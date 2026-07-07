import torch
import triton
import triton.language as tl
from fla.ops.utils.constant import RCP_LN2 as _RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.softplus import softplus

# Triton requires module-level constants used inside @jit kernels to be
# wrapped in tl.constexpr.
RCP_LN2 = tl.constexpr(_RCP_LN2)


@triton.jit
def _gate_l2norm_fused_kernel(
    # Pointers
    g_ptr,
    A_log_ptr,
    dt_bias_ptr,  # gate inputs
    q_ptr,
    k_ptr,  # qk inputs
    g_out_ptr,  # gate output (fp32 cumsum)
    yq_ptr,
    yk_ptr,  # qk outputs (bf16)
    rstd_q_ptr,
    rstd_k_ptr,  # qk rstd outputs (fp32)
    cu_seqlens_ptr,
    chunk_indices_ptr,
    # Scalars
    lower_bound,
    eps_l2,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices_ptr + i_t * 2).to(tl.int32)
        i_t_local = tl.load(chunk_indices_ptr + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        T_seq = eos - bos
        bt_base = bos + i_t_local * BT
        valid_t = (i_t_local * BT + tl.arange(0, BT)) < T_seq
    else:
        bt_base = (i_bh // H) * T + i_t * BT
        valid_t = (i_t * BT + tl.arange(0, BT)) < T

    rows = bt_base + tl.arange(0, BT)

    cols = tl.arange(0, BD)  # (BD,)
    valid_d = cols < D

    offs = rows[:, None] * (H * D) + i_h * D + cols[None, :]
    mask = valid_t[:, None] & valid_d[None, :]

    # ====================================================================
    # GATE: cumsum( transform(g + bias) ) * RCP_LN2
    # ====================================================================
    b_g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b_bias = tl.load(dt_bias_ptr + i_h * D + cols, mask=valid_d, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    b_A = tl.load(A_log_ptr + i_h).to(tl.float32)
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
    else:
        b_gate = -tl.exp(b_A) * softplus(b_g)

    b_gate_cs = tl.cumsum(b_gate, axis=0) * RCP_LN2
    # zero out the tokens beyond T so we don't pollute g_out tail rows
    b_gate_cs = tl.where(mask, b_gate_cs, 0.0)
    tl.store(g_out_ptr + offs, b_gate_cs, mask=mask)

    # ====================================================================
    # L2-NORM Q  — per-row normalisation along D, rstd written to (B*T*H,)
    # ====================================================================
    b_q = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b_q_sq = tl.sum(b_q * b_q, axis=1)  # (BT,)
    b_rstd_q = 1.0 / tl.sqrt(b_q_sq + eps_l2)  # (BT,)
    b_yq = b_q * b_rstd_q[:, None]
    tl.store(yq_ptr + offs, b_yq.to(yq_ptr.dtype.element_ty), mask=mask)

    # rstd is shape (B, T, H,) contiguous in (b*T*H + t*H + h) order
    rstd_offs = rows * H + i_h  # (BT,)
    tl.store(rstd_q_ptr + rstd_offs, b_rstd_q, mask=valid_t)

    # ====================================================================
    # L2-NORM K
    # ====================================================================
    b_k = tl.load(k_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b_k_sq = tl.sum(b_k * b_k, axis=1)
    b_rstd_k = 1.0 / tl.sqrt(b_k_sq + eps_l2)
    b_yk = b_k * b_rstd_k[:, None]
    tl.store(yk_ptr + offs, b_yk.to(yk_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_k_ptr + rstd_offs, b_rstd_k, mask=valid_t)


def gate_l2norm_fused_fwd(
    g: torch.Tensor,  # (B, T, H, D) bf16
    q: torch.Tensor,  # (B, T, H, D) bf16
    k: torch.Tensor,  # (B, T, H, D) bf16
    A_log: torch.Tensor,  # (H,) fp32
    dt_bias: torch.Tensor | None,  # (H*D,) fp32 or None
    lower_bound: float | None,  # if not None and safe_gate -> use lb*sigmoid path
    chunk_size: int = 64,
    eps_l2: float = 1e-6,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
):
    """One-launch fused preprocessing.

    Returns:
        g_out: (B, T, H, D) fp32  -- gate cumsum * RCP_LN2
        y_q:   (B, T, H, D) bf16  -- l2-normalised q
        y_k:   (B, T, H, D) bf16  -- l2-normalised k
        rstd_q, rstd_k: (B, T, H) fp32  -- 1/sqrt(sum^2 + eps)
    """
    assert g.shape == q.shape == k.shape, f"shapes must match: g{g.shape} q{q.shape} k{k.shape}"
    assert g.is_contiguous() and q.is_contiguous() and k.is_contiguous(), "all inputs must be contiguous"
    B, T, H, D = g.shape
    assert chunk_size == 64, "only chunk_size=64 supported (matches SM90 main kernel)"

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1, "varlen path expects packed B=1 layout"
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if dt_bias is None:
        dt_bias_arg = A_log  # any valid fp32 pointer; HAS_BIAS=False suppresses use
        has_bias = False
    else:
        dt_bias_arg = dt_bias
        has_bias = True

    use_lower_bound = lower_bound is not None
    if not use_lower_bound:
        lower_bound = 0.0  # unused, but Triton needs a value

    g_out = torch.empty_like(g, dtype=torch.float32)
    y_q = torch.empty_like(q)
    y_k = torch.empty_like(k)
    rstd_q = torch.empty((B, T, H), dtype=torch.float32, device=q.device)
    rstd_k = torch.empty((B, T, H), dtype=torch.float32, device=q.device)

    BD = triton.next_power_of_2(D)
    if is_varlen:
        NT = chunk_indices.shape[0]
        grid = (NT, H)
        cu_seqlens_arg = cu_seqlens
        chunk_indices_arg = chunk_indices
    else:
        NT = triton.cdiv(T, chunk_size)  # ceil — partial last chunk handled by mask
        grid = (NT, B * H)
        cu_seqlens_arg = A_log
        chunk_indices_arg = A_log
    num_warps = 1 if BD <= 128 else (2 if BD <= 256 else 4)

    _gate_l2norm_fused_kernel[grid](
        g_ptr=g,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias_arg,
        q_ptr=q,
        k_ptr=k,
        g_out_ptr=g_out,
        yq_ptr=y_q,
        yk_ptr=y_k,
        rstd_q_ptr=rstd_q,
        rstd_k_ptr=rstd_k,
        cu_seqlens_ptr=cu_seqlens_arg,
        chunk_indices_ptr=chunk_indices_arg,
        lower_bound=lower_bound,
        eps_l2=eps_l2,
        T=T,
        H=H,
        D=D,
        BT=chunk_size,
        BD=BD,
        HAS_BIAS=has_bias,
        USE_LOWER_BOUND=use_lower_bound,
        IS_VARLEN=is_varlen,
        num_warps=num_warps,
    )
    return g_out, y_q, y_k, rstd_q, rstd_k
