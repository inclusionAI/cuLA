import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus
from fla.utils import autotune_cache_kwargs, input_guard

BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]


def naive_gdn_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Torch reference implementation for KDA gate computation.

    Computes: g = -A_log.exp().unsqueeze(-1) * softplus(g + dt_bias.view(g.shape[-1]))

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., H]`.
        A_log (torch.Tensor):
            Parameter tensor with `H` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[H]`.

    Returns:
        Output tensor of shape `[..., H]` .
    """
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias
    g = (-A_log.float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


# naive gdn lowerbound method based off of fla.ops.kda.gate
def naive_gdn_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias
    g = lower_bound * F.sigmoid(A_log.exp() * g)
    return g.to(output_dtype)


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8]],
    key=["H", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def gdn_gate_chunk_cumsum_scalar_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0)).to(tl.float32)

    # Apply dt_bias if exists
    if HAS_BIAS:
        b_bias = tl.load(dt_bias + i_h).to(tl.float32)
        b_s = b_s + b_bias

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        # Apply gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # Apply chunk local cumsum
    if REVERSE:
        b_o = tl.cumsum(b_gate, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@input_guard
def gdn_gate_chunk_cumsum_lowerbound(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    lower_bound: float | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch_size 1 is supported when cu_seqlens is provided"
    assert len(g.shape) == 3
    B, T, H = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return (NT, B * H)

    gdn_gate_chunk_cumsum_scalar_kernel[grid](
        s=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        BT=BT,
        REVERSE=False,
    )
    return g
