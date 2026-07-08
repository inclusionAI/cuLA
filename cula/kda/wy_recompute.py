from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [2, 4, 8] for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kda_recompute_wuk_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
):
    """K = V = 128, BT = 64 specialized. BK = K, BV = V (no inner loop)."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    # Per-head pointer offsets
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    # ----- u = A @ (β · v) -----
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
    b_u = tl.dot(b_A, b_vb)
    p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0))
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    # ----- Load k, gk, compute β·exp2(gk)·k and kg -----
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
    b_exp_gk = tl.math.exp2(b_gk)

    # w = A @ (β · exp2(gk) · k)
    b_kb = b_k * b_b[:, None] * b_exp_gk
    b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
    p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    # qg = q · exp2(gk)  (optional)
    if STORE_QG:
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * b_exp_gk
        p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

    # kg = β · exp2(g_chunk_end - gk) · k  (optional, needed by cp_h0 path)
    if STORE_KG:
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        o_k = tl.arange(0, K)
        b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=o_k < K, other=0.0).to(tl.float32)
        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_kg = b_k * tl.where(m_t[:, None], tl.math.exp2(b_gn[None, :] - b_gk), 0.0)
        p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))


def kda_recompute_w_u(
    k: torch.Tensor,  # bf16 [B, T, H, K]
    v: torch.Tensor,  # bf16 [B, T, H, V]
    beta: torch.Tensor,  # bf16 [B, T, H]
    A: torch.Tensor,  # bf16 [B, T, H, BT]
    q: torch.Tensor | None,  # bf16 [B, T, H, K], or None
    gk: torch.Tensor,  # fp32 [B, T, H, K]
    chunk_size: int = 64,
    out_w: torch.Tensor | None = None,
    out_u: torch.Tensor | None = None,
    out_kg: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    assert K == 128 and V == 128, f"specialized for K=V=128, got K={K} V={V}"
    assert A.shape[-1] == BT, f"expected A.shape[-1]={BT}, got {A.shape[-1]}"

    w = out_w[:B, :T] if out_w is not None else torch.empty_like(k)
    u = out_u[:B, :T] if out_u is not None else torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    if gk is not None:
        kg = out_kg[:B, :T] if out_kg is not None else torch.empty_like(k)
    else:
        kg = None

    NT = triton.cdiv(T, BT)
    grid = (NT, B * H)
    _kda_recompute_wuk_kernel[grid](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return w, u, qg, kg
