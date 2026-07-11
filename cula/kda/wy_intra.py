from __future__ import annotations

import torch
import triton
import triton.language as tl

from cula.kda.wy_recompute import kda_recompute_w_u


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [1, 2, 4] for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "BC"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_intra_sub_chunk_kernel(
    k,
    g,
    beta,
    Akkd,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    # Per-head pointer offsets
    k_base = k + (bos * H + i_h) * K
    g_base = g + (bos * H + i_h) * K
    beta_base = beta + bos * H + i_h
    Akkd_base = Akkd + (bos * H + i_h) * BC

    # Load k, g (BC × BK), beta (BC) for this sub-chunk
    p_k = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g_base, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta_base, (T,), (H,), (i_ti,), (BC,), (0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_beta = tl.load(p_beta, boundary_check=(0,))

    o_gn = i_ti + tl.minimum(BC // 2, T - i_ti - 1)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    b_gn = tl.load(g + (bos * H + i_h) * K + o_gn * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)

    b_gm = (b_g - b_gn[None, :]).to(tl.float32)
    b_gq = tl.where(m_c[:, None], tl.math.exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], tl.math.exp2(-b_gm), 0.0)

    b_kgt = tl.trans(b_k * b_gk)
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Akkd = tl.make_block_ptr(Akkd_base, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Akkd, b_Akk.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()

    b_Ai = -b_Akk
    for i in range(2, tl.minimum(BC, T - i_ti)):
        b_a = -tl.load(Akkd_base + (i_ti + i) * (H * BC) + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)

    b_Ai += m_I

    tl.store(p_Akkd, b_Ai.to(Akkd.dtype.element_ty), boundary_check=(0, 1))


_SOLVE_DOT_PRECISION = tl.constexpr("tf32")


@triton.autotune(
    configs=[triton.Config({"BK": BK}, num_warps=num_warps) for BK in [32, 64] for num_warps in [1, 2, 4]],
    key=["H", "K", "BC"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_intra_inter_solve_kernel(
    k,
    g,
    beta,
    Akkd,
    Akk,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k_base = k + (bos * H + i_h) * K
    g_base = g + (bos * H + i_h) * K
    Akk_base = Akk + (bos * H + i_h) * BT
    Akkd_base = Akkd + (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        # sub-chunk 1 (vs sub-chunk 0)
        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_gn1 = tl.load(g + (bos * H + i_h) * K + i_tc1 * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], tl.math.exp2(b_g1 - b_gn1[None, :]), 0.0)
            b_kgt = tl.trans(b_k0 * tl.math.exp2(b_gn1[None, :] - b_g0))
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            # sub-chunk 2 (vs 0 and 1)
            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                b_gn2 = tl.load(g + (bos * H + i_h) * K + i_tc2 * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_gqn2 = tl.where(m_tc2[:, None], tl.math.exp2(b_g2 - b_gn2[None, :]), 0.0)
                b_kg2 = b_k2 * b_gqn2
                b_kgt0 = tl.trans(b_k0 * tl.math.exp2(b_gn2[None, :] - b_g0))
                b_Akk20 += tl.dot(b_kg2, b_kgt0)
                b_kgt1 = tl.trans(b_k1 * tl.math.exp2(b_gn2[None, :] - b_g1))
                b_Akk21 += tl.dot(b_kg2, b_kgt1)

                # sub-chunk 3 (vs 0, 1, 2)
                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(k_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    b_gn3 = tl.load(g + (bos * H + i_h) * K + i_tc3 * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
                    b_gqn3 = tl.where(m_tc3[:, None], tl.math.exp2(b_g3 - b_gn3[None, :]), 0.0)
                    b_kg3 = b_k3 * b_gqn3
                    b_kgt0 = tl.trans(b_k0 * tl.math.exp2(b_gn3[None, :] - b_g0))
                    b_Akk30 += tl.dot(b_kg3, b_kgt0)
                    b_kgt1 = tl.trans(b_k1 * tl.math.exp2(b_gn3[None, :] - b_g1))
                    b_Akk31 += tl.dot(b_kg3, b_kgt1)
                    b_kgt2 = tl.trans(b_k2 * tl.math.exp2(b_gn3[None, :] - b_g2))
                    b_Akk32 += tl.dot(b_kg3, b_kgt2)

    beta_base = beta + bos * H + i_h
    if i_tc1 < T:
        p_b1 = tl.make_block_ptr(beta_base, (T,), (H,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_b2 = tl.make_block_ptr(beta_base, (T,), (H,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_b3 = tl.make_block_ptr(beta_base, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    # Load 4 inverted diagonal blocks (from sub_chunk kernel)
    p_Akk00 = tl.make_block_ptr(Akkd_base, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd_base, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd_base, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd_base, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=_SOLVE_DOT_PRECISION),
        b_Ai00,
        input_precision=_SOLVE_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=_SOLVE_DOT_PRECISION),
        b_Ai11,
        input_precision=_SOLVE_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=_SOLVE_DOT_PRECISION),
        b_Ai22,
        input_precision=_SOLVE_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=_SOLVE_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=_SOLVE_DOT_PRECISION),
        input_precision=_SOLVE_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=_SOLVE_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=_SOLVE_DOT_PRECISION),
        input_precision=_SOLVE_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=_SOLVE_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=_SOLVE_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=_SOLVE_DOT_PRECISION),
        input_precision=_SOLVE_DOT_PRECISION,
    )

    # Store 10 blocks to the full BT×BT Akk buffer.
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    tl.store(p, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    tl.store(p, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    tl.store(p, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    tl.store(p, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    tl.store(p, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    tl.store(p, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    tl.store(p, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    tl.store(p, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    tl.store(p, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    p = tl.make_block_ptr(Akk_base, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
    tl.store(p, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


def kda_intra_native(
    k: torch.Tensor,  # bf16 [1, T, H, K]
    v: torch.Tensor,  # bf16 [1, T, H, V]
    gk: torch.Tensor,  # fp32 [1, T, H, K]
    beta: torch.Tensor,  # bf16 [1, T, H]
    chunk_size: int = 64,
    q: torch.Tensor | None = None,  # bf16 [1, T, H, K], for qg (optional)
    need_qg: bool = False,
    out_w: torch.Tensor | None = None,  # bf16 [1, T, H, K]
    out_u: torch.Tensor | None = None,  # bf16 [1, T, H, V]
    out_kg: torch.Tensor | None = None,  # bf16 [1, T, H, K]
    out_Akkd: torch.Tensor | None = None,  # fp32 [1, T, H, BC]
    out_Akk: torch.Tensor | None = None,  # bf16 [1, T, H, BT]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    assert B == 1, "intra expects packed [1, T, H, K] input"
    assert K == 128 and V == 128, f"specialized for K=V=128, got K={K} V={V}"
    BT = chunk_size
    BC = 16
    NT = (T + BT - 1) // BT
    NC = BT // BC

    if out_Akkd is not None:
        Akkd = out_Akkd[:B, :T]
    else:
        Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)
    if out_Akk is not None:
        Akk = out_Akk[:B, :T]
        Akk.zero_()
    else:
        Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)

    # Step 1: per-sub-chunk diagonal Akk inversion
    BK_sub = triton.next_power_of_2(K)  # =128 for K=128
    grid_sub = (NT, NC, B * H)
    _kda_intra_sub_chunk_kernel[grid_sub](
        k=k,
        g=gk,
        beta=beta,
        Akkd=Akkd,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK_sub,
    )

    # Step 2: per-chunk off-diagonal + assemble full Akk_inv
    grid_inter = (NT, B * H)
    _kda_intra_inter_solve_kernel[grid_inter](
        k=k,
        g=gk,
        beta=beta,
        Akkd=Akkd,
        Akk=Akk,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    if out_Akkd is None:
        del Akkd

    # Step 3: recompute w, u, kg (and optionally qg) from Akk
    w, u, qg, kg = kda_recompute_w_u(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if need_qg else None,
        gk=gk,
        chunk_size=chunk_size,
        out_w=out_w,
        out_u=out_u,
        out_kg=out_kg,
    )
    # Akk is similarly dead after recompute's kernel is queued.
    if out_Akk is None:
        del Akk
    return w, u, qg, kg
