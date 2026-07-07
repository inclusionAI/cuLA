from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_EXIT_STATE": lambda args: args["exit_state"] is not None,
        "MULTI_RAW": lambda args: args["raw_seq_idx"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kda_h_boundary_kernel(
    kg,
    w,
    u,
    gk,
    h0,  # fp32 [H, V, K] OR [1, H, V, K] — state at chunk 0
    cp_h0_out,  # fp32 [num_cp, H, V, K], cuLA layout
    slot_map,  # int32 [NT], slot_map[i_t] = output slot or -1
    exit_state,  # fp32 [H, V, K] or None — end-of-tile state for next tile
    raw_h0_dense,  # fp32 [raw_batch, H, V, K] — per-raw-seq h0 for cross-seq resets
    raw_seq_idx,  # int32 [NT] — raw_seq_idx[i_t] = which raw seq chunk i_t belongs to
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_EXIT_STATE: tl.constexpr,
    MULTI_RAW: tl.constexpr,
):
    # Grid: (V_tiles, H). One CTA per (V-tile, head) walks all T chunks serially.
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    NT = tl.cdiv(T, BT)

    # State tiles: 2× [BV, 64] holds the (V_tile=BV, K=128) state split into K=64 halves.
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    b_h2 = tl.zeros([BV, 64], dtype=tl.float32)

    # Per-batch (B=1) offset into per-head buffers.
    kg_ptr = kg + i_h * K
    w_ptr = w + i_h * K
    u_ptr = u + i_h * V
    gk_ptr = gk + i_h * K
    h0_ptr = h0 + i_h * V * K if USE_INITIAL_STATE else h0  # h0 [H, V, K] or [1, H, V, K]

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        p_h0_2 = tl.make_block_ptr(h0_ptr, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    if MULTI_RAW:
        prev_raw = tl.load(raw_seq_idx + 0)
    else:
        prev_raw = 0  # unused

    for i_t in range(NT):
        if MULTI_RAW:
            if i_t > 0:
                cur_raw = tl.load(raw_seq_idx + i_t)
                if cur_raw != prev_raw:
                    rh_base = raw_h0_dense + cur_raw.to(tl.int64) * (H * V * K) + i_h * (V * K)
                    p_rh_1 = tl.make_block_ptr(rh_base, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
                    p_rh_2 = tl.make_block_ptr(rh_base, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
                    b_h1 = tl.load(p_rh_1, boundary_check=(0, 1)).to(tl.float32)
                    b_h2 = tl.load(p_rh_2, boundary_check=(0, 1)).to(tl.float32)
                    prev_raw = cur_raw

        # ----- Conditional boundary store -----
        # slot = slot_map[i_t]; if slot >= 0, write h to cp_h0_out[slot, i_h, ...]
        slot = tl.load(slot_map + i_t)
        is_boundary = slot >= 0
        # Use slot=0 as safe target when not boundary; mask blocks the store.
        safe_slot = tl.maximum(slot, 0).to(tl.int64)
        out_base = cp_h0_out + safe_slot * (H * V * K) + i_h * (V * K)
        p_out_1 = tl.make_block_ptr(out_base, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        p_out_2 = tl.make_block_ptr(out_base, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        if is_boundary:
            tl.store(p_out_1, b_h1, boundary_check=(0, 1))
            tl.store(p_out_2, b_h2, boundary_check=(0, 1))

        # ----- Compute v_new = u - w @ h -----
        # w [BT, K] split into 2× [BT, 64] for the matmuls
        p_w1 = tl.make_block_ptr(w_ptr, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_v_acc = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))
        p_w2 = tl.make_block_ptr(w_ptr, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_v_acc += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))

        # u [BT, V_tile]
        p_u = tl.make_block_ptr(u_ptr, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v_acc

        # ----- Apply gate decay: h *= exp2(gk_last) -----
        last_idx = tl.minimum((i_t + 1) * BT, T) - 1
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(gk_ptr + last_idx * (H * K) + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32)
        b_h1 *= tl.math.exp2(b_gk_last1)[None, :]
        o_k2 = 64 + o_k1
        b_gk_last2 = tl.load(gk_ptr + last_idx * (H * K) + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32)
        b_h2 *= tl.math.exp2(b_gk_last2)[None, :]

        # ----- State update: h += kg^T @ v_new   (transpose_state_layout=True form) -----
        b_v_bf = b_v.to(kg.dtype.element_ty)
        p_kg1 = tl.make_block_ptr(kg_ptr, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_kg1 = tl.load(p_kg1, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_kg1, b_v_bf))
        p_kg2 = tl.make_block_ptr(kg_ptr, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
        b_kg2 = tl.load(p_kg2, boundary_check=(0, 1))
        b_h2 += tl.trans(tl.dot(b_kg2, b_v_bf))

    # ----- After the loop: optionally store the end-of-tile state for the next tile -----
    if STORE_EXIT_STATE:
        es_ptr = exit_state + i_h * V * K
        p_es_1 = tl.make_block_ptr(es_ptr, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        p_es_2 = tl.make_block_ptr(es_ptr, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
        tl.store(p_es_1, b_h1, boundary_check=(0, 1))
        tl.store(p_es_2, b_h2, boundary_check=(0, 1))


def kda_cp_h0_boundary(
    kg: torch.Tensor,  # bf16 [1, T, H, K]
    w: torch.Tensor,  # bf16 [1, T, H, K]
    u: torch.Tensor,  # bf16 [1, T, H, V]
    g_cumsum: torch.Tensor,  # fp32 [1, T, H, K]
    h0: torch.Tensor | None,  # fp32 [H, V, K] (or [1, H, V, K]) — state at this tile's chunk 0
    slot_map: torch.Tensor,  # int32 [NT], slot_map[i_t] = output slot or -1
    num_cp: int,
    chunk_size: int = 64,
    cp_h0_out: torch.Tensor | None = None,  # pre-allocated [num_cp, H, V, K] fp32; allocated if None
    exit_state: torch.Tensor | None = None,  # if not None, kernel writes the end-of-T state to [H, V, K] fp32
    raw_h0_dense: torch.Tensor | None = None,  # fp32 [raw_batch, H, V, K] for cross-raw-seq resets
    raw_seq_idx: torch.Tensor | None = None,  # int32 [NT] mapping chunk → raw seq idx (None for single-raw)
) -> torch.Tensor:
    assert kg.is_contiguous() and w.is_contiguous() and u.is_contiguous()
    assert kg.size(0) == 1 and w.size(0) == 1 and u.size(0) == 1 and g_cumsum.size(0) == 1
    assert slot_map.dtype == torch.int32 and slot_map.is_contiguous()
    if raw_seq_idx is not None:
        assert raw_h0_dense is not None, "raw_h0_dense required when raw_seq_idx is set"
        assert raw_seq_idx.dtype == torch.int32 and raw_seq_idx.is_contiguous()

    T = kg.size(1)
    H = kg.size(2)
    K = kg.size(3)
    V = u.size(3)
    assert K == 128 and V == 128, f"Phase 1 kernel hard-codes K=V=128, got K={K} V={V}"

    if cp_h0_out is None:
        cp_h0_out = torch.empty(num_cp, H, V, K, dtype=torch.float32, device=kg.device)

    BT = chunk_size

    # Grid depends on BV (autotune-selected); use meta-grid.
    def grid(meta):
        return (V // meta["BV"], H)

    _kda_h_boundary_kernel[grid](
        kg,
        w,
        u,
        g_cumsum,
        h0,
        cp_h0_out,
        slot_map,
        exit_state,
        raw_h0_dense,
        raw_seq_idx,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return cp_h0_out
