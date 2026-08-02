# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightning Attention KVBuffer verify kernel (paper Eq. 7 for LA).

Closed-form parallel verification — computes each draft step's output directly
from (h0, k, v) without materializing the intermediate states:

    o_t = alpha^{t+1} * (h0 @ q_t * scale)                       <- "term1" (HQ)
        + sum_{i=0..t} alpha^{t-i} * (q_t . k_i) * scale * v_i   <- "term2" (QK·V)

The two dot-product GEMMs run on tensor cores via inline-PTX mma.sync.m16n8k8
(TF32). Operands are staged in fp32 SMEM (manual fragment addressing — no
LdMatrix/StMatrix). Everything downstream of the GEMMs is plain scalar math.

PARALLELISM
    Grid:  (B * HV * num_v_tiles, 1, 1)   — one block per (sequence, v-head, V-tile)
    Block: 128 threads = 4 warps.  Each warp owns `rows_per_group` output V-rows.

PIPELINE (per block)
    Stage 0  cooperative load q*scale, k -> SMEM (sQ, sK)
    Stage 1  GEMM2: QK[t,i] = q_t . k_i           (warp 0 only) -> s_qk_scaled
    Stage 2  per V-row-block: load h0 -> SMEM, GEMM1: HQ = h0 @ q_t,
             then scalar combine term1+term2 -> o

MMA m16n8k8 FRAGMENT MAP (lane = gid*4 + tig, gid=lane//4 in 0..7, tig=lane%4 in 0..3)
    A[16,8] row-major : a0=A[gid,tig]  a1=A[gid+8,tig]  a2=A[gid,tig+4]  a3=A[gid+8,tig+4]
    B[8,8]  col-major : b0=B[tig,gid]  b1=B[tig+4,gid]
    C[16,8]           : c0=C[gid,2tig] c1=C[gid,2tig+1] c2=C[gid+8,2tig] c3=C[gid+8,2tig+1]
    We only have 8 valid rows (BT=8), so A rows 8..15 are fed as zeros and the
    corresponding outputs c2,c3 / e2,e3 are unused padding.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)
from cutlass.cute.typing import Int32
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op

from cula.lightning.la_decode_mtp import (
    NUM_THREADS_MTP,
    hq_dot_pair,
)
from cula.utils import USE_FAST_MATH, get_device_sm_version

# Dispatch threshold between the two verify implementations.
# The MMA (tensor-core) kernel wins at T>=4 (matches at T=4, +45% at T=8 vs the
# shuffle kernel), but the shuffle kernel wins at small T (T<=2) where the MMA
# GEMMs are under-utilised and its larger SMEM footprint caps occupancy.
# See docs/la_verify_kvbuffer_dev_history.md §6 for the full benchmark.
MMA_MIN_T: int = 4


def get_mtp_config(B: int, T: int, HV: int, V: int) -> tuple:
    """Pick (tile_v, vec_size, ilp_rows) for the verify + state-update kernels.

    Grid-searched on the LA verify kernel (B200, H=HV=64, K=V=128,
    B ∈ [1..128], T ∈ [2,4,8]).  This is the *opposite* regime from the decode
    kernel's ``get_mtp_config``: the verify kernel runs tensor-core MMA GEMMs
    (T>=4) and is compute-bound, so it wants LARGE tiles with ilp_rows=8 — more
    V-rows per block amortize the q/k SMEM staging and fill the m16n8k8 MMA
    tiles.  Small tiles starve the tensor cores (75% output padding at tile_v=8).

    vs the old shared GDN thresholds (tile_v=64, ilp=4 for work_units>1024):
    up to 1.6x faster on the MMA path at large B (e.g. B=128,T=4: 0.170→0.125 ms;
    B=1024,T=4: 0.031→0.019 ms).  At small work_units the kernel is
    latency-bound, so the tile choice is immaterial (<2% spread).

    Returns a 3-tuple — the verify/state-update kernels have no ``use_smem_v``
    knob (they always stage v in SMEM).
    """
    work_units = B * HV
    vec_size = 4
    if work_units <= 512:
        tile_v, ilp_rows = 64, 8
    else:
        tile_v, ilp_rows = 128, 8

    tile_v = min(tile_v, V)
    rows_per_group = tile_v // 4
    assert rows_per_group % ilp_rows == 0, (
        f"tile_v={tile_v} / num_groups=4 / ilp_rows={ilp_rows} doesn't divide cleanly "
        f"(rows_per_group={rows_per_group}); the ILP loop would run zero iterations."
    )
    return tile_v, vec_size, ilp_rows


# ---------------------------------------------------------------------------
# Inline PTX mma.sync.m16n8k8.tf32 — copied from kda_decode_mtp_kvbuffer.py
# ---------------------------------------------------------------------------


@dsl_user_op
def _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """One mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; returns (d0,d1,d2,d3)."""
    f32 = _T.f32()
    i32 = _T.i32()

    def _bits(v):
        vv = v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v
        return _arith.bitcast(i32, vv, loc=loc, ip=ip)

    def _f(v):
        return v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v

    res_ty = _llvm.StructType.get_literal([f32, f32, f32, f32])
    res = _llvm.inline_asm(
        res_ty,
        [_bits(a0), _bits(a1), _bits(a2), _bits(a3), _bits(b0), _bits(b1), _f(c0), _f(c1), _f(c2), _f(c3)],
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = cutlass.Float32(_llvm.extractvalue(f32, res, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(_llvm.extractvalue(f32, res, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(_llvm.extractvalue(f32, res, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(_llvm.extractvalue(f32, res, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


BT: int = 8  # pad M and N dimensions to 8 for mma fragment


@cute.kernel
def la_verify_kvbuffer_kernel(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] fp32 (READ ONLY)
    decay_scales: cute.Tensor,  # [H] fp32
    q: cute.Tensor,  # [B, T, H,  K] fp32
    k: cute.Tensor,  # [B, T, H,  K] fp32
    v: cute.Tensor,  # [B, T, HV, V] fp32
    o: cute.Tensor,  # [B, T, HV, V] fp32 (WRITTEN)
    h0_indices: cute.Tensor,  # [B] int32
    k_buf: cute.Tensor,  # [pool_size, T, H, K] fp32 (WRITTEN when write_kv)
    v_buf: cute.Tensor,  # [pool_size, T, HV, V] fp32 (WRITTEN when write_kv)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    write_kv: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # MMA lane decomposition (see fragment map in module docstring).
    gid = lane_id // 4  # 0..7: row index within the MMA tile
    tig = lane_id % 4  # 0..3: k-pair within the current 8-wide K-slab

    # 4 warps/block; each warp owns a disjoint set of output V-rows. All 32 lanes
    # of a warp cooperate over the full K dimension (K=128, vec_size=4).
    NUM_WARPS: cutlass.Constexpr[int] = 4

    # Block -> (sequence n, v-head hv, V-tile i_v); i_h maps the v-head to its q/k head.
    block_idx, _, _ = cute.arch.block_idx()
    i_v = block_idx % num_v_tiles
    tmp = block_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]

    # ---- Per-lane registers ----
    r_decay_pow = cute.make_rmem_tensor(cute.make_layout((T + 1,), stride=(1,)), cutlass.Float32)
    r_q_f32 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k_f32 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

    # ---- SMEM (all fp32; MMA bitcasts fp32->TF32, no separate conversion) ----
    # KP = K+4 pads the row stride so 132%32=4: the gid*4+tig access pattern then
    # hits 32 distinct banks, giving conflict-free SMEM reads in both GEMMs.
    KP: cutlass.Constexpr[int] = K + 4
    smem = cutlass.utils.SmemAllocator()
    # GEMM operands. sQ holds q*scale, doubles as GEMM2-A and GEMM1-B.
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, KP), stride=(KP, 1)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, KP), stride=(KP, 1)), 16)
    # h0, one [BT, K] region per warp (each warp does GEMM1 for its own V-rows).
    sH0 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_WARPS, BT, KP), stride=(BT * KP, KP, 1)), 16)
    # Decay-masked QK coefficients [T, T], produced by GEMM2, consumed by every warp.
    s_qk_scaled = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16)
    # v is lane-invariant within a warp; stage it once in SMEM and broadcast-read.
    sVbuf = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_WARPS, T, BT), stride=(T * BT, BT, 1)), 16)

    if cache_idx >= 0:
        alpha = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)

        r_decay_pow[0] = cutlass.Float32(1.0)
        for t in cutlass.range_constexpr(1, T + 1):
            r_decay_pow[t] = r_decay_pow[t - 1] * alpha

        rows_per_group: cutlass.Constexpr[int] = tile_v // NUM_WARPS
        flat_state_idx = cache_idx * HV + i_hv

        # ================================================================
        # Stage 0: cooperative load q*scale, k -> SMEM (sQ, sK), fp32.
        # Warp w loads tokens {w, w+4, ...}; within a token, lane_id covers the
        # K dimension (vec_size contiguous elements each). Rows T..BT-1 are the
        # MMA M-padding and are zeroed.
        # ================================================================
        tokens_per_warp: cutlass.Constexpr[int] = (BT + NUM_WARPS - 1) // NUM_WARPS
        for tt in cutlass.range_constexpr(tokens_per_warp):
            t_tok = tt * NUM_WARPS + warp_idx
            if t_tok < T:
                q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
                k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
                cute.autovec_copy(q_tile, r_q_f32)
                cute.autovec_copy(k_tile, r_k_f32)
                for c in cutlass.range_constexpr(vec_size):
                    col = lane_id * vec_size + c
                    sQ[(t_tok, col)] = cutlass.Float32(r_q_f32[c]) * scale
                    sK[(t_tok, col)] = cutlass.Float32(r_k_f32[c])
                # Persist k to the pool buffer while it is already in registers.
                if cutlass.const_expr(write_kv):
                    if i_v == 0 and i_hv % (HV // H) == 0:
                        kb_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size), (cache_idx, t_tok, i_h, lane_id))
                        for c in cutlass.range_constexpr(vec_size):
                            kb_tile[c] = r_k_f32[c]
            if t_tok >= T and t_tok < BT:
                for c in cutlass.range_constexpr(vec_size):
                    col = lane_id * vec_size + c
                    sQ[(t_tok, col)] = cutlass.Float32(0.0)
                    sK[(t_tok, col)] = cutlass.Float32(0.0)

        cute.arch.barrier()

        # ================================================================
        # Stage 1: GEMM2 — QK[t,i] = q_t . k_i, accumulated over the full K.
        # A = Q[8,K] (rows = tokens), B = K[8,K] read col-major as K^T. Warp 0
        # alone has enough lanes (M=N=T<=8), so the other warps skip this.
        # ================================================================
        if warp_idx == 0:
            c0 = cutlass.Float32(0.0)
            c1 = cutlass.Float32(0.0)
            c2 = cutlass.Float32(0.0)  # c2,c3 = padding rows 8..15, unused
            c3 = cutlass.Float32(0.0)
            for ks in cutlass.range_constexpr(K // 8):
                kb = ks * 8
                a0 = sQ[(gid, kb + tig)]
                a1 = cutlass.Float32(0.0)
                a2 = sQ[(gid, kb + tig + 4)]
                a3 = cutlass.Float32(0.0)
                b0 = sK[(gid, kb + tig)]
                b1 = sK[(gid, kb + tig + 4)]
                c0, c1, c2, c3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)

            # c0,c1 hold QK[gid, 2tig], QK[gid, 2tig+1]. Keep the causal lower
            # triangle, pre-multiply by the decay alpha^{t-i}, store coefficients.
            for fi in cutlass.range_constexpr(2):
                row = gid
                col = 2 * tig + fi
                cv = c1 if cutlass.const_expr(fi == 1) else c0
                if row < T and col < T:
                    if col <= row:
                        s_qk_scaled[(row, col)] = r_decay_pow[row - col] * cv
                    else:
                        s_qk_scaled[(row, col)] = cutlass.Float32(0.0)

        cute.arch.barrier()

        # ================================================================
        # Stage 2: for each block of `ilp_rows` V-rows owned by this warp,
        # load h0 -> SMEM, run GEMM1 (HQ = h0 @ q_t), then combine the two terms.
        # ================================================================
        num_row_blocks: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for row_block in cutlass.range_constexpr(num_row_blocks):
            v_base = i_v * tile_v + warp_idx * rows_per_group + row_block * ilp_rows
            if v_base + (ilp_rows - 1) < V:
                # (a) Coalesced h0 load: lane_id indexes vec_size contiguous K
                # elements, so the 32 lanes read one full contiguous row per step
                # (no over-fetch). Each warp fills its own sH0 region.
                sH0_w = sH0[(warp_idx, None, None)]  # [BT, KP]
                gH0 = h0_source[(flat_state_idx, None, None)]  # [V, K]
                for row in cutlass.range_constexpr(ilp_rows):
                    h_g = cute.local_tile(gH0, (1, vec_size), (v_base + row, lane_id))
                    h_s = cute.local_tile(sH0_w, (1, vec_size), (row, lane_id))
                    cute.autovec_copy(h_g, h_s)
                # Zero the M-padding rows (ilp_rows..BT-1). GEMM1 reads all BT rows;
                # their outputs are unused, but leaving stale/NaN SMEM as MMA inputs
                # is unclean — explicitly zero so the fragment is well-defined.
                for row in cutlass.range_constexpr(ilp_rows, BT):
                    for c in cutlass.range_constexpr(vec_size):
                        sH0_w[(row, lane_id * vec_size + c)] = cutlass.Float32(0.0)
                cute.arch.sync_warp()  # make sH0 writes visible to this warp's GEMM1

                # (b) GEMM1: HQ[row, t] = h0_row . q_t, over the full K.
                # A = sH0 (this warp's V-rows), B = sQ read col-major as Q^T.
                e0 = cutlass.Float32(0.0)
                e1 = cutlass.Float32(0.0)
                e2 = cutlass.Float32(0.0)  # e2,e3 = padding rows 8..15, unused
                e3 = cutlass.Float32(0.0)
                for ks in cutlass.range_constexpr(K // 8):
                    kb = ks * 8
                    a0 = sH0[(warp_idx, gid, kb + tig)]
                    a1 = cutlass.Float32(0.0)
                    a2 = sH0[(warp_idx, gid, kb + tig + 4)]
                    a3 = cutlass.Float32(0.0)
                    b0 = sQ[(gid, kb + tig)]
                    b1 = sQ[(gid, kb + tig + 4)]
                    e0, e1, e2, e3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
                # e0,e1 now hold HQ[gid, 2tig], HQ[gid, 2tig+1] (gid = V-row index).

                # (c) Stage v in SMEM (lane-invariant within the warp) and persist it.
                if lane_id < ilp_rows:
                    for t in cutlass.range_constexpr(T):
                        vv = v[i_n, t, i_hv, v_base + lane_id]
                        sVbuf[(warp_idx, t, lane_id)] = cutlass.Float32(vv)
                        if cutlass.const_expr(write_kv):
                            v_buf[(cache_idx, t, i_hv, v_base + lane_id)] = cutlass.Float32(vv)

                # (d) Combine: o[t, row] = alpha^{t+1}*HQ[row,t] + sum_i qk[t,i]*v[i,row].
                # The (t, row) output grid has T*ilp_rows entries. Distribute them
                # across the 32 lanes in a grid-stride fashion: lane L handles outputs
                # L, L+32, L+64, ... so each lane emits ceil(T*ilp_rows/32) of them.
                # This keeps every lane doing useful work for ANY T (T=4 -> 1 each,
                # T=8 -> 2 each, T=2 -> half the lanes), with no redundant compute and
                # no SMEM reshuffle — HQ is fetched straight from its owner lane.
                num_out: cutlass.Constexpr[int] = T * ilp_rows
                outs_per_lane: cutlass.Constexpr[int] = (num_out + 31) // 32
                for oj in cutlass.range_constexpr(outs_per_lane):
                    out_idx = lane_id + oj * 32
                    my_t = out_idx // ilp_rows
                    my_slot = out_idx % ilp_rows
                    # shuffle_sync must execute on ALL lanes (warp-collective), so it
                    # stays outside the my_t<T guard. src_lane is always in [0,32) even
                    # for tail lanes. HQ[my_slot, my_t] lives on lane (gid=my_slot,
                    # tig=my_t//2) in register e0 (even my_t) or e1 (odd my_t).
                    src_lane = my_slot * 4 + (my_t // 2)
                    hq_e0 = cute.arch.shuffle_sync(e0, src_lane)
                    hq_e1 = cute.arch.shuffle_sync(e1, src_lane)
                    if my_t < T:  # guard the tail when T*ilp_rows isn't a multiple of 32
                        parity = cutlass.Float32(my_t % 2)
                        hq_val = hq_e0 * (cutlass.Float32(1.0) - parity) + hq_e1 * parity
                        acc = r_decay_pow[my_t + 1] * hq_val
                        for i in cutlass.range_constexpr(T):
                            if i <= my_t:
                                acc = acc + s_qk_scaled[(my_t, i)] * sVbuf[(warp_idx, i, my_slot)]
                        o[(i_n, my_t, i_hv, v_base + my_slot)] = acc


@cute.jit
def run_la_verify_kvbuffer_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
    grid_size: Int32,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    write_kv: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v

    # Mirror the kernel's SMEM allocations (all fp32 = 4 bytes). BT=8, 4 warps.
    KP: cutlass.Constexpr[int] = K + 4
    F32: cutlass.Constexpr[int] = 4
    smem_bytes = (
        BT * KP * F32  # sQ
        + BT * KP * F32  # sK
        + 4 * BT * KP * F32  # sH0 (one [BT, KP] per warp)
        + T * T * F32  # s_qk_scaled
        + 4 * T * BT * F32  # sVbuf
        + 5 * 16  # per-allocation 16B alignment padding (5 tensors)
    )

    la_verify_kvbuffer_kernel(
        h0_source,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        k_buf,
        v_buf,
        vec_size,
        num_v_tiles,
        tile_v,
        scale,
        T,
        H,
        HV,
        K,
        V,
        ilp_rows,
        write_kv,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _get_compiled_verify_kvbuffer_kernel(
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    softmax_scale: float,
    tile_v: int,
    vec_size: int,
    ilp_rows: int,
    write_kv: bool,
):
    return {}


def _verify_kvbuffer_compile_cache(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    softmax_scale: float,
    *,
    write_kv: bool,
    device: torch.device,
):
    """Return (cache dict, tile config) for the given launch parameters."""
    if T < MMA_MIN_T:
        tile_v, vec_size, ilp_rows = get_mtp_config(B, T, HV, V)
        assert V % ilp_rows == 0, f"V={V} % ilp_rows={ilp_rows} ≠ 0: partial row-blocks would be silently skipped"
        use_packed_fma = get_device_sm_version(device)[0] >= 10
        cache = _get_compiled_verify_kvbuffer_kernel_shuffle(
            T,
            H,
            HV,
            K,
            V,
            pool_size,
            softmax_scale,
            tile_v,
            vec_size,
            ilp_rows,
            use_packed_fma,
            write_kv,
        )
        return cache, (tile_v, vec_size, ilp_rows, use_packed_fma)

    tile_v, vec_size, ilp_rows = get_mtp_config(B, T, HV, V)
    assert T <= 8, f"T={T} > 8: MMA kernel's BT=8 token staging only covers T ≤ 8"
    assert V % ilp_rows == 0, f"V={V} % ilp_rows={ilp_rows} ≠ 0: partial row-blocks would be silently skipped"
    cache = _get_compiled_verify_kvbuffer_kernel(
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        softmax_scale,
        tile_v,
        vec_size,
        ilp_rows,
        write_kv,
    )
    return cache, (tile_v, vec_size, ilp_rows, None)


def get_compiled_verify_kvbuffer_handle(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    softmax_scale: float,
    *,
    write_kv: bool,
    device: torch.device,
):
    """Return a pre-compiled verify kernel handle (benchmark kernel-only path).

    Call ``linear_attention_verify_kvbuffer`` once with the same config first.
    The returned handle has one stable signature for both MMA and shuffle paths:
    ``(..., k_buf, v_buf, stream)``. It computes the runtime grid size internally.
    """
    cache, (tile_v, _, _, _) = _verify_kvbuffer_compile_cache(
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        softmax_scale,
        write_kv=write_kv,
        device=device,
    )
    compiled = cache.get("compiled")
    if compiled is None:
        raise RuntimeError("Verify kernel not compiled for this config; call linear_attention_verify_kvbuffer once first.")

    num_v_tiles = (V + tile_v - 1) // tile_v

    def run_compiled(
        h0_source,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        k_buf,
        v_buf,
        stream,
    ):
        grid_size = q.shape[0] * HV * num_v_tiles
        compiled(
            h0_source,
            decay_scales,
            q,
            k,
            v,
            o,
            h0_indices,
            k_buf,
            v_buf,
            Int32(grid_size),
            stream,
        )

    return run_compiled


def linear_attention_verify_kvbuffer(
    q: torch.Tensor,  # [B, T, H,  K] fp32
    k: torch.Tensor,  # [B, T, H,  K] fp32
    v: torch.Tensor,  # [B, T, HV, V] fp32
    s: torch.Tensor,  # [pool_size, HV, V, K] fp32, READ ONLY
    out: torch.Tensor,  # [B, T, HV, V] fp32, WRITTEN
    decay_scales: torch.Tensor,  # [H] fp32
    h0_indices: torch.Tensor,  # [B] int32, -1 to skip
    softmax_scale: float,
    T: int,
    k_buf: torch.Tensor | None = None,
    v_buf: torch.Tensor | None = None,
) -> None:
    """
    Closed-form parallel verify (KVBuffer Eq. 7). Writes out; does not touch s.

    When k_buf and v_buf are provided, also writes k,v to fp32 pool-indexed
    buffers so the caller can free the original k,v tensors after this call
    returns. This matches Ling/SGLang, where q/k/v arrive after fp32 RoPE and
    are committed into a fp32 temporal state.

    Dispatches between two equivalent implementations by draft depth T: the
    tensor-core MMA kernel below for T >= MMA_MIN_T, and the warp-shuffle kernel
    for smaller T (where MMA's GEMMs are under-utilised). Both share the same
    interface, grid, and KVBuffer write semantics.
    """
    if T < MMA_MIN_T:
        return linear_attention_verify_kvbuffer_shuffle(
            q,
            k,
            v,
            s,
            out,
            decay_scales,
            h0_indices,
            softmax_scale,
            T,
            k_buf=k_buf,
            v_buf=v_buf,
        )

    B, T_q, H, K = q.shape
    assert T_q == T, f"q.shape[1]={T_q} doesn't match T={T}"
    assert K == 128, f"K={K} != 128: kernel hardcodes K=128 (threads_per_group, KP=K+4, lane K-coverage)"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError(f"q/k/v must be torch.float32, got {q.dtype}/{k.dtype}/{v.dtype}")
    if s.dtype != torch.float32:
        raise ValueError(f"s must be torch.float32, got {s.dtype}")
    if out.dtype != torch.float32:
        raise ValueError(f"out must be torch.float32, got {out.dtype}")

    write_kv = k_buf is not None and v_buf is not None
    if (k_buf is None) != (v_buf is None):
        raise ValueError("k_buf and v_buf must both be None or both be provided")
    if write_kv and (k_buf.dtype != torch.float32 or v_buf.dtype != torch.float32):
        raise ValueError(f"k_buf/v_buf must be torch.float32, got {k_buf.dtype}/{v_buf.dtype}")

    cache, (tile_v, vec_size, ilp_rows, _) = _verify_kvbuffer_compile_cache(
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        softmax_scale,
        write_kv=write_kv,
        device=q.device,
    )

    h0_view = s.view(pool_size * HV, V, K)

    if not write_kv:
        k_buf_t = torch.empty(1, T, H, K, device=q.device, dtype=torch.float32)
        v_buf_t = torch.empty(1, T, HV, V, device=q.device, dtype=torch.float32)
    else:
        k_buf_t = k_buf
        v_buf_t = v_buf

    if "compiled" not in cache:
        # Use sym_int() for B so one compiled kernel handles all batch sizes
        # (no per-B cute.compile JIT). Pattern from prefill (lightning_attn_sm100).
        sym_b = cute.sym_int()
        q_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, H, K), stride_order=(3, 2, 1, 0), assumed_align=16)
        k_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, H, K), stride_order=(3, 2, 1, 0), assumed_align=16)
        v_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, HV, V), stride_order=(3, 2, 1, 0), assumed_align=16)
        o_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, HV, V), stride_order=(3, 2, 1, 0), assumed_align=16)
        h0_fake = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), V, K), stride_order=(2, 1, 0), assumed_align=16)
        decay_fake = make_fake_compact_tensor(cutlass.Float32, (H,), stride_order=(0,), assumed_align=16)
        idx_fake = make_fake_compact_tensor(cutlass.Int32, (sym_b,), stride_order=(0,), assumed_align=16)
        k_buf_fake = make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), T, H, K), stride_order=(3, 2, 1, 0), assumed_align=16
        )
        v_buf_fake = make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), T, HV, V), stride_order=(3, 2, 1, 0), assumed_align=16
        )
        stream_fake = make_fake_stream()

        compiled = cute.compile(
            run_la_verify_kvbuffer_kernel,
            h0_fake,
            decay_fake,
            q_fake,
            k_fake,
            v_fake,
            o_fake,
            idx_fake,
            k_buf_fake,
            v_buf_fake,
            Int32(1),  # grid_size (positional, before Constexpr kwargs)
            scale=softmax_scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            write_kv=write_kv,
            stream=stream_fake,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    num_v_tiles_rt = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles_rt
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_view,
        decay_scales,
        q,
        k,
        v,
        out,
        h0_indices,
        k_buf_t,
        v_buf_t,
        Int32(grid_size),
        stream,
    )


# ===========================================================================
# Warp-shuffle verify kernel (baseline). Dispatched for small T (T < MMA_MIN_T)
# by linear_attention_verify_kvbuffer above. Uses butterfly shuffle reduce for
# the dot products instead of tensor-core MMA — h0 stays in registers (no SMEM
# fragment staging), giving higher occupancy that wins when T is small.
# ===========================================================================


@cute.kernel
def la_verify_kvbuffer_shuffle_kernel(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] fp32 (READ ONLY)
    decay_scales: cute.Tensor,  # [H] fp32
    q: cute.Tensor,  # [B, T, H,  K] fp32
    k: cute.Tensor,  # [B, T, H,  K] fp32
    v: cute.Tensor,  # [B, T, HV, V] fp32
    o: cute.Tensor,  # [B, T, HV, V] fp32 (WRITTEN)
    h0_indices: cute.Tensor,  # [B] int32
    k_buf: cute.Tensor,  # [pool_size, T, H, K] fp32 (WRITTEN when write_kv)
    v_buf: cute.Tensor,  # [pool_size, T, HV, V] fp32 (WRITTEN when write_kv)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    write_kv: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 4

    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    block_idx, _, _ = cute.arch.block_idx()
    i_v = block_idx % num_v_tiles
    tmp = block_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]

    r_q_f32 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k_f32 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    r_decay_pow = cute.make_rmem_tensor(cute.make_layout((T + 1,), stride=(1,)), cutlass.Float32)
    o_partial = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)

    smem = cutlass.utils.SmemAllocator()
    s_qk_scaled = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16)
    # v staged to SMEM (block-shared over the whole v-tile). v has no K dim, so
    # keeping it in per-lane registers wasted 8*T regs/thread and capped occupancy;
    # SMEM costs only T*tile_v*4 bytes and is read warp-uniformly (broadcast).
    sVdata = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16)
    # q (scaled) and k staged to SMEM. They depend only on lane_in_group (NOT on
    # warp/group), so a single copy of 32 K-slices is shared by all 4 warps —
    # this also removes the redundant per-warp q/k loads. Lane-minor layout
    # (T, vec_size, 32) keeps the 32 lanes of a warp on consecutive banks
    # (conflict-free); cost is 2 * T*vec_size*32*4 bytes (~8KB at T=8).
    s_q = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((T, vec_size, threads_per_group), stride=(vec_size * threads_per_group, threads_per_group, 1)),
        16,
    )
    s_k = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((T, vec_size, threads_per_group), stride=(vec_size * threads_per_group, threads_per_group, 1)),
        16,
    )

    if cache_idx >= 0:
        alpha = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)

        # alpha^0 .. alpha^T  (T+1 powers; term1 uses alpha^{t+1})
        r_decay_pow[0] = cutlass.Float32(1.0)
        for t in cutlass.range_constexpr(1, T + 1):
            r_decay_pow[t] = r_decay_pow[t - 1] * alpha

        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        # Stage all T q (scaled) and k (fp32) into SMEM. q/k are warp-independent,
        # so only warp 0 (its 32 lanes cover the full K dim) loads them once.
        # The k_buf write is fused here, replacing the old per-warp redundant store.
        if warp_idx == 0:
            for t in cutlass.range_constexpr(T):
                q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t, i_h, lane_id))
                k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t, i_h, lane_id))
                cute.autovec_copy(q_tile, r_q_f32)
                cute.autovec_copy(k_tile, r_k_f32)
                for j in cutlass.range_constexpr(vec_size):
                    s_q[(t, j, lane_id)] = cutlass.Float32(r_q_f32[j]) * scale
                    s_k[(t, j, lane_id)] = cutlass.Float32(r_k_f32[j])

                # Write k to buffer — gated: only one block per (b, h, t) writes
                if cutlass.const_expr(write_kv):
                    if i_v == 0 and i_hv % (HV // H) == 0:
                        kb_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size), (cache_idx, t, i_h, lane_id))
                        cute.autovec_copy(r_k_f32, kb_tile)

        # Cooperative v load: first tile_v threads each stage one v-row for all T
        # steps into SMEM. v_buf write (when enabled) is fused here — every
        # (cache_idx, t, hv, v_row) is written exactly once by its owning thread.
        v_tile_start = i_v * tile_v
        for t in cutlass.range_constexpr(T):
            if tidx < tile_v:
                v_global_idx = v_tile_start + tidx
                if v_global_idx < V:
                    vv = v[i_n, t, i_hv, v_global_idx]
                    sVdata[(t, tidx)] = cutlass.Float32(vv)
                    if cutlass.const_expr(write_kv):
                        v_buf[(cache_idx, t, i_hv, v_global_idx)] = vv

        cute.arch.barrier()  # q/k/v staged → visible to all warps

        # Phase 1: cooperative QK matrix — 4 warps split T*(T+1)/2 qk dot products.
        # Warp w handles rows where min(t, T-1-t) % 4 == w (head-tail pairing) so that
        # each warp's total row-length is balanced: heavy tail rows are paired with light
        # head rows, making per-warp work ≈ T*(T+1)/8 regardless of T.
        for t_assign in cutlass.range_constexpr(T):
            if min(t_assign, T - 1 - t_assign) % 4 == warp_idx:
                for i in cutlass.range_constexpr(t_assign + 1):
                    qk_lo = cutlass.Float32(0.0)
                    qk_hi = cutlass.Float32(0.0)
                    for j in cutlass.range_constexpr(0, vec_size, 2):
                        qk_lo, qk_hi = hq_dot_pair(
                            s_q[t_assign, j, lane_in_group],
                            s_q[t_assign, j + 1, lane_in_group],
                            s_k[i, j, lane_in_group],
                            s_k[i, j + 1, lane_in_group],
                            qk_lo,
                            qk_hi,
                            use_packed_fma,
                        )
                    qk = qk_lo + qk_hi
                    for offset in [16, 8, 4, 2, 1]:
                        qk += cute.arch.shuffle_sync_bfly(qk, offset=offset, mask=-1, mask_and_clamp=31)
                    if lane_in_group == 0:
                        s_qk_scaled[(t_assign, i)] = r_decay_pow[t_assign - i] * qk

        cute.arch.barrier()  # s_qk_scaled written by Phase 1 → read by Phase 2

        num_row_blocks: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for row_block in cutlass.range_constexpr(num_row_blocks):
            v_base = i_v * tile_v + group_idx * rows_per_group + row_block * ilp_rows
            v_local = group_idx * rows_per_group + row_block * ilp_rows  # offset within sVdata's v-tile
            if v_base + (ilp_rows - 1) < V:
                # Load h_init rows (persistent across the T loop).
                for slot in cutlass.range_constexpr(ilp_rows):
                    h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_base + slot, lane_in_group))
                    cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                for t in cutlass.range_constexpr(T):
                    # term1: alpha^{t+1} * (h_init @ q_t)  (per-slot warp reduce)
                    for slot in cutlass.range_constexpr(ilp_rows):
                        hq_lo = cutlass.Float32(0.0)
                        hq_hi = cutlass.Float32(0.0)
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            hq_lo, hq_hi = hq_dot_pair(
                                r_h[slot, j],
                                r_h[slot, j + 1],
                                s_q[t, j, lane_in_group],
                                s_q[t, j + 1, lane_in_group],
                                hq_lo,
                                hq_hi,
                                use_packed_fma,
                            )
                        hq = hq_lo + hq_hi
                        for offset in [16, 8, 4, 2, 1]:
                            hq += cute.arch.shuffle_sync_bfly(hq, offset=offset, mask=-1, mask_and_clamp=31)
                        o_partial[slot] = r_decay_pow[t + 1] * hq

                    # term2: read pre-computed decay-scaled qk + staged v from SMEM
                    for i in cutlass.range_constexpr(t + 1):
                        coeff = s_qk_scaled[(t, i)]
                        for slot in cutlass.range_constexpr(ilp_rows):
                            o_partial[slot] = o_partial[slot] + coeff * sVdata[(i, v_local + slot)]

                    # writeback (all lanes hold the reduced value; lane 0 writes)
                    if lane_in_group == 0:
                        for slot in cutlass.range_constexpr(ilp_rows):
                            o[(i_n, t, i_hv, v_base + slot)] = o_partial[slot]


@cute.jit
def run_la_verify_kvbuffer_shuffle_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
    grid_size: Int32,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    write_kv: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v

    # s_qk_scaled[T][T] + sVdata[T][tile_v] + s_q/s_k[T][vec_size][32]
    threads_per_group = 32
    smem_bytes = (
        T * T * 4  # s_qk_scaled
        + T * tile_v * 4  # sVdata
        + 2 * T * vec_size * threads_per_group * 4  # s_q + s_k
        + 4 * 16  # per-allocation 16B alignment padding (4 tensors)
    )

    la_verify_kvbuffer_shuffle_kernel(
        h0_source,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        k_buf,
        v_buf,
        vec_size,
        num_v_tiles,
        tile_v,
        scale,
        T,
        H,
        HV,
        K,
        V,
        ilp_rows,
        use_packed_fma,
        write_kv,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _get_compiled_verify_kvbuffer_kernel_shuffle(
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    softmax_scale: float,
    tile_v: int,
    vec_size: int,
    ilp_rows: int,
    use_packed_fma: bool,
    write_kv: bool,
):
    return {}


def linear_attention_verify_kvbuffer_shuffle(
    q: torch.Tensor,  # [B, T, H,  K] fp32
    k: torch.Tensor,  # [B, T, H,  K] fp32
    v: torch.Tensor,  # [B, T, HV, V] fp32
    s: torch.Tensor,  # [pool_size, HV, V, K] fp32, READ ONLY
    out: torch.Tensor,  # [B, T, HV, V] fp32, WRITTEN
    decay_scales: torch.Tensor,  # [H] fp32
    h0_indices: torch.Tensor,  # [B] int32, -1 to skip
    softmax_scale: float,
    T: int,
    k_buf: torch.Tensor | None = None,  # [pool_size, T, H, K] fp32, WRITTEN
    v_buf: torch.Tensor | None = None,  # [pool_size, T, HV, V] fp32, WRITTEN
) -> None:
    """
    Closed-form parallel verify (KVBuffer Eq. 7). Writes out; does not touch s.

    When k_buf and v_buf are provided, also writes k,v to fp32 pool-indexed
    buffers so the caller can free the original k,v tensors after this call
    returns.

    For batch b with h0_indices[b] < 0, out[b] is LEFT UNCHANGED — callers must
    pre-initialize out if downstream code reads those slots.
    """
    B, T_q, H, K = q.shape
    assert T_q == T, f"q.shape[1]={T_q} doesn't match T={T}"
    assert K == 128, f"K={K} != 128: kernel hardcodes K=128 (threads_per_group, lane K-coverage)"
    _, _, HV, V = v.shape
    pool_size = s.shape[0]
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError(f"q/k/v must be torch.float32, got {q.dtype}/{k.dtype}/{v.dtype}")
    if s.dtype != torch.float32:
        raise ValueError(f"s must be torch.float32, got {s.dtype}")
    if out.dtype != torch.float32:
        raise ValueError(f"out must be torch.float32, got {out.dtype}")

    write_kv = k_buf is not None and v_buf is not None
    if (k_buf is None) != (v_buf is None):
        raise ValueError("k_buf and v_buf must both be None or both be provided")
    if write_kv and (k_buf.dtype != torch.float32 or v_buf.dtype != torch.float32):
        raise ValueError(f"k_buf/v_buf must be torch.float32, got {k_buf.dtype}/{v_buf.dtype}")

    cache, (tile_v, vec_size, ilp_rows, use_packed_fma) = _verify_kvbuffer_compile_cache(
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        softmax_scale,
        write_kv=write_kv,
        device=q.device,
    )

    h0_view = s.view(pool_size * HV, V, K)

    # Dummy fp32 tensors when write_kv=False (never accessed by kernel)
    if not write_kv:
        k_buf_t = torch.empty(1, T, H, K, device=q.device, dtype=torch.float32)
        v_buf_t = torch.empty(1, T, HV, V, device=q.device, dtype=torch.float32)
    else:
        k_buf_t = k_buf
        v_buf_t = v_buf

    if "compiled" not in cache:
        sym_b = cute.sym_int()
        q_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, H, K), stride_order=(3, 2, 1, 0), assumed_align=16)
        k_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, H, K), stride_order=(3, 2, 1, 0), assumed_align=16)
        v_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, HV, V), stride_order=(3, 2, 1, 0), assumed_align=16)
        o_fake = make_fake_compact_tensor(cutlass.Float32, (sym_b, T, HV, V), stride_order=(3, 2, 1, 0), assumed_align=16)
        h0_fake = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), V, K), stride_order=(2, 1, 0), assumed_align=16)
        decay_fake = make_fake_compact_tensor(cutlass.Float32, (H,), stride_order=(0,), assumed_align=16)
        idx_fake = make_fake_compact_tensor(cutlass.Int32, (sym_b,), stride_order=(0,), assumed_align=16)
        k_buf_fake = make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), T, H, K), stride_order=(3, 2, 1, 0), assumed_align=16
        )
        v_buf_fake = make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), T, HV, V), stride_order=(3, 2, 1, 0), assumed_align=16
        )
        stream_fake = make_fake_stream()
        compiled = cute.compile(
            run_la_verify_kvbuffer_shuffle_kernel,
            h0_fake,
            decay_fake,
            q_fake,
            k_fake,
            v_fake,
            o_fake,
            idx_fake,
            k_buf_fake,
            v_buf_fake,
            Int32(1),
            scale=softmax_scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_packed_fma=use_packed_fma,
            write_kv=write_kv,
            stream=stream_fake,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    num_v_tiles_rt = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles_rt
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_view,
        decay_scales,
        q,
        k,
        v,
        out,
        h0_indices,
        k_buf_t,
        v_buf_t,
        Int32(grid_size),
        stream,
    )
