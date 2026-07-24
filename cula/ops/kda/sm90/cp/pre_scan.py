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


"""Intracard-CP pre-scan kernel (SM90) — stage 1 (fused S+M chains).

Computes B_seg (S-chain, S0=0) and M_seg (M-chain, M0=I) per segment in one
pass; the per-tile recurrence reuses the FlashKDA-derived K2 math it imports.

160 threads = 4 MMA warps + 1 LOAD warp; outputs fp32 bhvk layout.
"""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.nvgpu.warpgroup import SmemLayoutAtomKind, make_smem_layout_atom
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cula.ops.kda.sm90._common import movm_t_b16
from cula.ops.kda.sm90.k2 import (
    CHUNK,
    D,
    _get_current_custream,
    _get_dummy_int32,
    _make_out_kinter_one_stage,
    _make_state_smem_layout,
)

THREADS_PER_CTA = 160  # 4 MMA warps + 1 LOAD warp
LOAD_WARP_IDX = 4


@cute.kernel
def pre_scan_kernel(
    tma_atom_v: cute.CopyAtom,
    tma_tensor_v: cute.Tensor,
    tma_atom_kd: cute.CopyAtom,
    tma_tensor_kd: cute.Tensor,
    tma_atom_kr: cute.CopyAtom,
    tma_tensor_kr: cute.Tensor,
    tma_atom_inv: cute.CopyAtom,
    tma_tensor_inv: cute.Tensor,
    tma_atom_gt: cute.CopyAtom,
    tma_tensor_gt: cute.Tensor,
    tma_atom_beta: cute.CopyAtom,
    tma_tensor_beta: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Int32,
    T_total: cutlass.Int32,
    seg_cu_tiles: cute.Tensor,
    seg_order: cute.Tensor,  # int32 [S]: launch-slot -> segment index
    b_state_g: cute.Tensor,  # flat fp32 [S*H*D*D] gmem, bhvk
    m_state_g: cute.Tensor,  # flat fp32 [S*H*D*D] gmem, bhvk
    v_tile_starts: cute.Tensor,
    v_tile_actual_lens: cute.Tensor,
    v_is_varlen: cutlass.Constexpr[bool],
):
    # Longest-first launch order; pure reordering.
    seg_slot, head_idx, _ = cute.arch.block_idx()
    seg_idx = cutlass.Int32(seg_order[seg_slot])
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()

    state_layout = _make_state_smem_layout()

    STAGES: cutlass.Constexpr[int] = 2
    cc_stage_layout = cute.make_layout((CHUNK, CHUNK, STAGES), stride=(CHUNK, 1, CHUNK * CHUNK))
    v_kinter_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    v_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    kd_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    kr_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    kr_mninter_atom = make_smem_layout_atom(SmemLayoutAtomKind.MN_INTER, cutlass.BFloat16)
    kr_t_stage_layout = cute.tile_to_shape(kr_mninter_atom, (D, CHUNK, STAGES), order=(1, 0, 2))

    sV = smem.allocate_tensor(cutlass.BFloat16, v_stage_layout, 128)
    sKd = smem.allocate_tensor(cutlass.BFloat16, kd_stage_layout, 128)
    sKr = smem.allocate_tensor(cutlass.BFloat16, kr_stage_layout, 128)
    sINV = smem.allocate_tensor(cutlass.BFloat16, cc_stage_layout, 128)
    sState = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    sM = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    sGt = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D, 1, STAGES), stride=(1, D, D)), 128)
    sBeta = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((CHUNK, 1, STAGES), stride=(1, 64, 64)), 128)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((STAGES,)), 16)
    sMbar_ptr = sMbar.iterator
    sMbarE = smem.allocate_tensor(cutlass.Int64, cute.make_layout((STAGES,)), 16)
    sMbarE_ptr = sMbarE.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(sMbar_ptr + cutlass.Int32(s), cutlass.Int32(1))
                cute.arch.mbarrier_init(sMbarE_ptr + cutlass.Int32(s), cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    gSrc_v = cute.local_tile(tma_tensor_v, (CHUNK, D), (None, None, None))
    tVs, tVg = cpasync.tma_partition(
        tma_atom_v,
        0,
        cute.make_layout(1),
        cute.group_modes(sV, 0, 2),
        cute.group_modes(gSrc_v, 0, 2),
    )
    gSrc_kd = cute.local_tile(tma_tensor_kd, (CHUNK, D), (None, None, None))
    tKDs, tKDg = cpasync.tma_partition(
        tma_atom_kd,
        0,
        cute.make_layout(1),
        cute.group_modes(sKd, 0, 2),
        cute.group_modes(gSrc_kd, 0, 2),
    )
    gSrc_kr = cute.local_tile(tma_tensor_kr, (CHUNK, D), (None, None, None))
    tKRs, tKRg = cpasync.tma_partition(
        tma_atom_kr,
        0,
        cute.make_layout(1),
        cute.group_modes(sKr, 0, 2),
        cute.group_modes(gSrc_kr, 0, 2),
    )
    gSrc_inv = cute.local_tile(tma_tensor_inv, (CHUNK, CHUNK), (None, None, None))
    tIs, tIg = cpasync.tma_partition(
        tma_atom_inv,
        0,
        cute.make_layout(1),
        cute.group_modes(sINV, 0, 2),
        cute.group_modes(gSrc_inv, 0, 2),
    )
    gSrc_gt = cute.local_tile(tma_tensor_gt, (D, 1), (None, None, None))
    tGTs, tGTg = cpasync.tma_partition(
        tma_atom_gt,
        0,
        cute.make_layout(1),
        cute.group_modes(sGt, 0, 2),
        cute.group_modes(gSrc_gt, 0, 2),
    )
    gSrc_beta = cute.local_tile(tma_tensor_beta, (CHUNK, 1), (None, None, None))
    tBs, tBg = cpasync.tma_partition(
        tma_atom_beta,
        0,
        cute.make_layout(1),
        cute.group_modes(sBeta, 0, 2),
        cute.group_modes(gSrc_beta, 0, 2),
    )

    # sState=0, sM=I
    if tidx < D:
        for e in cutlass.range_constexpr(D):
            sState[tidx, e] = cutlass.BFloat16(0.0)
            sM[tidx, e] = cutlass.BFloat16(0.0)
        sM[tidx, tidx] = cutlass.BFloat16(1.0)
    cute.arch.barrier()

    mma_atom = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 4, 1),
        permutation_mnk=(16, 32, 16),
    )
    thr_mma = tiled_mma.get_slice(tidx)

    copy_atom_AB = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_A = cute.make_tiled_copy_A(copy_atom_AB, tiled_mma)
    smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
    copy_atom_B_T = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_B_T = cute.make_tiled_copy_B(copy_atom_B_T, tiled_mma)
    smem_thr_copy_B_T = smem_tiled_copy_B_T.get_slice(tidx)

    tiled_mma_state = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 4, 1),
        permutation_mnk=(16, 32, 16),
    )
    thr_mma_state = tiled_mma_state.get_slice(tidx)
    smem_tiled_copy_A_state = cute.make_tiled_copy_A(copy_atom_B_T, tiled_mma_state)
    smem_thr_copy_A_state = smem_tiled_copy_A_state.get_slice(tidx)

    sKd_s0 = sKd[(None, None, 0)]
    sKd_tile0 = cute.flat_divide(sKd_s0, (CHUNK, 16))
    # state[K_in, D_out] transposed B-view for Phase 1
    sState_B_view = cute.make_tensor(sState.iterator, layout=cute.select(sState.layout, mode=[1, 0]))
    sState_tile = cute.flat_divide(sState_B_view, (D, 16))
    sM_B_view = cute.make_tensor(sM.iterator, layout=cute.select(sM.layout, mode=[1, 0]))
    sM_tile = cute.flat_divide(sM_B_view, (D, 16))

    sKd_ref = sKd_tile0[None, None, 0, 0]
    sState_ref = sState_tile[None, None, 0, 0]

    tCrKd = thr_mma.make_fragment_A(thr_mma.partition_A(sKd_ref))
    tCrState = thr_mma.make_fragment_B(thr_mma.partition_B(sState_ref))
    tCrU = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))
    # Separate accumulator for the M-chain's kd@M, hoisted ahead of the
    # S-chain's dependent stages instead of serializing behind them.
    tCrU_m = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))

    tCrKd_cv = smem_thr_copy_A.retile(tCrKd)
    tCrState_cv = smem_thr_copy_B_T.retile(tCrState)

    # sKr transposed view (MN_INTER swizzle, aliased storage)
    sKr_T_view = cute.make_tensor(sKr.iterator, kr_t_stage_layout)
    sKr_T_view_s0 = sKr_T_view[(None, None, 0)]
    sKr_T_ref = cute.flat_divide(sKr_T_view_s0, (D, CHUNK))[None, None, 0, 0]

    sKr_T_blk_for_frag = cute.flat_divide(sKr_T_view_s0, (CHUNK, CHUNK))[None, None, 0, 0]
    tCrKrA_state_blk = thr_mma_state.make_fragment_A(thr_mma_state.partition_A(sKr_T_blk_for_frag))
    tCrKrA_state_blk_cv = smem_thr_copy_A_state.retile(tCrKrA_state_blk)
    tCrUpd_blk = thr_mma_state.make_fragment_C(tiled_mma_state.partition_shape_C((CHUNK, D)))
    sState_blk_tile = cute.flat_divide(sState, (CHUNK, D))
    sM_blk_tile = cute.flat_divide(sM, (CHUNK, D))
    coord_state_blk = cute.make_identity_tensor((CHUNK, D))
    tCcState_blk = thr_mma_state.partition_C(coord_state_blk)

    tCrU_T = thr_mma.make_fragment_B(thr_mma.partition_B(sKr_T_ref))

    sINV_s0 = sINV[(None, None, 0)]
    sINV_tile0 = cute.flat_divide(sINV_s0, (CHUNK, CHUNK))
    sINV_ref = sINV_tile0[None, None, 0, 0]
    tCrInv = thr_mma.make_fragment_A(thr_mma.partition_A(sINV_ref))
    tCrInv_cv = smem_thr_copy_A.retile(tCrInv)
    tCrU_post = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))
    tCrU_T_post = cute.make_fragment_like(tCrU_T)
    tCrU_post_bf16 = cute.make_fragment_like(tCrU_post, cutlass.BFloat16)
    tCrU_pre_bf16 = cute.make_fragment_like(tCrU, cutlass.BFloat16)

    tile_base = seg_cu_tiles[seg_idx]
    t_tiles = seg_cu_tiles[seg_idx + 1] - tile_base
    TMA_BYTES: cutlass.Constexpr[int] = 3 * CHUNK * D * 2 + CHUNK * CHUNK * 2 + D * 4 + CHUNK * 2

    if warp_idx == LOAD_WARP_IDX:
        s_dyn_l = cutlass.Int32(0)
        phase_emp = cutlass.Int32(1)
        if cutlass.const_expr(v_is_varlen):
            seq_v_start = v_tile_starts[tile_base]
            tma_tensor_v_seq = cute.domain_offset((seq_v_start, 0, 0), tma_tensor_v)
            gSrc_v_seq = cute.local_tile(tma_tensor_v_seq, (CHUNK, D), (None, None, None))
            tVs_seq, tVg_seq = cpasync.tma_partition(
                tma_atom_v,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gSrc_v_seq, 0, 2),
            )
        for t in cutlass.range(t_tiles, unroll=1):
            cute.arch.mbarrier_wait(sMbarE_ptr + s_dyn_l, phase_emp)
            tg_l = tile_base + t
            wt_l = head_idx * total_tiles + tg_l
            bar_l = sMbar_ptr + s_dyn_l
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_l, cutlass.Int32(TMA_BYTES))
            if cutlass.const_expr(v_is_varlen):
                cute.copy(tma_atom_v, tVg_seq[(None, t, 0, head_idx)], tVs_seq[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            else:
                cute.copy(tma_atom_v, tVg[(None, tg_l, 0, head_idx)], tVs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_kd, tKDg[(None, 0, 0, wt_l)], tKDs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_kr, tKRg[(None, 0, 0, wt_l)], tKRs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_inv, tIg[(None, 0, 0, wt_l)], tIs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_gt, tGTg[(None, 0, 0, wt_l)], tGTs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_beta, tBg[(None, 0, 0, wt_l)], tBs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            s_dyn_l = s_dyn_l + cutlass.Int32(1)
            if s_dyn_l == cutlass.Int32(STAGES):
                s_dyn_l = cutlass.Int32(0)
                phase_emp = phase_emp ^ cutlass.Int32(1)
    else:
        phase_full = cutlass.Int32(0)
        s_dyn = cutlass.Int32(0)

        for t in cutlass.range(t_tiles, unroll=1):
            sV_s = sV[(None, None, s_dyn)]
            sKd_tile = cute.flat_divide(sKd[(None, None, s_dyn)], (CHUNK, 16))
            sINV_ref_s = cute.flat_divide(sINV[(None, None, s_dyn)], (CHUNK, CHUNK))[None, None, 0, 0]
            sGt_s = sGt[(None, 0, s_dyn)]
            sBeta_s = sBeta[(None, 0, s_dyn)]

            cute.arch.mbarrier_wait(sMbar_ptr + s_dyn, phase_full)

            if cutlass.const_expr(v_is_varlen):
                actual_len = v_tile_actual_lens[tile_base + t]
                if actual_len < cutlass.Int32(CHUNK):
                    if tidx < D:
                        col_v_tail = tidx
                        for r in cutlass.range_constexpr(CHUNK):
                            if actual_len <= cutlass.Int32(r):
                                sV_s[r, col_v_tail] = cutlass.BFloat16(0.0)
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)

            sKr_T_s = sKr_T_view[(None, None, s_dyn)]
            sKr_T_blk_tile_s = cute.flat_divide(sKr_T_s, (CHUNK, CHUNK))

            # kd @ state
            tCrU.fill(0.0)
            for k in cutlass.range_constexpr(D // 16):
                sKd_k = sKd_tile[None, None, 0, k]
                sState_k = sState_tile[None, None, 0, k]
                cute.copy(smem_tiled_copy_B_T, smem_thr_copy_B_T.partition_S(sState_k), tCrState_cv)
                cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sKd_k), tCrKd_cv)
                cute.gemm(tiled_mma, tCrU, tCrKd, tCrState, tCrU)

            # M-chain front half (kd @ M), hoisted to overlap the S-chain's
            # dependent sigmoid/movmatrix/INV stages.
            tCrU_m.fill(0.0)
            for k in cutlass.range_constexpr(D // 16):
                sKd_k = sKd_tile[None, None, 0, k]
                sM_k = sM_tile[None, None, 0, k]
                cute.copy(smem_tiled_copy_B_T, smem_thr_copy_B_T.partition_S(sM_k), tCrState_cv)
                cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sKd_k), tCrKd_cv)
                cute.gemm(tiled_mma, tCrU_m, tCrKd, tCrState, tCrU_m)

            # sigmoid(beta) * (v - u_pre)
            lane_in_warp = tidx % 32
            Rrow0 = lane_in_warp // 4
            Rrow1 = Rrow0 + 8
            b0 = cutlass.Float32(sBeta_s[Rrow0])
            b1 = cutlass.Float32(sBeta_s[Rrow1])
            sig0 = cutlass.Float32(0.5) * (cute.tanh(b0 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            sig1 = cutlass.Float32(0.5) * (cute.tanh(b1 * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
            tCsV = thr_mma.partition_C(sV_s)
            for i in cutlass.range_constexpr(cute.size(tCrU)):
                ii: cutlass.Constexpr[int] = i
                sub_i: cutlass.Constexpr[int] = (ii % 4) // 2
                sig = sig0 if sub_i == 0 else sig1
                diff = cutlass.Float32(tCsV[ii]) - tCrU[ii]
                tCrU_pre_bf16[ii] = cutlass.BFloat16(diff * sig)
            tCrU_pre_u32 = cute.recast_tensor(tCrU_pre_bf16, dtype=cutlass.Int32)
            tCrU_T_u32 = cute.recast_tensor(tCrU_T, dtype=cutlass.Int32)
            for i in cutlass.range_constexpr(cute.size(tCrU_pre_u32)):
                ii: cutlass.Constexpr[int] = i
                tCrU_T_u32[ii] = movm_t_b16(cutlass.Int32(tCrU_pre_u32[ii]))

            # U_post = INV @ U_pre
            cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sINV_ref_s), tCrInv_cv)
            tCrU_post.fill(0.0)
            cute.gemm(tiled_mma, tCrU_post, tCrInv, tCrU_T, tCrU_post)
            for i in cutlass.range_constexpr(cute.size(tCrU_post)):
                ii: cutlass.Constexpr[int] = i
                tCrU_post_bf16[ii] = cutlass.BFloat16(tCrU_post[ii])
            tCrU_post_u32 = cute.recast_tensor(tCrU_post_bf16, dtype=cutlass.Int32)
            tCrU_T_post_u32 = cute.recast_tensor(tCrU_T_post, dtype=cutlass.Int32)
            for i in cutlass.range_constexpr(cute.size(tCrU_post_u32)):
                ii: cutlass.Constexpr[int] = i
                tCrU_T_post_u32[ii] = movm_t_b16(cutlass.Int32(tCrU_post_u32[ii]))

            # State update: state = state*gt + kr^T @ U
            M_BLOCKS: cutlass.Constexpr[int] = D // CHUNK
            for mi in cutlass.range_constexpr(M_BLOCKS):
                sKr_T_blk_s = sKr_T_blk_tile_s[None, None, mi, 0]
                cute.copy(
                    smem_tiled_copy_A_state,
                    smem_thr_copy_A_state.partition_S(sKr_T_blk_s),
                    tCrKrA_state_blk_cv,
                )
                tCrUpd_blk.fill(0.0)
                cute.gemm(tiled_mma_state, tCrUpd_blk, tCrKrA_state_blk, tCrU_T_post, tCrUpd_blk)

                sState_blk = sState_blk_tile[None, None, mi, 0]
                tCsState_blk = thr_mma_state.partition_C(sState_blk)
                state_frag_blk = cute.make_fragment_like(tCsState_blk, cutlass.BFloat16)
                gt_frag_blk = cute.make_fragment_like(tCsState_blk, cutlass.Float32)
                m_off: cutlass.Constexpr[int] = mi * CHUNK
                for i in cutlass.range_constexpr(cute.size(state_frag_blk)):
                    ii: cutlass.Constexpr[int] = i
                    state_frag_blk[ii] = tCsState_blk[ii]
                    gt_frag_blk[ii] = sGt_s[m_off + tCcState_blk[ii][0]]
                for i in cutlass.range_constexpr(cute.size(tCrUpd_blk)):
                    ii: cutlass.Constexpr[int] = i
                    old = cutlass.Float32(state_frag_blk[ii]) * gt_frag_blk[ii]
                    tCsState_blk[ii] = cutlass.BFloat16(old + tCrUpd_blk[ii])

            # M-chain: same pipeline with V := 0 (duality).
            # (kd @ M already issued above.)

            # sigmoid(beta) * (0 - u_pre)
            for i in cutlass.range_constexpr(cute.size(tCrU_m)):
                ii: cutlass.Constexpr[int] = i
                sub_i: cutlass.Constexpr[int] = (ii % 4) // 2
                sig = sig0 if sub_i == 0 else sig1
                diff = cutlass.Float32(0.0) - tCrU_m[ii]
                tCrU_pre_bf16[ii] = cutlass.BFloat16(diff * sig)
            for i in cutlass.range_constexpr(cute.size(tCrU_pre_u32)):
                ii: cutlass.Constexpr[int] = i
                tCrU_T_u32[ii] = movm_t_b16(cutlass.Int32(tCrU_pre_u32[ii]))

            # U_post = INV @ U_pre
            tCrU_post.fill(0.0)
            cute.gemm(tiled_mma, tCrU_post, tCrInv, tCrU_T, tCrU_post)
            for i in cutlass.range_constexpr(cute.size(tCrU_post)):
                ii: cutlass.Constexpr[int] = i
                tCrU_post_bf16[ii] = cutlass.BFloat16(tCrU_post[ii])
            for i in cutlass.range_constexpr(cute.size(tCrU_post_u32)):
                ii: cutlass.Constexpr[int] = i
                tCrU_T_post_u32[ii] = movm_t_b16(cutlass.Int32(tCrU_post_u32[ii]))

            # State update': M = M*gt + kr^T @ U
            for mi in cutlass.range_constexpr(M_BLOCKS):
                sKr_T_blk_s = sKr_T_blk_tile_s[None, None, mi, 0]
                cute.copy(
                    smem_tiled_copy_A_state,
                    smem_thr_copy_A_state.partition_S(sKr_T_blk_s),
                    tCrKrA_state_blk_cv,
                )
                tCrUpd_blk.fill(0.0)
                cute.gemm(tiled_mma_state, tCrUpd_blk, tCrKrA_state_blk, tCrU_T_post, tCrUpd_blk)

                sM_blk = sM_blk_tile[None, None, mi, 0]
                tCsM_blk = thr_mma_state.partition_C(sM_blk)
                m_frag_blk = cute.make_fragment_like(tCsM_blk, cutlass.BFloat16)
                gt_frag_blk_m = cute.make_fragment_like(tCsM_blk, cutlass.Float32)
                m_off2: cutlass.Constexpr[int] = mi * CHUNK
                for i in cutlass.range_constexpr(cute.size(m_frag_blk)):
                    ii: cutlass.Constexpr[int] = i
                    m_frag_blk[ii] = tCsM_blk[ii]
                    gt_frag_blk_m[ii] = sGt_s[m_off2 + tCcState_blk[ii][0]]
                for i in cutlass.range_constexpr(cute.size(tCrUpd_blk)):
                    ii: cutlass.Constexpr[int] = i
                    old = cutlass.Float32(m_frag_blk[ii]) * gt_frag_blk_m[ii]
                    tCsM_blk[ii] = cutlass.BFloat16(old + tCrUpd_blk[ii])

            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            cute.arch.fence_view_async_shared()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(sMbarE_ptr + s_dyn)
            s_dyn = s_dyn + cutlass.Int32(1)
            if s_dyn == cutlass.Int32(STAGES):
                s_dyn = cutlass.Int32(0)
                phase_full = phase_full ^ cutlass.Int32(1)
    cute.arch.barrier()
    # Epilogue: write both states fp32. The index swap below (gmem[d_out, tidx] =
    # smem[tidx, d_out]) stores the TRANSPOSE S^T / M^T on purpose — merge consumes
    # this transposed convention (carry @ M^T), so do not "fix" the orientation here.
    state_base_f = cutlass.Int32(seg_idx) * cutlass.Int32(H * D * D) + cutlass.Int32(head_idx) * cutlass.Int32(D * D)
    if tidx < D:
        for d_out in cutlass.range_constexpr(D):
            b_state_g[state_base_f + cutlass.Int32(d_out * D) + cutlass.Int32(tidx)] = cutlass.Float32(sState[tidx, d_out])
            m_state_g[state_base_f + cutlass.Int32(d_out * D) + cutlass.Int32(tidx)] = cutlass.Float32(sM[tidx, d_out])


@cute.jit
def run_pre_scan(
    v: cute.Tensor,
    beta: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    seg_cu_tiles: cute.Tensor,
    seg_order: cute.Tensor,
    b_state_g: cute.Tensor,  # flat fp32 [S*H*D*D]
    m_state_g: cute.Tensor,  # flat fp32 [S*H*D*D]
    v_tile_starts: cute.Tensor,
    v_tile_actual_lens: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Int32,
    T_total: cutlass.Int32,
    v_is_varlen: cutlass.Constexpr[bool],
    S: cutlass.Int32,
    stream: cuda_drv.CUstream,
):
    cc_smem = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
    kinter_smem = _make_out_kinter_one_stage()

    def make_thd_atom(t, op):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(op, view, kinter_smem, (CHUNK, D))

    def make_ws_qkd_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, D, total_tiles * H), stride=(D, 1, CHUNK * D)),
        )
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), view, kinter_smem, (CHUNK, D))

    def make_ws_cc_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, CHUNK, total_tiles * H), stride=(CHUNK, 1, CHUNK * CHUNK)),
        )
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), view, cc_smem, (CHUNK, CHUNK))

    tma_atom_v, tma_tensor_v = make_thd_atom(v, cpasync.CopyBulkTensorTileG2SOp())
    tma_atom_kd, tma_tensor_kd = make_ws_qkd_atom(ws_kd)
    tma_atom_kr, tma_tensor_kr = make_ws_qkd_atom(ws_kr)
    tma_atom_inv, tma_tensor_inv = make_ws_cc_atom(ws_inv)

    gt_smem = cute.make_layout((D, 1), stride=(1, D))
    beta_smem = cute.make_layout((CHUNK, 1), stride=(1, 64))

    def make_gt_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((D, 1, total_tiles * H), stride=(1, D, D)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            view,
            gt_smem,
            (D, 1),
        )

    def make_beta_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, 1, total_tiles * H), stride=(1, CHUNK, CHUNK)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            view,
            beta_smem,
            (CHUNK, 1),
        )

    tma_atom_gt, tma_tensor_gt = make_gt_atom(ws_gt)
    tma_atom_beta, tma_tensor_beta = make_beta_atom(beta)

    STAGES_LOCAL = 2
    smem_bytes = (
        2 * D * D * 2
        + STAGES_LOCAL * 3 * (CHUNK * D * 2)
        + STAGES_LOCAL * (CHUNK * CHUNK * 2)
        + STAGES_LOCAL * (D * 4)
        + STAGES_LOCAL * (64 * 2)
        + STAGES_LOCAL * 8
        + STAGES_LOCAL * 8
        + 2048
    )

    pre_scan_kernel(
        tma_atom_v,
        tma_tensor_v,
        tma_atom_kd,
        tma_tensor_kd,
        tma_atom_kr,
        tma_tensor_kr,
        tma_atom_inv,
        tma_tensor_inv,
        tma_atom_gt,
        tma_tensor_gt,
        tma_atom_beta,
        tma_tensor_beta,
        H,
        total_tiles,
        T_total,
        seg_cu_tiles,
        seg_order,
        b_state_g,
        m_state_g,
        v_tile_starts,
        v_tile_actual_lens,
        v_is_varlen,
    ).launch(
        grid=(S, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# Compile cache keyed on CONFIG ONLY — total_tiles/T_total/S are dynamic
# cutlass.Int32, so one compiled kernel serves every batch shape.
_prescan_kernel_cache: dict = {}


def _compile_pre_scan(H, v_is_varlen):
    sym_t = cute.sym_int()  # T_total (v token extent)
    sym_bt = cute.sym_int()  # beta/ws_beta (total_tiles*CHUNK*H)
    sym_qk = cute.sym_int()  # ws_kd/kr (total_tiles*H*CHUNK*D)
    sym_gt = cute.sym_int()  # ws_gt    (total_tiles*H*D)
    sym_cc = cute.sym_int()  # ws_inv   (total_tiles*H*CHUNK*CHUNK)
    sym_seg = cute.sym_int()  # seg_cu_tiles (S+1)
    sym_so = cute.sym_int()  # seg_order (S) — own sym: length differs from seg_cu_tiles
    sym_st = cute.sym_int()  # b_flat / m_flat (S*H*D*D)
    sym_vs = cute.sym_int()  # v_tile_starts
    sym_va = cute.sym_int()  # v_tile_actual_lens

    v_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, H, D), stride_order=(2, 1, 0), assumed_align=16)
    beta_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_bt,), assumed_align=16)
    ws_kd_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_kr_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_gt_fake = make_fake_compact_tensor(cutlass.Float32, (sym_gt,), assumed_align=16)
    ws_inv_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_cc,), assumed_align=16)
    seg_fake = make_fake_compact_tensor(cutlass.Int32, (sym_seg,), assumed_align=4)
    so_fake = make_fake_compact_tensor(cutlass.Int32, (sym_so,), assumed_align=4)
    b_fake = make_fake_compact_tensor(cutlass.Float32, (sym_st,), assumed_align=16)
    m_fake = make_fake_compact_tensor(cutlass.Float32, (sym_st,), assumed_align=16)
    vts_fake = make_fake_compact_tensor(cutlass.Int32, (sym_vs,), assumed_align=4)
    vtal_fake = make_fake_compact_tensor(cutlass.Int32, (sym_va,), assumed_align=4)
    stream_fake = make_fake_stream()

    return cute.compile(
        run_pre_scan,
        v_fake,
        beta_fake,
        ws_kd_fake,
        ws_kr_fake,
        ws_gt_fake,
        ws_inv_fake,
        seg_fake,
        so_fake,
        b_fake,
        m_fake,
        vts_fake,
        vtal_fake,
        H,  # Constexpr -> baked
        cutlass.Int32(1),  # total_tiles -> runtime (placeholder)
        cutlass.Int32(1),  # T_total -> runtime
        v_is_varlen,  # Constexpr
        cutlass.Int32(1),  # S -> runtime
        stream_fake,
        options="--enable-tvm-ffi",
    )


def _get_compiled_pre_scan(H, v_is_varlen):
    key = (H, v_is_varlen)
    cached = _prescan_kernel_cache.get(key)
    if cached is None:
        cached = _compile_pre_scan(H, v_is_varlen)
        _prescan_kernel_cache[key] = cached
    return cached


def launch_pre_scan(
    v: torch.Tensor,
    beta: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_kr: torch.Tensor,
    ws_gt: torch.Tensor,
    ws_inv: torch.Tensor,
    b_state: torch.Tensor,  # [S, H, D, D] fp32, written (bhvk)
    m_state: torch.Tensor,  # [S, H, D, D] fp32, written (bhvk)
    seg_cu_tiles: torch.Tensor,  # int32 [S+1], global tile prefix sum
    *,
    total_tiles: int,
    seg_order: torch.Tensor,  # int32 launch list
    v_tile_starts: torch.Tensor | None = None,  # per-tile packed offset (native varlen)
    v_tile_actual_lens: torch.Tensor | None = None,  # per-tile valid rows (partial-tile mask)
) -> None:
    """Launch fused pre_scan: B_seg and M_seg for the segments in seg_order."""
    B, T, H, K = v.shape
    T_total = B * T
    v_is_varlen = v_tile_starts is not None
    if not v_is_varlen:
        dummy = _get_dummy_int32(v.device)
        v_tile_starts = dummy
        v_tile_actual_lens = dummy
    compiled_fn = _get_compiled_pre_scan(H, v_is_varlen)
    stream = _get_current_custream()
    # TMA descriptors are (re)built inside run_pre_scan from the dynamic Int32
    # dims every launch, so one compiled kernel serves all shapes.
    compiled_fn(
        v.view(T_total, H, D),
        beta,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        seg_cu_tiles,
        seg_order,
        b_state.reshape(-1),
        m_state.reshape(-1),
        v_tile_starts,
        v_tile_actual_lens,
        total_tiles,
        T_total,
        seg_order.numel(),
        stream,
    )
