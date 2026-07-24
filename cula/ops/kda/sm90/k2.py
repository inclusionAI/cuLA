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

# Copyright (c) 2026 MoonshotAI
# Licensed under the MIT License.
# Based on MoonshotAI/FlashKDA (https://github.com/MoonshotAI/FlashKDA)

"""
CuteDSL port of FlashKDA K2 (Recurrence).

Produces workspace tensors consumed by subsequent stages.
"""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.nvgpu.warpgroup import SmemLayoutAtomKind, make_smem_layout_atom
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

CHUNK: int = 16
D: int = 128


def _make_state_smem_layout():
    atom = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3),
        0,
        cute.make_layout((8, 64), stride=(64, 1)),
    )
    return cute.tile_to_shape(atom, (D, D), (0, 1))


from cula.ops.kda.sm90._common import _stream_key, movm_t_b16  # noqa: E402


def _make_out_kinter_one_stage():
    """K_INTER swizzled (CHUNK, D) bf16 SMEM layout."""
    atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    return cute.tile_to_shape(atom, (CHUNK, D), order=(0, 1))


THREADS_PER_CTA = 192  # 128 compute (4 MMA warps) + 32 load + 32 store
N_WARPS = 4
LOAD_WARP_IDX = 4
STORE_WARP_IDX = 5


@cute.kernel
def k2_kernel(
    tma_atom_v: cute.CopyAtom,
    tma_tensor_v: cute.Tensor,
    tma_atom_kd: cute.CopyAtom,
    tma_tensor_kd: cute.Tensor,
    tma_atom_qd: cute.CopyAtom,
    tma_tensor_qd: cute.Tensor,
    tma_atom_kr: cute.CopyAtom,
    tma_tensor_kr: cute.Tensor,
    tma_atom_inv: cute.CopyAtom,
    tma_tensor_inv: cute.Tensor,
    tma_atom_mqk: cute.CopyAtom,
    tma_tensor_mqk: cute.Tensor,
    tma_atom_out: cute.CopyAtom,
    tma_tensor_out: cute.Tensor,
    out_gmem: cute.Tensor,
    tma_atom_gt: cute.CopyAtom,
    tma_tensor_gt: cute.Tensor,
    tma_atom_beta: cute.CopyAtom,
    tma_tensor_beta: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Int32,
    cu_seqlens_tiles: cute.Tensor,
    seq_order: cute.Tensor,  # int32 [N]: launch-slot -> sequence/segment index
    v_tile_starts: cute.Tensor,
    v_tile_actual_lens: cute.Tensor,
    initial_state_g: cute.Tensor,  # flat fp32 [N*H*D*D] gmem (layout per state_transposed)
    final_state_g: cute.Tensor,  # flat fp32 [N*H*D*D] gmem (layout per state_transposed)
    has_initial_state: cutlass.Constexpr[bool],
    has_final_state: cutlass.Constexpr[bool],
    state_transposed: cutlass.Constexpr[bool],
    v_is_varlen: cutlass.Constexpr[bool],
):
    # Longest-first launch order: pure reordering — no math changes.
    seq_slot, head_idx, _ = cute.arch.block_idx()
    seq_idx = cutlass.Int32(seq_order[seq_slot])
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()

    state_layout = _make_state_smem_layout()

    STAGES: cutlass.Constexpr[int] = 2
    OUT_STAGES: cutlass.Constexpr[int] = 2
    cc_stage_layout = cute.make_layout((CHUNK, CHUNK, STAGES), stride=(CHUNK, 1, CHUNK * CHUNK))
    out_kinter_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    out_stage_layout = cute.tile_to_shape(out_kinter_atom, (CHUNK, D, OUT_STAGES), order=(0, 1, 2))
    v_kinter_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    v_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    kd_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    qd_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    kr_stage_layout = cute.tile_to_shape(v_kinter_atom, (CHUNK, D, STAGES), order=(0, 1, 2))
    # MN_INTER transposed view of sKr for state update (same bytes, transposed swizzle).
    kr_mninter_atom = make_smem_layout_atom(SmemLayoutAtomKind.MN_INTER, cutlass.BFloat16)
    kr_t_stage_layout = cute.tile_to_shape(kr_mninter_atom, (D, CHUNK, STAGES), order=(1, 0, 2))

    sV = smem.allocate_tensor(cutlass.BFloat16, v_stage_layout, 128)
    sKd = smem.allocate_tensor(cutlass.BFloat16, kd_stage_layout, 128)
    sQd = smem.allocate_tensor(cutlass.BFloat16, qd_stage_layout, 128)
    sKr = smem.allocate_tensor(cutlass.BFloat16, kr_stage_layout, 128)
    sINV = smem.allocate_tensor(cutlass.BFloat16, cc_stage_layout, 128)
    sMqk = smem.allocate_tensor(cutlass.BFloat16, cc_stage_layout, 128)
    sOut = smem.allocate_tensor(cutlass.BFloat16, out_stage_layout, 128)
    sState = smem.allocate_tensor(cutlass.BFloat16, state_layout, 128)
    sGt = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D, 1, STAGES), stride=(1, D, D)), 128)
    sBeta = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((CHUNK, 1, STAGES), stride=(1, 64, 64)), 128)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((STAGES,)), 16)
    sMbar_ptr = sMbar.iterator
    sMbarE = smem.allocate_tensor(cutlass.Int64, cute.make_layout((STAGES,)), 16)
    sMbarE_ptr = sMbarE.iterator
    sMbarSF = smem.allocate_tensor(cutlass.Int64, cute.make_layout((OUT_STAGES,)), 16)
    sMbarSF_ptr = sMbarSF.iterator
    sMbarSE = smem.allocate_tensor(cutlass.Int64, cute.make_layout((OUT_STAGES,)), 16)
    sMbarSE_ptr = sMbarSE.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(sMbar_ptr + cutlass.Int32(s), cutlass.Int32(1))
                cute.arch.mbarrier_init(sMbarE_ptr + cutlass.Int32(s), cutlass.Int32(1))
            for s in cutlass.range_constexpr(OUT_STAGES):
                cute.arch.mbarrier_init(sMbarSF_ptr + cutlass.Int32(s), cutlass.Int32(1))
                cute.arch.mbarrier_init(sMbarSE_ptr + cutlass.Int32(s), cutlass.Int32(1))
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
    gSrc_qd = cute.local_tile(tma_tensor_qd, (CHUNK, D), (None, None, None))
    tQDs, tQDg = cpasync.tma_partition(
        tma_atom_qd,
        0,
        cute.make_layout(1),
        cute.group_modes(sQd, 0, 2),
        cute.group_modes(gSrc_qd, 0, 2),
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
    gSrc_mqk = cute.local_tile(tma_tensor_mqk, (CHUNK, CHUNK), (None, None, None))
    tMs, tMg = cpasync.tma_partition(
        tma_atom_mqk,
        0,
        cute.make_layout(1),
        cute.group_modes(sMqk, 0, 2),
        cute.group_modes(gSrc_mqk, 0, 2),
    )
    gDst_o = cute.local_tile(tma_tensor_out, (CHUNK, D), (None, None, None))
    tOs, tOg = cpasync.tma_partition(
        tma_atom_out,
        0,
        cute.make_layout(1),
        cute.group_modes(sOut, 0, 2),
        cute.group_modes(gDst_o, 0, 2),
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

    # Load initial_state -> sState[K_in, D_out]
    if has_initial_state:
        state_base = cutlass.Int32(seq_idx) * cutlass.Int32(H * D * D) + cutlass.Int32(head_idx) * cutlass.Int32(D * D)
        if tidx < D:
            if state_transposed:
                for k_in in cutlass.range_constexpr(D):
                    sState[k_in, tidx] = cutlass.BFloat16(
                        initial_state_g[state_base + cutlass.Int32(k_in * D) + cutlass.Int32(tidx)]
                    )
            else:
                for d_out in cutlass.range_constexpr(D):
                    sState[tidx, d_out] = cutlass.BFloat16(
                        initial_state_g[state_base + cutlass.Int32(d_out * D) + cutlass.Int32(tidx)]
                    )
    else:
        if tidx < D:
            for e in cutlass.range_constexpr(D):
                sState[tidx, e] = cutlass.BFloat16(0.0)
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

    copy_atom_stsm = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    smem_tiled_store_C = cute.make_tiled_copy_C_atom(copy_atom_stsm, tiled_mma)
    smem_thr_store_C = smem_tiled_store_C.get_slice(tidx)

    tiled_mma_state = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 4, 1),
        permutation_mnk=(16, 32, 16),
    )
    thr_mma_state = tiled_mma_state.get_slice(tidx)
    smem_tiled_copy_A_state = cute.make_tiled_copy_A(copy_atom_B_T, tiled_mma_state)
    smem_thr_copy_A_state = smem_tiled_copy_A_state.get_slice(tidx)

    sKd_s0 = sKd[(None, None, 0)]
    sQd_s0 = sQd[(None, None, 0)]
    sKd_tile0 = cute.flat_divide(sKd_s0, (CHUNK, 16))
    sQd_tile0 = cute.flat_divide(sQd_s0, (CHUNK, 16))
    # sState[K_in, D_out] transposed view for B-operand in Phase 1/4.
    sState_B_view = cute.make_tensor(sState.iterator, layout=cute.select(sState.layout, mode=[1, 0]))
    sState_tile = cute.flat_divide(sState_B_view, (D, 16))

    sKd_ref = sKd_tile0[None, None, 0, 0]
    sQd_ref = sQd_tile0[None, None, 0, 0]
    sState_ref = sState_tile[None, None, 0, 0]

    tCrKd = thr_mma.make_fragment_A(thr_mma.partition_A(sKd_ref))
    tCrQd = thr_mma.make_fragment_A(thr_mma.partition_A(sQd_ref))
    tCrState = thr_mma.make_fragment_B(thr_mma.partition_B(sState_ref))
    tCrU = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))
    tCrOut = thr_mma.make_fragment_C(tiled_mma.partition_shape_C((CHUNK, D)))

    tCrKd_cv = smem_thr_copy_A.retile(tCrKd)
    tCrQd_cv = smem_thr_copy_A.retile(tCrQd)
    tCrState_cv = smem_thr_copy_B_T.retile(tCrState)

    sMqk_s0 = sMqk[(None, None, 0)]
    sMqk_tile0 = cute.flat_divide(sMqk_s0, (CHUNK, CHUNK))
    sMqk_ref = sMqk_tile0[None, None, 0, 0]
    tCrMqk = thr_mma.make_fragment_A(thr_mma.partition_A(sMqk_ref))
    tCrMqk_cv = smem_thr_copy_A.retile(tCrMqk)

    # State update: MN_INTER transposed view of sKr.
    sKr_T_view = cute.make_tensor(sKr.iterator, kr_t_stage_layout)
    sKr_T_view_s0 = sKr_T_view[(None, None, 0)]
    sKr_T_ref = cute.flat_divide(sKr_T_view_s0, (D, CHUNK))[None, None, 0, 0]

    sKr_T_blk_for_frag = cute.flat_divide(sKr_T_view_s0, (CHUNK, CHUNK))[None, None, 0, 0]
    tCrKrA_state_blk = thr_mma_state.make_fragment_A(thr_mma_state.partition_A(sKr_T_blk_for_frag))
    tCrKrA_state_blk_cv = smem_thr_copy_A_state.retile(tCrKrA_state_blk)
    tCrUpd_blk = thr_mma_state.make_fragment_C(tiled_mma_state.partition_shape_C((CHUNK, D)))
    sState_blk_tile = cute.flat_divide(sState, (CHUNK, D))
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

    tile_base = cu_seqlens_tiles[seq_idx]
    t_tiles = cu_seqlens_tiles[seq_idx + 1] - tile_base  # dynamic tile count for this sequence
    TMA_BYTES: cutlass.Constexpr[int] = 4 * CHUNK * D * 2 + 2 * CHUNK * CHUNK * 2 + D * 4 + CHUNK * 2

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
            cute.copy(tma_atom_qd, tQDg[(None, 0, 0, wt_l)], tQDs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_kr, tKRg[(None, 0, 0, wt_l)], tKRs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_inv, tIg[(None, 0, 0, wt_l)], tIs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_mqk, tMg[(None, 0, 0, wt_l)], tMs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_gt, tGTg[(None, 0, 0, wt_l)], tGTs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            cute.copy(tma_atom_beta, tBg[(None, 0, 0, wt_l)], tBs[(None, s_dyn_l)], tma_bar_ptr=bar_l)
            s_dyn_l = s_dyn_l + cutlass.Int32(1)
            if s_dyn_l == cutlass.Int32(STAGES):
                s_dyn_l = cutlass.Int32(0)
                phase_emp = phase_emp ^ cutlass.Int32(1)
    elif warp_idx == STORE_WARP_IDX:
        if cutlass.const_expr(v_is_varlen):
            universal_copy_bits: cutlass.Constexpr[int] = 128
            async_copy_elems: cutlass.Constexpr[int] = universal_copy_bits // cutlass.BFloat16.width
            atom_universal_copy_o = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=universal_copy_bits,
            )
            o_thr_layout = cute.make_ordered_layout(
                (THREADS_PER_CTA // 6 // (D // async_copy_elems), D // async_copy_elems),
                order=(1, 0),
            )
            o_val_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_o = cute.make_tiled_copy_tv(
                atom_universal_copy_o,
                o_thr_layout,
                o_val_layout,
            )
            gmem_thr_copy_o = gmem_tiled_copy_o.get_slice(tidx % 32)
            seq_out_start = v_tile_starts[tile_base]
            tma_tensor_out_seq = cute.domain_offset((seq_out_start, 0, 0), tma_tensor_out)
            gDst_o_seq = cute.local_tile(tma_tensor_out_seq, (CHUNK, D), (None, None, None))
            tOs_seq, tOg_seq = cpasync.tma_partition(
                tma_atom_out,
                0,
                cute.make_layout(1),
                cute.group_modes(sOut, 0, 2),
                cute.group_modes(gDst_o_seq, 0, 2),
            )

        s_out_s = cutlass.Int32(0)
        phase_sf = cutlass.Int32(0)
        for t in cutlass.range(t_tiles, unroll=1):
            cute.arch.mbarrier_wait(sMbarSF_ptr + s_out_s, phase_sf)
            t_g_s = tile_base + t
            if cutlass.const_expr(v_is_varlen):
                actual_len_o = v_tile_actual_lens[t_g_s]
                if actual_len_o == cutlass.Int32(CHUNK):
                    cute.copy(tma_atom_out, tOs_seq[(None, s_out_s)], tOg_seq[(None, t, 0, head_idx)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                else:
                    out_start = v_tile_starts[t_g_s]
                    sOut_stage = sOut[(None, None, s_out_s)]
                    tOsO = gmem_thr_copy_o.partition_S(sOut_stage)
                    cO = cute.make_identity_tensor((CHUNK, D))
                    tOcO = gmem_thr_copy_o.partition_S(cO)
                    tOrO = cute.make_fragment_like(tOsO, cutlass.BFloat16)
                    cute.autovec_copy(tOsO, tOrO)

                    out_chunk_raw = out_gmem.iterator + out_start * H * D + head_idx * D
                    out_chunk_ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        out_chunk_raw.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    out_stride_t: cutlass.Constexpr[int] = H * D
                    gOut_chunk = cute.make_tensor(
                        out_chunk_ptr,
                        cute.make_layout((CHUNK, D), stride=(out_stride_t, 1)),
                    )
                    tOgO = gmem_thr_copy_o.partition_D(gOut_chunk)
                    for m1 in cutlass.range_constexpr(cute.size(tOsO.shape[1])):
                        row_coord = tOcO[(0, 0), m1, 0][0]
                        if row_coord < actual_len_o:
                            cute.autovec_copy(tOrO[(None, m1, None)], tOgO[(None, m1, None)])
            else:
                cute.copy(tma_atom_out, tOs[(None, s_out_s)], tOg[(None, t_g_s, 0, head_idx)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(sMbarSE_ptr + s_out_s)
            s_out_s = s_out_s + cutlass.Int32(1)
            if s_out_s == cutlass.Int32(OUT_STAGES):
                s_out_s = cutlass.Int32(0)
                phase_sf = phase_sf ^ cutlass.Int32(1)
    else:
        phase_full = cutlass.Int32(0)
        s_dyn = cutlass.Int32(0)
        s_out = cutlass.Int32(0)
        phase_se = cutlass.Int32(1)

        for t in cutlass.range(t_tiles, unroll=1):
            sV_s = sV[(None, None, s_dyn)]
            sKd_tile = cute.flat_divide(sKd[(None, None, s_dyn)], (CHUNK, 16))
            sQd_tile = cute.flat_divide(sQd[(None, None, s_dyn)], (CHUNK, 16))
            sINV_ref_s = cute.flat_divide(sINV[(None, None, s_dyn)], (CHUNK, CHUNK))[None, None, 0, 0]
            sMqk_ref_s = cute.flat_divide(sMqk[(None, None, s_dyn)], (CHUNK, CHUNK))[None, None, 0, 0]
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

            # kd @ state and qd @ state share the same state tile load.
            tCrU.fill(0.0)
            tCrOut.fill(0.0)
            for k in cutlass.range_constexpr(D // 16):
                sKd_k = sKd_tile[None, None, 0, k]
                sQd_k = sQd_tile[None, None, 0, k]
                sState_k = sState_tile[None, None, 0, k]
                cute.copy(smem_tiled_copy_B_T, smem_thr_copy_B_T.partition_S(sState_k), tCrState_cv)
                cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sKd_k), tCrKd_cv)
                cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sQd_k), tCrQd_cv)
                cute.gemm(tiled_mma, tCrU, tCrKd, tCrState, tCrU)
                cute.gemm(tiled_mma, tCrOut, tCrQd, tCrState, tCrOut)

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

            # out += Mqk @ U_post
            cute.copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(sMqk_ref_s), tCrMqk_cv)
            cute.gemm(tiled_mma, tCrOut, tCrMqk, tCrU_T_post, tCrOut)

            cute.arch.mbarrier_wait(sMbarSE_ptr + s_out, phase_se)
            sOut_s = sOut[(None, None, s_out)]
            tCrOut_bf16 = cute.make_fragment_like(tCrOut, cutlass.BFloat16)
            for i in cutlass.range_constexpr(cute.size(tCrOut)):
                tCrOut_bf16[i] = cutlass.BFloat16(tCrOut[i])
            cute.copy(
                smem_tiled_store_C,
                smem_thr_store_C.retile(tCrOut_bf16),
                smem_thr_store_C.partition_D(sOut_s),
            )

            # State update: state = state*gt + kr^T @ U (blocked M-loop)
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
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            cute.arch.fence_view_async_shared()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(sMbarSF_ptr + s_out)
                    cute.arch.mbarrier_arrive(sMbarE_ptr + s_dyn)
            s_dyn = s_dyn + cutlass.Int32(1)
            if s_dyn == cutlass.Int32(STAGES):
                s_dyn = cutlass.Int32(0)
                phase_full = phase_full ^ cutlass.Int32(1)
            s_out = s_out + cutlass.Int32(1)
            if s_out == cutlass.Int32(OUT_STAGES):
                s_out = cutlass.Int32(0)
                phase_se = phase_se ^ cutlass.Int32(1)
    cute.arch.barrier()
    if has_final_state:
        state_base_f = cutlass.Int32(seq_idx) * cutlass.Int32(H * D * D) + cutlass.Int32(head_idx) * cutlass.Int32(D * D)
        if tidx < D:
            if state_transposed:
                for k_in in cutlass.range_constexpr(D):
                    final_state_g[state_base_f + cutlass.Int32(k_in * D) + cutlass.Int32(tidx)] = cutlass.Float32(
                        sState[k_in, tidx]
                    )
            else:
                for d_out in cutlass.range_constexpr(D):
                    final_state_g[state_base_f + cutlass.Int32(d_out * D) + cutlass.Int32(tidx)] = cutlass.Float32(
                        sState[tidx, d_out]
                    )


@cute.jit
def run_k2(
    v: cute.Tensor,
    beta: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    ws_mqk: cute.Tensor,
    out: cute.Tensor,
    cu_seqlens_tiles: cute.Tensor,
    seq_order: cute.Tensor,
    initial_state_g: cute.Tensor,  # flat fp32 [N*H*D*D] or dummy [1]
    final_state_g: cute.Tensor,  # flat fp32 [N*H*D*D] or dummy [1]
    v_tile_starts: cute.Tensor,
    v_tile_actual_lens: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Int32,
    O_T_total: cutlass.Int32,
    V_T_total: cutlass.Int32,
    N: cutlass.Int32,
    has_initial_state: cutlass.Constexpr[bool],
    has_final_state: cutlass.Constexpr[bool],
    state_transposed: cutlass.Constexpr[bool],
    v_is_varlen: cutlass.Constexpr[bool],
    stream: cuda_drv.CUstream,
):
    cc_smem = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))
    kinter_smem = _make_out_kinter_one_stage()

    def make_thd_atom(t, op, t_total: cutlass.Int32):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((t_total, D, H), stride=(H * D, 1, D)),
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

    tma_atom_v, tma_tensor_v = make_thd_atom(v, cpasync.CopyBulkTensorTileG2SOp(), V_T_total)
    tma_atom_out, tma_tensor_out = make_thd_atom(out, cpasync.CopyBulkTensorTileS2GOp(), O_T_total)
    tma_atom_kd, tma_tensor_kd = make_ws_qkd_atom(ws_kd)
    tma_atom_qd, tma_tensor_qd = make_ws_qkd_atom(ws_qd)
    tma_atom_kr, tma_tensor_kr = make_ws_qkd_atom(ws_kr)
    tma_atom_inv, tma_tensor_inv = make_ws_cc_atom(ws_inv)
    tma_atom_mqk, tma_tensor_mqk = make_ws_cc_atom(ws_mqk)

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
    OUT_STAGES_LOCAL = 2
    smem_bytes = (
        D * D * 2
        + STAGES_LOCAL * 4 * (CHUNK * D * 2)
        + STAGES_LOCAL * 2 * (CHUNK * CHUNK * 2)
        + OUT_STAGES_LOCAL * (CHUNK * D * 2)
        + STAGES_LOCAL * (D * 4)
        + STAGES_LOCAL * (64 * 2)
        + (STAGES_LOCAL + OUT_STAGES_LOCAL) * 2 * 8
        + 2048
    )

    k2_kernel(
        tma_atom_v,
        tma_tensor_v,
        tma_atom_kd,
        tma_tensor_kd,
        tma_atom_qd,
        tma_tensor_qd,
        tma_atom_kr,
        tma_tensor_kr,
        tma_atom_inv,
        tma_tensor_inv,
        tma_atom_mqk,
        tma_tensor_mqk,
        tma_atom_out,
        tma_tensor_out,
        out,
        tma_atom_gt,
        tma_tensor_gt,
        tma_atom_beta,
        tma_tensor_beta,
        H,
        total_tiles,
        cu_seqlens_tiles,
        seq_order,
        v_tile_starts,
        v_tile_actual_lens,
        initial_state_g,
        final_state_g,
        has_initial_state,
        has_final_state,
        state_transposed,
        v_is_varlen,
    ).launch(
        grid=(N, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# Compile cache keyed on CONFIG ONLY — the per-batch shape dims
# (total_tiles, O_T_total, V_T_total, N) are dynamic cutlass.Int32, so one
# compiled kernel serves every batch shape.
_k2_kernel_cache: dict = {}
_DUMMY_FP32_CACHE: dict[tuple, torch.Tensor] = {}
_DUMMY_INT32_CACHE: dict[tuple, torch.Tensor] = {}
_CU_STREAM_CACHE: dict[int, object] = {}
_CU_STREAM_CACHE_MAXSIZE = 64
_DUMMY_CACHE_MAXSIZE = 64
_FIXED_CU_TILES_CACHE: dict[tuple, torch.Tensor] = {}
_FIXED_CU_TILES_CACHE_MAXSIZE = 64
_IDENTITY_ORDER_CACHE: dict[tuple, torch.Tensor] = {}
_IDENTITY_ORDER_CACHE_MAXSIZE = 64


def _compile_k2(H, has_initial_state, has_final_state, state_transposed, v_is_varlen):
    # One sym_int per dynamic tensor extent; shape scalars are runtime Int32.
    sym_vt = cute.sym_int()  # V_T_total (v token extent)
    sym_ot = cute.sym_int()  # O_T_total (out token extent; differs from V in varlen)
    sym_qk = cute.sym_int()  # ws_qd/kd/kr  (total_tiles*H*CHUNK*D)
    sym_cc = cute.sym_int()  # ws_inv/mqk   (total_tiles*H*CHUNK*CHUNK)
    sym_gt = cute.sym_int()  # ws_gt        (total_tiles*H*D)
    sym_bt = cute.sym_int()  # beta/ws_beta (total_tiles*CHUNK*H)
    sym_cu = cute.sym_int()  # cu_seqlens_tiles (N+1)
    sym_so = cute.sym_int()  # seq_order (N) — own sym: length differs from cu_seqlens_tiles
    # init/final need separate syms: their runtime sizes differ when one is
    # the 1-element dummy, and tvm-ffi binds each sym to one value.
    sym_sti = cute.sym_int()  # initial state flat (N*H*D*D or 1)
    sym_stf = cute.sym_int()  # final state flat (N*H*D*D or 1)
    sym_vs = cute.sym_int()  # v_tile_starts (total_tiles or 1)
    sym_va = cute.sym_int()  # v_tile_actual_lens (total_tiles or 1)

    v_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_vt, H, D), stride_order=(2, 1, 0), assumed_align=16)
    beta_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_bt,), assumed_align=16)
    ws_qd_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_kd_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_kr_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_gt_fake = make_fake_compact_tensor(cutlass.Float32, (sym_gt,), assumed_align=16)
    ws_inv_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_cc,), assumed_align=16)
    ws_mqk_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_cc,), assumed_align=16)
    out_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_ot, H, D), stride_order=(2, 1, 0), assumed_align=16)
    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=4)
    so_fake = make_fake_compact_tensor(cutlass.Int32, (sym_so,), assumed_align=4)
    init_fake = make_fake_compact_tensor(cutlass.Float32, (sym_sti,), assumed_align=16)
    final_fake = make_fake_compact_tensor(cutlass.Float32, (sym_stf,), assumed_align=16)
    vts_fake = make_fake_compact_tensor(cutlass.Int32, (sym_vs,), assumed_align=4)
    vtal_fake = make_fake_compact_tensor(cutlass.Int32, (sym_va,), assumed_align=4)
    stream_fake = make_fake_stream()

    return cute.compile(
        run_k2,
        v_fake,
        beta_fake,
        ws_qd_fake,
        ws_kd_fake,
        ws_kr_fake,
        ws_gt_fake,
        ws_inv_fake,
        ws_mqk_fake,
        out_fake,
        cu_fake,
        so_fake,
        init_fake,
        final_fake,
        vts_fake,
        vtal_fake,
        H,  # Constexpr -> baked
        cutlass.Int32(1),  # total_tiles -> runtime (placeholder)
        cutlass.Int32(1),  # O_T_total
        cutlass.Int32(1),  # V_T_total
        cutlass.Int32(1),  # N
        has_initial_state,  # Constexpr
        has_final_state,  # Constexpr
        state_transposed,  # Constexpr
        v_is_varlen,  # Constexpr
        stream_fake,
        options="--enable-tvm-ffi",
    )


def _get_compiled_k2(H, has_initial_state, has_final_state, state_transposed, v_is_varlen):
    key = (H, has_initial_state, has_final_state, state_transposed, v_is_varlen)
    cached = _k2_kernel_cache.get(key)
    if cached is None:
        cached = _compile_k2(H, has_initial_state, has_final_state, state_transposed, v_is_varlen)
        _k2_kernel_cache[key] = cached
    return cached


def _get_current_custream():
    stream_ptr = int(torch.cuda.current_stream().cuda_stream)
    cached = _CU_STREAM_CACHE.get(stream_ptr)
    if cached is not None:
        return cached
    if len(_CU_STREAM_CACHE) >= _CU_STREAM_CACHE_MAXSIZE:
        _CU_STREAM_CACHE.pop(next(iter(_CU_STREAM_CACHE)))
    cached = cuda_drv.CUstream(stream_ptr)
    _CU_STREAM_CACHE[stream_ptr] = cached
    return cached


def _get_dummy_fp32(device: torch.device) -> torch.Tensor:
    key = _stream_key(device)
    cached = _DUMMY_FP32_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_DUMMY_FP32_CACHE) >= _DUMMY_CACHE_MAXSIZE:
        _DUMMY_FP32_CACHE.pop(next(iter(_DUMMY_FP32_CACHE)))
    cached = torch.zeros(1, dtype=torch.float32, device=device)
    _DUMMY_FP32_CACHE[key] = cached
    return cached


def _get_dummy_int32(device: torch.device) -> torch.Tensor:
    key = _stream_key(device)
    cached = _DUMMY_INT32_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_DUMMY_INT32_CACHE) >= _DUMMY_CACHE_MAXSIZE:
        _DUMMY_INT32_CACHE.pop(next(iter(_DUMMY_INT32_CACHE)))
    cached = torch.zeros(1, dtype=torch.int32, device=device)
    _DUMMY_INT32_CACHE[key] = cached
    return cached


def _get_identity_order(n: int, device: torch.device) -> torch.Tensor:
    """Cached identity launch order [0..n) for serial/uniform batches."""
    key = (n, _stream_key(device))
    cached = _IDENTITY_ORDER_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_IDENTITY_ORDER_CACHE) >= _IDENTITY_ORDER_CACHE_MAXSIZE:
        _IDENTITY_ORDER_CACHE.pop(next(iter(_IDENTITY_ORDER_CACHE)))
    cached = torch.arange(n, dtype=torch.int32, device=device)
    _IDENTITY_ORDER_CACHE[key] = cached
    return cached


def _get_fixed_cu_seqlens_tiles(B: int, t_tiles_per_seq: int, device: torch.device) -> torch.Tensor:
    """Cached [0, t, 2t, ..., B*t] tile offsets for the fixed-length path."""
    key = (B, t_tiles_per_seq, _stream_key(device))
    cached = _FIXED_CU_TILES_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_FIXED_CU_TILES_CACHE) >= _FIXED_CU_TILES_CACHE_MAXSIZE:
        _FIXED_CU_TILES_CACHE.pop(next(iter(_FIXED_CU_TILES_CACHE)))
    cached = torch.arange(
        0,
        (B + 1) * t_tiles_per_seq,
        t_tiles_per_seq,
        dtype=torch.int32,
        device=device,
    )
    _FIXED_CU_TILES_CACHE[key] = cached
    return cached


def launch_k2(
    v: torch.Tensor,
    beta: torch.Tensor,
    ws_qd: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_kr: torch.Tensor,
    ws_gt: torch.Tensor,
    ws_inv: torch.Tensor,
    ws_mqk: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_tiles: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,  # [N, H, V, K] fp32 bhvk
    final_state: torch.Tensor | None = None,  # [N, H, V, K] fp32 bhvk (written in-place)
    state_transposed: bool = False,
    v_tile_starts: torch.Tensor | None = None,
    v_tile_actual_lens: torch.Tensor | None = None,
    seq_order: torch.Tensor | None = None,
) -> None:
    """Run K2 recurrence. Supports fixed-len and varlen inputs.

    state_transposed: False=[N,H,V,K] (default), True=[N,H,K,V].
    seq_order: optional int32 [N] launch-slot -> sequence index permutation
        (longest-first for ragged CP batches); None = identity.
    """
    B, T, H, _ = v.shape
    V_T_total = B * T
    O_T_total = out.shape[0] * out.shape[1]
    v_is_varlen = v_tile_starts is not None
    if v_is_varlen:
        total_tiles = v_tile_starts.numel()
    else:
        O_T_total = V_T_total
        total_tiles = V_T_total // CHUNK
        dummy = _get_dummy_int32(v.device)
        v_tile_starts = dummy
        v_tile_actual_lens = dummy

    if cu_seqlens_tiles is None:
        cu_seqlens_tiles = _get_fixed_cu_seqlens_tiles(B, T // CHUNK, v.device)
        N_seqs = B
    else:
        N_seqs = cu_seqlens_tiles.numel() - 1

    has_initial_state_flag = initial_state is not None
    has_final_state_flag = final_state is not None

    _dummy = _get_dummy_fp32(v.device)
    if has_initial_state_flag:
        initial_state_fp32 = initial_state.reshape(-1)
    else:
        initial_state_fp32 = _dummy
    if has_final_state_flag:
        final_state_fp32 = final_state.reshape(-1)
    else:
        final_state_fp32 = _dummy

    if seq_order is None:
        seq_order = _get_identity_order(N_seqs, v.device)

    compiled_fn = _get_compiled_k2(
        H,
        has_initial_state_flag,
        has_final_state_flag,
        state_transposed,
        v_is_varlen,
    )
    stream = _get_current_custream()
    # tvm-ffi launch: torch tensors pass straight through with NO shape/dtype
    # validation of the positional args. TMA descriptors are (re)built inside
    # run_k2 from the dynamic Int32 dims every launch, so one compiled kernel
    # serves all shapes.
    compiled_fn(
        v.view(V_T_total, H, D),
        beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        out.view(O_T_total, H, D),
        cu_seqlens_tiles,
        seq_order,
        initial_state_fp32,
        final_state_fp32,
        v_tile_starts,
        v_tile_actual_lens,
        total_tiles,
        O_T_total,
        V_T_total,
        N_seqs,
        stream,
    )
