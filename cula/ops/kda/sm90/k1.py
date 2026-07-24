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
CuteDSL port of FlashKDA K1.

Produces 6 workspace tensors consumed by K2: ws_qd, ws_kd, ws_kr, ws_gt, ws_inv, ws_mqk.
"""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.nvgpu.warpgroup import SmemLayoutAtomKind, make_smem_layout_atom
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cula.ops.kda.sm90._common import _stream_key, add_f16x2_u32, movm_t_b16

CHUNK: int = 16
D: int = 128
THREADS_PER_CTA: int = 256


@cute.kernel
def k1_kernel(
    tma_atom_q: cute.CopyAtom,
    tma_tensor_q: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_tensor_k: cute.Tensor,
    tma_atom_g: cute.CopyAtom,
    tma_tensor_g: cute.Tensor,
    tma_atom_ws_qd: cute.CopyAtom,
    tma_tensor_ws_qd: cute.Tensor,
    tma_atom_ws_kd: cute.CopyAtom,
    tma_tensor_ws_kd: cute.Tensor,
    tma_atom_ws_kr: cute.CopyAtom,
    tma_tensor_ws_kr: cute.Tensor,
    tma_atom_ws_inv: cute.CopyAtom,
    tma_tensor_ws_inv: cute.Tensor,
    tma_atom_ws_mqk: cute.CopyAtom,
    tma_tensor_ws_mqk: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    ws_mqk: cute.Tensor,
    ws_beta: cute.Tensor,
    tile_starts: cute.Tensor,
    tile_actual_lens: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Int32,
    T_total: cutlass.Int32,
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
    is_varlen: cutlass.Constexpr[bool],
):
    tile_idx, head_idx, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((CHUNK, D), stride=(D, 1))
    kinter_atom_qk = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    kinter_qk_layout = cute.tile_to_shape(kinter_atom_qk, (CHUNK, D), order=(0, 1))
    cc_layout = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))

    sQ = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sK = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sG_raw = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    sG_cumsum = smem.allocate_tensor(cutlass.Float32, qk_layout, 128)
    s_g_total = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 128)
    # Outputs (K_INTER swizzled, alias input bytes after barrier)
    s_k_inv = cute.make_tensor(cute.recast_ptr(sG_cumsum.iterator, dtype=cutlass.BFloat16), kinter_qk_layout)
    s_q_decayed = cute.make_tensor(sQ.iterator, kinter_qk_layout)
    s_k_decayed = cute.make_tensor(sK.iterator, kinter_qk_layout)
    s_k_restored = cute.make_tensor(sG_raw.iterator, kinter_qk_layout)
    sL_bf16 = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sMqk_bf16 = smem.allocate_tensor(cutlass.BFloat16, cc_layout, 128)
    sBetaSig = smem.allocate_tensor(cutlass.Float32, cute.make_layout((CHUNK,)), 128)
    # Neumann buffers (alias sG_cumsum after decay_apply barrier)
    sG_cumsum_fp16_ptr = cute.recast_ptr(sG_cumsum.iterator, dtype=cutlass.Float16)
    sG_cumsum_bf16_ptr = cute.recast_ptr(sG_cumsum.iterator, dtype=cutlass.BFloat16)
    sL_fp16 = cute.make_tensor(sG_cumsum_fp16_ptr, cc_layout)
    sINV_fp16 = cute.make_tensor(sG_cumsum_fp16_ptr + (CHUNK * CHUNK), cc_layout)
    sINV_bf16 = cute.make_tensor(sG_cumsum_bf16_ptr + (2 * CHUNK * CHUNK), cc_layout)
    sMbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), 8)
    sMbar_ptr = sMbar.iterator

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(sMbar_ptr, cutlass.Int32(1))
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    token_start = tile_idx * CHUNK
    actual_len = cutlass.Int32(CHUNK)
    load_tile_idx = tile_idx
    tma_tensor_q_use = tma_tensor_q
    tma_tensor_k_use = tma_tensor_k
    tma_tensor_g_use = tma_tensor_g
    if cutlass.const_expr(is_varlen):
        token_start = tile_starts[tile_idx]
        actual_len = tile_actual_lens[tile_idx]
        load_tile_idx = cutlass.Int32(0)
        tma_tensor_q_use = cute.domain_offset((token_start, 0, 0), tma_tensor_q)
        tma_tensor_k_use = cute.domain_offset((token_start, 0, 0), tma_tensor_k)
        tma_tensor_g_use = cute.domain_offset((token_start, 0, 0), tma_tensor_g)

    gSrc_q = cute.local_tile(tma_tensor_q_use, (CHUNK, D), (None, None, None))
    gSrc_k = cute.local_tile(tma_tensor_k_use, (CHUNK, D), (None, None, None))
    gSrc_g = cute.local_tile(tma_tensor_g_use, (CHUNK, D), (None, None, None))
    tQs, tQg = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 2),
        cute.group_modes(gSrc_q, 0, 2),
    )
    tKs, tKg = cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        cute.group_modes(sK, 0, 2),
        cute.group_modes(gSrc_k, 0, 2),
    )
    tGs, tGg = cpasync.tma_partition(
        tma_atom_g,
        0,
        cute.make_layout(1),
        cute.group_modes(sG_raw, 0, 2),
        cute.group_modes(gSrc_g, 0, 2),
    )

    gDst_qd = cute.local_tile(tma_tensor_ws_qd, (CHUNK, D), (None, None, None))
    tQDws_s, tQDws_g = cpasync.tma_partition(
        tma_atom_ws_qd,
        0,
        cute.make_layout(1),
        cute.group_modes(s_q_decayed, 0, 2),
        cute.group_modes(gDst_qd, 0, 2),
    )
    gDst_kd = cute.local_tile(tma_tensor_ws_kd, (CHUNK, D), (None, None, None))
    tKDws_s, tKDws_g = cpasync.tma_partition(
        tma_atom_ws_kd,
        0,
        cute.make_layout(1),
        cute.group_modes(s_k_decayed, 0, 2),
        cute.group_modes(gDst_kd, 0, 2),
    )
    gDst_kr = cute.local_tile(tma_tensor_ws_kr, (CHUNK, D), (None, None, None))
    tKRws_s, tKRws_g = cpasync.tma_partition(
        tma_atom_ws_kr,
        0,
        cute.make_layout(1),
        cute.group_modes(s_k_restored, 0, 2),
        cute.group_modes(gDst_kr, 0, 2),
    )
    gDst_inv = cute.local_tile(tma_tensor_ws_inv, (CHUNK, CHUNK), (None, None, None))
    tINVws_s, tINVws_g = cpasync.tma_partition(
        tma_atom_ws_inv,
        0,
        cute.make_layout(1),
        cute.group_modes(sINV_bf16, 0, 2),
        cute.group_modes(gDst_inv, 0, 2),
    )
    gDst_mqk = cute.local_tile(tma_tensor_ws_mqk, (CHUNK, CHUNK), (None, None, None))
    tMQKws_s, tMQKws_g = cpasync.tma_partition(
        tma_atom_ws_mqk,
        0,
        cute.make_layout(1),
        cute.group_modes(sMqk_bf16, 0, 2),
        cute.group_modes(gDst_mqk, 0, 2),
    )
    ws_slot = head_idx * total_tiles + tile_idx

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(sMbar_ptr, cutlass.Int32(3 * CHUNK * D * 2))
        cute.copy(tma_atom_q, tQg[(None, load_tile_idx, 0, head_idx)], tQs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_k, tKg[(None, load_tile_idx, 0, head_idx)], tKs[(None,)], tma_bar_ptr=sMbar_ptr)
        cute.copy(tma_atom_g, tGg[(None, load_tile_idx, 0, head_idx)], tGs[(None,)], tma_bar_ptr=sMbar_ptr)

    cute.arch.mbarrier_wait(sMbar_ptr, cutlass.Int32(0))

    if tidx < 128:
        col_tail = tidx
        for r in cutlass.range_constexpr(CHUNK):
            if actual_len <= cutlass.Int32(r):
                sQ[r, col_tail] = cutlass.BFloat16(0.0)
                sK[r, col_tail] = cutlass.BFloat16(0.0)
                sG_raw[r, col_tail] = cutlass.BFloat16(0.0)
    cute.arch.barrier()

    row = tidx // 16
    sQ_tile = cute.flat_divide(sQ, (1, 8))  # ((1,8), CHUNK, D//8)
    sK_tile = cute.flat_divide(sK, (1, 8))
    cb = tidx % 16
    sQ_my = sQ_tile[(None, None, row, cb)]
    sK_my = sK_tile[(None, None, row, cb)]
    r_q_bf = cute.make_rmem_tensor(cute.make_layout((1, 8)), cutlass.BFloat16)
    r_k_bf = cute.make_rmem_tensor(cute.make_layout((1, 8)), cutlass.BFloat16)
    cute.autovec_copy(sQ_my, r_q_bf)
    cute.autovec_copy(sK_my, r_k_bf)
    q_sq = cutlass.Float32(0.0)
    k_sq = cutlass.Float32(0.0)
    q_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    k_vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), cutlass.Float32)
    for j in cutlass.range_constexpr(8):
        qv = cutlass.Float32(r_q_bf[0, j])
        kv = cutlass.Float32(r_k_bf[0, j])
        q_vals[j] = qv
        k_vals[j] = kv
        q_sq = q_sq + qv * qv
        k_sq = k_sq + kv * kv
    q_sq = cute.arch.warp_reduction(q_sq, lambda a, b: a + b, threads_in_group=16)
    k_sq = cute.arch.warp_reduction(k_sq, lambda a, b: a + b, threads_in_group=16)
    q_inv = cute.rsqrt(q_sq + cutlass.Float32(1.0e-6), fastmath=True)
    k_inv = cute.rsqrt(k_sq + cutlass.Float32(1.0e-6), fastmath=True)
    for j in cutlass.range_constexpr(8):
        r_q_bf[0, j] = cutlass.BFloat16(q_vals[j] * q_inv)
        r_k_bf[0, j] = cutlass.BFloat16(k_vals[j] * k_inv)
    cute.autovec_copy(r_q_bf, sQ_my)
    cute.autovec_copy(r_k_bf, sK_my)
    a_log_exp = cute.exp(cutlass.Float32(a_log[head_idx]), fastmath=True)
    if tidx < 128:
        col_c = tidx
        dt = cutlass.Float32(dt_bias[head_idx, col_c])
        s = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(CHUNK):
            if actual_len > cutlass.Int32(r):
                x = cutlass.Float32(sG_raw[r, col_c]) + dt
                x = a_log_exp * x
                sig = cutlass.Float32(0.5) * (cute.tanh(x * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
                s = s + cutlass.Float32(gate_scale) * sig
            sG_cumsum[r, col_c] = s
        s_g_total[col_c] = cute.exp(s, fastmath=True)
    cute.arch.barrier()

    # Pre-compute per-row sigmoid(beta). beta is the original packed [T, H]
    # layout, so no host-side transpose is needed.
    if tidx < CHUNK:
        bv = cutlass.Float32(-80.0)
        if actual_len > tidx:
            bv = cutlass.Float32(beta[(token_start + tidx) * H + head_idx])
        sBetaSig[tidx] = cutlass.Float32(0.5) * (cute.tanh(bv * cutlass.Float32(0.5), fastmath=True) + cutlass.Float32(1.0))
        # Emit raw beta into the compact wt_l workspace (tail rows = -80 -> sigmoid~0).
        ws_beta[(head_idx * total_tiles + tile_idx) * CHUNK + tidx] = cutlass.BFloat16(bv)

    lane = tidx % 32
    warp_id = tidx // 32
    group = lane // 4
    t_in_group = lane % 4
    N_M: cutlass.Constexpr[int] = CHUNK // 8  # = 2
    N_N: cutlass.Constexpr[int] = D // 64  # = 2
    N_TILES: cutlass.Constexpr[int] = N_M * N_N  # = 4

    reg_g = cute.make_rmem_tensor(cute.make_layout((N_TILES, 2)), cutlass.Float32)
    reg_q = cute.make_rmem_tensor(cute.make_layout((N_TILES, 2)), cutlass.BFloat16)
    reg_k = cute.make_rmem_tensor(cute.make_layout((N_TILES, 2)), cutlass.BFloat16)
    reg_gt = cute.make_rmem_tensor(cute.make_layout((N_TILES, 2)), cutlass.Float32)
    sG_cumsum_zipped = cute.zipped_divide(sG_cumsum, (1, 2))
    sQ_zipped = cute.zipped_divide(sQ, (1, 2))
    sK_zipped = cute.zipped_divide(sK, (1, 2))
    s_g_total_zipped = cute.zipped_divide(s_g_total, (2,))
    for m_blk in cutlass.range_constexpr(0, CHUNK, 8):
        for n_blk in cutlass.range_constexpr(0, D, 64):
            tile_idx_d: cutlass.Constexpr[int] = (m_blk // 8) * N_N + (n_blk // 64)
            row = m_blk + ((warp_id + group) % 8)
            col = n_blk + group * 8 + t_in_group * 2
            cute.autovec_copy(sG_cumsum_zipped[None, (row, col // 2)], reg_g[tile_idx_d, None])
            cute.autovec_copy(sQ_zipped[None, (row, col // 2)], reg_q[tile_idx_d, None])
            cute.autovec_copy(sK_zipped[None, (row, col // 2)], reg_k[tile_idx_d, None])
            cute.autovec_copy(s_g_total_zipped[None, col // 2], reg_gt[tile_idx_d, None])

    cute.arch.barrier()

    r_qd_pack = cute.make_rmem_tensor(cute.make_layout((1, 2)), cutlass.BFloat16)
    r_kd_pack = cute.make_rmem_tensor(cute.make_layout((1, 2)), cutlass.BFloat16)
    r_ki_pack = cute.make_rmem_tensor(cute.make_layout((1, 2)), cutlass.BFloat16)
    r_kr_pack = cute.make_rmem_tensor(cute.make_layout((1, 2)), cutlass.BFloat16)
    s_q_decayed_zipped = cute.zipped_divide(s_q_decayed, (1, 2))
    s_k_decayed_zipped = cute.zipped_divide(s_k_decayed, (1, 2))
    s_k_inv_zipped = cute.zipped_divide(s_k_inv, (1, 2))
    s_k_restored_zipped = cute.zipped_divide(s_k_restored, (1, 2))
    for m_blk in cutlass.range_constexpr(0, CHUNK, 8):
        for n_blk in cutlass.range_constexpr(0, D, 64):
            tile_idx_d: cutlass.Constexpr[int] = (m_blk // 8) * N_N + (n_blk // 64)
            row = m_blk + ((warp_id + group) % 8)
            col = n_blk + group * 8 + t_in_group * 2
            for v in cutlass.range_constexpr(2):
                vv: cutlass.Constexpr[int] = v
                gv = reg_g[tile_idx_d, vv]
                qv = cutlass.Float32(reg_q[tile_idx_d, vv])
                kv = cutlass.Float32(reg_k[tile_idx_d, vv])
                gtv = reg_gt[tile_idx_d, vv]
                exp_pos = cute.exp(gv, fastmath=True)
                inv_pos = cutlass.Float32(1.0) / exp_pos
                r_qd_pack[0, vv] = cutlass.BFloat16(qv * exp_pos * cutlass.Float32(scale))
                r_kd_pack[0, vv] = cutlass.BFloat16(kv * exp_pos)
                r_ki_pack[0, vv] = cutlass.BFloat16(kv * inv_pos)
                r_kr_pack[0, vv] = cutlass.BFloat16(kv * gtv * inv_pos)
            cute.autovec_copy(r_qd_pack, s_q_decayed_zipped[None, (row, col // 2)])
            cute.autovec_copy(r_kd_pack, s_k_decayed_zipped[None, (row, col // 2)])
            cute.autovec_copy(r_ki_pack, s_k_inv_zipped[None, (row, col // 2)])
            cute.autovec_copy(r_kr_pack, s_k_restored_zipped[None, (row, col // 2)])

    if tidx < 128:
        gt_base = (head_idx * total_tiles + tile_idx) * D
        ws_gt[gt_base + tidx] = s_g_total[tidx]
    cute.arch.barrier()

    # All 5 TMA stores must come from one thread (cp.async.bulk groups are per-thread).

    mma_atom_mask_mma = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma_mask_mma = cute.make_tiled_mma(
        mma_atom_mask_mma,
        atom_layout_mnk=(1, 1, 1),
        permutation_mnk=(16, 16, 16),
    )
    warp_lane = tidx % 32
    thr_mma_mask_mma = tiled_mma_mask_mma.get_slice(warp_lane)

    copy_atom_AB_mask_mma = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16,
    )
    smem_tiled_copy_A_mask_mma = cute.make_tiled_copy_A(copy_atom_AB_mask_mma, tiled_mma_mask_mma)
    smem_tiled_copy_B_mask_mma = cute.make_tiled_copy_B(copy_atom_AB_mask_mma, tiled_mma_mask_mma)
    smem_thr_copy_A_mask_mma = smem_tiled_copy_A_mask_mma.get_slice(warp_lane)
    smem_thr_copy_B_mask_mma = smem_tiled_copy_B_mask_mma.get_slice(warp_lane)

    copy_atom_stsm_mask_mma = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    smem_tiled_store_mask_mma = cute.make_tiled_copy_C_atom(copy_atom_stsm_mask_mma, tiled_mma_mask_mma)
    smem_thr_store_mask_mma = smem_tiled_store_mask_mma.get_slice(warp_lane)

    sB_tile = cute.flat_divide(s_k_inv, (CHUNK, 16))  # ((16,16), 1, D//16)
    sB_ref = sB_tile[None, None, 0, 0]

    if warp_idx == 0:
        sA_tile_l = cute.flat_divide(s_k_decayed, (CHUNK, 16))
        sA_ref_l = sA_tile_l[None, None, 0, 0]
        tCrA_l = thr_mma_mask_mma.make_fragment_A(thr_mma_mask_mma.partition_A(sA_ref_l))
        tCrB_l = thr_mma_mask_mma.make_fragment_B(thr_mma_mask_mma.partition_B(sB_ref))
        tCrC_l = thr_mma_mask_mma.make_fragment_C(tiled_mma_mask_mma.partition_shape_C((CHUNK, CHUNK)))
        tCrA_l_cv = smem_thr_copy_A_mask_mma.retile(tCrA_l)
        tCrB_l_cv = smem_thr_copy_B_mask_mma.retile(tCrB_l)

        tCrC_l.fill(0.0)
        for k_blk in cutlass.range_constexpr(D // 16):
            sA_k = sA_tile_l[None, None, 0, k_blk]
            sB_k = sB_tile[None, None, 0, k_blk]
            cute.copy(smem_tiled_copy_A_mask_mma, smem_thr_copy_A_mask_mma.partition_S(sA_k), tCrA_l_cv)
            cute.copy(smem_tiled_copy_B_mask_mma, smem_thr_copy_B_mask_mma.partition_S(sB_k), tCrB_l_cv)
            cute.gemm(tiled_mma_mask_mma, tCrC_l, tCrA_l, tCrB_l, tCrC_l)

        # L mask fold: factor = float(m > n) * sigmoid(beta[m]).
        coord_Cl = cute.make_identity_tensor((CHUNK, CHUNK))
        tCcC_l = thr_mma_mask_mma.partition_C(coord_Cl)
        for ii in cutlass.range_constexpr(cute.size(tCrC_l)):
            crd = tCcC_l[ii]
            m = crd[0]
            n = crd[1]
            keep = cutlass.Float32(1.0) if m > n else cutlass.Float32(0.0)
            tCrC_l[ii] = tCrC_l[ii] * keep * sBetaSig[m]

        tCrC_l_bf16 = cute.make_fragment_like(tCrC_l, cutlass.BFloat16)
        for ii in cutlass.range_constexpr(cute.size(tCrC_l)):
            tCrC_l_bf16[ii] = cutlass.BFloat16(tCrC_l[ii])
        cute.copy(
            smem_tiled_store_mask_mma,
            smem_thr_store_mask_mma.retile(tCrC_l_bf16),
            smem_thr_store_mask_mma.partition_D(sL_bf16),
        )
    elif warp_idx == 1:
        sA_tile_m = cute.flat_divide(s_q_decayed, (CHUNK, 16))
        sA_ref_m = sA_tile_m[None, None, 0, 0]
        tCrA_m = thr_mma_mask_mma.make_fragment_A(thr_mma_mask_mma.partition_A(sA_ref_m))
        tCrB_m = thr_mma_mask_mma.make_fragment_B(thr_mma_mask_mma.partition_B(sB_ref))
        tCrC_m = thr_mma_mask_mma.make_fragment_C(tiled_mma_mask_mma.partition_shape_C((CHUNK, CHUNK)))
        tCrA_m_cv = smem_thr_copy_A_mask_mma.retile(tCrA_m)
        tCrB_m_cv = smem_thr_copy_B_mask_mma.retile(tCrB_m)

        tCrC_m.fill(0.0)
        for k_blk in cutlass.range_constexpr(D // 16):
            sA_k = sA_tile_m[None, None, 0, k_blk]
            sB_k = sB_tile[None, None, 0, k_blk]
            cute.copy(smem_tiled_copy_A_mask_mma, smem_thr_copy_A_mask_mma.partition_S(sA_k), tCrA_m_cv)
            cute.copy(smem_tiled_copy_B_mask_mma, smem_thr_copy_B_mask_mma.partition_S(sB_k), tCrB_m_cv)
            cute.gemm(tiled_mma_mask_mma, tCrC_m, tCrA_m, tCrB_m, tCrC_m)

        # Mqk mask fold: factor = float(m >= n).
        coord_Cm = cute.make_identity_tensor((CHUNK, CHUNK))
        tCcC_m = thr_mma_mask_mma.partition_C(coord_Cm)
        for ii in cutlass.range_constexpr(cute.size(tCrC_m)):
            crd = tCcC_m[ii]
            m = crd[0]
            n = crd[1]
            keep = cutlass.Float32(1.0) if m >= n else cutlass.Float32(0.0)
            tCrC_m[ii] = tCrC_m[ii] * keep

        tCrC_m_bf16 = cute.make_fragment_like(tCrC_m, cutlass.BFloat16)
        for ii in cutlass.range_constexpr(cute.size(tCrC_m)):
            tCrC_m_bf16[ii] = cutlass.BFloat16(tCrC_m[ii])
        cute.copy(
            smem_tiled_store_mask_mma,
            smem_thr_store_mask_mma.retile(tCrC_m_bf16),
            smem_thr_store_mask_mma.partition_D(sMqk_bf16),
        )
    cute.arch.barrier()

    i = tidx // CHUNK
    col = tidx % CHUNK
    l_bf = cutlass.Float32(sL_bf16[i, col])
    sL_fp16[i, col] = cutlass.Float16(l_bf)
    inv_init = cutlass.Float32(1.0 if i == col else 0.0) - l_bf
    sINV_fp16[i, col] = cutlass.Float16(inv_init)
    cute.arch.barrier()

    if warp_idx == 0:
        mma_atom_neumann = warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float16, (16, 8, 16))
        tiled_mma_neumann = cute.make_tiled_mma(
            mma_atom_neumann,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )
        thr_mma_neumann = tiled_mma_neumann.get_slice(tidx)

        copy_atom_A_neumann = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.Float16,
        )
        smem_tiled_copy_A_neumann = cute.make_tiled_copy_A(copy_atom_A_neumann, tiled_mma_neumann)
        smem_thr_copy_A_neumann = smem_tiled_copy_A_neumann.get_slice(tidx)

        tCrL = thr_mma_neumann.make_fragment_A(thr_mma_neumann.partition_A(sL_fp16))
        tCrL_cv = smem_thr_copy_A_neumann.retile(tCrL)
        cute.copy(smem_tiled_copy_A_neumann, smem_thr_copy_A_neumann.partition_S(sL_fp16), tCrL_cv)

        tCrInv = thr_mma_neumann.make_fragment_A(thr_mma_neumann.partition_A(sINV_fp16))
        tCrInv_cv = smem_thr_copy_A_neumann.retile(tCrInv)
        cute.copy(smem_tiled_copy_A_neumann, smem_thr_copy_A_neumann.partition_S(sINV_fp16), tCrInv_cv)

        tCrLpowB = thr_mma_neumann.make_fragment_B(thr_mma_neumann.partition_B(sL_fp16))
        tCrLpow = thr_mma_neumann.make_fragment_C(tiled_mma_neumann.partition_shape_C((CHUNK, CHUNK)))
        tCrDelta = thr_mma_neumann.make_fragment_C(tiled_mma_neumann.partition_shape_C((CHUNK, CHUNK)))
        tCrLpowA = thr_mma_neumann.make_fragment_A(thr_mma_neumann.partition_A(sL_fp16))

        tCrL_u32 = cute.recast_tensor(tCrL, dtype=cutlass.Int32)
        tCrInv_u32 = cute.recast_tensor(tCrInv, dtype=cutlass.Int32)
        tCrLpowB_u32 = cute.recast_tensor(tCrLpowB, dtype=cutlass.Int32)
        tCrLpow_u32 = cute.recast_tensor(tCrLpow, dtype=cutlass.Int32)
        tCrDelta_u32 = cute.recast_tensor(tCrDelta, dtype=cutlass.Int32)
        tCrLpowA_u32 = cute.recast_tensor(tCrLpowA, dtype=cutlass.Int32)

        N_REGS_U32: cutlass.Constexpr[int] = 4  # 8 fp16 / thread = 4 u32

        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrLpowB_u32[ii] = movm_t_b16(cutlass.Int32(tCrL_u32[ii]))
        tCrLpow.fill(0.0)
        cute.gemm(tiled_mma_neumann, tCrLpow, tCrL, tCrLpowB, tCrLpow)

        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrLpowB_u32[ii] = movm_t_b16(cutlass.Int32(tCrLpow_u32[ii]))
        tCrDelta.fill(0.0)
        cute.gemm(tiled_mma_neumann, tCrDelta, tCrInv, tCrLpowB, tCrDelta)
        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrInv_u32[ii] = add_f16x2_u32(cutlass.Int32(tCrInv_u32[ii]), cutlass.Int32(tCrDelta_u32[ii]))

        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrLpowA_u32[ii] = tCrLpow_u32[ii]
        tCrLpow.fill(0.0)
        cute.gemm(tiled_mma_neumann, tCrLpow, tCrLpowA, tCrLpowB, tCrLpow)

        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrLpowB_u32[ii] = movm_t_b16(cutlass.Int32(tCrLpow_u32[ii]))
        tCrDelta.fill(0.0)
        cute.gemm(tiled_mma_neumann, tCrDelta, tCrInv, tCrLpowB, tCrDelta)
        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrInv_u32[ii] = add_f16x2_u32(cutlass.Int32(tCrInv_u32[ii]), cutlass.Int32(tCrDelta_u32[ii]))

        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrLpowA_u32[ii] = tCrLpow_u32[ii]
        tCrLpow.fill(0.0)
        cute.gemm(tiled_mma_neumann, tCrLpow, tCrLpowA, tCrLpowB, tCrLpow)

        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrLpowB_u32[ii] = movm_t_b16(cutlass.Int32(tCrLpow_u32[ii]))
        tCrDelta.fill(0.0)
        cute.gemm(tiled_mma_neumann, tCrDelta, tCrInv, tCrLpowB, tCrDelta)
        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrInv_u32[ii] = add_f16x2_u32(cutlass.Int32(tCrInv_u32[ii]), cutlass.Int32(tCrDelta_u32[ii]))

        tCrInvC = thr_mma_neumann.make_fragment_C(tiled_mma_neumann.partition_shape_C((CHUNK, CHUNK)))
        tCrInvC_u32 = cute.recast_tensor(tCrInvC, dtype=cutlass.Int32)
        for ii in cutlass.range_constexpr(N_REGS_U32):
            tCrInvC_u32[ii] = tCrInv_u32[ii]
        tCrInvC_bf16 = cute.make_fragment_like(tCrInvC, cutlass.BFloat16)
        for ii in cutlass.range_constexpr(cute.size(tCrInvC)):
            tCrInvC_bf16[ii] = cutlass.BFloat16(cutlass.Float32(tCrInvC[ii]))

        copy_atom_stsm = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
            cutlass.BFloat16,
        )
        smem_tiled_store_C = cute.make_tiled_copy_C_atom(copy_atom_stsm, tiled_mma_neumann)
        smem_thr_store_C = smem_tiled_store_C.get_slice(tidx)
        cute.copy(
            smem_tiled_store_C,
            smem_thr_store_C.retile(tCrInvC_bf16),
            smem_thr_store_C.partition_D(sINV_bf16),
        )
    cute.arch.barrier()

    # TMA bulk store all 5 workspace tensors (one elect_one, one thread).
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.copy(tma_atom_ws_qd, tQDws_s[(None,)], tQDws_g[(None, 0, 0, ws_slot)])
            cute.copy(tma_atom_ws_kd, tKDws_s[(None,)], tKDws_g[(None, 0, 0, ws_slot)])
            cute.copy(tma_atom_ws_kr, tKRws_s[(None,)], tKRws_g[(None, 0, 0, ws_slot)])
            cute.copy(tma_atom_ws_inv, tINVws_s[(None,)], tINVws_g[(None, 0, 0, ws_slot)])
            cute.copy(tma_atom_ws_mqk, tMQKws_s[(None,)], tMQKws_g[(None, 0, 0, ws_slot)])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def run_k1(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_kr: cute.Tensor,
    ws_gt: cute.Tensor,
    ws_inv: cute.Tensor,
    ws_mqk: cute.Tensor,
    ws_beta: cute.Tensor,
    tile_starts: cute.Tensor,
    tile_actual_lens: cute.Tensor,
    H: cutlass.Constexpr[int],
    total_tiles: cutlass.Int32,
    T_total: cutlass.Int32,
    scale: cutlass.Constexpr[float],
    gate_scale: cutlass.Constexpr[float],
    is_varlen: cutlass.Constexpr[bool],
    stream: cuda_drv.CUstream,
):
    smem_layout_qk = cute.make_layout((CHUNK, D), stride=(D, 1))
    # K_INTER swizzled layout — must match kernel SMEM layout for TMA stores.
    kinter_atom = make_smem_layout_atom(SmemLayoutAtomKind.K_INTER, cutlass.BFloat16)
    smem_layout_qk_kinter = cute.tile_to_shape(kinter_atom, (CHUNK, D), order=(0, 1))

    def make_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((T_total, D, H), stride=(H * D, 1, D)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            view,
            smem_layout_qk,
            (CHUNK, D),
        )

    def make_ws_store_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout((CHUNK, D, total_tiles * H), stride=(D, 1, CHUNK * D)),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            view,
            smem_layout_qk_kinter,
            (CHUNK, D),
        )

    # (CHUNK, CHUNK) bf16 plain layout for ws_inv / ws_mqk TMA bulk store.
    smem_layout_cc = cute.make_layout((CHUNK, CHUNK), stride=(CHUNK, 1))

    def make_ws_cc_store_atom(t):
        view = cute.make_tensor(
            t.iterator,
            cute.make_layout(
                (CHUNK, CHUNK, total_tiles * H),
                stride=(CHUNK, 1, CHUNK * CHUNK),
            ),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            view,
            smem_layout_cc,
            (CHUNK, CHUNK),
        )

    tma_atom_q, tma_tensor_q = make_atom(q)
    tma_atom_k, tma_tensor_k = make_atom(k)
    tma_atom_g, tma_tensor_g = make_atom(g)
    tma_atom_ws_qd, tma_tensor_ws_qd = make_ws_store_atom(ws_qd)
    tma_atom_ws_kd, tma_tensor_ws_kd = make_ws_store_atom(ws_kd)
    tma_atom_ws_kr, tma_tensor_ws_kr = make_ws_store_atom(ws_kr)
    tma_atom_ws_inv, tma_tensor_ws_inv = make_ws_cc_store_atom(ws_inv)
    tma_atom_ws_mqk, tma_tensor_ws_mqk = make_ws_cc_store_atom(ws_mqk)

    smem_bytes = 24 * 1024

    k1_kernel(
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_g,
        tma_tensor_g,
        tma_atom_ws_qd,
        tma_tensor_ws_qd,
        tma_atom_ws_kd,
        tma_tensor_ws_kd,
        tma_atom_ws_kr,
        tma_tensor_ws_kr,
        tma_atom_ws_inv,
        tma_tensor_ws_inv,
        tma_atom_ws_mqk,
        tma_tensor_ws_mqk,
        a_log,
        dt_bias,
        beta,
        ws_gt,
        ws_inv,
        ws_mqk,
        ws_beta,
        tile_starts,
        tile_actual_lens,
        H,
        total_tiles,
        T_total,
        scale,
        gate_scale,
        is_varlen,
    ).launch(
        grid=(total_tiles, H, 1),
        block=[THREADS_PER_CTA, 1, 1],
        smem=smem_bytes,
        stream=stream,
        min_blocks_per_mp=8,
    )


# Compile cache keyed on CONFIG ONLY — total_tiles/T_total are dynamic
# cutlass.Int32, so one compiled kernel serves every batch shape.
_k1_kernel_cache: dict = {}
_CU_STREAM_CACHE: dict[int, object] = {}
_DUMMY_INT32_CACHE: dict[tuple, torch.Tensor] = {}
_K1_KERNEL_CACHE_MAXSIZE = 32
_CU_STREAM_CACHE_MAXSIZE = 64
_DUMMY_INT32_CACHE_MAXSIZE = 64


def _compile_k1(H, scale, gate_scale, is_varlen):
    sym_t = cute.sym_int()  # T_total (q/k/g token extent)
    sym_al = cute.sym_int()  # a_log
    sym_bt = cute.sym_int()  # beta, original packed [T_total, H] flattened (T_total*H)
    sym_qk = cute.sym_int()  # ws_qd/kd/kr  (total_tiles*H*CHUNK*D)
    sym_gt = cute.sym_int()  # ws_gt        (total_tiles*H*D)
    sym_cc = cute.sym_int()  # ws_inv/mqk   (total_tiles*H*CHUNK*CHUNK)
    sym_wb = cute.sym_int()  # ws_beta      (total_tiles*CHUNK*H)
    sym_ts = cute.sym_int()  # tile_starts
    sym_tal = cute.sym_int()  # tile_actual_lens

    q_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, H, D), stride_order=(2, 1, 0), assumed_align=16)
    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, H, D), stride_order=(2, 1, 0), assumed_align=16)
    g_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, H, D), stride_order=(2, 1, 0), assumed_align=16)
    alog_fake = make_fake_compact_tensor(cutlass.Float32, (sym_al,), assumed_align=16)
    dt_fake = make_fake_compact_tensor(cutlass.Float32, (H, D), stride_order=(1, 0), assumed_align=16)
    beta_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_bt,), assumed_align=16)
    ws_qd_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_kd_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_kr_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_qk,), assumed_align=16)
    ws_gt_fake = make_fake_compact_tensor(cutlass.Float32, (sym_gt,), assumed_align=16)
    ws_inv_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_cc,), assumed_align=16)
    ws_mqk_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_cc,), assumed_align=16)
    ws_beta_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_wb,), assumed_align=16)
    ts_fake = make_fake_compact_tensor(cutlass.Int32, (sym_ts,), assumed_align=4)
    tal_fake = make_fake_compact_tensor(cutlass.Int32, (sym_tal,), assumed_align=4)
    stream_fake = make_fake_stream()

    return cute.compile(
        run_k1,
        q_fake,
        k_fake,
        g_fake,
        alog_fake,
        dt_fake,
        beta_fake,
        ws_qd_fake,
        ws_kd_fake,
        ws_kr_fake,
        ws_gt_fake,
        ws_inv_fake,
        ws_mqk_fake,
        ws_beta_fake,
        ts_fake,
        tal_fake,
        H,  # Constexpr -> baked
        cutlass.Int32(1),  # total_tiles -> runtime (placeholder)
        cutlass.Int32(1),  # T_total -> runtime
        scale,  # Constexpr
        gate_scale,  # Constexpr
        is_varlen,  # Constexpr
        stream_fake,
        options="--enable-tvm-ffi --opt-level=3",
    )


def _get_compiled_k1(H, scale, gate_scale, is_varlen):
    key = (H, scale, gate_scale, is_varlen)
    cached = _k1_kernel_cache.get(key)
    if cached is None:
        if len(_k1_kernel_cache) >= _K1_KERNEL_CACHE_MAXSIZE:
            _k1_kernel_cache.pop(next(iter(_k1_kernel_cache)))
        cached = _compile_k1(H, scale, gate_scale, is_varlen)
        _k1_kernel_cache[key] = cached
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


def _get_dummy_int32(device: torch.device) -> torch.Tensor:
    key = _stream_key(device)
    cached = _DUMMY_INT32_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_DUMMY_INT32_CACHE) >= _DUMMY_INT32_CACHE_MAXSIZE:
        _DUMMY_INT32_CACHE.pop(next(iter(_DUMMY_INT32_CACHE)))
    cached = torch.zeros(1, dtype=torch.int32, device=device)
    _DUMMY_INT32_CACHE[key] = cached
    return cached


def launch_k1(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    gate_scale: float,
    ws_qd: torch.Tensor,
    ws_kd: torch.Tensor,
    ws_kr: torch.Tensor,
    ws_gt: torch.Tensor,
    ws_inv: torch.Tensor,
    ws_mqk: torch.Tensor,
    ws_beta: torch.Tensor,
    *,
    tile_starts: torch.Tensor | None = None,
    tile_actual_lens: torch.Tensor | None = None,
    total_tiles: int | None = None,
    is_varlen: bool = False,
) -> None:
    """Run K1 pipeline; produces all 6 K2-ready workspace tensors."""
    B, T, H, _ = q.shape
    T_total = B * T
    if not is_varlen:
        total_tiles = (B * T) // CHUNK
        dummy = _get_dummy_int32(q.device)
        tile_starts = dummy
        tile_actual_lens = dummy

    compiled_fn = _get_compiled_k1(H, scale, gate_scale, is_varlen)
    stream = _get_current_custream()
    # TMA descriptors are (re)built inside run_k1 from the dynamic Int32 dims
    # every launch, so one compiled kernel serves all shapes.
    compiled_fn(
        q.view(T_total, H, D),
        k.view(T_total, H, D),
        g.view(T_total, H, D),
        A_log,
        dt_bias,
        beta,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        ws_inv,
        ws_mqk,
        ws_beta,
        tile_starts,
        tile_actual_lens,
        total_tiles,
        T_total,
        stream,
    )
