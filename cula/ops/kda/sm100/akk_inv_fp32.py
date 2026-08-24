# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Standalone fp32-workspace Akk inverse for K123.

Input:
  A_phys [B, T, H, 64] fp32, K123 physical block-transposed pre-inverse layout.
Output:
  A_out  [B, T, H, 64] bf16, logical lower_tri((I + L)^-1), no beta[col] epilogue.

This is a baseline kernel for K123 -> fp32 GMEM workspace -> standalone inverse.
It converts K123's physical layout to logical lower-triangular fp32 in SMEM,
then reuses the same TF32 Schur helper structure as kda_akk_inv_tf32.py.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from cula.ops.kda.sm100.akk_inv_tf32 import (
    _add_store_C16_smem,
    _invert_diag_forward16,
    _matmul16_smem_smem,
    _matmul16_tmp_smem,
    _store_C16_smem,
    _store_C16_tmp,
)

BS = 64
SB = 16
THREADS = 128
AKK_STRIDE = BS + 4
TMP_STRIDE = SB + 4
TMP_SLOTS = 4


@cute.kernel
def akk_inv_fp32_physical_kernel(
    mA_phys: cute.Tensor,
    mA_out: cute.Tensor,
    smat_layout: cute.Layout,
    stmp_layout: cute.Layout,
    NT: int,
    H: int,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    WAIT_ON_PDL: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_id = tidx % 32
    h_idx, nt_idx, b_idx = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    sAkk = smem.allocate_tensor(cutlass.Float32, smat_layout, 128)
    sTmp = smem.allocate_tensor(cutlass.Float32, stmp_layout, 128)

    chunk_start = nt_idx * BS
    eos = cutlass.Int32(chunk_start + BS)
    if IS_VARLEN:
        seq_id = cutlass.Int32(mChunkIndices[nt_idx, 0])
        local = cutlass.Int32(mChunkIndices[nt_idx, 1])
        bos = cutlass.Int32(mCuSeqlens[seq_id])
        eos = cutlass.Int32(mCuSeqlens[seq_id + 1])
        chunk_start = bos + local * BS

    if cutlass.const_expr(WAIT_ON_PDL != 0):
        # The setup above is independent. Wait only before reading K123's fp32
        # workspace.
        cute.arch.griddepcontrol_wait()

    for i in range((BS * BS) // THREADS):
        linear = tidx + i * THREADS
        row = linear // BS
        col = linear % BS
        t_row = chunk_start + row
        value = cutlass.Float32(0.0)
        if row >= col:
            if row == col:
                value = cutlass.Float32(1.0)
            else:
                row_blk = row // SB
                col_blk = col // SB
                src_row = row
                src_col = col
                if row_blk != col_blk:
                    src_row = col_blk * SB + row % SB
                    src_col = row_blk * SB + col % SB
                if IS_VARLEN:
                    if t_row < eos:
                        value = cutlass.Float32(mA_phys[b_idx, chunk_start + src_row, h_idx, src_col])
                else:
                    value = cutlass.Float32(mA_phys[b_idx, chunk_start + src_row, h_idx, src_col])
        sAkk[row, col] = value
    cute.arch.barrier()

    if tidx < 64:
        _invert_diag_forward16(sAkk, tidx // 16, tidx)
    cute.arch.barrier()

    if tidx < 64:
        block32 = tidx // 32
        row_base = block32 * 32
        c0, c1, c2, c3, c4, c5, c6, c7 = _matmul16_smem_smem(
            sAkk, row_base + 16, row_base + 16, sAkk, row_base + 16, row_base, lane_id
        )
        _store_C16_tmp(sTmp, block32, -c0, -c1, -c2, -c3, -c4, -c5, -c6, -c7, lane_id)
    cute.arch.barrier()

    if tidx < 64:
        block32 = tidx // 32
        row_base = block32 * 32
        c0, c1, c2, c3, c4, c5, c6, c7 = _matmul16_tmp_smem(sTmp, block32, sAkk, row_base, row_base, lane_id)
        _store_C16_smem(sAkk, row_base + 16, row_base, c0, c1, c2, c3, c4, c5, c6, c7, lane_id)
    cute.arch.barrier()

    x = warp_idx // 2
    y = warp_idx % 2
    slot = warp_idx
    row_o = 32 + y * 16
    col_c = x * 16

    p0, p1, p2, p3, p4, p5, p6, p7 = _matmul16_smem_smem(sAkk, row_o, 32, sAkk, 32, col_c, lane_id)
    q0, q1, q2, q3, q4, q5, q6, q7 = _matmul16_smem_smem(sAkk, row_o, 48, sAkk, 48, col_c, lane_id)
    _store_C16_tmp(
        sTmp,
        slot,
        -(p0 + q0),
        -(p1 + q1),
        -(p2 + q2),
        -(p3 + q3),
        -(p4 + q4),
        -(p5 + q5),
        -(p6 + q6),
        -(p7 + q7),
        lane_id,
    )
    cute.arch.barrier()

    o0, o1, o2, o3, o4, o5, o6, o7 = _matmul16_tmp_smem(sTmp, slot, sAkk, x * 16, 0, lane_id)
    r0, r1, r2, r3, r4, r5, r6, r7 = _matmul16_tmp_smem(sTmp, slot, sAkk, x * 16, 16, lane_id)
    if x == 0:
        _store_C16_smem(sAkk, row_o, 0, o0, o1, o2, o3, o4, o5, o6, o7, lane_id)
        _store_C16_smem(sAkk, row_o, 16, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
    cute.arch.barrier()
    if x == 1:
        _add_store_C16_smem(sAkk, row_o, 0, o0, o1, o2, o3, o4, o5, o6, o7, lane_id)
        _add_store_C16_smem(sAkk, row_o, 16, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
    cute.arch.barrier()

    row_start = warp_idx * SB
    for ri in range(SB):
        row = row_start + ri
        col0 = lane_id * 2
        col1 = col0 + 1
        t_row = chunk_start + row
        val0 = cutlass.Float32(sAkk[row, col0])
        val1 = cutlass.Float32(sAkk[row, col1])
        if row < col0:
            val0 = cutlass.Float32(0.0)
        if row < col1:
            val1 = cutlass.Float32(0.0)
        if IS_VARLEN:
            if t_row < eos:
                mA_out[b_idx, t_row, h_idx, col0] = val0.to(cutlass.BFloat16)
                mA_out[b_idx, t_row, h_idx, col1] = val1.to(cutlass.BFloat16)
        else:
            mA_out[b_idx, t_row, h_idx, col0] = val0.to(cutlass.BFloat16)
            mA_out[b_idx, t_row, h_idx, col1] = val1.to(cutlass.BFloat16)

    if cutlass.const_expr(WAIT_ON_PDL != 0):
        cute.arch.barrier()
        cute.arch.fence_acq_rel_gpu()
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def akk_inv_fp32_physical_host(
    A_phys: cute.Tensor,
    A_out: cute.Tensor,
    B: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    T_VAL: cutlass.Constexpr[int],
    WAIT_ON_PDL: cutlass.Constexpr[int] = 1,
):
    in_layout = cute.make_layout((B, T_VAL, H, BS), stride=(T_VAL * H * BS, H * BS, BS, 1))
    out_layout = cute.make_layout((B, T_VAL, H, BS), stride=(T_VAL * H * BS, H * BS, BS, 1))
    gA_phys = cute.make_tensor(A_phys.iterator, in_layout)
    gA_out = cute.make_tensor(A_out.iterator, out_layout)

    smat_layout = cute.make_layout((BS, BS), stride=(AKK_STRIDE, 1))
    stmp_layout = cute.make_layout((TMP_SLOTS, SB, SB), stride=(SB * TMP_STRIDE, TMP_STRIDE, 1))
    smem_bytes = BS * AKK_STRIDE * 4 + TMP_SLOTS * SB * TMP_STRIDE * 4 + 256

    akk_inv_fp32_physical_kernel(
        gA_phys,
        gA_out,
        smat_layout,
        stmp_layout,
        NT,
        H,
        mCuSeqlens,
        mChunkIndices,
        IS_VARLEN,
        WAIT_ON_PDL,
    ).launch(
        grid=(H, NT, B),
        block=(THREADS, 1, 1),
        smem=smem_bytes,
        use_pdl=WAIT_ON_PDL != 0,
    )
