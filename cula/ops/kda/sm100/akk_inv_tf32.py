# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Akk 64x64 lower triangular inverse using the CUDA C TF32 Schur structure.

This is an experimental CuTeDSL port of
`kerutils::CollectiveInverseTF32<cutlass::tfloat32_t, true, false, false>`.

Input:
  A_kk [B, T, H, 64] bf16, logical lower-triangular form, diag may be garbage.
Output:
  A_kk [B, T, H, 64] bf16, lower_tri((I + L)^-1), no beta epilogue by default.

Precision choices:
  - SMEM is fp32 with the CUDA C padded stride 68.
  - Diagonal 16x16 blocks use forward substitution in fp32.
  - Off-diagonal Schur products use mma.sync f32.tf32.tf32.f32.
  - Intermediate Schur products are stored/reloaded as fp32 scratch in this
    first CuTeDSL version; TF32 truncation is still done only by MMA hardware.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

BS = 64
SB = 16
THREADS = 128
AKK_STRIDE = BS + 4
TMP_STRIDE = SB + 4
TMP_SLOTS = 4


@dsl_user_op
def mma_tf32_m16n8k8(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    *,
    loc=None,
    ip=None,
):
    """Register-level TF32 MMA with fp32 accumulator, shape m16n8k8."""
    a0b = llvm.bitcast(T.i32(), a0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a1b = llvm.bitcast(T.i32(), a1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a2b = llvm.bitcast(T.i32(), a2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a3b = llvm.bitcast(T.i32(), a3.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b0b = llvm.bitcast(T.i32(), b0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b1b = llvm.bitcast(T.i32(), b1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0b,
            a1b,
            a2b,
            a3b,
            b0b,
            b1b,
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        """{
            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {$0, $1, $2, $3},
                {$4, $5, $6, $7},
                {$8, $9},
                {$10, $11, $12, $13};
        }""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


@dsl_user_op
def _invert_diag_forward16(sAkk: cute.Tensor, block_idx, tidx, *, loc=None, ip=None):
    """CUDA C compute_diagonal_inverse_NxN<16>, in fp32."""
    tid_in_group = tidx % 16
    group_base = (block_idx % 2) * 16
    base = block_idx * 16
    row = cute.make_rmem_tensor(cute.make_layout((SB,), stride=(1,)), cutlass.Float32)

    for i in range(SB):
        val = cutlass.Float32(sAkk[base + tid_in_group, base + i])
        is_lower = cutlass.Float32(i < tid_in_group)
        is_diag = cutlass.Float32(i == tid_in_group)
        row[i] = val * is_lower + is_diag

    for src_row in range(SB - 1):
        row_scale = -row[src_row]
        target_lane = group_base + src_row
        active = cutlass.Float32(tid_in_group > src_row)
        for i in range(src_row):
            src_row_value = cute.arch.shuffle_sync_op(
                value=row[i],
                offset=target_lane,
                mask=0xFFFFFFFF,
                mask_and_clamp=31,
            )
            row[i] = row[i] + active * row_scale * src_row_value
        row[src_row] = active * row_scale + (cutlass.Float32(1.0) - active) * row[src_row]

    for i in range(SB):
        sAkk[base + tid_in_group, base + i] = row[i]


@dsl_user_op
def _matmul16_smem_smem(
    sA: cute.Tensor,
    a_row,
    a_col,
    sB: cute.Tensor,
    b_row,
    b_col,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    """Return a 16x16 tile C = A@B using two m16n8k8 TF32 MMAs."""
    gid = lane_id // 4
    tid = lane_id % 4
    z = cutlass.Float32(0.0)

    a0 = cutlass.Float32(sA[a_row + gid, a_col + 2 * tid])
    a1 = cutlass.Float32(sA[a_row + gid + 8, a_col + 2 * tid])
    a2 = cutlass.Float32(sA[a_row + gid, a_col + 2 * tid + 1])
    a3 = cutlass.Float32(sA[a_row + gid + 8, a_col + 2 * tid + 1])
    b0n0 = cutlass.Float32(sB[b_row + 2 * tid, b_col + gid])
    b1n0 = cutlass.Float32(sB[b_row + 2 * tid + 1, b_col + gid])
    b0n1 = cutlass.Float32(sB[b_row + 2 * tid, b_col + 8 + gid])
    b1n1 = cutlass.Float32(sB[b_row + 2 * tid + 1, b_col + 8 + gid])
    c0, c1, c2, c3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n0, b1n0, z, z, z, z)
    c4, c5, c6, c7 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n1, b1n1, z, z, z, z)

    a0 = cutlass.Float32(sA[a_row + gid, a_col + 8 + 2 * tid])
    a1 = cutlass.Float32(sA[a_row + gid + 8, a_col + 8 + 2 * tid])
    a2 = cutlass.Float32(sA[a_row + gid, a_col + 8 + 2 * tid + 1])
    a3 = cutlass.Float32(sA[a_row + gid + 8, a_col + 8 + 2 * tid + 1])
    b0n0 = cutlass.Float32(sB[b_row + 8 + 2 * tid, b_col + gid])
    b1n0 = cutlass.Float32(sB[b_row + 8 + 2 * tid + 1, b_col + gid])
    b0n1 = cutlass.Float32(sB[b_row + 8 + 2 * tid, b_col + 8 + gid])
    b1n1 = cutlass.Float32(sB[b_row + 8 + 2 * tid + 1, b_col + 8 + gid])
    c0, c1, c2, c3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n0, b1n0, c0, c1, c2, c3)
    c4, c5, c6, c7 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n1, b1n1, c4, c5, c6, c7)
    return c0, c1, c2, c3, c4, c5, c6, c7


@dsl_user_op
def _matmul16_tmp_smem(
    sTmp: cute.Tensor,
    slot,
    sB: cute.Tensor,
    b_row,
    b_col,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    z = cutlass.Float32(0.0)

    a0 = cutlass.Float32(sTmp[slot, gid, 2 * tid])
    a1 = cutlass.Float32(sTmp[slot, gid + 8, 2 * tid])
    a2 = cutlass.Float32(sTmp[slot, gid, 2 * tid + 1])
    a3 = cutlass.Float32(sTmp[slot, gid + 8, 2 * tid + 1])
    b0n0 = cutlass.Float32(sB[b_row + 2 * tid, b_col + gid])
    b1n0 = cutlass.Float32(sB[b_row + 2 * tid + 1, b_col + gid])
    b0n1 = cutlass.Float32(sB[b_row + 2 * tid, b_col + 8 + gid])
    b1n1 = cutlass.Float32(sB[b_row + 2 * tid + 1, b_col + 8 + gid])
    c0, c1, c2, c3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n0, b1n0, z, z, z, z)
    c4, c5, c6, c7 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n1, b1n1, z, z, z, z)

    a0 = cutlass.Float32(sTmp[slot, gid, 8 + 2 * tid])
    a1 = cutlass.Float32(sTmp[slot, gid + 8, 8 + 2 * tid])
    a2 = cutlass.Float32(sTmp[slot, gid, 8 + 2 * tid + 1])
    a3 = cutlass.Float32(sTmp[slot, gid + 8, 8 + 2 * tid + 1])
    b0n0 = cutlass.Float32(sB[b_row + 8 + 2 * tid, b_col + gid])
    b1n0 = cutlass.Float32(sB[b_row + 8 + 2 * tid + 1, b_col + gid])
    b0n1 = cutlass.Float32(sB[b_row + 8 + 2 * tid, b_col + 8 + gid])
    b1n1 = cutlass.Float32(sB[b_row + 8 + 2 * tid + 1, b_col + 8 + gid])
    c0, c1, c2, c3 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n0, b1n0, c0, c1, c2, c3)
    c4, c5, c6, c7 = mma_tf32_m16n8k8(a0, a1, a2, a3, b0n1, b1n1, c4, c5, c6, c7)
    return c0, c1, c2, c3, c4, c5, c6, c7


@dsl_user_op
def _store_C16_tmp(
    sTmp: cute.Tensor,
    slot,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    sTmp[slot, gid, 2 * tid] = c0
    sTmp[slot, gid, 2 * tid + 1] = c1
    sTmp[slot, gid + 8, 2 * tid] = c2
    sTmp[slot, gid + 8, 2 * tid + 1] = c3
    sTmp[slot, gid, 8 + 2 * tid] = c4
    sTmp[slot, gid, 8 + 2 * tid + 1] = c5
    sTmp[slot, gid + 8, 8 + 2 * tid] = c6
    sTmp[slot, gid + 8, 8 + 2 * tid + 1] = c7


@dsl_user_op
def _store_C16_smem(
    sDst: cute.Tensor,
    row,
    col,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    sDst[row + gid, col + 2 * tid] = c0
    sDst[row + gid, col + 2 * tid + 1] = c1
    sDst[row + gid + 8, col + 2 * tid] = c2
    sDst[row + gid + 8, col + 2 * tid + 1] = c3
    sDst[row + gid, col + 8 + 2 * tid] = c4
    sDst[row + gid, col + 8 + 2 * tid + 1] = c5
    sDst[row + gid + 8, col + 8 + 2 * tid] = c6
    sDst[row + gid + 8, col + 8 + 2 * tid + 1] = c7


@dsl_user_op
def _add_store_C16_smem(
    sDst: cute.Tensor,
    row,
    col,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    sDst[row + gid, col + 2 * tid] = cutlass.Float32(sDst[row + gid, col + 2 * tid]) + c0
    sDst[row + gid, col + 2 * tid + 1] = cutlass.Float32(sDst[row + gid, col + 2 * tid + 1]) + c1
    sDst[row + gid + 8, col + 2 * tid] = cutlass.Float32(sDst[row + gid + 8, col + 2 * tid]) + c2
    sDst[row + gid + 8, col + 2 * tid + 1] = cutlass.Float32(sDst[row + gid + 8, col + 2 * tid + 1]) + c3
    sDst[row + gid, col + 8 + 2 * tid] = cutlass.Float32(sDst[row + gid, col + 8 + 2 * tid]) + c4
    sDst[row + gid, col + 8 + 2 * tid + 1] = cutlass.Float32(sDst[row + gid, col + 8 + 2 * tid + 1]) + c5
    sDst[row + gid + 8, col + 8 + 2 * tid] = cutlass.Float32(sDst[row + gid + 8, col + 8 + 2 * tid]) + c6
    sDst[row + gid + 8, col + 8 + 2 * tid + 1] = cutlass.Float32(sDst[row + gid + 8, col + 8 + 2 * tid + 1]) + c7


@cute.kernel
def akk_inv_tf32_kernel(
    mA_in: cute.Tensor,
    mA_out: cute.Tensor,
    mBeta: cute.Tensor,
    smat_layout: cute.Layout,
    stmp_layout: cute.Layout,
    NT: int,
    H: int,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    APPLY_BETA_EPILOGUE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_id = tidx % 32
    h_idx, nt_idx, b_idx = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    sAkk = smem.allocate_tensor(cutlass.Float32, smat_layout, 128)
    sTmp = smem.allocate_tensor(cutlass.Float32, stmp_layout, 128)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BS,), stride=(1,)), 128)

    chunk_start = nt_idx * BS
    eos = cutlass.Int32(chunk_start + BS)
    if IS_VARLEN:
        seq_id = cutlass.Int32(mChunkIndices[nt_idx, 0])
        local = cutlass.Int32(mChunkIndices[nt_idx, 1])
        bos = cutlass.Int32(mCuSeqlens[seq_id])
        eos = cutlass.Int32(mCuSeqlens[seq_id + 1])
        chunk_start = bos + local * BS

    for i in range((BS * BS) // THREADS):
        linear = tidx + i * THREADS
        row = linear // BS
        col = linear % BS
        t_row = chunk_start + row
        value = cutlass.Float32(0.0)
        if IS_VARLEN:
            if t_row < eos:
                value = cutlass.Float32(mA_in[b_idx, t_row, h_idx, col])
        else:
            value = cutlass.Float32(mA_in[b_idx, t_row, h_idx, col])
        if row == col:
            value = cutlass.Float32(1.0)
        if row < col:
            value = cutlass.Float32(0.0)
        sAkk[row, col] = value

    if tidx < BS:
        beta_t = chunk_start + tidx
        beta_val = cutlass.Float32(0.0)
        if IS_VARLEN:
            if beta_t < eos:
                beta_val = cutlass.Float32(mBeta[b_idx, beta_t, h_idx])
        else:
            beta_val = cutlass.Float32(mBeta[b_idx, beta_t, h_idx])
        sBeta[tidx] = beta_val
    cute.arch.barrier()

    if tidx < 64:
        _invert_diag_forward16(sAkk, tidx // 16, tidx)
    cute.arch.barrier()

    # 16x16 -> 32x32 for the two diagonal 32x32 blocks.
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

    # 32x32 -> 64x64. Four warps compute partials over x and reduce by y.
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
        if APPLY_BETA_EPILOGUE:
            val0 = val0 * cutlass.Float32(sBeta[col0])
            val1 = val1 * cutlass.Float32(sBeta[col1])
        if IS_VARLEN:
            if t_row < eos:
                mA_out[b_idx, t_row, h_idx, col0] = val0.to(cutlass.BFloat16)
                mA_out[b_idx, t_row, h_idx, col1] = val1.to(cutlass.BFloat16)
        else:
            mA_out[b_idx, t_row, h_idx, col0] = val0.to(cutlass.BFloat16)
            mA_out[b_idx, t_row, h_idx, col1] = val1.to(cutlass.BFloat16)


@cute.jit
def akk_inv_tf32_host(
    A_in: cute.Tensor,
    A_out: cute.Tensor,
    Beta_in: cute.Tensor,
    B: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    T_VAL: cutlass.Constexpr[int],
    APPLY_BETA_EPILOGUE: cutlass.Constexpr[int] = 0,
):
    in_layout = cute.make_layout((B, T_VAL, H, BS), stride=(T_VAL * H * BS, H * BS, BS, 1))
    out_layout = cute.make_layout((B, T_VAL, H, BS), stride=(T_VAL * H * BS, H * BS, BS, 1))
    gA_in = cute.make_tensor(A_in.iterator, in_layout)
    gA_out = cute.make_tensor(A_out.iterator, out_layout)
    smat_layout = cute.make_layout((BS, BS), stride=(AKK_STRIDE, 1))
    stmp_layout = cute.make_layout((TMP_SLOTS, SB, SB), stride=(SB * TMP_STRIDE, TMP_STRIDE, 1))
    smem_bytes = BS * AKK_STRIDE * 4 + TMP_SLOTS * SB * TMP_STRIDE * 4 + BS * 4 + 256

    akk_inv_tf32_kernel(
        gA_in,
        gA_out,
        Beta_in,
        smat_layout,
        stmp_layout,
        NT,
        H,
        mCuSeqlens,
        mChunkIndices,
        IS_VARLEN,
        APPLY_BETA_EPILOGUE,
    ).launch(
        grid=(H, NT, B),
        block=(THREADS, 1, 1),
        smem=smem_bytes,
    )


_compile_cache = {}


def _wrap_tensor(t: torch.Tensor, element_type, *, dynamic: bool, assumed_align: int = 16):
    ct = from_dlpack(t, assumed_align=assumed_align)
    if dynamic:
        ct = ct.mark_layout_dynamic()
    ct.element_type = element_type
    return ct


def _make_eqlen_metadata(device: torch.device):
    cu = torch.empty(2, dtype=torch.int64, device=device)
    ci = torch.empty(1, 2, dtype=torch.int64, device=device)
    cu_ct = _wrap_tensor(cu, cutlass.Int64, dynamic=True, assumed_align=4)
    ci_ct = _wrap_tensor(ci, cutlass.Int64, dynamic=True, assumed_align=4)
    return cu, ci, cu_ct, ci_ct


def akk_inv_tf32(
    a_logical_lower: torch.Tensor,
    beta: torch.Tensor,
    *,
    apply_beta_epilogue: bool = False,
) -> torch.Tensor:
    if a_logical_lower.dtype is not torch.bfloat16:
        raise TypeError("a_logical_lower must be torch.bfloat16")
    if beta.dtype is not torch.bfloat16:
        raise TypeError("beta must be torch.bfloat16")
    if a_logical_lower.ndim != 4 or a_logical_lower.shape[-1] != BS:
        raise ValueError("a_logical_lower must have shape [B, T, H, 64]")
    B, T_VAL, H, _ = a_logical_lower.shape
    if T_VAL % BS != 0:
        raise ValueError("equal-length path requires T to be a multiple of 64")
    out = torch.empty_like(a_logical_lower)
    cu, ci, cu_ct, ci_ct = _make_eqlen_metadata(a_logical_lower.device)
    a_in = _wrap_tensor(a_logical_lower, cutlass.BFloat16, dynamic=False)
    a_out = _wrap_tensor(out, cutlass.BFloat16, dynamic=False)
    beta_ct = _wrap_tensor(beta, cutlass.BFloat16, dynamic=True)
    key = (
        "tf32",
        a_logical_lower.device.index or 0,
        B,
        T_VAL // BS,
        H,
        bool(apply_beta_epilogue),
    )
    if key not in _compile_cache:
        _compile_cache[key] = cute.compile(
            akk_inv_tf32_host,
            a_in,
            a_out,
            beta_ct,
            B,
            T_VAL // BS,
            H,
            cu_ct,
            ci_ct,
            0,
            T_VAL,
            int(apply_beta_epilogue),
        )
    _compile_cache[key](a_in, a_out, beta_ct, cu_ct, ci_ct)
    del cu, ci
    return out

