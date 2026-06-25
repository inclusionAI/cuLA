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

# ruff: noqa: E402, E702, F841

"""
CuteDSL implementation of kda chunk intra (SM100 Blackwell).

Computes intra-chunk backward gradients dq, dk, dg, db for the KDA
attention mechanism.

Warp specialization (384 threads / CTA = 12 warps = 3 warp-groups × 4 warps):
  warps 0-7    : Epilogue (WG0 = warps 0-3, WG1 = warps 4-7)
  warp  8      : Mma    — elect_one executes all tcgen05.mma instructions
  warp  9      : Load   — TMA loads, persistent tile scheduling
  warps 10-11  : Empty  — loads buf_beta, signals mbar_mask_rdy

Pipeline (per tile, 4 K-iterations):
  [Load]      TMA   → SMEM (K, G double-buf; dA single-buf)
  [Epilogue]  mask_A / mask_At  → TMEM
  [Epilogue]  setup kg_all / qkg_all B-matrices in SMEM
  [Mma]       dAqk × kg   → TMEM[dq],  dAqk_t × qkg → TMEM[dkt]
  [Epilogue]  TMEM → scale → output dq / dk / dg / db
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.tcgen05.helpers import (
    SmemLayoutAtomKind,
    make_smem_layout_atom,
)
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import (
    BFloat16,
    Float32,
    Int32,
    Int64,
    TFloat32,
)
from cutlass.cutlass_dsl import dsl_user_op

from cula.ops.ptx_umma_ext import (
    Tcgen05SmemDescriptor,
    initialize_tcgen05_descriptor,
    tcgen05mma_ts_mask0,
    tcgen05mma_ts_mask02,
    tcgen05mma_ts_mask1,
    tcgen05mma_ts_mask2,
    tcgen05mma_ts_mask3,
    tcgen05mma_ts_mask13,
)

# ============================================================
# Constants
# ============================================================

SUB_T_TILE: int = 16
T_TILE: int = 64
K_SIZE: int = 128
K_TILE: int = 32
K_ITERATION: int = K_SIZE // K_TILE  # = 4
NUM_BUF_A: int = 1
NUM_BUF_VALUE: int = 2
NUM_THREADS: int = 128 * 3  # 384
REG_COMPUTE: int = 184
REG_LOAD: int = 136
CHUNK_SIZE: int = T_TILE

# TMEM column constants
# TMEM ACC Layout F, offset = 16 lanes
LANE16_STRIDE: int = 16 * 65536  # = 1048576
DQ_02: int = 0
DQ_13: int = LANE16_STRIDE
DQ2_02: int = 32
DQ2_13: int = 32 + LANE16_STRIDE
DKT_02: int = 64
DKT_13: int = 64 + LANE16_STRIDE
DAQK_02: int = 96
DAQK_13: int = 96 + LANE16_STRIDE
DAQK_T_02: int = 352
DAQK_T_13: int = 352 + LANE16_STRIDE

# IDESC: M=64, N=32 (=K_TILE), TF32, K-maj A, MN-maj B
IDESC_M64_N32: int = (4 << 24) | (4 << 17) | (1 << 16) | (2 << 10) | (2 << 7) | (1 << 4)

# SMEM B-matrix descriptor constants for MN_SW128_32B with N=32
# LBO = N * sizeof(TF32) / 16 = 32 * 4 / 16 = 8
# SBO = atom_size_bytes / 16 = (32 * 8 * 4) / 16 = 64
B_LBO_N32: int = 8
B_SBO: int = 32
B_SW_MN128_32B: int = 1  # SWIZZLE_128B_BASE32B — required for TF32 + b_major=MN
B_K_STEP_BYTES: int = 1024  # bytes per K-atom step (atom [32,8] × 4 bytes)
KG_SLOT_BYTES: int = 2048  # bytes per kg_intra/inter slot: [32,16] × 4 = 2048
QKG_SLOT_BYTES: int = 4096  # bytes per qkg_intra/inter slot: [32,32] × 4 = 4096

#
ROLE_EMPTY: int = 0x0
ROLE_LOAD: int = 0x1
ROLE_MMA: int = 0x2
ROLE_EPILOGUE: int = 0x3


# ============================================================
# Helpers: _ir, Float32 conversion
# ============================================================


def _ir(val, loc=None, ip=None):
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


@dsl_user_op
def bf16_to_f32(val, *, loc=None, ip=None):
    """Convert a BFloat16 value to Float32 using arith.extf (no inline asm)."""
    bf16_ir = BFloat16(val).ir_value(loc=loc, ip=ip)
    f32_ir = _arith.extf(ir.F32Type.get(), bf16_ir, loc=loc, ip=ip)
    return Float32(f32_ir)


@dsl_user_op
def f32_to_bf16(val, *, loc=None, ip=None):
    """Convert a Float32 value to BFloat16 using native arith.truncf."""
    f32_ir = Float32(val).ir_value(loc=loc, ip=ip)
    bf16_ir = _arith.truncf(BFloat16.mlir_type, f32_ir, loc=loc, ip=ip)
    return BFloat16(bf16_ir)


# ============================================================
# TMEM load/store PTX wrappers
# ============================================================


def _tmem_ptr(addr, loc, ip):
    """Convert an i32 TMEM address to !llvm.ptr<6> (TMEM address space)."""
    tmem_ptr_t = llvm.PointerType.get(address_space=6)
    return llvm.inttoptr(tmem_ptr_t, _ir(Int32(addr), loc, ip), loc=loc, ip=ip)


def _f32_to_i32(val, loc, ip):
    """Bitcast f32 -> i32 for TMEM store ops."""
    return llvm.bitcast(ir.IntegerType.get_signless(32), _ir(Float32(val), loc, ip), loc=loc, ip=ip)


def _i32_to_f32(val, loc, ip):
    """Bitcast i32 -> f32 for TMEM load results."""
    return llvm.bitcast(ir.F32Type.get(), val, loc=loc, ip=ip)


@dsl_user_op
def tmem_ld_x16(addr: Int32, *, loc=None, ip=None):
    """Load 16 float32 from TMEM. tcgen05.ld.sync.aligned.32x32b.x16"""
    i32_t = ir.IntegerType.get_signless(32)
    vec_t = ir.VectorType.get([16], i32_t)
    result = _nvvm.tcgen05_ld(
        res=vec_t,
        shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
        num=16,
        tmem_addr=_tmem_ptr(addr, loc, ip),
        loc=loc,
        ip=ip,
    )
    vals = tuple(
        Float32(_i32_to_f32(llvm.extractelement(result, position=_ir(Int32(i), loc, ip), loc=loc, ip=ip), loc, ip))
        for i in range(16)
    )
    return vals


@dsl_user_op
def tmem_ld_x32(addr: Int32, *, loc=None, ip=None):
    """Load 32 float32 from TMEM. tcgen05.ld.sync.aligned.32x32b.x32"""
    i32_t = ir.IntegerType.get_signless(32)
    vec_t = ir.VectorType.get([32], i32_t)
    result = _nvvm.tcgen05_ld(
        res=vec_t,
        shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
        num=32,
        tmem_addr=_tmem_ptr(addr, loc, ip),
        loc=loc,
        ip=ip,
    )
    vals = tuple(
        Float32(_i32_to_f32(llvm.extractelement(result, position=_ir(Int32(i), loc, ip), loc=loc, ip=ip), loc, ip))
        for i in range(32)
    )
    return vals


@dsl_user_op
def tmem_st_x32(
    addr: Int32,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    v8: Float32,
    v9: Float32,
    v10: Float32,
    v11: Float32,
    v12: Float32,
    v13: Float32,
    v14: Float32,
    v15: Float32,
    v16: Float32,
    v17: Float32,
    v18: Float32,
    v19: Float32,
    v20: Float32,
    v21: Float32,
    v22: Float32,
    v23: Float32,
    v24: Float32,
    v25: Float32,
    v26: Float32,
    v27: Float32,
    v28: Float32,
    v29: Float32,
    v30: Float32,
    v31: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 32 float32 to TMEM. tcgen05.st.sync.aligned.32x32b.x32"""
    i32_t = ir.IntegerType.get_signless(32)
    vec_t = ir.VectorType.get([32], i32_t)
    all_vals = [
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        v8,
        v9,
        v10,
        v11,
        v12,
        v13,
        v14,
        v15,
        v16,
        v17,
        v18,
        v19,
        v20,
        v21,
        v22,
        v23,
        v24,
        v25,
        v26,
        v27,
        v28,
        v29,
        v30,
        v31,
    ]
    vec = llvm.mlir_undef(vec_t, loc=loc, ip=ip)
    for i, v in enumerate(all_vals):
        vec = llvm.insertelement(vec, _f32_to_i32(v, loc, ip), position=_ir(Int32(i), loc, ip), loc=loc, ip=ip)
    _nvvm.tcgen05_st(
        shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
        num=32,
        tmem_addr=_tmem_ptr(addr, loc, ip),
        r=vec,
        loc=loc,
        ip=ip,
    )


@cute.jit
def tcgen05_fence_before():
    """tcgen05.fence::before_thread_sync — non-blocking ordering fence."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_fence_after():
    """tcgen05.fence::after_thread_sync — non-blocking ordering fence."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


@cute.jit
def umma_arrive_noelect(mbar_ptr: cute.Pointer):
    """tcgen05.commit.cta_group::1.mbarrier::arrive::one — signal MMA done."""
    tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)


# ============================================================
# B-matrix write helpers (SMEM setup)
# Each thread writes 4 TF32 values to the B-matrix.
# ============================================================


@cute.jit
def write_b4(buf_Btens, col_base: Int32, row: Int32, v0: Float32, v1: Float32, v2: Float32, v3: Float32):
    """
    Write 4 consecutive TF32 values at (col_base..col_base+3, row) in B-matrix
    with SWIZZLE_128B_BASE32B (Swizzle<2,5,2>) applied at the BYTE level.

    The hardware MMA descriptor applies SWIZZLE_128B_BASE32B on BYTE offsets:
    XOR byte_addr bits[5:6] with bits[7:8].  For the B-matrix layout
    (N=32 TF32 stride-1, K stride-32), byte = col*4 + k_within*128.
    bits[7:8] = (k_within & 3) = (row & 3), so the column swizzle is:
      swizzled_col = col ^ ((row & 3) << 3)
    The row coordinate is unchanged.

    col_base is always 4-aligned, so (col_base + i) ^ X = (col_base ^ X) + i
    whenever X only touches bits >= 3 (which is true since X = 0, 8, 16, or 24).
    """
    xor_bits = (row & Int32(3)) << Int32(3)
    swizzled_base = col_base ^ xor_bits
    buf_Btens[(swizzled_base, row)] = TFloat32(v0)
    buf_Btens[(swizzled_base + Int32(1), row)] = TFloat32(v1)
    buf_Btens[(swizzled_base + Int32(2), row)] = TFloat32(v2)
    buf_Btens[(swizzled_base + Int32(3), row)] = TFloat32(v3)


# ============================================================
# B-matrix setup functions
# ============================================================


@cute.jit
def setup_kg_intra(
    buf_G_raw: cute.Pointer,  # Float32 raw SMEM ptr (no swizzle)
    buf_K_raw: cute.Pointer,  # BFloat16 raw SMEM ptr
    buf_KG_in,  # [K_TILE, 6*SUB_T_TILE] TF32 SMEM tensor (B-matrix)
    tile_j: cutlass.Constexpr,
    idx_in_warpgroup: Int32,
    gn0: Float32,
    gn1: Float32,
    gn2: Float32,
    gn3: Float32,  # gn float4
    kg_index: cutlass.Constexpr,
):
    """
    B-matrix setup for kg_intra (vectorized SMEM loads).
    """
    KG_OFFSET: cutlass.Constexpr = SUB_T_TILE
    x = idx_in_warpgroup // Int32(8) + Int32(tile_j * 16)
    col_base = (idx_in_warpgroup % Int32(8)) * Int32(4)
    n_base = idx_in_warpgroup // Int32(8)
    n_col = n_base + Int32(kg_index * KG_OFFSET)

    # Vectorized SMEM loads
    g = smem_load_f32x4_sw128(buf_G_raw, x, col_base)
    k_bf = smem_load_bf16x4_sw64(buf_K_raw, x, col_base)
    k0 = bf16_to_f32(k_bf[0])
    k1 = bf16_to_f32(k_bf[1])
    k2 = bf16_to_f32(k_bf[2])
    k3 = bf16_to_f32(k_bf[3])

    v0 = cute.arch.exp2(gn0 - g[0]) * k0
    v1 = cute.arch.exp2(gn1 - g[1]) * k1
    v2 = cute.arch.exp2(gn2 - g[2]) * k2
    v3 = cute.arch.exp2(gn3 - g[3]) * k3

    write_b4(buf_KG_in, col_base, n_col, v0, v1, v2, v3)


@cute.jit
def setup_kg_intra_2gn(
    buf_G_raw: cute.Pointer,
    buf_K_raw: cute.Pointer,
    buf_KG_in,
    tile_j: cutlass.Constexpr,
    idx_in_warpgroup: Int32,
    gn1_0: Float32,
    gn1_1: Float32,
    gn1_2: Float32,
    gn1_3: Float32,
    gn2_0: Float32,
    gn2_1: Float32,
    gn2_2: Float32,
    gn2_3: Float32,
    kg_index1: cutlass.Constexpr,
    kg_index2: cutlass.Constexpr,
):
    """Two kg_intra outputs from one row load (vectorized SMEM loads)."""
    KG_OFFSET: cutlass.Constexpr = SUB_T_TILE
    x = idx_in_warpgroup // Int32(8) + Int32(tile_j * 16)
    col_base = (idx_in_warpgroup % Int32(8)) * Int32(4)
    n_base = idx_in_warpgroup // Int32(8)

    g = smem_load_f32x4_sw128(buf_G_raw, x, col_base)
    k_bf = smem_load_bf16x4_sw64(buf_K_raw, x, col_base)
    k0 = bf16_to_f32(k_bf[0])
    k1 = bf16_to_f32(k_bf[1])
    k2 = bf16_to_f32(k_bf[2])
    k3 = bf16_to_f32(k_bf[3])

    # Output 1
    s1a0 = cute.arch.exp2(gn1_0 - g[0]) * k0
    s1a1 = cute.arch.exp2(gn1_1 - g[1]) * k1
    s1a2 = cute.arch.exp2(gn1_2 - g[2]) * k2
    s1a3 = cute.arch.exp2(gn1_3 - g[3]) * k3
    n1 = n_base + Int32(kg_index1 * KG_OFFSET)
    write_b4(buf_KG_in, col_base, n1, s1a0, s1a1, s1a2, s1a3)

    # Output 2
    s2a0 = cute.arch.exp2(gn2_0 - g[0]) * k0
    s2a1 = cute.arch.exp2(gn2_1 - g[1]) * k1
    s2a2 = cute.arch.exp2(gn2_2 - g[2]) * k2
    s2a3 = cute.arch.exp2(gn2_3 - g[3]) * k3
    n2 = n_base + Int32(kg_index2 * KG_OFFSET)
    write_b4(buf_KG_in, col_base, n2, s2a0, s2a1, s2a2, s2a3)


@cute.jit
def setup_intra_fused(
    buf_G_raw: cute.Pointer,
    buf_K_raw: cute.Pointer,
    buf_Q_raw: cute.Pointer,
    buf_KG_in,  # [K_TILE, 6*SUB_T_TILE] TF32
    buf_QKG_in,  # [K_TILE, 6*2*SUB_T_TILE] TF32
    tile_j: cutlass.Constexpr,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    gn_kg0: Float32,
    gn_kg1: Float32,
    gn_kg2: Float32,
    gn_kg3: Float32,
    gn_qkg0: Float32,
    gn_qkg1: Float32,
    gn_qkg2: Float32,
    gn_qkg3: Float32,
    beta0: Float32,
    beta1: Float32,
    kg_index: cutlass.Constexpr,
    qkg_index: cutlass.Constexpr,
):
    """Fused kg_intra + qkg_intra (vectorized SMEM loads)."""
    KG_OFFSET: cutlass.Constexpr = SUB_T_TILE
    QKG_OFFSET: cutlass.Constexpr = 2 * SUB_T_TILE
    x = idx_in_warpgroup // Int32(8) + Int32(tile_j * 16)
    col_base = (idx_in_warpgroup % Int32(8)) * Int32(4)
    n_base = idx_in_warpgroup // Int32(8)

    if x < sub_seq_len:
        g = smem_load_f32x4_sw128(buf_G_raw, x, col_base)
        k_bf = smem_load_bf16x4_sw64(buf_K_raw, x, col_base)
        k0 = bf16_to_f32(k_bf[0])
        k1 = bf16_to_f32(k_bf[1])
        k2 = bf16_to_f32(k_bf[2])
        k3 = bf16_to_f32(k_bf[3])
        q_bf = smem_load_bf16x4_sw64(buf_Q_raw, x, col_base)
        q0 = bf16_to_f32(q_bf[0])
        q1 = bf16_to_f32(q_bf[1])
        q2 = bf16_to_f32(q_bf[2])
        q3 = bf16_to_f32(q_bf[3])

        # kg_intra: exp2f(gn_kg - g) * k
        sk0 = cute.arch.exp2(gn_kg0 - g[0])
        sk1 = cute.arch.exp2(gn_kg1 - g[1])
        sk2 = cute.arch.exp2(gn_kg2 - g[2])
        sk3 = cute.arch.exp2(gn_kg3 - g[3])
        n_kg = n_base + Int32(kg_index * KG_OFFSET)
        write_b4(buf_KG_in, col_base, n_kg, sk0 * k0, sk1 * k1, sk2 * k2, sk3 * k3)

        # qkg_intra q-part: exp2f(g - gn_qkg) * q
        sq0 = cute.arch.exp2(g[0] - gn_qkg0)
        sq1 = cute.arch.exp2(g[1] - gn_qkg1)
        sq2 = cute.arch.exp2(g[2] - gn_qkg2)
        sq3 = cute.arch.exp2(g[3] - gn_qkg3)
        n_qkg = n_base + Int32(qkg_index * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n_qkg, sq0 * q0, sq1 * q1, sq2 * q2, sq3 * q3)

        # qkg_intra k-beta part: exp2f(g - gn_qkg) * k * beta
        n_qkg_k = n_base + Int32(qkg_index * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n_qkg_k, sq0 * k0 * beta0, sq1 * k1 * beta1, sq2 * k2 * beta0, sq3 * k3 * beta1)
    else:
        zero = Float32(0.0)
        n_kg = n_base + Int32(kg_index * KG_OFFSET)
        write_b4(buf_KG_in, col_base, n_kg, zero, zero, zero, zero)
        n_qkg = n_base + Int32(qkg_index * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n_qkg, zero, zero, zero, zero)
        n_qkg_k = n_base + Int32(qkg_index * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n_qkg_k, zero, zero, zero, zero)


@cute.jit
def setup_inter_fused(
    buf_G_raw: cute.Pointer,
    buf_K_raw: cute.Pointer,
    buf_Q_raw: cute.Pointer,
    buf_KG_ex,  # [K_TILE, 4*SUB_T_TILE] TF32
    buf_QKG_ex,  # [K_TILE, 4*2*SUB_T_TILE] TF32
    sub_tile_i: cutlass.Constexpr,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    beta0: Float32,
    beta1: Float32,
):
    """Fused kg_inter + qkg_inter (vectorized SMEM loads)."""
    KG_OFFSET: cutlass.Constexpr = SUB_T_TILE
    QKG_OFFSET: cutlass.Constexpr = 2 * SUB_T_TILE

    col_base = (idx_in_warpgroup % Int32(8)) * Int32(4)
    n_base = idx_in_warpgroup // Int32(8)
    mid_row_val = cutlass.select_(
        Int32(sub_tile_i * 16 + 8) < sub_seq_len,
        Int32(sub_tile_i * 16 + 8),
        sub_seq_len - Int32(1),
    )

    gn_h = smem_load_f32x4_sw128(buf_G_raw, mid_row_val, col_base)

    x = idx_in_warpgroup // Int32(8) + Int32(sub_tile_i * 16)

    if x < sub_seq_len:
        g = smem_load_f32x4_sw128(buf_G_raw, x, col_base)
        k_bf = smem_load_bf16x4_sw64(buf_K_raw, x, col_base)
        k0 = bf16_to_f32(k_bf[0])
        k1 = bf16_to_f32(k_bf[1])
        k2 = bf16_to_f32(k_bf[2])
        k3 = bf16_to_f32(k_bf[3])
        q_bf = smem_load_bf16x4_sw64(buf_Q_raw, x, col_base)
        q0 = bf16_to_f32(q_bf[0])
        q1 = bf16_to_f32(q_bf[1])
        q2 = bf16_to_f32(q_bf[2])
        q3 = bf16_to_f32(q_bf[3])

        sub0 = g[0] - gn_h[0]
        sub1 = g[1] - gn_h[1]
        sub2 = g[2] - gn_h[2]
        sub3 = g[3] - gn_h[3]

        exp0 = cute.arch.exp2(sub0)
        exp1 = cute.arch.exp2(sub1)
        exp2 = cute.arch.exp2(sub2)
        exp3 = cute.arch.exp2(sub3)

        neg_exp0 = cute.arch.exp2(-sub0)
        neg_exp1 = cute.arch.exp2(-sub1)
        neg_exp2 = cute.arch.exp2(-sub2)
        neg_exp3 = cute.arch.exp2(-sub3)

        n_kg = n_base + Int32(sub_tile_i * KG_OFFSET)
        write_b4(buf_KG_ex, col_base, n_kg, neg_exp0 * k0, neg_exp1 * k1, neg_exp2 * k2, neg_exp3 * k3)

        n_qkg = n_base + Int32(sub_tile_i * QKG_OFFSET)
        write_b4(buf_QKG_ex, col_base, n_qkg, exp0 * q0, exp1 * q1, exp2 * q2, exp3 * q3)

        n_qkg_k = n_base + Int32(sub_tile_i * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_ex, col_base, n_qkg_k, exp0 * k0 * beta0, exp1 * k1 * beta1, exp2 * k2 * beta0, exp3 * k3 * beta1)
    else:
        zero = Float32(0.0)
        n_kg = n_base + Int32(sub_tile_i * KG_OFFSET)
        write_b4(buf_KG_ex, col_base, n_kg, zero, zero, zero, zero)
        n_qkg = n_base + Int32(sub_tile_i * QKG_OFFSET)
        write_b4(buf_QKG_ex, col_base, n_qkg, zero, zero, zero, zero)
        n_qkg_k = n_base + Int32(sub_tile_i * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_ex, col_base, n_qkg_k, zero, zero, zero, zero)


@cute.jit
def setup_qkg_intra(
    buf_G_raw: cute.Pointer,
    buf_Q_raw: cute.Pointer,
    buf_K_raw: cute.Pointer,
    buf_QKG_in,  # [K_TILE, 6*2*SUB_T_TILE] TF32
    tile_j: cutlass.Constexpr,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    beta0: Float32,
    beta1: Float32,
    gn0: Float32,
    gn1: Float32,
    gn2: Float32,
    gn3: Float32,
    qkg_index: cutlass.Constexpr,
):
    """qkg_intra setup (vectorized SMEM loads)."""
    QKG_OFFSET: cutlass.Constexpr = 2 * SUB_T_TILE
    x = idx_in_warpgroup // Int32(8) + Int32(tile_j * 16)
    col_base = (idx_in_warpgroup % Int32(8)) * Int32(4)
    n_base = idx_in_warpgroup // Int32(8)

    if x < sub_seq_len:
        g = smem_load_f32x4_sw128(buf_G_raw, x, col_base)
        k_bf = smem_load_bf16x4_sw64(buf_K_raw, x, col_base)
        k0 = bf16_to_f32(k_bf[0])
        k1 = bf16_to_f32(k_bf[1])
        k2 = bf16_to_f32(k_bf[2])
        k3 = bf16_to_f32(k_bf[3])
        q_bf = smem_load_bf16x4_sw64(buf_Q_raw, x, col_base)
        q0 = bf16_to_f32(q_bf[0])
        q1 = bf16_to_f32(q_bf[1])
        q2 = bf16_to_f32(q_bf[2])
        q3 = bf16_to_f32(q_bf[3])

        sq0 = cute.arch.exp2(g[0] - gn0)
        sq1 = cute.arch.exp2(g[1] - gn1)
        sq2 = cute.arch.exp2(g[2] - gn2)
        sq3 = cute.arch.exp2(g[3] - gn3)

        n_q = n_base + Int32(qkg_index * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n_q, sq0 * q0, sq1 * q1, sq2 * q2, sq3 * q3)

        n_k = n_base + Int32(qkg_index * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n_k, sq0 * k0 * beta0, sq1 * k1 * beta1, sq2 * k2 * beta0, sq3 * k3 * beta1)
    else:
        zero = Float32(0.0)
        n_q = n_base + Int32(qkg_index * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n_q, zero, zero, zero, zero)
        n_k = n_base + Int32(qkg_index * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n_k, zero, zero, zero, zero)


@cute.jit
def setup_qkg_intra_2gn(
    buf_G_raw: cute.Pointer,
    buf_Q_raw: cute.Pointer,
    buf_K_raw: cute.Pointer,
    buf_QKG_in,
    tile_j: cutlass.Constexpr,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    beta0: Float32,
    beta1: Float32,
    gn1_0: Float32,
    gn1_1: Float32,
    gn1_2: Float32,
    gn1_3: Float32,
    gn2_0: Float32,
    gn2_1: Float32,
    gn2_2: Float32,
    gn2_3: Float32,
    qkg_index1: cutlass.Constexpr,
    qkg_index2: cutlass.Constexpr,
):
    """Two qkg_intra outputs (vectorized SMEM loads)."""
    QKG_OFFSET: cutlass.Constexpr = 2 * SUB_T_TILE
    x = idx_in_warpgroup // Int32(8) + Int32(tile_j * 16)
    col_base = (idx_in_warpgroup % Int32(8)) * Int32(4)
    n_base = idx_in_warpgroup // Int32(8)

    if x < sub_seq_len:
        g = smem_load_f32x4_sw128(buf_G_raw, x, col_base)
        k_bf = smem_load_bf16x4_sw64(buf_K_raw, x, col_base)
        k0 = bf16_to_f32(k_bf[0])
        k1 = bf16_to_f32(k_bf[1])
        k2 = bf16_to_f32(k_bf[2])
        k3 = bf16_to_f32(k_bf[3])
        q_bf = smem_load_bf16x4_sw64(buf_Q_raw, x, col_base)
        q0 = bf16_to_f32(q_bf[0])
        q1 = bf16_to_f32(q_bf[1])
        q2 = bf16_to_f32(q_bf[2])
        q3 = bf16_to_f32(q_bf[3])

        # Output 1 with gn1
        s1_0 = cute.arch.exp2(g[0] - gn1_0)
        s1_1 = cute.arch.exp2(g[1] - gn1_1)
        s1_2 = cute.arch.exp2(g[2] - gn1_2)
        s1_3 = cute.arch.exp2(g[3] - gn1_3)

        n1_q = n_base + Int32(qkg_index1 * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n1_q, s1_0 * q0, s1_1 * q1, s1_2 * q2, s1_3 * q3)
        n1_k = n_base + Int32(qkg_index1 * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n1_k, s1_0 * k0 * beta0, s1_1 * k1 * beta1, s1_2 * k2 * beta0, s1_3 * k3 * beta1)

        # Output 2 with gn2
        s2_0 = cute.arch.exp2(g[0] - gn2_0)
        s2_1 = cute.arch.exp2(g[1] - gn2_1)
        s2_2 = cute.arch.exp2(g[2] - gn2_2)
        s2_3 = cute.arch.exp2(g[3] - gn2_3)

        n2_q = n_base + Int32(qkg_index2 * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n2_q, s2_0 * q0, s2_1 * q1, s2_2 * q2, s2_3 * q3)
        n2_k = n_base + Int32(qkg_index2 * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n2_k, s2_0 * k0 * beta0, s2_1 * k1 * beta1, s2_2 * k2 * beta0, s2_3 * k3 * beta1)
    else:
        zero = Float32(0.0)
        n1_q = n_base + Int32(qkg_index1 * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n1_q, zero, zero, zero, zero)
        n1_k = n_base + Int32(qkg_index1 * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n1_k, zero, zero, zero, zero)
        n2_q = n_base + Int32(qkg_index2 * QKG_OFFSET)
        write_b4(buf_QKG_in, col_base, n2_q, zero, zero, zero, zero)
        n2_k = n_base + Int32(qkg_index2 * QKG_OFFSET + SUB_T_TILE)
        write_b4(buf_QKG_in, col_base, n2_k, zero, zero, zero, zero)


# ============================================================
# mask_A / mask_At: apply triangular mask → TMEM store
# ============================================================


@cute.jit
def mask_A_tensor(
    sDA,  # [T_TILE, T_TILE] float32 SMEM tensor (dAqk or dAkk)
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    tmem_col_base: Int32,  # base TMEM column address
    offset: Int32,
):
    """
    Load 32 elements of dA (lower-triangular masked), store to TMEM[dAqk].
    """
    x = idx_in_warpgroup % Int32(64)
    # Load + mask 32 values: 8 groups of 4 elements
    res = [Float32(0.0)] * 32
    for i in cutlass.range_constexpr(8):
        y0 = i * 4 + offset
        v0 = sDA[(x, Int32(y0))]
        v1 = sDA[(x, Int32(y0 + 1))]
        v2 = sDA[(x, Int32(y0 + 2))]
        v3 = sDA[(x, Int32(y0 + 3))]
        # Mask: zero if x >= sub_seq_len OR x < y+j OR y+j >= sub_seq_len
        v0 = cutlass.select_(
            (x >= sub_seq_len) | (x < Int32(y0)) | (Int32(y0) >= sub_seq_len),
            Float32(0.0),
            v0,
        )
        v1 = cutlass.select_(
            (x >= sub_seq_len) | (x < Int32(y0 + 1)) | (Int32(y0 + 1) >= sub_seq_len),
            Float32(0.0),
            v1,
        )
        v2 = cutlass.select_(
            (x >= sub_seq_len) | (x < Int32(y0 + 2)) | (Int32(y0 + 2) >= sub_seq_len),
            Float32(0.0),
            v2,
        )
        v3 = cutlass.select_(
            (x >= sub_seq_len) | (x < Int32(y0 + 3)) | (Int32(y0 + 3) >= sub_seq_len),
            Float32(0.0),
            v3,
        )
        res[i * 4] = v0
        res[i * 4 + 1] = v1
        res[i * 4 + 2] = v2
        res[i * 4 + 3] = v3

    # TMEM store x32 at dAqk_02/13 + 256*buf + offset
    tmem_st_x32(
        tmem_col_base + Int32(offset),
        res[0],
        res[1],
        res[2],
        res[3],
        res[4],
        res[5],
        res[6],
        res[7],
        res[8],
        res[9],
        res[10],
        res[11],
        res[12],
        res[13],
        res[14],
        res[15],
        res[16],
        res[17],
        res[18],
        res[19],
        res[20],
        res[21],
        res[22],
        res[23],
        res[24],
        res[25],
        res[26],
        res[27],
        res[28],
        res[29],
        res[30],
        res[31],
    )


@cute.jit
def mask_At_tensor(
    buf_DAqk,  # [T_TILE, T_TILE] float32 SMEM tensor
    buf_DAkk,  # [T_TILE, T_TILE] float32 SMEM tensor
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    tmem_col_base: Int32,  # DAQK_T_02 or DAQK_T_13
    offset: Int32,
):
    """
    Transpose dA and apply upper-triangular mask → TMEM[dAqk_t].
    Two stages of 32 stores each.
    """
    x = idx_in_warpgroup % Int32(64)
    # Two stages of 32 elements (TILE_SIZE=64, stage 0 and stage 1)
    for stage in cutlass.range_constexpr(2):
        res = [Float32(0.0)] * 32
        for i in cutlass.range_constexpr(16):
            y = i + 16 * stage + offset // 2
            # res[i] = buf_DAqk[y, x] if x <= y and x,y < sub_seq_len else 0
            mask_val = (x >= sub_seq_len) | (x > Int32(y)) | (Int32(y) >= sub_seq_len)
            v_aqk = buf_DAqk[(Int32(y), x)]
            v_akk = buf_DAkk[(Int32(y), x)]
            res[i] = cutlass.select_(mask_val, Float32(0.0), v_aqk)
            res[i + 16] = cutlass.select_(mask_val, Float32(0.0), v_akk)

        tmem_addr = tmem_col_base + Int32(stage * 32 + offset)
        tmem_st_x32(
            tmem_addr,
            res[0],
            res[1],
            res[2],
            res[3],
            res[4],
            res[5],
            res[6],
            res[7],
            res[8],
            res[9],
            res[10],
            res[11],
            res[12],
            res[13],
            res[14],
            res[15],
            res[16],
            res[17],
            res[18],
            res[19],
            res[20],
            res[21],
            res[22],
            res[23],
            res[24],
            res[25],
            res[26],
            res[27],
            res[28],
            res[29],
            res[30],
            res[31],
        )
        cute.arch.fence_view_async_tmem_store()


# ============================================================
# MMA warp B-descriptor initialization
# Called once per tile in the MMA warp body.
# ============================================================


@cute.jit
def build_b_desc(smem_ptr: cute.Pointer) -> Tcgen05SmemDescriptor:
    """Build MN_SW128_32B B-matrix descriptor from SMEM pointer."""
    desc = Tcgen05SmemDescriptor()
    initialize_tcgen05_descriptor(desc, smem_ptr, B_LBO_N32, B_SBO, 0, False, B_SW_MN128_32B)
    return desc


# ============================================================
# Issue MMA calls for each group (unrolled K-steps)
# ============================================================


@cute.jit
def mma_kg_intra_call1(
    tmem_a_base: Int32,
    desc_kg_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
):
    """
    kg_intra call 1: MASK02, A=dAqk_13, B=intra[0], K=1 sub-tile (2 K-atoms), C=dq.
    scale_out=0 first step (clear chunks 0,2), then 1 (accumulate).
    """
    for ks in cutlass.range_constexpr(2):
        scale = 0 if ks == 0 else 1
        tmem_a = tmem_a_base + Int32(ks * 8)
        desc_b = desc_kg_base + (ks * B_K_STEP_BYTES)
        tcgen05mma_ts_mask02(tmem_a, desc_b, tmem_c, IDESC_M64_N32, scale)


@cute.jit
def mma_kg_intra_call2(
    tmem_a_base: Int32,
    desc_kg_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
):
    """
    kg_intra call 2: MASK13, A=dAqk_02, B=intra[1..2], K=2 sub-tiles (4 K-atoms), C=dq.
    scale_out=0 first step (clear chunks 1,3), then 1 (accumulate).
    """
    for ks in cutlass.range_constexpr(4):
        scale = 0 if ks == 0 else 1
        tmem_a = tmem_a_base + Int32(ks * 8)
        desc_b = desc_kg_base + (ks * B_K_STEP_BYTES)
        tcgen05mma_ts_mask13(tmem_a, desc_b, tmem_c, IDESC_M64_N32, scale)


@cute.jit
def mma_kg_intra_call3(
    tmem_a_base: Int32,
    desc_kg_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
):
    """
    kg_intra call 3: MASK13, A=dAqk_13, B=intra[3..5], K=3 sub-tiles (6 K-atoms), C=dq.
    scale_out=0 first step (clear chunks 1,3), then 1 (accumulate).
    """
    for ks in cutlass.range_constexpr(6):
        scale = 0 if ks == 0 else 1
        tmem_a = tmem_a_base + Int32(ks * 8)
        desc_b = desc_kg_base + (ks * B_K_STEP_BYTES)
        tcgen05mma_ts_mask13(tmem_a, desc_b, tmem_c, IDESC_M64_N32, scale)


@cute.jit
def mma_kg_inter_call(
    tmem_a: Int32,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    mask_type: cutlass.Constexpr,
):
    """
    kg_inter single call: 1 sub-tile (2 K-atoms), any mask type.
    mask_type: 0=MASK02, 1=MASK13
    """
    for ks in cutlass.range_constexpr(2):
        scale = 0 if ks == 0 else 1
        a = tmem_a + Int32(ks * 8)
        d = desc_b + (ks * B_K_STEP_BYTES)
        if cutlass.const_expr(mask_type == 0):
            tcgen05mma_ts_mask02(a, d, tmem_c, IDESC_M64_N32, scale)
        else:
            tcgen05mma_ts_mask13(a, d, tmem_c, IDESC_M64_N32, scale)


@cute.jit
def mma_qkg_intra_call0(
    tmem_a_base: Int32,
    desc_qkg_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
):
    """
    qkg_intra call 0: MASK0, A=dAqk_t_02+32, B=qkg_intra[0..5]=12 K-atoms, C=dkt_02.
    """
    for ks in cutlass.range_constexpr(12):
        scale = 0 if ks == 0 else 1
        a = tmem_a_base + Int32(ks * 8)
        d = desc_qkg_base + (ks * B_K_STEP_BYTES)
        tcgen05mma_ts_mask0(a, d, tmem_c, IDESC_M64_N32, scale)


@cute.jit
def mma_qkg_intra_call1(
    tmem_a_base: Int32,
    desc_qkg_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
):
    """
    qkg_intra call 1: MASK0, A=dAqk_t_13+64, B=qkg_intra[3..6]=8 K-atoms, C=dkt_13.
    """
    for ks in cutlass.range_constexpr(8):
        scale = 0 if ks == 0 else 1
        a = tmem_a_base + Int32(ks * 8)
        d = desc_qkg_base + (ks * B_K_STEP_BYTES)
        tcgen05mma_ts_mask0(a, d, tmem_c, IDESC_M64_N32, scale)


@cute.jit
def mma_qkg_intra_call2(
    tmem_a_base: Int32,
    desc_qkg_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
):
    """
    qkg_intra call 2: MASK1, A=dAqk_t_02+96, B=qkg_intra[5..6]=4 K-atoms, C=dkt_02.
    """
    for ks in cutlass.range_constexpr(4):
        scale = 0 if ks == 0 else 1
        a = tmem_a_base + Int32(ks * 8)
        d = desc_qkg_base + (ks * B_K_STEP_BYTES)
        tcgen05mma_ts_mask1(a, d, tmem_c, IDESC_M64_N32, scale)


@cute.jit
def mma_qkg_inter_call(
    tmem_a: Int32,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    mask_type: cutlass.Constexpr,
):
    """
    qkg_inter single call: 2 sub-tile groups (4 K-atoms), mask_type 2 or 3.
    """
    for ks in cutlass.range_constexpr(4):
        scale = 0 if ks == 0 else 1
        a = tmem_a + Int32(ks * 8)
        d = desc_b + (ks * B_K_STEP_BYTES)
        if cutlass.const_expr(mask_type == 2):
            tcgen05mma_ts_mask2(a, d, tmem_c, IDESC_M64_N32, scale)
        else:
            tcgen05mma_ts_mask3(a, d, tmem_c, IDESC_M64_N32, scale)


# ============================================================
# Vectorized SMEM load helpers
# Replaces scalar buf_G[(row, col)] with 4-wide loads by computing
# the swizzled address manually and using make_ptr with align hint.
# ============================================================


@cute.jit
def smem_load_f32x4_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 4 consecutive float32 from SMEM with Swizzle<3,4,3> layout.
    raw_ptr: Float32 SMEM base pointer (NOT recast_ptr — raw buffer start)
    row: row index in [0, T_TILE)
    col_base: 4-aligned column index
    row_stride: K_TILE = 32 elements

    Swizzle<3,4,3> on float32 with stride=32:
      swizzled_col = col_base ^ ((row & 7) << 2)
    4 consecutive elements at aligned col_base are provably contiguous.
    """
    swizzled_col = col_base ^ ((row & Int32(7)) << Int32(2))
    elem_offset = row * Int32(K_TILE) + swizzled_col
    aligned_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    t = cute.make_tensor(aligned_ptr, cute.make_layout((4,), stride=(1,)))
    vals = t.load()
    return (vals[0], vals[1], vals[2], vals[3])


@cute.jit
def smem_load_bf16x4_sw64(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 4 consecutive bf16 from SMEM with Swizzle<2,4,3> layout.
    raw_ptr: BFloat16 SMEM base pointer
    row_stride: K_TILE = 32 elements = 64 bytes

    Swizzle<2,4,3> on bf16 at byte level:
      byte_off = row * 64 + col * 2
      phys_byte = byte_off ^ (((byte_off >> 7) & 3) << 4)
    At element level for 4-aligned col_base:
      swizzled_col = col_base ^ (((row * 64 + col_base * 2) >> 7 & 3) << 3)
    But simpler: compute byte offset, XOR, convert back.
    """
    byte_off = row * Int32(64) + col_base * Int32(2)
    xor_val = ((byte_off >> Int32(7)) & Int32(3)) << Int32(4)
    phys_byte_off = byte_off ^ xor_val
    phys_elem_off = phys_byte_off >> Int32(1)
    aligned_ptr = cute.make_ptr(
        BFloat16,
        (raw_ptr + phys_elem_off).toint(),
        cute.AddressSpace.smem,
        assumed_align=8,
    )
    t = cute.make_tensor(aligned_ptr, cute.make_layout((4,), stride=(1,)))
    vals = t.load()
    return (vals[0], vals[1], vals[2], vals[3])


@cute.jit
def smem_load_f32x4_sw64_da(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 4 consecutive float32 from SMEM with Swizzle<2,4,3> layout.
    For dA tensors: row_stride = T_TILE = 64 elements = 256 bytes.
    """
    byte_off = row * Int32(256) + col_base * Int32(4)
    xor_val = ((byte_off >> Int32(7)) & Int32(3)) << Int32(4)
    phys_byte_off = byte_off ^ xor_val
    phys_elem_off = phys_byte_off >> Int32(2)
    aligned_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + phys_elem_off).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    t = cute.make_tensor(aligned_ptr, cute.make_layout((4,), stride=(1,)))
    vals = t.load()
    return (vals[0], vals[1], vals[2], vals[3])


@cute.jit
def smem_load_f32x4_noswizzle(raw_ptr: cute.Pointer, row: Int32, col_base: Int32, row_stride: Int32):
    """
    Load 4 consecutive float32 from SMEM without swizzle.
    For sDKT buffers: row_stride = 36 elements = 144 bytes.
    col_base must be 4-aligned for 16-byte alignment.
    """
    elem_offset = row * row_stride + col_base
    aligned_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    t = cute.make_tensor(aligned_ptr, cute.make_layout((4,), stride=(1,)))
    vals = t.load()
    return (vals[0], vals[1], vals[2], vals[3])


@cute.jit
def smem_store_f32x4_noswizzle(
    raw_ptr: cute.Pointer, row: Int32, col_base: Int32, row_stride: Int32, v0: Float32, v1: Float32, v2: Float32, v3: Float32
):
    """
    Store 4 consecutive float32 to SMEM without swizzle (vectorized STS.128).
    row_stride must be a multiple of 4 for 16-byte alignment.
    col_base must be 4-aligned for 16-byte alignment.
    """
    elem_offset = row * row_stride + col_base
    smem_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(smem_ptr, cute.make_layout((4,), stride=(1,)))
    rmem_t = cute.make_fragment_like(smem_t)
    rmem_t[(0,)] = v0
    rmem_t[(1,)] = v1
    rmem_t[(2,)] = v2
    rmem_t[(3,)] = v3
    cute.autovec_copy(rmem_t, smem_t)


@cute.jit
def smem_store_f32x4_sw128(
    raw_ptr: cute.Pointer, row: Int32, col_base: Int32, v0: Float32, v1: Float32, v2: Float32, v3: Float32
):
    """
    Store 4 consecutive float32 to SMEM with Swizzle<3,4,3> layout.
    Uses K_TILE (32) as row stride, same swizzle as buf_G buffers.
    """
    swizzled_col = col_base ^ ((row & Int32(7)) << Int32(2))
    elem_offset = row * Int32(K_TILE) + swizzled_col
    smem_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(smem_ptr, cute.make_layout((4,), stride=(1,)))
    rmem_t = cute.make_fragment_like(smem_t)
    rmem_t[(0,)] = v0
    rmem_t[(1,)] = v1
    rmem_t[(2,)] = v2
    rmem_t[(3,)] = v3
    cute.autovec_copy(rmem_t, smem_t)


# ============================================================
# (Vectorized GMEM store helpers removed — using cute.autovec_copy)
# ============================================================


# ============================================================
# Fused scale computation
# ============================================================


@cute.jit
def epilogue_compute_scales_fused(
    buf_G_raw_ptr: cute.Pointer,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
):
    """
    Fused intra + inter scale computation.
    Reads buf_G[row, col] once, shares with both intra and inter ref.
    Returns 32-tuple: (intra_sc[0..15], inter_sc[0..15]).
    """
    local = idx_in_warpgroup % Int32(64)
    upper = local >= Int32(16)
    intra_ref = (local // Int32(16)) * Int32(16)
    inter_ref = cutlass.select_(
        local // Int32(16) * Int32(16) + Int32(8) < sub_seq_len,
        local // Int32(16) * Int32(16) + Int32(8),
        sub_seq_len - Int32(1),
    )

    intra_res = [Float32(0.0)] * 16
    inter_res = [Float32(0.0)] * 16

    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        # Read row values (shared between both scales)
        bg = smem_load_f32x4_sw128(buf_G_raw_ptr, local, Int32(col))
        # Read inter ref values
        bi = smem_load_f32x4_sw128(buf_G_raw_ptr, inter_ref, Int32(col))
        # Read intra ref values
        bgn = smem_load_f32x4_sw128(buf_G_raw_ptr, intra_ref, Int32(col))

        # Inter scale: always computed
        inter_res[i * 4] = cute.arch.exp2(bg[0] - bi[0])
        inter_res[i * 4 + 1] = cute.arch.exp2(bg[1] - bi[1])
        inter_res[i * 4 + 2] = cute.arch.exp2(bg[2] - bi[2])
        inter_res[i * 4 + 3] = cute.arch.exp2(bg[3] - bi[3])

        # Intra scale: zero for rows < 16
        intra_res[i * 4] = cutlass.select_(upper, cute.arch.exp2(bg[0] - bgn[0]), Float32(0.0))
        intra_res[i * 4 + 1] = cutlass.select_(upper, cute.arch.exp2(bg[1] - bgn[1]), Float32(0.0))
        intra_res[i * 4 + 2] = cutlass.select_(upper, cute.arch.exp2(bg[2] - bgn[2]), Float32(0.0))
        intra_res[i * 4 + 3] = cutlass.select_(upper, cute.arch.exp2(bg[3] - bgn[3]), Float32(0.0))

    return (*intra_res, *inter_res)


@cute.jit
def compute_intra_scale_vec(
    buf_G_raw_ptr: cute.Pointer,
    idx_in_warpgroup: Int32,
    k_off: Int32,
):
    """
    Compute intra scale: exp2(g[row] - g[row/16*16]) for rows >= 16.
    Uses vectorized SMEM loads via smem_load_f32x4_sw128.
    Returns 16-tuple of Float32 (zeros for rows 0..15).
    """
    local = idx_in_warpgroup % Int32(64)
    upper = local >= Int32(16)
    intra_ref = (local // Int32(16)) * Int32(16)

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg = smem_load_f32x4_sw128(buf_G_raw_ptr, local, Int32(col))
        bgn = smem_load_f32x4_sw128(buf_G_raw_ptr, intra_ref, Int32(col))
        res[i * 4] = cutlass.select_(upper, cute.arch.exp2(bg[0] - bgn[0]), Float32(0.0))
        res[i * 4 + 1] = cutlass.select_(upper, cute.arch.exp2(bg[1] - bgn[1]), Float32(0.0))
        res[i * 4 + 2] = cutlass.select_(upper, cute.arch.exp2(bg[2] - bgn[2]), Float32(0.0))
        res[i * 4 + 3] = cutlass.select_(upper, cute.arch.exp2(bg[3] - bgn[3]), Float32(0.0))
    return tuple(res)


@cute.jit
def compute_inter_scale_vec(
    buf_G_raw_ptr: cute.Pointer,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
):
    """
    Compute inter scale: exp2(g[row] - g[inter_ref]).
    Uses vectorized SMEM loads via smem_load_f32x4_sw128.
    Returns 16-tuple of Float32.
    """
    local = idx_in_warpgroup % Int32(64)
    inter_ref = cutlass.select_(
        local // Int32(16) * Int32(16) + Int32(8) < sub_seq_len,
        local // Int32(16) * Int32(16) + Int32(8),
        sub_seq_len - Int32(1),
    )

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg = smem_load_f32x4_sw128(buf_G_raw_ptr, local, Int32(col))
        bi = smem_load_f32x4_sw128(buf_G_raw_ptr, inter_ref, Int32(col))
        res[i * 4] = cute.arch.exp2(bg[0] - bi[0])
        res[i * 4 + 1] = cute.arch.exp2(bg[1] - bi[1])
        res[i * 4 + 2] = cute.arch.exp2(bg[2] - bi[2])
        res[i * 4 + 3] = cute.arch.exp2(bg[3] - bi[3])
    return tuple(res)


@cute.jit
def epilogue_compute_dkt_scale_vec(
    buf_G_raw_ptr: cute.Pointer,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
):
    """
    Vectorized dkt scale using smem_load_f32x4_sw128.
    """
    local = idx_in_warpgroup % Int32(64)
    is_lower = idx_in_warpgroup < Int32(64)

    lower_ref = cutlass.select_(
        (local // Int32(16) + Int32(1)) * Int32(16) < sub_seq_len,
        (local // Int32(16) + Int32(1)) * Int32(16),
        sub_seq_len - Int32(1),
    )
    lower_zero = ((local // Int32(16) + Int32(1)) * Int32(16)) >= sub_seq_len

    upper_ref = cutlass.select_(
        local // Int32(16) * Int32(16) + Int32(8) < sub_seq_len,
        local // Int32(16) * Int32(16) + Int32(8),
        sub_seq_len - Int32(1),
    )
    upper_zero = local >= sub_seq_len

    ref_row = cutlass.select_(is_lower, lower_ref, upper_ref)
    should_zero = cutlass.select_(is_lower, lower_zero, upper_zero)

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg_ref = smem_load_f32x4_sw128(buf_G_raw_ptr, ref_row, Int32(col))
        bg = smem_load_f32x4_sw128(buf_G_raw_ptr, local, Int32(col))
        res[i * 4] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref[0] - bg[0]))
        res[i * 4 + 1] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref[1] - bg[1]))
        res[i * 4 + 2] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref[2] - bg[2]))
        res[i * 4 + 3] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref[3] - bg[3]))
    return tuple(res)


# ============================================================
# Apply-only epilogue functions (use precomputed scales)
# ============================================================


@cute.jit
def apply_dq_precomputed(
    idx_in_warpgroup: Int32,
    tmem_dq_addr: Int32,
    tmem_dq2_addr: Int32,
    # intra_scale[16]
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
    s8: Float32,
    s9: Float32,
    s10: Float32,
    s11: Float32,
    s12: Float32,
    s13: Float32,
    s14: Float32,
    s15: Float32,
    # inter_scale[16]
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
    x4: Float32,
    x5: Float32,
    x6: Float32,
    x7: Float32,
    x8: Float32,
    x9: Float32,
    x10: Float32,
    x11: Float32,
    x12: Float32,
    x13: Float32,
    x14: Float32,
    x15: Float32,
):
    """Apply precomputed intra+inter scales to TMEM dq/dq2 results.
    res[i] = select(upper, dq[i] * intra_scale[i], 0) + dq2[i] * inter_scale[i]
    """
    intra = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15)
    inter = (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)

    local = idx_in_warpgroup % Int32(64)
    upper = local >= Int32(16)

    tcgen05_fence_after()
    dq_vals = tmem_ld_x16(tmem_dq_addr)
    dq2_vals = tmem_ld_x16(tmem_dq2_addr)
    cute.arch.fence_view_async_tmem_load()
    tcgen05_fence_before()

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(16):
        # select_ guards against NaN in dq_vals for rows < 16
        intra_contrib = cutlass.select_(upper, dq_vals[i] * intra[i], Float32(0.0))
        res[i] = intra_contrib + dq2_vals[i] * inter[i]
    return tuple(res)


@cute.jit
def apply_dkt_precomputed(
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    tmem_dkt_addr: Int32,
    # dkt_scale[16]
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
    s8: Float32,
    s9: Float32,
    s10: Float32,
    s11: Float32,
    s12: Float32,
    s13: Float32,
    s14: Float32,
    s15: Float32,
):
    """Apply precomputed scale to TMEM dkt results.
    res[i] = select(should_zero, 0, dkt[i] * scale[i])
    """
    scale = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15)

    local = idx_in_warpgroup % Int32(64)
    is_lower = idx_in_warpgroup < Int32(64)
    lower_zero = ((local // Int32(16) + Int32(1)) * Int32(16)) >= sub_seq_len
    upper_zero = local >= sub_seq_len
    should_zero = cutlass.select_(is_lower, lower_zero, upper_zero)

    v = tmem_ld_x16(tmem_dkt_addr)
    cute.arch.fence_view_async_tmem_load()

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(16):
        res[i] = cutlass.select_(should_zero, Float32(0.0), v[i] * scale[i])
    return tuple(res)


# ============================================================
# Fused epilogue functions (register-pressure-optimized)
# ============================================================


@cute.jit
def epilogue_dq_scaled(
    buf_G_raw_ptr: cute.Pointer,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
    tmem_dq_addr: Int32,
    tmem_dq2_addr: Int32,
):
    """
    Merged epilogue: load both TMEM accumulators under one fence pair,
    then process 4 elements at a time (intra mul + inter FMA) to keep
    peak live registers modest while avoiding duplicate SMEM loads.
    Returns 16 float values: dq_intra * intra_scale + dq_inter * inter_scale.
    """
    local = idx_in_warpgroup % Int32(64)
    upper = local >= Int32(16)
    intra_ref = (local // Int32(16)) * Int32(16)
    inter_ref = cutlass.select_(
        local // Int32(16) * Int32(16) + Int32(8) < sub_seq_len,
        local // Int32(16) * Int32(16) + Int32(8),
        sub_seq_len - Int32(1),
    )

    # Load both TMEM accumulators under single fence pair
    tcgen05_fence_after()
    dq_vals = tmem_ld_x16(tmem_dq_addr)
    dq2_vals = tmem_ld_x16(tmem_dq2_addr)
    cute.arch.fence_view_async_tmem_load()
    tcgen05_fence_before()

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg = smem_load_f32x4_sw128(buf_G_raw_ptr, local, Int32(col))
        bgn = smem_load_f32x4_sw128(buf_G_raw_ptr, intra_ref, Int32(col))
        bi = smem_load_f32x4_sw128(buf_G_raw_ptr, inter_ref, Int32(col))

        # intra mul then inter FMA — compiler can fuse to fma(dq2, inter, res)
        res[i * 4] = cutlass.select_(upper, dq_vals[i * 4] * cute.arch.exp2(bg[0] - bgn[0]), Float32(0.0))
        res[i * 4 + 1] = cutlass.select_(upper, dq_vals[i * 4 + 1] * cute.arch.exp2(bg[1] - bgn[1]), Float32(0.0))
        res[i * 4 + 2] = cutlass.select_(upper, dq_vals[i * 4 + 2] * cute.arch.exp2(bg[2] - bgn[2]), Float32(0.0))
        res[i * 4 + 3] = cutlass.select_(upper, dq_vals[i * 4 + 3] * cute.arch.exp2(bg[3] - bgn[3]), Float32(0.0))

        res[i * 4] = dq2_vals[i * 4] * cute.arch.exp2(bg[0] - bi[0]) + res[i * 4]
        res[i * 4 + 1] = dq2_vals[i * 4 + 1] * cute.arch.exp2(bg[1] - bi[1]) + res[i * 4 + 1]
        res[i * 4 + 2] = dq2_vals[i * 4 + 2] * cute.arch.exp2(bg[2] - bi[2]) + res[i * 4 + 2]
        res[i * 4 + 3] = dq2_vals[i * 4 + 3] * cute.arch.exp2(bg[3] - bi[3]) + res[i * 4 + 3]
    return tuple(res)


@cute.jit
def epilogue_dkt_scaled(
    buf_G_raw_ptr: cute.Pointer,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
    tmem_dkt_addr: Int32,
):
    """
    Fused: compute dkt scales inline and apply to TMEM dkt values.
    Replaces epilogue_compute_dkt_scale_vec + epilogue_process_dkt.
    Returns 16 float values: scaled dkt.
    """
    local = idx_in_warpgroup % Int32(64)
    is_lower = idx_in_warpgroup < Int32(64)

    lower_ref = cutlass.select_(
        (local // Int32(16) + Int32(1)) * Int32(16) < sub_seq_len,
        (local // Int32(16) + Int32(1)) * Int32(16),
        sub_seq_len - Int32(1),
    )
    lower_zero = ((local // Int32(16) + Int32(1)) * Int32(16)) >= sub_seq_len

    upper_ref = cutlass.select_(
        local // Int32(16) * Int32(16) + Int32(8) < sub_seq_len,
        local // Int32(16) * Int32(16) + Int32(8),
        sub_seq_len - Int32(1),
    )
    upper_zero = local >= sub_seq_len

    ref_row = cutlass.select_(is_lower, lower_ref, upper_ref)
    should_zero = cutlass.select_(is_lower, lower_zero, upper_zero)

    # Load TMEM dkt (no fence needed — already fenced by dq epilogue)
    v = tmem_ld_x16(tmem_dkt_addr)
    cute.arch.fence_view_async_tmem_load()

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg_ref = smem_load_f32x4_sw128(buf_G_raw_ptr, ref_row, Int32(col))
        bg = smem_load_f32x4_sw128(buf_G_raw_ptr, local, Int32(col))

        sc0 = cute.arch.exp2(bg_ref[0] - bg[0])
        sc1 = cute.arch.exp2(bg_ref[1] - bg[1])
        sc2 = cute.arch.exp2(bg_ref[2] - bg[2])
        sc3 = cute.arch.exp2(bg_ref[3] - bg[3])

        res[i * 4] = cutlass.select_(should_zero, Float32(0.0), v[i * 4] * sc0)
        res[i * 4 + 1] = cutlass.select_(should_zero, Float32(0.0), v[i * 4 + 1] * sc1)
        res[i * 4 + 2] = cutlass.select_(should_zero, Float32(0.0), v[i * 4 + 2] * sc2)
        res[i * 4 + 3] = cutlass.select_(should_zero, Float32(0.0), v[i * 4 + 3] * sc3)
    return tuple(res)


# ============================================================
# Epilogue helper functions
# All use HALF_K=16 elements per warpgroup half.
# ============================================================

HALF_K: int = K_TILE // 2  # = 16, elements per WG per tile


@cute.jit
def epilogue_apply_dq_intra(
    idx_in_warpgroup: Int32,
    tmem_dq_addr: Int32,
    scale_0: Float32,
    scale_1: Float32,
    scale_2: Float32,
    scale_3: Float32,
    scale_4: Float32,
    scale_5: Float32,
    scale_6: Float32,
    scale_7: Float32,
    scale_8: Float32,
    scale_9: Float32,
    scale_10: Float32,
    scale_11: Float32,
    scale_12: Float32,
    scale_13: Float32,
    scale_14: Float32,
    scale_15: Float32,
):
    """
    1) tcgen05_after_thread_sync (non-blocking ordering fence)
    2) Load 16 floats from TMEM[dq]
    3) fence_view_async_tmem_load
    4) tcgen05_before_thread_sync
    5) if idx%64 >= 16: res[i] *= scale[i]  else: res[i] = 0
    Returns 16 Float32 values: (r0..r15)
    """
    tcgen05_fence_after()
    vals = tmem_ld_x16(tmem_dq_addr)
    cute.arch.fence_view_async_tmem_load()
    tcgen05_fence_before()

    local = idx_in_warpgroup % Int32(64)
    upper = local >= Int32(16)

    r0 = cutlass.select_(upper, vals[0] * scale_0, Float32(0.0))
    r1 = cutlass.select_(upper, vals[1] * scale_1, Float32(0.0))
    r2 = cutlass.select_(upper, vals[2] * scale_2, Float32(0.0))
    r3 = cutlass.select_(upper, vals[3] * scale_3, Float32(0.0))
    r4 = cutlass.select_(upper, vals[4] * scale_4, Float32(0.0))
    r5 = cutlass.select_(upper, vals[5] * scale_5, Float32(0.0))
    r6 = cutlass.select_(upper, vals[6] * scale_6, Float32(0.0))
    r7 = cutlass.select_(upper, vals[7] * scale_7, Float32(0.0))
    r8 = cutlass.select_(upper, vals[8] * scale_8, Float32(0.0))
    r9 = cutlass.select_(upper, vals[9] * scale_9, Float32(0.0))
    r10 = cutlass.select_(upper, vals[10] * scale_10, Float32(0.0))
    r11 = cutlass.select_(upper, vals[11] * scale_11, Float32(0.0))
    r12 = cutlass.select_(upper, vals[12] * scale_12, Float32(0.0))
    r13 = cutlass.select_(upper, vals[13] * scale_13, Float32(0.0))
    r14 = cutlass.select_(upper, vals[14] * scale_14, Float32(0.0))
    r15 = cutlass.select_(upper, vals[15] * scale_15, Float32(0.0))
    return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)


@cute.jit
def epilogue_combine_dq_inter(
    tmem_dq2_addr: Int32,
    r0: Float32,
    r1: Float32,
    r2: Float32,
    r3: Float32,
    r4: Float32,
    r5: Float32,
    r6: Float32,
    r7: Float32,
    r8: Float32,
    r9: Float32,
    r10: Float32,
    r11: Float32,
    r12: Float32,
    r13: Float32,
    r14: Float32,
    r15: Float32,
    sc0: Float32,
    sc1: Float32,
    sc2: Float32,
    sc3: Float32,
    sc4: Float32,
    sc5: Float32,
    sc6: Float32,
    sc7: Float32,
    sc8: Float32,
    sc9: Float32,
    sc10: Float32,
    sc11: Float32,
    sc12: Float32,
    sc13: Float32,
    sc14: Float32,
    sc15: Float32,
):
    """res[i] += res2[i] * inter_scale[i]   (fma)"""
    tcgen05_fence_after()
    v = tmem_ld_x16(tmem_dq2_addr)
    cute.arch.fence_view_async_tmem_load()
    tcgen05_fence_before()

    o0 = v[0] * sc0 + r0
    o1 = v[1] * sc1 + r1
    o2 = v[2] * sc2 + r2
    o3 = v[3] * sc3 + r3
    o4 = v[4] * sc4 + r4
    o5 = v[5] * sc5 + r5
    o6 = v[6] * sc6 + r6
    o7 = v[7] * sc7 + r7
    o8 = v[8] * sc8 + r8
    o9 = v[9] * sc9 + r9
    o10 = v[10] * sc10 + r10
    o11 = v[11] * sc11 + r11
    o12 = v[12] * sc12 + r12
    o13 = v[13] * sc13 + r13
    o14 = v[14] * sc14 + r14
    o15 = v[15] * sc15 + r15
    return (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)


@cute.jit
def epilogue_compute_intra_scale(
    buf_G,
    idx_in_warpgroup: Int32,
    k_off: Int32,
):
    """
    Compute intra scale: exp2(g[row] - g[row/16*16]) for rows >= 16.
    k_off = WG_IDX * HALF_K — column offset into buf_G.
    Returns 16-tuple of Float32 (zeros for rows 0..15).
    """
    local = idx_in_warpgroup % Int32(64)
    upper = local >= Int32(16)
    intra_ref = (local // Int32(16)) * Int32(16)

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg0 = buf_G[(local, Int32(col))]
        bg1 = buf_G[(local, Int32(col + 1))]
        bg2 = buf_G[(local, Int32(col + 2))]
        bg3 = buf_G[(local, Int32(col + 3))]
        bgn0 = buf_G[(intra_ref, Int32(col))]
        bgn1 = buf_G[(intra_ref, Int32(col + 1))]
        bgn2 = buf_G[(intra_ref, Int32(col + 2))]
        bgn3 = buf_G[(intra_ref, Int32(col + 3))]
        res[i * 4] = cutlass.select_(upper, cute.arch.exp2(bg0 - bgn0), Float32(0.0))
        res[i * 4 + 1] = cutlass.select_(upper, cute.arch.exp2(bg1 - bgn1), Float32(0.0))
        res[i * 4 + 2] = cutlass.select_(upper, cute.arch.exp2(bg2 - bgn2), Float32(0.0))
        res[i * 4 + 3] = cutlass.select_(upper, cute.arch.exp2(bg3 - bgn3), Float32(0.0))
    return tuple(res)


@cute.jit
def epilogue_compute_inter_scale(
    buf_G,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
):
    """
    Compute inter scale: exp2(g[row] - g[inter_ref]).
    inter_ref = min(row/16*16+8, sub_seq_len-1)
    Returns 16-tuple of Float32.
    """
    local = idx_in_warpgroup % Int32(64)
    inter_ref = cutlass.select_(
        local // Int32(16) * Int32(16) + Int32(8) < sub_seq_len,
        local // Int32(16) * Int32(16) + Int32(8),
        sub_seq_len - Int32(1),
    )
    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg0 = buf_G[(local, Int32(col))]
        bg1 = buf_G[(local, Int32(col + 1))]
        bg2 = buf_G[(local, Int32(col + 2))]
        bg3 = buf_G[(local, Int32(col + 3))]
        bi0 = buf_G[(inter_ref, Int32(col))]
        bi1 = buf_G[(inter_ref, Int32(col + 1))]
        bi2 = buf_G[(inter_ref, Int32(col + 2))]
        bi3 = buf_G[(inter_ref, Int32(col + 3))]
        res[i * 4] = cute.arch.exp2(bg0 - bi0)
        res[i * 4 + 1] = cute.arch.exp2(bg1 - bi1)
        res[i * 4 + 2] = cute.arch.exp2(bg2 - bi2)
        res[i * 4 + 3] = cute.arch.exp2(bg3 - bi3)
    return tuple(res)


@cute.jit
def epilogue_compute_dkt_scale(
    buf_G,
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
):
    """
    Lower half (idx<64): scale = exp2(g[next_block_start] - g[row])
                         zero if (row/16+1)*16 >= sub_seq_len
    Upper half (idx>=64): scale = exp2(g[half_row] - g[row])
                          zero if row >= sub_seq_len
    Returns 16-tuple Float32.
    """
    local = idx_in_warpgroup % Int32(64)
    is_lower = idx_in_warpgroup < Int32(64)

    # Lower: ref_row = min((row/16+1)*16, sub_seq_len-1)
    lower_ref = cutlass.select_(
        (local // Int32(16) + Int32(1)) * Int32(16) < sub_seq_len,
        (local // Int32(16) + Int32(1)) * Int32(16),
        sub_seq_len - Int32(1),
    )
    # Lower: zero if (row/16+1)*16 >= sub_seq_len
    lower_zero = ((local // Int32(16) + Int32(1)) * Int32(16)) >= sub_seq_len

    # Upper: ref_row = min(row/16*16+8, sub_seq_len-1)
    upper_ref = cutlass.select_(
        local // Int32(16) * Int32(16) + Int32(8) < sub_seq_len,
        local // Int32(16) * Int32(16) + Int32(8),
        sub_seq_len - Int32(1),
    )
    upper_zero = local >= sub_seq_len

    ref_row = cutlass.select_(is_lower, lower_ref, upper_ref)
    should_zero = cutlass.select_(is_lower, lower_zero, upper_zero)

    res = [Float32(0.0)] * 16
    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        bg_ref0 = buf_G[(ref_row, Int32(col))]
        bg_ref1 = buf_G[(ref_row, Int32(col + 1))]
        bg_ref2 = buf_G[(ref_row, Int32(col + 2))]
        bg_ref3 = buf_G[(ref_row, Int32(col + 3))]
        bg0 = buf_G[(local, Int32(col))]
        bg1 = buf_G[(local, Int32(col + 1))]
        bg2 = buf_G[(local, Int32(col + 2))]
        bg3 = buf_G[(local, Int32(col + 3))]
        res[i * 4] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref0 - bg0))
        res[i * 4 + 1] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref1 - bg1))
        res[i * 4 + 2] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref2 - bg2))
        res[i * 4 + 3] = cutlass.select_(should_zero, Float32(0.0), cute.arch.exp2(bg_ref3 - bg3))
    return tuple(res)


@cute.jit
def epilogue_process_dkt(
    idx_in_warpgroup: Int32,
    tmem_dkt_addr: Int32,
    sub_seq_len: Int32,
    sc0: Float32,
    sc1: Float32,
    sc2: Float32,
    sc3: Float32,
    sc4: Float32,
    sc5: Float32,
    sc6: Float32,
    sc7: Float32,
    sc8: Float32,
    sc9: Float32,
    sc10: Float32,
    sc11: Float32,
    sc12: Float32,
    sc13: Float32,
    sc14: Float32,
    sc15: Float32,
):
    """Load TMEM dkt, multiply by scale (or zero for invalid rows)."""
    v = tmem_ld_x16(tmem_dkt_addr)
    cute.arch.fence_view_async_tmem_load()

    local = idx_in_warpgroup % Int32(64)
    is_lower = idx_in_warpgroup < Int32(64)
    lower_zero = ((local // Int32(16) + Int32(1)) * Int32(16)) >= sub_seq_len
    upper_zero = local >= sub_seq_len
    should_zero = cutlass.select_(is_lower, lower_zero, upper_zero)

    def _e(vi, si):
        return cutlass.select_(should_zero, Float32(0.0), vi * si)

    r0 = _e(v[0], sc0)
    r1 = _e(v[1], sc1)
    r2 = _e(v[2], sc2)
    r3 = _e(v[3], sc3)
    r4 = _e(v[4], sc4)
    r5 = _e(v[5], sc5)
    r6 = _e(v[6], sc6)
    r7 = _e(v[7], sc7)
    r8 = _e(v[8], sc8)
    r9 = _e(v[9], sc9)
    r10 = _e(v[10], sc10)
    r11 = _e(v[11], sc11)
    r12 = _e(v[12], sc12)
    r13 = _e(v[13], sc13)
    r14 = _e(v[14], sc14)
    r15 = _e(v[15], sc15)
    return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)


@cute.jit
def epilogue_output_dq(
    buf_Q_raw: cute.Pointer,  # bf16 SMEM raw pointer (sw64)
    buf_DQ_raw: cute.Pointer,  # f32 SMEM raw pointer (sw128)
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
    dq_out_base_ptr: cute.Pointer,  # bf16 pointer into output buffer
    # res[16] via 16 individual args (also updated output for dg prep: res *= q)
    r0: Float32,
    r1: Float32,
    r2: Float32,
    r3: Float32,
    r4: Float32,
    r5: Float32,
    r6: Float32,
    r7: Float32,
    r8: Float32,
    r9: Float32,
    r10: Float32,
    r11: Float32,
    r12: Float32,
    r13: Float32,
    r14: Float32,
    r15: Float32,
):
    """
    For rows idx%64 < sub_seq_len:
      - Load dq_prev from sDQ, add res, write bf16 to dq_out
      - Multiply res by q (for dg path) → returned updated res
    """
    local = idx_in_warpgroup % Int32(64)

    # Write dq and prepare res*=q
    new_r = [Float32(0.0)] * 16
    res_raw = (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)

    # Create rmem fragment for bf16 dq output (populated inside loop, stored after)
    gmem_ptr = cute.make_ptr(
        BFloat16,
        dq_out_base_ptr.toint(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    gmem_t = cute.make_tensor(gmem_ptr, cute.make_layout((16,), stride=(1,)))
    rmem_t = cute.make_fragment_like(gmem_t)

    if local < sub_seq_len:
        for i in cutlass.range_constexpr(4):
            base = k_off + i * 4
            # Vectorized load dq_prev (f32, sw128) + q (bf16, sw64)
            dqp = smem_load_f32x4_sw128(buf_DQ_raw, local, base)
            q = smem_load_bf16x4_sw64(buf_Q_raw, local, base)

            # Compute dq_out = (dq_prev + res) as bf16 → store to rmem fragment
            rmem_t[(i * 4,)] = f32_to_bf16(dqp[0] + res_raw[i * 4])
            rmem_t[(i * 4 + 1,)] = f32_to_bf16(dqp[1] + res_raw[i * 4 + 1])
            rmem_t[(i * 4 + 2,)] = f32_to_bf16(dqp[2] + res_raw[i * 4 + 2])
            rmem_t[(i * 4 + 3,)] = f32_to_bf16(dqp[3] + res_raw[i * 4 + 3])

            # Update res for dg path: res *= q
            new_r[i * 4] = res_raw[i * 4] * bf16_to_f32(q[0])
            new_r[i * 4 + 1] = res_raw[i * 4 + 1] * bf16_to_f32(q[1])
            new_r[i * 4 + 2] = res_raw[i * 4 + 2] * bf16_to_f32(q[2])
            new_r[i * 4 + 3] = res_raw[i * 4 + 3] * bf16_to_f32(q[3])

        # Vectorized GMEM store via autovec_copy
        cute.autovec_copy(rmem_t, gmem_t)

    return tuple(new_r)


@cute.jit
def epilogue_accumulate_db(
    buf_K_raw: cute.Pointer,  # bf16 SMEM raw pointer (sw64)
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
    beta_val: Float32,
    db_in: Float32,
    r0: Float32,
    r1: Float32,
    r2: Float32,
    r3: Float32,
    r4: Float32,
    r5: Float32,
    r6: Float32,
    r7: Float32,
    r8: Float32,
    r9: Float32,
    r10: Float32,
    r11: Float32,
    r12: Float32,
    r13: Float32,
    r14: Float32,
    r15: Float32,
):
    """
    Accumulate db += dot(k, res), then scale res by beta (for dk path).
    Returns (new_db, scaled_res[16]).
    """
    local = idx_in_warpgroup % Int32(64)
    res_raw = (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)

    db = db_in
    new_r = list(res_raw)

    for i in cutlass.range_constexpr(4):
        base = k_off + i * 4
        kv = smem_load_bf16x4_sw64(buf_K_raw, local, base)
        k0 = bf16_to_f32(kv[0])
        k1 = bf16_to_f32(kv[1])
        k2 = bf16_to_f32(kv[2])
        k3 = bf16_to_f32(kv[3])

        db = db + k0 * res_raw[i * 4]
        db = db + k1 * res_raw[i * 4 + 1]
        db = db + k2 * res_raw[i * 4 + 2]
        db = db + k3 * res_raw[i * 4 + 3]

        # Scale res by beta for dk path
        new_r[i * 4] = res_raw[i * 4] * beta_val
        new_r[i * 4 + 1] = res_raw[i * 4 + 1] * beta_val
        new_r[i * 4 + 2] = res_raw[i * 4 + 2] * beta_val
        new_r[i * 4 + 3] = res_raw[i * 4 + 3] * beta_val

    return (db, *new_r)


@cute.jit
def epilogue_exchange_dkt(
    buf_DKT0_raw: cute.Pointer,  # f32 SMEM raw pointer for sDKT_0 (sw128, stride=K_TILE)
    buf_DKT1_raw: cute.Pointer,  # f32 SMEM raw pointer for sDKT_1 (sw128, stride=K_TILE)
    idx_in_warpgroup: Int32,
    k_off: Int32,
    # res[16]
    r0: Float32,
    r1: Float32,
    r2: Float32,
    r3: Float32,
    r4: Float32,
    r5: Float32,
    r6: Float32,
    r7: Float32,
    r8: Float32,
    r9: Float32,
    r10: Float32,
    r11: Float32,
    r12: Float32,
    r13: Float32,
    r14: Float32,
    r15: Float32,
    # res_dkt[16]
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    d4: Float32,
    d5: Float32,
    d6: Float32,
    d7: Float32,
    d8: Float32,
    d9: Float32,
    d10: Float32,
    d11: Float32,
    d12: Float32,
    d13: Float32,
    d14: Float32,
    d15: Float32,
):
    """
    Lower half (idx<64): write res_dkt to sDKT_0
    Upper half (idx>=64): write (res - res_dkt) to sDKT_1
    """
    local = idx_in_warpgroup % Int32(64)
    is_lower = idx_in_warpgroup < Int32(64)

    dkt_vals = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15)
    res_vals = (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)

    for i in cutlass.range_constexpr(4):
        col = k_off + i * 4
        if is_lower:
            # Lower writes res_dkt to sDKT_0 (vectorized STS.128 with swizzle)
            smem_store_f32x4_sw128(
                buf_DKT0_raw, local, col, dkt_vals[i * 4], dkt_vals[i * 4 + 1], dkt_vals[i * 4 + 2], dkt_vals[i * 4 + 3]
            )
        else:
            # Upper writes (res - res_dkt) to sDKT_1 (vectorized STS.128 with swizzle)
            smem_store_f32x4_sw128(
                buf_DKT1_raw,
                local,
                col,
                res_vals[i * 4] - dkt_vals[i * 4],
                res_vals[i * 4 + 1] - dkt_vals[i * 4 + 1],
                res_vals[i * 4 + 2] - dkt_vals[i * 4 + 2],
                res_vals[i * 4 + 3] - dkt_vals[i * 4 + 3],
            )


@cute.jit
def epilogue_output_dg(
    buf_K_raw: cute.Pointer,  # bf16 SMEM raw pointer (sw64)
    buf_DG_raw: cute.Pointer,  # f32 SMEM raw pointer (sw128)
    buf_DKT1_raw: cute.Pointer,  # f32 SMEM raw pointer (sw128, stride=K_TILE)
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
    dg_out_base_ptr: cute.Pointer,  # f32 output pointer
    # res (=res*q after epilogue_output_dq)
    r0: Float32,
    r1: Float32,
    r2: Float32,
    r3: Float32,
    r4: Float32,
    r5: Float32,
    r6: Float32,
    r7: Float32,
    r8: Float32,
    r9: Float32,
    r10: Float32,
    r11: Float32,
    r12: Float32,
    r13: Float32,
    r14: Float32,
    r15: Float32,
    # res_dkt
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    d4: Float32,
    d5: Float32,
    d6: Float32,
    d7: Float32,
    d8: Float32,
    d9: Float32,
    d10: Float32,
    d11: Float32,
    d12: Float32,
    d13: Float32,
    d14: Float32,
    d15: Float32,
):
    """
    dg = res + (dk_sub_dkt - res_dkt) * k + dg_prev
    Output to dg_out_base_ptr[0..15] as float32.
    Only for local rows < sub_seq_len.
    """
    local = idx_in_warpgroup % Int32(64)
    res_raw = (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
    dkt_raw = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15)

    # Create rmem fragment for f32 dg output (tensor setup outside control flow)
    gmem_ptr = cute.make_ptr(
        Float32,
        dg_out_base_ptr.toint(),
        cute.AddressSpace.gmem,
        assumed_align=64,
    )
    gmem_t = cute.make_tensor(gmem_ptr, cute.make_layout((16,), stride=(1,)))
    rmem_t = cute.make_fragment_like(gmem_t)

    if local < sub_seq_len:
        for i in cutlass.range_constexpr(4):
            base = k_off + i * 4
            dk_sub = smem_load_f32x4_sw128(buf_DKT1_raw, local, base)
            kv = smem_load_bf16x4_sw64(buf_K_raw, local, base)
            dg_p = smem_load_f32x4_sw128(buf_DG_raw, local, base)

            # diff = dk_sub_dkt - res_dkt
            diff0 = dk_sub[0] - dkt_raw[i * 4]
            diff1 = dk_sub[1] - dkt_raw[i * 4 + 1]
            diff2 = dk_sub[2] - dkt_raw[i * 4 + 2]
            diff3 = dk_sub[3] - dkt_raw[i * 4 + 3]

            # out = (res + dg_prev) + diff * k → store to rmem fragment
            rmem_t[(i * 4,)] = (res_raw[i * 4] + dg_p[0]) + diff0 * bf16_to_f32(kv[0])
            rmem_t[(i * 4 + 1,)] = (res_raw[i * 4 + 1] + dg_p[1]) + diff1 * bf16_to_f32(kv[1])
            rmem_t[(i * 4 + 2,)] = (res_raw[i * 4 + 2] + dg_p[2]) + diff2 * bf16_to_f32(kv[2])
            rmem_t[(i * 4 + 3,)] = (res_raw[i * 4 + 3] + dg_p[3]) + diff3 * bf16_to_f32(kv[3])

        # Vectorized GMEM store via autovec_copy
        cute.autovec_copy(rmem_t, gmem_t)


@cute.jit
def epilogue_output_dk(
    buf_DK_raw: cute.Pointer,  # f32 SMEM raw pointer (sw128)
    buf_DKT0_raw: cute.Pointer,  # f32 SMEM raw pointer (sw128, stride=K_TILE)
    idx_in_warpgroup: Int32,
    sub_seq_len: Int32,
    k_off: Int32,
    dk_out_base_ptr: cute.Pointer,  # bf16 output pointer
    # res (already beta-scaled dq for dk path)
    r0: Float32,
    r1: Float32,
    r2: Float32,
    r3: Float32,
    r4: Float32,
    r5: Float32,
    r6: Float32,
    r7: Float32,
    r8: Float32,
    r9: Float32,
    r10: Float32,
    r11: Float32,
    r12: Float32,
    r13: Float32,
    r14: Float32,
    r15: Float32,
    # res_dkt
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    d4: Float32,
    d5: Float32,
    d6: Float32,
    d7: Float32,
    d8: Float32,
    d9: Float32,
    d10: Float32,
    d11: Float32,
    d12: Float32,
    d13: Float32,
    d14: Float32,
    d15: Float32,
):
    """
    dk_out = dk_prev + res_dkt + dkt_sub + res  (= beta-scaled dq)
    Only for rows < sub_seq_len.
    """
    local = idx_in_warpgroup % Int32(64)
    res_raw = (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
    dkt_raw = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15)

    # Create rmem fragment for bf16 dk output (tensor setup outside control flow)
    gmem_ptr = cute.make_ptr(
        BFloat16,
        dk_out_base_ptr.toint(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    gmem_t = cute.make_tensor(gmem_ptr, cute.make_layout((16,), stride=(1,)))
    rmem_t = cute.make_fragment_like(gmem_t)

    if local < sub_seq_len:
        for i in cutlass.range_constexpr(4):
            base = k_off + i * 4
            dkt_sub = smem_load_f32x4_sw128(buf_DKT0_raw, local, base)
            dk_p = smem_load_f32x4_sw128(buf_DK_raw, local, base)

            # result = res + res_dkt + dkt_sub + dk_prev → store as bf16 to rmem fragment
            rmem_t[(i * 4,)] = f32_to_bf16(res_raw[i * 4] + dkt_raw[i * 4] + dkt_sub[0] + dk_p[0])
            rmem_t[(i * 4 + 1,)] = f32_to_bf16(res_raw[i * 4 + 1] + dkt_raw[i * 4 + 1] + dkt_sub[1] + dk_p[1])
            rmem_t[(i * 4 + 2,)] = f32_to_bf16(res_raw[i * 4 + 2] + dkt_raw[i * 4 + 2] + dkt_sub[2] + dk_p[2])
            rmem_t[(i * 4 + 3,)] = f32_to_bf16(res_raw[i * 4 + 3] + dkt_raw[i * 4 + 3] + dkt_sub[3] + dk_p[3])

        # Vectorized GMEM store via autovec_copy
        cute.autovec_copy(rmem_t, gmem_t)


# ============================================================
# ComputeEpilogue warpgroup body
# ============================================================


@cute.jit
def compute_epilogue_body(
    # SMEM tensors
    buf_Q_buf,  # [NUM_BUF_VALUE, T_TILE, K_TILE] bf16
    buf_K_buf,  # same
    buf_G_buf,  # [NUM_BUF_VALUE, T_TILE, K_TILE] f32
    buf_KG_in,  # [K_TILE, 6*SUB_T_TILE] TF32 (all 6 intra slots)
    buf_KG_ex,  # [K_TILE, 4*SUB_T_TILE] TF32 (all 4 inter slots)
    buf_QKG_in,  # [K_TILE, 6*2*SUB_T_TILE] TF32
    buf_QKG_ex,  # [K_TILE, 4*2*SUB_T_TILE] TF32
    buf_DAqk,  # [NUM_BUF_A, T_TILE, T_TILE] f32
    buf_DAkk,  # [NUM_BUF_A, T_TILE, T_TILE] f32
    buf_DQ_buf,  # [NUM_BUF_VALUE, T_TILE, K_TILE] f32
    buf_DK_buf,  # same
    buf_DG_buf,  # same
    buf_DKT0_buf,  # [NUM_BUF_VALUE, T_TILE, K_TILE] f32 (b_k_neg_exp)
    buf_DKT1_buf,  # [NUM_BUF_VALUE, T_TILE, K_TILE] f32 (b_k_exp)
    buf_Beta,  # [2, T_TILE] f32 (buf_beta double-buffered by A_phase)
    buf_DBpart,  # [2, T_TILE] f32 (buf_dbpart, indexed by 0)
    # mbarrier pointers (raw Int64 SMEM pointers)
    mbar_kg_tma_ready_ptr,  # [NUM_BUF_VALUE]
    mbar_qb_tma_ptr,  # [NUM_BUF_VALUE]
    mbar_dkg_tma_ready_ptr,  # [NUM_BUF_VALUE]
    mbar_kg_done_ready_ptr,  # single
    mbar_qkg_done_ready_ptr,  # single
    mbar_dq_fin_ptr,  # single
    mbar_dkt_fin_ptr,  # single
    mbar_dA_rdy_ptr,  # [NUM_BUF_A]
    mbar_dAt_rdy_ptr,  # [NUM_BUF_A]
    mbar_val_free_ptr,  # [NUM_BUF_VALUE]
    # Output GMEM pointers
    dq_out_ptr: cute.Pointer,  # bf16
    dk_out_ptr: cute.Pointer,  # bf16
    dg_out_ptr: cute.Pointer,  # f32
    db_out_ptr: cute.Pointer,  # f32
    db_ptr: cute.Pointer,  # f32 (previous db)
    # Tile parameters
    start_offset: Int32,
    tile_idx: Int32,
    head_idx: Int32,
    sub_seq_len: Int32,
    h_param: Int32,
    k_size_param: Int32,
    beta_buf: Int32,  # = A_phase
    # TMEM base address (retrieved from SMEM hold register)
    tmem_base: Int32,
    # Mutable state (pre-initialized by caller)
    state_phase: Int32,
    buf_idx_A: Int32,
    buf_idx_value: Int32,
    # WG index (compile-time)
    wg_idx: cutlass.Constexpr,
):
    """Per-tile Epilogue body for one warpgroup (wg_idx=0 or 1)."""
    K_OFF: cutlass.Constexpr = wg_idx * HALF_K
    DKT_BAR_ID: cutlass.Constexpr = wg_idx * 2

    idx = cute.arch.thread_idx()[0] % Int32(128)
    local = idx % Int32(64)
    b_phase = Int32(0)

    # Only WG1 loads initial db to avoid double-counting
    db = Float32(0.0)
    if cutlass.const_expr(wg_idx == 1):
        if (idx >= Int32(64)) & (local < sub_seq_len):
            token_row = start_offset + tile_idx * Int32(T_TILE) + local
            flat_idx = token_row * h_param + head_idx
            db = db_ptr[flat_idx]

    # Pre-loop: mask_A
    sDA_cur = buf_DAqk[(buf_idx_A, None, None)] if idx < Int32(64) else buf_DAkk[(buf_idx_A, None, None)]
    mask_A_tensor(
        sDA_cur,
        idx,
        sub_seq_len,
        tmem_base + Int32(DAQK_02),
        wg_idx * 32,
    )
    cute.arch.fence_view_async_tmem_store()
    tcgen05_fence_before()
    cute.arch.mbarrier_arrive(mbar_dA_rdy_ptr + buf_idx_A)

    # Wait for beta
    cute.arch.mbarrier_wait(buf_Beta.iterator, b_phase)  # mbar_mask_rdy

    # Convenience: beta buffer for this tile
    beta_tile = buf_Beta[(beta_buf, None)]

    # ===== K-ITERATION LOOP =====
    for k_idx in cutlass.range(K_ITERATION, unroll_full=False):
        local_phase = (state_phase >> buf_idx_value) & Int32(1)

        # --- Wait for K/G ---
        cute.arch.mbarrier_wait(mbar_kg_tma_ready_ptr + buf_idx_value, local_phase)
        buf_K = buf_K_buf[(buf_idx_value, None, None)]
        buf_G = buf_G_buf[(buf_idx_value, None, None)]

        y = idx % Int32(8) * Int32(4)

        # --- kg_intra (pre-Q phase) ---
        if cutlass.const_expr(wg_idx == 0):
            gn3_0 = buf_G[(Int32(48), y)]
            gn3_1 = buf_G[(Int32(48), y + Int32(1))]
            gn3_2 = buf_G[(Int32(48), y + Int32(2))]
            gn3_3 = buf_G[(Int32(48), y + Int32(3))]
            setup_kg_intra(buf_G, buf_K, buf_KG_in, 0, idx, gn3_0, gn3_1, gn3_2, gn3_3, 3)
        else:
            gn1_0 = buf_G[(Int32(16), y)]
            gn1_1 = buf_G[(Int32(16), y + Int32(1))]
            gn1_2 = buf_G[(Int32(16), y + Int32(2))]
            gn1_3 = buf_G[(Int32(16), y + Int32(3))]
            gn2_0 = buf_G[(Int32(32), y)]
            gn2_1 = buf_G[(Int32(32), y + Int32(1))]
            gn2_2 = buf_G[(Int32(32), y + Int32(2))]
            gn2_3 = buf_G[(Int32(32), y + Int32(3))]
            setup_kg_intra_2gn(buf_G, buf_K, buf_KG_in, 0, idx, gn1_0, gn1_1, gn1_2, gn1_3, gn2_0, gn2_1, gn2_2, gn2_3, 0, 1)
            setup_kg_intra(buf_G, buf_K, buf_KG_in, 1, idx, gn2_0, gn2_1, gn2_2, gn2_3, 2)

        # --- Wait for Q ---
        cute.arch.mbarrier_wait(mbar_qb_tma_ptr + buf_idx_value, local_phase)
        buf_Q = buf_Q_buf[(buf_idx_value, None, None)]
        sDQ = buf_DQ_buf[(buf_idx_value, None, None)]

        # Load beta values
        beta_base = idx // Int32(8)
        beta1 = Float32(0.0)
        beta2 = Float32(0.0)
        beta3 = Float32(0.0)
        if beta_base + Int32(16) < sub_seq_len:
            beta1 = beta_tile[beta_base + Int32(16)]
        if beta_base + Int32(32) < sub_seq_len:
            beta2 = beta_tile[beta_base + Int32(32)]
        if beta_base + Int32(48) < sub_seq_len:
            beta3 = beta_tile[beta_base + Int32(48)]

        # --- Fused kg_intra + qkg_intra ---
        if cutlass.const_expr(wg_idx == 0):
            gn3_0 = buf_G[(Int32(48), y)]
            gn3_1 = buf_G[(Int32(48), y + Int32(1))]
            gn3_2 = buf_G[(Int32(48), y + Int32(2))]
            gn3_3 = buf_G[(Int32(48), y + Int32(3))]
            gn1_0 = buf_G[(Int32(16), y)]
            gn1_1 = buf_G[(Int32(16), y + Int32(1))]
            gn1_2 = buf_G[(Int32(16), y + Int32(2))]
            gn1_3 = buf_G[(Int32(16), y + Int32(3))]
            # tile_j=1: kg uses gn3, qkg uses gn1
            setup_intra_fused(
                buf_G,
                buf_K,
                buf_Q,
                buf_KG_in,
                buf_QKG_in,
                1,
                idx,
                sub_seq_len,
                gn3_0,
                gn3_1,
                gn3_2,
                gn3_3,
                gn1_0,
                gn1_1,
                gn1_2,
                gn1_3,
                beta1,
                beta1,
                4,
                0,
            )
            # tile_j=2
            setup_intra_fused(
                buf_G,
                buf_K,
                buf_Q,
                buf_KG_in,
                buf_QKG_in,
                2,
                idx,
                sub_seq_len,
                gn3_0,
                gn3_1,
                gn3_2,
                gn3_3,
                gn1_0,
                gn1_1,
                gn1_2,
                gn1_3,
                beta2,
                beta2,
                5,
                1,
            )
        # WG1 has no fused intra (tile_j=1,2 are shared with WG0 in kg; WG1 does tile_j=0,1,3 only)

        # --- Fused kg_inter + qkg_inter ---
        if cutlass.const_expr(wg_idx == 0):
            beta0 = Float32(0.0)
            if beta_base < sub_seq_len:
                beta0 = beta_tile[beta_base]
            setup_inter_fused(buf_G, buf_K, buf_Q, buf_KG_ex, buf_QKG_ex, 0, idx, sub_seq_len, beta0, beta0)
            setup_inter_fused(buf_G, buf_K, buf_Q, buf_KG_ex, buf_QKG_ex, 3, idx, sub_seq_len, beta3, beta3)
        else:
            setup_inter_fused(buf_G, buf_K, buf_Q, buf_KG_ex, buf_QKG_ex, 1, idx, sub_seq_len, beta1, beta1)
            setup_inter_fused(buf_G, buf_K, buf_Q, buf_KG_ex, buf_QKG_ex, 2, idx, sub_seq_len, beta2, beta2)

        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(mbar_kg_done_ready_ptr)

        # --- mask_At (first k_idx only) ---
        if k_idx == Int32(0):
            mask_At_tensor(
                buf_DAqk[(buf_idx_A, None, None)],
                buf_DAkk[(buf_idx_A, None, None)],
                idx,
                sub_seq_len,
                tmem_base + Int32(DAQK_T_02),
                wg_idx * 64,
            )
            tcgen05_fence_before()
            cute.arch.mbarrier_arrive(mbar_dAt_rdy_ptr + buf_idx_A)

        # --- Precompute intra + inter scales (overlap with MMA kg phase) ---
        row = idx % Int32(64)
        intra_sc = epilogue_compute_intra_scale(buf_G, idx, K_OFF)
        inter_sc = epilogue_compute_inter_scale(buf_G, idx, sub_seq_len, K_OFF)

        # --- qkg_intra (non-overlapping rows) ---
        if cutlass.const_expr(wg_idx == 0):
            gn1_0 = buf_G[(Int32(16), y)]
            gn1_1 = buf_G[(Int32(16), y + Int32(1))]
            gn1_2 = buf_G[(Int32(16), y + Int32(2))]
            gn1_3 = buf_G[(Int32(16), y + Int32(3))]
            setup_qkg_intra(buf_G, buf_Q, buf_K, buf_QKG_in, 3, idx, sub_seq_len, beta3, beta3, gn1_0, gn1_1, gn1_2, gn1_3, 2)
        else:
            gn2_0 = buf_G[(Int32(32), y)]
            gn2_1 = buf_G[(Int32(32), y + Int32(1))]
            gn2_2 = buf_G[(Int32(32), y + Int32(2))]
            gn2_3 = buf_G[(Int32(32), y + Int32(3))]
            gn3_0 = buf_G[(Int32(48), y)]
            gn3_1 = buf_G[(Int32(48), y + Int32(1))]
            gn3_2 = buf_G[(Int32(48), y + Int32(2))]
            gn3_3 = buf_G[(Int32(48), y + Int32(3))]
            setup_qkg_intra(buf_G, buf_Q, buf_K, buf_QKG_in, 2, idx, sub_seq_len, beta2, beta2, gn2_0, gn2_1, gn2_2, gn2_3, 3)
            setup_qkg_intra_2gn(
                buf_G,
                buf_Q,
                buf_K,
                buf_QKG_in,
                3,
                idx,
                sub_seq_len,
                beta3,
                beta3,
                gn2_0,
                gn2_1,
                gn2_2,
                gn2_3,
                gn3_0,
                gn3_1,
                gn3_2,
                gn3_3,
                4,
                5,
            )

        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(mbar_qkg_done_ready_ptr)

        # --- Wait dq results from MMA ---
        cute.arch.mbarrier_wait(mbar_dq_fin_ptr, b_phase)

        tmem_dq_base = tmem_base + Int32(DQ_02) + Int32(K_OFF) + Int32(256) * buf_idx_value
        res = epilogue_apply_dq_intra(
            idx,
            tmem_dq_base,
            *intra_sc,
        )

        tmem_dq2_base = tmem_base + Int32(DQ2_02) + Int32(K_OFF) + Int32(256) * buf_idx_value
        res = epilogue_combine_dq_inter(
            tmem_dq2_base,
            *res,
            *inter_sc,
        )

        # Compute dkt scale (reuses intra_sc storage)
        dkt_sc = epilogue_compute_dkt_scale(buf_G, idx, sub_seq_len, K_OFF)

        # --- Output dq / accumulate db ---
        if idx >= Int32(64):
            beta_val = Float32(0.0)
            if local < sub_seq_len:
                beta_val = beta_tile[local]
            result_db = epilogue_accumulate_db(buf_K, idx, sub_seq_len, K_OFF, beta_val, db, *res)
            db = result_db[0]
            res = result_db[1:]
        else:
            token_row = start_offset + tile_idx * Int32(T_TILE) + local
            dq_stride = h_param * Int32(K_SIZE)
            dq_ptr_base = dq_out_ptr + (
                token_row * dq_stride + head_idx * Int32(K_SIZE) + Int32(k_idx) * Int32(K_TILE) + Int32(K_OFF)
            )
            res = epilogue_output_dq(buf_Q, sDQ, idx, sub_seq_len, K_OFF, dq_ptr_base, *res)

        # --- Wait dkt results from MMA ---
        cute.arch.mbarrier_wait(mbar_dkt_fin_ptr, b_phase)

        tmem_dkt_base = tmem_base + Int32(DKT_02) + Int32(K_OFF) + Int32(256) * buf_idx_value
        res_dkt = epilogue_process_dkt(idx, tmem_dkt_base, sub_seq_len, *dkt_sc)

        # --- DKT exchange (intra-WG) ---
        cute.arch.barrier(barrier_id=DKT_BAR_ID, number_of_threads=128)
        buf_DKT0 = buf_DKT0_buf[(buf_idx_value, None, None)]
        buf_DKT1 = buf_DKT1_buf[(buf_idx_value, None, None)]
        epilogue_exchange_dkt(buf_DKT0, buf_DKT1, idx, K_OFF, *res, *res_dkt)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=DKT_BAR_ID, number_of_threads=128)

        # --- Wait for dkg data, then output dg / dk ---
        cute.arch.mbarrier_wait(mbar_dkg_tma_ready_ptr + buf_idx_value, local_phase)
        sDG = buf_DG_buf[(buf_idx_value, None, None)]
        sDK = buf_DK_buf[(buf_idx_value, None, None)]

        if idx < Int32(64):
            if local < sub_seq_len:
                token_row = start_offset + tile_idx * Int32(T_TILE) + local
                dg_stride = h_param * Int32(K_SIZE)
                dg_ptr_base = dg_out_ptr + (
                    token_row * dg_stride + head_idx * Int32(K_SIZE) + Int32(k_idx) * Int32(K_TILE) + Int32(K_OFF)
                )
                epilogue_output_dg(buf_K, sDG, buf_DKT1, idx, sub_seq_len, K_OFF, dg_ptr_base, *res, *res_dkt)
        else:
            token_row = start_offset + tile_idx * Int32(T_TILE) + local
            dk_stride = h_param * Int32(K_SIZE)
            dk_ptr_base = dk_out_ptr + (
                token_row * dk_stride + head_idx * Int32(K_SIZE) + Int32(k_idx) * Int32(K_TILE) + Int32(K_OFF)
            )
            epilogue_output_dk(sDK, buf_DKT0, idx, sub_seq_len, K_OFF, dk_ptr_base, *res, *res_dkt)

        cute.arch.mbarrier_arrive(mbar_val_free_ptr + buf_idx_value)

        b_phase = b_phase ^ Int32(1)
        state_phase = state_phase ^ (Int32(1) << buf_idx_value)
        buf_idx_value = (buf_idx_value + Int32(1)) % Int32(NUM_BUF_VALUE)

    # ===== POST-LOOP: DB REDUCE =====
    if idx >= Int32(64):
        if cutlass.const_expr(wg_idx == 0):
            if local < sub_seq_len:
                buf_DBpart[(Int32(0), local)] = db
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        if cutlass.const_expr(wg_idx == 1):
            if local < sub_seq_len:
                db = db + buf_DBpart[(Int32(0), local)]

    # DB output (WG1 only)
    if cutlass.const_expr(wg_idx == 1):
        if (idx >= Int32(64)) & (local < sub_seq_len):
            token_row = start_offset + tile_idx * Int32(T_TILE) + local
            flat_idx = token_row * h_param + head_idx
            db_out_ptr[flat_idx] = db

    state_phase = state_phase ^ (Int32(1) << (buf_idx_A + Int32(NUM_BUF_VALUE)))
    buf_idx_A = (buf_idx_A + Int32(1)) % Int32(NUM_BUF_A)

    return state_phase, buf_idx_A, buf_idx_value


# ============================================================
# MMA warp body
# (elect_one) executes all UMMA calls per tile.
# ============================================================


@cute.jit
def mma_warp_body(
    buf_KG_in,  # [K_TILE, 6*SUB_T_TILE] TF32
    buf_KG_ex,  # [K_TILE, 4*SUB_T_TILE] TF32
    buf_QKG_in,  # [K_TILE, 6*2*SUB_T_TILE] TF32
    buf_QKG_ex,  # [K_TILE, 4*2*SUB_T_TILE] TF32
    mbar_kg_done_ready_ptr,
    mbar_qkg_done_ready_ptr,
    mbar_dA_rdy_ptr,
    mbar_dAt_rdy_ptr,
    mbar_dq_fin_ptr,
    mbar_dkt_fin_ptr,
    tmem_base: Int32,
    state_phase: Int32,
    buf_idx_A: Int32,
    buf_idx_value: Int32,
    A_phase: Int32,
    sub_seq_len: Int32,
):
    """One tile's MMA work executed by a single elected thread."""
    b_phase = Int32(0)

    # Build B-matrix descriptors from SMEM pointers (once per tile)
    desc_kg_intra = build_b_desc(buf_KG_in.iterator)
    desc_kg_inter = build_b_desc(buf_KG_ex.iterator)
    desc_qkg_intra = build_b_desc(buf_QKG_in.iterator)
    desc_qkg_inter = build_b_desc(buf_QKG_ex.iterator)

    tcgen05_fence_after()

    # Wait for dAt (mask_At) once before the loop — Epilogue now signals before its loop
    cute.arch.mbarrier_wait(mbar_dAt_rdy_ptr + buf_idx_A, A_phase)

    for k_idx in cutlass.range(K_ITERATION, unroll_full=False):
        local_phase = (state_phase >> buf_idx_value) & Int32(1)

        # ===== KG PHASE =====
        cute.arch.mbarrier_wait(mbar_kg_done_ready_ptr, b_phase)

        tmem_dq_02 = tmem_base + Int32(DQ_02) + Int32(256) * buf_idx_value
        tmem_dq_13 = tmem_base + Int32(DQ_13) + Int32(256) * buf_idx_value
        tmem_dq2_02 = tmem_base + Int32(DQ2_02) + Int32(256) * buf_idx_value
        tmem_dq2_13 = tmem_base + Int32(DQ2_13) + Int32(256) * buf_idx_value

        tmem_a_02 = tmem_base + Int32(DAQK_02) + Int32(256) * buf_idx_A
        tmem_a_13 = tmem_base + Int32(DAQK_13) + Int32(256) * buf_idx_A

        # kg_intra call 1: MASK02, A=dAqk_13, B=intra[0], C=dq_13, 2 K-atoms
        mma_kg_intra_call1(tmem_a_13, desc_kg_intra, tmem_dq_13)

        # kg_intra call 2: MASK13, A=dAqk_02, B=intra[1..2], C=dq_02, 4 K-atoms
        mma_kg_intra_call2(tmem_a_02, desc_kg_intra + KG_SLOT_BYTES, tmem_dq_02)

        # kg_intra call 3: MASK13, A=dAqk_13, B=intra[3..5], C=dq_13, 6 K-atoms
        mma_kg_intra_call3(tmem_a_13, desc_kg_intra + 3 * KG_SLOT_BYTES, tmem_dq_13)

        tcgen05_fence_after()

        # kg_inter call 4: MASK02, A=dAqk_02, B=inter[0], C=dq2_02, 2 K-atoms
        mma_kg_inter_call(tmem_a_02, desc_kg_inter, tmem_dq2_02, 0)

        # kg_inter call 5: MASK02, A=dAqk_13+16, B=inter[1], C=dq2_13, 2 K-atoms
        mma_kg_inter_call(tmem_a_13 + Int32(16), desc_kg_inter + KG_SLOT_BYTES, tmem_dq2_13, 0)

        # kg_inter call 6: MASK13, A=dAqk_02+32, B=inter[2], C=dq2_02, 2 K-atoms
        mma_kg_inter_call(tmem_a_02 + Int32(32), desc_kg_inter + 2 * KG_SLOT_BYTES, tmem_dq2_02, 1)

        # kg_inter call 7: MASK13, A=dAqk_13+48, B=inter[3], C=dq2_13, 2 K-atoms
        mma_kg_inter_call(tmem_a_13 + Int32(48), desc_kg_inter + 3 * KG_SLOT_BYTES, tmem_dq2_13, 1)

        umma_arrive_noelect(mbar_dq_fin_ptr)

        tcgen05_fence_after()

        # ===== QKG PHASE =====
        cute.arch.mbarrier_wait(mbar_qkg_done_ready_ptr, b_phase)

        tmem_dkt_02 = tmem_base + Int32(DKT_02) + Int32(256) * buf_idx_value
        tmem_dkt_13 = tmem_base + Int32(DKT_13) + Int32(256) * buf_idx_value

        # All qkg A offsets use NO double-buffering (dAqk_t is single-write)
        tmem_at_02 = tmem_base + Int32(DAQK_T_02)
        tmem_at_13 = tmem_base + Int32(DAQK_T_13)

        # qkg_intra call 0: MASK0, A=dAqk_t_02+32, B=qkg_intra[0..5], 12 K-atoms
        mma_qkg_intra_call0(tmem_at_02 + Int32(32), desc_qkg_intra, tmem_dkt_02)

        # qkg_intra call 1: MASK0, A=dAqk_t_13+64, B=qkg_intra[3..6], 8 K-atoms
        mma_qkg_intra_call1(tmem_at_13 + Int32(64), desc_qkg_intra + 3 * QKG_SLOT_BYTES, tmem_dkt_13)

        # qkg_intra call 2: MASK1, A=dAqk_t_02+96, B=qkg_intra[5..6], 4 K-atoms
        mma_qkg_intra_call2(tmem_at_02 + Int32(96), desc_qkg_intra + 5 * QKG_SLOT_BYTES, tmem_dkt_02)

        tcgen05_fence_after()
        tcgen05_fence_before()

        # qkg_inter call 0: MASK2, A=dAqk_t_02, B=qkg_inter[0..1], 4 K-atoms
        mma_qkg_inter_call(tmem_at_02, desc_qkg_inter, tmem_dkt_02, 2)

        # qkg_inter call 1: MASK2, A=dAqk_t_13+32, B=qkg_inter[1..2], 4 K-atoms
        mma_qkg_inter_call(tmem_at_13 + Int32(32), desc_qkg_inter + QKG_SLOT_BYTES, tmem_dkt_13, 2)

        # qkg_inter call 2: MASK3, A=dAqk_t_02+64, B=qkg_inter[2..3], 4 K-atoms
        mma_qkg_inter_call(tmem_at_02 + Int32(64), desc_qkg_inter + 2 * QKG_SLOT_BYTES, tmem_dkt_02, 3)

        # qkg_inter call 3: MASK3, A=dAqk_t_13+96, B=qkg_inter[3..4], 4 K-atoms
        mma_qkg_inter_call(tmem_at_13 + Int32(96), desc_qkg_inter + 3 * QKG_SLOT_BYTES, tmem_dkt_13, 3)

        umma_arrive_noelect(mbar_dkt_fin_ptr)

        tcgen05_fence_after()
        b_phase = b_phase ^ Int32(1)
        state_phase = state_phase ^ (Int32(1) << buf_idx_value)
        buf_idx_value = (buf_idx_value + Int32(1)) % Int32(NUM_BUF_VALUE)

    state_phase = state_phase ^ (Int32(1) << (buf_idx_A + Int32(NUM_BUF_VALUE)))
    buf_idx_A = (buf_idx_A + Int32(1)) % Int32(NUM_BUF_A)
    return state_phase, buf_idx_A, buf_idx_value


# ============================================================
# TMA helper sizes (in elements)
# ============================================================

# BF16 [T_TILE, K_TILE] K_SW64: 2048 elements
SMEM_QK_ELEMS = T_TILE * K_TILE  # 2048 BF16 per buffer
SMEM_GF32_ELEMS = T_TILE * K_TILE  # 2048 F32 per buffer
SMEM_DA_ELEMS = T_TILE * T_TILE  # 4096 F32 per buffer
SMEM_KG_SLOT_ELEMS = K_TILE * SUB_T_TILE  # 512 TF32 per intra slot
SMEM_QKG_SLOT_ELEMS = K_TILE * SUB_T_TILE * 2  # 1024 TF32 per qkg slot
SMEM_DKT_ELEMS = T_TILE * K_TILE  # 2048 F32

SMEM_KG_IN_ELEMS = 6 * SMEM_KG_SLOT_ELEMS
SMEM_KG_EX_ELEMS = 4 * SMEM_KG_SLOT_ELEMS
SMEM_QKG_IN_ELEMS = 6 * SMEM_QKG_SLOT_ELEMS
SMEM_QKG_EX_ELEMS = 4 * SMEM_QKG_SLOT_ELEMS

# TMA byte sizes for each barrier group
TMA_KG_BYTES = SMEM_QK_ELEMS * 2 + SMEM_GF32_ELEMS * 4  # K(bf16) + G(f32) = 4096+8192 = 12288
TMA_QB_BYTES = SMEM_QK_ELEMS * 2 + SMEM_GF32_ELEMS * 4  # Q(bf16) + DQ(f32)
TMA_DKG_BYTES = SMEM_GF32_ELEMS * 4 * 2  # DK(f32) + DG(f32)
TMA_DA_BYTES = SMEM_DA_ELEMS * 4 * 2  # DAqk(f32) + DAkk(f32)

ROLE_EMPTY = 0
ROLE_LOAD = 1
ROLE_MMA = 2
ROLE_EPILOGUE = 5


# ============================================================
# KDA Backward Intra SM100 kernel class
# ============================================================

from cutlass.cute.nvgpu import cpasync


class KDABwdIntraSM100:
    """Host-side driver for kda_bwd_intra_sm100 CuteDSL kernel."""

    def __init__(self):
        pass  # SharedStorage is defined lazily inside @cute.jit __call__

    @cute.kernel
    def _kernel(
        self,
        # TMA copy atoms
        tma_atom_q: cute.CopyAtom,
        tma_atom_k: cute.CopyAtom,
        tma_atom_g: cute.CopyAtom,
        tma_atom_dAqk: cute.CopyAtom,
        tma_atom_dAkk: cute.CopyAtom,
        tma_atom_dq: cute.CopyAtom,
        tma_atom_dk: cute.CopyAtom,
        tma_atom_dg: cute.CopyAtom,
        # TMA gmem tensors
        tma_tensor_q: cute.Tensor,
        tma_tensor_k: cute.Tensor,
        tma_tensor_g: cute.Tensor,
        tma_tensor_dAqk: cute.Tensor,
        tma_tensor_dAkk: cute.Tensor,
        tma_tensor_dq: cute.Tensor,
        tma_tensor_dk: cute.Tensor,
        tma_tensor_dg: cute.Tensor,
        # Output / input raw GMEM pointers
        dq_out_ptr: cute.Pointer,  # BFloat16
        dk_out_ptr: cute.Pointer,  # BFloat16
        dg_out_ptr: cute.Pointer,  # Float32
        db_out_ptr: cute.Pointer,  # Float32
        db_in_ptr: cute.Pointer,  # Float32 (previous db)
        beta_ptr: cute.Pointer,  # BFloat16 or Float32 [total_q, H]
        tile_counter_ptr: cute.Pointer,  # Int32
        cu_seqlens_ptr: cute.Pointer,  # Int32 [B+1]
        chunk_indices_ptr: cute.Pointer,  # Int32 [num_chunks*2]
        total_tiles: Int32,
        H_param: Int32,
        K_param: Int32,
    ):
        # ---- Thread identification ----
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread_idx = cute.arch.thread_idx()[0]
        warpgroup_idx = cute.arch.make_warp_uniform(
            thread_idx // Int32(128)
        )  # shfl-uniform so ptxas can prove warpgroup-convergence

        # Decode warp role: KWARD_ASSIGNMENT is 40-bit so Int32 would truncate.
        # Instead use direct per-warp role lookup (warps 0-7=Epilogue, 8=MMA, 9=LOAD, 10-11=EMPTY).
        role = cutlass.select_(
            warp_idx >= Int32(10),
            Int32(ROLE_EMPTY),
            cutlass.select_(
                warp_idx == Int32(9),
                Int32(ROLE_LOAD),
                cutlass.select_(warp_idx == Int32(8), Int32(ROLE_MMA), Int32(ROLE_EPILOGUE)),
            ),
        )

        # ---- SMEM allocation ----
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Barrier array pointers
        p_kg_tma = storage.mbar_kg_tma.data_ptr()
        p_dA_tma = storage.mbar_dA_tma.data_ptr()
        p_qb_tma = storage.mbar_qb_tma.data_ptr()
        p_dkg_tma = storage.mbar_dkg_tma.data_ptr()
        p_kg_done = storage.mbar_kg_done.data_ptr()
        p_qkg_done = storage.mbar_qkg_done.data_ptr()
        p_dq_fin = storage.mbar_dq_fin.data_ptr()
        p_dkt_fin = storage.mbar_dkt_fin.data_ptr()
        p_dA_rdy = storage.mbar_dA_rdy.data_ptr()
        p_dAt_rdy = storage.mbar_dAt_rdy.data_ptr()
        p_val_free = storage.mbar_val_free.data_ptr()
        p_mask_rdy = storage.mbar_mask_rdy.data_ptr()

        # All SMEM data field pointers — must be extracted before any dynamic if/while
        tmem_addr_ptr = storage.tmem_addr.data_ptr()
        tile_id_ptr = storage.tile_id.data_ptr()
        buf_DAqk_ptr = storage.buf_DAqk.data_ptr()
        buf_DAkk_ptr = storage.buf_DAkk.data_ptr()
        buf_beta_base = storage.buf_beta.data_ptr()
        buf_K_ptr = storage.buf_K.data_ptr()
        buf_G_ptr = storage.buf_G.data_ptr()
        buf_Q_ptr = storage.buf_Q.data_ptr()
        buf_DQ_ld_ptr = storage.buf_DQ_ld.data_ptr()
        buf_DK_ld_ptr = storage.buf_DK_ld.data_ptr()
        buf_DG_ld_ptr = storage.buf_DG_ld.data_ptr()
        buf_DKT0_ptr = storage.buf_DKT0.data_ptr()
        buf_DKT1_ptr = storage.buf_DKT1.data_ptr()
        buf_dbpart_ptr = storage.buf_dbpart.data_ptr()
        buf_KG_in_ptr = storage.buf_KG_in.data_ptr()
        buf_KG_ex_ptr = storage.buf_KG_ex.data_ptr()
        buf_QKG_in_ptr = storage.buf_QKG_in.data_ptr()
        buf_QKG_ex_ptr = storage.buf_QKG_ex.data_ptr()

        # Tensor wrappers for fields requiring subscript reads/writes
        tile_id_smem = cute.make_tensor(tile_id_ptr, cute.make_layout((2,)))
        buf_dbpart_smem = cute.make_tensor(buf_dbpart_ptr, cute.make_layout((64,)))

        # ---- TMA descriptor prefetch (warp 0, elect_one) ----
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_q)
                cpasync.prefetch_descriptor(tma_atom_k)
                cpasync.prefetch_descriptor(tma_atom_g)
                cpasync.prefetch_descriptor(tma_atom_dAqk)
                cpasync.prefetch_descriptor(tma_atom_dAkk)
                cpasync.prefetch_descriptor(tma_atom_dq)
                cpasync.prefetch_descriptor(tma_atom_dk)
                cpasync.prefetch_descriptor(tma_atom_dg)

        # ---- Barrier initialization (warp 0, elect_one) ----
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                for i in cutlass.range(NUM_BUF_VALUE, unroll_full=True):
                    cute.arch.mbarrier_init(p_kg_tma + i, Int32(1))
                    cute.arch.mbarrier_init(p_qb_tma + i, Int32(1))
                    cute.arch.mbarrier_init(p_dkg_tma + i, Int32(1))
                    cute.arch.mbarrier_init(p_val_free + i, Int32(256))
                cute.arch.mbarrier_init(p_kg_done, Int32(256))
                cute.arch.mbarrier_init(p_qkg_done, Int32(256))
                cute.arch.mbarrier_init(p_dq_fin, Int32(1))
                cute.arch.mbarrier_init(p_dkt_fin, Int32(1))
                for i in cutlass.range(NUM_BUF_A, unroll_full=True):
                    cute.arch.mbarrier_init(p_dA_tma + i, Int32(1))
                    cute.arch.mbarrier_init(p_dA_rdy + i, Int32(256))
                    cute.arch.mbarrier_init(p_dAt_rdy + i, Int32(256))
                cute.arch.mbarrier_init(p_mask_rdy, Int32(64))
                cute.arch.mbarrier_init_fence()

            # TMEM allocation (warp 0 only - all warps in warp 0)
            cute.arch.alloc_tmem(Int32(512), tmem_addr_ptr)
            cute.arch.relinquish_tmem_alloc_permit()

        cute.arch.sync_threads()

        # Retrieve TMEM base address (from SMEM hold register)
        tmem_col_base = cute.make_tensor(tmem_addr_ptr, cute.make_layout((1,)))[(Int32(0),)]

        # ---- Persistent state ----
        state_phase = Int32(0)
        buf_idx_A = Int32(0)
        buf_idx_value = Int32(0)
        tile_phase = Int32(0)

        # Shared layout helpers
        # Swizzled SMEM: separate swizzle (in pointer) + outer layout (in tensor)
        # K_SW64 for bf16, K_SW128 for f32, K_SW64 for dA f32
        sw_bf16 = cute.make_swizzle(2, 4, 3)  # Swizzle<2,4,3> = K_SW64 for bf16
        sw_f32 = cute.make_swizzle(3, 4, 3)  # Swizzle<3,4,3> = K_SW128 for f32
        sw_da = cute.make_swizzle(2, 4, 3)  # Swizzle<2,4,3> = K_SW64 for dA f32
        # Outer layouts: tile_to_shape on non-swizzled atoms
        # K-major atom: (M=8, K=N) with stride (N, 1) — M first, K contiguous
        layout_qk = cute.tile_to_shape(cute.make_layout((8, K_TILE), stride=(K_TILE, 1)), (T_TILE, K_TILE), (0, 1))
        layout_gf = cute.tile_to_shape(cute.make_layout((8, K_TILE), stride=(K_TILE, 1)), (T_TILE, K_TILE), (0, 1))
        layout_da = cute.tile_to_shape(cute.make_layout((8, 16), stride=(16, 1)), (T_TILE, T_TILE), (0, 1))
        layout_dkt = cute.make_layout((T_TILE, K_TILE), stride=(K_TILE, 1))
        layout_kg_intra_v = cute.make_layout((K_TILE, 6 * SUB_T_TILE), stride=(1, K_TILE))
        layout_kg_inter_v = cute.make_layout((K_TILE, 4 * SUB_T_TILE), stride=(1, K_TILE))
        layout_qkg_intra_v = cute.make_layout((K_TILE, 6 * 2 * SUB_T_TILE), stride=(1, K_TILE))
        layout_qkg_inter_v = cute.make_layout((K_TILE, 4 * 2 * SUB_T_TILE), stride=(1, K_TILE))

        buf_KG_in = cute.make_tensor(buf_KG_in_ptr, layout_kg_intra_v)
        buf_KG_ex = cute.make_tensor(buf_KG_ex_ptr, layout_kg_inter_v)
        buf_QKG_in = cute.make_tensor(buf_QKG_in_ptr, layout_qkg_intra_v)
        buf_QKG_ex = cute.make_tensor(buf_QKG_ex_ptr, layout_qkg_inter_v)

        # ==================================================================
        # EPILOGUE WARPGROUP (warps 0-7)
        # ==================================================================
        if role == Int32(ROLE_EPILOGUE):
            cute.arch.setmaxregister_increase(REG_COMPUTE)

            wg_idx_local = warpgroup_idx  # 0 or 1
            idx = thread_idx % Int32(128)
            local = idx % Int32(64)

            # Epilogue warp: fetch first tile before entering persistent loop
            A_phase = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
            cute.arch.mbarrier_wait(p_dA_tma + buf_idx_A, A_phase)
            tile_id_v = tile_id_smem[(A_phase,)]

            while tile_id_v < total_tiles:
                # Decode tile: tile_id encodes head as tile_id % H, chunk as tile_id // H
                chunk_off_v = (tile_id_v // H_param) * Int32(2)
                batch_idx_v = cute.make_tensor(chunk_indices_ptr + chunk_off_v, cute.make_layout((1,)))[(Int32(0),)]
                tile_idx_v = cute.make_tensor(chunk_indices_ptr + chunk_off_v + Int32(1), cute.make_layout((1,)))[(Int32(0),)]
                head_idx_v = tile_id_v % H_param
                start_offset = cute.make_tensor(cu_seqlens_ptr + batch_idx_v, cute.make_layout((1,)))[(Int32(0),)]
                seq_len_v = (
                    cute.make_tensor(cu_seqlens_ptr + batch_idx_v + Int32(1), cute.make_layout((1,)))[(Int32(0),)]
                    - start_offset
                )
                sub_seq_len = min(Int32(T_TILE), seq_len_v - tile_idx_v * Int32(T_TILE))

                # WG-specific K_OFF
                K_OFF = wg_idx_local * Int32(HALF_K)

                # WG1 only: load initial db
                db = Float32(0.0)
                if wg_idx_local == Int32(1):
                    if (idx >= Int32(64)) & (local < sub_seq_len):
                        flat_idx = (start_offset + tile_idx_v * Int32(T_TILE) + local) * H_param + head_idx_v
                        db = cute.make_tensor(db_in_ptr + flat_idx, cute.make_layout((1,)))[(Int32(0),)]

                # --- Mask dA into TMEM ---
                # --- Mask dA into TMEM ---
                # Within each warpgroup, first 64 threads load dAqk, second 64 load dAkk
                # (first 64 threads → dAqk, second 64 → dAkk)
                # Offset is per-warpgroup: WG0=0 (cols 0-31), WG1=32 (cols 32-63)
                da_buf_off = buf_idx_A * Int32(T_TILE * T_TILE)
                buf_DAqk_view = cute.make_tensor(cute.recast_ptr(buf_DAqk_ptr + da_buf_off, sw_da), layout_da)
                buf_DAkk_view = cute.make_tensor(cute.recast_ptr(buf_DAkk_ptr + da_buf_off, sw_da), layout_da)

                if idx < Int32(64):
                    mask_A_tensor(buf_DAqk_view, idx, sub_seq_len, tmem_col_base + Int32(DAQK_02), wg_idx_local * Int32(32))
                else:
                    mask_A_tensor(buf_DAkk_view, idx, sub_seq_len, tmem_col_base + Int32(DAQK_02), wg_idx_local * Int32(32))
                cute.arch.fence_view_async_tmem_store()
                tcgen05_fence_before()

                cute.arch.mbarrier_arrive(p_dA_rdy + buf_idx_A)

                # === mask_At: transpose dA mask into TMEM (hoisted from loop) ===
                mask_At_tensor(
                    cute.make_tensor(cute.recast_ptr(buf_DAqk_ptr + da_buf_off, sw_da), layout_da),
                    cute.make_tensor(cute.recast_ptr(buf_DAkk_ptr + da_buf_off, sw_da), layout_da),
                    idx,
                    sub_seq_len,
                    tmem_col_base + Int32(DAQK_T_02),
                    wg_idx_local * Int32(64),
                )
                tcgen05_fence_before()
                cute.arch.mbarrier_arrive(p_dAt_rdy + buf_idx_A)

                # Wait for buf_beta
                cute.arch.mbarrier_wait(p_mask_rdy, tile_phase)
                buf_beta_ptr = buf_beta_base + A_phase * Int32(T_TILE)

                # Hoist loop-invariant: y, beta loads, output base address
                y = idx % Int32(8) * Int32(4)
                beta_row = idx // Int32(8)
                beta0_v = cutlass.select_(
                    beta_row < sub_seq_len,
                    cute.make_tensor(buf_beta_ptr + beta_row, cute.make_layout((1,)))[(Int32(0),)],
                    Float32(0.0),
                )
                beta1_v = cutlass.select_(
                    beta_row + Int32(16) < sub_seq_len,
                    cute.make_tensor(buf_beta_ptr + (beta_row + Int32(16)), cute.make_layout((1,)))[(Int32(0),)],
                    Float32(0.0),
                )
                beta2_v = cutlass.select_(
                    beta_row + Int32(32) < sub_seq_len,
                    cute.make_tensor(buf_beta_ptr + (beta_row + Int32(32)), cute.make_layout((1,)))[(Int32(0),)],
                    Float32(0.0),
                )
                beta3_v = cutlass.select_(
                    beta_row + Int32(48) < sub_seq_len,
                    cute.make_tensor(buf_beta_ptr + (beta_row + Int32(48)), cute.make_layout((1,)))[(Int32(0),)],
                    Float32(0.0),
                )
                # beta for db accumulation (upper half only, but safe to load unconditionally)
                beta_val_loc = cutlass.select_(
                    local < sub_seq_len,
                    cute.make_tensor(buf_beta_ptr + local, cute.make_layout((1,)))[(Int32(0),)],
                    Float32(0.0),
                )
                # Precompute output base address (saves H_param*K_param multiply per k-iter)
                out_base_v = (
                    (start_offset + tile_idx_v * Int32(T_TILE) + local) * H_param * K_param + head_idx_v * K_param + K_OFF
                )

                # Precompute TMEM base addresses (loop-invariant)
                tmem_dq_fixed = tmem_col_base + Int32(DQ_02) + K_OFF
                tmem_dq2_fixed = tmem_col_base + Int32(DQ2_02) + K_OFF
                tmem_dkt_fixed = tmem_col_base + Int32(DKT_02) + K_OFF

                # Saved gn values to eliminate redundant SMEM loads across if/else blocks
                saved_gn_0 = Float32(0.0)
                saved_gn_1 = Float32(0.0)
                saved_gn_2 = Float32(0.0)
                saved_gn_3 = Float32(0.0)

                b_phase = Int32(0)

                for k_idx in cutlass.range(K_ITERATION, unroll=2):
                    local_phase = (state_phase >> buf_idx_value) & Int32(1)

                    # Buffer offsets for this k-step
                    off_qk = buf_idx_value * Int32(T_TILE * K_TILE)

                    # Raw pointers for vectorized SMEM loads (early-use only)
                    buf_G_raw = buf_G_ptr + off_qk
                    buf_K_raw = buf_K_ptr + off_qk

                    # === Wait for K/G ===
                    cute.arch.mbarrier_wait(p_kg_tma + buf_idx_value, local_phase)

                    # === Setup kg_intra ===
                    if wg_idx_local == Int32(0):
                        gn3 = smem_load_f32x4_sw128(buf_G_raw, Int32(48), y)
                        saved_gn_0 = gn3[0]
                        saved_gn_1 = gn3[1]
                        saved_gn_2 = gn3[2]
                        saved_gn_3 = gn3[3]
                        setup_kg_intra(buf_G_raw, buf_K_raw, buf_KG_in, 0, idx, gn3[0], gn3[1], gn3[2], gn3[3], 3)
                    else:
                        gn1 = smem_load_f32x4_sw128(buf_G_raw, Int32(16), y)
                        gn2 = smem_load_f32x4_sw128(buf_G_raw, Int32(32), y)
                        saved_gn_0 = gn2[0]
                        saved_gn_1 = gn2[1]
                        saved_gn_2 = gn2[2]
                        saved_gn_3 = gn2[3]
                        setup_kg_intra_2gn(
                            buf_G_raw,
                            buf_K_raw,
                            buf_KG_in,
                            0,
                            idx,
                            gn1[0],
                            gn1[1],
                            gn1[2],
                            gn1[3],
                            gn2[0],
                            gn2[1],
                            gn2[2],
                            gn2[3],
                            0,
                            1,
                        )
                        setup_kg_intra(buf_G_raw, buf_K_raw, buf_KG_in, 1, idx, gn2[0], gn2[1], gn2[2], gn2[3], 2)

                    # === Wait for Q/DQ ===
                    cute.arch.mbarrier_wait(p_qb_tma + buf_idx_value, local_phase)
                    buf_Q_raw = buf_Q_ptr + off_qk

                    # === kg_intra fused with qkg_intra (WG0) ===
                    if wg_idx_local == Int32(0):
                        # Reuse saved gn3 (row 48) from Phase 1; load gn1 (row 16) fresh
                        gn1 = smem_load_f32x4_sw128(buf_G_raw, Int32(16), y)
                        setup_intra_fused(
                            buf_G_raw,
                            buf_K_raw,
                            buf_Q_raw,
                            buf_KG_in,
                            buf_QKG_in,
                            1,
                            idx,
                            sub_seq_len,
                            saved_gn_0,
                            saved_gn_1,
                            saved_gn_2,
                            saved_gn_3,
                            gn1[0],
                            gn1[1],
                            gn1[2],
                            gn1[3],
                            beta1_v,
                            beta1_v,
                            4,
                            0,
                        )
                        setup_intra_fused(
                            buf_G_raw,
                            buf_K_raw,
                            buf_Q_raw,
                            buf_KG_in,
                            buf_QKG_in,
                            2,
                            idx,
                            sub_seq_len,
                            saved_gn_0,
                            saved_gn_1,
                            saved_gn_2,
                            saved_gn_3,
                            gn1[0],
                            gn1[1],
                            gn1[2],
                            gn1[3],
                            beta2_v,
                            beta2_v,
                            5,
                            1,
                        )
                        # Save gn1 (row 16) for qkg_intra remaining phase
                        saved_gn_0 = gn1[0]
                        saved_gn_1 = gn1[1]
                        saved_gn_2 = gn1[2]
                        saved_gn_3 = gn1[3]

                    # === Inter setup ===
                    if wg_idx_local == Int32(0):
                        setup_inter_fused(
                            buf_G_raw, buf_K_raw, buf_Q_raw, buf_KG_ex, buf_QKG_ex, 0, idx, sub_seq_len, beta0_v, beta0_v
                        )
                        setup_inter_fused(
                            buf_G_raw, buf_K_raw, buf_Q_raw, buf_KG_ex, buf_QKG_ex, 3, idx, sub_seq_len, beta3_v, beta3_v
                        )
                    else:
                        setup_inter_fused(
                            buf_G_raw, buf_K_raw, buf_Q_raw, buf_KG_ex, buf_QKG_ex, 1, idx, sub_seq_len, beta1_v, beta1_v
                        )
                        setup_inter_fused(
                            buf_G_raw, buf_K_raw, buf_Q_raw, buf_KG_ex, buf_QKG_ex, 2, idx, sub_seq_len, beta2_v, beta2_v
                        )

                    cute.arch.fence_view_async_shared()

                    cute.arch.mbarrier_arrive(p_kg_done)

                    # === qkg_intra remaining rows ===
                    if wg_idx_local == Int32(0):
                        # Reuse saved gn1 (row 16) from kg_intra fused phase
                        setup_qkg_intra(
                            buf_G_raw,
                            buf_Q_raw,
                            buf_K_raw,
                            buf_QKG_in,
                            3,
                            idx,
                            sub_seq_len,
                            beta3_v,
                            beta3_v,
                            saved_gn_0,
                            saved_gn_1,
                            saved_gn_2,
                            saved_gn_3,
                            2,
                        )
                    else:
                        # Reuse saved gn2 (row 32) from Phase 1; only load gn3 (row 48) fresh
                        gn3 = smem_load_f32x4_sw128(buf_G_raw, Int32(48), y)
                        setup_qkg_intra(
                            buf_G_raw,
                            buf_Q_raw,
                            buf_K_raw,
                            buf_QKG_in,
                            2,
                            idx,
                            sub_seq_len,
                            beta2_v,
                            beta2_v,
                            saved_gn_0,
                            saved_gn_1,
                            saved_gn_2,
                            saved_gn_3,
                            3,
                        )
                        setup_qkg_intra_2gn(
                            buf_G_raw,
                            buf_Q_raw,
                            buf_K_raw,
                            buf_QKG_in,
                            3,
                            idx,
                            sub_seq_len,
                            beta3_v,
                            beta3_v,
                            saved_gn_0,
                            saved_gn_1,
                            saved_gn_2,
                            saved_gn_3,
                            gn3[0],
                            gn3[1],
                            gn3[2],
                            gn3[3],
                            4,
                            5,
                        )

                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(p_qkg_done)

                    # === Wait for dq MMA results ===
                    cute.arch.mbarrier_wait(p_dq_fin, b_phase)

                    tmem_dq_base = tmem_dq_fixed + Int32(256) * buf_idx_value
                    tmem_dq2_base = tmem_dq2_fixed + Int32(256) * buf_idx_value

                    # Fused: compute scales inline + apply to TMEM dq/dq2
                    res = epilogue_dq_scaled(buf_G_raw, idx, sub_seq_len, K_OFF, tmem_dq_base, tmem_dq2_base)

                    # === Output dq / accumulate db ===
                    if idx >= Int32(64):
                        result_db_t = epilogue_accumulate_db(buf_K_raw, idx, sub_seq_len, K_OFF, beta_val_loc, db, *res)
                        db = result_db_t[0]
                        res = result_db_t[1:]
                    else:
                        if local < sub_seq_len:
                            dq_ptr = dq_out_ptr + out_base_v + k_idx * Int32(K_TILE)
                            buf_DQ_raw = buf_DQ_ld_ptr + off_qk
                            res = epilogue_output_dq(buf_Q_raw, buf_DQ_raw, idx, sub_seq_len, K_OFF, dq_ptr, *res)

                    # === Wait for dkt MMA results + DKG data (overlap TMA with epilogue) ===
                    cute.arch.mbarrier_wait(p_dkg_tma + buf_idx_value, local_phase)
                    cute.arch.mbarrier_wait(p_dkt_fin, b_phase)

                    # Fused: compute dkt scales inline + apply to TMEM dkt
                    tmem_dkt_base = tmem_dkt_fixed + Int32(256) * buf_idx_value
                    res_dkt = epilogue_dkt_scaled(buf_G_raw, idx, sub_seq_len, K_OFF, tmem_dkt_base)

                    # === DKT exchange ===
                    off_dkt = buf_idx_value * Int32(T_TILE * K_TILE)
                    buf_DKT0_raw = buf_DKT0_ptr + off_dkt
                    buf_DKT1_raw = buf_DKT1_ptr + off_dkt
                    # No barrier needed before exchange: dkt_scaled reads TMEM/buf_G only,
                    # exchange writes to thread-local sDKT rows (no cross-thread hazard).
                    epilogue_exchange_dkt(buf_DKT0_raw, buf_DKT1_raw, idx, K_OFF, *res, *res_dkt)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=wg_idx_local * Int32(2), number_of_threads=128)

                    if idx < Int32(64):
                        if local < sub_seq_len:
                            dg_ptr = dg_out_ptr + out_base_v + k_idx * Int32(K_TILE)
                            buf_DG_raw = buf_DG_ld_ptr + off_qk
                            epilogue_output_dg(
                                buf_K_raw, buf_DG_raw, buf_DKT1_raw, idx, sub_seq_len, K_OFF, dg_ptr, *res, *res_dkt
                            )
                    else:
                        if local < sub_seq_len:
                            dk_ptr = dk_out_ptr + out_base_v + k_idx * Int32(K_TILE)
                            buf_DK_raw = buf_DK_ld_ptr + off_qk
                            epilogue_output_dk(buf_DK_raw, buf_DKT0_raw, idx, sub_seq_len, K_OFF, dk_ptr, *res, *res_dkt)

                    cute.arch.mbarrier_arrive(p_val_free + buf_idx_value)
                    b_phase = b_phase ^ Int32(1)
                    state_phase = state_phase ^ (Int32(1) << buf_idx_value)
                    buf_idx_value = (buf_idx_value + Int32(1)) % Int32(NUM_BUF_VALUE)

                # === Post-loop db reduce ===
                if idx >= Int32(64):
                    if wg_idx_local == Int32(0):
                        if local < sub_seq_len:
                            buf_dbpart_smem[(local,)] = db
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    if wg_idx_local == Int32(1):
                        if local < sub_seq_len:
                            db = db + buf_dbpart_smem[(local,)]

                # === DB output (WG1 only) ===
                if (wg_idx_local == Int32(1)) & (idx >= Int32(64)) & (local < sub_seq_len):
                    flat_idx_db = (start_offset + tile_idx_v * Int32(T_TILE) + local) * H_param + head_idx_v
                    cute.make_tensor(db_out_ptr + flat_idx_db, cute.make_layout((1,)))[(Int32(0),)] = db

                state_phase = state_phase ^ (Int32(1) << (buf_idx_A + Int32(NUM_BUF_VALUE)))
                buf_idx_A = (buf_idx_A + Int32(1)) % Int32(NUM_BUF_A)
                tile_phase = tile_phase ^ Int32(1)

                # Fetch next tile for loop condition check
                A_phase = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
                cute.arch.mbarrier_wait(p_dA_tma + buf_idx_A, A_phase)
                tile_id_v = tile_id_smem[(A_phase,)]

            # Epilogue warp termination: signal MMA warp to stop
            cute.arch.mbarrier_arrive(p_dA_rdy + buf_idx_A)

        # ==================================================================
        # MMA WARP
        # ==================================================================
        elif role == Int32(ROLE_MMA):
            cute.arch.setmaxregister_decrease(REG_LOAD)
            with cute.arch.elect_one():
                A_phase = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
                cute.arch.mbarrier_wait(p_dA_rdy + buf_idx_A, A_phase)
                tile_id_m = tile_id_smem[(A_phase,)]

                while tile_id_m < total_tiles:
                    sp = mma_warp_body(
                        buf_KG_in,
                        buf_KG_ex,
                        buf_QKG_in,
                        buf_QKG_ex,
                        p_kg_done,
                        p_qkg_done,
                        p_dA_rdy,
                        p_dAt_rdy,
                        p_dq_fin,
                        p_dkt_fin,
                        tmem_col_base,
                        state_phase,
                        buf_idx_A,
                        buf_idx_value,
                        A_phase,
                        Int32(T_TILE),  # placeholder sub_seq_len (MMA doesn't use it)
                    )
                    state_phase = sp[0]
                    buf_idx_A = sp[1]
                    buf_idx_value = sp[2]

                    # Break SSA – force re-read after mma_warp_body
                    state_phase = state_phase | Int32(0)
                    buf_idx_A = buf_idx_A | Int32(0)
                    buf_idx_value = buf_idx_value | Int32(0)

                    A_phase = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
                    cute.arch.mbarrier_wait(p_dA_rdy + buf_idx_A, A_phase)
                    tile_id_m = tile_id_smem[(A_phase,)]

                # MMA done: deallocate TMEM (must happen after all MMA instructions complete)
                tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_addr_ptr)
                cute.arch.dealloc_tmem(tmem_ptr, 512)

            # Break SSA: elect_one yields nothing; reset to prevent dominance violation
            state_phase = Int32(0)
            buf_idx_A = Int32(0)
            buf_idx_value = Int32(0)

        # ==================================================================
        # LOAD WARP
        # ==================================================================
        elif role == Int32(ROLE_LOAD):
            cute.arch.setmaxregister_decrease(REG_LOAD)

            with cute.arch.elect_one():
                A_phase = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
                tile_id_l = cute.arch.atomic_add(tile_counter_ptr, Int32(1))
                tile_id_smem[(A_phase,)] = tile_id_l
                cute.arch.fence_acq_rel_cta()

                k_total = Int32(0)  # total k-iterations processed (used to bootstrap p_val_free wait)
                while tile_id_l < total_tiles:
                    chunk_off_l = (tile_id_l // H_param) * Int32(2)
                    batch_idx_l = cute.make_tensor(chunk_indices_ptr + chunk_off_l, cute.make_layout((1,)))[(Int32(0),)]
                    tile_idx_l = cute.make_tensor(chunk_indices_ptr + chunk_off_l + Int32(1), cute.make_layout((1,)))[
                        (Int32(0),)
                    ]
                    head_idx_l = tile_id_l % H_param
                    tok_off_l = cute.make_tensor(cu_seqlens_ptr + batch_idx_l, cute.make_layout((1,)))[(Int32(0),)]

                    # Wait for previous dAt to protect dA SMEM.
                    # Skip on the very first tile (k_total==0): dA SMEM is fresh, no prior Epilogue.
                    # From tile 1+ onward (k_total>0): Epilogue must have finished mask_At for the
                    # PREVIOUS tile before LOAD can reuse the single-buffered dA SMEM slot.
                    # NOTE: We must NOT use a flag initialized to 1 before the while loop,
                    # because MLIR constant-propagates 'flag=0' in the loop body, overwriting
                    # the initial value of 1 → all iterations see flag=0 → deadlock on tile 0.
                    # k_total is an accumulating variable (cannot be constant-folded).
                    if k_total != Int32(0):
                        cute.arch.mbarrier_wait(p_dAt_rdy + buf_idx_A, A_phase ^ Int32(1))

                    # TMA load dAqk + dAkk
                    da_off_l = buf_idx_A * Int32(T_TILE * T_TILE)
                    buf_DAqk_l = cute.make_tensor(cute.recast_ptr(buf_DAqk_ptr + da_off_l, sw_da), layout_da)
                    buf_DAkk_l = cute.make_tensor(cute.recast_ptr(buf_DAkk_ptr + da_off_l, sw_da), layout_da)

                    tma_dAqk_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_dAqk)
                    tma_dAkk_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_dAkk)

                    gDAqk_2d = tma_dAqk_v[None, None, head_idx_l]
                    gDAkk_2d = tma_dAkk_v[None, None, head_idx_l]
                    gDAqk_t = cute.local_tile(gDAqk_2d, (T_TILE, T_TILE), (tile_idx_l, 0))
                    gDAkk_t = cute.local_tile(gDAkk_2d, (T_TILE, T_TILE), (tile_idx_l, 0))

                    tXbuf_DAqk, tXgDAqk = cpasync.tma_partition(
                        tma_atom_dAqk,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(buf_DAqk_l, 0, 2),
                        cute.group_modes(gDAqk_t, 0, 2),
                    )
                    tXbuf_DAkk, tXgDAkk = cpasync.tma_partition(
                        tma_atom_dAkk,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(buf_DAkk_l, 0, 2),
                        cute.group_modes(gDAkk_t, 0, 2),
                    )

                    cute.arch.mbarrier_arrive_and_expect_tx(p_dA_tma + buf_idx_A, Int32(TMA_DA_BYTES))
                    cute.copy(tma_atom_dAqk, tXgDAqk[None], tXbuf_DAqk[None], tma_bar_ptr=p_dA_tma + buf_idx_A)
                    cute.copy(tma_atom_dAkk, tXgDAkk[None], tXbuf_DAkk[None], tma_bar_ptr=p_dA_tma + buf_idx_A)

                    # Per k_idx loads
                    tma_q_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_q)
                    tma_k_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_k)
                    tma_g_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_g)
                    tma_dq_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_dq)
                    tma_dk_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_dk)
                    tma_dg_v = cute.domain_offset((tok_off_l, 0, 0), tma_tensor_dg)

                    for k_idx_l in cutlass.range(K_ITERATION, unroll_full=False):
                        lp = (state_phase >> buf_idx_value) & Int32(1)
                        # Skip p_val_free back-pressure wait for the first NUM_BUF_VALUE k-iterations
                        # across the entire kernel (bootstrap: buffers are initially free).
                        # From k_total >= NUM_BUF_VALUE onward, Epilogue must have released the buffer.
                        if k_total >= Int32(NUM_BUF_VALUE):
                            cute.arch.mbarrier_wait(p_val_free + buf_idx_value, lp ^ Int32(1))
                        k_total = k_total + Int32(1)

                        off_qkl = buf_idx_value * Int32(T_TILE * K_TILE)
                        off_gfl = buf_idx_value * Int32(T_TILE * K_TILE)

                        sKl = cute.make_tensor(cute.recast_ptr(buf_K_ptr + off_qkl, sw_bf16), layout_qk)
                        sGl = cute.make_tensor(cute.recast_ptr(buf_G_ptr + off_gfl, sw_f32), layout_gf)
                        sQl = cute.make_tensor(cute.recast_ptr(buf_Q_ptr + off_qkl, sw_bf16), layout_qk)
                        sDQl = cute.make_tensor(cute.recast_ptr(buf_DQ_ld_ptr + off_gfl, sw_f32), layout_gf)
                        sDKl = cute.make_tensor(cute.recast_ptr(buf_DK_ld_ptr + off_gfl, sw_f32), layout_gf)
                        sDGl = cute.make_tensor(cute.recast_ptr(buf_DG_ld_ptr + off_gfl, sw_f32), layout_gf)

                        gK_2dl = tma_k_v[None, None, head_idx_l]
                        gG_2dl = tma_g_v[None, None, head_idx_l]
                        gQ_2dl = tma_q_v[None, None, head_idx_l]
                        gDQ_2dl = tma_dq_v[None, None, head_idx_l]
                        gDK_2dl = tma_dk_v[None, None, head_idx_l]
                        gDG_2dl = tma_dg_v[None, None, head_idx_l]

                        gKt = cute.local_tile(gK_2dl, (T_TILE, K_TILE), (tile_idx_l, k_idx_l))
                        gGt = cute.local_tile(gG_2dl, (T_TILE, K_TILE), (tile_idx_l, k_idx_l))
                        gQt = cute.local_tile(gQ_2dl, (T_TILE, K_TILE), (tile_idx_l, k_idx_l))
                        gDQt = cute.local_tile(gDQ_2dl, (T_TILE, K_TILE), (tile_idx_l, k_idx_l))
                        gDKt = cute.local_tile(gDK_2dl, (T_TILE, K_TILE), (tile_idx_l, k_idx_l))
                        gDGt = cute.local_tile(gDG_2dl, (T_TILE, K_TILE), (tile_idx_l, k_idx_l))

                        tXsKl, tXgKl = cpasync.tma_partition(
                            tma_atom_k, 0, cute.make_layout(1), cute.group_modes(sKl, 0, 2), cute.group_modes(gKt, 0, 2)
                        )
                        tXsGl, tXgGl = cpasync.tma_partition(
                            tma_atom_g, 0, cute.make_layout(1), cute.group_modes(sGl, 0, 2), cute.group_modes(gGt, 0, 2)
                        )
                        tXsQl, tXgQl = cpasync.tma_partition(
                            tma_atom_q, 0, cute.make_layout(1), cute.group_modes(sQl, 0, 2), cute.group_modes(gQt, 0, 2)
                        )
                        tXsDQl, tXgDQl = cpasync.tma_partition(
                            tma_atom_dq, 0, cute.make_layout(1), cute.group_modes(sDQl, 0, 2), cute.group_modes(gDQt, 0, 2)
                        )
                        tXsDKl, tXgDKl = cpasync.tma_partition(
                            tma_atom_dk, 0, cute.make_layout(1), cute.group_modes(sDKl, 0, 2), cute.group_modes(gDKt, 0, 2)
                        )
                        tXsDGl, tXgDGl = cpasync.tma_partition(
                            tma_atom_dg, 0, cute.make_layout(1), cute.group_modes(sDGl, 0, 2), cute.group_modes(gDGt, 0, 2)
                        )

                        cute.arch.mbarrier_arrive_and_expect_tx(p_kg_tma + buf_idx_value, Int32(TMA_KG_BYTES))
                        cute.copy(tma_atom_k, tXgKl[None], tXsKl[None], tma_bar_ptr=p_kg_tma + buf_idx_value)
                        cute.copy(tma_atom_g, tXgGl[None], tXsGl[None], tma_bar_ptr=p_kg_tma + buf_idx_value)

                        cute.arch.mbarrier_arrive_and_expect_tx(p_qb_tma + buf_idx_value, Int32(TMA_QB_BYTES))
                        cute.copy(tma_atom_q, tXgQl[None], tXsQl[None], tma_bar_ptr=p_qb_tma + buf_idx_value)
                        cute.copy(tma_atom_dq, tXgDQl[None], tXsDQl[None], tma_bar_ptr=p_qb_tma + buf_idx_value)

                        cute.arch.mbarrier_arrive_and_expect_tx(p_dkg_tma + buf_idx_value, Int32(TMA_DKG_BYTES))
                        cute.copy(tma_atom_dk, tXgDKl[None], tXsDKl[None], tma_bar_ptr=p_dkg_tma + buf_idx_value)
                        cute.copy(tma_atom_dg, tXgDGl[None], tXsDGl[None], tma_bar_ptr=p_dkg_tma + buf_idx_value)

                        state_phase = state_phase ^ (Int32(1) << buf_idx_value)
                        buf_idx_value = (buf_idx_value + Int32(1)) % Int32(NUM_BUF_VALUE)

                    state_phase = state_phase ^ (Int32(1) << (buf_idx_A + Int32(NUM_BUF_VALUE)))
                    buf_idx_A = (buf_idx_A + Int32(1)) % Int32(NUM_BUF_A)

                    # Fetch next tile for loop condition check
                    A_phase = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
                    tile_id_l = cute.arch.atomic_add(tile_counter_ptr, Int32(1))
                    tile_id_smem[(A_phase,)] = tile_id_l
                    cute.arch.fence_acq_rel_cta()

                # Load warp termination: signal Epilogue/Empty warps with 0-byte TMA
                cute.arch.mbarrier_arrive_and_expect_tx(p_dA_tma + buf_idx_A, Int32(0))

            # Break SSA: elect_one yields nothing; reset to prevent dominance violation
            state_phase = Int32(0)
            buf_idx_A = Int32(0)
            buf_idx_value = Int32(0)
            # tile_id_smem is written inside elect_one's while; refresh from ptr in outer scope
            tile_id_smem = cute.make_tensor(tile_id_ptr, cute.make_layout((2,)))

        # ==================================================================
        # EMPTY WARP (warps 10-11): load buf_beta
        # ==================================================================
        else:
            cute.arch.setmaxregister_decrease(REG_LOAD)

            empty_idx = thread_idx - Int32(NUM_THREADS - 64)  # 0..63

            A_phase_e = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
            cute.arch.mbarrier_wait(p_dA_tma + buf_idx_A, A_phase_e)
            tile_id_e = tile_id_smem[(A_phase_e,)]

            while tile_id_e < total_tiles:
                chunk_off_e = (tile_id_e // H_param) * Int32(2)
                batch_idx_e = cute.make_tensor(chunk_indices_ptr + chunk_off_e, cute.make_layout((1,)))[(Int32(0),)]
                tile_idx_e = cute.make_tensor(chunk_indices_ptr + chunk_off_e + Int32(1), cute.make_layout((1,)))[(Int32(0),)]
                head_idx_e = tile_id_e % H_param
                tok_off_e = cute.make_tensor(cu_seqlens_ptr + batch_idx_e, cute.make_layout((1,)))[(Int32(0),)]
                seq_len_e = (
                    cute.make_tensor(cu_seqlens_ptr + batch_idx_e + Int32(1), cute.make_layout((1,)))[(Int32(0),)] - tok_off_e
                )
                sub_len_e = min(Int32(T_TILE), seq_len_e - tile_idx_e * Int32(T_TILE))

                buf_beta_w = buf_beta_base + A_phase_e * Int32(T_TILE)

                if empty_idx < Int32(T_TILE):
                    beta_f32 = Float32(0.0)
                    if empty_idx < sub_len_e:
                        flat_e = (tok_off_e + tile_idx_e * Int32(T_TILE) + empty_idx) * H_param + head_idx_e
                        beta_f32 = Float32(cute.make_tensor(beta_ptr + flat_e, cute.make_layout((1,)))[(Int32(0),)])
                    cute.make_tensor(buf_beta_w + empty_idx, cute.make_layout((1,)))[(Int32(0),)] = beta_f32

                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(p_mask_rdy)

                state_phase = state_phase ^ (Int32(1) << (buf_idx_A + Int32(NUM_BUF_VALUE)))
                buf_idx_A = (buf_idx_A + Int32(1)) % Int32(NUM_BUF_A)

                # Fetch next tile for loop condition check
                A_phase_e = (state_phase >> (buf_idx_A + Int32(NUM_BUF_VALUE))) & Int32(1)
                cute.arch.mbarrier_wait(p_dA_tma + buf_idx_A, A_phase_e)
                tile_id_e = tile_id_smem[(A_phase_e,)]

        # ==================================================================
        # Cleanup: TMEM deallocation
        # ==================================================================
        # Cannot use sync_threads here: non-elected MMA/LOAD threads (62 total)
        # reach bar.sync 0 immediately, blocking TMA completion.
        # TMEM dealloc is done inside the Epilogue branch above (thread 0) after
        # the pipeline has fully completed.
        pass

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,  # [1, total_q, H, K] bf16
        k_in: cute.Tensor,  # [1, total_q, H, K] bf16
        g_in: cute.Tensor,  # [1, total_q, H, K] f32
        dAqk_in: cute.Tensor,  # [1, total_q, H, BT] f32
        dAkk_in: cute.Tensor,  # [1, total_q, H, BT] f32
        dq_in: cute.Tensor,  # [1, total_q, H, K] f32
        dk_in: cute.Tensor,  # [1, total_q, H, K] f32
        dg_in: cute.Tensor,  # [1, total_q, H, K] f32
        db_in: cute.Tensor,  # [1, total_q, H] f32
        beta_in: cute.Tensor,  # [1, total_q, H] bf16 or f32
        dq_out_in: cute.Tensor,  # [1, total_q, H, K] bf16
        dk_out_in: cute.Tensor,  # [1, total_q, H, K] bf16
        dg_out_in: cute.Tensor,  # [1, total_q, H, K] f32
        db_out_in: cute.Tensor,  # [1, total_q, H] f32
        tile_counter_in: cute.Tensor,  # [1] i32
        cu_seqlens_in: cute.Tensor,  # [N+1] i32
        chunk_indices_in: cute.Tensor,  # [2*num_chunks] i32
        problem_size: tuple[Int32, Int32, Int32],  # (total_q, H_sz, K_sz)
        total_tiles: Int32,
        stream,
    ):
        total_q, H_sz, K_sz = problem_size
        BT = T_TILE

        q_ptr = q_in.iterator
        k_ptr = k_in.iterator
        g_ptr = g_in.iterator
        dAqk_ptr = dAqk_in.iterator
        dAkk_ptr = dAkk_in.iterator
        dq_ptr = dq_in.iterator
        dk_ptr = dk_in.iterator
        dg_ptr = dg_in.iterator

        # Build gmem tensors using token-indexed flat layouts
        # shape: (total_q, K, H), stride: (H*K, 1, K)  [heads-last, K contiguous]
        qkg_layout = cute.make_layout(
            (total_q, K_sz, H_sz),
            stride=(H_sz * K_sz, 1, K_sz),
        )
        da_layout = cute.make_layout(
            (total_q, BT, H_sz),
            stride=(H_sz * BT, 1, BT),
        )

        gmem_q = cute.make_tensor(q_ptr, qkg_layout)
        gmem_k = cute.make_tensor(k_ptr, qkg_layout)
        gmem_g = cute.make_tensor(g_ptr, qkg_layout)
        gmem_dAqk = cute.make_tensor(dAqk_ptr, da_layout)
        gmem_dAkk = cute.make_tensor(dAkk_ptr, da_layout)
        gmem_dq = cute.make_tensor(dq_ptr, qkg_layout)
        gmem_dk = cute.make_tensor(dk_ptr, qkg_layout)
        gmem_dg = cute.make_tensor(dg_ptr, qkg_layout)

        # Swizzled SMEM layouts:
        #   BF16 (Q/K):       K_SW64  → Swizzle<2,4,3>
        #   F32  (G/DQ/DK/DG): K_SW128 → Swizzle<3,4,3>
        #   F32  (dAqk/dAkk):  K_SW64  → Swizzle<2,4,3>
        # K-major atoms have shape (M=8, K=N), matching our (T, K) indexing convention.
        layout_bf16 = cute.tile_to_shape(make_smem_layout_atom(SmemLayoutAtomKind.K_SW64, BFloat16), (T_TILE, K_TILE), (0, 1))
        layout_f32 = cute.tile_to_shape(make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, Float32), (T_TILE, K_TILE), (0, 1))
        layout_da_h = cute.tile_to_shape(make_smem_layout_atom(SmemLayoutAtomKind.K_SW64, Float32), (T_TILE, T_TILE), (0, 1))

        tma_q_a, tma_q_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gmem_q, layout_bf16, (T_TILE, K_TILE)
        )
        tma_k_a, tma_k_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gmem_k, layout_bf16, (T_TILE, K_TILE)
        )
        tma_g_a, tma_g_t = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), gmem_g, layout_f32, (T_TILE, K_TILE))
        tma_dAqk_a, tma_dAqk_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gmem_dAqk, layout_da_h, (T_TILE, T_TILE)
        )
        tma_dAkk_a, tma_dAkk_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gmem_dAkk, layout_da_h, (T_TILE, T_TILE)
        )
        tma_dq_a, tma_dq_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gmem_dq, layout_f32, (T_TILE, K_TILE)
        )
        tma_dk_a, tma_dk_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gmem_dk, layout_f32, (T_TILE, K_TILE)
        )
        tma_dg_a, tma_dg_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gmem_dg, layout_f32, (T_TILE, K_TILE)
        )

        import torch

        num_sm = torch.cuda.get_device_properties(0).multi_processor_count

        # Define SharedStorage here (inside @cute.jit context) so @cute.struct
        # fields are registered with MLIR before being used in @cute.kernel
        @cute.struct
        class SharedStorage:
            buf_Q: cute.struct.Align[cute.struct.MemRange[BFloat16, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            buf_K: cute.struct.Align[cute.struct.MemRange[BFloat16, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            buf_G: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            buf_KG_in: cute.struct.Align[cute.struct.MemRange[Float32, 6 * K_TILE * SUB_T_TILE], 128]
            buf_KG_ex: cute.struct.Align[cute.struct.MemRange[Float32, 4 * K_TILE * SUB_T_TILE], 128]
            buf_QKG_in: cute.struct.Align[cute.struct.MemRange[Float32, 6 * K_TILE * SUB_T_TILE * 2], 128]
            buf_QKG_ex: cute.struct.Align[cute.struct.MemRange[Float32, 4 * K_TILE * SUB_T_TILE * 2], 128]
            buf_DAqk: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_A * T_TILE * T_TILE], 128]
            buf_DAkk: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_A * T_TILE * T_TILE], 128]
            buf_DQ_ld: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            buf_DK_ld: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            buf_DG_ld: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            buf_DKT0: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            buf_DKT1: cute.struct.Align[cute.struct.MemRange[Float32, NUM_BUF_VALUE * T_TILE * K_TILE], 128]
            mbar_kg_tma: cute.struct.Align[cute.struct.MemRange[Int64, NUM_BUF_VALUE], 16]
            mbar_dA_tma: cute.struct.Align[cute.struct.MemRange[Int64, NUM_BUF_A], 16]
            mbar_qb_tma: cute.struct.Align[cute.struct.MemRange[Int64, NUM_BUF_VALUE], 16]
            mbar_dkg_tma: cute.struct.Align[cute.struct.MemRange[Int64, NUM_BUF_VALUE], 16]
            mbar_kg_done: cute.struct.Align[cute.struct.MemRange[Int64, 1], 16]
            mbar_qkg_done: cute.struct.Align[cute.struct.MemRange[Int64, 1], 16]
            mbar_dq_fin: cute.struct.Align[cute.struct.MemRange[Int64, 1], 16]
            mbar_dkt_fin: cute.struct.Align[cute.struct.MemRange[Int64, 1], 16]
            mbar_dA_rdy: cute.struct.Align[cute.struct.MemRange[Int64, NUM_BUF_A], 16]
            mbar_dAt_rdy: cute.struct.Align[cute.struct.MemRange[Int64, NUM_BUF_A], 16]
            mbar_val_free: cute.struct.Align[cute.struct.MemRange[Int64, NUM_BUF_VALUE], 16]
            mbar_mask_rdy: cute.struct.Align[cute.struct.MemRange[Int64, 1], 16]
            buf_beta: cute.struct.Align[cute.struct.MemRange[Float32, 2 * T_TILE], 16]
            tile_id: cute.struct.MemRange[Int32, 2]
            buf_dbpart: cute.struct.Align[cute.struct.MemRange[Float32, 2 * T_TILE], 16]
            tmem_addr: cute.struct.Align[cute.struct.MemRange[Int32, 1], 4]

        self.shared_storage = SharedStorage

        self._kernel(
            tma_q_a,
            tma_k_a,
            tma_g_a,
            tma_dAqk_a,
            tma_dAkk_a,
            tma_dq_a,
            tma_dk_a,
            tma_dg_a,
            tma_q_t,
            tma_k_t,
            tma_g_t,
            tma_dAqk_t,
            tma_dAkk_t,
            tma_dq_t,
            tma_dk_t,
            tma_dg_t,
            dq_out_in.iterator,
            dk_out_in.iterator,
            dg_out_in.iterator,
            db_out_in.iterator,
            db_in.iterator,
            beta_in.iterator,
            tile_counter_in.iterator,
            cu_seqlens_in.iterator,
            chunk_indices_in.iterator,
            total_tiles,
            H_sz,
            K_sz,
        ).launch(
            grid=[num_sm, 1, 1],
            block=[NUM_THREADS, 1, 1],
            smem=216 * 1024,
            stream=stream,
            min_blocks_per_mp=1,
        )


# ============================================================================
# TVM-FFI compile cache
# ============================================================================

_bwd_intra_cache = {}


_TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def compile_kda_bwd_intra(H, K=K_SIZE, BT=T_TILE, beta_dtype=torch.bfloat16):
    """Pre-compile KDABwdIntraSM100 with TVM-FFI for zero-overhead invocation.

    Args:
        H: Number of heads.
        K: Head dimension (default 128).
        BT: Chunk / tile size (default 64).
        beta_dtype: ``torch.dtype`` for the beta tensor — ``torch.bfloat16``
            or ``torch.float32``.  Default: ``torch.bfloat16``.

    Returns a compiled callable that accepts raw torch tensors directly
    (no from_dlpack needed)::

        compiled_fn = compile_kda_bwd_intra(H=4, beta_dtype=torch.float32)
        compiled_fn(
            q, k, g, dAqk, dAkk, dq, dk, dg, db, beta,
            dq_out, dk_out, dg_out, db_out,
            tile_counter, cu_seqlens, chunk_indices,
            (Int32(T), Int32(H), Int32(K)),
            Int32(num_tiles),
        )
    """
    key = (H, K, BT, beta_dtype)
    if key in _bwd_intra_cache:
        return _bwd_intra_cache[key]

    beta_cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[beta_dtype]

    kernel_obj = KDABwdIntraSM100()

    sym_t = cute.sym_int()  # total_tokens (dynamic)
    sym_cu = cute.sym_int()  # cu_seqlens length
    sym_ci = cute.sym_int()  # chunk_indices length

    # Inputs:  [1, total_q, H, K] or [1, total_q, H, BT]
    q_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    g_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dAqk_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H, BT), stride_order=(3, 2, 1, 0), assumed_align=128)
    dAkk_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H, BT), stride_order=(3, 2, 1, 0), assumed_align=128)
    dq_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dk_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dg_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    # db_in: [1, total_q, H]  f32;  beta_in: [1, total_q, H]  beta_cutlass_dtype
    db_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H), stride_order=(2, 1, 0), assumed_align=128)
    beta_fake = make_fake_compact_tensor(beta_cutlass_dtype, (1, sym_t, H), stride_order=(2, 1, 0), assumed_align=128)
    # Outputs
    dq_out_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dk_out_fake = make_fake_compact_tensor(cutlass.BFloat16, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    dg_out_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    db_out_fake = make_fake_compact_tensor(cutlass.Float32, (1, sym_t, H), stride_order=(2, 1, 0), assumed_align=128)
    # Scalars
    tc_fake = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=128)
    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=128)
    ci_fake = make_fake_compact_tensor(cutlass.Int32, (sym_ci,), assumed_align=128)

    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        g_fake,
        dAqk_fake,
        dAkk_fake,
        dq_fake,
        dk_fake,
        dg_fake,
        db_fake,
        beta_fake,
        dq_out_fake,
        dk_out_fake,
        dg_out_fake,
        db_out_fake,
        tc_fake,
        cu_fake,
        ci_fake,
        (Int32(1), Int32(H), Int32(K)),
        Int32(1),
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )
    _bwd_intra_cache[key] = compiled_fn
    return compiled_fn
