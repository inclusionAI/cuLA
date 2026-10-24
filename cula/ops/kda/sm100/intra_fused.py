# ruff: noqa
"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Persistent K2+K3 Kernel for KDA.

Consumes the FP32 chunk-cumulative gate tensor produced by the public KDA
pipeline and computes intra sub-chunk Aqk/Akk (K2) plus the optional merged
inverse (K3) in a single persistent kernel.

Grid: (NUM_SMS, 1, 1) — 148 persistent blocks, each loops over work units
  Total work units = (NT/4) * H * B, distributed round-robin across SMs
  Block i processes work units i, i+NUM_SMS, i+2*NUM_SMS, ...
Block: 480 threads (15 warps), warp-specialized by role:
  Warps 0-9: K2/K3 MMA compute (one lower-triangular tile per warp)
  Warp 10: dedicated TMA producer for Q/K/G and beta
  Warps 11-14: vectorized Aqk/Akk stores

Pipeline (single for_generate, warp groups separated by if-blocks):
  per work unit:
    Warp 10: prefetch Q/K/G and beta, reusing stages after MMA releases them
    Warps 0-9: wait(input_ready)+wait(store_done), MMA,
               arrive(mma_done+stage_reuse)
    Warps 11-14: wait(mma_done), store sAqk/sAkk→GMEM, arrive(store_done)
  All warp-group invariants are computed inside each group's if-block (not hoisted)
  to eliminate cross-group register pressure — same budget as the _all version.
  Mbarrier phases self-reset after 4 iterations (2 stages × 2 phases).

Mbarriers:
  tma_mbars[2]:          count=1, TMA completion → MMA waits for Q/K/G
  stage_reuse_mbars[2]:  count=320, MMA(10 warps) → TMA waits before reuse
  input_ready_mbars[2]:  count=32, TMA warp → MMA waits for staged beta
  mma_done_mbars[2]:     count=320, MMA(10 warps) → Store waits for sAqk/sAkk ready
  store_done_mbars[2]:   count=128, Store(4 warps) → MMA waits for sAqk/sAkk stage free

SMEM: ~184KB (Q/K and FP32 G double buffers, Aqk/Akk double buffers,
      staged beta, and pipeline barriers).

Inputs:
  gk      [B,T,H,K]   fp32  activated, chunk-cumulative gate
  k       [B,T,H,K]   bf16
  q       [B,T,H,K]   bf16
  beta    [B,T,H]      bf16/fp32 used for Akk unit lower triangular
  scale                fp32  1/sqrt(K)

Outputs:
  A_qk       [B,T,H,BT]  bf16  full merged (diagonal + off-diagonal)
  A_kk       [B,T,H,BT]  bf16  pre-inverse output by default; bf16 inverse
                              output when fp32_akk_inv uses an fp32 workspace
"""


import weakref

import cutlass
import cutlass.cute as cute
import torch
from cutlass import for_generate, yield_out
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from fla.ops.utils import prepare_chunk_indices

from cula.ops.kda.sm100.akk_inv_fp32 import akk_inv_fp32_physical_host as _akk_inv_fp32_physical_host
from cula.ops.kda.sm100.akk_inv_tf32 import _add_store_C16_smem as _tf32_add_store_C16_smem
from cula.ops.kda.sm100.akk_inv_tf32 import _invert_diag_forward16 as _tf32_invert_diag_forward16
from cula.ops.kda.sm100.akk_inv_tf32 import _matmul16_smem_smem as _tf32_matmul16_smem_smem
from cula.ops.kda.sm100.akk_inv_tf32 import _matmul32_smem_smem as _tf32_matmul32_smem_smem
from cula.ops.kda.sm100.akk_inv_tf32 import _matmul16_tmp_smem as _tf32_matmul16_tmp_smem
from cula.ops.kda.sm100.akk_inv_tf32 import _store_C16_smem as _tf32_store_C16_smem
from cula.ops.kda.sm100.akk_inv_tf32 import _store_C16_tmp as _tf32_store_C16_tmp

# (Test-only references to upstream act_cumsum / intra_parallel / fla triton
# kernels were stripped during flashinfer migration — the fused fast path
# does not depend on them.)

BT = 64
BC = 16
K_DIM = 128
K_PAD = 8
K_STRIDE = K_DIM + K_PAD  # 136, padded row stride to avoid bank conflicts
CHUNKS_PER_BLOCK = 4
NUM_SMS = 148  # Persistent kernel: one resident block per SM

NUM_MMA_WARPS = 11  # 10 active MMA warps + 1 TMA producer
NUM_MMA_ACTIVE = 10  # mma_warp 0..9: actual MMA work
TMA_WARP_ID = NUM_MMA_ACTIVE  # warp 10 = dedicated TMA producer
NUM_STORE_WARPS = 4  # Warps 11-14: vectorized stores / optional in-CTA inverse
STORE_WARP_BASE = NUM_MMA_WARPS
NUM_WARPS = STORE_WARP_BASE + NUM_STORE_WARPS  # 15
THREADS = NUM_WARPS * 32  # 480

MMA_K_TILE = 16
NUM_MMA_K_TILES = K_DIM // MMA_K_TILE  # 8  (bf16 m16n8k16)
AQK_TILE_PAD = 8
AQK_TILE_STRIDE = BT + AQK_TILE_PAD  # 72 — sAqk now 64x72 row-major (same shape as sAkk)

# AKK_PAD=4 → stride 68 to avoid SMEM bank conflicts on MMA reads of sAkk
AKK_PAD = 4
AKK_STRIDE = BT + AKK_PAD  # 68 (padded, bank-conflict-free for MMA)
VEC = 2
NUM_STAGES = 2
INV_TMP_STRIDE = BC + 4


@dsl_user_op
def store_internal_barrier(*, loc=None, ip=None):
    """Named barrier for the four store/inverse warps. barrier_id=3."""
    llvm.inline_asm(
        T.i32(),
        [],
        "membar.cta; bar.sync 3, 128; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_bf16x2_f32(hi_f32, lo_f32, *, loc=None, ip=None):
    """Pack two fp32 values into a bf16x2 u32 register.
    Returns u32 with bf16(hi) in bits [31:16] and bf16(lo) in bits [15:0].
    PTX: cvt.rn.bf16x2.f32 d, a, b  -> d[31:16]=bf16(a), d[15:0]=bf16(b)
    """
    result = llvm.inline_asm(
        T.i32(),
        [hi_f32.ir_value(loc=loc, ip=ip), lo_f32.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def mma_bf16_m16n8k16(
    a0,
    a1,
    a2,
    a3,  # 4 u32 (bf16x2 packed) A operands
    b0,
    b1,  # 2 u32 (bf16x2 packed) B operands
    c0,
    c1,
    c2,
    c3,  # 4 fp32 accumulators
    *,
    loc=None,
    ip=None,
):
    """bf16 MMA with fp32 accumulator, shape m16n8k16.
    D_fp32 = A_bf16 * B_bf16 + C_fp32
    """
    # a/b already i32 (from pack_bf16x2_f32) -> no bitcast needed
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0.ir_value(loc=loc, ip=ip),
            a1.ir_value(loc=loc, ip=ip),
            a2.ir_value(loc=loc, ip=ip),
            a3.ir_value(loc=loc, ip=ip),
            b0.ir_value(loc=loc, ip=ip),
            b1.ir_value(loc=loc, ip=ip),
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        """{
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
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
def mma_tf32_m16n8k8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
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


# ============================================================
# Akk inverse helpers (fused from Akk_inverse_lower_triangle_bf16.py)
# ============================================================
@dsl_user_op
def _ak_pack_bf16x2(lo_f32, hi_f32, *, loc=None, ip=None):
    """cvt.rn.bf16x2.f32 -- pack two fp32 into bf16x2 (as fp32 bitcast view)."""
    result = llvm.inline_asm(
        T.i32(),
        [lo_f32.ir_value(loc=loc, ip=ip), hi_f32.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_movmatrix_trans(src, *, loc=None, ip=None):
    src_b = llvm.bitcast(T.i32(), src.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [src_b],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_mask_packed_ltri(packed, row, pair, *, loc=None, ip=None):
    """Mask packed bf16x2 to lower-tri: zero elements where row < col."""
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p, row.ir_value(loc=loc, ip=ip), pair.ir_value(loc=loc, ip=ip)],
        """{
            .reg .b32 %c0, %c1, %mlo, %mhi, %mask;
            .reg .pred %p0, %p1;
            shl.b32 %c0, $3, 1;
            add.u32 %c1, %c0, 1;
            setp.ge.s32 %p0, $2, %c0;
            setp.ge.s32 %p1, $2, %c1;
            selp.b32 %mlo, 0xFFFF, 0, %p0;
            selp.b32 %mhi, 0xFFFF0000, 0, %p1;
            or.b32 %mask, %mlo, %mhi;
            and.b32 $0, $1, %mask;
        }""",
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_unpack_bf16x2_lo(packed, *, loc=None, ip=None):
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p],
        "shl.b32 $0, $1, 16;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_unpack_bf16x2_hi(packed, *, loc=None, ip=None):
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p],
        "and.b32 $0, $1, 0xFFFF0000;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """BF16 MMA m16n8k16, args are Float32 (bitcast-viewed as i32 for bf16x2)."""
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
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
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
def _ak_invert_diag_neumann(sAkk, block_idx, lane_id, *, loc=None, ip=None):
    """Invert 16x16 diag block via Neumann series (I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8).
    Reads/writes sAkk as fp32 packed bf16x2 (stride 36). Each block handled by 1 warp."""
    r_off = block_idx * 16
    c_off = block_idx * 8  # 16 cols = 8 pairs
    gid = lane_id // 4
    tid = lane_id % 4

    packed0 = cutlass.Float32(sAkk[r_off + gid, c_off + tid])
    packed1 = cutlass.Float32(sAkk[r_off + gid + 8, c_off + tid])
    packed2 = cutlass.Float32(sAkk[r_off + gid, c_off + 4 + tid])
    packed3 = cutlass.Float32(sAkk[r_off + gid + 8, c_off + 4 + tid])

    A_f0 = _ak_unpack_bf16x2_lo(packed0)
    A_f1 = _ak_unpack_bf16x2_hi(packed0)
    A_f2 = _ak_unpack_bf16x2_lo(packed1)
    A_f3 = _ak_unpack_bf16x2_hi(packed1)
    A_f4 = _ak_unpack_bf16x2_lo(packed2)
    A_f5 = _ak_unpack_bf16x2_hi(packed2)
    A_f6 = _ak_unpack_bf16x2_lo(packed3)
    A_f7 = _ak_unpack_bf16x2_hi(packed3)

    _one = cutlass.Float32(1.0)
    _zero = cutlass.Float32(0.0)
    I_f0 = _one * cutlass.Float32(gid == 2 * tid) + _zero * cutlass.Float32(gid != 2 * tid)
    I_f1 = _one * cutlass.Float32(gid == 2 * tid + 1) + _zero * cutlass.Float32(gid != 2 * tid + 1)
    I_f2 = _one * cutlass.Float32(gid + 8 == 2 * tid) + _zero * cutlass.Float32(gid + 8 != 2 * tid)
    I_f3 = _one * cutlass.Float32(gid + 8 == 2 * tid + 1) + _zero * cutlass.Float32(gid + 8 != 2 * tid + 1)
    I_f4 = _one * cutlass.Float32(gid == 8 + 2 * tid) + _zero * cutlass.Float32(gid != 8 + 2 * tid)
    I_f5 = _one * cutlass.Float32(gid == 8 + 2 * tid + 1) + _zero * cutlass.Float32(gid != 8 + 2 * tid + 1)
    I_f6 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid) + _zero * cutlass.Float32(gid + 8 != 8 + 2 * tid)
    I_f7 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid + 1) + _zero * cutlass.Float32(gid + 8 != 8 + 2 * tid + 1)

    L_f0 = A_f0 - I_f0
    L_f1 = A_f1 - I_f1
    L_f2 = A_f2 - I_f2
    L_f3 = A_f3 - I_f3
    L_f4 = A_f4 - I_f4
    L_f5 = A_f5 - I_f5
    L_f6 = A_f6 - I_f6
    L_f7 = A_f7 - I_f7
    INV_f0 = I_f0 - L_f0
    INV_f1 = I_f1 - L_f1
    INV_f2 = I_f2 - L_f2
    INV_f3 = I_f3 - L_f3
    INV_f4 = I_f4 - L_f4
    INV_f5 = I_f5 - L_f5
    INV_f6 = I_f6 - L_f6
    INV_f7 = I_f7 - L_f7

    _zf = cutlass.Float32(0.0)
    L_a0 = _ak_pack_bf16x2(L_f0, L_f1)
    L_a1 = _ak_pack_bf16x2(L_f2, L_f3)
    L_a2 = _ak_pack_bf16x2(L_f4, L_f5)
    L_a3 = _ak_pack_bf16x2(L_f6, L_f7)
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 1: L^2, INV += INV*L^2
    L_b0 = _ak_movmatrix_trans(L_a0)
    L_b1 = _ak_movmatrix_trans(L_a1)
    L_b2 = _ak_movmatrix_trans(L_a2)
    L_b3 = _ak_movmatrix_trans(L_a3)
    Lp_c0, Lp_c1, Lp_c2, Lp_c3 = _ak_mma(L_a0, L_a1, L_a2, L_a3, L_b0, L_b1, _zf, _zf, _zf, _zf)
    Lp_c4, Lp_c5, Lp_c6, Lp_c7 = _ak_mma(L_a0, L_a1, L_a2, L_a3, L_b2, L_b3, _zf, _zf, _zf, _zf)
    Lp_a0 = _ak_pack_bf16x2(Lp_c0, Lp_c1)
    Lp_a1 = _ak_pack_bf16x2(Lp_c2, Lp_c3)
    Lp_a2 = _ak_pack_bf16x2(Lp_c4, Lp_c5)
    Lp_a3 = _ak_pack_bf16x2(Lp_c6, Lp_c7)
    Lp_b0 = _ak_movmatrix_trans(Lp_a0)
    Lp_b1 = _ak_movmatrix_trans(Lp_a1)
    Lp_b2 = _ak_movmatrix_trans(Lp_a2)
    Lp_b3 = _ak_movmatrix_trans(Lp_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf)
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf)
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 2: L^4, INV += INV*L^4
    L4_c0, L4_c1, L4_c2, L4_c3 = _ak_mma(Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf)
    L4_c4, L4_c5, L4_c6, L4_c7 = _ak_mma(Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf)
    L4_a0 = _ak_pack_bf16x2(L4_c0, L4_c1)
    L4_a1 = _ak_pack_bf16x2(L4_c2, L4_c3)
    L4_a2 = _ak_pack_bf16x2(L4_c4, L4_c5)
    L4_a3 = _ak_pack_bf16x2(L4_c6, L4_c7)
    L4_b0 = _ak_movmatrix_trans(L4_a0)
    L4_b1 = _ak_movmatrix_trans(L4_a1)
    L4_b2 = _ak_movmatrix_trans(L4_a2)
    L4_b3 = _ak_movmatrix_trans(L4_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf)
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf)
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 3: L^8, INV += INV*L^8
    L8_c0, L8_c1, L8_c2, L8_c3 = _ak_mma(L4_a0, L4_a1, L4_a2, L4_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf)
    L8_c4, L8_c5, L8_c6, L8_c7 = _ak_mma(L4_a0, L4_a1, L4_a2, L4_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf)
    L8_a0 = _ak_pack_bf16x2(L8_c0, L8_c1)
    L8_a1 = _ak_pack_bf16x2(L8_c2, L8_c3)
    L8_a2 = _ak_pack_bf16x2(L8_c4, L8_c5)
    L8_a3 = _ak_pack_bf16x2(L8_c6, L8_c7)
    L8_b0 = _ak_movmatrix_trans(L8_a0)
    L8_b1 = _ak_movmatrix_trans(L8_a1)
    L8_b2 = _ak_movmatrix_trans(L8_a2)
    L8_b3 = _ak_movmatrix_trans(L8_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L8_b0, L8_b1, _zf, _zf, _zf, _zf)
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L8_b2, L8_b3, _zf, _zf, _zf, _zf)
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7

    sAkk[r_off + gid, c_off + tid] = _ak_pack_bf16x2(INV_f0, INV_f1)
    sAkk[r_off + gid + 8, c_off + tid] = _ak_pack_bf16x2(INV_f2, INV_f3)
    sAkk[r_off + gid, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f4, INV_f5)
    sAkk[r_off + gid + 8, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f6, INV_f7)


@dsl_user_op
def _ak_invert_diag_neumann_inreg(
    sAkk,
    block_idx,
    lane_id,
    raw_f0,
    raw_f1,
    raw_f2,
    raw_f3,
    raw_f4,
    raw_f5,
    raw_f6,
    raw_f7,
    *,
    loc=None,
    ip=None,
):
    """In-register variant: invert 16x16 diag block from K2 acc fragment values
    (no SMEM read). Applies I+L mask, runs Neumann iterations, writes INV to sAkk.

    raw_f0..raw_f7 are K2 fp32 acc*beta values in C-fragment layout for m16n8k16:
      raw_f0,raw_f1 = (gid,   2*tid),   (gid,   2*tid+1)
      raw_f2,raw_f3 = (gid+8, 2*tid),   (gid+8, 2*tid+1)
      raw_f4,raw_f5 = (gid,   2*tid+8), (gid,   2*tid+9)
      raw_f6,raw_f7 = (gid+8, 2*tid+8), (gid+8, 2*tid+9)
    """
    r_off = block_idx * 16
    c_off = block_idx * 8
    gid = lane_id // 4
    tid = lane_id % 4

    _one = cutlass.Float32(1.0)
    _zero = cutlass.Float32(0.0)

    # Identity pattern (matches the SMEM-read variant's I_f computation)
    I_f0 = _one * cutlass.Float32(gid == 2 * tid) + _zero * cutlass.Float32(gid != 2 * tid)
    I_f1 = _one * cutlass.Float32(gid == 2 * tid + 1) + _zero * cutlass.Float32(gid != 2 * tid + 1)
    I_f2 = _one * cutlass.Float32(gid + 8 == 2 * tid) + _zero * cutlass.Float32(gid + 8 != 2 * tid)  # always 0
    I_f3 = _one * cutlass.Float32(gid + 8 == 2 * tid + 1) + _zero * cutlass.Float32(gid + 8 != 2 * tid + 1)  # always 0
    I_f4 = _one * cutlass.Float32(gid == 8 + 2 * tid) + _zero * cutlass.Float32(gid != 8 + 2 * tid)  # always 0
    I_f5 = _one * cutlass.Float32(gid == 8 + 2 * tid + 1) + _zero * cutlass.Float32(gid != 8 + 2 * tid + 1)  # always 0
    I_f6 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid) + _zero * cutlass.Float32(gid + 8 != 8 + 2 * tid)
    I_f7 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid + 1) + _zero * cutlass.Float32(gid + 8 != 8 + 2 * tid + 1)

    # Apply I+L mask to raw K2 values:
    #   diag (row==col): replace with 1
    #   strict upper (row<col within block): replace with 0
    #   strict lower (row>col within block): keep raw value (= L value)
    m_gt_a = cutlass.Float32(gid > 2 * tid)  # for cols 2*tid (with row=gid)
    m_gt_b = cutlass.Float32(gid > 2 * tid + 1)  # for cols 2*tid+1
    # (gid, 2*tid) and (gid, 2*tid+1): mask conditional
    A_f0 = m_gt_a * raw_f0 + I_f0
    A_f1 = m_gt_b * raw_f1 + I_f1
    # (gid+8, 2*tid) and (gid+8, 2*tid+1): always strict lower (gid+8 > 2*tid+1 always)
    A_f2 = raw_f2
    A_f3 = raw_f3
    # (gid, 2*tid+8) and (gid, 2*tid+9): always strict upper (gid <= 7 < 8 <= 2*tid+8)
    A_f4 = _zero
    A_f5 = _zero
    # (gid+8, 2*tid+8): row-col offset = gid+8 - (2*tid+8) = gid - 2*tid -> same as A_f0 mask
    # (gid+8, 2*tid+9): similar -> same as A_f1 mask
    A_f6 = m_gt_a * raw_f6 + I_f6
    A_f7 = m_gt_b * raw_f7 + I_f7

    # L = A - I, INV = I - L (first two Neumann terms)
    L_f0 = A_f0 - I_f0
    L_f1 = A_f1 - I_f1
    L_f2 = A_f2 - I_f2
    L_f3 = A_f3 - I_f3
    L_f4 = A_f4 - I_f4
    L_f5 = A_f5 - I_f5
    L_f6 = A_f6 - I_f6
    L_f7 = A_f7 - I_f7
    INV_f0 = I_f0 - L_f0
    INV_f1 = I_f1 - L_f1
    INV_f2 = I_f2 - L_f2
    INV_f3 = I_f3 - L_f3
    INV_f4 = I_f4 - L_f4
    INV_f5 = I_f5 - L_f5
    INV_f6 = I_f6 - L_f6
    INV_f7 = I_f7 - L_f7

    _zf = cutlass.Float32(0.0)
    L_a0 = _ak_pack_bf16x2(L_f0, L_f1)
    L_a1 = _ak_pack_bf16x2(L_f2, L_f3)
    L_a2 = _ak_pack_bf16x2(L_f4, L_f5)
    L_a3 = _ak_pack_bf16x2(L_f6, L_f7)
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 1: L^2, INV += INV*L^2
    L_b0 = _ak_movmatrix_trans(L_a0)
    L_b1 = _ak_movmatrix_trans(L_a1)
    L_b2 = _ak_movmatrix_trans(L_a2)
    L_b3 = _ak_movmatrix_trans(L_a3)
    Lp_c0, Lp_c1, Lp_c2, Lp_c3 = _ak_mma(L_a0, L_a1, L_a2, L_a3, L_b0, L_b1, _zf, _zf, _zf, _zf)
    Lp_c4, Lp_c5, Lp_c6, Lp_c7 = _ak_mma(L_a0, L_a1, L_a2, L_a3, L_b2, L_b3, _zf, _zf, _zf, _zf)
    Lp_a0 = _ak_pack_bf16x2(Lp_c0, Lp_c1)
    Lp_a1 = _ak_pack_bf16x2(Lp_c2, Lp_c3)
    Lp_a2 = _ak_pack_bf16x2(Lp_c4, Lp_c5)
    Lp_a3 = _ak_pack_bf16x2(Lp_c6, Lp_c7)
    Lp_b0 = _ak_movmatrix_trans(Lp_a0)
    Lp_b1 = _ak_movmatrix_trans(Lp_a1)
    Lp_b2 = _ak_movmatrix_trans(Lp_a2)
    Lp_b3 = _ak_movmatrix_trans(Lp_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf)
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf)
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 2: L^4, INV += INV*L^4
    L4_c0, L4_c1, L4_c2, L4_c3 = _ak_mma(Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf)
    L4_c4, L4_c5, L4_c6, L4_c7 = _ak_mma(Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf)
    L4_a0 = _ak_pack_bf16x2(L4_c0, L4_c1)
    L4_a1 = _ak_pack_bf16x2(L4_c2, L4_c3)
    L4_a2 = _ak_pack_bf16x2(L4_c4, L4_c5)
    L4_a3 = _ak_pack_bf16x2(L4_c6, L4_c7)
    L4_b0 = _ak_movmatrix_trans(L4_a0)
    L4_b1 = _ak_movmatrix_trans(L4_a1)
    L4_b2 = _ak_movmatrix_trans(L4_a2)
    L4_b3 = _ak_movmatrix_trans(L4_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf)
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf)
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 3: L^8, INV += INV*L^8
    L8_c0, L8_c1, L8_c2, L8_c3 = _ak_mma(L4_a0, L4_a1, L4_a2, L4_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf)
    L8_c4, L8_c5, L8_c6, L8_c7 = _ak_mma(L4_a0, L4_a1, L4_a2, L4_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf)
    L8_a0 = _ak_pack_bf16x2(L8_c0, L8_c1)
    L8_a1 = _ak_pack_bf16x2(L8_c2, L8_c3)
    L8_a2 = _ak_pack_bf16x2(L8_c4, L8_c5)
    L8_a3 = _ak_pack_bf16x2(L8_c6, L8_c7)
    L8_b0 = _ak_movmatrix_trans(L8_a0)
    L8_b1 = _ak_movmatrix_trans(L8_a1)
    L8_b2 = _ak_movmatrix_trans(L8_a2)
    L8_b3 = _ak_movmatrix_trans(L8_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L8_b0, L8_b1, _zf, _zf, _zf, _zf)
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(INV_a0, INV_a1, INV_a2, INV_a3, L8_b2, L8_b3, _zf, _zf, _zf, _zf)
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7

    sAkk[r_off + gid, c_off + tid] = _ak_pack_bf16x2(INV_f0, INV_f1)
    sAkk[r_off + gid + 8, c_off + tid] = _ak_pack_bf16x2(INV_f2, INV_f3)
    sAkk[r_off + gid, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f4, INV_f5)
    sAkk[r_off + gid + 8, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f6, INV_f7)


@dsl_user_op
def _ak_matmul_AB(sAkk, br_A, bc_A, br_B, bc_B, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 8
    rB = br_B * 16
    cB = bc_B * 8
    a0 = cutlass.Float32(sAkk[rA + gid, cA + tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 4 + tid])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 4 + tid])
    bA0 = cutlass.Float32(sAkk[rB + gid, cB + tid])
    bA1 = cutlass.Float32(sAkk[rB + gid + 8, cB + tid])
    bA2 = cutlass.Float32(sAkk[rB + gid, cB + 4 + tid])
    bA3 = cutlass.Float32(sAkk[rB + gid + 8, cB + 4 + tid])
    b0 = _ak_movmatrix_trans(bA0)
    b1 = _ak_movmatrix_trans(bA1)
    b2 = _ak_movmatrix_trans(bA2)
    b3 = _ak_movmatrix_trans(bA3)
    cn0_0, cn0_1, cn0_2, cn0_3 = _ak_mma(a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf)
    cn1_0, cn1_1, cn1_2, cn1_3 = _ak_mma(a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf)
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _ak_chain_mma_B(sAkk, br_B, bc_B, a0, a1, a2, a3, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rB = br_B * 16
    cB = bc_B * 8
    bA0 = cutlass.Float32(sAkk[rB + gid, cB + tid])
    bA1 = cutlass.Float32(sAkk[rB + gid + 8, cB + tid])
    bA2 = cutlass.Float32(sAkk[rB + gid, cB + 4 + tid])
    bA3 = cutlass.Float32(sAkk[rB + gid + 8, cB + 4 + tid])
    b0 = _ak_movmatrix_trans(bA0)
    b1 = _ak_movmatrix_trans(bA1)
    b2 = _ak_movmatrix_trans(bA2)
    b3 = _ak_movmatrix_trans(bA3)
    cn0_0, cn0_1, cn0_2, cn0_3 = _ak_mma(a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf)
    cn1_0, cn1_1, cn1_2, cn1_3 = _ak_mma(a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf)
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _ak_chain_mma_A(sAkk, br_A, bc_A, b0, b1, b2, b3, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 8
    a0 = cutlass.Float32(sAkk[rA + gid, cA + tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 4 + tid])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 4 + tid])
    cn0_0, cn0_1, cn0_2, cn0_3 = _ak_mma(a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf)
    cn1_0, cn1_1, cn1_2, cn1_3 = _ak_mma(a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf)
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _ak_store_neg_C(sAkk, br, bc, c0, c1, c2, c3, c4, c5, c6, c7, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    r = br * 16
    c = bc * 8
    sAkk[r + gid, c + tid] = _ak_pack_bf16x2(-c0, -c1)
    sAkk[r + gid + 8, c + tid] = _ak_pack_bf16x2(-c2, -c3)
    sAkk[r + gid, c + 4 + tid] = _ak_pack_bf16x2(-c4, -c5)
    sAkk[r + gid + 8, c + 4 + tid] = _ak_pack_bf16x2(-c6, -c7)


@dsl_user_op
def _ak_pack_C_to_A(c0, c1, c2, c3, c4, c5, c6, c7, *, loc=None, ip=None):
    a0 = _ak_pack_bf16x2(c0, c1)
    a1 = _ak_pack_bf16x2(c2, c3)
    a2 = _ak_pack_bf16x2(c4, c5)
    a3 = _ak_pack_bf16x2(c6, c7)
    return a0, a1, a2, a3


@dsl_user_op
def _ak_pack_C_to_B(c0, c1, c2, c3, c4, c5, c6, c7, *, loc=None, ip=None):
    a0 = _ak_pack_bf16x2(c0, c1)
    a1 = _ak_pack_bf16x2(c2, c3)
    a2 = _ak_pack_bf16x2(c4, c5)
    a3 = _ak_pack_bf16x2(c6, c7)
    b0 = _ak_movmatrix_trans(a0)
    b1 = _ak_movmatrix_trans(a1)
    b2 = _ak_movmatrix_trans(a2)
    b3 = _ak_movmatrix_trans(a3)
    return b0, b1, b2, b3


@dsl_user_op
def _ak_store_C_temp(sT, buf, c0, c1, c2, c3, c4, c5, c6, c7, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    sT[gid, 2 * tid, buf] = c0
    sT[gid, 2 * tid + 1, buf] = c1
    sT[gid + 8, 2 * tid, buf] = c2
    sT[gid + 8, 2 * tid + 1, buf] = c3
    sT[gid, 8 + 2 * tid, buf] = c4
    sT[gid, 8 + 2 * tid + 1, buf] = c5
    sT[gid + 8, 8 + 2 * tid, buf] = c6
    sT[gid + 8, 8 + 2 * tid + 1, buf] = c7


@dsl_user_op
def _ak_load_C_temp(sT, buf, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    c0 = cutlass.Float32(sT[gid, 2 * tid, buf])
    c1 = cutlass.Float32(sT[gid, 2 * tid + 1, buf])
    c2 = cutlass.Float32(sT[gid + 8, 2 * tid, buf])
    c3 = cutlass.Float32(sT[gid + 8, 2 * tid + 1, buf])
    c4 = cutlass.Float32(sT[gid, 8 + 2 * tid, buf])
    c5 = cutlass.Float32(sT[gid, 8 + 2 * tid + 1, buf])
    c6 = cutlass.Float32(sT[gid + 8, 8 + 2 * tid, buf])
    c7 = cutlass.Float32(sT[gid + 8, 8 + 2 * tid + 1, buf])
    return c0, c1, c2, c3, c4, c5, c6, c7


@dsl_user_op
def opaque_zero_from_work_id(*, loc=None, ip=None):
    """
    Return 0 via opaque side-effectful asm (no inputs needed).

    Because has_side_effects=True, MLIR LICM treats this as having memory
    effects and will NOT hoist it outside the for_generate loop.  Any value
    computed from the result (_oz) therefore appears loop-variant to LICM,
    preventing get_slice() and scalar layout-invariant computations from being
    hoisted to the kernel prologue.  This keeps prologue register pressure < 64
    and eliminates the 440-byte stack frame that caused ~300-cycle L2 LDL
    penalties per iteration.
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        "mov.b32 $0, 0;",  # output = 0 (opaque to compiler constant-folding)
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@cute.kernel
def intra_kernel(
    tma_atom_Q: cute.CopyAtom,
    tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom,
    tma_tensor_K: cute.Tensor,
    tma_atom_G: cute.CopyAtom,
    tma_tensor_G: cute.Tensor,
    mBeta: cute.Tensor,
    scale: cutlass.Float32,
    mAqk: cute.Tensor,
    mAkk: cute.Tensor,
    mAqk_v2: cute.Tensor,  # 5D vec2 view for coalesced row stores
    mAkk_v2: cute.Tensor,
    tiled_mma_k2,
    tiled_copy_mma_A,
    tiled_copy_mma_B,
    tiled_copy_Gcum_norm,
    tiled_copy_Gcum_gate,
    qk_smem_layout,
    g_cumsum_layout,
    num_chunks: int,
    num_heads: int,
    num_qk_heads: int,
    batch_size: int,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    FUSE_AKK_INV: cutlass.Constexpr[int] = 0,
    FP32_AKK_WORKSPACE: cutlass.Constexpr[int] = 0,
    BETA_FP32: cutlass.Constexpr[int] = 0,
):
    block_id, _, _ = cute.arch.block_idx()
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32

    total_cgs_per_head = cutlass.Int32(0)
    cgs_per_head = cutlass.Int32(0)
    total_cgs = cutlass.Int32(0)
    if IS_VARLEN:
        total_cgs_per_head = (num_chunks + CHUNKS_PER_BLOCK - 1) // CHUNKS_PER_BLOCK
        total_cgs = total_cgs_per_head * num_heads
    else:
        cgs_per_head = num_chunks // CHUNKS_PER_BLOCK
        total_cgs = cgs_per_head * num_heads * batch_size

    # =====================================================================
    # SMEM allocation
    # =====================================================================
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner)
    sK = smem.allocate_tensor(cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner)
    sGcum = smem.allocate_tensor(cutlass.Float32, g_cumsum_layout.outer, 128, swizzle=g_cumsum_layout.inner)

    # sAqk: 64x72 row-major (BT rows, BT+pad cols), same shape as sAkk.
    # Each sub-tile (i_q, i_k) sits at SMEM rows [i_q*BC, (i_q+1)*BC) cols
    # [i_k*BC, (i_k+1)*BC) — directly mirrors the 64x64 attention matrix.
    aqk_tile_layout = cute.make_layout((BT, AQK_TILE_STRIDE, NUM_STAGES), stride=(AQK_TILE_STRIDE, 1, BT * AQK_TILE_STRIDE))
    sAqk = smem.allocate_tensor(cutlass.BFloat16, aqk_tile_layout, 128)

    # Akk FP32 with stride=32 (4-way bank conflict vs 16-way with stride=64)
    # shape [64, 32, 2] - stores two 64x32 sub-tiles, 64x64 Akk read via two MMAs
    akk_tile_layout = cute.make_layout((BT, AKK_STRIDE, NUM_STAGES), stride=(AKK_STRIDE, 1, BT * AKK_STRIDE))
    sAkk = smem.allocate_tensor(cutlass.Float32, akk_tile_layout, 128)

    # The TMA warp loads each beta value once; all ten MMA tile warps reuse it.
    beta_smem_layout = cute.make_layout((BT, NUM_STAGES), stride=(1, BT))
    if cutlass.const_expr(BETA_FP32 != 0):
        sBeta = smem.allocate_tensor(cutlass.Float32, beta_smem_layout, 128)
    else:
        sBeta = smem.allocate_tensor(cutlass.BFloat16, beta_smem_layout, 128)

    # =====================================================================
    # Mbarrier allocation & init
    # =====================================================================
    tma_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    stage_reuse_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    input_ready_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    mma_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    store_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)

    bytes_per_stage = BT * K_DIM * (2 + 2 + 4)

    if tidx == 0:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_init(tma_mbars + s, 1)
            cute.arch.mbarrier_init(stage_reuse_mbars + s, NUM_MMA_ACTIVE * 32)
            cute.arch.mbarrier_init(input_ready_mbars + s, 32)
            cute.arch.mbarrier_init(mma_done_mbars + s, NUM_MMA_ACTIVE * 32)
            cute.arch.mbarrier_init(store_done_mbars + s, NUM_STORE_WARPS * 32)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # =====================================================================
    # SMEM init: zero out sAqk and sAkk valid 64x64 region (cols 64..71 are
    # padding for SMEM bank-conflict avoidance, never read or written by MMA
    # or store warps). Required for downstream row-major store optimizations
    # — positions outside MMA-written sub-tiles stay at 0.
    #
    # Cooperative pattern: each active warp owns a contiguous row group.
    #   - For sAqk bf16: each lane owns 2 contiguous bf16 cols
    #   - For sAkk fp32: each lane owns 1 fp32 col (stride 68)
    # =====================================================================
    _warp_id_in_cta = tidx >> 5  # tidx // 32, range 0..31
    _lane_id_warp = tidx & 31  # tidx % 32, range 0..31
    _rows_per_init_warp = (BT + NUM_WARPS - 1) // NUM_WARPS
    _row_base = _warp_id_in_cta * _rows_per_init_warp
    for _s in cutlass.range_constexpr(NUM_STAGES):
        for _ri in cutlass.range_constexpr(_rows_per_init_warp):
            _row = _row_base + _ri
            if _row < BT:
                _col_lo_bf16 = _lane_id_warp * 2
                _col_hi_bf16 = _col_lo_bf16 + 1
                sAqk[_row, _col_lo_bf16, _s] = cutlass.BFloat16(0.0)
                sAqk[_row, _col_hi_bf16, _s] = cutlass.BFloat16(0.0)
                sAkk[_row, _lane_id_warp, _s] = cutlass.Float32(0.0)
    cute.arch.barrier()

    # =====================================================================
    # Pre-arrive all MMA readers so the first producer use of each stage does
    # not wait.
    # store_done_mbars:  MMA waits before Store arrives → pre-arrive first 4 MMA warps
    # =====================================================================
    if warp_idx < NUM_MMA_ACTIVE:
        mma_warp_tmp = warp_idx
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_arrive(stage_reuse_mbars + s)
            if mma_warp_tmp < NUM_STORE_WARPS:
                cute.arch.mbarrier_arrive(store_done_mbars + s)

    # =================================================================
    # Persistent outer loop. Single for_generate at top level (required).
    # Opaque asm barrier on work_id prevents MLIR LICM from hoisting
    # get_slice() and scalar layout invariants to the kernel prologue,
    # keeping register pressure < 64 and eliminating prologue spill.
    # =================================================================
    for work_id in for_generate(block_id, total_cgs, NUM_SMS):
        i_cg = cutlass.Int32(0)
        i_h = cutlass.Int32(0)
        i_b = cutlass.Int32(0)
        chunk_base = cutlass.Int32(0)
        if IS_VARLEN:
            i_cg = work_id % total_cgs_per_head
            i_h = work_id // total_cgs_per_head
            i_b = cutlass.Int32(0)
            chunk_base = i_cg * CHUNKS_PER_BLOCK
        else:
            i_cg = work_id % cgs_per_head
            i_h = (work_id // cgs_per_head) % num_heads
            i_b = work_id // (cgs_per_head * num_heads)
            chunk_base = i_cg * CHUNKS_PER_BLOCK
        # Tiles are scheduled in V-head space. Under GVA, a consecutive
        # group of V heads shares one Q/K head, matching the csrc scheduler.
        i_h_qk = i_h // (num_heads // num_qk_heads)

        # Anti-LICM barrier: _oz is always 0 but appears to depend on work_id.
        # Because this asm has side_effects=True, it stays inside the loop.
        # Any value computed from _oz/_lane/_warp is also loop-variant from
        # LICM's perspective → get_slice() and scalar invariants stay in-loop.
        _oz = opaque_zero_from_work_id()
        _lane = lane_id + _oz
        _warp = warp_idx + _oz

        # =============================================================
        # Warp 10: dedicated TMA producer.
        # Waits stage_reuse (gated by MMA arrives), issues TMA for Q/K/G,
        # signals tma_mbar and input_ready.
        # =============================================================
        if warp_idx == TMA_WARP_ID:
            gQ_head = tma_tensor_Q[(None, None, i_h_qk)]
            gK_head = tma_tensor_K[(None, None, i_h_qk)]
            gG_head = tma_tensor_G[(None, None, i_h)]

            # Prefetch chunk 0 -> stage 0 (stage_reuse[0] pre-arrived)
            pf_cs = cutlass.Int32(0)
            pf_eos = cutlass.Int32(BT)
            if IS_VARLEN:
                pf_seq_id_0 = cutlass.Int32(mChunkIndices[chunk_base, 0])
                pf_local_0 = cutlass.Int32(mChunkIndices[chunk_base, 1])
                pf_bos_0 = cutlass.Int32(mCuSeqlens[pf_seq_id_0])
                pf_eos = cutlass.Int32(mCuSeqlens[pf_seq_id_0 + 1])
                pf_cs = pf_bos_0 + pf_local_0 * BT
            else:
                pf_cs = i_b * num_chunks * BT + chunk_base * BT
            cute.arch.mbarrier_wait(stage_reuse_mbars, 0)
            if lane_id == 0:
                cute.arch.mbarrier_expect_tx(tma_mbars, bytes_per_stage)
            sQ_pf = sQ[(None, None, 0)]
            gQ_pf = cute.local_tile(cute.domain_offset((pf_cs, 0), gQ_head), (BT, K_DIM), (0, 0))
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_pf, 0, 2), cute.group_modes(gQ_pf, 0, 2)
            )
            cute.copy(tma_atom_Q, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            sK_pf = sK[(None, None, 0)]
            gK_pf = cute.local_tile(cute.domain_offset((pf_cs, 0), gK_head), (BT, K_DIM), (0, 0))
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK_pf, 0, 2), cute.group_modes(gK_pf, 0, 2)
            )
            cute.copy(tma_atom_K, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            sG_pf = sGcum[(None, None, 0)]
            gG_pf = cute.local_tile(cute.domain_offset((pf_cs, 0), gG_head), (BT, K_DIM), (0, 0))
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_G, 0, cute.make_layout(1), cute.group_modes(sG_pf, 0, 2), cute.group_modes(gG_pf, 0, 2)
            )
            cute.copy(tma_atom_G, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            if lane_id == 0:
                cute.arch.mbarrier_arrive(tma_mbars)
            sBeta_pf = sBeta[(None, 0)]
            beta_cs = pf_cs
            if not IS_VARLEN:
                beta_cs = chunk_base * BT
            for beta_iter in cutlass.range_constexpr(BT // 32):
                beta_row = beta_iter * 32 + lane_id
                beta_load_row = beta_cs + beta_row
                if IS_VARLEN:
                    if beta_load_row >= pf_eos:
                        beta_load_row = pf_eos - 1
                sBeta_pf[beta_row] = mBeta[i_b, beta_load_row, i_h]
            cute.arch.mbarrier_arrive(input_ready_mbars)

            # Issue TMAs for chunks 1..CHUNKS_PER_BLOCK-1
            for next_i in cutlass.range_constexpr(1, CHUNKS_PER_BLOCK):
                next_stage = next_i % NUM_STAGES
                next_phase = next_i // NUM_STAGES % 2
                next_cs = cutlass.Int32(0)
                next_eos = cutlass.Int32(BT)
                if IS_VARLEN:
                    next_eos = cutlass.Int32(1)
                    next_chunk_idx = chunk_base + next_i
                    if next_chunk_idx < num_chunks:
                        _nsid = cutlass.Int32(mChunkIndices[next_chunk_idx, 0])
                        next_cs = cutlass.Int32(mCuSeqlens[_nsid]) + cutlass.Int32(mChunkIndices[next_chunk_idx, 1]) * BT
                        next_eos = cutlass.Int32(mCuSeqlens[_nsid + 1])
                else:
                    next_cs = i_b * num_chunks * BT + (chunk_base + next_i) * BT
                cute.arch.mbarrier_wait(stage_reuse_mbars + next_stage, next_phase)
                if lane_id == 0:
                    cute.arch.mbarrier_expect_tx(tma_mbars + next_stage, bytes_per_stage)
                sQ_ns = sQ[(None, None, next_stage)]
                gQ_ns = cute.local_tile(cute.domain_offset((next_cs, 0), gQ_head), (BT, K_DIM), (0, 0))
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_ns, 0, 2), cute.group_modes(gQ_ns, 0, 2)
                )
                cute.copy(tma_atom_Q, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                sK_ns = sK[(None, None, next_stage)]
                gK_ns = cute.local_tile(cute.domain_offset((next_cs, 0), gK_head), (BT, K_DIM), (0, 0))
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_K, 0, cute.make_layout(1), cute.group_modes(sK_ns, 0, 2), cute.group_modes(gK_ns, 0, 2)
                )
                cute.copy(tma_atom_K, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                sG_ns = sGcum[(None, None, next_stage)]
                gG_ns = cute.local_tile(cute.domain_offset((next_cs, 0), gG_head), (BT, K_DIM), (0, 0))
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_G, 0, cute.make_layout(1), cute.group_modes(sG_ns, 0, 2), cute.group_modes(gG_ns, 0, 2)
                )
                cute.copy(tma_atom_G, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                if lane_id == 0:
                    cute.arch.mbarrier_arrive(tma_mbars + next_stage)
                sBeta_ns = sBeta[(None, next_stage)]
                beta_cs = next_cs
                if not IS_VARLEN:
                    beta_cs = (chunk_base + next_i) * BT
                for beta_iter in cutlass.range_constexpr(BT // 32):
                    beta_row = beta_iter * 32 + lane_id
                    beta_load_row = beta_cs + beta_row
                    if IS_VARLEN:
                        if beta_load_row >= next_eos:
                            beta_load_row = next_eos - 1
                    sBeta_ns[beta_row] = mBeta[i_b, beta_load_row, i_h]
                cute.arch.mbarrier_arrive(input_ready_mbars + next_stage)

        # =============================================================
        # Warps 0-9: K2/K3 MMA compute
        # =============================================================
        if warp_idx < NUM_MMA_ACTIVE:
            # Warp-layout invariants (scope-local → no cross-group register spill)
            _tid_in_group = _lane % 4
            _group_id = _lane // 4
            mma_warp = _warp
            my_i_q = cutlass.Int32(0)
            my_i_k = cutlass.Int32(0)
            if mma_warp < 1:
                my_i_q = cutlass.Int32(0)
                my_i_k = mma_warp
            elif mma_warp < 3:
                my_i_q = cutlass.Int32(1)
                my_i_k = mma_warp - 1
            elif mma_warp < 6:
                my_i_q = cutlass.Int32(2)
                my_i_k = mma_warp - 3
            elif mma_warp < NUM_MMA_ACTIVE:
                my_i_q = cutlass.Int32(3)
                my_i_k = mma_warp - 6
            q_row_base = my_i_q * BC
            k_row_base = my_i_k * BC
            akk_row_base = k_row_base
            akk_col_base = q_row_base
            norm_row = q_row_base
            if my_i_q == my_i_k:
                norm_row = q_row_base + cutlass.Int32(BC // 2)
            row0 = _group_id
            row1 = _group_id + 8
            col0 = _tid_in_group * 2
            col1 = _tid_in_group * 2 + 1
            col2 = 8 + _tid_in_group * 2
            col3 = 8 + _tid_in_group * 2 + 1
            thr_mma = tiled_mma_k2.get_slice(_lane)
            thr_copy_A = tiled_copy_mma_A.get_slice(_lane)
            thr_copy_B = tiled_copy_mma_B.get_slice(_lane)
            thr_copy_Gn = tiled_copy_Gcum_norm.get_slice(_tid_in_group)
            thr_copy_Ggate = tiled_copy_Gcum_gate.get_slice(_lane)

            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                s = chunk_iter % NUM_STAGES
                phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                valid_rows = cutlass.Int32(BT)
                if IS_VARLEN:
                    valid_rows = cutlass.Int32(1)
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = cutlass.Int32(mCuSeqlens[_sid]) + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        _mma_eos = cutlass.Int32(mCuSeqlens[_sid + 1])
                        valid_rows = _mma_eos - chunk_start
                        if valid_rows > BT:
                            valid_rows = cutlass.Int32(BT)
                else:
                    chunk_start = chunk_idx * BT

                # csrc clamps each 16-row gate reference to the final valid
                # row of a partial chunk. Reading the nominal row (8/16/32/48)
                # changes the gated K operands at a tail and can move FP32-beta
                # Akk by one BF16 ULP after inversion.
                norm_row_cur = norm_row
                if IS_VARLEN:
                    if norm_row_cur >= valid_rows:
                        norm_row_cur = valid_rows - 1

                cute.arch.mbarrier_wait(tma_mbars + s, phase)
                cute.arch.mbarrier_wait(input_ready_mbars + s, phase)
                cute.arch.mbarrier_wait(store_done_mbars + s, phase)

                if mma_warp < NUM_MMA_ACTIVE:
                    csQ = sQ[(None, None, s)]
                    csK = sK[(None, None, s)]
                    csGcum = sGcum[(None, None, s)]
                    csAqk = sAqk[(None, None, s)]
                    csAkk = sAkk[(None, None, s)]
                    csBeta = sBeta[(None, s)]

                    _z = cutlass.Float32(0.0)

                    beta_row0 = csBeta[q_row_base + row0].to(cutlass.Float32)
                    beta_row1 = csBeta[q_row_base + row1].to(cutlass.Float32)

                    acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = _z, _z, _z, _z
                    acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = _z, _z, _z, _z
                    acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = _z, _z, _z, _z
                    acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = _z, _z, _z, _z

                    # bf16 m16n8k16 MMA: each k_block covers k=16.
                    for k_block in cutlass.range_constexpr(NUM_MMA_K_TILES):
                        # ---- Load Q/Kq bf16 fragments (16x16, 8 bf16/thread) ----
                        sQ_tile = cute.local_tile(csQ, tiler=(16, 16), coord=(my_i_q, k_block))
                        tCrQ = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sQ_tile))
                        cute.copy(tiled_copy_mma_A, thr_copy_A.partition_S(sQ_tile), thr_copy_A.retile(tCrQ))

                        sKq_tile = cute.local_tile(csK, tiler=(16, 16), coord=(my_i_q, k_block))
                        tCrKq = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sKq_tile))
                        cute.copy(tiled_copy_mma_A, thr_copy_A.partition_S(sKq_tile), thr_copy_A.retile(tCrKq))

                        # ---- Issue K n0/n1 LDSMs early for better ILP ----
                        sK_tile_n0 = cute.local_tile(csK, tiler=(8, 16), coord=(my_i_k * 2, k_block))
                        tCrK_n0 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n0))
                        cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n0), thr_copy_B.retile(tCrK_n0))

                        sK_tile_n1 = cute.local_tile(csK, tiler=(8, 16), coord=(my_i_k * 2 + 1, k_block))
                        tCrK_n1 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n1))
                        cute.copy(tiled_copy_mma_B, thr_copy_B.partition_S(sK_tile_n1), thr_copy_B.retile(tCrK_n1))

                        # ---- Gate norm (2x k=8 covers k=16) ----
                        sGn_a = cute.local_tile(csGcum, tiler=(1, 8), coord=(norm_row_cur, k_block * 2))
                        tCsGn_a = thr_copy_Gn.partition_S(sGn_a)
                        tCrGn_a = cute.make_fragment_like(tCsGn_a, cutlass.Float32)
                        cute.copy(tiled_copy_Gcum_norm, tCsGn_a, thr_copy_Gn.retile(tCrGn_a))
                        gn_a0 = tCrGn_a[0]
                        gn_a1 = tCrGn_a[1]

                        sGn_b = cute.local_tile(csGcum, tiler=(1, 8), coord=(norm_row_cur, k_block * 2 + 1))
                        tCsGn_b = thr_copy_Gn.partition_S(sGn_b)
                        tCrGn_b = cute.make_fragment_like(tCsGn_b, cutlass.Float32)
                        cute.copy(tiled_copy_Gcum_norm, tCsGn_b, thr_copy_Gn.retile(tCrGn_b))
                        gn_b0 = tCrGn_b[0]
                        gn_b1 = tCrGn_b[1]

                        # ---- Gate Q (2x (16,8) partition_C covers m=16,k=16) ----
                        sGq_a = cute.local_tile(csGcum, tiler=(16, 8), coord=(my_i_q, k_block * 2))
                        tCrGq_a = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGq_a))
                        cute.copy(tiled_copy_Gcum_gate, thr_copy_Ggate.partition_S(sGq_a), thr_copy_Ggate.retile(tCrGq_a))

                        sGq_b = cute.local_tile(csGcum, tiler=(16, 8), coord=(my_i_q, k_block * 2 + 1))
                        tCrGq_b = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGq_b))
                        cute.copy(tiled_copy_Gcum_gate, thr_copy_Ggate.partition_S(sGq_b), thr_copy_Ggate.retile(tCrGq_b))

                        # 8 Q gate values per thread (matching A bf16 m16n8k16 layout):
                        # first half k=0..7 (tCrGq_a): a0=(r0,c0) a1=(r0,c0+1) a2=(r0+8,c0) a3=(r0+8,c0+1)
                        # second half k=8..15 (tCrGq_b): a4=(r0,c0+8) a5=(r0,c0+9) a6=(r0+8,c0+8) a7=(r0+8,c0+9)
                        gate_q_0 = cute.exp2(tCrGq_a[0] - gn_a0, fastmath=True)
                        gate_q_1 = cute.exp2(tCrGq_a[1] - gn_a1, fastmath=True)
                        gate_q_2 = cute.exp2(tCrGq_a[2] - gn_a0, fastmath=True)
                        gate_q_3 = cute.exp2(tCrGq_a[3] - gn_a1, fastmath=True)
                        gate_q_4 = cute.exp2(tCrGq_b[0] - gn_b0, fastmath=True)
                        gate_q_5 = cute.exp2(tCrGq_b[1] - gn_b1, fastmath=True)
                        gate_q_6 = cute.exp2(tCrGq_b[2] - gn_b0, fastmath=True)
                        gate_q_7 = cute.exp2(tCrGq_b[3] - gn_b1, fastmath=True)

                        # qa fp32 = Q*gate (8 per thread). tCrQ indexing assumption:
                        # [0..3] first k-chunk (k=0..7), [4..7] second k-chunk (k=8..15).
                        qa0 = tCrQ[0].to(cutlass.Float32) * gate_q_0
                        qa1 = tCrQ[1].to(cutlass.Float32) * gate_q_1
                        qa2 = tCrQ[2].to(cutlass.Float32) * gate_q_2
                        qa3 = tCrQ[3].to(cutlass.Float32) * gate_q_3
                        qa4 = tCrQ[4].to(cutlass.Float32) * gate_q_4
                        qa5 = tCrQ[5].to(cutlass.Float32) * gate_q_5
                        qa6 = tCrQ[6].to(cutlass.Float32) * gate_q_6
                        qa7 = tCrQ[7].to(cutlass.Float32) * gate_q_7

                        ka0 = tCrKq[0].to(cutlass.Float32) * gate_q_0
                        ka1 = tCrKq[1].to(cutlass.Float32) * gate_q_1
                        ka2 = tCrKq[2].to(cutlass.Float32) * gate_q_2
                        ka3 = tCrKq[3].to(cutlass.Float32) * gate_q_3
                        ka4 = tCrKq[4].to(cutlass.Float32) * gate_q_4
                        ka5 = tCrKq[5].to(cutlass.Float32) * gate_q_5
                        ka6 = tCrKq[6].to(cutlass.Float32) * gate_q_6
                        ka7 = tCrKq[7].to(cutlass.Float32) * gate_q_7

                        # ---- Gate K (2x (16,8) partition_C covers m=16,k=16) ----
                        sGk_a = cute.local_tile(csGcum, tiler=(16, 8), coord=(my_i_k, k_block * 2))
                        tCrGk_a = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGk_a))
                        cute.copy(tiled_copy_Gcum_gate, thr_copy_Ggate.partition_S(sGk_a), thr_copy_Ggate.retile(tCrGk_a))

                        sGk_b = cute.local_tile(csGcum, tiler=(16, 8), coord=(my_i_k, k_block * 2 + 1))
                        tCrGk_b = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGk_b))
                        cute.copy(tiled_copy_Gcum_gate, thr_copy_Ggate.partition_S(sGk_b), thr_copy_Ggate.retile(tCrGk_b))

                        # n0 uses rows 0..7 of (16,*) tile (tCrGk_*[0,1])
                        # n1 uses rows 8..15 of (16,*) tile (tCrGk_*[2,3])
                        gk_n0_0 = cute.exp2(gn_a0 - tCrGk_a[0], fastmath=True)
                        gk_n0_1 = cute.exp2(gn_a1 - tCrGk_a[1], fastmath=True)
                        gk_n0_2 = cute.exp2(gn_b0 - tCrGk_b[0], fastmath=True)
                        gk_n0_3 = cute.exp2(gn_b1 - tCrGk_b[1], fastmath=True)

                        gk_n1_0 = cute.exp2(gn_a0 - tCrGk_a[2], fastmath=True)
                        gk_n1_1 = cute.exp2(gn_a1 - tCrGk_a[3], fastmath=True)
                        gk_n1_2 = cute.exp2(gn_b0 - tCrGk_b[2], fastmath=True)
                        gk_n1_3 = cute.exp2(gn_b1 - tCrGk_b[3], fastmath=True)

                        # tCrK_n0 bf16 fragment: 4 elems/thread at (n_row, c0), (n_row, c0+1),
                        # (n_row, c0+8), (n_row, c0+9) — k-adjacent pairs
                        k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                        k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1
                        k_n0_b2 = tCrK_n0[2].to(cutlass.Float32) * gk_n0_2
                        k_n0_b3 = tCrK_n0[3].to(cutlass.Float32) * gk_n0_3

                        k_n1_b0 = tCrK_n1[0].to(cutlass.Float32) * gk_n1_0
                        k_n1_b1 = tCrK_n1[1].to(cutlass.Float32) * gk_n1_1
                        k_n1_b2 = tCrK_n1[2].to(cutlass.Float32) * gk_n1_2
                        k_n1_b3 = tCrK_n1[3].to(cutlass.Float32) * gk_n1_3

                        # TF32 m16n8k8 consumes one k8 half per call.  The
                        # loaded BF16-fragment value order is (r0,k0),
                        # (r0,k1), (r8,k0), (r8,k1), so reorder A to the
                        # register MMA order used by kda_fuse_k4.py:
                        # (r0,k0), (r8,k0), (r0,k1), (r8,k1).
                        acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = mma_tf32_m16n8k8(
                            qa0, qa2, qa1, qa3, k_n0_b0, k_n0_b1, acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3
                        )
                        acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = mma_tf32_m16n8k8(
                            qa4, qa6, qa5, qa7, k_n0_b2, k_n0_b3, acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3
                        )

                        acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = mma_tf32_m16n8k8(
                            qa0, qa2, qa1, qa3, k_n1_b0, k_n1_b1, acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3
                        )
                        acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = mma_tf32_m16n8k8(
                            qa4, qa6, qa5, qa7, k_n1_b2, k_n1_b3, acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3
                        )

                        acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = mma_tf32_m16n8k8(
                            ka0, ka2, ka1, ka3, k_n0_b0, k_n0_b1, acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3
                        )
                        acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = mma_tf32_m16n8k8(
                            ka4, ka6, ka5, ka7, k_n0_b2, k_n0_b3, acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3
                        )

                        acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = mma_tf32_m16n8k8(
                            ka0, ka2, ka1, ka3, k_n1_b0, k_n1_b1, acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3
                        )
                        acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = mma_tf32_m16n8k8(
                            ka4, ka6, ka5, ka7, k_n1_b2, k_n1_b3, acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3
                        )

                    # sQ/sK/sG reads done, signal TMA before SMEM writes
                    cute.arch.mbarrier_arrive(stage_reuse_mbars + s)

                    # Varlen writes apply causal/diagonal masks here so partial
                    # chunks are final in SMEM. Equal-length writes all values;
                    # the row-major store masks the upper triangle.
                    _z16 = cutlass.BFloat16(0.0)
                    _one16 = cutlass.BFloat16(1.0)
                    _z32 = cutlass.Float32(0.0)
                    _one32 = cutlass.Float32(1.0)
                    if IS_VARLEN:
                        if my_i_q == my_i_k:
                            # Diagonal sub-tile: causal-mask sAqk and write
                            # diag=1 / strict-lower=MMA*beta / strict-upper=0
                            # to sAkk so SMEM is in final form.
                            _v_q00 = (acc_aqk_n0_0 * scale).to(cutlass.BFloat16)
                            if row0 < col0:
                                _v_q00 = _z16
                            csAqk[q_row_base + row0, k_row_base + col0] = _v_q00
                            _v_q01 = (acc_aqk_n0_1 * scale).to(cutlass.BFloat16)
                            if row0 < col1:
                                _v_q01 = _z16
                            csAqk[q_row_base + row0, k_row_base + col1] = _v_q01
                            _v_q02 = (acc_aqk_n0_2 * scale).to(cutlass.BFloat16)
                            if row1 < col0:
                                _v_q02 = _z16
                            csAqk[q_row_base + row1, k_row_base + col0] = _v_q02
                            _v_q03 = (acc_aqk_n0_3 * scale).to(cutlass.BFloat16)
                            if row1 < col1:
                                _v_q03 = _z16
                            csAqk[q_row_base + row1, k_row_base + col1] = _v_q03
                            _v_q04 = (acc_aqk_n1_0 * scale).to(cutlass.BFloat16)
                            if row0 < col2:
                                _v_q04 = _z16
                            csAqk[q_row_base + row0, k_row_base + col2] = _v_q04
                            _v_q05 = (acc_aqk_n1_1 * scale).to(cutlass.BFloat16)
                            if row0 < col3:
                                _v_q05 = _z16
                            csAqk[q_row_base + row0, k_row_base + col3] = _v_q05
                            _v_q06 = (acc_aqk_n1_2 * scale).to(cutlass.BFloat16)
                            if row1 < col2:
                                _v_q06 = _z16
                            csAqk[q_row_base + row1, k_row_base + col2] = _v_q06
                            _v_q07 = (acc_aqk_n1_3 * scale).to(cutlass.BFloat16)
                            if row1 < col3:
                                _v_q07 = _z16
                            csAqk[q_row_base + row1, k_row_base + col3] = _v_q07

                            if cutlass.const_expr(FP32_AKK_WORKSPACE != 0 or FUSE_AKK_INV != 0):
                                _v_k00 = acc_akk_n0_0 * beta_row0
                                if row0 == col0:
                                    _v_k00 = _one32
                                if row0 < col0:
                                    _v_k00 = _z32
                                csAkk[akk_row_base + row0, akk_col_base + col0] = _v_k00
                            else:
                                _v_k00 = (acc_akk_n0_0 * beta_row0).to(cutlass.BFloat16)
                                if row0 == col0:
                                    _v_k00 = _one16
                                if row0 < col0:
                                    _v_k00 = _z16
                                csAkk[akk_row_base + row0, akk_col_base + col0] = _v_k00.to(cutlass.Float32)  # BF16 -> FP32
                            _v_k01 = acc_akk_n0_1 * beta_row0  # Keep FP32
                            if row0 == col1:
                                _v_k01 = _one32
                            if row0 < col1:
                                _v_k01 = _z32
                            csAkk[akk_row_base + row0, akk_col_base + col1] = _v_k01
                            _v_k02 = acc_akk_n0_2 * beta_row1  # Keep FP32
                            if row1 == col0:
                                _v_k02 = _one32
                            if row1 < col0:
                                _v_k02 = _z32
                            csAkk[akk_row_base + row1, akk_col_base + col0] = _v_k02
                            _v_k03 = acc_akk_n0_3 * beta_row1  # Keep FP32
                            if row1 == col1:
                                _v_k03 = _one32
                            if row1 < col1:
                                _v_k03 = _z32
                            csAkk[akk_row_base + row1, akk_col_base + col1] = _v_k03
                            _v_k04 = acc_akk_n1_0 * beta_row0  # Keep FP32
                            if row0 == col2:
                                _v_k04 = _one32
                            if row0 < col2:
                                _v_k04 = _z32
                            csAkk[akk_row_base + row0, akk_col_base + col2] = _v_k04
                            _v_k05 = acc_akk_n1_1 * beta_row0  # Keep FP32
                            if row0 == col3:
                                _v_k05 = _one32
                            if row0 < col3:
                                _v_k05 = _z32
                            csAkk[akk_row_base + row0, akk_col_base + col3] = _v_k05
                            _v_k06 = acc_akk_n1_2 * beta_row1  # Keep FP32
                            if row1 == col2:
                                _v_k06 = _one32
                            if row1 < col2:
                                _v_k06 = _z32
                            csAkk[akk_row_base + row1, akk_col_base + col2] = _v_k06
                            _v_k07 = acc_akk_n1_3 * beta_row1  # Keep FP32
                            if row1 == col3:
                                _v_k07 = _one32
                            if row1 < col3:
                                _v_k07 = _z32
                            csAkk[akk_row_base + row1, akk_col_base + col3] = _v_k07
                        else:
                            # Non-diag (i_q > i_k): write all 16x16 unchanged.
                            csAqk[q_row_base + row0, k_row_base + col0] = (acc_aqk_n0_0 * scale).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col1] = (acc_aqk_n0_1 * scale).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col0] = (acc_aqk_n0_2 * scale).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col1] = (acc_aqk_n0_3 * scale).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col2] = (acc_aqk_n1_0 * scale).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col3] = (acc_aqk_n1_1 * scale).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col2] = (acc_aqk_n1_2 * scale).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col3] = (acc_aqk_n1_3 * scale).to(cutlass.BFloat16)
                            if cutlass.const_expr(FP32_AKK_WORKSPACE != 0 or FUSE_AKK_INV != 0):
                                csAkk[akk_row_base + row0, akk_col_base + col0] = acc_akk_n0_0 * beta_row0
                                csAkk[akk_row_base + row0, akk_col_base + col1] = acc_akk_n0_1 * beta_row0
                                csAkk[akk_row_base + row1, akk_col_base + col0] = acc_akk_n0_2 * beta_row1
                                csAkk[akk_row_base + row1, akk_col_base + col1] = acc_akk_n0_3 * beta_row1
                                csAkk[akk_row_base + row0, akk_col_base + col2] = acc_akk_n1_0 * beta_row0
                                csAkk[akk_row_base + row0, akk_col_base + col3] = acc_akk_n1_1 * beta_row0
                                csAkk[akk_row_base + row1, akk_col_base + col2] = acc_akk_n1_2 * beta_row1
                                csAkk[akk_row_base + row1, akk_col_base + col3] = acc_akk_n1_3 * beta_row1
                            else:
                                csAkk[akk_row_base + row0, akk_col_base + col0] = (acc_akk_n0_0 * beta_row0).to(
                                    cutlass.BFloat16
                                )
                                csAkk[akk_row_base + row0, akk_col_base + col1] = (acc_akk_n0_1 * beta_row0).to(
                                    cutlass.BFloat16
                                )
                                csAkk[akk_row_base + row1, akk_col_base + col0] = (acc_akk_n0_2 * beta_row1).to(
                                    cutlass.BFloat16
                                )
                                csAkk[akk_row_base + row1, akk_col_base + col1] = (acc_akk_n0_3 * beta_row1).to(
                                    cutlass.BFloat16
                                )
                                csAkk[akk_row_base + row0, akk_col_base + col2] = (acc_akk_n1_0 * beta_row0).to(
                                    cutlass.BFloat16
                                )
                                csAkk[akk_row_base + row0, akk_col_base + col3] = (acc_akk_n1_1 * beta_row0).to(
                                    cutlass.BFloat16
                                )
                                csAkk[akk_row_base + row1, akk_col_base + col2] = (acc_akk_n1_2 * beta_row1).to(
                                    cutlass.BFloat16
                                )
                                csAkk[akk_row_base + row1, akk_col_base + col3] = (acc_akk_n1_3 * beta_row1).to(
                                    cutlass.BFloat16
                                )
                    else:
                        # Equal-length path: write all 16x16 values and apply
                        # the upper-triangle mask in the row-major store.
                        csAqk[q_row_base + row0, k_row_base + col0] = (acc_aqk_n0_0 * scale).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col1] = (acc_aqk_n0_1 * scale).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col0] = (acc_aqk_n0_2 * scale).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col1] = (acc_aqk_n0_3 * scale).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col2] = (acc_aqk_n1_0 * scale).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col3] = (acc_aqk_n1_1 * scale).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col2] = (acc_aqk_n1_2 * scale).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col3] = (acc_aqk_n1_3 * scale).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col0] = acc_akk_n0_0 * beta_row0  # Keep FP32
                        csAkk[akk_row_base + row0, akk_col_base + col1] = acc_akk_n0_1 * beta_row0  # Keep FP32
                        csAkk[akk_row_base + row1, akk_col_base + col0] = acc_akk_n0_2 * beta_row1  # Keep FP32
                        csAkk[akk_row_base + row1, akk_col_base + col1] = acc_akk_n0_3 * beta_row1  # Keep FP32
                        csAkk[akk_row_base + row0, akk_col_base + col2] = acc_akk_n1_0 * beta_row0  # Keep FP32
                        csAkk[akk_row_base + row0, akk_col_base + col3] = acc_akk_n1_1 * beta_row0  # Keep FP32
                        csAkk[akk_row_base + row1, akk_col_base + col2] = acc_akk_n1_2 * beta_row1  # Keep FP32
                        csAkk[akk_row_base + row1, akk_col_base + col3] = acc_akk_n1_3 * beta_row1  # Keep FP32
                else:
                    cute.arch.mbarrier_arrive(stage_reuse_mbars + s)

                cute.arch.mbarrier_arrive(mma_done_mbars + s)

        # =============================================================
        # Warps 11-14: Store warps / optional in-CTA inverse workers
        # =============================================================
        if warp_idx >= STORE_WARP_BASE and warp_idx < STORE_WARP_BASE + NUM_STORE_WARPS:
            store_warp = warp_idx - STORE_WARP_BASE
            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                s = chunk_iter % NUM_STAGES
                phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                st_eos = cutlass.Int32(0)
                if IS_VARLEN:
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = cutlass.Int32(mCuSeqlens[_sid]) + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        st_eos = cutlass.Int32(mCuSeqlens[_sid + 1])
                else:
                    chunk_start = chunk_idx * BT
                    st_eos = chunk_start + BT

                cute.arch.mbarrier_wait(mma_done_mbars + s, phase)

                csAqk = sAqk[(None, None, s)]
                csAkk = sAkk[(None, None, s)]

                # mAqk_v2 / mAkk_v2 shape (B, T, H, BT/2, 2): last dim is vec2.
                col_lo = lane_id * 2
                col_hi = col_lo + 1
                col_vec_idx = lane_id
                rAqkOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
                if cutlass.const_expr(FP32_AKK_WORKSPACE != 0):
                    rAkkFp32Out = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
                else:
                    rAkkOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)

                # Row-major full-row vec autovec + row mask. SMEM upper-tri is
                # zero (MMA-masked at write), so writing all 64 cols is correct.
                row_base_warp = store_warp * (BT // NUM_STORE_WARPS)
                for ri in cutlass.range_constexpr(BT // NUM_STORE_WARPS):
                    full_row = row_base_warp + ri
                    full_abs_row = chunk_start + full_row
                    rAqkOut[0] = csAqk[full_row, col_lo]
                    rAqkOut[1] = csAqk[full_row, col_hi]
                    if full_row < col_lo:
                        rAqkOut[0] = cutlass.BFloat16(0.0)
                    if full_row < col_hi:
                        rAqkOut[1] = cutlass.BFloat16(0.0)
                    if full_abs_row < st_eos:
                        cute.autovec_copy(rAqkOut, mAqk_v2[i_b, full_abs_row, i_h, col_vec_idx, None])
                        if cutlass.const_expr(FUSE_AKK_INV != 0):
                            pass
                        elif cutlass.const_expr(FP32_AKK_WORKSPACE != 0):
                            rAkkFp32Out[0] = cutlass.Float32(csAkk[full_row, col_lo])
                            rAkkFp32Out[1] = cutlass.Float32(csAkk[full_row, col_hi])
                            cute.autovec_copy(rAkkFp32Out, mAkk_v2[i_b, full_abs_row, i_h, col_vec_idx, None])
                        else:
                            rAkkOut[0] = csAkk[full_row, col_lo].to(cutlass.BFloat16)
                            rAkkOut[1] = csAkk[full_row, col_hi].to(cutlass.BFloat16)
                            cute.autovec_copy(rAkkOut, mAkk_v2[i_b, full_abs_row, i_h, col_vec_idx, None])

                if cutlass.const_expr(FUSE_AKK_INV != 0):
                    # Aqk has reached GMEM, so its current-stage storage can
                    # safely become the 4x16x16 FP32 Schur workspace. The
                    # stage remains owned by these four warps until
                    # store_done, hence the next MMA stage cannot race this
                    # alias. 64x72 bf16 = 9216 B; the workspace needs 5120 B.
                    store_internal_barrier()
                    inv_tmp_layout = cute.make_layout(
                        (NUM_STORE_WARPS, BC, BC),
                        stride=(BC * INV_TMP_STRIDE, INV_TMP_STRIDE, 1),
                    )
                    sInvTmp = cute.make_tensor(
                        cute.recast_ptr(csAqk.iterator, dtype=cutlass.Float32),
                        inv_tmp_layout,
                    )
                    store_tid = store_warp * 32 + lane_id

                    # Convert the block-transposed intra accumulator into a
                    # logical unit-lower matrix before the TF32 Schur inverse.
                    for norm_i in cutlass.range_constexpr((BT * BT) // (NUM_STORE_WARPS * 32)):
                        linear = store_tid + norm_i * (NUM_STORE_WARPS * 32)
                        norm_row = linear // BT
                        norm_col = linear % BT
                        if norm_row >= norm_col:
                            norm_val = cutlass.Float32(0.0)
                            if norm_row == norm_col:
                                norm_val = cutlass.Float32(1.0)
                            else:
                                norm_row_blk = norm_row // BC
                                norm_col_blk = norm_col // BC
                                src_row = norm_row
                                src_col = norm_col
                                if norm_row_blk != norm_col_blk:
                                    src_row = norm_col_blk * BC + norm_row % BC
                                    src_col = norm_row_blk * BC + norm_col % BC
                                norm_val = cutlass.Float32(csAkk[src_row, src_col])
                                if IS_VARLEN:
                                    if chunk_start + norm_row >= st_eos:
                                        norm_val = cutlass.Float32(0.0)
                            csAkk[norm_row, norm_col] = norm_val
                    store_internal_barrier()

                    # The off-diagonal source tiles occupy the physical
                    # upper triangle.  They must be cleared after every
                    # logical-lower copy has consumed them; otherwise the
                    # TF32 inverse observes stale intra accumulator values.
                    for clear_i in cutlass.range_constexpr((BT * BT) // (NUM_STORE_WARPS * 32)):
                        linear = store_tid + clear_i * (NUM_STORE_WARPS * 32)
                        clear_row = linear // BT
                        clear_col = linear % BT
                        if clear_row < clear_col:
                            csAkk[clear_row, clear_col] = cutlass.Float32(0.0)
                    store_internal_barrier()

                    if store_tid < 64:
                        _tf32_invert_diag_forward16(csAkk, store_tid // 16, store_tid)
                    store_internal_barrier()

                    # Invert each diagonal 32x32 block.
                    if store_tid < 64:
                        block32 = store_tid // 32
                        row_base = block32 * 32
                        c0, c1, c2, c3, c4, c5, c6, c7 = _tf32_matmul16_smem_smem(
                            csAkk,
                            row_base + 16,
                            row_base + 16,
                            csAkk,
                            row_base + 16,
                            row_base,
                            lane_id,
                        )
                        _tf32_store_C16_tmp(
                            sInvTmp,
                            block32,
                            -c0,
                            -c1,
                            -c2,
                            -c3,
                            -c4,
                            -c5,
                            -c6,
                            -c7,
                            lane_id,
                        )
                    store_internal_barrier()

                    if store_tid < 64:
                        block32 = store_tid // 32
                        row_base = block32 * 32
                        c0, c1, c2, c3, c4, c5, c6, c7 = _tf32_matmul16_tmp_smem(
                            sInvTmp,
                            block32,
                            csAkk,
                            row_base,
                            row_base,
                            lane_id,
                        )
                        _tf32_store_C16_smem(
                            csAkk,
                            row_base + 16,
                            row_base,
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                            lane_id,
                        )
                    store_internal_barrier()

                    # Schur solve for the lower-left 32x32 block. Each warp
                    # computes one 16x16 quadrant.
                    inv_x = store_warp // 2
                    inv_y = store_warp % 2
                    inv_slot = store_warp
                    inv_row_o = 32 + inv_y * 16
                    inv_col_c = inv_x * 16

                    p0, p1, p2, p3, p4, p5, p6, p7 = _tf32_matmul32_smem_smem(
                        csAkk, inv_row_o, 32, csAkk, 32, inv_col_c, lane_id
                    )
                    _tf32_store_C16_tmp(
                        sInvTmp,
                        inv_slot,
                        -p0,
                        -p1,
                        -p2,
                        -p3,
                        -p4,
                        -p5,
                        -p6,
                        -p7,
                        lane_id,
                    )
                    store_internal_barrier()

                    o0, o1, o2, o3, o4, o5, o6, o7 = _tf32_matmul16_tmp_smem(
                        sInvTmp,
                        inv_slot,
                        csAkk,
                        inv_x * 16,
                        0,
                        lane_id,
                    )
                    r0, r1, r2, r3, r4, r5, r6, r7 = _tf32_matmul16_tmp_smem(
                        sInvTmp,
                        inv_slot,
                        csAkk,
                        inv_x * 16,
                        16,
                        lane_id,
                    )
                    if inv_x == 0:
                        _tf32_store_C16_smem(
                            csAkk,
                            inv_row_o,
                            0,
                            o0,
                            o1,
                            o2,
                            o3,
                            o4,
                            o5,
                            o6,
                            o7,
                            lane_id,
                        )
                        _tf32_store_C16_smem(
                            csAkk,
                            inv_row_o,
                            16,
                            r0,
                            r1,
                            r2,
                            r3,
                            r4,
                            r5,
                            r6,
                            r7,
                            lane_id,
                        )
                    store_internal_barrier()
                    if inv_x == 1:
                        _tf32_add_store_C16_smem(
                            csAkk,
                            inv_row_o,
                            0,
                            o0,
                            o1,
                            o2,
                            o3,
                            o4,
                            o5,
                            o6,
                            o7,
                            lane_id,
                        )
                        _tf32_add_store_C16_smem(
                            csAkk,
                            inv_row_o,
                            16,
                            r0,
                            r1,
                            r2,
                            r3,
                            r4,
                            r5,
                            r6,
                            r7,
                            lane_id,
                        )
                    store_internal_barrier()

                    row_base_warp = store_warp * (BT // NUM_STORE_WARPS)
                    for ri in cutlass.range_constexpr(BT // NUM_STORE_WARPS):
                        local_row = row_base_warp + ri
                        abs_row = chunk_start + local_row
                        val0 = cutlass.Float32(csAkk[local_row, col_lo])
                        val1 = cutlass.Float32(csAkk[local_row, col_hi])
                        if local_row < col_lo:
                            val0 = cutlass.Float32(0.0)
                        if local_row < col_hi:
                            val1 = cutlass.Float32(0.0)
                        rAkkOut[0] = val0.to(cutlass.BFloat16)
                        rAkkOut[1] = val1.to(cutlass.BFloat16)
                        if IS_VARLEN:
                            if abs_row < st_eos:
                                cute.autovec_copy(rAkkOut, mAkk_v2[i_b, abs_row, i_h, col_vec_idx, None])
                        else:
                            cute.autovec_copy(rAkkOut, mAkk_v2[i_b, abs_row, i_h, col_vec_idx, None])

                cute.arch.mbarrier_arrive(store_done_mbars + s)

        yield_out()


# =========================================================================
# Host function
# =========================================================================
def make_host_function(
    B,
    NT,
    H_qk,
    H_v=None,
    is_varlen=False,
    T_padded=None,
    fuse_akk_inv=False,
    fp32_akk_workspace=False,
    beta_fp32=False,
):
    """Build the exact FP32-gk intra host used by the public SM100 path."""
    if H_v is None:
        H_v = H_qk
    if H_qk <= 0 or H_v <= 0 or H_v % H_qk != 0:
        raise ValueError(f"H_v ({H_v}) must be a positive multiple of H_qk ({H_qk})")
    _B, _NT, _H_QK, _H_V = B, NT, H_qk, H_v
    _IS_VARLEN = 1 if is_varlen else 0
    _FUSE_AKK_INV = 1 if fuse_akk_inv else 0
    _FP32_AKK_WORKSPACE = 1 if fp32_akk_workspace else 0
    _BETA_FP32 = 1 if beta_fp32 else 0
    if _FUSE_AKK_INV and _FP32_AKK_WORKSPACE:
        raise ValueError("fuse_akk_inv and fp32_akk_workspace are mutually exclusive")
    if is_varlen:
        assert _B == 1, "Varlen requires B=1"
        assert T_padded is not None, "T_padded required for varlen"
        _T = T_padded
    else:
        _T = _NT * BT

    if is_varlen:
        _total_cgs_val = ((_NT + CHUNKS_PER_BLOCK - 1) // CHUNKS_PER_BLOCK) * _H_V
    else:
        _total_cgs_val = (_NT // CHUNKS_PER_BLOCK) * _H_V * _B

    # Q/K live in QK-head space; G and every output live in V-head space.
    _T_total = _B * _T
    s_col = 1
    s_h = K_DIM

    @cute.jit
    def host_fn(
        mQ,
        mK,
        mG,
        mBeta,
        scale,
        mAqk,
        mAkk,
        mCuSeqlens,
        mChunkIndices,
        stream=None,
    ):
        # 3D TMA view: (B*T, K_DIM, H). domain_offset in kernel shifts to
        # arbitrary chunk_start — no BT alignment required for varlen.
        qk_view_layout_3d = cute.make_layout(
            (_T_total, K_DIM, _H_QK),
            stride=(_H_QK * K_DIM, s_col, s_h),
        )
        g_view_layout_3d = cute.make_layout(
            (_T_total, K_DIM, _H_V),
            stride=(_H_V * K_DIM, s_col, s_h),
        )
        mQ_view = cute.make_tensor(mQ.iterator, qk_view_layout_3d)
        mK_view = cute.make_tensor(mK.iterator, qk_view_layout_3d)
        mG_view = cute.make_tensor(mG.iterator, g_view_layout_3d)

        smem_atom_qk = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16)
        qk_smem_2d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM), order=(0, 1))
        qk_smem_3d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM, NUM_STAGES), order=(0, 1, 2))

        smem_atom_gk = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float32)
        gk_smem_2d = cute.tile_to_shape(smem_atom_gk, (BT, K_DIM), order=(0, 1))
        tma_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_op, mQ_view, qk_smem_2d, cute.product_each(qk_smem_2d.shape), num_multicast=1
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_op, mK_view, qk_smem_2d, cute.product_each(qk_smem_2d.shape), num_multicast=1
        )
        tma_atom_G, tma_tensor_G = cpasync.make_tiled_tma_atom(
            tma_op, mG_view, gk_smem_2d, cute.product_each(gk_smem_2d.shape), num_multicast=1
        )
        g_cumsum_layout = cute.tile_to_shape(smem_atom_gk, (BT, K_DIM, NUM_STAGES), order=(0, 1, 2))

        # vec2 views for mAqk / mAkk (BT dimension instead of K_DIM).
        # Shape: (B, T, H, BT/2, 2). Each VEC=2 slot = 2 contiguous bf16 = 1
        # fp32-aligned 4-byte unit. Used by store warp's autovec_copy →
        # STG.E.32 with 32 lanes coalesced to 1 cache line per row.
        BT_VEC = BT // VEC  # 32
        akk_v2_layout = cute.make_layout(
            (_B, _T, _H_V, BT_VEC, VEC),
            stride=(_T * _H_V * BT, _H_V * BT, BT, VEC, 1),
        )
        mAqk_v2 = cute.make_tensor(mAqk.iterator, akk_v2_layout)
        mAkk_v2 = cute.make_tensor(mAkk.iterator, akk_v2_layout)

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        tiled_mma_k2 = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 16))

        tiled_copy_mma_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16), tiled_mma_k2
        )
        tiled_copy_mma_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16), tiled_mma_k2
        )

        copy_atom_Gcum = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64)
        tiled_copy_Gcum_norm = cute.make_tiled_copy_tv(
            copy_atom_Gcum, thr_layout=cute.make_layout((1, 4)), val_layout=cute.make_layout((1, 2))
        )
        tiled_copy_Gcum_gate = cute.make_tiled_copy_C(copy_atom_Gcum, tiled_mma_k2)

        # Q/K bf16: 64 * 128 * 2 * 2 * NUM_STAGES = 65,536 B
        # sAkk fp32: 64 * 68 * 4 * NUM_STAGES = 34,816 B
        # G cumsum: 64 * 136 * 4 * NUM_STAGES = 69,632 B
        # Aqk SMEM: 64 * 72 * 2 * NUM_STAGES = 18,432 B
        # Barriers: 5 * 2 * 8 = 80 B
        smem_size = (
            BT * K_DIM * 2 * 2 * NUM_STAGES  # Q/K bf16: 65,536 B
            + BT * AKK_STRIDE * 4 * NUM_STAGES  # sAkk fp32: 34,816 B
            + BT * K_STRIDE * 4 * NUM_STAGES  # G cumsum fp32: 69,632 B
            + BT * AQK_TILE_STRIDE * 2 * NUM_STAGES  # sAqk bf16: 18,432 B
            + BT * NUM_STAGES * (4 if _BETA_FP32 else 2)
            + 5 * NUM_STAGES * 8  # Barriers: 80 B
        )

        _grid_x = min(NUM_SMS, _total_cgs_val)

        kernel = intra_kernel(
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_G,
            tma_tensor_G,
            mBeta,
            scale,
            mAqk,
            mAkk,
            mAqk_v2,
            mAkk_v2,
            tiled_mma_k2,
            tiled_copy_mma_A,
            tiled_copy_mma_B,
            tiled_copy_Gcum_norm,
            tiled_copy_Gcum_gate,
            qk_smem_3d,
            g_cumsum_layout,
            _NT,
            _H_V,
            _H_QK,
            _B,
            mCuSeqlens,
            mChunkIndices,
            _IS_VARLEN,
            _FUSE_AKK_INV,
            _FP32_AKK_WORKSPACE,
            _BETA_FP32,
        )
        # Standalone callers compile this host with the default stream and
        # therefore pass Python None.  A composed host passes a real DSL
        # stream value.  Omitting the keyword in the former case is required:
        # MLIR launch operands cannot contain None.
        if cutlass.const_expr(stream is None):
            kernel.launch(
                grid=(_grid_x, 1, 1),
                block=(THREADS, 1, 1),
                smem=smem_size,
            )
        else:
            kernel.launch(
                grid=(_grid_x, 1, 1),
                block=(THREADS, 1, 1),
                smem=smem_size,
                stream=stream,
            )

    return host_fn


# ============================================================================
# High-level Python wrappers for training usage
# ============================================================================

from fla.ops.utils import prepare_chunk_indices

_INTRA_CACHE = {}
_VARLEN_PAD_CACHE = {}
_EXACT_PORT_OUTPUT_POOLS = {}
_MAX_EXACT_PORT_POOLED_BYTES = 128 * 1024 * 1024


def _tensor_version(t: torch.Tensor) -> int:
    try:
        return t._version
    except RuntimeError:
        return 0


def _ct_cached(t: torch.Tensor, dtype) -> torch.Tensor:
    """Cache CuTe tensor wrappers to avoid repeated wrapping overhead."""
    version = _tensor_version(t)
    key = ("ct", id(t), dtype)
    entry = _INTRA_CACHE.get(key)
    if entry is None or entry[0]() is not t or entry[1] != version:
        from cutlass.cute.runtime import from_dlpack

        # Use assumed_align and mark_layout_dynamic for optimal performance
        wrapper = from_dlpack(t, assumed_align=16).mark_layout_dynamic()
        wrapper.element_type = dtype
        entry = (weakref.ref(t), version, wrapper)
        _INTRA_CACHE[key] = entry
    return entry[2]


def _ct_uncached(t: torch.Tensor, dtype) -> torch.Tensor:
    """Wrap a call-owned tensor without retaining its allocation globally."""
    from cutlass.cute.runtime import from_dlpack

    wrapper = from_dlpack(t, assumed_align=16).mark_layout_dynamic()
    wrapper.element_type = dtype
    return wrapper


def _pad_varlen_inputs(q, k, g, beta, T_padded):
    """Zero-pad packed varlen inputs to 4*BT boundary."""
    inputs = (q, k, g, beta)
    if all(t.shape[1] == T_padded for t in inputs):
        return inputs

    versions = tuple(_tensor_version(t) for t in inputs)
    key = (*map(id, inputs), T_padded)
    entry = _VARLEN_PAD_CACHE.get(key)
    if entry is None or any(ref() is not tensor for ref, tensor in zip(entry[0], inputs, strict=True)) or entry[1] != versions:

        def _pad(t):
            if t.shape[1] == T_padded:
                return t
            pad = t.new_zeros((t.shape[0], T_padded, *t.shape[2:]))
            pad[:, : t.shape[1]].copy_(t)
            return pad

        entry = (tuple(weakref.ref(t) for t in inputs), versions, tuple(_pad(t) for t in inputs))
        _VARLEN_PAD_CACHE[key] = entry
    return entry[2]


def _get_eqlen_dummies(device):
    """Dummy varlen tensors for equal-length launches."""
    key = ("eqlen_dummies", device.index if device.index is not None else 0)
    entry = _INTRA_CACHE.get(key)
    if entry is None:
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
        chunk_indices = torch.tensor([[0, 0]], dtype=torch.int32, device=device)
        cu_ct = _ct_cached(cu_seqlens, cutlass.Int32)
        ci_ct = _ct_cached(chunk_indices, cutlass.Int32)
        entry = (cu_seqlens, chunk_indices, cu_ct, ci_ct)
        _INTRA_CACHE[key] = entry
    return entry[2], entry[3]


class _ExactPortOutputEntry:
    def __init__(self, specs, device):
        self.specs = specs
        self.buffers = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs)
        self.busy = False

    def is_free(self):
        if self.busy:
            return False
        # The temporary UntypedStorage wrapper contributes one reference and
        # the pooled tensor contributes the other. Any returned view, slice,
        # or detached alias adds another StorageImpl reference.
        return all(torch._C._storage_Use_Count(buffer.untyped_storage()._cdata) == 2 for buffer in self.buffers)

    def acquire(self):
        if self.busy:
            raise RuntimeError("attempted to acquire busy KDA intra outputs")
        self.busy = True
        return self.buffers

    def attach(self):
        self.busy = False

    def release_after_error(self):
        self.busy = False


def _acquire_exact_port_outputs(device, B, T, H):
    shape = (B, T, H, BT)
    specs = ((shape, torch.bfloat16), (shape, torch.bfloat16), (shape, torch.float32))
    nbytes = sum(dtype.itemsize * torch.Size(tensor_shape).numel() for tensor_shape, dtype in specs)
    if nbytes > _MAX_EXACT_PORT_POOLED_BYTES:
        return None, tuple(torch.empty(tensor_shape, dtype=dtype, device=device) for tensor_shape, dtype in specs)

    stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
    pool = _EXACT_PORT_OUTPUT_POOLS.setdefault((device.index, stream_ptr), [])
    for entry in pool:
        if entry.is_free() and entry.specs == specs:
            return entry, entry.acquire()
    pool[:] = [entry for entry in pool if not entry.is_free()]
    entry = _ExactPortOutputEntry(specs, device)
    pool.append(entry)
    return entry, entry.acquire()


def _export_exact_port_outputs(entry, A_qk, A_kk):
    if entry is None:
        return A_qk, A_kk
    outputs = (A_qk.view_as(A_qk), A_kk.view_as(A_kk))
    entry.attach()
    return outputs


def _run_akk_inv_fp32_physical(
    A_phys: torch.Tensor,
    A_out: torch.Tensor,
    B: int,
    NT: int,
    H: int,
    cu_ct,
    ci_ct,
    *,
    is_varlen: bool,
    T_val: int,
    cache_wrappers: bool = True,
):
    dev = A_phys.device.index if A_phys.device.index is not None else 0
    wrap = _ct_cached if cache_wrappers else _ct_uncached
    phys_ct = wrap(A_phys, cutlass.Float32)
    out_ct = wrap(A_out, cutlass.BFloat16)
    cache_key = ("akk_inv_fp32_physical", dev, B, NT, H, bool(is_varlen), T_val)
    kernel = _INTRA_CACHE.get(cache_key)
    if kernel is None:
        kernel = cute.compile(
            _akk_inv_fp32_physical_host,
            phys_ct,
            out_ct,
            B,
            NT,
            H,
            cu_ct,
            ci_ct,
            1 if is_varlen else 0,
            T_val,
        )
        _INTRA_CACHE[cache_key] = kernel
    kernel(phys_ct, out_ct, cu_ct, ci_ct)


def chunk_kda_fwd_intra_sm100_from_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = BT,
    use_tf32_inverse: bool = True,
    fp32_akk_inv: bool = True,
):
    """CuTeDSL implementation of the exact SM100 intra contract.

    This entry point consumes the already activated and chunk-cumsummed FP32
    ``gk`` tensor, exactly like the former SM100 native API. Gate fusion is not
    part of this precision contract.

    ``fp32_akk_inv=True`` keeps the pre-inverse Akk workspace in FP32 and runs
    the standalone TF32 inverse. ``fp32_akk_inv=False`` enables the matching
    in-CTA TF32 inverse for aligned equal-length inputs.
    """
    if chunk_size != BT:
        raise NotImplementedError(f"chunk_size must be {BT}, got {chunk_size}.")
    if q.dim() != 4 or k.dim() != 4 or gk.dim() != 4 or beta.dim() != 3:
        raise ValueError("Expected q/k/gk [B,T,H,K] and beta [B,T,H].")
    if q.shape != k.shape:
        raise ValueError(f"q/k shapes must match, got q={q.shape}, k={k.shape}.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("q and k must be bfloat16, matching the csrc specialization.")
    if gk.dtype != torch.float32:
        raise TypeError(f"gk must be float32, got {gk.dtype}.")
    if beta.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"beta must be bfloat16 or float32, got {beta.dtype}.")
    B, T_total, H_qk, K = q.shape
    if gk.shape[:2] != (B, T_total) or gk.shape[3] != K:
        raise ValueError(f"gk must have shape [B, T, H_v, K], got {tuple(gk.shape)}.")
    H_v = gk.shape[2]
    if H_qk <= 0 or H_v <= 0 or H_v % H_qk != 0:
        raise ValueError(f"H_v ({H_v}) must be a positive multiple of H_qk ({H_qk}).")
    if K != K_DIM:
        raise NotImplementedError(f"K must be {K_DIM}, got {K}.")
    if beta.shape != (B, T_total, H_v):
        raise ValueError(f"beta shape must be {(B, T_total, H_v)}, got {tuple(beta.shape)}.")
    if not use_tf32_inverse:
        raise NotImplementedError("The one-to-one baseline currently ports the default csrc TF32 inverse only.")
    if not fp32_akk_inv and (cu_seqlens is not None or T_total % (CHUNKS_PER_BLOCK * BT) != 0):
        raise NotImplementedError("The in-CTA inverse currently requires the aligned equal-length scheduler.")
    # The compiled kernel constructs fixed compact views from each base
    # pointer, so normalize strided slices before either scheduler path.
    q = q.contiguous()
    k = k.contiguous()
    gk = gk.contiguous()
    beta = beta.contiguous()
    if scale is None:
        scale = K_DIM**-0.5

    # The equal-length specialization groups four chunks per CTA and avoids
    # scheduler decoding. Any explicit packed layout, or an equal-length tail
    # smaller than four chunks, uses the same arithmetic kernel through its
    # varlen scheduler. This is also how the csrc API represents tails.
    if cu_seqlens is not None or T_total % (CHUNKS_PER_BLOCK * BT) != 0:
        if cu_seqlens is None:
            cu_seqlens = torch.arange(B + 1, dtype=torch.int32, device=q.device) * T_total
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}.")
        cu_seqlens = cu_seqlens.contiguous()
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        if chunk_indices.dtype != torch.int32:
            raise TypeError(f"chunk_indices must be int32, got {chunk_indices.dtype}.")
        chunk_indices = chunk_indices.contiguous()

        T_flat = B * T_total
        T_padded = ((T_flat + CHUNKS_PER_BLOCK * BT - 1) // (CHUNKS_PER_BLOCK * BT)) * (CHUNKS_PER_BLOCK * BT)
        q_pack = q.reshape(1, T_flat, H_qk, K)
        k_pack = k.reshape(1, T_flat, H_qk, K)
        gk_pack = gk.reshape(1, T_flat, H_v, K)
        beta_pack = beta.reshape(1, T_flat, H_v)
        q_pad, k_pad, gk_pad, beta_pad = _pad_varlen_inputs(q_pack, k_pack, gk_pack, beta_pack, T_padded)

        NT = chunk_indices.shape[0]
        dev = q.device.index if q.device.index is not None else 0
        cache_key = ("exact_port_varlen", dev, T_padded, NT, H_qk, H_v, beta.dtype, fp32_akk_inv)
        output_entry, (A_qk, A_kk, A_kk_workspace) = _acquire_exact_port_outputs(q.device, 1, T_padded, H_v)
        try:
            q_ct = _ct_uncached(q_pad, cutlass.BFloat16)
            k_ct = _ct_uncached(k_pad, cutlass.BFloat16)
            gk_in_ct = _ct_uncached(gk_pad, cutlass.Float32)
            beta_type = cutlass.Float32 if beta.dtype == torch.float32 else cutlass.BFloat16
            beta_ct = _ct_uncached(beta_pad, beta_type)
            aqk_ct = _ct_uncached(A_qk, cutlass.BFloat16)
            akk_ct = _ct_uncached(A_kk_workspace, cutlass.Float32)
            cu_ct = _ct_uncached(cu_seqlens, cutlass.Int32)
            ci_ct = _ct_uncached(chunk_indices, cutlass.Int32)
            ct_args = (
                q_ct,
                k_ct,
                gk_in_ct,
                beta_ct,
                float(scale),
                aqk_ct,
                akk_ct,
                cu_ct,
                ci_ct,
            )
            kernel = _INTRA_CACHE.get(("kernel", cache_key))
            if kernel is None:
                host_fn = make_host_function(
                    1,
                    NT,
                    H_qk,
                    H_v,
                    is_varlen=True,
                    T_padded=T_padded,
                    fp32_akk_workspace=True,
                    beta_fp32=beta.dtype == torch.float32,
                )
                kernel = cute.compile(host_fn, *ct_args)
                _INTRA_CACHE[("kernel", cache_key)] = kernel
            kernel(*ct_args)
            _run_akk_inv_fp32_physical(
                A_kk_workspace,
                A_kk,
                1,
                NT,
                H_v,
                cu_ct,
                ci_ct,
                is_varlen=True,
                T_val=T_padded,
                cache_wrappers=False,
            )
        except Exception:
            if output_entry is not None:
                output_entry.release_after_error()
            raise
        return _export_exact_port_outputs(
            output_entry,
            A_qk[:, :T_flat].view(B, T_total, H_v, BT),
            A_kk[:, :T_flat].view(B, T_total, H_v, BT),
        )

    NT = T_total // BT
    dev = q.device.index if q.device.index is not None else 0
    cache_key = ("exact_port", dev, B, NT, H_qk, H_v, beta.dtype, fp32_akk_inv)
    output_entry, (A_qk, A_kk, A_kk_workspace) = _acquire_exact_port_outputs(q.device, B, T_total, H_v)
    try:
        q_ct = _ct_uncached(q, cutlass.BFloat16)
        k_ct = _ct_uncached(k, cutlass.BFloat16)
        gk_in_ct = _ct_uncached(gk, cutlass.Float32)
        beta_type = cutlass.Float32 if beta.dtype == torch.float32 else cutlass.BFloat16
        beta_ct = _ct_uncached(beta, beta_type)
        aqk_ct = _ct_uncached(A_qk, cutlass.BFloat16)
        akk_storage = A_kk_workspace if fp32_akk_inv else A_kk
        akk_ct = _ct_uncached(akk_storage, cutlass.Float32 if fp32_akk_inv else cutlass.BFloat16)
        cu_ct, ci_ct = _get_eqlen_dummies(q.device)
        ct_args = (
            q_ct,
            k_ct,
            gk_in_ct,
            beta_ct,
            float(scale),
            aqk_ct,
            akk_ct,
            cu_ct,
            ci_ct,
        )
        kernel = _INTRA_CACHE.get(("kernel", cache_key))
        if kernel is None:
            host_fn = make_host_function(
                B,
                NT,
                H_qk,
                H_v,
                is_varlen=False,
                fuse_akk_inv=not fp32_akk_inv,
                fp32_akk_workspace=fp32_akk_inv,
                beta_fp32=beta.dtype == torch.float32,
            )
            kernel = cute.compile(host_fn, *ct_args)
            _INTRA_CACHE[("kernel", cache_key)] = kernel
        kernel(*ct_args)
        if fp32_akk_inv:
            _run_akk_inv_fp32_physical(
                A_kk_workspace,
                A_kk,
                B,
                NT,
                H_v,
                cu_ct,
                ci_ct,
                is_varlen=False,
                T_val=T_total,
                cache_wrappers=False,
            )
    except Exception:
        if output_entry is not None:
            output_entry.release_after_error()
        raise
    return _export_exact_port_outputs(output_entry, A_qk, A_kk)
