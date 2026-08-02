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

"""SM100 (Blackwell) Tensor Memory intrinsics and UMMA extensions for CuTeDSL."""

__all__ = [
    # TMEM load/store/copy
    "tcgen05_ld_32x32b",
    "tcgen05_st_32x32b",
    "tcgen05_cp_128x256b",
    "tcgen05_cp_128x128b",
    "tcgen05_fence_before",
    "tcgen05_fence_after",
    "umma_arrive",
    "umma_arrive_noelect",
    # descriptor helpers
    "Tcgen05SmemDescriptor",
    "initialize_tcgen05_descriptor",
    # low-level MMA primitives
    "tcgen05mma_ss",
    "tcgen05mma_ts",
    "tcgen05mma_ws_ss_tf32",
    "tcgen05mma_ws_ts_tf32",
    "tcgen05mma_ws_ss_f16",
    "tcgen05mma_ws_ts_f16",
    # SS named wrappers
    "tcgen05mma_ss_no_mask",
    "tcgen05mma_ss_mask0",
    "tcgen05mma_ss_mask1",
    "tcgen05mma_ss_mask2",
    "tcgen05mma_ss_mask3",
    # TS named wrappers
    "tcgen05mma_ts_no_mask",
    "tcgen05mma_ts_mask0",
    "tcgen05mma_ts_mask1",
    "tcgen05mma_ts_mask2",
    "tcgen05mma_ts_mask3",
    "tcgen05mma_ts_mask02",
    "tcgen05mma_ts_mask13",
    # collector enums (re-exported for convenience)
    "CollectorBBuffer",
    "CollectorOp",
]

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.arch import elect_one
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Int32
from cutlass.cutlass_dsl import dsl_user_op

CollectorBBuffer = _nvvm.Tcgen05MMACollectorBBuffer
CollectorOp = _nvvm.Tcgen05MMACollectorOp


def _to_ir(val, loc=None, ip=None):
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


# ===========================================================================
# Tcgen05SmemDescriptor — 64-bit SMEM descriptor stored as 2×Int32
# ===========================================================================


class Tcgen05SmemDescriptor:
    """64-bit shared-memory descriptor for tcgen05 MMA (Blackwell / SM100).

    The descriptor encodes SMEM base address, leading/stride byte offsets,
    swizzle mode, and other fields required by the ``tcgen05.mma`` PTX
    instruction to locate a matrix tile in shared memory.

    64-bit layout (PTX ISA Table 40)::

      Bit 63                                                      Bit 0
      ┌──────────┬────────┬─────┬──────────┬────┬──────────┬──────┬──────────────┐
      │ 63    61 │ 60  53 │  52 │ 51    49 │ 48 │ 45    32 │31 30 │ 29   16│15 14│ 13     0│
      │layout_typ│ reservd│l_abs│base_offst│ 46 │   SBO    │ rsvd │  LBO   │rsvd │start_adr│
      │  (3 bit) │ (8 bit)│(1b) │  (3 bit) │=0b001│(14 bit)│(2 b) │(14 bit)│(2b) │(14 bit) │
      └──────────┴────────┴─────┴──────────┴────┴──────────┴──────┴────────┴─────┴─────────┘

    Storage: two Int32 registers (desc[0] = low 32 bits, desc[1] = high 32
    bits), recast to a single Int64 for the PTX ``l``-constraint operand.

    Usage inside a @cute.jit kernel::

        desc = Tcgen05SmemDescriptor()
        initialize_tcgen05_descriptor(desc, smem_ptr, lbo, sbo, 0, True, swizzle)
    """

    def __init__(self, desc_64: cute.Int64 = None):
        self.desc = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        self.desc_i64 = cute.make_tensor(cute.recast_ptr(self.desc.iterator, dtype=cute.Int64), (1,))
        if desc_64 is not None:
            self.desc_i64[0] = desc_64

    def __add__(self, byte_offset):
        """Return a new descriptor offset by ``byte_offset`` bytes."""
        res = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        res_i64 = cute.make_tensor(cute.recast_ptr(res.iterator, dtype=cute.Int64), (1,))
        res[0] = self.desc[0] + (byte_offset >> 4)
        res[1] = self.desc[1]
        return Tcgen05SmemDescriptor(res_i64[0])


def initialize_tcgen05_descriptor(
    desc,
    start_address,
    leading_byte_offset,
    stride_byte_offset,
    base_offset,
    leading_abs,
    swizzle_mode,
):
    """Pack SMEM descriptor bitfields into *desc* (a Tcgen05SmemDescriptor).

    All address/offset fields must be pre-divided by 16 (``>> 4``) before
    passing, because the hardware stores them in 16-byte granularity.

    Args:
        desc:                 Tcgen05SmemDescriptor to fill.
        start_address:        CuTeDSL Pointer to the SMEM tile start.
        leading_byte_offset:  Leading-dimension byte offset, already >> 4.
        stride_byte_offset:   Stride  byte offset, already >> 4.
        base_offset:          Swizzle alignment correction (raw int, bits 17-19).
        leading_abs:          Bool — True → LBO is absolute address.
        swizzle_mode:         Swizzle layout_type integer (bits 29-31).
    """
    ptr_val = start_address.toint() >> 4

    desc.desc[0] = cutlass.Int32(ptr_val) | cutlass.Int32(cutlass.Int32(leading_byte_offset) << 16)

    desc.desc[1] = (
        cutlass.Int32(stride_byte_offset)
        | cutlass.Int32(1 << 14)
        | cutlass.Int32(cutlass.Int32(base_offset & 0x7) << 17)
        | cutlass.Int32(cutlass.Int32(int(leading_abs)) << 20)
        | cutlass.Int32(cutlass.Int32(swizzle_mode & 0x7) << 29)
    )


# ===========================================================================
# TMEM load / store / copy  (tcgen05.ld / tcgen05.st / tcgen05.cp)
# ===========================================================================


@cute.jit
def tcgen05_ld_32x32b(num: int, taddr: int):
    """Load *num* × 32-bit values from TMEM → an opaque ``vector<N x i32>``."""

    @dsl_user_op
    def _do(addr_val, *, loc=None, ip=None):
        i32_ty = ir.IntegerType.get_signless(32)
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        vec_i32_ty = ir.VectorType.get([num], i32_ty)
        return _nvvm.tcgen05_ld(
            res=vec_i32_ty,
            shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
            num=num,
            tmem_addr=tmem_ptr,
            loc=loc,
            ip=ip,
        )

    return _do(Int32(taddr))


@cute.jit
def tcgen05_st_32x32b(num: int, taddr: int, vec):
    """Store *num* × 32-bit values from an opaque vector → TMEM."""

    @dsl_user_op
    def _do(addr_val, vec_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        _nvvm.tcgen05_st(
            shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
            num=num,
            tmem_addr=tmem_ptr,
            r=_to_ir(vec_val, loc, ip),
            loc=loc,
            ip=ip,
        )

    _do(Int32(taddr), vec)


@cute.jit
def tcgen05_cp_128x256b(taddr: int, smem_desc: Tcgen05SmemDescriptor):
    """Async copy SMEM → TMEM with shape ``128x256b`` (``cta_group::1``)."""

    @dsl_user_op
    def _do(addr_val, desc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        _nvvm.tcgen05_cp(
            shape=_nvvm.Tcgen05CpShape.SHAPE_128x256b,
            taddr=tmem_ptr,
            smem_desc=_to_ir(desc_val, loc, ip),
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            loc=loc,
            ip=ip,
        )

    _do(Int32(taddr), smem_desc.desc_i64[0])


@cute.jit
def tcgen05_cp_128x128b(taddr: int, smem_desc: Tcgen05SmemDescriptor):
    """Async copy SMEM → TMEM with shape ``128x128b`` (``cta_group::1``)."""

    @dsl_user_op
    def _do(addr_val, desc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        _nvvm.tcgen05_cp(
            shape=_nvvm.Tcgen05CpShape.SHAPE_128x128b,
            taddr=tmem_ptr,
            smem_desc=_to_ir(desc_val, loc, ip),
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            loc=loc,
            ip=ip,
        )

    _do(Int32(taddr), smem_desc.desc_i64[0])


@cute.jit
def tcgen05_fence_before():
    """tcgen05.fence::before_thread_sync — non-blocking ordering fence."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_fence_after():
    """tcgen05.fence::after_thread_sync — non-blocking ordering fence."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


@cute.jit
def umma_arrive(mbar_ptr: cute.Pointer):
    """tcgen05.commit.cta_group::1.mbarrier::arrive::one — signal MMA done."""
    with elect_one():
        tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)


@cute.jit
def umma_arrive_noelect(mbar_ptr: cute.Pointer):
    """tcgen05.commit.cta_group::1.mbarrier::arrive::one — signal MMA done."""
    tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)


# ===========================================================================
# Disable-output-lane mask constants (4 × uint32)
# ===========================================================================

_ALL_ACTIVE = 0x00000000
_ALL_OFF = 0xFFFFFFFF

# SS masks (SMEM A, SMEM B)
SS_NO_MASK = (_ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE)
SS_MASK0 = (_ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)
SS_MASK1 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE)
SS_MASK2 = (_ALL_OFF, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)
SS_MASK3 = (_ALL_OFF, _ALL_OFF, _ALL_OFF, _ALL_ACTIVE)

# TS masks (TMEM A, SMEM B)
TS_NO_MASK = (_ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE)
TS_MASK0 = (_ALL_ACTIVE, _ALL_OFF, _ALL_OFF, _ALL_OFF)
TS_MASK1 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_OFF)
TS_MASK2 = (_ALL_OFF, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)
TS_MASK3 = (_ALL_OFF, _ALL_OFF, _ALL_OFF, _ALL_ACTIVE)
TS_MASK02 = (_ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)
TS_MASK13 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE)


# ===========================================================================
# Low-level MMA primitives
# ===========================================================================


@cute.jit
def tcgen05mma_ss(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
):
    """Issue ``tcgen05.mma.cta_group::1.kind::tf32`` with SMEM operands."""

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, m0_val, m1_val, m2_val, m3_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i32_ty = ir.IntegerType.get_signless(32)
        i1_ty = ir.IntegerType.get_signless(1)
        vec4i32_ty = ir.VectorType.get([4], i32_ty)

        c_ir = _to_ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        da_ir = _to_ir(da_val, loc, ip)
        db_ir = _to_ir(db_val, loc, ip)
        dv_ir = _to_ir(dv_val, loc, ip)
        sc_ir = _to_ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        m0_ir = _to_ir(m0_val, loc, ip)
        m1_ir = _to_ir(m1_val, loc, ip)
        m2_ir = _to_ir(m2_val, loc, ip)
        m3_ir = _to_ir(m3_val, loc, ip)

        undef = llvm.mlir_undef(vec4i32_ty, loc=loc, ip=ip)
        idx0 = _arith.constant(i32_ty, 0, loc=loc, ip=ip)
        idx1 = _arith.constant(i32_ty, 1, loc=loc, ip=ip)
        idx2 = _arith.constant(i32_ty, 2, loc=loc, ip=ip)
        idx3 = _arith.constant(i32_ty, 3, loc=loc, ip=ip)
        v = llvm.InsertElementOp(undef, m0_ir, idx0, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m1_ir, idx1, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m2_ir, idx2, loc=loc, ip=ip)
        mask = llvm.InsertElementOp(v, m3_ir, idx3, loc=loc, ip=ip)

        _nvvm.tcgen05_mma(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            d=d_ptr,
            a=da_ir,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            write_disable_mask=mask,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


@cute.jit
def tcgen05mma_ts(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
):
    """Issue ``tcgen05.mma.cta_group::1.kind::tf32`` with TMEM A operand."""

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, m0_val, m1_val, m2_val, m3_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i32_ty = ir.IntegerType.get_signless(32)
        i1_ty = ir.IntegerType.get_signless(1)
        vec4i32_ty = ir.VectorType.get([4], i32_ty)

        c_ir = _to_ir(c_val, loc, ip)
        a_ir = _to_ir(a_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        a_ptr = llvm.inttoptr(ptr6_ty, a_ir, loc=loc, ip=ip)
        b_ir = _to_ir(db_val, loc, ip)
        dv_ir = _to_ir(dv_val, loc, ip)
        sc_ir = _to_ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        m0_ir = _to_ir(m0_val, loc, ip)
        m1_ir = _to_ir(m1_val, loc, ip)
        m2_ir = _to_ir(m2_val, loc, ip)
        m3_ir = _to_ir(m3_val, loc, ip)

        undef = llvm.mlir_undef(vec4i32_ty, loc=loc, ip=ip)
        idx0 = _arith.constant(i32_ty, 0, loc=loc, ip=ip)
        idx1 = _arith.constant(i32_ty, 1, loc=loc, ip=ip)
        idx2 = _arith.constant(i32_ty, 2, loc=loc, ip=ip)
        idx3 = _arith.constant(i32_ty, 3, loc=loc, ip=ip)
        v = llvm.InsertElementOp(undef, m0_ir, idx0, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m1_ir, idx1, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m2_ir, idx2, loc=loc, ip=ip)
        mask = llvm.InsertElementOp(v, m3_ir, idx3, loc=loc, ip=ip)

        _nvvm.tcgen05_mma(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            d=d_ptr,
            a=a_ptr,
            b=b_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            write_disable_mask=mask,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


# ---------------------------------------------------------------------------
# Weight-stationary variants
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ss_tf32(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::tf32`` (weight-stationary, SS)."""

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _to_ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        da_ir = _to_ir(da_val, loc, ip)
        db_ir = _to_ir(db_val, loc, ip)
        dv_ir = _to_ir(dv_val, loc, ip)
        sc_ir = _to_ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            d=d_ptr,
            a=da_ir,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


@cute.jit
def tcgen05mma_ws_ss_f16(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::f16`` (weight-stationary, SS)."""

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _to_ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        da_ir = _to_ir(da_val, loc, ip)
        db_ir = _to_ir(db_val, loc, ip)
        dv_ir = _to_ir(dv_val, loc, ip)
        sc_ir = _to_ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.F16,
            d=d_ptr,
            a=da_ir,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


@cute.jit
def tcgen05mma_ws_ts_tf32(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::tf32`` with TMEM A (weight-stationary)."""

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _to_ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        a_ir = _to_ir(a_val, loc, ip)
        a_ptr = llvm.inttoptr(ptr6_ty, a_ir, loc=loc, ip=ip)
        db_ir = _to_ir(db_val, loc, ip)
        dv_ir = _to_ir(dv_val, loc, ip)
        sc_ir = _to_ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            d=d_ptr,
            a=a_ptr,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


@cute.jit
def tcgen05mma_ws_ts_f16(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::f16`` with TMEM A (weight-stationary)."""

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _to_ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        a_ir = _to_ir(a_val, loc, ip)
        a_ptr = llvm.inttoptr(ptr6_ty, a_ir, loc=loc, ip=ip)
        db_ir = _to_ir(db_val, loc, ip)
        dv_ir = _to_ir(dv_val, loc, ip)
        sc_ir = _to_ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.F16,
            d=d_ptr,
            a=a_ptr,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ===========================================================================
# Named convenience wrappers (pre-set mask constants)
# ===========================================================================

# ---------------------------------------------------------------------------
# SS named wrappers  (SMEM A)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ss_no_mask(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA with no output-lane disable (all rows active)."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_NO_MASK[0], SS_NO_MASK[1], SS_NO_MASK[2], SS_NO_MASK[3])


@cute.jit
def tcgen05mma_ss_mask0(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0, 0xF…, 0, 0xF…} — groups 0,2 active (1,3 disabled)."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK0[0], SS_MASK0[1], SS_MASK0[2], SS_MASK0[3])


@cute.jit
def tcgen05mma_ss_mask1(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0, 0xF…, 0} — groups 1,3 active (0,2 disabled)."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK1[0], SS_MASK1[1], SS_MASK1[2], SS_MASK1[3])


@cute.jit
def tcgen05mma_ss_mask2(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0xF…, 0, 0xF…} — group 2 only active."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK2[0], SS_MASK2[1], SS_MASK2[2], SS_MASK2[3])


@cute.jit
def tcgen05mma_ss_mask3(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0xF…, 0xF…, 0} — group 3 only active."""
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out, SS_MASK3[0], SS_MASK3[1], SS_MASK3[2], SS_MASK3[3])


# ---------------------------------------------------------------------------
# TS named wrappers  (TMEM A)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ts_no_mask(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA with no output-lane disable (all rows active)."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_NO_MASK[0], TS_NO_MASK[1], TS_NO_MASK[2], TS_NO_MASK[3])


@cute.jit
def tcgen05mma_ts_mask0(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0, 0xF…, 0xF…, 0xF…} — group 0 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK0[0], TS_MASK0[1], TS_MASK0[2], TS_MASK0[3])


@cute.jit
def tcgen05mma_ts_mask1(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0, 0xF…, 0xF…} — group 1 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK1[0], TS_MASK1[1], TS_MASK1[2], TS_MASK1[3])


@cute.jit
def tcgen05mma_ts_mask2(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0xF…, 0, 0xF…} — group 2 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK2[0], TS_MASK2[1], TS_MASK2[2], TS_MASK2[3])


@cute.jit
def tcgen05mma_ts_mask3(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0xF…, 0xF…, 0} — group 3 only active."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK3[0], TS_MASK3[1], TS_MASK3[2], TS_MASK3[3])


@cute.jit
def tcgen05mma_ts_mask02(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0, 0xF…, 0, 0xF…} — groups 0,2 active (1,3 disabled)."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK02[0], TS_MASK02[1], TS_MASK02[2], TS_MASK02[3])


@cute.jit
def tcgen05mma_ts_mask13(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0, 0xF…, 0} — groups 1,3 active (0,2 disabled)."""
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out, TS_MASK13[0], TS_MASK13[1], TS_MASK13[2], TS_MASK13[3])
