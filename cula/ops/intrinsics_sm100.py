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

"""NVVM wrappers for SM100 (Blackwell) Tensor Memory intrinsics.

Provides low-level, CuteDSL-compatible helpers that move data between
Tensor Memory (TMEM) and registers / shared memory via the native
``nvvm.tcgen05.*`` MLIR ops.

**T2R / R2T** – ``tcgen05.ld`` / ``tcgen05.st`` with ``.32x32b`` shape.
**S2T**       – ``tcgen05.cp`` with ``.128x256b`` shape (SMEM → TMEM)
PTX reference
-------------
    tcgen05.ld.sync.aligned.32x32b.xN.b32  {r0, ..., rN-1}, [taddr];
    tcgen05.st.sync.aligned.32x32b.xN.b32  [taddr], {r0, ..., rN-1};

where ``N ∈ {2, 4, 8, 16, 32, 64, 128}`` and each ``r`` is a 32-bit
register.  ``taddr`` encodes both the TMEM column index (bits [15:0])
and the lane index (bits [31:16]).

See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld

Usage inside a ``@cute.kernel`` or ``@cute.jit`` function::

    from cula.ops.intrinsics_sm100 import (
        tcgen05_ld_32x32b, tcgen05_st_32x32b,
        reinterpret_cast, subvec, store_256b,
    )
    from cutlass.cute.typing import Float32, Int32

    # Load 32 × 32-bit values from TMEM → opaque vector<32 x i32>
    vec_i32 = tcgen05_ld_32x32b(32, taddr)

    # Zero-cost reinterpret as f32 (single vector.bitcast, no instructions)
    vec_f32 = reinterpret_cast(vec_i32, Int32, 32, Float32)

    # Store to global via store_256b (4 × 256-bit stores)
    # store_256b takes vector<8 x i32>, so reinterpret back and slice
    vec_i32_back = reinterpret_cast(vec_f32, Float32, 32, Int32)
    for chunk in range(4):  # 32 / 8 = 4 chunks
        store_256b(gmem_addr + chunk * 32, subvec(vec_i32_back, chunk * 8, 8))

    # Store back to TMEM
    tcgen05_st_32x32b(32, taddr, vec_i32_back)
"""

__all__ = [
    "tcgen05_ld_32x32b",
    "tcgen05_st_32x32b",
    "tcgen05_cp_128x256b",
    "reinterpret_cast",
    "subvec",
    "store_256b",
]

import cutlass.cute as cute
from cutlass._mlir import ir as _ir_mod
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass._mlir.dialects import vector as _vector
from cutlass.cute.typing import Int32
from cutlass.cutlass_dsl import dsl_user_op


def _to_ir(val, loc=None, ip=None):
    """Extract raw MLIR IR value from a CuteDSL wrapper."""
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


# ---------------------------------------------------------------------------
# tcgen05.ld.sync.aligned.32x32b.xN.b32  (via nvvm.tcgen05.ld)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05_ld_32x32b(num: int, taddr: int):
    """Load *num* × 32-bit values from TMEM → an opaque ``vector<N x i32>``.

    ``num`` must be a **compile-time constant** in {2, 4, 8, 16, 32, 64, 128}.
    Returns a single opaque MLIR vector value (``vector<num x i32>``).

    Use :func:`reinterpret_cast` to reinterpret the element type (zero-cost),
    and :func:`subvec` to slice a contiguous sub-vector.

    Parameters
    ----------
    num : int
        Number of 32-bit registers to load.  Must be a compile-time constant.
    taddr : int
        TMEM address (bits [31:16] = lane, bits [15:0] = column).
    """

    @dsl_user_op
    def _do(addr_val, *, loc=None, ip=None):
        i32_ty = _ir_mod.IntegerType.get_signless(32)
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        vec_i32_ty = _ir_mod.VectorType.get([num], i32_ty)
        return _nvvm.tcgen05_ld(
            res=vec_i32_ty,
            shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
            num=num,
            tmem_addr=tmem_ptr,
            loc=loc,
            ip=ip,
        )

    return _do(Int32(taddr))


# ---------------------------------------------------------------------------
# tcgen05.st.sync.aligned.32x32b.xN.b32  (via nvvm.tcgen05.st)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05_st_32x32b(num: int, taddr: int, vec):
    """Store *num* × 32-bit values from an opaque vector → TMEM.

    ``num`` must be a **compile-time constant** in {2, 4, 8, 16, 32, 64, 128}.

    Parameters
    ----------
    num : int
        Number of 32-bit registers to store.  Must be a compile-time constant.
    taddr : int
        TMEM address (bits [31:16] = lane, bits [15:0] = column).
    vec : opaque vector
        An opaque ``vector<num x i32>`` value (from :func:`tcgen05_ld_32x32b`
        or :func:`reinterpret_cast`).
    """

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


# ---------------------------------------------------------------------------
# reinterpret_cast  (zero-cost vector.bitcast)
# ---------------------------------------------------------------------------


@cute.jit
def reinterpret_cast(vec, src_type, src_num, tgt_type):
    """Zero-cost reinterpret of a vector's element type (single ``vector.bitcast``).

    Analogous to C++ ``reinterpret_cast``: no instructions emitted, just
    re-labels the bits.  The total bit-width is preserved:
    ``src_num * src_type.width == tgt_num * tgt_type.width``.

    Parameters
    ----------
    vec : opaque vector
        Source vector (e.g. ``vector<N x i32>`` from :func:`tcgen05_ld_32x32b`).
    src_type : CuTeDSL type
        Element type of *vec* (e.g. ``Int32``).
    src_num : int
        Number of elements in *vec* (compile-time constant).
    tgt_type : CuTeDSL type
        Desired element type (e.g. ``Float32``, ``BFloat16``, ``Float16``).

    Returns
    -------
    opaque vector
        ``vector<M x tgt_type>`` where ``M = src_num * src_type.width // tgt_type.width``.

    Examples
    --------
    ::

        vec_i32  = tcgen05_ld_32x32b(8, taddr)                     # vector<8 x i32>
        vec_f32  = reinterpret_cast(vec_i32, Int32, 8, Float32)    # vector<8 x f32>
        vec_bf16 = reinterpret_cast(vec_i32, Int32, 8, BFloat16)   # vector<16 x bf16>
        vec_back = reinterpret_cast(vec_bf16, BFloat16, 16, Int32) # vector<8 x i32>
    """
    tgt_num = src_num * src_type.width // tgt_type.width

    @dsl_user_op
    def _do(v, *, loc=None, ip=None):
        tgt_vec_ty = _ir_mod.VectorType.get([tgt_num], tgt_type.mlir_type)
        return _vector.bitcast(tgt_vec_ty, _to_ir(v, loc, ip), loc=loc, ip=ip)

    return _do(vec)


# ---------------------------------------------------------------------------
# subvec  (extract a contiguous sub-vector)
# ---------------------------------------------------------------------------


@cute.jit
def subvec(vec, offset, size):
    """Extract a contiguous sub-vector (``vector.extract_strided_slice``).

    Parameters
    ----------
    vec : opaque vector
        Source vector.
    offset : int
        Starting element index (compile-time constant).
    size : int
        Number of elements to extract (compile-time constant).

    Returns
    -------
    opaque vector
        ``vector<size x elem_type>``.
    """

    @dsl_user_op
    def _do(v, *, loc=None, ip=None):
        ir_v = _to_ir(v, loc, ip)
        elem_ty = _ir_mod.VectorType(ir_v.type).element_type
        res_ty = _ir_mod.VectorType.get([size], elem_ty)
        return _vector.extract_strided_slice(
            res_ty,
            ir_v,
            offsets=[offset],
            sizes=[size],
            strides=[1],
            loc=loc,
            ip=ip,
        )

    return _do(vec)


# ---------------------------------------------------------------------------
# st.global.L1::no_allocate.v8.f32  (256-bit direct R2G store)
# ---------------------------------------------------------------------------

_STORE_256B_ASM = "st.global.L1::no_allocate.v8.f32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
_STORE_256B_CONSTRAINTS = "l,r,r,r,r,r,r,r,r"


@cute.jit
def store_256b(gmem_ptr, vec):
    """Store 256 bits (8 × 32-bit) to global memory, bypassing L1 allocation.

    Issues ``st.global.L1::no_allocate.v8.f32`` with ``"r"`` (integer register)
    constraints — type-agnostic, just like C++ ``reinterpret_cast<uint32_t*>``.

    Parameters
    ----------
    gmem_ptr : pointer
        Global-memory destination address (must be 32-byte aligned).
    vec : opaque vector
        A ``vector<8 x i32>`` (use :func:`subvec` to slice from a larger vector).
    """

    @dsl_user_op
    def _do(addr, v, *, loc=None, ip=None):
        i32_ty = _ir_mod.IntegerType.get_signless(32)
        ir_v = _to_ir(v, loc, ip)
        elems = [
            _vector.extractelement(
                ir_v,
                position=_arith.constant(i32_ty, i, loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
            for i in range(8)
        ]
        operands = [_to_ir(addr, loc, ip)] + elems
        llvm.inline_asm(
            _ir_mod.Type.parse("!llvm.void"),
            operands,
            _STORE_256B_ASM,
            _STORE_256B_CONSTRAINTS,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(gmem_ptr, vec)


# ---------------------------------------------------------------------------
# tcgen05.cp.cta_group::1.128x256b  (via nvvm.tcgen05.cp)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05_cp_128x256b(taddr: int, smem_desc):
    """Async copy SMEM → TMEM with shape ``128x256b`` (``cta_group::1``).

    Issues ``tcgen05.cp.cta_group::1.128x256b  [taddr], s-desc;``
    via the native ``nvvm.tcgen05.cp`` MLIR op.

    The instruction copies a 128-row × 256-bit tile from shared memory
    (described by *smem_desc*) into Tensor Memory at *taddr*.  The copy
    is **asynchronous** — use ``tcgen05.commit`` + ``mbarrier.wait`` to
    synchronize.

    PTX reference
    -------------
        tcgen05.cp.cta_group::1.128x256b  [taddr], s-desc;

    Parameters
    ----------
    taddr : int
        TMEM destination address (uint32, passed as ``!llvm.ptr<6>``).
    smem_desc : i64
        64-bit SMEM matrix descriptor (same format as ``tcgen05.mma``
        descriptors — see ``Tcgen05SmemDescriptor``).
    """

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

    _do(Int32(taddr), smem_desc)
