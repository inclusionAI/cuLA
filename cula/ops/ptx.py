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

"""Shared inline PTX and MLIR helpers for CuTeDSL kernels (SM80+)."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import vector as _vector
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op


def _to_ir(v, loc=None, ip=None):
    if hasattr(v, "ir_value"):
        return v.ir_value(loc=loc, ip=ip)
    return v


@cutlass.dsl_user_op
def cvt_f32_to_tf32(f, *, loc=None, ip=None):
    f_ir = _to_ir(f, loc=loc, ip=ip)
    result = _llvm.inline_asm(
        _T.i32(),
        [f_ir],
        "cvt.rna.tf32.f32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@cutlass.dsl_user_op
def mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    ins = [_to_ir(x, loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)]
    struct_ty = ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>")
    ret = _llvm.inline_asm(
        struct_ty,
        ins,
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = _llvm.extractvalue(_T.f32(), ret, [0], loc=loc, ip=ip)
    d1 = _llvm.extractvalue(_T.f32(), ret, [1], loc=loc, ip=ip)
    d2 = _llvm.extractvalue(_T.f32(), ret, [2], loc=loc, ip=ip)
    d3 = _llvm.extractvalue(_T.f32(), ret, [3], loc=loc, ip=ip)
    return (
        cutlass.Float32(d0),
        cutlass.Float32(d1),
        cutlass.Float32(d2),
        cutlass.Float32(d3),
    )


# ---------------------------------------------------------------------------
# MLIR vector utilities (architecture-independent)
# ---------------------------------------------------------------------------


@cute.jit
def reinterpret_cast(vec, src_type, src_num, tgt_type):
    """Zero-cost reinterpret of a vector's element type (single vector.bitcast)."""
    tgt_num = src_num * src_type.width // tgt_type.width

    @dsl_user_op
    def _do(v, *, loc=None, ip=None):
        tgt_vec_ty = ir.VectorType.get([tgt_num], tgt_type.mlir_type)
        return _vector.bitcast(tgt_vec_ty, _to_ir(v, loc, ip), loc=loc, ip=ip)

    return _do(vec)


@cute.jit
def subvec(vec, offset, size):
    """Extract a contiguous sub-vector (vector.extract_strided_slice)."""

    @dsl_user_op
    def _do(v, *, loc=None, ip=None):
        ir_v = _to_ir(v, loc, ip)
        elem_ty = ir.VectorType(ir_v.type).element_type
        res_ty = ir.VectorType.get([size], elem_ty)
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


_STORE_256B_ASM = "st.global.L1::no_allocate.v8.f32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
_STORE_256B_CONSTRAINTS = "l,r,r,r,r,r,r,r,r"


@cute.jit
def store_256b(gmem_ptr, vec):
    """Store 256 bits (8 x 32-bit) to global memory, bypassing L1 allocation."""

    @dsl_user_op
    def _do(addr, v, *, loc=None, ip=None):
        i32_ty = ir.IntegerType.get_signless(32)
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
        _llvm.inline_asm(
            ir.Type.parse("!llvm.void"),
            operands,
            _STORE_256B_ASM,
            _STORE_256B_CONSTRAINTS,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(gmem_ptr, vec)
