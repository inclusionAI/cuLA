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

"""Portable mma.sync CuTeDSL implementation of KDA intra-chunk backward.

The kernel uses PTX ``mma.sync.m16n8k8`` rather than architecture-specific
warp-group or tcgen05 operations, so the same implementation runs on SM90
and SM100/SM103. One 128-thread CTA owns one ``(chunk, head)`` tile, and
its four warps own the four 16-token subchunks.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import T as _T

BT = 64
BC = 16
BD = 32
THREADS = 128
_cache: dict[tuple[int, ...], object] = {}


@cutlass.dsl_user_op
def _to_tf32_bits(value, *, loc=None, ip=None):
    result = _llvm.inline_asm(
        _T.i32(),
        [Float32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 bits; mov.b32 bits, $1; "
        "and.b32 $0, bits, 0xffffe000; }",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@cutlass.dsl_user_op
def _store_bf16x2(pointer, value0, value1, *, loc=None, ip=None):
    pointer_i64 = pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    _llvm.inline_asm(
        None,
        [
            pointer_i64,
            Float32(value0).ir_value(loc=loc, ip=ip),
            Float32(value1).ir_value(loc=loc, ip=ip),
        ],
        (
            "{ .reg .b16 lo, hi; .reg .b32 packed; "
            "cvt.rn.bf16.f32 lo, $1; "
            "cvt.rn.bf16.f32 hi, $2; "
            "mov.b32 packed, {lo, hi}; "
            "st.global.cg.u32 [$0], packed; }"
        ),
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _store_f32x2(pointer, value0, value1, *, loc=None, ip=None):
    pointer_i64 = pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    _llvm.inline_asm(
        None,
        [
            pointer_i64,
            Float32(value0).ir_value(loc=loc, ip=ip),
            Float32(value1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cg.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _store_f32(pointer, value, *, loc=None, ip=None):
    pointer_i64 = pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    _llvm.inline_asm(
        None,
        [
            pointer_i64,
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cg.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def _load_shared_f32x4(pointer, *, loc=None, ip=None):
    pointer_i32 = pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = _llvm.inline_asm(
        _llvm.StructType.get_literal(
            [_T.f32(), _T.f32(), _T.f32(), _T.f32()]
        ),
        [pointer_i32],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(_llvm.extractvalue(_T.f32(), result, [idx]))
        for idx in range(4)
    )


@cutlass.dsl_user_op
def _load_shared_bf16x4(pointer, *, loc=None, ip=None):
    pointer_i32 = pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = _llvm.inline_asm(
        _llvm.StructType.get_literal(
            [_T.f32(), _T.f32(), _T.f32(), _T.f32()]
        ),
        [pointer_i32],
        (
            "{ .reg .b32 p0, p1; .reg .b16 h0, h1, h2, h3; "
            "ld.shared.v2.b32 {p0, p1}, [$4]; "
            "mov.b32 {h0, h1}, p0; "
            "mov.b32 {h2, h3}, p1; "
            "cvt.f32.bf16 $0, h0; "
            "cvt.f32.bf16 $1, h1; "
            "cvt.f32.bf16 $2, h2; "
            "cvt.f32.bf16 $3, h3; }"
        ),
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(_llvm.extractvalue(_T.f32(), result, [idx]))
        for idx in range(4)
    )


@cutlass.dsl_user_op
def _mma_tf32(
    c0,
    c1,
    c2,
    c3,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    *,
    loc=None,
    ip=None,
):
    result = _llvm.inline_asm(
        _llvm.StructType.get_literal(
            [_T.f32(), _T.f32(), _T.f32(), _T.f32()]
        ),
        [
            Float32(c0).ir_value(loc=loc, ip=ip),
            Float32(c1).ir_value(loc=loc, ip=ip),
            Float32(c2).ir_value(loc=loc, ip=ip),
            Float32(c3).ir_value(loc=loc, ip=ip),
            Int32(a0).ir_value(loc=loc, ip=ip),
            Int32(a1).ir_value(loc=loc, ip=ip),
            Int32(a2).ir_value(loc=loc, ip=ip),
            Int32(a3).ir_value(loc=loc, ip=ip),
            Int32(b0).ir_value(loc=loc, ip=ip),
            Int32(b1).ir_value(loc=loc, ip=ip),
        ],
        (
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{$0,$1,$2,$3}, {$8,$9,$10,$11}, {$12,$13}, "
            "{$4,$5,$6,$7};"
        ),
        "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(_llvm.extractvalue(_T.f32(), result, [idx]))
        for idx in range(4)
    )


@cute.jit
def _tf32_gemm_dual(
    s_daq: cute.Tensor,
    s_dak: cute.Tensor,
    s_b: cute.Tensor,
    acc_q: cute.Tensor,
    acc_k: cute.Tensor,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    transpose: cutlass.Constexpr[bool],
    causal: cutlass.Constexpr[bool],
    sub_len: cutlass.Int32,
    lane: cutlass.Int32,
):
    gid = lane // 4
    lane4 = lane % 4
    for kk in cutlass.range_constexpr(2):
        a_col = kk * 8 + lane4 * 2
        if cutlass.const_expr(transpose):
            aq0 = s_daq[
                (row_block * BC + a_col, col_block * BC + gid)
            ]
            aq1 = s_daq[
                (row_block * BC + a_col, col_block * BC + gid + 8)
            ]
            aq2 = s_daq[
                (row_block * BC + a_col + 1, col_block * BC + gid)
            ]
            aq3 = s_daq[
                (
                    row_block * BC + a_col + 1,
                    col_block * BC + gid + 8,
                )
            ]
            ak0 = s_dak[
                (row_block * BC + a_col, col_block * BC + gid)
            ]
            ak1 = s_dak[
                (row_block * BC + a_col, col_block * BC + gid + 8)
            ]
            ak2 = s_dak[
                (row_block * BC + a_col + 1, col_block * BC + gid)
            ]
            ak3 = s_dak[
                (
                    row_block * BC + a_col + 1,
                    col_block * BC + gid + 8,
                )
            ]
        else:
            aq0 = s_daq[
                (row_block * BC + gid, col_block * BC + a_col)
            ]
            aq1 = s_daq[
                (row_block * BC + gid + 8, col_block * BC + a_col)
            ]
            aq2 = s_daq[
                (row_block * BC + gid, col_block * BC + a_col + 1)
            ]
            aq3 = s_daq[
                (
                    row_block * BC + gid + 8,
                    col_block * BC + a_col + 1,
                )
            ]
            ak0 = s_dak[
                (row_block * BC + gid, col_block * BC + a_col)
            ]
            ak1 = s_dak[
                (row_block * BC + gid + 8, col_block * BC + a_col)
            ]
            ak2 = s_dak[
                (row_block * BC + gid, col_block * BC + a_col + 1)
            ]
            ak3 = s_dak[
                (
                    row_block * BC + gid + 8,
                    col_block * BC + a_col + 1,
                )
            ]
        if cutlass.const_expr(causal):
            if cutlass.const_expr(transpose):
                if not (
                    gid <= a_col
                    and a_col < sub_len
                    and gid < sub_len
                ):
                    aq0 = 0.0
                    ak0 = 0.0
                if not (
                    gid + 8 <= a_col
                    and a_col < sub_len
                    and gid + 8 < sub_len
                ):
                    aq1 = 0.0
                    ak1 = 0.0
                if not (
                    gid <= a_col + 1
                    and a_col + 1 < sub_len
                    and gid < sub_len
                ):
                    aq2 = 0.0
                    ak2 = 0.0
                if not (
                    gid + 8 <= a_col + 1
                    and a_col + 1 < sub_len
                    and gid + 8 < sub_len
                ):
                    aq3 = 0.0
                    ak3 = 0.0
            else:
                if not (
                    a_col <= gid
                    and gid < sub_len
                    and a_col < sub_len
                ):
                    aq0 = 0.0
                    ak0 = 0.0
                if not (
                    a_col <= gid + 8
                    and gid + 8 < sub_len
                    and a_col < sub_len
                ):
                    aq1 = 0.0
                    ak1 = 0.0
                if not (
                    a_col + 1 <= gid
                    and gid < sub_len
                    and a_col + 1 < sub_len
                ):
                    aq2 = 0.0
                    ak2 = 0.0
                if not (
                    a_col + 1 <= gid + 8
                    and gid + 8 < sub_len
                    and a_col + 1 < sub_len
                ):
                    aq3 = 0.0
                    ak3 = 0.0
        aq_bits = (
            _to_tf32_bits(aq0),
            _to_tf32_bits(aq1),
            _to_tf32_bits(aq2),
            _to_tf32_bits(aq3),
        )
        ak_bits = (
            _to_tf32_bits(ak0),
            _to_tf32_bits(ak1),
            _to_tf32_bits(ak2),
            _to_tf32_bits(ak3),
        )
        b_k = kk * 8 + lane4 * 2
        for nt in cutlass.range_constexpr(4):
            base = nt * 4
            n_base = nt * 8
            b0 = _to_tf32_bits(s_b[(n_base + gid, b_k)])
            b1 = _to_tf32_bits(s_b[(n_base + gid, b_k + 1)])
            q_result = _mma_tf32(
                acc_q[base],
                acc_q[base + 1],
                acc_q[base + 2],
                acc_q[base + 3],
                aq_bits[0],
                aq_bits[1],
                aq_bits[2],
                aq_bits[3],
                b0,
                b1,
            )
            k_result = _mma_tf32(
                acc_k[base],
                acc_k[base + 1],
                acc_k[base + 2],
                acc_k[base + 3],
                ak_bits[0],
                ak_bits[1],
                ak_bits[2],
                ak_bits[3],
                b0,
                b1,
            )
            for value_idx in cutlass.range_constexpr(4):
                acc_q[base + value_idx] = q_result[value_idx]
                acc_k[base + value_idx] = k_result[value_idx]


@cute.jit
def _tf32_gemm_single(
    s_a: cute.Tensor,
    s_b: cute.Tensor,
    acc: cute.Tensor,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    transpose: cutlass.Constexpr[bool],
    causal: cutlass.Constexpr[bool],
    sub_len: cutlass.Int32,
    lane: cutlass.Int32,
):
    gid = lane // 4
    lane4 = lane % 4
    for kk in cutlass.range_constexpr(2):
        a_col = kk * 8 + lane4 * 2
        if cutlass.const_expr(transpose):
            a0 = s_a[
                (row_block * BC + a_col, col_block * BC + gid)
            ]
            a1 = s_a[
                (row_block * BC + a_col, col_block * BC + gid + 8)
            ]
            a2 = s_a[
                (row_block * BC + a_col + 1, col_block * BC + gid)
            ]
            a3 = s_a[
                (
                    row_block * BC + a_col + 1,
                    col_block * BC + gid + 8,
                )
            ]
        else:
            a0 = s_a[
                (row_block * BC + gid, col_block * BC + a_col)
            ]
            a1 = s_a[
                (row_block * BC + gid + 8, col_block * BC + a_col)
            ]
            a2 = s_a[
                (row_block * BC + gid, col_block * BC + a_col + 1)
            ]
            a3 = s_a[
                (
                    row_block * BC + gid + 8,
                    col_block * BC + a_col + 1,
                )
            ]
        if cutlass.const_expr(causal):
            if cutlass.const_expr(transpose):
                if not (
                    gid <= a_col
                    and a_col < sub_len
                    and gid < sub_len
                ):
                    a0 = 0.0
                if not (
                    gid + 8 <= a_col
                    and a_col < sub_len
                    and gid + 8 < sub_len
                ):
                    a1 = 0.0
                if not (
                    gid <= a_col + 1
                    and a_col + 1 < sub_len
                    and gid < sub_len
                ):
                    a2 = 0.0
                if not (
                    gid + 8 <= a_col + 1
                    and a_col + 1 < sub_len
                    and gid + 8 < sub_len
                ):
                    a3 = 0.0
            else:
                if not (
                    a_col <= gid
                    and gid < sub_len
                    and a_col < sub_len
                ):
                    a0 = 0.0
                if not (
                    a_col <= gid + 8
                    and gid + 8 < sub_len
                    and a_col < sub_len
                ):
                    a1 = 0.0
                if not (
                    a_col + 1 <= gid
                    and gid < sub_len
                    and a_col + 1 < sub_len
                ):
                    a2 = 0.0
                if not (
                    a_col + 1 <= gid + 8
                    and gid + 8 < sub_len
                    and a_col + 1 < sub_len
                ):
                    a3 = 0.0
        a_bits = (
            _to_tf32_bits(a0),
            _to_tf32_bits(a1),
            _to_tf32_bits(a2),
            _to_tf32_bits(a3),
        )
        b_k = kk * 8 + lane4 * 2
        for nt in cutlass.range_constexpr(4):
            base = nt * 4
            n_base = nt * 8
            b0 = _to_tf32_bits(s_b[(n_base + gid, b_k)])
            b1 = _to_tf32_bits(s_b[(n_base + gid, b_k + 1)])
            result = _mma_tf32(
                acc[base],
                acc[base + 1],
                acc[base + 2],
                acc[base + 3],
                a_bits[0],
                a_bits[1],
                a_bits[2],
                a_bits[3],
                b0,
                b1,
            )
            for value_idx in cutlass.range_constexpr(4):
                acc[base + value_idx] = result[value_idx]


@cute.jit
def _tf32_gemm_two_b(
    s_daq: cute.Tensor,
    s_dak: cute.Tensor,
    s_bq: cute.Tensor,
    s_bk: cute.Tensor,
    acc: cute.Tensor,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    transpose: cutlass.Constexpr[bool],
    causal: cutlass.Constexpr[bool],
    sub_len: cutlass.Int32,
    lane: cutlass.Int32,
):
    _tf32_gemm_single(
        s_daq,
        s_bq,
        acc,
        row_block,
        col_block,
        transpose,
        causal,
        sub_len,
        lane,
    )
    _tf32_gemm_single(
        s_dak,
        s_bk,
        acc,
        row_block,
        col_block,
        transpose,
        causal,
        sub_len,
        lane,
    )


@cute.jit
def _gemm_dual(
    tiled_mma: cute.TiledMma,
    s_aq: cute.Tensor,
    s_ak: cute.Tensor,
    s_b: cute.Tensor,
    acc_q: cute.Tensor,
    acc_k: cute.Tensor,
    lane: cutlass.Int32,
):
    thr_mma = tiled_mma.get_slice(lane)
    r_aq = tiled_mma.make_fragment_A(thr_mma.partition_A(s_aq))
    r_ak = tiled_mma.make_fragment_A(thr_mma.partition_A(s_ak))
    r_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_b))

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        cutlass.BFloat16,
    )
    copy_a = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    copy_b = cute.make_tiled_copy_B(copy_atom, tiled_mma)
    thr_a = copy_a.get_slice(lane)
    thr_b = copy_b.get_slice(lane)
    saq = thr_a.partition_S(s_aq)
    sak = thr_a.partition_S(s_ak)
    sb = thr_b.partition_S(s_b)
    raq = thr_a.retile(r_aq)
    rak = thr_a.retile(r_ak)
    rb = thr_b.retile(r_b)

    for kb in cutlass.range_constexpr(cute.size(r_aq, mode=[2])):
        cute.copy(copy_b, sb[(None, None, kb)], rb[(None, None, kb)])
        cute.copy(copy_a, saq[(None, None, kb)], raq[(None, None, kb)])
        cute.gemm(
            tiled_mma,
            acc_q,
            r_aq[(None, None, kb)],
            r_b[(None, None, kb)],
            acc_q,
        )
        cute.copy(copy_a, sak[(None, None, kb)], rak[(None, None, kb)])
        cute.gemm(
            tiled_mma,
            acc_k,
            r_ak[(None, None, kb)],
            r_b[(None, None, kb)],
            acc_k,
        )


@cute.jit
def _gemm_two_b(
    tiled_mma: cute.TiledMma,
    s_aq: cute.Tensor,
    s_ak: cute.Tensor,
    s_bq: cute.Tensor,
    s_bk: cute.Tensor,
    acc: cute.Tensor,
    lane: cutlass.Int32,
):
    thr_mma = tiled_mma.get_slice(lane)
    r_aq = tiled_mma.make_fragment_A(thr_mma.partition_A(s_aq))
    r_ak = tiled_mma.make_fragment_A(thr_mma.partition_A(s_ak))
    r_bq = tiled_mma.make_fragment_B(thr_mma.partition_B(s_bq))
    r_bk = tiled_mma.make_fragment_B(thr_mma.partition_B(s_bk))

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        cutlass.BFloat16,
    )
    copy_a = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    copy_b = cute.make_tiled_copy_B(copy_atom, tiled_mma)
    thr_a = copy_a.get_slice(lane)
    thr_b = copy_b.get_slice(lane)
    saq = thr_a.partition_S(s_aq)
    sak = thr_a.partition_S(s_ak)
    sbq = thr_b.partition_S(s_bq)
    sbk = thr_b.partition_S(s_bk)
    raq = thr_a.retile(r_aq)
    rak = thr_a.retile(r_ak)
    rbq = thr_b.retile(r_bq)
    rbk = thr_b.retile(r_bk)

    for kb in cutlass.range_constexpr(cute.size(r_aq, mode=[2])):
        cute.copy(copy_a, saq[(None, None, kb)], raq[(None, None, kb)])
        cute.copy(copy_b, sbq[(None, None, kb)], rbq[(None, None, kb)])
        cute.gemm(
            tiled_mma,
            acc,
            r_aq[(None, None, kb)],
            r_bq[(None, None, kb)],
            acc,
        )
        cute.copy(copy_a, sak[(None, None, kb)], rak[(None, None, kb)])
        cute.copy(copy_b, sbk[(None, None, kb)], rbk[(None, None, kb)])
        cute.gemm(
            tiled_mma,
            acc,
            r_ak[(None, None, kb)],
            r_bk[(None, None, kb)],
            acc,
        )


@cute.jit
def _stage_a(
    s_daq: cute.Tensor,
    s_dak: cute.Tensor,
    s_aq: cute.Tensor,
    s_ak: cute.Tensor,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    transpose: cutlass.Constexpr[bool],
    lane: cutlass.Int32,
):
    for item in cutlass.range_constexpr(8):
        linear = lane + item * 32
        row = linear // BC
        col = linear % BC
        src_row = row_block * BC + row
        src_col = col_block * BC + col
        if cutlass.const_expr(transpose):
            s_aq[(col, row)] = s_daq[(src_row, src_col)]
            s_ak[(col, row)] = s_dak[(src_row, src_col)]
        else:
            s_aq[(row, col)] = s_daq[(src_row, src_col)]
            s_ak[(row, col)] = s_dak[(src_row, src_col)]
    cute.arch.sync_warp()


@cute.jit
def _fill_lower_b(
    s_k: cute.Tensor,
    s_g: cute.Tensor,
    s_b: cute.Tensor,
    source_block: cutlass.Int32,
    output_block: cutlass.Int32,
    anchor_row: cutlass.Int32,
    tile_len: cutlass.Int32,
    lane: cutlass.Int32,
):
    for item in cutlass.range_constexpr(4):
        row = item * 4 + lane // 8
        feature = (lane % 8) * 4
        source_row = source_block * BC + row
        anchor = output_block * BC + anchor_row
        for value_idx in cutlass.range_constexpr(4):
            value = cutlass.Float32(0.0)
            if source_row < tile_len:
                gate = cute.math.exp2(
                    cutlass.Float32(s_g[(anchor, feature + value_idx)])
                    - cutlass.Float32(
                        s_g[(source_row, feature + value_idx)]
                    ),
                    fastmath=True,
                )
                value = (
                    cutlass.Float32(
                        s_k[(source_row, feature + value_idx)]
                    )
                    * gate
                )
            s_b[(feature + value_idx, row)] = cutlass.BFloat16(value)
    cute.arch.sync_warp()


@cute.jit
def _fill_upper_b(
    s_q: cute.Tensor,
    s_k: cute.Tensor,
    s_g: cute.Tensor,
    s_beta: cute.Tensor,
    s_bq: cute.Tensor,
    s_bk: cute.Tensor,
    source_block: cutlass.Int32,
    output_block: cutlass.Int32,
    anchor_row: cutlass.Int32,
    tile_len: cutlass.Int32,
    lane: cutlass.Int32,
):
    for item in cutlass.range_constexpr(4):
        row = item * 4 + lane // 8
        feature = (lane % 8) * 4
        source_row = source_block * BC + row
        anchor = output_block * BC + anchor_row
        beta_value = cutlass.Float32(0.0)
        if source_row < tile_len:
            beta_value = cutlass.Float32(s_beta[source_row])
        for value_idx in cutlass.range_constexpr(4):
            q_value = cutlass.Float32(0.0)
            k_value = cutlass.Float32(0.0)
            if source_row < tile_len:
                gate = cute.math.exp2(
                    cutlass.Float32(
                        s_g[(source_row, feature + value_idx)]
                    )
                    - cutlass.Float32(
                        s_g[(anchor, feature + value_idx)]
                    ),
                    fastmath=True,
                )
                q_value = (
                    cutlass.Float32(
                        s_q[(source_row, feature + value_idx)]
                    )
                    * gate
                )
                k_value = (
                    cutlass.Float32(
                        cutlass.BFloat16(
                            cutlass.Float32(
                                s_k[(source_row, feature + value_idx)]
                            )
                            * beta_value
                        )
                    )
                    * gate
                )
            s_bq[(feature + value_idx, row)] = cutlass.BFloat16(q_value)
            s_bk[(feature + value_idx, row)] = cutlass.BFloat16(k_value)
    cute.arch.sync_warp()


@cute.jit
def _load_qkg(
    s_q: cute.Tensor,
    s_k: cute.Tensor,
    s_g: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    chunk_start: cutlass.Int32,
    tile_len: cutlass.Int32,
    head: cutlass.Int32,
    feature_start: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bf16_async = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    f32_async = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    bf16_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    f32_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=128,
    )

    for item in cutlass.range_constexpr(2):
        group = tidx + item * THREADS
        row = group // 4
        feature = (group % 4) * 8
        sq = cute.make_tensor(
            (s_q.iterator + s_q.layout((row, feature))).align(16),
            cute.make_layout((8,), stride=(1,)),
        )
        sk = cute.make_tensor(
            (s_k.iterator + s_k.layout((row, feature))).align(16),
            cute.make_layout((8,), stride=(1,)),
        )
        if row < tile_len:
            token = chunk_start + row
            gq = cute.make_tensor(
                (
                    q.iterator
                    + q.layout((0, token, head, feature_start + feature))
                ).align(16),
                cute.make_layout((8,), stride=(1,)),
            )
            gk = cute.make_tensor(
                (
                    k.iterator
                    + k.layout((0, token, head, feature_start + feature))
                ).align(16),
                cute.make_layout((8,), stride=(1,)),
            )
            cute.copy(bf16_async, gq, sq)
            cute.copy(bf16_async, gk, sk)
        else:
            zero = cute.make_rmem_tensor(
                cute.make_layout((8,), stride=(1,)),
                cutlass.BFloat16,
            )
            zero.fill(0.0)
            cute.copy(bf16_copy, zero, sq)
            cute.copy(bf16_copy, zero, sk)

    for item in cutlass.range_constexpr(4):
        group = tidx + item * THREADS
        row = group // 8
        feature = (group % 8) * 4
        sg = cute.make_tensor(
            (s_g.iterator + s_g.layout((row, feature))).align(16),
            cute.make_layout((4,), stride=(1,)),
        )
        if row < tile_len:
            token = chunk_start + row
            gg = cute.make_tensor(
                (
                    g.iterator
                    + g.layout((0, token, head, feature_start + feature))
                ).align(16),
                cute.make_layout((4,), stride=(1,)),
            )
            cute.copy(f32_async, gg, sg)
        else:
            zero = cute.make_rmem_tensor(
                cute.make_layout((4,), stride=(1,)),
                cutlass.Float32,
            )
            zero.fill(0.0)
            cute.copy(f32_copy, zero, sg)

    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()


@cute.jit
def _fill_lower_b_f32(
    s_k: cute.Tensor,
    s_g: cute.Tensor,
    s_b: cute.Tensor,
    source_block: cutlass.Int32,
    output_block: cutlass.Int32,
    anchor_row: cutlass.Int32,
    tile_len: cutlass.Int32,
    lane: cutlass.Int32,
):
    copy_128 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    feature = (lane % 8) * 4
    anchor = output_block * BC + anchor_row
    anchor_values = _load_shared_f32x4(
        s_g.iterator + s_g.layout((anchor, feature))
    )
    for item in cutlass.range_constexpr(4):
        row = item * 4 + lane // 8
        source_row = source_block * BC + row
        k_values = (
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
        )
        source_values = k_values
        if source_row < tile_len:
            k_values = _load_shared_bf16x4(
                s_k.iterator + s_k.layout((source_row, feature))
            )
            source_values = _load_shared_f32x4(
                s_g.iterator + s_g.layout((source_row, feature))
            )
        values = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)),
            cutlass.Float32,
        )
        for value_idx in cutlass.range_constexpr(4):
            value = cutlass.Float32(0.0)
            if source_row < tile_len:
                value = (
                    k_values[value_idx]
                    * cute.math.exp2(
                        anchor_values[value_idx]
                        - source_values[value_idx],
                        fastmath=True,
                    )
                )
            values[value_idx] = value
        destination = cute.make_tensor(
            (s_b.iterator + s_b.layout((feature, row))).align(16),
            cute.make_layout((4,), stride=(1,)),
        )
        cute.copy(copy_128, values, destination)
    cute.arch.sync_warp()


@cute.jit
def _fill_upper_b_f32(
    s_q: cute.Tensor,
    s_k: cute.Tensor,
    s_g: cute.Tensor,
    s_beta: cute.Tensor,
    s_bq: cute.Tensor,
    s_bk: cute.Tensor,
    source_block: cutlass.Int32,
    output_block: cutlass.Int32,
    anchor_row: cutlass.Int32,
    tile_len: cutlass.Int32,
    lane: cutlass.Int32,
    beta_is_f32: cutlass.Constexpr[bool],
):
    copy_128 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    feature = (lane % 8) * 4
    anchor = output_block * BC + anchor_row
    anchor_values = _load_shared_f32x4(
        s_g.iterator + s_g.layout((anchor, feature))
    )
    for item in cutlass.range_constexpr(4):
        row = item * 4 + lane // 8
        source_row = source_block * BC + row
        beta_value = cutlass.Float32(0.0)
        if source_row < tile_len:
            beta_value = cutlass.Float32(s_beta[source_row])
        zero_values = (
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
        )
        q_input = zero_values
        k_input = zero_values
        source_values = zero_values
        if source_row < tile_len:
            q_input = _load_shared_bf16x4(
                s_q.iterator + s_q.layout((source_row, feature))
            )
            k_input = _load_shared_bf16x4(
                s_k.iterator + s_k.layout((source_row, feature))
            )
            source_values = _load_shared_f32x4(
                s_g.iterator + s_g.layout((source_row, feature))
            )
        q_values = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)),
            cutlass.Float32,
        )
        k_values = cute.make_rmem_tensor_like(
            q_values, cutlass.Float32
        )
        for value_idx in cutlass.range_constexpr(4):
            q_value = cutlass.Float32(0.0)
            k_value = cutlass.Float32(0.0)
            if source_row < tile_len:
                gate = cute.math.exp2(
                    source_values[value_idx]
                    - anchor_values[value_idx],
                    fastmath=True,
                )
                q_value = q_input[value_idx] * gate
                k_beta = k_input[value_idx] * beta_value
                if cutlass.const_expr(not beta_is_f32):
                    k_beta = cutlass.Float32(
                        cutlass.BFloat16(k_beta)
                    )
                k_value = k_beta * gate
            q_values[value_idx] = q_value
            k_values[value_idx] = k_value
        q_destination = cute.make_tensor(
            (s_bq.iterator + s_bq.layout((feature, row))).align(16),
            cute.make_layout((4,), stride=(1,)),
        )
        k_destination = cute.make_tensor(
            (s_bk.iterator + s_bk.layout((feature, row))).align(16),
            cute.make_layout((4,), stride=(1,)),
        )
        cute.copy(copy_128, q_values, q_destination)
        cute.copy(copy_128, k_values, k_destination)
    cute.arch.sync_warp()


@cute.jit
def _load_gradient_block(
    s_destination: cute.Tensor,
    gradient: cute.Tensor,
    chunk_start: cutlass.Int32,
    output_block: cutlass.Int32,
    head: cutlass.Int32,
    feature_start: cutlass.Int32,
    sub_len: cutlass.Int32,
    lane: cutlass.Int32,
):
    async_copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    regular_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    for item in cutlass.range_constexpr(4):
        row = item * 4 + lane // 8
        feature = (lane % 8) * 4
        destination = cute.make_tensor(
            (
                s_destination.iterator
                + s_destination.layout((feature, row))
            ).align(16),
            cute.make_layout((4,), stride=(1,)),
        )
        if row < sub_len:
            token = chunk_start + output_block * BC + row
            source = cute.make_tensor(
                (
                    gradient.iterator
                    + gradient.layout(
                        (0, token, head, feature_start + feature)
                    )
                ).align(16),
                cute.make_layout((4,), stride=(1,)),
            )
            cute.copy(async_copy, source, destination)
        else:
            zero = cute.make_rmem_tensor(
                cute.make_layout((4,), stride=(1,)),
                cutlass.Float32,
            )
            zero.fill(0.0)
            cute.copy(regular_copy, zero, destination)
    cute.arch.cp_async_commit_group()


@cute.kernel
def _kernel(
    tiled_mma: cute.TiledMma,
    a_tile_layout: cute.Layout,
    a_all_layout: cute.Layout,
    b_tile_layout: cute.Layout,
    b_all_layout: cute.Layout,
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    d_aq: cute.Tensor,
    d_ak: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    db: cute.Tensor,
    dg: cute.Tensor,
    cu_seqlens: cute.Tensor,
    chunk_indices: cute.Tensor,
    dq_out: cute.Tensor,
    dk_out: cute.Tensor,
    db_out: cute.Tensor,
    dg_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    chunk_idx, head, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32

    seq_idx = cutlass.Int32(chunk_indices[(chunk_idx, 0)])
    chunk_in_seq = cutlass.Int32(chunk_indices[(chunk_idx, 1)])
    seq_start = cutlass.Int32(cu_seqlens[seq_idx])
    seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
    chunk_start = seq_start + chunk_in_seq * BT
    tile_len = seq_end - chunk_start
    if tile_len > BT:
        tile_len = cutlass.Int32(BT)

    tile_input_layout = cute.make_layout(
        (1, BT, 1, q.shape[3]),
        stride=(0, q.shape[2] * q.shape[3], 0, 1),
    )
    q_tile = cute.make_tensor(
        (
            q.iterator + q.layout((0, chunk_start, head, 0))
        ).align(16),
        tile_input_layout,
    )
    k_tile = cute.make_tensor(
        (
            k.iterator + k.layout((0, chunk_start, head, 0))
        ).align(16),
        tile_input_layout,
    )
    g_tile = cute.make_tensor(
        (
            g.iterator + g.layout((0, chunk_start, head, 0))
        ).align(16),
        tile_input_layout,
    )
    smem = cutlass.utils.SmemAllocator()
    qkg_layout = cute.make_layout((BT, BD), stride=(BD, 1))
    da_layout = cute.make_layout((BT, BT), stride=(BT + 4, 1))
    s_q = smem.allocate_tensor(cutlass.BFloat16, qkg_layout, 128)
    s_k = smem.allocate_tensor(cutlass.BFloat16, qkg_layout, 128)
    s_g = smem.allocate_tensor(cutlass.Float32, qkg_layout, 128)
    s_beta = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BT,), stride=(1,)),
        128,
    )
    s_daq = smem.allocate_tensor(cutlass.BFloat16, da_layout, 128)
    s_dak = smem.allocate_tensor(cutlass.BFloat16, da_layout, 128)
    s_bq_all = smem.allocate_tensor(
        cutlass.BFloat16, b_all_layout, 128
    )
    s_bk_all = smem.allocate_tensor(
        cutlass.BFloat16, b_all_layout, 128
    )
    s_aq_all = smem.allocate_tensor(
        cutlass.BFloat16, a_all_layout, 128
    )
    s_ak_all = smem.allocate_tensor(
        cutlass.BFloat16, a_all_layout, 128
    )

    if tidx < BT:
        value = cutlass.Float32(0.0)
        if tidx < tile_len:
            value = cutlass.Float32(beta[(0, chunk_start + tidx, head)])
        s_beta[tidx] = value

    f32_copy = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    for item in cutlass.range_constexpr(8):
        group = tidx + item * THREADS
        row = group // 16
        col = (group % 16) * 4
        aq = cute.make_rmem_tensor(
            cute.make_layout((4,), stride=(1,)),
            cutlass.Float32,
        )
        ak = cute.make_rmem_tensor_like(aq, cutlass.Float32)
        aq.fill(0.0)
        ak.fill(0.0)
        if row < tile_len:
            token = chunk_start + row
            gaq = cute.make_tensor(
                (
                    d_aq.iterator
                    + d_aq.layout((0, token, head, col))
                ).align(16),
                cute.make_layout((4,), stride=(1,)),
            )
            gak = cute.make_tensor(
                (
                    d_ak.iterator
                    + d_ak.layout((0, token, head, col))
                ).align(16),
                cute.make_layout((4,), stride=(1,)),
            )
            cute.copy(f32_copy, gaq, aq)
            cute.copy(f32_copy, gak, ak)
        for value_idx in cutlass.range_constexpr(4):
            valid = col + value_idx <= row
            s_daq[(row, col + value_idx)] = (
                cutlass.BFloat16(aq[value_idx])
                if valid
                else cutlass.BFloat16(0.0)
            )
            s_dak[(row, col + value_idx)] = (
                cutlass.BFloat16(ak[value_idx])
                if valid
                else cutlass.BFloat16(0.0)
            )
    cute.arch.sync_threads()

    s_aq = cute.make_tensor(
        s_aq_all.iterator + s_aq_all.layout((warp * BC, 0)),
        a_tile_layout,
    )
    s_ak = cute.make_tensor(
        s_ak_all.iterator + s_ak_all.layout((warp * BC, 0)),
        a_tile_layout,
    )
    s_bq = cute.make_tensor(
        s_bq_all.iterator + s_bq_all.layout((warp * BD, 0)),
        b_tile_layout,
    )
    s_bk = cute.make_tensor(
        s_bk_all.iterator + s_bk_all.layout((warp * BD, 0)),
        b_tile_layout,
    )

    thr_mma = tiled_mma.get_slice(lane)
    acc_shape = thr_mma.partition_shape_C((BC, BD))
    ident = cute.make_identity_tensor((BC, BD))
    coord = thr_mma.partition_C(ident)

    for feature_tile in cutlass.range(4, unroll=1):
        feature_start = feature_tile * BD
        _load_qkg(
            s_q,
            s_k,
            s_g,
            q_tile,
            k_tile,
            g_tile,
            cutlass.Int32(0),
            tile_len,
            cutlass.Int32(0),
            feature_start,
        )

        sub_len = tile_len - warp * BC
        if sub_len > BC:
            sub_len = cutlass.Int32(BC)

        if sub_len > 0:
            acc_dq = tiled_mma.make_fragment_C(acc_shape)
            acc_dkl = tiled_mma.make_fragment_C(acc_shape)
            acc_dku = tiled_mma.make_fragment_C(acc_shape)
            acc_dq.fill(0.0)
            acc_dkl.fill(0.0)
            acc_dku.fill(0.0)

            for source in cutlass.range(4, unroll=1):
                if source < warp:
                    _stage_a(
                        s_daq,
                        s_dak,
                        s_aq,
                        s_ak,
                        warp,
                        source,
                        False,
                        lane,
                    )
                    _fill_lower_b(
                        s_k,
                        s_g,
                        s_bq,
                        source,
                        warp,
                        cutlass.Int32(0),
                        tile_len,
                        lane,
                    )
                    _gemm_dual(
                        tiled_mma,
                        s_aq,
                        s_ak,
                        s_bq,
                        acc_dq,
                        acc_dkl,
                        lane,
                    )
                    cute.arch.sync_warp()

            for idx in cutlass.range_constexpr(cute.size(acc_dq)):
                row, feature = coord[idx]
                scale = cute.math.exp2(
                    cutlass.Float32(
                        s_g[(warp * BC + row, feature)]
                    )
                    - cutlass.Float32(s_g[(warp * BC, feature)]),
                    fastmath=True,
                )
                acc_dq[idx] *= scale
                acc_dkl[idx] *= scale

            mid = cutlass.Int32(BC // 2)
            if mid >= sub_len:
                mid = sub_len - 1
            tmp_q = tiled_mma.make_fragment_C(acc_shape)
            tmp_k = tiled_mma.make_fragment_C(acc_shape)
            tmp_q.fill(0.0)
            tmp_k.fill(0.0)
            _stage_a(
                s_daq,
                s_dak,
                s_aq,
                s_ak,
                warp,
                warp,
                False,
                lane,
            )
            _fill_lower_b(
                s_k,
                s_g,
                s_bq,
                warp,
                warp,
                mid,
                tile_len,
                lane,
            )
            _gemm_dual(
                tiled_mma,
                s_aq,
                s_ak,
                s_bq,
                tmp_q,
                tmp_k,
                lane,
            )
            for idx in cutlass.range_constexpr(cute.size(acc_dq)):
                row, feature = coord[idx]
                scale = cute.math.exp2(
                    cutlass.Float32(
                        s_g[(warp * BC + row, feature)]
                    )
                    - cutlass.Float32(
                        s_g[(warp * BC + mid, feature)]
                    ),
                    fastmath=True,
                )
                acc_dq[idx] += tmp_q[idx] * scale
                acc_dkl[idx] += tmp_k[idx] * scale
            cute.arch.sync_warp()

            last = sub_len - 1
            for source in cutlass.range(4, unroll=1):
                if source > warp and source * BC < tile_len:
                    _stage_a(
                        s_daq,
                        s_dak,
                        s_aq,
                        s_ak,
                        source,
                        warp,
                        True,
                        lane,
                    )
                    _fill_upper_b(
                        s_q,
                        s_k,
                        s_g,
                        s_beta,
                        s_bq,
                        s_bk,
                        source,
                        warp,
                        last,
                        tile_len,
                        lane,
                    )
                    _gemm_two_b(
                        tiled_mma,
                        s_aq,
                        s_ak,
                        s_bq,
                        s_bk,
                        acc_dku,
                        lane,
                    )
                    cute.arch.sync_warp()

            for idx in cutlass.range_constexpr(cute.size(acc_dku)):
                row, feature = coord[idx]
                acc_dku[idx] *= cute.math.exp2(
                    cutlass.Float32(
                        s_g[(warp * BC + last, feature)]
                    )
                    - cutlass.Float32(
                        s_g[(warp * BC + row, feature)]
                    ),
                    fastmath=True,
                )

            tmp_u = tiled_mma.make_fragment_C(acc_shape)
            tmp_u.fill(0.0)
            _stage_a(
                s_daq,
                s_dak,
                s_aq,
                s_ak,
                warp,
                warp,
                True,
                lane,
            )
            _fill_upper_b(
                s_q,
                s_k,
                s_g,
                s_beta,
                s_bq,
                s_bk,
                warp,
                warp,
                mid,
                tile_len,
                lane,
            )
            _gemm_two_b(
                tiled_mma,
                s_aq,
                s_ak,
                s_bq,
                s_bk,
                tmp_u,
                lane,
            )

            row0 = lane // 4
            row1 = row0 + 8
            db0 = cutlass.Float32(0.0)
            db1 = cutlass.Float32(0.0)
            for idx in cutlass.range_constexpr(cute.size(acc_dq)):
                row, feature = coord[idx]
                upper_scale = cute.math.exp2(
                    cutlass.Float32(
                        s_g[(warp * BC + mid, feature)]
                    )
                    - cutlass.Float32(
                        s_g[(warp * BC + row, feature)]
                    ),
                    fastmath=True,
                )
                dku = cutlass.Float32(acc_dku[idx]) + cutlass.Float32(
                    tmp_u[idx]
                ) * upper_scale
                dqi = cutlass.Float32(acc_dq[idx])
                dkl = cutlass.Float32(acc_dkl[idx])
                beta_value = cutlass.Float32(s_beta[warp * BC + row])
                dkl_beta = dkl * beta_value
                q_value = cutlass.Float32(
                    s_q[(warp * BC + row, feature)]
                )
                k_value = cutlass.Float32(
                    s_k[(warp * BC + row, feature)]
                )
                token = chunk_start + warp * BC + row
                dim = feature_start + feature

                dq_out[(0, token, head, dim)] = cutlass.BFloat16(
                    cutlass.Float32(dq[(0, token, head, dim)]) + dqi
                )
                dk_out[(0, token, head, dim)] = cutlass.BFloat16(
                    cutlass.Float32(dk[(0, token, head, dim)])
                    + dkl_beta
                    + dku
                )
                dg_out[(0, token, head, dim)] = (
                    cutlass.Float32(dg[(0, token, head, dim)])
                    + q_value * dqi
                    + (dkl_beta - dku) * k_value
                )
                contribution = dkl * k_value
                if row == row0:
                    db0 += contribution
                if row == row1:
                    db1 += contribution

            db0 += cute.arch.shuffle_sync_bfly(
                db0, offset=1, mask=-1, mask_and_clamp=31
            )
            db0 += cute.arch.shuffle_sync_bfly(
                db0, offset=2, mask=-1, mask_and_clamp=31
            )
            db1 += cute.arch.shuffle_sync_bfly(
                db1, offset=1, mask=-1, mask_and_clamp=31
            )
            db1 += cute.arch.shuffle_sync_bfly(
                db1, offset=2, mask=-1, mask_and_clamp=31
            )
            if lane % 4 == 0:
                token0 = chunk_start + warp * BC + row0
                value0 = db0
                if feature_tile == 0:
                    value0 += cutlass.Float32(db[(0, token0, head)])
                ptr0 = (
                    db_out.iterator
                    + db_out.layout((0, token0, head))
                )
                cute.arch.atomic_add(
                    ptr0.llvm_ptr,
                    value0,
                    sem="relaxed",
                    scope="gpu",
                )
                if row1 < sub_len:
                    token1 = chunk_start + warp * BC + row1
                    value1 = db1
                    if feature_tile == 0:
                        value1 += cutlass.Float32(db[(0, token1, head)])
                    ptr1 = (
                        db_out.iterator
                        + db_out.layout((0, token1, head))
                    )
                    cute.arch.atomic_add(
                        ptr1.llvm_ptr,
                        value1,
                        sem="relaxed",
                        scope="gpu",
                    )
        cute.arch.sync_threads()


@cute.jit
def _process_tf32_tile(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    d_aq: cute.Tensor,
    d_ak: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    db: cute.Tensor,
    dg: cute.Tensor,
    cu_seqlens: cute.Tensor,
    chunk_indices: cute.Tensor,
    dq_out: cute.Tensor,
    dk_out: cute.Tensor,
    db_out: cute.Tensor,
    dg_out: cute.Tensor,
    chunk_idx: cutlass.Int32,
    head: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    gid = lane // 4
    lane4 = lane % 4

    seq_idx = cutlass.Int32(chunk_indices[(chunk_idx, 0)])
    chunk_in_seq = cutlass.Int32(chunk_indices[(chunk_idx, 1)])
    seq_start = cutlass.Int32(cu_seqlens[seq_idx])
    seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
    chunk_start = seq_start + chunk_in_seq * BT
    tile_len = seq_end - chunk_start
    if tile_len > BT:
        tile_len = cutlass.Int32(BT)

    tile_input_layout = cute.make_layout(
        (1, BT, 1, q.shape[3]),
        stride=(0, q.shape[2] * q.shape[3], 0, 1),
    )
    q_tile = cute.make_tensor(
        (
            q.iterator + q.layout((0, chunk_start, head, 0))
        ).align(16),
        tile_input_layout,
    )
    k_tile = cute.make_tensor(
        (
            k.iterator + k.layout((0, chunk_start, head, 0))
        ).align(16),
        tile_input_layout,
    )
    g_tile = cute.make_tensor(
        (
            g.iterator + g.layout((0, chunk_start, head, 0))
        ).align(16),
        tile_input_layout,
    )
    smem = cutlass.utils.SmemAllocator()
    qk_layout = cute.make_layout((BT, BD), stride=(BD, 1))
    g_layout = cute.make_layout((BT, BD), stride=(BD + 4, 1))
    da_layout = cute.make_layout((BT, BT), stride=(BT + 4, 1))
    b_all_layout = cute.make_layout(
        (4, BD, BC),
        stride=(BC * (BD + 4), 1, BD + 4),
    )
    b_tile_layout = cute.make_layout(
        (BD, BC), stride=(1, BD + 4)
    )
    s_q = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    s_k = smem.allocate_tensor(cutlass.BFloat16, qk_layout, 128)
    s_g = smem.allocate_tensor(cutlass.Float32, g_layout, 128)
    s_beta = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((BT,), stride=(1,)),
        128,
    )
    s_daq = smem.allocate_tensor(cutlass.Float32, da_layout, 128)
    s_dak = smem.allocate_tensor(cutlass.Float32, da_layout, 128)
    s_bq_all = smem.allocate_tensor(
        cutlass.Float32, b_all_layout, 128
    )
    s_bk_all = smem.allocate_tensor(
        cutlass.Float32, b_all_layout, 128
    )
    s_load_barrier = smem.allocate_tensor(
        cutlass.Int64,
        cute.make_layout((1,), stride=(1,)),
        8,
    )

    if tidx < BT:
        value = cutlass.Float32(0.0)
        if tidx < tile_len:
            value = cutlass.Float32(beta[(0, chunk_start + tidx, head)])
        s_beta[tidx] = value

    if tile_len == BT:
        bulk_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyBulkG2SOp(),
            cutlass.Float32,
            num_bits_per_copy=BT * 32,
        )
        if tidx == 0:
            cute.arch.mbarrier_init(s_load_barrier.iterator, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(
                s_load_barrier.iterator,
                BT * BT * 4 * 2,
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if tidx < BT:
            row = tidx
            token = chunk_start + row
            gaq = cute.make_tensor(
                (
                    d_aq.iterator
                    + d_aq.layout((0, token, head, 0))
                ).align(16),
                cute.make_layout((BT,), stride=(1,)),
            )
            gak = cute.make_tensor(
                (
                    d_ak.iterator
                    + d_ak.layout((0, token, head, 0))
                ).align(16),
                cute.make_layout((BT,), stride=(1,)),
            )
            saq = cute.make_tensor(
                (
                    s_daq.iterator + s_daq.layout((row, 0))
                ).align(16),
                cute.make_layout((BT,), stride=(1,)),
            )
            sak = cute.make_tensor(
                (
                    s_dak.iterator + s_dak.layout((row, 0))
                ).align(16),
                cute.make_layout((BT,), stride=(1,)),
            )
            cute.copy(
                bulk_copy,
                gaq,
                saq,
                mbar_ptr=s_load_barrier.iterator,
            )
            cute.copy(
                bulk_copy,
                gak,
                sak,
                mbar_ptr=s_load_barrier.iterator,
            )
        _load_qkg(
            s_q,
            s_k,
            s_g,
            q_tile,
            k_tile,
            g_tile,
            cutlass.Int32(0),
            tile_len,
            cutlass.Int32(0),
            cutlass.Int32(0),
        )
        cute.arch.mbarrier_wait(s_load_barrier.iterator, 0)
    else:
        f32_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        for item in cutlass.range_constexpr(8):
            group = tidx + item * THREADS
            row = group // 16
            col = (group % 16) * 4
            aq = cute.make_rmem_tensor(
                cute.make_layout((4,), stride=(1,)),
                cutlass.Float32,
            )
            ak = cute.make_rmem_tensor_like(aq, cutlass.Float32)
            aq.fill(0.0)
            ak.fill(0.0)
            if row < tile_len:
                token = chunk_start + row
                gaq = cute.make_tensor(
                    (
                        d_aq.iterator
                        + d_aq.layout((0, token, head, col))
                    ).align(16),
                    cute.make_layout((4,), stride=(1,)),
                )
                gak = cute.make_tensor(
                    (
                        d_ak.iterator
                        + d_ak.layout((0, token, head, col))
                    ).align(16),
                    cute.make_layout((4,), stride=(1,)),
                )
                cute.copy(f32_copy, gaq, aq)
                cute.copy(f32_copy, gak, ak)
            for value_idx in cutlass.range_constexpr(4):
                if col + value_idx <= row:
                    aq[value_idx] = aq[value_idx]
                    ak[value_idx] = ak[value_idx]
                else:
                    aq[value_idx] = 0.0
                    ak[value_idx] = 0.0
            saq = cute.make_tensor(
                (
                    s_daq.iterator
                    + s_daq.layout((row, col))
                ).align(16),
                cute.make_layout((4,), stride=(1,)),
            )
            sak = cute.make_tensor(
                (
                    s_dak.iterator
                    + s_dak.layout((row, col))
                ).align(16),
                cute.make_layout((4,), stride=(1,)),
            )
            cute.copy(f32_copy, aq, saq)
            cute.copy(f32_copy, ak, sak)
        cute.arch.sync_threads()
        _load_qkg(
            s_q,
            s_k,
            s_g,
            q_tile,
            k_tile,
            g_tile,
            cutlass.Int32(0),
            tile_len,
            cutlass.Int32(0),
            cutlass.Int32(0),
        )

    s_bq = cute.make_tensor(
        s_bq_all.iterator + s_bq_all.layout((warp, 0, 0)),
        b_tile_layout,
    )
    s_bk = cute.make_tensor(
        s_bk_all.iterator + s_bk_all.layout((warp, 0, 0)),
        b_tile_layout,
    )

    sub_len = tile_len - warp * BC
    if sub_len > BC:
        sub_len = cutlass.Int32(BC)
    db0 = cutlass.Float32(0.0)
    db1 = cutlass.Float32(0.0)

    for feature_tile in cutlass.range(4, unroll=1):
        feature_start = feature_tile * BD
        if feature_tile > 0:
            _load_qkg(
                s_q,
                s_k,
                s_g,
                q_tile,
                k_tile,
                g_tile,
                cutlass.Int32(0),
                tile_len,
                cutlass.Int32(0),
                feature_start,
        )

        if sub_len > 0:
            _load_gradient_block(
                s_bk,
                dq,
                chunk_start,
                warp,
                head,
                feature_start,
                sub_len,
                lane,
            )
            acc_dq = cute.make_rmem_tensor(
                cute.make_layout((16,), stride=(1,)),
                cutlass.Float32,
            )
            acc_dkl = cute.make_rmem_tensor_like(
                acc_dq, cutlass.Float32
            )
            acc_dq.fill(0.0)
            acc_dkl.fill(0.0)

            for source in cutlass.range(4, unroll=1):
                if source < warp:
                    _fill_lower_b_f32(
                        s_k,
                        s_g,
                        s_bq,
                        source,
                        warp,
                        cutlass.Int32(0),
                        tile_len,
                        lane,
                    )
                    _tf32_gemm_dual(
                        s_daq,
                        s_dak,
                        s_bq,
                        acc_dq,
                        acc_dkl,
                        warp,
                        source,
                        False,
                        False,
                        cutlass.Int32(BC),
                        lane,
                    )

            mid = cutlass.Int32(BC // 2)
            if mid >= sub_len:
                mid = sub_len - 1
            for nt in cutlass.range_constexpr(4):
                base = nt * 4
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                anchor0 = cutlass.Float32(s_g[(warp * BC, col0)])
                anchor1 = cutlass.Float32(s_g[(warp * BC, col1)])
                mid_anchor0 = cutlass.Float32(
                    s_g[(warp * BC + mid, col0)]
                )
                mid_anchor1 = cutlass.Float32(
                    s_g[(warp * BC + mid, col1)]
                )
                scale0 = cute.math.exp2(
                    mid_anchor0 - anchor0,
                    fastmath=True,
                )
                scale1 = cute.math.exp2(
                    mid_anchor1 - anchor1,
                    fastmath=True,
                )
                scales = (
                    scale0,
                    scale1,
                    scale0,
                    scale1,
                )
                for value_idx in cutlass.range_constexpr(4):
                    acc_dq[base + value_idx] *= scales[value_idx]
                    acc_dkl[base + value_idx] *= scales[value_idx]

            _fill_lower_b_f32(
                s_k,
                s_g,
                s_bq,
                warp,
                warp,
                mid,
                tile_len,
                lane,
            )
            _tf32_gemm_dual(
                s_daq,
                s_dak,
                s_bq,
                acc_dq,
                acc_dkl,
                warp,
                warp,
                False,
                True,
                sub_len,
                lane,
            )
            for nt in cutlass.range_constexpr(4):
                base = nt * 4
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                anchor0 = cutlass.Float32(
                    s_g[(warp * BC + mid, col0)]
                )
                anchor1 = cutlass.Float32(
                    s_g[(warp * BC + mid, col1)]
                )
                scales = (
                    cute.math.exp2(
                        cutlass.Float32(
                            s_g[(warp * BC + gid, col0)]
                        )
                        - anchor0,
                        fastmath=True,
                    ),
                    cute.math.exp2(
                        cutlass.Float32(
                            s_g[(warp * BC + gid, col1)]
                        )
                        - anchor1,
                        fastmath=True,
                    ),
                    cute.math.exp2(
                        cutlass.Float32(
                            s_g[(warp * BC + gid + 8, col0)]
                        )
                        - anchor0,
                        fastmath=True,
                    ),
                    cute.math.exp2(
                        cutlass.Float32(
                            s_g[(warp * BC + gid + 8, col1)]
                        )
                        - anchor1,
                        fastmath=True,
                    ),
                )
                for value_idx in cutlass.range_constexpr(4):
                    acc_dq[base + value_idx] *= scales[value_idx]
                    acc_dkl[base + value_idx] *= scales[value_idx]
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()
            for nt in cutlass.range_constexpr(4):
                base = nt * 4
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                if gid < sub_len:
                    token0 = chunk_start + warp * BC + gid
                    value0 = (
                        cutlass.Float32(s_bk[(col0, gid)])
                        + cutlass.Float32(acc_dq[base])
                    )
                    value1 = (
                        cutlass.Float32(s_bk[(col1, gid)])
                        + cutlass.Float32(acc_dq[base + 1])
                    )
                    destination0 = (
                        dq_out.iterator
                        + dq_out.layout(
                            (0, token0, head, feature_start + col0)
                        )
                    )
                    _store_bf16x2(destination0, value0, value1)
                if gid + 8 < sub_len:
                    token1 = chunk_start + warp * BC + gid + 8
                    value2 = (
                        cutlass.Float32(s_bk[(col0, gid + 8)])
                        + cutlass.Float32(acc_dq[base + 2])
                    )
                    value3 = (
                        cutlass.Float32(s_bk[(col1, gid + 8)])
                        + cutlass.Float32(acc_dq[base + 3])
                    )
                    destination1 = (
                        dq_out.iterator
                        + dq_out.layout(
                            (0, token1, head, feature_start + col0)
                        )
                    )
                    _store_bf16x2(destination1, value2, value3)

            acc_dku = cute.make_rmem_tensor_like(
                acc_dq, cutlass.Float32
            )
            acc_dku.fill(0.0)
            last = sub_len - 1
            for source in cutlass.range(4, unroll=1):
                if source > warp and source * BC < tile_len:
                    _fill_upper_b_f32(
                        s_q,
                        s_k,
                        s_g,
                        s_beta,
                        s_bq,
                        s_bk,
                        source,
                        warp,
                        last,
                        tile_len,
                        lane,
                        beta.element_type == cutlass.Float32,
                    )
                    _tf32_gemm_two_b(
                        s_daq,
                        s_dak,
                        s_bq,
                        s_bk,
                        acc_dku,
                        source,
                        warp,
                        True,
                        False,
                        cutlass.Int32(BC),
                        lane,
                    )

            for nt in cutlass.range_constexpr(4):
                base = nt * 4
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                anchor0 = cutlass.Float32(
                    s_g[(warp * BC + last, col0)]
                )
                anchor1 = cutlass.Float32(
                    s_g[(warp * BC + last, col1)]
                )
                mid_anchor0 = cutlass.Float32(
                    s_g[(warp * BC + mid, col0)]
                )
                mid_anchor1 = cutlass.Float32(
                    s_g[(warp * BC + mid, col1)]
                )
                scale0 = cute.math.exp2(
                    anchor0 - mid_anchor0,
                    fastmath=True,
                )
                scale1 = cute.math.exp2(
                    anchor1 - mid_anchor1,
                    fastmath=True,
                )
                scales = (
                    scale0,
                    scale1,
                    scale0,
                    scale1,
                )
                for value_idx in cutlass.range_constexpr(4):
                    acc_dku[base + value_idx] *= scales[value_idx]

            _fill_upper_b_f32(
                s_q,
                s_k,
                s_g,
                s_beta,
                s_bq,
                s_bk,
                warp,
                warp,
                mid,
                tile_len,
                lane,
                beta.element_type == cutlass.Float32,
            )
            _tf32_gemm_two_b(
                s_daq,
                s_dak,
                s_bq,
                s_bk,
                acc_dku,
                warp,
                warp,
                True,
                True,
                sub_len,
                lane,
            )

            _load_gradient_block(
                s_bq,
                dk,
                chunk_start,
                warp,
                head,
                feature_start,
                sub_len,
                lane,
            )
            _load_gradient_block(
                s_bk,
                dg,
                chunk_start,
                warp,
                head,
                feature_start,
                sub_len,
                lane,
            )
            for nt in cutlass.range_constexpr(4):
                base = nt * 4
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                anchor0 = cutlass.Float32(
                    s_g[(warp * BC + mid, col0)]
                )
                anchor1 = cutlass.Float32(
                    s_g[(warp * BC + mid, col1)]
                )
                upper_scales = (
                    cute.math.exp2(
                        anchor0
                        - cutlass.Float32(
                            s_g[(warp * BC + gid, col0)]
                        ),
                        fastmath=True,
                    ),
                    cute.math.exp2(
                        anchor1
                        - cutlass.Float32(
                            s_g[(warp * BC + gid, col1)]
                        ),
                        fastmath=True,
                    ),
                    cute.math.exp2(
                        anchor0
                        - cutlass.Float32(
                            s_g[(warp * BC + gid + 8, col0)]
                        ),
                        fastmath=True,
                    ),
                    cute.math.exp2(
                        anchor1
                        - cutlass.Float32(
                            s_g[(warp * BC + gid + 8, col1)]
                        ),
                        fastmath=True,
                    ),
                )
                for value_idx in cutlass.range_constexpr(4):
                    acc_dku[base + value_idx] *= upper_scales[value_idx]
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()

            for nt in cutlass.range_constexpr(4):
                base = nt * 4
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                for row_group in cutlass.range_constexpr(2):
                    row = gid + row_group * 8
                    value_idx = row_group * 2
                    beta_value = cutlass.Float32(
                        s_beta[warp * BC + row]
                    )
                    dqi0 = cutlass.Float32(
                        acc_dq[base + value_idx]
                    )
                    dqi1 = cutlass.Float32(
                        acc_dq[base + value_idx + 1]
                    )
                    dkl0 = cutlass.Float32(
                        acc_dkl[base + value_idx]
                    )
                    dkl1 = cutlass.Float32(
                        acc_dkl[base + value_idx + 1]
                    )
                    dku0 = cutlass.Float32(
                        acc_dku[base + value_idx]
                    )
                    dku1 = cutlass.Float32(
                        acc_dku[base + value_idx + 1]
                    )
                    dkl_beta0 = dkl0 * beta_value
                    dkl_beta1 = dkl1 * beta_value
                    q_value0 = cutlass.Float32(
                        s_q[(warp * BC + row, col0)]
                    )
                    q_value1 = cutlass.Float32(
                        s_q[(warp * BC + row, col1)]
                    )
                    k_value0 = cutlass.Float32(
                        s_k[(warp * BC + row, col0)]
                    )
                    k_value1 = cutlass.Float32(
                        s_k[(warp * BC + row, col1)]
                    )
                    if row < sub_len:
                        token = chunk_start + warp * BC + row
                        dk_value0 = (
                            cutlass.Float32(s_bq[(col0, row)])
                            + dkl_beta0
                            + dku0
                        )
                        dk_value1 = (
                            cutlass.Float32(s_bq[(col1, row)])
                            + dkl_beta1
                            + dku1
                        )
                        dg_value0 = (
                            cutlass.Float32(s_bk[(col0, row)])
                            + q_value0 * dqi0
                            + (dkl_beta0 - dku0) * k_value0
                        )
                        dg_value1 = (
                            cutlass.Float32(s_bk[(col1, row)])
                            + q_value1 * dqi1
                            + (dkl_beta1 - dku1) * k_value1
                        )
                        dk_destination = (
                            dk_out.iterator
                            + dk_out.layout(
                                (0, token, head, feature_start + col0)
                            )
                        )
                        dg_destination = (
                            dg_out.iterator
                            + dg_out.layout(
                                (0, token, head, feature_start + col0)
                            )
                        )
                        _store_bf16x2(
                            dk_destination, dk_value0, dk_value1
                        )
                        _store_f32x2(
                            dg_destination, dg_value0, dg_value1
                        )
                        contribution = (
                            dkl0 * k_value0 + dkl1 * k_value1
                        )
                        if row_group == 0:
                            db0 += contribution
                        else:
                            db1 += contribution
        cute.arch.sync_threads()

    if sub_len > 0:
        db0 += cute.arch.shuffle_sync_bfly(
            db0, offset=1, mask=-1, mask_and_clamp=31
        )
        db0 += cute.arch.shuffle_sync_bfly(
            db0, offset=2, mask=-1, mask_and_clamp=31
        )
        db1 += cute.arch.shuffle_sync_bfly(
            db1, offset=1, mask=-1, mask_and_clamp=31
        )
        db1 += cute.arch.shuffle_sync_bfly(
            db1, offset=2, mask=-1, mask_and_clamp=31
        )
        if lane4 == 0:
            if gid < sub_len:
                token0 = chunk_start + warp * BC + gid
                value0 = (
                    db0 + cutlass.Float32(db[(0, token0, head)])
                )
                ptr0 = (
                    db_out.iterator
                    + db_out.layout((0, token0, head))
                )
                _store_f32(ptr0, value0)
            if gid + 8 < sub_len:
                token1 = chunk_start + warp * BC + gid + 8
                value1 = (
                    db1 + cutlass.Float32(db[(0, token1, head)])
                )
                ptr1 = (
                    db_out.iterator
                    + db_out.layout((0, token1, head))
                )
                _store_f32(ptr1, value1)


@cute.kernel
def _kernel_tf32(
    q: cute.Tensor,
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    d_aq: cute.Tensor,
    d_ak: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    db: cute.Tensor,
    dg: cute.Tensor,
    cu_seqlens: cute.Tensor,
    chunk_indices: cute.Tensor,
    dq_out: cute.Tensor,
    dk_out: cute.Tensor,
    db_out: cute.Tensor,
    dg_out: cute.Tensor,
    chunks: cutlass.Int32,
    heads: cutlass.Constexpr[int],
):
    block, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    total_tiles = chunks * heads
    for tile_id in cutlass.range(block, total_tiles, grid_x, unroll=1):
        chunk_idx = tile_id // heads
        head = tile_id % heads
        _process_tf32_tile(
            q,
            k,
            g,
            beta,
            d_aq,
            d_ak,
            dq,
            dk,
            db,
            dg,
            cu_seqlens,
            chunk_indices,
            dq_out,
            dk_out,
            db_out,
            dg_out,
            chunk_idx,
            head,
        )


def _make_jit():
    @cute.jit
    def launch(
        q: cute.Tensor,
        k: cute.Tensor,
        g: cute.Tensor,
        beta: cute.Tensor,
        d_aq: cute.Tensor,
        d_ak: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        db: cute.Tensor,
        dg: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        dq_out: cute.Tensor,
        dk_out: cute.Tensor,
        db_out: cute.Tensor,
        dg_out: cute.Tensor,
        chunks: cutlass.Int32,
        heads: cutlass.Constexpr[int],
        persistent_ctas: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        _kernel_tf32(
            q,
            k,
            g,
            beta,
            d_aq,
            d_ak,
            dq,
            dk,
            db,
            dg,
            cu_seqlens,
            chunk_indices,
            dq_out,
            dk_out,
            db_out,
            dg_out,
            chunks,
            heads,
        ).launch(
            grid=(
                cutlass.min(chunks * heads, persistent_ctas),
                1,
                1,
            ),
            block=(THREADS, 1, 1),
            smem=72 * 1024,
            stream=stream,
        )

    return launch


_launcher = None


def _tensor(value: torch.Tensor):
    return from_dlpack(value.detach(), assumed_align=16)


def _launch(tensors):
    global _launcher
    q = tensors[0]
    chunks = tensors[11].shape[0]
    heads = q.shape[2]
    major, minor = torch.cuda.get_device_capability(q.device)
    persistent_ctas = (
        torch.cuda.get_device_properties(q.device).multi_processor_count * 3
    )
    key = (
        q.device.index,
        major,
        minor,
        q.shape[1],
        heads,
        chunks,
        tensors[10].shape[0],
        int(tensors[3].dtype == torch.float32),
        persistent_ctas,
    )
    stream = cuda.CUstream(
        torch.cuda.current_stream(q.device).cuda_stream
    )
    compiled = _cache.get(key)
    if compiled is None:
        if _launcher is None:
            _launcher = _make_jit()
        cute_tensors = [_tensor(value) for value in tensors]
        compiled = cute.compile(
            _launcher,
            *cute_tensors,
            cutlass.Int32(chunks),
            heads=heads,
            persistent_ctas=persistent_ctas,
            stream=stream,
        )
        _cache[key] = compiled
    cute_tensors = [_tensor(value) for value in tensors]
    compiled(
        *cute_tensors,
        cutlass.Int32(chunks),
        heads,
        persistent_ctas,
        stream,
    )


def kda_bwd_intra_mma(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    db_out: torch.Tensor,
    dg_out: torch.Tensor,
    chunk_size: int,
    tile_counter: torch.Tensor | None = None,
):
    """Run the portable mma.sync CuTeDSL kernel."""
    del tile_counter
    if chunk_size != BT:
        raise ValueError("CuTe DSL KDA backward requires chunk_size=64")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("q and k must have shape [1, T, H, 128]")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("q and k must be bfloat16")
    if beta.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("beta must be bfloat16 or float32")
    if dq_out.dtype != torch.bfloat16 or dk_out.dtype != torch.bfloat16:
        raise TypeError("dq_out and dk_out must be bfloat16")
    float_tensors = (
        g,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        db_out,
        dg_out,
    )
    if any(value.dtype != torch.float32 for value in float_tensors):
        raise TypeError("g, dA, input gradients, db_out and dg_out must be fp32")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be int32")
    if chunk_indices.dtype != torch.int32:
        raise TypeError("chunk_indices must be int32")
    if not torch.cuda.is_available():
        raise RuntimeError("CuTeDSL KDA backward requires a CUDA GPU")
    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) not in ((9, 0), (10, 0), (10, 3)):
        raise RuntimeError(
            "mma.sync CuTeDSL KDA backward requires SM90, SM100, or SM103; "
            f"got SM{major}{minor}"
        )

    chunk_indices_2d = chunk_indices.reshape(-1, 2)
    tensors = (
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices_2d,
        dq_out,
        dk_out,
        db_out,
        dg_out,
    )
    if any(not value.is_cuda for value in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if any(value.device != q.device for value in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if any(not value.is_contiguous() for value in tensors):
        raise ValueError("all tensors must be contiguous")

    with torch.cuda.device(q.device):
        _launch(tensors)
    return dq_out, dk_out, db_out, dg_out


__all__ = ["kda_bwd_intra_mma"]
