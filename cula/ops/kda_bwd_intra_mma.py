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

"""Portable native-CuTe implementation of KDA intra-chunk backward.

The tensor-core path is described by a ``TiledMma``, its A/B/C thread-value
fragments, and ``cute.gemm``. CuTe lowers that description to warp-level
``mma.sync.m16n8k8`` instructions, without WGMMA or tcgen05, so one kernel
runs on SM90 and SM100/SM103. One 128-thread CTA owns one ``(chunk, head)``
tile; its four symmetric warps own the four 16-token output blocks.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import vector as _vector
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import T as _T

from ._cute_tf32 import make_tf32_tiled_mma

BT = 64
BC = 16
BD = 32
THREADS = 128
MMA_ATOM_N = 8
MMA_ATOM_K = 8
MMA_N_ATOMS = BD // MMA_ATOM_N
MMA_K_ATOMS = BC // MMA_ATOM_K
_cache: dict[tuple[int, ...], object] = {}


@cutlass.dsl_user_op
def _as_tf32_register(value, *, loc=None, ip=None):
    """Adapt F32 to CuTeDSL 4.4's integer-backed TF32 fragment storage.

    KDA's established numerical contract uses the CUTLASS
    ``round_toward_zero`` TF32 encoding. CuTeDSL 4.4 exposes the fragment as
    I32, so clear the 13 unused F32 mantissa bits with ordinary DSL integer
    arithmetic. The MMA and its thread-value mapping remain native CuTe.
    """

    bits = _llvm.bitcast(
        _T.i32(),
        Float32(value).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return Int32(bits) & cutlass.Int32(-0x2000)


@cutlass.dsl_user_op
def _store_bf16x2(pointer, value0, value1, *, loc=None, ip=None):
    """Round, pack, and store two BF16 values with the required CG policy.

    CuTeDSL 4.4.2 extends the live range of its BF16 vector conversion and
    raises this kernel from 130 to 136 registers. Keep this one leaf adapter
    until that lowering is fixed; the other memory helpers use native APIs.
    """

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
    """Issue one native two-element CG store."""

    values = cute.make_rmem_tensor((2,), Float32)
    values[0] = value0
    values[1] = value1
    cute.arch.store(pointer, values.load(), cop="cg")


@cutlass.dsl_user_op
def _store_f32(pointer, value, *, loc=None, ip=None):
    """Issue one native scalar CG store."""

    cute.arch.store(pointer, Float32(value), cop="cg")


@cutlass.dsl_user_op
def _load_shared_f32x4(pointer, *, loc=None, ip=None):
    """Issue one native 128-bit shared-memory load."""

    vector_type = _ir.VectorType.get([4], Float32.mlir_type, loc=loc)
    values = cute.TensorSSA(
        cute.arch.load(
            pointer, vector_type, ss="cta", loc=loc, ip=ip
        ),
        (4,),
        Float32,
    )
    return tuple(Float32(values[idx]) for idx in range(4))


@cutlass.dsl_user_op
def _load_shared_bf16x4(pointer, *, loc=None, ip=None):
    """Issue one native 64-bit shared load and widen BF16 values to F32.

    Loading ``vector<4xbf16>`` directly crashes CuTeDSL 4.4.2, so load the
    same 64 payload bits as two I32 values and use a typed vector bitcast.
    """

    packed_type = _ir.VectorType.get([2], Int32.mlir_type, loc=loc)
    bf16_type = _ir.VectorType.get(
        [4], cutlass.BFloat16.mlir_type, loc=loc
    )
    packed = cute.arch.load(
        pointer, packed_type, ss="cta", loc=loc, ip=ip
    )
    unpacked = _vector.bitcast(
        bf16_type, packed, loc=loc, ip=ip
    )
    values = cute.TensorSSA(
        unpacked, (4,), cutlass.BFloat16
    ).to(Float32)
    return tuple(Float32(values[idx]) for idx in range(4))


@cute.jit
def _tf32_gemm_dual_tv(
    tiled_mma: cute.TiledMma,
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
    """Accumulate two GEMMs that share B using CuTe A/B/C TV fragments."""

    row_in_atom = lane // 4
    reduction_lane = (lane % 4) * 2
    a_shape = tiled_mma.partition_shape_A((BC, BC))
    thr_mma = tiled_mma.get_slice(lane)
    r_aq = tiled_mma.make_fragment_A(a_shape)
    r_ak = tiled_mma.make_fragment_A(a_shape)
    r_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_b))

    # local_tile selects register slots from CuTe's m16n8k8 TV layout. Only
    # the shared-memory source coordinates stay explicit, because transpose
    # and causal masking change them at runtime.
    for k_atom in cutlass.range_constexpr(MMA_K_ATOMS):
        a_q_atom = cute.local_tile(
            r_aq, (4, 1, 1), (0, 0, k_atom)
        )
        a_k_atom = cute.local_tile(
            r_ak, (4, 1, 1), (0, 0, k_atom)
        )
        reduction_base = k_atom * MMA_ATOM_K + reduction_lane
        for reduction_pair in cutlass.range_constexpr(2):
            reduction = reduction_base + reduction_pair
            for row_group in cutlass.range_constexpr(2):
                row = row_in_atom + row_group * (BC // 2)
                value_idx = reduction_pair * 2 + row_group
                if cutlass.const_expr(transpose):
                    aq = s_daq[
                        (
                            row_block * BC + reduction,
                            col_block * BC + row,
                        )
                    ]
                    ak = s_dak[
                        (
                            row_block * BC + reduction,
                            col_block * BC + row,
                        )
                    ]
                else:
                    aq = s_daq[
                        (
                            row_block * BC + row,
                            col_block * BC + reduction,
                        )
                    ]
                    ak = s_dak[
                        (
                            row_block * BC + row,
                            col_block * BC + reduction,
                        )
                    ]
                aq_bits = _as_tf32_register(aq)
                ak_bits = _as_tf32_register(ak)
                if cutlass.const_expr(causal):
                    if cutlass.const_expr(transpose):
                        valid = row <= reduction
                    else:
                        valid = reduction <= row
                    valid = (
                        valid
                        and row < sub_len
                        and reduction < sub_len
                    )
                    if not valid:
                        aq_bits = cutlass.Int32(0)
                        ak_bits = cutlass.Int32(0)
                a_q_atom[value_idx] = aq_bits
                a_k_atom[value_idx] = ak_bits

        # B has two lane values for each (N atom, K atom) pair.
        for n_atom in cutlass.range_constexpr(MMA_N_ATOMS):
            b_atom = cute.local_tile(
                r_b, (2, 1, 1), (0, n_atom, k_atom)
            )
            feature_row = n_atom * MMA_ATOM_N + row_in_atom
            for reduction_pair in cutlass.range_constexpr(2):
                b = s_b[
                    (feature_row, reduction_base + reduction_pair)
                ]
                b_atom[reduction_pair] = _as_tf32_register(b)
        cute.gemm(
            tiled_mma,
            acc_q,
            r_aq[(None, None, k_atom)],
            r_b[(None, None, k_atom)],
            acc_q,
        )
        cute.gemm(
            tiled_mma,
            acc_k,
            r_ak[(None, None, k_atom)],
            r_b[(None, None, k_atom)],
            acc_k,
        )


@cute.jit
def _tf32_gemm_single_tv(
    tiled_mma: cute.TiledMma,
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
    """Accumulate a 16x32 GEMM through native CuTe TV fragments."""

    row_in_atom = lane // 4
    reduction_lane = (lane % 4) * 2
    a_shape = tiled_mma.partition_shape_A((BC, BC))
    thr_mma = tiled_mma.get_slice(lane)
    r_a = tiled_mma.make_fragment_A(a_shape)
    r_b = tiled_mma.make_fragment_B(thr_mma.partition_B(s_b))

    for k_atom in cutlass.range_constexpr(MMA_K_ATOMS):
        a_atom = cute.local_tile(
            r_a, (4, 1, 1), (0, 0, k_atom)
        )
        reduction_base = k_atom * MMA_ATOM_K + reduction_lane
        for reduction_pair in cutlass.range_constexpr(2):
            reduction = reduction_base + reduction_pair
            for row_group in cutlass.range_constexpr(2):
                row = row_in_atom + row_group * (BC // 2)
                value_idx = reduction_pair * 2 + row_group
                if cutlass.const_expr(transpose):
                    value = s_a[
                        (
                            row_block * BC + reduction,
                            col_block * BC + row,
                        )
                    ]
                else:
                    value = s_a[
                        (
                            row_block * BC + row,
                            col_block * BC + reduction,
                        )
                    ]
                value_bits = _as_tf32_register(value)
                if cutlass.const_expr(causal):
                    if cutlass.const_expr(transpose):
                        valid = row <= reduction
                    else:
                        valid = reduction <= row
                    valid = (
                        valid
                        and row < sub_len
                        and reduction < sub_len
                    )
                    if not valid:
                        value_bits = cutlass.Int32(0)
                a_atom[value_idx] = value_bits

        for n_atom in cutlass.range_constexpr(MMA_N_ATOMS):
            b_atom = cute.local_tile(
                r_b, (2, 1, 1), (0, n_atom, k_atom)
            )
            feature_row = n_atom * MMA_ATOM_N + row_in_atom
            for reduction_pair in cutlass.range_constexpr(2):
                b = s_b[
                    (feature_row, reduction_base + reduction_pair)
                ]
                b_atom[reduction_pair] = _as_tf32_register(b)
        cute.gemm(
            tiled_mma,
            acc,
            r_a[(None, None, k_atom)],
            r_b[(None, None, k_atom)],
            acc,
        )


@cute.jit
def _tf32_gemm_two_b_tv(
    tiled_mma: cute.TiledMma,
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
    """Accumulate A_q @ B_q and A_k @ B_k into one C TV fragment."""

    _tf32_gemm_single_tv(
        tiled_mma,
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
    _tf32_gemm_single_tv(
        tiled_mma,
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


@cute.jit
def _process_tf32_tile(
    tiled_mma: cute.TiledMma,
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
    """Process one (64-token chunk, head) tile with four symmetric warps.

    Warp ``w`` owns output rows ``[16*w, 16*(w+1))``. Every warp performs
    lower/diagonal/upper staging and its own epilogue; there is no dedicated
    producer/consumer warp specialization. Source-block counts differ by
    warp, but the tensor-core instruction count stays balanced.
    """

    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    gid = lane // 4
    lane4 = lane % 4
    thr_mma = tiled_mma.get_slice(lane)
    acc_shape = thr_mma.partition_shape_C((BC, BD))

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

    # Stage the CTA-wide 64x64 dA matrices. Full chunks use bulk async copy;
    # tail chunks use predicated vector copies and explicit causal masking.
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

    # Each warp gets private 32x16 B staging buffers for its output block.
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

    # Stream the 128 feature columns in four 32-column tiles.
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
            # The MMA and scalar epilogue both address native C TV fragments;
            # local_tile selects one 8-column atom without flattening layout.
            acc_dq_tv = tiled_mma.make_fragment_C(acc_shape)
            acc_dkl_tv = tiled_mma.make_fragment_C(acc_shape)
            acc_dq_tv.fill(0.0)
            acc_dkl_tv.fill(0.0)

            # Earlier source blocks plus the causal diagonal produce dq and
            # the lower-triangular component of dk.
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
                    _tf32_gemm_dual_tv(
                        tiled_mma,
                        s_daq,
                        s_dak,
                        s_bq,
                        acc_dq_tv,
                        acc_dkl_tv,
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
                acc_dq_atom = cute.local_tile(
                    acc_dq_tv, (4, 1, 1), (0, 0, nt)
                )
                acc_dkl_atom = cute.local_tile(
                    acc_dkl_tv, (4, 1, 1), (0, 0, nt)
                )
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
                    acc_dq_atom[value_idx] *= scales[value_idx]
                    acc_dkl_atom[value_idx] *= scales[value_idx]

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
            _tf32_gemm_dual_tv(
                tiled_mma,
                s_daq,
                s_dak,
                s_bq,
                acc_dq_tv,
                acc_dkl_tv,
                warp,
                warp,
                False,
                True,
                sub_len,
                lane,
            )
            for nt in cutlass.range_constexpr(4):
                acc_dq_atom = cute.local_tile(
                    acc_dq_tv, (4, 1, 1), (0, 0, nt)
                )
                acc_dkl_atom = cute.local_tile(
                    acc_dkl_tv, (4, 1, 1), (0, 0, nt)
                )
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
                    acc_dq_atom[value_idx] *= scales[value_idx]
                    acc_dkl_atom[value_idx] *= scales[value_idx]
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()
            for nt in cutlass.range_constexpr(4):
                acc_dq_atom = cute.local_tile(
                    acc_dq_tv, (4, 1, 1), (0, 0, nt)
                )
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                if gid < sub_len:
                    token0 = chunk_start + warp * BC + gid
                    value0 = (
                        cutlass.Float32(s_bk[(col0, gid)])
                        + cutlass.Float32(acc_dq_atom[0])
                    )
                    value1 = (
                        cutlass.Float32(s_bk[(col1, gid)])
                        + cutlass.Float32(acc_dq_atom[1])
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
                        + cutlass.Float32(acc_dq_atom[2])
                    )
                    value3 = (
                        cutlass.Float32(s_bk[(col1, gid + 8)])
                        + cutlass.Float32(acc_dq_atom[3])
                    )
                    destination1 = (
                        dq_out.iterator
                        + dq_out.layout(
                            (0, token1, head, feature_start + col0)
                        )
                    )
                    _store_bf16x2(destination1, value2, value3)

            # Later source blocks plus the transposed causal diagonal produce
            # the upper-triangular component of dk and dg.
            acc_dku_tv = tiled_mma.make_fragment_C(acc_shape)
            acc_dku_tv.fill(0.0)
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
                    _tf32_gemm_two_b_tv(
                        tiled_mma,
                        s_daq,
                        s_dak,
                        s_bq,
                        s_bk,
                        acc_dku_tv,
                        source,
                        warp,
                        True,
                        False,
                        cutlass.Int32(BC),
                        lane,
                    )

            for nt in cutlass.range_constexpr(4):
                acc_dku_atom = cute.local_tile(
                    acc_dku_tv, (4, 1, 1), (0, 0, nt)
                )
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
                    acc_dku_atom[value_idx] *= scales[value_idx]

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
            _tf32_gemm_two_b_tv(
                tiled_mma,
                s_daq,
                s_dak,
                s_bq,
                s_bk,
                acc_dku_tv,
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
                acc_dku_atom = cute.local_tile(
                    acc_dku_tv, (4, 1, 1), (0, 0, nt)
                )
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
                    acc_dku_atom[value_idx] *= upper_scales[value_idx]
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()

            for nt in cutlass.range_constexpr(4):
                acc_dq_atom = cute.local_tile(
                    acc_dq_tv, (4, 1, 1), (0, 0, nt)
                )
                acc_dkl_atom = cute.local_tile(
                    acc_dkl_tv, (4, 1, 1), (0, 0, nt)
                )
                acc_dku_atom = cute.local_tile(
                    acc_dku_tv, (4, 1, 1), (0, 0, nt)
                )
                col0 = nt * 8 + lane4 * 2
                col1 = col0 + 1
                for row_group in cutlass.range_constexpr(2):
                    row = gid + row_group * 8
                    value_idx = row_group * 2
                    beta_value = cutlass.Float32(
                        s_beta[warp * BC + row]
                    )
                    dqi0 = cutlass.Float32(
                        acc_dq_atom[value_idx]
                    )
                    dqi1 = cutlass.Float32(
                        acc_dq_atom[value_idx + 1]
                    )
                    dkl0 = cutlass.Float32(
                        acc_dkl_atom[value_idx]
                    )
                    dkl1 = cutlass.Float32(
                        acc_dkl_atom[value_idx + 1]
                    )
                    dku0 = cutlass.Float32(
                        acc_dku_atom[value_idx]
                    )
                    dku1 = cutlass.Float32(
                        acc_dku_atom[value_idx + 1]
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
    tiled_mma: cute.TiledMma,
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
            tiled_mma,
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
        tiled_mma = make_tf32_tiled_mma()
        _kernel_tf32(
            tiled_mma,
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
    """Run the portable native-TiledMma CuTeDSL kernel."""
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
            "TiledMma CuTeDSL KDA backward requires SM90, SM100, or SM103; "
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
