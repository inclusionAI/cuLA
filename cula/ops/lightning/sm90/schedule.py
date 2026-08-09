#!/usr/bin/env python3
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

"""Shared SM90a schedule constants and WGMMA register-layout helpers.

The production kernel uses three warp groups in ``LdSt / Math0 / Math1``
roles, 24/240/240 register redistribution, and Q/K/V/O stage counts 3/3/2/3.
The public wrapper is responsible for architecture dispatch and validation.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cula.ops._mlir_compat import llvm
from cutlass.cutlass_dsl import T

TARGET_ARCH = "sm_90a"

THREADS_PER_WARP_GROUP = 128
MATH0_WARP_GROUP_INDEX = 1
MATH_THREAD_COUNT = 256
MATH_SIGNALING_THREADS = 8

THREADS_PER_CTA = 384
REGISTER_TARGETS = (24, 240, 240)
EPILOGUE_THREADS = 32
STORE_WARP = 3

CHUNK_TOKENS = 64
HEAD_DIM = 128
VALUE_DIM = 128
Q_STAGES = 3
K_STAGES = 3
V_STAGES = 2
O_STAGES = 3

QK_SHAPE = (CHUNK_TOKENS, CHUNK_TOKENS)
STATE_SHAPE = (VALUE_DIM, HEAD_DIM)
O_SHAPE = (VALUE_DIM, CHUNK_TOKENS)
QK_TILE_SHAPE = (CHUNK_TOKENS, CHUNK_TOKENS, HEAD_DIM)
STATE_TILE_SHAPE = (VALUE_DIM, HEAD_DIM, CHUNK_TOKENS)
O1_TILE_SHAPE = (VALUE_DIM, CHUNK_TOKENS, HEAD_DIM)
O2_TILE_SHAPE = (VALUE_DIM, CHUNK_TOKENS, CHUNK_TOKENS)
SMEM_ALIGNMENT_BYTES = 1024

QK_R2S_TRANSPOSE = False
QK_R2S_NUM_MATRICES = 4
RS_A_S2R_TRANSPOSE = True
RS_A_S2R_NUM_MATRICES = 4

Q_BARRIER_COUNT = 2 * Q_STAGES
K_BARRIER_COUNT = 2 * K_STAGES
V_BARRIER_COUNT = 2 * V_STAGES
O_BARRIER_COUNT = 2 * O_STAGES
Q_BYTES_PER_STAGE = CHUNK_TOKENS * HEAD_DIM * 2
K_BYTES_PER_STAGE = CHUNK_TOKENS * HEAD_DIM * 2
V_BYTES_PER_STAGE = VALUE_DIM * CHUNK_TOKENS * 2
QK_PUBLICATION_BYTES = CHUNK_TOKENS * CHUNK_TOKENS * 2
O_BYTES_PER_STAGE = VALUE_DIM * CHUNK_TOKENS * 2
TENSOR_PAYLOAD_BYTES = (
    Q_STAGES * Q_BYTES_PER_STAGE
    + K_STAGES * K_BYTES_PER_STAGE
    + V_STAGES * V_BYTES_PER_STAGE
    + QK_PUBLICATION_BYTES
    + O_STAGES * O_BYTES_PER_STAGE
)
# The first 176 barrier bytes are rounded up to the first 1024-byte aligned
# tensor allocation.  Every tensor allocation is already a 1024-byte multiple.
DYNAMIC_SMEM_ESTIMATE_BYTES = SMEM_ALIGNMENT_BYTES + TENSOR_PAYLOAD_BYTES
SM90_OPTIN_SMEM_LIMIT_BYTES = 227328

QK_PUBLISHED_BARRIER_ID = 1
QK_CONSUMED_BARRIER_ID = 2


class LightningSm90PrefillSchedule:
    """Shared register-fragment helpers for the production prefill kernel."""

    def __init__(self):
        self.threads_per_cta = THREADS_PER_CTA
        self.epilogue_threads = EPILOGUE_THREADS
        self.store_warp = STORE_WARP
        self.register_targets = REGISTER_TARGETS
        self.shared_storage = None
        self.dynamic_smem_bytes = None

    @staticmethod
    def convert_c_layout_to_a_layout(c_layout, a_value_layout):
        """Exact CUTLASS Hopper C-fragment to A-fragment reshape."""

        return cute.make_layout(
            (
                a_value_layout,
                c_layout.shape[1],
                (
                    c_layout.shape[2],
                    cute.size(c_layout, mode=[0]) // cute.size(a_value_layout),
                ),
            ),
            stride=(
                c_layout.stride[0],
                c_layout.stride[1],
                (
                    c_layout.stride[2],
                    cute.size(a_value_layout, mode=[2]) * c_layout.stride[0][2],
                ),
            ),
        )

    @staticmethod
    @cute.jit
    def make_a_register_fragment(accumulator, operand_layout_tv, element_type):
        return cute.make_rmem_tensor_like(
            LightningSm90PrefillSchedule.convert_c_layout_to_a_layout(
                accumulator.layout,
                operand_layout_tv.shape[1],
            ),
            element_type,
        )

    @staticmethod
    @cute.jit
    def make_acc_into_op(accumulator, operand_layout_tv, element_type):
        operand = LightningSm90PrefillSchedule.make_a_register_fragment(
            accumulator,
            operand_layout_tv,
            element_type,
        )
        operand_as_accumulator = cute.make_tensor(operand.iterator, accumulator.layout)
        operand_as_accumulator.store(accumulator.load().to(element_type))
        return operand

    @staticmethod
    @cute.jit
    def issue_wgmma_ss(tiled_mma, a, b, accumulator):
        # Match the production GDN SM90 helper: only the operand K mode is
        # structurally relevant here.  The C fragment's outer grouping is a
        # property of the selected tiled MMA and must not be constrained by an
        # compile-time rank assertion.
        for k_block_idx in cutlass.range(cute.size(a, mode=[2]), unroll_full=True):
            tiled_mma.set(
                cute.nvgpu.warpgroup.Field.ACCUMULATE,
                k_block_idx != 0,
            )
            cute.gemm(
                tiled_mma,
                accumulator,
                a[None, None, k_block_idx],
                b[None, None, k_block_idx],
                accumulator,
            )

    @staticmethod
    @cute.jit
    def issue_wgmma_rs_zero(tiled_mma, a, b, accumulator):
        for k_block_idx in cutlass.range(cute.size(a, mode=[2]), unroll_full=True):
            tiled_mma.set(
                cute.nvgpu.warpgroup.Field.ACCUMULATE,
                k_block_idx != 0,
            )
            cute.gemm(
                tiled_mma,
                accumulator,
                a[None, None, k_block_idx],
                b[None, None, k_block_idx],
                accumulator,
            )

    @staticmethod
    @cute.jit
    def issue_wgmma_rs_accumulate(tiled_mma, a, b, accumulator):
        for k_block_idx in cutlass.range(cute.size(a, mode=[2]), unroll_full=True):
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.gemm(
                tiled_mma,
                accumulator,
                a[None, None, k_block_idx],
                b[None, None, k_block_idx],
                accumulator,
            )

    @staticmethod
    @cute.jit
    def _fence_f32_register(reg: cutlass.Float32) -> cutlass.Float32:
        return cutlass.Float32(
            llvm.inline_asm(
                T.f32(),
                [reg.ir_value()],
                "",
                "=f,0",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @staticmethod
    @cute.jit
    def _fence_u32_register(reg: cutlass.Uint32) -> cutlass.Uint32:
        return cutlass.Uint32(
            llvm.inline_asm(
                T.i32(),
                [reg.ir_value()],
                "",
                "=r,0",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @staticmethod
    @cute.jit
    def fence_register_fragment(fragment: cute.Tensor):
        if cutlass.const_expr(fragment.element_type is cutlass.Float32):
            values = cute.recast_tensor(fragment, cutlass.Float32)
            for item in cutlass.range_constexpr(cute.size(values)):
                values[item] = LightningSm90PrefillSchedule._fence_f32_register(values[item])
        else:
            values = cute.recast_tensor(fragment, cutlass.Uint32)
            for item in cutlass.range_constexpr(cute.size(values)):
                values[item] = LightningSm90PrefillSchedule._fence_u32_register(values[item])
