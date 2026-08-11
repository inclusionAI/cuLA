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

"""Production SM90a CuTe DSL kernel for Lightning Attention prefill.

The kernel supports fixed and packed variable-length inputs, optional initial
and final state, GVA head mapping, runtime output scale, and persistent packed
scheduling. It uses a 384-thread ``LdSt / Math0 / Math1`` schedule with
3/3/2/3-stage Q/K/V/O rings and a distributed FP32 recurrent state.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.utils.tensormap_manager import TensorMapManager, TensorMapUpdateMode

from .schedule import (
    DYNAMIC_SMEM_ESTIMATE_BYTES,
    EPILOGUE_THREADS,
    HEAD_DIM,
    K_BARRIER_COUNT,
    K_STAGES,
    MATH0_WARP_GROUP_INDEX,
    MATH_SIGNALING_THREADS,
    MATH_THREAD_COUNT,
    O_BARRIER_COUNT,
    O_SHAPE,
    O_STAGES,
    Q_BARRIER_COUNT,
    Q_STAGES,
    QK_CONSUMED_BARRIER_ID,
    QK_PUBLISHED_BARRIER_ID,
    QK_R2S_NUM_MATRICES,
    QK_R2S_TRANSPOSE,
    QK_SHAPE,
    QK_TILE_SHAPE,
    REGISTER_TARGETS,
    RS_A_S2R_NUM_MATRICES,
    RS_A_S2R_TRANSPOSE,
    SM90_OPTIN_SMEM_LIMIT_BYTES,
    SMEM_ALIGNMENT_BYTES,
    STATE_SHAPE,
    STATE_TILE_SHAPE,
    STORE_WARP,
    THREADS_PER_CTA,
    THREADS_PER_WARP_GROUP,
    V_BARRIER_COUNT,
    V_STAGES,
    VALUE_DIM,
    LightningSm90PrefillSchedule,
)

SUPPORTED_CUTLASS_DSL_SPECIFIER = ">=4.4.2,<4.7,!=4.5.0"
DECAY_LUT_ENTRIES = 65
TENSORMAP_BYTES = 128


@cute.jit
def _swap_first_two_modes(tensor: cute.Tensor) -> cute.Tensor:
    """Return the C-fragment view of a row-major public BHVK state tile."""

    return cute.make_tensor(
        tensor.iterator.align(tensor.iterator.max_alignment),
        cute.make_layout(
            (tensor.layout.shape[1], tensor.layout.shape[0]) + tensor.layout.shape[2:],
            stride=(tensor.layout.stride[1], tensor.layout.stride[0]) + tensor.layout.stride[2:],
        ),
    )


@cute.jit
def _tensormap_replace_global_dim_1(
    tensormap_ptr: cute.Pointer,
    new_extent: cutlass.Int32,
):
    llvm.inline_asm(
        None,
        [tensormap_ptr.toint().ir_value(), new_extent.ir_value()],
        "tensormap.replace.tile.global_dim.global.b1024.b32 [$0], 1, $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class LightningSm90PrefillKernel(LightningSm90PrefillSchedule):
    """Compile-time specialization for fixed or packed prefill semantics."""

    def __init__(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        heads: int | None = None,
        qk_heads: int | None = None,
        value_heads: int | None = None,
        decay_heads: int | None = None,
        needs_initial_state: bool = False,
        needs_final_state: bool = False,
        is_varlen: bool = False,
        num_sequences: int | None = None,
        state_pool_size: int | None = None,
        persistent: bool = False,
        persistent_ctas: int | None = None,
    ):
        super().__init__()
        if heads is not None:
            if qk_heads is not None or value_heads is not None:
                raise ValueError("heads cannot be combined with qk_heads/value_heads")
            qk_heads = heads
            value_heads = heads
        if qk_heads is None or value_heads is None:
            raise ValueError("qk_heads and value_heads are required")
        if decay_heads is None:
            decay_heads = value_heads
        for name, value in (
            ("batch_size", batch_size),
            ("sequence_length", sequence_length),
            ("qk_heads", qk_heads),
            ("value_heads", value_heads),
            ("decay_heads", decay_heads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, found {value!r}")
        if value_heads < qk_heads or value_heads % qk_heads:
            raise ValueError("value_heads must be >= qk_heads with an integral GVA group")
        if decay_heads not in {qk_heads, value_heads}:
            raise ValueError("decay_heads must equal qk_heads or value_heads")
        if not isinstance(needs_initial_state, bool) or not isinstance(
            needs_final_state,
            bool,
        ):
            raise TypeError("state specialization flags must be boolean")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen must be boolean")
        if not isinstance(persistent, bool):
            raise TypeError("persistent must be boolean")
        if is_varlen:
            if batch_size != 1:
                raise ValueError("packed varlen requires physical batch size one")
            for name, value in (
                ("num_sequences", num_sequences),
                ("state_pool_size", state_pool_size),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer for packed varlen")
            if not needs_initial_state or not needs_final_state:
                raise ValueError("packed varlen requires in-place initial/final state support")
            total_work_units = num_sequences * value_heads
            if persistent:
                if isinstance(persistent_ctas, bool) or not isinstance(persistent_ctas, int):
                    raise ValueError("persistent_ctas must be an integer for persistent packed mode")
                if persistent_ctas < 1 or persistent_ctas > total_work_units:
                    raise ValueError("persistent_ctas must be within the packed work-unit count")
            elif persistent_ctas is not None:
                raise ValueError("persistent_ctas is invalid for non-persistent packed mode")
        elif num_sequences is not None or state_pool_size is not None:
            raise ValueError("packed scheduling metadata is invalid for fixed length")
        elif persistent or persistent_ctas is not None:
            raise ValueError("persistent scheduling is valid only for packed varlen")
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.heads = value_heads
        self.qk_heads = qk_heads
        self.value_heads = value_heads
        self.decay_heads = decay_heads
        self.group_size = value_heads // qk_heads
        self.needs_initial_state = needs_initial_state
        self.needs_final_state = needs_final_state
        self.is_varlen = is_varlen
        self.num_sequences = num_sequences
        self.state_pool_size = state_pool_size
        self.persistent = persistent
        self.persistent_ctas = persistent_ctas
        self.threads_per_cta = THREADS_PER_CTA
        self.epilogue_threads = EPILOGUE_THREADS
        self.store_warp = STORE_WARP
        self.register_targets = REGISTER_TARGETS

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        v_in: cute.Tensor,
        decay_s: cute.Tensor,
        initial_state_in: cute.Tensor,
        final_state_in: cute.Tensor,
        cu_seqlens_in: cute.Tensor,
        initial_state_indices_in: cute.Tensor,
        tensormaps_in: cute.Tensor,
        scale: cutlass.Float32,
        sequence_length: cutlass.Int32,
        output_in: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(q_in.element_type != cutlass.BFloat16):
            raise TypeError("Q must use BF16 elements")
        if cutlass.const_expr(k_in.element_type != cutlass.BFloat16):
            raise TypeError("K must use BF16 elements")
        if cutlass.const_expr(v_in.element_type != cutlass.BFloat16):
            raise TypeError("V must use BF16 elements")
        if cutlass.const_expr(decay_s.element_type != cutlass.Float32):
            raise TypeError("decay_s must use FP32 elements")
        if cutlass.const_expr(initial_state_in.element_type != cutlass.Float32):
            raise TypeError("initial state placeholder must use FP32 elements")
        if cutlass.const_expr(final_state_in.element_type != cutlass.Float32):
            raise TypeError("final state placeholder must use FP32 elements")
        if cutlass.const_expr(self.is_varlen):
            if cutlass.const_expr(cu_seqlens_in.element_type != cutlass.Int32):
                raise TypeError("cu_seqlens must use INT32 elements")
            if cutlass.const_expr(initial_state_indices_in.element_type != cutlass.Int32):
                raise TypeError("initial_state_indices must use INT32 elements")
            if cutlass.const_expr(tensormaps_in.element_type != cutlass.Uint8):
                raise TypeError("TensorMap workspace must use UINT8 elements")
        if cutlass.const_expr(output_in.element_type != cutlass.BFloat16):
            raise TypeError("output must use BF16 elements")

        B = self.batch_size
        # Keep T runtime-typed while building TensorMaps.  With a static T=1,
        # CuTe DSL 4.5.1 coalesces the (D,T) descriptor to 1D and emits
        # UTMALDG.1D for a still-2D 128x64 tile, which traps on Hopper.  The
        # runtime extent preserves the intended 2D descriptor for every T,
        # including one, without widening its logical allocation bound.
        T = sequence_length
        specialized_T = self.sequence_length
        H = self.qk_heads
        HV = self.value_heads
        D = HEAD_DIM
        expected_qk_elements = B * specialized_T * H * D
        expected_vo_elements = B * specialized_T * HV * D
        state_batches = self.state_pool_size if self.is_varlen else B
        expected_state_elements = state_batches * HV * VALUE_DIM * HEAD_DIM
        if cutlass.const_expr(cute.size(q_in) != expected_qk_elements):
            raise ValueError("Q size does not match the compiled specialization")
        if cutlass.const_expr(cute.size(k_in) != expected_qk_elements):
            raise ValueError("K size does not match the compiled specialization")
        if cutlass.const_expr(cute.size(v_in) != expected_vo_elements):
            raise ValueError("V size does not match the compiled specialization")
        if cutlass.const_expr(cute.size(output_in) != expected_vo_elements):
            raise ValueError("output size does not match the compiled specialization")
        if cutlass.const_expr(cute.size(decay_s) != self.decay_heads):
            raise ValueError("decay_s size does not match the compiled specialization")
        if cutlass.const_expr(self.needs_initial_state):
            if cutlass.const_expr(
                cute.size(initial_state_in) != expected_state_elements,
            ):
                raise ValueError("initial state size does not match the compiled specialization")
        if cutlass.const_expr(self.needs_final_state):
            if cutlass.const_expr(cute.size(final_state_in) != expected_state_elements):
                raise ValueError("final state size does not match the compiled specialization")
        if cutlass.const_expr(self.is_varlen):
            if cutlass.const_expr(cute.size(cu_seqlens_in) != self.num_sequences + 1):
                raise ValueError("cu_seqlens size does not match the compiled specialization")
            if cutlass.const_expr(cute.size(initial_state_indices_in) != self.num_sequences):
                raise ValueError("initial_state_indices size does not match the compiled specialization")

        # Torch row-major [B,T,H,D] becomes the exact two logical views used by
        # the warp-specialized schedule. Q uses (T,D); K/V/O use (D,T), with the
        # final nested mode selecting one (head,batch) CTA.
        q_layout = cute.make_layout(
            (T, D, (H, B)),
            stride=(D * H, 1, (D, D * H * T)),
        )
        k_layout = cute.make_layout(
            (D, T, (H, B)),
            stride=(1, D * H, (D, D * H * T)),
        )
        vo_layout = cute.make_layout(
            (D, T, (HV, B)),
            stride=(1, D * HV, (D, D * HV * T)),
        )
        # Public torch state is row-major [B,HV,V,K].  The first logical mode
        # below is K (unit stride); _swap_first_two_modes exposes VxK to the
        # distributed state-C fragment without a physical transpose.
        state_layout = cute.make_layout(
            (HEAD_DIM, VALUE_DIM, (HV, state_batches)),
            stride=(1, HEAD_DIM, (HEAD_DIM * VALUE_DIM, HV * HEAD_DIM * VALUE_DIM)),
        )
        q = cute.make_tensor(q_in.iterator, q_layout)
        k_t = cute.make_tensor(k_in.iterator, k_layout)
        v_t = cute.make_tensor(v_in.iterator, vo_layout)
        output = cute.make_tensor(output_in.iterator, vo_layout)
        initial_state = cute.make_tensor(initial_state_in.iterator, state_layout)
        final_state = cute.make_tensor(final_state_in.iterator, state_layout)

        bf16 = cutlass.BFloat16
        f32 = cutlass.Float32
        k_major = warpgroup.OperandMajorMode.K
        mn_major = warpgroup.OperandMajorMode.MN
        row_major = utils.LayoutEnum.ROW_MAJOR
        column_major = utils.LayoutEnum.COL_MAJOR

        qk_ss_mma = sm90_utils.make_trivial_tiled_mma(
            bf16,
            bf16,
            k_major,
            k_major,
            f32,
            (1, 1, 1),
            (64, 64),
            warpgroup.OperandSource.SMEM,
        )
        state_rs_mma = sm90_utils.make_trivial_tiled_mma(
            bf16,
            bf16,
            k_major,
            mn_major,
            f32,
            (2, 1, 1),
            (64, HEAD_DIM),
            warpgroup.OperandSource.RMEM,
        )
        o1_rs_mma = sm90_utils.make_trivial_tiled_mma(
            bf16,
            bf16,
            k_major,
            k_major,
            f32,
            (2, 1, 1),
            (64, 64),
            warpgroup.OperandSource.RMEM,
        )
        o2_rs_mma = sm90_utils.make_trivial_tiled_mma(
            bf16,
            bf16,
            k_major,
            k_major,
            f32,
            (2, 1, 1),
            (64, 64),
            warpgroup.OperandSource.RMEM,
        )
        assert qk_ss_mma.size == THREADS_PER_WARP_GROUP
        assert state_rs_mma.size == MATH_THREAD_COUNT
        assert o1_rs_mma.size == MATH_THREAD_COUNT
        assert o2_rs_mma.size == MATH_THREAD_COUNT

        q_layout_staged = sm90_utils.make_smem_layout_a(
            row_major,
            QK_TILE_SHAPE,
            bf16,
            Q_STAGES,
        )
        k_state_layout_staged = sm90_utils.make_smem_layout_b(
            column_major,
            STATE_TILE_SHAPE,
            bf16,
            K_STAGES,
        )
        k_qk_layout_staged = cute.select(k_state_layout_staged, mode=[1, 0, 2])
        v_layout_staged = sm90_utils.make_smem_layout_a(
            column_major,
            STATE_TILE_SHAPE,
            bf16,
            V_STAGES,
        )
        o_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            bf16,
        )
        o_layout_staged = cute.tile_to_shape(
            o_layout_atom,
            (VALUE_DIM, 64, O_STAGES),
            order=(1, 0, 2),
        )
        assert cute.cosize(k_state_layout_staged) == cute.cosize(k_qk_layout_staged)

        qk_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_INTER,
            bf16,
        )
        qk_publication_layout = cute.tile_to_shape(
            qk_layout_atom,
            QK_SHAPE,
            order=(0, 1),
        )
        qk_r2s_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                transpose=QK_R2S_TRANSPOSE,
                num_matrices=QK_R2S_NUM_MATRICES,
            ),
            bf16,
        )
        qk_r2s_tiled_copy = cute.make_tiled_copy_C(qk_r2s_atom, qk_ss_mma)
        rs_a_s2r_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(
                transpose=RS_A_S2R_TRANSPOSE,
                num_matrices=RS_A_S2R_NUM_MATRICES,
            ),
            bf16,
        )
        state_a_s2r_tiled_copy = cute.make_tiled_copy_A(rs_a_s2r_atom, state_rs_mma)
        o2_a_s2r_tiled_copy = cute.make_tiled_copy_A(rs_a_s2r_atom, o2_rs_mma)

        q_stage_layout = cute.slice_(q_layout_staged, (None, None, 0))
        k_stage_layout = cute.slice_(k_state_layout_staged, (None, None, 0))
        v_stage_layout = cute.slice_(v_layout_staged, (None, None, 0))
        o_stage_layout = cute.slice_(o_layout_staged, (None, None, 0))
        q_tma_atom, q_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            q,
            q_stage_layout,
            (64, HEAD_DIM),
        )
        k_tma_atom, k_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_t,
            k_stage_layout,
            (HEAD_DIM, 64),
        )
        v_tma_atom, v_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            v_t,
            v_stage_layout,
            (VALUE_DIM, 64),
        )
        o_tma_atom, o_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            output,
            o_stage_layout,
            O_SHAPE,
        )

        self.q_tma_bytes = cute.size_in_bytes(bf16, q_stage_layout)
        self.k_tma_bytes = cute.size_in_bytes(bf16, k_stage_layout)
        self.v_tma_bytes = cute.size_in_bytes(bf16, v_stage_layout)

        @cute.struct
        class SharedStorage:
            q_mbarriers: cute.struct.MemRange[cutlass.Int64, Q_BARRIER_COUNT]
            k_mbarriers: cute.struct.MemRange[cutlass.Int64, K_BARRIER_COUNT]
            v_mbarriers: cute.struct.MemRange[cutlass.Int64, V_BARRIER_COUNT]
            o_mbarriers: cute.struct.MemRange[cutlass.Int64, O_BARRIER_COUNT]
            # The 260-byte LUT fits inside the inherited 848-byte alignment
            # gap between the 176 barrier bytes and Q's 1024-byte boundary.
            decay_lut: cute.struct.Align[
                cute.struct.MemRange[f32, DECAY_LUT_ENTRIES],
                16,
            ]
            q: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(q_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            k: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(k_state_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            v: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(v_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            qk: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(qk_publication_layout)],
                SMEM_ALIGNMENT_BYTES,
            ]
            o: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(o_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage
        self.dynamic_smem_bytes = SharedStorage.size_in_bytes()
        assert self.dynamic_smem_bytes == DYNAMIC_SMEM_ESTIMATE_BYTES
        assert self.dynamic_smem_bytes <= SM90_OPTIN_SMEM_LIMIT_BYTES

        kernel_args = (
            decay_s,
            initial_state,
            final_state,
            cu_seqlens_in,
            initial_state_indices_in,
            tensormaps_in,
            scale,
            sequence_length,
            qk_ss_mma,
            state_rs_mma,
            o1_rs_mma,
            o2_rs_mma,
            qk_r2s_tiled_copy,
            state_a_s2r_tiled_copy,
            o2_a_s2r_tiled_copy,
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            v_tma_atom,
            v_tma_tensor,
            o_tma_atom,
            o_tma_tensor,
            q_layout_staged,
            k_state_layout_staged,
            k_qk_layout_staged,
            v_layout_staged,
            qk_publication_layout,
            o_layout_staged,
        )
        grid = (1, HV, B)
        if cutlass.const_expr(self.is_varlen):
            grid = (1, HV, self.num_sequences)
        if cutlass.const_expr(self.persistent):
            kernel = self.kernel_varlen_persistent(*kernel_args)
            grid = (self.persistent_ctas, 1, 1)
        else:
            kernel = self.kernel_nonpersistent(*kernel_args)
        kernel.launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            smem=self.dynamic_smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def populate_decay_lut(
        self,
        tidx: cutlass.Int32,
        decay_s: cute.Tensor,
        decay_head_idx: cutlass.Int32,
        decay_lut: cute.Tensor,
    ):
        """Build ``lambda**k`` with the frozen C++ warp-scan rounding order."""

        if tidx < cutlass.Int32(32):
            lane_id = tidx % cutlass.Int32(32)
            decay_lambda = cutlass.Float32(0.0)
            if lane_id == cutlass.Int32(0):
                decay_lambda = cute.exp(
                    -decay_s[decay_head_idx],
                    fastmath=False,
                )
            decay_lambda = cute.arch.shuffle_sync(
                decay_lambda,
                0,
                mask=-1,
                mask_and_clamp=31,
            )

            for base in [0, 32, 64]:
                product = decay_lambda
                for offset in [1, 2, 4, 8, 16]:
                    shuffled = cute.arch.shuffle_sync_bfly(
                        product,
                        offset=offset,
                        mask=-1,
                        mask_and_clamp=31,
                    )
                    if lane_id > cutlass.Int32(offset):
                        product = product * shuffled

                if lane_id == cutlass.Int32(0):
                    product = cutlass.Float32(1.0)
                if cutlass.const_expr(base != 0):
                    product = product * (decay_lambda * decay_lut[cutlass.Int32(base - 1)])

                index = cutlass.Int32(base) + lane_id
                if index < cutlass.Int32(DECAY_LUT_ENTRIES):
                    decay_lut[index] = product
                if cutlass.const_expr(base != 64):
                    # Order the shared carry before the next segment reads it.
                    cute.arch.sync_warp()

    @cute.kernel
    def kernel_nonpersistent(
        self,
        decay_s: cute.Tensor,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state_indices: cute.Tensor,
        g_tensormaps: cute.Tensor,
        scale: cutlass.Float32,
        sequence_length: cutlass.Int32,
        qk_ss_mma: cute.TiledMma,
        state_rs_mma: cute.TiledMma,
        o1_rs_mma: cute.TiledMma,
        o2_rs_mma: cute.TiledMma,
        qk_r2s_tiled_copy: cute.TiledCopy,
        state_a_s2r_tiled_copy: cute.TiledCopy,
        o2_a_s2r_tiled_copy: cute.TiledCopy,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        v_tma_atom: cute.CopyAtom,
        v_tma_tensor: cute.Tensor,
        o_tma_atom: cute.CopyAtom,
        o_tma_tensor: cute.Tensor,
        q_layout_staged: cute.ComposedLayout,
        k_state_layout_staged: cute.ComposedLayout,
        k_qk_layout_staged: cute.ComposedLayout,
        v_layout_staged: cute.ComposedLayout,
        qk_publication_layout: cute.ComposedLayout,
        o_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // THREADS_PER_WARP_GROUP)
        _, value_head_idx, batch_idx = cute.arch.block_idx()
        tensormap_workspace_slot = cutlass.Int32(0)
        if cutlass.const_expr(self.is_varlen):
            # The (sequence, value-head) work-unit ID is unique in grid (1, HV, N).
            tensormap_workspace_slot = batch_idx * cutlass.Int32(self.value_heads) + value_head_idx
        tensor_batch_idx = batch_idx
        state_idx = batch_idx
        sequence_bos = cutlass.Int32(0)
        sequence_length_use = sequence_length
        if cutlass.const_expr(self.is_varlen):
            tensor_batch_idx = cutlass.Int32(0)
            sequence_bos = cu_seqlens[batch_idx]
            sequence_length_use = cu_seqlens[batch_idx + cutlass.Int32(1)] - sequence_bos
            state_idx = initial_state_indices[batch_idx]
        qk_head_idx = value_head_idx // cutlass.Int32(self.group_size)
        decay_head_idx = value_head_idx
        if cutlass.const_expr(self.decay_heads == self.qk_heads):
            decay_head_idx = qk_head_idx

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        s_decay_lut = storage.decay_lut.get_tensor(cute.make_layout((DECAY_LUT_ENTRIES,)))
        s_q = storage.q.get_tensor(q_layout_staged.outer, swizzle=q_layout_staged.inner)
        s_k_state = storage.k.get_tensor(
            k_state_layout_staged.outer,
            swizzle=k_state_layout_staged.inner,
        )
        s_k_qk = storage.k.get_tensor(
            k_qk_layout_staged.outer,
            swizzle=k_qk_layout_staged.inner,
        )
        s_v = storage.v.get_tensor(v_layout_staged.outer, swizzle=v_layout_staged.inner)
        s_qk = storage.qk.get_tensor(
            qk_publication_layout.outer,
            swizzle=qk_publication_layout.inner,
        )
        s_o = storage.o.get_tensor(o_layout_staged.outer, swizzle=o_layout_staged.inner)

        # One CTA owns one value/state head, so the decay load is CTA-uniform.
        # Compute k=0..64 before register redistribution; the shared LUT then
        # has no live register dependency across either role branch.
        self.populate_decay_lut(
            tidx,
            decay_s,
            decay_head_idx,
            s_decay_lut,
        )
        cute.arch.sync_threads()

        q_head = q_tma_tensor[(None, None, (qk_head_idx, tensor_batch_idx))]
        k_head = k_tma_tensor[(None, None, (qk_head_idx, tensor_batch_idx))]
        v_head = v_tma_tensor[(None, None, (value_head_idx, tensor_batch_idx))]
        if cutlass.const_expr(self.is_varlen):
            q_head = cute.domain_offset((sequence_bos, cutlass.Int32(0)), q_head)
            k_head = cute.domain_offset((cutlass.Int32(0), sequence_bos), k_head)
            v_head = cute.domain_offset((cutlass.Int32(0), sequence_bos), v_head)
        tiled_q = cute.local_tile(q_head, (64, HEAD_DIM), (None, None))
        tiled_k = cute.local_tile(k_head, (HEAD_DIM, 64), (None, None))
        tiled_v = cute.local_tile(v_head, (VALUE_DIM, 64), (None, None))
        q_smem, q_gmem = cpasync.tma_partition(
            q_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_q, 0, 2),
            cute.group_modes(tiled_q, 0, 2),
        )
        k_smem, k_gmem = cpasync.tma_partition(
            k_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_k_state, 0, 2),
            cute.group_modes(tiled_k, 0, 2),
        )
        v_smem, v_gmem = cpasync.tma_partition(
            v_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_v, 0, 2),
            cute.group_modes(tiled_v, 0, 2),
        )
        num_chunks = (sequence_length_use + cutlass.Int32(63)) // cutlass.Int32(64)

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            MATH_SIGNALING_THREADS,
        )
        math_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, MATH_THREAD_COUNT)
        epilogue_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.epilogue_threads,
        )
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbarriers.data_ptr(),
            num_stages=Q_STAGES,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.q_tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            defer_sync=True,
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbarriers.data_ptr(),
            num_stages=K_STAGES,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.k_tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            defer_sync=True,
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_mbarriers.data_ptr(),
            num_stages=V_STAGES,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.v_tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            defer_sync=True,
        )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.o_mbarriers.data_ptr(),
            num_stages=O_STAGES,
            producer_group=math_group,
            consumer_group=epilogue_group,
            defer_sync=False,
        )
        q_producer, q_consumer = q_pipeline.make_participants()
        k_producer, k_consumer = k_pipeline.make_participants()
        v_producer, v_consumer = v_pipeline.make_participants()
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=O_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            O_STAGES,
        )
        o_wait_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            O_STAGES,
        )
        o_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            O_STAGES,
        )
        qk_published_barrier = pipeline.NamedBarrier(
            barrier_id=QK_PUBLISHED_BARRIER_ID,
            num_threads=MATH_THREAD_COUNT,
        )
        qk_consumed_barrier = pipeline.NamedBarrier(
            barrier_id=QK_CONSUMED_BARRIER_ID,
            num_threads=MATH_THREAD_COUNT,
        )
        o_epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.epilogue_threads,
        )

        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.register_targets[0])
            if warp_idx == 0:
                self.run_tma_load_producer(
                    q_producer,
                    num_chunks,
                    q_tma_atom,
                    q_gmem,
                    q_smem,
                    True,
                )
            elif warp_idx == 1:
                self.run_tma_load_producer(
                    k_producer,
                    num_chunks,
                    k_tma_atom,
                    k_gmem,
                    k_smem,
                    False,
                )
            elif warp_idx == 2:
                self.run_tma_load_producer(
                    v_producer,
                    num_chunks,
                    v_tma_atom,
                    v_gmem,
                    v_smem,
                    False,
                )
            else:
                self.run_epilogue_store(
                    tidx,
                    warp_idx,
                    sequence_length_use,
                    sequence_bos,
                    num_chunks,
                    o_wait_state,
                    o_release_state,
                    o_pipeline,
                    tma_store_pipeline,
                    o_epilogue_barrier,
                    o_tma_atom,
                    o_tma_tensor,
                    value_head_idx,
                    tensor_batch_idx,
                    batch_idx,
                    tensormap_workspace_slot,
                    g_tensormaps,
                    s_o,
                )
        else:
            cute.arch.setmaxregister_increase(self.register_targets[1])
            self.run_math(
                tidx,
                warp_group_idx,
                s_decay_lut,
                sequence_length_use,
                num_chunks,
                o_producer_state,
                q_consumer,
                k_consumer,
                v_consumer,
                o_pipeline,
                qk_published_barrier,
                qk_consumed_barrier,
                qk_ss_mma,
                state_rs_mma,
                o1_rs_mma,
                o2_rs_mma,
                qk_r2s_tiled_copy,
                state_a_s2r_tiled_copy,
                o2_a_s2r_tiled_copy,
                s_q,
                s_k_state,
                s_k_qk,
                s_v,
                s_qk,
                s_o,
                initial_state,
                final_state,
                value_head_idx,
                state_idx,
                scale,
            )

    @cute.kernel
    def kernel_varlen_persistent(
        self,
        decay_s: cute.Tensor,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state_indices: cute.Tensor,
        g_tensormaps: cute.Tensor,
        scale: cutlass.Float32,
        sequence_length: cutlass.Int32,
        qk_ss_mma: cute.TiledMma,
        state_rs_mma: cute.TiledMma,
        o1_rs_mma: cute.TiledMma,
        o2_rs_mma: cute.TiledMma,
        qk_r2s_tiled_copy: cute.TiledCopy,
        state_a_s2r_tiled_copy: cute.TiledCopy,
        o2_a_s2r_tiled_copy: cute.TiledCopy,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        v_tma_atom: cute.CopyAtom,
        v_tma_tensor: cute.Tensor,
        o_tma_atom: cute.CopyAtom,
        o_tma_tensor: cute.Tensor,
        q_layout_staged: cute.ComposedLayout,
        k_state_layout_staged: cute.ComposedLayout,
        k_qk_layout_staged: cute.ComposedLayout,
        v_layout_staged: cute.ComposedLayout,
        qk_publication_layout: cute.ComposedLayout,
        o_layout_staged: cute.ComposedLayout,
    ):
        """Static-strided persistent scheduling over packed (sequence, HV) work units."""

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // THREADS_PER_WARP_GROUP)
        cta_idx, _, _ = cute.arch.block_idx()
        # The launch-time CTA ID owns one descriptor slot across its drained work units.

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        s_decay_lut = storage.decay_lut.get_tensor(cute.make_layout((DECAY_LUT_ENTRIES,)))
        s_q = storage.q.get_tensor(q_layout_staged.outer, swizzle=q_layout_staged.inner)
        s_k_state = storage.k.get_tensor(
            k_state_layout_staged.outer,
            swizzle=k_state_layout_staged.inner,
        )
        s_k_qk = storage.k.get_tensor(
            k_qk_layout_staged.outer,
            swizzle=k_qk_layout_staged.inner,
        )
        s_v = storage.v.get_tensor(v_layout_staged.outer, swizzle=v_layout_staged.inner)
        s_qk = storage.qk.get_tensor(
            qk_publication_layout.outer,
            swizzle=qk_publication_layout.inner,
        )
        s_o = storage.o.get_tensor(o_layout_staged.outer, swizzle=o_layout_staged.inner)

        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.register_targets[0])
        else:
            cute.arch.setmaxregister_increase(self.register_targets[1])

        self.run_persistent_scheduler(
            tidx,
            warp_idx,
            warp_group_idx,
            cta_idx,
            decay_s,
            initial_state,
            final_state,
            cu_seqlens,
            initial_state_indices,
            g_tensormaps,
            scale,
            qk_ss_mma,
            state_rs_mma,
            o1_rs_mma,
            o2_rs_mma,
            qk_r2s_tiled_copy,
            state_a_s2r_tiled_copy,
            o2_a_s2r_tiled_copy,
            q_tma_atom,
            q_tma_tensor,
            k_tma_atom,
            k_tma_tensor,
            v_tma_atom,
            v_tma_tensor,
            o_tma_atom,
            o_tma_tensor,
            s_decay_lut,
            s_q,
            s_k_state,
            s_k_qk,
            s_v,
            s_qk,
            s_o,
            storage.q_mbarriers.data_ptr(),
            storage.k_mbarriers.data_ptr(),
            storage.v_mbarriers.data_ptr(),
            storage.o_mbarriers.data_ptr(),
        )

    @cute.jit
    def run_persistent_scheduler(
        self,
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        warp_group_idx: cutlass.Int32,
        cta_idx: cutlass.Int32,
        decay_s: cute.Tensor,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state_indices: cute.Tensor,
        g_tensormaps: cute.Tensor,
        scale: cutlass.Float32,
        qk_ss_mma: cute.TiledMma,
        state_rs_mma: cute.TiledMma,
        o1_rs_mma: cute.TiledMma,
        o2_rs_mma: cute.TiledMma,
        qk_r2s_tiled_copy: cute.TiledCopy,
        state_a_s2r_tiled_copy: cute.TiledCopy,
        o2_a_s2r_tiled_copy: cute.TiledCopy,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        v_tma_atom: cute.CopyAtom,
        v_tma_tensor: cute.Tensor,
        o_tma_atom: cute.CopyAtom,
        o_tma_tensor: cute.Tensor,
        s_decay_lut: cute.Tensor,
        s_q: cute.Tensor,
        s_k_state: cute.Tensor,
        s_k_qk: cute.Tensor,
        s_v: cute.Tensor,
        s_qk: cute.Tensor,
        s_o: cute.Tensor,
        q_mbarriers,
        k_mbarriers,
        v_mbarriers,
        o_mbarriers,
    ):
        """Static-strided scheduling over packed work units."""

        work_idx = cutlass.Int32(cta_idx)
        total_work_units = cutlass.Int32(self.num_sequences * self.value_heads)
        work_stride = cutlass.Int32(self.persistent_ctas)
        while work_idx < total_work_units:
            self.run_persistent_work_unit(
                tidx,
                warp_idx,
                warp_group_idx,
                work_idx,
                cta_idx,
                decay_s,
                initial_state,
                final_state,
                cu_seqlens,
                initial_state_indices,
                g_tensormaps,
                scale,
                qk_ss_mma,
                state_rs_mma,
                o1_rs_mma,
                o2_rs_mma,
                qk_r2s_tiled_copy,
                state_a_s2r_tiled_copy,
                o2_a_s2r_tiled_copy,
                q_tma_atom,
                q_tma_tensor,
                k_tma_atom,
                k_tma_tensor,
                v_tma_atom,
                v_tma_tensor,
                o_tma_atom,
                o_tma_tensor,
                s_decay_lut,
                s_q,
                s_k_state,
                s_k_qk,
                s_v,
                s_qk,
                s_o,
                q_mbarriers,
                k_mbarriers,
                v_mbarriers,
                o_mbarriers,
            )
            # WGMMA/TMA use the async shared-memory proxy.  Complete that
            # proxy's view before the CTA barrier permits the next work unit
            # to overwrite reused shared tiles and pipeline storage.
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
            work_idx = work_idx + work_stride

    @cute.jit
    def run_persistent_work_unit(
        self,
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        warp_group_idx: cutlass.Int32,
        work_idx: cutlass.Int32,
        tensormap_workspace_slot: cutlass.Int32,
        decay_s: cute.Tensor,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state_indices: cute.Tensor,
        g_tensormaps: cute.Tensor,
        scale: cutlass.Float32,
        qk_ss_mma: cute.TiledMma,
        state_rs_mma: cute.TiledMma,
        o1_rs_mma: cute.TiledMma,
        o2_rs_mma: cute.TiledMma,
        qk_r2s_tiled_copy: cute.TiledCopy,
        state_a_s2r_tiled_copy: cute.TiledCopy,
        o2_a_s2r_tiled_copy: cute.TiledCopy,
        q_tma_atom: cute.CopyAtom,
        q_tma_tensor: cute.Tensor,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        v_tma_atom: cute.CopyAtom,
        v_tma_tensor: cute.Tensor,
        o_tma_atom: cute.CopyAtom,
        o_tma_tensor: cute.Tensor,
        s_decay_lut: cute.Tensor,
        s_q: cute.Tensor,
        s_k_state: cute.Tensor,
        s_k_qk: cute.Tensor,
        s_v: cute.Tensor,
        s_qk: cute.Tensor,
        s_o: cute.Tensor,
        q_mbarriers,
        k_mbarriers,
        v_mbarriers,
        o_mbarriers,
    ):
        """Run one fully drained work unit without leaking pipeline objects to the scheduler."""

        value_head_idx = work_idx % cutlass.Int32(self.value_heads)
        sequence_idx = work_idx // cutlass.Int32(self.value_heads)
        qk_head_idx = value_head_idx // cutlass.Int32(self.group_size)
        decay_head_idx = value_head_idx
        if cutlass.const_expr(self.decay_heads == self.qk_heads):
            decay_head_idx = qk_head_idx
        sequence_bos = cu_seqlens[sequence_idx]
        sequence_length_use = cu_seqlens[sequence_idx + cutlass.Int32(1)] - sequence_bos
        state_idx = initial_state_indices[sequence_idx]

        # Reused persistent shared storage must have an explicit CTA
        # rendezvous immediately before the next work unit rewrites its
        # per-head decay table.  The scheduler-level drain orders the async
        # proxies; this local rendezvous makes the generic shared-memory
        # read-to-write handoff explicit at the reuse site as well.
        cute.arch.sync_threads()
        self.populate_decay_lut(
            tidx,
            decay_s,
            decay_head_idx,
            s_decay_lut,
        )
        cute.arch.sync_threads()

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            MATH_SIGNALING_THREADS,
        )
        math_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, MATH_THREAD_COUNT)
        epilogue_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.epilogue_threads,
        )
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=q_mbarriers,
            num_stages=Q_STAGES,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.q_tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            defer_sync=True,
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=k_mbarriers,
            num_stages=K_STAGES,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.k_tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            defer_sync=True,
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=v_mbarriers,
            num_stages=V_STAGES,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.v_tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            defer_sync=True,
        )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=o_mbarriers,
            num_stages=O_STAGES,
            producer_group=math_group,
            consumer_group=epilogue_group,
            defer_sync=False,
        )
        q_producer, q_consumer = q_pipeline.make_participants()
        k_producer, k_consumer = k_pipeline.make_participants()
        v_producer, v_consumer = v_pipeline.make_participants()
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=O_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            O_STAGES,
        )
        o_wait_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            O_STAGES,
        )
        o_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            O_STAGES,
        )
        qk_published_barrier = pipeline.NamedBarrier(
            barrier_id=QK_PUBLISHED_BARRIER_ID,
            num_threads=MATH_THREAD_COUNT,
        )
        qk_consumed_barrier = pipeline.NamedBarrier(
            barrier_id=QK_CONSUMED_BARRIER_ID,
            num_threads=MATH_THREAD_COUNT,
        )
        o_epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.epilogue_threads,
        )

        q_head = q_tma_tensor[(None, None, (qk_head_idx, cutlass.Int32(0)))]
        k_head = k_tma_tensor[(None, None, (qk_head_idx, cutlass.Int32(0)))]
        v_head = v_tma_tensor[(None, None, (value_head_idx, cutlass.Int32(0)))]
        q_head = cute.domain_offset((sequence_bos, cutlass.Int32(0)), q_head)
        k_head = cute.domain_offset((cutlass.Int32(0), sequence_bos), k_head)
        v_head = cute.domain_offset((cutlass.Int32(0), sequence_bos), v_head)
        tiled_q = cute.local_tile(q_head, (64, HEAD_DIM), (None, None))
        tiled_k = cute.local_tile(k_head, (HEAD_DIM, 64), (None, None))
        tiled_v = cute.local_tile(v_head, (VALUE_DIM, 64), (None, None))
        q_smem, q_gmem = cpasync.tma_partition(
            q_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_q, 0, 2),
            cute.group_modes(tiled_q, 0, 2),
        )
        k_smem, k_gmem = cpasync.tma_partition(
            k_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_k_state, 0, 2),
            cute.group_modes(tiled_k, 0, 2),
        )
        v_smem, v_gmem = cpasync.tma_partition(
            v_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_v, 0, 2),
            cute.group_modes(tiled_v, 0, 2),
        )
        num_chunks = (sequence_length_use + cutlass.Int32(63)) // cutlass.Int32(64)

        if warp_group_idx == 0:
            if warp_idx == 0:
                self.run_tma_load_producer(
                    q_producer,
                    num_chunks,
                    q_tma_atom,
                    q_gmem,
                    q_smem,
                    True,
                )
            elif warp_idx == 1:
                self.run_tma_load_producer(
                    k_producer,
                    num_chunks,
                    k_tma_atom,
                    k_gmem,
                    k_smem,
                    False,
                )
            elif warp_idx == 2:
                self.run_tma_load_producer(
                    v_producer,
                    num_chunks,
                    v_tma_atom,
                    v_gmem,
                    v_smem,
                    False,
                )
            else:
                self.run_epilogue_store(
                    tidx,
                    warp_idx,
                    sequence_length_use,
                    sequence_bos,
                    num_chunks,
                    o_wait_state,
                    o_release_state,
                    o_pipeline,
                    tma_store_pipeline,
                    o_epilogue_barrier,
                    o_tma_atom,
                    o_tma_tensor,
                    value_head_idx,
                    cutlass.Int32(0),
                    sequence_idx,
                    tensormap_workspace_slot,
                    g_tensormaps,
                    s_o,
                )
        else:
            self.run_math(
                tidx,
                warp_group_idx,
                s_decay_lut,
                sequence_length_use,
                num_chunks,
                o_producer_state,
                q_consumer,
                k_consumer,
                v_consumer,
                o_pipeline,
                qk_published_barrier,
                qk_consumed_barrier,
                qk_ss_mma,
                state_rs_mma,
                o1_rs_mma,
                o2_rs_mma,
                qk_r2s_tiled_copy,
                state_a_s2r_tiled_copy,
                o2_a_s2r_tiled_copy,
                s_q,
                s_k_state,
                s_k_qk,
                s_v,
                s_qk,
                s_o,
                initial_state,
                final_state,
                value_head_idx,
                state_idx,
                scale,
            )

    @cute.jit
    def run_tma_load_producer(
        self,
        load_producer,
        num_chunks: cutlass.Int32,
        tma_atom: cute.CopyAtom,
        gmem_partition: cute.Tensor,
        smem_partition: cute.Tensor,
        q_orientation: bool,
    ):
        for chunk in cutlass.range(num_chunks, unroll=1):
            handle = load_producer.acquire_and_advance()
            if cutlass.const_expr(q_orientation):
                source = gmem_partition[(None, chunk, 0)]
            else:
                source = gmem_partition[(None, 0, chunk)]
            cute.copy(
                tma_atom,
                source,
                smem_partition[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            handle.commit()
        return load_producer

    @cute.jit
    def run_math(
        self,
        tidx: cutlass.Int32,
        warp_group_idx: cutlass.Int32,
        decay_lut: cute.Tensor,
        sequence_length: cutlass.Int32,
        num_chunks: cutlass.Int32,
        o_state,
        q_consumer,
        k_consumer,
        v_consumer,
        o_pipeline,
        qk_published_barrier: pipeline.NamedBarrier,
        qk_consumed_barrier: pipeline.NamedBarrier,
        qk_ss_mma: cute.TiledMma,
        state_rs_mma: cute.TiledMma,
        o1_rs_mma: cute.TiledMma,
        o2_rs_mma: cute.TiledMma,
        qk_r2s_tiled_copy: cute.TiledCopy,
        state_a_s2r_tiled_copy: cute.TiledCopy,
        o2_a_s2r_tiled_copy: cute.TiledCopy,
        s_q: cute.Tensor,
        s_k_state: cute.Tensor,
        s_k_qk: cute.Tensor,
        s_v: cute.Tensor,
        s_qk: cute.Tensor,
        s_o: cute.Tensor,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        value_head_idx: cutlass.Int32,
        state_idx: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        local_math_tid = tidx - MATH0_WARP_GROUP_INDEX * THREADS_PER_WARP_GROUP
        state_thread = state_rs_mma.get_slice(local_math_tid)
        o1_thread = o1_rs_mma.get_slice(local_math_tid)
        o2_thread = o2_rs_mma.get_slice(local_math_tid)
        o_r2s_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                transpose=True,
                num_matrices=4,
            ),
            cutlass.BFloat16,
        )
        o_r2s_tiled_copy = cute.make_tiled_copy_C(o_r2s_atom, o1_rs_mma)

        state_accumulator = state_thread.make_fragment_C(
            state_thread.partition_shape_C(STATE_SHAPE),
        )
        for item in cutlass.range_constexpr(cute.size(state_accumulator)):
            state_accumulator[item] = cutlass.Float32(0.0)
        if cutlass.const_expr(self.needs_initial_state):
            initial_state_head = initial_state[(None, None, (value_head_idx, state_idx))]
            self.load_state_fragment(
                state_accumulator,
                initial_state_head,
                state_rs_mma,
                local_math_tid,
            )

        for chunk in cutlass.range(num_chunks, unroll=1):
            q_handle = q_consumer.wait_and_advance()
            k_handle = k_consumer.wait_and_advance()
            v_handle = v_consumer.wait_and_advance()
            valid_tokens = sequence_length - chunk * cutlass.Int32(64)
            if chunk < num_chunks - cutlass.Int32(1):
                valid_tokens = cutlass.Int32(64)

            if warp_group_idx == MATH0_WARP_GROUP_INDEX:
                qk_local_tid = tidx - MATH0_WARP_GROUP_INDEX * THREADS_PER_WARP_GROUP
                qk_thread = qk_ss_mma.get_slice(qk_local_tid)
                q_fragment_staged = qk_thread.make_fragment_A(qk_thread.partition_A(s_q))
                k_fragment_staged = qk_thread.make_fragment_B(qk_thread.partition_B(s_k_qk))
                q_fragment = q_fragment_staged[(None, None, None, q_handle.index)]
                k_fragment = k_fragment_staged[(None, None, None, k_handle.index)]
                qk_accumulator = qk_thread.make_fragment_C(
                    qk_thread.partition_shape_C(QK_SHAPE),
                )
                warpgroup.fence()
                self.issue_wgmma_ss(qk_ss_mma, q_fragment, k_fragment, qk_accumulator)
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                qk_coordinates = qk_thread.partition_C(cute.make_identity_tensor(QK_SHAPE))
                for item in cutlass.range_constexpr(cute.size(qk_accumulator)):
                    coordinate = qk_coordinates[item]
                    query_token = coordinate[0]
                    key_token = coordinate[1]
                    transformed = cutlass.Float32(0.0)
                    if query_token < valid_tokens:
                        if key_token < valid_tokens:
                            if query_token >= key_token:
                                transformed = qk_accumulator[item] * decay_lut[query_token - key_token]
                    qk_accumulator[item] = transformed

                qk_copy_thread = qk_r2s_tiled_copy.get_slice(qk_local_tid)
                qk_copy_source = qk_copy_thread.retile(qk_accumulator)
                qk_publication_fragment = cute.make_fragment_like(
                    qk_copy_source,
                    cutlass.BFloat16,
                )
                qk_publication_fragment.store(
                    qk_copy_source.load().to(cutlass.BFloat16),
                )
                qk_copy_destination = qk_copy_thread.partition_D(s_qk)
                cute.copy(
                    qk_r2s_tiled_copy,
                    qk_publication_fragment,
                    qk_copy_destination,
                )
                cute.arch.fence_view_async_shared()

            qk_published_barrier.sync()

            o1_a = self.make_acc_into_op(
                state_accumulator,
                o1_rs_mma.tv_layout_A,
                cutlass.BFloat16,
            )
            o1_b_staged = o1_thread.make_fragment_B(o1_thread.partition_B(s_q))
            o1_b = o1_b_staged[(None, None, None, q_handle.index)]
            o_accumulator = o1_thread.make_fragment_C(o1_thread.partition_shape_C(O_SHAPE))
            self.fence_register_fragment(o1_a)
            warpgroup.fence()
            self.issue_wgmma_rs_zero(o1_rs_mma, o1_a, o1_b, o_accumulator)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

            o_coordinates = o1_thread.partition_C(cute.make_identity_tensor(O_SHAPE))
            for item in cutlass.range_constexpr(cute.size(o_accumulator)):
                token = o_coordinates[item][1]
                o_accumulator[item] = o_accumulator[item] * decay_lut[token + 1]

            q_handle.release()

            o2_a = o2_thread.make_fragment_A(
                o2_thread.partition_A(s_v[(None, None, 0)]),
            )
            o2_copy_thread = o2_a_s2r_tiled_copy.get_slice(local_math_tid)
            o2_copy_source_staged = o2_copy_thread.partition_S(s_v)
            cute.copy(
                o2_a_s2r_tiled_copy,
                o2_copy_source_staged[(None, None, None, v_handle.index)],
                o2_copy_thread.retile(o2_a),
            )
            o2_b = o2_thread.make_fragment_B(o2_thread.partition_B(s_qk))
            self.fence_register_fragment(o2_a)
            self.fence_register_fragment(o_accumulator)
            warpgroup.fence()
            self.issue_wgmma_rs_accumulate(o2_rs_mma, o2_a, o2_b, o_accumulator)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

            if chunk < num_chunks - cutlass.Int32(1):
                if warp_group_idx == MATH0_WARP_GROUP_INDEX:
                    qk_consumed_barrier.wait_unaligned()
                else:
                    qk_consumed_barrier.arrive_unaligned()

            o_pipeline.producer_acquire(o_state)
            o_copy_thread = o_r2s_tiled_copy.get_slice(local_math_tid)
            o_copy_source = o_copy_thread.retile(o_accumulator)
            o_publication_fragment = cute.make_fragment_like(
                o_copy_source,
                cutlass.BFloat16,
            )
            for item in cutlass.range_constexpr(cute.size(o_copy_source)):
                o_publication_fragment[item] = cutlass.BFloat16(o_copy_source[item] * scale)
            o_copy_destination_staged = o_copy_thread.partition_D(s_o)
            o_copy_destination = o_copy_destination_staged[(None, None, None, o_state.index)]
            # SharedStorage aligns the O ring to 1024 bytes, and every 16 KiB
            # stage preserves the 16-byte alignment required by STSM.  Retain
            # that proof after partitioning and runtime stage selection, where
            # the CuTe type would otherwise fall back to element alignment.
            o_copy_destination = cute.make_tensor(
                o_copy_destination.iterator.align(16),
                o_copy_destination.layout,
            )
            cute.copy(
                o_r2s_tiled_copy,
                o_publication_fragment,
                o_copy_destination,
            )
            cute.arch.fence_view_async_shared()
            o_pipeline.producer_commit(o_state)
            o_state.advance()

            for item in cutlass.range_constexpr(cute.size(state_accumulator)):
                state_accumulator[item] = state_accumulator[item] * decay_lut[valid_tokens]

            state_a = state_thread.make_fragment_A(
                state_thread.partition_A(s_v[(None, None, 0)]),
            )
            state_copy_thread = state_a_s2r_tiled_copy.get_slice(local_math_tid)
            state_copy_source_staged = state_copy_thread.partition_S(s_v)
            cute.copy(
                state_a_s2r_tiled_copy,
                state_copy_source_staged[(None, None, None, v_handle.index)],
                state_copy_thread.retile(state_a),
            )
            state_a_coordinates = state_thread.partition_A(
                cute.make_identity_tensor((VALUE_DIM, 64)),
            )
            for item in cutlass.range_constexpr(cute.size(state_a)):
                token = state_a_coordinates[item][1]
                weighted = cutlass.Float32(0.0)
                if token < valid_tokens:
                    weighted = cutlass.Float32(state_a[item]) * decay_lut[valid_tokens - cutlass.Int32(1) - token]
                state_a[item] = cutlass.BFloat16(weighted)
            state_b_staged = state_thread.make_fragment_B(state_thread.partition_B(s_k_state))
            state_b = state_b_staged[(None, None, None, k_handle.index)]
            self.fence_register_fragment(state_a)
            self.fence_register_fragment(state_accumulator)
            warpgroup.fence()
            self.issue_wgmma_rs_accumulate(
                state_rs_mma,
                state_a,
                state_b,
                state_accumulator,
            )
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            k_handle.release()
            v_handle.release()

        o_pipeline.producer_tail(o_state)
        if cutlass.const_expr(self.needs_final_state):
            final_state_head = final_state[(None, None, (value_head_idx, state_idx))]
            self.store_state_fragment(
                state_accumulator,
                final_state_head,
                state_rs_mma,
                local_math_tid,
            )
        return q_consumer, k_consumer, v_consumer, o_state

    @cute.jit
    def load_state_fragment(
        self,
        state_accumulator: cute.Tensor,
        global_state: cute.Tensor,
        state_rs_mma: cute.TiledMma,
        local_math_tid: cutlass.Int32,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            global_state.element_type,
        )
        tiled_copy = cute.make_tiled_copy_C(copy_atom, state_rs_mma)
        thread_copy = tiled_copy.get_slice(local_math_tid)
        source = thread_copy.partition_S(_swap_first_two_modes(global_state))
        cute.copy(tiled_copy, source, state_accumulator)

    @cute.jit
    def store_state_fragment(
        self,
        state_accumulator: cute.Tensor,
        global_state: cute.Tensor,
        state_rs_mma: cute.TiledMma,
        local_math_tid: cutlass.Int32,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            global_state.element_type,
        )
        tiled_copy = cute.make_tiled_copy_C(copy_atom, state_rs_mma)
        thread_copy = tiled_copy.get_slice(local_math_tid)
        destination = thread_copy.partition_D(_swap_first_two_modes(global_state))
        cute.copy(tiled_copy, state_accumulator, destination)

    @cute.jit
    def tail_tensormap_gmem_ptr(
        self,
        g_tensormaps: cute.Tensor,
        workspace_slot: cutlass.Int32,
    ):
        manager = TensorMapManager(TensorMapUpdateMode.GMEM, TENSORMAP_BYTES)
        return manager.get_tensormap_ptr(g_tensormaps.iterator + workspace_slot * cutlass.Int32(TENSORMAP_BYTES))

    @cute.jit
    def tail_tensormap_generic_ptr(
        self,
        g_tensormaps: cute.Tensor,
        workspace_slot: cutlass.Int32,
    ):
        manager = TensorMapManager(TensorMapUpdateMode.GMEM, TENSORMAP_BYTES)
        return manager.get_tensormap_ptr(
            g_tensormaps.iterator + workspace_slot * cutlass.Int32(TENSORMAP_BYTES),
            address_space=_cute_ir.AddressSpace.generic,
        )

    @cute.jit
    def create_tail_tensormap(
        self,
        o_tma_atom: cute.CopyAtom,
        g_tensormaps: cute.Tensor,
        workspace_slot: cutlass.Int32,
        sequence_end: cutlass.Int32,
    ):
        tail_ptr = self.tail_tensormap_gmem_ptr(g_tensormaps, workspace_slot)
        with cute.arch.elect_one():
            cpasync.copy_tensormap(o_tma_atom, tail_ptr)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            _tensormap_replace_global_dim_1(tail_ptr, sequence_end)
        cute.arch.sync_warp()
        cpasync.fence_tma_desc_release()

    @cute.jit
    def run_epilogue_store(
        self,
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        sequence_length: cutlass.Int32,
        sequence_bos: cutlass.Int32,
        num_chunks: cutlass.Int32,
        wait_state,
        release_state,
        o_pipeline,
        tma_store_pipeline,
        o_epilogue_barrier: pipeline.NamedBarrier,
        o_tma_atom: cute.CopyAtom,
        o_tma_tensor: cute.Tensor,
        head_idx: cutlass.Int32,
        batch_idx: cutlass.Int32,
        sequence_idx: cutlass.Int32,
        tensormap_workspace_slot: cutlass.Int32,
        g_tensormaps: cute.Tensor,
        s_o: cute.Tensor,
    ):
        output_head = o_tma_tensor[(None, None, (head_idx, batch_idx))]
        needs_tail_tensormap = False
        if cutlass.const_expr(self.is_varlen):
            needs_tail_tensormap = sequence_idx < cutlass.Int32(self.num_sequences - 1) and sequence_length % cutlass.Int32(
                64
            ) != cutlass.Int32(0)
            if needs_tail_tensormap:
                self.create_tail_tensormap(
                    o_tma_atom,
                    g_tensormaps,
                    tensormap_workspace_slot,
                    sequence_bos + sequence_length,
                )
        for chunk in cutlass.range(num_chunks, unroll=1):
            valid_tokens = sequence_length - chunk * cutlass.Int32(64)
            if chunk < num_chunks - cutlass.Int32(1):
                valid_tokens = cutlass.Int32(64)
            use_tail_tensormap = needs_tail_tensormap and valid_tokens != cutlass.Int32(64)
            if warp_idx == self.store_warp:
                tma_store_pipeline.producer_acquire()
            o_epilogue_barrier.sync()
            if chunk >= O_STAGES:
                o_pipeline.consumer_release(release_state)
                release_state.advance()

            o_pipeline.consumer_wait(wait_state)
            cute.arch.fence_view_async_shared()
            o_epilogue_barrier.sync()
            if warp_idx == self.store_warp:
                output_tile = cute.domain_offset(
                    (
                        cutlass.Int32(0),
                        sequence_bos + chunk * cutlass.Int32(64),
                    ),
                    output_head,
                )
                output_tile = cute.zipped_divide(output_tile, O_SHAPE)[
                    (
                        (None, None),
                        (cutlass.Int32(0), cutlass.Int32(0)),
                    )
                ]
                o_smem, o_gmem = cpasync.tma_partition(
                    o_tma_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(s_o[(None, None, wait_state.index)], 0, 2),
                    cute.group_modes(output_tile, 0, 2),
                )
                if use_tail_tensormap:
                    tail_gmem_ptr = self.tail_tensormap_gmem_ptr(
                        g_tensormaps,
                        tensormap_workspace_slot,
                    )
                    tail_generic_ptr = self.tail_tensormap_generic_ptr(
                        g_tensormaps,
                        tensormap_workspace_slot,
                    )
                    cpasync.fence_tma_desc_acquire(tail_gmem_ptr)
                    cute.copy(
                        o_tma_atom,
                        o_smem,
                        o_gmem,
                        tma_desc_ptr=tail_generic_ptr,
                    )
                else:
                    cute.copy(o_tma_atom, o_smem, o_gmem)
                tma_store_pipeline.producer_commit()
            o_epilogue_barrier.sync()
            wait_state.advance()

        if warp_idx == self.store_warp:
            tma_store_pipeline.producer_tail()
        o_epilogue_barrier.sync()
        remaining_stages = num_chunks
        if remaining_stages > cutlass.Int32(O_STAGES):
            remaining_stages = cutlass.Int32(O_STAGES)
        for _ in cutlass.range(remaining_stages, unroll=1):
            o_pipeline.consumer_release(release_state)
            release_state.advance()
        return wait_state, release_state
