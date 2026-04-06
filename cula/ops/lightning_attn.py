# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Linear Attention with Headwise Exponential Decay using CuTe DSL

This module implements chunkwise linear attention WITH per-head exponential decay factors
for the NVIDIA Blackwell SM100 architecture using CUTE DSL.

The implementation supports:
- Chunkwise computation for improved GPU utilization
- Per-head exponential decay coefficients (λ_h = exp(-s_h)) for temporal modeling
- Efficient state accumulation with decay across chunks
- Position-aware decay masking for intra-chunk attention
- Input/output format: [Batch, Sequence, Heads, Dim]

Mathematical formulation:

For each head h with decay parameter s_h:
- λ_h = exp(-s_h) where s_h > 0
- Intra-chunk: O_intra = (QK^T ⊙ D) · V where D_ij = exp(-s·(i-j)) for i≥j
- Inter-chunk state: S_i = λ^C · S_{i-1} + K_i^T V_i
- Inter-chunk output: O_inter = (Q · S) ⊙ exp(-s·offset)
- Final output: O = O_intra + O_inter
"""

import argparse
import time

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cutlass_dsl import T as _T

from cula.utils import USE_FAST_MATH, assert_blackwell


@cutlass.dsl_user_op
def _atomic_add_global_i32(ptr_i64, addend_i32, *, loc=None, ip=None):
    """Atomic add on global memory int32. Returns old value."""
    result = _llvm.inline_asm(
        _T.i32(),
        [ptr_i64, addend_i32],
        "atom.global.add.s32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


PRINT_DEBUG = False


class MaskEnum:
    """Enumeration for different mask types."""

    NONE = 0
    PADDING = 1
    CAUSAL = 2


class LinearAttentionChunkwiseDecay:
    """
    Chunkwise Linear Attention with Per-Head Exponential Decay using CuTe DSL

    Implements the Lightning Attention algorithm with headwise exponential decay factors.
    Decomposes attention into intra-chunk (local) and inter-chunk (global) components,
    applying exponential decay to both components.

    Args:
        chunk_size: Size of each attention chunk (default: 64)
        acc_dtype: Accumulator data type for all MMA computations (default: Float32)
        io_dtype: Input/output data type (default: BFloat16)
        H: Number of attention heads
        K: Key head dimension (must be 128)
        V: Value head dimension (must be 128)
        scale: Scaling factor for queries
    """

    def __init__(
        self,
        chunk_size: int = 64,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        has_initial_state: bool = False,
        output_final_state: bool = False,
        H: int = 64,
        K: int = 128,
        V: int = 128,
        scale: float = 1.0,
        is_varlen: bool = False,
        persistent: bool = True,
        use_fast_math: bool = True,
    ):
        assert K == 128 and V == 128, f"K and V must both be 128, got K={K}, V={V}"
        assert_blackwell()
        self.use_fast_math = use_fast_math
        self.chunk_size = chunk_size
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.is_varlen = is_varlen
        # Persistent kernel only meaningful for varlen
        self.persistent = persistent if is_varlen else False
        # For varlen mode, always enable state support (INPLACE_UPDATE)
        if is_varlen:
            self.has_initial_state = True
            self.output_final_state = True
        else:
            self.has_initial_state = has_initial_state
            self.output_final_state = output_final_state
        self.H = H
        self.K = K
        self.V = V
        self.D = K  # Internal shorthand: K == V == D
        self.scale = scale

        # Warp specialization
        self.num_load_warps = 1
        self.num_compute_warps = 4
        self.num_correction_warps = 4
        self.threads_per_warp = 32

        # MMA tile shapes
        # C: chunk_size, D: head_dim — both compile-time constants
        C = chunk_size
        D = self.D
        # (C, C, D)
        self.qk_mma_tiler = (C, C, D)  # (M, N, K)
        # (D, C, C)
        self.vp_mma_tiler = (D, C, C)  # (M, N, K)
        # (D, D, C)
        self.kv_mma_tiler = (D, D, C)  # (M, N, K)
        # (D, C, D)
        # State as operand A since it's in TMEM
        # Q now as operand B
        self.sq_mma_tiler = (D, C, D)  # (M, N, K)

        # one-cta cluster shape
        self.cluster_shape_mnk = (1, 1, 1)
        # For masking & decay.
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.epilogue_warp_id = 6
        self.empty_warp_id = 7

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cuda_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.empty_warp_id,
            )
        )

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )

        # NamedBarrier for synchronizing CUDA core warps only (128 threads)
        # Used for LUT init cross-warp visibility and other CUDA-only sync points
        self.cuda_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len(self.cuda_warp_ids),  # 128
        )

        self.buffer_align_bytes = 1024

    @staticmethod
    def _plan_tmem_offsets(
        tiled_mma_qk,
        tile_shape_mnk_qk,
        tiled_mma_pv,
        tile_shape_mnk_pv,
        tiled_mma_kv,
        tile_shape_mnk_kv,
        tiled_mma_sq,
        tile_shape_mnk_sq,
        acc_stages,
        kv_stages=1,
    ):
        """Compute TMEM offsets for various tensors used in the kernel."""
        SM100_TMEM_CAPACITY_COLS = 512
        BITS_PER_TMEM_COL = 32

        # (MMA, MMA_M, MMA_N)
        acc_shape_qk = tiled_mma_qk.partition_shape_C(tile_shape_mnk_qk[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAccQK_fake = tiled_mma_qk.make_fragment_C(cute.append(acc_shape_qk, acc_stages))
        tCtAccQK_fake2 = tiled_mma_qk.make_fragment_C(cute.append(acc_shape_qk, 1))
        num_qk_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccQK_fake)
        num_qk_acc_cols2 = tcgen05.find_tmem_tensor_col_offset(tCtAccQK_fake2)
        # NOTE: 64dp makes the datapath utilization halved
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tCtAccQK_fake={tCtAccQK_fake}, num_qk_acc_cols={num_qk_acc_cols}, num_qk_acc_cols2={num_qk_acc_cols2}")

        acc_shape_pv = tiled_mma_pv.partition_shape_C(tile_shape_mnk_pv[:2])
        tCtAccPV_fake = tiled_mma_pv.make_fragment_C(cute.append(acc_shape_pv, acc_stages))
        num_pv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccPV_fake)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tCtAccPV_fake={tCtAccPV_fake}, num_pv_acc_cols={num_pv_acc_cols}")

        # KV state with configurable stages for pipeline optimization
        acc_shape_kv = tiled_mma_kv.partition_shape_C(tile_shape_mnk_kv[:2])
        tCtAccKV_fake = tiled_mma_kv.make_fragment_C(cute.append(acc_shape_kv, kv_stages))
        num_kv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccKV_fake)
        # KV16 needs separate allocation (cannot trivially reuse KV due to layout differences)
        # BF16 has half columns of FP32
        num_kv16_acc_cols = num_kv_acc_cols // 2
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tCtAccKV_fake={tCtAccKV_fake}, num_kv_acc_cols={num_kv_acc_cols}, num_kv16_acc_cols={num_kv16_acc_cols}")

        acc_shape_sq = tiled_mma_sq.partition_shape_C(tile_shape_mnk_sq[:2])
        # No Stage for QS since state has no stages.
        tCtAccSQ_fake = tiled_mma_sq.make_fragment_C(cute.append(acc_shape_sq, 1))
        num_sq_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccSQ_fake)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tCtAccSQ_fake={tCtAccSQ_fake}, num_sq_acc_cols={num_sq_acc_cols}")

        num_qk_acc_cols_offset = 0
        num_pv_acc_cols_offset = num_qk_acc_cols_offset + num_qk_acc_cols
        num_kv_acc_cols_offset = num_pv_acc_cols_offset + num_pv_acc_cols
        num_kv16_acc_cols_offset = num_kv_acc_cols_offset + num_kv_acc_cols
        num_qs_acc_cols_offset = num_kv16_acc_cols_offset + num_kv16_acc_cols

        num_tmem_cols_total_tmp = num_qs_acc_cols_offset + num_sq_acc_cols
        # Turn num_tmem_cols_total to the nearest power of 2
        num_tmem_cols_total = 1
        while num_tmem_cols_total < num_tmem_cols_total_tmp:
            num_tmem_cols_total *= 2
        assert num_tmem_cols_total <= SM100_TMEM_CAPACITY_COLS

        if cutlass.const_expr(PRINT_DEBUG):
            # Always print TMEM allocation details for capacity analysis
            print("=" * 80)
            print("TMEM Allocation Details:")
            print(f"  QK acc:      {num_qk_acc_cols:4d} cols @ offset {num_qk_acc_cols_offset:4d} (stages={acc_stages})")
            print(f"  PV acc:      {num_pv_acc_cols:4d} cols @ offset {num_pv_acc_cols_offset:4d} (stages={acc_stages})")
            print(f"  KV acc:      {num_kv_acc_cols:4d} cols @ offset {num_kv_acc_cols_offset:4d} (stages={kv_stages})")
            print(f"  KV16:        {num_kv16_acc_cols:4d} cols @ offset {num_kv16_acc_cols_offset:4d} (stages={kv_stages})")
            print(f"  SQ acc:      {num_sq_acc_cols:4d} cols @ offset {num_qs_acc_cols_offset:4d} (stages=1)")
            print("  ---")
            print(f"  Total (raw): {num_tmem_cols_total_tmp:4d} cols")
            print(f"  Total (pow2):{num_tmem_cols_total:4d} cols")
            print(f"  Capacity:    {SM100_TMEM_CAPACITY_COLS:4d} cols")
            print(f"  Usage:       {num_tmem_cols_total / SM100_TMEM_CAPACITY_COLS * 100:5.1f}%")
            print(f"  Margin:      {SM100_TMEM_CAPACITY_COLS - num_tmem_cols_total:4d} cols available")
            print(
                f"  Size (KB):   {num_tmem_cols_total * BITS_PER_TMEM_COL / 8 / 1024:.1f} KB / {SM100_TMEM_CAPACITY_COLS * BITS_PER_TMEM_COL / 8 / 1024:.1f} KB"
            )
            print("=" * 80)

        return (
            num_qk_acc_cols_offset,
            num_pv_acc_cols_offset,
            num_kv_acc_cols_offset,
            num_kv16_acc_cols_offset,
            num_qs_acc_cols_offset,
            num_tmem_cols_total,
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the linear attention kernel."""
        self.q_stage = 2
        self.k_stage = 2
        self.v_stage = 2
        self.o_stage = 2
        self.epi_stage = 2
        self.acc_stage = 2
        self.o_inter_stage = 1
        self.o_intra_stage = 2
        self.kv_stage = 1  # Keep at 1 for now - KV16 reuse needs careful layout handling

    def _compute_grid(
        self,
        o_shape: cute.Shape,
        chunk_size: int,
    ) -> cute.Shape:
        """Compute tile scheduler parameters based on the chunk size and MMA tiler."""
        # (D, S, (H, B))
        return (
            1,
            # H
            cute.size(o_shape[2][0]),
            # B
            cute.size(o_shape[2][1]),
        )

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        v_in: cute.Tensor,
        o_in: cute.Tensor,
        decay_in: cute.Tensor,
        initial_state_in: cute.Tensor,
        final_state_in: cute.Tensor,
        cu_seqlens_in: cute.Tensor,  # [N+1] int32, varlen only (None otherwise)
        initial_state_indices_in: cute.Tensor,  # [N] int32, varlen only (None otherwise)
        o_tensor_in: cute.Tensor,  # Output tensor for varlen CopyUniversal tail store
        workspace_in: cute.Tensor,  # Workspace for persistent kernel atomic counter
        problem_size: tuple[Int32, Int32],  # (N, T) for varlen, (B, S) for non-varlen
        stream,  # CUstream type annotation removed to avoid import issues
    ):
        """
        Execute the Chunkwise Linear Attention operation on the provided tensors.

        With --enable-tvm-ffi, all tensor args accept torch.Tensor directly
        (zero-copy C-level dlpack). Pass None for initial_state_in / final_state_in
        when has_initial_state / output_final_state is False.

        scale, H, D are compile-time constants stored in self.__init__.

        Args:
            q_in: Query tensor [B, S, H, D] or [1, T, H, D] for varlen
            k_in: Key tensor [B, S, H, D] or [1, T, H, D] for varlen
            v_in: Value tensor [B, S, H, D] or [1, T, H, D] for varlen
            o_in: Output tensor [B, S, H, D] or [1, T, H, D] for varlen
            decay_in: Per-head decay tensor [H] (FP32)
            initial_state_in: Initial state [B, H, D, D] or state pool [pool, H, D, D] (FP32)
            final_state_in: Final state [B, H, D, D] (FP32) or None (varlen uses INPLACE_UPDATE)
            cu_seqlens_in: [N+1] int32 cumulative sequence lengths (varlen only)
            initial_state_indices_in: [N] int32 indices into state pool (varlen only)
            problem_size: (N, T) for varlen or (B, S) dynamic problem dimensions
            stream: CUDA stream
        """
        B, S = problem_size
        H = self.H
        D = self.D

        # Setup attributes
        self._setup_attributes()

        self.cta_group = tcgen05.CtaGroup.ONE

        # It's ok since torch tensor is row major, hence we've layout=(B,S,H,D):(DHS, DH, D, 1).
        # Below are just permutation tricks to ease the later processing.
        # For varlen: input is [1, T, H, D] → view as (T, D, H) with stride (D*H, 1, D)
        # For non-varlen: input is [B, S, H, D] → view as (S, D, (H,B))
        if cutlass.const_expr(self.is_varlen):
            # Varlen: B=N (num_seqs), S=T (total_tokens), no batch stride
            q_layout = cute.make_layout((S, D, H), stride=(D * H, 1, D))
            k_layout = cute.make_layout((S, D, H), stride=(D * H, 1, D))
            v_layout = cute.make_layout((D, S, H), stride=(1, D * H, D))
            o_layout = cute.make_layout((D, S, H), stride=(1, D * H, D))
        else:
            q_layout = cute.make_layout(
                (S, D, (H, B)),
                stride=(D * H, 1, (D, D * H * S)),
            )
            k_layout = cute.make_layout(
                (S, D, (H, B)),
                stride=(D * H, 1, (D, D * H * S)),
            )
            v_layout = cute.make_layout(
                (D, S, (H, B)),
                stride=(1, D * H, (D, D * H * S)),
            )
            o_layout = cute.make_layout(
                (D, S, (H, B)),
                stride=(1, D * H, (D, D * H * S)),
            )
        q = cute.make_tensor(q_in.iterator, q_layout)
        k = cute.make_tensor(k_in.iterator, k_layout)
        v = cute.make_tensor(v_in.iterator, v_layout)
        o = cute.make_tensor(o_in.iterator, o_layout)

        # Initial state / final state: [B, H, D, D] stored as row-major FP32
        # When has_initial_state / output_final_state is False, None is passed
        # and the parameter is eliminated at compile time via const_expr guards.
        # For varlen: state pool is [pool_size, H, D, D]. We use B (=N) as the
        # pool dimension — strides are correct regardless of actual pool_size.
        fstate_layout = cute.make_layout((D, D, (H, B)),
                                         stride=(D, 1, (D * D, D * D * H)),
                                         )
        if cutlass.const_expr(self.has_initial_state):
            initial_state = cute.make_tensor(initial_state_in.iterator, fstate_layout)
        else:
            initial_state = initial_state_in
        if cutlass.const_expr(self.is_varlen):
            # Varlen INPLACE_UPDATE: final_state writes back to initial_state pool
            final_state = initial_state
        elif cutlass.const_expr(self.output_final_state):
            final_state = cute.make_tensor(final_state_in.iterator, fstate_layout)
        else:
            final_state = final_state_in

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.k_major_mode_kv = tcgen05.OperandMajorMode.MN  # For V^T*K, S dimension coalesced
        # TMEM register output results as (D, C)
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.o_layout != utils.LayoutEnum.COL_MAJOR):
            raise RuntimeError("The layout of o is not supported")
        if cutlass.const_expr(self.k_major_mode == self.k_major_mode_kv):
            raise RuntimeError("The layout of k & k^t should be different")

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.qk_mma_tiler[:2],
        )
        # V^T*K, majorness
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.v_major_mode,
            self.k_major_mode_kv,
            self.acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
        )
        # State^T Q^T
        sq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            # State is in TMEM, always K major, TODO
            tcgen05.OperandMajorMode.K,
            self.q_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.sq_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        p_major_mode = tcgen05.OperandMajorMode.K
        vp_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_major_mode,
            p_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.vp_mma_tiler[:2],
        )

        (
            self.tmem_qk_cols_offset,
            self.tmem_pv_cols_offset,
            self.tmem_kv_cols_offset,
            self.tmem_kv16_cols_offset,
            self.tmem_sq_cols_offset,
            self.tmem_total_cols,
        ) = self._plan_tmem_offsets(
            qk_tiled_mma,
            self.qk_mma_tiler,
            vp_tiled_mma,
            self.vp_mma_tiler,
            kv_tiled_mma,
            self.kv_mma_tiler,
            sq_tiled_mma,
            self.sq_mma_tiler,
            # Try double buffer
            self.acc_stage,
            self.kv_stage,
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        # Output shape, (D, C)

        # TODO: check transpose here
        self.epi_tile = (self.vp_mma_tiler[0], self.vp_mma_tiler[1])  # (D, S)
        self.qk_epi_tile = (self.qk_mma_tiler[0], self.qk_mma_tiler[1])  # qk

        # Q&K^T
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        # V^T*K
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(
            vp_tiled_mma,
            self.vp_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )
        kv_k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        # Single-stage layout for separate sK_weighted buffer (used by KV GEMM)
        kv_k_smem_layout_single = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.k_dtype,
            num_stages=1,
        )
        # V^T*P
        p_smem_layout_staged = sm100_utils.make_smem_layout_b(
            vp_tiled_mma,
            self.vp_mma_tiler,
            self.v_dtype,
            self.acc_stage,
        )
        state_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            sq_tiled_mma,
            self.sq_mma_tiler,
            self.q_dtype,
            num_stages=1,
        )
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.acc_stage,
        )

        # TMA operations
        # TODO: multicast check, (1,1,1) cluster indicates no multicast
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # TMA load for Q
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            v,
            v_smem_layout,
            self.vp_mma_tiler,
            vp_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_smem_layout,
            self.epi_tile,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_k_bytes = k_copy_size
        self.tma_copy_v_bytes = k_copy_size

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"q_layout: {cute.pretty_str(q_layout)}")
            print(f"q: {cute.pretty_str(q)}")
            print(f"k_layout: {cute.pretty_str(k_layout)}")
            print(f"k: {cute.pretty_str(k)}")
            print(f"v_layout: {cute.pretty_str(v_layout)}")
            print(f"v: {cute.pretty_str(v)}")
            print(f"o_layout: {cute.pretty_str(o_layout)}")
            print(f"o: {cute.pretty_str(o)}")
            print(f"qk_tiled_mma: {cute.pretty_str(qk_tiled_mma)}")
            print(f"kv_tiled_mma: {cute.pretty_str(kv_tiled_mma)}")
            print(f"vp_tiled_mma: {cute.pretty_str(vp_tiled_mma)}")
            print(f"sq_tiled_mma: {cute.pretty_str(sq_tiled_mma)}")
            print(f"cluster_layout_vmnk: {cute.pretty_str(cluster_layout_vmnk)}")
            print(f"epi_tile: {cute.pretty_str(self.epi_tile)}")
            print(f"q_smem_layout: {cute.pretty_str(q_smem_layout)}")
            print(f"k_smem_layout: {cute.pretty_str(k_smem_layout)}")
            print(f"v_smem_layout: {cute.pretty_str(v_smem_layout)}")
            print(f"q_smem_layout_staged: {cute.pretty_str(q_smem_layout_staged)}")
            print(f"k_smem_layout_staged: {cute.pretty_str(k_smem_layout_staged)}")
            print(f"k_smem_layout_staged.swzzle: {cute.pretty_str(k_smem_layout_staged.inner)}")
            print(f"k_smem_layout_staged.outer: {cute.pretty_str(k_smem_layout_staged.outer)}")
            print(f"kv_k_smem_layout_staged: {cute.pretty_str(kv_k_smem_layout_staged)}")
            print(f"kv_k_smem_layout_staged.swzzle: {cute.pretty_str(kv_k_smem_layout_staged.inner)}")
            print(f"kv_k_smem_layout_staged.outer: {cute.pretty_str(kv_k_smem_layout_staged.outer)}")
            print(f"v_smem_layout_staged: {cute.pretty_str(v_smem_layout_staged)}")
            print(f"o_smem_layout_staged: {cute.pretty_str(o_smem_layout_staged)}")
            print(f"p_smem_layout_staged: {cute.pretty_str(p_smem_layout_staged)}")
            print(f"tma_atom_q: {cute.pretty_str(tma_atom_q)}")
            print(f"tma_atom_k: {cute.pretty_str(tma_atom_k)}")
            print(f"tma_atom_v: {cute.pretty_str(tma_atom_v)}")
            print(f"tma_tensor_q: {cute.pretty_str(tma_tensor_q)}")
            print(f"tma_tensor_k: {cute.pretty_str(tma_tensor_k)}")
            print(f"tma_tensor_v: {cute.pretty_str(tma_tensor_v)}")

            print(f"tma_atom_o: {cute.pretty_str(tma_atom_o)}")
            print(f"o_smem_layout: {cute.pretty_str(o_smem_layout)}")
            print(f"tma_tensor_o: {cute.pretty_str(tma_tensor_o)}")

            print(f"q_copy_size: {q_copy_size}")
            print(f"k_copy_size: {k_copy_size}")
            print(f"v_copy_size: {v_copy_size}")

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            # Inputs
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]  # type: ignore
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]  # type: ignore
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # type: ignore
            # Masking
            s_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            # KV
            kv_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            kv16_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            p_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            o_intra_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            o_inter_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
            smem_o_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            k_weighted_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(v_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Store QK
            sP: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(p_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Decay lookup table: exp(-s*k) for k=0..C-1 (64 FP32 = 256 bytes)
            sDecayLUT: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.chunk_size],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Separate weighted K buffer for KV GEMM (single stage)
            # Allows K weighting to run in parallel with QK GEMM
            sK_weighted: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(kv_k_smem_layout_single)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Double-buffered work index for persistent dynamic scheduling
            sWorkIdx: cute.struct.MemRange[Int32, 2]
            # Double-buffered scheduling mbarriers (count=1 each, Load warp elect_one arrives)
            sched_mbar: cute.struct.MemRange[Int64, 2]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(self.is_varlen and self.persistent):
            import torch as _torch

            sm_count = _torch.cuda.get_device_properties(0).multi_processor_count
            self.grid = (sm_count, 1, 1)
        elif cutlass.const_expr(self.is_varlen):
            # Varlen grid: (1, H, N) where B = N = num_sequences
            self.grid = (1, H, B)
        else:
            self.grid = self._compute_grid(
                o_shape=cute.shape(o),
                chunk_size=self.chunk_size,
            )

        # Prepare cu_seqlens and initial_state_indices for kernel
        if cutlass.const_expr(self.is_varlen):
            cu_seqlens_tensor = cu_seqlens_in
            initial_state_indices_tensor = initial_state_indices_in
        else:
            cu_seqlens_tensor = cu_seqlens_in
            initial_state_indices_tensor = initial_state_indices_in

        self.kernel(
            qk_tiled_mma,
            kv_tiled_mma,
            vp_tiled_mma,
            sq_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            decay_in.iterator,
            initial_state,
            final_state,
            cu_seqlens_tensor,
            initial_state_indices_tensor,
            o_tensor_in,
            workspace_in.iterator,
            q_smem_layout_staged,
            k_smem_layout_staged,
            kv_k_smem_layout_staged,
            kv_k_smem_layout_single,
            v_smem_layout_staged,
            o_smem_layout_staged,
            p_smem_layout_staged,
            state_tmem_layout_staged,
            problem_size,
        ).launch(
            grid=self.grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        vp_tiled_mma: cute.TiledMma,
        sq_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        decay: cute.Pointer,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        cu_seqlens: cute.Tensor,  # [N+1] int32, varlen only
        initial_state_indices: cute.Tensor,  # [N] int32, varlen only
        o_tensor: cute.Tensor,  # Output tensor for varlen CopyUniversal tail store
        workspace_iter: cute.Pointer,  # Workspace for persistent kernel atomic counter
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        kv_k_smem_layout_staged: cute.ComposedLayout,
        kv_k_smem_layout_single: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        state_tmem_layout_staged: cute.ComposedLayout,
        problem_size: tuple[Int32, Int32],  # (B, S)
    ):
        """Kernel for chunkwise linear attention with per-position decay."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # Allocate shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.s_mbar_ptr.data_ptr(),
        ).make_participants()
        # Notify cuda core to convert 32-bit accumulator to 16-bit
        kv_producer, kv_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.kv_stage,  # Keep configurable for future optimization
            producer_group=make_thread_cooperative_group(
                len([self.mma_warp_id]),
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.kv_mbar_ptr.data_ptr(),
        ).make_participants()
        # Notify mma warp that 16bit state is ready for mma as operand A
        kv16_producer, kv16_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.kv_stage,  # Keep configurable for future optimization
            producer_group=make_thread_cooperative_group(
                len(self.cuda_warp_ids),
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len([self.mma_warp_id])),
            barrier_storage=storage.kv16_mbar_ptr.data_ptr(),
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len([self.mma_warp_id])),
            barrier_storage=storage.p_mbar_ptr.data_ptr(),
        ).make_participants()
        o_intra_producer, o_intra_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.o_intra_mbar_ptr.data_ptr(),
        ).make_participants()
        o_inter_producer, o_inter_consumer = pipeline.PipelineUmmaAsync.create(
            # NO STAGE for Q*STATE
            num_stages=1,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.o_inter_mbar_ptr.data_ptr(),
        ).make_participants()
        smem_o_producer, smem_o_consumer = pipeline.PipelineAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len([self.epilogue_warp_id])),
            barrier_storage=storage.smem_o_mbar_ptr.data_ptr(),
        ).make_participants()
        # K weighted signal: CUDA core weights K in SMEM, MMA waits before KV GEMM
        k_weighted_producer, k_weighted_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len([self.mma_warp_id])),
            barrier_storage=storage.k_weighted_mbar_ptr.data_ptr(),
        ).make_participants()
        # (state_load pipeline removed — h0/ht now use direct GMEM↔RMEM)

        # TMEM
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total_cols)

        # Barrier before retrieve tensor memory ptr from shared memory
        tmem.wait_for_alloc()

        # Retrieve tmem ptr
        tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

        # Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, STAGE_Q)
        # sQ: ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)>
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        q_as_b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            sq_tiled_mma,
            self.sq_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        sQ_sq = storage.sQ.get_tensor(q_as_b_smem_layout_staged.outer, swizzle=q_as_b_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, STAGE_K)
        # sK: tensor<ptr<bf16, smem, align<1024>, S<3,4,3>> o
        # ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)>
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sK_kv = storage.sK.get_tensor(kv_k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        # Separate weighted K buffer for KV GEMM (allows parallel K weighting)
        sK_weighted = storage.sK_weighted.get_tensor(kv_k_smem_layout_single.outer, swizzle=kv_k_smem_layout_single.inner)
        # Decay lookup table
        sDecayLUT = cute.make_tensor(storage.sDecayLUT.data_ptr(), cute.make_layout(self.chunk_size))
        # (((64,2),16),1,4,2):(((1,4096),64),0,1024,8192)>
        sV = storage.sV.get_tensor(v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sP = storage.sP.get_tensor(p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE_O)
        sO = storage.sO.get_tensor(o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)

        qk_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.qk_mma_tiler[:2],
            self.acc_stage,
        )
        sQK = storage.sP.get_tensor(
            qk_smem_layout_staged.outer,
            swizzle=qk_smem_layout_staged.inner,
        )

        # CopyUniversal epilogue SMEM layout for varlen tail O store
        # (Must be outside warp if/elif to avoid SharedStorage serialization)
        if cutlass.const_expr(self.is_varlen):
            o_epi_staged = sm100_utils.make_smem_layout_epi(
                self.io_dtype,
                self.o_layout,
                (self.D, self.chunk_size),
                self.acc_stage,
            )
            sO_epi = storage.sO.get_tensor(o_epi_staged.outer, swizzle=o_epi_staged.inner)

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"sQ: {cute.pretty_str(sQ)}")
            print(f"sK: {cute.pretty_str(sK)}")
            print(f"sK_kv: {cute.pretty_str(sK_kv)}")
            print(f"sV: {cute.pretty_str(sV)}")
            print(f"sO: {cute.pretty_str(sO)}")
            print(f"sP: {cute.pretty_str(sP)}")
            print(f"sQK: {cute.pretty_str(sQK)}")

        self.num_regs_other = 32
        self.num_regs_cuda = 256

        B, S = problem_size
        H = self.H
        D = self.D
        C = self.chunk_size
        scale = cutlass.Float32(self.scale)

        # ===================== Block indices =====================
        if cutlass.const_expr(self.is_varlen):
            if cutlass.const_expr(self.persistent):
                # 1D grid work decode: persistent (grid=SM_count)
                total_work_units = H * B
                num_iters = Int32(0)  # not used, while loop controls iteration
                # Pre-initialize variables reassigned inside persistent loop (CuTe DSL requirement)
                hidx = Int32(0)
                bidx = Int32(0)
                bos = Int32(0)
                eos = Int32(0)
                seq_len = Int32(0)
                state_idx = Int32(0)
            else:
                # Non-persistent varlen: 3D grid (1, H, N)
                (_, hidx, bidx) = cute.arch.block_idx()
                bos = cu_seqlens[bidx]
                eos = cu_seqlens[bidx + 1]
                seq_len = eos - bos
                state_idx = initial_state_indices[bidx]
                num_iters = Int32(1)
        else:
            (_, hidx, bidx) = cute.arch.block_idx()
            seq_len = S
            state_idx = bidx
            num_iters = Int32(1)

        # ===================== Scheduling mbarrier init (persistent varlen) =====================
        if cutlass.const_expr(self.is_varlen and self.persistent):
            sched_mbar_base = storage.sched_mbar.data_ptr()
            # Init 2 mbarriers with count=1: only Load warp elect_one arrives
            if warp_idx == 0:
                cute.arch.mbarrier_init(sched_mbar_base, 1)
                cute.arch.mbarrier_init(sched_mbar_base + 1, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier(barrier_id=0, number_of_threads=self.threads_per_cta)

        # Load per-head decay parameter (s_h > 0) to register
        # λ_h = exp(-s_h) is the decay factor
        # For persistent: will be recomputed per-WU inside warp loops
        if cutlass.const_expr(self.is_varlen and self.persistent):
            # Persistent varlen: block_decay computed per-WU inside CUDA warp loop.
            # Pre-initialize here so DSL sees it defined before warp if/elif blocks.
            block_decay = Float32(0.0)
        else:
            # Non-varlen and non-persistent varlen: hidx known at CTA start
            decay_tensor = cute.make_tensor(decay, cute.make_layout(H))
            decay_s = decay_tensor[hidx]
            # Block-level decay: λ^C for inter-chunk state accumulation
            block_decay = cute.exp(-decay_s * cutlass.Float32(C), fastmath=self.use_fast_math)

        # Make fragments/tmem for QK MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCrQ, tCrK, tCtAccQK = self.mma_partition_ss(
            qk_tiled_mma,
            self.qk_mma_tiler,
            sQ,
            sK,
            tmem_ptr_base + self.tmem_qk_cols_offset,
            self.acc_stage,
        )

        # Make fragments/tmem for KV MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        # Use sK_weighted (separate buffer) so K weighting can run in parallel with QK GEMM
        tCrV, tCrK_kv, tCtAccKV = self.mma_partition_ss(
            kv_tiled_mma,
            self.kv_mma_tiler,
            sV,
            sK_weighted,  # Changed from sK_kv: read from separate weighted K buffer
            tmem_ptr_base + self.tmem_kv_cols_offset,
            1,  # acc_stages=1 for sK_weighted (single stage)
        )

        # Make fragments/tmem for SQ MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCrState, tCrQ_sq, tCtAccSQ = self.mma_partition_ts(
            tiled_mma=sq_tiled_mma,
            tile_shape_mnk=self.sq_mma_tiler,
            a_tmem_layout=state_tmem_layout_staged,
            smem_b=sQ_sq,
            tmem_a_ptr=tmem_ptr_base + self.tmem_kv16_cols_offset,
            tmem_acc_ptr=tmem_ptr_base + self.tmem_sq_cols_offset,
            acc_stages=1,
        )

        ############################################
        kv_mma_tiler2 = (self.kv_mma_tiler[0], self.kv_mma_tiler[1] // 2, self.kv_mma_tiler[2])
        fake_kv_tiled_mma_acc32 = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.v_major_mode,
            self.k_major_mode_kv,
            self.acc_dtype,
            self.cta_group,
            kv_mma_tiler2[:2],
        )
        tCtStateAsF32 = self.mma_partition_c(
            fake_kv_tiled_mma_acc32, kv_mma_tiler2, tmem_ptr_base + self.tmem_kv16_cols_offset, 1
        )
        ############################################

        # Make fragments/tmem for VP MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCrV_dup, tCrP, tCtAccPV = self.mma_partition_ss(
            vp_tiled_mma,
            self.vp_mma_tiler,
            sV,
            sP,
            tmem_ptr_base + self.tmem_pv_cols_offset,
            self.acc_stage,
        )

        # ========================================================
        # K decay weighting: compile-time S2R/R2S setup
        # (Must be outside warp if/elif to avoid SharedStorage serialization)
        # Use self.chunk_size and self.kv_mma_tiler[0] as compile-time D
        # ========================================================
        _C = self.chunk_size  # compile-time 64
        _D = self.kv_mma_tiler[0]  # compile-time 128
        HALF_D = _D // 2  # 64
        HALF_SMEM_ELEMS = _C * HALF_D  # 64 * 64 = 4096

        # Epilogue-style flat SMEM layout for S2R access
        k_smem_layout_epi = sm100_utils.make_smem_layout_epi(
            self.k_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (_C, _D),
            self.k_stage,
        )
        k_smem_layout_coalesce = cute.coalesce(
            k_smem_layout_epi,
            target_profile=(1, 1, 1),
        )
        sK_flat_s2r = storage.sK.get_tensor(k_smem_layout_coalesce.outer, swizzle=k_smem_layout_coalesce.inner)

        # Half-size MMA for partitioned S2R: process (_C, HALF_D) per pass
        mma_op_half = cute.nvgpu.warp.MmaF16BF16Op(ab_dtype=self.io_dtype, acc_dtype=self.acc_dtype, shape_mnk=(16, 8, 16))
        k_s2r_tiler_half = (_C, _C, HALF_D)  # (M=64, N=64, K=64)
        tiled_mma_k_half = cute.make_tiled_mma(
            mma_op_half,
            atom_layout_mnk=(4, 1, 1),  # 4 warps
            permutation_mnk=k_s2r_tiler_half,
        )

        # S2R (LdMatrix) and R2S (StMatrix) copy atoms
        copy_op_k_s2r = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
        copy_op_k_r2s = cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4)
        tiled_load_k_half = cute.make_tiled_copy_A(
            cute.make_copy_atom(copy_op_k_s2r, self.io_dtype),
            tiled_mma_k_half,
        )
        tiled_store_k_half = cute.make_tiled_copy_A(
            cute.make_copy_atom(copy_op_k_r2s, self.io_dtype),
            tiled_mma_k_half,
        )

        # Half-width SMEM views split along D
        k_sml_epi_half = sm100_utils.make_smem_layout_epi(
            self.k_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (_C, HALF_D),
            self.k_stage,
        )
        k_sml_half = cute.coalesce(k_sml_epi_half, target_profile=(1, 1, 1))
        # Fix stage stride: half layout has stride _C*HALF_D but actual SMEM uses _C*_D
        k_half_outer = cute.make_layout(
            k_sml_half.outer.shape, stride=(*k_sml_half.outer.stride[:-1], k_sml_half.outer.stride[-1] * 2)
        )
        sK_s2r_h0 = cute.make_tensor(sK_flat_s2r.iterator, layout=k_half_outer)
        sK_s2r_h1 = cute.make_tensor(sK_flat_s2r.iterator + HALF_SMEM_ELEMS, layout=k_half_outer)
        sK_s2r_halves = [sK_s2r_h0, sK_s2r_h1]

        # sK_weighted half views (single stage) for CUDA core R2S writes
        # Use epilogue layout for S2R/R2S compatibility
        k_sml_epi_single = sm100_utils.make_smem_layout_epi(
            self.k_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (_C, HALF_D),
            1,  # single stage
        )
        k_sml_single = cute.coalesce(k_sml_epi_single, target_profile=(1, 1, 1))
        # Create flat view of sK_weighted with epilogue layout
        k_weighted_epi = sm100_utils.make_smem_layout_epi(
            self.k_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (_C, _D),
            1,  # single stage
        )
        k_weighted_coalesce = cute.coalesce(k_weighted_epi, target_profile=(1, 1, 1))
        sK_weighted_flat = storage.sK_weighted.get_tensor(k_weighted_coalesce.outer, swizzle=k_weighted_coalesce.inner)
        # Single-stage half layout (no stage stride adjustment needed)
        sK_weighted_h0 = cute.make_tensor(sK_weighted_flat.iterator, layout=k_sml_single.outer)
        sK_weighted_h1 = cute.make_tensor(sK_weighted_flat.iterator + HALF_SMEM_ELEMS, layout=k_sml_single.outer)
        sK_weighted_halves = [sK_weighted_h0, sK_weighted_h1]

        # Identity tensor for mapping thread elements to (row=position, col=dim)
        k_shape_half = (_C, HALF_D)
        cK_half = cute.make_identity_tensor(k_shape_half)

        # =========================================================================
        # DYNAMIC SCHEDULING: initial work_idx fetch (persistent varlen only)
        # Load warp elect_one does atomicAdd → sWorkIdx[buf] → fence → arrive mbar[buf].
        # Other warps wait on mbar[buf] at the correct phase, then read sWorkIdx[buf].
        # Double-buffered to avoid phase racing.
        # =========================================================================
        if cutlass.const_expr(self.is_varlen and self.persistent):
            sWorkIdx = storage.sWorkIdx.get_tensor(cute.make_layout((2,)))
            if warp_idx == self.load_warp_id:
                with cute.arch.elect_one():
                    first_work_idx = _atomic_add_global_i32(workspace_iter.toint().ir_value(), Int32(1).ir_value())
                    sWorkIdx[(0,)] = first_work_idx
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.mbarrier_arrive(sched_mbar_base)
            else:
                # All other warps: wait for Load warp's signal on mbar[0], phase=0
                cute.arch.mbarrier_wait(sched_mbar_base, Int32(0))

        # ///////////////////////////////////////////////////////////////////////////////
        # LOAD WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            wu_iter = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                # Load warp wrote sWorkIdx[0], read it back
                cute.arch.sync_warp()
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)  # next buffer to write
                should_continue = work_idx < total_work_units
            elif cutlass.const_expr(self.is_varlen):
                work_idx = Int32(0)
                should_continue = wu_iter < num_iters
            else:
                should_continue = True  # non-varlen: single iteration

            while should_continue:
                # --- Work decode (persistent only) ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    hidx = work_idx % H
                    bidx = work_idx // H
                    bos = cu_seqlens[bidx]
                    eos = cu_seqlens[bidx + 1]
                    seq_len = eos - bos

                # For varlen: apply domain_offset to shift TMA tensors by bos
                # so chunk_idx 0 maps to tokens [bos, bos+C), etc.
                # TMA has built-in OOB zeroing for tail chunks.
                if cutlass.const_expr(self.is_varlen):
                    # Q: (S, D, H) → offset S (mode 0) by bos
                    tma_tensor_q_use = cute.domain_offset((bos, 0, 0), tma_tensor_q)
                    # K: (S, D, H) → offset S (mode 0) by bos
                    tma_tensor_k_use = cute.domain_offset((bos, 0, 0), tma_tensor_k)
                    # V: (D, S, H) → offset S (mode 1) by bos
                    tma_tensor_v_use = cute.domain_offset((0, bos, 0), tma_tensor_v)
                else:
                    tma_tensor_q_use = tma_tensor_q
                    tma_tensor_k_use = tma_tensor_k
                    tma_tensor_v_use = tma_tensor_v

                # ((ATOM_V, REST_V), INPUT_STAGE)
                # ((ATOM_V, REST_V), TILES_N, TILES_K)
                tQsQ, tQgQ = self.tma_partition_for_mma_operand(
                    tma_atom_q,
                    tma_tensor_q_use,
                    sQ,
                    self.qk_mma_tiler,
                    qk_tiled_mma,
                    operand_mode="A",
                    hidx=hidx,
                    bidx=bidx,
                    debug_name="Q",
                )

                tKsK, tKgK = self.tma_partition_for_mma_operand(
                    tma_atom_k,
                    tma_tensor_k_use,
                    sK,
                    self.qk_mma_tiler,
                    qk_tiled_mma,
                    operand_mode="B",
                    hidx=hidx,
                    bidx=bidx,
                    debug_name="K",
                )

                tVsV, tVgV = self.tma_partition_for_mma_operand(
                    tma_atom_v,
                    tma_tensor_v_use,
                    sV,
                    self.vp_mma_tiler,
                    vp_tiled_mma,
                    operand_mode="A",
                    hidx=hidx,
                    bidx=bidx,
                    debug_name="V",
                )

                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    # Chunk iterate over TILES_M, TILES_K is 1 in our case since max D is 128
                    # With domain_offset, idx starts from 0 (already offset by bos)
                    idx = chunk_start // C

                    # Qi
                    q_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_q,
                        src=tQgQ[None, idx, 0],
                        dst=tQsQ[None, q_handle.index],
                        tma_bar_ptr=q_handle.barrier,
                    )

                    # Ki
                    k_handle = load_k_producer.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_k,
                        src=tKgK[None, idx, 0],
                        dst=tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )

                    # Vi
                    v_handle = load_v_producer.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_v,
                        src=tVgV[None, 0, idx],
                        dst=tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )

                # --- End-of-WU: Load warp fetches next work_idx and signals ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    # CTA barrier: sync all warps before WU transition
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_per_cta)
                    with cute.arch.elect_one():
                        next_idx = _atomic_add_global_i32(workspace_iter.toint().ir_value(), Int32(1).ir_value())
                        sWorkIdx[(sched_buf,)] = next_idx
                        cute.arch.fence_acq_rel_cta()
                        cute.arch.mbarrier_arrive(sched_mbar_base + sched_buf)
                    cute.arch.sync_warp()
                    work_idx = sWorkIdx[(sched_buf,)]
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                elif cutlass.const_expr(self.is_varlen):
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters
                else:
                    should_continue = False

        # ///////////////////////////////////////////////////////////////////////////////
        # COMPUTE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            wu_iter = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)
                sched_phase0 = Int32(1)
                sched_phase1 = Int32(0)
                should_continue = work_idx < total_work_units
            elif cutlass.const_expr(self.is_varlen):
                work_idx = Int32(0)
                should_continue = wu_iter < num_iters
            else:
                should_continue = True

            while should_continue:
                # --- Work decode (MMA only needs seq_len) ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    bidx_mma = work_idx // H
                    seq_len = cu_seqlens[bidx_mma + 1] - cu_seqlens[bidx_mma]

                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    # Process chunk from chunk_start to chunk_start + chunk_size
                    idx = chunk_start // C

                    # Wait for Qi.
                    q_handle = load_q_consumer.wait_and_advance()

                    if chunk_start != 0 or cutlass.const_expr(self.has_initial_state):
                        kv16_handle = kv16_consumer.wait_and_advance()
                        o_inter_handle = o_inter_producer.acquire_and_advance()

                        # Compute SQ once Qi is ready.
                        sq_tiled_mma = self.exec_mma(
                            tiled_mma=sq_tiled_mma,
                            tCtAcc=tCtAccSQ,
                            tCrA=tCrState,
                            tCrB=tCrQ_sq,
                            a_stage_idx=0,
                            b_stage_idx=q_handle.index,
                            acc_stage_idx=0,
                        )
                        o_inter_handle.commit()
                        kv16_handle.release()

                    # Wait for Ki.
                    k_handle = load_k_consumer.wait_and_advance()
                    # Acquire empty S0 buffer
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # GEMM
                    qk_tiled_mma = self.exec_mma(
                        tiled_mma=qk_tiled_mma,
                        tCtAcc=tCtAccQK,
                        tCrA=tCrQ,
                        tCrB=tCrK,
                        a_stage_idx=q_handle.index,
                        b_stage_idx=k_handle.index,
                        acc_stage_idx=s0_handle.index,
                    )
                    # Release Q.
                    q_handle.release()
                    # Commit S = QK.
                    s0_handle.commit()

                    # Wait for CUDA core to finish weighting K for per-position decay
                    kw_handle = k_weighted_consumer.wait_and_advance()

                    # Wait for V, then execute KV GEMM (moved earlier)
                    v_handle = load_v_consumer.wait_and_advance()
                    kv_handle = kv_producer.acquire_and_advance()

                    # Execute KV GEMM: State = K_weighted^T @ V
                    kv_tiled_mma = self.exec_mma(
                        tiled_mma=kv_tiled_mma,
                        tCtAcc=tCtAccKV,
                        tCrA=tCrV,
                        tCrB=tCrK_kv,
                        a_stage_idx=v_handle.index,
                        b_stage_idx=0,
                        acc_stage_idx=0,
                        always_acc=True if (chunk_start != 0 or cutlass.const_expr(self.has_initial_state)) else False,  # noqa: SIM210 -- Cute DSL: const_expr requires explicit True/False form
                    )
                    kv_handle.commit()
                    k_handle.release()
                    kw_handle.release()

                    # Now wait for P and execute VP GEMM
                    p_handle = p_consumer.wait_and_advance()
                    o_intra_handle = o_intra_producer.acquire_and_advance()

                    # VP GEMM: O_intra = P @ V
                    vp_tiled_mma = self.exec_mma(
                        tiled_mma=vp_tiled_mma,
                        tCtAcc=tCtAccPV,
                        tCrA=tCrV,
                        tCrB=tCrP,
                        a_stage_idx=v_handle.index,
                        b_stage_idx=p_handle.index,
                        acc_stage_idx=o_intra_handle.index,
                    )
                    p_handle.release()
                    o_intra_handle.commit()

                    # Release K V here
                    v_handle.release()

                # --- End-of-WU: wait for Load warp's scheduling signal ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    # CTA barrier: sync all warps before WU transition
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_per_cta)
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                elif cutlass.const_expr(self.is_varlen):
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters
                else:
                    should_continue = False

        # ///////////////////////////////////////////////////////////////////////////////
        # CUDA CORE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.setmaxregister_increase(self.num_regs_cuda)

            # ----------------------------------------------------------
            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))

            # constant mask tensor
            cM = cute.make_identity_tensor(self.qk_mma_tiler[:2])

            # With ACC_STAGE
            # O1
            (
                tiled_copy_t2r_pv,
                tTR_tAcc_base_pv,
                tTR_rAcc_pv,
            ) = self.epilog_tmem_copy_and_partition(tidx, tCtAccPV, self.vp_mma_tiler, use_2cta_instrs=False)

            # ((ATOM_V, REST_V), EPI_M, EPI_N)
            tTR_rO = cute.make_rmem_tensor(tTR_rAcc_pv.shape, self.io_dtype)
            tiled_copy_r2s_o, tRS_rO, tRS_sO = self.epilog_smem_copy_and_partition_o(tiled_copy_t2r_pv, tTR_rO, tidx, sO)

            thr_copy_r2s_o = tiled_copy_r2s_o.get_slice(tidx)

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"thr_copy_r2s_o: {cute.pretty_str(thr_copy_r2s_o)}")
                print(f"sO: {cute.pretty_str(sO)}")
                print(f"tTR_rO: {cute.pretty_str(tTR_rO)}")
                print(f"tRS_rO: {cute.pretty_str(tRS_rO)}")
                print(f"tRS_sO: {cute.pretty_str(tRS_sO)}")

            # O2, i.e. O_INTER
            # SQ: (128, 64), (D, C)
            (
                tiled_copy_t2r_sq,
                tTR_tAcc_base_sq,
                tTR_rAcc_sq,
            ) = self.epilog_tmem_copy_and_partition(tidx, tCtAccSQ, self.sq_mma_tiler, use_2cta_instrs=False)

            # Position identity tensor for SQ output (D, C)
            # Used to apply per-position inter-chunk decay: exp(-s*(pos+1))
            cM_sq = cute.make_identity_tensor(self.sq_mma_tiler[:2])
            thr_copy_t2r_sq_thread = tiled_copy_t2r_sq.get_slice(tidx)
            tTR_cSQ = thr_copy_t2r_sq_thread.partition_D(cM_sq)

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tiled_copy_t2r_pv: {tiled_copy_t2r_pv}")
                print(f"tTR_tAcc_base_pv: {tTR_tAcc_base_pv}")
                print(f"tTR_rAcc_pv: {tTR_rAcc_pv}")
                print(f"tiled_copy_t2r_sq: {tiled_copy_t2r_sq}")
                print(f"tTR_tAcc_base_sq: {tTR_tAcc_base_sq}")
                print(f"tTR_rAcc_sq: {tTR_rAcc_sq}")

                print(f"tCtAccQK: {tCtAccQK}")
                print(f"tCtAccSQ: {tCtAccSQ}")
                print(f"tCtAccPV: {tCtAccPV}")

            # HANDLE P = QK^T
            (
                tiled_t2r_S,
                thr_t2r_S,
                # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, STAGES)
                tTR_tS,
                # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
                tTR_rS,
            ) = self.tmem_load_and_partition_qk(
                local_tidx,
                # (MMA, MMA_M, MMA_N, N_STAGE)
                # (MMA_M, MMA_N, STAGE)
                tCtAccQK[((None, None), 0, 0, None)],
            )
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            tTR_cS = thr_t2r_S.partition_D(cM)

            tTR_rP = cute.make_rmem_tensor_like(
                src=tTR_rS,
                dtype=self.q_dtype,
            )
            (
                tiled_r2s_P,
                thr_r2s_P,
                tRS_rP,
                tRS_sP,
            ) = self.smem_store_and_partition_qk(
                local_tidx=local_tidx,
                smem_p=sQK,
                tiled_t2r_qk=tiled_t2r_S,
                tPrP_t2r=tTR_rP,
                tiled_mma=qk_tiled_mma,
            )

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tTR_tS: {tTR_tS}")
                print(f"tTR_cS: {tTR_cS}")
                print(f"tTR_rS: {tTR_rS}")
                print(f"tTR_rP: {tTR_rP}")
                print(f"tRS_rP: {tRS_rP}")
                print(f"tRS_sP: {tRS_sP}")
            # -------------------------------------------------------

            # With ACC_STAGE
            # KV
            # (EPITILE_M, EPITILE_N, STAGES)
            tCtAccKV_slice = tCtAccKV[((None, None), 0, 0, None)]
            (
                tiled_copy_t2r_kv,
                _,  # thr_t2r
                tTR_tKV,
                tTR_rKV,
            ) = self.tmem_load_partition_kv(
                mma_tiler=self.kv_mma_tiler,
                tState=tCtAccKV_slice,
                local_tidx=local_tidx,
            )

            ############################################################
            # FP32 TMEM store partition (for h0 initial state loading)
            (
                tmem_store_kv_f32,
                tmem_store_tAccKV_f32,
                tmem_store_rAccKV_f32,
            ) = self.tmem_store_and_partition_acc(
                local_tidx,
                tCtAcc=tCtAccKV,
            )
            tmem_store_rKV = cute.make_tensor(tTR_rKV.iterator, layout=tmem_store_rAccKV_f32.layout)

            # BF16 TMEM store partition (for kv16 state used by SQ MMA)
            (
                tmem_store_kv,
                tmem_store_tAccKV,
                tmem_store_rAccKV,
            ) = self.tmem_store_and_partition_acc(
                local_tidx,
                tCtAcc=tCtStateAsF32,
            )
            tmem_store_rAccKVAsBF16 = cute.recast_tensor(tmem_store_rAccKV, dtype=self.io_dtype)
            ############################################################

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tiled_copy_t2r_kv: {tiled_copy_t2r_kv}")
                print(f"LOAD tTR_tKV: {tTR_tKV}")
                print(f"LOAD tTR_rKV: {tTR_rKV}")

            # -------------------------------------------------------
            # K decay weighting: thread-local partitioning (needs local_tidx)
            # -------------------------------------------------------
            thr_load_k_half = tiled_load_k_half.get_slice(local_tidx)
            thr_store_k_half = tiled_store_k_half.get_slice(local_tidx)

            # Partition SMEM halves for S2R load and R2S store
            tKsK_s2r_h = [thr_load_k_half.partition_S(h) for h in sK_s2r_halves]
            # Partition for writing to separate sK_weighted buffer (single stage)
            tKsK_r2s_h_weighted = [thr_store_k_half.partition_D(h) for h in sK_weighted_halves]

            # Register fragment prototype (for make_fragment_like)
            thr_mma_k_half = tiled_mma_k_half.get_slice(local_tidx)
            sK_s2r_h0_0 = sK_s2r_h0[None, None, 0]
            tKrK_half_proto = thr_mma_k_half.make_fragment_A(thr_mma_k_half.partition_A(sK_s2r_h0_0))

            # Identity tensor partition for mapping thread elements to (row, col)
            tKcK_half = thr_mma_k_half.partition_A(cK_half)

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"sK_flat_s2r: {cute.pretty_str(sK_flat_s2r)}")
                print(f"sK_s2r_h0: {cute.pretty_str(sK_s2r_h0)}")
                print(f"tiled_load_k_half: {cute.pretty_str(tiled_load_k_half)}")
                print(f"tKsK_s2r_h[0]: {cute.pretty_str(tKsK_s2r_h[0])}")
                print(f"tKrK_half_proto: {cute.pretty_str(tKrK_half_proto)}")
                print(f"tKcK_half: {cute.pretty_str(tKcK_half)}")
            # -------------------------------------------------------

            # RMEM flat view of KV state (D FP32 values per thread)
            init_flat = cute.make_tensor(tTR_rKV.iterator, layout=cute.make_layout(_D))

            # Identity tensor for O/PV output position tracking (D, C)
            # Used for tail chunk zeroing in varlen mode
            if cutlass.const_expr(self.is_varlen):
                cM_o = cute.make_identity_tensor(self.vp_mma_tiler[:2])  # (D, C)
                thr_t2r_pv_thread = tiled_copy_t2r_pv.get_slice(tidx)
                tTR_cO = thr_t2r_pv_thread.partition_D(cM_o)

            # ===== Persistent outer loop =====
            wu_iter = Int32(0)
            decay_s_cuda = Float32(0.0)
            block_decay = Float32(0.0)
            # Cumulative K stage offset: counts total K pipeline produces
            # across all WUs so far. Used to track which K SMEM stage
            # (out of k_stage=2) the Load warp wrote to for each chunk.
            # Must persist across WU boundaries in persistent mode.
            k_stage_offset = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)
                sched_phase0 = Int32(1)
                sched_phase1 = Int32(0)
                should_continue = work_idx < total_work_units
            elif cutlass.const_expr(self.is_varlen):
                work_idx = Int32(0)
                should_continue = wu_iter < num_iters
            else:
                should_continue = True

            while should_continue:
                # --- Work decode (persistent only) ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    hidx = work_idx % H
                    bidx = work_idx // H
                    bos = cu_seqlens[bidx]
                    eos = cu_seqlens[bidx + 1]
                    seq_len = eos - bos
                    state_idx = initial_state_indices[bidx]

                # Load per-head decay parameter to register (s_h > 0)
                # For persistent: hidx was decoded above; for non-persistent: hidx from block_idx
                decay_tensor_cuda = cute.make_tensor(decay, cute.make_layout(H))
                decay_s_cuda = decay_tensor_cuda[hidx]
                block_decay = cute.exp(-decay_s_cuda * cutlass.Float32(C), fastmath=self.use_fast_math)

                # -------------- Initial State Loading (h0) ----------------
                if cutlass.const_expr(self.has_initial_state):
                    gState_h0 = initial_state[None, None, (hidx, state_idx)]
                    gCol_h0 = cute.make_tensor(gState_h0.iterator + local_tidx * _D, cute.make_layout(_D, stride=1))
                    cute.autovec_copy(gCol_h0, init_flat)

                    # Store raw h0 as BF16 to kv16 TMEM for SQ MMA at idx=0
                    tmem_store_rAccKVAsBF16.store(tTR_rKV.load().to(self.io_dtype))
                    init_kv16_handle = kv16_producer.acquire_and_advance()
                    init_tmem_store_tAccKVi = tmem_store_tAccKV[None, None, None, None, init_kv16_handle.index]
                    cute.copy(tmem_store_kv, tmem_store_rAccKV, init_tmem_store_tAccKVi)
                    cute.arch.fence_view_async_tmem_store()
                    init_kv16_handle.commit()

                    # Store h0 * block_decay to FP32 TMEM accumulator
                    h0_decayed = tTR_rKV.load() * block_decay
                    tTR_rKV.store(h0_decayed)
                    init_tmem_store_tKVi = tmem_store_tAccKV_f32[None, None, None, None, 0]
                    cute.copy(tmem_store_kv_f32, tmem_store_rKV, init_tmem_store_tKVi)
                    cute.arch.fence_view_async_tmem_store()

                # ============================================================
                # Populate decay lookup table per work unit
                # sDecayLUT[k] = exp(-s*k) for k=0..C-1
                # ============================================================
                for lut_k in cutlass.range(_C, unroll_full=True):
                    if local_tidx == lut_k:
                        sDecayLUT[lut_k] = cute.exp(-decay_s_cuda * cutlass.Float32(lut_k), fastmath=self.use_fast_math)
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.cuda_sync_barrier.sync()

                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C

                    # Compute valid positions in this chunk for tail handling (varlen)
                    if cutlass.const_expr(self.is_varlen):
                        valid_len_chunk = seq_len - chunk_start
                        if valid_len_chunk > C:
                            valid_len_chunk = Int32(C)

                    ####################################################
                    # Apply block decay with type conversion
                    ####################################################
                    if chunk_start != 0:
                        kv_handle = kv_consumer.wait_and_advance()
                        tTR_tKVi = tTR_tKV[(None, None, None, kv_handle.index)]
                        cute.copy(tiled_copy_t2r_kv, tTR_tKVi, tTR_rKV)
                        cute.arch.fence_view_async_tmem_load()
                        kv_handle.release()

                        kv16_handle = kv16_producer.acquire_and_advance()
                        tmem_store_rAccKVAsBF16.store(tTR_rKV.load().to(self.io_dtype))
                        tmem_store_tAccKVi = tmem_store_tAccKV[None, None, None, None, kv16_handle.index]
                        cute.copy(tmem_store_kv, tmem_store_rAccKV, tmem_store_tAccKVi)
                        cute.arch.fence_view_async_tmem_store()

                        kv_state_decayed = tTR_rKV.load() * block_decay
                        tTR_rKV.store(kv_state_decayed)
                        tmem_store_tKVi_f32 = tmem_store_tAccKV_f32[None, None, None, None, 0]
                        cute.copy(tmem_store_kv_f32, tmem_store_rKV, tmem_store_tKVi_f32)
                        cute.arch.fence_view_async_tmem_store()

                        kv16_handle.commit()

                    # Weight K to separate buffer (sK_weighted)
                    # Use k_stage_offset + idx to derive the correct K SMEM stage,
                    # staying in sync with K TMA pipeline across WU boundaries.
                    k_stage_idx = (idx + k_stage_offset) % 2

                    for half_idx in cutlass.range_constexpr(2):
                        tKrK_half = cute.make_fragment_like(tKrK_half_proto, self.io_dtype)
                        tKrK_half_cv = thr_load_k_half.retile(tKrK_half)
                        cute.copy(tiled_load_k_half, tKsK_s2r_h[half_idx][None, None, None, k_stage_idx], tKrK_half_cv)

                        for i in cutlass.range_constexpr(cute.size(tKcK_half)):
                            row, col = tKcK_half[i]
                            weight = sDecayLUT[_C - 1 - row]
                            k_val = tKrK_half[i].to(cutlass.Float32)
                            tKrK_half[i] = (k_val * weight).to(self.io_dtype)
                            if cutlass.const_expr(self.is_varlen):
                                if row >= valid_len_chunk:
                                    tKrK_half[i] = self.io_dtype(0.0)

                        tKrK_half_cv_dst = thr_store_k_half.retile(tKrK_half)
                        cute.copy(tiled_store_k_half, tKrK_half_cv_dst, tKsK_r2s_h_weighted[half_idx][None, None, None, 0])

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    kw_prod_handle = k_weighted_producer.acquire_and_advance()
                    kw_prod_handle.commit()

                    # Wait for S = QK^T
                    s0_handle = mma_s0_consumer.wait_and_advance()

                    tTR_tSi = tTR_tS[None, None, None, s0_handle.index]
                    cute.copy(tiled_t2r_S, tTR_tSi, tTR_rS)
                    cute.arch.fence_view_async_tmem_load()

                    # Apply exponential decay mask and convert to BF16
                    self.apply_decay_mask(tTR_rS, tTR_cS, tTR_rP, sDecayLUT, debug=False)

                    # Zero P at invalid positions for tail chunk (varlen)
                    if cutlass.const_expr(self.is_varlen):
                        if valid_len_chunk < C:
                            for i in cutlass.range_constexpr(cute.size(tTR_rP)):
                                q_pos, k_pos = tTR_cS[i]
                                if q_pos >= valid_len_chunk or k_pos >= valid_len_chunk:
                                    tTR_rP[i] = self.q_dtype(0.0)

                    # Write P to SMEM
                    p_handle = p_producer.acquire_and_advance()
                    tRS_sPi = tRS_sP[(None, None, None, p_handle.index)]
                    cute.copy(tiled_r2s_P, tRS_rP, tRS_sPi)
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    s0_handle.release()
                    p_handle.commit()

                    # Wait for O_INTRA (PV result)
                    o_intra_handle = o_intra_consumer.wait_and_advance()
                    tTR_tAcc_pv_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, o_intra_handle.index)]
                    cute.copy(tiled_copy_t2r_pv, tTR_tAcc_pv_i, tTR_rAcc_pv)
                    cute.arch.fence_view_async_tmem_load()
                    o_intra_handle.release()

                    # Wait for O_INTER (SQ result)
                    if chunk_start != 0 or cutlass.const_expr(self.has_initial_state):
                        o_inter_handle = o_inter_consumer.wait_and_advance()
                        tTR_tAcc_sq_i = tTR_tAcc_base_sq[(None, None, None, 0, 0, o_inter_handle.index)]
                        cute.copy(tiled_copy_t2r_sq, tTR_tAcc_sq_i, tTR_rAcc_sq)
                        cute.arch.fence_view_async_tmem_load()
                        o_inter_handle.release()

                    # Combine: O = (O_INTRA + O_INTER_decayed) * scale
                    acc_vec = tTR_rAcc_pv.load()
                    if chunk_start != 0 or cutlass.const_expr(self.has_initial_state):
                        self.apply_inter_chunk_decay(tTR_rAcc_sq, tTR_cSQ, sDecayLUT)
                        acc_vec = acc_vec + tTR_rAcc_sq.load()
                    acc_vec = acc_vec * scale
                    tTR_rO.store(acc_vec.to(self.io_dtype))

                    # Zero O at invalid positions for tail chunk (varlen)
                    if cutlass.const_expr(self.is_varlen):
                        if valid_len_chunk < C:
                            for i in cutlass.range_constexpr(cute.size(tTR_rO)):
                                _, c_pos = tTR_cO[i]
                                if c_pos >= valid_len_chunk:
                                    tTR_rO[i] = self.io_dtype(0.0)

                    # Store output to smem
                    smem_o_handle = smem_o_producer.acquire_and_advance()
                    cute.copy(tiled_copy_r2s_o, tRS_rO, tRS_sO[(None, None, None, smem_o_handle.index)])
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    smem_o_handle.commit()

                # -------------- Final State Output (ht) ----------------
                if cutlass.const_expr(self.output_final_state):
                    kv_handle = kv_consumer.wait_and_advance()
                    tTR_tKVi = tTR_tKV[(None, None, None, kv_handle.index)]
                    cute.copy(tiled_copy_t2r_kv, tTR_tKVi, tTR_rKV)
                    cute.arch.fence_view_async_tmem_load()
                    kv_handle.release()

                    if cutlass.const_expr(self.is_varlen):
                        gState_ht = initial_state[None, None, (hidx, state_idx)]
                    else:
                        gState_ht = final_state[None, None, (hidx, state_idx)]
                  
                    gCol_ht = cute.make_tensor(gState_ht.iterator + local_tidx * _D, cute.make_layout(_D, stride=1))
                    out_flat = cute.make_tensor(tTR_rKV.iterator, layout=cute.make_layout(_D))
                    cute.autovec_copy(out_flat, gCol_ht)

                # Advance k_stage_offset by number of chunks in this WU
                # so next WU's k_stage_idx stays in sync with the K pipeline.
                num_chunks_wu = (seq_len + Int32(C - 1)) // Int32(C)
                k_stage_offset = k_stage_offset + num_chunks_wu

                # --- End-of-WU: wait for Load warp's scheduling signal ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    # CTA barrier: sync all warps before WU transition
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_per_cta)
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                elif cutlass.const_expr(self.is_varlen):
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters
                else:
                    should_continue = False

        elif warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            # --- CopyUniversal setup for varlen tail O store ---
            if cutlass.const_expr(self.is_varlen):
                epi_local_tidx = tidx % self.threads_per_warp  # 0..31
                universal_copy_bits = 128
                async_copy_elems = universal_copy_bits // self.io_dtype.width  # 8 for bf16
                atom_universal_copy_o = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.io_dtype,
                    num_bits_per_copy=universal_copy_bits,
                )
                o_thr_dim0 = D // async_copy_elems  # 128/8 = 16
                o_thr_dim1 = self.threads_per_warp // o_thr_dim0  # 32/16 = 2
                o_thr_layout = cute.make_ordered_layout(
                    (o_thr_dim0, o_thr_dim1),
                    order=(0, 1),
                )
                o_val_layout = cute.make_layout((async_copy_elems, 1))
                gmem_tiled_copy_o = cute.make_tiled_copy_tv(
                    atom_universal_copy_o,
                    o_thr_layout,
                    o_val_layout,
                )

            wu_iter = Int32(0)
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)
                sched_phase0 = Int32(1)
                sched_phase1 = Int32(0)
                should_continue = work_idx < total_work_units
            elif cutlass.const_expr(self.is_varlen):
                work_idx = Int32(0)
                should_continue = wu_iter < num_iters
            else:
                should_continue = True

            while should_continue:
                # --- Work decode (persistent only) ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    hidx = work_idx % H
                    bidx = work_idx // H
                    bos = cu_seqlens[bidx]
                    eos = cu_seqlens[bidx + 1]
                    seq_len = eos - bos

                # TMA STORE
                if cutlass.const_expr(self.is_varlen):
                    tma_tensor_o_use = cute.domain_offset((0, bos, 0), tma_tensor_o)
                else:
                    tma_tensor_o_use = tma_tensor_o

                gO_pre_partition = cute.flat_divide(tma_tensor_o_use, cute.select(self.vp_mma_tiler, mode=[0, 1]))

                if cutlass.const_expr(self.is_varlen):
                    gO_pre_partition = gO_pre_partition[None, None, None, None, hidx]
                else:
                    gO_pre_partition = gO_pre_partition[None, None, None, None, (hidx, bidx)]

                tma_atom_o, bSG_sO, bSG_gO = self.epilog_gmem_copy_and_partition(
                    tma_atom_o,
                    gO_pre_partition,
                    self.epi_tile,
                    sO,
                )

                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C

                    smem_o_handle = smem_o_consumer.wait_and_advance()

                    if cutlass.const_expr(self.is_varlen):
                        remaining = seq_len - chunk_start
                        if remaining >= C:
                            cute.copy(tma_atom_o, bSG_sO[None, smem_o_handle.index], bSG_gO[(None, 0, 0, 0, idx)])
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        else:
                            sO_stage = sO_epi[None, None, smem_o_handle.index]

                            gmem_thr_copy_o = gmem_tiled_copy_o.get_slice(epi_local_tidx)
                            tOsO = gmem_thr_copy_o.partition_S(sO_stage)

                            cO = cute.make_identity_tensor((D, C))
                            tOcO = gmem_thr_copy_o.partition_S(cO)

                            tOrO = cute.make_fragment_like(tOsO, self.io_dtype)
                            cute.autovec_copy(tOsO, tOrO)

                            o_chunk_raw = o_tensor.iterator + (bos + chunk_start) * D * H + hidx * D
                            o_chunk_ptr = cute.make_ptr(
                                self.io_dtype,
                                o_chunk_raw.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            o_stride_c = D * H
                            gO_chunk = cute.make_tensor(
                                o_chunk_ptr,
                                cute.make_layout(
                                    (D, C),
                                    stride=(1, o_stride_c),
                                ),
                            )

                            tOgO = gmem_thr_copy_o.partition_D(gO_chunk)

                            for rest_c in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                                c_coord = tOcO[0, 0, rest_c][1]
                                if c_coord < remaining:
                                    cute.copy(
                                        gmem_tiled_copy_o,
                                        tOrO[None, None, rest_c],
                                        tOgO[None, None, rest_c],
                                    )
                    else:
                        cute.copy(tma_atom_o, bSG_sO[None, smem_o_handle.index], bSG_gO[(None, 0, 0, 0, idx)])
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)

                    smem_o_handle.release()

                # --- End-of-WU: wait for Load warp's scheduling signal ---
                if cutlass.const_expr(self.is_varlen and self.persistent):
                    # CTA barrier: sync all warps before WU transition
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_per_cta)
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    sched_buf = Int32(1) - sched_buf
                    should_continue = work_idx < total_work_units
                elif cutlass.const_expr(self.is_varlen):
                    wu_iter = wu_iter + 1
                    should_continue = wu_iter < num_iters
                else:
                    should_continue = False
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Dynamic scheduling: wait on double-buffered mbarriers for each WU
            if cutlass.const_expr(self.is_varlen and self.persistent):
                work_idx = sWorkIdx[(0,)]
                sched_buf = Int32(1)
                sched_phase0 = Int32(1)
                sched_phase1 = Int32(0)
                should_continue_else = work_idx < total_work_units
                while should_continue_else:
                    # CTA barrier: sync all warps before WU transition
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_per_cta)
                    if sched_buf == 0:
                        cute.arch.mbarrier_wait(sched_mbar_base, sched_phase0)
                        sched_phase0 = Int32(1) - sched_phase0
                    else:
                        cute.arch.mbarrier_wait(sched_mbar_base + 1, sched_phase1)
                        sched_phase1 = Int32(1) - sched_phase1
                    work_idx = sWorkIdx[(sched_buf,)]
                    sched_buf = Int32(1) - sched_buf
                    should_continue_else = work_idx < total_work_units
        # Release tensor memory allocation lock
        tmem.relinquish_alloc_permit()
        # Sync before deallocating tmem
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        # Dealloc tmem buffer
        tmem.free(tmem_ptr_base)

        return

    def epilog_smem_copy_and_partition_o(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rO: cute.Tensor,
        tidx: cutlass.Int32,
        sO: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source)
        and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rO: The partitioned accumulator tensor
        :type tTR_rO: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sO: The shared memory tensor to be copied and partitioned
        :type sO: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rO: The partitioned tensor C (register source)
            - tRS_sO: The partitioned tensor C (smem destination)
        :rtype: tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.o_layout, self.io_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sO = thr_copy_r2s.partition_D(sO)
        # (R2S, R2S_M, R2S_N)
        tRS_rO = tiled_copy_r2s.retile(tTR_rO)
        return tiled_copy_r2s, tRS_rO, tRS_sO

    def epilog_gmem_copy_and_partition(
        self,
        atom: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """
        Partitions source and destination tensors for a global memory store.

        This method generates a tiled copy for storing results to global memory
        and partitions the source (register or shared memory) and destination
        (global memory) tensors accordingly. The behavior varies based on whether
        TMA store is enabled.

        :param tidx: The thread index in epilogue warp groups.
        :type tidx: cutlass.Int32
        :param atom: The copy atom to be used (TMA or universal).
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler.
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor C.
        :return: A tuple containing the appropriate copy atom and partitioned
                 source and destination tensors for the store operation.
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # gc_mnl: (D, C, TILES_D, TILES_C)
        # gC_epi: (D, C, 1, 1, TILES_D, TILES_C)
        gC_epi = cute.flat_divide(
            # (D, C, TILES_D, TILES_C)
            gC_mnl,
            # epi: (D, C)
            epi_tile,
        )

        tma_atom_c = atom
        # ((D, C), STAGE)
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        # ((D, C), 1, 1, TILES_D, TILES_C)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), STAGE)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, TILES_D, TILES_C)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    def tmem_load_partition_kv(self, mma_tiler, tState, local_tidx):
        # Make tiledCopy for tensor memory load
        # D,D
        # KV: V^T*K, K-Major
        # Q: K-Major
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            mma_tiler,
            utils.LayoutEnum.ROW_MAJOR,
            self.io_dtype,
            self.acc_dtype,
            mma_tiler[:2],
            use_2cta_instrs=False,
        )
        fake_sState = cute.make_tensor(
            cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
            cute.dice(self.kv_mma_tiler, (1, 1, None)),
        )
        return self.make_tmem_load_and_partition(copy_atom_t2r, tState, (None, None, 0), local_tidx, fake_sState)

    def make_tmem_load_and_partition(self, copy_atom_t2r, tmem_tensor, tmem_tile_coord, local_tidx, smem_tensor):
        dtype = tmem_tensor.element_type
        # TMEM: (EPITILE_M, EPITILE_N, STAGES)
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tmem_tensor[tmem_tile_coord])
        thr_t2r = tiled_t2r.get_slice(local_tidx)
        # Partition tmem/shared tensor for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, STAGES)
        tTR_t = thr_t2r.partition_S(tmem_tensor)
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_s = thr_t2r.partition_D(smem_tensor)
        # Make register fragments for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_r = cute.make_rmem_tensor(
            tTR_s.shape,
            dtype,
        )
        return tiled_t2r, thr_t2r, tTR_t, tTR_r

    def tmem_store_and_partition_kv(self, local_tidx, tCtState):
        dtype = tCtState.element_type
        # Make tiledCopy for tensor memory store
        # (D, D), 128,128, 16b
        copy_atom_r2t = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.UNPACK_32b_IN_16b),
            dtype,
        )

        tiled_r2t_kv = tcgen05.make_tmem_copy(copy_atom_r2t, tCtState)
        thr_r2t_kv = tiled_r2t_kv.get_slice(local_tidx)

        # Partition tmem/register tensor for tensor memory store INTRA2_Q
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ...)
        tCtState_partitioned = thr_r2t_kv.partition_S(tCtState)
        tRT_rKV16 = cute.make_rmem_tensor(
            cute.slice_(tCtState_partitioned.shape, (None, None, None, 0)),
            dtype,
        )
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ..., INTERNAL_STAGE)
        tRT_tKV16 = thr_r2t_kv.partition_D(tCtState)

        return tiled_r2t_kv, tRT_tKV16, tRT_rKV16

    def tmem_store_and_partition_acc(self, local_tidx, tCtAcc):
        dtype = tCtAcc.element_type
        copy_atom_r2t = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE),
            dtype,
        )

        tiled_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, tCtAcc)
        thr_r2t = tiled_r2t.get_slice(local_tidx)

        # Partition tmem/register tensor for tensor memory store INTRA2_Q
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ...)
        tCtAcc_partitioned = thr_r2t.partition_S(tCtAcc)
        tRT_rAcc = cute.make_rmem_tensor(
            cute.slice_(tCtAcc_partitioned.shape, (None, None, None, None, 0)),
            dtype,
        )
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ..., INTERNAL_STAGE)
        tRT_tAcc = thr_r2t.partition_D(tCtAcc)

        return tiled_r2t, tRT_tAcc, tRT_rAcc

    @cute.jit
    def tmem_load_and_partition_qk(
        self,
        local_tidx,
        tQK,
    ):
        # 64,64
        copy_atom_t2r_qk = cute.make_copy_atom(
            # 32b x 8 x 8
            tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        fake_sQK = cute.make_tensor(
            cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
            cute.dice(self.qk_mma_tiler, (1, 1, None)),
        )
        # tQK: (EPITILE_M, EPITILE_N, STAGES)
        # Tiled Copy for one stage
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r_qk, tQK[None, None, 0])
        thr_t2r = tiled_t2r.get_slice(local_tidx)
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, STAGES)
        tTR_t = thr_t2r.partition_S(tQK)
        # (EPITILE_M, EPITILE_N)
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_s = thr_t2r.partition_D(fake_sQK)
        tTR_r = cute.make_rmem_tensor(
            tTR_s.shape,
            tQK.element_type,
        )
        return tiled_t2r, thr_t2r, tTR_t, tTR_r

    @cute.jit
    def smem_store_and_partition_qk(
        self,
        local_tidx,
        smem_p,
        tiled_t2r_qk,
        tPrP_t2r,
        tiled_mma=None,
    ):
        copy_atom_r2s_qk = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.from_tensor(smem_p), self.io_dtype, self.acc_dtype, tiled_t2r_qk
        )
        # num_dp, num_bits, num_rep, pack = sm100_utils.get_tmem_copy_properties(tiled_t2r_qk)
        tiled_r2s_qk = cute.make_tiled_copy_D(copy_atom_r2s_qk, tiled_t2r_qk)
        thr_r2s_qk = tiled_r2s_qk.get_slice(local_tidx)

        # ((V, R), M, N, N_STAGE)
        tPsP_r2s = thr_r2s_qk.partition_D(smem_p)

        # ((V, R), M, N)
        tPrP_r2s = thr_r2s_qk.retile(tPrP_t2r)

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"copy_atom_r2s_qk: {copy_atom_r2s_qk}")
            print(f"tiled_t2r_qk: {tiled_t2r_qk}")
            print(f"thr_t2r_qk: {thr_r2s_qk}")
            print(f"before partition_D: {smem_p}")
            print(f"after partition_D, tPsP_r2s: {tPsP_r2s}")
            print(f"before retile tPrP_t2r: {tPrP_t2r}")
            print(f"after retile tPrP_r2s: {tPrP_r2s}")

        return tiled_r2s_qk, thr_r2s_qk, tPrP_r2s, tPsP_r2s

    @cute.jit
    def smem_store_acc_as_ab_and_partition_x(
        self,
        local_tidx,
        smem_x,
        tiled_t2r_x,
        tXrX_t2r,
    ):
        copy_atom_r2s_x = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.from_tensor(smem_x), self.io_dtype, self.acc_dtype, tiled_t2r_x
        )
        tiled_r2s_x = cute.make_tiled_copy_D(copy_atom_r2s_x, tiled_t2r_x)
        thr_r2s_x = tiled_r2s_x.get_slice(local_tidx)

        # ((V, R), M, N, N_STAGE)
        tXsX_r2s = thr_r2s_x.partition_D(smem_x)

        # ((V, R), M, N)
        tXrX_r2s = thr_r2s_x.retile(tXrX_t2r)

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"copy_atom_r2s_x: {copy_atom_r2s_x}")
            print(f"tiled_t2r_x: {tiled_t2r_x}")
            print(f"thr_t2r_x: {thr_r2s_x}")
            print(f"before partition_D: {smem_x}")
            print(f"after partition_D, tXsX_r2s: {tXsX_r2s}")
            print(f"before retile tXrX_t2r: {tXrX_t2r}")
            print(f"after retile tXrX_r2s: {tXrX_r2s}")

        return tiled_r2s_x, thr_r2s_x, tXrX_r2s, tXsX_r2s

    def epilog_tmem_load_and_partition_acc(self, local_tidx, tIntra, smem_y):
        copy_atom_t2r_inter2_intra2 = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        return self.make_tmem_load_and_partition_acc(
            copy_atom_t2r_inter2_intra2,
            tIntra,
            (None, None, 0, 0, 0),
            local_tidx,
            smem_y[None, None, 0],
        )

    def make_tmem_load_and_partition_acc(self, copy_atom_t2r, tmem_tensor, tmem_tile_coord, local_tidx, smem_tensor):
        dtype = tmem_tensor.element_type
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tmem_tensor[tmem_tile_coord])
        thr_t2r = tiled_t2r.get_slice(local_tidx)
        # Partition tmem/shared tensor for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_t = thr_t2r.partition_S(tmem_tensor)
        tTR_s = thr_t2r.partition_D(smem_tensor)
        # Make register fragments for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_r = cute.make_rmem_tensor(
            tTR_s.shape,
            dtype,
        )
        return tiled_t2r, tTR_t, tTR_r

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        mma_tiler: cute.Tile,
        use_2cta_instrs: cutlass.Boolean | bool,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Partitions source and destination tensors for a tensor memory load.

        This method generates a tiled copy for loading accumulators from tensor
        memory and partitions the source (tensor memory) and destination
        (register) tensors accordingly.

        :param tidx: The thread index in epilogue warp groups.
        :param tAcc: The accumulator tensor to be copied and partitioned.
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled.
        :return: A tuple containing the tiled copy for the load operation and
                 the partitioned source and destination tensors.
        """
        # Make tiledCopy for tensor memory load
        epitile = mma_tiler[:2]
        assert epitile[0] == 128
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        # (EPI_TILE_M, EPI_TILE_N, 1, 1, STAGE)
        tAcc_epi = cute.flat_divide(
            # ((EPI_TILE_M, EPI_TILE_N), EPI_M, EPI_N, STAGE)
            tAcc[((None, None), 0, 0, None)],
            epitile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N)
        fake_sAcc = cute.make_tensor(
            cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.smem),
            cute.dice(mma_tiler, (1, 1, None)),
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN)
        tTR_sAcc = thr_copy_t2r.partition_D(fake_sAcc)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_sAcc.shape,
            tAcc.element_type,
        )

        if cutlass.const_expr(PRINT_DEBUG):
            print("------------ EPILOG TMEM COPY AND PARTITION BEGIN --------------")
            print(f"tAcc: {tAcc}")
            print(f"tAcc_epi: {tAcc_epi}")
            print(f"copy_atom_t2r: {copy_atom_t2r}")
            print(f"tiled_copy_t2r: {tiled_copy_t2r}")
            print(f"thr_copy_t2r: {thr_copy_t2r}")
            print(f"tTR_tAcc: {tTR_tAcc}")
            print(f"tTR_rAcc: {tTR_rAcc}")
            print("------------ EPILOG TMEM COPY AND PARTITION END --------------")

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    @cute.jit
    def apply_decay_mask(
        self,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        p: cute.Tensor,
        decay_lut: cute.Tensor,  # Decay LUT: sDecayLUT[k] = exp(-s*k) for k=0..C-1
        debug: bool = False,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        # Apply exponential decay mask: D_ij = exp(-s * (i-j)) for i >= j
        # Uses precomputed decay LUT instead of exp() calls
        # Phase 3 Optimization 3.5: Loop unrolling by 2x for better ILP
        size = cute.size(acc_qk)

        # Process pairs of elements (unrolled by 2) for instruction-level parallelism
        for i in cutlass.range_constexpr(0, size - 1, 2):
            # Element i
            index_q0, index_k0 = index_transform(*index_qk[i])
            distance0 = index_q0 - index_k0

            if debug:
                cute.printf("index_qk[{}], {},{}", i, index_q0, index_k0)

            if index_q0 < index_k0:
                acc_qk[i] = cutlass.Float32(0.0)
                p[i] = cutlass.BFloat16(0.0)
            else:
                # Use LUT: decay_lut[distance] = exp(-s*distance)
                p[i] = (acc_qk[i] * decay_lut[distance0]).to(self.q_dtype)

            # Element i+1 (unrolled for ILP)
            index_q1, index_k1 = index_transform(*index_qk[i + 1])
            distance1 = index_q1 - index_k1

            if index_q1 < index_k1:
                acc_qk[i + 1] = cutlass.Float32(0.0)
                p[i + 1] = cutlass.BFloat16(0.0)
            else:
                p[i + 1] = (acc_qk[i + 1] * decay_lut[distance1]).to(self.q_dtype)

        # Handle last element if size is odd
        if size % 2 == 1:
            i = size - 1
            index_q, index_k = index_transform(*index_qk[i])
            distance = index_q - index_k

            if index_q < index_k:
                acc_qk[i] = cutlass.Float32(0.0)
                p[i] = cutlass.BFloat16(0.0)
            else:
                p[i] = (acc_qk[i] * decay_lut[distance]).to(self.q_dtype)

    @cute.jit
    def scale_tmem_accumulator(self, tmem_acc: cute.Tensor, scale_factor):
        """Scale TMEM accumulator tensor by a scalar factor (for block-level decay)."""
        for i in cutlass.range_constexpr(cute.size(tmem_acc)):
            tmem_acc[i] = tmem_acc[i] * scale_factor

    @cute.jit
    def apply_inter_chunk_decay(
        self,
        acc_sq: cute.Tensor,
        index_sq: cute.Tensor,
        decay_lut: cute.Tensor,  # Decay LUT: sDecayLUT[k] = exp(-s*k) for k=0..C-1
    ):
        """Apply per-position inter-chunk decay to SQ MMA output.

        For position t within the chunk, multiply by exp(-s*(t+1)).
        Uses decay_lut[pos] * decay_lut[1] = exp(-s*pos) * exp(-s) = exp(-s*(pos+1))

        Args:
            acc_sq: SQ MMA output register tensor (FP32)
            index_sq: Position identity partition giving (d_idx, pos_idx) per element
            decay_lut: Precomputed decay LUT where decay_lut[k] = exp(-s*k), k=0..C-1
        """
        size = cute.size(acc_sq)
        decay_one = decay_lut[1]  # exp(-s)

        for i in cutlass.range_constexpr(0, size - 1, 2):
            # Element i: get position within chunk (column index of SQ output)
            pos0 = index_sq[i][1]
            acc_sq[i] = cutlass.Float32(acc_sq[i]) * decay_lut[pos0] * decay_one

            # Element i+1 (unrolled for ILP)
            pos1 = index_sq[i + 1][1]
            acc_sq[i + 1] = cutlass.Float32(acc_sq[i + 1]) * decay_lut[pos1] * decay_one

        # Handle last element if size is odd
        if size % 2 == 1:
            i_last = size - 1
            pos_last = index_sq[i_last][1]
            acc_sq[i_last] = cutlass.Float32(acc_sq[i_last]) * decay_lut[pos_last] * decay_one

    @cute.jit
    def make_tmem_store_and_partition(self, copy_atom_r2t, tmem_tensor, local_tidx):
        tiled_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, tmem_tensor)
        thr_r2t = tiled_r2t.get_slice(local_tidx)
        tRT_t = thr_r2t.partition_D(tmem_tensor)
        return tiled_r2t, tRT_t

    @cute.jit
    def mma_partition_ss(
        self,
        tiled_mma: cute.TiledMma,
        tile_shape_mnk: cute.Tile,
        smem_a,
        smem_b,
        tmem_acc_ptr,
        acc_stages,
    ):
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        tCrA = tiled_mma.make_fragment_A(smem_a)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        tCrB = tiled_mma.make_fragment_B(smem_b)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCtAcc = self.mma_partition_c(tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages)
        return tCrA, tCrB, tCtAcc

    @cute.jit
    def mma_partition_ts(
        self,
        tiled_mma,
        tile_shape_mnk,
        a_tmem_layout,
        smem_b,
        tmem_a_ptr,
        tmem_acc_ptr,
        acc_stages,
    ):
        # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
        tCrA = self.mma_partition_a_tmem(tiled_mma, a_tmem_layout, tmem_a_ptr)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        tCrB = tiled_mma.make_fragment_B(smem_b)
        # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
        tCtAcc = self.mma_partition_c(tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages)
        return tCrA, tCrB, tCtAcc

    @cute.jit
    def mma_partition_a_tmem(self, tiled_mma, a_tmem_layout, tmem_a_ptr):
        tCrA_fake = tiled_mma.make_fragment_A(a_tmem_layout.outer.shape)
        tCrA = cute.make_tensor(
            cute.recast_ptr(
                tmem_a_ptr,
                dtype=tCrA_fake.element_type,
            ),
            tCrA_fake.layout,
        )
        return tCrA

    @cute.jit
    def mma_partition_c(self, tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages):
        acc_shape = tiled_mma.partition_shape_C(tile_shape_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))
        # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
        tCtAcc = cute.make_tensor(tmem_acc_ptr, tCtAcc_fake.layout)
        return tCtAcc

    @cute.jit
    def exec_mma(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        a_stage_idx,
        b_stage_idx,
        acc_stage_idx,
        always_acc=False,
    ):
        for kphase_idx in cutlass.range(cute.size(tCrB, mode=[2]), unroll_full=True):
            # set accu = 1
            tiled_mma.set(
                tcgen05.Field.ACCUMULATE,
                cutlass.Boolean(kphase_idx != 0 or always_acc),
            )
            cute.gemm(
                tiled_mma,
                tCtAcc[None, None, None, acc_stage_idx],
                tCrA[None, None, kphase_idx, a_stage_idx],
                tCrB[None, None, kphase_idx, b_stage_idx],
                tCtAcc[None, None, None, acc_stage_idx],
            )
        return tiled_mma

    @cute.jit
    def local_tile_partition_for_mma_operand(
        self,
        tensor_x,
        tile_shape,
        tiled_mma,
        operand_mode,
        hidx,
        bidx,
        debug_name=None,
        no_cta_coord=False,
    ):
        # Local_tile partition global tensors
        # x: (0,0,0,0) o (M,K,(H,B)):(1@1,1@0,(1@2,1@3))
        # (MMATile_M, MMATile_K, TILES_M, TILES_K, (H, B))
        operand_mode = operand_mode.upper()
        coord = None
        if cutlass.const_expr(operand_mode == "B"):
            coord = (0, None, None)
        elif cutlass.const_expr(operand_mode == "C"):
            coord = (None, None, 0)
        elif cutlass.const_expr(operand_mode == "A"):
            coord = (None, 0, None)
        else:
            raise RuntimeError(f"unknown operand mode: {operand_mode}")

        gX = cute.local_tile(
            tensor_x,
            cute.slice_(tile_shape, coord),  #
            (None, None, hidx)
            if cutlass.const_expr(self.is_varlen) and not no_cta_coord
            else (None, None, (hidx, bidx))
            if not no_cta_coord
            else (None, None, None),
        )
        # Partition global tensor with regard to TiledMMA
        thr_mma = tiled_mma.get_slice(0)
        # tCgX: (MMA, MMA_M, MMA_K, TILES_M, TILES_K)
        if cutlass.const_expr(operand_mode == "A"):
            tCgX = thr_mma.partition_A(gX)
        elif cutlass.const_expr(operand_mode == "B"):
            tCgX = thr_mma.partition_B(gX)
        elif cutlass.const_expr(operand_mode == "C"):
            tCgX = thr_mma.partition_C(gX)
        else:
            raise RuntimeError("unknown operand mode")
        return tCgX

    @cute.jit
    def tma_partition_for_mma_operand(
        self,
        tma_atom_x,
        tma_tensor_x,
        smem_x,
        tile_shape,
        tiled_mma,
        operand_mode,
        hidx,
        bidx,
        debug_name=None,
    ):
        tCgX = self.local_tile_partition_for_mma_operand(
            tensor_x=tma_tensor_x,
            tile_shape=tile_shape,
            tiled_mma=tiled_mma,
            operand_mode=operand_mode,
            hidx=hidx,
            bidx=bidx,
            debug_name=debug_name,
        )
        # Partition shared tensor with regard to TMA
        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), TILES_N, TILES_K)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            0,  # no multicast
            cute.make_layout(1),
            cute.group_modes(smem_x, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX


def make_thread_cooperative_group(size: int):
    """Helper to create thread cooperative groups for pipeline synchronization."""
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


# ---------------------------------------------------------------------------
# Compile cache + TVM-FFI API
# ---------------------------------------------------------------------------

# Internal cache: maps (has_initial_state, output_final_state, H, D, scale, chunk_size) → compiled_fn
_kernel_cache: dict = {}


def _compile_single_variant(has_initial_state, output_final_state, H, D, scale, chunk_size):
    """Compile one kernel variant. Returns the compiled TVM-FFI callable.

    Uses make_fake_compact_tensor and make_fake_stream for compilation with
    TVM-FFI.  At runtime, torch tensors are passed directly (zero-copy).
    Uses sym_int() for dynamic B and S dimensions so one compiled kernel
    handles all batch-size / sequence-length combinations.
    """
    kernel_obj = LinearAttentionChunkwiseDecay(
        chunk_size=chunk_size,
        acc_dtype=cutlass.Float32,
        io_dtype=cutlass.BFloat16,
        has_initial_state=has_initial_state,
        output_final_state=output_final_state,
        H=H,
        K=D,
        V=D,
        scale=scale,
        use_fast_math=USE_FAST_MATH,
    )

    sym_b = cute.sym_int()
    sym_s = cute.sym_int()

    # Q, K, V, O: (B, S, H, D) row-major bf16
    q_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_s, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    k_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_s, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    v_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_s, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    o_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_b, sym_s, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )

    # decay: (H,) float32
    decay_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (H,),
        assumed_align=128,
    )

    # initial_state / final_state: (B, H, D, D) float32 or None
    h0_fake = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_b, H, D, D),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        if has_initial_state
        else None
    )
    ht_fake = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_b, H, D, D),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        if output_final_state
        else None
    )

    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    # Dummy cu_seqlens / initial_state_indices for non-varlen
    # (never dereferenced — guarded by const_expr(is_varlen)),
    # but CuteDSL requires valid tensors, not None.
    dummy_cu_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (2,),
        assumed_align=128,
    )
    dummy_idx_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=128,
    )

    # Dummy workspace for non-varlen (never dereferenced, persistent=False)
    workspace_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=128,
    )

    compiled_fn = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        v_fake,
        o_fake,
        decay_fake,
        h0_fake,
        ht_fake,
        dummy_cu_fake,
        dummy_idx_fake,
        o_fake,  # o_tensor (reuse o_fake — never dereferenced for non-varlen)
        workspace_fake,
        (Int32(1), Int32(1)),  # dummy (B, S)
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_fn


def _get_compiled_kernel(has_initial_state, output_final_state, H, D, scale, chunk_size):
    """Get a compiled kernel with on-demand (lazy) compilation.

    Each variant is compiled exactly once and cached.  Compilation is deferred
    until the variant is actually needed so that cute.compile is always
    immediately followed by execution — this avoids a CuTe DSL runtime issue
    where a subsequent cute.compile can invalidate previously compiled but
    not-yet-executed functions.

    Cache key: (has_initial_state, output_final_state, H, D, scale, chunk_size, USE_FAST_MATH)
    """
    key = (has_initial_state, output_final_state, H, D, scale, chunk_size, USE_FAST_MATH)
    if key not in _kernel_cache:
        _kernel_cache[key] = _compile_single_variant(
            has_initial_state,
            output_final_state,
            H,
            D,
            scale,
            chunk_size,
        )
    return _kernel_cache[key]


def lightning_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    decay: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lightning Attention forward pass with compile cache and TVM-FFI.

    Uses make_fake_compact_tensor for compilation (no GC issues).
    At runtime, torch tensors are passed directly via TVM-FFI.
    sym_int() is used for B and S so a single compilation handles all
    batch-size / sequence-length combinations.

    Cache key: (has_initial_state, output_final_state, H, D, scale, chunk_size)

    Args:
        Q: (B, S, H, D) bf16 query
        K: (B, S, H, D) bf16 key
        V: (B, S, H, D) bf16 value
        decay: (H,) f32 per-head decay coefficients
        scale: attention scale factor (default: 1.0)
        initial_state: (B, H, D, D) f32 initial state in BHVK layout or None
        output_final_state: whether to output final state
        chunk_size: chunk size (default: 64)

    Returns:
        (O, ht): output tensor (B,S,H,D) bf16, final state (B,H,D,D) f32 or None
    """
    B, S, H, D = Q.shape
    O = torch.zeros_like(Q)

    has_initial_state = initial_state is not None

    compiled_fn = _get_compiled_kernel(
        has_initial_state,
        output_final_state,
        H,
        D,
        scale,
        chunk_size,
    )

    if output_final_state:
        ht = torch.zeros(B, H, D, D, dtype=torch.float32, device=Q.device)
    else:
        ht = None

    # TVM-FFI: pass torch tensors directly; stream is auto-provided
    # by make_fake_stream(use_tvm_ffi_env_stream=True).
    # Dummy cu_seqlens/indices for non-varlen (never dereferenced).
    _dummy_cu = torch.zeros(2, dtype=torch.int32, device=Q.device)
    _dummy_idx = torch.zeros(1, dtype=torch.int32, device=Q.device)
    _dummy_workspace = torch.zeros(1, dtype=torch.int32, device=Q.device)
    compiled_fn(
        Q,
        K,
        V,
        O,
        decay,
        initial_state,
        ht,
        _dummy_cu,
        _dummy_idx,
        O,  # o_tensor (reuse O — never dereferenced for non-varlen)
        _dummy_workspace,
        (Int32(B), Int32(S)),
    )

    return O, ht


# ---------------------------------------------------------------------------
# Varlen Compile cache + TVM-FFI API
# ---------------------------------------------------------------------------

_varlen_kernel_cache: dict = {}


def _compile_single_variant_varlen(H, D, scale, chunk_size, persistent=True):
    """Compile one varlen kernel variant. Returns the compiled TVM-FFI callable.

    Varlen kernel always has initial state and output_final_state (INPLACE_UPDATE).
    """
    kernel_obj = LinearAttentionChunkwiseDecay(
        chunk_size=chunk_size,
        acc_dtype=cutlass.Float32,
        io_dtype=cutlass.BFloat16,
        has_initial_state=True,
        output_final_state=True,
        H=H,
        K=D,
        V=D,
        scale=scale,
        is_varlen=True,
        persistent=persistent,
        use_fast_math=USE_FAST_MATH,
    )

    sym_n = cute.sym_int()  # N: number of sequences
    sym_t = cute.sym_int()  # T: total packed tokens

    # Q, K, V, O: [1, T, H, D] row-major bf16
    # For varlen, B=1 in the physical tensor but we view as (T, D, H)
    q_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_t, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    k_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_t, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    v_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_t, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    o_fake = make_fake_compact_tensor(
        cutlass.BFloat16,
        (1, sym_t, H, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )

    # decay: (H,) float32
    decay_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (H,),
        assumed_align=128,
    )

    # State pool: [pool_size, H, D, D] float32 — always present for varlen
    # Use sym_n as pool dimension (actual pool may be larger, strides are correct)
    h0_fake = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_n, H, D, D),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    # final_state is same as initial_state for INPLACE_UPDATE
    ht_fake = h0_fake

    # cu_seqlens: [N+1] int32 — requires separate sym because size differs from N
    sym_cu = cute.sym_int()
    cu_seqlens_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_cu,),
        assumed_align=128,
    )

    # initial_state_indices: [N] int32
    sym_idx = cute.sym_int()
    indices_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_idx,),
        assumed_align=128,
    )

    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    # Workspace for persistent kernel atomic counter (int32, zeroed before each call)
    workspace_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=128,
    )

    compiled_fn = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        v_fake,
        o_fake,
        decay_fake,
        h0_fake,
        ht_fake,
        cu_seqlens_fake,
        indices_fake,
        o_fake,  # o_tensor for CopyUniversal tail store
        workspace_fake,
        (Int32(1), Int32(1)),  # dummy (N, T)
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_fn


def _get_compiled_kernel_varlen(H, D, scale, chunk_size, persistent=True):
    """Get a compiled varlen kernel with on-demand compilation.

    Cache key: (H, D, scale, chunk_size, persistent, USE_FAST_MATH)
    """
    key = (H, D, scale, chunk_size, persistent, USE_FAST_MATH)
    if key not in _varlen_kernel_cache:
        _varlen_kernel_cache[key] = _compile_single_variant_varlen(
            H,
            D,
            scale,
            chunk_size,
            persistent=persistent,
        )
    return _varlen_kernel_cache[key]


def lightning_attn_fwd_varlen(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    decay: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float = 1.0,
    state_pool: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    chunk_size: int = 64,
    persistent: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lightning Attention forward pass with varlen (packed variable-length sequences).

    Supports SGLang extend mode:
    - Variable-length sequences packed into a single tensor
    - State cache with indirect indexing (INPLACE_UPDATE)
    - Per-request initial state via state pool

    Supports non-aligned sequence lengths (tail chunks handled via CopyUniversal).

    Args:
        Q: (1, T, H, D) bf16 query — packed tokens from all sequences
        K: (1, T, H, D) bf16 key
        V: (1, T, H, D) bf16 value
        decay: (H,) f32 per-head decay coefficients
        cu_seqlens: (N+1,) int32 cumulative sequence lengths
        scale: attention scale factor (default: 1.0)
       state_pool: (pool_size, H, D, D) f32 state pool in BHVK layout, or None
            If None, a zero state pool is allocated with pool_size=N.
            States are updated in-place (INPLACE_UPDATE).
        initial_state_indices: (N,) int32 indices into state_pool per sequence.
            If None, defaults to arange(N).
        chunk_size: chunk size (default: 64)

    Returns:
        (O, state_pool): output tensor (1,T,H,D) bf16, updated state pool (pool_size,H,D,D) f32
    """
    _, T, H, D = Q.shape
    N = cu_seqlens.shape[0] - 1
    O = torch.zeros_like(Q)

    # Allocate state pool if not provided
    if state_pool is None:
        state_pool = torch.zeros(N, H, D, D, dtype=torch.float32, device=Q.device).transpose(-1,-2)

    # Default indices: identity mapping
    if initial_state_indices is None:
        initial_state_indices = torch.arange(N, dtype=torch.int32, device=Q.device)

    # Ensure int32 types
    cu_seqlens = cu_seqlens.to(torch.int32)
    initial_state_indices = initial_state_indices.to(torch.int32)

    compiled_fn = _get_compiled_kernel_varlen(H, D, scale, chunk_size, persistent=persistent)

    # Workspace for persistent kernel atomic counter (zeroed before each call)
    workspace = torch.zeros(1, dtype=torch.int32, device=Q.device)

    # TVM-FFI: pass torch tensors directly
    compiled_fn(
        Q,
        K,
        V,
        O,
        decay,
        state_pool,
        state_pool,  # initial_state = final_state = state_pool (INPLACE_UPDATE)
        cu_seqlens,
        initial_state_indices,
        O,  # o_tensor for CopyUniversal tail store
        workspace,
        (Int32(N), Int32(T)),
    )

    return O, state_pool


def main():
    """
    Example usage of LinearAttentionChunkwise with CuTe DSL
    """
    parser = argparse.ArgumentParser(description="Chunkwise Linear Attention with Headwise Decay")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=64, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size")
    parser.add_argument("--decay", type=float, default=0.95, help="Decay factor")
    parser.add_argument("--io_dtype", type=cutlass.dtype, default=cutlass.BFloat16, help="Input/output data type")
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32, help="Accumulation data type")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=1, help="Benchmark iterations")

    args = parser.parse_args()

    print("Running Chunkwise Linear Attention with CuTe DSL:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Head dimension: {args.head_dim}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Decay factor: {args.decay}")
    print(f"  IO dtype: {args.io_dtype}")
    print(f"  Accumulation dtype: {args.acc_dtype}")
    print(f"  Warmup iterations: {args.warmup_iterations}")
    print(f"  Benchmark iterations: {args.iterations}")

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    # Create inputs
    B, S, H, D = args.batch_size, args.seq_len, args.num_heads, args.head_dim

    # Input tensors in format [B, S, H, D]
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    # Per-head decay coefficients [H]
    decay = torch.full((H,), args.decay, device="cuda", dtype=torch.float32)

    scale = 1.0 / (D**0.5)

    # Compile with TVM-FFI cache (first call compiles, subsequent calls reuse)
    start_time = time.time()
    O, ht = lightning_attn_fwd(
        Q,
        K,
        V,
        decay,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        chunk_size=args.chunk_size,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation + first run time: {compilation_time:.4f} seconds")

    print(f"B, S, H, D: {(B, S, H, D)}")

    # Warmup (uses cached kernel — no recompilation)
    for _ in range(args.warmup_iterations):
        O, ht = lightning_attn_fwd(
            Q,
            K,
            V,
            decay,
            scale=scale,
            initial_state=None,
            output_final_state=True,
            chunk_size=args.chunk_size,
        )

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(args.iterations):
        O, ht = lightning_attn_fwd(
            Q,
            K,
            V,
            decay,
            scale=scale,
            initial_state=None,
            output_final_state=True,
            chunk_size=args.chunk_size,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\nExecution time: {elapsed * 1000 / args.iterations:.2f} ms (average over {args.iterations} iterations)")
    print("\nPASS")


if __name__ == "__main__":
    main()
