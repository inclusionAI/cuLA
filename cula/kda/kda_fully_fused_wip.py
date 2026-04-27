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
KDA (Kimi Delta Attention) using CuTe DSL [WIP]

Status: This module is a work in progress (WIP) and is not considered a stable production path yet.

This module implements chunkwise KDA with gate-based attention for
the NVIDIA Blackwell SM100 architecture using CUTE DSL.

The implementation supports:
- Chunkwise computation with M (intra-chunk attention matrix)
- Gate-based temporal modeling with g_cumsum
- WY-representation for efficient state updates (W, U matrices)
- Input/output format: [Batch, Sequence, Heads, Dim]

Step 1 Implementation (current):
- Gate processing: g -> g_cumsum (chunkwise cumulative sum)
- Elementwise operations: Q' = Q * exp(g_cumsum), K' = K * exp(g_cumsum), K'^T = K^T * exp(-g_cumsum)
- Validation against torch reference

Mathematical formulation (chunkwise KDA):
- M = Akk (intra-chunk attention with gate-based causal mask)
- M^{-1} via triangular solve
- W = M^{-1} @ (K * beta * exp(g))
- U = M^{-1} @ (V * beta)
- State update: S_new = exp(g_last) * S + W^T @ U
- Output: O = P @ V (where P includes both intra and inter contributions)
"""

import argparse
import time

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64
from fla.modules.l2norm import l2norm_fwd

from cula.utils import assert_blackwell

# Global debug switch - set to False to disable ALL print statements
# When False, cutlass.const_expr(PRINT_DEBUG) will eliminate code at compile time
PRINT_DEBUG = False
# Fine-grained debug switches for specific warp groups
ENABLE_MMA_WARP_PRINT = False  # MMA warp debug prints
ENABLE_CUDA_WG_PRINT = False  # CUDA warpgroup debug prints


class Constant:
    """Common constants used in KDA implementation."""

    WARP_SIZE = 32
    MAX_TMEM_COLS_SM100 = 512
    C = 64  # chunk size
    SC = 16  # subchunk size
    D = 128  # head dim
    HALF_D = 64  # half head dim for partitioned S2R
    SCALE = float(D) ** -0.5
    BK_SC = 64  # tile size in subchunk MMA


class MaskEnum:
    """Enumeration for different mask types."""

    NONE = 0
    PADDING = 1
    CAUSAL = 2


class KDAChunkwise:
    """
    Chunkwise KDA (Kimi Delta Attention) using CuTe DSL

    Implements KDA with gate-based attention and WY-representation.

    Step 1 (current): Gate processing with g_cumsum and elementwise operations
    Future steps: M computation, triangular solve, W/U, state updates

    Args:
        chunk_size: Size of each attention chunk (default: 64)
        qk_acc_dtype: Accumulator data type for QK computation (default: Float32)
        kv_acc_dtype: Accumulator data type for PV computation (default: Float32)
        io_dtype: Input/output data type (default: BFloat16)
    """

    def __init__(
        self,
        chunk_size: int = 64,
        qk_acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        kv_acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        scale: cutlass.Float32 = 1.0,
        safe_gate: bool = False,
        has_initial_state: bool = False,
        output_final_state: bool = False,
        is_varlen: bool = False,
        use_fast_math: bool = True,
        # num_regs_cuda: int = 248,
        num_regs_cuda: int = 224,
        num_regs_subchunk: int = 192,
        num_regs_others: int = 64,  # Optimized: best config from comprehensive sweep
    ):
        assert_blackwell()
        # make scale a constant
        self.scale = scale
        self.safe_gate = safe_gate
        self.has_initial_state = has_initial_state
        self.output_final_state = output_final_state
        self.is_varlen = is_varlen
        self.use_fast_math = use_fast_math

        self.chunk_size = chunk_size
        self.subchunk_size = 16
        self.qk_acc_dtype = qk_acc_dtype
        self.kv_acc_dtype = kv_acc_dtype
        self.pv_acc_dtype = kv_acc_dtype
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.mv_acc_stage = 1
        self.inverse_dtype = cutlass.Float16  # For inverse
        self.beta_dtype = cutlass.Float32

        # Register allocation configuration
        self.num_regs_cuda = num_regs_cuda
        self.num_regs_subchunk = num_regs_subchunk
        # Reg count for load, MMA, epilogue and empty warp, maybe tuned
        self.num_regs_others = num_regs_others

        self.threads_per_warp = 32

        # MMA tile shapes
        # C: 64, choose chunk size as 64 for enough spaces to do double buffering
        # Q: (64, 128)
        # K: (64, 128)
        # V: (64, 128)
        C, D = (Constant.C, Constant.D)
        HALF_D = Constant.HALF_D
        # (C, C, D)
        self.qk_mma_tiler = (C, C, D)  # (M, N, K)
        # Half-width tiler for partitioned S2R in elementwise gating
        self.qk_mma_tiler_half = (C, C, HALF_D)  # (M, N, K/2)
        self.kk_mma_tiler = (C, C, D)  # (M, N, K)
        # (D, C, C)
        self.vp_mma_tiler = (D, C, C)  # (M, N, K)
        self.mv_mma_tiler = (D, C, C)  # (M, N, K)
        # (D, D, C)
        self.kv_mma_tiler = (D, D, C)  # (M, N, K)
        # (D, C, D)
        # State as operand A since it's in TMEM
        # Q now as operand B
        self.sq_mma_tiler = (D, C, D)  # (M, N, K)
        self.ks_mma_tiler = (D, C, D)  # (M, N, K)

        # subchunk MMA
        SC, BK_SC = (Constant.SC, Constant.BK_SC)
        self.qk_kk_subchunk_mma_tiler = (SC, SC, BK_SC)  # (M, N, K)
        self.NK_SC = D // BK_SC  # number of iterations for MMA K dimension
        assert self.NK_SC == 2

        # one-cta cluster shape
        self.cluster_shape_mnk = (1, 1, 1)
        # For masking & decay.
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.cuda_subchunk_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.epilogue_warp_id = 10
        # NOTE: setmaxnreg is a warpgroup-wide instruction, so we force thread number to be multiply of warpgroup
        # https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg
        self.load_beta_warp_id = 11

        self.threads_per_warp = 32
        self.cuda_core_threads = self.threads_per_warp * (len(self.cuda_warp_ids))
        self.cuda_core_subchunk_threads = self.threads_per_warp * (len(self.cuda_subchunk_warp_ids))
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cuda_warp_ids,
                *self.cuda_subchunk_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.load_beta_warp_id,
            )
        )

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )

        self.cuda_wg_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.cuda_core_threads,
        )

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=32 * len([self.mma_warp_id]),
        )

        self.cuda_subchunk_wg_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.cuda_core_subchunk_threads,
        )

        self.buffer_align_bytes = 1024

    def get_config(self) -> dict:
        """
        Get current KDA configuration.

        Returns:
            dict: Configuration dictionary with kernel parameters
        """
        return {
            "chunk_size": self.chunk_size,
            "qk_acc_dtype": str(self.qk_acc_dtype),
            "kv_acc_dtype": str(self.kv_acc_dtype),
            "acc_dtype": str(self.acc_dtype),
            "io_dtype": str(self.io_dtype),
            "num_cuda_core_threads": self.cuda_core_threads,
            "threads_per_cta": self.threads_per_cta,
        }

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
    ):
        """Compute TMEM offsets for various tensors used in the kernel."""
        SM100_TMEM_CAPACITY_COLS = 512

        # (MMA, MMA_M, MMA_N)
        acc_shape_qk = tiled_mma_qk.partition_shape_C(tile_shape_mnk_qk[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAccQK_fake = tiled_mma_qk.make_fragment_C(cute.append(acc_shape_qk, acc_stages))
        num_qk_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccQK_fake)
        num_kk_acc_cols = num_qk_acc_cols

        acc_shape_pv = tiled_mma_pv.partition_shape_C(tile_shape_mnk_pv[:2])
        tCtAccPV_fake = tiled_mma_pv.make_fragment_C(cute.append(acc_shape_pv, acc_stages))
        num_pv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccPV_fake)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tCtAccPV_fake={tCtAccPV_fake}, num_pv_acc_cols={num_pv_acc_cols}")

        # No stage for linear state.
        acc_shape_kv = tiled_mma_kv.partition_shape_C(tile_shape_mnk_kv[:2])
        tCtAccKV_fake = tiled_mma_kv.make_fragment_C(cute.append(acc_shape_kv, 1))
        num_kv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccKV_fake)
        # Cannot reuse KV since we need to accumulate KV in FP32.
        # We setup a separated tmem space for KV16 as operand A for mma.
        num_kv16_acc_cols = num_kv_acc_cols // 2  # BF16 has half columns
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tCtAccKV_fake={tCtAccKV_fake}, num_kv_acc_cols={num_kv_acc_cols}, num_kv16_acc_cols={num_kv16_acc_cols}")

        acc_shape_sq = tiled_mma_sq.partition_shape_C(tile_shape_mnk_sq[:2])
        # No Stage for QS since state has no stages.
        tCtAccSQ_fake = tiled_mma_sq.make_fragment_C(cute.append(acc_shape_sq, 1))
        num_qs_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccSQ_fake)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tCtAccSQ_fake={tCtAccSQ_fake}, num_qs_acc_cols={num_qs_acc_cols}")

        num_qk_acc_cols_offset = 0
        num_pv_acc_cols_offset = num_qk_acc_cols_offset + num_qk_acc_cols
        num_kv_acc_cols_offset = num_pv_acc_cols_offset + num_pv_acc_cols
        num_kv16_acc_cols_offset = num_kv_acc_cols_offset + num_kv_acc_cols
        num_qs_acc_cols_offset = num_kv16_acc_cols_offset + num_kv16_acc_cols
        num_kk_acc_cols_offset = num_qs_acc_cols_offset + num_qs_acc_cols

        num_tmem_cols_total_tmp = num_kk_acc_cols_offset + num_kk_acc_cols
        # Turn num_tmem_cols_total to the nearest power of 2
        num_tmem_cols_total = 1
        while num_tmem_cols_total < num_tmem_cols_total_tmp:
            num_tmem_cols_total *= 2
        assert num_tmem_cols_total <= SM100_TMEM_CAPACITY_COLS

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"num_qk_acc_cols_offset: {num_qk_acc_cols_offset}")
            print(f"num_pv_acc_cols_offset: {num_pv_acc_cols_offset}")
            print(f"num_kv_acc_cols_offset: {num_kv_acc_cols_offset}")
            print(f"num_kv16_acc_cols_offset: {num_kv16_acc_cols_offset}")
            print(f"num_qs_acc_cols_offset: {num_qs_acc_cols_offset}")
            print(f"num_tmem_cols_total: {num_tmem_cols_total}")

        return (
            num_qk_acc_cols_offset,
            num_pv_acc_cols_offset,
            num_kv_acc_cols_offset,
            num_kv16_acc_cols_offset,
            num_qs_acc_cols_offset,
            num_kk_acc_cols_offset,
            num_tmem_cols_total,
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the linear attention kernel."""

        self.q_stage = 2
        self.k_stage = 2
        self.v_stage = 1
        self.o_stage = 2
        self.g_stage = 2  # Single stage for g (CUDA warp processes immediately)
        self.beta_stage = 1  # TODO: two stage ?
        self.q_k_scaled_stage = 1  # only single stage here due to smem limitation

        self.epi_stage = 2
        self.acc_stage = 2
        self.ks_stage = 1
        self.o_inter_stage = 1
        self.o_intra_stage = 2

    def _compute_grid(
        self,
        o_shape: cute.Shape,
        chunk_size: int,
    ) -> cute.Shape:
        """Compute tile scheduler parameters based on the chunk size and MMA tiler."""
        # (D, S, (H, B))
        return (
            # S / CHUNK
            # cute.ceil_div(o_shape[0], chunk_size),
            # For Loop to tile over chunk size,
            # TODO: varlen will make parallelism good enough
            1,
            # H
            cute.size(o_shape[2][0]),
            # B
            cute.size(o_shape[2][1]),
        )

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        g_iter: cute.Pointer,  # NEW: gate values
        o_iter: cute.Pointer,
        beta_iter: cute.Pointer,  # NEW: beta tensor [B, S, H]
        initial_state_iter: cute.Pointer,  # Initial state [B, H, D, D], float32 or nullptr
        final_state_iter: cute.Pointer,  # Final state [B, H, D, D], float32 or nullptr
        cu_seqlens_iter: cute.Pointer,  # Cumulative seq lengths [num_seqs+1], int32 (varlen)
        workspace_iter: cute.Pointer,  # Workspace buffer for TMA descriptor modification
        problem_size: tuple[Int32, Int32, Int32, Int32],  # (B/num_seqs, S/total_tokens, H, D)
        stream: cuda.CUstream,
        options=None,  # compile options
    ):
        """
        Execute the Chunkwise KDA operation on the provided tensors.

        Step 1: Gate processing with g_cumsum and elementwise operations

        Args:
            q_iter: Query tensor [B, S, H, D] or [1, total_tokens, H, D] for varlen
            k_iter: Key tensor [B, S, H, D] or [1, total_tokens, H, D] for varlen
            v_iter: Value tensor [B, S, H, D] or [1, total_tokens, H, D] for varlen
            g_iter: Gate tensor [B, S, H, D] or [1, total_tokens, H, D] for varlen
            o_iter: Output tensor [B, S, H, D] or [1, total_tokens, H, D] for varlen
            beta_iter: Beta tensor [B, S, H] or [1, total_tokens, H] for varlen
            initial_state_iter: Initial state [N, H, D, D] or nullptr (N=B or num_seqs)
            final_state_iter: Final state [N, H, D, D] or nullptr
            cu_seqlens_iter: Cumulative seq lengths [num_seqs+1], int32 (varlen only)
            workspace_iter: Workspace buffer for TMA descriptor modification (varlen tail tiles)
            problem_size: (N, S, H, D) where N=B or num_seqs, S=seq_len or total_tokens
            stream: CUDA stream
            options: compile options for the kernel
        """
        B, S, H, D = problem_size

        # Setup attributes
        self._setup_attributes()

        # TODO: try two-cta
        self.cta_group = tcgen05.CtaGroup.ONE

        # For varlen: B=num_seqs, S=total_tokens. Data tensors use data_B=1, state uses B=num_seqs.
        # For non-varlen: B=batch_size, S=seq_len. data_B=B.
        if cutlass.const_expr(self.is_varlen):
            data_B = 1
        else:
            data_B = B

        # It's ok since torch tensor is row major, hence we've layout=(B,S,H,D):(DHS, DH, D, 1).
        # Below are just permutation tricks to ease the later processing.
        # For varlen, data_B=1 since tokens are concatenated in the S dimension.
        q_layout = cute.make_layout(
            (S, D, (H, data_B)),
            stride=(D * H, 1, (D, D * H * S)),
        )
        q = cute.make_tensor(q_iter, q_layout)
        # (S, D, (H, data_B))
        k_layout = cute.make_layout(
            (S, D, (H, data_B)),
            stride=(D * H, 1, (D, D * H * S)),
        )
        k = cute.make_tensor(k_iter, k_layout)
        kt_layout = cute.make_layout(
            (D, S, (H, data_B)),
            stride=(1, D * H, (D, D * H * S)),
        )
        kt = cute.make_tensor(k_iter, kt_layout)
        # v
        v_layout = cute.make_layout(
            (D, S, (H, data_B)),
            stride=(1, D * H, (D, D * H * S)),
        )
        v = cute.make_tensor(v_iter, v_layout)

        # g (gate) - NEW for KDA, same layout as Q/K
        g_layout = cute.make_layout(
            (S, D, (H, data_B)),
            stride=(D * H, 1, (D, D * H * S)),
        )
        g = cute.make_tensor(g_iter, g_layout)

        # beta - NEW for KDA, shape (B, S, H) or (1, total_tokens, H) for varlen
        beta_layout = cute.make_layout(
            (S, (H, data_B)),
            stride=(H, (1, H * S)),
        )
        beta = cute.make_tensor(beta_iter, beta_layout)

        o_layout = cute.make_layout(
            (D, S, (H, data_B)),
            stride=(1, D * H, (D, D * H * S)),
        )
        o = cute.make_tensor(o_iter, o_layout)

        # Initial state / final state: [N, H, D, D] stored as row-major
        # N = B (non-varlen) or num_seqs (varlen). Always uses B from problem_size.
        fstate_layout = cute.make_layout(
            (D, D, (H, B)),
            stride=(1, D, (D * D, D * D * H)),
        )
        initial_state = cute.make_tensor(initial_state_iter, fstate_layout)
        final_state = cute.make_tensor(final_state_iter, fstate_layout)

        # cu_seqlens tensor for varlen
        if cutlass.const_expr(self.is_varlen):
            cu_seqlens = cute.make_tensor(cu_seqlens_iter, cute.make_layout((B + 1,)))
        else:
            cu_seqlens = cute.make_tensor(cu_seqlens_iter, cute.make_layout((2,)))

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.g_dtype = g.element_type  # NEW for KDA
        self.o_dtype = o.element_type

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.g_major_mode = utils.LayoutEnum.from_tensor(g).mma_major_mode()  # NEW for KDA
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
            self.qk_acc_dtype,
            self.cta_group,
            self.qk_mma_tiler[:2],
        )
        kk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            # SHOULE BE both K-major
            self.k_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.kk_mma_tiler[:2],
        )
        # V^T*K, majorness
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.v_major_mode,
            self.k_major_mode_kv,
            self.kv_acc_dtype,
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
        ks_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            # State is in TMEM, always K major, TODO
            tcgen05.OperandMajorMode.K,
            # State is in TMEM, always K major, TODO
            self.k_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.ks_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )

        m_major_mode = tcgen05.OperandMajorMode.K
        mv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_major_mode,
            m_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mv_mma_tiler[:2],
        )

        p_major_mode = tcgen05.OperandMajorMode.K
        vp_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_major_mode,
            p_major_mode,
            self.pv_acc_dtype,
            self.cta_group,
            self.vp_mma_tiler[:2],
        )

        (
            self.tmem_qk_cols_offset,
            self.tmem_pv_cols_offset,
            self.tmem_kv_cols_offset,
            self.tmem_kv16_cols_offset,
            self.tmem_sq_cols_offset,
            self.tmem_kk_cols_offset,
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
        q_k_scaled_smem_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_k_scaled_stage,
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
        # G (gate) - NEW for KDA
        # Use same layout as Q since g has same shape and memory layout as Q
        # This ensures TMA compatibility
        # ((MMA_ATOM_M, MMA_ATOM_K), MMA_M, MMA_K, STAGES)
        g_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.g_dtype,
            self.g_stage,
        )
        if PRINT_DEBUG:
            print(f"g_smem_layout_staged: {g_smem_layout_staged}")
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
        # For M, same shape as KK^T
        m_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            # SMEM M = KK^T should be row-major
            utils.LayoutEnum.ROW_MAJOR,
            self.kk_mma_tiler[:2],
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
        kv_k_smem_layout = cute.select(kv_k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            kt,
            kv_k_smem_layout,
            self.kv_mma_tiler,
            kv_tiled_mma,
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
        # TMA load for G (gate) - NEW for KDA
        # Use same TMA atom as Q since g has same layout as Q
        g_smem_layout = cute.select(g_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_g, tma_tensor_g = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            g,
            g_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        if PRINT_DEBUG:
            print(f"tma_atom_g: {cute.pretty_str(tma_atom_g)}")
            print(f"g_smem_layout: {cute.pretty_str(g_smem_layout)}")

        # NOTE: G's last row will be extracted from sG in CUDA warp after TMA load
        # No separate TMA needed for G last row - we extract it from the full G tile

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
        g_copy_size = cute.size_in_bytes(self.g_dtype, g_smem_layout)  # NEW for KDA
        if PRINT_DEBUG:
            print(
                f"q_copy_size: {q_copy_size}, k_copy_size: {k_copy_size}, v_copy_size: {v_copy_size}, g_copy_size: {g_copy_size}"
            )
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_k_bytes = k_copy_size
        # self.tma_copy_v_bytes = v_copy_size
        self.tma_copy_v_bytes = k_copy_size
        self.tma_copy_g_bytes = g_copy_size  # NEW for KDA

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

        beta_layout = cute.make_layout((Constant.C, self.beta_stage), stride=(1, Constant.C))
        g_last_layout = cute.make_layout((Constant.D, self.g_stage), stride=(1, Constant.D))

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            # Inputs
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]  # type: ignore
            load_q2_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]  # type: ignore
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]  # type: ignore
            load_k2_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]  # type: ignore
            load_kt2_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]  # type: ignore
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # type: ignore
            load_v2_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # type: ignore
            load_v3_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # type: ignore
            pseudo_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # type: ignore
            end_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # type: ignore
            load_g_mbar_ptr: cute.struct.MemRange[Int64, self.g_stage * 2]  # type: ignore  # NEW for KDA
            load_beta_mbar_ptr: cute.struct.MemRange[Int64, self.beta_stage * 2]  # type: ignore
            load_q_scaled_mbar_ptr: cute.struct.MemRange[Int64, self.q_k_scaled_stage * 2]  # type: ignore
            load_k_scaled_mbar_ptr: cute.struct.MemRange[Int64, self.q_k_scaled_stage * 2]  # type: ignore
            load_k_scaled2_mbar_ptr: cute.struct.MemRange[Int64, self.q_k_scaled_stage * 2]  # type: ignore
            # KDA gating sync: CUDA warp notifies MMA warp that Q'/K' are ready
            kda_gate_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]  # type: ignore  # NEW for KDA
            # Masking
            s_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            # MMA
            kk_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            # Write to SMEM
            smem_kk_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            # KV
            kv_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
            kv16_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
            p_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            o_intra_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
            ks_mbar_ptr: cute.struct.MemRange[Int64, self.ks_stage * 2]  # type: ignore
            o_inter_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
            smem_o_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # type: ignore
            kv_decay_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
            kv_decay_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # type: ignore
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
            # Store the exp scaled Q/K for MMA
            sQ_K_scaled: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_k_scaled_smem_layout)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # G (gate) - NEW for KDA
            sG: cute.struct.Align[
                cute.struct.MemRange[self.g_dtype, cute.cosize(g_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Store QK
            sP: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(p_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Store KK
            sM: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(m_smem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Store last row of exp(g) for KDA
            # 4B * 128 = 512B
            sG_last: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(g_last_layout)],  # type: ignore
                self.buffer_align_bytes,
            ]
            # Store the beta chunk
            # 4B * 64 * 2stage = 512B
            sBeta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_layout)],  # type: ignore
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        if PRINT_DEBUG:
            print(f"size of storage: {SharedStorage.__sizeof__()}")
            print(f"m_smem_layout_staged: {m_smem_layout_staged}")

        if cutlass.const_expr(self.is_varlen):
            self.grid = (1, H, B)
            # TensorMapManager for TMA descriptor modification in varlen tail tiles
            self._tensormap_mgr = utils.TensorMapManager(utils.TensorMapUpdateMode.GMEM, 128)
        else:
            self.grid = self._compute_grid(
                # (D, S, (H, B))
                o_shape=cute.shape(o),
                chunk_size=self.chunk_size,
            )
        if PRINT_DEBUG:
            print(f"grid: {self.grid}")

        self.kernel(
            qk_tiled_mma,
            kk_tiled_mma,
            kv_tiled_mma,
            vp_tiled_mma,
            sq_tiled_mma,
            ks_tiled_mma,
            mv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_kt,
            tma_tensor_kt,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_g,  # NEW for KDA
            tma_tensor_g,  # NEW for KDA
            tma_atom_o,
            tma_tensor_o,
            beta,  # NEW for KDA
            q_smem_layout_staged,
            k_smem_layout_staged,
            kv_k_smem_layout_staged,
            v_smem_layout_staged,
            g_smem_layout_staged,  # NEW for KDA
            o_smem_layout_staged,
            p_smem_layout_staged,
            m_smem_layout_staged,
            g_last_layout,
            beta_layout,
            state_tmem_layout_staged,
            initial_state,
            final_state,
            cu_seqlens,
            o,
            workspace_iter,
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
        kk_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        vp_tiled_mma: cute.TiledMma,
        sq_tiled_mma: cute.TiledMma,
        ks_tiled_mma: cute.TiledMma,
        mv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        tma_tensor_kt: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_g: cute.CopyAtom,  # NEW for KDA
        tma_tensor_g: cute.Tensor,  # NEW for KDA
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        beta: cute.Tensor,  # NEW for KDA - shape (S, (H, B))
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        kv_k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        g_smem_layout_staged: cute.ComposedLayout,  # NEW for KDA
        o_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        m_smem_layout_staged: cute.ComposedLayout,
        g_last_layout: cute.Layout,
        beta_layout: cute.Layout,
        state_tmem_layout_staged: cute.ComposedLayout,
        initial_state: cute.Tensor,  # (D, D, (H, B)), float32
        final_state: cute.Tensor,  # (D, D, (H, B)), float32
        cu_seqlens: cute.Tensor,  # int32 tensor for varlen
        o_gmem: cute.Tensor,  # raw GMEM output tensor (D, S, (H, data_B)) for tail tile handling
        workspace_iter: cute.Pointer,  # workspace buffer for TMA descriptor modification
        problem_size: tuple[Int32, Int32, Int32, Int32],  # (B, S, H, D)
    ):
        """
        KDA Kernel - Step 1: Gate processing

        Current implementation:
        - Load g via TMA
        - Compute g_cumsum (chunkwise cumulative sum)
        - Apply elementwise: Q*exp(g), K*exp(g), K^T*exp(-g)
        - Store outputs for validation

        Future: M computation, triangular solve, W/U, state updates
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_g)  # NEW for KDA
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # Allocate shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len([*self.cuda_warp_ids, *self.cuda_subchunk_warp_ids])
            ),  # CUDA cores will consume
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
        ).make_participants()
        load_q2_producer, load_q2_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(32 * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_q2_mbar_ptr.data_ptr(),
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len([*self.cuda_warp_ids, *self.cuda_subchunk_warp_ids])
            ),  # CUDA cores will consume
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
        ).make_participants()
        load_k2_producer, load_k2_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(32 * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_k2_mbar_ptr.data_ptr(),
        ).make_participants()
        load_kt2_producer, load_kt2_consumer = pipeline.PipelineAsyncUmma.create(
            # TODO: add assert that g and k have the same stage count
            num_stages=self.g_stage,
            producer_group=make_thread_cooperative_group(32 * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_kt2_mbar_ptr.data_ptr(),
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cuda_warp_ids)),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
        ).make_participants()
        load_v2_producer, load_v2_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(32 * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_v2_mbar_ptr.data_ptr(),
        ).make_participants()
        pseudo_v_producer, pseudo_v_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.pseudo_v_mbar_ptr.data_ptr(),
        ).make_participants()
        load_v3_producer, load_v3_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(32 * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_v3_mbar_ptr.data_ptr(),
        ).make_participants()
        end_v_producer, end_v_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(32 * len(self.cuda_warp_ids)),
            barrier_storage=storage.end_v_mbar_ptr.data_ptr(),
        ).make_participants()
        # G (gate/g_cumsum) - NEW for KDA
        load_g_producer, load_g_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.g_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len([*self.cuda_warp_ids, *self.cuda_subchunk_warp_ids])
            ),  # CUDA cores will consume
            tx_count=self.tma_copy_g_bytes,
            barrier_storage=storage.load_g_mbar_ptr.data_ptr(),
        ).make_participants()
        load_beta_producer, load_beta_consumer = pipeline.PipelineAsync.create(
            num_stages=self.beta_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len([self.load_beta_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * (len(self.cuda_warp_ids) + len(self.cuda_subchunk_warp_ids))
                if not self.safe_gate
                else self.threads_per_warp * len(self.cuda_subchunk_warp_ids)
            ),
            barrier_storage=storage.load_beta_mbar_ptr.data_ptr(),
        ).make_participants()
        # for Q/K prologue and Q@S, K@S, K^T@NewV MMA
        load_q_scaled_producer, load_q_scaled_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.q_k_scaled_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_q_scaled_mbar_ptr.data_ptr(),
        ).make_participants()
        load_k_scaled_producer, load_k_scaled_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.q_k_scaled_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_k_scaled_mbar_ptr.data_ptr(),
        ).make_participants()
        load_k_scaled2_producer, load_k_scaled2_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.q_k_scaled_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.load_k_scaled2_mbar_ptr.data_ptr(),
        ).make_participants()

        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.s_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_kk_producer, mma_kk_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.kk_mbar_ptr.data_ptr(),
        ).make_participants()
        # Notify cuda core to convert 32-bit accumulator to 16-bit
        kv_producer, kv_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                len([self.mma_warp_id]),
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.kv_mbar_ptr.data_ptr(),
        ).make_participants()
        # Notify mma warp that 16bit state is ready for mma as operand A
        kv16_producer, kv16_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                32 * len(self.cuda_warp_ids),
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.kv16_mbar_ptr.data_ptr(),
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.acc_stage,  # TODO: check p stages
            producer_group=make_thread_cooperative_group(
                # FIXME: change to only subchunk warps as producer when subchunk is ready
                self.threads_per_warp * (len(self.cuda_warp_ids) + len(self.cuda_subchunk_warp_ids))
                if not self.safe_gate
                else self.threads_per_warp * len(self.cuda_subchunk_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mbar_ptr.data_ptr(),
        ).make_participants()
        smem_kk_producer, smem_kk_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * (len(self.cuda_warp_ids) + len(self.cuda_subchunk_warp_ids))
                if not self.safe_gate
                else self.threads_per_warp * len(self.cuda_subchunk_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.smem_kk_mbar_ptr.data_ptr(),
        ).make_participants()
        o_intra_producer, o_intra_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.o_intra_mbar_ptr.data_ptr(),
        ).make_participants()
        ks_producer, ks_consumer = pipeline.PipelineUmmaAsync.create(
            # NO STAGE for Kg * STATE
            num_stages=1,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.ks_mbar_ptr.data_ptr(),
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
        # T2R & R2T sync in S decay
        kv_decay_producer, kv_decay_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.kv_decay_mbar_ptr.data_ptr(),
        ).make_participants()

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
        tmem_ptr_base = tmem.retrieve_ptr(self.qk_acc_dtype)

        # NOTE: only used for safe_gate=True with separate scaled Q/K smem storage
        k_smem_mma_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.q_k_scaled_stage,
        )
        kv_k_smem_mma_layout_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.k_dtype,
            self.q_k_scaled_stage,
        )
        # Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, STAGE_Q)
        # sQ: ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)>
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        q_as_b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            sq_tiled_mma,
            self.sq_mma_tiler,
            self.q_dtype,
            self.q_stage if not self.safe_gate else self.q_k_scaled_stage,
        )
        sQ_sel = storage.sQ if not self.safe_gate else storage.sQ_K_scaled
        sQ_sq = sQ_sel.get_tensor(q_as_b_smem_layout_staged.outer, swizzle=q_as_b_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, STAGE_K)
        # sK: tensor<ptr<bf16, smem, align<1024>, S<3,4,3>> o
        # ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)>
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        # NOTE: reuse same smem as sK
        sK_g = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sK_kv = storage.sQ_K_scaled.get_tensor(kv_k_smem_mma_layout_staged.outer, swizzle=kv_k_smem_mma_layout_staged.inner)
        sK_ks = storage.sQ_K_scaled.get_tensor(k_smem_mma_layout_staged.outer, swizzle=k_smem_mma_layout_staged.inner)
        # NOTE: reuse same smem as sG
        sK_neg_g_f32 = storage.sG.get_tensor(
            # kv_k_smem_layout_staged.outer, swizzle=kv_k_smem_layout_staged.inner
            # NOTE: same swizzle atom (k-major) as k_smem_layout_staged
            k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
        )
        # NOTE: recast as bf16 since operand B is BF16
        # CRITICAL FIX: sK_neg_g's stage stride must match sG's byte stride
        # sG (F32) has stage stride = 8192 elements = 32768 bytes
        # sK_neg_g (BF16) must have stage stride = 32768 bytes = 16384 BF16 elements
        # Original bug: using sK_g.layout which has stage stride = 8192 BF16 elements = 16384 bytes
        # This caused sK_neg_g stage 1 to overlap with sG stage 0's second half!

        # Get the base layout from sK_g but double the stage stride
        sK_g_outer = sK_g.layout
        # Create new layout with corrected stage stride (16384 BF16 elements instead of 8192)
        # sK_g layout is: ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)
        # We need: ((64,16),1,(4,2),2):((64,1),0,(16,4096),16384)
        sK_neg_g_layout = cute.make_layout(
            sK_g_outer.shape,
            stride=(*sK_g_outer.stride[:-1], sK_g_outer.stride[-1] * 2),  # Double the stage stride
        )
        sK_neg_g = cute.make_tensor(
            cute.recast_ptr(sK_neg_g_f32.iterator, swizzle_=k_smem_layout_staged.inner, dtype=self.io_dtype),
            layout=sK_neg_g_layout,
        )

        # Same fix for sK_neg_g_b
        sK_neg_g_b_outer = kv_k_smem_layout_staged.outer
        sK_neg_g_b_layout = cute.make_layout(
            sK_neg_g_b_outer.shape,
            stride=(*sK_neg_g_b_outer.stride[:-1], sK_neg_g_b_outer.stride[-1] * 2),  # Double the stage stride
        )
        sK_neg_g_b = cute.make_tensor(
            cute.recast_ptr(sK_neg_g_f32.iterator, swizzle_=kv_k_smem_layout_staged.inner, dtype=self.io_dtype),
            layout=sK_neg_g_b_layout,
        )
        # sK_neg_g = cute.make_tensor(
        #     cute.recast_ptr(
        #         sK_neg_g_f32.iterator,
        #         swizzle_=k_smem_layout_staged.inner,
        #         dtype=self.io_dtype),
        #     layout=sK_neg_g_f32.layout)
        if PRINT_DEBUG:
            print(f"sK_neg_g: {cute.pretty_str(sK_neg_g)}")
            print(f"sK_neg_g_b: {cute.pretty_str(sK_neg_g_b)}")
            print(f"sK_g: {cute.pretty_str(sK_g)}")
        # (((64,2),16),1,4,2):(((1,4096),64),0,1024,8192)>
        sV = storage.sV.get_tensor(v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner)
        # G (gate/g_cumsum) - NEW for KDA
        sG = storage.sG.get_tensor(g_smem_layout_staged.outer, swizzle=g_smem_layout_staged.inner)
        # No swizzling for last row of exp(G)
        sG_last = self.get_smem_tensor_sG_last(storage, g_last_layout)
        if PRINT_DEBUG:
            print(f"sG_last: {sG_last}")

        # NOTE: optimize swizzle
        sBeta = storage.sBeta.get_tensor(beta_layout, swizzle=None)
        if PRINT_DEBUG:
            print(f"sBeta: {sBeta}")

        # (MMA, MMA_N, MMA_K, STAGE)
        sP = storage.sP.get_tensor(p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE_O)
        sO = storage.sO.get_tensor(o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)
        # sM = storage.sM.get_tensor(
        #     m_smem_layout_staged.outer, swizzle=sM_swizzle
        # )
        sM = storage.sM.get_tensor(m_smem_layout_staged.outer, swizzle=m_smem_layout_staged.inner)
        # ROW MAJOR
        sM_flat_layout = cute.make_layout((64, 64, self.k_stage), stride=(64, 1, 4096))
        sM_flat = storage.sM.get_tensor(
            sM_flat_layout,
            swizzle=m_smem_layout_staged.inner,
        )
        sM_opB = storage.sM.get_tensor(p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner)
        sM_f16 = cute.make_tensor(
            cute.recast_ptr(sM.iterator, swizzle_=m_smem_layout_staged.inner, dtype=cutlass.Float16), layout=sM.layout
        )
        sM_f16_flat = cute.make_tensor(
            cute.recast_ptr(sM_flat.iterator, swizzle_=m_smem_layout_staged.inner, dtype=cutlass.Float16),
            layout=sM_flat.layout,
        )

        # (MMA, MMA_M, MMA_K, STAGE_O)

        # NOTE: row major has the same majorness as mma operand B
        qk_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            # utils.LayoutEnum.from_tensor(sP),
            self.qk_mma_tiler[:2],
            self.acc_stage,
        )
        sQK = storage.sP.get_tensor(
            qk_smem_layout_staged.outer,
            swizzle=qk_smem_layout_staged.inner,
        )
        # ROW MAJOR
        sQK_flat_layout = cute.make_layout((64, 64, self.q_stage), stride=(64, 1, 4096))
        sQK_flat = storage.sP.get_tensor(
            sQK_flat_layout,
            swizzle=qk_smem_layout_staged.inner,
        )
        # ROW MAJOR
        # TODO:
        v_smem_layout_epi = sm100_utils.make_smem_layout_epi(
            self.v_dtype,
            # utils.LayoutEnum.ROW_MAJOR,
            utils.LayoutEnum.COL_MAJOR,
            (Constant.D, Constant.C),
            self.v_stage,
        )
        v_smem_layout_coalesce = cute.coalesce(
            v_smem_layout_epi,
            target_profile=(1, 1, 1),
        )
        sV_flat_s2r = storage.sV.get_tensor(v_smem_layout_coalesce.outer, swizzle=v_smem_layout_coalesce.inner)
        sV_epi = storage.sV.get_tensor(v_smem_layout_epi.outer, swizzle=v_smem_layout_epi.inner)

        # ((64,16),1,(4,2),2):
        # ((64,1),0,(16,4096),8192)>
        sK_flat_layout = cute.make_layout((64, (64, 2), self.k_stage), stride=(64, (1, 4096), 8192))
        sK_flat = storage.sK.get_tensor(
            sK_flat_layout,
            swizzle=k_smem_layout_staged.inner,
        )

        # Q and K flat layout for s2r - similar to g_smem_layout_epi
        # Q/K: (C, D) = (64, 128), layout is row-major, 2 stages
        # Use make_smem_layout_epi to create compatible layout
        q_smem_layout_epi = sm100_utils.make_smem_layout_epi(
            self.q_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (Constant.C, Constant.D),
            self.q_stage,
        )
        q_smem_layout_coalesce = cute.coalesce(
            q_smem_layout_epi,
            target_profile=(1, 1, 1),
        )
        sQ_flat = storage.sQ.get_tensor(q_smem_layout_coalesce.outer, swizzle=q_smem_layout_coalesce.inner)

        q_k_scaled_layout_epi = sm100_utils.make_smem_layout_epi(
            self.q_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (Constant.C, Constant.D),
            self.q_k_scaled_stage,
        )
        q_k_scaled_smem_layout_coalesce = cute.coalesce(
            q_k_scaled_layout_epi,
            target_profile=(1, 1, 1),
        )
        sQ_K_scaled_flat = storage.sQ_K_scaled.get_tensor(
            q_k_scaled_smem_layout_coalesce.outer, swizzle=q_k_scaled_smem_layout_coalesce.inner
        )

        k_smem_layout_epi = sm100_utils.make_smem_layout_epi(
            self.k_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (Constant.C, Constant.D),
            self.k_stage,
        )
        k_smem_layout_coalesce = cute.coalesce(
            k_smem_layout_epi,
            target_profile=(1, 1, 1),
        )
        sK_flat_s2r = storage.sK.get_tensor(k_smem_layout_coalesce.outer, swizzle=k_smem_layout_coalesce.inner)

        sG_flat_s2r_f32_fake = storage.sG.get_tensor(k_smem_layout_coalesce.outer, swizzle=k_smem_layout_coalesce.inner)
        # CRITICAL FIX: When recasting F32 to BF16, we must double the stage stride
        # so that the byte offset remains the same.
        # F32 stage stride = 8192 elements = 32768 bytes
        # BF16 stage stride should = 16384 elements = 32768 bytes
        k_smem_layout_bf16_outer = k_smem_layout_coalesce.outer
        k_smem_layout_bf16_fixed = cute.make_layout(
            k_smem_layout_bf16_outer.shape,
            stride=(*k_smem_layout_bf16_outer.stride[:-1], k_smem_layout_bf16_outer.stride[-1] * 2),
        )
        sG_flat_bf16 = cute.make_tensor(
            cute.recast_ptr(sG_flat_s2r_f32_fake.iterator, swizzle_=k_smem_layout_coalesce.inner, dtype=self.io_dtype),
            layout=k_smem_layout_bf16_fixed,
        )

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"sQ: {cute.pretty_str(sQ)}")
            print(f"sK: {cute.pretty_str(sK)}")
            print(f"sV: {cute.pretty_str(sV)}")
            print(f"sO: {cute.pretty_str(sO)}")
            print(f"sP: {cute.pretty_str(sP)}")
            print(f"sQK: {cute.pretty_str(sQK)}")

        (_, hidx, bidx) = cute.arch.block_idx()
        B, S, H, D = problem_size
        C = self.chunk_size

        # Varlen: compute per-CTA sequence boundary and domain offsets
        if cutlass.const_expr(self.is_varlen):
            tok_offset = cu_seqlens[bidx]
            seq_len = cu_seqlens[bidx + 1] - tok_offset
            data_bidx = Int32(0)
        else:
            tok_offset = Int32(0)
            seq_len = S
            data_bidx = bidx

        # -------------------------------------------------------------
        # Make fragments for MMAs.

        # Make fragments/tmem for QK MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCrQ, tCrK, tCtAccQK = self.mma_partition_ss(
            qk_tiled_mma,
            self.qk_mma_tiler,
            sQ,
            sK_neg_g,
            tmem_ptr_base + self.tmem_qk_cols_offset,
            self.acc_stage,
        )

        # Make fragments/tmem for QK MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCrKG, tCrKNegG, tCtAccKK = self.mma_partition_ss(
            kk_tiled_mma,
            self.kk_mma_tiler,
            sK_g,
            sK_neg_g,
            tmem_ptr_base + self.tmem_kk_cols_offset,
            self.acc_stage,
        )

        # Make fragments/tmem for KV MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCrV, tCrK_kv, tCtAccKV = self.mma_partition_ss(
            kv_tiled_mma,
            self.kv_mma_tiler,
            sV,
            sK_neg_g_b if not self.safe_gate else sK_kv,
            tmem_ptr_base + self.tmem_kv_cols_offset,
            1,  # NOTE: no stage for state accum
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

        # REUSE PV TMEM FOR KS MMA
        tCrState_KS, tCrK_KS, tCtAccKS = self.mma_partition_ts(
            tiled_mma=ks_tiled_mma,
            tile_shape_mnk=self.ks_mma_tiler,
            a_tmem_layout=state_tmem_layout_staged,
            smem_b=sK_g if not self.safe_gate else sK_ks,
            tmem_a_ptr=tmem_ptr_base + self.tmem_kv16_cols_offset,
            tmem_acc_ptr=tmem_ptr_base + self.tmem_pv_cols_offset,
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

        # Make fragments/tmem for MV MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        if PRINT_DEBUG:
            print(f"sP: {cute.pretty_str(sP)}")
            print(f"sM: {cute.pretty_str(sM)}")
        tCrV_corr, tCrM, tCtAccMV = self.mma_partition_ss(
            mv_tiled_mma,
            self.mv_mma_tiler,
            sV,
            sM_opB,
            tmem_ptr_base + self.tmem_pv_cols_offset,
            self.mv_acc_stage,
        )

        # -------------------------------------------------------
        # ((SWIZZLE_ATOM_M, REST_M), (SWIZZLE_ATOM_N, REST_N), (1, STAGES))
        g_smem_layout_epi = sm100_utils.make_smem_layout_epi(
            self.g_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            # G SMEM has the shape of
            (Constant.C, Constant.D),
            self.g_stage,
        )
        # (C, (SWIZZLE_ATOM_N, REST_N), STAGES)
        g_smem_layout_coalesce = cute.coalesce(
            g_smem_layout_epi,
            target_profile=(1, 1, 1),
        )
        # ROW MAJOR
        sG_flat = storage.sG.get_tensor(g_smem_layout_coalesce.outer, swizzle=g_smem_layout_coalesce.inner)
        # HALF SPACE - CRITICAL FIX: Double the stage stride for BF16
        sG_flat_layout = sG_flat.layout
        sG_flat_bf16_layout = cute.make_layout(
            sG_flat_layout.shape, stride=(*sG_flat_layout.stride[:-1], sG_flat_layout.stride[-1] * 2)
        )
        sG_flat_as_bf16 = cute.make_tensor(cute.recast_ptr(sG_flat.iterator, dtype=self.io_dtype), layout=sG_flat_bf16_layout)
        if PRINT_DEBUG:
            print(f"sG_flat: {cute.pretty_str(sG_flat)}")
            print(f"sG_flat_as_bf16: {cute.pretty_str(sG_flat_as_bf16)}")
            print(f"g_smem_layout_epi: {g_smem_layout_epi}")
            print(f"g_smem_layout_coalesce: {g_smem_layout_coalesce}")

        # ///////////////////////////////////////////////////////////////////////////////
        # LOAD WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            # Apply domain_offset for varlen TMA tensors
            tma_tensor_q_v = tma_tensor_q
            tma_tensor_k_v = tma_tensor_k
            tma_tensor_v_v = tma_tensor_v
            tma_tensor_g_v = tma_tensor_g
            if cutlass.const_expr(self.is_varlen):
                tma_tensor_q_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_q)
                tma_tensor_k_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_k)
                tma_tensor_v_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_v)
                tma_tensor_g_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_g)

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), TILES_N, TILES_K)
            tQsQ, tQgQ = self.tma_partition_for_mma_operand(
                tma_atom_q,
                tma_tensor_q_v,
                sQ,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="A",
                debug_name="Q",
                batch_idx=data_bidx,
            )

            tKsK, tKgK = self.tma_partition_for_mma_operand(
                tma_atom_k,
                tma_tensor_k_v,
                sK,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="B",
                debug_name="K",
                batch_idx=data_bidx,
            )

            tVsV, tVgV = self.tma_partition_for_mma_operand(
                tma_atom_v,
                tma_tensor_v_v,
                sV,
                self.vp_mma_tiler,
                vp_tiled_mma,
                operand_mode="A",
                debug_name="V",
                batch_idx=data_bidx,
            )

            # G (gate) - NEW for KDA
            tGsG, tGgG = self.tma_partition_for_mma_operand(
                tma_atom_g,
                tma_tensor_g_v,
                sG,
                self.qk_mma_tiler,  # Same as Q
                qk_tiled_mma,
                operand_mode="A",
                debug_name="G",
                batch_idx=data_bidx,
            )

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tKsK={tKsK}")
                print(f"tKgK={tKgK}")
                print(f"tVsV={tVsV}")
                print(f"tVgV={tVgV}")

            for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                # Chunk iterate over TILES_M, TILES_K is 1 in our case since max D is 128
                idx = chunk_start // C
                should_debug = PRINT_DEBUG and tidx == warp_idx * 32 and hidx == 0 and bidx == 0

                # Gi (gate/g_cumsum) - NEW for KDA
                g_handle = load_g_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_g,
                    src=tGgG[None, idx, 0],
                    dst=tGsG[None, g_handle.index],
                    tma_bar_ptr=g_handle.barrier,
                )

                # Qi
                # SRC: ((ATOM_V, REST_V), TILES_M, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                q_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_q,
                    src=tQgQ[None, idx, 0],  # source
                    dst=tQsQ[None, q_handle.index],  # which stage
                    tma_bar_ptr=q_handle.barrier,
                )

                # Ki
                # SRC: ((ATOM_V, REST_V), TILES_N, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                k_handle = load_k_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_k,
                    src=tKgK[None, idx, 0],
                    dst=tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )

                # Vi
                # SRC: ((ATOM_V, REST_V), TILES_M, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                v_handle = load_v_producer.acquire_and_advance()
                if cutlass.const_expr(PRINT_DEBUG) and should_debug:
                    cute.printf("TMA v producer idx={}, v_handle={}", idx, v_handle.index)
                cute.copy(
                    atom=tma_atom_v,
                    src=tVgV[None, 0, idx],
                    dst=tVsV[None, v_handle.index],
                    tma_bar_ptr=v_handle.barrier,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        # COMPUTE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            should_debug = PRINT_DEBUG and tidx == warp_idx * 32 and hidx == 0 and bidx == 0
            should_debug_f = ENABLE_MMA_WARP_PRINT and tidx == warp_idx * 32 and hidx == 0 and bidx == 0

            if cutlass.const_expr(self.safe_gate):
                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C

                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        # wait bf16 State for MMA
                        kv16_handle = kv16_consumer.wait_and_advance()
                        # wait for sQ_K_scaled ready
                        # q_k_scaled_handle = load_q_k_scaled_consumer.wait_and_advance()
                        q_scaled_handle = load_q_scaled_consumer.wait_and_advance()

                        # launch Q@S MMA
                        o_inter_handle = o_inter_producer.acquire_and_advance()

                        # Compute SQ once Qi is ready.
                        sq_tiled_mma = self.exec_mma(
                            tiled_mma=sq_tiled_mma,
                            tCtAcc=tCtAccSQ,
                            tCrA=tCrState,
                            tCrB=tCrQ_sq,
                            a_stage_idx=0,
                            b_stage_idx=q_scaled_handle.index,
                            acc_stage_idx=0,
                        )
                        o_inter_handle.commit()
                        q_scaled_handle.release()

                        # wait for sQ_K_scaled ready
                        k_scaled_handle = load_k_scaled_consumer.wait_and_advance()

                        # launch K@S MMA
                        ks_handle = ks_producer.acquire_and_advance()
                        ks_tiled_mma = self.exec_mma(
                            tiled_mma=ks_tiled_mma,
                            # REUSE TMEM of PV
                            tCtAcc=tCtAccKS,
                            tCrA=tCrState_KS,
                            # NOTE: S^T * K^T,  this is still k-major, its ok to reuse A-style K here
                            tCrB=tCrK_KS,
                            a_stage_idx=kv16_handle.index,
                            b_stage_idx=k_scaled_handle.index,
                            acc_stage_idx=ks_handle.index,
                        )

                        ks_handle.commit()
                        k_scaled_handle.release()
                        kv16_handle.release()

                    # wait for T(inv(KK)) & V' ready
                    v2_handle = load_v2_consumer.wait_and_advance()
                    kk_handle = smem_kk_consumer.wait_and_advance()
                    # produce pseudo V
                    pseudo_v_handle = pseudo_v_producer.acquire_and_advance()

                    # NewV=T@V'
                    mv_tiled_mma = self.exec_mma(
                        tiled_mma=mv_tiled_mma,
                        # Reuse TMEM and shape of PV
                        tCtAcc=tCtAccMV,
                        tCrA=tCrV_corr,
                        tCrB=tCrM,
                        a_stage_idx=v2_handle.index,
                        b_stage_idx=kk_handle.index,
                        acc_stage_idx=pseudo_v_handle.index,
                    )

                    v2_handle.release()
                    kk_handle.release()
                    pseudo_v_handle.commit()

                    # wait NewV convert from Acc to operand
                    v3_handle = load_v3_consumer.wait_and_advance()
                    # wait Q@K^T
                    p_handle = p_consumer.wait_and_advance()
                    # producer O_intra
                    o_intra_handle = o_intra_producer.acquire_and_advance()

                    # O2=P@NewV
                    vp_tiled_mma = self.exec_mma(
                        tiled_mma=vp_tiled_mma,
                        tCtAcc=tCtAccPV,
                        tCrA=tCrV,
                        tCrB=tCrP,
                        a_stage_idx=v3_handle.index,
                        b_stage_idx=p_handle.index,
                        acc_stage_idx=o_intra_handle.index,
                    )

                    p_handle.release()
                    o_intra_handle.commit()

                    # wait for sQ_K_scaled and decay(S) ready
                    k_scaled2_handle = load_k_scaled2_consumer.wait_and_advance()
                    kv_decay_handle = kv_decay_consumer.wait_and_advance()
                    kv_handle = kv_producer.acquire_and_advance()

                    # launch S=K^T@NewV MMA
                    # NOTE: Always ACC to avoid adding in cuda core.
                    kv_tiled_mma = self.exec_mma(
                        tiled_mma=kv_tiled_mma,
                        tCtAcc=tCtAccKV,
                        tCrA=tCrV,
                        # tCrB=tCrK,
                        tCrB=tCrK_kv,
                        a_stage_idx=v3_handle.index,
                        b_stage_idx=k_scaled2_handle.index,
                        acc_stage_idx=0,
                        always_acc=True if (idx != 0 or cutlass.const_expr(self.has_initial_state)) else False,  # noqa: SIM210 -- Cute DSL: const_expr requires explicit True/False form # always accumulate states
                    )

                    k_scaled2_handle.release()
                    kv_decay_handle.release()
                    kv_handle.commit()

                    # NOTE: Add a signal to notify the end of v consumption.
                    end_v_handle = end_v_producer.acquire_and_advance()
                    end_v_handle.commit()
                    # release NewV
                    v3_handle.release()

            else:
                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C

                    k_handle = load_k2_consumer.wait_and_advance()
                    kt_handle = load_kt2_consumer.wait_and_advance()
                    mma_kk_handle = mma_kk_producer.acquire_and_advance()

                    if should_debug:
                        cute.printf("chunk idx={}, got k2 consumer={}", idx, k_handle.index)
                        cute.printf("chunk idx={}, got kt2 consumer={}", idx, kt_handle.index)
                        cute.printf("chunk idx={}, got mma_kk producer={}", idx, mma_kk_handle.index)

                    # GEMM KK
                    kk_tiled_mma = self.exec_mma(
                        tiled_mma=kk_tiled_mma,
                        tCtAcc=tCtAccKK,
                        tCrA=tCrKG,
                        tCrB=tCrKNegG,
                        a_stage_idx=k_handle.index,
                        b_stage_idx=kt_handle.index,
                        acc_stage_idx=mma_kk_handle.index,
                    )

                    # Commit KK
                    mma_kk_handle.commit()
                    # Wait for Qi (TMA load complete).
                    q_handle = load_q2_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got q2 consumer={}", idx, q_handle.index)

                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        #############################################
                        # HANDLE KS
                        kv16_handle = kv16_consumer.wait_and_advance()
                        if should_debug:
                            cute.printf("chunk idx={}, got kv16 consumer={}", idx, kv16_handle.index)
                        # Calculate MMA(State, Kg)
                        # Produce KS, need to create a operand B style
                        ks_handle = ks_producer.acquire_and_advance()
                        if should_debug:
                            cute.printf("chunk idx={}, got ks producer={}", idx, ks_handle.index)
                        ks_tiled_mma = self.exec_mma(
                            tiled_mma=ks_tiled_mma,
                            # REUSE TMEM of PV
                            tCtAcc=tCtAccKS,
                            tCrA=tCrState_KS,
                            # NOTE: S^T * K^T,  this is still k-major, its ok to reuse A-style K here
                            tCrB=tCrK_KS,
                            a_stage_idx=kv16_handle.index,
                            b_stage_idx=k_handle.index,
                            acc_stage_idx=ks_handle.index,
                        )
                        ks_handle.commit()
                        if should_debug:
                            cute.printf("chunk idx={}, committed ks={}", idx, ks_handle.index)

                        #############################################
                        # HANDLE QS
                        o_inter_handle = o_inter_producer.acquire_and_advance()
                        if should_debug:
                            cute.printf("chunk idx={}, got o_inter producer={}", idx, o_inter_handle.index)

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

                    # Acquire empty S0 buffer
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got s0 producer={}", idx, s0_handle.index)
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
                    # Release Kng = K*exp(-g) After QK and KK
                    kt_handle.release()
                    # Commit S = QK.
                    s0_handle.commit()
                    # End of GEMM (Qi, Ki) -> S0i

                    # -------------------------------------------------------------
                    # PRODUCE PSEUDO_V HERE
                    kk_handle = smem_kk_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got kk consumer={}", idx, kk_handle.index)
                    v2_handle = load_v2_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got v2 consumer={}", idx, v2_handle.index)
                    pseudo_v_handle = pseudo_v_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got pseudo_v producer={}", idx, pseudo_v_handle.index)
                    # Produce Pseudo V
                    mv_tiled_mma = self.exec_mma(
                        tiled_mma=mv_tiled_mma,
                        # Reuse TMEM and shape of PV
                        tCtAcc=tCtAccMV,
                        tCrA=tCrV_corr,
                        tCrB=tCrM,
                        a_stage_idx=v2_handle.index,
                        b_stage_idx=kk_handle.index,
                        acc_stage_idx=pseudo_v_handle.index,
                    )

                    v2_handle.release()
                    kk_handle.release()
                    pseudo_v_handle.commit()

                    # Wait for PV, produce ointra
                    v3_handle = load_v3_consumer.wait_and_advance()
                    p_handle = p_consumer.wait_and_advance()
                    o_intra_handle = o_intra_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got o_intra producer={}", idx, o_intra_handle.index)
                    # both v and p are in smem

                    # FIXME
                    self.mma_sync_barrier.arrive_and_wait()
                    if should_debug_f:
                        cute.printf("sV_epi before PV mma:\n")
                        cute.print_tensor(sV_epi[None, None, v3_handle.index])
                    self.mma_sync_barrier.arrive_and_wait()

                    vp_tiled_mma = self.exec_mma(
                        tiled_mma=vp_tiled_mma,
                        tCtAcc=tCtAccPV,
                        tCrA=tCrV,
                        tCrB=tCrP,
                        a_stage_idx=v3_handle.index,
                        b_stage_idx=p_handle.index,
                        acc_stage_idx=o_intra_handle.index,
                    )
                    p_handle.release()
                    o_intra_handle.commit()

                    ##########################################################
                    # NOTE: Generate next state for all chunks (including last when output_final_state)
                    if idx != ((seq_len + C - 1) // C) - 1 or cutlass.const_expr(self.output_final_state):
                        kv_handle = kv_producer.acquire_and_advance()

                        self.mma_sync_barrier.arrive_and_wait()
                        if should_debug_f:
                            cute.printf("chunk idx={}, got kv producer={}", idx, kv_handle.index)
                            cute.printf("sK_flat before state mma:\n")
                            cute.print_tensor(sK_flat[None, None, k_handle.index])
                            cute.printf("sV_epi before state mma:\n")
                            cute.print_tensor(sV_epi[None, None, v3_handle.index])
                            cute.printf("Full sV_epi before state mma:\n")
                            cute.print_tensor(sV_epi)
                        self.mma_sync_barrier.arrive_and_wait()

                        # NOTE: Always ACC to avoid adding in cuda core.
                        kv_tiled_mma = self.exec_mma(
                            tiled_mma=kv_tiled_mma,
                            tCtAcc=tCtAccKV,
                            tCrA=tCrV,
                            # tCrB=tCrK,
                            tCrB=tCrK_kv,
                            a_stage_idx=v3_handle.index,
                            b_stage_idx=k_handle.index,
                            acc_stage_idx=0,
                            always_acc=True if (idx != 0 or cutlass.const_expr(self.has_initial_state)) else False,  # noqa: SIM210 -- Cute DSL: const_expr requires explicit True/False form # always accumulate states
                        )
                        # Release K V here
                        kv_handle.commit()

                    # NOTE: Add a signal to notify the end of v consumption.
                    end_v_handle = end_v_producer.acquire_and_advance()
                    end_v_handle.commit()

                    k_handle.release()
                    v3_handle.release()

        # ///////////////////////////////////////////////////////////////////////////////
        # CUDA CORE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            # ----------------------------------------------------------
            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))

            should_debug = cutlass.const_expr(PRINT_DEBUG) and hidx == 0 and bidx == 0 and local_tidx == 0

            should_debug_f = ENABLE_CUDA_WG_PRINT and hidx == 0 and bidx == 0 and tidx == 32 * self.cuda_warp_ids[0]

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

            # HANDLE KK = K*K^T
            (
                tiled_t2r_KK,
                thr_t2r_KK,
                # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, STAGES)
                tTR_tKK,
                # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
                tTR_rKK,
            ) = self.tmem_load_and_partition_kk(
                local_tidx,
                # (MMA, MMA_M, MMA_N, N_STAGE)
                # (MMA_M, MMA_N, STAGE)
                tCtAccKK[((None, None), 0, 0, None)],
            )

            # TODO: check reuse of rKK and rP
            inverse_type = cutlass.Float16
            tTR_rKK_f16 = cute.make_rmem_tensor_like(
                src=tTR_rKK,
                dtype=inverse_type,
            )
            (
                tiled_r2s_KK,
                thr_r2s_KK,
                tRS_rKK,
                tRS_sKK,
            ) = self.smem_store_acc_as_ab_and_partition_x(
                local_tidx, sM_f16, tiled_t2r_KK, tTR_rKK_f16, show_debug_info=PRINT_DEBUG, debug_name="KK"
            )

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tiled_t2r_S: {tiled_t2r_S}")
                print(f"tTR_tS: {tTR_tS}")
                print(f"tTR_rS: {tTR_rS}")
                print(f"tiled_t2r_KK: {tiled_t2r_KK}")
                print(f"tTR_tKK: {tTR_tKK}")
                print(f"tTR_rKK: {tTR_rKK}")

            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            tTR_cMask = thr_t2r_S.partition_D(cM)

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

            # To Store Pseudo-V
            tTR_rPseudoV = cute.make_rmem_tensor_like(
                src=tTR_rAcc_pv,
                dtype=self.v_dtype,
            )
            (
                tiled_r2s_pseudo_v,
                thr_r2s_pseudo_v,
                tRS_rPseudoV,
                tRS_sPseudoV,
            ) = self.smem_store_and_partition_x(
                local_tidx=local_tidx,
                smem_x=sV_epi,
                tiled_t2r_x=tiled_copy_t2r_pv,
                tXrX_t2r=tTR_rPseudoV,
            )

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tTR_tS: {tTR_tS}")
                print(f"tTR_cMask: {tTR_cMask}")
                print(f"tTR_rS: {tTR_rS}")
                print(f"tTR_rP: {tTR_rP}")
                print(f"tRS_rPseudoV: {tRS_rPseudoV}")
                print(f"tRS_sPseudoV: {tRS_sPseudoV}")
                print(f"tTR_rPseudoV: {tTR_rPseudoV}")
                print(f"sV_epi: {sV_epi}")
                print(f"sV_flat_s2r: {sV_flat_s2r}")
                print(f"tiled_r2s_pseudo_v: {tiled_r2s_pseudo_v}")
                print(f"thr_r2s_pseudo_v: {thr_r2s_pseudo_v}")
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
            (
                tmem_store_kv_f32,
                tmem_store_tAccKV_f32,
                tmem_store_rAccKV_f32,
            ) = self.tmem_store_and_partition_acc(
                local_tidx,
                tCtAcc=tCtAccKV,
            )
            tmem_store_rKV = cute.make_tensor(tTR_rKV.iterator, layout=tmem_store_rAccKV_f32.layout)

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
                print(f"tmem_store_rKV: {tmem_store_rKV}")
                print(f"tmem_store_tAccKV_f32: {tmem_store_tAccKV_f32}")
                print(f"tmem_store_rAccKV_f32: {tmem_store_rAccKV_f32}")
                print(f"tmem_store_tAccKV: {tmem_store_tAccKV}")
                print(f"tmem_store_rAccKV: {tmem_store_rAccKV}")

            # -------------------------------------------------------
            # G (gate/g_cumsum) - NEW for KDA Step 1

            # TODO: replace these partitions with Ampere-MMA based partition, better perf
            shape_g = (Constant.C, Constant.D)
            # LOAD AS F32
            (
                tiled_s2r_g,
                thr_s2r_g,
                tRS_sG,
                tRS_rG,
            ) = self.make_s2r_partitions_prologue(
                local_tidx,
                sG_flat,
                shape_g,
            )
            # STORE k * exp(-g) AS BF16
            (
                tiled_s2r_g_bf16,
                thr_s2r_g_bf16,
                tRS_sG_bf16,
                tRS_rG_bf16,
            ) = self.make_s2r_partitions_prologue(
                local_tidx,
                sG_flat_bf16,
                shape_g,
            )
            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tiled_s2r_g: {tiled_s2r_g}")
                print(f"tRS_sG: {tRS_sG}")
                print(f"tRS_rG: {tRS_rG}")
                print(f"tiled_s2r_g_bf16: {tiled_s2r_g_bf16}")
                print(f"tRS_sG_bf16: {tRS_sG_bf16}")
                print(f"tRS_rG_bf16: {tRS_rG_bf16}")
            # -------------------------------------------------------

            # -------------------------------------------------------
            # Q s2r partitions - NEW for KDA elementwise processing
            shape_q = (Constant.C, Constant.D)
            (
                tiled_s2r_q,
                thr_s2r_q,
                tRS_sQ,  # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
                tRS_rQ,  # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
            ) = self.make_s2r_partitions_prologue(
                local_tidx,
                sQ_flat,
                shape_q,
            )
            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tiled_s2r_q: {tiled_s2r_q}")
                print(f"tRS_sQ: {tRS_sQ}")
                print(f"tRS_rQ: {tRS_rQ}")
            # -------------------------------------------------------

            # -------------------------------------------------------
            # K s2r partitions - NEW for KDA elementwise processing
            shape_k = (Constant.C, Constant.D)
            (
                tiled_s2r_k,
                thr_s2r_k,
                tRS_sK,  # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
                tRS_rK,  # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
            ) = self.make_s2r_partitions_prologue(
                local_tidx,
                sK_flat_s2r,
                shape_k,
            )
            # Create additional RMEM tensor for K_inter = K * exp(g)
            # tRS_rK_inter = cute.make_rmem_tensor_like(tRS_rK, dtype=self.io_dtype)
            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tiled_s2r_k: {tiled_s2r_k}")
                print(f"tRS_sK: {tRS_sK}")
                print(f"tRS_rK: {tRS_rK}")
            # print(f"tRS_rK_inter: {tRS_rK_inter}")
            # -------------------------------------------------------

            # -------------------------------------------------------
            # V s2r partitions - NEW for KDA elementwise processing
            # shape_v = (Constant.C, Constant.D)
            shape_v = (Constant.D, Constant.C)
            (
                tiled_s2r_v,
                thr_s2r_v,
                tRS_sV,  # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
                tRS_rV,  # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
            ) = self.make_s2r_partitions_v(
                local_tidx,
                sV_flat_s2r,
                shape_v,
            )
            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tiled_s2r_v: {tiled_s2r_v}")
                print(f"tRS_sV: {tRS_sV}")
                print(f"tRS_rV: {tRS_rV}")

            mma_op = cute.nvgpu.warp.MmaF16BF16Op(ab_dtype=self.q_dtype, acc_dtype=self.acc_dtype, shape_mnk=(16, 8, 16))
            # Half-size MMA for partitioned S2R: K dim halved from D=128 to HALF_D=64
            # This reduces per-thread register fragment by 2x during S2R loading
            tiled_mma_epi_half = cute.make_tiled_mma(
                mma_op,
                atom_layout_mnk=(4, 1, 1),  # NOTE: 4 warps to process prologue
                permutation_mnk=self.qk_mma_tiler_half,  # (64, 64, 64)
            )
            thr_mma_epi_half = tiled_mma_epi_half.get_slice(local_tidx)
            copy_op_qk_s2r = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
            copy_op_qk_r2s = cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4)
            # FIXME: only 2 FP32 elements (64 bits) compatible with ldmatrix, how to change to 128?
            copy_atom_g = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.g_dtype, num_bits_per_copy=64)
            # Half-size tiled copies for partitioned S2R
            tiled_load_g_half = cute.make_tiled_copy_A(copy_atom_g, tiled_mma_epi_half)
            thr_load_g_half = tiled_load_g_half.get_slice(local_tidx)
            tiled_load_qk_half = cute.make_tiled_copy_A(cute.make_copy_atom(copy_op_qk_s2r, self.io_dtype), tiled_mma_epi_half)
            thr_load_qk_half = tiled_load_qk_half.get_slice(local_tidx)
            tiled_store_qk_half = cute.make_tiled_copy_A(
                cute.make_copy_atom(copy_op_qk_r2s, self.io_dtype), tiled_mma_epi_half
            )
            thr_store_qk_half = tiled_store_qk_half.get_slice(local_tidx)

            # ---- Half-width SMEM views for partitioned S2R ----
            # Each full SMEM tensor (C, D, stage) is split into two halves along D.
            # First half: D=[0, HALF_D), same base pointer
            # Second half: D=[HALF_D, D), base pointer + C*HALF_D elements
            # NOTE: Use existing tensor iterators (NOT storage.get_tensor) to avoid
            # SharedStorage reference leaking into DSL if-block iter args
            HALF_SMEM_ELEMS = Constant.C * Constant.HALF_D  # 4096

            # Q half SMEM views (from sQ_flat iterator)
            q_sml_epi_half = sm100_utils.make_smem_layout_epi(
                self.q_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (Constant.C, Constant.HALF_D),
                self.q_stage,
            )
            q_sml_half = cute.coalesce(q_sml_epi_half, target_profile=(1, 1, 1))
            # Fix stage stride: half layout has stride C*HALF_D but actual SMEM uses C*D
            q_half_outer = cute.make_layout(
                q_sml_half.outer.shape, stride=(*q_sml_half.outer.stride[:-1], q_sml_half.outer.stride[-1] * 2)
            )
            sQ_flat_h0 = cute.make_tensor(sQ_flat.iterator, layout=q_half_outer)
            sQ_flat_h1 = cute.make_tensor(sQ_flat.iterator + HALF_SMEM_ELEMS, layout=q_half_outer)

            # K half SMEM views (from sK_flat_s2r iterator)
            k_sml_epi_half = sm100_utils.make_smem_layout_epi(
                self.k_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (Constant.C, Constant.HALF_D),
                self.k_stage,
            )
            k_sml_half = cute.coalesce(k_sml_epi_half, target_profile=(1, 1, 1))
            k_half_outer = cute.make_layout(
                k_sml_half.outer.shape, stride=(*k_sml_half.outer.stride[:-1], k_sml_half.outer.stride[-1] * 2)
            )
            sK_flat_h0 = cute.make_tensor(sK_flat_s2r.iterator, layout=k_half_outer)
            sK_flat_h1 = cute.make_tensor(sK_flat_s2r.iterator + HALF_SMEM_ELEMS, layout=k_half_outer)

            # G half SMEM views (FP32, from sG_flat iterator)
            g_sml_epi_half = sm100_utils.make_smem_layout_epi(
                self.g_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (Constant.C, Constant.HALF_D),
                self.g_stage,
            )
            g_sml_half = cute.coalesce(g_sml_epi_half, target_profile=(1, 1, 1))
            g_half_outer = cute.make_layout(
                g_sml_half.outer.shape, stride=(*g_sml_half.outer.stride[:-1], g_sml_half.outer.stride[-1] * 2)
            )
            sG_flat_h0 = cute.make_tensor(sG_flat.iterator, layout=g_half_outer)
            sG_flat_h1 = cute.make_tensor(sG_flat.iterator + HALF_SMEM_ELEMS, layout=g_half_outer)

            # Q_K_scaled half SMEM views (from sQ_K_scaled_flat iterator)
            qks_sml_epi_half = sm100_utils.make_smem_layout_epi(
                self.q_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (Constant.C, Constant.HALF_D),
                self.q_k_scaled_stage,
            )
            qks_sml_half = cute.coalesce(qks_sml_epi_half, target_profile=(1, 1, 1))
            qks_half_outer = cute.make_layout(
                qks_sml_half.outer.shape, stride=(*qks_sml_half.outer.stride[:-1], qks_sml_half.outer.stride[-1] * 2)
            )
            sQKS_flat_h0 = cute.make_tensor(sQ_K_scaled_flat.iterator, layout=qks_half_outer)
            sQKS_flat_h1 = cute.make_tensor(sQ_K_scaled_flat.iterator + HALF_SMEM_ELEMS, layout=qks_half_outer)

            # Lists for iteration in half-loops
            sQ_flat_halves = [sQ_flat_h0, sQ_flat_h1]
            sK_flat_halves = [sK_flat_h0, sK_flat_h1]
            sG_flat_halves = [sG_flat_h0, sG_flat_h1]
            sQKS_flat_halves = [sQKS_flat_h0, sQKS_flat_h1]

            # Partition half SMEM views with half-size tiled copies
            tQsQ_h = [thr_load_qk_half.partition_S(h) for h in sQ_flat_halves]
            tQsK_h = [thr_load_qk_half.partition_S(h) for h in sK_flat_halves]
            tQsG_h = [thr_load_g_half.partition_S(h) for h in sG_flat_halves]
            tQsQKS_h = [thr_store_qk_half.partition_D(h) for h in sQKS_flat_halves]

            # Half-size fragment prototype
            sQ_h0_0 = sQ_flat_h0[None, None, 0]
            tQrQ_half_0 = thr_mma_epi_half.make_fragment_A(thr_mma_epi_half.partition_A(sQ_h0_0))

            # Half-size index tensor: maps thread elements to (row, col) in [0,C) x [0,HALF_D)
            q_shape_half = (self.qk_mma_tiler_half[0], self.qk_mma_tiler_half[2])
            cM_half = cute.make_identity_tensor(q_shape_half)
            tQcMq_half = thr_mma_epi_half.partition_A(cM_half)

            def index_transform_half(index_q, index_k):
                return (
                    index_q,
                    index_k,
                )

            # -------------------------------------------------------

            # -------------- DEBUG -------------
            # if tidx == 0:
            #     cute.printf("-------------------- sG_flat raw:")
            #     cute.print_tensor(sG_flat)
            # #     cute.printf("-------------------- sQ_flat raw:")
            # #     cute.print_tensor(sQ_flat)
            #     cute.printf("-------------------- sK_flat raw:")
            #     cute.print_tensor(sK_flat)
            # # # Note: add barrier here to make sure print make sense since we'll overwrite sG
            # self.cuda_wg_sync_barrier.arrive_and_wait()

            # -------------- Initial State Loading -------------
            # If has_initial_state, load initial state from GMEM into TMEM
            # State shape: (D, D) per (H, B), stored as FP32
            if cutlass.const_expr(self.has_initial_state):
                # Load initial state from GMEM to RMEM respecting TMEM partition.
                # TMEM stores S^T (transposed), so flat[i] = state[local_tidx, i]
                # Each thread owns key position local_tidx, D elements cover value positions.
                init_state_chunk = initial_state[None, None, (hidx, bidx)]
                init_flat = cute.make_tensor(tTR_rKV.iterator, layout=cute.make_layout(Constant.D))
                for init_i in cutlass.range(0, Constant.D, unroll=0):
                    init_flat[init_i] = init_state_chunk[local_tidx, init_i]

                # Store FP32 state to TMEM for accumulation (tCtAccKV)
                init_tmem_store_tKVi = tmem_store_tAccKV_f32[None, None, None, None, 0]
                cute.copy(tmem_store_kv_f32, tmem_store_rKV, init_tmem_store_tKVi)
                cute.arch.fence_view_async_tmem_store()

                # Also prepare BF16 version for Q@S and K@S MMA
                tmem_store_rAccKVAsBF16.store(tTR_rKV.load().to(self.io_dtype))
                init_kv16_handle = kv16_producer.acquire_and_advance()
                init_tmem_store_tAccKVi = tmem_store_tAccKV[None, None, None, None, init_kv16_handle.index]
                cute.copy(tmem_store_kv, tmem_store_rAccKV, init_tmem_store_tAccKVi)
                cute.arch.fence_view_async_tmem_store()
                init_kv16_handle.commit()

            final_blk = (seq_len + C - 1) // C - 1
            if cutlass.const_expr(self.safe_gate):
                # safe_gate version
                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C
                    if cutlass.const_expr(self.is_varlen):
                        valid_len_chunk = seq_len - chunk_start
                    else:
                        valid_len_chunk = C

                    # ============================================================
                    # KDA cuda core logic
                    # 1. prologue for Q/K with G: exp(g)*Q/K for Q@S, K@S, exp(g_last-g)*K for K^T@NewV
                    # 2. element-wise operations: V'=V-KS, S=S*g_last, O=O1+O2
                    # ============================================================

                    # ====================================================
                    # Partitioned S2R: G loading + g_last + Q gating
                    # Each half loads HALF_D=64 elements vs full D=128,
                    # reducing per-thread register fragment by 2x during S2R.
                    # exp2(g) persists in tQrG_persists[] across Q→K gating
                    # to avoid redundant G SMEM reloads (flashkda pattern).
                    # ====================================================

                    # wait G
                    g_handle = load_g_consumer.wait_and_advance()
                    g_stage_idx = g_handle.index

                    # wait Q (always wait to advance pipeline)
                    q_handle = load_q_consumer.wait_and_advance()

                    # Pre-allocate persistent G register fragments (outside if blocks)
                    # These hold exp2(g) after Q gating for reuse in K gating
                    tQrG_persists = []
                    for half_idx in cutlass.range_constexpr(2):
                        tQrG_persists.append(cute.make_fragment_like(tQrQ_half_0, dtype=self.g_dtype))

                    # Merged g_last + Q gating path: single G half-load per half
                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        q_stage_idx = q_handle.index
                        q_scaled_handle = load_q_scaled_producer.acquire_and_advance()
                        q_scaled_stage_idx = q_scaled_handle.index

                        for half_idx in cutlass.range_constexpr(2):
                            k_offset = half_idx * Constant.HALF_D

                            # S2R G half into persistent fragment
                            tQrG_half_cv = thr_load_g_half.retile(tQrG_persists[half_idx])
                            cute.copy(tiled_load_g_half, tQsG_h[half_idx][None, None, None, g_stage_idx], tQrG_half_cv)

                            # Write g_last half (before exp transforms g values)
                            for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                index_q, index_k = index_transform_half(*tQcMq_half[i])
                                if cutlass.const_expr(self.is_varlen):
                                    if valid_len_chunk < C:
                                        if index_q == valid_len_chunk - 1:
                                            sG_last[index_k + k_offset, g_stage_idx] = tQrG_persists[half_idx][i]
                                    else:
                                        if index_q == Constant.C - 1:
                                            sG_last[index_k + k_offset, g_stage_idx] = tQrG_persists[half_idx][i]
                                else:
                                    if index_q == Constant.C - 1:
                                        sG_last[index_k + k_offset, g_stage_idx] = tQrG_persists[half_idx][i]

                            # exp(g) half in-place — persists for K gating reuse
                            for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                tQrG_persists[half_idx][i] = cute.exp2(tQrG_persists[half_idx][i], fastmath=self.use_fast_math)

                            # S2R Q half
                            tQrQ_half = cute.make_fragment_like(tQrQ_half_0, self.q_dtype)
                            tQrQ_half_cv = thr_load_qk_half.retile(tQrQ_half)
                            cute.copy(tiled_load_qk_half, tQsQ_h[half_idx][None, None, None, q_stage_idx], tQrQ_half_cv)

                            # Zero Q for invalid positions (varlen only)
                            if cutlass.const_expr(self.is_varlen):
                                if valid_len_chunk < C:
                                    for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                        index_q, index_k = index_transform_half(*tQcMq_half[i])
                                        if index_q >= valid_len_chunk:
                                            tQrQ_half[i] = self.q_dtype(0.0)

                            # Q gating: Q' = Q * exp(g) * scale
                            for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                q_i = tQrQ_half[i].to(cutlass.Float32)
                                tQrQ_half[i] = (q_i * tQrG_persists[half_idx][i] * self.scale).to(self.io_dtype)

                            # R2S Q' half to sQ_K_scaled
                            tQrQ_half_cv_src = thr_store_qk_half.retile(tQrQ_half)
                            cute.copy(
                                tiled_store_qk_half, tQrQ_half_cv_src, tQsQKS_h[half_idx][None, None, None, q_scaled_stage_idx]
                            )

                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        q_scaled_handle.commit()
                    else:
                        # idx==0 without initial state: only write g_last (no Q gating)
                        for half_idx in cutlass.range_constexpr(2):
                            k_offset = half_idx * Constant.HALF_D
                            tQrG_half_cv = thr_load_g_half.retile(tQrG_persists[half_idx])
                            cute.copy(tiled_load_g_half, tQsG_h[half_idx][None, None, None, g_stage_idx], tQrG_half_cv)
                            for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                index_q, index_k = index_transform_half(*tQcMq_half[i])
                                if cutlass.const_expr(self.is_varlen):
                                    if valid_len_chunk < C:
                                        if index_q == valid_len_chunk - 1:
                                            sG_last[index_k + k_offset, g_stage_idx] = tQrG_persists[half_idx][i]
                                    else:
                                        if index_q == Constant.C - 1:
                                            sG_last[index_k + k_offset, g_stage_idx] = tQrG_persists[half_idx][i]
                                else:
                                    if index_q == Constant.C - 1:
                                        sG_last[index_k + k_offset, g_stage_idx] = tQrG_persists[half_idx][i]

                    # ====================================================
                    # Partitioned S2R: K gating (2 half-passes)
                    # K' = K * exp(g), reusing exp2(g) from tQrG_persists[]
                    # (no redundant G SMEM reload!)
                    # ====================================================

                    # wait K
                    k_handle = load_k_consumer.wait_and_advance()
                    k_stage_idx = k_handle.index
                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        # Wait Q@S MMA before writing K' to sQ_K_scaled
                        o_inter_handle = o_inter_consumer.wait()
                        k_scaled_handle = load_k_scaled_producer.acquire_and_advance()
                        k_scaled_stage_idx = k_scaled_handle.index

                        for half_idx in cutlass.range_constexpr(2):
                            k_offset = half_idx * Constant.HALF_D

                            # S2R K half
                            tQrK_half = cute.make_fragment_like(tQrQ_half_0, self.q_dtype)
                            tQrK_half_cv = thr_load_qk_half.retile(tQrK_half)
                            cute.copy(tiled_load_qk_half, tQsK_h[half_idx][None, None, None, k_stage_idx], tQrK_half_cv)

                            # Zero K for invalid positions (varlen only)
                            if cutlass.const_expr(self.is_varlen):
                                if valid_len_chunk < C:
                                    for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                        index_q, index_k = index_transform_half(*tQcMq_half[i])
                                        if index_q >= valid_len_chunk:
                                            tQrK_half[i] = self.q_dtype(0.0)

                            # K gating: K' = K * exp(g) — reuse persisted exp2(g)
                            for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                k_i = tQrK_half[i].to(cutlass.Float32)
                                tQrK_half[i] = (k_i * tQrG_persists[half_idx][i]).to(self.io_dtype)

                            # R2S K' half to sQ_K_scaled
                            tQrK_half_cv_src = thr_store_qk_half.retile(tQrK_half)
                            cute.copy(
                                tiled_store_qk_half, tQrK_half_cv_src, tQsQKS_h[half_idx][None, None, None, k_scaled_stage_idx]
                            )

                        # Commit K' after both halves written
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        k_scaled_handle.commit()

                    # release Q
                    q_handle.release()

                    # wait V
                    v_handle = load_v_consumer.wait_and_advance()
                    # self.cuda_wg_sync_barrier.arrive_and_wait()
                    # if should_debug2:
                    #     cute.printf("chunk idx={}, V:", idx)
                    #     cute.print_tensor(sV_flat_s2r[None, None, v_handle.index])
                    # self.cuda_wg_sync_barrier.arrive_and_wait()

                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        # load V to reg
                        cute.copy(tiled_s2r_v, tRS_sV[(None, None, None, v_handle.index)], tRS_rV)
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )

                        # TODO: copy KS from TMEM to RMEM, NOTE KS reuse TMEM of PV, since they have the same shapes
                        # TODO: check the layout equivalence, should be row-major
                        tTR_rAcc_ks = cute.make_tensor(tTR_rAcc_pv.iterator, layout=tRS_rV.layout)

                        # wait K@S MMA
                        ks_handle = ks_consumer.wait_and_advance()

                        # T2R K@S acc
                        # TODO: confirm pv capacity is enough
                        # Load KS from TMEM to RMEM, (128,64)
                        tTR_tAcc_ks_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, ks_handle.index)]  # KS HANDLE INDEX == 0
                        cute.copy(tiled_copy_t2r_sq, tTR_tAcc_ks_i, tTR_rAcc_pv)
                        cute.arch.fence_view_async_tmem_load()

                        ks_handle.release()
                        # if should_debug2:
                        #     cute.printf("chunk idx={}, KS:", idx)
                        #     cute.print_tensor(tTR_rAcc_pv)

                        # Produce Vcorr = V - KS
                        v2_handle = load_v2_producer.acquire_and_advance()

                        # V'=V-KS
                        v_corrected = tRS_rV.load().to(cutlass.Float32)
                        # if should_debug2:
                        #     cute.printf("chunk idx={}, KS:", idx)
                        #     cute.print_tensor(tTR_rAcc_pv)
                        ks = tTR_rAcc_ks.load()
                        v_corrected -= ks
                        # Convert to BF16 and store v_corrected to SMEM
                        tRS_rV.store(v_corrected.to(self.v_dtype))

                        # FIXME: write V to TMEM as operand A
                        # write V back to SMEM
                        cute.copy(tiled_s2r_v, tRS_rV, tRS_sV[(None, None, None, v2_handle.index)])
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )

                        # notify T@V' MMA
                        # Commit Vcorr = V - KS to trigger M*Vcorr
                        v2_handle.commit()
                    else:
                        # first chunk, notify T@V' directly
                        v2_handle = load_v2_producer.acquire_and_advance()
                        v2_handle.commit()

                    # wait NewV=T@V' MMA
                    pseudo_v_handle = pseudo_v_consumer.wait_and_advance()
                    pseudo_v_stage_idx = pseudo_v_handle.index

                    # T2R NewV
                    # REUSE TMEM of PV, LOAD Pseudo-V from TMEM to RMEM
                    tTR_tAcc_pv_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, pseudo_v_stage_idx)]
                    cute.copy(tiled_copy_t2r_pv, tTR_tAcc_pv_i, tTR_rAcc_pv)

                    # release
                    cute.arch.fence_view_async_tmem_load()
                    pseudo_v_handle.release()

                    # convert Acc to V dtype
                    tTR_rPseudoV.store(tTR_rAcc_pv.load().to(self.v_dtype))

                    # R2S NewV and notify O2=P@NewV MMA
                    v3_handle = load_v3_producer.acquire_and_advance()

                    cute.copy(tiled_r2s_pseudo_v, tRS_rPseudoV, tRS_sPseudoV[(None, None, None, pseudo_v_stage_idx)])

                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    # FIXME: when printing, accuracy issues, Why?
                    # self.cuda_wg_sync_barrier.arrive_and_wait()
                    # if should_debug2:
                    #     cute.printf("chunk idx={}, NewV:", idx)
                    #     cute.print_tensor(sV_epi[None, None, v3_handle.index])
                    # self.cuda_wg_sync_barrier.arrive_and_wait()
                    v3_handle.commit()

                    # decay S, S=S*g_last
                    # FIXME: currently do not support initial state,
                    # so only decay S after first block K^T@NewV
                    kv_decay_handle = kv_decay_producer.acquire_and_advance()
                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        # NOTE: TMEM S is always ready here
                        # T2R S
                        tTR_tKVi = tTR_tKV[(None, None, None, 0)]  # kv stage == 1
                        cute.copy(tiled_copy_t2r_kv, tTR_tKVi, tTR_rKV)
                        cute.arch.fence_view_async_tmem_load()

                        # decay S
                        flat = cute.make_tensor(tTR_rKV.iterator, layout=cute.make_layout(Constant.D))
                        self.scale_state(flat, sG_last[None, g_stage_idx])

                        # R2T S
                        # NOTE: TMEM STORE DECAY KV STATE to enable accumulation over chunks
                        tmem_store_tKVi = tmem_store_tAccKV_f32[None, None, None, None, 0]
                        cute.copy(tmem_store_kv_f32, tmem_store_rKV, tmem_store_tKVi)
                        cute.arch.fence_view_async_tmem_store()

                    kv_decay_handle.commit()

                    # ====================================================
                    # Partitioned S2R: K^T gating — exp(g_last-g)*K
                    # ====================================================
                    k_scaled2_handle = load_k_scaled2_producer.acquire_and_advance()
                    k_scaled2_stage_idx = k_scaled2_handle.index

                    for half_idx in cutlass.range_constexpr(2):
                        k_offset = half_idx * Constant.HALF_D

                        # S2R G half
                        tQrG_half = cute.make_fragment_like(tQrQ_half_0, dtype=self.g_dtype)
                        tQrG_half_cv = thr_load_g_half.retile(tQrG_half)
                        cute.copy(tiled_load_g_half, tQsG_h[half_idx][None, None, None, g_stage_idx], tQrG_half_cv)

                        # S2R K half
                        tQrK_half = cute.make_fragment_like(tQrQ_half_0, dtype=self.k_dtype)
                        tQrK_half_cv = thr_load_qk_half.retile(tQrK_half)
                        cute.copy(tiled_load_qk_half, tQsK_h[half_idx][None, None, None, k_stage_idx], tQrK_half_cv)

                        # Zero K half for invalid positions (varlen only)
                        if cutlass.const_expr(self.is_varlen):
                            if valid_len_chunk < C:
                                for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                                    index_q, index_k = index_transform_half(*tQcMq_half[i])
                                    if index_q >= valid_len_chunk:
                                        tQrK_half[i] = self.k_dtype(0.0)

                        # K^T gating: exp(g_last - g) * K
                        for i in cutlass.range_constexpr(cute.size(tQcMq_half)):
                            index_q, index_k = index_transform_half(*tQcMq_half[i])
                            g_last_val = sG_last[index_k + k_offset, g_stage_idx]
                            k_i = tQrK_half[i].to(cutlass.Float32)
                            g_i = tQrG_half[i]
                            tQrK_half[i] = (cute.exp2(g_last_val - g_i, fastmath=self.use_fast_math) * k_i).to(self.k_dtype)

                        # R2S K^T half to sQ_K_scaled
                        tQrK_half_cv_src = thr_store_qk_half.retile(tQrK_half)
                        cute.copy(
                            tiled_store_qk_half, tQrK_half_cv_src, tQsQKS_h[half_idx][None, None, None, k_scaled2_stage_idx]
                        )

                    # notify K^T@NewV MMA
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    k_scaled2_handle.commit()

                    # wait O2=P@NewV MMA, reuse TMEM of K@S
                    o_intra_handle = o_intra_consumer.wait_and_advance()
                    # T2R O2
                    # Load O_INTRA from TMEM to RMEM
                    tTR_tAcc_pv_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, o_intra_handle.index)]
                    cute.copy(tiled_copy_t2r_pv, tTR_tAcc_pv_i, tTR_rAcc_pv)

                    cute.arch.fence_view_async_tmem_load()
                    o_intra_handle.release()

                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        # T2R Q@S (O_inter)
                        # NOTE: stage=1, always 0
                        tTR_tAcc_sq_i = tTR_tAcc_base_sq[(None, None, None, 0, 0, 0)]
                        # Load O_INTER from TMEM to RMEM
                        cute.copy(tiled_copy_t2r_sq, tTR_tAcc_sq_i, tTR_rAcc_sq)
                        cute.arch.fence_view_async_tmem_load()
                        o_inter_consumer.release()
                        o_inter_consumer.advance()

                    # O=O1+O2
                    acc_vec = tTR_rAcc_pv.load()
                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        acc_vec_inter = tTR_rAcc_sq.load()
                        acc_vec = acc_vec + acc_vec_inter
                    tTR_rO.store(acc_vec.to(self.io_dtype))

                    # notify epilogue O Store
                    # NOTE: overlapped with S=K^T@NewV
                    smem_o_handle = smem_o_producer.acquire_and_advance()

                    cute.copy(tiled_copy_r2s_o, tRS_rO, tRS_sO[(None, None, None, smem_o_handle.index)])

                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    smem_o_handle.commit()

                    # wait S=K^T@NewV MMA
                    kv_handle = kv_consumer.wait_and_advance()

                    # release G, K
                    g_handle.release()
                    k_handle.release()

                    # convert State to BF16, except for final block
                    if idx != final_blk or cutlass.const_expr(self.output_final_state):
                        # T2R FP32 S
                        tTR_tKVi = tTR_tKV[(None, None, None, kv_handle.index)]  # kv stage == 1
                        cute.copy(tiled_copy_t2r_kv, tTR_tKVi, tTR_rKV)
                        cute.arch.fence_view_async_tmem_load()

                    if idx != final_blk:
                        # Store as a separated BF16 state for QS and KS MMA before decay
                        # tmem_store_rAccKVAsBF16 point to the same rmem as tmem_store_rKV
                        tmem_store_rAccKVAsBF16.store(tTR_rKV.load().to(self.io_dtype))

                        # Prepare bf16 state for tcgen05.mma
                        kv16_handle = kv16_producer.acquire_and_advance()
                        # convert to BF16
                        # V^T*K -> (Dv, Dk)
                        tmem_store_tAccKVi = tmem_store_tAccKV[None, None, None, None, kv16_handle.index]

                        # R2T BF16 S
                        # tmem_store_rAccKV is just an FP32 recast view of tmem_store_rAccKVAsBF16
                        cute.copy(tmem_store_kv, tmem_store_rAccKV, tmem_store_tAccKVi)

                        cute.arch.fence_view_async_tmem_store()
                        kv16_handle.commit()
                    else:
                        # idx == final_blk: output final state immediately to minimize tTR_rKV lifetime
                        if cutlass.const_expr(self.output_final_state):
                            # Write FP32 state from RMEM to GMEM
                            # TMEM stores S^T (transposed), so flat[i] = state[local_tidx, i]
                            state_out = final_state[None, None, (hidx, bidx)]
                            out_flat = cute.make_tensor(tTR_rKV.iterator, layout=cute.make_layout(Constant.D))
                            for out_i in cutlass.range(0, Constant.D, unroll=0):
                                state_out[local_tidx, out_i] = out_flat[out_i]

                    # release KV
                    kv_handle.release()

                    # release V
                    # NOTE: only release v after PV and State=KV has been consumed
                    end_v_handle = end_v_consumer.wait_and_advance()
                    end_v_handle.release()
                    # TODO: move V to TMEM as operand A
                    v_handle.release()

            else:
                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C
                    if cutlass.const_expr(self.is_varlen):
                        valid_len_chunk = seq_len - chunk_start
                    else:
                        valid_len_chunk = C

                    # ============================================================
                    # KDA Prologue: Load g, Q, K from SMEM to RMEM for elementwise
                    # ============================================================

                    self.cuda_wg_sync_barrier.arrive_and_wait()

                    # Load g (g_cumsum) - NEW for KDA Step 1
                    g_handle = load_g_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got g consumer={}", idx, g_handle.index)

                    g_stage_idx = g_handle.index
                    cute.copy(tiled_s2r_g, tRS_sG[(None, None, None, g_stage_idx)], tRS_rG)
                    # Fence for shared memory reads
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    # Load Q from SMEM to RMEM for KDA elementwise processing
                    q_handle = load_q_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got q consumer={}", idx, q_handle.index)
                    q_stage_idx = q_handle.index
                    cute.copy(tiled_s2r_q, tRS_sQ[(None, None, None, q_stage_idx)], tRS_rQ)
                    # Fence for shared memory reads
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    # Load K from SMEM to RMEM for KDA elementwise processing
                    k_handle = load_k_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got k consumer={}", idx, k_handle.index)
                    k_stage_idx = k_handle.index
                    cute.copy(tiled_s2r_k, tRS_sK[(None, None, None, k_stage_idx)], tRS_rK)

                    # Fence for shared memory reads
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    q_handle.release()
                    k_handle.release()

                    # ============================================================
                    # KDA Step 2: Compute exp(g) and apply to Q, K
                    # Q' = Q * exp(g)
                    # K_inter = K * exp(g)   (for inter-chunk: K'^T V -> state update)
                    # K_intra = K * exp(-g)  (for intra-chunk: Q' K''^T computation)
                    # ============================================================

                    # TODO: subchunk pipeline v1
                    # 1. g'=exp(g), store in reg
                    # 2. g'*Q, write to sQ_K_scaled
                    # 3. wait for QS finish, g'*K, write to sQ_K_scaled
                    # 3. wait for KS finish, exp(g_last-g)*K, write to sQ_K_scaled

                    # Load g, Q, K and compute gated values element-wise
                    # Element-wise processing avoids bulk .load()/.to() creating
                    # ~300+ register SSA vectors from G, Q, K simultaneously
                    for _zr in cutlass.range(0, Constant.C, unroll_full=True):
                        if cutlass.const_expr(self.is_varlen):
                            if valid_len_chunk < C and _zr >= valid_len_chunk:
                                tRS_rQ[0, _zr, 0] = self.io_dtype(0.0)
                                tRS_rK[0, _zr, 0] = self.io_dtype(0.0)
                                tRS_rG_bf16[0, _zr, 0] = self.io_dtype(0.0)
                            else:
                                g_i = tRS_rG[0, _zr, 0]
                                exp_g_i = cute.exp2(g_i, fastmath=self.use_fast_math)
                                q_i = tRS_rQ[0, _zr, 0].to(cutlass.Float32)
                                tRS_rQ[0, _zr, 0] = (q_i * exp_g_i * self.scale).to(self.io_dtype)
                                k_i = tRS_rK[0, _zr, 0].to(cutlass.Float32)
                                tRS_rK[0, _zr, 0] = (k_i * exp_g_i).to(self.io_dtype)
                                tRS_rG_bf16[0, _zr, 0] = (k_i * cute.exp2(-g_i, fastmath=self.use_fast_math)).to(self.io_dtype)
                        else:
                            g_i = tRS_rG[0, _zr, 0]
                            exp_g_i = cute.exp2(g_i, fastmath=self.use_fast_math)
                            q_i = tRS_rQ[0, _zr, 0].to(cutlass.Float32)
                            tRS_rQ[0, _zr, 0] = (q_i * exp_g_i * self.scale).to(self.io_dtype)
                            k_i = tRS_rK[0, _zr, 0].to(cutlass.Float32)
                            tRS_rK[0, _zr, 0] = (k_i * exp_g_i).to(self.io_dtype)
                            tRS_rG_bf16[0, _zr, 0] = (k_i * cute.exp2(-g_i, fastmath=self.use_fast_math)).to(self.io_dtype)

                    # ============================================================
                    # KDA Step 3: Write gated Q', K_inter, K_intra back to SMEM
                    # - Q' = Q * exp(g) -> SMEM[Q]
                    # - K_inter = K * exp(g) -> SMEM[K]
                    # - K_intra = K * exp(-g) -> SMEM[G] (overwrite g)
                    # ============================================================

                    # Make sure the read from sK and sG are completed.
                    self.cuda_wg_sync_barrier.arrive_and_wait()

                    # TODO: check q2 & k2 stage equivalence
                    # TODO: remove, only write to a seperate smem
                    k2_handle = load_k2_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got k2 producer={}", idx, k2_handle.index)
                    # Write K_inter = K * exp(g) to SMEM[K]
                    if not cutlass.const_expr(self.safe_gate):
                        cute.copy(tiled_s2r_k, tRS_rK, tRS_sK[(None, None, None, k_stage_idx)])
                    # produce k * exp(g) for mma
                    k2_handle.commit()

                    # Write K_intra = K * exp(-g) to SMEM[G] (overwrite g)
                    # TODO: remove
                    kt2_handle = load_kt2_producer.acquire_and_advance()
                    if should_debug_f:
                        cute.printf("chunk idx={}, got kt2 producer={}, g_stage_idx={}", idx, kt2_handle.index, g_stage_idx)
                    # BUG FIX: use kt2_handle.index instead of g_stage_idx to ensure producer/consumer index match
                    if not cutlass.const_expr(self.safe_gate):
                        cute.copy(tiled_s2r_g_bf16, tRS_rG_bf16, tRS_sG_bf16[(None, None, None, kt2_handle.index)])
                    # produce k^t * exp(-g) for mma
                    kt2_handle.commit()

                    q2_handle = load_q2_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got q2 producer={}", idx, q2_handle.index)
                    # Write Q' from RMEM to SMEM (same location as original Q)
                    if not cutlass.const_expr(self.safe_gate):
                        cute.copy(tiled_s2r_q, tRS_rQ, tRS_sQ[(None, None, None, q_stage_idx)])
                    # produce q * exp(g) for mma
                    q2_handle.commit()

                    # Fence for shared memory writes
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    self.cuda_wg_sync_barrier.arrive_and_wait()
                    if should_debug_f:
                        cute.printf("-------------------- sQ_flat: q * exp(g)")
                        cute.print_tensor(sQ_flat[None, None, q_stage_idx])
                        cute.printf("-------------------- k * exp(g)")
                        # cute.print_tensor(sK_flat[63, None, k_stage_idx], verbose=True)
                        cute.print_tensor(sK_flat[None, None, k_stage_idx])
                        cute.printf("-------------------- k * exp(-g):")
                        cute.print_tensor(sG_flat_bf16[None, None, g_stage_idx])

                    # ============================================================
                    # KDA End of Prologue
                    # ============================================================

                    # ------------------------------------------------------------
                    # NOTE: Save exp(g) of last VALID row to rG_last for state update in next chunk
                    # For full chunks, directly use C-1; only loop for partial chunks (varlen only)
                    if cutlass.const_expr(self.is_varlen):
                        if valid_len_chunk < C:
                            rG_last = exp_g[valid_len_chunk - 1]
                        else:
                            rG_last = exp_g[Constant.C - 1]
                    else:
                        rG_last = exp_g[Constant.C - 1]
                    # NOTE: each thread save one element
                    sG_last[local_tidx, g_stage_idx] = rG_last

                    mma_kk_handle = mma_kk_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got mma_kk consumer={}", idx, mma_kk_handle.index)

                    cute.copy(tiled_t2r_KK, tTR_tKK[None, None, None, mma_kk_handle.index], tTR_rKK)
                    cute.arch.fence_view_async_tmem_load()
                    # GEMM KK done, can release
                    mma_kk_handle.release()

                    # Inplace modify tTR_rKK to M matrix
                    beta_handle = load_beta_consumer.wait_and_advance()
                    # step1: M = I + StrictTril(beta*KK^T), save M in smem
                    # TODO: drop assignment for tTR_rKK
                    self.apply_M_transform(tTR_rKK, sBeta, tTR_cMask, tTR_rKK_f16)

                    # STORE M as F16
                    smem_kk_handle = smem_kk_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got smem_kk producer={}", idx, smem_kk_handle.index)
                    tRS_sKKi = tRS_sKK[(None, None, None, smem_kk_handle.index)]
                    if not cutlass.const_expr(self.safe_gate):
                        cute.copy(tiled_r2s_KK, tRS_rKK, tRS_sKKi)
                    # Fence
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    if cutlass.const_expr(PRINT_DEBUG):
                        print(f"sM: {sM}")
                        print(f"sM_f16: {sM_f16}")
                    # self.cuda_wg_sync_barrier.arrive_and_wait()
                    # if should_debug2 and not cutlass.const_expr(self.safe_gate):
                    #     tiler_acc_qk_kk = (16, 16)
                    #     sM_f16_slice = cute.flat_divide(sM_f16[None, None, smem_kk_handle.index], tiler_acc_qk_kk)
                    #     cute.printf("sM_f16_0_0")
                    #     cute.print_tensor(sM_f16_slice[None, None, 0, 0])
                    curr_sM = sM[None, None, smem_kk_handle.index]
                    curr_sM_f16 = sM_f16[None, None, smem_kk_handle.index]
                    if not cutlass.const_expr(self.safe_gate):
                        self.compute_matrix_inverse_64x64(curr_sM_f16)

                        # S2R, scale with beta, convert to BF16, store back to smem `sM`
                        # TODO: make a repro for the cutedsl team
                        self.scale_M_inverse_with_beta(local_tidx, sBeta, curr_sM_f16, curr_sM)

                    # Make sure the inverse is done.
                    self.cuda_wg_sync_barrier.arrive_and_wait()
                    if should_debug_f:
                        cute.printf("--------------- M after inverse and beta scale:")
                        cute.print_tensor(curr_sM)
                    # FIXME
                    self.cuda_wg_sync_barrier.arrive_and_wait()

                    # Notify end of smem_kk
                    smem_kk_handle.commit()
                    # release Beta
                    beta_handle.release()

                    # TODO: LOAD V via S2R, need to have the same tv-layout as TMEM, since we need to perform a elementwise reduce here.
                    # Need to make it (128,64) and row-major
                    v_handle = load_v_consumer.wait_and_advance()

                    self.cuda_wg_sync_barrier.arrive_and_wait()
                    if should_debug_f:
                        cute.printf("chunk idx={}, got v consumer={}, sV:", idx, v_handle.index)
                        cute.print_tensor(sV_epi)
                    self.cuda_wg_sync_barrier.arrive_and_wait()

                    cute.copy(tiled_s2r_v, tRS_sV[(None, None, None, v_handle.index)], tRS_rV)
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    # TODO: copy KS from TMEM to RMEM, NOTE KS reuse TMEM of PV, since they have the same shapes
                    # TODO: check the layout equivalence, should be row-major
                    tTR_rAcc_ks = cute.make_tensor(tTR_rAcc_pv.iterator, layout=tRS_rV.layout)
                    if cutlass.const_expr(PRINT_DEBUG):
                        print(f"FIXME tTR_rAcc_pv: {tTR_rAcc_pv}")
                        print(f"FIXME tTR_rAcc_ks: {tTR_rAcc_ks}")
                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        # Wait for KS
                        ks_handle = ks_consumer.wait_and_advance()

                        if should_debug:
                            cute.printf("chunk idx={}, got ks consumer={}", idx, ks_handle.index)
                        # TODO: confirm pv capacity is enough
                        # Load KS from TMEM to RMEM, (128,64)
                        tTR_tAcc_ks_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, ks_handle.index)]  # KS HANDLE INDEX == 0
                        cute.copy(tiled_copy_t2r_sq, tTR_tAcc_ks_i, tTR_rAcc_pv)
                        cute.arch.fence_view_async_tmem_load()

                        ks_handle.release()

                        # if should_debug2:
                        #     cute.printf("chunk idx={}, KS:", idx)
                        #     cute.print_tensor(tTR_rAcc_pv)

                    # Perform addition and store to tmem for Vcorr^T * M^T
                    # TODO: now we just store V back to SMEM, in the future we could save them to TMEM to enable double stage of V
                    # Produce Vcorr = V - KS
                    v2_handle = load_v2_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got v2 producer={}", idx, v2_handle.index)

                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        # First V could be kept no changes, since the initial state is None
                        v_corrected = tRS_rV.load().to(cutlass.Float32)

                        self.cuda_wg_sync_barrier.arrive_and_wait()
                        if should_debug_f:
                            cute.printf("chunk idx={}, V rmem chunk:", idx)
                            cute.print_tensor(tRS_rV)
                        self.cuda_wg_sync_barrier.arrive_and_wait()

                        ks = tTR_rAcc_ks.load()
                        v_corrected -= ks
                        # TODO: why store back to SMEM?
                        # Convert to BF16 and store v_corrected to SMEM
                        tRS_rV.store(v_corrected.to(self.v_dtype))

                        self.cuda_wg_sync_barrier.arrive_and_wait()
                        if should_debug_f:
                            cute.printf("chunk idx={}, V-KS rmem chunk:", idx)
                            cute.print_tensor(tRS_rV)
                        self.cuda_wg_sync_barrier.arrive_and_wait()

                        # Store Vcorr back to SMEM
                        cute.copy(tiled_s2r_v, tRS_rV, tRS_sV[(None, None, None, v_handle.index)])
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                    # Commit Vcorr = V - KS to trigger M*Vcorr
                    v2_handle.commit()

                    # Let Kg = K * exp(g_cumsum)
                    # Let Kn = K * exp(-g_cumsum)
                    # Let Qg = Q * exp(g_cumsum)
                    #
                    # Produce pseudo-V = M * (V - Kg * States)
                    # {Pseudo-V} ^ T = (V^T - State^T * Kg^T) * M^T
                    # Store Pseudo-V back to V-SMEM
                    pseudo_v_handle = pseudo_v_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got pseudo_v consumer={}", idx, pseudo_v_handle.index)

                    pseudo_v_stage_idx = pseudo_v_handle.index
                    # REUSE TMEM of PV, LOAD Pseudo-V from TMEM to RMEM
                    tTR_tAcc_pv_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, pseudo_v_stage_idx)]
                    cute.copy(tiled_copy_t2r_pv, tTR_tAcc_pv_i, tTR_rAcc_pv)
                    cute.arch.fence_view_async_tmem_load()
                    pseudo_v_handle.release()

                    # Convert to V dtype
                    tTR_rPseudoV.store(tTR_rAcc_pv.load().to(self.v_dtype))

                    v3_handle = load_v3_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got v3 producer={}", idx, v3_handle.index)

                    cute.copy(tiled_r2s_pseudo_v, tRS_rPseudoV, tRS_sPseudoV[(None, None, None, pseudo_v_stage_idx)])
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    v3_handle.commit()

                    self.cuda_wg_sync_barrier.arrive_and_wait()
                    if should_debug_f:
                        cute.printf("------------ begin pseudov dump, idx={}", idx)
                        cute.print_tensor(sV_epi)
                        cute.printf("------------ end pseudov dump")
                    self.cuda_wg_sync_barrier.arrive_and_wait()

                    # Maintain of S:
                    # S_{t+1} = G_last*S_{t} + Kg^T* PseudoV
                    #
                    # O_Inter = Qg * S
                    # O_Intra = Tril(Qg * Kn^T) * PseudoV
                    # O = O_Inter + O_Intra

                    # Wait for S = MMA(exp(g)*Q, (K*exp(-g))^T)
                    s0_handle = mma_s0_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got s0 consumer={}", idx, s0_handle.index)

                    # NOTE: Only Allow next TMA Load for G after k^exp(-g) has been consumed by QK & KK
                    # TODO: This pipeline is too complex, we need to refactor it.
                    # TODO: We might need to introduce multiple cuda warpgroups to achieve better parallelism.
                    # TODO: G has been consumed in multiple places, need to track them carefully and use updated pipeline instances.
                    g_handle.release()

                    # (MMA, MMA_M, MMA_N, ACC_STAGE)
                    tTR_tSi = tTR_tS[None, None, None, s0_handle.index]
                    # Load S from TMEM to RMEM
                    cute.copy(tiled_t2r_S, tTR_tSi, tTR_rS)
                    cute.arch.fence_view_async_tmem_load()

                    # TODO: Apply strict causal mask and comput inverse of M
                    self.apply_mask(tTR_rS, tTR_cMask, tTR_rP, debug=False)

                    # Write P to SMEM
                    p_handle = p_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got p producer={}", idx, p_handle.index)

                    # Store P from RMEM to SMEM
                    tRS_sPi = tRS_sP[(None, None, None, p_handle.index)]
                    if not cutlass.const_expr(self.safe_gate):
                        cute.copy(tiled_r2s_P, tRS_rP, tRS_sPi)
                    # Fence
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    s0_handle.release()
                    p_handle.commit()

                    # self.cuda_wg_sync_barrier.arrive_and_wait()
                    # if should_debug2 and not cutlass.const_expr(self.safe_gate):
                    #     tiler_acc_qk_kk = (16, 16)
                    #     sQK_slice = cute.flat_divide(sQK_flat[None, None, p_handle.index], tiler_acc_qk_kk)
                    #     cute.printf("sQK_0_0")
                    #     cute.print_tensor(sQK_slice[None, None, 0, 0])

                    if cutlass.const_expr(PRINT_DEBUG):
                        self.cuda_wg_sync_barrier.arrive_and_wait()
                        if should_debug_f:
                            cute.printf("------- smem QK:")
                            cute.print_tensor(sQK_flat)
                            cute.printf("------- smem QK, last row:")
                            cute.print_tensor(sQK_flat[63, None, None], verbose=True)

                    # Wait for O_INTRA
                    o_intra_handle = o_intra_consumer.wait_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got o_intra consumer={}", idx, o_intra_handle.index)
                    # Load O_INTRA from TMEM to RMEM
                    tTR_tAcc_pv_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, o_intra_handle.index)]
                    cute.copy(tiled_copy_t2r_pv, tTR_tAcc_pv_i, tTR_rAcc_pv)
                    cute.arch.fence_view_async_tmem_load()
                    o_intra_handle.release()

                    # Wait for O_INTER
                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        o_inter_handle = o_inter_consumer.wait_and_advance()
                        if should_debug:
                            cute.printf("chunk idx={}, got o_inter consumer={}", idx, o_inter_handle.index)
                        tTR_tAcc_sq_i = tTR_tAcc_base_sq[(None, None, None, 0, 0, o_inter_handle.index)]
                        # Load O_INTER from TMEM to RMEM
                        cute.copy(tiled_copy_t2r_sq, tTR_tAcc_sq_i, tTR_rAcc_sq)
                        cute.arch.fence_view_async_tmem_load()
                        o_inter_handle.release()

                    # Perform addition and store to gmem
                    acc_vec = tTR_rAcc_pv.load()
                    if idx != 0 or cutlass.const_expr(self.has_initial_state):
                        acc_vec_inter = tTR_rAcc_sq.load()
                        acc_vec = acc_vec + acc_vec_inter
                    tTR_rO.store(acc_vec.to(self.io_dtype))

                    # Store output to smem
                    smem_o_handle = smem_o_producer.acquire_and_advance()
                    if should_debug:
                        cute.printf("chunk idx={}, got smem_o producer={}", idx, smem_o_handle.index)
                    cute.copy(tiled_copy_r2s_o, tRS_rO, tRS_sO[(None, None, None, smem_o_handle.index)])
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    smem_o_handle.commit()

                    # ------------------------------------------------------------
                    # 1. Decay the state (T2R FP32 read from TMEM to RMEM)
                    # 2. Output final state if this is the last chunk
                    # We split the T2R read (needed for final state) from kv16 produce (only for non-final)
                    if idx != ((seq_len + C - 1) // C - 1) or cutlass.const_expr(self.output_final_state):
                        # Wait for kv mma from `idx-1` round of mma warp
                        kv_handle = kv_consumer.wait_and_advance()
                        if should_debug:
                            cute.printf("chunk idx={}, got kv consumer={}", idx, kv_handle.index)

                        tTR_tKVi = tTR_tKV[(None, None, None, kv_handle.index)]  # kv stage == 1
                        cute.copy(tiled_copy_t2r_kv, tTR_tKVi, tTR_rKV)
                        cute.arch.fence_view_async_tmem_load()

                        self.cuda_wg_sync_barrier.arrive_and_wait()
                        if should_debug_f:
                            cute.printf("--------------- before decay KV state chunk idx={}", idx)
                            cute.print_tensor(tTR_rKV)
                        self.cuda_wg_sync_barrier.arrive_and_wait()

                        flat = cute.make_tensor(tTR_rKV.iterator, layout=cute.make_layout(Constant.D))

                        # FIXME
                        self.cuda_wg_sync_barrier.arrive_and_wait()

                        # Then decay the FP32 version state
                        self.scale_state(flat, sG_last[None, g_stage_idx])

                        self.cuda_wg_sync_barrier.arrive_and_wait()
                        if should_debug_f:
                            cute.printf("--------------- after decay KV state chunk idx={}", idx)
                            cute.print_tensor(tTR_rKV)
                        self.cuda_wg_sync_barrier.arrive_and_wait()

                        # Store as a separated BF16 state for QS and KS MMA before decay
                        # tmem_store_rAccKVAsBF16 point to the same rmem as tmem_store_rKV
                        tmem_store_rAccKVAsBF16.store(tTR_rKV.load().to(self.io_dtype))

                        # NOTE: TMEM STORE DECAY KV STATE to enable accumulation over chunks
                        tmem_store_tKVi = tmem_store_tAccKV_f32[None, None, None, None, kv_handle.index]
                        cute.copy(tmem_store_kv_f32, tmem_store_rKV, tmem_store_tKVi)
                        cute.arch.fence_view_async_tmem_store()
                        kv_handle.release()

                    # 3. Convert to BF16 and produce to kv16 pipeline (only for non-final blocks)
                    if idx != ((seq_len + C - 1) // C - 1):
                        # Prepare bf16 state for tcgen05.mma
                        kv16_handle = kv16_producer.acquire_and_advance()
                        if should_debug:
                            cute.printf("chunk idx={}, got kv16 producer={}", idx, kv16_handle.index)
                        #####################################################################
                        # V^T*K -> (Dv, Dk)
                        tmem_store_tAccKVi = tmem_store_tAccKV[None, None, None, None, kv16_handle.index]
                        # tmem_store_rAccKV is just an FP32 recast view of tmem_store_rAccKVAsBF16
                        cute.copy(tmem_store_kv, tmem_store_rAccKV, tmem_store_tAccKVi)
                        cute.arch.fence_view_async_tmem_store()
                        #####################################################################
                        kv16_handle.commit()
                    else:
                        # idx == final: output final state immediately to minimize tTR_rKV lifetime
                        if cutlass.const_expr(self.output_final_state):
                            state_out = final_state[None, None, (hidx, bidx)]
                            out_flat = cute.make_tensor(tTR_rKV.iterator, layout=cute.make_layout(Constant.D))
                            for out_i in cutlass.range(0, Constant.D, unroll=0):
                                state_out[local_tidx, out_i] = out_flat[out_i]

                    # NOTE: only release v after PV and State=KV has been consumed
                    end_v_handle = end_v_consumer.wait_and_advance()
                    end_v_handle.release()

                    # Finally let us release v.
                    v_handle.release()

        # CUDA core warps for subchunk computation
        elif warp_idx in self.cuda_subchunk_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_subchunk)
            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_subchunk_warp_ids))
            should_debug = local_tidx == 0 and hidx == 0 and bidx == 0
            subchunk_tidx = local_tidx
            if local_tidx >= 64:
                subchunk_tidx = local_tidx - 64
            if cutlass.const_expr(self.safe_gate):
                # define TiledMMA and TiledCopy
                mma_op = cute.nvgpu.warp.MmaF16BF16Op(ab_dtype=self.q_dtype, acc_dtype=self.acc_dtype, shape_mnk=(16, 8, 16))
                tiled_mma_subchunk = cute.make_tiled_mma(
                    mma_op,
                    atom_layout_mnk=(1, 2, 1),  # 2 warps compute MMA
                    permutation_mnk=self.qk_kk_subchunk_mma_tiler,
                )
                thr_mma_subchunk = tiled_mma_subchunk.get_slice(subchunk_tidx)
                copy_op_A_s2r = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
                copy_op_B_s2r = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
                copy_op_r2s = cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2)
                # FIXME: only 2 FP32 elements (64 bits) compatible with ldmatrix, how to change to 128?
                copy_g_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.g_dtype, num_bits_per_copy=64)
                G_Q_tiled_copy = cute.make_tiled_copy_A(copy_g_atom, tiled_mma_subchunk)
                G_Kt_tiled_copy = cute.make_tiled_copy_B(copy_g_atom, tiled_mma_subchunk)
                Q_tiled_copy = cute.make_tiled_copy_A(cute.make_copy_atom(copy_op_A_s2r, self.q_dtype), tiled_mma_subchunk)
                Kt_tiled_copy = cute.make_tiled_copy_B(cute.make_copy_atom(copy_op_B_s2r, self.k_dtype), tiled_mma_subchunk)
                O_tiled_copy = cute.make_tiled_copy_C(cute.make_copy_atom(copy_op_r2s, self.q_dtype), tiled_mma_subchunk)
                O_tiled_copy_kk = cute.make_tiled_copy_C(
                    cute.make_copy_atom(copy_op_r2s, self.inverse_dtype), tiled_mma_subchunk
                )
                G_Q_thr_copy = G_Q_tiled_copy.get_slice(subchunk_tidx)
                G_Kt_thr_copy = G_Kt_tiled_copy.get_slice(subchunk_tidx)
                Q_thr_copy = Q_tiled_copy.get_slice(subchunk_tidx)
                Kt_thr_copy = Kt_tiled_copy.get_slice(subchunk_tidx)
                O_thr_copy = O_tiled_copy.get_slice(subchunk_tidx)
                O_thr_copy_kk = O_tiled_copy_kk.get_slice(subchunk_tidx)

                # index tensor
                cMqk_subchunk = cute.make_identity_tensor(self.qk_kk_subchunk_mma_tiler[:2])
                tQKcMqk_subchunk = thr_mma_subchunk.partition_C(cMqk_subchunk)

                def index_transform(index_q, index_k):
                    return (
                        index_q,
                        index_k,
                    )

                # epilogue
                tiled_mma_epi_fake = cute.make_tiled_mma(
                    mma_op,
                    atom_layout_mnk=(4, 1, 1),  # NOTE: 4 warps to process QK&KK
                    permutation_mnk=self.qk_mma_tiler,
                )
                thr_mma_epi_fake = tiled_mma_epi_fake.get_slice(local_tidx)
                tQKrQK_fake = tiled_mma_epi_fake.make_fragment_C(tiled_mma_epi_fake.partition_shape_C(self.qk_mma_tiler[:2]))
                copy_op_epi_s2r = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
                copy_op_epi_r2s = cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4)
                tiled_load_qk = cute.make_tiled_copy_C(cute.make_copy_atom(copy_op_epi_s2r, self.io_dtype), tiled_mma_epi_fake)
                tiled_load_kk = cute.make_tiled_copy_C(
                    cute.make_copy_atom(copy_op_epi_s2r, self.inverse_dtype), tiled_mma_epi_fake
                )
                tiled_store_qk = cute.make_tiled_copy_C(
                    cute.make_copy_atom(copy_op_epi_r2s, self.io_dtype), tiled_mma_epi_fake
                )
                tiled_store_kk = cute.make_tiled_copy_C(
                    cute.make_copy_atom(copy_op_epi_r2s, self.inverse_dtype), tiled_mma_epi_fake
                )
                thr_load_qk = tiled_load_qk.get_slice(local_tidx)
                thr_load_kk = tiled_load_kk.get_slice(local_tidx)
                thr_store_qk = tiled_store_qk.get_slice(local_tidx)
                thr_store_kk = tiled_store_kk.get_slice(local_tidx)
                tQKrQK_cv = thr_load_qk.retile(tQKrQK_fake)
                # index tensor
                cM = cute.make_identity_tensor(self.qk_mma_tiler[:2])
                tQKcMqk = thr_mma_epi_fake.partition_C(cM)

                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C

                    # subchunk computation
                    # divide input/output
                    tiler_subchunk_g = (16, (32, 2))
                    tiler_subchunk_qk = (16, (64, 1))
                    tiler_subchunk_beta = (16,)
                    # TODO: change to pipeline.PipelineState declaration, hack currently
                    sQqk_curr = sQ_flat[None, None, load_q_consumer._PipelineConsumer__state.index]
                    sKqk_curr = sK_flat[None, None, load_k_consumer._PipelineConsumer__state.index]
                    sGqkq_curr = sG_flat[None, None, load_g_consumer._PipelineConsumer__state.index]
                    sBeta_curr = sBeta[None, load_beta_consumer._PipelineConsumer__state.index]

                    # (_16,(_32,_2),_4,(_1,_2)):(_32,(_1,_2048),_512,(_0,_4096))
                    sQqk_slice = cute.flat_divide(sQqk_curr, tiler_subchunk_qk)
                    sKqk_slice = cute.flat_divide(sKqk_curr, tiler_subchunk_qk)
                    # (_16,(_64,_1),_4,(_1,_2)):(_64,(_1,_0),_1024,(_0,_4096))
                    sGqkq_slice = cute.flat_divide(sGqkq_curr, tiler_subchunk_g)
                    sBeta_slice = cute.flat_divide(sBeta_curr, tiler_subchunk_beta)

                    # Acc results
                    tiler_acc_qk_kk = (16, 16)
                    p_stage_idx = p_producer._PipelineProducer__state.index
                    smem_kk_stage_idx = smem_kk_producer._PipelineProducer__state.index
                    sQK_curr = sQK_flat[None, None, p_stage_idx]
                    sQK_slice = cute.flat_divide(sQK_curr, tiler_acc_qk_kk)
                    sKK_inv_curr = sM_f16_flat[None, None, smem_kk_stage_idx]
                    sKK_inv_slice = cute.flat_divide(sKK_inv_curr, tiler_acc_qk_kk)
                    curr_sM = sM[None, None, smem_kk_stage_idx]
                    curr_sM_f16 = sM_f16[None, None, smem_kk_stage_idx]

                    # if should_debug:
                    #     cute.printf("q_pipe={}", load_q_consumer._PipelineConsumer__state.index)
                    #     cute.printf("k_pipe={}", load_k_consumer._PipelineConsumer__state.index)
                    #     cute.printf("g_pipe={}", load_g_consumer._PipelineConsumer__state.index)
                    #     cute.printf("beta_pipe={}", load_beta_consumer._PipelineConsumer__state.index)
                    #     cute.printf("qk_pipe={}", load_q_consumer._PipelineConsumer__state.index)
                    #     cute.printf("kk_pipe={}", smem_kk_producer._PipelineProducer__state.index)

                    sGqkq_0_0 = sGqkq_slice[None, None, 0, (0, 0)]
                    layout_g_first = cute.make_layout(sGqkq_0_0.shape, stride=(0, sGqkq_0_0.stride[1]))

                    # used for make_fragment_like in G
                    sQqk_1_0 = sQqk_slice[None, None, 1, (0, 0)]
                    sKqk_1_0 = sKqk_slice[None, None, 1, (0, 0)]
                    tQKrQ_1_0 = thr_mma_subchunk.make_fragment_A(thr_mma_subchunk.partition_A(sQqk_1_0))
                    tQKrKt_1_0 = thr_mma_subchunk.make_fragment_B(thr_mma_subchunk.partition_B(sKqk_1_0))
                    tv_layout_mma_A = tQKrQ_1_0.layout
                    tv_layout_mma_B = tQKrKt_1_0.layout

                    # g_i_j/q_i_j/k_i_j: the j-th head dim slice of the i-th subchunk
                    if local_tidx < 64:
                        # Q/K0@K0, Q/K3@K3, Q/K3@K0, Q/K3@K1, Q/K3@K2
                        # NOTE: tensor core MMA for safe gate with lower_bound >= -5
                        # Q/K0@K0
                        tQKrQK_0_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_0_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        # first subchunk, wait for data ready
                        g_handle = load_g_consumer.wait_and_advance()
                        q_handle = load_q_consumer.wait_and_advance()
                        k_handle = load_k_consumer.wait_and_advance()

                        for j in cutlass.range(self.NK_SC):
                            tQKrQ_0_j, tQKrK_0_j = self.s2r_compute_subchunk_operand_A(
                                0,
                                j,
                                G_Q_tiled_copy,
                                G_Q_thr_copy,
                                Q_tiled_copy,
                                Q_thr_copy,
                                tv_layout_mma_A,
                                layout_g_first,
                                sGqkq_slice,
                                sQqk_slice,
                                sKqk_slice,
                            )

                            # S2R g_0_j_first again (different layouts for operand B)
                            sGqkq_0_j = sGqkq_slice[None, None, 0, (0, j)]
                            sG_first_0_j = cute.make_tensor(sGqkq_0_j.iterator, layout=layout_g_first)
                            tGsGfirst_0_j_kt = G_Kt_thr_copy.partition_S(sG_first_0_j)
                            tGrGfirst_0_j_kt = cute.make_fragment_like(tv_layout_mma_B, dtype=self.g_dtype)
                            tGrGfirst_0_j_kt_cv = G_Kt_thr_copy.retile(tGrGfirst_0_j_kt)
                            cute.copy(G_Kt_tiled_copy, tGsGfirst_0_j_kt, tGrGfirst_0_j_kt_cv)

                            tQKrKt_0_j = self.s2r_compute_subchunk_operand_B(
                                0,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_0_j_kt,
                            )

                            # q_0_j/k_0_j @ k_0_j, accumulate acc_3_3
                            cute.gemm(tiled_mma_subchunk, tQKrQK_0_0, tQKrQ_0_j, tQKrKt_0_j, tQKrQK_0_0)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_0_0, tQKrK_0_j, tQKrKt_0_j, tKKrKK_0_0)

                        qk_0_0_val = tQKrQK_0_0.load()
                        qk_0_0_val = qk_0_0_val * self.scale
                        tQKrQK_0_0.store(qk_0_0_val)
                        # first subchunk, wait for data ready
                        beta_handle = load_beta_consumer.wait()
                        # Fence due to normal load
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        sBeta_0 = sBeta_slice[None, 0]
                        for i in cutlass.range_constexpr(cute.size(tQKcMqk_subchunk)):
                            s, t = index_transform(*tQKcMqk_subchunk[i])
                            b = sBeta_0[s]
                            tKKrKK_0_0[i] *= b

                        # R2S qk_0_0, kk_0_0
                        # first subchunk, acquire data
                        smem_kk_producer.acquire()
                        self.r2s_subchunk_acc(
                            0, 0, tKKrKK_0_0, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )

                        p_producer.acquire()
                        self.r2s_subchunk_acc(0, 0, tQKrQK_0_0, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)

                        # Q/K3@K0, Q/K3@K1, Q/K3@K2, Q/K3@K3
                        tQKrQK_3_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_3_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tQKrQK_3_1 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_3_1 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tQKrQK_3_2 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_3_2 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tQKrQK_3_3 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_3_3 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )

                        for j in cutlass.range(self.NK_SC):
                            tQKrQ_3_j, tQKrK_3_j = self.s2r_compute_subchunk_operand_A(
                                3,
                                j,
                                G_Q_tiled_copy,
                                G_Q_thr_copy,
                                Q_tiled_copy,
                                Q_thr_copy,
                                tv_layout_mma_A,
                                layout_g_first,
                                sGqkq_slice,
                                sQqk_slice,
                                sKqk_slice,
                            )

                            # S2R g_0_j, g_3_j_first again (different layouts for operand B)
                            sGqkq_3_j = sGqkq_slice[None, None, 3, (0, j)]
                            sG_first_3_j = cute.make_tensor(sGqkq_3_j.iterator, layout=layout_g_first)
                            tGsGfirst_3_j_kt = G_Kt_thr_copy.partition_S(sG_first_3_j)
                            tGrGfirst_3_j_kt = cute.make_fragment_like(tv_layout_mma_B, dtype=self.g_dtype)
                            tGrGfirst_3_j_kt_cv = G_Kt_thr_copy.retile(tGrGfirst_3_j_kt)
                            cute.copy(G_Kt_tiled_copy, tGsGfirst_3_j_kt, tGrGfirst_3_j_kt_cv)

                            tQKrKt_0_j = self.s2r_compute_subchunk_operand_B(
                                0,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_3_j_kt,
                            )

                            # q_3_j/k_3_j @ k_0_j, accumulate acc_3_0
                            cute.gemm(tiled_mma_subchunk, tQKrQK_3_0, tQKrQ_3_j, tQKrKt_0_j, tQKrQK_3_0)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_3_0, tQKrK_3_j, tQKrKt_0_j, tKKrKK_3_0)

                            tQKrKt_1_j = self.s2r_compute_subchunk_operand_B(
                                1,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_3_j_kt,
                            )

                            # q_3_j/k_3_j @ k_1_j, accumulate acc_3_1
                            cute.gemm(tiled_mma_subchunk, tQKrQK_3_1, tQKrQ_3_j, tQKrKt_1_j, tQKrQK_3_1)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_3_1, tQKrK_3_j, tQKrKt_1_j, tKKrKK_3_1)

                            tQKrKt_2_j = self.s2r_compute_subchunk_operand_B(
                                2,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_3_j_kt,
                            )

                            # q_3_j/k_3_j @ k_2_j, accumulate acc_3_2
                            cute.gemm(tiled_mma_subchunk, tQKrQK_3_2, tQKrQ_3_j, tQKrKt_2_j, tQKrQK_3_2)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_3_2, tQKrK_3_j, tQKrKt_2_j, tKKrKK_3_2)

                            tQKrKt_3_j = self.s2r_compute_subchunk_operand_B(
                                3,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_3_j_kt,
                            )

                            # q_3_j/k_3_j @ k_3_j, accumulate acc_3_3
                            cute.gemm(tiled_mma_subchunk, tQKrQK_3_3, tQKrQ_3_j, tQKrKt_3_j, tQKrQK_3_3)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_3_3, tQKrK_3_j, tQKrKt_3_j, tKKrKK_3_3)

                        qk_3_0_val = tQKrQK_3_0.load()
                        qk_3_0_val = qk_3_0_val * self.scale
                        tQKrQK_3_0.store(qk_3_0_val)
                        qk_3_1_val = tQKrQK_3_1.load()
                        qk_3_1_val = qk_3_1_val * self.scale
                        tQKrQK_3_1.store(qk_3_1_val)
                        qk_3_2_val = tQKrQK_3_2.load()
                        qk_3_2_val = qk_3_2_val * self.scale
                        tQKrQK_3_2.store(qk_3_2_val)
                        qk_3_3_val = tQKrQK_3_3.load()
                        qk_3_3_val = qk_3_3_val * self.scale
                        tQKrQK_3_3.store(qk_3_3_val)
                        sBeta_3 = sBeta_slice[None, 3]
                        for i in cutlass.range_constexpr(cute.size(tQKcMqk_subchunk)):
                            s, t = index_transform(*tQKcMqk_subchunk[i])
                            b = sBeta_3[s]
                            tKKrKK_3_0[i] *= b
                            tKKrKK_3_1[i] *= b
                            tKKrKK_3_2[i] *= b
                            tKKrKK_3_3[i] *= b

                        # R2S qk_3_0, kk_3_0
                        self.r2s_subchunk_acc(
                            3, 0, tKKrKK_3_0, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(3, 0, tQKrQK_3_0, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)
                        # R2S qk_3_1, kk_3_1
                        self.r2s_subchunk_acc(
                            3, 1, tKKrKK_3_1, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(3, 1, tQKrQK_3_1, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)
                        # R2S qk_3_2, kk_3_2
                        self.r2s_subchunk_acc(
                            3, 2, tKKrKK_3_2, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(3, 2, tQKrQK_3_2, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)
                        # R2S qk_3_3, kk_3_3
                        self.r2s_subchunk_acc(
                            3, 3, tKKrKK_3_3, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(3, 3, tQKrQK_3_3, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)

                        # release Q, K, G
                        g_handle.release()
                        q_handle.release()
                        k_handle.release()

                    else:
                        # Q/K1@K0, Q/K2@K0, Q/K2@K1, Q/K2@K2, Q/K1@K1

                        # Q/K1@K0, Q/K1@K1
                        tQKrQK_1_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_1_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tQKrQK_1_1 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_1_1 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        # first subchunk, wait for data ready
                        g_handle = load_g_consumer.wait_and_advance()
                        q_handle = load_q_consumer.wait_and_advance()
                        k_handle = load_k_consumer.wait_and_advance()

                        for j in cutlass.range(self.NK_SC):
                            tQKrQ_1_j, tQKrK_1_j = self.s2r_compute_subchunk_operand_A(
                                1,
                                j,
                                G_Q_tiled_copy,
                                G_Q_thr_copy,
                                Q_tiled_copy,
                                Q_thr_copy,
                                tv_layout_mma_A,
                                layout_g_first,
                                sGqkq_slice,
                                sQqk_slice,
                                sKqk_slice,
                            )

                            # S2R g_0_j, g_1_j_first again (different layouts for operand B)
                            sGqkq_1_j = sGqkq_slice[None, None, 1, (0, j)]
                            sG_first_1_j = cute.make_tensor(sGqkq_1_j.iterator, layout=layout_g_first)
                            tGsGfirst_1_j_kt = G_Kt_thr_copy.partition_S(sG_first_1_j)
                            tGrGfirst_1_j_kt = cute.make_fragment_like(tv_layout_mma_B, dtype=self.g_dtype)
                            tGrGfirst_1_j_kt_cv = G_Kt_thr_copy.retile(tGrGfirst_1_j_kt)
                            cute.copy(G_Kt_tiled_copy, tGsGfirst_1_j_kt, tGrGfirst_1_j_kt_cv)

                            tQKrKt_0_j = self.s2r_compute_subchunk_operand_B(
                                0,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_1_j_kt,
                            )

                            # q_1_j/k_1_j @ k_0_j, accumulate acc_1_0
                            cute.gemm(tiled_mma_subchunk, tQKrQK_1_0, tQKrQ_1_j, tQKrKt_0_j, tQKrQK_1_0)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_1_0, tQKrK_1_j, tQKrKt_0_j, tKKrKK_1_0)

                            tQKrKt_1_j = self.s2r_compute_subchunk_operand_B(
                                1,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_1_j_kt,
                            )

                            # q_1_j/k_1_j @ k_1_j, accumulate acc_1_1
                            cute.gemm(tiled_mma_subchunk, tQKrQK_1_1, tQKrQ_1_j, tQKrKt_1_j, tQKrQK_1_1)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_1_1, tQKrK_1_j, tQKrKt_1_j, tKKrKK_1_1)

                        qk_1_0_val = tQKrQK_1_0.load()
                        qk_1_0_val = qk_1_0_val * self.scale
                        tQKrQK_1_0.store(qk_1_0_val)
                        qk_1_1_val = tQKrQK_1_1.load()
                        qk_1_1_val = qk_1_1_val * self.scale
                        tQKrQK_1_1.store(qk_1_1_val)
                        # first subchunk, wait for data ready
                        beta_handle = load_beta_consumer.wait()
                        # Fence due to normal load
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        sBeta_1 = sBeta_slice[None, 1]
                        for i in cutlass.range_constexpr(cute.size(tQKcMqk_subchunk)):
                            s, t = index_transform(*tQKcMqk_subchunk[i])
                            b = sBeta_1[s]
                            tKKrKK_1_0[i] *= b
                            tKKrKK_1_1[i] *= b

                        # R2S qk_1_0, kk_1_0
                        # first subchunk, acquire data
                        smem_kk_producer.acquire()
                        self.r2s_subchunk_acc(
                            1, 0, tKKrKK_1_0, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        p_producer.acquire()
                        self.r2s_subchunk_acc(1, 0, tQKrQK_1_0, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)
                        # R2S qk_1_1, kk_1_1
                        self.r2s_subchunk_acc(
                            1, 1, tKKrKK_1_1, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(1, 1, tQKrQK_1_1, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)

                        # Q/K2@K0, Q/K2@K1, Q/K2@K2
                        tQKrQK_2_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_2_0 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tQKrQK_2_1 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_2_1 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tQKrQK_2_2 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )
                        tKKrKK_2_2 = self.mma_sync_partition_c(
                            tiled_mma_subchunk, self.qk_kk_subchunk_mma_tiler, zero_fill=True
                        )

                        for j in cutlass.range(self.NK_SC):
                            tQKrQ_2_j, tQKrK_2_j = self.s2r_compute_subchunk_operand_A(
                                2,
                                j,
                                G_Q_tiled_copy,
                                G_Q_thr_copy,
                                Q_tiled_copy,
                                Q_thr_copy,
                                tv_layout_mma_A,
                                layout_g_first,
                                sGqkq_slice,
                                sQqk_slice,
                                sKqk_slice,
                            )

                            # S2R g_0_j, g_2_j_first again (different layouts for operand B)
                            sGqkq_2_j = sGqkq_slice[None, None, 2, (0, j)]
                            sG_first_2_j = cute.make_tensor(sGqkq_2_j.iterator, layout=layout_g_first)
                            tGsGfirst_2_j_kt = G_Kt_thr_copy.partition_S(sG_first_2_j)
                            tGrGfirst_2_j_kt = cute.make_fragment_like(tv_layout_mma_B, dtype=self.g_dtype)
                            tGrGfirst_2_j_kt_cv = G_Kt_thr_copy.retile(tGrGfirst_2_j_kt)
                            cute.copy(G_Kt_tiled_copy, tGsGfirst_2_j_kt, tGrGfirst_2_j_kt_cv)

                            tQKrKt_0_j = self.s2r_compute_subchunk_operand_B(
                                0,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_2_j_kt,
                            )

                            # q_2_j/k_2_j @ k_0_j, accumulate acc_2_0
                            cute.gemm(tiled_mma_subchunk, tQKrQK_2_0, tQKrQ_2_j, tQKrKt_0_j, tQKrQK_2_0)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_2_0, tQKrK_2_j, tQKrKt_0_j, tKKrKK_2_0)

                            tQKrKt_1_j = self.s2r_compute_subchunk_operand_B(
                                1,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_2_j_kt,
                            )

                            # q_2_j/k_2_j @ k_1_j, accumulate acc_2_1
                            cute.gemm(tiled_mma_subchunk, tQKrQK_2_1, tQKrQ_2_j, tQKrKt_1_j, tQKrQK_2_1)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_2_1, tQKrK_2_j, tQKrKt_1_j, tKKrKK_2_1)

                            tQKrKt_2_j = self.s2r_compute_subchunk_operand_B(
                                2,
                                j,
                                G_Kt_tiled_copy,
                                G_Kt_thr_copy,
                                Kt_tiled_copy,
                                Kt_thr_copy,
                                tv_layout_mma_B,
                                sGqkq_slice,
                                sKqk_slice,
                                tGrGfirst_2_j_kt,
                            )

                            # q_2_j/k_2_j @ k_2_j, accumulate acc_2_2
                            cute.gemm(tiled_mma_subchunk, tQKrQK_2_2, tQKrQ_2_j, tQKrKt_2_j, tQKrQK_2_2)
                            cute.gemm(tiled_mma_subchunk, tKKrKK_2_2, tQKrK_2_j, tQKrKt_2_j, tKKrKK_2_2)

                        qk_2_0_val = tQKrQK_2_0.load()
                        qk_2_0_val = qk_2_0_val * self.scale
                        tQKrQK_2_0.store(qk_2_0_val)
                        qk_2_1_val = tQKrQK_2_1.load()
                        qk_2_1_val = qk_2_1_val * self.scale
                        tQKrQK_2_1.store(qk_2_1_val)
                        qk_2_2_val = tQKrQK_2_2.load()
                        qk_2_2_val = qk_2_2_val * self.scale
                        tQKrQK_2_2.store(qk_2_2_val)
                        sBeta_2 = sBeta_slice[None, 2]
                        for i in cutlass.range_constexpr(cute.size(tQKcMqk_subchunk)):
                            s, t = index_transform(*tQKcMqk_subchunk[i])
                            b = sBeta_2[s]
                            tKKrKK_2_0[i] *= b
                            tKKrKK_2_1[i] *= b
                            tKKrKK_2_2[i] *= b

                        # R2S qk_2_0, kk_2_0
                        self.r2s_subchunk_acc(
                            2, 0, tKKrKK_2_0, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(2, 0, tQKrQK_2_0, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)
                        # R2S qk_2_1, kk_2_1
                        self.r2s_subchunk_acc(
                            2, 1, tKKrKK_2_1, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(2, 1, tQKrQK_2_1, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)
                        # R2S qk_2_2, kk_2_2
                        self.r2s_subchunk_acc(
                            2, 2, tKKrKK_2_2, sKK_inv_slice, O_tiled_copy_kk, O_thr_copy_kk, self.inverse_dtype
                        )
                        self.r2s_subchunk_acc(2, 2, tQKrQK_2_2, sQK_slice, O_tiled_copy, O_thr_copy, self.io_dtype)

                        # release Q, K, G
                        g_handle.release()
                        q_handle.release()
                        k_handle.release()

                    # wait for QK/KK ready
                    self.cuda_subchunk_wg_sync_barrier.arrive_and_wait()

                    # QK/KK epilogue mask
                    tQKrQK = cute.make_fragment_like(tQKrQK_cv, dtype=self.io_dtype)
                    tKKrKK = cute.make_fragment_like(tQKrQK_cv, dtype=self.inverse_dtype)
                    cute.copy(tiled_load_qk, thr_load_qk.partition_S(sQK_curr), tQKrQK)
                    cute.copy(tiled_load_kk, thr_load_kk.partition_S(sKK_inv_curr), tKKrKK)
                    # triangular mask and boundary mask
                    if cutlass.const_expr(self.is_varlen):
                        valid_len_chunk = seq_len - chunk_start
                    else:
                        valid_len_chunk = C
                    self.apply_qk_kk_mask(tQKcMqk, tQKrQK, tKKrKK, valid_len_chunk)
                    # R2S QK/KK
                    cute.copy(tiled_store_qk, thr_store_qk.retile(tQKrQK), thr_store_qk.partition_D(sQK_curr))
                    cute.copy(tiled_store_kk, thr_store_kk.retile(tKKrKK), thr_store_kk.partition_D(sKK_inv_curr))

                    # QK is ready to consume
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    # self.cuda_subchunk_wg_sync_barrier.arrive_and_wait()
                    # if should_debug:
                    #     cute.printf("chunk_idx={}, QK={}", idx, p_stage_idx)
                    #     cute.print_tensor(sQK_curr)
                    # self.cuda_subchunk_wg_sync_barrier.arrive_and_wait()
                    p_producer.commit()
                    p_producer.advance()

                    # NOTE: make sure KK mask is ready
                    self.cuda_subchunk_wg_sync_barrier.arrive_and_wait()

                    # inverse KK
                    self.compute_matrix_inverse_64x64(curr_sM_f16)

                    # S2R, scale with beta, convert to BF16, store back to smem `sM`
                    # TODO: make a repro for the cutedsl team
                    self.scale_M_inverse_with_beta(local_tidx, sBeta, curr_sM_f16, curr_sM)

                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    smem_kk_producer.commit()
                    smem_kk_producer.advance()
                    # self.cuda_subchunk_wg_sync_barrier.arrive_and_wait()
                    # if should_debug:
                    #     cute.printf("chunk_idx={}, inv(KK)={}", idx, smem_kk_stage_idx)
                    #     cute.print_tensor(curr_sM)
                    # self.cuda_subchunk_wg_sync_barrier.arrive_and_wait()

                    # after inverse KK, release Beta
                    load_beta_consumer.release()
                    load_beta_consumer.advance()

            else:
                for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                    idx = chunk_start // C
                    # wait for Q, K, G, Beta load
                    g_handle = load_g_consumer.wait_and_advance()
                    q_handle = load_q_consumer.wait_and_advance()
                    k_handle = load_k_consumer.wait_and_advance()
                    beta_handle = load_beta_consumer.wait_and_advance()

                    p_handle = p_producer.acquire_and_advance()
                    p_handle.commit()
                    smem_kk_handle = smem_kk_producer.acquire_and_advance()
                    smem_kk_handle.commit()

                    g_handle.release()
                    q_handle.release()
                    k_handle.release()
                    # release Beta
                    beta_handle.release()

        elif warp_idx == self.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            should_debug = PRINT_DEBUG and tidx == warp_idx * 32 and hidx == 0 and bidx == 0
            # TMA STORE
            # O: (D, S), column major
            # Apply domain_offset for varlen (use tok_offset for both input and output)
            tma_tensor_o_v = tma_tensor_o
            if cutlass.const_expr(self.is_varlen):
                tma_tensor_o_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_o)
            # (MMA_M, MMA_N, TILES_M, TILES_N, (H, B))
            gO_pre_partition = cute.flat_divide(tma_tensor_o_v, cute.select(self.vp_mma_tiler, mode=[0, 1]))

            # (MMA_M, MMA_N, TILES_M, TILES_N)
            gO_pre_partition = gO_pre_partition[None, None, None, None, (hidx, data_bidx)]

            # bSG_sO: ((ATOM_V, REST_V), STAGE)
            # bSG_gO: ((ATOM_V, REST_V), EPI_M, EPI_N, TILES_D, TILES_C)
            tma_atom_o, bSG_sO, bSG_gO = self.epilog_gmem_copy_and_partition(
                tma_atom_o,
                gO_pre_partition,
                self.epi_tile,
                sO,
            )

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tma_tensor_o: {tma_tensor_o}")
                print(f"bSG_sO: {bSG_sO}")
                print(f"bSG_gO_partitioned: {bSG_gO}")

            # Varlen tail tile handling: prepare modified TMA descriptor
            # For the last chunk of a non-last sequence that isn't chunk-aligned,
            # the TMA store would write past the sequence boundary into the next
            # sequence's output region. We prevent this by creating a modified TMA
            # descriptor with a truncated S dimension (like flashkda's approach).
            if cutlass.const_expr(self.is_varlen):
                # Get workspace pointer with proper TMA descriptor type and alignment
                ws_desc_ptr = self._tensormap_mgr.get_tensormap_ptr((workspace_iter + bidx * 128).align(128))
                need_tail_fixup = (seq_len % C != 0) & (bidx < B - 1)
                if need_tail_fixup:
                    # Create a GMEM tensor view with truncated S dimension
                    new_S = tok_offset + seq_len
                    o_tail = cute.make_tensor(
                        o_gmem.iterator,
                        cute.make_layout(
                            (D, new_S, (H, Int32(1))),
                            stride=(Int32(1), D * H, (D, D * H * S)),
                        ),
                    )
                    # Initialize: copy original TMA descriptor to workspace
                    cpasync.copy_tensormap(tma_atom_o, ws_desc_ptr)
                    # Update with truncated tensor (changes shape/stride in descriptor)
                    cpasync.update_tma_descriptor(tma_atom_o, o_tail, ws_desc_ptr)
                    # Fence: make descriptor update visible to subsequent TMA operations
                    cpasync.fence_tma_desc_release()

            # TMA STORE
            for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                idx = chunk_start // C

                smem_o_handle = smem_o_consumer.wait_and_advance()
                # TMA STORE O: SMEM -> GMEM
                if cutlass.const_expr(self.is_varlen):
                    remaining = seq_len - chunk_start
                    is_tail = (remaining < C) & (bidx < B - 1)
                    if is_tail:
                        # Use modified TMA descriptor for tail tile
                        # The modified descriptor has S bound = tok_offset + seq_len,
                        # so elements beyond the sequence boundary are silently dropped
                        cpasync.fence_tma_desc_acquire(ws_desc_ptr)
                        cute.copy(
                            tma_atom_o,
                            bSG_sO[None, smem_o_handle.index],
                            bSG_gO[(None, 0, 0, 0, idx)],
                            tma_desc_ptr=self._tensormap_mgr.get_tensormap_ptr(ws_desc_ptr, cute.AddressSpace.generic),
                        )
                    else:
                        cute.copy(tma_atom_o, bSG_sO[None, smem_o_handle.index], bSG_gO[(None, 0, 0, 0, idx)])
                else:
                    cute.copy(tma_atom_o, bSG_sO[None, smem_o_handle.index], bSG_gO[(None, 0, 0, 0, idx)])
                # Ensure smem_o has been released.
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                smem_o_handle.release()

        elif warp_idx == self.load_beta_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)
            local_tidx = tidx % (self.threads_per_warp * len([self.load_beta_warp_id]))
            for chunk_start in cutlass.range(0, seq_len, C, unroll=0):
                idx = chunk_start // C
                # Load beta into smem
                beta_handle = load_beta_producer.acquire_and_advance()
                # Fence due to normal load
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )

                s_idx = idx * C
                # Apply domain_offset for varlen beta access
                beta_v = beta
                if cutlass.const_expr(self.is_varlen):
                    beta_v = cute.domain_offset((tok_offset, (0, 0)), beta)
                beta_chunk = beta_v[(None, (hidx, data_bidx))]
                beta_chunk_layout = cute.make_layout((C, 1), stride=(H, 0))
                beta_chunk = cute.make_tensor(beta_chunk.iterator + s_idx * H, layout=beta_chunk_layout)

                if cutlass.const_expr(PRINT_DEBUG):
                    print(f"sBeta: {sBeta}")
                    print(f"beta_chunk: {beta_chunk}")
                # Load beta with boundary check for partial last chunk (varlen only)
                if cutlass.const_expr(self.is_varlen):
                    valid_len_beta = seq_len - chunk_start
                    for data_idx in cutlass.range(local_tidx, Constant.C, self.threads_per_warp):
                        if data_idx < valid_len_beta:
                            sBeta[data_idx, 0] = beta_chunk[data_idx, 0]
                        else:
                            sBeta[data_idx, 0] = cutlass.Float32(0.0)
                else:
                    for data_idx in cutlass.range(local_tidx, Constant.C, self.threads_per_warp):
                        sBeta[data_idx, 0] = beta_chunk[data_idx, 0]

                # Fence
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                beta_handle.commit()

        # Release tensor memory allocation lock
        tmem.relinquish_alloc_permit()
        # Sync before deallocating tmem
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        # Dealloc tmem buffer
        tmem.free(tmem_ptr_base)

        return

    @cute.jit
    def scale_state(self, flat: cute.Tensor, sG_last: cute.Tensor) -> cute.Tensor:
        # tTR_rKV: (Dk, Dv)
        # kv_f32 = flat.load()
        kv_f32 = flat
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"kv_f32: {kv_f32}")
        for i in cutlass.range(0, Constant.D, unroll_full=True):
            if not cutlass.const_expr(self.safe_gate):
                kv_f32[i] = kv_f32[i] * sG_last[i]
            else:
                # NOTE: when safe_gate=True, sG_last stores the original G values
                kv_f32[i] = kv_f32[i] * cute.exp2(sG_last[i], fastmath=self.use_fast_math)
        return kv_f32

    def tmem_load_kv16(self, local_tidx, tState):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.kv_mma_tiler,
            utils.LayoutEnum.ROW_MAJOR,
            self.io_dtype,
            self.io_dtype,
            self.kv_mma_tiler[:2],
            use_2cta_instrs=False,
        )
        fake_sState = cute.make_tensor(
            cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
            cute.dice(self.kv_mma_tiler, (1, 1, None)),
        )
        return self.make_tmem_load_and_partition(copy_atom_t2r, tState, (None, None, 0), local_tidx, fake_sState)

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
        # copy_atom_r2s = cute.make_copy_atom(
        #     # NOTE: TRANSPOSE
        #     cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
        #     self.io_dtype,
        # )
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
        # copy_atom_t2r = sm100_utils.get_tmem_load_op(
        #     mma_tiler,
        #     utils.LayoutEnum.ROW_MAJOR,
        #     self.io_dtype,
        #     self.acc_dtype,
        #     mma_tiler[:2],
        #     use_2cta_instrs=False,
        # )

        # In KDA, we need to make tv-layout row-wise to perform diagonal op.
        copy_atom_t2r = cute.make_copy_atom(
            # 32b x 32, TODO: ADJUST RMEM PEAK
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE),
            self.acc_dtype,
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
    def tmem_load_and_partition_kk(
        self,
        local_tidx,
        tKK,
    ):
        # 64,64
        copy_atom_t2r_kk = cute.make_copy_atom(
            # 32b x 8 x 8
            tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        fake_sKK = cute.make_tensor(
            cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
            cute.dice(self.kk_mma_tiler, (1, 1, None)),
        )
        # tKK: (EPITILE_M, EPITILE_N, STAGES)
        # Tiled Copy for one stage
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r_kk, tKK[None, None, 0])
        thr_t2r = tiled_t2r.get_slice(local_tidx)
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, STAGES)
        tTR_t = thr_t2r.partition_S(tKK)
        # (EPITILE_M, EPITILE_N)
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_s = thr_t2r.partition_D(fake_sKK)
        tTR_r = cute.make_rmem_tensor(
            tTR_s.shape,
            tKK.element_type,
        )
        return tiled_t2r, thr_t2r, tTR_t, tTR_r

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
        # copy_atom_t2r_qk = sm100_utils.get_tmem_load_op(
        #     self.qk_mma_tiler,
        #     # TODO:
        #     utils.LayoutEnum.ROW_MAJOR,
        #     self.io_dtype,
        #     self.acc_dtype,
        #     self.qk_epi_tile,
        #     use_2cta_instrs=False,
        # )
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
    def smem_store_and_partition_x(
        self,
        local_tidx: cutlass.Int32,
        smem_x: cute.Tensor,
        tiled_t2r_x: cute.TiledCopy,
        tXrX_t2r: cute.Tensor,
    ):
        # TODO: figure out why the selected smem store atom is not correct
        copy_atom_r2s_x = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.from_tensor(smem_x), self.io_dtype, self.acc_dtype, tiled_t2r_x
        )
        if cutlass.const_expr(PRINT_DEBUG):
            num_dp, num_bits, num_rep, pack = sm100_utils.get_tmem_copy_properties(tiled_t2r_x)
            print(f"num_dp={num_dp}, num_bits={num_bits}, num_rep={num_rep}, pack={pack}")
            print(f"copy_atom_r2s_x: {copy_atom_r2s_x}")
            print(f"tiled_t2r_x.layout_dst_tv_tiled: {tiled_t2r_x.layout_dst_tv_tiled}")
            print(f"tiled_t2r_x.tiler_mn: {tiled_t2r_x.tiler_mn}")

        tiled_r2s_x = cute.make_tiled_copy_D(copy_atom_r2s_x, tiled_t2r_x)
        thr_r2s_x = tiled_r2s_x.get_slice(local_tidx)

        # ((V, R), M, N, N_STAGE)
        tXsX_r2s = thr_r2s_x.partition_D(smem_x)

        # ((V, R), M, N)
        tXrX_r2s = thr_r2s_x.retile(tXrX_t2r)

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"copy_atom_r2s_x: {copy_atom_r2s_x}")
            print(f"tiled_t2r_x: {tiled_t2r_x}")
            print(f"tiled_r2s_x: {tiled_r2s_x}")
            print(f"thr_r2s_x: {thr_r2s_x}")
            print(f"before partition_D: {smem_x}")
            print(f"after partition_D, tXsX_r2s: {tXsX_r2s}")
            print(f"before retile tXrX_t2r: {tXrX_t2r}")
            print(f"after retile tXrX_r2s: {tXrX_r2s}")

        return tiled_r2s_x, thr_r2s_x, tXrX_r2s, tXsX_r2s

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
        show_debug_info=False,
        debug_name="X",
    ):
        copy_atom_r2s_x = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.from_tensor(smem_x), smem_x.element_type, self.acc_dtype, tiled_t2r_x
        )
        tiled_r2s_x = cute.make_tiled_copy_D(copy_atom_r2s_x, tiled_t2r_x)
        thr_r2s_x = tiled_r2s_x.get_slice(local_tidx)

        # ((V, R), M, N, N_STAGE)
        tXsX_r2s = thr_r2s_x.partition_D(smem_x)

        # ((V, R), M, N)
        tXrX_r2s = thr_r2s_x.retile(tXrX_t2r)

        if show_debug_info:
            print(f"-------------------- SMEM STORE: {debug_name}")
            print(f"copy_atom_r2s_x: {copy_atom_r2s_x}")
            print(f"tiled_t2r_x: {tiled_t2r_x}")
            print(f"thr_t2r_x: {thr_r2s_x}")
            print(f"before partition_D: {smem_x}")
            print(f"after partition_D, tXsX_r2s: {tXsX_r2s}")
            print(f"before retile tXrX_t2r: {tXrX_t2r}")
            print(f"after retile tXrX_r2s: {tXrX_r2s}")

        return tiled_r2s_x, thr_r2s_x, tXrX_r2s, tXsX_r2s

    def epilog_tmem_load_and_partition_acc(self, local_tidx, tIntra, smem_y):
        # copy_atom_t2r_inter2_intra2 = cute.make_copy_atom(
        #     tcgen05.Ld16x256bOp(tcgen05.Repetition(4), tcgen05.Pack.NONE),
        #     self.acc_dtype,
        # )
        # TODO: 16x256b might be faster, but 32x32b is easier to debug now.
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
        # TODO: 32dp ease DEBUGGING
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        # copy_atom_t2r = sm100_utils.get_tmem_load_op(
        #     mma_tiler,
        #     # self.o_layout,
        #     # TODO
        #     utils.LayoutEnum.ROW_MAJOR,
        #     self.io_dtype,
        #     self.acc_dtype,
        #     epitile,
        #     use_2cta_instrs,
        # )
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
    def apply_M_transform(
        self,
        kk_mat: cute.Tensor,  # (C, C) matrix from K*K^T, stored in registers - INPLACE MODIFIED
        beta_vec: cute.Tensor,  # (C, 1) vector of beta values
        index_kk: cute.Tensor,  # Index tensor with row/col information for strict triangular structure
        kk_f16_mat: cute.Tensor,
    ):
        """
        Compute M = I + StrictTril(beta * KK^T) INPLACE

        Modifies kk_mat to become M matrix.

        Args:
            kk_mat: K*K^T matrix from GEMM, shape (C, C) - MODIFIED INPLACE to M matrix
            beta_vec: KDA beta scaling factor, shape (C, 1)
            index_kk: Index tensor with row/col information (similar to apply_mask usage)

        The formula is:
        - M[i,j] = 1.0 if i == j (identity on diagonal)
        - M[i,j] = beta[i] * KK[i,j] if i > j (strict lower triangular)
        - M[i,j] = 0.0 if i < j (upper triangular is zero)
        """
        # Iterate through all elements of the matrix using index information
        for i in cutlass.range_constexpr(cute.size(kk_mat)):
            # Get row and column indices from index tensor
            row, col = index_kk[i]

            if row == col:
                # Diagonal: M[i,i] = 1.0 (identity)
                # kk_mat[i] = cutlass.Float32(1.0)
                kk_f16_mat[i] = cutlass.Float16(1.0)
            elif row > col:
                # Strict lower triangular: M[i,j] = beta[row] * KK[i,j]
                # TODO: cache beta to register since row does not change here.
                beta_val = beta_vec[row].to(cutlass.Float32)
                kk_val = kk_mat[i].to(cutlass.Float32)
                # kk_mat[i] = beta_val * kk_val
                kk_f16_mat[i] = (beta_val * kk_val).to(cutlass.Float16)
            else:
                # Upper triangular: M[i,j] = 0.0
                # kk_mat[i] = cutlass.Float32(0.0)
                kk_f16_mat[i] = cutlass.Float16(0.0)

    @cute.jit
    def compute_matrix_inverse_64x64(
        self,
        s_mat: cute.Tensor,  # Input M matrix in smem, shape (64, 64)
    ):
        """
        Compute M^{-1} for 64x64 lower triangular matrix using block-wise Schur complement.

        Based on flat_collective_inverse.hpp algorithm:
        1. Divide into 8x8 blocks
        2. Compute 8x8 diagonal block inverses (lower triangular blocks)
        3. Use Schur complement for below-diagonal blocks
        4. Progressively combine: 8x8 -> 16x16 -> 32x32 -> 64x64

        For a lower triangular matrix L:
            inv([A  0 ]) = [inv(A)  0                    ]
               [C  D ]   [-inv(D)C*inv(A)  inv(D)        ]

        Args:
            s_mat: Input M matrix (lower triangular) in smem, shape (64, 64)
        """
        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx % 128

        # Stage 1: Invert all 8 diagonal 8x8 blocks
        t8x8mat = cute.flat_divide(s_mat, (8, 8))
        if cutlass.const_expr(PRINT_DEBUG):
            if tidx == 0:
                cute.printf("---------- Raw first 8x8 block:")
                t4x4mat = cute.flat_divide(t8x8mat[None, None, 0, 0], (4, 4))
                cute.print_tensor(t4x4mat[None, None, 0, 0])
                cute.print_tensor(t4x4mat[None, None, 0, 1])
                cute.print_tensor(t4x4mat[None, None, 1, 0])
                cute.print_tensor(t4x4mat[None, None, 1, 1])

        # Block size progression: 8x8 -> 16x16 -> 32x32 -> 64x64
        # Process in stages

        # Stage 1: Invert all 8 diagonal 8x8 blocks
        t8x8mat = cute.flat_divide(s_mat, (8, 8))
        if tidx < 64:
            self.compute_diagonal_inverse_8x8(
                t8x8mat[None, None, tidx // 8, tidx // 8],
                tidx % 8,
                tidx // 8,  # Pass block_id for correct shuffle masking
            )
        self.cuda_wg_sync_barrier.arrive_and_wait()

        if cutlass.const_expr(PRINT_DEBUG) and tidx == 0:
            cute.printf("---------- After Stage1, first 8x8 block:")
            cute.print_tensor(t8x8mat[None, None, 0, 0])

        # Stage 2: Invert all 4 diagonal 16x16 blocks
        t16x16mat = cute.flat_divide(s_mat, (16, 16))
        self.compute_diagonal_inverse_8x8_to_16x16(
            t16x16mat[None, None, tidx // 32, tidx // 32],
        )
        # Synchronize after stage 2
        self.cuda_wg_sync_barrier.arrive_and_wait()

        if cutlass.const_expr(PRINT_DEBUG) and tidx == 0:
            cute.printf("-------------- After stage 2 inverse 16x16, first 16x16 block")
            cute.print_tensor(t16x16mat[None, None, 0, 0])

        # Stage 3: Invert all 2 diagonal 32x32 blocks
        t32x32mat = cute.flat_divide(s_mat, (32, 32))
        if tidx < 64:
            self.compute_diagonal_inverse_16x16_to_32x32(
                t32x32mat[None, None, tidx // 32, tidx // 32],
            )
        self.cuda_wg_sync_barrier.arrive_and_wait()
        if cutlass.const_expr(PRINT_DEBUG) and tidx == 0:
            cute.printf("-------------- After stage 3 inverse 32x32, first 32x32 block")
            cute.print_tensor(t32x32mat[None, None, 0, 0])

        # Stage 4: Invert the full 64x64 matrix
        self.compute_diagonal_inverse_32x32_to_64x64(s_mat)
        if cutlass.const_expr(PRINT_DEBUG) and tidx == 0:
            cute.printf("-------------- Final inverse 64x64 block\n")
            cute.print_tensor(s_mat, verbose=False)

    @cute.jit
    def scale_M_inverse_with_beta(
        self,
        local_tidx,
        beta_vec: cute.Tensor,  # (C, 1) vector of beta values from gmem
        sM_f16: cute.Tensor,  # Input M^{-1} in smem, Float16, shape (64, 64, STAGE)
        sM: cute.Tensor,  # Output M^{-1}*beta in smem, BFloat16, shape (64, 64, STAGE)
    ):
        """
        Scale M^{-1} by beta and convert from Float16 to BFloat16.

        For each row i: M_scaled[i, :] = beta[i] * M_inverse[i, :]

        Uses 128 threads (4 warps × 32 threads) to process 64×64 matrix.
        Each thread processes one row (64 elements), with 2 rows per thread
        across all 128 threads covering 64 rows in 2 passes, or each thread
        handles half a row.

        Strategy: 128 threads, 64 rows, each pair of threads handles one row.
        Thread 2*i and 2*i+1 handle row i, with 32 elements each.

        Args:
            beta_vec: KDA beta scaling factor, shape (C, 1)
            sM_f16: Input M^{-1} matrix in smem (Float16), shape (64, 64, STAGE)
            sM: Output scaled matrix in smem (BFloat16), shape (64, 64, STAGE)
        """

        # Each thread handles half a row (32 elements)
        # Thread layout: thread i handles row (i//2), columns [(i%2)*32 : (i%2+1)*32]
        col_start = (local_tidx % 2) * 32  # Column offset: 0 or 32

        # Create copy atom for smem <-> rmem transfers (32 elements at once)
        # FIXME: figure out why num__bits_per_copy == 512 does not work here
        num_bits_per_copy = 128
        copy_atom_s2r_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            sM_f16.element_type,
            # (64,64) ; copy half row
            num_bits_per_copy=num_bits_per_copy,
        )
        copy_atom_r2s_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            sM.element_type,
            # (64,64) ; copy half row
            num_bits_per_copy=num_bits_per_copy,
        )

        shape_x = cute.coalesce(sM_f16.layout, target_profile=(1, 1)).shape

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"shape_x: {shape_x}")

        num_elements_per_thread = 32
        num_threads_per_row = shape_x[1] // num_elements_per_thread
        # NOTE: Assume 128 cuda core threads
        num_threads_per_col = 128 // num_threads_per_row
        # (64,2):(2,1)
        thread_layout = cute.make_layout(
            (num_threads_per_col, num_threads_per_row),
            stride=(num_threads_per_row, 1),
        )
        val_layout = cute.make_layout((1, num_elements_per_thread))
        tiled_s2r_x = cute.make_tiled_copy_tv(
            copy_atom_s2r_x,
            thread_layout,
            val_layout,
        )
        thr_s2r_x = tiled_s2r_x.get_slice(local_tidx)

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"thr_s2r_x: {thr_s2r_x}")

        # Partition shared tensor for smem load Bt
        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tXsX_s2r = thr_s2r_x.partition_S(sM_f16)

        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tXrX_s2r = cute.make_rmem_tensor(
            tXsX_s2r.shape,
            sM_f16.element_type,
        )
        tXrX_bf16 = cute.make_rmem_tensor_like(
            tXsX_s2r,
            sM.element_type,
        )

        cute.copy(tiled_s2r_x, tXsX_s2r, tXrX_s2r)
        # TODO: check fence
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"------ DOT tXrX_s2r: {tXrX_s2r}")
        # M^{-1} * diag(beta): Load values and prepare beta scaling vector
        m = cute.make_rmem_tensor_like(tXrX_s2r, cutlass.Float32)
        m.store(tXrX_s2r.load().to(cutlass.Float32))
        # Each thread loads its corresponding beta values (32 elements)
        for j in cutlass.range_constexpr(32):
            m[j] = m[j] * cutlass.Float32(beta_vec[col_start + j])

        assert sM.element_type == cutlass.BFloat16
        tXrX_bf16.store(m.load().to(sM.element_type))

        tiled_r2s_x = cute.make_tiled_copy_tv(
            copy_atom_r2s_x,
            thread_layout,
            val_layout,
        )
        thr_r2s_x = tiled_r2s_x.get_slice(local_tidx)

        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tXsX_r2s = thr_r2s_x.partition_D(sM)

        cute.copy(tiled_r2s_x, tXrX_bf16, tXsX_r2s)

        # TODO: check fence
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def compute_diagonal_inverse_8x8(
        self,
        s_block: cute.Tensor,  # Input 8x8 block in smem
        lane_id: int,
        block_id: int = -1,  # Block ID for correct shuffle mask calculation
    ):
        """
        Compute inverse of a diagonal 8x8 lower triangular block.

        Each warp processes one 8x8 block. Each lane (0-7) computes one column.
        """
        # Every thread fix a row
        row = self.load_row_mat8x8(s_block, lane_id)

        # Calculate the actual physical starting position of this block in the warp
        # Blocks 0-3 are in Warp 0 (tidx 0-31), Blocks 4-7 are in Warp 1 (tidx 32-63)
        # Each block starts at: warp_base + block_index_in_warp * 8
        # where block_index_in_warp = block_id % 4
        actual_block_base = (block_id % 4) * 8  # 0, 8, 16, or 24 within the warp

        for src_row in cutlass.range_constexpr(8 - 1):
            row_scale = row[src_row] * cutlass.Float32(-1)
            for i in cutlass.range(src_row, unroll_full=True):
                # Correct target lane: within the block, we need threads at positions
                # actual_block_base + 0, actual_block_base + 1, ..., actual_block_base + 7
                # The src_row thread (0-7 within the block) maps to actual_block_base + src_row
                # in the warp
                target_lane = actual_block_base + src_row

                src_row_value = cute.arch.shuffle_sync_op(
                    value=row[i],
                    offset=target_lane,  # Use actual position, not just src_row
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
                if lane_id > src_row:
                    row[i] += row_scale * src_row_value
            if lane_id > src_row:
                row[src_row] = row_scale
        # Store row
        self.store_row_mat8x8(s_block, row, lane_id)

    @cute.jit
    def load_row_mat8x8(
        self,
        mat: cute.Tensor,
        idx: int,
    ) -> cute.Tensor:
        copy_atom_s2r_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mat.element_type,
            num_bits_per_copy=mat.element_type.width * 8,
        )
        row_tensor = cute.make_rmem_tensor(cute.make_layout(8), mat.element_type)
        cute.copy(copy_atom_s2r_x, mat[idx, None], row_tensor[None])
        row_f32_tensor = cute.make_rmem_tensor_like(row_tensor, cutlass.Float32)
        row_f32_tensor.store(row_tensor.load().to(cutlass.Float32))
        return row_f32_tensor

    @cute.jit
    def store_row_mat8x8(
        self,
        mat: cute.Tensor,
        row: cute.Tensor,
        idx: int,
    ) -> cute.Tensor:
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"store_row_mat8x8: idx={idx}, row={row}, mat={mat}")
        row_bf16 = cute.make_rmem_tensor_like(row, mat.element_type)
        row_bf16.store(row.load().to(mat.element_type))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mat.element_type,
            num_bits_per_copy=mat.element_type.width * 8,
        )
        cute.copy(copy_atom, row_bf16[None], mat[idx, None])

    @cute.jit
    def make_op_a_from_acc_rmem_16x8x8(
        self,
        acc_dtype: cute.Numeric,
        ab_dtype: cute.Numeric,
        acc_tensor: cute.Tensor,
    ):
        # For 16x8x8, ACC result 16x8 can directly serve as operand A
        a_tensor = cute.make_rmem_tensor_like(
            acc_tensor,
            dtype=ab_dtype,
        )
        a_tensor.store(acc_tensor.load().to(ab_dtype))
        return a_tensor

    def compute_diagonal_inverse_8x8_to_16x16(
        self,
        mat: cute.Tensor,  # Input 16x16 block in smem
    ):
        """
        Compute inverse of a diagonal 16x16 lower triangular block using
        two 8x8 blocks and Schur complement.
        """
        dtype = mat.element_type
        mat8x8_2x2 = cute.flat_divide(mat, (8, 8))

        mma_atom_shape = (16, 8, 8)
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=dtype,
            acc_dtype=self.acc_dtype,
            shape_mnk=mma_atom_shape,
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
        )
        # TODO: check transpose for column major mat
        copy_op_s2r = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=1)
        copy_op_s2r_t = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=1)
        copy_op_r2s = cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=1)
        copy_atom_s2r = cute.make_copy_atom(copy_op_s2r, mat.element_type)
        copy_atom_s2r_t = cute.make_copy_atom(copy_op_s2r_t, mat.element_type)
        copy_atom_r2s = cute.make_copy_atom(copy_op_r2s, mat.element_type)

        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32

        thr_mma = tiled_mma.get_slice(lane_id)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"thr_mma: {thr_mma}")
            print(f"tiled_mma: {tiled_mma}")

        D_tiled_copy = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        C_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        A_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        O_tiled_copy = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        sDInv = mat8x8_2x2[None, None, 1, 1]
        sC = mat8x8_2x2[None, None, 1, 0]
        sAInv = mat8x8_2x2[None, None, 0, 0]
        sO = mat8x8_2x2[None, None, 1, 0]

        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1, 0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1, 0]))

        # Padding by broadcast, need to figure out why cutlass version choose a operand of (2,0)
        sDInv_bcast = cute.make_tensor(
            sDInv.iterator, cute.blocked_product(sDInv.layout, cute.make_layout((2, 1), stride=(0, 0)))
        )
        sO_bcast = cute.make_tensor(sO.iterator, cute.blocked_product(sO.layout, cute.make_layout((2, 1), stride=(0, 0))))

        a_shape = cute.dice(mma_atom_shape, (1, None, 1))
        c_shape = cute.dice(mma_atom_shape, (1, 1, None))
        tOrDInv = thr_mma.make_fragment_A(tiled_mma.partition_shape_A(a_shape))
        tOrC = thr_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAInv = thr_mma.make_fragment_B(thr_mma.partition_B(sAInv))

        tDCrDC = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))
        tOrO = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))

        # ((vecsize, numvec), M, N)
        tOsDInv = D_thr_copy.partition_S(sDInv_bcast)
        tOrDInv_cv = D_thr_copy.retile(tOrDInv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAInv = A_thr_copy.partition_S(sAInv)
        tOrAInv_cv = A_thr_copy.retile(tOrAInv)
        tOsO = O_thr_copy.partition_D(sO_bcast)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(C_tiled_copy, tOsC, tOrC_cv)

        tDInv_src = tOsDInv
        tDInv_dst = tOrDInv_cv
        # tDInv_src = tOsDInv[(None, 0), None, None]
        # tDInv_dst = tOrDInv_cv[(None, 0), None, None]
        cute.copy(D_tiled_copy, tDInv_src, tDInv_dst)

        tDCrDC.fill(0.0)  # Clear C for D = A*B + C
        cute.gemm(tiled_mma, tDCrDC, tOrDInv, tOrC, tDCrDC)
        tDCrDC.store(tDCrDC.load() * cutlass.Float32(-1))

        tOrDC = self.make_op_a_from_acc_rmem_16x8x8(
            self.acc_dtype,
            mat.element_type,
            tDCrDC,
        )
        cute.copy(A_tiled_copy, tOsAInv, tOrAInv_cv)
        tOrO_cv.fill(0.0)  # Clear O for O = A*B + O
        cute.gemm(tiled_mma, tOrO_cv, tOrDC, tOrAInv, tOrO_cv)

        tOrO_cv_half = tOrO_cv[(None, 0), None, None]
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tOrO_cv_half: {tOrO_cv_half}")
        tOrO_f16 = cute.make_rmem_tensor_like(tOrO_cv_half, mat.element_type)
        tOrO_f16.store(tOrO_cv[(None, 0), None, None].load().to(mat.element_type))

        # NOTE: group here to make cutedsl happy
        src_shape = tOrO_f16.shape
        src_stride = tOrO_f16.layout.stride
        dst_shape = tOsO[(None, 0), None, None].shape
        dst_stride = tOsO[(None, 0), None, None].layout.stride
        tOrO_src = cute.make_tensor(
            tOrO_f16.iterator,
            layout=cute.make_layout(
                ((src_shape[0], 1), src_shape[1], src_shape[2]), stride=((src_stride[0], 0), src_stride[1], src_stride[2])
            ),
        )
        tOsO_dst = cute.make_tensor(
            tOsO.iterator,
            layout=cute.make_layout(
                ((dst_shape[0], 1), dst_shape[1], dst_shape[2]), stride=((dst_stride[0], 0), dst_stride[1], dst_stride[2])
            ),
        )
        # tOrO_src = tOrO_f16
        # tOsO_dst = tOsO[(None, 0), None, None]
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"tOrO_src: {tOrO_src}")
            print(f"tOsO_dst: {tOsO_dst}")
        cute.copy(O_tiled_copy, tOrO_src, tOsO_dst)

    def canonical_lane_id(self):
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        return lane_id

    def compute_diagonal_inverse_16x16_to_32x32(
        self,
        mat: cute.Tensor,  # Input 16x16 block in smem
    ):
        """
        Compute inverse of a diagonal 16x16 lower triangular block using
        two 8x8 blocks and Schur complement.
        """
        dtype = mat.element_type
        lane_id = self.canonical_lane_id()
        mat16x16_2x2 = cute.flat_divide(mat, (16, 16))
        mma_atom_shape = (16, 8, 16)
        mma_tiler = (16, 16, 16)
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=dtype,
            acc_dtype=self.acc_dtype,
            shape_mnk=mma_atom_shape,
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=mma_tiler,
        )
        thr_mma = tiled_mma.get_slice(lane_id)
        copy_atom_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
            mat.element_type,
        )
        copy_atom_s2r_t = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2),
            mat.element_type,
        )
        copy_atom_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
            mat.element_type,
        )
        D_tiled_copy = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        C_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        A_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        O_tiled_copy = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma)
        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        sDInv = mat16x16_2x2[None, None, 1, 1]
        sC = mat16x16_2x2[None, None, 1, 0]
        sAInv = mat16x16_2x2[None, None, 0, 0]
        sO = mat16x16_2x2[None, None, 1, 0]

        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1, 0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1, 0]))

        c_shape = cute.dice(mma_tiler, (1, 1, None))

        tOrDInv = thr_mma.make_fragment_A(thr_mma.partition_A(sDInv))
        tOrC = thr_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAInv = thr_mma.make_fragment_B(thr_mma.partition_B(sAInv))

        tDCrDC = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))
        tOrO = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))

        tOsDInv = D_thr_copy.partition_S(sDInv)
        tOrDInv_cv = D_thr_copy.retile(tOrDInv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAInv = A_thr_copy.partition_S(sAInv)
        tOrAInv_cv = A_thr_copy.retile(tOrAInv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(D_tiled_copy, tOsDInv, tOrDInv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)

        tDCrDC.fill(0.0)  # Clear C for D = A*B + C
        cute.gemm(tiled_mma, tDCrDC, tOrDInv, tOrC, tDCrDC)
        tDCrDC.store(tDCrDC.load() * cutlass.Float32(-1))

        tOrDC = self.make_acc_as_a(
            tDCrDC,
            tiled_mma,
            mat.element_type,
        )
        cute.copy(A_tiled_copy, tOsAInv, tOrAInv_cv)
        tOrO.fill(0.0)  # Clear O for O = A*B + O
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAInv, tOrO)

        tOrO_f16 = cute.make_rmem_tensor_like(tOrO_cv, mat.element_type)
        tOrO_f16.store(tOrO_cv.load().to(mat.element_type))
        cute.copy(O_tiled_copy, tOrO_f16, tOsO)

    def convert_layout_c_to_a(
        self,
        c_layout: cute.Layout,
        tiled_mma: cute.TiledMma,
    ):
        # TODO: VERIFY THIS
        cfrag_atom_size = cute.size(c_layout.shape[0])
        afrag_atom_size = cute.size(tiled_mma.tv_layout_A.shape[1])
        ratio = afrag_atom_size // cfrag_atom_size
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"cfrag_atom_size: {cfrag_atom_size}, afrag_atom_size: {afrag_atom_size}")

        if ratio == 1:
            return c_layout

        divided = cute.logical_divide(c_layout, (None, None, ratio))
        a_layout = cute.make_layout(
            (cute.flatten((divided.shape[0], divided.shape[2][0])), divided.shape[1], divided.shape[2][1])
        )
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"c_layout: {c_layout}")
            print(f"a_layout: {a_layout}")
        return a_layout

    def make_acc_as_a(self, acc: cute.Tensor, tiled_mma: cute.TiledMma, dtype: cute.Numeric):
        a_layout = self.convert_layout_c_to_a(acc.layout, tiled_mma)
        a_tensor = cute.make_rmem_tensor(a_layout, dtype=dtype)
        op_as_acc = cute.make_tensor(a_tensor.iterator, layout=acc.layout)
        op_as_acc.store(acc.load().to(dtype))
        return a_tensor

    @cute.jit
    def compute_diagonal_inverse_32x32_to_64x64(
        self,
        mat: cute.Tensor,  # Input 16x16 block in smem
    ):
        """
        Compute inverse of a diagonal 16x16 lower triangular block using
        two 8x8 blocks and Schur complement.
        """
        # TODO: try tcgen05 tensor core here since M is 32
        mat32x32_2x2 = cute.flat_divide(mat, (32, 32))
        mat_16x2_2x2 = cute.logical_divide(mat32x32_2x2, (16, 16))

        warp_id_wg = cute.arch.warp_idx() % 4
        x = warp_id_wg // 2
        y = warp_id_wg % 2

        lane_id = self.canonical_lane_id()
        mma_atom_shape = (16, 8, 16)
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=mat.element_type,
            acc_dtype=self.acc_dtype,
            shape_mnk=mma_atom_shape,
        )
        mma_tiler1 = (16, 16, 32)
        mma_tiler2 = (16, 32, 16)
        tiled_mma1 = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=mma_tiler1,
        )
        tiled_mma2 = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=mma_tiler2,
        )
        thr_mma1 = tiled_mma1.get_slice(lane_id)
        thr_mma2 = tiled_mma2.get_slice(lane_id)
        copy_atom_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mat.element_type,
        )
        copy_atom_s2r_t = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mat.element_type,
        )
        copy_atom_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mat.element_type,
        )

        D_tiled_copy = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma1)
        C_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma1)
        A_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma2)
        O_tiled_s2r = cute.make_tiled_copy_C(copy_atom_s2r, tiled_mma2)
        O_tiled_r2s = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma2)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_s2r = O_tiled_s2r.get_slice(lane_id)
        O_thr_r2s = O_tiled_r2s.get_slice(lane_id)

        sDInv = mat_16x2_2x2[(None, y), None, 1, 1]
        sC = mat_16x2_2x2[None, (None, x), 1, 0]
        sAInv = mat_16x2_2x2[(None, x), None, 0, 0]
        sO = mat_16x2_2x2[(None, y), None, 1, 0]

        # Make oprand B Col-Major
        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1, 0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1, 0]))

        tOrDInv = thr_mma1.make_fragment_A(thr_mma1.partition_A(sDInv))
        tOrC = thr_mma1.make_fragment_B(thr_mma1.partition_B(sC))
        tOrAInv = thr_mma2.make_fragment_B(thr_mma2.partition_B(sAInv))

        tDCrDC = thr_mma1.make_fragment_C(tiled_mma1.partition_shape_C((16, 16)))
        tOrO = thr_mma2.make_fragment_C(tiled_mma2.partition_shape_C((16, 32)))

        tOsDInv = D_thr_copy.partition_S(sDInv)
        tOrDInv_cv = D_thr_copy.retile(tOrDInv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAInv = A_thr_copy.partition_S(sAInv)
        tOrAInv_cv = A_thr_copy.retile(tOrAInv)

        cute.copy(D_tiled_copy, tOsDInv, tOrDInv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)

        tDCrDC.fill(0.0)  # Clear C for D = A*B + C
        cute.gemm(tiled_mma1, tDCrDC, tOrDInv, tOrC, tDCrDC)
        tDCrDC.store(tDCrDC.load() * cutlass.Float32(-1))

        tOrDC = self.make_acc_as_a(
            tDCrDC,
            tiled_mma2,
            mat.element_type,
        )

        cute.copy(A_tiled_copy, tOsAInv, tOrAInv_cv)
        tOrO.fill(0.0)  # Clear O for O = A*B
        cute.gemm(tiled_mma2, tOrO, tOrDC, tOrAInv, tOrO)

        tOrO_f16 = cute.make_rmem_tensor_like(tOrO, mat.element_type)
        tOrO_f16.store(tOrO.load().to(mat.element_type))

        # Make sure tOsC are consumed since we've 4 warps here.
        self.cuda_wg_sync_barrier.arrive_and_wait()

        tOsO = O_thr_r2s.partition_D(sO)
        tOrO_cvt_cv = O_thr_r2s.retile(tOrO_f16)

        # Warp with x=0 writes the result
        if x == 0:
            cute.copy(O_tiled_r2s, tOrO_cvt_cv, tOsO)

        self.cuda_wg_sync_barrier.arrive_and_wait()

        # Warp with x=1 reads and accumulates (reduce operation)
        if x == 1:
            # reduce to get correct results
            tOrO_red = cute.make_rmem_tensor_like(tOrO_f16)
            tOsO_s = O_thr_s2r.partition_S(sO)
            tOrO_red_cv = O_thr_s2r.retile(tOrO_red)
            cute.copy(O_tiled_s2r, tOsO_s, tOrO_red_cv)
            tOrO_f16.store(tOrO_f16.load() + tOrO_red.load())
            cute.copy(O_tiled_r2s, tOrO_cvt_cv, tOsO)

        self.cuda_wg_sync_barrier.arrive_and_wait()

    @cute.jit
    def apply_mask(
        self,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        p: cute.Tensor,
        debug: bool = False,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        # Apply causal mask
        for i in cutlass.range_constexpr(cute.size(acc_qk)):
            index_q, index_k = index_transform(*index_qk[i])
            if debug:
                cute.printf("index_qk, {},{}", index_q, index_k)
            # Mask causal
            if index_q < index_k:
                acc_qk[i] = cutlass.Float32(0.0)
                p[i] = cutlass.BFloat16(0.0)
            else:
                p[i] = (acc_qk[i]).to(self.q_dtype)

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
        debug_name=None,
        no_cta_coord=False,
        batch_idx=None,
    ):
        _, hidx, bidx = cute.arch.block_idx()
        if batch_idx is not None:
            bidx = batch_idx
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
            (None, None, (hidx, bidx)) if not no_cta_coord else (None, None, None),
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
    def get_smem_tensor_sG_last(
        self,
        storage,
        layout: cute.Layout,
        swizzle=None,
    ):
        sG_last = storage.sG_last.get_tensor(layout, swizzle=None)
        return sG_last

    @cute.jit
    def tma_partition_for_mma_operand(
        self,
        tma_atom_x,
        tma_tensor_x,
        smem_x,
        tile_shape,
        tiled_mma,
        operand_mode,
        debug_name=None,
        batch_idx=None,
    ):
        tCgX = self.local_tile_partition_for_mma_operand(
            tensor_x=tma_tensor_x,
            tile_shape=tile_shape,
            tiled_mma=tiled_mma,
            operand_mode=operand_mode,
            debug_name=debug_name,
            batch_idx=batch_idx,
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

    # ===========
    # Utility functions for Ampere-style mma.sync, used for subchunk computation
    @cute.jit
    def mma_sync_partition_c(
        self, tiled_mma: cute.atom.TiledMma, tile_shape_mnk: cute.Shape, zero_fill: cutlass.Constexpr[bool] = True
    ):
        acc_shape = tiled_mma.partition_shape_C(tile_shape_mnk[:2])
        tCrC = tiled_mma.make_fragment_C(acc_shape)
        if cutlass.const_expr(zero_fill):
            tCrC.fill(0.0)
        return tCrC

    @cute.jit
    def s2r_compute_subchunk_operand_A(
        self,
        subchunk_idx: cutlass.Constexpr[int],  # [0, 3] subchunk index
        nk: cutlass.Constexpr[int],  # [0, 1] tile size for head dim 128
        g_tiled_copy: cute.atom.TiledCopy,
        g_thr_copy: cute.atom.ThrCopy,
        q_k_tiled_copy: cute.atom.TiledCopy,
        q_k_thr_copy: cute.atom.ThrCopy,
        tv_layout_mma_A: cute.Layout,
        layout_g_first: cute.Layout,  # for make g_first tensor
        sG_slice: cute.Tensor,
        sQ_slice: cute.Tensor,
        sK_slice: cute.Tensor,
    ):
        # S2R g, g_first
        sG = sG_slice[None, None, subchunk_idx, (0, nk)]
        tQKsG = g_thr_copy.partition_S(sG)
        tQKrG = cute.make_fragment_like(tv_layout_mma_A, dtype=self.g_dtype)
        tQKrG_cv = g_thr_copy.retile(tQKrG)
        cute.copy(g_tiled_copy, tQKsG, tQKrG_cv)

        # TODO: do register shuffle g to get g_first, reduce smem load
        sG_first = cute.make_tensor(sG.iterator, layout=layout_g_first)
        tQKsGfirst = g_thr_copy.partition_S(sG_first)
        tQKrGfirst = cute.make_fragment_like(tv_layout_mma_A, dtype=self.g_dtype)
        tQKrGfirst_cv = g_thr_copy.retile(tQKrGfirst)
        cute.copy(g_tiled_copy, tQKsGfirst, tQKrGfirst_cv)

        # gqn = exp2(g - g_first[None, :]), reuse g
        g_val = tQKrG.load()
        g_first_val = tQKrGfirst.load()
        g_val = cute.exp2(g_val - g_first_val, fastmath=self.use_fast_math)
        tQKrG.store(g_val)

        # S2R q, k
        sQ = sQ_slice[None, None, subchunk_idx, (0, nk)]
        sK = sK_slice[None, None, subchunk_idx, (0, nk)]
        tQKrQ = cute.make_fragment_like(tv_layout_mma_A, dtype=self.q_dtype)
        tQKrK = cute.make_fragment_like(tv_layout_mma_A, dtype=self.k_dtype)
        tQKsQ = q_k_thr_copy.partition_S(sQ)
        tQKsK = q_k_thr_copy.partition_S(sK)
        tQKrQ_cv = q_k_thr_copy.retile(tQKrQ)
        tQKrK_cv = q_k_thr_copy.retile(tQKrK)
        cute.copy(q_k_tiled_copy, tQKsQ, tQKrQ_cv)
        cute.copy(q_k_tiled_copy, tQKsK, tQKrK_cv)

        # compute q/k * gqn, reuse q/k
        q_val = tQKrQ.load().to(cutlass.Float32)
        q_val = q_val * g_val
        q_val = q_val.to(self.io_dtype)
        tQKrQ.store(q_val)
        k_val = tQKrK.load().to(cutlass.Float32)
        k_val = k_val * g_val
        k_val = k_val.to(self.io_dtype)
        tQKrK.store(k_val)

        return tQKrQ, tQKrK

    @cute.jit
    def s2r_compute_subchunk_operand_B(
        self,
        subchunk_idx: cutlass.Constexpr[int],  # [0, 3] subchunk index
        nk: cutlass.Constexpr[int],  # [0, 1] tile size for head dim 128
        g_tiled_copy: cute.atom.TiledCopy,
        g_thr_copy: cute.atom.ThrCopy,
        kt_tiled_copy: cute.atom.TiledCopy,
        kt_thr_copy: cute.atom.ThrCopy,
        tv_layout_mma_B: cute.Layout,
        sG_slice: cute.Tensor,
        sK_slice: cute.Tensor,
        rG_first: cute.Tensor,
    ):
        # S2R g
        sG = sG_slice[None, None, subchunk_idx, (0, nk)]
        tQKsG = g_thr_copy.partition_S(sG)
        tQKrG = cute.make_fragment_like(tv_layout_mma_B, dtype=self.g_dtype)
        tQKrG_cv = g_thr_copy.retile(tQKrG)
        cute.copy(g_tiled_copy, tQKsG, tQKrG_cv)

        # compute gktn = exp2(g_first - g), reuse g
        g_val = tQKrG.load()
        g_first_val = rG_first.load()
        g_val = cute.exp2(g_first_val - g_val, fastmath=self.use_fast_math)
        tQKrG.store(g_val)

        # S2R k
        sK = sK_slice[None, None, subchunk_idx, (0, nk)]
        tQKrKt = cute.make_fragment_like(tv_layout_mma_B, dtype=self.k_dtype)
        tQKsKt = kt_thr_copy.partition_S(sK)
        tQKrKt_cv = kt_thr_copy.retile(tQKrKt)
        cute.copy(kt_tiled_copy, tQKsKt, tQKrKt_cv)

        # compute k * gktn
        kt_val = tQKrKt.load().to(cutlass.Float32)
        kt_val = kt_val * g_val
        kt_val = kt_val.to(self.io_dtype)
        tQKrKt.store(kt_val)

        return tQKrKt

    @cute.jit
    def r2s_subchunk_acc(
        self,
        r: cutlass.Constexpr[int],
        c: cutlass.Constexpr[int],
        src: cute.Tensor,
        dst: cute.Tensor,
        tiled_copy: cute.atom.TiledCopy,
        thr_copy: cute.atom.ThrCopy,
        out_dtype,
    ):
        dst_r_c = dst[None, None, r, c]
        tSdst_r_c = thr_copy.partition_D(dst_r_c)
        tRsrc = thr_copy.retile(src)
        tRsrc_cvt = cute.make_fragment_like(tRsrc, dtype=out_dtype)
        src_val = tRsrc.load()
        src_val_out = src_val.to(out_dtype)
        tRsrc_cvt.store(src_val_out)
        cute.copy(tiled_copy, tRsrc_cvt, tSdst_r_c)

    @cute.jit
    def apply_qk_kk_mask(
        self,
        index_qk: cute.Tensor,
        qk: cute.Tensor,
        kk: cute.Tensor,
        valid_len_chunk: cutlass.Int32,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
        # FIXME: boundary mask for the final block
        is_final_block: cutlass.Constexpr[bool] = False,
    ):
        for i in cutlass.range_constexpr(cute.size(index_qk)):
            index_q, index_k = index_transform(*index_qk[i])
            # triangular for qk & kk
            if index_q < index_k:
                qk[i] = cutlass.BFloat16(0.0)
                kk[i] = cutlass.Float16(0.0)
            # boundary mask for non-aligned chunks (only needed for partial chunks)
            if valid_len_chunk < Constant.C:
                if index_q >= valid_len_chunk:
                    qk[i] = cutlass.BFloat16(0.0)
                    kk[i] = cutlass.Float16(0.0)
                if index_k >= valid_len_chunk:
                    qk[i] = cutlass.BFloat16(0.0)
                    kk[i] = cutlass.Float16(0.0)
            # fill 1.0 for kk diagonal
            if index_q == index_k:
                kk[i] = cutlass.Float16(1.0)

    # ===========

    @cute.jit
    def make_s2r_partitions_v(
        self,
        local_tidx: cutlass.Int32,
        smem_x: cute.Tensor,
        shape_x: cute.Tile,
    ):
        # Assume X is row-major
        dtype = smem_x.element_type
        copy_atom_s2r_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            # NOTE: choose 8x2B
            # num_bits_per_copy=dtype.width * 8,
            num_bits_per_copy=dtype.width * 1,
        )
        num_elements_per_thread = Constant.C
        num_threads_per_row = shape_x[1] // num_elements_per_thread
        # NOTE: Assume 128 cuda core threads
        num_threads_per_col = 128 // num_threads_per_row
        thread_layout = cute.make_layout(
            (num_threads_per_col, num_threads_per_row),
            stride=(num_threads_per_row, 1),
        )
        val_layout = cute.make_layout((1, num_elements_per_thread))
        tiled_s2r_x = cute.make_tiled_copy_tv(
            copy_atom_s2r_x,
            thread_layout,
            val_layout,
        )
        thr_s2r_x = tiled_s2r_x.get_slice(local_tidx)

        # Partition shared tensor for smem load Bt
        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
        tXsX_s2r = thr_s2r_x.partition_S(smem_x)

        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tXrX_s2r = cute.make_rmem_tensor(
            cute.slice_(tXsX_s2r.shape, (None, None, None, 0)),
            dtype,
        )
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"v s2r tiled: {tiled_s2r_x}")
        return tiled_s2r_x, thr_s2r_x, tXsX_s2r, tXrX_s2r

    @cute.jit
    def make_s2r_partitions_prologue(
        self,
        local_tidx: cutlass.Int32,
        smem_x: cute.Tensor,
        shape_x: cute.Tile,
    ):
        # Assume X is row-major
        dtype = smem_x.element_type
        copy_atom_s2r_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            # NOTE: make wanna make sure every thread loads one element
            num_bits_per_copy=dtype.width,
        )
        # num_elements_per_thread = 16 // dtype.width
        num_elements_per_thread = 1
        num_threads_per_row = shape_x[1] // num_elements_per_thread
        # NOTE: Assume 128 cuda core threads
        num_threads_per_col = 128 // num_threads_per_row
        thread_layout = cute.make_layout(
            (num_threads_per_col, num_threads_per_row),
            stride=(num_threads_per_row, 1),
        )
        val_layout = cute.make_layout((1, num_elements_per_thread))
        tiled_s2r_x = cute.make_tiled_copy_tv(
            copy_atom_s2r_x,
            thread_layout,
            val_layout,
        )
        thr_s2r_x = tiled_s2r_x.get_slice(local_tidx)

        # Partition shared tensor for smem load Bt
        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
        tXsX_s2r = thr_s2r_x.partition_S(smem_x)

        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tXrX_s2r = cute.make_rmem_tensor(
            cute.slice_(tXsX_s2r.shape, (None, None, None, 0)),
            dtype,
        )
        return tiled_s2r_x, thr_s2r_x, tXsX_s2r, tXrX_s2r


def make_thread_cooperative_group(size: int):
    """Helper to create thread cooperative groups for pipeline synchronization."""
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


def main():
    """
    Example usage of LinearAttentionChunkwise with CuTe DSL
    """
    parser = argparse.ArgumentParser(description="Chunkwise Linear Attention with Headwise Decay")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size")
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
    print(f"  IO dtype: {args.io_dtype}")
    print(f"  Accumulation dtype: {args.acc_dtype}")
    print(f"  Warmup iterations: {args.warmup_iterations}")
    print(f"  Benchmark iterations: {args.iterations}")

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create inputs
    B, S, H, D = args.batch_size, args.seq_len, args.num_heads, args.head_dim

    # Input tensors in format [B, S, H, D]
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    G = torch.nn.functional.logsigmoid(
        torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    )  # Gate tensor for KDA (logsigmoid initialized)
    # Beta tensor for KDA: shape (B, S, H)
    # Each position in the sequence can have its own beta value per batch and head
    beta_tensor = torch.randn(B, S, H, device="cuda", dtype=torch.float32).sigmoid()

    # Apply cumsum within each chunk (chunk_size=64) and multiply by 1/ln2 for G before passing to kernel
    chunk_size = 64
    num_chunks = S // chunk_size
    G = G.float().view(B, num_chunks, chunk_size, H, D).cumsum(dim=2).view(B, S, H, D) * 1.4426950216

    # QK, L2 Norm
    Q, Q_rstd = l2norm_fwd(Q)
    K, K_rstd = l2norm_fwd(K)

    # Convert to dlpack for CuTe
    q_cute = from_dlpack(Q)
    k_cute = from_dlpack(K)
    v_cute = from_dlpack(V)
    g_cute = from_dlpack(G)
    beta_cute = from_dlpack(beta_tensor)

    o_cute = from_dlpack(torch.empty_like(Q))

    # Create kernel instance
    attn_kernel = KDAChunkwise(
        chunk_size=args.chunk_size,
        qk_acc_dtype=args.acc_dtype,
        kv_acc_dtype=args.acc_dtype,
        io_dtype=args.io_dtype,
        scale=args.head_dim**-0.5,
    )

    # Get default stream
    stream = cutlass_torch.default_stream()

    start_time = time.time()
    compiled = cute.compile(
        attn_kernel,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        g_cute.iterator,
        o_cute.iterator,
        beta_cute.iterator,
        (B, S, H, D),
        stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print(f"B, S, H, D: {(B, S, H, D)}")

    # Warmup
    for _ in range(args.warmup_iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            (B, S, H, D),
            stream,
        )

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(args.iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            (B, S, H, D),
            stream,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\nExecution time: {elapsed * 1000 / args.iterations:.2f} ms (average over {args.iterations} iterations)")
    print(f"Throughput: {(B * S * H * D * args.iterations) / (elapsed * 1e9):.2f} GB/s")
    print("\nPASS")


if __name__ == "__main__":
    main()
