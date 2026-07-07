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
Linear Attention with Headwise Decay using CuTe DSL

This module implements chunkwise linear attention with per-head decay factors
for the NVIDIA Blackwell SM100 architecture using CUTE DSL.

The implementation supports:
- Chunkwise computation for improved GPU utilization
- Per-head decay coefficients for flexible dependency modeling
- Efficient state accumulation across chunks
- Input/output format: [Batch, Sequence, Heads, Dim]

To run this example:

.. code-block:: bash

    python examples/blackwell/linear_attn.py \\
      --batch_size 4 --seq_len 1024 --num_heads 8 --head_dim 64 \\
      --chunk_size 64 --decay 0.95

Mathematical formulation:

O_h(i) = (Q_i^T * (global_num + block_num)) / (Q_i^T * (global_denom + block_denom) + ε)

where:
- block_num(t) = λ_h * block_num(t-1) + K_t V_t^T
- block_denom(t) = λ_h * block_denom(t-1) + K_t
- global_state = λ_h^L * global_state + block_state (inter-chunk accumulation)
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

from cula.utils import assert_blackwell

PRINT_DEBUG = False


class MaskEnum:
    """Enumeration for different mask types."""

    NONE = 0
    PADDING = 1
    CAUSAL = 2


class LinearAttentionChunkwise:
    """
    Chunkwise Linear Attention with Per-Head Decay using CuTe DSL

    Implements the Lightning Attention algorithm with headwise decay factors.
    Decomposes attention into intra-chunk (local) and inter-chunk (global) components.

    Args:
        chunk_size: Size of each attention chunk (default: 64)
        qk_acc_dtype: Accumulator data type for QK computation (default: Float32)
        kv_acc_dtype: Accumulator data type for PV computation (default: Float32)
        io_dtype: Input/output data type (default: Float16)
    """

    def __init__(
        self,
        chunk_size: int = 64,
        qk_acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        kv_acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        assert_blackwell()
        self.chunk_size = chunk_size
        self.qk_acc_dtype = qk_acc_dtype
        self.kv_acc_dtype = kv_acc_dtype
        self.pv_acc_dtype = kv_acc_dtype
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype

        # Warp specialization
        self.num_load_warps = 1
        self.num_compute_warps = 4
        self.num_correction_warps = 4
        self.threads_per_warp = 32

        # MMA tile shapes
        # C: 64, choose chunk size as 64 for enough spaces to do double buffering
        # Q: (64, 128)
        # K: (64, 128)
        # V: (64, 128)
        # TODO: READ from input
        C, D = (64, 128)
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
        # self.empty_warp_id = 7

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cuda_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
            )
        )

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
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
    ):
        """Compute TMEM offsets for various tensors used in the kernel."""
        SM100_TMEM_CAPACITY_COLS = 512

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
        o_iter: cute.Pointer,
        decay: cute.Pointer,
        problem_size: tuple[Int32, Int32, Int32, Int32],  # (B, S, H, D)
        # problem_size: tuple[int, int, int, int],  # (B, S, H, D)
        stream: cuda.CUstream,
    ):
        """
        Execute the Chunkwise Linear Attention operation on the provided tensors.

        Args:
            q_iter: Query tensor
            k_iter: Key tensor
            v_iter: Value tensor
            o_iter: Output tensor
            decay_iter: Per-head decay coefficients pointer [H]
            problem_size: (B, S, H, D) problem dimensions
            stream: CUDA stream
        """
        B, S, H, D = problem_size

        # Setup attributes
        self._setup_attributes()

        # TODO: try two-cta
        self.cta_group = tcgen05.CtaGroup.ONE

        # It's ok since torch tensor is row major, hence we've layout=(B,S,H,D):(DHS, DH, D, 1).
        # Below are just permutation tricks to ease the later processing.
        q_layout = cute.make_layout(
            (S, D, (H, B)),
            stride=(D * H, 1, (D, D * H * S)),
        )
        q = cute.make_tensor(q_iter, q_layout)
        # (S, D, (H,B))
        k_layout = cute.make_layout(
            (S, D, (H, B)),
            stride=(D * H, 1, (D, D * H * S)),
        )
        k = cute.make_tensor(k_iter, k_layout)
        kt_layout = cute.make_layout(
            (D, S, (H, B)),
            stride=(1, D * H, (D, D * H * S)),
        )
        kt = cute.make_tensor(k_iter, kt_layout)
        # v
        v_layout = cute.make_layout(
            (D, S, (H, B)),
            stride=(1, D * H, (D, D * H * S)),
        )
        # v_layout = cute.make_layout(
        #     (S, D, (H,B)),
        #     stride=(D*H, 1, (D, D*H*S)),
        # )
        v = cute.make_tensor(v_iter, v_layout)

        o_layout = cute.make_layout(
            (D, S, (H, B)),
            stride=(1, D * H, (D, D * H * S)),
        )
        o = cute.make_tensor(o_iter, o_layout)

        # TODO: output final state

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
            self.qk_acc_dtype,
            self.cta_group,
            self.qk_mma_tiler[:2],
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
        # self.tma_copy_v_bytes = v_copy_size
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
            # DEBUG: TODO: drop me
            sKV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(state_tmem_layout_staged)],  # type: ignore
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        print(f"size of storage: {SharedStorage.__sizeof__()}")

        self.grid = self._compute_grid(
            # (D, S, (H, B))
            o_shape=cute.shape(o),
            chunk_size=self.chunk_size,
        )
        print(f"grid: {self.grid}")

        self.kernel(
            qk_tiled_mma,
            kv_tiled_mma,
            vp_tiled_mma,
            sq_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_kt,
            tma_tensor_kt,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            decay,
            q_smem_layout_staged,
            k_smem_layout_staged,
            kv_k_smem_layout_staged,
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
        tma_atom_kt: cute.CopyAtom,
        tma_tensor_kt: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        decay: cute.Pointer,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        kv_k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        state_tmem_layout_staged: cute.ComposedLayout,
        problem_size: tuple[Int32, Int32, Int32, Int32],  # (B, S, H, D)
    ):
        """Kernel for linear attention.

        Args:
            qk_tiled_mma (cute.TiledMma): qk tiled mma
            kv_tiled_mma (cute.TiledMma): kv tiled mma
            vp_tiled_mma (cute.TiledMma): pv tiled mma
            tma_atom_q (cute.CopyAtom): _description_
            tma_tensor_q (cute.Tensor): _description_
            tma_atom_k (cute.CopyAtom): _description_
            tma_tensor_k (cute.Tensor): _description_
            tma_atom_v (cute.CopyAtom): _description_
            mV_vdl (cute.Tensor): _description_
            o (cute.Tensor): _description_
            decay (cute.Pointer): _description_
            q_smem_layout_staged (cute.ComposedLayout): _description_
            k_smem_layout_staged (cute.ComposedLayout): _description_
            v_smem_layout_staged (cute.ComposedLayout): _description_
            o_smem_layout_staged (cute.ComposedLayout): _description_
            p_smem_layout_staged (cute.ComposedLayout): _description_
            chunk_size (int): _description_
        """
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
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                len([self.mma_warp_id]),
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.kv_mbar_ptr.data_ptr(),
        ).make_participants()
        # Notify mma warp that 16bit state is ready for mma as operand A
        kv16_producer, kv16_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                len(self.cuda_warp_ids),
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len([self.mma_warp_id])),
            barrier_storage=storage.kv16_mbar_ptr.data_ptr(),
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsync.create(
            num_stages=self.acc_stage,  # TODO: check p stages
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
        # TODO: Avoid duplicated loading of k even through L2 cache might make it fine.
        sK_kv = storage.sK.get_tensor(
            # kv_k_smem_layout_staged.outer, swizzle=kv_k_smem_layout_staged.inner
            # NOTE: same swizzle atom (k-major) as k_smem_layout_staged
            kv_k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
        )
        # (((64,2),16),1,4,2):(((1,4096),64),0,1024,8192)>
        sV = storage.sV.get_tensor(v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sP = storage.sP.get_tensor(p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE_O)
        sO = storage.sO.get_tensor(o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)

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

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"sQ: {cute.pretty_str(sQ)}")
            print(f"sK: {cute.pretty_str(sK)}")
            print(f"sK_kv: {cute.pretty_str(sK_kv)}")
            print(f"sV: {cute.pretty_str(sV)}")
            print(f"sO: {cute.pretty_str(sO)}")
            print(f"sP: {cute.pretty_str(sP)}")
            print(f"sQK: {cute.pretty_str(sQK)}")

        self.num_regs_other = 24
        self.num_regs_epilogue_warps = 24
        self.num_regs_mma = 24
        self.num_regs_cuda = 160

        (_, hidx, bidx) = cute.arch.block_idx()
        B, S, H, D = problem_size
        C = self.chunk_size

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
            sK,
            tmem_ptr_base + self.tmem_qk_cols_offset,
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
            sK_kv,
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

        # ///////////////////////////////////////////////////////////////////////////////
        # LOAD WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), TILES_N, TILES_K)
            tQsQ, tQgQ = self.tma_partition_for_mma_operand(
                tma_atom_q,
                tma_tensor_q,
                sQ,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="A",
                debug_name="Q",
            )

            tKsK, tKgK = self.tma_partition_for_mma_operand(
                tma_atom_k,
                tma_tensor_k,
                sK,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="B",
                debug_name="K",
            )

            tVsV, tVgV = self.tma_partition_for_mma_operand(
                tma_atom_v,
                tma_tensor_v,
                sV,
                self.vp_mma_tiler,
                vp_tiled_mma,
                operand_mode="A",
                debug_name="V",
            )

            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tKsK={tKsK}")
                print(f"tKgK={tKgK}")
                print(f"tVsV={tVsV}")
                print(f"tVgV={tVgV}")

            for chunk_start in cutlass.range(0, S, C, unroll=0):
                # Chunk iterate over TILES_M, TILES_K is 1 in our case since max D is 128
                idx = chunk_start // C
                should_debug = PRINT_DEBUG and tidx == warp_idx * 32 and hidx == 0 and bidx == 0

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
                cute.copy(
                    atom=tma_atom_v,
                    src=tVgV[None, 0, idx],
                    dst=tVsV[None, v_handle.index],
                    tma_bar_ptr=v_handle.barrier,
                )

                if should_debug:
                    # cute.printf("tKgK={}", tKgK)
                    # cute.printf("tKgKT={}", tKgKT)
                    # cute.printf("tVgV={}", tVgV)
                    # cute.printf("tVsV={}", tVsV)
                    # cute.printf("tma produce v, index={}", v_handle.index)
                    # # cute.print_tensor(tVsV[None, v_handle.index])
                    # cute.printf("Full SV")
                    # cute.print_tensor(sV_flat)
                    # cute.printf("Full SK")
                    # cute.print_tensor(sK_flat)
                    pass

        # ///////////////////////////////////////////////////////////////////////////////
        # COMPUTE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)

            should_debug = PRINT_DEBUG and tidx == warp_idx * 32 and hidx == 0 and bidx == 0

            for chunk_start in cutlass.range(0, S, C, unroll=0):
                # Process chunk from chunk_start to chunk_start + chunk_size
                idx = chunk_start // C

                # Wait for Qi.
                q_handle = load_q_consumer.wait_and_advance()

                if idx != 0:
                    kv16_handle = kv16_consumer.wait_and_advance()
                    o_inter_handle = o_inter_producer.acquire_and_advance()

                    # TODO: support initial state
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
                # End of GEMM (Qi, Ki) -> S0i

                # Wait for PV, produce ointra
                v_handle = load_v_consumer.wait_and_advance()
                p_handle = p_consumer.wait_and_advance()
                o_intra_handle = o_intra_producer.acquire_and_advance()
                # both v and p are in smem
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

                ####################################################
                kv_handle = kv_producer.acquire_and_advance()
                # NOTE: Always ACC to avoid adding in cuda core.
                kv_tiled_mma = self.exec_mma(
                    tiled_mma=kv_tiled_mma,
                    tCtAcc=tCtAccKV,
                    tCrA=tCrV,
                    # tCrB=tCrK,
                    tCrB=tCrK_kv,
                    a_stage_idx=v_handle.index,
                    b_stage_idx=k_handle.index,
                    acc_stage_idx=0,
                    always_acc=True if idx != 0 else False,  # noqa: SIM210 -- Cute DSL: const_expr requires explicit True/False form # always accumulate states
                )
                # Release K V here
                k_handle.release()
                v_handle.release()
                kv_handle.commit()

        # ///////////////////////////////////////////////////////////////////////////////
        # CUDA CORE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

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

            ### ############################################
            ### # DEBUG BEGIN
            ### tAccSQ_epi = cute.flat_divide(
            ###     tCtAccSQ[(None, None), 0, 0, None],
            ###     self.epi_tile,
            ### )
            ### fake_smem_sq_layout = sm100_utils.make_smem_layout_epi(
            ###     self.io_dtype,
            ###     utils.LayoutEnum.ROW_MAJOR,
            ###     self.sq_mma_tiler[:2],
            ###     1,
            ### )
            ### fake_smem_sq = cute.make_tensor(
            ###     cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
            ###     fake_smem_sq_layout,
            ### )
            ### (
            ###     tmem_load_sq,
            ###     tmem_load_tAccSQ,
            ###     tmem_load_rAccSQ,
            ### ) = self.epilog_tmem_load_and_partition_acc(local_tidx, tAccSQ_epi, fake_smem_sq)

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
            # tCcM = qk_thr_mma.partition_C(cM)
            tTR_cS = thr_t2r_S.partition_D(cM)
            # print(f"thr_t2r_S: {thr_t2r_S}")
            # print(f"tCcM: {tCcM}")
            # print(f"tTR_cS: {tTR_cS}")

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
                # tiled_mma=vp_tiled_mma,
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

            should_debug = PRINT_DEBUG and tidx == self.cuda_warp_ids[0] * 32 and hidx == 0 and bidx == 0

            for chunk_start in cutlass.range(0, S, C, unroll=0):
                idx = chunk_start // C

                # Convert KV to KV16
                if idx != 0:
                    kv_handle = kv_consumer.wait_and_advance()
                    tTR_tKVi = tTR_tKV[(None, None, None, kv_handle.index)]  # kv stage == 1
                    cute.copy(tiled_copy_t2r_kv, tTR_tKVi, tTR_rKV)
                    cute.arch.fence_view_async_tmem_load()
                    kv_handle.release()

                    # if tidx == 0:
                    #     # cute.printf("idx={}, tidx={}, LOAD tKV", idx, tidx)
                    #     # tidx_rKV = cute.make_tensor(tTR_rKV.iterator, cute.make_layout(D))
                    #     # cute.print_tensor(tidx_rKV)
                    #     pass

                    kv16_handle = kv16_producer.acquire_and_advance()
                    #####################################################################3
                    tmem_store_rAccKVAsBF16.store(tTR_rKV.load().to(self.io_dtype))
                    tmem_store_tAccKVi = tmem_store_tAccKV[None, None, None, None, kv16_handle.index]
                    cute.copy(tmem_store_kv, tmem_store_rAccKV, tmem_store_tAccKVi)
                    cute.arch.fence_view_async_tmem_store()
                    # if tidx == 0:
                    #     cute.printf("idx={}, tidx={} STORE tKV16", idx, tidx)
                    #     tidx_rKV = cute.make_tensor(tmem_store_rAccKVAsBF16.iterator, cute.make_layout(D))
                    #     cute.print_tensor(tidx_rKV)
                    #     pass
                    #####################################################################3
                    kv16_handle.commit()

                # Wait for S = QK^T
                s0_handle = mma_s0_consumer.wait_and_advance()

                # (MMA, MMA_M, MMA_N, ACC_STAGE)
                tTR_tSi = tTR_tS[None, None, None, s0_handle.index]
                # Load S from TMEM to RMEM
                cute.copy(tiled_t2r_S, tTR_tSi, tTR_rS)
                cute.arch.fence_view_async_tmem_load()

                # Apply mask and convert to BF16
                # TODO: check causal correctness
                self.apply_mask(tTR_rS, tTR_cS, tTR_rP, debug=False)

                # Write P to SMEM
                p_handle = p_producer.acquire_and_advance()

                # Store P from RMEM to SMEM
                tRS_sPi = tRS_sP[(None, None, None, p_handle.index)]
                cute.copy(tiled_r2s_P, tRS_rP, tRS_sPi)
                # Fence
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                s0_handle.release()
                p_handle.commit()

                # Wait for O_INTRA
                o_intra_handle = o_intra_consumer.wait_and_advance()

                # Load O_INTRA from TMEM to RMEM
                tTR_tAcc_pv_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, o_intra_handle.index)]
                cute.copy(tiled_copy_t2r_pv, tTR_tAcc_pv_i, tTR_rAcc_pv)
                cute.arch.fence_view_async_tmem_load()
                o_intra_handle.release()

                # Wait for O_INTER
                if idx != 0:
                    o_inter_handle = o_inter_consumer.wait_and_advance()
                    tTR_tAcc_sq_i = tTR_tAcc_base_sq[(None, None, None, 0, 0, o_inter_handle.index)]
                    # Load O_INTER from TMEM to RMEM
                    cute.copy(tiled_copy_t2r_sq, tTR_tAcc_sq_i, tTR_rAcc_sq)
                    cute.arch.fence_view_async_tmem_load()
                    o_inter_handle.release()

                # Perform addition and store to gmem
                acc_vec = tTR_rAcc_pv.load()
                if idx != 0:
                    acc_vec_inter = tTR_rAcc_sq.load()
                    acc_vec = acc_vec + acc_vec_inter
                tTR_rO.store(acc_vec.to(self.io_dtype))

                # Store output to smem
                smem_o_handle = smem_o_producer.acquire_and_advance()
                cute.copy(tiled_copy_r2s_o, tRS_rO, tRS_sO[(None, None, None, smem_o_handle.index)])
                # Fence and barrier to make sure shared memory store is visible to TMA store
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                smem_o_handle.commit()

        elif warp_idx == self.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_epilogue_warps)

            should_debug = PRINT_DEBUG and tidx == warp_idx * 32 and hidx == 0 and bidx == 0
            # TMA STORE
            # O: (D, S), column major
            # (MMA_M, MMA_N, TILES_M, TILES_N, (H, B))
            gO_pre_partition = cute.flat_divide(tma_tensor_o, cute.select(self.vp_mma_tiler, mode=[0, 1]))

            # (MMA_M, MMA_N, TILES_M, TILES_N)
            gO_pre_partition = gO_pre_partition[None, None, None, None, (hidx, bidx)]

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

            # TMA LOAD
            for chunk_start in cutlass.range(0, S, C, unroll=0):
                idx = chunk_start // C

                smem_o_handle = smem_o_consumer.wait_and_advance()
                # TMA STORE O: SMEM -> GMEM
                cute.copy(tma_atom_o, bSG_sO[None, smem_o_handle.index], bSG_gO[(None, 0, 0, 0, idx)])
                # Ensure smem_o has been released.
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                smem_o_handle.release()

        # Release tensor memory allocation lock
        tmem.relinquish_alloc_permit()
        # Sync before deallocating tmem
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        # Dealloc tmem buffer
        tmem.free(tmem_ptr_base)

        return

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
            # TODO: check unpack
            # tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
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
                p[i] = acc_qk[i].to(self.q_dtype)

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
    ):
        _, hidx, bidx = cute.arch.block_idx()
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
    def tma_partition_for_mma_operand(
        self,
        tma_atom_x,
        tma_tensor_x,
        smem_x,
        tile_shape,
        tiled_mma,
        operand_mode,
        debug_name=None,
    ):
        tCgX = self.local_tile_partition_for_mma_operand(
            tensor_x=tma_tensor_x,
            tile_shape=tile_shape,
            tiled_mma=tiled_mma,
            operand_mode=operand_mode,
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

    # Convert to dlpack for CuTe
    q_cute = from_dlpack(Q)
    k_cute = from_dlpack(K)
    v_cute = from_dlpack(V)
    decay_cute = from_dlpack(decay)

    o_cute = from_dlpack(torch.zeros_like(Q))

    # Create kernel instance
    attn_kernel = LinearAttentionChunkwise(
        chunk_size=args.chunk_size,
        qk_acc_dtype=args.acc_dtype,
        kv_acc_dtype=args.acc_dtype,
        io_dtype=args.io_dtype,
    )

    # Get default stream
    stream = cutlass_torch.default_stream()

    start_time = time.time()
    compiled = cute.compile(
        attn_kernel,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        o_cute.iterator,
        decay_cute.iterator,
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
            o_cute.iterator,
            decay_cute.iterator,
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
            o_cute.iterator,
            decay_cute.iterator,
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
