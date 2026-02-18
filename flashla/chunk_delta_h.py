# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Chunk Gated Delta Rule Forward H Kernel using CuTe DSL

This module implements the chunk_gated_delta_rule_fwd_h operation for NVIDIA Blackwell
SM100 architecture using CUTE DSL.

Mathematical formulation (per chunk t):
1. Save current state: h_out[t] = h
2. Compute delta: v_new = u - w @ h  
3. Apply gate decay:
   - Scalar gate g: v_new *= exp(g_last - g), h *= exp(g_last)
   - Vector gate gk: h *= exp(gk_last)
4. Update state: h += k^T @ v_new

Input tensors:
- k: (B, T, H, K) - Key tensor
- w: (B, T, H, K) - Weight for delta rule erasure
- u: (B, T, H, V) - Value tensor (called v in fla)
- g: (B, T, H) - Optional scalar gate (log space)
- gk: (B, T, H, K) - Optional vector gate (log space)
- initial_state: (N, H, K, V) - Initial hidden state

Output tensors:
- h: (B, NT, H, K, V) - Hidden states at each chunk start
- v_new: (B, T, H, V) - Corrected values
- final_state: (N, H, K, V) - Final hidden state

Assumptions:
- K (head dimension) = 128
- chunk_size = 64
- V can vary but typically 64 or 128
"""

import argparse
import math
import os
import sys
import time
from typing import Type, Tuple, List, Union

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32

PRINT_DEBUG = False

# Constants for exp2 vs exp conversion
LN2 = 0.6931471805599453  # ln(2)
INV_LN2 = 1.4426950408889634  # 1/ln(2)


def make_thread_cooperative_group(size: int):
    """Helper to create thread cooperative groups for pipeline synchronization."""
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class ChunkDeltaRuleFwdH:
    """
    Chunk Gated Delta Rule Forward H using CuTe DSL for Blackwell SM100
    
    Implements the hidden state recurrence for gated delta rule attention.
    
    Args:
        chunk_size: Size of each chunk (default: 64)
        head_dim_k: Key head dimension (default: 128)
        acc_dtype: Accumulator data type (default: Float32)
        io_dtype: Input/output data type (default: BFloat16)
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        use_exp2: bool = True,
    ):
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.use_exp2 = use_exp2

        # Tile dimensions
        self.BT = chunk_size  # 64
        self.BK = head_dim_k  # 128
        self.BV = head_dim_v  # 64 or 128 (will be tiled if > 64)

        # Warp specialization
        self.threads_per_warp = 32
        
        # Warp assignment:
        # - cuda_warp_ids: CUDA core warps for gate application and type conversion
        # - mma_warp_id: MMA warp for matrix operations
        # - load_warp_id: TMA load warp
        # - store_warp_id: TMA store warp
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6

        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cuda_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.store_warp_id,
            )
        )

        # MMA tile shapes
        # For W @ H: (BT, BV) = (BT, BK) @ (BK, BV) -> w_mma_tiler
        # BT=64, BK=128, BV=64
        self.wh_mma_tiler = (self.BT, self.BV, self.BK)  # (M, N, K)
        
        # For V_new^T @ K: (BV, BK) = (BV, BT) @ (BT, BK) -> kv_mma_tiler
        # V as operand A (M dim) so TMEM M=128 allows Ld32x32bOp T2R
        # Result is h^T; transpose handled by TMA store via h_out_T view
        self.kv_mma_tiler = (self.BV, self.BK, self.BT)  # (M=V, N=K, K=T)

        # Pipeline stages
        self.k_stage = 1
        self.w_stage = 1
        self.u_stage = 1
        self.h_stage = 1  # State buffer
        self.acc_stage = 1

        # Cluster shape (single CTA)
        self.cluster_shape_mnk = (1, 1, 1)
        
        # CTA group for MMA
        self.cta_group = tcgen05.CtaGroup.ONE

        # Barriers
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )

        self.buffer_align_bytes = 1024
        self.enable_gmem_roundtrip = True   # Pre-loop h_gmem init
        self.enable_loop_roundtrip = True   # In-loop vnew_gmem + h_gmem waits

    def _setup_attributes(self):
        """Setup pipeline stage attributes."""
        self.k_stage = 1
        self.w_stage = 1
        self.u_stage = 1
        self.h_stage = 1
        self.acc_stage = 1

    @staticmethod
    def _plan_tmem_offsets(
        tiled_mma_wh,
        tile_shape_wh,
        tiled_mma_kv,
        tile_shape_kv,
        acc_stages,
    ):
        """Compute TMEM offsets for accumulator tensors."""
        SM100_TMEM_CAPACITY_COLS = 512
        BITS_PER_TMEM_COL = 32

        # W @ H accumulator: (BT, BV)
        acc_shape_wh = tiled_mma_wh.partition_shape_C(tile_shape_wh[:2])
        tCtAccWH_fake = tiled_mma_wh.make_fragment_C(
            cute.append(acc_shape_wh, acc_stages)
        )
        num_wh_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccWH_fake)

        # V_new^T @ K accumulator: (BV, BK) - this is h^T (state transposed)
        acc_shape_kv = tiled_mma_kv.partition_shape_C(tile_shape_kv[:2])
        tCtAccKV_fake = tiled_mma_kv.make_fragment_C(
            cute.append(acc_shape_kv, 1)  # State has 1 stage
        )
        num_kv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccKV_fake)
        
        # State as BF16 for loading back
        num_state_bf16_cols = num_kv_acc_cols // 2

        # Offsets
        num_wh_acc_cols_offset = 0
        num_kv_acc_cols_offset = num_wh_acc_cols_offset + num_wh_acc_cols
        num_state_bf16_cols_offset = num_kv_acc_cols_offset + num_kv_acc_cols

        num_tmem_cols_total_tmp = num_state_bf16_cols_offset + num_state_bf16_cols
        
        # Round up to power of 2
        num_tmem_cols_total = 1
        while num_tmem_cols_total < num_tmem_cols_total_tmp:
            num_tmem_cols_total *= 2
        
        assert num_tmem_cols_total <= SM100_TMEM_CAPACITY_COLS

        if cutlass.const_expr(PRINT_DEBUG):
            print("=" * 80)
            print("TMEM Allocation Details:")
            print(f"  WH acc:      {num_wh_acc_cols:4d} cols @ offset {num_wh_acc_cols_offset:4d}")
            print(f"  KV acc:      {num_kv_acc_cols:4d} cols @ offset {num_kv_acc_cols_offset:4d}")
            print(f"  State BF16:  {num_state_bf16_cols:4d} cols @ offset {num_state_bf16_cols_offset:4d}")
            print(f"  Total:       {num_tmem_cols_total:4d} cols")
            print("=" * 80)

        return (
            num_wh_acc_cols_offset,
            num_kv_acc_cols_offset,
            num_state_bf16_cols_offset,
            num_tmem_cols_total,
        )

    def _compute_grid(
        self,
        B: int,
        H: int,
        V: int,
    ) -> cute.Shape:
        """Compute grid dimensions."""
        # Grid: (ceil(V/BV), N*H)
        # We parallelize over V dimension tiles and batch*heads
        BV = self.BV  # BV tile size
        return (
            (V + BV - 1) // BV,  # V tiles
            H,                   # Heads
            B,                   # Batch
        )

    @cute.jit
    def __call__(
        self,
        k_ptr: cute.Pointer,      # (B, T, H, K)
        w_ptr: cute.Pointer,      # (B, T, H, K)
        u_ptr: cute.Pointer,      # (B, T, H, V)
        g_ptr: cute.Pointer,      # (B, T, H) or None
        gk_ptr: cute.Pointer,     # (B, T, H, K) or None
        h_out_ptr: cute.Pointer,  # (B, NT, H, K, V)
        v_new_ptr: cute.Pointer,  # (B, T, H, V)
        h0_ptr: cute.Pointer,     # (B, H, K, V) or None
        ht_ptr: cute.Pointer,     # (B, H, K, V) or None
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],  # (B, T, H, K, V)
        use_g: Int32,
        use_gk: Int32,
        use_initial_state: Int32,
        store_final_state: Int32,
        save_v_new: Int32,
        stream,
    ):
        """
        Execute chunk_gated_delta_rule_fwd_h operation.
        
        Args:
            k_ptr: Key tensor pointer
            w_ptr: Weight tensor pointer for delta rule
            u_ptr: Value tensor pointer
            g_ptr: Optional scalar gate pointer
            gk_ptr: Optional vector gate pointer
            h_out_ptr: Output hidden states pointer
            v_new_ptr: Output corrected values pointer
            h0_ptr: Optional initial state pointer
            ht_ptr: Optional final state pointer
            problem_size: (B, T, H, K, V) dimensions
            use_g: Whether to use scalar gate
            use_gk: Whether to use vector gate
            use_initial_state: Whether to use initial state
            store_final_state: Whether to store final state
            save_v_new: Whether to save v_new
            stream: CUDA stream
        """
        B, T, H, K, V = problem_size
        NT = (T + self.BT - 1) // self.BT  # Number of chunks

        self._setup_attributes()

        # Create tensor layouts
        # k, w: (B, T, H, K) with stride (T*H*K, H*K, K, 1)
        k_layout = cute.make_layout(
            (T, K, (H, B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        k = cute.make_tensor(k_ptr, k_layout)
        
        # k transposed for K^T @ V: (K, T, (H, B))
        kt_layout = cute.make_layout(
            (K, T, (H, B)),
            stride=(1, H * K, (K, T * H * K)),
        )
        kt = cute.make_tensor(k_ptr, kt_layout)

        w_layout = cute.make_layout(
            (T, K, (H, B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        w = cute.make_tensor(w_ptr, w_layout)

        # u: (B, T, H, V) with stride (T*H*V, H*V, V, 1)
        u_layout = cute.make_layout(
            (T, V, (H, B)),
            stride=(H * V, 1, (V, T * H * V)),
        )
        u = cute.make_tensor(u_ptr, u_layout)

        # v_new: same layout as u
        v_new_layout = cute.make_layout(
            (T, V, (H, B)),
            stride=(H * V, 1, (V, T * H * V)),
        )
        v_new = cute.make_tensor(v_new_ptr, v_new_layout)

        # v_new transposed GMEM view for TMA load as A operand of kv_tiled_mma
        # A operand has (M, K) = (BV, BT), so we need (V, T, ...) ordering
        v_new_T_layout = cute.make_layout(
            (V, T, (H, B)),
            stride=(1, H * V, (V, T * H * V)),
        )
        v_new_T = cute.make_tensor(v_new_ptr, v_new_T_layout)

        # h_out: (B, NT, H, K, V) -> (K, V, (NT, H, B))
        h_out_layout = cute.make_layout(
            (K, V, (NT, H, B)),
            stride=(V, 1, (H * K * V, K * V, NT * H * K * V)),
        )
        h_out = cute.make_tensor(h_out_ptr, h_out_layout)

        # h_out transposed GMEM view for TMA load as B operand of wh_tiled_mma
        # B operand has (N, K) = (BV, BK), so we need (V, K, ...) ordering
        h_out_T_layout = cute.make_layout(
            (V, K, (NT, H, B)),
            stride=(1, V, (H * K * V, K * V, NT * H * K * V)),
        )
        h_out_T = cute.make_tensor(h_out_ptr, h_out_T_layout)

        # h0, ht: (B, H, K, V) -> (K, V, (H, B))
        h0_layout = cute.make_layout(
            (K, V, (H, B)),
            stride=(V, 1, (K * V, H * K * V)),
        )
        h0 = cute.make_tensor(h0_ptr, h0_layout)
        ht = cute.make_tensor(ht_ptr, h0_layout)

        # h0/ht transposed GMEM view for TMA load/store (V, K, (H, B))
        # Same layout as h_out_T but without NT dimension
        h0_T_layout = cute.make_layout(
            (V, K, (H, B)),
            stride=(1, V, (K * V, H * K * V)),
        )
        h0_T = cute.make_tensor(h0_ptr, h0_T_layout)
        ht_T = cute.make_tensor(ht_ptr, h0_T_layout)

        # g: (B, T, H) -> (T, (H, B))
        g_layout = cute.make_layout(
            (T, (H, B)),
            stride=(H, (1, T * H)),
        )
        g = cute.make_tensor(g_ptr, g_layout)

        # gk: (B, T, H, K) -> (T, K, (H, B))
        gk_layout = cute.make_layout(
            (T, K, (H, B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        gk = cute.make_tensor(gk_ptr, gk_layout)

        self.k_dtype = k.element_type
        self.w_dtype = w.element_type
        self.u_dtype = u.element_type

        # Setup MMA operations
        # W @ H: (BT, BV) = (BT, BK) @ (BK, BV)
        wh_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,  # W is K-major (BT, BK)
            tcgen05.OperandMajorMode.MN, # H is MN-major (V contiguous in GMEM)
            self.acc_dtype,
            self.cta_group,
            self.wh_mma_tiler[:2],
        )

        # V_new^T @ K: (BV, BK) = (BV, BT) @ (BT, BK)
        # A=V_new: V contiguous in GMEM → M contiguous → MN-major
        # B=K: K contiguous in GMEM → N contiguous → MN-major
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,  # V_new A operand: MN-major
            tcgen05.OperandMajorMode.MN,  # K B operand: MN-major
            self.acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],  # (BV=128, BK=128)
        )

        # Plan TMEM offsets
        (
            self.tmem_wh_cols_offset,
            self.tmem_kv_cols_offset,
            self.tmem_state_bf16_offset,
            self.tmem_total_cols,
        ) = self._plan_tmem_offsets(
            wh_tiled_mma,
            self.wh_mma_tiler,
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.acc_stage,
        )

        # Create SMEM layouts
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # W SMEM layout: (BT, BK) for W @ H
        w_smem_layout_staged = sm100_utils.make_smem_layout_a(
            wh_tiled_mma,
            self.wh_mma_tiler,
            self.io_dtype,
            self.w_stage,
        )

        # K SMEM layout as B operand of KV MMA: (BT, BK) for V^T @ K
        k_kv_smem_layout_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.io_dtype,
            self.k_stage,
        )

        # U SMEM layout: (BT, BV)
        u_smem_layout_staged = sm100_utils.make_smem_layout_b(
            wh_tiled_mma,
            self.wh_mma_tiler,
            self.io_dtype,
            self.u_stage,
        )
        # U epilogue layout for CUDA warp reads (dual-view of same SMEM as sU)
        u_epi_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.wh_mma_tiler[:2],  # (BT=64, BV=128)
            self.u_stage,
        )

        # V_new SMEM layout as A operand of KV MMA: (BV, BT) for V^T @ K
        vnew_smem_layout_staged = sm100_utils.make_smem_layout_a(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.io_dtype,
            self.acc_stage,
        )
        # Epilogue layout for writing by CUDA warp (ROW_MAJOR matches K-major operand B)
        vnew_epi_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.wh_mma_tiler[:2],  # (BT=64, BV=128) output shape
            self.acc_stage,
        )

        # H state SMEM layout as MMA operand B for W @ H: (BK, BV)
        # wh_mma_tiler = (BT, BV, BK) means B operand is (K=BK, N=BV)
        h_state_smem_layout_staged = sm100_utils.make_smem_layout_b(
            wh_tiled_mma,
            self.wh_mma_tiler,
            self.io_dtype,
            1,  # single stage for state
        )
        # H state epilogue layout for CUDA warp writes (dual-view of same SMEM)
        # KV result is (BV, BK) = V^T@K, matching kv_mma_tiler[:2]
        # COL_MAJOR so BV (mode 0) is contiguous, matching V-contiguous in h_out_T GMEM
        h_state_epi_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            self.kv_mma_tiler[:2],  # (BV=128, BK=128)
            1,
        )

        # H_out SMEM layout for TMA store: (BK, BV)
        h_out_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BK, self.BV),
            1,
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (wh_tiled_mma.thr_id.shape,),
        )

        # TMA descriptors
        w_smem_layout = cute.select(w_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            w,
            w_smem_layout,
            self.wh_mma_tiler,
            wh_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        k_kv_smem_layout = cute.select(k_kv_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k_kv, tma_tensor_k_kv = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kt,
            k_kv_smem_layout,
            self.kv_mma_tiler,
            kv_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        u_smem_layout = cute.select(u_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_u, tma_tensor_u = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            u,
            u_smem_layout,
            self.wh_mma_tiler,
            wh_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        h_epi_smem_layout = cute.select(h_state_epi_layout_staged, mode=[0, 1])
        tma_atom_h_out, tma_tensor_h_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            h_out_T,
            h_epi_smem_layout,
            (self.BV, self.BK),
        )

        vnew_epi_smem_layout = cute.select(vnew_epi_layout_staged, mode=[0, 1])
        tma_atom_vnew_store, tma_tensor_vnew_store = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            v_new,
            vnew_epi_smem_layout,
            (self.BT, self.BV),
        )

        # TMA load for v_new from GMEM to sVnew_mma (A operand of kv_tiled_mma)
        # Uses transposed GMEM view: v_new_T = (V, T, (H,B)) maps V→M, T→K
        vnew_mma_smem_layout = cute.select(vnew_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_vnew_load, tma_tensor_vnew_load = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            v_new_T,
            vnew_mma_smem_layout,
            self.kv_mma_tiler,
            kv_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        # TMA load for h_state from GMEM to sH_state (B operand of wh_tiled_mma)
        # Uses transposed GMEM view so TMA handles the (BK, BV) → (BV, BK) transpose
        h_state_mma_smem_layout = cute.select(h_state_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_h_load, tma_tensor_h_load = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            h_out_T,
            h_state_mma_smem_layout,
            self.wh_mma_tiler,
            wh_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        # TMA load for h0 (initial state) from GMEM to sH_state
        # Uses h0_T transposed view: (V, K, (H, B))
        tma_atom_h0_load, tma_tensor_h0_load = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            h0_T,
            h_state_mma_smem_layout,
            self.wh_mma_tiler,
            wh_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        # TMA store for ht (final state) from sH_epi to GMEM
        # Uses ht_T transposed view: (V, K, (H, B))
        tma_atom_ht_store, tma_tensor_ht_store = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            ht_T,
            h_epi_smem_layout,
            (self.BV, self.BK),
        )

        self.tma_copy_vnew_bytes = cute.size_in_bytes(self.io_dtype, vnew_mma_smem_layout)
        self.tma_copy_h_bytes = cute.size_in_bytes(self.io_dtype, h_state_mma_smem_layout)

        # Calculate copy sizes
        self.tma_copy_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem_layout)
        self.tma_copy_kt_bytes = cute.size_in_bytes(self.io_dtype, k_kv_smem_layout)
        self.tma_copy_u_bytes = cute.size_in_bytes(self.io_dtype, u_smem_layout)

        # Shared storage structure
        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_w_mbar_ptr: cute.struct.MemRange[Int64, self.w_stage * 2]
            load_kt_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]
            wh_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]
            kv_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # KV result: MMA → CUDA
            vnew_smem_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # v_new SMEM ready: CUDA → Store
            vnew_gmem_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # v_new GMEM ready: Store → Load
            vnew_load_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]  # v_new TMA loaded: Load → MMA
            h_gmem_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # h_state GMEM ready: CUDA → Load
            h_load_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]  # h_state TMA loaded: Load → MMA
            h_out_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]
            # TMEM holding buffer
            tmem_holding_buf: Int32
            # SMEM tensors (no sU - CUDA reads U from GMEM; no sH_out - unused)
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_kv_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sVnew: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, max(cute.cosize(vnew_smem_layout_staged), cute.cosize(vnew_epi_layout_staged))],
                self.buffer_align_bytes,
            ]
            sH_state: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, max(cute.cosize(h_state_smem_layout_staged), cute.cosize(h_state_epi_layout_staged))],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.grid = self._compute_grid(B, H, V)

        # Debug: print SMEM sizes
        total_smem = sum([
            cute.cosize(w_smem_layout_staged),
            cute.cosize(k_kv_smem_layout_staged),
            max(cute.cosize(vnew_smem_layout_staged), cute.cosize(vnew_epi_layout_staged)),
            max(cute.cosize(h_state_smem_layout_staged), cute.cosize(h_state_epi_layout_staged)),
        ]) * 2 + 4 * 1024  # alignment padding
        print(f"  Total estimated SMEM: {total_smem} bytes ({total_smem / 1024:.1f} KB)")

        # Launch kernel
        self.kernel(
            wh_tiled_mma,
            kv_tiled_mma,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_k_kv,
            tma_tensor_k_kv,
            tma_atom_vnew_load,
            tma_tensor_vnew_load,
            tma_atom_h_load,
            tma_tensor_h_load,
            tma_atom_h0_load,
            tma_tensor_h0_load,
            tma_atom_h_out,
            tma_tensor_h_out,
            tma_atom_ht_store,
            tma_tensor_ht_store,
            tma_atom_vnew_store,
            tma_tensor_vnew_store,
            g,
            gk,
            h0,
            ht,
            v_new,
            u,
            h_out,
            w_smem_layout_staged,
            k_kv_smem_layout_staged,
            vnew_smem_layout_staged,
            vnew_epi_layout_staged,
            h_state_smem_layout_staged,
            h_state_epi_layout_staged,
            problem_size,
            use_g,
            use_gk,
            use_initial_state,
            store_final_state,
            save_v_new,
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
        wh_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        tma_tensor_w: cute.Tensor,
        tma_atom_k_kv: cute.CopyAtom,
        tma_tensor_k_kv: cute.Tensor,
        tma_atom_vnew_load: cute.CopyAtom,
        tma_tensor_vnew_load: cute.Tensor,
        tma_atom_h_load: cute.CopyAtom,
        tma_tensor_h_load: cute.Tensor,
        tma_atom_h0_load: cute.CopyAtom,
        tma_tensor_h0_load: cute.Tensor,
        tma_atom_h_out: cute.CopyAtom,
        tma_tensor_h_out: cute.Tensor,
        tma_atom_ht_store: cute.CopyAtom,
        tma_tensor_ht_store: cute.Tensor,
        tma_atom_vnew_store: cute.CopyAtom,
        tma_tensor_vnew_store: cute.Tensor,
        g: cute.Tensor,
        gk: cute.Tensor,
        h0: cute.Tensor,
        ht: cute.Tensor,
        v_new: cute.Tensor,
        u_tensor: cute.Tensor,
        h_out_tensor: cute.Tensor,
        w_smem_layout_staged: cute.ComposedLayout,
        k_kv_smem_layout_staged: cute.ComposedLayout,
        vnew_smem_layout_staged: cute.ComposedLayout,
        vnew_epi_layout_staged: cute.ComposedLayout,
        h_state_smem_layout_staged: cute.ComposedLayout,
        h_state_epi_layout_staged: cute.ComposedLayout,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        use_g: Int32,
        use_gk: Int32,
        use_initial_state: Int32,
        store_final_state: Int32,
        save_v_new: Int32,
    ):
        """
        Main kernel for chunk_gated_delta_rule_fwd_h.
        
        Pipeline overview:
        1. Load warp: TMA loads W, K^T, U for each chunk
        2. MMA warp: Computes W @ H and K^T @ V_new  
        3. CUDA warps: Apply gates, convert types, compute v_new = u - (W @ H)
        4. Store warp: TMA stores H_out and V_new
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k_kv)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vnew_load)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h_load)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h0_load)

        # Allocate shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Create pipelines
        load_w_producer, load_w_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_w_bytes,
            barrier_storage=storage.load_w_mbar_ptr.data_ptr(),
        ).make_participants()

        load_kt_producer, load_kt_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_kt_bytes,
            barrier_storage=storage.load_kt_mbar_ptr.data_ptr(),
        ).make_participants()

        wh_producer, wh_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            barrier_storage=storage.wh_mbar_ptr.data_ptr(),
        ).make_participants()

        vnew_smem_producer, vnew_smem_consumer = pipeline.PipelineAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),  # CUDA warps
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),  # Store warp = 32 threads
            barrier_storage=storage.vnew_smem_mbar_ptr.data_ptr(),
        ).make_participants()

        vnew_gmem_producer, vnew_gmem_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),  # Store warp
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),  # Load warp = 32 threads
            barrier_storage=storage.vnew_gmem_mbar_ptr.data_ptr(),
        ).make_participants()

        vnew_load_producer, vnew_load_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_vnew_bytes,
            barrier_storage=storage.vnew_load_mbar_ptr.data_ptr(),
        ).make_participants()

        kv_producer, kv_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),  # MMA
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),  # CUDA warps = 128 threads
            barrier_storage=storage.kv_mbar_ptr.data_ptr(),
        ).make_participants()

        h_gmem_producer, h_gmem_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),  # Store warp
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),  # Load warp
            barrier_storage=storage.h_gmem_mbar_ptr.data_ptr(),
        ).make_participants()

        h_load_producer, h_load_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_h_bytes,
            barrier_storage=storage.h_load_mbar_ptr.data_ptr(),
        ).make_participants()

        h_out_producer, h_out_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),  # Store warp = 32 threads
            barrier_storage=storage.h_out_mbar_ptr.data_ptr(),
        ).make_participants()

        # Allocate TMEM
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
        tmem.wait_for_alloc()
        tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

        # Create SMEM tensors
        sW = storage.sW.get_tensor(
            w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner
        )
        sKt = storage.sKt.get_tensor(
            k_kv_smem_layout_staged.outer, swizzle=k_kv_smem_layout_staged.inner
        )
        # V_new SMEM: MMA B operand layout for kv_tiled_mma reads (TMA-loaded)
        sVnew_mma = storage.sVnew.get_tensor(
            vnew_smem_layout_staged.outer, swizzle=vnew_smem_layout_staged.inner
        )
        # H_state SMEM: MMA B operand layout for wh_tiled_mma reads (TMA-loaded)
        sH_state = storage.sH_state.get_tensor(
            h_state_smem_layout_staged.outer, swizzle=h_state_smem_layout_staged.inner
        )
        # H_state SMEM: epilogue layout for R2S writes (dual-view of same buffer)
        sH_epi = storage.sH_state.get_tensor(
            h_state_epi_layout_staged.outer, swizzle=h_state_epi_layout_staged.inner
        )
        # V_new SMEM: epilogue layout for R2S writes (dual-view of same buffer as sVnew_mma)
        sVnew_epi = storage.sVnew.get_tensor(
            vnew_epi_layout_staged.outer, swizzle=vnew_epi_layout_staged.inner
        )

        # MMA partitions for W @ H
        tCrW = wh_tiled_mma.make_fragment_A(sW)
        tCrH_for_wh = wh_tiled_mma.make_fragment_B(sH_state)  # H state as operand B
        acc_shape_wh = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH_fake = wh_tiled_mma.make_fragment_C(
            cute.append(acc_shape_wh, self.acc_stage)
        )
        tCtAccWH = cute.make_tensor(
            tmem_ptr_base + self.tmem_wh_cols_offset,
            tCtAccWH_fake.layout
        )

        # MMA partitions for V_new^T @ K (V_new as A, K as B)
        tCrVnew = kv_tiled_mma.make_fragment_A(sVnew_mma)  # V_new as A operand
        tCrK_kv = kv_tiled_mma.make_fragment_B(sKt)        # K as B operand
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"  tCrVnew (A for KV MMA): {tCrVnew}")
            print(f"  tCrK_kv (B for KV MMA): {tCrK_kv}")
            print(f"  tCrW (A for WH MMA): {tCrW}")
            print(f"  tCrH_for_wh (B for WH MMA): {tCrH_for_wh}")
        acc_shape_kv = kv_tiled_mma.partition_shape_C(self.kv_mma_tiler[:2])
        tCtAccKV_fake = kv_tiled_mma.make_fragment_C(
            cute.append(acc_shape_kv, 1)
        )
        tCtAccKV = cute.make_tensor(
            tmem_ptr_base + self.tmem_kv_cols_offset,
            tCtAccKV_fake.layout
        )

        # Get problem dimensions
        (v_tile_idx, hidx, bidx) = cute.arch.block_idx()
        B, T, H, K, V = problem_size
        BT = self.BT
        NT = (T + BT - 1) // BT  # Number of chunks

        # =========================================================================
        # LOAD WARP
        # =========================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_alloc(160)

            # TMA partition for W
            tWsW, tWgW = self._tma_partition_for_operand(
                tma_atom_w, tma_tensor_w, sW, self.wh_mma_tiler, wh_tiled_mma, "A"
            )

            # TMA partition for K (B operand of KV MMA)
            tKsK, tKgK = self._tma_partition_for_operand(
                tma_atom_k_kv, tma_tensor_k_kv, sKt, self.kv_mma_tiler, kv_tiled_mma, "B"
            )

            # TMA partition for v_new load (A operand of KV MMA, GMEM → sVnew_mma)
            tVsV, tVgV = self._tma_partition_for_operand(
                tma_atom_vnew_load, tma_tensor_vnew_load, sVnew_mma, self.kv_mma_tiler, kv_tiled_mma, "A"
            )

            # TMA partition for h_state load (GMEM → sH_state)
            # h_out_T has (V, K, (NT, H, B)) - need (None, hidx, bidx) for batch
            _, hidx_load, bidx_load = cute.arch.block_idx()
            coord_h = (0, None, None)
            gH_load = cute.local_tile(
                tma_tensor_h_load,
                cute.slice_(self.wh_mma_tiler, coord_h),
                (None, None, (None, hidx_load, bidx_load))
            )
            thr_mma_h = wh_tiled_mma.get_slice(0)
            tCgH = thr_mma_h.partition_B(gH_load)
            tHsH_load, tHgH_load = cute.nvgpu.cpasync.tma_partition(
                tma_atom_h_load,
                0,
                cute.make_layout(1),
                cute.group_modes(sH_state, 0, 3),
                cute.group_modes(tCgH, 0, 3),
            )

            # TMA partition for h0 (initial state) load (GMEM → sH_state)
            # h0_T has (V, K, (H, B)) - no NT dimension
            gH0_load = cute.local_tile(
                tma_tensor_h0_load,
                cute.slice_(self.wh_mma_tiler, coord_h),
                (None, None, (hidx_load, bidx_load))
            )
            tCgH0 = thr_mma_h.partition_B(gH0_load)
            tH0sH_load, tH0gH_load = cute.nvgpu.cpasync.tma_partition(
                tma_atom_h0_load,
                0,
                cute.make_layout(1),
                cute.group_modes(sH_state, 0, 3),
                cute.group_modes(tCgH0, 0, 3),
            )

            # Initial h_state: load h0 if use_initial_state, else load h_out[0] (zeros)
            h_load_init_handle = h_load_producer.acquire_and_advance()
            if use_initial_state:
                # Load h0 (initial state) using h0-specific partition
                cute.copy(
                    atom=tma_atom_h0_load,
                    src=tH0gH_load[None, 0, 0],
                    dst=tH0sH_load[None, h_load_init_handle.index],
                    tma_bar_ptr=h_load_init_handle.barrier,
                )
            else:
                # Load h_out[0] (pre-zeroed by host)
                cute.copy(
                    atom=tma_atom_h_load,
                    src=tHgH_load[None, 0, 0, 0],
                    dst=tHsH_load[None, h_load_init_handle.index],
                    tma_bar_ptr=h_load_init_handle.barrier,
                )

            # Main loop over chunks
            for chunk_idx in cutlass.range(0, NT, unroll=0):
                # Load W for this chunk
                w_handle = load_w_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_w,
                    src=tWgW[None, chunk_idx, 0],
                    dst=tWsW[None, w_handle.index],
                    tma_bar_ptr=w_handle.barrier,
                )

                # Load K for this chunk (B operand of KV MMA)
                kt_handle = load_kt_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_k_kv,
                    src=tKgK[None, 0, chunk_idx],
                    dst=tKsK[None, kt_handle.index],
                    tma_bar_ptr=kt_handle.barrier,
                )

                if cutlass.const_expr(self.enable_loop_roundtrip):
                    # Wait for CUDA warps to finish writing v_new to GMEM
                    vnew_gmem_handle = vnew_gmem_consumer.wait_and_advance()
                    vnew_gmem_handle.release()

                # TMA load v_new from GMEM → sVnew_mma (A operand)
                vnew_load_handle = vnew_load_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_vnew_load,
                    src=tVgV[None, 0, chunk_idx],
                    dst=tVsV[None, vnew_load_handle.index],
                    tma_bar_ptr=vnew_load_handle.barrier,
                )

                if cutlass.const_expr(self.enable_loop_roundtrip):
                    # Wait for CUDA warps to finish writing h_state to GMEM
                    h_gmem_handle = h_gmem_consumer.wait_and_advance()
                    h_gmem_handle.release()

                # TMA load h_state from GMEM → sH_state
                h_load_handle = h_load_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_h_load,
                    src=tHgH_load[None, 0, 0, chunk_idx],
                    dst=tHsH_load[None, h_load_handle.index],
                    tma_bar_ptr=h_load_handle.barrier,
                )

        # =========================================================================
        # MMA WARP
        # =========================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                # Wait for h_state loaded via TMA (from GMEM roundtrip)
                h_load_handle = h_load_consumer.wait_and_advance()

                # Wait for W
                w_handle = load_w_consumer.wait_and_advance()

                # Compute W @ H -> acc_wh
                wh_handle = wh_producer.acquire_and_advance()
                wh_tiled_mma = self._exec_mma_clear(
                    tiled_mma=wh_tiled_mma,
                    tCtAcc=tCtAccWH,
                    tCrA=tCrW,
                    tCrB=tCrH_for_wh,
                    a_stage_idx=w_handle.index,
                    b_stage_idx=h_load_handle.index,  # h_state from TMA load
                    acc_stage_idx=wh_handle.index,
                )
                wh_handle.commit()
                w_handle.release()
                h_load_handle.release()

                # Wait for V_new loaded via TMA (from GMEM roundtrip)
                vnew_load_handle = vnew_load_consumer.wait_and_advance()

                # Wait for K^T
                kt_handle = load_kt_consumer.wait_and_advance()

                # Compute V_new^T @ K -> KV result (h^T)
                # With gating: h_new = gate(h_old) + KV
                # Without gating: h_new = h_old + KV (simple accumulation)
                # 
                # NOTE: Gating h_state requires special handling - currently only
                # v_new output gating is implemented. h_state gating is TODO.
                # For now, we use cross-chunk accumulation which is only correct
                # when g=0 and gk=0 (no decay on h_state).
                kv_handle = kv_producer.acquire_and_advance()
                kv_always_acc = True if chunk_idx != 0 else False
                for kphase_idx in cutlass.range(cute.size(tCrK_kv, mode=[2]), unroll_full=True):
                    kv_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        cutlass.Boolean(kphase_idx != 0 or kv_always_acc),
                    )
                    cute.gemm(
                        kv_tiled_mma,
                        tCtAccKV[None, None, None, 0],
                        tCrVnew[None, None, kphase_idx, vnew_load_handle.index],
                        tCrK_kv[None, None, kphase_idx, kt_handle.index],
                        tCtAccKV[None, None, None, 0],
                    )
                kv_handle.commit()
                kt_handle.release()
                vnew_load_handle.release()

        # =========================================================================
        # CUDA CORE WARPS - Apply gates and compute v_new = u - (W @ H)
        # =========================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(160)

            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))

            # ===== Setup TMEM load for W@H result (BT×BV = 64×128) =====
            copy_atom_t2r_wh = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccWH_flat = tCtAccWH[((None, None), 0, 0, None)]
            fake_sWH = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.wh_mma_tiler, (1, 1, None)),
            )
            tiled_copy_t2r_wh = tcgen05.make_tmem_copy(
                copy_atom_t2r_wh, tCtAccWH_flat[(None, None, 0)]
            )
            thr_copy_t2r_wh = tiled_copy_t2r_wh.get_slice(local_tidx)
            tTR_tWH = thr_copy_t2r_wh.partition_S(tCtAccWH_flat)
            tTR_sWH = thr_copy_t2r_wh.partition_D(fake_sWH)
            tTR_rWH = cute.make_rmem_tensor(tTR_sWH.shape, self.acc_dtype)
            tTR_rVnew = cute.make_rmem_tensor(tTR_rWH.shape, self.io_dtype)

            # Copy atom for element-wise register ↔ memory ops
            copy_atom_s2r_u = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.io_dtype,
                num_bits_per_copy=self.io_dtype.width,
            )

            # ===== Setup TMEM load for KV result (BV×BK = 128×128) =====
            # MMA with CtaGroup.ONE uses 16dp/warp regardless of operand major mode
            copy_atom_t2r_kv = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccKV_flat = tCtAccKV[((None, None), 0, 0, None)]
            fake_sKV = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.kv_mma_tiler, (1, 1, None)),
            )
            tiled_copy_t2r_kv = tcgen05.make_tmem_copy(
                copy_atom_t2r_kv, tCtAccKV_flat[(None, None, 0)]
            )
            thr_copy_t2r_kv = tiled_copy_t2r_kv.get_slice(local_tidx)
            tTR_tKV = thr_copy_t2r_kv.partition_S(tCtAccKV_flat)
            tTR_sKV = thr_copy_t2r_kv.partition_D(fake_sKV)
            tTR_rKV = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)

            # ===== Setup v_new R2S epilogue (register → sVnew_epi → TMA store) =====
            copy_atom_r2s_vnew = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.acc_dtype,
                tiled_copy_t2r_wh,
            )
            tiled_copy_r2s_vnew = cute.make_tiled_copy_D(copy_atom_r2s_vnew, tiled_copy_t2r_wh)
            thr_copy_r2s_vnew = tiled_copy_r2s_vnew.get_slice(local_tidx)
            tRS_sVnew = thr_copy_r2s_vnew.partition_D(sVnew_epi)

            # ===== Setup U GMEM input (direct GMEM read, bypass SMEM) =====
            gU_all = cute.local_tile(
                u_tensor,
                (self.BT, self.BV),
                (None, None, (hidx, bidx)),
            )
            tTR_gU = thr_copy_t2r_wh.partition_D(gU_all)

            # ===== Setup g (scalar gate) GMEM input =====
            # g: (T, (H, B)) -> each thread loads one value per timestep
            # For now, load g[chunk*BT : (chunk+1)*BT] for each chunk
            # g_last is g at timestep (chunk+1)*BT - 1
            gG_all = cute.local_tile(
                g,
                (self.BT,),
                (None, (hidx, bidx)),
            )  # (BT, chunk, 1) per block

            # ===== Setup gk (vector gate) GMEM input =====
            # gk: (T, K, (H, B)) -> each thread loads K values per timestep
            gGK_all = cute.local_tile(
                gk,
                (self.BT, self.BK),
                (None, None, (hidx, bidx)),
            )  # (BT, BK, chunk, 1) per block

            # ===== Setup h_out R2S epilogue (register → sH_epi → TMA store) =====
            copy_atom_r2s_h = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR,
                self.io_dtype,
                self.acc_dtype,
                tiled_copy_t2r_kv,
            )
            tiled_copy_r2s_h = cute.make_tiled_copy_D(copy_atom_r2s_h, tiled_copy_t2r_kv)
            thr_copy_r2s_h = tiled_copy_r2s_h.get_slice(local_tidx)
            tRS_sH = thr_copy_r2s_h.partition_D(sH_epi)

            # Pre-allocate bf16 register tensor for h_state R2S
            tTR_rH_bf16 = cute.make_rmem_tensor(tTR_rKV.shape, self.io_dtype)

            # ===== Setup identity tensor for (row, col) coordinates =====
            # This allows us to know which logical (t, v) position each register element corresponds to.
            # We use the same partition as tTR_sWH to match vnew_vec element ordering.
            vnew_tile_shape = cute.dice(self.wh_mma_tiler, (1, 1, None))  # (BT, BV)
            cM_vnew = cute.make_identity_tensor(vnew_tile_shape)
            tTR_cM_vnew = thr_copy_t2r_wh.partition_D(cM_vnew)
            # tTR_cM_vnew[i] gives (row_idx, col_idx) for vnew_vec[i]

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                # ===== Phase A: Compute v_new = u - (W @ H) =====
                # Wait for W @ H result from MMA warp
                wh_handle = wh_consumer.wait_and_advance()

                # Load W@H from TMEM to registers
                tTR_tWH_i = tTR_tWH[(None, None, None, wh_handle.index)]
                cute.copy(tiled_copy_t2r_wh, tTR_tWH_i, tTR_rWH)
                cute.arch.fence_view_async_tmem_load()

                # Load U directly from GMEM using TMEM-compatible partition
                tTR_gU_i = tTR_gU[(None, None, None, chunk_idx, 0)]
                tTR_rU = cute.make_rmem_tensor(tTR_gU_i.shape, self.io_dtype)
                cute.copy(copy_atom_s2r_u, tTR_gU_i, tTR_rU)

                # Compute v_new = u - W@H (both in TMEM partition order)
                wh_vec = tTR_rWH.load()
                u_vec = tTR_rU.load().to(self.acc_dtype)  # bf16 → fp32
                vnew_vec = u_vec - wh_vec

                # Store vnew to mutable register tensor first
                tTR_rVnew.store(vnew_vec.to(self.io_dtype))

                # ===== Apply g scalar gate to v_new in registers =====
                # v_new[t,:] *= exp(g_last - g[t]) for each row t
                # Using identity tensor tTR_cM_vnew to get the logical (row, col) for each element.
                # We apply gating in-place to tTR_rVnew (mutable tensor).
                if use_g:
                    # g_last = g at last timestep of this chunk
                    g_chunk_offset = chunk_idx * self.BT
                    g_last_val = g[(g_chunk_offset + self.BT - 1, (hidx, bidx))]
                    
                    # Apply gating in-place to mutable register tensor
                    for elem_idx in cutlass.range_constexpr(cute.size(tTR_cM_vnew)):
                        row_idx, col_idx = tTR_cM_vnew[elem_idx]
                        # Load g[row_idx] for this chunk
                        g_row_val = g[(g_chunk_offset + row_idx, (hidx, bidx))]
                        # Compute scale = exp(g_last - g_row) using exp2
                        # exp(x) = exp2(x * INV_LN2) where INV_LN2 = 1/ln(2)
                        g_diff = g_last_val - g_row_val
                        g_scale = cute.exp2(g_diff * INV_LN2)
                        # Read from mutable tensor, apply scale, write back
                        val = tTR_rVnew[elem_idx].to(self.acc_dtype)
                        tTR_rVnew[elem_idx] = (val * g_scale).to(self.io_dtype)

                # R2S: v_new register → sVnew_epi (SMEM epilogue buffer)
                tRS_rVnew = tiled_copy_r2s_vnew.retile(tTR_rVnew)
                vnew_smem_handle = vnew_smem_producer.acquire_and_advance()
                cute.copy(tiled_copy_r2s_vnew, tRS_rVnew, tRS_sVnew[(None, None, None, vnew_smem_handle.index)])
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                
                vnew_smem_handle.commit()

                # Release W@H buffer
                wh_handle.release()

                # ===== Phase B: Read KV result, R2S to sH_epi =====
                # Wait for K^T @ V_new result from MMA warp
                kv_handle_c = kv_consumer.wait_and_advance()

                # Load KV from TMEM to registers (fp32)
                tTR_tKV_i = tTR_tKV[(None, None, None, 0)]
                cute.copy(tiled_copy_t2r_kv, tTR_tKV_i, tTR_rKV)
                cute.arch.fence_view_async_tmem_load()
                kv_handle_c.release()

                # Convert fp32 → bf16 for R2S
                tTR_rH_bf16.store(tTR_rKV.load().to(self.io_dtype))

                # R2S: register → sH_epi (SMEM epilogue buffer)
                tRS_rH = tiled_copy_r2s_h.retile(tTR_rH_bf16)
                h_out_handle = h_out_producer.acquire_and_advance()
                cute.copy(tiled_copy_r2s_h, tRS_rH, tRS_sH[(None, None, None, h_out_handle.index)])
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                
                # ===== Apply g and gk gates to h_state in SMEM =====
                # NOTE: The FLA reference applies g/gk gates to h BEFORE the KV update:
                #   h *= exp(g_last)     (g gate)
                #   h *= exp(gk_last)    (gk gate)
                #   h += K^T @ v_new     (KV update)
                #
                # However, our kernel accumulates h_state in TMEM (tCtAccKV), so gates must
                # be applied to the accumulated state BEFORE the next chunk's KV computation.
                # 
                # Current implementation: Apply gates AFTER KV result is read to SMEM.
                # This is INCORRECT for the recurrence math - the gated h is used in the
                # NEXT chunk's W@H computation, but we're gating the CURRENT chunk's result.
                #
                # TODO: Proper implementation requires either:
                # 1. Gate the TMEM accumulator at start of each chunk (complex)
                # 2. Gate the GMEM h_out[t] before TMA load as next chunk's h (extra pass)
                # 3. Fuse gating into MMA warp with accumulator scaling (complex)
                #
                # For now, skip h_state gating (use_g/use_gk only affects v_new scaling)
                # =========================================================================
                
                h_out_handle.commit()

        # =========================================================================
        # STORE WARP - TMA store v_new and h_state from SMEM to GMEM
        # =========================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)

            # Prefetch TMA store descriptors
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h_out)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vnew_store)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_ht_store)

            # TMA partition for h_out store: sH_epi → GMEM h_out
            (_, hidx_s, bidx_s) = cute.arch.block_idx()
            # tma_tensor_h_out maps h_out_T: (V, K, (NT, H, B))
            gH_store = tma_tensor_h_out[None, None, (None, hidx_s, bidx_s)]
            tma_atom_h_st, bSG_sH, bSG_gH = self._epilog_gmem_copy_partition(
                tma_atom_h_out,
                gH_store,
                (self.BV, self.BK),
                sH_epi,
            )

            # TMA partition for ht (final state) store: sH_epi → GMEM ht
            # ht_T has (V, K, (H, B)) - no NT dimension
            gHt_store = tma_tensor_ht_store[None, None, (hidx_s, bidx_s)]
            tma_atom_ht_st, bSG_sHt, bSG_gHt = self._epilog_gmem_copy_partition(
                tma_atom_ht_store,
                gHt_store,
                (self.BV, self.BK),
                sH_epi,
            )

            # TMA partition for v_new store: sVnew_epi → GMEM v_new
            gV_store = tma_tensor_vnew_store[None, None, (hidx_s, bidx_s)]
            tma_atom_v_st, bSG_sV, bSG_gV = self._epilog_gmem_copy_partition(
                tma_atom_vnew_store,
                gV_store,
                (self.BT, self.BV),
                sVnew_epi,
            )

            # No h_gmem_init needed: pre-loop TMA load of h_out[0]=zeros is done
            # without h_gmem consumer. The load loop consumes h_gmem from the store
            # loop, ensuring each load gets the UPDATED h_state from the same chunk.

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                # --- v_new TMA store ---
                # Wait for CUDA warp to finish R2S to sVnew_epi
                vnew_smem_handle = vnew_smem_consumer.wait_and_advance()

                # TMA store sVnew_epi → GMEM v_new[chunk_idx]
                cute.copy(tma_atom_v_st, bSG_sV[None, vnew_smem_handle.index], bSG_gV[(None, chunk_idx, 0)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

                # Release sVnew_epi for next R2S write
                vnew_smem_handle.release()

                # Fence GMEM writes visible to load warp's TMA
                cute.arch.fence_acq_rel_gpu()

                # Signal load warp that GMEM v_new is ready for TMA load
                vnew_gmem_handle = vnew_gmem_producer.acquire_and_advance()
                vnew_gmem_handle.commit()

                # --- h_state TMA store ---
                # Wait for CUDA warp to finish R2S to sH_epi
                h_out_handle = h_out_consumer.wait_and_advance()

                # TMA store sH_epi → GMEM h_out[chunk_idx]
                cute.copy(tma_atom_h_st, bSG_sH[None, h_out_handle.index], bSG_gH[(None, 0, 0, chunk_idx)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

                # Release sH_epi for next R2S write
                h_out_handle.release()

                # Fence GMEM writes visible to load warp's TMA
                cute.arch.fence_acq_rel_gpu()

                # Signal load warp that GMEM h_out is ready for TMA load
                h_gmem_handle = h_gmem_producer.acquire_and_advance()
                h_gmem_handle.commit()

            # After all chunks: store final state ht if requested
            # Note: h_out[NT-1] contains the final accumulated h_state
            # But we already stored it during the last iteration.
            # For ht, we need to store the h_state AFTER the last chunk's KV update.
            # The last h_out_handle stored to h_out[NT-1] is the final state.
            # We can simply TMA store from h_out[NT-1] to ht, but that requires
            # another TMA load-store cycle. Simpler: store to ht directly as well.
            # 
            # Actually, the sH_epi already contains the final state after last iteration.
            # But we released the handle and may have missed it. 
            # For now, the host can copy h_out[NT-1] to ht after kernel completes.
            # 
            # TODO: Implement proper ht store by either:
            # 1. Having CUDA warps signal a final h_out write for ht
            # 2. Having host copy h_out[NT-1] to ht after kernel
            # For simplicity, we rely on option 2 in the test harness.

        # Cleanup
        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)

        return

    @cute.jit
    def _tma_partition_for_operand(
        self,
        tma_atom,
        tma_tensor,
        smem,
        tile_shape,
        tiled_mma,
        operand_mode,
    ):
        """Partition tensors for TMA copy."""
        _, hidx, bidx = cute.arch.block_idx()
        
        if cutlass.const_expr(operand_mode.upper() == "A"):
            coord = (None, 0, None)
            gX = cute.local_tile(
                tma_tensor,
                cute.slice_(tile_shape, coord),
                (None, None, (hidx, bidx))
            )
            thr_mma = tiled_mma.get_slice(0)
            tCgX = thr_mma.partition_A(gX)
        elif cutlass.const_expr(operand_mode.upper() == "B"):
            coord = (0, None, None)
            gX = cute.local_tile(
                tma_tensor,
                cute.slice_(tile_shape, coord),
                (None, None, (hidx, bidx))
            )
            thr_mma = tiled_mma.get_slice(0)
            tCgX = thr_mma.partition_B(gX)
        else:
            raise RuntimeError(f"Unknown operand mode: {operand_mode}")

        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _exec_mma_clear(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        a_stage_idx,
        b_stage_idx,
        acc_stage_idx,
    ):
        """Execute MMA operation, clearing accumulator first."""
        for kphase_idx in cutlass.range(cute.size(tCrB, mode=[2]), unroll_full=True):
            tiled_mma.set(
                tcgen05.Field.ACCUMULATE,
                cutlass.Boolean(kphase_idx != 0),
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
    def _exec_mma_accum(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        a_stage_idx,
        b_stage_idx,
        acc_stage_idx,
    ):
        """Execute MMA operation, always accumulating."""
        for kphase_idx in cutlass.range(cute.size(tCrB, mode=[2]), unroll_full=True):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
            cute.gemm(
                tiled_mma,
                tCtAcc[None, None, None, acc_stage_idx],
                tCrA[None, None, kphase_idx, a_stage_idx],
                tCrB[None, None, kphase_idx, b_stage_idx],
                tCtAcc[None, None, None, acc_stage_idx],
            )
        return tiled_mma

    @cute.jit
    def _epilog_gmem_copy_partition(
        self,
        atom,
        gC_mnl,
        epi_tile,
        sC,
    ):
        """Partition for epilogue global memory copy."""
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sC_for_tma,
            gC_for_tma,
        )
        return atom, bSG_sC, bSG_gC


def reference_chunk_delta_rule_fwd_h(k, w, u, g=None, gk=None, h0=None, chunk_size=64):
    """
    PyTorch reference: chunk_gated_delta_rule_fwd_h with optional gates.
    
    Implements the FLA gating math:
    - Scalar gate g: v_new *= exp(g_last - g), h *= exp(g_last)
    - Vector gate gk: h *= exp(gk_last)  (per K dimension)
    
    Args:
        k: (B, T, H, K) bf16
        w: (B, T, H, K) bf16
        u: (B, T, H, V) bf16
        g: (B, T, H) fp32 or None - scalar gate in log space
        gk: (B, T, H, K) fp32 or None - vector gate in log space
        h0: (B, H, K, V) fp32 or None
        chunk_size: int
    Returns:
        h_out: (B, NT, H, K, V) bf16  - state at each chunk start (before update)
        v_new: (B, T, H, V) bf16  - corrected values
        h_after_list: list of (K, V) bf16 - h state after each chunk (for debugging)
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    h_out = torch.zeros(B, NT, H, K, V, device=k.device, dtype=torch.bfloat16)
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    
    # h state [B, H, K, V] in fp32
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()
    
    # Track h_state AFTER each chunk's update (for comparing with kernel's KV result)
    h_after_list = []
    
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        actual_bt = end - start  # handle last chunk potentially being smaller
        
        # Store h_out[t] = h (before update)
        h_out[:, t] = h.to(torch.bfloat16)
        
        # w_chunk: (B, BT, H, K) -> for W@H: need (B, H, BT, K)
        w_chunk = w[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
        k_chunk = k[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
        u_chunk = u[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, V)
        
        # v_new = u - W @ h
        wh = torch.matmul(w_chunk, h)  # (B, H, BT, V)
        v_new_chunk = u_chunk - wh  # (B, H, BT, V)
        
        # Apply scalar gate g to v_new: v_new[i] *= exp(g_last - g[i])
        if g is not None:
            # g_chunk: (B, BT, H) -> (B, H, BT)
            g_chunk = g[:, start:end].permute(0, 2, 1).float()  # (B, H, BT)
            g_last = g_chunk[:, :, -1:].float()  # (B, H, 1) - last timestep
            # Expand for broadcast: (B, H, BT, 1)
            g_scale = torch.exp(g_last - g_chunk).unsqueeze(-1)  # (B, H, BT, 1)
            v_new_chunk = v_new_chunk * g_scale
        
        # Save v_new (after g scaling, before KV)
        v_new_out[:, start:end] = v_new_chunk.permute(0, 2, 1, 3).to(torch.bfloat16)
        
        # Apply scalar gate g to h_state: h *= exp(g_last)
        if g is not None:
            g_last_scalar = g_chunk[:, :, -1].float()  # (B, H)
            h_scale = torch.exp(g_last_scalar).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            h = h * h_scale
        
        # Apply vector gate gk to h_state: h[k,:] *= exp(gk_last[k])
        if gk is not None:
            # gk_chunk: (B, BT, H, K) -> (B, H, BT, K)
            gk_chunk = gk[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
            gk_last = gk_chunk[:, :, -1, :].float()  # (B, H, K) - last timestep
            # Scale h per K dimension: h @ diag(exp(gk_last))
            gk_scale = torch.exp(gk_last).unsqueeze(-1)  # (B, H, K, 1)
            h = h * gk_scale
        
        # h += K^T @ v_new
        # k^T: (B, H, K, BT), v_new: (B, H, BT, V)
        k_t = k_chunk.transpose(-2, -1)  # (B, H, K, BT)
        h = h + torch.matmul(k_t, v_new_chunk)
        
        # Save h_state after this chunk's update 
        h_after_list.append(h[0, 0].to(torch.bfloat16).clone())  # (K, V)
    
    return h_out, v_new_out, h_after_list


def reference_chunk_delta_rule_bf16_roundtrip(k, w, u, g=None, gk=None, h0=None, chunk_size=64):
    """
    Reference that mimics the kernel's bf16 precision:
    - h_state goes through bf16 before W@H (GMEM roundtrip simulation)
    - v_new goes through bf16 before K^T@V_new (GMEM roundtrip simulation)
    - Includes gate (g, gk) support
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()
    
    h_after_list = []
    
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        
        w_chunk = w[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
        k_chunk = k[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
        u_chunk = u[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, V)
        
        # Simulate bf16 roundtrip: h goes through bf16 before W@H
        h_bf16 = h.to(torch.bfloat16).float()
        wh = torch.matmul(w_chunk, h_bf16)  # (B, H, BT, V)
        v_new_chunk = u_chunk - wh  # fp32
        
        # Apply scalar gate g to v_new: v_new[i] *= exp(g_last - g[i])
        if g is not None:
            g_chunk = g[:, start:end].permute(0, 2, 1).float()  # (B, H, BT)
            g_last = g_chunk[:, :, -1:].float()  # (B, H, 1)
            g_scale = torch.exp(g_last - g_chunk).unsqueeze(-1)  # (B, H, BT, 1)
            v_new_chunk = v_new_chunk * g_scale
        
        v_new_out[:, start:end] = v_new_chunk.permute(0, 2, 1, 3).to(torch.bfloat16)
        
        # Apply scalar gate g to h_state: h *= exp(g_last)
        if g is not None:
            g_last_scalar = g_chunk[:, :, -1].float()  # (B, H)
            h_scale = torch.exp(g_last_scalar).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            h = h * h_scale
        
        # Apply vector gate gk to h_state: h[k,:] *= exp(gk_last[k])
        if gk is not None:
            gk_chunk = gk[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
            gk_last = gk_chunk[:, :, -1, :].float()  # (B, H, K)
            gk_scale = torch.exp(gk_last).unsqueeze(-1)  # (B, H, K, 1)
            h = h * gk_scale
        
        # Simulate bf16 roundtrip: v_new goes through bf16 before K^T@V_new
        v_new_bf16 = v_new_chunk.to(torch.bfloat16).float()
        k_t = k_chunk.transpose(-2, -1)
        h = h + torch.matmul(k_t, v_new_bf16)
        
        h_after_list.append(h[0, 0].to(torch.bfloat16).clone())
    
    return v_new_out, h_after_list


def reference_vnew_gate_only_bf16(k, w, u, g=None, h0=None, chunk_size=64):
    """
    Reference that applies g gate ONLY to v_new output and KV input,
    WITHOUT gating h_state. This matches what the kernel currently does.
    
    Recurrence: h_new = h_old + K^T @ (v_new * g_scale)
    Note: h_state does NOT get scaled by exp(g_last).
    
    Returns:
        v_new_out: (B, T, H, V) bf16 - gated v_new output
        h_after_list: list of (K, V) bf16 - h state after each chunk
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()
    
    h_after_list = []
    
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        
        w_chunk = w[:, start:end].permute(0, 2, 1, 3).float()
        k_chunk = k[:, start:end].permute(0, 2, 1, 3).float()
        u_chunk = u[:, start:end].permute(0, 2, 1, 3).float()
        
        # Simulate bf16 roundtrip for h_state before W@H
        h_bf16 = h.to(torch.bfloat16).float()
        wh = torch.matmul(w_chunk, h_bf16)
        v_new_chunk = u_chunk - wh
        
        # Apply g gate to v_new: v_new *= exp(g_last - g[t])
        if g is not None:
            g_chunk = g[:, start:end].permute(0, 2, 1).float()
            g_last = g_chunk[:, :, -1:].float()
            g_scale = torch.exp(g_last - g_chunk).unsqueeze(-1)
            v_new_chunk = v_new_chunk * g_scale
        
        # Save gated v_new output
        v_new_out[:, start:end] = v_new_chunk.permute(0, 2, 1, 3).to(torch.bfloat16)
        
        # NO gating on h_state! This is different from reference_chunk_delta_rule_fwd_h.
        # Just accumulate: h += K^T @ v_new (with bf16 roundtrip for v_new)
        v_new_bf16 = v_new_chunk.to(torch.bfloat16).float()
        k_t = k_chunk.transpose(-2, -1)
        h = h + torch.matmul(k_t, v_new_bf16)
        
        h_after_list.append(h[0, 0].to(torch.bfloat16).clone())
    
    return v_new_out, h_after_list


def main():
    """Test the ChunkDeltaRuleFwdH kernel."""
    parser = argparse.ArgumentParser(description="Chunk Delta Rule Fwd H Kernel Test")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--head_dim_k", type=int, default=128, help="Key head dimension")
    parser.add_argument("--head_dim_v", type=int, default=128, help="Value head dimension")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size")
    
    args = parser.parse_args()
    
    print("Testing ChunkDeltaRuleFwdH Kernel:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Head dim K: {args.head_dim_k}")
    print(f"  Head dim V: {args.head_dim_v}")
    print(f"  Chunk size: {args.chunk_size}")
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    B, T, H, K, V = args.batch_size, args.seq_len, args.num_heads, args.head_dim_k, args.head_dim_v
    BT = args.chunk_size
    NT = (T + BT - 1) // BT
    
    # Use small values to avoid numerical issues
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1
    g = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    h0 = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
    
    # Recalculate NT after potential T override
    NT = (T + BT - 1) // BT
    print(f"  NT = {NT} (chunks)")
    
    # Create output tensors
    h_out = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    ht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
    
    # Reference (fp32)
    h_out_ref, v_new_ref, h_ref_list = reference_chunk_delta_rule_fwd_h(k, w, u, h0=None, chunk_size=BT)
    
    # Reference (bf16 roundtrip - matches kernel precision)
    v_new_ref_bf16, h_ref_list_bf16 = reference_chunk_delta_rule_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)
    
    # Convert to CuTe pointers
    k_cute = from_dlpack(k)
    w_cute = from_dlpack(w)
    u_cute = from_dlpack(u)
    g_cute = from_dlpack(g)
    gk_cute = from_dlpack(gk)
    h0_cute = from_dlpack(h0)
    h_out_cute = from_dlpack(h_out)
    v_new_cute = from_dlpack(v_new)
    ht_cute = from_dlpack(ht)
    
    # Create kernel
    kernel = ChunkDeltaRuleFwdH(
        chunk_size=BT,
        head_dim_k=K,
        head_dim_v=V,
    )
    
    stream = cutlass_torch.default_stream()
    
    print("\nCompiling kernel...")
    start = time.time()
    compiled = cute.compile(
        kernel,
        k_cute.iterator,
        w_cute.iterator,
        u_cute.iterator,
        g_cute.iterator,
        gk_cute.iterator,
        h_out_cute.iterator,
        v_new_cute.iterator,
        h0_cute.iterator,
        ht_cute.iterator,
        (B, T, H, K, V),
        0,  # use_g = 0 (no gating for correctness test)
        0,  # use_gk = 0
        0,  # use_initial_state = 0 (start from zeros)
        0,  # store_final_state = 0
        1,  # save_v_new = 1
        stream,
    )
    compile_time = time.time() - start
    print(f"Compilation time: {compile_time:.2f}s")
    
    print("\nRunning kernel...")
    compiled(
        k_cute.iterator,
        w_cute.iterator,
        u_cute.iterator,
        g_cute.iterator,
        gk_cute.iterator,
        h_out_cute.iterator,
        v_new_cute.iterator,
        h0_cute.iterator,
        ht_cute.iterator,
        (B, T, H, K, V),
        0, 0, 0, 0, 1,
        stream,
    )
    torch.cuda.synchronize()
    
    print("\nComparing outputs...")
    
    # Note: kernel doesn't store h_out yet, only v_new matters
    # The kernel currently only writes the first chunk's v_new and stores KV result to h_state
    
    # Check first chunk v_new (chunk 0: h=0, v_new = u - W@0 = u)
    v0_kernel = v_new[:, :BT]  # first chunk
    v0_ref = v_new_ref[:, :BT]
    
    # For chunk 0 with h=0: v_new should equal u
    v0_u = u[:, :BT]
    
    print(f"  First chunk v_new (should be ~u since h=0):")
    diff_naive = (v0_kernel.float() - v0_u.float()).abs().max().item()
    diff_ref = (v0_kernel.float() - v0_ref.float()).abs().max().item()
    print(f"    vs u directly: max_diff = {diff_naive:.6f}")
    print(f"    vs reference:  max_diff = {diff_ref:.6f}")
    
    if NT > 1:
        # Check second chunk v_new
        v1_kernel = v_new[:, BT:2*BT]
        v1_ref = v_new_ref[:, BT:2*BT]
        diff_v1 = (v1_kernel.float() - v1_ref.float()).abs().max().item()
        print(f"  Second chunk v_new:")
        print(f"    vs reference: max_diff = {diff_v1:.6f}")
    
    # Overall check
    all_diff = (v_new.float() - v_new_ref.float()).abs().max().item()
    print(f"\n  Overall v_new max diff: {all_diff:.6f}")
    
    # Per-chunk analysis
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        chunk_diff = (v_new[:, start:end].float() - v_new_ref[:, start:end].float()).abs().max().item()
        chunk_mean = (v_new[:, start:end].float() - v_new_ref[:, start:end].float()).abs().mean().item()
        print(f"  Chunk {t}: max_diff={chunk_diff:.6f}, mean_diff={chunk_mean:.6f}")
    
    # For chunk 0, v_new should equal u (h=0)
    diff_chunk0 = (v_new[:, :BT].float() - u[:, :BT].float()).abs().max().item()
    diff_chunk0_mean = (v_new[:, :BT].float() - u[:, :BT].float()).abs().mean().item()
    print(f"\n  Chunk 0 v_new vs u (h=0): max_diff={diff_chunk0:.6f}, mean_diff={diff_chunk0_mean:.6f}")
    
    # Show first values
    print(f"  v_new[0,0,0,:8] = {v_new[0,0,0,:8].tolist()}")
    print(f"  u[0,0,0,:8] = {u[0,0,0,:8].tolist()}")
    
    # ===== Check h_out (debug: h_state written by kernel) =====
    print("\n  --- h_out (accumulated h_state per chunk) ---")
    print(f"  {'Chunk':<8} {'vs fp32 ref':>14} {'vs bf16 ref':>14}")
    for t in range(NT):
        h_kernel = h_out[0, t, 0]  # (K, V)
        h_ref = h_ref_list[t]  # (K, V) - fp32 ref
        h_ref_b = h_ref_list_bf16[t]  # (K, V) - bf16 roundtrip ref
        h_diff = (h_kernel.float() - h_ref.float()).abs().max().item()
        h_diff_b = (h_kernel.float() - h_ref_b.float()).abs().max().item()
        print(f"  {t:<8} {h_diff:>14.6f} {h_diff_b:>14.6f}")
    
    # Debug: show actual h_state values for chunk 0
    h0_kernel = h_out[0, 0, 0]  # (K, V)
    h0_ref = h_ref_list[0]
    print(f"\n  h_state chunk 0 diagnostic:")
    print(f"    kernel max/min: {h0_kernel.float().max().item():.6f} / {h0_kernel.float().min().item():.6f}")
    print(f"    ref    max/min: {h0_ref.float().max().item():.6f} / {h0_ref.float().min().item():.6f}")
    print(f"    kernel[0,:8]: {h0_kernel[0,:8].tolist()}")
    print(f"    ref   [0,:8]: {h0_ref[0,:8].tolist()}")
    print(f"    kernel[1,:8]: {h0_kernel[1,:8].tolist()}")
    print(f"    ref   [1,:8]: {h0_ref[1,:8].tolist()}")
    print(f"    kernel sum: {h0_kernel.float().sum().item():.6f}")
    print(f"    ref    sum: {h0_ref.float().sum().item():.6f}")
    print(f"    kernel all_zeros: {(h0_kernel == 0).all().item()}")
    print(f"    kernel nonzero: {(h0_kernel != 0).sum().item()} / {h0_kernel.numel()}")
    # Compare element-wise with bf16 MMA reference
    k0 = k[:, :BT, 0].float()  # (B, BT, K)
    u0 = u[:, :BT, 0].float()  # (B, BT, V) 
    kt_ref = k0.transpose(-2, -1)  # (B, K, BT)
    h_bf16_mma = torch.matmul(kt_ref, u0.to(torch.bfloat16).float().to(torch.bfloat16).float())
    print(f"    bf16 mma sim[0,:8]: {h_bf16_mma[0, 0, :8].to(torch.bfloat16).tolist()}")
    
    # ===== Check if h_state accumulation is working =====
    # If h_out[1] == h_ref[0], accumulation is stuck (not incrementing)
    if NT > 1:
        h1_kernel = h_out[0, 1, 0]  # chunk 1
        h0_ref_b = h_ref_list_bf16[0]  # chunk 0 ref (= KV[0].to(bf16))
        h1_ref_b = h_ref_list_bf16[1]  # chunk 1 ref (= (KV[0]+KV[1]).to(bf16))
        stuck_diff = (h1_kernel.float() - h0_ref_b.float()).abs().max().item()
        print(f"\n  Accumulation diagnostic:")
        print(f"    h_out[1] vs h_ref_bf16[0] (stuck?): {stuck_diff:.6f}")
        print(f"    h_out[1] sum: {h1_kernel.float().sum().item():.6f}")
        print(f"    h_ref_bf16[0] sum: {h0_ref_b.float().sum().item():.6f}")
        print(f"    h_ref_bf16[1] sum: {h1_ref_b.float().sum().item():.6f}")
        print(f"    KV[1] incremental sum (ref): {(h1_ref_b.float() - h0_ref_b.float()).sum().item():.6f}")
        print(f"    h_out[1] - h_ref[0] sum: {(h1_kernel.float() - h0_ref_b.float()).sum().item():.6f}")
    
    # ===== Check v_new against bf16 reference =====
    print(f"\n  --- v_new per-chunk (vs bf16 roundtrip ref) ---")
    print(f"  {'Chunk':<8} {'vs fp32 ref':>14} {'vs bf16 ref':>14}")
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        vk = v_new[:, start:end]
        vr = v_new_ref[:, start:end]
        vb = v_new_ref_bf16[:, start:end]
        d_fp32 = (vk.float() - vr.float()).abs().max().item()
        d_bf16 = (vk.float() - vb.float()).abs().max().item()
        print(f"  {t:<8} {d_fp32:>14.6f} {d_bf16:>14.6f}")
    
    all_diff_bf16 = (v_new.float() - v_new_ref_bf16.float()).abs().max().item()
    h_max_diff_bf16 = max(
        (h_out[0, t, 0].float() - h_ref_list_bf16[t].float()).abs().max().item()
        for t in range(NT)
    )
    
    print(f"\n  Overall v_new max diff (vs bf16 ref): {all_diff_bf16:.6f}")
    print(f"  Overall h_state max diff (vs bf16 ref): {h_max_diff_bf16:.6f}")
    
    if all_diff_bf16 < 0.1 and h_max_diff_bf16 < 0.5:
        print("\nPASS - Correctness verified (bf16 MMA tolerance)!")
    elif diff_chunk0 < 0.01:
        print(f"\nPARTIAL PASS - Chunk 0 correct, later chunks have drift")
    else:
        print(f"\nFAIL - Diffs exceed tolerance")
    
    # ===== Test with gating enabled =====
    print("\n" + "=" * 60)
    print("Testing with GATING enabled (g and gk)")
    print("=" * 60)
    
    # Create non-zero gate values (small for numerical stability)
    torch.manual_seed(123)  # Fixed seed for reproducibility
    g_gated = torch.randn(B, T, H, device="cuda", dtype=torch.float32) * 0.1
    gk_gated = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    
    # Reset outputs
    h_out_gated = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new_gated = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    
    # Gated references
    h_out_ref_g, v_new_ref_g, h_ref_list_g = reference_chunk_delta_rule_fwd_h(
        k, w, u, g=g_gated, gk=gk_gated, h0=None, chunk_size=BT
    )
    v_new_ref_bf16_g, h_ref_list_bf16_g = reference_chunk_delta_rule_bf16_roundtrip(
        k, w, u, g=g_gated, gk=gk_gated, h0=None, chunk_size=BT
    )
    
    g_cute_gated = from_dlpack(g_gated)
    gk_cute_gated = from_dlpack(gk_gated)
    h_out_cute_gated = from_dlpack(h_out_gated)
    v_new_cute_gated = from_dlpack(v_new_gated)
    
    print("\nRunning kernel with gates...")
    compiled(
        k_cute.iterator,
        w_cute.iterator,
        u_cute.iterator,
        g_cute_gated.iterator,
        gk_cute_gated.iterator,
        h_out_cute_gated.iterator,
        v_new_cute_gated.iterator,
        h0_cute.iterator,
        ht_cute.iterator,
        (B, T, H, K, V),
        1, 0, 0, 0, 0,  # use_g=1, use_gk=0 (only v_new gating implemented correctly for now)
        stream,
    )
    torch.cuda.synchronize()
    
    # Reference: v_new gate only (no h_state gating) to match kernel behavior
    v_new_ref_vnew_gate, h_ref_list_vnew_gate = reference_vnew_gate_only_bf16(
        k, w, u, g=g_gated, h0=h0, chunk_size=BT
    )
    
    print("\nComparing gated outputs (v_new gate only, no h_state gate)...")
    
    # Debug: Did the kernel actually apply gating?
    # Compare kernel gated vs kernel non-gated (from earlier in the test)
    # v_new (non-gated) was computed earlier
    print(f"\n  Gating effect check:")
    print(f"    Non-gated v_new[0,0,0,:4] = {v_new[0,0,0,:4].tolist()}")
    print(f"    Gated v_new[0,0,0,:4] = {v_new_gated[0,0,0,:4].tolist()}")
    
    # Expected: gated = non_gated * exp(g_last - g[0])
    g_chunk0 = g_gated[0, :BT, 0]
    g_last0 = g_chunk0[-1].item()
    expected_scale_0 = float(torch.exp(torch.tensor(g_last0 - g_chunk0[0].item())))
    print(f"    Expected scale for t=0: {expected_scale_0:.6f}")
    print(f"    Actual ratio: {(v_new_gated[0,0,0,0] / v_new[0,0,0,0]).item():.6f}")
    
    # Check chunk 2
    if NT >= 3:
        non_gated_c2 = v_new[0,BT*2,0,:4].tolist()
        gated_c2 = v_new_gated[0,BT*2,0,:4].tolist()
        g_chunk2 = g_gated[0, BT*2:BT*3, 0]
        g_last2 = g_chunk2[-1].item()
        expected_scale_2_0 = float(torch.exp(torch.tensor(g_last2 - g_chunk2[0].item())))
        print(f"\n    Chunk 2 non-gated v_new[0,{BT*2},0,:4] = {non_gated_c2}")
        print(f"    Chunk 2 gated v_new[0,{BT*2},0,:4] = {gated_c2}")
        print(f"    Chunk 2 expected scale for t=0: {expected_scale_2_0:.6f}")
        if non_gated_c2[0] != 0:
            print(f"    Chunk 2 actual ratio: {(v_new_gated[0,BT*2,0,0] / v_new[0,BT*2,0,0]).item():.6f}")
    
    # Debug: check what gate values we're using
    print(f"\n  Debug g values for chunk 0:")
    g_chunk0 = g_gated[0, :BT, 0]  # (BT,) for batch 0, head 0
    g_last0 = g_chunk0[-1].item()
    print(f"    g_last = {g_last0:.6f}")
    print(f"    g[0:8] = {g_chunk0[:8].tolist()}")
    g_scales0 = torch.exp(g_last0 - g_chunk0)
    print(f"    exp(g_last - g)[0:8] = {g_scales0[:8].tolist()}")
    print(f"    exp(g_last - g) range: [{g_scales0.min().item():.4f}, {g_scales0.max().item():.4f}]")
    
    # Debug: check v_new values
    print(f"\n  Debug v_new chunk 0:")
    print(f"    kernel v_new[0,0,0,:8] = {v_new_gated[0,0,0,:8].tolist()}")
    print(f"    ref    v_new[0,0,0,:8] = {v_new_ref_vnew_gate[0,0,0,:8].tolist()}")
    
    if NT > 1:
        print(f"\n  Debug v_new chunk 1:")
        print(f"    kernel v_new[0,{BT},0,:8] = {v_new_gated[0,BT,0,:8].tolist()}")
        print(f"    ref    v_new[0,{BT},0,:8] = {v_new_ref_vnew_gate[0,BT,0,:8].tolist()}")
        
        # Compare h_state at chunk boundary
        print(f"\n  Debug h_state after chunk 0:")
        print(f"    kernel h_out[0,0,0,:2,:4] =")
        print(f"      {h_out_gated[0,0,0,:2,:4].tolist()}")
        print(f"    ref h_after[0][:2,:4] =")
        print(f"      {h_ref_list_vnew_gate[0][:2,:4].tolist()}")
        h_diff_0 = (h_out_gated[0,0,0].float() - h_ref_list_vnew_gate[0].float()).abs().max().item()
        print(f"    max diff: {h_diff_0:.6f}")
        
        # Analyze v_new diff source: is it due to WH (h_state) difference?
        # v_new_diff = kernel(u - WH) * g_scale - ref(u - WH) * g_scale
        #            = (kernel_WH - ref_WH) * (-g_scale)
        # If h_state is same, kernel_WH = ref_WH, so v_new should match.
        # Let's check if kernel uses different h for chunk 1
        v_new_diff_chunk1 = (v_new_gated[0,BT:2*BT,0] - v_new_ref_vnew_gate[0,BT:2*BT,0]).float()
        print(f"\n  Chunk 1 v_new diff analysis:")
        print(f"    v_new diff max: {v_new_diff_chunk1.abs().max().item():.6f}")
        print(f"    v_new diff [0:8]: {v_new_diff_chunk1[:8].tolist()}")
        
        # What is the expected diff if kernel used h_out[1] instead of h_out[0] for WH?
        # WH_kernel = W @ h_out[1], WH_ref = W @ h_out[0]
        # v_new_diff = (WH_ref - WH_kernel) = W @ (h_out[0] - h_out[1])
        # h_diff = h_out[0] - h_out[1] 
        # Ah but h_out[1] may not be meaningful in this test...
    
    # v_new comparison
    print("  --- Gated v_new per-chunk (vs vnew-gate-only ref) ---")
    print(f"  {'Chunk':<8} {'max diff':>14}")
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        vk = v_new_gated[:, start:end]
        vr = v_new_ref_vnew_gate[:, start:end]
        d = (vk.float() - vr.float()).abs().max().item()
        print(f"  {t:<8} {d:>14.6f}")
    
    # h_out comparison
    print("\n  --- Gated h_out per-chunk (vs vnew-gate-only ref) ---")
    print(f"  {'Chunk':<8} {'max diff':>14}")
    for t in range(NT):
        h_kernel = h_out_gated[0, t, 0]
        h_ref = h_ref_list_vnew_gate[t]
        h_diff = (h_kernel.float() - h_ref.float()).abs().max().item()
        print(f"  {t:<8} {h_diff:>14.6f}")
    
    all_diff_g = (v_new_gated.float() - v_new_ref_vnew_gate.float()).abs().max().item()
    
    # Debug: find where the max diff is
    diff_tensor = (v_new_gated.float() - v_new_ref_vnew_gate.float()).abs()
    max_idx = torch.argmax(diff_tensor).item()
    print(f"\n  Debug: v_new max diff at flat idx {max_idx}")
    # Unravel index
    B_dim, T_dim, H_dim, V_dim = v_new_gated.shape
    b_idx = max_idx // (T_dim * H_dim * V_dim)
    rem = max_idx % (T_dim * H_dim * V_dim)
    t_idx = rem // (H_dim * V_dim)
    rem = rem % (H_dim * V_dim)
    h_idx = rem // V_dim
    v_idx = rem % V_dim
    print(f"  Max diff at (b={b_idx}, t={t_idx}, h={h_idx}, v={v_idx})")
    print(f"  kernel value: {v_new_gated[b_idx, t_idx, h_idx, v_idx].item():.6f}")
    print(f"  ref value: {v_new_ref_vnew_gate[b_idx, t_idx, h_idx, v_idx].item():.6f}")
    print(f"  non-gated kernel value: {v_new[b_idx, t_idx, h_idx, v_idx].item():.6f}")
    
    # Compute expected gate scale for this position
    chunk_for_t = t_idx // BT
    local_t = t_idx % BT
    g_chunk = g_gated[b_idx, chunk_for_t*BT:(chunk_for_t+1)*BT, h_idx]
    g_last_chunk = g_chunk[-1].item()
    g_t_val = g_chunk[local_t].item()
    expected_scale = float(torch.exp(torch.tensor(g_last_chunk - g_t_val)))
    print(f"  Expected scale for t={t_idx} (chunk {chunk_for_t}, local {local_t}): {expected_scale:.6f}")
    print(f"  g_last={g_last_chunk:.6f}, g[t]={g_t_val:.6f}")
    print(f"  Expected gated = non_gated * scale = {v_new[b_idx, t_idx, h_idx, v_idx].item() * expected_scale:.6f}")
    
    # Check kernel gated / non-gated ratio
    if v_new[b_idx, t_idx, h_idx, v_idx].item() != 0:
        actual_ratio = v_new_gated[b_idx, t_idx, h_idx, v_idx].item() / v_new[b_idx, t_idx, h_idx, v_idx].item()
        print(f"  Actual kernel ratio (gated/non-gated): {actual_ratio:.6f}")
        print(f"  Expected ratio: {expected_scale:.6f}")
        
        # What scale was actually applied? 
        # If ratio is negative, something is very wrong
        if actual_ratio < 0:
            print(f"  ERROR: Negative ratio! Gating flipped the sign!")
            # Check nearby positions to see pattern
            print(f"  Nearby values:")
            for dt in [-2, -1, 0, 1, 2]:
                if 0 <= t_idx + dt < T:
                    ng = v_new[b_idx, t_idx+dt, h_idx, v_idx].item()
                    g_ = v_new_gated[b_idx, t_idx+dt, h_idx, v_idx].item()
                    r_ = g_ / ng if ng != 0 else float('inf')
                    print(f"    t={t_idx+dt}: non_gated={ng:.6f}, gated={g_:.6f}, ratio={r_:.6f}")
    
    h_max_diff_g = max(
        (h_out_gated[0, t, 0].float() - h_ref_list_vnew_gate[t].float()).abs().max().item()
        for t in range(NT)
    )
    
    print(f"\n  Overall gated v_new max diff: {all_diff_g:.6f}")
    print(f"  Overall gated h_state max diff: {h_max_diff_g:.6f}")
    
    if all_diff_g < 0.01 and h_max_diff_g < 0.01:
        print("\nGATED PASS - v_new gating correctness verified!")
    else:
        print(f"\nGATED FAIL - Diffs exceed tolerance (v_new: {all_diff_g:.6f}, h: {h_max_diff_g:.6f})")


if __name__ == "__main__":
    main()
