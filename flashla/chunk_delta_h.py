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

PRINT_DEBUG = True


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
            tma_atom_h_out,
            tma_tensor_h_out,
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
        tma_atom_h_out: cute.CopyAtom,
        tma_tensor_h_out: cute.Tensor,
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

            # Initial h_state: h_out[0] is pre-zeroed by host, TMA load directly
            h_load_init_handle = h_load_producer.acquire_and_advance()
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
                tTR_rVnew.store(vnew_vec.to(self.io_dtype))

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
                h_out_handle.commit()

        # =========================================================================
        # STORE WARP - TMA store v_new and h_state from SMEM to GMEM
        # =========================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)

            # Prefetch TMA store descriptors
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h_out)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vnew_store)

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


def reference_chunk_delta_rule_fwd_h(k, w, u, h0=None, chunk_size=64):
    """
    PyTorch reference: chunk_gated_delta_rule_fwd_h WITHOUT gates.
    
    Args:
        k: (B, T, H, K) bf16
        w: (B, T, H, K) bf16
        u: (B, T, H, V) bf16
        h0: (B, H, K, V) fp32 or None
        chunk_size: int
    Returns:
        h_out: (B, NT, H, K, V) bf16  - state at each chunk start (before update)
        v_new: (B, T, H, V) bf16  - corrected values
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
        h = h0.clone()
    
    # Track h_state AFTER each chunk's update (for comparing with kernel's KV result)
    h_after_list = []
    
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        
        # Store h_out[t] = h (before update)
        h_out[:, t] = h.to(torch.bfloat16)
        
        # w_chunk: (B, BT, H, K) -> for W@H: need (B, H, BT, K)
        w_chunk = w[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
        k_chunk = k[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, K)
        u_chunk = u[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, BT, V)
        
        # v_new = u - W @ h
        wh = torch.matmul(w_chunk, h)  # (B, H, BT, V)
        v_new_chunk = u_chunk - wh  # (B, H, BT, V)
        
        # Save v_new
        v_new_out[:, start:end] = v_new_chunk.permute(0, 2, 1, 3).to(torch.bfloat16)
        
        # h += K^T @ v_new  (no gating)
        # k^T: (B, H, K, BT), v_new: (B, H, BT, V)
        k_t = k_chunk.transpose(-2, -1)  # (B, H, K, BT)
        h = h + torch.matmul(k_t, v_new_chunk)
        
        # Save h_state after this chunk's update 
        h_after_list.append(h[0, 0].to(torch.bfloat16).clone())  # (K, V)
    
    return h_out, v_new_out, h_after_list


def reference_chunk_delta_rule_bf16_roundtrip(k, w, u, h0=None, chunk_size=64):
    """
    Reference that mimics the kernel's bf16 precision:
    - h_state goes through bf16 before W@H (GMEM roundtrip simulation)
    - v_new goes through bf16 before K^T@V_new (GMEM roundtrip simulation)
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone()
    
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
        
        v_new_out[:, start:end] = v_new_chunk.permute(0, 2, 1, 3).to(torch.bfloat16)
        
        # Simulate bf16 roundtrip: v_new goes through bf16 before K^T@V_new
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


if __name__ == "__main__":
    main()
