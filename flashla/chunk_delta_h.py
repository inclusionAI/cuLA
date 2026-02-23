# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Chunk Gated Delta Rule Forward H Kernel V2 - No GMEM Roundtrip (Register-Carry)

Optimized version eliminating GMEM roundtrip:
- h_state: carried in CUDA registers across chunks (no R2T needed)
- v_new: computed in registers → R2S to sVnew (SMEM) → KV MMA A operand

Both MMAs share M=BV=64 and use SS (SMEM×SMEM) operand mode:
- WH MMA: state(BV,BK) @ W(BT,BK) → WH_acc(BV,BT)
- KV MMA: v_new^T(BV,BT) @ K^T(BK,BT) → update(BV,BK)  [ACCUMULATE=False]

After KV MMA:  h_new = G * h + update  (in registers)
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

LN2 = 0.6931471805599453
INV_LN2 = 1.4426950408889634


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class ChunkDeltaRuleFwdH:
    """
    V2: No GMEM roundtrip. Both MMAs share M=BV=64, SS operand mode.
    h carried in CUDA registers; KV MMA only computes update term.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        is_varlen: bool = False,
        BV: int = None,
        num_stages: int = 2,
        min_occupancy: int = 1,
        persistent: bool = True,
    ):
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.is_varlen = is_varlen

        self.BT = chunk_size   # 64
        self.BK = head_dim_k   # 128
        self.BV = BV if BV is not None else head_dim_v  # V tiling (default: no tiling)

        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        self.empty_warp_id = 7
        # Register allocation:
        # - occ=1: 208 regs (varlen) / 232 regs (non-varlen) for CUDA warps
        #   208 is the minimum to eliminate all register spilling in varlen mode
        # - occ=2: 128 regs (just enough for 64 h-state regs + 64 spare)
        self.min_occupancy = min_occupancy
        self.persistent = persistent if is_varlen else False  # only meaningful for varlen
        if min_occupancy >= 2:
            self.num_regs_cuda = 128
        else:
            self.num_regs_cuda = 208 if is_varlen else 232
        self.num_regs_others = 40
        self.threads_per_cta = self.threads_per_warp * 8

        # WH MMA tiler: (M=BV=64, N=BT=64, K=BK=128), A & B both SS
        self.wh_mma_tiler = (self.BV, self.BT, self.BK)
        # KV MMA tiler: (M=BV=64, N=BK=128, K=BT=64), A & B both SS
        self.kv_mma_tiler = (self.BV, self.BK, self.BT)

        self.k_stage = num_stages
        self.w_stage = num_stages
        # For occ>=2: reduce CUDA→Store / Load→CUDA stages to 1
        # to fit SMEM under 114KB (228KB/2)
        if min_occupancy >= 2:
            self.u_stage = 1
            self.h_out_stage = 1
            self.vnew_store_stage = 1
        else:
            self.u_stage = 2
            self.h_out_stage = 2
            self.vnew_store_stage = 2
        self.acc_stage = 1
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        self.gk_stage = 2

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta,
        )
        # Barrier for CUDA warp-group sync during cooperative gk_scale precomputation
        self.gk_precompute_bar = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len(self.cuda_warp_ids),  # 128
        )
        self.buffer_align_bytes = 128

    @staticmethod
    def _plan_tmem_offsets(tiled_mma_wh, tile_wh, tiled_mma_kv, tile_kv, state_tmem_layout, vnew_tmem_layout, acc_stages):
        SM100_TMEM_CAPACITY_COLS = 512
        # WH acc: (BV=64, BT=64) FP32
        wh_shape = tiled_mma_wh.partition_shape_C(tile_wh[:2])
        wh_fake = tiled_mma_wh.make_fragment_C(cute.append(wh_shape, acc_stages))
        num_wh = tcgen05.find_tmem_tensor_col_offset(wh_fake)
        # State TMEM A operand for WH MMA: (BV=64, BK=128) BF16
        tCrState_fake = tiled_mma_wh.make_fragment_A(state_tmem_layout.outer.shape)
        num_state = tcgen05.find_tmem_tensor_col_offset(tCrState_fake)
        # v_new TMEM A operand for KV MMA: (BV=64, BT=64) BF16
        tCrVnew_fake = tiled_mma_kv.make_fragment_A(vnew_tmem_layout.outer.shape)
        num_vnew = tcgen05.find_tmem_tensor_col_offset(tCrVnew_fake)
        # KV acc: (BV=64, BK=128) FP32
        kv_shape = tiled_mma_kv.partition_shape_C(tile_kv[:2])
        kv_fake = tiled_mma_kv.make_fragment_C(cute.append(kv_shape, 1))
        num_kv = tcgen05.find_tmem_tensor_col_offset(kv_fake)

        wh_off = 0
        state_off = wh_off + num_wh
        vnew_off = state_off + num_state
        kv_off = vnew_off + num_vnew
        total_tmp = kv_off + num_kv
        total = 1
        while total < total_tmp:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"  TMEM: WH={num_wh}@{wh_off}, State={num_state}@{state_off}, Vnew={num_vnew}@{vnew_off}, KV={num_kv}@{kv_off}, total={total}")
        return wh_off, state_off, vnew_off, kv_off, total

    def _compute_grid(self, B, H, V):
        num_v_tiles = (V + self.BV - 1) // self.BV
        if self.is_varlen:
            if self.persistent:
                import torch
                sm_count = torch.cuda.get_device_properties(0).multi_processor_count
                # Scale grid by min_occupancy: more CTAs to fill higher occupancy
                return (sm_count * self.min_occupancy, 1, 1)
            else:
                # Non-persistent: one CTA per work unit, free HW scheduling
                total_work_units = num_v_tiles * H * B
                return (total_work_units, 1, 1)
        return (num_v_tiles, H, B)

    @cute.jit
    def __call__(
        self,
        k_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        u_ptr: cute.Pointer,
        g_ptr: cute.Pointer,
        gk_ptr: cute.Pointer,
        h_out_ptr: cute.Pointer,
        v_new_ptr: cute.Pointer,
        h0_ptr: cute.Pointer,
        ht_ptr: cute.Pointer,
        cu_seqlens_ptr: cute.Pointer,
        chunk_offsets_ptr: cute.Pointer,
        workspace_ptr: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,
        use_g: Int32,
        use_gk: Int32,
        use_initial_state: Int32,
        store_final_state: Int32,
        save_v_new: Int32,
        stream,
    ):
        B, T, H, K, V = problem_size

        # For varlen: B=num_seqs, T=total_tokens, data tensors use data_B=1.
        # For non-varlen: data_B=B, NT=ceil(T/BT).
        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
            NT = total_nt  # total number of chunks across all sequences
        else:
            data_B = B
            NT = (T + self.BT - 1) // self.BT

        # ===================== GMEM layouts =====================
        # Data tensors use data_B for batch dimension (1 for varlen, B for non-varlen)
        kt_layout = cute.make_layout((K, T, (H, data_B)), stride=(1, H * K, (K, T * H * K)))
        kt = cute.make_tensor(k_ptr, kt_layout)

        w_layout = cute.make_layout((T, K, (H, data_B)), stride=(H * K, 1, (K, T * H * K)))
        w = cute.make_tensor(w_ptr, w_layout)

        u_layout = cute.make_layout((T, V, (H, data_B)), stride=(H * V, 1, (V, T * H * V)))
        u = cute.make_tensor(u_ptr, u_layout)

        v_new = cute.make_tensor(v_new_ptr, u_layout)

        # h_out: for varlen, NT=total_chunks and data_B=1; for non-varlen, NT=per-seq chunks and data_B=B
        h_out_T_layout = cute.make_layout(
            (V, K, (NT, H, data_B)),
            stride=(1, V, (H * K * V, K * V, NT * H * K * V)),
        )
        h_out_T = cute.make_tensor(h_out_ptr, h_out_T_layout)

        # h0/ht always use B=num_seqs (same for both varlen and non-varlen)
        h0_layout = cute.make_layout((K, V, (H, B)), stride=(V, 1, (K * V, H * K * V)))
        h0 = cute.make_tensor(h0_ptr, h0_layout)

        ht_T_layout = cute.make_layout((V, K, (H, B)), stride=(1, V, (K * V, H * K * V)))
        ht_T = cute.make_tensor(ht_ptr, ht_T_layout)

        gk_layout = cute.make_layout((T, K, (H, data_B)), stride=(H * K, 1, (K, T * H * K)))
        gk = cute.make_tensor(gk_ptr, gk_layout)

        # gk K-first view for TMA: (K, T, (H, data_B)) with K contiguous
        gk_K_layout = cute.make_layout((K, T, (H, data_B)), stride=(1, H * K, (K, T * H * K)))
        gk_K = cute.make_tensor(gk_ptr, gk_K_layout)

        # Transposed U view: (V, T, (H, data_B)) to match WH acc shape (M=BV, N=BT)
        u_T_layout = cute.make_layout((V, T, (H, data_B)), stride=(1, H * V, (V, T * H * V)))
        u_T = cute.make_tensor(u_ptr, u_T_layout)

        self.k_dtype = kt.element_type
        self.w_dtype = w.element_type
        self.u_dtype = u.element_type

        # ===================== MMA setup =====================
        # WH MMA: A=state(TMEM, K-major), B=W(SMEM, K-major)
        wh_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,   # A: state, K-major (required for TMEM source)
            tcgen05.OperandMajorMode.K,   # B: W, K-major (BK contiguous)
            self.acc_dtype,
            self.cta_group,
            self.wh_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,   # A operand from TMEM (zero-copy)
        )

        # KV MMA: A=v_new^T(TMEM, K-major required), B=K^T(SMEM, MN-major)
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,   # A: v_new, K-major (required for TMEM source)
            tcgen05.OperandMajorMode.MN,  # B: K^T, MN-major (BK contiguous)
            self.acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,   # A operand from TMEM (zero-copy)
        )

        # v_new TMEM layout for KV MMA A operand
        vnew_tmem_layout = sm100_utils.make_smem_layout_a(
            kv_tiled_mma, self.kv_mma_tiler, self.io_dtype, 1,
        )
        # State TMEM layout for WH MMA A operand
        state_tmem_layout = sm100_utils.make_smem_layout_a(
            wh_tiled_mma, self.wh_mma_tiler, self.io_dtype, 1,
        )

        # ===================== TMEM offsets =====================
        (self.tmem_wh_off, self.tmem_state_off, self.tmem_vnew_off, self.tmem_kv_off, self.tmem_total) = self._plan_tmem_offsets(
            wh_tiled_mma, self.wh_mma_tiler,
            kv_tiled_mma, self.kv_mma_tiler,
            state_tmem_layout, vnew_tmem_layout,
            self.acc_stage,
        )

        # ===================== SMEM layouts =====================
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # W as B operand of WH MMA
        w_smem_staged = sm100_utils.make_smem_layout_b(
            wh_tiled_mma, self.wh_mma_tiler, self.io_dtype, self.w_stage,
        )
        # K^T as B operand of KV MMA
        kt_smem_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma, self.kv_mma_tiler, self.io_dtype, self.k_stage,
        )
        # State A operand now from TMEM (no SMEM layout needed)
        # v_new A operand now from TMEM (no SMEM layout needed for MMA path)
        # h_out epilogue for TMA store
        # COL_MAJOR for (BV, BK): BV contiguous → matches V stride 1 in h_out_T GMEM
        h_out_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BK),  # (64, 128)
            self.h_out_stage,
        )
        # U SMEM for TMA load — COL_MAJOR (BV, BT), BV contiguous matches u_T GMEM
        u_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),  # (64, 64)
            self.u_stage,
        )
        # v_new store SMEM — COL_MAJOR (BV, BT) for TMA S2G
        vnew_store_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),  # (64, 64)
            self.vnew_store_stage,
        )

        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (wh_tiled_mma.thr_id.shape,),
        )

        # ===================== TMA descriptors =====================
        w_smem = cute.select(w_smem_staged, mode=[0, 1, 2])
        tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, w, w_smem, self.wh_mma_tiler, wh_tiled_mma, cluster_layout.shape,
        )

        kt_smem = cute.select(kt_smem_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, kt, kt_smem, self.kv_mma_tiler, kv_tiled_mma, cluster_layout.shape,
        )

        h_epi_smem = cute.select(h_out_epi_staged, mode=[0, 1])
        tma_atom_h_out, tma_tensor_h_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op, h_out_T, h_epi_smem, (self.BV, self.BK),
        )

        tma_atom_ht, tma_tensor_ht = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op, ht_T, h_epi_smem, (self.BV, self.BK),
        )

        # TMA descriptor for U load (G2S) — non-MMA operand
        u_smem = cute.select(u_epi_staged, mode=[0, 1])
        tma_atom_u, tma_tensor_u = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op, u_T, u_smem, (self.BV, self.BT),
        )

        # v_new transposed GMEM view: (V, T, (H, data_B)) for TMA store
        v_new_T_layout = cute.make_layout(
            (V, T, (H, data_B)), stride=(1, H * V, (V, T * H * V)),
        )
        v_new_T = cute.make_tensor(v_new_ptr, v_new_T_layout)

        # cu_seqlens and chunk_offsets tensors for varlen
        cu_seqlens = cute.make_tensor(
            cu_seqlens_ptr, cute.make_layout((B + 1,))
        )
        chunk_offsets = cute.make_tensor(
            chunk_offsets_ptr, cute.make_layout((B + 1,))
        )

        # TMA descriptor for v_new store (S2G) — used only for non-varlen
        vnew_store_smem = cute.select(vnew_store_epi_staged, mode=[0, 1])
        tma_atom_vnew_st, tma_tensor_vnew_st = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op, v_new_T, vnew_store_smem, (self.BV, self.BT),
        )

        # Direct GMEM write TiledCopy for v_new — used for varlen mode
        # (avoids TMA store tail fixup descriptor corruption bug)
        # sVnew_store is COL_MAJOR (BV, BT): BV contiguous (mode 0)
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.io_dtype.width  # 8 for bf16
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # Thread layout for 32 store-warp threads over (BV, BT) tile:
        # - BV (mode 0, contiguous): 8 thread-groups × 8 values = 64
        # - BT (mode 1): 4 threads × 1 value = 4 per iteration, 16 repeats
        vnew_thr_dim0 = self.BV // async_copy_elems  # 8
        vnew_thr_dim1 = self.threads_per_warp // vnew_thr_dim0  # 4
        assert self.BT % vnew_thr_dim1 == 0
        vnew_thr_layout = cute.make_ordered_layout(
            (vnew_thr_dim0, vnew_thr_dim1), order=(0, 1),
        )  # (8, 4), BV-groups faster → coalesced GMEM writes
        vnew_val_layout = cute.make_layout((async_copy_elems, 1))  # (8, 1)
        gmem_tiled_copy_vnew = cute.make_tiled_copy_tv(
            atom_universal_copy, vnew_thr_layout, vnew_val_layout,
        )

        # TMA descriptor for gk load (G2S) — 2D tile (BK, 1) along K dimension
        gk_smem_2d = cute.make_layout((self.BK, 1))
        tma_atom_gk, tma_tensor_gk = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op, gk_K, gk_smem_2d, (self.BK, 1),
        )

        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem)
        self.tma_kt_bytes = cute.size_in_bytes(self.io_dtype, kt_smem)
        self.tma_u_bytes = cute.size_in_bytes(self.io_dtype, u_smem)
        self.tma_gk_bytes = self.BK * 4  # BK Float32 elements

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            load_w_mbar: cute.struct.MemRange[Int64, self.w_stage * 2]
            load_kt_mbar: cute.struct.MemRange[Int64, self.k_stage * 2]
            load_u_mbar: cute.struct.MemRange[Int64, self.u_stage * 2]         # Load→CUDA: sU ready
            load_gk_mbar: cute.struct.MemRange[Int64, self.gk_stage * 2]       # Load→CUDA: sGK ready
            state_tmem_mbar: cute.struct.MemRange[Int64, 1 * 2]       # CUDA→MMA: state TMEM ready
            wh_done_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # MMA→CUDA: WH done
            vnew_smem_mbar: cute.struct.MemRange[Int64, 1 * 2]        # CUDA→MMA: sVnew ready
            kv_done_mbar: cute.struct.MemRange[Int64, 1 * 2]          # MMA→CUDA: KV done
            h_out_mbar: cute.struct.MemRange[Int64, self.h_out_stage * 2]  # CUDA→Store: sH_epi ready
            vnew_store_mbar: cute.struct.MemRange[Int64, self.vnew_store_stage * 2]  # CUDA→Store: sVnew_store ready
            tmem_holding_buf: Int32
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_staged)],
                self.buffer_align_bytes,
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(kt_smem_staged)],
                self.buffer_align_bytes,
            ]
            # sState removed: state now goes through TMEM, not SMEM
            # sVnew removed: v_new now goes through TMEM, not SMEM
            sH_epi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(h_out_epi_staged)],
                self.buffer_align_bytes,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(u_epi_staged)],
                self.buffer_align_bytes,
            ]
            sVnew_store: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vnew_store_epi_staged)],
                self.buffer_align_bytes,
            ]
            sGK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK * 2],
                128,
            ]

        self.shared_storage = SharedStorage
        self.grid = self._compute_grid(B, H, V)

        self.kernel(
            wh_tiled_mma, kv_tiled_mma,
            tma_atom_w, tma_tensor_w,
            tma_atom_kt, tma_tensor_kt,
            tma_atom_h_out, tma_tensor_h_out,
            tma_atom_ht, tma_tensor_ht,
            tma_atom_u, tma_tensor_u,
            tma_atom_vnew_st, tma_tensor_vnew_st,
            tma_atom_gk, tma_tensor_gk,
            gmem_tiled_copy_vnew,
            h0, u, u_T, h_out_T, v_new,
            w_smem_staged, kt_smem_staged,
            state_tmem_layout, vnew_tmem_layout,
            h_out_epi_staged,
            u_epi_staged, vnew_store_epi_staged,
            cu_seqlens, chunk_offsets,
            workspace_ptr,
            problem_size,
            use_gk, use_initial_state, store_final_state, save_v_new,
        ).launch(
            grid=self.grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=self.min_occupancy,
        )

    @cute.kernel
    def kernel(
        self,
        wh_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        tma_tensor_w: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        tma_tensor_kt: cute.Tensor,
        tma_atom_h_out: cute.CopyAtom,
        tma_tensor_h_out: cute.Tensor,
        tma_atom_ht: cute.CopyAtom,
        tma_tensor_ht: cute.Tensor,
        tma_atom_u: cute.CopyAtom,
        tma_tensor_u: cute.Tensor,
        tma_atom_vnew_st: cute.CopyAtom,
        tma_tensor_vnew_st: cute.Tensor,
        tma_atom_gk: cute.CopyAtom,
        tma_tensor_gk: cute.Tensor,
        gmem_tiled_copy_vnew: cute.TiledCopy,
        h0: cute.Tensor,
        u_tensor: cute.Tensor,
        u_T_tensor: cute.Tensor,
        h_out_T_tensor: cute.Tensor,
        v_new_tensor: cute.Tensor,
        w_smem_staged: cute.ComposedLayout,
        kt_smem_staged: cute.ComposedLayout,
        state_tmem_layout: cute.ComposedLayout,
        vnew_tmem_layout: cute.ComposedLayout,
        h_out_epi_staged: cute.ComposedLayout,
        u_epi_staged: cute.ComposedLayout,
        vnew_store_epi_staged: cute.ComposedLayout,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        workspace_iter: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        use_gk: Int32,
        use_initial_state: Int32,
        store_final_state: Int32,
        save_v_new: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_u)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_gk)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sGK_smem = storage.sGK.get_tensor(cute.make_layout((self.BK, self.gk_stage)))
        # 3D SMEM view for _epilog_partition in Load warp: (BK, 1, gk_stage)
        sGK_3d = storage.sGK.get_tensor(cute.make_layout(
            (self.BK, 1, self.gk_stage), stride=(1, self.BK, self.BK)))

        # ===================== Pipelines =====================
        load_w_P, load_w_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.load_w_mbar.data_ptr(),
        ).make_participants()

        load_kt_P, load_kt_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_kt_bytes,
            barrier_storage=storage.load_kt_mbar.data_ptr(),
        ).make_participants()

        state_smem_P, state_smem_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.state_tmem_mbar.data_ptr(),
        ).make_participants()

        wh_done_P, wh_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.wh_done_mbar.data_ptr(),
        ).make_participants()

        vnew_smem_P, vnew_smem_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.vnew_smem_mbar.data_ptr(),
        ).make_participants()

        kv_done_P, kv_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.kv_done_mbar.data_ptr(),
        ).make_participants()

        h_out_P, h_out_C = pipeline.PipelineAsync.create(
            num_stages=self.h_out_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.h_out_mbar.data_ptr(),
        ).make_participants()

        load_u_P, load_u_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.u_stage,
            producer_group=make_thread_cooperative_group(
                len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.cuda_warp_ids)),
            tx_count=self.tma_u_bytes,
            barrier_storage=storage.load_u_mbar.data_ptr(),
        ).make_participants()

        vnew_store_P, vnew_store_C = pipeline.PipelineAsync.create(
            num_stages=self.vnew_store_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.vnew_store_mbar.data_ptr(),
        ).make_participants()

        load_gk_P, load_gk_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.gk_stage,
            producer_group=make_thread_cooperative_group(
                len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.cuda_warp_ids)),
            tx_count=self.tma_gk_bytes,
            barrier_storage=storage.load_gk_mbar.data_ptr(),
        ).make_participants()

        # ===================== TMEM =====================
        tmem_alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # ===================== SMEM views =====================
        sW = storage.sW.get_tensor(w_smem_staged.outer, swizzle=w_smem_staged.inner)
        sKt = storage.sKt.get_tensor(kt_smem_staged.outer, swizzle=kt_smem_staged.inner)
        sH_epi = storage.sH_epi.get_tensor(h_out_epi_staged.outer, swizzle=h_out_epi_staged.inner)
        sU_epi = storage.sU.get_tensor(u_epi_staged.outer, swizzle=u_epi_staged.inner)
        sVnew_store_epi = storage.sVnew_store.get_tensor(
            vnew_store_epi_staged.outer, swizzle=vnew_store_epi_staged.inner,
        )

        # ===================== MMA fragments =====================
        # WH MMA: A=state(TMEM), B=sW, acc=WH TMEM
        tCrState_fake = wh_tiled_mma.make_fragment_A(state_tmem_layout.outer.shape)
        tCrState = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_off, dtype=tCrState_fake.element_type),
            tCrState_fake.layout,
        )
        tCrW = wh_tiled_mma.make_fragment_B(sW)
        wh_shape = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH_fake = wh_tiled_mma.make_fragment_C(cute.append(wh_shape, self.acc_stage))
        tCtAccWH = cute.make_tensor(tmem_ptr + self.tmem_wh_off, tCtAccWH_fake.layout)

        # KV MMA: A=v_new(TMEM), B=sKt, acc=KV TMEM
        # Create v_new TMEM A fragment (Mamba2-style: get layout from fake, bind TMEM ptr)
        tCrVnew_fake = kv_tiled_mma.make_fragment_A(vnew_tmem_layout.outer.shape)
        tCrVnew = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_vnew_off, dtype=tCrVnew_fake.element_type),
            tCrVnew_fake.layout,
        )
        tCrKt = kv_tiled_mma.make_fragment_B(sKt)
        kv_shape = kv_tiled_mma.partition_shape_C(self.kv_mma_tiler[:2])
        tCtAccKV_fake = kv_tiled_mma.make_fragment_C(cute.append(kv_shape, 1))
        tCtAccKV = cute.make_tensor(tmem_ptr + self.tmem_kv_off, tCtAccKV_fake.layout)

        # ===================== Block indices =====================
        B, T, H, K, V = problem_size
        BT = self.BT

        if cutlass.const_expr(self.is_varlen):
            # 1D grid work decode: persistent (grid=SM_count, multi-iter) or
            # non-persistent (grid=total_work_units, single iter per CTA)
            block_idx_x = cute.arch.block_idx()[0]
            grid_dim_x = cute.arch.grid_dim()[0]
            num_v_tiles = (V + self.BV - 1) // self.BV
            total_work_units = num_v_tiles * H * B
            num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x
            # Pre-initialize variables reassigned inside persistent loop (CuTe DSL requirement)
            v_tile_idx = Int32(0)
            hidx = Int32(0)
            bidx = Int32(0)
            tok_offset = Int32(0)
            seq_len = Int32(0)
            NT = Int32(0)
            data_bidx = Int32(0)
            chunk_off = Int32(0)
        else:
            (v_tile_idx, hidx, bidx) = cute.arch.block_idx()
            tok_offset = Int32(0)
            seq_len = T
            NT = (T + BT - 1) // BT
            data_bidx = bidx
            chunk_off = Int32(0)
            num_iters = Int32(1)

        # =========================================================================
        # LOAD WARP
        # =========================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # --- Persistent work decode ---
                if cutlass.const_expr(self.is_varlen):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    v_tile_idx = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    hidx = temp_work % H
                    bidx = temp_work // H
                    tok_offset = cu_seqlens[bidx]
                    seq_len = cu_seqlens[bidx + 1] - tok_offset
                    NT = (seq_len + BT - 1) // BT
                    data_bidx = Int32(0)

                # Apply domain_offset for varlen TMA tensors (shift T dim by tok_offset)
                if cutlass.const_expr(self.is_varlen):
                    tma_tensor_w_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_w)
                    tma_tensor_kt_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_kt)
                    tma_tensor_u_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_u)
                    tma_tensor_gk_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_gk)
                else:
                    tma_tensor_w_v = tma_tensor_w
                    tma_tensor_kt_v = tma_tensor_kt
                    tma_tensor_u_v = tma_tensor_u
                    tma_tensor_gk_v = tma_tensor_gk

                tWsW, tWgW = self._tma_partition_B(
                    tma_atom_w, tma_tensor_w_v, sW, self.wh_mma_tiler, wh_tiled_mma, data_bidx, hidx,
                )
                tKsK, tKgK = self._tma_partition_B(
                    tma_atom_kt, tma_tensor_kt_v, sKt, self.kv_mma_tiler, kv_tiled_mma, data_bidx, hidx,
                )

                # U TMA load partition (non-MMA, epilog-style)
                gU_ld = tma_tensor_u_v[None, None, (hidx, data_bidx)]
                _, bSG_sU, bSG_gU = self._epilog_partition(
                    tma_atom_u, gU_ld, (self.BV, self.BT), sU_epi,
                )

                # gk TMA load partition: gk_K shape (K, T, (H, data_B)), load (BK, 1) per timestep
                gGK_ld = tma_tensor_gk_v[None, None, (hidx, data_bidx)]  # (K, T)
                _, bSG_sGK, bSG_gGK = self._epilog_partition(
                    tma_atom_gk, gGK_ld, (self.BK, 1), sGK_3d,
                )

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    w_h = load_w_P.acquire_and_advance()
                    cute.copy(atom=tma_atom_w, src=tWgW[None, chunk_idx, 0],
                              dst=tWsW[None, w_h.index], tma_bar_ptr=w_h.barrier)

                    kt_h = load_kt_P.acquire_and_advance()
                    cute.copy(atom=tma_atom_kt, src=tKgK[None, 0, chunk_idx],
                              dst=tKsK[None, kt_h.index], tma_bar_ptr=kt_h.barrier)

                    u_h = load_u_P.acquire_and_advance()
                    cute.copy(atom=tma_atom_u,
                              src=bSG_gU[(None, v_tile_idx, chunk_idx)],
                              dst=bSG_sU[None, u_h.index],
                              tma_bar_ptr=u_h.barrier)

                    # TMA load gk for this chunk (BK Float32 values)
                    if use_gk:
                        # For tail chunk in varlen, use last valid position
                        gk_t_idx = chunk_idx * self.BT + self.BT - 1
                        if cutlass.const_expr(self.is_varlen):
                            remaining = seq_len - chunk_idx * self.BT
                            if remaining < self.BT:
                                gk_t_idx = seq_len - 1
                        gk_h = load_gk_P.acquire_and_advance()
                        cute.copy(atom=tma_atom_gk,
                                  src=bSG_gGK[(None, 0, gk_t_idx)],
                                  dst=bSG_sGK[None, gk_h.index],
                                  tma_bar_ptr=gk_h.barrier)

        # =========================================================================
        # MMA WARP
        # =========================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # --- Persistent work decode (MMA only needs NT) ---
                if cutlass.const_expr(self.is_varlen):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    bidx_mma = (work_idx // num_v_tiles) // H
                    tok_off_mma = cu_seqlens[bidx_mma]
                    NT = (cu_seqlens[bidx_mma + 1] - tok_off_mma + BT - 1) // BT

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    # --- WH MMA: state(SMEM) × W(SMEM) → acc_wh ---
                    state_h = state_smem_C.wait_and_advance()
                    w_h = load_w_C.wait_and_advance()

                    wh_h = wh_done_P.acquire_and_advance()
                    for kp in cutlass.range(cute.size(tCrW, mode=[2]), unroll_full=True):
                        wh_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(wh_tiled_mma,
                                  tCtAccWH[None, None, None, wh_h.index],
                                  tCrState[None, None, kp, state_h.index],
                                  tCrW[None, None, kp, w_h.index],
                                  tCtAccWH[None, None, None, wh_h.index])
                    wh_h.commit()
                    w_h.release()
                    state_h.release()

                    # --- KV MMA: v_new(TMEM) × K^T(SMEM) → update (ACCUMULATE=False always) ---
                    vnew_h = vnew_smem_C.wait_and_advance()
                    kt_h = load_kt_C.wait_and_advance()

                    kv_h = kv_done_P.acquire_and_advance()
                    for kp in cutlass.range(cute.size(tCrKt, mode=[2]), unroll_full=True):
                        # Always ACCUMULATE=False: we only compute the update term
                        kv_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(kv_tiled_mma,
                                  tCtAccKV[None, None, None, 0],
                                  tCrVnew[None, None, kp, vnew_h.index],
                                  tCrKt[None, None, kp, kt_h.index],
                                  tCtAccKV[None, None, None, 0])
                    kv_h.commit()
                    kt_h.release()
                    vnew_h.release()

        # =========================================================================
        # CUDA CORE WARPS
        # =========================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))

            # ----- T2R setup for KV acc (BV=64, BK=128 FP32) -----
            t2r_atom_kv = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccKV_flat = tCtAccKV[((None, None), 0, 0, None)]
            fake_sKV = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.kv_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_kv = tcgen05.make_tmem_copy(t2r_atom_kv, tCtAccKV_flat[(None, None, 0)])
            thr_t2r_kv = tiled_t2r_kv.get_slice(local_tidx)
            tTR_tKV = thr_t2r_kv.partition_S(tCtAccKV_flat)
            tTR_sKV = thr_t2r_kv.partition_D(fake_sKV)
            # h state in registers (persistent across chunks)
            tTR_rKV = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)

            # ----- T2R setup for WH acc (BV=64, BT=64 FP32) -----
            t2r_atom_wh = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccWH_flat = tCtAccWH[((None, None), 0, 0, None)]
            fake_sWH = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.wh_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_wh = tcgen05.make_tmem_copy(t2r_atom_wh, tCtAccWH_flat[(None, None, 0)])
            thr_t2r_wh = tiled_t2r_wh.get_slice(local_tidx)
            tTR_tWH = thr_t2r_wh.partition_S(tCtAccWH_flat)
            tTR_sWH = thr_t2r_wh.partition_D(fake_sWH)

            # ----- R2T: h state regs → TMEM for WH MMA A operand -----
            copy_atom_r2t_state = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(16), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_state = tcgen05.make_tmem_copy(copy_atom_r2t_state, tCrState)
            thr_r2t_state = tiled_r2t_state.get_slice(local_tidx)
            r2t_state_shape = cute.slice_(thr_r2t_state.partition_S(tCrState).shape, (None, None, None, None, 0))
            tRT_tState = thr_r2t_state.partition_D(tCrState)

            # ----- R2S: KV T2R regs → sH_epi (COL_MAJOR, BV×BK) -----
            r2s_atom_h = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r_kv,
            )
            tiled_r2s_h = cute.make_tiled_copy_D(r2s_atom_h, tiled_t2r_kv)
            thr_r2s_h = tiled_r2s_h.get_slice(local_tidx)
            tRS_sH = thr_r2s_h.partition_D(sH_epi)

            # ----- R2S: WH T2R regs → sVnew_store_epi (COL_MAJOR, BV×BT) for TMA store -----
            r2s_atom_vnew = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r_wh,
            )
            tiled_r2s_vnew = cute.make_tiled_copy_D(r2s_atom_vnew, tiled_t2r_wh)
            thr_r2s_vnew = tiled_r2s_vnew.get_slice(local_tidx)
            tRS_sVnew_store = thr_r2s_vnew.partition_D(sVnew_store_epi)

            # ----- R2T: v_new regs → TMEM for KV MMA A operand -----
            copy_atom_r2t_vnew = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_vnew = tcgen05.make_tmem_copy(copy_atom_r2t_vnew, tCrVnew)
            thr_r2t_vnew = tiled_r2t_vnew.get_slice(local_tidx)
            r2t_vnew_shape = cute.slice_(thr_r2t_vnew.partition_S(tCrVnew).shape, (None, None, None, None, 0))
            tRT_tVnew = thr_r2t_vnew.partition_D(tCrVnew)

            # ----- Identity tensor for WH tile (BV, BT) → v_new coords -----
            vnew_tile = cute.dice(self.wh_mma_tiler, (1, 1, None))  # (BV, BT)
            cM_vnew = cute.make_identity_tensor(vnew_tile)
            tTR_cM = thr_t2r_wh.partition_D(cM_vnew)

            # ----- Identity tensor for KV tile (BV, BK) → h coords -----
            h_tile = cute.dice(self.kv_mma_tiler, (1, 1, None))  # (BV, BK)
            cM_h = cute.make_identity_tensor(h_tile)
            tTR_cM_h = thr_t2r_kv.partition_D(cM_h)

            # ===== Persistent outer loop =====
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # --- Persistent work decode ---
                if cutlass.const_expr(self.is_varlen):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    v_tile_idx = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    hidx = temp_work % H
                    bidx = temp_work // H
                    tok_offset = cu_seqlens[bidx]
                    seq_len = cu_seqlens[bidx + 1] - tok_offset
                    NT = (seq_len + BT - 1) // BT

                # ===== Initialize h in registers =====
                if use_initial_state:
                    # Load h0 from GMEM into registers using identity tensor mapping
                    gH0 = h0[None, None, (hidx, bidx)]  # (K, V)
                    for ei in cutlass.range_constexpr(cute.size(tTR_rKV)):
                        v_coord, k_coord = tTR_cM_h[ei]
                        tTR_rKV[ei] = gH0[k_coord, v_coord + v_tile_idx * self.BV].to(self.acc_dtype)
                else:
                    for ei in cutlass.range_constexpr(cute.size(tTR_rKV)):
                        tTR_rKV[ei] = Float32(0.0)

                # ===== Main loop (gk-only optimized pipeline) =====
                # Pipeline: Phase1(R2T+R2S+gk_decay)→WH MMA→Phase2(v_new)→KV MMA→Phase4(h update)
                # gk decay moved to Phase1 to overlap with longer WH MMA (K=128) window

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    # ========================================
                    # Phase 1: Publish h for WH MMA + h_out store
                    # ========================================
                    # Declare per-phase register tensors at point of use
                    # to help compiler see non-overlapping lifetimes
                    tTR_rKV_bf16 = cute.make_rmem_tensor(tTR_rKV.shape, self.io_dtype)
                    tRT_rState = cute.make_rmem_tensor(r2t_state_shape, self.io_dtype)
                    h_vec = tTR_rKV.load()
                    tTR_rKV_bf16.store(h_vec.to(self.io_dtype))

                    # R2T h state → TMEM (triggers WH MMA — zero-copy A operand)
                    tRT_rState.store(h_vec.to(self.io_dtype))
                    state_h = state_smem_P.acquire_and_advance()
                    cute.copy(tiled_r2t_state, tRT_rState, tRT_tState[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    state_h.commit()  # WH MMA can start now!

                    # R2S to sH_epi (overlaps with WH MMA)
                    tRS_rH = tiled_r2s_h.retile(tTR_rKV_bf16)
                    h_handle = h_out_P.acquire_and_advance()
                    cute.copy(tiled_r2s_h, tRS_rH, tRS_sH[(None, None, None, h_handle.index)])
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    h_handle.commit()

                    # gk decay: h *= exp(gk) — cooperative precomputation
                    # 128 CUDA threads cooperatively compute 128 gk_scale values (1 per K position)
                    # then each thread applies only SMEM reads (no redundant exp2/SFU ops)
                    if use_gk:
                        gk_h = load_gk_C.wait_and_advance()
                        # Step 1: Each CUDA thread (tidx 0-127) computes one exp2 and overwrites sGK in-place
                        gk_raw = sGK_smem[(tidx, gk_h.index)]
                        sGK_smem[(tidx, gk_h.index)] = cute.exp2(gk_raw * INV_LN2)
                        # Step 2: Sync all 4 CUDA warps so all 128 scales are visible
                        self.gk_precompute_bar.arrive_and_wait()
                        # Step 3: Apply precomputed scales (SMEM read only, no exp2)
                        for ei in cutlass.range_constexpr(cute.size(tTR_rKV)):
                            v_coord, k_coord = tTR_cM_h[ei]
                            tTR_rKV[ei] = tTR_rKV[ei] * sGK_smem[(k_coord, gk_h.index)]
                        gk_h.release()

                    # ========================================
                    # Phase 2: v_new from WH result → triggers KV MMA
                    # ========================================
                    wh_h = wh_done_C.wait_and_advance()
                    tTR_rWH = cute.make_rmem_tensor(tTR_sWH.shape, self.acc_dtype)
                    cute.copy(tiled_t2r_wh, tTR_tWH[(None, None, None, wh_h.index)], tTR_rWH)
                    cute.arch.fence_view_async_tmem_load()
                    wh_h.release()

                    # Inline U-load + v_new = u - WH (no g-scaling needed)
                    u_handle = load_u_C.wait_and_advance()
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        v_coord, t_coord = tTR_cM[ei]
                        u_val = sU_epi[(v_coord, t_coord, u_handle.index)].to(self.acc_dtype)
                        tTR_rWH[ei] = u_val - tTR_rWH[ei]
                    u_handle.release()

                    # Zero v_new for positions beyond sequence boundary (varlen tail chunk)
                    if cutlass.const_expr(self.is_varlen):
                        valid_len_chunk = seq_len - chunk_idx * self.BT
                        if valid_len_chunk < self.BT:
                            for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                                v_coord, t_coord = tTR_cM[ei]
                                if t_coord >= valid_len_chunk:
                                    tTR_rWH[ei] = Float32(0.0)

                    # Prepare bf16 v_new for both R2T and R2S
                    tTR_rVnew_bf16 = cute.make_rmem_tensor(tTR_rWH.shape, self.io_dtype)
                    tTR_rVnew_bf16.store(tTR_rWH.load().to(self.io_dtype))

                    # R2T v_new → TMEM FIRST (triggers KV MMA — zero-copy A operand)
                    tRT_rVnew = cute.make_rmem_tensor(r2t_vnew_shape, self.io_dtype)
                    tRT_rVnew.store(tTR_rWH.load().to(self.io_dtype))
                    vnew_h = vnew_smem_P.acquire_and_advance()
                    cute.copy(tiled_r2t_vnew, tRT_rVnew, tRT_tVnew[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    vnew_h.commit()  # KV MMA starts now!

                    # Save v_new to SMEM for TMA store (overlaps with KV MMA)
                    if save_v_new:
                        tRS_rVnew_st = tiled_r2s_vnew.retile(tTR_rVnew_bf16)
                        vnew_st_h = vnew_store_P.acquire_and_advance()
                        cute.copy(tiled_r2s_vnew, tRS_rVnew_st,
                                  tRS_sVnew_store[(None, None, None, vnew_st_h.index)])
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        vnew_st_h.commit()

                    # ========================================
                    # Phase 4: KV update → h
                    # ========================================
                    kv_h = kv_done_C.wait_and_advance()
                    tTR_rUpdate = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)
                    cute.copy(tiled_t2r_kv, tTR_tKV[(None, None, None, 0)], tTR_rUpdate)
                    cute.arch.fence_view_async_tmem_load()
                    kv_h.release()

                    h_vec = tTR_rKV.load()
                    update_vec = tTR_rUpdate.load()
                    tTR_rKV.store(h_vec + update_vec)

                # ===== After main loop: store final state ht =====
                if store_final_state:
                    tTR_rKV_bf16 = cute.make_rmem_tensor(tTR_rKV.shape, self.io_dtype)
                    h_vec = tTR_rKV.load()
                    tTR_rKV_bf16.store(h_vec.to(self.io_dtype))
                    tRS_rH = tiled_r2s_h.retile(tTR_rKV_bf16)
                    h_handle = h_out_P.acquire_and_advance()
                    cute.copy(tiled_r2s_h, tRS_rH, tRS_sH[(None, None, None, h_handle.index)])
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    h_handle.commit()

        # =========================================================================
        # STORE WARP
        # =========================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h_out)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_ht)
            if cutlass.const_expr(not self.is_varlen):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vnew_st)

            # For varlen: prepare direct GMEM write infrastructure for v_new
            # Store warp local thread index (0..31)
            store_local_tidx = tidx - self.store_warp_id * self.threads_per_warp

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # --- Persistent work decode ---
                if cutlass.const_expr(self.is_varlen):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    v_tile_idx = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    hidx = temp_work % H
                    bidx = temp_work // H
                    tok_offset = cu_seqlens[bidx]
                    seq_len = cu_seqlens[bidx + 1] - tok_offset
                    NT = (seq_len + BT - 1) // BT
                    data_bidx = Int32(0)
                    chunk_off = chunk_offsets[bidx]

                # Apply domain_offset for varlen store TMA tensors
                if cutlass.const_expr(self.is_varlen):
                    tma_tensor_h_out_v = cute.domain_offset(
                        (0, 0, (chunk_off, 0, 0)), tma_tensor_h_out)
                else:
                    tma_tensor_h_out_v = tma_tensor_h_out

                gH_st = tma_tensor_h_out_v[None, None, (None, hidx, data_bidx)]
                tma_h_st, bSG_sH, bSG_gH = self._epilog_partition(
                    tma_atom_h_out, gH_st, (self.BV, self.BK), sH_epi,
                )

                # ht uses B=num_seqs always, bidx is correct
                gHt_st = tma_tensor_ht[None, None, (hidx, bidx)]
                tma_ht_st, bSG_sHt, bSG_gHt = self._epilog_partition(
                    tma_atom_ht, gHt_st, (self.BV, self.BK), sH_epi,
                )

                # v_new store partition: TMA for non-varlen, direct GMEM for varlen
                if cutlass.const_expr(not self.is_varlen):
                    tma_tensor_vnew_v = tma_tensor_vnew_st
                    gVnew_st = tma_tensor_vnew_v[None, None, (hidx, data_bidx)]
                    tma_vnew_st, bSG_sVnew_st, bSG_gVnew_st = self._epilog_partition(
                        tma_atom_vnew_st, gVnew_st, (self.BV, self.BT), sVnew_store_epi,
                    )

                for chunk_idx in cutlass.range(0, NT, unroll=0):
                    h_handle = h_out_C.wait_and_advance()

                    cute.copy(tma_h_st, bSG_sH[None, h_handle.index],
                              bSG_gH[(None, v_tile_idx, 0, chunk_idx)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

                    h_handle.release()

                    # v_new store
                    if save_v_new:
                        vnew_handle = vnew_store_C.wait_and_advance()
                        if cutlass.const_expr(self.is_varlen):
                            # Direct SMEM → REG → GMEM write with per-row bounds check
                            remaining = seq_len - chunk_idx * self.BT

                            # Get SMEM stage view (BV, BT)
                            sVnew_stage = sVnew_store_epi[None, None, vnew_handle.index]

                            # Partition SMEM source with gmem_tiled_copy_vnew
                            gmem_thr_copy = gmem_tiled_copy_vnew.get_slice(store_local_tidx)
                            tOsVnew = gmem_thr_copy.partition_S(sVnew_stage)

                            # Identity tensor for coordinate tracking
                            cVnew = cute.make_identity_tensor((self.BV, self.BT))
                            tOcVnew = gmem_thr_copy.partition_S(cVnew)

                            # SMEM → REG (handles swizzle via autovec_copy)
                            tOrVnew = cute.make_fragment_like(tOsVnew, self.io_dtype)
                            cute.autovec_copy(tOsVnew, tOrVnew)

                            # Construct GMEM tile for this chunk
                            # v_new layout: (T, V, (H, 1)) stride (H*V, 1, (V, T*H*V))
                            # For (BV, BT) tile: BV contiguous (stride=1), BT stride = H*V
                            vnew_chunk_raw = (v_new_tensor.iterator
                                + (tok_offset + chunk_idx * BT) * H * V
                                + hidx * V
                                + v_tile_idx * self.BV)
                            # Re-annotate pointer as 128-bit (16-byte) aligned
                            # (safe: torch tensors are ≥256-byte aligned, offsets are
                            #  multiples of V≥128 which ≥ 8 bf16 = 16 bytes)
                            vnew_chunk_ptr = cute.make_ptr(
                                self.io_dtype, vnew_chunk_raw.toint(),
                                cute.AddressSpace.gmem, assumed_align=16,
                            )
                            # Assume non-contiguous stride divisible by 8 bf16 = 128 bits
                            vnew_stride_t = cute.assume(
                                H * V, divby=128 // self.io_dtype.width,
                            )
                            gVnew_chunk = cute.make_tensor(
                                vnew_chunk_ptr,
                                cute.make_layout(
                                    (self.BV, self.BT), stride=(1, vnew_stride_t),
                                ),
                            )

                            # Partition GMEM destination
                            tOgVnew = gmem_thr_copy.partition_D(gVnew_chunk)

                            # REG → GMEM with per-BT-row bounds check
                            for rest_bt in cutlass.range_constexpr(
                                cute.size(tOrVnew.shape[2])
                            ):
                                # BT coordinate for this thread at this rest iteration
                                bt_coord = tOcVnew[0, 0, rest_bt][1]
                                if bt_coord < remaining:
                                    cute.copy(
                                        gmem_tiled_copy_vnew,
                                        tOrVnew[None, None, rest_bt],
                                        tOgVnew[None, None, rest_bt],
                                    )
                        else:
                            cute.copy(tma_vnew_st,
                                      bSG_sVnew_st[None, vnew_handle.index],
                                      bSG_gVnew_st[(None, v_tile_idx, chunk_idx)])
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        vnew_handle.release()

                # Store final state ht
                if store_final_state:
                    h_handle = h_out_C.wait_and_advance()
                    cute.copy(tma_ht_st, bSG_sHt[None, h_handle.index],
                              bSG_gHt[(None, v_tile_idx, 0)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    h_handle.release()

        # =========================================================================
        # EMPTY WARP
        # =========================================================================
        elif warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr)

    @cute.jit
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        """Partition B operand tensors for TMA copy."""
        coord = (0, None, None)
        gX = cute.local_tile(
            tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx))
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1),
            cute.group_modes(smem, 0, 3), cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _epilog_partition(self, atom, gC_mnl, epi_tile, sC):
        """Partition for epilogue TMA store."""
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), sC_g, gC_g,
        )
        return atom, bSG_sC, bSG_gC


# ===================== Reference implementations =====================

def reference_chunk_delta_rule_fwd_h(k, w, u, g=None, gk=None, h0=None, chunk_size=64):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    h_out = torch.zeros(B, NT, H, K, V, device=k.device, dtype=torch.bfloat16)
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()
    h_after = []
    for t in range(NT):
        s, e = t * BT, min((t + 1) * BT, T)
        h_out[:, t] = h.to(torch.bfloat16)
        wc = w[:, s:e].permute(0, 2, 1, 3).float()
        kc = k[:, s:e].permute(0, 2, 1, 3).float()
        uc = u[:, s:e].permute(0, 2, 1, 3).float()
        wh = torch.matmul(wc, h)
        vnc = uc - wh
        if g is not None:
            gc = g[:, s:e].permute(0, 2, 1).float()
            gl = gc[:, :, -1:].float()
            vnc = vnc * torch.exp(gl - gc).unsqueeze(-1)
        v_new_out[:, s:e] = vnc.permute(0, 2, 1, 3).to(torch.bfloat16)
        if g is not None:
            gls = gc[:, :, -1].float()
            h = h * torch.exp(gls).unsqueeze(-1).unsqueeze(-1)
        if gk is not None:
            gkc = gk[:, s:e].permute(0, 2, 1, 3).float()
            gkl = gkc[:, :, -1, :].float()
            h = h * torch.exp(gkl).unsqueeze(-1)
        h = h + torch.matmul(kc.transpose(-2, -1), vnc)
        h_after.append(h[0, 0].to(torch.bfloat16).clone())
    return h_out, v_new_out, h_after


def reference_bf16_roundtrip(k, w, u, g=None, gk=None, h0=None, chunk_size=64):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=torch.bfloat16)
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()
    h_after = []
    for t in range(NT):
        s, e = t * BT, min((t + 1) * BT, T)
        wc = w[:, s:e].permute(0, 2, 1, 3).float()
        kc = k[:, s:e].permute(0, 2, 1, 3).float()
        uc = u[:, s:e].permute(0, 2, 1, 3).float()
        h_bf16 = h.to(torch.bfloat16).float()
        wh = torch.matmul(wc, h_bf16)
        vnc = uc - wh
        if g is not None:
            gc = g[:, s:e].permute(0, 2, 1).float()
            gl = gc[:, :, -1:].float()
            vnc = vnc * torch.exp(gl - gc).unsqueeze(-1)
        v_new_out[:, s:e] = vnc.permute(0, 2, 1, 3).to(torch.bfloat16)
        if g is not None:
            gls = gc[:, :, -1].float()
            h = h * torch.exp(gls).unsqueeze(-1).unsqueeze(-1)
        if gk is not None:
            gkc = gk[:, s:e].permute(0, 2, 1, 3).float()
            gkl = gkc[:, :, -1, :].float()
            h = h * torch.exp(gkl).unsqueeze(-1)
        vn_bf16 = vnc.to(torch.bfloat16).float()
        h = h + torch.matmul(kc.transpose(-2, -1), vn_bf16)
        h_after.append(h[0, 0].to(torch.bfloat16).clone())
    return v_new_out, h_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--head_dim_k", type=int, default=128)
    parser.add_argument("--head_dim_v", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    B, T, H, K, V = args.batch_size, args.seq_len, args.num_heads, args.head_dim_k, args.head_dim_v
    BT = args.chunk_size
    NT = (T + BT - 1) // BT

    print(f"V2 Test: B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}, NT={NT}")

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1

    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
    stream = cutlass_torch.default_stream()

    compiled_kernel = None

    # Dummy cu_seqlens/chunk_offsets/workspace for non-varlen mode
    cu_seqlens_dummy = torch.zeros(2, dtype=torch.int32, device="cuda")
    chunk_offsets_dummy = torch.zeros(2, dtype=torch.int32, device="cuda")
    workspace_dummy = torch.zeros(128, dtype=torch.uint8, device="cuda")
    cu_seqlens_c = from_dlpack(cu_seqlens_dummy)
    chunk_offsets_c = from_dlpack(chunk_offsets_dummy)
    workspace_c = from_dlpack(workspace_dummy)

    def run_kernel(k_t, w_t, u_t, g_t, gk_t, h0_t, use_g_val, use_gk_val, use_h0, store_ht, do_save_vnew=0):
        nonlocal compiled_kernel
        h_out = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
        v_new = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
        ht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.bfloat16)

        kc, wc, uc = from_dlpack(k_t), from_dlpack(w_t), from_dlpack(u_t)
        gc, gkc = from_dlpack(g_t), from_dlpack(gk_t)
        h0c = from_dlpack(h0_t)
        hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new), from_dlpack(ht)

        args_tuple = (
            kc.iterator, wc.iterator, uc.iterator,
            gc.iterator, gkc.iterator,
            hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
            cu_seqlens_c.iterator, chunk_offsets_c.iterator,
            workspace_c.iterator,
            (B, T, H, K, V), NT,
            int(use_g_val), int(use_gk_val), int(use_h0), int(store_ht), int(do_save_vnew),
            stream,
        )

        if compiled_kernel is None:
            print("Compiling...")
            t0 = time.time()
            compiled_kernel = cute.compile(kernel, *args_tuple)
            print(f"Compiled in {time.time()-t0:.2f}s")

        compiled_kernel(*args_tuple)
        torch.cuda.synchronize()
        return h_out, v_new, ht

    all_pass = True

    # ===== Test 1: No gating, no h0 =====
    print("\n" + "="*60)
    print("Test 1: No gating, no h0")
    g_z = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk_z = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    h0_z = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_z, 0, 0, 0, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t1_pass = max_diff < 0.5
    print(f"  {'PASS' if t1_pass else 'FAIL'}")
    all_pass = all_pass and t1_pass

    # ===== Test 2: With gk + h0 =====
    print("\n" + "="*60)
    print("Test 2: With gk + h0")
    gk_val = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    gk_val = -torch.abs(gk_val)
    gk_val = gk_val.cumsum(dim=1)
    h0_val = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.01

    compiled_kernel = None  # recompile since use_gk changes
    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_val, 0, 1, 1, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=h0_val, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t2_pass = max_diff < 0.5
    print(f"  {'PASS' if t2_pass else 'FAIL'}")
    all_pass = all_pass and t2_pass

    # ===== Test 3: With gk gating =====
    print("\n" + "="*60)
    print("Test 3: With gk gating")
    gk_val = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    gk_val = -torch.abs(gk_val)
    gk_val = gk_val.cumsum(dim=1)

    compiled_kernel = None  # recompile
    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_z, 0, 1, 0, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=None, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t3_pass = max_diff < 0.5
    print(f"  {'PASS' if t3_pass else 'FAIL'}")
    all_pass = all_pass and t3_pass

    # ===== Test 4: With h0 initial state =====
    print("\n" + "="*60)
    print("Test 4: With h0 initial state")
    h0_val = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.01

    compiled_kernel = None  # recompile
    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_val, 0, 0, 1, 0)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, h0=h0_val, chunk_size=BT)

    # h_out[0] should be h0 (bf16 rounded)
    h0_bf16 = h0_val.to(torch.bfloat16)
    d0 = (h_out[0, 0, 0].float() - h0_bf16[0, 0].float()).abs().max().item()
    print(f"  h_out[0] vs h0 bf16: {d0:.6f}")

    max_diff = d0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t4_pass = max_diff < 0.5
    print(f"  {'PASS' if t4_pass else 'FAIL'}")
    all_pass = all_pass and t4_pass

    # ===== Test 5: With store_final_state (ht) =====
    print("\n" + "="*60)
    print("Test 5: store_final_state")

    compiled_kernel = None  # recompile
    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_z, 0, 0, 0, 1)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)

    # ht should match the last h_ref (after all chunks)
    ht_ref = h_ref_bf16[-1]  # last chunk's state
    # ht layout: (B, H, K, V) but kernel writes in transposed (V, K) format
    # Compare ht[0, 0] with ht_ref
    d_ht = (ht[0, 0].float() - ht_ref.float()).abs().max().item()
    print(f"  ht vs ref: {d_ht:.6f}")
    t5_pass = d_ht < 0.5
    print(f"  {'PASS' if t5_pass else 'FAIL'}")
    all_pass = all_pass and t5_pass

    # ===== Test 6: gk + h0 + ht (all features) =====
    print("\n" + "="*60)
    print("Test 6: gk + h0 + ht (all features)")

    compiled_kernel = None
    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_val, 0, 1, 1, 1)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=h0_val, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        d = (h_out[0, t + 1, 0].float() - h_ref_bf16[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    d_ht = (ht[0, 0].float() - h_ref_bf16[-1].float()).abs().max().item()
    max_diff = max(max_diff, d_ht)
    print(f"  max diff (h_out + ht): {max_diff:.6f}")
    t6_pass = max_diff < 0.5
    print(f"  {'PASS' if t6_pass else 'FAIL'}")
    all_pass = all_pass and t6_pass

    # ===== Test 7: Larger config =====
    print("\n" + "="*60)
    print("Test 7: B=2, T=512, H=4 (no gating)")
    B2, T2, H2 = 2, 512, 4
    NT2 = (T2 + BT - 1) // BT
    torch.manual_seed(123)
    k2 = torch.randn(B2, T2, H2, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(B2, T2, H2, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u2 = torch.randn(B2, T2, H2, V, device="cuda", dtype=torch.bfloat16) * 0.1
    g_z2 = torch.zeros(B2, T2, H2, device="cuda", dtype=torch.float32)
    gk_z2 = torch.zeros(B2, T2, H2, K, device="cuda", dtype=torch.float32)
    h0_z2 = torch.zeros(B2, H2, K, V, device="cuda", dtype=torch.float32)

    # Need new kernel instance for different B/T/H
    kernel2 = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
    h_out2 = torch.zeros(B2, NT2, H2, K, V, device="cuda", dtype=torch.bfloat16)
    v_new2 = torch.zeros(B2, T2, H2, V, device="cuda", dtype=torch.bfloat16)
    ht2 = torch.zeros(B2, H2, K, V, device="cuda", dtype=torch.bfloat16)

    kc2, wc2, uc2 = from_dlpack(k2), from_dlpack(w2), from_dlpack(u2)
    gc2, gkc2 = from_dlpack(g_z2), from_dlpack(gk_z2)
    h0c2 = from_dlpack(h0_z2)
    hc2, vnc2, htc2 = from_dlpack(h_out2), from_dlpack(v_new2), from_dlpack(ht2)
    cu_seqlens_d2 = torch.zeros(2, dtype=torch.int32, device="cuda")
    chunk_offsets_d2 = torch.zeros(2, dtype=torch.int32, device="cuda")
    workspace_d2 = torch.zeros(128, dtype=torch.uint8, device="cuda")
    csd2 = from_dlpack(cu_seqlens_d2)
    cod2 = from_dlpack(chunk_offsets_d2)
    wsd2 = from_dlpack(workspace_d2)

    compiled2 = cute.compile(
        kernel2,
        kc2.iterator, wc2.iterator, uc2.iterator,
        gc2.iterator, gkc2.iterator,
        hc2.iterator, vnc2.iterator, h0c2.iterator, htc2.iterator,
        csd2.iterator, cod2.iterator, wsd2.iterator,
        (B2, T2, H2, K, V), NT2, 0, 0, 0, 0, 0, stream,
    )
    compiled2(
        kc2.iterator, wc2.iterator, uc2.iterator,
        gc2.iterator, gkc2.iterator,
        hc2.iterator, vnc2.iterator, h0c2.iterator, htc2.iterator,
        csd2.iterator, cod2.iterator, wsd2.iterator,
        (B2, T2, H2, K, V), NT2, 0, 0, 0, 0, 0, stream,
    )
    torch.cuda.synchronize()

    _, h_ref2 = reference_bf16_roundtrip(k2, w2, u2, h0=None, chunk_size=BT)

    max_diff = 0.0
    for t in range(min(NT2 - 1, len(h_ref2))):
        d = (h_out2[0, t + 1, 0].float() - h_ref2[t].float()).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"  max diff h_out: {max_diff:.6f}")
    t7_pass = max_diff < 0.5
    print(f"  {'PASS' if t7_pass else 'FAIL'}")
    all_pass = all_pass and t7_pass

    # ===== Test 8: v_new output (no gating) =====
    print("\n" + "="*60)
    print("Test 8: v_new output (no gating)")

    compiled_kernel = None  # recompile
    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_z, h0_z, 0, 0, 0, 0, do_save_vnew=1)
    vnew_ref, _ = reference_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)

    d_vnew = (v_new.float() - vnew_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vnew:.6f}")
    t8_pass = d_vnew < 0.5
    print(f"  {'PASS' if t8_pass else 'FAIL'}")
    all_pass = all_pass and t8_pass

    # ===== Test 9: v_new output (with gk gating) =====
    print("\n" + "="*60)
    print("Test 9: v_new output (with gk gating)")

    compiled_kernel = None  # recompile
    h_out, v_new, ht = run_kernel(k, w, u, g_z, gk_val, h0_z, 0, 1, 0, 0, do_save_vnew=1)
    vnew_ref, _ = reference_bf16_roundtrip(k, w, u, gk=gk_val, h0=None, chunk_size=BT)

    d_vnew = (v_new.float() - vnew_ref.float()).abs().max().item()
    print(f"  v_new max diff: {d_vnew:.6f}")
    t9_pass = d_vnew < 0.5
    print(f"  {'PASS' if t9_pass else 'FAIL'}")
    all_pass = all_pass and t9_pass

    # ===== Summary =====
    print("\n" + "="*60)
    results = [t1_pass, t2_pass, t3_pass, t4_pass, t5_pass, t6_pass, t7_pass, t8_pass, t9_pass]
    names = ["No gate", "gk + h0", "gk gate", "h0 init", "ht store", "All features", "Larger config", "v_new (no gk)", "v_new (gk)"]
    for i, (name, r) in enumerate(zip(names, results)):
        print(f"  Test {i+1} ({name}): {'PASS' if r else 'FAIL'}")
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} tests passed")
    print("ALL PASS" if all_pass else "SOME FAILED")

    # ===== Benchmark =====
    print("\n" + "="*60)
    print("Benchmark: B=4, T=4096, H=64, K=128, V=128")
    Bb, Tb, Hb = 4, 4096, 64
    NTb = (Tb + BT - 1) // BT
    torch.manual_seed(999)
    kb = torch.randn(Bb, Tb, Hb, K, device="cuda", dtype=torch.bfloat16) * 0.1
    wb = torch.randn(Bb, Tb, Hb, K, device="cuda", dtype=torch.bfloat16) * 0.1
    ub = torch.randn(Bb, Tb, Hb, V, device="cuda", dtype=torch.bfloat16) * 0.1
    gb = torch.zeros(Bb, Tb, Hb, device="cuda", dtype=torch.float32)
    gkb = torch.zeros(Bb, Tb, Hb, K, device="cuda", dtype=torch.float32)
    h0b = torch.zeros(Bb, Hb, K, V, device="cuda", dtype=torch.float32)
    h_outb = torch.zeros(Bb, NTb, Hb, K, V, device="cuda", dtype=torch.bfloat16)
    v_newb = torch.zeros(Bb, Tb, Hb, V, device="cuda", dtype=torch.bfloat16)
    htb = torch.zeros(Bb, Hb, K, V, device="cuda", dtype=torch.bfloat16)

    kernelb = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)

    kcb, wcb, ucb = from_dlpack(kb), from_dlpack(wb), from_dlpack(ub)
    gcb, gkcb = from_dlpack(gb), from_dlpack(gkb)
    h0cb = from_dlpack(h0b)
    hcb, vncb, htcb = from_dlpack(h_outb), from_dlpack(v_newb), from_dlpack(htb)
    cu_seqlens_db = torch.zeros(2, dtype=torch.int32, device="cuda")
    chunk_offsets_db = torch.zeros(2, dtype=torch.int32, device="cuda")
    workspace_db = torch.zeros(128, dtype=torch.uint8, device="cuda")
    csdb = from_dlpack(cu_seqlens_db)
    codb = from_dlpack(chunk_offsets_db)
    wsdb = from_dlpack(workspace_db)

    bench_args = (
        kcb.iterator, wcb.iterator, ucb.iterator,
        gcb.iterator, gkcb.iterator,
        hcb.iterator, vncb.iterator, h0cb.iterator, htcb.iterator,
        csdb.iterator, codb.iterator, wsdb.iterator,
        (Bb, Tb, Hb, K, V), NTb, 0, 0, 0, 0, 0, stream,
    )
    compiled_b = cute.compile(kernelb, *bench_args)

    # Warmup
    for _ in range(3):
        compiled_b(*bench_args)
    torch.cuda.synchronize()

    # Benchmark
    n_iter = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        compiled_b(*bench_args)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / n_iter
    print(f"  V2 kernel: {elapsed_ms:.3f} ms")

    # FLA h-kernel reference (apples-to-apples: h-state recurrence only)
    try:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h
        # Warmup
        for _ in range(3):
            fla_fwd_h(
                k=kb, w=wb, u=ub,
                g=None, gk=None,
                initial_state=None,
                output_final_state=False,
                chunk_size=BT,
                save_new_value=True,
            )
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n_iter):
            fla_fwd_h(
                k=kb, w=wb, u=ub,
                g=None, gk=None,
                initial_state=None,
                output_final_state=False,
                chunk_size=BT,
                save_new_value=True,
            )
        end_event.record()
        torch.cuda.synchronize()
        fla_ms = start_event.elapsed_time(end_event) / n_iter
        print(f"  FLA h-kernel: {fla_ms:.3f} ms")
        print(f"  Speedup vs FLA h-kernel: {fla_ms / elapsed_ms:.2f}x")
    except Exception as e:
        print(f"  FLA not available: {e}")


if __name__ == "__main__":
    main()
