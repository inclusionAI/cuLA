# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
CuTeDSL kernel for KDA recompute_w_u_fwd.

Computes (always with gk, always recompute):
  w  = A @ (k * beta * exp2(gk))     — [BT, BK] per tile
  u  = A @ (v * beta)                — [BT, BV] per tile
  kg = k * exp2(gn - gk)             — [BT, BK] per tile

MMA layout (tcgen05):
  C[M, N] = A_mma[M, K] @ B_mma[N, K]^T
  A_mma(TMEM) = B_proc^T[BN, BT],  B_mma(SMEM) = A_mat[BT, BT]
  Result = output^T[BN, BT] -> transpose in epilogue writes.
  MMA tiler = (BN, BT, BT).

Persistent mode: grid = SM_count; each CTA loops over multiple work units
  (one WU = one chunk x one head). With 2-stage TMA pipelines, the load
  warp pre-fetches the NEXT chunk's data while CUDA warps compute the
  current chunk, maximising TMA<->compute overlap.

Varlen mode: variable sequence lengths. cu_seqlens[N+1] gives token
  offsets; chunk_indices[total_nt*2] gives (batch_idx, chunk_in_seq)
  pairs for each global chunk index. TMA uses domain_offset for per-WU
  alignment, matching the fwd_o.py pattern.

Warp assignment:
  0-3: CUDA core warps (element-wise compute, GMEM store w/u/kg)
  4:   MMA warp (tcgen05 GEMM)
  5:   Load warp (TMA for A_mat, k, v, gk)
  6:   Store warp (idle)
  7:   Empty warp (idle)
"""

import math
import os
import time
from typing import Type, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Int32, Int64, Float32


def _make_coop_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class KDARecomputeWU:

    def __init__(
        self,
        K: int = 128,
        V: int = 128,
        chunk_size: int = 64,
        block_k: int = None,
        block_v: int = None,
        io_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        is_varlen: bool = False,
        persistent: bool = False,
    ):
        self.K = K
        self.V = V
        self.BT = chunk_size
        self.BK = block_k if block_k is not None else K
        self.BV = block_v if block_v is not None else V
        self.NK = (K + self.BK - 1) // self.BK
        self.NV = (V + self.BV - 1) // self.BV
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype
        self.is_varlen = is_varlen
        self.persistent = persistent

        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        self.empty_warp_id = 7
        self.threads_per_cta = self.threads_per_warp * 8  # 256
        self.num_cuda_warps = len(self.cuda_warp_ids)
        self.num_cuda_threads = self.threads_per_warp * self.num_cuda_warps  # 128

        self.BN = max(self.BK, self.BV)
        self.mma_tiler = (self.BN, self.BT, self.BT)

        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mnk = (1, 1, 1)
        self.buffer_align_bytes = 1024

        self.bproc_stage = 1
        self.acc_stage = 2

        # Staging buffer for coalesced GMEM stores (transpose via SMEM)
        # sStage[BT, BN + PAD] bf16 — PAD avoids bank conflicts
        self.stage_pad = 4
        self.stage_stride_row = self.BN + self.stage_pad  # stride in elements for BT dim
        self.stage_elems = self.BT * self.stage_stride_row

        if persistent:
            # occ=2 persistent: keep 2-CTA-per-SM parallelism
            # SMEM: sA(2×8=16KB) + sK(16KB) + sV(16KB) + sGK(32KB) = 80KB < 114KB ✓
            # 2-stage A: prefetch next WU's A_mat while current WU computes
            self.min_occupancy = 2
            self.num_regs_cuda = 232
            self.num_regs_others = 24
            self.a_stage = 2
            self.kgk_stage = 1
            self.v_tma_stage = 1
        else:
            # occ=2: 114KB SMEM limit; single-stage
            # SMEM: sA(8KB) + sK(16KB) + sV(16KB) + sGK(32KB) = 72KB < 114KB
            self.min_occupancy = 2
            self.num_regs_cuda = 232
            self.num_regs_others = 24
            self.a_stage = 1
            self.kgk_stage = 1
            self.v_tma_stage = 1

    @staticmethod
    def _plan_tmem_offsets(tiled_mma, mma_tiler, tmem_a_layout, acc_stages, io_dtype, acc_dtype):
        SM100_TMEM_CAPACITY_COLS = 512
        tCrA_fake = tiled_mma.make_fragment_A(tmem_a_layout.outer.shape)
        num_a = tcgen05.find_tmem_tensor_col_offset(tCrA_fake)
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))
        num_acc = tcgen05.find_tmem_tensor_col_offset(tCtAcc_fake)
        acc_off = 0
        a_off = acc_off + num_acc
        total_tmp = a_off + num_a
        total = 1
        while total < total_tmp:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS
        return acc_off, a_off, total

    @cute.jit
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        coord = (0, None, None)
        gX = cute.local_tile(
            tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx))
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1),
            cute.group_modes(smem, 0, 3), cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _data_tma_partition(self, atom, tma_tensor_3d, tile_shape, smem, head_idx, batch_idx):
        """Partition for non-MMA TMA load (epilog-style)."""
        gmem_2d = tma_tensor_3d[None, None, (head_idx, batch_idx)]
        gC_tiled = cute.local_tile(gmem_2d, tile_shape, (None, None))
        sC_g = cute.group_modes(smem, 0, 2)
        gC_g = cute.group_modes(gC_tiled, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), sC_g, gC_g,
        )
        return bSG_sC, bSG_gC

    @cute.jit
    def _epilog_partition_varlen(self, atom, gC_2d, epi_tile, sC):
        """Partition for varlen epilog TMA load (2D tensor with domain_offset).
        Uses local_tile to correctly handle domain_offset coordinates.
        """
        gC_tiled = cute.local_tile(gC_2d, epi_tile, (None, None))
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_tiled, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), sC_g, gC_g,
        )
        return bSG_sC, bSG_gC

    @cute.jit
    def __call__(
        self,
        k_in: cute.Tensor,
        v_in: cute.Tensor,
        beta_in: cute.Tensor,
        A_in: cute.Tensor,
        gk_in: cute.Tensor,
        w_in: cute.Tensor,
        u_in: cute.Tensor,
        kg_in: cute.Tensor,
        cu_seqlens_in: cute.Tensor,
        chunk_indices_in: cute.Tensor,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,
        stream,
    ):
        k_ptr = k_in.iterator
        v_ptr = v_in.iterator
        beta_ptr = beta_in.iterator
        A_ptr = A_in.iterator
        gk_ptr = gk_in.iterator
        w_ptr = w_in.iterator
        u_ptr = u_in.iterator
        kg_ptr = kg_in.iterator
        cu_seqlens_ptr = cu_seqlens_in.iterator
        chunk_indices_ptr = chunk_indices_in.iterator

        B, T, H, K, V = problem_size
        BT = self.BT

        # For varlen: data_B=1, T=T_total
        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
        else:
            data_B = B

        # ---------- MMA setup ----------
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        tmem_a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.io_dtype, 1,
        )

        (self.tmem_acc_off, self.tmem_a_off, self.tmem_total) = self._plan_tmem_offsets(
            tiled_mma, self.mma_tiler, tmem_a_layout,
            self.acc_stage, self.io_dtype, self.acc_dtype,
        )

        # ---------- TMA load op ----------
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        # ---------- SMEM layout: A_mat (MMA B operand) ----------
        a_smem_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, self.a_stage,
        )

        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma.thr_id.shape,),
        )

        # ---------- SMEM layouts: k (bf16), v (bf16), gk (fp32) ----------
        k_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK), self.kgk_stage,
        )
        v_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BV), self.v_tma_stage,
        )
        gk_epi_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK), self.kgk_stage,
        )

        # ---------- GMEM tensors (token-indexed) ----------
        # varlen: T=T_total, data_B=1
        # non-varlen: T=seq_len, data_B=B
        A_layout = cute.make_layout(
            (T, BT, (H, data_B)),
            stride=(H * BT, 1, (BT, T * H * BT)),
        )
        A_gmem = cute.make_tensor(A_ptr, A_layout)

        k_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        k_gmem = cute.make_tensor(k_ptr, k_layout)

        v_layout = cute.make_layout(
            (T, V, (H, data_B)),
            stride=(H * V, 1, (V, T * H * V)),
        )
        v_gmem = cute.make_tensor(v_ptr, v_layout)

        gk_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        gk_gmem = cute.make_tensor(gk_ptr, gk_layout)

        # ---------- TMA descriptors ----------
        a_smem_one = cute.select(a_smem_staged, mode=[0, 1, 2])
        tma_atom_A, tma_tensor_A = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, A_gmem, a_smem_one, self.mma_tiler, tiled_mma,
            cluster_layout.shape,
        )
        self.tma_A_bytes = cute.size_in_bytes(self.io_dtype, a_smem_one)

        k_epi_smem = cute.select(k_epi_staged, mode=[0, 1])
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op, k_gmem, k_epi_smem, (self.BT, self.BK),
        )

        v_epi_smem = cute.select(v_epi_staged, mode=[0, 1])
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op, v_gmem, v_epi_smem, (self.BT, self.BV),
        )

        gk_epi_smem = cute.select(gk_epi_staged, mode=[0, 1])
        tma_atom_gk, tma_tensor_gk = cpasync.make_tiled_tma_atom(
            tma_load_op, gk_gmem, gk_epi_smem, (self.BT, self.BK),
        )

        # ---------- TMA byte counts ----------
        self.tma_bytes_k = cute.size_in_bytes(self.io_dtype, k_epi_smem)
        self.tma_bytes_v = cute.size_in_bytes(self.io_dtype, v_epi_smem)
        self.tma_bytes_gk = cute.size_in_bytes(self.acc_dtype, gk_epi_smem)
        self.tma_bytes_kgkv = self.tma_bytes_k + self.tma_bytes_gk + self.tma_bytes_v

        # ---------- SharedStorage ----------
        @cute.struct
        class SharedStorage:
            load_A_mbar: cute.struct.MemRange[Int64, self.a_stage * 2]
            load_kgkv_mbar: cute.struct.MemRange[Int64, self.kgk_stage * 2]
            bproc_mbar: cute.struct.MemRange[Int64, self.bproc_stage * 2]
            acc_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]
            tmem_holding_buf: Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_epi_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_epi_staged)],
                self.buffer_align_bytes,
            ]
            sGK: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(gk_epi_staged)],
                self.buffer_align_bytes,
            ]
            sBeta: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, self.BT + 2],
                128,
            ]
            sStage: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, self.stage_elems],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # ---------- cu_seqlens / chunk_indices tensors ----------
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((B + 1,)))
        chunk_indices = cute.make_tensor(chunk_indices_ptr, cute.make_layout((total_nt * 2,)))

        # ---------- Grid ----------
        if cutlass.const_expr(self.persistent):
            sm_count = Int32(torch.cuda.get_device_properties(0).multi_processor_count)
            # 2×SM_count: keep occ=2 (two CTAs per SM), each looping over many WUs
            # Block-contiguous scheduling for cache locality
            grid = (sm_count * 2, 1, 1)
        elif cutlass.const_expr(self.is_varlen):
            grid = (total_nt * H, 1, 1)
        else:
            NT = (T + BT - 1) // BT
            grid = (NT, H, B)

        self.kernel(
            tiled_mma,
            tma_atom_A, tma_tensor_A,
            tma_atom_k, tma_tensor_k,
            tma_atom_v, tma_tensor_v,
            tma_atom_gk, tma_tensor_gk,
            tmem_a_layout,
            a_smem_staged,
            k_epi_staged,
            v_epi_staged,
            gk_epi_staged,
            beta_ptr,
            w_ptr, u_ptr, kg_ptr,
            cu_seqlens, chunk_indices,
            problem_size,
            total_nt,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=self.min_occupancy,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_A: cute.CopyAtom,
        tma_tensor_A: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_gk: cute.CopyAtom,
        tma_tensor_gk: cute.Tensor,
        tmem_a_layout: cute.ComposedLayout,
        a_smem_staged: cute.ComposedLayout,
        k_epi_staged: cute.ComposedLayout,
        v_epi_staged: cute.ComposedLayout,
        gk_epi_staged: cute.ComposedLayout,
        beta_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        u_ptr: cute.Pointer,
        kg_ptr: cute.Pointer,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,
    ):
        B, T, H, K, V = problem_size
        BT = self.BT

        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
        else:
            data_B = B

        # ===================== Work-unit decode =====================
        if cutlass.const_expr(self.persistent):
            block_idx_x = cute.arch.block_idx()[0]
            grid_dim_x = cute.arch.grid_dim()[0]
            total_work_units = total_nt * H
            # Block-contiguous: each CTA gets a contiguous range of WUs
            # → same (head, batch) within a CTA → better HBM cache locality
            work_per_cta_base = total_work_units // grid_dim_x
            extra = total_work_units % grid_dim_x
            # CTAs [0, extra) get (work_per_cta_base+1) WUs; rest get work_per_cta_base
            # Pre-initialize before conditional flow
            wu_start = Int32(0)
            num_iters = work_per_cta_base
            if block_idx_x < extra:
                wu_start = block_idx_x * (work_per_cta_base + 1)
                num_iters = work_per_cta_base + 1
            else:
                wu_start = extra * (work_per_cta_base + 1) + (block_idx_x - extra) * work_per_cta_base
                num_iters = work_per_cta_base
            i_h = Int32(0)
            i_b = Int32(0)
            i_t = Int32(0)
            tok_offset = Int32(0)
            data_bidx = Int32(0)
            chunk_global = Int32(0)
            if cutlass.const_expr(not self.is_varlen):
                NT = (T + BT - 1) // BT
        else:
            if cutlass.const_expr(self.is_varlen):
                work_idx_x = cute.arch.block_idx()[0]
                chunk_global = work_idx_x % total_nt
                i_h = work_idx_x // total_nt
                i_b = chunk_indices[chunk_global * 2]
                i_t = chunk_indices[chunk_global * 2 + 1]
                tok_offset = cu_seqlens[i_b]
                data_bidx = Int32(0)
            else:
                NT = (T + BT - 1) // BT
                i_t, i_h, i_b = cute.arch.block_idx()
                tok_offset = i_b * T
                data_bidx = i_b
                chunk_global = i_b * NT + i_t
            num_iters = Int32(1)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_gk)

        # ---------- SMEM ----------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA  = storage.sA.get_tensor(a_smem_staged.outer,  swizzle=a_smem_staged.inner)
        sK  = storage.sK.get_tensor(k_epi_staged.outer,   swizzle=k_epi_staged.inner)
        sV  = storage.sV.get_tensor(v_epi_staged.outer,   swizzle=v_epi_staged.inner)
        sGK = storage.sGK.get_tensor(gk_epi_staged.outer, swizzle=gk_epi_staged.inner)
        sBeta = cute.make_tensor(
            cute.make_ptr(self.io_dtype, storage.sBeta.data_ptr().toint(),
                          cute.AddressSpace.smem),
            cute.make_layout((self.BT,), stride=(1,)),
        )
        sStage_ptr = cute.make_ptr(
            self.io_dtype, storage.sStage.data_ptr().toint(),
            cute.AddressSpace.smem,
        )
        sStage = cute.make_tensor(
            sStage_ptr,
            cute.make_layout((self.BT, self.BN), stride=(self.stage_stride_row, 1)),
        )

        # ---------- Pipelines ----------
        load_A_P, load_A_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.a_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(1),
            tx_count=self.tma_A_bytes,
            barrier_storage=storage.load_A_mbar.data_ptr(),
        ).make_participants()

        load_kgk_P, load_kgk_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.kgk_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(self.num_cuda_warps),
            tx_count=self.tma_bytes_kgkv,
            barrier_storage=storage.load_kgkv_mbar.data_ptr(),
        ).make_participants()

        bproc_P, bproc_C = pipeline.PipelineAsyncUmma.create(
            num_stages=self.bproc_stage,
            producer_group=_make_coop_group(self.num_cuda_threads),
            consumer_group=_make_coop_group(1),
            barrier_storage=storage.bproc_mbar.data_ptr(),
        ).make_participants()

        acc_done_P, acc_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(self.num_cuda_threads),
            barrier_storage=storage.acc_mbar.data_ptr(),
        ).make_participants()

        # ---------- TMEM ----------
        tmem_alloc_bar = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        tCrA_fake = tiled_mma.make_fragment_A(tmem_a_layout.outer.shape)
        tCrA = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_a_off, dtype=tCrA_fake.element_type),
            tCrA_fake.layout,
        )
        tCrB = tiled_mma.make_fragment_B(sA)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stage))
        tCtAcc = cute.make_tensor(tmem_ptr + self.tmem_acc_off, tCtAcc_fake.layout)

        # =====================================================================
        # LOAD WARP
        # =====================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # --- Persistent work decode (block-contiguous) ---
                if cutlass.const_expr(self.persistent):
                    work_idx = wu_start + wu_iter
                    chunk_global = work_idx % total_nt
                    i_h = work_idx // total_nt
                    if cutlass.const_expr(self.is_varlen):
                        i_b = chunk_indices[chunk_global * 2]
                        i_t = chunk_indices[chunk_global * 2 + 1]
                        tok_offset = cu_seqlens[i_b]
                        data_bidx = Int32(0)
                    else:
                        i_b = chunk_global // NT
                        i_t = chunk_global % NT
                        tok_offset = i_b * T
                        data_bidx = i_b

                # --- Domain offset (varlen) or alias (non-varlen) ---
                if cutlass.const_expr(self.is_varlen):
                    tma_k_v  = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_k)
                    tma_v_v  = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_v)
                    tma_gk_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_gk)
                    tma_A_v  = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_A)
                else:
                    tma_k_v  = tma_tensor_k
                    tma_v_v  = tma_tensor_v
                    tma_gk_v = tma_tensor_gk
                    tma_A_v  = tma_tensor_A

                # --- TMA partitions per WU ---
                if cutlass.const_expr(self.is_varlen):
                    bSG_sK, bSG_gK = self._epilog_partition_varlen(
                        tma_atom_k,  tma_k_v[None, None, (i_h, data_bidx)],  (self.BT, self.BK), sK,
                    )
                    bSG_sV, bSG_gV = self._epilog_partition_varlen(
                        tma_atom_v,  tma_v_v[None, None, (i_h, data_bidx)],  (self.BT, self.BV), sV,
                    )
                    bSG_sGK, bSG_gGK = self._epilog_partition_varlen(
                        tma_atom_gk, tma_gk_v[None, None, (i_h, data_bidx)], (self.BT, self.BK), sGK,
                    )
                    tAsA, tAgA = self._tma_partition_B(
                        tma_atom_A, tma_A_v, sA, self.mma_tiler, tiled_mma, data_bidx, i_h,
                    )
                else:
                    bSG_sK, bSG_gK = self._data_tma_partition(
                        tma_atom_k,  tma_k_v,  (self.BT, self.BK), sK,  i_h, data_bidx,
                    )
                    bSG_sV, bSG_gV = self._data_tma_partition(
                        tma_atom_v,  tma_v_v,  (self.BT, self.BV), sV,  i_h, data_bidx,
                    )
                    bSG_sGK, bSG_gGK = self._data_tma_partition(
                        tma_atom_gk, tma_gk_v, (self.BT, self.BK), sGK, i_h, data_bidx,
                    )
                    tAsA, tAgA = self._tma_partition_B(
                        tma_atom_A, tma_A_v, sA, self.mma_tiler, tiled_mma, data_bidx, i_h,
                    )

                # --- Issue TMA loads ---
                h_a = load_A_P.acquire_and_advance()
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, i_t, 0)],
                    tAsA[(None, h_a.index)],
                    tma_bar_ptr=h_a.barrier,
                )

                for i_kv in cutlass.range(0, self.NK):
                    kgk_h = load_kgk_P.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        bSG_gK[(None, i_t, i_kv)],
                        bSG_sK[None, kgk_h.index],
                        tma_bar_ptr=kgk_h.barrier,
                    )
                    cute.copy(
                        tma_atom_gk,
                        bSG_gGK[(None, i_t, i_kv)],
                        bSG_sGK[None, kgk_h.index],
                        tma_bar_ptr=kgk_h.barrier,
                    )
                    cute.copy(
                        tma_atom_v,
                        bSG_gV[(None, i_t, i_kv)],
                        bSG_sV[None, kgk_h.index],
                        tma_bar_ptr=kgk_h.barrier,
                    )

        # =====================================================================
        # STORE WARP -- idle
        # =====================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

        # =====================================================================
        # EMPTY WARP -- idle
        # =====================================================================
        elif warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

        # =====================================================================
        # MMA WARP
        # =====================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            num_kblks = cute.size(tCrB, mode=[2])

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # Wait for A_mat
                a_h = load_A_C.wait_and_advance()
                a_h.release()

                for i_kv in cutlass.range(0, self.NK):
                    # K pass: w = A_mat @ B_proc(k)
                    bp_h = bproc_C.wait_and_advance()
                    acc_h = acc_done_P.acquire_and_advance()
                    for kblk in cutlass.range(num_kblks, unroll_full=True):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kblk != 0))
                        cute.gemm(
                            tiled_mma,
                            tCtAcc[(None, None, None, acc_h.index)],
                            tCrA[(None, None, kblk, 0)],
                            tCrB[(None, None, kblk, 0)],
                            tCtAcc[(None, None, None, acc_h.index)],
                        )
                    acc_h.commit()
                    bp_h.release()

                    # V pass: u = A_mat @ B_proc(v)
                    bp_h2 = bproc_C.wait_and_advance()
                    acc_h2 = acc_done_P.acquire_and_advance()
                    for kblk in cutlass.range(num_kblks, unroll_full=True):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kblk != 0))
                        cute.gemm(
                            tiled_mma,
                            tCtAcc[(None, None, None, acc_h2.index)],
                            tCrA[(None, None, kblk, 0)],
                            tCrB[(None, None, kblk, 0)],
                            tCtAcc[(None, None, None, acc_h2.index)],
                        )
                    acc_h2.commit()
                    bp_h2.release()

        # =====================================================================
        # CUDA CORE WARPS
        # =====================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            local_tidx = tidx % (self.threads_per_warp * self.num_cuda_warps)

            # ---- Hoist loop-invariant T2R/R2T setup ----
            t2r_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
            fake_sOut = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.mma_tiler, (1, 1, None)),
            )
            tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
            thr_t2r = tiled_t2r.get_slice(local_tidx)
            tTR_tAcc = thr_t2r.partition_S(tCtAcc_flat)
            tTR_sOut = thr_t2r.partition_D(fake_sOut)

            r2t_atom = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t = tcgen05.make_tmem_copy(r2t_atom, tCrA)
            thr_r2t = tiled_r2t.get_slice(local_tidx)
            r2t_src_shape = cute.slice_(
                thr_r2t.partition_S(tCrA).shape, (None, None, None, None, 0)
            )
            tRT_tA = thr_r2t.partition_D(tCrA)

            out_tile = cute.dice(self.mma_tiler, (1, 1, None))
            cM_id = cute.make_identity_tensor(out_tile)
            tTR_cM = thr_t2r.partition_D(cM_id)

            # Rmem tensors (hoisted outside WU loop to minimise register lifetime)
            tTR_rAcc   = cute.make_rmem_tensor(tTR_sOut.shape, self.acc_dtype)
            tTR_rBproc = cute.make_rmem_tensor(tTR_sOut.shape, self.io_dtype)
            tRT_rBproc = cute.make_rmem_tensor(r2t_src_shape, self.io_dtype)
            tTR_rKg    = cute.make_rmem_tensor(tTR_sOut.shape, self.io_dtype)

            cuda_sync = pipeline.NamedBarrier(
                barrier_id=2, num_threads=self.num_cuda_threads)

            # ====== Per-WU loop ======
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # --- Persistent work decode (block-contiguous) ---
                if cutlass.const_expr(self.persistent):
                    work_idx = wu_start + wu_iter
                    chunk_global = work_idx % total_nt
                    i_h = work_idx // total_nt
                    if cutlass.const_expr(self.is_varlen):
                        i_b = chunk_indices[chunk_global * 2]
                        i_t = chunk_indices[chunk_global * 2 + 1]
                        tok_offset = cu_seqlens[i_b]
                    else:
                        i_b = chunk_global // NT
                        i_t = chunk_global % NT
                        tok_offset = i_b * T

                stride_k = cute.assume(H * K, divby=1)
                stride_v = cute.assume(H * V, divby=1)
                stride_beta = cute.assume(H, divby=1)

                time_base = (tok_offset + i_t * BT) * H + i_h

                # Load beta cooperatively (no A wait needed: CUDA warps don't read sA)
                beta_gmem_p = cute.make_ptr(
                    self.io_dtype, (beta_ptr + time_base).toint(),
                    cute.AddressSpace.gmem, assumed_align=2,
                )
                beta_gmem = cute.make_tensor(
                    beta_gmem_p,
                    cute.make_layout((self.BT,), stride=(stride_beta,)),
                )
                beta_load_idx = local_tidx % self.BT
                sBeta[beta_load_idx] = beta_gmem[beta_load_idx]
                cuda_sync.arrive_and_wait()

                for i_kv in cutlass.range(0, self.NK):
                    # ==== K pass: compute k*beta*exp2(gk) + kg ====
                    kgk_h = load_kgk_C.wait_and_advance()

                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        m_coord, k_coord = tTR_cM[ei]
                        k_val    = sK[(k_coord,  m_coord, kgk_h.index)].to(self.acc_dtype)
                        gk_val   = sGK[(k_coord, m_coord, kgk_h.index)]
                        gn_val   = sGK[(self.BT - 1, m_coord, kgk_h.index)]
                        beta_val = sBeta[k_coord].to(self.acc_dtype)
                        tTR_rBproc[ei] = (k_val * beta_val * cute.exp2(gk_val)).to(self.io_dtype)
                        tTR_rKg[ei]    = (k_val * cute.exp2(gn_val - gk_val)).to(self.io_dtype)

                    # R2T K bproc -> TMEM -> signal MMA K start
                    tRT_rBproc.store(tTR_rBproc.load())
                    bproc_h = bproc_P.acquire_and_advance()
                    cute.copy(tiled_r2t, tRT_rBproc, tRT_tA[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    bproc_h.commit()

                    # === Overlap with MMA K: precompute V + write kg ===
                    k_off = time_base * K + i_kv * self.BK

                    # Precompute v*beta FIRST (while sK/sV/sGK still held)
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        m_coord, k_coord = tTR_cM[ei]
                        v_val    = sV[(k_coord, m_coord, kgk_h.index)].to(self.acc_dtype)
                        beta_val = sBeta[k_coord].to(self.acc_dtype)
                        tTR_rBproc[ei] = (v_val * beta_val).to(self.io_dtype)
                    kgk_h.release()
                    tRT_rBproc.store(tTR_rBproc.load())

                    # Stage kg through SMEM for coalesced GMEM writes
                    kg_tile_p = cute.make_ptr(
                        self.io_dtype, (kg_ptr + k_off).toint(),
                        cute.AddressSpace.gmem, assumed_align=2,
                    )
                    kg_tile = cute.make_tensor(
                        kg_tile_p,
                        cute.make_layout((self.BT, self.BK), stride=(stride_k, 1)),
                    )
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        m_coord, k_coord = tTR_cM[ei]
                        sStage[(k_coord, m_coord)] = tTR_rKg[ei]
                    cuda_sync.arrive_and_wait()
                    for si in cutlass.range_constexpr(self.BT * self.BK // self.num_cuda_threads):
                        flat_idx = local_tidx + si * self.num_cuda_threads
                        s_row = flat_idx // self.BK
                        s_col = flat_idx % self.BK
                        kg_tile[(s_row, s_col)] = sStage[(s_row, s_col)]
                    cuda_sync.arrive_and_wait()

                    # === K MMA done: R2T V first to unblock MMA V, then read K result ===

                    # R2T V bproc -> TMEM -> signal MMA V start (BEFORE reading K result)
                    bproc_h2 = bproc_P.acquire_and_advance()
                    cute.copy(tiled_r2t, tRT_rBproc, tRT_tA[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    bproc_h2.commit()

                    # Now read K result from acc (MMA V can run on acc[1] concurrently)
                    acc_h = acc_done_C.wait_and_advance()
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, acc_h.index)], tTR_rAcc)
                    cute.arch.fence_view_async_tmem_load()
                    acc_h.release()

                    # Write w to GMEM via SMEM staging (overlaps with MMA V)
                    w_tile_p = cute.make_ptr(
                        self.io_dtype, (w_ptr + k_off).toint(),
                        cute.AddressSpace.gmem, assumed_align=2,
                    )
                    w_tile = cute.make_tensor(
                        w_tile_p,
                        cute.make_layout((self.BT, self.BK), stride=(stride_k, 1)),
                    )
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        m_coord, n_coord = tTR_cM[ei]
                        sStage[(n_coord, m_coord)] = tTR_rAcc[ei].to(self.io_dtype)
                    cuda_sync.arrive_and_wait()
                    for si in cutlass.range_constexpr(self.BT * self.BK // self.num_cuda_threads):
                        flat_idx = local_tidx + si * self.num_cuda_threads
                        s_row = flat_idx // self.BK
                        s_col = flat_idx % self.BK
                        w_tile[(s_row, s_col)] = sStage[(s_row, s_col)]
                    cuda_sync.arrive_and_wait()

                    # === V MMA done ===
                    acc_h2 = acc_done_C.wait_and_advance()
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, acc_h2.index)], tTR_rAcc)
                    cute.arch.fence_view_async_tmem_load()
                    acc_h2.release()

                    v_off = time_base * V + i_kv * self.BV
                    u_tile_p = cute.make_ptr(
                        self.io_dtype, (u_ptr + v_off).toint(),
                        cute.AddressSpace.gmem, assumed_align=2,
                    )
                    u_tile = cute.make_tensor(
                        u_tile_p,
                        cute.make_layout((self.BT, self.BV), stride=(stride_v, 1)),
                    )
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        m_coord, n_coord = tTR_cM[ei]
                        sStage[(n_coord, m_coord)] = tTR_rAcc[ei].to(self.io_dtype)
                    cuda_sync.arrive_and_wait()
                    for si in cutlass.range_constexpr(self.BT * self.BV // self.num_cuda_threads):
                        flat_idx = local_tidx + si * self.num_cuda_threads
                        s_row = flat_idx // self.BV
                        s_col = flat_idx % self.BV
                        u_tile[(s_row, s_col)] = sStage[(s_row, s_col)]
                    cuda_sync.arrive_and_wait()

        # ---------- TMEM cleanup ----------
        tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)


# ============================================================================
# Compile cache
# ============================================================================

_recompute_wu_cache = {}
_dummy_cu_seqlens = None
_dummy_chunk_indices = None


def _compile_recompute_wu(H, K, V, chunk_size=64, block_k=None, block_v=None, persistent=False, is_varlen=False):
    key = (H, K, V, chunk_size, block_k, block_v, persistent, is_varlen)
    if key in _recompute_wu_cache:
        return _recompute_wu_cache[key]

    kernel_obj = KDARecomputeWU(
        K=K, V=V, chunk_size=chunk_size,
        block_k=block_k, block_v=block_v,
        persistent=persistent, is_varlen=is_varlen,
    )

    sym_a = cute.sym_int()
    sym_b = cute.sym_int()
    sym_nt = cute.sym_int()
    sym_cu = cute.sym_int()
    sym_ci = cute.sym_int()
    BT = chunk_size

    if is_varlen:
        k_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, K),  stride_order=(2,1,0), assumed_align=128)
        v_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, V),  stride_order=(2,1,0), assumed_align=128)
        beta_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H),     stride_order=(1,0),   assumed_align=128)
        A_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, BT), stride_order=(2,1,0), assumed_align=128)
        gk_fake   = make_fake_compact_tensor(cutlass.Float32,  (sym_a, H, K),  stride_order=(2,1,0), assumed_align=128)
        w_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, K),  stride_order=(2,1,0), assumed_align=128)
        u_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, V),  stride_order=(2,1,0), assumed_align=128)
        kg_fake   = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, K),  stride_order=(2,1,0), assumed_align=128)
    else:
        k_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K),  stride_order=(3,2,1,0), assumed_align=128)
        v_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, V),  stride_order=(3,2,1,0), assumed_align=128)
        beta_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H),     stride_order=(2,1,0),   assumed_align=128)
        A_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, BT), stride_order=(3,2,1,0), assumed_align=128)
        gk_fake   = make_fake_compact_tensor(cutlass.Float32,  (sym_a, sym_b, H, K),  stride_order=(3,2,1,0), assumed_align=128)
        w_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K),  stride_order=(3,2,1,0), assumed_align=128)
        u_fake    = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, V),  stride_order=(3,2,1,0), assumed_align=128)
        kg_fake   = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K),  stride_order=(3,2,1,0), assumed_align=128)

    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=128)
    ci_fake = make_fake_compact_tensor(cutlass.Int32, (sym_ci,), assumed_align=128)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        k_fake, v_fake, beta_fake, A_fake, gk_fake,
        w_fake, u_fake, kg_fake,
        cu_fake, ci_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        Int32(1),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    _recompute_wu_cache[key] = compiled_fn
    return compiled_fn


# ============================================================================
# Public API
# ============================================================================

def recompute_w_u_fwd(k, v, beta, A, gk,
                      cu_seqlens=None, chunk_indices=None,
                      persistent=False, block_k=None, block_v=None):
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert chunk_indices is not None
        T_total  = k.shape[0]
        H, K     = k.shape[1], k.shape[2]
        V        = v.shape[2]
        BT       = A.shape[-1]
        num_seqs = cu_seqlens.shape[0] - 1
        total_nt = chunk_indices.shape[0] // 2
        ps = (Int32(num_seqs), Int32(T_total), Int32(H), Int32(K), Int32(V))
        cu_s = cu_seqlens
        ci_s = chunk_indices
    else:
        B, T, H, K = k.shape
        V    = v.shape[-1]
        BT   = A.shape[-1]
        NT   = (T + BT - 1) // BT
        total_nt = B * NT
        ps = (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V))
        global _dummy_cu_seqlens, _dummy_chunk_indices
        if _dummy_cu_seqlens is None or _dummy_cu_seqlens.device != k.device:
            _dummy_cu_seqlens = torch.zeros(2, dtype=torch.int32, device=k.device)
        if _dummy_chunk_indices is None or _dummy_chunk_indices.device != k.device:
            _dummy_chunk_indices = torch.zeros(2, dtype=torch.int32, device=k.device)
        cu_s = _dummy_cu_seqlens
        ci_s = _dummy_chunk_indices

    w  = torch.empty_like(k)
    u  = torch.empty_like(v)
    kg = torch.empty_like(k)

    compiled_fn = _compile_recompute_wu(
        H, K, V, chunk_size=BT, block_k=block_k, block_v=block_v,
        persistent=persistent, is_varlen=is_varlen,
    )
    compiled_fn(k, v, beta, A, gk, w, u, kg, cu_s, ci_s, ps, Int32(total_nt))
    return w, u, None, kg


# ============================================================================
# Reference
# ============================================================================

def recompute_w_u_fwd_ref(k, v, beta, A, gk):
    """Reference implementation supporting both [B,T,H,K] and [T_total,H,K] inputs."""
    if k.dim() == 4:
        B, T, H, K = k.shape
        V  = v.shape[-1]
        BT = A.shape[-1]
        NT = (T + BT - 1) // BT
        w  = torch.empty_like(k)
        u  = torch.empty_like(v)
        kg = torch.empty_like(k)
        for b in range(B):
            for h in range(H):
                for it in range(NT):
                    t0, t1 = it * BT, min((it + 1) * BT, T)
                    b_A    = A[b, t0:t1, h, :t1-t0].float()
                    b_beta = beta[b, t0:t1, h].float()
                    b_k    = k[b, t0:t1, h, :].float()
                    b_gk   = gk[b, t0:t1, h, :].float()
                    b_v    = v[b, t0:t1, h, :].float()
                    w[b, t0:t1, h, :]  = (b_A @ (b_k * b_beta[:, None] * 2.0**b_gk)).to(k.dtype)
                    u[b, t0:t1, h, :]  = (b_A @ (b_v * b_beta[:, None])).to(v.dtype)
                    b_gn = gk[b, t1-1, h, :].float()
                    kg[b, t0:t1, h, :] = (b_k * 2.0**(b_gn - b_gk)).to(k.dtype)
        return w, u, None, kg
    else:
        # varlen: k is [T_total, H, K]
        T_total, H, K = k.shape
        V  = v.shape[-1]
        BT = A.shape[-1]
        NT = (T_total + BT - 1) // BT
        w  = torch.empty_like(k)
        u  = torch.empty_like(v)
        kg = torch.empty_like(k)
        for h in range(H):
            for it in range(NT):
                t0, t1 = it * BT, min((it + 1) * BT, T_total)
                b_A    = A[t0:t1, h, :t1-t0].float()
                b_beta = beta[t0:t1, h].float()
                b_k    = k[t0:t1, h, :].float()
                b_gk   = gk[t0:t1, h, :].float()
                b_v    = v[t0:t1, h, :].float()
                w[t0:t1, h, :]  = (b_A @ (b_k * b_beta[:, None] * 2.0**b_gk)).to(k.dtype)
                u[t0:t1, h, :]  = (b_A @ (b_v * b_beta[:, None])).to(v.dtype)
                b_gn = gk[t1-1, h, :].float()
                kg[t0:t1, h, :] = (b_k * 2.0**(b_gn - b_gk)).to(k.dtype)
        return w, u, None, kg


def build_chunk_indices_wu(seq_lens, BT=64, device='cuda'):
    """Build chunk_indices [NT*2] for varlen recompute_wu."""
    pairs = []
    for seq_idx, sl in enumerate(seq_lens):
        for c in range((sl + BT - 1) // BT):
            pairs.extend([seq_idx, c])
    return torch.tensor(pairs, dtype=torch.int32, device=device)


# ============================================================================
# Test & Benchmark
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["correctness", "benchmark", "both"], default="both")
    parser.add_argument("--persistent", type=int, default=0)
    args = parser.parse_args()
    do_persistent = bool(args.persistent)

    print(f"persistent={do_persistent}")

    # ---- Non-varlen correctness ----
    print("\n=== Non-varlen correctness ===")
    B, T, H, K, V = 1, 128, 1, 64, 64
    BT = 64
    NT = T // BT
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16))
    gk = (-torch.abs(torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)) * 0.1).cumsum(dim=1)
    A = torch.tril(torch.randn(B, NT, H, BT, BT, device="cuda", dtype=torch.bfloat16) * 0.1).reshape(B, T, H, BT)

    w_ref, u_ref, _, kg_ref = recompute_w_u_fwd_ref(k, v, beta, A, gk)
    torch.cuda.synchronize()
    w, u, _, kg = recompute_w_u_fwd(k, v, beta, A, gk, persistent=do_persistent)
    torch.cuda.synchronize()
    dw  = (w.float()  - w_ref.float()).abs().max().item()
    du  = (u.float()  - u_ref.float()).abs().max().item()
    dkg = (kg.float() - kg_ref.float()).abs().max().item()
    ok  = dw < 1.0 and du < 1.0 and dkg < 1.0
    print(f"  K=64 V=64: w={dw:.6f} u={du:.6f} kg={dkg:.6f} {'PASS' if ok else 'FAIL'}")

    # Test K=128,V=128
    B2, T2, H2, K2, V2 = 1, 128, 1, 128, 128
    BT2 = 64; NT2 = T2 // BT2
    torch.manual_seed(43)
    k2 = torch.randn(B2, T2, H2, K2, device="cuda", dtype=torch.bfloat16) * 0.1
    v2 = torch.randn(B2, T2, H2, V2, device="cuda", dtype=torch.bfloat16) * 0.1
    beta2 = torch.sigmoid(torch.randn(B2, T2, H2, device="cuda", dtype=torch.bfloat16))
    gk2 = (-torch.abs(torch.randn(B2, T2, H2, K2, device="cuda", dtype=torch.float32)) * 0.1).cumsum(dim=1)
    A2 = torch.tril(torch.randn(B2, NT2, H2, BT2, BT2, device="cuda", dtype=torch.bfloat16) * 0.1).reshape(B2, T2, H2, BT2)
    w2_ref, u2_ref, _, kg2_ref = recompute_w_u_fwd_ref(k2, v2, beta2, A2, gk2)
    w2, u2, _, kg2 = recompute_w_u_fwd(k2, v2, beta2, A2, gk2, persistent=do_persistent)
    torch.cuda.synchronize()
    dw2 = (w2.float() - w2_ref.float()).abs().max().item()
    du2 = (u2.float() - u2_ref.float()).abs().max().item()
    dkg2 = (kg2.float() - kg2_ref.float()).abs().max().item()
    ok2 = dw2 < 1.0 and du2 < 1.0 and dkg2 < 1.0
    print(f"  K=128 V=128: w={dw2:.6f} u={du2:.6f} kg={dkg2:.6f} {'PASS' if ok2 else 'FAIL'}")

    all_ok = ok and ok2

    # ---- Varlen correctness ----
    print("\n=== Varlen correctness ===")
    # Use multiples of BT3 so chunks don't cross seq boundaries
    test_seqlens = [[64], [128], [64, 128], [128, 64], [64, 64], [128, 128]]
    BT3, H3, K3, V3 = 64, 4, 128, 128
    for seq_lens in test_seqlens:
        T_total = sum(seq_lens)
        cu_list = [0]
        for sl in seq_lens:
            cu_list.append(cu_list[-1] + sl)
        cu_t = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
        ci_t = build_chunk_indices_wu(seq_lens, BT=BT3, device="cuda")

        torch.manual_seed(77)
        k_f    = torch.randn(T_total, H3, K3, device="cuda", dtype=torch.bfloat16) * 0.1
        v_f    = torch.randn(T_total, H3, V3, device="cuda", dtype=torch.bfloat16) * 0.1
        beta_f = torch.sigmoid(torch.randn(T_total, H3, device="cuda", dtype=torch.bfloat16))
        gk_f   = (-torch.abs(torch.randn(T_total, H3, K3, device="cuda", dtype=torch.float32)) * 0.1).cumsum(dim=0)
        NT_total = T_total // BT3
        A_f = (
            torch.tril(torch.randn(NT_total, H3, BT3, BT3, device="cuda", dtype=torch.bfloat16) * 0.1)
            .permute(0, 2, 1, 3).contiguous().reshape(T_total, H3, BT3)
        )

        w_rf = torch.empty_like(k_f); u_rf = torch.empty_like(v_f); kg_rf = torch.empty_like(k_f)
        for si, sl in enumerate(seq_lens):
            s, e = cu_list[si], cu_list[si+1]
            wr, ur, _, kgr = recompute_w_u_fwd_ref(k_f[s:e], v_f[s:e], beta_f[s:e], A_f[s:e], gk_f[s:e])
            w_rf[s:e] = wr; u_rf[s:e] = ur; kg_rf[s:e] = kgr

        wf, uf, _, kgf = recompute_w_u_fwd(
            k_f, v_f, beta_f, A_f, gk_f,
            cu_seqlens=cu_t, chunk_indices=ci_t, persistent=do_persistent,
        )
        torch.cuda.synchronize()
        dwf  = (wf.float()  - w_rf.float()).abs().max().item()
        duf  = (uf.float()  - u_rf.float()).abs().max().item()
        dkgf = (kgf.float() - kg_rf.float()).abs().max().item()
        okf  = dwf < 1.0 and duf < 1.0 and dkgf < 1.0
        all_ok = all_ok and okf
        print(f"  seq_lens={seq_lens}: w={dwf:.6f} u={duf:.6f} kg={dkgf:.6f} {'PASS' if okf else 'FAIL'}")

    # ---- Benchmark ----
    if all_ok and args.test in ("benchmark", "both"):
        print("\n=== Benchmark (B=4, T=4096, H=64, K=128, V=128) ===")
        Bb, Tb, Hb, Kb, Vb = 4, 4096, 64, 128, 128
        BTb = 64; NTb = Tb // BTb
        torch.manual_seed(999)
        kb    = torch.randn(Bb, Tb, Hb, Kb, device="cuda", dtype=torch.bfloat16) * 0.1
        vb    = torch.randn(Bb, Tb, Hb, Vb, device="cuda", dtype=torch.bfloat16) * 0.1
        betab = torch.sigmoid(torch.randn(Bb, Tb, Hb, device="cuda", dtype=torch.bfloat16))
        gkb   = (-torch.abs(torch.randn(Bb, Tb, Hb, Kb, device="cuda", dtype=torch.float32)) * 0.1).cumsum(dim=1)
        Ab    = torch.tril(torch.randn(Bb, NTb, Hb, BTb, BTb, device="cuda", dtype=torch.bfloat16) * 0.1).reshape(Bb, Tb, Hb, BTb)

        n_iter = 20
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        for mode, p in [("non-persistent (occ=2, 1-stage)", False), ("persistent   (occ=2, 2-stage A)", True)]:
            for _ in range(3):
                recompute_w_u_fwd(kb, vb, betab, Ab, gkb, persistent=p)
            torch.cuda.synchronize()
            start.record()
            for _ in range(n_iter):
                recompute_w_u_fwd(kb, vb, betab, Ab, gkb, persistent=p)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / n_iter
            print(f"  CuTeDSL {mode}: {ms:.3f} ms")

        try:
            from fla.ops.kda.wy_fast import recompute_w_u_fwd as fla_recompute
            for _ in range(3):
                fla_recompute(kb, vb, betab, Ab, gk=gkb)
            torch.cuda.synchronize()
            start.record()
            for _ in range(n_iter):
                fla_recompute(kb, vb, betab, Ab, gk=gkb)
            end.record()
            torch.cuda.synchronize()
            fla_ms = start.elapsed_time(end) / n_iter
            print(f"  FLA Triton:                      {fla_ms:.3f} ms")
        except Exception as e:
            print(f"  FLA not available: {e}")


if __name__ == "__main__":
    main()
