# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
chunk_gla_fwd_o — CuTe DSL Implementation for Blackwell SM100

Computes output O for chunkwise gated linear attention in KDA forward pass:

    O = scale * (q ⊙ 2^g) @ h  +  tril(Aqk) @ v_new

Inputs:
  q:  [B, T, H, K]           bf16  — query
  v:  [B, T, H, V]           bf16  — value (v_new from delta-rule)
  g:  [B, T, H, K]           bf16  — cumulative gate (log2 domain)
  h:  [NT_total, H, K, V]    bf16  — inter-chunk recurrent state
  A:  [B, T, H, BT]          bf16  — intra-chunk attention matrix (Aqk)

Output:
  o:  [B, T, H, V]           bf16

For KDA: K=V=128, BT=64, use_exp2=True always.
Scale is folded into the gating: qg = q * 2^g * scale.

Kernel design (TMEM A-operand approach):
  Grid: (ceil(V/BV), NT, B*H)
  6 warps = 192 threads

  Warp specialization:
    Warps 0-3 (CUDA):
        - Read q, g from epilog SMEM, compute qg = q * exp2(g) * scale
        - R2T write qg to TMEM (QG A-operand)
        - Read A from epilog SMEM, apply causal mask
        - R2T write A_masked to TMEM (AM A-operand)
    Warp 4 (MMA):
        - QH MMA: qg(TMEM) × h(SMEM) → acc(TMEM)
        - AV MMA: am(TMEM) × v(SMEM) → acc(TMEM, ACCUMULATE)
        - tcgen05.st: acc → sO
        - TMA S2G: sO → GMEM
    Warp 5 (Load):
        - TMA G2S: q, g, A → epilog SMEM
        - TMA G2S: h, v → MMA B-operand SMEM

  TMEM layout:
    ACC:   (BT, BV) fp32 — shared accumulator for QH and AV
    QG_A:  (BT, BK) bf16 — A-operand for QH MMA
    AM_A:  (BT, BT) bf16 — A-operand for AV MMA

  Pipeline:
    Load→CUDA: q, g, A  (PipelineTmaAsync, 2-stage)
    Load→MMA:  h, v      (PipelineTmaUmma, 2-stage)
    CUDA→MMA:  qg_ready  (PipelineAsyncUmma, 1-stage)
    CUDA→MMA:  am_ready  (PipelineAsyncUmma, 1-stage)
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
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32

PRINT_DEBUG = False

LN2 = 0.6931471805599453
RCP_LN2 = 1.4426950408889634


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class ChunkGlaFwdO:
    """
    CuTE DSL kernel for chunk_gla_fwd_o:
      o = scale * (q ⊙ 2^g) @ h  +  tril(Aqk) @ v_new

    Targeting KDA forward: K=V=128, BT=64, use_exp2=True.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        scale: float = 1.0,
        is_varlen: bool = False,
        BK: int = 128,
        BV: int = 128,
    ):
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.scale = scale
        self.is_varlen = is_varlen

        self.BT = chunk_size    # 64
        self.BK = BK            # 128
        self.BV = BV            # 128

        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.threads_per_cta = self.threads_per_warp * 6  # 192

        self.num_regs_cuda = 232
        self.num_regs_others = 40

        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        # Number of K tiles for QH MMA (K=BK for KDA)
        self.num_k_tiles = (head_dim_k + BK - 1) // BK  # 1

        # Pipeline stages
        self.q_stage = 2
        self.g_stage = 2
        self.h_stage = 2
        self.v_stage = 2
        self.a_stage = 2
        self.acc_stage = 1

        # MMA tiler shapes:
        # QH: qg(BT, BK) @ h(BK, BV) → (BT, BV)
        self.qh_mma_tiler = (self.BT, self.BV, self.BK)
        # AV: A(BT, BT) @ v(BT, BV) → (BT, BV)
        self.av_mma_tiler = (self.BT, self.BV, self.BT)

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta,
        )
        self.buffer_align_bytes = 128

    @staticmethod
    def _plan_tmem_offsets(
        qh_tiled_mma, tile_qh,
        qg_tmem_layout, am_tmem_layout,
        acc_stages,
    ):
        """Plan TMEM offsets for ACC, QG A-operand, AM A-operand."""
        SM100_TMEM_CAPACITY_COLS = 512

        # ACC: (BT, BV) FP32
        acc_shape = qh_tiled_mma.partition_shape_C(tile_qh[:2])
        acc_fake = qh_tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))
        num_acc = tcgen05.find_tmem_tensor_col_offset(acc_fake)

        # QG A-operand: (BT, BK) BF16
        tCrQG_fake = qh_tiled_mma.make_fragment_A(qg_tmem_layout.outer.shape)
        num_qg = tcgen05.find_tmem_tensor_col_offset(tCrQG_fake)

        # AM A-operand: (BT, BT) BF16
        # Use av_tiled_mma for this — passed in via am_tmem_layout

        acc_off = 0
        qg_off = acc_off + num_acc
        am_off = qg_off + num_qg

        # For AM, we need to find its column count from the layout
        # We'll compute it at the call site and pass it in
        # For now, estimate as num_qg // 2 (BT=64 vs BK=128)
        total_tmp = am_off + num_qg  # conservative estimate
        total = 1
        while total < total_tmp:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS, f"TMEM overflow: {total} > {SM100_TMEM_CAPACITY_COLS}"
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"  TMEM: ACC={num_acc}@{acc_off}, QG={num_qg}@{qg_off}, AM@{am_off}, total={total}")
        return acc_off, qg_off, am_off, total

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,           # [B, T, H, K]
        v_ptr: cute.Pointer,           # [B, T, H, V]
        g_ptr: cute.Pointer,           # [B, T, H, K]
        h_ptr: cute.Pointer,           # [NT_total, H, K, V]
        o_ptr: cute.Pointer,           # [B, T, H, V]
        A_ptr: cute.Pointer,           # [B, T, H, BT]
        cu_seqlens_ptr: cute.Pointer,  # [N+1] int32 (unused for non-varlen)
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        stream,
    ):
        B, T, H, K, V = problem_size
        BT = self.BT

        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
        else:
            data_B = B
        NT = (T + BT - 1) // BT

        # ===================== GMEM layouts =====================
        # q, g: row-major (T, K, (H, data_B)) for TMA epilog load
        qg_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        q = cute.make_tensor(q_ptr, qg_layout)
        g = cute.make_tensor(g_ptr, qg_layout)

        # v, o: row-major (T, V, (H, data_B))
        v_layout = cute.make_layout(
            (T, V, (H, data_B)),
            stride=(H * V, 1, (V, T * H * V)),
        )
        v = cute.make_tensor(v_ptr, v_layout)
        o = cute.make_tensor(o_ptr, v_layout)

        # v transposed for MMA B TMA: (V, T, (H, data_B)) — V contiguous
        v_T_layout = cute.make_layout(
            (V, T, (H, data_B)),
            stride=(1, H * V, (V, T * H * V)),
        )
        v_T = cute.make_tensor(v_ptr, v_T_layout)

        # h: stored as [NT_total, H, K, V] — V contiguous
        # Transposed view for MMA B TMA: (V, K, (H, B*NT)) — V contiguous
        h_T_layout = cute.make_layout(
            (V, K, (H, B * NT)),
            stride=(1, V, (K * V, H * K * V)),
        )
        h_T = cute.make_tensor(h_ptr, h_T_layout)

        # A: (T, BT, (H, data_B))
        a_layout = cute.make_layout(
            (T, BT, (H, data_B)),
            stride=(H * BT, 1, (BT, T * H * BT)),
        )
        A = cute.make_tensor(A_ptr, a_layout)

        # ===================== MMA setup =====================
        # QH MMA: A=qg from TMEM, B=h from SMEM
        # B is MN-major because h_T GMEM has V(=N) contiguous
        qh_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,    # A: K-major (TMEM requires K-major)
            tcgen05.OperandMajorMode.MN,   # B: MN-major (V contiguous in GMEM)
            self.acc_dtype,
            self.cta_group,
            self.qh_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,    # A from TMEM
        )

        # AV MMA: A=A_masked from TMEM, B=v from SMEM
        # B is MN-major because v_T GMEM has V(=N) contiguous
        av_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,    # A: K-major (TMEM requires K-major)
            tcgen05.OperandMajorMode.MN,   # B: MN-major (V contiguous in GMEM)
            self.acc_dtype,
            self.cta_group,
            self.av_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,    # A from TMEM
        )

        # ===================== TMEM layouts =====================
        # QG A-operand TMEM layout
        qg_tmem_layout = sm100_utils.make_smem_layout_a(
            qh_tiled_mma, self.qh_mma_tiler, self.io_dtype, 1,
        )
        # AM A-operand TMEM layout
        am_tmem_layout = sm100_utils.make_smem_layout_a(
            av_tiled_mma, self.av_mma_tiler, self.io_dtype, 1,
        )

        # ===================== TMEM offsets =====================
        (self.tmem_acc_off, self.tmem_qg_off, self.tmem_am_off, self.tmem_total) = \
            self._plan_tmem_offsets(
                qh_tiled_mma, self.qh_mma_tiler,
                qg_tmem_layout, am_tmem_layout,
                self.acc_stage,
            )

        # ===================== SMEM layouts =====================
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        # Epilog SMEM for q, g (ROW_MAJOR BT×BK for CUDA warp reading)
        q_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK), self.q_stage,
        )
        g_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK), self.g_stage,
        )
        # Epilog SMEM for A (ROW_MAJOR BT×BT)
        a_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BT), self.a_stage,
        )
        # MMA B-operand SMEM for h (QH MMA B)
        h_smem_staged = sm100_utils.make_smem_layout_b(
            qh_tiled_mma, self.qh_mma_tiler, self.io_dtype, self.h_stage,
        )
        # MMA B-operand SMEM for v (AV MMA B)
        v_smem_staged = sm100_utils.make_smem_layout_b(
            av_tiled_mma, self.av_mma_tiler, self.io_dtype, self.v_stage,
        )
        # Output epilog for TMA store (ROW_MAJOR BT×BV)
        o_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BV), 1,
        )

        # ===================== Cluster layout =====================
        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qh_tiled_mma.thr_id.shape,),
        )

        # ===================== TMA descriptors =====================
        # q, g, A: non-MMA epilog TMA (simple 2D tiles)
        q_epi_smem = cute.select(q_epi_staged, mode=[0, 1])
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op, q, q_epi_smem, (self.BT, self.BK),
        )
        g_epi_smem = cute.select(g_epi_staged, mode=[0, 1])
        tma_atom_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            tma_load_op, g, g_epi_smem, (self.BT, self.BK),
        )
        a_epi_smem = cute.select(a_epi_staged, mode=[0, 1])
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op, A, a_epi_smem, (self.BT, self.BT),
        )

        # h: MMA B TMA (transposed view)
        h_smem_1 = cute.select(h_smem_staged, mode=[0, 1, 2])
        tma_atom_h, tma_tensor_h = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, h_T, h_smem_1, self.qh_mma_tiler, qh_tiled_mma,
            cluster_layout.shape,
        )

        # v: MMA B TMA (transposed view)
        v_smem_1 = cute.select(v_smem_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, v_T, v_smem_1, self.av_mma_tiler, av_tiled_mma,
            cluster_layout.shape,
        )

        # O: TMA store
        o_epi_smem = cute.select(o_epi_staged, mode=[0, 1])
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op, o, o_epi_smem, (self.BT, self.BV),
        )

        # ===================== TMA byte counts =====================
        self.tma_bytes_q = cute.size_in_bytes(self.io_dtype, q_epi_smem)
        self.tma_bytes_g = cute.size_in_bytes(self.io_dtype, g_epi_smem)
        self.tma_bytes_h = cute.size_in_bytes(self.io_dtype, h_smem_1)
        self.tma_bytes_v = cute.size_in_bytes(self.io_dtype, v_smem_1)
        self.tma_bytes_a = cute.size_in_bytes(self.io_dtype, a_epi_smem)

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            load_q_mbar: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_g_mbar: cute.struct.MemRange[Int64, self.g_stage * 2]
            load_h_mbar: cute.struct.MemRange[Int64, self.h_stage * 2]
            load_v_mbar: cute.struct.MemRange[Int64, self.v_stage * 2]
            load_a_mbar: cute.struct.MemRange[Int64, self.a_stage * 2]
            qg_mbar: cute.struct.MemRange[Int64, 1 * 2]   # CUDA→MMA: qg ready
            am_mbar: cute.struct.MemRange[Int64, 1 * 2]   # CUDA→MMA: am ready
            acc_done_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # MMA→CUDA: acc done
            o_ready_mbar: cute.struct.MemRange[Int64, 1 * 2]   # CUDA→Load: o ready

            sQ_epi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_epi_staged)],
                self.buffer_align_bytes,
            ]
            sG_epi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(g_epi_staged)],
                self.buffer_align_bytes,
            ]
            sA_epi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(a_epi_staged)],
                self.buffer_align_bytes,
            ]
            sH: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(h_smem_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_smem_staged)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(o_epi_staged)],
                self.buffer_align_bytes,
            ]
            tmem_holding_buf: Int32

        # ===================== AM coord MMA =====================
        # Helper MMA for AM (BT, BT) tile — used only for T2R coordinate mapping
        am_coord_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            (self.BT, self.BT),
            tcgen05.OperandSource.TMEM,
        )

        # ===================== Grid =====================
        num_v_tiles = (V + self.BV - 1) // self.BV
        grid = (num_v_tiles, NT, B * H)

        self.shared_storage = SharedStorage

        # ===================== Launch =====================
        self.kernel(
            qh_tiled_mma, av_tiled_mma, am_coord_mma,
            tma_atom_q, tma_tensor_q,
            tma_atom_g, tma_tensor_g,
            tma_atom_h, tma_tensor_h,
            tma_atom_v, tma_tensor_v,
            tma_atom_a, tma_tensor_a,
            tma_atom_o, tma_tensor_o,
            q_epi_staged, g_epi_staged, a_epi_staged,
            h_smem_staged, v_smem_staged,
            o_epi_staged,
            qg_tmem_layout, am_tmem_layout,
            problem_size,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        qh_tiled_mma, av_tiled_mma, am_coord_mma,
        tma_atom_q, tma_tensor_q,
        tma_atom_g, tma_tensor_g,
        tma_atom_h, tma_tensor_h,
        tma_atom_v, tma_tensor_v,
        tma_atom_a, tma_tensor_a,
        tma_atom_o, tma_tensor_o,
        q_epi_staged, g_epi_staged, a_epi_staged,
        h_smem_staged, v_smem_staged,
        o_epi_staged,
        qg_tmem_layout, am_tmem_layout,
        problem_size,
    ):
        B, T, H, K, V = problem_size
        BT = self.BT

        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
        else:
            data_B = B
        NT = (T + BT - 1) // BT

        # Grid indices
        i_v = cute.arch.block_idx()[0]
        i_t = cute.arch.block_idx()[1]
        i_bh = cute.arch.block_idx()[2]
        i_b = i_bh // H
        i_h = i_bh % H
        bos = i_b * T
        i_tg = i_b * NT + i_t

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # ---- SMEM ----
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ_epi = storage.sQ_epi.get_tensor(q_epi_staged.outer, swizzle=q_epi_staged.inner)
        sG_epi = storage.sG_epi.get_tensor(g_epi_staged.outer, swizzle=g_epi_staged.inner)
        sA_epi = storage.sA_epi.get_tensor(a_epi_staged.outer, swizzle=a_epi_staged.inner)
        sH = storage.sH.get_tensor(h_smem_staged.outer, swizzle=h_smem_staged.inner)
        sV = storage.sV.get_tensor(v_smem_staged.outer, swizzle=v_smem_staged.inner)
        sO = storage.sO.get_tensor(o_epi_staged.outer, swizzle=o_epi_staged.inner)

        # ---- TMEM ----
        tmem_alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # ---- TMEM tensors ----
        # ACC: (BT, BV) fp32
        acc_shape = qh_tiled_mma.partition_shape_C(self.qh_mma_tiler[:2])
        tCtAcc_fake = qh_tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stage))
        tCtAcc = cute.make_tensor(tmem_ptr + self.tmem_acc_off, tCtAcc_fake.layout)

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"acc_shape: {acc_shape}")
            print(f"tCtAcc: {tCtAcc}")
            print(f"tCtAcc_fake: {tCtAcc_fake}")

        # QG A-operand: TMEM fragment (BF16) - use FP32 ptr + offset, then recast
        tCrQG = qh_tiled_mma.make_fragment_A(qg_tmem_layout.outer.shape)
        tCrQG_tmem = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_qg_off, dtype=self.io_dtype),
            tCrQG.layout,
        )

        # AM A-operand: TMEM fragment (BF16) - use FP32 ptr + offset, then recast
        tCrAM = av_tiled_mma.make_fragment_A(am_tmem_layout.outer.shape)
        tCrAM_tmem = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_am_off, dtype=self.io_dtype),
            tCrAM.layout,
        )

        # ---- MMA B fragments (from SMEM) ----
        tCrH_B = qh_tiled_mma.make_fragment_B(sH)
        tCrV_B = av_tiled_mma.make_fragment_B(sV)

        # ---- TMA partitions for q, g, A, O (epilog style) ----
        gQ = tma_tensor_q[None, None, (i_h, i_b)]
        _, bSG_sQ, bSG_gQ = self._epilog_partition(
            tma_atom_q, gQ, (self.BT, self.BK), sQ_epi,
        )
        gG = tma_tensor_g[None, None, (i_h, i_b)]
        _, bSG_sG, bSG_gG = self._epilog_partition(
            tma_atom_g, gG, (self.BT, self.BK), sG_epi,
        )
        gA = tma_tensor_a[None, None, (i_h, i_b)]
        _, bSG_sA, bSG_gA = self._epilog_partition(
            tma_atom_a, gA, (self.BT, self.BT), sA_epi,
        )
        gO = tma_tensor_o[None, None, (i_h, i_b)]
        _, bSG_sO, bSG_gO = self._epilog_partition(
            tma_atom_o, gO, (self.BT, self.BV), sO,
        )

        # ---- Pipelines ----
        num_cuda_threads = self.threads_per_warp * len(self.cuda_warp_ids)

        load_q_P, load_q_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_cuda_threads),
            tx_count=self.tma_bytes_q,
            barrier_storage=storage.load_q_mbar.data_ptr(),
        ).make_participants()

        load_g_P, load_g_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.g_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_cuda_threads),
            tx_count=self.tma_bytes_g,
            barrier_storage=storage.load_g_mbar.data_ptr(),
        ).make_participants()

        load_a_P, load_a_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.a_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_cuda_threads),
            tx_count=self.tma_bytes_a,
            barrier_storage=storage.load_a_mbar.data_ptr(),
        ).make_participants()

        load_h_P, load_h_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.h_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_bytes_h,
            barrier_storage=storage.load_h_mbar.data_ptr(),
        ).make_participants()

        load_v_P, load_v_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_bytes_v,
            barrier_storage=storage.load_v_mbar.data_ptr(),
        ).make_participants()

        qg_P, qg_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(num_cuda_threads),
            consumer_group=make_thread_cooperative_group(1),
            barrier_storage=storage.qg_mbar.data_ptr(),
        ).make_participants()

        am_P, am_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(num_cuda_threads),
            consumer_group=make_thread_cooperative_group(1),
            barrier_storage=storage.am_mbar.data_ptr(),
        ).make_participants()

        acc_done_P, acc_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_cuda_threads),
            barrier_storage=storage.acc_done_mbar.data_ptr(),
        ).make_participants()

        o_ready_P, o_ready_C = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(num_cuda_threads),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.o_ready_mbar.data_ptr(),
        ).make_participants()

        # =====================================================================
        # LOAD WARP
        # =====================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_g)
            cpasync.prefetch_descriptor(tma_atom_h)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_a)

            # TMA B partitions for h and v (with batch/head fixed)
            # h_T: (V, K, (H, B*NT)) → after fixing (i_h, i_tg), remains (V_tiles, K_tiles)
            tHsH, tHgH = self._tma_partition_B(
                tma_atom_h, tma_tensor_h, sH, self.qh_mma_tiler, qh_tiled_mma, i_tg, i_h,
            )
            # v_T: (V, T, (H, B)) → after fixing (i_h, i_b), remains (V_tiles, T_tiles)
            tVsV, tVgV = self._tma_partition_B(
                tma_atom_v, tma_tensor_v, sV, self.av_mma_tiler, av_tiled_mma, i_b, i_h,
            )

            # Load Q, G, H for each K tile (only 1 tile for KDA: K=BK=128)
            for i_k in cutlass.range(self.num_k_tiles, unroll_full=True):
                q_h = load_q_P.acquire_and_advance()
                cute.copy(atom=tma_atom_q,
                          src=bSG_gQ[(None, i_t, 0)],
                          dst=bSG_sQ[None, q_h.index],
                          tma_bar_ptr=q_h.barrier)

                g_h = load_g_P.acquire_and_advance()
                cute.copy(atom=tma_atom_g,
                          src=bSG_gG[(None, i_t, 0)],
                          dst=bSG_sG[None, g_h.index],
                          tma_bar_ptr=g_h.barrier)

                h_h = load_h_P.acquire_and_advance()
                cute.copy(atom=tma_atom_h,
                          src=tHgH[None, i_v, 0],
                          dst=tHsH[None, h_h.index],
                          tma_bar_ptr=h_h.barrier)

            # Load V and A (once, not per K tile)
            v_h = load_v_P.acquire_and_advance()
            cute.copy(atom=tma_atom_v,
                      src=tVgV[None, i_v, i_t],
                      dst=tVsV[None, v_h.index],
                      tma_bar_ptr=v_h.barrier)

            a_h = load_a_P.acquire_and_advance()
            cute.copy(atom=tma_atom_a,
                      src=bSG_gA[(None, i_t, 0)],
                      dst=bSG_sA[None, a_h.index],
                      tma_bar_ptr=a_h.barrier)

            # Wait for CUDA warps to write O to SMEM, then TMA store
            cpasync.prefetch_descriptor(tma_atom_o)
            o_h = o_ready_C.wait_and_advance()
            cute.copy(tma_atom_o, bSG_sO[None, 0], bSG_gO[(None, i_t, i_v)])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            o_h.release()

        # =====================================================================
        # MMA WARP
        # =====================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            # Phase 1: QH MMA — qg(TMEM) × h(SMEM) → acc(TMEM)
            qg_h = qg_C.wait_and_advance()

            for i_k in cutlass.range(self.num_k_tiles, unroll_full=True):
                h_h = load_h_C.wait_and_advance()

                for kp in cutlass.range(cute.size(tCrH_B, mode=[2]), unroll_full=True):
                    qh_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        cutlass.Boolean(kp != 0 or i_k != 0),
                    )
                    cute.gemm(
                        qh_tiled_mma,
                        tCtAcc[None, None, None, 0],
                        tCrQG_tmem[None, None, kp, 0],
                        tCrH_B[None, None, kp, h_h.index],
                        tCtAcc[None, None, None, 0],
                    )

                h_h.release()

            qg_h.release()

            # Phase 2: AV MMA — am(TMEM) × v(SMEM) → acc(TMEM, ACCUMULATE)
            am_h = am_C.wait_and_advance()
            v_h = load_v_C.wait_and_advance()

            for kp in cutlass.range(cute.size(tCrV_B, mode=[2]), unroll_full=True):
                av_tiled_mma.set(
                    tcgen05.Field.ACCUMULATE,
                    cutlass.Boolean(True),
                )
                cute.gemm(
                    av_tiled_mma,
                    tCtAcc[None, None, None, 0],
                    tCrAM_tmem[None, None, kp, 0],
                    tCrV_B[None, None, kp, v_h.index],
                    tCtAcc[None, None, None, 0],
                )

            am_h.release()
            v_h.release()

            # Phase 3: Signal ACC done to CUDA warps (they'll do T2R+R2S)
            acc_h = acc_done_P.acquire_and_advance()
            acc_h.commit()

        # =====================================================================
        # CUDA WARPS: Gating + Masking → TMEM
        # =====================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))
            scale_f32 = Float32(self.scale)

            # ----- T2R for ACC (FP32, BT×BV=64×128) — for QG coordinate mapping -----
            t2r_atom_acc = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tCtAcc_flat: {tCtAcc_flat}")
            fake_sQG = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.qh_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_acc = tcgen05.make_tmem_copy(t2r_atom_acc, tCtAcc_flat[(None, None, 0)])
            thr_t2r_acc = tiled_t2r_acc.get_slice(local_tidx)

            # QG identity tensor: (BT, BK) coords
            qg_tile = cute.dice(self.qh_mma_tiler, (1, 1, None))  # (BT, BK)
            cM_qg = cute.make_identity_tensor(qg_tile)
            tTR_cM_qg = thr_t2r_acc.partition_D(cM_qg)

            # QG R2T: bf16 registers → QG TMEM
            r2t_atom_qg = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(16), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_qg = tcgen05.make_tmem_copy(r2t_atom_qg, tCrQG_tmem)
            thr_r2t_qg = tiled_r2t_qg.get_slice(local_tidx)
            r2t_qg_shape = cute.slice_(thr_r2t_qg.partition_S(tCrQG_tmem).shape, (None, None, None, None, 0))
            tRT_tQG = thr_r2t_qg.partition_D(tCrQG_tmem)

            # Register tensors for QG computation
            tTR_rQG_fp32 = cute.make_rmem_tensor(thr_t2r_acc.partition_D(fake_sQG).shape, self.acc_dtype)
            tRT_rQG_bf16 = cute.make_rmem_tensor(r2t_qg_shape, self.io_dtype)

            # ----- T2R for AM coordinate mapping (FP32, BT×BT=64×64) -----
            # Create fake (BT, BT) FP32 TMEM accumulator via am_coord_mma
            fake_am_acc = am_coord_mma.make_fragment_C(
                am_coord_mma.partition_shape_C((self.BT, self.BT))
            )
            t2r_atom_am_coord = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            fake_am_flat = fake_am_acc[((None, None), 0, 0)]
            tiled_t2r_am = tcgen05.make_tmem_copy(t2r_atom_am_coord, fake_am_flat)
            thr_t2r_am = tiled_t2r_am.get_slice(local_tidx)

            # AM identity tensor: (BT, BT) coords
            am_tile = cute.dice(self.av_mma_tiler, (1, 1, None))  # (BT, BT)
            cM_am = cute.make_identity_tensor(am_tile)
            tTR_cM_am = thr_t2r_am.partition_D(cM_am)

            # AM R2T: bf16 registers → AM TMEM
            r2t_atom_am = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_am = tcgen05.make_tmem_copy(r2t_atom_am, tCrAM_tmem)
            thr_r2t_am = tiled_r2t_am.get_slice(local_tidx)
            r2t_am_shape = cute.slice_(thr_r2t_am.partition_S(tCrAM_tmem).shape, (None, None, None, None, 0))
            tRT_tAM = thr_r2t_am.partition_D(tCrAM_tmem)
            tRT_rAM = cute.make_rmem_tensor(r2t_am_shape, self.io_dtype)

            # Fake SMEM tensor for AM T2R destination sizing
            fake_sAM = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                am_tile,
            )
            tTR_sAM = thr_t2r_am.partition_D(fake_sAM)

            # ----- R2S: ACC T2R regs → sO (ROW_MAJOR, BT×BV) -----
            r2s_atom_o = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r_acc,
            )
            tiled_r2s_o = cute.make_tiled_copy_D(r2s_atom_o, tiled_t2r_acc)
            thr_r2s_o = tiled_r2s_o.get_slice(local_tidx)
            tRS_sO = thr_r2s_o.partition_D(sO)

            # ============ Compute QG: q * exp2(g) * scale ============
            for i_k in cutlass.range(self.num_k_tiles, unroll_full=True):
                q_h = load_q_C.wait_and_advance()
                g_h = load_g_C.wait_and_advance()

                # Read q, g using identity coords and compute qg
                for ei in cutlass.range_constexpr(cute.size(tTR_rQG_fp32)):
                    bt_coord, bk_coord = tTR_cM_qg[ei]
                    q_val = sQ_epi[(bt_coord, bk_coord, q_h.index)].to(self.acc_dtype)
                    g_val = sG_epi[(bt_coord, bk_coord, g_h.index)].to(self.acc_dtype)
                    tTR_rQG_fp32[ei] = q_val * cute.exp2(g_val) * scale_f32

                q_h.release()
                g_h.release()

                # Convert to BF16 and R2T to TMEM
                tRT_rQG_bf16.store(tTR_rQG_fp32.load().to(self.io_dtype))
                qg_h = qg_P.acquire_and_advance()
                cute.copy(tiled_r2t_qg, tRT_rQG_bf16, tRT_tQG[(None, None, None, None, 0)])
                cute.arch.fence_view_async_tmem_store()
                qg_h.commit()

            # ============ Compute AM: tril(A) ============
            a_h = load_a_C.wait_and_advance()

            # Read A, apply causal mask, write to AM TMEM
            for ei in cutlass.range_constexpr(cute.size(tRT_rAM)):
                row, col = tTR_cM_am[ei]
                if row >= col:
                    tRT_rAM[ei] = sA_epi[(row, col, a_h.index)]
                else:
                    tRT_rAM[ei] = Float32(0.0).to(self.io_dtype)

            a_h.release()

            am_h = am_P.acquire_and_advance()
            cute.copy(tiled_r2t_am, tRT_rAM, tRT_tAM[(None, None, None, None, 0)])
            cute.arch.fence_view_async_tmem_store()
            am_h.commit()

            # ============ Output Epilog: ACC → T2R → R2S → sO ============
            tTR_tAcc = thr_t2r_acc.partition_S(tCtAcc_flat)

            # Wait for MMA to finish writing ACC
            acc_h = acc_done_C.wait_and_advance()
            # T2R: read ACC TMEM → FP32 registers
            tTR_rAcc = cute.make_rmem_tensor(thr_t2r_acc.partition_D(fake_sQG).shape, self.acc_dtype)
            cute.copy(tiled_t2r_acc, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)
            cute.arch.fence_view_async_tmem_load()
            acc_h.release()

            # Convert FP32 → BF16
            tTR_rAcc_bf16 = cute.make_rmem_tensor(tTR_rAcc.shape, self.io_dtype)
            tTR_rAcc_bf16.store(tTR_rAcc.load().to(self.io_dtype))

            # Retile BF16 regs for R2S and copy to sO
            tRS_rO = tiled_r2s_o.retile(tTR_rAcc_bf16)
            if cutlass.const_expr(PRINT_DEBUG):
                print(f"tRS_rO: {tRS_rO}")

            o_h = o_ready_P.acquire_and_advance()
            cute.copy(tiled_r2s_o, tRS_rO, tRS_sO[(None, None, None, 0)])
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            o_h.commit()

        # ---- TMEM cleanup ----
        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr)

    @cute.jit
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, head_idx):
        """Partition B operand for TMA."""
        coord = (0, None, None)
        gX = cute.local_tile(
            tma_tensor, cute.slice_(tile_shape, coord), (None, None, (head_idx, batch_idx))
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1),
            cute.group_modes(smem, 0, 3), cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _epilog_partition(self, atom, gC_mnl, epi_tile, sC):
        """Partition for epilog TMA load/store."""
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), sC_g, gC_g,
        )
        return atom, bSG_sC, bSG_gC


# =====================================================================
# Reference implementation
# =====================================================================

def reference_chunk_gla_fwd_o(q, v, g, h, A, scale, chunk_size=64):
    """
    Pure PyTorch reference for chunk_gla_fwd_o_gk.

    Args:
        q: [B, T, H, K]
        v: [B, T, H, V] (v_new)
        g: [B, T, H, K] cumulative gate (log2 domain)
        h: [NT_total, H, K, V] recurrent state
        A: [B, T, H, BT] intra-chunk attention matrix (Aqk)
        scale: float
        chunk_size: int

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT

    o = torch.zeros_like(v)

    for b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            actual_bt = t_end - t_start

            for i_h in range(H):
                q_chunk = q[b, t_start:t_end, i_h, :]
                g_chunk = g[b, t_start:t_end, i_h, :].float()
                qg = (q_chunk.float() * (2.0 ** g_chunk)).to(q.dtype)

                i_tg = b * NT + i_t
                h_state = h[i_tg, i_h, :, :]

                o_inter = scale * (qg.float() @ h_state.float())

                A_chunk = A[b, t_start:t_end, i_h, :actual_bt]
                mask = torch.tril(torch.ones(actual_bt, actual_bt, device=A.device))
                A_masked = (A_chunk * mask).to(v.dtype)

                v_chunk = v[b, t_start:t_end, i_h, :]
                o_intra = A_masked.float() @ v_chunk.float()

                o[b, t_start:t_end, i_h, :] = (o_inter + o_intra).to(o.dtype)

    return o


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Chunk GLA FWD O kernel test")
    parser.add_argument("--test", type=str, default="correctness",
                        choices=["correctness", "benchmark", "both"])
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    if args.scale is None:
        args.scale = args.K ** -0.5

    B, T, H, K, V = args.B, args.T, args.H, args.K, args.V
    BT = args.chunk_size
    scale = args.scale
    NT = (T + BT - 1) // BT
    dtype, device = torch.bfloat16, "cuda"

    print(f"Config: B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}, scale={scale:.4f}")
    print(f"  Chunks per seq: {NT}, Total chunks: {B*NT}")

    if args.test in ("correctness", "both"):
        print("\n=== Correctness Test ===")
        torch.manual_seed(42)

        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        g = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
        h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
        A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

        o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

        try:
            sys.path.insert(0, "/ossfs/workspace/flash-linear-attention")
            from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
            o_triton = chunk_gla_fwd_o_gk(
                q=q, v=v, g=g, A=A, h=h,
                scale=scale, chunk_size=BT, use_exp2=True,
            )
            max_diff = (o_ref.float() - o_triton.float()).abs().max().item()
            print(f"  Triton vs Reference: max_diff = {max_diff:.6f}")
        except Exception as e:
            print(f"  Triton not available: {e}")

        try:
            stream = cutlass_torch.default_stream()
            kernel = ChunkGlaFwdO(
                chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale,
            )
            q_cute = from_dlpack(q.detach())
            v_cute = from_dlpack(v.detach())
            g_cute = from_dlpack(g.detach())
            h_cute = from_dlpack(h.detach())
            A_cute = from_dlpack(A.detach())
            o_out = torch.zeros_like(v)
            o_cute = from_dlpack(o_out.detach())
            cu = torch.zeros(2, dtype=torch.int32, device=device)
            cu_cute = from_dlpack(cu.detach())
            ps = (B, T, H, K, V)

            compiled = cute.compile(
                kernel,
                q_cute.iterator, v_cute.iterator, g_cute.iterator,
                h_cute.iterator, o_cute.iterator, A_cute.iterator,
                cu_cute.iterator, ps, stream,
                options="--generate-line-info --ptxas-options '--verbose'",
            )
            compiled(
                q_cute.iterator, v_cute.iterator, g_cute.iterator,
                h_cute.iterator, o_cute.iterator, A_cute.iterator,
                cu_cute.iterator, ps, stream,
            )
            torch.cuda.synchronize()

            max_diff = (o_ref.float() - o_out.float()).abs().max().item()
            print(f"  CuTE DSL vs Reference: max_diff = {max_diff:.6f}")
            if max_diff < 0.02:
                print("  PASSED!")
            else:
                print("  FAILED!")
                for b_idx in range(min(1, B)):
                    for h_idx in range(min(1, H)):
                        print(f"  [B={b_idx}, H={h_idx}] ref[:4,:8]:\n{o_ref[b_idx, :4, h_idx, :8]}")
                        print(f"  [B={b_idx}, H={h_idx}] out[:4,:8]:\n{o_out[b_idx, :4, h_idx, :8]}")

        except Exception as e:
            import traceback
            print(f"  CuTE DSL failed: {e}")
            traceback.print_exc()

    if args.test in ("benchmark", "both"):
        print("\n=== Benchmark ===")
        torch.manual_seed(42)
        for bench_T in [1024, 2048, 4096]:
            bench_NT = (bench_T + BT - 1) // BT
            q = torch.randn(B, bench_T, H, K, dtype=dtype, device=device)
            v = torch.randn(B, bench_T, H, V, dtype=dtype, device=device)
            g = torch.randn(B, bench_T, H, K, dtype=dtype, device=device) * 0.1
            h = torch.randn(B * bench_NT, H, K, V, dtype=dtype, device=device) * 0.01
            A = torch.randn(B, bench_T, H, BT, dtype=dtype, device=device) * 0.1

            try:
                sys.path.insert(0, "/ossfs/workspace/flash-linear-attention")
                from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
                for _ in range(10):
                    chunk_gla_fwd_o_gk(q=q, v=v, g=g, A=A, h=h,
                                       scale=scale, chunk_size=BT, use_exp2=True)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                N_iters = 100
                start.record()
                for _ in range(N_iters):
                    chunk_gla_fwd_o_gk(q=q, v=v, g=g, A=A, h=h,
                                       scale=scale, chunk_size=BT, use_exp2=True)
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / N_iters
                print(f"  Triton T={bench_T}: {ms:.3f} ms")
            except Exception as e:
                print(f"  Triton benchmark failed: {e}")


if __name__ == "__main__":
    main()
