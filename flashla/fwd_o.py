# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
chunk_gla_fwd_o — CuTe DSL Implementation for Blackwell SM100

Computes output O for chunkwise gated linear attention in KDA forward pass:

    O = scale * (q ⊙ 2^g) @ h  +  tril(Aqk) @ v_new

Inputs:
  q:  [B, T, H, K]           bf16  — query
  v:  [B, T, H, V]           bf16  — value (v_new from delta-rule)
  g:  [B, T, H, K]           fp32  — cumulative gate (log2 domain)
  h:  [NT_total, H, K, V]    bf16  — inter-chunk recurrent state
  A:  [B, T, H, BT]          bf16  — intra-chunk attention matrix (Aqk)

Output:
  o:  [B, T, H, V]           bf16

For KDA: K=V=128, BT=64, use_exp2=True always.
Scale is folded into the gating: qg = q * 2^g * scale.

Kernel design (TMEM A-operand approach):
  Grid: (ceil(V/BV), NT, B*H)
  8 warps = 256 threads (occ=2 enabled)

  Warp specialization:
    Warps 0-3 (CUDA):
        - Read q, g from epilog SMEM, compute qg = q * exp2(g) * scale
        - R2T write qg to TMEM (QG A-operand)
        - Read A from epilog SMEM, apply causal mask
        - R2T write A_masked to TMEM (AM A-operand)
    Warp 4 (MMA):
        - QH MMA: qg(TMEM) × h(SMEM) → acc(TMEM)
        - AV MMA: am(TMEM) × v(SMEM) → acc(TMEM, ACCUMULATE)
    Warp 5 (Load):
        - TMA G2S: q, g, A → epilog SMEM
        - TMA G2S: h, v → MMA B-operand SMEM
    Warp 6 (Store):
        - TMA S2G: sO → GMEM
    Warp 7 (Empty):
        - Required for warp group register redistribution

  TMEM layout:
    ACC:   (BT, BV) fp32 — shared accumulator for QH and AV
    QG_A:  (BT, BK) bf16 — A-operand for QH MMA
    AM_A:  (BT, BT) bf16 — A-operand for AV MMA

  Pipeline:
    Load→CUDA: q, g, A  (PipelineTmaAsync, 1-stage)
    Load→MMA:  h, v      (PipelineTmaUmma, 1-stage)
    CUDA→MMA:  qg_ready  (PipelineAsyncUmma, 1-stage)
    CUDA→MMA:  am_ready  (PipelineAsyncUmma, 1-stage)
    MMA→CUDA:  acc_done  (PipelineUmmaAsync, 1-stage)
    CUDA→Store: o_ready  (PipelineAsync, 1-stage)

  Output epilog:
    CUDA warps: T2R (ACC TMEM → FP32 regs) → FP32→BF16 → R2S (regs → sO)
    Load warp:  TMA store sO → GMEM
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
PRINT_SMEM_DEBUG = False  # Print SMEM contents after TMA loads for non-aligned varlen debug

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
        g_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        scale: float = 1.0,
        is_varlen: bool = False,
        BK: int = 128,
        BV: int = 128,
        min_occupancy: int = 2,
        persistent: bool = True,
    ):
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.g_dtype = g_dtype
        self.scale = scale
        self.is_varlen = is_varlen
        self.persistent = persistent

        self.BT = chunk_size    # 64
        self.BK = BK            # 128
        self.BV = BV            # 128

        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        self.empty_warp_id = 7
        self.threads_per_cta = self.threads_per_warp * 8  # 256

        # Register allocation for occ=2:
        # Per CTA: 4×208×32 + 4×40×32 = 31,744 ≤ 32,768
        self.num_regs_cuda = 208
        self.num_regs_others = 40
        self.min_occupancy = min_occupancy

        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        # Number of K tiles for QH MMA (K=BK for KDA)
        self.num_k_tiles = (head_dim_k + BK - 1) // BK  # 1

        # Pipeline stages for TMA inputs.
        # Non-persistent (occ=2): single-buffered to keep SMEM ≤ 114K (228K/2).
        #   q=16K + g=32K + h=32K + v=16K + A=8K + O=16K = ~120K ✓
        # Persistent (occ=1): double-buffer to overlap TMA prefetch with compute.
        #   q=32K + g=32K + h=64K + v=32K + A=16K + O=16K = ~192K < 228K ✓
        #   g is kept 1-stage (32K fp32 too expensive to double).
        self.o_stage = 1
        self.acc_stage = 1
        if self.persistent:
            self.min_occupancy = 1
            self.q_stage = 2
            self.g_stage = 1
            self.h_stage = 2
            self.v_stage = 2
            self.a_stage = 2
            # With occ=1 (65536 regs/CTA), we can give more registers to
            # the store warp so it can hold the full O tile partition in
            # registers (~128 regs for 256 bf16), enabling bulk SMEM→REG
            # prefetch before GMEM writes.
            # Budget: 4×32×208 + 4×32×168 = 48128 ≤ 65536 ✓
            self.num_regs_others = 168
        else:
            self.q_stage = 1
            self.g_stage = 1
            self.h_stage = 1
            self.v_stage = 1
            self.a_stage = 1

        # MMA tiler shapes:
        # QH: qg(BT, BK) @ h(BK, BV) → (BT, BV)
        self.qh_mma_tiler = (self.BT, self.BV, self.BK)
        # AV: A(BT, BT) @ v(BT, BV) → (BT, BV)
        self.av_mma_tiler = (self.BT, self.BV, self.BT)

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta,
        )
        self.buffer_align_bytes = 1024

    def _compute_grid(self, B, T, H, V, total_nt=None):
        """Compute grid dimensions for kernel launch."""
        num_v_tiles = (V + self.BV - 1) // self.BV
        if self.persistent:
            # Persistent kernel: grid = SM_count.  Each CTA loops over
            # multiple work units via grid-stride.
            import torch
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
            return (sm_count, 1, 1)
        elif self.is_varlen:
            # Non-persistent varlen: one CTA per work unit.
            total_work_units = num_v_tiles * total_nt * H
            return (total_work_units, 1, 1)
        NT = (T + self.BT - 1) // self.BT
        return (num_v_tiles, NT, B * H)

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
        cu_seqlens_ptr: cute.Pointer,  # [N+1] int32
        chunk_indices_ptr: cute.Pointer, # [NT*2] int32 — (batch_idx, chunk_seq_idx) pairs
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,               # total chunks across all seqs (varlen)
        stream,
    ):
        B, T, H, K, V = problem_size
        BT = self.BT

        # For varlen: B=num_seqs, T=max_seqlen (or total_tokens), data_B=1
        # For non-varlen: data_B=B, NT=ceil(T/BT)
        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
            NT = total_nt
        else:
            data_B = B
            NT = (T + BT - 1) // BT

        # ===================== GMEM layouts =====================
        # q layout: token-indexed (T, K, (H, data_B)) — bf16
        #   varlen: data_B=1, T=T_total
        #   non-varlen: data_B=B
        q_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        q = cute.make_tensor(q_ptr, q_layout)

        # g layout: token-indexed (T, K, (H, data_B)) — fp32 (separate from q)
        g_layout = cute.make_layout(
            (T, K, (H, data_B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        g = cute.make_tensor(g_ptr, g_layout)

        # o: row-major (T, V, (H, data_B)) — token-indexed for direct GMEM write (varlen)
        o_layout = cute.make_layout(
            (T, V, (H, data_B)),
            stride=(H * V, 1, (V, T * H * V)),
        )
        o = cute.make_tensor(o_ptr, o_layout)

        # v transposed for MMA B TMA: token-indexed (V, T, (data_B, H))
        # NOTE: Mode 2 uses (batch, H) order — NOT (H, batch) — so that
        # the batch dimension occupies TMA coordinate 2.  When H=1 the
        # TMA descriptor collapses the degenerate H dim; keeping batch
        # at coord-2 guarantees it always maps to an existing TMA dim.
        v_T_layout = cute.make_layout(
            (V, T, (data_B, H)),
            stride=(1, H * V, (T * H * V, V)),
        )
        v_T = cute.make_tensor(v_ptr, v_T_layout)

        # h: stored as [NT_total, H, K, V] — V contiguous
        # Transposed view for MMA B TMA: (V, K, (H, NT_total)) — V contiguous
        # non-varlen: NT_total = B * NT;  varlen: NT_total = total_nt (already flat)
        if cutlass.const_expr(self.is_varlen):
            h_nt_total = NT  # = total_nt
        else:
            h_nt_total = B * NT
        # NOTE: Mode 2 uses (batch, H) order — see v_T comment above.
        h_T_layout = cute.make_layout(
            (V, K, (h_nt_total, H)),
            stride=(1, V, (H * K * V, K * V)),
        )
        h_T = cute.make_tensor(h_ptr, h_T_layout)

        # A layout: token-indexed (T, BT, (H, data_B))
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

        # Epilog SMEM for q (ROW_MAJOR BT×BK, bf16)
        q_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK), self.q_stage,
        )
        # Epilog SMEM for g (ROW_MAJOR BT×BK, fp32)
        g_epi_staged = sm100_utils.make_smem_layout_epi(
            self.g_dtype, utils.LayoutEnum.ROW_MAJOR,
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
            (self.BT, self.BV), self.o_stage,
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
        self.tma_bytes_g = cute.size_in_bytes(self.g_dtype, g_epi_smem)
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
            qg_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]   # CUDA→MMA: qg ready
            am_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]   # CUDA→MMA: am ready
            acc_done_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # MMA→CUDA: acc done
            o_ready_mbar: cute.struct.MemRange[Int64, self.o_stage * 2]   # CUDA→Load: o ready

            sQ_epi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_epi_staged)],
                self.buffer_align_bytes,
            ]
            sG_epi: cute.struct.Align[
                cute.struct.MemRange[self.g_dtype, cute.cosize(g_epi_staged)],
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
        # to write A_masked into TMEM as av_tiled_mma's A operand.
        # B operand majorness must match av_tiled_mma for C layout compatibility.
        am_coord_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            (self.BT, self.BT),
            tcgen05.OperandSource.TMEM,
        )

        # ===================== Grid =====================
        grid = self._compute_grid(B, T, H, V, total_nt=total_nt)

        # ===================== cu_seqlens / chunk_indices tensors =====================
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((B + 1,)))
        chunk_indices = cute.make_tensor(chunk_indices_ptr, cute.make_layout((total_nt * 2,)))

        # ===================== Direct GMEM write for varlen O store =====================
        # For varlen tail chunks, TMA store would write beyond sequence boundary.
        # Use CopyUniversalOp with per-row bounds check instead.
        if cutlass.const_expr(self.is_varlen):
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.io_dtype.width  # 8 for bf16
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.io_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            # Thread layout for store warp (32 threads) over (BT, BV) tile:
            # Mode 0 (BT=64): 2 threads × 1 value = 2 per rest → 32 rest iters
            # Mode 1 (BV=128): 16 threads × 8 values = 128 (fully covered)
            o_thr_dim0 = self.threads_per_warp // (self.BV // async_copy_elems)  # 32/16 = 2
            o_thr_dim1 = self.BV // async_copy_elems  # 128/8 = 16
            assert self.BT % o_thr_dim0 == 0
            o_thr_layout = cute.make_ordered_layout(
                (o_thr_dim0, o_thr_dim1), order=(1, 0),
            )  # mode 1 (BV) faster → coalesced GMEM writes
            o_val_layout = cute.make_layout((1, async_copy_elems))  # (1, 8)
            gmem_tiled_copy_o = cute.make_tiled_copy_tv(
                atom_universal_copy, o_thr_layout, o_val_layout,
            )
        else:
            gmem_tiled_copy_o = None

        # ===================== O GMEM tensor for varlen direct write =====================
        o_tensor = cute.make_tensor(o_ptr, o_layout)

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
            cu_seqlens, chunk_indices,
            o_tensor,
            gmem_tiled_copy_o,
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
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        o_tensor: cute.Tensor,
        gmem_tiled_copy_o,
        problem_size,
        total_nt,
    ):
        B, T, H, K, V = problem_size
        BT = self.BT

        if cutlass.const_expr(self.is_varlen):
            data_B = Int32(1)
        else:
            data_B = B

        # ===================== Work decode =====================
        num_v_tiles = (V + self.BV - 1) // self.BV

        if cutlass.const_expr(self.persistent):
            # Persistent kernel: 1D grid, work decoded inside each warp's loop
            block_idx_x = cute.arch.block_idx()[0]
            grid_dim_x = cute.arch.grid_dim()[0]
            total_work_units = num_v_tiles * total_nt * H
            num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x
            # Pre-initialize persistent loop variables (CuTe DSL requirement)
            i_v = Int32(0)
            chunk_global_idx = Int32(0)
            i_h = Int32(0)
            i_b = Int32(0)
            i_t = Int32(0)
            tok_offset = Int32(0)
            seq_len = Int32(0)
            remaining = Int32(BT)
            i_tg = Int32(0)
            data_bidx = Int32(0)
            if cutlass.const_expr(not self.is_varlen):
                NT = (T + BT - 1) // BT
        else:
            NT = (T + BT - 1) // BT
            i_v = cute.arch.block_idx()[0]
            i_t = cute.arch.block_idx()[1]
            i_bh = cute.arch.block_idx()[2]
            i_b = i_bh // H
            i_h = i_bh % H
            tok_offset = i_b * T
            seq_len = T
            data_bidx = i_b
            i_tg = i_b * NT + i_t
            num_iters = Int32(1)

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

        # TMA partitions: computed per-iteration inside warp loops following
        # the persistent kernel pattern (domain_offset for varlen, alias for non-varlen).

        # ---- Pipelines ----
        num_cuda_threads = self.threads_per_warp * len(self.cuda_warp_ids)
        num_cuda_warps = len(self.cuda_warp_ids)

        # NOTE: PipelineTmaAsync consumer_group size must equal the number of
        # signalling threads in consumer_release (1 per warp for single-CTA),
        # NOT the total consumer thread count.  See chunk_delta_h for reference.
        load_q_P, load_q_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_cuda_warps),
            tx_count=self.tma_bytes_q,
            barrier_storage=storage.load_q_mbar.data_ptr(),
        ).make_participants()

        load_g_P, load_g_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.g_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_cuda_warps),
            tx_count=self.tma_bytes_g,
            barrier_storage=storage.load_g_mbar.data_ptr(),
        ).make_participants()

        load_a_P, load_a_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.a_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(num_cuda_warps),
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

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # --- Persistent work decode ---
                if cutlass.const_expr(self.persistent):
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_v = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    chunk_flat = temp_work % total_nt
                    i_h = temp_work // total_nt
                    if cutlass.const_expr(self.is_varlen):
                        i_b = chunk_indices[chunk_flat * 2]
                        i_t = chunk_indices[chunk_flat * 2 + 1]
                        tok_offset = cu_seqlens[i_b]
                        data_bidx = Int32(0)
                    else:
                        i_b = chunk_flat // NT
                        i_t = chunk_flat % NT
                        tok_offset = i_b * T
                        data_bidx = i_b
                    i_tg = chunk_flat

                # --- Domain offset for varlen, alias for non-varlen ---
                if cutlass.const_expr(self.is_varlen):
                    tma_q_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_q)
                    tma_g_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_g)
                    tma_a_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_a)
                    tma_v_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_v)
                else:
                    tma_q_v = tma_tensor_q
                    tma_g_v = tma_tensor_g
                    tma_a_v = tma_tensor_a
                    tma_v_v = tma_tensor_v

                # --- Unconditional TMA partitions ---
                bSG_sQ, bSG_gQ = self._epilog_partition_varlen(
                    tma_atom_q, tma_q_v[None, None, (i_h, data_bidx)], (self.BT, self.BK), sQ_epi,
                )
                bSG_sG, bSG_gG = self._epilog_partition_varlen(
                    tma_atom_g, tma_g_v[None, None, (i_h, data_bidx)], (self.BT, self.BK), sG_epi,
                )
                bSG_sA, bSG_gA = self._epilog_partition_varlen(
                    tma_atom_a, tma_a_v[None, None, (i_h, data_bidx)], (self.BT, self.BT), sA_epi,
                )
                tHsH, tHgH = self._tma_partition_B(
                    tma_atom_h, tma_tensor_h, sH, self.qh_mma_tiler, qh_tiled_mma, i_tg, i_h,
                )
                tVsV, tVgV = self._tma_partition_B(
                    tma_atom_v, tma_v_v, sV, self.av_mma_tiler, av_tiled_mma, data_bidx, i_h,
                )

                epi_tile_t = i_t
                for i_k in cutlass.range(self.num_k_tiles, unroll_full=True):
                    q_h = load_q_P.acquire_and_advance()
                    cute.copy(atom=tma_atom_q, src=bSG_gQ[(None, epi_tile_t, 0)],
                              dst=bSG_sQ[None, q_h.index], tma_bar_ptr=q_h.barrier)
                    g_h = load_g_P.acquire_and_advance()
                    cute.copy(atom=tma_atom_g, src=bSG_gG[(None, epi_tile_t, 0)],
                              dst=bSG_sG[None, g_h.index], tma_bar_ptr=g_h.barrier)
                    h_h = load_h_P.acquire_and_advance()
                    cute.copy(atom=tma_atom_h, src=tHgH[None, i_v, 0],
                              dst=tHsH[None, h_h.index], tma_bar_ptr=h_h.barrier)

                v_h = load_v_P.acquire_and_advance()
                cute.copy(atom=tma_atom_v, src=tVgV[None, i_v, i_t],
                          dst=tVsV[None, v_h.index], tma_bar_ptr=v_h.barrier)
                a_h = load_a_P.acquire_and_advance()
                cute.copy(atom=tma_atom_a, src=bSG_gA[(None, epi_tile_t, 0)],
                          dst=bSG_sA[None, a_h.index], tma_bar_ptr=a_h.barrier)

        # =====================================================================
        # STORE WARP
        # =====================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            cpasync.prefetch_descriptor(tma_atom_o)

            if cutlass.const_expr(self.is_varlen):
                # ---- Persistent varlen store ----
                # With num_regs_others=168 (persistent, occ=1), the store warp
                # can hold the full O tile partition in registers (~128 regs
                # for 256 bf16).  Bulk SMEM→REG prefetch so GMEM writes don't
                # stall on SMEM reads.
                store_local_tidx = tidx % self.threads_per_warp
                gmem_thr_copy = gmem_tiled_copy_o.get_slice(store_local_tidx)
                sO_stage = sO[(None, None, 0)]
                tOsO = gmem_thr_copy.partition_S(sO_stage)
                cO = cute.make_identity_tensor((self.BT, self.BV))
                tOcO = gmem_thr_copy.partition_S(cO)
                tOrO = cute.make_fragment_like(tOsO, self.io_dtype)

                for wu_iter in cutlass.range(0, num_iters, unroll=0):
                    o_h = o_ready_C.wait_and_advance()

                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_v = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    chunk_global_idx = temp_work % total_nt
                    i_h = temp_work // total_nt
                    i_b = chunk_indices[chunk_global_idx * 2]
                    i_t = chunk_indices[chunk_global_idx * 2 + 1]
                    tok_offset = cu_seqlens[i_b]
                    seq_len = cu_seqlens[i_b + 1] - tok_offset
                    remaining = seq_len - i_t * BT
                    remaining = cutlass.select_(remaining > BT, Int32(BT), remaining)

                    # Bulk prefetch: SMEM → registers (all 256 bf16 at once)
                    cute.autovec_copy(tOsO, tOrO)
                    o_chunk_raw = (o_tensor.iterator
                        + (tok_offset + i_t * BT) * H * V
                        + i_h * V
                        + i_v * self.BV)
                    o_chunk_ptr = cute.make_ptr(
                        self.io_dtype, o_chunk_raw.toint(),
                        cute.AddressSpace.gmem, assumed_align=16,
                    )
                    o_stride_bt = cute.assume(
                        H * V, divby=128 // self.io_dtype.width,
                    )
                    gO_chunk = cute.make_tensor(
                        o_chunk_ptr,
                        cute.make_layout(
                            (self.BT, self.BV), stride=(o_stride_bt, 1),
                        ),
                    )
                    tOgO = gmem_thr_copy.partition_D(gO_chunk)

                    # Registers → GMEM with bounds check
                    for m1 in cutlass.range_constexpr(cute.size(tOsO.shape[1])):
                        bt_coord = tOcO[(0, 0), m1, 0][0]
                        if bt_coord < remaining:
                            cute.autovec_copy(tOrO[(None, m1, None)], tOgO[(None, m1, None)])

                    o_h.release()
            elif cutlass.const_expr(self.persistent):
                # ---- Persistent non-varlen: TMA store per WU ----
                for wu_iter in cutlass.range(0, num_iters, unroll=0):
                    o_h = o_ready_C.wait_and_advance()

                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_v = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    chunk_flat = temp_work % total_nt
                    i_h = temp_work // total_nt
                    i_b = chunk_flat // NT
                    i_t = chunk_flat % NT
                    data_bidx = i_b

                    gO = tma_tensor_o[None, None, (i_h, data_bidx)]
                    _, bSG_sO, bSG_gO = self._epilog_partition(
                        tma_atom_o, gO, (self.BT, self.BV), sO,
                    )
                    cute.copy(tma_atom_o, bSG_sO[None, 0], bSG_gO[(None, i_t, i_v)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o_h.release()
            else:
                # ---- Non-persistent non-varlen: single TMA store ----
                gO = tma_tensor_o[None, None, (i_h, data_bidx)]
                _, bSG_sO, bSG_gO = self._epilog_partition(
                    tma_atom_o, gO, (self.BT, self.BV), sO,
                )
                for wu_iter in cutlass.range(0, num_iters, unroll=0):
                    o_h = o_ready_C.wait_and_advance()
                    cute.copy(tma_atom_o, bSG_sO[None, 0], bSG_gO[(None, i_t, i_v)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o_h.release()

        # =====================================================================
        # EMPTY WARP
        # =====================================================================
        elif warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

        # =====================================================================
        # MMA WARP
        # =====================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # Phase 1: QH MMA — qg(TMEM) × h(SMEM) → acc(TMEM)
                qg_h = qg_C.wait_and_advance()
                acc_h = acc_done_P.acquire_and_advance()

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

                # Phase 3: Signal ACC done to CUDA warps
                acc_h.commit()

        # =====================================================================
        # CUDA WARPS: Gating + Masking → TMEM
        # =====================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))
            scale_f32 = Float32(self.scale)

            # ====== Persistent computation loop ======
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                if cutlass.const_expr(self.persistent and self.is_varlen):
                    # Work decode for remaining (persistent varlen)
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_v = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    chunk_global_idx = temp_work % total_nt
                    i_h = temp_work // total_nt
                    i_b = chunk_indices[chunk_global_idx * 2]
                    i_t = chunk_indices[chunk_global_idx * 2 + 1]
                    tok_offset = cu_seqlens[i_b]
                    seq_len = cu_seqlens[i_b + 1] - tok_offset
                    remaining = seq_len - i_t * BT
                    remaining = cutlass.select_(remaining > BT, Int32(BT), remaining)

                # ----- T2R for ACC (FP32, BT×BV=64×128) — for QG coordinate mapping -----
                t2r_atom_acc = cute.make_copy_atom(
                    tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
                    self.acc_dtype,
                )
                tCtAcc_flat = tCtAcc[((None, None), 0, 0, None)]
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

                # ----- AM R2T: bf16 registers → AM TMEM -----
                r2t_atom_am = cute.make_copy_atom(
                    tcgen05.St16x128bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                    self.io_dtype,
                )
                tiled_r2t_am = tcgen05.make_tmem_copy(r2t_atom_am, tCrAM_tmem)
                thr_r2t_am = tiled_r2t_am.get_slice(local_tidx)
                r2t_am_shape = cute.slice_(thr_r2t_am.partition_S(tCrAM_tmem).shape, (None, None, None, None, 0))
                tRT_tAM = thr_r2t_am.partition_D(tCrAM_tmem)
                tRT_rAM = cute.make_rmem_tensor(r2t_am_shape, self.io_dtype)

                # AM coordinate mapping via R2T partition_S(identity)
                cM_am_r4 = cute.make_identity_tensor(tCrAM_tmem.layout.shape)
                tRS_cM_am_full = thr_r2t_am.partition_S(cM_am_r4)
                tRS_cM_am = cute.slice_(tRS_cM_am_full, (None, None, None, None, 0))

                # ----- R2S: ACC T2R regs → sO (ROW_MAJOR, BT×BV) -----
                r2s_atom_o = sm100_utils.get_smem_store_op(
                    utils.LayoutEnum.ROW_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r_acc,
                )
                tiled_r2s_o = cute.make_tiled_copy_D(r2s_atom_o, tiled_t2r_acc)
                thr_r2s_o = tiled_r2s_o.get_slice(local_tidx)
                tRS_sO = thr_r2s_o.partition_D(sO)

                # Output epilog setup
                tTR_tAcc = thr_t2r_acc.partition_S(tCtAcc_flat)

                # ============ Compute QG: q * exp2(g) * scale ============
                for i_k in cutlass.range(self.num_k_tiles, unroll_full=True):
                    q_h = load_q_C.wait_and_advance()
                    g_h = load_g_C.wait_and_advance()

                    for ei in cutlass.range_constexpr(cute.size(tTR_rQG_fp32)):
                        bt_coord, bk_coord = tTR_cM_qg[ei]
                        if cutlass.const_expr(self.is_varlen):
                            if bt_coord < remaining:
                                q_val = sQ_epi[(bt_coord, bk_coord, q_h.index)].to(self.acc_dtype)
                                g_val = sG_epi[(bt_coord, bk_coord, g_h.index)]
                                tTR_rQG_fp32[ei] = q_val * cute.exp2(g_val) * scale_f32
                            else:
                                tTR_rQG_fp32[ei] = Float32(0.0)
                        else:
                            q_val = sQ_epi[(bt_coord, bk_coord, q_h.index)].to(self.acc_dtype)
                            g_val = sG_epi[(bt_coord, bk_coord, g_h.index)]
                            tTR_rQG_fp32[ei] = q_val * cute.exp2(g_val) * scale_f32

                    q_h.release()
                    g_h.release()

                    tRT_rQG_bf16.store(tTR_rQG_fp32.load().to(self.io_dtype))
                    qg_h = qg_P.acquire_and_advance()
                    cute.copy(tiled_r2t_qg, tRT_rQG_bf16, tRT_tQG[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    qg_h.commit()

                # ============ Compute AM: tril(A) with varlen boundary mask ============
                a_h = load_a_C.wait_and_advance()

                for ei in cutlass.range_constexpr(cute.size(tRT_rAM)):
                    coord_val = tRS_cM_am[ei]
                    m0, m1, m2, m3 = coord_val
                    sub0, sub1 = m0
                    sub0_0, sub0_1 = sub0
                    row = sub0_0 + sub0_1 * 16
                    col = sub1 + m2 * 16
                    if cutlass.const_expr(self.is_varlen):
                        if row >= col and row < remaining and col < remaining:
                            tRT_rAM[ei] = sA_epi[(row, col, a_h.index)]
                        else:
                            tRT_rAM[ei] = Float32(0.0).to(self.io_dtype)
                    else:
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
                acc_h = acc_done_C.wait_and_advance()

                tTR_rAcc = cute.make_rmem_tensor(thr_t2r_acc.partition_D(fake_sQG).shape, self.acc_dtype)
                cute.copy(tiled_t2r_acc, tTR_tAcc[(None, None, None, 0)], tTR_rAcc)
                cute.arch.fence_view_async_tmem_load()
                acc_h.release()

                tTR_rAcc_bf16 = cute.make_rmem_tensor(tTR_rAcc.shape, self.io_dtype)
                tTR_rAcc_bf16.store(tTR_rAcc.load().to(self.io_dtype))

                tRS_rO = tiled_r2s_o.retile(tTR_rAcc_bf16)
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
        """Partition B operand for TMA.
        
        The GMEM layout mode 2 is (batch, H), so the coord is (batch_idx, head_idx).
        """
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_tma_partition_B: tma_tensor = {tma_tensor}")
            print(f"_tma_partition_B: tile_shape = {tile_shape}")
        coord = (0, None, None)
        tiler = cute.slice_(tile_shape, coord)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_tma_partition_B: tiler (sliced) = {tiler}")
        gX = cute.local_tile(
            tma_tensor, tiler, (None, None, (batch_idx, head_idx))
        )
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_tma_partition_B: gX (local_tile result) = {gX}")
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_tma_partition_B: tCgX (partition_B result) = {tCgX}")
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1),
            cute.group_modes(smem, 0, 3), cute.group_modes(tCgX, 0, 3),
        )
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_tma_partition_B: tXsX = {tXsX}")
            print(f"_tma_partition_B: tXgX = {tXgX}")
        return tXsX, tXgX

    @cute.jit
    def _epilog_partition_3d(self, atom, tma_tensor_3d, epi_tile, sC, head_idx, batch_idx):
        """Partition for epilog TMA load, operating on the full 3D TMA tensor.

        Uses local_tile on the 3D tensor (T, F, (H, B)) to preserve mode2 coordinate
        information for tma_partition.  This is critical when domain_offset is used
        (varlen), because slicing mode2 first would bake the head offset into the
        pointer and lose the coordinate — causing tma_partition to generate wrong
        TMA coordinates for heads > 0.

        This follows the same pattern as _tma_partition_B and kda.py's
        local_tile_partition_for_mma_operand.
        """
        # local_tile on 3D: tile mode0 by epi_tile[0], tile mode1 by epi_tile[1],
        # select mode2 = (head_idx, batch_idx)
        gC = cute.local_tile(
            tma_tensor_3d,
            epi_tile,   # (BT, BK) or (BT, BT) — tiles first 2 modes
            (None, None, (head_idx, batch_idx)),  # keep T/K tiles, fix mode2
        )
        # gC: (BT, BK/BT, NT, NK) — mode2 consumed by coord selection
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), sC_g, gC_g,
        )
        return bSG_sC, bSG_gC

    @cute.jit
    def _epilog_partition(self, atom, gC_mnl, epi_tile, sC):
        """Partition for epilog TMA load/store (2D tensor, used for O store)."""
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_epilog_partition: gC_mnl = {gC_mnl}")
            print(f"_epilog_partition: epi_tile = {epi_tile}")
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_epilog_partition: gC_epi (flat_divide result) = {gC_epi}")
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_epi, 0, 2)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_epilog_partition: gC_g (grouped) = {gC_g}")
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), sC_g, gC_g,
        )
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_epilog_partition: bSG_gC (tma_partition result) = {bSG_gC}")
        return atom, bSG_sC, bSG_gC

    @cute.jit
    def _epilog_partition_varlen(self, atom, gC_2d, epi_tile, sC):
        """Partition for varlen epilog TMA load (2D tensor with domain_offset).

        Uses local_tile instead of flat_divide to correctly preserve TMA basis
        stride coordinates through domain_offset.  Matches Flash Attention's
        pattern: slice mode2 → domain_offset(2D) → local_tile → tma_partition.

        Uses (None, None) to keep all tile-count modes, producing the same
        rank as _epilog_partition (flat_divide) so copy indexing is unchanged.
        """
        gC_tiled = cute.local_tile(gC_2d, epi_tile, (None, None))
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_tiled, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), sC_g, gC_g,
        )
        return bSG_sC, bSG_gC


# =====================================================================
# Varlen preprocessing helpers
# =====================================================================

def prepare_chunked(tensor, cu_seqlens, chunk_offsets, BT=64):
    """
    Preprocess token-indexed tensor to chunk-indexed layout for varlen TMA.

    Converts (T_total, H, F) → (total_nt, BT, H, F), where each chunk's data
    starts at a BT-aligned position in the output buffer.  This eliminates
    the need for `domain_offset` in the kernel for all varlen tensors (q, g, v, A).

    Args:
        tensor: [T_total, H, F] — token-level tensor (any feature dim F)
        cu_seqlens: [N+1] int32 — cumulative sequence lengths
        chunk_offsets: [N+1] int32 — cumulative chunk counts
        BT: chunk size (default 64)

    Returns:
        chunked: [total_nt, BT, H, F] — chunk-indexed tensor with zero-padding
    """
    import torch
    cu_seqlens_cpu = cu_seqlens.cpu().tolist() if isinstance(cu_seqlens, torch.Tensor) else list(cu_seqlens)
    chunk_offsets_cpu = chunk_offsets.cpu().tolist() if isinstance(chunk_offsets, torch.Tensor) else list(chunk_offsets)

    num_seqs = len(cu_seqlens_cpu) - 1
    total_nt = chunk_offsets_cpu[-1]
    H = tensor.shape[1]
    F = tensor.shape[2]

    chunked = torch.zeros(total_nt, BT, H, F, dtype=tensor.dtype, device=tensor.device)
    for i in range(num_seqs):
        tok_off = cu_seqlens_cpu[i]
        seq_len = cu_seqlens_cpu[i + 1] - tok_off
        co = chunk_offsets_cpu[i]
        nt = (seq_len + BT - 1) // BT
        for c in range(nt):
            src_start = tok_off + c * BT
            chunk_len = min(BT, seq_len - c * BT)
            chunked[co + c, :chunk_len] = tensor[src_start:src_start + chunk_len]
    return chunked


def prepare_v_chunked(v, cu_seqlens, chunk_offsets, BT=64):
    """Backward-compatible wrapper: v has an extra leading batch dim [1, T_total, H, V]."""
    return prepare_chunked(v[0], cu_seqlens, chunk_offsets, BT)


def build_chunk_indices(seq_lens, BT=64, device='cuda'):
    """
    Build chunk_indices tensor in the same format as FLA's prepare_chunk_indices.

    Returns a flat int32 tensor of shape [NT*2], where each pair is
    (batch_idx, chunk_seq_idx).  This matches the kda_bwd decode_tile_coord
    scheme: chunk_indices[i*2] = batch_idx, chunk_indices[i*2+1] = chunk_in_seq.

    Args:
        seq_lens: list of sequence lengths
        BT: chunk size (default 64)
        device: torch device

    Returns:
        chunk_indices: [NT*2] int32 tensor
    """
    import torch
    pairs = []
    for seq_idx, sl in enumerate(seq_lens):
        nt = (sl + BT - 1) // BT
        for c in range(nt):
            pairs.extend([seq_idx, c])
    return torch.tensor(pairs, dtype=torch.int32, device=device)


def build_chunk_offsets(seq_lens, BT=64):
    """Build chunk_offsets list [N+1] from sequence lengths (for reference h indexing)."""
    offsets = [0]
    for sl in seq_lens:
        offsets.append(offsets[-1] + (sl + BT - 1) // BT)
    return offsets


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
        print("\n=== Correctness Test (skipped, testing varlen directly) ===")

        # ----- Varlen correctness test -----
        print("\n=== Varlen Correctness Test ===")

        # Diagnostic: Step-function A to prove wrong T-position load
        print("\n  --- TMA address verification test ---")
        try:
            seq_lens_d = [100, 128]
            T_d = 228
            cu_d = [0, 100, 228]
            total_nt_d = 4
            cu_t = torch.tensor(cu_d, dtype=torch.int32, device=device)
            ci_t = build_chunk_indices(seq_lens_d, BT=BT, device=device)
            cu_c = from_dlpack(cu_t.detach())
            ci_c = from_dlpack(ci_t.detach())
            ch_offsets_d = build_chunk_offsets(seq_lens_d, BT=BT)
            stream = cutlass_torch.default_stream()
            H_test = 1  # Use H=1 to reduce SMEM debug output
            ps_d = (2, T_d, H_test, K, V)

            # Use q[t, h, k] = float(t) so SMEM values reveal the actual global token loaded
            q_tma_test = torch.zeros(T_d, H_test, K, dtype=dtype, device=device)
            for t in range(T_d):
                q_tma_test[t, :, :] = float(t)
            # g[t, h, k] = float(t) * 0.001 (distinctive but small)
            g_tma_test = torch.zeros(T_d, H_test, K, dtype=torch.float32, device=device)
            for t in range(T_d):
                g_tma_test[t, :, :] = float(t) * 0.001
            # A[t, h, j] = float(t) + j*0.01
            A_tma_test = torch.zeros(T_d, H_test, BT, dtype=dtype, device=device)
            for t in range(T_d):
                for j in range(BT):
                    A_tma_test[t, :, j] = float(t) + j * 0.01
            v_tma_test = torch.zeros(T_d, H_test, V, dtype=dtype, device=device)
            for t in range(T_d):
                v_tma_test[t, :, :] = float(t)
            h_tma_test = torch.zeros(total_nt_d, H_test, K, V, dtype=dtype, device=device)
            o_tma_test = torch.zeros(T_d, H_test, V, dtype=dtype, device=device)

            # Print expected values for comparison
            print("  Expected SMEM values (q[t]=t, g[t]=t*0.001, A[t,j]=t+j*0.01):")
            for si, sl in enumerate(seq_lens_d):
                s = cu_d[si]
                nt = (sl + BT - 1) // BT
                for c in range(nt):
                    tok = s + c * BT
                    rem = min(BT, sl - c * BT)
                    print(f"    Seq{si} c{c}: tok_offset={s} local_chunk={c} global_tok={tok} remaining={rem}")
                    print(f"      Expected q[0..3,0] = {tok:.0f} {tok+1:.0f} {tok+2:.0f} {tok+3:.0f}")
                    print(f"      Expected g[0..3,0] = {tok*0.001:.4f} {(tok+1)*0.001:.4f} {(tok+2)*0.001:.4f} {(tok+3)*0.001:.4f}")
                    print(f"      Expected A[0..3,0] = {tok:.2f} {tok+1:.2f} {tok+2:.2f} {tok+3:.2f}")
                    if rem < BT:
                        print(f"      Expected q[rem-1={rem-1},0] = {tok+rem-1:.0f}")
                        print(f"      Expected q[rem={rem},0] = 0.0 (TMA zero-fill)")

            k_tma = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True)
            c_tma = cute.compile(
                k_tma,
                from_dlpack(q_tma_test.detach()).iterator, from_dlpack(v_tma_test.detach()).iterator,
                from_dlpack(g_tma_test.detach()).iterator, from_dlpack(h_tma_test.detach()).iterator,
                from_dlpack(o_tma_test.detach()).iterator, from_dlpack(A_tma_test.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_d, total_nt_d, stream,
            )
            c_tma(
                from_dlpack(q_tma_test.detach()).iterator, from_dlpack(v_tma_test.detach()).iterator,
                from_dlpack(g_tma_test.detach()).iterator, from_dlpack(h_tma_test.detach()).iterator,
                from_dlpack(o_tma_test.detach()).iterator, from_dlpack(A_tma_test.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_d, total_nt_d, stream,
            )
            torch.cuda.synchronize()
            print("  (Check SMEM debug prints above for actual loaded values)")
        except Exception as e:
            import traceback
            print(f"  TMA verification test failed: {e}")
            traceback.print_exc()

        print("\n  TMA verification test done.")

        print("\n  --- Step-function A test (prove T-position bug) ---")
        try:
            seq_lens_d = [100, 128]
            T_d = 228
            cu_d = [0, 100, 228]
            total_nt_d = 4
            cu_t = torch.tensor(cu_d, dtype=torch.int32, device=device)
            ci_t = build_chunk_indices(seq_lens_d, BT=BT, device=device)
            cu_c = from_dlpack(cu_t.detach())
            ci_c = from_dlpack(ci_t.detach())
            ch_offsets_d = build_chunk_offsets(seq_lens_d, BT=BT)
            stream = cutlass_torch.default_stream()
            ps_d = (2, T_d, H, K, V)

            # A step: A=1 for t < 100, A=0 for t >= 100
            # For Seq1 (t >= 100): all A=0 → tril(0) @ v = 0. Any non-zero output = wrong T position.
            A_step = torch.ones(T_d, H, BT, dtype=dtype, device=device)
            A_step[100:, :, :] = 0.0
            v_step = torch.ones(T_d, H, V, dtype=dtype, device=device)
            q_step = torch.zeros(T_d, H, K, dtype=dtype, device=device)
            g_step = torch.zeros(T_d, H, K, dtype=torch.float32, device=device)
            h_step = torch.zeros(total_nt_d, H, K, V, dtype=dtype, device=device)

            o_step = torch.zeros(T_d, H, V, dtype=dtype, device=device)
            k_step = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True)
            c_step = cute.compile(
                k_step,
                from_dlpack(q_step.detach()).iterator, from_dlpack(v_step.detach()).iterator,
                from_dlpack(g_step.detach()).iterator, from_dlpack(h_step.detach()).iterator,
                from_dlpack(o_step.detach()).iterator, from_dlpack(A_step.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_d, total_nt_d, stream,
            )
            c_step(
                from_dlpack(q_step.detach()).iterator, from_dlpack(v_step.detach()).iterator,
                from_dlpack(g_step.detach()).iterator, from_dlpack(h_step.detach()).iterator,
                from_dlpack(o_step.detach()).iterator, from_dlpack(A_step.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_d, total_nt_d, stream,
            )
            torch.cuda.synchronize()

            print("    A-step test: A=1 for t<100, A=0 for t>=100")
            for seq_idx, sl in enumerate(seq_lens_d):
                s = cu_d[seq_idx]
                nt = (sl + BT - 1) // BT
                for c_idx in range(nt):
                    cs = s + c_idx * BT
                    ce = min(cs + BT, cu_d[seq_idx + 1])
                    cl = ce - cs
                    o_chunk = o_step[cs:ce, 0, 0]
                    mx = o_chunk.abs().max().item()
                    first_nonzero = -1
                    for r in range(cl):
                        if abs(o_chunk[r].item()) > 0.01:
                            first_nonzero = r
                            break
                    nz_str = f" first_nonzero_row={first_nonzero}" if first_nonzero >= 0 else ""
                    # Show a few values
                    vals = [f"r{r}:{o_chunk[r].item():.2f}" for r in [0, 1, 2, min(cl-1, 63)]]
                    print(f"    Seq{seq_idx} c{c_idx} cs={cs}: max={mx:.4f} {' '.join(vals)}{nz_str}")
                    # For Seq1: any non-zero means wrong T position
                    if seq_idx == 1 and mx > 0.01:
                        print(f"      → PROOF: kernel loaded A from T<100 for Seq1 c{c_idx}!")

            # V step: v=1 for t < 100, v=0 for t >= 100, A=ones
            # For Seq1: v=0 → tril(A)@0 = 0. Any non-zero = wrong v T position.
            print("    V-step test: v=1 for t<100, v=0 for t>=100, A=ones")
            v_step2 = torch.ones(T_d, H, V, dtype=dtype, device=device)
            v_step2[100:, :, :] = 0.0
            A_step2 = torch.ones(T_d, H, BT, dtype=dtype, device=device)

            o_step2 = torch.zeros(T_d, H, V, dtype=dtype, device=device)
            k_step2 = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True)
            c_step2 = cute.compile(
                k_step2,
                from_dlpack(q_step.detach()).iterator, from_dlpack(v_step2.detach()).iterator,
                from_dlpack(g_step.detach()).iterator, from_dlpack(h_step.detach()).iterator,
                from_dlpack(o_step2.detach()).iterator, from_dlpack(A_step2.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_d, total_nt_d, stream,
            )
            c_step2(
                from_dlpack(q_step.detach()).iterator, from_dlpack(v_step2.detach()).iterator,
                from_dlpack(g_step.detach()).iterator, from_dlpack(h_step.detach()).iterator,
                from_dlpack(o_step2.detach()).iterator, from_dlpack(A_step2.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_d, total_nt_d, stream,
            )
            torch.cuda.synchronize()

            for seq_idx, sl in enumerate(seq_lens_d):
                s = cu_d[seq_idx]
                nt = (sl + BT - 1) // BT
                for c_idx in range(nt):
                    cs = s + c_idx * BT
                    ce = min(cs + BT, cu_d[seq_idx + 1])
                    cl = ce - cs
                    o_chunk = o_step2[cs:ce, 0, 0]
                    mx = o_chunk.abs().max().item()
                    first_nonzero = -1
                    for r in range(cl):
                        if abs(o_chunk[r].item()) > 0.01:
                            first_nonzero = r
                            break
                    nz_str = f" first_nonzero_row={first_nonzero}" if first_nonzero >= 0 else ""
                    vals = [f"r{r}:{o_chunk[r].item():.2f}" for r in [0, 1, 2, min(cl-1, 63)]]
                    print(f"    Seq{seq_idx} c{c_idx} cs={cs}: max={mx:.4f} {' '.join(vals)}{nz_str}")
                    if seq_idx == 1 and mx > 0.01:
                        print(f"      → PROOF: kernel loaded v from T<100 for Seq1 c{c_idx}!")

        except Exception as e:
            import traceback
            print(f"  Step test failed: {e}")
            traceback.print_exc()

        # Test: Random data with occ=1 to check multi-CTA interference
        print("\n  --- [100,128] random data with occ=1 ---")
        try:
            seq_lens_o1 = [100, 128]
            T_o1 = 228
            cu_o1 = [0, 100, 228]
            ch_offsets_o1 = build_chunk_offsets(seq_lens_o1, BT=BT)
            nt_o1 = 4
            cu_t = torch.tensor(cu_o1, dtype=torch.int32, device=device)
            ci_t = build_chunk_indices(seq_lens_o1, BT=BT, device=device)
            cu_c = from_dlpack(cu_t.detach())
            ci_c = from_dlpack(ci_t.detach())
            stream = cutlass_torch.default_stream()
            ps_o1 = (2, T_o1, H, K, V)

            torch.manual_seed(42)
            q_o1 = torch.randn(T_o1, H, K, dtype=dtype, device=device)
            v_o1 = torch.randn(T_o1, H, V, dtype=dtype, device=device)
            g_o1 = torch.randn(T_o1, H, K, dtype=torch.float32, device=device) * 0.1
            h_o1 = torch.randn(nt_o1, H, K, V, dtype=dtype, device=device) * 0.01
            A_o1 = torch.randn(T_o1, H, BT, dtype=dtype, device=device) * 0.1

            o_ref_o1 = torch.zeros(T_o1, H, V, dtype=dtype, device=device)
            for si, sl in enumerate(seq_lens_o1):
                s, e = cu_o1[si], cu_o1[si + 1]
                co = ch_offsets_o1[si]
                nt_s = (sl + BT - 1) // BT
                o_s = reference_chunk_gla_fwd_o(
                    q_o1[s:e].unsqueeze(0), v_o1[s:e].unsqueeze(0),
                    g_o1[s:e].unsqueeze(0), h_o1[co:co+nt_s],
                    A_o1[s:e].unsqueeze(0), scale, BT)
                o_ref_o1[s:e] = o_s[0]

            o_out1 = torch.zeros(T_o1, H, V, dtype=dtype, device=device)
            k_occ1 = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True, min_occupancy=1)
            c_occ1 = cute.compile(
                k_occ1,
                from_dlpack(q_o1.detach()).iterator, from_dlpack(v_o1.detach()).iterator,
                from_dlpack(g_o1.detach()).iterator, from_dlpack(h_o1.detach()).iterator,
                from_dlpack(o_out1.detach()).iterator, from_dlpack(A_o1.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_o1, nt_o1, stream,
            )
            c_occ1(
                from_dlpack(q_o1.detach()).iterator, from_dlpack(v_o1.detach()).iterator,
                from_dlpack(g_o1.detach()).iterator, from_dlpack(h_o1.detach()).iterator,
                from_dlpack(o_out1.detach()).iterator, from_dlpack(A_o1.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_o1, nt_o1, stream,
            )
            torch.cuda.synchronize()
            for si, sl in enumerate(seq_lens_o1):
                s = cu_o1[si]
                nt_s = (sl + BT - 1) // BT
                for ci in range(nt_s):
                    cs = s + ci * BT
                    ce = min(cs + BT, cu_o1[si + 1])
                    cd = (o_ref_o1[cs:ce].float() - o_out1[cs:ce].float()).abs().max().item()
                    status = "PASS" if cd < 0.02 else "FAIL"
                    print(f"    occ=1 Seq{si} c{ci} (tok={cs}) max_diff={cd:.4f} [{status}]")
        except Exception as e:
            import traceback
            print(f"  occ=1 test failed: {e}")
            traceback.print_exc()

        # Test: compile once, run with [100, 128] isolating inter vs intra
        print("\n  --- Isolate inter vs intra for [100, 128] ---")
        try:
            seq_lens = [100, 128]
            num_seqs = 2
            T_total = 228
            cu_seqlens_list = [0, 100, 228]
            chunk_offsets_list = build_chunk_offsets(seq_lens, BT=BT)
            total_nt_val = chunk_offsets_list[-1]
            cu_seqlens_t = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
            ci_t = build_chunk_indices(seq_lens, BT=BT, device=device)
            cu_cute = from_dlpack(cu_seqlens_t.detach())
            ci_cute = from_dlpack(ci_t.detach())
            ps = (num_seqs, T_total, H, K, V)
            stream = cutlass_torch.default_stream()

            torch.manual_seed(42)
            q_flat = torch.randn(T_total, H, K, dtype=dtype, device=device)
            v_flat = torch.randn(T_total, H, V, dtype=dtype, device=device)
            g_flat = torch.randn(T_total, H, K, dtype=torch.float32, device=device) * 0.1
            h_flat = torch.randn(total_nt_val, H, K, V, dtype=dtype, device=device) * 0.01
            A_flat = torch.randn(T_total, H, BT, dtype=dtype, device=device) * 0.1

            def run_and_check(label, q, v, g, h, A):
                o_ref = torch.zeros(T_total, H, V, dtype=dtype, device=device)
                for seq_idx, sl in enumerate(seq_lens):
                    s, e = cu_seqlens_list[seq_idx], cu_seqlens_list[seq_idx + 1]
                    co = chunk_offsets_list[seq_idx]
                    nt_seq = (sl + BT - 1) // BT
                    o_seq = reference_chunk_gla_fwd_o(
                        q[s:e].unsqueeze(0), v[s:e].unsqueeze(0),
                        g[s:e].unsqueeze(0), h[co:co+nt_seq],
                        A[s:e].unsqueeze(0), scale, BT)
                    o_ref[s:e] = o_seq[0]

                o_flat = torch.zeros(T_total, H, V, dtype=dtype, device=device)
                kernel_vl = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True)
                compiled_vl = cute.compile(
                    kernel_vl,
                    from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
                    from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
                    from_dlpack(o_flat.detach()).iterator, from_dlpack(A.detach()).iterator,
                    cu_cute.iterator, ci_cute.iterator,
                    ps, total_nt_val, stream,
                )
                compiled_vl(
                    from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
                    from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
                    from_dlpack(o_flat.detach()).iterator, from_dlpack(A.detach()).iterator,
                    cu_cute.iterator, ci_cute.iterator,
                    ps, total_nt_val, stream,
                )
                torch.cuda.synchronize()
                # Per chunk
                for seq_idx, sl in enumerate(seq_lens):
                    s = cu_seqlens_list[seq_idx]
                    co = chunk_offsets_list[seq_idx]
                    nt_seq = (sl + BT - 1) // BT
                    for c in range(nt_seq):
                        cs = s + c * BT
                        ce = min(cs + BT, cu_seqlens_list[seq_idx + 1])
                        cd = (o_ref[cs:ce].float() - o_flat[cs:ce].float()).abs().max().item()
                        if cd > 0.02:
                            print(f"    {label}: Seq{seq_idx} c{c} (tok={cs}) max_diff={cd:.4f} FAIL")
                md = (o_ref.float() - o_flat.float()).abs().max().item()
                status = "PASS" if md < 0.02 else "FAIL"
                print(f"    {label}: max_diff={md:.6f} [{status}]")

            # Full test
            run_and_check("FULL", q_flat, v_flat, g_flat, h_flat, A_flat)

            # INTRA only: q=0, h=0 → only tril(A) @ v
            q_zero = torch.zeros_like(q_flat)
            h_zero = torch.zeros_like(h_flat)
            run_and_check("INTRA(q=0,h=0)", q_zero, v_flat, g_flat, h_zero, A_flat)

            # INTER only: A=0 → only scale * qg @ h
            A_zero = torch.zeros_like(A_flat)
            run_and_check("INTER(A=0)", q_flat, v_flat, g_flat, h_flat, A_zero)

            # V-only: q=0, h=0, A=identity-like (first column = 1, rest=0)
            # This tests purely the v load
            A_ones = torch.zeros_like(A_flat)
            A_ones[:, :, 0] = 1.0  # tril row 0 col 0 = 1
            run_and_check("V-LOAD(A=diag)", q_zero, v_flat, g_flat, h_zero, A_ones)

        except Exception as e:
            import traceback
            print(f"  Test failed: {e}")
            traceback.print_exc()

        # Diagnostic: Fresh-compile A-column test (no TMA descriptor reuse)
        print("\n  --- Fresh A-column diagnostic ---")
        try:
            seq_lens_diag = [100, 128]
            T_diag = 228
            cu_diag = [0, 100, 228]
            total_nt_diag = 4
            cu_t = torch.tensor(cu_diag, dtype=torch.int32, device=device)
            ci_t = build_chunk_indices(seq_lens_diag, BT=BT, device=device)
            cu_c = from_dlpack(cu_t.detach())
            ci_c = from_dlpack(ci_t.detach())
            stream = cutlass_torch.default_stream()

            # v = ones, q=0, g=0, h=0
            v_diag = torch.ones(T_diag, H, V, dtype=dtype, device=device)
            q_diag = torch.zeros(T_diag, H, K, dtype=dtype, device=device)
            g_diag = torch.zeros(T_diag, H, K, dtype=torch.float32, device=device)
            h_diag = torch.zeros(total_nt_diag, H, K, V, dtype=dtype, device=device)
            # A[t, h, j] = (j+1) * 0.01
            A_diag = torch.zeros(T_diag, H, BT, dtype=dtype, device=device)
            for j in range(BT):
                A_diag[:, :, j] = (j + 1) * 0.01

            o_diag = torch.zeros(T_diag, H, V, dtype=dtype, device=device)
            ps_diag = (2, T_diag, H, K, V)

            # FRESH compile with these exact tensors
            kvl = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True)
            compiled_d = cute.compile(
                kvl,
                from_dlpack(q_diag.detach()).iterator, from_dlpack(v_diag.detach()).iterator,
                from_dlpack(g_diag.detach()).iterator, from_dlpack(h_diag.detach()).iterator,
                from_dlpack(o_diag.detach()).iterator, from_dlpack(A_diag.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_diag, total_nt_diag, stream,
            )
            compiled_d(
                from_dlpack(q_diag.detach()).iterator, from_dlpack(v_diag.detach()).iterator,
                from_dlpack(g_diag.detach()).iterator, from_dlpack(h_diag.detach()).iterator,
                from_dlpack(o_diag.detach()).iterator, from_dlpack(A_diag.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_diag, total_nt_diag, stream,
            )
            torch.cuda.synchronize()

            # Expected: o[cs+i, h, v] = sum_{j=0}^{i} (j+1)*0.01 * 1 = 0.01*(i+1)*(i+2)/2
            for seq_idx, sl in enumerate(seq_lens_diag):
                s = cu_diag[seq_idx]
                nt = (sl + BT - 1) // BT
                for c in range(nt):
                    cs = s + c * BT
                    ce = min(cs + BT, cu_diag[seq_idx + 1])
                    cl = ce - cs
                    vals = []
                    first_bad = -1
                    for row in range(cl):
                        ex = 0.01 * (row + 1) * (row + 2) / 2.0
                        ac = o_diag[cs + row, 0, 0].item()
                        d = abs(ac - ex)
                        if row < 4 or row >= cl - 2:
                            vals.append(f"r{row}:{ac:.4f}/{ex:.4f}")
                        if d > max(0.1, abs(ex) * 0.05) and first_bad < 0:
                            first_bad = row
                    bad_str = f" first_bad={first_bad}" if first_bad >= 0 else " ALL_OK"
                    print(f"    Seq{seq_idx} c{c} cs={cs}: {' '.join(vals)}{bad_str}")
                    if first_bad >= 0:
                        for row in range(max(0, first_bad-1), min(cl, first_bad+5)):
                            ex = 0.01 * (row + 1) * (row + 2) / 2.0
                            ac = o_diag[cs + row, 0, 0].item()
                            print(f"      r{row}: act={ac:.6f} exp={ex:.6f} diff={ac-ex:.6f}")

            # Also fresh A-ones test for sanity
            print("    --- Fresh A-ones test ---")
            A_ones = torch.ones(T_diag, H, BT, dtype=dtype, device=device)
            o_ones = torch.zeros(T_diag, H, V, dtype=dtype, device=device)
            kvl2 = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale, is_varlen=True)
            compiled_d2 = cute.compile(
                kvl2,
                from_dlpack(q_diag.detach()).iterator, from_dlpack(v_diag.detach()).iterator,
                from_dlpack(g_diag.detach()).iterator, from_dlpack(h_diag.detach()).iterator,
                from_dlpack(o_ones.detach()).iterator, from_dlpack(A_ones.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_diag, total_nt_diag, stream,
            )
            compiled_d2(
                from_dlpack(q_diag.detach()).iterator, from_dlpack(v_diag.detach()).iterator,
                from_dlpack(g_diag.detach()).iterator, from_dlpack(h_diag.detach()).iterator,
                from_dlpack(o_ones.detach()).iterator, from_dlpack(A_ones.detach()).iterator,
                cu_c.iterator, ci_c.iterator,
                ps_diag, total_nt_diag, stream,
            )
            torch.cuda.synchronize()
            # Expected: o[cs+i] = sum_{j=0}^{i} 1*1 = i+1
            for seq_idx, sl in enumerate(seq_lens_diag):
                s = cu_diag[seq_idx]
                nt = (sl + BT - 1) // BT
                for c in range(nt):
                    cs = s + c * BT
                    ce = min(cs + BT, cu_diag[seq_idx + 1])
                    cl = ce - cs
                    md = max(abs(o_ones[cs + row, 0, 0].item() - float(row + 1)) for row in range(cl))
                    status = "OK" if md < 0.5 else "FAIL"
                    print(f"    Seq{seq_idx} c{c} cs={cs}: max_diff={md:.2f} [{status}]")

        except Exception as e:
            import traceback
            print(f"  Fresh A-col test failed: {e}")
            traceback.print_exc()

        # Original varlen tests
        test_configs = [
            [64],               # perfect single chunk
            [128],              # 2 chunks aligned
            [100],              # single seq, tail chunk
            [100, 128],         # non-aligned (was failing)
            [128, 100],         # different non-aligned
            [192, 100, 256, 50],  # mixed
            [228],              # single seq
            [128, 128],         # aligned control
        ]
        for run_id, seq_lens in enumerate(test_configs):
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.manual_seed(123)
                num_seqs = len(seq_lens)
                T_total = sum(seq_lens)
                cu_seqlens_list = [0]
                for sl in seq_lens:
                    cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
                chunk_offsets_list = build_chunk_offsets(seq_lens, BT=BT)
                total_nt_val = chunk_offsets_list[-1]

                cu_seqlens_t = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
                ci_t = build_chunk_indices(seq_lens, BT=BT, device=device)
                cu_cute = from_dlpack(cu_seqlens_t.detach())
                ci_cute = from_dlpack(ci_t.detach())
                ps = (num_seqs, T_total, H, K, V)

                # Random token-indexed inputs (used for reference)
                q_flat = torch.randn(T_total, H, K, dtype=dtype, device=device)
                v_flat = torch.randn(T_total, H, V, dtype=dtype, device=device)
                g_flat = torch.randn(T_total, H, K, dtype=torch.float32, device=device) * 0.1
                h_flat = torch.randn(total_nt_val, H, K, V, dtype=dtype, device=device) * 0.01
                A_flat = torch.randn(T_total, H, BT, dtype=dtype, device=device) * 0.1
                o_flat = torch.zeros(T_total, H, V, dtype=dtype, device=device)

                # Reference per-sequence (uses original token-indexed data)
                o_ref_flat = torch.zeros_like(o_flat)
                for seq_idx, sl in enumerate(seq_lens):
                    s = cu_seqlens_list[seq_idx]
                    e = cu_seqlens_list[seq_idx + 1]
                    co = chunk_offsets_list[seq_idx]
                    nt_seq = (sl + BT - 1) // BT
                    o_seq = reference_chunk_gla_fwd_o(
                        q_flat[s:e].unsqueeze(0), v_flat[s:e].unsqueeze(0),
                        g_flat[s:e].unsqueeze(0), h_flat[co:co+nt_seq],
                        A_flat[s:e].unsqueeze(0), scale, BT)
                    o_ref_flat[s:e] = o_seq[0]

                # === Test 1: varlen kernel (token-indexed inputs + domain_offset) ===
                stream = cutlass_torch.default_stream()
                kernel_vl = ChunkGlaFwdO(
                    chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale,
                    is_varlen=True,
                )
                compiled_vl = cute.compile(
                    kernel_vl,
                    from_dlpack(q_flat.detach()).iterator,
                    from_dlpack(v_flat.detach()).iterator,
                    from_dlpack(g_flat.detach()).iterator,
                    from_dlpack(h_flat.detach()).iterator,
                    from_dlpack(o_flat.detach()).iterator,
                    from_dlpack(A_flat.detach()).iterator,
                    cu_cute.iterator, ci_cute.iterator,
                    ps, total_nt_val, stream,
                )
                compiled_vl(
                    from_dlpack(q_flat.detach()).iterator,
                    from_dlpack(v_flat.detach()).iterator,
                    from_dlpack(g_flat.detach()).iterator,
                    from_dlpack(h_flat.detach()).iterator,
                    from_dlpack(o_flat.detach()).iterator,
                    from_dlpack(A_flat.detach()).iterator,
                    cu_cute.iterator, ci_cute.iterator,
                    ps, total_nt_val, stream,
                )
                torch.cuda.synchronize()

                max_diff = (o_ref_flat.float() - o_flat.float()).abs().max().item()
                status = "PASS" if max_diff < 0.02 else "FAIL"
                # Show tok offsets
                tok_offs = [cu_seqlens_list[i] for i in range(num_seqs)]
                aligned = all(t % BT == 0 for t in tok_offs)
                print(f"  seq_lens={seq_lens} T={T_total} tok_offs={tok_offs} aligned={aligned}: max_diff={max_diff:.6f} [{status}]")

                if max_diff >= 0.02:
                    # Per-chunk, per-head breakdown
                    for seq_idx, sl in enumerate(seq_lens):
                        s = cu_seqlens_list[seq_idx]
                        co = chunk_offsets_list[seq_idx]
                        nt_seq = (sl + BT - 1) // BT
                        for c in range(nt_seq):
                            cs = s + c * BT
                            ce = min(cs + BT, cu_seqlens_list[seq_idx + 1])
                            gc = co + c
                            diffs = [
                                (o_ref_flat[cs:ce, hh, :].float() - o_flat[cs:ce, hh, :].float()).abs().max().item()
                                for hh in range(H)
                            ]
                            if max(diffs) > 0.02:
                                dstr = " ".join(f"h{i}={d:.4f}" for i, d in enumerate(diffs))
                                print(f"    Seq{seq_idx} c{c} gc={gc} tok={cs} rem={sl-c*BT}: {dstr}")

            except Exception as e:
                import traceback
                print(f"  seq_lens={seq_lens}: ERROR - {e}")
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
