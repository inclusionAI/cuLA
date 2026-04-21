# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
chunk_gla_fwd_o — CuTe DSL Implementation for Blackwell SM100

Computes output O for chunkwise gated linear attention in KDA forward pass:

    O = scale * (q ⊙ 2^g) @ h  +  tril(Aqk) @ v_new

Inputs:
  q:  [B, T, H, K]           bf16  — query
  v:  [B, T, H, V]           bf16  — value (v_new from delta-rule)
  g:  [B, T, H, K]           fp32  — cumulative gate (log2 domain)
  h:  [B, NT, H, K, V]       bf16  — inter-chunk recurrent state
  A:  [B, T, H, BT]          bf16  — intra-chunk attention matrix (Aqk)

Output:
  o:  [B, T, H, V]           bf16

For KDA: K=V=128, BT=64, use_exp2=True always.
Scale is deferred to post-accumulation for better bf16 precision:
  qg = q * 2^g (no scale), AM = tril(A) / scale, output = scale * acc.

Kernel design (TMEM A-operand approach):
  Grid: (ceil(V/BV), NT, B*H)
  8 warps = 256 threads (occ=2 enabled)

  Warp specialization:
    Warps 0-3 (CUDA):
        - Read q, g from epilog SMEM, compute qg = q * exp2(g)
        - R2T write qg to TMEM (QG A-operand)
        - Read A from epilog SMEM, apply causal mask
        - R2T write A_masked to TMEM (AM A-operand)
        - After QH MMA: T2R read acc, multiply by scale, save to regs
        - After AV MMA: T2R read acc, add scaled QH, convert bf16
    Warp 4 (MMA):
        - QH MMA: qg(TMEM) × h(SMEM) → acc(TMEM)
        - Signal QH done, wait for CUDA read
        - AV MMA: am(TMEM) × v(SMEM) → acc(TMEM, fresh)
    Warp 5 (Load):
        - TMA G2S: q, g, A → epilog SMEM
        - TMA G2S: h, v → MMA B-operand SMEM
    Warp 6 (Store):
        - TMA S2G: sO → GMEM
    Warp 7 (Empty):
        - Required for warp group register redistribution

  TMEM layout (dual-ACC):
    ACC_QH: (BT, BV) fp32 — accumulator for QH MMA (qg@h)
    ACC_AV: (BT, BV) fp32 — accumulator for AV MMA (am@v)
    QG_A:   (BT, BK) bf16 — A-operand for QH MMA
    AM_A:   (BT, BT) bf16 — A-operand for AV MMA

  Pipeline:
    Load→CUDA: q, g, A   (PipelineTmaAsync, 1-stage)
    Load→MMA:  h, v       (PipelineTmaUmma, 1-stage)
    CUDA→MMA:  qg_ready   (PipelineAsyncUmma, 1-stage)
    CUDA→MMA:  am_ready   (PipelineAsyncUmma, 1-stage)
    MMA→CUDA:  acc_done   (PipelineUmmaAsync, 1-stage)
    CUDA→Store: o_ready   (PipelineAsync, 1-stage)

  Output epilog (dual-ACC approach):
    MMA: QH→acc_qh, AV→acc_av back-to-back, signal acc_done
    CUDA: T2R(acc_qh) + T2R(acc_av), o = scale*qh + av → BF16 → R2S
    Load warp: TMA store sO → GMEM
"""

import argparse

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64

from cula.utils import USE_FAST_MATH, assert_blackwell, prepare_chunk_indices

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
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        g_dtype: type[cutlass.Numeric] = cutlass.Float32,
        scale: float = 1.0,
        is_varlen: bool = False,
        BK: int = 128,
        BV: int = 128,
        min_occupancy: int = 2,
        persistent: bool = True,
        use_fast_math: bool = True,
    ):
        assert head_dim_k == 128 and head_dim_v == 128, (
            f"head_dim_k and head_dim_v must both be 128, got head_dim_k={head_dim_k}, head_dim_v={head_dim_v}"
        )
        assert_blackwell()
        self.use_fast_math = use_fast_math
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.g_dtype = g_dtype
        self.scale = scale
        self.is_varlen = is_varlen
        self.persistent = persistent

        self.BT = chunk_size  # 64
        self.BK = BK  # 128
        self.BV = BV  # 128

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
            # With occ=1 (65536 regs/CTA), give CUDA warps maximum registers
            # to eliminate register spilling (peak ~200 regs for dual-ACC
            # T2R epilog: 2×64 fp32 + bf16 buffers + overhead).
            # Store warp needs ~168 for bulk O tile prefetch (varlen).
            # Budget: 4×32×256 + 4×32×168 = 54272 ≤ 65536 ✓
            self.num_regs_cuda = 256
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
            barrier_id=2,
            num_threads=self.threads_per_cta,
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
        qh_tiled_mma,
        tile_qh,
        qg_tmem_layout,
        am_tmem_layout,
        acc_stages,
    ):
        """Plan TMEM offsets for dual-ACC, QG A-operand, AM A-operand.

        Dual-ACC layout: two separate ACC regions (one for QH MMA, one for AV MMA)
        so MMA can run QH→AV back-to-back without blocking on CUDA warp reads.
        """
        SM100_TMEM_CAPACITY_COLS = 512

        # ACC: (BT, BV) FP32
        acc_shape = qh_tiled_mma.partition_shape_C(tile_qh[:2])
        acc_fake = qh_tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))
        num_acc = tcgen05.find_tmem_tensor_col_offset(acc_fake)

        # QG A-operand: (BT, BK) BF16
        tCrQG_fake = qh_tiled_mma.make_fragment_A(qg_tmem_layout.outer.shape)
        num_qg = tcgen05.find_tmem_tensor_col_offset(tCrQG_fake)

        # Dual-ACC: acc_qh for QH MMA, acc_av for AV MMA
        acc_qh_off = 0
        acc_av_off = acc_qh_off + num_acc
        qg_off = acc_av_off + num_acc
        am_off = qg_off + num_qg

        total_tmp = am_off + num_qg  # conservative estimate
        total = 1
        while total < total_tmp:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS, f"TMEM overflow: {total} > {SM100_TMEM_CAPACITY_COLS}"
        if cutlass.const_expr(PRINT_DEBUG):
            print(
                f"  TMEM: ACC_QH={num_acc}@{acc_qh_off}, ACC_AV={num_acc}@{acc_av_off}, QG={num_qg}@{qg_off}, AM@{am_off}, total={total}"
            )
        return acc_qh_off, acc_av_off, qg_off, am_off, total

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,  # [B, T, H, K] (B=1 for varlen)
        v_in: cute.Tensor,  # [B, T, H, V] (B=1 for varlen)
        g_in: cute.Tensor,  # [B, T, H, K] fp32 (B=1 for varlen)
        h_in: cute.Tensor,  # [B, NT, H, K, V] (B=1 for varlen)
        o_in: cute.Tensor,  # [B, T, H, V] (B=1 for varlen)
        A_in: cute.Tensor,  # [B, T, H, BT] (B=1 for varlen)
        cu_seqlens_in: cute.Tensor,  # [N+1] int32
        chunk_indices_in: cute.Tensor,  # [NT, 2] int32
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,  # total chunks across all seqs (varlen)
        stream,
    ):
        # Extract pointers from tensor args (TVM-FFI compatible)
        q_ptr = q_in.iterator
        v_ptr = v_in.iterator
        g_ptr = g_in.iterator
        h_ptr = h_in.iterator
        o_ptr = o_in.iterator
        A_ptr = A_in.iterator
        cu_seqlens_ptr = cu_seqlens_in.iterator
        chunk_indices_ptr = chunk_indices_in.iterator

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

        # h: stored as [B, NT, H, K, V] — V contiguous
        # Transposed view for MMA B TMA: (V, K, (h_nt_total, H)) — V contiguous
        # non-varlen: h_nt_total = B * NT;  varlen: h_nt_total = total_nt (B=1)
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
            tcgen05.OperandMajorMode.K,  # A: K-major (TMEM requires K-major)
            tcgen05.OperandMajorMode.MN,  # B: MN-major (V contiguous in GMEM)
            self.acc_dtype,
            self.cta_group,
            self.qh_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,  # A from TMEM
        )

        # AV MMA: A=A_masked from TMEM, B=v from SMEM
        # B is MN-major because v_T GMEM has V(=N) contiguous
        av_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,  # A: K-major (TMEM requires K-major)
            tcgen05.OperandMajorMode.MN,  # B: MN-major (V contiguous in GMEM)
            self.acc_dtype,
            self.cta_group,
            self.av_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,  # A from TMEM
        )

        # ===================== TMEM layouts =====================
        # QG A-operand TMEM layout
        qg_tmem_layout = sm100_utils.make_smem_layout_a(
            qh_tiled_mma,
            self.qh_mma_tiler,
            self.io_dtype,
            1,
        )
        # AM A-operand TMEM layout
        am_tmem_layout = sm100_utils.make_smem_layout_a(
            av_tiled_mma,
            self.av_mma_tiler,
            self.io_dtype,
            1,
        )

        # ===================== TMEM offsets =====================
        (self.tmem_acc_qh_off, self.tmem_acc_av_off, self.tmem_qg_off, self.tmem_am_off, self.tmem_total) = (
            self._plan_tmem_offsets(
                qh_tiled_mma,
                self.qh_mma_tiler,
                qg_tmem_layout,
                am_tmem_layout,
                self.acc_stage,
            )
        )

        # ===================== SMEM layouts =====================
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        # Epilog SMEM for q (ROW_MAJOR BT×BK, bf16)
        q_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.q_stage,
        )
        # Epilog SMEM for g (ROW_MAJOR BT×BK, fp32)
        g_epi_staged = sm100_utils.make_smem_layout_epi(
            self.g_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.g_stage,
        )
        # Epilog SMEM for A (ROW_MAJOR BT×BT)
        a_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BT),
            self.a_stage,
        )
        # MMA B-operand SMEM for h (QH MMA B)
        h_smem_staged = sm100_utils.make_smem_layout_b(
            qh_tiled_mma,
            self.qh_mma_tiler,
            self.io_dtype,
            self.h_stage,
        )
        # MMA B-operand SMEM for v (AV MMA B)
        v_smem_staged = sm100_utils.make_smem_layout_b(
            av_tiled_mma,
            self.av_mma_tiler,
            self.io_dtype,
            self.v_stage,
        )
        # Output epilog for TMA store (ROW_MAJOR BT×BV)
        o_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BV),
            self.o_stage,
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
            tma_load_op,
            q,
            q_epi_smem,
            (self.BT, self.BK),
        )
        g_epi_smem = cute.select(g_epi_staged, mode=[0, 1])
        tma_atom_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            tma_load_op,
            g,
            g_epi_smem,
            (self.BT, self.BK),
        )
        a_epi_smem = cute.select(a_epi_staged, mode=[0, 1])
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            A,
            a_epi_smem,
            (self.BT, self.BT),
        )

        # h: MMA B TMA (transposed view)
        h_smem_1 = cute.select(h_smem_staged, mode=[0, 1, 2])
        tma_atom_h, tma_tensor_h = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            h_T,
            h_smem_1,
            self.qh_mma_tiler,
            qh_tiled_mma,
            cluster_layout.shape,
        )

        # v: MMA B TMA (transposed view)
        v_smem_1 = cute.select(v_smem_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v_T,
            v_smem_1,
            self.av_mma_tiler,
            av_tiled_mma,
            cluster_layout.shape,
        )

        # O: TMA store
        o_epi_smem = cute.select(o_epi_staged, mode=[0, 1])
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_epi_smem,
            (self.BT, self.BV),
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
            qg_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # CUDA→MMA: qg ready
            am_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # CUDA→MMA: am ready
            acc_done_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # MMA→CUDA: both ACC done
            o_ready_mbar: cute.struct.MemRange[Int64, self.o_stage * 2]  # CUDA→Load: o ready

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
        chunk_indices = cute.make_tensor(chunk_indices_ptr, cute.make_layout((total_nt, 2), stride=(2, 1)))

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
                (o_thr_dim0, o_thr_dim1),
                order=(1, 0),
            )  # mode 1 (BV) faster → coalesced GMEM writes
            o_val_layout = cute.make_layout((1, async_copy_elems))  # (1, 8)
            gmem_tiled_copy_o = cute.make_tiled_copy_tv(
                atom_universal_copy,
                o_thr_layout,
                o_val_layout,
            )
        else:
            gmem_tiled_copy_o = None

        # ===================== O GMEM tensor for varlen direct write =====================
        o_tensor = cute.make_tensor(o_ptr, o_layout)

        self.shared_storage = SharedStorage

        # ===================== Launch =====================
        self.kernel(
            qh_tiled_mma,
            av_tiled_mma,
            am_coord_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_g,
            tma_tensor_g,
            tma_atom_h,
            tma_tensor_h,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_o,
            tma_tensor_o,
            q_epi_staged,
            g_epi_staged,
            a_epi_staged,
            h_smem_staged,
            v_smem_staged,
            o_epi_staged,
            qg_tmem_layout,
            am_tmem_layout,
            cu_seqlens,
            chunk_indices,
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
        qh_tiled_mma,
        av_tiled_mma,
        am_coord_mma,
        tma_atom_q,
        tma_tensor_q,
        tma_atom_g,
        tma_tensor_g,
        tma_atom_h,
        tma_tensor_h,
        tma_atom_v,
        tma_tensor_v,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_o,
        tma_tensor_o,
        q_epi_staged,
        g_epi_staged,
        a_epi_staged,
        h_smem_staged,
        v_smem_staged,
        o_epi_staged,
        qg_tmem_layout,
        am_tmem_layout,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        o_tensor: cute.Tensor,
        gmem_tiled_copy_o,
        problem_size,
        total_nt,
    ):
        B, T, H, K, V = problem_size
        BT = self.BT

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
        # Dual ACC: separate regions for QH MMA and AV MMA results
        acc_shape = qh_tiled_mma.partition_shape_C(self.qh_mma_tiler[:2])
        tCtAcc_fake = qh_tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stage))
        tCtAcc_qh = cute.make_tensor(tmem_ptr + self.tmem_acc_qh_off, tCtAcc_fake.layout)
        tCtAcc_av = cute.make_tensor(tmem_ptr + self.tmem_acc_av_off, tCtAcc_fake.layout)

        if cutlass.const_expr(PRINT_DEBUG):
            print(f"acc_shape: {acc_shape}")
            print(f"tCtAcc_qh: {tCtAcc_qh}")
            print(f"tCtAcc_av: {tCtAcc_av}")

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
            cute.arch.setmaxregister_decrease(self.num_regs_others)

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
                        i_b = chunk_indices[(chunk_flat, 0)]
                        i_t = chunk_indices[(chunk_flat, 1)]
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
                    tma_atom_q,
                    tma_q_v[None, None, (i_h, data_bidx)],
                    (self.BT, self.BK),
                    sQ_epi,
                )
                bSG_sG, bSG_gG = self._epilog_partition_varlen(
                    tma_atom_g,
                    tma_g_v[None, None, (i_h, data_bidx)],
                    (self.BT, self.BK),
                    sG_epi,
                )
                bSG_sA, bSG_gA = self._epilog_partition_varlen(
                    tma_atom_a,
                    tma_a_v[None, None, (i_h, data_bidx)],
                    (self.BT, self.BT),
                    sA_epi,
                )
                tHsH, tHgH = self._tma_partition_B(
                    tma_atom_h,
                    tma_tensor_h,
                    sH,
                    self.qh_mma_tiler,
                    qh_tiled_mma,
                    i_tg,
                    i_h,
                )
                tVsV, tVgV = self._tma_partition_B(
                    tma_atom_v,
                    tma_v_v,
                    sV,
                    self.av_mma_tiler,
                    av_tiled_mma,
                    data_bidx,
                    i_h,
                )

                epi_tile_t = i_t
                for i_k in cutlass.range(self.num_k_tiles, unroll_full=True):
                    q_h = load_q_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_q,
                        src=bSG_gQ[(None, epi_tile_t, 0)],
                        dst=bSG_sQ[None, q_h.index],
                        tma_bar_ptr=q_h.barrier,
                    )
                    g_h = load_g_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_g,
                        src=bSG_gG[(None, epi_tile_t, 0)],
                        dst=bSG_sG[None, g_h.index],
                        tma_bar_ptr=g_h.barrier,
                    )
                    h_h = load_h_P.acquire_and_advance()
                    cute.copy(atom=tma_atom_h, src=tHgH[None, i_v, 0], dst=tHsH[None, h_h.index], tma_bar_ptr=h_h.barrier)

                v_h = load_v_P.acquire_and_advance()
                cute.copy(atom=tma_atom_v, src=tVgV[None, i_v, i_t], dst=tVsV[None, v_h.index], tma_bar_ptr=v_h.barrier)
                a_h = load_a_P.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_a, src=bSG_gA[(None, epi_tile_t, 0)], dst=bSG_sA[None, a_h.index], tma_bar_ptr=a_h.barrier
                )

        # =====================================================================
        # STORE WARP
        # =====================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

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
                    i_b = chunk_indices[(chunk_global_idx, 0)]
                    i_t = chunk_indices[(chunk_global_idx, 1)]
                    tok_offset = cu_seqlens[i_b]
                    seq_len = cu_seqlens[i_b + 1] - tok_offset
                    remaining = seq_len - i_t * BT
                    remaining = cutlass.select_(remaining > BT, Int32(BT), remaining)

                    # Bulk prefetch: SMEM → registers (all 256 bf16 at once)
                    cute.autovec_copy(tOsO, tOrO)
                    o_chunk_raw = o_tensor.iterator + (tok_offset + i_t * BT) * H * V + i_h * V + i_v * self.BV
                    o_chunk_ptr = cute.make_ptr(
                        self.io_dtype,
                        o_chunk_raw.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    o_stride_bt = cute.assume(
                        H * V,
                        divby=128 // self.io_dtype.width,
                    )
                    gO_chunk = cute.make_tensor(
                        o_chunk_ptr,
                        cute.make_layout(
                            (self.BT, self.BV),
                            stride=(o_stride_bt, 1),
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
                        tma_atom_o,
                        gO,
                        (self.BT, self.BV),
                        sO,
                    )
                    cute.copy(tma_atom_o, bSG_sO[None, 0], bSG_gO[(None, i_t, i_v)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o_h.release()
            else:
                # ---- Non-persistent non-varlen: single TMA store ----
                gO = tma_tensor_o[None, None, (i_h, data_bidx)]
                _, bSG_sO, bSG_gO = self._epilog_partition(
                    tma_atom_o,
                    gO,
                    (self.BT, self.BV),
                    sO,
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
            cute.arch.setmaxregister_decrease(self.num_regs_others)

        # =====================================================================
        # MMA WARP
        # =====================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                # Phase 1: QH MMA — qg(TMEM) × h(SMEM) → acc_qh(TMEM)
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
                            tCtAcc_qh[None, None, None, 0],
                            tCrQG_tmem[None, None, kp, 0],
                            tCrH_B[None, None, kp, h_h.index],
                            tCtAcc_qh[None, None, None, 0],
                        )

                    h_h.release()

                qg_h.release()

                # Phase 2: AV MMA — am(TMEM) × v(SMEM) → acc_av(TMEM)
                # No barrier needed: QH writes to acc_qh, AV writes to acc_av (separate TMEM regions)
                am_h = am_C.wait_and_advance()
                acc_h = acc_done_P.acquire_and_advance()
                v_h = load_v_C.wait_and_advance()

                for kp in cutlass.range(cute.size(tCrV_B, mode=[2]), unroll_full=True):
                    av_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        cutlass.Boolean(kp != 0),
                    )
                    cute.gemm(
                        av_tiled_mma,
                        tCtAcc_av[None, None, None, 0],
                        tCrAM_tmem[None, None, kp, 0],
                        tCrV_B[None, None, kp, v_h.index],
                        tCtAcc_av[None, None, None, 0],
                    )

                am_h.release()
                v_h.release()

                # Signal both ACCs done to CUDA warps
                acc_h.commit()

        # =====================================================================
        # CUDA WARPS: Gating + Masking → TMEM
        # =====================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.setmaxregister_increase(self.num_regs_cuda)

            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))
            scale_f32 = Float32(self.scale)

            # ---- Hoist loop-invariant T2R/R2T/R2S setup ----
            # All of these depend only on compile-time constants and local_tidx.
            # Hoisting them out of the persistent loop reduces loop body size
            # and instruction cache pressure.

            # T2R for ACC (FP32, BT×BV=64×128) — for QG coordinate mapping
            t2r_atom_acc = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAcc_qh_flat = tCtAcc_qh[((None, None), 0, 0, None)]
            tCtAcc_av_flat = tCtAcc_av[((None, None), 0, 0, None)]
            fake_sQG = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.qh_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_acc = tcgen05.make_tmem_copy(t2r_atom_acc, tCtAcc_qh_flat[(None, None, 0)])
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

            # Register buffer for AV acc result
            tTR_rAV_fp32 = cute.make_rmem_tensor(tTR_rQG_fp32.shape, self.acc_dtype)

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

            # AM coordinate mapping via R2T partition_S(identity)
            cM_am_r4 = cute.make_identity_tensor(tCrAM_tmem.layout.shape)
            tRS_cM_am_full = thr_r2t_am.partition_S(cM_am_r4)
            tRS_cM_am = cute.slice_(tRS_cM_am_full, (None, None, None, None, 0))

            # R2S: ACC T2R regs → sO (ROW_MAJOR, BT×BV)
            r2s_atom_o = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.acc_dtype,
                tiled_t2r_acc,
            )
            tiled_r2s_o = cute.make_tiled_copy_D(r2s_atom_o, tiled_t2r_acc)
            thr_r2s_o = tiled_r2s_o.get_slice(local_tidx)
            tRS_sO = thr_r2s_o.partition_D(sO)

            # Output epilog setup — dual T2R sources for QH and AV accumulators
            tTR_tAcc_qh = thr_t2r_acc.partition_S(tCtAcc_qh_flat)
            tTR_tAcc_av = thr_t2r_acc.partition_S(tCtAcc_av_flat)

            # ====== Persistent computation loop ======
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                if cutlass.const_expr(self.persistent and self.is_varlen):
                    # Work decode for remaining (persistent varlen)
                    work_idx = block_idx_x + wu_iter * grid_dim_x
                    i_v = work_idx % num_v_tiles
                    temp_work = work_idx // num_v_tiles
                    chunk_global_idx = temp_work % total_nt
                    i_h = temp_work // total_nt
                    i_b = chunk_indices[(chunk_global_idx, 0)]
                    i_t = chunk_indices[(chunk_global_idx, 1)]
                    tok_offset = cu_seqlens[i_b]
                    seq_len = cu_seqlens[i_b + 1] - tok_offset
                    remaining = seq_len - i_t * BT
                    remaining = cutlass.select_(remaining > BT, Int32(BT), remaining)

                # ============ Compute QG: q * exp2(g) ============
                # Scale is deferred to post-accumulation for better bf16 precision.
                # Varlen: unconditional SMEM loads + branchless select.
                for i_k in cutlass.range(self.num_k_tiles, unroll_full=True):
                    q_h = load_q_C.wait_and_advance()
                    g_h = load_g_C.wait_and_advance()

                    for ei in cutlass.range(cute.size(tTR_rQG_fp32), unroll_full=True):
                        bt_coord, bk_coord = tTR_cM_qg[ei]
                        if cutlass.const_expr(self.is_varlen):
                            # Unconditional loads — safe because SMEM always
                            # contains valid data (from this or next sequence).
                            # Branchless select zeros out-of-bounds results.
                            q_val = sQ_epi[(bt_coord, bk_coord, q_h.index)].to(self.acc_dtype)
                            g_val = sG_epi[(bt_coord, bk_coord, g_h.index)]
                            result = q_val * cute.exp2(g_val, fastmath=self.use_fast_math)
                            tTR_rQG_fp32[ei] = cutlass.select_(bt_coord < remaining, result, Float32(0.0))
                        else:
                            q_val = sQ_epi[(bt_coord, bk_coord, q_h.index)].to(self.acc_dtype)
                            g_val = sG_epi[(bt_coord, bk_coord, g_h.index)]
                            tTR_rQG_fp32[ei] = q_val * cute.exp2(g_val, fastmath=self.use_fast_math)

                    q_h.release()
                    g_h.release()

                    tRT_rQG_bf16.store(tTR_rQG_fp32.load().to(self.io_dtype))
                    qg_h = qg_P.acquire_and_advance()
                    cute.copy(tiled_r2t_qg, tRT_rQG_bf16, tRT_tQG[(None, None, None, None, 0)])
                    cute.arch.fence_view_async_tmem_store()
                    qg_h.commit()

                # ============ Compute AM: tril(A) with varlen boundary mask ============
                a_h = load_a_C.wait_and_advance()

                for ei in cutlass.range(cute.size(tRT_rAM), unroll_full=True):
                    coord_val = tRS_cM_am[ei]
                    m0, m1, m2, m3 = coord_val
                    sub0, sub1 = m0
                    sub0_0, sub0_1 = sub0
                    row = sub0_0 + sub0_1 * 16
                    col = sub1 + m2 * 16
                    if cutlass.const_expr(self.is_varlen):
                        # row >= col is constexpr; row < remaining implies
                        # col < remaining since col <= row.
                        if row >= col:
                            a_val = sA_epi[(row, col, a_h.index)]
                            tRT_rAM[ei] = cutlass.select_(row < remaining, a_val, Float32(0.0).to(self.io_dtype))
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

                # ============ Dual-ACC Epilog: read both accumulators, combine ============
                # Sequential T2R to reduce peak register pressure:
                # Phase 1: read QH acc, fence, scale in-place (64 fp32 regs live)
                # Phase 2: read AV acc, fence, add to scaled QH (128 fp32 regs briefly)
                acc_h = acc_done_C.wait_and_advance()

                # Read QH accumulator (qg@h) from TMEM → fp32 registers
                cute.copy(tiled_t2r_acc, tTR_tAcc_qh[(None, None, None, 0)], tTR_rQG_fp32)
                cute.arch.fence_view_async_tmem_load()
                # Scale QH in-place while only 64 fp32 regs are live
                tTR_rQG_fp32.store(tTR_rQG_fp32.load() * Float32(scale_f32))

                # Read AV accumulator (am@v) from TMEM → fp32 registers
                cute.copy(tiled_t2r_acc, tTR_tAcc_av[(None, None, None, 0)], tTR_rAV_fp32)
                cute.arch.fence_view_async_tmem_load()
                acc_h.release()

                # o = scaled_qh + av
                tTR_rQG_fp32.store(tTR_rQG_fp32.load() + tTR_rAV_fp32.load())

                tTR_rAcc_bf16 = cute.make_rmem_tensor(tTR_rQG_fp32.shape, self.io_dtype)
                tTR_rAcc_bf16.store(tTR_rQG_fp32.load().to(self.io_dtype))

                tRS_rO = tiled_r2s_o.retile(tTR_rAcc_bf16)
                o_h = o_ready_P.acquire_and_advance()
                cute.copy(tiled_r2s_o, tRS_rO, tRS_sO[(None, None, None, 0)])
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
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
        gX = cute.local_tile(tma_tensor, tiler, (None, None, (batch_idx, head_idx)))
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_tma_partition_B: gX (local_tile result) = {gX}")
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"_tma_partition_B: tCgX (partition_B result) = {tCgX}")
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
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
            epi_tile,  # (BT, BK) or (BT, BT) — tiles first 2 modes
            (None, None, (head_idx, batch_idx)),  # keep T/K tiles, fix mode2
        )
        # gC: (BT, BK/BT, NT, NK) — mode2 consumed by coord selection
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
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
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
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
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
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
            chunked[co + c, :chunk_len] = tensor[src_start : src_start + chunk_len]
    return chunked


def prepare_v_chunked(v, cu_seqlens, chunk_offsets, BT=64):
    """Backward-compatible wrapper: v has an extra leading batch dim [1, T_total, H, V]."""
    return prepare_chunked(v[0], cu_seqlens, chunk_offsets, BT)


def build_chunk_indices(seq_lens, BT=64, device="cuda"):
    """
    Build chunk_indices tensor in the same format as FLA's prepare_chunk_indices.

    Returns an int32 tensor of shape [NT, 2], where each row is
    (batch_idx, chunk_seq_idx).  This matches the kda_bwd decode_tile_coord
    scheme: chunk_indices[i, 0] = batch_idx, chunk_indices[i, 1] = chunk_in_seq.

    Args:
        seq_lens: list of sequence lengths
        BT: chunk size (default 64)
        device: torch device

    Returns:
        chunk_indices: [NT, 2] int32 tensor
    """
    import torch

    pairs = []
    for seq_idx, sl in enumerate(seq_lens):
        nt = (sl + BT - 1) // BT
        for c in range(nt):
            pairs.append([seq_idx, c])
    return torch.tensor(pairs, dtype=torch.int32, device=device).reshape(-1, 2)


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
        h: [B, NT, H, K, V] recurrent state
        A: [B, T, H, BT] intra-chunk attention matrix (Aqk)
        scale: float
        chunk_size: int

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
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
                qg = (q_chunk.float() * (2.0**g_chunk)).to(q.dtype)

                h_state = h[b, i_t, i_h, :, :]

                o_inter = scale * (qg.float() @ h_state.float())

                A_chunk = A[b, t_start:t_end, i_h, :actual_bt]
                mask = torch.tril(torch.ones(actual_bt, actual_bt, device=A.device))
                A_masked = (A_chunk * mask).to(v.dtype)

                v_chunk = v[b, t_start:t_end, i_h, :]
                o_intra = A_masked.float() @ v_chunk.float()

                o[b, t_start:t_end, i_h, :] = (o_inter + o_intra).to(o.dtype)

    return o


# ---------------------------------------------------------------------------
# Compile cache + TVM-FFI API
# ---------------------------------------------------------------------------

# Internal cache: maps (is_varlen, persistent, H, K, V, scale, chunk_size) → compiled_fn
_fwd_o_kernel_cache: dict = {}

# Pre-allocated dummy tensors for non-varlen path (avoid per-call torch.zeros)
_fwd_o_dummy_cu_seqlens: torch.Tensor = None
_fwd_o_dummy_chunk_indices: torch.Tensor = None


def _compile_fwd_o_variant(is_varlen, persistent, H, K, V, scale, chunk_size, use_fast_math):
    """Compile one ChunkGlaFwdO kernel variant. Returns the compiled TVM-FFI callable.

    Uses make_fake_compact_tensor and make_fake_stream for compilation with
    TVM-FFI.  At runtime, torch tensors are passed directly (zero-copy).
    Uses sym_int() for dynamic B, T, NT dimensions so one compiled kernel
    handles all batch-size / sequence-length combinations.
    """
    kernel_obj = ChunkGlaFwdO(
        chunk_size=chunk_size,
        head_dim_k=K,
        head_dim_v=V,
        scale=scale,
        is_varlen=is_varlen,
        persistent=persistent,
        use_fast_math=use_fast_math,
    )

    sym_a = cute.sym_int()  # B (non-varlen: dynamic B; varlen: fixed 1)
    sym_b = cute.sym_int()  # T (non-varlen) or T_total (varlen)
    sym_nt = cute.sym_int()  # NT_total
    sym_cu = cute.sym_int()  # cu_seqlens size
    sym_ci = cute.sym_int()  # chunk_indices size

    BT = chunk_size

    if is_varlen:
        # varlen: tensors are [1, T_total, H, ...] (4D with B=1)
        # This avoids squeeze(0) CPU overhead at the call site.
        q_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (1, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        v_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (1, sym_b, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        g_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (1, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        o_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (1, sym_b, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        A_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (1, sym_b, H, BT),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
    else:
        # non-varlen: tensors are [B, T, H, ...] (4D)
        q_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        v_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        g_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (sym_a, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        o_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        A_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_b, H, BT),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )

    if is_varlen:
        # varlen: h is [1, NT_total, H, K, V] (5D with B=1)
        h_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (1, sym_nt, H, K, V),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=128,
        )
    else:
        # non-varlen: h is [B, NT, H, K, V] (5D)
        h_fake = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_a, sym_nt, H, K, V),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=128,
        )
    # chunk_indices is [NT, 2]
    ci_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_ci, 2),
        stride_order=(1, 0),
        assumed_align=128,
    )
    cu_fake = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_cu,),
        assumed_align=128,
    )
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        q_fake,
        v_fake,
        g_fake,
        h_fake,
        o_fake,
        A_fake,
        cu_fake,
        ci_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        Int32(1),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_fn


def _get_compiled_fwd_o(is_varlen, persistent, H, K, V, scale, chunk_size):
    """Get a compiled ChunkGlaFwdO kernel with on-demand (lazy) compilation.

    Each variant is compiled exactly once and cached.  Compilation is deferred
    until the variant is actually needed so that cute.compile is always
    immediately followed by execution — this avoids a CuTe DSL runtime issue
    where a subsequent cute.compile can invalidate previously compiled but
    not-yet-executed functions.

    Cache key: (is_varlen, persistent, H, K, V, scale, chunk_size, USE_FAST_MATH)
    """
    key = (is_varlen, persistent, H, K, V, scale, chunk_size, USE_FAST_MATH)
    if key not in _fwd_o_kernel_cache:
        _fwd_o_kernel_cache[key] = _compile_fwd_o_variant(
            is_varlen,
            persistent,
            H,
            K,
            V,
            scale,
            chunk_size,
            USE_FAST_MATH,
        )
    return _fwd_o_kernel_cache[key]


def chunk_gla_fwd_o(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor = None,
    chunk_indices: torch.Tensor = None,
    is_varlen: bool = False,
    persistent: bool = True,
) -> None:
    """
    ChunkGlaFwdO forward pass with compile cache and TVM-FFI.

    Computes:  o = scale * (q ⊙ 2^g) @ h  +  tril(Aqk) @ v_new

    Uses make_fake_compact_tensor for compilation (no GC issues).
    At runtime, torch tensors are passed directly via TVM-FFI.
    sym_int() is used for B, T, NT so a single compilation handles all
    batch-size / sequence-length combinations.

    Cache key: (is_varlen, persistent, H, K, V, scale, chunk_size)

    Args:
        q: query tensor — [B, T, H, K] bf16 (both non-varlen and varlen with B=1)
        v: value tensor — [B, T, H, V] bf16 (both non-varlen and varlen with B=1)
        g: gate tensor — [B, T, H, K] fp32 (both non-varlen and varlen with B=1)
        h: state tensor — [B, NT, H, K, V] bf16 (B=1 for varlen)
        o: output tensor (pre-allocated) — same shape as q but with V dim
        A: attention matrix — [B, T, H, BT] bf16 (both non-varlen and varlen with B=1)
        scale: attention scale factor
        chunk_size: chunk size (default: 64)
        cu_seqlens: cumulative sequence lengths [N+1] int32 (varlen only)
        chunk_indices: chunk index pairs [NT, 2] int32
        is_varlen: whether to use varlen mode
        persistent: whether to use persistent kernel (default: True)
    """
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if is_varlen:
        assert cu_seqlens is not None and chunk_indices is not None, (
            "cu_seqlens and chunk_indices are required for varlen mode"
        )
        assert q.dim() == 4 and q.shape[0] == 1, f"varlen mode expects [1, T_total, H, K] input, got shape {q.shape}"
        assert h.dim() == 5 and h.shape[0] == 1, f"varlen mode expects [1, NT_total, H, K, V] for h, got shape {h.shape}"
        T_total = q.shape[1]
        H = q.shape[2]
        K = q.shape[3]
        V = v.shape[3]
        num_seqs = cu_seqlens.shape[0] - 1
        total_nt_val = chunk_indices.shape[0]
        ps = (Int32(num_seqs), Int32(T_total), Int32(H), Int32(K), Int32(V))
    else:
        B, T, H, K = q.shape
        V = v.shape[3]
        NT = (T + chunk_size - 1) // chunk_size
        total_nt_val = B * NT
        ps = (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V))
        if cu_seqlens is None:
            global _fwd_o_dummy_cu_seqlens
            if _fwd_o_dummy_cu_seqlens is None or _fwd_o_dummy_cu_seqlens.device != q.device:
                _fwd_o_dummy_cu_seqlens = torch.zeros(2, dtype=torch.int32, device=q.device)
            cu_seqlens = _fwd_o_dummy_cu_seqlens
        if chunk_indices is None:
            global _fwd_o_dummy_chunk_indices
            if _fwd_o_dummy_chunk_indices is None or _fwd_o_dummy_chunk_indices.device != q.device:
                _fwd_o_dummy_chunk_indices = torch.zeros((1, 2), dtype=torch.int32, device=q.device)
            chunk_indices = _fwd_o_dummy_chunk_indices

    compiled_fn = _get_compiled_fwd_o(
        is_varlen,
        persistent,
        H,
        K,
        V,
        scale,
        chunk_size,
    )

    # TVM-FFI: pass torch tensors directly; stream is auto-provided
    # by make_fake_stream(use_tvm_ffi_env_stream=True).
    compiled_fn(
        q,
        v,
        g,
        h,
        o,
        A,
        cu_seqlens,
        chunk_indices,
        ps,
        Int32(total_nt_val),
    )


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Chunk GLA FWD O kernel test")
    parser.add_argument("--test", type=str, default="correctness", choices=["correctness", "benchmark", "both"])
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    if args.scale is None:
        args.scale = args.K**-0.5
    B, T, H, K, V = args.B, args.T, args.H, args.K, args.V
    BT = args.chunk_size
    scale = args.scale
    NT = (T + BT - 1) // BT
    dtype, device = torch.bfloat16, "cuda"

    print(f"Config: B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}, scale={scale:.4f}")
    print(f"  Chunks per seq: {NT}, Total chunks: {B * NT}")

    if args.test in ("correctness", "both"):
        all_pass = True

        # ----- Non-varlen correctness test -----
        print("\n=== Non-Varlen Correctness Test ===")
        torch.manual_seed(42)
        q_nv = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v_nv = torch.randn(B, T, H, V, dtype=dtype, device=device)
        g_nv = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
        h_nv = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
        A_nv = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

        o_ref_nv = reference_chunk_gla_fwd_o(q_nv, v_nv, g_nv, h_nv, A_nv, scale, BT)
        o_nv = torch.zeros(B, T, H, V, dtype=dtype, device=device)

        chunk_gla_fwd_o(
            q=q_nv,
            v=v_nv,
            g=g_nv,
            h=h_nv,
            o=o_nv,
            A=A_nv,
            scale=scale,
            chunk_size=BT,
            is_varlen=False,
        )
        torch.cuda.synchronize()

        max_diff = (o_ref_nv.float() - o_nv.float()).abs().max().item()
        status = "PASS" if max_diff < 0.02 else "FAIL"
        print(f"  Non-varlen: max_diff={max_diff:.6f} [{status}]")
        all_pass = all_pass and (max_diff < 0.02)

        # ----- Varlen correctness test -----
        print("\n=== Varlen Correctness Test ===")
        test_configs = [
            [64],
            [128],
            [100],
            [100, 128],
            [128, 100],
            [192, 100, 256, 50],
            [228],
            [128, 128],
        ]
        for seq_lens in test_configs:
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

                q_flat = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
                v_flat = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
                g_flat = torch.randn(1, T_total, H, K, dtype=torch.float32, device=device) * 0.1
                h_flat = torch.randn(1, total_nt_val, H, K, V, dtype=dtype, device=device) * 0.01
                A_flat = torch.randn(1, T_total, H, BT, dtype=dtype, device=device) * 0.1
                o_flat = torch.zeros(1, T_total, H, V, dtype=dtype, device=device)

                # Reference per-sequence
                o_ref_flat = torch.zeros_like(o_flat)
                for seq_idx, sl in enumerate(seq_lens):
                    s = cu_seqlens_list[seq_idx]
                    e = cu_seqlens_list[seq_idx + 1]
                    co = chunk_offsets_list[seq_idx]
                    nt_seq = (sl + BT - 1) // BT
                    o_seq = reference_chunk_gla_fwd_o(
                        q_flat[:, s:e], v_flat[:, s:e], g_flat[:, s:e], h_flat[:, co : co + nt_seq], A_flat[:, s:e], scale, BT
                    )
                    o_ref_flat[:, s:e] = o_seq

                chunk_gla_fwd_o(
                    q=q_flat,
                    v=v_flat,
                    g=g_flat,
                    h=h_flat,
                    o=o_flat,
                    A=A_flat,
                    scale=scale,
                    chunk_size=BT,
                    cu_seqlens=cu_seqlens_t,
                    chunk_indices=ci_t,
                    is_varlen=True,
                )
                torch.cuda.synchronize()

                max_diff = (o_ref_flat.float() - o_flat.float()).abs().max().item()
                status = "PASS" if max_diff < 0.02 else "FAIL"
                tok_offs = [cu_seqlens_list[i] for i in range(num_seqs)]
                aligned = all(t % BT == 0 for t in tok_offs)
                print(f"  seq_lens={seq_lens} T={T_total} aligned={aligned}: max_diff={max_diff:.6f} [{status}]")
                all_pass = all_pass and (max_diff < 0.02)

            except Exception as e:
                import traceback

                print(f"  seq_lens={seq_lens}: ERROR - {e}")
                traceback.print_exc()
                all_pass = False

        # ----- Cache reuse test (same variant, different data) -----
        print("\n=== Cache Reuse Test ===")
        for i in range(3):
            torch.manual_seed(i * 100)
            q_cr = torch.randn(B, T, H, K, dtype=dtype, device=device)
            v_cr = torch.randn(B, T, H, V, dtype=dtype, device=device)
            g_cr = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
            h_cr = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
            A_cr = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1
            o_cr = torch.zeros(B, T, H, V, dtype=dtype, device=device)
            o_ref_cr = reference_chunk_gla_fwd_o(q_cr, v_cr, g_cr, h_cr, A_cr, scale, BT)

            chunk_gla_fwd_o(
                q=q_cr,
                v=v_cr,
                g=g_cr,
                h=h_cr,
                o=o_cr,
                A=A_cr,
                scale=scale,
                chunk_size=BT,
                is_varlen=False,
            )
            torch.cuda.synchronize()
            md = (o_ref_cr.float() - o_cr.float()).abs().max().item()
            status = "PASS" if md < 0.02 else "FAIL"
            print(f"  Run {i + 1}/3: max_diff={md:.6f} [{status}]")
            all_pass = all_pass and (md < 0.02)

        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")

    if args.test in ("benchmark", "both"):
        print("\n=== Benchmark ===")
        torch.manual_seed(42)
        for bench_T in [1024, 2048, 4096]:
            bench_NT = (bench_T + BT - 1) // BT
            q_b = torch.randn(B, bench_T, H, K, dtype=dtype, device=device)
            v_b = torch.randn(B, bench_T, H, V, dtype=dtype, device=device)
            g_b = torch.randn(B, bench_T, H, K, dtype=torch.float32, device=device) * 0.1
            h_b = torch.randn(B, bench_NT, H, K, V, dtype=dtype, device=device) * 0.01
            A_b = torch.randn(B, bench_T, H, BT, dtype=dtype, device=device) * 0.1
            o_b = torch.zeros(B, bench_T, H, V, dtype=dtype, device=device)

            # Warmup (also triggers lazy compilation if needed)
            for _ in range(3):
                chunk_gla_fwd_o(
                    q=q_b,
                    v=v_b,
                    g=g_b,
                    h=h_b,
                    o=o_b,
                    A=A_b,
                    scale=scale,
                    chunk_size=BT,
                    is_varlen=False,
                )
            torch.cuda.synchronize()

            # Benchmark
            N_iters = 100
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(N_iters):
                chunk_gla_fwd_o(
                    q=q_b,
                    v=v_b,
                    g=g_b,
                    h=h_b,
                    o=o_b,
                    A=A_b,
                    scale=scale,
                    chunk_size=BT,
                    is_varlen=False,
                )
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / N_iters
            print(f"  CuTe DSL T={bench_T}: {ms:.3f} ms")

            try:
                from fla.ops.gla.chunk import chunk_gla_fwd_o_gk

                for _ in range(10):
                    chunk_gla_fwd_o_gk(
                        q=q_b, v=v_b, g=g_b, A=A_b, h=h_b.flatten(0, 1), scale=scale, chunk_size=BT, use_exp2=True
                    )
                torch.cuda.synchronize()
                start.record()
                for _ in range(N_iters):
                    chunk_gla_fwd_o_gk(
                        q=q_b, v=v_b, g=g_b, A=A_b, h=h_b.flatten(0, 1), scale=scale, chunk_size=BT, use_exp2=True
                    )
                end.record()
                torch.cuda.synchronize()
                ms_fla = start.elapsed_time(end) / N_iters
                print(f"  Triton T={bench_T}: {ms_fla:.3f} ms  speedup={ms_fla / ms:.2f}x")
            except Exception as e:
                print(f"  Triton benchmark failed: {e}")


if __name__ == "__main__":
    main()
