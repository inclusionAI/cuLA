# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
CuTeDSL kernel for KDA recompute_w_u_fwd.

Computes (always with gk, always recompute):
  w  = A @ (k * beta * exp2(gk))     — [BT, BK] per tile
  u  = A @ (v * beta)                — [BT, BV] per tile
  kg = k * exp2(gn - gk)             — [BT, BK] per tile

MMA layout (tcgen05 SS):
  C[M, N] = A_mma[M, K] @ B_mma[N, K]^T
  A_mma(SMEM) = A_mat[BT, BT], B_mma(SMEM) = B_proc^T[BN, BT]
  Result = output[BT, BN], matching the time-major GMEM layout.
  MMA tiler = (BT, BN, BT).

The implementation mirrors the csrc SM100 persistent kernel: one CTA per SM,
three warp-groups, double-buffered input TMA pipelines, auxiliary beta loads,
and CUDA-core W/U stores directly from TMEM.

Varlen mode: variable sequence lengths. cu_seqlens[N+1] gives token
  offsets; chunk_indices[total_nt*2] gives (batch_idx, chunk_in_seq)
  pairs for each global chunk index. TMA uses domain_offset for per-WU
  alignment, matching the fwd_o.py pattern.

Warp assignment:
  0-3: K prologue + kg output warpgroup
  4-7: V prologue + w/u epilogue warpgroup
  8:   MMA warp
  9:   Load warp
  10-11: beta-load auxiliary warps
"""

import weakref

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import Int32, Int64
from fla.ops.utils import prepare_chunk_indices

from cula.ops.ptx import reinterpret_cast, store_256b, subvec
from cula.ops.sm100.ptx import tcgen05_fence_after, tcgen05_fence_before, tcgen05_ld_32x32b
from cula.utils import USE_FAST_MATH, assert_blackwell


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
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        beta_dtype: type[cutlass.Numeric] = cutlass.Float32,
        is_varlen: bool = False,
        preprocessed_k: bool = False,
        persistent: bool = False,
        use_fast_math: bool = True,
    ):
        assert K == 128 and V == 128, f"K and V must both be 128, got K={K}, V={V}"
        assert_blackwell()
        self.use_fast_math = use_fast_math
        self.K = K
        self.V = V
        self.BT = chunk_size
        self.BK = block_k if block_k is not None else K
        self.BV = block_v if block_v is not None else V
        self.NK = (K + self.BK - 1) // self.BK
        self.NV = (V + self.BV - 1) // self.BV
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype
        self.beta_dtype = beta_dtype
        self.is_varlen = is_varlen
        self.preprocessed_k = preprocessed_k

        self.threads_per_warp = 32
        self.prologue_warp_ids = (0, 1, 2, 3)
        self.epilogue_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.aux_warp_ids = (10, 11)
        self.threads_per_cta = self.threads_per_warp * 12  # 384
        self.num_cuda_warps = 4
        self.num_cuda_threads = self.threads_per_warp * self.num_cuda_warps  # 128 per compute WG

        self.BN = max(self.BK, self.BV)
        self.mma_tiler = (self.BT, self.BN, self.BT)

        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mnk = (1, 1, 1)
        self.buffer_align_bytes = 1024

        self.bproc_stage = 1
        self.acc_pipe_stage = 1

        # Match the csrc 384-thread, one-CTA-per-SM register split.
        self.min_occupancy = 1
        self.num_regs_prologue = 224
        self.num_regs_epilogue = 200
        self.num_regs_others = 80
        self.a_stage = 2
        self.k_stage = 2
        self.g_stage = 2
        self.v_tma_stage = 2
        self.beta_stage = 2
        self.num_sm = utils.HardwareInfo().get_device_multiprocessor_count()

    @staticmethod
    def _plan_tmem(tiled_mma, mma_tiler, acc_stages):
        SM100_TMEM_CAPACITY_COLS = 512
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))
        num_acc = tcgen05.find_tmem_tensor_col_offset(tCtAcc_fake)
        total = 1
        while total < num_acc:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS
        return total

    @cute.jit
    def _tma_partition_A(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        coord = (None, 0, None)
        gX = cute.local_tile(tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx)))
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_A(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        coord = (0, None, None)
        gX = cute.local_tile(tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx)))
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
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
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
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
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
        )
        return bSG_sC, bSG_gC

    @cute.jit
    def _decode_persistent_work(self, work_idx, total_nt, H, T, BT, cu_seqlens, chunk_indices):
        chunk_global = work_idx // H
        i_h = work_idx - chunk_global * H
        if cutlass.const_expr(self.is_varlen):
            i_b = chunk_indices[chunk_global * 2]
            i_t = chunk_indices[chunk_global * 2 + 1]
            tok_offset = cu_seqlens[i_b]
            data_bidx = Int32(0)
            seq_end = cu_seqlens[i_b + 1]
            remaining = seq_end - (tok_offset + i_t * BT)
            remaining = cutlass.select_(remaining > BT, Int32(BT), remaining)
        else:
            NT = (T + BT - 1) // BT
            i_b = chunk_global // NT
            i_t = chunk_global - i_b * NT
            tok_offset = i_b * T
            data_bidx = i_b
            remaining = Int32(BT)
        return i_b, i_t, i_h, tok_offset, data_bidx, remaining

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
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
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
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # csrc reserves the full 512 TMEM columns and aliases W/U by dp-lane:
        # W starts at lane 0, U starts at lane 16.
        self.tmem_total = 512

        # Both MMA operands are resident in SMEM. A is the triangular Akk
        # matrix; B is the time-transposed preprocessed K/V tile.
        a_smem_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.io_dtype,
            self.a_stage,
        )
        b_smem_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.io_dtype,
            self.bproc_stage,
        )
        b_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BN),
            self.bproc_stage,
        )
        assert cute.cosize(b_smem_staged) == cute.cosize(b_epi_staged)

        # ---------- TMA load op ----------
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma.thr_id.shape,),
        )

        # ---------- SMEM layouts: k (bf16), v (bf16), gk (fp32) ----------
        k_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.k_stage,
        )
        v_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BV),
            self.v_tma_stage,
        )
        gk_epi_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.g_stage,
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
        tma_atom_A, tma_tensor_A = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            A_gmem,
            a_smem_one,
            self.mma_tiler,
            tiled_mma,
            cluster_layout.shape,
        )
        self.tma_A_bytes = cute.size_in_bytes(self.io_dtype, a_smem_one)

        k_epi_smem = cute.select(k_epi_staged, mode=[0, 1])
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            k_gmem,
            k_epi_smem,
            (self.BT, self.BK),
        )

        v_epi_smem = cute.select(v_epi_staged, mode=[0, 1])
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op,
            v_gmem,
            v_epi_smem,
            (self.BT, self.BV),
        )

        gk_epi_smem = cute.select(gk_epi_staged, mode=[0, 1])
        if cutlass.const_expr(self.preprocessed_k):
            # The preprocessed path does not consume gk. Reusing the k
            # descriptor also lets the public wrapper pass k as the unused
            # placeholder without constructing an fp32 tensor.
            tma_atom_gk, tma_tensor_gk = tma_atom_k, tma_tensor_k
        else:
            tma_atom_gk, tma_tensor_gk = cpasync.make_tiled_tma_atom(
                tma_load_op,
                gk_gmem,
                gk_epi_smem,
                (self.BT, self.BK),
            )

        # ---------- TMA byte counts ----------
        self.tma_bytes_k = cute.size_in_bytes(self.io_dtype, k_epi_smem)
        self.tma_bytes_v = cute.size_in_bytes(self.io_dtype, v_epi_smem)
        self.tma_bytes_gk = cute.size_in_bytes(self.acc_dtype, gk_epi_smem)
        # ---------- CUDA-core KG output staging ----------
        output_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BN),
            1,
        )

        # ---------- SharedStorage ----------
        @cute.struct
        class SharedStorage:
            load_A_mbar: cute.struct.MemRange[Int64, self.a_stage * 2]
            load_k_mbar: cute.struct.MemRange[Int64, self.k_stage * 2]
            load_g_mbar: cute.struct.MemRange[Int64, self.g_stage * 2]
            load_v_mbar: cute.struct.MemRange[Int64, self.v_tma_stage * 2]
            beta_mbar: cute.struct.MemRange[Int64, self.beta_stage * 2]
            prologue_ready_mbar: cute.struct.MemRange[Int64, self.bproc_stage * 2]
            acc_mbar: cute.struct.MemRange[Int64, self.acc_pipe_stage * 2]
            tmem_holding_buf: Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sBV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(b_smem_staged)],
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
            sOut: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(output_epi_staged)],
                self.buffer_align_bytes,
            ]
            sBeta: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, self.beta_stage * self.BT],
                128,
            ]

        self.shared_storage = SharedStorage

        # ---------- cu_seqlens / chunk_indices tensors ----------
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((B + 1,)))
        chunk_indices = cute.make_tensor(chunk_indices_ptr, cute.make_layout((total_nt * 2,)))

        # ---------- Grid ----------
        # csrc launches exactly one persistent CTA per SM.
        grid = (self.num_sm, 1, 1)

        self.kernel(
            tiled_mma,
            tma_atom_A,
            tma_tensor_A,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_gk,
            tma_tensor_gk,
            a_smem_staged,
            b_smem_staged,
            b_epi_staged,
            k_epi_staged,
            v_epi_staged,
            gk_epi_staged,
            output_epi_staged,
            beta_ptr,
            w_ptr,
            u_ptr,
            kg_ptr,
            cu_seqlens,
            chunk_indices,
            problem_size,
            total_nt,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
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
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        b_epi_staged: cute.ComposedLayout,
        k_epi_staged: cute.ComposedLayout,
        v_epi_staged: cute.ComposedLayout,
        gk_epi_staged: cute.ComposedLayout,
        output_epi_staged: cute.ComposedLayout,
        beta_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        u_ptr: cute.Pointer,
        kg_ptr: cute.Pointer,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        total_nt: Int32,
    ):
        B, T, H, K, V = problem_size
        BT = self.BT

        # Match csrc StaticPersistentTileScheduler: one CTA per SM,
        # tile_id = blockIdx.x + iteration * gridDim.x.
        block_idx_x = cute.arch.block_idx()[0]
        grid_dim_x = cute.arch.grid_dim()[0]
        total_work_units = total_nt * H
        num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            if cutlass.const_expr(not self.preprocessed_k):
                cpasync.prefetch_descriptor(tma_atom_gk)

        # ---------- SMEM ----------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sBK = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sBKTime = storage.sB.get_tensor(b_epi_staged.outer, swizzle=b_epi_staged.inner)
        sBV = storage.sBV.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sBVTime = storage.sBV.get_tensor(b_epi_staged.outer, swizzle=b_epi_staged.inner)
        sK = storage.sK.get_tensor(k_epi_staged.outer, swizzle=k_epi_staged.inner)
        sV = storage.sV.get_tensor(v_epi_staged.outer, swizzle=v_epi_staged.inner)
        sGK = storage.sGK.get_tensor(gk_epi_staged.outer, swizzle=gk_epi_staged.inner)
        sOut = storage.sOut.get_tensor(output_epi_staged.outer, swizzle=output_epi_staged.inner)
        sBeta = cute.make_tensor(
            cute.make_ptr(self.acc_dtype, storage.sBeta.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BT, self.beta_stage), stride=(1, self.BT)),
        )

        # ---------- Pipelines ----------
        load_A_P, load_A_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.a_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(1),
            tx_count=self.tma_A_bytes,
            barrier_storage=storage.load_A_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_k_P, load_k_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.k_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(self.num_cuda_warps),
            tx_count=self.tma_bytes_k,
            barrier_storage=storage.load_k_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_g_P, load_g_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.g_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(self.num_cuda_warps),
            tx_count=self.tma_bytes_gk,
            barrier_storage=storage.load_g_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_v_P, load_v_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.v_tma_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(self.num_cuda_warps),
            tx_count=self.tma_bytes_v,
            barrier_storage=storage.load_v_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        beta_P, beta_C = pipeline.PipelineAsync.create(
            num_stages=self.beta_stage,
            producer_group=_make_coop_group(2 * self.threads_per_warp),
            consumer_group=_make_coop_group(2 * self.num_cuda_threads),
            barrier_storage=storage.beta_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        prologue_ready_P, prologue_ready_C = pipeline.PipelineAsyncUmma.create(
            num_stages=self.bproc_stage,
            producer_group=_make_coop_group(2 * self.num_cuda_threads),
            consumer_group=_make_coop_group(1),
            barrier_storage=storage.prologue_ready_mbar.data_ptr(),
            defer_sync=True,
        ).make_participants()

        acc_done_P, acc_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_pipe_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(self.num_cuda_threads),
            barrier_storage=storage.acc_mbar.data_ptr(),
        ).make_participants()

        # ---------- TMEM ----------
        tmem_alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrBK = tiled_mma.make_fragment_B(sBK)
        tCrBV = tiled_mma.make_fragment_B(sBV)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tCtAccW = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tCtAccU = cute.make_tensor(tmem_ptr + 16 * 65536, tCtAcc_fake.layout)
        # Layout-only staged view used to derive the same 128-thread R2S
        # mapping as the MMA output. W/U stores do not read through this view.
        tCtAccMap_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
        tCtAccMap = cute.make_tensor(tmem_ptr, tCtAccMap_fake.layout)

        # =====================================================================
        # LOAD WARP
        # =====================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                i_b, i_t, i_h, tok_offset, data_bidx, remaining = self._decode_persistent_work(
                    work_idx, total_nt, H, T, BT, cu_seqlens, chunk_indices
                )
                # --- Domain offset (varlen) or alias (non-varlen) ---
                if cutlass.const_expr(self.is_varlen):
                    tma_k_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_k)
                    tma_v_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_v)
                    tma_gk_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_gk)
                    tma_A_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_A)
                else:
                    tma_k_v = tma_tensor_k
                    tma_v_v = tma_tensor_v
                    tma_gk_v = tma_tensor_gk
                    tma_A_v = tma_tensor_A

                # --- TMA partitions ---
                if cutlass.const_expr(self.is_varlen):
                    bSG_sK, bSG_gK = self._epilog_partition_varlen(
                        tma_atom_k,
                        tma_k_v[None, None, (i_h, data_bidx)],
                        (self.BT, self.BK),
                        sK,
                    )
                    bSG_sV, bSG_gV = self._epilog_partition_varlen(
                        tma_atom_v,
                        tma_v_v[None, None, (i_h, data_bidx)],
                        (self.BT, self.BV),
                        sV,
                    )
                    if cutlass.const_expr(not self.preprocessed_k):
                        bSG_sGK, bSG_gGK = self._epilog_partition_varlen(
                            tma_atom_gk,
                            tma_gk_v[None, None, (i_h, data_bidx)],
                            (self.BT, self.BK),
                            sGK,
                        )
                    tAsA, tAgA = self._tma_partition_A(
                        tma_atom_A,
                        tma_A_v,
                        sA,
                        self.mma_tiler,
                        tiled_mma,
                        data_bidx,
                        i_h,
                    )
                else:
                    bSG_sK, bSG_gK = self._data_tma_partition(
                        tma_atom_k,
                        tma_k_v,
                        (self.BT, self.BK),
                        sK,
                        i_h,
                        data_bidx,
                    )
                    bSG_sV, bSG_gV = self._data_tma_partition(
                        tma_atom_v,
                        tma_v_v,
                        (self.BT, self.BV),
                        sV,
                        i_h,
                        data_bidx,
                    )
                    if cutlass.const_expr(not self.preprocessed_k):
                        bSG_sGK, bSG_gGK = self._data_tma_partition(
                            tma_atom_gk,
                            tma_gk_v,
                            (self.BT, self.BK),
                            sGK,
                            i_h,
                            data_bidx,
                        )
                    tAsA, tAgA = self._tma_partition_A(
                        tma_atom_A,
                        tma_A_v,
                        sA,
                        self.mma_tiler,
                        tiled_mma,
                        data_bidx,
                        i_h,
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
                    k_h = load_k_P.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        bSG_gK[(None, i_t, i_kv)],
                        bSG_sK[None, k_h.index],
                        tma_bar_ptr=k_h.barrier,
                    )
                    if cutlass.const_expr(not self.preprocessed_k):
                        g_h = load_g_P.acquire_and_advance()
                        cute.copy(
                            tma_atom_gk,
                            bSG_gGK[(None, i_t, i_kv)],
                            bSG_sGK[None, g_h.index],
                            tma_bar_ptr=g_h.barrier,
                        )
                    v_h = load_v_P.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        bSG_gV[(None, i_t, i_kv)],
                        bSG_sV[None, v_h.index],
                        tma_bar_ptr=v_h.barrier,
                    )

        # =====================================================================
        # AUX WARPS — 64-thread beta producer, matching csrc LoadAux
        # =====================================================================
        elif warp_idx in self.aux_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)
            aux_tidx = tidx % (2 * self.threads_per_warp)
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                i_b, i_t, i_h, tok_offset, data_bidx, remaining = self._decode_persistent_work(
                    work_idx, total_nt, H, T, BT, cu_seqlens, chunk_indices
                )
                beta_h = beta_P.acquire_and_advance()
                beta_base = (tok_offset + i_t * BT) * H + i_h
                beta_gmem = cute.make_tensor(
                    cute.make_ptr(
                        self.beta_dtype,
                        (beta_ptr + beta_base).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=2,
                    ),
                    cute.make_layout((self.BT,), stride=(H,)),
                )
                sBeta[(aux_tidx, beta_h.index)] = cutlass.select_(
                    aux_tidx < remaining,
                    beta_gmem[aux_tidx].to(self.acc_dtype),
                    self.acc_dtype(0.0),
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                beta_h.commit()

        # =====================================================================
        # MMA WARP
        # =====================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            num_kblks = cute.size(tCrBK, mode=[2])

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                i_b, i_t, i_h, tok_offset, data_bidx, remaining = self._decode_persistent_work(
                    work_idx, total_nt, H, T, BT, cu_seqlens, chunk_indices
                )
                # Wait for A_mat — hold handle until all GEMMs finish reading sA
                a_h = load_A_C.wait_and_advance()

                for i_kv in cutlass.range(0, self.NK):
                    # Match csrc: dispatch W and U into two fixed TMEM regions and
                    # publish a single completion generation for the pair.
                    bp_h = prologue_ready_C.wait_and_advance()
                    acc_h = acc_done_P.acquire_and_advance()
                    for kblk in cutlass.range(num_kblks, unroll_full=True):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kblk != 0))
                        cute.gemm(
                            tiled_mma,
                            tCtAccW,
                            tCrA[(None, None, kblk, a_h.index)],
                            tCrBK[(None, None, kblk, bp_h.index)],
                            tCtAccW,
                        )

                    for kblk in cutlass.range(num_kblks, unroll_full=True):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kblk != 0))
                        cute.gemm(
                            tiled_mma,
                            tCtAccU,
                            tCrA[(None, None, kblk, a_h.index)],
                            tCrBV[(None, None, kblk, bp_h.index)],
                            tCtAccU,
                        )
                    acc_h.commit()
                    bp_h.release()

                # Release A after all GEMMs that read sA are dispatched
                a_h.release()

        # =====================================================================
        # K PROLOGUE + KG OUTPUT WARPGROUP
        # =====================================================================
        elif warp_idx in self.prologue_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_prologue)
            local_tidx = tidx % self.num_cuda_threads
            t2r_atom = cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), self.acc_dtype)
            tCtAcc_flat = tCtAccMap[((None, None), 0, 0, None)]
            fake_out = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem), cute.dice(self.mma_tiler, (1, 1, None))
            )
            tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
            thr_t2r = tiled_t2r.get_slice(local_tidx)
            tTR_sOut = thr_t2r.partition_D(fake_out)
            tTR_cM = thr_t2r.partition_D(cute.make_identity_tensor(cute.dice(self.mma_tiler, (1, 1, None))))
            r2s_atom = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r)
            tiled_r2s = cute.make_tiled_copy_D(r2s_atom, tiled_t2r)
            thr_r2s = tiled_r2s.get_slice(local_tidx)
            tRS_sBK = thr_r2s.partition_D(sBKTime)
            tRS_sOut = thr_r2s.partition_D(sOut[(None, None, 0)])
            r_bproc = cute.make_rmem_tensor(tTR_sOut.shape, self.io_dtype)
            r_kg = cute.make_rmem_tensor(tTR_sOut.shape, self.io_dtype)

            # csrc GmemTiledCopyO: 128 threads, 8 bf16 values per vector.
            async_copy_elems = 128 // self.io_dtype.width
            atom_ucopy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_dtype, num_bits_per_copy=128)
            n_threads_bn = self.BN // async_copy_elems
            n_threads_bt = self.num_cuda_threads // n_threads_bn
            r2g_tiled_copy = cute.make_tiled_copy_tv(
                atom_ucopy,
                cute.make_ordered_layout((n_threads_bt, n_threads_bn), order=(1, 0)),
                cute.make_layout((1, async_copy_elems)),
            )
            r2g_thr_copy = r2g_tiled_copy.get_slice(local_tidx)
            tOsOut = r2g_thr_copy.partition_S(sOut[(None, None, 0)])
            cOut = cute.make_identity_tensor((self.BT, self.BN))
            tOcOut = r2g_thr_copy.partition_S(cOut)
            r2g_frag = cute.make_fragment_like(tOsOut[None, 0, None], self.io_dtype)
            kg_sync = pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_cuda_threads)

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                i_b, i_t, i_h, tok_offset, data_bidx, remaining = self._decode_persistent_work(
                    work_idx, total_nt, H, T, BT, cu_seqlens, chunk_indices
                )
                beta_h = beta_C.wait_and_advance()
                token_base = tok_offset + i_t * BT
                kg_base = token_base * H * K + i_h * K
                gKg = cute.make_tensor(
                    cute.make_ptr(self.io_dtype, (kg_ptr + kg_base).toint(), cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout(
                        (self.BT, self.BK),
                        stride=(cute.assume(H * K, divby=async_copy_elems), 1),
                    ),
                )
                tOgKg = r2g_thr_copy.partition_D(gKg)

                for i_kv in cutlass.range(0, self.NK):
                    k_h = load_k_C.wait_and_advance()
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        m_coord, n_coord = tTR_cM[ei]
                        k_val = sK[(m_coord, n_coord, k_h.index)].to(self.acc_dtype)
                        beta_val = sBeta[(m_coord, beta_h.index)]
                        if cutlass.const_expr(self.preprocessed_k):
                            r_bproc[ei] = (k_val * beta_val).to(self.io_dtype)
                        else:
                            # csrc loads K into registers before waiting for G,
                            # overlapping the independent TMA pipelines.
                            r_bproc[ei] = k_val.to(self.io_dtype)

                    if cutlass.const_expr(not self.preprocessed_k):
                        g_h = load_g_C.wait_and_advance()
                        for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                            m_coord, n_coord = tTR_cM[ei]
                            k_val = r_bproc[ei].to(self.acc_dtype)
                            beta_val = sBeta[(m_coord, beta_h.index)]
                            g_val = sGK[(m_coord, n_coord, g_h.index)]
                            last_row = cutlass.select_(remaining < Int32(BT), remaining - 1, Int32(BT - 1))
                            gn_val = sGK[(last_row, n_coord, g_h.index)]
                            bp = (k_val * beta_val * cute.exp2(g_val, fastmath=self.use_fast_math)).to(self.io_dtype)
                            kg = (k_val * cute.exp2(gn_val - g_val, fastmath=self.use_fast_math)).to(self.io_dtype)
                            if cutlass.const_expr(self.is_varlen):
                                r_bproc[ei] = cutlass.select_(m_coord < remaining, bp, self.io_dtype(0.0))
                                r_kg[ei] = cutlass.select_(m_coord < remaining, kg, self.io_dtype(0.0))
                            else:
                                r_bproc[ei] = bp
                                r_kg[ei] = kg

                    bp_h = prologue_ready_P.acquire_and_advance()
                    r2s_b = tiled_r2s.retile(r_bproc)
                    cute.copy(tiled_r2s, r2s_b, tRS_sBK[(None, None, None, bp_h.index)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    bp_h.commit()
                    k_h.release()

                    if cutlass.const_expr(not self.preprocessed_k):
                        kg_sync.arrive_and_wait()
                        r2s_kg = tiled_r2s.retile(r_kg)
                        cute.copy(tiled_r2s, r2s_kg, tRS_sOut)
                        g_h.release()
                        kg_sync.arrive_and_wait()
                        for m1 in cutlass.range_constexpr(cute.size(tOsOut.shape[1])):
                            row = tOcOut[(0, 0), m1, 0][0]
                            if row < remaining:
                                cute.autovec_copy(tOsOut[None, m1, None], r2g_frag)
                                cute.autovec_copy(r2g_frag, tOgKg[None, m1, None])

                beta_h.release()

        # =====================================================================
        # V PROLOGUE + W/U EPILOGUE WARPGROUP
        # =====================================================================
        elif warp_idx in self.epilogue_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_epilogue)
            local_tidx = tidx % self.num_cuda_threads
            t2r_atom = cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), self.acc_dtype)
            tCtAcc_flat = tCtAccMap[((None, None), 0, 0, None)]
            fake_out = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem), cute.dice(self.mma_tiler, (1, 1, None))
            )
            tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tCtAcc_flat[(None, None, 0)])
            thr_t2r = tiled_t2r.get_slice(local_tidx)
            tTR_sOut = thr_t2r.partition_D(fake_out)
            tTR_cM = thr_t2r.partition_D(cute.make_identity_tensor(cute.dice(self.mma_tiler, (1, 1, None))))
            r2s_atom = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r)
            tiled_r2s = cute.make_tiled_copy_D(r2s_atom, tiled_t2r)
            thr_r2s = tiled_r2s.get_slice(local_tidx)
            tRS_sBV = thr_r2s.partition_D(sBVTime)
            r_bproc = cute.make_rmem_tensor(tTR_sOut.shape, self.io_dtype)

            # Exact csrc dp-lane mapping: alternating 16-lane groups store W/U,
            # and four warps cover the 64 output rows.
            acc_idx = (local_tidx // 16) & 1
            output_row = (local_tidx // 32) * 16 + (local_tidx % 16)
            quar_k = self.BN // 4

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                i_b, i_t, i_h, tok_offset, data_bidx, remaining = self._decode_persistent_work(
                    work_idx, total_nt, H, T, BT, cu_seqlens, chunk_indices
                )
                beta_h = beta_C.wait_and_advance()
                for i_kv in cutlass.range(0, self.NK):
                    v_h = load_v_C.wait_and_advance()
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        m_coord, n_coord = tTR_cM[ei]
                        value = (sV[(m_coord, n_coord, v_h.index)].to(self.acc_dtype) * sBeta[(m_coord, beta_h.index)]).to(
                            self.io_dtype
                        )
                        if cutlass.const_expr(self.is_varlen):
                            r_bproc[ei] = cutlass.select_(m_coord < remaining, value, self.io_dtype(0.0))
                        else:
                            r_bproc[ei] = value
                    bp_h = prologue_ready_P.acquire_and_advance()
                    r2s_b = tiled_r2s.retile(r_bproc)
                    cute.copy(tiled_r2s, r2s_b, tRS_sBV[(None, None, None, bp_h.index)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    bp_h.commit()
                    v_h.release()

                    wu_h = acc_done_C.wait_and_advance()
                    token_base = tok_offset + i_t * BT
                    output_addr = Int64(0)
                    if acc_idx == 0:
                        output_addr = (w_ptr + (token_base + output_row) * H * K + i_h * K + i_kv * self.BK).toint()
                    else:
                        output_addr = (u_ptr + (token_base + output_row) * H * V + i_h * V + i_kv * self.BV).toint()

                    tcgen05_fence_after()
                    for quar in cutlass.range_constexpr(4):
                        acc_i32 = tcgen05_ld_32x32b(
                            quar_k,
                            acc_idx * (16 * 65536) + wu_h.index * 256 + quar * quar_k,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        if output_row < remaining:
                            acc_f32 = reinterpret_cast(acc_i32, Int32, quar_k, self.acc_dtype)
                            acc_bf16 = TensorSSA(acc_f32, (quar_k,), self.acc_dtype).to(self.io_dtype)
                            out_i32 = reinterpret_cast(acc_bf16, self.io_dtype, quar_k, Int32)
                            for store_idx in cutlass.range_constexpr(quar_k // 16):
                                store_256b(
                                    output_addr + quar * quar_k * 2 + store_idx * 32,
                                    subvec(out_i32, store_idx * 8, 8),
                                )
                    tcgen05_fence_before()
                    wu_h.release()

                beta_h.release()

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
_uniform_cu_cache = {}


def _uniform_problem(cu_seqlens: torch.Tensor) -> tuple[int, int] | None:
    """Return ``(batch, sequence_length)`` for a uniform packed batch."""
    key = id(cu_seqlens)
    entry = _uniform_cu_cache.get(key)
    if entry is None or entry[0]() is not cu_seqlens or entry[1] != cu_seqlens._version:
        offsets = cu_seqlens.detach().cpu().tolist()
        lengths = [end - start for start, end in zip(offsets, offsets[1:])]
        result = (len(lengths), lengths[0]) if lengths and len(set(lengths)) == 1 else None
        _uniform_cu_cache[key] = (weakref.ref(cu_seqlens), cu_seqlens._version, result)
        return result
    return entry[2]


def _compile_recompute_wu(
    H,
    K,
    V,
    chunk_size=64,
    block_k=None,
    block_v=None,
    persistent=True,
    is_varlen=False,
    beta_dtype=cutlass.Float32,
    preprocessed_k=False,
):
    key = (
        H,
        K,
        V,
        chunk_size,
        block_k,
        block_v,
        persistent,
        is_varlen,
        beta_dtype,
        preprocessed_k,
        USE_FAST_MATH,
    )
    if key in _recompute_wu_cache:
        return _recompute_wu_cache[key]

    kernel_obj = KDARecomputeWU(
        K=K,
        V=V,
        chunk_size=chunk_size,
        block_k=block_k,
        block_v=block_v,
        beta_dtype=beta_dtype,
        is_varlen=is_varlen,
        preprocessed_k=preprocessed_k,
        use_fast_math=USE_FAST_MATH,
    )

    sym_a = cute.sym_int()
    sym_b = cute.sym_int()
    sym_cu = cute.sym_int()
    sym_ci = cute.sym_int()
    BT = chunk_size

    if is_varlen:
        k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, K), stride_order=(2, 1, 0), assumed_align=128)
        v_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, V), stride_order=(2, 1, 0), assumed_align=128)
        beta_fake = make_fake_compact_tensor(beta_dtype, (sym_a, H), stride_order=(1, 0), assumed_align=128)
        A_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, BT), stride_order=(2, 1, 0), assumed_align=128)
        gk_fake = make_fake_compact_tensor(
            cutlass.BFloat16 if preprocessed_k else cutlass.Float32,
            (sym_a, H, K),
            stride_order=(2, 1, 0),
            assumed_align=128,
        )
        w_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, K), stride_order=(2, 1, 0), assumed_align=128)
        u_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, V), stride_order=(2, 1, 0), assumed_align=128)
        kg_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, H, K), stride_order=(2, 1, 0), assumed_align=128)
    else:
        k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
        v_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
        beta_fake = make_fake_compact_tensor(beta_dtype, (sym_a, sym_b, H), stride_order=(2, 1, 0), assumed_align=128)
        A_fake = make_fake_compact_tensor(
            cutlass.BFloat16, (sym_a, sym_b, H, BT), stride_order=(3, 2, 1, 0), assumed_align=128
        )
        gk_fake = make_fake_compact_tensor(
            cutlass.BFloat16 if preprocessed_k else cutlass.Float32,
            (sym_a, sym_b, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        w_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
        u_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
        kg_fake = make_fake_compact_tensor(
            cutlass.BFloat16, (sym_a, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128
        )

    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=128)
    ci_fake = make_fake_compact_tensor(cutlass.Int32, (sym_ci,), assumed_align=128)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        k_fake,
        v_fake,
        beta_fake,
        A_fake,
        gk_fake,
        w_fake,
        u_fake,
        kg_fake,
        cu_fake,
        ci_fake,
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


def recompute_w_u_fwd(k, v, beta, A, gk, cu_seqlens=None, chunk_indices=None, block_k=None, block_v=None):
    is_varlen = cu_seqlens is not None
    packed_4d = is_varlen and k.dim() == 4
    restore_packed = False
    if packed_4d:
        if k.shape[0] != 1:
            raise ValueError("varlen inputs must be packed with batch dimension 1")
        k, v, beta, A, gk = (x.squeeze(0) for x in (k, v, beta, A, gk))

        uniform = _uniform_problem(cu_seqlens)
        if uniform is not None and uniform[1] % A.shape[-1] == 0:
            batch, seq_len = uniform
            k = k.view(batch, seq_len, *k.shape[1:])
            v = v.view(batch, seq_len, *v.shape[1:])
            beta = beta.view(batch, seq_len, *beta.shape[1:])
            A = A.view(batch, seq_len, *A.shape[1:])
            gk = gk.view(batch, seq_len, *gk.shape[1:])
            is_varlen = False
            packed_4d = False
            restore_packed = True

    if is_varlen:
        BT = A.shape[-1]
        T_total, H, K = k.shape
        V = v.shape[2]
        num_seqs = cu_seqlens.shape[0] - 1

        # Single-seq varlen with aligned T → dispatch as non-varlen for TMA S2G speed
        if num_seqs == 1 and T_total % BT == 0:
            k_4d = k.unsqueeze(0)
            v_4d = v.unsqueeze(0)
            beta_4d = beta.unsqueeze(0)
            A_4d = A.unsqueeze(0)
            gk_4d = gk.unsqueeze(0)
            w_4d, u_4d, _, kg_4d = recompute_w_u_fwd(
                k_4d,
                v_4d,
                beta_4d,
                A_4d,
                gk_4d,
                block_k=block_k,
                block_v=block_v,
            )
            return w_4d.squeeze(0), u_4d.squeeze(0), None, kg_4d.squeeze(0)

        if chunk_indices is not None:
            ci_s = chunk_indices.reshape(-1)
        else:
            ci_s = prepare_chunk_indices(cu_seqlens, BT).reshape(-1)

        total_nt = ci_s.shape[0] // 2
        ps = (Int32(num_seqs), Int32(T_total), Int32(H), Int32(K), Int32(V))
        cu_s = cu_seqlens
    else:
        B, T, H, K = k.shape
        V = v.shape[-1]
        BT = A.shape[-1]
        NT = (T + BT - 1) // BT
        total_nt = B * NT
        ps = (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V))
        global _dummy_cu_seqlens, _dummy_chunk_indices
        if _dummy_cu_seqlens is None or _dummy_cu_seqlens.device != k.device:
            _dummy_cu_seqlens = torch.zeros(2, dtype=torch.int32, device=k.device)
        if _dummy_chunk_indices is None or _dummy_chunk_indices.device != k.device:
            _dummy_chunk_indices = torch.zeros(2, dtype=torch.int32, device=k.device)
        cu_s = _dummy_cu_seqlens
        ci_s = _dummy_chunk_indices

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)

    compiled_fn = _compile_recompute_wu(
        H,
        K,
        V,
        chunk_size=BT,
        block_k=block_k,
        block_v=block_v,
        is_varlen=is_varlen,
        beta_dtype=cutlass.Float32 if beta.dtype == torch.float32 else cutlass.BFloat16,
    )

    compiled_fn(k, v, beta, A, gk, w, u, kg, cu_s, ci_s, ps, Int32(total_nt))

    if restore_packed:
        w, u, kg = (x.flatten(0, 1).unsqueeze(0) for x in (w, u, kg))
    elif packed_4d:
        w, u, kg = (x.unsqueeze(0) for x in (w, u, kg))
    return w, u, None, kg


def recompute_w_u_from_preprocessed(
    k_scaled,
    v,
    beta,
    A,
    cu_seqlens=None,
    chunk_indices=None,
    block_k=None,
    block_v=None,
):
    """Compute only ``w`` and ``u`` when fused intra already produced scaled k.

    ``k_scaled`` is ``k * exp2(gk)`` and ``A`` is the inverted intra-chunk
    matrix. The companion fused intra kernel already produced ``kg``, so this
    path avoids loading the fp32 cumulative gate and writing ``kg`` again.
    """
    is_varlen = cu_seqlens is not None
    packed_4d = is_varlen and k_scaled.dim() == 4
    restore_packed = False
    if packed_4d:
        if k_scaled.shape[0] != 1:
            raise ValueError("varlen inputs must be packed with batch dimension 1")
        k_scaled, v, beta, A = (x.squeeze(0) for x in (k_scaled, v, beta, A))

        uniform = _uniform_problem(cu_seqlens)
        if uniform is not None and uniform[1] % A.shape[-1] == 0:
            batch, seq_len = uniform
            k_scaled = k_scaled.view(batch, seq_len, *k_scaled.shape[1:])
            v = v.view(batch, seq_len, *v.shape[1:])
            beta = beta.view(batch, seq_len, *beta.shape[1:])
            A = A.view(batch, seq_len, *A.shape[1:])
            is_varlen = False
            packed_4d = False
            restore_packed = True

    if is_varlen:
        BT = A.shape[-1]
        T_total, H, K = k_scaled.shape
        V = v.shape[2]
        num_seqs = cu_seqlens.shape[0] - 1
        ci_s = chunk_indices.reshape(-1) if chunk_indices is not None else prepare_chunk_indices(cu_seqlens, BT).reshape(-1)
        total_nt = ci_s.shape[0] // 2
        ps = (Int32(num_seqs), Int32(T_total), Int32(H), Int32(K), Int32(V))
        cu_s = cu_seqlens
    else:
        B, T, H, K = k_scaled.shape
        V = v.shape[-1]
        BT = A.shape[-1]
        total_nt = B * ((T + BT - 1) // BT)
        ps = (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V))
        global _dummy_cu_seqlens, _dummy_chunk_indices
        if _dummy_cu_seqlens is None or _dummy_cu_seqlens.device != k_scaled.device:
            _dummy_cu_seqlens = torch.zeros(2, dtype=torch.int32, device=k_scaled.device)
        if _dummy_chunk_indices is None or _dummy_chunk_indices.device != k_scaled.device:
            _dummy_chunk_indices = torch.zeros(2, dtype=torch.int32, device=k_scaled.device)
        cu_s = _dummy_cu_seqlens
        ci_s = _dummy_chunk_indices

    w = torch.empty_like(k_scaled)
    u = torch.empty_like(v)
    compiled_fn = _compile_recompute_wu(
        H,
        K,
        V,
        chunk_size=BT,
        block_k=block_k,
        block_v=block_v,
        is_varlen=is_varlen,
        beta_dtype=cutlass.Float32 if beta.dtype == torch.float32 else cutlass.BFloat16,
        preprocessed_k=True,
    )

    # gk and kg are unused compile-signature placeholders for this
    # specialization. Reusing existing bf16 buffers avoids extra allocations.
    compiled_fn(k_scaled, v, beta, A, k_scaled, w, u, w, cu_s, ci_s, ps, Int32(total_nt))

    if restore_packed:
        w, u = (x.flatten(0, 1).unsqueeze(0) for x in (w, u))
    elif packed_4d:
        w, u = (x.unsqueeze(0) for x in (w, u))
    return w, u
