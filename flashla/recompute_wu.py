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
  Result = output^T[BN, BT] → transpose in epilogue writes.
  MMA tiler = (64, 64, 64).

Optimization: cooperative SMEM loading.
  All 128 CUDA threads cooperatively load k/v/gk tiles from GMEM→SMEM
  using coalesced 128-bit copies, then read from SMEM for element-wise compute.
  This eliminates scattered GMEM loads (stride H*K between rows) and reduces
  register spilling from long GMEM latency.

Warp assignment:
  0-3: CUDA core warps (cooperative load, element-wise compute, GMEM store)
  4:   MMA warp (tcgen05 GEMM)
  5:   Load warp (TMA for A_mat)
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
        io_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.K = K
        self.V = V
        self.BT = chunk_size
        self.BK = min(64, K)
        self.BV = min(64, V)
        self.NK = (K + self.BK - 1) // self.BK
        self.NV = (V + self.BV - 1) // self.BV
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype

        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        self.empty_warp_id = 7
        self.threads_per_cta = self.threads_per_warp * 8  # 256
        self.num_cuda_warps = len(self.cuda_warp_ids)

        self.num_regs_cuda = 208
        self.num_regs_others = 40
        self.min_occupancy = 2

        self.BN = max(self.BK, self.BV)  # 64
        self.mma_tiler = (self.BN, self.BT, self.BT)

        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mnk = (1, 1, 1)
        self.buffer_align_bytes = 1024

        self.bproc_stage = 1
        self.acc_stage = 1

        # Cooperative copy parameters
        self.num_cuda_threads = self.threads_per_warp * self.num_cuda_warps  # 128
        # For 128-bit vectorized copies:
        self.vec_bf16 = 8   # 8 bf16 = 128 bits
        self.vec_fp32 = 4   # 4 fp32 = 128 bits

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
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
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

        B, T, H, K, V = problem_size
        BT = self.BT

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

        # ---------- SMEM layouts for MMA B operand (A_mat) ----------
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        a_smem_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, 1,
        )

        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma.thr_id.shape,),
        )

        # ---------- GMEM tensor for A_mat ----------
        A_layout = cute.make_layout(
            (T, BT, (H, B)),
            stride=(H * BT, 1, (BT, T * H * BT)),
        )
        A_gmem = cute.make_tensor(A_ptr, A_layout)

        # ---------- TMA descriptor for A_mat ----------
        a_smem_one = cute.select(a_smem_staged, mode=[0, 1, 2])
        tma_atom_A, tma_tensor_A = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, A_gmem, a_smem_one, self.mma_tiler, tiled_mma,
            cluster_layout.shape,
        )
        self.tma_A_bytes = cute.size_in_bytes(self.io_dtype, a_smem_one)

        # ---------- Cooperative copy atoms for CUDA warps (128 threads) ----------
        copy_atom_bf16 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        copy_atom_fp32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.acc_dtype,
            num_bits_per_copy=128,
        )
        # Thread layout: 128 threads across (16 rows, 8 col_groups)
        # order=(1,0): col_groups vary fastest → threads 0-7 cover same row → coalesced
        coop_thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))
        coop_val_bf16 = cute.make_layout((1, self.vec_bf16))   # (1, 8) → 128 bits
        coop_val_fp32 = cute.make_layout((1, self.vec_fp32))   # (1, 4) → 128 bits

        coop_copy_bf16 = cute.make_tiled_copy_tv(
            copy_atom_bf16, coop_thr_layout, coop_val_bf16,
        )
        coop_copy_fp32 = cute.make_tiled_copy_tv(
            copy_atom_fp32, coop_thr_layout, coop_val_fp32,
        )

        # Padded SMEM strides to avoid bank conflicts during T2R-mapped reads:
        # bf16: stride(BN+2, 1) → bank = (row + col/2) % 32 → no conflict
        # fp32: stride(BK+1, 1) → bank = (row + col) % 32 → no conflict
        self.smem_bf16_stride = self.BN + 2   # 66
        self.smem_fp32_stride = self.BK + 1   # 65

        # ---------- SharedStorage ----------
        @cute.struct
        class SharedStorage:
            load_A_mbar: cute.struct.MemRange[Int64, 1 * 2]
            bproc_mbar: cute.struct.MemRange[Int64, self.bproc_stage * 2]
            acc_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]
            tmem_holding_buf: Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            # Cooperative loading buffers (padded for bank conflict avoidance)
            sDataBF16: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, self.BT * self.smem_bf16_stride],
                self.buffer_align_bytes,
            ]
            sDataFP32: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, self.BT * self.smem_fp32_stride],
                self.buffer_align_bytes,
            ]
            # Beta buffer (64 bf16 with 2-element padding for alignment)
            sBeta: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, self.BT + 2],
                128,
            ]

        self.shared_storage = SharedStorage
        NT = (T + BT - 1) // BT
        self.grid = (NT, H, B)

        self.kernel(
            tiled_mma,
            tma_atom_A, tma_tensor_A,
            tmem_a_layout,
            a_smem_staged,
            coop_copy_bf16,
            coop_copy_fp32,
            k_ptr, v_ptr, beta_ptr, gk_ptr,
            w_ptr, u_ptr, kg_ptr,
            problem_size,
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
        tiled_mma: cute.TiledMma,
        tma_atom_A: cute.CopyAtom,
        tma_tensor_A: cute.Tensor,
        tmem_a_layout: cute.ComposedLayout,
        a_smem_staged: cute.ComposedLayout,
        coop_copy_bf16: cute.TiledCopy,
        coop_copy_fp32: cute.TiledCopy,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        gk_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        u_ptr: cute.Pointer,
        kg_ptr: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_A)

        # ---------- SMEM ----------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)

        # Cooperative loading SMEM with padded strides (no swizzle)
        sDataBF16 = cute.make_tensor(
            cute.make_ptr(self.io_dtype, storage.sDataBF16.data_ptr().toint(),
                          cute.AddressSpace.smem),
            cute.make_layout((self.BT, self.BN),
                             stride=(self.smem_bf16_stride, 1)),
        )
        sDataFP32 = cute.make_tensor(
            cute.make_ptr(self.acc_dtype, storage.sDataFP32.data_ptr().toint(),
                          cute.AddressSpace.smem),
            cute.make_layout((self.BT, self.BK),
                             stride=(self.smem_fp32_stride, 1)),
        )
        sBeta = cute.make_tensor(
            cute.make_ptr(self.io_dtype, storage.sBeta.data_ptr().toint(),
                          cute.AddressSpace.smem),
            cute.make_layout((self.BT,), stride=(1,)),
        )

        # ---------- Pipelines ----------
        load_A_P, load_A_C = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(self.num_cuda_warps + 1),
            tx_count=self.tma_A_bytes,
            barrier_storage=storage.load_A_mbar.data_ptr(),
        ).make_participants()

        bproc_P, bproc_C = pipeline.PipelineAsyncUmma.create(
            num_stages=self.bproc_stage,
            producer_group=_make_coop_group(
                self.threads_per_warp * self.num_cuda_warps),
            consumer_group=_make_coop_group(1),
            barrier_storage=storage.bproc_mbar.data_ptr(),
        ).make_participants()

        acc_done_P, acc_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=_make_coop_group(1),
            consumer_group=_make_coop_group(
                self.threads_per_warp * self.num_cuda_warps),
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

        # ---------- Block indices ----------
        B, T, H, K, V = problem_size
        BT = self.BT
        BK = self.BK
        BV = self.BV
        NK = self.NK
        NV = self.NV

        i_t, i_h, i_b = cute.arch.block_idx()
        bos = i_b * T

        # ---------- TMA partition for A_mat ----------
        tAsA, tAgA = self._tma_partition_B(
            tma_atom_A, tma_tensor_A, sA,
            self.mma_tiler, tiled_mma, i_b, i_h,
        )

        # =================================================================
        # LOAD WARP
        # =================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)
            h_a = load_A_P.acquire_and_advance()
            cute.copy(
                tma_atom_A,
                tAgA[(None, i_t, 0)],
                tAsA[(None, 0)],
                tma_bar_ptr=h_a.barrier,
            )

        # =================================================================
        # STORE WARP — idle
        # =================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

        # =================================================================
        # EMPTY WARP — idle
        # =================================================================
        elif warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

        # =================================================================
        # MMA WARP
        # =================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            a_h = load_A_C.wait_and_advance()
            a_h.release()

            num_kblks = cute.size(tCrB, mode=[2])
            num_tiles = Int32(NK + NV)

            for _tile in cutlass.range(0, num_tiles, unroll=0):
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

        # =================================================================
        # CUDA CORE WARPS (last branch — heaviest)
        # =================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            local_tidx = tidx % (self.threads_per_warp * self.num_cuda_warps)

            # ---- T2R: TMEM acc → registers ----
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

            # ---- R2T: registers → TMEM A-operand ----
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

            # ---- Coordinate mapping ----
            out_tile = cute.dice(self.mma_tiler, (1, 1, None))
            cM_id = cute.make_identity_tensor(out_tile)
            tTR_cM = thr_t2r.partition_D(cM_id)

            # ---- Rmem tensors (hoisted) ----
            tTR_rAcc = cute.make_rmem_tensor(tTR_sOut.shape, self.acc_dtype)
            tTR_rBproc = cute.make_rmem_tensor(tTR_sOut.shape, self.io_dtype)
            tRT_rBproc = cute.make_rmem_tensor(r2t_src_shape, self.io_dtype)

            # ---- Cooperative copy thread views ----
            thr_bf16 = coop_copy_bf16.get_slice(local_tidx)
            thr_fp32 = coop_copy_fp32.get_slice(local_tidx)

            # SMEM destination partitions (constant across tiles)
            tCsDataBF16 = thr_bf16.partition_D(sDataBF16)
            tCsDataFP32 = thr_fp32.partition_D(sDataFP32)

            # ---- NamedBarrier for CUDA warp sync ----
            cuda_sync = pipeline.NamedBarrier(
                barrier_id=2, num_threads=self.num_cuda_threads)

            # Wait for A_mat
            a_h = load_A_C.wait_and_advance()
            a_h.release()

            # GMEM strides
            stride_k = cute.assume(H * K, divby=1)
            stride_v = cute.assume(H * V, divby=1)
            stride_beta = cute.assume(H, divby=1)

            time_base = (bos + i_t * BT) * H + i_h
            last_t = cutlass.min(i_t * BT + BT, T) - Int32(1)

            num_k_tiles = Int32(NK)
            num_v_tiles = Int32(NV)

            # -- Load beta into SMEM once (shared across all K/V tiles) --
            # (sBeta tensor created before warp dispatch to avoid DSLTreeFlattenError)
            # Cooperative load: first BT threads each load 1 beta element
            beta_gmem_p = cute.make_ptr(self.io_dtype, (beta_ptr + time_base).toint(),
                                        cute.AddressSpace.gmem, assumed_align=2)
            beta_gmem = cute.make_tensor(beta_gmem_p,
                cute.make_layout((self.BT,), stride=(stride_beta,)))
            # Use threads 0..BT-1 to load beta; others get redundant copy
            beta_load_idx = local_tidx % self.BT  # constexpr divisor
            sBeta[beta_load_idx] = beta_gmem[beta_load_idx]
            cuda_sync.arrive_and_wait()

            # ==============================================================
            # K-tile loop: compute w and kg
            # ==============================================================
            for i_k in cutlass.range(0, num_k_tiles, unroll=0):
                k_off = time_base * K + i_k * BK

                # -- GMEM tensors for cooperative load --
                k_tile_p = cute.make_ptr(self.io_dtype, (k_ptr + k_off).toint(),
                                         cute.AddressSpace.gmem, assumed_align=2)
                gmem_k = cute.make_tensor(k_tile_p,
                    cute.make_layout((self.BT, self.BK), stride=(stride_k, 1)))

                gk_tile_p = cute.make_ptr(self.acc_dtype, (gk_ptr + k_off).toint(),
                                          cute.AddressSpace.gmem, assumed_align=4)
                gmem_gk = cute.make_tensor(gk_tile_p,
                    cute.make_layout((self.BT, self.BK), stride=(stride_k, 1)))

                # -- Cooperative load: GMEM → SMEM (coalesced) --
                tCgK = thr_bf16.partition_S(gmem_k)
                cute.autovec_copy(tCgK, tCsDataBF16)

                tCgGK = thr_fp32.partition_S(gmem_gk)
                cute.autovec_copy(tCgGK, tCsDataFP32)

                # Sync: ensure all SMEM writes complete before reads
                cuda_sync.arrive_and_wait()

                # -- Compute B_proc from SMEM (k, gk, beta all from SMEM) --
                for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                    m_coord, k_coord = tTR_cM[ei]
                    k_val = sDataBF16[(k_coord, m_coord)].to(self.acc_dtype)
                    gk_val = sDataFP32[(k_coord, m_coord)]
                    beta_val = sBeta[k_coord].to(self.acc_dtype)
                    tTR_rBproc[ei] = (k_val * beta_val * cute.exp2(gk_val)).to(self.io_dtype)

                # -- R2T: B_proc → TMEM --
                tRT_rBproc.store(tTR_rBproc.load())
                bproc_h = bproc_P.acquire_and_advance()
                cute.copy(tiled_r2t, tRT_rBproc, tRT_tA[(None, None, None, None, 0)])
                cute.arch.fence_view_async_tmem_store()
                bproc_h.commit()

                # -- Compute kg from SMEM (overlaps with MMA) --
                gn_off = (bos + last_t) * H * K + i_h * K + i_k * BK
                gn_p = cute.make_ptr(self.acc_dtype, (gk_ptr + gn_off).toint(),
                                     cute.AddressSpace.gmem, assumed_align=4)
                gn_row = cute.make_tensor(gn_p,
                    cute.make_layout((self.BK,), stride=(1,)))

                kg_tile_p = cute.make_ptr(self.io_dtype, (kg_ptr + k_off).toint(),
                                          cute.AddressSpace.gmem, assumed_align=2)
                kg_tile = cute.make_tensor(kg_tile_p,
                    cute.make_layout((self.BT, self.BK), stride=(stride_k, 1)))

                for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                    m_coord, k_coord = tTR_cM[ei]
                    k_val2 = sDataBF16[(k_coord, m_coord)].to(self.acc_dtype)
                    gk_val2 = sDataFP32[(k_coord, m_coord)]
                    gn_val = gn_row[m_coord]
                    kg_tile[(k_coord, m_coord)] = (k_val2 * cute.exp2(gn_val - gk_val2)).to(self.io_dtype)

                # Sync: ensure all SMEM reads done before next cooperative load
                cuda_sync.arrive_and_wait()

                # -- Wait for MMA, T2R → w --
                acc_h = acc_done_C.wait_and_advance()
                cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, acc_h.index)], tTR_rAcc)
                cute.arch.fence_view_async_tmem_load()
                acc_h.release()

                # -- Write w to GMEM --
                w_tile_p = cute.make_ptr(self.io_dtype, (w_ptr + k_off).toint(),
                                         cute.AddressSpace.gmem, assumed_align=2)
                w_tile = cute.make_tensor(w_tile_p,
                    cute.make_layout((self.BT, self.BK), stride=(stride_k, 1)))
                for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                    m_coord, n_coord = tTR_cM[ei]
                    w_tile[(n_coord, m_coord)] = tTR_rAcc[ei].to(self.io_dtype)

            # ==============================================================
            # V-tile loop: compute u
            # ==============================================================
            for i_v in cutlass.range(0, num_v_tiles, unroll=0):
                v_off = time_base * V + i_v * BV

                # -- Cooperative load v → SMEM --
                v_tile_p = cute.make_ptr(self.io_dtype, (v_ptr + v_off).toint(),
                                         cute.AddressSpace.gmem, assumed_align=2)
                gmem_v = cute.make_tensor(v_tile_p,
                    cute.make_layout((self.BT, self.BV), stride=(stride_v, 1)))

                tCgV = thr_bf16.partition_S(gmem_v)
                cute.autovec_copy(tCgV, tCsDataBF16)

                cuda_sync.arrive_and_wait()

                # -- Compute B_proc = v * beta from SMEM --
                for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                    m_coord, k_coord = tTR_cM[ei]
                    v_val = sDataBF16[(k_coord, m_coord)].to(self.acc_dtype)
                    beta_val = sBeta[k_coord].to(self.acc_dtype)
                    tTR_rBproc[ei] = (v_val * beta_val).to(self.io_dtype)

                tRT_rBproc.store(tTR_rBproc.load())
                bproc_h2 = bproc_P.acquire_and_advance()
                cute.copy(tiled_r2t, tRT_rBproc, tRT_tA[(None, None, None, None, 0)])
                cute.arch.fence_view_async_tmem_store()
                bproc_h2.commit()

                # Sync: ensure all SMEM reads done
                cuda_sync.arrive_and_wait()

                # -- Wait for MMA, T2R → u --
                acc_h2 = acc_done_C.wait_and_advance()
                cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, acc_h2.index)], tTR_rAcc)
                cute.arch.fence_view_async_tmem_load()
                acc_h2.release()

                u_tile_p = cute.make_ptr(self.io_dtype, (u_ptr + v_off).toint(),
                                         cute.AddressSpace.gmem, assumed_align=2)
                u_tile = cute.make_tensor(u_tile_p,
                    cute.make_layout((self.BT, self.BV), stride=(stride_v, 1)))
                for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                    m_coord, n_coord = tTR_cM[ei]
                    u_tile[(n_coord, m_coord)] = tTR_rAcc[ei].to(self.io_dtype)

        # ---------- TMEM cleanup ----------
        tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)


# ============================================================================
# Compile cache
# ============================================================================

_recompute_wu_cache = {}


def _compile_recompute_wu(H, K, V, chunk_size=64):
    key = (H, K, V, chunk_size)
    if key in _recompute_wu_cache:
        return _recompute_wu_cache[key]

    kernel_obj = KDARecomputeWU(K=K, V=V, chunk_size=chunk_size)

    sym_a = cute.sym_int()
    sym_b = cute.sym_int()

    k_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_a, sym_b, H, K),
        stride_order=(3, 2, 1, 0), assumed_align=128,
    )
    v_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_a, sym_b, H, V),
        stride_order=(3, 2, 1, 0), assumed_align=128,
    )
    beta_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_a, sym_b, H),
        stride_order=(2, 1, 0), assumed_align=128,
    )
    A_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_a, sym_b, H, chunk_size),
        stride_order=(3, 2, 1, 0), assumed_align=128,
    )
    gk_fake = make_fake_compact_tensor(
        cutlass.Float32, (sym_a, sym_b, H, K),
        stride_order=(3, 2, 1, 0), assumed_align=128,
    )
    w_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_a, sym_b, H, K),
        stride_order=(3, 2, 1, 0), assumed_align=128,
    )
    u_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_a, sym_b, H, V),
        stride_order=(3, 2, 1, 0), assumed_align=128,
    )
    kg_fake = make_fake_compact_tensor(
        cutlass.BFloat16, (sym_a, sym_b, H, K),
        stride_order=(3, 2, 1, 0), assumed_align=128,
    )
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        k_fake, v_fake, beta_fake, A_fake, gk_fake,
        w_fake, u_fake, kg_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    _recompute_wu_cache[key] = compiled_fn
    return compiled_fn


# ============================================================================
# Public API
# ============================================================================

def recompute_w_u_fwd(k, v, beta, A, gk, cu_seqlens=None, chunk_indices=None):
    assert cu_seqlens is None, "varlen mode not yet implemented"
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)

    compiled_fn = _compile_recompute_wu(H, K, V, chunk_size=BT)
    compiled_fn(k, v, beta, A, gk, w, u, kg,
                (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V)))
    return w, u, None, kg


# ============================================================================
# Reference
# ============================================================================

def recompute_w_u_fwd_ref(k, v, beta, A, gk):
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    NT = (T + BT - 1) // BT

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)

    for b in range(B):
        for h in range(H):
            for it in range(NT):
                t0 = it * BT
                t1 = min(t0 + BT, T)
                tlen = t1 - t0
                b_A = A[b, t0:t1, h, :tlen].float()
                b_beta = beta[b, t0:t1, h].float()
                b_k = k[b, t0:t1, h, :].float()
                b_gk = gk[b, t0:t1, h, :].float()
                b_v = v[b, t0:t1, h, :].float()

                b_kb = b_k * b_beta[:, None] * (2.0 ** b_gk)
                w[b, t0:t1, h, :] = (b_A @ b_kb).to(k.dtype)

                b_vb = b_v * b_beta[:, None]
                u[b, t0:t1, h, :] = (b_A @ b_vb).to(v.dtype)

                b_gn = gk[b, t1 - 1, h, :].float()
                kg[b, t0:t1, h, :] = (b_k * (2.0 ** (b_gn[None, :] - b_gk))).to(k.dtype)

    return w, u, None, kg


# ============================================================================
# Test & Benchmark
# ============================================================================

def main():
    B, T, H, K, V = 1, 128, 1, 64, 64
    BT = 64
    NT = T // BT

    print(f"Test: B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}")

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16))
    gk_raw = -torch.abs(torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)) * 0.1
    gk = gk_raw.cumsum(dim=1)
    A_full = torch.randn(B, NT, H, BT, BT, device="cuda", dtype=torch.bfloat16) * 0.1
    A_full = torch.tril(A_full)
    A = A_full.reshape(B, T, H, BT)

    w_ref, u_ref, _, kg_ref = recompute_w_u_fwd_ref(k, v, beta, A, gk)
    torch.cuda.synchronize()

    w, u, _, kg = recompute_w_u_fwd(k, v, beta, A, gk)
    torch.cuda.synchronize()

    dw = (w.float() - w_ref.float()).abs().max().item()
    du = (u.float() - u_ref.float()).abs().max().item()
    dkg = (kg.float() - kg_ref.float()).abs().max().item()
    print(f"  w  max diff: {dw:.6f}")
    print(f"  u  max diff: {du:.6f}")
    print(f"  kg max diff: {dkg:.6f}")

    ok = dw < 1.0 and du < 1.0 and dkg < 1.0
    print(f"  {'PASS' if ok else 'FAIL'}")

    if ok:
        print("\nBenchmark:")
        B2, T2, H2, K2, V2 = 4, 4096, 64, 128, 128
        BT2 = 64
        NT2 = T2 // BT2
        torch.manual_seed(999)
        k2 = torch.randn(B2, T2, H2, K2, device="cuda", dtype=torch.bfloat16) * 0.1
        v2 = torch.randn(B2, T2, H2, V2, device="cuda", dtype=torch.bfloat16) * 0.1
        beta2 = torch.sigmoid(torch.randn(B2, T2, H2, device="cuda", dtype=torch.bfloat16))
        gk2 = (-torch.abs(torch.randn(B2, T2, H2, K2, device="cuda", dtype=torch.float32)) * 0.1).cumsum(dim=1)
        A2 = torch.tril(torch.randn(B2, NT2, H2, BT2, BT2, device="cuda", dtype=torch.bfloat16) * 0.1).reshape(B2, T2, H2, BT2)

        for _ in range(3):
            recompute_w_u_fwd(k2, v2, beta2, A2, gk2)
        torch.cuda.synchronize()

        n_iter = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            recompute_w_u_fwd(k2, v2, beta2, A2, gk2)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_iter
        print(f"  CuTeDSL: {ms:.3f} ms")

        try:
            from fla.ops.kda.wy_fast import recompute_w_u_fwd as fla_recompute
            for _ in range(3):
                fla_recompute(k2, v2, beta2, A2, gk=gk2)
            torch.cuda.synchronize()
            start.record()
            for _ in range(n_iter):
                fla_recompute(k2, v2, beta2, A2, gk=gk2)
            end.record()
            torch.cuda.synchronize()
            fla_ms = start.elapsed_time(end) / n_iter
            print(f"  FLA Triton: {fla_ms:.3f} ms")
            print(f"  Speedup: {fla_ms / ms:.2f}x")
        except Exception as e:
            print(f"  FLA not available: {e}")


if __name__ == "__main__":
    main()
