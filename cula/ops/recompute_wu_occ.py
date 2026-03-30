# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CuTeDSL kernel for KDA recompute_w_u_fwd — high-occupancy variant.

Uses Ampere-style register MMA (m16n8k16) WITHOUT TMEM or warp
specialization. All 128 threads (4 warps) cooperate on data loads,
elementwise preprocessing, MMA, and GMEM stores.

Tile sizes: BT=64, BK=BV=64, NK=NV=2.
SMEM budget ~16.6 KB per CTA.

Computes (same semantics as recompute_wu.py):
  w  = A @ (k * beta * exp2(gk))
  u  = A @ (v * beta)
  kg = k * exp2(gn - gk)

Key optimizations:
  - JIT A fragment loading in MMA loop (reduces peak register pressure)
  - 4-wide elementwise thread mapping with precomputed gn values
  - min_blocks_per_mp=7 to balance register budget vs occupancy
  - Full unroll of elementwise outer loop (8 iterations)

MMA layout (m16n8k16):
  C[BT, BK] = A[BT, BT] @ B_proc[BT, BK]
  K inner dimension = BT = 64 → 4 k-tiles of 16.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Int32


class KDARecomputeWUOcc:
    """Occupancy-8 cooperative kernel for recompute_wu."""

    def __init__(
        self,
        K: int = 128,
        V: int = 128,
        chunk_size: int = 64,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        assert K == 128 and V == 128, f"K,V must be 128, got K={K}, V={V}"
        self.K = K
        self.V = V
        self.BT = chunk_size
        self.BK = 64
        self.BV = 64
        self.NK = K // self.BK  # 2
        self.NV = V // self.BV  # 2
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype

        self.num_threads = 128
        self.num_warps = 4
        self.min_occupancy = 6

        # MMA: m16n8k16 bf16
        self.mma_inst = (16, 8, 16)

        # Swizzle atom dimensions for SMEM
        self.smem_k_block = 64  # columns per swizzle atom
        self.swizzle_bits = 3  # log2(8) → 8 rows per atom
        self.copy_bits = 128  # 128-bit (16B) copies for cp.async / ldmatrix

        self.buffer_align = 128

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
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
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

        # ---- SMEM layouts with swizzle ----
        # sA: A_mat (MMA-A operand), shape (BT, BT) = (64, 64)
        sA_atom = cute.make_composed_layout(
            cute.make_swizzle(self.swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, self.smem_k_block), stride=(self.smem_k_block, 1)),
        )
        sA_layout = cute.tile_to_shape(sA_atom, (self.BT, self.BT), (0, 1))

        # sB: B_proc (element-wise preprocessed), shape (BT, BK) = (64, 64)
        sB_atom = cute.make_composed_layout(
            cute.make_swizzle(self.swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, self.smem_k_block), stride=(self.smem_k_block, 1)),
        )
        sB_layout = cute.tile_to_shape(sB_atom, (self.BT, self.BK), (0, 1))

        # ---- Tiled MMA ----
        mma_op = warp.MmaF16BF16Op(self.io_dtype, self.acc_dtype, self.mma_inst)
        atom_layout_mnk = (self.num_warps, 1, 1)
        permutation_mnk = (
            atom_layout_mnk[0] * self.mma_inst[0],
            atom_layout_mnk[1] * self.mma_inst[1] * 2,
            atom_layout_mnk[2] * self.mma_inst[2],
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        # ---- g2s copy atom for A (regular cooperative loads, not cp.async) ----
        async_copy_elems = self.copy_bits // self.io_dtype.width  # 8
        atom_copy_A = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_dtype,
        )
        thr_dim1 = sA_atom.outer.shape[1] // async_copy_elems  # 64/8 = 8
        thr_layout_A = cute.make_layout(
            (self.num_threads // thr_dim1, thr_dim1),
            stride=(thr_dim1, 1),
        )
        val_layout_A = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_A = cute.make_tiled_copy_tv(atom_copy_A, thr_layout_A, val_layout_A)

        # ---- SMEM → register copy atoms for MMA operands ----
        smem_copy_atom_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.io_dtype,
        )
        smem_copy_atom_Bt = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.io_dtype,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_Bt = cute.make_tiled_copy_B(smem_copy_atom_Bt, tiled_mma)

        # ---- epilogue store atom (direct register → GMEM) ----
        # (no smem_tiled_copy_C or gmem_tiled_copy_C needed)

        # ---- SharedStorage ----
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(sA_layout)],
                self.buffer_align,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(sB_layout)],
                self.buffer_align,
            ]
            sBeta: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, self.BT + 2],
                self.buffer_align,
            ]

        # ---- GMEM tensor for A (used with cp.async) ----
        A_layout = cute.make_layout(
            (T, self.BT, (H, B)),
            stride=(H * self.BT, 1, (self.BT, T * H * self.BT)),
        )
        A_gmem = cute.make_tensor(A_ptr, A_layout)

        # Grid
        NT = (T + BT - 1) // BT
        grid = (NT, H, B)

        self.kernel(
            tiled_mma,
            gmem_tiled_copy_A,
            smem_tiled_copy_A,
            smem_tiled_copy_Bt,
            sA_layout,
            sB_layout,
            SharedStorage,
            A_gmem,
            k_ptr,
            v_ptr,
            beta_ptr,
            gk_ptr,
            w_ptr,
            u_ptr,
            kg_ptr,
            problem_size,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            min_blocks_per_mp=self.min_occupancy,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        gmem_tiled_copy_A: cute.TiledCopy,
        smem_tiled_copy_A: cute.TiledCopy,
        smem_tiled_copy_Bt: cute.TiledCopy,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
        A_gmem: cute.Tensor,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        gk_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        u_ptr: cute.Pointer,
        kg_ptr: cute.Pointer,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
    ):
        B, T, H, K, V = problem_size
        BT = self.BT
        BK = self.BK
        BV = self.BV

        i_t, i_h, i_b = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        tok_offset = i_b * T

        # ---- SMEM alloc ----
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)
        sBeta = cute.make_tensor(
            cute.make_ptr(self.io_dtype, storage.sBeta.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BT,), stride=(1,)),
        )

        # Transposed view of sB for MMA-B operand: (BK, BT)
        sBt = cute.composition(
            sB,
            cute.make_layout(
                (self.BK, self.BT),
                stride=(self.BT, 1),
            ),
        )

        # ---- Setup CTA sync ----
        cta_sync = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_threads)

        # =====================================================================
        # 1. Load A_mat → sA via cp.async
        # =====================================================================
        # gA for this chunk: (BT, BT) tile from A_gmem
        gA = cute.local_tile(
            A_gmem[None, None, (i_h, i_b)],
            (self.BT, self.BT),
            (i_t, 0),
        )
        # Ensure 16-byte alignment for cp.async 128-bit copies
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)

        thr_copy_A = gmem_tiled_copy_A.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        # Load A tile using cooperative element-wise copies
        tArA_tmp = cute.make_fragment_like(tAsA, self.io_dtype)
        cute.copy(gmem_tiled_copy_A, tAgA, tArA_tmp)
        cute.copy(gmem_tiled_copy_A, tArA_tmp, tAsA)

        # =====================================================================
        # 2. Load beta → sBeta
        # =====================================================================
        stride_beta = cute.assume(H, divby=1)
        beta_time_base = (tok_offset + i_t * BT) * H + i_h
        beta_gmem_p = cute.make_ptr(
            self.io_dtype,
            (beta_ptr + beta_time_base).toint(),
            cute.AddressSpace.gmem,
            assumed_align=2,
        )
        beta_gmem = cute.make_tensor(
            beta_gmem_p,
            cute.make_layout((self.BT,), stride=(stride_beta,)),
        )
        if tidx < self.BT:
            sBeta[tidx] = beta_gmem[tidx]

        # Wait for A and beta
        cta_sync.arrive_and_wait()

        # =====================================================================
        # 3. Setup MMA partitions (A JIT-loaded per MMA iter, B per-pass)
        # =====================================================================
        thr_mma = tiled_mma.get_slice(tidx)

        # --- A from sA (JIT loaded inside MMA loop to reduce register pressure) ---
        tCsA_mma = thr_mma.partition_A(sA)
        tCrA = tiled_mma.make_fragment_A(tCsA_mma)

        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        tSsA = smem_thr_copy_A.partition_S(sA)
        tSrA_copy_view = smem_thr_copy_A.retile(tCrA)

        # NOTE: A fragments loaded JIT inside MMA loop (not persistent)
        # This frees ~16 registers for elementwise, avoiding local memory spills.

        # --- B from sBt (transposed view) ---
        tCsBt_mma = thr_mma.partition_B(sBt)
        tCrBt = tiled_mma.make_fragment_B(tCsBt_mma)

        smem_thr_copy_Bt = smem_tiled_copy_Bt.get_slice(tidx)
        tSsBt = smem_thr_copy_Bt.partition_S(sBt)
        tSrBt_copy_view = smem_thr_copy_Bt.retile(tCrBt)

        # --- Accumulator shape ---
        acc_shape = thr_mma.partition_shape_C((self.BT, self.BK))

        # --- Direct store atom for register → SMEM epilogue ---
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_dtype)

        # --- sB partition for MMA output → SMEM staging (reuse sB after MMA consumes it) ---
        tCsB = thr_mma.partition_C(sB)

        # sB base SMEM pointer (byte addr) for manual swizzled offset computation
        sB_base_int = storage.sB.data_ptr().toint()

        # =====================================================================
        # 4. K passes (NK = K/BK iterations)
        # =====================================================================
        H_K = cute.assume(H * K, divby=128)

        for i_k in cutlass.range(0, self.NK):
            k_col_off = i_k * BK
            time_base = (tok_offset + i_t * BT) * H + i_h
            k_base_off = time_base * K + k_col_off

            # ------- 4a. Elementwise: B_proc + kg (vectorized 1D row loads) -------
            ELEMS_K = 8
            threads_per_row_k = BK // ELEMS_K  # 8
            thread_col_base_k = (tidx % threads_per_row_k) * ELEMS_K
            thread_row_base_k = tidx // threads_per_row_k  # 0..15
            rows_per_iter_k = self.num_threads // threads_per_row_k  # 16

            # Vectorized gn load (8 fp32 from row BT-1)
            gn_row_off = k_base_off + (BT - 1) * H_K + thread_col_base_k
            r_gn = cute.make_rmem_tensor((ELEMS_K,), self.acc_dtype)
            g_gn_row = cute.make_tensor(
                cute.make_ptr(self.acc_dtype, (gk_ptr + gn_row_off).toint(), cute.AddressSpace.gmem, assumed_align=32),
                cute.make_layout((ELEMS_K,)),
            )
            cute.autovec_copy(g_gn_row, r_gn)

            # 1D row layouts
            row_layout_k = cute.make_layout((ELEMS_K,))
            r_row_k = cute.make_rmem_tensor((ELEMS_K,), self.io_dtype)
            r_row_gk = cute.make_rmem_tensor((ELEMS_K,), self.acc_dtype)
            r_row_kg = cute.make_rmem_tensor((ELEMS_K,), self.io_dtype)
            r_row_sB = cute.make_rmem_tensor((ELEMS_K,), self.io_dtype)

            for row_iter in cutlass.range(0, self.BT // rows_per_iter_k, unroll=4):
                row = thread_row_base_k + row_iter * rows_per_iter_k
                row_off = k_base_off + row * H_K + thread_col_base_k

                # Vectorized GMEM loads
                g_k_row = cute.make_tensor(
                    cute.make_ptr(self.io_dtype, (k_ptr + row_off).toint(), cute.AddressSpace.gmem, assumed_align=16),
                    row_layout_k,
                )
                g_gk_row = cute.make_tensor(
                    cute.make_ptr(self.acc_dtype, (gk_ptr + row_off).toint(), cute.AddressSpace.gmem, assumed_align=32),
                    cute.make_layout((ELEMS_K,)),
                )
                cute.autovec_copy(g_k_row, r_row_k)
                cute.autovec_copy(g_gk_row, r_row_gk)

                beta_val = sBeta[row].to(self.acc_dtype)
                for j in cutlass.range_constexpr(ELEMS_K):
                    k_val = r_row_k[j].to(self.acc_dtype)
                    gk_val = r_row_gk[j]
                    r_row_sB[j] = (k_val * beta_val * cute.exp2(gk_val, fastmath=True)).to(self.io_dtype)
                    r_row_kg[j] = (k_val * cute.exp2(r_gn[j] - gk_val, fastmath=True)).to(self.io_dtype)

                # Vectorized SMEM store for sB (manual swizzle address)
                row_in_atom = row & 7
                swizzled_col = (thread_col_base_k & 63) ^ (row_in_atom << 3)
                sB_phys_elem_off = (row >> 3) * 512 + row_in_atom * 64 + swizzled_col
                sB_row_ptr = cute.make_ptr(
                    self.io_dtype,
                    sB_base_int + sB_phys_elem_off * 2,
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                s_row_sB = cute.make_tensor(sB_row_ptr, cute.make_layout((ELEMS_K,)))
                cute.autovec_copy(r_row_sB, s_row_sB)

                # Vectorized GMEM store for kg
                g_kg_row = cute.make_tensor(
                    cute.make_ptr(self.io_dtype, (kg_ptr + row_off).toint(), cute.AddressSpace.gmem, assumed_align=16),
                    row_layout_k,
                )
                cute.autovec_copy(r_row_kg, g_kg_row)

            cta_sync.arrive_and_wait()

            # ------- 4b. MMA: acc_w = sA @ sBt (JIT A loading) -------
            acc_w = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            acc_w.fill(0.0)

            num_k_mma = cute.size(tSsBt, mode=[2])
            cute.copy(smem_tiled_copy_A, tSsA[None, None, 0], tSrA_copy_view[None, None, 0])
            cute.copy(smem_tiled_copy_Bt, tSsBt[None, None, 0], tSrBt_copy_view[None, None, 0])
            for k_mma in cutlass.range_constexpr(num_k_mma):
                k_next = (k_mma + 1) % num_k_mma
                cute.copy(smem_tiled_copy_A, tSsA[None, None, k_next], tSrA_copy_view[None, None, k_next])
                cute.copy(smem_tiled_copy_Bt, tSsBt[None, None, k_next], tSrBt_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc_w, tCrA[None, None, k_mma], tCrBt[None, None, k_mma], acc_w)

            # ------- 4c. Store w via SMEM staging → coalesced GMEM writes -------
            rW = cute.make_fragment_like(acc_w, self.io_dtype)
            rW.store(acc_w.load().to(self.io_dtype))
            cute.copy(atom_store, rW, tCsB)  # MMA output → sB (safe: MMA already consumed sB)
            cta_sync.arrive_and_wait()

            w_base_off = k_base_off
            for row_iter in cutlass.range(0, self.BT // rows_per_iter_k, unroll=4):
                row = thread_row_base_k + row_iter * rows_per_iter_k
                row_in_atom = row & 7
                swizzled_col = (thread_col_base_k & 63) ^ (row_in_atom << 3)
                sB_phys_off = (row >> 3) * 512 + row_in_atom * 64 + swizzled_col
                sB_rd_ptr = cute.make_ptr(
                    self.io_dtype,
                    sB_base_int + sB_phys_off * 2,
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                s_row_rd = cute.make_tensor(sB_rd_ptr, cute.make_layout((ELEMS_K,)))
                r_row_w = cute.make_rmem_tensor((ELEMS_K,), self.io_dtype)
                cute.autovec_copy(s_row_rd, r_row_w)

                w_row_off = w_base_off + row * H_K + thread_col_base_k
                g_w_row = cute.make_tensor(
                    cute.make_ptr(self.io_dtype, (w_ptr + w_row_off).toint(), cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((ELEMS_K,)),
                )
                cute.autovec_copy(r_row_w, g_w_row)

            cta_sync.arrive_and_wait()

        # =====================================================================
        # 5. V passes (NV = V/BV iterations)
        # =====================================================================
        H_V = cute.assume(H * V, divby=128)

        for i_v in cutlass.range(0, self.NV):
            v_col_off = i_v * BV
            time_base = (tok_offset + i_t * BT) * H + i_h
            v_base_off = time_base * V + v_col_off

            # ------- 5a. Elementwise: v * beta → sB (vectorized 1D row loads) -------
            ELEMS_V = 8
            threads_per_row_v = BV // ELEMS_V  # 8
            thread_col_base_v = (tidx % threads_per_row_v) * ELEMS_V
            thread_row_base_v = tidx // threads_per_row_v  # 0..15
            rows_per_iter_v = self.num_threads // threads_per_row_v  # 16

            r_row_v = cute.make_rmem_tensor((ELEMS_V,), self.io_dtype)
            r_row_sB_v = cute.make_rmem_tensor((ELEMS_V,), self.io_dtype)
            row_layout_v = cute.make_layout((ELEMS_V,))

            for row_iter in cutlass.range(0, self.BT // rows_per_iter_v, unroll=4):
                row = thread_row_base_v + row_iter * rows_per_iter_v
                row_off_v = v_base_off + row * H_V + thread_col_base_v
                g_v_row = cute.make_tensor(
                    cute.make_ptr(self.io_dtype, (v_ptr + row_off_v).toint(), cute.AddressSpace.gmem, assumed_align=16),
                    row_layout_v,
                )
                cute.autovec_copy(g_v_row, r_row_v)

                beta_val = sBeta[row].to(self.acc_dtype)
                for j in cutlass.range_constexpr(ELEMS_V):
                    v_val = r_row_v[j].to(self.acc_dtype)
                    r_row_sB_v[j] = (v_val * beta_val).to(self.io_dtype)

                # Vectorized SMEM store for sB
                row_in_atom_v = row & 7
                swizzled_col_v = (thread_col_base_v & 63) ^ (row_in_atom_v << 3)
                sB_phys_elem_off_v = (row >> 3) * 512 + row_in_atom_v * 64 + swizzled_col_v
                sB_row_ptr_v = cute.make_ptr(
                    self.io_dtype,
                    sB_base_int + sB_phys_elem_off_v * 2,
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                s_row_sB_v = cute.make_tensor(sB_row_ptr_v, cute.make_layout((ELEMS_V,)))
                cute.autovec_copy(r_row_sB_v, s_row_sB_v)

            cta_sync.arrive_and_wait()

            # ------- 5b. MMA: acc_u = sA @ sBt (JIT A loading) -------
            acc_u = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            acc_u.fill(0.0)

            num_k_mma_v = cute.size(tSsBt, mode=[2])
            cute.copy(smem_tiled_copy_A, tSsA[None, None, 0], tSrA_copy_view[None, None, 0])
            cute.copy(smem_tiled_copy_Bt, tSsBt[None, None, 0], tSrBt_copy_view[None, None, 0])
            for k_mma in cutlass.range_constexpr(num_k_mma_v):
                k_next = (k_mma + 1) % num_k_mma_v
                cute.copy(smem_tiled_copy_A, tSsA[None, None, k_next], tSrA_copy_view[None, None, k_next])
                cute.copy(smem_tiled_copy_Bt, tSsBt[None, None, k_next], tSrBt_copy_view[None, None, k_next])
                cute.gemm(tiled_mma, acc_u, tCrA[None, None, k_mma], tCrBt[None, None, k_mma], acc_u)

            # ------- 5c. Store u via SMEM staging → coalesced GMEM writes -------
            rU = cute.make_fragment_like(acc_u, self.io_dtype)
            rU.store(acc_u.load().to(self.io_dtype))
            cute.copy(atom_store, rU, tCsB)  # MMA output → sB
            cta_sync.arrive_and_wait()

            for row_iter in cutlass.range(0, self.BT // rows_per_iter_v, unroll=4):
                row = thread_row_base_v + row_iter * rows_per_iter_v
                row_in_atom_v = row & 7
                swizzled_col_v = (thread_col_base_v & 63) ^ (row_in_atom_v << 3)
                sB_phys_off_v = (row >> 3) * 512 + row_in_atom_v * 64 + swizzled_col_v
                sB_rd_ptr_v = cute.make_ptr(
                    self.io_dtype,
                    sB_base_int + sB_phys_off_v * 2,
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                s_row_rd_v = cute.make_tensor(sB_rd_ptr_v, cute.make_layout((ELEMS_V,)))
                r_row_u = cute.make_rmem_tensor((ELEMS_V,), self.io_dtype)
                cute.autovec_copy(s_row_rd_v, r_row_u)

                u_row_off = v_base_off + row * H_V + thread_col_base_v
                g_u_row = cute.make_tensor(
                    cute.make_ptr(self.io_dtype, (u_ptr + u_row_off).toint(), cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((ELEMS_V,)),
                )
                cute.autovec_copy(r_row_u, g_u_row)

            cta_sync.arrive_and_wait()


# ============================================================================
# Compile & public API
# ============================================================================

_occ_cache = {}


def _compile_occ(H, K, V, chunk_size=64):
    key = (H, K, V, chunk_size)
    if key in _occ_cache:
        return _occ_cache[key]

    kernel_obj = KDARecomputeWUOcc(K=K, V=V, chunk_size=chunk_size)

    sym_a = cute.sym_int()
    sym_b = cute.sym_int()
    BT = chunk_size

    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    v_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    beta_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H), stride_order=(2, 1, 0), assumed_align=128)
    A_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, BT), stride_order=(3, 2, 1, 0), assumed_align=128)
    gk_fake = make_fake_compact_tensor(cutlass.Float32, (sym_a, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    w_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
    u_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, V), stride_order=(3, 2, 1, 0), assumed_align=128)
    kg_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_a, sym_b, H, K), stride_order=(3, 2, 1, 0), assumed_align=128)
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
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    _occ_cache[key] = compiled_fn
    return compiled_fn


def recompute_w_u_fwd_occ(k, v, beta, A, gk):
    """Public API — occupancy=8 variant (non-varlen only)."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)

    ps = (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V))
    compiled_fn = _compile_occ(H, K, V, chunk_size=BT)
    compiled_fn(k, v, beta, A, gk, w, u, kg, ps)
    return w, u, None, kg


# ============================================================================
# Test & Benchmark
# ============================================================================


def main():
    import argparse

    from cula.ops.recompute_wu import recompute_w_u_fwd_ref

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["correctness", "benchmark", "both"], default="both")
    args = parser.parse_args()

    # ---- Non-varlen correctness ----
    print("\n=== Occ=8 kernel correctness ===")
    for B, T, H, K, V in [(1, 128, 1, 128, 128), (2, 256, 4, 128, 128), (1, 64, 1, 128, 128)]:
        BT = 64
        NT = T // BT
        torch.manual_seed(42)
        k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16))
        gk = (-torch.abs(torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)) * 0.1).cumsum(dim=1)
        A = torch.tril(torch.randn(B, NT, H, BT, BT, device="cuda", dtype=torch.bfloat16) * 0.1).reshape(B, T, H, BT)

        w_ref, u_ref, _, kg_ref = recompute_w_u_fwd_ref(k, v, beta, A, gk)
        w, u, _, kg = recompute_w_u_fwd_occ(k, v, beta, A, gk)
        torch.cuda.synchronize()

        dw = (w.float() - w_ref.float()).abs().max().item()
        du = (u.float() - u_ref.float()).abs().max().item()
        dkg = (kg.float() - kg_ref.float()).abs().max().item()
        ok = dw < 1.0 and du < 1.0 and dkg < 1.0
        print(f"  B={B} T={T} H={H}: w={dw:.6f} u={du:.6f} kg={dkg:.6f} {'PASS' if ok else 'FAIL'}")

    # ---- Benchmark ----
    if args.test in ("benchmark", "both"):
        print("\n=== Benchmark (B=4, T=4096, H=64, K=128, V=128) ===")
        Bb, Tb, Hb, Kb, Vb = 4, 4096, 64, 128, 128
        BTb = 64
        NTb = Tb // BTb
        torch.manual_seed(999)
        kb = torch.randn(Bb, Tb, Hb, Kb, device="cuda", dtype=torch.bfloat16) * 0.1
        vb = torch.randn(Bb, Tb, Hb, Vb, device="cuda", dtype=torch.bfloat16) * 0.1
        betab = torch.sigmoid(torch.randn(Bb, Tb, Hb, device="cuda", dtype=torch.bfloat16))
        gkb = (-torch.abs(torch.randn(Bb, Tb, Hb, Kb, device="cuda", dtype=torch.float32)) * 0.1).cumsum(dim=1)
        Ab = torch.tril(torch.randn(Bb, NTb, Hb, BTb, BTb, device="cuda", dtype=torch.bfloat16) * 0.1).reshape(Bb, Tb, Hb, BTb)

        n_iter = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(3):
            recompute_w_u_fwd_occ(kb, vb, betab, Ab, gkb)
        torch.cuda.synchronize()

        start.record()
        for _ in range(n_iter):
            recompute_w_u_fwd_occ(kb, vb, betab, Ab, gkb)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_iter
        print(f"  CuTeDSL occ=8: {ms:.3f} ms")

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
            print(f"  FLA Triton:    {fla_ms:.3f} ms")
            print(f"  Speedup:       {fla_ms / ms:.2f}x")
        except Exception as e:
            print(f"  FLA not available: {e}")


if __name__ == "__main__":
    main()
