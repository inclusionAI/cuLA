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
    ):
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype

        self.BT = chunk_size   # 64
        self.BK = head_dim_k   # 128
        self.BV = 64           # V tiling

        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        self.empty_warp_id = 7
        self.num_regs_cuda = 232
        self.num_regs_others = 40
        self.threads_per_cta = self.threads_per_warp * 8

        # WH MMA tiler: (M=BV=64, N=BT=64, K=BK=128), A & B both SS
        self.wh_mma_tiler = (self.BV, self.BT, self.BK)
        # KV MMA tiler: (M=BV=64, N=BK=128, K=BT=64), A & B both SS
        self.kv_mma_tiler = (self.BV, self.BK, self.BT)

        self.k_stage = 1
        self.w_stage = 1
        self.acc_stage = 1
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta,
        )
        self.buffer_align_bytes = 1024

    @staticmethod
    def _plan_tmem_offsets(tiled_mma_wh, tile_wh, tiled_mma_kv, tile_kv, acc_stages):
        SM100_TMEM_CAPACITY_COLS = 512
        # WH acc: (BV=64, BT=64) FP32
        wh_shape = tiled_mma_wh.partition_shape_C(tile_wh[:2])
        wh_fake = tiled_mma_wh.make_fragment_C(cute.append(wh_shape, acc_stages))
        num_wh = tcgen05.find_tmem_tensor_col_offset(wh_fake)
        # KV acc: (BV=64, BK=128) FP32
        kv_shape = tiled_mma_kv.partition_shape_C(tile_kv[:2])
        kv_fake = tiled_mma_kv.make_fragment_C(cute.append(kv_shape, 1))
        num_kv = tcgen05.find_tmem_tensor_col_offset(kv_fake)

        wh_off = 0
        kv_off = wh_off + num_wh
        total_tmp = kv_off + num_kv
        total = 1
        while total < total_tmp:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS
        if cutlass.const_expr(PRINT_DEBUG):
            print(f"  TMEM: WH={num_wh}@{wh_off}, KV={num_kv}@{kv_off}, total={total}")
        return wh_off, kv_off, total

    def _compute_grid(self, B, H, V):
        return ((V + self.BV - 1) // self.BV, H, B)

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
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        use_g: Int32,
        use_gk: Int32,
        use_initial_state: Int32,
        store_final_state: Int32,
        save_v_new: Int32,
        stream,
    ):
        B, T, H, K, V = problem_size
        NT = (T + self.BT - 1) // self.BT

        # ===================== GMEM layouts =====================
        kt_layout = cute.make_layout((K, T, (H, B)), stride=(1, H * K, (K, T * H * K)))
        kt = cute.make_tensor(k_ptr, kt_layout)

        w_layout = cute.make_layout((T, K, (H, B)), stride=(H * K, 1, (K, T * H * K)))
        w = cute.make_tensor(w_ptr, w_layout)

        u_layout = cute.make_layout((T, V, (H, B)), stride=(H * V, 1, (V, T * H * V)))
        u = cute.make_tensor(u_ptr, u_layout)

        v_new = cute.make_tensor(v_new_ptr, u_layout)

        h_out_T_layout = cute.make_layout(
            (V, K, (NT, H, B)),
            stride=(1, V, (H * K * V, K * V, NT * H * K * V)),
        )
        h_out_T = cute.make_tensor(h_out_ptr, h_out_T_layout)

        h0_layout = cute.make_layout((K, V, (H, B)), stride=(V, 1, (K * V, H * K * V)))
        h0 = cute.make_tensor(h0_ptr, h0_layout)

        ht_T_layout = cute.make_layout((V, K, (H, B)), stride=(1, V, (K * V, H * K * V)))
        ht_T = cute.make_tensor(ht_ptr, ht_T_layout)

        g_layout = cute.make_layout((T, (H, B)), stride=(H, (1, T * H)))
        g = cute.make_tensor(g_ptr, g_layout)

        gk_layout = cute.make_layout((T, K, (H, B)), stride=(H * K, 1, (K, T * H * K)))
        gk = cute.make_tensor(gk_ptr, gk_layout)

        # Transposed U view: (V, T, (H,B)) to match WH acc shape (M=BV, N=BT)
        u_T_layout = cute.make_layout((V, T, (H, B)), stride=(1, H * V, (V, T * H * V)))
        u_T = cute.make_tensor(u_ptr, u_T_layout)

        self.k_dtype = kt.element_type
        self.w_dtype = w.element_type
        self.u_dtype = u.element_type

        # ===================== MMA setup =====================
        # WH MMA: A=state(SMEM, K-major), B=W(SMEM, K-major)
        wh_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,   # A: state, K-major (BK contiguous)
            tcgen05.OperandMajorMode.K,   # B: W, K-major (BK contiguous)
            self.acc_dtype,
            self.cta_group,
            self.wh_mma_tiler[:2],
        )

        # KV MMA: A=v_new^T(SMEM, MN-major), B=K^T(SMEM, MN-major)
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,  # A: v_new^T, MN-major (BV contiguous)
            tcgen05.OperandMajorMode.MN,  # B: K^T, MN-major (BK contiguous)
            self.acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
        )

        # ===================== TMEM offsets =====================
        (self.tmem_wh_off, self.tmem_kv_off, self.tmem_total) = self._plan_tmem_offsets(
            wh_tiled_mma, self.wh_mma_tiler,
            kv_tiled_mma, self.kv_mma_tiler,
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
        # State (h^T BF16) as A operand of WH MMA — MMA read view
        state_mma_staged = sm100_utils.make_smem_layout_a(
            wh_tiled_mma, self.wh_mma_tiler, self.io_dtype, 1,
        )
        # State epilogue for CUDA R2S writes — dual-view of same buffer
        # ROW_MAJOR for (BV, BK): BK contiguous → K-major, matches A operand layout
        state_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BV, self.BK),  # (64, 128)
            1,
        )
        # v_new^T as A operand of KV MMA — MMA read view
        vnew_mma_staged = sm100_utils.make_smem_layout_a(
            kv_tiled_mma, self.kv_mma_tiler, self.io_dtype, 1,
        )
        # v_new epilogue for CUDA R2S writes — dual-view of same buffer
        # COL_MAJOR for (BV, BT): BV contiguous → MN-major for KV A operand
        vnew_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),  # (64, 64)
            1,
        )
        # h_out epilogue for TMA store
        # COL_MAJOR for (BV, BK): BV contiguous → matches V stride 1 in h_out_T GMEM
        h_out_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BK),  # (64, 128)
            1,
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

        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem)
        self.tma_kt_bytes = cute.size_in_bytes(self.io_dtype, kt_smem)

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            load_w_mbar: cute.struct.MemRange[Int64, self.w_stage * 2]
            load_kt_mbar: cute.struct.MemRange[Int64, self.k_stage * 2]
            state_smem_mbar: cute.struct.MemRange[Int64, 1 * 2]       # CUDA→MMA: sState ready
            wh_done_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]  # MMA→CUDA: WH done
            vnew_smem_mbar: cute.struct.MemRange[Int64, 1 * 2]        # CUDA→MMA: sVnew ready
            kv_done_mbar: cute.struct.MemRange[Int64, 1 * 2]          # MMA→CUDA: KV done
            h_out_mbar: cute.struct.MemRange[Int64, 1 * 2]            # CUDA→Store: sH_epi ready
            tmem_holding_buf: Int32
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_staged)],
                self.buffer_align_bytes,
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(kt_smem_staged)],
                self.buffer_align_bytes,
            ]
            sState: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, max(cute.cosize(state_mma_staged), cute.cosize(state_epi_staged))],
                self.buffer_align_bytes,
            ]
            sVnew: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, max(cute.cosize(vnew_mma_staged), cute.cosize(vnew_epi_staged))],
                self.buffer_align_bytes,
            ]
            sH_epi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(h_out_epi_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.grid = self._compute_grid(B, H, V)

        self.kernel(
            wh_tiled_mma, kv_tiled_mma,
            tma_atom_w, tma_tensor_w,
            tma_atom_kt, tma_tensor_kt,
            tma_atom_h_out, tma_tensor_h_out,
            tma_atom_ht, tma_tensor_ht,
            g, gk, h0, u, u_T, h_out_T, v_new,
            w_smem_staged, kt_smem_staged,
            state_mma_staged, state_epi_staged,
            vnew_mma_staged, vnew_epi_staged,
            h_out_epi_staged,
            problem_size,
            use_g, use_gk, use_initial_state, store_final_state, save_v_new,
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
        tma_atom_kt: cute.CopyAtom,
        tma_tensor_kt: cute.Tensor,
        tma_atom_h_out: cute.CopyAtom,
        tma_tensor_h_out: cute.Tensor,
        tma_atom_ht: cute.CopyAtom,
        tma_tensor_ht: cute.Tensor,
        g: cute.Tensor,
        gk: cute.Tensor,
        h0: cute.Tensor,
        u_tensor: cute.Tensor,
        u_T_tensor: cute.Tensor,
        h_out_T_tensor: cute.Tensor,
        v_new_tensor: cute.Tensor,
        w_smem_staged: cute.ComposedLayout,
        kt_smem_staged: cute.ComposedLayout,
        state_mma_staged: cute.ComposedLayout,
        state_epi_staged: cute.ComposedLayout,
        vnew_mma_staged: cute.ComposedLayout,
        vnew_epi_staged: cute.ComposedLayout,
        h_out_epi_staged: cute.ComposedLayout,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        use_g: Int32,
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

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

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
            barrier_storage=storage.state_smem_mbar.data_ptr(),
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
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.h_out_mbar.data_ptr(),
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
        sState_mma = storage.sState.get_tensor(state_mma_staged.outer, swizzle=state_mma_staged.inner)
        sState_epi = storage.sState.get_tensor(state_epi_staged.outer, swizzle=state_epi_staged.inner)
        sVnew_mma = storage.sVnew.get_tensor(vnew_mma_staged.outer, swizzle=vnew_mma_staged.inner)
        sVnew_epi = storage.sVnew.get_tensor(vnew_epi_staged.outer, swizzle=vnew_epi_staged.inner)
        sH_epi = storage.sH_epi.get_tensor(h_out_epi_staged.outer, swizzle=h_out_epi_staged.inner)

        # ===================== MMA fragments =====================
        # WH MMA: A=sState, B=sW, acc=WH TMEM
        tCrState = wh_tiled_mma.make_fragment_A(sState_mma)
        tCrW = wh_tiled_mma.make_fragment_B(sW)
        wh_shape = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH_fake = wh_tiled_mma.make_fragment_C(cute.append(wh_shape, self.acc_stage))
        tCtAccWH = cute.make_tensor(tmem_ptr + self.tmem_wh_off, tCtAccWH_fake.layout)

        # KV MMA: A=sVnew, B=sKt, acc=KV TMEM
        tCrVnew = kv_tiled_mma.make_fragment_A(sVnew_mma)
        tCrKt = kv_tiled_mma.make_fragment_B(sKt)
        kv_shape = kv_tiled_mma.partition_shape_C(self.kv_mma_tiler[:2])
        tCtAccKV_fake = kv_tiled_mma.make_fragment_C(cute.append(kv_shape, 1))
        tCtAccKV = cute.make_tensor(tmem_ptr + self.tmem_kv_off, tCtAccKV_fake.layout)

        # ===================== Block indices =====================
        (v_tile_idx, hidx, bidx) = cute.arch.block_idx()
        B, T, H, K, V = problem_size
        BT = self.BT
        NT = (T + BT - 1) // BT

        # =========================================================================
        # LOAD WARP
        # =========================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            tWsW, tWgW = self._tma_partition_B(
                tma_atom_w, tma_tensor_w, sW, self.wh_mma_tiler, wh_tiled_mma,
            )
            tKsK, tKgK = self._tma_partition_B(
                tma_atom_kt, tma_tensor_kt, sKt, self.kv_mma_tiler, kv_tiled_mma,
            )

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                w_h = load_w_P.acquire_and_advance()
                cute.copy(atom=tma_atom_w, src=tWgW[None, chunk_idx, 0],
                          dst=tWsW[None, w_h.index], tma_bar_ptr=w_h.barrier)

                kt_h = load_kt_P.acquire_and_advance()
                cute.copy(atom=tma_atom_kt, src=tKgK[None, 0, chunk_idx],
                          dst=tKsK[None, kt_h.index], tma_bar_ptr=kt_h.barrier)

        # =========================================================================
        # MMA WARP
        # =========================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

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

                # --- KV MMA: v_new(SMEM) × K^T(SMEM) → update (ACCUMULATE=False always) ---
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
            # Update from KV MMA (temporary)
            tTR_rUpdate = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)

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
            tTR_rWH = cute.make_rmem_tensor(tTR_sWH.shape, self.acc_dtype)

            # ----- R2S: KV T2R regs → sState_epi (ROW_MAJOR, BV×BK) -----
            r2s_atom_state = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r_kv,
            )
            tiled_r2s_state = cute.make_tiled_copy_D(r2s_atom_state, tiled_t2r_kv)
            thr_r2s_state = tiled_r2s_state.get_slice(local_tidx)
            tRS_sState = thr_r2s_state.partition_D(sState_epi)

            # ----- R2S: KV T2R regs → sH_epi (COL_MAJOR, BV×BK) -----
            r2s_atom_h = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r_kv,
            )
            tiled_r2s_h = cute.make_tiled_copy_D(r2s_atom_h, tiled_t2r_kv)
            thr_r2s_h = tiled_r2s_h.get_slice(local_tidx)
            tRS_sH = thr_r2s_h.partition_D(sH_epi)

            # ----- R2S: WH T2R regs → sVnew_epi (COL_MAJOR, BV×BT) -----
            r2s_atom_vnew = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR, self.io_dtype, self.acc_dtype, tiled_t2r_wh,
            )
            tiled_r2s_vnew = cute.make_tiled_copy_D(r2s_atom_vnew, tiled_t2r_wh)
            thr_r2s_vnew = tiled_r2s_vnew.get_slice(local_tidx)
            tRS_sVnew = thr_r2s_vnew.partition_D(sVnew_epi)

            # ----- BF16 register tensors -----
            tTR_rKV_bf16 = cute.make_rmem_tensor(tTR_rKV.shape, self.io_dtype)
            tTR_rVnew_bf16 = cute.make_rmem_tensor(tTR_rWH.shape, self.io_dtype)

            # ----- U GMEM (direct read, bypass SMEM) -----
            # Use transposed U view (V, T, ...) to match WH acc layout (M=BV, N=BT)
            gU_all = cute.local_tile(u_T_tensor, (self.BV, self.BT), (None, None, (hidx, bidx)))
            tTR_gU = thr_t2r_wh.partition_D(gU_all)
            copy_atom_s2r = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.io_dtype,
                num_bits_per_copy=self.io_dtype.width,
            )

            # ----- g/gk GMEM -----
            gG = cute.local_tile(g, (self.BT,), (None, (hidx, bidx)))
            gGK = cute.local_tile(gk, (self.BT, self.BK), (None, None, (hidx, bidx)))

            # ----- Identity tensor for element coordinates -----
            vnew_tile = cute.dice(self.wh_mma_tiler, (1, 1, None))  # (BV, BT)
            cM_vnew = cute.make_identity_tensor(vnew_tile)
            tTR_cM = thr_t2r_wh.partition_D(cM_vnew)

            # ===== Initialize h = 0 in registers =====
            for ei in cutlass.range_constexpr(cute.size(tTR_rKV)):
                tTR_rKV[ei] = Float32(0.0)

            # ===== Main loop =====
            for chunk_idx in cutlass.range(0, NT, unroll=0):
                # ========================================
                # Phase 1: h state → sState + sH_epi
                # ========================================
                # Convert h (FP32) to BF16 for R2S
                h_vec = tTR_rKV.load()
                tTR_rKV_bf16.store(h_vec.to(self.io_dtype))

                # R2S to sH_epi (for h_out TMA store)
                tRS_rH = tiled_r2s_h.retile(tTR_rKV_bf16)
                h_handle = h_out_P.acquire_and_advance()
                cute.copy(tiled_r2s_h, tRS_rH, tRS_sH[(None, None, None, h_handle.index)])
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                h_handle.commit()

                # R2S to sState (for WH MMA A operand)
                tRS_rState = tiled_r2s_state.retile(tTR_rKV_bf16)
                state_h = state_smem_P.acquire_and_advance()
                cute.copy(tiled_r2s_state, tRS_rState, tRS_sState[(None, None, None, state_h.index)])
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                state_h.commit()

                # Decay h in registers (R2S already done, safe to modify)
                if use_g:
                    g_off = chunk_idx * self.BT
                    g_last = g[(g_off + self.BT - 1, (hidx, bidx))]
                    g_scale_h = cute.exp2(g_last * INV_LN2)
                    h_vec = h_vec * g_scale_h
                    tTR_rKV.store(h_vec)

                # ========================================
                # Phase 2: v_new from WH result
                # ========================================
                wh_h = wh_done_C.wait_and_advance()

                # T2R: WH acc FP32 → registers
                cute.copy(tiled_t2r_wh, tTR_tWH[(None, None, None, wh_h.index)], tTR_rWH)
                cute.arch.fence_view_async_tmem_load()
                wh_h.release()

                # Load U from GMEM (u_T indexed: v_tile, chunk)
                tTR_gU_i = tTR_gU[(None, None, None, v_tile_idx, chunk_idx)]
                tTR_rU = cute.make_rmem_tensor(tTR_gU_i.shape, self.io_dtype)
                cute.copy(copy_atom_s2r, tTR_gU_i, tTR_rU)

                # v_new = u - WH (FP32)
                wh_vec = tTR_rWH.load()
                u_vec = tTR_rU.load().to(self.acc_dtype)
                vnew_vec = u_vec - wh_vec

                # Apply g gate to v_new (per-timestep scaling)
                tTR_rVnew_bf16.store(vnew_vec.to(self.io_dtype))
                if use_g:
                    g_off = chunk_idx * self.BT
                    g_last_v = g[(g_off + self.BT - 1, (hidx, bidx))]
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        v_coord, t_coord = tTR_cM[ei]  # (BV dim, BT dim)
                        g_row = g[(g_off + t_coord, (hidx, bidx))]  # time index
                        g_diff = g_last_v - g_row
                        gs = cute.exp2(g_diff * INV_LN2)
                        val = tTR_rVnew_bf16[ei].to(self.acc_dtype)
                        tTR_rVnew_bf16[ei] = (val * gs).to(self.io_dtype)

                # R2S: v_new → sVnew_epi
                tRS_rVnew = tiled_r2s_vnew.retile(tTR_rVnew_bf16)
                vnew_h = vnew_smem_P.acquire_and_advance()
                cute.copy(tiled_r2s_vnew, tRS_rVnew, tRS_sVnew[(None, None, None, vnew_h.index)])
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                vnew_h.commit()

                # ========================================
                # Phase 3: KV update → h
                # ========================================
                kv_h = kv_done_C.wait_and_advance()

                # T2R: KV acc (update = K^T × v_new) → tTR_rUpdate
                cute.copy(tiled_t2r_kv, tTR_tKV[(None, None, None, 0)], tTR_rUpdate)
                cute.arch.fence_view_async_tmem_load()
                kv_h.release()

                # h_new = h_decayed + update (vectorized in registers)
                h_vec = tTR_rKV.load()
                update_vec = tTR_rUpdate.load()
                tTR_rKV.store(h_vec + update_vec)

        # =========================================================================
        # STORE WARP
        # =========================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_others)

            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h_out)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_ht)

            gH_st = tma_tensor_h_out[None, None, (None, hidx, bidx)]
            tma_h_st, bSG_sH, bSG_gH = self._epilog_partition(
                tma_atom_h_out, gH_st, (self.BV, self.BK), sH_epi,
            )

            gHt_st = tma_tensor_ht[None, None, (hidx, bidx)]
            tma_ht_st, bSG_sHt, bSG_gHt = self._epilog_partition(
                tma_atom_ht, gHt_st, (self.BV, self.BK), sH_epi,
            )

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                h_handle = h_out_C.wait_and_advance()

                cute.copy(tma_h_st, bSG_sH[None, h_handle.index],
                          bSG_gH[(None, v_tile_idx, 0, chunk_idx)])
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
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma):
        """Partition B operand tensors for TMA copy."""
        _, hidx, bidx = cute.arch.block_idx()
        coord = (0, None, None)
        gX = cute.local_tile(
            tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, bidx))
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
    g = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    h0 = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
    h_out = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    ht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

    _, _, h_ref_fp32 = reference_chunk_delta_rule_fwd_h(k, w, u, h0=None, chunk_size=BT)
    _, h_ref_bf16 = reference_bf16_roundtrip(k, w, u, h0=None, chunk_size=BT)

    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
    stream = cutlass_torch.default_stream()

    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g), from_dlpack(gk)
    h0c, hc, vnc, htc = from_dlpack(h0), from_dlpack(h_out), from_dlpack(v_new), from_dlpack(ht)

    print("\nCompiling...")
    t0 = time.time()
    compiled = cute.compile(
        kernel,
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        (B, T, H, K, V), 0, 0, 0, 0, 0, stream,
    )
    print(f"Compiled in {time.time()-t0:.2f}s")

    print("Running...")
    compiled(
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        (B, T, H, K, V), 0, 0, 0, 0, 0, stream,
    )
    torch.cuda.synchronize()

    # h_out[0] should be zeros (h=0 before first chunk)
    h0_ok = (h_out[0, 0, 0] == 0).all().item()
    print(f"\nh_out[0] all zeros: {h0_ok}")

    # Compare: h_out[t+1] should match h_ref[t] (state after chunk t)
    print(f"\n{'Chunk':<8} {'h_out[t+1] vs bf16_ref[t]':>28}")
    max_diff = 0.0
    for t in range(min(NT - 1, len(h_ref_bf16))):
        hk = h_out[0, t + 1, 0]
        hr = h_ref_bf16[t]
        d = (hk.float() - hr.float()).abs().max().item()
        max_diff = max(max_diff, d)
        print(f"  {t:<6} {d:>28.6f}")

    if NT > 1:
        print(f"\n  kernel h_out[1][0,:8]: {h_out[0, 1, 0, 0, :8].tolist()}")
        print(f"  ref    h_bf16[0][0,:8]: {h_ref_bf16[0][0,:8].tolist()}")
    if NT > 2:
        print(f"\n  kernel h_out[2][0,:8]: {h_out[0, 2, 0, 0, :8].tolist()}")
        print(f"  ref    h_bf16[1][0,:8]: {h_ref_bf16[1][0,:8].tolist()}")

    print(f"\nMax diff (h_out[t+1] vs ref[t]): {max_diff:.6f}")
    if NT > 2:
        d_consecutive = (h_out[0, 2, 0].float() - h_out[0, 1, 0].float()).abs().max().item()
        print(f"h_out[2] vs h_out[1] max diff: {d_consecutive:.8f}")
    print("PASS" if max_diff < 0.5 else "FAIL")


if __name__ == "__main__":
    main()
