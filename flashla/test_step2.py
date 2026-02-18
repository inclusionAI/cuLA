#!/usr/bin/env python3
"""Incremental test: add MMA + pipeline to minimal kernel."""

import sys
sys.path.insert(0, '/ossfs/workspace/flashla/flashla')

import torch
import time
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32
from chunk_delta_h import ChunkDeltaRuleFwdH


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class StepTest:
    """Test pipeline + MMA step by step."""

    def __init__(self):
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.BT = 64
        self.BK = 128
        self.BV = 128
        
        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        
        self.threads_per_cta = self.threads_per_warp * 7
        
        self.wh_mma_tiler = (self.BT, self.BV, self.BK)
        self.kv_mma_tiler = (self.BK, self.BV, self.BT)
        
        self.w_stage = 2
        self.k_stage = 2
        self.acc_stage = 2
        
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.buffer_align_bytes = 1024
        
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )

    @cute.jit
    def __call__(self, w_ptr: cute.Pointer, kt_ptr: cute.Pointer, out_ptr: cute.Pointer,
                 problem_size: tuple, stream):
        B, T, H, K, V = problem_size
        NT = (T + self.BT - 1) // self.BT

        wh_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.wh_mma_tiler[:2],
        )
        
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
        )
        
        (
            self.tmem_wh_cols_offset,
            self.tmem_kv_cols_offset,
            _,
            self.tmem_total_cols,
        ) = ChunkDeltaRuleFwdH._plan_tmem_offsets(
            wh_tiled_mma, self.wh_mma_tiler,
            kv_tiled_mma, self.kv_mma_tiler,
            self.acc_stage,
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        # W layout: (T, K, (H, B))
        w_layout = cute.make_layout(
            (T, K, (H, B)),
            stride=(H * K, 1, (K, T * H * K)),
        )
        w = cute.make_tensor(w_ptr, w_layout)

        # W SMEM layout
        w_smem_layout_staged = sm100_utils.make_smem_layout_a(
            wh_tiled_mma, self.wh_mma_tiler, self.io_dtype, self.w_stage,
        )
        
        # H_state SMEM layout for MMA B operand: (BK, BV) 
        h_state_smem_layout_staged = sm100_utils.make_smem_layout_b(
            wh_tiled_mma, self.wh_mma_tiler, self.io_dtype, 1,
        )
        
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (wh_tiled_mma.thr_id.shape,),
        )

        w_smem_layout = cute.select(w_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, w, w_smem_layout,
            self.wh_mma_tiler, wh_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        self.tma_copy_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem_layout)

        @cute.struct
        class SharedStorage:
            load_w_mbar_ptr: cute.struct.MemRange[Int64, self.w_stage * 2]
            wh_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]
            vnew_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2]
            tmem_holding_buf: Int32
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sH_state: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(h_state_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            wh_tiled_mma,
            tma_atom_w,
            tma_tensor_w,
            w_smem_layout_staged,
            h_state_smem_layout_staged,
            out_ptr,
            problem_size,
        ).launch(
            grid=(1, H, B),
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        wh_tiled_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        tma_tensor_w: cute.Tensor,
        w_smem_layout_staged: cute.ComposedLayout,
        h_state_smem_layout_staged: cute.ComposedLayout,
        out_ptr: cute.Pointer,
        problem_size: tuple,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # TMEM
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total_cols)
        tmem.wait_for_alloc()
        tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

        # Pipeline for W load
        load_w_producer, load_w_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_w_bytes,
            barrier_storage=storage.load_w_mbar_ptr.data_ptr(),
        ).make_participants()

        # Pipeline for W@H result
        wh_producer, wh_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            barrier_storage=storage.wh_mbar_ptr.data_ptr(),
        ).make_participants()

        # Test: add vnew pipeline creation
        vnew_producer, vnew_consumer = pipeline.PipelineAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.vnew_mbar_ptr.data_ptr(),
        ).make_participants()

        # SMEM tensors
        sW = storage.sW.get_tensor(
            w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner
        )
        sH_state = storage.sH_state.get_tensor(
            h_state_smem_layout_staged.outer, swizzle=h_state_smem_layout_staged.inner
        )

        # MMA partitions
        tCrW = wh_tiled_mma.make_fragment_A(sW)
        tCrH_for_wh = wh_tiled_mma.make_fragment_B(sH_state)
        acc_shape_wh = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH = cute.make_tensor(
            tmem_ptr_base + self.tmem_wh_cols_offset,
            wh_tiled_mma.make_fragment_C(cute.append(acc_shape_wh, self.acc_stage)).layout
        )

        (_, hidx, bidx) = cute.arch.block_idx()
        B, T, H, K, V = problem_size
        BT = self.BT
        NT = (T + BT - 1) // BT

        # ======================== LOAD WARP ========================
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_alloc(160)

            tWsW, tWgW = self._tma_partition_for_a(
                tma_atom_w, tma_tensor_w, sW, self.wh_mma_tiler, wh_tiled_mma
            )

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                w_handle = load_w_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_w,
                    src=tWgW[None, chunk_idx, 0],
                    dst=tWsW[None, w_handle.index],
                    tma_bar_ptr=w_handle.barrier,
                )

        # ======================== MMA WARP ========================
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                w_handle = load_w_consumer.wait_and_advance()

                wh_handle = wh_producer.acquire_and_advance()
                # W @ H_state
                for kphase_idx in cutlass.range(cute.size(tCrH_for_wh, mode=[2]), unroll_full=True):
                    wh_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE,
                        cutlass.Boolean(kphase_idx != 0),
                    )
                    cute.gemm(
                        wh_tiled_mma,
                        tCtAccWH[None, None, None, wh_handle.index],
                        tCrW[None, None, kphase_idx, w_handle.index],
                        tCrH_for_wh[None, None, kphase_idx, 0],
                        tCtAccWH[None, None, None, wh_handle.index],
                    )
                wh_handle.commit()
                w_handle.release()

                # Wait for vnew from CUDA warps
                vnew_handle = vnew_consumer.wait_and_advance()
                vnew_handle.release()

        # ======================== CUDA WARPS ========================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(160)
            
            for chunk_idx in cutlass.range(0, NT, unroll=0):
                wh_handle = wh_consumer.wait_and_advance()
                wh_handle.release()
                
                vnew_handle = vnew_producer.acquire_and_advance()
                vnew_handle.commit()

        # ======================== STORE WARP ========================
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)

        # Cleanup
        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)

    @cute.jit
    def _tma_partition_for_a(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma):
        _, hidx, bidx = cute.arch.block_idx()
        gX = cute.local_tile(
            tma_tensor,
            cute.slice_(tile_shape, (None, 0, None)),
            (None, None, (hidx, bidx))
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_A(gX)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX


def main():
    print("Testing MMA pipeline kernel...")
    
    B, T, H, K, V = 1, 256, 1, 128, 128
    NT = T // 64
    
    w = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    kt = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
    out = torch.zeros(1, device='cuda', dtype=torch.float32)
    
    w_cute = from_dlpack(w)
    kt_cute = from_dlpack(kt)
    out_cute = from_dlpack(out)
    
    kernel = StepTest()
    stream = cutlass_torch.default_stream()
    
    print("Compiling...")
    start = time.time()
    compiled = cute.compile(kernel, w_cute.iterator, kt_cute.iterator, out_cute.iterator,
                           (B, T, H, K, V), stream)
    print(f"Compiled in {time.time()-start:.2f}s")
    
    print("Running...")
    compiled(w_cute.iterator, kt_cute.iterator, out_cute.iterator, (B, T, H, K, V), stream)
    torch.cuda.synchronize()
    print("PASS: MMA pipeline kernel works!")


if __name__ == "__main__":
    main()
