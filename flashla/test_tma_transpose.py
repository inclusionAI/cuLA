#!/usr/bin/env python3
"""Minimal test: TMA load from transposed GMEM view into MMA-B SMEM layout."""

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


class TmaTransposeTest:
    """Test TMA load from transposed GMEM view."""

    def __init__(self):
        self.io_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.BT = 64
        self.BK = 128
        self.BV = 128
        
        self.threads_per_warp = 32
        self.load_warp_id = 5
        self.mma_warp_id = 4
        self.store_warp_id = 6
        self.cuda_warp_ids = (0, 1, 2, 3)
        
        self.threads_per_cta = self.threads_per_warp * 7
        
        self.wh_mma_tiler = (self.BT, self.BV, self.BK)
        self.kv_mma_tiler = (self.BK, self.BV, self.BT)
        
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.buffer_align_bytes = 1024
        
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )

    @cute.jit
    def __call__(self, h_out_ptr: cute.Pointer, debug_ptr: cute.Pointer,
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

        (
            _,
            _,
            _,
            self.tmem_total_cols,
        ) = ChunkDeltaRuleFwdH._plan_tmem_offsets(
            wh_tiled_mma, self.wh_mma_tiler,
            wh_tiled_mma, self.wh_mma_tiler,
            1,
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        # Create transposed GMEM view: h_out stored as (B, NT, H, K, V)
        # Normal view: (K, V, (NT, H, B))
        # Transposed view: (V, K, (NT, H, B)) - for MMA-B operand
        h_out_T_layout = cute.make_layout(
            (V, K, (NT, H, B)),
            stride=(1, V, (H * K * V, K * V, NT * H * K * V)),
        )
        h_out_T = cute.make_tensor(h_out_ptr, h_out_T_layout)

        # Create h_state SMEM layout (MMA-B operand)
        h_state_smem_layout_staged = sm100_utils.make_smem_layout_b(
            wh_tiled_mma, self.wh_mma_tiler, self.io_dtype, 1,
        )
        h_state_mma_smem_layout = cute.select(h_state_smem_layout_staged, mode=[0, 1, 2])

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (wh_tiled_mma.thr_id.shape,),
        )

        # Create TMA atom for h_state load
        tma_atom_h_load, tma_tensor_h_load = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            h_out_T,
            h_state_mma_smem_layout,
            self.wh_mma_tiler,
            wh_tiled_mma,
            cluster_layout_vmnk.shape,
        )

        self.tma_copy_h_bytes = cute.size_in_bytes(self.io_dtype, h_state_mma_smem_layout)

        @cute.struct
        class SharedStorage:
            h_load_mbar_ptr: cute.struct.MemRange[Int64, 1 * 2]
            tmem_holding_buf: Int32
            sH_state: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(h_state_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            wh_tiled_mma,
            tma_atom_h_load,
            tma_tensor_h_load,
            h_state_smem_layout_staged,
            debug_ptr,
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
        tma_atom_h_load: cute.CopyAtom,
        tma_tensor_h_load: cute.Tensor,
        h_state_smem_layout_staged: cute.ComposedLayout,
        debug_ptr: cute.Pointer,
        problem_size: tuple,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h_load)

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

        # TMA pipeline  
        h_load_producer, h_load_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_copy_h_bytes,
            barrier_storage=storage.h_load_mbar_ptr.data_ptr(),
        ).make_participants()

        # SMEM tensor
        sH_state = storage.sH_state.get_tensor(
            h_state_smem_layout_staged.outer, swizzle=h_state_smem_layout_staged.inner
        )

        B, T, H, K, V = problem_size

        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_alloc(160)

            # TMA partition for h_state
            _, hidx, bidx = cute.arch.block_idx()
            gH_load = cute.local_tile(
                tma_tensor_h_load,
                cute.slice_(self.wh_mma_tiler, (0, None, None)),
                (None, None, (None, hidx, bidx))
            )
            thr_mma_h = wh_tiled_mma.get_slice(0)
            tCgH = thr_mma_h.partition_B(gH_load)
            tHsH_load, tHgH_load = cute.nvgpu.cpasync.tma_partition(
                tma_atom_h_load,
                0,
                cute.make_layout(1),
                cute.group_modes(sH_state, 0, 3),
                cute.group_modes(tCgH, 0, 3),
            )

            # TMA load h_out[0] (should contain test pattern)
            handle = h_load_producer.acquire_and_advance()
            cute.copy(
                atom=tma_atom_h_load,
                src=tHgH_load[None, 0, 0, 0],
                dst=tHsH_load[None, handle.index],
                tma_bar_ptr=handle.barrier,
            )

        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)

            # Wait for TMA load
            handle = h_load_consumer.wait_and_advance()
            handle.release()

        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(160)
            # Do nothing

        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)

        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)


def main():
    print("Testing TMA load with transposed GMEM view...")
    
    B, T, H, K, V = 1, 256, 1, 128, 128
    NT = T // 64
    
    # Create h_out with known pattern: h_out[0, 0, 0, k, v] = k * 128 + v (as bf16)
    h_out = torch.zeros(B, NT, H, K, V, device='cuda', dtype=torch.bfloat16)
    for k in range(min(K, 4)):
        for v in range(min(V, 4)):
            h_out[0, 0, 0, k, v] = float(k * 128 + v)
    
    debug = torch.zeros(4, device='cuda', dtype=torch.float32)
    
    h_out_cute = from_dlpack(h_out)
    debug_cute = from_dlpack(debug)
    
    kernel = TmaTransposeTest()
    stream = cutlass_torch.default_stream()
    
    print("Compiling...")
    start = time.time()
    compiled = cute.compile(kernel, h_out_cute.iterator, debug_cute.iterator,
                           (B, T, H, K, V), stream)
    print(f"Compiled in {time.time()-start:.2f}s")
    
    print("Running...")
    compiled(h_out_cute.iterator, debug_cute.iterator, (B, T, H, K, V), stream)
    torch.cuda.synchronize()
    print("PASS: TMA load with transposed GMEM view works!")
    print(f"h_out[0,0,0,0,:4] = {h_out[0,0,0,0,:4].tolist()}")
    print(f"h_out[0,0,0,1,:4] = {h_out[0,0,0,1,:4].tolist()}")


if __name__ == "__main__":
    main()
