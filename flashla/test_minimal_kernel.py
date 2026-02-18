#!/usr/bin/env python3
"""Minimal test to isolate kernel launch issues."""

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
import sys
sys.path.insert(0, '/ossfs/workspace/flashla/flashla')
from chunk_delta_h import ChunkDeltaRuleFwdH


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class MinimalTest:
    """Minimal warp-specialized kernel to test basic infrastructure."""

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
        
        self.threads_per_cta = self.threads_per_warp * 7  # 224
        
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
    def __call__(self, out_ptr: cute.Pointer, N: Int32, stream):
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
        
        # Plan TMEM
        SM100_TMEM_CAPACITY_COLS = 512
        acc_shape_wh = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH_fake = wh_tiled_mma.make_fragment_C(cute.append(acc_shape_wh, 2))
        num_wh_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccWH_fake)
        
        acc_shape_kv = kv_tiled_mma.partition_shape_C(self.kv_mma_tiler[:2])
        tCtAccKV_fake = kv_tiled_mma.make_fragment_C(cute.append(acc_shape_kv, 1))
        num_kv_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccKV_fake)
        
        total_cols = 1
        while total_cols < num_wh_cols + num_kv_cols:
            total_cols *= 2
        
        # Pre-compute TMEM sizes (compile-time constants)
        acc_shape_wh = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH_fake = wh_tiled_mma.make_fragment_C(cute.append(acc_shape_wh, 2))
        num_wh_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccWH_fake)
        
        acc_shape_kv = kv_tiled_mma.partition_shape_C(self.kv_mma_tiler[:2])
        tCtAccKV_fake = kv_tiled_mma.make_fragment_C(cute.append(acc_shape_kv, 1))
        num_kv_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccKV_fake)
        
        (
            _,
            _,
            _,
            self.tmem_total_cols,
        ) = ChunkDeltaRuleFwdH._plan_tmem_offsets(
            wh_tiled_mma, self.wh_mma_tiler,
            kv_tiled_mma, self.kv_mma_tiler,
            2,  # acc_stage
        )
        
        self.kernel(wh_tiled_mma, kv_tiled_mma, out_ptr, N).launch(
            grid=(1, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, wh_tiled_mma: cute.TiledMma, kv_tiled_mma: cute.TiledMma,
               out_ptr: cute.Pointer, N: Int32):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        
        @cute.struct
        class Storage:
            tmem_holding_buf: Int32
        
        smem = utils.SmemAllocator()
        storage = smem.allocate(Storage)
        
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
        
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_alloc(160)
            # Do nothing, just exist
            
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)
            # Do nothing
            
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(160)
            # Write output to signal kernel executed
            if tidx == 0:
                out_layout = cute.make_layout((N,), stride=(1,))
                out = cute.make_tensor(out_ptr, out_layout)
                out[0] = cutlass.Float32(42.0)
            
        elif warp_idx == self.store_warp_id:
            cute.arch.warpgroup_reg_dealloc(24)
            # Do nothing
        
        # Cleanup
        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)


def main():
    print("Testing minimal kernel...")
    
    out = torch.zeros(1, dtype=torch.float32, device='cuda')
    out_cute = from_dlpack(out)
    
    kernel = MinimalTest()
    stream = cutlass_torch.default_stream()
    
    print("Compiling...")
    start = time.time()
    compiled = cute.compile(kernel, out_cute.iterator, 1, stream)
    print(f"Compiled in {time.time()-start:.2f}s")
    
    print("Running...")
    compiled(out_cute.iterator, 1, stream)
    torch.cuda.synchronize()
    
    print(f"Output: {out[0].item()}")
    if out[0].item() == 42.0:
        print("PASS: Minimal kernel works!")
    else:
        print("FAIL: Output mismatch")


if __name__ == "__main__":
    main()
