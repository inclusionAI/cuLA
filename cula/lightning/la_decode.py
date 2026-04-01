# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Linear Attention Decode Kernel - Single Token Generation

This file implements a fused CUDA kernel using CUTLASS CuTe DSL for executing
Linear Attention updates during decode phase. Simplified compared to GDN.

Architecture Design:
- Uses TMA (Tensor Memory Accelerator) for efficient Global Memory → Shared Memory transfers
- Employs 2-stage pipeline to overlap loading and computation, hiding memory latency
- Each block uses 128 threads (4 warps), with each warp processing one matrix row
- Tile size: 8x128 (TILE_V x TILE_K)

Computation Flow:
1. Warp 0 handles TMA prefetch, loading data from GMEM to SMEM
2. All warps compute in parallel: state update with exponential decay
3. Each warp processes one row of data, completing h_new = exp(decay) * h + k ⊗ v
4. Uses warp-level shuffle for efficient reduction operations
5. Results are vectorized and written back to Global Memory

Core Formula:
    state_new = exp(decay) * state_old + k ⊗ v
    output = q @ state_new
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from cula.utils import USE_FAST_MATH

# ============================================================================
# Global configuration
# ============================================================================
TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128  # 4 warps
NUM_BLOCKS_PER_STATE = 8


@cute.kernel
def la_decode_kernel_small_batch_pretranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    decay_scales: cute.Tensor,  # [H]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

    HV = H
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // NUM_BLOCKS_PER_STATE
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_v = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_decay_scale = -cutlass.Float32(decay_scales[i_h])
    r_decay = cute.exp(r_decay_scale, fastmath=USE_FAST_MATH)

    cute.arch.barrier()

    # Get current batch
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K)
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # split tiles in V-dimension
    gSrc = cute.local_tile(gSrc_batch, (TILE_V, TILE_K), (None, 0))  # (TILE_V, TILE_K, num_v_tiles)

    # Partition for load
    thr_copy_load = tiled_copy_load.get_slice(tidx)

    # ===================================================================
    # Prefetch: All threads participate in cp.async load
    # ===================================================================
    start_v_tiles = batch_inner * num_v_tiles_per_block
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
    for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        gSrc_tile = gSrc[(None, None, v_tiles)]
        sData_stage = sData[(None, None, stage)]

        thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
        thr_sData = thr_copy_load.partition_D(sData_stage)

        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()

    for i in range(vec_size):
        r_q[i] = cutlass.Float32(q[i_n, i_h, i * 32 + lane_id])
        r_k[i] = cutlass.Float32(k[i_n, i_h, i * 32 + lane_id])
        r_v[i] = cutlass.Float32(v[i_n, i_hv, i * 32 + lane_id])

    cute.arch.barrier()  # Ensure all threads finish writing to sV

    # Apply scaling in Float32
    for i in range(vec_size):
        r_q[i] = r_q[i] * scale

    # ===================================================================
    # Mainloop: All threads participate
    # ===================================================================
    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        # Step 1: Wait for current stage to complete
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Step 2: Issue async load for next tile (after compute)
        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < end_v_tiles:
            next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES

            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]

            thr_gSrc = thr_copy_load.partition_S(gSrc_next)
            thr_sData = thr_copy_load.partition_D(sData_next)

            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Step 3: Compute using data from current stage
        for row in range(0, TILE_V, 4):
            row_offset = tidx // 32

            v_idx = v_tiles * TILE_V + row + row_offset
            v_row = cute.arch.shuffle_sync(r_v[v_idx // 32], v_idx % 32, mask=-1, mask_and_clamp=31)

            sum_hq = 0.0
            for i in range(vec_size):
                r_h[i] = sData[(row + row_offset, i * 32 + lane_id, stage)]
                r_h[i] = r_h[i] * r_decay
                r_h[i] += r_k[i] * v_row
                gDst[(0, row + row_offset, i * 32 + lane_id, v_tiles)] = r_h[i]
                sum_hq += r_h[i] * r_q[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset, mask=-1, mask_and_clamp=31)

            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # ===================================================================
    # Final writeback: Copy output from shared memory to global memory
    # All threads write (V=128, NUM_THREADS=128)
    # ===================================================================
    cute.arch.barrier()  # Ensure all writes to sOutput are complete
    if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
        o[(i_n, i_hv, tidx)] = sOutput[tidx]


@cute.kernel
def la_decode_kernel_big_batch_pretranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    decay_scales: cute.Tensor,  # [H]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

    HV = H
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    batch_idx, _, _ = cute.arch.block_idx()
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_v = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

    cute.arch.barrier()

    # Get current batch
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K)
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # split tiles in V-dimension
    gSrc = cute.local_tile(gSrc_batch, (TILE_V, TILE_K), (None, 0))  # (TILE_V, TILE_K, num_v_tiles)

    # Partition for load
    thr_copy_load = tiled_copy_load.get_slice(tidx)

    # ===================================================================
    # Prefetch: All threads participate in cp.async load
    # ===================================================================
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
    for v_tiles in range(prefetch_count):
        stage = v_tiles % NUM_STAGES

        gSrc_tile = gSrc[(None, None, v_tiles)]
        sData_stage = sData[(None, None, stage)]

        thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
        thr_sData = thr_copy_load.partition_D(sData_stage)

        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()

    for i in range(vec_size):
        r_q[i] = cutlass.Float32(q[i_n, i_h, i * 32 + lane_id])
        r_k[i] = cutlass.Float32(k[i_n, i_h, i * 32 + lane_id])
        r_v[i] = cutlass.Float32(v[i_n, i_hv, i * 32 + lane_id])

    cute.arch.barrier()  # Ensure all threads finish writing to sV

    # ===================================================================
    # Compute g and beta (scalar values)
    # ===================================================================
    # Apply scaling in Float32
    for i in range(vec_size):
        r_q[i] = r_q[i] * scale

    r_g = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)

    # ===================================================================
    # Mainloop: All threads participate
    # ===================================================================
    for v_tiles in range(num_v_tiles):
        stage = v_tiles % NUM_STAGES

        # Step 1: Wait for current stage to complete
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Step 2: Issue async load for next tile (after compute)
        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < num_v_tiles:
            next_stage = next_v_tiles % NUM_STAGES

            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]

            thr_gSrc = thr_copy_load.partition_S(gSrc_next)
            thr_sData = thr_copy_load.partition_D(sData_next)

            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Step 3: Compute using data from current stage
        for row in range(0, TILE_V, 4):
            row_offset = tidx // 32

            v_idx = v_tiles * TILE_V + row + row_offset
            v_row = cute.arch.shuffle_sync(r_v[v_idx // 32], v_idx % 32, mask=-1, mask_and_clamp=31)

            sum_hq = 0.0
            for i in range(vec_size):
                r_h[i] = sData[(row + row_offset, i * 32 + lane_id, stage)]
                r_h[i] = r_h[i] * r_g
                r_h[i] += r_k[i] * v_row
                gDst[(0, row + row_offset, i * 32 + lane_id, v_tiles)] = r_h[i]
                sum_hq += r_h[i] * r_q[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset, mask=-1, mask_and_clamp=31)

            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # ===================================================================
    # Final writeback: Copy output from shared memory to global memory
    # All threads write (V=128, NUM_THREADS=128)
    # ===================================================================
    cute.arch.barrier()  # Ensure all writes to sOutput are complete

    if tidx < V:
        o[(i_n, i_hv, tidx)] = sOutput[tidx]


@cute.jit
def run_la_decode_kernel_big_batch_pretranspose(
    h0_source: cute.Tensor,  # [B*H, V, K]
    decay_scales: cute.Tensor,  # [H]
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softmax_scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    # h0_source: (B*HV, V, K)
    batch_size, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),  # 4 rows, 32 threads/row
        stride=(32, 1),
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)

    vec_size = TILE_K // 32  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    # print(f"Batched CP.ASYNC Load + Store (bypass L1 cache)")
    # print(f"  {batch_size} batches x {v_dim}x{k_dim} matrices")
    # print(f"  Tile: {TILE_V}x{TILE_K}, {num_v_tiles} tiles/batch")
    # print(f"  Threads: {NUM_THREADS} ({NUM_THREADS // 32} warps), vec_size: {vec_size}")
    # print(f"  Total: {total_data_mb:.1f} MB\n")

    # Create SMEM layout
    smem_layout_staged = cute.make_layout((TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K))

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 2 * v_dim + 32

    la_decode_kernel_big_batch_pretranspose(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        softmax_scale,
        B,
        T,
        H,
        K,
        V,
    ).launch(
        grid=(batch_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def run_la_decode_kernel_small_batch_pretranspose(
    h0_source: cute.Tensor,  # [B*H, V, K]
    decay_scales: cute.Tensor,  # [H]
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softmax_scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    # h0_source: (B*H, V, K)
    batch_size, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),  # 4 rows, 32 threads/row
        stride=(32, 1),
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)

    vec_size = TILE_K // 32  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    # Create SMEM layout
    smem_layout_staged = cute.make_layout((TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K))

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sOutput: TILE_V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 2 * v_dim + 32

    la_decode_kernel_small_batch_pretranspose(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        decay_scales,
        q,
        k,
        v,
        o,
        h0_indices,
        softmax_scale,
        B,
        T,
        H,
        K,
        V,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _get_compiled_kernel(B: int, T: int, H: int, K: int, V: int, softmax_scale: float, use_fast_math: bool = True):
    """Get or create compiled kernel cache."""
    return {}


def linear_attention_decode(
    q: torch.Tensor,  # [B, 1, H, HEAD_DIM], same as [B, 1, H, K]
    k: torch.Tensor,  # [B, 1, H, HEAD_DIM], same as [B, 1, H, K]
    v: torch.Tensor,  # [B, 1, H, HEAD_DIM], same as [B, 1, H, V]
    s: torch.Tensor,  # [pool_size, heads, V, K]
    out: torch.Tensor,  # [B, 1, H, HEAD_DIM]
    softmax_scale: float,
    stride_q: int,
    stride_k: int,
    stride_v: int,
    stride_s: int,
    stride_o: int,
    s_offsets: torch.Tensor,  # [B] - state pool indices
    decay_scales: torch.Tensor,  # [H]
    HEAD_DIM: int,
    K_SPLIT_DIM: int,
    V_SPLIT_DIM: int,
) -> None:
    """
    Linear Attention Decode using CuTe DSL.
    Compatible with Triton seg_la_d_kernel interface.

    Args:
        q: Query tensor [B, H, HEAD_DIM]
        k: Key tensor [B, H, HEAD_DIM]
        v: Value tensor [B, H, HEAD_DIM]
        s: State pool tensor [pool_size, heads, K*V]
        out: Output tensor [k_dim_block, length, heads, HEAD_DIM]
        softmax_scale: Softmax scale factor
        stride_q: Stride of q tensor
        stride_k: Stride of k tensor
        stride_v: Stride of v tensor
        stride_s: Stride of s tensor
        stride_o: Stride of out tensor
        s_offsets: State pool indices [B]
        decay_scales: Decay scales per head [H]
        HEAD_DIM: Head dimension
        K_SPLIT_DIM: K split dimension (must be HEAD_DIM for no split)
        V_SPLIT_DIM: V split dimension (must be HEAD_DIM for no split)

    Returns:
        None (modifies out and s in-place)
    """
    B = q.shape[0]
    H = q.shape[1]

    k_dim_block = HEAD_DIM // K_SPLIT_DIM
    if k_dim_block > 1:
        raise NotImplementedError(f"CuTe kernel doesn't support K splitting (k_dim_block={k_dim_block})")

    # Get compiled kernel (cached)
    cache_key = (B, 1, H, HEAD_DIM, HEAD_DIM, softmax_scale, USE_FAST_MATH)
    cache = _get_compiled_kernel(*cache_key)

    h0_source = s
    # First-time compilation
    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        if B <= 32:
            run_func = run_la_decode_kernel_small_batch_pretranspose
        else:
            run_func = run_la_decode_kernel_big_batch_pretranspose

        # Create views for compilation
        q_view = q
        k_view = k
        v_view = v
        o_view = out

        # Use s_offsets directly (pass to kernel but not actually used in current implementation)
        h0_indices = s_offsets

        # Convert to CuTe format for compilation
        h0_tensor = from_dlpack(h0_source, assumed_align=16)
        decay_tensor = from_dlpack(decay_scales, assumed_align=16)
        q_tensor = from_dlpack(q_view, assumed_align=16)
        k_tensor = from_dlpack(k_view, assumed_align=16)
        v_tensor = from_dlpack(v_view, assumed_align=16)
        o_tensor = from_dlpack(o_view, assumed_align=16)
        h0_idx_tensor = from_dlpack(h0_indices, assumed_align=16)

        compiled = cute.compile(
            run_func,
            h0_tensor,
            decay_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            h0_idx_tensor,
            softmax_scale=softmax_scale,
            H=H,
            B=B,
            T=1,
            K=HEAD_DIM,
            V=HEAD_DIM,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled(h0_source, decay_scales, q, k, v, out, s_offsets, stream)


def seg_la_d_kernel_cute(
    q: torch.Tensor,  # [B, 1, heads, HEAD_DIM]
    k: torch.Tensor,  # [B, 1, heads, HEAD_DIM]
    v: torch.Tensor,  # [B, 1, heads, HEAD_DIM]
    s: torch.Tensor,  # [pool_size, heads, K*V]
    out: torch.Tensor,  # [B, 1, heads, HEAD_DIM]
    softmax_scale: float,
    stride_q: int,
    stride_k: int,
    stride_v: int,
    stride_s: int,
    stride_o: int,
    s_offsets: torch.Tensor,  # [B] - state pool indices
    decay_scales: torch.Tensor,  # [H]
    HEAD_DIM: int,
    K_SPLIT_DIM: int,
    V_SPLIT_DIM: int,
) -> None:
    """
    CuTe wrapper function compatible with Triton seg_la_d_kernel interface.
    """
    # Call the main implementation
    linear_attention_decode(
        q,
        k,
        v,
        s,
        out,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        stride_s,
        stride_o,
        s_offsets,
        decay_scales,
        HEAD_DIM,
        K_SPLIT_DIM,
        V_SPLIT_DIM,
    )
