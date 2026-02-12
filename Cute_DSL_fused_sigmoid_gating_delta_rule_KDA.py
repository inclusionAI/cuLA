# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Batched TMA Pipeline Load - Each Warp Processes One Row

This file implements a fused CUDA kernel using CUTLASS CuTe DSL for executing
Sigmoid Gating Delta Rule updates. Key features:

Architecture Design:
- Uses TMA (Tensor Memory Accelerator) for efficient Global Memory → Shared Memory transfers
- Employs 4-stage pipeline to overlap loading and computation, hiding memory latency
- Each block uses 128 threads (4 warps), with each warp processing one matrix row
- Tile size: 8x128 (TILE_V x TILE_K)

Computation Flow:
1. Warp 0 handles TMA prefetch, loading data from GMEM to SMEM
2. All warps compute in parallel: softplus, L2 normalization, delta rule updates
3. Each warp processes one row of data, completing h_new = g*h + k*(beta*(v - h@k))
4. Uses warp-level shuffle for efficient reduction operations
5. Results are vectorized and written back to Global Memory

Performance Optimizations:
- Vectorized memory access: each thread processes vec_size=4 elements (128-bit aligned)
- Warp-level collective operations: shuffle-based reduction
- Pipeline overlap: load stage N+1 while computing stage N
- L2 cache management: flush L2 before benchmarking to measure true bandwidth
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import argparse


# Global configuration
TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128  # 4 warps
NUM_BLOCKS_PER_STATE = 8

@cute.kernel
def cpasync_pipeline_kernel_small_batch(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,      # [HV]
    a: cute.Tensor,          # [B, T, HV, K] - KDA: per-K dimension gate input
    dt_bias: cute.Tensor,    # [HV, K] - KDA: per-K dimension bias
    q: cute.Tensor,          # [B, T, H, K]
    k: cute.Tensor,          # [B, T, H, K]
    v: cute.Tensor,          # [B, T, HV, V]
    b: cute.Tensor,          # [B, T, HV]
    o: cute.Tensor,          # [B, T, HV, V] - output
    h0_indices: cute.Tensor, # [B] - initial state indices
    cu_seqlens: cute.Tensor, # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // NUM_BLOCKS_PER_STATE
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # Allocate shared memory for v values (size V, to reduce register usage)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)
    # Allocate shared memory for a, dt_bias, and g values using tensors
    # KDA: g is per-K dimension (K=128), matching SGLang's IS_KDA behavior
    sA_layout = cute.make_layout((K,), stride=(1,))
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sDtBias = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)


    r_k = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)),
            cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)),
        cutlass.Float32
    )
    # r_v moved to shared memory (sV)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)),
        cutlass.Float32
    )

    # ===================================================================
    # Load a and dt_bias to shared memory using all 128 threads
    # ===================================================================
    sA[(tidx,)] = cutlass.Float32(a[i_n, i_t, i_hv, tidx])
    sDtBias[(tidx,)] = cutlass.Float32(dt_bias[i_hv, tidx])

    cute.arch.barrier()

    # Get current batch
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K)
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # V 方向分 tiles
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
        r_q[i] = cutlass.Float32(q[i_n, i_t, i_h, i * 32 + in_warp_tid])
        r_k[i] = cutlass.Float32(k[i_n, i_t, i_h, i * 32 + in_warp_tid])
        # Store v to shared memory instead of register
        v_val = cutlass.Float32(v[i_n, i_t, i_hv, i * 32 + in_warp_tid])
        sV[i * 32 + in_warp_tid] = v_val

    cute.arch.barrier()  # Ensure all threads finish writing to sV

    # ===================================================================
    # Compute beta (scalar value, per HV) and g (vector, per HV)
    # ===================================================================
    r_beta = 0.0
    if in_warp_tid == 0:
        # Compute beta = 1 / (1 + exp(-b))
        r_beta = 1.0 / (1.0 + cute.exp(-r_b))

    # Each thread computes one g value
    x = sA[(tidx,)] + sDtBias[(tidx,)]
    beta_x = softplus_beta * x
    softplus_x = 0.0

    if beta_x <= softplus_threshold:
        # softplus(x) = (1/beta) * log(1 + exp(beta*x))
        exp_beta_x = cute.exp(beta_x)
        log_input = cutlass.Float32(1.0 + exp_beta_x)
        log_result = cutlass.Float32(cute.log(log_input))
        softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
    else:
        softplus_x = x

    # Compute g = exp(- exp(A_log) * softplus_x)
    r_g_value = - cute.exp(r_A_log) * softplus_x
    sG[(tidx,)] = cute.exp(r_g_value)

    cute.arch.barrier()

    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    if use_qk_l2norm:
        # Compute L2 norm of q and k
        sum_q = 0.0
        sum_k = 0.0
        for i in range(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        # Warp-level reduction using butterfly shuffle
        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
            sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)

        norm_q = cute.sqrt(sum_q + 1e-6)
        norm_k = cute.sqrt(sum_k + 1e-6)
        for i in range(vec_size):
            r_q[i] = r_q[i] / norm_q
            r_k[i] = r_k[i] / norm_k

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
            # All 128 threads in parallel
            row_offset = tidx // 32

            # ===================================================================
            # Load pre-computed gate g from shared memory (per-K dimension)
            # Each thread processes vec_size K elements, load corresponding g values
            # SGLang KDA: b_h *= exp(b_gk[:, None]) - g is per-K, broadcast to V
            # ===================================================================
            for i in range(vec_size):  # Process 4 rows at once
                r_h[i] = sData[(row + row_offset, i * 32 + in_warp_tid, stage)]  # SMEM → Register
                # Load g for this K position and apply decay
                r_g_k = sG[(i * 32 + in_warp_tid,)]
                r_h[i] = r_h[i] * r_g_k

            sum_hk = 0.0
            for i in range(vec_size):
                sum_hk += r_h[i] * r_k[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(sum_hk, offset=offset, mask=-1, mask_and_clamp=31)

            v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk  # Fixed: added row_offset
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in range(vec_size):
                r_h[i] += r_k[i] * v_new
                gDst[(0, row + row_offset, i * 32 + in_warp_tid, v_tiles)] = r_h[i]
                sum_hq += r_h[i] * r_q[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset, mask=-1, mask_and_clamp=31)

            o_idx = v_tiles * TILE_V + row + row_offset
            if in_warp_tid == 0:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # ===================================================================
    # Final writeback: Copy output from shared memory to global memory
    # All threads write (V=128, NUM_THREADS=128)
    # ===================================================================
    cute.arch.barrier()  # Ensure all writes to sOutput are complete
    if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
        o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]


@cute.kernel
def cpasync_pipeline_kernel_big_batch(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,      # [HV]
    a: cute.Tensor,          # [B, T, HV, K] - KDA: per-K dimension gate input
    dt_bias: cute.Tensor,    # [HV, K] - KDA: per-K dimension bias
    q: cute.Tensor,          # [B, T, H, K]
    k: cute.Tensor,          # [B, T, H, K]
    v: cute.Tensor,          # [B, T, HV, V]
    b: cute.Tensor,          # [B, T, HV]
    o: cute.Tensor,          # [B, T, HV, V] - output
    h0_indices: cute.Tensor, # [B] - initial state indices
    cu_seqlens: cute.Tensor, # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    batch_idx, _, _ = cute.arch.block_idx()
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # Allocate shared memory for v values (size V, to reduce register usage)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    # Allocate shared memory for a, dt_bias, and g values using tensors
    # KDA: g is per-K dimension (K=128), matching SGLang's IS_KDA behavior
    sA_layout = cute.make_layout((K,), stride=(1,))
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sDtBias = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, sA_layout, 16)


    r_k = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)),
            cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)),
        cutlass.Float32
    )
    # r_v moved to shared memory (sV)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)),
        cutlass.Float32
    )

    # ===================================================================
    # Load a and dt_bias to shared memory using all 128 threads
    # ===================================================================
    sA[(tidx,)] = cutlass.Float32(a[i_n, i_t, i_hv, tidx])
    sDtBias[(tidx,)] = cutlass.Float32(dt_bias[i_hv, tidx])


    cute.arch.barrier()

    # Get current batch
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K)
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # V 方向分 tiles
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
        r_q[i] = cutlass.Float32(q[i_n, i_t, i_h, i * 32 + in_warp_tid])
        r_k[i] = cutlass.Float32(k[i_n, i_t, i_h, i * 32 + in_warp_tid])
        # Store v to shared memory instead of in_warp_tid
        v_val = cutlass.Float32(v[i_n, i_t, i_hv, i * 32 + in_warp_tid])
        sV[i * 32 + in_warp_tid] = v_val

    cute.arch.barrier()  # Ensure all threads finish writing to sV

    # ===================================================================
    # Compute beta (scalar value, per HV) and g (vector, per HV)
    # ===================================================================
    r_beta = 0.0
    if in_warp_tid == 0:
        # Compute beta = 1 / (1 + exp(-b))
        r_beta = 1.0 / (1.0 + cute.exp(-r_b))

    # Each thread computes one g value
    x = sA[(tidx,)] + sDtBias[(tidx,)]
    beta_x = softplus_beta * x
    softplus_x = 0.0

    if beta_x <= softplus_threshold:
        # softplus(x) = (1/beta) * log(1 + exp(beta*x))
        exp_beta_x = cute.exp(beta_x)
        log_input = cutlass.Float32(1.0 + exp_beta_x)
        log_result = cutlass.Float32(cute.log(log_input))
        softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
    else:
        softplus_x = x

    # Compute g = exp(- exp(A_log) * softplus_x)
    r_g_value = - cute.exp(r_A_log) * softplus_x
    sG[(tidx,)] = cute.exp(r_g_value)

    cute.arch.barrier()

    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    if use_qk_l2norm:
        # Compute L2 norm of q and k
        sum_q = 0.0
        sum_k = 0.0
        for i in range(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        # Warp-level reduction using butterfly shuffle
        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
            sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)

        norm_q = cute.sqrt(sum_q + 1e-6)
        norm_k = cute.sqrt(sum_k + 1e-6)
        for i in range(vec_size):
            r_q[i] = r_q[i] / norm_q
            r_k[i] = r_k[i] / norm_k

    # Apply scaling in Float32
    for i in range(vec_size):
        r_q[i] = r_q[i] * scale

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
            # All 128 threads in parallel
            row_offset = tidx // 32

            # ===================================================================
            # Load pre-computed gate g from shared memory (per-K dimension)
            # Each thread processes vec_size K elements, load corresponding g values
            # SGLang KDA: b_h *= exp(b_gk[:, None]) - g is per-K, broadcast to V
            # ===================================================================
            for i in range(vec_size):
                r_h[i] = sData[(row + row_offset, i * 32 + in_warp_tid, stage)]
                # Load g for this K position and apply decay
                r_g_k = sG[(i * 32 + in_warp_tid,)]
                r_h[i] = r_h[i] * r_g_k

            sum_hk = 0.0
            for i in range(vec_size):
                sum_hk += r_h[i] * r_k[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(sum_hk, offset=offset, mask=-1, mask_and_clamp=31)

            v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk  # Fixed: added row_offset
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in range(vec_size):
                r_h[i] += r_k[i] * v_new
                gDst[(0, row + row_offset, i * 32 + in_warp_tid, v_tiles)] = r_h[i]
                sum_hq += r_h[i] * r_q[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset, mask=-1, mask_and_clamp=31)

            o_idx = v_tiles * TILE_V + row + row_offset
            if in_warp_tid == 0:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # ===================================================================
    # Final writeback: Copy output from shared memory to global memory
    # All threads write (V=128, NUM_THREADS=128)
    # ===================================================================
    cute.arch.barrier()  # Ensure all writes to sOutput are complete

    o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]


@cute.jit
def run_batched_cpasync_small_batch(
    h0_source: cute.Tensor,  # [B*HV, K, V]
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    # h0_source: (B*HV, V, K)
    batch_size, v_dim, k_dim = h0_source.layout.shape[0], h0_source.layout.shape[1], h0_source.layout.shape[2]

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),       # 4 rows, 32 threads/row
        stride=(32, 1)
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    total_data_mb = v_dim * k_dim * batch_size * 4 / 1024 / 1024

    vec_size = TILE_K // 32  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    print(f"Batched CP.ASYNC Load + Store (bypass L1 cache)")
    print(f"  {batch_size} batches x {v_dim}x{k_dim} matrices")
    print(f"  Tile: {TILE_V}x{TILE_K}, {num_v_tiles} tiles/batch")
    print(f"  Threads: {NUM_THREADS} ({NUM_THREADS // 32} warps), vec_size: {vec_size}")
    print(f"  Total: {total_data_mb:.1f} MB\n")

    # Create SMEM layout
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES),
        stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sV: K * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 2 * v_dim + 32
    cpasync_pipeline_kernel_small_batch(
        tiled_copy_load, h0_source, smem_layout_staged, vec_size, num_v_tiles,
        A_log, a, dt_bias, q, k, v, b, o,
        h0_indices, cu_seqlens,
        softplus_beta, softplus_threshold, scale,
        HV, B, T, H, K, V,
        use_initial_state, use_qk_l2norm, is_varlen
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        stream=stream
    )

@cute.jit
def run_batched_cpasync_big_batch(
    h0_source: cute.Tensor,  # [B*HV, K, V]
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    # h0_source: (B*HV, V, K)
    batch_size, v_dim, k_dim = h0_source.layout.shape[0], h0_source.layout.shape[1], h0_source.layout.shape[2]

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),       # 4 rows, 32 threads/row
        stride=(32, 1)
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    total_data_mb = v_dim * k_dim * batch_size * 4 / 1024 / 1024

    vec_size = TILE_K // 32  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    print(f"Batched CP.ASYNC Load + Store (bypass L1 cache)")
    print(f"  {batch_size} batches x {v_dim}x{k_dim} matrices")
    print(f"  Tile: {TILE_V}x{TILE_K}, {num_v_tiles} tiles/batch")
    print(f"  Threads: {NUM_THREADS} ({NUM_THREADS // 32} warps), vec_size: {vec_size}")
    print(f"  Total: {total_data_mb:.1f} MB\n")

    # Create SMEM layout
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES),
        stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sV: K * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 2 * v_dim + 32
    cpasync_pipeline_kernel_big_batch(
        tiled_copy_load, h0_source, smem_layout_staged, vec_size, num_v_tiles,
        A_log, a, dt_bias, q, k, v, b, o,
        h0_indices, cu_seqlens,
        softplus_beta, softplus_threshold, scale,
        HV, B, T, H, K, V,
        use_initial_state, use_qk_l2norm, is_varlen
    ).launch(
        grid=(batch_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        stream=stream
    )

if __name__ == "__main__":

    warmup_iters = 0
    test_iters = 1

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CuTe DSL Fused Sigmoid Gating Delta Rule Update Kernel')
    parser.add_argument('--B', type=int, default=16, help='Batch size')
    parser.add_argument('--T', type=int, default=1, help='Sequence length')
    parser.add_argument('--H', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--K', type=int, default=128, help='Key/Query dimension')
    parser.add_argument('--V', type=int, default=128, help='Value dimension')
    parser.add_argument('--HV', type=int, default=32, help='Number of value heads')
    args = parser.parse_args()

    print("CuTe DSL Fused Sigmoid Gating Delta Rule Update Kernel")
    print("=" * 60)
    print(f"Config: B={args.B}, T={args.T}, H={args.H}, K={args.K}, V={args.V}, HV={args.HV}")
    print("=" * 60)

    B, T, H, K, V, HV = args.B, args.T, args.H, args.K, args.V, args.HV
    scale = K ** -0.5

    # Verify dimensions are multiples of 4 for 128-bit (4 x float32) vectorized loads
    assert K % 4 == 0, f"K must be multiple of 4 for vectorized loads, got K={K}"
    assert V % 4 == 0, f"V must be multiple of 4 for vectorized loads, got V={V}"
    assert HV % 4 == 0, f"HV must be multiple of 4 for vectorized loads, got HV={HV}"

    # Initialize with sequential values (0, 1, 2, 3...) for easier debugging
    A_log = torch.arange(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.arange(HV * V, dtype=torch.float32, device="cuda").reshape(HV, V)
    a = torch.arange(B * T * HV * V, dtype=torch.float16, device="cuda").reshape(B, T, HV, V)
    b = torch.arange(B * T * HV, dtype=torch.float16, device="cuda").reshape(B, T, HV)
    q = torch.arange(B * T * H * K, dtype=torch.float16, device="cuda").reshape(B, T, H, K)
    k = torch.arange(B * T * H * K, dtype=torch.float16, device="cuda").reshape(B, T, H, K)
    v = torch.arange(B * T * HV * V, dtype=torch.float16, device="cuda").reshape(B, T, HV, V)


    # Create initial state with proper shape [B, HV, K, V]
    initial_state_source = torch.arange(B * HV * K * V, dtype=torch.float32, device="cuda").reshape(B, HV, K, V)
    initial_state_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    # Create data: merge first two dimensions, transpose last two dimensions
    # [B, HV, K, V] -> [B*HV, K, V] -> [B*HV, V, K]
    h0_source = initial_state_source.reshape(B * HV, K, V).transpose(1, 2).contiguous()  # [B*HV, V, K]
    batch = B * HV  # Update batch size

    # Create output tensor
    o = torch.zeros_like(v)

    # Convert to CuTe tensors (element_type is automatically inferred from PyTorch dtype)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    a_tensor = from_dlpack(a, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    q_tensor = from_dlpack(q, assumed_align=16)
    k_tensor = from_dlpack(k, assumed_align=16)
    v_tensor = from_dlpack(v, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    o_tensor = from_dlpack(o, assumed_align=16)
    h0_indices_tensor = from_dlpack(initial_state_indices, assumed_align=16)

    # Create dummy cu_seqlens tensor
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

    # Compile
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    softplus_beta = 1.0
    softplus_threshold = 20.0
    use_initial_state = True
    use_qk_l2norm = True
    is_varlen = False
    
    compiled = cute.compile(
        run_batched_cpasync_small_batch,
        h0_source_tensor,
        A_log_tensor, a_tensor, dt_bias_tensor,
        q_tensor, k_tensor, v_tensor, b_tensor, o_tensor,
        h0_indices_tensor, cu_seqlens_tensor,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV, B=B, T=T, H=H, K=K, V=V,
        use_initial_state=use_initial_state,
        use_qk_l2norm=use_qk_l2norm,
        is_varlen=is_varlen,
        stream=stream,
        options="--keep-cubin"
    )

    # Warmup
    print(f"Warmup: {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        compiled(
            h0_source_tensor,
            A_log_tensor, a_tensor, dt_bias_tensor,
            q_tensor, k_tensor, v_tensor, b_tensor, o_tensor,
            h0_indices_tensor, cu_seqlens_tensor,
            stream
        )
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    print(f"Benchmarking: {test_iters} iterations (cold L2)...")

    # Create CUDA events
    events_start = [torch.cuda.Event(enable_timing=True) for _ in range(test_iters)]
    events_end = [torch.cuda.Event(enable_timing=True) for _ in range(test_iters)]

    # Create large buffer to flush L2
    l2_size_mb = 80
    dummy_size = int(l2_size_mb * 1024 * 1024 / 4)
    dummy_buffer = torch.empty(dummy_size, dtype=torch.float32, device='cuda')

    times = []
    for i in range(test_iters):
        _ = dummy_buffer.sum()
        torch.cuda.synchronize()

        events_start[i].record()
        compiled(
            h0_source_tensor,
            A_log_tensor, a_tensor, dt_bias_tensor,
            q_tensor, k_tensor, v_tensor, b_tensor, o_tensor,
            h0_indices_tensor, cu_seqlens_tensor,
            stream
        )
        events_end[i].record()
        torch.cuda.synchronize()

    torch.cuda.synchronize()

    # Calculate timings
    for i in range(test_iters):
        times.append(events_start[i].elapsed_time(events_end[i]))  # ms

    # Statistics
    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()
    min_time = times.min().item()

    # Calculate bandwidth: TMA load (read) + register store (write)
    data_per_iter_mb = V * K * batch * 4 / 1024 / 1024  # Data per iteration
    total_data_mb = data_per_iter_mb * 2  # Read + Write

    mean_bw = total_data_mb / (mean_time / 1000) / 1024  # GB/s
    peak_bw = total_data_mb / (min_time / 1000) / 1024   # GB/s

    print("=" * 70)
    print(f"✓ Performance Results:")
    print(f"  Mean time: {mean_time:.3f} ms (±{std_time:.3f} ms)")
    print(f"  Min time:  {min_time:.3f} ms")
    print(f"  Data per iter: {data_per_iter_mb:.1f} MB (read) + {data_per_iter_mb:.1f} MB (write)")
    print(f"  Mean bandwidth: {mean_bw:.1f} GB/s")
    print(f"  Peak bandwidth: {peak_bw:.1f} GB/s")
    print("=" * 70)

    # Print output tensor 'o' (4D: B, T, HV, V)
    print(f"\nOutput shape: {o.shape}")
    print("=" * 60)

    # Print o[0, 0, 0, :] (only batch=0, T=0, HV=0)
    print("\n=== Output Tensor 'o' Values (batch=0, T=0, HV=0) ===")
    B_out, T_out, HV_out, V_out = o.shape
    print(f"o shape: {o.shape}  (B={B_out}, T={T_out}, HV={HV_out}, V={V_out}) NUM_BLOCKS_PER_STATE={NUM_BLOCKS_PER_STATE}")
    print(f"\nPrinting o[0, 0, 0, :] - V dimension ({V_out} values):\n")

    for i_v in range(V_out):  # Print first 32 values
        print(f"  o[0,0,0,{i_v:3d}] = {o[0, 0, 0, i_v].item():10.2f}", end="")
        if (i_v + 1) % 8 == 0:  # Print 8 values per line
            print()
    if min(V_out, 32) % 8 != 0:
        print()  # New line if last row wasn't complete

    print("=" * 60)
    print("\n=== Output Tensor 'o' Values (batch=0, T=0, HV=1) ===")
    B_out, T_out, HV_out, V_out = o.shape
    print(f"o shape: {o.shape}  (B={B_out}, T={T_out}, HV={HV_out}, V={V_out})")
    print(f"\nPrinting o[0, 0, 1, :] - V dimension ({V_out} values):\n")

    for i_v in range(V_out):  # Print first 32 values
        print(f"  o[0,0,1,{i_v:3d}] = {o[0, 0, 1, i_v].item():10.2f}", end="")
        if (i_v + 1) % 8 == 0:  # Print 8 values per line
            print()
    if min(V_out, 32) % 8 != 0:
        print()  # New line if last row wasn't complete

    print("=" * 60)

    # Print initial_state_source (updated H state) [0, 0, 0, :]
    result = h0_source.transpose(1, 2).reshape(B, HV, K, V).cpu()
    print("\n=== Initial State Source 'result' (batch=0, HV=0, K=0) ===")
    print(f"result shape: {result.shape}  (B={result.shape[0]}, HV={result.shape[1]}, K={result.shape[2]}, V={result.shape[3]})")
    print(f"\nPrinting result[0, 0, 0, :] - V dimension ({result.shape[3]} values):\n")

    for i_v in range(result.shape[3]):
        print(f"  h[0,0,0,{i_v:3d}] = {result[0, 0, 0, i_v].item():10.2f}", end="")
        if (i_v + 1) % 8 == 0:  # Print 8 values per line
            print()
    if min(result.shape[3], 32) % 8 != 0:
        print()  # New line if last row wasn't complete

    print("=" * 60)
    print("\n=== Initial State Source 'result' (batch=0, HV=0, K=1) ===")
    print(f"result shape: {result.shape}  (B={result.shape[0]}, HV={result.shape[1]}, K={result.shape[2]}, V={result.shape[3]})")
    print(f"\nPrinting result[0, 0, 1, :] - V dimension ({result.shape[3]} values):\n")

    for i_v in range(result.shape[3]):
        print(f"  h[0,0,1,{i_v:3d}] = {result[0, 0, 1, i_v].item():10.2f}", end="")
        if (i_v + 1) % 8 == 0:  # Print 8 values per line
            print()
    print("=" * 60)
    print("Execution completed!")

    del compiled, h0_source_tensor

