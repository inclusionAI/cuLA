// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "qwen35_decode_common.cuh"
#include "qwen35_scalar_kda_mainloop.hpp"

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace cula::qwen35::decode::kernel {

using namespace cute;

template <int kBytes = 16>
CUTE_DEVICE void cp_async_ca_shared_global(void* smem_ptr, const void* gmem_ptr) {
  static_assert(kBytes == 16, "Only 16-byte cp.async copies are supported here.");
  const unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
}

CUTE_DEVICE void cp_async_bulk_shared_global(
    void* smem_ptr,
    const void* gmem_ptr,
    uint32_t bytes,
    cutlass::arch::ClusterTransactionBarrier::ValueType* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  const uint32_t barrier_addr = cute::cast_smem_ptr_to_uint(barrier);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n"
      :
      : "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(barrier_addr)
      : "memory");
#else
  (void)smem_ptr;
  (void)gmem_ptr;
  (void)bytes;
  (void)barrier;
#endif
}

CUTE_DEVICE void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

CUTE_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

CUTE_DEVICE void cp_async_wait_group_1() {
  asm volatile("cp.async.wait_group 1;\n" ::);
}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads>
struct Qwen35ScalarKdaDecodeKernel {
  using Shape = cula::qwen35::decode::Qwen35DecodeLocalShape<kLocalVHeads>;
  static_assert(kLocalQKHeads == Shape::kLocalQKHeads);
  // Decode-first design:
  // - 1 CTA owns 1 (token_idx, hv)
  // - 1 warpgroup (128 threads) per CTA
  // - recurrent state stays fp32 and is traversed as 16x16 tiles over the
  //   internal [V, K] view
  // - the intended optimized path is fp32 FFMA on CUDA cores, not a forced
  //   Tensor Core lowering
  static constexpr int kThreads = Shape::kKdaThreads;
  static constexpr int kWarpGroupThreads = Shape::kKdaThreads;
  static constexpr int kTileV = Shape::kKdaTileV;
  static constexpr int kTileK = Shape::kKdaTileK;
  static constexpr int kTilesPerV = kHeadDimV / kTileV;
  static constexpr int kTilesPerK = kHeadDimQK / kTileK;

  static_assert(kLocalQKHeads < kLocalVHeads);
  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);
  static_assert(kThreads == kWarpGroupThreads);
  static_assert(kHeadDimV % kTileV == 0);
  static_assert(kHeadDimQK % kTileK == 0);

  struct SharedStorage {
    // Shared staging plan for the fp32 decode path:
    // - q/k/v are staged once per CTA
    // - proj/out intermediates remain in fp32
    // - recurrent state itself remains in fp32 global storage
    alignas(16) float q_smem[kHeadDimQK];
    alignas(16) float k_smem[kHeadDimQK];
    alignas(16) scalar_t v_smem[kHeadDimV];
    alignas(16) float norm_smem[2];
    alignas(16) float proj_smem[kHeadDimV];
    alignas(16) float out_smem[kHeadDimV];
  };

  static dim3 block_shape() {
    return dim3(kThreads, 1, 1);
  }

  static dim3 grid_shape(int token_count) {
    // One block owns one (token_idx, hv) pair in the first implementation.
    return dim3(static_cast<unsigned int>(Shape::kLocalVHeads), static_cast<unsigned int>(token_count), 1);
  }

  template <typename Mainloop>
  CUTE_DEVICE static void run_device(
      const scalar_t* __restrict__ q_rep,
      const scalar_t* __restrict__ k_rep,
      const scalar_t* __restrict__ v,
      const scalar_t* __restrict__ a_kernel,
      const scalar_t* __restrict__ b_kernel,
      const float* __restrict__ A_log,
      const float* __restrict__ dt_bias,
      float* __restrict__ recurrent_state,
      const int32_t* __restrict__ pool_idx,
      scalar_t* __restrict__ out,
      int token_count,
      SharedStorage& storage) {
    const int hv = static_cast<int>(blockIdx.x);
    const int token_idx = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (token_idx >= token_count || hv >= kLocalVHeads) {
      return;
    }

    // Internal tensor-view contract fixed for the first implementation pass:
    //
    // 1. q_rep / k_rep / v / out stay in their external contiguous layouts:
    //    - q_rep : [N, HV, K] with stride (HV*K, K, 1)
    //    - k_rep : [N, HV, K] with stride (HV*K, K, 1)
    //    - v     : [N, HV, V] with stride (HV*V, V, 1)
    //    - out   : [N, HV, V] with stride (HV*V, V, 1)
    //
    // 2. a_kernel / b_kernel are treated as:
    //    - [N, HV] with stride (HV, 1)
    //
    // 3. A_log / dt_bias are treated as:
    //    - [HV] with stride (1)
    //
    // 4. recurrent_state keeps the external physical storage contract:
    //    - [pool, HV, K, V]
    //    but the kernel's main computation will use an internal VK view:
    //    - [pool, HV, V, K]
    //
    // This lets the recurrent update consume one V-row of state against q/k
    // more naturally in the first mainloop design, while preserving the
    // existing external state ABI.
    //
    // The current block owns exactly one (token_idx, hv) pair. That means one
    // warpgroup-sized CTA updates one 128x128 recurrent-state tile for one
    // v-head.
    //
    // TODO(qwen35-scalar-kda-opt):
    // - Likely next optimization path: keep one CTA per (token_idx, hv), but
    //   tile the 128x128 state more aggressively inside the block (for example
    //   along V tiles or KxV subtiles assigned per warp).
    // - More complex alternative: split one (token_idx, hv) tile across
    //   multiple CTAs and coordinate updates. Not a first-pass target.
    // - After the fp32 decode path is stable, evaluate warp specialization:
    //   dedicated producer/load warp(s) vs consumer/compute warp(s), instead
    //   of introducing that complexity before the math path itself is stable.

    auto q_layout = make_layout(
        make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimQK>{}),
        make_stride(kLocalVHeads * kHeadDimQK, kHeadDimQK, Int<1>{}));
    auto v_layout = make_layout(
        make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
        make_stride(kLocalVHeads * kHeadDimV, kHeadDimV, Int<1>{}));
    auto head_layout = make_layout(
        make_shape(token_count, Int<kLocalVHeads>{}),
        make_stride(kLocalVHeads, Int<1>{}));
    auto hv_layout = make_layout(make_shape(Int<kLocalVHeads>{}), make_stride(Int<1>{}));
    auto state_layout_kv = make_layout(
        make_shape(_, Int<kLocalVHeads>{}, Int<kHeadDimQK>{}, Int<kHeadDimV>{}),
        make_stride(Int<kLocalVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, kHeadDimV, Int<1>{}));
    auto state_layout_vk = make_layout(
        make_shape(_, Int<kLocalVHeads>{}, Int<kHeadDimV>{}, Int<kHeadDimQK>{}),
        make_stride(Int<kLocalVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, Int<1>{}, kHeadDimV));

    auto gQ = make_tensor(make_gmem_ptr(q_rep), q_layout);
    auto gK = make_tensor(make_gmem_ptr(k_rep), q_layout);
    auto gV = make_tensor(make_gmem_ptr(v), v_layout);
    auto gO = make_tensor(make_gmem_ptr(out), v_layout);
    auto gA = make_tensor(make_gmem_ptr(a_kernel), head_layout);
    auto gB = make_tensor(make_gmem_ptr(b_kernel), head_layout);
    auto gAlog = make_tensor(make_gmem_ptr(A_log), hv_layout);
    auto gDt = make_tensor(make_gmem_ptr(dt_bias), hv_layout);
    auto gH_kv = make_tensor(make_gmem_ptr(recurrent_state), state_layout_kv);
    auto gH_vk = make_tensor(make_gmem_ptr(recurrent_state), state_layout_vk);
    (void)gH_kv; // Keep the physical KV view documented and available.

    const int state_row = pool_idx[token_idx];
    if (state_row < 0) {
      return;
    }

    auto q_vec = gQ(token_idx, hv, _);
    auto k_vec = gK(token_idx, hv, _);
    auto v_vec = gV(token_idx, hv, _);
    auto out_vec = gO(token_idx, hv, _);
    auto a_scalar = gA(token_idx, hv);
    auto b_scalar = gB(token_idx, hv);
    auto A_log_scalar = gAlog(hv);
    auto dt_bias_scalar = gDt(hv);
    auto state_vk = gH_vk(state_row, hv, _, _);

    Mainloop::run(
        q_vec,
        k_vec,
        v_vec,
        a_scalar,
        b_scalar,
        A_log_scalar,
        dt_bias_scalar,
        state_vk,
        out_vec,
        storage,
        tid,
        kThreads);
  }

  template <typename Mainloop>
  CUTE_DEVICE static void run_layout_device(
      const scalar_t* __restrict__ mixed_qkv_conv,
      const scalar_t* __restrict__ a,
      const scalar_t* __restrict__ b,
      const float* __restrict__ A_log,
      const float* __restrict__ dt_bias,
      float* __restrict__ recurrent_state,
      const int32_t* __restrict__ pool_idx,
      scalar_t* __restrict__ out,
      int token_count,
      SharedStorage& storage) {
    constexpr int kRepeatFactor = Shape::kRepeatFactor;
    constexpr int kLocalQDim = Shape::kLocalQDim;
    constexpr int kLocalKDim = Shape::kLocalKDim;
    constexpr int kLocalMixedQKVDim = Shape::kLocalMixedQKVDim;

    const int hv = static_cast<int>(blockIdx.x);
    const int token_idx = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (token_idx >= token_count || hv >= kLocalVHeads) {
      return;
    }

    const int mapped_h = hv / kRepeatFactor;

    auto qk_src_layout = make_layout(
        make_shape(token_count, Int<kLocalQKHeads>{}, Int<kHeadDimQK>{}),
        make_stride(kLocalMixedQKVDim, kHeadDimQK, Int<1>{}));
    auto v_src_layout = make_layout(
        make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
        make_stride(kLocalMixedQKVDim, kHeadDimV, Int<1>{}));
    auto out_layout = make_layout(
        make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
        make_stride(kLocalVHeads * kHeadDimV, kHeadDimV, Int<1>{}));
    auto head_layout = make_layout(
        make_shape(token_count, Int<kLocalVHeads>{}),
        make_stride(kLocalVHeads, Int<1>{}));
    auto hv_layout = make_layout(make_shape(Int<kLocalVHeads>{}), make_stride(Int<1>{}));
    auto state_layout_kv = make_layout(
        make_shape(_, Int<kLocalVHeads>{}, Int<kHeadDimQK>{}, Int<kHeadDimV>{}),
        make_stride(Int<kLocalVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, kHeadDimV, Int<1>{}));
    auto state_layout_vk = make_layout(
        make_shape(_, Int<kLocalVHeads>{}, Int<kHeadDimV>{}, Int<kHeadDimQK>{}),
        make_stride(Int<kLocalVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, Int<1>{}, kHeadDimV));

    const scalar_t* q_src = mixed_qkv_conv;
    const scalar_t* k_src = mixed_qkv_conv + kLocalQDim;
    const scalar_t* v_src = mixed_qkv_conv + kLocalQDim + kLocalKDim;

    auto gQ = make_tensor(make_gmem_ptr(q_src), qk_src_layout);
    auto gK = make_tensor(make_gmem_ptr(k_src), qk_src_layout);
    auto gV = make_tensor(make_gmem_ptr(v_src), v_src_layout);
    auto gO = make_tensor(make_gmem_ptr(out), out_layout);
    auto gA = make_tensor(make_gmem_ptr(a), head_layout);
    auto gB = make_tensor(make_gmem_ptr(b), head_layout);
    auto gAlog = make_tensor(make_gmem_ptr(A_log), hv_layout);
    auto gDt = make_tensor(make_gmem_ptr(dt_bias), hv_layout);
    auto gH_kv = make_tensor(make_gmem_ptr(recurrent_state), state_layout_kv);
    auto gH_vk = make_tensor(make_gmem_ptr(recurrent_state), state_layout_vk);
    (void)gH_kv;

    const int state_row = pool_idx[token_idx];
    if (state_row < 0) {
      return;
    }

    auto q_vec = gQ(token_idx, mapped_h, _);
    auto k_vec = gK(token_idx, mapped_h, _);
    auto v_vec = gV(token_idx, hv, _);
    auto out_vec = gO(token_idx, hv, _);
    auto a_scalar = gA(token_idx, hv);
    auto b_scalar = gB(token_idx, hv);
    auto A_log_scalar = gAlog(hv);
    auto dt_bias_scalar = gDt(hv);
    auto state_vk = gH_vk(state_row, hv, _, _);

    Mainloop::run(
        q_vec,
        k_vec,
        v_vec,
        a_scalar,
        b_scalar,
        A_log_scalar,
        dt_bias_scalar,
        state_vk,
        out_vec,
        storage,
        tid,
        kThreads);
  }
};

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads, typename Mainloop = Qwen35ScalarKdaDecodeMainloop<scalar_t>>
__global__ void qwen35_scalar_kda_decode_kernel(
    const scalar_t* __restrict__ q_rep,
    const scalar_t* __restrict__ k_rep,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ a_kernel,
    const scalar_t* __restrict__ b_kernel,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ recurrent_state,
    const int32_t* __restrict__ pool_idx,
    scalar_t* __restrict__ out,
    int token_count) {
  __shared__ typename Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::SharedStorage storage;
  Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::template run_device<Mainloop>(
      q_rep,
      k_rep,
      v,
      a_kernel,
      b_kernel,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count,
      storage);
}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads, typename Mainloop = Qwen35ScalarKdaDecodeMainloop<scalar_t>>
void launch_qwen35_scalar_kda_decode_kernel(
    cudaStream_t stream,
    const scalar_t* q_rep,
    const scalar_t* k_rep,
    const scalar_t* v,
    const scalar_t* a_kernel,
    const scalar_t* b_kernel,
    const float* A_log,
    const float* dt_bias,
    float* recurrent_state,
    const int32_t* pool_idx,
    scalar_t* out,
    int token_count) {
  if (token_count >= 64) {
    auto grid = Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::grid_shape(token_count);
    auto block = Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::block_shape();
    qwen35_scalar_kda_decode_kernel<
        scalar_t,
        kLocalQKHeads,
        kLocalVHeads,
        Qwen35ScalarKdaDecodeLongMainloop<scalar_t>><<<grid, block, 0, stream>>>(
        q_rep,
        k_rep,
        v,
        a_kernel,
        b_kernel,
        A_log,
        dt_bias,
        recurrent_state,
        pool_idx,
        out,
        token_count);
    return;
  }

  auto grid = Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::grid_shape(token_count);
  auto block = Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::block_shape();
  qwen35_scalar_kda_decode_kernel<scalar_t, kLocalQKHeads, kLocalVHeads, Mainloop><<<grid, block, 0, stream>>>(
      q_rep,
      k_rep,
      v,
      a_kernel,
      b_kernel,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count);
}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads, typename Mainloop = Qwen35ScalarKdaDecodeMainloop<scalar_t>>
__global__ void qwen35_layout_scalar_kda_decode_kernel(
    const scalar_t* __restrict__ mixed_qkv_conv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ recurrent_state,
    const int32_t* __restrict__ pool_idx,
    scalar_t* __restrict__ out,
    int token_count) {
  __shared__ typename Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::SharedStorage storage;
  Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::template run_layout_device<Mainloop>(
      mixed_qkv_conv,
      a,
      b,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count,
      storage);
}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads, int kPipeTileK_ = 16>
__global__ void qwen35_layout_scalar_kda_decode_long_kernel(
    const scalar_t* __restrict__ mixed_qkv_conv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ recurrent_state,
    const int32_t* __restrict__ pool_idx,
    scalar_t* __restrict__ out,
    int token_count) {
  using Shape = cula::qwen35::decode::Qwen35DecodeLocalShape<kLocalVHeads>;
  constexpr int kRepeatFactor = Shape::kRepeatFactor;
  constexpr int kLocalQDim = Shape::kLocalQDim;
  constexpr int kLocalKDim = Shape::kLocalKDim;
  constexpr int kLocalMixedQKVDim = Shape::kLocalMixedQKVDim;
  constexpr int kWarpTileV = 32;
  constexpr int kWarpSize = 32;
  constexpr int kThreads = 128;
  constexpr int kWarps = kThreads / kWarpSize;
  constexpr int kPipeTileK = kPipeTileK_;
  constexpr int kPipeStages = 2;
  constexpr int kVecFloats = 4;
  constexpr int kStatePipeStrideV = kHeadDimV + 4;

  static_assert(kLocalQKHeads == Shape::kLocalQKHeads);
  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);
  static_assert(kHeadDimV % kWarpTileV == 0);
  static_assert(kWarps * kWarpTileV == kHeadDimV);
  static_assert(kHeadDimQK % kPipeTileK == 0);
  static_assert(kPipeTileK == 16 || kPipeTileK == 32);

  __shared__ float q_smem[kHeadDimQK];
  __shared__ float k_smem[kHeadDimQK];
  __shared__ float norm_smem[2 * kWarps];
  __shared__ float state_pipe[kPipeStages][kPipeTileK][kStatePipeStrideV];

  const int hv = static_cast<int>(blockIdx.x);
  const int token_idx = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  const int mapped_h = hv / kRepeatFactor;
  const int v_row = warp_id * kWarpTileV + lane;

  auto qk_src_layout = make_layout(
      make_shape(token_count, Int<kLocalQKHeads>{}, Int<kHeadDimQK>{}),
      make_stride(kLocalMixedQKVDim, kHeadDimQK, Int<1>{}));
  auto v_src_layout = make_layout(
      make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
      make_stride(kLocalMixedQKVDim, kHeadDimV, Int<1>{}));
  auto out_layout = make_layout(
      make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
      make_stride(kLocalVHeads * kHeadDimV, kHeadDimV, Int<1>{}));
  auto head_layout = make_layout(
      make_shape(token_count, Int<kLocalVHeads>{}),
      make_stride(kLocalVHeads, Int<1>{}));
  auto hv_layout = make_layout(make_shape(Int<kLocalVHeads>{}), make_stride(Int<1>{}));
  auto state_layout_vk = make_layout(
      make_shape(_, Int<kLocalVHeads>{}, Int<kHeadDimV>{}, Int<kHeadDimQK>{}),
      make_stride(Int<kLocalVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, Int<1>{}, kHeadDimV));

  const scalar_t* q_src = mixed_qkv_conv;
  const scalar_t* k_src = mixed_qkv_conv + kLocalQDim;
  const scalar_t* v_src = mixed_qkv_conv + kLocalQDim + kLocalKDim;

  auto gQ = make_tensor(make_gmem_ptr(q_src), qk_src_layout);
  auto gK = make_tensor(make_gmem_ptr(k_src), qk_src_layout);
  auto gV = make_tensor(make_gmem_ptr(v_src), v_src_layout);
  auto gO = make_tensor(make_gmem_ptr(out), out_layout);
  auto gA = make_tensor(make_gmem_ptr(a), head_layout);
  auto gB = make_tensor(make_gmem_ptr(b), head_layout);
  auto gAlog = make_tensor(make_gmem_ptr(A_log), hv_layout);
  auto gDt = make_tensor(make_gmem_ptr(dt_bias), hv_layout);
  auto gH_vk = make_tensor(make_gmem_ptr(recurrent_state), state_layout_vk);

  const int state_row = pool_idx[token_idx];
  if (state_row < 0) {
    return;
  }

  auto q_vec = gQ(token_idx, mapped_h, _);
  auto k_vec = gK(token_idx, mapped_h, _);
  auto v_vec = gV(token_idx, hv, _);
  auto out_vec = gO(token_idx, hv, _);
  auto state_vk = gH_vk(state_row, hv, _, _);

  const float a_val = static_cast<float>(gA(token_idx, hv));
  const float b_val = static_cast<float>(gB(token_idx, hv));
  const float g = -expf(static_cast<float>(gAlog(hv))) *
      Qwen35ScalarKdaDecodeMainloop<scalar_t>::softplusf_approx(a_val + static_cast<float>(gDt(hv)));
  const float decay = expf(g);
  const float beta = 1.f / (1.f + expf(-b_val));

  const float q_raw = static_cast<float>(q_vec(tid));
  const float k_raw = static_cast<float>(k_vec(tid));
  q_smem[tid] = q_raw;
  k_smem[tid] = k_raw;

  float q_norm_sq = q_raw * q_raw;
  float k_norm_sq = k_raw * k_raw;
  q_norm_sq = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(q_norm_sq);
  k_norm_sq = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(k_norm_sq);
  if (lane == 0) {
    norm_smem[warp_id] = q_norm_sq;
    norm_smem[kWarps + warp_id] = k_norm_sq;
  }
  __syncthreads();

  float q_block_sum = lane < kWarps ? norm_smem[lane] : 0.f;
  float k_block_sum = lane < kWarps ? norm_smem[kWarps + lane] : 0.f;
  if (warp_id == 0) {
    q_block_sum = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(q_block_sum);
    k_block_sum = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(k_block_sum);
    if (lane == 0) {
      norm_smem[0] = rsqrtf(q_block_sum + 1e-6f) * rsqrtf(static_cast<float>(kHeadDimQK));
      norm_smem[1] = rsqrtf(k_block_sum + 1e-6f);
    }
  }
  __syncthreads();

  const float q_normed = q_raw * norm_smem[0];
  const float k_normed = k_raw * norm_smem[1];
  q_smem[tid] = q_normed;
  k_smem[tid] = k_normed;

  float qk_dot = q_normed * k_normed;
  qk_dot = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(qk_dot);
  if (lane == 0) {
    norm_smem[warp_id] = qk_dot;
  }
  __syncthreads();
  float qk_block_sum = lane < kWarps ? norm_smem[lane] : 0.f;
  if (warp_id == 0) {
    qk_block_sum = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(qk_block_sum);
    if (lane == 0) {
      norm_smem[2] = qk_block_sum;
    }
  }
  __syncthreads();

  auto load_state_pipe_tile = [&](int stage, int k_base) {
#pragma unroll 1
    for (int elem = tid * kVecFloats; elem < kPipeTileK * kHeadDimV; elem += kThreads * kVecFloats) {
      const int k_local = elem / kHeadDimV;
      const int v_base = elem - k_local * kHeadDimV;
      cp_async_ca_shared_global<16>(
          &state_pipe[stage][k_local][v_base],
          &state_vk(v_base, k_base + k_local));
    }
  };

  float proj_acc0 = 0.f;
  float proj_acc1 = 0.f;
  float proj_acc2 = 0.f;
  float proj_acc3 = 0.f;
  float out_acc0 = 0.f;
  float out_acc1 = 0.f;
  float out_acc2 = 0.f;
  float out_acc3 = 0.f;
  int pipe_stage = 0;
  load_state_pipe_tile(0, 0);
  cp_async_commit_group();
  if (kPipeTileK < kHeadDimQK) {
    load_state_pipe_tile(1, kPipeTileK);
    cp_async_commit_group();
  }

#pragma unroll 1
  for (int k_base = 0; k_base < kHeadDimQK; k_base += kPipeTileK) {
    const int next_k_base = k_base + kPipeTileK;
    const int next_stage = pipe_stage ^ 1;
    const int prefetch_k_base = k_base + 2 * kPipeTileK;
    if (next_k_base < kHeadDimQK) {
      cp_async_wait_group_1();
    } else {
      cp_async_wait_all();
    }
    __syncthreads();

    float q_regs[kPipeTileK];
    float k_regs[kPipeTileK];
#pragma unroll
    for (int kk = 0; kk < kPipeTileK; ++kk) {
      q_regs[kk] = q_smem[k_base + kk];
      k_regs[kk] = k_smem[k_base + kk];
    }

#pragma unroll
    for (int k_local = 0; k_local < kPipeTileK; k_local += 4) {
      const float state0 = state_pipe[pipe_stage][k_local + 0][v_row];
      const float state1 = state_pipe[pipe_stage][k_local + 1][v_row];
      const float state2 = state_pipe[pipe_stage][k_local + 2][v_row];
      const float state3 = state_pipe[pipe_stage][k_local + 3][v_row];
      proj_acc0 += state0 * k_regs[k_local + 0];
      proj_acc1 += state1 * k_regs[k_local + 1];
      proj_acc2 += state2 * k_regs[k_local + 2];
      proj_acc3 += state3 * k_regs[k_local + 3];
      out_acc0 += state0 * q_regs[k_local + 0];
      out_acc1 += state1 * q_regs[k_local + 1];
      out_acc2 += state2 * q_regs[k_local + 2];
      out_acc3 += state3 * q_regs[k_local + 3];
    }
    if (prefetch_k_base < kHeadDimQK) {
      __syncthreads();
      load_state_pipe_tile(pipe_stage, prefetch_k_base);
      cp_async_commit_group();
    }
    pipe_stage = next_stage;
  }

  const float proj_row = (proj_acc0 + proj_acc1) + (proj_acc2 + proj_acc3);
  const float out_old_row = (out_acc0 + out_acc1) + (out_acc2 + out_acc3);
  const float v_val = static_cast<float>(v_vec(v_row));
  const float v_new_row = beta * (v_val - decay * proj_row);
  out_vec(v_row) = static_cast<scalar_t>(decay * out_old_row + v_new_row * norm_smem[2]);

  pipe_stage = 0;
  load_state_pipe_tile(0, 0);
  cp_async_commit_group();
  if (kPipeTileK < kHeadDimQK) {
    load_state_pipe_tile(1, kPipeTileK);
    cp_async_commit_group();
  }

#pragma unroll 1
  for (int k_base = 0; k_base < kHeadDimQK; k_base += kPipeTileK) {
    const int next_k_base = k_base + kPipeTileK;
    const int next_stage = pipe_stage ^ 1;
    const int prefetch_k_base = k_base + 2 * kPipeTileK;
    if (next_k_base < kHeadDimQK) {
      cp_async_wait_group_1();
    } else {
      cp_async_wait_all();
    }
    __syncthreads();

    float k_regs[kPipeTileK];
#pragma unroll
    for (int kk = 0; kk < kPipeTileK; ++kk) {
      k_regs[kk] = k_smem[k_base + kk];
    }

#pragma unroll
    for (int k_local = 0; k_local < kPipeTileK; k_local += 4) {
      const float state_new0 = decay * state_pipe[pipe_stage][k_local + 0][v_row] + v_new_row * k_regs[k_local + 0];
      const float state_new1 = decay * state_pipe[pipe_stage][k_local + 1][v_row] + v_new_row * k_regs[k_local + 1];
      const float state_new2 = decay * state_pipe[pipe_stage][k_local + 2][v_row] + v_new_row * k_regs[k_local + 2];
      const float state_new3 = decay * state_pipe[pipe_stage][k_local + 3][v_row] + v_new_row * k_regs[k_local + 3];
      state_vk(v_row, k_base + k_local + 0) = state_new0;
      state_vk(v_row, k_base + k_local + 1) = state_new1;
      state_vk(v_row, k_base + k_local + 2) = state_new2;
      state_vk(v_row, k_base + k_local + 3) = state_new3;
    }
    if (prefetch_k_base < kHeadDimQK) {
      __syncthreads();
      load_state_pipe_tile(pipe_stage, prefetch_k_base);
      cp_async_commit_group();
    }
    pipe_stage = next_stage;
  }

}


template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads, int kTileV>
__global__ void qwen35_layout_scalar_kda_decode_long_vtile_kernel(
    const scalar_t* __restrict__ mixed_qkv_conv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ recurrent_state,
    const int32_t* __restrict__ pool_idx,
    scalar_t* __restrict__ out,
    int token_count) {
  using Shape = cula::qwen35::decode::Qwen35DecodeLocalShape<kLocalVHeads>;
  constexpr int kRepeatFactor = Shape::kRepeatFactor;
  constexpr int kLocalQDim = Shape::kLocalQDim;
  constexpr int kLocalKDim = Shape::kLocalKDim;
  constexpr int kLocalMixedQKVDim = Shape::kLocalMixedQKVDim;
  constexpr int kWarpSize = 32;
  constexpr int kThreads = kTileV;
  constexpr int kWarps = kThreads / kWarpSize;
  constexpr int kVTiles = kHeadDimV / kTileV;
  constexpr int kKPerThread = kHeadDimQK / kThreads;

  static_assert(kLocalQKHeads == Shape::kLocalQKHeads);
  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);
  static_assert(kTileV == 32 || kTileV == 64);
  static_assert(kHeadDimV % kTileV == 0);
  static_assert(kHeadDimQK % kThreads == 0);

  __shared__ float q_smem[kHeadDimQK];
  __shared__ float k_smem[kHeadDimQK];
  __shared__ float state_smem[kHeadDimQK][kTileV];
  __shared__ float norm_smem[3];
  __shared__ float warp_reduce_smem[2 * kWarps];
  __shared__ cutlass::arch::ClusterTransactionBarrier::ValueType state_barrier;

  const int hv_tile = static_cast<int>(blockIdx.x);
  const int token_idx = static_cast<int>(blockIdx.y);
  const int hv = hv_tile / kVTiles;
  const int v_tile = hv_tile - hv * kVTiles;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  const int mapped_h = hv / kRepeatFactor;
  const int v_row = v_tile * kTileV + tid;

  auto qk_src_layout = make_layout(
      make_shape(token_count, Int<kLocalQKHeads>{}, Int<kHeadDimQK>{}),
      make_stride(kLocalMixedQKVDim, kHeadDimQK, Int<1>{}));
  auto v_src_layout = make_layout(
      make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
      make_stride(kLocalMixedQKVDim, kHeadDimV, Int<1>{}));
  auto out_layout = make_layout(
      make_shape(token_count, Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
      make_stride(kLocalVHeads * kHeadDimV, kHeadDimV, Int<1>{}));
  auto head_layout = make_layout(
      make_shape(token_count, Int<kLocalVHeads>{}),
      make_stride(kLocalVHeads, Int<1>{}));
  auto hv_layout = make_layout(make_shape(Int<kLocalVHeads>{}), make_stride(Int<1>{}));
  auto state_layout_vk = make_layout(
      make_shape(_, Int<kLocalVHeads>{}, Int<kHeadDimV>{}, Int<kHeadDimQK>{}),
      make_stride(Int<kLocalVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, Int<1>{}, kHeadDimV));

  const scalar_t* q_src = mixed_qkv_conv;
  const scalar_t* k_src = mixed_qkv_conv + kLocalQDim;
  const scalar_t* v_src = mixed_qkv_conv + kLocalQDim + kLocalKDim;

  auto gQ = make_tensor(make_gmem_ptr(q_src), qk_src_layout);
  auto gK = make_tensor(make_gmem_ptr(k_src), qk_src_layout);
  auto gV = make_tensor(make_gmem_ptr(v_src), v_src_layout);
  auto gO = make_tensor(make_gmem_ptr(out), out_layout);
  auto gA = make_tensor(make_gmem_ptr(a), head_layout);
  auto gB = make_tensor(make_gmem_ptr(b), head_layout);
  auto gAlog = make_tensor(make_gmem_ptr(A_log), hv_layout);
  auto gDt = make_tensor(make_gmem_ptr(dt_bias), hv_layout);
  auto gH_vk = make_tensor(make_gmem_ptr(recurrent_state), state_layout_vk);

  const int state_row = pool_idx[token_idx];
  if (state_row < 0) {
    return;
  }

  auto q_vec = gQ(token_idx, mapped_h, _);
  auto k_vec = gK(token_idx, mapped_h, _);
  auto v_vec = gV(token_idx, hv, _);
  auto out_vec = gO(token_idx, hv, _);
  auto state_vk = gH_vk(state_row, hv, _, _);

  float q_norm_sq = 0.f;
  float k_norm_sq = 0.f;
#pragma unroll
  for (int i = 0; i < kKPerThread; ++i) {
    const int k_idx = i * kThreads + tid;
    const float q_raw = static_cast<float>(q_vec(k_idx));
    const float k_raw = static_cast<float>(k_vec(k_idx));
    q_smem[k_idx] = q_raw;
    k_smem[k_idx] = k_raw;
    q_norm_sq += q_raw * q_raw;
    k_norm_sq += k_raw * k_raw;
  }
  q_norm_sq = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(q_norm_sq);
  k_norm_sq = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(k_norm_sq);
  if (lane == 0) {
    warp_reduce_smem[warp_id] = q_norm_sq;
    warp_reduce_smem[kWarps + warp_id] = k_norm_sq;
  }
  __syncthreads();
  if (warp_id == 0) {
    float q_block_sum = lane < kWarps ? warp_reduce_smem[lane] : 0.f;
    float k_block_sum = lane < kWarps ? warp_reduce_smem[kWarps + lane] : 0.f;
    q_block_sum = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(q_block_sum);
    k_block_sum = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(k_block_sum);
    if (lane == 0) {
      norm_smem[0] = rsqrtf(q_block_sum + 1e-6f) * rsqrtf(static_cast<float>(kHeadDimQK));
      norm_smem[1] = rsqrtf(k_block_sum + 1e-6f);
    }
  }
  __syncthreads();

  float qk_dot = 0.f;
#pragma unroll
  for (int i = 0; i < kKPerThread; ++i) {
    const int k_idx = i * kThreads + tid;
    const float q_normed = q_smem[k_idx] * norm_smem[0];
    const float k_normed = k_smem[k_idx] * norm_smem[1];
    q_smem[k_idx] = q_normed;
    k_smem[k_idx] = k_normed;
    qk_dot += q_normed * k_normed;
  }
  qk_dot = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(qk_dot);
  if (lane == 0) {
    warp_reduce_smem[warp_id] = qk_dot;
  }
  __syncthreads();
  if (warp_id == 0) {
    float qk_block_sum = lane < kWarps ? warp_reduce_smem[lane] : 0.f;
    qk_block_sum = Qwen35ScalarKdaDecodeMainloop<scalar_t>::warp_sum(qk_block_sum);
    if (lane == 0) {
      norm_smem[2] = qk_block_sum;
    }
  }
  __syncthreads();

  const float a_val = static_cast<float>(gA(token_idx, hv));
  const float b_val = static_cast<float>(gB(token_idx, hv));
  const float g = -expf(static_cast<float>(gAlog(hv))) *
      Qwen35ScalarKdaDecodeMainloop<scalar_t>::softplusf_approx(a_val + static_cast<float>(gDt(hv)));
  const float decay = expf(g);
  const float beta = 1.f / (1.f + expf(-b_val));

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (tid == 0) {
    cutlass::arch::ClusterTransactionBarrier::init(&state_barrier, 1);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        &state_barrier, kHeadDimQK * kTileV * sizeof(float));
  }
  __syncthreads();
#pragma unroll 1
  for (int k_idx = tid; k_idx < kHeadDimQK; k_idx += kThreads) {
    cp_async_bulk_shared_global(
        &state_smem[k_idx][0],
        &state_vk(v_tile * kTileV, k_idx),
        kTileV * sizeof(float),
        &state_barrier);
  }
  cutlass::arch::ClusterTransactionBarrier::wait(&state_barrier, 0);
  __syncthreads();
#else
#pragma unroll 1
  for (int elem = tid * 4; elem < kHeadDimQK * kTileV; elem += kThreads * 4) {
    const int k_idx = elem / kTileV;
    const int v_base = elem - k_idx * kTileV;
    cp_async_ca_shared_global<16>(
        &state_smem[k_idx][v_base],
        &state_vk(v_tile * kTileV + v_base, k_idx));
  }
  cp_async_commit_group();
  cp_async_wait_all();
  __syncthreads();
#endif

  float proj_acc0 = 0.f;
  float proj_acc1 = 0.f;
  float proj_acc2 = 0.f;
  float proj_acc3 = 0.f;
  float proj_acc4 = 0.f;
  float proj_acc5 = 0.f;
  float proj_acc6 = 0.f;
  float proj_acc7 = 0.f;
  float out_acc0 = 0.f;
  float out_acc1 = 0.f;
  float out_acc2 = 0.f;
  float out_acc3 = 0.f;
  float out_acc4 = 0.f;
  float out_acc5 = 0.f;
  float out_acc6 = 0.f;
  float out_acc7 = 0.f;
#pragma unroll 1
  for (int k_idx = 0; k_idx < kHeadDimQK; k_idx += 8) {
    const float state0 = state_smem[k_idx + 0][tid];
    const float state1 = state_smem[k_idx + 1][tid];
    const float state2 = state_smem[k_idx + 2][tid];
    const float state3 = state_smem[k_idx + 3][tid];
    const float state4 = state_smem[k_idx + 4][tid];
    const float state5 = state_smem[k_idx + 5][tid];
    const float state6 = state_smem[k_idx + 6][tid];
    const float state7 = state_smem[k_idx + 7][tid];
    const float k0 = k_smem[k_idx + 0];
    const float k1 = k_smem[k_idx + 1];
    const float k2 = k_smem[k_idx + 2];
    const float k3 = k_smem[k_idx + 3];
    const float k4 = k_smem[k_idx + 4];
    const float k5 = k_smem[k_idx + 5];
    const float k6 = k_smem[k_idx + 6];
    const float k7 = k_smem[k_idx + 7];
    const float q0 = q_smem[k_idx + 0];
    const float q1 = q_smem[k_idx + 1];
    const float q2 = q_smem[k_idx + 2];
    const float q3 = q_smem[k_idx + 3];
    const float q4 = q_smem[k_idx + 4];
    const float q5 = q_smem[k_idx + 5];
    const float q6 = q_smem[k_idx + 6];
    const float q7 = q_smem[k_idx + 7];
    proj_acc0 += state0 * k0;
    proj_acc1 += state1 * k1;
    proj_acc2 += state2 * k2;
    proj_acc3 += state3 * k3;
    proj_acc4 += state4 * k4;
    proj_acc5 += state5 * k5;
    proj_acc6 += state6 * k6;
    proj_acc7 += state7 * k7;
    out_acc0 += state0 * q0;
    out_acc1 += state1 * q1;
    out_acc2 += state2 * q2;
    out_acc3 += state3 * q3;
    out_acc4 += state4 * q4;
    out_acc5 += state5 * q5;
    out_acc6 += state6 * q6;
    out_acc7 += state7 * q7;
  }

  const float proj_row =
      ((proj_acc0 + proj_acc1) + (proj_acc2 + proj_acc3)) +
      ((proj_acc4 + proj_acc5) + (proj_acc6 + proj_acc7));
  const float out_old_row =
      ((out_acc0 + out_acc1) + (out_acc2 + out_acc3)) +
      ((out_acc4 + out_acc5) + (out_acc6 + out_acc7));
  const float v_val = static_cast<float>(v_vec(v_row));
  const float v_new_row = beta * (v_val - decay * proj_row);
  out_vec(v_row) = static_cast<scalar_t>(decay * out_old_row + v_new_row * norm_smem[2]);

#pragma unroll 1
  for (int k_idx = 0; k_idx < kHeadDimQK; k_idx += 8) {
    const float state0 = state_smem[k_idx + 0][tid];
    const float state1 = state_smem[k_idx + 1][tid];
    const float state2 = state_smem[k_idx + 2][tid];
    const float state3 = state_smem[k_idx + 3][tid];
    const float state4 = state_smem[k_idx + 4][tid];
    const float state5 = state_smem[k_idx + 5][tid];
    const float state6 = state_smem[k_idx + 6][tid];
    const float state7 = state_smem[k_idx + 7][tid];
    const float state_new0 = decay * state0 + v_new_row * k_smem[k_idx + 0];
    const float state_new1 = decay * state1 + v_new_row * k_smem[k_idx + 1];
    const float state_new2 = decay * state2 + v_new_row * k_smem[k_idx + 2];
    const float state_new3 = decay * state3 + v_new_row * k_smem[k_idx + 3];
    const float state_new4 = decay * state4 + v_new_row * k_smem[k_idx + 4];
    const float state_new5 = decay * state5 + v_new_row * k_smem[k_idx + 5];
    const float state_new6 = decay * state6 + v_new_row * k_smem[k_idx + 6];
    const float state_new7 = decay * state7 + v_new_row * k_smem[k_idx + 7];
    state_vk(v_row, k_idx + 0) = state_new0;
    state_vk(v_row, k_idx + 1) = state_new1;
    state_vk(v_row, k_idx + 2) = state_new2;
    state_vk(v_row, k_idx + 3) = state_new3;
    state_vk(v_row, k_idx + 4) = state_new4;
    state_vk(v_row, k_idx + 5) = state_new5;
    state_vk(v_row, k_idx + 6) = state_new6;
    state_vk(v_row, k_idx + 7) = state_new7;
  }

}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads>
void launch_qwen35_layout_scalar_kda_decode_long_kernel(
    cudaStream_t stream,
    const scalar_t* mixed_qkv_conv,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    float* recurrent_state,
    const int32_t* pool_idx,
    scalar_t* out,
    int token_count) {
  constexpr int kWarpTileV = 32;
  (void)kWarpTileV;
  if (token_count == 64 || token_count == 128) {
    constexpr int kLongTileV = 64;
    dim3 grid(kLocalVHeads * (kHeadDimV / kLongTileV), token_count, 1);
    dim3 block(kLongTileV, 1, 1);
    qwen35_layout_scalar_kda_decode_long_vtile_kernel<scalar_t, kLocalQKHeads, kLocalVHeads, kLongTileV>
        <<<grid, block, 0, stream>>>(
        mixed_qkv_conv,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        pool_idx,
        out,
        token_count);
    return;
  }

  dim3 grid(kLocalVHeads, token_count, 1);
  dim3 block(128, 1, 1);
  qwen35_layout_scalar_kda_decode_long_kernel<scalar_t, kLocalQKHeads, kLocalVHeads, 16><<<grid, block, 0, stream>>>(
      mixed_qkv_conv,
      a,
      b,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count);
}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads, typename Mainloop = Qwen35ScalarKdaDecodeMainloop<scalar_t>>
void launch_qwen35_layout_scalar_kda_decode_kernel(
    cudaStream_t stream,
    const scalar_t* mixed_qkv_conv,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    float* recurrent_state,
    const int32_t* pool_idx,
    scalar_t* out,
    int token_count) {
  if (token_count >= 64) {
    launch_qwen35_layout_scalar_kda_decode_long_kernel<scalar_t, kLocalQKHeads, kLocalVHeads>(
        stream,
        mixed_qkv_conv,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        pool_idx,
        out,
        token_count);
    return;
  }

  auto grid = Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::grid_shape(token_count);
  auto block = Qwen35ScalarKdaDecodeKernel<scalar_t, kLocalQKHeads, kLocalVHeads>::block_shape();
  qwen35_layout_scalar_kda_decode_kernel<scalar_t, kLocalQKHeads, kLocalVHeads, Mainloop><<<grid, block, 0, stream>>>(
      mixed_qkv_conv,
      a,
      b,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count);
}

} // namespace cula::qwen35::decode::kernel
