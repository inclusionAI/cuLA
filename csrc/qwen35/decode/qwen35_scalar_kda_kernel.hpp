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

namespace cula::qwen35::decode::kernel {

using namespace cute;

template <int kBytes = 16>
CUTE_DEVICE void cp_async_ca_shared_global(void* smem_ptr, const void* gmem_ptr) {
  static_assert(kBytes == 16, "Only 16-byte cp.async copies are supported here.");
  const unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
}

CUTE_DEVICE void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

CUTE_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
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

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads>
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
  constexpr int kPipeTileK = 16;
  constexpr int kPipeStages = 2;
  constexpr int kVecFloats = 4;
  constexpr int kStatePipeStrideV = kHeadDimV + 4;

  static_assert(kLocalQKHeads == Shape::kLocalQKHeads);
  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);
  static_assert(kHeadDimV % kWarpTileV == 0);
  static_assert(kWarps * kWarpTileV == kHeadDimV);
  static_assert(kHeadDimQK % kPipeTileK == 0);

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

  q_smem[tid] = q_smem[tid] * norm_smem[0];
  k_smem[tid] = k_smem[tid] * norm_smem[1];
  __syncthreads();

  float qk_dot = q_smem[tid] * k_smem[tid];
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

  float proj_row = 0.f;
  float out_old_row = 0.f;
  int pipe_stage = 0;
  load_state_pipe_tile(pipe_stage, 0);
  cp_async_commit_group();

#pragma unroll 1
  for (int k_base = 0; k_base < kHeadDimQK; k_base += kPipeTileK) {
    cp_async_wait_all();
    __syncthreads();

    const int next_k_base = k_base + kPipeTileK;
    const int next_stage = pipe_stage ^ 1;
    if (next_k_base < kHeadDimQK) {
      load_state_pipe_tile(next_stage, next_k_base);
      cp_async_commit_group();
    }

#pragma unroll
    for (int k_local = 0; k_local < kPipeTileK; k_local += 4) {
      const float state0 = state_pipe[pipe_stage][k_local + 0][v_row];
      const float state1 = state_pipe[pipe_stage][k_local + 1][v_row];
      const float state2 = state_pipe[pipe_stage][k_local + 2][v_row];
      const float state3 = state_pipe[pipe_stage][k_local + 3][v_row];
      proj_row += state0 * k_smem[k_base + k_local + 0];
      proj_row += state1 * k_smem[k_base + k_local + 1];
      proj_row += state2 * k_smem[k_base + k_local + 2];
      proj_row += state3 * k_smem[k_base + k_local + 3];
      out_old_row += state0 * q_smem[k_base + k_local + 0];
      out_old_row += state1 * q_smem[k_base + k_local + 1];
      out_old_row += state2 * q_smem[k_base + k_local + 2];
      out_old_row += state3 * q_smem[k_base + k_local + 3];
    }
    __syncthreads();
    pipe_stage = next_stage;
  }

  const float v_val = static_cast<float>(v_vec(v_row));
  const float v_new_row = beta * (v_val - decay * proj_row);
  out_vec(v_row) = static_cast<scalar_t>(decay * out_old_row + v_new_row * norm_smem[2]);

  pipe_stage = 0;
  load_state_pipe_tile(pipe_stage, 0);
  cp_async_commit_group();

#pragma unroll 1
  for (int k_base = 0; k_base < kHeadDimQK; k_base += kPipeTileK) {
    cp_async_wait_all();
    __syncthreads();

    const int next_k_base = k_base + kPipeTileK;
    const int next_stage = pipe_stage ^ 1;
    if (next_k_base < kHeadDimQK) {
      load_state_pipe_tile(next_stage, next_k_base);
      cp_async_commit_group();
    }

#pragma unroll
    for (int k_local = 0; k_local < kPipeTileK; k_local += 4) {
      const float state_new0 = decay * state_pipe[pipe_stage][k_local + 0][v_row] + v_new_row * k_smem[k_base + k_local + 0];
      const float state_new1 = decay * state_pipe[pipe_stage][k_local + 1][v_row] + v_new_row * k_smem[k_base + k_local + 1];
      const float state_new2 = decay * state_pipe[pipe_stage][k_local + 2][v_row] + v_new_row * k_smem[k_base + k_local + 2];
      const float state_new3 = decay * state_pipe[pipe_stage][k_local + 3][v_row] + v_new_row * k_smem[k_base + k_local + 3];
      state_vk(v_row, k_base + k_local + 0) = state_new0;
      state_vk(v_row, k_base + k_local + 1) = state_new1;
      state_vk(v_row, k_base + k_local + 2) = state_new2;
      state_vk(v_row, k_base + k_local + 3) = state_new3;
    }
    __syncthreads();
    pipe_stage = next_stage;
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
  dim3 grid(kLocalVHeads, token_count, 1);
  dim3 block(128, 1, 1);
  qwen35_layout_scalar_kda_decode_long_kernel<scalar_t, kLocalQKHeads, kLocalVHeads><<<grid, block, 0, stream>>>(
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
