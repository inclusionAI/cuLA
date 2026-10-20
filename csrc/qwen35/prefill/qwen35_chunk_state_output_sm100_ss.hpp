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

// Standalone Blackwell SS-UMMA prototype for the state/output portion of the
// Qwen3.5 scalar prefill path.  Keeping this header independent makes it
// possible to compile and inspect the replacement without changing the
// production WMMA kernel or its launcher.

#if defined(CULA_SM100_ENABLED)

#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include "kerutils/kerutils.cuh"

namespace cula::qwen35::prefill::kernel::sm100_ss {

using namespace cute;

using bf16 = kerutils::bf16;

static constexpr int kHeadDim = 128;
static constexpr int kChunk = 64;
static constexpr int kValueTile = 64;
static constexpr int kThreads = 128;
// Columns 0..63 hold the two 64-row output accumulators (lower/upper DP
// halves); columns 64..127 hold the independent M128 state update.
static constexpr int kTmemColumns = 128;
static constexpr uint32_t kOutputUpperDp = 16u * 65536u;
static constexpr uint32_t kStateUpdateColumn = 64u;

// UMMA sees both operands as logical matrices.  W/Qg/Aqk/KgT are A
// operands, so K-major is the natural row-major representation.  State and
// Vnew are B operands with logical shape [N, K]; MN-major is required here,
// rather than treating their physical [K, V] input representation as an
// ordinary K-major matrix.
using SmemLayoutA64x128K = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<kChunk>, Int<kHeadDim>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayoutStateB64x128MN = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_MN_SW128_Atom<bf16>{},
        Shape<Int<kValueTile>, Int<kHeadDim>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayoutA64x64K = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<kChunk>, Int<kChunk>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayoutVnewB64x64MN = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_MN_SW128_Atom<bf16>{},
        Shape<Int<kValueTile>, Int<kChunk>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayoutKgT128x64K = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<kHeadDim>, Int<kChunk>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

static_assert(cosize_v<SmemLayoutA64x128K> == kChunk * kHeadDim);
static_assert(cosize_v<SmemLayoutStateB64x128MN> == kValueTile * kHeadDim);
static_assert(cosize_v<SmemLayoutA64x64K> == kChunk * kChunk);
static_assert(cosize_v<SmemLayoutVnewB64x64MN> == kValueTile * kChunk);
static_assert(cosize_v<SmemLayoutKgT128x64K> == kHeadDim * kChunk);

using TiledMma64x64 = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<
        bf16,
        bf16,
        float,
        kChunk,
        kValueTile,
        UMMA::Major::K,
        UMMA::Major::MN>{}));

// State update is Kg^T[M=128,K=64] @ Vnew^T[N=64,K=64].  Keeping M=128 in
// one instruction is important: splitting it into two M64 updates repeats
// the Vnew descriptor traffic and completion synchronization.
using TiledMma128x64 = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS<
        bf16,
        bf16,
        float,
        kHeadDim,
        kValueTile,
        UMMA::Major::K,
        UMMA::Major::MN>{}));

using CompletionPipeline = cutlass::PipelineUmmaAsync<1>;
using CompletionPipelineState = cutlass::PipelineState<CompletionPipeline::Stages>;
using ClusterShape = Shape<_1, _1, _1>;

struct alignas(128) Qwen35ChunkStateOutputSm100SsShared {
  // Persistent state is laid out as B[N=V,K=head_dim].
  alignas(128) bf16 state[kValueTile * kHeadDim];
  // W/Aqk use this K-major A buffer.  It is reused as Aqk after the first
  // dual UMMA pair has completed.
  alignas(128) bf16 operand_a[kHeadDim * kChunk];
  // Qg/Kg^T use a second K-major A buffer.  Keeping the two A operands
  // separate allows each pair of contractions to share one completion wait.
  alignas(128) bf16 operand_a_aux[kHeadDim * kChunk];
  // Vnew is the MN-major B operand for both Aqk and the M128 state update.
  alignas(128) bf16 vnew[kValueTile * kChunk];
  float gate_exp[kChunk];
  alignas(16) uint32_t tmem_base_ptr;
  alignas(16) typename CompletionPipeline::SharedStorage completion;
};

static_assert(
    sizeof(Qwen35ChunkStateOutputSm100SsShared) <= 64 * 1024,
    "SS-UMMA state/output prototype must remain below 64 KiB shared memory");

CUTE_DEVICE void release_ss_mma_result(
    CompletionPipeline& completion,
    CompletionPipelineState& consumer_state) {
  // TMEM loads performed by the epilogue must become visible before the
  // consumer marks the single pipeline stage reusable.
  kerutils::tcgen05_before_thread_sync();
  completion.consumer_release(consumer_state);
  ++consumer_state;
  __syncthreads();
}

template <int kLocalVHeads>
__global__ __launch_bounds__(kThreads, 1) void qwen35_chunk_state_output_sm100_ss_kernel(
    const __nv_bfloat16* __restrict__ q_norm,
    const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ Aqk,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ kg,
    const float* __restrict__ initial_state,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    bool has_initial_state) {
  extern __shared__ char shared_bytes[];
  auto& shared = *reinterpret_cast<Qwen35ChunkStateOutputSm100SsShared*>(shared_bytes);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;

  int work = static_cast<int>(blockIdx.x);
  const int value_tile = work % (kHeadDim / kValueTile);
  work /= (kHeadDim / kValueTile);
  const int hv = work % kLocalVHeads;
  const int seq = work / kLocalVHeads;
  if (seq >= batch_size) {
    return;
  }

  const int qk_h = hv / (kLocalVHeads / qk_heads);
  const int v_base = value_tile * kValueTile;
  const int state_global_base = (seq * kLocalVHeads + hv) * kHeadDim * kHeadDim;

  auto s_state = make_tensor(
      make_smem_ptr(shared.state), SmemLayoutStateB64x128MN{});
  auto s_a_64x128 = make_tensor(
      make_smem_ptr(shared.operand_a), SmemLayoutA64x128K{});
  auto s_a_qg = make_tensor(
      make_smem_ptr(shared.operand_a_aux), SmemLayoutA64x128K{});
  auto s_a_aqk = make_tensor(
      make_smem_ptr(shared.operand_a), SmemLayoutA64x64K{});
  auto s_a_kg = make_tensor(
      make_smem_ptr(shared.operand_a_aux), SmemLayoutKgT128x64K{});
  auto s_a_128x64 = make_tensor(
      make_smem_ptr(shared.operand_a), SmemLayoutKgT128x64K{});
  auto s_vnew = make_tensor(
      make_smem_ptr(shared.vnew), SmemLayoutVnewB64x64MN{});

  for (int index = tid; index < kValueTile * kHeadDim; index += kThreads) {
    const int vv = index / kHeadDim;
    const int kk = index % kHeadDim;
    const float value = has_initial_state
        ? initial_state[state_global_base + kk * kHeadDim + v_base + vv]
        : 0.0f;
    s_state(vv, kk) = bf16(value);
  }

  // One completion pipeline is sufficient because the four contractions are
  // issued as two dual-UMMA pairs.  Warp 0 both issues UMMA and participates
  // in the 128-thread epilogue.
  typename CompletionPipeline::Params completion_params;
  completion_params.producer_arv_count = 1;
  completion_params.consumer_arv_count = kThreads;
  completion_params.initializing_warp = 0;
  completion_params.role = warp == 0
      ? CompletionPipeline::ThreadCategory::ProducerConsumer
      : CompletionPipeline::ThreadCategory::Consumer;
  CompletionPipeline completion(
      shared.completion, completion_params, ClusterShape{});

  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (warp == 0) {
    tmem_allocator.allocate(kTmemColumns, &shared.tmem_base_ptr);
    tmem_allocator.release_allocation_lock();
  }
  __syncthreads();

  TiledMma64x64 mma_64x64;
  TiledMma128x64 mma_128x64;
  auto t_acc_64x64_lower = partition_fragment_C(
      mma_64x64, Shape<Int<kChunk>, Int<kValueTile>>{});
  auto t_acc_64x64_upper = partition_fragment_C(
      mma_64x64, Shape<Int<kChunk>, Int<kValueTile>>{});
  auto t_acc_128x64_state = partition_fragment_C(
      mma_128x64, Shape<Int<kHeadDim>, Int<kValueTile>>{});
  t_acc_64x64_lower.data() = shared.tmem_base_ptr;
  t_acc_64x64_upper.data() = shared.tmem_base_ptr + kOutputUpperDp;
  t_acc_128x64_state.data() =
      shared.tmem_base_ptr + kStateUpdateColumn;

  auto c64 = make_identity_tensor(
      Shape<Int<kChunk>, Int<kValueTile>>{});
  auto c128 = make_identity_tensor(
      Shape<Int<kHeadDim>, Int<kValueTile>>{});
  auto t_c64 = mma_64x64.get_slice(_0{}).partition_C(c64);
  auto t_c128 = mma_128x64.get_slice(_0{}).partition_C(c128);

  CompletionPipelineState producer_state =
      cutlass::make_producer_start_state<CompletionPipeline>();
  CompletionPipelineState consumer_state;

  const auto* q_bf16 = reinterpret_cast<const bf16*>(q_norm);
  const auto* aqk_bf16 = reinterpret_cast<const bf16*>(Aqk);
  const auto* w_bf16 = reinterpret_cast<const bf16*>(w);
  const auto* u_bf16 = reinterpret_cast<const bf16*>(u);
  const auto* kg_bf16 = reinterpret_cast<const bf16*>(kg);
  const int chunk_count = (seq_len + kChunk - 1) / kChunk;
  const float q_scale = rsqrtf(static_cast<float>(kHeadDim));

  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int chunk_start = chunk * kChunk;
    const int valid_rows = min(kChunk, seq_len - chunk_start);

    if (tid < kChunk) {
      if (tid < valid_rows) {
        const int token = seq * seq_len + chunk_start + tid;
        shared.gate_exp[tid] = exp2f(g[token * kLocalVHeads + hv]);
      } else {
        shared.gate_exp[tid] = 0.0f;
      }
    }

    // Pair 1: projection = W[64,128] @ state^T[128,64] in the lower DP
    // half, and output_base = Qg[64,128] @ state^T in the upper DP half.
    // Both A operands are staged before the single UMMA completion signal.
    for (int index = tid; index < kChunk * kHeadDim; index += kThreads) {
      const int row = index / kHeadDim;
      const int kk = index % kHeadDim;
      bf16 value = bf16(0.0f);
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        value = w_bf16[(token * kLocalVHeads + hv) * kHeadDim + kk];
      }
      s_a_64x128(row, kk) = value;
      float q_value = 0.0f;
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        q_value = static_cast<float>(
                      q_bf16[(token * qk_heads + qk_h) * kHeadDim + kk]) *
            shared.gate_exp[row] * q_scale;
      }
      s_a_qg(row, kk) = bf16(q_value);
    }
    __syncthreads();

    if (warp == 0) {
      completion.producer_acquire(producer_state);
      cutlass::arch::fence_view_async_shared();
      kerutils::utcmma_ss(
          mma_64x64,
          s_a_64x128,
          s_state,
          t_acc_64x64_lower,
          true);
      kerutils::tcgen05_after_thread_sync();
      kerutils::utcmma_ss(
          mma_64x64,
          s_a_qg,
          s_state,
          t_acc_64x64_upper,
          true);
      completion.producer_commit(producer_state);
      ++producer_state;
    }
    completion.consumer_wait(consumer_state);
    kerutils::tcgen05_after_thread_sync();

    {
      auto tiled_t2r = make_tmem_copy(
          SM100_TMEM_LOAD_16dp256b8x{}, t_acc_64x64_lower);
      auto thr_t2r = tiled_t2r.get_slice(tid);
      auto t_src = thr_t2r.partition_S(t_acc_64x64_lower);
      auto t_coord = thr_t2r.partition_D(t_c64);
      auto r_acc = make_tensor<float>(shape(t_coord));
      copy(tiled_t2r, t_src, r_acc);
      cutlass::arch::fence_view_async_tmem_load();
      CUTE_UNROLL
      for (int item = 0; item < size(r_acc); ++item) {
        const auto coord = t_coord(item);
        const int row = static_cast<int>(get<0>(coord));
        const int vv = static_cast<int>(get<1>(coord));
        float value = 0.0f;
        if (row < valid_rows) {
          const int token = seq * seq_len + chunk_start + row;
          value = static_cast<float>(
                      u_bf16[(token * kLocalVHeads + hv) * kHeadDim + v_base + vv]) -
              r_acc(item);
        }
        s_vnew(vv, row) = bf16(value);
      }
    }
    cutlass::arch::fence_view_async_shared();
    release_ss_mma_result(completion, consumer_state);

    // Pair 2: output += Aqk[64,64] @ Vnew^T in the upper DP half, while the
    // state update Kg^T[128,64] @ Vnew^T is accumulated in a separate M128
    // TMEM fragment.  The two independent UMMAs share one completion wait.
    for (int index = tid; index < kChunk * kChunk; index += kThreads) {
      const int row = index / kChunk;
      const int col = index % kChunk;
      bf16 value = bf16(0.0f);
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        value = aqk_bf16[(token * kLocalVHeads + hv) * kChunk + col];
      }
      s_a_aqk(row, col) = value;
    }
    for (int index = tid; index < kHeadDim * kChunk; index += kThreads) {
      const int kk = index / kChunk;
      const int row = index % kChunk;
      bf16 value = bf16(0.0f);
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        value = kg_bf16[(token * kLocalVHeads + hv) * kHeadDim + kk];
      }
      s_a_kg(kk, row) = value;
    }
    const int last_token = seq * seq_len + chunk_start + valid_rows - 1;
    const float chunk_decay = exp2f(g[last_token * kLocalVHeads + hv]);
    __syncthreads();

    if (warp == 0) {
      completion.producer_acquire(producer_state);
      cutlass::arch::fence_view_async_shared();
      kerutils::utcmma_ss(
          mma_64x64,
          s_a_aqk,
          s_vnew,
          t_acc_64x64_upper,
          false);
      kerutils::tcgen05_after_thread_sync();
      kerutils::utcmma_ss(
          mma_128x64,
          s_a_kg,
          s_vnew,
          t_acc_128x64_state,
          true);
      completion.producer_commit(producer_state);
      ++producer_state;
    }
    completion.consumer_wait(consumer_state);
    kerutils::tcgen05_after_thread_sync();

    {
      auto tiled_t2r = make_tmem_copy(
          SM100_TMEM_LOAD_16dp256b8x{}, t_acc_64x64_upper);
      auto thr_t2r = tiled_t2r.get_slice(tid);
      auto t_src = thr_t2r.partition_S(t_acc_64x64_upper);
      auto t_coord = thr_t2r.partition_D(t_c64);
      auto r_acc = make_tensor<float>(shape(t_coord));
      copy(tiled_t2r, t_src, r_acc);
      cutlass::arch::fence_view_async_tmem_load();
      CUTE_UNROLL
      for (int item = 0; item < size(r_acc); ++item) {
        const auto coord = t_coord(item);
        const int row = static_cast<int>(get<0>(coord));
        const int vv = static_cast<int>(get<1>(coord));
        if (row < valid_rows) {
          const int token = seq * seq_len + chunk_start + row;
          out[(token * kLocalVHeads + hv) * kHeadDim + v_base + vv] =
              __float2bfloat16_rn(r_acc(item));
        }
      }
    }

    {
      auto tiled_t2r = make_tmem_copy(
          SM100_TMEM_LOAD_32dp32b16x{}, t_acc_128x64_state);
      auto thr_t2r = tiled_t2r.get_slice(tid);
      auto t_src = thr_t2r.partition_S(t_acc_128x64_state);
      auto t_coord = thr_t2r.partition_D(t_c128);
      auto r_acc = make_tensor<float>(shape(t_coord));
      copy(tiled_t2r, t_src, r_acc);
      cutlass::arch::fence_view_async_tmem_load();
      const bool is_last_chunk = chunk + 1 == chunk_count;
      CUTE_UNROLL
      for (int item = 0; item < size(r_acc); ++item) {
        const auto coord = t_coord(item);
        const int kk = static_cast<int>(get<0>(coord));
        const int vv = static_cast<int>(get<1>(coord));
        const float updated =
            r_acc(item) + chunk_decay * static_cast<float>(s_state(vv, kk));
        const bf16 quantized = bf16(updated);
        s_state(vv, kk) = quantized;
        if (is_last_chunk) {
          final_state[state_global_base + kk * kHeadDim + v_base + vv] =
              static_cast<float>(quantized);
        }
      }
    }
    cutlass::arch::fence_view_async_shared();
    release_ss_mma_result(completion, consumer_state);
  }

  __syncthreads();
  if (warp == 0) {
    tmem_allocator.free(shared.tmem_base_ptr, kTmemColumns);
  }
}

template <int kLocalVHeads>
inline void launch_qwen35_chunk_state_output_sm100_ss(
    cudaStream_t stream,
    const __nv_bfloat16* q_norm,
    const float* g,
    const __nv_bfloat16* Aqk,
    const __nv_bfloat16* w,
    const __nv_bfloat16* u,
    const __nv_bfloat16* kg,
    const float* initial_state,
    __nv_bfloat16* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    bool has_initial_state) {
  auto kernel_fn = &qwen35_chunk_state_output_sm100_ss_kernel<kLocalVHeads>;
  constexpr size_t shared_bytes =
      sizeof(Qwen35ChunkStateOutputSm100SsShared);
  cudaFuncSetAttribute(
      kernel_fn,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  const int grid =
      batch_size * kLocalVHeads * (kHeadDim / kValueTile);
  kernel_fn<<<grid, kThreads, shared_bytes, stream>>>(
      q_norm,
      g,
      Aqk,
      w,
      u,
      kg,
      initial_state,
      out,
      final_state,
      batch_size,
      seq_len,
      qk_heads,
      has_initial_state);
}

}  // namespace cula::qwen35::prefill::kernel::sm100_ss

#endif  // CULA_SM100_ENABLED
