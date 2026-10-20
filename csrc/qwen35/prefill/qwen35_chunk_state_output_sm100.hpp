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

// This header is intentionally independent from
// qwen35_scalar_kda_prefill_kernel.hpp.  It is a Blackwell-only prototype for
// replacing that file's WMMA chunk state/output stage without perturbing the
// recurrent fallback or the in-flight scalar-kernel work.

#if defined(CULA_SM100_ENABLED)

#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include "kerutils/kerutils.cuh"

namespace cula::qwen35::prefill::kernel::sm100_ts {

using namespace cute;

using bf16 = kerutils::bf16;

struct alignas(16) Bf16x8 {
  bf16 values[8];
};

static constexpr int kHeadDim = 128;
static constexpr int kChunk = 64;
#ifndef CULA_QWEN35_TS_VALUE_TILE
#define CULA_QWEN35_TS_VALUE_TILE 128
#endif
static constexpr int kValueTile = CULA_QWEN35_TS_VALUE_TILE;
static_assert(kValueTile == 64 || kValueTile == 128);
static constexpr int kTmemThreads = 128;
#ifndef CULA_QWEN35_TS_THREADS
#define CULA_QWEN35_TS_THREADS 352
#endif
static constexpr int kThreads = CULA_QWEN35_TS_THREADS;

// TMEM is addressed in 32-bit columns.  TS-UMMA requires its M=64 accumulator
// to start at datapath zero, so projection and output use separate 64-column
// regions.  (The DP16 packing accepted by the SS recompute mainloop is not a
// legal destination for this TS instruction.)
struct TmemAllocation {
  static constexpr uint32_t kStateF32 = 0;      // DP 0..31, 128 columns
  static constexpr uint32_t kStateBf16 = 128;   // DP 0..31,  64 columns
  static constexpr uint32_t kResult = 192;      // DP 0..31,  64 columns
  static constexpr uint32_t kOutput = 256;      // DP 0..31,  64 columns
  static constexpr int kColumns = 512;
};

using SmemLayout64x128K = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<kChunk>, Int<kHeadDim>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayout64x64K = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<kChunk>, Int<kChunk>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

// Shape-only layouts used to construct the TMEM A fragments.  They are kept
// separate from the 64-row B layouts above because the full-value path uses
// an M=128 TS-UMMA while W/Qg/Aqk still have 64 token rows.
using SmemLayoutValuex128K = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<kValueTile>, Int<kHeadDim>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayoutValuex64K = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<kValueTile>, Int<kChunk>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

// Logical shape [N=128, K=64].  N/MN-major storage makes both the source KG
// loads (KG is physically [token, K]) and the SMEM stores coalesced while UMMA
// performs the required transpose internally.
using SmemLayout128x64MN = decltype(coalesce(
    tile_to_shape(
        UMMA::Layout_MN_SW128_Atom<bf16>{},
        Shape<Int<kHeadDim>, Int<kChunk>>{},
        Step<_1, _2>{}),
    Shape<_1, _1>{}));

using TiledMma64x64K = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_TS<
        bf16,
        bf16,
        float,
        kValueTile,
        kChunk,
        UMMA::Major::K,
        UMMA::Major::K>{}));

using TiledMma64x128MN = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_TS<
        bf16,
        bf16,
        float,
        kValueTile,
        kHeadDim,
        UMMA::Major::K,
        UMMA::Major::MN>{}));

struct alignas(128) Qwen35ChunkStateOutputSm100Shared {
  // Two buffers are required because the two independent TS contractions are
  // issued before a single UMMA completion wait.
  alignas(128) bf16 operand_b0[kChunk * kHeadDim];
  alignas(128) bf16 operand_b1[kChunk * kHeadDim];
  float gate_exp[kChunk];
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;
};

static_assert(
    sizeof(Qwen35ChunkStateOutputSm100Shared) <= 48 * 1024,
    "SM100 state/output prototype should remain below 48 KiB shared memory");

CUTE_DEVICE uint32_t pack_bf16_pair(float x0, float x1) {
  union Bf16Bits {
    __nv_bfloat16 value;
    uint16_t bits;
  } lo{}, hi{};
  lo.value = __float2bfloat16_rn(x0);
  hi.value = __float2bfloat16_rn(x1);
  return static_cast<uint32_t>(lo.bits) | (static_cast<uint32_t>(hi.bits) << 16);
}

// Store a [K=128, V=64] FP32 state tile in its transposed TMEM representation
// [V=64, K=128], and create the packed BF16 operand-A shadow at the same time.
CUTE_DEVICE void initialize_state_tmem(
    uint32_t tmem_base,
    const float* __restrict__ initial_state,
    int state_global_base,
    int v_base,
    bool has_initial_state) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  constexpr int kValuesPerWarp = kValueTile / 4;
  const bool active = lane < kValuesPerWarp;
  const int vv = warp * kValuesPerWarp + (lane & (kValuesPerWarp - 1));

#pragma unroll
  for (int kk0 = 0; kk0 < kHeadDim; kk0 += 16) {
    float state_values[16];
    uint32_t state_bf16[8];
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      state_values[item] = active && has_initial_state
          ? initial_state[state_global_base + (kk0 + item) * kHeadDim + v_base + vv]
          : 0.0f;
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      state_bf16[item] = pack_bf16_pair(state_values[2 * item], state_values[2 * item + 1]);
    }
    kerutils::tmem_st_32dp32bNx<16>(tmem_base + TmemAllocation::kStateF32 + kk0, state_values);
    kerutils::tmem_st_32dp32bNx<8>(tmem_base + TmemAllocation::kStateBf16 + kk0 / 2, state_bf16);
  }
  cutlass::arch::fence_view_async_tmem_store();
  kerutils::tcgen05_before_thread_sync();
}

template <int kLocalVHeads, bool kPrefetchGate>
__global__ __launch_bounds__(kThreads, 1) void qwen35_chunk_state_output_sm100_ts_kernel(
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
  // CUDA only guarantees the base alignment of an untyped dynamic-shared
  // declaration.  UMMA SW128 descriptors require 128-byte alignment, so align
  // the struct explicitly instead of relying on alignas to move the runtime
  // base address.
  const uintptr_t shared_addr = reinterpret_cast<uintptr_t>(shared_bytes);
  const uintptr_t shared_aligned_addr = (shared_addr + 127u) & ~uintptr_t(127u);
  auto& shared = *reinterpret_cast<Qwen35ChunkStateOutputSm100Shared*>(shared_aligned_addr);

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  int work = static_cast<int>(blockIdx.x);
  const int value_tile = work % (kHeadDim / kValueTile);
  work /= (kHeadDim / kValueTile);
  const int hv = work % kLocalVHeads;
  const int seq = work / kLocalVHeads;
  if (seq >= batch_size) {
    return;
  }

  const int heads_per_group = kLocalVHeads / qk_heads;
  const int qk_h = hv / heads_per_group;
  const int v_base = value_tile * kValueTile;
  const int state_global_base = (seq * kLocalVHeads + hv) * kHeadDim * kHeadDim;

  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (warp == 0) {
    tmem_allocator.allocate(TmemAllocation::kColumns, &shared.tmem_base_ptr);
    // Do not hold the per-SM allocation permit for the whole kernel.
    tmem_allocator.release_allocation_lock();
  }
  if (tid == 0) {
    cute::initialize_barrier(shared.mma_barrier, 1);
  }
  __syncthreads();

  const uint32_t tmem_base = shared.tmem_base_ptr;
  if (tid < kTmemThreads) {
    initialize_state_tmem(
        tmem_base,
        initial_state,
        state_global_base,
        v_base,
        has_initial_state);
  }
  __syncthreads();
  kerutils::tcgen05_after_thread_sync();

  TiledMma64x64K mma_64x64;
  TiledMma64x128MN mma_64x128;

  // Construct correctly-shaped TMEM A fragments from fake SMEM tensors.  The
  // fragment data pointers are then redirected to the explicit TMEM plan.
  // A TS operand is a TMEM fragment; the SMEM tensor is shape-only.  It must
  // use a null SMEM pointer so no real shared-memory base/swizzle offset leaks
  // into the generated TMEM fragment layout.
  auto fake_state = make_tensor(
      make_smem_ptr(static_cast<bf16*>(nullptr)), SmemLayoutValuex128K{});
  auto fake_vnew = make_tensor(
      make_smem_ptr(static_cast<bf16*>(nullptr)), SmemLayoutValuex64K{});
  auto t_state_a = mma_64x64.get_slice(_0{}).partition_fragment_A(fake_state);
  t_state_a.data() = tmem_base + TmemAllocation::kStateBf16;
  auto t_vnew_a_64 = mma_64x64.get_slice(_0{}).partition_fragment_A(fake_vnew);
  t_vnew_a_64.data() = tmem_base + TmemAllocation::kStateBf16;
  auto t_vnew_a_128 = mma_64x128.get_slice(_0{}).partition_fragment_A(fake_vnew);
  t_vnew_a_128.data() = tmem_base + TmemAllocation::kStateBf16;

  auto t_projection = partition_fragment_C(
      mma_64x64, Shape<Int<kValueTile>, Int<kChunk>>{});
  t_projection.data() = tmem_base + TmemAllocation::kResult;
  auto t_output = partition_fragment_C(
      mma_64x64, Shape<Int<kValueTile>, Int<kChunk>>{});
  t_output.data() = tmem_base + TmemAllocation::kOutput;
  auto t_state_acc = partition_fragment_C(
      mma_64x128, Shape<Int<kValueTile>, Int<kHeadDim>>{});
  t_state_acc.data() = tmem_base + TmemAllocation::kStateF32;

  int barrier_phase = 0;
  const int chunk_count = (seq_len + kChunk - 1) / kChunk;
  const float q_scale = rsqrtf(static_cast<float>(kHeadDim));
  const auto* q_norm_bf16 = reinterpret_cast<const bf16*>(q_norm);
  const auto* Aqk_bf16 = reinterpret_cast<const bf16*>(Aqk);
  const auto* w_bf16 = reinterpret_cast<const bf16*>(w);
  const auto* kg_bf16 = reinterpret_cast<const bf16*>(kg);

  if constexpr (kPrefetchGate) {
    // Seed the first chunk's scalar gate.  Later chunks are prefetched while
    // the first UMMA pair is in flight, which removes one block barrier from
    // every recurrent chunk transition.
    if (tid < kChunk) {
      if (tid < min(kChunk, seq_len)) {
        const int token = seq * seq_len + tid;
        shared.gate_exp[tid] = exp2f(g[token * kLocalVHeads + hv]);
      } else {
        shared.gate_exp[tid] = 0.0f;
      }
    }
    __syncthreads();
  }

  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int chunk_start = chunk * kChunk;
    const int valid_rows = min(kChunk, seq_len - chunk_start);

    if constexpr (!kPrefetchGate) {
      // Keep the short-sequence specialization identical to the lower-latency
      // original path; prefetching only pays back after several chunks.
      if (tid < kChunk) {
        if (tid < valid_rows) {
          const int token = seq * seq_len + chunk_start + tid;
          shared.gate_exp[tid] = exp2f(g[token * kLocalVHeads + hv]);
        } else {
          shared.gate_exp[tid] = 0.0f;
        }
      }
      __syncthreads();
    }

    // First pair: transpose(W @ state) and transpose(Qg @ state).
    auto s_w = make_tensor(make_smem_ptr(shared.operand_b0), SmemLayout64x128K{});
    auto s_qg = make_tensor(make_smem_ptr(shared.operand_b1), SmemLayout64x128K{});
    constexpr int kBf16PerVector = 8;
    constexpr int kHeadVectors = kHeadDim / kBf16PerVector;
    for (int vector_idx = tid;
         vector_idx < kChunk * kHeadVectors;
         vector_idx += kThreads) {
      const int row = vector_idx / kHeadVectors;
      const int kk = (vector_idx % kHeadVectors) * kBf16PerVector;
      Bf16x8 w_values{};
      Bf16x8 qg_values{};
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        w_values = *reinterpret_cast<const Bf16x8*>(
            w_bf16 + (token * kLocalVHeads + hv) * kHeadDim + kk);
        const Bf16x8 q_values = *reinterpret_cast<const Bf16x8*>(
            q_norm_bf16 + (token * qk_heads + qk_h) * kHeadDim + kk);
#pragma unroll
        for (int item = 0; item < kBf16PerVector; ++item) {
          qg_values.values[item] = bf16(
              static_cast<float>(q_values.values[item]) *
              shared.gate_exp[row] * q_scale);
        }
      }
      *reinterpret_cast<Bf16x8*>(&s_w(row, kk)) = w_values;
      *reinterpret_cast<Bf16x8*>(&s_qg(row, kk)) = qg_values;
    }
    __syncthreads();

    if (warp == 0) {
      cutlass::arch::fence_view_async_shared();
      kerutils::utcmma_ts(mma_64x64, t_state_a, s_w, t_projection, true);
      kerutils::tcgen05_after_thread_sync();
      kerutils::utcmma_ts(mma_64x64, t_state_a, s_qg, t_output, true);
      cutlass::arch::umma_arrive(&shared.mma_barrier);
    }
    // Once W/Qg have been staged, gate_exp is dead for this chunk.  Use two
    // non-issuer warps to prepare the next chunk while UMMA consumes SMEM.
    // The epilogue's block barrier below makes these writes visible before
    // the next iteration starts loading Qg.
    if constexpr (kPrefetchGate) {
      if (chunk + 1 < chunk_count && tid >= 32 && tid < 32 + kChunk) {
        const int next_row = tid - 32;
        const int next_start = chunk_start + kChunk;
        const int next_valid_rows = min(kChunk, seq_len - next_start);
        if (next_row < next_valid_rows) {
          const int token = seq * seq_len + next_start + next_row;
          shared.gate_exp[next_row] = exp2f(g[token * kLocalVHeads + hv]);
        } else {
          shared.gate_exp[next_row] = 0.0f;
        }
      }
    }
    cute::wait_barrier(shared.mma_barrier, barrier_phase);
    barrier_phase ^= 1;

    // Load the transposed projection in 16-column slices and create Vnew.
    constexpr int kValuesPerWarp = kValueTile / 4;
    const bool active_value = lane < kValuesPerWarp;
    const int vv = warp * kValuesPerWarp + (lane & (kValuesPerWarp - 1));
    if (tid < kTmemThreads) {
      kerutils::tcgen05_after_thread_sync();
#pragma unroll
      for (int row0 = 0; row0 < kChunk; row0 += 16) {
        float pair_values[16];
        uint32_t vnew_bf16[8];
        kerutils::tmem_ld_32dp32bNx<16>(
            tmem_base + TmemAllocation::kResult + row0,
            pair_values);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          float v0 = 0.0f;
          float v1 = 0.0f;
          if (active_value) {
            const int row_a = row0 + 2 * item;
            const int row_b = row_a + 1;
            if (row_a < valid_rows) {
              const int token = seq * seq_len + chunk_start + row_a;
              v0 = __bfloat162float(
                       u[(token * kLocalVHeads + hv) * kHeadDim + v_base + vv]) -
                  pair_values[2 * item];
            }
            if (row_b < valid_rows) {
              const int token = seq * seq_len + chunk_start + row_b;
              v1 = __bfloat162float(
                       u[(token * kLocalVHeads + hv) * kHeadDim + v_base + vv]) -
                  pair_values[2 * item + 1];
            }
          }
          vnew_bf16[item] = pack_bf16_pair(v0, v1);
        }
        kerutils::tmem_st_32dp32bNx<8>(
            tmem_base + TmemAllocation::kStateBf16 + row0 / 2,
            vnew_bf16);
      }

      // Apply the chunk decay to the persistent FP32 state before accumulating
      // the KG^T @ Vnew update.  DP 16..31 are unused for this tile.
      const int last_token = seq * seq_len + chunk_start + valid_rows - 1;
      const float chunk_decay = exp2f(g[last_token * kLocalVHeads + hv]);
#pragma unroll
      for (int kk0 = 0; kk0 < kHeadDim; kk0 += 16) {
        float state_values[16];
        kerutils::tmem_ld_32dp32bNx<16>(
            tmem_base + TmemAllocation::kStateF32 + kk0,
            state_values);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int item = 0; item < 16; ++item) {
          state_values[item] *= chunk_decay;
        }
        kerutils::tmem_st_32dp32bNx<16>(
            tmem_base + TmemAllocation::kStateF32 + kk0,
            state_values);
      }
      cutlass::arch::fence_view_async_tmem_store();
      kerutils::tcgen05_before_thread_sync();
    }
    __syncthreads();
    if (tid < kTmemThreads) {
      kerutils::tcgen05_after_thread_sync();
    }

    // Second pair: Aqk @ Vnew accumulates into output, while KG^T @ Vnew
    // accumulates directly into the decayed FP32 state tile.
    auto s_aqk = make_tensor(make_smem_ptr(shared.operand_b0), SmemLayout64x64K{});
    auto s_kg = make_tensor(make_smem_ptr(shared.operand_b1), SmemLayout128x64MN{});
    constexpr int kChunkVectors = kChunk / kBf16PerVector;
    for (int vector_idx = tid;
         vector_idx < kChunk * kChunkVectors;
         vector_idx += kThreads) {
      const int row = vector_idx / kChunkVectors;
      const int col = (vector_idx % kChunkVectors) * kBf16PerVector;
      Bf16x8 values{};
      if (row < valid_rows && col < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        values = *reinterpret_cast<const Bf16x8*>(
            Aqk_bf16 + (token * kLocalVHeads + hv) * kChunk + col);
      }
      *reinterpret_cast<Bf16x8*>(&s_aqk(row, col)) = values;
    }
    // Iterate in KG's physical [token, K] order; MN-major B storage keeps the
    // destination N coordinate contiguous too.
    for (int vector_idx = tid;
         vector_idx < kChunk * kHeadVectors;
         vector_idx += kThreads) {
      const int row = vector_idx / kHeadVectors;
      const int kk = (vector_idx % kHeadVectors) * kBf16PerVector;
      Bf16x8 values{};
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        values = *reinterpret_cast<const Bf16x8*>(
            kg_bf16 + (token * kLocalVHeads + hv) * kHeadDim + kk);
      }
#pragma unroll
      for (int item = 0; item < kBf16PerVector; ++item) {
        s_kg(kk + item, row) = values.values[item];
      }
    }
    __syncthreads();

    if (warp == 0) {
      cutlass::arch::fence_view_async_shared();
      kerutils::utcmma_ts(mma_64x64, t_vnew_a_64, s_aqk, t_output, false);
      kerutils::tcgen05_after_thread_sync();
      kerutils::utcmma_ts(mma_64x128, t_vnew_a_128, s_kg, t_state_acc, false);
      cutlass::arch::umma_arrive(&shared.mma_barrier);
    }
    cute::wait_barrier(shared.mma_barrier, barrier_phase);
    barrier_phase ^= 1;

    // Store the completed transposed output.  Within each warp, lower-half
    // lanes write consecutive V columns for a fixed token.
    if (tid < kTmemThreads) {
      kerutils::tcgen05_after_thread_sync();
#pragma unroll
      for (int row0 = 0; row0 < kChunk; row0 += 16) {
        float pair_values[16];
        kerutils::tmem_ld_32dp32bNx<16>(
            tmem_base + TmemAllocation::kOutput + row0,
            pair_values);
        cutlass::arch::fence_view_async_tmem_load();
        if (active_value) {
#pragma unroll
          for (int item = 0; item < 16; ++item) {
            const int row = row0 + item;
            if (row < valid_rows) {
              const int token = seq * seq_len + chunk_start + row;
              out[(token * kLocalVHeads + hv) * kHeadDim + v_base + vv] =
                  __float2bfloat16_rn(pair_values[item]);
            }
          }
        }
      }

      // Refresh the BF16 state shadow for the next chunk.  The final chunk also
      // writes the exact FP32 persistent state to the public output tensor, but
      // does not need a shadow refresh because no later chunk consumes it.
      const bool is_last_chunk = chunk + 1 == chunk_count;
      if (!is_last_chunk) {
#pragma unroll
        for (int kk0 = 0; kk0 < kHeadDim; kk0 += 16) {
          float state_values[16];
          uint32_t state_bf16[8];
          kerutils::tmem_ld_32dp32bNx<16>(
              tmem_base + TmemAllocation::kStateF32 + kk0,
              state_values);
          cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
          for (int item = 0; item < 8; ++item) {
            state_bf16[item] = pack_bf16_pair(state_values[2 * item], state_values[2 * item + 1]);
          }
          kerutils::tmem_st_32dp32bNx<8>(
              tmem_base + TmemAllocation::kStateBf16 + kk0 / 2,
              state_bf16);
        }
        cutlass::arch::fence_view_async_tmem_store();
        kerutils::tcgen05_before_thread_sync();
      } else {
#pragma unroll
        for (int kk0 = 0; kk0 < kHeadDim; kk0 += 16) {
          float state_values[16];
          kerutils::tmem_ld_32dp32bNx<16>(
              tmem_base + TmemAllocation::kStateF32 + kk0,
              state_values);
          cutlass::arch::fence_view_async_tmem_load();
          if (active_value) {
#pragma unroll
            for (int item = 0; item < 16; ++item) {
              final_state[state_global_base + (kk0 + item) * kHeadDim + v_base + vv] =
                  state_values[item];
            }
          }
        }
      }
    }
    __syncthreads();
    if (tid < kTmemThreads) {
      kerutils::tcgen05_after_thread_sync();
    }
  }

  __syncthreads();
  if (warp == 0) {
    tmem_allocator.free(tmem_base, TmemAllocation::kColumns);
  }
}

template <int kLocalVHeads, bool kPrefetchGate>
inline void launch_qwen35_chunk_state_output_sm100_ts_variant(
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
  auto kernel_fn =
      &qwen35_chunk_state_output_sm100_ts_kernel<kLocalVHeads, kPrefetchGate>;
  constexpr size_t shared_bytes = sizeof(Qwen35ChunkStateOutputSm100Shared) + 127;
  cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  const int grid = batch_size * kLocalVHeads * (kHeadDim / kValueTile);
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

template <int kLocalVHeads>
inline void launch_qwen35_chunk_state_output_sm100_ts(
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
  if (seq_len >= 256) {
    launch_qwen35_chunk_state_output_sm100_ts_variant<kLocalVHeads, true>(
        stream, q_norm, g, Aqk, w, u, kg, initial_state, out, final_state,
        batch_size, seq_len, qk_heads, has_initial_state);
  } else {
    launch_qwen35_chunk_state_output_sm100_ts_variant<kLocalVHeads, false>(
        stream, q_norm, g, Aqk, w, u, kg, initial_state, out, final_state,
        batch_size, seq_len, qk_heads, has_initial_state);
  }
}

} // namespace cula::qwen35::prefill::kernel::sm100_ts

#endif // CULA_SM100_ENABLED
