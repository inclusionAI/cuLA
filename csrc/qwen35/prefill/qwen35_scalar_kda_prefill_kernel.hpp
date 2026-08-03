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

#include "qwen35_prefill_common.cuh"

#include <cute/tensor.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace cula::qwen35::prefill::kernel {

using namespace cute;

template <typename scalar_t, int kLocalVHeads, int kWarpsPerBlock>
struct Qwen35ScalarKdaPrefillKernel {
  static constexpr int kWarpSize = 32;
  static constexpr int kWarps = kWarpsPerBlock;
  static constexpr int kThreads = kWarps * kWarpSize;
  static constexpr int kKValuesPerLane = kHeadDimQK / kWarpSize;
  static constexpr int kHeadDim = kHeadDimQK;
  static constexpr int kColumnsPerWarp = 1;
  // Each warp owns one independent V column and keeps its complete recurrent
  // state in registers.  All warps in the CTA reuse one normalized Q/K vector
  // and one scalar gate through shared memory.
  static constexpr int kVTile = kWarps;
  static constexpr int kNumVTiles = kHeadDimV / kVTile;

  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);
  static_assert(kThreads % kWarpSize == 0);
  static_assert(kHeadDimQK % kWarpSize == 0);
  static_assert(kHeadDimV % kVTile == 0);

  struct SharedStorage {
    float q_norm[kHeadDimQK];
    float k_norm[kHeadDimQK];
    float decay;
    float beta;
    int unsafe_gate;
  };

  static dim3 block_shape() {
    return dim3(kThreads, 1, 1);
  }

  CUTE_HOST_DEVICE static auto make_v_work_tiles(int sequence_count) {
    auto problem_layout = make_layout(
        make_shape(Int<kHeadDimV>{}, Int<kLocalVHeads>{}, sequence_count),
        make_stride(Int<1>{}, Int<kHeadDimV>{}, Int<kHeadDimV * kLocalVHeads>{}));
    return zipped_divide(problem_layout, make_shape(Int<kVTile>{}, Int<1>{}, Int<1>{}));
  }

  static dim3 grid_shape(int sequence_count) {
    auto v_work_tiles = make_v_work_tiles(sequence_count);
    return dim3(static_cast<unsigned int>(size<1>(v_work_tiles)), 1, 1);
  }

  CUTE_DEVICE static float load_as_float(scalar_t value) {
    return static_cast<float>(value);
  }

  CUTE_DEVICE static scalar_t cast_output(float value) {
    return static_cast<scalar_t>(value);
  }

  // A one-warp specialization avoids CTA barriers altogether.  It is useful
  // for the high-HV/small-T regime where the extra Q/K/gate work is cheaper
  // than synchronizing a multi-warp V tile.
  CUTE_DEVICE static void run_warp_only(
      const scalar_t* __restrict__ q,
      const scalar_t* __restrict__ k,
      const scalar_t* __restrict__ v,
      const scalar_t* __restrict__ a,
      const scalar_t* __restrict__ b,
      const float* __restrict__ A_log,
      const float* __restrict__ dt_bias,
      const float* __restrict__ initial_state,
      const int32_t* __restrict__ cu_seqlens,
      scalar_t* __restrict__ out,
      float* __restrict__ final_state,
      int batch_size,
      int seq_len,
      int qk_heads,
      int sequence_count,
      bool is_varlen,
      bool has_initial_state,
      const float* __restrict__ precomputed_g = nullptr,
      const float* __restrict__ precomputed_beta = nullptr,
      const int32_t* __restrict__ unsafe_gate_flags = nullptr) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    int work = static_cast<int>(blockIdx.x);
    const int v_row = work % kHeadDimV;
    work /= kHeadDimV;
    const int hv = work % kLocalVHeads;
    const int seq_idx = work / kLocalVHeads;
    if (seq_idx >= sequence_count) {
      return;
    }
    const int repeat = kLocalVHeads / qk_heads;
    const int qk_h = hv / repeat;
    const int token_begin = is_varlen ? static_cast<int>(cu_seqlens[seq_idx]) : seq_idx * seq_len;
    const int token_end = is_varlen ? static_cast<int>(cu_seqlens[seq_idx + 1]) : token_begin + seq_len;
    const int state_base = ((seq_idx * kLocalVHeads + hv) * kHeadDimQK) * kHeadDimV;
    if (precomputed_g != nullptr) {
      if (unsafe_gate_flags != nullptr) {
        if (unsafe_gate_flags[seq_idx * kLocalVHeads + hv] == 0) {
          return;
        }
      } else {
        bool unsafe_gate = false;
        for (int token = token_begin + lane; token < token_end; token += kWarpSize) {
          const float gate = precomputed_g[token * kLocalVHeads + hv];
          unsafe_gate = unsafe_gate || !isfinite(gate) || gate < -5.0f || gate > 0.0f;
        }
        if (!__any_sync(0xffffffffu, unsafe_gate)) {
          return;
        }
      }
    }
    float state_vals[kKValuesPerLane];
#pragma unroll
    for (int item = 0; item < kKValuesPerLane; ++item) {
      const int kk = lane + item * kWarpSize;
      state_vals[item] = has_initial_state ? initial_state[state_base + kk * kHeadDimV + v_row] : 0.0f;
    }
    const float scale = rsqrtf(static_cast<float>(kHeadDimQK));
    const float exp_A = precomputed_g == nullptr ? expf(A_log[hv]) : 0.0f;
    const float dt = precomputed_g == nullptr ? dt_bias[hv] : 0.0f;
    for (int token = token_begin; token < token_end; ++token) {
      const int local_t = token - token_begin;
      const int qk_base = ((token * qk_heads + qk_h) * kHeadDimQK);
      const int v_input_base = ((token * kLocalVHeads + hv) * kHeadDimV);
      float q_vals[kKValuesPerLane];
      float k_vals[kKValuesPerLane];
      float q_norm_sq = 0.0f;
      float k_norm_sq = 0.0f;
#pragma unroll
      for (int item = 0; item < kKValuesPerLane; ++item) {
        const int kk = lane + item * kWarpSize;
        q_vals[item] = load_as_float(q[qk_base + kk]);
        k_vals[item] = load_as_float(k[qk_base + kk]);
        q_norm_sq += q_vals[item] * q_vals[item];
        k_norm_sq += k_vals[item] * k_vals[item];
      }
      // The SM90 speculative path passes its BF16-normalized Q/K workspace to
      // the exact overwrite. Reuse those values verbatim so both paths have
      // identical normalization and rounding semantics.
      const float q_rnorm = precomputed_g != nullptr
          ? scale
          : rsqrtf(fmaxf(warp_sum(q_norm_sq), 1.0e-20f)) * scale;
      const float k_rnorm = precomputed_g != nullptr
          ? 1.0f
          : rsqrtf(fmaxf(warp_sum(k_norm_sq), 1.0e-20f));
      float decay = 0.0f;
      float beta = 0.0f;
      if (lane == 0) {
        const int gate_base = token * kLocalVHeads + hv;
        if (precomputed_g != nullptr) {
          decay = expf(precomputed_g[gate_base]);
          beta = precomputed_beta[gate_base];
        } else {
          decay = expf(-exp_A * softplus(load_as_float(a[gate_base]) + dt));
          beta = 1.0f / (1.0f + expf(-load_as_float(b[gate_base])));
        }
      }
      decay = __shfl_sync(0xffffffffu, decay, 0);
      beta = __shfl_sync(0xffffffffu, beta, 0);
#pragma unroll
      for (int item = 0; item < kKValuesPerLane; ++item) {
        k_vals[item] *= k_rnorm;
        q_vals[item] *= q_rnorm;
      }
      float proj_partial = 0.0f;
      float out_partial = 0.0f;
      const float v_val = __shfl_sync(0xffffffffu, lane == 0 ? load_as_float(v[v_input_base + v_row]) : 0.0f, 0);
#pragma unroll
      for (int item = 0; item < kKValuesPerLane; ++item) {
        proj_partial += state_vals[item] * k_vals[item];
      }
      const float proj = warp_sum(proj_partial);
      const float v_new = beta * (v_val - decay * proj);
#pragma unroll
      for (int item = 0; item < kKValuesPerLane; ++item) {
        state_vals[item] = decay * state_vals[item] + k_vals[item] * v_new;
        out_partial += state_vals[item] * q_vals[item];
      }
      const float out_acc = warp_sum(out_partial);
      if (lane == 0) {
        const int out_off = (token * kLocalVHeads + hv) * kHeadDimV + v_row;
        out[out_off] = cast_output(out_acc);
      }
    }
#pragma unroll
    for (int item = 0; item < kKValuesPerLane; ++item) {
      const int kk = lane + item * kWarpSize;
      final_state[state_base + kk * kHeadDimV + v_row] = state_vals[item];
    }
    (void)batch_size;
  }

  CUTE_DEVICE static float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
  }

  CUTE_DEVICE static float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
  }

  CUTE_DEVICE static void run_device(
      const scalar_t* __restrict__ q,
      const scalar_t* __restrict__ k,
      const scalar_t* __restrict__ v,
      const scalar_t* __restrict__ a,
      const scalar_t* __restrict__ b,
      const float* __restrict__ A_log,
      const float* __restrict__ dt_bias,
      const float* __restrict__ initial_state,
      const int32_t* __restrict__ cu_seqlens,
      scalar_t* __restrict__ out,
      float* __restrict__ final_state,
      int batch_size,
      int seq_len,
      int qk_heads,
      int sequence_count,
      bool is_varlen,
      bool has_initial_state,
      const float* __restrict__ precomputed_g,
      const float* __restrict__ precomputed_beta,
      const int32_t* __restrict__ unsafe_gate_flags,
      SharedStorage& storage) {
    if constexpr (kWarpsPerBlock == 1) {
      run_warp_only(
          q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state,
          batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state,
          precomputed_g, precomputed_beta, unsafe_gate_flags);
      return;
    }
    auto v_work_tiles = make_v_work_tiles(sequence_count);
    auto work_layout = make_layout(get<1>(v_work_tiles.shape()), LayoutLeft{});
    auto work_coord = work_layout.get_hier_coord(static_cast<int>(blockIdx.x));
    const int v_tile_idx = static_cast<int>(get<0>(work_coord));
    const int hv = static_cast<int>(get<1>(work_coord));
    const int seq_idx = static_cast<int>(get<2>(work_coord));
    const int v_base = v_tile_idx * kVTile;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;

    if (hv >= kLocalVHeads || seq_idx >= sequence_count) {
      return;
    }

    const int token_begin = is_varlen ? static_cast<int>(cu_seqlens[seq_idx]) : seq_idx * seq_len;
    const int token_end = is_varlen ? static_cast<int>(cu_seqlens[seq_idx + 1]) : token_begin + seq_len;
    const int state_base = ((seq_idx * kLocalVHeads + hv) * kHeadDimQK) * kHeadDimV;
    const int qk_h = hv / (kLocalVHeads / qk_heads);

    // A compact SM90 safe-gate chunk is exact for per-token log gates in
    // [-5, 0].  This recurrent fallback is launched after the fast kernel and
    // overwrites only heads whose raw gate falls outside that domain.
    if (precomputed_g != nullptr) {
      if (unsafe_gate_flags != nullptr) {
        if (unsafe_gate_flags[seq_idx * kLocalVHeads + hv] == 0) {
          return;
        }
      } else {
        if (tid == 0) {
          storage.unsafe_gate = 0;
        }
        __syncthreads();
        for (int token = token_begin + tid; token < token_end; token += kThreads) {
          const float gate = precomputed_g[token * kLocalVHeads + hv];
          if (!isfinite(gate) || gate < -5.0f || gate > 0.0f) {
            atomicExch(&storage.unsafe_gate, 1);
          }
        }
        __syncthreads();
        if (storage.unsafe_gate == 0) {
          return;
        }
      }
    }

    float state_vals[kColumnsPerWarp][kKValuesPerLane];

#pragma unroll
    for (int column = 0; column < kColumnsPerWarp; ++column) {
      const int v_row = v_base + warp * kColumnsPerWarp + column;
#pragma unroll
      for (int item = 0; item < kKValuesPerLane; ++item) {
        const int kk = lane + item * kWarpSize;
        const int state_off = state_base + kk * kHeadDimV + v_row;
        state_vals[column][item] = has_initial_state ? initial_state[state_off] : 0.0f;
      }
    }

    const float scale = rsqrtf(static_cast<float>(kHeadDimQK));
    const float exp_A = precomputed_g == nullptr ? expf(A_log[hv]) : 0.0f;
    const float dt = precomputed_g == nullptr ? dt_bias[hv] : 0.0f;

    for (int token = token_begin; token < token_end; ++token) {
      const int qk_base = ((token * qk_heads + qk_h) * kHeadDimQK);
      const int v_base_input = ((token * kLocalVHeads + hv) * kHeadDimV);
      const int gate_base = token * kLocalVHeads + hv;

      if (warp == 0) {
        float q_vals_raw[kKValuesPerLane];
        float k_vals_raw[kKValuesPerLane];
        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
#pragma unroll
        for (int item = 0; item < kKValuesPerLane; ++item) {
          const int kk = lane + item * kWarpSize;
          q_vals_raw[item] = load_as_float(q[qk_base + kk]);
          k_vals_raw[item] = load_as_float(k[qk_base + kk]);
          q_norm_sq += q_vals_raw[item] * q_vals_raw[item];
          k_norm_sq += k_vals_raw[item] * k_vals_raw[item];
        }
        q_norm_sq = warp_sum(q_norm_sq);
        k_norm_sq = warp_sum(k_norm_sq);
        const float q_rnorm = precomputed_g != nullptr
            ? scale
            : rsqrtf(fmaxf(q_norm_sq, 1.0e-20f)) * scale;
        const float k_rnorm = precomputed_g != nullptr
            ? 1.0f
            : rsqrtf(fmaxf(k_norm_sq, 1.0e-20f));
#pragma unroll
        for (int item = 0; item < kKValuesPerLane; ++item) {
          const int kk = lane + item * kWarpSize;
          storage.q_norm[kk] = q_vals_raw[item] * q_rnorm;
          storage.k_norm[kk] = k_vals_raw[item] * k_rnorm;
        }
        if (lane == 0) {
          if (precomputed_g != nullptr) {
            storage.decay = expf(precomputed_g[gate_base]);
            storage.beta = precomputed_beta[gate_base];
          } else {
            storage.decay = expf(-exp_A * softplus(load_as_float(a[gate_base]) + dt));
            storage.beta = 1.0f / (1.0f + expf(-load_as_float(b[gate_base])));
          }
        }
      }
      __syncthreads();

      float q_vals[kKValuesPerLane];
      float k_vals[kKValuesPerLane];
#pragma unroll
      for (int item = 0; item < kKValuesPerLane; ++item) {
        const int kk = lane + item * kWarpSize;
        q_vals[item] = storage.q_norm[kk];
        k_vals[item] = storage.k_norm[kk];
      }
      const float decay = storage.decay;
      const float beta = storage.beta;

#pragma unroll
      for (int column = 0; column < kColumnsPerWarp; ++column) {
        const int v_row = v_base + warp * kColumnsPerWarp + column;
        float proj_partial = 0.0f;
#pragma unroll
        for (int item = 0; item < kKValuesPerLane; ++item) {
          proj_partial += state_vals[column][item] * k_vals[item];
        }
        const float proj = warp_sum(proj_partial);

        float v_val = lane == 0 ? load_as_float(v[v_base_input + v_row]) : 0.0f;
        v_val = __shfl_sync(0xffffffffu, v_val, 0);
        const float v_new = beta * (v_val - decay * proj);

        float out_partial = 0.0f;
#pragma unroll
        for (int item = 0; item < kKValuesPerLane; ++item) {
          const float state_new = decay * state_vals[column][item] + k_vals[item] * v_new;
          state_vals[column][item] = state_new;
          out_partial += state_new * q_vals[item];
        }
        const float out_acc = warp_sum(out_partial);

        if (lane == 0) {
          const int out_off = (token * kLocalVHeads + hv) * kHeadDimV + v_row;
          out[out_off] = cast_output(out_acc);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int column = 0; column < kColumnsPerWarp; ++column) {
      const int v_row = v_base + warp * kColumnsPerWarp + column;
#pragma unroll
      for (int item = 0; item < kKValuesPerLane; ++item) {
        const int kk = lane + item * kWarpSize;
        const int state_off = state_base + kk * kHeadDimV + v_row;
        final_state[state_off] = state_vals[column][item];
      }
    }

    (void)batch_size;
  }
};

// The recurrent scalar kernel above is still the lowest-latency path for very
// short prompts.  For real prefill lengths, process time in 64-token chunks
// and use tensor cores for the state/output contractions.  The preceding
// native-GVA intra/WY stages provide Aqk, W, U, and Kg for this kernel.
static constexpr int kChunkSize = 64;
static constexpr int kValueTile = 64;
static constexpr int kValueTilesPerBlock = kValueTile / 16;
static constexpr int kChunkOutputWarps = 16;
static constexpr int kChunkStateWarps = kChunkOutputWarps;

struct alignas(128) Qwen35ChunkStateOutputShared {
  __nv_bfloat16 state[kHeadDimQK * kValueTile];
  __nv_bfloat16 matrix[kChunkSize * kHeadDimQK];
  __nv_bfloat16 v_new[kChunkSize * kValueTile];
  float gate_exp[kChunkSize];
  float accum[kHeadDimQK * kValueTile];
  __nv_bfloat16 aqk[kChunkSize * kChunkSize];
};

__device__ __forceinline__ float qwen35_bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 qwen35_float_to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float qwen35_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void qwen35_chunk_qk_norm_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ q_norm,
    __nv_bfloat16* __restrict__ k_norm,
    int vector_count) {
  const int vector_idx = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (vector_idx >= vector_count) {
    return;
  }
  const int base = vector_idx * kHeadDimQK;
  float q_values[4];
  float k_values[4];
  float q_sq = 0.0f;
  float k_sq = 0.0f;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int kk = lane + item * 32;
    q_values[item] = qwen35_bf16_to_float(q[base + kk]);
    k_values[item] = qwen35_bf16_to_float(k[base + kk]);
    q_sq += q_values[item] * q_values[item];
    k_sq += k_values[item] * k_values[item];
  }
  const float q_rnorm = rsqrtf(qwen35_warp_sum(q_sq) + 1.0e-6f);
  const float k_rnorm = rsqrtf(qwen35_warp_sum(k_sq) + 1.0e-6f);
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int kk = lane + item * 32;
    q_norm[base + kk] = qwen35_float_to_bf16(q_values[item] * q_rnorm);
    k_norm[base + kk] = qwen35_float_to_bf16(k_values[item] * k_rnorm);
  }
}

__device__ __forceinline__ float qwen35_softplus(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

__global__ void qwen35_chunk_gate_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ g,
    float* __restrict__ beta,
    int32_t* __restrict__ cu_seqlens,
    int32_t* __restrict__ chunk_indices,
    int batch_size,
    int seq_len,
    int v_heads,
    int chunks_per_sequence) {
  __shared__ float scan[kChunkSize];
  const int tid = static_cast<int>(threadIdx.x);
  int work = static_cast<int>(blockIdx.x);
  const int hv = work % v_heads;
  work /= v_heads;
  const int chunk = work % chunks_per_sequence;
  const int seq = work / chunks_per_sequence;
  const int local_t = chunk * kChunkSize + tid;
  const bool valid = tid < kChunkSize && local_t < seq_len;
  const int token = seq * seq_len + local_t;

  if (tid < kChunkSize) {
    float log2_decay = 0.0f;
    if (valid) {
      const int gate_offset = token * v_heads + hv;
      const float raw_a = qwen35_bf16_to_float(a[gate_offset]);
      const float raw_b = qwen35_bf16_to_float(b[gate_offset]);
      const float log_decay = -expf(A_log[hv]) * qwen35_softplus(raw_a + dt_bias[hv]);
      log2_decay = log_decay * 1.4426950408889634f;
      beta[gate_offset] = 1.0f / (1.0f + expf(-raw_b));
    }
    scan[tid] = log2_decay;
  }
  __syncthreads();

#pragma unroll
  for (int offset = 1; offset < kChunkSize; offset <<= 1) {
    float addend = 0.0f;
    if (tid < kChunkSize && tid >= offset) {
      addend = scan[tid - offset];
    }
    __syncthreads();
    if (tid < kChunkSize) {
      scan[tid] += addend;
    }
    __syncthreads();
  }

  if (tid < kChunkSize) {
    const int row_t = chunk * kChunkSize + tid;
    if (row_t < seq_len) {
      const int row_token = seq * seq_len + row_t;
      g[row_token * v_heads + hv] = scan[tid];
    }
  }

  if (hv == 0 && tid == 0) {
    const int chunk_idx = seq * chunks_per_sequence + chunk;
    chunk_indices[chunk_idx * 2] = seq;
    chunk_indices[chunk_idx * 2 + 1] = chunk;
    if (chunk == 0) {
      cu_seqlens[seq] = seq * seq_len;
    }
    if (chunk == chunks_per_sequence - 1) {
      cu_seqlens[seq + 1] = (seq + 1) * seq_len;
    }
  }
}

// The Qwen prefill path always needs both normalization and scalar-gate
// preprocessing.  Running the two small kernels back-to-back leaves roughly
// 2--3 us of avoidable serialization at short prefill lengths.  This fused
// launcher assigns four warps to four Q/K vectors for the first block range,
// then reuses the same 128-thread block shape for the gate scan blocks.  The
// two branches are block-uniform, so the gate barriers never involve norm
// threads from another branch and the prefix-sum semantics are unchanged.
#ifndef CULA_QWEN35_FAST_GATE_SCAN
#define CULA_QWEN35_FAST_GATE_SCAN 1
#endif
#ifndef CULA_QWEN35_PREPROCESS_THREADS
#define CULA_QWEN35_PREPROCESS_THREADS 256
#endif
#ifndef CULA_QWEN35_PREPROCESS_GATE_FIRST
#define CULA_QWEN35_PREPROCESS_GATE_FIRST 1
#endif
static_assert(CULA_QWEN35_PREPROCESS_THREADS >= 64,
              "Qwen35 preprocess requires at least two warps");
static_assert(CULA_QWEN35_PREPROCESS_THREADS % 32 == 0,
              "Qwen35 preprocess threads must be a multiple of 32");
template <bool UsePrecomputedGate>
__global__ void qwen35_chunk_preprocess_fused_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ gate_raw,
    const float* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ q_norm,
    __nv_bfloat16* __restrict__ k_norm,
    float* __restrict__ g,
    float* __restrict__ g_raw_output,
    float* __restrict__ beta,
    int32_t* __restrict__ unsafe_gate_flags,
    int32_t* __restrict__ cu_seqlens,
    int32_t* __restrict__ chunk_indices,
    int batch_size,
    int seq_len,
    int qk_heads,
    int v_heads,
    int vector_count,
    int gate_blocks,
    int norm_blocks,
    int chunks_per_sequence) {
  const int block = static_cast<int>(blockIdx.x);
#if CULA_QWEN35_PREPROCESS_GATE_FIRST
  if (block >= gate_blocks) {
    const int norm_block = block - gate_blocks;
#else
  if (block < norm_blocks) {
    const int norm_block = block;
#endif
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int vector_idx = norm_block * (static_cast<int>(blockDim.x) / 32) + warp;
    if (vector_idx >= vector_count) {
      return;
    }
    const int base = vector_idx * kHeadDimQK;
    float q_values[4];
    float k_values[4];
    float q_sq = 0.0f;
    float k_sq = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int kk = lane + item * 32;
      q_values[item] = qwen35_bf16_to_float(q[base + kk]);
      k_values[item] = qwen35_bf16_to_float(k[base + kk]);
      q_sq += q_values[item] * q_values[item];
      k_sq += k_values[item] * k_values[item];
    }
    const float q_rnorm = rsqrtf(qwen35_warp_sum(q_sq) + 1.0e-6f);
    const float k_rnorm = rsqrtf(qwen35_warp_sum(k_sq) + 1.0e-6f);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int kk = lane + item * 32;
      q_norm[base + kk] = qwen35_float_to_bf16(q_values[item] * q_rnorm);
      k_norm[base + kk] = qwen35_float_to_bf16(k_values[item] * k_rnorm);
    }
    return;
  }

  __shared__ float scan[kChunkSize + 2];
  const int tid = static_cast<int>(threadIdx.x);
#if CULA_QWEN35_PREPROCESS_GATE_FIRST
  int work = block;
#else
  int work = block - norm_blocks;
#endif
  const int hv = work % v_heads;
  work /= v_heads;
  const int chunk = work % chunks_per_sequence;
  const int seq = work / chunks_per_sequence;
  const int local_t = chunk * kChunkSize + tid;
  const bool valid = tid < kChunkSize && local_t < seq_len;
  const int token = seq * seq_len + local_t;

  if (tid < kChunkSize) {
    float log2_decay = 0.0f;
    if (valid) {
      const int gate_offset = token * v_heads + hv;
      float raw_log_decay;
      if constexpr (UsePrecomputedGate) {
        raw_log_decay = gate_raw[gate_offset];
        beta[gate_offset] = beta_in[gate_offset];
      } else {
        const float raw_a = qwen35_bf16_to_float(a[gate_offset]);
        const float raw_b = qwen35_bf16_to_float(b[gate_offset]);
        raw_log_decay = -expf(A_log[hv]) * qwen35_softplus(raw_a + dt_bias[hv]);
        beta[gate_offset] = 1.0f / (1.0f + expf(-raw_b));
      }
      log2_decay = raw_log_decay * 1.4426950408889634f;
      if (g_raw_output != nullptr) {
        g_raw_output[gate_offset] = raw_log_decay;
      }
      if (unsafe_gate_flags != nullptr &&
          (!isfinite(raw_log_decay) || raw_log_decay < -5.0f || raw_log_decay > 0.0f)) {
        atomicExch(&unsafe_gate_flags[seq * v_heads + hv], 1);
      }
    }
    scan[tid] = log2_decay;
  }
  __syncthreads();

#if CULA_QWEN35_FAST_GATE_SCAN
  if (tid < kChunkSize) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float prefix = scan[tid];
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float addend = __shfl_up_sync(0xffffffffu, prefix, offset);
      if (lane >= offset) {
        prefix += addend;
      }
    }
    if (lane == 31) {
      scan[kChunkSize + warp] = prefix;
    }
    scan[tid] = prefix;
  }
  __syncthreads();
  if (tid >= 32 && tid < kChunkSize) {
    scan[tid] += scan[kChunkSize];
  }
  __syncthreads();
#else
#pragma unroll
  for (int offset = 1; offset < kChunkSize; offset <<= 1) {
    float addend = 0.0f;
    if (tid < kChunkSize && tid >= offset) {
      addend = scan[tid - offset];
    }
    __syncthreads();
    if (tid < kChunkSize) {
      scan[tid] += addend;
    }
    __syncthreads();
  }
#endif

  if (tid < kChunkSize) {
    const int row_t = chunk * kChunkSize + tid;
    if (row_t < seq_len) {
      const int row_token = seq * seq_len + row_t;
      g[row_token * v_heads + hv] = scan[tid];
    }
  }

  if (hv == 0 && tid == 0) {
    const int chunk_idx = seq * chunks_per_sequence + chunk;
    chunk_indices[chunk_idx * 2] = seq;
    chunk_indices[chunk_idx * 2 + 1] = chunk;
    if (chunk == 0) {
      cu_seqlens[seq] = seq * seq_len;
    }
    if (chunk == chunks_per_sequence - 1) {
      cu_seqlens[seq + 1] = (seq + 1) * seq_len;
    }
  }
}

template <int kLocalVHeads>
__global__ __launch_bounds__(kChunkStateWarps * 32, 1) void qwen35_chunk_state_output_kernel(
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
  using namespace nvcuda;
  extern __shared__ char shared_bytes[];
  auto& shared = *reinterpret_cast<Qwen35ChunkStateOutputShared*>(shared_bytes);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  int work = static_cast<int>(blockIdx.x);
  const int value_tile = work % (kHeadDimV / kValueTile);
  work /= (kHeadDimV / kValueTile);
  const int hv = work % kLocalVHeads;
  const int seq = work / kLocalVHeads;
  if (seq >= batch_size) {
    return;
  }
  const int qk_h = hv / (kLocalVHeads / qk_heads);
  const int v_base = value_tile * kValueTile;
  const int state_global_base = (seq * kLocalVHeads + hv) * kHeadDimQK * kHeadDimV;

  for (int index = tid; index < kHeadDimQK * kValueTile; index += static_cast<int>(blockDim.x)) {
    const int kk = index / kValueTile;
    const int vv = index % kValueTile;
    shared.state[index] = qwen35_float_to_bf16(
        has_initial_state ? initial_state[state_global_base + kk * kHeadDimV + v_base + vv] : 0.0f);
  }
  __syncthreads();

  const int chunk_count = (seq_len + kChunkSize - 1) / kChunkSize;
  const float q_scale = rsqrtf(static_cast<float>(kHeadDimQK));
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int chunk_start = chunk * kChunkSize;
    const int valid_rows = min(kChunkSize, seq_len - chunk_start);

    for (int row = tid; row < kChunkSize; row += static_cast<int>(blockDim.x)) {
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        shared.gate_exp[row] = exp2f(g[token * kLocalVHeads + hv]);
      } else {
        shared.gate_exp[row] = 0.0f;
      }
    }
    __syncthreads();
    for (int index = tid; index < kChunkSize * kHeadDimQK; index += static_cast<int>(blockDim.x)) {
      const int row = index / kHeadDimQK;
      const int kk = index % kHeadDimQK;
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        shared.matrix[index] = w[(token * kLocalVHeads + hv) * kHeadDimQK + kk];
      } else {
        shared.matrix[index] = qwen35_float_to_bf16(0.0f);
      }
    }
    __syncthreads();

    for (int tile = warp; tile < (kChunkSize / 16) * kValueTilesPerBlock; tile += kChunkOutputWarps) {
      const int tile_m = tile / kValueTilesPerBlock;
      const int tile_n = tile % kValueTilesPerBlock;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
      wmma::fill_fragment(frag_c, 0.0f);
#pragma unroll
      for (int kk = 0; kk < kHeadDimQK; kk += 16) {
        wmma::load_matrix_sync(frag_a, shared.matrix + tile_m * 16 * kHeadDimQK + kk, kHeadDimQK);
        wmma::load_matrix_sync(frag_b, shared.state + kk * kValueTile + tile_n * 16, kValueTile);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
      }
      wmma::store_matrix_sync(
          shared.accum + tile_m * 16 * kValueTile + tile_n * 16,
          frag_c,
          kValueTile,
          wmma::mem_row_major);
    }
    __syncthreads();

    for (int index = tid; index < kChunkSize * kValueTile; index += static_cast<int>(blockDim.x)) {
      const int row = index / kValueTile;
      const int vv = index % kValueTile;
      float value = 0.0f;
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        value = qwen35_bf16_to_float(u[(token * kLocalVHeads + hv) * kHeadDimV + v_base + vv]) -
            shared.accum[index];
      }
      shared.v_new[index] = qwen35_float_to_bf16(value);
    }
    for (int index = tid; index < kChunkSize * kHeadDimQK; index += static_cast<int>(blockDim.x)) {
      const int row = index / kHeadDimQK;
      const int kk = index % kHeadDimQK;
      float value = 0.0f;
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        value = qwen35_bf16_to_float(q_norm[(token * qk_heads + qk_h) * kHeadDimQK + kk]) *
            shared.gate_exp[row] * q_scale;
      }
      shared.matrix[index] = qwen35_float_to_bf16(value);
    }
    __syncthreads();

    for (int tile = warp; tile < (kChunkSize / 16) * kValueTilesPerBlock; tile += kChunkOutputWarps) {
      const int tile_m = tile / kValueTilesPerBlock;
      const int tile_n = tile % kValueTilesPerBlock;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
      wmma::fill_fragment(frag_c, 0.0f);
#pragma unroll
      for (int kk = 0; kk < kHeadDimQK; kk += 16) {
        wmma::load_matrix_sync(frag_a, shared.matrix + tile_m * 16 * kHeadDimQK + kk, kHeadDimQK);
        wmma::load_matrix_sync(frag_b, shared.state + kk * kValueTile + tile_n * 16, kValueTile);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
      }
      wmma::store_matrix_sync(
          shared.accum + tile_m * 16 * kValueTile + tile_n * 16,
          frag_c,
          kValueTile,
          wmma::mem_row_major);
    }
    __syncthreads();

    for (int index = tid; index < kChunkSize * kChunkSize; index += static_cast<int>(blockDim.x)) {
      const int row = index / kChunkSize;
      const int col = index % kChunkSize;
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        shared.aqk[index] = Aqk[(token * kLocalVHeads + hv) * kChunkSize + col];
      } else {
        shared.aqk[index] = qwen35_float_to_bf16(0.0f);
      }
    }
    __syncthreads();

    if (warp < kChunkOutputWarps) {
      const int tile_m = warp / kValueTilesPerBlock;
      const int tile_n = warp % kValueTilesPerBlock;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> output_acc;
      wmma::load_matrix_sync(
          output_acc,
          shared.accum + tile_m * 16 * kValueTile + tile_n * 16,
          kValueTile,
          wmma::mem_row_major);
#pragma unroll
      for (int kk = 0; kk < kChunkSize; kk += 16) {
        wmma::load_matrix_sync(frag_a, shared.aqk + tile_m * 16 * kChunkSize + kk, kChunkSize);
        wmma::load_matrix_sync(frag_b, shared.v_new + kk * kValueTile + tile_n * 16, kValueTile);
        wmma::mma_sync(output_acc, frag_a, frag_b, output_acc);
      }
      wmma::store_matrix_sync(
          shared.accum + tile_m * 16 * kValueTile + tile_n * 16,
          output_acc,
          kValueTile,
          wmma::mem_row_major);
    }
    __syncthreads();

    for (int index = tid; index < valid_rows * kValueTile; index += static_cast<int>(blockDim.x)) {
      const int row = index / kValueTile;
      const int vv = index % kValueTile;
      const int token = seq * seq_len + chunk_start + row;
      out[(token * kLocalVHeads + hv) * kHeadDimV + v_base + vv] =
          qwen35_float_to_bf16(shared.accum[index]);
    }
    for (int index = tid; index < kChunkSize * kHeadDimQK; index += static_cast<int>(blockDim.x)) {
      const int row = index / kHeadDimQK;
      const int kk = index % kHeadDimQK;
      if (row < valid_rows) {
        const int token = seq * seq_len + chunk_start + row;
        shared.matrix[index] = kg[(token * kLocalVHeads + hv) * kHeadDimQK + kk];
      } else {
        shared.matrix[index] = qwen35_float_to_bf16(0.0f);
      }
    }
    const int last_token = seq * seq_len + chunk_start + valid_rows - 1;
    const float chunk_decay = exp2f(g[last_token * kLocalVHeads + hv]);
    __syncthreads();

    for (int tile = warp; tile < (kHeadDimQK / 16) * kValueTilesPerBlock; tile += kChunkStateWarps) {
      const int tile_m = tile / kValueTilesPerBlock;
      const int tile_n = tile % kValueTilesPerBlock;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
      wmma::fill_fragment(frag_c, 0.0f);
#pragma unroll
      for (int tt = 0; tt < kChunkSize; tt += 16) {
        wmma::load_matrix_sync(frag_a, shared.matrix + tt * kHeadDimQK + tile_m * 16, kHeadDimQK);
        wmma::load_matrix_sync(frag_b, shared.v_new + tt * kValueTile + tile_n * 16, kValueTile);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
      }
      wmma::store_matrix_sync(
          shared.accum + tile_m * 16 * kValueTile + tile_n * 16,
          frag_c,
          kValueTile,
          wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = tid; index < kHeadDimQK * kValueTile; index += static_cast<int>(blockDim.x)) {
      const float updated = shared.accum[index] + chunk_decay * qwen35_bf16_to_float(shared.state[index]);
      shared.state[index] = qwen35_float_to_bf16(updated);
    }
    __syncthreads();
  }

  for (int index = tid; index < kHeadDimQK * kValueTile; index += static_cast<int>(blockDim.x)) {
    const int kk = index / kValueTile;
    const int vv = index % kValueTile;
    final_state[state_global_base + kk * kHeadDimV + v_base + vv] = qwen35_bf16_to_float(shared.state[index]);
  }
}

inline void launch_qwen35_chunk_preprocess(
    cudaStream_t stream,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const float* A_log,
    const float* dt_bias,
    const float* gate_raw,
    const float* beta_in,
    bool use_precomputed_gate,
    __nv_bfloat16* q_norm,
    __nv_bfloat16* k_norm,
    float* g,
    float* g_raw_output,
    float* beta,
    int32_t* unsafe_gate_flags,
    int32_t* cu_seqlens,
    int32_t* chunk_indices,
    int batch_size,
    int seq_len,
    int qk_heads,
    int v_heads) {
  const int vector_count = batch_size * seq_len * qk_heads;
  const int chunks = (seq_len + kChunkSize - 1) / kChunkSize;
  const int gate_blocks = batch_size * chunks * v_heads;
  constexpr int kNormVectorsPerBlock = CULA_QWEN35_PREPROCESS_THREADS / 32;
  const int norm_blocks = (vector_count + kNormVectorsPerBlock - 1) / kNormVectorsPerBlock;
  if (use_precomputed_gate) {
    qwen35_chunk_preprocess_fused_kernel<true><<<
        norm_blocks + gate_blocks, CULA_QWEN35_PREPROCESS_THREADS, 0, stream>>>(
        q, k, a, b, A_log, dt_bias, gate_raw, beta_in,
        q_norm, k_norm, g, g_raw_output, beta, unsafe_gate_flags,
        cu_seqlens, chunk_indices, batch_size, seq_len, qk_heads, v_heads,
        vector_count, gate_blocks, norm_blocks, chunks);
  } else {
    qwen35_chunk_preprocess_fused_kernel<false><<<
        norm_blocks + gate_blocks, CULA_QWEN35_PREPROCESS_THREADS, 0, stream>>>(
        q, k, a, b, A_log, dt_bias, gate_raw, beta_in,
        q_norm, k_norm, g, g_raw_output, beta, unsafe_gate_flags,
        cu_seqlens, chunk_indices, batch_size, seq_len, qk_heads, v_heads,
        vector_count, gate_blocks, norm_blocks, chunks);
  }
}

template <int kLocalVHeads>
inline void launch_qwen35_chunk_state_output(
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
  auto kernel_fn = &qwen35_chunk_state_output_kernel<kLocalVHeads>;
  constexpr size_t shared_bytes = sizeof(Qwen35ChunkStateOutputShared);
  cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  const int grid = batch_size * kLocalVHeads * (kHeadDimV / kValueTile);
  kernel_fn<<<grid, kChunkStateWarps * 32, shared_bytes, stream>>>(
      q_norm, g, Aqk, w, u, kg, initial_state, out, final_state,
      batch_size, seq_len, qk_heads, has_initial_state);
}


template <typename scalar_t, int kLocalVHeads, int kWarpsPerBlock>
__global__ void qwen35_scalar_kda_prefill_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
      int batch_size,
      int seq_len,
      int qk_heads,
      int sequence_count,
    bool is_varlen,
    bool has_initial_state,
    const float* __restrict__ precomputed_g,
    const float* __restrict__ precomputed_beta,
    const int32_t* __restrict__ unsafe_gate_flags) {
  __shared__ typename Qwen35ScalarKdaPrefillKernel<scalar_t, kLocalVHeads, kWarpsPerBlock>::SharedStorage storage;
  Qwen35ScalarKdaPrefillKernel<scalar_t, kLocalVHeads, kWarpsPerBlock>::run_device(
      q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      initial_state,
      cu_seqlens,
      out,
      final_state,
      batch_size,
      seq_len,
      qk_heads,
      sequence_count,
      is_varlen,
      has_initial_state,
      precomputed_g,
      precomputed_beta,
      unsafe_gate_flags,
      storage);
}

template <typename scalar_t, int kLocalVHeads, int kWarpsPerBlock>
void launch_qwen35_scalar_kda_prefill_kernel_variant(
    cudaStream_t stream,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    const int32_t* cu_seqlens,
    scalar_t* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    int sequence_count,
    bool is_varlen,
    bool has_initial_state) {
  using Kernel = Qwen35ScalarKdaPrefillKernel<scalar_t, kLocalVHeads, kWarpsPerBlock>;
  const auto grid = Kernel::grid_shape(sequence_count);
  const auto block = Kernel::block_shape();
  qwen35_scalar_kda_prefill_kernel<scalar_t, kLocalVHeads, kWarpsPerBlock><<<grid, block, 0, stream>>>(
      q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      initial_state,
      cu_seqlens,
      out,
      final_state,
      batch_size,
      seq_len,
      qk_heads,
      sequence_count,
      is_varlen,
      has_initial_state,
      nullptr,
      nullptr,
      nullptr);
}

template <typename scalar_t, int kLocalVHeads>
void launch_qwen35_scalar_kda_prefill_precomputed_fallback(
    cudaStream_t stream,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const float* g,
    const float* beta,
    const float* initial_state,
    scalar_t* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    bool has_initial_state,
    const int32_t* unsafe_gate_flags) {
  // Safe inputs make this launch an early-return guard.  A 16-warp CTA keeps
  // that fixed cost to only 8 CTAs per V head, while still providing an exact
  // recurrent overwrite for the rare unsafe head.
  constexpr int kFallbackWarps = 16;
  using Kernel = Qwen35ScalarKdaPrefillKernel<scalar_t, kLocalVHeads, kFallbackWarps>;
  qwen35_scalar_kda_prefill_kernel<scalar_t, kLocalVHeads, kFallbackWarps>
      <<<Kernel::grid_shape(batch_size), Kernel::block_shape(), 0, stream>>>(
          q,
          k,
          v,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          initial_state,
          nullptr,
          out,
          final_state,
          batch_size,
          seq_len,
          qk_heads,
          batch_size,
          false,
          has_initial_state,
          g,
          beta,
          unsafe_gate_flags);
}

template <typename scalar_t, int kLocalVHeads>
void launch_qwen35_scalar_kda_prefill_kernel(
    cudaStream_t stream,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    const int32_t* cu_seqlens,
    scalar_t* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    int sequence_count,
    bool is_varlen,
    bool has_initial_state) {
  launch_qwen35_scalar_kda_prefill_kernel_variant<scalar_t, kLocalVHeads, 4>(
      stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state,
      batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
}

} // namespace cula::qwen35::prefill::kernel
