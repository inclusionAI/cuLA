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

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <cstdint>

namespace cula::qwen35::decode {

inline constexpr int kNumQKHeads = 16;
inline constexpr int kNumVHeads = 48;
inline constexpr int kHeadDimQK = 128;
inline constexpr int kHeadDimV = 128;
inline constexpr int kConvKernelSize = 4;
inline constexpr int kQDim = kNumQKHeads * kHeadDimQK;
inline constexpr int kKDim = kNumQKHeads * kHeadDimQK;
inline constexpr int kVDim = kNumVHeads * kHeadDimV;
inline constexpr int kMixedQKVDim = kQDim + kKDim + kVDim;

inline constexpr int local_qk_heads_from_v_heads(int local_v_heads) {
  return local_v_heads / (kNumVHeads / kNumQKHeads);
}

inline constexpr int local_q_dim(int local_qk_heads) {
  return local_qk_heads * kHeadDimQK;
}

inline constexpr int local_v_dim(int local_v_heads) {
  return local_v_heads * kHeadDimV;
}

inline constexpr int local_mixed_qkv_dim(int local_qk_heads, int local_v_heads) {
  return 2 * local_q_dim(local_qk_heads) + local_v_dim(local_v_heads);
}

inline constexpr bool is_supported_local_v_heads(int local_v_heads) {
  return local_v_heads == 48 || local_v_heads == 24 || local_v_heads == 12 || local_v_heads == 6;
}

template <int kLocalVHeads_>
struct Qwen35DecodeLocalShape {
  static_assert(is_supported_local_v_heads(kLocalVHeads_), "Unsupported Qwen3.5 local V-head count.");
  static constexpr int kLocalVHeads = kLocalVHeads_;
  static constexpr int kLocalQKHeads = local_qk_heads_from_v_heads(kLocalVHeads);
  static constexpr int kRepeatFactor = kLocalVHeads / kLocalQKHeads;
  static constexpr int kLocalQDim = local_q_dim(kLocalQKHeads);
  static constexpr int kLocalKDim = local_q_dim(kLocalQKHeads);
  static constexpr int kLocalVDim = local_v_dim(kLocalVHeads);
  static constexpr int kLocalMixedQKVDim = local_mixed_qkv_dim(kLocalQKHeads, kLocalVHeads);

  // Decode shape policy. Head dimension is fixed at 128 for Qwen3.5, but keep
  // these knobs with the local-head traits so future TP-shape tuning has one
  // place to specialize.
  static constexpr int kLayoutVec = 4;
  static constexpr int kLayoutThreads = kHeadDimQK / kLayoutVec;
  static constexpr int kKdaThreads = 128;
  static constexpr int kKdaTileV = 16;
  static constexpr int kKdaTileK = 16;

  static_assert(kLocalVHeads % kLocalQKHeads == 0);
  static_assert(kHeadDimQK == kHeadDimV);
  static_assert(kHeadDimQK % kLayoutVec == 0);
  static_assert(kHeadDimV % kKdaTileV == 0);
  static_assert(kHeadDimQK % kKdaTileK == 0);
};

struct ConvDecodeParams {
  at::Tensor mixed_qkv;   // [B, 1, local_conv_dim]
  at::Tensor conv_state;  // [B, local_conv_dim, 4]
  at::Tensor conv_weight; // [local_conv_dim, 4]
  at::Tensor out;         // [B, 1, local_conv_dim]
};

struct LayoutDecodeParams {
  at::Tensor mixed_qkv_conv; // [N, local_conv_dim]
  at::Tensor a;              // [N, local_v_heads]
  at::Tensor b;              // [N, local_v_heads]
  at::Tensor q_rep;          // [N, local_v_heads, 128]
  at::Tensor k_rep;          // [N, local_v_heads, 128]
  at::Tensor v;              // [N, local_v_heads, 128]
  at::Tensor a_kernel;       // [N, local_v_heads]
  at::Tensor b_kernel;       // [N, local_v_heads]
};

struct ScalarKdaDecodeParams {
  // Dtype contract for the first implementation:
  // - activations / outputs: half or bf16
  //     q_rep, k_rep, v, a_kernel, b_kernel, out
  // - recurrent parameters / state: float32
  //     A_log, dt_bias, recurrent_state
  at::Tensor q_rep;            // [N, local_v_heads, 128]
  at::Tensor k_rep;            // [N, local_v_heads, 128]
  at::Tensor v;                // [N, local_v_heads, 128]
  at::Tensor a_kernel;         // [N, local_v_heads]
  at::Tensor b_kernel;         // [N, local_v_heads]
  at::Tensor A_log;            // [local_v_heads], float32
  at::Tensor dt_bias;          // [local_v_heads], float32
  at::Tensor recurrent_state;  // [pool, local_v_heads, 128, 128], float32
  at::Tensor pool_idx;         // [N], int32
  at::Tensor out;              // [N, local_v_heads, 128]
};

struct LayoutScalarKdaDecodeParams {
  at::Tensor mixed_qkv_conv;    // [N, local_conv_dim]
  at::Tensor a;                 // [N, local_v_heads]
  at::Tensor b;                 // [N, local_v_heads]
  at::Tensor A_log;             // [local_v_heads], float32
  at::Tensor dt_bias;           // [local_v_heads], float32
  at::Tensor recurrent_state;   // [pool, local_v_heads, 128, 128], float32
  at::Tensor pool_idx;          // [N], int32
  at::Tensor out;               // [N, local_v_heads, 128]
};

void run_qwen35_conv1d_decode(ConvDecodeParams& params);
void run_qwen35_layout_decode(LayoutDecodeParams& params);
void run_qwen35_scalar_kda_decode(ScalarKdaDecodeParams& params);
void run_qwen35_layout_scalar_kda_decode(LayoutScalarKdaDecodeParams& params);

} // namespace cula::qwen35::decode
