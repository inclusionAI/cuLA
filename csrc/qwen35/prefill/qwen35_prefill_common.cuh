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

#include "qwen35/decode/qwen35_decode_common.cuh"

#include <ATen/core/TensorBody.h>

namespace cula::qwen35::prefill {

using decode::kHeadDimQK;
using decode::kHeadDimV;
using decode::kKDim;
using decode::kMixedQKVDim;
using decode::kNumQKHeads;
using decode::kNumVHeads;
using decode::kQDim;
using decode::kVDim;

struct LayoutPrefillParams {
  at::Tensor mixed_qkv_conv; // [N, local_conv_dim]
  at::Tensor a;              // [N, local_v_heads]
  at::Tensor b;              // [N, local_v_heads]
  at::Tensor q_rep;          // [N, local_v_heads, 128]
  at::Tensor k_rep;          // [N, local_v_heads, 128]
  at::Tensor v;              // [N, local_v_heads, 128]
  at::Tensor a_kernel;       // [N, local_v_heads]
  at::Tensor b_kernel;       // [N, local_v_heads]
};

struct ScalarKdaPrefillParams {
  at::Tensor q;                // [B, T, local_qk_heads, 128]
  at::Tensor k;                // [B, T, local_qk_heads, 128]
  at::Tensor v;                // [B, T, local_v_heads, 128]
  at::Tensor a;                // [B, T, local_v_heads]
  at::Tensor b;                // [B, T, local_v_heads]
  at::Tensor A_log;            // [local_v_heads], float32
  at::Tensor dt_bias;          // [local_v_heads], float32
  at::Tensor initial_state;    // [N, local_v_heads, 128, 128], float32, may be empty
  at::Tensor cu_seqlens;       // [N + 1], int32, may be empty
  at::Tensor out;              // [B, T, local_v_heads, 128]
  at::Tensor final_state;      // [N, local_v_heads, 128, 128], float32
};

// Core-only ABI used for apples-to-apples comparison with SGLang's
// TritonGDNKernel.extend.  g/beta are the already materialized per-token
// scalar gate and beta tensors; q/k normalization and chunk-local gate scan
// remain inside the CUDA prefill calculation.
struct ScalarKdaPrefillCoreParams {
  at::Tensor q;                // [B, T, local_qk_heads, 128], bf16
  at::Tensor k;                // [B, T, local_qk_heads, 128], bf16
  at::Tensor v;                // [B, T, local_v_heads, 128], bf16
  at::Tensor g;                // [B, T, local_v_heads], float32, natural-log gate
  at::Tensor beta;             // [B, T, local_v_heads], float32
  at::Tensor initial_state;    // [N, local_v_heads, 128, 128], float32, may be empty
  at::Tensor cu_seqlens;       // [N + 1], int32, may be empty
  at::Tensor out;              // [B, T, local_v_heads, 128], bf16
  at::Tensor final_state;      // [N, local_v_heads, 128, 128], float32
};

// All local V-head counts produced by the downloaded Qwen3.5/Qwen3.6
// configurations at TP={1,2,4,8}.  The scalar path accepts compact native-GVA
// Q/K heads and maps each V head to its Q/K group inside the kernel.
inline constexpr bool is_supported_scalar_prefill_v_heads(int local_v_heads) {
  return local_v_heads == 64 || local_v_heads == 48 || local_v_heads == 32 ||
      local_v_heads == 24 || local_v_heads == 16 || local_v_heads == 12 ||
      local_v_heads == 8 || local_v_heads == 6 || local_v_heads == 4 ||
      local_v_heads == 2;
}

void run_qwen35_scalar_kda_prefill(ScalarKdaPrefillParams& params);
void run_qwen35_scalar_kda_prefill_core(ScalarKdaPrefillCoreParams& params);
void run_qwen35_layout_prefill(LayoutPrefillParams& params);

} // namespace cula::qwen35::prefill

namespace cula::qwen35::prefill::sm90 {

void qwen35_chunk_qk_prefill_sm90(at::Tensor q, at::Tensor k, at::Tensor out);

} // namespace cula::qwen35::prefill::sm90
