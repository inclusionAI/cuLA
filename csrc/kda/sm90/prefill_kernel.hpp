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

#include <cstdint>

#include <cuda_runtime_api.h>

namespace kda::sm90 {

template <
    typename ArchTag,  // TODO: hide this
    typename TO,
    typename TQKV,
    typename TState,
    typename TBeta = float>
void
launch_kda_fwd_prefill_kernel(
    cudaStream_t stream,
    TO* output,
    TState* output_state,
    TQKV const* q,
    TQKV const* k,
    TQKV const* v,
    TState const* input_state,
    float const* alpha,
    TBeta const* beta,
    int32_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    bool safe_gate,
    int32_t sm_count = 0,
    /// Number of physical Q/K heads. Q and K always share one head count in
    /// KDA because they interact in the Q*K^T matmul. When 0 (default) or
    /// equal to num_heads the kernel runs as plain MHA. When num_heads >
    /// num_k_heads and num_heads % num_k_heads == 0, the kernel runs as KDA's
    /// "multi-value" attention flavor: k_group_size = num_heads / num_k_heads
    /// V/state heads share each physical Q/K head. Q and K tensors are
    /// expected to be laid out as [total_seqlen, num_k_heads, head_size]; V
    /// and O keep the [total_seqlen, num_heads, head_size] layout. This
    /// matches e.g. Qwen3.5-A3B's Gated DeltaNet.
    int32_t num_k_heads = 0);

}  // namespace kda::sm90
