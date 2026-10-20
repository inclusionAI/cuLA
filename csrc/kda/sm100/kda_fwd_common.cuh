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

#include "kda/sm100/kda_config.hpp"

namespace kda::sm100 {

// Presents a compact per-row scalar gate as the float4-addressable 2-D view
// used by the existing gated Q/K helpers. Each row stores four identical
// values; all logical K columns intentionally alias that four-float record.
struct ScalarGateView {
    float* ptr;

    __device__ __forceinline__ float&
    operator()(int row, int) const {
        return ptr[row * 4];
    }
};

// KDA forward kernels

// KDA forward intra-chunk kernel
void
run_kda_fwd_intra_sm100(KDA_fwd_intra_params& params, cudaStream_t stream);

void
run_kda_fwd_intra_sm100_qwen_scalar_g(KDA_fwd_intra_params& params, cudaStream_t stream);

// KDA forward recompute W & U kernel
void
run_kda_fwd_recomp_w_u_sm100(KDA_fwd_recomp_w_u_params& params, cudaStream_t stream);

void
run_kda_fwd_recomp_w_u_sm100_qwen_scalar_g(KDA_fwd_recomp_w_u_params& params, cudaStream_t stream);

}  // namespace kda::sm100
