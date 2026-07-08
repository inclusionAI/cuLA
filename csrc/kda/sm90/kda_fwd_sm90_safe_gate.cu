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

#include <cute/numeric/numeric_types.hpp>
#include <cutlass/arch/arch.h>

#include "kda/sm90/prefill_kernel_kda_fwd_sm90.cuh"
#include "kda/sm90/utils/common.hpp"

namespace kda::sm90 {

using namespace cute;
using bf16 = cute::bfloat16_t;

#define INSTANTIATE_GBAI(NeedsBeta, NeedsAlpha, InitState, SafeGate, TBeta) \
    template void launch_kda_fwd_prefill_kernel_gbai<                       \
        NeedsBeta,                                                          \
        NeedsAlpha,                                                         \
        InitState,                                                          \
        SafeGate,                                                           \
        cutlass::arch::Sm90,                                                \
        bf16,                                                               \
        bf16,                                                               \
        float,                                                              \
        TBeta>(                                                             \
        cudaStream_t,                                                       \
        bf16*,                                                              \
        float*,                                                             \
        bf16 const*,                                                        \
        bf16 const*,                                                        \
        bf16 const*,                                                        \
        float const*,                                                       \
        float const*,                                                       \
        TBeta const*,                                                       \
        int32_t const*,                                                     \
        uint8_t*,                                                           \
        int32_t,                                                            \
        int32_t,                                                            \
        int32_t,                                                            \
        int32_t,                                                            \
        int64_t,                                                            \
        float,                                                              \
        int32_t,                                                            \
        int32_t const*,                                                     \
        int32_t const*,                                                     \
        int32_t)

INSTANTIATE_GBAI(true, true, false, true, float);
INSTANTIATE_GBAI(true, true, true, true, float);
INSTANTIATE_GBAI(true, true, false, true, bf16);
INSTANTIATE_GBAI(true, true, true, true, bf16);

#undef INSTANTIATE_GBAI

}  // namespace kda::sm90
