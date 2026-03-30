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

#include "kda/sm100/kda_fwd_common.cuh"
#include "kda/sm100/kda_fwd_intra_kernel_sm100.hpp"
#include "kda/sm100/kda_fwd_recomp_w_u_kernel_sm100.hpp"

namespace kda::sm100 {

void
run_kda_fwd_intra_sm100(KDA_fwd_intra_params& params, cudaStream_t stream) {
    kda::sm100::run_kda_fwd_intra_sm100_impl(params, stream);
}

void
run_kda_fwd_recomp_w_u_sm100(KDA_fwd_recomp_w_u_params& params, cudaStream_t stream) {
    kda::sm100::run_kda_fwd_recomp_w_u_sm100_impl(params, stream);
}

}  // namespace kda::sm100
