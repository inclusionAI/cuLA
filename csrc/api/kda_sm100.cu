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

#include <torch/python.h>
#include "kda/sm100/kda_fwd_common.cuh"
#include "cutlass/cutlass.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

void ChunkKDAFwdIntra(
    at::Tensor q,
    at::Tensor k,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor Aqk_out,
    at::Tensor Akk_out,
    at::Tensor tile_counter,
    float scale,
    int chunk_size,
    bool use_tf32_inverse,
    bool unified_gref) {

    KDA_fwd_intra_params params;
    params.total_q_len = q.size(0) * q.size(1);
    params.b = cu_seqlens.size(0) - 1;
    params.h = q.size(2);
    params.d = q.size(3);
    params.chunk_size = chunk_size;
    params.scale = scale;
    params.use_tf32_inverse = use_tf32_inverse;
    params.unified_gref = unified_gref;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.g_ptr = g.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    params.Aqk_out_ptr = Aqk_out.data_ptr();
    params.Akk_out_ptr = Akk_out.data_ptr();
    params.shape_Akk = cute::make_shape(params.total_q_len, params.chunk_size, params.h);
    params.stride_Akk = cute::make_stride(params.chunk_size * params.h, cute::_1{}, params.chunk_size);
    int tile_num = chunk_indices.size(0);
    auto device_prop = at::cuda::getCurrentDeviceProperties();
    params.num_sm = device_prop->multiProcessorCount;
    params.tile_scheduler_params = StaticPersistentTileScheduler::Params{tile_num, params.h, params.num_sm, (int*)tile_counter.data_ptr()};

    kda::sm100::run_kda_fwd_intra_sm100(params, at::cuda::getCurrentCUDAStream());
}