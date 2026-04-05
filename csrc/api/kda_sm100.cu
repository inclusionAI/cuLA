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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cutlass.h>
#include <torch/python.h>

#include "kda/sm100/kda_fwd_common.cuh"

void
ChunkKDAFwdIntra(
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
    params.tile_scheduler_params =
        StaticPersistentTileScheduler::Params{tile_num, params.h, params.num_sm, (int*)tile_counter.data_ptr()};

    kda::sm100::run_kda_fwd_intra_sm100(params, at::cuda::getCurrentCUDAStream());
}

void
ChunkKDAFwdRecompWU(
    at::Tensor k,
    at::Tensor v,
    at::Tensor beta,
    at::Tensor A,
    at::Tensor g,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor w_out,
    at::Tensor u_out,
    at::Tensor kg_out,
    int chunk_size,
    std::optional<at::Tensor> q,
    std::optional<at::Tensor> qg_out) {
    KDA_fwd_recomp_w_u_params params;
    params.total_len = k.size(0) * k.size(1);
    params.b = cu_seqlens.size(0) - 1;
    params.h = k.size(2);
    params.d = k.size(3);
    params.chunk_size = chunk_size;
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.g_ptr = g.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    params.w_out_ptr = w_out.data_ptr();
    params.u_out_ptr = u_out.data_ptr();
    params.kg_out_ptr = kg_out.data_ptr();
    const bool has_q = q.has_value();
    const bool has_qg_out = qg_out.has_value();
    TORCH_CHECK(
        has_q == has_qg_out, "ChunkKDAFwdRecompWU: q and qg_out must either both be provided or both be omitted.");
    params.store_qg = has_q && has_qg_out;
    params.q_ptr = params.store_qg ? q->data_ptr() : nullptr;
    params.qg_out_ptr = params.store_qg ? qg_out->data_ptr() : nullptr;
    params.shape_wukg = cute::make_shape(params.total_len, params.d, params.h);
    params.stride_wukg = cute::make_stride(params.d * params.h, cute::_1{}, params.d);
    int tile_num = chunk_indices.size(0);
    auto device_prop = at::cuda::getCurrentDeviceProperties();
    params.num_sm = device_prop->multiProcessorCount;
    params.tile_scheduler_params = StaticPersistentTileScheduler::Params{tile_num, params.h, params.num_sm, nullptr};

    kda::sm100::run_kda_fwd_recomp_w_u_sm100(params, at::cuda::getCurrentCUDAStream());
}