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
    // GVA: Q/K are in h_qk head space (from q.size(2)); g/beta/Aqk/Akk are in h_v head
    // space (from g.size(2)). When HV == HQK, heads_per_group == 1 and behaviour matches
    // the pre-GVA path.
    params.h_qk = q.size(2);
    params.h_v = g.size(2);
    TORCH_CHECK(
        k.size(2) == params.h_qk,
        "ChunkKDAFwdIntra: k.size(2) (",
        k.size(2),
        ") must match q.size(2) (",
        params.h_qk,
        ") under GVA (Q/K share h_qk).");
    TORCH_CHECK(
        beta.size(-1) == params.h_v,
        "ChunkKDAFwdIntra: beta.size(-1) (",
        beta.size(-1),
        ") must equal h_v (",
        params.h_v,
        ").");
    TORCH_CHECK(
        params.h_qk > 0 && params.h_v > 0 && params.h_v % params.h_qk == 0,
        "ChunkKDAFwdIntra: h_v (",
        params.h_v,
        ") must be a positive multiple of h_qk (",
        params.h_qk,
        ").");
    params.heads_per_group = params.h_v / params.h_qk;
    params.d = q.size(3);
    params.chunk_size = chunk_size;
    params.scale = scale;
    params.use_tf32_inverse = use_tf32_inverse;
    params.unified_gref = unified_gref;
    TORCH_CHECK(
        beta.dtype() == torch::kFloat32 || beta.dtype() == torch::kBFloat16,
        "beta must be float32 or bfloat16, got ",
        beta.dtype());
    params.is_beta_bf16 = (beta.dtype() == torch::kBFloat16);
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.g_ptr = g.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    params.Aqk_out_ptr = Aqk_out.data_ptr();
    params.Akk_out_ptr = Akk_out.data_ptr();
    // Akk is laid out per v-head: (total_len, chunk_size, h_v).
    params.shape_Akk = cute::make_shape(params.total_q_len, params.chunk_size, params.h_v);
    params.stride_Akk = cute::make_stride(params.chunk_size * params.h_v, cute::_1{}, params.chunk_size);
    int tile_num = chunk_indices.size(0);
    auto device_prop = at::cuda::getCurrentDeviceProperties();
    params.num_sm = device_prop->multiProcessorCount;
    // Tiles are enumerated in v-head space.
    params.tile_scheduler_params = StaticPersistentTileScheduler::Params{
        tile_num, params.h_v, params.heads_per_group, params.num_sm, (int*)tile_counter.data_ptr()};

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
    // GVA: K (and optional Q) live in h_qk space; V/G/beta/A/w/u/kg/qg live in h_v space.
    params.h_qk = k.size(2);
    params.h_v = v.size(2);
    TORCH_CHECK(
        g.size(2) == params.h_v,
        "ChunkKDAFwdRecompWU: g.size(2) (",
        g.size(2),
        ") must equal v.size(2) (",
        params.h_v,
        ").");
    TORCH_CHECK(
        beta.size(-1) == params.h_v,
        "ChunkKDAFwdRecompWU: beta.size(-1) (",
        beta.size(-1),
        ") must equal h_v (",
        params.h_v,
        ").");
    TORCH_CHECK(
        params.h_qk > 0 && params.h_v > 0 && params.h_v % params.h_qk == 0,
        "ChunkKDAFwdRecompWU: h_v (",
        params.h_v,
        ") must be a positive multiple of h_qk (",
        params.h_qk,
        ").");
    params.heads_per_group = params.h_v / params.h_qk;
    params.d = k.size(3);
    params.chunk_size = chunk_size;
    TORCH_CHECK(
        beta.dtype() == torch::kFloat32 || beta.dtype() == torch::kBFloat16,
        "beta must be float32 or bfloat16, got ",
        beta.dtype());
    params.is_beta_bf16 = (beta.dtype() == torch::kBFloat16);
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
    if (params.store_qg) {
        TORCH_CHECK(
            q->size(2) == params.h_qk,
            "ChunkKDAFwdRecompWU: q.size(2) (",
            q->size(2),
            ") must equal h_qk (",
            params.h_qk,
            ").");
        TORCH_CHECK(
            qg_out->size(2) == params.h_v,
            "ChunkKDAFwdRecompWU: qg_out.size(2) (",
            qg_out->size(2),
            ") must equal h_v (",
            params.h_v,
            ").");
    }
    params.q_ptr = params.store_qg ? q->data_ptr() : nullptr;
    params.qg_out_ptr = params.store_qg ? qg_out->data_ptr() : nullptr;
    // w/u/kg/qg are per v-head: (total_len, d, h_v).
    params.shape_wukg = cute::make_shape(params.total_len, params.d, params.h_v);
    params.stride_wukg = cute::make_stride(params.d * params.h_v, cute::_1{}, params.d);
    int tile_num = chunk_indices.size(0);
    auto device_prop = at::cuda::getCurrentDeviceProperties();
    params.num_sm = device_prop->multiProcessorCount;
    params.tile_scheduler_params =
        StaticPersistentTileScheduler::Params{tile_num, params.h_v, params.heads_per_group, params.num_sm, nullptr};

    kda::sm100::run_kda_fwd_recomp_w_u_sm100(params, at::cuda::getCurrentCUDAStream());
}