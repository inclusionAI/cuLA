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
#include <torch/nn/functional.h>
#include <torch/python.h>

#include "qwen35/decode/qwen35_decode_common.cuh"
#include "qwen35/prefill/qwen35_prefill_common.cuh"

#if defined(CULA_SM100_ENABLED) || defined(CULA_SM103_ENABLED)
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
    bool unified_gref);
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
    std::optional<at::Tensor> qg_out);
#endif

#if defined(CULA_SM90A_ENABLED)
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
kda_fwd_prefill(
    std::optional<torch::Tensor> output_,
    std::optional<torch::Tensor> output_state_,
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    std::optional<torch::Tensor> input_state_,
    std::optional<torch::Tensor> alpha_,
    std::optional<torch::Tensor> beta_,
    torch::Tensor const& cu_seqlens,
    torch::Tensor workspace_buffer,
    float scale,
    bool output_final_state,
    bool safe_gate,
    std::optional<torch::Tensor> cp_seq_map_,
    std::optional<torch::Tensor> raw_cu_seqlens_);
#endif

void
qwen35_conv1d_decode(
    at::Tensor mixed_qkv,
    at::Tensor conv_state,
    at::Tensor conv_weight,
    at::Tensor out) {
    cula::qwen35::decode::ConvDecodeParams params{
        mixed_qkv,
        conv_state,
        conv_weight,
        out,
    };
    cula::qwen35::decode::run_qwen35_conv1d_decode(params);
}

void
qwen35_layout_decode(
    at::Tensor mixed_qkv_conv,
    at::Tensor a,
    at::Tensor b,
    at::Tensor q_rep,
    at::Tensor k_rep,
    at::Tensor v,
    at::Tensor a_kernel,
    at::Tensor b_kernel) {
    cula::qwen35::decode::LayoutDecodeParams params{
        mixed_qkv_conv,
        a,
        b,
        q_rep,
        k_rep,
        v,
        a_kernel,
        b_kernel,
    };
    cula::qwen35::decode::run_qwen35_layout_decode(params);
}

void
qwen35_scalar_kda_decode(
    at::Tensor q_rep,
    at::Tensor k_rep,
    at::Tensor v,
    at::Tensor a_kernel,
    at::Tensor b_kernel,
    at::Tensor A_log,
    at::Tensor dt_bias,
    at::Tensor recurrent_state,
    at::Tensor pool_idx,
    at::Tensor out) {
    cula::qwen35::decode::ScalarKdaDecodeParams params{
        q_rep,
        k_rep,
        v,
        a_kernel,
        b_kernel,
        A_log,
        dt_bias,
        recurrent_state,
        pool_idx,
        out,
    };
    cula::qwen35::decode::run_qwen35_scalar_kda_decode(params);
}

void
qwen35_layout_scalar_kda_decode(
    at::Tensor mixed_qkv_conv,
    at::Tensor a,
    at::Tensor b,
    at::Tensor A_log,
    at::Tensor dt_bias,
    at::Tensor recurrent_state,
    at::Tensor pool_idx,
    at::Tensor out) {
    cula::qwen35::decode::LayoutScalarKdaDecodeParams params{
        mixed_qkv_conv,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
        pool_idx,
        out,
    };
    cula::qwen35::decode::run_qwen35_layout_scalar_kda_decode(params);
}

void
qwen35_scalar_kda_prefill(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor A_log,
    at::Tensor dt_bias,
    at::Tensor initial_state,
    at::Tensor cu_seqlens,
    at::Tensor out,
    at::Tensor final_state) {
    cula::qwen35::prefill::ScalarKdaPrefillParams params{
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
    };
    cula::qwen35::prefill::run_qwen35_scalar_kda_prefill(params);
}

void
qwen35_layout_prefill(
    at::Tensor mixed_qkv_conv,
    at::Tensor a,
    at::Tensor b,
    at::Tensor q_rep,
    at::Tensor k_rep,
    at::Tensor v,
    at::Tensor a_kernel,
    at::Tensor b_kernel) {
    cula::qwen35::prefill::LayoutPrefillParams params{
        mixed_qkv_conv,
        a,
        b,
        q_rep,
        k_rep,
        v,
        a_kernel,
        b_kernel,
    };
    cula::qwen35::prefill::run_qwen35_layout_prefill(params);
}

void
qwen35_chunk_qk_prefill_sm90(at::Tensor q, at::Tensor k, at::Tensor out) {
    cula::qwen35::prefill::sm90::qwen35_chunk_qk_prefill_sm90(q, k, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuLA";
#if defined(CULA_SM100_ENABLED) || defined(CULA_SM103_ENABLED)
    m.def("chunk_kda_fwd_intra_cuda", &ChunkKDAFwdIntra);
    m.def("recompute_w_u_cuda", &ChunkKDAFwdRecompWU);
#endif
#if defined(CULA_SM90A_ENABLED)
    m.def(
        "kda_fwd_prefill",
        &kda_fwd_prefill,
        pybind11::arg("output_"),
        pybind11::arg("output_state_"),
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"),
        pybind11::arg("input_state_"),
        pybind11::arg("alpha_"),
        pybind11::arg("beta_"),
        pybind11::arg("cu_seqlens"),
        pybind11::arg("workspace_buffer"),
        pybind11::arg("scale"),
        pybind11::arg("output_final_state"),
        pybind11::arg("safe_gate"),
        pybind11::arg("cp_seq_map_") = std::nullopt,
        pybind11::arg("raw_cu_seqlens_") = std::nullopt);
#endif
    m.def("qwen35_conv1d_decode", &qwen35_conv1d_decode);
    m.def("qwen35_layout_decode", &qwen35_layout_decode);
    m.def("qwen35_scalar_kda_decode", &qwen35_scalar_kda_decode);
    m.def("qwen35_layout_scalar_kda_decode", &qwen35_layout_scalar_kda_decode);
    m.def("qwen35_layout_prefill", &qwen35_layout_prefill);
    m.def("qwen35_scalar_kda_prefill", &qwen35_scalar_kda_prefill);
    m.def("qwen35_chunk_qk_prefill_sm90", &qwen35_chunk_qk_prefill_sm90);
}
