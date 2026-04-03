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
std::tuple<torch::Tensor, torch::Tensor>
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
    bool safe_gate);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuLA";
#if defined(CULA_SM100_ENABLED) || defined(CULA_SM103_ENABLED)
    m.def("chunk_kda_fwd_intra_cuda", &ChunkKDAFwdIntra);
    m.def("recompute_w_u_cuda", &ChunkKDAFwdRecompWU);
#endif
#if defined(CULA_SM90A_ENABLED)
    m.def("kda_fwd_prefill", &kda_fwd_prefill);
#endif
}
