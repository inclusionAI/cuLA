#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#if defined(CULA_SM100_ENABLED)
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
    bool use_tf32_inverse);
#endif

#if defined(CULA_SM90A_ENABLED)
std::tuple<torch::Tensor, torch::Tensor> kda_fwd_prefill(
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
#if defined(CULA_SM100_ENABLED)
    m.def("chunk_kda_fwd_intra_cuda", &ChunkKDAFwdIntra);
#endif
#if defined(CULA_SM90A_ENABLED)
    m.def("kda_fwd_prefill", &kda_fwd_prefill);
#endif
}
