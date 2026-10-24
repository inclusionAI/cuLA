#include <torch/extension.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <tuple>

#include "kda_prefill_io.hpp"

namespace {

void check_supported_dims(int64_t k, int64_t v, int64_t chunk) {
  // (128,128,64) is not supported by the current device kernel configuration.
  const bool ok =
      (k == 64 && v == 64 && (chunk == 32 || chunk == 64)) ||
      (k == 128 && v == 128 && chunk == 32);
  TORCH_CHECK(
      ok,
      "kda_prefill: unsupported (K,V,chunk_size)=(",
      k,
      ",",
      v,
      ",",
      chunk,
      "); supported: (64,64,32|64), (128,128,32)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> kda_prefill_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    int64_t chunk_size,
    c10::optional<at::Tensor> w_opt,
    c10::optional<at::Tensor> u_opt,
    c10::optional<at::Tensor> o_opt) {
  TORCH_CHECK(q.is_cuda(), "kda_prefill: q must be CUDA");
  TORCH_CHECK(
      q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
          v.scalar_type() == at::kFloat && g.scalar_type() == at::kFloat &&
          beta.scalar_type() == at::kFloat,
      "kda_prefill: all tensors must be float32");

  TORCH_CHECK(q.dim() == 4, "kda_prefill: q must be [B,H,T,K]");
  const int64_t B = q.size(0);
  const int64_t H = q.size(1);
  const int64_t T = q.size(2);
  const int64_t K = q.size(3);
  TORCH_CHECK(
      k.sizes() == q.sizes(), "kda_prefill: k must match q shape ", q.sizes());
  TORCH_CHECK(
      g.sizes() == q.sizes(), "kda_prefill: g must match q shape ", q.sizes());
  TORCH_CHECK(
      v.size(0) == B && v.size(1) == H && v.size(2) == T,
      "kda_prefill: v must be [B,H,T,V] with same B,H,T as q");
  const int64_t V = v.size(3);
  TORCH_CHECK(
      beta.sizes() == at::IntArrayRef({B, H, T}),
      "kda_prefill: beta must be [B,H,T]");

  TORCH_CHECK(chunk_size > 0, "kda_prefill: chunk_size must be positive");
  check_supported_dims(K, V, chunk_size);

  TORCH_CHECK(q.is_contiguous(), "kda_prefill: q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "kda_prefill: k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "kda_prefill: v must be contiguous");
  TORCH_CHECK(g.is_contiguous(), "kda_prefill: g must be contiguous");
  TORCH_CHECK(beta.is_contiguous(), "kda_prefill: beta must be contiguous");

  const int64_t num_chunks = (T + chunk_size - 1) / chunk_size;
  auto opts = q.options();

  at::Tensor w;
  at::Tensor u;
  at::Tensor o;
  if (w_opt.has_value()) {
    w = *w_opt;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kFloat, "kda_prefill: w");
    TORCH_CHECK(
        w.sizes() == at::IntArrayRef({B, H, num_chunks, chunk_size, K}),
        "kda_prefill: w must be [B,H,num_chunks,C,K]");
    TORCH_CHECK(w.is_contiguous(), "kda_prefill: w must be contiguous");
  } else {
    w = at::empty({B, H, num_chunks, chunk_size, K}, opts);
  }
  if (u_opt.has_value()) {
    u = *u_opt;
    TORCH_CHECK(u.is_cuda() && u.scalar_type() == at::kFloat, "kda_prefill: u");
    TORCH_CHECK(
        u.sizes() == at::IntArrayRef({B, H, num_chunks, chunk_size, V}),
        "kda_prefill: u must be [B,H,num_chunks,C,V]");
    TORCH_CHECK(u.is_contiguous(), "kda_prefill: u must be contiguous");
  } else {
    u = at::empty({B, H, num_chunks, chunk_size, V}, opts);
  }
  if (o_opt.has_value()) {
    o = *o_opt;
    TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kFloat, "kda_prefill: o");
    TORCH_CHECK(
        o.sizes() == at::IntArrayRef({B, H, T, V}),
        "kda_prefill: o must be [B,H,T,V]");
    TORCH_CHECK(o.is_contiguous(), "kda_prefill: o must be contiguous");
  } else {
    o = at::empty({B, H, T, V}, opts);
  }

  kda::api::KdaPrefillIO<float> io{};
  io.q_ptr = q.data_ptr<float>();
  io.k_ptr = k.data_ptr<float>();
  io.v_ptr = v.data_ptr<float>();
  io.g_ptr = g.data_ptr<float>();
  io.beta_ptr = beta.data_ptr<float>();
  io.w_ptr = w.data_ptr<float>();
  io.u_ptr = u.data_ptr<float>();
  io.o_ptr = o.data_ptr<float>();
  io.batch_size = static_cast<int>(B);
  io.num_heads = static_cast<int>(H);
  io.seq_len = static_cast<int>(T);
  io.head_dim = static_cast<int>(K);
  io.value_dim = static_cast<int>(V);
  io.chunk_size = static_cast<int>(chunk_size);
  io.num_chunks = static_cast<int>(num_chunks);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device());
  cudaError_t err = kda_prefill_f32(&io, stream);
  TORCH_CHECK(
      err == cudaSuccess,
      "kda_prefill: ",
      cudaGetErrorString(err),
      " (check K,V,chunk_size, GPU sm>=89, and tensor layouts)");
  return std::make_tuple(std::move(o), std::move(w), std::move(u));
}

}  // namespace

TORCH_LIBRARY(kda_prefill, m) {
  m.def(
      "forward(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int chunk_size, Tensor? w, Tensor? u, Tensor? o) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(kda_prefill, CUDA, m) {
  m.impl("forward", &kda_prefill_forward_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
