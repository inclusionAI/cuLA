#include <cuda_runtime.h>

#include "kda_api.hpp"

namespace {

template <int K_DIM, int V_DIM, int CHUNK>
cudaError_t launch_kda_prefill_impl(
    const kda::api::KdaPrefillIO<float>& io,
    cudaStream_t stream) {
  using Params = kda::KdaPrefillParams<float, K_DIM, V_DIM, CHUNK>;
  Params params = kda::make_prefill_params<float, K_DIM, V_DIM, CHUNK>(
      io.q_ptr,
      io.k_ptr,
      io.v_ptr,
      io.g_ptr,
      io.beta_ptr,
      io.w_ptr,
      io.u_ptr,
      io.o_ptr,
      io.batch_size,
      io.num_heads,
      io.seq_len);
  kda::api::launch_kda_prefill_kernel(params, stream);
  return cudaGetLastError();
}

}  // namespace

namespace kda::api {

cudaError_t launch_kda_prefill_f32(const KdaPrefillIO<float>& io,
                                   cudaStream_t stream) {
  if (io.head_dim == 64 && io.value_dim == 64 && io.chunk_size == 64) {
    return launch_kda_prefill_impl<64, 64, 64>(io, stream);
  }
  if (io.head_dim == 64 && io.value_dim == 64 && io.chunk_size == 32) {
    return launch_kda_prefill_impl<64, 64, 32>(io, stream);
  }
  if (io.head_dim == 128 && io.value_dim == 128 && io.chunk_size == 64) {
    return launch_kda_prefill_impl<128, 128, 64>(io, stream);
  }
  if (io.head_dim == 128 && io.value_dim == 128 && io.chunk_size == 32) {
    return launch_kda_prefill_impl<128, 128, 32>(io, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace kda::api

extern "C" cudaError_t kda_prefill_f32(
    const kda::api::KdaPrefillIO<float>* io,
    cudaStream_t stream) {
  if (io == nullptr) {
    return cudaErrorInvalidValue;
  }
  try {
    return kda::api::launch_kda_prefill(*io, stream);
  } catch (...) {
    return cudaErrorInvalidValue;
  }
}
