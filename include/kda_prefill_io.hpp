#pragma once
#include <cuda_runtime.h>

namespace kda {
namespace api {

// Host-side prefill IO contract (shared by CUDA launcher and PyTorch binding).
//
// Inputs:
//   q/k/g: [B, H, T, K]
//   v:     [B, H, T, V]
//   beta:  [B, H, T]
//
// Intermediates:
//   w: [B, H, num_chunks, C, K]
//   u: [B, H, num_chunks, C, V]
//
// Output:
//   o: [B, H, T, V]
template <typename T>
struct KdaPrefillIO {
  const T* q_ptr = nullptr;
  const T* k_ptr = nullptr;
  const T* v_ptr = nullptr;
  const T* g_ptr = nullptr;
  const T* beta_ptr = nullptr;

  T* w_ptr = nullptr;
  T* u_ptr = nullptr;
  T* o_ptr = nullptr;

  int batch_size = 0;  // B
  int num_heads = 0;   // H
  int seq_len = 0;     // T
  int head_dim = 0;    // K
  int value_dim = 0;   // V
  int chunk_size = 0;  // C
  int num_chunks = 0;
};

}  // namespace api
}  // namespace kda

extern "C" cudaError_t kda_prefill_f32(
    const kda::api::KdaPrefillIO<float>* io,
    cudaStream_t stream);
