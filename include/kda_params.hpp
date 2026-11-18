#pragma once

#include <cuda_runtime.h>

namespace kda {

template <typename T, int K_DIM_SIZE, int V_DIM_SIZE, int CHUNK_SIZE>
struct KdaPrefillParams {
  using element_type = T;
  static constexpr int K_DIM = K_DIM_SIZE;
  static constexpr int V_DIM = V_DIM_SIZE;
  static constexpr int C = CHUNK_SIZE;

  // q/k/g: [B, H, T, K_DIM], v/o: [B, H, T, V_DIM], beta: [B, H, T]
  const T* __restrict__ q_ptr;
  const T* __restrict__ k_ptr;
  const T* __restrict__ v_ptr;
  const T* __restrict__ g_ptr;
  const T* __restrict__ beta_ptr;

  // Intra-chunk intermediates: [B, H, num_chunks, C, K_DIM/V_DIM]
  T* __restrict__ w_ptr;
  T* __restrict__ u_ptr;

  // Final prefill output [B, H, T, V_DIM]
  T* __restrict__ o_ptr;

  int batch_size;
  int num_heads;
  int seq_len;
  int num_chunks;
  int inter_v_shards;
  int inter_v_tile;
};

template <typename T, int K_DIM_SIZE, int V_DIM_SIZE, int CHUNK_SIZE>
inline KdaPrefillParams<T, K_DIM_SIZE, V_DIM_SIZE, CHUNK_SIZE> make_prefill_params(
    const T* q_ptr,
    const T* k_ptr,
    const T* v_ptr,
    const T* g_ptr,
    const T* beta_ptr,
    T* w_ptr,
    T* u_ptr,
    T* o_ptr,
    int batch_size,
    int num_heads,
    int seq_len) {
  KdaPrefillParams<T, K_DIM_SIZE, V_DIM_SIZE, CHUNK_SIZE> params{};
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.g_ptr = g_ptr;
  params.beta_ptr = beta_ptr;
  params.w_ptr = w_ptr;
  params.u_ptr = u_ptr;
  params.o_ptr = o_ptr;
  params.batch_size = batch_size;
  params.num_heads = num_heads;
  params.seq_len = seq_len;
  params.num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
  params.inter_v_shards = 1;
  params.inter_v_tile = V_DIM_SIZE;
  return params;
}

}  // namespace kda