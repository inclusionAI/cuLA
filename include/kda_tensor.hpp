// layout to tensor-view helpers
#pragma once

#include "kda_layouts.hpp"

namespace kda {
namespace tensor {

template <typename T, int Dim>
struct FeatureTensorView {
  T* ptr;
  int B, H, T_len;
  int b, h;

  KDA_HOST_DEVICE T& operator()(int t, int d) const {
    return ptr[layouts::FeatureLayout<Dim>::offset(b, h, t, d, B, H, T_len)];
  }

  KDA_HOST_DEVICE T* data() const {
    return ptr + layouts::FeatureLayout<Dim>::offset(b, h, 0, 0, B, H, T_len);
  }
};

template <typename T>
struct GateTensorView {
  T* ptr;
  int B, H, T_len;
  int b, h;

  KDA_HOST_DEVICE T& operator()(int t) const {
    return ptr[layouts::GateLayout::offset(b, h, t, B, H, T_len)];
  }
};

template <int C, int Dim, typename T>
struct ChunkTensorView {
  T* ptr;
  int B, H, num_chunks;
  int b, h;

  KDA_HOST_DEVICE T& operator()(int chunk_idx, int row, int d) const {
    const std::size_t base = (((static_cast<std::size_t>(b) * H + h) * num_chunks + chunk_idx) * C * Dim);
    return ptr[base + row * Dim + d];
  }
};

template <int Dim, typename T>
KDA_HOST_DEVICE FeatureTensorView<T, Dim> get_feature_tensor(
    T* ptr, int B, int H, int T_len, int batch_idx, int head_idx) {
  return {ptr, B, H, T_len, batch_idx, head_idx};
}

template <typename T>
KDA_HOST_DEVICE GateTensorView<T> get_gate_tensor(
    T* ptr, int B, int H, int T_len, int batch_idx, int head_idx) {
  return {ptr, B, H, T_len, batch_idx, head_idx};
}

template <int C, int Dim, typename T>
KDA_HOST_DEVICE ChunkTensorView<C, Dim, T> get_chunk_tensor(
    T* ptr, int B, int H, int num_chunks, int batch_idx, int head_idx) {
  return {ptr, B, H, num_chunks, batch_idx, head_idx};
}

template <int C, int Dim, typename T>
KDA_HOST_DEVICE T* get_chunk_slice(
    T* ptr, int B, int H, int num_chunks, int batch_idx, int head_idx, int chunk_idx) {
  const std::size_t base = (((static_cast<std::size_t>(batch_idx) * H + head_idx) * num_chunks + chunk_idx) * C * Dim);
  (void)B;
  return ptr + base;
}

template <int C, int Dim, typename T>
KDA_HOST_DEVICE const T* get_chunk_slice(
    const T* ptr, int B, int H, int num_chunks, int batch_idx, int head_idx, int chunk_idx) {
  const std::size_t base = (((static_cast<std::size_t>(batch_idx) * H + head_idx) * num_chunks + chunk_idx) * C * Dim);
  (void)B;
  return ptr + base;
}

}  // namespace tensor
}  // namespace kda