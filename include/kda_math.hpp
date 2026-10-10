#pragma once

#include <cmath>
#include <cuda_runtime.h>

#ifndef KDA_HOST_DEVICE
#define KDA_HOST_DEVICE __forceinline__ __host__ __device__
#endif

namespace kda {
namespace math {

using fp32 = float;

KDA_HOST_DEVICE float fast_exp_f32(float x) {
#if defined(__CUDA_ARCH__)
  return __expf(x);
#else
  return std::exp(x);
#endif
}

KDA_HOST_DEVICE float fast_log_f32(float x) {
#if defined(__CUDA_ARCH__)
  return __logf(x);
#else
  return std::log(x);
#endif
}

template <typename T>
KDA_HOST_DEVICE T sigmoid(T x) {
  const float x_f = static_cast<float>(x);
  const float res = 1.0f / (1.0f + fast_exp_f32(-x_f));
  return static_cast<T>(res);
}

template <typename T>
KDA_HOST_DEVICE T swish(T x) {
  const float x_f = static_cast<float>(x);
  return static_cast<T>(x_f * static_cast<float>(sigmoid(x_f)));
}

template <typename T>
KDA_HOST_DEVICE T softplus(T x) {
  const float x_f = static_cast<float>(x);
  const float res = (x_f > 20.0f) ? x_f : fast_log_f32(1.0f + fast_exp_f32(x_f));
  return static_cast<T>(res);
}

KDA_HOST_DEVICE fp32 warp_reduce_sum(fp32 val) {
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
#else
  return val;
#endif
}

template <typename T>
KDA_HOST_DEVICE T fast_exp(T x) {
  return static_cast<T>(fast_exp_f32(static_cast<float>(x)));
}

}  // namespace math
}  // namespace kda