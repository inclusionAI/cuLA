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

#include "qwen35_decode_common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

template <typename T>
__device__ inline float to_float(T x) {
  return static_cast<float>(x);
}

template <>
__device__ inline float to_float<c10::Half>(c10::Half x) {
  return __half2float(static_cast<__half>(x));
}

template <>
__device__ inline float to_float<c10::BFloat16>(c10::BFloat16 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162float(static_cast<__nv_bfloat16>(x));
#else
  return static_cast<float>(x);
#endif
}

template <typename T>
__device__ inline T from_float(float x) {
  return static_cast<T>(x);
}

template <>
__device__ inline c10::Half from_float<c10::Half>(float x) {
  return c10::Half(__float2half_rn(x));
}

template <>
__device__ inline c10::BFloat16 from_float<c10::BFloat16>(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return c10::BFloat16(__float2bfloat16(x));
#else
  return c10::BFloat16(x);
#endif
}

template <typename scalar_t>
__global__ void qwen35_conv1d_decode_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    scalar_t* __restrict__ conv_state,
    const scalar_t* __restrict__ conv_weight,
    scalar_t* __restrict__ out,
    int batch_size,
    int conv_dim) {
  constexpr int kThreads = 256;
  const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch_size) * conv_dim;
  if (linear_idx >= total) {
    return;
  }

  const int64_t b = linear_idx / conv_dim;
  const int64_t c = linear_idx % conv_dim;

  const int64_t x_idx = b * conv_dim + c;
  const int64_t state_base =
      (b * conv_dim + c) * cula::qwen35::decode::kConvKernelSize;
  const int64_t weight_base = c * cula::qwen35::decode::kConvKernelSize;

  const float s0 = to_float(conv_state[state_base + 1]);
  const float s1 = to_float(conv_state[state_base + 2]);
  const float s2 = to_float(conv_state[state_base + 3]);
  const float s3 = to_float(mixed_qkv[x_idx]);

  const float w0 = to_float(conv_weight[weight_base + 0]);
  const float w1 = to_float(conv_weight[weight_base + 1]);
  const float w2 = to_float(conv_weight[weight_base + 2]);
  const float w3 = to_float(conv_weight[weight_base + 3]);

  const float conv = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;
  const float silu = conv / (1.f + expf(-conv));

  conv_state[state_base + 0] = from_float<scalar_t>(s0);
  conv_state[state_base + 1] = from_float<scalar_t>(s1);
  conv_state[state_base + 2] = from_float<scalar_t>(s2);
  conv_state[state_base + 3] = from_float<scalar_t>(s3);
  out[x_idx] = from_float<scalar_t>(silu);
}

void check_tensor_device(const at::Tensor& tensor, const char* name, const at::Device& device) {
  TORCH_CHECK(tensor.device() == device, name, " must be on device ", device, ".");
}

} // namespace

namespace cula::qwen35::decode {

void run_qwen35_conv1d_decode(ConvDecodeParams& params) {
  const at::Tensor& mixed_qkv = params.mixed_qkv;
  const at::Tensor& conv_state = params.conv_state;
  const at::Tensor& conv_weight = params.conv_weight;
  const at::Tensor& out = params.out;

  TORCH_CHECK(mixed_qkv.is_cuda(), "mixed_qkv must be a CUDA tensor.");
  const at::Device device = mixed_qkv.device();

  check_tensor_device(conv_state, "conv_state", device);
  check_tensor_device(conv_weight, "conv_weight", device);
  check_tensor_device(out, "out", device);

  TORCH_CHECK(mixed_qkv.is_contiguous(), "mixed_qkv must be contiguous.");
  TORCH_CHECK(conv_state.is_contiguous(), "conv_state must be contiguous.");
  TORCH_CHECK(conv_weight.is_contiguous(), "conv_weight must be contiguous.");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous.");

  TORCH_CHECK(
      mixed_qkv.scalar_type() == conv_state.scalar_type() &&
          mixed_qkv.scalar_type() == conv_weight.scalar_type() &&
          mixed_qkv.scalar_type() == out.scalar_type(),
      "mixed_qkv/conv_state/conv_weight/out must share the same dtype.");

  TORCH_CHECK(
      mixed_qkv.scalar_type() == at::kHalf || mixed_qkv.scalar_type() == at::kBFloat16,
      "conv decode only supports half/bfloat16.");

  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t conv_dim = mixed_qkv.size(2);
  TORCH_CHECK(conv_dim > 0, "conv_dim must be positive.");
  TORCH_CHECK(
      mixed_qkv.dim() == 3 && mixed_qkv.sizes() == at::IntArrayRef({batch_size, 1, conv_dim}),
      "mixed_qkv must have shape [B, 1, local_conv_dim].");
  TORCH_CHECK(
      conv_state.dim() == 3 &&
          conv_state.sizes() == at::IntArrayRef({batch_size, conv_dim, kConvKernelSize}),
      "conv_state must have shape [B, local_conv_dim, 4].");
  TORCH_CHECK(
      (conv_weight.dim() == 2 && conv_weight.sizes() == at::IntArrayRef({conv_dim, kConvKernelSize})) ||
          (conv_weight.dim() == 3 &&
           conv_weight.sizes() == at::IntArrayRef({conv_dim, 1, kConvKernelSize})),
      "conv_weight must have shape [local_conv_dim, 4] or [local_conv_dim, 1, 4].");
  TORCH_CHECK(
      out.dim() == 3 && out.sizes() == at::IntArrayRef({batch_size, 1, conv_dim}),
      "out must have shape [B, 1, local_conv_dim].");

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(device.index());

  const at::Tensor mixed_qkv_2d = mixed_qkv.view({batch_size, conv_dim});
  const at::Tensor out_2d = out.view({batch_size, conv_dim});
  const at::Tensor weight_2d =
      conv_weight.dim() == 3 ? conv_weight.view({conv_dim, kConvKernelSize}) : conv_weight;

  constexpr int kThreads = 256;
  const int64_t total = batch_size * conv_dim;
  const dim3 block(kThreads, 1, 1);
  const dim3 grid(static_cast<unsigned int>((total + kThreads - 1) / kThreads), 1, 1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      mixed_qkv.scalar_type(),
      "qwen35_conv1d_decode_kernel",
      [&] {
        qwen35_conv1d_decode_kernel<scalar_t><<<grid, block, 0, stream>>>(
            mixed_qkv_2d.data_ptr<scalar_t>(),
            conv_state.data_ptr<scalar_t>(),
            weight_2d.data_ptr<scalar_t>(),
            out_2d.data_ptr<scalar_t>(),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cula::qwen35::decode
