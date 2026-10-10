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
#include "qwen35_layout_kernel.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

void check_tensor_device(const at::Tensor& tensor, const char* name, const at::Device& device) {
  TORCH_CHECK(tensor.device() == device, name, " must be on device ", device, ".");
}

void check_tensor_shape_2d(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.dim() == 2,
      name,
      " must have rank 2, but got rank ",
      tensor.dim(),
      ".");
}

template <typename scalar_t, int kLocalVHeads>
void launch_layout_decode_for_heads(
    cudaStream_t stream,
    const scalar_t* mixed_qkv_conv,
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* q_rep,
    scalar_t* k_rep,
    scalar_t* v,
    scalar_t* a_kernel,
    scalar_t* b_kernel,
    int64_t batch_size) {
  using Shape = cula::qwen35::decode::Qwen35DecodeLocalShape<kLocalVHeads>;
  dim3 grid(Shape::kLocalVHeads, static_cast<unsigned int>(batch_size), 1);
  cula::qwen35::decode::qwen35_layout_decode_kernel_cute<scalar_t, Shape::kLocalQKHeads, Shape::kLocalVHeads>
      <<<grid, Shape::kLayoutThreads, 0, stream>>>(
          mixed_qkv_conv,
          a,
          b,
          q_rep,
          k_rep,
          v,
          a_kernel,
          b_kernel,
          batch_size);
}

} // namespace

namespace cula::qwen35::decode {

void run_qwen35_layout_decode(LayoutDecodeParams& params) {
  const at::Tensor& mixed_qkv_conv = params.mixed_qkv_conv;
  const at::Tensor& a = params.a;
  const at::Tensor& b = params.b;
  const at::Tensor& q_rep = params.q_rep;
  const at::Tensor& k_rep = params.k_rep;
  const at::Tensor& v = params.v;
  const at::Tensor& a_kernel = params.a_kernel;
  const at::Tensor& b_kernel = params.b_kernel;

  TORCH_CHECK(mixed_qkv_conv.is_cuda(), "mixed_qkv_conv must be a CUDA tensor.");
  TORCH_CHECK(mixed_qkv_conv.is_contiguous(), "mixed_qkv_conv must be contiguous.");
  TORCH_CHECK(
      mixed_qkv_conv.scalar_type() == a.scalar_type() &&
          mixed_qkv_conv.scalar_type() == b.scalar_type() &&
          mixed_qkv_conv.scalar_type() == q_rep.scalar_type() &&
          mixed_qkv_conv.scalar_type() == k_rep.scalar_type() &&
          mixed_qkv_conv.scalar_type() == v.scalar_type() &&
          mixed_qkv_conv.scalar_type() == a_kernel.scalar_type() &&
          mixed_qkv_conv.scalar_type() == b_kernel.scalar_type(),
      "All layout decode tensors must share the same dtype.");

  check_tensor_shape_2d(a, "a");
  check_tensor_shape_2d(b, "b");

  const int64_t batch_size = mixed_qkv_conv.size(0);
  const int64_t local_v_heads = a.size(1);
  TORCH_CHECK(is_supported_local_v_heads(static_cast<int>(local_v_heads)), "local V heads must be one of {48, 24, 12, 6}, got ", local_v_heads, ".");
  const int local_qk_heads = local_qk_heads_from_v_heads(static_cast<int>(local_v_heads));
  const int local_mixed_dim = local_mixed_qkv_dim(local_qk_heads, static_cast<int>(local_v_heads));
  TORCH_CHECK(
      mixed_qkv_conv.dim() == 2 && mixed_qkv_conv.size(1) == local_mixed_dim,
      "mixed_qkv_conv must have shape [N, local_conv_dim=", local_mixed_dim, "], got ",
      mixed_qkv_conv.sizes(), ".");
  const at::Device device = mixed_qkv_conv.device();

  check_tensor_device(a, "a", device);
  check_tensor_device(b, "b", device);
  check_tensor_device(q_rep, "q_rep", device);
  check_tensor_device(k_rep, "k_rep", device);
  check_tensor_device(v, "v", device);
  check_tensor_device(a_kernel, "a_kernel", device);
  check_tensor_device(b_kernel, "b_kernel", device);

  TORCH_CHECK(q_rep.is_contiguous(), "q_rep must be contiguous.");
  TORCH_CHECK(k_rep.is_contiguous(), "k_rep must be contiguous.");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous.");
  TORCH_CHECK(a_kernel.is_contiguous(), "a_kernel must be contiguous.");
  TORCH_CHECK(b_kernel.is_contiguous(), "b_kernel must be contiguous.");

  TORCH_CHECK(
      q_rep.dim() == 3 && q_rep.sizes() == at::IntArrayRef({batch_size, local_v_heads, kHeadDimQK}),
      "q_rep must have shape [N, local_v_heads, 128].");
  TORCH_CHECK(
      k_rep.dim() == 3 && k_rep.sizes() == at::IntArrayRef({batch_size, local_v_heads, kHeadDimQK}),
      "k_rep must have shape [N, local_v_heads, 128].");
  TORCH_CHECK(
      v.dim() == 3 && v.sizes() == at::IntArrayRef({batch_size, local_v_heads, kHeadDimV}),
      "v must have shape [N, local_v_heads, 128].");
  TORCH_CHECK(
      a_kernel.dim() == 2 && a_kernel.sizes() == at::IntArrayRef({batch_size, local_v_heads}),
      "a_kernel must have shape [N, local_v_heads].");
  TORCH_CHECK(
      b_kernel.dim() == 2 && b_kernel.sizes() == at::IntArrayRef({batch_size, local_v_heads}),
      "b_kernel must have shape [N, local_v_heads].");

  TORCH_CHECK(a.sizes() == at::IntArrayRef({batch_size, local_v_heads}), "a must have shape [N, local_v_heads].");
  TORCH_CHECK(b.sizes() == at::IntArrayRef({batch_size, local_v_heads}), "b must have shape [N, local_v_heads].");

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(device.index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      mixed_qkv_conv.scalar_type(),
      "qwen35_layout_decode_kernel_cute",
      [&] {
        switch (local_v_heads) {
          case 48:
            launch_layout_decode_for_heads<scalar_t, 48>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), batch_size);
            break;
          case 24:
            launch_layout_decode_for_heads<scalar_t, 24>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), batch_size);
            break;
          case 12:
            launch_layout_decode_for_heads<scalar_t, 12>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), batch_size);
            break;
          case 6:
            launch_layout_decode_for_heads<scalar_t, 6>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), batch_size);
            break;
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cula::qwen35::decode
