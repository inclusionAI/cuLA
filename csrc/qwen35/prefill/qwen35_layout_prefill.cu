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

#include "qwen35_layout_prefill_kernel.hpp"
#include "qwen35_prefill_common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

namespace cula::qwen35::prefill {

namespace {

void check_tensor_device(const at::Tensor& tensor, const char* name, const at::Device& device) {
  TORCH_CHECK(tensor.device() == device, name, " must be on device ", device, ".");
}

void check_rank_2(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank 2, got rank ", tensor.dim(), ".");
}

template <typename scalar_t, int kLocalVHeads>
void dispatch_layout_prefill_for_heads(
    cudaStream_t stream,
    const scalar_t* mixed_qkv_conv,
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* q_rep,
    scalar_t* k_rep,
    scalar_t* v,
    scalar_t* a_kernel,
    scalar_t* b_kernel,
    int64_t token_count) {
  constexpr int kLocalQKHeads = decode::local_qk_heads_from_v_heads(kLocalVHeads);
  kernel::launch_qwen35_layout_prefill_kernel<scalar_t, kLocalQKHeads, kLocalVHeads>(
      stream,
      mixed_qkv_conv,
      a,
      b,
      q_rep,
      k_rep,
      v,
      a_kernel,
      b_kernel,
      token_count);
}

template <typename scalar_t>
void dispatch_layout_prefill(
    int64_t local_v_heads,
    cudaStream_t stream,
    const scalar_t* mixed_qkv_conv,
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* q_rep,
    scalar_t* k_rep,
    scalar_t* v,
    scalar_t* a_kernel,
    scalar_t* b_kernel,
    int64_t token_count) {
  switch (local_v_heads) {
    case 48:
      dispatch_layout_prefill_for_heads<scalar_t, 48>(stream, mixed_qkv_conv, a, b, q_rep, k_rep, v, a_kernel, b_kernel, token_count);
      break;
    case 24:
      dispatch_layout_prefill_for_heads<scalar_t, 24>(stream, mixed_qkv_conv, a, b, q_rep, k_rep, v, a_kernel, b_kernel, token_count);
      break;
    case 12:
      dispatch_layout_prefill_for_heads<scalar_t, 12>(stream, mixed_qkv_conv, a, b, q_rep, k_rep, v, a_kernel, b_kernel, token_count);
      break;
    case 6:
      dispatch_layout_prefill_for_heads<scalar_t, 6>(stream, mixed_qkv_conv, a, b, q_rep, k_rep, v, a_kernel, b_kernel, token_count);
      break;
  }
}

} // namespace

void run_qwen35_layout_prefill(LayoutPrefillParams& params) {
  const at::Tensor& mixed_qkv_conv = params.mixed_qkv_conv;
  const at::Tensor& a = params.a;
  const at::Tensor& b = params.b;
  const at::Tensor& q_rep = params.q_rep;
  const at::Tensor& k_rep = params.k_rep;
  const at::Tensor& v = params.v;
  const at::Tensor& a_kernel = params.a_kernel;
  const at::Tensor& b_kernel = params.b_kernel;

  TORCH_CHECK(mixed_qkv_conv.is_cuda(), "mixed_qkv_conv must be a CUDA tensor.");
  const at::Device device = mixed_qkv_conv.device();

  check_tensor_device(a, "a", device);
  check_tensor_device(b, "b", device);
  check_tensor_device(q_rep, "q_rep", device);
  check_tensor_device(k_rep, "k_rep", device);
  check_tensor_device(v, "v", device);
  check_tensor_device(a_kernel, "a_kernel", device);
  check_tensor_device(b_kernel, "b_kernel", device);

  TORCH_CHECK(mixed_qkv_conv.is_contiguous(), "mixed_qkv_conv must be contiguous.");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous.");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous.");
  TORCH_CHECK(q_rep.is_contiguous(), "q_rep must be contiguous.");
  TORCH_CHECK(k_rep.is_contiguous(), "k_rep must be contiguous.");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous.");
  TORCH_CHECK(a_kernel.is_contiguous(), "a_kernel must be contiguous.");
  TORCH_CHECK(b_kernel.is_contiguous(), "b_kernel must be contiguous.");

  TORCH_CHECK(
      mixed_qkv_conv.scalar_type() == a.scalar_type() &&
          mixed_qkv_conv.scalar_type() == b.scalar_type() &&
          mixed_qkv_conv.scalar_type() == q_rep.scalar_type() &&
          mixed_qkv_conv.scalar_type() == k_rep.scalar_type() &&
          mixed_qkv_conv.scalar_type() == v.scalar_type() &&
          mixed_qkv_conv.scalar_type() == a_kernel.scalar_type() &&
          mixed_qkv_conv.scalar_type() == b_kernel.scalar_type(),
      "All layout prefill tensors must share the same dtype.");
  TORCH_CHECK(
      mixed_qkv_conv.scalar_type() == at::kHalf || mixed_qkv_conv.scalar_type() == at::kBFloat16,
      "mixed_qkv_conv must be float16 or bfloat16.");

  check_rank_2(mixed_qkv_conv, "mixed_qkv_conv");
  check_rank_2(a, "a");
  check_rank_2(b, "b");

  const int64_t token_count = mixed_qkv_conv.size(0);
  const int64_t local_v_heads = a.size(1);
  TORCH_CHECK(decode::is_supported_local_v_heads(static_cast<int>(local_v_heads)), "local V heads must be one of {48, 24, 12, 6}, got ", local_v_heads, ".");
  const int local_qk_heads = decode::local_qk_heads_from_v_heads(static_cast<int>(local_v_heads));
  const int local_mixed_dim = decode::local_mixed_qkv_dim(local_qk_heads, static_cast<int>(local_v_heads));
  TORCH_CHECK(mixed_qkv_conv.size(1) == local_mixed_dim, "mixed_qkv_conv must be [N, local_conv_dim=", local_mixed_dim, "].");
  TORCH_CHECK(a.sizes() == at::IntArrayRef({token_count, local_v_heads}), "a must be [N, local_v_heads].");
  TORCH_CHECK(b.sizes() == at::IntArrayRef({token_count, local_v_heads}), "b must be [N, local_v_heads].");
  TORCH_CHECK(
      q_rep.dim() == 3 && q_rep.sizes() == at::IntArrayRef({token_count, local_v_heads, kHeadDimQK}),
      "q_rep must be [N, local_v_heads, 128].");
  TORCH_CHECK(k_rep.sizes() == q_rep.sizes(), "k_rep must match q_rep shape.");
  TORCH_CHECK(v.sizes() == q_rep.sizes(), "v must match q_rep shape.");
  TORCH_CHECK(a_kernel.sizes() == a.sizes(), "a_kernel must match a shape.");
  TORCH_CHECK(b_kernel.sizes() == b.sizes(), "b_kernel must match b shape.");

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(device.index());

  if (mixed_qkv_conv.scalar_type() == at::kHalf) {
    dispatch_layout_prefill<c10::Half>(
        local_v_heads,
        stream,
        mixed_qkv_conv.data_ptr<c10::Half>(),
        a.data_ptr<c10::Half>(),
        b.data_ptr<c10::Half>(),
        q_rep.data_ptr<c10::Half>(),
        k_rep.data_ptr<c10::Half>(),
        v.data_ptr<c10::Half>(),
        a_kernel.data_ptr<c10::Half>(),
        b_kernel.data_ptr<c10::Half>(),
        token_count);
  } else {
    dispatch_layout_prefill<c10::BFloat16>(
        local_v_heads,
        stream,
        mixed_qkv_conv.data_ptr<c10::BFloat16>(),
        a.data_ptr<c10::BFloat16>(),
        b.data_ptr<c10::BFloat16>(),
        q_rep.data_ptr<c10::BFloat16>(),
        k_rep.data_ptr<c10::BFloat16>(),
        v.data_ptr<c10::BFloat16>(),
        a_kernel.data_ptr<c10::BFloat16>(),
        b_kernel.data_ptr<c10::BFloat16>(),
        token_count);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cula::qwen35::prefill
