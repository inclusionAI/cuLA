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
#include "qwen35_scalar_kda_kernel.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

namespace cula::qwen35::decode {

namespace {

void check_tensor_device(const at::Tensor& tensor, const char* name, const at::Device& device) {
  TORCH_CHECK(tensor.device() == device, name, " must be on device ", device, ".");
}

template <typename scalar_t, int kLocalVHeads>
void dispatch_scalar_decode_for_heads(
    cudaStream_t stream,
    const scalar_t* q_rep,
    const scalar_t* k_rep,
    const scalar_t* v,
    const scalar_t* a_kernel,
    const scalar_t* b_kernel,
    const float* A_log,
    const float* dt_bias,
    float* recurrent_state,
    const int32_t* pool_idx,
    scalar_t* out,
    int token_count) {
  using Shape = Qwen35DecodeLocalShape<kLocalVHeads>;
  kernel::launch_qwen35_scalar_kda_decode_kernel<scalar_t, Shape::kLocalQKHeads, Shape::kLocalVHeads>(
      stream,
      q_rep,
      k_rep,
      v,
      a_kernel,
      b_kernel,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count);
}

template <typename scalar_t, int kLocalVHeads>
void dispatch_layout_scalar_decode_for_heads(
    cudaStream_t stream,
    const scalar_t* mixed_qkv_conv,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    float* recurrent_state,
    const int32_t* pool_idx,
    scalar_t* out,
    int token_count) {
  using Shape = Qwen35DecodeLocalShape<kLocalVHeads>;
  kernel::launch_qwen35_layout_scalar_kda_decode_kernel<scalar_t, Shape::kLocalQKHeads, Shape::kLocalVHeads>(
      stream,
      mixed_qkv_conv,
      a,
      b,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count);
}

} // namespace

void run_qwen35_scalar_kda_decode(ScalarKdaDecodeParams& params) {
  const at::Tensor& q_rep = params.q_rep;
  const at::Tensor& k_rep = params.k_rep;
  const at::Tensor& v = params.v;
  const at::Tensor& a_kernel = params.a_kernel;
  const at::Tensor& b_kernel = params.b_kernel;
  const at::Tensor& A_log = params.A_log;
  const at::Tensor& dt_bias = params.dt_bias;
  const at::Tensor& recurrent_state = params.recurrent_state;
  const at::Tensor& pool_idx = params.pool_idx;
  const at::Tensor& out = params.out;

  TORCH_CHECK(q_rep.is_cuda(), "q_rep must be a CUDA tensor.");
  const at::Device device = q_rep.device();

  check_tensor_device(k_rep, "k_rep", device);
  check_tensor_device(v, "v", device);
  check_tensor_device(a_kernel, "a_kernel", device);
  check_tensor_device(b_kernel, "b_kernel", device);
  check_tensor_device(A_log, "A_log", device);
  check_tensor_device(dt_bias, "dt_bias", device);
  check_tensor_device(recurrent_state, "recurrent_state", device);
  check_tensor_device(pool_idx, "pool_idx", device);
  check_tensor_device(out, "out", device);

  TORCH_CHECK(q_rep.is_contiguous(), "q_rep must be contiguous.");
  TORCH_CHECK(k_rep.is_contiguous(), "k_rep must be contiguous.");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous.");
  TORCH_CHECK(a_kernel.is_contiguous(), "a_kernel must be contiguous.");
  TORCH_CHECK(b_kernel.is_contiguous(), "b_kernel must be contiguous.");
  TORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous.");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous.");
  TORCH_CHECK(recurrent_state.is_contiguous(), "recurrent_state must be contiguous.");
  TORCH_CHECK(pool_idx.is_contiguous(), "pool_idx must be contiguous.");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous.");

  TORCH_CHECK(
      q_rep.scalar_type() == k_rep.scalar_type() && q_rep.scalar_type() == v.scalar_type() &&
          q_rep.scalar_type() == a_kernel.scalar_type() && q_rep.scalar_type() == b_kernel.scalar_type() &&
          q_rep.scalar_type() == out.scalar_type(),
      "q_rep/k_rep/v/a_kernel/b_kernel/out must share the same dtype.");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat, "A_log must be float32.");
  TORCH_CHECK(dt_bias.scalar_type() == at::kFloat, "dt_bias must be float32.");
  TORCH_CHECK(recurrent_state.scalar_type() == at::kFloat, "recurrent_state must be float32.");
  TORCH_CHECK(pool_idx.scalar_type() == at::kInt, "pool_idx must be int32.");

  TORCH_CHECK(q_rep.dim() == 3, "q_rep must have shape [N, local_v_heads, 128].");
  const int64_t token_count = q_rep.size(0);
  const int64_t local_v_heads = q_rep.size(1);
  TORCH_CHECK(is_supported_local_v_heads(static_cast<int>(local_v_heads)), "local V heads must be one of {48, 24, 12, 6}, got ", local_v_heads, ".");
  TORCH_CHECK(
      q_rep.sizes() == at::IntArrayRef({token_count, local_v_heads, kHeadDimQK}),
      "q_rep must have shape [N, local_v_heads, 128].");
  TORCH_CHECK(
      k_rep.dim() == 3 && k_rep.sizes() == at::IntArrayRef({token_count, local_v_heads, kHeadDimQK}),
      "k_rep must have shape [N, local_v_heads, 128].");
  TORCH_CHECK(
      v.dim() == 3 && v.sizes() == at::IntArrayRef({token_count, local_v_heads, kHeadDimV}),
      "v must have shape [N, local_v_heads, 128].");
  TORCH_CHECK(
      a_kernel.dim() == 2 && a_kernel.sizes() == at::IntArrayRef({token_count, local_v_heads}),
      "a_kernel must have shape [N, local_v_heads].");
  TORCH_CHECK(
      b_kernel.dim() == 2 && b_kernel.sizes() == at::IntArrayRef({token_count, local_v_heads}),
      "b_kernel must have shape [N, local_v_heads].");
  TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == local_v_heads, "A_log must have shape [local_v_heads].");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == local_v_heads, "dt_bias must have shape [local_v_heads].");
  TORCH_CHECK(
      recurrent_state.dim() == 4 &&
          recurrent_state.size(1) == local_v_heads &&
          recurrent_state.size(2) == kHeadDimQK &&
          recurrent_state.size(3) == kHeadDimV,
      "recurrent_state must have shape [pool, local_v_heads, 128, 128].");
  TORCH_CHECK(pool_idx.dim() == 1 && pool_idx.size(0) == token_count, "pool_idx must have shape [N].");
  TORCH_CHECK(
      out.dim() == 3 && out.sizes() == at::IntArrayRef({token_count, local_v_heads, kHeadDimV}),
      "out must have shape [N, local_v_heads, 128].");

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(device.index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q_rep.scalar_type(),
      "launch_qwen35_scalar_kda_decode_kernel",
      [&] {
        switch (local_v_heads) {
          case 48:
            dispatch_scalar_decode_for_heads<scalar_t, 48>(stream, q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
          case 24:
            dispatch_scalar_decode_for_heads<scalar_t, 24>(stream, q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
          case 12:
            dispatch_scalar_decode_for_heads<scalar_t, 12>(stream, q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
          case 6:
            dispatch_scalar_decode_for_heads<scalar_t, 6>(stream, q_rep.data_ptr<scalar_t>(), k_rep.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a_kernel.data_ptr<scalar_t>(), b_kernel.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_qwen35_layout_scalar_kda_decode(LayoutScalarKdaDecodeParams& params) {
  const at::Tensor& mixed_qkv_conv = params.mixed_qkv_conv;
  const at::Tensor& a = params.a;
  const at::Tensor& b = params.b;
  const at::Tensor& A_log = params.A_log;
  const at::Tensor& dt_bias = params.dt_bias;
  const at::Tensor& recurrent_state = params.recurrent_state;
  const at::Tensor& pool_idx = params.pool_idx;
  const at::Tensor& out = params.out;

  TORCH_CHECK(mixed_qkv_conv.is_cuda(), "mixed_qkv_conv must be a CUDA tensor.");
  const at::Device device = mixed_qkv_conv.device();

  check_tensor_device(a, "a", device);
  check_tensor_device(b, "b", device);
  check_tensor_device(A_log, "A_log", device);
  check_tensor_device(dt_bias, "dt_bias", device);
  check_tensor_device(recurrent_state, "recurrent_state", device);
  check_tensor_device(pool_idx, "pool_idx", device);
  check_tensor_device(out, "out", device);

  TORCH_CHECK(mixed_qkv_conv.is_contiguous(), "mixed_qkv_conv must be contiguous.");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous.");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous.");
  TORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous.");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous.");
  TORCH_CHECK(recurrent_state.is_contiguous(), "recurrent_state must be contiguous.");
  TORCH_CHECK(pool_idx.is_contiguous(), "pool_idx must be contiguous.");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous.");

  TORCH_CHECK(
      mixed_qkv_conv.scalar_type() == a.scalar_type() &&
          mixed_qkv_conv.scalar_type() == b.scalar_type() &&
          mixed_qkv_conv.scalar_type() == out.scalar_type(),
      "mixed_qkv_conv/a/b/out must share the same dtype.");
  TORCH_CHECK(
      mixed_qkv_conv.scalar_type() == at::kHalf || mixed_qkv_conv.scalar_type() == at::kBFloat16,
      "mixed_qkv_conv must be float16 or bfloat16.");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat, "A_log must be float32.");
  TORCH_CHECK(dt_bias.scalar_type() == at::kFloat, "dt_bias must be float32.");
  TORCH_CHECK(recurrent_state.scalar_type() == at::kFloat, "recurrent_state must be float32.");
  TORCH_CHECK(pool_idx.scalar_type() == at::kInt, "pool_idx must be int32.");

  TORCH_CHECK(mixed_qkv_conv.dim() == 2, "mixed_qkv_conv must have shape [N, local_conv_dim].");
  TORCH_CHECK(a.dim() == 2, "a must have shape [N, local_v_heads].");
  const int64_t token_count = mixed_qkv_conv.size(0);
  const int64_t local_v_heads = a.size(1);
  TORCH_CHECK(is_supported_local_v_heads(static_cast<int>(local_v_heads)), "local V heads must be one of {48, 24, 12, 6}, got ", local_v_heads, ".");
  const int local_qk_heads = local_qk_heads_from_v_heads(static_cast<int>(local_v_heads));
  const int local_mixed_dim = local_mixed_qkv_dim(local_qk_heads, static_cast<int>(local_v_heads));
  TORCH_CHECK(
      mixed_qkv_conv.sizes() == at::IntArrayRef({token_count, local_mixed_dim}),
      "mixed_qkv_conv must have shape [N, local_conv_dim].");
  TORCH_CHECK(
      a.sizes() == at::IntArrayRef({token_count, local_v_heads}),
      "a must have shape [N, local_v_heads].");
  TORCH_CHECK(
      b.dim() == 2 && b.sizes() == at::IntArrayRef({token_count, local_v_heads}),
      "b must have shape [N, local_v_heads].");
  TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == local_v_heads, "A_log must have shape [local_v_heads].");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == local_v_heads, "dt_bias must have shape [local_v_heads].");
  TORCH_CHECK(
      recurrent_state.dim() == 4 &&
          recurrent_state.size(1) == local_v_heads &&
          recurrent_state.size(2) == kHeadDimQK &&
          recurrent_state.size(3) == kHeadDimV,
      "recurrent_state must have shape [pool, local_v_heads, 128, 128].");
  TORCH_CHECK(pool_idx.dim() == 1 && pool_idx.size(0) == token_count, "pool_idx must have shape [N].");
  TORCH_CHECK(
      out.dim() == 3 && out.sizes() == at::IntArrayRef({token_count, local_v_heads, kHeadDimV}),
      "out must have shape [N, local_v_heads, 128].");

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(device.index());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      mixed_qkv_conv.scalar_type(),
      "launch_qwen35_layout_scalar_kda_decode_kernel",
      [&] {
        switch (local_v_heads) {
          case 48:
            dispatch_layout_scalar_decode_for_heads<scalar_t, 48>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
          case 24:
            dispatch_layout_scalar_decode_for_heads<scalar_t, 24>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
          case 12:
            dispatch_layout_scalar_decode_for_heads<scalar_t, 12>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
          case 6:
            dispatch_layout_scalar_decode_for_heads<scalar_t, 6>(stream, mixed_qkv_conv.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), recurrent_state.data_ptr<float>(), pool_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), static_cast<int>(token_count));
            break;
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cula::qwen35::decode
