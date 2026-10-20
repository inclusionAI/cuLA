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

#pragma once

#include "qwen35_prefill_common.cuh"

#include <cutlass/array.h>
#include <cute/tensor.hpp>
#include <cstdint>
#include <cuda_runtime.h>

namespace cula::qwen35::prefill::kernel {

using namespace cute;

template <typename scalar_t, int kVec>
CUTE_DEVICE void copy_prefill_vec_contiguous(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src) {
  constexpr int kBytes = sizeof(scalar_t) * kVec;
  if constexpr (kBytes == 16 || kBytes == 8) {
    using VecType = cutlass::AlignedArray<scalar_t, kVec>;
    const auto dst_addr = reinterpret_cast<uintptr_t>(dst);
    const auto src_addr = reinterpret_cast<uintptr_t>(src);
    if ((dst_addr % alignof(VecType) == 0) && (src_addr % alignof(VecType) == 0)) {
      *reinterpret_cast<VecType*>(dst) = *reinterpret_cast<const VecType*>(src);
      return;
    }
  }

#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    dst[i] = src[i];
  }
}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads>
__global__ void qwen35_layout_prefill_kernel(
    const scalar_t* __restrict__ mixed_qkv_conv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ q_rep,
    scalar_t* __restrict__ k_rep,
    scalar_t* __restrict__ v_out,
    scalar_t* __restrict__ a_kernel,
    scalar_t* __restrict__ b_kernel,
    int64_t token_count) {
  static_assert(kLocalVHeads % kLocalQKHeads == 0);
  static_assert(kHeadDimQK == kHeadDimV);
  constexpr int kRepeatFactor = kLocalVHeads / kLocalQKHeads;
  constexpr int kLocalQDim = kLocalQKHeads * kHeadDimQK;
  constexpr int kLocalKDim = kLocalQKHeads * kHeadDimQK;
  constexpr int kLocalMixedQKVDim = 2 * kLocalQDim + kLocalVHeads * kHeadDimV;
  constexpr int kVec = 4;
  static_assert(kHeadDimQK % kVec == 0);

  const int hv = static_cast<int>(blockIdx.x);
  const int token_idx = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  if (token_idx >= token_count || hv >= kLocalVHeads) {
    return;
  }

  const int mapped_h = hv / kRepeatFactor;

  auto qk_src_layout = make_layout(
      make_shape(Int<kLocalQKHeads>{}, Int<kHeadDimQK>{}),
      make_stride(Int<kHeadDimQK>{}, Int<1>{}));
  auto v_src_layout = make_layout(
      make_shape(Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
      make_stride(Int<kHeadDimV>{}, Int<1>{}));
  auto hv_layout = make_layout(
      make_shape(Int<kLocalVHeads>{}, Int<kHeadDimV>{}),
      make_stride(Int<kHeadDimV>{}, Int<1>{}));
  auto head_layout = make_layout(make_shape(Int<kLocalVHeads>{}), make_stride(Int<1>{}));

  const scalar_t* token_ptr = mixed_qkv_conv + static_cast<int64_t>(token_idx) * kLocalMixedQKVDim;
  const scalar_t* q_src_ptr = token_ptr;
  const scalar_t* k_src_ptr = token_ptr + kLocalQDim;
  const scalar_t* v_src_ptr = token_ptr + kLocalQDim + kLocalKDim;

  scalar_t* q_dst_ptr = q_rep + static_cast<int64_t>(token_idx) * kLocalVHeads * kHeadDimQK;
  scalar_t* k_dst_ptr = k_rep + static_cast<int64_t>(token_idx) * kLocalVHeads * kHeadDimQK;
  scalar_t* v_dst_ptr = v_out + static_cast<int64_t>(token_idx) * kLocalVHeads * kHeadDimV;

  for (int vec_idx = tid; vec_idx < kHeadDimQK / kVec; vec_idx += blockDim.x) {
    const int d = vec_idx * kVec;
    const int q_src_idx = crd2idx(make_coord(mapped_h, d), qk_src_layout);
    const int k_src_idx = crd2idx(make_coord(mapped_h, d), qk_src_layout);
    const int v_src_idx = crd2idx(make_coord(hv, d), v_src_layout);
    const int dst_idx = crd2idx(make_coord(hv, d), hv_layout);

    copy_prefill_vec_contiguous<scalar_t, kVec>(q_dst_ptr + dst_idx, q_src_ptr + q_src_idx);
    copy_prefill_vec_contiguous<scalar_t, kVec>(k_dst_ptr + dst_idx, k_src_ptr + k_src_idx);
    copy_prefill_vec_contiguous<scalar_t, kVec>(v_dst_ptr + dst_idx, v_src_ptr + v_src_idx);
  }

  if (tid == 0) {
    const int head_idx = crd2idx(make_coord(hv), head_layout);
    const int64_t token_head_offset = static_cast<int64_t>(token_idx) * kLocalVHeads + head_idx;
    a_kernel[token_head_offset] = a[token_head_offset];
    b_kernel[token_head_offset] = b[token_head_offset];
  }
}

template <typename scalar_t, int kLocalQKHeads, int kLocalVHeads>
void launch_qwen35_layout_prefill_kernel(
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
  constexpr int kThreads = 32;
  dim3 grid(kLocalVHeads, static_cast<unsigned int>(token_count), 1);
  qwen35_layout_prefill_kernel<scalar_t, kLocalQKHeads, kLocalVHeads><<<grid, kThreads, 0, stream>>>(
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

} // namespace cula::qwen35::prefill::kernel
