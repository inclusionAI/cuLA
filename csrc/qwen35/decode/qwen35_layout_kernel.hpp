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

#include "qwen35_decode_common.cuh"

#include <cutlass/array.h>
#include <cute/tensor.hpp>
#include <cstdint>

namespace cula::qwen35::decode {

using namespace cute;

template <typename scalar_t, int kVec>
CUTE_DEVICE void copy_vec_contiguous(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src) {
  constexpr int kBytes = sizeof(scalar_t) * kVec;
  if constexpr (kBytes == 16 || kBytes == 8) {
    using VecType = cutlass::AlignedArray<scalar_t, kVec>;
    auto dst_addr = reinterpret_cast<uintptr_t>(dst);
    auto src_addr = reinterpret_cast<uintptr_t>(src);
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
__global__ void qwen35_layout_decode_kernel_cute(
    const scalar_t* __restrict__ mixed_qkv_conv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ q_rep,
    scalar_t* __restrict__ k_rep,
    scalar_t* __restrict__ v_out,
    scalar_t* __restrict__ a_kernel,
    scalar_t* __restrict__ b_kernel,
    int64_t token_count) {
  using Shape = Qwen35DecodeLocalShape<kLocalVHeads>;
  static_assert(kLocalQKHeads == Shape::kLocalQKHeads);
  constexpr int kRepeatFactor = Shape::kRepeatFactor;
  constexpr int kLocalQDim = Shape::kLocalQDim;
  constexpr int kLocalKDim = Shape::kLocalKDim;
  constexpr int kLocalMixedQKVDim = Shape::kLocalMixedQKVDim;
  // TODO(qwen35-layout-opt):
  // - Re-evaluate whether Vec=8 is profitable for bf16/fp16 on the target GPUs.
  // - Push more of the q/k repeat mapping into compile-time CuTe layout transforms.
  // - Revisit whether a shared-memory staging path is worthwhile after profiling.
  // - Consider widening the a/b writeback path if it shows up in profiling.
  constexpr int kVec = Shape::kLayoutVec;
  static_assert(kHeadDimV % kVec == 0);
  static_assert(kHeadDimQK == kHeadDimV);
  static_assert(kHeadDimQK % kVec == 0);

  const int token_idx = static_cast<int>(blockIdx.y);
  const int hv = static_cast<int>(blockIdx.x);
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
  auto out_layout = make_layout(
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

  // Current version uses a direct GMEM->GMEM vector copy path. This keeps the
  // kernel simple while already removing the scalar-copy bottleneck from the
  // first draft. More aggressive staging/copy strategies should be driven by
  // profiling rather than added pre-emptively.
  for (int vec_idx = tid; vec_idx < kHeadDimV / kVec; vec_idx += blockDim.x) {
    const int d = vec_idx * kVec;
    const int q_src_idx = crd2idx(make_coord(mapped_h, d), qk_src_layout);
    const int k_src_idx = crd2idx(make_coord(mapped_h, d), qk_src_layout);
    const int v_src_idx = crd2idx(make_coord(hv, d), v_src_layout);
    const int dst_idx = crd2idx(make_coord(hv, d), out_layout);

    copy_vec_contiguous<scalar_t, kVec>(q_dst_ptr + dst_idx, q_src_ptr + q_src_idx);
    copy_vec_contiguous<scalar_t, kVec>(k_dst_ptr + dst_idx, k_src_ptr + k_src_idx);
    copy_vec_contiguous<scalar_t, kVec>(v_dst_ptr + dst_idx, v_src_ptr + v_src_idx);
  }

  if (tid == 0) {
    // TODO(qwen35-layout-opt): If a/b copy becomes measurable, fuse a wider
    // per-head copy path here instead of scalar head writes.
    const int head_idx = crd2idx(make_coord(hv), head_layout);
    const int64_t token_head_offset = static_cast<int64_t>(token_idx) * kLocalVHeads + head_idx;
    a_kernel[token_head_offset] = a[token_head_offset];
    b_kernel[token_head_offset] = b[token_head_offset];
  }
}

} // namespace cula::qwen35::decode
