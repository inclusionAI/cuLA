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

#include "qwen35/prefill/qwen35_prefill_common.cuh"

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/layout.h>

namespace cula::qwen35::prefill::sm90 {

using namespace cute;

// First SM90 chunk shape for Qwen3.5 prefill.
//
// This intentionally only describes the TMA/WGMMA tiles.  The full chunk
// algorithm still needs a local chunk recurrence and inter-chunk state scan;
// those should be built on top of these traits instead of extending the scalar
// fallback kernel.
template <int kBlockT_ = 64, int kBlockV_ = 64, int kStages_ = 2>
struct Qwen35ChunkPrefillSm90Traits {
  static constexpr int kBlockT = kBlockT_;
  static constexpr int kBlockV = kBlockV_;
  static constexpr int kStages = kStages_;

  static_assert(kBlockT == 64 || kBlockT == 128, "GMMA chunk tiles expect BT=64 or BT=128.");
  static_assert(kBlockV == 64 || kBlockV == 128, "V chunk tiles expect BV=64 or BV=128.");
  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);

  using Element = cutlass::bfloat16_t;
  using Accumulator = float;
  static constexpr int kAlignment = 16 / sizeof(Element);

  using ClusterShape = Shape<_1, _1, _1>;
  using StageCount = cutlass::gemm::collective::StageCount<kStages>;

  // q/k/v are materialized by qwen35_layout_prefill as contiguous
  // [total_tokens, 48, 128].  The TMA tensor view below exposes them as
  // (token, dim, head), with dynamic strides:
  //   token stride = 48 * 128
  //   dim stride   = 1
  //   head stride  = 128
  using GmemStrideTDH = cute::tuple<int64_t, _1, int32_t>;

  using TileShapeQK = decltype(make_shape(Int<kBlockT>{}, Int<kBlockT>{}, Int<kHeadDimQK>{}));
  using TileShapeOV = decltype(make_shape(Int<kBlockT>{}, Int<kBlockV>{}, Int<kHeadDimQK>{}));

  // Q @ K^T => [BT, BT].  CollectiveBuilder selects GMMA and TMA-compatible
  // shared-memory layouts for SM90.
  using CollectiveQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      GmemStrideTDH,
      kAlignment,
      Element,
      GmemStrideTDH,
      kAlignment,
      Accumulator,
      TileShapeQK,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using TiledMmaQK = typename CollectiveQK::TiledMma;
  using SmemLayoutQ = typename CollectiveQK::SmemLayoutA;
  using SmemLayoutK = typename CollectiveQK::SmemLayoutB;
  using TmaQ = typename CollectiveQK::Params::TMA_A;
  using TmaK = typename CollectiveQK::Params::TMA_B;

  // Q @ state / local_value => [BT, BV].  This is the second core WGMMA shape
  // needed once chunk-local state summaries are available.
  using CollectiveOV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      GmemStrideTDH,
      kAlignment,
      Element,
      GmemStrideTDH,
      kAlignment,
      Accumulator,
      TileShapeOV,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using TiledMmaOV = typename CollectiveOV::TiledMma;
  using SmemLayoutOV_A = typename CollectiveOV::SmemLayoutA;
  using SmemLayoutOV_B = typename CollectiveOV::SmemLayoutB;
};

using Qwen35ChunkPrefillSm90DefaultTraits = Qwen35ChunkPrefillSm90Traits<64, 64, 2>;

} // namespace cula::qwen35::prefill::sm90
