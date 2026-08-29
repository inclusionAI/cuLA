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

#include "qwen35_chunk_prefill_traits_sm90.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <torch/extension.h>

namespace cula::qwen35::prefill::sm90 {

namespace {

using DefaultTraits = Qwen35ChunkPrefillSm90DefaultTraits;

static_assert(DefaultTraits::kBlockT == 64);
static_assert(DefaultTraits::kBlockV == 64);
static_assert(DefaultTraits::kStages == 2);
static_assert(size(typename DefaultTraits::TiledMmaQK{}) == 128);
static_assert(size(typename DefaultTraits::TiledMmaOV{}) == 128);
static_assert(cosize(typename DefaultTraits::SmemLayoutQ{}) > 0);
static_assert(cosize(typename DefaultTraits::SmemLayoutK{}) > 0);

void check_cutlass_status(cutlass::Status status, const char* what) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, what, " failed with CUTLASS status ", static_cast<int>(status));
}

template <typename scalar_t>
void run_qwen35_chunk_qk_prefill_sm90_impl(const at::Tensor& q, const at::Tensor& k, const at::Tensor& out) {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cute::tuple<int64_t, cute::_1, int64_t>;
  using LayoutB = cute::tuple<int64_t, cute::_1, int64_t>;
  using LayoutC = cute::tuple<int64_t, cute::_1, int64_t>;
  using LayoutD = LayoutC;

  constexpr int kAlignmentA = 16 / sizeof(ElementA);
  constexpr int kAlignmentB = 16 / sizeof(ElementB);
  constexpr int kAlignmentC = 16 / sizeof(ElementC);
  constexpr int kAlignmentD = 16 / sizeof(ElementD);

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
#if defined(CULA_SM100_ENABLED) || defined(CULA_SM103_ENABLED)
  using ArchTag = cutlass::arch::Sm100;
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
#else
  using ArchTag = cutlass::arch::Sm90;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
#endif
#if defined(CULA_SM100_ENABLED) || defined(CULA_SM103_ENABLED)
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
#else
  using EpilogueTileType = decltype(cute::take<0, 2>(TileShape{}));
#endif
  using FusionOperation =
      typename cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      kAlignmentC,
      ElementD,
      LayoutD,
      kAlignmentD,
      EpilogueSchedule,
      FusionOperation>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t HV = q.size(2);
  constexpr int K = kHeadDimQK;
  const int64_t L = B * HV;

  LayoutA stride_A{HV * K, cute::_1{}, K};
  LayoutB stride_B{HV * K, cute::_1{}, K};
  LayoutC stride_C{T, cute::_1{}, T * T};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(T), static_cast<int>(T), K, static_cast<int>(L)},
      {
          reinterpret_cast<ElementA const*>(q.data_ptr<scalar_t>()),
          stride_A,
          reinterpret_cast<ElementB const*>(k.data_ptr<scalar_t>()),
          stride_B,
      },
      {
          {1.0f, 0.0f},
          out.data_ptr<float>(),
          stride_C,
          out.data_ptr<float>(),
          stride_C,
      },
  };

  Gemm gemm;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size)}, q.options().dtype(at::kByte));
  check_cutlass_status(gemm.can_implement(arguments), "qwen35_chunk_qk_prefill_sm90 can_implement");
  check_cutlass_status(gemm.initialize(arguments, workspace.data_ptr(), at::cuda::getCurrentCUDAStream(q.device().index())), "qwen35_chunk_qk_prefill_sm90 initialize");
  check_cutlass_status(gemm.run(at::cuda::getCurrentCUDAStream(q.device().index())), "qwen35_chunk_qk_prefill_sm90 run");
}

} // namespace

void qwen35_chunk_qk_prefill_sm90(at::Tensor q, at::Tensor k, at::Tensor out) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous [B,T,HV,128]");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous [B,T,HV,128]");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous [B,HV,T,T]");
  TORCH_CHECK(q.dim() == 4, "q must be [B,T,HV,128]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q");
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t HV = q.size(2);
  TORCH_CHECK(decode::is_supported_local_v_heads(static_cast<int>(HV)), "expected local HV in {48, 24, 12, 6}, got ", HV);
  TORCH_CHECK(q.size(3) == kHeadDimQK, "expected D=128");
  TORCH_CHECK(out.sizes() == at::IntArrayRef({B, HV, T, T}), "out must be [B,HV,T,T]");

  const at::cuda::OptionalCUDAGuard device_guard(q.device());
  run_qwen35_chunk_qk_prefill_sm90_impl<c10::BFloat16>(q, k, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cula::qwen35::prefill::sm90
