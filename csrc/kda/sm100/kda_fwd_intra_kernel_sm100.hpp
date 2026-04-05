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

#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include "kerutils/kerutils.cuh"

#include "kda/sm100/kda_fwd_common.cuh"
#include "kda/sm100/kda_fwd_intra_mainloop_sm100.hpp"
#include "kda/sm100/sm100_umma_ext.hpp"

namespace kda::sm100 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using namespace cute;

// ===================================================================
// Kernel struct: KdaChunkFwdIntraKernelSm100
// Templated on Mainloop. Owns only kernel-level config (register
// counts, warp role dispatch) and delegates everything else to Mainloop.
// ===================================================================
template <typename Mainloop_>
struct KdaChunkFwdIntraKernelSm100 {
    // ===================== Mainloop alias =====================
    using Mainloop = Mainloop_;

    // ===================== Import types from Mainloop =====================
    using SharedMemoryPlan = typename Mainloop::SharedMemoryPlan;
    using TileScheduler = typename Mainloop::TileScheduler;

    // SMEM layouts (for TMA descriptor construction in host launcher)
    using SmemLayoutInputBF16 = typename Mainloop::SmemLayoutInputBF16;
    using SmemLayoutInputFP32 = typename Mainloop::SmemLayoutInputFP32;

    // TMA params (for host launcher)
    template <typename ShapeQKG, typename TMA_Q, typename TMA_K, typename TMA_G>
    using TmaParams = typename Mainloop::template TmaParams<ShapeQKG, TMA_Q, TMA_K, TMA_G>;

    // Pipeline types (for construction in operator())
    using PipelineQKG = typename Mainloop::PipelineQKG;
    using PipelineBeta = typename Mainloop::PipelineBeta;
    using PipelineQKGInterReady = typename Mainloop::PipelineQKGInterReady;
    using PipelineQKDone = typename Mainloop::PipelineQKDone;
    using PipelineKKInvReady = typename Mainloop::PipelineKKInvReady;

    // Pipeline state types
    using PipelineStateQKG = typename Mainloop::PipelineStateQKG;
    using PipelineStateBeta = typename Mainloop::PipelineStateBeta;
    using PipelineStateQKGInter = typename Mainloop::PipelineStateQKGInter;
    using PipelineStateQKDone = typename Mainloop::PipelineStateQKDone;
    using PipelineStateKKInv = typename Mainloop::PipelineStateKKInv;

    using ClusterShape = Shape<_1, _1, _1>;

    // ===================== Thread Count Constants =====================
    static constexpr int NumTotalThreads = 128 * 4;  // 512
    static constexpr int NumCudaCoreThreads = 256;   // warp 0-7 (2 warpgroups)
    static constexpr int NumInverseThreads = 128;    // warp 8-11 (1 warpgroup)
    static constexpr int NumMmaThreads = 32;         // warp 12
    static constexpr int NumLoadTmaThreads = 1;      // elect_one in warp 13
    static constexpr int NumLoadBetaThreads = 64;    // warp 14-15

    // ===================== Kernel-only Constants =====================
    static constexpr int NumCudaCoreRegs = 160;
    static constexpr int NumLoadRegs = 80;
    static constexpr int NumInverseRegs = 104;

    // ===================== Warp Roles =====================
    enum class WarpRole { ComputeCudaCore, Inverse, Mma, Load, LoadBeta, Empty };

    // Warp layout (16 warps, 512 threads):
    //   warp  0- 7  (thread   0-255): ComputeCudaCore   — WG0+WG1
    //   warp  8-11  (thread 256-383): Inverse           — 1 warpgroup for inv(KK)
    //   warp  12    (thread 384-415): Mma               — 1 warp
    //   warp  13    (thread 416-447): Load              — 1 warp, elect_one
    //   warp 14-15  (thread 448-511): LoadBeta          — 2 warps for beta loading
    CUTLASS_DEVICE static WarpRole
    warp_idx_to_role(int warp_idx) {
        int wg_idx = warp_idx / 4;
        if (wg_idx == 0 || wg_idx == 1)
            return WarpRole::ComputeCudaCore;
        if (wg_idx == 2)
            return WarpRole::Inverse;
        if (warp_idx == 12)
            return WarpRole::Mma;
        if (warp_idx == 13)
            return WarpRole::Load;
        if (warp_idx == 14 || warp_idx == 15)
            return WarpRole::LoadBeta;
        return WarpRole::Empty;  // not used
    }

    // ===================================================================
    // operator(): the kernel entry point
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    operator()(const KDA_fwd_intra_params& params, const TmaParamsT& tma_params) {
        const int warpgroup_idx = cutlass::canonical_warp_group_idx();
        const int warp_idx = cutlass::canonical_warp_idx_sync();
        auto role = warp_idx_to_role(warp_idx);
        int lane_predicate = cute::elect_one_sync();
        TileScheduler tile_scheduler(params.tile_scheduler_params);

        extern __shared__ char shared_buf[];
        SharedMemoryPlan* shared_plan = reinterpret_cast<SharedMemoryPlan*>(shared_buf);

        // Prefetch TMA descriptors
        if (warp_idx == 0 && lane_predicate) {
            cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
        }

        // Allocate TMEM (warp 0 only)
        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().allocate(512, shared_plan->tmem_start_addr.data());
            cute::TMEM::Allocator1Sm().release_allocation_lock();
        }

        // ---------------------------------------------------------------
        // Configure pipeline params per role
        // ---------------------------------------------------------------

        // === Unified TMA load pipeline: Q + K + G ===
        typename PipelineQKG::Params qkg_load_pipe_params;
        qkg_load_pipe_params.transaction_bytes = sizeof(ku::bf16) * cosize_v<SmemLayoutInputBF16> +  // Q
                                                 sizeof(ku::bf16) * cosize_v<SmemLayoutInputBF16> +  // K
                                                 sizeof(float) * cosize_v<SmemLayoutInputFP32>;      // G
        qkg_load_pipe_params.is_leader = lane_predicate && (role == WarpRole::Load);
        qkg_load_pipe_params.num_consumers = NumCudaCoreThreads;

        if (role == WarpRole::Load) {
            qkg_load_pipe_params.role = PipelineQKG::ThreadCategory::Producer;
        } else if (role == WarpRole::ComputeCudaCore) {
            qkg_load_pipe_params.role = PipelineQKG::ThreadCategory::Consumer;
        }

        // === Beta pipeline ===
        typename PipelineBeta::Params beta_pipe_params;
        beta_pipe_params.producer_arv_count = NumLoadBetaThreads;
        beta_pipe_params.consumer_arv_count = NumCudaCoreThreads;
        if (role == WarpRole::LoadBeta) {
            beta_pipe_params.role = PipelineBeta::ThreadCategory::Producer;
        } else if (role == WarpRole::ComputeCudaCore) {
            beta_pipe_params.role = PipelineBeta::ThreadCategory::Consumer;
        }

        // === CudaCore -> MMA pipelines ===
        typename PipelineQKGInterReady::Params qkg_inter_pipe_params;
        qkg_inter_pipe_params.producer_arv_count = NumCudaCoreThreads;
        // NOTE: only one threads calls consumer_release (umma_arrive)
        qkg_inter_pipe_params.consumer_arv_count = 1;

        if (role == WarpRole::ComputeCudaCore) {
            qkg_inter_pipe_params.role = PipelineQKGInterReady::ThreadCategory::Producer;
        } else if (role == WarpRole::Mma) {
            qkg_inter_pipe_params.role = PipelineQKGInterReady::ThreadCategory::Consumer;
        }

        // === MMA -> CudaCore pipelines (UMMA) ===
        typename PipelineQKDone::Params qk_done_pipe_params;
        qk_done_pipe_params.producer_arv_count = NumMmaThreads;
        qk_done_pipe_params.consumer_arv_count = NumCudaCoreThreads;

        if (role == WarpRole::Mma) {
            qk_done_pipe_params.role = PipelineQKDone::ThreadCategory::Producer;
        } else if (role == WarpRole::ComputeCudaCore) {
            qk_done_pipe_params.role = PipelineQKDone::ThreadCategory::Consumer;
        }

        // === CudaCore -> Inverse pipeline ===
        typename PipelineKKInvReady::Params kk_inv_pipe_params;
        kk_inv_pipe_params.producer_arv_count = NumCudaCoreThreads;
        kk_inv_pipe_params.consumer_arv_count = NumInverseThreads;
        if (role == WarpRole::ComputeCudaCore) {
            kk_inv_pipe_params.role = PipelineKKInvReady::ThreadCategory::Producer;
        } else if (role == WarpRole::Inverse) {
            kk_inv_pipe_params.role = PipelineKKInvReady::ThreadCategory::Consumer;
        }

        // ---------------------------------------------------------------
        // Construct pipeline objects
        // ---------------------------------------------------------------
        PipelineQKG qkg_load_pipeline(shared_plan->pipe_qkg_load_storage, qkg_load_pipe_params, ClusterShape{});

        PipelineBeta beta_pipeline(
            shared_plan->pipe_beta_storage,
            beta_pipe_params,
            /*InitBarriers*/ cute::true_type{});

        // PipelineQKGInterReady   qkg_inter_pipeline(shared_plan->pipe_qkg_inter_storage, qkg_inter_pipe_params,
        // /*InitBarriers*/cute::true_type{});
        PipelineQKGInterReady qkg_inter_pipeline(
            shared_plan->pipe_qkg_inter_storage, qkg_inter_pipe_params, ClusterShape{});

        PipelineQKDone qk_done_pipeline(
            shared_plan->pipe_qk_done_storage,
            qk_done_pipe_params,
            /*InitBarriers*/ cute::true_type{});

        PipelineKKInvReady kk_inv_pipeline(
            shared_plan->pipe_kk_inv_storage,
            kk_inv_pipe_params,
            /*InitBarriers*/ cute::true_type{});

        // ---------------------------------------------------------------
        // Initialize pipeline states
        // ---------------------------------------------------------------
        PipelineStateQKG qkg_load_pipe_state_read;
        PipelineStateQKG qkg_load_pipe_state_write = cutlass::make_producer_start_state<PipelineQKG>();

        PipelineStateBeta beta_pipe_state_read;
        PipelineStateBeta beta_pipe_state_write = cutlass::make_producer_start_state<PipelineBeta>();

        PipelineStateQKGInter qkg_inter_pipe_state_read;
        PipelineStateQKGInter qkg_inter_pipe_state_write = cutlass::make_producer_start_state<PipelineQKGInterReady>();

        PipelineStateQKDone qk_done_pipe_state_read;
        PipelineStateQKDone qk_done_pipe_state_write = cutlass::make_producer_start_state<PipelineQKDone>();

        PipelineStateKKInv kk_inv_pipe_state_read;
        PipelineStateKKInv kk_inv_pipe_state_write = cutlass::make_producer_start_state<PipelineKKInvReady>();

        // Barrier sync after pipeline construction
        __syncthreads();

        // =======================================================================
        // Dispatch to warp-specialized persistent loops (Mainloop)
        // =======================================================================
        Mainloop mainloop;

        if (role == WarpRole::ComputeCudaCore) {
            cutlass::arch::warpgroup_reg_alloc<NumCudaCoreRegs>();
            mainloop.compute_cudacore_loop(
                params,
                tma_params,
                shared_plan,
                tile_scheduler,
                qkg_load_pipeline,
                qkg_load_pipe_state_read,
                qkg_inter_pipeline,
                qkg_inter_pipe_state_write,
                qk_done_pipeline,
                qk_done_pipe_state_read,
                beta_pipeline,
                beta_pipe_state_read,
                kk_inv_pipeline,
                kk_inv_pipe_state_write);

        } else if (role == WarpRole::Mma) {
            cutlass::arch::warpgroup_reg_dealloc<NumLoadRegs>();
            mainloop.mma_loop(
                params,
                tma_params,
                shared_plan,
                tile_scheduler,
                qkg_inter_pipeline,
                qkg_inter_pipe_state_read,
                qk_done_pipeline,
                qk_done_pipe_state_write);

        } else if (role == WarpRole::Load) {
            cutlass::arch::warpgroup_reg_dealloc<NumLoadRegs>();
            mainloop.load_loop(
                params, tma_params, shared_plan, tile_scheduler, qkg_load_pipeline, qkg_load_pipe_state_write);

        } else if (role == WarpRole::Inverse) {
            cutlass::arch::warpgroup_reg_dealloc<NumInverseRegs>();
            mainloop.inverse_loop(
                params, tma_params, shared_plan, tile_scheduler, kk_inv_pipeline, kk_inv_pipe_state_read);

        } else {
            cutlass::arch::warpgroup_reg_dealloc<NumLoadRegs>();
            mainloop.load_beta_loop(
                params, tma_params, shared_plan, tile_scheduler, beta_pipeline, beta_pipe_state_write);
        }

        // === CLEANUP ===
        __syncthreads();
        if (warp_idx == 0 && cute::elect_one_sync()) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    }
};

// ===================================================================
// Default Kernel types: parameterized by UseTF32Inverse × UnifiedGRef
// ===================================================================
using KdaChunkFwdIntraKernelSm100_TF32 = KdaChunkFwdIntraKernelSm100<KdaChunkFwdIntraMainloopSm100<true, false, true>>;

using KdaChunkFwdIntraKernelSm100_FP16 = KdaChunkFwdIntraKernelSm100<KdaChunkFwdIntraMainloopSm100<false, false, true>>;

using KdaChunkFwdIntraKernelSm100_TF32_GHalf =
    KdaChunkFwdIntraKernelSm100<KdaChunkFwdIntraMainloopSm100<true, false, false>>;

using KdaChunkFwdIntraKernelSm100_FP16_GHalf =
    KdaChunkFwdIntraKernelSm100<KdaChunkFwdIntraMainloopSm100<false, false, false>>;

// ===================================================================
// __global__ kernel wrapper (free function — CUDA requires this)
// ===================================================================
template <typename KernelT, typename TmaParamsT>
__global__ void
__launch_bounds__(512, 1, 1) kda_fwd_intra_sm100_kernel_entry(
    __grid_constant__ const KDA_fwd_intra_params params, __grid_constant__ const TmaParamsT tma_params) {
    KernelT kernel_obj;
    kernel_obj(params, tma_params);
}

// ===================================================================
// Host-side launcher: constructs TMA descriptors and launches kernel
// ===================================================================
template <typename Kernel>
inline void
run_kda_fwd_intra_sm100_impl_dispatch(KDA_fwd_intra_params& params, cudaStream_t stream) {
    auto shape_QKG = make_shape(params.total_q_len, params.d, params.h);
    auto stride_QKG = make_stride(params.h * params.d, _1{}, params.d);

    // --- Build TMA descriptors ---
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((ku::bf16*)params.q_ptr), make_layout(shape_QKG, stride_QKG)),
        typename Kernel::SmemLayoutInputBF16{});

    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((ku::bf16*)params.k_ptr), make_layout(shape_QKG, stride_QKG)),
        typename Kernel::SmemLayoutInputBF16{});

    auto tma_G = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.g_ptr), make_layout(shape_QKG, stride_QKG)),
        typename Kernel::SmemLayoutInputFP32{});

    // --- Pack TMA params ---
    typename Kernel::template TmaParams<decltype(shape_QKG), decltype(tma_Q), decltype(tma_K), decltype(tma_G)>
        tma_params = {
            shape_QKG,
            tma_Q,
            tma_K,
            tma_G,
        };

    // --- Launch config ---
    auto kernel_fn = &kda_fwd_intra_sm100_kernel_entry<Kernel, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_dim(Kernel::TileScheduler::get_grid_shape(params.tile_scheduler_params));
    dim3 block_dim(Kernel::NumTotalThreads, 1, 1);
    kernel_fn<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

// ===================================================================
// Runtime dispatch based on params.use_tf32_inverse and params.unified_gref
// ===================================================================
inline void
run_kda_fwd_intra_sm100_impl(KDA_fwd_intra_params& params, cudaStream_t stream) {
    if (params.use_tf32_inverse) {
        if (params.unified_gref) {
            run_kda_fwd_intra_sm100_impl_dispatch<KdaChunkFwdIntraKernelSm100_TF32>(params, stream);
        } else {
            run_kda_fwd_intra_sm100_impl_dispatch<KdaChunkFwdIntraKernelSm100_TF32_GHalf>(params, stream);
        }
    } else {
        if (params.unified_gref) {
            run_kda_fwd_intra_sm100_impl_dispatch<KdaChunkFwdIntraKernelSm100_FP16>(params, stream);
        } else {
            run_kda_fwd_intra_sm100_impl_dispatch<KdaChunkFwdIntraKernelSm100_FP16_GHalf>(params, stream);
        }
    }
}

}  // namespace kda::sm100