#pragma once

// NOTE: This header is included from .cu files in csrc/ (parent directory).
// All includes use csrc/ as the root include path.
#include "kda_fwd_common.cuh"
#include "kda_bwd/helpers.h"
#include "kda_bwd/gemm.h"
#include "kda_bwd/utils.h"

#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "kda_bwd/kda_fwd_intra_mainloop_sm100.hpp"

namespace sm100 {

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
    using SharedMemoryPlan       = typename Mainloop::SharedMemoryPlan;
    using TileScheduler          = typename Mainloop::TileScheduler;
    using ClusterShape           = typename Mainloop::ClusterShape;

    // SMEM layouts (for TMA descriptor construction in host launcher)
    using SmemLayoutInputBF16    = typename Mainloop::SmemLayoutInputBF16;
    using SmemLayoutInputFP32    = typename Mainloop::SmemLayoutInputFP32;

    // TMA params (for host launcher)
    template <typename ShapeQKG, typename TMA_Q, typename TMA_K, typename TMA_G>
    using TmaParams = typename Mainloop::template TmaParams<ShapeQKG, TMA_Q, TMA_K, TMA_G>;

    // Pipeline types (for construction in operator())
    using PipelineQ              = typename Mainloop::PipelineQ;
    using PipelineK              = typename Mainloop::PipelineK;
    using PipelineG              = typename Mainloop::PipelineG;
    using PipelineBeta           = typename Mainloop::PipelineBeta;
    using PipelineQKGInterReady    = typename Mainloop::PipelineQKGInterReady;
    using PipelineQKDone         = typename Mainloop::PipelineQKDone;
    using PipelineKKInvReady     = typename Mainloop::PipelineKKInvReady;

    // Pipeline state types
    using PipelineStateQ         = typename Mainloop::PipelineStateQ;
    using PipelineStateK         = typename Mainloop::PipelineStateK;
    using PipelineStateG         = typename Mainloop::PipelineStateG;
    using PipelineStateBeta      = typename Mainloop::PipelineStateBeta;
    using PipelineStateQKGInter  = typename Mainloop::PipelineStateQKGInter;
    using PipelineStateQKDone    = typename Mainloop::PipelineStateQKDone;
    using PipelineStateKKInv     = typename Mainloop::PipelineStateKKInv;

    // Constants forwarded from Mainloop
    static constexpr int NUM_THREADS         = Mainloop::NUM_THREADS;
    static constexpr int NUM_CE_THREADS      = Mainloop::NUM_CE_THREADS;
    static constexpr int NUM_INVERSE_THREADS = Mainloop::NUM_INVERSE_THREADS;
    static constexpr int NUM_MMA_THREADS     = Mainloop::NUM_MMA_THREADS;
    static constexpr int NUM_LOAD_THREADS    = Mainloop::NUM_LOAD_THREADS;
    static constexpr int NUM_EMPTY_THREADS   = Mainloop::NUM_EMPTY_THREADS;
    static constexpr int NUM_TILE_CONSUMERS  = Mainloop::NUM_TILE_CONSUMERS;
    static constexpr int NUM_BUF_VALUE       = Mainloop::NUM_BUF_VALUE;

    // ===================== Kernel-only Constants =====================
    static constexpr int REG_COMPUTE  = 168;
    static constexpr int REG_LOAD     = 64;
    static constexpr int REG_INVERSE  = 104;

    // ===================== Warp Roles =====================
    enum class WarpRole {
        Empty = 0x0, Load = 0x1, Mma = 0x2, Compute = 0x3, Epilogue = 0x4,
        ComputeEpilogue = 0x5, Inverse = 0x6
    };

    // Warp layout (16 warps, 512 threads):
    //   warp  0- 7  (thread   0-255): ComputeEpilogue  — WG0+WG1
    //   warp  8-11  (thread 256-383): Inverse           — 1 warpgroup for inv(KK)
    //   warp  12    (thread 384-415): Mma               — 1 warp, elect_one
    //   warp  13    (thread 416-447): Load              — 1 warp, elect_one
    //   warp 14-15  (thread 448-511): Empty             — 2 warps for beta loading
    static constexpr unsigned long long kWarpAssignment = 0x12'6666'5555'5555ull;

    CUTLASS_DEVICE static WarpRole warp_idx_to_role(int warp_idx) {
        return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
    }

    // ===================================================================
    // operator(): the kernel entry point
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void operator()(
        const KDA_fwd_intra_params &params,
        const TmaParamsT &tma_params) {

        const int warpgroup_idx    = cutlass::canonical_warp_group_idx();
        const int idx_in_warpgroup = threadIdx.x % 128;
        const int warp_idx         = cutlass::canonical_warp_idx_sync();
        const int idx_in_warp      = threadIdx.x % 32;
        auto role = warp_idx_to_role(warp_idx);
        int lane_predicate = cute::elect_one_sync();
        TileScheduler tile_scheduler(params.tile_scheduler_params);

        extern __shared__ char shared_buf[];
        SharedMemoryPlan *shared_plan = reinterpret_cast<SharedMemoryPlan*>(shared_buf);

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

        // === TMA load pipelines: Q, K, G ===
        typename PipelineQ::Params q_pipe_params;
        q_pipe_params.transaction_bytes = sizeof(bf16) * cosize_v<SmemLayoutInputBF16>;
        q_pipe_params.is_leader    = lane_predicate && (role == WarpRole::Load);
        q_pipe_params.num_consumers = NUM_CE_THREADS;

        typename PipelineK::Params k_pipe_params;
        k_pipe_params.transaction_bytes = sizeof(bf16) * cosize_v<SmemLayoutInputBF16>;
        k_pipe_params.is_leader    = lane_predicate && (role == WarpRole::Load);
        k_pipe_params.num_consumers = NUM_CE_THREADS;

        typename PipelineG::Params g_pipe_params;
        g_pipe_params.transaction_bytes = sizeof(float) * cosize_v<SmemLayoutInputFP32>;
        g_pipe_params.is_leader    = lane_predicate && (role == WarpRole::Load);
        g_pipe_params.num_consumers = NUM_CE_THREADS;

        if (role == WarpRole::Load) {
            q_pipe_params.role = PipelineQ::ThreadCategory::Producer;
            k_pipe_params.role = PipelineK::ThreadCategory::Producer;
            g_pipe_params.role = PipelineG::ThreadCategory::Producer;
        } else if (role == WarpRole::ComputeEpilogue) {
            q_pipe_params.role = PipelineQ::ThreadCategory::Consumer;
            k_pipe_params.role = PipelineK::ThreadCategory::Consumer;
            g_pipe_params.role = PipelineG::ThreadCategory::Consumer;
        }

        // === Beta pipeline ===
        typename PipelineBeta::Params beta_pipe_params;
        beta_pipe_params.producer_arv_count = NUM_EMPTY_THREADS;
        beta_pipe_params.consumer_arv_count = NUM_CE_THREADS;
        if (role == WarpRole::Empty) {
            beta_pipe_params.role = PipelineBeta::ThreadCategory::Producer;
        } else if (role == WarpRole::ComputeEpilogue) {
            beta_pipe_params.role = PipelineBeta::ThreadCategory::Consumer;
        }

        // === CE -> MMA pipelines ===
        typename PipelineQKGInterReady::Params qkg_inter_pipe_params;
        qkg_inter_pipe_params.producer_arv_count = NUM_CE_THREADS;
        qkg_inter_pipe_params.consumer_arv_count = NUM_MMA_THREADS;

        if (role == WarpRole::ComputeEpilogue) {
            qkg_inter_pipe_params.role   = PipelineQKGInterReady::ThreadCategory::Producer;
        } else if (role == WarpRole::Mma) {
            qkg_inter_pipe_params.role   = PipelineQKGInterReady::ThreadCategory::Consumer;
        }

        // === MMA -> CE pipelines (UMMA) ===
        typename PipelineQKDone::Params qk_done_pipe_params;
        qk_done_pipe_params.producer_arv_count = NUM_MMA_THREADS;
        qk_done_pipe_params.consumer_arv_count = NUM_CE_THREADS;

        if (role == WarpRole::Mma) {
            qk_done_pipe_params.role = PipelineQKDone::ThreadCategory::Producer;
        } else if (role == WarpRole::ComputeEpilogue) {
            qk_done_pipe_params.role = PipelineQKDone::ThreadCategory::Consumer;
        }

        // === CE -> Inverse pipeline ===
        typename PipelineKKInvReady::Params kk_inv_pipe_params;
        kk_inv_pipe_params.producer_arv_count = NUM_CE_THREADS;
        kk_inv_pipe_params.consumer_arv_count = NUM_INVERSE_THREADS;
        if (role == WarpRole::ComputeEpilogue) {
            kk_inv_pipe_params.role = PipelineKKInvReady::ThreadCategory::Producer;
        } else if (role == WarpRole::Inverse) {
            kk_inv_pipe_params.role = PipelineKKInvReady::ThreadCategory::Consumer;
        }

        // ---------------------------------------------------------------
        // Construct pipeline objects
        // ---------------------------------------------------------------
        PipelineQ   q_pipeline(shared_plan->pipe_q_storage, q_pipe_params, ClusterShape{});
        PipelineK   k_pipeline(shared_plan->pipe_k_storage, k_pipe_params, ClusterShape{});
        PipelineG   g_pipeline(shared_plan->pipe_g_storage, g_pipe_params, ClusterShape{});

        PipelineBeta beta_pipeline(shared_plan->pipe_beta_storage, beta_pipe_params, /*InitBarriers*/cute::true_type{});

        // PipelineQKGInterReady   qkg_inter_pipeline(shared_plan->pipe_qkg_inter_storage, qkg_inter_pipe_params, /*InitBarriers*/cute::true_type{});
        PipelineQKGInterReady   qkg_inter_pipeline(shared_plan->pipe_qkg_inter_storage, qkg_inter_pipe_params, ClusterShape{});

        PipelineQKDone qk_done_pipeline(shared_plan->pipe_qk_done_storage, qk_done_pipe_params, /*InitBarriers*/cute::true_type{});

        PipelineKKInvReady kk_inv_pipeline(shared_plan->pipe_kk_inv_storage, kk_inv_pipe_params, /*InitBarriers*/cute::true_type{});

        // ---------------------------------------------------------------
        // Initialize pipeline states
        // ---------------------------------------------------------------
        PipelineStateQ q_pipe_state_read;
        PipelineStateQ q_pipe_state_write = cutlass::make_producer_start_state<PipelineQ>();
        PipelineStateK k_pipe_state_read;
        PipelineStateK k_pipe_state_write = cutlass::make_producer_start_state<PipelineK>();
        PipelineStateG g_pipe_state_read;
        PipelineStateG g_pipe_state_write = cutlass::make_producer_start_state<PipelineG>();

        PipelineStateBeta beta_pipe_state_read;
        PipelineStateBeta beta_pipe_state_write = cutlass::make_producer_start_state<PipelineBeta>();

        PipelineStateQKGInter   qkg_inter_pipe_state_read;
        PipelineStateQKGInter   qkg_inter_pipe_state_write = cutlass::make_producer_start_state<PipelineQKGInterReady>();

        PipelineStateQKDone qk_done_pipe_state_read;
        PipelineStateQKDone qk_done_pipe_state_write = cutlass::make_producer_start_state<PipelineQKDone>();

        PipelineStateKKInv kk_inv_pipe_state_read;
        PipelineStateKKInv kk_inv_pipe_state_write = cutlass::make_producer_start_state<PipelineKKInvReady>();

        // Barrier sync after pipeline construction
        __syncthreads();

        int *chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int *cu_len_ptr = (int*)params.cu_seqlens_ptr;
        int total_tiles = tile_scheduler.total_tiles();

        // =======================================================================
        // Dispatch to warp-specialized persistent loops (Mainloop)
        // =======================================================================
        Mainloop mainloop;

        if (role == WarpRole::ComputeEpilogue) {
            cutlass::arch::warpgroup_reg_alloc<REG_COMPUTE>();
            mainloop.compute_epilogue_loop(
                params, tma_params, shared_plan, tile_scheduler,
                q_pipeline, q_pipe_state_read,
                k_pipeline, k_pipe_state_read,
                g_pipeline, g_pipe_state_read,
                qkg_inter_pipeline, qkg_inter_pipe_state_write,
                qk_done_pipeline, qk_done_pipe_state_read,
                beta_pipeline, beta_pipe_state_read,
                kk_inv_pipeline, kk_inv_pipe_state_write,
                chunk_indices_ptr, cu_len_ptr, total_tiles
            );

        } else if (role == WarpRole::Mma) {
            cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
            mainloop.mma_loop(
                params, tma_params, shared_plan, tile_scheduler,
                qkg_inter_pipeline, qkg_inter_pipe_state_read,
                qk_done_pipeline, qk_done_pipe_state_write,
                chunk_indices_ptr, cu_len_ptr, total_tiles
            );

        } else if (role == WarpRole::Load) {
            cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
            mainloop.load_loop(
                params, tma_params, shared_plan, tile_scheduler,
                q_pipeline, q_pipe_state_write,
                k_pipeline, k_pipe_state_write,
                g_pipeline, g_pipe_state_write,
                chunk_indices_ptr, cu_len_ptr, total_tiles
            );

        } else if (role == WarpRole::Inverse) {
            cutlass::arch::warpgroup_reg_dealloc<REG_INVERSE>();
            mainloop.inverse_loop(
                params, tma_params, shared_plan, tile_scheduler,
                kk_inv_pipeline, kk_inv_pipe_state_read,
                chunk_indices_ptr, cu_len_ptr, total_tiles
            );

        } else {
            // WarpRole::Empty — beta loading
            cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
            mainloop.empty_loop(
                params, tma_params, shared_plan, tile_scheduler,
                beta_pipeline, beta_pipe_state_write,
                chunk_indices_ptr, cu_len_ptr, total_tiles
            );
        }

        // === CLEANUP ===
        __syncthreads();
        if (warp_idx == 0 && cute::elect_one_sync()) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    }

};

// ===================================================================
// Default Kernel type: uses the self-contained mainloop
// ===================================================================
using KdaChunkFwdIntraKernelSm100Default =
    KdaChunkFwdIntraKernelSm100<KdaChunkFwdIntraMainloopSm100>;

// ===================================================================
// __global__ kernel wrapper (free function — CUDA requires this)
// ===================================================================
template <typename KernelT, typename TmaParamsT>
__global__ void __launch_bounds__(512, 1, 1)
kda_fwd_intra_sm100_kernel_entry(
    __grid_constant__ const KDA_fwd_intra_params params,
    __grid_constant__ const TmaParamsT tma_params) {
    KernelT kernel_obj;
    kernel_obj(params, tma_params);
}

// ===================================================================
// Host-side launcher: constructs TMA descriptors and launches kernel
// ===================================================================
inline void run_kda_fwd_intra_sm100_v2(KDA_fwd_intra_params &params, cudaStream_t stream) {
    using Kernel = KdaChunkFwdIntraKernelSm100Default;
    KDA_ASSERT(params.d % 32 == 0);

    auto shape_QKG  = make_shape(params.total_q_len, params.d, params.h);
    auto stride_QKG = make_stride(params.h * params.d, _1{}, params.d);

    // --- Build TMA descriptors ---
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q_ptr),
            make_layout(shape_QKG, stride_QKG)
        ),
        typename Kernel::SmemLayoutInputBF16{}
    );

    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.k_ptr),
            make_layout(shape_QKG, stride_QKG)
        ),
        typename Kernel::SmemLayoutInputBF16{}
    );

    auto tma_G = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.g_ptr),
            make_layout(shape_QKG, stride_QKG)
        ),
        typename Kernel::SmemLayoutInputFP32{}
    );

    // --- Pack TMA params ---
    typename Kernel::template TmaParams<
        decltype(shape_QKG),
        decltype(tma_Q), decltype(tma_K), decltype(tma_G)
    > tma_params = {
        shape_QKG,
        tma_Q,
        tma_K,
        tma_G,
    };

    // --- Launch config ---
    auto kernel_fn = &kda_fwd_intra_sm100_kernel_entry<Kernel, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_dim(Kernel::TileScheduler::get_grid_shape(params.tile_scheduler_params));
    dim3 block_dim(Kernel::NUM_THREADS, 1, 1);
    kernel_fn<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params);
}

} // namespace sm100