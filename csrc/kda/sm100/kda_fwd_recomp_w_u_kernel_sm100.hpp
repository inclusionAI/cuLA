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
#include "kda/sm100/kda_fwd_recomp_w_u_mainloop_sm100.hpp"

namespace kda::sm100 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using ku::bf16;
using namespace cute;

// ===================================================================
// Kernel struct: KdaChunkFwdRecompWUKernelSm100
// Templated on Mainloop. Owns only kernel-level config (register
// counts, warp role dispatch) and delegates everything else to Mainloop.
// ===================================================================
template <typename Mainloop_>
struct KdaChunkFwdRecompWUKernelSm100 {
    // ===================== Mainloop alias =====================
    using Mainloop = Mainloop_;

    // ===================== Import types from Mainloop =====================
    using SharedMemoryPlan = typename Mainloop::SharedMemoryPlan;
    using TileScheduler = typename Mainloop::TileScheduler;

    // SMEM layouts (for TMA descriptor construction in host launcher)
    using SmemLayoutInputBF16 = typename Mainloop::SmemLayoutInputBF16;
    using SmemLayoutInputFP32 = typename Mainloop::SmemLayoutInputFP32;
    using SmemLayoutInputAkkBF16 = typename Mainloop::SmemLayoutInputAkkBF16;

    // TMA params (for host launcher)
    template <
        typename ShapeKVG,
        typename ShapeAkk,
        typename TMA_V,
        typename TMA_K,
        typename TMA_G,
        typename TMA_Akk,
        typename TMA_Q = int>
    using TmaParams = typename Mainloop::template TmaParams<ShapeKVG, ShapeAkk, TMA_V, TMA_K, TMA_G, TMA_Akk, TMA_Q>;

    // Pipeline types (for construction in operator())
    using PipelineA = typename Mainloop::PipelineA;
    using PipelineK = typename Mainloop::PipelineK;
    using PipelineG = typename Mainloop::PipelineG;
    using PipelineV = typename Mainloop::PipelineV;
    using PipelineQ = typename Mainloop::PipelineQ;
    using PipelineBeta = typename Mainloop::PipelineBeta;
    using PipelinePrologueReady = typename Mainloop::PipelinePrologueReady;
    using PipelineAccDone = typename Mainloop::PipelineAccDone;

    // Pipeline state types
    using PipelineStateA = typename Mainloop::PipelineStateA;
    using PipelineStateK = typename Mainloop::PipelineStateK;
    using PipelineStateG = typename Mainloop::PipelineStateG;
    using PipelineStateV = typename Mainloop::PipelineStateV;
    using PipelineStateQ = typename Mainloop::PipelineStateQ;
    using PipelineStateBeta = typename Mainloop::PipelineStateBeta;
    using PipelineStatePrologueReady = typename Mainloop::PipelineStatePrologueReady;
    using PipelineStateAccDone = typename Mainloop::PipelineStateAccDone;

    using ClusterShape = Shape<_1, _1, _1>;

    // ===================== Thread Count Constants =====================
    // Layout: 384 threads = 12 warps = 3 Warp Groups
    //   WG0 (warp 0-3,   thread   0-127): Prologue (element-wise K_proc/V_proc → signal MMA)
    //   WG1 (warp 4-7,   thread 128-255): Epilogue (kg element-wise + MMA result store w/u → GMEM)
    //   WG2 (warp 8-11,  thread 256-383): Load/MMA/Aux
    //     warp 8    (thread 256-287): MMA warp (elect_one executes UMMA)
    //     warp 9    (thread 288-319): Load warp (elect_one executes TMA)
    //     warp 10-11 (thread 320-383): Aux warps (beta loading)
    static constexpr int NumTotalThreads = 384;
    static constexpr int NumPrologueThreads = cutlass::NumThreadsPerWarpGroup;  // 128 threads (WG0, warp 0-3)
    static constexpr int NumEpilogueThreads = cutlass::NumThreadsPerWarpGroup;  // 128 threads (WG1, warp 4-7)
    static constexpr int NumMmaThreads = 32;                                    // warp 8
    static constexpr int NumLoadTmaThreads = 1;                                 // elect_one in warp 9
    static constexpr int NumLoadAuxThreads = 64;                                // warp 10-11

    // ===================== Kernel-only Constants =====================
    static constexpr bool StoreQG = Mainloop::StoreQG;

    // NOTE: NVCC 12.9 and 13.0 have performance diffs on the same register config based on our testing
#if CUDA_VERSION_CHECK >= 13000
    static constexpr int NumPrologueRegs = StoreQG ? 232 : 224;  // WG0: element-wise + R2T Akk
    static constexpr int NumEpilogueRegs = StoreQG ? 192 : 200;  // WG1: T2R acc + R2G store + kg
#else
    static constexpr int NumPrologueRegs = StoreQG ? 216 : 224;  // WG0: element-wise + R2T Akk
    static constexpr int NumEpilogueRegs = StoreQG ? 208 : 200;  // WG1: T2R acc + R2G store + kg
#endif
    static constexpr int NumLoadRegs = 80;  // WG2: TMA load + MMA + Aux

    // ===================== Warp Roles =====================
    enum class WarpRole {
        Prologue,  // WG0: warp 0-3, element-wise K_proc/V_proc → signal MMA
        Epilogue,  // WG1: warp 4-7, kg + w/u store → GMEM
        Mma,       // warp 8, UMMA instructions
        Load,      // warp 9, TMA loads
        LoadAux,   // warp 10-11, beta
        Empty
    };

    // Warp layout (12 warps, 384 threads):
    //   warp 0-3  (thread   0-127): Prologue (WG0)
    //   warp 4-7  (thread 128-255): Epilogue (WG1)
    //   warp 8    (thread 256-287): Mma
    //   warp 9    (thread 288-319): Load (TMA, elect_one)
    //   warp 10-11 (thread 320-383): LoadAux
    CUTLASS_DEVICE static WarpRole
    warp_idx_to_role(int warp_idx) {
        if (warp_idx <= 3)
            return WarpRole::Prologue;
        if (warp_idx >= 4 && warp_idx <= 7)
            return WarpRole::Epilogue;
        if (warp_idx == 8)
            return WarpRole::Mma;
        if (warp_idx == 9)
            return WarpRole::Load;
        if (warp_idx == 10 || warp_idx == 11)
            return WarpRole::LoadAux;
        return WarpRole::Empty;
    }

    // ===================================================================
    // operator(): the kernel entry point
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    operator()(const KDA_fwd_recomp_w_u_params& params, const TmaParamsT& tma_params) {
        const int warp_idx = cutlass::canonical_warp_idx_sync();
        auto role = warp_idx_to_role(warp_idx);
        int lane_predicate = cute::elect_one_sync();
        TileScheduler tile_scheduler(params.tile_scheduler_params);

        extern __shared__ char shared_buf[];
        SharedMemoryPlan* shared_plan = reinterpret_cast<SharedMemoryPlan*>(shared_buf);

        // Prefetch TMA descriptors
        if (warp_idx == 0 && lane_predicate) {
            cute::prefetch_tma_descriptor(tma_params.tma_akk.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_v.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
            if constexpr (StoreQG) {
                cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
            }
        }

        // Allocate TMEM (warp 0 only)
        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().allocate(512, shared_plan->tmem_start_addr.data());
            cute::TMEM::Allocator1Sm().release_allocation_lock();
        }

        // ---------------------------------------------------------------
        // Configure pipeline params per role
        // ---------------------------------------------------------------

        // === TMA load pipelines: A (Akk), K, V, G ===
        // PipelineA: Load(producer) → MMA warp(consumer)
        typename PipelineA::Params a_pipe_params;
        a_pipe_params.transaction_bytes = sizeof(bf16) * cosize_v<SmemLayoutInputAkkBF16>;
        a_pipe_params.is_leader = lane_predicate && (role == WarpRole::Load);
        a_pipe_params.num_consumers = cutlass::NumThreadsPerWarp;
        if (role == WarpRole::Load) {
            a_pipe_params.role = PipelineA::ThreadCategory::Producer;
        } else if (role == WarpRole::Mma) {
            a_pipe_params.role = PipelineA::ThreadCategory::Consumer;
        }

        // PipelineV: Load(producer) → Epilogue(consumer, 128 threads)
        // WG1 (Epilogue) needs V for V_proc computation
        typename PipelineV::Params v_pipe_params;
        v_pipe_params.transaction_bytes = sizeof(bf16) * cosize_v<SmemLayoutInputBF16>;
        v_pipe_params.is_leader = lane_predicate && (role == WarpRole::Load);
        v_pipe_params.num_consumers = NumEpilogueThreads;
        if (role == WarpRole::Load) {
            v_pipe_params.role = PipelineV::ThreadCategory::Producer;
        } else if (role == WarpRole::Epilogue) {
            v_pipe_params.role = PipelineV::ThreadCategory::Consumer;
        }

        // PipelineK: Load(producer) → Prologue(128) consumers
        typename PipelineK::Params k_pipe_params;
        k_pipe_params.transaction_bytes = sizeof(bf16) * cosize_v<SmemLayoutInputBF16>;
        k_pipe_params.is_leader = lane_predicate && (role == WarpRole::Load);
        k_pipe_params.num_consumers = NumPrologueThreads;
        if (role == WarpRole::Load) {
            k_pipe_params.role = PipelineK::ThreadCategory::Producer;
        } else if (role == WarpRole::Prologue) {
            k_pipe_params.role = PipelineK::ThreadCategory::Consumer;
        }

        // PipelineG: Load(producer) → Prologue(128) consumers
        typename PipelineG::Params g_pipe_params;
        g_pipe_params.transaction_bytes = sizeof(float) * cosize_v<SmemLayoutInputFP32>;
        g_pipe_params.is_leader = lane_predicate && (role == WarpRole::Load);
        g_pipe_params.num_consumers = NumPrologueThreads;
        if (role == WarpRole::Load) {
            g_pipe_params.role = PipelineG::ThreadCategory::Producer;
        } else if (role == WarpRole::Prologue) {
            g_pipe_params.role = PipelineG::ThreadCategory::Consumer;
        }

        // PipelineQ: Load(producer) → Prologue(128) consumers (only meaningful when StoreQG=true)
        typename PipelineQ::Params q_pipe_params;
        q_pipe_params.transaction_bytes = sizeof(bf16) * cosize_v<SmemLayoutInputBF16>;
        q_pipe_params.is_leader = lane_predicate && (role == WarpRole::Load);
        q_pipe_params.num_consumers = NumPrologueThreads;
        if (role == WarpRole::Load) {
            q_pipe_params.role = PipelineQ::ThreadCategory::Producer;
        } else if (role == WarpRole::Prologue) {
            q_pipe_params.role = PipelineQ::ThreadCategory::Consumer;
        }

        // === Beta pipeline: LoadAux(producer, 64 threads) → Prologue+Epilogue(consumer, 256 threads) ===
        // Both WG0 (Prologue) and WG1 (Epilogue) need beta
        typename PipelineBeta::Params beta_pipe_params;
        beta_pipe_params.producer_arv_count = NumLoadAuxThreads;
        beta_pipe_params.consumer_arv_count = NumPrologueThreads + NumEpilogueThreads;
        if (role == WarpRole::LoadAux) {
            beta_pipe_params.role = PipelineBeta::ThreadCategory::Producer;
        } else if (role == WarpRole::Prologue || role == WarpRole::Epilogue) {
            beta_pipe_params.role = PipelineBeta::ThreadCategory::Consumer;
        }

        // === Prologue → MMA pipelines ===

        // PipelinePrologueReady: Prologue+Epilogue(producer, 256 threads) → Mma(consumer, 32 threads)
        // Unified pipeline for both K and V prologue ready (co-produced by Prologue and Epilogue)
        typename PipelinePrologueReady::Params prologue_ready_pipe_params;
        prologue_ready_pipe_params.producer_arv_count = NumPrologueThreads + NumEpilogueThreads;
        prologue_ready_pipe_params.consumer_arv_count = NumMmaThreads;
        if (role == WarpRole::Prologue || role == WarpRole::Epilogue) {
            prologue_ready_pipe_params.role = PipelinePrologueReady::ThreadCategory::Producer;
        } else if (role == WarpRole::Mma) {
            prologue_ready_pipe_params.role = PipelinePrologueReady::ThreadCategory::Consumer;
        }

        // === MMA → Epilogue pipelines ===
        // PipelineAccDone: Mma(producer, elect_one = 1 thread) → Epilogue(consumer, 128 threads)
        // Unified pipeline for both W and U acc done (used sequentially)
        typename PipelineAccDone::Params acc_done_pipe_params;
        acc_done_pipe_params.producer_arv_count = 1;  // elect_one in MMA warp
        acc_done_pipe_params.consumer_arv_count = NumEpilogueThreads;
        if (role == WarpRole::Mma) {
            acc_done_pipe_params.role = PipelineAccDone::ThreadCategory::Producer;
        } else if (role == WarpRole::Epilogue) {
            acc_done_pipe_params.role = PipelineAccDone::ThreadCategory::Consumer;
        }

        // ---------------------------------------------------------------
        // Construct pipeline objects
        // ---------------------------------------------------------------
        // TMA pipelines (PipelineTmaAsync uses ClusterShape for barrier init)
        PipelineA a_pipeline(shared_plan->pipe_a_storage, a_pipe_params, ClusterShape{});
        PipelineK k_pipeline(shared_plan->pipe_k_storage, k_pipe_params, ClusterShape{});
        PipelineG g_pipeline(shared_plan->pipe_g_storage, g_pipe_params, ClusterShape{});
        PipelineV v_pipeline(shared_plan->pipe_v_storage, v_pipe_params, ClusterShape{});
        PipelineQ q_pipeline(shared_plan->pipe_q_storage, q_pipe_params, ClusterShape{});

        // PipelineAsync pipelines (use true_type for barrier init)
        PipelineBeta beta_pipeline(
            shared_plan->pipe_beta_storage,
            beta_pipe_params,
            /*InitBarriers*/ cute::true_type{});

        PipelinePrologueReady prologue_ready_pipeline(
            shared_plan->pipe_prologue_ready_storage, prologue_ready_pipe_params, /*InitBarriers*/ cute::true_type{});

        PipelineAccDone acc_done_pipeline(shared_plan->pipe_acc_done_storage, acc_done_pipe_params, ClusterShape{});

        // ---------------------------------------------------------------
        // Initialize pipeline states
        // ---------------------------------------------------------------
        PipelineStateA a_pipe_state_read;
        PipelineStateA a_pipe_state_write = cutlass::make_producer_start_state<PipelineA>();
        PipelineStateK k_pipe_state_read;
        PipelineStateK k_pipe_state_write = cutlass::make_producer_start_state<PipelineK>();
        PipelineStateG g_pipe_state_read;
        PipelineStateG g_pipe_state_write = cutlass::make_producer_start_state<PipelineG>();
        PipelineStateV v_pipe_state_read;
        PipelineStateV v_pipe_state_write = cutlass::make_producer_start_state<PipelineV>();
        PipelineStateQ q_pipe_state_read;
        PipelineStateQ q_pipe_state_write = cutlass::make_producer_start_state<PipelineQ>();

        PipelineStateBeta beta_pipe_state_read;
        PipelineStateBeta beta_pipe_state_write = cutlass::make_producer_start_state<PipelineBeta>();

        PipelineStatePrologueReady prologue_ready_pipe_state_read;
        PipelineStatePrologueReady prologue_ready_pipe_state_write =
            cutlass::make_producer_start_state<PipelinePrologueReady>();

        PipelineStateAccDone acc_done_pipe_state_read;
        PipelineStateAccDone acc_done_pipe_state_write = cutlass::make_producer_start_state<PipelineAccDone>();

        // Barrier sync after pipeline construction
        __syncthreads();

        // =======================================================================
        // Dispatch to warp-specialized persistent loops (Mainloop)
        // =======================================================================
        Mainloop mainloop;

        if (role == WarpRole::Prologue) {
            // WG0 (warp 0-3, 128 threads): Element-wise K_proc → co-signal MMA, KG → GMEM
            cutlass::arch::warpgroup_reg_alloc<NumPrologueRegs>();
            mainloop.prologue_loop(
                params,
                tma_params,
                shared_plan,
                tile_scheduler,
                // TMA pipelines (consumer): K, G
                k_pipeline,
                k_pipe_state_read,
                g_pipeline,
                g_pipe_state_read,
                // Beta pipeline (consumer)
                beta_pipeline,
                beta_pipe_state_read,
                // TMA pipeline (consumer): Q (only used when StoreQG=true)
                q_pipeline,
                q_pipe_state_read,
                // Prologue -> MMA pipeline (co-producer with Epilogue)
                prologue_ready_pipeline,
                prologue_ready_pipe_state_write);

        } else if (role == WarpRole::Epilogue) {
            // WG1 (warp 4-7, 128 threads): V_proc → co-signal MMA, w/u store → GMEM
            cutlass::arch::warpgroup_reg_alloc<NumEpilogueRegs>();
            mainloop.epilogue_loop(
                params,
                tma_params,
                shared_plan,
                tile_scheduler,
                // TMA pipeline (consumer): V
                v_pipeline,
                v_pipe_state_read,
                // Beta pipeline (consumer)
                beta_pipeline,
                beta_pipe_state_read,
                // Prologue -> MMA pipeline (co-producer with Prologue)
                prologue_ready_pipeline,
                prologue_ready_pipe_state_write,
                // MMA -> Epilogue pipeline (consumer)
                acc_done_pipeline,
                acc_done_pipe_state_read);

        } else if (role == WarpRole::Mma) {
            cutlass::arch::warpgroup_reg_dealloc<NumLoadRegs>();
            mainloop.mma_loop(
                params,
                tma_params,
                shared_plan,
                tile_scheduler,
                // Load -> MMA pipelines (consumer)
                a_pipeline,
                a_pipe_state_read,
                // Prologue -> MMA pipeline (consumer)
                prologue_ready_pipeline,
                prologue_ready_pipe_state_read,
                // MMA -> Epilogue pipeline (producer)
                acc_done_pipeline,
                acc_done_pipe_state_write);

        } else if (role == WarpRole::Load) {
            cutlass::arch::warpgroup_reg_dealloc<NumLoadRegs>();
            mainloop.load_loop(
                params,
                tma_params,
                shared_plan,
                tile_scheduler,
                // TMA pipelines (producer)
                a_pipeline,
                a_pipe_state_write,
                k_pipeline,
                k_pipe_state_write,
                g_pipeline,
                g_pipe_state_write,
                v_pipeline,
                v_pipe_state_write,
                q_pipeline,
                q_pipe_state_write);

        } else if (role == WarpRole::LoadAux) {
            cutlass::arch::warpgroup_reg_dealloc<NumLoadRegs>();
            mainloop.load_aux_loop(
                params,
                tma_params,
                shared_plan,
                tile_scheduler,
                // Beta pipeline (producer)
                beta_pipeline,
                beta_pipe_state_write);
        }

        // === CLEANUP ===
        __syncthreads();
        if (warp_idx == 0 && cute::elect_one_sync()) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    }
};

// ===================================================================
// __global__ kernel wrapper (free function — CUDA requires this)
// ===================================================================
template <typename KernelT, typename TmaParamsT>
__global__ void
__launch_bounds__(384, 1, 1) kda_fwd_recomp_w_u_sm100_kernel_entry(
    __grid_constant__ const KDA_fwd_recomp_w_u_params params, __grid_constant__ const TmaParamsT tma_params) {
    KernelT kernel_obj;
    kernel_obj(params, tma_params);
}

// ===================================================================
// Host-side launcher: constructs TMA descriptors and launches kernel
// ===================================================================
template <typename Kernel>
inline void
run_kda_fwd_recomp_w_u_sm100_impl_dispatch(KDA_fwd_recomp_w_u_params& params, cudaStream_t stream) {
    auto shape_KVG = make_shape(params.total_len, params.d, params.h);
    auto stride_KVG = make_stride(params.h * params.d, _1{}, params.d);
    auto shape_Akk = make_shape(params.total_len, params.chunk_size, params.h);
    auto stride_Akk = make_stride(params.h * params.chunk_size, _1{}, params.chunk_size);

    // --- Build TMA descriptors ---
    auto tma_V = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.v_ptr), make_layout(shape_KVG, stride_KVG)),
        typename Kernel::SmemLayoutInputBF16{});

    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.k_ptr), make_layout(shape_KVG, stride_KVG)),
        typename Kernel::SmemLayoutInputBF16{});

    auto tma_G = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((float*)params.g_ptr), make_layout(shape_KVG, stride_KVG)),
        typename Kernel::SmemLayoutInputFP32{});

    auto tma_Akk = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr((bf16*)params.A_ptr), make_layout(shape_Akk, stride_Akk)),
        typename Kernel::SmemLayoutInputAkkBF16{});

    // Q TMA descriptor (only meaningful when StoreQG=true)
    auto tma_Q = [&]() {
        if constexpr (Kernel::StoreQG) {
            return cute::make_tma_copy(
                SM90_TMA_LOAD{},
                make_tensor(make_gmem_ptr((bf16*)params.q_ptr), make_layout(shape_KVG, stride_KVG)),
                typename Kernel::SmemLayoutInputBF16{});
        } else {
            return 0;  // placeholder, not used
        }
    }();

    // --- Pack TMA params ---
    typename Kernel::template TmaParams<
        decltype(shape_KVG),
        decltype(shape_Akk),
        decltype(tma_V),
        decltype(tma_K),
        decltype(tma_G),
        decltype(tma_Akk),
        decltype(tma_Q)>
        tma_params = {shape_KVG, shape_Akk, tma_V, tma_K, tma_G, tma_Akk, tma_Q};

    // --- Launch config ---
    auto kernel_fn = &kda_fwd_recomp_w_u_sm100_kernel_entry<Kernel, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_dim(Kernel::TileScheduler::get_grid_shape(params.tile_scheduler_params));
    dim3 block_dim(Kernel::NumTotalThreads, 1, 1);
    kernel_fn<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

inline void
run_kda_fwd_recomp_w_u_sm100_impl(KDA_fwd_recomp_w_u_params& params, cudaStream_t stream) {
    BETA_TYPE_SWITCH(params.is_beta_bf16, BetaType, [&] {
        BOOL_SWITCH(params.store_qg, kStoreQG, [&] {
            using Kernel = KdaChunkFwdRecompWUKernelSm100<KdaChunkFwdRecompWUMainloopSm100<kStoreQG, BetaType>>;
            run_kda_fwd_recomp_w_u_sm100_impl_dispatch<Kernel>(params, stream);
        });
    });
}

}  // namespace kda::sm100