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

namespace sm100 {

using cutlass::arch::fence_view_async_shared;
using namespace cute;

// ===================================================================
// Mainloop struct: KdaChunkFwdIntraMainloopSm100
// Self-contained: owns all pipeline types, SMEM layouts, SharedMemoryPlan,
// constants, and the persistent loop bodies for each warp role.
// The Kernel struct is templated on this Mainloop.
// ===================================================================
struct KdaChunkFwdIntraMainloopSm100 {

    // ===================== Tile / Buffer Constants =====================
    static constexpr int SUB_T_TILE   = 16;
    static constexpr int T_TILE       = 64;
    static constexpr int K_SIZE       = 128;
    static constexpr int K_TILE       = 32;
    static constexpr int K_ITERATION  = K_SIZE / K_TILE;
    static constexpr int NUM_BUF_A    = 1;
    static constexpr int NUM_BUF_VALUE = 2;
    static constexpr int CHUNK_SIZE   = 64;
    static constexpr int NUM_THREADS  = 128 * 4;  // 512

    // ===================== Thread Count Constants =====================
    static constexpr int NUM_CE_THREADS      = 256; // warp 0-7 (2 warpgroups)
    static constexpr int NUM_INVERSE_THREADS = 128; // warp 8-11 (1 warpgroup)
    static constexpr int NUM_MMA_THREADS     = 1;   // elect_one in warp 12
    static constexpr int NUM_LOAD_THREADS    = 1;   // elect_one in warp 13
    static constexpr int NUM_EMPTY_THREADS   = 64;  // warp 14-15
    static constexpr int NUM_TILE_CONSUMERS  = NUM_CE_THREADS + NUM_INVERSE_THREADS + NUM_MMA_THREADS + NUM_EMPTY_THREADS;

    using ClusterShape = Shape<_1, _1, _1>;
    using TileScheduler = StaticPersistentTileScheduler;

    // ===================== SMEM Layouts =====================
    // Q, K (bf16)
    using SmemLayoutInputBF16 = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW64_Atom<bf16>{},
        Shape<Int<T_TILE>, Int<K_TILE>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    // G (fp32)
    using SmemLayoutInputFP32 = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<float>{},
        Shape<Int<T_TILE>, Int<K_TILE>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    // Gated MMA K^T (tf32)
    template<int NUM_TILES>
    using SmemLayoutMatBTF32 = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<tf32>{},
        Shape<Int<SUB_T_TILE * NUM_TILES>, Int<K_TILE>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    // inv(KK) output (bf16)
    using SmemLayoutOutputBF16 = decltype(tile_to_shape(
        UMMA::Layout_K_INTER_Atom<bf16>{},
        Shape<Int<T_TILE>, Int<T_TILE>>{}
    ));

    // inv(KK) (fp16)
    using SmemLayoutOutputFP16 = decltype(tile_to_shape(
        UMMA::Layout_K_INTER_Atom<fp16>{},
        Shape<Int<T_TILE>, Int<T_TILE>>{}
    ));

    // inv(KK) (tf32)
    using SmemLayoutOutputTF32 = decltype(tile_to_shape(
        UMMA::Layout_K_INTER_Atom<tf32>{},
        Shape<Int<T_TILE>, Int<T_TILE>>{}
    ));

    // ===================== Shared Memory Plan =====================
    struct SharedMemoryPlan {
        // Q, K, G double buffer
        array_aligned<bf16,  cosize_v<SmemLayoutInputBF16>> q[NUM_BUF_VALUE];
        array_aligned<bf16,  cosize_v<SmemLayoutInputBF16>> k[NUM_BUF_VALUE];
        array_aligned<float, cosize_v<SmemLayoutInputFP32>> g[NUM_BUF_VALUE];

        // Gated MMA K^T, single buffer
        struct {
            array_aligned<tf32, cosize_v<SmemLayoutMatBTF32<1>>> inter[6];
            array_aligned<tf32, cosize_v<SmemLayoutMatBTF32<1>>> intra[4];
        } kg_all;

        // inv(KK), single buffer
        array_aligned<fp16, cosize_v<SmemLayoutOutputFP16>> kk[NUM_BUF_A];

        // ---- Pipeline shared storage ----
        alignas(16) typename cutlass::PipelineTmaAsync<NUM_BUF_VALUE>::SharedStorage pipe_q_storage;
        alignas(16) typename cutlass::PipelineTmaAsync<NUM_BUF_VALUE>::SharedStorage pipe_k_storage;
        alignas(16) typename cutlass::PipelineTmaAsync<NUM_BUF_VALUE>::SharedStorage pipe_g_storage;

        alignas(16) typename cutlass::PipelineAsync<2>::SharedStorage          pipe_beta_storage;

        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_qkg_all_storage;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_ktg_inter_storage;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_ktg_intra_storage;

        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_qk_done_storage;
        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_kk_done_storage;

        alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_kk_inv_storage;

        alignas(16) float beta_smem[2][T_TILE];
        array_aligned<uint32_t, 1> tmem_start_addr;
    };

    // ===================== TMA Params =====================
    template<
        typename ShapeQKG,
        typename TMA_Q,
        typename TMA_K,
        typename TMA_G>
    struct TmaParams {
        ShapeQKG shape_qkg;
        TMA_Q tma_q;
        TMA_K tma_k;
        TMA_G tma_g;
    };

    // ===================== Pipeline Types =====================
    using PipelineQ = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;
    using PipelineK = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;
    using PipelineG = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;

    using PipelineBeta = cutlass::PipelineAsync<2>;

    using PipelineQKGAllReady   = cutlass::PipelineAsync<1>;
    using PipelineKTGInterReady = cutlass::PipelineAsync<1>;
    using PipelineKTGIntraReady = cutlass::PipelineAsync<1>;

    using PipelineQKDone = cutlass::PipelineUmmaAsync<1>;
    using PipelineKKDone = cutlass::PipelineUmmaAsync<1>;

    using PipelineKKInvReady = cutlass::PipelineAsync<1>;

    // ===================== Pipeline State Types =====================
    using PipelineStateQ        = cutlass::PipelineState<PipelineQ::Stages>;
    using PipelineStateK        = cutlass::PipelineState<PipelineK::Stages>;
    using PipelineStateG        = cutlass::PipelineState<PipelineG::Stages>;
    using PipelineStateBeta     = cutlass::PipelineState<PipelineBeta::Stages>;
    using PipelineStateQKGAll   = cutlass::PipelineState<PipelineQKGAllReady::Stages>;
    using PipelineStateKTGInter = cutlass::PipelineState<PipelineKTGInterReady::Stages>;
    using PipelineStateKTGIntra = cutlass::PipelineState<PipelineKTGIntraReady::Stages>;
    using PipelineStateQKDone   = cutlass::PipelineState<PipelineQKDone::Stages>;
    using PipelineStateKKDone   = cutlass::PipelineState<PipelineKKDone::Stages>;
    using PipelineStateKKInv    = cutlass::PipelineState<PipelineKKInvReady::Stages>;

    // ===================================================================
    // ComputeEpilogue warp persistent loop (warp 0-7, 2 warpgroups)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void compute_epilogue_loop(
        const KDA_fwd_intra_params &params,
        const TmaParamsT &tma_params,
        SharedMemoryPlan *shared_plan,
        TileScheduler &tile_scheduler,
        // TMA pipelines (consumer)
        PipelineQ &q_pipeline, PipelineStateQ &q_pipe_state_read,
        PipelineK &k_pipeline, PipelineStateK &k_pipe_state_read,
        PipelineG &g_pipeline, PipelineStateG &g_pipe_state_read,
        // CE -> MMA pipelines (producer)
        PipelineQKGAllReady   &qkg_all_pipeline,   PipelineStateQKGAll   &qkg_all_pipe_state_write,
        PipelineKTGInterReady &ktg_inter_pipeline,  PipelineStateKTGInter &ktg_inter_pipe_state_write,
        PipelineKTGIntraReady &ktg_intra_pipeline,  PipelineStateKTGIntra &ktg_intra_pipe_state_write,
        // MMA -> CE pipelines (consumer)
        PipelineQKDone &qk_done_pipeline, PipelineStateQKDone &qk_done_pipe_state_read,
        PipelineKKDone &kk_done_pipeline, PipelineStateKKDone &kk_done_pipe_state_read,
        // Beta pipeline (consumer)
        PipelineBeta &beta_pipeline, PipelineStateBeta &beta_pipe_state_read,
        // CE -> Inverse pipeline (producer)
        PipelineKKInvReady &kk_inv_pipeline, PipelineStateKKInv &kk_inv_pipe_state_write,
        // Tile decode helpers
        int *chunk_indices_ptr, int *cu_len_ptr, int total_tiles)
    {
        // === PERSISTENT CE LOOP (static scheduling, no tile pipeline) ===
        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx  = get<1>(blk_coord);
            int tile_idx  = get<2>(blk_coord);

            // TODO: CE computation body
            beta_pipeline.consumer_wait(beta_pipe_state_read);

            beta_pipeline.consumer_release(beta_pipe_state_read);
            ++beta_pipe_state_read;
        }
    }

    // ===================================================================
    // MMA warp persistent loop (warp 12, elect_one)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void mma_loop(
        const KDA_fwd_intra_params &params,
        const TmaParamsT &tma_params,
        SharedMemoryPlan *shared_plan,
        TileScheduler &tile_scheduler,
        // CE -> MMA pipelines (consumer)
        PipelineQKGAllReady   &qkg_all_pipeline,   PipelineStateQKGAll   &qkg_all_pipe_state_read,
        PipelineKTGInterReady &ktg_inter_pipeline,  PipelineStateKTGInter &ktg_inter_pipe_state_read,
        PipelineKTGIntraReady &ktg_intra_pipeline,  PipelineStateKTGIntra &ktg_intra_pipe_state_read,
        // MMA -> CE pipelines (producer)
        PipelineQKDone &qk_done_pipeline, PipelineStateQKDone &qk_done_pipe_state_write,
        PipelineKKDone &kk_done_pipeline, PipelineStateKKDone &kk_done_pipe_state_write,
        // Tile decode helpers
        int *chunk_indices_ptr, int *cu_len_ptr, int total_tiles)
    {
        if (cute::elect_one_sync()) {
            // === PERSISTENT MMA LOOP (static scheduling, no tile pipeline) ===
            CUTE_NO_UNROLL
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int tid = tile_scheduler.get_current_tile_id();

                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx  = get<1>(blk_coord);
                int tile_idx  = get<2>(blk_coord);

                // TODO: MMA computation body
            }
        }
    }

    // ===================================================================
    // Load warp persistent loop (warp 13, elect_one, TMA producer)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void load_loop(
        const KDA_fwd_intra_params &params,
        const TmaParamsT &tma_params,
        SharedMemoryPlan *shared_plan,
        TileScheduler &tile_scheduler,
        // TMA pipelines (producer)
        PipelineQ &q_pipeline, PipelineStateQ &q_pipe_state_write,
        PipelineK &k_pipeline, PipelineStateK &k_pipe_state_write,
        PipelineG &g_pipeline, PipelineStateG &g_pipe_state_write,
        // Tile decode helpers
        int *chunk_indices_ptr, int *cu_len_ptr, int total_tiles)
    {
        if (cute::elect_one_sync()) {
            // === PERSISTENT LOAD LOOP (static scheduling, no tile pipeline) ===
            CUTE_NO_UNROLL
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int tid = tile_scheduler.get_current_tile_id();

                // Decode tile coordinates
                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx    = get<0>(blk_coord);
                int head_idx     = get<1>(blk_coord);
                int tile_idx     = get<2>(blk_coord);

                // TODO: TMA load body (Q, K, G)
            }
        }
    }

    // ===================================================================
    // Inverse warpgroup persistent loop (warp 8-11, 1 warpgroup)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void inverse_loop(
        const KDA_fwd_intra_params &params,
        const TmaParamsT &tma_params,
        SharedMemoryPlan *shared_plan,
        TileScheduler &tile_scheduler,
        // CE -> Inverse pipeline (consumer)
        PipelineKKInvReady &kk_inv_pipeline,
        PipelineStateKKInv &kk_inv_pipe_state_read,
        // Tile decode helpers
        int *chunk_indices_ptr, int *cu_len_ptr, int total_tiles)
    {
        // === PERSISTENT INVERSE LOOP (static scheduling, no tile pipeline) ===
        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx  = get<1>(blk_coord);
            int tile_idx  = get<2>(blk_coord);

            // TODO: Inverse computation body
        }
    }

    // ===================================================================
    // Empty warp persistent loop (warp 14-15, beta loading)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void empty_loop(
        const KDA_fwd_intra_params &params,
        const TmaParamsT &tma_params,
        SharedMemoryPlan *shared_plan,
        TileScheduler &tile_scheduler,
        // Beta pipeline (producer)
        PipelineBeta &beta_pipeline,
        PipelineStateBeta &beta_pipe_state_write,
        // Tile decode helpers
        int *chunk_indices_ptr, int *cu_len_ptr, int total_tiles)
    {
        // === PERSISTENT EMPTY WARP LOOP (static scheduling, no tile pipeline) ===
        int empty_idx = threadIdx.x - (NUM_THREADS - 64); // 0..63
        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx    = get<0>(blk_coord);
            int head_idx     = get<1>(blk_coord);
            int tile_idx     = get<2>(blk_coord);
            int token_offset = cu_len_ptr[batch_idx];
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

            // TODO: Beta loading body
            beta_pipeline.producer_acquire(beta_pipe_state_write);
            if (empty_idx < T_TILE) {
                shared_plan->beta_smem[beta_pipe_state_write.index()][empty_idx] = (empty_idx < sub_seq_len)
                    ? reinterpret_cast<float*>(params.beta_ptr)[(token_offset + tile_idx * T_TILE + empty_idx) * params.h + head_idx]
                    : float(0);
            }
            fence_view_async_shared();
            beta_pipeline.producer_commit(beta_pipe_state_write);
            ++beta_pipe_state_write;
        }
    }
};

} // namespace sm100