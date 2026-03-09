#pragma once

// NOTE: This header is included from .cu files in csrc/ (parent directory).
// All includes use csrc/ as the root include path.
#include "kda_fwd_common.cuh"
#include "kda_bwd/helpers.h"
#include "kda_bwd/gemm.h"
#include "kda_bwd/utils.h"
#include "kda_bwd/fwd_util_func.h"

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

struct KdaChunkFwdIntraSm100NamedBarriers {
    static constexpr int ComputePrologue = 0;
};

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
    // static constexpr int NUM_TILE_CONSUMERS  = NUM_CE_THREADS + NUM_INVERSE_THREADS + NUM_MMA_THREADS + NUM_EMPTY_THREADS;

    // TODO: allocate TMEM
    enum class TmemAllocation : uint32_t {
        Q = 0
    };

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

    // Gated MMA B-matrix (tf32) — non-transposed, K-major layout
    // Forward computes Q/K @ K^T, Backward computes dAqk/dAkk @ K.
    // Forward MMA:  64 × X × 32 (M×N×K), reduces head dim (K=32), B = K^T
    //   B-matrix shape = (N × K) = (SUB_T_TILE × K_TILE), K-major
    //   Store pattern: sKG(x_local, y), where x_local = row within sub_tile, y = col group
    // Backward MMA: 64 × 32 × X (M×N×K), reduces chunk dim (K=SUB_T_TILE), B = K
    //   B-matrix shape = (K × N) = (SUB_T_TILE × K_TILE), MN-major
    //   Uses SmemLayoutMatBTF32Tranposed (Layout_MN_SW128_32B_Atom), stored as sKG(y, x_local)
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

    // ===================== Pipeline Types =====================
    using PipelineQ = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;
    using PipelineK = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;
    using PipelineG = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;

    using PipelineBeta = cutlass::PipelineAsync<2>;

    // TODO: update to PipelineUmmaConsumerAsync after finishing umma
    using PipelineQKGInterReady = cutlass::PipelineAsync<1>;
    using PipelineQKGIntraReady = cutlass::PipelineAsync<1>;

    using PipelineQKDone = cutlass::PipelineAsync<1>;

    using PipelineKKInvReady = cutlass::PipelineAsync<1>;

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
        alignas(16) typename PipelineQ::SharedStorage pipe_q_storage;
        alignas(16) typename PipelineK::SharedStorage pipe_k_storage;
        alignas(16) typename PipelineG::SharedStorage pipe_g_storage;

        alignas(16) typename PipelineBeta::SharedStorage pipe_beta_storage;

        alignas(16) typename PipelineQKGInterReady::SharedStorage pipe_qkg_inter_storage;
        alignas(16) typename PipelineQKGIntraReady::SharedStorage pipe_qkg_intra_storage;

        alignas(16) typename PipelineQKDone::SharedStorage pipe_qk_done_storage;

        alignas(16) typename PipelineKKInvReady::SharedStorage pipe_kk_inv_storage;

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

    // ===================== Pipeline State Types =====================
    using PipelineStateQ        = cutlass::PipelineState<PipelineQ::Stages>;
    using PipelineStateK        = cutlass::PipelineState<PipelineK::Stages>;
    using PipelineStateG        = cutlass::PipelineState<PipelineG::Stages>;
    using PipelineStateBeta     = cutlass::PipelineState<PipelineBeta::Stages>;
    using PipelineStateQKGInter = cutlass::PipelineState<PipelineQKGInterReady::Stages>;
    using PipelineStateQKGIntra = cutlass::PipelineState<PipelineQKGIntraReady::Stages>;
    using PipelineStateQKDone   = cutlass::PipelineState<PipelineQKDone::Stages>;
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
        PipelineQKGInterReady   &qkg_inter_pipeline,   PipelineStateQKGInter   &qkg_inter_pipe_state_write,
        PipelineQKGIntraReady &qkg_intra_pipeline,  PipelineStateQKGIntra &qkg_intra_pipe_state_write,
        // MMA -> CE pipelines (consumer)
        PipelineQKDone &qk_done_pipeline, PipelineStateQKDone &qk_done_pipe_state_read,
        // Beta pipeline (consumer)
        PipelineBeta &beta_pipeline, PipelineStateBeta &beta_pipe_state_read,
        // CE -> Inverse pipeline (producer)
        PipelineKKInvReady &kk_inv_pipeline, PipelineStateKKInv &kk_inv_pipe_state_write,
        // Tile decode helpers
        int *chunk_indices_ptr, int *cu_len_ptr, int total_tiles)
    {
        // === PERSISTENT CE LOOP (static scheduling, no tile pipeline) ===
        //
        // CE warpgroups: WG0 = thread [0,128), WG1 = thread [128,256)
        // idx_in_warpgroup: 0..127 within each WG
        //
        // B-matrix (R2S) thread mapping:
        //   128 threads per WG cover 16 rows × 8 col-groups = one sub_tile per call
        //   x_local = idx_in_warpgroup / 8  (row 0..15 within sub_tile)
        //   y       = idx_in_warpgroup % 8 * 4  (col group 0..28 step 4)
        //
        // Lower-triangular 4×4 subchunk matrix (10 total, i=row, j=col):
        //          j=0         j=1         j=2         j=3
        //   i=0  intra[0]
        //   i=1  inter[0]   intra[1]
        //   i=2  inter[1]   inter[2]   intra[2]
        //   i=3  inter[3]   inter[4]   inter[5]   intra[3]
        //
        // B-matrix formula for block (i, j) with i >= j:
        //   if i > j (inter): B = exp2(g_first_i - g_j[x]) * K_j[x]  (g_first_i = g[i*16])
        //   if i == j (intra): B = exp2(g_half_i - g_i[x]) * K_i[x]  (g_half_i = g[i*16+8])
        //
        // Column-based processing with fused helpers (load K_j + G once per column):
        //   col0_4out: intra(0,0) + inter(1,0) + inter(2,0) + inter(3,0)  (4 outputs)
        //   col1_3out: intra(1,1) + inter(2,1) + inter(3,1)               (3 outputs)
        //   col2_2out: intra(2,2) + inter(3,2)                            (2 outputs)
        //   col3_1out: intra(3,3)                                         (1 output)
        //
        // Work distribution across 2 WGs (balanced at 5 outputs each):
        //   WG0: col0 (4 outputs) + col3 (1 output) = 5 outputs
        //         via fwd_setup_kg_col0_4out + fwd_setup_kg_col3_1out
        //   WG1: col1 (3 outputs) + col2 (2 outputs) = 5 outputs
        //         via fwd_setup_kg_col1_3out + fwd_setup_kg_col2_2out
        //
        // Benefits over old column-split approach:
        //   - Each column's K_j + G data loaded exactly ONCE (was 2× for col0-2)
        //   - No WG idle time (old design: WG1 idle on col3)
        //   - Perfect 5:5 output balance
        //
        // Result: kg_all.inter[0..5] and kg_all.intra[0..3] in SMEM (tf32).
        //
        const int idx_in_warpgroup = threadIdx.x % 128;
        const int wg_idx = threadIdx.x / 128;  // 0 or 1 within CE

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx  = get<1>(blk_coord);
            int tile_idx  = get<2>(blk_coord);
            int start_offset = cu_len_ptr[batch_idx];
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

            for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                // ============================================================
                // Step 1: Wait for K, G data from TMA Load warp
                // ============================================================
                g_pipeline.consumer_wait(g_pipe_state_read);
                k_pipeline.consumer_wait(k_pipe_state_read);
                q_pipeline.consumer_wait(q_pipe_state_read);

                // ============================================================
                // Step 2: Create SMEM tensor views for this buffer slot
                // ============================================================
                Tensor sK = make_tensor(make_smem_ptr(shared_plan->k[k_pipe_state_read.index()].data()), SmemLayoutInputBF16{});
                Tensor sG = make_tensor(make_smem_ptr(shared_plan->g[g_pipe_state_read.index()].data()), SmemLayoutInputFP32{});

                // B-matrix SMEM views (single-buffered kg_all)
                // Each sub_tile occupies one SmemLayoutMatBTF32<1> = (16 × 32)
                // inter[i] and intra[i] are indexed by KG_OFFSET * index inside the helper
                Tensor sKG_inter = make_tensor(make_smem_ptr(shared_plan->kg_all.inter[0].data()), SmemLayoutMatBTF32<1>{});
                Tensor sKG_intra = make_tensor(make_smem_ptr(shared_plan->kg_all.intra[0].data()), SmemLayoutMatBTF32<1>{});

                constexpr int kg_offset = SUB_T_TILE * K_TILE;  // stride between sub_tile buffers

                // All 6 inter + 4 intra B-matrices are ready → signal both pipelines
                qkg_inter_pipeline.producer_acquire(qkg_inter_pipe_state_write);
                qkg_intra_pipeline.producer_acquire(qkg_intra_pipe_state_write);
                // ============================================================
                // Step 3: Compute all 10 B-matrix subchunks (R2S)
                // ============================================================
                // Lower-triangular 4×4 pattern, column-by-column processing.
                // Each fused helper loads K_j + G data ONCE and produces ALL outputs for column j.
                //
                // Buffer mapping:
                //   inter[0]=(1,0), inter[1]=(2,0), inter[2]=(2,1),
                //   inter[3]=(3,0), inter[4]=(3,1), inter[5]=(3,2)
                //   intra[0]=(0,0), intra[1]=(1,1), intra[2]=(2,2), intra[3]=(3,3)
                //
                // Work distribution (balanced, 5 outputs each):
                //   WG0: col0 (4 outputs) + col3 (1 output) = 5 outputs
                //   WG1: col1 (3 outputs) + col2 (2 outputs) = 5 outputs
                {
                    if (wg_idx == 0) {
                        // ---- WG0: Column j=0 (4 outputs) ----
                        // intra(0,0) + inter(1,0) + inter(2,0) + inter(3,0)
                        float4 g_half_0, g_first_1, g_first_2, g_first_3;
                        fwd_setup_kg_col0_4out<decltype(sG), decltype(sK), decltype(sKG_inter), kg_offset>(
                            sG, sK, sKG_inter, sKG_intra,
                            idx_in_warpgroup, sub_seq_len,
                            g_half_0, g_first_1, g_first_2, g_first_3);

                        // ---- WG0: Column j=3 (1 output) ----
                        // intra(3,3)
                        float4 g_half_3;
                        fwd_setup_kg_col3_1out<decltype(sG), decltype(sK), decltype(sKG_intra), kg_offset>(
                            sG, sK, sKG_intra,
                            idx_in_warpgroup, sub_seq_len,
                            g_half_3);
                    } else {
                        // ---- WG1: Column j=1 (3 outputs) ----
                        // intra(1,1) + inter(2,1) + inter(3,1)
                        float4 g_half_1, g_first_2, g_first_3;
                        fwd_setup_kg_col1_3out<decltype(sG), decltype(sK), decltype(sKG_inter), kg_offset>(
                            sG, sK, sKG_inter, sKG_intra,
                            idx_in_warpgroup, sub_seq_len,
                            g_half_1, g_first_2, g_first_3);

                        // ---- WG1: Column j=2 (2 outputs) ----
                        // intra(2,2) + inter(3,2)
                        float4 g_half_2, g_first_3_2;
                        fwd_setup_kg_col2_2out<decltype(sG), decltype(sK), decltype(sKG_inter), kg_offset>(
                            sG, sK, sKG_inter, sKG_intra,
                            idx_in_warpgroup, sub_seq_len,
                            g_half_2, g_first_3_2);
                    }
                }

                // =====DEBUG=====
                // wait for smem write finished
                // cutlass::arch::NamedBarrier::arrive_and_wait(128 * 2, KdaChunkFwdIntraSm100NamedBarriers::ComputePrologue);
                // if (threadIdx.x == 0) {
                //     printf("Iter=%d\n", k_idx);
                //     printf("sKG_inter (0, 0)");
                //     cute::print_tensor(sKG_inter);
                //     printf("sKG_intra (0, 0)");
                //     cute::print_tensor(sKG_intra);
                // }

                // TODO: debug

                // ============================================================
                // Step 4: Fence SMEM writes and signal MMA
                // ============================================================
                fence_view_async_shared();
                qkg_inter_pipeline.producer_commit(qkg_inter_pipe_state_write);
                ++qkg_inter_pipe_state_write;

                fence_view_async_shared();
                qkg_intra_pipeline.producer_commit(qkg_intra_pipe_state_write);
                ++qkg_intra_pipe_state_write;

                // ============================================================
                // Step 5: Release Q, K, G smem buffers back to TMA Load warp
                // ============================================================
                g_pipeline.consumer_release(g_pipe_state_read);
                ++g_pipe_state_read;
                k_pipeline.consumer_release(k_pipe_state_read);
                ++k_pipe_state_read;
                q_pipeline.consumer_release(q_pipe_state_read);
                ++q_pipe_state_read;
            }

            // ============================================================
            // Post-loop: wait for MMA results, epilogue, signal downstream
            // ============================================================
            // TODO: A-matrix (R2T) prologue — will be added after B-matrix validation
            // TODO: wait for MMA ready, fence tmem load
            qk_done_pipeline.consumer_wait(qk_done_pipe_state_read);

            // TODO: T2R kk

            // TODO: kk epilogue and notify KK inverse
            beta_pipeline.consumer_wait(beta_pipe_state_read);
            fence_view_async_shared();

            kk_inv_pipeline.producer_acquire(kk_inv_pipe_state_write);

            fence_view_async_shared();
            kk_inv_pipeline.producer_commit(kk_inv_pipe_state_write);
            ++kk_inv_pipe_state_write;

            beta_pipeline.consumer_release(beta_pipe_state_read);
            ++beta_pipe_state_read;

            // TODO: T2R qk and notify tmem read finished
            qk_done_pipeline.consumer_release(qk_done_pipe_state_read);
            ++qk_done_pipe_state_read;

            // TODO: qk epilogue and R2G qk

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
        PipelineQKGInterReady &qkg_inter_pipeline,  PipelineStateQKGInter &qkg_inter_pipe_state_read,
        PipelineQKGIntraReady &qkg_intra_pipeline,  PipelineStateQKGIntra &qkg_intra_pipe_state_read,
        // MMA -> CE pipelines (producer)
        PipelineQKDone &qk_done_pipeline, PipelineStateQKDone &qk_done_pipe_state_write,
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
                for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                    qkg_inter_pipeline.consumer_wait(qkg_inter_pipe_state_read);
                    
                    // TODO: qkg inter MMA

                    qkg_inter_pipeline.consumer_release(qkg_inter_pipe_state_read);
                    ++qkg_inter_pipe_state_read;

                    // TODO: tcgen05 thread sync

                    qkg_intra_pipeline.consumer_wait(qkg_intra_pipe_state_read);

                    // TODO: qkg intra MMA

                    qkg_intra_pipeline.consumer_release(qkg_intra_pipe_state_read);
                    ++qkg_intra_pipe_state_read;
                }
                // notify MMA finished to CE
                qk_done_pipeline.producer_acquire(qk_done_pipe_state_write);
                qk_done_pipeline.producer_commit(qk_done_pipe_state_write);
                ++qk_done_pipe_state_write;
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
                int token_offset = cu_len_ptr[batch_idx];
                int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
                int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

                // TMA load body (Q, K, G)
                for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                    Tensor sQ = make_tensor(make_smem_ptr(
                        shared_plan->q[q_pipe_state_write.index()].data()
                    ), SmemLayoutInputBF16{});
                    Tensor sK = make_tensor(make_smem_ptr(
                        shared_plan->k[k_pipe_state_write.index()].data()
                    ), SmemLayoutInputBF16{});
                    Tensor sG = make_tensor(make_smem_ptr(
                        shared_plan->g[g_pipe_state_write.index()].data()
                    ), SmemLayoutInputFP32{});

                    Tensor mQ = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_q.get_tma_tensor(tma_params.shape_qkg));
                    Tensor mK = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_k.get_tma_tensor(tma_params.shape_qkg));
                    Tensor mG = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_g.get_tma_tensor(tma_params.shape_qkg));

                    Tensor gK = local_tile(mK(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    Tensor gG = local_tile(mG(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    Tensor gQ = local_tile(mQ(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    
                    g_pipeline.producer_acquire(g_pipe_state_write);
                    launch_tma_copy(tma_params.tma_g, gG, sG, *g_pipeline.producer_get_barrier(g_pipe_state_write));
                    ++g_pipe_state_write;
                    k_pipeline.producer_acquire(k_pipe_state_write);
                    launch_tma_copy(tma_params.tma_k, gK, sK, *k_pipeline.producer_get_barrier(k_pipe_state_write));
                    ++k_pipe_state_write;
                    q_pipeline.producer_acquire(q_pipe_state_write);
                    launch_tma_copy(tma_params.tma_q, gQ, sQ, *q_pipeline.producer_get_barrier(q_pipe_state_write));
                    ++q_pipe_state_write;

                }
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
            kk_inv_pipeline.consumer_wait(kk_inv_pipe_state_read);
            fence_view_async_shared();

            kk_inv_pipeline.consumer_release(kk_inv_pipe_state_read);
            ++kk_inv_pipe_state_read;
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

            // Beta loading body
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