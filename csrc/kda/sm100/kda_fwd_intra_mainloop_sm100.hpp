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

#include "kda/sm100/fwd_helpers.hpp"
#include "kda/sm100/kda_fwd_common.cuh"
#include "kda/sm100/sm100_umma_ext.hpp"

namespace kda::sm100 {

using cutlass::arch::fence_view_async_shared;
using namespace cute;

struct KdaChunkFwdIntraSm100NamedBarriers {
    static constexpr int ComputeCudaCore = 0;
    static constexpr int InverseMath = 1;
};

// ===================================================================
// Mainloop struct: KdaChunkFwdIntraMainloopSm100
// Self-contained: owns all pipeline types, SMEM layouts, SharedMemoryPlan,
// constants, and the persistent loop bodies for each warp role.
// The Kernel struct is templated on this Mainloop.
// ===================================================================
template <bool UseTF32Inverse_ = true, bool RoundingTF32_ = false, bool UnifiedGRef_ = false, typename ElementBeta_ = float>
struct KdaChunkFwdIntraMainloopSm100 {
    // ===================== Tile / Buffer Constants =====================
    static constexpr int SubTileT = 16;
    static constexpr int TileT = 64;
    static constexpr int HeadDim = 128;
    static constexpr int TileK = 32;
    static constexpr int NumKIters = HeadDim / TileK;
    static constexpr int ChunkSize = 64;
    static constexpr int StagesLoad = 3;
    static constexpr int StagesMma = 3;
    static constexpr int StagesAcc = 3;

    // matrix inversion config
    // TODO: optimize perf, larger band conflict for TF32 inverse
    // NOTE: using TF32 inverse gets better accuracy, causes about 7% kernel time increase currently
    // default to true for better precision
    static constexpr bool UseTF32Inverse = UseTF32Inverse_;
    // NOTE: when enabling RoundingTF32=true, do x+=0x1000u for rounding,
    // theoretically better precision, but lower performance
    // otherwise, better performance but theoretically lower precision
    // default to false, because FLA impl uses tl.dot directly which does not use rounding
    // ref: https://triton-lang.org/main/python-api/generated/triton.language.dot.html
    static constexpr bool RoundingTF32 = RoundingTF32_;
    // When true, intra B-matrix uses g_first (=g[sub_tile_i*16]) instead of g_half (=g[sub_tile_i*16+8]).
    // This makes inter and intra A-matrices identical, allowing the intra A-matrix to be skipped entirely.
    // Saves 50% of A-matrix exp2f computation and one TMEM store per k-iteration.
    static constexpr bool UnifiedGRef = UnifiedGRef_;

    // double buffer in TMEM, overlap prologue A matrix with MMA
    enum class TmemAllocation : uint32_t {
        QK = 0,                                  // [0, 64]
        QK_02 = QK,                              // [0, 64]
        QK_13 = QK_02 + 16 * 65536,              // [0, 64] +lane16
        QG_INTER = QK + TileT,                   // [64, 96]
        QG_INTER_02 = QG_INTER,                  // [64, 96]
        QG_INTER_13 = QG_INTER_02 + 16 * 65536,  // [64, 96] +lane16
        QG_INTRA = QG_INTER + TileK,             // [96, 128]
        QG_INTRA_02 = QG_INTRA,                  // [96, 128]
        QG_INTRA_13 = QG_INTRA_02 + 16 * 65536,  // [96, 128] +lane16
    };

    using TileScheduler = StaticPersistentTileScheduler;

    // ===================== SMEM Layouts =====================
    // Q, K (bf16)
    using SmemLayoutInputBF16 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW64_Atom<ku::bf16>{}, Shape<Int<TileT>, Int<TileK>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    // G (fp32)
    using SmemLayoutInputFP32 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW128_Atom<float>{}, Shape<Int<TileT>, Int<TileK>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    template <int NumTiles>
    using SmemLayoutMatBTF32 = decltype(coalesce(
        tile_to_shape(
            UMMA::Layout_K_SW128_Atom<ku::tf32>{}, Shape<Int<SubTileT * NumTiles>, Int<TileK>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    // QK/inv(KK) output (bf16)
    using SmemLayoutOutputBF16 =
        decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<ku::bf16>{}, Shape<Int<TileT>, Int<TileT>>{}));

    // inv(KK) (fp16)
    using SmemLayoutOutputFP16 =
        decltype(tile_to_shape(UMMA::Layout_K_INTER_Atom<ku::fp16>{}, Shape<Int<TileT>, Int<TileT>>{}));

    // inv(KK) (tf32) — padded row-major layout to avoid SMEM bank conflicts.
    // With SW128 swizzle (SmemLayoutOutputTF32), row stride = 64*4 = 256 bytes,
    // and (row*64) % 32 == 0 for all rows, causing 32-way bank conflicts when
    // threads write/read the same column across different rows.
    // Padded stride 68: bank(r,c) = (r*68+c) % 32 = (r*4+c) % 32, cycling
    // through 8 banks per 8 rows → max 4-way conflict (8x better).
    // Each row is 68*4 = 272 bytes, which is 16-byte aligned (128-bit).
    // Extra SMEM per buffer: 4*64*4 = 1024 bytes (negligible).
    using SmemLayoutInvKK_TF32_Padded = Layout<Shape<Int<TileT>, Int<TileT>>, Stride<Int<TileT + 4>, _1>>;

    using SmemLayoutInvKK = std::conditional_t<!UseTF32Inverse, SmemLayoutOutputFP16, SmemLayoutInvKK_TF32_Padded>;

    using TiledMMA_KDAqk_N16_MASK02 = decltype(make_tiled_mma(
        SM100_MMA_TF32_TS_MASK02<ku::tf32, ku::tf32, float, TileT, SubTileT, UMMA::Major::K, UMMA::Major::K>{}));

    using TiledMMA_KDAqk_N16_MASK13 = decltype(make_tiled_mma(
        SM100_MMA_TF32_TS_MASK13<ku::tf32, ku::tf32, float, TileT, SubTileT, UMMA::Major::K, UMMA::Major::K>{}));

    using TiledMMA_KDAqk_N32_MASK02 = decltype(make_tiled_mma(
        SM100_MMA_TF32_TS_MASK02<ku::tf32, ku::tf32, float, TileT, SubTileT * 2, UMMA::Major::K, UMMA::Major::K>{}));

    using TiledMMA_KDAqk_N32_MASK13 = decltype(make_tiled_mma(
        SM100_MMA_TF32_TS_MASK13<ku::tf32, ku::tf32, float, TileT, SubTileT * 2, UMMA::Major::K, UMMA::Major::K>{}));

    using TiledMMA_KDAqk_N48_MASK02 = decltype(make_tiled_mma(
        SM100_MMA_TF32_TS_MASK02<ku::tf32, ku::tf32, float, TileT, SubTileT * 3, UMMA::Major::K, UMMA::Major::K>{}));

    using TiledMMA_KDAqk_N48_MASK13 = decltype(make_tiled_mma(
        SM100_MMA_TF32_TS_MASK13<ku::tf32, ku::tf32, float, TileT, SubTileT * 3, UMMA::Major::K, UMMA::Major::K>{}));

    // ===================== Pipeline Types =====================
    using PipelineQKG = cutlass::PipelineTmaAsync<StagesLoad>;

    using PipelineBeta = cutlass::PipelineAsync<StagesAcc>;

    using PipelineQKGInterReady = cutlass::PipelineUmmaConsumerAsync<StagesMma>;

    using PipelineQKDone = cutlass::PipelineAsync<StagesAcc>;

    using PipelineKKInvReady = cutlass::PipelineAsync<StagesAcc>;

    // ===================== Matrix Inverse =====================
    using InverseType = std::conditional_t<!UseTF32Inverse, cutlass::half_t, cutlass::tfloat32_t>;
    // NOTE: avoid tfloat32 cast in R2G store
    using InverseOutputType = std::conditional_t<!UseTF32Inverse, cutlass::half_t, float>;
    using CollectiveInverse = std::conditional_t<
        !UseTF32Inverse,
        ku::CollectiveInverse<InverseType, true, false>,
        ku::CollectiveInverseTF32<InverseType, true, false, RoundingTF32>>;

    // ===================== GMEM Store ===========
    // Akk: R2G store bf16
    using TileShapeKK = decltype(make_shape(_64{}, _64{}, _128{}));
    using Element = cutlass::bfloat16_t;
    // Adapted from
    // https://github.com/Dao-AILab/flash-attention/blob/9b6dbaceb658f576ea81e2b0189f4b5707a39aae/hopper/epilogue_fwd.hpp#L51
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);  // 16/2=8
    static_assert(TileT % kGmemElemsPerStore == 0, "Chunk size must be a multiple of kGmemElemsPerStore for Aqk/Akk");
    static constexpr int kBytePerRow = TileT * sizeof(Element);  // 64x2=128
    static constexpr int kBlockKGmem =
        (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);  // 128/2=64
    // Number of threads required to collaboratively read/write one (128-byte, 64-byte, or 32-byte) block
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;  // 8
    static constexpr int NumEpilogueThreads = cutlass::NumThreadsPerWarpGroup;
    static_assert(
        NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    // Layout of Epilogue threads, named GmemLayoutAtom
    using GmemLayoutAtom = Layout<
        Shape<Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTileCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        GmemTileCopyAtomO{},
        GmemLayoutAtom{},
        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));  // Val layout, 8 or 16 vals per store

    // ===================== Shared Memory Plan =====================
    struct SharedMemoryPlan {
        // Q, K, G double buffer
        array_aligned<ku::bf16, cosize_v<SmemLayoutInputBF16>> q[StagesLoad];  // 12KB
        array_aligned<ku::bf16, cosize_v<SmemLayoutInputBF16>> k[StagesLoad];  // 12KB
        array_aligned<float, cosize_v<SmemLayoutInputFP32>> g[StagesLoad];     // 24KB

        // Gated MMA K^T, double buffer
        struct {
            array_aligned<ku::tf32, cosize_v<SmemLayoutMatBTF32<1>>> inter[6];
            array_aligned<ku::tf32, cosize_v<SmemLayoutMatBTF32<1>>> intra[4];
        } kg_all[StagesMma];  // 60KB

        // inv(KK), double buffer
        array_aligned<InverseType, cosize_v<SmemLayoutInvKK>> kk[StagesAcc];  // 48KB

        // ---- Pipeline shared storage ----
        alignas(16) typename PipelineQKG::SharedStorage pipe_qkg_load_storage;

        alignas(16) typename PipelineBeta::SharedStorage pipe_beta_storage;

        alignas(16) typename PipelineQKGInterReady::SharedStorage pipe_qkg_inter_storage;

        alignas(16) typename PipelineQKDone::SharedStorage pipe_qk_done_storage;

        alignas(16) typename PipelineKKInvReady::SharedStorage pipe_kk_inv_storage;

        // TODO: support bf16 beta
        alignas(16) float beta_smem[StagesAcc][TileT];
        array_aligned<uint32_t, 1> tmem_start_addr;
    };

    // ===================== TMA Params =====================
    template <typename ShapeQKG, typename TMA_Q, typename TMA_K, typename TMA_G>
    struct TmaParams {
        ShapeQKG shape_qkg;
        TMA_Q tma_q;
        TMA_K tma_k;
        TMA_G tma_g;
    };

    // ===================== Pipeline State Types =====================
    using PipelineStateQKG = cutlass::PipelineState<PipelineQKG::Stages>;
    using PipelineStateBeta = cutlass::PipelineState<PipelineBeta::Stages>;
    using PipelineStateQKGInter = cutlass::PipelineState<PipelineQKGInterReady::Stages>;
    using PipelineStateQKDone = cutlass::PipelineState<PipelineQKDone::Stages>;
    using PipelineStateKKInv = cutlass::PipelineState<PipelineKKInvReady::Stages>;

    // ===================================================================
    // ComputeCudaCore warp persistent loop (warp 0-7, 2 warpgroups)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    compute_cudacore_loop(
        const KDA_fwd_intra_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // Unified TMA pipeline for Q+K+G (consumer)
        PipelineQKG& qkg_load_pipeline,
        PipelineStateQKG& qkg_load_pipe_state_read,
        // CudaCore -> MMA pipelines (producer)
        PipelineQKGInterReady& qkg_inter_pipeline,
        PipelineStateQKGInter& qkg_inter_pipe_state_write,
        // MMA -> CudaCore pipelines (consumer)
        PipelineQKDone& qk_done_pipeline,
        PipelineStateQKDone& qk_done_pipe_state_read,
        // Beta pipeline (consumer)
        PipelineBeta& beta_pipeline,
        PipelineStateBeta& beta_pipe_state_read,
        // CudaCore -> Inverse pipeline (producer)
        PipelineKKInvReady& kk_inv_pipeline,
        PipelineStateKKInv& kk_inv_pipe_state_write) {
        // === PERSISTENT CudaCore LOOP (static scheduling, no tile pipeline) ===
        //
        // CudaCore warpgroups: WG0 = thread [0,128), WG1 = thread [128,256)
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
        const int wg_idx = threadIdx.x / 128;  // 0 or 1 within CudaCore
        int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;
        constexpr int HalfK = TileK / 2;
        int k_offset = wg_idx * HalfK;

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_seqlens_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int seq_len = cu_seqlens_ptr[batch_idx + 1] - cu_seqlens_ptr[batch_idx];
            int sub_seq_len = min(TileT, seq_len - tile_idx * TileT);

            constexpr int kg_offset = SubTileT * TileK;  // stride between sub_tile buffers

            CUTE_NO_UNROLL
            for (int k_idx = 0; k_idx < NumKIters; ++k_idx) {
                // ============================================================
                // Step 1: Wait for Q, K, G data from TMA Load warp (unified pipeline)
                // ============================================================
                qkg_load_pipeline.consumer_wait(qkg_load_pipe_state_read);
                int buf_load_idx = qkg_load_pipe_state_read.index();

                // ============================================================
                // Step 2: Create SMEM tensor views for this buffer slot
                // ============================================================
                Tensor sK = make_tensor(make_smem_ptr(shared_plan->k[buf_load_idx].data()), SmemLayoutInputBF16{});
                Tensor sG = make_tensor(make_smem_ptr(shared_plan->g[buf_load_idx].data()), SmemLayoutInputFP32{});

                qkg_inter_pipeline.producer_acquire(qkg_inter_pipe_state_write);
                int buf_idx = qkg_inter_pipe_state_write.index();
                // B-matrix SMEM views (single-buffered kg_all)
                // Each sub_tile occupies one SmemLayoutMatBTF32<1> = (16 × 32)
                // inter[i] and intra[i] are indexed by KG_OFFSET * index inside the helper
                Tensor sKG_inter =
                    make_tensor(make_smem_ptr(shared_plan->kg_all[buf_idx].inter[0].data()), SmemLayoutMatBTF32<1>{});
                Tensor sKG_intra =
                    make_tensor(make_smem_ptr(shared_plan->kg_all[buf_idx].intra[0].data()), SmemLayoutMatBTF32<1>{});

                // ============================================================
                // A-matrix prologue: gated Q/K → TMEM (R2T) for all 4 subchunks
                // ============================================================
                // Thread mapping (128 threads per WG, 4 warps):
                //   row = idx_in_warpgroup % 64, sub_tile_i = row / 16
                //   Threads 0-63:   gated Q → TMEM QG_INTER / QG_INTRA (lanes 0-63)
                //   Threads 64-127: gated K → TMEM QG_INTER / QG_INTRA (lanes 64-127)
                //
                // Inter-chunk: qg/kg_i = q/k_i * exp2(g_i - g_first_i), g_first_i = g[sub_tile_i * 16]
                // Intra-chunk: qg/kg_i = q/k_i * exp2(g_i - g_half_i),  g_half_i  = g[sub_tile_i * 16 + 8]
                {
                    Tensor sQ = make_tensor(make_smem_ptr(shared_plan->q[buf_load_idx].data()), SmemLayoutInputBF16{});

                    if constexpr (UnifiedGRef) {
                        // Inter-only A-matrix: B-matrix intra also uses g_first reference,
                        // so inter and intra A-matrices are identical. Only compute inter.
                        // Saves 50% of A-matrix exp2f computation and one TMEM store.
                        fwd_setup_A_inter_all_QK<HalfK>(
                            sG,
                            sQ,
                            sK,
                            idx_in_warpgroup,
                            sub_seq_len,
                            k_offset,
                            static_cast<int>(TmemAllocation::QG_INTER) + buf_idx * 128);
                    } else {
                        // Fused inter+intra A-matrix: reads sG and sQ/sK ONCE per iteration,
                        // producing both QG_INTER and QG_INTRA in a single pass.
                        fwd_setup_A_inter_intra_all_QK<HalfK>(
                            sG,
                            sQ,
                            sK,
                            idx_in_warpgroup,
                            sub_seq_len,
                            k_offset,
                            static_cast<int>(TmemAllocation::QG_INTER) + buf_idx * 128,
                            static_cast<int>(TmemAllocation::QG_INTRA) + buf_idx * 128);
                    }

                    // NOTE: TMEM fence (tcgen05.wait::st, blocking) deferred past B-matrix
                    // to overlap TMEM store latency with B-matrix SMEM computation.
                    // Safe because B-matrix only touches SMEM, not TMEM.
                }

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
                        fwd_setup_kg_col0_4out<
                            decltype(sG),
                            decltype(sK),
                            decltype(sKG_inter),
                            kg_offset,
                            TileK,
                            UnifiedGRef>(sG, sK, sKG_inter, sKG_intra, idx_in_warpgroup, sub_seq_len);

                        // ---- WG0: Column j=3 (1 output) ----
                        // intra(3,3)
                        fwd_setup_kg_col3_1out<
                            decltype(sG),
                            decltype(sK),
                            decltype(sKG_intra),
                            kg_offset,
                            TileK,
                            UnifiedGRef>(sG, sK, sKG_intra, idx_in_warpgroup, sub_seq_len);
                    } else {
                        // ---- WG1: Column j=1 (3 outputs) ----
                        // intra(1,1) + inter(2,1) + inter(3,1)
                        fwd_setup_kg_col1_3out<
                            decltype(sG),
                            decltype(sK),
                            decltype(sKG_inter),
                            kg_offset,
                            TileK,
                            UnifiedGRef>(sG, sK, sKG_inter, sKG_intra, idx_in_warpgroup, sub_seq_len);

                        // ---- WG1: Column j=2 (2 outputs) ----
                        // intra(2,2) + inter(3,2)
                        fwd_setup_kg_col2_2out<
                            decltype(sG),
                            decltype(sK),
                            decltype(sKG_inter),
                            kg_offset,
                            TileK,
                            UnifiedGRef>(sG, sK, sKG_inter, sKG_intra, idx_in_warpgroup, sub_seq_len);
                    }
                }

                // ============================================================
                // Step 4: Fence TMEM + SMEM writes and signal MMA
                // ============================================================
                // TMEM fence deferred from after A-matrix to here, overlapping
                // TMEM store latency with B-matrix computation above.
                cutlass::arch::fence_view_async_tmem_store();
                ku::tcgen05_before_thread_sync();
                fence_view_async_shared();
                qkg_inter_pipeline.producer_commit(qkg_inter_pipe_state_write);
                ++qkg_inter_pipe_state_write;

                // ============================================================
                // Step 5: Release Q, K, G smem buffers back to TMA Load warp (unified pipeline)
                // ============================================================
                qkg_load_pipeline.consumer_release(qkg_load_pipe_state_read);
                ++qkg_load_pipe_state_read;
            }

            // ============================================================
            // Post-loop: wait for MMA results, epilogue, signal downstream
            // ============================================================
            // Reorder blocking waits: start waiting for inverse pipeline slot
            // and beta data before waiting for MMA, overlapping independent waits.
            kk_inv_pipeline.producer_acquire(kk_inv_pipe_state_write);

            beta_pipeline.consumer_wait(beta_pipe_state_read);

            qk_done_pipeline.consumer_wait(qk_done_pipe_state_read);
            int buf_acc_idx = qk_done_pipe_state_read.index();

            fence_view_async_shared();

            // ============================================================
            // QK + KK epilogue: T2R + mask + scale/beta → global / SMEM
            // ============================================================
            // Lower 64 threads in WG0: QK epilogue (scale + causal mask → global bf16)
            // Upper 64 threads in WG0: KK epilogue (beta + causal mask → SMEM tf32)
            //
            // TMEM address: QK = 0 (the MMA wrote QK/KK results at QK_02/QK_13;
            // tmem_ld_32dp32bNx reads all 64 rows correctly from the base
            // address QK = 0 for lower 64 threads, and KK occupies the
            // upper 64 TMEM lanes accessed by upper 64 threads).
            //
            // KK epilogue applies per-row beta scaling: KK[i, :] *= beta[i]
            // Beta is loaded from beta_smem[pipe_index][row] (Tx1 vector).
            {
                int token_offset = cu_seqlens_ptr[batch_idx];
                int row = idx_in_warpgroup % 64;
                int BT = TileT;
                int H = params.h;
                __nv_bfloat16* Aqk_base = reinterpret_cast<__nv_bfloat16*>(params.Aqk_out_ptr);
                __nv_bfloat16* qk_out_row =
                    Aqk_base + static_cast<int64_t>(token_offset + tile_idx * TileT + row) * H * BT + head_idx * BT;

                // Read per-row beta for KK scaling
                float beta_row = shared_plan->beta_smem[beta_pipe_state_read.index()][row];

                // Create SMEM tensor view for KK output (tf32/fp16)
                Tensor sKK = make_tensor(make_smem_ptr(shared_plan->kk[buf_acc_idx].data()), SmemLayoutInvKK{});

                if (wg_idx == 0) {
                    fwd_epilogue_qk_kk<TileT, UseTF32Inverse, RoundingTF32>(
                        static_cast<int>(TmemAllocation::QK) + buf_acc_idx * 128,
                        idx_in_warpgroup,
                        sub_seq_len,
                        params.scale,
                        beta_row,
                        qk_out_row,
                        sKK);
                }
            }

            fence_view_async_shared();
            kk_inv_pipeline.producer_commit(kk_inv_pipe_state_write);
            ++kk_inv_pipe_state_write;

            beta_pipeline.consumer_release(beta_pipe_state_read);
            ++beta_pipe_state_read;

            qk_done_pipeline.consumer_release(qk_done_pipe_state_read);
            ++qk_done_pipe_state_read;
        }
    }

    // ===================================================================
    // MMA warp persistent loop (warp 12, elect_one)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    mma_loop(
        const KDA_fwd_intra_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // CudaCore -> MMA pipelines (consumer)
        PipelineQKGInterReady& qkg_inter_pipeline,
        PipelineStateQKGInter& qkg_inter_pipe_state_read,
        // MMA -> CudaCore pipelines (producer)
        PipelineQKDone& qk_done_pipeline,
        PipelineStateQKDone& qk_done_pipe_state_write) {
        // === PERSISTENT MMA LOOP (static scheduling, no tile pipeline) ===
        int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;

        TiledMMA tile_mma_qk_n16_mask02 = TiledMMA_KDAqk_N16_MASK02{};
        TiledMMA tile_mma_qk_n16_mask13 = TiledMMA_KDAqk_N16_MASK13{};
        TiledMMA tile_mma_qk_n32_mask02 = TiledMMA_KDAqk_N32_MASK02{};
        TiledMMA tile_mma_qk_n32_mask13 = TiledMMA_KDAqk_N32_MASK13{};
        TiledMMA tile_mma_qk_n48_mask02 = TiledMMA_KDAqk_N48_MASK02{};
        TiledMMA tile_mma_qk_n48_mask13 = TiledMMA_KDAqk_N48_MASK13{};

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_seqlens_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);

            // MMA computation body
            qk_done_pipeline.producer_acquire(qk_done_pipe_state_write);
            int buf_acc_idx = qk_done_pipe_state_write.index();
            CUTE_NO_UNROLL
            for (int k_idx = 0; k_idx < NumKIters; ++k_idx) {
                qkg_inter_pipeline.consumer_wait(qkg_inter_pipe_state_read);
                int buf_idx = qkg_inter_pipe_state_read.index();

                // inter-chunk: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2)
                Tensor tQK_row1 =
                    partition_fragment_C(tile_mma_qk_n16_mask02, make_shape(Int<TileT>{}, Int<SubTileT>{}));
                Tensor tQK_row2 =
                    partition_fragment_C(tile_mma_qk_n32_mask02, make_shape(Int<TileT>{}, Int<SubTileT * 2>{}));
                Tensor tQK_row3 =
                    partition_fragment_C(tile_mma_qk_n48_mask02, make_shape(Int<TileT>{}, Int<SubTileT * 3>{}));
                // row1, (1,0), qk[1, 3]-kk[1, 3], mask02
                tQK_row1.data() = uint32_t(TmemAllocation::QK_13) + buf_acc_idx * 128;
                // row2, (2,0) (2,1), qk[0, 2]-kk[0, 2], mask13
                tQK_row2.data() = uint32_t(TmemAllocation::QK_02) + buf_acc_idx * 128;
                // row3, (3,0) (3,1) (3,2), qk[1, 3]-kk[1, 3], mask13
                tQK_row3.data() = uint32_t(TmemAllocation::QK_13) + buf_acc_idx * 128;

                ku::tcgen05_after_thread_sync();
                // inter-chunk: 3 MMA calls
                // clear_accum only on first k_idx iteration; accumulate across NumKIters
                {
                    bool first_iter = (k_idx == 0);
                    Tensor tQ_1 = tile_mma_qk_n16_mask02.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tile_mma_qk_n16_mask02, Shape<Int<ChunkSize>, Int<TileK>>{}));
                    tQ_1.data() = uint32_t(TmemAllocation::QG_INTER_13) + buf_idx * 128;
                    Tensor sKG_1 = make_tensor(
                        make_smem_ptr(shared_plan->kg_all[buf_idx].inter[0].data()), SmemLayoutMatBTF32<1>{});
                    ku::utcmma_ts(tile_mma_qk_n16_mask02, tQ_1, sKG_1, tQK_row1, first_iter);

                    Tensor tQ_2 = tile_mma_qk_n32_mask13.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tile_mma_qk_n32_mask13, Shape<Int<ChunkSize>, Int<TileK>>{}));
                    tQ_2.data() = uint32_t(TmemAllocation::QG_INTER_02) + buf_idx * 128;
                    Tensor sKG_2 = make_tensor(
                        make_smem_ptr(shared_plan->kg_all[buf_idx].inter[1].data()), SmemLayoutMatBTF32<2>{});
                    ku::utcmma_ts(tile_mma_qk_n32_mask13, tQ_2, sKG_2, tQK_row2, first_iter);

                    Tensor tQ_3 = tile_mma_qk_n48_mask13.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tile_mma_qk_n48_mask13, Shape<Int<ChunkSize>, Int<TileK>>{}));
                    tQ_3.data() = uint32_t(TmemAllocation::QG_INTER_13) + buf_idx * 128;
                    Tensor sKG_3 = make_tensor(
                        make_smem_ptr(shared_plan->kg_all[buf_idx].inter[3].data()), SmemLayoutMatBTF32<3>{});
                    ku::utcmma_ts(tile_mma_qk_n48_mask13, tQ_3, sKG_3, tQK_row3, first_iter);
                }

                ku::tcgen05_after_thread_sync();

                // intra-chunk: 4 MMA calls
                // intra-chunk: (0,0), (1,1), (2,2), (3,3)
                Tensor tQK_00 = partition_fragment_C(tile_mma_qk_n16_mask02, make_shape(Int<TileT>{}, Int<SubTileT>{}));
                Tensor tQK_11 = partition_fragment_C(tile_mma_qk_n16_mask02, make_shape(Int<TileT>{}, Int<SubTileT>{}));
                Tensor tQK_22 = partition_fragment_C(tile_mma_qk_n16_mask02, make_shape(Int<TileT>{}, Int<SubTileT>{}));
                Tensor tQK_33 = partition_fragment_C(tile_mma_qk_n16_mask02, make_shape(Int<TileT>{}, Int<SubTileT>{}));
                // (0,0) qk[0, 2]-kk[0, 2], mask02, column offset 0
                tQK_00.data() = uint32_t(TmemAllocation::QK_02) + buf_acc_idx * 128;
                // (1,1) qk[1, 3]-kk[1, 3], mask02, column offset 16
                tQK_11.data() = uint32_t(TmemAllocation::QK_13) + 16 + buf_acc_idx * 128;
                // (2,2) qk[0, 2]-kk[0, 2], mask13, column offset 32
                tQK_22.data() = uint32_t(TmemAllocation::QK_02) + 32 + buf_acc_idx * 128;
                // (3,3) qk[1, 3]-kk[1, 3], mask13, column offset 48
                tQK_33.data() = uint32_t(TmemAllocation::QK_13) + 48 + buf_acc_idx * 128;

                {
                    bool first_iter = (k_idx == 0);
                    constexpr auto IntraA_02 = UnifiedGRef ? TmemAllocation::QG_INTER_02 : TmemAllocation::QG_INTRA_02;
                    constexpr auto IntraA_13 = UnifiedGRef ? TmemAllocation::QG_INTER_13 : TmemAllocation::QG_INTRA_13;

                    Tensor tQ_0 = tile_mma_qk_n16_mask02.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tile_mma_qk_n16_mask02, Shape<Int<ChunkSize>, Int<TileK>>{}));
                    tQ_0.data() = uint32_t(IntraA_02) + buf_idx * 128;
                    Tensor sKG_0 = make_tensor(
                        make_smem_ptr(shared_plan->kg_all[buf_idx].intra[0].data()), SmemLayoutMatBTF32<1>{});
                    ku::utcmma_ts(tile_mma_qk_n16_mask02, tQ_0, sKG_0, tQK_00, first_iter);

                    Tensor tQ_1 = tile_mma_qk_n16_mask02.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tile_mma_qk_n16_mask02, Shape<Int<ChunkSize>, Int<TileK>>{}));
                    tQ_1.data() = uint32_t(IntraA_13) + buf_idx * 128;
                    Tensor sKG_1 = make_tensor(
                        make_smem_ptr(shared_plan->kg_all[buf_idx].intra[1].data()), SmemLayoutMatBTF32<1>{});
                    ku::utcmma_ts(tile_mma_qk_n16_mask02, tQ_1, sKG_1, tQK_11, first_iter);

                    Tensor tQ_2 = tile_mma_qk_n16_mask02.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tile_mma_qk_n16_mask02, Shape<Int<ChunkSize>, Int<TileK>>{}));
                    tQ_2.data() = uint32_t(IntraA_02) + buf_idx * 128;
                    Tensor sKG_2 = make_tensor(
                        make_smem_ptr(shared_plan->kg_all[buf_idx].intra[2].data()), SmemLayoutMatBTF32<1>{});
                    ku::utcmma_ts(tile_mma_qk_n16_mask13, tQ_2, sKG_2, tQK_22, first_iter);

                    Tensor tQ_3 = tile_mma_qk_n16_mask02.get_slice(_0{}).make_fragment_A(
                        partition_shape_A(tile_mma_qk_n16_mask02, Shape<Int<ChunkSize>, Int<TileK>>{}));
                    tQ_3.data() = uint32_t(IntraA_13) + buf_idx * 128;
                    Tensor sKG_3 = make_tensor(
                        make_smem_ptr(shared_plan->kg_all[buf_idx].intra[3].data()), SmemLayoutMatBTF32<1>{});
                    ku::utcmma_ts(tile_mma_qk_n16_mask13, tQ_3, sKG_3, tQK_33, first_iter);
                }

                qkg_inter_pipeline.consumer_release(qkg_inter_pipe_state_read);
                ++qkg_inter_pipe_state_read;
            }
            // notify MMA finished to CudaCore
            qk_done_pipeline.producer_commit(qk_done_pipe_state_write);
            ++qk_done_pipe_state_write;
        }
    }

    // ===================================================================
    // Load warp persistent loop (warp 13, elect_one, TMA producer)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    load_loop(
        const KDA_fwd_intra_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // Unified TMA pipeline for Q+K+G (producer)
        PipelineQKG& qkg_load_pipeline,
        PipelineStateQKG& qkg_load_pipe_state_write) {
        if (cute::elect_one_sync()) {
            int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
            int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;
            // === PERSISTENT LOAD LOOP (static scheduling, no tile pipeline) ===
            CUTE_NO_UNROLL
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int tid = tile_scheduler.get_current_tile_id();

                // Decode tile coordinates
                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_seqlens_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);
                int token_offset = cu_seqlens_ptr[batch_idx];
                int seq_len = cu_seqlens_ptr[batch_idx + 1] - cu_seqlens_ptr[batch_idx];
                int sub_seq_len = min(TileT, seq_len - tile_idx * TileT);

                Tensor mQ = domain_offset(
                    make_coord(token_offset, _0{}, _0{}), tma_params.tma_q.get_tma_tensor(tma_params.shape_qkg));
                Tensor mK = domain_offset(
                    make_coord(token_offset, _0{}, _0{}), tma_params.tma_k.get_tma_tensor(tma_params.shape_qkg));
                Tensor mG = domain_offset(
                    make_coord(token_offset, _0{}, _0{}), tma_params.tma_g.get_tma_tensor(tma_params.shape_qkg));

                // TMA load body (Q, K, G — unified pipeline, single barrier per stage)
                CUTE_NO_UNROLL
                for (int k_idx = 0; k_idx < NumKIters; ++k_idx) {
                    int buf_idx = qkg_load_pipe_state_write.index();
                    Tensor sQ = make_tensor(make_smem_ptr(shared_plan->q[buf_idx].data()), SmemLayoutInputBF16{});
                    Tensor sK = make_tensor(make_smem_ptr(shared_plan->k[buf_idx].data()), SmemLayoutInputBF16{});
                    Tensor sG = make_tensor(make_smem_ptr(shared_plan->g[buf_idx].data()), SmemLayoutInputFP32{});

                    Tensor gK = local_tile(
                        mK(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, k_idx));
                    Tensor gG = local_tile(
                        mG(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, k_idx));
                    Tensor gQ = local_tile(
                        mQ(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, k_idx));

                    // Single acquire for all three TMA copies
                    qkg_load_pipeline.producer_acquire(qkg_load_pipe_state_write);
                    auto& barrier = *qkg_load_pipeline.producer_get_barrier(qkg_load_pipe_state_write);
                    ku::launch_tma_copy(tma_params.tma_g, gG, sG, barrier);
                    ku::launch_tma_copy(tma_params.tma_k, gK, sK, barrier);
                    ku::launch_tma_copy(tma_params.tma_q, gQ, sQ, barrier);
                    ++qkg_load_pipe_state_write;
                }
            }
        }
    }

    // ===================================================================
    // Inverse warpgroup persistent loop (warp 8-11, 1 warpgroup)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    inverse_loop(
        const KDA_fwd_intra_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // CudaCore -> Inverse pipeline (consumer)
        PipelineKKInvReady& kk_inv_pipeline,
        PipelineStateKKInv& kk_inv_pipe_state_read) {
        // === PERSISTENT INVERSE LOOP (static scheduling, no tile pipeline) ===
        int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;
        static_assert(UseTF32Inverse || sizeof(InverseType) == sizeof(Element));

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_seqlens_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int token_offset = cu_seqlens_ptr[batch_idx];
            int seq_len = cu_seqlens_ptr[batch_idx + 1] - cu_seqlens_ptr[batch_idx];
            int sub_seq_len = min(TileT, seq_len - tile_idx * TileT);
            int token_offset_cur = token_offset + tile_idx * TileT;

            // KK R2G Store
            Tensor mO = make_tensor(
                make_gmem_ptr(reinterpret_cast<Element*>(params.Akk_out_ptr)),
                make_layout(params.shape_Akk, params.stride_Akk))(_, _, head_idx);
            // NOTE: currently hardcode to _0{} chunk because each tile only processes one chunk
            Tensor gO = local_tile(
                cute::domain_offset(make_coord(token_offset_cur, _0{}), mO),
                select<0, 1>(TileShapeKK{}),
                make_coord(_0{}, _0{}));

            // Inverse computation body
            kk_inv_pipeline.consumer_wait(kk_inv_pipe_state_read);
            fence_view_async_shared();

            // Create SMEM tensor view for KK output (fp16)
            Tensor sKK_inv =
                make_tensor(make_smem_ptr(shared_plan->kk[kk_inv_pipe_state_read.index()].data()), SmemLayoutInvKK{});
            // if (thread_idx == 0) {
            //     printf("sKK, Before Inverse\n");
            //     cute::print_tensor(sKK_inv);
            // }
            auto sKK_inv_pipe_slice = sKK_inv(_, _);
            auto sKK_out_pipe_slice = [&]() {
                if constexpr (!UseTF32Inverse) {
                    return sKK_inv_pipe_slice;
                } else {
                    return recast<float>(sKK_inv_pipe_slice);
                }
            }();
            auto collective_inverse = CollectiveInverse(KdaChunkFwdIntraSm100NamedBarriers::InverseMath);
            collective_inverse.compute(sKK_inv_pipe_slice);

            // cast to Element in registers, then R2G directly — no extra R2S + S2R round-trip
            using GmemTileCopyAtomInv = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, InverseOutputType>;
            using GmemTiledCopyInv = decltype(make_tiled_copy(
                GmemTileCopyAtomInv{}, GmemLayoutAtom{}, Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));

            GmemTiledCopyInv gmem_tiled_copy_inv;
            auto gmem_thr_copy_inv = gmem_tiled_copy_inv.get_thread_slice(thread_idx);

            GmemTiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

            // Initialize tOcO and tOpO to predict OOB access
            Tensor tOcO = gmem_thr_copy_O.partition_D(make_identity_tensor(select<0, 1>(TileShapeKK{})));
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
#pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_Akk);
            }
            // Initialize tOgO to store O to gmem
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

            // wait for inverse done
            cutlass::arch::NamedBarrier::arrive_and_wait(
                cutlass::NumThreadsPerWarpGroup, KdaChunkFwdIntraSm100NamedBarriers::InverseMath);
            // if (thread_idx == 0) {
            //     printf("sKK, After Inverse\n");
            //     cute::print_tensor(sKK_inv);
            // }

            // S2R with GmemTiledCopy layout, reading InverseType from smem
            Tensor tOsInv = gmem_thr_copy_inv.partition_S(sKK_out_pipe_slice);
            Tensor tOrInv = make_fragment_like(tOsInv);
            cute::copy(gmem_tiled_copy_inv, tOsInv, tOrInv);

            // Cast InverseType -> Element in registers, then R2G directly
            Tensor tOrFinalO = make_fragment_like<Element>(tOrInv);
#pragma unroll
            for (int i = 0; i < size(tOrInv); ++i) {
                tOrFinalO(i) = Element(tOrInv(i));
            }

            // R2G directly
            ku::copy_pred</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrFinalO, tOgO, tOcO, tOpO, sub_seq_len);

            kk_inv_pipeline.consumer_release(kk_inv_pipe_state_read);
            ++kk_inv_pipe_state_read;
        }
    }

    // ===================================================================
    // Load beta warp persistent loop (warp 14-15, beta loading)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    load_beta_loop(
        const KDA_fwd_intra_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // Beta pipeline (producer)
        PipelineBeta& beta_pipeline,
        PipelineStateBeta& beta_pipe_state_write) {
        // === PERSISTENT LOAD BETA WARP LOOP (static scheduling, no tile pipeline) ===
        int thread_idx = threadIdx.x % 64;  // 0..63
        int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_seqlens_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int token_offset = cu_seqlens_ptr[batch_idx];
            int seq_len = cu_seqlens_ptr[batch_idx + 1] - cu_seqlens_ptr[batch_idx];
            int sub_seq_len = min(TileT, seq_len - tile_idx * TileT);

            // Beta loading body
            beta_pipeline.producer_acquire(beta_pipe_state_write);
            if (thread_idx < TileT) {
                shared_plan->beta_smem[beta_pipe_state_write.index()][thread_idx] =
                    (thread_idx < sub_seq_len)
                        ? float(reinterpret_cast<ElementBeta_*>(
                              params.beta_ptr)[(token_offset + tile_idx * TileT + thread_idx) * params.h + head_idx])
                        : float(0);
            }
            fence_view_async_shared();
            beta_pipeline.producer_commit(beta_pipe_state_write);
            ++beta_pipe_state_write;
        }
    }
};

}  // namespace kda::sm100