#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include "kerutils/kerutils.cuh"

#include "kda/sm100/kda_fwd_common.cuh"

namespace kda::sm100 {

using cutlass::arch::fence_view_async_shared;
using ku::bf16;
using ku::float2_mul;
using ku::nvbf16x4;
using ku::store_256b;
using namespace cute;

struct KdaChunkFwdRecompWUSm100NamedBarriers {
    static constexpr int PrologueCudaCore = 0;
    static constexpr int EpilogueCudaCore = 1;
};

// ===================================================================
// Mainloop struct: KdaChunkFwdRecompWUMainloopSm100
// Self-contained: owns all pipeline types, SMEM layouts, SharedMemoryPlan,
// constants, and the persistent loop bodies for each warp role.
// The Kernel struct is templated on this Mainloop.
// ===================================================================
struct KdaChunkFwdRecompWUMainloopSm100 {
    // ===================== Tile / Buffer Constants =====================
    static constexpr int TileT = 64;
    static constexpr int HeadDim = 128;
    static constexpr int TileK = 128;  // increase to 128, greater perf
    static constexpr int NumKIters = HeadDim / TileK;
    static constexpr int ChunkSize = 64;
    static constexpr int StagesLoadStore = 2;  // increase to 2, greater perf
    static constexpr int StagesA = 2;          // increase to 2, greater perf
    static constexpr int StagesMma = 1;

    // TODO: double buffer for TMEM acc
    enum class TmemAllocation : uint32_t {
        W = 0,               // W, acc, single buffer, [0, 64],
        U = W + 16 * 65536,  // U, acc, [0, 64] +lane16
    };

    using TileScheduler = StaticPersistentTileScheduler;

    // ===================== SMEM Layouts =====================
    // Q, K (bf16)
    using SmemLayoutInputBF16 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<TileT>, Int<TileK>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    // Akk (bf16)
    using SmemLayoutInputAkkBF16 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<TileT>, Int<TileT>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    // G (fp32)
    using SmemLayoutInputFP32 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW128_Atom<float>{}, Shape<Int<TileT>, Int<TileK>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    // MMA B-operand layout: K_proc/V_proc in SMEM, shape [N=TileK, K=TileT] MN-major
    // MMA semantics: C[M,N] = A[M,K] @ B[N,K]^T
    //   A = Akk [M=64, K=64] in TMEM (K-major)
    //   B = K_proc [N=32, K=64] in SMEM (MN-major, UMMA transposes internally)
    //   C = w [M=64, N=32] in TMEM accumulator
    // Since K dim = BT = 64 (reduce over chunk), N dim = BK = 32 (head dim slice),
    // B-operand is stored as [N=TileK, K=TileT] = [32, 64] in MN-major layout.
    using SmemLayoutMatBBF16 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<Int<TileK>, Int<TileT>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    // W/U/KG output (bf16)
    using SmemLayoutOutputBF16 = decltype(coalesce(
        tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<TileT>, Int<TileK>>{}, Step<_1, _2>{}),
        Shape<_1, _1>{}));

    using TiledMMA_KDAak = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS<bf16, bf16, float, TileT, TileK, UMMA::Major::K, UMMA::Major::MN>{}));

    // ===================== Pipeline Types =====================
    // TMA load -> MMA (Akk)
    using PipelineA = cutlass::PipelineTmaAsync<StagesA>;
    // TMA load -> Compute (merged prologue+epilogue)
    using PipelineV = cutlass::PipelineTmaAsync<StagesLoadStore>;
    // TMA load -> Compute (merged K+G, two TMA copies share one barrier)
    using PipelineKG = cutlass::PipelineTmaAsync<StagesLoadStore>;
    // Aux load -> Compute
    using PipelineBeta = cutlass::PipelineAsync<StagesA>;

    // Unified pipeline: Compute -> MMA (K/V prologue ready share one pipeline, used sequentially)
    using PipelinePrologueReady = cutlass::PipelineAsync<StagesMma>;
    // Unified pipeline: MMA -> Compute (W/U acc done share one pipeline, used sequentially)
    using PipelineAccDone = cutlass::PipelineUmmaAsync<StagesMma>;

    // ===================== GMEM Store ===========
    // W/U/KG: R2G store bf16, (TileT, TileK)
    using Element = cutlass::bfloat16_t;
    // Adapted from
    // https://github.com/Dao-AILab/flash-attention/blob/9b6dbaceb658f576ea81e2b0189f4b5707a39aae/hopper/epilogue_fwd.hpp#L51
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);  // 16/2=8
    static_assert(TileK % kGmemElemsPerStore == 0, "Chunk size must be a multiple of kGmemElemsPerStore for Aqk/Akk");
    static constexpr int kBytePerRow = TileK * sizeof(Element);  // 128x2=256
    static constexpr int kBlockKGmem =
        (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);  // 128/2=64
    // Number of threads required to collaboratively read/write one (128-byte, 64-byte, or 32-byte) block
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;  // 8
    static constexpr int NumPrologueThreads = cutlass::NumThreadsPerWarpGroup;   // 128 threads (WG0, warp 0-3)
    static constexpr int NumEpilogueThreads = cutlass::NumThreadsPerWarpGroup;   // 128 threads (WG1, warp 4-7)
    static_assert(
        NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    // Layout of Epilogue threads for GMEM store, named GmemLayoutAtom
    using GmemLayoutAtom = Layout<
        Shape<Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTileCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        GmemTileCopyAtomO{},
        GmemLayoutAtom{},
        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));  // Val layout, 8 vals per store

    // ===================== Dummy MMA for R2S and S2R ==============
    using MMA = SM80_16x8x16_F32BF16BF16F32_TN;
    // one warpgroup to load (TileT, TileK) data
    using TileShape_S2R = Shape<_64, _64, _128>;
    using TiledMma_S2R = decltype(make_tiled_mma(MMA{}, Layout<Shape<_4, _1, _1>>{}, TileShape_S2R{}));
    using CopyGAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>;
    using CopyS2RAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using CopyR2SAtom = Copy_Atom<SM90_U32x4_STSM_N, Element>;

    // ===================== Shared Memory Plan =====================
    struct SharedMemoryPlan {
        // Akk, single buffer
        array_aligned<bf16, cosize_v<SmemLayoutInputAkkBF16>> akk[StagesA];  // 16KB
        // K, V, G double buffer
        array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> k[StagesLoadStore];   // 32KB
        array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> v[StagesLoadStore];   // 32KB
        array_aligned<float, cosize_v<SmemLayoutInputFP32>> g[StagesLoadStore];  // 64KB
        // MMA B-operand staging: K_proc/V_proc after prologue, [N=TileK, K=TileT] MN-major, double buffer
        array_aligned<bf16, cosize_v<SmemLayoutMatBBF16>> k_mma[StagesMma];  // 16KB
        array_aligned<bf16, cosize_v<SmemLayoutMatBBF16>> v_mma[StagesMma];  // 16KB
        // Epilogue store buffer
        array_aligned<bf16, cosize_v<SmemLayoutOutputBF16>> out[1];  // 16KB

        alignas(16) float beta_smem[StagesA][TileT];
        array_aligned<uint32_t, 1> tmem_start_addr;

        // Pipeline shared storage
        alignas(16) typename PipelineA::SharedStorage pipe_a_storage;
        alignas(16) typename PipelineKG::SharedStorage pipe_kg_storage;
        alignas(16) typename PipelineV::SharedStorage pipe_v_storage;
        alignas(16) typename PipelineBeta::SharedStorage pipe_beta_storage;
        alignas(16) typename PipelinePrologueReady::SharedStorage pipe_prologue_ready_storage;
        alignas(16) typename PipelineAccDone::SharedStorage pipe_acc_done_storage;
    };

    // ===================== TMA Params =====================
    template <typename ShapeKVG, typename ShapeAkk, typename TMA_V, typename TMA_K, typename TMA_G, typename TMA_Akk>
    struct TmaParams {
        ShapeKVG shape_kvg;
        ShapeAkk shape_Akk;
        TMA_V tma_v;
        TMA_K tma_k;
        TMA_G tma_g;
        TMA_Akk tma_akk;
    };

    // ===================== Pipeline State Types =====================
    using PipelineStateA = cutlass::PipelineState<PipelineA::Stages>;
    using PipelineStateKG = cutlass::PipelineState<PipelineKG::Stages>;
    using PipelineStateV = cutlass::PipelineState<PipelineV::Stages>;
    using PipelineStateBeta = cutlass::PipelineState<PipelineBeta::Stages>;
    using PipelineStatePrologueReady = cutlass::PipelineState<PipelinePrologueReady::Stages>;
    using PipelineStateAccDone = cutlass::PipelineState<PipelineAccDone::Stages>;

    // ===================================================================
    // WG0: Prologue persistent loop (warp 0-3, 128 threads, 1 WG)
    // Element-wise K_proc/V_proc computation → signal MMA
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    prologue_loop(
        const KDA_fwd_recomp_w_u_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // TMA pipelines (consumer): KG (merged K+G), V
        PipelineKG& kg_pipeline,
        PipelineStateKG& kg_pipe_state_read,
        PipelineV& v_pipeline,
        PipelineStateV& v_pipe_state_read,
        // Beta pipeline (consumer, 1×/WU)
        PipelineBeta& beta_pipeline,
        PipelineStateBeta& beta_pipe_state_read,
        // Prologue -> MMA pipeline (producer, used for both K and V sequentially)
        PipelinePrologueReady& prologue_ready_pipeline,
        PipelineStatePrologueReady& prologue_ready_pipe_state_write) {
        // === PERSISTENT PROLOGUE LOOP (WG0, 128 threads) ===
        int idx_in_wg = threadIdx.x % NumPrologueThreads;  // 0..127
        int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();
            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_seqlens_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int seq_len = cu_seqlens_ptr[batch_idx + 1] - cu_seqlens_ptr[batch_idx];
            int sub_seq_len = min(TileT, seq_len - tile_idx * TileT);

            // ============================================================
            // Once per WU: Wait for beta
            // ============================================================
            beta_pipeline.consumer_wait(beta_pipe_state_read);
            fence_view_async_shared();

            // ============================================================
            // Per i_k iteration: K_proc, V_proc element-wise → signal MMA
            // ============================================================
            CUTE_NO_UNROLL
            for (int i_k = 0; i_k < NumKIters; ++i_k) {
                // Wait for K, V, G data from TMA Load warp
                kg_pipeline.consumer_wait(kg_pipe_state_read);

                Tensor sK = make_tensor(
                    make_smem_ptr(shared_plan->k[kg_pipe_state_read.index()].data()), SmemLayoutInputBF16{});
                Tensor sG = make_tensor(
                    make_smem_ptr(shared_plan->g[kg_pipe_state_read.index()].data()), SmemLayoutInputFP32{});

                // ---- K_proc: K * beta * exp2(G) → R2S → k_mma ----
                prologue_ready_pipeline.producer_acquire(prologue_ready_pipe_state_write);
                int buf_idx = prologue_ready_pipe_state_write.index();
                Tensor sK_dst = make_tensor(make_smem_ptr(shared_plan->k_mma[buf_idx].data()), SmemLayoutMatBBF16{});
                {
                    // 128 threads cooperate using setup_kg-style thread mapping:
                    //   x_local = idx_in_wg / 8  → row within 16-row group (0..15)
                    //   y_base  = idx_in_wg % 8 * 4 → column group base (0,4,..,28)
                    // Each thread processes float4 (4 floats) + nvbf16x4 (4 bf16) per iteration.
                    // Nested loops over t_iter (row tiles: 0,16,32,48) and y_iter (col chunks: 0,32,64,96)
                    // cover the full [TileT=64, TileK=128] tile.
                    int x_local = idx_in_wg / 8;     // 0..15
                    int y_base = idx_in_wg % 8 * 4;  // 0,4,8,...,28

#pragma unroll
                    for (int t_iter = 0; t_iter < TileT; t_iter += 16) {
                        int t = x_local + t_iter;
                        float beta_val = shared_plan->beta_smem[beta_pipe_state_read.index()][t];
                        float2 beta2 = {beta_val, beta_val};

#pragma unroll
                        for (int y_iter = 0; y_iter < TileK; y_iter += 32) {
                            int y = y_base + y_iter;
                            if (t < sub_seq_len) {
                                float4 g = *reinterpret_cast<float4*>(&sG(t, y));
                                nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(t, y));
                                float2 kf_a = __bfloat1622float2(k.a);
                                float2 kf_b = __bfloat1622float2(k.b);
                                float2 g_a = {exp2f(g.x), exp2f(g.y)};
                                float2 g_b = {exp2f(g.z), exp2f(g.w)};
                                float2 res_a = float2_mul(float2_mul(kf_a, beta2), g_a);
                                float2 res_b = float2_mul(float2_mul(kf_b, beta2), g_b);
                                nvbf16x4 out;
                                out.a = __float22bfloat162_rn(res_a);
                                out.b = __float22bfloat162_rn(res_b);
                                *reinterpret_cast<nvbf16x4*>(&sK_dst(y, t)) = out;
                            } else {
                                nvbf16x4 zero;
                                zero.a = __float2bfloat162_rn(0.0f);
                                zero.b = __float2bfloat162_rn(0.0f);
                                *reinterpret_cast<nvbf16x4*>(&sK_dst(y, t)) = zero;
                            }
                        }
                    }
                }

                v_pipeline.consumer_wait(v_pipe_state_read);
                Tensor sV =
                    make_tensor(make_smem_ptr(shared_plan->v[v_pipe_state_read.index()].data()), SmemLayoutInputBF16{});
                // ---- V_proc: V * beta → R2S → v_mma ----
                Tensor sV_dst = make_tensor(make_smem_ptr(shared_plan->v_mma[buf_idx].data()), SmemLayoutMatBBF16{});
                {
                    // Same thread mapping as K_proc:
                    //   x_local = idx_in_wg / 8  → row within 16-row group (0..15)
                    //   y_base  = idx_in_wg % 8 * 4 → column group base (0,4,..,28)
                    int x_local = idx_in_wg / 8;
                    int y_base = idx_in_wg % 8 * 4;

#pragma unroll
                    for (int t_iter = 0; t_iter < TileT; t_iter += 16) {
                        int t = x_local + t_iter;
                        float beta_val = shared_plan->beta_smem[beta_pipe_state_read.index()][t];
                        float2 beta2 = {beta_val, beta_val};

#pragma unroll
                        for (int y_iter = 0; y_iter < TileK; y_iter += 32) {
                            int y = y_base + y_iter;
                            if (t < sub_seq_len) {
                                nvbf16x4 v = *reinterpret_cast<nvbf16x4*>(&sV(t, y));
                                float2 vf_a = __bfloat1622float2(v.a);
                                float2 vf_b = __bfloat1622float2(v.b);
                                float2 res_a = float2_mul(vf_a, beta2);
                                float2 res_b = float2_mul(vf_b, beta2);
                                nvbf16x4 out;
                                out.a = __float22bfloat162_rn(res_a);
                                out.b = __float22bfloat162_rn(res_b);
                                *reinterpret_cast<nvbf16x4*>(&sV_dst(y, t)) = out;
                            } else {
                                nvbf16x4 zero;
                                zero.a = __float2bfloat162_rn(0.0f);
                                zero.b = __float2bfloat162_rn(0.0f);
                                *reinterpret_cast<nvbf16x4*>(&sV_dst(y, t)) = zero;
                            }
                        }
                    }
                }

                // Tensor sK_dst_kmajor = make_tensor(make_smem_ptr(shared_plan->k_mma[buf_idx].data()),
                // SmemLayoutInputBF16{});
                // =====DEBUG=======
                // cutlass::arch::NamedBarrier::arrive_and_wait(NumPrologueThreads,
                // KdaChunkFwdRecompWUSm100NamedBarriers::PrologueCudaCore); if (idx_in_wg == 0) {
                //     printf("sK_dst\n");
                //     cute::print_tensor(sK_dst);
                //     printf("sK_dst_kmajor\n");
                //     cute::print_tensor(sK_dst_kmajor);
                // }

                fence_view_async_shared();
                prologue_ready_pipeline.producer_commit(prologue_ready_pipe_state_write);
                ++prologue_ready_pipe_state_write;

                kg_pipeline.consumer_release(kg_pipe_state_read);
                ++kg_pipe_state_read;
                // TODO: add store QG for disable_recompute=true
                // Release V, KG SMEM back to Load warp (prologue is done with them)
                v_pipeline.consumer_release(v_pipe_state_read);
                ++v_pipe_state_read;
            }

            // Release beta at end of WU
            beta_pipeline.consumer_release(beta_pipe_state_read);
            ++beta_pipe_state_read;
        }
    }

    // ===================================================================
    // WG1: Epilogue persistent loop (warp 4-7, 128 threads, 1 WG)
    // kg element-wise + MMA result store (w, u) → GMEM
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    epilogue_loop(
        const KDA_fwd_recomp_w_u_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // TMA pipeline (consumer): KG (merged K+G, for kg computation)
        PipelineKG& kg_pipeline,
        PipelineStateKG& kg_pipe_state_read,
        // MMA -> Epilogue pipeline (consumer, used for both W and U sequentially)
        PipelineAccDone& acc_done_pipeline,
        PipelineStateAccDone& acc_done_pipe_state_read) {
        // === PERSISTENT EPILOGUE LOOP (WG1, 128 threads) ===
        // idx_in_wg: 0..127 within this warp group
        int idx_in_wg = threadIdx.x % NumEpilogueThreads;  // 0..127
        int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;

        // Setup autovec GMEM store tiled copy (for epilogue S2G)
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx_in_wg);

        // S2R & R2S exp(g_last-g)*k
        auto tiledmma_s2r = TiledMma_S2R{};
        auto thr_mma_s2r = tiledmma_s2r.get_thread_slice(idx_in_wg);
        auto tiled_load_k = make_tiled_copy_A(CopyS2RAtom{}, thr_mma_s2r);
        auto thr_load_k = tiled_load_k.get_thread_slice(idx_in_wg);
        auto tiled_load_g = make_tiled_copy_A(CopyGAtom{}, thr_mma_s2r);
        auto thr_load_g = tiled_load_g.get_thread_slice(idx_in_wg);
        auto tiled_store_k = make_tiled_copy_A(CopyR2SAtom{}, thr_mma_s2r);
        auto thr_store_k = tiled_store_k.get_thread_slice(idx_in_wg);

        auto cMk = make_identity_tensor(select<0, 2>(TileShape_S2R{}));
        auto tKcMk = thr_mma_s2r.partition_A(cMk);

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

            // GMEM output tensors (sliced by head)
            Tensor mKg = make_tensor(
                make_gmem_ptr(reinterpret_cast<Element*>(params.kg_out_ptr)),
                make_layout(params.shape_wukg, params.stride_wukg))(_, _, head_idx);
            Tensor mW = make_tensor(
                make_gmem_ptr(reinterpret_cast<Element*>(params.w_out_ptr)),
                make_layout(params.shape_wukg, params.stride_wukg))(_, _, head_idx);
            Tensor mU = make_tensor(
                make_gmem_ptr(reinterpret_cast<Element*>(params.u_out_ptr)),
                make_layout(params.shape_wukg, params.stride_wukg))(_, _, head_idx);

            // ============================================================
            // Per i_k iteration: kg compute, then wait w/u from MMA
            // ============================================================
            CUTE_NO_UNROLL
            for (int i_k = 0; i_k < NumKIters; ++i_k) {
                // Wait for K, G data from TMA Load warp (needed for kg)
                kg_pipeline.consumer_wait(kg_pipe_state_read);
                int buf_idx = kg_pipe_state_read.index();

                Tensor sK = make_tensor(make_smem_ptr(shared_plan->k[buf_idx].data()), SmemLayoutInputBF16{});
                Tensor sG = make_tensor(make_smem_ptr(shared_plan->g[buf_idx].data()), SmemLayoutInputFP32{});

                // ---- kg output: kg = K * exp2(g_last - G) ----
                Tensor sO = make_tensor(make_smem_ptr(shared_plan->out[0].data()), SmemLayoutOutputBF16{});
                // Ensure all 128 epilogue threads have finished writing sO
                cutlass::arch::NamedBarrier::arrive_and_wait(
                    NumEpilogueThreads, KdaChunkFwdRecompWUSm100NamedBarriers::EpilogueCudaCore);
                {
                    auto tKrK = thr_mma_s2r.partition_fragment_A(sK);
                    // S2R g, compute g'=exp2(g_last - g)
                    auto tKrG = make_fragment_like<float>(tKrK);
                    auto tKsG = thr_mma_s2r.partition_A(sG);
                    copy(CopyGAtom{}, tKsG, tKrG);

                    for_each(make_int_sequence<size(tKcMk)>{}, [&](auto i) {
                        auto coord = tKcMk(i);
                        auto [s, t] = coord;
                        auto g = tKrG(i);
                        auto g_last = sG(sub_seq_len - 1, t);
                        if (s < sub_seq_len) {
                            tKrG(i) = exp2f(g_last - g);
                        } else {
                            tKrG(i) = 0.0f;
                        }
                    });
                    // S2R k, compute kg = k*g'
                    auto tKsK = thr_load_k.partition_S(sK);
                    auto tKrK_cv = thr_load_k.retile_D(tKrK);
                    copy(tiled_load_k, tKsK, tKrK_cv);

                    cute::transform(tKrK, tKrG, tKrK, [&](auto k, auto g) { return Element(float(k) * g); });

                    // R2S kg
                    auto tKsK_out = thr_store_k.partition_D(sO);
                    auto tKrK_out_cv = thr_store_k.retile_S(tKrK);
                    copy(tiled_store_k, tKrK_out_cv, tKsK_out);
                }

                // Ensure all 128 epilogue threads have finished writing sO
                cutlass::arch::NamedBarrier::arrive_and_wait(
                    NumEpilogueThreads, KdaChunkFwdRecompWUSm100NamedBarriers::EpilogueCudaCore);
                // if (idx_in_wg == 0) {
                //     printf("sO: exp(g_last-g)*K\n");
                //     cute::print_tensor(sO);
                // }
                // cutlass::arch::NamedBarrier::arrive_and_wait(NumEpilogueThreads,
                // KdaChunkFwdRecompWUSm100NamedBarriers::EpilogueCudaCore);

                // S2G: sO → GMEM kg
                Tensor gKg = local_tile(
                    cute::domain_offset(make_coord(token_offset_cur, _0{}), mKg),
                    Shape<Int<TileT>, Int<TileK>>{},
                    make_coord(_0{}, _0{}));
                Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
                Tensor tOgKg = gmem_thr_copy_O.partition_D(gKg);
                Tensor tOcO = gmem_thr_copy_O.partition_D(make_identity_tensor(Shape<Int<TileT>, Int<TileK>>{}));
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
#pragma unroll
                for (int k = 0; k < size(tOpO); ++k) {
                    tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < params.d;
                }
                ku::copy_pred<
                    /*Is_even_MN=*/false,
                    /*Is_even_K=*/false,
                    /*Clear_OOB_MN=*/false,
                    /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOsO, tOgKg, tOcO, tOpO, sub_seq_len);

                // Release KG SMEM after kg computation
                // ---- w output: wait K-GEMM acc → T2R → bf16 → R2G ----
                // ---- u output: wait V-GEMM acc → T2R → bf16 → R2G ----
                kg_pipeline.consumer_release(kg_pipe_state_read);
                ++kg_pipe_state_read;

                acc_done_pipeline.consumer_wait(acc_done_pipe_state_read);
                int buf_mma_idx = acc_done_pipe_state_read.index();
                // TODO: use other tmem ld to load 64x128 W/U acc, and then direct R2G, remove reg spill
                float res[TileK];
                ku::tcgen05_after_thread_sync();
                ku::tmem_ld_32dp32bNx<TileK>(uint32_t(TmemAllocation::W) + buf_mma_idx * 256, res);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                acc_done_pipeline.consumer_release(acc_done_pipe_state_read);
                ++acc_done_pipe_state_read;

                int acc_idx = (idx_in_wg / 16) & 1;
                __nv_bfloat16* out_ptr_base =
                    reinterpret_cast<__nv_bfloat16*>(acc_idx == 0 ? params.w_out_ptr : params.u_out_ptr);
                // lane: 0-15, 32-47, 64-79, 96-111 stores W acc
                // lane: 16-31, 48-63, 80-95, 112-127 stores U acc
                // TMEM dp-lane to row mapping (4 warps × 32 lanes → 64 rows):
                //   row = (idx_in_wg / 32) * 16 + (idx_in_wg % 16)
                // each thread processes one row of W/U (TileK columns)
                int row = (idx_in_wg / 32) * 16 + (idx_in_wg % 16);

                // GMEM output address: layout [total_len, d, h], stride [d*h, 1, d]
                __nv_bfloat16* out_row_base =
                    out_ptr_base + (token_offset_cur + row) * params.d * params.h + head_idx * params.d;

                // Convert float acc to bf16 and store to GMEM via store_256b
                // store_256b writes 16 bf16 values (256 bits) per call
                // TileK / 16 iterations to cover the full row
                if (row < sub_seq_len) {
#pragma unroll
                    for (int i = 0; i < TileK / 16; ++i) {
                        ku::bf16x16 out;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            reinterpret_cast<__nv_bfloat162*>(&out)[j] =
                                __float22bfloat162_rn(reinterpret_cast<float2*>(&res[i * 16])[j]);
                        }
                        store_256b(&out, out_row_base + i * 16);
                    }
                }
            }
        }
    }

    // ===================================================================
    // MMA warp persistent loop (warp 8, elect_one)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    mma_loop(
        const KDA_fwd_recomp_w_u_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // Load -> MMA pipelines (consumer)
        PipelineA& a_pipeline,
        PipelineStateA& a_pipe_state_read,
        // Prologue -> MMA pipeline (consumer, used for both K and V sequentially)
        PipelinePrologueReady& prologue_ready_pipeline,
        PipelineStatePrologueReady& prologue_ready_pipe_state_read,
        // MMA -> Epilogue pipeline (producer, used for both W and U sequentially)
        PipelineAccDone& acc_done_pipeline,
        PipelineStateAccDone& acc_done_pipe_state_write) {
        // === PERSISTENT MMA LOOP (warp 8, elect_one executes UMMA) ===
        TiledMMA tile_mma_ak = TiledMMA_KDAak{};

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            // int tid = tile_scheduler.get_current_tile_id();
            // auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_seqlens_ptr);

            // ============================================================
            // Once per WU: Wait for Akk in SMEM (from Load warp)
            // ============================================================
            a_pipeline.consumer_wait(a_pipe_state_read);
            int a_idx = a_pipe_state_read.index();
            Tensor sA = make_tensor(make_smem_ptr(shared_plan->akk[a_idx].data()), SmemLayoutInputAkkBF16{});

            // ============================================================
            // Per i_k iteration: K-GEMM then V-GEMM (serial)
            // ============================================================
            CUTE_NO_UNROLL
            for (int i_k = 0; i_k < NumKIters; ++i_k) {
                // ---- K-GEMM: acc = Akk @ K_proc^T → w ----
                prologue_ready_pipeline.consumer_wait(prologue_ready_pipe_state_read);
                fence_view_async_shared();

                acc_done_pipeline.producer_acquire(acc_done_pipe_state_write);
                int buf_mma_idx = acc_done_pipe_state_write.index();
                {
                    Tensor tAK = partition_fragment_C(tile_mma_ak, make_shape(Int<TileT>{}, Int<TileK>{}));
                    tAK.data() = uint32_t(TmemAllocation::W);
                    // k_mma is MMA B-operand: [N=TileK, K=TileT] MN-major
                    Tensor sKmma =
                        make_tensor(make_smem_ptr(shared_plan->k_mma[buf_mma_idx].data()), SmemLayoutMatBBF16{});
                    ku::utcmma_ss(tile_mma_ak, sA, sKmma, tAK, true);
                }

                ku::tcgen05_after_thread_sync();

                {
                    Tensor tAV = partition_fragment_C(tile_mma_ak, make_shape(Int<TileT>{}, Int<TileK>{}));
                    tAV.data() = uint32_t(TmemAllocation::U);
                    // v_mma is MMA B-operand: [N=TileK, K=TileT] MN-major
                    Tensor sVmma =
                        make_tensor(make_smem_ptr(shared_plan->v_mma[buf_mma_idx].data()), SmemLayoutMatBBF16{});
                    ku::utcmma_ss(tile_mma_ak, sA, sVmma, tAV, true);
                }
                acc_done_pipeline.producer_commit(acc_done_pipe_state_write);
                ++acc_done_pipe_state_write;

                prologue_ready_pipeline.consumer_release(prologue_ready_pipe_state_read);
                ++prologue_ready_pipe_state_read;
            }
            a_pipeline.consumer_release(a_pipe_state_read);
            ++a_pipe_state_read;
        }
    }

    // ===================================================================
    // Load warp persistent loop (warp 9, elect_one, TMA producer)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    load_loop(
        const KDA_fwd_recomp_w_u_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // TMA pipelines (producer): Akk, KG (merged K+G), V
        PipelineA& a_pipeline,
        PipelineStateA& a_pipe_state_write,
        PipelineKG& kg_pipeline,
        PipelineStateKG& kg_pipe_state_write,
        PipelineV& v_pipeline,
        PipelineStateV& v_pipe_state_write) {
        if (cute::elect_one_sync()) {
            int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
            int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;

            // === PERSISTENT LOAD LOOP (static scheduling) ===
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

                // Build GMEM tensor views (with domain offset for batch)
                Tensor mK = domain_offset(
                    make_coord(token_offset, _0{}, _0{}), tma_params.tma_k.get_tma_tensor(tma_params.shape_kvg));
                Tensor mV = domain_offset(
                    make_coord(token_offset, _0{}, _0{}), tma_params.tma_v.get_tma_tensor(tma_params.shape_kvg));
                Tensor mG = domain_offset(
                    make_coord(token_offset, _0{}, _0{}), tma_params.tma_g.get_tma_tensor(tma_params.shape_kvg));
                Tensor mA = domain_offset(
                    make_coord(token_offset, _0{}, _0{}), tma_params.tma_akk.get_tma_tensor(tma_params.shape_Akk));

                // ============================================================
                // Once per WU: TMA Akk[BT, BT] → sA
                // ============================================================
                {
                    Tensor sA = make_tensor(
                        make_smem_ptr(shared_plan->akk[a_pipe_state_write.index()].data()), SmemLayoutInputAkkBF16{});
                    Tensor gA = local_tile(
                        mA(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileT>{}), make_coord(tile_idx, _0{}));
                    a_pipeline.producer_acquire(a_pipe_state_write);
                    ku::launch_tma_copy(
                        tma_params.tma_akk, gA, sA, *a_pipeline.producer_get_barrier(a_pipe_state_write));
                    ++a_pipe_state_write;
                }

                // ============================================================
                // Per i_k: TMA K+G, V → sK, sG, sV (double-buffered)
                // K and G share one pipeline barrier (merged KG pipeline)
                // ============================================================
                CUTE_NO_UNROLL
                for (int i_k = 0; i_k < NumKIters; ++i_k) {
                    Tensor sK = make_tensor(
                        make_smem_ptr(shared_plan->k[kg_pipe_state_write.index()].data()), SmemLayoutInputBF16{});
                    Tensor sV = make_tensor(
                        make_smem_ptr(shared_plan->v[v_pipe_state_write.index()].data()), SmemLayoutInputBF16{});
                    Tensor sG = make_tensor(
                        make_smem_ptr(shared_plan->g[kg_pipe_state_write.index()].data()), SmemLayoutInputFP32{});

                    Tensor gK = local_tile(
                        mK(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, i_k));
                    Tensor gV = local_tile(
                        mV(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, i_k));
                    Tensor gG = local_tile(
                        mG(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, i_k));

                    // KG: Load K and G on the same barrier → Compute
                    kg_pipeline.producer_acquire(kg_pipe_state_write);
                    auto& kg_barrier = *kg_pipeline.producer_get_barrier(kg_pipe_state_write);
                    ku::launch_tma_copy(tma_params.tma_k, gK, sK, kg_barrier);
                    ku::launch_tma_copy(tma_params.tma_g, gG, sG, kg_barrier);
                    ++kg_pipe_state_write;

                    // V: Load → Compute
                    v_pipeline.producer_acquire(v_pipe_state_write);
                    ku::launch_tma_copy(tma_params.tma_v, gV, sV, *v_pipeline.producer_get_barrier(v_pipe_state_write));
                    ++v_pipe_state_write;
                }
            }
        }
    }

    // ===================================================================
    // Load aux warp persistent loop (warp 10-11, beta loading)
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    load_aux_loop(
        const KDA_fwd_recomp_w_u_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // Beta pipeline (producer, 1×/WU)
        PipelineBeta& beta_pipeline,
        PipelineStateBeta& beta_pipe_state_write) {
        // === PERSISTENT LOAD AUX LOOP (warp 10-11, 64 threads) ===
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

            // ============================================================
            // Once per WU: beta[0:BT] → sBeta (64 threads, each loads 1 element)
            // ============================================================
            beta_pipeline.producer_acquire(beta_pipe_state_write);
            if (thread_idx < TileT) {
                float beta_val =
                    (thread_idx < sub_seq_len)
                        ? reinterpret_cast<float*>(
                              params.beta_ptr)[(token_offset + tile_idx * TileT + thread_idx) * params.h + head_idx]
                        : float(0);
                shared_plan->beta_smem[beta_pipe_state_write.index()][thread_idx] = beta_val;
            }
            fence_view_async_shared();
            beta_pipeline.producer_commit(beta_pipe_state_write);
            ++beta_pipe_state_write;
        }
    }
};

}  // namespace kda::sm100