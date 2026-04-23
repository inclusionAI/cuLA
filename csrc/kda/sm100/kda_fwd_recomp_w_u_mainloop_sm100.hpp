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
using ku::bf16x8;
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
template <bool StoreQG_ = false, typename ElementBeta_ = float>
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
    static constexpr int StagesQ = 1;

    static constexpr bool StoreQG = StoreQG_;
    using ElementBeta = ElementBeta_;

    // TODO: try optimization with tcgen05.mma.ws
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
    // TMA load -> Compute (K)
    using PipelineK = cutlass::PipelineTmaAsync<StagesLoadStore>;
    // TMA load -> Compute (G)
    using PipelineG = cutlass::PipelineTmaAsync<StagesLoadStore>;
    // TMA load -> Compute (Q, only used when StoreQG=true)
    using PipelineQ = cutlass::PipelineTmaAsync<StagesQ>;
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
        cutlass::NumThreadsPerWarpGroup % kGmemThreadsPerRow == 0,
        "Store thread counts must be a multiple of kGmemThreadsPerRow");
    // Layout of Prologue threads for GMEM store, named GmemLayoutAtom
    using GmemLayoutAtom = Layout<
        Shape<Int<cutlass::NumThreadsPerWarpGroup / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
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
    // Helper: conditionally include Q SMEM buffer (only when StoreQG=true)
    struct QSmemBufferEnabled {
        array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> q[StagesQ];
    };
    struct QSmemBufferDisabled {};  // empty, zero-cost
    using QSmemBuffer = cute::conditional_t<StoreQG, QSmemBufferEnabled, QSmemBufferDisabled>;

    struct SharedMemoryPlan {
        // Akk, single buffer
        array_aligned<bf16, cosize_v<SmemLayoutInputAkkBF16>> akk[StagesA];  // 16KB
        // K, V, G double buffer
        array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> k[StagesLoadStore];   // 32KB
        array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> v[StagesLoadStore];   // 32KB
        array_aligned<float, cosize_v<SmemLayoutInputFP32>> g[StagesLoadStore];  // 64KB
        // Q double buffer (only present when StoreQG=true)
        QSmemBuffer q_buf;
        // MMA B-operand staging: K_proc/V_proc after prologue, [N=TileK, K=TileT] MN-major, double buffer
        array_aligned<bf16, cosize_v<SmemLayoutMatBBF16>> k_mma[StagesMma];  // 16KB
        array_aligned<bf16, cosize_v<SmemLayoutMatBBF16>> v_mma[StagesMma];  // 16KB
        // Epilogue store buffer
        array_aligned<bf16, cosize_v<SmemLayoutOutputBF16>> out[1];  // 16KB

        alignas(16) float beta_smem[StagesA][TileT];
        array_aligned<uint32_t, 1> tmem_start_addr;

        // Pipeline shared storage
        alignas(16) typename PipelineA::SharedStorage pipe_a_storage;
        alignas(16) typename PipelineK::SharedStorage pipe_k_storage;
        alignas(16) typename PipelineG::SharedStorage pipe_g_storage;
        alignas(16) typename PipelineV::SharedStorage pipe_v_storage;
        alignas(16) typename PipelineQ::SharedStorage pipe_q_storage;  // Only meaningful when StoreQG=true
        alignas(16) typename PipelineBeta::SharedStorage pipe_beta_storage;
        alignas(16) typename PipelinePrologueReady::SharedStorage pipe_prologue_ready_storage;
        alignas(16) typename PipelineAccDone::SharedStorage pipe_acc_done_storage;
    };

    // ===================== TMA Params =====================
    template <
        typename ShapeKVG,
        typename ShapeAkk,
        typename TMA_V,
        typename TMA_K,
        typename TMA_G,
        typename TMA_Akk,
        typename TMA_Q = int>
    struct TmaParams {
        ShapeKVG shape_kvg;
        ShapeAkk shape_Akk;
        TMA_V tma_v;
        TMA_K tma_k;
        TMA_G tma_g;
        TMA_Akk tma_akk;
        TMA_Q tma_q{};  // Only meaningful when StoreQG=true; default-initialized
    };

    // ===================== Pipeline State Types =====================
    using PipelineStateA = cutlass::PipelineState<PipelineA::Stages>;
    using PipelineStateK = cutlass::PipelineState<PipelineK::Stages>;
    using PipelineStateG = cutlass::PipelineState<PipelineG::Stages>;
    using PipelineStateV = cutlass::PipelineState<PipelineV::Stages>;
    using PipelineStateQ = cutlass::PipelineState<PipelineQ::Stages>;
    using PipelineStateBeta = cutlass::PipelineState<PipelineBeta::Stages>;
    using PipelineStatePrologueReady = cutlass::PipelineState<PipelinePrologueReady::Stages>;
    using PipelineStateAccDone = cutlass::PipelineState<PipelineAccDone::Stages>;

    // ===================================================================
    // WG0: Prologue persistent loop (warp 0-3, 128 threads, 1 WG)
    // Element-wise K_proc computation → co-signal MMA (with Epilogue), then KG output → GMEM
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    prologue_loop(
        const KDA_fwd_recomp_w_u_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // TMA pipelines (consumer): K, G (separate pipelines)
        PipelineK& k_pipeline,
        PipelineStateK& k_pipe_state_read,
        PipelineG& g_pipeline,
        PipelineStateG& g_pipe_state_read,
        // Beta pipeline (consumer, 1×/WU)
        PipelineBeta& beta_pipeline,
        PipelineStateBeta& beta_pipe_state_read,
        // TMA pipeline (consumer): Q (only used when StoreQG=true)
        PipelineQ& q_pipeline,
        PipelineStateQ& q_pipe_state_read,
        // Prologue -> MMA pipeline (co-producer with Epilogue)
        PipelinePrologueReady& prologue_ready_pipeline,
        PipelineStatePrologueReady& prologue_ready_pipe_state_write) {
        // === PERSISTENT PROLOGUE LOOP (WG0, 128 threads) ===
        int idx_in_wg = threadIdx.x % NumPrologueThreads;  // 0..127
        int* chunk_indices_ptr = (int*)params.chunk_indices_ptr;
        int* cu_seqlens_ptr = (int*)params.cu_seqlens_ptr;

        // Setup autovec GMEM store tiled copy (for KG S2G)
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx_in_wg);

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

            // ============================================================
            // Once per WU: Wait for beta
            // ============================================================
            beta_pipeline.consumer_wait(beta_pipe_state_read);
            fence_view_async_shared();

            // ============================================================
            // Per i_k iteration: K_proc element-wise → co-signal MMA, then KG → GMEM
            // ============================================================
            CUTE_NO_UNROLL
            for (int i_k = 0; i_k < NumKIters; ++i_k) {
                // Wait for K data from TMA Load warp
                k_pipeline.consumer_wait(k_pipe_state_read);

                Tensor sK =
                    make_tensor(make_smem_ptr(shared_plan->k[k_pipe_state_read.index()].data()), SmemLayoutInputBF16{});

                Tensor sKG_out = make_tensor(make_smem_ptr(shared_plan->out[0].data()), SmemLayoutOutputBF16{});

                // ---- Step 1: Load K/G from SMEM to RMEM ----
                // K uses 16x64 thread mapping with bf16x8 (128-bit) loads to reduce bank conflict and uncoalesced
                // shared accesses:
                //   x_local = idx_in_wg / 8  → row within 16-row group (0..15)
                //   k_y_base = idx_in_wg % 8 * 8 → column group base (0,8,16,...,56)
                //   Each thread loads bf16x8 (8 bf16 = 128 bits) per iteration.
                //   Inner loop: TileK/64 = 2 column iterations (stride 64)
                // G uses same 16x64 column mapping as K (k_y_base), but loads two float4 (lo+hi)
                //   per iteration to keep K/G columns aligned for register-only computation.
                int x_local = idx_in_wg / 8;       // 0..15
                int k_y_base = idx_in_wg % 8 * 8;  // 0,8,16,...,56  (16x64 mapping)

                // K registers: [row_iters][col_iters], each bf16x8 = 8 bf16 = 128 bits
                bf16x8 k_reg[TileT / 16][TileK / 64];
                // G registers: [row_iters][col_iters][lo/hi], each float4 = 4 floats
                // g_reg[ti][k_yi][0] = lo half (cols y..y+3), g_reg[ti][k_yi][1] = hi half (cols y+4..y+7)
                float4 g_reg[TileT / 16][TileK / 64][2];

                // Load K with 16x64 mapping using bf16x8 (128-bit) loads
#pragma unroll
                for (int ti = 0; ti < TileT / 16; ++ti) {
                    int t = x_local + ti * 16;
#pragma unroll
                    for (int k_yi = 0; k_yi < TileK / 64; ++k_yi) {
                        int y = k_y_base + k_yi * 64;
                        if (t < sub_seq_len) {
                            k_reg[ti][k_yi] = *reinterpret_cast<bf16x8*>(&sK(t, y));
                        } else {
                            k_reg[ti][k_yi].a01 = __float2bfloat162_rn(0.0f);
                            k_reg[ti][k_yi].a23 = __float2bfloat162_rn(0.0f);
                            k_reg[ti][k_yi].a45 = __float2bfloat162_rn(0.0f);
                            k_reg[ti][k_yi].a67 = __float2bfloat162_rn(0.0f);
                        }
                    }
                }
                // Release K SMEM back to Load warp (done reading K)
                k_pipeline.consumer_release(k_pipe_state_read);
                ++k_pipe_state_read;

                g_pipeline.consumer_wait(g_pipe_state_read);
                Tensor sG =
                    make_tensor(make_smem_ptr(shared_plan->g[g_pipe_state_read.index()].data()), SmemLayoutInputFP32{});
                // Load G with same 16x64 column mapping as K (two float4 per iteration)
#pragma unroll
                for (int ti = 0; ti < TileT / 16; ++ti) {
                    int t = x_local + ti * 16;
#pragma unroll
                    for (int k_yi = 0; k_yi < TileK / 64; ++k_yi) {
                        int y = k_y_base + k_yi * 64;
                        if (t < sub_seq_len) {
                            g_reg[ti][k_yi][0] = *reinterpret_cast<float4*>(&sG(t, y));
                            g_reg[ti][k_yi][1] = *reinterpret_cast<float4*>(&sG(t, y + 4));
                        } else {
                            g_reg[ti][k_yi][0] = {0.0f, 0.0f, 0.0f, 0.0f};
                            g_reg[ti][k_yi][1] = {0.0f, 0.0f, 0.0f, 0.0f};
                        }
                    }
                }

                // ---- Step 2: Compute K_proc = K * beta * exp2(G) → write to k_mma (MN-major) → signal MMA ----
                // Co-acquire prologue_ready with Epilogue (both WGs must acquire before MMA can consume)
                // Deferred to here so that K/G SMEM→RMEM loads can overlap with MMA consuming previous k_mma
                prologue_ready_pipeline.producer_acquire(prologue_ready_pipe_state_write);
                int buf_idx = prologue_ready_pipe_state_write.index();
                Tensor sK_dst = make_tensor(make_smem_ptr(shared_plan->k_mma[buf_idx].data()), SmemLayoutMatBBF16{});

                // K and G registers are column-aligned; compute all 8 bf16 elements and store as bf16x8 (128-bit)
                {
#pragma unroll
                    for (int ti = 0; ti < TileT / 16; ++ti) {
                        int t = x_local + ti * 16;
                        float beta_val = shared_plan->beta_smem[beta_pipe_state_read.index()][t];
                        float2 beta2 = {beta_val, beta_val};

#pragma unroll
                        for (int k_yi = 0; k_yi < TileK / 64; ++k_yi) {
                            int y = k_y_base + k_yi * 64;
                            if (t < sub_seq_len) {
                                // lo half (cols y..y+3): k_reg a01, a23 + g_reg lo
                                float2 kf_01 = __bfloat1622float2(k_reg[ti][k_yi].a01);
                                float2 kf_23 = __bfloat1622float2(k_reg[ti][k_yi].a23);
                                float2 g_01 = {exp2f(g_reg[ti][k_yi][0].x), exp2f(g_reg[ti][k_yi][0].y)};
                                float2 g_23 = {exp2f(g_reg[ti][k_yi][0].z), exp2f(g_reg[ti][k_yi][0].w)};
                                float2 res_01 = float2_mul(float2_mul(kf_01, beta2), g_01);
                                float2 res_23 = float2_mul(float2_mul(kf_23, beta2), g_23);
                                // hi half (cols y+4..y+7): k_reg a45, a67 + g_reg hi
                                float2 kf_45 = __bfloat1622float2(k_reg[ti][k_yi].a45);
                                float2 kf_67 = __bfloat1622float2(k_reg[ti][k_yi].a67);
                                float2 g_45 = {exp2f(g_reg[ti][k_yi][1].x), exp2f(g_reg[ti][k_yi][1].y)};
                                float2 g_67 = {exp2f(g_reg[ti][k_yi][1].z), exp2f(g_reg[ti][k_yi][1].w)};
                                float2 res_45 = float2_mul(float2_mul(kf_45, beta2), g_45);
                                float2 res_67 = float2_mul(float2_mul(kf_67, beta2), g_67);
                                // Single 128-bit store
                                bf16x8 out;
                                out.a01 = __float22bfloat162_rn(res_01);
                                out.a23 = __float22bfloat162_rn(res_23);
                                out.a45 = __float22bfloat162_rn(res_45);
                                out.a67 = __float22bfloat162_rn(res_67);
                                *reinterpret_cast<bf16x8*>(&sK_dst(y, t)) = out;
                            } else {
                                bf16x8 zero;
                                zero.a01 = __float2bfloat162_rn(0.0f);
                                zero.a23 = __float2bfloat162_rn(0.0f);
                                zero.a45 = __float2bfloat162_rn(0.0f);
                                zero.a67 = __float2bfloat162_rn(0.0f);
                                *reinterpret_cast<bf16x8*>(&sK_dst(y, t)) = zero;
                            }
                        }
                    }
                }

                fence_view_async_shared();
                prologue_ready_pipeline.producer_commit(prologue_ready_pipe_state_write);
                ++prologue_ready_pipe_state_write;

                // ---- Step 3: Compute KG = K * exp2(g_last - G) → write to out (K-major) → store to GMEM ----
                // Load g_last from SMEM into registers (only need last valid row)
                float4 g_last_reg[TileK / 64][2];
#pragma unroll
                for (int k_yi = 0; k_yi < TileK / 64; ++k_yi) {
                    int y = k_y_base + k_yi * 64;
                    g_last_reg[k_yi][0] = *reinterpret_cast<float4*>(&sG(sub_seq_len - 1, y));
                    g_last_reg[k_yi][1] = *reinterpret_cast<float4*>(&sG(sub_seq_len - 1, y + 4));
                }

                g_pipeline.consumer_release(g_pipe_state_read);
                ++g_pipe_state_read;

                // Need NamedBarrier to ensure all 128 prologue threads finish previous sKG_out writes
                cutlass::arch::NamedBarrier::arrive_and_wait(
                    NumPrologueThreads, KdaChunkFwdRecompWUSm100NamedBarriers::PrologueCudaCore);

                // Compute KG using k_reg, g_reg, and g_last_reg (all from registers)
                // Compute all 8 bf16 elements and store as bf16x8 (128-bit)
#pragma unroll
                for (int ti = 0; ti < TileT / 16; ++ti) {
                    int t = x_local + ti * 16;
#pragma unroll
                    for (int k_yi = 0; k_yi < TileK / 64; ++k_yi) {
                        int y = k_y_base + k_yi * 64;
                        if (t < sub_seq_len) {
                            // lo half (cols y..y+3): k_reg a01, a23
                            float2 kf_01 = __bfloat1622float2(k_reg[ti][k_yi].a01);
                            float2 kf_23 = __bfloat1622float2(k_reg[ti][k_yi].a23);
                            float2 gd_01 = {
                                exp2f(g_last_reg[k_yi][0].x - g_reg[ti][k_yi][0].x),
                                exp2f(g_last_reg[k_yi][0].y - g_reg[ti][k_yi][0].y)};
                            float2 gd_23 = {
                                exp2f(g_last_reg[k_yi][0].z - g_reg[ti][k_yi][0].z),
                                exp2f(g_last_reg[k_yi][0].w - g_reg[ti][k_yi][0].w)};
                            float2 res_01 = float2_mul(kf_01, gd_01);
                            float2 res_23 = float2_mul(kf_23, gd_23);
                            // hi half (cols y+4..y+7): k_reg a45, a67
                            float2 kf_45 = __bfloat1622float2(k_reg[ti][k_yi].a45);
                            float2 kf_67 = __bfloat1622float2(k_reg[ti][k_yi].a67);
                            float2 gd_45 = {
                                exp2f(g_last_reg[k_yi][1].x - g_reg[ti][k_yi][1].x),
                                exp2f(g_last_reg[k_yi][1].y - g_reg[ti][k_yi][1].y)};
                            float2 gd_67 = {
                                exp2f(g_last_reg[k_yi][1].z - g_reg[ti][k_yi][1].z),
                                exp2f(g_last_reg[k_yi][1].w - g_reg[ti][k_yi][1].w)};
                            float2 res_45 = float2_mul(kf_45, gd_45);
                            float2 res_67 = float2_mul(kf_67, gd_67);
                            // Single 128-bit store
                            bf16x8 out;
                            out.a01 = __float22bfloat162_rn(res_01);
                            out.a23 = __float22bfloat162_rn(res_23);
                            out.a45 = __float22bfloat162_rn(res_45);
                            out.a67 = __float22bfloat162_rn(res_67);
                            *reinterpret_cast<bf16x8*>(&sKG_out(t, y)) = out;
                        } else {
                            bf16x8 zero;
                            zero.a01 = __float2bfloat162_rn(0.0f);
                            zero.a23 = __float2bfloat162_rn(0.0f);
                            zero.a45 = __float2bfloat162_rn(0.0f);
                            zero.a67 = __float2bfloat162_rn(0.0f);
                            *reinterpret_cast<bf16x8*>(&sKG_out(t, y)) = zero;
                        }
                    }
                }

                // Ensure all 128 prologue threads have finished writing sKG_out
                cutlass::arch::NamedBarrier::arrive_and_wait(
                    NumPrologueThreads, KdaChunkFwdRecompWUSm100NamedBarriers::PrologueCudaCore);

                // S2G: sKG_out → GMEM kg via GmemTiledCopyO + copy_pred
                {
                    Tensor mKg = make_tensor(
                        make_gmem_ptr(reinterpret_cast<Element*>(params.kg_out_ptr)),
                        make_layout(params.shape_wukg, params.stride_wukg))(_, _, head_idx);
                    Tensor gKg = local_tile(
                        cute::domain_offset(make_coord(token_offset_cur, _0{}), mKg),
                        Shape<Int<TileT>, Int<TileK>>{},
                        make_coord(_0{}, _0{}));
                    Tensor tOsO = gmem_thr_copy_O.partition_S(sKG_out);
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
                }

                // ---- Step 4 (StoreQG only): Compute QG = Q * exp2(G) → write to out → store to GMEM ----
                if constexpr (StoreQG) {
                    // Wait for Q data from TMA Load warp
                    q_pipeline.consumer_wait(q_pipe_state_read);
                    Tensor sQ = make_tensor(
                        make_smem_ptr(shared_plan->q_buf.q[q_pipe_state_read.index()].data()), SmemLayoutInputBF16{});

                    // Load Q with same 16x64 mapping as K
                    bf16x8 q_reg[TileT / 16][TileK / 64];
#pragma unroll
                    for (int ti = 0; ti < TileT / 16; ++ti) {
                        int t = x_local + ti * 16;
#pragma unroll
                        for (int k_yi = 0; k_yi < TileK / 64; ++k_yi) {
                            int y = k_y_base + k_yi * 64;
                            if (t < sub_seq_len) {
                                q_reg[ti][k_yi] = *reinterpret_cast<bf16x8*>(&sQ(t, y));
                            } else {
                                q_reg[ti][k_yi].a01 = __float2bfloat162_rn(0.0f);
                                q_reg[ti][k_yi].a23 = __float2bfloat162_rn(0.0f);
                                q_reg[ti][k_yi].a45 = __float2bfloat162_rn(0.0f);
                                q_reg[ti][k_yi].a67 = __float2bfloat162_rn(0.0f);
                            }
                        }
                    }
                    // Release Q SMEM back to Load warp
                    q_pipeline.consumer_release(q_pipe_state_read);
                    ++q_pipe_state_read;

                    // Need NamedBarrier to ensure all 128 prologue threads finish previous sKG_out writes (from KG
                    // store)
                    cutlass::arch::NamedBarrier::arrive_and_wait(
                        NumPrologueThreads, KdaChunkFwdRecompWUSm100NamedBarriers::PrologueCudaCore);

                    // Compute QG = Q * exp2(G) and write to sKG_out (reuse output buffer)
#pragma unroll
                    for (int ti = 0; ti < TileT / 16; ++ti) {
                        int t = x_local + ti * 16;
#pragma unroll
                        for (int k_yi = 0; k_yi < TileK / 64; ++k_yi) {
                            int y = k_y_base + k_yi * 64;
                            if (t < sub_seq_len) {
                                // lo half (cols y..y+3)
                                float2 qf_01 = __bfloat1622float2(q_reg[ti][k_yi].a01);
                                float2 qf_23 = __bfloat1622float2(q_reg[ti][k_yi].a23);
                                float2 g_01 = {exp2f(g_reg[ti][k_yi][0].x), exp2f(g_reg[ti][k_yi][0].y)};
                                float2 g_23 = {exp2f(g_reg[ti][k_yi][0].z), exp2f(g_reg[ti][k_yi][0].w)};
                                float2 res_01 = float2_mul(qf_01, g_01);
                                float2 res_23 = float2_mul(qf_23, g_23);
                                // hi half (cols y+4..y+7)
                                float2 qf_45 = __bfloat1622float2(q_reg[ti][k_yi].a45);
                                float2 qf_67 = __bfloat1622float2(q_reg[ti][k_yi].a67);
                                float2 g_45 = {exp2f(g_reg[ti][k_yi][1].x), exp2f(g_reg[ti][k_yi][1].y)};
                                float2 g_67 = {exp2f(g_reg[ti][k_yi][1].z), exp2f(g_reg[ti][k_yi][1].w)};
                                float2 res_45 = float2_mul(qf_45, g_45);
                                float2 res_67 = float2_mul(qf_67, g_67);
                                // Single 128-bit store
                                bf16x8 out;
                                out.a01 = __float22bfloat162_rn(res_01);
                                out.a23 = __float22bfloat162_rn(res_23);
                                out.a45 = __float22bfloat162_rn(res_45);
                                out.a67 = __float22bfloat162_rn(res_67);
                                *reinterpret_cast<bf16x8*>(&sKG_out(t, y)) = out;
                            } else {
                                bf16x8 zero;
                                zero.a01 = __float2bfloat162_rn(0.0f);
                                zero.a23 = __float2bfloat162_rn(0.0f);
                                zero.a45 = __float2bfloat162_rn(0.0f);
                                zero.a67 = __float2bfloat162_rn(0.0f);
                                *reinterpret_cast<bf16x8*>(&sKG_out(t, y)) = zero;
                            }
                        }
                    }

                    // Ensure all 128 prologue threads have finished writing QG to sKG_out
                    cutlass::arch::NamedBarrier::arrive_and_wait(
                        NumPrologueThreads, KdaChunkFwdRecompWUSm100NamedBarriers::PrologueCudaCore);

                    // S2G: sKG_out → GMEM qg via GmemTiledCopyO + copy_pred
                    {
                        Tensor mQg = make_tensor(
                            make_gmem_ptr(reinterpret_cast<Element*>(params.qg_out_ptr)),
                            make_layout(params.shape_wukg, params.stride_wukg))(_, _, head_idx);
                        Tensor gQg = local_tile(
                            cute::domain_offset(make_coord(token_offset_cur, _0{}), mQg),
                            Shape<Int<TileT>, Int<TileK>>{},
                            make_coord(_0{}, _0{}));
                        Tensor tOsO_qg = gmem_thr_copy_O.partition_S(sKG_out);
                        Tensor tOgQg = gmem_thr_copy_O.partition_D(gQg);
                        Tensor tOcO_qg =
                            gmem_thr_copy_O.partition_D(make_identity_tensor(Shape<Int<TileT>, Int<TileK>>{}));
                        Tensor tOpO_qg = make_tensor<bool>(make_shape(size<2>(tOcO_qg)));
#pragma unroll
                        for (int k = 0; k < size(tOpO_qg); ++k) {
                            tOpO_qg(k) = get<1>(tOcO_qg(_0{}, _0{}, k)) < params.d;
                        }
                        ku::copy_pred<
                            /*Is_even_MN=*/false,
                            /*Is_even_K=*/false,
                            /*Clear_OOB_MN=*/false,
                            /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOsO_qg, tOgQg, tOcO_qg, tOpO_qg, sub_seq_len);
                    }
                }  // end if constexpr (StoreQG)
            }

            // Release beta at end of WU
            beta_pipeline.consumer_release(beta_pipe_state_read);
            ++beta_pipe_state_read;
        }
    }

    // ===================================================================
    // WG1: Epilogue persistent loop (warp 4-7, 128 threads, 1 WG)
    // V_proc computation → co-signal MMA (with Prologue), then W/U store → GMEM
    // ===================================================================
    template <typename TmaParamsT>
    CUTLASS_DEVICE void
    epilogue_loop(
        const KDA_fwd_recomp_w_u_params& params,
        const TmaParamsT& tma_params,
        SharedMemoryPlan* shared_plan,
        TileScheduler& tile_scheduler,
        // TMA pipeline (consumer): V
        PipelineV& v_pipeline,
        PipelineStateV& v_pipe_state_read,
        // Beta pipeline (consumer, 1×/WU)
        PipelineBeta& beta_pipeline,
        PipelineStateBeta& beta_pipe_state_read,
        // Prologue -> MMA pipeline (co-producer with Prologue)
        PipelinePrologueReady& prologue_ready_pipeline,
        PipelineStatePrologueReady& prologue_ready_pipe_state_write,
        // MMA -> Epilogue pipeline (consumer, used for both W and U sequentially)
        PipelineAccDone& acc_done_pipeline,
        PipelineStateAccDone& acc_done_pipe_state_read) {
        // === PERSISTENT EPILOGUE LOOP (WG1, 128 threads) ===
        // idx_in_wg: 0..127 within this warp group
        int idx_in_wg = threadIdx.x % NumEpilogueThreads;  // 0..127
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
            int token_offset_cur = token_offset + tile_idx * TileT;

            // ============================================================
            // Once per WU: Wait for beta
            // ============================================================
            beta_pipeline.consumer_wait(beta_pipe_state_read);
            fence_view_async_shared();

            // ============================================================
            // Per i_k iteration: V_proc → co-signal MMA, then wait w/u → store to GMEM
            // ============================================================
            CUTE_NO_UNROLL
            for (int i_k = 0; i_k < NumKIters; ++i_k) {
                // Wait for V data from TMA Load warp
                v_pipeline.consumer_wait(v_pipe_state_read);
                Tensor sV =
                    make_tensor(make_smem_ptr(shared_plan->v[v_pipe_state_read.index()].data()), SmemLayoutInputBF16{});

                // Co-acquire prologue_ready with Prologue (both WGs must acquire before MMA can consume)
                prologue_ready_pipeline.producer_acquire(prologue_ready_pipe_state_write);
                int buf_idx = prologue_ready_pipe_state_write.index();

                // ---- V_proc: V * beta → R2S → v_mma (MN-major) ----
                // Uses 16x64 thread mapping with bf16x8 (128-bit) loads/stores, same as K
                Tensor sV_dst = make_tensor(make_smem_ptr(shared_plan->v_mma[buf_idx].data()), SmemLayoutMatBBF16{});
                {
                    int x_local = idx_in_wg / 8;     // 0..15
                    int y_base = idx_in_wg % 8 * 8;  // 0,8,16,...,56  (16x64 mapping)

#pragma unroll
                    for (int t_iter = 0; t_iter < TileT; t_iter += 16) {
                        int t = x_local + t_iter;
                        float beta_val = shared_plan->beta_smem[beta_pipe_state_read.index()][t];
                        float2 beta2 = {beta_val, beta_val};

#pragma unroll
                        for (int y_iter = 0; y_iter < TileK; y_iter += 64) {
                            int y = y_base + y_iter;
                            if (t < sub_seq_len) {
                                bf16x8 v = *reinterpret_cast<bf16x8*>(&sV(t, y));
                                // lo half (cols y..y+3)
                                float2 vf_01 = __bfloat1622float2(v.a01);
                                float2 vf_23 = __bfloat1622float2(v.a23);
                                float2 res_01 = float2_mul(vf_01, beta2);
                                float2 res_23 = float2_mul(vf_23, beta2);
                                // hi half (cols y+4..y+7)
                                float2 vf_45 = __bfloat1622float2(v.a45);
                                float2 vf_67 = __bfloat1622float2(v.a67);
                                float2 res_45 = float2_mul(vf_45, beta2);
                                float2 res_67 = float2_mul(vf_67, beta2);
                                // Single 128-bit store
                                bf16x8 out;
                                out.a01 = __float22bfloat162_rn(res_01);
                                out.a23 = __float22bfloat162_rn(res_23);
                                out.a45 = __float22bfloat162_rn(res_45);
                                out.a67 = __float22bfloat162_rn(res_67);
                                *reinterpret_cast<bf16x8*>(&sV_dst(y, t)) = out;
                            } else {
                                bf16x8 zero;
                                zero.a01 = __float2bfloat162_rn(0.0f);
                                zero.a23 = __float2bfloat162_rn(0.0f);
                                zero.a45 = __float2bfloat162_rn(0.0f);
                                zero.a67 = __float2bfloat162_rn(0.0f);
                                *reinterpret_cast<bf16x8*>(&sV_dst(y, t)) = zero;
                            }
                        }
                    }
                }

                // Release V SMEM back to Load warp (done reading sV, all in sV_dst now)
                v_pipeline.consumer_release(v_pipe_state_read);
                ++v_pipe_state_read;

                // Co-commit prologue_ready with Prologue → MMA can now consume k_mma + v_mma
                fence_view_async_shared();
                prologue_ready_pipeline.producer_commit(prologue_ready_pipe_state_write);
                ++prologue_ready_pipe_state_write;

                // ---- w/u output: wait K-GEMM & V-GEMM acc → T2R → bf16 → R2G ----
                // Split into 2 iterations of TileK/2 to reduce register pressure (avoid spill)
                acc_done_pipeline.consumer_wait(acc_done_pipe_state_read);
                int buf_mma_idx = acc_done_pipe_state_read.index();

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

                constexpr int QuarK = TileK / 4;

                ku::tcgen05_after_thread_sync();
#pragma unroll
                for (int quar = 0; quar < 4; ++quar) {
                    float res_quar[QuarK];
                    ku::tmem_ld_32dp32bNx<QuarK>(
                        uint32_t(TmemAllocation::W) + buf_mma_idx * 256 + quar * QuarK, res_quar);
                    cutlass::arch::fence_view_async_tmem_load();

                    if (row < sub_seq_len) {
#pragma unroll
                        for (int i = 0; i < QuarK / 16; ++i) {
                            ku::bf16x16 out;
#pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                reinterpret_cast<__nv_bfloat162*>(&out)[j] =
                                    __float22bfloat162_rn(reinterpret_cast<float2*>(&res_quar[i * 16])[j]);
                            }
                            store_256b(&out, out_row_base + quar * QuarK + i * 16);
                        }
                    }
                }
                ku::tcgen05_before_thread_sync();

                acc_done_pipeline.consumer_release(acc_done_pipe_state_read);
                ++acc_done_pipe_state_read;
            }

            // Release beta at end of WU
            beta_pipeline.consumer_release(beta_pipe_state_read);
            ++beta_pipe_state_read;
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
        // TMA pipelines (producer): Akk, K, G, V, Q (Q only used when StoreQG=true)
        PipelineA& a_pipeline,
        PipelineStateA& a_pipe_state_write,
        PipelineK& k_pipeline,
        PipelineStateK& k_pipe_state_write,
        PipelineG& g_pipeline,
        PipelineStateG& g_pipe_state_write,
        PipelineV& v_pipeline,
        PipelineStateV& v_pipe_state_write,
        PipelineQ& q_pipeline,
        PipelineStateQ& q_pipe_state_write) {
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

                // Q GMEM tensor (only used when StoreQG=true)
                [[maybe_unused]] auto mQ = [&]() {
                    if constexpr (StoreQG) {
                        return domain_offset(
                            make_coord(token_offset, _0{}, _0{}),
                            tma_params.tma_q.get_tma_tensor(tma_params.shape_kvg));
                    } else {
                        return 0;  // unused placeholder
                    }
                }();

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
                // Per i_k: TMA K, G, V → sK, sG, sV (double-buffered)
                // K and G have separate pipelines
                // ============================================================
                CUTE_NO_UNROLL
                for (int i_k = 0; i_k < NumKIters; ++i_k) {
                    Tensor sK = make_tensor(
                        make_smem_ptr(shared_plan->k[k_pipe_state_write.index()].data()), SmemLayoutInputBF16{});
                    Tensor sV = make_tensor(
                        make_smem_ptr(shared_plan->v[v_pipe_state_write.index()].data()), SmemLayoutInputBF16{});
                    Tensor sG = make_tensor(
                        make_smem_ptr(shared_plan->g[g_pipe_state_write.index()].data()), SmemLayoutInputFP32{});

                    Tensor gK = local_tile(
                        mK(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, i_k));
                    Tensor gV = local_tile(
                        mV(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, i_k));
                    Tensor gG = local_tile(
                        mG(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, i_k));

                    // K: Load → Compute
                    k_pipeline.producer_acquire(k_pipe_state_write);
                    ku::launch_tma_copy(tma_params.tma_k, gK, sK, *k_pipeline.producer_get_barrier(k_pipe_state_write));
                    ++k_pipe_state_write;

                    // G: Load → Compute
                    g_pipeline.producer_acquire(g_pipe_state_write);
                    ku::launch_tma_copy(tma_params.tma_g, gG, sG, *g_pipeline.producer_get_barrier(g_pipe_state_write));
                    ++g_pipe_state_write;

                    // V: Load → Compute
                    v_pipeline.producer_acquire(v_pipe_state_write);
                    ku::launch_tma_copy(tma_params.tma_v, gV, sV, *v_pipeline.producer_get_barrier(v_pipe_state_write));
                    ++v_pipe_state_write;

                    // Q: Load → Compute (only when StoreQG=true)
                    if constexpr (StoreQG) {
                        Tensor sQ = make_tensor(
                            make_smem_ptr(shared_plan->q_buf.q[q_pipe_state_write.index()].data()),
                            SmemLayoutInputBF16{});
                        Tensor gQ = local_tile(
                            mQ(_, _, head_idx), make_shape(Int<TileT>{}, Int<TileK>{}), make_coord(tile_idx, i_k));
                        q_pipeline.producer_acquire(q_pipe_state_write);
                        ku::launch_tma_copy(
                            tma_params.tma_q, gQ, sQ, *q_pipeline.producer_get_barrier(q_pipe_state_write));
                        ++q_pipe_state_write;
                    }
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
                        ? float(reinterpret_cast<ElementBeta*>(
                              params.beta_ptr)[(token_offset + tile_idx * TileT + thread_idx) * params.h + head_idx])
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