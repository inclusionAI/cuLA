#include "kda_bwd_intra_sm100.cuh"
#include "kda_bwd/helpers.h"
#include "kda_bwd/gemm.h"
#include "kda_bwd/utils.h"
#include "kda_bwd/util_func.h"

#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

namespace sm100 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using namespace cute;

// ===================== NaN DEBUG UTILITIES =====================
// Only check blockIdx.x == 0 to limit output volume
#define NAN_DEBUG_ENABLED 0

#if NAN_DEBUG_ENABLED
__device__ inline bool check_nan_array(const float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        if (__isnanf(arr[i])) return true;
    }
    return false;
}

__device__ inline bool check_nan_inf_array(const float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        if (__isnanf(arr[i]) || __isinff(arr[i])) return true;
    }
    return false;
}

// Print first NaN/Inf location in an array
__device__ inline void print_nan_detail(const char* name, const float* arr, int size, int idx_in_wg, int k_idx) {
    for (int i = 0; i < size; ++i) {
        if (__isnanf(arr[i])) {
            printf("[NaN] %s[%d]=NaN thread=%d k_idx=%d blk=%d\n", name, i, idx_in_wg, k_idx, blockIdx.x);
            return;
        }
        if (__isinff(arr[i])) {
            printf("[Inf] %s[%d]=Inf thread=%d k_idx=%d blk=%d\n", name, i, idx_in_wg, k_idx, blockIdx.x);
            return;
        }
    }
}

#define DEBUG_CHECK_NAN(name, arr, size, idx_in_wg, k_idx) \
    do { \
        if (blockIdx.x == 0 && check_nan_inf_array((const float*)(arr), (size))) { \
            print_nan_detail(name, (const float*)(arr), (size), idx_in_wg, k_idx); \
        } \
    } while(0)

// Check smem float tensor for NaN (one thread checks its row)
#define DEBUG_CHECK_SMEM_ROW(name, tensor, row, ncols, idx_in_wg, k_idx) \
    do { \
        if (blockIdx.x == 0) { \
            for (int _c = 0; _c < (ncols); ++_c) { \
                float _v = (tensor)((row), _c); \
                if (__isnanf(_v) || __isinff(_v)) { \
                    printf("[NaN/Inf] %s(%d,%d)=%f thread=%d k_idx=%d blk=%d\n", \
                           name, (row), _c, _v, idx_in_wg, k_idx, blockIdx.x); \
                    break; \
                } \
            } \
        } \
    } while(0)
#else
#define DEBUG_CHECK_NAN(name, arr, size, idx_in_wg, k_idx)
#define DEBUG_CHECK_SMEM_ROW(name, tensor, row, ncols, idx_in_wg, k_idx)
#endif
// ===================== END NaN DEBUG =====================

template<
typename ShapeQKG, typename ShapeDA, 
typename TMA_Q,
typename TMA_K,
typename TMA_G,
typename TMA_DAqk,
typename TMA_DAkk,
typename TMA_DQ,
typename TMA_DK,
typename TMA_DG>
struct TmaParams{
    ShapeQKG shape_qkg;
    ShapeDA shape_da;
    TMA_Q tma_q;
    TMA_K tma_k;
    TMA_G tma_g;
    TMA_DAqk tma_dAqk;
    TMA_DAkk tma_dAkk;
    TMA_DQ tma_dq;
    TMA_DK tma_dk;
    TMA_DG tma_dg;
};

constexpr int SUB_T_TILE = 16;
constexpr int T_TILE = 64;
constexpr int K_SIZE = 128;
constexpr int K_TILE = 32;
constexpr int K_ITERATION = K_SIZE / K_TILE;
constexpr int NUM_BUF_A = 1;
constexpr int NUM_BUF_VALUE = 2;
constexpr int NUM_THREADS = 128 * 3;
constexpr int CHUNK_SIZE = 64;
constexpr int REG_COMPUTE = 200;
constexpr int REG_LOAD = 96;

namespace tmem_addr {
    constexpr int dq = 0; //[0, 32] [256, 288]
    constexpr int dq_02 = dq; //[0, 32] [256, 288]
    constexpr int dq_13 = dq_02 + 16 * 65536; //[0, 32] [256, 288] (+lane16 offset) 
    constexpr int dq2 = dq + K_TILE; //[32, 64], [288, 320]
    constexpr int dq2_02 = dq_02 + K_TILE; //[32, 64] [288, 320]
    constexpr int dq2_13 = dq_13 + K_TILE; //[32, 64] [288, 320] (+lane16 offset)
    constexpr int dkt = dq2 + K_TILE; //[64, 96] [320, 352]
    constexpr int dAqk = dkt + K_TILE; //[96, 160]
    constexpr int dAqk_02 = dAqk; //[96, 160] 
    constexpr int dAqk_13 = dAqk_02 + 16 * 65536; //[96, 160]
    constexpr int dAqk_t = dAqk + 256; // Aqk_t: [352, 368], [384, 400], [416, 432], [448, 464]  Akk_t:[368, 384], [400, 416], [432, 448], [464, 480] 
    constexpr int dAqk_t_02 = dAqk_t; // Aqk_t: [352, 368], [384, 400], [416, 432], [448, 464]  Akk_t:[368, 384], [400, 416], [432, 448], [464, 480] 
    constexpr int dAqk_t_13 = dAqk_t + 16 * 65536; // lane_16 
};

enum class WarpRole {
    Empty = 0x0, Load = 0x1, Mma = 0x2, Compute = 0x3, Epilogue = 0x4,
    ComputeEpilogue = 0x5
};

static constexpr unsigned long long kWarpAssignment = 0x12'5555'5555ull;

__forceinline__ __device__ WarpRole warp_idx_to_role(int warp_idx) {
    return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
}

using SmemLayoutInputBF16 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<T_TILE>, Int<K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutMatBTF32Tranposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_32B_Atom<tf32>{},
    Shape<Int<K_TILE>, Int<SUB_T_TILE * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutInputFP32 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<T_TILE>, Int<K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutInputTF32tmp = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<tf32>{},
    Shape<Int<K_TILE>, Int<T_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutDA_SUB = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<float>{},
    Shape<Int<T_TILE>, Int<16 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutOutputBF16 = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{}, 
    Shape<Int<T_TILE>, Int<K_TILE>>{}
));

using SmemLayoutOutputTF32 = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<tf32>{}, 
    Shape<Int<T_TILE>, Int<K_TILE>>{}
));

using SmemLayoutBeta = Layout<
    Shape<Int<K_TILE>>,
    Stride<_1>
>;

using SmemLayoutDB = Layout<
    Shape<Int<T_TILE>>,
    Stride<_1>
>;

using SmemLayoutDA = SmemLayoutDA_SUB<4>;


struct SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> q[NUM_BUF_VALUE];
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> k[NUM_BUF_VALUE];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> g[NUM_BUF_VALUE];
    struct {
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<1>>> intra[6];
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<1>>> inter[4];
    } kg_all;  // 20480 bytes, single-buffered
    struct {
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<2>>> intra[6];
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32Tranposed<2>>> inter[4];
    } qkg_all; // 40960 bytes, single-buffered
    array_aligned<float, cosize_v<SmemLayoutDA>> dAqk[NUM_BUF_A];
    array_aligned<float, cosize_v<SmemLayoutDA>> dAkk[NUM_BUF_A];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> dq[NUM_BUF_VALUE];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> dk[NUM_BUF_VALUE];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> dg[NUM_BUF_VALUE];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> b_k_exp[NUM_BUF_VALUE];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> b_k_neg_exp[NUM_BUF_VALUE];
    // array_aligned<float, cosize_v<SmemLayoutBeta>> beta[NUM_BUF_VALUE];
    // array_aligned<float, cosize_v<SmemLayoutDB>> db;
    alignas(16) cute::uint64_t bar_load_kg_ready[NUM_BUF_VALUE], bar_load_dA_ready[NUM_BUF_A], bar_load_qb[NUM_BUF_VALUE], bar_load_dkg_ready[NUM_BUF_VALUE]; //load
    alignas(16) cute::uint64_t bar_kg_all_ready, bar_qkg_all_ready;     // CE → MMA: B-matrix data ready (phase-tracked with b_phase)
    alignas(16) cute::uint64_t bar_dq_done, bar_dkt_done;               // MMA → CE: tmem results ready (phase-tracked with b_phase)
    alignas(16) cute::uint64_t bar_dA_ready[NUM_BUF_A], bar_dAt_ready[NUM_BUF_A], bar_dA_free[NUM_BUF_A], bar_dAt_free[NUM_BUF_A], bar_dvalue_free[NUM_BUF_VALUE]; //epilogue
    alignas(16) cute::uint64_t bar_dA_mask_ready[NUM_BUF_A];

    alignas(16) __nv_bfloat16 beta_smem[2][T_TILE]; // double-buffered per-tile beta, indexed by A_phase
    int tile_id[2]; // double-buffered persistent tile ID (written by Load warp, read by all)
    array_aligned<float, 64> db_partial[2];
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TiledMMA_KDAqk_MASK02 = decltype(make_tiled_mma(
    SM100_MMA_TF32_TS_MASK02_NOELECT<tf32, tf32, float, T_TILE, K_TILE, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_KDAqk_MASK13 = decltype(make_tiled_mma(
    SM100_MMA_TF32_TS_MASK13_NOELECT<tf32, tf32, float, T_TILE, K_TILE, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_KDAqk_MASK0 = decltype(make_tiled_mma(
    SM100_MMA_TF32_TS_MASK0_NOELECT<tf32, tf32, float, T_TILE, K_TILE, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_KDAqk_MASK1 = decltype(make_tiled_mma(
    SM100_MMA_TF32_TS_MASK1_NOELECT<tf32, tf32, float, T_TILE, K_TILE, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_KDAqk_MASK2 = decltype(make_tiled_mma(
    SM100_MMA_TF32_TS_MASK2_NOELECT<tf32, tf32, float, T_TILE, K_TILE, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_KDAqk_MASK3 = decltype(make_tiled_mma(
    SM100_MMA_TF32_TS_MASK3_NOELECT<tf32, tf32, float, T_TILE, K_TILE, UMMA::Major::K, UMMA::Major::MN>{}
));

using TileScheduler = NaiveTileScheduler;

template<int OFFSET, int TILE_SIZE = 64>
__forceinline__ __device__ void mask_A_tensor(float* smem_ptr, int idx_in_warpgroup, int sub_len, int tmem_addr) {
    float res[TILE_SIZE];
    Tensor s = make_tensor(make_smem_ptr(smem_ptr), SmemLayoutDA{});
    int x = idx_in_warpgroup % 64;
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / 4; ++i) {
        int y = i * 4 + OFFSET;
        
        reinterpret_cast<float4*>(&res[i*4])[0] = *reinterpret_cast<float4*>(&s(x, y));

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (x >= sub_len || x < y + j || y + j >= sub_len) {
                res[i*4 + j] = 0.0f;
            }
        }
    }
    tmem_st_32dp32bNx<TILE_SIZE>(tmem_addr + OFFSET, res);   
}

template<int OFFSET = 0, int TILE_SIZE = 128>
__forceinline__ __device__ void mask_At_tensor(float* smem_ptr_1, float* smem_ptr_2, int idx_in_warpgroup, int sub_len, int tmem_addr) {
    float res[32];
    Tensor s_1 = make_tensor(make_smem_ptr(smem_ptr_1), SmemLayoutDA{});
    Tensor s_2 = make_tensor(make_smem_ptr(smem_ptr_2), SmemLayoutDA{});
    int x = idx_in_warpgroup % 64;
    // #pragma unroll
    for (int stage = 0; stage < TILE_SIZE / 32; ++stage) {
        for (int i = 0; i < 16; ++i) {
            int y = i + 16 * stage + OFFSET / 2;

            res[i] = s_1(y, x);
            res[i + 16] = s_2(y, x);
            if (x >= sub_len || x > y || y >= sub_len) {
                res[i] = 0.0f;
                res[i + 16] = 0.0f;
            }
        }
        tmem_st_32dp32bNx<32>(tmem_addr + stage * 32 + OFFSET, res);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

template <int WG_IDX>
__forceinline__ __device__ void compute_epilogue_body(
    SharedMemoryPlan *shared_plan,
    const KDA_bwd_intra_params &params,
    int idx_in_warpgroup,
    int &state_phase, int &buf_idx_A, int &buf_idx_value,
    int batch_idx, int head_idx, int tile_idx,
    int start_offset, int sub_seq_len, int tile_phase, int beta_buf) {

    constexpr int HALF_K = K_TILE / 2;
    constexpr int K_OFF = WG_IDX * HALF_K;
    constexpr int DKT_BAR_ID = WG_IDX * 2;

    int offset = start_offset;
    int local_idx = idx_in_warpgroup % 64;
    int b_phase = 0; // B-matrix phase tracking

    // Only WG1 loads db_init from global; WG0 starts at 0 to avoid double-counting after reduce
    float db = 0.0f;
    if constexpr (WG_IDX == 1) {
        db = (idx_in_warpgroup >= 64 && local_idx < sub_seq_len)
        ? reinterpret_cast<float*>(params.db_ptr)[(start_offset + tile_idx * T_TILE + local_idx) * params.h + head_idx]
        : 0.0f;
    } 

    // [DEBUG-INPUT] Check dAqk and dAkk loaded from TMA
    {
        float *dA_ptr = idx_in_warpgroup < 64 ? shared_plan->dAqk[buf_idx_A].data() : shared_plan->dAkk[buf_idx_A].data();
        Tensor s_dA = make_tensor(make_smem_ptr(dA_ptr), SmemLayoutDA{});
        int dA_row = idx_in_warpgroup % 64;
        DEBUG_CHECK_SMEM_ROW("INPUT:dA", s_dA, dA_row, 64, idx_in_warpgroup, -1);
    }

    // Pre-loop: mask_A — dA already loaded by Load warp (bar_load_dA_ready already waited by caller)
    mask_A_tensor<WG_IDX * 32, 32>(idx_in_warpgroup < 64 ? shared_plan->dAqk[buf_idx_A].data() : shared_plan->dAkk[buf_idx_A].data(), idx_in_warpgroup, sub_seq_len, tmem_addr::dAqk);
    cutlass::arch::fence_view_async_tmem_store();
    tcgen05_before_thread_sync();
    cute::arrive_barrier(shared_plan->bar_dA_ready[buf_idx_A]);

    // Wait for beta_smem (loaded by Empty warp)
    cute::wait_barrier(shared_plan->bar_dA_mask_ready[0], tile_phase);

    for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
        int local_phase = (state_phase >> buf_idx_value) & 1;

        // === Protect B-matrix buffers: wait for MMA to finish reading previous qkg_all ===
        // cute::wait_barrier(shared_plan->bar_dkt_done, b_phase ^ 1);

        // === Wait for K/G data ===
        cute::wait_barrier(shared_plan->bar_load_kg_ready[buf_idx_value], local_phase);

        Tensor sK = make_tensor(make_smem_ptr(shared_plan->k[buf_idx_value].data()), SmemLayoutInputBF16{});
        Tensor sG = make_tensor(make_smem_ptr(shared_plan->g[buf_idx_value].data()), SmemLayoutInputFP32{});

        // [DEBUG-INPUT-G] Check sG for NaN (float tensor, easy to check)
        DEBUG_CHECK_SMEM_ROW("INPUT:sG", sG, idx_in_warpgroup % 64, K_TILE, idx_in_warpgroup, k_idx);

        int y = idx_in_warpgroup % 8 * 4;
        constexpr int kg_offset = SUB_T_TILE * K_TILE;

        // === COMPUTE: kg_intra (non-overlapping rows only, before Q is needed) ===
        Tensor sKG_intra = make_tensor(make_smem_ptr(shared_plan->kg_all.intra[0].data()), SmemLayoutMatBTF32Tranposed<1>{});
        Tensor sQKG_intra = make_tensor(make_smem_ptr(shared_plan->qkg_all.intra[0].data()), SmemLayoutMatBTF32Tranposed<2>{});
        constexpr int qkg_offset = SUB_T_TILE * K_TILE * 2;
        if constexpr (WG_IDX == 0) {
            float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
            // tile_j=0 unique to kg (no qkg match)
            setup_kg_intra<decltype(sG), decltype(sK), decltype(sKG_intra), kg_offset>(sG, sK, sKG_intra, 0, idx_in_warpgroup, gn3, 3);
        } else {
            float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
            float4 gn2 = *reinterpret_cast<float4*>(&sG(32, y));
            // tile_j=0 with 2 gn values: fuse into 1 call (shared row idx/8)
            setup_kg_intra_2gn<decltype(sG), decltype(sK), decltype(sKG_intra), kg_offset>(sG, sK, sKG_intra, 0, idx_in_warpgroup, gn1, gn2, 0, 1);
            // tile_j=1 unique to kg
            setup_kg_intra<decltype(sG), decltype(sK), decltype(sKG_intra), kg_offset>(sG, sK, sKG_intra, 1, idx_in_warpgroup, gn2, 2);
        }

        // === Wait for Q data ===
        cute::wait_barrier(shared_plan->bar_load_qb[buf_idx_value], local_phase);
        Tensor sQ = make_tensor(make_smem_ptr(shared_plan->q[buf_idx_value].data()), SmemLayoutInputBF16{});

        // === COMPUTE: fused kg_intra + qkg_intra for shared rows (saves sK+sG reloads) ===
        {
            float2 beta[4];
            if constexpr (WG_IDX == 0) {
                float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
                float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
                for (int j = 1; j <= 2; ++j) {
                    int x = idx_in_warpgroup / 8 + j * 16;
                    if (x < sub_seq_len) beta[j] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x]));
                }
                // tile_j=1: shared row idx/8+16 (kg uses gn3, qkg uses gn1)
                setup_intra_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_intra), decltype(sQKG_intra), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_intra, sQKG_intra, 1, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta[1], 4, 0);
                // tile_j=2: shared row idx/8+32 (kg uses gn3, qkg uses gn1)
                setup_intra_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_intra), decltype(sQKG_intra), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_intra, sQKG_intra, 2, idx_in_warpgroup, sub_seq_len, gn3, gn1, beta[2], 5, 1);
            }
        }

        // === COMPUTE: fused kg_inter + qkg_inter ===
        {
            Tensor sKG_inter = make_tensor(make_smem_ptr(shared_plan->kg_all.inter[0].data()), SmemLayoutMatBTF32Tranposed<1>{});
            Tensor sQKG_inter = make_tensor(make_smem_ptr(shared_plan->qkg_all.inter[0].data()), SmemLayoutMatBTF32Tranposed<2>{});
            float2 beta[4];

            if constexpr (WG_IDX == 0) {
                beta[0] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][idx_in_warpgroup / 8]));
                int x3 = idx_in_warpgroup / 8 + 48;
                if (x3 < sub_seq_len) beta[3] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x3]));
                float4 gn_half_0, gn_half_3;
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 0, idx_in_warpgroup, sub_seq_len, beta[0], gn_half_0);
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 3, idx_in_warpgroup, sub_seq_len, beta[3], gn_half_3);
            } else {
                int x1 = idx_in_warpgroup / 8 + 16;
                if (x1 < sub_seq_len) beta[1] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x1]));
                int x2 = idx_in_warpgroup / 8 + 32;
                if (x2 < sub_seq_len) beta[2] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x2]));
                float4 gn_half_1, gn_half_2;
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 1, idx_in_warpgroup, sub_seq_len, beta[1], gn_half_1);
                setup_inter_fused<decltype(sG), decltype(sK), decltype(sQ), decltype(sKG_inter), decltype(sQKG_inter), kg_offset, qkg_offset>(
                    sG, sK, sQ, sKG_inter, sQKG_inter, 2, idx_in_warpgroup, sub_seq_len, beta[2], gn_half_2);
            }
        }

        fence_view_async_shared();
        cute::arrive_barrier(shared_plan->bar_kg_all_ready); // kg_all complete (intra + inter)

        // === COMPUTE: mask_At (first k_idx only) ===
        if (k_idx == 0) {
            mask_At_tensor<WG_IDX * 64, 64>(shared_plan->dAqk[buf_idx_A].data(), shared_plan->dAkk[buf_idx_A].data(), idx_in_warpgroup, sub_seq_len, tmem_addr::dAqk_t);
            tcgen05_before_thread_sync();
            cute::arrive_barrier(shared_plan->bar_dAt_ready[buf_idx_A]);
        }

        // === EPILOGUE: compute intra scale (can overlap with MMA kg phase) ===
        float scale[HALF_K];
        epilogue_compute_intra_scale<HALF_K, K_OFF>(sG, idx_in_warpgroup, scale);

        // === COMPUTE: qkg_intra (non-overlapping rows only) ===
        {
            float2 beta[4];
            if constexpr (WG_IDX == 0) {
                float4 gn1 = *reinterpret_cast<float4*>(&sG(16, y));
                int x3 = idx_in_warpgroup / 8 + 48;
                if (x3 < sub_seq_len) beta[3] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x3]));
                setup_qkg_intra<decltype(sG), decltype(sQ), decltype(sK), decltype(sQKG_intra), qkg_offset>(sG, sQ, sK, sQKG_intra, 3, idx_in_warpgroup, sub_seq_len, beta[3], gn1, 2);
            } else {
                float4 gn2 = *reinterpret_cast<float4*>(&sG(32, y));
                float4 gn3 = *reinterpret_cast<float4*>(&sG(48, y));
                int x2 = idx_in_warpgroup / 8 + 32;
                if (x2 < sub_seq_len) beta[2] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x2]));
                int x3 = idx_in_warpgroup / 8 + 48;
                if (x3 < sub_seq_len) beta[3] = __bfloat1622float2(__bfloat162bfloat162(shared_plan->beta_smem[beta_buf][x3]));
                setup_qkg_intra<decltype(sG), decltype(sQ), decltype(sK), decltype(sQKG_intra), qkg_offset>(sG, sQ, sK, sQKG_intra, 2, idx_in_warpgroup, sub_seq_len, beta[2], gn2, 3);
                setup_qkg_intra_2gn<decltype(sG), decltype(sQ), decltype(sK), decltype(sQKG_intra), qkg_offset>(sG, sQ, sK, sQKG_intra, 3, idx_in_warpgroup, sub_seq_len, beta[3], gn2, gn3, 4, 5);
            }
        }

        fence_view_async_shared();
        cute::arrive_barrier(shared_plan->bar_qkg_all_ready); // all qkg data ready for MMA

        // === EPILOGUE: wait for dq+dq2 results from MMA ===
        cute::wait_barrier(shared_plan->bar_dq_done, b_phase);

        float res[HALF_K];
        epilogue_apply_dq_intra<HALF_K>(idx_in_warpgroup, tmem_addr::dq + K_OFF + 256 * buf_idx_value, res, scale);

        // [DEBUG-A] Check res after dq_intra (before inter combine)
        DEBUG_CHECK_NAN("A:res_dq_intra", res, HALF_K, idx_in_warpgroup, k_idx);

        // Compute sBkExp scale directly from sG
        {
            int row = idx_in_warpgroup % 64;
            int g_half_row = min(row / 16 * 16 + 8, sub_seq_len - 1);
            for (int i = 0; i < HALF_K / 4; ++i) {
                float4 bg = *reinterpret_cast<float4*>(&sG(row, K_OFF + i * 4));
                float4 bg_half = *reinterpret_cast<float4*>(&sG(g_half_row, K_OFF + i * 4));
                float2 diff0 = float2_sub(reinterpret_cast<float2*>(&bg)[0], reinterpret_cast<float2*>(&bg_half)[0]);
                float2 diff1 = float2_sub(reinterpret_cast<float2*>(&bg)[1], reinterpret_cast<float2*>(&bg_half)[1]);
                scale[i * 4]     = exp2f(diff0.x);
                scale[i * 4 + 1] = exp2f(diff0.y);
                scale[i * 4 + 2] = exp2f(diff1.x);
                scale[i * 4 + 3] = exp2f(diff1.y);
            }
        }

        // [DEBUG-B] Check inter scale (exp2f could overflow)
        DEBUG_CHECK_NAN("B:inter_scale", scale, HALF_K, idx_in_warpgroup, k_idx);

        epilogue_combine_dq_inter<HALF_K>(tmem_addr::dq2 + K_OFF + 256 * buf_idx_value, res, scale);

        // [DEBUG-C] Check res after dq_inter combine
        DEBUG_CHECK_NAN("C:res_dq_combined", res, HALF_K, idx_in_warpgroup, k_idx);

        // === EPILOGUE: output dq / accumulate db ===
        {
            Tensor sQ = make_tensor(make_smem_ptr(shared_plan->q[buf_idx_value].data()), SmemLayoutInputBF16{});
            if (idx_in_warpgroup >= 64) {
                // [DEBUG-D] Check res before beta scaling (this is the dq result for upper half)
                DEBUG_CHECK_NAN("D:res_pre_beta(upper)", res, HALF_K, idx_in_warpgroup, k_idx);

                bf16 beta_val = (local_idx < sub_seq_len)
                    ? (bf16)(__nv_bfloat16)shared_plan->beta_smem[beta_buf][local_idx]
                    : bf16{};
                float *db_out_addr = nullptr;
                epilogue_accumulate_db<HALF_K, K_OFF>(sK, idx_in_warpgroup, sub_seq_len, res, db, false, db_out_addr, beta_val);

                // [DEBUG-E] Check res after beta scaling (res = dq_result * beta, used for dk)
                DEBUG_CHECK_NAN("E:res_post_beta(upper)", res, HALF_K, idx_in_warpgroup, k_idx);
            } else {
                Tensor sDQ = make_tensor(make_smem_ptr(shared_plan->dq[buf_idx_value].data()), SmemLayoutInputFP32{});
                __nv_bfloat16 *dq_out_base = (__nv_bfloat16*)(params.dq_out_ptr)
                    + (start_offset + tile_idx * T_TILE + local_idx) * params.h * K_SIZE
                    + head_idx * K_SIZE + k_idx * K_TILE + K_OFF;
                epilogue_output_dq<HALF_K, K_OFF>(sQ, sDQ, idx_in_warpgroup, sub_seq_len, res, dq_out_base);
            }
        }

        // === PER-ITERATION DB REDUCE: WG0 partial -> smem -> WG1 accumulates ===
        if (idx_in_warpgroup >= 64) {
            if (WG_IDX == 0 && local_idx < sub_seq_len) {
                shared_plan->db_partial[0][local_idx] = db;
            }
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, 1);
            if (WG_IDX == 1 && local_idx < sub_seq_len) {
                db += shared_plan->db_partial[0][local_idx];
            }
            if constexpr (WG_IDX == 0) {
                db = 0.0f;
            }
        }

        // === EPILOGUE: compute dkt scale (directly from sG) ===
        epilogue_compute_dkt_scale<HALF_K, K_OFF>(sG, idx_in_warpgroup, sub_seq_len, scale);

        // [DEBUG-F] Check dkt scale
        DEBUG_CHECK_NAN("F:dkt_scale", scale, HALF_K, idx_in_warpgroup, k_idx);

        // === EPILOGUE: wait for dkt results from MMA ===
        cute::wait_barrier(shared_plan->bar_dkt_done, b_phase);

        float res_dkt[HALF_K];
        epilogue_process_dkt<HALF_K>(idx_in_warpgroup, tmem_addr::dkt + K_OFF + 256 * buf_idx_value, res_dkt, scale, sub_seq_len);

        // [DEBUG-G] Check res_dkt after process (TMEM result * scale)
        DEBUG_CHECK_NAN("G:res_dkt_processed", res_dkt, HALF_K, idx_in_warpgroup, k_idx);

        // dkt exchange: each WG syncs within itself (128 threads)
        NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);
        Tensor sDKT_0 = make_tensor(make_smem_ptr(shared_plan->b_k_neg_exp[buf_idx_value].data()), SmemLayoutInputFP32{});
        Tensor sDKT_1 = make_tensor(make_smem_ptr(shared_plan->b_k_exp[buf_idx_value].data()), SmemLayoutInputFP32{});
        epilogue_exchange_dkt<HALF_K, K_OFF>(sDKT_0, sDKT_1, idx_in_warpgroup, res, res_dkt);
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, DKT_BAR_ID);

        // === EPILOGUE: output dg / dk ===
        cute::wait_barrier(shared_plan->bar_load_dkg_ready[buf_idx_value], local_phase);
        if (idx_in_warpgroup < 64) {
            if (local_idx < sub_seq_len) {
                Tensor sDG = make_tensor(make_smem_ptr(shared_plan->dg[buf_idx_value].data()), SmemLayoutInputFP32{});
                float *dg_out_base = (float*)(params.dg_out_ptr)
                    + (start_offset + tile_idx * T_TILE + local_idx) * params.h * K_SIZE
                    + head_idx * K_SIZE + k_idx * K_TILE + K_OFF;
                epilogue_output_dg<HALF_K, K_OFF>(sK, sDG, sDKT_1, idx_in_warpgroup, res, res_dkt, dg_out_base);
            }
        } else {
            Tensor sDK = make_tensor(make_smem_ptr(shared_plan->dk[buf_idx_value].data()), SmemLayoutInputFP32{});

            // [DEBUG-H] Check all dk inputs (upper half only)
            DEBUG_CHECK_NAN("H:dk_res(beta*dq)", res, HALF_K, idx_in_warpgroup, k_idx);
            DEBUG_CHECK_NAN("H:dk_res_dkt", res_dkt, HALF_K, idx_in_warpgroup, k_idx);
            // Check sDKT_0 (from lower half exchange)
            if (local_idx < sub_seq_len) {
                float dkt_sub_check[HALF_K];
                for (int _i = 0; _i < HALF_K / 4; ++_i) {
                    *reinterpret_cast<float4*>(&dkt_sub_check[_i * 4]) = *reinterpret_cast<float4*>(&sDKT_0(local_idx, K_OFF + _i * 4));
                }
                DEBUG_CHECK_NAN("H:dk_dkt_sub(from_lower)", dkt_sub_check, HALF_K, idx_in_warpgroup, k_idx);
                // Check dk_input from TMA
                float dk_input_check[HALF_K];
                for (int _i = 0; _i < HALF_K / 4; ++_i) {
                    *reinterpret_cast<float4*>(&dk_input_check[_i * 4]) = *reinterpret_cast<float4*>(&sDK(local_idx, K_OFF + _i * 4));
                }
                DEBUG_CHECK_NAN("H:dk_input(TMA)", dk_input_check, HALF_K, idx_in_warpgroup, k_idx);
            }

            __nv_bfloat16 *dk_out_base = (__nv_bfloat16*)(params.dk_out_ptr)
                + (start_offset + tile_idx * T_TILE + local_idx) * params.h * K_SIZE
                + head_idx * K_SIZE + k_idx * K_TILE + K_OFF;
            epilogue_output_dk<HALF_K, K_OFF>(sDK, sDKT_0, idx_in_warpgroup, sub_seq_len, res, res_dkt, dk_out_base);
        }

        cute::arrive_barrier(shared_plan->bar_dvalue_free[buf_idx_value]);
        b_phase ^= 1;
        state_phase ^= 1 << buf_idx_value;
        buf_idx_value = (buf_idx_value + 1) % NUM_BUF_VALUE;
    }

    // === DB OUTPUT: WG1 already holds the fully-reduced db (per-iteration reduce done above) ===
    if (idx_in_warpgroup >= 64 && WG_IDX == 1 && local_idx < sub_seq_len) {
        reinterpret_cast<float*>(params.db_out_ptr)[(start_offset + tile_idx * T_TILE + local_idx) * params.h + head_idx] = db;
    }

    state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
    buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
}

template<typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1) 
kda_bwd_intra_sm100_kernel(__grid_constant__ const KDA_bwd_intra_params params, __grid_constant__ const TmaParams tma_params) {
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warp = threadIdx.x % 32;
    auto role = warp_idx_to_role(warp_idx);
    TileScheduler tile_scheduler{params.tile_scheduler_params};

    extern __shared__ char shared_buf[];
    SharedMemoryPlan *shared_plan = reinterpret_cast<SharedMemoryPlan*>(shared_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dAqk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dAkk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dq.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dk.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dg.get_tma_descriptor());
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            for (int i = 0; i < NUM_BUF_VALUE; ++i) {
                cute::initialize_barrier(shared_plan->bar_load_kg_ready[i], 1);
                cute::initialize_barrier(shared_plan->bar_load_qb[i], 1);
                cute::initialize_barrier(shared_plan->bar_load_dkg_ready[i], 1);
                cute::initialize_barrier(shared_plan->bar_dvalue_free[i], 256);
            }
            // B-matrix barriers: single-buffered, phase-tracked
            cute::initialize_barrier(shared_plan->bar_kg_all_ready, 256);   // CE(256) → MMA
            cute::initialize_barrier(shared_plan->bar_qkg_all_ready, 256);  // CE(256) → MMA
            cute::initialize_barrier(shared_plan->bar_dq_done, 1);          // MMA(1) → CE
            cute::initialize_barrier(shared_plan->bar_dkt_done, 1);         // MMA(1) → CE
            for (int i = 0; i < NUM_BUF_A; ++i) {
                cute::initialize_barrier(shared_plan->bar_load_dA_ready[i], 1);
                cute::initialize_barrier(shared_plan->bar_dA_ready[i], 256);
                cute::initialize_barrier(shared_plan->bar_dAt_ready[i], 256);
                cute::initialize_barrier(shared_plan->bar_dA_mask_ready[i], 64); // repurposed: beta load by Empty warps (2 warps = 64 threads)
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, shared_plan->tmem_start_addr.data());
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    int state_phase = 0;
    int buf_idx_A = 0;
    int buf_idx_value = 0;
    int tile_phase = 0; // for beta barrier (bar_dA_mask_ready), flips each tile
    int *chunk_indices_ptr = (int*)params.chunk_indices_ptr;
    int *cu_len_ptr = (int*)params.cu_seqlens_ptr;
    int total_tiles = tile_scheduler.total_tiles();

    if (role == WarpRole::ComputeEpilogue) {
        cutlass::arch::warpgroup_reg_alloc<REG_COMPUTE>();
        // === PERSISTENT CE LOOP ===
        for (;;) {
            int A_phase = (state_phase >> (buf_idx_A + NUM_BUF_VALUE)) & 1;
            // Wait for Load warp to write tile_id + TMA dA
            cute::wait_barrier(shared_plan->bar_load_dA_ready[buf_idx_A], A_phase);
            int tid = shared_plan->tile_id[A_phase];
            if (tid >= total_tiles) {
                // Signal MMA to terminate: arrive bar_dA_ready so MMA can read sentinel
                cute::arrive_barrier(shared_plan->bar_dA_ready[buf_idx_A]);
                break;
            }
            // Decode tile coordinates from tile_id
            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int start_offset = cu_len_ptr[batch_idx];
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);
            if (warpgroup_idx == 0) {
                compute_epilogue_body<0>(shared_plan, params, idx_in_warpgroup, state_phase, buf_idx_A, buf_idx_value,
                    batch_idx, head_idx, tile_idx, start_offset, sub_seq_len, tile_phase, A_phase);
            } else {
                compute_epilogue_body<1>(shared_plan, params, idx_in_warpgroup, state_phase, buf_idx_A, buf_idx_value,
                    batch_idx, head_idx, tile_idx, start_offset, sub_seq_len, tile_phase, A_phase);
            }
            tile_phase ^= 1;
        }
    } else if (role == WarpRole::Mma) {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
        if (elect_one_sync()) {
            // === PERSISTENT MMA LOOP ===
            for (;;) {
                int A_phase = (state_phase >> (buf_idx_A + NUM_BUF_VALUE)) & 1;
                // Wait for CE to finish mask_A (signals bar_dA_ready)
                cute::wait_barrier(shared_plan->bar_dA_ready[buf_idx_A], A_phase);
                int tid = shared_plan->tile_id[A_phase];
                if (tid >= total_tiles) break;
                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);
                int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
                int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);
                int b_phase = 0;

                tcgen05_after_thread_sync();
            
                for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                    int local_phase = (state_phase >> buf_idx_value) & 1;

                    // === KG PHASE === 
                    // Launder shared_plan pointer to prevent compiler from hoisting qkg addresses into kg phase
                    {
                        SharedMemoryPlan *sp = shared_plan;
                        asm volatile("" : "+l"(sp) :: "memory");

                        cute::wait_barrier(sp->bar_kg_all_ready, b_phase);
                    
                        TiledMMA tile_mma_dqk_mask02 = TiledMMA_KDAqk_MASK02{};
                        TiledMMA tile_mma_dqk_mask13 = TiledMMA_KDAqk_MASK13{};
                        Tensor tDQ_02 = partition_fragment_C(tile_mma_dqk_mask02, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
                        Tensor tDQ_13 = partition_fragment_C(tile_mma_dqk_mask02, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
                        tDQ_02.data().get() = tmem_addr::dq_02 + 256 * buf_idx_value;
                        tDQ_13.data().get() = tmem_addr::dq_13 + 256 * buf_idx_value;
                        
                        // kg_intra: 3 MMA calls
                        {
                            Tensor tAqk_1 = tile_mma_dqk_mask02.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask02, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE>>{})
                            );
                            tAqk_1.data().get() = tmem_addr::dAqk_13 + 256 * buf_idx_A;
                            Tensor sKG_1 = make_tensor(make_smem_ptr(sp->kg_all.intra[0].data()), SmemLayoutMatBTF32Tranposed<1>{});
                            utcmma_ts(tile_mma_dqk_mask02, tAqk_1, sKG_1, tDQ_13, true);

                            Tensor tAqk_2 = tile_mma_dqk_mask02.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask02, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 2>>{})
                            );
                            tAqk_2.data().get() = tmem_addr::dAqk_02 + 256 * buf_idx_A;
                            Tensor sKG_2 = make_tensor(make_smem_ptr(sp->kg_all.intra[1].data()), SmemLayoutMatBTF32Tranposed<2>{});
                            utcmma_ts(tile_mma_dqk_mask13, tAqk_2, sKG_2, tDQ_02, true);

                            Tensor tAqk_3 = tile_mma_dqk_mask02.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask02, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 3>>{})
                            );
                            tAqk_3.data().get() = tmem_addr::dAqk_13 + 256 * buf_idx_A;
                            Tensor sKG_3 = make_tensor(make_smem_ptr(sp->kg_all.intra[3].data()), SmemLayoutMatBTF32Tranposed<3>{});
                            utcmma_ts(tile_mma_dqk_mask13, tAqk_3, sKG_3, tDQ_13, true);
                        }
                        
                        tcgen05_after_thread_sync();

                        // Re-launder for kg_inter to separate intra/inter address computation
                        asm volatile("" : "+l"(sp) :: "memory");

                        // kg_inter: 4 MMA calls
                        Tensor tDQ2_02 = partition_fragment_C(tile_mma_dqk_mask02, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
                        Tensor tDQ2_13 = partition_fragment_C(tile_mma_dqk_mask02, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
                        tDQ2_02.data().get() = tmem_addr::dq2_02 + 256 * buf_idx_value;
                        tDQ2_13.data().get() = tmem_addr::dq2_13 + 256 * buf_idx_value;
                        
                        {
                            Tensor tAqk_0 = tile_mma_dqk_mask02.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask02, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE>>{})
                            );
                            tAqk_0.data().get() = tmem_addr::dAqk_02 + 256 * buf_idx_A;
                            Tensor sKG_0 = make_tensor(make_smem_ptr(sp->kg_all.inter[0].data()), SmemLayoutMatBTF32Tranposed<1>{});
                            utcmma_ts(tile_mma_dqk_mask02, tAqk_0, sKG_0, tDQ2_02, true);

                            Tensor tAqk_1 = tile_mma_dqk_mask02.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask02, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE>>{})
                            );
                            tAqk_1.data().get() = tmem_addr::dAqk_13 + 16 + 256 * buf_idx_A;
                            Tensor sKG_1 = make_tensor(make_smem_ptr(sp->kg_all.inter[1].data()), SmemLayoutMatBTF32Tranposed<1>{});
                            utcmma_ts(tile_mma_dqk_mask02, tAqk_1, sKG_1, tDQ2_13, true);

                            Tensor tAqk_2 = tile_mma_dqk_mask02.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask02, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE>>{})
                            );
                            tAqk_2.data().get() = tmem_addr::dAqk_02 + 32 + 256 * buf_idx_A;
                            Tensor sKG_2 = make_tensor(make_smem_ptr(sp->kg_all.inter[2].data()), SmemLayoutMatBTF32Tranposed<1>{});
                            utcmma_ts(tile_mma_dqk_mask13, tAqk_2, sKG_2, tDQ2_02, true);

                            Tensor tAqk_3 = tile_mma_dqk_mask02.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask02, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE>>{})
                            );
                            tAqk_3.data().get() = tmem_addr::dAqk_13 + 48 + 256 * buf_idx_A;
                            Tensor sKG_3 = make_tensor(make_smem_ptr(sp->kg_all.inter[3].data()), SmemLayoutMatBTF32Tranposed<1>{});
                            utcmma_ts(tile_mma_dqk_mask13, tAqk_3, sKG_3, tDQ2_13, true);
                        }
                        
                        umma_arrive_noelect(sp->bar_dq_done);
                    } // === end KG scope ===

                    tcgen05_after_thread_sync();

                    // === QKG PHASE ===
                    // Fresh laundered pointer — compiler cannot CSE with kg phase addresses
                    {
                        SharedMemoryPlan *sp = shared_plan;
                        asm volatile("" : "+l"(sp) :: "memory");

                        if (k_idx == 0) {
                            cute::wait_barrier(sp->bar_dAt_ready[buf_idx_A], A_phase);
                        }
                        cute::wait_barrier(sp->bar_qkg_all_ready, b_phase);
                        
                        TiledMMA tile_mma_dqk_mask0 = TiledMMA_KDAqk_MASK0{};
                        TiledMMA tile_mma_dqk_mask1 = TiledMMA_KDAqk_MASK1{};
                        TiledMMA tile_mma_dqk_mask2 = TiledMMA_KDAqk_MASK2{};
                        TiledMMA tile_mma_dqk_mask3 = TiledMMA_KDAqk_MASK3{};
                        Tensor tDKT_02 = partition_fragment_C(tile_mma_dqk_mask0, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
                        Tensor tDKT_13 = partition_fragment_C(tile_mma_dqk_mask0, make_shape(Int<T_TILE>{}, Int<K_TILE>{}));
                        tDKT_02.data().get() = tmem_addr::dkt + 256 * buf_idx_value;
                        tDKT_13.data().get() = tmem_addr::dkt + 16*65536 + 256 * buf_idx_value;
                        
                        // qkg_intra: 3 MMA calls
                        {
                            Tensor tAqk_1 = tile_mma_dqk_mask0.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask0, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 6>>{})
                            );
                            tAqk_1.data().get() = tmem_addr::dAqk_t_02 + 32;
                            Tensor sKG_1 = make_tensor(make_smem_ptr(sp->qkg_all.intra[0].data()), SmemLayoutMatBTF32Tranposed<6>{});
                            utcmma_ts(tile_mma_dqk_mask0, tAqk_1, sKG_1, tDKT_02, true);

                            Tensor tAqk_2 = tile_mma_dqk_mask0.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask0, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 4>>{})
                            );
                            tAqk_2.data().get() = tmem_addr::dAqk_t_13 + 64;
                            Tensor sKG_2 = make_tensor(make_smem_ptr(sp->qkg_all.intra[3].data()), SmemLayoutMatBTF32Tranposed<4>{});
                            utcmma_ts(tile_mma_dqk_mask0, tAqk_2, sKG_2, tDKT_13, true);

                            Tensor tAqk_3 = tile_mma_dqk_mask0.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask0, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 2>>{})
                            );
                            tAqk_3.data().get() = tmem_addr::dAqk_t_02 + 96;
                            Tensor sKG_3 = make_tensor(make_smem_ptr(sp->qkg_all.intra[5].data()), SmemLayoutMatBTF32Tranposed<2>{});
                            utcmma_ts(tile_mma_dqk_mask1, tAqk_3, sKG_3, tDKT_02, true);
                        }
                        
                        tcgen05_after_thread_sync();
                        
                        // Re-launder for qkg_inter
                        asm volatile("" : "+l"(sp) :: "memory");
                        tcgen05_before_thread_sync();
                        
                        // qkg_inter: 4 MMA calls
                        {
                            Tensor tAqk_0 = tile_mma_dqk_mask2.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask2, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 2>>{})
                            );
                            tAqk_0.data().get() = tmem_addr::dAqk_t_02;
                            Tensor sKG_0 = make_tensor(make_smem_ptr(sp->qkg_all.inter[0].data()), SmemLayoutMatBTF32Tranposed<2>{});
                            utcmma_ts(tile_mma_dqk_mask2, tAqk_0, sKG_0, tDKT_02, true);

                            Tensor tAqk_1 = tile_mma_dqk_mask2.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask2, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 2>>{})
                            );
                            tAqk_1.data().get() = tmem_addr::dAqk_t_13 + 32;
                            Tensor sKG_1 = make_tensor(make_smem_ptr(sp->qkg_all.inter[1].data()), SmemLayoutMatBTF32Tranposed<2>{});
                            utcmma_ts(tile_mma_dqk_mask2, tAqk_1, sKG_1, tDKT_13, true);

                            Tensor tAqk_2 = tile_mma_dqk_mask2.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask2, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 2>>{})
                            );
                            tAqk_2.data().get() = tmem_addr::dAqk_t_02 + 64;
                            Tensor sKG_2 = make_tensor(make_smem_ptr(sp->qkg_all.inter[2].data()), SmemLayoutMatBTF32Tranposed<2>{});
                            utcmma_ts(tile_mma_dqk_mask3, tAqk_2, sKG_2, tDKT_02, true);

                            Tensor tAqk_3 = tile_mma_dqk_mask2.get_slice(_0{}).make_fragment_A(
                                partition_shape_A(tile_mma_dqk_mask2, Shape<Int<CHUNK_SIZE>, Int<SUB_T_TILE * 2>>{})
                            );
                            tAqk_3.data().get() = tmem_addr::dAqk_t_13 + 96;
                            Tensor sKG_3 = make_tensor(make_smem_ptr(sp->qkg_all.inter[3].data()), SmemLayoutMatBTF32Tranposed<2>{});
                            utcmma_ts(tile_mma_dqk_mask3, tAqk_3, sKG_3, tDKT_13, true);
                        }
                        umma_arrive_noelect(sp->bar_dkt_done);
                    } // === end QKG scope ===

                    tcgen05_after_thread_sync();
                    b_phase ^= 1;
                    state_phase ^= 1 << buf_idx_value;
                    buf_idx_value = (buf_idx_value + 1) % NUM_BUF_VALUE;
                }
                state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
                buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
            } // end persistent MMA tile
        }
    } else if (role == WarpRole::Load) {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
        if (elect_one_sync()) {
            // === PERSISTENT LOAD LOOP ===
            for (;;) {
                int A_phase = (state_phase >> (buf_idx_A + NUM_BUF_VALUE)) & 1;

                // Fetch next tile via atomicAdd
                int tid = tile_scheduler.get_next_tile_id();
                shared_plan->tile_id[A_phase] = tid; // write to double-buffered smem (indexed by A_phase)
                __threadfence_block(); // ensure tile_id visible to all CTA threads before TMA barrier fires

                if (tid >= total_tiles) {
                    // Signal CE+Empty: atomic arrive + expect_tx=0 to trigger the TMA transaction barrier
                    uint32_t bar_addr = cute::cast_smem_ptr_to_uint(&shared_plan->bar_load_dA_ready[buf_idx_A]);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 0;" :: "r"(bar_addr));
                    break;
                }

                // Decode tile coordinates
                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);
                int token_offset = cu_len_ptr[batch_idx];
                int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
                int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

                // Wait for CE to finish using dA from previous tile (protects dA smem)
                cute::wait_barrier(shared_plan->bar_dAt_ready[buf_idx_A], A_phase ^ 1);

                // TMA load dA
                Tensor sDAqk = make_tensor(make_smem_ptr(shared_plan->dAqk[buf_idx_A].data()), SmemLayoutDA{});
                Tensor sDAkk = make_tensor(make_smem_ptr(shared_plan->dAkk[buf_idx_A].data()), SmemLayoutDA{});
                int tma_bytes_dakk = sizeof(make_tensor_like(sDAkk));
                int tma_bytes_daqk = sizeof(make_tensor_like(sDAqk));
                int tma_transaction_A_bytes = tma_bytes_dakk + tma_bytes_daqk;
                Tensor mDakk = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_dAkk.get_tma_tensor(tma_params.shape_da));
                Tensor mDaqk = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_dAqk.get_tma_tensor(tma_params.shape_da));
                Tensor gDakk = local_tile(mDakk(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<T_TILE>{}), make_coord(tile_idx, _0{}));
                Tensor gDaqk = local_tile(mDaqk(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<T_TILE>{}), make_coord(tile_idx, _0{}));
                cute::set_barrier_transaction_bytes(shared_plan->bar_load_dA_ready[buf_idx_A], tma_transaction_A_bytes);
                launch_tma_copy(tma_params.tma_dAkk, gDakk, sDAkk, shared_plan->bar_load_dA_ready[buf_idx_A]);
                launch_tma_copy(tma_params.tma_dAqk, gDaqk, sDAqk, shared_plan->bar_load_dA_ready[buf_idx_A]);

                // TMA load per-k_idx data
                for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                    int local_phase = (state_phase >> buf_idx_value) & 1;
                    cute::wait_barrier(shared_plan->bar_dvalue_free[buf_idx_value], local_phase^1);

                    Tensor sQ = make_tensor(make_smem_ptr(shared_plan->q[buf_idx_value].data()), SmemLayoutInputBF16{});
                    Tensor sDQ = make_tensor(make_smem_ptr(shared_plan->dq[buf_idx_value].data()), SmemLayoutInputFP32{});
                    Tensor sK = make_tensor(make_smem_ptr(shared_plan->k[buf_idx_value].data()), SmemLayoutInputBF16{});
                    Tensor sG = make_tensor(make_smem_ptr(shared_plan->g[buf_idx_value].data()), SmemLayoutInputFP32{});
                    Tensor sDK = make_tensor(make_smem_ptr(shared_plan->dk[buf_idx_value].data()), SmemLayoutInputFP32{});
                    Tensor sDG = make_tensor(make_smem_ptr(shared_plan->dg[buf_idx_value].data()), SmemLayoutInputFP32{});
                    int tma_transaction_kg_bytes = sizeof(make_tensor_like(sK)) + sizeof(make_tensor_like(sG));
                    int tma_transaction_q_bytes = sizeof(make_tensor_like(sQ)) + sizeof(make_tensor_like(sDQ));
                    int tma_transaction_dkg_bytes = sizeof(make_tensor_like(sDK)) + sizeof(make_tensor_like(sDG));
                    Tensor mK = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_k.get_tma_tensor(tma_params.shape_qkg));
                    Tensor mG = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_g.get_tma_tensor(tma_params.shape_qkg));
                    Tensor mQ = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_q.get_tma_tensor(tma_params.shape_qkg));
                    Tensor mDQ = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_dq.get_tma_tensor(tma_params.shape_qkg));
                    Tensor mDK = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_dk.get_tma_tensor(tma_params.shape_qkg));
                    Tensor mDG = domain_offset(make_coord(token_offset, _0{}, _0{}), tma_params.tma_dg.get_tma_tensor(tma_params.shape_qkg));

                    Tensor gK = local_tile(mK(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    Tensor gG = local_tile(mG(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    Tensor gQ = local_tile(mQ(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    Tensor gDQ = local_tile(mDQ(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    Tensor gDK = local_tile(mDK(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));
                    Tensor gDG = local_tile(mDG(_, _, head_idx), make_shape(Int<T_TILE>{}, Int<K_TILE>{}), make_coord(tile_idx, k_idx));

                    cute::set_barrier_transaction_bytes(shared_plan->bar_load_kg_ready[buf_idx_value], tma_transaction_kg_bytes);
                    launch_tma_copy(tma_params.tma_k, gK, sK, shared_plan->bar_load_kg_ready[buf_idx_value]);
                    launch_tma_copy(tma_params.tma_g, gG, sG, shared_plan->bar_load_kg_ready[buf_idx_value]);

                    cute::set_barrier_transaction_bytes(shared_plan->bar_load_qb[buf_idx_value], tma_transaction_q_bytes);
                    launch_tma_copy(tma_params.tma_q, gQ, sQ, shared_plan->bar_load_qb[buf_idx_value]);
                    launch_tma_copy(tma_params.tma_dq, gDQ, sDQ, shared_plan->bar_load_qb[buf_idx_value]);

                    cute::set_barrier_transaction_bytes(shared_plan->bar_load_dkg_ready[buf_idx_value], tma_transaction_dkg_bytes);
                    launch_tma_copy(tma_params.tma_dk, gDK, sDK, shared_plan->bar_load_dkg_ready[buf_idx_value]);
                    launch_tma_copy(tma_params.tma_dg, gDG, sDG, shared_plan->bar_load_dkg_ready[buf_idx_value]);

                    state_phase ^= 1 << buf_idx_value;
                    buf_idx_value = (buf_idx_value + 1) % NUM_BUF_VALUE;
                }
                state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
                buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
            } // end persistent Load tile
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
        // === PERSISTENT EMPTY WARP LOOP (beta loading) ===
        int empty_idx = threadIdx.x - (NUM_THREADS - 64); // 0..63
        for (;;) {
            int A_phase = (state_phase >> (buf_idx_A + NUM_BUF_VALUE)) & 1;
            // Wait for Load to write tile_id + dA (same barrier CE uses)
            cute::wait_barrier(shared_plan->bar_load_dA_ready[buf_idx_A], A_phase);
            int tid = shared_plan->tile_id[A_phase];
            if (tid >= total_tiles) break;

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);
            int token_offset = cu_len_ptr[batch_idx];
            int seq_len = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            int sub_seq_len = min(T_TILE, seq_len - tile_idx * T_TILE);

            if (empty_idx < T_TILE) {
                shared_plan->beta_smem[A_phase][empty_idx] = (empty_idx < sub_seq_len)
                    ? reinterpret_cast<__nv_bfloat16*>(params.beta_ptr)[(token_offset + tile_idx * T_TILE + empty_idx) * params.h + head_idx]
                    : __nv_bfloat16(0);
            }
            fence_view_async_shared();
            cute::arrive_barrier(shared_plan->bar_dA_mask_ready[0]); // signal: beta_smem ready

            state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
            buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
        } // end persistent Empty tile
    }
    // === CLEANUP (once per CTA) ===
    __syncthreads();
    if (warp_idx == 0 && elect_one_sync()) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
    return;
}

void run_kda_bwd_intra_sm100(KDA_bwd_intra_params &params, cudaStream_t stream) {
    KDA_ASSERT(params.d % 32 == 0);
    int total_q_len = params.total_q_len;
    int H = params.h;
    int D = params.d;
    int BT = params.chunk_size;

    auto shape_QKG = make_shape(total_q_len, D, H);
    auto stride_QKG = make_stride(H * D, _1{}, D);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q_ptr),
            make_layout(
                shape_QKG,
                stride_QKG
            )
        ),
        SmemLayoutInputBF16{}
    );

    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.k_ptr),
            make_layout(
                shape_QKG,
                stride_QKG
            )
        ),
        SmemLayoutInputBF16{}
    );

    auto tma_G = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.g_ptr),
            make_layout(
                shape_QKG,
                stride_QKG
            )
        ),
        SmemLayoutInputFP32{}
    );

    auto shape_DA = make_shape(total_q_len, BT, H);
    auto stride_DA = make_stride(H * BT, _1{}, BT);
    auto tma_DAqk = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.dAqk_ptr),
            make_layout(
                shape_DA,
                stride_DA
            )
        ),
        SmemLayoutDA{}
    );

    auto tma_DAkk = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.dAkk_ptr),
            make_layout(
                shape_DA,
                stride_DA
            )
        ),
        SmemLayoutDA{}
    );

    auto tma_DQ = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.dq_ptr),
            make_layout(
                shape_QKG,
                stride_QKG
            )
        ),
        SmemLayoutInputFP32{}
    );

    auto tma_DK = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.dk_ptr),
            make_layout(
                shape_QKG,
                stride_QKG
            )
        ),
        SmemLayoutInputFP32{}
    );

    auto tma_DG = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.dg_ptr),
            make_layout(
                shape_QKG,
                stride_QKG
            )
        ),
        SmemLayoutInputFP32{}
    );

    TmaParams<
        decltype(shape_QKG), decltype(shape_DA), 
        decltype(tma_Q), decltype(tma_K), decltype(tma_G), 
        decltype(tma_DAqk), decltype(tma_DAkk), decltype(tma_DQ), decltype(tma_DK), decltype(tma_DG)
    > tma_params = {
        shape_QKG,
        shape_DA,
        tma_Q,
        tma_K,
        tma_G,
        tma_DAqk,
        tma_DAkk,
        tma_DQ,
        tma_DK,
        tma_DG,
    };

    auto kda_kernel = &kda_bwd_intra_sm100_kernel<decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_dim(TileScheduler::get_grid_shape(params.tile_scheduler_params));
    dim3 block_dim(NUM_THREADS, 1, 1);
    kda_kernel<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params);
    return;
}
}