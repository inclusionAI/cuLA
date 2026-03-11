#include "kda_fwd_common.cuh"
#include "helpers.h"
#include "gemm.h"
#include "kda_fwd_intra_kernel_sm100.hpp"

#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>
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

constexpr int SUB_T_TILE = 16;
constexpr int T_TILE = 64;
constexpr int K_SIZE = 128;
constexpr int K_TILE = 32;
constexpr int K_ITERATION = K_SIZE / K_TILE;
constexpr int NUM_BUF_A = 1;
constexpr int NUM_BUF_VALUE = 2;
constexpr int NUM_THREADS = 128 * 4;
constexpr int CHUNK_SIZE = 64;
constexpr int REG_COMPUTE = 168;
constexpr int REG_LOAD = 64;
constexpr int REG_INVERSE = 104;

namespace tmem_addr {

};

enum class WarpRole {
    Empty = 0x0, Load = 0x1, Mma = 0x2, Compute = 0x3, Epilogue = 0x4,
    ComputeEpilogue = 0x5, Inverse = 0x6
};

// Warp layout (16 warps, 512 threads total):
//   warp  0- 7  (thread   0-255): ComputeEpilogue  — WG0 (warp 0-3) + WG1 (warp 4-7)
//   warp  8-11  (thread 256-383): Inverse          — 1 warpgroup (128 threads) for inv(KK)
//   warp  12    (thread 384-415): Mma              — 1 warp, uses elect_one_sync
//   warp  13    (thread 416-447): Load             — 1 warp, uses elect_one_sync
//   warp 14-15  (thread 448-511): Empty            - 2 warps, used for beta loading
static constexpr unsigned long long kWarpAssignment = 0x12'6666'5555'5555ull;

__forceinline__ __device__ WarpRole warp_idx_to_role(int warp_idx) {
    return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
}

// for input Q, K
using SmemLayoutInputBF16 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<T_TILE>, Int<K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// for input G
using SmemLayoutInputFP32 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<T_TILE>, Int<K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// for gated MMA K^T
// M=T_TILE, N=SUB_T_TILE*NUM_TILES, K=K_TILE
template<int NUM_TILES>
using SmemLayoutMatBTF32 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<tf32>{},
    Shape<Int<SUB_T_TILE * NUM_TILES>, Int<K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// for inverse(KK) output
using SmemLayoutOutputBF16 = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<T_TILE>, Int<T_TILE>>{}
));

// for inverse(KK)
using SmemLayoutOutputFP16 = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<fp16>{},
    Shape<Int<T_TILE>, Int<T_TILE>>{}
));

// TODO: for inverse(KK) with TF32 precision
using SmemLayoutOutputTF32 = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<tf32>{},
    Shape<Int<T_TILE>, Int<T_TILE>>{}
));

struct SharedMemoryPlan {
    // Q,K,G, double buffer
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> q[NUM_BUF_VALUE];
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> k[NUM_BUF_VALUE];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> g[NUM_BUF_VALUE];

    // gated MMA K^T, single buffer
    struct {
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32<1>>> inter[6];
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32<1>>> intra[4];
    } kg_all;

    // inv(KK), single buffer
    array_aligned<fp16, cosize_v<SmemLayoutOutputFP16>> kk[NUM_BUF_A];

    // ---------------------------------------------------------------
    // Pipeline shared storage (barriers managed by cutlass Pipeline API)
    // ---------------------------------------------------------------

    // TMA load pipelines: Load(TMA producer) -> CE/MMA/Inverse (consumers)
    alignas(16) typename cutlass::PipelineTmaAsync<NUM_BUF_VALUE>::SharedStorage pipe_q_storage;
    alignas(16) typename cutlass::PipelineTmaAsync<NUM_BUF_VALUE>::SharedStorage pipe_k_storage;
    alignas(16) typename cutlass::PipelineTmaAsync<NUM_BUF_VALUE>::SharedStorage pipe_g_storage;

    // Beta pipeline: Empty warps (producer, 64 threads) -> CE (consumer, 256 threads)
    alignas(16) typename cutlass::PipelineAsync<2>::SharedStorage pipe_beta_storage;

    // CE -> MMA: signal that kg_all is ready for MMA consumption
    alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_qkg_all_storage;
    alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_ktg_inter_storage;
    alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_ktg_intra_storage;

    // MMA -> CE: signal that QK / KK MMA results are ready in tmem
    // PipelineUmmaAsync: producer uses umma_arrive, consumer uses regular barrier wait
    alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_qk_done_storage;
    alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_kk_done_storage;

    // CE -> Inverse: signal that KK is ready for inversion
    alignas(16) typename cutlass::PipelineAsync<1>::SharedStorage pipe_kk_inv_storage;

    alignas(16) float beta_smem[2][T_TILE]; // double-buffered per-tile beta
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TileScheduler = StaticPersistentTileScheduler;

// ---------------------------------------------------------------
// Pipeline type aliases
// ---------------------------------------------------------------
// TMA load pipeline: Load warp (TMA producer) -> CE/MMA/Inverse (consumers)
using PipelineQ = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;
using PipelineK = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;
using PipelineG = cutlass::PipelineTmaAsync<NUM_BUF_VALUE>;

// Beta pipeline: Empty warps (producer) -> CE (consumer)
using PipelineBeta = cutlass::PipelineAsync<2>;

// CE -> MMA: kg_all matrices ready
using PipelineQKGAllReady   = cutlass::PipelineAsync<1>;
using PipelineKTGInterReady = cutlass::PipelineAsync<1>;
using PipelineKTGIntraReady = cutlass::PipelineAsync<1>;

// MMA (UMMA) -> CE: QK / KK results ready in tmem
using PipelineQKDone = cutlass::PipelineUmmaAsync<1>;
using PipelineKKDone = cutlass::PipelineUmmaAsync<1>;

// CE -> Inverse: KK matrix ready for inversion
using PipelineKKInvReady = cutlass::PipelineAsync<1>;

// Pipeline state type aliases
using PipelineStateQ   = cutlass::PipelineState<PipelineQ::Stages>;
using PipelineStateK   = cutlass::PipelineState<PipelineK::Stages>;
using PipelineStateG   = cutlass::PipelineState<PipelineG::Stages>;
using PipelineStateBeta = cutlass::PipelineState<PipelineBeta::Stages>;
using PipelineStateQKGAll   = cutlass::PipelineState<PipelineQKGAllReady::Stages>;
using PipelineStateKTGInter = cutlass::PipelineState<PipelineKTGInterReady::Stages>;
using PipelineStateKTGIntra = cutlass::PipelineState<PipelineKTGIntraReady::Stages>;
using PipelineStateQKDone = cutlass::PipelineState<PipelineQKDone::Stages>;
using PipelineStateKKDone = cutlass::PipelineState<PipelineKKDone::Stages>;
using PipelineStateKKInv  = cutlass::PipelineState<PipelineKKInvReady::Stages>;

// Thread count constants for pipeline params
constexpr int NUM_CE_THREADS      = 256; // warp 0-7 (2 warpgroups)
constexpr int NUM_INVERSE_THREADS = 128; // warp 8-11 (1 warpgroup)
constexpr int NUM_MMA_THREADS     = 1;   // elect_one in warp 12
constexpr int NUM_LOAD_THREADS    = 1;   // elect_one in warp 13
constexpr int NUM_EMPTY_THREADS   = 64;  // warp 14-15

// Total consumer threads for tile pipeline = CE + Inverse + MMA + Empty
// constexpr int NUM_TILE_CONSUMERS  = NUM_CE_THREADS + NUM_INVERSE_THREADS + NUM_MMA_THREADS + NUM_EMPTY_THREADS;

using ClusterShape = Shape<_1, _1, _1>;

template <int WG_IDX>
__forceinline__ __device__ void compute_prologue_body(
    SharedMemoryPlan *shared_plan,
    const KDA_fwd_intra_params &params,
    int idx_in_warpgroup,
    int &state_phase, int &buf_idx_A, int &buf_idx_value,
    int batch_idx, int head_idx, int tile_idx,
    int start_offset, int sub_seq_len, int tile_phase, int beta_buf) {

    // Wait for beta_smem (loaded by Empty warp) — handled via PipelineBeta in kernel

}

template <typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
kda_fwd_intra_sm100_kernel(__grid_constant__ const KDA_fwd_intra_params params, __grid_constant__ const TmaParams tma_params) {
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warp = threadIdx.x % 32;
    auto role = warp_idx_to_role(warp_idx);
    int lane_predicate = cute::elect_one_sync();
    TileScheduler tile_scheduler(params.tile_scheduler_params);

    extern __shared__ char shared_buf[];
    auto& shared_plan = *reinterpret_cast<SharedMemoryPlan*>(shared_buf);

    // Prefetch TMA descriptors
    if (warp_idx == 0 && lane_predicate) {
        cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
    }

    // ---------------------------------------------------------------
    // Configure pipeline params per role
    // ---------------------------------------------------------------

    // === TMA load pipelines: Q, K, G ===
    // Load warp is producer (TMA), CE+MMA+Inverse are consumers
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

    // Set roles
    if (role == WarpRole::Load) {
        q_pipe_params.role = PipelineQ::ThreadCategory::Producer;
        k_pipe_params.role = PipelineK::ThreadCategory::Producer;
        g_pipe_params.role = PipelineG::ThreadCategory::Producer;
    } else if (role == WarpRole::ComputeEpilogue) {
        q_pipe_params.role = PipelineQ::ThreadCategory::Consumer;
        k_pipe_params.role = PipelineK::ThreadCategory::Consumer;
        g_pipe_params.role = PipelineG::ThreadCategory::Consumer;
    }

    // === Beta pipeline: Empty warps(producer) -> CE(consumer) ===
    typename PipelineBeta::Params beta_pipe_params;
    beta_pipe_params.producer_arv_count = NUM_EMPTY_THREADS;
    beta_pipe_params.consumer_arv_count = NUM_CE_THREADS;
    if (role == WarpRole::Empty) {
        beta_pipe_params.role = PipelineBeta::ThreadCategory::Producer;
    } else if (role == WarpRole::ComputeEpilogue) {
        beta_pipe_params.role = PipelineBeta::ThreadCategory::Consumer;
    }

    // === CE -> MMA pipelines: qkg_all, ktg_inter, ktg_intra ===
    typename PipelineQKGAllReady::Params qkg_all_pipe_params;
    qkg_all_pipe_params.producer_arv_count = NUM_CE_THREADS;
    qkg_all_pipe_params.consumer_arv_count = NUM_MMA_THREADS;

    typename PipelineKTGInterReady::Params ktg_inter_pipe_params;
    ktg_inter_pipe_params.producer_arv_count = NUM_CE_THREADS;
    ktg_inter_pipe_params.consumer_arv_count = NUM_MMA_THREADS;

    typename PipelineKTGIntraReady::Params ktg_intra_pipe_params;
    ktg_intra_pipe_params.producer_arv_count = NUM_CE_THREADS;
    ktg_intra_pipe_params.consumer_arv_count = NUM_MMA_THREADS;

    if (role == WarpRole::ComputeEpilogue) {
        qkg_all_pipe_params.role   = PipelineQKGAllReady::ThreadCategory::Producer;
        ktg_inter_pipe_params.role = PipelineKTGInterReady::ThreadCategory::Producer;
        ktg_intra_pipe_params.role = PipelineKTGIntraReady::ThreadCategory::Producer;
    } else if (role == WarpRole::Mma) {
        qkg_all_pipe_params.role   = PipelineQKGAllReady::ThreadCategory::Consumer;
        ktg_inter_pipe_params.role = PipelineKTGInterReady::ThreadCategory::Consumer;
        ktg_intra_pipe_params.role = PipelineKTGIntraReady::ThreadCategory::Consumer;
    }

    // === MMA -> CE pipelines: qk_done, kk_done (UMMA arrive) ===
    typename PipelineQKDone::Params qk_done_pipe_params;
    qk_done_pipe_params.producer_arv_count = NUM_MMA_THREADS;
    qk_done_pipe_params.consumer_arv_count = NUM_CE_THREADS;

    typename PipelineKKDone::Params kk_done_pipe_params;
    kk_done_pipe_params.producer_arv_count = NUM_MMA_THREADS;
    kk_done_pipe_params.consumer_arv_count = NUM_CE_THREADS;

    if (role == WarpRole::Mma) {
        qk_done_pipe_params.role = PipelineQKDone::ThreadCategory::Producer;
        kk_done_pipe_params.role = PipelineKKDone::ThreadCategory::Producer;
    } else if (role == WarpRole::ComputeEpilogue) {
        qk_done_pipe_params.role = PipelineQKDone::ThreadCategory::Consumer;
        kk_done_pipe_params.role = PipelineKKDone::ThreadCategory::Consumer;
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
    // Construct pipeline objects (initializes barriers internally)
    // ---------------------------------------------------------------
    PipelineQ   q_pipeline(shared_plan.pipe_q_storage, q_pipe_params, ClusterShape{});
    PipelineK   k_pipeline(shared_plan.pipe_k_storage, k_pipe_params, ClusterShape{});
    PipelineG   g_pipeline(shared_plan.pipe_g_storage, g_pipe_params, ClusterShape{});

    PipelineBeta beta_pipeline(shared_plan.pipe_beta_storage, beta_pipe_params, cute::true_type{});

    PipelineQKGAllReady   qkg_all_pipeline(shared_plan.pipe_qkg_all_storage, qkg_all_pipe_params, cute::true_type{});
    PipelineKTGInterReady ktg_inter_pipeline(shared_plan.pipe_ktg_inter_storage, ktg_inter_pipe_params, cute::true_type{});
    PipelineKTGIntraReady ktg_intra_pipeline(shared_plan.pipe_ktg_intra_storage, ktg_intra_pipe_params, cute::true_type{});

    PipelineQKDone qk_done_pipeline(shared_plan.pipe_qk_done_storage, qk_done_pipe_params, ClusterShape{});
    PipelineKKDone kk_done_pipeline(shared_plan.pipe_kk_done_storage, kk_done_pipe_params, ClusterShape{});

    PipelineKKInvReady kk_inv_pipeline(shared_plan.pipe_kk_inv_storage, kk_inv_pipe_params, cute::true_type{});

    // ---------------------------------------------------------------
    // Initialize pipeline states
    // ---------------------------------------------------------------
    // TMA pipelines: producers start with phase=1 (buffers initially empty)
    PipelineStateQ q_pipe_state_read;
    PipelineStateQ q_pipe_state_write = cutlass::make_producer_start_state<PipelineQ>();
    PipelineStateK k_pipe_state_read;
    PipelineStateK k_pipe_state_write = cutlass::make_producer_start_state<PipelineK>();
    PipelineStateG g_pipe_state_read;
    PipelineStateG g_pipe_state_write = cutlass::make_producer_start_state<PipelineG>();

    // Beta pipeline
    PipelineStateBeta beta_pipe_state_read;
    PipelineStateBeta beta_pipe_state_write = cutlass::make_producer_start_state<PipelineBeta>();

    // CE -> MMA pipelines
    PipelineStateQKGAll   qkg_all_pipe_state_read;
    PipelineStateQKGAll   qkg_all_pipe_state_write = cutlass::make_producer_start_state<PipelineQKGAllReady>();
    PipelineStateKTGInter ktg_inter_pipe_state_read;
    PipelineStateKTGInter ktg_inter_pipe_state_write = cutlass::make_producer_start_state<PipelineKTGInterReady>();
    PipelineStateKTGIntra ktg_intra_pipe_state_read;
    PipelineStateKTGIntra ktg_intra_pipe_state_write = cutlass::make_producer_start_state<PipelineKTGIntraReady>();

    // MMA -> CE pipelines (UMMA)
    PipelineStateQKDone qk_done_pipe_state_read;
    PipelineStateQKDone qk_done_pipe_state_write = cutlass::make_producer_start_state<PipelineQKDone>();
    PipelineStateKKDone kk_done_pipe_state_read;
    PipelineStateKKDone kk_done_pipe_state_write = cutlass::make_producer_start_state<PipelineKKDone>();

    // CE -> Inverse
    PipelineStateKKInv kk_inv_pipe_state_read;
    PipelineStateKKInv kk_inv_pipe_state_write = cutlass::make_producer_start_state<PipelineKKInvReady>();

    // Allocate TMEM (warp 0 only)
    if (warp_idx == 0) {
        cute::TMEM::Allocator1Sm().allocate(512, shared_plan.tmem_start_addr.data());
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    // Barrier sync after pipeline construction (all barriers initialized)
    __syncthreads();

    int *chunk_indices_ptr = (int*)params.chunk_indices_ptr;
    int *cu_len_ptr = (int*)params.cu_seqlens_ptr;
    int total_tiles = tile_scheduler.total_tiles();

    // =======================================================================
    // WARP-SPECIALIZED PERSISTENT LOOPS
    // =======================================================================

    if (role == WarpRole::ComputeEpilogue) {
        cutlass::arch::warpgroup_reg_alloc<REG_COMPUTE>();

        // === PERSISTENT CE LOOP (static scheduling, no tile pipeline) ===
        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx  = get<1>(blk_coord);
            int tile_idx  = get<2>(blk_coord);


            // --- Wait for Q, K, G TMA loads from Load warp ---
            // Iterate over K_ITERATION sub-tiles
            for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                // q_pipeline.consumer_wait(q_pipe_state_read);
                // k_pipeline.consumer_wait(k_pipe_state_read);
                // g_pipeline.consumer_wait(g_pipe_state_read);

                // TODO: compute gated KG products, fill kg_all smem

                // Release Q, K, G smem buffers back to Load warp
                // q_pipeline.consumer_release(q_pipe_state_read);
                // k_pipeline.consumer_release(k_pipe_state_read);
                // g_pipeline.consumer_release(g_pipe_state_read);
                // ++q_pipe_state_read;
                // ++k_pipe_state_read;
                // ++g_pipe_state_read;
            }

            // --- Signal MMA that kg_all is ready ---
            // qkg_all_pipeline.producer_acquire(qkg_all_pipe_state_write);
            // (kg_all data already written to smem above)
            // qkg_all_pipeline.producer_commit(qkg_all_pipe_state_write);
            // ++qkg_all_pipe_state_write;

            // --- Wait for QK MMA result from MMA warp ---
            // qk_done_pipeline.consumer_wait(qk_done_pipe_state_read);
            // TODO: epilogue for QK result in tmem
            // qk_done_pipeline.consumer_release(qk_done_pipe_state_read);
            // ++qk_done_pipe_state_read;

            // --- Wait for KK MMA result from MMA warp ---
            // kk_done_pipeline.consumer_wait(kk_done_pipe_state_read);
            // TODO: epilogue for KK result in tmem
            // kk_done_pipeline.consumer_release(kk_done_pipe_state_read);
            // ++kk_done_pipe_state_read;

            // --- Wait for beta from Empty warps ---
            // beta_pipeline.consumer_wait(beta_pipe_state_read);
            // TODO: use beta_smem for epilogue
            // beta_pipeline.consumer_release(beta_pipe_state_read);
            // ++beta_pipe_state_read;

            // --- Signal Inverse warpgroup that KK is ready ---
            // kk_inv_pipeline.producer_acquire(kk_inv_pipe_state_write);
            // (KK data written to smem above)
            // kk_inv_pipeline.producer_commit(kk_inv_pipe_state_write);
            // ++kk_inv_pipe_state_write;
        }

    } else if (role == WarpRole::Mma) {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();

        if (elect_one_sync()) {
            // === PERSISTENT MMA LOOP (static scheduling, no tile pipeline) ===
            CUTE_NO_UNROLL
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int tid = tile_scheduler.get_current_tile_id();

                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx  = get<1>(blk_coord);
                int tile_idx  = get<2>(blk_coord);

                // --- Wait for kg_all from CE ---
                // qkg_all_pipeline.consumer_wait(qkg_all_pipe_state_read);

                // TODO: issue UMMA for QK and KK using kg_all smem

                // qkg_all_pipeline.consumer_release(qkg_all_pipe_state_read);
                // ++qkg_all_pipe_state_read;

                // --- Signal CE that QK result is ready (UMMA arrive) ---
                // qk_done_pipeline.producer_acquire(qk_done_pipe_state_write);
                // qk_done_pipeline.producer_commit(qk_done_pipe_state_write);
                // ++qk_done_pipe_state_write;

                // --- Signal CE that KK result is ready (UMMA arrive) ---
                // kk_done_pipeline.producer_acquire(kk_done_pipe_state_write);
                // kk_done_pipeline.producer_commit(kk_done_pipe_state_write);
                // ++kk_done_pipe_state_write;
            }
        }

    } else if (role == WarpRole::Load) {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();

        if (elect_one_sync()) {
            // === PERSISTENT LOAD LOOP (static scheduling, no tile pipeline) ===
            CUTE_NO_UNROLL
            for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
                int tid = tile_scheduler.get_current_tile_id();

                // Decode tile coordinates
                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx    = get<0>(blk_coord);
                int head_idx     = get<1>(blk_coord);
                int tile_idx     = get<2>(blk_coord);
                // int token_offset = cu_len_ptr[batch_idx];
                // int seq_len      = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
                // int sub_seq_len  = min(T_TILE, seq_len - tile_idx * T_TILE);

                // TMA load per-k_idx data (Q, K, G)
                for (int k_idx = 0; k_idx < K_ITERATION; ++k_idx) {
                    // Acquire Q/K/G smem slots (wait for consumers to release)
                    // q_pipeline.producer_acquire(q_pipe_state_write);
                    // k_pipeline.producer_acquire(k_pipe_state_write);
                    // g_pipeline.producer_acquire(g_pipe_state_write);

                    // TODO: issue TMA copies for Q, K, G into smem buffers
                    // cute::copy(tma_params.tma_q.with(...), ...)
                    // using BarrierType = typename PipelineQ::ProducerBarrierType;
                    // BarrierType* tma_barrier_q = q_pipeline.producer_get_barrier(q_pipe_state_write);
                    // BarrierType* tma_barrier_k = k_pipeline.producer_get_barrier(k_pipe_state_write);
                    // BarrierType* tma_barrier_g = g_pipeline.producer_get_barrier(g_pipe_state_write);

                    // Commit (for TMA pipeline, TMA hardware completes the barrier)
                    // q_pipeline.producer_commit(q_pipe_state_write, uint32_t(0));
                    // k_pipeline.producer_commit(k_pipe_state_write, uint32_t(0));
                    // g_pipeline.producer_commit(g_pipe_state_write, uint32_t(0));
                    // ++q_pipe_state_write;
                    // ++k_pipe_state_write;
                    // ++g_pipe_state_write;
                }
            }
        }

    } else if (role == WarpRole::Inverse) {
        cutlass::arch::warpgroup_reg_dealloc<REG_INVERSE>();

        // === PERSISTENT INVERSE LOOP (static scheduling, no tile pipeline) ===
        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx  = get<1>(blk_coord);
            int tile_idx  = get<2>(blk_coord);

            // --- Wait for KK from CE ---
            // kk_inv_pipeline.consumer_wait(kk_inv_pipe_state_read);

            // TODO: compute inverse of KK matrix

            // kk_inv_pipeline.consumer_release(kk_inv_pipe_state_read);
            // ++kk_inv_pipe_state_read;
        }

    } else {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();

        // === PERSISTENT EMPTY WARP LOOP (static scheduling, no tile pipeline) ===
        int empty_idx = threadIdx.x - (NUM_THREADS - 64); // 0..63

        CUTE_NO_UNROLL
        for (; tile_scheduler.is_valid(); tile_scheduler.advance()) {
            int tid = tile_scheduler.get_current_tile_id();

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx    = get<0>(blk_coord);
            int head_idx     = get<1>(blk_coord);
            int tile_idx     = get<2>(blk_coord);
            // int token_offset = cu_len_ptr[batch_idx];
            // int seq_len      = cu_len_ptr[batch_idx + 1] - cu_len_ptr[batch_idx];
            // int sub_seq_len  = min(T_TILE, seq_len - tile_idx * T_TILE);

            // --- Produce beta: acquire slot, write beta_smem, commit ---
            // beta_pipeline.producer_acquire(beta_pipe_state_write);

            // if (empty_idx < T_TILE) {
            //     shared_plan.beta_smem[beta_pipe_state_write.index()][empty_idx] = (empty_idx < sub_seq_len)
            //         ? reinterpret_cast<float*>(params.beta_ptr)[(token_offset + tile_idx * T_TILE + empty_idx) * params.h + head_idx]
            //         : float(0);
            // }
            // fence_view_async_shared();


            // beta_pipeline.producer_commit(beta_pipe_state_write);
            // ++beta_pipe_state_write;
        }
    }
    // === CLEANUP (once per CTA) ===
    __syncthreads();
    if (warp_idx == 0 && elect_one_sync()) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
    return;
}

void run_kda_fwd_intra_sm100(KDA_fwd_intra_params &params, cudaStream_t stream) {
    // KDA_ASSERT(params.d % 32 == 0);
    // int total_q_len = params.total_q_len;
    // int H = params.h;
    // int D = params.d;
    // int BT = params.chunk_size;

    // auto shape_QKG = make_shape(total_q_len, D, H);
    // auto stride_QKG = make_stride(H * D, _1{}, D);
    // auto tma_Q = cute::make_tma_copy(
    //     SM90_TMA_LOAD{},
    //     make_tensor(
    //         make_gmem_ptr((bf16*)params.q_ptr),
    //         make_layout(
    //             shape_QKG,
    //             stride_QKG
    //         )
    //     ),
    //     SmemLayoutInputBF16{}
    // );

    // auto tma_K = cute::make_tma_copy(
    //     SM90_TMA_LOAD{},
    //     make_tensor(
    //         make_gmem_ptr((bf16*)params.k_ptr),
    //         make_layout(
    //             shape_QKG,
    //             stride_QKG
    //         )
    //     ),
    //     SmemLayoutInputBF16{}
    // );

    // auto tma_G = cute::make_tma_copy(
    //     SM90_TMA_LOAD{},
    //     make_tensor(
    //         make_gmem_ptr((float*)params.g_ptr),
    //         make_layout(
    //             shape_QKG,
    //             stride_QKG
    //         )
    //     ),
    //     SmemLayoutInputFP32{}
    // );

    // TmaParams<
    //     decltype(shape_QKG), 
    //     decltype(tma_Q), decltype(tma_K), decltype(tma_G)
    // > tma_params = {
    //     shape_QKG,
    //     tma_Q,
    //     tma_K,
    //     tma_G,
    // };

    // auto kda_kernel = &kda_fwd_intra_sm100_kernel<decltype(tma_params)>;
    // constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    // CHECK_CUDA(cudaFuncSetAttribute(kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // dim3 grid_dim(TileScheduler::get_grid_shape(params.tile_scheduler_params));
    // dim3 block_dim(NUM_THREADS, 1, 1);
    // kda_kernel<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params);
    // return;
    sm100::run_kda_fwd_intra_sm100_v2(params, stream);
}

} // namespace sm100