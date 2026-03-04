#include "kda_fwd_common.cuh"
#include "kda_bwd/helpers.h"
#include "kda_bwd/gemm.h"
#include "kda_bwd/utils.h"

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
//   warp  8     (thread 256-287): Mma              — 1 warp, uses elect_one_sync
//   warp  9     (thread 288-319): Load             — 1 warp, uses elect_one_sync
//   warp 10-13  (thread 320-447): Inverse          — 1 warpgroup (128 threads) for inv(KK)
//   warp 14-15  (thread 448-511): Empty            - 2 warps, used for beta loading
static constexpr unsigned long long kWarpAssignment = 0x66'6612'5555'5555ull;

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
    // Q,K,G, double buffer, 32MB
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> q[NUM_BUF_VALUE];
    array_aligned<bf16, cosize_v<SmemLayoutInputBF16>> k[NUM_BUF_VALUE];
    array_aligned<float, cosize_v<SmemLayoutInputFP32>> g[NUM_BUF_VALUE];

    // gated MMA K^T, single buffer, 20MB
    struct {
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32<1>>> inter[6];
        array_aligned<tf32, cosize_v<SmemLayoutMatBTF32<1>>> intra[4];
    } kg_all;

    // inv(KK), single buffer, 8MB
    array_aligned<fp16, cosize_v<SmemLayoutOutputFP16>> kk[NUM_BUF_A];

    // pipeline
    // load
    alignas(16) cute::uint64_t bar_load_g_ready[NUM_BUF_VALUE], bar_load_k_ready[NUM_BUF_VALUE], bar_load_q_ready[NUM_BUF_VALUE];
    alignas(16) cute::uint64_t bar_load_tile_ready[NUM_BUF_A];
    alignas(16) cute::uint64_t bar_load_beta_ready[NUM_BUF_A];
    // CUDA Core (prologue) -> MMA
    alignas(16) cute::uint64_t bar_qkg_all_ready, bar_ktg_inter_ready, bar_ktg_intra_ready;
    // MMA -> CUDA Core (epilogue)
    alignas(16) cute::uint64_t bar_qk_done, bar_kk_done;
    // CUDA Core (epilogue) -> Inverse
    alignas(16) cute::uint64_t bar_kk_inv_ready;

    alignas(16) __nv_bfloat16 beta_smem[2][T_TILE]; // double-buffered per-tile beta, indexed by A_phase
    int tile_id[2]; // double-buffered persistent tile ID (written by Load warp, read by all)
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TileScheduler = NaiveTileScheduler;

template <typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
kda_fwd_intra_sm100_kernel(__grid_constant__ const KDA_fwd_intra_params params, __grid_constant__ const TmaParams tma_params) {
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warp = threadIdx.x % 32;
    auto role = warp_idx_to_role(warp_idx);
    TileScheduler tile_scheduler(params.tile_scheduler_params);

    extern __shared__ char shared_buf[];
    SharedMemoryPlan *shared_plan = reinterpret_cast<SharedMemoryPlan*>(shared_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            for (int i = 0; i < NUM_BUF_VALUE; ++i) {
                cute::initialize_barrier(shared_plan->bar_load_g_ready[i], 1);
                cute::initialize_barrier(shared_plan->bar_load_k_ready[i], 1);
                cute::initialize_barrier(shared_plan->bar_load_q_ready[i], 1);
            }
            cute::initialize_barrier(shared_plan->bar_qkg_all_ready, 256); // CE(256) -> MMA
            cute::initialize_barrier(shared_plan->bar_ktg_inter_ready, 256); // CE(256) -> MMA
            cute::initialize_barrier(shared_plan->bar_ktg_intra_ready, 256); // CE(256) -> MMA
            cute::initialize_barrier(shared_plan->bar_qk_done, 1); // MMA(1) -> CE
            cute::initialize_barrier(shared_plan->bar_kk_done, 1); // MMA(1) -> CE
            cute::initialize_barrier(shared_plan->bar_kk_inv_ready, 256); // CE(256) -> Inverse
            for (int i = 0; i < NUM_BUF_A; ++i) {
                cute::initialize_barrier(shared_plan->bar_load_tile_ready[i], 1); // Load(1) -> All
                cute::initialize_barrier(shared_plan->bar_load_beta_ready[i], 64); // repurposed: beta load by Empty warps (2 warps = 64 threads)
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, shared_plan->tmem_start_addr.data());
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    // After fence barrier init
    // Note : It must be composed with an appropriate sync instruction with the right scope
    // to ensure visibility eg. __syncthreads() or a cluster_arrive() + cluster_wait()
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
            // Wait for Load warp to write tile_id
            cute::wait_barrier(shared_plan->bar_load_tile_ready[buf_idx_A], A_phase);
            int tid = shared_plan->tile_id[A_phase];
            if (tid >= total_tiles) break;

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);

            // update A phase for next tile
            state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
            buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
        }

    } else if (role == WarpRole::Mma) {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
        if (elect_one_sync()) {
            // === PERSISTENT MMA LOOP ===
            for (;;) {
                int A_phase = (state_phase >> (buf_idx_A + NUM_BUF_VALUE)) & 1;
                // Wait for Load warp to write tile_id
                cute::wait_barrier(shared_plan->bar_load_tile_ready[buf_idx_A], A_phase);
                int tid = shared_plan->tile_id[A_phase];
                if (tid >= total_tiles) break;

                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);

                // update A phase for next tile
                state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
                buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
            }
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
                    // Signal CE+Empty+Inverse: atomic arrive
                    cute::arrive_barrier(shared_plan->bar_load_tile_ready[buf_idx_A]);
                    break;
                }
                // signal other warps that tile_id is ready
                cute::arrive_barrier(shared_plan->bar_load_tile_ready[buf_idx_A]);

                // Decode tile coordinates
                auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
                int batch_idx = get<0>(blk_coord);
                int head_idx = get<1>(blk_coord);
                int tile_idx = get<2>(blk_coord);

                // update A phase for next tile
                state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
                buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
            }
        }

    } else if (role == WarpRole::Inverse) {
        cutlass::arch::warpgroup_reg_dealloc<REG_INVERSE>();
        // === PERSISTENT INVERSE LOOP ===
        for (;;) {
            int A_phase = (state_phase >> (buf_idx_A + NUM_BUF_VALUE)) & 1;
            // Wait for Load warp to write tile_id
            cute::wait_barrier(shared_plan->bar_load_tile_ready[buf_idx_A], A_phase);
            int tid = shared_plan->tile_id[A_phase];
            if (tid >= total_tiles) break;

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);

            // update A phase for next tile
            state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
            buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
        }

    } else {
        cutlass::arch::warpgroup_reg_dealloc<REG_LOAD>();
        // === PERSISTENT EMPTY WARP LOOP (beta loading) ===
        int empty_idx = threadIdx.x - (NUM_THREADS - 64); // 0..63
        for (;;) {
            int A_phase = (state_phase >> (buf_idx_A + NUM_BUF_VALUE)) & 1;
            // Wait for Load warp to write tile_id
            cute::wait_barrier(shared_plan->bar_load_tile_ready[buf_idx_A], A_phase);
            int tid = shared_plan->tile_id[A_phase];
            if (tid >= total_tiles) break;

            auto blk_coord = TileScheduler::decode_tile_coord(tid, params.h, chunk_indices_ptr, cu_len_ptr);
            int batch_idx = get<0>(blk_coord);
            int head_idx = get<1>(blk_coord);
            int tile_idx = get<2>(blk_coord);

            // update A phase for next tile
            state_phase ^= 1 << (buf_idx_A + NUM_BUF_VALUE);
            buf_idx_A = (buf_idx_A + 1) % NUM_BUF_A;
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

    TmaParams<
        decltype(shape_QKG), 
        decltype(tma_Q), decltype(tma_K), decltype(tma_G)
    > tma_params = {
        shape_QKG,
        tma_Q,
        tma_K,
        tma_G,
    };

    auto kda_kernel = &kda_fwd_intra_sm100_kernel<decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_dim(TileScheduler::get_grid_shape(params.tile_scheduler_params));
    dim3 block_dim(NUM_THREADS, 1, 1);
    kda_kernel<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params);
    return;
}

} // namespace sm100