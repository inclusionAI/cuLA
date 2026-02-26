// KDA Backward WY dqkg fused kernel for SM100 (Blackwell architecture)
// This kernel fuses the WY backward computation with dq/dk/dg gradient computation
//
// Corresponds to: chunk_kda_bwd_kernel_wy_dqkg_fused in chunk_bwd.py

#include "kda_bwd_common.cuh"
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

// =============================================================================
// Constants and Configuration
// =============================================================================

constexpr int WY_T_TILE = 64;        // Chunk size
constexpr int WY_K_SIZE = 128;       // K dimension
constexpr int WY_V_SIZE = 128;       // V dimension
constexpr int WY_K_TILE = 32;        // K tile size
constexpr int WY_V_TILE = 64;        // V tile size
constexpr int WY_NUM_BUF = 2;        // Number of buffers for pipelining
constexpr int WY_NUM_THREADS = 128 * 3;  // Number of threads

// =============================================================================
// TMA Parameters Structure
// =============================================================================

template<
    typename ShapeQKG, 
    typename ShapeV,
    typename ShapeA,
    typename ShapeH,
    typename TMA_Q,
    typename TMA_K,
    typename TMA_V,
    typename TMA_V_NEW,
    typename TMA_G,
    typename TMA_BETA,
    typename TMA_A,
    typename TMA_H,
    typename TMA_DO,
    typename TMA_DH,
    typename TMA_DV>
struct WYTmaParams {
    ShapeQKG shape_qkg;
    ShapeV shape_v;
    ShapeA shape_a;
    ShapeH shape_h;
    TMA_Q tma_q;
    TMA_K tma_k;
    TMA_V tma_v;
    TMA_V_NEW tma_v_new;
    TMA_G tma_g;
    TMA_BETA tma_beta;
    TMA_A tma_A;
    TMA_H tma_h;
    TMA_DO tma_do;
    TMA_DH tma_dh;
    TMA_DV tma_dv;
};

// =============================================================================
// Shared Memory Layout Definitions
// =============================================================================

// Layout for BF16 inputs [T_TILE, K_TILE]
using WYSmemLayoutInputBF16_K = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<WY_T_TILE>, Int<WY_K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Layout for BF16 inputs [T_TILE, V_TILE]
using WYSmemLayoutInputBF16_V = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<WY_T_TILE>, Int<WY_V_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Layout for FP32 inputs [T_TILE, K_TILE]
using WYSmemLayoutInputFP32_K = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<WY_T_TILE>, Int<WY_K_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Layout for FP32 inputs [T_TILE, V_TILE]
using WYSmemLayoutInputFP32_V = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<WY_T_TILE>, Int<WY_V_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Layout for A matrix [T_TILE, BT] - bf16 (Akk matrix)
using WYSmemLayoutA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<WY_T_TILE>, Int<WY_T_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Layout for H matrix [K_TILE, V_TILE]
using WYSmemLayoutH = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<WY_K_TILE>, Int<WY_V_TILE>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// Layout for beta [T_TILE]
using WYSmemLayoutBeta = Layout<
    Shape<Int<WY_T_TILE>>,
    Stride<_1>
>;

// =============================================================================
// Shared Memory Plan
// =============================================================================

struct WYSharedMemoryPlan {
    // Input buffers
    array_aligned<bf16, cosize_v<WYSmemLayoutInputBF16_K>> q[WY_NUM_BUF];
    array_aligned<bf16, cosize_v<WYSmemLayoutInputBF16_K>> k[WY_NUM_BUF];
    array_aligned<bf16, cosize_v<WYSmemLayoutInputBF16_V>> v[WY_NUM_BUF];
    array_aligned<bf16, cosize_v<WYSmemLayoutInputBF16_V>> v_new[WY_NUM_BUF];
    array_aligned<float, cosize_v<WYSmemLayoutInputFP32_K>> g[WY_NUM_BUF];
    array_aligned<float, cosize_v<WYSmemLayoutBeta>> beta[WY_NUM_BUF];
    array_aligned<bf16, cosize_v<WYSmemLayoutA>> A[WY_NUM_BUF];  // bf16 (Akk matrix)
    
    // Hidden state buffers
    array_aligned<float, cosize_v<WYSmemLayoutH>> h[WY_NUM_BUF];
    array_aligned<float, cosize_v<WYSmemLayoutH>> dh[WY_NUM_BUF];
    
    // Gradient buffers
    array_aligned<bf16, cosize_v<WYSmemLayoutInputBF16_V>> do_buf[WY_NUM_BUF];
    array_aligned<bf16, cosize_v<WYSmemLayoutInputBF16_V>> dv[WY_NUM_BUF];

    // Barriers
    alignas(16) cute::uint64_t bar_load_qkg[WY_NUM_BUF];
    alignas(16) cute::uint64_t bar_load_v[WY_NUM_BUF];
    alignas(16) cute::uint64_t bar_load_A[WY_NUM_BUF];
    alignas(16) cute::uint64_t bar_load_h[WY_NUM_BUF];
    alignas(16) cute::uint64_t bar_compute_done[WY_NUM_BUF];
    alignas(16) cute::uint64_t bar_output_ready[WY_NUM_BUF];

    array_aligned<uint32_t, 1> tmem_start_addr;
};

// =============================================================================
// Warp Role Assignment
// =============================================================================

enum class WYWarpRole {
    Empty = 0x0, Load = 0x1, Mma = 0x2, Compute = 0x3, Epilogue = 0x4
};

static constexpr unsigned long long kWYWarpAssignment = 0x12'3333'4444ull;
static constexpr int kWYNumComputeWarps = 4;
static constexpr int kWYNumEpilogueWarps = 4;

__forceinline__ __device__ WYWarpRole wy_warp_idx_to_role(int warp_idx) {
    return static_cast<WYWarpRole>((kWYWarpAssignment >> (4 * warp_idx)) & 0xF);
}

// =============================================================================
// Tile Scheduler
// =============================================================================

using WYTileScheduler = NaiveTileScheduler;

// =============================================================================
// Main Kernel
// =============================================================================

template<typename TmaParams>
__global__ void __launch_bounds__(WY_NUM_THREADS, 1, 1) 
kda_bwd_wy_dqkg_fused_sm100_kernel(
    __grid_constant__ const KDA_bwd_wy_dqkg_fused_params params, 
    __grid_constant__ const TmaParams tma_params
) {
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warp = threadIdx.x % 32;
    auto role = wy_warp_idx_to_role(warp_idx);
    WYTileScheduler tile_scheduler{params.tile_scheduler_params};

    extern __shared__ char shared_buf[];
    WYSharedMemoryPlan *shared_plan = reinterpret_cast<WYSharedMemoryPlan*>(shared_buf);

    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_v.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_v_new.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_g.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_beta.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_A.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_h.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_do.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dh.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dv.get_tma_descriptor());
    }

    // Initialize barriers
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            for (int i = 0; i < WY_NUM_BUF; ++i) {
                cute::initialize_barrier(shared_plan->bar_load_qkg[i], 1);
                cute::initialize_barrier(shared_plan->bar_load_v[i], 1);
                cute::initialize_barrier(shared_plan->bar_load_A[i], 1);
                cute::initialize_barrier(shared_plan->bar_load_h[i], 1);
                cute::initialize_barrier(shared_plan->bar_compute_done[i], 128);
                cute::initialize_barrier(shared_plan->bar_output_ready[i], 128);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, shared_plan->tmem_start_addr.data());
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    int state_phase = 0;
    int buf_idx = 0;

    // TODO: Implement the kernel logic here
    // The following is a placeholder structure showing the expected flow:
    
    if (role == WYWarpRole::Compute) {
        // Compute warp: Handle compute operations
        // - Compute dq, dk, dv, db, dg, dA
        // - Fuse WY backward with gradient computation
        
    } else if (role == WYWarpRole::Epilogue) {
        // Epilogue warp: Handle output operations
        // - Write dq, dk, dv, db, dg, dA to global memory
        
    } else if (role == WYWarpRole::Mma) {
        // MMA warp: Handle matrix multiply operations
        // - dq += do @ h.T * exp2(g) * scale
        // - dk += v_new @ dh.T * exp2(gn - g)
        // - dA matrix operations
        
    } else if (role == WYWarpRole::Load) {
        // Load warp: Handle TMA loads
        // - Load q, k, v, v_new, g, beta, A, h, do, dh, dv
        
    }

    __syncthreads();
    
    if (warp_idx == 0 && elect_one_sync()) {
        cute::TMEM::Allocator1Sm().free(0, 512);
    }
    return;
}

// =============================================================================
// Host Launch Function
// =============================================================================

void run_kda_bwd_wy_dqkg_fused_sm100(KDA_bwd_wy_dqkg_fused_params &params, cudaStream_t stream) {
    KDA_ASSERT(params.d % 32 == 0);
    
    int total_q_len = params.total_q_len;
    int H = params.h;
    int K = params.d;
    int V = params.d_v;
    int BT = params.chunk_size;

    // Shape for Q, K, G tensors: [total_q_len, K, H]
    auto shape_QKG = make_shape(total_q_len, K, H);
    auto stride_QKG = make_stride(H * K, _1{}, K);

    // Shape for V tensors: [total_q_len, V, H]
    auto shape_V = make_shape(total_q_len, V, H);
    auto stride_V = make_stride(H * V, _1{}, V);

    // Shape for A tensor: [total_q_len, BT, H]
    auto shape_A = make_shape(total_q_len, BT, H);
    auto stride_A = make_stride(H * BT, _1{}, BT);

    // Shape for H tensor: [NT, K, V, H] - simplified for TMA
    int NT = (total_q_len + BT - 1) / BT; // Number of chunks
    auto shape_H = make_shape(NT * H, K, V);
    auto stride_H = make_stride(K * V, V, _1{});

    // Create TMA descriptors for input tensors
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q_ptr),
            make_layout(shape_QKG, stride_QKG)
        ),
        WYSmemLayoutInputBF16_K{}
    );

    auto tma_K = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.k_ptr),
            make_layout(shape_QKG, stride_QKG)
        ),
        WYSmemLayoutInputBF16_K{}
    );

    auto tma_V = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.v_ptr),
            make_layout(shape_V, stride_V)
        ),
        WYSmemLayoutInputBF16_V{}
    );

    auto tma_V_NEW = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.v_new_ptr),
            make_layout(shape_V, stride_V)
        ),
        WYSmemLayoutInputBF16_V{}
    );

    auto tma_G = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.g_ptr),
            make_layout(shape_QKG, stride_QKG)
        ),
        WYSmemLayoutInputFP32_K{}
    );

    // Beta is 1D per token: [total_q_len, H]
    auto shape_Beta = make_shape(total_q_len, H);
    auto stride_Beta = make_stride(H, _1{});
    auto tma_Beta = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.beta_ptr),
            make_layout(shape_Beta, stride_Beta)
        ),
        WYSmemLayoutBeta{}
    );

    auto tma_A = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.A_ptr),  // bf16 (Akk matrix)
            make_layout(shape_A, stride_A)
        ),
        WYSmemLayoutA{}
    );

    auto tma_H = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.h_ptr),
            make_layout(shape_H, stride_H)
        ),
        WYSmemLayoutH{}
    );

    auto tma_DO = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.do_ptr),
            make_layout(shape_V, stride_V)
        ),
        WYSmemLayoutInputBF16_V{}
    );

    auto tma_DH = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((float*)params.dh_ptr),
            make_layout(shape_H, stride_H)
        ),
        WYSmemLayoutH{}
    );

    auto tma_DV = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.dv_ptr),
            make_layout(shape_V, stride_V)
        ),
        WYSmemLayoutInputBF16_V{}
    );

    WYTmaParams<
        decltype(shape_QKG), decltype(shape_V), decltype(shape_A), decltype(shape_H),
        decltype(tma_Q), decltype(tma_K), decltype(tma_V), decltype(tma_V_NEW),
        decltype(tma_G), decltype(tma_Beta), decltype(tma_A), decltype(tma_H),
        decltype(tma_DO), decltype(tma_DH), decltype(tma_DV)
    > tma_params = {
        shape_QKG,
        shape_V,
        shape_A,
        shape_H,
        tma_Q,
        tma_K,
        tma_V,
        tma_V_NEW,
        tma_G,
        tma_Beta,
        tma_A,
        tma_H,
        tma_DO,
        tma_DH,
        tma_DV,
    };

    auto kda_kernel = &kda_bwd_wy_dqkg_fused_sm100_kernel<decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(WYSharedMemoryPlan);
    
    CHECK_CUDA(cudaFuncSetAttribute(kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid_dim(WYTileScheduler::get_grid_shape(params.tile_scheduler_params));
    printf("grid_dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    dim3 block_dim(WY_NUM_THREADS, 1, 1);
    
    kda_kernel<<<grid_dim, block_dim, smem_size, stream>>>(params, tma_params);
    return;
}

} // namespace sm100
