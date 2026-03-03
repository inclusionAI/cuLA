#include "kda_fwd_common.cuh"
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

void run_kda_fwd_intra_sm100(KDA_fwd_intra_params &params, cudaStream_t stream) {
    KDA_ASSERT(params.d % 32 == 0);
    int total_q_len = params.total_q_len;
    int H = params.h;
    int D = params.d;
    int BT = params.chunk_size;

    auto shape_QKG = make_shape(total_q_len, D, H);
    auto stride_QKG = make_stride(H * D, _1{}, D);
}

} // namespace sm100