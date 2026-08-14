// KDA backward intra-chunk kernel for SM90 (Hopper) - v16
// 128 threads (4 warps), each warp handles one sub-chunk
// dA matrices cached in shared memory across K-iterations
// Persistent kernel: eliminates wave quantization overhead

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAException.h>

#include "kda/sm90/bwd/kda_config.h"

namespace sm90 {

constexpr int BT = 64;
constexpr int BC = 16;
constexpr int BK = 32;
constexpr int K_SIZE = 128;
constexpr int NC = BT / BC;
constexpr int NK = K_SIZE / BK;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = NC * WARP_SIZE;
constexpr int NT = BK / 8;
constexpr uint32_t TF32_MASK = 0xFFFFE000u;
constexpr int BK_S = BK + 4;
constexpr int BT_S = BT + 4;

struct WarpWork {
    float B_a[BC][BK_S];
    float B_b[BC][BK_S];
};

struct SmemLayout {
    __nv_bfloat16 q_s[BT][BK];
    __nv_bfloat16 k_s[BT][BK];
    float g_s[BT][BK_S];
    float beta_s[BT];
    float dAqk_cache[BT][BT_S];
    float dAkk_cache[BT][BT_S];
    WarpWork ww[NC];
    int s_tile_id;
};

__device__ __forceinline__ void
cp_async_16(void* smem, const void* global) {
    uint32_t sa = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(global));
}
__device__ __forceinline__ void
cp_async_8(void* smem, const void* global) {
    uint32_t sa = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(sa), "l"(global));
}
__device__ __forceinline__ float
bf2f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ float4
load_bf16x4(const __nv_bfloat16* p) {
    __nv_bfloat16 tmp[4];
    *reinterpret_cast<uint2*>(tmp) = *reinterpret_cast<const uint2*>(p);
    return {__bfloat162float(tmp[0]), __bfloat162float(tmp[1]), __bfloat162float(tmp[2]), __bfloat162float(tmp[3])};
}
__device__ __forceinline__ void
cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void
cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}
__device__ __forceinline__ void
st_global_cg_u32(void* addr, uint32_t val) {
    asm volatile("st.global.cg.u32 [%0], %1;\n" ::"l"(addr), "r"(val));
}
__device__ __forceinline__ void
st_global_cg_f32(void* addr, float val) {
    asm volatile("st.global.cg.f32 [%0], %1;\n" ::"l"(addr), "f"(val));
}
__device__ __forceinline__ void
st_global_cg_f32x2(void* addr, float v0, float v1) {
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};\n" ::"l"(addr), "f"(v0), "f"(v1));
}

__device__ __forceinline__ void
mma_m16n8k8_acc(
    float& c0,
    float& c1,
    float& c2,
    float& c3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void
matmul_1warp_2A_from_cache(
    float acc1[],
    float acc2[],
    const float cacheA1[][BT_S],
    const float cacheA2[][BT_S],
    int row_off,
    int col_off,
    const float B[][BK_S],
    int gid,
    int tid_in_grp,
    bool apply_mask,
    int sub_seq_len) {
    float a1[8], a2[8];
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        const int a_col = kk * 8 + 2 * tid_in_grp;
        a1[kk * 4 + 0] = cacheA1[row_off + gid][col_off + a_col];
        a1[kk * 4 + 1] = cacheA1[row_off + gid + 8][col_off + a_col];
        a1[kk * 4 + 2] = cacheA1[row_off + gid][col_off + a_col + 1];
        a1[kk * 4 + 3] = cacheA1[row_off + gid + 8][col_off + a_col + 1];
        a2[kk * 4 + 0] = cacheA2[row_off + gid][col_off + a_col];
        a2[kk * 4 + 1] = cacheA2[row_off + gid + 8][col_off + a_col];
        a2[kk * 4 + 2] = cacheA2[row_off + gid][col_off + a_col + 1];
        a2[kk * 4 + 3] = cacheA2[row_off + gid + 8][col_off + a_col + 1];
        if (apply_mask) {
            int col0 = a_col, col1 = a_col + 1;
            if (!(col0 <= gid && gid < sub_seq_len && col0 < sub_seq_len)) {
                a1[kk * 4 + 0] = 0.0f;
                a2[kk * 4 + 0] = 0.0f;
            }
            if (!(col0 <= gid + 8 && gid + 8 < sub_seq_len && col0 < sub_seq_len)) {
                a1[kk * 4 + 1] = 0.0f;
                a2[kk * 4 + 1] = 0.0f;
            }
            if (!(col1 <= gid && gid < sub_seq_len && col1 < sub_seq_len)) {
                a1[kk * 4 + 2] = 0.0f;
                a2[kk * 4 + 2] = 0.0f;
            }
            if (!(col1 <= gid + 8 && gid + 8 < sub_seq_len && col1 < sub_seq_len)) {
                a1[kk * 4 + 3] = 0.0f;
                a2[kk * 4 + 3] = 0.0f;
            }
        }
        a1[kk * 4 + 0] = __uint_as_float(__float_as_uint(a1[kk * 4 + 0]) & TF32_MASK);
        a1[kk * 4 + 1] = __uint_as_float(__float_as_uint(a1[kk * 4 + 1]) & TF32_MASK);
        a1[kk * 4 + 2] = __uint_as_float(__float_as_uint(a1[kk * 4 + 2]) & TF32_MASK);
        a1[kk * 4 + 3] = __uint_as_float(__float_as_uint(a1[kk * 4 + 3]) & TF32_MASK);
        a2[kk * 4 + 0] = __uint_as_float(__float_as_uint(a2[kk * 4 + 0]) & TF32_MASK);
        a2[kk * 4 + 1] = __uint_as_float(__float_as_uint(a2[kk * 4 + 1]) & TF32_MASK);
        a2[kk * 4 + 2] = __uint_as_float(__float_as_uint(a2[kk * 4 + 2]) & TF32_MASK);
        a2[kk * 4 + 3] = __uint_as_float(__float_as_uint(a2[kk * 4 + 3]) & TF32_MASK);
    }
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int n_base = nt << 3;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int b_k = kk * 8 + 2 * tid_in_grp;
            uint32_t ub0 = __float_as_uint(B[b_k][n_base + gid]) & TF32_MASK;
            uint32_t ub1 = __float_as_uint(B[b_k + 1][n_base + gid]) & TF32_MASK;
            mma_m16n8k8_acc(
                acc1[nt * 4],
                acc1[nt * 4 + 1],
                acc1[nt * 4 + 2],
                acc1[nt * 4 + 3],
                __float_as_uint(a1[kk * 4 + 0]),
                __float_as_uint(a1[kk * 4 + 1]),
                __float_as_uint(a1[kk * 4 + 2]),
                __float_as_uint(a1[kk * 4 + 3]),
                ub0,
                ub1);
            mma_m16n8k8_acc(
                acc2[nt * 4],
                acc2[nt * 4 + 1],
                acc2[nt * 4 + 2],
                acc2[nt * 4 + 3],
                __float_as_uint(a2[kk * 4 + 0]),
                __float_as_uint(a2[kk * 4 + 1]),
                __float_as_uint(a2[kk * 4 + 2]),
                __float_as_uint(a2[kk * 4 + 3]),
                ub0,
                ub1);
        }
    }
}

__device__ __forceinline__ void
matmul_1warp_2B_transA_from_cache(
    float acc[],
    const float cacheA1[][BT_S],
    const float cacheA2[][BT_S],
    int row_off,
    int col_off,
    const float B_x[][BK_S],
    const float B_y[][BK_S],
    int gid,
    int tid_in_grp,
    bool apply_mask,
    int sub_seq_len) {
    {
        float a[8];
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int a_col = kk * 8 + 2 * tid_in_grp;
            a[kk * 4 + 0] = cacheA1[row_off + a_col][col_off + gid];
            a[kk * 4 + 1] = cacheA1[row_off + a_col][col_off + gid + 8];
            a[kk * 4 + 2] = cacheA1[row_off + a_col + 1][col_off + gid];
            a[kk * 4 + 3] = cacheA1[row_off + a_col + 1][col_off + gid + 8];
            if (apply_mask) {
                int row_A0 = a_col, row_A1 = a_col + 1;
                if (!(gid <= row_A0 && row_A0 < sub_seq_len && gid < sub_seq_len))
                    a[kk * 4 + 0] = 0.0f;
                if (!(gid + 8 <= row_A0 && row_A0 < sub_seq_len && gid + 8 < sub_seq_len))
                    a[kk * 4 + 1] = 0.0f;
                if (!(gid <= row_A1 && row_A1 < sub_seq_len && gid < sub_seq_len))
                    a[kk * 4 + 2] = 0.0f;
                if (!(gid + 8 <= row_A1 && row_A1 < sub_seq_len && gid + 8 < sub_seq_len))
                    a[kk * 4 + 3] = 0.0f;
            }
            a[kk * 4 + 0] = __uint_as_float(__float_as_uint(a[kk * 4 + 0]) & TF32_MASK);
            a[kk * 4 + 1] = __uint_as_float(__float_as_uint(a[kk * 4 + 1]) & TF32_MASK);
            a[kk * 4 + 2] = __uint_as_float(__float_as_uint(a[kk * 4 + 2]) & TF32_MASK);
            a[kk * 4 + 3] = __uint_as_float(__float_as_uint(a[kk * 4 + 3]) & TF32_MASK);
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int n_base = nt << 3;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int b_k = kk * 8 + 2 * tid_in_grp;
                uint32_t ub0 = __float_as_uint(B_x[b_k][n_base + gid]) & TF32_MASK;
                uint32_t ub1 = __float_as_uint(B_x[b_k + 1][n_base + gid]) & TF32_MASK;
                mma_m16n8k8_acc(
                    acc[nt * 4],
                    acc[nt * 4 + 1],
                    acc[nt * 4 + 2],
                    acc[nt * 4 + 3],
                    __float_as_uint(a[kk * 4 + 0]),
                    __float_as_uint(a[kk * 4 + 1]),
                    __float_as_uint(a[kk * 4 + 2]),
                    __float_as_uint(a[kk * 4 + 3]),
                    ub0,
                    ub1);
            }
        }
    }
    {
        float a[8];
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int a_col = kk * 8 + 2 * tid_in_grp;
            a[kk * 4 + 0] = cacheA2[row_off + a_col][col_off + gid];
            a[kk * 4 + 1] = cacheA2[row_off + a_col][col_off + gid + 8];
            a[kk * 4 + 2] = cacheA2[row_off + a_col + 1][col_off + gid];
            a[kk * 4 + 3] = cacheA2[row_off + a_col + 1][col_off + gid + 8];
            if (apply_mask) {
                int row_A0 = a_col, row_A1 = a_col + 1;
                if (!(gid <= row_A0 && row_A0 < sub_seq_len && gid < sub_seq_len))
                    a[kk * 4 + 0] = 0.0f;
                if (!(gid + 8 <= row_A0 && row_A0 < sub_seq_len && gid + 8 < sub_seq_len))
                    a[kk * 4 + 1] = 0.0f;
                if (!(gid <= row_A1 && row_A1 < sub_seq_len && gid < sub_seq_len))
                    a[kk * 4 + 2] = 0.0f;
                if (!(gid + 8 <= row_A1 && row_A1 < sub_seq_len && gid + 8 < sub_seq_len))
                    a[kk * 4 + 3] = 0.0f;
            }
            a[kk * 4 + 0] = __uint_as_float(__float_as_uint(a[kk * 4 + 0]) & TF32_MASK);
            a[kk * 4 + 1] = __uint_as_float(__float_as_uint(a[kk * 4 + 1]) & TF32_MASK);
            a[kk * 4 + 2] = __uint_as_float(__float_as_uint(a[kk * 4 + 2]) & TF32_MASK);
            a[kk * 4 + 3] = __uint_as_float(__float_as_uint(a[kk * 4 + 3]) & TF32_MASK);
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int n_base = nt << 3;
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int b_k = kk * 8 + 2 * tid_in_grp;
                uint32_t ub0 = __float_as_uint(B_y[b_k][n_base + gid]) & TF32_MASK;
                uint32_t ub1 = __float_as_uint(B_y[b_k + 1][n_base + gid]) & TF32_MASK;
                mma_m16n8k8_acc(
                    acc[nt * 4],
                    acc[nt * 4 + 1],
                    acc[nt * 4 + 2],
                    acc[nt * 4 + 3],
                    __float_as_uint(a[kk * 4 + 0]),
                    __float_as_uint(a[kk * 4 + 1]),
                    __float_as_uint(a[kk * 4 + 2]),
                    __float_as_uint(a[kk * 4 + 3]),
                    ub0,
                    ub1);
            }
        }
    }
}

__device__ __forceinline__ void
load_block_cp_async(float dst[][BK_S], const float* src, int row_base, int stride, int tile_seq_len, int tid) {
#pragma unroll
    for (int pass = 0; pass < 4; pass++) {
        int f4_r = pass * 4 + (tid >> 3);
        int f4_c = (tid & 7) << 2;
        int r = row_base + f4_r;
        if (r < tile_seq_len) {
            cp_async_16(&dst[f4_r][f4_c], &src[r * stride + f4_c]);
        } else {
            float4 zero = {0, 0, 0, 0};
            *(float4*)(&dst[f4_r][f4_c]) = zero;
        }
    }
}

__global__ void
__launch_bounds__(BLOCK_THREADS, 3) kda_bwd_intra_sm90_kernel(const KDA_bwd_intra_params params) {
    extern __shared__ char shared_buf[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(shared_buf);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int tid = threadIdx.x % WARP_SIZE;
    const int i_i = warp_id;

    const int gid = tid >> 2;
    const int tid_in_grp = tid & 3;

    const int* chunk_indices = (const int*)params.chunk_indices_ptr;
    const int* cu_seqlens = (const int*)params.cu_seqlens_ptr;
    int* tile_counter = (int*)params.tile_counter_ptr;

    const int H = params.h;
    const int K = params.d;
    const int stride_qk = H * K;
    const int stride_dA = H * BT;
    const int stride_b = H;
    const int total_tiles = params.num_chunks * H;

    const __nv_bfloat16(*my_q)[BK] = &smem->q_s[i_i * BC];
    const __nv_bfloat16(*my_k)[BK] = &smem->k_s[i_i * BC];
    const float(*my_g)[BK_S] = (const float(*)[BK_S]) & smem->g_s[i_i * BC];
    WarpWork& ww = smem->ww[i_i];
    const int row0 = gid;
    const int row1 = gid + 8;

    while (true) {
        // ==================== PERSISTENT TILE DISPATCH ====================
        if (threadIdx.x == 0) {
            smem->s_tile_id = atomicAdd(tile_counter, 1);
        }
        __syncthreads();
        const int tile_id = smem->s_tile_id;
        if (tile_id >= total_tiles)
            return;

        const int i_t = tile_id / H;
        const int i_h = tile_id % H;

        const int batch_idx = chunk_indices[i_t * 2];
        const int seq_idx = chunk_indices[i_t * 2 + 1];
        const int start_offset = cu_seqlens[batch_idx];
        const int seq_len = cu_seqlens[batch_idx + 1] - start_offset;

        const int tile_seq_len = min(BT, seq_len - seq_idx * BT);

        const int tile_offset = start_offset + seq_idx * BT;

        const float* beta_base = (const float*)params.beta_ptr + tile_offset * stride_b + i_h;
        const float* dAqk_base = (const float*)params.dAqk_ptr + tile_offset * stride_dA + i_h * BT;
        const float* dAkk_base = (const float*)params.dAkk_ptr + tile_offset * stride_dA + i_h * BT;

        const int sub_seq_len = min(BC, tile_seq_len - i_i * BC);
        const bool warp_active = (sub_seq_len > 0);
        const int NC_actual = min(NC, (tile_seq_len + BC - 1) / BC);

        // Load beta (k-independent)
        if (threadIdx.x < BT) {
            smem->beta_s[threadIdx.x] = (threadIdx.x < tile_seq_len) ? beta_base[threadIdx.x * stride_b] : 0.0f;
        }

// ==================== dA CACHE LOAD ====================
#pragma unroll 1
        for (int elem4 = threadIdx.x; elem4 < BT * (BT >> 2); elem4 += BLOCK_THREADS) {
            int r = elem4 >> 4;
            int c = (elem4 & 15) << 2;
            if (r < tile_seq_len) {
                cp_async_16(&smem->dAqk_cache[r][c], &dAqk_base[r * stride_dA + c]);
                cp_async_16(&smem->dAkk_cache[r][c], &dAkk_base[r * stride_dA + c]);
            } else {
                float4 zero = {0, 0, 0, 0};
                *reinterpret_cast<float4*>(&smem->dAqk_cache[r][c]) = zero;
                *reinterpret_cast<float4*>(&smem->dAkk_cache[r][c]) = zero;
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // ==================== K-SLICE LOOP ====================
        for (int i_k = 0; i_k < NK; i_k++) {
            const int k_off = i_k * BK;

            // Load Q, K, G for this K-iteration
            {
                const __nv_bfloat16* q_base =
                    (const __nv_bfloat16*)params.q_ptr + tile_offset * stride_qk + i_h * K + k_off;
                const __nv_bfloat16* k_base =
                    (const __nv_bfloat16*)params.k_ptr + tile_offset * stride_qk + i_h * K + k_off;
                const float* g_base = (const float*)params.g_ptr + tile_offset * stride_qk + i_h * K + k_off;
                for (int elem = threadIdx.x; elem < BT * (BK / 8); elem += BLOCK_THREADS) {
                    int r = elem >> 2;
                    int c = (elem & 3) << 3;
                    if (r < tile_seq_len) {
                        int goff = r * stride_qk + c;
                        cp_async_16(&smem->q_s[r][c], &q_base[goff]);
                        cp_async_16(&smem->k_s[r][c], &k_base[goff]);
                    } else {
                        uint4 zero4 = {0, 0, 0, 0};
                        *reinterpret_cast<uint4*>(&smem->q_s[r][c]) = zero4;
                        *reinterpret_cast<uint4*>(&smem->k_s[r][c]) = zero4;
                    }
                }
                for (int elem = threadIdx.x; elem < BT * (BK / 4); elem += BLOCK_THREADS) {
                    int r = elem >> 3;
                    int c = (elem & 7) << 2;
                    if (r < tile_seq_len) {
                        cp_async_16(&smem->g_s[r][c], &g_base[r * stride_qk + c]);
                    } else {
                        float4 zero = {0, 0, 0, 0};
                        *reinterpret_cast<float4*>(&smem->g_s[r][c]) = zero;
                    }
                }
            }
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();

            const float* dq_base = (const float*)params.dq_ptr + tile_offset * stride_qk + i_h * K + k_off;
            const float* dk_base = (const float*)params.dk_ptr + tile_offset * stride_qk + i_h * K + k_off;
            const float* dg_base = (const float*)params.dg_ptr + tile_offset * stride_qk + i_h * K + k_off;
            __nv_bfloat16* dq_out = (__nv_bfloat16*)params.dq_out_ptr + tile_offset * stride_qk + i_h * K + k_off;
            __nv_bfloat16* dk_out = (__nv_bfloat16*)params.dk_out_ptr + tile_offset * stride_qk + i_h * K + k_off;
            float* db2_base = (float*)params.db2_ptr + i_k * params.total_q_len * H + tile_offset * stride_b + i_h;
            float* dg_out = (float*)params.dg_out_ptr + tile_offset * stride_qk + i_h * K + k_off;

            if (warp_active) {
                // ==================== FORWARD OFF-DIAGONAL ====================
                float dq2[16] = {0};
                float dk2[16] = {0};

                if (i_i > 0) {
#pragma unroll 1
                    for (int j = 0; j < i_i; j++) {
#pragma unroll
                        for (int pass = 0; pass < 4; pass++) {
                            int f4_r = pass * 4 + (tid >> 3);
                            int f4_c = (tid & 7) << 2;
                            float4 ks = load_bf16x4(&smem->k_s[j * BC + f4_r][f4_c]);
                            float4 gs_j = *reinterpret_cast<const float4*>(&smem->g_s[j * BC + f4_r][f4_c]);
                            float4 gs_gn = *reinterpret_cast<const float4*>(&my_g[0][f4_c]);
                            float4 ba;
                            ba.x = ks.x * exp2f(gs_gn.x - gs_j.x);
                            ba.y = ks.y * exp2f(gs_gn.y - gs_j.y);
                            ba.z = ks.z * exp2f(gs_gn.z - gs_j.z);
                            ba.w = ks.w * exp2f(gs_gn.w - gs_j.w);
                            *reinterpret_cast<float4*>(&ww.B_a[f4_r][f4_c]) = ba;
                        }
                        __syncwarp();
                        matmul_1warp_2A_from_cache(
                            dq2,
                            dk2,
                            smem->dAqk_cache,
                            smem->dAkk_cache,
                            i_i * BC,
                            j * BC,
                            ww.B_a,
                            gid,
                            tid_in_grp,
                            false,
                            BC);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        int col0 = nt * 8 + tid_in_grp * 2;
                        int col1 = col0 + 1;
                        float gn0 = my_g[0][col0], gn1 = my_g[0][col1];
                        float s00 = exp2f(my_g[row0][col0] - gn0);
                        float s01 = exp2f(my_g[row0][col1] - gn1);
                        float s10 = exp2f(my_g[row1][col0] - gn0);
                        float s11 = exp2f(my_g[row1][col1] - gn1);
                        dq2[nt * 4 + 0] *= s00;
                        dq2[nt * 4 + 1] *= s01;
                        dq2[nt * 4 + 2] *= s10;
                        dq2[nt * 4 + 3] *= s11;
                        dk2[nt * 4 + 0] *= s00;
                        dk2[nt * 4 + 1] *= s01;
                        dk2[nt * 4 + 2] *= s10;
                        dk2[nt * 4 + 3] *= s11;
                    }
                }

                // ==================== FORWARD DIAGONAL ====================
                {
                    int gn_row = min(BC / 2, sub_seq_len - 1);

#pragma unroll
                    for (int pass = 0; pass < 4; pass++) {
                        int f4_r = pass * 4 + (tid >> 3);
                        int f4_c = (tid & 7) << 2;
                        if (f4_r < sub_seq_len) {
                            float4 ks = load_bf16x4(&my_k[f4_r][f4_c]);
                            float4 gs_gn = *reinterpret_cast<const float4*>(&my_g[gn_row][f4_c]);
                            float4 gs_r = *reinterpret_cast<const float4*>(&my_g[f4_r][f4_c]);
                            float4 ba;
                            ba.x = ks.x * exp2f(gs_gn.x - gs_r.x);
                            ba.y = ks.y * exp2f(gs_gn.y - gs_r.y);
                            ba.z = ks.z * exp2f(gs_gn.z - gs_r.z);
                            ba.w = ks.w * exp2f(gs_gn.w - gs_r.w);
                            *reinterpret_cast<float4*>(&ww.B_a[f4_r][f4_c]) = ba;
                        } else {
                            float4 zero = {0, 0, 0, 0};
                            *reinterpret_cast<float4*>(&ww.B_a[f4_r][f4_c]) = zero;
                        }
                    }
                    __syncwarp();

                    float dqd[16] = {0};
                    float dkd[16] = {0};
                    matmul_1warp_2A_from_cache(
                        dqd,
                        dkd,
                        smem->dAqk_cache,
                        smem->dAkk_cache,
                        i_i * BC,
                        i_i * BC,
                        ww.B_a,
                        gid,
                        tid_in_grp,
                        true,
                        sub_seq_len);

                    // Start dq_in load into B_a (MMA is done reading B_a)
                    load_block_cp_async(ww.B_a, dq_base, i_i * BC, stride_qk, tile_seq_len, tid);
                    cp_async_commit();

#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        int col0 = nt * 8 + tid_in_grp * 2;
                        int col1 = col0 + 1;
                        float gn0 = my_g[gn_row][col0], gn1 = my_g[gn_row][col1];
                        float s00 = exp2f(my_g[row0][col0] - gn0);
                        float s01 = exp2f(my_g[row0][col1] - gn1);
                        float s10 = exp2f(my_g[row1][col0] - gn0);
                        float s11 = exp2f(my_g[row1][col1] - gn1);
                        dq2[nt * 4 + 0] += dqd[nt * 4 + 0] * s00;
                        dq2[nt * 4 + 1] += dqd[nt * 4 + 1] * s01;
                        dq2[nt * 4 + 2] += dqd[nt * 4 + 2] * s10;
                        dq2[nt * 4 + 3] += dqd[nt * 4 + 3] * s11;
                        dk2[nt * 4 + 0] += dkd[nt * 4 + 0] * s00;
                        dk2[nt * 4 + 1] += dkd[nt * 4 + 1] * s01;
                        dk2[nt * 4 + 2] += dkd[nt * 4 + 2] * s10;
                        dk2[nt * 4 + 3] += dkd[nt * 4 + 3] * s11;
                    }
                }

                // Wait for dq_in load
                cp_async_wait_all();
                __syncwarp();

                // Write dq_out = dq2 + dq_in, then reuse dq2 for dg_p = q * dq2
                {
                    int tile_r0 = i_i * BC + row0;
                    int tile_r1 = i_i * BC + row1;
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        int col0 = nt * 8 + tid_in_grp * 2;
                        if (row0 < sub_seq_len) {
                            __nv_bfloat162 pair = {
                                __float2bfloat16(dq2[nt * 4 + 0] + ww.B_a[row0][col0]),
                                __float2bfloat16(dq2[nt * 4 + 1] + ww.B_a[row0][col0 + 1])};
                            st_global_cg_u32(&dq_out[tile_r0 * stride_qk + col0], *reinterpret_cast<uint32_t*>(&pair));
                        }
                        if (row1 < sub_seq_len) {
                            __nv_bfloat162 pair = {
                                __float2bfloat16(dq2[nt * 4 + 2] + ww.B_a[row1][col0]),
                                __float2bfloat16(dq2[nt * 4 + 3] + ww.B_a[row1][col0 + 1])};
                            st_global_cg_u32(&dq_out[tile_r1 * stride_qk + col0], *reinterpret_cast<uint32_t*>(&pair));
                        }
                        dq2[nt * 4 + 0] = bf2f(my_q[row0][col0]) * dq2[nt * 4 + 0];
                        dq2[nt * 4 + 1] = bf2f(my_q[row0][col0 + 1]) * dq2[nt * 4 + 1];
                        dq2[nt * 4 + 2] = bf2f(my_q[row1][col0]) * dq2[nt * 4 + 2];
                        dq2[nt * 4 + 3] = bf2f(my_q[row1][col0 + 1]) * dq2[nt * 4 + 3];
                    }
                }

                // db reduction
                {
                    float db0 = 0, db1 = 0;
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        int col0 = nt * 8 + tid_in_grp * 2;
                        int col1 = col0 + 1;
                        db0 += dk2[nt * 4 + 0] * bf2f(my_k[row0][col0]) + dk2[nt * 4 + 1] * bf2f(my_k[row0][col1]);
                        db1 += dk2[nt * 4 + 2] * bf2f(my_k[row1][col0]) + dk2[nt * 4 + 3] * bf2f(my_k[row1][col1]);
                    }
                    db0 += __shfl_xor_sync(0xFFFFFFFF, db0, 1);
                    db0 += __shfl_xor_sync(0xFFFFFFFF, db0, 2);
                    db1 += __shfl_xor_sync(0xFFFFFFFF, db1, 1);
                    db1 += __shfl_xor_sync(0xFFFFFFFF, db1, 2);
                    if (tid_in_grp == 0) {
                        if (row0 < sub_seq_len)
                            st_global_cg_f32(&db2_base[(i_i * BC + row0) * stride_b], db0);
                        if (row1 < sub_seq_len)
                            st_global_cg_f32(&db2_base[(i_i * BC + row1) * stride_b], db1);
                    }
                }

                // Scale dk2 by beta
                {
                    float beta0 = smem->beta_s[i_i * BC + row0];
                    float beta1 = smem->beta_s[i_i * BC + row1];
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        dk2[nt * 4 + 0] *= beta0;
                        dk2[nt * 4 + 1] *= beta0;
                        dk2[nt * 4 + 2] *= beta1;
                        dk2[nt * 4 + 3] *= beta1;
                    }
                }

                // ==================== BACKWARD OFF-DIAGONAL ====================
                float dkt[16] = {0};

                if (i_i < NC_actual - 1) {
                    int gn_bwd_row = min(BC - 1, sub_seq_len - 1);

#pragma unroll 1
                    for (int j = i_i + 1; j < NC_actual; j++) {
                        int j_sub_seq = min(BC, tile_seq_len - j * BC);
#pragma unroll
                        for (int pass = 0; pass < 4; pass++) {
                            int f4_r = pass * 4 + (tid >> 3);
                            int f4_c = (tid & 7) << 2;
                            if (f4_r < j_sub_seq) {
                                float4 qs = load_bf16x4(&smem->q_s[j * BC + f4_r][f4_c]);
                                float4 ks = load_bf16x4(&smem->k_s[j * BC + f4_r][f4_c]);
                                float4 gs_j = *reinterpret_cast<const float4*>(&smem->g_s[j * BC + f4_r][f4_c]);
                                float4 gs_gn = *reinterpret_cast<const float4*>(&my_g[gn_bwd_row][f4_c]);
                                float beta_val = smem->beta_s[j * BC + f4_r];

                                float eg0 = exp2f(gs_j.x - gs_gn.x), eg1 = exp2f(gs_j.y - gs_gn.y),
                                      eg2 = exp2f(gs_j.z - gs_gn.z), eg3 = exp2f(gs_j.w - gs_gn.w);

                                float4 ba = {qs.x * eg0, qs.y * eg1, qs.z * eg2, qs.w * eg3};
                                float4 bb;
                                bb.x = __bfloat162float(__float2bfloat16(ks.x * beta_val)) * eg0;
                                bb.y = __bfloat162float(__float2bfloat16(ks.y * beta_val)) * eg1;
                                bb.z = __bfloat162float(__float2bfloat16(ks.z * beta_val)) * eg2;
                                bb.w = __bfloat162float(__float2bfloat16(ks.w * beta_val)) * eg3;
                                *reinterpret_cast<float4*>(&ww.B_a[f4_r][f4_c]) = ba;
                                *reinterpret_cast<float4*>(&ww.B_b[f4_r][f4_c]) = bb;
                            } else {
                                float4 zero = {0, 0, 0, 0};
                                *reinterpret_cast<float4*>(&ww.B_a[f4_r][f4_c]) = zero;
                                *reinterpret_cast<float4*>(&ww.B_b[f4_r][f4_c]) = zero;
                            }
                        }
                        __syncwarp();
                        matmul_1warp_2B_transA_from_cache(
                            dkt,
                            smem->dAqk_cache,
                            smem->dAkk_cache,
                            j * BC,
                            i_i * BC,
                            ww.B_a,
                            ww.B_b,
                            gid,
                            tid_in_grp,
                            false,
                            BC);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        int col0 = nt * 8 + tid_in_grp * 2;
                        int col1 = col0 + 1;
                        float gn0 = my_g[gn_bwd_row][col0], gn1 = my_g[gn_bwd_row][col1];
                        float s00 = exp2f(gn0 - my_g[row0][col0]);
                        float s01 = exp2f(gn1 - my_g[row0][col1]);
                        float s10 = exp2f(gn0 - my_g[row1][col0]);
                        float s11 = exp2f(gn1 - my_g[row1][col1]);
                        dkt[nt * 4 + 0] *= s00;
                        dkt[nt * 4 + 1] *= s01;
                        dkt[nt * 4 + 2] *= s10;
                        dkt[nt * 4 + 3] *= s11;
                    }
                }

                // ==================== BACKWARD DIAGONAL ====================
                {
                    int gn_row = min(BC / 2, sub_seq_len - 1);

#pragma unroll
                    for (int pass = 0; pass < 4; pass++) {
                        int f4_r = pass * 4 + (tid >> 3);
                        int f4_c = (tid & 7) << 2;
                        if (f4_r < sub_seq_len) {
                            float4 qs = load_bf16x4(&my_q[f4_r][f4_c]);
                            float4 ks = load_bf16x4(&my_k[f4_r][f4_c]);
                            float4 gs_r = *reinterpret_cast<const float4*>(&my_g[f4_r][f4_c]);
                            float4 gs_gn = *reinterpret_cast<const float4*>(&my_g[gn_row][f4_c]);
                            float beta_r = smem->beta_s[i_i * BC + f4_r];

                            float eg0 = exp2f(gs_r.x - gs_gn.x), eg1 = exp2f(gs_r.y - gs_gn.y),
                                  eg2 = exp2f(gs_r.z - gs_gn.z), eg3 = exp2f(gs_r.w - gs_gn.w);

                            float4 ba = {qs.x * eg0, qs.y * eg1, qs.z * eg2, qs.w * eg3};
                            float4 bb;
                            bb.x = __bfloat162float(__float2bfloat16(ks.x * beta_r)) * eg0;
                            bb.y = __bfloat162float(__float2bfloat16(ks.y * beta_r)) * eg1;
                            bb.z = __bfloat162float(__float2bfloat16(ks.z * beta_r)) * eg2;
                            bb.w = __bfloat162float(__float2bfloat16(ks.w * beta_r)) * eg3;
                            *reinterpret_cast<float4*>(&ww.B_a[f4_r][f4_c]) = ba;
                            *reinterpret_cast<float4*>(&ww.B_b[f4_r][f4_c]) = bb;
                        } else {
                            float4 zero = {0, 0, 0, 0};
                            *reinterpret_cast<float4*>(&ww.B_a[f4_r][f4_c]) = zero;
                            *reinterpret_cast<float4*>(&ww.B_b[f4_r][f4_c]) = zero;
                        }
                    }
                    __syncwarp();

                    float dktd[16] = {0};
                    matmul_1warp_2B_transA_from_cache(
                        dktd,
                        smem->dAqk_cache,
                        smem->dAkk_cache,
                        i_i * BC,
                        i_i * BC,
                        ww.B_a,
                        ww.B_b,
                        gid,
                        tid_in_grp,
                        true,
                        sub_seq_len);

                    // Start epilogue loads now (B_a/B_b no longer needed by MMA)
                    load_block_cp_async(ww.B_a, dk_base, i_i * BC, stride_qk, tile_seq_len, tid);
                    load_block_cp_async(ww.B_b, dg_base, i_i * BC, stride_qk, tile_seq_len, tid);
                    cp_async_commit();

#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        int col0 = nt * 8 + tid_in_grp * 2;
                        int col1 = col0 + 1;
                        float gn0 = my_g[gn_row][col0], gn1 = my_g[gn_row][col1];
                        float s00 = exp2f(gn0 - my_g[row0][col0]);
                        float s01 = exp2f(gn1 - my_g[row0][col1]);
                        float s10 = exp2f(gn0 - my_g[row1][col0]);
                        float s11 = exp2f(gn1 - my_g[row1][col1]);
                        dkt[nt * 4 + 0] += dktd[nt * 4 + 0] * s00;
                        dkt[nt * 4 + 1] += dktd[nt * 4 + 1] * s01;
                        dkt[nt * 4 + 2] += dktd[nt * 4 + 2] * s10;
                        dkt[nt * 4 + 3] += dktd[nt * 4 + 3] * s11;
                    }
                }

                // ==================== EPILOGUE ====================
                cp_async_wait_all();
                __syncwarp();

                {
                    int tile_r0 = i_i * BC + row0;
                    int tile_r1 = i_i * BC + row1;

#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        int col0 = nt * 8 + tid_in_grp * 2;
                        if (row0 < sub_seq_len) {
                            int off0 = tile_r0 * stride_qk + col0;
                            __nv_bfloat162 dk_pair = {
                                __float2bfloat16(ww.B_a[row0][col0] + dk2[nt * 4 + 0] + dkt[nt * 4 + 0]),
                                __float2bfloat16(ww.B_a[row0][col0 + 1] + dk2[nt * 4 + 1] + dkt[nt * 4 + 1])};
                            st_global_cg_u32(&dk_out[off0], *reinterpret_cast<uint32_t*>(&dk_pair));
                            st_global_cg_f32x2(
                                &dg_out[off0],
                                dq2[nt * 4 + 0] + (dk2[nt * 4 + 0] - dkt[nt * 4 + 0]) * bf2f(my_k[row0][col0]) +
                                    ww.B_b[row0][col0],
                                dq2[nt * 4 + 1] + (dk2[nt * 4 + 1] - dkt[nt * 4 + 1]) * bf2f(my_k[row0][col0 + 1]) +
                                    ww.B_b[row0][col0 + 1]);
                        }
                        if (row1 < sub_seq_len) {
                            int off0 = tile_r1 * stride_qk + col0;
                            __nv_bfloat162 dk_pair = {
                                __float2bfloat16(ww.B_a[row1][col0] + dk2[nt * 4 + 2] + dkt[nt * 4 + 2]),
                                __float2bfloat16(ww.B_a[row1][col0 + 1] + dk2[nt * 4 + 3] + dkt[nt * 4 + 3])};
                            st_global_cg_u32(&dk_out[off0], *reinterpret_cast<uint32_t*>(&dk_pair));
                            st_global_cg_f32x2(
                                &dg_out[off0],
                                dq2[nt * 4 + 2] + (dk2[nt * 4 + 2] - dkt[nt * 4 + 2]) * bf2f(my_k[row1][col0]) +
                                    ww.B_b[row1][col0],
                                dq2[nt * 4 + 3] + (dk2[nt * 4 + 3] - dkt[nt * 4 + 3]) * bf2f(my_k[row1][col0 + 1]) +
                                    ww.B_b[row1][col0 + 1]);
                        }
                    }
                }

            }  // warp_active

            __syncthreads();

        }  // k_idx loop

    }  // persistent while loop
}

void
run_kda_bwd_intra_sm90(KDA_bwd_intra_params& params, cudaStream_t stream) {
    constexpr size_t smem_size = sizeof(SmemLayout);
    auto kernel = &kda_bwd_intra_sm90_kernel;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int num_blocks_per_sm;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, BLOCK_THREADS, smem_size));

    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int num_sms;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

    int total_tiles = params.num_chunks * params.h;
    int num_blocks = min(num_sms * num_blocks_per_sm, total_tiles);

    int* tile_counter;
    C10_CUDA_CHECK(cudaMallocAsync(&tile_counter, sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(tile_counter, 0, sizeof(int), stream));
    params.tile_counter_ptr = tile_counter;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(BLOCK_THREADS, 1, 1);
    kernel<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaFreeAsync(tile_counter, stream));
}

}  // namespace sm90
