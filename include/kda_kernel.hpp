#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>
#include "kda_math.hpp"
#include "kda_params.hpp"
#include "kda_tensor.hpp"

namespace kda {
namespace kernel {
namespace prefill {

namespace wmma = nvcuda::wmma;

template <typename T>
__forceinline__ __device__ float row_l2_norm_sq(
    const T* __restrict__ data,
    int row,
    int dim) {
    float norm_sq = 0.0f;
    const int base = row * dim;
    #pragma unroll
    for (int d = 0; d < dim; ++d) {
        float val = static_cast<float>(data[base + d]);
        norm_sq += val * val;
    }
    return norm_sq;
}

template <int ROWS, int DIM, typename T>
__forceinline__ __device__ void compute_sigmoid_prefix_products(
    const T* __restrict__ data,
    float* __restrict__ prefix_products,
    int valid_rows,
    int tid,
    int num_threads) {
    for (int d = tid; d < DIM; d += num_threads) {
        float prefix = 1.0f;
        #pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            if (row < valid_rows) {
                prefix *= static_cast<float>(math::sigmoid(data[row * DIM + d]));
                prefix_products[row * DIM + d] = prefix;
            } else {
                prefix_products[row * DIM + d] = 0.0f;
            }
        }
    }
    __syncthreads();
}

template <int ROWS, int DIM, typename T>
__forceinline__ __device__ void normalize_rows_inplace(
    T* __restrict__ data,
    float* __restrict__ row_scales,
    int valid_rows,
    float extra_scale,
    int tid,
    int num_threads) {
    for (int row = tid; row < ROWS; row += num_threads) {
        if (row >= valid_rows) {
            row_scales[row] = 0.0f;
            continue;
        }
        float norm_sq = row_l2_norm_sq(data, row, DIM);
        row_scales[row] = rsqrtf(norm_sq + 1.0e-6f) * extra_scale;
    }
    __syncthreads();

    for (int idx = tid; idx < ROWS * DIM; idx += num_threads) {
        int row = idx / DIM;
        if (row >= valid_rows) {
            data[idx] = static_cast<T>(0);
            continue;
        }
        float val = static_cast<float>(data[idx]) * row_scales[row];
        data[idx] = static_cast<T>(val);
    }
}

template <typename T>
__forceinline__ __device__ void copy_async_global_to_shared(
    T* __restrict__ dst,
    const T* __restrict__ src) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned dst_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
            "r"(dst_addr), "l"(src), "n"(sizeof(T)));
    #else
    *dst = *src;
    #endif
}

__forceinline__ __device__ void cp_async_commit_group() {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
    #endif
}

__forceinline__ __device__ void cp_async_wait_group_0() {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
    #endif
}

template <typename T>
__forceinline__ __device__ float row_l2_norm_sq_strided(
    const T* __restrict__ data,
    int row,
    int dim,
    int stride) {
    float norm_sq = 0.0f;
    const int base = row * stride;
    #pragma unroll
    for (int d = 0; d < dim; ++d) {
        float val = static_cast<float>(data[base + d]);
        norm_sq += val * val;
    }
    return norm_sq;
}

template <int ROWS, int DIM, typename T>
__forceinline__ __device__ void normalize_rows_inplace_strided(
    T* __restrict__ data,
    float* __restrict__ row_scales,
    int valid_rows,
    float extra_scale,
    int tid,
    int num_threads,
    int stride) {
    for (int row = tid; row < ROWS; row += num_threads) {
        if (row >= valid_rows) {
            row_scales[row] = 0.0f;
            continue;
        }
        float norm_sq = row_l2_norm_sq_strided(data, row, DIM, stride);
        row_scales[row] = rsqrtf(norm_sq + 1.0e-6f) * extra_scale;
    }
    __syncthreads();

    for (int idx = tid; idx < ROWS * DIM; idx += num_threads) {
        int row = idx / DIM;
        int d = idx % DIM;
        if (row >= valid_rows) {
            data[row * stride + d] = static_cast<T>(0);
            continue;
        }
        float val = static_cast<float>(data[row * stride + d]) * row_scales[row];
        data[row * stride + d] = static_cast<T>(val);
    }
}

template <typename T>
__forceinline__ __device__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T>
__forceinline__ __device__ void normalize_rows_warp_32x64(
    T* __restrict__ data,
    float* __restrict__ row_scales,
    int valid_rows,
    float extra_scale,
    int tid,
    int stride) {
    constexpr int rows = 32;
    constexpr int dim = 64;
    constexpr int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane = tid % warp_size;
    const int num_warps = blockDim.x / warp_size;

    for (int row = warp_id; row < rows; row += num_warps) {
        float norm_sq = 0.0f;
        if (row < valid_rows) {
            #pragma unroll
            for (int d = lane; d < dim; d += warp_size) {
                float val = static_cast<float>(data[row * stride + d]);
                norm_sq += val * val;
            }
        }
        norm_sq = warp_sum<float>(norm_sq);
        if (lane == 0) {
            row_scales[row] = (row < valid_rows)
                ? rsqrtf(norm_sq + 1.0e-6f) * extra_scale
                : 0.0f;
        }
    }
    __syncthreads();

    for (int idx = tid; idx < rows * dim; idx += blockDim.x) {
        int row = idx / dim;
        int d = idx % dim;
        if (row >= valid_rows) {
            data[row * stride + d] = static_cast<T>(0);
            continue;
        }
        float val = static_cast<float>(data[row * stride + d]) * row_scales[row];
        data[row * stride + d] = static_cast<T>(val);
    }
}

template <typename T>
__forceinline__ __device__ void compute_sigmoid_prefix_products_warp_32x64(
    const T* __restrict__ data,
    float* __restrict__ prefix_products,
    int valid_rows,
    int tid,
    int data_stride,
    int out_stride) {
    constexpr int dim = 64;
    constexpr int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane = tid % warp_size;
    const int num_warps = blockDim.x / warp_size;

    for (int d = warp_id; d < dim; d += num_warps) {
        float value = (lane < valid_rows)
            ? static_cast<float>(math::sigmoid(data[lane * data_stride + d]))
            : 1.0f;
        #pragma unroll
        for (int offset = 1; offset < warp_size; offset <<= 1) {
            float prev = __shfl_up_sync(0xffffffffu, value, offset);
            if (lane >= offset) {
                value *= prev;
            }
        }
        prefix_products[lane * out_stride + d] = (lane < valid_rows) ? value : 0.0f;
    }
    __syncthreads();
}

template <int ROWS, int DIM, typename T>
__forceinline__ __device__ void compute_sigmoid_prefix_products_strided(
    const T* __restrict__ data,
    float* __restrict__ prefix_products,
    int valid_rows,
    int tid,
    int num_threads,
    int data_stride,
    int out_stride) {
    for (int d = tid; d < DIM; d += num_threads) {
        float prefix = 1.0f;
        #pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            if (row < valid_rows) {
                prefix *= static_cast<float>(math::sigmoid(data[row * data_stride + d]));
                prefix_products[row * out_stride + d] = prefix;
            } else {
                prefix_products[row * out_stride + d] = 0.0f;
            }
        }
    }
    __syncthreads();
}

template <int C, int K_DIM, typename T>
__forceinline__ __device__ void compute_intra_chunk_kkt_strided(
    const T* __restrict__ s_K,
    const float* __restrict__ s_G_prefix,
    const T* __restrict__ s_Beta,
    float* __restrict__ s_M,
    int valid_c,
    int tid,
    int num_threads,
    int k_stride,
    int g_stride,
    int m_stride) {
    const int total_elements = C * C;
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        const int row = idx / C;
        const int col = idx % C;

        if (row >= valid_c || col >= valid_c || row <= col) {
            s_M[row * m_stride + col] = 0.0f;
            continue;
        }

        float dot_product = 0.0f;
        #pragma unroll
        for (int d = 0; d < K_DIM; ++d) {
            float k_i = static_cast<float>(s_K[row * k_stride + d]);
            float k_j = static_cast<float>(s_K[col * k_stride + d]);
            float decay = s_G_prefix[row * g_stride + d] / s_G_prefix[col * g_stride + d];
            dot_product = fmaf(k_i * k_j, decay, dot_product);
        }

        float beta_i = static_cast<float>(math::softplus(s_Beta[row]));
        s_M[row * m_stride + col] = dot_product * beta_i;
    }
}

template <int C>
__forceinline__ __device__ void invert_lower_triangular_strided(
    float* __restrict__ s_M,
    float* __restrict__ row_cache,
    int valid_c,
    int tid,
    int num_threads,
    int m_stride) {
    for (int i = tid; i < C; i += num_threads) {
        s_M[i * m_stride + i] = (i < valid_c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 1; i < valid_c; ++i) {
        for (int k = tid; k < i; k += num_threads) {
            row_cache[k] = s_M[i * m_stride + k];
        }
        __syncthreads();

        for (int j = tid; j < i; j += num_threads) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < C; ++k) {
                if (k < j || k >= i) {
                    continue;
                }
                float m_kj = (k == j) ? 1.0f : s_M[k * m_stride + j];
                sum += row_cache[k] * m_kj;
            }
            s_M[i * m_stride + j] = -sum;
        }
        __syncthreads();
    }
}

template <int C, int K_DIM, typename T>
__forceinline__ __device__ void compute_W_and_U_local_strided(
    const float* __restrict__ s_M,
    const T* __restrict__ s_K,
    const T* __restrict__ s_V,
    const T* __restrict__ s_G,
    T* __restrict__ s_W,
    T* __restrict__ s_U,
    int valid_c,
    int v_local_dim,
    int tid,
    int num_threads,
    int m_stride,
    int k_stride,
    int v_stride) {
    const int total_w = C * K_DIM;
    for (int idx = tid; idx < total_w; idx += num_threads) {
        const int row = idx / K_DIM;
        const int d = idx % K_DIM;
        if (row >= valid_c) {
            s_W[row * k_stride + d] = static_cast<T>(0);
            continue;
        }

        float w_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j <= row; ++j) {
            float m_val = s_M[row * m_stride + j];
            float k_val = static_cast<float>(s_K[j * k_stride + d]);
            float gamma_val = static_cast<float>(math::swish(s_G[j * k_stride + d]));
            w_acc = fmaf(m_val, gamma_val * k_val, w_acc);
        }
        s_W[row * k_stride + d] = static_cast<T>(w_acc);
    }

    const int total_u = C * v_local_dim;
    for (int idx = tid; idx < total_u; idx += num_threads) {
        const int row = idx / v_local_dim;
        const int d = idx % v_local_dim;
        if (row >= valid_c) {
            s_U[row * v_stride + d] = static_cast<T>(0);
            continue;
        }

        float u_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j <= row; ++j) {
            float m_val = s_M[row * m_stride + j];
            float v_val = static_cast<float>(s_V[j * v_stride + d]);
            u_acc = fmaf(m_val, v_val, u_acc);
        }
        s_U[row * v_stride + d] = static_cast<T>(u_acc);
    }
}

template <int C, int K_DIM, int V_DIM, typename T, typename Params>
__forceinline__ __device__ void prefetch_chunk_stage_async(
    T* __restrict__ s_K,
    T* __restrict__ s_Q,
    T* __restrict__ s_G,
    T* __restrict__ s_V,
    T* __restrict__ s_Beta,
    const Params& params,
    int batch_idx,
    int head_idx,
    int chunk_idx,
    int v_begin,
    int v_local_dim,
    int tid,
    int num_threads,
    int k_stride,
    int v_stride) {
    const int start_token = chunk_idx * C;
    const int remain = params.seq_len - start_token;
    const int valid_c = (remain > 0) ? ((remain < C) ? remain : C) : 0;
    const std::size_t qk_base =
        ((static_cast<std::size_t>(batch_idx) * params.num_heads + head_idx) * params.seq_len + start_token) *
        K_DIM;
    const std::size_t v_base =
        ((static_cast<std::size_t>(batch_idx) * params.num_heads + head_idx) * params.seq_len + start_token) *
        V_DIM;
    const std::size_t beta_base =
        ((static_cast<std::size_t>(batch_idx) * params.num_heads + head_idx) * params.seq_len + start_token);

    for (int idx = tid; idx < C * K_DIM; idx += num_threads) {
        int row = idx / K_DIM;
        int d = idx % K_DIM;
        if (row < valid_c) {
            copy_async_global_to_shared(&s_K[row * k_stride + d], &params.k_ptr[qk_base + row * K_DIM + d]);
            copy_async_global_to_shared(&s_Q[row * k_stride + d], &params.q_ptr[qk_base + row * K_DIM + d]);
            copy_async_global_to_shared(&s_G[row * k_stride + d], &params.g_ptr[qk_base + row * K_DIM + d]);
        } else {
            s_K[row * k_stride + d] = static_cast<T>(0);
            s_Q[row * k_stride + d] = static_cast<T>(0);
            s_G[row * k_stride + d] = static_cast<T>(0);
        }
    }

    for (int idx = tid; idx < C * v_local_dim; idx += num_threads) {
        int row = idx / v_local_dim;
        int d = idx % v_local_dim;
        if (row < valid_c) {
            copy_async_global_to_shared(
                &s_V[row * v_stride + d],
                &params.v_ptr[v_base + row * V_DIM + (v_begin + d)]);
        } else {
            s_V[row * v_stride + d] = static_cast<T>(0);
        }
    }

    for (int row = tid; row < C; row += num_threads) {
        if (row < valid_c) {
            copy_async_global_to_shared(&s_Beta[row], &params.beta_ptr[beta_base + row]);
        } else {
            s_Beta[row] = static_cast<T>(0);
        }
    }

    cp_async_commit_group();
}

// Strictly lower-triangular KKT dot products inside the chunk.
template <int C, int K_DIM, typename T>
__forceinline__ __device__ void compute_intra_chunk_kkt(
    const T* __restrict__ s_K,
    const float* __restrict__ s_G_prefix,
    const T* __restrict__ s_Beta,
    float* __restrict__ s_M,
    int valid_c,
    int tid,
    int num_threads) {
    int total_elements = C * C;
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        int row = idx / C;
        int col = idx % C;

        if (row >= valid_c || col >= valid_c || row <= col) {
            s_M[row * C + col] = 0.0f;
            continue;
        }

        float dot_product = 0.0f;
        #pragma unroll
        for (int d = 0; d < K_DIM; ++d) {
            float k_i = static_cast<float>(s_K[row * K_DIM + d]);
            float k_j = static_cast<float>(s_K[col * K_DIM + d]);
            const float prefix_row = s_G_prefix[row * K_DIM + d];
            const float prefix_col = s_G_prefix[col * K_DIM + d];
            float decay = prefix_row / prefix_col;
            dot_product = fmaf(k_i * k_j, decay, dot_product);
        }

        float beta_i = static_cast<float>(math::softplus(s_Beta[row]));
        s_M[row * C + col] = dot_product * beta_i;
    }
}

// In-place inverse of (I + L) via forward substitution; result in s_M.
template <int C>
__forceinline__ __device__ void invert_lower_triangular(
    float* __restrict__ s_M,
    float* __restrict__ row_cache,
    int valid_c,
    int tid,
    int num_threads) {
    for (int i = tid; i < C; i += num_threads) {
        s_M[i * C + i] = (i < valid_c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 1; i < valid_c; ++i) {
        for (int k = tid; k < i; k += num_threads) {
            row_cache[k] = s_M[i * C + k];
        }
        __syncthreads();

        for (int j = tid; j < i; j += num_threads) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < C; ++k) {
                if (k < j || k >= i) {
                    continue;
                }
                float m_kj = (k == j) ? 1.0f : s_M[k * C + j];
                sum += row_cache[k] * m_kj;
            }
            s_M[i * C + j] = -sum;
        }
        __syncthreads();
    }
}

// Effective W/U from inverted system; write chunk slices to global w/u.
template <int C, int K_DIM, int V_DIM, typename T, typename Params>
__forceinline__ __device__ void compute_W_and_U_and_store(
    const float* __restrict__ s_M,
    const T* __restrict__ s_K,
    const T* __restrict__ s_V,
    const T* __restrict__ s_G,
    Params& params,
    int chunk_idx, int head_idx, int batch_idx,
    int valid_c,
    int tid,
    int num_threads) {
    T* W_global = tensor::get_chunk_slice<C, K_DIM>(
        params.w_ptr, params.batch_size, params.num_heads, params.num_chunks, batch_idx, head_idx, chunk_idx);
    T* U_global = tensor::get_chunk_slice<C, V_DIM>(
        params.u_ptr, params.batch_size, params.num_heads, params.num_chunks, batch_idx, head_idx, chunk_idx);

    int total_w = C * K_DIM;
    for (int idx = tid; idx < total_w; idx += num_threads) {
        int row = idx / K_DIM;
        int d   = idx % K_DIM;

        if (row >= valid_c) {
            W_global[row * K_DIM + d] = static_cast<T>(0);
            continue;
        }

        float w_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j <= row; ++j) {
            float m_val = s_M[row * C + j];
            int mem_idx = j * K_DIM + d;
            float k_val = static_cast<float>(s_K[mem_idx]);
            float gamma_val = static_cast<float>(math::swish(s_G[mem_idx]));
            w_acc = fmaf(m_val, gamma_val * k_val, w_acc);
        }
        W_global[row * K_DIM + d] = static_cast<T>(w_acc);
    }

    int total_u = C * V_DIM;
    for (int idx = tid; idx < total_u; idx += num_threads) {
        int row = idx / V_DIM;
        int d = idx % V_DIM;
        if (row >= valid_c) {
            U_global[row * V_DIM + d] = static_cast<T>(0);
            continue;
        }
        float u_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j <= row; ++j) {
            float m_val = s_M[row * C + j];
            float v_val = static_cast<float>(s_V[j * V_DIM + d]);
            u_acc = fmaf(m_val, v_val, u_acc);
        }
        U_global[row * V_DIM + d] = static_cast<T>(u_acc);
    }
}

template <typename T>
__forceinline__ __device__ void wmma_store_accumulator_row_major(
    float* __restrict__ dst,
    const wmma::fragment<wmma::accumulator, 16, 16, 8, float>& frag) {
    wmma::store_matrix_sync(dst, frag, 16, wmma::mem_row_major);
}

template <typename T>
__forceinline__ __device__ void inter_chunk_wmma_64x16(
    float* __restrict__ s_S_matrix,
    T* __restrict__ s_Q,
    T* __restrict__ s_W,
    T* __restrict__ s_U,
    float* __restrict__ s_G_prefix,
    float* __restrict__ s_tile_scratch,
    T* __restrict__ o_ptr,
    std::size_t o_base,
    int valid_c,
    int v_begin,
    int tid,
    int v_stride,
    int k_stride) {
    const int warp_id = tid / 32;

    for (int idx = tid; idx < 32 * 64; idx += blockDim.x) {
        int row = idx / 64;
        int k = idx % 64;
        if (row < valid_c) {
            s_Q[row * k_stride + k] =
                static_cast<T>(static_cast<float>(s_Q[row * k_stride + k]) *
                               s_G_prefix[row * k_stride + k]);
        } else {
            s_Q[row * k_stride + k] = static_cast<T>(0);
        }
    }
    __syncthreads();

    if (warp_id < 2) {
        const int row_tile = warp_id * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        #pragma unroll
        for (int kk = 0; kk < 64; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
            const float* a_ptr = reinterpret_cast<const float*>(s_Q) + row_tile * k_stride + kk;
            const float* b_ptr = s_S_matrix + kk * v_stride;
            wmma::load_matrix_sync(a_frag, a_ptr, k_stride);
            wmma::load_matrix_sync(b_frag, b_ptr, v_stride);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        wmma_store_accumulator_row_major<T>(s_tile_scratch + row_tile * 16, acc);
    }
    __syncthreads();

    for (int idx = tid; idx < 32 * 16; idx += blockDim.x) {
        int row = idx / 16;
        int v_local = idx % 16;
        if (row < valid_c) {
            float out = s_tile_scratch[row * 16 + v_local] +
                        static_cast<float>(s_U[row * v_stride + v_local]);
            o_ptr[o_base + row * 64 + (v_begin + v_local)] = static_cast<T>(out);
        }
    }
    __syncthreads();

    if (warp_id < 4) {
        const int row_tile = warp_id * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        #pragma unroll
        for (int kk = 0; kk < 32; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
            const float* a_ptr = reinterpret_cast<const float*>(s_W) + kk * k_stride + row_tile;
            const float* b_ptr = reinterpret_cast<const float*>(s_U) + kk * v_stride;
            wmma::load_matrix_sync(a_frag, a_ptr, k_stride);
            wmma::load_matrix_sync(b_frag, b_ptr, v_stride);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        wmma_store_accumulator_row_major<T>(s_tile_scratch + row_tile * 16, acc);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
        int k = idx / 16;
        int v_local = idx % 16;
        float decay_val = s_G_prefix[(valid_c - 1) * k_stride + k];
        s_S_matrix[k * v_stride + v_local] =
            s_S_matrix[k * v_stride + v_local] * decay_val + s_tile_scratch[idx];
    }
    __syncthreads();
}

template <int C, int K_DIM, int V_DIM, typename T, typename Params>
__forceinline__ __device__ void compute_W_and_U_and_store_strided(
    const float* __restrict__ s_M,
    const T* __restrict__ s_K,
    const T* __restrict__ s_V,
    const T* __restrict__ s_G,
    Params& params,
    int chunk_idx, int head_idx, int batch_idx,
    int valid_c,
    int tid,
    int num_threads,
    int m_stride,
    int k_stride,
    int v_stride) {
    T* W_global = tensor::get_chunk_slice<C, K_DIM>(
        params.w_ptr, params.batch_size, params.num_heads, params.num_chunks, batch_idx, head_idx, chunk_idx);
    T* U_global = tensor::get_chunk_slice<C, V_DIM>(
        params.u_ptr, params.batch_size, params.num_heads, params.num_chunks, batch_idx, head_idx, chunk_idx);

    int total_w = C * K_DIM;
    for (int idx = tid; idx < total_w; idx += num_threads) {
        int row = idx / K_DIM;
        int d = idx % K_DIM;
        if (row >= valid_c) {
            W_global[row * K_DIM + d] = static_cast<T>(0);
            continue;
        }

        float w_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j <= row; ++j) {
            float m_val = s_M[row * m_stride + j];
            float k_val = static_cast<float>(s_K[j * k_stride + d]);
            float gamma_val = static_cast<float>(math::swish(s_G[j * k_stride + d]));
            w_acc = fmaf(m_val, gamma_val * k_val, w_acc);
        }
        W_global[row * K_DIM + d] = static_cast<T>(w_acc);
    }

    int total_u = C * V_DIM;
    for (int idx = tid; idx < total_u; idx += num_threads) {
        int row = idx / V_DIM;
        int d = idx % V_DIM;
        if (row >= valid_c) {
            U_global[row * V_DIM + d] = static_cast<T>(0);
            continue;
        }

        float u_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j <= row; ++j) {
            float m_val = s_M[row * m_stride + j];
            float v_val = static_cast<float>(s_V[j * v_stride + d]);
            u_acc = fmaf(m_val, v_val, u_acc);
        }
        U_global[row * V_DIM + d] = static_cast<T>(u_acc);
    }
}

// Intra-chunk: KKT, invert, W/U, write intermediates.
template <typename Params>
__global__ void kda_intra_chunk_kernel(Params params) {
    static_assert(Params::V_DIM >= Params::K_DIM,
                  "temporary intra-chunk prefix buffer requires V_DIM >= K_DIM");
    constexpr int k_stride = Params::K_DIM + 1;
    constexpr int v_stride = Params::V_DIM + 1;
    constexpr int m_stride = Params::C + 1;
    int chunk_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int batch_idx = blockIdx.z;
    int tid = threadIdx.x;        
    int num_threads = blockDim.x;  

    extern __shared__ char smem[];
    using T = typename Params::element_type; 
    
    T* s_K     = reinterpret_cast<T*>(smem);
    T* s_V     = s_K + (Params::C * k_stride);
    T* s_G     = s_V + (Params::C * v_stride);
    T* s_Beta  = s_G + (Params::C * k_stride);
    float* s_M = reinterpret_cast<float*>(s_Beta + Params::C);
    float* s_row_scales = s_M + (Params::C * m_stride);
    float* s_G_prefix = reinterpret_cast<float*>(s_V);

    const int start_token = chunk_idx * Params::C;
    const int remain = params.seq_len - start_token;
    const int valid_c = (remain > 0) ? ((remain < Params::C) ? remain : Params::C) : 0;
    if (valid_c <= 0) return;

    const int k_base = ((batch_idx * params.num_heads + head_idx) * params.seq_len + start_token) * Params::K_DIM;
    const int v_base = ((batch_idx * params.num_heads + head_idx) * params.seq_len + start_token) * Params::V_DIM;
    const int b_base = ((batch_idx * params.num_heads + head_idx) * params.seq_len + start_token);

    for (int idx = tid; idx < Params::C * Params::K_DIM; idx += num_threads) {
        int r = idx / Params::K_DIM;
        int d = idx % Params::K_DIM;
        if (r < valid_c) {
            copy_async_global_to_shared(&s_K[r * k_stride + d], &params.k_ptr[k_base + r * Params::K_DIM + d]);
            copy_async_global_to_shared(&s_G[r * k_stride + d], &params.g_ptr[k_base + r * Params::K_DIM + d]);
        } else {
            s_K[r * k_stride + d] = static_cast<T>(0);
            s_G[r * k_stride + d] = static_cast<T>(0);
        }
    }
    for (int r = tid; r < Params::C; r += num_threads) {
        if (r < valid_c) {
            copy_async_global_to_shared(&s_Beta[r], &params.beta_ptr[b_base + r]);
        } else {
            s_Beta[r] = static_cast<T>(0);
        }
    }
    cp_async_commit_group();
    cp_async_wait_group_0();
    __syncthreads();

    normalize_rows_inplace_strided<Params::C, Params::K_DIM, T>(
        s_K, s_row_scales, valid_c, 1.0f, tid, num_threads, k_stride);
    __syncthreads();

    compute_sigmoid_prefix_products_strided<Params::C, Params::K_DIM, T>(
        s_G, s_G_prefix, valid_c, tid, num_threads, k_stride, v_stride);
    compute_intra_chunk_kkt_strided<Params::C, Params::K_DIM, T>(
        s_K, s_G_prefix, s_Beta, s_M, valid_c, tid, num_threads, k_stride, v_stride, m_stride);
    __syncthreads(); 

    invert_lower_triangular_strided<Params::C>(
        s_M, s_row_scales, valid_c, tid, num_threads, m_stride);
    __syncthreads(); 

    for (int idx = tid; idx < Params::C * Params::V_DIM; idx += num_threads) {
        int r = idx / Params::V_DIM;
        int d = idx % Params::V_DIM;
        if (r < valid_c) {
            copy_async_global_to_shared(&s_V[r * v_stride + d], &params.v_ptr[v_base + r * Params::V_DIM + d]);
        } else {
            s_V[r * v_stride + d] = static_cast<T>(0);
        }
    }
    cp_async_commit_group();
    cp_async_wait_group_0();
    __syncthreads();

    compute_W_and_U_and_store_strided<Params::C, Params::K_DIM, Params::V_DIM, T, Params>(
        s_M, s_K, s_V, s_G, params, chunk_idx, head_idx, batch_idx, valid_c, tid, num_threads,
        m_stride, k_stride, v_stride);
}

// Inter-chunk RNN: hidden state S in shared memory, serial over chunks.
template <typename Params>
__global__ void kda_inter_chunk_rnn_kernel(Params params) {
    constexpr int k_stride = Params::K_DIM + 1;
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int shard_idx = blockIdx.z;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem[];
    using T = typename Params::element_type;

    const int v_tile = params.inter_v_tile;
    const int v_begin = shard_idx * v_tile;
    const int v_end = (v_begin + v_tile < Params::V_DIM) ? (v_begin + v_tile) : Params::V_DIM;
    const int v_local_dim = (v_end > v_begin) ? (v_end - v_begin) : 0;
    if (v_local_dim <= 0) return;

    float* s_S_matrix = reinterpret_cast<float*>(smem);  // [K_DIM, v_tile+1]
    T* s_Q = reinterpret_cast<T*>(s_S_matrix + (Params::K_DIM * (v_tile + 1)));
    T* s_W = s_Q + (Params::C * k_stride);
    T* s_U = s_W + (Params::C * k_stride);  // [C, v_tile+1]
    float* s_row_scales = reinterpret_cast<float*>(s_U + (Params::C * (v_tile + 1)));
    float* s_G_prefix = s_row_scales + Params::C;
    float* s_tile_scratch = s_G_prefix + (Params::C * k_stride);

    int S_elements = Params::K_DIM * (v_tile + 1);
    for (int i = tid; i < S_elements; i += num_threads) {
        s_S_matrix[i] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < params.num_chunks; ++t) {
        const int start_token = t * Params::C;
        const int remain = params.seq_len - start_token;
        const int valid_c = (remain > 0) ? ((remain < Params::C) ? remain : Params::C) : 0;
        if (valid_c <= 0) break;

        const std::size_t q_base =
            ((static_cast<std::size_t>(batch_idx) * params.num_heads + head_idx) * params.seq_len + start_token) *
            Params::K_DIM;
        const T* W_global = tensor::get_chunk_slice<Params::C, Params::K_DIM>(
            params.w_ptr, params.batch_size, params.num_heads, params.num_chunks, batch_idx, head_idx, t);
        const T* U_global = tensor::get_chunk_slice<Params::C, Params::V_DIM>(
            params.u_ptr, params.batch_size, params.num_heads, params.num_chunks, batch_idx, head_idx, t);

        for (int idx = tid; idx < Params::C * Params::K_DIM; idx += num_threads) {
            int r = idx / Params::K_DIM;
            int d = idx % Params::K_DIM;
            if (r < valid_c) {
                copy_async_global_to_shared(&s_Q[r * k_stride + d], &params.q_ptr[q_base + r * Params::K_DIM + d]);
                copy_async_global_to_shared(&s_W[r * k_stride + d], &W_global[r * Params::K_DIM + d]);
            } else {
                s_Q[r * k_stride + d] = static_cast<T>(0);
                s_W[r * k_stride + d] = static_cast<T>(0);
            }
        }
        for (int idx = tid; idx < Params::C * v_tile; idx += num_threads) {
            int r = idx / v_tile;
            int d_local = idx % v_tile;
            int d_global = v_begin + d_local;
            if (r < valid_c && d_global < Params::V_DIM) {
                copy_async_global_to_shared(
                    &s_U[r * (v_tile + 1) + d_local], &U_global[r * Params::V_DIM + d_global]);
            } else {
                s_U[r * (v_tile + 1) + d_local] = static_cast<T>(0);
            }
        }
        cp_async_commit_group();
        cp_async_wait_group_0();
        __syncthreads();

        if constexpr (Params::C == 32 && Params::K_DIM == 64) {
            normalize_rows_warp_32x64(
                s_Q,
                s_row_scales,
                valid_c,
                rsqrtf(static_cast<float>(Params::K_DIM)),
                tid,
                k_stride);
        } else {
            normalize_rows_inplace_strided<Params::C, Params::K_DIM, T>(
                s_Q,
                s_row_scales,
                valid_c,
                rsqrtf(static_cast<float>(Params::K_DIM)),
                tid,
                num_threads,
                k_stride);
        }
        __syncthreads();

        if constexpr (Params::C == 32 && Params::K_DIM == 64) {
            compute_sigmoid_prefix_products_warp_32x64(
                params.g_ptr + q_base, s_G_prefix, valid_c, tid, Params::K_DIM, k_stride);
        } else {
            compute_sigmoid_prefix_products_strided<Params::C, Params::K_DIM, T>(
                params.g_ptr + q_base, s_G_prefix, valid_c, tid, num_threads, Params::K_DIM, k_stride);
        }

        const std::size_t o_base =
            ((static_cast<std::size_t>(batch_idx) * params.num_heads + head_idx) * params.seq_len + start_token) *
            Params::V_DIM;

        if constexpr (std::is_same_v<T, float> && Params::C == 32 && Params::K_DIM == 64 && Params::V_DIM == 64) {
            if (v_local_dim == 16) {
                inter_chunk_wmma_64x16<T>(
                    s_S_matrix,
                    s_Q,
                    s_W,
                    s_U,
                    s_G_prefix,
                    s_tile_scratch,
                    params.o_ptr,
                    o_base,
                    valid_c,
                    v_begin,
                    tid,
                    v_tile + 1,
                    k_stride);
            } else {
                int O_elements = Params::C * v_local_dim;
                for (int idx = tid; idx < O_elements; idx += num_threads) {
                    int row = idx / v_local_dim;
                    int v_local = idx % v_local_dim;
                    int v_global = v_begin + v_local;

                    if (row >= valid_c) continue;
                    float o_acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < Params::K_DIM; ++k) {
                        float q_val = static_cast<float>(s_Q[row * k_stride + k]);
                        float s_val = s_S_matrix[k * (v_tile + 1) + v_local];
                        float prefix_decay = s_G_prefix[row * k_stride + k];
                        o_acc = fmaf(q_val, s_val * prefix_decay, o_acc);
                    }
                    o_acc += static_cast<float>(s_U[row * (v_tile + 1) + v_local]);
                    params.o_ptr[o_base + row * Params::V_DIM + v_global] = static_cast<T>(o_acc);
                }
                __syncthreads();

                for (int idx = tid; idx < S_elements; idx += num_threads) {
                    int k = idx / (v_tile + 1);
                    int v_local = idx % (v_tile + 1);
                    if (k >= Params::K_DIM || v_local >= v_local_dim) {
                        continue;
                    }
                    float decay_val = s_G_prefix[(valid_c - 1) * k_stride + k];

                    float update = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < valid_c; ++c) {
                        float w_val = static_cast<float>(s_W[c * k_stride + k]);
                        float u_val = static_cast<float>(s_U[c * (v_tile + 1) + v_local]);
                        update = fmaf(w_val, u_val, update);
                    }
                    s_S_matrix[k * (v_tile + 1) + v_local] =
                        (s_S_matrix[k * (v_tile + 1) + v_local] * decay_val) + update;
                }
                __syncthreads();
            }
        } else {
            int O_elements = Params::C * v_local_dim;
            for (int idx = tid; idx < O_elements; idx += num_threads) {
                int row = idx / v_local_dim;
                int v_local = idx % v_local_dim;
                int v_global = v_begin + v_local;

                if (row >= valid_c) continue;
                float o_acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < Params::K_DIM; ++k) {
                    float q_val = static_cast<float>(s_Q[row * k_stride + k]);
                    float s_val = s_S_matrix[k * (v_tile + 1) + v_local];
                    float prefix_decay = s_G_prefix[row * k_stride + k];
                    o_acc = fmaf(q_val, s_val * prefix_decay, o_acc);
                }
                o_acc += static_cast<float>(s_U[row * (v_tile + 1) + v_local]);
                params.o_ptr[o_base + row * Params::V_DIM + v_global] = static_cast<T>(o_acc);
            }
            __syncthreads();

            for (int idx = tid; idx < S_elements; idx += num_threads) {
                int k = idx / (v_tile + 1);
                int v_local = idx % (v_tile + 1);
                if (k >= Params::K_DIM || v_local >= v_local_dim) {
                    continue;
                }
                float decay_val = s_G_prefix[(valid_c - 1) * k_stride + k];

                float update = 0.0f;
                #pragma unroll
                for (int c = 0; c < valid_c; ++c) {
                    float w_val = static_cast<float>(s_W[c * k_stride + k]);
                    float u_val = static_cast<float>(s_U[c * (v_tile + 1) + v_local]);
                    update = fmaf(w_val, u_val, update);
                }
                s_S_matrix[k * (v_tile + 1) + v_local] =
                    (s_S_matrix[k * (v_tile + 1) + v_local] * decay_val) + update;
            }
            __syncthreads();
        }
    }
}

template <typename Params>
__global__ void kda_fused_prefill_kernel(Params params) {
    static_assert(Params::K_DIM == 64 && Params::V_DIM == 64 && Params::C == 32,
                  "fused prefill kernel currently specializes 64x64x32 only");
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int shard_idx = blockIdx.z;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    using T = typename Params::element_type;
    constexpr int k_stride = Params::K_DIM + 1;
    constexpr int m_stride = Params::C + 1;
    const int v_tile = params.inter_v_tile;
    const int v_stride = v_tile + 1;
    const int v_begin = shard_idx * v_tile;
    const int v_end = (v_begin + v_tile < Params::V_DIM) ? (v_begin + v_tile) : Params::V_DIM;
    const int v_local_dim = (v_end > v_begin) ? (v_end - v_begin) : 0;
    if (v_local_dim <= 0) {
        return;
    }

    extern __shared__ char smem_raw[];
    char* smem = smem_raw;

    T* s_K_stage0 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * k_stride;
    T* s_K_stage1 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * k_stride;
    T* s_Q_stage0 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * k_stride;
    T* s_Q_stage1 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * k_stride;
    T* s_G_stage0 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * k_stride;
    T* s_G_stage1 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * k_stride;
    T* s_V_stage0 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * v_stride;
    T* s_V_stage1 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * v_stride;
    T* s_Beta_stage0 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C;
    T* s_Beta_stage1 = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C;

    float* s_S_matrix = reinterpret_cast<float*>(smem);
    smem += sizeof(float) * Params::K_DIM * v_stride;
    float* s_M = reinterpret_cast<float*>(smem);
    smem += sizeof(float) * Params::C * m_stride;
    float* s_G_prefix = reinterpret_cast<float*>(smem);
    smem += sizeof(float) * Params::C * k_stride;
    float* s_row_scales = reinterpret_cast<float*>(smem);
    smem += sizeof(float) * Params::C;
    T* s_W = reinterpret_cast<T*>(smem);
    smem += sizeof(T) * Params::C * k_stride;
    T* s_U = reinterpret_cast<T*>(smem);

    T* s_K_stages[2] = {s_K_stage0, s_K_stage1};
    T* s_Q_stages[2] = {s_Q_stage0, s_Q_stage1};
    T* s_G_stages[2] = {s_G_stage0, s_G_stage1};
    T* s_V_stages[2] = {s_V_stage0, s_V_stage1};
    T* s_Beta_stages[2] = {s_Beta_stage0, s_Beta_stage1};

    const int S_elements = Params::K_DIM * v_stride;
    for (int i = tid; i < S_elements; i += num_threads) {
        s_S_matrix[i] = 0.0f;
    }
    __syncthreads();

    if (params.num_chunks <= 0) {
        return;
    }

    int current_stage = 0;
    prefetch_chunk_stage_async<Params::C, Params::K_DIM, Params::V_DIM, T>(
        s_K_stages[current_stage],
        s_Q_stages[current_stage],
        s_G_stages[current_stage],
        s_V_stages[current_stage],
        s_Beta_stages[current_stage],
        params,
        batch_idx,
        head_idx,
        0,
        v_begin,
        v_local_dim,
        tid,
        num_threads,
        k_stride,
        v_stride);
    cp_async_wait_group_0();
    __syncthreads();

    for (int t = 0; t < params.num_chunks; ++t) {
        const int start_token = t * Params::C;
        const int remain = params.seq_len - start_token;
        const int valid_c = (remain > 0) ? ((remain < Params::C) ? remain : Params::C) : 0;
        if (valid_c <= 0) {
            break;
        }

        T* s_K = s_K_stages[current_stage];
        T* s_Q = s_Q_stages[current_stage];
        T* s_G = s_G_stages[current_stage];
        T* s_V = s_V_stages[current_stage];
        T* s_Beta = s_Beta_stages[current_stage];

        normalize_rows_inplace_strided<Params::C, Params::K_DIM, T>(
            s_K, s_row_scales, valid_c, 1.0f, tid, num_threads, k_stride);
        __syncthreads();

        compute_sigmoid_prefix_products_strided<Params::C, Params::K_DIM, T>(
            s_G, s_G_prefix, valid_c, tid, num_threads, k_stride, k_stride);
        compute_intra_chunk_kkt_strided<Params::C, Params::K_DIM, T>(
            s_K, s_G_prefix, s_Beta, s_M, valid_c, tid, num_threads, k_stride, k_stride, m_stride);
        __syncthreads();

        invert_lower_triangular_strided<Params::C>(
            s_M, s_row_scales, valid_c, tid, num_threads, m_stride);
        __syncthreads();

        compute_W_and_U_local_strided<Params::C, Params::K_DIM, T>(
            s_M, s_K, s_V, s_G, s_W, s_U, valid_c, v_local_dim, tid, num_threads, m_stride, k_stride, v_stride);
        __syncthreads();

        if (t + 1 < params.num_chunks) {
            const int next_stage = current_stage ^ 1;
            prefetch_chunk_stage_async<Params::C, Params::K_DIM, Params::V_DIM, T>(
                s_K_stages[next_stage],
                s_Q_stages[next_stage],
                s_G_stages[next_stage],
                s_V_stages[next_stage],
                s_Beta_stages[next_stage],
                params,
                batch_idx,
                head_idx,
                t + 1,
                v_begin,
                v_local_dim,
                tid,
                num_threads,
                k_stride,
                v_stride);
        }

        normalize_rows_inplace_strided<Params::C, Params::K_DIM, T>(
            s_Q,
            s_row_scales,
            valid_c,
            rsqrtf(static_cast<float>(Params::K_DIM)),
            tid,
            num_threads,
            k_stride);
        __syncthreads();

        const std::size_t o_base =
            ((static_cast<std::size_t>(batch_idx) * params.num_heads + head_idx) * params.seq_len + start_token) *
            Params::V_DIM;
        const int O_elements = Params::C * v_local_dim;
        for (int idx = tid; idx < O_elements; idx += num_threads) {
            int row = idx / v_local_dim;
            int v_local = idx % v_local_dim;
            if (row >= valid_c) {
                continue;
            }

            float o_acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < Params::K_DIM; ++k) {
                float q_val = static_cast<float>(s_Q[row * k_stride + k]);
                float s_val = s_S_matrix[k * v_stride + v_local];
                float prefix_decay = s_G_prefix[row * k_stride + k];
                o_acc = fmaf(q_val, s_val * prefix_decay, o_acc);
            }
            o_acc += static_cast<float>(s_U[row * v_stride + v_local]);
            params.o_ptr[o_base + row * Params::V_DIM + (v_begin + v_local)] = static_cast<T>(o_acc);
        }
        __syncthreads();

        for (int idx = tid; idx < Params::K_DIM * v_local_dim; idx += num_threads) {
            int k = idx / v_local_dim;
            int v_local = idx % v_local_dim;
            float decay_val = s_G_prefix[(valid_c - 1) * k_stride + k];
            float update = 0.0f;
            #pragma unroll
            for (int c = 0; c < Params::C; ++c) {
                if (c >= valid_c) {
                    continue;
                }
                float w_val = static_cast<float>(s_W[c * k_stride + k]);
                float u_val = static_cast<float>(s_U[c * v_stride + v_local]);
                update = fmaf(w_val, u_val, update);
            }
            s_S_matrix[k * v_stride + v_local] =
                s_S_matrix[k * v_stride + v_local] * decay_val + update;
        }
        __syncthreads();

        if (t + 1 < params.num_chunks) {
            cp_async_wait_group_0();
            __syncthreads();
            current_stage ^= 1;
        }
    }
}

} // namespace prefill
} // namespace kernel
} // namespace kda