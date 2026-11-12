#pragma once
#include <cuda_runtime.h>

#include "kda_prefill_io.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kda_kernel.hpp"
#include "kda_params.hpp"

namespace kda {
namespace api {

inline void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

inline void check_sm89_or_newer(const cudaDeviceProp& prop) {
  const int sm = prop.major * 10 + prop.minor;
  if (sm < 89) {
    throw std::runtime_error(
        "KDA prefill requires an sm89-or-newer GPU, but current device is sm_" +
        std::to_string(prop.major) + std::to_string(prop.minor));
  }
}

template <typename T>
inline void normalize_prefill_io(KdaPrefillIO<T>& io) {
  if (io.chunk_size <= 0) {
    throw std::runtime_error("prefill io chunk_size must be positive");
  }
  if (io.num_chunks <= 0) {
    io.num_chunks = (io.seq_len + io.chunk_size - 1) / io.chunk_size;
  }
}

template <typename T>
inline void validate_prefill_io(const KdaPrefillIO<T>& io) {
  if (!io.q_ptr || !io.k_ptr || !io.v_ptr || !io.g_ptr || !io.beta_ptr ||
      !io.w_ptr || !io.u_ptr || !io.o_ptr) {
    throw std::runtime_error("prefill io contains null pointers");
  }
  if (io.batch_size <= 0 || io.num_heads <= 0 || io.seq_len <= 0 ||
      io.head_dim <= 0 || io.value_dim <= 0 || io.chunk_size <= 0) {
    throw std::runtime_error("prefill io dimensions must be positive");
  }
}

// Split V across inter-chunk blocks to bound per-block shared memory.
inline int choose_inter_v_shards(int value_dim,
                                 int base_inter_blocks,
                                 int target_inter_blocks) {
  if (value_dim <= 0 || base_inter_blocks <= 0) {
    return 1;
  }

  int desired_shards =
      (target_inter_blocks + base_inter_blocks - 1) / base_inter_blocks;
  desired_shards = std::max(1, std::min(desired_shards, value_dim));

  for (int shards = desired_shards; shards <= value_dim; ++shards) {
    if (value_dim % shards == 0) {
      return shards;
    }
  }
  return desired_shards;
}


inline int read_env_threads_override(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

inline int read_env_positive_override(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  const int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

template <typename Params>
inline size_t intra_chunk_smem_bytes() {
  using T = typename Params::element_type;
  return sizeof(T) *
             (Params::C * (Params::K_DIM + 1) + Params::C * (Params::V_DIM + 1) +
              Params::C * (Params::K_DIM + 1) + Params::C) +
         sizeof(float) * (Params::C * (Params::C + 1) + Params::C);
}

template <typename Params>
inline size_t inter_chunk_smem_bytes(const Params& params) {
  using T = typename Params::element_type;
  const size_t wmma_scratch =
      (Params::K_DIM == 64 && Params::V_DIM == 64 && Params::C == 32 &&
       params.inter_v_tile == 16)
          ? sizeof(float) * Params::K_DIM * params.inter_v_tile
          : 0;
  return sizeof(float) * (Params::K_DIM * (params.inter_v_tile + 1)) +
         sizeof(T) * (Params::C * (Params::K_DIM + 1) + Params::C * (Params::K_DIM + 1) +
                      Params::C * (params.inter_v_tile + 1)) +
         sizeof(float) * (Params::C + Params::C * (Params::K_DIM + 1)) +
         wmma_scratch;
}

template <typename Params>
inline size_t fused_prefill_smem_bytes(const Params& params) {
  using T = typename Params::element_type;
  const size_t k_stride = Params::K_DIM + 1;
  const size_t v_stride = params.inter_v_tile + 1;
  const size_t m_stride = Params::C + 1;
  return sizeof(T) *
             (2 * Params::C * k_stride + 2 * Params::C * k_stride +
              2 * Params::C * k_stride + 2 * Params::C * v_stride +
              2 * Params::C + Params::C * k_stride + Params::C * v_stride) +
         sizeof(float) *
             (Params::K_DIM * v_stride + Params::C * m_stride +
              Params::C * k_stride + Params::C);
}

template <typename Params>
inline void launch_kda_prefill_kernel(Params params,
                                      cudaStream_t stream = 0,
                                      int intra_threads = 256,
                                      int default_inter_threads = 256) {
  if (params.seq_len <= 0 || params.batch_size <= 0 || params.num_heads <= 0) {
    return;
  }

  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "get current device failed");
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, dev), "get device properties failed");
  check_sm89_or_newer(prop);

  const int base_inter_blocks = params.batch_size * params.num_heads;
  const int target_inter_blocks =
      std::max(prop.multiProcessorCount * 2, base_inter_blocks);
  int shards =
      choose_inter_v_shards(Params::V_DIM, base_inter_blocks, target_inter_blocks);
  const char* inter_shards_env = std::getenv("KDA_INTER_SHARDS");
  if (inter_shards_env != nullptr && *inter_shards_env != '\0') {
    const int requested_shards = std::atoi(inter_shards_env);
    if (requested_shards > 0 && Params::V_DIM % requested_shards == 0) {
      shards = requested_shards;
    }
  } else if (Params::K_DIM == 64 && Params::V_DIM == 64 && Params::C == 32) {
    shards = 4;
  }

  params.inter_v_shards = shards;
  params.inter_v_tile = (Params::V_DIM + shards - 1) / shards;

  const int intra_smem = static_cast<int>(intra_chunk_smem_bytes<Params>());
  const int inter_smem = static_cast<int>(inter_chunk_smem_bytes(params));
  constexpr bool use_fused_kernel = false;
  const int fused_smem =
      use_fused_kernel ? static_cast<int>(fused_prefill_smem_bytes(params)) : 0;

  cudaError_t attr_err = cudaSuccess;
  if constexpr (use_fused_kernel) {
    attr_err = cudaFuncSetAttribute(
        kernel::prefill::kda_fused_prefill_kernel<Params>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fused_smem);
    if (attr_err != cudaSuccess && fused_smem <= 48 * 1024) {
      check_cuda(attr_err, "set fused shared memory attribute failed");
    }
  } else {
    attr_err = cudaFuncSetAttribute(
        kernel::prefill::kda_intra_chunk_kernel<Params>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        intra_smem);
    if (attr_err != cudaSuccess && intra_smem <= 48 * 1024) {
      check_cuda(attr_err, "set intra shared memory attribute failed");
    }
    attr_err = cudaFuncSetAttribute(
        kernel::prefill::kda_inter_chunk_rnn_kernel<Params>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        inter_smem);
    if (attr_err != cudaSuccess && inter_smem <= 48 * 1024) {
      check_cuda(attr_err, "set inter shared memory attribute failed");
    }
  }

  if constexpr (use_fused_kernel) {
    dim3 grid_fused(params.num_heads, params.batch_size, params.inter_v_shards);
    kernel::prefill::kda_fused_prefill_kernel<Params>
        <<<grid_fused, default_inter_threads, fused_smem, stream>>>(params);
    check_cuda(cudaGetLastError(), "launch kda_fused_prefill_kernel failed");
    return;
  }

  dim3 grid_intra(params.num_chunks, params.num_heads, params.batch_size);
  intra_threads = read_env_threads_override("KDA_INTRA_THREADS", intra_threads);
  kernel::prefill::kda_intra_chunk_kernel<Params>
      <<<grid_intra, intra_threads, intra_smem, stream>>>(params);
  check_cuda(cudaGetLastError(), "launch kda_intra_chunk_kernel failed");

  int inter_threads =
      (Params::K_DIM == 64 && Params::V_DIM == 64 && Params::C == 32) ? default_inter_threads : intra_threads;
  inter_threads = read_env_threads_override("KDA_INTER_THREADS", inter_threads);
  dim3 grid_inter(params.num_heads, params.batch_size, params.inter_v_shards);
  kernel::prefill::kda_inter_chunk_rnn_kernel<Params>
      <<<grid_inter, inter_threads, inter_smem, stream>>>(params);
  check_cuda(cudaGetLastError(), "launch kda_inter_chunk_rnn_kernel failed");
}

cudaError_t launch_kda_prefill_f32(const KdaPrefillIO<float>& io,
                                   cudaStream_t stream = 0);

inline cudaError_t launch_kda_prefill(KdaPrefillIO<float> io,
                                      cudaStream_t stream = 0) {
  normalize_prefill_io(io);
  validate_prefill_io(io);
  return launch_kda_prefill_f32(io, stream);
}

}  // namespace api
}  // namespace kda