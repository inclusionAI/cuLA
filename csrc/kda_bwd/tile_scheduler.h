#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"
#include "cute/tensor.hpp"

struct NaiveTileScheduler {
    struct Params {
      int num_blocks;
      int num_heads;
      int num_k;

      int num_sm;
      int *tile_counter; // persistent: global atomicAdd counter (device memory, init to 0)
    };
    
    int block_idx;
    Params params;
    int k_idx;
    bool is_valid_flag;
  
    CUTLASS_DEVICE
    NaiveTileScheduler(Params const& params)
        : block_idx(blockIdx.x),
          k_idx(blockIdx.z),
          params(params),
          is_valid_flag(true) {
          }
  
    static dim3 get_grid_shape(Params const& params) {
      // Persistent: launch num_sm CTAs
      dim3 grid(params.num_sm, 1, 1);
      return grid;
    }

    // Get next tile via atomicAdd (called by Load warp elected thread)
    CUTLASS_DEVICE
    int get_next_tile_id() {
      return atomicAdd(params.tile_counter, 1);
    }

    CUTLASS_DEVICE
    int total_tiles() const {
      return params.num_blocks * params.num_heads;
    }
  
    CUTLASS_DEVICE
    bool is_valid() { return is_valid_flag; }

    // Decode tile_id into (batch_idx, head_idx, seq_idx)
    CUTLASS_DEVICE
    static auto decode_tile_coord(int tile_id, int num_heads, int *chunk_indices_ptr, int *cu_seqlens_ptr) {
      using namespace cute;
      int tile_idx_raw = tile_id / num_heads;
      int head_idx = tile_id % num_heads;
      int batch_idx = chunk_indices_ptr[tile_idx_raw * 2];
      int seq_idx = chunk_indices_ptr[tile_idx_raw * 2 + 1];
      return make_coord(batch_idx, head_idx, seq_idx, 0);
    }
  
    CUTLASS_DEVICE
    auto get_block_coord(int *chunk_indices_ptr) {
      using namespace cute;
      int tile_idx = block_idx / params.num_heads;
      int head_idx = block_idx % params.num_heads;
      int batch_idx = chunk_indices_ptr[tile_idx * 2];
      int seq_idx = chunk_indices_ptr[tile_idx * 2 + 1];
      return make_coord(batch_idx, head_idx, seq_idx, k_idx);
    }
  
    CUTLASS_DEVICE
    NaiveTileScheduler& operator++() {
      is_valid_flag = false;
      return *this;
    }
  };