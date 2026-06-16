// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/kernel_hardware_info.h>

// TODO: implement DynamicPersistentTileScheduler with atomic added tile_counter

// ===================================================================
// StaticPersistentTileScheduler
// No smem synchronization needed — every CTA processes tiles starting
// at blockIdx.x and striding by gridDim.x. All warps within a CTA
// independently maintain the same tile_id, so no tile pipeline is needed.
//
// GVA (Grouped V-head Attention) support:
//   Q/K are sized by `num_qk_heads`; V, g, beta, O and state tensors are
//   sized by `num_v_heads`. We enumerate tiles by `num_v_heads` so that
//   each v-head is scheduled independently, and derive the companion
//   `qk_head_idx = v_head_idx / heads_per_group` on the device side.
//   `heads_per_group = num_v_heads / num_qk_heads` is precomputed on the
//   host to avoid a per-tile integer division.
// ===================================================================
struct StaticPersistentTileScheduler {
    struct Params {
        int num_blocks;       // number of sequence chunks (from chunk_indices)
        int num_heads;        // == num_v_heads; tiles are enumerated by v-head
        int heads_per_group;  // == num_v_heads / num_qk_heads, precomputed on host

        int num_sm;
        int* tile_counter;  // unused
    };

    int current_tile_id;
    Params params;

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(Params const& params) : current_tile_id(blockIdx.x), params(params) {
    }

    static dim3
    get_grid_shape(Params const& params) {
        dim3 grid(params.num_sm, 1, 1);
        return grid;
    }

    // Total number of tiles
    CUTLASS_DEVICE
    int
    total_tiles() const {
        return params.num_blocks * params.num_heads;
    }

    // Get the current tile id (no atomicAdd, purely local)
    CUTLASS_DEVICE
    int
    get_current_tile_id() const {
        return current_tile_id;
    }

    // Advance to the next tile for this CTA (stride by gridDim.x)
    CUTLASS_DEVICE
    void
    advance() {
        current_tile_id += gridDim.x;
    }

    // Check if current tile is valid
    CUTLASS_DEVICE
    bool
    is_valid() const {
        return current_tile_id < total_tiles();
    }

    // Decode tile_id -> (batch_idx, v_head_idx, seq_idx, qk_head_idx).
    // `num_v_heads` is the number of V/O/g/beta heads; tile enumeration is
    // done in v-head space. `heads_per_group` (= num_v_heads/num_qk_heads)
    // is used to derive the companion Q/K head index for GVA.
    // For backward compatibility, when HV == HQK, `heads_per_group == 1`
    // and `qk_head_idx == v_head_idx`.
    CUTLASS_DEVICE
    static auto
    decode_tile_coord(
        int tile_id, int num_v_heads, int heads_per_group, int* chunk_indices_ptr, int* /*cu_seqlens_ptr*/) {
        using namespace cute;
        int tile_idx_raw = tile_id / num_v_heads;
        int v_head_idx = tile_id % num_v_heads;
        int qk_head_idx = v_head_idx / heads_per_group;
        int batch_idx = chunk_indices_ptr[tile_idx_raw * 2];
        int seq_idx = chunk_indices_ptr[tile_idx_raw * 2 + 1];
        return make_coord(batch_idx, v_head_idx, seq_idx, qk_head_idx);
    }
};