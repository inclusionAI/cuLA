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

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/kernel_hardware_info.h>

namespace kda::sm90::kernel {

using namespace cute;

struct WorkDesc {
    // coord
    int32_t seq_idx;      // which sequence to process
    int32_t qk_head_idx;  // head idx for Q/K (the representative of the GVA group)
    int32_t head_idx;     // head idx for V/O/g/beta
    int64_t tok_offset;   // start offset of this sequence in the packed tensor

    // shape
    int64_t seq_len;  // length of this sequence

    // update by mainloop
    int32_t tile_idx = 0;  // current tile index (mutated by the mainloop)

    template <typename Params>
    CUTE_DEVICE bool
    is_valid(Params const& params) {
        return seq_idx >= 0 && seq_idx < params.num_seqs;
    }

    CUTE_DEVICE int32_t
    q_head_idx() const {
        return qk_head_idx;
    }
    CUTE_DEVICE int32_t
    k_head_idx() const {
        return qk_head_idx;
    }
    CUTE_DEVICE int32_t
    v_head_idx() const {
        return head_idx;
    }
    CUTE_DEVICE int32_t
    o_head_idx() const {
        return head_idx;
    }

    // compatible interface, for work without ChunkWiseParallel, chunk_len equals to seq_len
    CUTE_DEVICE int32_t
    chunk_len() const {
        return seq_len;
    }
};

// Each block handles a single (seq, v_head) work item; CTAs do not cooperate.
// GVA optimization: heads_per_group is precomputed on the host and stored in
// Params, so the device side does not redo the integer division per CTA.
struct IndividualTileScheduler {
    struct Params {
        dim3 grid;
        int32_t num_seqs;
        int32_t num_v_heads;
        int32_t heads_per_group;  // = num_v_heads / num_qk_heads, precomputed on host
    };

    bool scheduled = false;  // a once flag

    CUTE_DEVICE
    IndividualTileScheduler(Params const& params) {
    }

    template <typename ProblemSize, typename ClusterShape, typename TileShape>
    static Params
    to_underlying_arguments(
        ProblemSize const& problem_size,
        cutlass::KernelHardwareInfo const& hw_info,
        ClusterShape const& cluster_shape,
        TileShape const& tile_shape) {
        // Compute heads_per_group once on the host so every CTA does not have to redo
        // the integer division.
        int32_t const heads_per_group = problem_size.num_v_heads / problem_size.num_qk_heads;
        dim3 grid(0, 1, 1);
        grid.x = problem_size.num_seqs * problem_size.num_v_heads;
        DPRINTF(
            "to_underlying_arguments: grid:{.x:%d, .y:%d, .z:%d}, num_seqs:%d, num_qk_heads:%d, num_v_heads:%d, "
            "heads_per_group:%d\n",
            grid.x,
            grid.y,
            grid.z,
            problem_size.num_seqs,
            problem_size.num_qk_heads,
            problem_size.num_v_heads,
            heads_per_group);
        return {
            .grid = grid,
            .num_seqs = problem_size.num_seqs,
            .num_v_heads = problem_size.num_v_heads,
            .heads_per_group = heads_per_group,
        };
    }

    static dim3
    get_grid_shape(Params const& params) {
        return params.grid;
    }

    template <typename ProblemSize>
    CUTE_DEVICE WorkDesc
    get_next_work(Params params, ProblemSize const& problem_size) {
        int32_t seq_idx = blockIdx.x / params.num_v_heads;
        int32_t head_idx = blockIdx.x % params.num_v_heads;
        // GVA: use the host-precomputed heads_per_group to avoid device-side division.
        int32_t qk_head_idx = head_idx / params.heads_per_group;

        int32_t s = problem_size.cu_seqlens[seq_idx];
        int32_t e = problem_size.cu_seqlens[seq_idx + 1];
        int32_t seq_len = e - s;

        if (scheduled) {
            seq_idx = -1;
        } else {
            scheduled = true;
            DPRINTF0_W(
                "get_next_work: this_work={seq_idx:%d qk_head_idx:%d head_idx:%d tok_offset:%lld seq_len:%lld}\n",
                seq_idx,
                qk_head_idx,
                head_idx,
                s,
                seq_len);
        }

        return {
            .seq_idx = seq_idx,
            .qk_head_idx = qk_head_idx,
            .head_idx = head_idx,
            .tok_offset = s,
            .seq_len = seq_len,
        };
    }
};

}  // namespace kda::sm90::kernel
