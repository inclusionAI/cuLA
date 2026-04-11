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
    int32_t seq_idx;
    int32_t head_idx;          // state/V head index in [0, num_heads)
    int32_t k_group_size = 1;  // num_heads / num_k_heads. Q and K share the
                               // same physical head = head_idx / k_group_size.
                               // k_group_size == 1 is plain MHA.
    int64_t tok_offset;        // offset to the start of the start

    // shape
    int64_t seq_len;

    // update by mainloop
    int32_t tile_idx = 0;

    template <typename Params>
    CUTE_DEVICE bool
    is_valid(Params const& params) {
        return seq_idx >= 0 && seq_idx < params.num_seqs;
    }

    // Q and K share one physical head count in KDA. In the "multi-value"
    // flavor (Qwen3.5-style), k_group_size > 1 state heads share each
    // physical Q/K head; in plain MHA k_group_size == 1 and every state head
    // gets its own Q and K head.
    CUTE_DEVICE int32_t
    q_head_idx() const {
        return head_idx / k_group_size;
    }
    CUTE_DEVICE int32_t
    k_head_idx() const {
        return head_idx / k_group_size;
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

struct IndividualTileScheduler {
    struct Params {
        dim3 grid;
        int32_t num_seqs;
        int32_t num_heads;
        int32_t k_group_size;  // num_heads / num_k_heads (>=1, ==1 for MHA)
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
        dim3 grid(0, 1, 1);
        grid.x = problem_size.num_seqs * problem_size.num_heads;
        // The host entry point normalises a sentinel num_k_heads==0 to
        // `num_heads` before building the launcher Arguments, so
        // `problem_size.num_k_heads` is guaranteed positive here (== num_heads
        // in plain MHA). Dividing is therefore always safe.
        int32_t k_group_size = problem_size.num_heads / problem_size.num_k_heads;
        DPRINTF(
            "to_underlying_arguments: grid:{.x:%d, .y:%d, .z:%d}, num_seqs:%d, "
            "num_heads:%d, num_k_heads:%d, k_group_size:%d\n",
            grid.x,
            grid.y,
            grid.z,
            problem_size.num_seqs,
            problem_size.num_heads,
            problem_size.num_k_heads,
            k_group_size);
        return {
            .grid = grid,
            .num_seqs = problem_size.num_seqs,
            .num_heads = problem_size.num_heads,
            .k_group_size = k_group_size,
        };
    }

    static dim3
    get_grid_shape(Params const& params) {
        return params.grid;
    }

    template <typename ProblemSize>
    CUTE_DEVICE WorkDesc
    get_next_work(Params params, ProblemSize const& problem_size) {
        int32_t seq_idx = blockIdx.x / params.num_heads;
        int32_t head_idx = blockIdx.x % params.num_heads;

        int32_t s = problem_size.cu_seqlens[seq_idx];
        int32_t e = problem_size.cu_seqlens[seq_idx + 1];
        int32_t seq_len = e - s;

        if (scheduled) {
            seq_idx = -1;
        } else {
            scheduled = true;
            DPRINTF0_W(
                "get_next_work: this_work={seq_idx:%d head_idx:%d tok_offset:%lld seq_len:%lld}\n",
                seq_idx,
                head_idx,
                s,
                seq_len);
        }

        return {
            .seq_idx = seq_idx,
            .head_idx = head_idx,
            .k_group_size = params.k_group_size,
            .tok_offset = s,
            .seq_len = seq_len,
        };
    }
};

}  // namespace kda::sm90::kernel
