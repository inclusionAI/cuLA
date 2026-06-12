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

#include "qwen35_decode_common.cuh"

#include <cutlass/arch/arch.h>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>

namespace cula::qwen35::decode::kernel {

using namespace cute;

template <typename scalar_t>
struct Qwen35ScalarKdaDecodeMainloop {
  // Decode design decision:
  // - recurrent_state remains fp32 both physically and mathematically
  // - decode is treated as a register-level recurrent GEMV/rank-1-update
  //   problem, not as a Tensor Core GEMM problem
  // - you can think of the target implementation style as a
  //   flash_linear_decode_kernel: pure CUDA Core math, warp-shuffle vector
  //   sharing, and fp32 state kept live as long as possible during one token
  //   update
  //
  // Reason:
  // - state participates in a long recurrent chain; lowering master-state
  //   precision is risky and usually not worth it
  // - decode operates on a single-token q/k vector, so the dominant kernels
  //   are GEMV-like:
  //     proj = state @ k
  //     out  = state' @ q
  //   This is typically register / memory bound rather than Tensor-Core bound
  //
  // Practical consequence:
  // - the first production-worthy decode path should be built around fp32 FFMA
  //   on CUDA cores
  // - tile structure is still useful, but it should serve register ownership,
  //   reduction, and cache behavior instead of forcing a GMMA lowering
  //
  // The remaining work for this kernel is therefore:
  // 1. tighten thread ownership of the fp32 state tile
  // 2. optimize proj / update / out reductions
  // 3. evaluate warp-specialized load/compute roles only after the fp32 path
  //    is stable and measured
  static constexpr int kTileV = 16;
  static constexpr int kTileK = 16;
  static constexpr int kTilesPerV = kHeadDimV / kTileV;
  static constexpr int kTilesPerK = kHeadDimQK / kTileK;
  static constexpr int kWarpSize = 32;
  static constexpr int kRowsPerTile = kTileV;
  static constexpr int kWarpsPerCta = 4;
  static constexpr int kRowsPerWarp = kWarpSize;
  static constexpr int kRowsPerThread = 1;

  static_assert(kHeadDimV == 128);
  static_assert(kHeadDimQK == 128);
  static_assert(kWarpsPerCta * kRowsPerWarp == kHeadDimV);

  // First concrete decode threading plan:
  //
  // - 1 CTA = 1 (token, hv)
  // - 128 threads = 4 warps
  // - 1 thread owns exactly 1 V-row of the 128x128 recurrent state
  // - Therefore one CTA covers all 128 V-rows exactly once
  //
  // For the owned row, the thread streams over K in 16-wide tiles:
  //
  //   state_row[0:15]   -> registers
  //   state_row[16:31]  -> registers
  //   ...
  //   state_row[112:127]-> registers
  //
  // This means the first concrete fp32 path does NOT attempt to keep the
  // whole 128-float row resident in registers at once. Instead it keeps the
  // current K tile resident:
  //
  // - state_regs[16] : current fp32 state tile
  // - k_regs[16]     : current key tile
  // - q_regs[16]     : current query tile
  //
  // plus a handful of scalar accumulators:
  //
  // - proj_row
  // - out_row
  // - v_new_row
  // - gate scalars
  //
  // This is a practical first step toward the user's desired "state stays in
  // registers for the current token" behavior while keeping register pressure
  // manageable.
  //
  // Reduction policy for this first concrete plan:
  //
  // - proj/out are row-local, so no warp reduction is required
  // - each row is fully owned by one thread across all K tiles
  // - warp shuffle is reserved for future vector-broadcast refinements if we
  //   decide to move q/k staging from shared memory into warp-register paths
  struct ThreadRowPlan {
    int warp_id;
    int lane_id;
    int v_row;
    bool owns_row;
  };

  struct TileCoords {
    int v_base;
    int k_base;
  };

  CUTE_DEVICE static ThreadRowPlan make_thread_row_plan(int tid) {
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    const int v_row = warp_id * kRowsPerWarp + lane_id;
    const bool owns_row = v_row < kHeadDimV;
    return ThreadRowPlan{warp_id, lane_id, v_row, owns_row};
  }

  template <typename TensorVec>
  CUTE_DEVICE static void load_vec_tile_to_regs(
      TensorVec const& vec,
      TileCoords coords,
      float (&regs)[kTileK]) {
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      regs[kk] = static_cast<float>(vec(coords.k_base + kk));
    }
  }

  template <typename TensorState>
  CUTE_DEVICE static void load_state_row_tile_to_regs(
      TensorState const& state_vk,
      int v_row,
      TileCoords coords,
      float (&state_regs)[kTileK]) {
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      state_regs[kk] = static_cast<float>(state_vk(v_row, coords.k_base + kk));
    }
  }

  template <typename TensorState>
  CUTE_DEVICE static void store_state_row_tile_from_regs(
      TensorState& state_vk,
      int v_row,
      TileCoords coords,
      float const (&state_regs)[kTileK]) {
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      state_vk(v_row, coords.k_base + kk) = state_regs[kk];
    }
  }

  struct RowTileProjPlan {
    int v_base;
    int k_base;
    int warp_id;
    int lane_id;
    bool owns_row;
    int row_in_tile;
    int v_row;
  };

  CUTE_DEVICE static TileCoords make_tile_coords(int tile_v, int tile_k) {
    return TileCoords{tile_v * kTileV, tile_k * kTileK};
  }

  CUTE_DEVICE static RowTileProjPlan make_row_tile_proj_plan(
      TileCoords coords,
      int warp_id,
      int lane_id) {
    const bool owns_row = lane_id < kTileV;
    const int row_in_tile = lane_id;
    const int v_row = coords.v_base + row_in_tile;
    return RowTileProjPlan{
        coords.v_base,
        coords.k_base,
        warp_id,
        lane_id,
        owns_row,
        row_in_tile,
        v_row,
    };
  }

  struct RowTileUpdatePlan {
    TileCoords coords;
    int warp_id;
    int lane_id;
    bool owns_row;
    int row_in_tile;
    int v_row;
  };

  CUTE_DEVICE static RowTileUpdatePlan make_row_tile_update_plan(
      TileCoords coords,
      int warp_id,
      int lane_id) {
    const bool owns_row = lane_id < kTileV;
    const int row_in_tile = lane_id;
    const int v_row = coords.v_base + row_in_tile;
    return RowTileUpdatePlan{
        coords,
        warp_id,
        lane_id,
        owns_row,
        row_in_tile,
        v_row,
    };
  }

  template <typename TensorState, typename TensorKTile>
  CUTE_DEVICE static float accumulate_proj_row_tile(
      TensorState const& state_vk,
      TensorKTile const& k_smem,
      int v_row,
      TileCoords coords) {
    float state_regs[kTileK];
    float k_regs[kTileK];
    load_state_row_tile_to_regs(state_vk, v_row, coords, state_regs);
    load_vec_tile_to_regs(k_smem, coords, k_regs);

    float accum = 0.f;
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      accum += state_regs[kk] * k_regs[kk];
    }
    return accum;
  }

  template <typename TensorState, typename TensorKTile, typename TensorQTile>
  CUTE_DEVICE static float update_state_row_tile_and_accumulate_out(
      TensorState& state_vk,
      TensorKTile const& k_smem,
      TensorQTile const& q_smem,
      int v_row,
      TileCoords coords,
      float decay,
      float v_new) {
    float state_regs[kTileK];
    float k_regs[kTileK];
    float q_regs[kTileK];
    load_state_row_tile_to_regs(state_vk, v_row, coords, state_regs);
    load_vec_tile_to_regs(k_smem, coords, k_regs);
    load_vec_tile_to_regs(q_smem, coords, q_regs);

    float out_acc = 0.f;
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      const float state_new = decay * state_regs[kk] + v_new * k_regs[kk];
      state_regs[kk] = state_new;
      out_acc += state_new * q_regs[kk];
    }
    store_state_row_tile_from_regs(state_vk, v_row, coords, state_regs);
    return out_acc;
  }

  template <typename TensorState, typename TensorKTile>
  CUTE_DEVICE static float project_row_tile(
      TensorState const& state_vk,
      TensorKTile const& k_smem,
      int v_row,
      TileCoords coords) {
    // Current decode path:
    // - one thread owns one full V-row
    // - this helper computes the row-local proj contribution for one K tile
    // - no cross-thread reduction is needed
    return accumulate_proj_row_tile(state_vk, k_smem, v_row, coords);
  }

  template <typename TensorState, typename TensorKTile>
  CUTE_DEVICE static float project_row_tile(
      TensorState const& state_vk,
      TensorKTile const& k_smem,
      RowTileProjPlan const& plan) {
    if (!plan.owns_row) {
      return 0.f;
    }
    return project_row_tile(
        state_vk, k_smem, plan.v_row, TileCoords{plan.v_base, plan.k_base});
  }

  template <typename TensorState, typename TensorKTile, typename TensorQTile>
  CUTE_DEVICE static float update_and_output_row_tile(
      TensorState& state_vk,
      TensorKTile const& k_smem,
      TensorQTile const& q_smem,
      int v_row,
      TileCoords coords,
      float decay,
      float v_new) {
    // Current decode path:
    // - read one 16-wide state tile for the owned row into registers
    // - apply decay and rank-1 update in fp32
    // - accumulate the matching out contribution against q
    // - write the updated state tile back
    return update_state_row_tile_and_accumulate_out(
        state_vk, k_smem, q_smem, v_row, coords, decay, v_new);
  }

  template <typename TensorState, typename TensorKTile, typename TensorQTile>
  CUTE_DEVICE static float update_and_output_row_tile(
      TensorState& state_vk,
      TensorKTile const& k_smem,
      TensorQTile const& q_smem,
      RowTileUpdatePlan const& plan,
      float decay,
      float v_new) {
    if (!plan.owns_row) {
      return 0.f;
    }
    return update_and_output_row_tile(
        state_vk,
        k_smem,
        q_smem,
        plan.v_row,
        plan.coords,
        decay,
        v_new);
  }

  CUTE_DEVICE static float softplusf_approx(float x) {
    return x > 20.f ? x : log1pf(expf(x));
  }

  template <
      typename TensorQ,
      typename TensorK,
      typename TensorV,
      typename TensorA,
      typename TensorB,
      typename TensorAlog,
      typename TensorDt,
      typename TensorHvk,
      typename TensorOut,
      typename SharedStorage>
  CUTE_DEVICE static void run(
      TensorQ const& q_vec,
      TensorK const& k_vec,
      TensorV const& v_vec,
      TensorA const& a_scalar,
      TensorB const& b_scalar,
      TensorAlog const& A_log_scalar,
      TensorDt const& dt_bias_scalar,
      TensorHvk& state_vk,
      TensorOut& out_vec,
      SharedStorage& storage,
      int tid,
      int num_threads) {
    // Decode organization:
    // - 1 warpgroup owns the full [128, 128] state tile for one (token, hv)
    // - state is traversed as 16x16 tiles over the internal VK view
    // - q/k/v are staged once into shared memory
    // - proj/out are accumulated over K tiles
    // - rank-1 update is applied tile-by-tile in the same traversal order
    //
    // This pass establishes the tile-first organization for the final fp32
    // decode kernel. The next implementation step should optimize the scalar
    // inner loops with better register ownership / reductions rather than
    // forcing Tensor Core math.
    //
    // TODO(qwen35-decode-fp32):
    // - evaluate whether q/k should move from shared-memory staging to
    //   warp-shuffle broadcast
    // - evaluate whether one thread should own more than one V-row
    // - evaluate whether some parts of the state row can remain resident in
    //   registers across both proj and update/out passes with acceptable
    //   register pressure

    const float a_val = static_cast<float>(a_scalar);
    const float b_val = static_cast<float>(b_scalar);
    const float A_log_val = static_cast<float>(A_log_scalar);
    const float dt_bias_val = static_cast<float>(dt_bias_scalar);

    const float g = -expf(A_log_val) * softplusf_approx(a_val + dt_bias_val);
    const float decay = expf(g);
    const float beta = 1.f / (1.f + expf(-b_val));

    auto q_smem = make_tensor(make_smem_ptr(storage.q_smem), make_layout(make_shape(Int<kHeadDimQK>{})));
    auto k_smem = make_tensor(make_smem_ptr(storage.k_smem), make_layout(make_shape(Int<kHeadDimQK>{})));
    auto v_smem = make_tensor(make_smem_ptr(storage.v_smem), make_layout(make_shape(Int<kHeadDimV>{})));
    auto norm_smem = make_tensor(make_smem_ptr(storage.norm_smem), make_layout(make_shape(Int<2>{})));
    auto proj_smem = make_tensor(make_smem_ptr(storage.proj_smem), make_layout(make_shape(Int<kHeadDimV>{})));
    auto out_smem = make_tensor(make_smem_ptr(storage.out_smem), make_layout(make_shape(Int<kHeadDimV>{})));

    // Stage q/k/v once per CTA for the current decode token.
    for (int idx = tid; idx < kHeadDimQK; idx += num_threads) {
      q_smem(idx) = static_cast<float>(q_vec(idx));
      k_smem(idx) = static_cast<float>(k_vec(idx));
    }
    for (int idx = tid; idx < kHeadDimV; idx += num_threads) {
      v_smem(idx) = v_vec(idx);
      proj_smem(idx) = 0.f;
      out_smem(idx) = 0.f;
    }
    __syncthreads();

    if (tid == 0) {
      float q_norm_sq = 0.f;
      float k_norm_sq = 0.f;
#pragma unroll
      for (int idx = 0; idx < kHeadDimQK; ++idx) {
        const float q_val = q_smem(idx);
        const float k_val = k_smem(idx);
        q_norm_sq += q_val * q_val;
        k_norm_sq += k_val * k_val;
      }
      norm_smem(0) = rsqrtf(q_norm_sq + 1e-6f) * rsqrtf(static_cast<float>(kHeadDimQK));
      norm_smem(1) = rsqrtf(k_norm_sq + 1e-6f);
    }
    __syncthreads();

    for (int idx = tid; idx < kHeadDimQK; idx += num_threads) {
      q_smem(idx) = q_smem(idx) * norm_smem(0);
      k_smem(idx) = k_smem(idx) * norm_smem(1);
    }
    __syncthreads();

    ThreadRowPlan row_plan = make_thread_row_plan(tid);

    // First concrete ownership model:
    // - each thread owns one full state row across all 128 K columns
    // - the row is streamed tile-by-tile through registers
    // - no cross-thread reduction is needed for proj/out because the full row
    //   stays with one thread for the duration of the token update
    if (row_plan.owns_row) {
      float proj_row = 0.f;
      for (int tile_k = 0; tile_k < kTilesPerK; ++tile_k) {
        TileCoords coords = TileCoords{(row_plan.v_row / kTileV) * kTileV, tile_k * kTileK};
        RowTileProjPlan proj_plan = make_row_tile_proj_plan(coords, row_plan.warp_id, row_plan.lane_id);
        proj_plan.v_row = row_plan.v_row;
        proj_plan.owns_row = true;
        proj_row += project_row_tile(state_vk, k_smem, proj_plan);
      }

      proj_smem(row_plan.v_row) = proj_row;

      const float v_val = static_cast<float>(v_smem(row_plan.v_row));
      const float decayed_proj_row = decay * proj_row;
      const float v_new_row = beta * (v_val - decayed_proj_row);

      float out_row = 0.f;
      for (int tile_k = 0; tile_k < kTilesPerK; ++tile_k) {
        TileCoords coords = TileCoords{(row_plan.v_row / kTileV) * kTileV, tile_k * kTileK};
        RowTileUpdatePlan update_plan = make_row_tile_update_plan(coords, row_plan.warp_id, row_plan.lane_id);
        update_plan.v_row = row_plan.v_row;
        update_plan.owns_row = true;
        out_row += update_and_output_row_tile(
            state_vk, k_smem, q_smem, update_plan, decay, v_new_row);
      }

      out_smem(row_plan.v_row) = out_row;
    }
    __syncthreads();

    for (int idx = tid; idx < kHeadDimV; idx += num_threads) {
      out_vec(idx) = static_cast<scalar_t>(out_smem(idx));
    }
  }
};

} // namespace cula::qwen35::decode::kernel
