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

#include "kerutils/kerutils.cuh"

namespace kda::sm100 {

using ku::float2_mul;
using ku::float2_sub;
using ku::nvbf16x4;
using ku::store_128b;
using namespace cute;

// ============================================================
// Forward Prologue: B-matrix (SMEM) helper functions
// ============================================================
//
// B-matrix formula (stored to SMEM for MMA consumption):
//   inter: exp2(g_first - g[x]) * K[x]     (g_first = g[sub_tile_i * 16])
//   intra: exp2(g_half  - g[x]) * K[x]     (g_half  = g[sub_tile_i * 16 + 8])
//
// SmemLayoutMatBTF32<1> = (SubTileT, TileK) = (16, TileK), K-major layout
//
// Thread mapping (128 threads per WG, 16 rows per sub_tile):
//   x_local = idx_in_warpgroup / 8           (row within sub_tile, 0..15)
//   y_base  = idx_in_warpgroup % 8 * 4       (column group base, 0..28 step 4)
//   Each thread writes 4 consecutive tf32 values (128 bits) to SMEM.
//   When TileK > 32, each thread loops over y_iter in [0, 32, ...] to cover all columns.
//
// Store pattern: sKG(x_local, y) + KG_OFFSET * index
//   where KG_OFFSET = SubTileT * TileK (stride between sub_tile buffers)

// ============================================================
// Column-based fused B-matrix helpers (1 load → N outputs per column)
// ============================================================
//
// New approach: process the lower-triangular 4×4 subchunk matrix column-by-column.
// Each helper loads K_j + G data ONCE and produces ALL outputs for that column.
// This maximizes SMEM bandwidth reuse.
//
//          j=0         j=1         j=2         j=3
//   i=0  intra[0]
//   i=1  inter[0]   intra[1]
//   i=2  inter[1]   inter[2]   intra[2]
//   i=3  inter[3]   inter[4]   inter[5]   intra[3]
//
// Work distribution (balanced at 5 outputs each):
//   WG0: col0 (4 outputs) + col3 (1 output) = 5 outputs
//   WG1: col1 (3 outputs) + col2 (2 outputs) = 5 outputs
//
// Helper summary:
//   fwd_setup_kg_col0_4out: col j=0 → intra(0,0) + inter(1,0) + inter(2,0) + inter(3,0)
//   fwd_setup_kg_col1_3out: col j=1 → intra(1,1) + inter(2,1) + inter(3,1)
//   fwd_setup_kg_col2_2out: col j=2 → intra(2,2) + inter(3,2)
//   fwd_setup_kg_col3_1out: col j=3 → intra(3,3)

// fwd_setup_kg_col0_4out: column j=0, 4 outputs (1 intra + 3 inter)
//   Loads K_0 + G data once, computes:
//     intra(0,0): exp2(g_half_0 - g_0[x]) * K_0[x]  → sKG_intra index 0
//     inter(1,0): exp2(g_first_1 - g_0[x]) * K_0[x] → sKG_inter index 0
//     inter(2,0): exp2(g_first_2 - g_0[x]) * K_0[x] → sKG_inter index 1
//     inter(3,0): exp2(g_first_3 - g_0[x]) * K_0[x] → sKG_inter index 3
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET, int TileK_, bool UnifiedGRef = true>
__forceinline__ __device__ void
fwd_setup_kg_col0_4out(
    G_TENSOR& sG, K_TENSOR& sK, KG_TENSOR& sKG_inter, KG_TENSOR& sKG_intra, int idx_in_warpgroup, int sub_seq_len) {
    constexpr int intra_offset = UnifiedGRef ? 0 : 8;
    int y_base = idx_in_warpgroup % 8 * 4;
    // K data from sub_tile j=0
    int x = idx_in_warpgroup / 8 + 0 * 16;
// Loop over column chunks to cover full TileK width
// When TileK_=32: single iteration (y_iter=0). When TileK_=64: two iterations.
#pragma unroll
    for (int y_iter = 0; y_iter < TileK_; y_iter += 32) {
        int y = y_base + y_iter;
        // Load 4 g_ref values for this column chunk
        float4 g_first_0_local = *reinterpret_cast<float4*>(&sG(min(0 * 16 + intra_offset, sub_seq_len - 1), y));
        float4 g_first_1_local = *reinterpret_cast<float4*>(&sG(min(1 * 16, sub_seq_len - 1), y));
        float4 g_first_2_local = *reinterpret_cast<float4*>(&sG(min(2 * 16, sub_seq_len - 1), y));
        float4 g_first_3_local = *reinterpret_cast<float4*>(&sG(min(3 * 16, sub_seq_len - 1), y));
        if (x < sub_seq_len) {
            float4 g = *reinterpret_cast<float4*>(&sG(x, y));
            nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
            float2 kf_a = __bfloat1622float2(k.a);
            float2 kf_b = __bfloat1622float2(k.b);
            float2 g_a = reinterpret_cast<float2*>(&g)[0];
            float2 g_b = reinterpret_cast<float2*>(&g)[1];
            // intra(0,0): exp2(g_first_0 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_0_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_0_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 0, res);
            }
            // inter(1,0): exp2(g_first_1 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_1_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_1_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 0, res);
            }
            // inter(2,0): exp2(g_first_2 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_2_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_2_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 1, res);
            }
            // inter(3,0): exp2(g_first_3 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 3, res);
            }
        } else {
            float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 0, z);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 0, z);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 1, z);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 3, z);
        }
    }
}

// fwd_setup_kg_col1_3out: column j=1, 3 outputs (1 intra + 2 inter)
//   Loads K_1 + G data once, computes:
//     intra(1,1): exp2(g_half_1 - g_1[x]) * K_1[x]  → sKG_intra index 1
//     inter(2,1): exp2(g_first_2 - g_1[x]) * K_1[x] → sKG_inter index 2
//     inter(3,1): exp2(g_first_3 - g_1[x]) * K_1[x] → sKG_inter index 4
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET, int TileK_, bool UnifiedGRef = true>
__forceinline__ __device__ void
fwd_setup_kg_col1_3out(
    G_TENSOR& sG, K_TENSOR& sK, KG_TENSOR& sKG_inter, KG_TENSOR& sKG_intra, int idx_in_warpgroup, int sub_seq_len) {
    constexpr int intra_offset = UnifiedGRef ? 0 : 8;
    int y_base = idx_in_warpgroup % 8 * 4;
    // K data from sub_tile j=1
    int x = idx_in_warpgroup / 8 + 1 * 16;
#pragma unroll
    for (int y_iter = 0; y_iter < TileK_; y_iter += 32) {
        int y = y_base + y_iter;
        // Load 3 g_ref values for this column chunk
        float4 g_first_1_local = *reinterpret_cast<float4*>(&sG(min(1 * 16 + intra_offset, sub_seq_len - 1), y));
        float4 g_first_2_local = *reinterpret_cast<float4*>(&sG(min(2 * 16, sub_seq_len - 1), y));
        float4 g_first_3_local = *reinterpret_cast<float4*>(&sG(min(3 * 16, sub_seq_len - 1), y));
        if (x < sub_seq_len) {
            float4 g = *reinterpret_cast<float4*>(&sG(x, y));
            nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
            float2 kf_a = __bfloat1622float2(k.a);
            float2 kf_b = __bfloat1622float2(k.b);
            float2 g_a = reinterpret_cast<float2*>(&g)[0];
            float2 g_b = reinterpret_cast<float2*>(&g)[1];
            // intra(1,1): exp2(g_first_1 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_1_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_1_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 1, res);
            }
            // inter(2,1): exp2(g_first_2 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_2_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_2_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 2, res);
            }
            // inter(3,1): exp2(g_first_3 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 4, res);
            }
        } else {
            float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 1, z);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 2, z);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 4, z);
        }
    }
}

// fwd_setup_kg_col2_2out: column j=2, 2 outputs (1 intra + 1 inter)
//   Loads K_2 + G data once, computes:
//     intra(2,2): exp2(g_half_2 - g_2[x]) * K_2[x]  → sKG_intra index 2
//     inter(3,2): exp2(g_first_3 - g_2[x]) * K_2[x] → sKG_inter index 5
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET, int TileK_, bool UnifiedGRef = true>
__forceinline__ __device__ void
fwd_setup_kg_col2_2out(
    G_TENSOR& sG, K_TENSOR& sK, KG_TENSOR& sKG_inter, KG_TENSOR& sKG_intra, int idx_in_warpgroup, int sub_seq_len) {
    constexpr int intra_offset = UnifiedGRef ? 0 : 8;
    int y_base = idx_in_warpgroup % 8 * 4;
    // K data from sub_tile j=2
    int x = idx_in_warpgroup / 8 + 2 * 16;
#pragma unroll
    for (int y_iter = 0; y_iter < TileK_; y_iter += 32) {
        int y = y_base + y_iter;
        // Load 2 g_ref values for this column chunk
        float4 g_first_2_local = *reinterpret_cast<float4*>(&sG(min(2 * 16 + intra_offset, sub_seq_len - 1), y));
        float4 g_first_3_local = *reinterpret_cast<float4*>(&sG(min(3 * 16, sub_seq_len - 1), y));
        if (x < sub_seq_len) {
            float4 g = *reinterpret_cast<float4*>(&sG(x, y));
            nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
            float2 kf_a = __bfloat1622float2(k.a);
            float2 kf_b = __bfloat1622float2(k.b);
            float2 g_a = reinterpret_cast<float2*>(&g)[0];
            float2 g_b = reinterpret_cast<float2*>(&g)[1];
            // intra(2,2): exp2(g_first_2 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_2_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_2_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 2, res);
            }
            // inter(3,2): exp2(g_first_3 - g[x]) * K[x]
            {
                float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[0], g_a);
                float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[1], g_b);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                float4 res;
                reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
                reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
                store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 5, res);
            }
        } else {
            float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 2, z);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 5, z);
        }
    }
}

// fwd_setup_kg_col3_1out: column j=3, 1 output (intra only)
//   Loads K_3 + G data once, computes:
//     intra(3,3): exp2(g_half_3 - g_3[x]) * K_3[x]  → sKG_intra index 3
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET, int TileK_, bool UnifiedGRef = true>
__forceinline__ __device__ void
fwd_setup_kg_col3_1out(G_TENSOR& sG, K_TENSOR& sK, KG_TENSOR& sKG_intra, int idx_in_warpgroup, int sub_seq_len) {
    constexpr int intra_offset = UnifiedGRef ? 0 : 8;
    int y_base = idx_in_warpgroup % 8 * 4;
    // K data from sub_tile j=3
    int x = idx_in_warpgroup / 8 + 3 * 16;
#pragma unroll
    for (int y_iter = 0; y_iter < TileK_; y_iter += 32) {
        int y = y_base + y_iter;
        // Load 1 g_ref value for this column chunk
        float4 g_first_3_local = *reinterpret_cast<float4*>(&sG(min(3 * 16 + intra_offset, sub_seq_len - 1), y));
        if (x < sub_seq_len) {
            float4 g = *reinterpret_cast<float4*>(&sG(x, y));
            nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
            // intra(3,3): exp2(g_first_3 - g[x]) * K[x]
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[0], reinterpret_cast<float2*>(&g)[0]);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_3_local)[1], reinterpret_cast<float2*>(&g)[1]);
            s1.x = exp2f(s1.x);
            s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x);
            s2.y = exp2f(s2.y);
            float4 res;
            reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, __bfloat1622float2(k.a));
            reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, __bfloat1622float2(k.b));
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 3, res);
        } else {
            float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 3, z);
        }
    }
}

// fwd_setup_A_inter_intra_all: fused inter+intra, single Vec (Q or K)
template <int TileK, typename G_TENSOR, typename VEC_TENSOR>
__forceinline__ __device__ void
fwd_setup_A_inter_intra_all(
    G_TENSOR& sG,
    VEC_TENSOR& sVec,
    int idx_in_warpgroup,
    int sub_seq_len,
    int k_offset,
    int tmem_addr_inter,
    int tmem_addr_intra) {
    int row = idx_in_warpgroup % 64;
    int sub_tile_i = row / 16;  // 0, 1, 2, or 3
    float res_inter[TileK];
    float res_intra[TileK];
    if (row < sub_seq_len) {
        int g_first_row = min(sub_tile_i * 16, sub_seq_len - 1);
        int g_half_row = min(sub_tile_i * 16 + 8, sub_seq_len - 1);
#pragma unroll
        for (int i = 0; i < TileK / 4; ++i) {
            int y = i * 4 + k_offset;
            // Read g[row] and Vec[row] ONCE (shared between inter and intra)
            float4 g = *reinterpret_cast<float4*>(&sG(row, y));
            nvbf16x4 v = *reinterpret_cast<nvbf16x4*>(&sVec(row, y));
            float2 va = __bfloat1622float2(v.a);
            float2 vb = __bfloat1622float2(v.b);
            float2 g_a = reinterpret_cast<float2*>(&g)[0];
            float2 g_b = reinterpret_cast<float2*>(&g)[1];
            // Inter: exp2(g[row] - g_first) * Vec[row]
            {
                float4 g_ref = *reinterpret_cast<float4*>(&sG(g_first_row, y));
                float2 s1 = float2_sub(g_a, reinterpret_cast<float2*>(&g_ref)[0]);
                float2 s2 = float2_sub(g_b, reinterpret_cast<float2*>(&g_ref)[1]);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                reinterpret_cast<float2*>(&res_inter[i * 4])[0] = float2_mul(s1, va);
                reinterpret_cast<float2*>(&res_inter[i * 4])[1] = float2_mul(s2, vb);
            }
            // Intra: exp2(g[row] - g_half) * Vec[row]
            {
                float4 g_ref = *reinterpret_cast<float4*>(&sG(g_half_row, y));
                float2 s1 = float2_sub(g_a, reinterpret_cast<float2*>(&g_ref)[0]);
                float2 s2 = float2_sub(g_b, reinterpret_cast<float2*>(&g_ref)[1]);
                s1.x = exp2f(s1.x);
                s1.y = exp2f(s1.y);
                s2.x = exp2f(s2.x);
                s2.y = exp2f(s2.y);
                reinterpret_cast<float2*>(&res_intra[i * 4])[0] = float2_mul(s1, va);
                reinterpret_cast<float2*>(&res_intra[i * 4])[1] = float2_mul(s2, vb);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < TileK; ++i) {
            res_inter[i] = 0.0f;
            res_intra[i] = 0.0f;
        }
    }
    ku::tmem_st_32dp32bNx<TileK>(tmem_addr_inter, res_inter);
    ku::tmem_st_32dp32bNx<TileK>(tmem_addr_intra, res_intra);
}

// fwd_setup_A_inter_all: inter-only, single Vec (Q or K)
// Only computes gated Vec with inter reference (g_first), skips intra entirely.
template <int TileK, typename G_TENSOR, typename VEC_TENSOR>
__forceinline__ __device__ void
fwd_setup_A_inter_all(
    G_TENSOR& sG, VEC_TENSOR& sVec, int idx_in_warpgroup, int sub_seq_len, int k_offset, int tmem_addr_inter) {
    int row = idx_in_warpgroup % 64;
    int sub_tile_i = row / 16;  // 0, 1, 2, or 3
    float res_inter[TileK];
    if (row < sub_seq_len) {
        int g_first_row = min(sub_tile_i * 16, sub_seq_len - 1);
#pragma unroll
        for (int i = 0; i < TileK / 4; ++i) {
            int y = i * 4 + k_offset;
            float4 g = *reinterpret_cast<float4*>(&sG(row, y));
            nvbf16x4 v = *reinterpret_cast<nvbf16x4*>(&sVec(row, y));
            float2 va = __bfloat1622float2(v.a);
            float2 vb = __bfloat1622float2(v.b);
            float4 g_ref = *reinterpret_cast<float4*>(&sG(g_first_row, y));
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g)[0], reinterpret_cast<float2*>(&g_ref)[0]);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g)[1], reinterpret_cast<float2*>(&g_ref)[1]);
            s1.x = exp2f(s1.x);
            s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x);
            s2.y = exp2f(s2.y);
            reinterpret_cast<float2*>(&res_inter[i * 4])[0] = float2_mul(s1, va);
            reinterpret_cast<float2*>(&res_inter[i * 4])[1] = float2_mul(s2, vb);
        }
    } else {
#pragma unroll
        for (int i = 0; i < TileK; ++i) {
            res_inter[i] = 0.0f;
        }
    }
    ku::tmem_st_32dp32bNx<TileK>(tmem_addr_inter, res_inter);
}

// fwd_setup_A_inter_all_QK: inter-only, combined Q + K
// Threads 0-63 → gated Q (inter), Threads 64-127 → gated K (inter)
template <int TileK, typename G_TENSOR, typename Q_TENSOR, typename K_TENSOR>
__forceinline__ __device__ void
fwd_setup_A_inter_all_QK(
    G_TENSOR& sG,
    Q_TENSOR& sQ,
    K_TENSOR& sK,
    int idx_in_warpgroup,
    int sub_seq_len,
    int k_offset,
    int tmem_addr_inter) {
    if (idx_in_warpgroup < 64) {
        fwd_setup_A_inter_all<TileK>(sG, sQ, idx_in_warpgroup, sub_seq_len, k_offset, tmem_addr_inter + k_offset);
    } else {
        fwd_setup_A_inter_all<TileK>(sG, sK, idx_in_warpgroup, sub_seq_len, k_offset, tmem_addr_inter + k_offset);
    }
}

// fwd_setup_A_inter_intra_all_QK: fused inter+intra, combined Q + K
// Threads 0-63 → gated Q (inter + intra), Threads 64-127 → gated K (inter + intra)
template <int TileK, typename G_TENSOR, typename Q_TENSOR, typename K_TENSOR>
__forceinline__ __device__ void
fwd_setup_A_inter_intra_all_QK(
    G_TENSOR& sG,
    Q_TENSOR& sQ,
    K_TENSOR& sK,
    int idx_in_warpgroup,
    int sub_seq_len,
    int k_offset,
    int tmem_addr_inter,
    int tmem_addr_intra) {
    if (idx_in_warpgroup < 64) {
        fwd_setup_A_inter_intra_all<TileK>(
            sG, sQ, idx_in_warpgroup, sub_seq_len, k_offset, tmem_addr_inter + k_offset, tmem_addr_intra + k_offset);
    } else {
        fwd_setup_A_inter_intra_all<TileK>(
            sG, sK, idx_in_warpgroup, sub_seq_len, k_offset, tmem_addr_inter + k_offset, tmem_addr_intra + k_offset);
    }
}

// ============================================================
// Forward Epilogue: T2R + Causal Mask + R2G / R2S helper functions
// ============================================================
//
// After MMA finishes NumKIters accumulations, the full 64×64 QK and KK
// matrices reside in TMEM. The CudaCore warpgroups perform:
//   1. T2R: read TMEM → registers (ku::tmem_ld_32dp32bNx<TileT>)
//   2. Causal mask: apply lower-triangular mask in registers (zero j > i)
//   3. R2G: convert float → bf16 and store to global memory (store_256b)
//   4. R2S (KK only): write masked KK to SMEM for inverse warpgroup
//
// Thread mapping for T2R (128 threads per WG, 2 WGs):
//   row = idx_in_warpgroup % 64     (each thread owns one row of 64×64)
//   Lower 64 threads (idx < 64):    QK[row, 0..63] from TMEM QK lanes
//   Upper 64 threads (idx >= 64):   KK[row, 0..63] from TMEM QK lanes (offset by 64 TMEM lanes)
//
// TMEM lane mapping:
//   QK_02 = lanes 0,2 → threads with (warp_in_wg % 2 == 0), i.e. row 0-15 and 32-47
//   QK_13 = lanes 1,3 → threads with (warp_in_wg % 2 == 1), i.e. row 16-31 and 48-63
//
// Causal mask (lower-triangular):
//   QK[i, j] = 0 when j > i (only keep j <= i)
//   KK[i, j] = 0 when j > i (only keep j <= i)
//   Applied element-wise in registers after T2R.
//
// The 64 columns in TMEM map to the 64 sequence positions within the chunk.
// Column layout mirrors the MMA accumulation pattern:
//   col 0-15:  sub_tile 0 (positions 0-15)
//   col 16-31: sub_tile 1 (positions 16-31)
//   col 32-47: sub_tile 2 (positions 32-47)
//   col 48-63: sub_tile 3 (positions 48-63)

// fwd_epilogue_t2r_qk: T2R for QK result, apply scale + causal mask, R2G to global memory
// Called by threads responsible for QK output in each CudaCore warpgroup.
// Each thread reads its row of the 64×64 QK matrix from TMEM,
// multiplies every element by `scale`, applies lower-triangular causal mask,
// and writes bf16 to global memory.
//
// TMEM row mapping (64 threads → 64 rows):
//   Warp 0 (threads  0-31): dp-lanes 0-31  → rows 0-31
//   Warp 1 (threads 32-63): dp-lanes 0-31  → rows 32-63  (lane16 bank offset)
//   A single ku::tmem_ld_32dp32bNx<TileT> call per thread reads all TileT columns
//   for that thread's row.
//
// Output addressing (Aqk_out_ptr layout: [total_tokens, H, BT]):
//   qk_out_row_base = Aqk_out_ptr + (token_offset + tile_idx * TileT + row) * H * BT
//                                  + head_idx * BT
//   Each thread writes its full row (TileT bf16 values) via store_256b.
//
// tmem_qk_addr: TMEM base address for QK data (= TmemAllocation::QK)
// idx_in_warpgroup: 0..127 within this WG (only threads 0..63 should call this)
// sub_seq_len: actual sequence length within this tile (for bounds checking)
// scale: per-element scaling factor (params.scale)
// qk_out_base: global memory pointer for this thread's row output
template <int TileT>
__forceinline__ __device__ void
fwd_epilogue_t2r_qk(int tmem_qk_addr, int idx_in_warpgroup, int sub_seq_len, float scale, __nv_bfloat16* qk_out_base) {
    // T2R: read full row (TileT floats) from TMEM
    float res[TileT];
    ku::tcgen05_after_thread_sync();
    ku::tmem_ld_32dp32bNx<TileT>(tmem_qk_addr, res);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    int row = idx_in_warpgroup % 64;

    // Apply scale and causal mask:
    //   - Multiply valid elements (j <= row && j < sub_seq_len) by scale
    //   - Zero out elements in the strictly upper triangle (j > row)
    //   - Zero out out-of-bounds columns (j >= sub_seq_len)
    int mask_limit = min(row, sub_seq_len - 1);  // last valid column
#pragma unroll
    for (int j = 0; j < TileT; ++j) {
        if (j > mask_limit) {
            res[j] = 0.0f;
        } else {
            res[j] *= scale;
        }
    }

    // R2G: convert to bf16 and write to global memory
    // Only write if this row is valid (row < sub_seq_len)
    // NOTE: we do not need write zeros here because this is a direct R2G copy, writing zero causes oob write
    if (row < sub_seq_len) {
#pragma unroll
        for (int i = 0; i < TileT / 16; ++i) {
            ku::bf16x16 out;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                reinterpret_cast<__nv_bfloat162*>(&out)[j] =
                    __float22bfloat162_rn(reinterpret_cast<float2*>(&res[i * 16])[j]);
            }
            ku::store_256b(&out, qk_out_base + i * 16);
        }
    }
}

// fwd_epilogue_t2r_kk: T2R for KK result, apply beta scaling + causal mask, R2S
// Called by upper 64 threads (idx_in_warpgroup >= 64) of each CudaCore warpgroup.
// Each thread reads its row of the 64×64 KK matrix from TMEM,
// applies beta scaling (row i *= beta[i]), lower-triangular causal mask,
// converts to fp16, and writes to SMEM for the inverse warpgroup.
//
// tmem_kk_addr: TMEM base address for this thread's KK data
// idx_in_warpgroup: 0..127 within this WG (only threads 64..127 should call this)
// sub_seq_len: actual sequence length within this tile
// beta_row: beta scaling factor for this thread's row (from beta_smem)
// sKK: SMEM tensor for inverse warpgroup (fp16)
template <int TileT, bool UseTF32Inverse, bool RoundingTF32, typename KK_SMEM_TENSOR>
__forceinline__ __device__ void
fwd_epilogue_t2r_kk(int tmem_kk_addr, int idx_in_warpgroup, int sub_seq_len, float beta_row, KK_SMEM_TENSOR& sKK) {
    // T2R: read full row (TileT floats) from TMEM
    float res[TileT];
    ku::tcgen05_after_thread_sync();
    ku::tmem_ld_32dp32bNx<TileT>(tmem_kk_addr, res);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    int row = idx_in_warpgroup % 64;

    // Apply beta scaling and causal mask:
    //   - Multiply valid elements (j <= row && j < sub_seq_len) by beta_row
    //   - Zero out elements in the strictly upper triangle (j > row)
    //   - Zero out out-of-bounds columns (j >= sub_seq_len)
    int mask_limit = min(row, sub_seq_len - 1);
#pragma unroll
    for (int j = 0; j < TileT; ++j) {
        if (j > mask_limit) {
            res[j] = 0.0f;
        } else {
            res[j] *= float(beta_row);
        }
    }

    if (row < sub_seq_len) {
// R2S: convert to fp16 and write to SMEM for inverse warpgroup
#pragma unroll
        for (int i = 0; i < TileT / 4; ++i) {
            // Convert 4 floats → 4 fp16 values, store as 2×half2
            float2 f01 = reinterpret_cast<float2*>(res)[i * 2];
            float2 f23 = reinterpret_cast<float2*>(res)[i * 2 + 1];
            if constexpr (!UseTF32Inverse) {
                __half2 h01 = __float22half2_rn(f01);
                __half2 h23 = __float22half2_rn(f23);
                *reinterpret_cast<__half2*>(&sKK(row, i * 4)) = h01;
                *reinterpret_cast<__half2*>(&sKK(row, i * 4 + 2)) = h23;
            } else {
                if constexpr (RoundingTF32) {
                    sKK(row, i * 4) = tfloat32_t(f01.x);
                    sKK(row, i * 4 + 1) = tfloat32_t(f01.y);
                    sKK(row, i * 4 + 2) = tfloat32_t(f23.x);
                    sKK(row, i * 4 + 3) = tfloat32_t(f23.y);
                } else {
                    *reinterpret_cast<float2*>(&sKK(row, i * 4)) = f01;
                    *reinterpret_cast<float2*>(&sKK(row, i * 4 + 2)) = f23;
                }
            }
        }
    } else {
// NOTE: must write zeros to SMEM of invalid token postitions in the current chunk
// R2S zero
#pragma unroll
        for (int i = 0; i < TileT; ++i) {
            if constexpr (!UseTF32Inverse) {
                sKK(row, i) = half_t(0.0f);
            } else {
                if constexpr (RoundingTF32) {
                    sKK(row, i) = tfloat32_t(0.0f);
                } else {
                    *reinterpret_cast<float*>(&sKK(row, i)) = 0.0f;
                }
            }
        }
    }
}

// fwd_epilogue_qk_kk: combined T2R + causal mask + output for both QK and KK
// Called by all 128 threads in each CudaCore warpgroup.
// Lower 64 threads handle QK (scale + mask + R2G), upper 64 threads handle KK (beta + mask + R2S).
//
// tmem_qk_addr: TMEM base address for QK / KK data (same region, different dp-lane banks)
// idx_in_warpgroup: 0..127
// sub_seq_len: actual sequence length within this tile
// scale: per-element scaling factor for QK (params.scale)
// beta_row: beta scaling factor for this thread's row (from beta_smem)
// qk_out_base: global memory pointer for QK row output (bf16, lower threads only)
// sKK: SMEM tensor for KK inverse (fp16, upper threads only)
template <int TileT, bool UseTF32Inverse, bool RoundingTF32, typename KK_SMEM_TENSOR>
__forceinline__ __device__ void
fwd_epilogue_qk_kk(
    int tmem_qk_addr,
    int idx_in_warpgroup,
    int sub_seq_len,
    float scale,
    float beta_row,
    __nv_bfloat16* qk_out_base,
    KK_SMEM_TENSOR& sKK) {
    if (idx_in_warpgroup < 64) {
        fwd_epilogue_t2r_qk<TileT>(tmem_qk_addr, idx_in_warpgroup, sub_seq_len, scale, qk_out_base);
    } else {
        fwd_epilogue_t2r_kk<TileT, UseTF32Inverse, RoundingTF32>(
            tmem_qk_addr, idx_in_warpgroup, sub_seq_len, beta_row, sKK);
    }
}

}  // namespace kda::sm100
