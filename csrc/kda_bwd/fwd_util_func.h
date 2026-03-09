#pragma once

#include <cute/tensor.hpp>
#include "basic.h"
#include "helpers.h"

namespace sm100 {

using namespace cute;

// ============================================================
// Forward Prologue: B-matrix (SMEM) helper functions
// ============================================================
//
// B-matrix formula (stored to SMEM for MMA consumption):
//   inter: exp2(g_first - g[x]) * K[x]     (g_first = g[sub_tile_i * 16])
//   intra: exp2(g_half  - g[x]) * K[x]     (g_half  = g[sub_tile_i * 16 + 8])
//
// Backward vs Forward B-matrix layout difference:
//   Forward computes Q/K @ K^T, Backward computes dAqk/dAkk @ K.
//   Forward MMA:  64 × X × 32 (M×N×K), reduces head dim (K=32), B = K^T
//     B-matrix shape = (N × K) = (SUB_T_TILE × K_TILE), K-major
//     uses SmemLayoutMatBTF32 = Layout_K_SW128_Atom, stored as sKG(x_local, y)
//   Backward MMA: 64 × 32 × X (M×N×K), reduces chunk dim (K=SUB_T_TILE), B = K
//     B-matrix shape = (K × N) = (SUB_T_TILE × K_TILE), MN-major
//     uses SmemLayoutMatBTF32Tranposed = Layout_MN_SW128_32B_Atom, stored as sKG(y, x_local)
//
// SmemLayoutMatBTF32<1> = (SUB_T_TILE, K_TILE) = (16, 32), K-major layout
//
// Thread mapping (128 threads per WG, 16 rows per sub_tile):
//   x_local = idx_in_warpgroup / 8           (row within sub_tile, 0..15)
//   y       = idx_in_warpgroup % 8 * 4       (column group, 0..28 step 4)
//   Each thread writes 4 consecutive tf32 values (128 bits) to SMEM.
//
// Store pattern: sKG(x_local, y) + KG_OFFSET * index
//   where KG_OFFSET = SUB_T_TILE * K_TILE (stride between sub_tile buffers)

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
//   Returns g_half_0, g_first_1, g_first_2, g_first_3 for potential A-matrix reuse.
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void fwd_setup_kg_col0_4out(
    G_TENSOR &sG, K_TENSOR &sK,
    KG_TENSOR &sKG_inter, KG_TENSOR &sKG_intra,
    int idx_in_warpgroup, int sub_seq_len,
    float4 &g_half_0 /*out*/, float4 &g_first_1 /*out*/,
    float4 &g_first_2 /*out*/, float4 &g_first_3 /*out*/) {
    int y = idx_in_warpgroup % 8 * 4;
    // Load 4 g_ref values (3 SMEM reads for g_ref; g_half_0 needs row 8)
    g_half_0  = *reinterpret_cast<float4*>(&sG(min(0 * 16 + 8, sub_seq_len - 1), y));
    g_first_1 = *reinterpret_cast<float4*>(&sG(min(1 * 16, sub_seq_len - 1), y));
    g_first_2 = *reinterpret_cast<float4*>(&sG(min(2 * 16, sub_seq_len - 1), y));
    g_first_3 = *reinterpret_cast<float4*>(&sG(min(3 * 16, sub_seq_len - 1), y));
    // K data from sub_tile j=0
    int x = idx_in_warpgroup / 8 + 0 * 16;
    if (x < sub_seq_len) {
        float4 g = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        float2 kf_a = __bfloat1622float2(k.a);
        float2 kf_b = __bfloat1622float2(k.b);
        float2 g_a = reinterpret_cast<float2*>(&g)[0];
        float2 g_b = reinterpret_cast<float2*>(&g)[1];
        // intra(0,0): exp2(g_half_0 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_half_0)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_half_0)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            float4 res;
            reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
            reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 0, res);
        }
        // inter(1,0): exp2(g_first_1 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_1)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_1)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            float4 res;
            reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
            reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 0, res);
        }
        // inter(2,0): exp2(g_first_2 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_2)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_2)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            float4 res;
            reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
            reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 1, res);
        }
        // inter(3,0): exp2(g_first_3 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_3)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_3)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
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

// fwd_setup_kg_col1_3out: column j=1, 3 outputs (1 intra + 2 inter)
//   Loads K_1 + G data once, computes:
//     intra(1,1): exp2(g_half_1 - g_1[x]) * K_1[x]  → sKG_intra index 1
//     inter(2,1): exp2(g_first_2 - g_1[x]) * K_1[x] → sKG_inter index 2
//     inter(3,1): exp2(g_first_3 - g_1[x]) * K_1[x] → sKG_inter index 4
//   Returns g_half_1, g_first_2, g_first_3 for potential A-matrix reuse.
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void fwd_setup_kg_col1_3out(
    G_TENSOR &sG, K_TENSOR &sK,
    KG_TENSOR &sKG_inter, KG_TENSOR &sKG_intra,
    int idx_in_warpgroup, int sub_seq_len,
    float4 &g_half_1 /*out*/, float4 &g_first_2 /*out*/,
    float4 &g_first_3 /*out*/) {
    int y = idx_in_warpgroup % 8 * 4;
    // Load 3 g_ref values
    g_half_1  = *reinterpret_cast<float4*>(&sG(min(1 * 16 + 8, sub_seq_len - 1), y));
    g_first_2 = *reinterpret_cast<float4*>(&sG(min(2 * 16, sub_seq_len - 1), y));
    g_first_3 = *reinterpret_cast<float4*>(&sG(min(3 * 16, sub_seq_len - 1), y));
    // K data from sub_tile j=1
    int x = idx_in_warpgroup / 8 + 1 * 16;
    if (x < sub_seq_len) {
        float4 g = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        float2 kf_a = __bfloat1622float2(k.a);
        float2 kf_b = __bfloat1622float2(k.b);
        float2 g_a = reinterpret_cast<float2*>(&g)[0];
        float2 g_b = reinterpret_cast<float2*>(&g)[1];
        // intra(1,1): exp2(g_half_1 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_half_1)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_half_1)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            float4 res;
            reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
            reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 1, res);
        }
        // inter(2,1): exp2(g_first_2 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_2)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_2)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            float4 res;
            reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
            reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
            store_128b(&sKG_inter(idx_in_warpgroup / 8, y) + KG_OFFSET * 2, res);
        }
        // inter(3,1): exp2(g_first_3 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_3)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_3)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
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

// fwd_setup_kg_col2_2out: column j=2, 2 outputs (1 intra + 1 inter)
//   Loads K_2 + G data once, computes:
//     intra(2,2): exp2(g_half_2 - g_2[x]) * K_2[x]  → sKG_intra index 2
//     inter(3,2): exp2(g_first_3 - g_2[x]) * K_2[x] → sKG_inter index 5
//   Returns g_half_2, g_first_3 for potential A-matrix reuse.
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void fwd_setup_kg_col2_2out(
    G_TENSOR &sG, K_TENSOR &sK,
    KG_TENSOR &sKG_inter, KG_TENSOR &sKG_intra,
    int idx_in_warpgroup, int sub_seq_len,
    float4 &g_half_2 /*out*/, float4 &g_first_3 /*out*/) {
    int y = idx_in_warpgroup % 8 * 4;
    // Load 2 g_ref values
    g_half_2  = *reinterpret_cast<float4*>(&sG(min(2 * 16 + 8, sub_seq_len - 1), y));
    g_first_3 = *reinterpret_cast<float4*>(&sG(min(3 * 16, sub_seq_len - 1), y));
    // K data from sub_tile j=2
    int x = idx_in_warpgroup / 8 + 2 * 16;
    if (x < sub_seq_len) {
        float4 g = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        float2 kf_a = __bfloat1622float2(k.a);
        float2 kf_b = __bfloat1622float2(k.b);
        float2 g_a = reinterpret_cast<float2*>(&g)[0];
        float2 g_b = reinterpret_cast<float2*>(&g)[1];
        // intra(2,2): exp2(g_half_2 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_half_2)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_half_2)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            float4 res;
            reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, kf_a);
            reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, kf_b);
            store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 2, res);
        }
        // inter(3,2): exp2(g_first_3 - g[x]) * K[x]
        {
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_first_3)[0], g_a);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_first_3)[1], g_b);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
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

// fwd_setup_kg_col3_1out: column j=3, 1 output (intra only)
//   Loads K_3 + G data once, computes:
//     intra(3,3): exp2(g_half_3 - g_3[x]) * K_3[x]  → sKG_intra index 3
//   Returns g_half_3 for potential A-matrix reuse.
template <typename G_TENSOR, typename K_TENSOR, typename KG_TENSOR, int KG_OFFSET>
__forceinline__ __device__ void fwd_setup_kg_col3_1out(
    G_TENSOR &sG, K_TENSOR &sK,
    KG_TENSOR &sKG_intra,
    int idx_in_warpgroup, int sub_seq_len,
    float4 &g_half_3 /*out*/) {
    int y = idx_in_warpgroup % 8 * 4;
    // Load 1 g_ref value
    g_half_3 = *reinterpret_cast<float4*>(&sG(min(3 * 16 + 8, sub_seq_len - 1), y));
    // K data from sub_tile j=3
    int x = idx_in_warpgroup / 8 + 3 * 16;
    if (x < sub_seq_len) {
        float4 g = *reinterpret_cast<float4*>(&sG(x, y));
        nvbf16x4 k = *reinterpret_cast<nvbf16x4*>(&sK(x, y));
        // intra(3,3): exp2(g_half_3 - g[x]) * K[x]
        float2 s1 = float2_sub(reinterpret_cast<float2*>(&g_half_3)[0], reinterpret_cast<float2*>(&g)[0]);
        float2 s2 = float2_sub(reinterpret_cast<float2*>(&g_half_3)[1], reinterpret_cast<float2*>(&g)[1]);
        s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
        s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
        float4 res;
        reinterpret_cast<float2*>(&res)[0] = float2_mul(s1, __bfloat1622float2(k.a));
        reinterpret_cast<float2*>(&res)[1] = float2_mul(s2, __bfloat1622float2(k.b));
        store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 3, res);
    } else {
        float4 z = {0.0f, 0.0f, 0.0f, 0.0f};
        store_128b(&sKG_intra(idx_in_warpgroup / 8, y) + KG_OFFSET * 3, z);
    }
}

// ============================================================
// Forward Prologue: A-matrix (TMEM) helper functions
// ============================================================
//
// A-matrix formula (stored to TMEM for MMA consumption):
//   inter: exp2(g[row] - g_first) * Vec[row]   (Vec = Q for QK MMA, K for KK MMA)
//   intra: exp2(g[row] - g_half)  * Vec[row]   (Vec = Q for QK MMA, K for KK MMA)
//
// Thread mapping for TMEM store (32 dp-lanes):
//   row = idx_in_warpgroup % 64  (each thread owns one row of the 64-row tile)
//   Lower 64 threads (idx < 64): write gated Q → TMEM for QK MMA
//   Upper 64 threads (idx >= 64): write gated K → TMEM for KK MMA
//
// Each thread computes K_TILE float values for its row, then does a single
// tmem_st_32dp32bNx<K_TILE> store.
//
// The tmem_addr parameter specifies the TMEM base address for this A-matrix.
// Different sub_tiles write to different TMEM address offsets.

// fwd_setup_A_inter: compute exp2(g[row] - g_first) * Vec[row] → TMEM
// Covers all 64 rows; only rows in [sub_tile_i*16, sub_tile_i*16+16) get real data.
// Other rows are zeroed. Vec is Q or K (bf16 from SMEM).
template <int K_TILE, typename G_TENSOR, typename VEC_TENSOR>
__forceinline__ __device__ void fwd_setup_A_inter(
    G_TENSOR &sG, VEC_TENSOR &sVec,
    int sub_tile_i, int idx_in_warpgroup, int sub_seq_len,
    int tmem_addr) {
    int row = idx_in_warpgroup % 64;
    float res[K_TILE];
    if (row >= sub_tile_i * 16 && row < sub_tile_i * 16 + 16 && row < sub_seq_len) {
        int g_first_row = min(sub_tile_i * 16, sub_seq_len - 1);
        #pragma unroll
        for (int i = 0; i < K_TILE / 4; ++i) {
            int y = i * 4;
            float4 g     = *reinterpret_cast<float4*>(&sG(row, y));
            float4 g_ref = *reinterpret_cast<float4*>(&sG(g_first_row, y));
            nvbf16x4 v   = *reinterpret_cast<nvbf16x4*>(&sVec(row, y));
            // exp2(g[row] - g_first)
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g)[0], reinterpret_cast<float2*>(&g_ref)[0]);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g)[1], reinterpret_cast<float2*>(&g_ref)[1]);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            reinterpret_cast<float2*>(&res[i * 4])[0] = float2_mul(s1, __bfloat1622float2(v.a));
            reinterpret_cast<float2*>(&res[i * 4])[1] = float2_mul(s2, __bfloat1622float2(v.b));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < K_TILE; ++i) res[i] = 0.0f;
    }
    tmem_st_32dp32bNx<K_TILE>(tmem_addr, res);
}

// fwd_setup_A_intra: compute exp2(g[row] - g_half) * Vec[row] → TMEM
template <int K_TILE, typename G_TENSOR, typename VEC_TENSOR>
__forceinline__ __device__ void fwd_setup_A_intra(
    G_TENSOR &sG, VEC_TENSOR &sVec,
    int sub_tile_i, int idx_in_warpgroup, int sub_seq_len,
    int tmem_addr) {
    int row = idx_in_warpgroup % 64;
    float res[K_TILE];
    if (row >= sub_tile_i * 16 && row < sub_tile_i * 16 + 16 && row < sub_seq_len) {
        int g_half_row = min(sub_tile_i * 16 + 8, sub_seq_len - 1);
        #pragma unroll
        for (int i = 0; i < K_TILE / 4; ++i) {
            int y = i * 4;
            float4 g     = *reinterpret_cast<float4*>(&sG(row, y));
            float4 g_ref = *reinterpret_cast<float4*>(&sG(g_half_row, y));
            nvbf16x4 v   = *reinterpret_cast<nvbf16x4*>(&sVec(row, y));
            // exp2(g[row] - g_half)
            float2 s1 = float2_sub(reinterpret_cast<float2*>(&g)[0], reinterpret_cast<float2*>(&g_ref)[0]);
            float2 s2 = float2_sub(reinterpret_cast<float2*>(&g)[1], reinterpret_cast<float2*>(&g_ref)[1]);
            s1.x = exp2f(s1.x); s1.y = exp2f(s1.y);
            s2.x = exp2f(s2.x); s2.y = exp2f(s2.y);
            reinterpret_cast<float2*>(&res[i * 4])[0] = float2_mul(s1, __bfloat1622float2(v.a));
            reinterpret_cast<float2*>(&res[i * 4])[1] = float2_mul(s2, __bfloat1622float2(v.b));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < K_TILE; ++i) res[i] = 0.0f;
    }
    tmem_st_32dp32bNx<K_TILE>(tmem_addr, res);
}

// fwd_setup_A_inter_QK: compute both gated Q and gated K for inter A-matrices
// Lower 64 threads handle Q (for QK MMA), upper 64 handle K (for KK MMA)
// Both use the same exponent: exp2(g[row] - g_first)
template <int K_TILE, typename G_TENSOR, typename Q_TENSOR, typename K_TENSOR>
__forceinline__ __device__ void fwd_setup_A_inter_QK(
    G_TENSOR &sG, Q_TENSOR &sQ, K_TENSOR &sK,
    int sub_tile_i, int idx_in_warpgroup, int sub_seq_len,
    int tmem_addr_q, int tmem_addr_k) {
    // Lower 64 threads → gated Q → tmem_addr_q
    // Upper 64 threads → gated K → tmem_addr_k
    if (idx_in_warpgroup < 64) {
        fwd_setup_A_inter<K_TILE>(sG, sQ, sub_tile_i, idx_in_warpgroup, sub_seq_len, tmem_addr_q);
    } else {
        fwd_setup_A_inter<K_TILE>(sG, sK, sub_tile_i, idx_in_warpgroup, sub_seq_len, tmem_addr_k);
    }
}

// fwd_setup_A_intra_QK: compute both gated Q and gated K for intra A-matrices
template <int K_TILE, typename G_TENSOR, typename Q_TENSOR, typename K_TENSOR>
__forceinline__ __device__ void fwd_setup_A_intra_QK(
    G_TENSOR &sG, Q_TENSOR &sQ, K_TENSOR &sK,
    int sub_tile_i, int idx_in_warpgroup, int sub_seq_len,
    int tmem_addr_q, int tmem_addr_k) {
    if (idx_in_warpgroup < 64) {
        fwd_setup_A_intra<K_TILE>(sG, sQ, sub_tile_i, idx_in_warpgroup, sub_seq_len, tmem_addr_q);
    } else {
        fwd_setup_A_intra<K_TILE>(sG, sK, sub_tile_i, idx_in_warpgroup, sub_seq_len, tmem_addr_k);
    }
}

} // namespace sm100
