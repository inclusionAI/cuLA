// Compute the full BF16→TF32 permutation table for operand A and B.
//
// For each (thread, value) pair in TF32 layout, find the corresponding (thread, value)
// pair in BF16 layout that maps to the same logical tile position. This tells us exactly
// which cross-thread shuffles are needed.
//
// Key findings:
//   - All shuffles stay within the same warp (no cross-warp communication needed)
//   - Operand A: shuffles happen within groups of 4 threads (t0 dimension)
//   - Operand B: same pattern as A
//
// Build:
//   nvcc -std=c++17 -I<cutlass_include_path> analyze_permutation.cu -o analyze_permutation

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cstdio>

using namespace cute;

// Given LayoutA_TV: (thread, value) -> tile_idx
// We want to find: for each (thread, value) in TF32, which (thread, value) in BF16 maps to the same tile position

int main() {
    // BF16 MMA 16x8x8 - LayoutA_TV: (thread, value) -> tile
    // Thread layout: (_4,_8) with stride (_32,_1)  => tid = t0*32 + t1
    // Value layout:  (_2,_2) with stride (_16,_8)  => val_idx = v0*16 + v1*8
    // Combined: tile_pos = t0*32 + t1 + v0*16 + v1*8

    // TF32 MMA 16x8x8 - LayoutA_TV: (thread, value) -> tile
    // Thread layout: (_4,_8) with stride (_16,_1)  => tid = t0*16 + t1
    // Value layout:  (_2,_2) with stride (_8,_64) => val_idx = v0*8 + v1*64
    // Combined: tile_pos = t0*16 + t1 + v0*8 + v1*64

    printf("=== LayoutA_TV Analysis ===\n");
    printf("BF16: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))\n");
    printf("TF32: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))\n\n");

    int bf16_map[64][4]; // bf16_map[tid][vid] = tile_pos
    int tf32_map[64][4]; // tf32_map[tid][vid] = tile_pos

    for (int tid = 0; tid < 64; tid++) {
        int t0 = tid / 8;
        int t1 = tid % 8;
        if (tid >= 32) continue;
        
        for (int vid = 0; vid < 4; vid++) {
            int v0 = vid % 2;
            int v1 = vid / 2;

            bf16_map[tid][vid] = t0*32 + t1 + v0*16 + v1*8;
            tf32_map[tid][vid] = t0*16 + t1 + v0*8 + v1*64;
        }
    }

    printf("BF16 -> TF32 permutation for operand A (32 threads, 4 values each):\n");
    printf("For each (bf16_tid, bf16_vid), find (tf32_tid, tf32_vid) with same tile_pos\n\n");
    
    // Build reverse map: tile_pos -> (tid, vid) for TF32
    int tf32_reverse_tid[512], tf32_reverse_vid[512];
    for (int i = 0; i < 512; i++) { tf32_reverse_tid[i] = -1; tf32_reverse_vid[i] = -1; }
    for (int tid = 0; tid < 32; tid++) {
        for (int vid = 0; vid < 4; vid++) {
            tf32_reverse_tid[tf32_map[tid][vid]] = tid;
            tf32_reverse_vid[tf32_map[tid][vid]] = vid;
        }
    }

    // For each thread in BF16, find where its values need to go in TF32
    printf("BF16(tid,vid) -> tile_pos -> TF32(tid,vid)\n");
    for (int tid = 0; tid < 32; tid++) {
        for (int vid = 0; vid < 4; vid++) {
            int pos = bf16_map[tid][vid];
            int dst_tid = tf32_reverse_tid[pos];
            int dst_vid = tf32_reverse_vid[pos];
            if (dst_tid >= 0) {
                printf("BF16(%2d,%d) -> pos=%3d -> TF32(%2d,%d)", tid, vid, pos, dst_tid, dst_vid);
                if (tid != dst_tid) printf("  ** NEED SHUFFLE **");
                printf("\n");
            } else {
                printf("BF16(%2d,%d) -> pos=%3d -> NOT IN TF32\n", tid, vid, pos);
            }
        }
    }

    // Same for operand B
    printf("\n=== LayoutB_TV Analysis ===\n");
    printf("BF16: ((_4,_8),_2):((_16,_1),_8)\n");
    printf("TF32: ((_4,_8),_2):((_8,_1),_32)\n\n");

    // B tile is 8xK (N=8, K=8 per atom)
    // BF16: pos = t0*16 + t1 + v0*8
    // TF32: pos = t0*8 + t1 + v0*32

    int bf16_mapB[32][2], tf32_mapB[32][2];
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8;
        int t1 = tid % 8;
        for (int vid = 0; vid < 2; vid++) {
            bf16_mapB[tid][vid] = t0*16 + t1 + vid*8;
            tf32_mapB[tid][vid] = t0*8 + t1 + vid*32;
        }
    }

    int tf32_reverse_tidB[256], tf32_reverse_vidB[256];
    for (int i = 0; i < 256; i++) { tf32_reverse_tidB[i] = -1; tf32_reverse_vidB[i] = -1; }
    for (int tid = 0; tid < 32; tid++) {
        for (int vid = 0; vid < 2; vid++) {
            tf32_reverse_tidB[tf32_mapB[tid][vid]] = tid;
            tf32_reverse_vidB[tf32_mapB[tid][vid]] = vid;
        }
    }

    printf("BF16(tid,vid) -> tile_pos -> TF32(tid,vid)\n");
    for (int tid = 0; tid < 32; tid++) {
        for (int vid = 0; vid < 2; vid++) {
            int pos = bf16_mapB[tid][vid];
            int dst_tid = tf32_reverse_tidB[pos];
            int dst_vid = tf32_reverse_vidB[pos];
            if (dst_tid >= 0) {
                printf("BF16(%2d,%d) -> pos=%3d -> TF32(%2d,%d)", tid, vid, pos, dst_tid, dst_vid);
                if (tid != dst_tid) printf("  ** SHUFFLE (src_tid=%d -> dst_tid=%d) **", tid, dst_tid);
                printf("\n");
            } else {
                printf("BF16(%2d,%d) -> pos=%3d -> NOT IN TF32\n", tid, vid, pos);
            }
        }
    }

    // Summary: check if all values stay within the same warp (same group of 32 threads)
    printf("\n=== Summary ===\n");
    bool all_within_warp_A = true, all_same_thread_A = true;
    for (int tid = 0; tid < 32; tid++) {
        for (int vid = 0; vid < 4; vid++) {
            int pos = bf16_map[tid][vid];
            int dst_tid = tf32_reverse_tid[pos];
            if (dst_tid < 0) continue;
            if (dst_tid != tid) all_same_thread_A = false;
            if (dst_tid / 32 != tid / 32) all_within_warp_A = false;
        }
    }
    printf("Operand A: all_same_thread=%d, all_within_warp=%d\n", all_same_thread_A, all_within_warp_A);

    bool all_within_warp_B = true, all_same_thread_B = true;
    for (int tid = 0; tid < 32; tid++) {
        for (int vid = 0; vid < 2; vid++) {
            int pos = bf16_mapB[tid][vid];
            int dst_tid = tf32_reverse_tidB[pos];
            if (dst_tid < 0) continue;
            if (dst_tid != tid) all_same_thread_B = false;
            if (dst_tid / 32 != tid / 32) all_within_warp_B = false;
        }
    }
    printf("Operand B: all_same_thread=%d, all_within_warp=%d\n", all_same_thread_B, all_within_warp_B);

    return 0;
}
