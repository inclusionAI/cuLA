// Derive per-thread (row, k) ownership for BF16 and TF32 MMA operand A & B,
// and compute the exact __shfl_sync source lane pattern.
//
// This is the key analysis program that led to the shuffle algorithm:
//   - Shows exactly which (M-row, K-col) each thread holds for each value index
//   - Builds the reverse map: for each TF32 (tid, vid), find the BF16 source (tid, vid)
//   - Reveals the pattern: shuffles happen within t0 groups of 4 threads
//
// Build:
//   nvcc -std=c++17 -I<cutlass_include_path> analyze_shuffle.cu -o analyze_shuffle

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cstdio>

using namespace cute;

int main() {
    // Focus on understanding the register layout in the physical per-thread fragment
    // Both MMA types use fragment layout ((_2,_2),_1,_4):((_1,_2),_0,_4)
    // This means each thread holds 16 values (4 values * 4 k-iterations), arranged as:
    //   val[0..3] for k_iter=0, val[4..7] for k_iter=1, etc.
    // Within each k_iter, the 4 values are (v0,v1) with v0 in {0,1}, v1 in {0,1}
    //   physical register idx = v0 + v1*2 + k_iter*4

    // The MMA atom is M16xN8xK8, tiled by Layout<Shape<_1,_2,_1>> giving 16x16x8
    // The K dimension is tiled 4x to give K=32 total
    
    // For operand A (M x K = 16 x 32), each thread sees (M_frag x 1 x K_iter) = (4 x 1 x 4)
    // The question is: how do the 4 values per K-iteration map to (row, col) in the tile?

    // BF16 LayoutA_TV: ((_4,_8), (_2,_2)) : ((_32,_1), (_16,_8))
    //   For thread tid: t0 = (tid / 8) % 4, t1 = tid % 8
    //   For value vid:  v0 = vid % 2,      v1 = vid / 2
    //   tile_pos = t0*32 + t1*1 + v0*16 + v1*8
    //   For M16xK8 atom (col-major): m = pos % 16, k = pos / 16

    // TF32 LayoutA_TV: ((_4,_8), (_2,_2)) : ((_16,_1), (_8,_64))
    //   tile_pos = t0*16 + t1*1 + v0*8 + v1*64
    //   m = pos % 16, k = pos / 16
    
    printf("Per-thread (row,col) for BF16 MMA atom operand A:\n");
    printf("tid | v=(0,0)   v=(1,0)   v=(0,1)   v=(1,1)\n");
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8;
        int t1 = tid % 8;
        printf("%3d |", tid);
        for (int v1 = 0; v1 < 2; v1++) {
            for (int v0 = 0; v0 < 2; v0++) {
                int pos = t0*32 + t1 + v0*16 + v1*8;
                int m = pos % 16;
                int k = pos / 16;
                printf(" (%2d,%d)", m, k);
            }
        }
        printf("\n");
    }
    
    printf("\nPer-thread (row,col) for TF32 MMA atom operand A:\n");
    printf("tid | v=(0,0)   v=(1,0)   v=(0,1)   v=(1,1)\n");
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8;
        int t1 = tid % 8;
        printf("%3d |", tid);
        for (int v1 = 0; v1 < 2; v1++) {
            for (int v0 = 0; v0 < 2; v0++) {
                int pos = t0*16 + t1 + v0*8 + v1*64;
                int m = pos % 16;
                int k = pos / 16;
                printf(" (%2d,%d)", m, k);
            }
        }
        printf("\n");
    }

    // Now let's analyze the shuffle pattern more carefully
    // For each value position, which thread in BF16 holds the data that TF32 needs?
    printf("\n=== Shuffle pattern for operand A (tid 0..31) ===\n");
    printf("For TF32 thread tid, value vid: needs data from BF16 thread src_tid, value src_vid\n");
    
    // Build BF16 reverse map: pos -> (tid, vid)
    int bf16_rev_tid[128], bf16_rev_vid[128];
    for (int i = 0; i < 128; i++) { bf16_rev_tid[i] = -1; bf16_rev_vid[i] = -1; }
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8, t1 = tid % 8;
        for (int vid = 0; vid < 4; vid++) {
            int v0 = vid % 2, v1 = vid / 2;
            int pos = t0*32 + t1 + v0*16 + v1*8;
            bf16_rev_tid[pos] = tid;
            bf16_rev_vid[pos] = vid;
        }
    }
    
    printf("TF32(tid, vid) needs BF16(src_tid, src_vid), delta = src_tid XOR tid\n");
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8, t1 = tid % 8;
        printf("tid=%2d:", tid);
        for (int vid = 0; vid < 4; vid++) {
            int v0 = vid % 2, v1 = vid / 2;
            int pos = t0*16 + t1 + v0*8 + v1*64;
            int src_tid = bf16_rev_tid[pos];
            int src_vid = bf16_rev_vid[pos];
            printf("  v%d<-BF(%2d,v%d)[xor=%2d]", vid, src_tid, src_vid, tid ^ src_tid);
        }
        printf("\n");
    }

    // Same analysis for operand B
    printf("\n=== Shuffle pattern for operand B (tid 0..31) ===\n");
    // LayoutB_TV: BF16: ((_4,_8),_2):((_16,_1),_8)
    //             TF32: ((_4,_8),_2):((_8,_1),_32)
    // B tile shape per atom: (N=8, K=8) = 64 elements
    // BF16: pos = t0*16 + t1 + v*8
    // TF32: pos = t0*8 + t1 + v*32
    // n = pos % 8, k = pos / 8
    
    printf("Per-thread (n,k) for BF16 MMA atom operand B:\n");
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8, t1 = tid % 8;
        printf("tid=%2d:", tid);
        for (int v = 0; v < 2; v++) {
            int pos = t0*16 + t1 + v*8;
            printf("  v%d->(%d,%d)", v, pos%8, pos/8);
        }
        printf("\n");
    }
    
    printf("\nPer-thread (n,k) for TF32 MMA atom operand B:\n");
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8, t1 = tid % 8;
        printf("tid=%2d:", tid);
        for (int v = 0; v < 2; v++) {
            int pos = t0*8 + t1 + v*32;
            printf("  v%d->(%d,%d)", v, pos%8, pos/8);
        }
        printf("\n");
    }

    // Build BF16 B reverse map
    int bf16B_rev_tid[64], bf16B_rev_vid[64];
    for (int i = 0; i < 64; i++) { bf16B_rev_tid[i] = -1; bf16B_rev_vid[i] = -1; }
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8, t1 = tid % 8;
        for (int v = 0; v < 2; v++) {
            int pos = t0*16 + t1 + v*8;
            bf16B_rev_tid[pos] = tid;
            bf16B_rev_vid[pos] = v;
        }
    }
    
    printf("\nTF32(tid, vid) needs BF16(src_tid, src_vid), delta = src_tid XOR tid\n");
    for (int tid = 0; tid < 32; tid++) {
        int t0 = tid / 8, t1 = tid % 8;
        printf("tid=%2d:", tid);
        for (int v = 0; v < 2; v++) {
            int pos = t0*8 + t1 + v*32;
            int src_tid = bf16B_rev_tid[pos];
            int src_vid = bf16B_rev_vid[pos];
            printf("  v%d<-BF(%2d,v%d)[xor=%2d]", v, src_tid, src_vid, tid ^ src_tid);
        }
        printf("\n");
    }

    return 0;
}
