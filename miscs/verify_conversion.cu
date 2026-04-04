// GPU verification: print the actual TV→(row,k) mapping for all 32 threads in a warp.
// This program runs on device (SM80+) and uses CuTe's layout evaluation to confirm
// the layout formulas derived in the host analysis programs.
//
// Build:
//   nvcc -std=c++17 -arch=sm_80 -I<cutlass_include_path> verify_conversion.cu -o verify_conversion

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cstdio>

using namespace cute;

__global__ void test_conversion() {
    int tid = threadIdx.x;
    if (tid >= 32) return;  // single warp

    // BF16 MMA: SM80_16x8x8_F32BF16BF16F32_TN
    using MMA_BF16 = SM80_16x8x8_F32BF16BF16F32_TN;
    using TiledMma_BF16 = decltype(make_tiled_mma(MMA_BF16{}, Layout<Shape<_1, _1, _1>>{}, Shape<_16, _16, _8>{}));
    // TF32 MMA: SM80_16x8x8_F32TF32TF32F32_TN
    using MMA_TF32 = SM80_16x8x8_F32TF32TF32F32_TN;
    using TiledMma_TF32 = decltype(make_tiled_mma(MMA_TF32{}, Layout<Shape<_1, _1, _1>>{}, Shape<_16, _16, _8>{}));

    auto tiled_mma_bf16 = TiledMma_BF16{};
    auto tiled_mma_tf32 = TiledMma_TF32{};

    // Print TV layouts
    if (tid == 0) {
        printf("=== BF16 MMA Atom ===\n");
        printf("LayoutA_TV: "); print(typename MMA_Atom<MMA_BF16>::LayoutA_TV{}); printf("\n");
        printf("LayoutB_TV: "); print(typename MMA_Atom<MMA_BF16>::LayoutB_TV{}); printf("\n");
        printf("LayoutC_TV: "); print(typename MMA_Atom<MMA_BF16>::LayoutC_TV{}); printf("\n");

        printf("\n=== TF32 MMA Atom ===\n");
        printf("LayoutA_TV: "); print(typename MMA_Atom<MMA_TF32>::LayoutA_TV{}); printf("\n");
        printf("LayoutB_TV: "); print(typename MMA_Atom<MMA_TF32>::LayoutB_TV{}); printf("\n");
        printf("LayoutC_TV: "); print(typename MMA_Atom<MMA_TF32>::LayoutC_TV{}); printf("\n");
    }
    __syncwarp();

    // Print the (row, k) values each thread holds for operand A
    // BF16 LayoutA_TV: ((_4,_8), (_2,_2)) : ((_32,_1), (_16,_8))
    // TF32 LayoutA_TV: ((_4,_8), (_2,_2)) : ((_16,_1), (_8,_64))
    // For M16xK8 atom (col-major): pos = m + k*16, so m = pos % 16, k = pos / 16

    if (tid == 0) {
        printf("\n=== BF16 A: (thread, val) → (row, k) ===\n");
        auto layoutA_bf16 = typename MMA_Atom<MMA_BF16>::LayoutA_TV{};
        for (int t = 0; t < 32; t++) {
            printf("thread %2d: ", t);
            for (int v = 0; v < 4; v++) {
                int pos = layoutA_bf16(t, v);
                int row = pos / 8;  // 16x8 matrix
                int k = pos % 8;
                printf("v%d→(r%d,k%d) ", v, row, k);
            }
            printf("\n");
        }

        printf("\n=== TF32 A: (thread, val) → (row, k) ===\n");
        auto layoutA_tf32 = typename MMA_Atom<MMA_TF32>::LayoutA_TV{};
        for (int t = 0; t < 32; t++) {
            printf("thread %2d: ", t);
            for (int v = 0; v < 4; v++) {
                int pos = layoutA_tf32(t, v);
                int row = pos / 8;
                int k = pos % 8;
                printf("v%d→(r%d,k%d) ", v, row, k);
            }
            printf("\n");
        }

        printf("\n=== BF16 B: (thread, val) → position ===\n");
        auto layoutB_bf16 = typename MMA_Atom<MMA_BF16>::LayoutB_TV{};
        for (int t = 0; t < 32; t++) {
            printf("thread %2d: ", t);
            for (int v = 0; v < 2; v++) {
                int pos = layoutB_bf16(t, v);
                printf("v%d→pos%d ", v, pos);
            }
            printf("\n");
        }

        printf("\n=== TF32 B: (thread, val) → position ===\n");
        auto layoutB_tf32 = typename MMA_Atom<MMA_TF32>::LayoutB_TV{};
        for (int t = 0; t < 32; t++) {
            printf("thread %2d: ", t);
            for (int v = 0; v < 2; v++) {
                int pos = layoutB_tf32(t, v);
                printf("v%d→pos%d ", v, pos);
            }
            printf("\n");
        }
    }
}

int main() {
    test_conversion<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
