// Print CuTe TV layouts, fragment shapes, and tiled copy layouts for BF16 vs TF32 MMA.
// This is a host-only program — it uses CuTe's compile-time layout introspection (no GPU needed).
//
// Build:
//   nvcc -std=c++17 -I<cutlass_include_path> analyze_layout.cu -o analyze_layout
//
// Key outputs:
//   - LayoutA_TV / LayoutB_TV / LayoutC_TV for both BF16 and TF32 MMA
//   - Fragment shapes for thread 0
//   - Tiled copy layouts for LDSM (BF16) and AutoVec (TF32)

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

int main() {
    // BF16 MMA 16x8x8 (same shape as TF32)
    using MMA_BF16 = SM80_16x8x8_F32BF16BF16F32_TN;
    using TiledMma_BF16 = decltype(make_tiled_mma(MMA_BF16{}, Layout<Shape<_1, _2, _1>>{}, Shape<_16, _16, _32>{}));
    
    // TF32 MMA 16x8x8
    using MMA_TF32 = SM80_16x8x8_F32TF32TF32F32_TN;
    using TiledMma_TF32 = decltype(make_tiled_mma(MMA_TF32{}, Layout<Shape<_1, _2, _1>>{}, Shape<_16, _16, _32>{}));
    
    auto bf16_mma = TiledMma_BF16{};
    auto tf32_mma = TiledMma_TF32{};
    
    // Print MMA atoms
    printf("=== BF16 MMA Atom ===\n");
    print(MMA_BF16{});
    printf("\n\n");
    
    printf("=== TF32 MMA Atom ===\n");
    print(MMA_TF32{});
    printf("\n\n");
    
    printf("=== BF16 TiledMMA ===\n");
    print(bf16_mma);
    printf("\n\n");
    
    printf("=== TF32 TiledMMA ===\n");
    print(tf32_mma);
    printf("\n\n");
    
    // Check LayoutA_TV and LayoutB_TV
    printf("=== BF16 LayoutA_TV ===\n");
    print(typename TiledMma_BF16::LayoutA_TV{});
    printf("\n\n");
    
    printf("=== TF32 LayoutA_TV ===\n");
    print(typename TiledMma_TF32::LayoutA_TV{});
    printf("\n\n");
    
    printf("=== BF16 LayoutB_TV ===\n");
    print(typename TiledMma_BF16::LayoutB_TV{});
    printf("\n\n");
    
    printf("=== TF32 LayoutB_TV ===\n");
    print(typename TiledMma_TF32::LayoutB_TV{});
    printf("\n\n");
    
    // Check C layout
    printf("=== BF16 LayoutC_TV ===\n");
    print(typename TiledMma_BF16::LayoutC_TV{});
    printf("\n\n");
    
    printf("=== TF32 LayoutC_TV ===\n");
    print(typename TiledMma_TF32::LayoutC_TV{});
    printf("\n\n");

    // Partition fragment for thread 0
    auto smem_a = make_counting_tensor(make_layout(make_shape(_16{}, _32{}), make_stride(_1{}, _16{})));
    auto smem_b = make_counting_tensor(make_layout(make_shape(_16{}, _32{}), make_stride(_1{}, _16{})));
    
    auto bf16_thr0 = bf16_mma.get_thread_slice(0);
    auto tf32_thr0 = tf32_mma.get_thread_slice(0);
    
    printf("=== BF16 thread 0 fragment_A ===\n");
    auto bf16_fragA = bf16_thr0.partition_fragment_A(smem_a);
    print(bf16_fragA.layout());
    printf("\n\n");
    
    printf("=== TF32 thread 0 fragment_A ===\n");
    auto tf32_fragA = tf32_thr0.partition_fragment_A(smem_a);
    print(tf32_fragA.layout());
    printf("\n\n");

    printf("=== BF16 thread 0 fragment_B ===\n");
    auto bf16_fragB = bf16_thr0.partition_fragment_B(smem_b);
    print(bf16_fragB.layout());
    printf("\n\n");
    
    printf("=== TF32 thread 0 fragment_B ===\n");
    auto tf32_fragB = tf32_thr0.partition_fragment_B(smem_b);
    print(tf32_fragB.layout());
    printf("\n\n");
    
    // Size comparison
    printf("BF16 fragA size: %d, TF32 fragA size: %d\n", (int)size(bf16_fragA), (int)size(tf32_fragA));
    printf("BF16 fragB size: %d, TF32 fragB size: %d\n", (int)size(bf16_fragB), (int)size(tf32_fragB));
    
    // Check tiled_copy layouts for BF16
    using CopyAtom_BF16 = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::bfloat16_t>;
    auto bf16_copy_A = make_tiled_copy_A(CopyAtom_BF16{}, bf16_mma);
    auto bf16_copy_B = make_tiled_copy_B(CopyAtom_BF16{}, bf16_mma);
    
    printf("=== BF16 tiled_copy_A ===\n");
    print(bf16_copy_A);
    printf("\n\n");
    
    printf("=== BF16 tiled_copy_B ===\n");
    print(bf16_copy_B);
    printf("\n\n");
    
    // Check tiled_copy layouts for TF32 (AutoVec)
    using CopyAtom_TF32 = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, cutlass::bfloat16_t>;
    auto tf32_copy_A = make_tiled_copy_A(CopyAtom_TF32{}, tf32_mma);
    auto tf32_copy_B = make_tiled_copy_B(CopyAtom_TF32{}, tf32_mma);
    
    printf("=== TF32 tiled_copy_A (AutoVec bf16) ===\n");
    print(tf32_copy_A);
    printf("\n\n");
    
    printf("=== TF32 tiled_copy_B (AutoVec bf16) ===\n");
    print(tf32_copy_B);
    printf("\n\n");
    
    return 0;
}
