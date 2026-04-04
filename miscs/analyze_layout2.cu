// Extended layout analysis: TV layouts, fragment shapes, retile_D layouts, and copy atom details.
// Builds on analyze_layout.cu with additional S2R retile and type size checks.
//
// Build:
//   nvcc -std=c++17 -I<cutlass_include_path> analyze_layout2.cu -o analyze_layout2

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
    
    printf("=== BF16 thread 0 fragment_A layout ===\n");
    auto bf16_fragA = bf16_thr0.partition_fragment_A(smem_a);
    print(bf16_fragA.layout());
    printf("\n");
    printf("BF16 fragA size: %d\n\n", (int)size(bf16_fragA));
    
    printf("=== TF32 thread 0 fragment_A layout ===\n");
    auto tf32_fragA = tf32_thr0.partition_fragment_A(smem_a);
    print(tf32_fragA.layout());
    printf("\n");
    printf("TF32 fragA size: %d\n\n", (int)size(tf32_fragA));

    printf("=== BF16 thread 0 fragment_B layout ===\n");
    auto bf16_fragB = bf16_thr0.partition_fragment_B(smem_b);
    print(bf16_fragB.layout());
    printf("\n");
    printf("BF16 fragB size: %d\n\n", (int)size(bf16_fragB));
    
    printf("=== TF32 thread 0 fragment_B layout ===\n");
    auto tf32_fragB = tf32_thr0.partition_fragment_B(smem_b);
    print(tf32_fragB.layout());
    printf("\n");
    printf("TF32 fragB size: %d\n\n", (int)size(tf32_fragB));

    // Check C fragment layout
    printf("=== BF16 thread 0 fragment_C layout ===\n");
    auto smem_c = make_counting_tensor(make_layout(make_shape(_16{}, _16{})));
    auto bf16_fragC = bf16_thr0.partition_fragment_C(smem_c);
    print(bf16_fragC.layout());
    printf("\n");
    printf("BF16 fragC size: %d\n\n", (int)size(bf16_fragC));
    
    printf("=== TF32 thread 0 fragment_C layout ===\n");
    auto tf32_fragC = tf32_thr0.partition_fragment_C(smem_c);
    print(tf32_fragC.layout());
    printf("\n");
    printf("TF32 fragC size: %d\n\n", (int)size(tf32_fragC));

    // Check tiled_copy_A for BF16 using LDSM
    using CopyAtom_LDSM = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::bfloat16_t>;
    auto bf16_copy_A = make_tiled_copy_A(CopyAtom_LDSM{}, bf16_mma);
    printf("=== BF16 tiled_copy_A (LDSM) ===\n");
    print(bf16_copy_A);
    printf("\n\n");
    
    // Check S2R retile for BF16 tiled_copy_A
    auto bf16_thr_copy_A = bf16_copy_A.get_thread_slice(0);
    auto bf16_retile_A = bf16_thr_copy_A.retile_D(bf16_fragA);
    printf("=== BF16 retile_D(fragA) layout ===\n");
    print(bf16_retile_A.layout());
    printf("\n\n");
    
    // Check tiled_copy_A for TF32 using AutoVec
    using CopyAtom_AutoVec = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, cutlass::bfloat16_t>;
    auto tf32_copy_A = make_tiled_copy_A(CopyAtom_AutoVec{}, tf32_mma);
    printf("=== TF32 tiled_copy_A (AutoVec bf16) ===\n");
    print(tf32_copy_A);
    printf("\n\n");
    
    // Check S2R retile for TF32 tiled_copy_A
    auto tf32_thr_copy_A = tf32_copy_A.get_thread_slice(0);
    auto tf32_retile_A = tf32_thr_copy_A.retile_D(tf32_fragA);
    printf("=== TF32 retile_D(fragA) layout ===\n");
    print(tf32_retile_A.layout());
    printf("\n\n");

    // Check tiled_copy_B for BF16 using LDSM
    auto bf16_copy_B = make_tiled_copy_B(CopyAtom_LDSM{}, bf16_mma);
    printf("=== BF16 tiled_copy_B (LDSM) ===\n");
    print(bf16_copy_B);
    printf("\n\n");

    // Check tiled_copy_B for TF32 using AutoVec
    auto tf32_copy_B = make_tiled_copy_B(CopyAtom_AutoVec{}, tf32_mma);
    printf("=== TF32 tiled_copy_B (AutoVec bf16) ===\n");
    print(tf32_copy_B);
    printf("\n\n");

    // Key check: are the ValLayouts the same?
    printf("=== BF16 MMA ValTypeA size: %d bytes ===\n", (int)sizeof(typename TiledMma_BF16::ValTypeA));
    printf("=== TF32 MMA ValTypeA size: %d bytes ===\n", (int)sizeof(typename TiledMma_TF32::ValTypeA));
    printf("=== BF16 MMA ValTypeB size: %d bytes ===\n", (int)sizeof(typename TiledMma_BF16::ValTypeB));
    printf("=== TF32 MMA ValTypeB size: %d bytes ===\n\n", (int)sizeof(typename TiledMma_TF32::ValTypeB));

    // Check the TV (thread,value) mapping directly
    printf("=== BF16 ThrLayoutVMNK ===\n");
    print(typename TiledMma_BF16::ThrLayoutVMNK{});
    printf("\n\n");
    printf("=== TF32 ThrLayoutVMNK ===\n");
    print(typename TiledMma_TF32::ThrLayoutVMNK{});
    printf("\n\n");

    return 0;
}
