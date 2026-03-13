// Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/detail/layout.hpp>

namespace flashla {

using namespace cute;

// tcgen05.fence::before_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

// tcgen05.fence::after_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// Perform SS UTCMMA
// sA and sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ss(
    TiledMMA &tiled_mma,
    TensorA sA,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            sA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// Perform TS UTCMMA
// sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tA_frag and tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ts(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// Load from tensor memory, 32 data path lanes, 32-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumElements>
__device__ __forceinline__
void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");
    // NOTE The following code crashes VSCode intellisense engine, so we disable it
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}

// Store into tensor memory, 32 data path lanes, 32-bit pattern, repeated N times. (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
template <int kNumElements>
__device__ __forceinline__
void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
    uint32_t const* data = (uint32_t const*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 || kNumElements == 32 || kNumElements == 64 || kNumElements == 128, "Invalid kNumElements");
#ifndef __VSCODE_IDE__
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1) {
            cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 2) {
            cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 4) {
            cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 8) {
            cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 16) {
            cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 32) {
            cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 64) {
            cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., tmem_start);
        } else if constexpr (kNumElements == 128) {
            cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., tmem_start);
        }
    }(cute::make_index_sequence<kNumElements>{});
#endif
}



}