#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/common.h"

namespace kerutils {

// ============================================================
// Vectorized float2 arithmetic
// ============================================================

// Vectorized addition for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)
CUTE_DEVICE
float2 float2_add(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "add.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b))
    );
    return c;
}

// Vectorized subtraction for float32
CUTE_DEVICE
float2 float2_sub(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "sub.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b))
    );
    return c;
}

// Vectorized multiplication for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul)
CUTE_DEVICE
float2 float2_mul(const float2 &a, const float2 &b) {
    float2 c;
    asm volatile(
        "mul.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

// Vectorized fused multiply-add for float32 (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)
CUTE_DEVICE
float2 float2_fma(const float2 &a, const float2 &b, const float2 &c) {
    // return a*b+c
    float2 d;
    asm volatile(
        "fma.rn.f32x2 %0, %1, %2, %3;\n"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)),
          "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

// Vectorized negation for float32
CUTE_DEVICE
float2 float2_neg(const float2 &a) {
    float2 t = {-1.0f, -1.0f};
    return float2_mul(a, t);
}

// ============================================================
// tcgen05 fence intrinsics (SM100)
// ============================================================

// tcgen05.fence::before_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

// tcgen05.fence::after_thread_sync (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// ============================================================
// Tensor memory (TMEM) load/store intrinsics (SM100)
// ============================================================

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

}  // namespace kerutils
