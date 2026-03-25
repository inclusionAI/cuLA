#pragma once

#include <cuda_bf16.h>

#include <cutlass/tfloat32.h>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cute/config.hpp>

namespace kerutils {

using nvbf16 = __nv_bfloat16;
using nvbf16x2 = __nv_bfloat162;

using tf32 = cutlass::tfloat32_t;
using bf16 = cutlass::bfloat16_t;
using fp16 = cutlass::half_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

struct bf16x4 {
    bf16 a, b, c, d;
};

struct nvbf16x4 {
    __nv_bfloat162 a, b;
};

struct tf32x4 {
    tf32 a, b, c, d;
};

struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

struct bf16x16 {
    __nv_bfloat162 a0;
    __nv_bfloat162 a1;
    __nv_bfloat162 a2;
    __nv_bfloat162 a3;
    __nv_bfloat162 a4;
    __nv_bfloat162 a5;
    __nv_bfloat162 a6;
    __nv_bfloat162 a7;
};

}  // namespace kerutils

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define KERUTILS_ENABLE_SM80
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
static_assert(false, "kerutils doesn't support SM architectures below SM80");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#define KERUTILS_ENABLE_SM90
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000))
#define KERUTILS_ENABLE_SM90A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
#define KERUTILS_ENABLE_SM100
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
#define KERUTILS_ENABLE_SM80
#define KERUTILS_ENABLE_SM90
#define KERUTILS_ENABLE_SM90A
#define KERUTILS_ENABLE_SM100
#define KERUTILS_ENABLE_SM100A
#endif
