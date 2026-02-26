#pragma once

#include <cutlass/tfloat32.h>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

namespace sm100 {
    using tf32 = cutlass::tfloat32_t;
    using bf16 = cutlass::bfloat16_t;
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

}