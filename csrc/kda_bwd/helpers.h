#pragma once

#include <cute/tensor.hpp>
#include "basic.h"


namespace sm100 {

using namespace cute;

    using Layout_K_SW128_32B_Atom_Bits = ComposedLayout<Swizzle<2,5,2>, smem_ptr_flag, Layout<Shape< _4, _1024>,Stride<_1024, _1>>>;

    template <class Type>
    using Layout_K_SW128_32B_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_32B_Atom_Bits{}));


CUTE_DEVICE
void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

CUTE_DEVICE
void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

CUTE_DEVICE
void umma_arrive_noelect(uint64_t &bar_ptr) {
  uint64_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar_ptr);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
    :
    :"l"(bar_intptr));
}

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

CUTE_DEVICE
__nv_bfloat162 bf16x2_add(const __nv_bfloat162 &a, const __nv_bfloat162 &b) {
    __nv_bfloat162 c;
    asm volatile(
        "add.bf16x2 %0, %1, %2;\n"
        : "=r"(reinterpret_cast<uint32_t&>(c))
        : "r"(reinterpret_cast<uint32_t const&>(a)),
          "r"(reinterpret_cast<uint32_t const&>(b))
    );
    return c;
}


CUTE_DEVICE 
void _store_256B(
    uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
    uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
    void *gmem_addr
) {
    asm volatile("st.global.L1::no_allocate.v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                :: "l"(gmem_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3), "r"(src4), "r"(src5), "r"(src6), "r"(src7));
}

CUTE_DEVICE
void store_256B(void *src, void *dst) {
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src);
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst);
    _store_256B(
        src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], src_ptr[6], src_ptr[7], dst);
}

template<
    typename TMA,
    typename Tensor0,
    typename Tensor1
>
CUTE_DEVICE
void launch_tma_copy(
    const TMA &tma_copy,
    Tensor0 src,
    Tensor1 dst,
    uint64_t &bar,  // 改为引用，TMA 需要 barrier 的 smem 地址
    const cute::TMA::CacheHintSm90 &cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL
) {
    auto thr_tma = tma_copy.get_slice(_0{});
    cute::copy(
        tma_copy.with(bar, 0, cache_hint),
        thr_tma.partition_S(src),
        thr_tma.partition_D(dst)
    );
}

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

#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)

// 32 data path lanes, 32-bit pattern, repeated N times
template <int N, typename T>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t const &src_addr, T* dst_ptr_) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128, "N must be a power of 2 and lies between 1 ~ 128");
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst_ptr_);

    if constexpr (N == 1) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    "{%0},"
                    "[%1];\n"
                    : "=r"(dst_ptr[0])
                    : "r"(src_addr));
    } else if constexpr (N == 2) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32"
                    "{%0, %1},"
                    "[%2];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1])
                    : "r"(src_addr));
    } else if constexpr (N == 4) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3])
                    : "r"(src_addr));
    } else if constexpr (N == 8) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
                    "{%0, %1, %2, %3, %4, %5, %6, %7},"
                    "[%8];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                    "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
                    : "r"(src_addr));
    } else if constexpr (N == 16) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32"
                    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15},"
                    "[%16];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                    "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
                    "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
                    "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
                    "=r"(dst_ptr[15])
                    : "r"(src_addr));
    } else if constexpr (N == 32) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
                    "%26, %27, %28, %29, %30, %31},"
                    "[%32];\n"
                    : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
                    "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
                    "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
                    "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
                    "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
                    "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
                    "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
                    "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
                    "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
                    "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
                    "=r"(dst_ptr[30]), "=r"(dst_ptr[31])
                    : "r"(src_addr));
    } else if constexpr (N == 64) {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x64.b32"
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63},"
            "[%64];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63])
            : "r"(src_addr));
    } else if constexpr (N == 128) {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x128.b32"
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, "
            "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, "
            "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, "
            "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
            "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
            "%121, %122, %123, %124, %125, %126, %127},"
            "[%128];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
            "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
            "=r"(dst_ptr[6]), "=r"(dst_ptr[7]), "=r"(dst_ptr[8]),
            "=r"(dst_ptr[9]), "=r"(dst_ptr[10]), "=r"(dst_ptr[11]),
            "=r"(dst_ptr[12]), "=r"(dst_ptr[13]), "=r"(dst_ptr[14]),
            "=r"(dst_ptr[15]), "=r"(dst_ptr[16]), "=r"(dst_ptr[17]),
            "=r"(dst_ptr[18]), "=r"(dst_ptr[19]), "=r"(dst_ptr[20]),
            "=r"(dst_ptr[21]), "=r"(dst_ptr[22]), "=r"(dst_ptr[23]),
            "=r"(dst_ptr[24]), "=r"(dst_ptr[25]), "=r"(dst_ptr[26]),
            "=r"(dst_ptr[27]), "=r"(dst_ptr[28]), "=r"(dst_ptr[29]),
            "=r"(dst_ptr[30]), "=r"(dst_ptr[31]), "=r"(dst_ptr[32]),
            "=r"(dst_ptr[33]), "=r"(dst_ptr[34]), "=r"(dst_ptr[35]),
            "=r"(dst_ptr[36]), "=r"(dst_ptr[37]), "=r"(dst_ptr[38]),
            "=r"(dst_ptr[39]), "=r"(dst_ptr[40]), "=r"(dst_ptr[41]),
            "=r"(dst_ptr[42]), "=r"(dst_ptr[43]), "=r"(dst_ptr[44]),
            "=r"(dst_ptr[45]), "=r"(dst_ptr[46]), "=r"(dst_ptr[47]),
            "=r"(dst_ptr[48]), "=r"(dst_ptr[49]), "=r"(dst_ptr[50]),
            "=r"(dst_ptr[51]), "=r"(dst_ptr[52]), "=r"(dst_ptr[53]),
            "=r"(dst_ptr[54]), "=r"(dst_ptr[55]), "=r"(dst_ptr[56]),
            "=r"(dst_ptr[57]), "=r"(dst_ptr[58]), "=r"(dst_ptr[59]),
            "=r"(dst_ptr[60]), "=r"(dst_ptr[61]), "=r"(dst_ptr[62]),
            "=r"(dst_ptr[63]), "=r"(dst_ptr[64]), "=r"(dst_ptr[65]),
            "=r"(dst_ptr[66]), "=r"(dst_ptr[67]), "=r"(dst_ptr[68]),
            "=r"(dst_ptr[69]), "=r"(dst_ptr[70]), "=r"(dst_ptr[71]),
            "=r"(dst_ptr[72]), "=r"(dst_ptr[73]), "=r"(dst_ptr[74]),
            "=r"(dst_ptr[75]), "=r"(dst_ptr[76]), "=r"(dst_ptr[77]),
            "=r"(dst_ptr[78]), "=r"(dst_ptr[79]), "=r"(dst_ptr[80]),
            "=r"(dst_ptr[81]), "=r"(dst_ptr[82]), "=r"(dst_ptr[83]),
            "=r"(dst_ptr[84]), "=r"(dst_ptr[85]), "=r"(dst_ptr[86]),
            "=r"(dst_ptr[87]), "=r"(dst_ptr[88]), "=r"(dst_ptr[89]),
            "=r"(dst_ptr[90]), "=r"(dst_ptr[91]), "=r"(dst_ptr[92]),
            "=r"(dst_ptr[93]), "=r"(dst_ptr[94]), "=r"(dst_ptr[95]),
            "=r"(dst_ptr[96]), "=r"(dst_ptr[97]), "=r"(dst_ptr[98]),
            "=r"(dst_ptr[99]), "=r"(dst_ptr[100]), "=r"(dst_ptr[101]),
            "=r"(dst_ptr[102]), "=r"(dst_ptr[103]), "=r"(dst_ptr[104]),
            "=r"(dst_ptr[105]), "=r"(dst_ptr[106]), "=r"(dst_ptr[107]),
            "=r"(dst_ptr[108]), "=r"(dst_ptr[109]), "=r"(dst_ptr[110]),
            "=r"(dst_ptr[111]), "=r"(dst_ptr[112]), "=r"(dst_ptr[113]),
            "=r"(dst_ptr[114]), "=r"(dst_ptr[115]), "=r"(dst_ptr[116]),
            "=r"(dst_ptr[117]), "=r"(dst_ptr[118]), "=r"(dst_ptr[119]),
            "=r"(dst_ptr[120]), "=r"(dst_ptr[121]), "=r"(dst_ptr[122]),
            "=r"(dst_ptr[123]), "=r"(dst_ptr[124]), "=r"(dst_ptr[125]),
            "=r"(dst_ptr[126]), "=r"(dst_ptr[127])
            : "r"(src_addr));
    } else {
        asm volatile ("trap");
    }
}

// 32 data path lanes, 32-bit pattern, repeated N times
template <int N, typename T>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t const &dst_addr, T* src_ptr_) {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= 128, "N must be a power of 2 and lies between 1 ~ 128");
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src_ptr_);

    if constexpr (N == 1) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                    "[%1], {%0};\n"
                    :
                    : "r"(src_ptr[0]),
                      "r"(dst_addr));
    } else if constexpr (N == 2) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32"
                    "[%2], {%0, %1};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]),
                      "r"(dst_addr));
    } else if constexpr (N == 4) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32"
                    "[%4], {%0, %1, %2, %3};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]),
                      "r"(dst_addr));
    } else if constexpr (N == 8) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32"
                    "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
                    "r"(src_ptr[6]), "r"(src_ptr[7]),
                      "r"(dst_addr));
    } else if constexpr (N == 16) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32"
                    "[%16], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
                    "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
                    "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
                    "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
                    "r"(src_ptr[15]),
                      "r"(dst_addr));
    } else if constexpr (N == 32) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32"
                    "[%32], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
                    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
                    "%26, %27, %28, %29, %30, %31};\n"
                    :
                    : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
                    "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
                    "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
                    "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
                    "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
                    "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]),
                    "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]),
                    "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]),
                    "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]),
                    "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]),
                    "r"(src_ptr[30]), "r"(src_ptr[31]),
                      "r"(dst_addr));
    } else if constexpr (N == 64) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x64.b32"
            "[%64], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
            "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
            "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
            "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
            "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
            "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]),
            "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]),
            "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]),
            "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]),
            "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]),
            "r"(src_ptr[30]), "r"(src_ptr[31]), "r"(src_ptr[32]),
            "r"(src_ptr[33]), "r"(src_ptr[34]), "r"(src_ptr[35]),
            "r"(src_ptr[36]), "r"(src_ptr[37]), "r"(src_ptr[38]),
            "r"(src_ptr[39]), "r"(src_ptr[40]), "r"(src_ptr[41]),
            "r"(src_ptr[42]), "r"(src_ptr[43]), "r"(src_ptr[44]),
            "r"(src_ptr[45]), "r"(src_ptr[46]), "r"(src_ptr[47]),
            "r"(src_ptr[48]), "r"(src_ptr[49]), "r"(src_ptr[50]),
            "r"(src_ptr[51]), "r"(src_ptr[52]), "r"(src_ptr[53]),
            "r"(src_ptr[54]), "r"(src_ptr[55]), "r"(src_ptr[56]),
            "r"(src_ptr[57]), "r"(src_ptr[58]), "r"(src_ptr[59]),
            "r"(src_ptr[60]), "r"(src_ptr[61]), "r"(src_ptr[62]),
            "r"(src_ptr[63]),
              "r"(dst_addr));
    } else if constexpr (N == 128) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x128.b32"
            "[%128], {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, "
            "%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, "
            "%43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
            "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, "
            "%71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, "
            "%85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, "
            "%99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
            "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
            "%121, %122, %123, %124, %125, %126, %127};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
            "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
            "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(src_ptr[8]),
            "r"(src_ptr[9]), "r"(src_ptr[10]), "r"(src_ptr[11]),
            "r"(src_ptr[12]), "r"(src_ptr[13]), "r"(src_ptr[14]),
            "r"(src_ptr[15]), "r"(src_ptr[16]), "r"(src_ptr[17]),
            "r"(src_ptr[18]), "r"(src_ptr[19]), "r"(src_ptr[20]),
            "r"(src_ptr[21]), "r"(src_ptr[22]), "r"(src_ptr[23]),
            "r"(src_ptr[24]), "r"(src_ptr[25]), "r"(src_ptr[26]),
            "r"(src_ptr[27]), "r"(src_ptr[28]), "r"(src_ptr[29]),
            "r"(src_ptr[30]), "r"(src_ptr[31]), "r"(src_ptr[32]),
            "r"(src_ptr[33]), "r"(src_ptr[34]), "r"(src_ptr[35]),
            "r"(src_ptr[36]), "r"(src_ptr[37]), "r"(src_ptr[38]),
            "r"(src_ptr[39]), "r"(src_ptr[40]), "r"(src_ptr[41]),
            "r"(src_ptr[42]), "r"(src_ptr[43]), "r"(src_ptr[44]),
            "r"(src_ptr[45]), "r"(src_ptr[46]), "r"(src_ptr[47]),
            "r"(src_ptr[48]), "r"(src_ptr[49]), "r"(src_ptr[50]),
            "r"(src_ptr[51]), "r"(src_ptr[52]), "r"(src_ptr[53]),
            "r"(src_ptr[54]), "r"(src_ptr[55]), "r"(src_ptr[56]),
            "r"(src_ptr[57]), "r"(src_ptr[58]), "r"(src_ptr[59]),
            "r"(src_ptr[60]), "r"(src_ptr[61]), "r"(src_ptr[62]),
            "r"(src_ptr[63]), "r"(src_ptr[64]), "r"(src_ptr[65]),
            "r"(src_ptr[66]), "r"(src_ptr[67]), "r"(src_ptr[68]),
            "r"(src_ptr[69]), "r"(src_ptr[70]), "r"(src_ptr[71]),
            "r"(src_ptr[72]), "r"(src_ptr[73]), "r"(src_ptr[74]),
            "r"(src_ptr[75]), "r"(src_ptr[76]), "r"(src_ptr[77]),
            "r"(src_ptr[78]), "r"(src_ptr[79]), "r"(src_ptr[80]),
            "r"(src_ptr[81]), "r"(src_ptr[82]), "r"(src_ptr[83]),
            "r"(src_ptr[84]), "r"(src_ptr[85]), "r"(src_ptr[86]),
            "r"(src_ptr[87]), "r"(src_ptr[88]), "r"(src_ptr[89]),
            "r"(src_ptr[90]), "r"(src_ptr[91]), "r"(src_ptr[92]),
            "r"(src_ptr[93]), "r"(src_ptr[94]), "r"(src_ptr[95]),
            "r"(src_ptr[96]), "r"(src_ptr[97]), "r"(src_ptr[98]),
            "r"(src_ptr[99]), "r"(src_ptr[100]), "r"(src_ptr[101]),
            "r"(src_ptr[102]), "r"(src_ptr[103]), "r"(src_ptr[104]),
            "r"(src_ptr[105]), "r"(src_ptr[106]), "r"(src_ptr[107]),
            "r"(src_ptr[108]), "r"(src_ptr[109]), "r"(src_ptr[110]),
            "r"(src_ptr[111]), "r"(src_ptr[112]), "r"(src_ptr[113]),
            "r"(src_ptr[114]), "r"(src_ptr[115]), "r"(src_ptr[116]),
            "r"(src_ptr[117]), "r"(src_ptr[118]), "r"(src_ptr[119]),
            "r"(src_ptr[120]), "r"(src_ptr[121]), "r"(src_ptr[122]),
            "r"(src_ptr[123]), "r"(src_ptr[124]), "r"(src_ptr[125]),
            "r"(src_ptr[126]), "r"(src_ptr[127]),
              "r"(dst_addr));
    } else {
        asm volatile ("trap");
    }
}

template<typename T>
CUTE_DEVICE
void store_128b(void* smem_ptr, const T &data) {
    static_assert(sizeof(T) == 16);
    *(__int128*)smem_ptr = *(__int128*)&data;
}

}
