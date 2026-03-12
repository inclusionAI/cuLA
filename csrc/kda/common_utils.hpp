#pragma once

#include <cutlass/tfloat32.h>
#include <cutlass/bfloat16.h>
#include <cute/tensor.hpp>
#include <cutlass/detail/layout.hpp>

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

namespace sm100 {

using namespace cute;

    using tf32 = cutlass::tfloat32_t;
    using bf16 = cutlass::bfloat16_t;
    using fp16 = cutlass::half_t;

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

template<typename T>
CUTE_DEVICE
void store_128b(void* smem_ptr, const T &data) {
    static_assert(sizeof(T) == 16);
    *(__int128*)smem_ptr = *(__int128*)&data;
}

template <int... Is, typename Layout>
__forceinline__ __host__ __device__ constexpr auto
select_layout(Layout&& l) {
  if constexpr (is_composed_layout<Layout>::value) {
    return make_composed_layout(
        l.layout_a(),
        l.offset(),
        select<Is...>(l.layout_b())
    );
  } else {
    return select<Is...>(l);
  }
}

template <int... Is, typename Tensor>
__forceinline__ __host__ __device__ constexpr auto
select_tensor(Tensor&& t) {
  if constexpr (is_composed_layout<decltype(t.layout())>::value) {
    return make_tensor(
        std::forward<Tensor>(t).data(),
        make_composed_layout(
            std::forward<Tensor>(t).layout().layout_a(),
            std::forward<Tensor>(t).layout().offset(),
            select<Is...>(std::forward<Tensor>(t).layout().layout_b())
        )
    );
  } else {
    return make_tensor(std::forward<Tensor>(t).data(), select<Is...>(t.layout()));
  }
}

template<class Layout>
CUTE_DEVICE constexpr size_t
alignment_for_swizzle(Layout&& layout) {
  return cutlass::detail::alignment_for_swizzle(std::forward<Layout>(layout));
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          class CopyAtom, class TV, class Tiler, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy_pred(TiledCopy<CopyAtom, TV, Tiler> const &tiled_copy, Tensor<Engine0, Layout0> const &S,
                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    // Decay TiledCopy to CopyAtom
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    auto has_with_bool = cute::is_valid([](auto t)->void_t<decltype(declval<typename decltype(t)::Traits>().with(true))>{}, copy_atom);
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool predicate_mn = Is_even_MN || get<0>(identity_MN(_0{}, m, _0{})) < max_MN;
        // NOTE: currently only this predicate is true because we set Clear_OOB_MN=false
        if constexpr (Is_even_MN || !Clear_OOB_MN) {
            if (Is_even_MN || predicate_mn) {
                #pragma unroll
                for (int k = 0; k < size<2>(S); ++k) {
                    if constexpr (Is_even_K || !Clear_OOB_K) {
                        if (Is_even_K || predicate_K(k)) { cute::copy(copy_atom, S(_, m, k), D(_, m, k)); }
                    } else {  // Clear_OOB_K == true && Is_even_K == false
                        // If copy traits can be transformed with a predicate value, do it, otherwise branch here
                        if constexpr (has_with_bool) {
                            cute::copy(copy_atom.with(predicate_K(k)), S(_, m, k), D(_, m, k));
                        } else {
                            if (predicate_K(k)) {
                                cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                            } else {
                                cute::clear(D(_, m, k));
                            }
                        }
                    }
                }
            }
        } else {  // Clear_OOB_MN == true && Is_even_MN == false, also implies Clear_OOB_K == true
            if constexpr (!has_with_bool) {
                if (predicate_mn) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(S); ++k) {
                        if (Is_even_K || predicate_K(k)) {
                            cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                        } else if (Clear_OOB_K) {
                            cute::clear(D(_, m, k));
                        }
                    }
                } else {
                    cute::clear(D(_, m, _));
                }
            } else {  // combine the mn predicate with the k predicate
                #pragma unroll
                for (int k = 0; k < size<2>(S); ++k) {
                    cute::copy(copy_atom.with(predicate_mn && (Is_even_K || predicate_K(k))), S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

} // namespace sm100