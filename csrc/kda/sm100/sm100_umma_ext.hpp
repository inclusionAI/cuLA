// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cute/arch/config.hpp>
#include <cute/tensor.hpp>

namespace cute {

template <
    class a_type,
    class b_type,
    class c_type,
    int M,
    int N,
    UMMA::Major a_major,
    UMMA::Major b_major,
    UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
    UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
    UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_TF32_TS_MASK02 {
    static_assert(
        M == 64 || M == 128, "SM100_MMA_TF32_TS_MASK02 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
    static_assert(
        (M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) || (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
        "SM100_MMA_TF32_TS_MASK02 N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");
    static_assert(a_major == UMMA::Major::K, "SM100_MMA_TF32_TS_MASK02 A from TMEM can't be transposed");

    using DRegisters = void;
    using ARegisters = uint32_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void
    fma(uint32_t const& tmem_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scaleC,
        uint64_t const& idescE) {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
        uint32_t mask[4] = {0, 0xFFFFFFFF, 0, 0xFFFFFFFF};
        if (cute::elect_one_sync()) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
                "}\n"
                :
                : "r"(tmem_c),
                  "r"(tmem_a),
                  "l"(desc_b),
                  "r"(uint32_t(idescE >> 32)),
                  "r"(scaleC),
                  "r"(mask[0]),
                  "r"(mask[1]),
                  "r"(mask[2]),
                  "r"(mask[3]));
        }
#else
        CUTE_INVALID_CONTROL_PATH(
            "Attempting to use SM100_MMA_TF32_TS_MASK02 without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
    }
};

template <
    class a_type,
    class b_type,
    class c_type,
    int M,
    int N,
    UMMA::Major a_major,
    UMMA::Major b_major,
    UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
    UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
    UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_TF32_TS_MASK13 {
    static_assert(
        M == 64 || M == 128, "SM100_MMA_TF32_TS_MASK13 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
    static_assert(
        (M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) || (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
        "SM100_MMA_TF32_TS_MASK13 N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");
    static_assert(a_major == UMMA::Major::K, "SM100_MMA_TF32_TS_MASK13 A from TMEM can't be transposed");

    using DRegisters = void;
    using ARegisters = uint32_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void
    fma(uint32_t const& tmem_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scaleC,
        uint64_t const& idescE) {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
        uint32_t mask[4] = {0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
        if (cute::elect_one_sync()) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
                "}\n"
                :
                : "r"(tmem_c),
                  "r"(tmem_a),
                  "l"(desc_b),
                  "r"(uint32_t(idescE >> 32)),
                  "r"(scaleC),
                  "r"(mask[0]),
                  "r"(mask[1]),
                  "r"(mask[2]),
                  "r"(mask[3]));
        }
#else
        CUTE_INVALID_CONTROL_PATH(
            "Attempting to use SM100_MMA_TF32_TS_MASK13 without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
    }
};

template <
    class a_type,
    class b_type,
    class c_type,
    int M,
    int N,
    UMMA::Major a_major,
    UMMA::Major b_major,
    UMMA::ScaleIn a_neg,
    UMMA::ScaleIn b_neg,
    UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_TF32_TS_MASK02<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>> {
    using ValTypeD = c_type;
    using ValTypeA = a_type;
    using ValTypeB = b_type;
    using ValTypeC = c_type;
    static_assert(
        cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32,
        "SM100_MMA_TF32_TS_MASK02 supports 32bit types");

    using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

    // Logical shape-K is always 256bits, transform to units of elements
    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
    using ThrID = Layout<_1>;
    using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
    using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
    using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

    // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

    template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
    CUTE_HOST_DEVICE constexpr friend void
    mma_unpack(
        MMA_Traits const& traits,
        Tensor<TD, DLayout>& D,
        Tensor<TA, ALayout> const& A,
        Tensor<TB, BLayout> const& B,
        Tensor<TC, CLayout> const& C) {
        static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
        static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

        uint64_t desc_a = raw_pointer_cast(A.data());
        uint64_t desc_b = B[0];
        uint32_t tmem_c = raw_pointer_cast(D.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

        SM100_MMA_TF32_TS_MASK02<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>::fma(
            desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
    }
};

template <
    class a_type,
    class b_type,
    class c_type,
    int M,
    int N,
    UMMA::Major a_major,
    UMMA::Major b_major,
    UMMA::ScaleIn a_neg,
    UMMA::ScaleIn b_neg,
    UMMA::Saturate c_sat>
struct MMA_Traits<SM100_MMA_TF32_TS_MASK13<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>> {
    using ValTypeD = c_type;
    using ValTypeA = a_type;
    using ValTypeB = b_type;
    using ValTypeC = c_type;
    static_assert(
        cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> && cute::sizeof_bits_v<b_type> == 32,
        "SM100_MMA_TF32_TS_MASK13 supports 32bit types");

    using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

    // Logical shape-K is always 256bits, transform to units of elements
    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
    using ThrID = Layout<_1>;
    using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
    using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
    using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

    // Accumulate or overwrite C.   1: read C, 0: ignore C [clear accumulators]
    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>();

    template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
    CUTE_HOST_DEVICE constexpr friend void
    mma_unpack(
        MMA_Traits const& traits,
        Tensor<TD, DLayout>& D,
        Tensor<TA, ALayout> const& A,
        Tensor<TB, BLayout> const& B,
        Tensor<TC, CLayout> const& C) {
        static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
        static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

        uint64_t desc_a = raw_pointer_cast(A.data());
        uint64_t desc_b = B[0];
        uint32_t tmem_c = raw_pointer_cast(D.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);

        SM100_MMA_TF32_TS_MASK13<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>::fma(
            desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
    }
};

}  // namespace cute