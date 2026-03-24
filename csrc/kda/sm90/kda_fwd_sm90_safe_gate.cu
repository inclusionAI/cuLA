#include "cute/numeric/numeric_types.hpp"
#include "cutlass/arch/arch.h"

#include "kda/sm90/utils/common.hpp"
#include "kda/sm90/prefill_kernel_kda_fwd_sm90.cuh"

namespace flat {

using namespace cute;
using bf16 = cute::bfloat16_t;

// SafeGate=true, InitState=false
template void launch_kda_fwd_prefill_kernel_gbai<
    false, true, true, false, true,
    cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, float const*, int64_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

// SafeGate=true, InitState=true
template void launch_kda_fwd_prefill_kernel_gbai<
    false, true, true, true, true,
    cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, float const*, int64_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, float, int32_t);

}  // namespace flat
