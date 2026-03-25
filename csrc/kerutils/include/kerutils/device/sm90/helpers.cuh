#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/common.h"

namespace kerutils {

// TMA copy helper with optional multicast support
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
    uint64_t &bar,
    const cute::TMA::CacheHintSm90 &cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL,
    const uint16_t &multicast_mask = 0
) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(bar, multicast_mask, cache_hint),
        thr_tma.partition_S(src),
        thr_tma.partition_D(dst)
    );
}

}  // namespace kerutils
