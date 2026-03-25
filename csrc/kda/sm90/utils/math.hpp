#pragma once

#include "cute/config.hpp"

namespace kda::sm90 {

namespace detail {

template <typename T>
CUTE_HOST_DEVICE constexpr T
ceil_log2(T n) {
  return n <= 1 ? 0 : 1 + ceil_log2((n + 1) / 2);
}

}  // namespace detail

template <typename T>
CUTE_HOST_DEVICE constexpr T
next_power_of_two(T n) {
  return static_cast<T>(1) << detail::ceil_log2(n);
}

}  // namespace kda::sm90
