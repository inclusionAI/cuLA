#pragma once

#include "cute/config.hpp"

namespace kda::sm90 {

struct Unused {
  using Params = Unused;
  using SharedStorage = char;
  static constexpr uint32_t Stages = 0;

  template <typename... Ts>
  CUTE_HOST_DEVICE
  Unused(Ts... vs) {}

  template <typename T>
  CUTE_HOST_DEVICE
  Unused operator=(T&& v) { return Unused{}; }
};

}
