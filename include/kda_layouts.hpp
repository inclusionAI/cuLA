// describe the "shape" and "stride".
#pragma once

#include <cstddef>

#ifndef KDA_HOST_DEVICE
#define KDA_HOST_DEVICE __forceinline__ __host__ __device__
#endif

namespace kda {
namespace layouts {

template <int Dim>
struct FeatureLayout {
  KDA_HOST_DEVICE static std::size_t offset(int b, int h, int t, int d, int B, int H, int T) {
    (void)B;
    return (((static_cast<std::size_t>(b) * H + h) * T + t) * Dim + d);
  }
};

struct GateLayout {
  KDA_HOST_DEVICE static std::size_t offset(int b, int h, int t, int B, int H, int T) {
    (void)B;
    return ((static_cast<std::size_t>(b) * H + h) * T + t);
  }
};

template <int C>
struct ChunkMatrixLayout {
  KDA_HOST_DEVICE static std::size_t offset(
      int b, int h, int cidx, int row, int col, int B, int H, int num_chunks) {
    (void)B;
    return ((((static_cast<std::size_t>(b) * H + h) * num_chunks + cidx) * C + row) * C + col);
  }
};

}  // namespace layouts
}  // namespace kda