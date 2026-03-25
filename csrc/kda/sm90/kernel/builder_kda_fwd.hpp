#pragma once

#include "kda/sm90/collective/mainloop_kda_fwd.hpp"
#include "kda/sm90/kernel/tile_scheduler.hpp"
#include "kda/sm90/kernel/kernel_kda_fwd.hpp"
#include "kda/sm90/kernel/options.hpp"

#include "kda/sm90/utils/type_traits.hpp"

namespace kda::sm90::kernel {

template <
    class Element_,
    class ElementAccumulatorQK_,
    class ElementAccumulatorPV_,
    class TileShape_,  // BlkSeqQO, BlkSeqKV, HeadSize
    class LayoutQ_,
    class LayoutK_,
    class LayoutV_,
    class LayoutO_,
    class DispatchPolicy,
    class Options = DefaultOptions>
struct FlatBuilderKdaFwd;

template <
    class Element,
    class ElementAccumulatorQK,
    class ElementAccumulatorPV,
    class TileShape,  // BlkSeqQO, BlkSeqKV, HeadSize
    class LayoutQ,
    class LayoutK,
    class LayoutV,
    class LayoutO,
    class Options>
struct FlatBuilderKdaFwd<
    Element,
    ElementAccumulatorQK,
    ElementAccumulatorPV,
    TileShape,
    LayoutQ,
    LayoutK,
    LayoutV,
    LayoutO,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    Options> {
  using CollectiveMainloop = kda::sm90::collective::FlatMainloopTmaWarpSpecializedKdaFwd<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, LayoutQ, LayoutK, LayoutV, LayoutO,
      Options>;

  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, false_type, Options>::value;
  static_assert(!kIsPersistent, "not implemented");

  using TileScheduler = kda::sm90::kernel::IndividualTileScheduler;
  // using TileScheduler = std::conditional_t<kIsPersistent, kda::sm90::kernel::PersistentTileScheduler, kda::sm90::kernel::IndividualTileScheduler>;

  using Kernel = kda::sm90::kernel::FlatKernelTmaWarpSpecializedKdaFwd<CollectiveMainloop, TileScheduler, Options>;
};

}  // namespace kda::sm90::kernel
