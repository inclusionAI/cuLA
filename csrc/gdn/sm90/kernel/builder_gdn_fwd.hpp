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

#include "gdn/sm90/collective/mainloop_gdn_fwd.hpp"
#include "gdn/sm90/kernel/kernel_gdn_fwd.hpp"
#include "gdn/sm90/kernel/options.hpp"
#include "gdn/sm90/kernel/tile_scheduler.hpp"
#include "gdn/sm90/utils/type_traits.hpp"

namespace gdn::sm90::kernel {

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
struct FlatBuilderGdnFwd;

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
struct FlatBuilderGdnFwd<
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
    using CollectiveMainloop = gdn::sm90::collective::FlatMainloopTmaWarpSpecializedGdnFwd<
        Element,
        ElementAccumulatorQK,
        ElementAccumulatorPV,
        TileShape,
        LayoutQ,
        LayoutK,
        LayoutV,
        LayoutO,
        Options>;

    static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, false_type, Options>::value;
    static_assert(!kIsPersistent, "not implemented");

    using TileScheduler = gdn::sm90::kernel::IndividualTileScheduler;
    // using TileScheduler = std::conditional_t<kIsPersistent, gdn::sm90::kernel::PersistentTileScheduler,
    // gdn::sm90::kernel::IndividualTileScheduler>;

    using Kernel = gdn::sm90::kernel::FlatKernelTmaWarpSpecializedGdnFwd<CollectiveMainloop, TileScheduler, Options>;
};

}  // namespace gdn::sm90::kernel
