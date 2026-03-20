#pragma once

#include "kda_config.hpp"

namespace flashla {

// KDA forward kernels

// KDA forward intra-chunk kernel
void run_kda_fwd_intra_sm100(KDA_fwd_intra_params &config, cudaStream_t stream);

} // namespace flashla