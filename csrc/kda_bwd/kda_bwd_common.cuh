#pragma once

#include "kda_config.h"

namespace sm100 {

// Forward declarations for KDA backward kernels

// KDA backward intra-chunk kernel
void run_kda_bwd_intra_sm100(KDA_bwd_intra_params &config, cudaStream_t stream);

// KDA backward WY dqkg fused kernel
void run_kda_bwd_wy_dqkg_fused_sm100(KDA_bwd_wy_dqkg_fused_params &config, cudaStream_t stream);

} // namespace sm100
