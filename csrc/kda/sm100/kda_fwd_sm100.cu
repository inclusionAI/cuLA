#include "kda_fwd_common.cuh"
#include "kda_fwd_intra_kernel_sm100.hpp"

namespace kda::sm100 {

void run_kda_fwd_intra_sm100(KDA_fwd_intra_params &params, cudaStream_t stream) {
    kda::sm100::run_kda_fwd_intra_sm100_impl(params, stream);
}

} // namespace kda::sm100