#include "kda_fwd_common.cuh"
#include "kda_fwd_intra_kernel_sm100.hpp"

namespace flashla {

void run_kda_fwd_intra_sm100(KDA_fwd_intra_params &params, cudaStream_t stream) {
    flashla::run_kda_fwd_intra_sm100_impl(params, stream);
}

} // namespace flashla