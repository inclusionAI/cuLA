#pragma once

#include <cstdint>
#include "cuda_runtime_api.h"

namespace flat {

template <
    typename ArchTag,  // TODO: hide this
    typename TO,
    typename TQKV,
    typename TState>
void launch_kda_fwd_prefill_kernel(
    cudaStream_t   stream,
    TO*            output,
    TState*        output_state,
    TQKV const*    q,
    TQKV const*    k,
    TQKV const*    v,
    TState const*  input_state,
    float const*   alpha,
    float const*   beta,
    int64_t const* cu_seqlens,
    uint8_t*       workspace_buffer,
    int32_t        num_seqs,
    int32_t        num_q_heads,
    int32_t        num_k_heads,
    int32_t        num_v_heads,
    int32_t        num_o_heads,
    int32_t        head_size,
    int64_t        total_seqlen,
    float          scale,
    bool           safe_gate,
    int32_t        sm_count = 0
);

}  // namespace flat
