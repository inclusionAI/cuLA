// Dispatch function only — does NOT include the .cuh to avoid
// implicit instantiation of all kernel variants in one TU.
// Each SafeGate variant is explicitly instantiated in its own .cu file.

#include "cute/numeric/numeric_types.hpp"
#include "cutlass/arch/arch.h"

namespace flat {

using namespace cute;

// Forward declaration of the per-variant launcher (defined in .cuh, instantiated in separate TUs)
template <
    bool IsGVA,
    bool NeedsBeta,
    bool NeedsAlpha,
    bool InitStateFromInput,
    bool SafeGate,
    typename ArchTag,
    typename TO,
    typename TQKV,
    typename TState>
void launch_kda_fwd_prefill_kernel_gbai(
    cudaStream_t   stream,
    TO*            output,
    TState*        output_state,
    TQKV const*    q,
    TQKV const*    k,
    TQKV const*    v,
    TState const*  input_state,
    float const*   alpha,
    float const*   beta,
    int32_t const* cu_seqlens,
    uint8_t*       workspace_buffer,
    int32_t        num_seqs,
    int32_t        num_q_heads,
    int32_t        num_k_heads,
    int32_t        num_v_heads,
    int32_t        num_o_heads,
    int32_t        head_size,
    int64_t        total_seqlen,
    float          scale,
    int32_t        sm_count
);

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
    int32_t const* cu_seqlens,
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
) {
  bool is_gva      = num_v_heads > num_q_heads;
  bool needs_beta  = beta != nullptr;
  bool needs_alpha = alpha != nullptr;
  bool init_state  = input_state != nullptr;

#define LAUNCH(is_gva, needs_beta, needs_alpha, init_state, safe_gate)                                       \
  launch_kda_fwd_prefill_kernel_gbai<is_gva, needs_beta, needs_alpha, init_state, safe_gate, ArchTag>(       \
      stream, output, output_state, q, k, v, input_state, alpha, beta, cu_seqlens, workspace_buffer,         \
      num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen, scale, sm_count \
  );
  if (init_state) {
    if (is_gva && needs_beta && needs_alpha) {
      // LAUNCH(true, true, true, true);
    } else if (is_gva && needs_beta && !needs_alpha) {
      // LAUNCH(true, true, false, true);
    } else if (is_gva && !needs_beta && needs_alpha) {
      // LAUNCH(true, false, true, true);
    } else if (is_gva && !needs_beta && !needs_alpha) {
      // LAUNCH(true, false, false, true);
    } else if (!is_gva && needs_beta && needs_alpha && safe_gate) {
      LAUNCH(false, true, true, true, true);
    } else if (!is_gva && needs_beta && needs_alpha && !safe_gate) {
      // LAUNCH(false, true, true, true, false);
    } else if (!is_gva && needs_beta && !needs_alpha) {
      // LAUNCH(false, true, false, true);
    } else if (!is_gva && !needs_beta && needs_alpha) {
      // LAUNCH(false, false, true, true);
    } else if (!is_gva && !needs_beta && !needs_alpha) {
      // LAUNCH(false, false, false, true);
    } else {
      throw std::runtime_error("unreachable");
    }
  } else {
    if (is_gva && needs_beta && needs_alpha) {
      // LAUNCH(true, true, true, false);
    } else if (is_gva && needs_beta && !needs_alpha) {
      // LAUNCH(true, true, false, false);
    } else if (is_gva && !needs_beta && needs_alpha) {
      // LAUNCH(true, false, true, false);
    } else if (is_gva && !needs_beta && !needs_alpha) {
      // LAUNCH(true, false, false, false);
    } else if (!is_gva && needs_beta && needs_alpha && safe_gate) {
      LAUNCH(false, true, true, false, true);
    } else if (!is_gva && needs_beta && needs_alpha && !safe_gate) {
      // LAUNCH(false, true, true, false, false);
    } else if (!is_gva && needs_beta && !needs_alpha) {
      // LAUNCH(false, true, false, false);
    } else if (!is_gva && !needs_beta && needs_alpha) {
      // LAUNCH(false, false, true, false);
    } else if (!is_gva && !needs_beta && !needs_alpha) {
      // LAUNCH(false, false, false, false);
    } else {
      throw std::runtime_error("unreachable");
    }
  }

#undef LAUNCH
}

using bf16 = cute::bfloat16_t;

template void launch_kda_fwd_prefill_kernel<cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t   stream,
    bf16*          output,
    float*         state,
    bf16 const*    q,
    bf16 const*    k,
    bf16 const*    v,
    float const*   input_state,
    float const*   alpha,
    float const*   beta,
    int32_t const* cu_seqlens,
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
    int32_t        sm_count
);

}  // namespace flat
