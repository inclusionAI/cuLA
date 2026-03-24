#include "torch/extension.h"
#include "ATen/cuda/CUDAContext.h"

#include "cute/numeric/numeric_types.hpp"
#include "cutlass/arch/arch.h"

#include "kda/sm90/prefill_kernel.hpp"

using OptionalTensor = std::optional<torch::Tensor>;

std::tuple<torch::Tensor, torch::Tensor>
kda_fwd_prefill(
    OptionalTensor       output_,
    OptionalTensor       output_state_,
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    OptionalTensor       input_state_,
    OptionalTensor       alpha_,
    OptionalTensor       beta_,
    torch::Tensor const& cu_seqlens,
    torch::Tensor        workspace_buffer,
    float                scale,
    bool                 safe_gate
) {
    // Q, K, V: [packed_seq, H, D] (already packed by Python layer)
    auto packed_seq   = q.size(0);
    auto num_q_heads  = q.size(1);
    auto head_size    = q.size(2);
    auto num_k_heads  = k.size(1);
    auto num_v_heads  = v.size(1);
    auto num_seqs     = cu_seqlens.size(0) - 1;

    // KDA constraint: num_q_heads == num_v_heads
    TORCH_CHECK(num_q_heads == num_v_heads,
        "KDA requires num_q_heads == num_v_heads, got ", num_q_heads, " vs ", num_v_heads);
    TORCH_CHECK(head_size == v.size(2),
        "KDA requires K == V head dim, got ", head_size, " vs ", v.size(2));

    // GQA check
    if (num_k_heads != num_v_heads) {
        TORCH_CHECK(num_q_heads % num_k_heads == 0,
            "GQA: num_q_heads must be divisible by num_k_heads");
    }

    auto num_o_heads   = num_q_heads;
    auto num_sab_heads = std::max(num_q_heads, num_v_heads);

    // Allocate output if not provided
    torch::Tensor output = output_.has_value() ? output_.value()
        : torch::empty({packed_seq, num_o_heads, head_size},
                       torch::TensorOptions().dtype(q.dtype()).device(q.device()));

    // Allocate output state if not provided
    torch::Tensor output_state = output_state_.has_value() ? output_state_.value()
        : torch::zeros({num_seqs, num_sab_heads, head_size, head_size},
                       torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

    // Validate dtypes
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt64, "cu_seqlens must be int64");

    // Validate contiguity
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output_state.is_contiguous(), "output_state must be contiguous");
    TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
    TORCH_CHECK(workspace_buffer.is_contiguous(), "workspace_buffer must be contiguous");

    // Extract optional pointers
    float const* alpha_ptr = nullptr;
    float const* beta_ptr  = nullptr;
    float const* input_state_ptr = nullptr;

    if (alpha_.has_value()) {
        auto& alpha = alpha_.value();
        TORCH_CHECK(alpha.dtype() == torch::kFloat32, "alpha must be float32");
        TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
        TORCH_CHECK(alpha.size(0) == packed_seq && alpha.size(1) == num_sab_heads && alpha.size(2) == head_size,
            "alpha shape must be [packed_seq, num_sab_heads, head_size]");
        alpha_ptr = alpha.data_ptr<float>();
    }
    if (beta_.has_value()) {
        auto& beta = beta_.value();
        TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be float32");
        TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
        TORCH_CHECK(beta.size(0) == packed_seq && beta.size(1) == num_sab_heads,
            "beta shape must be [packed_seq, num_sab_heads]");
        beta_ptr = beta.data_ptr<float>();
    }
    if (input_state_.has_value()) {
        auto& input_state = input_state_.value();
        TORCH_CHECK(input_state.dtype() == torch::kFloat32, "input_state must be float32");
        TORCH_CHECK(input_state.is_contiguous(), "input_state must be contiguous");
        input_state_ptr = input_state.data_ptr<float>();
    }

    // Auto-compute scale if 0
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    }

    auto stream   = at::cuda::getCurrentCUDAStream();
    auto sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    using bf16 = cute::bfloat16_t;
    using Sm90 = cutlass::arch::Sm90;

    flat::launch_kda_fwd_prefill_kernel<Sm90, bf16, bf16, float>(
        stream,
        reinterpret_cast<bf16*>(output.data_ptr()),
        output_state.data_ptr<float>(),
        reinterpret_cast<bf16 const*>(q.data_ptr()),
        reinterpret_cast<bf16 const*>(k.data_ptr()),
        reinterpret_cast<bf16 const*>(v.data_ptr()),
        input_state_ptr,
        alpha_ptr,
        beta_ptr,
        cu_seqlens.data_ptr<int64_t>(),
        workspace_buffer.data_ptr<uint8_t>(),
        static_cast<int32_t>(num_seqs),
        static_cast<int32_t>(num_q_heads),
        static_cast<int32_t>(num_k_heads),
        static_cast<int32_t>(num_v_heads),
        static_cast<int32_t>(num_o_heads),
        static_cast<int32_t>(head_size),
        static_cast<int64_t>(packed_seq),
        scale,
        safe_gate,
        static_cast<int32_t>(sm_count)
    );

    return {output, output_state};
}
