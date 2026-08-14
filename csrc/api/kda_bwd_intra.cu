// Copyright 2025-2026 Ant Group Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "kda/sm90/bwd/kda_config.h"

namespace sm90 {
void
run_kda_bwd_intra_sm90(KDA_bwd_intra_params& params, cudaStream_t stream);
}

namespace {

void
check_cuda_contiguous(at::Tensor const& tensor, char const* name, at::Device const& device) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device, ", got ", tensor.device());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void
check_same_shape(at::Tensor const& tensor, at::Tensor const& expected, char const* name, char const* expected_name) {
    TORCH_CHECK(
        tensor.sizes() == expected.sizes(),
        name,
        " must have the same shape as ",
        expected_name,
        ", got ",
        tensor.sizes(),
        " vs ",
        expected.sizes());
}

}  // namespace

void
ChunkKDABwdIntra(
    at::Tensor q,
    at::Tensor k,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor dAqk,
    at::Tensor dAkk,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor db,
    at::Tensor dg,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor dq_out,
    at::Tensor dk_out,
    at::Tensor db_out,
    at::Tensor dg_out,
    int64_t chunk_size) {
    auto const device = q.device();
    check_cuda_contiguous(q, "q", device);
    check_cuda_contiguous(k, "k", device);
    check_cuda_contiguous(g, "g", device);
    check_cuda_contiguous(beta, "beta", device);
    check_cuda_contiguous(dAqk, "dAqk", device);
    check_cuda_contiguous(dAkk, "dAkk", device);
    check_cuda_contiguous(dq, "dq", device);
    check_cuda_contiguous(dk, "dk", device);
    check_cuda_contiguous(db, "db", device);
    check_cuda_contiguous(dg, "dg", device);
    check_cuda_contiguous(cu_seqlens, "cu_seqlens", device);
    check_cuda_contiguous(chunk_indices, "chunk_indices", device);
    check_cuda_contiguous(dq_out, "dq_out", device);
    check_cuda_contiguous(dk_out, "dk_out", device);
    check_cuda_contiguous(db_out, "db_out", device);
    check_cuda_contiguous(dg_out, "dg_out", device);

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
    TORCH_CHECK(dq_out.scalar_type() == at::kBFloat16, "dq_out must be bfloat16");
    TORCH_CHECK(dk_out.scalar_type() == at::kBFloat16, "dk_out must be bfloat16");
    for (auto const& item : {
             std::pair<at::Tensor const*, char const*>{&g, "g"},
             {&dAqk, "dAqk"},
             {&dAkk, "dAkk"},
             {&dq, "dq"},
             {&dk, "dk"},
             {&db, "db"},
             {&dg, "dg"},
             {&db_out, "db_out"},
             {&dg_out, "dg_out"},
         }) {
        TORCH_CHECK(item.first->scalar_type() == at::kFloat, item.second, " must be float32");
    }
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.scalar_type() == at::kInt, "chunk_indices must be int32");

    TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, H, K]");
    check_same_shape(k, q, "k", "q");
    check_same_shape(g, q, "g", "q");
    check_same_shape(dq, q, "dq", "q");
    check_same_shape(dk, q, "dk", "q");
    check_same_shape(dg, q, "dg", "q");
    check_same_shape(dq_out, q, "dq_out", "q");
    check_same_shape(dk_out, q, "dk_out", "q");
    check_same_shape(dg_out, q, "dg_out", "q");
    TORCH_CHECK(beta.dim() == 3, "beta must have shape [B, T, H]");
    TORCH_CHECK(db.sizes() == beta.sizes(), "db must have the same shape as beta");
    TORCH_CHECK(
        beta.size(0) == q.size(0) && beta.size(1) == q.size(1) && beta.size(2) == q.size(2),
        "beta shape must match q[:3]");
    TORCH_CHECK(
        dAqk.dim() == 4 && dAqk.size(0) == q.size(0) && dAqk.size(1) == q.size(1) && dAqk.size(2) == q.size(2) &&
            dAqk.size(3) == chunk_size,
        "dAqk must have shape [B, T, H, chunk_size]");
    TORCH_CHECK(dAkk.sizes() == dAqk.sizes(), "dAkk must have the same shape as dAqk");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2, "cu_seqlens must have shape [num_sequences + 1]");
    TORCH_CHECK(
        chunk_indices.dim() == 2 && chunk_indices.size(1) == 2, "chunk_indices must have shape [num_chunks, 2]");
    TORCH_CHECK(db_out.sizes() == beta.sizes(), "db_out must have the same shape as beta");

    TORCH_CHECK(chunk_size == 64, "chunk_kda_bwd_intra_cuda supports only chunk_size=64, got ", chunk_size);
    TORCH_CHECK(q.size(3) == 128, "chunk_kda_bwd_intra_cuda supports only K=128, got ", q.size(3));
    TORCH_CHECK(q.numel() > 0, "q must be non-empty");
    auto const total_q_len = q.size(0) * q.size(1);
    TORCH_CHECK(total_q_len <= std::numeric_limits<int>::max(), "B*T exceeds int32 range");
    TORCH_CHECK(q.size(2) <= std::numeric_limits<int>::max(), "H exceeds int32 range");
    TORCH_CHECK(chunk_indices.size(0) > 0, "chunk_indices must contain at least one chunk");
    TORCH_CHECK(chunk_indices.size(0) <= std::numeric_limits<int>::max(), "number of chunks exceeds int32 range");
    TORCH_CHECK(
        chunk_indices.size(0) <= std::numeric_limits<int>::max() / q.size(2), "num_chunks * H exceeds int32 range");

    c10::cuda::CUDAGuard device_guard(device);
    KDA_bwd_intra_params params{};
    params.total_q_len = static_cast<int>(total_q_len);
    params.h = static_cast<int>(q.size(2));
    params.d = static_cast<int>(q.size(3));
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.g_ptr = g.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.dAqk_ptr = dAqk.data_ptr();
    params.dAkk_ptr = dAkk.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dg_ptr = dg.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    params.dq_out_ptr = dq_out.data_ptr();
    params.dk_out_ptr = dk_out.data_ptr();
    params.dg_out_ptr = dg_out.data_ptr();
    params.num_chunks = static_cast<int>(chunk_indices.size(0));

    auto const num_k_tiles = q.size(3) / 32;
    auto db_partials = at::zeros({num_k_tiles, beta.size(0), beta.size(1), beta.size(2)}, db.options());
    params.db2_ptr = db_partials.data_ptr();

    sm90::run_kda_bwd_intra_sm90(params, at::cuda::getCurrentCUDAStream());
    db_out.copy_(db_partials.sum(0).add_(db));
}
