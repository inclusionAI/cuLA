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

#include "qwen35_prefill_common.cuh"
#include "qwen35_scalar_kda_prefill_kernel.hpp"
#ifdef CULA_SM90A_ENABLED
#include "kda/sm90/prefill_kernel.hpp"
#endif
#ifdef CULA_SM100_ENABLED
#include "kda/sm100/kda_fwd_common.cuh"
#include "qwen35_chunk_state_output_sm100.hpp"
#include "qwen35_chunk_state_output_sm100_ss.hpp"
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include <algorithm>

namespace cula::qwen35::prefill {

namespace {

void check_tensor_device(const at::Tensor& tensor, const char* name, const at::Device& device) {
  if (tensor.defined() && tensor.numel() > 0) {
    TORCH_CHECK(tensor.device() == device, name, " must be on device ", device, ".");
  }
}

void check_contiguous(const at::Tensor& tensor, const char* name) {
  if (tensor.defined() && tensor.numel() > 0) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  }
}

template <typename scalar_t, int kLocalVHeads>
void dispatch_scalar_prefill_for_heads(
    cudaStream_t stream,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    const int32_t* cu_seqlens,
    scalar_t* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    int sequence_count,
    bool is_varlen,
    bool has_initial_state) {
  kernel::launch_qwen35_scalar_kda_prefill_kernel<scalar_t, kLocalVHeads>(
      stream,
      q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      initial_state,
      cu_seqlens,
      out,
      final_state,
      batch_size,
      seq_len,
      qk_heads,
      sequence_count,
      is_varlen,
      has_initial_state);
}

template <typename scalar_t>
void dispatch_scalar_prefill(
    int64_t local_v_heads,
    cudaStream_t stream,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    const int32_t* cu_seqlens,
    scalar_t* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    int sequence_count,
    bool is_varlen,
    bool has_initial_state) {
  switch (local_v_heads) {
    case 64:
      dispatch_scalar_prefill_for_heads<scalar_t, 64>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 48:
      dispatch_scalar_prefill_for_heads<scalar_t, 48>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 32:
      dispatch_scalar_prefill_for_heads<scalar_t, 32>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 24:
      dispatch_scalar_prefill_for_heads<scalar_t, 24>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 16:
      dispatch_scalar_prefill_for_heads<scalar_t, 16>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 12:
      dispatch_scalar_prefill_for_heads<scalar_t, 12>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 8:
      dispatch_scalar_prefill_for_heads<scalar_t, 8>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 6:
      dispatch_scalar_prefill_for_heads<scalar_t, 6>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 4:
      dispatch_scalar_prefill_for_heads<scalar_t, 4>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    case 2:
      dispatch_scalar_prefill_for_heads<scalar_t, 2>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, qk_heads, sequence_count, is_varlen, has_initial_state);
      break;
    default:
      TORCH_CHECK(false, "unsupported scalar prefill local V-head count: ", local_v_heads);
  }
}

template <typename scalar_t>
void dispatch_scalar_prefill_precomputed_fallback(
    cudaStream_t stream,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const float* g,
    const float* beta,
    const float* initial_state,
    scalar_t* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    int local_v_heads,
    bool has_initial_state,
    const int32_t* unsafe_gate_flags) {
#define CULA_QWEN35_FALLBACK_CASE(HV)                                                        \
  case HV:                                                                                   \
    kernel::launch_qwen35_scalar_kda_prefill_precomputed_fallback<scalar_t, HV>(             \
        stream, q, k, v, g, beta, initial_state, out, final_state,                           \
        batch_size, seq_len, qk_heads, has_initial_state, unsafe_gate_flags);                \
    break
  switch (local_v_heads) {
    CULA_QWEN35_FALLBACK_CASE(64);
    CULA_QWEN35_FALLBACK_CASE(48);
    CULA_QWEN35_FALLBACK_CASE(32);
    CULA_QWEN35_FALLBACK_CASE(24);
    CULA_QWEN35_FALLBACK_CASE(16);
    CULA_QWEN35_FALLBACK_CASE(12);
    CULA_QWEN35_FALLBACK_CASE(8);
    CULA_QWEN35_FALLBACK_CASE(6);
    CULA_QWEN35_FALLBACK_CASE(4);
    CULA_QWEN35_FALLBACK_CASE(2);
    default:
      TORCH_CHECK(false, "Unsupported local V head count: ", local_v_heads, ".");
  }
#undef CULA_QWEN35_FALLBACK_CASE
}

#ifdef CULA_SM90A_ENABLED
void run_qwen35_chunk_prefill_sm90_bf16(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& initial_state,
    const at::Tensor& out,
    const at::Tensor& final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    int local_v_heads,
    bool has_initial_state,
    cudaStream_t stream,
    const at::Tensor* precomputed_g = nullptr,
    const at::Tensor* precomputed_beta = nullptr) {
  const bool use_sm90_chunk = local_v_heads % 4 == 0;
  TORCH_CHECK(
      (precomputed_g == nullptr) == (precomputed_beta == nullptr),
      "precomputed gate and beta must be supplied together.");
  constexpr int chunk_size = kernel::kChunkSize;
  const int chunks_per_sequence = (seq_len + chunk_size - 1) / chunk_size;
  const int total_chunks = batch_size * chunks_per_sequence;
  const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len;
  const auto bf16_options = q.options().dtype(at::kBFloat16);
  const auto fp32_options = q.options().dtype(at::kFloat);
  const auto int_options = q.options().dtype(at::kInt);

  at::Tensor q_norm = at::empty_like(q, bf16_options);
  at::Tensor k_norm = at::empty_like(k, bf16_options);
  at::Tensor g = at::empty({batch_size, seq_len, local_v_heads}, fp32_options);
  at::Tensor g_raw = precomputed_g == nullptr
      ? at::empty({batch_size, seq_len, local_v_heads}, fp32_options)
      : *precomputed_g;
  // Always materialize beta into private workspace. The core ABI input may be
  // aliased or reused concurrently and must remain read-only.
  at::Tensor beta = at::empty({batch_size, seq_len, local_v_heads}, fp32_options);
  // HV=6/2 TP shards cannot satisfy Hopper TMA's four-adjacent-head scalar
  // gate transaction. Keep the core ABI correct by marking every head for the
  // exact recurrent path; divisible-by-four shapes use speculative SM90 KDA.
  at::Tensor unsafe_gate_flags = use_sm90_chunk
      ? at::zeros({batch_size, local_v_heads}, int_options)
      : at::ones({batch_size, local_v_heads}, int_options);
  at::Tensor cu_work = at::empty({batch_size + 1}, int_options);
  at::Tensor chunk_indices = at::empty({total_chunks, 2}, int_options);

  kernel::launch_qwen35_chunk_preprocess(
      stream,
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<c10::BFloat16>()),
      precomputed_g == nullptr
          ? reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<c10::BFloat16>())
          : nullptr,
      precomputed_g == nullptr
          ? reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<c10::BFloat16>())
          : nullptr,
      precomputed_g == nullptr ? A_log.data_ptr<float>() : nullptr,
      precomputed_g == nullptr ? dt_bias.data_ptr<float>() : nullptr,
      precomputed_g == nullptr ? nullptr : precomputed_g->data_ptr<float>(),
      precomputed_beta == nullptr ? nullptr : precomputed_beta->data_ptr<float>(),
      precomputed_g != nullptr,
      reinterpret_cast<__nv_bfloat16*>(q_norm.data_ptr<c10::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(k_norm.data_ptr<c10::BFloat16>()),
      g.data_ptr<float>(),
      precomputed_g == nullptr ? g_raw.data_ptr<float>() : nullptr,
      beta.data_ptr<float>(),
      unsafe_gate_flags.data_ptr<int32_t>(),
      cu_work.data_ptr<int32_t>(),
      chunk_indices.data_ptr<int32_t>(),
      batch_size,
      seq_len,
      qk_heads,
      local_v_heads);

  if (use_sm90_chunk) {
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    at::Tensor workspace =
        at::empty({static_cast<int64_t>(sm_count) * 128}, q.options().dtype(at::kByte));
    kda::sm90::launch_qwen35_scalar_kda_fwd_prefill_kernel(
        stream,
        out.data_ptr(),
        final_state.data_ptr<float>(),
        q_norm.data_ptr(),
        k_norm.data_ptr(),
        v.data_ptr(),
        has_initial_state ? initial_state.data_ptr<float>() : nullptr,
        g.data_ptr<float>(),
        beta.data_ptr<float>(),
        cu_work.data_ptr<int32_t>(),
        workspace.data_ptr<uint8_t>(),
        batch_size,
        qk_heads,
        local_v_heads,
        kHeadDimQK,
        total_tokens,
        rsqrtf(static_cast<float>(kHeadDimQK)),
        has_initial_state,
        sm_count);
  }

  // The SM90 fully-fused safe-gate algebra assumes raw log gates in [-5, 0].
  // Preprocess marks unsafe (sequence, V-head) pairs while it already reads
  // the raw gates. A lightweight recurrent launch returns immediately for
  // safe heads; unsafe heads overwrite the speculative fast result exactly,
  // without a host sync or Python-side branch.
  dispatch_scalar_prefill_precomputed_fallback<c10::BFloat16>(
      stream,
      q_norm.data_ptr<c10::BFloat16>(),
      k_norm.data_ptr<c10::BFloat16>(),
      v.data_ptr<c10::BFloat16>(),
      g_raw.data_ptr<float>(),
      beta.data_ptr<float>(),
      has_initial_state ? initial_state.data_ptr<float>() : nullptr,
      out.data_ptr<c10::BFloat16>(),
      final_state.data_ptr<float>(),
      batch_size,
      seq_len,
      qk_heads,
      local_v_heads,
      has_initial_state,
      unsafe_gate_flags.data_ptr<int32_t>());
}
#endif

#ifdef CULA_SM100_ENABLED
// The TS-UMMA implementation is the default SM100 chunk state/output path.
// Keep compile-time escape hatches for A/B comparisons with the WMMA and
// standalone SS prototypes without changing the Python ABI or launch args.
#ifndef CULA_QWEN35_USE_WMMA_CHUNK
#define CULA_QWEN35_USE_WMMA_CHUNK 0
#endif
#ifndef CULA_QWEN35_USE_TS_CHUNK
#define CULA_QWEN35_USE_TS_CHUNK 1
#endif

template <int kLocalVHeads>
void launch_chunk_state_output_for_heads(
    cudaStream_t stream,
    const at::Tensor& q_norm,
    const at::Tensor& g,
    const at::Tensor& Aqk,
    const at::Tensor& w,
    const at::Tensor& u,
    const at::Tensor& kg,
    const at::Tensor& initial_state,
    const at::Tensor& out,
    const at::Tensor& final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    bool has_initial_state) {
#if CULA_QWEN35_USE_WMMA_CHUNK
  kernel::launch_qwen35_chunk_state_output<kLocalVHeads>(
      stream,
      reinterpret_cast<const __nv_bfloat16*>(q_norm.data_ptr<c10::BFloat16>()),
      g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(Aqk.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(kg.data_ptr<c10::BFloat16>()),
      has_initial_state ? initial_state.data_ptr<float>() : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
      final_state.data_ptr<float>(),
      batch_size,
      seq_len,
      qk_heads,
      has_initial_state);
#elif CULA_QWEN35_USE_TS_CHUNK
  kernel::sm100_ts::launch_qwen35_chunk_state_output_sm100_ts<kLocalVHeads>(
      stream,
      reinterpret_cast<const __nv_bfloat16*>(q_norm.data_ptr<c10::BFloat16>()),
      g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(Aqk.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(kg.data_ptr<c10::BFloat16>()),
      has_initial_state ? initial_state.data_ptr<float>() : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
      final_state.data_ptr<float>(),
      batch_size,
      seq_len,
      qk_heads,
      has_initial_state);
#else
  kernel::sm100_ss::launch_qwen35_chunk_state_output_sm100_ss<kLocalVHeads>(
      stream,
      reinterpret_cast<const __nv_bfloat16*>(q_norm.data_ptr<c10::BFloat16>()),
      g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(Aqk.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(kg.data_ptr<c10::BFloat16>()),
      has_initial_state ? initial_state.data_ptr<float>() : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>()),
      final_state.data_ptr<float>(),
      batch_size,
      seq_len,
      qk_heads,
      has_initial_state);
#endif
}

void launch_chunk_state_output(
    int64_t local_v_heads,
    cudaStream_t stream,
    const at::Tensor& q_norm,
    const at::Tensor& g,
    const at::Tensor& Aqk,
    const at::Tensor& w,
    const at::Tensor& u,
    const at::Tensor& kg,
    const at::Tensor& initial_state,
    const at::Tensor& out,
    const at::Tensor& final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    bool has_initial_state) {
#define CULA_QWEN35_CHUNK_HEAD_CASE(HV)                                                               \
  case HV:                                                                                            \
    launch_chunk_state_output_for_heads<HV>(                                                          \
        stream, q_norm, g, Aqk, w, u, kg, initial_state, out, final_state, batch_size, seq_len,       \
        qk_heads, has_initial_state);                                                                 \
    break
  switch (local_v_heads) {
    CULA_QWEN35_CHUNK_HEAD_CASE(64);
    CULA_QWEN35_CHUNK_HEAD_CASE(48);
    CULA_QWEN35_CHUNK_HEAD_CASE(32);
    CULA_QWEN35_CHUNK_HEAD_CASE(24);
    CULA_QWEN35_CHUNK_HEAD_CASE(16);
    CULA_QWEN35_CHUNK_HEAD_CASE(12);
    CULA_QWEN35_CHUNK_HEAD_CASE(8);
    CULA_QWEN35_CHUNK_HEAD_CASE(6);
    CULA_QWEN35_CHUNK_HEAD_CASE(4);
    CULA_QWEN35_CHUNK_HEAD_CASE(2);
    default:
      TORCH_CHECK(false, "unsupported chunk prefill local V-head count: ", local_v_heads);
  }
#undef CULA_QWEN35_CHUNK_HEAD_CASE
}

void run_qwen35_chunk_prefill_bf16(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& initial_state,
    const at::Tensor& out,
    const at::Tensor& final_state,
    int batch_size,
    int seq_len,
    int qk_heads,
    int local_v_heads,
    bool has_initial_state,
    cudaStream_t stream,
    const at::Tensor* precomputed_g = nullptr,
    const at::Tensor* precomputed_beta = nullptr) {
  TORCH_CHECK(
      (precomputed_g == nullptr) == (precomputed_beta == nullptr),
      "precomputed gate and beta must be supplied together.");
  constexpr int chunk_size = kernel::kChunkSize;
  const int chunks_per_sequence = (seq_len + chunk_size - 1) / chunk_size;
  const int total_chunks = batch_size * chunks_per_sequence;
  const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len;
  const auto bf16_options = q.options().dtype(at::kBFloat16);
  const auto fp32_options = q.options().dtype(at::kFloat);
  const auto int_options = q.options().dtype(at::kInt);

  // These tensors are genuine CUDA workspaces; no Python/reference operation
  // participates in the numerical result.  They are deliberately explicit
  // while the chunk path is stabilized, and can later be supplied by an
  // inference workspace pool without changing the kernels.
  at::Tensor q_norm = at::empty({batch_size, seq_len, qk_heads, kHeadDimQK}, bf16_options);
  at::Tensor k_norm = at::empty_like(q_norm);
  // Qwen GDN has one scalar gate per token/value-head. Keep it compact and
  // route only this adapter through the scalar-G KDA specializations.
  at::Tensor g = at::empty({batch_size, seq_len, local_v_heads}, fp32_options);
  at::Tensor beta = at::empty({batch_size, seq_len, local_v_heads}, fp32_options);
  at::Tensor cu_work = at::empty({batch_size + 1}, int_options);
  at::Tensor chunk_indices = at::empty({total_chunks, 2}, int_options);
  at::Tensor Aqk = at::empty({batch_size, seq_len, local_v_heads, chunk_size}, bf16_options);
  at::Tensor Akk = at::empty_like(Aqk);
  at::Tensor w = at::empty({batch_size, seq_len, local_v_heads, kHeadDimQK}, bf16_options);
  at::Tensor u = at::empty_like(w);
  at::Tensor kg = at::empty_like(w);

  kernel::launch_qwen35_chunk_preprocess(
      stream,
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<c10::BFloat16>()),
      precomputed_g == nullptr
          ? reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<c10::BFloat16>())
          : nullptr,
      precomputed_g == nullptr
          ? reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<c10::BFloat16>())
          : nullptr,
      precomputed_g == nullptr ? A_log.data_ptr<float>() : nullptr,
      precomputed_g == nullptr ? dt_bias.data_ptr<float>() : nullptr,
      precomputed_g == nullptr ? nullptr : precomputed_g->data_ptr<float>(),
      precomputed_beta == nullptr ? nullptr : precomputed_beta->data_ptr<float>(),
      precomputed_g != nullptr && precomputed_beta != nullptr,
      reinterpret_cast<__nv_bfloat16*>(q_norm.data_ptr<c10::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(k_norm.data_ptr<c10::BFloat16>()),
      g.data_ptr<float>(),
      nullptr,
      beta.data_ptr<float>(),
      nullptr,
      cu_work.data_ptr<int32_t>(),
      chunk_indices.data_ptr<int32_t>(),
      batch_size,
      seq_len,
      qk_heads,
      local_v_heads);

  auto* device_prop = at::cuda::getCurrentDeviceProperties();
  const int scheduler_tiles = total_chunks * local_v_heads;
  const int scheduler_sms = std::min(device_prop->multiProcessorCount, scheduler_tiles);
  KDA_fwd_intra_params intra{};
  intra.total_q_len = static_cast<int>(total_tokens);
  intra.b = batch_size;
  intra.h_qk = qk_heads;
  intra.h_v = local_v_heads;
  intra.heads_per_group = local_v_heads / qk_heads;
  intra.d = kHeadDimQK;
  intra.chunk_size = chunk_size;
  intra.scale = rsqrtf(static_cast<float>(kHeadDimQK));
  intra.use_tf32_inverse = false;
  intra.unified_gref = true;
  intra.is_beta_bf16 = false;
  intra.q_ptr = q_norm.data_ptr();
  intra.k_ptr = k_norm.data_ptr();
  intra.g_ptr = g.data_ptr();
  intra.beta_ptr = beta.data_ptr();
  intra.Aqk_out_ptr = Aqk.data_ptr();
  intra.Akk_out_ptr = Akk.data_ptr();
  intra.cu_seqlens_ptr = cu_work.data_ptr();
  intra.chunk_indices_ptr = chunk_indices.data_ptr();
  intra.shape_Akk = cute::make_shape(intra.total_q_len, chunk_size, local_v_heads);
  intra.stride_Akk = cute::make_stride(chunk_size * local_v_heads, cute::_1{}, chunk_size);
  intra.num_sm = scheduler_sms;
  intra.tile_scheduler_params = StaticPersistentTileScheduler::Params{
      total_chunks,
      local_v_heads,
      intra.heads_per_group,
      intra.num_sm,
      nullptr};
  kda::sm100::run_kda_fwd_intra_sm100_qwen_scalar_g(intra, stream);

  KDA_fwd_recomp_w_u_params recomp{};
  recomp.total_len = static_cast<int>(total_tokens);
  recomp.b = batch_size;
  recomp.h_qk = qk_heads;
  recomp.h_v = local_v_heads;
  recomp.heads_per_group = local_v_heads / qk_heads;
  recomp.d = kHeadDimQK;
  recomp.chunk_size = chunk_size;
  recomp.is_beta_bf16 = false;
  recomp.k_ptr = k_norm.data_ptr();
  recomp.v_ptr = v.data_ptr();
  recomp.q_ptr = q_norm.data_ptr();
  recomp.beta_ptr = beta.data_ptr();
  recomp.A_ptr = Akk.data_ptr();
  recomp.g_ptr = g.data_ptr();
  recomp.cu_seqlens_ptr = cu_work.data_ptr();
  recomp.chunk_indices_ptr = chunk_indices.data_ptr();
  recomp.w_out_ptr = w.data_ptr();
  recomp.u_out_ptr = u.data_ptr();
  recomp.kg_out_ptr = kg.data_ptr();
  recomp.qg_out_ptr = nullptr;
  recomp.store_qg = false;
  recomp.shape_wukg = cute::make_shape(recomp.total_len, kHeadDimQK, local_v_heads);
  recomp.stride_wukg = cute::make_stride(kHeadDimQK * local_v_heads, cute::_1{}, kHeadDimQK);
  recomp.num_sm = scheduler_sms;
  recomp.tile_scheduler_params = StaticPersistentTileScheduler::Params{
      total_chunks, local_v_heads, recomp.heads_per_group, recomp.num_sm, nullptr};
  kda::sm100::run_kda_fwd_recomp_w_u_sm100_qwen_scalar_g(recomp, stream);

  launch_chunk_state_output(
      local_v_heads,
      stream,
      q_norm,
      g,
      Aqk,
      w,
      u,
      kg,
      initial_state,
      out,
      final_state,
      batch_size,
      seq_len,
      qk_heads,
      has_initial_state);
}
#endif

} // namespace

void run_qwen35_scalar_kda_prefill(ScalarKdaPrefillParams& params) {
  const at::Tensor& q = params.q;
  const at::Tensor& k = params.k;
  const at::Tensor& v = params.v;
  const at::Tensor& a = params.a;
  const at::Tensor& b = params.b;
  const at::Tensor& A_log = params.A_log;
  const at::Tensor& dt_bias = params.dt_bias;
  const at::Tensor& initial_state = params.initial_state;
  const at::Tensor& cu_seqlens = params.cu_seqlens;
  const at::Tensor& out = params.out;
  const at::Tensor& final_state = params.final_state;

  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor.");
  const at::Device device = q.device();

  check_tensor_device(k, "k", device);
  check_tensor_device(v, "v", device);
  check_tensor_device(a, "a", device);
  check_tensor_device(b, "b", device);
  check_tensor_device(A_log, "A_log", device);
  check_tensor_device(dt_bias, "dt_bias", device);
  check_tensor_device(initial_state, "initial_state", device);
  check_tensor_device(cu_seqlens, "cu_seqlens", device);
  check_tensor_device(out, "out", device);
  check_tensor_device(final_state, "final_state", device);

  check_contiguous(q, "q");
  check_contiguous(k, "k");
  check_contiguous(v, "v");
  check_contiguous(a, "a");
  check_contiguous(b, "b");
  check_contiguous(A_log, "A_log");
  check_contiguous(dt_bias, "dt_bias");
  check_contiguous(initial_state, "initial_state");
  check_contiguous(cu_seqlens, "cu_seqlens");
  check_contiguous(out, "out");
  check_contiguous(final_state, "final_state");

  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
          q.scalar_type() == a.scalar_type() && q.scalar_type() == b.scalar_type() &&
          q.scalar_type() == out.scalar_type(),
      "q/k/v/a/b/out must share the same dtype.");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "q must be float16 or bfloat16.");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat, "A_log must be float32.");
  TORCH_CHECK(dt_bias.scalar_type() == at::kFloat, "dt_bias must be float32.");
  TORCH_CHECK(final_state.scalar_type() == at::kFloat, "final_state must be float32.");
  TORCH_CHECK(
      !initial_state.defined() || initial_state.numel() == 0 || initial_state.scalar_type() == at::kFloat,
      "initial_state must be float32 when provided.");
  TORCH_CHECK(
      !cu_seqlens.defined() || cu_seqlens.numel() == 0 || cu_seqlens.scalar_type() == at::kInt,
      "cu_seqlens must be int32 when provided.");

  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D.");
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t qk_heads = q.size(2);
  const int64_t local_v_heads = v.size(2);
  TORCH_CHECK(qk_heads > 0 && local_v_heads > 0, "q/k/v head counts must be positive.");
  TORCH_CHECK(local_v_heads % qk_heads == 0, "local V heads must be divisible by local Q/K heads.");
  TORCH_CHECK(
      is_supported_scalar_prefill_v_heads(static_cast<int>(local_v_heads)),
      "unsupported Qwen scalar prefill local V-head count: ", local_v_heads, ".");
  TORCH_CHECK(
      q.sizes() == at::IntArrayRef({B, T, qk_heads, kHeadDimQK}),
      "q must have shape [B, T, local_qk_heads, 128].");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape.");
  TORCH_CHECK(
      v.sizes() == at::IntArrayRef({B, T, local_v_heads, kHeadDimV}),
      "v must have shape [B, T, local_v_heads, 128].");
  TORCH_CHECK(a.dim() == 3 && a.sizes() == at::IntArrayRef({B, T, local_v_heads}), "a must be [B, T, local_v_heads].");
  TORCH_CHECK(b.sizes() == a.sizes(), "b must match a shape.");
  TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == local_v_heads, "A_log must be [local_v_heads].");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == local_v_heads, "dt_bias must be [local_v_heads].");
  TORCH_CHECK(out.sizes() == v.sizes(), "out must match v shape.");

  const bool is_varlen = cu_seqlens.defined() && cu_seqlens.numel() > 0;
  const int64_t sequence_count = is_varlen ? cu_seqlens.numel() - 1 : B;
  TORCH_CHECK(sequence_count > 0, "sequence_count must be positive.");
  if (is_varlen) {
    TORCH_CHECK(B == 1, "cu_seqlens mode expects flattened q/k/v with batch size 1.");
  }

  TORCH_CHECK(
      final_state.dim() == 4 &&
          final_state.sizes() == at::IntArrayRef({sequence_count, local_v_heads, kHeadDimQK, kHeadDimV}),
      "final_state must be [N, local_v_heads, 128, 128].");
  const bool has_initial_state = initial_state.defined() && initial_state.numel() > 0;
  if (has_initial_state) {
    TORCH_CHECK(initial_state.sizes() == final_state.sizes(), "initial_state must match final_state shape.");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

#if defined(CULA_SM100_ENABLED) || defined(CULA_SM90A_ENABLED)
  // A single packed sequence is equivalent to the fixed-length B=1 layout,
  // so it can use the same chunk scheduler without reading cu_seqlens back to
  // the host.  True multi-sequence varlen remains on the recurrent fallback.
  const bool fixed_like_layout = !is_varlen || sequence_count == 1;
#ifdef CULA_SM90A_ENABLED
  const bool chunk_head_supported = local_v_heads % 4 == 0;
#else
  constexpr bool chunk_head_supported = true;
#endif
  if (q.scalar_type() == at::kBFloat16 && T >= 32 && fixed_like_layout && chunk_head_supported) {
#ifdef CULA_SM100_ENABLED
    run_qwen35_chunk_prefill_bf16(
#else
    run_qwen35_chunk_prefill_sm90_bf16(
#endif
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        final_state,
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(qk_heads),
        static_cast<int>(local_v_heads),
        has_initial_state,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif

  if (q.scalar_type() == at::kHalf) {
    dispatch_scalar_prefill<c10::Half>(
        local_v_heads,
        stream,
        q.data_ptr<c10::Half>(),
        k.data_ptr<c10::Half>(),
        v.data_ptr<c10::Half>(),
        a.data_ptr<c10::Half>(),
        b.data_ptr<c10::Half>(),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        has_initial_state ? initial_state.data_ptr<float>() : nullptr,
        is_varlen ? cu_seqlens.data_ptr<int32_t>() : nullptr,
        out.data_ptr<c10::Half>(),
        final_state.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(qk_heads),
        static_cast<int>(sequence_count),
        is_varlen,
        has_initial_state);
  } else {
    dispatch_scalar_prefill<c10::BFloat16>(
        local_v_heads,
        stream,
        q.data_ptr<c10::BFloat16>(),
        k.data_ptr<c10::BFloat16>(),
        v.data_ptr<c10::BFloat16>(),
        a.data_ptr<c10::BFloat16>(),
        b.data_ptr<c10::BFloat16>(),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        has_initial_state ? initial_state.data_ptr<float>() : nullptr,
        is_varlen ? cu_seqlens.data_ptr<int32_t>() : nullptr,
        out.data_ptr<c10::BFloat16>(),
        final_state.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(qk_heads),
        static_cast<int>(sequence_count),
        is_varlen,
        has_initial_state);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_qwen35_scalar_kda_prefill_core(ScalarKdaPrefillCoreParams& params) {
#if !defined(CULA_SM100_ENABLED) && !defined(CULA_SM90A_ENABLED)
  TORCH_CHECK(false, "Qwen scalar GDN prefill core requires an SM90 or SM100 build.");
#else
  const at::Tensor& q = params.q;
  const at::Tensor& k = params.k;
  const at::Tensor& v = params.v;
  const at::Tensor& gate_raw = params.g;
  const at::Tensor& beta_raw = params.beta;
  const at::Tensor& initial_state = params.initial_state;
  const at::Tensor& cu_seqlens = params.cu_seqlens;
  const at::Tensor& out = params.out;
  const at::Tensor& final_state = params.final_state;

  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor.");
  const at::Device device = q.device();
  check_tensor_device(k, "k", device);
  check_tensor_device(v, "v", device);
  check_tensor_device(gate_raw, "g", device);
  check_tensor_device(beta_raw, "beta", device);
  check_tensor_device(initial_state, "initial_state", device);
  check_tensor_device(cu_seqlens, "cu_seqlens", device);
  check_tensor_device(out, "out", device);
  check_tensor_device(final_state, "final_state", device);

  check_contiguous(q, "q");
  check_contiguous(k, "k");
  check_contiguous(v, "v");
  check_contiguous(gate_raw, "g");
  check_contiguous(beta_raw, "beta");
  check_contiguous(initial_state, "initial_state");
  check_contiguous(cu_seqlens, "cu_seqlens");
  check_contiguous(out, "out");
  check_contiguous(final_state, "final_state");

  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
          v.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
      "q/k/v/out must be bfloat16 for the chunk core path.");
  TORCH_CHECK(gate_raw.scalar_type() == at::kFloat, "g must be float32.");
  TORCH_CHECK(beta_raw.scalar_type() == at::kFloat, "beta must be float32.");
  TORCH_CHECK(final_state.scalar_type() == at::kFloat, "final_state must be float32.");
  TORCH_CHECK(
      !initial_state.defined() || initial_state.numel() == 0 || initial_state.scalar_type() == at::kFloat,
      "initial_state must be float32 when provided.");
  TORCH_CHECK(
      !cu_seqlens.defined() || cu_seqlens.numel() == 0 || cu_seqlens.scalar_type() == at::kInt,
      "cu_seqlens must be int32 when provided.");

  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D.");
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t qk_heads = q.size(2);
  const int64_t local_v_heads = v.size(2);
  TORCH_CHECK(T >= 32, "the chunk core path requires sequence length >= 32.");
  TORCH_CHECK(qk_heads > 0 && local_v_heads > 0, "q/k/v head counts must be positive.");
  TORCH_CHECK(local_v_heads % qk_heads == 0, "local V heads must be divisible by local Q/K heads.");
  TORCH_CHECK(
      is_supported_scalar_prefill_v_heads(static_cast<int>(local_v_heads)),
      "unsupported Qwen scalar prefill local V-head count: ", local_v_heads, ".");
  TORCH_CHECK(
      q.sizes() == at::IntArrayRef({B, T, qk_heads, kHeadDimQK}),
      "q must have shape [B, T, local_qk_heads, 128].");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape.");
  TORCH_CHECK(
      v.sizes() == at::IntArrayRef({B, T, local_v_heads, kHeadDimV}),
      "v must have shape [B, T, local_v_heads, 128].");
  TORCH_CHECK(
      gate_raw.sizes() == at::IntArrayRef({B, T, local_v_heads}),
      "g must be [B, T, local_v_heads].");
  TORCH_CHECK(beta_raw.sizes() == gate_raw.sizes(), "beta must match g shape.");
  TORCH_CHECK(out.sizes() == v.sizes(), "out must match v shape.");

  const bool is_varlen = cu_seqlens.defined() && cu_seqlens.numel() > 0;
  const int64_t sequence_count = is_varlen ? cu_seqlens.numel() - 1 : B;
  TORCH_CHECK(sequence_count > 0, "sequence_count must be positive.");
  TORCH_CHECK(!is_varlen || (B == 1 && sequence_count == 1),
              "the chunk core path supports fixed batches or one packed sequence.");
  TORCH_CHECK(
      final_state.dim() == 4 &&
          final_state.sizes() == at::IntArrayRef({sequence_count, local_v_heads, kHeadDimQK, kHeadDimV}),
      "final_state must be [N, local_v_heads, 128, 128].");
  const bool has_initial_state = initial_state.defined() && initial_state.numel() > 0;
  if (has_initial_state) {
    TORCH_CHECK(initial_state.sizes() == final_state.sizes(), "initial_state must match final_state shape.");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  const at::Tensor empty;
#ifdef CULA_SM100_ENABLED
  run_qwen35_chunk_prefill_bf16(
#else
  run_qwen35_chunk_prefill_sm90_bf16(
#endif
      q,
      k,
      v,
      empty,
      empty,
      empty,
      empty,
      initial_state,
      out,
      final_state,
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(qk_heads),
      static_cast<int>(local_v_heads),
      has_initial_state,
      stream,
      &gate_raw,
      &beta_raw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

} // namespace cula::qwen35::prefill
