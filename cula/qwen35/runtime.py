# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime dispatch for Qwen3.5 linear-attention kernels."""

from __future__ import annotations

import torch

try:
    import cuda.bindings.driver as cuda
except ImportError:  # pragma: no cover - optional runtime dependency
    cuda = None

from cula.qwen35.common import (
    DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    Qwen35LinearAttentionConfig,
    infer_local_config,
    validate_mixed_qkv,
    validate_scalar_gate_inputs,
    validate_state_tensors,
)
from cula.ops.qwen35_conv1d_decode import qwen35_conv1d_decode_update
from cula.ops.qwen35_conv1d_prefill import qwen35_conv1d_prefill
from cula.ops.qwen35_layout_decode import qwen35_layout_decode
from cula.ops.qwen35_layout_prefill import qwen35_layout_prefill
from cula.ops.qwen35_scalar_kda_decode import (
    has_qwen35_layout_scalar_kda_decode_cudac,
    qwen35_layout_scalar_kda_decode,
    qwen35_scalar_kda_decode,
)
from cula.ops.qwen35_scalar_kda_prefill import qwen35_scalar_kda_prefill

_stream_cache: dict[tuple[str, int], object] = {}


def _get_cached_stream(device: torch.device) -> object:
    if cuda is None:
        raise RuntimeError("cuda.bindings.driver is not available in this environment.")
    stream_id = int(torch.cuda.current_stream(device=device).cuda_stream)
    cache_key = (str(device), stream_id)
    if cache_key not in _stream_cache:
        _stream_cache[cache_key] = cuda.CUstream(stream_id)
    return _stream_cache[cache_key]


def _torch_qwen35_scalar_kda_decode_reference(
    q_rep: torch.Tensor,
    k_rep: torch.Tensor,
    v: torch.Tensor,
    a_kernel: torch.Tensor,
    b_kernel: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure torch reference for Qwen3.5 scalar-gated decode."""
    tokens, num_v_heads, head_k_dim = q_rep.shape
    head_v_dim = v.shape[-1]
    state_out = recurrent_state.clone()
    out = torch.empty(tokens, num_v_heads, head_v_dim, device=q_rep.device, dtype=q_rep.dtype)

    scale = q_rep.shape[-1] ** -0.5
    q_f = torch.nn.functional.normalize(q_rep.float(), dim=-1) * scale
    k_f = torch.nn.functional.normalize(k_rep.float(), dim=-1)
    v_f = v.float()
    a_f = a_kernel.float()
    b_f = b_kernel.float()

    for token_idx in range(tokens):
        pool_idx = int(state_indices[token_idx].item())
        for hv in range(num_v_heads):
            state_kv = state_out[pool_idx, hv]
            state_vk = state_kv.transpose(0, 1).contiguous()

            decay_pre = a_f[token_idx, hv] + dt_bias[hv]
            decay = torch.exp(-torch.exp(A_log[hv]) * torch.nn.functional.softplus(decay_pre))
            beta = torch.sigmoid(b_f[token_idx, hv])

            k_vec = k_f[token_idx, hv]
            q_vec = q_f[token_idx, hv]

            proj = decay * (state_vk @ k_vec)
            v_new = beta * (v_f[token_idx, hv] - proj)
            state_vk_new = decay * state_vk + v_new.unsqueeze(1) * k_vec.unsqueeze(0)
            out[token_idx, hv] = (state_vk_new @ q_vec).to(out.dtype)
            state_out[pool_idx, hv] = state_vk_new.transpose(0, 1).contiguous()

    return out, state_out


def qwen35_linear_attention_decode_reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure torch reference for the full Qwen3.5 decode chain."""
    tokens = mixed_qkv.shape[0]
    if state_indices is None:
        state_indices = torch.arange(tokens, device=mixed_qkv.device, dtype=torch.int32)
    else:
        state_indices = state_indices.to(device=mixed_qkv.device, dtype=torch.int32)

    conv_out, conv_state_out = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        activation="silu",
        backend="reference",
    )
    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_prefill(
        conv_out,
        a,
        b,
        config=config,
        backend="reference",
    )
    core_attn_out, recurrent_state_out = _torch_qwen35_scalar_kda_decode_reference(
        q_rep,
        k_rep,
        v,
        a_kernel,
        b_kernel,
        A_log.float(),
        dt_bias.float(),
        recurrent_state.float(),
        state_indices,
    )
    return core_attn_out.reshape(tokens, -1), conv_state_out, recurrent_state_out


def qwen35_linear_attention_prefill(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    cu_seqlens: torch.Tensor | None = None,
    recurrent_state: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Qwen3.5 prefill wrapper.

    Args:
        mixed_qkv: flattened [tokens, local_conv_dim]
        a, b: [tokens, local_num_v_heads]
        conv_weight: [local_conv_dim, 1, 4] or [local_conv_dim, 4]
        A_log, dt_bias: [local_num_v_heads]
        cu_seqlens: optional int32 sequence offsets for flattened input
        recurrent_state: optional initial recurrent state [num_sequences, HV, 128, 128]

    Returns:
        - core_attn_out_flat: [tokens, local_value_dim]
        - final conv_state: [num_sequences, local_conv_dim, 4]
        - final recurrent_state: [num_sequences, local_num_v_heads, 128, 128]
    """

    validate_mixed_qkv(mixed_qkv, config)
    validate_scalar_gate_inputs(a, b, config)
    validate_state_tensors(conv_state, recurrent_state, config)
    if mixed_qkv.is_cuda:
        _get_cached_stream(mixed_qkv.device)
    if conv_state is not None:
        raise NotImplementedError("Qwen3.5 prefill with non-empty conv_state is not implemented yet.")
    if mixed_qkv.shape[0] != a.shape[0]:
        raise ValueError(f"Token dimension mismatch, got mixed_qkv={tuple(mixed_qkv.shape)} a={tuple(a.shape)}")
    if A_log.ndim != 1 or dt_bias.ndim != 1 or A_log.shape != dt_bias.shape:
        raise ValueError(f"A_log and dt_bias must be matching 1D tensors, got {tuple(A_log.shape)} and {tuple(dt_bias.shape)}")
    if cu_seqlens is not None and (cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32):
        raise ValueError(f"cu_seqlens must be 1D int32, got {tuple(cu_seqlens.shape)} {cu_seqlens.dtype}")

    tokens = mixed_qkv.shape[0]
    local_num_v_heads = a.shape[1]
    _, local_value_dim, _ = infer_local_config(
        mixed_qkv.shape[1],
        local_num_v_heads,
        config=config,
    )
    if A_log.numel() != local_num_v_heads:
        raise ValueError(f"A_log must match local_num_v_heads={local_num_v_heads}, got {A_log.numel()}")

    conv_out, conv_state_out = qwen35_conv1d_prefill(
        mixed_qkv,
        conv_weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_prefill(
        conv_out,
        a,
        b,
        config=config,
        backend="reference" if backend == "reference" else "auto",
    )
    core_attn_out, recurrent_state_out = qwen35_scalar_kda_prefill(
        q=q_rep.unsqueeze(0).contiguous(),
        k=k_rep.unsqueeze(0).contiguous(),
        v=v.unsqueeze(0).contiguous(),
        a=a_kernel.unsqueeze(0).contiguous(),
        b=b_kernel.unsqueeze(0).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=recurrent_state,
        cu_seqlens=cu_seqlens,
        backend=backend,
    )
    return core_attn_out.reshape(tokens, local_value_dim), conv_state_out, recurrent_state_out


def qwen35_linear_attention_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen3.5 decode wrapper.

    Args:
        mixed_qkv: [tokens, local_conv_dim]
        a, b: [tokens, local_num_v_heads]
        conv_weight: [local_conv_dim, 1, 4] or [local_conv_dim, 4]
        A_log, dt_bias: [local_num_v_heads]
        conv_state: [tokens, local_conv_dim, 4]
        recurrent_state: [pool, local_num_v_heads, 128, 128]

    Returns:
        - core_attn_out_flat: [tokens, local_value_dim]
        - updated_conv_state
        - updated_recurrent_state
    """

    validate_mixed_qkv(mixed_qkv, config)
    validate_scalar_gate_inputs(a, b, config)
    validate_state_tensors(conv_state, recurrent_state, config)
    if mixed_qkv.is_cuda:
        _get_cached_stream(mixed_qkv.device)

    if mixed_qkv.shape[0] != a.shape[0]:
        raise ValueError(f"Token dimension mismatch, got mixed_qkv={tuple(mixed_qkv.shape)} a={tuple(a.shape)}")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError(f"A_log and dt_bias must be 1D, got {tuple(A_log.shape)} and {tuple(dt_bias.shape)}")
    if A_log.shape != dt_bias.shape:
        raise ValueError(f"A_log and dt_bias must have the same shape, got {tuple(A_log.shape)} vs {tuple(dt_bias.shape)}")

    tokens = mixed_qkv.shape[0]
    local_num_v_heads = a.shape[1]
    local_key_dim, local_value_dim, local_num_k_heads = infer_local_config(
        mixed_qkv.shape[1],
        local_num_v_heads,
        config=config,
    )
    if conv_state.shape != (tokens, mixed_qkv.shape[1], config.conv_kernel_size):
        raise ValueError(
            f"conv_state must be [tokens, local_conv_dim, {config.conv_kernel_size}], got {tuple(conv_state.shape)}"
        )
    if recurrent_state.shape[1:] != (local_num_v_heads, config.head_k_dim, config.head_v_dim):
        raise ValueError(
            "recurrent_state must be [pool, local_num_v_heads, head_k_dim, head_v_dim], "
            f"got {tuple(recurrent_state.shape)}"
        )
    if A_log.numel() != local_num_v_heads:
        raise ValueError(f"A_log must match local_num_v_heads={local_num_v_heads}, got {A_log.numel()}")

    if backend == "auto" and not mixed_qkv.is_cuda:
        backend = "reference"

    if backend == "reference":
        return qwen35_linear_attention_decode_reference(
            mixed_qkv,
            a,
            b,
            conv_weight,
            A_log,
            dt_bias,
            config=config,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            state_indices=state_indices,
        )

    conv_out, conv_state_out = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        activation="silu",
        backend=backend,
    )
    use_fused_layout_kda = (
        backend in ("auto", "cudac")
        and mixed_qkv.is_cuda
        and has_qwen35_layout_scalar_kda_decode_cudac()
    )
    if backend == "cudac" and not use_fused_layout_kda:
        raise RuntimeError("Requested backend='cudac' but qwen35_layout_scalar_kda_decode is not available.")

    if use_fused_layout_kda:
        core_attn_out, recurrent_state_out = qwen35_layout_scalar_kda_decode(
            mixed_qkv_conv=conv_out,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            recurrent_state=recurrent_state,
            state_indices=state_indices,
            config=config,
            backend=backend,
        )
        core_attn_out = core_attn_out.reshape(tokens, local_value_dim)
        return core_attn_out, conv_state_out, recurrent_state_out

    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_decode(
        conv_out,
        a,
        b,
        config=config,
        backend=backend,
    )
    q = q_rep.unsqueeze(1).contiguous()
    k = k_rep.unsqueeze(1).contiguous()
    v = v.unsqueeze(1).contiguous()

    core_attn_out, recurrent_state_out = qwen35_scalar_kda_decode(
        q=q,
        k=k,
        v=v,
        a=a_kernel,
        b=b_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend=backend,
    )
    core_attn_out = core_attn_out.reshape(tokens, local_value_dim)
    return core_attn_out, conv_state_out, recurrent_state_out
