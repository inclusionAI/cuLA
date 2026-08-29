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

"""Qwen3.5 layout decode wrapper."""

from __future__ import annotations

import torch

from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG, Qwen35LinearAttentionConfig, infer_local_config

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def qwen35_layout_decode_reference(
    mixed_qkv_conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = mixed_qkv_conv.shape[0]
    local_num_v_heads = a.shape[1]
    local_key_dim, _, local_num_k_heads = infer_local_config(
        mixed_qkv_conv.shape[1],
        local_num_v_heads,
        config=config,
    )

    q_end = local_key_dim
    k_end = q_end + local_key_dim
    q_flat = mixed_qkv_conv[:, :q_end]
    k_flat = mixed_qkv_conv[:, q_end:k_end]
    v_flat = mixed_qkv_conv[:, k_end:]

    q = q_flat.view(tokens, local_num_k_heads, config.head_k_dim)
    k = k_flat.view(tokens, local_num_k_heads, config.head_k_dim)
    v = v_flat.view(tokens, local_num_v_heads, config.head_v_dim)

    repeat_factor = local_num_v_heads // local_num_k_heads
    q_rep = q.repeat_interleave(repeat_factor, dim=1).contiguous()
    k_rep = k.repeat_interleave(repeat_factor, dim=1).contiguous()
    return q_rep, k_rep, v.contiguous(), a.contiguous(), b.contiguous()


def qwen35_layout_decode(
    mixed_qkv_conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    use_cudac = (
        backend in ("auto", "cudac")
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_layout_decode")
        and mixed_qkv_conv.is_cuda
    )
    if backend == "cudac" and not use_cudac:
        raise RuntimeError("Requested backend='cudac' but qwen35_layout_decode is not available.")

    if use_cudac:
        tokens = mixed_qkv_conv.shape[0]
        local_num_v_heads = a.shape[1]
        infer_local_config(mixed_qkv_conv.shape[1], local_num_v_heads, config=config)
        q_rep = torch.empty(
            tokens,
            local_num_v_heads,
            config.head_k_dim,
            device=mixed_qkv_conv.device,
            dtype=mixed_qkv_conv.dtype,
        )
        k_rep = torch.empty_like(q_rep)
        v = torch.empty(
            tokens,
            local_num_v_heads,
            config.head_v_dim,
            device=mixed_qkv_conv.device,
            dtype=mixed_qkv_conv.dtype,
        )
        a_kernel = torch.empty_like(a)
        b_kernel = torch.empty_like(b)
        cula_cuda.qwen35_layout_decode(
            mixed_qkv_conv.contiguous(),
            a.contiguous(),
            b.contiguous(),
            q_rep,
            k_rep,
            v,
            a_kernel,
            b_kernel,
        )
        return q_rep, k_rep, v, a_kernel, b_kernel

    if backend not in ("auto", "reference"):
        raise ValueError(f"Unsupported backend={backend}")
    return qwen35_layout_decode_reference(mixed_qkv_conv, a, b, config=config)
