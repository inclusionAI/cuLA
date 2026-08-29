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

"""Shared constants and validation helpers for Qwen3.5 linear attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Qwen35LinearAttentionConfig:
    """Minimal runtime config for Qwen3.5 linear-attention kernels."""

    hidden_size: int = 5120
    conv_kernel_size: int = 4
    num_k_heads: int = 16
    num_v_heads: int = 48
    head_k_dim: int = 128
    head_v_dim: int = 128
    qkv_dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    @property
    def qk_repeat_factor(self) -> int:
        assert self.num_v_heads % self.num_k_heads == 0
        return self.num_v_heads // self.num_k_heads


DEFAULT_QWEN35_LINEAR_ATTN_CONFIG = Qwen35LinearAttentionConfig()


def validate_mixed_qkv(
    mixed_qkv: torch.Tensor,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
) -> None:
    if mixed_qkv.dtype != config.qkv_dtype:
        raise TypeError(f"mixed_qkv must be {config.qkv_dtype}, got {mixed_qkv.dtype}")
    if mixed_qkv.ndim != 2:
        raise ValueError(f"mixed_qkv must be 2D [tokens, conv_dim_local], got {tuple(mixed_qkv.shape)}")
    if mixed_qkv.shape[-1] <= 0:
        raise ValueError("mixed_qkv must have a non-zero channel dimension")
    if mixed_qkv.shape[-1] % config.conv_dim != 0 and mixed_qkv.shape[-1] != config.conv_dim:
        # In TP mode this is expected to be a local shard, so only require alignment
        # with the Qwen3.5 packed layout ratio.
        local_dim = mixed_qkv.shape[-1]
        expected_splits = (config.key_dim, config.key_dim, config.value_dim)
        if local_dim % sum(expected_splits) != 0:
            raise ValueError(f"mixed_qkv last dim must match packed local conv dim, got {local_dim}")


def validate_scalar_gate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
) -> None:
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got a={tuple(a.shape)} vs b={tuple(b.shape)}")
    if a.ndim != 2:
        raise ValueError(f"a and b must be 2D [tokens, num_v_heads_local], got {tuple(a.shape)}")
    if a.dtype != config.qkv_dtype or b.dtype != config.qkv_dtype:
        raise TypeError(f"a and b must be {config.qkv_dtype}, got a={a.dtype}, b={b.dtype}")


def validate_state_tensors(
    conv_state: torch.Tensor | None,
    recurrent_state: torch.Tensor | None,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
) -> None:
    if conv_state is not None:
        if conv_state.ndim != 3:
            raise ValueError(f"conv_state must be 3D [batch, channels, {config.conv_kernel_size}], got {tuple(conv_state.shape)}")
        if conv_state.dtype != config.qkv_dtype:
            raise TypeError(f"conv_state must be {config.qkv_dtype}, got {conv_state.dtype}")
    if recurrent_state is not None:
        if recurrent_state.ndim != 4:
            raise ValueError(f"recurrent_state must be 4D [batch, hv, k, v], got {tuple(recurrent_state.shape)}")
        if recurrent_state.dtype != config.state_dtype:
            raise TypeError(f"recurrent_state must be {config.state_dtype}, got {recurrent_state.dtype}")


def infer_local_config(
    mixed_qkv_dim: int,
    local_num_v_heads: int,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
) -> tuple[int, int, int]:
    """Infer local packed dims from runtime shard sizes.

    Returns:
    - local_key_dim
    - local_value_dim
    - local_num_k_heads
    """

    local_value_dim = local_num_v_heads * config.head_v_dim
    remaining = mixed_qkv_dim - local_value_dim
    if remaining <= 0 or remaining % 2 != 0:
        raise ValueError(
            f"Cannot infer local q/k dims from mixed_qkv_dim={mixed_qkv_dim}, local_num_v_heads={local_num_v_heads}"
        )
    local_key_dim = remaining // 2
    if local_key_dim % config.head_k_dim != 0:
        raise ValueError(f"Local key dim must be divisible by head_k_dim={config.head_k_dim}, got {local_key_dim}")
    local_num_k_heads = local_key_dim // config.head_k_dim
    if local_num_v_heads % local_num_k_heads != 0:
        raise ValueError(
            f"Local num_v_heads={local_num_v_heads} must be divisible by local num_k_heads={local_num_k_heads}"
        )
    return local_key_dim, local_value_dim, local_num_k_heads
