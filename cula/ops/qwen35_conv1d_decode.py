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

"""Qwen3.5 single-token conv-state update wrapper."""

from __future__ import annotations

import torch

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def qwen35_conv1d_decode_reference(
    x_t: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure torch reference for Qwen3.5 single-token depthwise conv decode."""
    if weight.ndim == 3:
        weight = weight.squeeze(1)

    state_tail = conv_state[..., 1:].to(torch.float32)
    x_last = x_t.unsqueeze(-1).to(torch.float32)
    window = torch.cat([state_tail, x_last], dim=-1)
    conv = (window * weight.to(torch.float32).unsqueeze(0)).sum(dim=-1)
    y = torch.nn.functional.silu(conv).to(dtype=x_t.dtype)

    conv_state_out = conv_state.clone()
    conv_state_out[..., 0] = conv_state[..., 1]
    conv_state_out[..., 1] = conv_state[..., 2]
    conv_state_out[..., 2] = conv_state[..., 3]
    conv_state_out[..., 3] = x_t
    return y, conv_state_out


def qwen35_conv1d_decode_update(
    x_t: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "silu",
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token depthwise causal conv1d update.

    Expected shapes:
    - x_t: [B, C]
    - conv_state: [B, C, 4]
    - weight: [C, 1, 4] or [C, 4]
    """

    if activation != "silu":
        raise ValueError(f"Only silu activation is currently supported, got {activation}")
    if x_t.ndim != 2:
        raise ValueError(f"x_t must be 2D [batch, channels], got {tuple(x_t.shape)}")
    if conv_state.ndim != 3:
        raise ValueError(f"conv_state must be 3D [batch, channels, kernel], got {tuple(conv_state.shape)}")
    if conv_state.shape[:2] != x_t.shape:
        raise ValueError(f"conv_state batch/channel dims must match x_t, got x_t={tuple(x_t.shape)} conv_state={tuple(conv_state.shape)}")
    kernel_size = conv_state.shape[-1]
    if kernel_size != 4:
        raise ValueError(f"Expected kernel_size=4 for Qwen3.5, got {kernel_size}")

    if weight.ndim == 3:
        if weight.shape[1] != 1 or weight.shape[2] != kernel_size:
            raise ValueError(f"weight must be [channels,1,{kernel_size}], got {tuple(weight.shape)}")
        weight_2d = weight.squeeze(1)
    elif weight.ndim == 2:
        if weight.shape[1] != kernel_size:
            raise ValueError(f"weight must be [channels,{kernel_size}], got {tuple(weight.shape)}")
        weight_2d = weight
    else:
        raise ValueError(f"weight must be 2D or 3D, got {tuple(weight.shape)}")

    if weight_2d.shape[0] != x_t.shape[1]:
        raise ValueError(f"weight channels must match x_t channels, got weight={tuple(weight_2d.shape)} x_t={tuple(x_t.shape)}")

    x_t = x_t.contiguous()
    conv_state = conv_state.contiguous()
    weight_2d = weight_2d.contiguous()

    use_cudac = (
        backend in ("auto", "cudac")
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_conv1d_decode")
        and x_t.is_cuda
    )
    if backend == "cudac" and not use_cudac:
        raise RuntimeError("Requested backend='cudac' but qwen35_conv1d_decode is not available.")

    if use_cudac:
        mixed_qkv_3d = x_t.unsqueeze(1).contiguous()
        out_3d = torch.empty_like(mixed_qkv_3d)
        conv_state_out = conv_state.clone()
        cula_cuda.qwen35_conv1d_decode(
            mixed_qkv_3d,
            conv_state_out,
            weight_2d,
            out_3d,
        )
        return out_3d.squeeze(1), conv_state_out

    if backend not in ("auto", "reference"):
        raise ValueError(f"Unsupported backend={backend}")
    return qwen35_conv1d_decode_reference(x_t, conv_state, weight_2d)
