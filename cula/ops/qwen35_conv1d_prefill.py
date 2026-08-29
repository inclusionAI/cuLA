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

"""Qwen3.5 depthwise causal conv1d prefill wrapper."""

from __future__ import annotations

import torch


def qwen35_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "silu",
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv1d over a full sequence.

    Expected shapes:
    - x: [B, C, S] or flattened [T, C]
    - weight: [C, 1, 4] or [C, 4]
    """

    if activation != "silu":
        raise ValueError(f"Unsupported activation={activation}")
    if weight.ndim == 3:
        if weight.shape[1] != 1:
            raise ValueError(f"weight must be [C,1,K] or [C,K], got {tuple(weight.shape)}")
        weight_2d = weight[:, 0, :]
    elif weight.ndim == 2:
        weight_2d = weight
    else:
        raise ValueError(f"weight must be [C,1,K] or [C,K], got {tuple(weight.shape)}")

    kernel_size = weight_2d.shape[1]

    def _conv_one(seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # seq: [S, C]
        if seq.ndim != 2 or seq.shape[1] != weight_2d.shape[0]:
            raise ValueError(f"sequence must be [S,C={weight_2d.shape[0]}], got {tuple(seq.shape)}")
        seq_f = seq.float()
        weight_f = weight_2d.float()
        out = torch.empty_like(seq)
        for t in range(seq.shape[0]):
            acc = torch.zeros(seq.shape[1], device=seq.device, dtype=torch.float32)
            for kk in range(kernel_size):
                src_t = t - (kernel_size - 1 - kk)
                if src_t >= 0:
                    acc = acc + seq_f[src_t] * weight_f[:, kk]
            out[t] = torch.nn.functional.silu(acc).to(seq.dtype)

        state = torch.zeros(seq.shape[1], kernel_size, device=seq.device, dtype=seq.dtype)
        take = min(kernel_size, seq.shape[0])
        if take > 0:
            state[:, kernel_size - take :] = seq[-take:].transpose(0, 1)
        return out, state

    if x.ndim == 3:
        # Public op shape follows the Qwen conv convention [B, C, S].
        if x.shape[1] != weight_2d.shape[0]:
            raise ValueError(f"x channel dim must match weight, got x={tuple(x.shape)} weight={tuple(weight_2d.shape)}")
        y = torch.empty_like(x)
        states = torch.empty(x.shape[0], x.shape[1], kernel_size, device=x.device, dtype=x.dtype)
        for bidx in range(x.shape[0]):
            y_b, state_b = _conv_one(x[bidx].transpose(0, 1).contiguous())
            y[bidx] = y_b.transpose(0, 1).contiguous()
            states[bidx] = state_b
        return (y, states) if output_final_state else y

    if x.ndim == 2:
        if cu_seqlens is None:
            y, state = _conv_one(x)
            return (y, state.unsqueeze(0)) if output_final_state else y
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError(f"cu_seqlens must be 1D int32, got {tuple(cu_seqlens.shape)} {cu_seqlens.dtype}")
        y = torch.empty_like(x)
        states = torch.empty(cu_seqlens.numel() - 1, x.shape[1], kernel_size, device=x.device, dtype=x.dtype)
        for sidx in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[sidx].item())
            end = int(cu_seqlens[sidx + 1].item())
            y_s, state_s = _conv_one(x[start:end])
            y[start:end] = y_s
            states[sidx] = state_s
        return (y, states) if output_final_state else y

    raise ValueError(f"x must be [B,C,S] or [T,C], got {tuple(x.shape)}")
