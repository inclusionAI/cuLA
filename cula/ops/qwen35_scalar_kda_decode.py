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

"""CuTe DSL placeholder for Qwen3.5 scalar-gated KDA decode."""

from __future__ import annotations

import torch

from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG, Qwen35LinearAttentionConfig

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def has_qwen35_layout_scalar_kda_decode_cudac() -> bool:
    return cula_cuda is not None and hasattr(cula_cuda, "qwen35_layout_scalar_kda_decode")


def _validate_cudac_state_indices(state_indices: torch.Tensor, *, rows: int, pool_size: int) -> None:
    if state_indices.ndim != 1 or state_indices.numel() != rows:
        raise ValueError(f"state_indices must be 1D with {rows} entries, got {tuple(state_indices.shape)}")
    if rows == 0:
        return
    min_idx = int(state_indices.min().item())
    max_idx = int(state_indices.max().item())
    if min_idx < 0 or max_idx >= pool_size:
        raise ValueError(f"state_indices must be in [0, {pool_size}), got min={min_idx} max={max_idx}")
    if torch.unique(state_indices).numel() != rows:
        raise ValueError(
            "backend='cudac' requires unique state_indices within one decode launch; "
            "duplicate rows need a sequential decode path."
        )


def qwen35_scalar_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    state_indices: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token scalar-gated delta-rule decode for Qwen3.5."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be 4D, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got q={tuple(q.shape)} vs k={tuple(k.shape)}")
    if q.shape[1] != 1 or v.shape[1] != 1:
        raise ValueError(f"Decode expects single-token sequence dim, got q={tuple(q.shape)} v={tuple(v.shape)}")

    N, _, HV, K = q.shape
    if a.ndim == 2:
        a = a.unsqueeze(1)
    if b.ndim == 2:
        b = b.unsqueeze(1)
    if a.shape != (N, 1, HV) or b.shape != (N, 1, HV):
        raise ValueError(f"a/b must be [N,1,HV], got a={tuple(a.shape)} b={tuple(b.shape)}")
    if A_log.shape != (HV,) or dt_bias.shape != (HV,):
        raise ValueError(f"A_log/dt_bias must be [HV], got A_log={tuple(A_log.shape)} dt_bias={tuple(dt_bias.shape)}")

    state_indices = (
        torch.arange(N, device=q.device, dtype=torch.int32)
        if state_indices is None
        else state_indices.to(device=q.device, dtype=torch.int32)
    )

    use_cudac = (
        backend in ("auto", "cudac")
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_scalar_kda_decode")
        and q.is_cuda
    )
    if backend == "cudac" and not use_cudac:
        raise RuntimeError("Requested backend='cudac' but qwen35_scalar_kda_decode is not available.")

    if use_cudac:
        _validate_cudac_state_indices(state_indices, rows=N, pool_size=recurrent_state.shape[0])
        q_rep = q.squeeze(1).contiguous()
        k_rep = k.squeeze(1).contiguous()
        v_rep = v.squeeze(1).contiguous()
        a_kernel = a.squeeze(1).contiguous()
        b_kernel = b.squeeze(1).contiguous()
        out = torch.empty_like(v_rep)
        recurrent_state_out = recurrent_state.clone()
        cula_cuda.qwen35_scalar_kda_decode(
            q_rep,
            k_rep,
            v_rep,
            a_kernel,
            b_kernel,
            A_log.contiguous(),
            dt_bias.contiguous(),
            recurrent_state_out,
            state_indices,
            out,
        )
        return out.unsqueeze(1), recurrent_state_out

    if backend not in ("auto", "generic_kda"):
        raise ValueError(f"Unsupported backend={backend}")

    from cula.ops.kda_decode import kda_decode

    a_expanded = a.unsqueeze(-1).expand(N, 1, HV, K)
    dt_bias_expanded = dt_bias[:, None].expand(HV, K).contiguous()
    o = kda_decode(
        A_log=A_log.contiguous(),
        dt_bias=dt_bias_expanded,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        a=a_expanded.contiguous(),
        b=b.contiguous(),
        initial_state_source=recurrent_state,
        initial_state_indices=state_indices,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        state_layout="kv",
    )
    return o, recurrent_state


def qwen35_layout_scalar_kda_decode(
    mixed_qkv_conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    state_indices: torch.Tensor | None = None,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Qwen3.5 layout decode + scalar-gated KDA decode."""
    if mixed_qkv_conv.ndim != 2:
        raise ValueError(f"mixed_qkv_conv must be 2D, got {tuple(mixed_qkv_conv.shape)}")
    if a.ndim == 3:
        if a.shape[1] != 1:
            raise ValueError(f"a sequence dim must be 1 for decode, got {tuple(a.shape)}")
        a = a.squeeze(1)
    if b.ndim == 3:
        if b.shape[1] != 1:
            raise ValueError(f"b sequence dim must be 1 for decode, got {tuple(b.shape)}")
        b = b.squeeze(1)

    N = mixed_qkv_conv.shape[0]
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != N:
        raise ValueError(f"a/b must be [N, HV], got a={tuple(a.shape)} b={tuple(b.shape)}")
    HV = a.shape[1]
    if A_log.shape != (HV,) or dt_bias.shape != (HV,):
        raise ValueError(f"A_log/dt_bias must be [HV], got A_log={tuple(A_log.shape)} dt_bias={tuple(dt_bias.shape)}")

    state_indices = (
        torch.arange(N, device=mixed_qkv_conv.device, dtype=torch.int32)
        if state_indices is None
        else state_indices.to(device=mixed_qkv_conv.device, dtype=torch.int32)
    )

    use_cudac = (
        backend in ("auto", "cudac")
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_layout_scalar_kda_decode")
        and mixed_qkv_conv.is_cuda
    )
    if backend == "cudac" and not use_cudac:
        raise RuntimeError("Requested backend='cudac' but qwen35_layout_scalar_kda_decode is not available.")

    if use_cudac:
        _validate_cudac_state_indices(state_indices, rows=N, pool_size=recurrent_state.shape[0])
        out = torch.empty(
            N,
            HV,
            recurrent_state.shape[-1],
            device=mixed_qkv_conv.device,
            dtype=mixed_qkv_conv.dtype,
        )
        recurrent_state_out = recurrent_state.clone()
        cula_cuda.qwen35_layout_scalar_kda_decode(
            mixed_qkv_conv.contiguous(),
            a.contiguous(),
            b.contiguous(),
            A_log.contiguous(),
            dt_bias.contiguous(),
            recurrent_state_out,
            state_indices,
            out,
        )
        return out.unsqueeze(1), recurrent_state_out

    if backend not in ("auto", "generic_kda"):
        raise ValueError(f"Unsupported backend={backend}")

    from cula.ops.qwen35_layout_decode import qwen35_layout_decode_reference

    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_decode_reference(mixed_qkv_conv, a, b, config=config)
    return qwen35_scalar_kda_decode(
        q=q_rep.unsqueeze(1).contiguous(),
        k=k_rep.unsqueeze(1).contiguous(),
        v=v.unsqueeze(1).contiguous(),
        a=a_kernel,
        b=b_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="generic_kda",
    )
