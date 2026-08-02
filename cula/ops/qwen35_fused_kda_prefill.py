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

"""Qwen3.5 adapter for the native-GVA fully-fused CuTe prefill core."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _resolve_fused_kda_prefill(device: torch.device | str | int | None = None):
    try:
        from cula.utils import get_kda_fused_fwd
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(f"Cannot import fused KDA selector: {exc}") from exc

    try:
        return get_kda_fused_fwd(device)
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve fused KDA prefill for device={device}: {exc}") from exc


def has_qwen35_fused_kda_prefill(device: torch.device | str | int | None = None) -> bool:
    try:
        _resolve_fused_kda_prefill(device)
    except Exception:
        return False
    return True


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[int, int, int, int, torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"q/k/v must be 4D [B,T,H,D], got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got q={tuple(q.shape)} k={tuple(k.shape)}")
    B, T, H, K = q.shape
    if v.shape[:2] != (B, T):
        raise ValueError(f"v must match q/k batch and token dimensions, got q={tuple(q.shape)} v={tuple(v.shape)}")
    HV, V = v.shape[2:]
    if K != 128 or V != 128:
        raise ValueError(f"Qwen3.5 fused prefill expects K=V=128, got K={K} V={V}")
    if HV % H != 0:
        raise ValueError(f"Qwen3.5 GVA expects HV to be divisible by H, got H={H} HV={HV}")
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    if a.shape != (B, T, HV) or b.shape != (B, T, HV):
        raise ValueError(f"a/b must be [B,T,HV], got a={tuple(a.shape)} b={tuple(b.shape)} expected={(B, T, HV)}")
    if A_log.shape != (HV,) or dt_bias.shape != (HV,):
        raise ValueError(f"A_log/dt_bias must be [HV], got A_log={tuple(A_log.shape)} dt_bias={tuple(dt_bias.shape)}")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("cu_seqlens mode expects flattened q/k/v with batch size 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError(f"cu_seqlens must be 1D int32, got {tuple(cu_seqlens.shape)} {cu_seqlens.dtype}")
    state_count = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state is not None and initial_state.shape != (state_count, HV, K, V):
        raise ValueError(f"initial_state must be [{state_count},{HV},{K},{V}], got {tuple(initial_state.shape)}")
    return B, T, HV, K, a, b


def qwen35_fused_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3.5 scalar-gated KDA prefill through the fully-fused CuTe core.

    Qwen3.5 uses native grouped value attention: Q/K have H heads while V and
    the scalar GDN gate have HV heads (globally H=16 and HV=48).  The scalar
    gate is passed to the CuTe specialization without materializing a D=128
    broadcast; Q/K likewise remain in their native, non-repeated layout.

    State is exposed in Qwen layout [N, HV, K, V]. The fused core consumes the
    transposed initial state and returns final state in Qwen layout.
    """

    if not q.is_cuda:
        raise RuntimeError("qwen35_fused_kda_prefill requires CUDA tensors.")
    B, T, HV, K, a, b = _validate_inputs(q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens)

    fused_kda_prefill = _resolve_fused_kda_prefill(q.device)

    log_gate_scalar = -torch.exp(A_log.float()).view(1, 1, HV, 1) * F.softplus(
        a.float().unsqueeze(-1) + dt_bias.float().view(1, 1, HV, 1)
    )
    log_gate = log_gate_scalar.squeeze(-1).contiguous()
    beta = torch.sigmoid(b.float()).contiguous()

    # A single [0, T] sequence is an ordinary equal-length prefill, not a
    # variable-length batch.  Keep the fast non-varlen SM100 launch in this
    # common Qwen inference case; the varlen path carries extra indirection
    # and workspace overhead.
    kernel_cu_seqlens = cu_seqlens
    if cu_seqlens is not None and q.shape[0] == 1 and cu_seqlens.numel() == 2:
        kernel_cu_seqlens = None

    initial_state_vk = None
    if initial_state is not None:
        initial_state_vk = initial_state.float().transpose(-1, -2).contiguous()

    out, final_state_vk = fused_kda_prefill(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=log_gate,
        beta=beta,
        scale=K**-0.5,
        initial_state=initial_state_vk,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        cu_seqlens=kernel_cu_seqlens,
        safe_gate=True,
        lower_bound=-5.0,
        scalar_gate=True,
    )
    final_state = None if final_state_vk is None else final_state_vk.contiguous()
    return out, final_state
