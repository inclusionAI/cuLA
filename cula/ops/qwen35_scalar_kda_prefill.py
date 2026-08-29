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

"""Qwen3.5 scalar-gated KDA prefill wrapper."""

from __future__ import annotations

import torch

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def qwen35_scalar_kda_prefill(
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
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked scalar-gated delta-rule prefill for Qwen3.5."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be 4D, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.shape != k.shape:
        raise ValueError(f"q/k must have the same shape, got q={tuple(q.shape)} k={tuple(k.shape)}")
    B, T, H, K = q.shape
    HV = v.shape[2]
    if K != 128 or v.shape[-1] != 128 or v.shape[:2] != q.shape[:2]:
        raise ValueError(f"Qwen3.5 prefill expects q/k=[B,T,H,128], v=[B,T,HV,128], got q={tuple(q.shape)} v={tuple(v.shape)}")
    if H <= 0 or HV <= 0 or HV % H:
        raise ValueError(f"local V heads must be divisible by local Q/K heads, got H={H} HV={HV}")
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
    if initial_state is not None and initial_state.shape[1:] != (HV, K, K):
        raise ValueError(f"initial_state must be [N,HV,128,128], got {tuple(initial_state.shape)}")

    use_cudac = (
        backend in ("auto", "cudac")
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_scalar_kda_prefill")
        and q.is_cuda
    )
    if backend == "cudac" and not use_cudac:
        raise RuntimeError("Requested backend='cudac' but qwen35_scalar_kda_prefill is not available.")

    if use_cudac:
        supported_hv = (64, 48, 32, 24, 16, 12, 8, 6, 4, 2)
        if HV not in supported_hv:
            raise ValueError(f"backend='cudac' supports Qwen local HV in {supported_hv}, got {HV}")
        state_count = B if cu_seqlens is None else cu_seqlens.numel() - 1
        out = torch.empty_like(v)
        final_state = torch.empty(state_count, HV, K, K, device=q.device, dtype=torch.float32)
        initial_state_arg = (
            torch.empty(0, device=q.device, dtype=torch.float32)
            if initial_state is None
            else initial_state.contiguous()
        )
        cu_seqlens_arg = (
            torch.empty(0, device=q.device, dtype=torch.int32)
            if cu_seqlens is None
            else cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        )
        cula_cuda.qwen35_scalar_kda_prefill(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            a.contiguous(),
            b.contiguous(),
            A_log.contiguous(),
            dt_bias.contiguous(),
            initial_state_arg,
            cu_seqlens_arg,
            out,
            final_state,
        )
        return out, final_state

    if backend not in ("auto", "reference"):
        raise ValueError(f"Unsupported backend={backend}")

    state_count = B if cu_seqlens is None else cu_seqlens.numel() - 1
    state = (
        torch.zeros(state_count, HV, K, K, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty_like(v)
    q_f = torch.nn.functional.normalize(q.float(), dim=-1) * (K**-0.5)
    k_f = torch.nn.functional.normalize(k.float(), dim=-1)
    v_f = v.float()
    a_f = a.float()
    b_f = b.float()
    A_log_f = A_log.float()
    dt_bias_f = dt_bias.float()

    def _run_sequence(batch_idx: int, state_idx: int, start: int, end: int) -> None:
        repeat = HV // H
        for t in range(start, end):
            for hv in range(HV):
                qk_h = hv // repeat
                state_kv = state[state_idx, hv]
                decay = torch.exp(-torch.exp(A_log_f[hv]) * torch.nn.functional.softplus(a_f[batch_idx, t, hv] + dt_bias_f[hv]))
                beta = torch.sigmoid(b_f[batch_idx, t, hv])
                k_vec = k_f[batch_idx, t, qk_h]
                q_vec = q_f[batch_idx, t, qk_h]
                proj = decay * (state_kv.transpose(0, 1) @ k_vec)
                v_new = beta * (v_f[batch_idx, t, hv] - proj)
                state_kv_new = decay * state_kv + k_vec.unsqueeze(1) * v_new.unsqueeze(0)
                out[batch_idx, t, hv] = (state_kv_new.transpose(0, 1) @ q_vec).to(out.dtype)
                state[state_idx, hv] = state_kv_new

    if cu_seqlens is None:
        for bidx in range(B):
            _run_sequence(bidx, bidx, 0, T)
    else:
        for sidx in range(state_count):
            _run_sequence(0, sidx, int(cu_seqlens[sidx].item()), int(cu_seqlens[sidx + 1].item()))
    return out, state


def qwen35_scalar_kda_prefill_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Qwen GDN calculation after scalar gate/beta preprocessing.

    ``g`` is the natural-log per-token gate before the chunk-local prefix
    scan, matching the tensors passed to SGLang's ``TritonGDNKernel.extend``.
    The CUDA core still performs Q/K normalization and the prefix scan, while
    the raw ``A_log/a/b/dt_bias`` conversion is intentionally outside timing.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape != k.shape:
        raise ValueError("q/k/v must be 4D and q/k must have identical shapes")
    B, T, H, K = q.shape
    HV = v.shape[2]
    if K != 128 or v.shape[:2] != q.shape[:2] or v.shape[-1] != 128 or HV % H:
        raise ValueError(f"invalid native-GVA shapes q={tuple(q.shape)} v={tuple(v.shape)}")
    if g.ndim == 2:
        g = g.unsqueeze(0)
    if beta.ndim == 2:
        beta = beta.unsqueeze(0)
    if g.shape != (B, T, HV) or beta.shape != g.shape:
        raise ValueError(f"g/beta must be [B,T,HV], got {tuple(g.shape)} {tuple(beta.shape)}")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("g and beta must be float32")
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError("cu_seqlens must be 1D int32 with B=1")
    if initial_state is not None and initial_state.shape[1:] != (HV, K, K):
        raise ValueError(f"initial_state must be [N,HV,128,128], got {tuple(initial_state.shape)}")

    use_cudac = (
        backend in ("auto", "cudac")
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_scalar_kda_prefill_core")
        and q.is_cuda
    )
    if backend == "cudac" and not use_cudac:
        raise RuntimeError("Requested backend='cudac' but qwen35_scalar_kda_prefill_core is unavailable")
    if not use_cudac:
        raise ValueError("qwen35_scalar_kda_prefill_core currently requires the CUDA backend")

    supported_hv = (64, 48, 32, 24, 16, 12, 8, 6, 4, 2)
    if HV not in supported_hv:
        raise ValueError(f"backend='cudac' supports local HV in {supported_hv}, got {HV}")
    state_count = B if cu_seqlens is None else cu_seqlens.numel() - 1
    out = torch.empty_like(v)
    final_state = torch.empty(state_count, HV, K, K, device=q.device, dtype=torch.float32)
    initial_state_arg = (
        torch.empty(0, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.contiguous()
    )
    cu_seqlens_arg = (
        torch.empty(0, device=q.device, dtype=torch.int32)
        if cu_seqlens is None
        else cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    )
    cula_cuda.qwen35_scalar_kda_prefill_core(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state_arg,
        cu_seqlens_arg,
        out,
        final_state,
    )
    return out, final_state
