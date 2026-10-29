# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Thin adapters for cuDNN Frontend's native THD KDA operations."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import torch


@cache
def _load_cudnn_kda() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from cudnn.linear_attention import kimi_delta_attention, kimi_delta_attention_summary
    except ImportError as exc:
        raise ImportError(
            "the cuDNN KDA backend requires the 'cudnn-cp' extra; install cuda-linear-attention[cudnn-cp]"
        ) from exc
    return kimi_delta_attention, kimi_delta_attention_summary


def _validate_thd(q, k, v, g, beta, cu_seqlens) -> None:
    tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "cu_seqlens": cu_seqlens}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g)):
        if tensor.ndim != 3:
            raise ValueError(f"{name} must use THD layout [total_tokens, heads, dim]")
    if beta.ndim != 2:
        raise ValueError("beta must use TH layout [total_tokens, heads]")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be one-dimensional")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must be int32 or int64")
    total = q.shape[0]
    if any(t.shape[0] != total for t in (k, v, g, beta)):
        raise ValueError("q, k, v, g, and beta must have the same token count")
    if any(t.device != q.device for t in (k, v, g, beta, cu_seqlens)):
        raise ValueError("all KDA tensors must be on the same device")


def cudnn_kda_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    gate_domain: str = "log",
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    batch_invariant: bool = False,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_dtype: torch.dtype | None = None,
    plan_name: str | None = None,
):
    """Run cuDNN Frontend KDA with its native packed THD contract.

    Recurrent states use cuDNN's V-first layout ``[N, HO, V, K]``.
    ``g`` is log decay unless ``safe_gate=True`` or
    ``gate_domain='linear'`` is selected.
    """
    _validate_thd(q, k, v, g, beta, cu_seqlens)
    kimi_delta_attention, _ = _load_cudnn_kda()
    checkpoint_kwargs = {}
    if checkpoint_dtype is not None:
        checkpoint_kwargs["checkpoint_dtype"] = checkpoint_dtype
    result = kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=batch_invariant,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=lower_bound,
        gate_domain=gate_domain,
        a_log=A_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        plan_name=plan_name,
        **checkpoint_kwargs,
    )
    if checkpoint_every_n_tokens > 0:
        output, final_state, checkpoints = result
        return output, final_state if output_final_state else None, checkpoints
    output, final_state = result
    return output, final_state if output_final_state else None


def cudnn_kda_summary_thd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    gate_domain: str = "log",
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    batch_invariant: bool = False,
    plan_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the zero-state span summary ``(H, M)`` for THD KDA.

    For a V-first state ``X``, the summary composes as ``X @ M + H``.
    Both outputs are float32 and non-differentiable in cuDNN Frontend.
    """
    # A summary has no q input, so reuse the common checks with k as q.
    _validate_thd(k, k, v, g, beta, cu_seqlens)
    _, kimi_delta_attention_summary = _load_cudnn_kda()
    return kimi_delta_attention_summary(
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=lower_bound,
        gate_domain=gate_domain,
        a_log=A_log,
        dt_bias=dt_bias,
        batch_invariant=batch_invariant,
        plan_name=plan_name,
    )
