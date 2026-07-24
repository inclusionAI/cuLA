# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from numbers import Real

import torch

from cula.backends import BackendRegistry
from cula.kda.backends.flashkda import FlashKDABackend
from cula.kda.backends.fully_fused import FullyFusedBackend
from cula.ops.kda.cp_mode import CPMode

_EXTRA_KWARGS = {
    "A_log",
    "dt_bias",
    "cu_seqlens_cpu",
    "use_beta_sigmoid_in_kernel",
    "use_intracard_cp",
    "out",
    "final_state",
}

kda_prefill_registry = BackendRegistry("kda_prefill")
kda_prefill_registry.register(FlashKDABackend())
kda_prefill_registry.register(FullyFusedBackend())


@torch.compiler.disable
def kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: float | None = -5.0,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    **kwargs,
):
    """Select the first SM90 prefill backend that accepts the call.

    This forward-only API does not implement backward. FlashKDA has priority
    over the fully-fused CUDA path; disabling either backend with its
    ``CULA_BACKEND_*`` variable makes dispatch try the remaining implementation.
    """
    tensors = (q, k, v, g, beta)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("q, k, v, g, and beta must be tensors")
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q and v must have rank 4")
    if q.shape[2] <= 0 or v.shape[2] <= 0:
        raise ValueError("q/k and v/g head counts must be positive")

    unknown = set(kwargs) - _EXTRA_KWARGS
    if unknown:
        raise TypeError(f"kda_prefill() got unexpected keyword arguments: {sorted(unknown)}")
    if use_gate_in_kernel and safe_gate:
        if lower_bound is None:
            raise ValueError("lower_bound must be specified when safe_gate=True and use_gate_in_kernel=True")
        if not isinstance(lower_bound, Real):
            raise TypeError(f"lower_bound must be a real number, got {type(lower_bound).__name__}")
        if not -5 <= lower_bound < 0:
            raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound!r}")
    use_beta_sigmoid = kwargs.get("use_beta_sigmoid_in_kernel", False)
    if not isinstance(use_beta_sigmoid, bool):
        raise TypeError("use_beta_sigmoid_in_kernel must be boolean")
    if "use_intracard_cp" in kwargs:
        kwargs["use_intracard_cp"] = CPMode.parse(kwargs["use_intracard_cp"])

    return kda_prefill_registry.dispatch(
        "kda_prefill",
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        **kwargs,
    )


__all__ = ["kda_prefill", "kda_prefill_registry"]
