# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Architecture dispatch for the public Lightning Attention prefill API."""

from __future__ import annotations

import functools
import importlib

import torch

SM100_FIXED_BACKEND_IDENTITY = "cula.lightning.sm100.cutedsl.prefill.fixed"
SM100_VARLEN_NONPERSISTENT_BACKEND_IDENTITY = "cula.lightning.sm100.cutedsl.prefill.varlen.nonpersistent"
SM100_VARLEN_PERSISTENT_BACKEND_IDENTITY = "cula.lightning.sm100.cutedsl.prefill.varlen.persistent"


def _cuda_device(value: torch.Tensor | torch.device | str | int) -> torch.device:
    if isinstance(value, torch.Tensor):
        if not value.is_cuda:
            raise ValueError("Lightning Attention prefill requires CUDA tensors")
        return value.device
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("Lightning Attention prefill requires a CUDA device")
    return device


@functools.cache
def _device_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _backend_name(value: torch.Tensor | torch.device | str | int) -> str:
    device = _cuda_device(value)
    capability = _device_capability(device)
    if capability == (9, 0):
        return "sm90"
    if capability in {(10, 0), (10, 3)}:
        return "sm100"
    raise RuntimeError(
        "Lightning Attention prefill supports SM90, SM100, and SM103; "
        f"found compute capability sm_{capability[0]}{capability[1]}"
    )


def _backend_module(value: torch.Tensor | torch.device | str | int):
    backend = _backend_name(value)
    module_name = "cula.ops.lightning.prefill_sm90" if backend == "sm90" else "cula.ops.lightning.prefill_sm100"
    return backend, importlib.import_module(module_name)


def get_lightning_attn_prefill_backend_identity(
    device: torch.Tensor | torch.device | str | int,
    *,
    varlen: bool = False,
    persistent: bool = True,
) -> str:
    """Resolve the exact backend identity from a CUDA tensor or device."""

    if not isinstance(varlen, bool) or not isinstance(persistent, bool):
        raise TypeError("varlen and persistent must be boolean")
    backend, module = _backend_module(device)
    if backend == "sm90":
        return module.get_sm90_lightning_attn_prefill_backend_identity(
            varlen=varlen,
            persistent=persistent,
        )
    if not varlen:
        return SM100_FIXED_BACKEND_IDENTITY
    return SM100_VARLEN_PERSISTENT_BACKEND_IDENTITY if persistent else SM100_VARLEN_NONPERSISTENT_BACKEND_IDENTITY


def lightning_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    decay: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch fixed-length Lightning Attention by ``Q.device``."""

    _, module = _backend_module(Q)
    return module.lightning_attn_fwd(
        Q,
        K,
        V,
        decay,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )


def lightning_attn_fwd_varlen(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    decay: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float = 1.0,
    state_pool: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    persistent: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch packed-varlen Lightning Attention by ``Q.device``."""

    _, module = _backend_module(Q)
    return module.lightning_attn_fwd_varlen(
        Q,
        K,
        V,
        decay,
        cu_seqlens,
        scale=scale,
        state_pool=state_pool,
        initial_state_indices=initial_state_indices,
        chunk_size=chunk_size,
        persistent=persistent,
    )


__all__ = [
    "get_lightning_attn_prefill_backend_identity",
    "lightning_attn_fwd",
    "lightning_attn_fwd_varlen",
]
