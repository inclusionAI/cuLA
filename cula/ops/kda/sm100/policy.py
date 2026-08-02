# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Intracard-CP dispatch for the SM100 KDA backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

from cula.ops.kda.cp_mode import CPMode, NotSplittableError


@dataclass(frozen=True)
class CPDecision:
    enabled: bool
    reason: str = ""
    force: bool = False
    # cu_seqlens materialized on host by the auto heuristic, returned so the
    # caller reuses it instead of paying a second D2H sync.
    cu_seqlens_cpu: torch.Tensor | None = None


def _env_default() -> CPMode:
    return CPMode.AUTO if os.environ.get("CULA_INTRACARD_CP", "0") != "0" else CPMode.OFF


def sm100_intracard_cp_decision(
    *,
    mode: CPMode | bool | str | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    g: torch.Tensor | None,
    num_qk_heads: int,
    chunk_size: int,
    is_inference: bool,
    sm_count_provider: Callable[[], int],
) -> CPDecision:
    mode = CPMode.parse(mode)
    if mode is None:
        mode = _env_default()
    if mode is CPMode.OFF:
        return CPDecision(False, "disabled")

    def declined(reason: str) -> CPDecision:
        if mode is CPMode.FORCE:
            raise NotSplittableError(reason)
        return CPDecision(False, reason)

    if cu_seqlens is None:
        return declined("SM100 intracard CP requires varlen cu_seqlens.")
    if g is not None:
        return declined("SM100 intracard CP requires g is None; pass gate through gk.")
    if not is_inference:
        return declined("SM100 intracard CP is inference-only.")

    if mode is CPMode.FORCE:
        return CPDecision(True, force=True)

    from cula.ops.kda.sm100.cp.chunk_delta_h import should_use_intracard_cp

    cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
    if should_use_intracard_cp(cpu, sm_count_provider(), num_qk_heads, chunk_size):
        return CPDecision(True, cu_seqlens_cpu=cpu)
    return CPDecision(False, "SM100 intracard CP heuristic declined for this shape.", cu_seqlens_cpu=cpu)
