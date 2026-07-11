# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel dispatch policy for KDA wrappers."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

IntracardCPMode = Literal["auto"] | bool


@dataclass(frozen=True)
class IntracardCPDecision:
    enabled: bool
    reason: str | None = None
    force: bool = False


class NotSplittableError(ValueError):
    """Raised when intracard CP cannot meaningfully split the given shape.

    Subclasses ValueError so existing ``except ValueError`` callers keep working,
    while new code can catch it narrowly and fall back to the serial path.
    """


def normalize_intracard_cp_mode(mode: IntracardCPMode) -> IntracardCPMode:
    # Identity checks (not `in`): `1 == True` / `0 == False` would match stray ints.
    if mode != "auto" and mode is not True and mode is not False:
        raise ValueError(f'use_intracard_cp must be "auto", True, or False, got {mode!r}')
    return mode


def resolve_intracard_cp_mode(
    use_intracard_cp: IntracardCPMode | None,
    use_cp_alias: IntracardCPMode | None,
) -> IntracardCPMode | None:
    if use_intracard_cp is not None and use_cp_alias is not None:
        raise TypeError("Pass only one of use_intracard_cp or use_cp.")
    mode = use_intracard_cp if use_intracard_cp is not None else use_cp_alias
    if mode is None:
        return None
    return normalize_intracard_cp_mode(mode)


def _reject_or_disable(mode: IntracardCPMode, reason: str) -> IntracardCPDecision:
    if mode is True:
        raise ValueError(reason)
    return IntracardCPDecision(False, reason)


def _sm100_env_cp_enabled() -> bool:
    return os.environ.get("CULA_INTRACARD_CP", "0") != "0"


def sm100_intracard_cp_decision(
    *,
    mode: IntracardCPMode | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    g: torch.Tensor | None,
    num_qk_heads: int,
    chunk_size: int,
    is_inference: bool,
    sm_count_provider: Callable[[], int],
    no_cp: bool = False,
) -> IntracardCPDecision:
    if mode is None:
        mode = "auto" if _sm100_env_cp_enabled() else False
    mode = normalize_intracard_cp_mode(mode)
    # no_cp is the recursion guard: intracard_fwd_h re-invokes fwd_h with _no_cp=True
    # so sub-sequences do not recursively re-trigger CP.
    if mode is False or no_cp:
        return IntracardCPDecision(False, "disabled")

    if cu_seqlens is None:
        return _reject_or_disable(mode, "SM100 intracard CP requires varlen cu_seqlens.")
    if g is not None:
        return _reject_or_disable(mode, "SM100 intracard CP requires g is None; pass gate through gk.")
    if not is_inference:
        return _reject_or_disable(mode, "SM100 intracard CP is inference-only.")

    if mode is True:
        return IntracardCPDecision(True, force=True)

    # auto: consult the CPU-only perf heuristic.
    from cula.ops.kda.sm100.cp.chunk_delta_h import should_use_intracard_cp

    cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
    if should_use_intracard_cp(cpu, sm_count_provider(), num_qk_heads, chunk_size):
        return IntracardCPDecision(True)
    return IntracardCPDecision(False, "SM100 intracard CP heuristic declined for this shape.")
