# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Logical-to-physical topology mapping for large-CP emulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from cula.distributed import KimiCPPlan


def representative_logical_rank(physical_rank: int, physical_world_size: int, logical_cp: int) -> int:
    """Spread physical processes over the logical CP rank interval."""
    if physical_world_size < 1:
        raise ValueError("physical_world_size must be positive")
    if logical_cp < physical_world_size:
        raise ValueError("logical_cp must be at least physical_world_size")
    if not 0 <= physical_rank < physical_world_size:
        raise ValueError("physical_rank must be in [0, physical_world_size)")
    if physical_world_size == 1:
        return logical_cp - 1
    return physical_rank * (logical_cp - 1) // (physical_world_size - 1)


def sample_chain_hops(global_cu_seqlens: Sequence[int], local_tokens: int) -> int:
    """Return the longest packed sample's number of rank-to-rank edges."""
    boundaries = tuple(global_cu_seqlens)
    if local_tokens < 1:
        raise ValueError("local_tokens must be positive")
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError("global_cu_seqlens must contain at least two entries and start at zero")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("global_cu_seqlens must be strictly increasing")
    max_ranks = 1
    for left, right in zip(boundaries, boundaries[1:]):
        first_rank = left // local_tokens
        last_rank = (right - 1) // local_tokens
        max_ranks = max(max_ranks, last_rank - first_rank + 1)
    return max_ranks - 1


@dataclass(frozen=True, slots=True)
class LogicalCPEmulationPlan:
    """Keep logical shard semantics separate from the physical process group.

    ``logical_plan`` describes the representative CP64 shard. ``runtime_plan``
    preserves that shard's THD boundary metadata while carrying the physical
    rank/world size required by cuLA's native process-group validation.
    """

    logical_plan: KimiCPPlan
    runtime_plan: KimiCPPlan
    logical_cp: int
    logical_rank: int
    physical_rank: int
    physical_world_size: int
    transport_hops: int
    transport_mode: str

    @classmethod
    def build(
        cls,
        global_cu_seqlens: Sequence[int],
        *,
        logical_cp: int,
        physical_rank: int,
        physical_world_size: int,
        transport_mode: str = "full-ring",
    ) -> LogicalCPEmulationPlan:
        boundaries = tuple(global_cu_seqlens)
        if logical_cp < 1:
            raise ValueError("logical_cp must be positive")
        if len(boundaries) < 2 or boundaries[0] != 0:
            raise ValueError("global_cu_seqlens must contain at least two entries and start at zero")
        if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("global_cu_seqlens must be strictly increasing")
        if boundaries[-1] % logical_cp:
            raise ValueError("global token count must be divisible by logical_cp")
        logical_rank = representative_logical_rank(physical_rank, physical_world_size, logical_cp)
        logical_plan = KimiCPPlan.from_global_cu_seqlens(
            boundaries,
            rank=logical_rank,
            world_size=logical_cp,
        )
        if transport_mode == "full-ring":
            hops = logical_cp - 1
        elif transport_mode == "sample-chain":
            hops = sample_chain_hops(boundaries, logical_plan.local_tokens)
        else:
            raise ValueError("transport_mode must be 'full-ring' or 'sample-chain'")
        runtime_plan = replace(
            logical_plan,
            rank=physical_rank,
            world_size=physical_world_size,
            token_start=physical_rank * logical_plan.local_tokens,
            token_end=(physical_rank + 1) * logical_plan.local_tokens,
        )
        return cls(
            logical_plan=logical_plan,
            runtime_plan=runtime_plan,
            logical_cp=logical_cp,
            logical_rank=logical_rank,
            physical_rank=physical_rank,
            physical_world_size=physical_world_size,
            transport_hops=hops,
            transport_mode=transport_mode,
        )
