# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Packed-THD shard planning for Kimi context parallelism.

The planner is deliberately CPU-only.  A framework supplies global sequence
boundaries and the contiguous token partition; the resulting immutable plan is
reusable by every KDA layer in the microbatch.  No CUDA tensor is inspected, so
building a plan never introduces a device-to-host synchronization.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import torch
import torch.distributed as dist


def _as_boundaries(values: torch.Tensor | Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor; copying CUDA metadata would synchronize the device")
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if values.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must have int32 or int64 dtype")
        result = tuple(int(value) for value in values.tolist())
    else:
        try:
            result = tuple(index(value) for value in values)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be an integer sequence") from error
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least a start and an end")
    if result[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _balanced_shards(total_tokens: int, world_size: int) -> tuple[int, ...]:
    if total_tokens < world_size:
        raise ValueError("packed Kimi CP requires at least one token on every rank")
    return tuple(index * total_tokens // world_size for index in range(world_size + 1))


def _validate_shards(shard_offsets: Sequence[int], *, total_tokens: int, world_size: int) -> tuple[int, ...]:
    try:
        offsets = tuple(index(value) for value in shard_offsets)
    except (TypeError, ValueError) as error:
        raise TypeError("shard_offsets must be an integer sequence") from error
    if len(offsets) != world_size + 1:
        raise ValueError(f"shard_offsets must contain world_size + 1 ({world_size + 1}) entries")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError("shard_offsets must cover the complete packed token range")
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("shard_offsets must be strictly increasing; empty CP shards are unsupported")
    return offsets


@dataclass(frozen=True, slots=True)
class KimiCPPlan:
    """Immutable metadata for one contiguous packed-THD CP shard.

    Only the first and last local sequences can cross rank boundaries.  The
    plan records their token spans and the ordered rank chains that contribute
    incoming forward state or outgoing backward state.  Tensor shapes and
    process-group identity are checked again at execution time.
    """

    rank: int
    world_size: int
    token_start: int
    token_end: int
    first_sequence: int
    last_sequence: int
    local_cu_seqlens: tuple[int, ...]
    first_boundary_end: int
    last_boundary_start: int
    pre_num_ranks: int
    post_num_ranks: int

    @property
    def local_tokens(self) -> int:
        return self.token_end - self.token_start

    @property
    def num_local_sequences(self) -> int:
        return len(self.local_cu_seqlens) - 1

    @property
    def first_boundary_tokens(self) -> int:
        return self.first_boundary_end

    @property
    def last_boundary_tokens(self) -> int:
        return self.local_tokens - self.last_boundary_start

    def validate_group(self, group: dist.ProcessGroup | None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            if group is not None or self.world_size != 1 or self.rank != 0:
                raise RuntimeError("the Kimi CP plan requires an initialized torch.distributed process group")
            return
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if (world_size, rank) != (self.world_size, self.rank):
            raise ValueError(
                "KimiCPPlan process-group mismatch: "
                f"plan has rank/world_size={self.rank}/{self.world_size}, "
                f"runtime has {rank}/{world_size}"
            )

    def validate_local_cu_seqlens(self, cu_seqlens: torch.Tensor, *, device: torch.device) -> None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be one-dimensional")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must have int32 or int64 dtype")
        if cu_seqlens.device != device:
            raise ValueError("cu_seqlens and KDA operands must be on the same device")
        if cu_seqlens.numel() != len(self.local_cu_seqlens):
            raise ValueError(
                "cu_seqlens does not match KimiCPPlan: "
                f"expected {len(self.local_cu_seqlens)} entries, got {cu_seqlens.numel()}"
            )
        if cu_seqlens.device.type == "cpu" and tuple(int(value) for value in cu_seqlens.tolist()) != self.local_cu_seqlens:
            raise ValueError("CPU cu_seqlens values do not match KimiCPPlan")

    @classmethod
    def from_global_cu_seqlens(
        cls,
        global_cu_seqlens: torch.Tensor | Sequence[int],
        *,
        rank: int,
        world_size: int,
        shard_offsets: Sequence[int] | None = None,
    ) -> KimiCPPlan:
        """Build a plan for a contiguous rank-ordered token partition.

        ``shard_offsets`` defaults to a balanced partition of the total packed
        tokens.  Frameworks with a different contiguous partition can pass the
        exact global token offsets for all ranks.
        """

        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size})")
        boundaries = _as_boundaries(global_cu_seqlens, name="global_cu_seqlens")
        total_tokens = boundaries[-1]
        offsets = (
            _balanced_shards(total_tokens, world_size)
            if shard_offsets is None
            else _validate_shards(shard_offsets, total_tokens=total_tokens, world_size=world_size)
        )
        start, end = offsets[rank], offsets[rank + 1]

        first_sequence = bisect_right(boundaries, start) - 1
        last_sequence = bisect_right(boundaries, end - 1) - 1
        local_boundaries = (0,) + tuple(value - start for value in boundaries[1:-1] if start < value < end) + (end - start,)

        first_sequence_rank = bisect_right(offsets, boundaries[first_sequence]) - 1
        last_sequence_rank = bisect_right(offsets, boundaries[last_sequence + 1] - 1) - 1
        pre_num_ranks = rank - first_sequence_rank
        post_num_ranks = last_sequence_rank - rank
        if pre_num_ranks < 0 or post_num_ranks < 0:
            raise RuntimeError("internal error while deriving Kimi CP boundary rank ranges")

        return cls(
            rank=rank,
            world_size=world_size,
            token_start=start,
            token_end=end,
            first_sequence=first_sequence,
            last_sequence=last_sequence,
            local_cu_seqlens=local_boundaries,
            first_boundary_end=local_boundaries[1],
            last_boundary_start=local_boundaries[-2],
            pre_num_ranks=pre_num_ranks,
            post_num_ranks=post_num_ranks,
        )


def build_kimi_cp_plan(
    global_cu_seqlens: torch.Tensor | Sequence[int],
    *,
    group: dist.ProcessGroup | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    shard_offsets: Sequence[int] | None = None,
) -> KimiCPPlan:
    """Build :class:`KimiCPPlan`, deriving rank identity from ``group`` by default."""

    if (rank is None) != (world_size is None):
        raise ValueError("rank and world_size must be supplied together")
    if rank is None:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank(group)
            world_size = dist.get_world_size(group)
        elif group is None:
            rank, world_size = 0, 1
        else:
            raise RuntimeError("a process group was supplied but torch.distributed is not initialized")
    return KimiCPPlan.from_global_cu_seqlens(
        global_cu_seqlens,
        rank=rank,
        world_size=world_size,
        shard_offsets=shard_offsets,
    )


__all__ = ["KimiCPPlan", "build_kimi_cp_plan"]
