# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Static packed-layout plans shared by cuDNN hybrid-attention operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import torch
import torch.distributed as dist

from cula._metadata import as_boundaries, balanced_shards, validate_shards


@dataclass(frozen=True, slots=True)
class IndexPoolPlan:
    """Static THD metadata for one cuDNN index-pooling invocation.

    A plan is built from host metadata once per packed microbatch.  It records
    the compacted block axis without reading CUDA values back to the host.  The
    corresponding device tensors can then be materialized outside a captured
    or allocation-free execution region and reused by every compatible layer.

    ``coff=1`` describes non-overlapping ``ratio``-token blocks. ``coff=2``
    selects cuDNN's overlapping compressor layout; it changes the projection
    width, not the number of compacted rows.
    """

    ratio: int
    coff: int
    head_dim: int
    cu_seqlens: tuple[int, ...]
    cu_seqlens_comp: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def total_comp(self) -> int:
        return self.cu_seqlens_comp[-1]

    @property
    def num_sequences(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def input_width(self) -> int:
        return self.coff * self.head_dim

    @classmethod
    def from_cu_seqlens(
        cls,
        cu_seqlens: torch.Tensor | Sequence[int],
        *,
        head_dim: int,
        ratio: int = 4,
        coff: int = 1,
    ) -> IndexPoolPlan:
        """Build a plan without accessing CUDA-resident metadata."""

        boundaries = as_boundaries(cu_seqlens, name="cu_seqlens")
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1:
            raise ValueError("head_dim must be a positive integer")
        if not isinstance(ratio, int) or isinstance(ratio, bool) or ratio < 1:
            raise ValueError("ratio must be a positive integer")
        if coff not in (1, 2):
            raise ValueError("coff must be 1 (own-block) or 2 (overlapping)")
        if boundaries[-1] >= 2**31:
            raise ValueError("cu_seqlens values must fit int32")

        compact = [0]
        for start, end in zip(boundaries, boundaries[1:]):
            compact.append(compact[-1] + (end - start) // ratio)
        if compact[-1] == 0:
            raise ValueError("the packed input contains no complete pooling block")
        return cls(
            ratio=ratio,
            coff=coff,
            head_dim=head_dim,
            cu_seqlens=boundaries,
            cu_seqlens_comp=tuple(compact),
        )

    def materialize(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Create reusable int32 metadata tensors on ``device``.

        This method allocates by design.  Call it during plan construction or
        warmup, not inside a CUDA Graph capture or the per-layer hot path.
        """

        resolved = torch.device(device)
        cu_seqlens = torch.tensor(self.cu_seqlens, dtype=torch.int32, device=resolved)
        cu_seqlens_comp = torch.tensor(self.cu_seqlens_comp, dtype=torch.int32, device=resolved)
        return cu_seqlens, cu_seqlens_comp


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _local_boundaries(
    global_cu_seqlens: tuple[int, ...], token_start: int, token_end: int
) -> tuple[int, ...]:
    return (
        (0,)
        + tuple(value - token_start for value in global_cu_seqlens[1:-1] if token_start < value < token_end)
        + (token_end - token_start,)
    )


def _interval_pool_geometry(
    boundaries: tuple[int, ...], token_start: int, token_end: int, ratio: int
) -> tuple[int, int, int, bool]:
    """Return owned blocks, left send, right receive, and direct-pool eligibility.

    A complete block is owned by the rank containing its first token.  A rank
    that starts in the middle of such a block contributes its prefix-adjacent
    rows to the previous owner; a rank ending in the middle of an owned block
    receives the remaining rows from the next rank.
    """

    owned_blocks = 0
    send_left_tokens = 0
    recv_right_tokens = 0
    direct = True
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        local_start = max(token_start, sequence_start)
        local_end = min(token_end, sequence_end)
        if local_start >= local_end:
            continue
        sequence_length = sequence_end - sequence_start
        complete_blocks = sequence_length // ratio
        complete_end = complete_blocks * ratio
        start_offset = local_start - sequence_start
        end_offset = local_end - sequence_start

        first_owned = _ceil_div(start_offset, ratio)
        last_owned = min(_ceil_div(end_offset, ratio), complete_blocks)
        owned_blocks += max(last_owned - first_owned, 0)

        if start_offset % ratio and start_offset < complete_end:
            direct = False
            block_end = min(_ceil_div(start_offset, ratio) * ratio, complete_end)
            send_left_tokens = max(send_left_tokens, max(block_end - start_offset, 0))
        # A sequence tail is intentionally dropped.  It needs no halo and does
        # not make direct local pooling invalid when the rank owns the tail.
        if local_end < sequence_end and end_offset % ratio and end_offset < complete_end:
            direct = False
            owned_block_end = min(_ceil_div(end_offset, ratio) * ratio, complete_end)
            recv_right_tokens = max(recv_right_tokens, max(owned_block_end - end_offset, 0))
    return owned_blocks, send_left_tokens, recv_right_tokens, direct


@dataclass(frozen=True, slots=True)
class IndexPoolShardPlan:
    """Distributed ratio-block ownership and halo metadata for one CP rank.

    The plan assigns each complete compressed block to the rank that owns its
    first token.  Aligned shards can call cuDNN pooling directly and gather the
    compact rows.  Misaligned shards expose the bounded seam exchange needed
    by a future/optional halo stitcher, avoiding a full raw-token all-gather.
    """

    rank: int
    world_size: int
    token_start: int
    token_end: int
    ratio: int
    coff: int
    head_dim: int
    local_cu_seqlens: tuple[int, ...]
    block_offsets: tuple[int, ...]
    send_left_tokens: int
    recv_right_tokens: int
    direct_local_pool: bool
    all_shards_direct: bool
    halo_supported: bool
    local_plan: IndexPoolPlan | None

    @property
    def local_tokens(self) -> int:
        return self.token_end - self.token_start

    @property
    def local_blocks(self) -> int:
        return self.block_offsets[self.rank + 1] - self.block_offsets[self.rank]

    @property
    def global_blocks(self) -> int:
        return self.block_offsets[-1]

    @property
    def block_counts(self) -> tuple[int, ...]:
        return tuple(
            right - left for left, right in zip(self.block_offsets, self.block_offsets[1:])
        )

    @property
    def equal_block_counts(self) -> bool:
        counts = self.block_counts
        return all(count == counts[0] for count in counts[1:])

    def validate_group(self, group: dist.ProcessGroup | None) -> None:
        """Check that the runtime process group matches this rank-local plan."""

        if not dist.is_available() or not dist.is_initialized():
            if group is not None or self.world_size != 1 or self.rank != 0:
                raise RuntimeError(
                    "the index-pooling shard plan requires an initialized "
                    "torch.distributed process group"
                )
            return
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if (world_size, rank) != (self.world_size, self.rank):
            raise ValueError(
                "IndexPoolShardPlan process-group mismatch: "
                f"plan has rank/world_size={self.rank}/{self.world_size}, "
                f"runtime has {rank}/{world_size}"
            )

    @classmethod
    def from_global_cu_seqlens(
        cls,
        global_cu_seqlens: torch.Tensor | Sequence[int],
        *,
        rank: int,
        world_size: int,
        head_dim: int,
        ratio: int = 4,
        coff: int = 1,
        shard_offsets: Sequence[int] | None = None,
    ) -> IndexPoolShardPlan:
        """Build a static distributed plan from global host metadata."""

        boundaries = as_boundaries(global_cu_seqlens, name="global_cu_seqlens")
        # Reuse IndexPoolPlan's option validation and compacted global layout.
        IndexPoolPlan.from_cu_seqlens(boundaries, head_dim=head_dim, ratio=ratio, coff=coff)
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size})")
        offsets = (
            balanced_shards(boundaries[-1], world_size)
            if shard_offsets is None
            else validate_shards(shard_offsets, total_tokens=boundaries[-1], world_size=world_size)
        )

        counts = []
        direct_by_rank = []
        geometry = []
        for shard_rank in range(world_size):
            values = _interval_pool_geometry(
                boundaries,
                offsets[shard_rank],
                offsets[shard_rank + 1],
                ratio,
            )
            geometry.append(values)
            counts.append(values[0])
            direct_by_rank.append(values[3])
        block_offsets = [0]
        for count in counts:
            block_offsets.append(block_offsets[-1] + count)

        start, end = offsets[rank], offsets[rank + 1]
        local_cu = _local_boundaries(boundaries, start, end)
        owned_blocks, send_left, recv_right, direct = geometry[rank]
        local_plan = None
        if direct and owned_blocks:
            local_plan = IndexPoolPlan.from_cu_seqlens(
                local_cu,
                head_dim=head_dim,
                ratio=ratio,
                coff=coff,
            )
            if local_plan.total_comp != owned_blocks:
                raise RuntimeError("internal error while deriving the direct local index-pooling plan")

        # With at least ``ratio`` tokens per shard, one compressed block can
        # cross at most one rank seam, so a single neighbor exchange suffices.
        halo_supported = min(right - left for left, right in zip(offsets, offsets[1:])) >= ratio
        return cls(
            rank=index(rank),
            world_size=index(world_size),
            token_start=start,
            token_end=end,
            ratio=ratio,
            coff=coff,
            head_dim=head_dim,
            local_cu_seqlens=local_cu,
            block_offsets=tuple(block_offsets),
            send_left_tokens=send_left,
            recv_right_tokens=recv_right,
            direct_local_pool=direct,
            all_shards_direct=all(direct_by_rank),
            halo_supported=halo_supported,
            local_plan=local_plan,
        )


__all__ = ["IndexPoolPlan", "IndexPoolShardPlan"]
