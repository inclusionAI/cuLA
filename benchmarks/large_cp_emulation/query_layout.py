# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Query-span layouts used to study causal-work balance independently of kernels."""

from __future__ import annotations

from dataclasses import dataclass

from cula._metadata import as_boundaries, balanced_shards
from cula.distributed.thd_planner import THDPlannerConfig, plan_thd_layout


@dataclass(frozen=True, slots=True)
class QuerySpan:
    sequence: int
    token_start: int
    token_end: int
    sequence_start: int
    sequence_end: int

    @property
    def tokens(self) -> int:
        return self.token_end - self.token_start

    @property
    def causal_offset(self) -> int:
        return self.token_start - self.sequence_start


@dataclass(frozen=True, slots=True)
class QueryLayout:
    rank: int
    world_size: int
    block_size: int
    spans: tuple[QuerySpan, ...]

    @property
    def local_tokens(self) -> int:
        return sum(span.tokens for span in self.spans)

    @property
    def local_cu_seqlens(self) -> tuple[int, ...]:
        result = [0]
        for span in self.spans:
            result.append(result[-1] + span.tokens)
        return tuple(result)

    @property
    def causal_offsets(self) -> tuple[int, ...]:
        return tuple(span.causal_offset for span in self.spans)

    @property
    def candidate_floats(self) -> int:
        return sum(_candidate_floats(span, self.block_size) for span in self.spans)


def _candidate_prefix(position: int, block_size: int) -> int:
    quotient, remainder = divmod(position, block_size)
    return block_size * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def _candidate_floats(span: QuerySpan, block_size: int) -> int:
    start = span.causal_offset
    return _candidate_prefix(start + span.tokens, block_size) - _candidate_prefix(
        start, block_size
    )


def _validate_layout_inputs(
    global_cu_seqlens: tuple[int, ...] | list[int],
    *,
    rank: int,
    world_size: int,
    block_size: int,
) -> tuple[int, ...]:
    boundaries = as_boundaries(global_cu_seqlens, name="global_cu_seqlens")
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if any((right - left) % block_size for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("every sequence length must be block aligned")
    return boundaries


def build_contiguous_query_layout(
    global_cu_seqlens: tuple[int, ...] | list[int],
    *,
    rank: int,
    world_size: int,
    block_size: int = 4,
) -> QueryLayout:
    boundaries = _validate_layout_inputs(
        global_cu_seqlens,
        rank=rank,
        world_size=world_size,
        block_size=block_size,
    )
    shard_offsets = balanced_shards(boundaries[-1], world_size)
    shard_start, shard_end = shard_offsets[rank : rank + 2]
    if shard_start % block_size or shard_end % block_size:
        raise ValueError("contiguous CP shard boundaries must be block aligned")
    spans = []
    for sequence, (sequence_start, sequence_end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        start = max(shard_start, sequence_start)
        end = min(shard_end, sequence_end)
        if start < end:
            spans.append(QuerySpan(sequence, start, end, sequence_start, sequence_end))
    return QueryLayout(rank, world_size, block_size, tuple(spans))


def build_alternating_sequence_query_layout(
    global_cu_seqlens: tuple[int, ...] | list[int],
    *,
    rank: int,
    world_size: int,
    block_size: int = 4,
    min_shard_tokens: int = 16 << 10,
) -> QueryLayout:
    """Keep short samples whole and alternate long-sequence shard direction.

    Long sequence 0 maps its ordered shards to ascending ranks, long sequence
    1 to descending ranks, and so on. Causal work from early and late regions
    therefore cancels across adjacent long samples. Whole short samples are
    greedily assigned to the rank with the fewest tokens, so they never pay a
    max-CP zigzag tax.
    """

    boundaries = _validate_layout_inputs(
        global_cu_seqlens,
        rank=rank,
        world_size=world_size,
        block_size=block_size,
    )
    if (
        not isinstance(min_shard_tokens, int)
        or isinstance(min_shard_tokens, bool)
        or min_shard_tokens < block_size
        or min_shard_tokens % block_size
    ):
        raise ValueError("min_shard_tokens must be a positive block-aligned integer")

    assignments: list[list[QuerySpan]] = [[] for _ in range(world_size)]
    assigned_tokens = [0] * world_size
    long_index = 0
    short_sequences = []
    for sequence, (sequence_start, sequence_end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        length = sequence_end - sequence_start
        if length // world_size < min_shard_tokens:
            short_sequences.append((sequence, sequence_start, sequence_end))
            continue
        block_offsets = balanced_shards(length // block_size, world_size)
        reverse = bool(long_index % 2)
        for owner in range(world_size):
            shard = world_size - 1 - owner if reverse else owner
            start = sequence_start + block_offsets[shard] * block_size
            end = sequence_start + block_offsets[shard + 1] * block_size
            assignments[owner].append(
                QuerySpan(sequence, start, end, sequence_start, sequence_end)
            )
            assigned_tokens[owner] += end - start
        long_index += 1

    for sequence, sequence_start, sequence_end in short_sequences:
        owner = min(range(world_size), key=lambda candidate: (assigned_tokens[candidate], candidate))
        assignments[owner].append(
            QuerySpan(sequence, sequence_start, sequence_end, sequence_start, sequence_end)
        )
        assigned_tokens[owner] += sequence_end - sequence_start

    spans = tuple(sorted(assignments[rank], key=lambda span: (span.sequence, span.token_start)))
    return QueryLayout(rank, world_size, block_size, spans)


def build_planned_query_layout(
    global_cu_seqlens: tuple[int, ...] | list[int],
    *,
    rank: int,
    world_size: int,
    block_size: int = 4,
) -> QueryLayout:
    """Adapt the production THD planner output to the DSA benchmark layout."""

    plan = plan_thd_layout(
        global_cu_seqlens,
        world_size=world_size,
        config=THDPlannerConfig(block_size=block_size),
    )
    spans = tuple(
        QuerySpan(
            span.sequence,
            span.token_start,
            span.token_end,
            span.sequence_start,
            span.sequence_end,
        )
        for span in plan.for_rank(rank).spans
    )
    return QueryLayout(rank, world_size, block_size, spans)


__all__ = [
    "QueryLayout",
    "QuerySpan",
    "build_alternating_sequence_query_layout",
    "build_contiguous_query_layout",
    "build_planned_query_layout",
]
