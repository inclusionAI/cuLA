# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.large_cp_emulation.query_layout import (
    build_alternating_sequence_query_layout,
    build_contiguous_query_layout,
)


def _workload_boundaries():
    lengths = [1024] * 20 + [514048] * 2
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def test_alternating_sequence_layout_keeps_short_samples_whole_and_tokens_balanced():
    boundaries = _workload_boundaries()
    layouts = [
        build_alternating_sequence_query_layout(boundaries, rank=rank, world_size=4)
        for rank in range(4)
    ]

    assert {layout.local_tokens for layout in layouts} == {1 << 18}
    assert all(len(layout.spans) == 7 for layout in layouts)
    assert all(sum(span.sequence < 20 for span in layout.spans) == 5 for layout in layouts)
    assert all(sum(span.sequence >= 20 for span in layout.spans) == 2 for layout in layouts)

    covered = sorted(
        (span.sequence, span.token_start, span.token_end)
        for layout in layouts
        for span in layout.spans
    )
    for sequence, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        sequence_spans = [span for span in covered if span[0] == sequence]
        assert sequence_spans[0][1] == start
        assert sequence_spans[-1][2] == end
        assert all(left[2] == right[1] for left, right in zip(sequence_spans, sequence_spans[1:]))


def test_alternating_sequence_layout_balances_causal_candidates_better_than_contiguous():
    boundaries = _workload_boundaries()
    contiguous = [
        build_contiguous_query_layout(boundaries, rank=rank, world_size=4)
        for rank in range(4)
    ]
    alternating = [
        build_alternating_sequence_query_layout(boundaries, rank=rank, world_size=4)
        for rank in range(4)
    ]

    contiguous_costs = [layout.candidate_floats for layout in contiguous]
    alternating_costs = [layout.candidate_floats for layout in alternating]
    assert max(contiguous_costs) / min(contiguous_costs) > 2.5
    assert max(alternating_costs) / min(alternating_costs) < 1.01
