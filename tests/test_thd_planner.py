# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import random
from collections import defaultdict

import pytest

from cula.distributed.thd_planner import THDPlannerConfig, plan_thd_layout


def _boundaries(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _assert_exact_cover(plan, boundaries):
    by_sequence = defaultdict(list)
    for rank_layout in plan.ranks:
        assert rank_layout.local_tokens > 0
        for span in rank_layout.spans:
            assert span.owner == rank_layout.rank
            by_sequence[span.sequence].append(span)
    for sequence, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        spans = sorted(by_sequence[sequence], key=lambda span: span.token_start)
        assert spans[0].token_start == start
        assert spans[-1].token_end == end
        assert all(left.token_end == right.token_start for left, right in zip(spans, spans[1:]))


@pytest.mark.parametrize(
    "lengths",
    [
        [1024] * 1024,
        [1024] * 20 + [514048] * 2,
        [1 << 20],
        [1024] * 256 + [786432],
        [4096] * 32 + [131072, 262144, 524288],
        [1024, 4096, 16384, 65536, 131072, 307200, 523264],
    ],
    ids=("all-short", "two-long", "single-long", "long-tail", "multi-long", "power-law"),
)
def test_planner_handles_representative_thd_distributions(lengths):
    boundaries = _boundaries(lengths)
    plan = plan_thd_layout(boundaries, world_size=4)

    _assert_exact_cover(plan, boundaries)
    scores = dict(plan.evaluated_scores)
    assert plan.metrics.score <= scores["contiguous"]
    assert max(plan.metrics.token_loads) - min(plan.metrics.token_loads) <= plan.block_size


def test_single_long_sequence_forms_balanced_multi_span_layout():
    plan = plan_thd_layout([0, 1 << 20], world_size=4)

    assert plan.policy == "cost-balanced-2x"
    assert plan.metrics.token_imbalance == pytest.approx(1.0)
    assert plan.metrics.causal_imbalance < 1.02
    assert all(1 <= len(rank.spans) <= 2 for rank in plan.ranks)


def test_short_sequences_are_not_fragmented_by_cost_balanced_candidate():
    lengths = [1024] * 20 + [514048] * 2
    boundaries = _boundaries(lengths)
    plan = plan_thd_layout(boundaries, world_size=4)

    assert plan.policy.startswith("cost-balanced-")
    short_counts = defaultdict(int)
    for rank in plan.ranks:
        for span in rank.spans:
            if span.sequence < 20:
                short_counts[span.sequence] += 1
    assert set(short_counts.values()) == {1}


def test_boundary_penalty_can_select_contiguous_fallback():
    config = THDPlannerConfig(boundary_weight=100.0)
    plan = plan_thd_layout([0, 1 << 20], world_size=4, config=config)

    assert plan.policy == "contiguous"


def test_cp64_long_sequence_uses_scaled_zigzag_chunks():
    plan = plan_thd_layout([0, 1 << 20], world_size=64)

    assert plan.policy == "cost-balanced-2x"
    assert plan.metrics.token_imbalance == pytest.approx(1.0)
    assert plan.metrics.causal_imbalance == pytest.approx(1.0)
    assert all(1 <= len(rank.spans) <= 2 for rank in plan.ranks)


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ([0, 3, 8], "block aligned"),
        ([0, 8], "at least one block per rank"),
    ],
)
def test_planner_rejects_invalid_partitions(boundaries, message):
    with pytest.raises(ValueError, match=message):
        plan_thd_layout(boundaries, world_size=8)


def test_random_thd_plans_preserve_coverage_and_never_lose_to_contiguous():
    generator = random.Random(20261104)
    for _ in range(100):
        sequence_count = generator.randint(1, 128)
        lengths = [4 * generator.randint(1, 1 << 15) for _ in range(sequence_count)]
        boundaries = _boundaries(lengths)
        plan = plan_thd_layout(boundaries, world_size=4)

        _assert_exact_cover(plan, boundaries)
        assert plan.metrics.score <= dict(plan.evaluated_scores)["contiguous"]
        assert max(plan.metrics.token_loads) - min(plan.metrics.token_loads) <= plan.block_size
