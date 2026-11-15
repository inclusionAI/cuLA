# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import random
from collections import defaultdict
from itertools import permutations, product

import pytest

from cula.distributed.thd_planner import (
    THDPlannerConfig,
    THDRankLayout,
    THDSelectionCostModel,
    THDSpan,
    _contiguous_items,
    _localize_spans,
    _minimum_assignment,
    _transfer_lower_bound,
    plan_thd_layout,
    plan_thd_selection,
)


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


def _cost_model(**overrides):
    values = dict(
        candidate_ns=0.1,
        query_ns=1.0,
        span_us=1.0,
        bytes_per_query=20608,
        intra_node_gbps=200.0,
        inter_node_gbps=20.0,
        packing_ns_per_query=20.0,
        rank_node_ids=(0, 0, 1, 1),
        exchange_us=10.0,
    )
    return THDSelectionCostModel(**(values | overrides))


def test_calibrated_model_rejects_migration_when_network_cost_dominates():
    lengths = [1 << 20]
    cheap = plan_thd_layout(_boundaries(lengths), world_size=4, cost_model=_cost_model())
    expensive = plan_thd_layout(
        _boundaries(lengths),
        world_size=4,
        cost_model=_cost_model(intra_node_gbps=0.001, inter_node_gbps=0.001),
    )
    assert cheap.policy != "contiguous"
    assert expensive.policy == "contiguous"
    assert cheap.score_unit == expensive.score_unit == "ms"
    _assert_exact_cover(cheap, _boundaries(lengths))
    assert cheap.metrics.score < dict(cheap.evaluated_scores)["contiguous"]


def test_identity_cost_contains_no_transport_and_margin_can_keep_it():
    boundaries = _boundaries([1024] * 1024)
    plan = plan_thd_layout(boundaries, world_size=4, cost_model=_cost_model())
    assert plan.policy == "contiguous"
    # One document's exact complete-block candidate count: sum(floor(i/4)).
    candidates = sum(i // 4 for i in range(1, 1025)) * 256
    expected = candidates * 0.1 / 1e6 + 262144 / 1e6 + 256 / 1e3
    assert plan.metrics.score == pytest.approx(expected)
    assert len(plan.evaluated_scores) == 1  # Proven lower bound prunes all migrations.
    conservative = plan_thd_layout([0, 1 << 20], world_size=4, cost_model=_cost_model(min_gain_fraction=0.99))
    assert conservative.policy == "contiguous"


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_ns": float("nan")},
        {"inter_node_gbps": 0},
        {"packing_ns_per_query": -1},
        {"min_gain_fraction": 1},
        {"bytes_per_query": True},
        {"rank_node_ids": (0, False)},
        {"peer_gbps": 0},
        {"peer_gbps": float("inf")},
        {"peer_gbps": True},
        {"min_gain_ms": -1},
        {"min_gain_ms": float("nan")},
        {"min_gain_ms": True},
    ],
)
def test_cost_model_rejects_invalid_profiles(overrides):
    with pytest.raises((ValueError, TypeError)):
        _cost_model(**overrides)


def test_cost_model_requires_matching_group_topology():
    with pytest.raises(ValueError, match="node map"):
        plan_thd_layout([0, 65536], world_size=8, cost_model=_cost_model())


@pytest.mark.parametrize("selection", [False, True])
def test_absolute_saving_budget_prunes_search_without_changing_relative_margin(selection):
    kwargs = {"max_tokens_per_rank": 524288} if selection else {}
    planner = plan_thd_selection if selection else plan_thd_layout
    reusable = planner([0, 1 << 20], world_size=4, cost_model=_cost_model(), **kwargs)
    assert reusable.policy != "contiguous"
    # Above the entire baseline cost, no possible redistribution can repay
    # the caller's preparation budget. No candidate search is necessary.
    one_use = planner([0, 1 << 20], world_size=4, cost_model=_cost_model(min_gain_ms=10000), **kwargs)
    assert one_use.policy == "contiguous"
    assert len(one_use.evaluated_scores) == 1
    savings = dict(reusable.evaluated_scores)["contiguous"] - reusable.metrics.score
    rejected = planner([0, 1 << 20], world_size=4, cost_model=_cost_model(min_gain_ms=savings + 1), **kwargs)
    assert rejected.policy == "contiguous"


def test_rank_matching_equals_exhaustive_optimum():
    generator = random.Random(20261113)
    for size in range(1, 7):
        for _ in range(8):
            costs = [[generator.random() * 100 for _ in range(size)] for _ in range(size)]
            assignment = _minimum_assignment(costs)
            actual = sum(costs[i][j] for i, j in enumerate(assignment))
            expected = min(sum(costs[i][j] for i, j in enumerate(p)) for p in permutations(range(size)))
            assert sorted(assignment) == list(range(size))
            assert actual == pytest.approx(expected)
    assert _minimum_assignment([[0.0] * 4 for _ in range(4)]) == (0, 1, 2, 3)


@pytest.mark.parametrize("total", [65536, 65540])
def test_localized_bins_preserve_capacity_and_do_not_worsen_chosen_score(total):
    plan = plan_thd_layout([0, total], world_size=4, cost_model=_cost_model(min_gain_fraction=0.0))
    _assert_exact_cover(plan, [0, total])
    assert max(plan.metrics.token_loads) - min(plan.metrics.token_loads) <= 4
    scores = dict(plan.evaluated_scores)
    assert plan.metrics.score <= min(value for name, value in scores.items() if not name.endswith("-local"))
    original = plan_thd_layout([0, total], world_size=4)
    localized = _localize_spans([s for r in original.ranks for s in r.spans], 4, 4, _cost_model())
    loads = [sum(s.tokens for s in localized if s.owner == r) for r in range(4)]
    assert tuple(loads) == original.metrics.token_loads


@pytest.mark.parametrize("capacity_factor", [1, 1.25, 2])
def test_selection_planner_preserves_cover_and_explicit_capacity(capacity_factor):
    rng = random.Random(62813)
    for _ in range(25):
        lengths = [4 * rng.randint(1, 12000) for _ in range(rng.randint(1, 40))]
        cu = _boundaries(lengths)
        capacity = 4 * int(((cu[-1] // 4 + 3) // 4) * capacity_factor)
        plan = plan_thd_selection(cu, world_size=4, max_tokens_per_rank=capacity, cost_model=_cost_model())
        # Empty target ranks are legal. Exact coverage is checked independently
        # of scheduling order and must survive inverse redistribution.
        spans = sorted((s.token_start, s.token_end) for r in plan.ranks for s in r.spans)
        assert spans[0][0] == 0 and spans[-1][1] == cu[-1]
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        for r in plan.ranks:
            assert r.local_tokens <= capacity
            for s in r.spans:
                assert s.owner == r.rank
                assert (s.sequence_start, s.sequence_end) == tuple(cu[s.sequence : s.sequence + 2])
                assert s.sequence_start <= s.token_start < s.token_end <= s.sequence_end
        assert plan.metrics.score <= dict(plan.evaluated_scores)["contiguous"]
        assert plan == plan_thd_selection(cu, world_size=4, max_tokens_per_rank=capacity, cost_model=_cost_model())


def test_selection_keeps_short_documents_in_source_ownership():
    cu = _boundaries([64] * 100 + [524288, 517888])
    plan = plan_thd_selection(cu, world_size=4, max_tokens_per_rank=524288, cost_model=_cost_model())
    assert plan.policy != "contiguous"
    for layout in plan.ranks:
        for span in layout.spans:
            if span.sequence < 100:
                assert span.owner == 0
                assert span.tokens == 64


@pytest.mark.parametrize("capacity,error", [(True, TypeError), (0, ValueError), (15, ValueError), (12, ValueError)])
def test_selection_rejects_invalid_row_capacity(capacity, error):
    with pytest.raises(error, match="max_tokens_per_rank"):
        plan_thd_selection([0, 64], world_size=4, max_tokens_per_rank=capacity, cost_model=_cost_model())


@pytest.mark.parametrize("lengths", [[32], [4, 12, 16], [4] * 8])
@pytest.mark.parametrize("bandwidth", [0.01, 1, 10000])
def test_transfer_lower_bound_never_exceeds_exhaustive_partition_optimum(lengths, bandwidth):
    # Enumerate every two-rank assignment of eight blocks, including empty
    # ranks and unequal counts. Coalesce before scoring to avoid counting
    # artificial fragment costs as necessary work.
    cu = tuple(_boundaries(lengths))
    model = _cost_model(rank_node_ids=(0, 1), intra_node_gbps=bandwidth, inter_node_gbps=bandwidth)
    source = _contiguous_items(cu, 2, 4)
    ranks = tuple(
        THDRankLayout(
            r,
            tuple(
                THDSpan(i.sequence, i.token_start, i.token_end, i.sequence_start, i.sequence_end, r)
                for i, owner in source
                if owner == r
            ),
        )
        for r in range(2)
    )
    total_work = sum(sum(i // 4 for i in range(1, length + 1)) for length in lengths)
    mean = (model.candidate_ns * total_work / 1e6 + model.query_ns * 32 / 1e6 + model.span_us * len(lengths) / 1e3) / 2
    bound = _transfer_lower_bound(source, 2, 4, model, mean, model.estimate_ms(ranks, block_size=4))
    optimum = float("inf")
    for owners in product(range(2), repeat=8):
        spans = []
        for seq, (start, end) in enumerate(zip(cu, cu[1:])):
            a = start
            for z in range(start + 4, end + 1, 4):
                if z == end or owners[z // 4] != owners[a // 4]:
                    spans.append(THDSpan(seq, a, z, start, end, owners[a // 4]))
                    a = z
        candidate = tuple(THDRankLayout(r, tuple(s for s in spans if s.owner == r)) for r in range(2))
        optimum = min(optimum, model.estimate_ms(candidate, block_size=4))
    assert bound <= optimum + 1e-12


def test_peer_limit_prices_aggregate_edge_rows_not_individual_fragments():
    from benchmarks.bench_thd_transport import transport_layout

    model = _cost_model(
        candidate_ns=0,
        query_ns=0,
        span_us=0,
        packing_ns_per_query=0,
        exchange_us=0,
        peer_gbps=10,
        intra_node_gbps=1000,
        inter_node_gbps=1000,
    )
    permutation = transport_layout(4, 128, 2, "inter")
    fragmented = tuple(
        THDRankLayout(
            r.rank,
            tuple(
                THDSpan(s.sequence, a, a + 64, s.sequence_start, s.sequence_end, s.owner)
                for s in r.spans
                for a in range(s.token_start, s.token_end, 64)
            ),
        )
        for r in permutation
    )
    expected = 128 * model.bytes_per_query / (10 * 1e6)
    assert model.estimate_ms(permutation, block_size=4) == pytest.approx(expected)
    assert model.estimate_ms(fragmented, block_size=4) == pytest.approx(expected)
    striped = transport_layout(4, 128, 2, "striped")
    assert model.estimate_ms(striped, block_size=4) == pytest.approx(expected / 4)


@pytest.mark.parametrize("world", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("peer_rate", [0.01, 12.0, 500.0])
def test_selection_peer_model_remains_bounded_on_unseen_topologies(world, peer_rate):
    rng = random.Random(73381 + world)
    lengths = [4 * rng.randint(1, 256) for _ in range(127)] + [131072, 262144]
    rng.shuffle(lengths)
    cu = _boundaries(lengths)
    capacity = 8 * ((cu[-1] // 4 + world - 1) // world)
    model = _cost_model(rank_node_ids=tuple(r // 8 for r in range(world)), peer_gbps=peer_rate)
    plan = plan_thd_selection(cu, world_size=world, max_tokens_per_rank=capacity, cost_model=model)
    spans = sorted((s.token_start, s.token_end) for r in plan.ranks for s in r.spans)
    assert spans[0][0] == 0 and spans[-1][1] == cu[-1]
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert max(plan.metrics.token_loads) <= capacity
    assert plan.metrics.score <= dict(plan.evaluated_scores)["contiguous"]
    assert plan.metrics.score == model.estimate_ms(plan.ranks, block_size=4)


def test_full_zigzag_can_win_even_when_all_documents_are_below_split_threshold():
    from benchmarks.cp_layouts import query_layouts

    cu = _boundaries([16384, 4096, 1024])
    model = _cost_model(
        candidate_ns=10,
        query_ns=0,
        span_us=0,
        packing_ns_per_query=0,
        exchange_us=0,
        intra_node_gbps=1e9,
        inter_node_gbps=1e9,
        min_gain_fraction=0,
    )
    plan = plan_thd_selection(cu, world_size=4, max_tokens_per_rank=16384, cost_model=model)
    zigzag, _ = query_layouts(cu, 4, "zigzag")
    assert plan.policy == "zigzag"
    assert plan.ranks == zigzag
    assert dict(plan.evaluated_scores)["zigzag"] == model.estimate_ms(zigzag, block_size=4)
    assert len(plan.evaluated_scores) == 2  # No long-document candidates.
    _assert_exact_cover(plan, cu)


def test_full_zigzag_obeys_capacity_even_when_short_chunks_concentrate_on_rank_zero():
    cu = _boundaries([4] * 8 + [64])
    model = _cost_model(candidate_ns=10000, min_gain_fraction=0, min_gain_ms=0)
    plan = plan_thd_selection(cu, world_size=4, max_tokens_per_rank=24, cost_model=model)
    assert "zigzag" not in dict(plan.evaluated_scores)
    assert max(plan.metrics.token_loads) <= 24


@pytest.mark.parametrize("world", [2, 4, 8, 16, 64])
def test_shared_full_zigzag_matches_explicit_early_late_block_definition(world):
    from cula.distributed import zigzag_thd_layout

    lengths = [4, 12, 64, 260, 16384]
    cu = _boundaries(lengths)
    ranks = zigzag_thd_layout(cu, world_size=world)
    for r in range(world):
        expected = []
        for d, length in enumerate(lengths):
            q, rem = divmod(length // 4, 2 * world)
            for piece in (r, 2 * world - 1 - r):
                begin = cu[d] + 4 * (piece * q + min(piece, rem))
                end = cu[d] + 4 * ((piece + 1) * q + min(piece + 1, rem))
                if end > begin:
                    expected.append((d, begin, end))
        assert [(s.sequence, s.token_start, s.token_end) for s in ranks[r].spans] == expected


def test_shared_node_bandwidth_and_peer_startup_are_not_per_fragment_costs():
    from benchmarks.bench_thd_transport import transport_layout

    model = _cost_model(
        candidate_ns=0,
        query_ns=0,
        span_us=0,
        packing_ns_per_query=0,
        exchange_us=0,
        intra_node_gbps=1e9,
        inter_node_gbps=1e9,
        inter_node_aggregate_gbps=10,
        peer_startup_us=7,
    )
    pair = transport_layout(4, 128, 2, "inter_pair")
    ring = transport_layout(4, 128, 2, "inter")
    dense = transport_layout(4, 128, 2, "inter_dense")
    row_ms = 128 * model.bytes_per_query / (10 * 1e6)
    assert model.estimate_ms(pair, block_size=4) == pytest.approx(row_ms + 0.007)
    assert model.estimate_ms(ring, block_size=4) == pytest.approx(2 * row_ms + 0.007)
    assert model.estimate_ms(dense, block_size=4) == pytest.approx(2 * row_ms + 0.014)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(peer_startup_us=-1),
        dict(peer_startup_us=True),
        dict(peer_startup_us=float("nan")),
        dict(inter_node_aggregate_gbps=0),
        dict(inter_node_aggregate_gbps=True),
        dict(inter_node_aggregate_gbps=float("inf")),
    ],
)
def test_topology_model_rejects_invalid_limits(overrides):
    with pytest.raises(ValueError):
        _cost_model(**overrides)


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("topology", ["contiguous", "interleaved", "irregular"])
def test_zigzag_bound_never_overprices_concrete_layout(world, overlap, topology):
    from cula.distributed.thd_planner import _zigzag_lower_bound, zigzag_thd_layout

    rng = random.Random(27113 + world)
    for _ in range(50):
        cu = _boundaries([4 * rng.randint(1, 8192) for _ in range(rng.randint(1, 50))])
        model = _cost_model(
            rank_node_ids=tuple(
                r // 3 if topology == "contiguous" else r % 3 if topology == "interleaved" else rng.randrange(world)
                for r in range(world)
            ),
            peer_startup_us=7,
            inter_node_aggregate_gbps=12,
            peer_gbps=8,
            overlap_intra_inter=overlap,
        )
        source = _contiguous_items(cu, world, 4)
        bound = _zigzag_lower_bound(cu, source, world, 4, model)
        actual = model.estimate_ms(zigzag_thd_layout(cu, world_size=world), block_size=4)
        assert bound <= actual + 1e-10


def test_transport_prices_only_executed_permutations():
    from dataclasses import replace

    from benchmarks.bench_thd_transport import transport_layout

    layouts = transport_layout(4, 128, 2, "striped")
    model = _cost_model(
        candidate_ns=0,
        query_ns=0,
        span_us=0,
        packing_ns_per_query=1000,
        exchange_us=0,
        intra_node_gbps=1000,
        inter_node_gbps=1000,
    )
    unpriced = replace(model, packing_ns_per_query=0)
    assert model.estimate_ms(layouts, block_size=4) == unpriced.estimate_ms(layouts, block_size=4)
    reversed_layouts = tuple(THDRankLayout(r.rank, tuple(reversed(r.spans))) for r in layouts)
    assert model.estimate_ms(reversed_layouts, block_size=4) - unpriced.estimate_ms(
        reversed_layouts, block_size=4
    ) == pytest.approx(0.128)


def test_independent_link_model_and_its_outgoing_bound():
    from dataclasses import replace

    from benchmarks.bench_thd_transport import transport_layout

    ranks = transport_layout(4, 128, 2, "striped")
    serial = _cost_model(
        candidate_ns=0, query_ns=0, span_us=0, packing_ns_per_query=0, exchange_us=0, intra_node_gbps=10, inter_node_gbps=10
    )
    parallel = replace(serial, overlap_intra_inter=True)
    assert parallel.estimate_ms(ranks, block_size=4) == pytest.approx(2 / 3 * serial.estimate_ms(ranks, block_size=4))
    assert parallel._outgoing_capacity_gbps(0) == 20
    assert serial._outgoing_capacity_gbps(0) == 10
    with pytest.raises(TypeError, match="boolean"):
        replace(serial, overlap_intra_inter=1)


def test_parallel_link_lower_bound_against_all_small_assignments():
    from cula.distributed.thd_planner import _coalesce, _make_item

    cu = (0, 4, 12, 24)
    world = 3
    model = _cost_model(
        rank_node_ids=(0, 0, 1),
        candidate_ns=10000,
        query_ns=10,
        overlap_intra_inter=True,
        peer_startup_us=2,
        inter_node_aggregate_gbps=25,
    )
    base = _contiguous_items(cu, world, 4)
    spans = _coalesce(base)
    source = tuple(THDRankLayout(r, tuple(s for s in spans if s.owner == r)) for r in range(world))
    mean = (
        model.candidate_ns * sum(i.causal_work for i, _ in base) + model.query_ns * 24
    ) / 1e6 / world + model.span_us * 3 / 1e3 / world
    bound = _transfer_lower_bound(base, world, 4, model, mean, model.estimate_ms(source, block_size=4))
    blocks = [_make_item(d, a, a + 4, cu[d], cu[d + 1], 4) for d in range(3) for a in range(cu[d], cu[d + 1], 4)]
    for owners in product(range(world), repeat=6):
        spans = _coalesce(zip(blocks, owners))
        ranks = tuple(THDRankLayout(r, tuple(s for s in spans if s.owner == r)) for r in range(world))
        assert bound <= model.estimate_ms(ranks, block_size=4) + 1e-10
