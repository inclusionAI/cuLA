# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.bench_thd_transport import transport_layout
from benchmarks.calibrate_thd_selection import nonnegative_fit, routing_critical_ms
from cula.distributed import THDRedistributionPlan


def test_nonnegative_fit_recovers_independent_costs():
    x = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 3, 4]]
    weights, rmse = nonnegative_fit(x, [2, 5, 7, 47])
    assert weights == pytest.approx([2, 5, 7])
    assert rmse == pytest.approx(0, abs=1e-10)


def test_fit_drops_negative_cost_instead_of_cancelling_compute():
    weights, _ = nonnegative_fit([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [2, -5, 7])
    assert weights == pytest.approx([2, 0, 7])


@pytest.mark.parametrize("features,targets", [([], []), ([[1, 2, 3]], []), ([[1, 2]], [3]), ([[float("nan"), 1, 2]], [3])])
def test_fit_rejects_incomplete_or_nonfinite_measurements(features, targets):
    with pytest.raises(ValueError):
        nonnegative_fit(features, targets)


def test_routing_fit_excludes_kda_and_selection_arrival_waits():
    def rank(kda, query, selection, restore):
        return dict(
            kda_fwd_ms=kda,
            query_redistribute_ms=query,
            pool_fwd_ms=1,
            pool_gather_ms=2,
            selection_key_layout_ms=0.5,
            selection_ms=selection,
            selection_restore_ms=restore,
        )

    # Rank 0 waits 80 ms before query exchange and 60 ms before restore.
    # Both exchanges take 5/3 ms after the last rank arrives, respectively.
    a, b = rank(20, 85, 10, 63), rank(100, 5, 70, 3)
    assert routing_critical_ms([a, b]) == pytest.approx(8.5)
    # Shifting the slow phase earlier on the other rank leaves transfer cost
    # unchanged. Neither the original waiting times nor rank identity matter.
    assert routing_critical_ms([b, a]) == pytest.approx(8.5)
    assert routing_critical_ms([rank(100, 5, 70, 3)] * 2) == pytest.approx(8.5)


def test_routing_fit_rejects_invalid_phase_measurements():
    with pytest.raises(ValueError):
        routing_critical_ms([])


@pytest.mark.parametrize("kind", ["intra", "intra_dense", "inter", "inter_dense", "striped"])
@pytest.mark.parametrize("world,per_node", [(4, 2), (8, 4), (16, 8)])
def test_transport_calibration_exercises_the_declared_links(kind, world, per_node):
    rows = 128
    ranks = transport_layout(world, rows, per_node, kind)
    for rank in range(world):
        p = THDRedistributionPlan.build(ranks, tuple(i * rows for i in range(world + 1)), rank)
        assert p.target_rows == p.source_rows == rows
        assert sum(p.send_counts) == sum(p.receive_counts) == rows
        assert not p.identity
        peers = [i for i, count in enumerate(p.send_counts) if count]
        if kind.startswith("intra"):
            assert all(i // per_node == rank // per_node for i in peers)
            assert (rank in peers) == (kind == "intra_dense")
        elif kind.startswith("inter"):
            assert all(i // per_node != rank // per_node for i in peers)
        else:
            assert peers == list(range(world))
            assert len(set(p.send_counts)) == 1


@pytest.mark.parametrize("world,per_node", [(4, 2), (8, 8), (16, 8)])
@pytest.mark.parametrize("kind", ["intra_pair", "intra", "intra_dense", "striped"])
@pytest.mark.parametrize("overlap", [False, True])
def test_transport_fit_prediction_uses_the_same_cost_model_as_planning(world, per_node, kind, overlap):
    from benchmarks.fit_thd_transport import predict
    from cula.distributed import THDSelectionCostModel

    rows = 128
    layouts = transport_layout(world, rows, per_node, kind)
    plans = [THDRedistributionPlan.build(layouts, tuple(i * rows for i in range(world + 1)), r) for r in range(world)]
    intra_out = [0] * world
    inter_out = [0] * world
    intra_in = [0] * world
    inter_in = [0] * world
    node_out = [0] * (world // per_node)
    node_in = node_out.copy()
    degrees_out = [0] * world
    degrees_in = [0] * world
    edge = []
    for s, p in enumerate(plans):
        for t, n in enumerate(p.send_counts):
            if s == t or not n:
                continue
            edge.append(n)
            degrees_out[s] += 1
            degrees_in[t] += 1
            if s // per_node == t // per_node:
                intra_out[s] += n
                intra_in[t] += n
            else:
                inter_out[s] += n
                inter_in[t] += n
                node_out[s // per_node] += n
                node_in[t // per_node] += n
    f = dict(
        rank_intra_rows=intra_out + intra_in,
        rank_inter_rows=inter_out + inter_in,
        peer_rows=max(edge),
        node_rows=max(node_out + node_in),
        peer_degree=max(degrees_out + degrees_in),
        copies=0,
    )
    coeff = [1, 3, 4, 2, 0.5, 0.007, 0.01]
    factor = 65536 * 20608 / 1e6
    m = THDSelectionCostModel(
        candidate_ns=0,
        query_ns=0,
        span_us=0,
        bytes_per_query=20608,
        rank_node_ids=tuple(r // per_node for r in range(world)),
        intra_node_gbps=factor / coeff[0],
        inter_node_gbps=factor / coeff[1],
        peer_gbps=factor / coeff[2],
        inter_node_aggregate_gbps=factor / coeff[3],
        packing_ns_per_query=coeff[4] * 1e6 / 65536,
        peer_startup_us=coeff[5] * 1000,
        exchange_us=coeff[6] * 1000,
        overlap_intra_inter=overlap,
    )
    assert predict(f, coeff, overlap) == pytest.approx(m.estimate_ms(layouts, block_size=4))


@pytest.fixture
def transport_fit_inputs(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    optimize = pytest.importorskip("scipy.optimize")
    # Keep export/metadata regressions independent of optimizer convergence.
    coefficients = np.array([1.0, 3.0, 4.0, 2.0, 0.5, 0.007, 0.01])
    monkeypatch.setattr(optimize, "differential_evolution", lambda *a, **kw: SimpleNamespace(x=coefficients.copy()))
    monkeypatch.setattr(optimize, "least_squares", lambda *a, **kw: SimpleNamespace(x=coefficients.copy()))
    profile = dict(model=dict(bytes_per_query=20608, rank_node_ids=[0] * 8, packing_ns_per_query=8.5))
    feature = dict(rank_intra_rows=[128] * 16, rank_inter_rows=[0] * 16, peer_rows=128, node_rows=0, copies=0, peer_degree=1)
    record = dict(heldout=False, marker_exact=True, critical_median_ms=1, features=feature, kind="intra", rows=128)
    return dict(world=8, ranks_per_node=8, records=[record.copy() for _ in range(6)]), profile


@pytest.mark.parametrize("world,per_node", [(8, 4), (16, 8), (8, 8), (4, 1)])
@pytest.mark.parametrize("old_per_node", [4, 8])
def test_transport_fit_exports_calibration_topology(transport_fit_inputs, world, per_node, old_per_node):
    from copy import deepcopy

    from benchmarks.fit_thd_transport import fit

    raw, profile = transport_fit_inputs
    profile["model"]["rank_node_ids"] = [r // old_per_node for r in range(8)]
    original = deepcopy(profile)
    raw.update(world=world, ranks_per_node=per_node)
    result = fit(raw, profile)
    assert result["model"]["rank_node_ids"] == [r // per_node for r in range(world)]
    assert result["transport_calibration"]["world_size"] == world
    assert result["transport_calibration"]["ranks_per_node"] == per_node
    assert profile == original


@pytest.mark.parametrize("world,per_node", [(0, 4), (1, 1), (8, 0), (8, 3), (8, 16), (8.0, 4), (8, 4.0), (True, 1), (8, True)])
def test_transport_fit_rejects_invalid_topology(transport_fit_inputs, world, per_node):
    from benchmarks.fit_thd_transport import fit

    raw, profile = transport_fit_inputs
    raw.update(world=world, ranks_per_node=per_node)
    with pytest.raises(ValueError, match="topology"):
        fit(raw, profile)


@pytest.mark.parametrize("heldout", [False, True])
@pytest.mark.parametrize("timing", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_transport_fit_rejects_invalid_timings_before_optimization(transport_fit_inputs, monkeypatch, heldout, timing):
    import scipy.optimize

    from benchmarks.fit_thd_transport import fit

    raw, profile = transport_fit_inputs
    raw["records"].append(dict(raw["records"][0], heldout=heldout, critical_median_ms=timing))

    def unexpected(*args, **kwargs):
        pytest.fail("invalid timing reached the optimizer")

    monkeypatch.setattr(scipy.optimize, "differential_evolution", unexpected)
    with pytest.raises(ValueError, match="finite positive"):
        fit(raw, profile)
