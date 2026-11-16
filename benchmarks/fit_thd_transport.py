# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fit nonnegative effective transport costs from independent routing sweeps.

SciPy is a calibration-only dependency. Production planning does not import it.
Held-out fixtures are reported, never included in the optimization objective.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def predict(features, coefficients, overlap_links=False):
    """Same ordered-phase resource model used by THDSelectionCostModel."""
    if len(coefficients) not in (7, 8):
        raise ValueError("transport prediction requires seven legacy or eight diagonal-aware coefficients")
    intra, inter, peer, node, packing, startup, fixed = coefficients[:7]
    local = coefficients[7] if len(coefficients) == 8 else 0.0
    unit = 65536
    combine = max if overlap_links else lambda a, b: a + b
    local_rows = features.get("rank_local_rows", [0] * len(features["rank_intra_rows"]))
    rank = (
        max(
            combine(a * intra + c * local, b * inter)
            for a, b, c in zip(features["rank_intra_rows"], features["rank_inter_rows"], local_rows)
        )
        / unit
    )
    transfer = max(rank, features["peer_rows"] * peer / unit, features["node_rows"] * node / unit)
    return transfer + features["copies"] * packing / unit + features["peer_degree"] * startup + fixed


def fit(raw, profile, *, overlap_links=False):
    from scipy.optimize import differential_evolution, least_squares

    world, per_node = raw["world"], raw["ranks_per_node"]
    if (
        isinstance(world, bool)
        or not isinstance(world, int)
        or isinstance(per_node, bool)
        or not isinstance(per_node, int)
        or world < 2
        or per_node < 1
        or world % per_node
    ):
        raise ValueError("calibration topology requires CP >= 2 and complete nodes")
    rows = [r for r in raw["records"] if not r["heldout"]]
    if len(rows) < 6 or any(
        not r["marker_exact"] or not np.isfinite(r["critical_median_ms"]) or r["critical_median_ms"] <= 0
        for r in raw["records"]
    ):
        raise ValueError("need complete, finite positive transport timings and exact row markers")
    y = np.array([r["critical_median_ms"] for r in rows])

    def residual(x):
        return (np.array([predict(r["features"], x, overlap_links) for r in rows]) - y) / np.maximum(y, 0.25)

    # Inactive max-resource branches have zero derivative. A bounded global
    # initialization avoids silently accepting a rank-bandwidth coefficient
    # that never became active during local least-squares optimization.
    scale = max(y)
    startup_limit = min(r["critical_median_ms"] / max(r["features"]["peer_degree"], 1) for r in rows)
    bounds = [(0, 2 * scale)] * 5 + [(0, startup_limit), (0, min(y))]
    fit_local = all("rank_local_rows" in r["features"] for r in rows) and any(
        any(r["features"]["rank_local_rows"]) for r in rows
    )
    if fit_local:
        bounds.append((0, 2 * scale))
    initialized = differential_evolution(
        lambda x: np.sum(residual(x) ** 2), bounds, seed=1, popsize=10, maxiter=300, tol=1e-8, polish=False
    )
    best = least_squares(residual, initialized.x, bounds=(0, np.inf), max_nfev=4000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    x = best.x
    # Ordered sweep fixtures elide every permutation. Their timings cannot
    # identify copy bandwidth; use a separately measured packing coefficient.
    if not any(r["features"]["copies"] for r in rows):
        x[4] = profile["model"].get("packing_ns_per_query", 0) * 65536 / 1e6
    # A single-node calibration cannot identify inter-node bandwidth.
    # Use the same unused fallback rate, not an arbitrary optimizer parameter.
    # The sweep assigns contiguous group ranks to nodes. Export that same
    # topology rather than retaining the input profile's deployment mapping.
    model = dict(
        profile["model"],
        overlap_intra_inter=overlap_links,
        rank_node_ids=[rank // per_node for rank in range(world)],
    )
    factor = 65536 * model["bytes_per_query"] / 1e6
    model.update(
        intra_node_gbps=factor / max(x[0], 1e-10),
        inter_node_gbps=factor / max(x[1] if raw["world"] > raw["ranks_per_node"] else x[0], 1e-10),
        peer_gbps=factor / max(x[2], 1e-10),
        inter_node_aggregate_gbps=factor / max(x[3], 1e-10) if raw["world"] > raw["ranks_per_node"] else None,
        packing_ns_per_query=x[4] * 1e6 / 65536,
        peer_startup_us=x[5] * 1000,
        exchange_us=x[6] * 1000,
        local_copy_gbps=factor / max(x[7], 1e-10) if fit_local else None,
    )
    predictions = [
        dict(
            kind=r["kind"],
            rows=r["rows"],
            heldout=r["heldout"],
            actual_ms=r["critical_median_ms"],
            predicted_ms=predict(r["features"], x, overlap_links),
            relative_error=predict(r["features"], x, overlap_links) / r["critical_median_ms"] - 1,
        )
        for r in raw["records"]
    ]
    return dict(
        profile,
        schema_version=6 if fit_local else 5,
        model=model,
        transport_calibration=dict(
            method="Nonnegative max-resource fit; independent eager transport timings",
            coefficients_ms_per_65536_rows=x.tolist(),
            predictions=predictions,
            ranks_per_node=raw["ranks_per_node"],
            world_size=raw["world"],
            heldout_rule="32768 rows, striped or local_half layout; never used in fitting",
            note="Effective coefficients include eager implementation costs; not physical bandwidth or universal optimality guarantees.",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overlap-links", action="store_true")
    a = p.parse_args()
    result = fit(json.loads(a.records.read_text()), json.loads(a.profile.read_text()), overlap_links=a.overlap_links)
    result["transport_calibration"]["source_sha256"] = hashlib.sha256(a.records.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["model"], indent=2))


if __name__ == "__main__":
    main()
