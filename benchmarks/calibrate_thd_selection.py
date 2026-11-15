# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fit explicit CPU planner coefficients from saved selection-routing runs.

This does not run GPU work or tune kernels. Use separate, held-out THD inputs
to evaluate the resulting policy; calibration fit alone is not validation.
"""

import argparse
import hashlib
import json
import math
import statistics
from itertools import accumulate, combinations
from pathlib import Path

from benchmarks.cp_layouts import query_layouts
from cula.distributed.thd_planner import _candidate_work


def _solve(matrix, rhs):
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(len(rhs)):
        pivot = max(range(col, len(rhs)), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-14:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [x / scale for x in rows[col]]
        for r in range(len(rhs)):
            if r != col:
                scale = rows[r][col]
                rows[r] = [a - scale * b for a, b in zip(rows[r], rows[col])]
    return [row[-1] for row in rows]


def nonnegative_fit(features, targets):
    """Exact active-set enumeration for three nonnegative cost coefficients."""
    if not features or len(features) != len(targets) or any(len(row) != 3 for row in features):
        raise ValueError("need matched, nonempty measurements with three features")
    if any(not math.isfinite(v) or v < 0 for row in features for v in row) or any(not math.isfinite(v) for v in targets):
        raise ValueError("fit inputs must be finite and feature counts nonnegative")
    best, error = [0.0] * 3, sum(y * y for y in targets)
    for size in range(1, 4):
        for active in combinations(range(3), size):
            matrix = [[sum(x[i] * x[j] for x in features) for j in active] for i in active]
            rhs = [sum(x[i] * y for x, y in zip(features, targets)) for i in active]
            solution = _solve(matrix, rhs)
            if solution is None or any(x < 0 for x in solution):
                continue
            weights = [0.0] * 3
            for i, value in zip(active, solution):
                weights[i] = value
            loss = sum((sum(a * b for a, b in zip(x, weights)) - y) ** 2 for x, y in zip(features, targets))
            if loss < error:
                best, error = weights, loss
    return best, math.sqrt(error / len(targets))


def routing_critical_ms(samples):
    """Critical-path increments, excluding pre-existing rank arrival skew.

    Each sample starts at the iteration's post-barrier event. At every phase,
    max(cumulative end) - max(cumulative start) is the phase's contribution to
    the distributed makespan. Summing local exchange durations instead would
    count waiting for slow KDA/selection ranks again as network time.
    """
    phases = (
        "query_redistribute_ms",
        "pool_fwd_ms",
        "pool_gather_ms",
        "selection_key_layout_ms",
        "selection_ms",
        "selection_restore_ms",
    )
    if not samples or any(not math.isfinite(s[k]) or s[k] < 0 for s in samples for k in ("kda_fwd_ms", *phases)):
        raise ValueError("routing phases must be finite, nonnegative and nonempty")
    arrival = [s["kda_fwd_ms"] for s in samples]
    duration = 0.0
    for phase in phases:
        before = max(arrival)
        arrival = [a + s[phase] for a, s in zip(arrival, samples)]
        if phase in ("query_redistribute_ms", "selection_key_layout_ms", "selection_restore_ms"):
            duration += max(arrival) - before
    return duration


def calibrate(paths, ranks_per_node, margin=0.05):
    features, targets, rates, sources, digests = [], [], [], [], {}
    world = None
    # Current benchmark: BF16 H64/D128 Q, BF16 H64 weights,
    # int32 IDs + FP32 scores for Top-K 512.
    payload = 2 * 64 * 128 + 2 * 64 + (4 + 4) * 512
    for path in paths:
        raw = json.loads(path.read_text())
        config = raw["config"]
        if config["check"]:
            continue
        if config.get("redistribution_scope") != "selection" or raw.get("planner_cost_model") is not None:
            raise ValueError("calibrate from unmodeled selection-only runs, not query-wide or already calibrated runs")
        if config["top_k"] != 512:
            raise ValueError("this benchmark profile requires indexer H64/D128 and Top-K 512")
        if world is not None and world != raw["world_size"]:
            raise ValueError("cannot combine different CP group sizes")
        world = raw["world_size"]
        bounds = (0, *accumulate(config["sequence_lengths"]))
        ranks, selected = query_layouts(bounds, world, config["layout"])
        observed = sorted(raw["ranks"], key=lambda x: x["rank"])
        if [r["rank"] for r in observed] != list(range(world)):
            raise ValueError("calibration requires exactly one result per process-group rank")
        if any(not all(r[k] for k in ("finite", "input_match", "redistribution_bitwise")) for r in observed):
            raise ValueError("cannot calibrate from a failed numerical/redistribution check")
        if any(
            not e["finite"] or not math.isfinite(e["relative_rms"]) or e["relative_rms"] >= 0.03
            for r in observed
            for e in (r["layout_vs_contiguous"] or {}).values()
        ):
            raise ValueError("cannot calibrate from failed output/gradient comparisons")
        if any(len(r["samples"]) != config["repeats"] for r in observed):
            raise ValueError("incomplete timing samples")
        if selected != observed[0]["selected_policy"]:
            raise ValueError("source planner differs from the measured assignment")
        for layout, result in zip(ranks, observed):
            candidates = sum(_candidate_work(s.causal_offset, s.tokens, 4) for s in layout.spans)
            features.append([candidates / 1e9, layout.local_tokens / 1e6, len(layout.spans) / 1000])
            targets.append(statistics.median(s["selection_ms"] for s in result["samples"]))
        repeats = len(observed[0]["samples"])
        duration = statistics.median(routing_critical_ms([r["samples"][i] for r in observed]) for i in range(repeats))
        remote_rows = max(r["remote_query_rows"] for r in observed)
        if remote_rows:
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("routing timings must be finite and positive")
            rates.append(remote_rows * payload / (duration * 1e6))
        sources.append(str(path))
        digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not features or not rates or world is None:
        raise ValueError("need measured selection work and at least one nonidentity routing case")
    if ranks_per_node < 1 or world % ranks_per_node:
        raise ValueError("ranks_per_node must divide the measured group size")
    coefficients, rmse = nonnegative_fit(features, targets)
    # The measurements do not isolate NVLink vs inter-node links. Use one
    # conservative effective rate, including packing, for both. Cumulative
    # critical increments remove pre-existing arrival skew from the fit.
    # Do not present it as a physical-link bandwidth measurement.
    rate = min(rates)
    return dict(
        schema_version=2,
        model=dict(
            candidate_ns=coefficients[0] / 1000,
            query_ns=coefficients[1],
            span_us=coefficients[2],
            bytes_per_query=payload,
            intra_node_gbps=rate,
            inter_node_gbps=rate,
            rank_node_ids=[r // ranks_per_node for r in range(world)],
            packing_ns_per_query=0.0,
            exchange_us=0.0,
            min_gain_fraction=margin,
        ),
        calibration=dict(
            sources=sources,
            source_sha256=digests,
            samples=len(targets),
            selection_fit_rmse_ms=rmse,
            observed_routing_effective_gbps=rates,
            transport_note="Critical phase increments remove pre-existing rank arrival skew. Shared conservative effective rate includes packing; links were not isolated.",
            validation_note="Evaluate held-out layouts and seeds; fitting is not a performance guarantee.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--ranks-per-node", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({p for root in args.records for p in root.glob("layout-1m-*.json")})
    result = calibrate(paths, args.ranks_per_node)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["model"], indent=2))
    print("selection fit RMSE ms", result["calibration"]["selection_fit_rmse_ms"])


if __name__ == "__main__":
    main()
