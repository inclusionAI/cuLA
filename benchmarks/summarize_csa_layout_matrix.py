# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Summarize complete paired full-CSA matrices, retaining worst-rank errors."""

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(directory):
    grouped = {}
    for path in sorted(directory.glob("layout-1m-*.json")):
        raw = json.loads(path.read_text())
        layout = raw["config"]["layout"]
        name = path.stem.removeprefix("layout-").removesuffix("-" + layout)
        ranks = raw["ranks"]
        if sorted(r["rank"] for r in ranks) != list(range(raw["world_size"])):
            raise ValueError(f"incomplete rank results: {path}")
        if any(not all(r[k] for k in ("finite", "input_match", "redistribution_bitwise")) for r in ranks):
            raise ValueError(f"failed correctness check: {path}")
        repeats = raw["config"]["repeats"]
        if not repeats or any(len(r["samples"]) != repeats for r in ranks):
            raise ValueError(f"incomplete timing samples: {path}")
        if any(
            not math.isfinite(s[k]) or s[k] < 0
            for r in ranks
            for s in r["samples"]
            for k in ("total_ms", "peak_allocated_gib", "peak_reserved_gib")
        ) or any(
            not math.isfinite(r[k]) or r[k] < 0
            for r in ranks
            for k in ("layout_prepare_ms", "common_prepare_ms", "kda_prepare_ms", "total_prepare_ms")
        ):
            raise ValueError(f"invalid timing/memory samples: {path}")
        if any(
            not math.isclose(r["total_prepare_ms"], r["layout_prepare_ms"] + r["common_prepare_ms"] + r["kda_prepare_ms"])
            for r in ranks
        ):
            raise ValueError(f"inconsistent preparation accounting: {path}")
        errors = {}
        for rank in ranks:
            if layout != "contiguous":
                required = {"output", "dq", "dkv", "d_index_q", "d_weights", "d_pool_kv", "d_pool_score", "d_ape"}
                if raw["config"].get("with_kda"):
                    required |= {"kda_output", "kda_dq", "kda_dk", "kda_dv", "kda_dg", "kda_dbeta"}
                if not required.issubset(rank.get("layout_vs_contiguous") or {}):
                    raise ValueError(f"missing output/gradient baseline comparison: {path}")
                delta = rank.get("loss_abs_diff")
                if delta is None or not math.isfinite(delta) or not 0 <= delta < 1e-5:
                    raise ValueError(f"failed loss comparison: {path}")
            for tensor, error in (rank["layout_vs_contiguous"] or {}).items():
                if not error["finite"] or not 0 <= error["relative_rms"] < 0.03:
                    raise ValueError(f"failed precision comparison: {path}: {tensor}")
                errors[tensor] = max(errors.get(tensor, 0.0), error["relative_rms"])
        critical = [max(r["samples"][i]["total_ms"] for r in ranks) for i in range(repeats)]
        row = dict(
            source=str(path),
            total_ms=statistics.median(critical),
            min_ms=min(critical),
            max_ms=max(critical),
            allocated_gib=max(s["peak_allocated_gib"] for r in ranks for s in r["samples"]),
            reserved_gib=max(s["peak_reserved_gib"] for r in ranks for s in r["samples"]),
            prepare_ms=max(r["total_prepare_ms"] for r in ranks),
            layout_prepare_ms=max(r["layout_prepare_ms"] for r in ranks),
            policy=ranks[0]["selected_policy"],
            relative_rms=errors,
            loss_abs_diff=max((r["loss_abs_diff"] or 0.0) for r in ranks),
            workspace_gib=max(r["selection_workspace_bytes"] for r in ranks) / 2**30,
            max_spans=max(r["selection_spans"] for r in ranks),
        )
        contract = {
            key: raw["config"].get(key)
            for key in (
                "heads",
                "top_k",
                "with_kda",
                "kda_batch_invariant",
                "kda_gate_scale",
                "candidate_mib",
                "redistribution_scope",
                "pool_backward_order",
                "repeats",
                "warmups",
            )
        }
        group = grouped.setdefault(
            name, dict(lengths=raw["config"]["sequence_lengths"], world_size=raw["world_size"], contract=contract, layouts={})
        )
        if (
            group["lengths"] != raw["config"]["sequence_lengths"]
            or group["world_size"] != raw["world_size"]
            or group["contract"] != contract
        ):
            raise ValueError(f"unpaired shape/group: {name}")
        group["layouts"][layout] = row
    if not grouped:
        raise ValueError("no 1M results found")
    for name, case in grouped.items():
        if set(case["layouts"]) != {"contiguous", "zigzag", "planner"}:
            raise ValueError(f"incomplete comparison: {name}")
        c, z, p = (case["layouts"][k] for k in ("contiguous", "zigzag", "planner"))
        case["speedup_vs_contiguous"] = c["total_ms"] / p["total_ms"]
        case["gap_to_best_baseline_pct"] = 100 * (p["total_ms"] / min(c["total_ms"], z["total_ms"]) - 1)
        case["prepare_plus_iteration_ms"] = {k: v["prepare_ms"] + v["total_ms"] for k, v in case["layouts"].items()}
        first = case["prepare_plus_iteration_ms"]
        case["prepare_inclusive_gap_pct"] = 100 * (first["planner"] / min(first["contiguous"], first["zigzag"]) - 1)
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = summarize(args.directory)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("| Case | Docs | Contiguous ms | Zigzag ms | Planner ms | Speedup | Planner allocated GiB | Prepare ms | Policy |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, case in results.items():
        c, z, p = (case["layouts"][k] for k in ("contiguous", "zigzag", "planner"))
        print(
            f"| {name} | {len(case['lengths'])} | {c['total_ms']:.2f} | {z['total_ms']:.2f} | {p['total_ms']:.2f} | "
            f"{case['speedup_vs_contiguous']:.3f}x | {p['allocated_gib']:.2f} | {p['prepare_ms']:.2f} | {p['policy']} |"
        )


if __name__ == "__main__":
    main()
