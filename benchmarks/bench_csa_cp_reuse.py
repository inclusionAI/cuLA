# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Four complete CSA instances sharing one prepared CP selection workspace.

Inputs differ across layers and are identical across policies. All forwards
precede reverse-order backwards. Layers are independent operator instances,
not a connected model: no KDA, projections, residuals, MoE or optimizer.
"""

import argparse
import copy
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from benchmarks.bench_csa_cp_windows import tensors
from benchmarks.bench_csa_layout_matrix import performance_cases
from benchmarks.bench_full_csa_cp import FullCSA, check_redistribution, check_reference, json_safe
from cula.distributed import THDPlannerConfig, THDSelectionCostModel
from cula.hybrid import CsaCPSelectionPlan, CsaCPSelectionWorkspace

POLICIES = ("contiguous", "zigzag", "planner")


def release(layer):
    layer.last = {}
    for value in (layer.indexer.local_kv, layer.indexer.pool_score, layer.indexer.ape):
        value.grad = None


def clone_holder(layer):
    """Share immutable inputs; isolate mutable Python execution bookkeeping."""
    result = copy.copy(layer)
    result.indexer = copy.copy(layer.indexer)
    result.events, result.last = [], {}
    return result


def bind(layers, prepared):
    for layer in layers:
        layer.use_prepared_layout(prepared)
    assert all(layer.cp_selection is prepared.cp_selection for layer in layers)


def run_stack(layers, inspect=None):
    def unexpected(*args, **kwargs):
        raise RuntimeError("selection preparation was invoked inside stack execution")

    # This guards the actual native run, not just a mocked host fixture.
    with (
        patch.object(CsaCPSelectionPlan, "from_cu_seqlens", unexpected),
        patch.object(CsaCPSelectionWorkspace, "from_plan", unexpected),
    ):
        for layer in layers:
            layer.forward()
        for i in reversed(range(len(layers))):
            layer = layers[i]
            # Exclude other layers' execution from this layer's stage breakdown.
            layer.mark("backward_start")
            layer.backward()
            if inspect is not None:
                inspect(i, layer)
            release(layer)


def iteration(layers):
    dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    wall = time.perf_counter()
    begin.record()
    run_stack(layers)
    end.record()
    end.synchronize()
    return dict(
        stack_ms=begin.elapsed_time(end),
        wall_ms=(time.perf_counter() - wall) * 1000,
        allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        layer_stages=[{k: v for k, v in layer.timings().items() if k != "backward_start_ms"} for layer in layers],
    )


def compare_tensors(actual, reference):
    """Chunked full-tensor FP64 statistics, normalized by the contiguous reference."""
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("mismatched full-tensor numerical reference")
    actual, reference = actual.detach().reshape(-1), reference.detach().reshape(-1)
    stats = torch.zeros(3, device=reference.device, dtype=torch.float64)
    finite = torch.ones((), device=reference.device, dtype=torch.bool)
    for start in range(0, reference.numel(), 1 << 22):
        a = actual[start : start + (1 << 22)].to(reference.device).float()
        b = reference[start : start + (1 << 22)].float()
        stats[0] += (a - b).square().sum(dtype=torch.float64)
        stats[1] += b.square().sum(dtype=torch.float64)
        stats[2] = torch.maximum(stats[2], (a - b).abs().max().double())
        finite &= torch.isfinite(a).all() & torch.isfinite(b).all()
    delta, norm, maximum = stats.cpu().tolist()
    return dict(relative_rms=math.sqrt(delta / max(norm, 1e-24)), max_abs=maximum, finite=bool(finite))


def qualify(layers, baseline, small):
    results = [None] * len(layers)

    def inspect(i, layer):
        # Retain one layer on device while evaluating its independent reference.
        # This avoids host staging without sampling or changing error precision.
        actual = {k: v.detach() for k, v in tensors(layer).items()}
        actual_loss = layer.last["loss"].detach().clone()
        oracle = check_reference(layer) if small else None
        release(layer)
        reference = clone_holder(layer)
        reference.use_prepared_layout(baseline)
        reference.forward()
        reference.backward()
        errors = {k: compare_tensors(actual[k], value) for k, value in tensors(reference).items()}
        loss_pair = torch.stack((actual_loss, reference.last["loss"].detach()))
        dist.all_reduce(loss_pair)
        loss, ref_loss = loss_pair.cpu().tolist()
        results[i] = dict(
            errors=errors,
            loss=loss,
            reference_loss=ref_loss,
            loss_abs_diff=abs(loss - ref_loss),
            fp32=oracle,
            okay=all(e["finite"] and e["relative_rms"] < 0.01 for e in errors.values())
            and math.isfinite(loss)
            and math.isfinite(ref_loss)
            and abs(loss - ref_loss) < 1e-5
            and (oracle is None or oracle["pass_"]),
        )
        release(reference)

    run_stack(layers, inspect)
    return results


def summarize(records, repeats, prepare_repeats, layers):
    result = {}
    for policy in POLICIES:
        critical = [max(r["samples"][policy][i]["stack_ms"] for r in records) for i in range(repeats)]
        prep = [max(sum(r["preparation"][policy][i].values()) for r in records) for i in range(prepare_repeats)]
        steady, prepare = statistics.median(critical), statistics.median(prep)
        result[policy] = dict(
            stack_ms=steady,
            prepare_ms=prepare,
            prepare_plus_stack_ms=prepare + steady,
            amortized_per_layer_ms=(prepare + steady) / layers,
            prepare_fraction=prepare / (prepare + steady),
            stack_min_ms=min(critical),
            stack_max_ms=max(critical),
            prepare_min_ms=min(prep),
            prepare_max_ms=max(prep),
            peak_allocated_gib=max(s["allocated_gib"] for r in records for s in r["samples"][policy]),
            peak_reserved_gib=max(s["reserved_gib"] for r in records for s in r["samples"][policy]),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--prepare-repeats", type=int, default=5)
    parser.add_argument("--min-gain-ms", type=float, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.layers < 2 or min(args.warmups, args.repeats, args.prepare_repeats) < 1:
        parser.error("at least two layers and positive measurement counts are required")
    suite = [(name, lengths) for name, lengths, _ in performance_cases("validation")]
    if args.cases:
        if set(args.cases) - {name for name, _ in suite}:
            parser.error("unknown validation case")
        suite = [(n, lengths) for n, lengths in suite if n in args.cases]
    if args.check_only:
        suite = [("check-single", [4096]), ("check-many", [4] * 1024), ("check-mixed", [4] * 64 + [1920] * 2)]
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model_data = json.loads(args.profile.read_text())["model"]
    model_data["rank_node_ids"] = tuple(model_data["rank_node_ids"])
    model = replace(THDSelectionCostModel(**model_data), min_gain_ms=args.min_gain_ms)
    if len(model.rank_node_ids) != world:
        raise ValueError("profile topology mismatch")
    config = THDPlannerConfig(min_chunk_tokens=1024, max_chunks_per_rank=6)
    root = Path(__file__).resolve().parents[1]
    import cudnn

    manifest = dict(
        schema_version=1,
        source_revision=args.source_revision,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        model=asdict(model),
        config=asdict(config),
        suite=[dict(name=n, lengths=ls) for n, ls in suite],
        software=dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
            cudnn_backend=str(cudnn.backend_version()),
        ),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("cula", "benchmarks")
            for p in sorted((root / folder).rglob("*.py"))
        },
    )
    manifests = [None] * world
    dist.all_gather_object(manifests, manifest)
    if any(m != manifests[0] for m in manifests):
        raise RuntimeError("source/config differs across ranks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for case_index, (name, lengths) in enumerate(suite):
        options = dict(
            heads=64,
            top_k=512,
            device=device,
            candidate_mib=0,
            selection_api=True,
            planner_cost_model=model,
            planner_strategy="selection",
            planner_config=config,
        )
        torch.manual_seed(20261117 + rank)
        first = FullCSA(lengths, **options)
        holders = {"contiguous": clone_holder(first)}
        for policy in POLICIES[1:]:
            holder = clone_holder(first)
            holder._layout_options = dict(first._layout_options, layout=policy)
            holder._prepare_layout(**holder._layout_options)
            holders[policy] = holder
        layers = [first]
        for i in range(1, args.layers):
            torch.manual_seed(20261117 + rank + i * 1000)
            layers.append(FullCSA(lengths, prepared_from=first, **options))
        if not all(check_redistribution(h) for h in holders.values()):
            raise RuntimeError("source/inverse marker check failed")
        if rank == 0:
            print("CASE", name, "selected", holders["planner"].selected_policy, flush=True)
        samples = {p: [] for p in POLICIES}
        orders = []
        for i in range(args.warmups + args.repeats):
            shift = (case_index + i) % len(POLICIES)
            order = POLICIES[shift:] + POLICIES[:shift]
            if i % 2:
                order = order[::-1]
            orders.append(order)
            for policy in order:
                bind(layers, holders[policy])
                value = iteration(layers)
                if i >= args.warmups:
                    samples[policy].append(value)
            if rank == 0:
                print("ITER", name, i, {p: round(samples[p][-1]["stack_ms"], 2) for p in POLICIES if samples[p]}, flush=True)
        checks = {}
        for policy in POLICIES:
            bind(layers, holders[policy])
            checks[policy] = qualify(layers, holders["contiguous"], args.check_only)
        preparation = {p: [] for p in POLICIES}
        prepare_orders = []
        for i in range(args.prepare_repeats):
            order = POLICIES[i % 3 :] + POLICIES[: i % 3]
            if i % 2:
                order = order[::-1]
            prepare_orders.append(order)
            for policy in order:
                dist.barrier(device_ids=[device.index])
                preparation[policy].append(holders[policy].prepare_metadata())
        layout = {
            p: dict(
                selected_policy=h.selected_policy,
                source_rows=h.source_n,
                selection_rows=h.selection_layout.local_tokens,
                spans=len(h.selection_layout.spans),
                candidate_workspace_bytes=0
                if h.indexer.select_plan is None
                else h.indexer.select_plan.candidate_workspace_floats * 4,
            )
            for p, h in holders.items()
        }
        record = dict(rank=rank, samples=samples, checks=checks, preparation=preparation, layout=layout)
        records = [None] * world
        dist.all_gather_object(records, record)
        okay = all(c["okay"] for r in records for policy in POLICIES for c in r["checks"][policy])
        if rank == 0:
            summary = summarize(records, args.repeats, args.prepare_repeats, args.layers)
            report = dict(
                case=name,
                lengths=lengths,
                world_size=world,
                layers=args.layers,
                okay=okay,
                orders=orders,
                prepare_orders=prepare_orders,
                summary=summary,
                ranks=records,
            )
            (args.output_dir / f"{name}.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
            print("RESULT", name, json.dumps(summary), "PASS", okay, flush=True)
        if not okay:
            raise RuntimeError("numerical qualification failed; report retained")
        del first, holder, holders, layers, records, record, checks, value
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
