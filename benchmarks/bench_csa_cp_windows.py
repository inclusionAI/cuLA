# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public CSA selection API: complete CSA F/B across packed context lengths.

No projections, KDA, MoE, optimizer, or checkpoint recomputation are included.
All variants use the same BF16 inputs and FP32 communication/reduction contract.
"""

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from benchmarks.bench_full_csa_cp import FullCSA, check_redistribution, check_reference, compare_cpu_reference, json_safe
from cula.distributed import THDPlannerConfig, THDSelectionCostModel


def cases(tokens):
    """Five block-aligned layouts, including 512/1000-document packs."""
    short = 64
    mixed = [short] * 998 + [(tokens - 998 * short) // 2] * 2
    # The two long documents have different positions in a reproducible pack.
    mixed = mixed[:199] + [mixed[-1]] + mixed[199:-2] + [mixed[-2]]
    tail = [64] * 800 + [256] * 199 + [tokens - 800 * 64 - 199 * 256]
    return {
        "single": [tokens],
        "uniform-1024": [tokens // 1024] * 1024,
        "near-uniform-512": [tokens // 512 - 4, tokens // 512 + 4] * 256,
        "mixed-1000": mixed,
        "long-tail-1000": tail,
    }


def tensors(csa):
    r, b = csa.last, csa.indexer
    return dict(
        output=r["source_output"],
        dq=r["source_dq"],
        dkv=r["local_dkv"],
        d_index_q=r["source_dqi"],
        d_weights=r["source_dw"],
        d_pool_kv=b.local_kv.grad,
        d_pool_score=b.pool_score.grad,
        d_ape=b.ape.grad,
    )


def iteration(csa):
    dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    wall = time.perf_counter()
    begin.record()
    csa.forward()
    csa.backward()
    end.record()
    end.synchronize()
    return dict(
        total_ms=begin.elapsed_time(end),
        wall_ms=1000 * (time.perf_counter() - wall),
        allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        **csa.timings(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128 << 10, 256 << 10])
    parser.add_argument("--cases", nargs="+", help="Subset of the five named distributions")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prepare-repeats", type=int, default=8)
    parser.add_argument("--min-gain-ms", type=float, help="Explicit preparation budget override, recorded in the report")
    parser.add_argument(
        "--min-chunk-tokens", type=int, default=1024, help="Minimum newly created planner span; independent of context length"
    )
    parser.add_argument("--max-chunks-per-rank", type=int, default=6)
    parser.add_argument("--check-only", action="store_true", help="Small independent FP32 oracle fixtures")
    args = parser.parse_args()
    if min(args.warmups, args.repeats, args.prepare_repeats) < 1 or any(t < 128 << 10 or t % 4096 for t in args.tokens):
        parser.error("warmups/repeats must be positive; contexts must be >= 128K and divisible by 4096")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    profiles = [json.loads(args.profile.read_text())["model"] if rank == 0 else None]
    dist.broadcast_object_list(profiles, src=0)
    profiles[0]["rank_node_ids"] = tuple(profiles[0]["rank_node_ids"])
    model = THDSelectionCostModel(**profiles[0])
    if args.min_gain_ms is not None:
        model = replace(model, min_gain_ms=args.min_gain_ms)
    if len(model.rank_node_ids) != world:
        raise ValueError("profile topology does not match world size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict

    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, model=asdict(model))
    source = Path(__file__).resolve().parents[1]
    import importlib.metadata

    try:
        frontend_version = importlib.metadata.version("nvidia-cudnn-frontend")
    except importlib.metadata.PackageNotFoundError:
        frontend_version = None  # Source-only frontend checkouts have no wheel metadata.
    manifest["software"] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "cudnn_frontend": frontend_version,
    }
    manifest["source_sha256"] = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in (source / "cula", source / "benchmarks")
        for p in sorted(directory.rglob("*.py"))
    }
    manifests = [None] * world
    dist.all_gather_object(manifests, manifest)
    if any(m != manifests[0] for m in manifests):
        raise RuntimeError("source/config differs across ranks")
    if rank == 0:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    suite = [
        (f"{t // 1024}k-{name}", lengths)
        for t in args.tokens
        for name, lengths in cases(t).items()
        if not args.cases or name in args.cases
    ]
    if args.check_only:
        suite = [("check-single", [4096]), ("check-many", [4] * 1024), ("check-mixed", [4] * 64 + [1920] * 2)]
    if not suite:
        raise ValueError("no matching cases")
    for name, lengths in suite:
        variants, samples, checks, preparation = {}, {}, {}, {}
        policies = ("legacy-contiguous", "contiguous", "zigzag", "planner")
        for policy in policies:
            torch.manual_seed(20261116 + rank)
            csa = FullCSA(
                lengths,
                64,
                512,
                device,
                0,
                layout="contiguous" if policy.startswith("legacy") else policy,
                planner_cost_model=model,
                planner_strategy="selection",
                selection_api=not policy.startswith("legacy"),
                planner_config=THDPlannerConfig(
                    min_chunk_tokens=args.min_chunk_tokens, max_chunks_per_rank=args.max_chunks_per_rank
                ),
            )
            variants[policy], samples[policy] = csa, []
            if not check_redistribution(csa):
                raise RuntimeError("row identity/inverse failed")
        if rank == 0:
            print("CASE", name, "selected", variants["planner"].selected_policy, flush=True)
        orders = []
        for i in range(args.warmups + args.repeats):
            order = policies[i % len(policies) :] + policies[: i % len(policies)]
            if i % 2:
                order = order[::-1]
            orders.append(order)
            for policy in order:
                sample = iteration(variants[policy])
                if i >= args.warmups:
                    samples[policy].append(sample)
            if rank == 0:
                print("ITER", name, i, {p: round(samples[p][-1]["total_ms"], 3) for p in policies if samples[p]}, flush=True)
        # Fresh metadata is measured after kernels/allocators are warm, in
        # rotated order. Retain cold measurements separately, without silently
        # substituting cache lookup time for fresh plan construction.
        cold_prepare = {p: dict(common_ms=c.common_prepare_ms, layout_ms=c.layout_prepare_ms) for p, c in variants.items()}
        prepare_orders = []
        for j in range(args.prepare_repeats):
            order = policies[j % len(policies) :] + policies[: j % len(policies)]
            if j % 2:
                order = order[::-1]
            prepare_orders.append(order)
            for p in order:
                dist.barrier(device_ids=[torch.cuda.current_device()])
                preparation.setdefault(p, []).append(variants[p].prepare_metadata())
        reference = {k: v.detach().cpu() for k, v in tensors(variants["legacy-contiguous"]).items()}
        # Loss is normalized by global tokens in FullCSA. Reduce its scalar
        # separately; selection routing must not change the objective.
        losses = torch.stack([variants[p].last["loss"].detach() for p in policies])
        dist.all_reduce(losses)
        loss_values = losses.cpu().tolist()
        okay = True
        for policy, csa in variants.items():
            errors = {k: compare_cpu_reference(v, reference[k]) for k, v in tensors(csa).items()}
            ref = check_reference(csa) if args.check_only else None
            okay &= all(e["finite"] and e["relative_rms"] < 0.01 for e in errors.values())
            loss = loss_values[policies.index(policy)]
            loss_delta = abs(loss - loss_values[0])
            okay &= ref is None or ref["pass_"]
            okay &= math.isfinite(loss) and loss_delta < 1e-5
            checks[policy] = dict(
                errors=errors,
                global_loss=loss,
                loss_abs_diff=loss_delta,
                fp32=ref,
                selected_policy=csa.selected_policy,
                cold_prepare=cold_prepare[policy],
                prepare_samples=preparation[policy],
                source_rows=csa.source_n,
                selection_rows=csa.selection_layout.local_tokens,
                selection_spans=len(csa.selection_layout.spans),
                workspace_bytes=0
                if csa.indexer.select_plan is None
                else csa.indexer.select_plan.candidate_workspace_floats * 4,
            )
        records = [None] * world
        dist.all_gather_object(records, dict(rank=rank, samples=samples, checks=checks, okay=okay))
        if rank == 0:
            summary = {}
            for policy in policies:
                step = statistics.median(
                    max(r["samples"][policy][i]["total_ms"] for r in records) for i in range(args.repeats)
                )
                prepare_samples = [
                    max(sum(r["checks"][policy]["prepare_samples"][j].values()) for r in records)
                    for j in range(args.prepare_repeats)
                ]
                prepare = statistics.median(prepare_samples)
                summary[policy] = dict(
                    step_ms=step,
                    prepare_ms=prepare,
                    one_use_ms=step + prepare,
                    prepare_min_ms=min(prepare_samples),
                    prepare_max_ms=max(prepare_samples),
                )
            report = dict(
                case=name,
                lengths=lengths,
                world_size=world,
                orders=orders,
                prepare_orders=prepare_orders,
                summary=summary,
                ranks=records,
            )
            (args.output_dir / f"{name}.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
            print("RESULT", name, json.dumps(summary), flush=True)
        if not all(r["okay"] for r in records):
            raise RuntimeError("numerical qualification failed; result retained")
        del variants, csa, reference, records
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
