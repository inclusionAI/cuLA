# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Paired KDA check for globally document-local CP shards, including backward.

The baseline conservatively forces the original boundary route. This compares
complete output/gradients on identical operands, retaining the baseline on CPU.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
from dataclasses import replace
from itertools import accumulate
from pathlib import Path

import torch
import torch.distributed as dist

from benchmarks.bench_full_csa_cp import compare_cpu_reference, json_safe
from cula.distributed import AllGatherKimiCPTransport, KimiCPPlan, cudnn_kda_cp


def paired_samples(run, *, warmups, repeats):
    """Warm both routes, then counterbalance each measured pair on all ranks."""
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive")
    samples = {"baseline": [], "candidate": []}
    for warming, count in ((True, warmups), (False, repeats)):
        for pair_index in range(count):
            order = ("baseline", "candidate") if pair_index % 2 == 0 else ("candidate", "baseline")
            for position, label in enumerate(order):
                sample = run(label)
                if not warming:
                    samples[label].append(dict(sample, pair_index=pair_index, execution_order=list(order), position=position))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2 or (1 << 20) % world:
        raise ValueError("require CP >= 2 dividing 1M")
    n = (1 << 20) // world
    cases = [(f"uniform-{length}", [length] * ((1 << 20) // length), False) for length in (1024, 4096, 16384)]
    cases += [
        ("near-uniform", [15360, 17408] * 32, False),
        ("weak-decay-uniform", [16384] * 64, True),
        ("isolated-ranks-with-crossing-peer", [n] * (world - 2) + [2 * n], False),
    ]
    results = []
    for name, lengths, weak in cases:
        plan = KimiCPPlan.from_global_cu_seqlens((0, *accumulate(lengths)), rank=rank, world_size=world)
        transport = AllGatherKimiCPTransport.prepare(plan, heads=32, key_dim=128, value_dim=128, device=device)
        cu = torch.tensor(plan.local_cu_seqlens, dtype=torch.int32, device=device)
        torch.manual_seed(88171 + rank)
        q, k = [torch.nn.functional.normalize(torch.randn(n, 32, 128, device=device), dim=-1).bfloat16() for _ in range(2)]
        v = torch.randn_like(q)
        g = (
            (-0.0001 - 0.0004 * torch.rand(n, 32, 128, device=device))
            if weak
            else (-0.01 - 0.04 * torch.rand(n, 32, 128, device=device))
        )
        beta = torch.sigmoid(torch.randn(n, 32, device=device))
        inputs = [x.requires_grad_() for x in (q, k, v, g, beta)]
        dout = torch.randn_like(v)
        plans = {"baseline": replace(plan, requires_boundary_exchange=True), "candidate": plan}

        def timed_iteration(label, inputs=inputs, cu=cu, transport=transport, dout=dout):
            for x in inputs:
                x.grad = None
            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin, forward, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            begin.record()
            out = cudnn_kda_cp(*inputs, cu, plan=plans[label], group=dist.group.WORLD, boundary_transport=transport)
            forward.record()
            out.backward(dout)
            end.record()
            end.synchronize()
            sample = dict(
                total_ms=begin.elapsed_time(end),
                fwd_ms=begin.elapsed_time(forward),
                bwd_ms=forward.elapsed_time(end),
                allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            del out
            return sample

        samples = paired_samples(timed_iteration, warmups=args.warmups, repeats=args.repeats)
        del timed_iteration
        # Run the numerical comparison separately, after both routes' timings.
        # CPU copies and reference comparisons cannot perturb a measured pair.
        reference = None
        for label, active_plan in plans.items():
            for x in inputs:
                x.grad = None
            out = cudnn_kda_cp(*inputs, cu, plan=active_plan, group=dist.group.WORLD, boundary_transport=transport)
            out.backward(dout)
            tensors = dict(output=out, **{key: x.grad for key, x in zip(("dq", "dk", "dv", "dg", "dbeta"), inputs)})
            errors = {} if reference is None else {key: compare_cpu_reference(x, reference[key]) for key, x in tensors.items()}
            if reference is None:
                reference = {key: x.detach().cpu() for key, x in tensors.items()}
            record = dict(
                case=name,
                label=label,
                rank=rank,
                requires_boundary_exchange=active_plan.requires_boundary_exchange,
                intrinsic_requires_boundary_exchange=plan.requires_boundary_exchange,
                samples=samples[label],
                errors=errors,
            )
            gathered = [None] * world
            dist.all_gather_object(gathered, record)
            passed = all(e["finite"] and e["max_abs"] == 0 for r in gathered for e in r["errors"].values())
            if rank == 0:
                results.append(
                    dict(
                        case=name,
                        label=label,
                        ranks=gathered,
                        passed=passed,
                        total_ms=statistics.median(
                            max(r["samples"][i]["total_ms"] for r in gathered) for i in range(args.repeats)
                        ),
                    )
                )
                print(name, label, results[-1]["total_ms"], "exact", passed, flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        json_safe(
                            dict(
                                world_size=world,
                                timing_protocol="paired-alternating-v1",
                                source_sha256=hashlib.sha256(
                                    (Path(__file__).resolve().parents[1] / "cula/distributed/kda.py").read_bytes()
                                ).hexdigest(),
                                results=results,
                            )
                        ),
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
            if not passed:
                raise RuntimeError("independent-shard path changed output or gradients; see saved errors")
            del out, tensors
            for x in inputs:
                x.grad = None
        del reference, inputs, q, k, v, g, beta, dout, transport, cu
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
