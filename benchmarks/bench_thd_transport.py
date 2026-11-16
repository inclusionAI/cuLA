# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate selection transport independently of KDA/selection arrival skew.

The caller supplies process-group topology. This uses the actual eager
redistributor and Q/weights/IDs/scores dtypes, with no attention kernels.
"""

import argparse
import datetime
import gc
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from cula.distributed import THDRankLayout, THDRedistributionPlan, THDRedistributor, THDSpan


def transport_layout(world, rows, ranks_per_node, kind):
    if world < 2 or rows < world or rows % world or rows % 4 or ranks_per_node < 1 or world % ranks_per_node:
        raise ValueError("need complete nodes and row counts divisible by CP")
    assignments = [[] for _ in range(world)]
    for source in range(world):
        if kind in ("local_quarter", "local_half", "local_three_quarters"):
            if ranks_per_node < 2 or rows % 16:
                raise ValueError("locality fixtures need multiple local ranks and 16-row alignment")
            fraction = {"local_quarter": 1, "local_half": 2, "local_three_quarters": 3}[kind]
            peer = source // ranks_per_node * ranks_per_node + (source + 1) % ranks_per_node
            cursor = source * rows
            for rank, count in ((source, rows // 4 * fraction), (peer, rows // 4 * (4 - fraction))):
                assignments[rank].append(THDSpan(0, cursor, cursor + count, 0, world * rows, rank))
                cursor += count
            continue
        if kind in ("mixed_dense_quarter", "mixed_dense_three_quarters"):
            if ranks_per_node == world or rows % (4 * ranks_per_node):
                raise ValueError("mixed validation needs multiple nodes and quarter-node aligned rows")
            local_start = source // ranks_per_node * ranks_per_node
            remote_start = (local_start + ranks_per_node) % world
            remote_rows = rows // 4 * (1 if kind == "mixed_dense_quarter" else 3)
            cursor = source * rows
            for node_start, count in ((local_start, rows - remote_rows), (remote_start, remote_rows)):
                for rank in range(node_start, node_start + ranks_per_node):
                    end = cursor + count // ranks_per_node
                    assignments[rank].append(THDSpan(0, cursor, end, 0, world * rows, rank))
                    cursor = end
            continue
        if kind in ("intra_pair", "inter_pair"):
            peer = 1 if kind == "intra_pair" else ranks_per_node
            if peer >= world or (kind == "intra_pair" and ranks_per_node < 2):
                raise ValueError("pair fixture needs two ranks on the declared link")
            destinations = [peer if source == 0 else 0 if source == peer else source]
        elif kind in ("intra", "intra_dense"):
            node_start = source // ranks_per_node * ranks_per_node
            destinations = (
                list(range(node_start, node_start + ranks_per_node))
                if kind == "intra_dense"
                else [node_start + (source + 1) % ranks_per_node]
            )
            if ranks_per_node == 1:
                raise ValueError("intra-node calibration needs at least two ranks per node")
        elif kind in ("inter", "inter_dense"):
            if ranks_per_node == world:
                raise ValueError("inter-node calibration needs at least two nodes")
            next_node = ((source // ranks_per_node + 1) * ranks_per_node) % world
            destinations = (
                list(range(next_node, next_node + ranks_per_node))
                if kind == "inter_dense"
                else [(source + ranks_per_node) % world]
            )
        elif kind == "striped":
            destinations = list(range(world))
        else:
            raise ValueError(f"unknown transport fixture: {kind}")
        for i, rank in enumerate(destinations):
            start = source * rows + i * rows // len(destinations)
            end = source * rows + (i + 1) * rows // len(destinations)
            assignments[rank].append(THDSpan(0, start, end, 0, world * rows, rank))
    return tuple(THDRankLayout(rank, tuple(spans)) for rank, spans in enumerate(assignments))


def ordered(ranges):
    cursor = 0
    for a, z in ranges:
        if a != cursor:
            return False
        cursor += z - a
    return True


def features(layouts, rows, rpn):
    plans = [
        THDRedistributionPlan.build(layouts, tuple(i * rows for i in range(len(layouts) + 1)), r) for r in range(len(layouts))
    ]
    edges = {(r, t): n for r, p in enumerate(plans) for t, n in enumerate(p.send_counts) if r != t and n}
    intra_out = [0] * len(plans)
    inter_out = intra_out.copy()
    intra_in = intra_out.copy()
    inter_in = intra_out.copy()
    node_out = [0] * (len(plans) // rpn)
    node_in = node_out.copy()
    degree_out = [0] * len(plans)
    degree_in = degree_out.copy()
    for (s, t), n in edges.items():
        a, b = s // rpn, t // rpn
        degree_out[s] += 1
        degree_in[t] += 1
        if a == b:
            intra_out[s] += n
            intra_in[t] += n
        else:
            inter_out[s] += n
            inter_in[t] += n
            node_out[a] += n
            node_in[b] += n
    return dict(
        rank_intra_rows=intra_out + intra_in,
        rank_inter_rows=inter_out + inter_in,
        rank_local_rows=[p.send_counts[r] for r, p in enumerate(plans)] * 2,
        peer_rows=max(edges.values(), default=0),
        node_rows=max(node_out + node_in),
        peer_degree=max(degree_out + degree_in),
        copies=max(
            (0 if ordered(p.send_ranges) else p.source_rows) + (0 if ordered(p.receive_ranges) else p.target_rows)
            for p in plans
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--ranks-per-node", type=int, required=True)
    parser.add_argument("--rows-per-rank", nargs="+", type=int, default=[256, 4096, 32768, 65536])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks_per_node < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("require positive ranks-per-node/repeats and nonnegative warmups")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    world, rank = dist.get_world_size(), dist.get_rank()
    if world < 2 or world % args.ranks_per_node:
        raise ValueError("require CP >= 2 and complete nodes")
    kinds = (
        ["intra_pair", "intra", "intra_dense", "local_quarter", "local_half", "local_three_quarters"]
        if args.ranks_per_node > 1
        else []
    )
    kinds += (["inter_pair", "inter", "inter_dense"] if world > args.ranks_per_node else []) + ["striped"]
    sizes = args.rows_per_rank
    if args.validation_only:
        sizes, kinds = [8192, 49152], ["mixed_dense_quarter", "mixed_dense_three_quarters"]
    # Validate every fixture before starting any timed GPU work.
    for rows in sizes:
        for kind in kinds:
            transport_layout(world, rows, args.ranks_per_node, kind)
    results = []
    for rows in sizes:
        for kind in kinds:
            layouts = transport_layout(world, rows, args.ranks_per_node, kind)
            plan = THDRedistributionPlan.build(layouts, tuple(r * rows for r in range(world + 1)), rank)
            exchange = THDRedistributor(plan, device, group=dist.group.WORLD)
            marker = torch.arange(rank * rows, (rank + 1) * rows, device=device)
            routed = exchange.distribute(marker)
            restored = exchange.restore(routed)
            expected = torch.cat([torch.arange(s.token_start, s.token_end, device=device) for s in layouts[rank].spans])
            okay = torch.tensor(int(torch.equal(marker, restored) and torch.equal(routed, expected)), device=device)
            dist.all_reduce(okay, op=dist.ReduceOp.MIN)
            if not okay.item():
                raise RuntimeError("transport calibration marker mismatch")
            # H64/D128 BF16 query, BF16 weights; Top-K512 int32 IDs and FP32 scores.
            source = [torch.zeros(rows, *shape, device=device, dtype=torch.bfloat16) for shape in [(64, 128), (64,)]]
            target = [torch.zeros(rows, 512, device=device, dtype=dtype) for dtype in (torch.int32, torch.float32)]
            samples = []
            for step in range(args.warmups + args.repeats):
                dist.barrier(device_ids=[local_rank])
                torch.cuda.synchronize()
                begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                outputs = [exchange.distribute(x) for x in source] + [exchange.restore(x) for x in target]
                end.record()
                end.synchronize()
                if step >= args.warmups:
                    samples.append(begin.elapsed_time(end))
                del outputs
            gathered = [None] * world
            dist.all_gather_object(gathered, samples)
            record = dict(
                kind=kind,
                rows=rows,
                features=features(layouts, rows, args.ranks_per_node),
                samples_ms=gathered,
                critical_median_ms=statistics.median(max(s[i] for s in gathered) for i in range(args.repeats)),
                marker_exact=True,
                heldout=args.validation_only or rows == 32768 or kind in ("striped", "local_half"),
            )
            results.append(record)
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(dict(world=world, ranks_per_node=args.ranks_per_node, records=results), indent=2) + "\n"
                )
                print(rows, kind, record["critical_median_ms"], flush=True)
            del source, target, marker, routed, expected, restored, exchange
            gc.collect()
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
