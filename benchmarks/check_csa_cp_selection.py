# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native distributed checks of source order, subgroups and retained results."""

import argparse
import datetime
import json
import math
import os
from itertools import accumulate
from pathlib import Path

import torch
import torch.distributed as dist

from cula.hybrid import CsaCPSelectionPlan, CsaCPSelectionWorkspace, cudnn_csa_select_cp


def canonical(result):
    ids, order = result.block_ids.sort(-1)
    scores = result.scores.gather(1, order).masked_fill(ids < 0, 0)
    return ids, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--reuse-layers", type=int, default=3, help="Distinct sequential inputs sharing a plan/workspace")
    args = parser.parse_args()
    if args.reuse_layers < 2:
        parser.error("reuse-layers must be at least two")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
    world = dist.get_world_size()
    if world < 4 or world % 2:
        raise ValueError("launch an even number of >= 4 ranks")
    groups = [dist.new_group(list(range(parity, world, 2))) for parity in (0, 1)]
    group = groups[dist.get_rank() % 2]
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    records = []
    # Aligned but unequal source lengths, document-crossing shards, and empty
    # zigzag destinations. All policies must restore the same native result.
    for lengths in ([4] * (2 * size), [4, 4 * (size + 1)], [4096], [4] * 20 + [984] * 2):
        cu = (0, *accumulate(lengths))
        plan = CsaCPSelectionPlan.from_cu_seqlens(cu, rank=rank, world_size=size, top_k=512, policy="contiguous")
        torch.manual_seed(20261116 + dist.get_rank())
        n = plan.exchange.source_rows
        qi = torch.randn(n, 64, 128, dtype=torch.bfloat16, device=device)
        dtype = torch.bfloat16 if args.weights_dtype == "bf16" else torch.float32
        weights = torch.randn(n, 64, dtype=dtype, device=device) / 64
        # Distinct replicated key sets per subgroup catch use of WORLD.
        torch.manual_seed(713 + dist.get_rank() % 2)
        keys = torch.randn(cu[-1] // 4, 1, 128, dtype=torch.bfloat16, device=device)
        baseline = None
        for policy in ("contiguous", "zigzag"):
            plan = CsaCPSelectionPlan.from_cu_seqlens(
                cu, rank=rank, world_size=size, top_k=512, policy=policy, max_tokens_per_rank=cu[-1]
            )
            workspace = CsaCPSelectionWorkspace.from_plan(plan, device, group=group)
            result = cudnn_csa_select_cp(qi, keys, weights, workspace=workspace)
            ids, scores = canonical(result)
            saved = [(result, result.block_ids.clone(), result.scores.clone())]
            reuse_ids_equal, reuse_error = True, 0.0
            for layer in range(1, args.reuse_layers):
                # Different layer inputs must not reuse the previous layer's
                # sparse selection. Compare with a fresh workspace, retaining
                # every earlier result as a training stack would for backward.
                layer_q, layer_keys, layer_weights = qi.roll(layer, 0), keys.roll(-layer, 0), weights.roll(-layer, 0)
                reused = cudnn_csa_select_cp(layer_q, layer_keys, layer_weights, workspace=workspace)
                fresh_workspace = CsaCPSelectionWorkspace.from_plan(plan, device, group=group)
                fresh = cudnn_csa_select_cp(layer_q, layer_keys, layer_weights, workspace=fresh_workspace)
                reused_ids, reused_scores = canonical(reused)
                fresh_ids, fresh_scores = canonical(fresh)
                reuse_ids_equal &= torch.equal(reused_ids, fresh_ids)
                error = ((reused_scores - fresh_scores).norm() / fresh_scores.norm().clamp_min(1e-12)).item()
                reuse_error = max(reuse_error, error) if math.isfinite(error) else math.inf
                saved.append((reused, reused.block_ids.clone(), reused.scores.clone()))
            retained = all(
                torch.equal(x.block_ids, ids_copy) and torch.equal(x.scores, scores_copy) for x, ids_copy, scores_copy in saved
            )
            if baseline is None:
                baseline = ids.clone(), scores.clone()
            ids_equal = torch.equal(ids, baseline[0])
            delta = (scores - baseline[1]).float()
            relative = (delta.norm() / baseline[1].norm().clamp_min(1e-12)).item()
            okay = (
                retained
                and ids_equal
                and reuse_ids_equal
                and reuse_error < 1e-5
                and bool(torch.isfinite(scores).all())
                and relative < 1e-5
            )
            records.append(
                dict(
                    lengths=lengths,
                    policy=policy,
                    weights_dtype=args.weights_dtype,
                    source_rows=n,
                    target_rows=plan.exchange.target_rows,
                    retained=retained,
                    reuse_layers=args.reuse_layers,
                    reuse_ids_equal=reuse_ids_equal,
                    reuse_max_score_relative_rms=reuse_error if math.isfinite(reuse_error) else None,
                    ids_equal=ids_equal,
                    score_relative_rms=relative if math.isfinite(relative) else None,
                    okay=okay,
                )
            )
    all_records = [None] * world
    dist.all_gather_object(all_records, records)
    if dist.get_rank() == 0:
        out = Path(os.environ.get("CSA_CHECK_OUTPUT", "csa-cp-checks.json"))
        out.write_text(json.dumps(all_records, indent=2, allow_nan=False) + "\n")
        print("CHECKS", sum(len(r) for r in all_records), "PASS", all(x["okay"] for r in all_records for x in r), flush=True)
    if not all(x["okay"] for r in all_records for x in r):
        raise RuntimeError("CSA CP selection qualification failed")
    for g in groups:
        if g != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(g)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
