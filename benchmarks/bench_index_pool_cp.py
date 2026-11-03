# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare gather-then-pool with cuLA's distributed pool-before-gather path."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cula.hybrid import (
    IndexPoolPlan,
    IndexPoolShardPlan,
    cudnn_index_pool,
    cudnn_index_pool_cp,
)


class _GatherReferenceRows(torch.autograd.Function):
    """Reference raw-row all-gather with FP32 gradient reduction."""

    @staticmethod
    def forward(ctx, local, group, offsets, rank):
        counts = tuple(right - left for left, right in zip(offsets, offsets[1:]))
        if any(count != counts[0] for count in counts[1:]):
            raise ValueError("the benchmark reference requires equal token shards")
        output = local.new_empty((sum(counts), *local.shape[1:]))
        if hasattr(dist, "all_gather_single"):
            dist.all_gather_single(output, local.contiguous(), group=group)
        else:
            dist.all_gather_into_tensor(output, local.contiguous(), group=group)
        ctx.group = group
        ctx.local_count = counts[rank]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        local_grad = torch.empty(
            (ctx.local_count, *grad_output.shape[1:]),
            dtype=torch.float32,
            device=grad_output.device,
        )
        source = grad_output.float().contiguous()
        if hasattr(dist, "reduce_scatter_single"):
            dist.reduce_scatter_single(local_grad, source, group=ctx.group)
        else:
            dist.reduce_scatter_tensor(local_grad, source, group=ctx.group)
        return local_grad.to(grad_output.dtype), None, None, None


def _boundaries(total: int, short_long_mix: bool) -> tuple[int, ...]:
    if not short_long_mix:
        return (0, total)
    short_total = 20 * 1024
    remaining = total - short_total
    if remaining <= 0 or remaining % 2:
        raise ValueError("20-short+2-long requires an even positive long-sequence remainder")
    lengths = [1024] * 20 + [remaining // 2] * 2
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return tuple(result)


def _measure_once(callable_) -> float:
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    del result
    return elapsed


def _measure_pair(left, right, repeats: int) -> tuple[list[float], list[float]]:
    """Interleave paths and reverse order to limit drift and ordering bias."""

    left_samples = []
    right_samples = []
    for repeat in range(repeats):
        if repeat % 2:
            right_samples.append(_measure_once(right))
            left_samples.append(_measure_once(left))
        else:
            left_samples.append(_measure_once(left))
            right_samples.append(_measure_once(right))
    return left_samples, right_samples


def _measure_incremental_peak(callable_) -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = callable_()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del result
    return peak


def _local_backward_oracle(kv, score, ape, grad_blocks, plan):
    members = []
    for start, end in zip(plan.cu_seqlens, plan.cu_seqlens[1:]):
        complete_end = start + (end - start) // plan.ratio * plan.ratio
        if complete_end > start:
            members.append(
                torch.arange(start, complete_end, device=kv.device).view(-1, plan.ratio)
            )
    members = torch.cat(members)
    grouped_kv = kv.index_select(0, members.flatten()).view(
        -1, plan.ratio, plan.head_dim
    ).float()
    grouped_score = score.index_select(0, members.flatten()).view_as(grouped_kv).float()
    weights = torch.softmax(grouped_score + ape.float().unsqueeze(0), dim=1)
    dy = grad_blocks.float().unsqueeze(1)
    dweight = grouped_kv * dy
    dscore_grouped = weights * (
        dweight - (dweight * weights).sum(dim=1, keepdim=True)
    )
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dscore = torch.zeros_like(score, dtype=torch.float32)
    dkv.index_copy_(0, members.flatten(), (weights * dy).flatten(0, 1))
    dscore.index_copy_(0, members.flatten(), dscore_grouped.flatten(0, 1))
    return dkv, dscore, dscore_grouped.sum(dim=0)


def _errors(actual, oracle):
    error = actual.float() - oracle.float()
    return {
        "max_abs": float(error.abs().max()),
        "rmse": float(error.square().mean().sqrt()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-tokens", type=int, default=1 << 17)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--thd-20s2l", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    boundaries = _boundaries(args.global_tokens, args.thd_20s2l)
    shard_plan = IndexPoolShardPlan.from_global_cu_seqlens(
        boundaries,
        rank=rank,
        world_size=world_size,
        head_dim=args.head_dim,
        ratio=args.ratio,
    )
    if not shard_plan.all_shards_direct or shard_plan.local_plan is None:
        raise ValueError("benchmark geometry must use the aligned pool-before-gather path")

    torch.manual_seed(1104 + rank)
    kv = torch.randn(
        shard_plan.local_tokens,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    score = torch.randn_like(kv)
    torch.manual_seed(47)
    ape = torch.randn(args.ratio, args.head_dim, dtype=torch.float32, device=device)
    local_cu, local_cb = shard_plan.local_plan.materialize(device)
    global_plan = IndexPoolPlan.from_cu_seqlens(
        boundaries,
        head_dim=args.head_dim,
        ratio=args.ratio,
    )
    global_cu, global_cb = global_plan.materialize(device)
    token_offsets = tuple(
        shard * args.global_tokens // world_size for shard in range(world_size + 1)
    )

    def reference(kv_, score_, ape_):
        global_kv = _GatherReferenceRows.apply(kv_, dist.group.WORLD, token_offsets, rank)
        global_score = _GatherReferenceRows.apply(
            score_, dist.group.WORLD, token_offsets, rank
        )
        return cudnn_index_pool(
            global_kv,
            global_score,
            ape_,
            global_cu,
            global_cb,
            plan=global_plan,
            deterministic=False,
        )

    def candidate(kv_, score_, ape_):
        return cudnn_index_pool_cp(
            kv_,
            score_,
            ape_,
            local_cu,
            local_cb,
            plan=shard_plan,
            group=dist.group.WORLD,
            deterministic=False,
        )

    ref_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (kv, score, ape))
    opt_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (kv, score, ape))
    ref_out = reference(*ref_inputs)
    opt_out = candidate(*opt_inputs)
    torch.manual_seed(9000 + rank)
    grad_out = torch.randn_like(ref_out)
    ref_grads = list(torch.autograd.grad(ref_out, ref_inputs, grad_out))
    opt_grads = list(torch.autograd.grad(opt_out, opt_inputs, grad_out))
    total_grad_out = grad_out.float()
    dist.all_reduce(total_grad_out)
    local_begin = shard_plan.block_offsets[rank]
    local_end = shard_plan.block_offsets[rank + 1]
    oracle_grads = list(
        _local_backward_oracle(
            kv,
            score,
            ape,
            total_grad_out[local_begin:local_end],
            shard_plan.local_plan,
        )
    )
    # APE is replicated, so compare the CP-global parameter gradient.
    dist.all_reduce(ref_grads[2])
    dist.all_reduce(opt_grads[2])
    dist.all_reduce(oracle_grads[2])
    correctness = {
        "out_bitwise": bool(torch.equal(ref_out, opt_out)),
        "dk_bitwise": bool(torch.equal(ref_grads[0], opt_grads[0])),
        "dk_max_abs": float((ref_grads[0].float() - opt_grads[0].float()).abs().max()),
        "dscore_bitwise": bool(torch.equal(ref_grads[1], opt_grads[1])),
        "dscore_max_abs": float(
            (ref_grads[1].float() - opt_grads[1].float()).abs().max()
        ),
        "dape_max_abs": float((ref_grads[2] - opt_grads[2]).abs().max()),
        "dape_max_rel": float(
            (
                (ref_grads[2] - opt_grads[2]).abs()
                / ref_grads[2].abs().clamp_min(1e-12)
            ).max()
        ),
        "reference_vs_fp32_oracle": {
            "dk": _errors(ref_grads[0], oracle_grads[0]),
            "dscore": _errors(ref_grads[1], oracle_grads[1]),
            "dape": _errors(ref_grads[2], oracle_grads[2]),
        },
        "candidate_vs_fp32_oracle": {
            "dk": _errors(opt_grads[0], oracle_grads[0]),
            "dscore": _errors(opt_grads[1], oracle_grads[1]),
            "dape": _errors(opt_grads[2], oracle_grads[2]),
        },
    }

    with torch.no_grad():
        reference(kv, score, ape)
        candidate(kv, score, ape)
        torch.cuda.synchronize()
        ref_samples, opt_samples = _measure_pair(
            lambda: reference(kv, score, ape),
            lambda: candidate(kv, score, ape),
            args.repeats,
        )

    def reference_forward_backward():
        inputs = tuple(t.detach().requires_grad_(True) for t in (kv, score, ape))
        output = reference(*inputs)
        return torch.autograd.grad(output, inputs, grad_out)

    def candidate_forward_backward():
        inputs = tuple(t.detach().requires_grad_(True) for t in (kv, score, ape))
        output = candidate(*inputs)
        return torch.autograd.grad(output, inputs, grad_out)

    reference_forward_backward()
    candidate_forward_backward()
    torch.cuda.synchronize()
    ref_peak = _measure_incremental_peak(reference_forward_backward)
    opt_peak = _measure_incremental_peak(candidate_forward_backward)
    ref_fb_samples, opt_fb_samples = _measure_pair(
        reference_forward_backward,
        candidate_forward_backward,
        args.repeats,
    )

    local = torch.tensor(
        [
            ref_samples,
            opt_samples,
            ref_fb_samples,
            opt_fb_samples,
            [ref_peak] * args.repeats,
            [opt_peak] * args.repeats,
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = torch.empty(
        world_size * 6 * args.repeats,
        dtype=torch.float64,
        device=device,
    )
    if hasattr(dist, "all_gather_single"):
        dist.all_gather_single(gathered, local.flatten())
    else:
        dist.all_gather_into_tensor(gathered, local.flatten())
    if rank == 0:
        samples = gathered.cpu().view(world_size, 6, args.repeats)
        ref_critical = samples[:, 0].max(dim=0).values.tolist()
        opt_critical = samples[:, 1].max(dim=0).values.tolist()
        ref_fb_critical = samples[:, 2].max(dim=0).values.tolist()
        opt_fb_critical = samples[:, 3].max(dim=0).values.tolist()
        ref_peak_bytes = int(samples[:, 4, 0].max())
        opt_peak_bytes = int(samples[:, 5, 0].max())
        ref_median = statistics.median(ref_critical)
        opt_median = statistics.median(opt_critical)
        ref_fb_median = statistics.median(ref_fb_critical)
        opt_fb_median = statistics.median(opt_fb_critical)
        print(
            json.dumps(
                {
                    "global_tokens": args.global_tokens,
                    "world_size": world_size,
                    "ratio": args.ratio,
                    "sequence_lengths": [
                        right - left for left, right in zip(boundaries, boundaries[1:])
                    ],
                    "raw_collective_bytes_per_rank": 2 * kv.numel() * kv.element_size(),
                    "pooled_collective_bytes_per_rank": (
                        shard_plan.local_blocks * args.head_dim * kv.element_size()
                    ),
                    "correctness": correctness,
                    "gather_then_pool_median_ms": ref_median,
                    "pool_then_gather_median_ms": opt_median,
                    "speedup": ref_median / opt_median,
                    "gather_then_pool_forward_backward_median_ms": ref_fb_median,
                    "pool_then_gather_forward_backward_median_ms": opt_fb_median,
                    "forward_backward_speedup": ref_fb_median / opt_fb_median,
                    "gather_then_pool_incremental_peak_bytes": ref_peak_bytes,
                    "pool_then_gather_incremental_peak_bytes": opt_peak_bytes,
                    "incremental_peak_reduction": ref_peak_bytes / opt_peak_bytes,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
