# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pool context-parallel index keys before exchanging compressed blocks."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from cula.hybrid.index_pool import _load_cudnn_index_pool_ops, _validate_inputs
from cula.hybrid.plan import IndexPoolShardPlan


def _reduce_scatter_tensor(
    output: torch.Tensor,
    tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup | None,
) -> None:
    """Use the public collective when available, retaining older torch support."""

    if hasattr(dist, "reduce_scatter_single"):
        dist.reduce_scatter_single(output, tensor, group=group)
    elif hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(output, tensor, group=group)
    else:  # pragma: no cover - compatibility with older PyTorch releases
        dist._reduce_scatter_base(output, tensor, group=group)


def _all_gather_block_rows(
    local: torch.Tensor,
    *,
    group: dist.ProcessGroup | None,
    block_offsets: tuple[int, ...],
    rank: int,
) -> torch.Tensor:
    """Gather compressed rows while padding only an uneven compressed axis."""

    counts = tuple(right - left for left, right in zip(block_offsets, block_offsets[1:]))
    if all(count == counts[0] for count in counts[1:]):
        output = local.new_empty((sum(counts), *local.shape[1:]))
        if hasattr(dist, "all_gather_single"):
            dist.all_gather_single(output, local.contiguous(), group=group)
        else:  # pragma: no cover - compatibility with older PyTorch releases
            dist.all_gather_into_tensor(output, local.contiguous(), group=group)
        return output

    max_count = max(counts)
    padded = local.new_zeros((max_count, *local.shape[1:]))
    padded[: counts[rank]].copy_(local)
    gathered = local.new_empty((len(counts), max_count, *local.shape[1:]))
    flat_gathered = gathered.flatten(0, 1)
    if hasattr(dist, "all_gather_single"):
        dist.all_gather_single(flat_gathered, padded, group=group)
    else:  # pragma: no cover - compatibility with older PyTorch releases
        dist.all_gather_into_tensor(flat_gathered, padded, group=group)

    output = local.new_empty((sum(counts), *local.shape[1:]))
    for owner, count in enumerate(counts):
        output[block_offsets[owner] : block_offsets[owner + 1]].copy_(gathered[owner, :count])
    return output


class _CudnnIndexPoolCP(torch.autograd.Function):
    """Distributed compressor with a precision-preserving reverse exchange."""

    @staticmethod
    def forward(
        ctx,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        plan: IndexPoolShardPlan,
        group,
        deterministic: bool,
    ) -> torch.Tensor:
        forward, _ = _load_cudnn_index_pool_ops()
        local_plan = plan.local_plan
        if local_plan is None:  # Validated by the public entry point.
            raise RuntimeError("the direct local pooling plan is unavailable")
        result = forward(
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            ratio=local_plan.ratio,
            head_dim=local_plan.head_dim,
            coff=local_plan.coff,
            total_comp=local_plan.total_comp,
        )
        local_blocks = result["out"]
        ctx.plan = plan
        ctx.group = group
        ctx.deterministic = deterministic
        ctx.save_for_backward(kv, score, ape, cu_seqlens, cu_seqlens_comp)
        if plan.world_size == 1:
            return local_blocks
        return _all_gather_block_rows(
            local_blocks,
            group=group,
            block_offsets=plan.block_offsets,
            rank=plan.rank,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        kv, score, ape, cu_seqlens, cu_seqlens_comp = ctx.saved_tensors
        plan = ctx.plan
        local_plan = plan.local_plan
        if local_plan is None:  # Validated by the public entry point.
            raise RuntimeError("the direct local pooling plan is unavailable")
        _, backward = _load_cudnn_index_pool_ops()

        if plan.world_size == 1:
            result = backward(
                kv,
                score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                grad_output.contiguous(),
                ratio=local_plan.ratio,
                head_dim=local_plan.head_dim,
                coff=local_plan.coff,
                deterministic=ctx.deterministic,
            )
            return (
                result["grad_kv"],
                result["grad_score"],
                result["grad_ape"],
                None,
                None,
                None,
                None,
                None,
            )

        # Each rank's loss supplies one gradient for every global compressed
        # block. Route each owner its block slice without merging loss ranks.
        # Keeping those BF16 slices separate is essential: merging them first
        # would round the CP sum before the nonlinear compressor backward.
        counts = plan.block_counts
        local_count = counts[plan.rank]
        received = grad_output.new_empty((plan.world_size * local_count, *grad_output.shape[1:]))
        dist.all_to_all_single(
            received,
            grad_output.contiguous(),
            output_split_sizes=[local_count] * plan.world_size,
            input_split_sizes=list(counts),
            group=ctx.group,
        )
        received = received.view(plan.world_size, local_count, *grad_output.shape[1:])

        grad_kv = torch.zeros_like(kv, dtype=torch.float32)
        grad_score = torch.zeros_like(score, dtype=torch.float32)
        grad_ape_by_loss = torch.empty((plan.world_size, *ape.shape), dtype=torch.float32, device=ape.device)
        for loss_rank in range(plan.world_size):
            result = backward(
                kv,
                score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                received[loss_rank].contiguous(),
                ratio=local_plan.ratio,
                head_dim=local_plan.head_dim,
                coff=local_plan.coff,
                deterministic=ctx.deterministic,
            )
            grad_kv.add_(result["grad_kv"])
            grad_score.add_(result["grad_score"])
            grad_ape_by_loss[loss_rank].copy_(result["grad_ape"])

        # APE is replicated. Return this loss rank's gradient summed over all
        # input owners, matching gather-then-pool autograd semantics before any
        # framework-level data-parallel gradient synchronization.
        grad_ape = torch.empty_like(ape, dtype=torch.float32)
        _reduce_scatter_tensor(
            grad_ape,
            grad_ape_by_loss.flatten(0, 1).contiguous(),
            group=ctx.group,
        )
        return (
            grad_kv.to(kv.dtype),
            grad_score.to(score.dtype),
            grad_ape,
            None,
            None,
            None,
            None,
            None,
        )


def cudnn_index_pool_cp(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    *,
    plan: IndexPoolShardPlan,
    group: dist.ProcessGroup | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """Pool local THD rows, then gather only compressed block rows.

    Complete ratio-sized blocks are owned by the rank containing their first
    token. The current execution path requires every CP seam to already be a
    block boundary; :class:`IndexPoolShardPlan` detects that condition without
    reading device metadata. Misaligned shards are rejected until the explicit
    neighbor-halo executor is selected, rather than silently pooling unrelated
    tokens across a rank-local seam.

    The collective is differentiable. Backward routes compressed block-gradient
    slices to their input owner without first merging BF16 loss-rank gradients,
    runs the local cuDNN backward once per loss rank, accumulates token gradients
    in FP32, and reduce-scatters FP32 APE gradients. This preserves the original
    gather-then-pool numerical order instead of trading precision for one fewer
    kernel launch. Rank block counts may differ; only the compressed axis uses
    uneven collective splits. ``coff=2`` overlapping windows are intentionally
    excluded because they need a left-window halo even when the owned block
    boundary itself is aligned.
    """

    if not isinstance(plan, IndexPoolShardPlan):
        raise TypeError("plan must be an IndexPoolShardPlan")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
    if plan.coff != 1 and plan.world_size > 1:
        raise NotImplementedError(
            "distributed pool-before-gather currently supports coff=1; coff=2 requires an overlapping-window halo"
        )
    plan.validate_group(group)
    if not plan.all_shards_direct or not plan.direct_local_pool:
        raise NotImplementedError(
            "this CP partition has a compressed block crossing a rank seam; "
            "use the neighbor-halo executor or gather token rows before pooling"
        )
    if any(count == 0 for count in plan.block_counts):
        raise NotImplementedError(
            "distributed pool-before-gather requires every rank to own at least one complete pooling block"
        )
    if plan.local_plan is None:
        raise RuntimeError("the direct local pooling plan is unavailable")
    if kv.shape[0] != plan.local_tokens:
        raise ValueError(f"kv must contain this rank's {plan.local_tokens} token rows, got {kv.shape[0]}")

    _validate_inputs(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        plan=plan.local_plan,
    )
    return _CudnnIndexPoolCP.apply(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        plan,
        group,
        deterministic,
    )


__all__ = ["cudnn_index_pool_cp"]
