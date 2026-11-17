# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Prepared CSA indexer selection with adaptive CP routing and inverse mapping."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from cula._metadata import _range_indices, as_boundaries, balanced_shards
from cula.distributed.redistribute import THDRedistributionPlan, THDRedistributor
from cula.distributed.thd_planner import (
    THDPlannerConfig,
    THDRankLayout,
    THDSelectionCostModel,
    THDSpan,
    plan_thd_selection,
    zigzag_thd_layout,
)
from cula.hybrid.dsa import DsaSelectionPlan, DsaSelectionWorkspace, cudnn_dsa_select


@dataclass(frozen=True, slots=True)
class CsaCPSelectionPlan:
    """Host-only selection plan for one rank of a packed THD batch.

    Source tokens are contiguous, balanced in complete pooling blocks. All
    ranks must use identical global boundaries, policy, capacity and costs.
    Documents must be block-aligned; selection destinations may be empty.
    No CUDA metadata is read, hardware probed, or process group accessed here.
    Build once per packed layout and reuse across compatible CSA layers;
    the plan contains no activations or selected block IDs.
    """

    global_cu_seqlens: tuple[int, ...]
    source_offsets: tuple[int, ...]
    ranks: tuple[THDRankLayout, ...]
    policy: str
    evaluated_scores: tuple[tuple[str, float], ...]
    top_k: int
    block_size: int
    exchange: THDRedistributionPlan
    selection: DsaSelectionPlan | None
    key_ranges: tuple[tuple[int, int], ...]
    cost_model: THDSelectionCostModel | None

    @classmethod
    def from_cu_seqlens(
        cls,
        global_cu_seqlens: torch.Tensor | Sequence[int],
        *,
        rank: int,
        world_size: int,
        top_k: int,
        policy: str = "planner",
        cost_model: THDSelectionCostModel | None = None,
        max_tokens_per_rank: int | None = None,
        config: THDPlannerConfig | None = None,
        candidate_budget_bytes: int | None = None,
    ) -> CsaCPSelectionPlan:
        """Choose a layout, then prepare native selection-window metadata.

        ``planner`` requires explicit calibrated costs. Fixed ``contiguous``
        and complete per-document ``zigzag`` policies need no cost model.
        The default destination row cap is twice the largest source shard;
        it is separate from the native compact-logit scratch budget.
        """
        for name, value, minimum in (("world_size", world_size, 1), ("rank", rank, 0), ("top_k", top_k, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if rank >= world_size or top_k > 2048:
            raise ValueError("rank must be below world_size and top_k must be <= 2048")
        config = THDPlannerConfig() if config is None else config
        if not isinstance(config, THDPlannerConfig):
            raise TypeError("config must be a THDPlannerConfig")
        block = config.block_size
        if isinstance(block, bool) or not isinstance(block, int) or block < 1:
            raise ValueError("config.block_size must be a positive integer")
        if policy not in ("contiguous", "zigzag", "planner"):
            raise ValueError("policy must be contiguous, zigzag, or planner")
        if cost_model is not None and not isinstance(cost_model, THDSelectionCostModel):
            raise TypeError("cost_model must be a THDSelectionCostModel")
        if cost_model is not None and len(cost_model.rank_node_ids) != world_size:
            raise ValueError("cost-model node map must match the process-group size")
        if policy == "planner" and cost_model is None:
            raise ValueError("planner requires an explicit calibrated cost_model")
        if candidate_budget_bytes is not None and (
            isinstance(candidate_budget_bytes, bool)
            or not isinstance(candidate_budget_bytes, int)
            or candidate_budget_bytes < 4
        ):
            raise ValueError("candidate_budget_bytes must hold at least one FP32 element")
        cu = as_boundaries(global_cu_seqlens, name="global_cu_seqlens")
        if cu[-1] >= 2**31 or any(x % block for x in cu):
            raise ValueError("document boundaries must be block-aligned and fit int32")
        offsets = tuple(x * block for x in balanced_shards(cu[-1] // block, world_size))
        source_cap = max(z - a for a, z in zip(offsets, offsets[1:]))
        cap = 2 * source_cap if max_tokens_per_rank is None else max_tokens_per_rank
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < source_cap or cap % block:
            raise ValueError("max_tokens_per_rank must be block-aligned and fit every source shard")
        scores = ()
        if policy == "planner":
            planned = plan_thd_selection(
                cu, world_size=world_size, max_tokens_per_rank=cap, cost_model=cost_model, config=config
            )
            ranks, policy, scores = planned.ranks, planned.policy, planned.evaluated_scores
        elif policy == "zigzag":
            ranks = zigzag_thd_layout(cu, world_size=world_size, block_size=block)
        else:
            ranks = tuple(
                THDRankLayout(
                    owner,
                    tuple(
                        THDSpan(seq, max(left, a), min(right, z), a, z, owner)
                        for seq in range(bisect_right(cu, left) - 1, bisect_left(cu, right))
                        for a, z in (cu[seq : seq + 2],)
                    ),
                )
                for owner, (left, right) in enumerate(zip(offsets, offsets[1:]))
            )
        if any(r.local_tokens > cap for r in ranks):
            raise ValueError("selected layout exceeds max_tokens_per_rank")
        local = ranks[rank]
        ranges = tuple((s.sequence_start // block, s.sequence_end // block) for s in local.spans)
        prefix = [0]
        for a, z in ranges:
            prefix.append(prefix[-1] + z - a)
        native = None
        if local.local_tokens:
            native = DsaSelectionPlan.from_cu_seqlens(
                local.local_cu_seqlens,
                prefix,
                top_k=top_k,
                block_size=block,
                q_causal_offsets=local.causal_offsets,
                candidate_budget_bytes=candidate_budget_bytes,
            )
        return cls(
            cu,
            offsets,
            ranks,
            policy,
            scores,
            top_k,
            block,
            THDRedistributionPlan.build(ranks, offsets, rank),
            native,
            ranges,
            cost_model,
        )


@dataclass(frozen=True, slots=True)
class CsaCPSelectionWorkspace:
    """Device metadata, transport and reusable selection scratch.

    Prepare/use on the same CUDA stream, or explicitly synchronize between
    streams. Use a separate workspace for overlapping invocations; the native
    scratch is mutable. Keep the process group alive through every invocation.
    This eager, allocating transport has no CUDA Graph capture contract.
    """

    plan: CsaCPSelectionPlan
    exchange: THDRedistributor
    selection: DsaSelectionWorkspace | None
    key_slice: slice
    key_indices: torch.Tensor | None

    @classmethod
    def from_plan(cls, plan: CsaCPSelectionPlan, device: torch.device | str, *, group=None) -> CsaCPSelectionWorkspace:
        """Materialize a plan once for sequential reuse on this device/group.

        The returned workspace retains the plan, device routing indices and
        native scratch. Pass it directly to every compatible layer's execution;
        do not call this factory inside the layer loop.
        """
        if not isinstance(plan, CsaCPSelectionPlan):
            raise TypeError("plan must be a CsaCPSelectionPlan")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CSA CP selection requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        exchange = THDRedistributor(plan.exchange, device, group=group)
        native = None if plan.selection is None else DsaSelectionWorkspace.from_plan(plan.selection, device)
        ranges, indices = plan.key_ranges, None
        key_slice = slice(0, 0)
        if ranges:
            if all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])):
                key_slice = slice(ranges[0][0], ranges[-1][1])
            else:
                indices = _range_indices(ranges, device)
        return cls(plan, exchange, native, key_slice, indices)


@dataclass(frozen=True, slots=True)
class CsaCPSelectionResult:
    """Source-ordered document-local int32 block IDs and FP32 raw scores.

    Outputs own their storage and survive workspace reuse. Invalid slots are
    -1 IDs; slot order is unspecified. Selection is discrete and has no autograd
    connection: downstream indexer training uses its explicit backward/loss.
    """

    block_ids: torch.Tensor
    scores: torch.Tensor


@torch.no_grad()
def cudnn_csa_select_cp(
    qi: torch.Tensor,
    global_kbar: torch.Tensor,
    weights: torch.Tensor,
    *,
    workspace: CsaCPSelectionWorkspace,
    sm_scale: float = 1.0,
    deterministic: bool = False,
) -> CsaCPSelectionResult:
    """Route local queries, select pooled blocks, and restore original rows.

    ``qi`` is contiguous BF16 [source_rows, H, D], ``weights`` is BF16/FP32
    [source_rows, H], and ``global_kbar`` is contiguous BF16 [total_blocks, 1, D]
    in original packed-document order, already gathered by the caller. Pooling,
    attention, projections and their gradients stay independently composable.
    No precision casts, head padding, or device-to-host metadata copies occur.

    ``workspace`` is required and already contains the plan. Execution never
    replans, rebuilds routing indices or materializes native scratch. Each call
    uses fresh layer inputs; transport buffers and retained results may allocate.

    All ranks must enter in the same order with matching tensor trailing
    dimensions/dtypes and scalar options, including empty destination ranks.
    Tensor validation is local, like the underlying PyTorch collectives.
    """
    if not isinstance(workspace, CsaCPSelectionWorkspace):
        raise TypeError("workspace must be a CsaCPSelectionWorkspace")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
    if isinstance(sm_scale, bool) or not isinstance(sm_scale, int | float) or not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be a finite host scalar")
    plan, exchange = workspace.plan, workspace.exchange
    if dist.get_rank(exchange.group) != plan.exchange.rank or dist.get_world_size(exchange.group) != plan.exchange.world_size:
        raise ValueError("workspace process group no longer matches the plan")
    if qi.ndim != 3 or qi.shape[0] != plan.exchange.source_rows or min(qi.shape[1:]) < 1:
        raise ValueError("qi must have shape [source_rows, H, D] with positive H/D")
    if weights.shape != qi.shape[:2] or global_kbar.shape != (plan.global_cu_seqlens[-1] // plan.block_size, 1, qi.shape[2]):
        raise ValueError("weights/global_kbar shapes do not match the source query contract")
    if (
        qi.dtype != torch.bfloat16
        or global_kbar.dtype != torch.bfloat16
        or weights.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise TypeError("qi/global_kbar must be BF16 and weights BF16 or FP32; no implicit quantization")
    for tensor in (qi, global_kbar, weights):
        if tensor.device != exchange.device or not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous and on the workspace device")
    if plan.cost_model is not None:
        payload = qi.shape[1] * qi.shape[2] * qi.element_size() + qi.shape[1] * weights.element_size() + plan.top_k * 8
        if payload != plan.cost_model.bytes_per_query:
            raise ValueError("cost_model.bytes_per_query must match Q/weights plus returning int32 IDs/FP32 scores")
    routed_q, routed_weights = exchange.distribute(qi), exchange.distribute(weights)
    if plan.selection is None:
        ids = torch.empty((0, plan.top_k), dtype=torch.int32, device=exchange.device)
        scores = torch.empty((0, plan.top_k), dtype=torch.float32, device=exchange.device)
    else:
        keys = (
            global_kbar[workspace.key_slice]
            if workspace.key_indices is None
            else global_kbar.index_select(0, workspace.key_indices)
        )
        selected = cudnn_dsa_select(
            routed_q,
            keys,
            routed_weights,
            plan=plan.selection,
            workspace=workspace.selection,
            sm_scale=sm_scale,
            deterministic=deterministic,
        )
        ids, scores = selected.block_ids, selected.scores
    ids, scores = exchange.restore(ids), exchange.restore(scores)
    # The identity transport returns aliases, whereas a real exchange owns its
    # outputs. Provide the same retained-result lifetime for every policy.
    if plan.exchange.identity:
        ids, scores = ids.clone(), scores.clone()
    return CsaCPSelectionResult(ids, scores)


__all__ = ["CsaCPSelectionPlan", "CsaCPSelectionWorkspace", "CsaCPSelectionResult", "cudnn_csa_select_cp"]
