# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Eager, real-CP KDA + full MLA-absorb CSA operator benchmark.

This experiment includes pooling, selection, attention, the detached MaxPool
teacher/KL indexer backward, and all CP activation/gradient collectives. It is
not a model benchmark: projection, residual, MLP and optimizer are excluded.
It deliberately reuses allocating cuDNN convenience APIs; no CUDA Graph or
bounded compilation-cache guarantee is made for this experimental harness.
Document lengths and CP cuts must be multiples of four. Per-query causal
partial blocks (including the first three queries of every doc) are supported.
"""

import argparse
import datetime
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from itertools import accumulate
from types import SimpleNamespace

import torch
import torch.distributed as dist

from benchmarks.cp_layouts import query_layouts
from cula._metadata import balanced_shards
from cula.distributed import (
    AllGatherKimiCPTransport,
    THDRedistributionPlan,
    THDRedistributor,
    THDSelectionCostModel,
    build_kimi_cp_plan,
    cudnn_kda_cp,
)
from cula.hybrid import (
    CsaCPSelectionPlan,
    CsaCPSelectionWorkspace,
    DsaSelectionPlan,
    DsaSelectionWorkspace,
    IndexPoolPlan,
    cudnn_csa_select_cp,
    cudnn_dsa_select,
    cudnn_index_pool,
)
from cula.hybrid.index_pool import _load_cudnn_index_pool_ops


def validate_source_alignment(lengths, world):
    """Make the same alignment decision on every balanced source rank."""
    if sum(lengths) % (4 * world) or any(n % 4 for n in lengths):
        raise ValueError("source CP shards and documents must be block aligned")


def json_safe(value):
    """Replace nonfinite measurements only in the serialized report copy."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {
        "max_abs": delta.abs().max().item(),
        "relative_rms": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def expand_blocks(block_ids, scores, doc_start, visible):
    """Compact valid block slots, expand members, append the causal tail.

    Sorting by validity preserves each valid (id, score) pairing. The returned
    block ids are still document-local; token ids are globally addressed.
    Invalid slots remain -1 even when topk_length is supplied to cuDNN.
    """
    order = (block_ids < 0).to(torch.int32).argsort(dim=-1, stable=True)
    ids = block_ids.gather(1, order)
    scores = scores.gather(1, order)
    valid = ids >= 0
    nblocks = valid.sum(-1, dtype=torch.int32)
    members = doc_start[:, None, None] + 4 * ids[:, :, None] + torch.arange(4, device=ids.device)
    members = members.masked_fill(~valid[:, :, None], -1).flatten(1).to(torch.int32)
    # The padded row is friendly to both the attention and teacher kernels.
    width = math.ceil((members.shape[1] + 3) / 128) * 128
    tokens = torch.full((ids.shape[0], width), -1, dtype=torch.int32, device=ids.device)
    tokens[:, : members.shape[1]] = members
    slots = nblocks[:, None] * 4 + torch.arange(3, device=ids.device)
    tail = doc_start[:, None] + visible[:, None] // 4 * 4 + torch.arange(3, device=ids.device)
    tail = tail.masked_fill(torch.arange(3, device=ids.device)[None] >= visible[:, None] % 4, -1)
    tokens.scatter_(1, slots.long(), tail.to(torch.int32))
    lengths = nblocks * 4 + visible % 4
    return ids, scores, members, tokens, lengths.to(torch.int32)


def reduce_scatter_fp32(value, local_rows):
    out = torch.empty((local_rows, *value.shape[1:]), device=value.device, dtype=torch.float32)
    dist.reduce_scatter_tensor(out, value.float().contiguous())
    return out


def query_metadata(layout, device):
    """Materialize all span addresses with one transfer, including empty ranks."""
    sizes = torch.tensor([span.tokens for span in layout.spans], dtype=torch.int64, device="cpu")
    starts = torch.tensor([span.sequence_start for span in layout.spans], dtype=torch.int32, device="cpu")
    shifts = torch.tensor(
        [span.token_start - offset for span, offset in zip(layout.spans, (0, *accumulate(sizes.tolist())))],
        dtype=torch.int32,
        device="cpu",
    )
    n = layout.local_tokens
    origin = torch.repeat_interleave(starts, sizes, output_size=n)
    rows = torch.arange(n, dtype=torch.int32, device="cpu") + torch.repeat_interleave(shifts, sizes, output_size=n)
    return torch.stack((origin, rows - origin + 1, rows)).to(device).unbind(0)


class FullCSA:
    def __init__(
        self,
        lengths,
        heads,
        top_k,
        device,
        candidate_mib,
        pool_backward_order="per-loss",
        layout="contiguous",
        redistribution_scope="selection",
        planner_cost_model=None,
        planner_strategy="balanced",
        selection_api=False,
        planner_config=None,
    ):
        import cudnn

        self.ops = cudnn.DSA
        self.top_k = top_k
        self.pool_backward_order = pool_backward_order
        self.redistribution_scope = redistribution_scope
        self.selection_api = selection_api
        if selection_api and (
            redistribution_scope != "selection" or (layout == "planner" and planner_strategy != "selection")
        ):
            raise ValueError("the public API requires selection-only routing and the selection planner")
        _, self.pool_backward = _load_cudnn_index_pool_ops()
        torch.cuda.synchronize(device)
        common_prepare_start = time.perf_counter()
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        validate_source_alignment(lengths, self.world)
        self.cu = (0, *accumulate(lengths))
        self.plan = build_kimi_cp_plan(self.cu, group=dist.group.WORLD)
        self.source_n, self.total = self.plan.local_tokens, self.cu[-1]
        self.scale, self.heads = 576**-0.5, heads
        n = self.source_n
        pool_plan = IndexPoolPlan.from_cu_seqlens(self.plan.local_cu_seqlens, head_dim=128, ratio=4)
        pool_cu_tokens, pool_cu_blocks = pool_plan.materialize(device)
        torch.cuda.synchronize(device)
        self.common_prepare_ms = (time.perf_counter() - common_prepare_start) * 1000
        # All random tensors are generated in SOURCE ownership, with identical
        # RNG consumption across layouts. No target-layout synthetic inputs.
        self.indexer = SimpleNamespace(
            qi=torch.randn(n, 64, 128, dtype=torch.bfloat16, device=device),
            local_kv=torch.randn(n, 128, dtype=torch.bfloat16, device=device),
            pool_score=torch.randn(n, 128, dtype=torch.bfloat16, device=device),
            ape=torch.randn(4, 128, dtype=torch.float32, device=device),
            weights=torch.randn(n, 64, dtype=torch.float32, device=device),
            global_kbar=torch.empty(self.total // 4, 128, dtype=torch.bfloat16, device=device),
            pool_plan=pool_plan,
            pool_cu_tokens=pool_cu_tokens,
            pool_cu_blocks=pool_cu_blocks,
            local_blocks=n // 4,
        )
        b = self.indexer
        # Match the native backward's BF16 weights contract from the outset;
        # do not quantize an FP32 forward operand only for backward.
        b.weights = (b.weights / (64 * math.sqrt(128))).bfloat16()
        self.source_qi, self.source_weights = b.qi, b.weights
        for x in (b.local_kv, b.pool_score, b.ape):
            x.requires_grad_(True)
        # APE is a replicated parameter, not an independent rank-local input.
        dist.broadcast(b.ape.detach(), src=0)
        self.source_q = torch.randn(n, heads, 576, device=device, dtype=torch.bfloat16) * 0.5
        self.kv = torch.randn(n, 576, device=device, dtype=torch.bfloat16) * 0.5
        self.source_dout = torch.randn(n, heads, 512, device=device, dtype=torch.bfloat16)
        self.global_kv = torch.empty(self.total, 576, device=device, dtype=torch.bfloat16)
        self.sink = torch.full((heads,), -torch.inf, device=device, dtype=torch.float32)
        self.unit_grad = torch.ones((), device=device, dtype=torch.float32)
        self._layout_options = dict(
            device=device,
            candidate_mib=candidate_mib,
            layout=layout,
            redistribution_scope=redistribution_scope,
            planner_cost_model=planner_cost_model,
            planner_strategy=planner_strategy,
            selection_api=selection_api,
            planner_config=planner_config,
        )
        self._prepare_layout(**self._layout_options)
        self.events = []
        self.last = {}

    def prepare_metadata(self):
        """Rebuild fresh metadata after warmup, without regenerating inputs.

        Synchronization includes H2D metadata copies and workspace allocation;
        compilation and RNG initialization are outside this preparation metric.
        """
        device = self.source_q.device
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        self.plan = build_kimi_cp_plan(self.cu, group=dist.group.WORLD)
        b = self.indexer
        b.pool_plan = IndexPoolPlan.from_cu_seqlens(self.plan.local_cu_seqlens, head_dim=128, ratio=4)
        b.pool_cu_tokens, b.pool_cu_blocks = b.pool_plan.materialize(device)
        torch.cuda.synchronize(device)
        self.common_prepare_ms = (time.perf_counter() - start) * 1000
        self._prepare_layout(**self._layout_options)
        return dict(common_ms=self.common_prepare_ms, layout_ms=self.layout_prepare_ms)

    def _prepare_layout(
        self,
        *,
        device,
        candidate_mib,
        layout,
        redistribution_scope,
        planner_cost_model,
        planner_strategy,
        selection_api,
        planner_config,
    ):
        b, top_k = self.indexer, self.top_k
        torch.cuda.synchronize(device)
        prepare_start = time.perf_counter()
        if planner_cost_model is not None and redistribution_scope != "selection":
            raise ValueError("the selection cost model is valid only for selection-only redistribution")
        if planner_strategy == "selection" and redistribution_scope != "selection":
            raise ValueError("selection planning requires selection-only redistribution")
        if selection_api:
            cp_selection_plan = CsaCPSelectionPlan.from_cu_seqlens(
                self.cu,
                rank=self.rank,
                world_size=self.world,
                top_k=top_k,
                policy=layout,
                cost_model=planner_cost_model,
                config=planner_config,
                max_tokens_per_rank=self.total if layout == "zigzag" else None,
                candidate_budget_bytes=None if candidate_mib == 0 else candidate_mib << 20,
            )
            self.cp_selection = CsaCPSelectionWorkspace.from_plan(cp_selection_plan, device, group=dist.group.WORLD)
            ranks, self.selected_policy = cp_selection_plan.ranks, cp_selection_plan.policy
        else:
            ranks, self.selected_policy = query_layouts(
                self.cu, self.world, layout, cost_model=planner_cost_model, planner_strategy=planner_strategy
            )
        if redistribution_scope == "queries" and any(r.local_tokens == 0 for r in ranks):
            raise ValueError(
                "query-wide redistribution requires nonempty attention ranks; use selection scope for empty ranks"
            )
        self.selection_layout = ranks[self.rank]
        source_offsets = balanced_shards(self.total, self.world)
        if selection_api:
            self.selection_exchange = self.cp_selection.exchange
        else:
            exchange_plan = THDRedistributionPlan.build(ranks, source_offsets, self.rank)
            self.selection_exchange = THDRedistributor(exchange_plan, device, group=dist.group.WORLD)
        if redistribution_scope == "selection":
            attention_ranks, _ = query_layouts(self.cu, self.world, "contiguous")
            self.layout = attention_ranks[self.rank]
            self.exchange = THDRedistributor(
                THDRedistributionPlan.build(attention_ranks, source_offsets, self.rank), device, group=dist.group.WORLD
            )
        else:
            self.layout = self.selection_layout
            self.exchange = self.selection_exchange
        self.n = self.layout.local_tokens
        self.doc_start, self.visible, self.query_global_rows = query_metadata(self.layout, device)
        key_prefix, key_ranges = [0], []
        for span in self.selection_layout.spans:
            key_prefix.append(key_prefix[-1] + (span.sequence_end - span.sequence_start) // 4)
            key_ranges.append((span.sequence_start // 4, span.sequence_end // 4))
        b.select_plan = None
        b.select_workspace = None
        if selection_api:
            b.select_plan, b.select_workspace = cp_selection_plan.selection, self.cp_selection.selection
        elif self.selection_layout.local_tokens:
            b.select_plan = DsaSelectionPlan.from_cu_seqlens(
                self.selection_layout.local_cu_seqlens,
                key_prefix,
                top_k=top_k,
                block_size=4,
                q_causal_offsets=self.selection_layout.causal_offsets,
                candidate_budget_bytes=None if candidate_mib == 0 else candidate_mib << 20,
            )
            b.select_workspace = DsaSelectionWorkspace.from_plan(b.select_plan, device)
        self.selection_key_ids = None
        if selection_api:
            self.selection_key_slice, self.selection_key_ids = self.cp_selection.key_slice, self.cp_selection.key_indices
        elif not key_ranges:
            self.selection_key_slice = slice(0, 0)
        elif all(left[1] == right[0] for left, right in zip(key_ranges, key_ranges[1:])):
            self.selection_key_slice = slice(key_ranges[0][0], key_ranges[-1][1])
        else:
            self.selection_key_ids = torch.cat([torch.arange(a, z, device="cpu") for a, z in key_ranges]).to(device)
        torch.cuda.synchronize(device)
        self.layout_prepare_ms = (time.perf_counter() - prepare_start) * 1000

    def mark(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append((name, event))
        if getattr(self, "record_memory", False):
            self.memory_stages[name] = allocator_usage()

    def forward(self):
        b = self.indexer
        self.events = []
        self.memory_stages = {}
        self.mark("start")
        # Release prior iteration references before allocating the next layout.
        self.last = {}
        with torch.cuda.nvtx.range("csa.query_redistribute"):
            self.q = self.exchange.distribute(self.source_q)
            selection_qi = self.source_qi if self.selection_api else self.selection_exchange.distribute(self.source_qi)
            selection_weights = (
                self.source_weights if self.selection_api else self.selection_exchange.distribute(self.source_weights)
            )
            b.qi = self.source_qi if self.redistribution_scope == "selection" else selection_qi
            b.weights = self.source_weights if self.redistribution_scope == "selection" else selection_weights
        self.mark("query_redistribute")
        for x in (b.local_kv, b.pool_score, b.ape):
            x.grad = None
        with torch.cuda.nvtx.range("csa.pool"):
            pooled = cudnn_index_pool(
                b.local_kv, b.pool_score, b.ape, b.pool_cu_tokens, b.pool_cu_blocks, plan=b.pool_plan, deterministic=False
            )
        self.mark("pool_fwd")
        with torch.cuda.nvtx.range("csa.pool_gather"):
            dist.all_gather_into_tensor(b.global_kbar, pooled.detach())
        self.mark("pool_gather")
        with torch.cuda.nvtx.range("csa.selection_key_layout"):
            select_keys = (
                None
                if self.selection_api
                else (
                    b.global_kbar[self.selection_key_slice]
                    if self.selection_key_ids is None
                    else b.global_kbar.index_select(0, self.selection_key_ids)
                )
            )
        self.mark("selection_key_layout")
        with torch.cuda.nvtx.range("csa.selection"):
            if self.selection_api:
                selected = cudnn_csa_select_cp(
                    self.source_qi,
                    b.global_kbar[:, None],
                    self.source_weights,
                    workspace=self.cp_selection,
                    deterministic=False,
                )
            elif b.select_plan is None:
                selected = SimpleNamespace(
                    block_ids=torch.empty((0, self.top_k), device=selection_qi.device, dtype=torch.int32),
                    scores=torch.empty((0, self.top_k), device=selection_qi.device, dtype=torch.float32),
                )
            else:
                selected = cudnn_dsa_select(
                    selection_qi,
                    select_keys[:, None],
                    selection_weights,
                    plan=b.select_plan,
                    workspace=b.select_workspace,
                    deterministic=False,
                )
        self.mark("selection")
        with torch.cuda.nvtx.range("csa.selection_restore"):
            if self.redistribution_scope == "selection" and not self.selection_api:
                selected_ids = self.selection_exchange.restore(selected.block_ids)
                selected_scores = self.selection_exchange.restore(selected.scores)
            else:
                selected_ids, selected_scores = selected.block_ids, selected.scores
            del selection_qi, selection_weights, select_keys, selected
        self.mark("selection_restore")
        with torch.cuda.nvtx.range("csa.expand_tail"):
            ids, scores, members, tokens, counts = expand_blocks(selected_ids, selected_scores, self.doc_start, self.visible)
            block_ids = (ids + self.doc_start[:, None] // 4).masked_fill(ids < 0, -1)
        self.mark("expand_tail")
        with torch.cuda.nvtx.range("csa.kv_gather"):
            dist.all_gather_into_tensor(self.global_kv, self.kv)
        self.mark("kv_gather")
        with torch.cuda.nvtx.range("csa.attention_fwd"):
            fw = self.ops.sparse_attention_forward_wrapper(
                self.q, self.global_kv, tokens, attn_sink=self.sink, topk_length=counts, softmax_scale=self.scale
            )
        self.mark("attention_fwd")
        with torch.cuda.nvtx.range("csa.teacher_kl"):
            teacher = self.ops.sparse_attn_score_recompute_wrapper(
                self.q[None],
                self.global_kv[None],
                fw["lse"][None],
                members[None],
                softmax_scale=self.scale,
                qhead_per_kv_head=self.heads,
                topk_indices_global=True,
            )["target"][0]
            target = teacher.view(self.n, -1, 4).amax(-1)
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-10)
            valid = ids >= 0
            # Empty selected sets (first three queries) have zero auxiliary loss.
            safe_scores = scores.masked_fill(~valid, -torch.inf)
            safe_scores = torch.where(valid.any(-1, keepdim=True), safe_scores, torch.zeros_like(safe_scores))
            log_predict = safe_scores.log_softmax(-1).masked_fill(~valid, 0)
            predict = log_predict.exp().masked_fill(~valid, 0)
            loss = (target * (target.clamp_min(1e-10).log() - log_predict)).sum() / self.total
        self.mark("teacher_kl")
        with torch.cuda.nvtx.range("csa.output_restore"):
            source_output = self.exchange.restore(fw["out"])
        self.mark("output_restore")
        self.last = dict(
            source_output=source_output,
            pooled=pooled,
            ids=ids,
            block_ids=block_ids,
            members=members,
            tokens=tokens,
            counts=counts,
            target=target,
            predict=predict,
            loss=loss,
            fw=fw,
        )
        return source_output

    def backward(self):
        b, r = self.indexer, self.last
        with torch.cuda.nvtx.range("csa.dout_redistribute"):
            self.dout = self.exchange.distribute(self.source_dout)
        self.mark("dout_redistribute")
        with torch.cuda.nvtx.range("csa.attention_bwd"):
            bw = self.ops.sparse_attention_backward_wrapper(
                self.q,
                self.global_kv,
                r["fw"]["out"],
                self.dout,
                r["fw"]["lse"],
                self.sink,
                r["tokens"],
                softmax_scale=self.scale,
                topk_length=r["counts"],
                deterministic=False,
            )
        self.mark("attention_bwd")
        with torch.cuda.nvtx.range("csa.kv_reduce_scatter"):
            local_dkv = reduce_scatter_fp32(bw["dkv"], self.source_n)
        self.mark("kv_reduce_scatter")
        with torch.cuda.nvtx.range("csa.indexer_bwd"):
            # The backend overwrites target with its score gradient. Keep the
            # reference target intact for numerical validation.
            ibw = self.ops.indexer_backward_wrapper(
                b.qi[None],
                b.weights[None],
                b.global_kbar[None],
                r["target"][None].clone(),
                r["predict"][None].clamp_min(1e-30).masked_fill(r["block_ids"][None] < 0, 0),
                r["block_ids"][None],
                self.unit_grad,
                loss_coeff=self.n / self.total,
                topk_indices_global=True,
                deterministic=False,
                d_index_k=torch.empty_like(b.global_kbar[None], dtype=torch.float32),
            )
        self.mark("indexer_bwd")
        with torch.cuda.nvtx.range("csa.pool_grad_exchange"):
            if self.pool_backward_order == "sum-first":
                local_dkbar = reduce_scatter_fp32(ibw["d_index_k"][0], b.local_blocks)
            else:
                # Preserve cuLA's gather-then-pool numerical ordering: do not
                # merge loss-rank gradients before the native BF16 pool API.
                incoming = ibw["d_index_k"][0].bfloat16()
                received = torch.empty_like(incoming)
                dist.all_to_all_single(received, incoming)
                received = received.view(self.world, b.local_blocks, 128)
                local_dkbar = received.float().sum(0)
        self.mark("pool_grad_exchange")
        with torch.cuda.nvtx.range("csa.pool_bwd"):
            if self.pool_backward_order == "sum-first":
                r["pooled"].backward(local_dkbar.bfloat16())
            else:
                dkv = torch.zeros_like(b.local_kv, dtype=torch.float32)
                ds = torch.zeros_like(b.pool_score, dtype=torch.float32)
                da = torch.zeros_like(b.ape)
                for loss_rank in range(self.world):
                    pool_bw = self.pool_backward(
                        b.local_kv,
                        b.pool_score,
                        b.ape,
                        b.pool_cu_tokens,
                        b.pool_cu_blocks,
                        received[loss_rank],
                        ratio=4,
                        head_dim=128,
                        coff=1,
                        deterministic=False,
                    )
                    dkv.add_(pool_bw["grad_kv"])
                    ds.add_(pool_bw["grad_score"])
                    da.add_(pool_bw["grad_ape"])
                b.local_kv.grad = dkv.bfloat16()
                b.pool_score.grad = ds.bfloat16()
                b.ape.grad = da
            dist.all_reduce(b.ape.grad)
        self.mark("pool_bwd")
        with torch.cuda.nvtx.range("csa.query_grad_restore"):
            source_dq = self.exchange.restore(bw["dq"])
            source_dqi = self.exchange.restore(ibw["d_index_q"][0])
            source_dw = self.exchange.restore(ibw["d_weights"][0])
        self.mark("query_grad_restore")
        r.update(source_dq=source_dq, source_dqi=source_dqi, source_dw=source_dw)
        r.update(bw=bw, ibw=ibw, local_dkv=local_dkv, local_dkbar=local_dkbar)

    def timings(self):
        return {
            name + "_ms": previous.elapsed_time(event) for (_, previous), (name, event) in zip(self.events, self.events[1:])
        }


def check_reference(csa):
    """Independent FP32 attention/teacher/KL and end-to-end pool gradients.

    Top-K is discrete: check the same selected set, plus exact document,
    causality, uniqueness and complete/tail coverage. This oracle does not
    independently validate selector ranking.
    """
    b, r = csa.indexer, csa.last
    ids, tokens = r["ids"], r["tokens"]
    valid = tokens >= 0
    global_q = csa.query_global_rows
    metadata_ok = bool(torch.all((~valid) | ((tokens >= csa.doc_start[:, None]) & (tokens <= global_q[:, None]))))
    sorted_tokens = tokens.sort(-1).values
    metadata_ok &= not bool(((sorted_tokens[:, 1:] == sorted_tokens[:, :-1]) & (sorted_tokens[:, 1:] >= 0)).any())
    metadata_ok &= bool(torch.equal(valid.sum(-1), r["counts"]))
    metadata_ok &= bool(torch.equal((ids >= 0).sum(-1), (csa.visible // 4).clamp(max=ids.shape[1])))
    # Every causal tail token appears exactly once, even with no full block.
    for offset in range(3):
        tail = csa.doc_start + csa.visible // 4 * 4 + offset
        found = (tokens == tail[:, None]).sum(-1)
        metadata_ok &= bool(torch.all(found == (offset < csa.visible % 4).long()))

    kv = csa.global_kv.detach().float().requires_grad_(True)
    kbar = b.global_kbar.detach().float().requires_grad_(True)
    outputs, dq, dqi, dw, targets = [], [], [], [], []
    reference_loss = 0.0
    for start in range(0, csa.n, 16):
        end = min(start + 16, csa.n)
        ti = tokens[start:end].long()
        mask = ti >= 0
        q = csa.q[start:end].float().detach().requires_grad_(True)
        selected_kv = kv[ti.clamp_min(0)]
        logits = torch.einsum("qhd,qkd->qhk", q, selected_kv) * csa.scale
        probs = logits.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        output = torch.einsum("qhk,qkv->qhv", probs, selected_kv[..., :512])
        output.backward(csa.dout[start:end].float())
        outputs.append(output.detach())
        dq.append(q.grad)
        # Complete members occupy the prefix; causal tail starts after the last
        # valid block. Mask before pooling to exclude tail slots from teacher.
        teacher = probs.detach().sum(1)[:, : ids.shape[1] * 4].reshape(end - start, -1, 4).amax(-1)
        teacher = teacher.masked_fill(ids[start:end] < 0, 0)
        teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-10)
        targets.append(teacher)
        qi = b.qi[start:end].float().detach().requires_grad_(True)
        w = b.weights[start:end].float().detach().requires_grad_(True)
        bi = r["block_ids"][start:end].long()
        selected_kbar = kbar[bi.clamp_min(0)]
        scores = (torch.einsum("qhd,qkd->qkh", qi, selected_kbar).relu() * w[:, None]).sum(-1)
        valid_blocks = bi >= 0
        scores = scores.masked_fill(~valid_blocks, -torch.inf)
        scores = torch.where(valid_blocks.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        log_predict = scores.log_softmax(-1).masked_fill(~valid_blocks, 0)
        loss = (teacher * (teacher.clamp_min(1e-10).log() - log_predict)).sum() / csa.total
        loss.backward()
        reference_loss += loss.item()
        dqi.append(qi.grad)
        dw.append(w.grad)
    local_dkv = reduce_scatter_fp32(kv.grad, csa.source_n)
    local_dkbar = reduce_scatter_fp32(kbar.grad, b.local_blocks)
    pool_kv, pool_score, ape = [x.detach().float().requires_grad_(True) for x in (b.local_kv, b.pool_score, b.ape)]
    weights = (pool_score.view(-1, 4, 128) + ape[None]).softmax(1)
    pooled = (weights * pool_kv.view(-1, 4, 128)).sum(1)
    pooled.backward(local_dkbar)
    dist.all_reduce(ape.grad)
    comparisons = {
        "output": (r["source_output"], csa.exchange.restore(torch.cat(outputs))),
        "dq": (r["source_dq"], csa.exchange.restore(torch.cat(dq))),
        "dkv": (r["local_dkv"], local_dkv),
        "teacher": (r["target"], torch.cat(targets)),
        "d_index_q": (r["source_dqi"], csa.exchange.restore(torch.cat(dqi))),
        "d_weights": (r["source_dw"], csa.exchange.restore(torch.cat(dw))),
        "d_kbar": (r["local_dkbar"], local_dkbar),
        "d_pool_kv": (b.local_kv.grad, pool_kv.grad),
        "d_pool_score": (b.pool_score.grad, pool_score.grad),
        "d_ape": (b.ape.grad, ape.grad),
    }
    errors = {key: metrics(actual, expected) for key, (actual, expected) in comparisons.items()}
    # Fixed BF16 envelope, reported alongside raw errors; no threshold tuning.
    passed = metadata_ok and all(x["finite"] and x["relative_rms"] < 0.03 for x in errors.values())
    loss_error = abs(r["loss"].item() - reference_loss)
    passed &= loss_error < 1e-5
    return dict(pass_=passed, metadata_ok=metadata_ok, errors=errors, loss_abs_error=loss_error)


def check_redistribution(csa):
    device = csa.source_q.device
    source_rows = torch.arange(csa.plan.token_start, csa.plan.token_end, device=device)
    source = (source_rows[:, None] * 3 + torch.arange(3, device=device)).float().requires_grad_(True)
    ok = True
    routes = [(csa.exchange, csa.layout)]
    if csa.selection_exchange is not csa.exchange:
        routes.append((csa.selection_exchange, csa.selection_layout))
    for exchange, layout in routes:
        rows = (
            torch.cat([torch.arange(s.token_start, s.token_end, device=device) for s in layout.spans])
            if layout.spans
            else torch.empty(0, device=device, dtype=torch.int64)
        )
        expected = (rows[:, None] * 3 + torch.arange(3, device=device)).float()
        source.grad = None
        shuffled = exchange.distribute(source)
        ok &= torch.equal(shuffled, expected)
        shuffled.backward(expected * 2)
        ok &= torch.equal(source.grad, source.detach() * 2)
        leaf = shuffled.detach().requires_grad_(True)
        restored = exchange.restore(leaf)
        ok &= torch.equal(restored, source.detach())
        restored.backward(source.detach() * 3)
        ok &= torch.equal(leaf.grad, expected * 3)
    flag = torch.tensor(int(ok), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError("row redistribution/inverse/autograd marker check failed")
    return True


def compare_cpu_reference(actual, reference):
    """Full-tensor error, streamed from lossless CPU storage; outside timing."""
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("baseline shape/dtype mismatch")
    actual = actual.detach().reshape(-1)
    reference = reference.reshape(-1)
    stats = torch.zeros(3, device=actual.device, dtype=torch.float64)
    finite = torch.ones((), device=actual.device, dtype=torch.bool)
    for start in range(0, actual.numel(), 1 << 22):
        a = actual[start : start + (1 << 22)].float()
        b = reference[start : start + (1 << 22)].to(actual.device).float()
        delta = a - b
        stats[0] += delta.square().sum(dtype=torch.float64)
        stats[1] += b.square().sum(dtype=torch.float64)
        stats[2] = torch.maximum(stats[2], delta.abs().max().double())
        finite &= torch.isfinite(a).all()
    values = stats.cpu().tolist()
    return dict(relative_rms=math.sqrt(values[0] / max(values[1], 1e-24)), max_abs=values[2], finite=bool(finite))


def fingerprint(tensor):
    flat = tensor.detach().reshape(-1)
    sums = torch.zeros(2, device=flat.device, dtype=torch.float64)
    for start in range(0, flat.numel(), 1 << 22):
        x = flat[start : start + (1 << 22)].float()
        sums[0] += x.sum(dtype=torch.float64)
        sums[1] += x.square().sum(dtype=torch.float64)
    return sums.cpu().tolist()


def allocator_usage():
    return dict(
        allocated_gib=torch.cuda.memory_allocated() / 2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        reserved_gib=torch.cuda.memory_reserved() / 2**30,
    )


def main(*, keep_group=False, baseline=None, save_baseline=False, comparison_name="layout_vs_contiguous"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[4096])
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--candidate-mib", type=int, default=0, help="0 uses cuLA's balanced scratch policy")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--with-kda", action="store_true")
    parser.add_argument("--kda-batch-invariant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--kda-gate-scale", type=float, default=1.0, help="Scale negative KDA log-decay for sensitivity checks"
    )
    parser.add_argument("--layout", choices=("contiguous", "zigzag", "planner"), default="contiguous")
    parser.add_argument("--redistribution-scope", choices=("selection", "queries"), default="selection")
    parser.add_argument("--planner-cost-model", help="JSON calibrated THDSelectionCostModel coefficients")
    parser.add_argument("--planner-strategy", choices=("balanced", "selection"), default="balanced")
    parser.add_argument("--selection-api", action="store_true", help="Use the public CSA CP selection API")
    parser.add_argument("--memory-stages", action="store_true", help="Diagnostic allocator counters at each phase boundary")
    parser.add_argument("--pool-backward-order", choices=("per-loss", "sum-first"), default="per-loss")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.top_k <= 0 or args.top_k % 128 or args.warmups < 0 or args.repeats < 1:
        parser.error("top-k must be positive and divisible by 128; warmups >= 0 and repeats >= 1")
    if any(n <= 0 or n % 4 for n in args.sequence_lengths):
        parser.error("this harness requires positive document lengths divisible by four")
    if args.candidate_mib < 0:
        parser.error("candidate-mib must be nonnegative")
    if not math.isfinite(args.kda_gate_scale) or args.kda_gate_scale <= 0:
        parser.error("kda-gate-scale must be finite and positive")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(20261112 + rank)
    # Disable TF32 in the independent FP32 oracle.
    torch.backends.cuda.matmul.allow_tf32 = False
    cost_model = None
    if args.planner_cost_model:
        with open(args.planner_cost_model) as f:
            parameters = json.load(f)["model"]
        parameters["rank_node_ids"] = tuple(parameters["rank_node_ids"])
        cost_model = THDSelectionCostModel(**parameters)
    csa = FullCSA(
        args.sequence_lengths,
        args.heads,
        args.top_k,
        device,
        args.candidate_mib,
        args.pool_backward_order,
        args.layout,
        args.redistribution_scope,
        cost_model,
        args.planner_strategy,
        args.selection_api,
    )
    csa.record_memory = args.memory_stages
    redistribution_ok = check_redistribution(csa)
    kda_prepare_ms = 0.0
    if args.with_kda:
        plan = csa.plan
        torch.cuda.synchronize(device)
        kda_prepare_start = time.perf_counter()
        transport = AllGatherKimiCPTransport.prepare(plan, heads=32, key_dim=128, value_dim=128, device=device)
        local_cu = torch.tensor(plan.local_cu_seqlens, device=device, dtype=torch.int32)
        torch.cuda.synchronize(device)
        kda_prepare_ms = (time.perf_counter() - kda_prepare_start) * 1000
        q, k = [
            torch.nn.functional.normalize(torch.randn(csa.source_n, 32, 128, device=device), dim=-1).bfloat16()
            for _ in range(2)
        ]
        v = torch.randn_like(q)
        g = -0.01 - 0.04 * torch.rand(csa.source_n, 32, 128, device=device)
        if args.kda_gate_scale != 1.0:
            g *= args.kda_gate_scale
        beta = torch.sigmoid(torch.randn(csa.source_n, 32, device=device))
        kda_inputs = [x.requires_grad_(True) for x in (q, k, v, g, beta)]
        kda_do = torch.randn_like(v)
    samples, check = [], None
    for iteration in range(args.warmups + args.repeats):
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin, kf, cb, end = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        wall = time.perf_counter()
        begin.record()
        if args.with_kda:
            for x in kda_inputs:
                x.grad = None
            with torch.cuda.nvtx.range("kda.forward"):
                kda_out = cudnn_kda_cp(
                    *kda_inputs,
                    local_cu,
                    plan=plan,
                    group=dist.group.WORLD,
                    boundary_transport=transport,
                    batch_invariant=args.kda_batch_invariant,
                )
        kf.record()
        memory_after_kda_forward = allocator_usage() if args.memory_stages else None
        csa.forward()
        csa.backward()
        cb.record()
        if args.with_kda:
            with torch.cuda.nvtx.range("kda.backward"):
                kda_out.backward(kda_do)
        end.record()
        end.synchronize()
        memory_after_kda_backward = allocator_usage() if args.memory_stages else None
        sample = dict(
            total_ms=begin.elapsed_time(end),
            wall_ms=(time.perf_counter() - wall) * 1000,
            kda_fwd_ms=begin.elapsed_time(kf),
            kda_bwd_ms=cb.elapsed_time(end),
            csa_ms=kf.elapsed_time(cb),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            **csa.timings(),
        )
        if iteration >= args.warmups:
            samples.append(sample)
        if rank == 0:
            print("ITER", iteration, json.dumps(json_safe(sample), allow_nan=False), flush=True)
        if args.check and check is None:
            check = check_reference(csa)
    finite = all(
        bool(torch.isfinite(x).all())
        for x in (
            csa.last["fw"]["out"],
            csa.last["bw"]["dq"],
            csa.last["local_dkv"],
            csa.last["ibw"]["d_index_q"],
            csa.last["ibw"]["d_weights"],
            csa.last["local_dkbar"],
            csa.last["target"],
            csa.last["predict"],
            csa.last["loss"],
            csa.indexer.local_kv.grad,
            csa.indexer.pool_score.grad,
            csa.indexer.ape.grad,
        )
    )
    if args.with_kda:
        finite &= all(bool(torch.isfinite(x).all()) for x in [kda_out, *(x.grad for x in kda_inputs)])
    r, b = csa.last, csa.indexer
    tensors = dict(
        output=r["source_output"],
        dq=r["source_dq"],
        dkv=r["local_dkv"],
        d_index_q=r["source_dqi"],
        d_weights=r["source_dw"],
        d_pool_kv=b.local_kv.grad,
        d_pool_score=b.pool_score.grad,
        d_ape=b.ape.grad,
    )
    if args.with_kda:
        tensors["kda_output"] = kda_out
        tensors.update({"kda_" + key: x.grad for key, x in zip(("dq", "dk", "dv", "dg", "dbeta"), kda_inputs)})
    global_loss = r["loss"].detach().clone()
    dist.all_reduce(global_loss)
    errors = (
        None
        if baseline is None
        else {key: compare_cpu_reference(value, baseline["tensors"][key]) for key, value in tensors.items()}
    )
    loss_delta = None if baseline is None else abs(global_loss.item() - baseline["global_loss"])
    # These operands verify identical source data across the three layouts.
    inputs = dict(
        q=csa.source_q,
        kv=csa.kv,
        qi=csa.source_qi,
        weights=csa.source_weights,
        pool_kv=b.local_kv,
        pool_score=b.pool_score,
        ape=b.ape,
        dout=csa.source_dout,
    )
    if args.with_kda:
        inputs.update({"kda_" + key: x for key, x in zip(("q", "k", "v", "g", "beta"), kda_inputs)})
        inputs["kda_dout"] = kda_do
    input_checksums = {key: fingerprint(value) for key, value in inputs.items()}
    input_match = baseline is None or input_checksums == baseline["input_checksums"]
    result = dict(
        rank=rank,
        samples=samples,
        check=check,
        finite=finite,
        loss=global_loss.item(),
        loss_abs_diff=loss_delta,
        input_match=input_match,
        redistribution_bitwise=redistribution_ok,
        layout_prepare_ms=csa.layout_prepare_ms,
        common_prepare_ms=csa.common_prepare_ms,
        kda_prepare_ms=kda_prepare_ms,
        total_prepare_ms=csa.common_prepare_ms + csa.layout_prepare_ms + kda_prepare_ms,
        selected_policy=csa.selected_policy,
        local_query_rows=csa.n,
        spans=len(csa.layout.spans),
        selection_spans=len(csa.selection_layout.spans),
        selection_query_rows=csa.selection_layout.local_tokens,
        remote_query_rows=csa.selection_exchange.plan.remote_rows,
        selection_workspace_bytes=0 if b.select_plan is None else b.select_plan.candidate_workspace_floats * 4,
        memory_stages=(
            dict(kda_forward=memory_after_kda_forward, **csa.memory_stages, kda_backward=memory_after_kda_backward)
            if args.memory_stages
            else None
        ),
    )
    result[comparison_name] = errors
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        critical = {
            key: statistics.median(max(r["samples"][i][key] for r in gathered) for i in range(args.repeats))
            for key in samples[0]
        }
        report = dict(config=vars(args), world_size=world, critical_rank_median=critical, ranks=gathered)
        report["planner_cost_model"] = None if cost_model is None else asdict(cost_model)
        with open(args.output, "w") as f:
            json.dump(json_safe(report), f, indent=2, allow_nan=False)
        print("RESULT", json.dumps(json_safe(critical), allow_nan=False), flush=True)
    okay = all(
        r["finite"]
        and r["input_match"]
        and r["redistribution_bitwise"]
        and (r["loss_abs_diff"] is None or 0 <= r["loss_abs_diff"] < 1e-5)
        and (r["check"] is None or r["check"]["pass_"])
        and (r[comparison_name] is None or all(e["finite"] and e["relative_rms"] < 0.03 for e in r[comparison_name].values()))
        for r in gathered
    )
    saved = (
        None
        if not save_baseline
        else dict(
            tensors={key: value.detach().cpu() for key, value in tensors.items()},
            global_loss=global_loss.item(),
            input_checksums=input_checksums,
        )
    )
    if not keep_group:
        dist.destroy_process_group()
    if not okay:
        raise RuntimeError("Numerical check failed; raw per-rank errors were saved")
    return saved


if __name__ == "__main__":
    main()
