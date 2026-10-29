# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Autograd-aware context parallelism for cuDNN Frontend KDA in THD layout."""

from __future__ import annotations

from functools import cache
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable


@cache
def _load_cudnn_cp_ops():
    try:
        from cudnn import kimi_cp_merge
        from cudnn.linear_attention import (
            kimi_delta_attention,
            kimi_delta_attention_summary,
            kimi_delta_attention_summary_bwd,
        )
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "cuDNN KDA context parallelism requires cuDNN Frontend >=1.32 "
            "with the KimiCPMerge API (NVIDIA/cudnn-frontend#1997)"
        ) from exc
    return (
        kimi_cp_merge,
        kimi_delta_attention,
        kimi_delta_attention_summary,
        kimi_delta_attention_summary_bwd,
    )


def _distributed_identity(group: dist.ProcessGroup | None) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        if group is not None:
            raise RuntimeError("a process group was supplied but torch.distributed is not initialized")
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


def _all_gather(tensor: torch.Tensor, group: dist.ProcessGroup | None, world_size: int) -> torch.Tensor:
    """Gather one compact, equal-shaped boundary payload per rank."""
    if world_size == 1:
        return tensor.unsqueeze(0)
    gathered = torch.empty(
        (world_size * tensor.shape[0], *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor.contiguous(), group=group)
    return gathered.view(world_size, *tensor.shape)


def _pack_boundary_summary(h: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
    """Pack H and M into one collective payload with shape ``[H, V+K, K]``."""
    if h.ndim != 3 or transition.ndim != 3:
        raise ValueError("KDA boundary summaries must have [heads, dim, dim_k] layout")
    if h.shape[0] != transition.shape[0] or h.shape[-1] != transition.shape[-1]:
        raise ValueError("KDA H and transition boundary shapes are incompatible")
    return torch.cat((h, transition), dim=-2).contiguous()


def _unpack_boundary_summaries(packed: torch.Tensor, dim_v: int) -> tuple[torch.Tensor, torch.Tensor]:
    return packed[..., :dim_v, :], packed[..., dim_v:, :]


def _merge_rank_summaries(h: torch.Tensor, transition: torch.Tensor, *, reverse: bool) -> torch.Tensor:
    """Merge caller-gathered FP32 summaries with cuDNN's IEEE state chain."""
    kimi_cp_merge, _, _, _ = _load_cudnn_cp_ops()
    merged = kimi_cp_merge(h.contiguous(), transition.contiguous(), reverse=reverse)
    return merged["initial_states"]


def _merge_boundary_chain(
    h_all: torch.Tensor,
    transition_all: torch.Tensor,
    *,
    rank: int,
    num_remote_ranks: int,
    reverse: bool,
) -> torch.Tensor:
    """Return the state entering or leaving this rank's boundary sequence."""
    if num_remote_ranks == 0:
        return torch.zeros_like(h_all[rank])
    if reverse:
        start, stop, target = rank, rank + num_remote_ranks + 1, 0
    else:
        start, stop, target = rank - num_remote_ranks, rank + 1, -1
    states = _merge_rank_summaries(
        h_all[start:stop],
        transition_all[start:stop],
        reverse=reverse,
    )
    return states[target]


def _op_kwargs(options: dict[str, Any], a_log: torch.Tensor | None, dt_bias: torch.Tensor | None) -> dict[str, Any]:
    return {
        "use_qk_l2norm_in_kernel": options["use_qk_l2norm_in_kernel"],
        "use_beta_sigmoid_in_kernel": options["use_beta_sigmoid_in_kernel"],
        "allow_neg_eigval": options["allow_neg_eigval"],
        "safe_gate": options["safe_gate"],
        "gate_lower_bound": options["lower_bound"],
        "gate_domain": options["gate_domain"],
        "a_log": a_log,
        "dt_bias": dt_bias,
        "batch_invariant": options["batch_invariant"],
    }


class _CudnnKdaCP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        a_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        group: dist.ProcessGroup | None,
        pre_num_ranks: int,
        post_num_ranks: int,
        options: dict[str, Any],
    ) -> torch.Tensor:
        _, kimi_delta_attention, kimi_delta_attention_summary, _ = _load_cudnn_cp_ops()
        world_size, rank = _distributed_identity(group)
        kwargs = _op_kwargs(options, a_log, dt_bias)

        h_local, transition_local = kimi_delta_attention_summary(
            k,
            v,
            g,
            beta,
            cu_seqlens,
            plan_name=options["summary_plan_name"],
            **kwargs,
        )
        dim_v = h_local.shape[-2]
        packed = _pack_boundary_summary(h_local[-1], transition_local[-1])
        packed_all = _all_gather(packed, group, world_size)
        h_all, transition_all = _unpack_boundary_summaries(packed_all, dim_v)
        incoming_boundary = _merge_boundary_chain(
            h_all,
            transition_all,
            rank=rank,
            num_remote_ranks=pre_num_ranks,
            reverse=False,
        )
        incoming = torch.zeros_like(h_local)
        incoming[0].copy_(incoming_boundary)

        with torch.enable_grad():
            leaves = [tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)]
            a_leaf = a_log.detach().requires_grad_(True) if a_log is not None else None
            dt_leaf = dt_bias.detach().requires_grad_(True) if dt_bias is not None else None
            incoming_leaf = incoming.detach().requires_grad_(True)
            inner_kwargs = _op_kwargs(options, a_leaf, dt_leaf)
            output, final_state = kimi_delta_attention(
                *leaves,
                cu_seqlens,
                scale=options["scale"],
                initial_state=incoming_leaf,
                output_final_state=True,
                plan_name=options["plan_name"],
                **inner_kwargs,
            )

        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.post_num_ranks = post_num_ranks
        ctx.options = options
        ctx.a_log_present = a_log is not None
        ctx.dt_bias_present = dt_bias is not None
        ctx.inner_leaves = (*leaves, incoming_leaf, a_leaf, dt_leaf)
        ctx.inner_outputs = (output, final_state)
        ctx.save_for_backward(
            q,
            k,
            g,
            beta,
            cu_seqlens,
            transition_local[0],
            *(tensor for tensor in (a_log, dt_bias) if tensor is not None),
        )
        return output.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        _, _, _, kimi_delta_attention_summary_bwd = _load_cudnn_cp_ops()
        saved = iter(ctx.saved_tensors)
        q = next(saved)
        k = next(saved)
        g = next(saved)
        beta = next(saved)
        cu_seqlens = next(saved)
        transition_first = next(saved)
        a_log = next(saved) if ctx.a_log_present else None
        dt_bias = next(saved) if ctx.dt_bias_present else None
        options = ctx.options
        kwargs = _op_kwargs(options, a_log, dt_bias)

        g_local, backward_transition = kimi_delta_attention_summary_bwd(
            q,
            k,
            g,
            beta,
            grad_output.contiguous(),
            cu_seqlens,
            scale=options["scale"],
            output_transition=True,
            plan_name=options["summary_plan_name"],
            **kwargs,
        )
        if backward_transition[0].shape != transition_first.shape:
            raise RuntimeError("cuDNN KDA summary transition shape changed unexpectedly")
        packed = _pack_boundary_summary(g_local[0], transition_first)
        packed_all = _all_gather(packed, ctx.group, ctx.world_size)
        g_all, transition_all = _unpack_boundary_summaries(packed_all, g_local.shape[-2])
        outgoing_boundary = _merge_boundary_chain(
            g_all,
            transition_all,
            rank=ctx.rank,
            num_remote_ranks=ctx.post_num_ranks,
            reverse=True,
        )
        grad_final_state = torch.zeros_like(g_local)
        grad_final_state[-1].copy_(outgoing_boundary)

        q_leaf, k_leaf, v_leaf, g_leaf, beta_leaf, incoming_leaf, a_leaf, dt_leaf = ctx.inner_leaves
        grad_inputs = [q_leaf, k_leaf, v_leaf, g_leaf, beta_leaf, incoming_leaf]
        if a_leaf is not None:
            grad_inputs.append(a_leaf)
        if dt_leaf is not None:
            grad_inputs.append(dt_leaf)
        with torch.enable_grad():
            grads = torch.autograd.grad(
                ctx.inner_outputs,
                grad_inputs,
                (grad_output.contiguous(), grad_final_state),
                allow_unused=True,
            )

        dq, dk, dv, dg, dbeta = grads[:5]
        index = 6  # The cross-rank state chain already consumed d(incoming).
        da_log = grads[index] if a_leaf is not None else None
        index += int(a_leaf is not None)
        ddt_bias = grads[index] if dt_leaf is not None else None
        ctx.inner_leaves = None
        ctx.inner_outputs = None
        return dq, dk, dv, dg, dbeta, None, da_log, ddt_bias, None, None, None, None


def cudnn_kda_cp_thd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    pre_num_ranks: int | None = None,
    post_num_ranks: int | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    gate_domain: str = "log",
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    batch_invariant: bool = True,
    plan_name: str = "kda_frost",
    summary_plan_name: str = "kda_summary_frost",
) -> torch.Tensor:
    """Run differentiable KDA over one contiguous packed-THD CP shard.

    Global packed tokens are partitioned into contiguous rank-ordered shards.
    ``cu_seqlens`` describes the clipped sequences in this local shard.
    ``pre_num_ranks`` and ``post_num_ranks`` are the numbers of adjacent ranks
    sharing this shard's first and last sequence, respectively. They are
    required for multi-rank execution because only those boundary chains may
    carry recurrent state across ranks.

    Forward and backward each communicate one packed FP32 H/M boundary through
    ``all_gather_into_tensor``. cuDNN's FP32-only ``kimi_cp_merge`` composes the
    state chain; the local KDA forward/backward remains cuDNN Frontend native.
    """
    tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "cu_seqlens": cu_seqlens}
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("all KDA operands must be torch.Tensor instances")
    if any(tensor.ndim != 3 for tensor in (q, k, v, g)) or beta.ndim != 2:
        raise ValueError("cuDNN KDA CP expects q/k/v/g in THD and beta in TH layout")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be one-dimensional")
    if any(tensor.shape[0] != q.shape[0] for tensor in (k, v, g, beta)):
        raise ValueError("all KDA operands must have the same local token count")
    if any(tensor.device != q.device for tensor in (k, v, g, beta, cu_seqlens)):
        raise ValueError("all KDA operands must be on the same device")

    world_size, rank = _distributed_identity(group)
    if world_size == 1:
        pre_num_ranks = 0 if pre_num_ranks is None else pre_num_ranks
        post_num_ranks = 0 if post_num_ranks is None else post_num_ranks
    elif pre_num_ranks is None or post_num_ranks is None:
        raise ValueError("multi-rank KDA CP requires pre_num_ranks and post_num_ranks")
    if not 0 <= pre_num_ranks <= rank:
        raise ValueError(f"invalid pre_num_ranks={pre_num_ranks} for CP rank {rank}")
    if not 0 <= post_num_ranks < world_size - rank:
        raise ValueError(f"invalid post_num_ranks={post_num_ranks} for CP rank {rank}")

    options = {
        "scale": scale,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
        "allow_neg_eigval": allow_neg_eigval,
        "safe_gate": safe_gate,
        "lower_bound": lower_bound,
        "gate_domain": gate_domain,
        "batch_invariant": batch_invariant,
        "plan_name": plan_name,
        "summary_plan_name": summary_plan_name,
    }
    return _CudnnKdaCP.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        cu_seqlens.contiguous(),
        A_log,
        dt_bias,
        group,
        pre_num_ranks,
        post_num_ranks,
        options,
    )
