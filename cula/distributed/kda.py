# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Autograd-aware cuDNN KDA context parallelism for packed THD shards."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from inspect import signature
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from cula.distributed.plan import KimiCPPlan


@cache
def _load_cudnn_cp_ops():
    try:
        cudnn = import_module("cudnn")
        linear_attention = import_module("cudnn.linear_attention")
        ops = linear_attention
        if not all(
            hasattr(ops, name)
            for name in (
                "kimi_delta_attention",
                "kimi_delta_attention_summary",
                "kimi_delta_attention_summary_bwd",
            )
        ):
            ops = import_module("cudnn.linear_attention.ops")
        raw_kimi_cp_merge = getattr(linear_attention, "kimi_cp_merge", None)
        if raw_kimi_cp_merge is None:
            try:
                raw_kimi_cp_merge = import_module("cudnn.linear_attention.merge").kimi_cp_merge
            except ImportError:
                raw_kimi_cp_merge = cudnn.kimi_cp_merge
        if "precision" in signature(raw_kimi_cp_merge).parameters:
            # Development builds predating the fixed-IEEE API exposed a
            # precision switch.  Preserve the runtime's FP32 invariant while
            # allowing those builds to validate the distributed layer.
            def kimi_cp_merge(*args, **kwargs):
                return raw_kimi_cp_merge(*args, precision="ieee", **kwargs)

        else:
            kimi_cp_merge = raw_kimi_cp_merge
        kimi_delta_attention = ops.kimi_delta_attention
        kimi_delta_attention_summary = ops.kimi_delta_attention_summary
        kimi_delta_attention_summary_bwd = ops.kimi_delta_attention_summary_bwd
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "cuDNN Kimi CP requires cuDNN Frontend with KDA summary APIs and KimiCPMerge; "
            "install cuda-linear-attention[cudnn-cp]"
        ) from error
    return kimi_cp_merge, kimi_delta_attention, kimi_delta_attention_summary, kimi_delta_attention_summary_bwd


@dataclass(frozen=True, slots=True)
class _KdaOptions:
    scale: float | None
    use_qk_l2norm_in_kernel: bool
    use_beta_sigmoid_in_kernel: bool
    allow_neg_eigval: bool
    safe_gate: bool
    lower_bound: float | None
    gate_domain: str
    batch_invariant: bool
    plan_name: str | None
    summary_plan_name: str | None

    def kwargs(self, a_log: torch.Tensor | None, dt_bias: torch.Tensor | None) -> dict[str, Any]:
        return {
            "use_qk_l2norm_in_kernel": self.use_qk_l2norm_in_kernel,
            "use_beta_sigmoid_in_kernel": self.use_beta_sigmoid_in_kernel,
            "allow_neg_eigval": self.allow_neg_eigval,
            "safe_gate": self.safe_gate,
            "gate_lower_bound": self.lower_bound,
            "gate_domain": self.gate_domain,
            "a_log": a_log,
            "dt_bias": dt_bias,
            "batch_invariant": self.batch_invariant,
        }


def _all_gather(tensor: torch.Tensor, group: dist.ProcessGroup | None, world_size: int) -> torch.Tensor:
    """Gather one equal-shaped boundary payload per rank."""
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
    """Pack FP32 H/M into one collective payload shaped ``[heads, V+K, K]``."""
    if h.ndim != 3 or transition.ndim != 3:
        raise ValueError("KDA boundary summaries must have [heads, dim, key_dim] layout")
    if h.dtype != torch.float32 or transition.dtype != torch.float32:
        raise TypeError("KDA boundary summaries must remain FP32")
    if h.shape[0] != transition.shape[0] or h.shape[-1] != transition.shape[-1]:
        raise ValueError("KDA H and transition boundary shapes are incompatible")
    return torch.cat((h, transition), dim=-2).contiguous()


def _unpack_boundary_summaries(packed: torch.Tensor, dim_v: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < dim_v < packed.shape[-2]:
        raise ValueError("invalid value dimension for packed KDA boundary summaries")
    return packed[..., :dim_v, :], packed[..., dim_v:, :]


def _merge_rank_summaries(h: torch.Tensor, transition: torch.Tensor, *, reverse: bool) -> torch.Tensor:
    """Compose an ordered remote state chain with cuDNN's IEEE FP32 merge."""
    kimi_cp_merge, _, _, _ = _load_cudnn_cp_ops()
    merged = kimi_cp_merge(
        h.contiguous(),
        transition.contiguous(),
        reverse=reverse,
    )
    return merged["final_state"][0]


def _merge_boundary_chain(
    h_all: torch.Tensor,
    transition_all: torch.Tensor,
    *,
    rank: int,
    num_remote_ranks: int,
    reverse: bool,
) -> torch.Tensor:
    """Return the state entering/leaving this rank's cross-rank sequence."""
    if num_remote_ranks == 0:
        return torch.zeros_like(h_all[rank])
    if reverse:
        start, stop = rank + 1, rank + num_remote_ranks + 1
    else:
        start, stop = rank - num_remote_ranks, rank
    return _merge_rank_summaries(h_all[start:stop], transition_all[start:stop], reverse=reverse)


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
        plan: KimiCPPlan,
        group: dist.ProcessGroup | None,
        options: _KdaOptions,
    ) -> torch.Tensor:
        _, kimi_delta_attention, kimi_delta_attention_summary, _ = _load_cudnn_cp_ops()
        kwargs = options.kwargs(a_log, dt_bias)

        # A contiguous shard can carry state only through its last local span.
        # Restricting the summary to that span avoids rescanning short samples.
        last_start = plan.last_boundary_start
        last_cu = cu_seqlens[-2:] - last_start
        h_local, transition_local = kimi_delta_attention_summary(
            k[last_start:],
            v[last_start:],
            g[last_start:],
            beta[last_start:],
            last_cu,
            plan_name=options.summary_plan_name,
            **kwargs,
        )
        dim_v = h_local.shape[-2]
        packed = _pack_boundary_summary(h_local[0], transition_local[0])
        packed_all = _all_gather(packed, group, plan.world_size)
        h_all, transition_all = _unpack_boundary_summaries(packed_all, dim_v)
        incoming_boundary = _merge_boundary_chain(
            h_all,
            transition_all,
            rank=plan.rank,
            num_remote_ranks=plan.pre_num_ranks,
            reverse=False,
        )
        incoming = h_local.new_zeros(
            plan.num_local_sequences,
            h_local.shape[1],
            h_local.shape[2],
            h_local.shape[3],
        )
        incoming[0].copy_(incoming_boundary)

        # Keep cuDNN's native local autograd graph.  Distributed backward below
        # supplies the state gradient arriving from subsequent ranks.
        with torch.enable_grad():
            # All inner operands require gradients even when an outer caller
            # does not.  torch.autograd.Function will discard unrequested
            # returned gradients, while the fixed inner list keeps backward
            # independent of the caller's requires_grad pattern.
            leaves = [tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)]
            a_leaf = a_log.detach().requires_grad_(True) if a_log is not None else None
            dt_leaf = dt_bias.detach().requires_grad_(True) if dt_bias is not None else None
            incoming_leaf = incoming.detach().requires_grad_(True)
            output, final_state = kimi_delta_attention(
                *leaves,
                cu_seqlens,
                scale=options.scale,
                initial_state=incoming_leaf,
                output_final_state=True,
                plan_name=options.plan_name,
                **options.kwargs(a_leaf, dt_leaf),
            )

        ctx.plan = plan
        ctx.group = group
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
        a_log = next(saved) if ctx.a_log_present else None
        dt_bias = next(saved) if ctx.dt_bias_present else None
        plan = ctx.plan
        options = ctx.options
        grad_output = grad_output.contiguous()

        # The reverse carry is determined only by the first local span.  cuDNN
        # returns its transition as M^T; reverse merge expects forward M.
        first_end = plan.first_boundary_end
        g_local, transition_backward = kimi_delta_attention_summary_bwd(
            q[:first_end],
            k[:first_end],
            g[:first_end],
            beta[:first_end],
            grad_output[:first_end],
            cu_seqlens[:2],
            scale=options.scale,
            output_transition=True,
            plan_name=options.summary_plan_name,
            **options.kwargs(a_log, dt_bias),
        )
        transition_first = transition_backward[0].transpose(-2, -1)
        packed = _pack_boundary_summary(g_local[0], transition_first)
        packed_all = _all_gather(packed, ctx.group, plan.world_size)
        g_all, transition_all = _unpack_boundary_summaries(packed_all, g_local.shape[-2])
        outgoing_boundary = _merge_boundary_chain(
            g_all,
            transition_all,
            rank=plan.rank,
            num_remote_ranks=plan.post_num_ranks,
            reverse=True,
        )
        grad_final_state = g_local.new_zeros(
            plan.num_local_sequences,
            g_local.shape[1],
            g_local.shape[2],
            g_local.shape[3],
        )
        grad_final_state[-1].copy_(outgoing_boundary)

        q_leaf, k_leaf, v_leaf, g_leaf, beta_leaf, incoming_leaf, a_leaf, dt_leaf = ctx.inner_leaves
        requested = [q_leaf, k_leaf, v_leaf, g_leaf, beta_leaf, incoming_leaf]
        optional_positions = []
        for leaf in (a_leaf, dt_leaf):
            if leaf is not None:
                optional_positions.append(len(requested))
                requested.append(leaf)
        with torch.enable_grad():
            grads = torch.autograd.grad(
                ctx.inner_outputs,
                requested,
                (grad_output, grad_final_state),
                allow_unused=True,
            )

        dq, dk, dv, dg, dbeta = grads[:5]
        optional_grads = iter(grads[index] for index in optional_positions)
        da_log = next(optional_grads) if a_leaf is not None else None
        ddt_bias = next(optional_grads) if dt_leaf is not None else None
        ctx.inner_leaves = None
        ctx.inner_outputs = None
        return dq, dk, dv, dg, dbeta, None, da_log, ddt_bias, None, None, None


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    plan: KimiCPPlan,
    group: dist.ProcessGroup | None,
    safe_gate: bool,
    a_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
) -> None:
    operands = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    if any(not isinstance(tensor, torch.Tensor) for tensor in operands.values()):
        raise TypeError("all KDA operands must be torch.Tensor instances")
    if any(tensor.ndim != 3 for tensor in (q, k, v, g)) or beta.ndim != 2:
        raise ValueError("cuDNN Kimi CP expects q/k/v/g in THD and beta in TH layout")
    if any(tensor.shape[0] != q.shape[0] for tensor in (k, v, g, beta)):
        raise ValueError("all KDA operands must have the same local token count")
    if q.shape[0] != plan.local_tokens:
        raise ValueError(f"KDA operands contain {q.shape[0]} local tokens but KimiCPPlan expects {plan.local_tokens}")
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        raise ValueError("all KDA operands must be on the same device")
    if q.device.type != "cuda":
        raise ValueError("cuDNN Kimi CP requires CUDA operands")
    plan.validate_group(group)
    plan.validate_local_cu_seqlens(cu_seqlens, device=q.device)
    if safe_gate:
        heads, key_dim = g.shape[-2:]
        if a_log is not None and tuple(a_log.shape) != (heads,):
            raise ValueError(f"safe-gate a_log must have shape [{heads}]")
        if dt_bias is not None and tuple(dt_bias.shape) != (heads, key_dim):
            raise ValueError(f"safe-gate dt_bias must have shape [{heads}, {key_dim}]")
    elif a_log is not None or dt_bias is not None:
        raise ValueError("a_log and dt_bias are accepted only with safe_gate=True; canonicalize non-safe gates upstream")


def cudnn_kda_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    plan: KimiCPPlan,
    group: dist.ProcessGroup | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    gate_domain: str = "log",
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    batch_invariant: bool = True,
    plan_name: str | None = "kda_frost",
    summary_plan_name: str | None = "kda_summary_frost",
) -> torch.Tensor:
    """Run differentiable cuDNN KDA over one contiguous packed-THD CP shard.

    cuLA owns boundary planning, collectives, ordered state propagation and
    distributed autograd.  cuDNN Frontend owns local KDA summaries, local
    forward/backward and the IEEE FP32 state merge.  The merge precision is an
    invariant rather than a user option so large-CP behavior cannot silently
    degrade.

    ``plan`` should be built once per packed microbatch and reused by all KDA
    layers.  Inputs use cuDNN's canonical THD contract; framework-specific
    layout and gate conversions belong in the framework adapter.
    """

    if not isinstance(plan, KimiCPPlan):
        raise TypeError("plan must be a KimiCPPlan")
    _validate_inputs(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        plan=plan,
        group=group,
        safe_gate=safe_gate,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    options = _KdaOptions(
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        gate_domain=gate_domain,
        batch_invariant=batch_invariant,
        plan_name=plan_name,
        summary_plan_name=summary_plan_name,
    )
    if plan.world_size == 1:
        _, kimi_delta_attention, _, _ = _load_cudnn_cp_ops()
        output, _ = kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            plan_name=plan_name,
            **options.kwargs(a_log, dt_bias),
        )
        return output
    return _CudnnKdaCP.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        cu_seqlens.contiguous(),
        a_log,
        dt_bias,
        plan,
        group,
        options,
    )


__all__ = ["cudnn_kda_cp"]
