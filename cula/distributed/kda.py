# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Autograd-aware cuDNN KDA context parallelism for packed THD shards."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from inspect import signature
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from cula.distributed.plan import KimiCPPlan


@cache
def _load_cudnn_kda_ops():
    try:
        linear_attention = import_module("cudnn.linear_attention")
        ops = linear_attention
        if not hasattr(ops, "kimi_delta_attention"):
            ops = import_module("cudnn.linear_attention.ops")
        kimi_delta_attention = ops.kimi_delta_attention
        if "initial_state" not in signature(kimi_delta_attention).parameters:
            raise AttributeError("KDA apply does not expose recurrent state inputs")
        kimi_delta_attention_bwd = torch.ops.cudnn.kimi_delta_attention_bwd
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "cuDNN Kimi CP requires cuDNN Frontend KDA state APIs; "
            "install cuda-linear-attention[cudnn-cp] and follow the compatible source-build instructions in USAGE.md"
        ) from error
    return kimi_delta_attention, kimi_delta_attention_bwd


@cache
def _load_cudnn_cp_ops():
    try:
        kimi_delta_attention, kimi_delta_attention_bwd = _load_cudnn_kda_ops()
        linear_attention = import_module("cudnn.linear_attention")
        ops = linear_attention
        if not all(hasattr(ops, name) for name in ("kimi_delta_attention_summary", "kimi_delta_attention_summary_bwd")):
            ops = import_module("cudnn.linear_attention.ops")
        kimi_delta_attention_summary = ops.kimi_delta_attention_summary
        kimi_delta_attention_summary_bwd = ops.kimi_delta_attention_summary_bwd
        if "preceding_state_update" not in signature(kimi_delta_attention).parameters:
            raise AttributeError("KDA apply does not accept concatenated CP summaries")
        summary_bwd_parameters = signature(kimi_delta_attention_summary_bwd).parameters
        if "output_context" not in summary_bwd_parameters or "checkpoint_every_n_tokens" not in summary_bwd_parameters:
            raise AttributeError("KDA backward summary does not expose the CP apply context")
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "cuDNN Kimi CP requires cuDNN Frontend with concatenated KDA summary apply APIs; "
            "install cuda-linear-attention[cudnn-cp] and follow the compatible source-build instructions in USAGE.md"
        ) from error
    return (
        kimi_delta_attention,
        kimi_delta_attention_summary,
        kimi_delta_attention_summary_bwd,
        kimi_delta_attention_bwd,
    )


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
    checkpoint_every_n_tokens: int
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

    def attention_kwargs(self, a_log: torch.Tensor | None, dt_bias: torch.Tensor | None) -> dict[str, Any]:
        kwargs = self.kwargs(a_log, dt_bias)
        kwargs["checkpoint_every_n_tokens"] = self.checkpoint_every_n_tokens
        return kwargs


def _validate_checkpoint_every_n_tokens(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("checkpoint_every_n_tokens must be an integer")
    if value not in (0, 16):
        raise ValueError("checkpoint_every_n_tokens must be 0 or the exact native chunk cadence 16")
    return value


def _all_gather_into(
    output: torch.Tensor,
    tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup | None,
) -> None:
    if hasattr(dist, "all_gather_single"):
        dist.all_gather_single(output, tensor, group=group)
    else:  # pragma: no cover - compatibility with older PyTorch releases
        dist.all_gather_into_tensor(output, tensor, group=group)


def _all_gather(tensor: torch.Tensor, group: dist.ProcessGroup | None, world_size: int) -> torch.Tensor:
    """Gather one equal-shaped boundary payload per rank."""
    if world_size == 1:
        return tensor.unsqueeze(0)
    gathered = torch.empty(
        (world_size * tensor.shape[0], *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    _all_gather_into(gathered, tensor.contiguous(), group=group)
    return gathered.view(world_size, *tensor.shape)


def _validate_local_boundary_summaries(h: torch.Tensor, transition: torch.Tensor) -> None:
    """Validate the two dense FP32 tensors emitted by a local summary op."""
    if h.ndim != 3 or transition.ndim != 3:
        raise ValueError("KDA boundary summaries must have [heads, dim, key_dim] layout")
    if h.dtype != torch.float32 or transition.dtype != torch.float32:
        raise TypeError("KDA boundary summaries must remain FP32")
    if h.shape[0] != transition.shape[0] or h.shape[-1] != transition.shape[-1]:
        raise ValueError("KDA H and transition boundary shapes are incompatible")
    if not h.is_contiguous() or not transition.is_contiguous():
        raise ValueError("local KDA boundary summaries must be contiguous")


def _select_boundary_summaries(
    h_all: torch.Tensor,
    transition_all: torch.Tensor,
    *,
    rank: int,
    num_remote_ranks: int,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the ordered summaries consumed by cuDNN's CP-aware KDA apply."""
    if reverse:
        start, stop = rank + 1, rank + num_remote_ranks + 1
    else:
        start, stop = rank - num_remote_ranks, rank
    return h_all[start:stop], transition_all[start:stop]


@dataclass(slots=True)
class _KimiCPCollectiveWorkspace:
    """Reusable FP32 payload storage for one Kimi CP collective stream.

    The workspace is temporary transport storage and is not retained by the
    local KDA autograd graph. One instance can therefore be reused across KDA
    layers executed sequentially on the same CUDA stream. Concurrent streams
    require separate instances.
    """

    world_size: int
    heads: int
    key_dim: int
    value_dim: int
    gathered_h: torch.Tensor
    gathered_transition: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        world_size: int,
        heads: int,
        key_dim: int,
        value_dim: int,
        device: torch.device | str | int,
    ) -> _KimiCPCollectiveWorkspace:
        dimensions = {
            "world_size": world_size,
            "heads": heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        resolved = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        gathered_h = torch.empty((world_size * heads, value_dim, key_dim), dtype=torch.float32, device=resolved)
        gathered_transition = torch.empty((world_size * heads, key_dim, key_dim), dtype=torch.float32, device=resolved)
        return cls(world_size, heads, key_dim, value_dim, gathered_h, gathered_transition)

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.gathered_h, self.gathered_transition))

    def validate(self, h: torch.Tensor, transition: torch.Tensor, *, world_size: int) -> None:
        expected_h = (self.heads, self.value_dim, self.key_dim)
        expected_transition = (self.heads, self.key_dim, self.key_dim)
        expected_gathered_h = (self.world_size * self.heads, self.value_dim, self.key_dim)
        expected_gathered_transition = (self.world_size * self.heads, self.key_dim, self.key_dim)
        if world_size != self.world_size:
            raise ValueError(f"workspace world_size={self.world_size} does not match KimiCPPlan world_size={world_size}")
        if h.dtype != torch.float32 or transition.dtype != torch.float32:
            raise TypeError("Kimi CP boundary summaries must remain FP32")
        if transition.device != h.device:
            raise ValueError("Kimi CP H and transition summaries must be on the same device")
        if tuple(h.shape) != expected_h or tuple(transition.shape) != expected_transition:
            raise ValueError(
                "workspace H/M shape mismatch: "
                f"expected {expected_h} and {expected_transition}, got {tuple(h.shape)} and {tuple(transition.shape)}"
            )
        for name, tensor, shape in (
            ("gathered_h", self.gathered_h, expected_gathered_h),
            ("gathered_transition", self.gathered_transition, expected_gathered_transition),
        ):
            if tensor.dtype != torch.float32 or tensor.device != h.device or tuple(tensor.shape) != shape:
                raise ValueError(f"workspace {name} must be FP32 on {h.device} with shape {shape}")
            if not tensor.is_contiguous():
                raise ValueError(f"workspace {name} must be contiguous")


@runtime_checkable
class KimiCPBoundaryTransport(Protocol):
    """KDA-specific contract for ordered cross-rank summary propagation."""

    def boundary_summaries(
        self,
        h: torch.Tensor,
        transition: torch.Tensor,
        *,
        plan: KimiCPPlan,
        group: dist.ProcessGroup | None,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(slots=True)
class AllGatherKimiCPTransport:
    """Exact ordered Kimi CP transport backed by typed H/M all-gathers.

    With no workspace, storage is allocated lazily on each call. Use
    :meth:`prepare` for stable large-CP memory and CUDA Graph integration.
    """

    workspace: _KimiCPCollectiveWorkspace | None = None

    def __post_init__(self) -> None:
        if self.workspace is not None and not isinstance(self.workspace, _KimiCPCollectiveWorkspace):
            raise TypeError("workspace must be created by AllGatherKimiCPTransport.prepare")

    @staticmethod
    def required_workspace_bytes(
        plan: KimiCPPlan,
        *,
        heads: int,
        key_dim: int,
        value_dim: int,
    ) -> int:
        """Return local plus gathered FP32 payload bytes without allocating."""
        if not isinstance(plan, KimiCPPlan):
            raise TypeError("plan must be a KimiCPPlan")
        for name, value in {"heads": heads, "key_dim": key_dim, "value_dim": value_dim}.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        return plan.world_size * heads * (value_dim + key_dim) * key_dim * torch.float32.itemsize

    @classmethod
    def prepare(
        cls,
        plan: KimiCPPlan,
        *,
        heads: int,
        key_dim: int,
        value_dim: int,
        device: torch.device | str | int,
    ) -> AllGatherKimiCPTransport:
        if not isinstance(plan, KimiCPPlan):
            raise TypeError("plan must be a KimiCPPlan")
        return cls(
            _KimiCPCollectiveWorkspace.allocate(
                world_size=plan.world_size,
                heads=heads,
                key_dim=key_dim,
                value_dim=value_dim,
                device=device,
            )
        )

    @property
    def workspace_bytes(self) -> int:
        return 0 if self.workspace is None else self.workspace.nbytes

    def boundary_summaries(
        self,
        h: torch.Tensor,
        transition: torch.Tensor,
        *,
        plan: KimiCPPlan,
        group: dist.ProcessGroup | None,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_local_boundary_summaries(h, transition)
        if self.workspace is None:
            h_all = _all_gather(h, group, plan.world_size)
            transition_all = _all_gather(transition, group, plan.world_size)
        else:
            workspace = self.workspace
            workspace.validate(h, transition, world_size=plan.world_size)
            if plan.world_size == 1:
                h_all = h.unsqueeze(0)
                transition_all = transition.unsqueeze(0)
            else:
                _all_gather_into(
                    workspace.gathered_h,
                    h,
                    group=group,
                )
                _all_gather_into(
                    workspace.gathered_transition,
                    transition,
                    group=group,
                )
                h_all = workspace.gathered_h.view(plan.world_size, *h.shape)
                transition_all = workspace.gathered_transition.view(plan.world_size, *transition.shape)
        return _select_boundary_summaries(
            h_all,
            transition_all,
            rank=plan.rank,
            num_remote_ranks=plan.post_num_ranks if reverse else plan.pre_num_ranks,
            reverse=reverse,
        )


_DEFAULT_BOUNDARY_TRANSPORT = AllGatherKimiCPTransport()


def _validate_boundary_summaries(
    summaries: tuple[torch.Tensor, torch.Tensor],
    h_reference: torch.Tensor,
    transition_reference: torch.Tensor,
    *,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(summaries, tuple) or len(summaries) != 2:
        raise TypeError("KimiCPBoundaryTransport must return an (H, transition) tuple")
    expected = (
        (rows, *h_reference.shape),
        (rows, *transition_reference.shape),
    )
    for name, tensor, reference, shape in zip(
        ("H", "transition"),
        summaries,
        (h_reference, transition_reference),
        expected,
        strict=True,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"KimiCPBoundaryTransport {name} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"KimiCPBoundaryTransport {name} must remain FP32")
        if tensor.device != reference.device or tuple(tensor.shape) != shape:
            raise ValueError(
                f"KimiCPBoundaryTransport {name} mismatch: expected shape/device "
                f"{shape} on {reference.device}, got {tuple(tensor.shape)} on {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"KimiCPBoundaryTransport {name} must be contiguous")
    return summaries


def _compose_affine_summaries(
    state_updates: torch.Tensor,
    transitions: torch.Tensor,
    *,
    reverse: bool,
) -> torch.Tensor:
    """Compose ordered FP32 state maps without changing their precision."""
    if state_updates.ndim != 4 or transitions.ndim != 4:
        raise ValueError("ordered KDA summaries must have [ranks, heads, rows, cols] layout")
    if state_updates.shape[0] != transitions.shape[0]:
        raise ValueError("ordered KDA state updates and transitions must have the same rank count")
    if state_updates.dtype != torch.float32 or transitions.dtype != torch.float32:
        raise TypeError("ordered KDA summary composition requires FP32 inputs")
    if state_updates.device != transitions.device:
        raise ValueError("ordered KDA summaries must be on the same device")
    state = state_updates.new_zeros(state_updates.shape[1:])
    indices = range(state_updates.shape[0] - 1, -1, -1) if reverse else range(state_updates.shape[0])
    for index in indices:
        state = torch.matmul(state, transitions[index]) + state_updates[index]
    return state.contiguous()


def _validate_hopper_state_probe_options(
    options: _KdaOptions,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> None:
    """Reject transformations unsupported by the Hopper backward engine."""
    unsupported = []
    if options.use_qk_l2norm_in_kernel:
        unsupported.append("use_qk_l2norm_in_kernel")
    if options.use_beta_sigmoid_in_kernel:
        unsupported.append("use_beta_sigmoid_in_kernel")
    if options.safe_gate:
        unsupported.append("safe_gate")
    if a_log is not None:
        unsupported.append("a_log")
    if dt_bias is not None:
        unsupported.append("dt_bias")
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            "SM90 Kimi CP requires canonicalized q/k, beta, and log-space g because "
            f"the cuDNN Hopper backward engine does not support in-kernel transforms: {names}"
        )


def _validate_ieee_fp32_composition(device: torch.device) -> None:
    if device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "Kimi CP boundary composition requires IEEE FP32; set torch.backends.cuda.matmul.allow_tf32 = False"
        )


def _state_probe_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    options: _KdaOptions,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover one local affine state map using the ordinary cuDNN KDA API.

    Hopper does not expose the FROST summary graph. KDA is affine in its
    initial state, so evaluating a zero state gives H and evaluating an
    embedded identity gives H + M. State tensors are FP32, hence the boundary
    contract remains high precision.
    """
    kimi_delta_attention, _ = _load_cudnn_kda_ops()
    heads = max(q.shape[1], v.shape[1])
    key_dim = k.shape[-1]
    value_dim = v.shape[-1]
    if value_dim < key_dim:
        raise ValueError(
            f"SM90 Kimi CP state probing requires value_dim >= key_dim; got value_dim={value_dim}, key_dim={key_dim}"
        )
    state_shape = (1, heads, value_dim, key_dim)
    zero = torch.zeros(state_shape, dtype=torch.float32, device=q.device)
    probe = zero.clone()
    probe[:, :, :key_dim, :].copy_(torch.eye(key_dim, dtype=torch.float32, device=q.device))
    kwargs = options.kwargs(None, None)

    def final_state(initial_state: torch.Tensor) -> torch.Tensor:
        outputs = kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=options.scale,
            initial_state=initial_state,
            output_final_state=True,
            plan_name=options.plan_name,
            **kwargs,
        )
        return outputs[1].float()

    state_update = final_state(zero)[0].contiguous()
    probed = final_state(probe)[0]
    transition = (probed[:, :key_dim, :] - state_update[:, :key_dim, :]).contiguous()
    _validate_local_boundary_summaries(state_update, transition)
    return state_update, transition


class _CudnnKdaCPHopper(torch.autograd.Function):
    """SM90 CP using ordinary Hopper state inputs instead of FROST summaries."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        plan: KimiCPPlan,
        group: dist.ProcessGroup | None,
        options: _KdaOptions,
        boundary_transport: KimiCPBoundaryTransport,
    ) -> torch.Tensor:
        _validate_hopper_state_probe_options(options)
        kimi_delta_attention, _ = _load_cudnn_kda_ops()

        last_start = plan.last_boundary_start
        last_cu = cu_seqlens[-2:] - last_start
        h_last, transition_last = _state_probe_summary(
            q[last_start:], k[last_start:], v[last_start:], g[last_start:], beta[last_start:], last_cu, options=options
        )
        preceding_h, preceding_transition = _validate_boundary_summaries(
            boundary_transport.boundary_summaries(h_last, transition_last, plan=plan, group=group, reverse=False),
            h_last,
            transition_last,
            rows=plan.pre_num_ranks,
        )
        incoming_state = _compose_affine_summaries(preceding_h, preceding_transition, reverse=False)
        initial_state = incoming_state.new_zeros((plan.num_local_sequences, *incoming_state.shape))
        initial_state[0].copy_(incoming_state)

        first_end = plan.first_boundary_end
        if last_start == 0 and first_end == q.shape[0]:
            transition_first = transition_last
        else:
            _, transition_first = _state_probe_summary(
                q[:first_end], k[:first_end], v[:first_end], g[:first_end], beta[:first_end], cu_seqlens[:2], options=options
            )

        outputs = kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=options.scale,
            initial_state=initial_state,
            output_final_state=False,
            plan_name=options.plan_name,
            **options.attention_kwargs(None, None),
        )
        output = outputs[0]
        state_checkpoints = outputs[2] if len(outputs) > 2 else q.new_empty(0)
        ctx.plan = plan
        ctx.group = group
        ctx.options = options
        ctx.boundary_transport = boundary_transport
        ctx.save_for_backward(q, k, v, g, beta, cu_seqlens, incoming_state, transition_first, state_checkpoints)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        _, kimi_delta_attention_bwd = _load_cudnn_kda_ops()
        q, k, v, g, beta, cu_seqlens, incoming_state, transition_first, state_checkpoints = ctx.saved_tensors
        plan = ctx.plan
        options = ctx.options
        _validate_ieee_fp32_composition(q.device)
        initial_state = incoming_state.new_zeros((plan.num_local_sequences, *incoming_state.shape))
        initial_state[0].copy_(incoming_state)
        grad_output = grad_output.contiguous()
        scale = options.scale if options.scale is not None else q.shape[-1] ** -0.5

        first_end = plan.first_boundary_end
        zero_first = initial_state.new_zeros((1, *initial_state.shape[1:]))
        first_grads = kimi_delta_attention_bwd(
            grad_output[:first_end],
            q[:first_end],
            k[:first_end],
            v[:first_end],
            g[:first_end],
            beta[:first_end],
            cu_seqlens[:2],
            float(scale),
            initial_state=zero_first,
            checkpoint_every_n_tokens=0,
            batch_invariant=options.batch_invariant,
            allow_neg_eigval=options.allow_neg_eigval,
            plan_name=options.plan_name,
        )
        g_local = first_grads[5][0].float().contiguous()
        transition_backward = transition_first.transpose(-1, -2).contiguous()
        following_h, following_transition = _validate_boundary_summaries(
            ctx.boundary_transport.boundary_summaries(g_local, transition_backward, plan=plan, group=ctx.group, reverse=True),
            g_local,
            transition_backward,
            rows=plan.post_num_ranks,
        )
        outgoing_grad = _compose_affine_summaries(following_h, following_transition, reverse=True)
        d_final_state = initial_state.new_zeros(initial_state.shape)
        d_final_state[-1].copy_(outgoing_grad)

        dq, dk, dv, dg, dbeta, _, _, _ = kimi_delta_attention_bwd(
            grad_output,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            float(scale),
            initial_state=initial_state,
            d_final_state=d_final_state,
            state_checkpoints=state_checkpoints if state_checkpoints.numel() else None,
            checkpoint_every_n_tokens=options.checkpoint_every_n_tokens,
            batch_invariant=options.batch_invariant,
            allow_neg_eigval=options.allow_neg_eigval,
            plan_name=options.plan_name,
        )
        return dq, dk, dv, dg, dbeta, None, None, None, None, None


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
        boundary_transport: KimiCPBoundaryTransport,
    ) -> torch.Tensor:
        kimi_delta_attention, kimi_delta_attention_summary, _, _ = _load_cudnn_cp_ops()
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
        preceding_h, preceding_transition = _validate_boundary_summaries(
            boundary_transport.boundary_summaries(
                h_local[0],
                transition_local[0],
                plan=plan,
                group=group,
                reverse=False,
            ),
            h_local[0],
            transition_local[0],
            rows=plan.pre_num_ranks,
        )

        # cuDNN owns the ordered FP32 summary composition. Passing an empty
        # leading dimension on the first rank keeps one stable return contract.
        attention_outputs = kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=options.scale,
            output_final_state=False,
            plan_name=options.plan_name,
            preceding_state_update=preceding_h,
            preceding_transition=preceding_transition,
            **options.attention_kwargs(a_log, dt_bias),
        )
        output, _, state_checkpoints, composed_initial_state = attention_outputs

        ctx.plan = plan
        ctx.group = group
        ctx.options = options
        ctx.boundary_transport = boundary_transport
        ctx.a_log_present = a_log is not None
        ctx.dt_bias_present = dt_bias is not None
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            state_checkpoints,
            composed_initial_state,
            *(tensor for tensor in (a_log, dt_bias) if tensor is not None),
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        _, _, kimi_delta_attention_summary_bwd, kimi_delta_attention_bwd = _load_cudnn_cp_ops()
        saved = iter(ctx.saved_tensors)
        q = next(saved)
        k = next(saved)
        v = next(saved)
        g = next(saved)
        beta = next(saved)
        cu_seqlens = next(saved)
        state_checkpoints = next(saved)
        composed_initial_state = next(saved)
        a_log = next(saved) if ctx.a_log_present else None
        dt_bias = next(saved) if ctx.dt_bias_present else None
        plan = ctx.plan
        options = ctx.options
        grad_output = grad_output.contiguous()

        # The reverse carry is determined only by the first local span. cuDNN
        # backward apply consumes the summary transition in its stored M^T domain.
        first_end = plan.first_boundary_end
        g_local, transition_backward, backward_context = kimi_delta_attention_summary_bwd(
            q[:first_end],
            k[:first_end],
            g[:first_end],
            beta[:first_end],
            grad_output[:first_end],
            cu_seqlens[:2],
            scale=options.scale,
            output_transition=True,
            output_context=True,
            checkpoint_every_n_tokens=options.checkpoint_every_n_tokens,
            plan_name=options.summary_plan_name,
            **options.kwargs(a_log, dt_bias),
        )
        following_h, following_transition = _validate_boundary_summaries(
            ctx.boundary_transport.boundary_summaries(
                g_local[0],
                transition_backward[0],
                plan=plan,
                group=ctx.group,
                reverse=True,
            ),
            g_local[0],
            transition_backward[0],
            rows=plan.post_num_ranks,
        )

        scale = options.scale if options.scale is not None else q.shape[-1] ** -0.5
        dq, dk, dv, dg, dbeta, _, da_log, ddt_bias = kimi_delta_attention_bwd(
            grad_output,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            float(scale),
            state_checkpoints=state_checkpoints if state_checkpoints.numel() else None,
            checkpoint_every_n_tokens=options.checkpoint_every_n_tokens,
            use_qk_l2norm_in_kernel=options.use_qk_l2norm_in_kernel,
            batch_invariant=options.batch_invariant,
            use_beta_sigmoid_in_kernel=options.use_beta_sigmoid_in_kernel,
            allow_neg_eigval=options.allow_neg_eigval,
            safe_gate=options.safe_gate,
            gate_lower_bound=options.lower_bound,
            gate_domain=options.gate_domain,
            a_log=a_log,
            dt_bias=dt_bias,
            plan_name=options.plan_name,
            following_state_grad=following_h,
            following_transition=following_transition,
            composed_initial_state=(composed_initial_state if composed_initial_state.numel() else None),
            context=backward_context,
        )
        da_log = da_log if ctx.a_log_present else None
        ddt_bias = ddt_bias if ctx.dt_bias_present else None
        return dq, dk, dv, dg, dbeta, None, da_log, ddt_bias, None, None, None, None


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


def _resolve_kimi_cp_backend(backend: str, device: torch.device) -> str:
    if backend not in ("auto", "frost-summary", "hopper-state-probe"):
        raise ValueError("backend must be 'auto', 'frost-summary', or 'hopper-state-probe'")
    if backend != "auto":
        return backend
    major, _ = torch.cuda.get_device_capability(device)
    if major == 9:
        return "hopper-state-probe"
    if major >= 10:
        return "frost-summary"
    raise RuntimeError(f"cuDNN Kimi CP does not support CUDA compute capability {major}.x")


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
    checkpoint_every_n_tokens: int = 0,
    plan_name: str | None = None,
    summary_plan_name: str | None = None,
    boundary_transport: KimiCPBoundaryTransport | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Run differentiable cuDNN KDA over one contiguous packed-THD CP shard.

    cuLA owns boundary planning, collectives, ordered summary propagation and
    distributed autograd.  cuDNN Frontend owns local KDA summaries, ordered
    FP32 summary composition, and local forward/backward.  The summary dtype is
    an invariant rather than a user option so large-CP behavior cannot silently
    degrade.

    ``plan`` should be built once per packed microbatch and reused by all KDA
    layers.  Inputs use cuDNN's canonical THD contract; framework-specific
    layout and gate conversions belong in the framework adapter.

    ``boundary_transport`` defaults to exact ordered summary all-gather with lazy
    storage. Large-CP callers should prepare one
    :class:`AllGatherKimiCPTransport` per concurrent CUDA stream and reuse it
    across layers to avoid repeated collective-buffer allocations.

    ``checkpoint_every_n_tokens=0`` lets cuDNN regenerate the chunk-state
    series during backward. Passing 16 retains every native chunk state for
    exact backward reuse. Coarser upstream cadences still recompute seeded
    spans, were slower in the 1M study, and are intentionally not exposed.
    Because exact checkpoints are large and live across the forward/backward
    interval, cuLA never enables them implicitly; callers should select 16
    only from measured whole-model memory/latency tradeoffs.
    """

    if not isinstance(plan, KimiCPPlan):
        raise TypeError("plan must be a KimiCPPlan")
    if backend not in ("auto", "frost-summary", "hopper-state-probe"):
        raise ValueError("backend must be 'auto', 'frost-summary', or 'hopper-state-probe'")
    resolved_transport = _DEFAULT_BOUNDARY_TRANSPORT if boundary_transport is None else boundary_transport
    if not isinstance(resolved_transport, KimiCPBoundaryTransport):
        raise TypeError("boundary_transport must implement KimiCPBoundaryTransport")
    checkpoint_every_n_tokens = _validate_checkpoint_every_n_tokens(checkpoint_every_n_tokens)
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
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        plan_name=plan_name,
        summary_plan_name=summary_plan_name,
    )
    if plan.world_size == 1:
        kimi_delta_attention, _ = _load_cudnn_kda_ops()
        attention_outputs = kimi_delta_attention(
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
            **options.attention_kwargs(a_log, dt_bias),
        )
        return attention_outputs[0]
    resolved_backend = _resolve_kimi_cp_backend(backend, q.device)
    if resolved_backend == "hopper-state-probe":
        _validate_hopper_state_probe_options(options, a_log, dt_bias)
        _validate_ieee_fp32_composition(q.device)
        return _CudnnKdaCPHopper.apply(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            cu_seqlens.contiguous(),
            plan,
            group,
            options,
            resolved_transport,
        )
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
        resolved_transport,
    )


__all__ = [
    "AllGatherKimiCPTransport",
    "KimiCPBoundaryTransport",
    "cudnn_kda_cp",
]
