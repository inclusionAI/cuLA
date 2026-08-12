# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SM90 KDA prefill wrapper for the two-kernel K1+K2 CuTeDSL path"""

from typing import Literal

import torch
from torch.amp import custom_bwd, custom_fwd

from cula.ops.kda.cp_mode import CPMode
from cula.ops.kda.sm90.cp.plan import plan_prefill
from cula.ops.kda.sm90.fwd import _seq_tiles_from_problem, _validate_inputs, _validate_launch_options, flash_kda_fwd
from cula.utils import assert_hopper


def _beta_logits_bf16(beta: torch.Tensor) -> torch.Tensor:
    return torch.logit(beta.float(), eps=1e-6).to(torch.bfloat16)


def _cast_beta_bf16(beta: torch.Tensor) -> torch.Tensor:
    return beta if beta.dtype == torch.bfloat16 else beta.to(torch.bfloat16)


def _guarded_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    cu_seqlens_cpu: torch.IntTensor | None = None,
    use_intracard_cp: CPMode | None = None,
    beta_is_logits: bool = False,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
):
    with torch.cuda.device_of(q):
        return HopperChunkKDAFunction._forward_impl(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state,
            lower_bound,
            cu_seqlens,
            cu_seqlens_cpu,
            use_intracard_cp,
            beta_is_logits,
            out,
            final_state,
        )


class HopperChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, *args):
        return _guarded_forward(*args)

    @staticmethod
    def _forward_impl(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        lower_bound,
        cu_seqlens,
        cu_seqlens_cpu,
        use_intracard_cp,
        beta_is_logits,
        out,
        final_state,
    ):
        batch_size, seq_len, num_heads, head_dim = q.shape

        if out is None:
            out = torch.empty_like(v)

        n_seqs = batch_size
        if cu_seqlens is not None:
            n_seqs = cu_seqlens.numel() - 1

        if not output_final_state:
            final_state = None
        elif final_state is None:
            final_state = torch.empty(
                n_seqs,
                num_heads,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=q.device,
            )

        if beta_is_logits:
            beta = _cast_beta_bf16(beta)
        else:
            beta = _beta_logits_bf16(beta)

        problem = _validate_inputs(q, k, v, g, beta, A_log, dt_bias, initial_state, final_state, cu_seqlens, cu_seqlens_cpu)
        _validate_launch_options(q, out, lower_bound, True)
        plan = plan_prefill(
            _seq_tiles_from_problem(problem),
            num_heads,
            q.device,
            use_intracard_cp,
        )
        if not plan.trivial:
            from cula.ops.kda.sm90.cp.driver import _run_cp

            _run_cp(
                plan,
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                out=out,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=cu_seqlens,
                state_transposed=False,
                _problem=problem,
            )
        else:
            flash_kda_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                out=out,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                state_transposed=False,
                use_gate_in_kernel=True,
                _problem=problem,
            )

        return out, final_state

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet.")


@torch.compiler.disable
def cula_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    use_gate_in_kernel: bool = True,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    use_intracard_cp: Literal["auto"] | bool | None = None,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    **kwargs,
):
    r"""
    Hopper (SM90) KDA forward prefill using CuTeDSL two-kernel pipeline.

    Gate preprocessing (A_log, dt_bias, lower_bound) and L2-norm are handled
    internally by the K1 kernel. This SM90 CuTeDSL path supports only the safe
    in-kernel gate mode: ``use_gate_in_kernel=True`` and ``safe_gate=True``.
    ``use_qk_l2norm_in_kernel`` is accepted for API compatibility; CuTeDSL
    always applies L2-norm internally.

    Tensor inputs must be contiguous and ``g`` must be bfloat16; no implicit
    copies or casts are made, non-conforming inputs raise.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Accepted for API compatibility; CuTeDSL always applies L2-norm
            internally. Default: `False`.
        use_beta_sigmoid_in_kernel (bool):
            When `True`, `beta` is pre-sigmoid logits and is passed straight
            to the kernel (which always applies sigmoid internally) — no
            host-side logit round-trip. When `False` (FLA convention),
            `beta` is post-sigmoid and is converted back to logits first.
            Default: `False`.
        use_gate_in_kernel (bool):
            Must be `True`; CuTeDSL computes the gate internally. Default: `True`.
        safe_gate (bool):
            Must be `True`; unsupported unsafe gate inputs are rejected. Default: `False`.
        lower_bound (Optional[float]):
            Lower bound for the forget gate activation function. Required when
            `safe_gate=True`; must be in `[-5, 0)`. Default: `None`.
        cu_seqlens (torch.IntTensor):
            Cumulative sequence lengths of shape `[N+1]`, int32.
        cu_seqlens_cpu (Optional[torch.IntTensor]):
            Optional CPU copy of `cu_seqlens` (same values), passed via kwargs, to
            skip the GPU->host sync when first building varlen metadata. Trusted,
            not verified (FLA convention). Default: `None`.
        chunk_indices (torch.IntTensor):
            Accepted for API compatibility; unused by CuTeDSL.
        use_intracard_cp (Literal["auto"] | bool):
            Whether to use the SM90 intracard-CP path when profitable. ``True``
            requires CP support and raises on rejection, ``"auto"`` falls back
            to the serial K1+K2 path, and ``False`` disables CP.
        out (Optional[torch.Tensor]):
            Preallocated output buffer, bf16, same shape as ``v``, written in
            place (also returned). ``None`` allocates per call. Default: `None`.
        final_state (Optional[torch.Tensor]):
            Preallocated final-state buffer, fp32, shape `[N, H, V, K]`,
            written in place (also returned). Requires
            ``output_final_state=True``. ``None`` allocates per call.
            Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.
    """
    assert_hopper(q.device)
    if not use_gate_in_kernel:
        raise NotImplementedError(
            "SM90 CuTeDSL KDA prefill only supports use_gate_in_kernel=True. "
            "Passing preprocessed gates would otherwise fall back to the slow reference path."
        )
    if not safe_gate:
        raise NotImplementedError("SM90 CuTeDSL KDA prefill only supports safe_gate=True.")
    num_qk_heads, head_dim = q.shape[2], q.shape[3]
    num_kv_heads = v.shape[2]
    A_log = kwargs.pop("A_log", None)
    dt_bias = kwargs.pop("dt_bias", None)
    cu_seqlens_cpu = kwargs.pop("cu_seqlens_cpu", None)
    use_intracard_cp = CPMode.parse(use_intracard_cp)
    if kwargs:
        raise TypeError(f"cula_kda_prefill got unexpected keyword arguments: {set(kwargs)}")
    if dt_bias is not None and dt_bias.ndim == 1:
        dt_bias = dt_bias.view(num_kv_heads, head_dim)

    if beta.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("beta must be in bfloat16 or float32.")
    if num_kv_heads != num_qk_heads:
        raise NotImplementedError(
            "SM90 CuTeDSL KDA prefill does not support grouped-value attention yet "
            f"(num_kv_heads={num_kv_heads} != num_qk_heads={num_qk_heads}); native GVA is a follow-up change."
        )

    if final_state is not None and not output_final_state:
        raise ValueError("final_state buffer requires output_final_state=True.")

    if scale is None:
        scale = k.shape[-1] ** -0.5
    fwd_args = (
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        lower_bound,
        cu_seqlens,
        cu_seqlens_cpu,
        use_intracard_cp,
        use_beta_sigmoid_in_kernel,
        out,
        final_state,
    )
    # Forward-only op: skip autograd.Function unless a graph could be recorded.
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, g, beta, A_log, dt_bias, initial_state)
    )
    if needs_grad:
        o, final_state = HopperChunkKDAFunction.apply(*fwd_args)
    else:
        o, final_state = _guarded_forward(*fwd_args)
    return o, final_state
