# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable cuDNN index pooling for packed THD inputs."""

from __future__ import annotations

from functools import cache
from importlib import import_module

import torch
from torch.autograd.function import once_differentiable

from cula.hybrid.plan import IndexPoolPlan


@cache
def _load_cudnn_index_pool_ops():
    try:
        cudnn = import_module("cudnn")
        namespace = cudnn.CSA
        forward = namespace.csa_compressor_forward_wrapper
        backward = namespace.csa_compressor_backward_wrapper
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "cuDNN index pooling requires cuDNN Frontend with the CSA compressor APIs; "
            "install cuda-linear-attention[cudnn-cp]"
        ) from error
    return forward, backward


def _validate_metadata(tensor: torch.Tensor, *, name: str, entries: int, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != (entries,):
        raise ValueError(f"{name} must have shape [{entries}], got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have int32 dtype")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    *,
    plan: IndexPoolPlan,
) -> None:
    for name, tensor in (("kv", kv), ("score", score), ("ape", ape)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if kv.device.type != "cuda":
        raise ValueError("cuDNN index pooling requires CUDA operands")
    if score.device != kv.device or ape.device != kv.device:
        raise ValueError("kv, score and ape must be on the same device")
    if kv.shape != (plan.total_tokens, plan.input_width):
        raise ValueError(
            f"kv must have shape [{plan.total_tokens}, {plan.input_width}], got {tuple(kv.shape)}"
        )
    if score.shape != kv.shape:
        raise ValueError(f"score shape {tuple(score.shape)} must match kv shape {tuple(kv.shape)}")
    if ape.shape != (plan.ratio, plan.input_width):
        raise ValueError(
            f"ape must have shape [{plan.ratio}, {plan.input_width}], got {tuple(ape.shape)}"
        )
    if kv.dtype != torch.bfloat16 or score.dtype != torch.bfloat16:
        raise TypeError("kv and score must have bfloat16 dtype")
    if ape.dtype != torch.float32:
        raise TypeError("ape must have float32 dtype")
    for name, tensor in (("kv", kv), ("score", score), ("ape", ape)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; implicit hot-path layout copies are not allowed")
    entries = plan.num_sequences + 1
    _validate_metadata(cu_seqlens, name="cu_seqlens", entries=entries, device=kv.device)
    _validate_metadata(cu_seqlens_comp, name="cu_seqlens_comp", entries=entries, device=kv.device)
    if cu_seqlens.device.type == "cpu":
        if tuple(int(value) for value in cu_seqlens.tolist()) != plan.cu_seqlens:
            raise ValueError("cu_seqlens values do not match IndexPoolPlan")
        if tuple(int(value) for value in cu_seqlens_comp.tolist()) != plan.cu_seqlens_comp:
            raise ValueError("cu_seqlens_comp values do not match IndexPoolPlan")


class _CudnnIndexPool(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        plan: IndexPoolPlan,
        deterministic: bool,
    ) -> torch.Tensor:
        forward, _ = _load_cudnn_index_pool_ops()
        result = forward(
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            ratio=plan.ratio,
            head_dim=plan.head_dim,
            coff=plan.coff,
            total_comp=plan.total_comp,
        )
        ctx.plan = plan
        ctx.deterministic = deterministic
        ctx.save_for_backward(kv, score, ape, cu_seqlens, cu_seqlens_comp)
        return result["out"]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        _, backward = _load_cudnn_index_pool_ops()
        kv, score, ape, cu_seqlens, cu_seqlens_comp = ctx.saved_tensors
        plan = ctx.plan
        result = backward(
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            grad_out.contiguous(),
            ratio=plan.ratio,
            head_dim=plan.head_dim,
            coff=plan.coff,
            deterministic=ctx.deterministic,
        )
        return result["grad_kv"], result["grad_score"], result["grad_ape"], None, None, None, None


def cudnn_index_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    *,
    plan: IndexPoolPlan,
    deterministic: bool = False,
) -> torch.Tensor:
    """Pool packed token rows with cuDNN's FP32-intermediate compressor.

    cuLA owns the static packed metadata and PyTorch autograd connection;
    cuDNN Frontend owns the fused forward/backward kernels.  Inputs must
    already use the native contiguous THD rows.  This function never performs
    an implicit layout conversion or changes the kernel's numerical contract.

    Set ``deterministic=True`` to use cuDNN's fixed-order dAPE reduction.  This
    affects reproducibility and workspace scheduling, not arithmetic precision.
    """

    if not isinstance(plan, IndexPoolPlan):
        raise TypeError("plan must be an IndexPoolPlan")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
    _validate_inputs(kv, score, ape, cu_seqlens, cu_seqlens_comp, plan=plan)
    return _CudnnIndexPool.apply(kv, score, ape, cu_seqlens, cu_seqlens_comp, plan, deterministic)


__all__ = ["cudnn_index_pool"]
