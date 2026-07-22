# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Independent host reference for the kernel's published auxiliary operands."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .collective_inverse_reference import MATRIX_SIZE, collective_inverse_reference


@dataclass(frozen=True)
class PublishedAuxReference:
    """Final consumer operands after the kernel's two BF16 publication stores."""

    qk_bf16: torch.Tensor
    inverse_kk_beta_bf16: torch.Tensor
    staged_inverse_fp16: torch.Tensor


def published_aux_reference(
    qk_epilogue: torch.Tensor,
    kk_physical: torch.Tensor,
    beta: torch.Tensor,
    valid_tokens: int,
) -> PublishedAuxReference:
    """Apply the kernel's post-inverse ``beta[col]`` and BF16 stores."""

    for name, tensor in (("qk_epilogue", qk_epilogue), ("kk_physical", kk_physical)):
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if tensor.shape != (MATRIX_SIZE, MATRIX_SIZE):
            raise ValueError(f"{name} must have shape ({MATRIX_SIZE}, {MATRIX_SIZE})")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype")
    if beta.device.type != "cpu":
        raise ValueError("beta must be a CPU tensor")
    if beta.ndim != 1 or beta.numel() != MATRIX_SIZE:
        raise ValueError(f"beta must have shape ({MATRIX_SIZE},)")
    if not beta.is_floating_point():
        raise ValueError("beta must use a floating-point dtype")
    if not 0 < valid_tokens <= MATRIX_SIZE:
        raise ValueError(f"valid_tokens must be in [1, {MATRIX_SIZE}]")
    if not bool((beta[valid_tokens:] == 0).all()):
        raise ValueError("beta's ragged tail must use the kernel's zero padding")

    inverse = collective_inverse_reference(kk_physical, valid_tokens).inverse
    inverse_kk_beta = inverse.float() * beta.float()[None, :]
    return PublishedAuxReference(
        qk_bf16=qk_epilogue.to(torch.bfloat16),
        inverse_kk_beta_bf16=inverse_kk_beta.to(torch.bfloat16),
        staged_inverse_fp16=inverse,
    )
