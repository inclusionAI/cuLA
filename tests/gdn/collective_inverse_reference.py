# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Independent staged-FP16 host reference for the 64x64 collective inverse.

The reference consumes the physical KK tile directly: the physical diagonal
is discarded, the strict upper triangle must already be exact zero, and a
ragged tail is promoted from zero padding to identity padding.

Every shared-memory or accumulator-to-operand half store in the device
collective is represented by an explicit FP16 round. The matrix products use
FP32 accumulation. The final 32-to-64 merge also preserves the two separately
rounded K=16 partial products before their FP16 reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MATRIX_SIZE = 64
BASE_BLOCK_SIZE = 8
STAGE_SIZES = (8, 16, 32, 64)


@dataclass(frozen=True)
class CollectiveInverseReference:
    """Staged operand, stage snapshots, and final inverse for one KK tile."""

    staged_kk_physical: torch.Tensor
    normalized_operand: torch.Tensor
    stage_8: torch.Tensor
    stage_16: torch.Tensor
    stage_32: torch.Tensor
    inverse: torch.Tensor
    inverse_residual: float


def _require_cpu_matrix(kk_physical: torch.Tensor) -> None:
    if kk_physical.device.type != "cpu":
        raise ValueError("kk_physical must be a CPU tensor")
    if kk_physical.shape != (MATRIX_SIZE, MATRIX_SIZE):
        raise ValueError(f"kk_physical must have shape ({MATRIX_SIZE}, {MATRIX_SIZE})")
    if not kk_physical.is_floating_point():
        raise ValueError("kk_physical must use a floating-point dtype")


def _round_fp16(value: torch.Tensor) -> torch.Tensor:
    """Model one device FP32-to-FP16 register or shared-memory store."""

    return value.to(torch.float16)


def _validate_physical_contract(staged: torch.Tensor, valid_tokens: int) -> None:
    row = torch.arange(MATRIX_SIZE)[:, None]
    col = torch.arange(MATRIX_SIZE)[None, :]
    active_strict_lower = (row < valid_tokens) & (col < valid_tokens) & (row > col)
    active_diagonal = (row == col) & (row < valid_tokens)

    # The diagonal is deliberately excluded because the device collective
    # replaces its physical value with one before inversion.
    if not bool(torch.isfinite(staged[active_strict_lower]).all()):
        raise ValueError("active strict-lower kk_physical entries must remain finite after FP16 staging")

    must_be_zero = ~(active_strict_lower | active_diagonal)
    if not bool((staged[must_be_zero] == 0).all()):
        raise ValueError("kk_physical upper triangle and ragged off-diagonal tail must be exact zero")


def _normalize_operand(staged: torch.Tensor) -> torch.Tensor:
    """Replace the garbage diagonal by identity and retain strict lower data."""

    identity = torch.eye(MATRIX_SIZE, dtype=torch.float16)
    return torch.tril(staged, diagonal=-1) + identity


def _inverse_unit_lower_8x8(block: torch.Tensor) -> torch.Tensor:
    """Model warp-shuffle row elimination for one 8x8 block."""

    rows = block.to(torch.float32).clone()
    for src_row in range(BASE_BLOCK_SIZE - 1):
        source_prefix = rows[src_row, :src_row].clone()
        for target_row in range(src_row + 1, BASE_BLOCK_SIZE):
            row_scale = -rows[target_row, src_row].clone()
            if src_row:
                rows[target_row, :src_row] += row_scale * source_prefix
            rows[target_row, src_row] = row_scale
    return _round_fp16(rows)


def _merge_inverse_blocks(working: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply one blockwise ``-inv(D) C inv(A)`` hierarchy level."""

    merged = working.clone()
    combined_size = 2 * block_size
    for start in range(0, MATRIX_SIZE, combined_size):
        middle = start + block_size
        end = start + combined_size
        a_inv = working[start:middle, start:middle]
        coupling = working[middle:end, start:middle]
        d_inv = working[middle:end, middle:end]

        # Convert the first FP32 HMMA accumulator into a half operand before
        # multiplying by inv(A).
        d_c = _round_fp16(-(d_inv.float() @ coupling.float()))
        if block_size == 32:
            # Four warps split the final second GEMM at K=16.  Each partial is
            # rounded before one warp reloads and reduces the pair to FP16.
            split = block_size // 2
            partial_0 = _round_fp16(d_c[:, :split].float() @ a_inv[:split, :].float())
            partial_1 = _round_fp16(d_c[:, split:].float() @ a_inv[split:, :].float())
            lower_left = _round_fp16(partial_0.float() + partial_1.float())
        else:
            lower_left = _round_fp16(d_c.float() @ a_inv.float())
        merged[middle:end, start:middle] = lower_left
    return merged


def inverse_residual(operand: torch.Tensor, inverse: torch.Tensor) -> float:
    """Return ``max(abs(operand @ inverse - I))`` with FP32 accumulation."""

    _require_cpu_matrix(operand)
    _require_cpu_matrix(inverse)
    identity = torch.eye(MATRIX_SIZE, dtype=torch.float32)
    residual = operand.float() @ inverse.float() - identity
    return float(residual.abs().max().item())


def collective_inverse_reference(
    kk_physical: torch.Tensor,
    valid_tokens: int,
) -> CollectiveInverseReference:
    """Compute the staged 64x64 half collective inverse.

    Args:
        kk_physical: One physical inclusive-lower KK tile.  The active
            diagonal may contain arbitrary values, including non-finite
            garbage, because the inverse replaces it by identity.  Upper and
            every ragged-tail entry must be exact zero.
        valid_tokens: Number of active rows and columns in ``[1, 64]``.
    """

    _require_cpu_matrix(kk_physical)
    if not 0 < valid_tokens <= MATRIX_SIZE:
        raise ValueError(f"valid_tokens must be in [1, {MATRIX_SIZE}]")

    staged = _round_fp16(kk_physical)
    _validate_physical_contract(staged, valid_tokens)
    normalized = _normalize_operand(staged)

    stage_8 = normalized.clone()
    for start in range(0, MATRIX_SIZE, BASE_BLOCK_SIZE):
        end = start + BASE_BLOCK_SIZE
        stage_8[start:end, start:end] = _inverse_unit_lower_8x8(normalized[start:end, start:end])

    stage_16 = _merge_inverse_blocks(stage_8, 8)
    stage_32 = _merge_inverse_blocks(stage_16, 16)
    stage_64 = _merge_inverse_blocks(stage_32, 32)
    residual = inverse_residual(normalized, stage_64)

    return CollectiveInverseReference(
        staged_kk_physical=staged,
        normalized_operand=normalized,
        stage_8=stage_8,
        stage_16=stage_16,
        stage_32=stage_32,
        inverse=stage_64,
        inverse_residual=residual,
    )
