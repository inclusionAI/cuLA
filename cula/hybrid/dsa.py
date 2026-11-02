# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Memory-bounded packed-THD block selection over cuDNN DSA kernels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from operator import index

import torch

from cula._metadata import as_boundaries

MEMORY_EFFICIENT_CANDIDATE_BUDGET_BYTES = 512 << 20
"""Lower-memory policy matching cuDNN QSA's serving-side selection budget."""

DEFAULT_CANDIDATE_BUDGET_BYTES = MEMORY_EFFICIENT_CANDIDATE_BUDGET_BYTES
"""Base FP32 compact-logit workspace used by the adaptive default policy."""

MAX_BALANCED_CANDIDATE_BUDGET_BYTES = 4 << 30
"""Maximum FP32 compact-logit workspace selected by the adaptive policy."""

_CANDIDATE_BUDGET_POSITION_QUANTUM = 128 << 10


def _as_nonnegative(
    values: torch.Tensor | Sequence[int], *, entries: int, name: str
) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor; copying CUDA metadata would synchronize the device")
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if values.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must have int32 or int64 dtype")
        result = tuple(int(value) for value in values.tolist())
    else:
        try:
            result = tuple(index(value) for value in values)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be an integer sequence") from error
    if len(result) != entries:
        raise ValueError(f"{name} must contain {entries} entries")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} entries must be non-negative")
    if any(value >= 2**31 for value in result):
        raise ValueError(f"{name} entries must fit int32")
    return result


def _as_block_prefix(values: torch.Tensor | Sequence[int]) -> tuple[int, ...]:
    """Read a host block prefix, permitting zero-block packed sequences."""

    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError("cu_seqlens_k must be a CPU tensor; copying CUDA metadata would synchronize the device")
        if values.ndim != 1:
            raise ValueError("cu_seqlens_k must be one-dimensional")
        if values.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens_k must have int32 or int64 dtype")
        result = tuple(int(value) for value in values.tolist())
    else:
        try:
            result = tuple(index(value) for value in values)
        except (TypeError, ValueError) as error:
            raise TypeError("cu_seqlens_k must be an integer sequence") from error
    if len(result) < 2 or result[0] != 0:
        raise ValueError("cu_seqlens_k must contain at least two entries and start at zero")
    if any(left > right for left, right in zip(result, result[1:])):
        raise ValueError("cu_seqlens_k must be non-decreasing")
    if result[-1] == 0:
        raise ValueError("the packed input contains no complete key block")
    return result


def _candidate_prefix(position: int, block_size: int) -> int:
    """Sum ``floor(i / block_size)`` for ``i`` in ``[0, position]``."""

    quotient, remainder = divmod(position, block_size)
    return block_size * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def _candidate_floats(position: int, rows: int, block_size: int) -> int:
    return _candidate_prefix(position + rows, block_size) - _candidate_prefix(position, block_size)


def _balanced_candidate_budget(cu_q: tuple[int, ...], offsets: tuple[int, ...]) -> int:
    max_causal_end = max(
        offset + right - left for left, right, offset in zip(cu_q[:-1], cu_q[1:], offsets, strict=True)
    )
    steps = max(1, (max_causal_end + _CANDIDATE_BUDGET_POSITION_QUANTUM - 1) // _CANDIDATE_BUDGET_POSITION_QUANTUM)
    return min(MAX_BALANCED_CANDIDATE_BUDGET_BYTES, steps * DEFAULT_CANDIDATE_BUDGET_BYTES)


@dataclass(frozen=True, slots=True)
class DsaSelectionWindow:
    """One cuDNN THD launch over contiguous packed query and block rows."""

    query_start: int
    query_end: int
    key_start: int
    key_end: int
    cu_seqlens_q: tuple[int, ...]
    cu_seqlens_k: tuple[int, ...]
    q_causal_offsets: tuple[int, ...]
    candidate_offsets: tuple[int, ...]

    @property
    def candidate_floats(self) -> int:
        return self.candidate_offsets[-1]

    @property
    def max_seqlen_q(self) -> int:
        return max(right - left for left, right in zip(self.cu_seqlens_q, self.cu_seqlens_q[1:]))

    @property
    def max_seqlen_k(self) -> int:
        return max(right - left for left, right in zip(self.cu_seqlens_k, self.cu_seqlens_k[1:]))


@dataclass(frozen=True, slots=True)
class DsaSelectionPlan:
    """Static, memory-bounded selection plan for packed indexer queries.

    ``cu_seqlens_k`` describes already-compressed block rows.  The plan keeps
    block ids local to each packed sequence, which is the native contract of
    cuDNN QSA and avoids expanding 512 block ids into 2048 token ids.

    Long sequences are split into exact causal row windows whose compact-logit
    scratch fits ``candidate_budget_bytes``.  Windowing changes neither input
    precision nor the selected set; deterministic tie-breaking remains enabled
    by default at execution.  Whole adjacent sequences are coalesced into one
    launch when they fit the same budget.
    """

    block_size: int
    top_k: int
    cu_seqlens_q: tuple[int, ...]
    cu_seqlens_k: tuple[int, ...]
    q_causal_offsets: tuple[int, ...]
    candidate_budget_bytes: int
    windows: tuple[DsaSelectionWindow, ...]

    @property
    def total_queries(self) -> int:
        return self.cu_seqlens_q[-1]

    @property
    def total_blocks(self) -> int:
        return self.cu_seqlens_k[-1]

    @property
    def num_sequences(self) -> int:
        return len(self.cu_seqlens_q) - 1

    @property
    def candidate_workspace_floats(self) -> int:
        return max(window.candidate_floats for window in self.windows)

    @classmethod
    def from_cu_seqlens(
        cls,
        cu_seqlens_q: torch.Tensor | Sequence[int],
        cu_seqlens_k: torch.Tensor | Sequence[int],
        *,
        top_k: int,
        block_size: int = 4,
        q_causal_offsets: torch.Tensor | Sequence[int] | None = None,
        candidate_budget_bytes: int | None = None,
    ) -> DsaSelectionPlan:
        """Create a plan entirely from host-visible packed metadata.

        A ``None`` budget selects the balanced 512 MiB-per-128K causal-reach
        policy, capped at 4 GiB. Pass an explicit byte count to enforce a fixed
        memory policy across ranks and microbatches.
        """

        cu_q = as_boundaries(cu_seqlens_q, name="cu_seqlens_q")
        cu_k = _as_block_prefix(cu_seqlens_k)
        if len(cu_q) != len(cu_k):
            raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same number of sequences")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 2048:
            raise ValueError("top_k must be an integer in [1, 2048]")
        if candidate_budget_bytes is not None and (
            not isinstance(candidate_budget_bytes, int)
            or isinstance(candidate_budget_bytes, bool)
            or candidate_budget_bytes < torch.float32.itemsize
        ):
            raise ValueError("candidate_budget_bytes must hold at least one FP32 element")

        batch = len(cu_q) - 1
        if cu_q[-1] >= 2**31 or cu_k[-1] >= 2**31:
            raise ValueError("packed query and block prefixes must fit int32")
        offsets = (
            (0,) * batch
            if q_causal_offsets is None
            else _as_nonnegative(q_causal_offsets, entries=batch, name="q_causal_offsets")
        )
        if candidate_budget_bytes is None:
            candidate_budget_bytes = _balanced_candidate_budget(cu_q, offsets)
        budget_floats = candidate_budget_bytes // torch.float32.itemsize
        atoms: list[tuple[int, int, int, int, int, int, int]] = []
        for sequence in range(batch):
            query_length = cu_q[sequence + 1] - cu_q[sequence]
            key_length = cu_k[sequence + 1] - cu_k[sequence]
            position = offsets[sequence]
            if position + query_length >= 2**31:
                raise ValueError(f"sequence {sequence} causal positions must fit int32")
            required_blocks = (position + query_length) // block_size
            if key_length < required_blocks:
                raise ValueError(
                    f"sequence {sequence} has {key_length} compressed blocks but its query geometry requires "
                    f"at least {required_blocks}"
                )
            max_row_candidates = (position + query_length) // block_size
            if max_row_candidates > budget_floats:
                raise ValueError(
                    f"candidate_budget_bytes is too small for one row of sequence {sequence}: "
                    f"need at least {max_row_candidates * torch.float32.itemsize} bytes"
                )

            row = 0
            while row < query_length:
                low, high = row + 1, query_length
                while low < high:
                    middle = (low + high + 1) // 2
                    needed = _candidate_floats(position + row, middle - row, block_size)
                    if needed <= budget_floats:
                        low = middle
                    else:
                        high = middle - 1
                row_end = low
                candidates = _candidate_floats(position + row, row_end - row, block_size)
                atoms.append((sequence, row, row_end, key_length, position + row, candidates, query_length))
                row = row_end

        windows: list[DsaSelectionWindow] = []
        atom = 0
        while atom < len(atoms):
            sequence, row_start, row_end, key_length, position, candidates, query_length = atoms[atom]
            whole = row_start == 0 and row_end == query_length
            group = [atoms[atom]]
            atom += 1
            while whole and atom < len(atoms):
                next_atom = atoms[atom]
                next_sequence, next_start, next_end, _, _, next_candidates, next_length = next_atom
                if (
                    next_sequence != group[-1][0] + 1
                    or next_start != 0
                    or next_end != next_length
                    or sum(item[5] for item in group) + next_candidates > budget_floats
                ):
                    break
                group.append(next_atom)
                atom += 1

            first_sequence = group[0][0]
            last_sequence = group[-1][0]
            query_start = cu_q[first_sequence] + group[0][1]
            query_end = cu_q[last_sequence] + group[-1][2]
            key_start = cu_k[first_sequence]
            key_end = cu_k[last_sequence + 1]
            window_cu_q = [0]
            window_cu_k = [0]
            window_offsets = []
            candidate_offsets = [0]
            for seq, start, end, seq_key_length, seq_position, seq_candidates, _ in group:
                window_cu_q.append(window_cu_q[-1] + end - start)
                window_cu_k.append(window_cu_k[-1] + seq_key_length)
                window_offsets.append(seq_position)
                candidate_offsets.append(candidate_offsets[-1] + seq_candidates)
            windows.append(
                DsaSelectionWindow(
                    query_start=query_start,
                    query_end=query_end,
                    key_start=key_start,
                    key_end=key_end,
                    cu_seqlens_q=tuple(window_cu_q),
                    cu_seqlens_k=tuple(window_cu_k),
                    q_causal_offsets=tuple(window_offsets),
                    candidate_offsets=tuple(candidate_offsets),
                )
            )

        return cls(
            block_size=block_size,
            top_k=top_k,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            q_causal_offsets=offsets,
            candidate_budget_bytes=candidate_budget_bytes,
            windows=tuple(windows),
        )


@dataclass(frozen=True, slots=True)
class _DsaWindowMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    q_causal_offsets: torch.Tensor | None
    candidate_offsets: torch.Tensor


@dataclass(frozen=True, slots=True)
class DsaSelectionWorkspace:
    """Caller-owned outputs, compact-logit scratch, and CUDA graph constants."""

    plan: DsaSelectionPlan
    candidates: torch.Tensor
    block_ids: torch.Tensor
    scores: torch.Tensor
    windows: tuple[_DsaWindowMetadata, ...]

    @classmethod
    def from_plan(cls, plan: DsaSelectionPlan, device: torch.device | str) -> DsaSelectionWorkspace:
        if not isinstance(plan, DsaSelectionPlan):
            raise TypeError("plan must be a DsaSelectionPlan")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("cuDNN DSA selection workspace must be allocated on CUDA")
        metadata = []
        for window in plan.windows:
            q_offsets = None
            if any(window.q_causal_offsets):
                q_offsets = torch.tensor(window.q_causal_offsets, dtype=torch.int32, device=resolved)
            metadata.append(
                _DsaWindowMetadata(
                    cu_seqlens_q=torch.tensor(window.cu_seqlens_q, dtype=torch.int32, device=resolved),
                    cu_seqlens_k=torch.tensor(window.cu_seqlens_k, dtype=torch.int32, device=resolved),
                    q_causal_offsets=q_offsets,
                    candidate_offsets=torch.tensor(window.candidate_offsets, dtype=torch.int64, device=resolved),
                )
            )
        return cls(
            plan=plan,
            candidates=torch.empty(plan.candidate_workspace_floats, dtype=torch.float32, device=resolved),
            block_ids=torch.empty((plan.total_queries, plan.top_k), dtype=torch.int32, device=resolved),
            scores=torch.empty((plan.total_queries, plan.top_k), dtype=torch.float32, device=resolved),
            windows=tuple(metadata),
        )


@dataclass(frozen=True, slots=True)
class DsaSelectionResult:
    """Workspace-owned local block ids and their FP32 selected logits.

    The `(block_id, score)` pairs form a deterministic set when selection uses
    deterministic tie-breaking. Their radix output slot order is unspecified.
    """

    block_ids: torch.Tensor
    scores: torch.Tensor


@cache
def _load_cudnn_dsa_select():
    try:
        cudnn = import_module("cudnn")
        namespace = getattr(cudnn, "DSA", None)
        if namespace is not None and hasattr(namespace, "indexer_forward_top_k_wrapper"):
            return namespace.indexer_forward_top_k_wrapper
        module = import_module("cudnn.deepseek_sparse_attention")
        return module.indexer_forward_top_k_wrapper
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "cuDNN DSA selection requires cuDNN Frontend with indexer_forward_top_k_wrapper; "
            "install cuda-linear-attention[cudnn-hybrid]"
        ) from error


def _validate_selection_inputs(
    qi: torch.Tensor,
    kbar: torch.Tensor,
    weights: torch.Tensor,
    *,
    plan: DsaSelectionPlan,
    workspace: DsaSelectionWorkspace,
) -> None:
    if workspace.plan != plan:
        raise ValueError("workspace was not created from the supplied DsaSelectionPlan")
    if qi.device.type != "cuda" or kbar.device != qi.device or weights.device != qi.device:
        raise ValueError("qi, kbar and weights must be CUDA tensors on one device")
    if qi.ndim != 3 or qi.shape[0] != plan.total_queries:
        raise ValueError(f"qi must have shape [{plan.total_queries}, H, D]")
    if kbar.shape != (plan.total_blocks, 1, qi.shape[2]):
        raise ValueError(
            f"kbar must have shape [{plan.total_blocks}, 1, {qi.shape[2]}], got {tuple(kbar.shape)}"
        )
    if weights.shape != qi.shape[:2]:
        raise ValueError(f"weights must have shape {tuple(qi.shape[:2])}, got {tuple(weights.shape)}")
    if qi.dtype != torch.bfloat16 or kbar.dtype != torch.bfloat16:
        raise TypeError("qi and kbar must be bfloat16; cuLA does not silently quantize indexer inputs")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("weights must be bfloat16 or float32")
    for name, tensor in (("qi", qi), ("kbar", kbar), ("weights", weights)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; implicit hot-path layout copies are not allowed")
    for name, tensor in (
        ("candidates", workspace.candidates),
        ("block_ids", workspace.block_ids),
        ("scores", workspace.scores),
    ):
        if tensor.device != qi.device:
            raise ValueError(f"workspace {name} must be on {qi.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"workspace {name} must be contiguous")
    if workspace.candidates.dtype != torch.float32 or workspace.candidates.numel() < plan.candidate_workspace_floats:
        raise ValueError("workspace candidates must be a sufficiently large FP32 tensor")
    if workspace.block_ids.dtype != torch.int32 or workspace.block_ids.shape != (plan.total_queries, plan.top_k):
        raise ValueError("workspace block_ids has the wrong dtype or shape")
    if workspace.scores.dtype != torch.float32 or workspace.scores.shape != (plan.total_queries, plan.top_k):
        raise ValueError("workspace scores has the wrong dtype or shape")
    if len(workspace.windows) != len(plan.windows):
        raise ValueError("workspace window metadata does not match the plan")
    for metadata, window in zip(workspace.windows, plan.windows):
        entries = len(window.cu_seqlens_q)
        for name, tensor, dtype in (
            ("cu_seqlens_q", metadata.cu_seqlens_q, torch.int32),
            ("cu_seqlens_k", metadata.cu_seqlens_k, torch.int32),
            ("candidate_offsets", metadata.candidate_offsets, torch.int64),
        ):
            if tensor.device != qi.device or tensor.dtype != dtype or tensor.shape != (entries,) or not tensor.is_contiguous():
                raise ValueError(f"workspace {name} metadata does not match its selection window")
        if metadata.q_causal_offsets is not None and (
            metadata.q_causal_offsets.device != qi.device
            or metadata.q_causal_offsets.dtype != torch.int32
            or metadata.q_causal_offsets.shape != (entries - 1,)
            or not metadata.q_causal_offsets.is_contiguous()
        ):
            raise ValueError("workspace q_causal_offsets metadata does not match its selection window")


def cudnn_dsa_select(
    qi: torch.Tensor,
    kbar: torch.Tensor,
    weights: torch.Tensor,
    *,
    plan: DsaSelectionPlan,
    workspace: DsaSelectionWorkspace,
    sm_scale: float = 1.0,
    deterministic: bool = True,
) -> DsaSelectionResult:
    """Select local compressed-block ids with bounded scratch and FP32 logits.

    The function consumes packed pooling output directly; it never pads blocks
    to ``[B, max_blocks, D]``.  Every long-sequence window reuses the same
    caller-owned compact-logit buffer.  Returned tensors alias ``workspace``
    and are overwritten by its next invocation.  ``deterministic=True`` makes
    the selected set reproducible; cuDNN's radix output slot order remains
    unspecified, so downstream code that requires bitwise accumulation order
    must canonicalize the pairs explicitly.
    """

    if not isinstance(plan, DsaSelectionPlan):
        raise TypeError("plan must be a DsaSelectionPlan")
    if not isinstance(workspace, DsaSelectionWorkspace):
        raise TypeError("workspace must be a DsaSelectionWorkspace")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
    if isinstance(sm_scale, bool) or not isinstance(sm_scale, int | float) or not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be a finite host scalar")
    _validate_selection_inputs(qi, kbar, weights, plan=plan, workspace=workspace)

    select = _load_cudnn_dsa_select()
    heads = int(qi.shape[1])
    for window, metadata in zip(plan.windows, workspace.windows):
        query_slice = slice(window.query_start, window.query_end)
        key_slice = slice(window.key_start, window.key_end)
        select(
            qi[query_slice],
            kbar[key_slice],
            weights[query_slice],
            top_k=plan.top_k,
            ratio=plan.block_size,
            qhead_per_kv_head=heads,
            sm_scale=float(sm_scale),
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=window.max_seqlen_q,
            max_seqlen_k=window.max_seqlen_k,
            q_causal_offsets=metadata.q_causal_offsets,
            topk_indices_global=False,
            cand_buffer=workspace.candidates,
            out_indices=workspace.block_ids[query_slice],
            out_logits=workspace.scores[query_slice],
            cand_batch_offsets=metadata.candidate_offsets,
            return_softmax=False,
            deterministic=deterministic,
        )
    return DsaSelectionResult(block_ids=workspace.block_ids, scores=workspace.scores)


__all__ = [
    "DEFAULT_CANDIDATE_BUDGET_BYTES",
    "MAX_BALANCED_CANDIDATE_BUDGET_BYTES",
    "MEMORY_EFFICIENT_CANDIDATE_BUDGET_BYTES",
    "DsaSelectionPlan",
    "DsaSelectionResult",
    "DsaSelectionWindow",
    "DsaSelectionWorkspace",
    "cudnn_dsa_select",
]
