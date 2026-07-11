# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Intra-Card Context Parallel (CP) for Chunk Delta H.

Overview:
    Long sequences on a single card are split into sub-sequences, each processed
    independently via cuLA's CuTeDSL chunk_delta_h kernel. A prefix-scan merge
    step propagates initial states across sub-sequences, eliminating the sequential
    bottleneck of the original single-pass recurrence.

Pipeline (3 stages):
    1. Pre-Scan: For each sub-sequence, compute packed (he, m) state:
         he [K, V] = cumulative delta-rule update (the "h-exit" state)
         m  [K, K] = cumulative decay matrix
       Packed as hm [S_split, H, K, K+V] where columns [0:V]=he, [V:V+K]=m

    2. Merge: Prefix scan across sub-sequences of the same original sequence.
       For sub-sequence j:  h0_j = m_j @ h0_{j-1} + he_j
       Produces per-sub-sequence initial states.

    3. Forward H: Run cuLA's existing chunk_gated_delta_rule_fwd_h on the
       split sub-sequences with the merged initial states.

Reference:
    - FLA intra-card CP: fla/ops/common/intracard_cp.py
    - FLA CP kernels:    fla/ops/cp/chunk_delta_h.py
    - cuLA chunk_delta_h: cula/ops/kda/sm100/delta_h.py
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import NamedTuple

import torch

from cula.utils import get_device_sm_count, get_pre_scan

# Lazy import to avoid circular dependency with cula.ops.kda.sm100.delta_h
_chunk_gated_delta_rule_fwd_h = None


def _get_fwd_h():
    global _chunk_gated_delta_rule_fwd_h
    if _chunk_gated_delta_rule_fwd_h is None:
        from cula.ops.kda.sm100.delta_h import chunk_gated_delta_rule_fwd_h

        _chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
    return _chunk_gated_delta_rule_fwd_h


class SplitSeqInfo(NamedTuple):
    """Metadata for sequences split into sub-sequences."""

    split_seq_ids: list[int]  # original sequence indices that were split
    start_subseq_idx: list[int]  # first sub-seq index in expanded cu_seqlens per split seq
    num_subseqs: list[int]  # number of sub-sequences per split seq


class _CacheEntry(NamedTuple):
    """Cached precomputed indices and GPU tensors for a given cu_seqlens layout."""

    cu_seqlens_ref: weakref.ref
    cu_seqlens_subseq_values: list[int]
    split_info: SplitSeqInfo
    total_subseqs: int
    non_first_indices: torch.Tensor  # [num_non_first] int64 GPU
    first_subseq_indices: torch.Tensor  # [N_orig] int64 GPU
    last_subseq_indices: torch.Tensor  # [N_orig] int64 GPU
    num_non_first: int
    merge_seq_starts: list[int]
    merge_seq_counts: list[int]
    merge_init_offsets: list[int]
    cu_seqlens_subseq_gpu: torch.Tensor
    chunk_indices_subseq: torch.Tensor  # [NT_subseq, 2] int32


_intracard_cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
_intracard_cache_lock = threading.Lock()
_INTRACARD_CACHE_MAXSIZE = 8


def _prepare_chunk_indices(
    cu_seqlens_values: list[int],
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build chunk_indices [NT, 2] int32 from cu_seqlens CPU list."""
    num_seqs = len(cu_seqlens_values) - 1
    seq_ids: list[int] = []
    chunk_ids: list[int] = []
    for i in range(num_seqs):
        nc = (cu_seqlens_values[i + 1] - cu_seqlens_values[i] + chunk_size - 1) // chunk_size
        seq_ids.extend([i] * nc)
        chunk_ids.extend(range(nc))
    return torch.stack(
        [torch.tensor(seq_ids, dtype=torch.int32, device=device), torch.tensor(chunk_ids, dtype=torch.int32, device=device)],
        dim=1,
    )


# Tunable thresholds — empirically calibrated on B200 SM100 (SM=152).
NUM_V_BLOCKS = 2  # fwd_h grid V-tile factor: grid = (NUM_V_BLOCKS, N*H)
MIN_SUBSEQ_CHUNKS = 16  # min chunks per sub-sequence
MIN_LONG_SEQ_CHUNKS = 256  # min chunks of the longest seq to consider CP
MAX_BE_H = 10  # max Be*H; above this CP gain < overhead (~3%)
MIN_SUBSEQ_CHUNKS_PER_HEAD = 12  # min expected sub-seq chunks per head (H-scaled Guard 3)


def should_use_intracard_cp(
    cu_seqlens_cpu: torch.Tensor,
    num_sms: int,
    H: int,
    chunk_size: int = 64,
) -> bool:
    """Pure-Python predicate: should we dispatch to intracard CP?

    Four cheap CPU-only guards (a fifth post-split guard lives in intracard_fwd_h):
      Guard 0: baseline already saturates SMs.
      Guard 1: longest sequence too short to amortize CP overhead.
      Guard 2: Be*H > MAX_BE_H — other seqs already provide enough parallelism.
      Guard 3: expected sub-seq too short — CP merge overhead exceeds gain.
    """
    cu_list = cu_seqlens_cpu.tolist()
    num_seqs = len(cu_list) - 1
    if num_seqs == 0:
        return False

    if NUM_V_BLOCKS * H * num_seqs >= num_sms:  # Guard 0
        return False

    chunks = [(cu_list[i + 1] - cu_list[i] + chunk_size - 1) // chunk_size for i in range(num_seqs)]
    max_c = max(chunks)

    if max_c < MIN_LONG_SEQ_CHUNKS:  # Guard 1
        return False

    # Guard 2: Be = effective batch size (as if every seq were max_c chunks long)
    Be = sum(chunks) / max_c
    if Be * H > MAX_BE_H:
        return False

    # Guard 3: expected sub-seq length must be long enough to amortise CP overhead.
    # CP merge work scales with H, so the minimum is proportional to H.
    per_seq_units = NUM_V_BLOCKS * H
    sm_budget = max(num_sms - per_seq_units * max(num_seqs - 1, 0), per_seq_units)
    target_splits = max(2, sm_budget // per_seq_units)
    expected_subseq_c = max((max_c + target_splits - 1) // target_splits, MIN_SUBSEQ_CHUNKS)
    return expected_subseq_c >= MIN_SUBSEQ_CHUNKS_PER_HEAD * H


def compute_subseq_len(
    seq_len: int,
    num_sms: int,
    num_heads: int,
    chunk_size: int = 64,
    num_seqs: int = 1,
) -> int:
    """Compute target sub-sequence length for intracard splitting.

    Targets enough splits to saturate remaining SMs after other sequences
    in the batch occupy their share. Uses floor division so that
    actual sub-seqs never exceed target_splits, guaranteeing Guard 3
    (total_subseqs * NUM_V_BLOCKS * H <= num_sms) is always satisfied
    for a single split sequence in the batch.
    Result is floored at MIN_SUBSEQ_CHUNKS * chunk_size.
    """
    seq_chunks = (seq_len + chunk_size - 1) // chunk_size

    if seq_chunks < 8:
        return seq_len

    per_seq_units = NUM_V_BLOCKS * num_heads
    sm_budget = max(num_sms - per_seq_units * max(num_seqs - 1, 0), per_seq_units)
    target_splits = max(2, sm_budget // per_seq_units)

    subseq_chunks = (seq_chunks + target_splits - 1) // target_splits
    subseq_chunks = max(subseq_chunks, MIN_SUBSEQ_CHUNKS)

    return subseq_chunks * chunk_size


def prepare_subseq_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    subseq_len: int,
    chunk_size: int = 64,
    max_splits: int = 32,
) -> tuple[list[int], SplitSeqInfo | bool, int]:
    """Insert sub-sequence split points into cu_seqlens.

    Sequences >= 3 * subseq_len are split into evenly-sized sub-sequences
    (each a multiple of chunk_size); shorter sequences are kept intact.
    Returns (expanded boundaries, SplitSeqInfo or False, total_subseqs).
    """
    cu_list = cu_seqlens_cpu.tolist()
    N = len(cu_list) - 1
    if N == 0:
        return cu_list, False, 0

    subseq_chunks = (subseq_len + chunk_size - 1) // chunk_size
    threshold_subseq_len = 3 * subseq_len

    split_seq_ids: list[int] = []
    start_subseq_idxs: list[int] = []
    num_subseqs_list: list[int] = []

    boundaries: list[int] = [0]
    cumsum_offset = 0

    for i in range(N):
        seq_start = cu_list[i]
        seq_end = cu_list[i + 1]
        seq_len_i = seq_end - seq_start
        seq_chunks_i = (seq_len_i + chunk_size - 1) // chunk_size

        if seq_len_i >= threshold_subseq_len:
            num_ss = min(max_splits, (seq_chunks_i + subseq_chunks - 1) // subseq_chunks)
            chunks_per = (seq_chunks_i + num_ss - 1) // num_ss
            actual_ssl = chunks_per * chunk_size
            split_seq_ids.append(i)
            start_subseq_idxs.append(cumsum_offset)
            num_subseqs_list.append(num_ss)
            for j in range(num_ss):
                boundary = min(seq_start + (j + 1) * actual_ssl, seq_end)
                boundaries.append(boundary)
            cumsum_offset += num_ss
        else:
            boundaries.append(seq_end)
            cumsum_offset += 1

    if not split_seq_ids:
        return cu_list, False, 0

    total_subseqs = cumsum_offset
    split_info = SplitSeqInfo(
        split_seq_ids=split_seq_ids,
        start_subseq_idx=start_subseq_idxs,
        num_subseqs=num_subseqs_list,
    )
    return boundaries, split_info, total_subseqs


class _PrecomputedIndices(NamedTuple):
    """Derived scatter/gather indices for the CP orchestrator."""

    non_first_indices: list[int]  # where to scatter merge results
    first_subseq_indices: list[int]  # first sub-seq index per original seq
    last_subseq_indices: list[int]  # last sub-seq index per original seq
    num_non_first: int
    merge_seq_starts: list[int]
    merge_seq_counts: list[int]
    merge_init_offsets: list[int]


def _precompute_intracard_indices(
    split_info: SplitSeqInfo,
    N_orig: int,
) -> _PrecomputedIndices:
    """Precompute scatter/gather indices from split metadata."""
    starts = split_info.start_subseq_idx
    num_ss = split_info.num_subseqs
    split_ids = split_info.split_seq_ids

    num_subseqs_per_seq = [1] * N_orig
    for sid, nss in zip(split_ids, num_ss):
        num_subseqs_per_seq[sid] = nss

    non_first_indices: list[int] = []
    for s, n in zip(starts, num_ss):
        for j in range(1, n):
            non_first_indices.append(s + j)

    first_subseq_indices: list[int] = [0]
    running = 0
    for i in range(N_orig - 1):
        running += num_subseqs_per_seq[i]
        first_subseq_indices.append(running)

    last_subseq_indices: list[int] = []
    running = 0
    for n in num_subseqs_per_seq:
        running += n
        last_subseq_indices.append(running - 1)

    # merge_seq_starts/counts use per-seq start indices (not CSR offsets) because
    # split sub-seqs may be non-contiguous in hm when unsplit seqs exist in between.
    merge_seq_starts: list[int] = list(starts)
    merge_seq_counts: list[int] = list(num_ss)
    merge_init_offsets: list[int] = [0]
    for n in num_ss:
        merge_init_offsets.append(merge_init_offsets[-1] + n - 1)
    num_non_first = merge_init_offsets[-1]

    return _PrecomputedIndices(
        non_first_indices=non_first_indices,
        first_subseq_indices=first_subseq_indices,
        last_subseq_indices=last_subseq_indices,
        num_non_first=num_non_first,
        merge_seq_starts=merge_seq_starts,
        merge_seq_counts=merge_seq_counts,
        merge_init_offsets=merge_init_offsets,
    )


def intracard_pre_scan(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor | None,
    cu_seqlens_subseq_split: torch.Tensor,
    S_split: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute packed (he, m) exit state for each sub-sequence.

    Returns hm [S_split, H, K, V+K] fp32 where columns [0:V]=he, [V:V+K]=m.
    Dispatches to the SM-appropriate pre_scan implementation via get_pre_scan().
    """
    chunk_delta_rule_pre_scan = get_pre_scan(k.device)
    return chunk_delta_rule_pre_scan(
        k=k,
        w=w,
        u=u,
        gk=gk,
        cu_seqlens_split=cu_seqlens_subseq_split,
        S_split=S_split,
        chunk_size=chunk_size,
    )


def intracard_merge(
    hm: torch.Tensor,
    split_info: SplitSeqInfo,
    num_non_first: int,
    merge_seq_starts: list[int],
    merge_seq_counts: list[int],
    merge_init_offsets: list[int],
    device: torch.device,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, int]:
    """Prefix scan across sub-sequences to produce per-sub-sequence initial states.

    For split seq [s0, s1, ..., s_{n-1}]: h0_sj = m_{j-1} @ h0_{j-1} + he_{j-1}.
    Returns (initial_states_merge [num_non_first, H, K, V] fp32, num_non_first).
    """
    from cula.ops.kda.sm100.cp.merge import launch_merge

    if num_non_first == 0:
        return None, 0

    initial_states_merge = launch_merge(
        hm=hm,
        seq_starts=merge_seq_starts,
        seq_counts=merge_seq_counts,
        init_offsets=merge_init_offsets,
        split_seq_ids=split_info.split_seq_ids,
        h0=initial_state,
        num_non_first=num_non_first,
    )

    return initial_states_merge, num_non_first


def _scatter_initial_states(
    initial_state: torch.Tensor | None,
    initial_states_merge: torch.Tensor | None,
    num_non_first: int,
    total_subseqs: int,
    first_subseq_indices: torch.Tensor,
    non_first_indices: torch.Tensor,
    H: int,
    K: int,
    V: int,
    device: torch.device,
) -> torch.Tensor:
    """Build initial_state_expanded [total_subseqs, H, K, V] for all sub-sequences."""
    initial_state_expanded = torch.zeros(total_subseqs, H, K, V, device=device, dtype=torch.float32)

    if initial_state is not None:
        initial_state_expanded[first_subseq_indices] = initial_state

    if initial_states_merge is not None and num_non_first > 0:
        initial_state_expanded[non_first_indices] = initial_states_merge

    return initial_state_expanded


def _gather_final_states(
    final_state_subseq: torch.Tensor | None,
    last_subseq_indices: torch.Tensor,
    output_final_state: bool,
) -> torch.Tensor | None:
    """Gather final state from last sub-sequence of each original sequence."""
    if not output_final_state or final_state_subseq is None:
        return None
    return final_state_subseq[last_subseq_indices]


def intracard_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    max_splits: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Intra-card CP chunk_delta_h forward; splits long sequences and runs
    pre_scan -> merge -> fwd_h on the sub-sequences.

    Pure CP executor: raises NotSplittableError when the shape cannot be
    meaningfully split. The caller owns the fallback-vs-raise policy (the
    pre-split heuristic lives in sm100_intracard_cp_decision, not here).
    """
    assert cu_seqlens is not None, "intracard_fwd_h requires cu_seqlens (varlen mode)"

    _, _, H, K = k.shape
    V = u.shape[3]
    device = k.device
    num_sms = get_device_sm_count(device)

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()

    cu_list = cu_seqlens_cpu.tolist()
    num_seqs = len(cu_list) - 1
    max_seq_len = max(cu_list[i + 1] - cu_list[i] for i in range(num_seqs))
    subseq_len = compute_subseq_len(max_seq_len, num_sms, H, chunk_size, num_seqs=num_seqs)

    cached = None
    cache_key = (id(cu_seqlens), subseq_len, chunk_size, max_splits, str(device))
    with _intracard_cache_lock:
        cached = _intracard_cache.get(cache_key)
        if cached is not None:
            if cached.cu_seqlens_ref() is cu_seqlens:
                _intracard_cache.move_to_end(cache_key)
            else:
                _intracard_cache.pop(cache_key, None)
                cached = None

    if cached is None:
        cu_seqlens_subseq_values, split_info, total_subseqs = prepare_subseq_cu_seqlens(
            cu_seqlens_cpu, subseq_len, chunk_size, max_splits=max_splits
        )
    else:
        split_info = cached.split_info
        total_subseqs = cached.total_subseqs

    # Post-split occupancy guard (total_subseqs only known after prepare_subseq_cu_seqlens)
    if split_info and total_subseqs * NUM_V_BLOCKS * H > num_sms:
        split_info = False

    if not split_info:
        from cula.ops.kda.policy import NotSplittableError

        raise NotSplittableError("SM100 intracard CP is not meaningfully splittable for this shape.")

    N_orig = len(cu_seqlens_cpu) - 1

    if cached is not None:
        cu_seqlens_subseq_values = cached.cu_seqlens_subseq_values
        total_subseqs = cached.total_subseqs
        non_first_indices = cached.non_first_indices
        first_subseq_indices = cached.first_subseq_indices
        last_subseq_indices = cached.last_subseq_indices
        num_non_first = cached.num_non_first
        merge_seq_starts = cached.merge_seq_starts
        merge_seq_counts = cached.merge_seq_counts
        merge_init_offsets = cached.merge_init_offsets
        cu_seqlens_subseq_gpu = cached.cu_seqlens_subseq_gpu
        chunk_indices_subseq = cached.chunk_indices_subseq
    else:
        (
            non_first_indices,
            first_subseq_indices,
            last_subseq_indices,
            num_non_first,
            merge_seq_starts,
            merge_seq_counts,
            merge_init_offsets,
        ) = _precompute_intracard_indices(split_info, N_orig)

        non_first_indices = torch.tensor(non_first_indices, dtype=torch.int64, device=device)
        first_subseq_indices = torch.tensor(first_subseq_indices, dtype=torch.int64, device=device)
        last_subseq_indices = torch.tensor(last_subseq_indices, dtype=torch.int64, device=device)

        cu_seqlens_subseq_gpu = torch.tensor(cu_seqlens_subseq_values, dtype=torch.int32, device=device)
        chunk_indices_subseq = _prepare_chunk_indices(cu_seqlens_subseq_values, chunk_size, device)

        with _intracard_cache_lock:
            _intracard_cache[cache_key] = _CacheEntry(
                cu_seqlens_ref=weakref.ref(cu_seqlens),
                cu_seqlens_subseq_values=cu_seqlens_subseq_values,
                split_info=split_info,
                total_subseqs=total_subseqs,
                non_first_indices=non_first_indices,
                first_subseq_indices=first_subseq_indices,
                last_subseq_indices=last_subseq_indices,
                num_non_first=num_non_first,
                merge_seq_starts=merge_seq_starts,
                merge_seq_counts=merge_seq_counts,
                merge_init_offsets=merge_init_offsets,
                cu_seqlens_subseq_gpu=cu_seqlens_subseq_gpu,
                chunk_indices_subseq=chunk_indices_subseq,
            )
            while len(_intracard_cache) > _INTRACARD_CACHE_MAXSIZE:
                _intracard_cache.popitem(last=False)

    hm = intracard_pre_scan(
        k=k,
        w=w,
        u=u,
        gk=gk,
        cu_seqlens_subseq_split=cu_seqlens_subseq_gpu,
        S_split=total_subseqs,
        chunk_size=chunk_size,
    )

    initial_states_merge, num_non_first = intracard_merge(
        hm=hm,
        split_info=split_info,
        num_non_first=num_non_first,
        merge_seq_starts=merge_seq_starts,
        merge_seq_counts=merge_seq_counts,
        merge_init_offsets=merge_init_offsets,
        device=device,
        initial_state=initial_state,
    )

    initial_state_expanded = _scatter_initial_states(
        initial_state=initial_state,
        initial_states_merge=initial_states_merge,
        num_non_first=num_non_first,
        total_subseqs=total_subseqs,
        first_subseq_indices=first_subseq_indices,
        non_first_indices=non_first_indices,
        H=H,
        K=K,
        V=V,
        device=device,
    )

    h, v_new, final_state_subseq = _get_fwd_h()(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=initial_state_expanded,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens_subseq_gpu,
        chunk_indices=chunk_indices_subseq,
        _no_cp=True,
    )

    final_state = _gather_final_states(
        final_state_subseq=final_state_subseq,
        last_subseq_indices=last_subseq_indices,
        output_final_state=output_final_state,
    )

    return h, v_new, final_state
