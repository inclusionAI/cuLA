# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent FP32 references for cuLA Lightning Attention.

The public recurrent-state layout is BHVK: ``[batch, value_head, V, K]``.
The internal mathematical state used here is BHKV.  Keeping the conversion at
the reference boundary makes orientation errors observable even though the
production specialization has ``K == V == 128``.

These functions intentionally do not import a product kernel or FLA.  The
tokenwise and chunkwise implementations have different decompositions so that
agreement is useful evidence rather than a self-comparison of one algorithm.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must all be rank-4 tensors")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have identical shapes, got q={tuple(q.shape)}, k={tuple(k.shape)}")

    batch, length, qk_heads, key_dim = q.shape
    if v.shape[:2] != (batch, length):
        raise ValueError(f"v must share q batch/length, got q={tuple(q.shape)}, v={tuple(v.shape)}")
    value_heads, value_dim = v.shape[2:]
    if value_heads < qk_heads or value_heads % qk_heads != 0:
        raise ValueError(f"value heads ({value_heads}) must be >= and divisible by q/k heads ({qk_heads})")
    return batch, length, qk_heads, value_heads, key_dim, value_dim


def _expand_qk_to_value_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qk_heads = q.shape[2]
    if value_heads == qk_heads:
        return q, k
    group_size = value_heads // qk_heads
    return q.repeat_interleave(group_size, dim=2), k.repeat_interleave(group_size, dim=2)


def _normalize_decay(decay_s: torch.Tensor, qk_heads: int, value_heads: int) -> torch.Tensor:
    if decay_s.ndim != 1:
        raise ValueError(f"decay_s must be rank 1, got shape {tuple(decay_s.shape)}")
    if decay_s.dtype != torch.float32:
        raise ValueError(f"decay_s must be FP32, got {decay_s.dtype}")
    if decay_s.shape[0] == value_heads:
        return decay_s
    if decay_s.shape[0] == qk_heads:
        return decay_s.repeat_interleave(value_heads // qk_heads)
    raise ValueError(f"decay_s must have shape ({qk_heads},) or ({value_heads},), got {tuple(decay_s.shape)}")


def _initial_state_bhkv(
    initial_state_bhvk: torch.Tensor | None,
    *,
    batch: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if initial_state_bhvk is None:
        return torch.zeros(batch, value_heads, key_dim, value_dim, dtype=torch.float32, device=device)
    if initial_state_bhvk.dtype != torch.float32:
        raise ValueError(f"initial_state must be FP32, got {initial_state_bhvk.dtype}")
    expected = (batch, value_heads, value_dim, key_dim)
    if initial_state_bhvk.shape != expected:
        raise ValueError(f"initial_state must use public BHVK shape {expected}, got {tuple(initial_state_bhvk.shape)}")
    return initial_state_bhvk.transpose(-1, -2).contiguous().float().clone()


def _public_state(state_bhkv: torch.Tensor | None) -> torch.Tensor | None:
    if state_bhkv is None:
        return None
    return state_bhkv.transpose(-1, -2).contiguous()


def tokenwise_lightning_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_s: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the normative recurrence one token at a time in FP32.

    ``initial_state`` and the returned state use public BHVK layout.  Output is
    FP32 even when the inputs are BF16 so callers can characterize projection
    error separately from recurrence error.
    """

    batch, length, qk_heads, value_heads, key_dim, value_dim = _validate_qkv(q, k, v)
    q, k = _expand_qk_to_value_heads(q, k, value_heads)
    decay_s = _normalize_decay(decay_s, qk_heads, value_heads).float()
    q, k, v = q.float(), k.float(), v.float()

    state = _initial_state_bhkv(
        initial_state,
        batch=batch,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        device=q.device,
    )
    decay = torch.exp(-decay_s).view(1, value_heads, 1, 1)
    output = torch.empty(batch, length, value_heads, value_dim, dtype=torch.float32, device=q.device)

    for token in range(length):
        key = k[:, token]
        value = v[:, token]
        state = state * decay + key.unsqueeze(-1) * value.unsqueeze(-2)
        output[:, token] = torch.einsum("bhk,bhkv->bhv", q[:, token], state) * scale

    return output, _public_state(state if output_final_state else None)


def chunkwise_lightning_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_s: torch.Tensor,
    *,
    scale: float = 1.0,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the recurrence with an independently derived chunk formula.

    The final partial chunk advances state by its actual valid length ``L``;
    padding to ``chunk_size`` never adds recurrent decay steps.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    batch, length, qk_heads, value_heads, key_dim, value_dim = _validate_qkv(q, k, v)
    q, k = _expand_qk_to_value_heads(q, k, value_heads)
    decay_s = _normalize_decay(decay_s, qk_heads, value_heads).float()
    q, k, v = q.float(), k.float(), v.float()

    state = _initial_state_bhkv(
        initial_state,
        batch=batch,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        device=q.device,
    )
    output = torch.empty(batch, length, value_heads, value_dim, dtype=torch.float32, device=q.device)

    for chunk_start in range(0, length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, length)
        actual_length = chunk_end - chunk_start
        q_chunk = q[:, chunk_start:chunk_end]
        k_chunk = k[:, chunk_start:chunk_end]
        v_chunk = v[:, chunk_start:chunk_end]

        query_position = torch.arange(actual_length, device=q.device).view(actual_length, 1)
        key_position = torch.arange(actual_length, device=q.device).view(1, actual_length)
        distance = query_position - key_position
        causal = distance >= 0
        decay_mask = torch.exp(
            -decay_s.view(1, value_heads, 1, 1) * distance.clamp_min(0).float().view(1, 1, actual_length, actual_length)
        )
        decay_mask = decay_mask * causal.view(1, 1, actual_length, actual_length)

        scores = torch.einsum("bthk,bshk->bhts", q_chunk, k_chunk)
        intra = torch.einsum("bhts,bshv->bthv", scores * decay_mask, v_chunk)

        position = torch.arange(actual_length, dtype=torch.float32, device=q.device)
        input_state_decay = torch.exp(-decay_s.view(1, 1, value_heads, 1) * (position.view(1, actual_length, 1, 1) + 1.0))
        inter = torch.einsum("bthk,bhkv->bthv", q_chunk, state) * input_state_decay
        output[:, chunk_start:chunk_end] = (intra + inter) * scale

        block_decay = torch.exp(-decay_s * actual_length).view(1, value_heads, 1, 1)
        key_weight = torch.exp(
            -decay_s.view(1, 1, value_heads, 1) * (actual_length - 1 - position).view(1, actual_length, 1, 1)
        )
        state = state * block_decay + torch.einsum("bthk,bthv->bhkv", k_chunk * key_weight, v_chunk)

    return output, _public_state(state if output_final_state else None)


def packed_tokenwise_lightning_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay_s: torch.Tensor,
    cu_seqlens: torch.Tensor | Sequence[int],
    *,
    scale: float = 1.0,
    state_pool: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference packed-varlen execution and state-pool updates.

    This is an oracle path and may inspect device values.  Product wrappers
    must not copy that behavior into their default metadata-only validation.
    """

    batch, total_length, _qk_heads, value_heads, key_dim, value_dim = _validate_qkv(q, k, v)
    if batch != 1:
        raise ValueError(f"packed input must have physical batch 1, got {batch}")

    boundaries = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else list(cu_seqlens)
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != total_length:
        raise ValueError(f"cu_seqlens must start at 0 and end at {total_length}, got {boundaries}")
    lengths = [end - begin for begin, end in zip(boundaries, boundaries[1:])]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"packed sequences must have positive lengths, got {lengths}")
    sequence_count = len(lengths)

    if initial_state_indices is None:
        indices = list(range(sequence_count))
    elif isinstance(initial_state_indices, torch.Tensor):
        indices = initial_state_indices.tolist()
    else:
        indices = list(initial_state_indices)
    if len(indices) != sequence_count:
        raise ValueError(f"expected {sequence_count} state indices, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError("packed execution requires unique state-pool indices")

    if state_pool is None:
        updated_pool = torch.zeros(
            sequence_count,
            value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        expected_tail = (value_heads, value_dim, key_dim)
        if state_pool.ndim != 4 or state_pool.shape[1:] != expected_tail:
            raise ValueError(f"state_pool must have shape [pool, {value_heads}, {value_dim}, {key_dim}]")
        if state_pool.dtype != torch.float32:
            raise ValueError(f"state_pool must be FP32, got {state_pool.dtype}")
        updated_pool = state_pool.clone()
    if any(index < 0 or index >= updated_pool.shape[0] for index in indices):
        raise ValueError(f"state index is outside pool size {updated_pool.shape[0]}: {indices}")

    output = torch.empty(1, total_length, value_heads, value_dim, dtype=torch.float32, device=q.device)
    for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        slot = indices[sequence]
        sequence_output, sequence_final_state = tokenwise_lightning_reference(
            q[:, begin:end],
            k[:, begin:end],
            v[:, begin:end],
            decay_s,
            scale=scale,
            initial_state=updated_pool[slot : slot + 1],
            output_final_state=True,
        )
        output[:, begin:end] = sequence_output
        assert sequence_final_state is not None
        updated_pool[slot] = sequence_final_state[0]

    return output, updated_pool
