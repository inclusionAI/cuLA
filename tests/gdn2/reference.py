# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Independent PyTorch reference for packed Gated DeltaNet-2 prefill."""

from __future__ import annotations

from itertools import pairwise

import torch


def _expand_qk_heads(
    tensor: torch.Tensor,
    value_heads: int,
) -> torch.Tensor:
    """Expand query-owned channels to their value-head owners in FP32."""

    group_size = value_heads // tensor.shape[1]
    owner = torch.arange(
        value_heads,
        device=tensor.device,
    ).div(group_size, rounding_mode="floor")
    return tensor.index_select(1, owner).float()


@torch.inference_mode()
def tokenwise_gdn2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the GDN2 recurrence token by token using only PyTorch ops.

    The public recurrent state is accepted and returned in ``[N,Hv,V,K]``
    orientation. Accumulation and the returned reference output use FP32.
    """

    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    value_heads = v.shape[1]
    key_size = q.shape[-1]
    value_size = v.shape[-1]
    qh = _expand_qk_heads(q, value_heads)
    kh = _expand_qk_heads(k, value_heads)
    gh = _expand_qk_heads(g, value_heads)
    bh = _expand_qk_heads(b, value_heads)
    vf = v.float()
    wf = w.float()
    output = torch.empty_like(v, dtype=torch.float32)
    final_states: list[torch.Tensor] = []

    for sequence, (start, end) in enumerate(pairwise(offsets)):
        if initial_state is None:
            state = torch.zeros(
                value_heads,
                key_size,
                value_size,
                device=v.device,
                dtype=torch.float32,
            )
        else:
            state = initial_state[sequence].transpose(-1, -2).contiguous().clone()
        for token in range(start, end):
            decayed = state * gh[token].exp().unsqueeze(-1)
            erase_key = bh[token] * kh[token]
            erase_read = torch.einsum(
                "hk,hkv->hv",
                erase_key,
                decayed,
            )
            new_value = wf[token] * vf[token] - erase_read
            state = decayed + kh[token].unsqueeze(-1) * new_value.unsqueeze(-2)
            output[token] = scale * torch.einsum(
                "hk,hkv->hv",
                qh[token],
                state,
            )
        if output_final_state:
            final_states.append(
                state.transpose(-1, -2).contiguous(),
            )

    return (
        output,
        (torch.stack(final_states) if output_final_state else None),
    )
