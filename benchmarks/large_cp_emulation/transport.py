# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-only transports that emulate logical CP dependency lengths."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist


def emulate_ring_hops(
    source: torch.Tensor,
    receive: torch.Tensor,
    *,
    physical_rank: int,
    physical_world_size: int,
    hops: int,
) -> None:
    """Execute dependent NCCL hops without pretending to exchange CP64 data."""
    if source.shape != receive.shape or source.dtype != receive.dtype or source.device != receive.device:
        raise ValueError("source and receive buffers must have identical shape, dtype, and device")
    if not 0 <= physical_rank < physical_world_size:
        raise ValueError("physical_rank must be in [0, physical_world_size)")
    if hops < 0:
        raise ValueError("hops must be non-negative")
    send, target = source, receive
    next_rank = (physical_rank + 1) % physical_world_size
    previous_rank = (physical_rank - 1 + physical_world_size) % physical_world_size
    for _ in range(hops):
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send, next_rank),
                dist.P2POp(dist.irecv, target, previous_rank),
            ]
        )
        for request in requests:
            request.wait()
        send, target = target, send


@dataclass(slots=True)
class LogicalCPRingTransport:
    """Implement the native KDA transport protocol with emulated ring cost."""

    logical_cp: int
    physical_rank: int
    physical_world_size: int
    ring_hops: int | None = field(default=None, kw_only=True)
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    local_payload: torch.Tensor | None = None
    receive_payload: torch.Tensor | None = None
    h_summaries: torch.Tensor | None = None
    transition_summaries: torch.Tensor | None = None

    def clear_events(self) -> None:
        self.events.clear()

    def _prepare(self, h: torch.Tensor, transition: torch.Tensor) -> None:
        if h.dtype != torch.float32 or transition.dtype != torch.float32:
            raise TypeError("logical KDA boundary summaries must remain FP32")
        if h.device != transition.device:
            raise ValueError("H and transition must be on the same device")
        payload_shape = (h.shape[0], h.shape[-2] + transition.shape[-2], h.shape[-1])
        if self.local_payload is not None:
            if self.local_payload.shape != payload_shape or self.local_payload.device != h.device:
                raise ValueError("logical CP transport cannot be reused with a different H/M shape or device")
            return
        self.local_payload = torch.empty(payload_shape, dtype=torch.float32, device=h.device)
        self.receive_payload = torch.empty_like(self.local_payload)
        self.h_summaries = torch.empty((self.logical_cp - 1, *h.shape), dtype=torch.float32, device=h.device)
        self.transition_summaries = torch.empty((self.logical_cp - 1, *transition.shape), dtype=torch.float32, device=h.device)

    @property
    def workspace_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.local_payload,
                self.receive_payload,
                self.h_summaries,
                self.transition_summaries,
            )
            if tensor is not None
        )

    def _ring(self) -> None:
        assert self.local_payload is not None and self.receive_payload is not None
        hops = self.logical_cp - 1 if self.ring_hops is None else self.ring_hops
        if not 0 <= hops < self.logical_cp:
            raise ValueError("ring_hops must be in [0, logical_cp)")
        emulate_ring_hops(
            self.local_payload,
            self.receive_payload,
            physical_rank=self.physical_rank,
            physical_world_size=self.physical_world_size,
            hops=hops,
        )

    def boundary_summaries(
        self,
        h: torch.Tensor,
        transition: torch.Tensor,
        *,
        plan,
        group,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del group
        self._prepare(h, transition)
        assert self.local_payload is not None
        assert self.h_summaries is not None and self.transition_summaries is not None
        self.local_payload[..., : h.shape[-2], :].copy_(h)
        self.local_payload[..., h.shape[-2] :, :].copy_(transition)
        start = torch.cuda.Event(enable_timing=True)
        after_comm = torch.cuda.Event(enable_timing=True)
        after_prepare = torch.cuda.Event(enable_timing=True)
        start.record()
        self._ring()
        after_comm.record()

        pieces = plan.post_num_ranks if reverse else plan.pre_num_ranks
        h_all = self.h_summaries[:pieces]
        transition_all = self.transition_summaries[:pieces]
        if pieces:
            h_all.copy_(h.unsqueeze(0))
            transition_all.copy_(transition.unsqueeze(0))
        after_prepare.record()
        self.events.append(("backward" if reverse else "forward", start, after_comm, after_prepare))
        return h_all, transition_all
