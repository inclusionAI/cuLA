# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Emulate a large logical Kimi CP group on fewer physical GPUs.

The benchmark executes the complete cuLA KDA path: local summary, boundary
transport and IEEE merge, local forward, and distributed backward. It maps one
representative logical rank to each physical process and runs ``CP - 1`` real
NCCL ring hops per direction. This is a critical-path emulator, not a claim
about real CP64 topology or contention.

Example:
  torchrun --standalone --nproc-per-node=4 benchmarks/bench_kimi_cp_e2e.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass, field, replace

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.distributed import KimiCPBoundaryTransport, KimiCPPlan, cudnn_kda_cp


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _load_merge():
    from cudnn.linear_attention import kimi_cp_merge

    if "precision" not in inspect.signature(kimi_cp_merge).parameters:
        return kimi_cp_merge

    def ieee_merge(*args, **kwargs):
        return kimi_cp_merge(*args, precision="ieee", **kwargs)

    return ieee_merge


@dataclass(slots=True)
class _LogicalCPRingTransport:
    """Benchmark-only logical-CP transport with real NCCL hop count."""

    logical_cp: int
    logical_rank: int
    physical_rank: int
    physical_world_size: int
    ring_hops: int | None = field(default=None, kw_only=True)
    merge: object = field(default_factory=_load_merge)
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    local_payload: torch.Tensor | None = None
    receive_payload: torch.Tensor | None = None
    h_summaries: torch.Tensor | None = None
    transition_summaries: torch.Tensor | None = None

    def clear_events(self) -> None:
        self.events.clear()

    def _prepare(self, h: torch.Tensor, transition: torch.Tensor) -> None:
        payload_shape = (h.shape[0], h.shape[-2] + transition.shape[-2], h.shape[-1])
        if self.local_payload is not None:
            if self.local_payload.shape != payload_shape or self.local_payload.device != h.device:
                raise ValueError("logical CP transport cannot be reused with a different H/M shape or device")
            return
        self.local_payload = torch.empty(payload_shape, dtype=torch.float32, device=h.device)
        self.receive_payload = torch.empty_like(self.local_payload)
        self.h_summaries = torch.empty(
            (self.logical_cp - 1, *h.shape), dtype=torch.float32, device=h.device
        )
        self.transition_summaries = torch.empty(
            (self.logical_cp - 1, *transition.shape), dtype=torch.float32, device=h.device
        )

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
        source, target = self.local_payload, self.receive_payload
        next_rank = (self.physical_rank + 1) % self.physical_world_size
        previous_rank = (self.physical_rank - 1 + self.physical_world_size) % self.physical_world_size
        hops = self.logical_cp - 1 if self.ring_hops is None else self.ring_hops
        if not 0 <= hops < self.logical_cp:
            raise ValueError("ring_hops must be in [0, logical_cp)")
        for _ in range(hops):
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, source, next_rank),
                    dist.P2POp(dist.irecv, target, previous_rank),
                ]
            )
            for request in requests:
                request.wait()
            source, target = target, source

    def boundary_state(
        self,
        h: torch.Tensor,
        transition: torch.Tensor,
        *,
        plan,
        group,
        reverse: bool,
    ) -> torch.Tensor:
        del group
        self._prepare(h, transition)
        assert self.local_payload is not None
        assert self.h_summaries is not None and self.transition_summaries is not None
        self.local_payload[..., : h.shape[-2], :].copy_(h)
        self.local_payload[..., h.shape[-2] :, :].copy_(transition)
        start = torch.cuda.Event(enable_timing=True)
        after_comm = torch.cuda.Event(enable_timing=True)
        after_merge = torch.cuda.Event(enable_timing=True)
        start.record()
        self._ring()
        after_comm.record()

        pieces = plan.post_num_ranks if reverse else plan.pre_num_ranks
        if pieces == 0:
            state = torch.zeros_like(h)
        else:
            h_all = self.h_summaries[:pieces]
            transition_all = self.transition_summaries[:pieces]
            h_all.copy_(h.unsqueeze(0))
            transition_all.copy_(transition.unsqueeze(0))
            state = self.merge(h_all, transition_all, reverse=reverse)["final_state"][0]
        after_merge.record()
        self.events.append(("backward" if reverse else "forward", start, after_comm, after_merge))
        return state


def _representative_logical_rank(physical_rank: int, physical_world_size: int, logical_cp: int) -> int:
    if physical_world_size == 1:
        return logical_cp - 1
    return physical_rank * (logical_cp - 1) // (physical_world_size - 1)


def _run(args) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("logical CP emulation requires at least two physical processes")
    if args.logical_cp < world_size:
        raise ValueError("logical_cp must be at least the physical world size")
    if args.global_tokens % args.logical_cp:
        raise ValueError("global_tokens must be divisible by logical_cp")

    local_tokens = args.global_tokens // args.logical_cp
    logical_rank = _representative_logical_rank(rank, world_size, args.logical_cp)
    torch.manual_seed(args.seed + logical_rank)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=device)

    q = torch.nn.functional.normalize(randn(local_tokens, args.heads, args.key_dim).float(), dim=-1).to(
        torch.bfloat16
    )
    k = torch.nn.functional.normalize(randn(local_tokens, args.heads, args.key_dim).float(), dim=-1).to(
        torch.bfloat16
    )
    v = randn(local_tokens, args.heads, args.value_dim)
    g = -(0.01 + 0.04 * torch.rand(local_tokens, args.heads, args.key_dim, dtype=torch.float32, device=device))
    beta = torch.sigmoid(randn(local_tokens, args.heads, dtype=torch.float32))
    grad_output = (randn(local_tokens, args.heads, args.value_dim) / float(args.global_tokens)).contiguous()

    # The physical plan supplies the local packed-THD shape. The benchmark
    # transport replaces only boundary propagation with logical-CP semantics.
    logical_plan = KimiCPPlan.from_global_cu_seqlens(
        [0, args.global_tokens],
        rank=logical_rank,
        world_size=args.logical_cp,
    )
    plan = replace(
        logical_plan,
        rank=rank,
        world_size=world_size,
        token_start=rank * local_tokens,
        token_end=(rank + 1) * local_tokens,
    )
    cu_seqlens = torch.tensor(plan.local_cu_seqlens, dtype=torch.int32, device=device)
    transport = _LogicalCPRingTransport(args.logical_cp, logical_rank, rank, world_size)
    assert isinstance(transport, KimiCPBoundaryTransport)

    def execute(measure: bool):
        inputs = [tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)]
        transport.clear_events()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if measure:
            start.record()
        output = cudnn_kda_cp(
            *inputs,
            cu_seqlens,
            plan=plan,
            group=dist.group.WORLD,
            boundary_transport=transport,
        )
        output.backward(grad_output)
        if measure:
            end.record()
            end.synchronize()
        else:
            torch.cuda.synchronize(device)
        transport_times = {}
        if measure:
            for direction, begin, after_comm, after_merge in transport.events:
                transport_times[f"comm_{direction}_ms"] = begin.elapsed_time(after_comm)
                transport_times[f"merge_{direction}_ms"] = after_comm.elapsed_time(after_merge)
        return {
            "cuda_total_ms": start.elapsed_time(end) if measure else None,
            "transport_ms": transport_times,
            "proxy_loss": (output.float() * grad_output.float()).sum().item(),
            "output_norm": output.float().norm().item(),
            "grad_norms": [tensor.grad.float().norm().item() for tensor in inputs],
            "finite": bool(torch.isfinite(output).all())
            and all(bool(torch.isfinite(tensor.grad).all()) for tensor in inputs),
        }

    dist.barrier(device_ids=[local_rank])
    execute(False)
    dist.barrier(device_ids=[local_rank])
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(args.repeats):
        dist.barrier(device_ids=[local_rank])
        wall_start = time.perf_counter()
        sample = execute(True)
        sample["wall_ms"] = (time.perf_counter() - wall_start) * 1000
        samples.append(sample)

    record = {
        "physical_rank": rank,
        "logical_rank": logical_rank,
        "samples": samples,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "transport_workspace_gib": transport.workspace_bytes / 2**30,
    }
    records = [None] * world_size
    dist.all_gather_object(records, record)
    if rank == 0:
        critical_cuda = [max(records[r]["samples"][i]["cuda_total_ms"] for r in range(world_size)) for i in range(args.repeats)]
        critical_wall = [max(records[r]["samples"][i]["wall_ms"] for r in range(world_size)) for i in range(args.repeats)]
        print(
            json.dumps(
                {
                    "mode": "kda_logical_cp_e2e",
                    "logical_global_tokens": args.global_tokens,
                    "logical_cp": args.logical_cp,
                    "local_tokens": local_tokens,
                    "physical_gpus": world_size,
                    "shape": {
                        "heads": args.heads,
                        "key_dim": args.key_dim,
                        "value_dim": args.value_dim,
                    },
                    "critical_cuda_samples_ms": critical_cuda,
                    "critical_cuda_median_ms": statistics.median(critical_cuda),
                    "critical_wall_samples_ms": critical_wall,
                    "critical_wall_median_ms": statistics.median(critical_wall),
                    "records": records,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-tokens", type=_positive_int, default=1 << 20)
    parser.add_argument("--logical-cp", type=_positive_int, default=64)
    parser.add_argument("--heads", type=_positive_int, default=32)
    parser.add_argument("--key-dim", type=_positive_int, default=128)
    parser.add_argument("--value-dim", type=_positive_int, default=128)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=20261102)
    return parser.parse_args()


if __name__ == "__main__":
    _run(_parse_args())
