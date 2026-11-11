# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark one cuDNN KDA + DSA-indexer layer on context-parallel shards.

Two modes deliberately answer different questions:

* ``real`` runs a real CP group. KDA uses cuLA's prepared all-gather
  transport, while the DSA branch pools local index keys and all-gathers the
  compressed rows before deterministic selection.
* ``logical`` maps representative ranks from a larger logical CP group onto
  the physical processes.  It keeps the real per-rank tensor shapes and runs
  real NCCL ring hops for both KDA boundary summaries and compressed DSA
  blocks.  ``full-ring`` uses ``logical_cp - 1`` hops as a conservative
  baseline.  ``sample-chain`` uses the longest packed sample's CP span, which
  models a sample-aware THD communicator.  Both are critical-path emulators,
  not claims about a real scale-out topology.

The benchmark compares a conservative serial schedule with an overlap schedule
that runs the independent DSA indexer branch on a dedicated CUDA stream while
KDA executes on the caller stream.  Arithmetic precision and deterministic
selection are identical in both schedules.

The DSA timing covers index pooling, compressed-block communication and Top-K
selection.  It does not claim to include the block-sparse attention kernel,
which is not yet part of cuLA's public hybrid runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.large_cp_emulation.topology import LogicalCPEmulationPlan  # noqa: E402
from benchmarks.large_cp_emulation.transport import (  # noqa: E402
    LogicalCPRingTransport,
    emulate_ring_hops,
)
from benchmarks.large_cp_emulation.workload import PackedTHDWorkload  # noqa: E402
from cula.distributed import (  # noqa: E402
    AllGatherKimiCPTransport,
    KimiCPPlan,
    build_kimi_cp_plan,
    cudnn_kda_cp,
)
from cula.hybrid import (  # noqa: E402
    DsaSelectionPlan,
    DsaSelectionWorkspace,
    IndexPoolPlan,
    cudnn_dsa_select,
    cudnn_index_pool,
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


@dataclass(slots=True)
class _DsaEvents:
    start: torch.cuda.Event
    after_pool: torch.cuda.Event
    after_comm: torch.cuda.Event
    end: torch.cuda.Event

    def timings(self) -> dict[str, float]:
        return {
            "pool_ms": self.start.elapsed_time(self.after_pool),
            "comm_ms": self.after_pool.elapsed_time(self.after_comm),
            "select_ms": self.after_comm.elapsed_time(self.end),
            "total_ms": self.start.elapsed_time(self.end),
        }


@dataclass(slots=True)
class _TimedKimiCPTransport:
    """Benchmark-only CUDA timing wrapper for KDA boundary propagation."""

    transport: object
    events: list[tuple[bool, torch.cuda.Event, torch.cuda.Event]]

    @classmethod
    def wrap(cls, transport: object) -> _TimedKimiCPTransport:
        return cls(transport=transport, events=[])

    @property
    def workspace_bytes(self) -> int:
        return self.transport.workspace_bytes

    def clear_events(self) -> None:
        self.events.clear()

    def boundary_summaries(
        self,
        h: torch.Tensor,
        transition: torch.Tensor,
        *,
        plan: KimiCPPlan,
        group: dist.ProcessGroup | None,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = self.transport.boundary_summaries(
            h,
            transition,
            plan=plan,
            group=group,
            reverse=reverse,
        )
        end.record()
        self.events.append((reverse, begin, end))
        return result

    def timings(self) -> dict[str, float]:
        totals = {False: 0.0, True: 0.0}
        for reverse, begin, end in self.events:
            totals[reverse] += begin.elapsed_time(end)
        return {
            "forward_ms": totals[False],
            "reverse_ms": totals[True],
            "total_ms": totals[False] + totals[True],
        }


@dataclass(slots=True)
class _DsaIndexerBranch:
    mode: str
    logical_cp: int
    logical_rank: int
    physical_rank: int
    physical_world_size: int
    transport_hops: int
    local_tokens: int
    local_blocks: int
    global_block_start: int
    key_start: int
    key_end: int
    qi: torch.Tensor
    local_kv: torch.Tensor
    pool_score: torch.Tensor
    ape: torch.Tensor
    weights: torch.Tensor
    pool_plan: IndexPoolPlan
    pool_cu_tokens: torch.Tensor
    pool_cu_blocks: torch.Tensor
    select_plan: DsaSelectionPlan
    select_workspace: DsaSelectionWorkspace
    global_kbar: torch.Tensor
    ring_receive: torch.Tensor
    last_result: object | None = None

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        logical_cp: int,
        logical_rank: int,
        physical_rank: int,
        physical_world_size: int,
        transport_hops: int,
        global_cu_seqlens: tuple[int, ...],
        shard_plan: KimiCPPlan,
        head_dim: int,
        index_heads: int,
        top_k: int,
        block_size: int,
        candidate_budget_bytes: int | None,
        device: torch.device,
    ) -> _DsaIndexerBranch:
        global_tokens = global_cu_seqlens[-1]
        local_tokens = shard_plan.local_tokens
        if local_tokens % block_size:
            raise ValueError("local_tokens must be divisible by the DSA block size")
        sequence_lengths = [right - left for left, right in zip(global_cu_seqlens, global_cu_seqlens[1:])]
        if any(length % block_size for length in sequence_lengths):
            raise ValueError("every packed sequence length must be divisible by the DSA block size")
        local_blocks = local_tokens // block_size
        global_blocks = global_tokens // block_size
        pool_plan = IndexPoolPlan.from_cu_seqlens(
            shard_plan.local_cu_seqlens,
            head_dim=head_dim,
            ratio=block_size,
        )
        if pool_plan.total_comp != local_blocks:
            raise ValueError("the CP shard must cut packed samples on DSA block boundaries")
        pool_cu_tokens, pool_cu_blocks = pool_plan.materialize(device)
        global_block_prefix = [0]
        for length in sequence_lengths:
            global_block_prefix.append(global_block_prefix[-1] + length // block_size)
        first_sequence = shard_plan.first_sequence
        last_sequence = shard_plan.last_sequence
        select_cu_k = [0]
        select_offsets = []
        for sequence in range(first_sequence, last_sequence + 1):
            select_cu_k.append(select_cu_k[-1] + global_block_prefix[sequence + 1] - global_block_prefix[sequence])
            local_sequence_start = max(shard_plan.token_start, global_cu_seqlens[sequence])
            select_offsets.append(local_sequence_start - global_cu_seqlens[sequence])
        select_plan = DsaSelectionPlan.from_cu_seqlens(
            shard_plan.local_cu_seqlens,
            select_cu_k,
            top_k=top_k,
            block_size=block_size,
            q_causal_offsets=select_offsets,
            candidate_budget_bytes=candidate_budget_bytes,
        )
        return cls(
            mode=mode,
            logical_cp=logical_cp,
            logical_rank=logical_rank,
            physical_rank=physical_rank,
            physical_world_size=physical_world_size,
            transport_hops=transport_hops,
            local_tokens=local_tokens,
            local_blocks=local_blocks,
            global_block_start=shard_plan.token_start // block_size,
            key_start=global_block_prefix[first_sequence],
            key_end=global_block_prefix[last_sequence + 1],
            qi=torch.randn(local_tokens, index_heads, head_dim, dtype=torch.bfloat16, device=device),
            local_kv=torch.randn(local_tokens, head_dim, dtype=torch.bfloat16, device=device),
            pool_score=torch.randn(local_tokens, head_dim, dtype=torch.bfloat16, device=device),
            ape=torch.randn(block_size, head_dim, dtype=torch.float32, device=device),
            weights=torch.randn(local_tokens, index_heads, dtype=torch.float32, device=device),
            pool_plan=pool_plan,
            pool_cu_tokens=pool_cu_tokens,
            pool_cu_blocks=pool_cu_blocks,
            select_plan=select_plan,
            select_workspace=DsaSelectionWorkspace.from_plan(select_plan, device),
            global_kbar=torch.empty(global_blocks, head_dim, dtype=torch.bfloat16, device=device),
            ring_receive=torch.empty(local_blocks, head_dim, dtype=torch.bfloat16, device=device),
        )

    @property
    def workspace_bytes(self) -> int:
        tensors = (
            self.select_workspace.candidates,
            self.select_workspace.block_ids,
            self.select_workspace.scores,
            self.global_kbar,
            self.ring_receive,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _communicate(self, local_kbar: torch.Tensor) -> None:
        if self.mode == "real":
            if hasattr(dist, "all_gather_single"):
                dist.all_gather_single(self.global_kbar, local_kbar.contiguous())
            else:  # pragma: no cover - compatibility with older PyTorch releases
                dist.all_gather_into_tensor(self.global_kbar, local_kbar.contiguous())
            return

        local_slice = self.global_kbar[self.global_block_start : self.global_block_start + self.local_blocks]
        local_slice.copy_(local_kbar)
        # The ring alternates buffers; local_kbar is scratch from here on.
        emulate_ring_hops(
            local_kbar,
            self.ring_receive,
            physical_rank=self.physical_rank,
            physical_world_size=self.physical_world_size,
            hops=self.transport_hops,
        )

    def execute(self) -> _DsaEvents:
        events = _DsaEvents(*(torch.cuda.Event(enable_timing=True) for _ in range(4)))
        events.start.record()
        with torch.cuda.nvtx.range("dsa.index_pool"):
            local_kbar = cudnn_index_pool(
                self.local_kv,
                self.pool_score,
                self.ape,
                self.pool_cu_tokens,
                self.pool_cu_blocks,
                plan=self.pool_plan,
                deterministic=True,
            )
        events.after_pool.record()
        with torch.cuda.nvtx.range("dsa.compressed_block_transport"):
            self._communicate(local_kbar)
        events.after_comm.record()
        with torch.cuda.nvtx.range("dsa.select"):
            self.last_result = cudnn_dsa_select(
                self.qi,
                self.global_kbar[self.key_start : self.key_end, None, :],
                self.weights,
                plan=self.select_plan,
                workspace=self.select_workspace,
                deterministic=True,
            )
        events.end.record()
        return events


def _run(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("the hybrid CP benchmark requires at least two physical processes")

    workload = (
        PackedTHDWorkload.short_long_mix(
            args.global_tokens,
            short_sequences=args.short_sequences,
            short_tokens=args.short_tokens,
            long_sequences=args.long_sequences,
        )
        if args.thd_20s2l
        else PackedTHDWorkload.single_sequence(args.global_tokens)
    )
    global_cu_seqlens = workload.global_cu_seqlens
    emulation = None
    if args.mode == "real":
        logical_cp = world_size
        if args.global_tokens % logical_cp:
            raise ValueError("global_tokens must be divisible by the CP size")
        local_tokens = args.global_tokens // logical_cp
        logical_rank = rank
        transport_hops = logical_cp - 1
    else:
        emulation = LogicalCPEmulationPlan.build(
            global_cu_seqlens,
            logical_cp=args.logical_cp,
            physical_rank=rank,
            physical_world_size=world_size,
            transport_mode=args.logical_transport,
        )
        logical_cp = emulation.logical_cp
        logical_rank = emulation.logical_rank
        local_tokens = emulation.logical_plan.local_tokens
        transport_hops = emulation.transport_hops
    torch.manual_seed(args.seed + logical_rank)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=device)

    q = torch.nn.functional.normalize(randn(local_tokens, args.heads, args.key_dim).float(), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(randn(local_tokens, args.heads, args.key_dim).float(), dim=-1).to(torch.bfloat16)
    v = randn(local_tokens, args.heads, args.value_dim)
    g = -(0.01 + 0.04 * torch.rand(local_tokens, args.heads, args.key_dim, dtype=torch.float32, device=device))
    beta = torch.sigmoid(randn(local_tokens, args.heads, dtype=torch.float32))
    grad_output = (randn(local_tokens, args.heads, args.value_dim) / float(args.global_tokens)).contiguous()

    if args.mode == "real":
        kda_plan = build_kimi_cp_plan(global_cu_seqlens, group=dist.group.WORLD)
        logical_shard_plan = kda_plan
    else:
        assert emulation is not None
        logical_shard_plan = emulation.logical_plan
        kda_plan = emulation.runtime_plan
    kda_cu_seqlens = torch.tensor(kda_plan.local_cu_seqlens, dtype=torch.int32, device=device)
    if args.mode == "real":
        kda_transport = AllGatherKimiCPTransport.prepare(
            kda_plan,
            heads=args.heads,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            device=device,
        )
        if args.profile_kda_transport:
            kda_transport = _TimedKimiCPTransport.wrap(kda_transport)
    else:
        kda_transport = LogicalCPRingTransport(
            logical_cp,
            rank,
            world_size,
            ring_hops=transport_hops,
        )

    candidate_budget_bytes = None if args.dsa_budget_mib == 0 else args.dsa_budget_mib << 20
    dsa = _DsaIndexerBranch.create(
        mode=args.mode,
        logical_cp=logical_cp,
        logical_rank=logical_rank,
        physical_rank=rank,
        physical_world_size=world_size,
        transport_hops=transport_hops,
        global_cu_seqlens=global_cu_seqlens,
        shard_plan=logical_shard_plan,
        head_dim=args.index_dim,
        index_heads=args.index_heads,
        top_k=args.top_k,
        block_size=args.block_size,
        candidate_budget_bytes=candidate_budget_bytes,
        device=device,
    )
    if args.mode == "logical":
        # Populate remote logical block ranges once. Timed iterations update
        # the local logical slice after emulating the selected transport span.
        dsa.global_kbar.normal_()

    dsa_stream = torch.cuda.Stream(device=device)
    profile_marker = torch.zeros(1, dtype=torch.int32, device=device)

    def execute(schedule: str, measure: bool) -> dict[str, object]:
        leaves = [tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)]
        if isinstance(kda_transport, LogicalCPRingTransport | _TimedKimiCPTransport):
            kda_transport.clear_events()
        start = torch.cuda.Event(enable_timing=True)
        kda_start = torch.cuda.Event(enable_timing=True)
        kda_forward_end = torch.cuda.Event(enable_timing=True)
        kda_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if args.cuda_profiler_capture:
            # Tiny kernels delimit the exact GPU interval in nsys.  An NVTX
            # range alone is projected onto child kernels and cannot represent
            # stream waits, which made the apparent idle fraction ambiguous.
            with torch.cuda.nvtx.range("hybrid.hotpath.begin_marker"):
                profile_marker.add_(1)
            torch.cuda.synchronize(device)
        hotpath_wall_start = time.perf_counter()
        torch.cuda.nvtx.range_push(f"hybrid.hotpath.{schedule}")
        start.record()

        if schedule == "overlap":
            dsa_stream.wait_event(start)
            with torch.cuda.stream(dsa_stream):
                dsa_events = dsa.execute()
            kda_start.record()
        else:
            dsa_events = dsa.execute()
            kda_start.record()

        with torch.cuda.nvtx.range("kda.forward_backward"):
            output = cudnn_kda_cp(
                *leaves,
                kda_cu_seqlens,
                plan=kda_plan,
                group=dist.group.WORLD,
                boundary_transport=kda_transport,
                checkpoint_every_n_tokens=args.kda_checkpoint_tokens,
            )
            kda_forward_end.record()
            output.backward(grad_output)
        kda_end.record()
        if schedule == "overlap":
            torch.cuda.current_stream(device).wait_stream(dsa_stream)
        end.record()
        torch.cuda.nvtx.range_pop()
        if args.cuda_profiler_capture:
            with torch.cuda.nvtx.range("hybrid.hotpath.end_marker"):
                profile_marker.add_(1)
        end.synchronize()
        hotpath_wall_ms = (time.perf_counter() - hotpath_wall_start) * 1000

        result = dsa.last_result
        assert result is not None
        sample: dict[str, object] = {
            "cuda_total_ms": start.elapsed_time(end) if measure else None,
            "wall_ms": hotpath_wall_ms if measure else None,
            "kda_ms": kda_start.elapsed_time(kda_end) if measure else None,
            "kda_forward_ms": kda_start.elapsed_time(kda_forward_end) if measure else None,
            "kda_backward_ms": kda_forward_end.elapsed_time(kda_end) if measure else None,
            "dsa_ms": dsa_events.timings() if measure else None,
            "proxy_loss": (output.float() * grad_output.float()).sum().item(),
            "output_norm": output.float().norm().item(),
            "grad_norms": [tensor.grad.float().norm().item() for tensor in leaves],
            "kda_finite": bool(torch.isfinite(output).all())
            and all(bool(torch.isfinite(tensor.grad).all()) for tensor in leaves),
            "dsa_has_nan": bool(torch.isnan(result.scores).any()),
            "dsa_id_sum": int(result.block_ids.to(torch.int64).sum().item()),
            "dsa_finite_score_sum": float(result.scores[torch.isfinite(result.scores)].double().sum().item()),
        }
        if measure and isinstance(kda_transport, LogicalCPRingTransport):
            sample["kda_transport_ms"] = {
                f"{direction}_{stage}_ms": (
                    begin.elapsed_time(after_comm) if stage == "comm" else after_comm.elapsed_time(after_prepare)
                )
                for direction, begin, after_comm, after_prepare in kda_transport.events
                for stage in ("comm", "prepare")
            }
        if measure and isinstance(kda_transport, _TimedKimiCPTransport):
            sample["kda_boundary_ms"] = kda_transport.timings()
        return sample

    schedules = ("serial", "overlap") if args.schedule == "both" else (args.schedule,)
    if args.cuda_profiler_capture and len(schedules) != 1:
        raise ValueError("--cuda-profiler-capture requires --schedule serial or --schedule overlap")
    all_samples: dict[str, list[dict[str, object]]] = {}
    memory_by_schedule = {}
    for schedule in schedules:
        dist.barrier(device_ids=[local_rank])
        for _ in range(args.warmups):
            execute(schedule, False)
        dist.barrier(device_ids=[local_rank])
        torch.cuda.reset_peak_memory_stats(device)
        if args.cuda_profiler_capture:
            torch.cuda.cudart().cudaProfilerStart()
        samples = []
        for _ in range(args.repeats):
            dist.barrier(device_ids=[local_rank])
            with torch.cuda.nvtx.range(f"hybrid.{schedule}"):
                sample = execute(schedule, True)
            samples.append(sample)
        if args.cuda_profiler_capture:
            torch.cuda.cudart().cudaProfilerStop()
        all_samples[schedule] = samples
        memory_by_schedule[schedule] = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }

    record = {
        "physical_rank": rank,
        "logical_rank": logical_rank,
        "samples": all_samples,
        "memory_by_schedule": memory_by_schedule,
        "kda_transport_workspace_gib": kda_transport.workspace_bytes / 2**30,
        "dsa_workspace_gib": dsa.workspace_bytes / 2**30,
        "dsa_windows": len(dsa.select_plan.windows),
        "dsa_candidate_workspace_gib": dsa.select_plan.candidate_workspace_floats * 4 / 2**30,
    }
    records = [None] * world_size
    dist.all_gather_object(records, record)
    if rank == 0:
        summary = {}
        for schedule in schedules:
            critical_cuda = [
                max(records[r]["samples"][schedule][i]["cuda_total_ms"] for r in range(world_size))
                for i in range(args.repeats)
            ]
            critical_wall = [
                max(records[r]["samples"][schedule][i]["wall_ms"] for r in range(world_size)) for i in range(args.repeats)
            ]
            summary[schedule] = {
                "critical_cuda_samples_ms": critical_cuda,
                "critical_cuda_median_ms": statistics.median(critical_cuda),
                "critical_wall_samples_ms": critical_wall,
                "critical_wall_median_ms": statistics.median(critical_wall),
                "rank_stage_medians_ms": [
                    {
                        "physical_rank": r,
                        "logical_rank": records[r]["logical_rank"],
                        "kda": statistics.median(sample["kda_ms"] for sample in records[r]["samples"][schedule]),
                        "kda_forward": statistics.median(
                            sample["kda_forward_ms"] for sample in records[r]["samples"][schedule]
                        ),
                        "kda_backward": statistics.median(
                            sample["kda_backward_ms"] for sample in records[r]["samples"][schedule]
                        ),
                        **(
                            {
                                f"kda_boundary_{stage}": statistics.median(
                                    sample["kda_boundary_ms"][stage] for sample in records[r]["samples"][schedule]
                                )
                                for stage in ("forward_ms", "reverse_ms", "total_ms")
                            }
                            if args.profile_kda_transport
                            else {}
                        ),
                        **{
                            f"dsa_{stage}": statistics.median(
                                sample["dsa_ms"][stage] for sample in records[r]["samples"][schedule]
                            )
                            for stage in ("pool_ms", "comm_ms", "select_ms", "total_ms")
                        },
                    }
                    for r in range(world_size)
                ],
            }
        if set(schedules) == {"serial", "overlap"}:
            serial = summary["serial"]["critical_cuda_median_ms"]
            overlap = summary["overlap"]["critical_cuda_median_ms"]
            summary["overlap"]["speedup_vs_serial"] = serial / overlap
        print(
            json.dumps(
                {
                    "mode": f"hybrid_kda_dsa_{args.mode}",
                    "global_tokens": args.global_tokens,
                    "logical_cp": logical_cp,
                    "logical_transport": args.logical_transport if args.mode == "logical" else "physical",
                    "kda_transport": "allgather" if args.mode == "real" else "logical-ring",
                    "transport_hops": transport_hops,
                    "physical_gpus": world_size,
                    "local_tokens": local_tokens,
                    "kda_checkpoint_tokens": args.kda_checkpoint_tokens,
                    "sequence_lengths": workload.sequence_lengths,
                    "schedules": summary,
                    "rank_resources": [
                        {
                            "physical_rank": r,
                            "logical_rank": records[r]["logical_rank"],
                            "dsa_windows": records[r]["dsa_windows"],
                            "dsa_candidate_workspace_gib": records[r]["dsa_candidate_workspace_gib"],
                            "dsa_workspace_gib": records[r]["dsa_workspace_gib"],
                            "kda_transport_workspace_gib": records[r]["kda_transport_workspace_gib"],
                            "memory_by_schedule": records[r]["memory_by_schedule"],
                        }
                        for r in range(world_size)
                    ],
                    "rank_checks": [
                        {
                            "physical_rank": r,
                            "logical_rank": records[r]["logical_rank"],
                            "kda_finite": all(
                                sample["kda_finite"] for schedule in schedules for sample in records[r]["samples"][schedule]
                            ),
                            "dsa_has_nan": any(
                                sample["dsa_has_nan"] for schedule in schedules for sample in records[r]["samples"][schedule]
                            ),
                            "repeat_stable": all(
                                all(
                                    sample[key] == records[r]["samples"][schedules[0]][0][key]
                                    for key in (
                                        "proxy_loss",
                                        "output_norm",
                                        "grad_norms",
                                        "dsa_id_sum",
                                        "dsa_finite_score_sum",
                                    )
                                )
                                for schedule in schedules
                                for sample in records[r]["samples"][schedule]
                            ),
                            **{
                                key: records[r]["samples"][schedules[0]][0][key]
                                for key in (
                                    "proxy_loss",
                                    "output_norm",
                                    "grad_norms",
                                    "dsa_id_sum",
                                    "dsa_finite_score_sum",
                                )
                            },
                        }
                        for r in range(world_size)
                    ],
                    "records": None if args.summary_only else records,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    real = subparsers.add_parser("real", help="run the physical processes as the real CP group")
    real.add_argument("--global-tokens", type=_positive_int, default=1 << 17)
    real.add_argument(
        "--profile-kda-transport",
        action="store_true",
        help="record CUDA time around real KDA boundary transport calls",
    )
    logical = subparsers.add_parser("logical", help="emulate a larger logical CP group")
    logical.add_argument("--global-tokens", type=_positive_int, default=1 << 20)
    logical.add_argument("--logical-cp", type=_positive_int, default=64)
    logical.add_argument(
        "--logical-transport",
        choices=("full-ring", "sample-chain"),
        default="full-ring",
        help="emulate all CP edges or only the longest packed sample's CP chain",
    )
    real.set_defaults(logical_transport="physical")
    logical.set_defaults(profile_kda_transport=False)
    for subparser in (real, logical):
        subparser.add_argument("--heads", type=_positive_int, default=32)
        subparser.add_argument("--key-dim", type=_positive_int, default=128)
        subparser.add_argument("--value-dim", type=_positive_int, default=128)
        subparser.add_argument("--index-heads", type=_positive_int, default=64)
        subparser.add_argument("--index-dim", type=_positive_int, default=128)
        subparser.add_argument("--top-k", type=_positive_int, default=512)
        subparser.add_argument("--block-size", type=_positive_int, default=4)
        subparser.add_argument(
            "--kda-checkpoint-tokens",
            type=int,
            choices=(0, 16),
            default=0,
            help="16 retains every native KDA chunk state; 0 recomputes them in backward",
        )
        subparser.add_argument(
            "--dsa-budget-mib",
            type=int,
            default=0,
            help="0 selects cuLA's adaptive exact-workspace policy",
        )
        subparser.add_argument("--schedule", choices=("serial", "overlap", "both"), default="both")
        subparser.add_argument(
            "--warmups",
            type=_positive_int,
            default=3,
            help="untimed iterations used to amortize cuDNN JIT and communicator setup",
        )
        subparser.add_argument("--repeats", type=_positive_int, default=5)
        subparser.add_argument("--seed", type=int, default=20261102)
        subparser.add_argument(
            "--thd-20s2l",
            action="store_true",
            help="pack 20 short samples plus two long samples into global_tokens",
        )
        subparser.add_argument("--short-sequences", type=_positive_int, default=20)
        subparser.add_argument("--short-tokens", type=_positive_int, default=1024)
        subparser.add_argument("--long-sequences", type=_positive_int, default=2)
        subparser.add_argument("--cuda-profiler-capture", action="store_true")
        subparser.add_argument(
            "--summary-only",
            action="store_true",
            help="omit per-iteration records from the JSON output",
        )
    return parser.parse_args()


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
