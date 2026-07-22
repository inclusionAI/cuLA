import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda as fla_chunk_kda

from cula.kda import chunk_kda as cula_chunk_kda


def _init_distributed(expected_world_size: int) -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    except TypeError:
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if world_size != expected_world_size:
        raise RuntimeError(f"This benchmark requires exactly {expected_world_size} ranks, got {world_size}.")
    return rank, local_rank, world_size


def run(args: argparse.Namespace) -> dict:
    rank, local_rank, world_size = _init_distributed(args.world_size)
    device = torch.device("cuda", local_rank)
    if args.sequence_length % world_size:
        raise ValueError("sequence length must be divisible by world size")

    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    shape = (1, args.sequence_length // world_size, args.heads, 128)
    q = torch.rand(shape, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.rand(shape, generator=generator, device=device, dtype=torch.bfloat16)
    q = F.normalize(q.float(), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(k.float(), p=2, dim=-1).to(torch.bfloat16)
    v = torch.rand(shape, generator=generator, device=device, dtype=torch.bfloat16)
    gate_logits = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    g = F.logsigmoid(gate_logits).clamp_(-5, 0).to(torch.bfloat16)
    beta_logits = torch.randn(shape[:-1], generator=generator, device=device, dtype=torch.float32)
    beta = beta_logits.sigmoid_().to(torch.bfloat16)
    a_log = torch.zeros(args.heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(args.heads * 128, device=device, dtype=torch.float32)
    global_cu_seqlens = torch.tensor([0, args.sequence_length], device=device, dtype=torch.long)
    cp_context = build_cp_context(cu_seqlens=global_cu_seqlens, group=dist.group.WORLD)
    os.environ["CULA_CP_COMM_BACKEND"] = args.backend
    os.environ["CULA_CP_OVERLAP"] = "1" if args.backend == "nvshmem" else "0"
    os.environ["CULA_CP_NVSHMEM_READY_WAIT"] = "1"
    implementation = fla_chunk_kda if args.backend == "fla_full" else cula_chunk_kda
    safe_gate = args.backend != "fla_full"

    def step() -> None:
        with torch.inference_mode():
            implementation(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                use_gate_in_kernel=False,
                safe_gate=safe_gate,
                lower_bound=-5.0 if safe_gate else None,
                cu_seqlens=cp_context.cu_seqlens,
                cu_seqlens_cpu=cp_context.cu_seqlens_cpu,
                cp_context=cp_context,
            )

    repeat_results = []
    for _ in range(args.repeats):
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize(device)
        dist.barrier()

        times = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

        local = torch.tensor(times, device=device)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        all_times = torch.stack(gathered).cpu()
        rank_max = all_times.amax(dim=0).tolist()
        repeat_results.append(
            {
                "global_p50_ms": round(float(all_times.flatten().median()), 4),
                "rank_p50_ms": [round(float(rank_times.median()), 4) for rank_times in all_times],
                "rank_max_median_ms": round(statistics.median(rank_max), 4),
                "rank_max_min_ms": round(min(rank_max), 4),
                "rank_max_max_ms": round(max(rank_max), 4),
            }
        )

    global_p50_values = [repeat["global_p50_ms"] for repeat in repeat_results]
    rank_max_p50_values = [repeat["rank_max_median_ms"] for repeat in repeat_results]
    result = {
        "backend": args.backend,
        "world_size": world_size,
        "sequence_length": args.sequence_length,
        "heads": args.heads,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "global_p50_ms": round(statistics.median(global_p50_values), 4),
        "rank_max_median_ms": round(statistics.median(rank_max_p50_values), 4),
        "repeat_results": repeat_results,
    }
    dist.barrier()
    dist.destroy_process_group()
    return result if rank == 0 else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=("fla_full", "fla", "nvshmem"), required=True)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    if result:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
