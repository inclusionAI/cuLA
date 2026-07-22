import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fla.ops.cp import build_cp_context

from cula.kda import chunk_kda


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
        raise RuntimeError(f"This check requires exactly {expected_world_size} ranks, got {world_size}.")
    return rank, local_rank, world_size


def _make_inputs(*, rank: int, world_size: int, device: torch.device, sequence_length: int, heads: int):
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    shape = (1, sequence_length // world_size, heads, 128)
    q = torch.rand(shape, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.rand(shape, generator=generator, device=device, dtype=torch.bfloat16)
    q = F.normalize(q.float(), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(k.float(), p=2, dim=-1).to(torch.bfloat16)
    v = torch.rand(shape, generator=generator, device=device, dtype=torch.bfloat16)
    gate_logits = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    g = F.logsigmoid(gate_logits).clamp_(-5, 0).to(torch.bfloat16)
    beta_logits = torch.randn(shape[:-1], generator=generator, device=device, dtype=torch.float32)
    beta = beta_logits.sigmoid_().to(torch.bfloat16)
    return q, k, v, g, beta


def _run_backend(*, backend: str, inputs, cp_context, heads: int):
    os.environ["CULA_CP_COMM_BACKEND"] = backend
    os.environ["CULA_CP_OVERLAP"] = "1" if backend == "nvshmem" else "0"
    os.environ["CULA_CP_NVSHMEM_READY_WAIT"] = "1"
    q, k, v, g, beta = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    a_log = torch.zeros(heads, device=q.device, dtype=torch.float32, requires_grad=True)
    dt_bias = torch.zeros(heads * 128, device=q.device, dtype=torch.float32, requires_grad=True)
    out, _ = chunk_kda(
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
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=cp_context.cu_seqlens,
        cu_seqlens_cpu=cp_context.cu_seqlens_cpu,
        cp_context=cp_context,
    )
    out.float().square().mean().backward()
    torch.cuda.synchronize(q.device)
    values = {
        "output": out.detach(),
        "grad_q": q.grad,
        "grad_k": k.grad,
        "grad_v": v.grad,
        "grad_g": g.grad,
        "grad_beta": beta.grad,
    }
    if any(value is None for value in values.values()):
        missing = [name for name, value in values.items() if value is None]
        raise RuntimeError(f"Missing active gradients for {missing}")
    return values


def _global_max(value: torch.Tensor) -> float:
    value = value.detach().float()
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def run(args: argparse.Namespace) -> dict:
    rank, local_rank, world_size = _init_distributed(args.world_size)
    device = torch.device("cuda", local_rank)
    if args.sequence_length % world_size:
        raise ValueError("sequence length must be divisible by world size")

    inputs = _make_inputs(
        rank=rank,
        world_size=world_size,
        device=device,
        sequence_length=args.sequence_length,
        heads=args.heads,
    )
    global_cu_seqlens = torch.tensor([0, args.sequence_length], device=device, dtype=torch.long)
    cp_context = build_cp_context(cu_seqlens=global_cu_seqlens, group=dist.group.WORLD)

    reference = _run_backend(backend="fla", inputs=inputs, cp_context=cp_context, heads=args.heads)
    dist.barrier()
    candidate = _run_backend(backend="nvshmem", inputs=inputs, cp_context=cp_context, heads=args.heads)

    differences = {
        name: _global_max((candidate[name].float() - reference[name].float()).abs().max()) for name in reference
    }
    passed = differences["output"] <= args.output_atol and max(
        difference for name, difference in differences.items() if name != "output"
    ) <= args.gradient_atol
    result = {
        "world_size": world_size,
        "sequence_length": args.sequence_length,
        "heads": args.heads,
        "output_atol": args.output_atol,
        "gradient_atol": args.gradient_atol,
        "max_abs_diff": differences,
        "passed": passed,
    }
    dist.barrier()
    dist.destroy_process_group()
    return result if rank == 0 else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--output-atol", type=float, default=1e-2)
    parser.add_argument("--gradient-atol", type=float, default=2e-2)
    args = parser.parse_args()
    result = run(args)
    if result:
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
