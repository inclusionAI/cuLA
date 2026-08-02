#!/usr/bin/env python3
"""Benchmark the fused cuLA Qwen3.5 decode kernel against upstream FLA.

The upstream FLA recurrent operator receives pre-laid-out Q/K/V tensors. cuLA
receives the packed Qwen3.5 ``mixed_qkv_conv`` tensor and performs layout plus
the recurrent update in one kernel. State reset is outside both timing windows.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

import cula.cudac as cula_cuda
from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG as GLOBAL_CONFIG
from cula.qwen35.common import Qwen35LinearAttentionConfig


def local_config_from_tp_size(tp_size: int) -> Qwen35LinearAttentionConfig:
    return Qwen35LinearAttentionConfig(
        hidden_size=GLOBAL_CONFIG.hidden_size // tp_size,
        conv_kernel_size=GLOBAL_CONFIG.conv_kernel_size,
        num_k_heads=GLOBAL_CONFIG.num_k_heads // tp_size,
        num_v_heads=GLOBAL_CONFIG.num_v_heads // tp_size,
        head_k_dim=GLOBAL_CONFIG.head_k_dim,
        head_v_dim=GLOBAL_CONFIG.head_v_dim,
        qkv_dtype=GLOBAL_CONFIG.qkv_dtype,
        state_dtype=GLOBAL_CONFIG.state_dtype,
    )


def benchmark_cuda(fn, *, setup=None, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        if setup is not None:
            setup()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    lo, hi = len(times) // 4, 3 * len(times) // 4
    return statistics.mean(times[lo:hi] or times)


def error_stats(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    reference = reference.float()
    actual = actual.float()
    diff = (reference - actual).abs()
    rel_rms = diff.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-8)
    return rel_rms.item(), diff.max().item()


def make_inputs(tokens: int, *, tp_size: int, seed: int, device: torch.device):
    config = local_config_from_tp_size(tp_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    hv, k_dim, v_dim = config.num_v_heads, config.head_k_dim, config.head_v_dim

    mixed_qkv = torch.randn(
        tokens,
        config.conv_dim,
        generator=generator,
        device=device,
        dtype=config.qkv_dtype,
    )
    a = torch.randn(tokens, hv, generator=generator, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, hv, generator=generator, device=device, dtype=config.qkv_dtype)
    A_log = -torch.rand(hv, generator=generator, device=device, dtype=torch.float32)
    dt_bias = torch.randn(hv, generator=generator, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(
        tokens,
        hv,
        k_dim,
        v_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * 0.01
    indices = torch.arange(tokens, device=device, dtype=torch.int32)
    return config, mixed_qkv, a, b, A_log, dt_bias, state, indices


def run_case(tokens: int, args, device: torch.device) -> dict[str, float | int]:
    config, mixed, a, b, A_log, dt_bias, state, indices = make_inputs(
        tokens,
        tp_size=args.tp_size,
        seed=args.seed,
        device=device,
    )
    qk_width = config.num_k_heads * config.head_k_dim
    q = mixed[:, :qk_width].view(tokens, config.num_k_heads, config.head_k_dim).contiguous()
    k = mixed[:, qk_width : 2 * qk_width].view(
        tokens, config.num_k_heads, config.head_k_dim
    ).contiguous()
    v = mixed[:, 2 * qk_width :].view(tokens, config.num_v_heads, config.head_v_dim).contiguous()
    q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    gate = a.contiguous().unsqueeze(1)
    beta = torch.sigmoid(b.float()).unsqueeze(1)

    state_cula = torch.empty_like(state)
    out_cula = torch.empty_like(v.squeeze(1))

    def run_fla():
        return fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            scale=config.head_k_dim**-0.5,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            transpose_state_layout=False,
        )

    def setup_cula() -> None:
        state_cula.copy_(state)

    def run_cula() -> None:
        cula_cuda.qwen35_layout_scalar_kda_decode(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            state_cula,
            indices,
            out_cula,
        )

    out_fla, state_fla = run_fla()
    setup_cula()
    run_cula()
    torch.cuda.synchronize()
    out_rel_rms, out_max_abs = error_stats(out_fla.squeeze(1), out_cula)
    state_rel_rms, state_max_abs = error_stats(state_fla, state_cula)

    fla_ms = benchmark_cuda(run_fla, warmup=args.warmup, rep=args.rep)
    cula_ms = benchmark_cuda(run_cula, setup=setup_cula, warmup=args.warmup, rep=args.rep)
    return {
        "tokens": tokens,
        "upstream_fla_ms": fla_ms,
        "cula_fused_ms": cula_ms,
        "speedup": fla_ms / cula_ms,
        "out_rel_rms": out_rel_rms,
        "out_max_abs": out_max_abs,
        "state_rel_rms": state_rel_rms,
        "state_max_abs": state_max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upstream FLA vs fused cuLA Qwen3.5 decode")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tp-size", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--csv", type=pathlib.Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    print(
        f"device={torch.cuda.get_device_name(device)} torch={torch.__version__} "
        f"cuda={torch.version.cuda} tp={args.tp_size} warmup/rep={args.warmup}/{args.rep}"
    )
    print("scope: upstream FLA recurrent operator vs cuLA fused Qwen layout + recurrent kernel")
    print("| tokens | upstream_fla_ms | cula_fused_ms | speedup | out_rel_rms | state_rel_rms |")
    print("|---:|---:|---:|---:|---:|---:|")

    rows = []
    for tokens in args.tokens:
        row = run_case(tokens, args, device)
        rows.append(row)
        print(
            f"| {tokens} | {row['upstream_fla_ms']:.4f} | {row['cula_fused_ms']:.4f} | "
            f"{row['speedup']:.2f}x | {row['out_rel_rms']:.3e} | {row['state_rel_rms']:.3e} |"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
