#!/usr/bin/env python3
"""Benchmark actual cuLA Qwen GDN decode against SGLang's packed inference path.

Only config.json is read. State reset is outside both CUDA event windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections.abc import Callable

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cula.cudac as cula_cuda


def load_shape(config_path: pathlib.Path, tp_size: int) -> dict[str, int | str]:
    with config_path.open(encoding="utf-8") as f:
        root = json.load(f)
    config = root.get("text_config", root)
    global_h = int(config["linear_num_key_heads"])
    global_hv = int(config["linear_num_value_heads"])
    if global_h % tp_size or global_hv % tp_size:
        raise ValueError(f"TP={tp_size} must divide H={global_h} and HV={global_hv}")
    h, hv = global_h // tp_size, global_hv // tp_size
    k = int(config["linear_key_head_dim"])
    v = int(config["linear_value_head_dim"])
    if hv % h or k != 128 or v != 128:
        raise ValueError(f"unsupported local GVA shape H={h} HV={hv} K={k} V={v}")
    return {
        "model": config_path.parent.name,
        "global_h": global_h,
        "global_hv": global_hv,
        "h": h,
        "hv": hv,
        "k": k,
        "v": v,
    }


def load_sglang(sglang_path: pathlib.Path):
    for candidate in (sglang_path, sglang_path / "python"):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    kernel = TritonGDNKernel()
    if not kernel.supports_packed_decode:
        raise RuntimeError("SGLang Triton packed GDN decode is unavailable")
    return kernel


def benchmark_cuda(
    fn: Callable[[], object],
    *,
    setup: Callable[[], None],
    warmup: int,
    rep: int,
) -> float:
    for _ in range(warmup):
        setup()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for start, end in zip(starts, ends, strict=True):
        setup()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    if len(samples) < 4:
        return statistics.mean(samples)
    return statistics.mean(samples[len(samples) // 4 : 3 * len(samples) // 4])


def relative_rms(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.float()
    diff = ref - actual.float()
    return (diff.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-8)).item()


def make_inputs(tokens: int, shape: dict[str, int | str], seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    h, hv, k, v = (int(shape[name]) for name in ("h", "hv", "k", "v"))
    conv_dim = 2 * h * k + hv * v
    state_kv = torch.randn(tokens, hv, k, v, device=device, dtype=torch.float32) * 0.01
    return {
        "mixed_qkv": torch.randn(tokens, conv_dim, device=device, dtype=torch.bfloat16),
        "a": torch.randn(tokens, hv, device=device, dtype=torch.bfloat16),
        "b": torch.randn(tokens, hv, device=device, dtype=torch.bfloat16),
        "A_log": -torch.rand(hv, device=device, dtype=torch.float32),
        "dt_bias": torch.randn(hv, device=device, dtype=torch.float32) * 0.1,
        "state_kv": state_kv,
        "state_vk": state_kv.transpose(-1, -2).contiguous(),
        "indices": torch.arange(tokens, device=device, dtype=torch.int32),
    }


@torch.inference_mode()
def run_case(tokens: int, shape, sglang_kernel, args) -> dict[str, float | int]:
    x = make_inputs(tokens, shape, args.seed)
    hv, k, v = (int(shape[name]) for name in ("hv", "k", "v"))
    state_cula = torch.empty_like(x["state_kv"])
    state_sglang = torch.empty_like(x["state_vk"])
    out_cula = torch.empty(tokens, hv, v, device="cuda", dtype=torch.bfloat16)

    def setup_cula():
        state_cula.copy_(x["state_kv"])

    def setup_sglang():
        state_sglang.copy_(x["state_vk"])

    def run_cula():
        cula_cuda.qwen35_layout_scalar_kda_decode(
            x["mixed_qkv"], x["a"], x["b"], x["A_log"], x["dt_bias"],
            state_cula, x["indices"], out_cula,
        )

    def run_sglang():
        return sglang_kernel.packed_decode(
            mixed_qkv=x["mixed_qkv"], a=x["a"], b=x["b"],
            A_log=x["A_log"], dt_bias=x["dt_bias"], scale=k**-0.5,
            ssm_states=state_sglang, cache_indices=x["indices"],
            num_v_heads=hv, head_v_dim=v,
        )

    setup_cula()
    run_cula()
    setup_sglang()
    out_sglang = run_sglang().squeeze(0)
    torch.cuda.synchronize()
    out_rrms = relative_rms(out_sglang, out_cula)
    state_rrms = relative_rms(state_sglang, state_cula.transpose(-1, -2))

    sglang_ms = benchmark_cuda(run_sglang, setup=setup_sglang, warmup=args.warmup, rep=args.rep)
    cula_ms = benchmark_cuda(run_cula, setup=setup_cula, warmup=args.warmup, rep=args.rep)
    return {
        "tokens": tokens,
        "sglang_packed_ms": sglang_ms,
        "cula_fused_ms": cula_ms,
        "speedup": sglang_ms / cula_ms,
        "out_rel_rms": out_rrms,
        "state_rel_rms": state_rrms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=pathlib.Path, required=True)
    parser.add_argument("--sglang-path", type=pathlib.Path, default=pathlib.Path("/sgl-workspace/sglang"))
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32, 64, 128))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=pathlib.Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = load_shape(args.config_json, args.tp_size)
    sglang_kernel = load_sglang(args.sglang_path)
    print(
        f"model={shape['model']} device={torch.cuda.get_device_name(0)} TP={args.tp_size} "
        f"global_H/HV={shape['global_h']}/{shape['global_hv']} "
        f"local_H/HV={shape['h']}/{shape['hv']} K/V={shape['k']}/{shape['v']}"
    )
    print("state reset is outside timing; SGLang packed decode vs cuLA fused packed decode")
    print("| tokens | sglang_packed_ms | cula_fused_ms | speedup | out_rrms | state_rrms |")
    print("|---:|---:|---:|---:|---:|---:|")
    rows = []
    for tokens in args.tokens:
        row = run_case(tokens, shape, sglang_kernel, args)
        rows.append(row)
        print(
            f"| {tokens} | {row['sglang_packed_ms']:.4f} | {row['cula_fused_ms']:.4f} | "
            f"{row['speedup']:.3f}x | {row['out_rel_rms']:.3e} | {row['state_rel_rms']:.3e} |"
        )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
