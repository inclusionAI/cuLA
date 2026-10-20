#!/usr/bin/env python3
"""Benchmark cuLA native-GVA Qwen3.5 prefill against SGLang's inference path.

Only config.json is read. Model weights are not loaded: tensors are generated
from the Qwen3.5 linear-attention shapes and dtype declared by the config.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections.abc import Callable

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cula.ops.qwen35_fused_kda_prefill import qwen35_fused_kda_prefill


def load_qwen35_shape(config_path: pathlib.Path, tp_size: int) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as f:
        root = json.load(f)
    config = root.get("text_config", root)

    required = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing Qwen3.5 fields: {missing}")

    global_h = int(config["linear_num_key_heads"])
    global_hv = int(config["linear_num_value_heads"])
    if global_h % tp_size or global_hv % tp_size:
        raise ValueError(f"TP={tp_size} must divide H={global_h} and HV={global_hv}")

    h = global_h // tp_size
    hv = global_hv // tp_size
    k = int(config["linear_key_head_dim"])
    v = int(config["linear_value_head_dim"])
    if hv % h:
        raise ValueError(f"Qwen3.5 GVA requires local HV % H == 0, got H={h} HV={hv}")
    if k != 128 or v != 128:
        raise ValueError(f"cuLA native-GVA prefill currently requires K=V=128, got K={k} V={v}")

    dtype_name = str(
        config.get("torch_dtype", config.get("dtype", root.get("torch_dtype", root.get("dtype", "bfloat16"))))
    ).lower()
    if dtype_name not in ("bfloat16", "bf16", "torch.bfloat16"):
        raise ValueError(f"This benchmark expects Qwen3.5 bf16 activations, got torch_dtype={dtype_name}")

    return {
        "model_type": config.get("model_type", root.get("model_type", "unknown")),
        "global_h": global_h,
        "global_hv": global_hv,
        "h": h,
        "hv": hv,
        "k": k,
        "v": v,
        "dtype": torch.bfloat16,
    }


def load_sglang(sglang_path: pathlib.Path | None):
    if sglang_path is not None:
        for candidate in (sglang_path, sglang_path / "python"):
            if candidate.exists():
                sys.path.insert(0, str(candidate))

    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    return fused_gdn_gating, TritonGDNKernel()


def benchmark_cuda(
    fn: Callable[[], object],
    warmup: int,
    rep: int,
    setup: Callable[[], None] | None = None,
) -> float:
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for start, end in zip(starts, ends, strict=True):
        if setup is not None:
            setup()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()

    samples = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    if len(samples) < 4:
        return statistics.mean(samples)
    return statistics.mean(samples[len(samples) // 4 : 3 * len(samples) // 4])


def relative_rms(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref_f = ref.float()
    diff_rms = (ref_f - out.float()).square().mean().sqrt()
    return (diff_rms / ref_f.square().mean().sqrt().clamp_min(1.0e-8)).item()


def make_inputs(
    batch: int,
    seq_len: int,
    shape: dict[str, object],
    device: torch.device,
    seed: int,
    random_initial_state: bool,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    total = batch * seq_len
    h, hv, k, v = (int(shape[name]) for name in ("h", "hv", "k", "v"))
    dtype = shape["dtype"]

    state = torch.zeros(batch, hv, k, v, device=device, dtype=torch.float32)
    if random_initial_state:
        state.normal_(mean=0.0, std=0.01)

    return {
        "q": torch.randn(1, total, h, k, device=device, dtype=dtype),
        "k": torch.randn(1, total, h, k, device=device, dtype=dtype),
        "v": torch.randn(1, total, hv, v, device=device, dtype=dtype),
        "a": torch.randn(total, hv, device=device, dtype=dtype),
        "b": torch.randn(total, hv, device=device, dtype=dtype),
        "A_log": -torch.rand(hv, device=device, dtype=torch.float32),
        "dt_bias": torch.randn(hv, device=device, dtype=torch.float32) * 0.1,
        "state_kv": state,
        "state_vk": state.transpose(-1, -2).contiguous(),
        "cu_seqlens": torch.arange(0, total + 1, seq_len, device=device, dtype=torch.int32),
        "cache_indices": torch.arange(batch, device=device, dtype=torch.int32),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=pathlib.Path, required=True)
    parser.add_argument("--sglang-path", type=pathlib.Path, default=pathlib.Path("/sgl-workspace/sglang"))
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=(128, 256, 512, 1024, 2048, 4096))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-initial-state", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    shape = load_qwen35_shape(args.config_json, args.tp_size)
    fused_gdn_gating, sglang_kernel = load_sglang(args.sglang_path)
    device = torch.device("cuda")

    print("Qwen3.5 GDN prefill: cuLA native GVA vs SGLang Triton inference path")
    print(f"config={args.config_json} model_type={shape['model_type']}")
    print(
        f"device={torch.cuda.get_device_name(device)} TP={args.tp_size} "
        f"global H/HV={shape['global_h']}/{shape['global_hv']} "
        f"local H/HV={shape['h']}/{shape['hv']} K/V={shape['k']}/{shape['v']}"
    )
    print(f"batch={args.batch} warmup={args.warmup} rep={args.rep}")
    print()
    print(f"{'T/seq':>8} {'tokens':>8} {'SGLang ms':>11} {'cuLA ms':>10} {'speedup':>9} {'out rrms':>11} {'state rrms':>12}")
    print("-" * 79)

    for seq_len in args.seq_lens:
        x = make_inputs(
            args.batch,
            seq_len,
            shape,
            device,
            args.seed,
            args.random_initial_state,
        )
        state_sglang = torch.empty_like(x["state_vk"])

        def setup_sglang():
            state_sglang.copy_(x["state_vk"])

        def run_cula():
            return qwen35_fused_kda_prefill(
                x["q"],
                x["k"],
                x["v"],
                x["a"],
                x["b"],
                x["A_log"],
                x["dt_bias"],
                initial_state=x["state_kv"],
                cu_seqlens=x["cu_seqlens"],
                output_final_state=True,
            )

        def run_sglang():
            g, beta = fused_gdn_gating(
                x["A_log"],
                x["a"],
                x["b"],
                x["dt_bias"],
            )
            return sglang_kernel.extend(
                x["q"],
                x["k"],
                x["v"],
                g,
                beta,
                ssm_states=state_sglang,
                cache_indices=x["cache_indices"],
                query_start_loc=x["cu_seqlens"],
            )

        rrms = float("nan")
        state_rrms = float("nan")
        if not args.skip_accuracy:
            out_cula, state_cula = run_cula()
            setup_sglang()
            out_sglang = run_sglang()[0]
            torch.cuda.synchronize()
            rrms = relative_rms(out_sglang, out_cula)
            state_rrms = relative_rms(state_sglang, state_cula.transpose(-1, -2))

        sglang_ms = benchmark_cuda(run_sglang, args.warmup, args.rep, setup=setup_sglang)
        cula_ms = benchmark_cuda(run_cula, args.warmup, args.rep)
        total = args.batch * seq_len
        print(
            f"{seq_len:8d} {total:8d} {sglang_ms:11.4f} {cula_ms:10.4f} "
            f"{sglang_ms / cula_ms:8.3f}x {rrms:11.3e} {state_rrms:12.3e}"
        )
        del x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
