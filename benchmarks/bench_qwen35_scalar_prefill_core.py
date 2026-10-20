#!/usr/bin/env python3
"""Benchmark the optimized legacy Qwen scalar prefill CUDA kernel.

The reference path is the real SGLang ``TritonGDNKernel.extend`` path.  Both
implementations receive compact native-GVA Q/K and full-HV V/gates.  The table
is deliberately a compute-kernel comparison: SGLang's
``fused_gdn_gating`` is warmed and evaluated before timing, while cuLA's
legacy kernel includes its raw ``a/b`` gate conversion, making the comparison
conservative for cuLA.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import pathlib
import statistics
import subprocess
import sys
from typing import Callable

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cula.ops.qwen35_scalar_kda_prefill import qwen35_scalar_kda_prefill, qwen35_scalar_kda_prefill_core


def _run_text(command: list[str], *, cwd: pathlib.Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_head(path: pathlib.Path) -> str:
    return _run_text(["git", "rev-parse", "HEAD"], cwd=path) or "unavailable"


def _tracked_source_state(path: pathlib.Path) -> str:
    status = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=path,
    )
    if status is None:
        return "unavailable"
    return "dirty" if status else "clean"


def load_shape(path: pathlib.Path, tp: int) -> dict[str, int | torch.dtype | str]:
    root = json.loads(path.read_text(encoding="utf-8"))
    cfg = root.get("text_config", root)
    h_global = int(cfg["linear_num_key_heads"])
    hv_global = int(cfg["linear_num_value_heads"])
    if h_global % tp or hv_global % tp:
        raise ValueError(f"TP={tp} must divide global H/HV={h_global}/{hv_global}")
    h, hv = h_global // tp, hv_global // tp
    if int(cfg["linear_key_head_dim"]) != 128 or int(cfg["linear_value_head_dim"]) != 128:
        raise ValueError("This scalar benchmark requires K=V=128")
    if hv % h:
        raise ValueError(f"local HV={hv} must be divisible by local H={h}")
    return {
        "model_type": str(cfg.get("model_type", root.get("model_type", "unknown"))),
        "h_global": h_global,
        "hv_global": hv_global,
        "h": h,
        "hv": hv,
        "dtype": torch.bfloat16,
    }


def _timed(fn: Callable[[], object], repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _capture_cuda_graph(fn: Callable[[], object]) -> Callable[[], object]:
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph.replay


def _rrms(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float(), b.float()
    return ((af - bf).square().mean().sqrt() / af.square().mean().sqrt().clamp_min(1.0e-8)).item()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", type=pathlib.Path, required=True)
    parser.add_argument("--sglang-path", type=pathlib.Path, default=pathlib.Path("/sgl-workspace/sglang"))
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=(1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--inner", type=int, default=1, help="kernel calls per CUDA event sample")
    parser.add_argument(
        "--preheat-iters",
        type=int,
        default=0,
        help="large BF16 GEMMs before timing to stabilize GPU clocks",
    )
    parser.add_argument(
        "--eager-timing",
        action="store_true",
        help="time Python launches directly instead of CUDA Graph replay",
    )
    parser.add_argument("--random-initial-state", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="compare the preprocessed g/beta calculation core; exclude raw a/b gate conversion on both sides",
    )
    parser.add_argument("--csv", type=pathlib.Path, help="write the exact per-shape medians, IQRs, and errors")
    parser.add_argument(
        "--min-speedup",
        type=float,
        help="fail unless every acceptance shape reaches this paired-median speedup; requires --core-only",
    )
    parser.add_argument(
        "--acceptance-seq-lens",
        type=int,
        nargs="+",
        default=(256, 512),
        help="sequence lengths checked by --min-speedup",
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help="fail when tracked source changes are present; ignored build artifacts are allowed",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch != 1:
        raise ValueError("The packed SGLang comparison currently requires --batch 1")
    if args.min_speedup is not None and not args.core_only:
        parser.error("--min-speedup is an acceptance gate and requires the apples-to-apples --core-only scope")
    missing_acceptance_shapes = sorted(set(args.acceptance_seq_lens) - set(args.seq_lens))
    if args.min_speedup is not None and missing_acceptance_shapes:
        parser.error(f"acceptance sequence lengths are missing from --seq-lens: {missing_acceptance_shapes}")
    source_state = _tracked_source_state(ROOT)
    if args.require_clean_source and source_state != "clean":
        parser.error(f"formal runs require clean tracked source, got source_state={source_state}")
    shape = load_shape(args.config_json, args.tp_size)
    import cula.cudac as cula_cuda

    for candidate in (args.sglang_path, args.sglang_path / "python"):
        if candidate.exists():
            sys.path.insert(0, str(candidate))

    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    device = torch.device("cuda")
    sg_kernel = TritonGDNKernel()
    if not hasattr(cula_cuda, "qwen35_scalar_kda_prefill_core"):
        raise RuntimeError("the loaded cuLA extension does not expose qwen35_scalar_kda_prefill_core")
    core_op = cula_cuda.qwen35_scalar_kda_prefill_core
    extension_module = sys.modules.get(getattr(core_op, "__module__", ""))
    extension_path = getattr(extension_module, "__file__", "unavailable")
    try:
        sglang_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        sglang_version = "unavailable"
    repo_head = _git_head(ROOT)
    sglang_head = _git_head(args.sglang_path)
    h, hv = int(shape["h"]), int(shape["hv"])
    print(
        f"repo_head={repo_head} tracked_source_state={source_state} "
        f"extension={extension_path}"
    )
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"sglang={sglang_version} sglang_head={sglang_head}"
    )
    print(
        f"device={torch.cuda.get_device_name(device)} config={args.config_json} "
        f"model_type={shape['model_type']} TP={args.tp_size} "
        f"global H/HV={shape['h_global']}/{shape['hv_global']} local H/HV={h}/{hv}"
    )
    print(
        f"batch={args.batch} warmup={args.warmup} rep={args.rep} inner={args.inner} "
        f"graph={'off' if args.eager_timing else 'on'} "
        f"(scope={'preprocessed core' if args.core_only else 'raw CULA gate vs SGLang core'})"
    )
    if args.preheat_iters:
        heat_a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
        heat_b = torch.randn_like(heat_a)
        for _ in range(args.preheat_iters):
            torch.mm(heat_a, heat_b)
        torch.cuda.synchronize()
        del heat_a, heat_b
    print(f"{'T':>6} {'SGLang ms':>25} {'cuLA ms':>25} {'speedup':>10} {'out rrms':>12} {'state rrms':>12}")
    print("-" * 110)
    rows: list[dict[str, int | float | str]] = []

    for seq_len in args.seq_lens:
        torch.manual_seed(7000 + seq_len + hv * 17 + args.tp_size)
        total = args.batch * seq_len
        q = torch.randn(args.batch, seq_len, h, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(args.batch, seq_len, hv, 128, device=device, dtype=torch.bfloat16)
        a = torch.randn(args.batch, seq_len, hv, device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = -torch.rand(hv, device=device, dtype=torch.float32)
        dt_bias = torch.randn(hv, device=device, dtype=torch.float32) * 0.1
        state_kv = torch.zeros(args.batch, hv, 128, 128, device=device, dtype=torch.float32)
        if args.random_initial_state:
            state_kv.normal_(mean=0.0, std=0.01)
        state_vk = state_kv.transpose(-1, -2).contiguous()
        state_sg = state_vk.clone()
        cu = torch.arange(0, total + 1, seq_len, device=device, dtype=torch.int32)
        cache_indices = torch.arange(args.batch, device=device, dtype=torch.int32)

        # Keep gating outside both timed calls.  This is the same input format
        # that SGLang's gdn_backend passes to TritonGDNKernel.extend.
        g, beta = fused_gdn_gating(A_log, a.reshape(total, hv), b.reshape(total, hv), dt_bias)
        g_core = g.reshape(args.batch, seq_len, hv).contiguous()
        beta_core = beta.reshape(args.batch, seq_len, hv).contiguous()

        out_cula = torch.empty_like(v)
        state_cula = torch.empty_like(state_kv)
        empty_initial = torch.empty(0, device=device, dtype=torch.float32)

        def run_cula() -> None:
            # Call the extension ABI directly: output/state allocation and
            # Python wrapper overhead are not part of the compute measurement.
            cula_state = state_kv
            if args.core_only:
                cula_cuda.qwen35_scalar_kda_prefill_core(
                    q, k, v, g_core, beta_core, cula_state, cu, out_cula, state_cula
                )
            else:
                cula_cuda.qwen35_scalar_kda_prefill(
                    q, k, v, a, b, A_log, dt_bias, cula_state, cu, out_cula, state_cula
                )

        def reset_sg() -> None:
            state_sg.copy_(state_vk)

        def run_sg() -> None:
            sg_kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=state_sg,
                cache_indices=cache_indices,
                query_start_loc=cu,
            )

        # Warm up each path independently, including the first Triton compile.
        for _ in range(args.warmup):
            run_cula()
            reset_sg()
            run_sg()
        torch.cuda.synchronize()

        out_rrms = float("nan")
        state_rrms = float("nan")
        if not args.skip_accuracy:
            run_cula()
            reset_sg()
            run_sg()
            torch.cuda.synchronize()
            # SGLang mutates [V,K], while cuLA writes [K,V].
            reset_sg()
            out_sg = sg_kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=state_sg,
                cache_indices=cache_indices,
                query_start_loc=cu,
            )[0]
            torch.cuda.synchronize()
            out_rrms = _rrms(out_sg, out_cula)
            state_rrms = _rrms(state_sg, state_cula.transpose(-1, -2))

        timed_sg: Callable[[], object] = run_sg
        timed_cula: Callable[[], object] = run_cula
        if not args.eager_timing:
            reset_sg()
            timed_sg = _capture_cuda_graph(run_sg)
            timed_cula = _capture_cuda_graph(run_cula)

        sg_ms: list[float] = []
        cu_ms: list[float] = []
        # Alternate order and restore the state outside each CUDA event.  This
        # avoids a systematic clock/thermal bias between the two kernels.
        for i in range(args.rep):
            if i & 1:
                reset_sg()
                cu_ms.append(_timed(timed_cula, args.inner) / args.inner)
                reset_sg()
                sg_ms.append(_timed(timed_sg, args.inner) / args.inner)
            else:
                reset_sg()
                sg_ms.append(_timed(timed_sg, args.inner) / args.inner)
                cu_ms.append(_timed(timed_cula, args.inner) / args.inner)

        def middle(xs: list[float]) -> tuple[float, float, float]:
            ys = sorted(xs)
            return statistics.median(ys), ys[len(ys) // 4], ys[(3 * len(ys)) // 4]

        sg_med, sg_q1, sg_q3 = middle(sg_ms)
        cu_med, cu_q1, cu_q3 = middle(cu_ms)
        paired = [s / c for s, c in zip(sg_ms, cu_ms)]
        speed_med, speed_q1, speed_q3 = middle(paired)
        rows.append(
            {
                "repo_head": repo_head,
                "source_state": source_state,
                "extension": extension_path,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "unavailable",
                "sglang_version": sglang_version,
                "sglang_head": sglang_head,
                "config_json": str(args.config_json.resolve()),
                "scope": "core" if args.core_only else "raw_cula_vs_sglang_core",
                "cuda_graph": not args.eager_timing,
                "random_initial_state": args.random_initial_state,
                "warmup": args.warmup,
                "rep": args.rep,
                "inner": args.inner,
                "seq_len": seq_len,
                "batch": args.batch,
                "tp_size": args.tp_size,
                "qk_heads": h,
                "v_heads": hv,
                "sglang_ms": sg_med,
                "sglang_q1_ms": sg_q1,
                "sglang_q3_ms": sg_q3,
                "cula_ms": cu_med,
                "cula_q1_ms": cu_q1,
                "cula_q3_ms": cu_q3,
                "paired_speedup": speed_med,
                "paired_speedup_q1": speed_q1,
                "paired_speedup_q3": speed_q3,
                "out_rrms": out_rrms,
                "state_rrms": state_rrms,
            }
        )
        print(
            f"{seq_len:6d} {sg_med:8.4f} [{sg_q1:8.4f},{sg_q3:8.4f}] "
            f"{cu_med:8.4f} [{cu_q1:8.4f},{cu_q3:8.4f}] "
            f"{speed_med:8.3f}x [{speed_q1:6.3f},{speed_q3:6.3f}] "
            f"{out_rrms:12.3e} {state_rrms:12.3e}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if args.min_speedup is not None:
        acceptance = {int(row["seq_len"]): float(row["paired_speedup"]) for row in rows}
        failures = {
            seq_len: acceptance[seq_len]
            for seq_len in args.acceptance_seq_lens
            if acceptance[seq_len] < args.min_speedup
        }
        if failures:
            formatted = ", ".join(f"T{seq_len}={speedup:.3f}x" for seq_len, speedup in failures.items())
            raise SystemExit(f"speedup acceptance failed (required {args.min_speedup:.3f}x): {formatted}")
        print(
            "speedup acceptance passed: "
            + ", ".join(
                f"T{seq_len}={acceptance[seq_len]:.3f}x" for seq_len in args.acceptance_seq_lens
            )
        )


if __name__ == "__main__":
    main()
