#!/usr/bin/env python3
"""
Benchmark Lightning Attention with decay: CuteDSL vs Triton (FLA).
Tests various prefill scenarios including with/without initial state (h0) and final state (ht).
"""

import torch
import time
import ctypes
import argparse
import sys
import numpy as np

from fla.ops.simple_gla.chunk import chunk_simple_gla_fwd
from fla.utils import device

sys.path.insert(0, '/ossfs/workspace/flashla')
from flashla.lightning_attn import lightning_attn_fwd


PRINT_DEBUG = False


def reset_cuda_error():
    """Reset CUDA error state after an error occurs."""
    try:
        torch.cuda.synchronize()
        libcudart = ctypes.CDLL('libcudart.so')
        libcudart.cudaGetLastError()
        torch.cuda.empty_cache()
    except Exception:
        pass


def compute_fla_decay(H, layer_idx, num_layers):
    """Compute FLA-style per-head decay. Returns (H,) tensor on CUDA."""
    # FLA: g_gamma = -(8/H * (1 - layer_idx/num_layers)) * range(H)
    # g_gamma is negative, our decay_s is positive => decay_s[h] = -g_gamma[h]
    return (8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Triton (FLA) runner
# ---------------------------------------------------------------------------
def run_triton(
    Q, K, V, decay,
    initial_state, output_final_state,
    warmup, iterations,
):
    """Run FLA chunk_simple_gla_fwd and return (output, final_state, elapsed_ms).

    Uses CUDA events for accurate GPU-only timing (excludes CPU/alloc overhead).
    FLA allocates output internally, so we measure end-to-end including that.
    """
    scale = 1.0
    g_gamma = -decay  # Our decay s > 0 => FLA g_gamma = -s

    for _ in range(warmup):
        chunk_simple_gla_fwd(
            q=Q, k=K, v=V,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=64,
        )
    torch.cuda.synchronize()

    # CUDA event timing — measures GPU time only
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        o_tri, ht_tri = chunk_simple_gla_fwd(
            q=Q, k=K, v=V,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=64,
        )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / iterations
    return o_tri, ht_tri, elapsed_ms


# ---------------------------------------------------------------------------
# CuteDSL runner
# ---------------------------------------------------------------------------
def run_cutedsl(
    Q, K, V, decay,
    has_initial_state, output_final_state,
    h0,
    warmup, iterations,
):
    """Run CuteDSL kernel and return (output, ht_tensor, elapsed_ms, compile_ms).

    Uses TVM-FFI compile cache: first call compiles, subsequent calls reuse.
    CUDA events for accurate GPU-only timing.
    """
    B, S, H, D = Q.shape
    scale = 1.0

    def _run():
        return lightning_attn_fwd(
            Q, K, V, decay, scale=scale,
            initial_state=h0.clone() if has_initial_state else None,
            output_final_state=output_final_state,
            chunk_size=64,
        )

    # First call triggers compilation if not cached
    t0 = time.time()
    _run()
    compile_ms = (time.time() - t0) * 1000

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()

    # CUDA event timing — measures GPU time only
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        O, ht = _run()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / iterations

    return O, ht, elapsed_ms, compile_ms


# ---------------------------------------------------------------------------
# Single config benchmark
# ---------------------------------------------------------------------------
def benchmark_config(
    B, T, H, D,
    layer_idx, num_layers,
    mode,  # "no_state" | "h0_ht"
    warmup=2, iterations=10,
):
    """
    Benchmark a single configuration.
    mode: "no_state"  - no initial/final state
          "h0_ht"     - provide random h0 and output ht
    Returns dict with timing and accuracy info.
    """
    torch.manual_seed(42)
    Q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    K = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    V = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)

    decay = compute_fla_decay(H, layer_idx, num_layers)

    has_initial_state = mode == "h0_ht"
    output_final_state = mode == "h0_ht"

    if has_initial_state:
        h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device) * 0.01
        initial_state_fla = h0.clone()
    else:
        h0 = None
        initial_state_fla = None

    result = {
        "B": B, "T": T, "H": H, "D": D,
        "mode": mode,
        "layer_idx": layer_idx,
        "num_layers": num_layers,
    }

    # --- Triton (FLA) ---
    triton_error = None
    try:
        o_tri, ht_tri, triton_ms = run_triton(
            Q, K, V, decay,
            initial_state=initial_state_fla,
            output_final_state=output_final_state,
            warmup=warmup, iterations=iterations,
        )
        result["triton_ms"] = triton_ms
    except Exception as e:
        triton_error = str(e)
        o_tri = None
        result["triton_ms"] = float("nan")
        reset_cuda_error()

    # --- CuteDSL ---
    cutedsl_error = None
    try:
        o_cute, ht_cute_out, cutedsl_ms, compile_ms = run_cutedsl(
            Q, K, V, decay,
            has_initial_state=has_initial_state,
            output_final_state=output_final_state,
            h0=h0,
            warmup=warmup, iterations=iterations,
        )
        result["cutedsl_ms"] = cutedsl_ms
        result["compile_ms"] = compile_ms
    except Exception as e:
        cutedsl_error = str(e)
        o_cute = None
        result["cutedsl_ms"] = float("nan")
        result["compile_ms"] = float("nan")
        reset_cuda_error()

    result["triton_error"] = triton_error
    result["cutedsl_error"] = cutedsl_error

    # --- Accuracy ---
    if o_tri is not None and o_cute is not None:
        diff_o = (o_tri - o_cute).abs()
        result["o_max_diff"] = diff_o.max().item()
        result["o_mean_diff"] = diff_o.mean().item()
        ref_mag = o_tri.abs().max().item() + 1e-8
        result["o_rel_error"] = result["o_max_diff"] / ref_mag

        if output_final_state and ht_tri is not None:
            diff_ht = (ht_tri - ht_cute_out).abs()
            result["ht_max_diff"] = diff_ht.max().item()
            result["ht_mean_diff"] = diff_ht.mean().item()
            ht_ref_mag = ht_tri.abs().max().item() + 1e-8
            result["ht_rel_error"] = result["ht_max_diff"] / ht_ref_mag
        else:
            result["ht_max_diff"] = float("nan")
            result["ht_mean_diff"] = float("nan")
            result["ht_rel_error"] = float("nan")
    else:
        result["o_max_diff"] = float("nan")
        result["o_mean_diff"] = float("nan")
        result["o_rel_error"] = float("nan")
        result["ht_max_diff"] = float("nan")
        result["ht_mean_diff"] = float("nan")
        result["ht_rel_error"] = float("nan")

    # --- Speedup ---
    if not np.isnan(result["triton_ms"]) and not np.isnan(result["cutedsl_ms"]):
        result["speedup"] = result["triton_ms"] / result["cutedsl_ms"]
    else:
        result["speedup"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def print_header():
    hdr = (
        f"{'Config':<30} {'Mode':<10} "
        f"{'Triton(ms)':>11} {'CuteDSL(ms)':>12} {'Speedup':>8} "
        f"{'O_maxdiff':>10} {'O_rel%':>8} "
        f"{'Ht_maxdiff':>11} {'Ht_rel%':>8} "
        f"{'Status':>8}"
    )
    print(hdr)
    print("-" * len(hdr))


def print_result(r):
    config = f"B={r['B']},T={r['T']},H={r['H']}"
    status = "OK"
    if r["triton_error"]:
        status = "TRI_ERR"
    elif r["cutedsl_error"]:
        status = "DSL_ERR"
    elif r["o_rel_error"] > 0.05:
        status = "HI_ERR"

    ht_max = f"{r['ht_max_diff']:.4f}" if not np.isnan(r['ht_max_diff']) else "-"
    ht_rel = f"{r['ht_rel_error']*100:.2f}" if not np.isnan(r['ht_rel_error']) else "-"

    tri_ms = f"{r['triton_ms']:.3f}" if not np.isnan(r['triton_ms']) else "ERR"
    dsl_ms = f"{r['cutedsl_ms']:.3f}" if not np.isnan(r['cutedsl_ms']) else "ERR"
    sp = f"{r['speedup']:.2f}x" if not np.isnan(r['speedup']) else "-"
    omd = f"{r['o_max_diff']:.4f}" if not np.isnan(r['o_max_diff']) else "-"
    orel = f"{r['o_rel_error']*100:.2f}%" if not np.isnan(r['o_rel_error']) else "-"

    print(
        f"{config:<30} {r['mode']:<10} "
        f"{tri_ms:>11} {dsl_ms:>12} {sp:>8} "
        f"{omd:>10} {orel:>8} "
        f"{ht_max:>11} {ht_rel:>7}% "
        f"{status:>8}"
    )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------
def run_benchmark_suite(args):
    """Run benchmarks across various prefill scenarios."""
    batch_sizes = args.batch_sizes
    seq_lens = args.seq_lens
    num_heads_list = args.num_heads
    D = args.head_dim
    layer_idx = args.layer_idx
    num_layers = args.num_layers
    iterations = args.iterations
    warmup = args.warmup
    modes = args.modes  # list of "no_state", "h0_ht"

    print("\n" + "=" * 80)
    print("Lightning Attention Benchmark: Prefill Scenarios")
    print("=" * 80)
    print(f"  Batch sizes:    {batch_sizes}")
    print(f"  Seq lengths:    {seq_lens}")
    print(f"  Num heads:      {num_heads_list}")
    print(f"  Head dim:       {D}")
    print(f"  Layer:          {layer_idx}/{num_layers}")
    print(f"  Modes:          {modes}")
    print(f"  Warmup/Iters:   {warmup}/{iterations}")
    print("=" * 80 + "\n")

    all_results = []
    print_header()

    for mode in modes:
        for B in batch_sizes:
            for T in seq_lens:
                for H in num_heads_list:
                    total_elems = B * T * H * D
                    if total_elems > 2_147_483_648:
                        continue
                    if T > 4096 and B > 2:
                        continue

                    r = benchmark_config(
                        B, T, H, D,
                        layer_idx, num_layers,
                        mode=mode,
                        warmup=warmup,
                        iterations=iterations,
                    )
                    all_results.append(r)
                    print_result(r)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY BY MODE")
    print("=" * 80)

    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode and not np.isnan(r["speedup"])]
        if not mode_results:
            print(f"\n  [{mode}]  No successful results.")
            continue

        speedups = [r["speedup"] for r in mode_results]
        triton_times = [r["triton_ms"] for r in mode_results]
        cutedsl_times = [r["cutedsl_ms"] for r in mode_results]
        o_rels = [r["o_rel_error"] * 100 for r in mode_results]

        print(f"\n  [{mode}]  ({len(mode_results)} configs)")
        print(f"    Speedup:       avg={np.mean(speedups):.2f}x  min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x")
        print(f"    Triton  (ms):  avg={np.mean(triton_times):.3f}  min={np.min(triton_times):.3f}  max={np.max(triton_times):.3f}")
        print(f"    CuteDSL (ms):  avg={np.mean(cutedsl_times):.3f}  min={np.min(cutedsl_times):.3f}  max={np.max(cutedsl_times):.3f}")
        print(f"    O rel err (%): avg={np.mean(o_rels):.2f}  max={np.max(o_rels):.2f}")

        if mode == "h0_ht":
            ht_rels = [r["ht_rel_error"] * 100 for r in mode_results if not np.isnan(r["ht_rel_error"])]
            if ht_rels:
                print(f"    Ht rel err(%): avg={np.mean(ht_rels):.2f}  max={np.max(ht_rels):.2f}")

    # --- Plot if requested ---
    if args.plot:
        plot_results(all_results, modes)

    # --- Markdown report ---
    if args.report:
        generate_report(all_results, modes, args)

    print()
    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(all_results, modes):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # Group by mode
    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 6), squeeze=False)
    fig.suptitle("CuteDSL vs Triton (FLA) -- Lightning Attention Prefill", fontsize=14, fontweight="bold")

    for col, mode in enumerate(modes):
        ax = axes[0, col]
        mode_r = [r for r in all_results if r["mode"] == mode and not np.isnan(r["speedup"])]
        if not mode_r:
            ax.set_title(f"{mode} (no data)")
            continue

        labels = [f"B{r['B']}T{r['T']}H{r['H']}" for r in mode_r]
        tri_ms = [r["triton_ms"] for r in mode_r]
        dsl_ms = [r["cutedsl_ms"] for r in mode_r]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, tri_ms, w, label="Triton", color="steelblue")
        ax.bar(x + w / 2, dsl_ms, w, label="CuteDSL", color="orange")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Mode: {mode}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add speedup labels
        for i, r in enumerate(mode_r):
            ax.text(i, max(tri_ms[i], dsl_ms[i]) * 1.02,
                    f"{r['speedup']:.1f}x", ha="center", va="bottom", fontsize=7, color="green")

    plt.tight_layout()
    out = "benchmark_results.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def generate_report(all_results, modes, args):
    from datetime import datetime
    path = "benchmark_report.md"
    with open(path, "w") as f:
        f.write("# Lightning Attention Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Batch sizes: {args.batch_sizes}\n")
        f.write(f"- Seq lengths: {args.seq_lens}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n")
        f.write(f"- Layer: {args.layer_idx}/{args.num_layers}\n")
        f.write(f"- Modes: {modes}\n")
        f.write(f"- Warmup/Iters: {args.warmup}/{args.iterations}\n\n")

        for mode in modes:
            mode_r = [r for r in all_results if r["mode"] == mode]
            f.write(f"## Mode: {mode}\n\n")

            if mode == "h0_ht":
                f.write("| Config | Triton(ms) | CuteDSL(ms) | Speedup | O_maxdiff | O_rel% | Ht_maxdiff | Ht_rel% | Status |\n")
                f.write("|--------|-----------|------------|---------|-----------|--------|------------|---------|--------|\n")
            else:
                f.write("| Config | Triton(ms) | CuteDSL(ms) | Speedup | O_maxdiff | O_rel% | Status |\n")
                f.write("|--------|-----------|------------|---------|-----------|--------|--------|\n")

            for r in mode_r:
                cfg = f"B={r['B']},T={r['T']},H={r['H']}"
                sp = f"{r['speedup']:.2f}x" if not np.isnan(r['speedup']) else "-"
                status = "OK"
                if r["triton_error"]:
                    status = "Triton Failed"
                elif r["cutedsl_error"]:
                    status = "CuteDSL Failed"

                tri = f"{r['triton_ms']:.3f}" if not np.isnan(r['triton_ms']) else "-"
                dsl = f"{r['cutedsl_ms']:.3f}" if not np.isnan(r['cutedsl_ms']) else "-"
                omd = f"{r['o_max_diff']:.4f}" if not np.isnan(r['o_max_diff']) else "-"
                orel = f"{r['o_rel_error']*100:.2f}%" if not np.isnan(r['o_rel_error']) else "-"

                if mode == "h0_ht":
                    htmd = f"{r['ht_max_diff']:.4f}" if not np.isnan(r['ht_max_diff']) else "-"
                    htrel = f"{r['ht_rel_error']*100:.2f}%" if not np.isnan(r['ht_rel_error']) else "-"
                    f.write(f"| {cfg} | {tri} | {dsl} | {sp} | {omd} | {orel} | {htmd} | {htrel} | {status} |\n")
                else:
                    f.write(f"| {cfg} | {tri} | {dsl} | {sp} | {omd} | {orel} | {status} |\n")
            f.write("\n")

        # Summary
        f.write("## Summary\n\n")
        for mode in modes:
            mode_r = [r for r in all_results if r["mode"] == mode and not np.isnan(r["speedup"])]
            if not mode_r:
                continue
            speedups = [r["speedup"] for r in mode_r]
            f.write(f"- **{mode}**: avg speedup {np.mean(speedups):.2f}x "
                    f"(min {np.min(speedups):.2f}x, max {np.max(speedups):.2f}x) "
                    f"over {len(mode_r)} configs\n")

        f.write("\n---\n*Generated by bench_lightning_attn.py*\n")

    print(f"Report saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Lightning Attention prefill scenarios")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8],
                    help="Batch sizes to test")
    p.add_argument("--seq-lens", type=int, nargs="+",
                    default=[256, 1024, 4096, 8192, 32768],
                    help="Sequence lengths to test")
    p.add_argument("--num-heads", type=int, nargs="+", default=[32, 64],
                    help="Number of heads to test")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layer-idx", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--modes", type=str, nargs="+",
                    default=["no_state", "h0_ht"],
                    choices=["no_state", "h0_ht"],
                    help="State modes to benchmark")
    p.add_argument("--plot", action="store_true", help="Save bar chart PNG")
    p.add_argument("--report", action="store_true", help="Generate markdown report")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark_suite(args)
