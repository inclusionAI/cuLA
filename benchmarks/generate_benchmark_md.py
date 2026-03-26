#!/usr/bin/env python3
"""
generate_benchmark_md.py — Run benchmarks and generate BENCHMARK.md

Runs cuLA benchmarks with representative configs and writes results
as formatted markdown tables.

Usage:
  python benchmarks/generate_benchmark_md.py

  # Generate with specific GPU id
  CUDA_VISIBLE_DEVICES=3 python benchmarks/generate_benchmark_md.py
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import numpy as np
from benchmarks.bench_kda import (
    bench_fixed as kda_bench_fixed,
    bench_varlen as kda_bench_varlen,
    gen_varlen_seqs,
    H as KDA_H, D as KDA_D,
)
from benchmarks.bench_lightning_attn import (
    benchmark_standard_config as la_bench_standard,
    benchmark_varlen_config as la_bench_varlen,
    gen_uniform, gen_skewed, gen_random,
    D_DEFAULT as LA_D, _valid,
)

BENCHMARK_MD = ROOT / "BENCHMARK.md"
README_MD = ROOT / "README.md"

# ============================================================
# Reduced configs (representative subset)
# ============================================================

# KDA fixed-length: B x T combos
KDA_FIXED_CONFIGS = [
    (1, 1024), (1, 4096), (1, 8192), (1, 16384),
    (2, 1024), (2, 4096), (2, 8192), (2, 16384),
]

# KDA varlen: representative set
KDA_VARLEN_CONFIGS = [
    # Single sequence
    ([4096], 4096),
    ([8192], 8192),
    ([16384], 16384),
    # Uniform (~20 seqs)
    (gen_varlen_seqs(4096, 20, seed=1), 4096),
    (gen_varlen_seqs(8192, 20, seed=2), 8192),
    (gen_varlen_seqs(16384, 20, seed=4), 16384),
    # Skewed (1 long + many short)
    ([4096 - 19 * 64] + [64] * 19, 4096),
    ([8192 - 19 * 64] + [64] * 19, 8192),
    ([16384 - 19 * 64] + [64] * 19, 16384),
    # Tail-heavy (many short + 1 long)
    ([64] * 19 + [4096 - 19 * 64], 4096),
    ([64] * 19 + [8192 - 19 * 64], 8192),
    ([64] * 19 + [16384 - 19 * 64], 16384),
]

# Lightning Attention standard: reduced set
LA_STANDARD_BATCH_SIZES = [1, 2]
LA_STANDARD_SEQ_LENS = [1024, 4096, 8192, 16384]
LA_H = 64
LA_WARMUP = 5
LA_ITERS = 20

# Lightning Attention varlen: reduced set
LA_VARLEN_N_VALUES = [5, 10, 16, 20, 25]
LA_VARLEN_T_VALUES = [1024, 8192, 16384, 32768]
LA_VARLEN_DISTS = ["uniform", "skewed", "random"]


# ============================================================
# Helpers
# ============================================================

def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "Unknown GPU"


def get_env_info():
    cuda_version = "Unknown"
    try:
        cuda_version = torch.version.cuda or "Unknown"
    except Exception:
        pass
    torch_version = torch.__version__
    return {
        "gpu": get_gpu_name(),
        "cuda": cuda_version,
        "torch": torch_version,
    }


def classify_varlen(seq_lens, total_len):
    """Classify a varlen config into a human-readable tag."""
    n = len(seq_lens)
    if n == 1:
        return f"1seq, T={total_len}"

    min_l, max_l = min(seq_lens), max(seq_lens)
    ratio = max_l / min_l if min_l > 0 else float('inf')

    if ratio < 3:
        dist = "uniform"
    elif seq_lens[0] == max_l:
        dist = "skewed"
    elif seq_lens[-1] == max_l:
        dist = "tail-heavy"
    else:
        dist = "mixed"

    return f"{n}seqs, T={total_len}, {dist}"


# ============================================================
# Run benchmarks
# ============================================================

def run_kda_benchmarks():
    print("\n>>> Running KDA benchmarks...")
    fixed_results = kda_bench_fixed(KDA_FIXED_CONFIGS)
    varlen_results = kda_bench_varlen(KDA_VARLEN_CONFIGS)
    return fixed_results, varlen_results


def run_lightning_attn_benchmarks():
    print("\n>>> Running Lightning Attention benchmarks...")

    # Standard prefill (no_state only — h0_ht is similar, skip for brevity)
    standard_results = []
    for B in LA_STANDARD_BATCH_SIZES:
        for T in LA_STANDARD_SEQ_LENS:
            r = la_bench_standard(B, T, LA_H, LA_D, 12, 24, "no_state", LA_WARMUP, LA_ITERS)
            standard_results.append(r)
            print(f"  LA prefill B={B} T={T}: {r.get('speedup', 0):.2f}x")

    # Varlen
    varlen_results = []
    for N in LA_VARLEN_N_VALUES:
        for T_total in LA_VARLEN_T_VALUES:
            if T_total // N < 1:
                continue
            for dist_name in LA_VARLEN_DISTS:
                if dist_name == "uniform":
                    seq_lens = gen_uniform(N, T_total)
                elif dist_name == "skewed":
                    seq_lens = gen_skewed(N, T_total)
                else:
                    seq_lens = gen_random(N, T_total)
                r = la_bench_varlen(N, seq_lens, LA_H, LA_D, LA_WARMUP, LA_ITERS, dist=dist_name)
                varlen_results.append(r)
                sp = r.get("p_vs_fla_vl_speedup", float("nan"))
                sp_str = f"{sp:.2f}x" if _valid(sp) else "N/A"
                print(f"  LA varlen N={N} T={T_total} {dist_name}: {sp_str}")

    return standard_results, varlen_results


# ============================================================
# Format markdown
# ============================================================

def format_benchmark_md(env, kda_fixed, kda_varlen, la_standard, la_varlen):
    lines = []
    w = lines.append

    w("# Benchmark Results\n")
    w(f"> Auto-generated by `benchmarks/generate_benchmark_md.py` on {datetime.now().strftime('%Y-%m-%d')}.\n")
    w(f"> **GPU:** {env['gpu']}  |  **CUDA:** {env['cuda']}  |  **PyTorch:** {env['torch']}\n")
    w(f"> FLA baseline: [flash-linear-attention v0.4.2](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.4.2)\n")
    w("")

    # -------------------------------------------------------------------
    # KDA
    # -------------------------------------------------------------------
    w("\n## KDA (Kimi Delta Attention)\n")

    # Fixed-length
    w(f"### Fixed-Length (H={KDA_H}, D={KDA_D}, bf16)\n")
    w("| B | T | FLA Triton (ms) | cuLA (ms) | Speedup |")
    w("|---|---|-----------------|-----------|---------|")
    for r in kda_fixed:
        sp = f"**{r['speedup']:.2f}x**"
        w(f"| {r['B']} | {r['T']} | {r['ms_fla']:.3f} | {r['ms_cula']:.3f} | {sp} |")

    # Varlen
    w(f"\n### Variable-Length (H={KDA_H}, D={KDA_D}, bf16)\n")
    w("| Config | FLA Triton (ms) | cuLA (ms) | Speedup |")
    w("|--------|-----------------|-----------|---------|")
    for r in kda_varlen:
        tag = r.get('tag', 'unknown')
        sp = f"**{r['speedup']:.2f}x**"
        w(f"| {tag} | {r['ms_fla']:.3f} | {r['ms_cula']:.3f} | {sp} |")

    w("\nTo reproduce:\n")
    w("```bash")
    w("python benchmarks/bench_kda.py --mode both")
    w("```\n")

    # -------------------------------------------------------------------
    # Lightning Attention
    # -------------------------------------------------------------------
    w("## Lightning Attention\n")

    # Standard prefill
    w(f"### Prefill (H={LA_H}, D={LA_D}, bf16)\n")
    w("| B | T | FLA Triton (ms) | cuLA (ms) | Speedup |")
    w("|---|---|-----------------|-----------|---------|")
    for r in la_standard:
        if not _valid(r.get("speedup", float("nan"))):
            continue
        sp = f"**{r['speedup']:.2f}x**"
        w(f"| {r['B']} | {r['T']} | {r['fla_ms']:.3f} | {r['cutedsl_ms']:.3f} | {sp} |")

    # Varlen summary (select representative rows to avoid verbosity)
    w(f"\n### Variable-Length (H={LA_H}, D={LA_D}, bf16)\n")
    w("Persistent CuTe DSL kernel vs FLA Triton varlen.\n")

    # Pick a representative subset: one per (N, T) — use "uniform" dist only for the table
    # but show summary stats across all dists
    representative = [r for r in la_varlen if r.get("dist") == "uniform"]
    w("| N (seqs) | T | cuLA (ms) | FLA Triton (ms) | Speedup |")
    w("|----------|---|-----------|-----------------|---------|")
    for r in representative:
        p_ms = r.get("persistent_ms", float("nan"))
        fla_ms = r.get("fla_varlen_ms", float("nan"))
        sp = r.get("p_vs_fla_vl_speedup", float("nan"))
        if not _valid(p_ms) or not _valid(fla_ms):
            continue
        sp_str = f"**{sp:.2f}x**" if _valid(sp) else "-"
        w(f"| {r['B']} | {r['T']} | {p_ms:.3f} | {fla_ms:.3f} | {sp_str} |")

    # Summary stats across all varlen configs
    all_sp = [r["p_vs_fla_vl_speedup"] for r in la_varlen
              if _valid(r.get("p_vs_fla_vl_speedup", float("nan")))]
    if all_sp:
        w(f"\nSummary ({len(all_sp)} configs across uniform/skewed/random): "
          f"**avg={np.mean(all_sp):.2f}x**, min={np.min(all_sp):.2f}x, max={np.max(all_sp):.2f}x.\n")

    w("To reproduce:\n")
    w("```bash")
    w("python benchmarks/bench_lightning_attn.py --modes no_state varlen")
    w("```\n")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def format_readme_section(env, kda_fixed, kda_varlen, la_standard, la_varlen):
    """Format a concise benchmark section for embedding in README.md.

    Skips the top-level heading (README already has ## Benchmarks) and
    uses ### subheadings so it nests properly.
    """
    lines = []
    w = lines.append

    w(f"All benchmarks run on a single **{env['gpu']}** GPU with "
      f"**CUDA {env['cuda']}**, **PyTorch {env['torch']}**.")
    w(f"FLA baseline: [flash-linear-attention v0.4.2]"
      f"(https://github.com/fla-org/flash-linear-attention/releases/tag/v0.4.2).\n")

    # --- KDA Fixed ---
    w(f"### KDA — Fixed-Length (H={KDA_H}, D={KDA_D}, bf16)\n")
    w("| B | T | FLA Triton (ms) | cuLA (ms) | Speedup |")
    w("|---|---|-----------------|-----------|---------|")
    for r in kda_fixed:
        sp = f"**{r['speedup']:.2f}x**"
        w(f"| {r['B']} | {r['T']} | {r['ms_fla']:.3f} | {r['ms_cula']:.3f} | {sp} |")

    # --- KDA Varlen ---
    w(f"\n### KDA — Variable-Length (H={KDA_H}, D={KDA_D}, bf16)\n")
    w("| Config | FLA Triton (ms) | cuLA (ms) | Speedup |")
    w("|--------|-----------------|-----------|---------|")
    for r in kda_varlen:
        tag = r.get('tag', 'unknown')
        sp = f"**{r['speedup']:.2f}x**"
        w(f"| {tag} | {r['ms_fla']:.3f} | {r['ms_cula']:.3f} | {sp} |")

    # --- Lightning Attention Prefill ---
    w(f"\n### Lightning Attention — Prefill (H={LA_H}, D={LA_D}, bf16)\n")
    w("| B | T | FLA Triton (ms) | cuLA (ms) | Speedup |")
    w("|---|---|-----------------|-----------|---------|")
    for r in la_standard:
        if not _valid(r.get('speedup', float('nan'))):
            continue
        sp = f"**{r['speedup']:.2f}x**"
        w(f"| {r['B']} | {r['T']} | {r['fla_ms']:.3f} | {r['cutedsl_ms']:.3f} | {sp} |")

    # --- Lightning Attention Varlen ---
    w(f"\n### Lightning Attention — Variable-Length (H={LA_H}, D={LA_D}, bf16)\n")
    w("Persistent CuTe DSL kernel vs FLA Triton varlen.\n")
    representative = [r for r in la_varlen if r.get('dist') == 'uniform']
    w("| N (seqs) | T | cuLA (ms) | FLA Triton (ms) | Speedup |")
    w("|----------|---|-----------|-----------------|---------|")
    for r in representative:
        p_ms = r.get('persistent_ms', float('nan'))
        fla_ms = r.get('fla_varlen_ms', float('nan'))
        sp = r.get('p_vs_fla_vl_speedup', float('nan'))
        if not _valid(p_ms) or not _valid(fla_ms):
            continue
        sp_str = f"**{sp:.2f}x**" if _valid(sp) else "-"
        w(f"| {r['B']} | {r['T']} | {p_ms:.3f} | {fla_ms:.3f} | {sp_str} |")

    all_sp = [r['p_vs_fla_vl_speedup'] for r in la_varlen
              if _valid(r.get('p_vs_fla_vl_speedup', float('nan')))]
    if all_sp:
        w(f"\nSummary ({len(all_sp)} configs across uniform/skewed/random): "
          f"**avg={np.mean(all_sp):.2f}x**, min={np.min(all_sp):.2f}x, max={np.max(all_sp):.2f}x.")

    return "\n".join(lines)


def inject_into_readme(readme_section):
    """Replace content between <!-- BENCHMARK_START --> and <!-- BENCHMARK_END --> in README.md."""
    START_MARKER = "<!-- BENCHMARK_START -->"
    END_MARKER = "<!-- BENCHMARK_END -->"

    if not README_MD.exists():
        print(f"WARNING: {README_MD} not found, skipping README injection.")
        return

    text = README_MD.read_text()
    start_idx = text.find(START_MARKER)
    end_idx = text.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        print(f"WARNING: Benchmark markers not found in README.md, skipping injection.")
        return

    new_text = (
        text[:start_idx + len(START_MARKER)]
        + "\n"
        + readme_section
        + "\n"
        + text[end_idx:]
    )
    README_MD.write_text(new_text)
    print(f"Injected benchmarks into {README_MD}")


def main():
    parser = argparse.ArgumentParser(description="Generate BENCHMARK.md")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to a JSON cache file. If exists, skip benchmarks and use cached results.")
    parser.add_argument("--save-cache", type=str, default=None,
                        help="Save benchmark results to JSON for future --cache use.")
    args = parser.parse_args()

    env = get_env_info()

    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached results from {args.cache}")
        with open(args.cache) as f:
            data = json.load(f)
        kda_fixed = data["kda_fixed"]
        kda_varlen = data["kda_varlen"]
        la_standard = data["la_standard"]
        la_varlen = data["la_varlen"]
    else:
        kda_fixed, kda_varlen = run_kda_benchmarks()
        la_standard, la_varlen = run_lightning_attn_benchmarks()

        if args.save_cache:
            cache_path = Path(args.save_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert results to JSON-serializable form
            def sanitize(results):
                out = []
                for r in results:
                    clean = {}
                    for k, v in r.items():
                        if isinstance(v, (list, tuple)):
                            clean[k] = list(v)
                        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                            clean[k] = None
                        else:
                            clean[k] = v
                    out.append(clean)
                return out

            with open(cache_path, "w") as f:
                json.dump({
                    "kda_fixed": sanitize(kda_fixed),
                    "kda_varlen": sanitize(kda_varlen),
                    "la_standard": sanitize(la_standard),
                    "la_varlen": sanitize(la_varlen),
                }, f, indent=2)
            print(f"Cached results to {cache_path}")

    md = format_benchmark_md(env, kda_fixed, kda_varlen, la_standard, la_varlen)

    BENCHMARK_MD.write_text(md)
    print(f"\nWrote {BENCHMARK_MD}")

    # Also inject concise version into README.md between markers
    readme_section = format_readme_section(env, kda_fixed, kda_varlen, la_standard, la_varlen)
    inject_into_readme(readme_section)


if __name__ == "__main__":
    main()
