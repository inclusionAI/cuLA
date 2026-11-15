# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Real-CP layout comparison with full output/gradient baselines in CPU RAM."""

import argparse
import gc
import sys
from pathlib import Path

from benchmarks.thd_1m_cases import DEFAULT_SEED, counted_cases, heldout_cases, statistics_cases


def performance_cases(suite="baseline", seed=DEFAULT_SEED):
    if suite == "validation":
        if seed != DEFAULT_SEED:
            raise ValueError("the validation suite fixes its seeds for reproducibility")
        return [
            *performance_cases("baseline"),
            *performance_cases("counted"),
            *(
                (f"{case.name}-seed{heldout_seed}", list(case.lengths), False)
                for heldout_seed in (20261114, 20261115, 20261116)
                for case in heldout_cases(heldout_seed)
            ),
        ]
    cases = []
    if suite in ("baseline", "all"):
        cases += [
            ("1m-single", [1048576], False),
            ("1m-uniform", [16384] * 64, False),
            ("1m-near-uniform", [15360, 17408] * 32, False),
            ("1m-20s2l", [1024] * 20 + [514048] * 2, False),
            ("1m-power-law", [1024, 4096, 16384, 65536, 131072, 307200, 523264], False),
        ]
    if suite in ("statistics", "all"):
        cases += [(case.name, list(case.lengths), False) for case in statistics_cases(seed)]
    if suite in ("counted", "all"):
        cases += [(case.name, list(case.lengths), False) for case in counted_cases(seed)]
    if suite in ("heldout", "all"):
        cases += [(case.name, list(case.lengths), False) for case in heldout_cases(seed)]
    if not cases:
        raise ValueError(f"unknown suite: {suite}")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument(
        "--suite", choices=("baseline", "statistics", "counted", "heldout", "all", "validation"), default="baseline"
    )
    parser.add_argument("--cases", nargs="+", help="Exact performance case names from the selected suite")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Synthetic THD generation seed")
    parser.add_argument("--list-cases", action="store_true", help="List shapes without initializing CUDA/distributed")
    parser.add_argument("--redistribution-scope", choices=("selection", "queries"), default="selection")
    parser.add_argument("--skip-checks", action="store_true", help="Skip small FP32 checks after a separately validated run")
    parser.add_argument(
        "--layouts", nargs="+", choices=("contiguous", "zigzag", "planner"), default=["contiguous", "zigzag", "planner"]
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--heads", type=int, default=64, help="Attention heads; native SM100 D576 requires 64")
    parser.add_argument("--planner-cost-model", help="JSON calibrated selection-routing coefficients")
    parser.add_argument("--memory-stages", action="store_true")
    parser.add_argument("--planner-strategy", choices=("balanced", "selection"), default="balanced")
    args = parser.parse_args()
    selected = performance_cases(args.suite, args.seed)
    if args.cases:
        unknown = set(args.cases) - {name for name, _, _ in selected}
        if unknown:
            parser.error(f"unknown cases for suite {args.suite}: {sorted(unknown)}")
        selected = [case for case in selected if case[0] in args.cases]
    if args.checks_only and args.cases:
        parser.error("--cases cannot be combined with --checks-only")
    if args.checks_only and args.skip_checks:
        parser.error("--checks-only and --skip-checks are mutually exclusive")
    if args.layouts[0] != "contiguous" or len(set(args.layouts)) != len(args.layouts):
        parser.error("layouts must start with the contiguous numerical baseline and contain no duplicates")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    if args.list_cases:
        for name, lengths, _ in selected:
            print(name, "documents", len(lengths), "tokens", sum(lengths))
        return

    import torch
    import torch.distributed as dist

    from benchmarks.bench_full_csa_cp import main as run_case

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("check-single", [32768], True),
        ("check-mixed", [4] * 20 + [2008] * 2, True),
        ("check-uniform", [256] * 16, True),
        ("check-dense", [4] * 256 + [32] * 96, True),
    ]
    if args.skip_checks:
        cases = []
    if not args.checks_only:
        cases += selected
    for case_index, (name, lengths, check) in enumerate(cases):
        baseline = None
        # Keep the numerical baseline first, alternate the remaining policies
        # to avoid always giving one policy the same thermal/cache order.
        layouts = args.layouts if case_index % 2 == 0 else [args.layouts[0], *reversed(args.layouts[1:])]
        for layout in layouts:
            print("CASE", name, layout, flush=True)
            sys.argv = [
                "bench_full_csa_cp.py",
                "--sequence-lengths",
                *map(str, lengths),
                "--heads",
                str(args.heads),
                "--top-k",
                "512",
                "--with-kda",
                "--layout",
                layout,
                "--redistribution-scope",
                args.redistribution_scope,
                "--planner-strategy",
                args.planner_strategy,
                "--pool-backward-order",
                "per-loss",
                "--output",
                str(args.output_dir / f"layout-{name}-{layout}.json"),
                "--warmups",
                "0" if check else str(args.warmups),
                "--repeats",
                "1" if check else str(args.repeats),
                "--candidate-mib",
                "64" if check else "0",
            ]
            if check:
                sys.argv.append("--check")
            if args.planner_cost_model:
                sys.argv += ["--planner-cost-model", args.planner_cost_model]
            if args.memory_stages:
                sys.argv.append("--memory-stages")
            saved = run_case(keep_group=True, baseline=baseline, save_baseline=layout == "contiguous")
            if saved is not None:
                baseline = saved
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier(device_ids=[torch.cuda.current_device()])
        del baseline, saved
        gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
