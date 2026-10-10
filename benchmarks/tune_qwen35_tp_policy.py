#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tune Qwen3.5 TP-local kernel policies.

This is a configuration-driven tuner. It benchmarks only policies that are
compiled into the current extension and records unsupported candidates in the
result file. The initial compiled policy is the decode traits currently used by
the CUDA/CuTe kernels:

  layout_vec=4, kda_threads=128, kda_tile_v=16, kda_tile_k=16, heads_per_cta=1

When more C++ policy specializations are added, extend `compiled_policy_key`
and the kernel dispatch path; this script can then sweep them without changing
the output format.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class DecodePolicy:
    name: str
    layout_vec: int
    kda_threads: int
    kda_tile_v: int
    kda_tile_k: int
    heads_per_cta: int = 1

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (self.layout_vec, self.kda_threads, self.kda_tile_v, self.kda_tile_k, self.heads_per_cta)


CURRENT_DECODE_POLICY = DecodePolicy(
    name="current",
    layout_vec=4,
    kda_threads=128,
    kda_tile_v=16,
    kda_tile_k=16,
    heads_per_cta=1,
)


def decode_benchmarks():
    from benchmarks import bench_qwen35_decode

    return bench_qwen35_decode


def compiled_policy_key(policy: DecodePolicy) -> str | None:
    """Return the compiled backend selector for a policy, or None if absent."""
    if policy.key == CURRENT_DECODE_POLICY.key:
        return "current"
    return None


def _list_from_json(data: dict[str, Any], key: str, default: list[int]) -> list[int]:
    value = data.get(key, default)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return [int(item) for item in value]


def load_decode_policies(path: pathlib.Path | None) -> list[DecodePolicy]:
    if path is None:
        return [CURRENT_DECODE_POLICY]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        policies = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Policy entry {idx} must be an object")
            policies.append(
                DecodePolicy(
                    name=str(item.get("name", f"policy_{idx}")),
                    layout_vec=int(item["layout_vec"]),
                    kda_threads=int(item["kda_threads"]),
                    kda_tile_v=int(item["kda_tile_v"]),
                    kda_tile_k=int(item["kda_tile_k"]),
                    heads_per_cta=int(item.get("heads_per_cta", 1)),
                )
            )
        return policies

    if not isinstance(data, dict):
        raise ValueError("Policy grid must be a JSON object or list")
    if data.get("mode", "decode") != "decode":
        raise ValueError("Only decode policy grids are supported by this tuner")

    policies = []
    for idx, combo in enumerate(
        itertools.product(
            _list_from_json(data, "layout_vec", [CURRENT_DECODE_POLICY.layout_vec]),
            _list_from_json(data, "kda_threads", [CURRENT_DECODE_POLICY.kda_threads]),
            _list_from_json(data, "kda_tile_v", [CURRENT_DECODE_POLICY.kda_tile_v]),
            _list_from_json(data, "kda_tile_k", [CURRENT_DECODE_POLICY.kda_tile_k]),
            _list_from_json(data, "heads_per_cta", [CURRENT_DECODE_POLICY.heads_per_cta]),
        )
    ):
        layout_vec, kda_threads, kda_tile_v, kda_tile_k, heads_per_cta = combo
        policies.append(
            DecodePolicy(
                name=f"p{idx}_lv{layout_vec}_th{kda_threads}_tv{kda_tile_v}_tk{kda_tile_k}_h{heads_per_cta}",
                layout_vec=layout_vec,
                kda_threads=kda_threads,
                kda_tile_v=kda_tile_v,
                kda_tile_k=kda_tile_k,
                heads_per_cta=heads_per_cta,
            )
        )
    return policies


def write_example_grid(path: pathlib.Path) -> None:
    example = {
        "mode": "decode",
        "layout_vec": [4, 8],
        "kda_threads": [64, 128, 256],
        "kda_tile_v": [8, 16, 32],
        "kda_tile_k": [8, 16, 32],
        "heads_per_cta": [1, 2, 4],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(example, f, indent=2, sort_keys=True)


def bucket_name(tokens: int) -> str:
    if tokens <= 4:
        return "tokens<=4"
    if tokens <= 16:
        return "tokens<=16"
    if tokens <= 64:
        return "tokens<=64"
    return "tokens>64"


def run_decode_policy(
    *,
    scope: str,
    tokens: int,
    tp_size: int,
    warmup: int,
    rep: int,
    seed: int,
    policy: DecodePolicy,
) -> dict[str, Any]:
    decode_bench = decode_benchmarks()
    config = decode_bench.local_config_from_tp_size(tp_size)
    compiled_key = compiled_policy_key(policy)
    row: dict[str, Any] = {
        "mode": "decode",
        "scope": scope,
        "tokens": tokens,
        "token_bucket": bucket_name(tokens),
        "tp_size": tp_size,
        "local_k_heads": config.num_k_heads,
        "local_v_heads": config.num_v_heads,
        "conv_dim": config.conv_dim,
        "policy": policy.name,
        "compiled_policy": compiled_key,
        **asdict(policy),
    }
    if compiled_key is None:
        row.update({"status": "unsupported", "ms": None, "us_per_token": None})
        return row

    device = decode_bench.accelerator_device()
    if scope == "core":
        ms = decode_bench.bench_native_core(tokens, device, warmup, rep, seed, config)
    elif scope == "fused":
        ms = decode_bench.bench_fused_layout_kda(tokens, device, warmup, rep, seed, config)
    elif scope == "full":
        ms = decode_bench.bench_full(tokens, device, warmup, rep, seed, config)
    else:
        raise ValueError(f"Unsupported decode scope={scope}")

    row.update({"status": "ok", "ms": ms, "us_per_token": ms * 1000.0 / tokens})
    return row


def choose_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["mode"], row["scope"], row["tp_size"], row["local_v_heads"], row["token_bucket"])
        groups.setdefault(key, []).append(row)

    best_rows = []
    for key, candidates in sorted(groups.items()):
        best = min(candidates, key=lambda row: float(row["ms"]))
        mode, scope, tp_size, local_v_heads, token_bucket = key
        best_rows.append(
            {
                "mode": mode,
                "scope": scope,
                "tp_size": tp_size,
                "local_v_heads": local_v_heads,
                "token_bucket": token_bucket,
                "policy": best["policy"],
                "compiled_policy": best["compiled_policy"],
                "ms": best["ms"],
                "us_per_token": best["us_per_token"],
                "layout_vec": best["layout_vec"],
                "kda_threads": best["kda_threads"],
                "kda_tile_v": best["kda_tile_v"],
                "kda_tile_k": best["kda_tile_k"],
                "heads_per_cta": best["heads_per_cta"],
            }
        )
    return best_rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "scope",
        "tokens",
        "token_bucket",
        "tp_size",
        "local_k_heads",
        "local_v_heads",
        "conv_dim",
        "policy",
        "compiled_policy",
        "status",
        "ms",
        "us_per_token",
        "layout_vec",
        "kda_threads",
        "kda_tile_v",
        "kda_tile_k",
        "heads_per_cta",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Qwen3.5 TP-local kernel policies.")
    parser.add_argument("--mode", choices=["decode"], default="decode")
    parser.add_argument("--scope", choices=["core", "fused", "full", "all"], default="fused")
    parser.add_argument("--tp-sizes", nargs="+", type=int, choices=[1, 2, 4, 8], default=[1, 2, 4, 8])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-grid", type=pathlib.Path, default=None)
    parser.add_argument("--write-example-grid", type=pathlib.Path, default=None)
    parser.add_argument("--output-json", type=pathlib.Path, default=pathlib.Path("tmp/qwen35_tp_policy_tune.json"))
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument("--fail-on-unsupported", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_example_grid is not None:
        write_example_grid(args.write_example_grid)
        print(f"wrote example policy grid: {args.write_example_grid}")
        return 0

    policies = load_decode_policies(args.policy_grid)
    scopes = ["core", "fused", "full"] if args.scope == "all" else [args.scope]
    decode_bench = decode_benchmarks()
    device = decode_bench.accelerator_device()
    device_name = decode_bench.accelerator_name(device)

    print(f"Qwen3.5 TP policy tuner: mode={args.mode} device={device_name}")
    print(f"tp_sizes={args.tp_sizes} tokens={args.tokens} scopes={scopes}")
    print(f"policies={len(policies)} compiled={sum(compiled_policy_key(p) is not None for p in policies)}")

    rows: list[dict[str, Any]] = []
    for policy in policies:
        compiled_key = compiled_policy_key(policy)
        if compiled_key is None:
            print(f"skip unsupported policy={policy.name} {asdict(policy)}")
        for scope in scopes:
            for tp_size in args.tp_sizes:
                for tokens in args.tokens:
                    row = run_decode_policy(
                        scope=scope,
                        tokens=tokens,
                        tp_size=tp_size,
                        warmup=args.warmup,
                        rep=args.rep,
                        seed=args.seed,
                        policy=policy,
                    )
                    rows.append(row)
                    if row["status"] == "ok":
                        print(
                            f"{scope:>5} tp={tp_size} hv={row['local_v_heads']:>2} tokens={tokens:>4} "
                            f"policy={policy.name} ms={row['ms']:.4f} us/tok={row['us_per_token']:.2f}"
                        )

    unsupported = [row for row in rows if row["status"] == "unsupported"]
    if unsupported and args.fail_on_unsupported:
        raise RuntimeError(f"{len(unsupported)} policy/shape rows are unsupported by the compiled extension")

    best_rows = choose_best(rows)
    result = {
        "device": device_name,
        "mode": args.mode,
        "warmup": args.warmup,
        "rep": args.rep,
        "seed": args.seed,
        "rows": rows,
        "best": best_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"wrote {args.output_json}")

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"wrote {args.csv}")

    if best_rows:
        print("best policies:")
        for row in best_rows:
            print(
                f"  {row['scope']:>5} tp={row['tp_size']} hv={row['local_v_heads']:>2} "
                f"{row['token_bucket']}: {row['policy']} {row['ms']:.4f} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
