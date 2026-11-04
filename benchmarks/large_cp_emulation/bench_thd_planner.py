# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Report modeled CP load balance over representative packed THD batches."""

from __future__ import annotations

from cula.distributed.thd_planner import plan_thd_layout

CASES = {
    "all-short": [1024] * 1024,
    "single-long": [1 << 20],
    "short+two-long": [1024] * 20 + [514048] * 2,
    "long-tail": [1024] * 256 + [786432],
    "multi-long": [4096] * 32 + [131072, 262144, 524288],
    "power-law": [1024, 4096, 16384, 65536, 131072, 307200, 523264],
}


def _boundaries(lengths: list[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def main() -> None:
    print("| THD distribution | selected | token max/avg | causal max/avg | score | contiguous score | remote boundaries |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, lengths in CASES.items():
        plan = plan_thd_layout(_boundaries(lengths), world_size=4)
        baseline = dict(plan.evaluated_scores)["contiguous"]
        print(
            f"| {name} | {plan.policy} | {plan.metrics.token_imbalance:.3f} | "
            f"{plan.metrics.causal_imbalance:.3f} | {plan.metrics.score:.3f} | "
            f"{baseline:.3f} | {sum(plan.metrics.boundary_loads) // 2} |"
        )


if __name__ == "__main__":
    main()
