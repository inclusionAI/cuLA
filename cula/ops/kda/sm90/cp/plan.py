# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SM90 intracard-CP segment planner.

Trivial plan means serial path. AUTO engages only when splitting shortens the
critical path enough to pay the pre_scan + segment-K2 re-run; constants below
are H100-fitted ratios of like chains (no per-SKU recalibration).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from cula.ops.kda.cp_mode import CPMode, NotSplittableError
from cula.utils import get_device_sm_count

CHUNK = 16

# Hand-crafted splits (plan_manual) stay non-degenerate.
MIN_SEG_TILES = 4

# Auto segment-length floor; H100-tuned.
AUTO_MIN_SEG_TILES = int(os.environ.get("CULA_KDA_CP_AUTO_MIN_SEG_TILES", "32"))

# CP re-run cost (pre_scan + segment-K2) vs serial K2, per tile; H100-tuned.
RERUN_RATIO = float(os.environ.get("CULA_KDA_CP_RERUN_RATIO", "3"))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class CPPlan:
    """Tile segmentation: segment i covers [seg_cu[i], seg_cu[i+1]);
    per_seq[s] = (first_segment, n_segments). Trivial => serial."""

    seq_tiles: tuple[int, ...]
    seg_cu: tuple[int, ...]
    per_seq: tuple[tuple[int, int], ...]
    reason: str = ""

    @classmethod
    def serial(cls, seq_tiles=(), reason: str = "") -> CPPlan:
        return cls(tuple(seq_tiles), (0,), (), reason)

    @property
    def trivial(self) -> bool:
        return self.n_seg_total <= self.n_seqs

    @property
    def n_seqs(self) -> int:
        return len(self.seq_tiles)

    @property
    def n_seg_total(self) -> int:
        return len(self.seg_cu) - 1

    @property
    def total_tiles(self) -> int:
        return self.seg_cu[-1]

    @property
    def seg_tiles(self) -> tuple[int, ...]:
        return tuple(self.seg_cu[i + 1] - self.seg_cu[i] for i in range(self.n_seg_total))

    @property
    def max_seg_tiles(self) -> int:
        return max(self.seg_tiles, default=0)


def _materialize(seq_tiles: list[int], n_segs: list[int]) -> CPPlan:
    seg_cu = [0]
    per_seq = []
    for tiles, n in zip(seq_tiles, n_segs):
        first = len(seg_cu) - 1
        base, rem = divmod(tiles, n)
        for i in range(n):
            seg_cu.append(seg_cu[-1] + base + (1 if i < rem else 0))
        per_seq.append((first, n))
    return CPPlan(tuple(seq_tiles), tuple(seg_cu), tuple(per_seq))


def split_balanced(seq_tiles: list[int], H: int, sm_count: int) -> CPPlan:
    """Spread total tiles over sm_count // H concurrent segment slots."""
    parallel_segs = max(1, sm_count // H)
    target_seg_tiles = max(AUTO_MIN_SEG_TILES, _ceil_div(sum(seq_tiles), parallel_segs))
    n_segs = [max(1, min(_ceil_div(tiles, target_seg_tiles), tiles // AUTO_MIN_SEG_TILES)) for tiles in seq_tiles]
    return _materialize(seq_tiles, n_segs)


def plan_auto(seq_tiles: list[int], H: int, sm_count: int) -> CPPlan:
    """split_balanced, then keep CP only if serial_wall >= RERUN_RATIO * cp_wall.

    Example: seq_tiles = [896] + [8] * 16, H=16, 132 SMs:
      parallel_segs    = 132 // 16 = 8
      target_seg_tiles = max(32, ceil(1024 / 8)) = 128
      split  = 896 -> 7 segments of 128; the 8-tile sequences stay whole
      engage = serial wall max(896, 128) >= CP wall 3 * max(128, 128) -> CP

    Counter-example: seq_tiles = [64], H=16 -> 2 segments of 32, but
    serial wall 64 < CP wall 3 * 32 = 96 -> serial.
    """
    plan = split_balanced(seq_tiles, H, sm_count)
    if plan.trivial:
        return CPPlan.serial(seq_tiles, "machine already full or sequences too short to split")
    parallel_segs = max(1, sm_count // H)
    load_bound = _ceil_div(sum(seq_tiles), parallel_segs)
    serial_wall = max(max(seq_tiles), load_bound)
    cp_wall = RERUN_RATIO * max(plan.max_seg_tiles, load_bound)
    if serial_wall < cp_wall:
        return CPPlan.serial(
            seq_tiles,
            f"serial wall {serial_wall} tiles vs CP wall ~{cp_wall:g}: "
            f"splitting does not pay the {RERUN_RATIO:g}x re-run cost",
        )
    return plan


def plan_manual(seq_tiles: list[int], s_split: int) -> CPPlan:
    """Up to s_split segments per sequence; no profitability check (tests)."""
    n_segs = [max(1, min(s_split, tiles // MIN_SEG_TILES)) for tiles in seq_tiles]
    return _materialize(seq_tiles, n_segs)


def plan_prefill(
    seq_tiles: list[int],
    H: int,
    device: torch.device,
    mode: CPMode | None = None,
) -> CPPlan:
    """Prefill planning entry. Trivial => serial. FORCE raises if unsplittable."""
    if mode is None or mode is CPMode.OFF:
        return CPPlan.serial((), "disabled")
    sm_count = get_device_sm_count(device)
    if mode is CPMode.FORCE:
        plan = split_balanced(seq_tiles, H, sm_count)
        if plan.trivial:
            raise NotSplittableError("SM90 intracard CP cannot split this shape.")
        return plan
    return plan_auto(seq_tiles, H, sm_count)
