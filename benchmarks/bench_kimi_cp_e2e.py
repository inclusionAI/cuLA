# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001

"""Compatibility entry point for the large-CP KDA emulator."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmarks.large_cp_emulation.bench_kda import main


if __name__ == "__main__":
    main()
