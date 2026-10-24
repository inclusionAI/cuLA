# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Normalize NVVM binding names across supported CuTeDSL releases."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any


def adapt_keywords(function: Callable[..., Any], renames: Mapping[str, str]) -> Callable[..., Any]:
    """Translate legacy keyword names only when the binding uses new names."""
    parameters = inspect.signature(function).parameters
    active = {old: new for old, new in renames.items() if old not in parameters and new in parameters}
    if not active:
        return function

    @wraps(function)
    def adapted(*args, **kwargs):
        for old, new in active.items():
            if old not in kwargs:
                continue
            if new in kwargs:
                raise TypeError(f"Both {old!r} and {new!r} were provided")
            kwargs[new] = kwargs.pop(old)
        return function(*args, **kwargs)

    return adapted


_MATRIX_KEYWORDS = {"mma_kind": "kind", "d": "matrix_d", "a": "matrix_a", "b": "matrix_b"}


def bind_mma_ops(nvvm):
    """Return CuTeDSL 4.6-style MMA callables for a 4.6 or 4.8 NVVM module."""
    return (
        adapt_keywords(
            nvvm.tcgen05_mma,
            {**_MATRIX_KEYWORDS, "write_disable_mask": "disable_output_lane"},
        ),
        adapt_keywords(
            nvvm.tcgen05_mma_ws,
            {**_MATRIX_KEYWORDS, "col_b_zero_mask": "zero_col_mask"},
        ),
    )
