# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared lazy-export support for public kernel namespaces."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def resolve(mapping: Mapping[str, tuple[str, str]], module_name: str, name: str) -> Any:
    target = mapping.get(name)
    if target is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    return getattr(importlib.import_module(target[0]), target[1])
