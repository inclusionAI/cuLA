# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import cula.kernels.kda as kda
import cula.kernels.lightning as lightning
from cula.kernels import get_kernel, list_kernels
from cula.ops.kda.index import prepare_chunk_indices, prepare_lens


def test_runtime_package_has_no_fla_or_triton_imports():
    package_root = Path(__file__).parents[1] / "cula"
    forbidden: list[tuple[Path, int, str]] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "fla" or name.startswith("fla.") or name == "triton" or name.startswith("triton."):
                    forbidden.append((path.relative_to(package_root), node.lineno, name))
    assert forbidden == []


def test_registry_names_are_exported():
    namespaces = {"kda": kda, "lightning": lightning}
    for spec in list_kernels():
        assert spec.name in namespaces[spec.family].__all__
        assert get_kernel(spec.name) is spec


def test_registry_filters_are_stable():
    kda_primitives = list_kernels(family="kda", level="primitive")
    assert kda_primitives
    assert all(spec.family == "kda" and spec.level == "primitive" for spec in kda_primitives)
    with pytest.raises(KeyError, match="unknown standalone kernel"):
        get_kernel("not_a_kernel")


def test_prepare_chunk_indices_cpu_contract():
    cu_seqlens = torch.tensor([0, 1, 65, 193], dtype=torch.int32)
    actual = prepare_chunk_indices(cu_seqlens, 64)
    expected = torch.tensor([[0, 0], [1, 0], [2, 0], [2, 1]], dtype=torch.int32)
    assert torch.equal(actual, expected)
    assert torch.equal(prepare_lens(cu_seqlens), torch.tensor([1, 64, 128], dtype=torch.int32))


def test_prepare_chunk_indices_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="positive"):
        prepare_chunk_indices(torch.tensor([0, 4], dtype=torch.int32), 0)
    with pytest.raises(ValueError, match="one-dimensional"):
        prepare_chunk_indices(torch.zeros((2, 2), dtype=torch.int32), 64)
