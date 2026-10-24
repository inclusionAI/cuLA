# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for FlashKDA's barrier-accounted workspace transport."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "run_function", "workspace_names"),
    [
        ("cula/ops/kda/sm90/k2.py", "run_k2", ("ws_kd", "ws_qd", "ws_kr")),
        ("cula/ops/kda/sm90/cp/pre_scan.py", "run_pre_scan", ("ws_kd", "ws_kr")),
    ],
)
def test_workspace_loads_use_barrier_accounted_plain_tma(relative_path, run_function, workspace_names):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == run_function)
    run_text = ast.unparse(run)

    assert "CopyBulkG2SOp" not in source
    assert "raw_qk_smem = cute.make_layout((CHUNK, D), stride=(D, 1))" in run_text
    for name in workspace_names:
        assert f"make_raw_ws_qk_atom({name})" in run_text

    kernel_text = ast.unparse(
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.endswith("kernel"))
    )
    assert "raw_qk_stage_layout = cute.make_layout((CHUNK, D, STAGES), stride=(D, 1, CHUNK * D))" in kernel_text
    for atom in ("tma_atom_kd", "tma_atom_kr"):
        assert f"cute.copy({atom}" in kernel_text
    if "ws_qd" in workspace_names:
        assert "cute.copy(tma_atom_qd" in kernel_text
    assert kernel_text.count("tma_bar_ptr=bar_l") >= len(workspace_names)
