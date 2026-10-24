# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for FlashKDA's barrier-accounted workspace transport."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "run_function", "workspace_names", "expected_tma_bytes"),
    [
        (
            "cula/ops/kda/sm90/k2.py",
            "run_k2",
            ("ws_kd", "ws_qd", "ws_kr"),
            "4 * CHUNK * D * 2 + 2 * CHUNK * CHUNK * 2 + D * 4 + CHUNK * 2",
        ),
        (
            "cula/ops/kda/sm90/cp/pre_scan.py",
            "run_pre_scan",
            ("ws_kd", "ws_kr"),
            "3 * CHUNK * D * 2 + CHUNK * CHUNK * 2 + D * 4 + CHUNK * 2",
        ),
    ],
)
def test_workspace_loads_use_barrier_accounted_plain_tma(relative_path, run_function, workspace_names, expected_tma_bytes):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == run_function)
    run_text = ast.unparse(run)

    assert "CopyBulkG2SOp" not in source
    assert "raw_qk_smem = cute.make_layout((CHUNK, D), stride=(D, 1))" in run_text
    for name in workspace_names:
        assert f"make_raw_ws_qk_atom({name})" in run_text

    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.endswith("kernel"))
    kernel_text = ast.unparse(kernel)
    assert "raw_qk_stage_layout = cute.make_layout((CHUNK, D, STAGES), stride=(D, 1, CHUNK * D))" in kernel_text

    tma_bytes = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "TMA_BYTES"
    )
    expected_bytes = ast.parse(expected_tma_bytes, mode="eval").body
    assert ast.dump(tma_bytes.value, include_attributes=False) == ast.dump(expected_bytes, include_attributes=False)

    copy_calls = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "cute.copy"
    ]
    for workspace_name in workspace_names:
        atom = f"tma_atom_{workspace_name.removeprefix('ws_')}"
        workspace_copies = [call for call in copy_calls if call.args and ast.unparse(call.args[0]) == atom]
        assert len(workspace_copies) == 1
        assert any(
            keyword.arg == "tma_bar_ptr" and ast.unparse(keyword.value) == "bar_l" for keyword in workspace_copies[0].keywords
        )
