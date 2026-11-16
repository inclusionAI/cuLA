# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.summarize_csa_layout_matrix import summarize


def _matrix(tmp_path):
    tensors = [
        "output",
        "dq",
        "dkv",
        "d_index_q",
        "d_weights",
        "d_pool_kv",
        "d_pool_score",
        "d_ape",
        "kda_output",
        "kda_dq",
        "kda_dk",
        "kda_dv",
        "kda_dg",
        "kda_dbeta",
    ]
    for layout, latency, setup in [("contiguous", 10, 1), ("zigzag", 8, 3), ("planner", 7, 5)]:
        rows = [
            dict(
                rank=rank,
                finite=True,
                input_match=True,
                redistribution_bitwise=True,
                samples=[dict(total_ms=latency + rank, peak_allocated_gib=1, peak_reserved_gib=2)],
                layout_prepare_ms=setup,
                common_prepare_ms=2,
                kda_prepare_ms=3,
                total_prepare_ms=setup + 5,
                selected_policy=layout,
                selection_workspace_bytes=0,
                selection_spans=1,
                layout_vs_contiguous=(
                    None if layout == "contiguous" else {k: dict(finite=True, relative_rms=0, max_abs=0) for k in tensors}
                ),
                loss_abs_diff=None if layout == "contiguous" else 0,
            )
            for rank in range(2)
        ]
        data = dict(world_size=2, config=dict(layout=layout, repeats=1, sequence_lengths=[1 << 20], with_kda=True), ranks=rows)
        (tmp_path / f"layout-1m-fixture-{layout}.json").write_text(json.dumps(data))


def test_report_distinguishes_steady_and_preparation_inclusive_cost(tmp_path):
    _matrix(tmp_path)
    case = summarize(tmp_path)["1m-fixture"]
    assert case["speedup_vs_contiguous"] == 11 / 8
    assert case["prepare_plus_iteration_ms"] == dict(contiguous=17, zigzag=17, planner=18)
    assert case["prepare_inclusive_gap_pct"] == pytest.approx(100 * (18 / 17 - 1))


@pytest.mark.parametrize(
    "broken",
    [
        "missing_all",
        "missing_dg",
        "bad_loss",
        "nan_loss",
        "missing_rank",
        "nan_time",
        "negative_memory",
        "heads",
        "kda_batch_invariant",
        "kda_gate_scale",
        "candidate_mib",
        "nan_prepare",
        "inconsistent_prepare",
    ],
)
def test_report_rejects_incomplete_precision_evidence(tmp_path, broken):
    _matrix(tmp_path)
    path = tmp_path / "layout-1m-fixture-planner.json"
    data = json.loads(path.read_text())
    row = data["ranks"][0]
    if broken == "missing_all":
        row["layout_vs_contiguous"] = None
    elif broken == "missing_dg":
        del row["layout_vs_contiguous"]["kda_dg"]
    elif broken == "bad_loss":
        row["loss_abs_diff"] = 1
    elif broken == "nan_loss":
        row["loss_abs_diff"] = float("nan")
    elif broken == "missing_rank":
        data["ranks"].pop()
    elif broken == "nan_time":
        row["samples"][0]["total_ms"] = float("nan")
    elif broken == "negative_memory":
        row["samples"][0]["peak_allocated_gib"] = -1
    elif broken in ("heads", "kda_batch_invariant", "kda_gate_scale", "candidate_mib"):
        data["config"][broken] = 999
    elif broken == "nan_prepare":
        row["total_prepare_ms"] = float("nan")
    elif broken == "inconsistent_prepare":
        row["total_prepare_ms"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize(tmp_path)


@pytest.mark.parametrize("layout", ["contiguous", "zigzag", "planner"])
@pytest.mark.parametrize("rank", [0, 1])
def test_report_rejects_failed_fp32_check(tmp_path, layout, rank):
    _matrix(tmp_path)
    path = tmp_path / f"layout-1m-fixture-{layout}.json"
    data = json.loads(path.read_text())
    data["ranks"][rank]["check"] = {"pass_": False}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="failed correctness check"):
        summarize(tmp_path)


@pytest.mark.parametrize("check", [None, {"pass_": True}])
def test_report_accepts_optional_or_successful_fp32_check(tmp_path, check):
    _matrix(tmp_path)
    for path in tmp_path.glob("*.json"):
        data = json.loads(path.read_text())
        for rank in data["ranks"]:
            rank["check"] = check
        path.write_text(json.dumps(data))
    assert "1m-fixture" in summarize(tmp_path)
