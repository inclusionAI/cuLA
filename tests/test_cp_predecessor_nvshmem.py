import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cula.kda import cp_overlap

REPO_ROOT = Path(__file__).resolve().parents[1]


def _json_payload(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"No JSON result found in output:\n{stdout}")


def test_cp_overlap_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CULA_CP_OVERLAP", raising=False)
    expected = object()
    monkeypatch.setattr(cp_overlap, "_fla_chunk_gated_delta_rule_fwd_h_pre_process", lambda **_: expected)

    result = cp_overlap.chunk_gated_delta_rule_fwd_h_pre_process_overlap(
        k=None,
        w=None,
        u=None,
        context=None,
    )

    assert not cp_overlap._cp_overlap_enabled()
    assert result is expected


@pytest.mark.skipif(
    os.getenv("CULA_RUN_NVSHMEM_TESTS") != "1",
    reason="Set CULA_RUN_NVSHMEM_TESTS=1 to run the NVSHMEM integration test",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs are required")
@pytest.mark.parametrize("world_size", [2, 4])
def test_nvshmem_predecessor_handoff_matches_fla_forward_and_backward(world_size):
    pytest.importorskip("nvshmem.core")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} GPUs are required")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_MAX_CONNECTIONS": "2",
            "CULA_CP_COMM_USE_CURRENT_STREAM": "1",
            "CULA_CP_NVSHMEM_DIRECT_STORE_CONN1_ONLY": "0",
            "CULA_CP_NVSHMEM_FUSED_REMOTE_MERGE": "1",
            "CULA_CP_NVSHMEM_READY_WAIT": "0",
            "NVSHMEM_DISABLE_CUDA_VMM": env.get("NVSHMEM_DISABLE_CUDA_VMM", "1"),
            "NVSHMEM_SYMMETRIC_SIZE": "256M",
            "NVSHMEM_IB_ENABLE": "0",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            "benchmarks/check_cp_predecessor_nvshmem.py",
            f"--world-size={world_size}",
            "--sequence-length=1024",
            "--heads=4",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert _json_payload(completed.stdout)["passed"]
