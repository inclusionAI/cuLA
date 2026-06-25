# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM90 correctness smoke tests for the public cula.kda.chunk_kda entrypoint."""

import torch

from benchmarks import bench_kda_fwd_bwd_e2e as bench
from benchmarks.utils import (
    SEED,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    relative_rms_error_rel_max_mean_abs,
    set_seed,
)

try:
    import pytest

    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

    class _DummyMark:
        def __getattr__(self, _name):
            return lambda *a, **kw: lambda f: f

    class _DummyPytest:
        mark = _DummyMark()

        @staticmethod
        def main(*_a, **_kw):
            raise SystemExit("pytest not installed; run via __main__ instead")

    pytest = _DummyPytest()  # type: ignore[assignment]


pytestmark = pytest.mark.sm90_only

D = 128
OUT_NAMES = ("o", "ht", "dq", "dk", "dv", "dg", "dbeta", "dh0")
MAX_REL_RMS = 0.05
MAX_REL_MAX = 0.25

SM90_CHUNK_CASES = (
    ("fixed_recompute_beta_fp32", 2, 64, (64, 64), 2, 2, torch.float32, False),
    ("fixed_saved_intermediates_beta_fp32", 1, 64, (64,), 2, 2, torch.float32, True),
    ("varlen_recompute_beta_bf16", 1, 96, (31, 65), 2, 2, torch.bfloat16, False),
)


def _make_common(batch_size, T, seq_lens, H, HV, beta_dtype):
    device = torch.device("cuda")
    cu_seqlens = torch.tensor(exclusive_cumsum(list(seq_lens)), dtype=torch.int32, device=device)
    inputs = prepare_safe_gate_inputs(
        batch_size,
        T,
        H,
        D,
        device,
        cu_seqlens=cu_seqlens,
        has_init_state=True,
        num_v_heads=HV,
    )
    inputs["beta"] = inputs["beta"].to(beta_dtype)

    set_seed(SEED + 1)
    do = torch.randn_like(inputs["v"])
    dht = torch.randn_like(inputs["init_state"])
    return dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        init_state=inputs["init_state"],
        cu_seqlens=cu_seqlens,
        lower_bound=inputs["lower_bound"],
        do=do,
        dht=dht,
    )


def _run_with_disable_recompute(common, disable_recompute):
    old_disable_recompute = bench.DISABLE_RECOMPUTE
    bench.DISABLE_RECOMPUTE = disable_recompute
    try:
        fla_results = bench.run_kda_e2e_with_grads(**common, fn=bench.fla_chunk_kda)
        cula_results = bench.run_kda_e2e_with_grads(**common, fn=bench.cula_chunk_kda)
        torch.cuda.synchronize()
    finally:
        bench.DISABLE_RECOMPUTE = old_disable_recompute
    return fla_results, cula_results


def _assert_results_match(case_id, ref, out):
    for name in OUT_NAMES:
        assert ref[name].shape == out[name].shape, f"{case_id}: {name} shape ref={ref[name].shape} out={out[name].shape}"
        assert torch.isfinite(out[name]).all(), f"{case_id}: cuLA {name} has non-finite values"
        rel_rms, rel_max, mean_abs = relative_rms_error_rel_max_mean_abs(ref[name], out[name])
        assert rel_rms < MAX_REL_RMS, f"{case_id}: {name} rel_rms={rel_rms:.6f} mean_abs={mean_abs:.6e}"
        assert rel_max < MAX_REL_MAX, f"{case_id}: {name} rel_max={rel_max:.6f} mean_abs={mean_abs:.6e}"


def _run_case(case):
    case_id, batch_size, T, seq_lens, H, HV, beta_dtype, disable_recompute = case
    common = _make_common(batch_size, T, seq_lens, H, HV, beta_dtype)
    ref, out = _run_with_disable_recompute(common, disable_recompute)
    _assert_results_match(case_id, ref, out)
    torch.cuda.empty_cache()


@pytest.mark.sm90_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", SM90_CHUNK_CASES, ids=[case[0] for case in SM90_CHUNK_CASES])
def test_chunk_kda_sm90_entry_matches_fla(case):
    _run_case(case)


if __name__ == "__main__":
    for _case in SM90_CHUNK_CASES:
        print(f"Running {_case[0]} ...", flush=True)
        _run_case(_case)
    print("SM90 chunk_kda entrypoint smoke PASS", flush=True)
