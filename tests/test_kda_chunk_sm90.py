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

"""SM90 correctness tests for the public ``cula.kda.chunk_kda`` entrypoint."""

import pytest
import torch

from benchmarks import bench_kda_fwd_bwd_e2e as bench
from benchmarks.utils import (
    SEED,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    relative_rms_error_rel_max_mean_abs,
    set_seed,
)

pytestmark = pytest.mark.sm90_only

D = 128
OUT_NAMES = ("o", "ht", "dq", "dk", "dv", "dg", "dbeta", "dh0")

SM90_CHUNK_CASES = (
    ("fixed_recompute_beta_fp32", 2, 64, (64, 64), torch.float32, False),
    ("fixed_saved_intermediates_beta_fp32", 1, 64, (64,), torch.float32, True),
    ("varlen_recompute_beta_bf16", 1, 96, (31, 65), torch.bfloat16, False),
)


def _make_inputs(batch_size, length, seq_lens, beta_dtype):
    device = torch.device("cuda")
    cu_seqlens = torch.tensor(exclusive_cumsum(list(seq_lens)), dtype=torch.int32, device=device)
    inputs = prepare_safe_gate_inputs(
        batch_size,
        length,
        2,
        D,
        device,
        cu_seqlens=cu_seqlens,
        has_init_state=True,
        num_v_heads=2,
    )
    inputs["beta"] = inputs["beta"].to(beta_dtype)
    set_seed(SEED + 1)
    return {
        "q": inputs["q"],
        "k": inputs["k"],
        "v": inputs["v"],
        "g": inputs["g"],
        "beta": inputs["beta"],
        "scale": inputs["scale"],
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "init_state": inputs["init_state"],
        "cu_seqlens": cu_seqlens,
        "lower_bound": inputs["lower_bound"],
        "do": torch.randn_like(inputs["v"]),
        "dht": torch.randn_like(inputs["init_state"]),
    }


def _run_case(case):
    case_id, batch_size, length, seq_lens, beta_dtype, disable_recompute = case
    inputs = _make_inputs(batch_size, length, seq_lens, beta_dtype)
    previous = bench.DISABLE_RECOMPUTE
    bench.DISABLE_RECOMPUTE = disable_recompute
    try:
        reference = bench.run_kda_e2e_with_grads(**inputs, fn=bench.fla_chunk_kda)
        actual = bench.run_kda_e2e_with_grads(**inputs, fn=bench.cula_chunk_kda)
        torch.cuda.synchronize()
    finally:
        bench.DISABLE_RECOMPUTE = previous

    for name in OUT_NAMES:
        assert reference[name].shape == actual[name].shape
        assert torch.isfinite(actual[name]).all(), f"{case_id}: {name} contains non-finite values"
        rel_rms, rel_max, mean_abs = relative_rms_error_rel_max_mean_abs(reference[name], actual[name])
        assert rel_rms < 0.05, f"{case_id}: {name} rel_rms={rel_rms:.6f}, mean_abs={mean_abs:.6e}"
        assert rel_max < 0.25, f"{case_id}: {name} rel_max={rel_max:.6f}, mean_abs={mean_abs:.6e}"


@pytest.mark.parametrize("case", SM90_CHUNK_CASES, ids=[case[0] for case in SM90_CHUNK_CASES])
def test_chunk_kda_sm90_entry_matches_fla(case):
    _run_case(case)
    torch.cuda.empty_cache()
