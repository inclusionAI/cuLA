"""Numerical tests for the SM90 fused WY-DqKG backward kernel.

The tests compare SM90 fused outputs against the FLA Triton fused kernel.
Both fixed-length and ragged varlen partial-chunk paths are covered.
"""

import os

import torch
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused as chunk_kda_bwd_triton

from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    BK as BENCHMARK_BK,
)
from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    BT as BENCHMARK_BT,
)
from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    BV as BENCHMARK_BV,
)
from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    MIN_OCC as BENCHMARK_MIN_OCC,
)
from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    SEED as BENCHMARK_SEED,
)
from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    K as BENCHMARK_K,
)
from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    V as BENCHMARK_V,
)
from benchmarks.bench_kda_bwd_wy_dqkg_sm90 import (
    benchmark_fixed_configs,
    benchmark_varlen_configs,
    prepare_bwd_wy_dqkg_fused_inputs,
)
from benchmarks.utils import exclusive_cumsum
from cula.ops.chunk_wy_dqkg_sm90 import chunk_kda_bwd_wy_dqkg_fused

# pytest is optional — when absent (e.g. minimal CI venv) we still allow
# the module to be executed directly via ``python tests/test_*.py``.
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


OUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dA")


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _benchmark_test_heads():
    heads = os.environ.get("CULA_BENCHMARK_TEST_HEADS")
    if heads is None:
        return (32,)
    return tuple(int(h.strip()) for h in heads.split(",") if h.strip())


def _benchmark_fixed_test_cases():
    return [(H, B, T) for H in _benchmark_test_heads() for B, T in benchmark_fixed_configs()]


def _benchmark_varlen_test_cases():
    return [
        (H, seq_lens, total_len, dist)
        for H in _benchmark_test_heads()
        for seq_lens, total_len, dist in benchmark_varlen_configs()
    ]


def _determinism_fixed_test_cases():
    min_t = _env_int("CULA_DETERMINISM_FIXED_MIN_T", 16384)
    return [(H, B, T) for H, B, T in _benchmark_fixed_test_cases() if min_t <= T]


def _determinism_varlen_test_cases():
    min_total_len = _env_int("CULA_DETERMINISM_VARLEN_MIN_TOTAL_LEN", 16384)
    return [
        (H, seq_lens, total_len, dist)
        for H, seq_lens, total_len, dist in _benchmark_varlen_test_cases()
        if total_len >= min_total_len
    ]


def _fixed_case_id(case):
    H, B, T = case
    return f"H{H}-B{B}-T{T}"


def _varlen_case_id(case):
    H, seq_lens, total_len, dist = case
    return f"H{H}-{dist}-{len(seq_lens)}seqs-T{total_len}-min{min(seq_lens)}-max{max(seq_lens)}"


BENCHMARK_FIXED_TEST_CASES = _benchmark_fixed_test_cases()
BENCHMARK_VARLEN_TEST_CASES = _benchmark_varlen_test_cases()
DETERMINISM_FIXED_TEST_CASES = _determinism_fixed_test_cases()
DETERMINISM_VARLEN_TEST_CASES = _determinism_varlen_test_cases()


def accuracy_stats(ref, out):
    """Compute err_ratio, relative max diff, and mean absolute difference."""
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    err = diff.flatten().pow(2).mean().sqrt().item()
    base = ref_f.flatten().pow(2).mean().sqrt().item()
    err_ratio = err / (base + 1e-8)
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.mean().item()
    return err_ratio, rel_max, mean_diff


def _print_tensor_stats(name, tensor, ref_tensor=None):
    """Print diagnostics for a tensor, optionally vs a reference."""
    f = tensor.float()
    print(f"{name} norm={f.norm().item():.6e}  min={f.min().item():.6e}  max={f.max().item():.6e}", flush=True)
    if ref_tensor is not None:
        err_ratio, rel_max, mean_diff = accuracy_stats(ref_tensor, tensor)
        print(f"  vs ref: err_ratio={err_ratio:.6f}  rel_max={rel_max:.6f}  mean_diff={mean_diff:.6e}", flush=True)


def _assert_outputs_match_fla(sm90_outputs, fla_outputs, case_id, *, max_err=0.05, verbose=False):
    for name, sm90, fla in zip(OUT_NAMES, sm90_outputs, fla_outputs):
        assert sm90.shape == fla.shape, f"{case_id}: {name} shape sm90={tuple(sm90.shape)} fla={tuple(fla.shape)}"
        err_ratio, rel_max, mean_diff = accuracy_stats(fla, sm90)
        if verbose:
            print(
                f"{case_id} {name}: err_ratio={err_ratio:.6f} rel_max={rel_max:.6f} mean_diff={mean_diff:.6e}",
                flush=True,
            )
            _print_tensor_stats(f"{name}_sm90", sm90, fla)
        assert err_ratio < max_err, f"{case_id}: {name} vs FLA err_ratio={err_ratio:.6f} too high"


def _run_matches_fla_fixed(
    B=1,
    T=64,
    H=4,
    K=128,
    V=128,
    BT=64,
    verbose=False,
    bk=32,
    bv=64,
    min_occupancy=2,
    beta_dtype=torch.float32,
):
    """Verify SM90 fused outputs on fixed-length inputs against FLA."""
    device = "cuda"
    dtype = torch.bfloat16
    scale = K**-0.5
    NT = T // BT

    torch.manual_seed(0)
    do_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    h_tensor = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
    g_tensor = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    q_tensor = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k_tensor = torch.randn(B, T, H, K, dtype=dtype, device=device)
    vnew_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    dh_tensor = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
    dv_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    v_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    A_tensor = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    beta_tensor = (torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.5 + 0.5).to(beta_dtype)

    if verbose:
        print(f"=== inputs: B={B} T={T} H={H} K={K} V={V} BT={BT} NT={NT} beta={beta_dtype} ===", flush=True)

    if verbose:
        print("=== invoking FLA Triton baseline ===", flush=True)
    fla_outputs = chunk_kda_bwd_triton(
        q=q_tensor,
        k=k_tensor,
        v=v_tensor,
        v_new=vnew_tensor,
        g=g_tensor,
        beta=beta_tensor,
        A=A_tensor,
        h=h_tensor,
        do=do_tensor,
        dh=dh_tensor,
        dv=dv_tensor,
        scale=scale,
        chunk_size=BT,
    )
    torch.cuda.synchronize()

    if verbose:
        print(f"A[0,0,0,:4]={A_tensor[0, 0, 0, :4].tolist()}", flush=True)
        print(f"A[0,1,0,:4]={A_tensor[0, 1, 0, :4].tolist()}", flush=True)
        print(f"A[0,:4,0,0]={A_tensor[0, :4, 0, 0].tolist()}", flush=True)
        print(f"A[0,:4,0,1]={A_tensor[0, :4, 0, 1].tolist()}", flush=True)

    if verbose:
        print("\n=== invoking SM90 fused wrapper ===", flush=True)
    sm90_outputs = chunk_kda_bwd_wy_dqkg_fused(
        q=q_tensor,
        k=k_tensor,
        v=v_tensor,
        v_new=vnew_tensor,
        g=g_tensor,
        beta=beta_tensor,
        A=A_tensor,
        h=h_tensor,
        do=do_tensor,
        dh=dh_tensor,
        dv=dv_tensor,
        scale=scale,
        chunk_size=BT,
        bk=bk,
        bv=bv,
        min_occupancy=min_occupancy,
    )
    torch.cuda.synchronize()

    _assert_outputs_match_fla(sm90_outputs, fla_outputs, f"fixed B={B} T={T} H={H}", verbose=verbose)
    return sm90_outputs


@pytest.mark.sm90_only
@pytest.mark.parametrize("B, T, H, K, V, BT", [(1, 64, 4, 128, 128, 64)])
@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=["beta_fp32", "beta_bf16"])
def test_matches_fla_fixed(B, T, H, K, V, BT, beta_dtype):
    _run_matches_fla_fixed(B, T, H, K, V, BT, beta_dtype=beta_dtype)


def _run_matches_fla_uniform_varlen(
    B=2,
    T=64,
    H=4,
    K=128,
    V=128,
    BT=64,
    verbose=False,
    bk=32,
    bv=64,
    min_occupancy=2,
):
    """Verify SM90 fused varlen path on uniform cu_seqlens (= prepare_uniform_cu_seqlens).

    Uses the explicit cu_seqlens / chunk_indices code path through the wrapper:
    feeds reshape-to-[1, B*T, ...] tensors plus cu_seqlens=[0, T, 2T, ..., B*T].
    """
    from fla.ops.utils.index import prepare_chunk_indices

    from cula.utils import prepare_uniform_cu_seqlens

    device = "cuda"
    dtype = torch.bfloat16
    scale = K**-0.5
    NT = T // BT

    torch.manual_seed(0)
    do_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    h_tensor = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
    g_tensor = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    q_tensor = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k_tensor = torch.randn(B, T, H, K, dtype=dtype, device=device)
    vnew_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    dh_tensor = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
    dv_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    v_tensor = torch.randn(B, T, H, V, dtype=dtype, device=device)
    A_tensor = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    beta_tensor = torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.5 + 0.5

    cu_seqlens = prepare_uniform_cu_seqlens(B, T, device, torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    fla_outputs = chunk_kda_bwd_triton(
        q=q_tensor,
        k=k_tensor,
        v=v_tensor,
        v_new=vnew_tensor,
        g=g_tensor,
        beta=beta_tensor,
        A=A_tensor,
        h=h_tensor,
        do=do_tensor,
        dh=dh_tensor,
        dv=dv_tensor,
        scale=scale,
        chunk_size=BT,
    )

    sm90_outputs = chunk_kda_bwd_wy_dqkg_fused(
        q=q_tensor,
        k=k_tensor,
        v=v_tensor,
        v_new=vnew_tensor,
        g=g_tensor,
        beta=beta_tensor,
        A=A_tensor,
        h=h_tensor,
        do=do_tensor,
        dh=dh_tensor,
        dv=dv_tensor,
        scale=scale,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        bk=bk,
        bv=bv,
        min_occupancy=min_occupancy,
    )
    torch.cuda.synchronize()

    if verbose:
        print(f"=== uniform varlen B={B} T={T} (T_total={B * T}) ===", flush=True)
    _assert_outputs_match_fla(sm90_outputs, fla_outputs, f"uniform varlen B={B} T={T} H={H}", verbose=verbose)
    return sm90_outputs


@pytest.mark.sm90_only
@pytest.mark.parametrize("B, T, H, K, V, BT", [(2, 64, 4, 128, 128, 64)])
def test_matches_fla_uniform_varlen(B, T, H, K, V, BT):
    _run_matches_fla_uniform_varlen(B, T, H, K, V, BT)


def _run_matches_fla_ragged_varlen(
    seq_lens,
    H=4,
    K=128,
    V=128,
    BT=64,
    verbose=False,
    bk=32,
    bv=64,
    min_occupancy=2,
):
    """Verify SM90 partial-chunk row mask on ragged cu_seqlens.

    All 6 outputs (dq, dk, dv, db, dg, dA) asserted per-sequence against
    the Triton FLA reference (which supports varlen via cu_seqlens).
    """
    import itertools

    from fla.ops.utils.index import prepare_chunk_indices

    device = "cuda"
    dtype = torch.bfloat16
    scale = K**-0.5

    T_total = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(seq_lens)),
        dtype=torch.int32,
        device=device,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT_total = chunk_indices.shape[0]

    torch.manual_seed(0)
    do_tensor = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
    h_tensor = torch.randn(1, NT_total, H, K, V, dtype=dtype, device=device) * 0.01
    g_tensor = torch.randn(1, T_total, H, K, dtype=torch.float32, device=device) * 0.1
    q_tensor = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
    k_tensor = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
    vnew_tensor = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
    dh_tensor = torch.randn(1, NT_total, H, K, V, dtype=dtype, device=device) * 0.01
    dv_tensor = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
    v_tensor = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
    A_tensor = torch.randn(1, T_total, H, BT, dtype=dtype, device=device)
    beta_tensor = torch.rand(1, T_total, H, dtype=torch.float32, device=device) * 0.5 + 0.5

    # FLA Triton reference (supports varlen via cu_seqlens)
    dq_fla, dk_fla, dv_fla, db_fla, dg_fla, dA_fla = chunk_kda_bwd_triton(
        q=q_tensor,
        k=k_tensor,
        v=v_tensor,
        v_new=vnew_tensor,
        g=g_tensor,
        beta=beta_tensor,
        A=A_tensor,
        h=h_tensor,
        do=do_tensor,
        dh=dh_tensor,
        dv=dv_tensor,
        scale=scale,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
    )

    dq_sm90, dk_sm90, dv_sm90, db_sm90, dg_sm90, dA_sm90 = chunk_kda_bwd_wy_dqkg_fused(
        q=q_tensor,
        k=k_tensor,
        v=v_tensor,
        v_new=vnew_tensor,
        g=g_tensor,
        beta=beta_tensor,
        A=A_tensor,
        h=h_tensor,
        do=do_tensor,
        dh=dh_tensor,
        dv=dv_tensor,
        scale=scale,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        bk=bk,
        bv=bv,
        min_occupancy=min_occupancy,
    )
    torch.cuda.synchronize()

    if verbose:
        print(f"=== ragged seq_lens={seq_lens} T_total={T_total} NT_total={NT_total} ===", flush=True)
        print(f"chunk_indices = {chunk_indices.tolist()}", flush=True)

    names_pairs = [
        ("dq", dq_sm90, dq_fla),
        ("dk", dk_sm90, dk_fla),
        ("dv", dv_sm90, dv_fla),
        ("db", db_sm90, db_fla),
        ("dg", dg_sm90, dg_fla),
        ("dA", dA_sm90, dA_fla),
    ]
    # Per-seq comparison for each of the 6 outputs
    for b in range(len(seq_lens)):
        start = cu_seqlens[b].item()
        end = cu_seqlens[b + 1].item()
        for name, sm90, ref in names_pairs:
            sm90_slice = sm90[0, start:end]
            ref_slice = ref[0, start:end]
            err_ratio, rel_max, mean_diff = accuracy_stats(ref_slice, sm90_slice)
            if verbose:
                print(
                    f"  seq[{b}] [{start},{end}) {name:>3}: "
                    f"err_ratio={err_ratio:.6f} rel_max={rel_max:.6f} mean_diff={mean_diff:.6e}",
                    flush=True,
                )
            assert err_ratio < 0.003, f"seq[{b}] {name} err_ratio={err_ratio:.6f} too high"

    return dq_sm90


@pytest.mark.sm90_only
@pytest.mark.parametrize("seq_lens", [[64, 96], [48, 112]])
def test_matches_fla_ragged_varlen(seq_lens):
    _run_matches_fla_ragged_varlen(seq_lens)


def _prepare_benchmark_fixed_inputs(B, T, H, K=BENCHMARK_K, V=BENCHMARK_V, BT=BENCHMARK_BT):
    device = torch.device("cuda")
    cu_seqlens = torch.tensor(exclusive_cumsum([T] * B), dtype=torch.int32, device=device)
    return prepare_bwd_wy_dqkg_fused_inputs(
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        chunk_size=BT,
        device=device,
        seed=BENCHMARK_SEED,
        cu_seqlens=cu_seqlens,
    )


def _prepare_benchmark_varlen_inputs(seq_lens, total_len, H, K=BENCHMARK_K, V=BENCHMARK_V, BT=BENCHMARK_BT):
    device = torch.device("cuda")
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    return prepare_bwd_wy_dqkg_fused_inputs(
        B=1,
        T=total_len,
        H=H,
        K=K,
        V=V,
        chunk_size=BT,
        device=device,
        seed=BENCHMARK_SEED,
        cu_seqlens=cu_seqlens,
    )


def _run_sm90_from_benchmark_inputs(
    inputs,
    *,
    chunk_size=BENCHMARK_BT,
    bk=BENCHMARK_BK,
    bv=BENCHMARK_BV,
    min_occupancy=BENCHMARK_MIN_OCC,
):
    outputs = chunk_kda_bwd_wy_dqkg_fused(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        v_new=inputs["v_new"],
        g=inputs["g"],
        beta=inputs["beta"],
        A=inputs["A"],
        h=inputs["h"],
        do=inputs["do"],
        dh=inputs["dh"],
        dv=inputs["dv"],
        scale=inputs["scale"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_size=chunk_size,
        chunk_indices=inputs["chunk_indices"],
        bk=bk,
        bv=bv,
        min_occupancy=min_occupancy,
    )
    torch.cuda.synchronize()
    return outputs


def _run_benchmark_input_determinism(
    inputs,
    case_id,
    *,
    iters=None,
    chunk_size=BENCHMARK_BT,
    bk=BENCHMARK_BK,
    bv=BENCHMARK_BV,
    min_occupancy=BENCHMARK_MIN_OCC,
):
    """Multiple SM90 calls on one benchmark-shaped input must be bitwise identical."""
    if iters is None:
        iters = _env_int("CULA_DETERMINISM_ITERS", 2)

    ref = tuple(
        out.clone()
        for out in _run_sm90_from_benchmark_inputs(
            inputs,
            chunk_size=chunk_size,
            bk=bk,
            bv=bv,
            min_occupancy=min_occupancy,
        )
    )
    for i in range(iters):
        actual = _run_sm90_from_benchmark_inputs(
            inputs,
            chunk_size=chunk_size,
            bk=bk,
            bv=bv,
            min_occupancy=min_occupancy,
        )
        for name, got, expected in zip(OUT_NAMES, actual, ref):
            assert torch.isfinite(got.float()).all(), f"{case_id}: {name} has non-finite values at iter {i}"
            assert torch.equal(got, expected), f"{case_id}: non-deterministic {name} at iter {i}"


def _run_determinism(B=1, T=64, H=4, K=128, V=128, BT=64, iters=2, bk=32, bv=64, min_occupancy=2):
    """Compatibility entry point for focused sanitizer/debug commands."""
    inputs = _prepare_benchmark_fixed_inputs(B, T, H, K=K, V=V, BT=BT)
    _run_benchmark_input_determinism(
        inputs,
        f"fixed-H{H}-B{B}-T{T}",
        iters=iters,
        chunk_size=BT,
        bk=bk,
        bv=bv,
        min_occupancy=min_occupancy,
    )


@pytest.mark.benchmark
@pytest.mark.sm90_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "H, B, T",
    DETERMINISM_FIXED_TEST_CASES,
    ids=[_fixed_case_id(case) for case in DETERMINISM_FIXED_TEST_CASES],
)
def test_determinism_benchmark_fixed_cases(H, B, T):
    inputs = _prepare_benchmark_fixed_inputs(B, T, H)
    _run_benchmark_input_determinism(inputs, f"fixed-H{H}-B{B}-T{T}")
    torch.cuda.empty_cache()


@pytest.mark.benchmark
@pytest.mark.sm90_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "H, seq_lens, total_len, dist",
    DETERMINISM_VARLEN_TEST_CASES,
    ids=[_varlen_case_id(case) for case in DETERMINISM_VARLEN_TEST_CASES],
)
def test_determinism_benchmark_varlen_cases(H, seq_lens, total_len, dist):
    inputs = _prepare_benchmark_varlen_inputs(seq_lens, total_len, H)
    case_id = _varlen_case_id((H, seq_lens, total_len, dist))
    _run_benchmark_input_determinism(inputs, case_id)
    torch.cuda.empty_cache()


def _run_benchmark_sanitizer_case(inputs, chunk_size=BENCHMARK_BT):
    outputs = _run_sm90_from_benchmark_inputs(inputs, chunk_size=chunk_size)
    _, total_len, H, K = inputs["q"].shape
    V = inputs["v"].shape[-1]
    expected_shapes = (
        (1, total_len, H, K),
        (1, total_len, H, K),
        (1, total_len, H, V),
        (1, total_len, H),
        (1, total_len, H, K),
        (1, total_len, H, chunk_size),
    )
    for out, expected_shape in zip(outputs, expected_shapes):
        assert tuple(out.shape) == expected_shape


@pytest.mark.sanitizer
@pytest.mark.benchmark
@pytest.mark.sm90_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    os.environ.get("CULA_RUN_SANITIZER_TESTS") != "1",
    reason="set CULA_RUN_SANITIZER_TESTS=1 and run under compute-sanitizer",
)
@pytest.mark.parametrize(
    "H, B, T",
    BENCHMARK_FIXED_TEST_CASES,
    ids=[_fixed_case_id(case) for case in BENCHMARK_FIXED_TEST_CASES],
)
def test_sanitizer_benchmark_fixed_cases(H, B, T):
    inputs = _prepare_benchmark_fixed_inputs(B, T, H)
    _run_benchmark_sanitizer_case(inputs)
    torch.cuda.empty_cache()


@pytest.mark.sanitizer
@pytest.mark.benchmark
@pytest.mark.sm90_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    os.environ.get("CULA_RUN_SANITIZER_TESTS") != "1",
    reason="set CULA_RUN_SANITIZER_TESTS=1 and run under compute-sanitizer",
)
@pytest.mark.parametrize(
    "H, seq_lens, total_len, dist",
    BENCHMARK_VARLEN_TEST_CASES,
    ids=[_varlen_case_id(case) for case in BENCHMARK_VARLEN_TEST_CASES],
)
def test_sanitizer_benchmark_varlen_cases(H, seq_lens, total_len, dist):
    inputs = _prepare_benchmark_varlen_inputs(seq_lens, total_len, H)
    _run_benchmark_sanitizer_case(inputs)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    bk = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    bv = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    occ = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print(f"Testing with bk={bk} bv={bv} min_occupancy={occ}")
    _run_matches_fla_fixed(verbose=True, bk=bk, bv=bv, min_occupancy=occ)
    print("\n✅ test_matches_fla_fixed PASSED")
    print("\n=== uniform varlen (B=2) ===")
    _run_matches_fla_uniform_varlen(verbose=True, bk=bk, bv=bv, min_occupancy=occ)
    print("\n✅ test_matches_fla_uniform_varlen PASSED")
