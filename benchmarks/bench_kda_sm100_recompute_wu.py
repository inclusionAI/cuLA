# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare SM100 C++ and CuTe DSL recompute-WU kernels on the same inputs."""

import argparse
import pathlib
import sys

import cutlass
import torch
from cutlass.cute.typing import Int32

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cula.cudac as cula_cuda
from benchmarks.bench_recompute_wu import prepare_recompute_wu_inputs
from benchmarks.utils import relative_rms_error_rel_max_mean_abs_rhs, triton_bench_fn
from cula.ops.kda.sm100 import recompute_wu as recompute_wu_module
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd


def _run_cpp(k, v, beta, A, gk, cu_seqlens, chunk_indices):
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    kg = torch.empty_like(v)
    cula_cuda.recompute_w_u_cuda(
        k,
        v,
        beta,
        A,
        gk,
        cu_seqlens,
        chunk_indices,
        w,
        u,
        kg,
        A.shape[-1],
        None,
        None,
    )
    return w, u, None, kg


def _make_cpp_launcher(k, v, beta, A, gk, cu_seqlens, chunk_indices):
    """Build an allocation-free launcher for measuring the kernel boundary."""
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    kg = torch.empty_like(v)

    def run():
        cula_cuda.recompute_w_u_cuda(
            k,
            v,
            beta,
            A,
            gk,
            cu_seqlens,
            chunk_indices,
            w,
            u,
            kg,
            A.shape[-1],
            None,
            None,
        )

    return run, (w, u, None, kg)


def _make_cutedsl_launcher(k, v, beta, A, gk):
    """Build an allocation-free launcher for measuring CuTeDSL dispatch."""
    B, T, H_qk, K = k.shape
    H_v = v.shape[2]
    V = v.shape[3]
    BT = A.shape[-1]
    total_nt = B * ((T + BT - 1) // BT)
    w = torch.empty((B, T, H_v, K), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    kg = torch.empty_like(w)
    cu = torch.zeros(2, dtype=torch.int32, device=k.device)
    ci = torch.zeros(2, dtype=torch.int32, device=k.device)
    compiled = recompute_wu_module._compile_recompute_wu(
        H_qk,
        H_v,
        K,
        V,
        chunk_size=BT,
        beta_dtype=cutlass.Float32 if beta.dtype == torch.float32 else cutlass.BFloat16,
    )
    problem_size = (Int32(B), Int32(T), Int32(H_qk), Int32(H_v), Int32(K), Int32(V))

    def run():
        compiled(k, v, beta, A, gk, w, u, kg, cu, ci, problem_size, Int32(total_nt))

    def run_pooled():
        run()
        return w.view_as(w), u.view_as(u), None, kg.view_as(kg)

    return run, run_pooled, (w, u, None, kg)


def _max_error(ref, out):
    stats = [relative_rms_error_rel_max_mean_abs_rhs(a, b) for a, b in zip(ref, out) if a is not None]
    return tuple(max(values) for values in zip(*stats))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--beta-bf16", action="store_true")
    parser.add_argument(
        "--force-varlen",
        action="store_true",
        help="Keep packed uniform inputs on the varlen kernel path (diagnostic only)",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 4096, 8192, 16384, 32768])
    parser.add_argument(
        "--profile",
        choices=("cpp", "ws"),
        help="Warm up, then launch exactly one selected kernel between CUDA profiler markers",
    )
    args = parser.parse_args()

    if args.force_varlen:
        recompute_wu_module._uniform_problem = lambda _cu_seqlens: None

    import benchmarks.bench_recompute_wu as common

    common.H = args.heads
    common.HV = args.heads
    device = torch.device("cuda")
    if args.profile:
        if len(args.lengths) != 1:
            parser.error("--profile requires exactly one value in --lengths")
        T = args.lengths[0]
        cu_seqlens = torch.tensor([0, T, 2 * T], dtype=torch.int32, device=device)
        _q, k, v, cu_gk, beta, A, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(
            2,
            T,
            device,
            cu_seqlens=cu_seqlens,
        )
        if args.beta_bf16:
            beta = beta.bfloat16()
        runners = {
            "cpp": lambda: _run_cpp(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices),
            "ws": lambda: recompute_w_u_fwd(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices),
        }
        runner = runners[args.profile]
        runner()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        runner()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(f"profiled {args.profile} at T={T}, H={args.heads}")
        return

    print(
        f"{'T':>8} {'C++ core':>10} {'DSL core':>10} {'core x':>8} "
        f"{'pooled':>10} {'C++ e2e':>10} {'DSL e2e':>10} {'e2e x':>8} {'rel_rmse':>12}"
    )
    for T in args.lengths:
        cu_seqlens = torch.tensor([0, T, 2 * T], dtype=torch.int32, device=device)
        _q, k, v, cu_gk, beta, A, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(2, T, device, cu_seqlens=cu_seqlens)
        if args.beta_bf16:
            beta = beta.bfloat16()
        cpp = _run_cpp(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices)
        ws = recompute_w_u_fwd(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices)
        ws_err = _max_error(cpp, ws)[0]
        if not torch.isfinite(torch.tensor(ws_err)):
            raise AssertionError(f"non-finite error at T={T}: ws={ws_err}")

        cpp_core, _ = _make_cpp_launcher(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices)
        dsl_core, dsl_pooled, _ = _make_cutedsl_launcher(k, v, beta, A, cu_gk)
        cpp_core_ms = triton_bench_fn(cpp_core)
        dsl_core_ms = triton_bench_fn(dsl_core)
        dsl_pooled_ms = triton_bench_fn(dsl_pooled)
        cpp_e2e_ms = triton_bench_fn(lambda: _run_cpp(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices))
        dsl_e2e_ms = triton_bench_fn(lambda: recompute_w_u_fwd(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices))
        print(
            f"{T:8d} {cpp_core_ms:10.4f} {dsl_core_ms:10.4f} {cpp_core_ms / dsl_core_ms:8.3f} "
            f"{dsl_pooled_ms:10.4f} {cpp_e2e_ms:10.4f} {dsl_e2e_ms:10.4f} "
            f"{cpp_e2e_ms / dsl_e2e_ms:8.3f} {ws_err:12.6g}"
        )


if __name__ == "__main__":
    main()
