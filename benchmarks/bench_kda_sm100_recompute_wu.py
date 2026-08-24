# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare SM100 C++ and CuTe DSL recompute-WU kernels on the same inputs."""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cula.cudac as cula_cuda
from benchmarks.bench_recompute_wu import prepare_recompute_wu_inputs
from benchmarks.utils import relative_rms_error_rel_max_mean_abs_rhs, triton_bench_fn
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd
from cula.ops.kda.sm100.recompute_wu_occ import recompute_w_u_fwd_occ


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


def _run_occ(k, v, beta, A, gk, cu_seqlens):
    offsets = cu_seqlens.cpu().tolist()
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if not lengths or len(set(lengths)) != 1:
        raise ValueError("The high-occupancy diagnostic kernel requires uniform sequences")
    B, T = len(lengths), lengths[0]
    args = [x.view(B, T, *x.shape[2:]) for x in (k, v, beta, A, gk)]
    outputs = recompute_w_u_fwd_occ(*args)
    return tuple(x.flatten(0, 1).unsqueeze(0) if x is not None else None for x in outputs)


def _max_error(ref, out):
    stats = [relative_rms_error_rel_max_mean_abs_rhs(a, b) for a, b in zip(ref, out) if a is not None]
    return tuple(max(values) for values in zip(*stats))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--beta-bf16", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 4096, 8192, 16384, 32768])
    parser.add_argument(
        "--profile",
        choices=("cpp", "ws", "occ"),
        help="Warm up, then launch exactly one selected kernel between CUDA profiler markers",
    )
    args = parser.parse_args()

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
            "occ": lambda: _run_occ(k, v, beta, A, cu_gk, cu_seqlens),
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

    print(f"{'T':>8} {'C++ (ms)':>12} {'WS (ms)':>12} {'Occ (ms)':>12} {'WS/C++':>10} {'Occ/C++':>10} {'rel_rmse':>12}")
    for T in args.lengths:
        cu_seqlens = torch.tensor([0, T, 2 * T], dtype=torch.int32, device=device)
        _q, k, v, cu_gk, beta, A, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(2, T, device, cu_seqlens=cu_seqlens)
        if args.beta_bf16:
            beta = beta.bfloat16()
        cpp = _run_cpp(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices)
        ws = recompute_w_u_fwd(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices)
        occ = _run_occ(k, v, beta, A, cu_gk, cu_seqlens)
        ws_err = _max_error(cpp, ws)[0]
        occ_err = _max_error(cpp, occ)[0]
        if not torch.isfinite(torch.tensor([ws_err, occ_err])).all():
            raise AssertionError(f"non-finite error at T={T}: ws={ws_err}, occ={occ_err}")

        cpp_ms = triton_bench_fn(lambda: _run_cpp(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices))
        ws_ms = triton_bench_fn(lambda: recompute_w_u_fwd(k, v, beta, A, cu_gk, cu_seqlens, chunk_indices))
        occ_ms = triton_bench_fn(lambda: _run_occ(k, v, beta, A, cu_gk, cu_seqlens))
        print(
            f"{T:8d} {cpp_ms:12.4f} {ws_ms:12.4f} {occ_ms:12.4f} "
            f"{cpp_ms / ws_ms:10.3f} {cpp_ms / occ_ms:10.3f} {max(ws_err, occ_err):12.6g}"
        )


if __name__ == "__main__":
    main()
