#!/usr/bin/env python3
"""
Benchmark KDA performance with pure backend isolation.

Compares:
- pure FLA backend
- pure cuLA backend

Important: cuLA backward path patches FLA bwd_intra at runtime. To keep comparison
"pure", this benchmark measures each backend in a separate subprocess.
"""

import argparse
import importlib
import json
import pathlib
import random
import subprocess
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fla.utils import assert_close

from benchmarks.utils import prepare_safe_gate_inputs


def gen_quasi_balanced_seqlens(total_tokens: int, num_seqs: int, max_ratio: float = 2.5, seed: int = 123) -> list[int]:
    rng = random.Random(seed)
    min_seq = 64
    weights = [rng.uniform(1.0, max_ratio) for _ in range(num_seqs)]
    w_sum = sum(weights)
    raw = [max(min_seq, int(w / w_sum * total_tokens)) for w in weights]
    diff = total_tokens - sum(raw)
    idxs = sorted(range(num_seqs), key=lambda i: raw[i], reverse=True)
    for i in range(abs(diff)):
        idx = idxs[i % num_seqs]
        raw[idx] += 1 if diff > 0 else -1
    return raw


def make_inputs(mode: str, h: int, d: int, dtype: torch.dtype, device: torch.device, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)

    if mode == "fixed":
        b, t = 2, 8192
        cu_seqlens = torch.arange(0, (b + 1) * t, t, device=device, dtype=torch.int32)
        inputs = prepare_safe_gate_inputs(
            batch_size=b,
            T=t,
            H=h,
            D=d,
            device=device,
            cu_seqlens=cu_seqlens,
            has_init_state=True,
            seed=seed,
        )
    elif mode == "varlen":
        total_t = 16384
        n = 20
        seqlens = gen_quasi_balanced_seqlens(total_t, n, seed=seed)
        cu_seqlens = torch.tensor([0] + [sum(seqlens[: i + 1]) for i in range(n)], dtype=torch.int32, device=device)
        inputs = prepare_safe_gate_inputs(
            batch_size=1,
            T=total_t,
            H=h,
            D=d,
            device=device,
            cu_seqlens=cu_seqlens,
            has_init_state=True,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
    h0 = inputs["init_state"]
    a_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
    scale = inputs["scale"]
    lower_bound = inputs["lower_bound"]

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    return q, k, v, g, beta, h0, cu_seqlens, do, dht, a_log, dt_bias, scale, lower_bound


def run_once(
    fn,
    q,
    k,
    v,
    g,
    beta,
    h0,
    cu_seqlens,
    do,
    dht,
    a_log,
    dt_bias,
    scale,
    lower_bound,
    phase,
):
    q.grad = None
    k.grad = None
    v.grad = None
    g.grad = None
    beta.grad = None
    h0.grad = None

    out, ht = fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    if phase == "e2e":
        loss = (out * do).sum() + (ht * dht).sum()
        loss.backward()
    torch.cuda.synchronize()


def correctness_check(fla_fn, cula_fn, inputs):
    q, k, v, g, beta, h0, cu_seqlens, do, dht, a_log, dt_bias, scale, lower_bound = inputs

    # FLA
    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)
    g1 = g.detach().clone().requires_grad_(True)
    b1 = beta.detach().clone().requires_grad_(True)
    h1 = h0.detach().clone().requires_grad_(True)
    o1, ht1 = fla_fn(
        q=q1,
        k=k1,
        v=v1,
        g=g1,
        beta=b1,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=h1,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    ((o1 * do).sum() + (ht1 * dht).sum()).backward()

    # cuLA
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    g2 = g.detach().clone().requires_grad_(True)
    b2 = beta.detach().clone().requires_grad_(True)
    h2 = h0.detach().clone().requires_grad_(True)
    o2, ht2 = cula_fn(
        q=q2,
        k=k2,
        v=v2,
        g=g2,
        beta=b2,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=h2,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    ((o2 * do).sum() + (ht2 * dht).sum()).backward()

    assert_close("o", o1, o2, 0.008)
    assert_close("ht", ht1, ht2, 0.008)
    assert_close("dq", q1.grad, q2.grad, 0.01)
    assert_close("dk", k1.grad, k2.grad, 0.01)
    assert_close("dv", v1.grad, v2.grad, 0.01)
    assert_close("dg", g1.grad, g2.grad, 0.03)
    assert_close("db", b1.grad, b2.grad, 0.03)


def measure_backend(backend: str, mode: str, h: int, d: int, warmup: int, rep: int, seed: int, phase: str):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if backend == "fla":
        from fla.ops.kda.chunk import chunk_kda as fn
    elif backend == "cula":
        from cula.kda import chunk_kda as fn
    else:
        raise ValueError(f"Unknown backend: {backend}")

    inputs = make_inputs(mode=mode, h=h, d=d, dtype=dtype, device=device, seed=seed)
    q0, k0, v0, g0, beta0, h00, cu_seqlens, do, dht, a_log, dt_bias, scale, lower_bound = inputs

    # Reuse the same leaf tensors across iterations to avoid benchmarking clone/alloc overhead.
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    g = g0.detach().clone().requires_grad_(True)
    beta = beta0.detach().clone().requires_grad_(True)
    h0 = h00.detach().clone().requires_grad_(True)

    def _bench_call():
        run_once(fn, q, k, v, g, beta, h0, cu_seqlens, do, dht, a_log, dt_bias, scale, lower_bound, phase)

    ms, _, _ = triton.testing.do_bench(_bench_call, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return {
        "backend": backend,
        "mode": mode,
        "h": h,
        "d": d,
        "phase": phase,
        "ms": float(ms),
    }


def run_subprocess(backend: str, mode: str, h: int, d: int, warmup: int, rep: int, seed: int, phase: str) -> dict:
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--backend",
        backend,
        "--mode",
        mode,
        "--heads",
        str(h),
        "--dim",
        str(d),
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
        "--seed",
        str(seed),
        "--phase",
        phase,
        "--json",
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description="Benchmark pure FLA vs pure cuLA KDA with bench_kda-aligned settings")
    parser.add_argument("--backend", choices=["fla", "cula"], default=None)
    parser.add_argument("--mode", choices=["fixed", "varlen"], default="varlen")
    parser.add_argument("--phase", choices=["forward", "e2e"], default="e2e")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    args = parser.parse_args()

    if args.backend is not None:
        result = measure_backend(args.backend, args.mode, args.heads, args.dim, args.warmup, args.rep, args.seed, args.phase)
        if args.json:
            print(json.dumps(result, ensure_ascii=True))
        else:
            print(result)
        return

    if args.check_correctness:
        # correctness check in isolated parent process before timing subprocesses
        cula_chunk_kda = importlib.import_module("cula.kda").chunk_kda
        fla_chunk_kda = importlib.import_module("fla.ops.kda.chunk").chunk_kda

        inputs = make_inputs(
            mode=args.mode,
            h=args.heads,
            d=args.dim,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            seed=args.seed,
        )
        correctness_check(fla_chunk_kda, cula_chunk_kda, inputs)
        print("Correctness check: PASS")

    fla_res = run_subprocess("fla", args.mode, args.heads, args.dim, args.warmup, args.rep, args.seed, args.phase)
    cula_res = run_subprocess("cula", args.mode, args.heads, args.dim, args.warmup, args.rep, args.seed, args.phase)

    fla_ms = fla_res["ms"]
    cula_ms = cula_res["ms"]
    speedup = fla_ms / cula_ms if cula_ms > 0 else float("inf")

    print(f"mode={args.mode} phase={args.phase} H={args.heads} D={args.dim} warmup={args.warmup} rep={args.rep}")
    print(f"FLA (pure) : {fla_ms:.3f} ms")
    print(f"cuLA (pure): {cula_ms:.3f} ms")
    print(f"Speedup (FLA/cuLA): {speedup:.3f}x")


if __name__ == "__main__":
    main()
