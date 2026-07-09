"""Sweep bench: fused KDA conv + MTP verify (cuLA auto) vs Triton, over T x N.

Real KDA shape: H == HV (num_q_heads == num_v_heads). One H=HV per GPU:
  CUDA_VISIBLE_DEVICES=0 python sweep_conv_mtp.py --H 8   > sweep_h8.csv
  ...--H 16 / --H 32 / --H 64 on cards 1/2/3.

Timing is CUDA-graph based (see ``t_graph_us``): the launch is captured into a
graph and timed with CUDA events, so at small batch we measure GPU kernel time
instead of a flat host launch/dispatch overhead (the reason eager timing showed
a constant Triton cost across small N). Small batch (N<16) packs ``--graph-calls``
calls per graph to amortize the fixed per-replay overhead.

Fair Triton timing: by default each (T,N) Triton point is measured in a FRESH
subprocess with an emptied, dedicated TRITON_CACHE_DIR, so the kernel compiles
and autotunes from scratch per config (no stale on-disk artifact, no carried-over
in-process autotune). cuLA is measured in-process (its CuTe kernels JIT-compile
once and are reused). Pass --triton-inproc to skip the subprocess (faster, less
rigorous: shared cache, one process).

Buffers are pre-allocated once per (T,N) and reused, so the graph captures stable
memory. accept=full (the recurrent verify processes all T draft tokens).
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

import torch

W = 4


# --------------------------------------------------------------------------- #
# Timing: CUDA-graph capture (mirrors benchmarks/bench_kda_decode_mtp.py)
# --------------------------------------------------------------------------- #
def t_graph_us(fn, warmup_iters, rep, graph_calls=1):
    """Return us/call, timed by replaying a CUDA graph with CUDA events."""
    # Warmup on a side stream so JIT compile / Triton autotune (which launch and
    # sync) finish BEFORE capture; autotune inside capture would corrupt it.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup_iters):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(graph_calls):
            fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep / graph_calls * 1e3  # ms -> us


def load_triton():
    path = os.environ["KDA_FUSED_TRI_FILE"]
    spec = importlib.util.spec_from_file_location("t", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.fused_kda_conv_gating_verify


def make(N, T, H, HV, K, V, lb, dev="cuda"):
    torch.manual_seed(0)
    D = 2 * H * K + HV * V
    f32, bf16 = torch.float32, torch.bfloat16
    idx = torch.arange(N, device=dev, dtype=torch.int32)
    g = dict(
        mixed=(torch.randn(N * T, D, device=dev, dtype=f32) * 0.5).to(bf16),
        cw=torch.randn(D, W, device=dev, dtype=f32) * 0.3,
        cb=torch.randn(D, device=dev, dtype=f32) * 0.1,
        cs_native=torch.randn(N, W - 1, D, device=dev, dtype=f32) * 0.3,
        a=(torch.randn(N, T, HV, K, device=dev, dtype=f32) * 0.5).to(bf16),
        b=(torch.randn(N, T, HV, device=dev, dtype=f32) * 0.5).to(bf16),
        Al=-torch.rand(HV, device=dev, dtype=f32) * 2.0,
        dtb=torch.randn(HV, K, device=dev, dtype=f32) * 0.1,
        ssm=torch.randn(N, HV, V, K, device=dev, dtype=f32) * 0.01,
        idx=idx, D=D, N=N, T=T, H=H, HV=HV, K=K, V=V, lb=lb,
    )
    return g


def cula_call(g):
    from cula.ops.kda.decode.mtp_conv import kda_conv_decode_mtp_verify
    dev = "cuda"
    cs = g["cs_native"].permute(0, 2, 1).contiguous()
    cw = torch.zeros(g["N"], g["T"], g["D"], W - 1, device=dev, dtype=torch.float32)
    ist = torch.zeros(g["N"], g["T"], g["HV"], g["V"], g["K"], device=dev, dtype=torch.float32)
    oo = torch.empty(g["N"], g["T"], g["HV"], g["V"], device=dev, dtype=torch.bfloat16)
    scale = g["K"] ** -0.5
    return lambda: kda_conv_decode_mtp_verify(
        mixed_qkv=g["mixed"], conv_weight=g["cw"], conv_bias=g["cb"], conv_state=cs,
        conv_state_indices=g["idx"], intermediate_conv_window=cw, intermediate_state_indices=g["idx"],
        a=g["a"], b=g["b"], A_log=g["Al"], dt_bias=g["dtb"], ssm_states=g["ssm"], cache_indices=g["idx"],
        intermediate_states_buffer=ist, scale=scale, T=g["T"], num_q_heads=g["H"], num_v_heads=g["HV"],
        head_k_dim=g["K"], head_v_dim=g["V"], lower_bound=g["lb"], variant="auto", out=oo)


def tri_call(fn, g):
    dev = "cuda"
    cs = g["cs_native"].clone().transpose(-1, -2)
    cw = torch.zeros(g["N"], g["T"], g["D"], W - 1, device=dev, dtype=torch.float32)
    ist = torch.zeros(g["N"], g["T"], g["HV"], g["V"], g["K"], device=dev, dtype=torch.float32)
    ssm = g["ssm"].clone()
    scale = g["K"] ** -0.5
    return lambda: fn(
        mixed_qkv=g["mixed"], conv_weight=g["cw"], conv_bias=g["cb"], conv_state=cs,
        conv_state_indices=g["idx"], intermediate_conv_window=cw, intermediate_state_indices=g["idx"],
        a=g["a"], b=g["b"], A_log=g["Al"], dt_bias=g["dtb"], ssm_states=ssm, cache_indices=g["idx"],
        intermediate_states_buffer=ist, scale=scale, T=g["T"], num_q_heads=g["H"], num_v_heads=g["HV"],
        head_k_dim=g["K"], head_v_dim=g["V"], lower_bound=g["lb"], num_warps=4)


def _gc(N, graph_calls):
    return 1 if N >= 16 else graph_calls  # amortize fixed replay overhead at small batch


# --------------------------------------------------------------------------- #
# --triton-one: subprocess entry measuring ONE (T,N) Triton point
# --------------------------------------------------------------------------- #
def run_triton_one(args):
    H = HV = args.H
    lb = args.lower_bound if args.gate == "safe" else None
    tri = load_triton()
    g = make(args.N, args.T, H, HV, args.K, args.V, lb)
    us = t_graph_us(tri_call(tri, g), args.warmup, args.rep, _gc(args.N, args.graph_calls))
    print(f"TRITON_US={us:.4f}", flush=True)


def _triton_subproc(args, T, N, tri_cache):
    """Measure Triton for (T,N) in a fresh subprocess with an emptied cache dir."""
    shutil.rmtree(tri_cache, ignore_errors=True)
    os.makedirs(tri_cache, exist_ok=True)
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = tri_cache
    cmd = [
        sys.executable, os.path.abspath(__file__), "--triton-one",
        "--H", str(args.H), "--K", str(args.K), "--V", str(args.V),
        "--T", str(T), "--N", str(N), "--gate", args.gate,
        "--lower-bound", str(args.lower_bound),
        "--rep", str(args.rep), "--warmup", str(args.warmup),
        "--graph-calls", str(args.graph_calls),
    ]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("TRITON_US="):
            return float(line.split("=", 1)[1])
    sys.stderr.write(
        f"[triton subproc FAILED T={T} N={N}] rc={out.returncode}\n"
        f"--stdout--\n{out.stdout}\n--stderr(tail)--\n{out.stderr[-2000:]}\n"
    )
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, required=True, help="H==HV (real KDA)")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--Ns", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--gate", choices=["safe", "softplus"], default="safe")
    ap.add_argument("--lower-bound", type=float, default=-5.0)
    ap.add_argument("--rep", type=int, default=300, help="CUDA-graph replays timed")
    ap.add_argument("--warmup", type=int, default=15, help="warmup iters before capture")
    ap.add_argument("--graph-calls", type=int, default=20,
                    help="calls/graph when N<16 to amortize fixed replay overhead")
    ap.add_argument("--triton-cache-dir", default=None,
                    help="dedicated Triton cache dir, EMPTIED before each config's "
                         "Triton subprocess (default: <TRITON_CACHE_DIR>_convsweep). "
                         "Must not be the shared cache.")
    ap.add_argument("--triton-inproc", action="store_true",
                    help="measure Triton in-process (skip fresh-subprocess/cache-clear; "
                         "faster but less rigorous)")
    # internal (--triton-one subprocess):
    ap.add_argument("--triton-one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--N", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--T", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.triton_one:
        run_triton_one(args)
        return

    H = HV = args.H
    lb = args.lower_bound if args.gate == "safe" else None

    base = os.environ.get("TRITON_CACHE_DIR")
    tri_cache = args.triton_cache_dir or (
        (base + "_convsweep") if base else "/tmp/kda_conv_convsweep_tricache")

    tri_inproc = None
    if args.triton_inproc:
        tri_inproc = load_triton()

    print(f"# H=HV={H} K={args.K} V={args.V} gate={args.gate} accept=full")
    print(f"# timing=cuda-graph rep={args.rep} warmup={args.warmup} "
          f"graph_calls(N<16)={args.graph_calls}")
    if args.triton_inproc:
        print("# triton: in-process (shared cache, no per-config clear)")
    else:
        print(f"# triton: fresh subprocess + emptied cache per config -> {tri_cache}")
    print("T,N,triton_us,cula_us,speedup")

    for T in args.Ts:
        for N in args.Ns:
            g = make(N, T, H, HV, args.K, args.V, lb)
            gc = _gc(N, args.graph_calls)
            c_us = t_graph_us(cula_call(g), args.warmup, args.rep, gc)

            if T < W - 1:  # triton fused verify requires T >= W-1
                print(f"{T},{N},NA,{c_us:.1f},NA", flush=True)
                g = None
                torch.cuda.empty_cache()
                continue

            if args.triton_inproc:
                t_us = t_graph_us(tri_call(tri_inproc, g), args.warmup, args.rep, gc)
            else:
                g = None  # free parent GPU buffers before the Triton subprocess
                torch.cuda.empty_cache()
                t_us = _triton_subproc(args, T, N, tri_cache)

            sp = (t_us / c_us) if (c_us and t_us == t_us) else float("nan")
            print(f"{T},{N},{t_us:.1f},{c_us:.1f},{sp:.2f}", flush=True)
            g = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
