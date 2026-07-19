"""Benchmark fused KDA conv + MTP verify against a torch reference.

Usage:
  python benchmarks/bench_fused_kda_conv_mtp.py --N 128 --T 4 --H 16 --HV 16
  python benchmarks/bench_fused_kda_conv_mtp.py --sweep --H 16 --HV 16

Set ``KDA_FUSED_TRI_FILE`` to benchmark a compatible Triton implementation:
  KDA_FUSED_TRI_FILE=/path/to/compatible_triton.py \
    python benchmarks/bench_fused_kda_conv_mtp.py --sweep --which both
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

import torch

W = 4  # conv width (KDA short_conv_kernel_size); fused paths hard-assume 4


# CUDA Graph timing
def t_graph_us(fn, warmup_iters, rep, graph_calls=1):
    """Measure microseconds per call with CUDA Graph replay."""
    # Complete compilation and autotuning before capture.
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


def clear_triton_cache():
    """Clear the configured Triton cache directory."""
    d = os.environ.get("TRITON_CACHE_DIR")
    if not d:
        print("  [warn] TRITON_CACHE_DIR unset; --rm-triton-cache is a no-op")
        return
    if os.path.isdir(d):
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
    print(f"  [cache] cleared Triton cache dir: {d}")


# Optional Triton loader
def load_triton_fused():
    path = os.environ.get("KDA_FUSED_TRI_FILE")
    if not path or not os.path.exists(path):
        raise RuntimeError(
            f"KDA_FUSED_TRI_FILE not set or missing: {path!r}. "
            "Point it at a compatible standalone Triton implementation."
        )
    spec = importlib.util.spec_from_file_location("kda_fused_tri", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_kda_conv_gating_verify


# Input generation
def make_inputs(N, T, H, HV, K, V, pool_size, gate, seed, device="cuda"):
    torch.manual_seed(seed)
    D = 2 * H * K + HV * V
    f32 = torch.float32
    bf16 = torch.bfloat16

    mixed_qkv = (torch.randn(N * T, D, device=device, dtype=f32) * 0.5).to(bf16)
    conv_weight = torch.randn(D, W, device=device, dtype=f32) * 0.3
    conv_bias = torch.randn(D, device=device, dtype=f32) * 0.1
    # conv_state native [lines, W-1, D]; row 0 = oldest history token.
    conv_state_native = torch.randn(pool_size, W - 1, D, device=device, dtype=f32) * 0.3

    a = (torch.randn(N, T, HV, K, device=device, dtype=f32) * 0.5).to(bf16)
    b = (torch.randn(N, T, HV, device=device, dtype=f32) * 0.5).to(bf16)
    A_log = -torch.rand(HV, device=device, dtype=f32) * 2.0  # A = exp(A_log) < 1
    dt_bias = torch.randn(HV, K, device=device, dtype=f32) * 0.1
    ssm_states = torch.randn(pool_size, HV, V, K, device=device, dtype=f32) * 0.01

    # one distinct mamba slot per request (chain verify: cache_indices[:N])
    cache_indices = torch.arange(N, device=device, dtype=torch.int32)

    return dict(
        D=D,
        mixed_qkv=mixed_qkv,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_state_native=conv_state_native,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        gate=gate,
    )


# Torch reference
def torch_reference(inp, N, T, H, HV, K, V, scale, lower_bound,
                    softplus_beta=1.0, softplus_threshold=20.0):
    dev = inp["mixed_qkv"].device
    D = inp["D"]
    mixed = inp["mixed_qkv"].float()             # [N*T, D]
    w = inp["conv_weight"]                        # [D, W]
    bias = inp["conv_bias"]                       # [D]
    hist0 = inp["conv_state_native"].float()      # [lines, W-1, D]
    a = inp["a"].float()
    b = inp["b"].float()
    A_log = inp["A_log"]
    dt_bias = inp["dt_bias"]
    ssm0 = inp["ssm_states"].float()              # [slots, HV, V, K]
    cidx = inp["cache_indices"]

    qk_dim = H * K
    o = torch.zeros(N, T, HV, V, device=dev, dtype=torch.float32)
    ssm_snap = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.float32)   # per-step S
    win_snap = torch.zeros(N, T, W - 1, D, device=dev, dtype=torch.float32)   # per-step conv window (raw input)
    conv_state_out = torch.zeros_like(hist0)

    for n in range(N):
        slot = int(cidx[n].item())
        x = mixed[n * T:(n + 1) * T]              # [T, D]
        hist = hist0[slot]                        # [W-1=3, D] oldest..newest
        xfull = torch.cat([hist, x], dim=0)       # [3+T, D]

        # depthwise causal conv (width 4) + SiLU + bf16 round-trip
        y = torch.zeros(T, D, device=dev, dtype=torch.float32)
        for t in range(T):
            acc = bias.clone()
            for j in range(W):
                acc = acc + w[:, j] * xfull[t + j]
            y[t] = torch.nn.functional.silu(acc)
            # window snapshot after consuming token t = xfull[t+1 .. t+3] (last 3)
            win_snap[n, t] = xfull[t + 1:t + 1 + (W - 1)]
        y = y.to(torch.bfloat16).float()          # round-trip

        # rolled conv_state after all T tokens = last W-1 raw inputs (all-accept temp)
        conv_state_out[slot] = xfull[-(W - 1):]

        qy = y[:, 0:qk_dim].view(T, H, K)
        ky = y[:, qk_dim:2 * qk_dim].view(T, H, K)
        vy = y[:, 2 * qk_dim:2 * qk_dim + HV * V].view(T, HV, V)

        for hv in range(HV):
            ih = hv // (HV // H)
            S = ssm0[slot, hv].clone()            # [V, K]
            eA = torch.exp(A_log[hv])
            for t in range(T):
                qb = qy[t, ih].clone()
                kb = ky[t, ih].clone()
                vb = vy[t, hv].clone()
                gx = a[n, t, hv] + dt_bias[hv]    # [K]
                if lower_bound is not None:
                    g = lower_bound * torch.sigmoid(eA * gx)
                else:
                    beta_x = softplus_beta * gx
                    sp = torch.where(beta_x <= softplus_threshold,
                                     (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
                                     gx)
                    g = -eA * sp
                beta = torch.sigmoid(b[n, t, hv])
                qb = qb / torch.sqrt((qb * qb).sum() + 1e-6) * scale
                kb = kb / torch.sqrt((kb * kb).sum() + 1e-6)
                S = S * torch.exp(g)[None, :]
                sv = S @ kb                        # [V]
                v_new = (vb - sv) * beta
                S = S + v_new[:, None] * kb[None, :]
                o[n, t, hv] = S @ qb
                ssm_snap[n, t, hv] = S

    return dict(o=o, ssm_snap=ssm_snap, win_snap=win_snap, conv_state_out=conv_state_out)


# Triton runner
def run_triton(fn, inp, N, T, H, HV, K, V, scale, lower_bound, num_warps=4):
    D = inp["D"]
    dev = inp["mixed_qkv"].device
    # Triton expects writable [lines, D, W-1] with stride(1) == 1.
    conv_state = inp["conv_state_native"].clone().transpose(-1, -2)   # [lines, D, W-1]
    assert conv_state.stride(1) == 1
    ssm = inp["ssm_states"].clone()
    # Dense [lines, steps, D, W-1] snapshots.
    conv_window = torch.zeros(N, T, D, W - 1, device=dev, dtype=torch.float32)
    inter_state_indices = inp["cache_indices"].clone()
    # SSM snapshots with contiguous K.
    inter_states = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.float32)

    out = fn(
        mixed_qkv=inp["mixed_qkv"],
        conv_weight=inp["conv_weight"],
        conv_bias=inp["conv_bias"],
        conv_state=conv_state,
        conv_state_indices=inp["cache_indices"],
        intermediate_conv_window=conv_window,
        intermediate_state_indices=inter_state_indices,
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        intermediate_states_buffer=inter_states,
        scale=scale,
        T=T,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        lower_bound=lower_bound,
        num_warps=num_warps,
    )
    return dict(o=out.view(N, T, HV, V), conv_state=conv_state, conv_window=conv_window,
                inter_states=inter_states)


# cuLA runner
def run_cula(inp, N, T, H, HV, K, V, scale, lower_bound, bv=-1, variant="auto"):
    from cula.ops.kda.decode.mtp_conv import kda_conv_decode_mtp_verify
    D = inp["D"]
    dev = inp["mixed_qkv"].device
    # cuLA conv_state: contiguous [lines, D, W-1], value[line,ch,w] = cs_native[line,w,ch]
    conv_state = inp["conv_state_native"].permute(0, 2, 1).contiguous()  # [lines, D, W-1]
    ssm = inp["ssm_states"].clone()
    conv_window = torch.zeros(N, T, D, W - 1, device=dev, dtype=torch.float32)
    inter_state_indices = inp["cache_indices"].clone()
    inter_states = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.float32)

    o = kda_conv_decode_mtp_verify(
        mixed_qkv=inp["mixed_qkv"],
        conv_weight=inp["conv_weight"],
        conv_bias=inp["conv_bias"],
        conv_state=conv_state,
        conv_state_indices=inp["cache_indices"],
        intermediate_conv_window=conv_window,
        intermediate_state_indices=inter_state_indices,
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        ssm_states=ssm,
        cache_indices=inp["cache_indices"],
        intermediate_states_buffer=inter_states,
        scale=scale,
        T=T,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        lower_bound=lower_bound,
        bv=bv,
        variant=variant,
    )
    return dict(o=o.view(N, T, HV, V), conv_state=conv_state, conv_window=conv_window,
                inter_states=inter_states)


def make_triton_bench_call(fn, inp, N, T, H, HV, K, V, scale, lower_bound):
    """Build a Triton call with stable buffers for CUDA Graph capture."""
    D = inp["D"]
    dev = inp["mixed_qkv"].device
    cs = inp["conv_state_native"].clone().transpose(-1, -2)
    cw = torch.zeros(N, T, D, W - 1, device=dev, dtype=torch.float32)
    istate = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.float32)
    ssm = inp["ssm_states"].clone()
    idx = inp["cache_indices"]

    return lambda: fn(
        mixed_qkv=inp["mixed_qkv"], conv_weight=inp["conv_weight"],
        conv_bias=inp["conv_bias"], conv_state=cs, conv_state_indices=idx,
        intermediate_conv_window=cw, intermediate_state_indices=idx, a=inp["a"],
        b=inp["b"], A_log=inp["A_log"], dt_bias=inp["dt_bias"], ssm_states=ssm,
        cache_indices=idx, intermediate_states_buffer=istate, scale=scale, T=T,
        num_q_heads=H, num_v_heads=HV, head_k_dim=K, head_v_dim=V,
        lower_bound=lower_bound, num_warps=4,
    )


def make_cula_bench_call(inp, N, T, H, HV, K, V, scale, lower_bound,
                         bv=-1, bvw=-1, variant="auto"):
    """Build a cuLA call with stable buffers for CUDA Graph capture."""
    from cula.ops.kda.decode.mtp_conv import kda_conv_decode_mtp_verify

    D = inp["D"]
    dev = inp["mixed_qkv"].device
    cs = inp["conv_state_native"].permute(0, 2, 1).contiguous()
    cw = torch.zeros(N, T, D, W - 1, device=dev, dtype=torch.float32)
    istate = torch.zeros(N, T, HV, V, K, device=dev, dtype=torch.float32)
    ssm = inp["ssm_states"].clone()
    idx = inp["cache_indices"]
    out = torch.empty(N, T, HV, V, device=dev, dtype=torch.bfloat16)

    return lambda: kda_conv_decode_mtp_verify(
        mixed_qkv=inp["mixed_qkv"], conv_weight=inp["conv_weight"],
        conv_bias=inp["conv_bias"], conv_state=cs, conv_state_indices=idx,
        intermediate_conv_window=cw, intermediate_state_indices=idx, a=inp["a"],
        b=inp["b"], A_log=inp["A_log"], dt_bias=inp["dt_bias"], ssm_states=ssm,
        cache_indices=idx, intermediate_states_buffer=istate, scale=scale, T=T,
        num_q_heads=H, num_v_heads=HV, head_k_dim=K, head_v_dim=V,
        lower_bound=lower_bound, bv=bv, bvw=bvw, variant=variant, out=out,
    )


def _graph_calls(N, graph_calls):
    return 1 if N >= 16 else graph_calls


def run_triton_one(args):
    """Subprocess entry for one fresh-cache Triton point."""
    lower_bound = args.lower_bound if args.gate == "safe" else None
    inp = make_inputs(
        args.N, args.T, args.H, args.HV, args.K, args.V, args.N,
        args.gate, args.seed,
    )
    fn = load_triton_fused()
    call = make_triton_bench_call(
        fn, inp, args.N, args.T, args.H, args.HV, args.K, args.V,
        args.K ** -0.5, lower_bound,
    )
    us = t_graph_us(
        call, args.warmup, args.rep, _graph_calls(args.N, args.graph_calls)
    )
    print(f"TRITON_US={us:.4f}", flush=True)


def _triton_subproc(args, T, N, cache_dir):
    """Measure one Triton point in a fresh process and dedicated cache."""
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = cache_dir
    cmd = [
        sys.executable, os.path.abspath(__file__), "--triton-one",
        "--N", str(N), "--T", str(T), "--H", str(args.H),
        "--HV", str(args.HV), "--K", str(args.K), "--V", str(args.V),
        "--gate", args.gate, "--lower-bound", str(args.lower_bound),
        "--seed", str(args.seed), "--rep", str(args.rep),
        "--warmup", str(args.warmup), "--graph-calls", str(args.graph_calls),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("TRITON_US="):
            return float(line.split("=", 1)[1])
    sys.stderr.write(
        f"[triton subprocess failed T={T} N={N}] rc={proc.returncode}\n"
        f"{proc.stdout}\n{proc.stderr[-2000:]}\n"
    )
    return float("nan")


def run_sweep(args):
    """Sweep the same benchmark over all requested T and batch sizes."""
    lower_bound = args.lower_bound if args.gate == "safe" else None
    want_cula = args.which in ("cula", "both")
    want_triton = args.which in ("triton", "both")
    tri_fn = load_triton_fused() if want_triton and args.triton_inproc else None
    base_cache = os.environ.get("TRITON_CACHE_DIR")
    tri_cache = args.triton_cache_dir or (
        f"{base_cache}_convsweep" if base_cache else "/tmp/kda_conv_convsweep_tricache"
    )

    print(f"# H={args.H} HV={args.HV} K={args.K} V={args.V} gate={args.gate}")
    print(f"# timing=cuda-graph rep={args.rep} warmup={args.warmup} "
          f"graph_calls(N<16)={args.graph_calls}")
    if want_triton:
        mode = "in-process" if args.triton_inproc else f"fresh process/cache {tri_cache}"
        print(f"# optional Triton baseline: {mode}")
    print("T,N,triton_us,cula_us,speedup")

    for T in args.Ts:
        for N in args.batch_sizes:
            inp = make_inputs(
                N, T, args.H, args.HV, args.K, args.V, N, args.gate, args.seed,
            )
            gc = _graph_calls(N, args.graph_calls)
            cula_us = float("nan")
            if want_cula:
                call = make_cula_bench_call(
                    inp, N, T, args.H, args.HV, args.K, args.V,
                    args.K ** -0.5, lower_bound, args.bv, args.bvw, args.variant,
                )
                cula_us = t_graph_us(call, args.warmup, args.rep, gc)

            triton_us = float("nan")
            if want_triton and T >= W - 1:
                if args.triton_inproc:
                    call = make_triton_bench_call(
                        tri_fn, inp, N, T, args.H, args.HV, args.K, args.V,
                        args.K ** -0.5, lower_bound,
                    )
                    triton_us = t_graph_us(call, args.warmup, args.rep, gc)
                else:
                    call = None
                    inp = None
                    torch.cuda.empty_cache()
                    triton_us = _triton_subproc(args, T, N, tri_cache)

            speedup = triton_us / cula_us if triton_us == triton_us and cula_us == cula_us else float("nan")
            tri_text = f"{triton_us:.1f}" if triton_us == triton_us else "NA"
            cula_text = f"{cula_us:.1f}" if cula_us == cula_us else "NA"
            speed_text = f"{speedup:.2f}" if speedup == speedup else "NA"
            print(f"{T},{N},{tri_text},{cula_text},{speed_text}", flush=True)
            inp = None
            torch.cuda.empty_cache()


# Compare helper
def report(name, ref, act, atol, rtol):
    ref = ref.float()
    act = act.float()
    diff = (ref - act).abs()
    denom = ref.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    ok = torch.allclose(ref, act, atol=atol, rtol=rtol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="sweep all --Ts x --batch-sizes")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--H", type=int, default=32)   # real KDA: num_q_heads==num_v_heads (H==HV); 32=full, 8=TP4
    ap.add_argument("--HV", type=int, default=32)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--gate", choices=["safe", "softplus"], default="safe")
    ap.add_argument("--lower-bound", type=float, default=-5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="run torch ref + correctness")
    ap.add_argument("--atol", type=float, default=3e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--rep", type=int, default=300, help="CUDA-graph replays timed")
    ap.add_argument("--warmup", type=int, default=15, help="warmup iters before capture")
    ap.add_argument("--graph-calls", type=int, default=20,
                    help="calls packed per CUDA graph when N<16 to amortize fixed "
                         "replay overhead at small batch (N>=16 uses 1)")
    ap.add_argument("--rm-triton-cache", action="store_true",
                    help="clear TRITON_CACHE_DIR before loading Triton (fresh "
                         "compile + autotune; use a dedicated cache dir)")
    ap.add_argument("--triton-cache-dir", default=None,
                    help="dedicated cache emptied before each Triton sweep point")
    ap.add_argument("--triton-inproc", action="store_true",
                    help="measure the optional Triton baseline in process")
    ap.add_argument("--which", choices=["triton", "cula", "both"], default="cula")
    ap.add_argument("--bv", type=int, default=-1, help="cuLA small_batch v-tile size (8/16/32; -1=auto)")
    ap.add_argument("--bvw", type=int, default=-1, help="cuLA large_batch v-cols/warp; -1=auto")
    ap.add_argument("--variant", choices=["auto", "small_batch", "large_batch"], default="auto")
    ap.add_argument("--triton-one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.triton_one:
        run_triton_one(args)
        return
    if args.sweep:
        run_sweep(args)
        return

    if args.rm_triton_cache and args.which in ("triton", "both"):
        clear_triton_cache()

    dev = "cuda"
    scale = args.K ** -0.5
    lower_bound = args.lower_bound if args.gate == "safe" else None
    pool_size = args.N
    print(f"config: N={args.N} T={args.T} H={args.H} HV={args.HV} K={args.K} V={args.V} "
          f"gate={args.gate} D={2*args.H*args.K + args.HV*args.V}")

    inp = make_inputs(args.N, args.T, args.H, args.HV, args.K, args.V, pool_size,
                      args.gate, args.seed, dev)

    tri_fn = None
    if args.which in ("triton", "both"):
        tri_fn = load_triton_fused()

    if args.check:
        ref = torch_reference(inp, args.N, args.T, args.H, args.HV, args.K, args.V,
                              scale, lower_bound)
        print("torch reference computed.")
        if tri_fn is not None:
            tri = run_triton(tri_fn, inp, args.N, args.T, args.H, args.HV, args.K, args.V,
                             scale, lower_bound, num_warps=4)
            print("triton vs torch-ref:")
            report("o", ref["o"], tri["o"], args.atol, args.rtol)
            report("inter_ssm", ref["ssm_snap"], tri["inter_states"], args.atol, args.rtol)
            report("conv_window", ref["win_snap"],
                   tri["conv_window"].transpose(-1, -2), args.atol, args.rtol)
            report("conv_state", ref["conv_state_out"],
                   tri["conv_state"].transpose(-1, -2), args.atol, args.rtol)
        if args.which in ("cula", "both"):
            cula = run_cula(inp, args.N, args.T, args.H, args.HV, args.K, args.V,
                            scale, lower_bound, bv=args.bv, variant=args.variant)
            print("cuLA vs torch-ref:")
            report("o", ref["o"], cula["o"], args.atol, args.rtol)
            report("inter_ssm", ref["ssm_snap"], cula["inter_states"], args.atol, args.rtol)
            report("conv_window", ref["win_snap"],
                   cula["conv_window"].transpose(-1, -2), args.atol, args.rtol)
            report("conv_state", ref["conv_state_out"],
                   cula["conv_state"].transpose(-1, -2), args.atol, args.rtol)

    # Reuse stable buffers during CUDA Graph replay.
    N, T, H, HV, K, V = args.N, args.T, args.H, args.HV, args.K, args.V
    gc = 1 if N >= 16 else args.graph_calls  # amortize replay overhead at small batch

    def bench(fn_run):
        return t_graph_us(fn_run, args.warmup, args.rep, gc)

    if tri_fn is not None:
        call = make_triton_bench_call(
            tri_fn, inp, N, T, H, HV, K, V, scale, lower_bound,
        )
        print(f"triton fused: {bench(call):.2f} us  (graph, rep={args.rep} gc={gc})")

    if args.which in ("cula", "both"):
        call = make_cula_bench_call(
            inp, N, T, H, HV, K, V, scale, lower_bound,
            args.bv, args.bvw, args.variant,
        )
        print(f"cuLA fused(var={args.variant},bv={args.bv}): {bench(call):.2f} us  "
              f"(graph, rep={args.rep} gc={gc})")


if __name__ == "__main__":
    main()
