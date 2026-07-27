"""Benchmark fused causal-conv1d + KDA MTP verify and full commit chains.

The unified benchmark covers Triton recurrent verify, cuLA small/large-batch
recurrent verify, fused KVBuffer verify, and their matching commit/flush path.
Timing uses CUDA Graphs and events; host overhead, compilation, allocation and
graph capture are excluded.
"""

import argparse
import csv
import importlib.util
import os
import pathlib
import shutil
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.ops.kda.decode.mtp_conv import kda_conv_decode_mtp_verify
from cula.ops.kda.decode.mtp_conv_kvbuffer import kda_conv_decode_mtp_kvbuffer
from cula.ops.kda.decode.mtp_kvbuffer import (
    kda_decode_mtp_tensor_core_kvbuffer,
    kda_flush_kvbuffer,
)

W = 4
K = 128
V = 128


def load_file_attr(path, attr, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


def load_triton_fused():
    path = os.environ.get("KDA_FUSED_TRI_FILE")
    if not path:
        raise RuntimeError("set KDA_FUSED_TRI_FILE to a compatible Triton fused conv MTP module")
    return load_file_attr(path, "fused_kda_conv_gating_verify", "_compatible_fused_kda_conv")


def load_scatter_commit():
    path = os.environ.get("KDA_SCATTER_FILE")
    if not path:
        raise RuntimeError("set KDA_SCATTER_FILE to a compatible state-commit module")
    return load_file_attr(path, "fused_mamba_state_scatter_with_mask", "_compatible_mamba_scatter")


def load_causal_conv_update():
    path = os.environ.get("KDA_CAUSAL_CONV_FILE")
    if not path:
        raise RuntimeError("set KDA_CAUSAL_CONV_FILE to a compatible causal-conv module")
    return load_file_attr(path, "causal_conv1d_update", "_compatible_causal_conv")


def clear_triton_cache():
    path = os.environ.get("TRITON_CACHE_DIR")
    if not path:
        raise RuntimeError("--rm-triton-cache requires TRITON_CACHE_DIR")
    root = pathlib.Path(path)
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    print(f"cleared Triton cache: {root}")


def graph_time_us(fn, *, warmup=20, rep=200, graph_calls=10):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_calls):
            fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / rep / graph_calls


def make_inputs(N, T, H, HV, seed=0):
    torch.manual_seed(seed)
    D = 2 * H * K + HV * V
    mixed = (torch.randn(N, T, D, device="cuda") * 0.5).to(torch.bfloat16)
    return {
        "mixed": mixed,
        "weight": torch.randn(D, W, device="cuda") * 0.3,
        "bias": torch.randn(D, device="cuda") * 0.1,
        "conv_state": torch.randn(N, D, W - 1, device="cuda") * 0.3,
        "a": (torch.randn(N, T, HV, K, device="cuda") * 0.5).to(torch.bfloat16),
        "b": (torch.randn(N, T, HV, device="cuda") * 0.5).to(torch.bfloat16),
        "A_log": -torch.rand(HV, device="cuda") * 2.0,
        "dt_bias": torch.randn(HV, K, device="cuda") * 0.1,
        "ssm": torch.randn(N, HV, V, K, device="cuda") * 0.01,
        "indices": torch.arange(N, device="cuda", dtype=torch.int32),
        "D": D,
    }


def alloc_ubufs(N, T, HV):
    return (
        torch.empty(N, T, HV, V, device="cuda"),
        torch.empty(N, T, HV, K, device="cuda"),
        torch.empty(N, T, HV, K, device="cuda"),
    )


def make_conv_only_call(inp, N, T):
    state = inp["conv_state"].clone()
    window = torch.empty(N, T, inp["D"], W - 1, device="cuda")
    conv_update = load_causal_conv_update()

    def call():
        return conv_update(
            inp["mixed"].transpose(1, 2),
            state,
            inp["weight"],
            inp["bias"],
            activation="silu",
            conv_state_indices=inp["indices"],
            intermediate_conv_window=window,
            intermediate_state_indices=inp["indices"],
        )

    return call


def make_kvbuffer_only_call(inp, N, T, H, HV):
    post = make_conv_only_call(inp, N, T)().transpose(1, 2)
    q_end = H * K
    q = post[..., :q_end].view(N, T, H, K)
    k = post[..., q_end : 2 * q_end].view(N, T, H, K)
    v = post[..., 2 * q_end :].view(N, T, HV, V)
    ssm = inp["ssm"].clone()
    out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
    bufs = alloc_ubufs(N, T, HV)

    def call():
        return kda_decode_mtp_tensor_core_kvbuffer(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            q=q,
            k=k,
            v=v,
            a=inp["a"],
            b=inp["b"],
            initial_state_source=ssm,
            initial_state_indices=inp["indices"],
            scale=K**-0.5,
            out=out,
            d_buffer=bufs[0],
            k_buffer=bufs[1],
            g_buffer=bufs[2],
            lower_bound=-5.0,
        )

    return call


def make_unfused_call(inp, N, T, H, HV):
    state = inp["conv_state"].clone()
    ssm = inp["ssm"].clone()
    window = torch.empty(N, T, inp["D"], W - 1, device="cuda")
    conv_update = load_causal_conv_update()
    q_end = H * K
    out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
    bufs = alloc_ubufs(N, T, HV)

    def call():
        post = conv_update(
            inp["mixed"].transpose(1, 2),
            state,
            inp["weight"],
            inp["bias"],
            activation="silu",
            conv_state_indices=inp["indices"],
            intermediate_conv_window=window,
            intermediate_state_indices=inp["indices"],
        ).transpose(1, 2)
        q = post[..., :q_end].view(N, T, H, K)
        k = post[..., q_end : 2 * q_end].view(N, T, H, K)
        v = post[..., 2 * q_end :].view(N, T, HV, V)
        kda_decode_mtp_tensor_core_kvbuffer(
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            q=q,
            k=k,
            v=v,
            a=inp["a"],
            b=inp["b"],
            initial_state_source=ssm,
            initial_state_indices=inp["indices"],
            scale=K**-0.5,
            out=out,
            d_buffer=bufs[0],
            k_buffer=bufs[1],
            g_buffer=bufs[2],
            lower_bound=-5.0,
        )
        return out

    return call, bufs, ssm


def make_fused_call(inp, N, T, H, HV, num_v_tiles=-1, opt_level=3):
    state = inp["conv_state"].clone()
    ssm = inp["ssm"].clone()
    window = torch.empty(N, T, inp["D"], W - 1, device="cuda")
    out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)
    bufs = alloc_ubufs(N, T, HV)

    def call():
        return kda_conv_decode_mtp_kvbuffer(
            mixed_qkv=inp["mixed"].view(N * T, inp["D"]),
            conv_weight=inp["weight"],
            conv_bias=inp["bias"],
            conv_state=state,
            conv_state_indices=inp["indices"],
            intermediate_conv_window=window,
            intermediate_state_indices=inp["indices"],
            a=inp["a"],
            b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            ssm_states=ssm,
            cache_indices=inp["indices"],
            scale=K**-0.5,
            T=T,
            num_q_heads=H,
            num_v_heads=HV,
            head_k_dim=K,
            head_v_dim=V,
            out=out,
            d_buffer=bufs[0],
            k_buffer=bufs[1],
            g_buffer=bufs[2],
            lower_bound=-5.0,
            opt_level=opt_level,
            num_v_tiles=num_v_tiles,
        )

    return call, bufs, ssm


def make_recurrent_call(inp, N, T, H, HV, variant="auto"):
    state = inp["conv_state"].clone()
    ssm = inp["ssm"].clone()
    window = torch.empty(N, T, inp["D"], W - 1, device="cuda")
    inter = torch.empty(N, T, HV, V, K, device="cuda")
    out = torch.empty(N, T, HV, V, device="cuda", dtype=torch.bfloat16)

    def call():
        return kda_conv_decode_mtp_verify(
            mixed_qkv=inp["mixed"].view(N * T, inp["D"]),
            conv_weight=inp["weight"],
            conv_bias=inp["bias"],
            conv_state=state,
            conv_state_indices=inp["indices"],
            intermediate_conv_window=window,
            intermediate_state_indices=inp["indices"],
            a=inp["a"],
            b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            ssm_states=ssm,
            cache_indices=inp["indices"],
            intermediate_states_buffer=inter,
            scale=K**-0.5,
            T=T,
            num_q_heads=H,
            num_v_heads=HV,
            head_k_dim=K,
            head_v_dim=V,
            lower_bound=-5.0,
            variant=variant,
            out=out,
        )

    return call, inter, ssm


def make_triton_fused_call(inp, N, T, H, HV):
    fn = load_triton_fused()
    state = inp["conv_state"].transpose(1, 2).contiguous().transpose(1, 2)
    ssm = inp["ssm"].clone()
    window = torch.empty(N, T, inp["D"], W - 1, device="cuda")
    inter = torch.empty(N, T, HV, V, K, device="cuda")

    def call():
        return fn(
            mixed_qkv=inp["mixed"].view(N * T, inp["D"]),
            conv_weight=inp["weight"],
            conv_bias=inp["bias"],
            conv_state=state,
            conv_state_indices=inp["indices"],
            intermediate_conv_window=window,
            intermediate_state_indices=inp["indices"],
            a=inp["a"],
            b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            ssm_states=ssm,
            cache_indices=inp["indices"],
            intermediate_states_buffer=inter,
            scale=K**-0.5,
            T=T,
            num_q_heads=H,
            num_v_heads=HV,
            head_k_dim=K,
            head_v_dim=V,
            lower_bound=-5.0,
            num_warps=4,
        )

    return call, inter, ssm


def chain_call(verify, ssm, indices, bufs, accept, flush_bv=-1):
    accept_tensor = torch.full((bufs[0].shape[0],), accept, device="cuda", dtype=torch.int32)

    def call():
        verify()
        kda_flush_kvbuffer(ssm, indices, *bufs, accept_len=accept_tensor, bv=flush_bv)

    return call


def recurrent_chain_call(verify, ssm, inter, accept):
    scatter = load_scatter_commit()
    N, T, HV = inter.shape[:3]
    dst = ssm.view(1, N, HV, V, K)
    src = inter.view(1, N, T, HV, V, K)
    dst_idx = torch.arange(N, device=ssm.device, dtype=torch.int32)
    step_idx = torch.full((N,), accept - 1, device=ssm.device, dtype=torch.int32)

    def call():
        verify()
        scatter(dst, src, dst_idx, step_idx)

    return call


def bench_one(N, T, H, HV, args):
    inp = make_inputs(N, T, H, HV, args.seed)
    graph_calls = args.graph_calls if N < 16 else 1
    rows = {}

    conv = make_conv_only_call(inp, N, T)
    kvbuffer = make_kvbuffer_only_call(inp, N, T, H, HV)
    unfused, unfused_bufs, unfused_ssm = make_unfused_call(inp, N, T, H, HV)
    fused, fused_bufs, fused_ssm = make_fused_call(
        inp,
        N,
        T,
        H,
        HV,
        num_v_tiles=args.num_v_tiles,
    )
    rows["conv"] = graph_time_us(conv, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
    rows["kvbuffer_tensor_core"] = graph_time_us(kvbuffer, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
    rows["unfused_tensor_core"] = graph_time_us(unfused, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
    rows["fused_tensor_core"] = graph_time_us(fused, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
    rows["unfused_tensor_core_chain"] = graph_time_us(
        chain_call(unfused, unfused_ssm, inp["indices"], unfused_bufs, args.accept or T, args.flush_bv),
        warmup=args.warmup,
        rep=args.rep,
        graph_calls=graph_calls,
    )
    rows["fused_tensor_core_chain"] = graph_time_us(
        chain_call(fused, fused_ssm, inp["indices"], fused_bufs, args.accept or T, args.flush_bv),
        warmup=args.warmup,
        rep=args.rep,
        graph_calls=graph_calls,
    )

    for variant in ("small_batch", "large_batch", "auto"):
        recurrent, inter, recurrent_ssm = make_recurrent_call(inp, N, T, H, HV, variant=variant)
        rows[f"cula_{variant}"] = graph_time_us(recurrent, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
        rows[f"cula_{variant}_chain"] = graph_time_us(
            recurrent_chain_call(recurrent, recurrent_ssm, inter, args.accept or T),
            warmup=args.warmup,
            rep=args.rep,
            graph_calls=graph_calls,
        )
    if T >= W - 1:
        triton, tri_inter, tri_ssm = make_triton_fused_call(inp, N, T, H, HV)
        rows["triton_fused"] = graph_time_us(triton, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
        rows["triton_fused_chain"] = graph_time_us(
            recurrent_chain_call(triton, tri_ssm, tri_inter, args.accept or T),
            warmup=args.warmup,
            rep=args.rep,
            graph_calls=graph_calls,
        )
    else:
        rows["triton_fused"] = float("nan")
        rows["triton_fused_chain"] = float("nan")
    return rows


def bench_summary(N, T, H, HV, args):
    """Time the three summary rows with their matching commit chain."""
    inp = make_inputs(N, T, H, HV, args.seed)
    graph_calls = args.graph_calls if N < 16 else 1
    rows = {}

    kvbuffer, bufs, kvb_ssm = make_fused_call(inp, N, T, H, HV, num_v_tiles=args.num_v_tiles)
    rows["kvbuffer"] = graph_time_us(kvbuffer, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
    rows["kvbuffer_chain"] = graph_time_us(
        chain_call(kvbuffer, kvb_ssm, inp["indices"], bufs, args.accept or T, args.flush_bv),
        warmup=args.warmup,
        rep=args.rep,
        graph_calls=graph_calls,
    )

    recurrent, inter, recurrent_ssm = make_recurrent_call(inp, N, T, H, HV, variant="auto")
    rows["non_kvbuffer_auto"] = graph_time_us(recurrent, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
    rows["non_kvbuffer_auto_chain"] = graph_time_us(
        recurrent_chain_call(recurrent, recurrent_ssm, inter, args.accept or T),
        warmup=args.warmup,
        rep=args.rep,
        graph_calls=graph_calls,
    )

    if T >= W - 1:
        triton, inter, triton_ssm = make_triton_fused_call(inp, N, T, H, HV)
        rows["triton"] = graph_time_us(triton, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
        rows["triton_chain"] = graph_time_us(
            recurrent_chain_call(triton, triton_ssm, inter, args.accept or T),
            warmup=args.warmup,
            rep=args.rep,
            graph_calls=graph_calls,
        )
    else:
        rows["triton"] = float("nan")
        rows["triton_chain"] = float("nan")
    return rows


def bench_fused_only(N, T, H, HV, args):
    """Time only the production fused tensor-core kernel and its flush chain."""
    inp = make_inputs(N, T, H, HV, args.seed)
    graph_calls = args.graph_calls if N < 16 else 1
    fused, bufs, ssm = make_fused_call(inp, N, T, H, HV, num_v_tiles=args.num_v_tiles)
    verify_us = graph_time_us(fused, warmup=args.warmup, rep=args.rep, graph_calls=graph_calls)
    chain_us = graph_time_us(
        chain_call(fused, ssm, inp["indices"], bufs, args.accept or T, args.flush_bv),
        warmup=args.warmup,
        rep=args.rep,
        graph_calls=graph_calls,
    )
    return {
        "fused_tensor_core": verify_us,
        "fused_tensor_core_chain": chain_us,
    }


def print_row(N, T, rows):
    fused = rows["fused_tensor_core"]
    unfused = rows["unfused_tensor_core"]
    tri = rows["triton_fused"]
    tri_chain = rows["triton_fused_chain"]
    print(
        f"N={N:3d} T={T:2d} | baseline verify us: triton={tri:8.3f} "
        f"fused-tc={fused:8.3f} ({tri / fused:5.2f}x) "
        f"cula-auto={rows['cula_auto']:8.3f}"
    )
    print(
        f"             | baseline chain  us: triton={tri_chain:8.3f} "
        f"fused-tc={rows['fused_tensor_core_chain']:8.3f} "
        f"({tri_chain / rows['fused_tensor_core_chain']:5.2f}x) "
        f"cula-auto={rows['cula_auto_chain']:8.3f}"
    )
    print(
        f"             | recurrent verify us: small={rows['cula_small_batch']:8.3f} "
        f"({tri / rows['cula_small_batch']:5.2f}x) "
        f"large={rows['cula_large_batch']:8.3f} "
        f"({tri / rows['cula_large_batch']:5.2f}x)"
    )
    print(
        f"             | recurrent chain  us: small={rows['cula_small_batch_chain']:8.3f} "
        f"({tri_chain / rows['cula_small_batch_chain']:5.2f}x) "
        f"large={rows['cula_large_batch_chain']:8.3f} "
        f"({tri_chain / rows['cula_large_batch_chain']:5.2f}x)"
    )
    print(
        f"             | unfused tensor us: verify={unfused:8.3f} -> {fused:8.3f} "
        f"({unfused / fused:5.2f}x); chain={rows['unfused_tensor_core_chain']:8.3f} "
        f"-> {rows['fused_tensor_core_chain']:8.3f}"
    )
    print(
        f"             | breakdown us: conv={rows['conv']:8.3f} "
        f"kvb-tc={rows['kvbuffer_tensor_core']:8.3f} "
        f"fused-over-kvb={fused - rows['kvbuffer_tensor_core']:8.3f}"
    )


def print_summary_row(N, T, rows):
    triton = rows["triton"]
    triton_chain = rows["triton_chain"]
    print(
        f"N={N:3d} T={T:2d} | verify us: triton={triton:8.3f} "
        f"non-kvb-auto={rows['non_kvbuffer_auto']:8.3f} "
        f"({triton / rows['non_kvbuffer_auto']:5.2f}x) "
        f"kvbuffer={rows['kvbuffer']:8.3f} ({triton / rows['kvbuffer']:5.2f}x)"
    )
    print(
        f"             | chain  us: triton={triton_chain:8.3f} "
        f"non-kvb-auto={rows['non_kvbuffer_auto_chain']:8.3f} "
        f"({triton_chain / rows['non_kvbuffer_auto_chain']:5.2f}x) "
        f"kvbuffer={rows['kvbuffer_chain']:8.3f} "
        f"({triton_chain / rows['kvbuffer_chain']:5.2f}x)"
    )


def profile_one(args):
    """Run one selected path repeatedly so an external profiler can wrap it."""
    N = args.batch_sizes[0] if args.batch_sizes else args.N
    T = args.Ts[0] if args.Ts else args.T
    inp = make_inputs(N, T, args.H, args.HV, args.seed)
    accept = args.accept or T
    name = args.profile.removesuffix("_chain")
    if name == "conv":
        verify = make_conv_only_call(inp, N, T)
        bufs = ssm = None
    elif name == "kvbuffer_tensor_core":
        verify = make_kvbuffer_only_call(inp, N, T, args.H, args.HV)
        bufs = ssm = None
    elif name == "unfused_tensor_core":
        verify, bufs, ssm = make_unfused_call(inp, N, T, args.H, args.HV)
    elif name == "fused_tensor_core":
        verify, bufs, ssm = make_fused_call(inp, N, T, args.H, args.HV, num_v_tiles=args.num_v_tiles)
    elif name in ("small_batch", "large_batch", "auto"):
        verify, inter, ssm = make_recurrent_call(inp, N, T, args.H, args.HV, variant=name)
        if args.profile.endswith("_chain"):
            verify = recurrent_chain_call(verify, ssm, inter, accept)
        bufs = None
    elif name == "triton_fused":
        verify, inter, ssm = make_triton_fused_call(inp, N, T, args.H, args.HV)
        if args.profile.endswith("_chain"):
            verify = recurrent_chain_call(verify, ssm, inter, accept)
        bufs = None
    elif name == "flush":
        prepare, bufs, ssm = make_fused_call(inp, N, T, args.H, args.HV, num_v_tiles=args.num_v_tiles)
        prepare()
        verify = chain_call(lambda: None, ssm, inp["indices"], bufs, accept, args.flush_bv)
    else:
        raise ValueError(f"unknown profile path {args.profile}")
    if args.profile.endswith("_chain") and name not in (
        "triton_fused",
        "small_batch",
        "large_batch",
        "auto",
    ):
        verify = chain_call(verify, ssm, inp["indices"], bufs, accept, args.flush_bv)
    for _ in range(5):
        verify()
    torch.cuda.synchronize()
    for _ in range(args.profile_iters):
        verify()
    torch.cuda.synchronize()
    print(f"profiled {args.profile} N={N} T={T} H={args.H} HV={args.HV} iters={args.profile_iters}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--HV", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--Ts", type=int, nargs="+", default=None)
    parser.add_argument("--accept", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-v-tiles", type=int, default=-1, choices=[-1, 1, 2, 4])
    parser.add_argument("--flush-bv", type=int, default=-1, choices=[-1, 8, 16, 32])
    parser.add_argument(
        "--profile",
        default="",
        choices=[
            "",
            "conv",
            "kvbuffer_tensor_core",
            "unfused_tensor_core",
            "fused_tensor_core",
            "small_batch",
            "large_batch",
            "auto",
            "triton_fused",
            "unfused_tensor_core_chain",
            "fused_tensor_core_chain",
            "small_batch_chain",
            "large_batch_chain",
            "auto_chain",
            "triton_fused_chain",
            "flush",
        ],
        help="run one path in a profiler-friendly launch loop",
    )
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--rm-triton-cache", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="time Triton, non-KVBuffer auto and KVBuffer verify+chain",
    )
    parser.add_argument(
        "--fused-only",
        action="store_true",
        help="time only fused tensor-core verify and verify+flush chain",
    )
    parser.add_argument("--output-csv", type=pathlib.Path, default=None)
    args = parser.parse_args()
    for T in args.Ts or [args.T]:
        if not 1 <= T <= 8:
            parser.error(f"fused conv KVBuffer requires T in [1, 8], got T={T}")
        if not 1 <= (args.accept or T) <= T:
            parser.error(f"--accept must be in [1, T], got accept={args.accept}, T={T}")
    if args.rm_triton_cache:
        clear_triton_cache()
    if args.profile:
        profile_one(args)
        return
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}; H={args.H}, HV={args.HV}, K=V=128")
    records = []
    for T in args.Ts or [args.T]:
        for N in args.batch_sizes or [args.N]:
            if args.summary_only:
                rows = bench_summary(N, T, args.H, args.HV, args)
                print_summary_row(N, T, rows)
                records.append({"H": args.H, "HV": args.HV, "N": N, "T": T, **rows})
                continue
            if args.fused_only:
                rows = bench_fused_only(N, T, args.H, args.HV, args)
                print(
                    f"N={N:3d} T={T:2d} | "
                    f"verify={rows['fused_tensor_core']:8.3f} us "
                    f"chain={rows['fused_tensor_core_chain']:8.3f} us"
                )
                records.append({"H": args.H, "HV": args.HV, "N": N, "T": T, **rows})
                continue
            rows = bench_one(N, T, args.H, args.HV, args)
            print_row(N, T, rows)
            records.append({"H": args.H, "HV": args.HV, "N": N, "T": T, **rows})
    if args.output_csv:
        if not args.output_csv.parent.is_dir():
            parser.error(f"CSV parent directory does not exist: {args.output_csv.parent}")
        with args.output_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
