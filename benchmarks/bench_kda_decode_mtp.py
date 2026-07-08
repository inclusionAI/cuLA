"""KDA MTP decode benchmark — recurrent vs KVBuffer (chunkwise) verify CHAIN.

Unified bench (supersedes the old forward-only bench_kda_decode_mtp and
bench_kda_kvbuffer). Variants, selectable via --only / --profile:
  recurrent verify: vk / tri (official Triton), all writing T*d^2 states;
  kvbuffer verify:  shuffle (token-parallel) / tensor_core (CuTe tensor-core GEMM
                    form, flat-in-T), both writing the compact u-buffer;
  forward-only baselines (no rollback cost, breakdown table only): kv / auto / loop.

Chain: REC = recurrent verify (writes T·d² intermediate states) + commit; KVB =
kvbuffer verify (emit output + write a compact u-buffer) + flush (rank-m rebuild of
S_m). spd = REC / KVB. The commit uses the REAL sglang fused_mamba_state_scatter_with_mask
(from KDA_SCATTER_FILE) so the recurrent rollback cost is official code, not a model.

Self-contained (inlines input/timing helpers). Triton recurrent baseline (numerical
check only) from KDA_TRITON_FILE; scatter commit from KDA_SCATTER_FILE.
"""

import argparse
import importlib.util
import os

import torch

from cula.ops.kda.decode.cute import kda_decode
from cula.ops.kda.decode.mtp import (
    kda_decode_mtp,
    kda_decode_mtp_recurrent,
    kda_decode_mtp_recurrent_ws,
)
from cula.ops.kda.decode.mtp_kvbuffer import kda_flush_kvbuffer

# shuffle-kvbuffer (token-parallel, structure B) is optional too.
try:
    from cula.ops.kda.decode.mtp_kvbuffer import kda_decode_mtp_shuffle_kvbuffer

    _HAVE_SHUFFLE = True
except Exception:
    _HAVE_SHUFFLE = False

# tensor_core-kvbuffer (CuTe tensor-core, flat-in-T verify).
try:
    from cula.ops.kda.decode.mtp_kvbuffer import kda_decode_mtp_tensor_core_kvbuffer

    _HAVE_TCORE = True
except Exception:
    _HAVE_TCORE = False


def _load_from_file(path, attr):
    """Load a single attribute from a standalone .py file via importlib."""
    spec = importlib.util.spec_from_file_location(f"_standalone_{attr}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


# Triton recurrent baseline (numerical check only).
_HAVE_TRITON, _TRITON_ERR = True, ""
fused_sigmoid_gating_delta_rule_update = None
try:
    _f = os.environ.get("KDA_TRITON_FILE", "")
    if _f and os.path.exists(_f):
        fused_sigmoid_gating_delta_rule_update = _load_from_file(_f, "fused_sigmoid_gating_delta_rule_update")
    else:
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
except Exception as e:
    _HAVE_TRITON, _TRITON_ERR = False, repr(e)

# Official sglang scatter commit (update_mamba_state_after_mtp_verify).
_HAVE_SCATTER, _SCATTER_ERR = True, ""
fused_mamba_state_scatter_with_mask = None
try:
    _f = os.environ.get("KDA_SCATTER_FILE", "")
    if _f and os.path.exists(_f):
        fused_mamba_state_scatter_with_mask = _load_from_file(_f, "fused_mamba_state_scatter_with_mask")
    else:
        from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
            fused_mamba_state_scatter_with_mask,
        )
except Exception as e:
    _HAVE_SCATTER, _SCATTER_ERR = False, repr(e)


def make_dense_inputs(N, T, H, HV, K, V, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16
    q = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    k = torch.randn(N, T, H, K, device=device, dtype=bf16, generator=g)
    v = torch.randn(N, T, HV, V, device=device, dtype=bf16, generator=g)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32, generator=g) * 0.1).to(bf16)
    b = torch.randn(N, T, HV, device=device, dtype=bf16, generator=g)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32, generator=g) * 2
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32, generator=g) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32, generator=g) * 0.01
    indices = torch.arange(N, device=device, dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, state, indices


def to_triton_varlen(q, k, v, a, b):
    N, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = N * T
    q_t = q.reshape(1, NT, H, K).contiguous()
    k_t = k.reshape(1, NT, H, K).contiguous()
    v_t = v.reshape(1, NT, HV, V).contiguous()
    a_t = a.reshape(1, NT, HV * K).contiguous()
    b_t = b.reshape(1, NT, HV).contiguous()
    cu_seqlens = torch.arange(0, (N + 1) * T, T, device=q.device, dtype=torch.int32)
    return q_t, k_t, v_t, a_t, b_t, cu_seqlens


def make_triton_call(
    qt,
    kt,
    vt,
    at,
    bt,
    cu_seqlens,
    A_log,
    dt_bias,
    state,
    indices,
    scale,
    dsu,
    inter_buf=None,
    inter_idx=None,
    cache_steps=None,
):
    """Official sglang recurrent verify. In verify mode (inter_buf set) it writes the T·d²
    intermediate_states_buffer, same rollback cost as our production recurrent_v."""

    def call():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=at,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=qt,
            k=kt,
            v=vt,
            b=bt,
            initial_state_source=state,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            is_kda=True,
            disable_state_update=dsu,
            intermediate_states_buffer=inter_buf,
            intermediate_state_indices=inter_idx,
            cache_steps=cache_steps,
            retrieve_parent_token=None,
            lower_bound=None,
        )

    return call


def warmup(fn, n):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def t_graph_ms(fn, warmup_iters, rep, graph_calls=1):
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
    return start.elapsed_time(end) / rep / graph_calls


_VK_BV = -1
_ONLY = set()  # empty = all variants


def _want(name):
    return not _ONLY or name in _ONLY


def make_vk_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, inter_buf=None):
    """Production recurrent vk. In verify mode (inter_buf set) it writes the T·d²
    intermediate_states_buffer — the rollback cost kvbuffer replaces with a u-buffer."""

    def call():
        return kda_decode_mtp_recurrent(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=dsu,
            variant="vk",
            bv=_VK_BV,
            intermediate_states_buffer=inter_buf,
        )

    return call


def make_recurrent_ws_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, inter_buf=None):
    """Production warp-spec recurrent (recurrent_ws). In verify mode (inter_buf set) it also writes T*d^2 states."""

    def call():
        return kda_decode_mtp_recurrent_ws(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=dsu,
            intermediate_states_buffer=inter_buf,
        )

    return call


def make_shuffle_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, ubufs=None):
    """shuffle-kvbuffer (token-parallel chunkwise, structure B) — target: verify latency ~flat in T.
    tile_v / ilp_rows overridable via env KDA_SHUFFLE_TILE_V / KDA_SHUFFLE_ILP_ROWS (-1 = auto)."""
    d_buf, k_buf, g_buf = ubufs if ubufs is not None else (None, None, None)
    _tv = int(os.environ.get("KDA_SHUFFLE_TILE_V", "-1"))
    _ilp = int(os.environ.get("KDA_SHUFFLE_ILP_ROWS", "-1"))

    def call():
        return kda_decode_mtp_shuffle_kvbuffer(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=dsu,
            emit_output=True,
            d_buffer=d_buf,
            k_buffer=k_buf,
            g_buffer=g_buf,
            tile_v=_tv,
            ilp_rows=_ilp,
        )

    return call


def make_tcore_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, ubufs=None):
    """CuTe tensor-core tensor_core-kvbuffer. env KDA_TCORE_BV / KDA_TCORE_NUM_V_TILES (-1 = auto)."""
    d_buf, k_buf, g_buf = ubufs if ubufs is not None else (None, None, None)
    _bv = int(os.environ.get("KDA_TCORE_BV", "32"))
    _num_v_tiles = int(os.environ.get("KDA_TCORE_NUM_V_TILES", "-1"))

    def call():
        return kda_decode_mtp_tensor_core_kvbuffer(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=dsu,
            emit_output=True,
            d_buffer=d_buf,
            k_buffer=k_buf,
            g_buffer=g_buf,
            bv=_bv,
            num_v_tiles=_num_v_tiles,
        )

    return call


def make_kv_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """Forward-only production kv (lane=V recurrent; no intermediate-state support)."""
    state_kv = state.transpose(-2, -1).contiguous()  # vk->kv once, outside timing

    def call():
        return kda_decode_mtp_recurrent(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state_kv,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=dsu,
            variant="kv",
        )

    return call


def make_auto_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, inter_buf=None):
    """kda_decode_mtp dispatch (recurrent vk)."""

    def call():
        return kda_decode_mtp(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=dsu,
            state_layout="vk",
            intermediate_states_buffer=inter_buf,
        )

    return call


def make_loop_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """Per-token kda_decode loop baseline (slices pre-cut; kda_decode always writes state)."""
    N, T = q.shape[0], q.shape[1]
    HV, V = v.shape[2], v.shape[3]
    qs = [q[:, t].unsqueeze(1).contiguous() for t in range(T)]
    ks = [k[:, t].unsqueeze(1).contiguous() for t in range(T)]
    vs = [v[:, t].unsqueeze(1).contiguous() for t in range(T)]
    as_ = [a[:, t].unsqueeze(1).contiguous() for t in range(T)]
    bs = [b[:, t].unsqueeze(1).contiguous() for t in range(T)]
    st = state.clone().contiguous()
    o = torch.empty(N, T, HV, V, device=q.device, dtype=torch.bfloat16)

    def call():
        for t in range(T):
            o_t = kda_decode(
                A_log=A_log,
                dt_bias=dt_bias,
                q=qs[t],
                k=ks[t],
                v=vs[t],
                a=as_[t],
                b=bs[t],
                initial_state_source=st,
                initial_state_indices=indices,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
            )
            o[:, t] = o_t.squeeze(1)
        return o

    return call


# ---- verify-chain components: commit (recurrent rollback) & flush (kvbuffer) ----
def make_scatter_commit_call(state_pool, inter_buf, m, N, T, HV, V, K):
    """Recurrent rollback via the OFFICIAL sglang fused_mamba_state_scatter_with_mask:
    gather each request's accepted-step state from the intermediate cache into the pool
    (num_layers=1; step = m-1 for all requests)."""
    dst = state_pool.view(1, N, HV, V, K)  # [layers, cache, *state]
    src = inter_buf.view(1, N, T, HV, V, K)  # [layers, req, step, *state]
    dst_idx = torch.arange(N, device=state_pool.device, dtype=torch.int32)
    step_idx = torch.full((N,), m - 1, device=state_pool.device, dtype=torch.int32)

    def call():
        fused_mamba_state_scatter_with_mask(dst, src, dst_idx, step_idx)
        return state_pool

    return call


def make_gather_commit_call(state_pool, inter_buf, m):
    """Recurrent rollback, strided gather model: copy inter_buf[:,m-1] (a T-strided view)
    into the pool. Less coalesced than the official kernel — kept for sensitivity only."""
    midx = m - 1

    def call():
        state_pool.copy_(inter_buf[:, midx])
        return state_pool

    return call


def make_flush_call(state_pool, indices, ubufs, m):
    """KVBuffer flush: read the compact u-buffer, rank-m rebuild S_m (no recompute)."""
    d_b, k_b, g_b = ubufs

    def call():
        return kda_flush_kvbuffer(state_pool, indices, d_b, k_b, g_b, m)

    return call


def _accept_len(T, accept, N=0):
    if accept == "full":
        return T
    if accept == "half":
        return max(1, (T + 1) // 2)
    if accept == "one":
        return 1
    if accept == "random":
        # Deterministic per-(N,T) accept length in [1,T] (real serving is per-req variable).
        g = torch.Generator().manual_seed(1000 * N + T)
        return int(torch.randint(1, T + 1, (1,), generator=g).item())
    return max(1, min(int(accept), T))


def _profile_one(args, DSU, device):
    """Run ONE method's kernel in a loop so ncu can wrap it. Shape = (batch_sizes[0], Ts[0])."""
    N, T = args.batch_sizes[0], args.Ts[0]
    q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(N, T, args.H, args.HV, args.K, args.V, device)
    scale = args.K**-0.5
    m = _accept_len(T, args.accept, N)
    inter_buf = torch.empty(N, T, args.HV, args.V, args.K, dtype=torch.float32, device=device)
    ubufs = (
        torch.empty(N, T, args.HV, args.V, dtype=torch.float32, device=device),
        torch.empty(N, T, args.HV, args.K, dtype=torch.float32, device=device),
        torch.empty(N, T, args.HV, args.K, dtype=torch.float32, device=device),
    )
    p = args.profile
    if p == "recurrent":
        fn = make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
    elif p == "recurrent_ws":
        fn = make_recurrent_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
    elif p == "shuffle":
        fn = make_shuffle_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
    elif p == "tensor_core":
        fn = make_tcore_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
    elif p == "triton":
        qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
        tri_idx = torch.arange(N, device=device, dtype=torch.int32)
        fn = make_triton_call(
            qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf, tri_idx, T
        )
    elif p == "commit":
        make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)()
        fn = make_scatter_commit_call(state0.clone(), inter_buf, m, N, T, args.HV, args.V, args.K)
    elif p == "recurrent_kv":
        fn = make_kv_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
    elif p == "auto":
        fn = make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
    elif p == "loop":
        fn = make_loop_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
    elif p == "flush":
        make_shuffle_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)()
        fn = make_flush_call(state0.clone(), indices, ubufs, m)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.profile_iters):
        fn()
    torch.cuda.synchronize()
    print(f"profiled {p} N={N} T={T} HV={args.HV} m={m} iters={args.profile_iters}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--HV", type=int, default=32)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--rep", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=5, help="warmup iters before each timed segment")
    ap.add_argument(
        "--graph-calls",
        type=int,
        default=20,
        help="ops per CUDA graph to amortize fixed launch overhead at small batch "
        "(N<16; N>=16 uses 1). needs idempotent dsu=1.",
    )
    ap.add_argument(
        "--dsu",
        type=int,
        default=1,
        choices=[0, 1],
        help="disable_state_update; 1=forward-only (idempotent, default), 0=write state",
    )
    ap.add_argument("--vk-bv", type=int, default=-1, choices=[-1, 8, 16, 32])
    ap.add_argument(
        "--accept", default="full", help="chain accept length m: full(=T)/half/one/random/<int>; drives commit/flush."
    )
    ap.add_argument(
        "--commit",
        default="scatter",
        choices=["scatter", "gather"],
        help="recurrent commit model: scatter=official sglang "
        "fused_mamba_state_scatter_with_mask (coalesced N·d², default); "
        "gather=strided copy (sensitivity). kvbuffer flush always counted.",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=[],
        choices=["recurrent", "recurrent_ws", "triton", "shuffle", "tensor_core", "recurrent_kv", "auto", "loop"],
        help="restrict check/timing to these verify variants (default: all). REC/spd columns show n/a for skipped baselines.",
    )
    ap.add_argument("--check", action="store_true", help="numerical check only, no timing")
    ap.add_argument("--atol", type=float, default=5e-2)
    ap.add_argument(
        "--profile",
        default="",
        choices=[
            "",
            "recurrent",
            "recurrent_ws",
            "shuffle",
            "tensor_core",
            "triton",
            "commit",
            "flush",
            "recurrent_kv",
            "auto",
            "loop",
        ],
        help="ncu profile mode: run one method's kernel in a loop (uses batch-sizes[0], Ts[0])",
    )
    ap.add_argument("--profile-iters", type=int, default=20, help="kernel launches in the profiled loop")
    args = ap.parse_args()

    global _VK_BV
    _VK_BV = args.vk_bv
    global _ONLY
    _ONLY = set(args.only)
    DSU = bool(args.dsu)
    device = "cuda"
    if args.profile:
        _profile_one(args, DSU, device)
        return
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"shape H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu={DSU} shuffle_impl={_HAVE_SHUFFLE} tensor_core_impl={_HAVE_TCORE}"
    )

    # ---------------- numerical check (vs Triton recurrent) ----------------
    if not _HAVE_TRITON:
        print(f"[warn] Triton baseline unavailable ({_TRITON_ERR}); skipping numerical check.")
    else:
        print(f"\n=== numerical check (max|Δ| vs Triton recurrent, threshold {args.atol}) ===")
        print(
            f"{'N':>4} {'T':>3} | {'Δ recurrent':>10} | {'Δ recurrent_ws':>10} | {'Δ shuffle':>10} | {'Δ tensor_core':>10} | flag"
        )
        for N in args.batch_sizes:
            for T in args.Ts:
                q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                    N, T, args.H, args.HV, args.K, args.V, device
                )
                scale = args.K**-0.5
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                o_tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(), indices, scale, True)()
                o_tri = o_tri.reshape(N, T, args.HV, args.V)
                d_recurrent = float("nan")
                if _want("recurrent"):
                    o_recurrent = make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)()
                    d_recurrent = (o_recurrent - o_tri).abs().max().item()
                d_recurrent_ws = float("nan")
                if _want("recurrent_ws"):
                    o_recurrent_ws = make_recurrent_ws_call(
                        q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True
                    )()
                    d_recurrent_ws = (o_recurrent_ws - o_tri).abs().max().item()
                d_shuffle = float("nan")
                if _HAVE_SHUFFLE and _want("shuffle"):
                    o_shuffle = make_shuffle_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)()
                    d_shuffle = (o_shuffle - o_tri).abs().max().item()
                d_tensor_core = float("nan")
                if _HAVE_TCORE and _want("tensor_core"):
                    o_tensor_core = make_tcore_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)()
                    d_tensor_core = (o_tensor_core - o_tri).abs().max().item()
                cand = [x for x in (d_recurrent, d_recurrent_ws, d_shuffle, d_tensor_core) if x == x]
                flag = ("OK" if max(cand) < args.atol else "DIFF!") if cand else "n/a"
                print(
                    f"{N:>4} {T:>3} | {d_recurrent:>10.2e} | {d_recurrent_ws:>10.2e} | {d_shuffle:>10.2e} | {d_tensor_core:>10.2e} | {flag}"
                )

    if args.check:
        return

    _timing_verify_chain(args, DSU, device)


def _timing_verify_chain(args, DSU, device):
    """Fair spec-decode verify CHAIN (each segment timed in its own CUDA graph, summed). All verify
    kernels run dsu=1 + verify-mode: recurrent vk/triton write the T·d² intermediate states,
    kvbuffer writes its compact u-buffer. REC = recurrent verify + commit; KVB = kvbuffer verify +
    flush. spd_recurrent = REC/KVB vs production recurrent; spd_bf = official triton REC chain
    / kvbuffer KVB chain. Prints chain totals + speedups first, per-segment breakdown after."""

    def us(x):
        return f"{x * 1e3:.1f}" if x else "n/a"

    def rat(a_, b_):
        return f"{a_ / b_:.2f}x" if (a_ and b_) else "n/a"

    if args.commit == "scatter" and not _HAVE_SCATTER:
        raise RuntimeError(
            f"commit=scatter needs the official sglang kernel; set KDA_SCATTER_FILE to "
            f"mamba_state_scatter_triton.py (load error: {_SCATTER_ERR})"
        )

    # ---- measure every segment for every (N, T) into `results` ----
    results = []
    for N in args.batch_sizes:
        for T in args.Ts:
            q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(N, T, args.H, args.HV, args.K, args.V, device)
            scale = args.K**-0.5
            m = _accept_len(T, args.accept, N)
            gc = 1 if N >= 16 else args.graph_calls  # amortize launch overhead at small batch
            inter_buf = torch.empty(N, T, args.HV, args.V, args.K, dtype=torch.float32, device=device)
            ubufs = (
                torch.empty(N, T, args.HV, args.V, dtype=torch.float32, device=device),
                torch.empty(N, T, args.HV, args.K, dtype=torch.float32, device=device),
                torch.empty(N, T, args.HV, args.K, dtype=torch.float32, device=device),
            )
            tg = {}

            def time_seg(fn):
                warmup(fn, args.warmup)
                return t_graph_ms(fn, args.warmup, args.rep, gc)

            # recurrent verify (dsu=1, writes T·d² states) + commit
            if _want("recurrent"):
                tg["recurrent_v"] = time_seg(
                    make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
                )
            if _want("recurrent_ws"):
                tg["recurrent_ws_v"] = time_seg(
                    make_recurrent_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
                )
            if _want("recurrent") or _want("recurrent_ws") or _want("triton"):
                if args.commit == "scatter":
                    fn_cmt = make_scatter_commit_call(state0.clone(), inter_buf, m, N, T, args.HV, args.V, args.K)
                else:
                    fn_cmt = make_gather_commit_call(state0.clone(), inter_buf, m)
                tg["cmt"] = time_seg(fn_cmt)
            # kvbuffer verify (dsu=1, writes u-buffer) + flush
            if _want("shuffle") or _want("tensor_core"):
                # flush needs a populated u-buffer: run one kvbuffer verify first to fill it
                if _HAVE_SHUFFLE and _want("shuffle"):
                    make_shuffle_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)()
                elif _HAVE_TCORE and _want("tensor_core"):
                    make_tcore_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)()
                tg["flush"] = time_seg(make_flush_call(state0.clone(), indices, ubufs, m))
            if _HAVE_SHUFFLE and _want("shuffle"):
                tg["shuffle_v"] = time_seg(
                    make_shuffle_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
                )
            if _HAVE_TCORE and _want("tensor_core"):
                tg["tensor_core_v"] = time_seg(
                    make_tcore_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
                )
            # official triton recurrent verify (dsu=1, writes T·d² states)
            if _HAVE_TRITON and _want("triton"):
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri_inter = torch.empty(N, T, args.HV, args.V, args.K, dtype=torch.float32, device=device)
                tri_idx = torch.arange(N, device=device, dtype=torch.int32)
                tg["triton_v"] = time_seg(
                    make_triton_call(
                        qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(), indices, scale, DSU, tri_inter, tri_idx, T
                    )
                )

            r = {"N": N, "T": T, "m": m, "tg": tg}

            def _sum(av, bv):
                return tg[av] + tg[bv] if (av in tg and bv in tg) else None

            r["REC_recurrent"] = _sum("recurrent_v", "cmt")
            r["REC_recurrent_ws"] = _sum("recurrent_ws_v", "cmt")
            r["KVB_shuffle"] = _sum("shuffle_v", "flush")
            r["KVB_tensor_core"] = _sum("tensor_core_v", "flush")
            r["REC_triton"] = _sum("triton_v", "cmt")
            results.append(r)

    # ---- table 1: chain totals + speedups ----
    print(f"\n=== verify-CHAIN total latency (us) + speedup — accept m={args.accept} commit={args.commit} ===")
    print("  REC_* = recurrent verify (writes T·d² states) + commit;  KVB_* = kvbuffer verify (u-buffer) + flush")
    print(
        "  spd_(recurrent/recurrent_ws/shuffle/tensor_core) = REC_triton (official triton) / (REC_recurrent/REC_recurrent_ws/KVB_shuffle/KVB_tensor_core) -- chain speedup over triton"
    )
    hdr = (
        f"{'N':>4} {'T':>3} {'m':>3} | {'REC_recurrent':>7} {'REC_recurrent_ws':>7} {'REC_triton':>7} | {'KVB_shuffle':>11} {'KVB_tensor_core':>9} | "
        f"{'spd_recurrent':>7} {'spd_recurrent_ws':>7} {'spd_shuffle':>11} {'spd_tensor_core':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['N']:>4} {r['T']:>3} {r['m']:>3} | {us(r['REC_recurrent']):>7} {us(r['REC_recurrent_ws']):>7} {us(r['REC_triton']):>7} | "
            f"{us(r['KVB_shuffle']):>11} {us(r['KVB_tensor_core']):>9} | "
            f"{rat(r['REC_triton'], r['REC_recurrent']):>7} {rat(r['REC_triton'], r['REC_recurrent_ws']):>7} {rat(r['REC_triton'], r['KVB_shuffle']):>11} {rat(r['REC_triton'], r['KVB_tensor_core']):>9}"
        )

    # ---- table 2: per-segment breakdown ----
    print("\n=== per-segment breakdown (us) — verify kernels + shared commit/flush ===")
    hdr2 = f"{'N':>4} {'T':>3} | {'recurrent_v':>6} {'recurrent_ws_v':>7} {'triton_v':>6} | {'shuffle_v':>9} {'tensor_core_v':>7} | {'cmt':>5} {'flush':>6}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in results:
        tg = r["tg"]
        print(
            f"{r['N']:>4} {r['T']:>3} | {us(tg.get('recurrent_v')):>6} {us(tg.get('recurrent_ws_v')):>7} {us(tg.get('triton_v')):>6} | "
            f"{us(tg.get('shuffle_v')):>9} {us(tg.get('tensor_core_v')):>7} | "
            f"{us(tg.get('cmt')):>5} {us(tg.get('flush')):>6}"
        )


if __name__ == "__main__":
    main()
