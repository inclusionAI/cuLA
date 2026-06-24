"""KDA MTP decode benchmark — recurrent vs KVBuffer (chunkwise) verify CHAIN.

Unified bench (supersedes the old forward-only bench_kda_decode_mtp and
bench_kda_kvbuffer). Variants, selectable via --only / --profile:
  recurrent verify: vk / ws / tri (official Triton), all writing T*d^2 states;
  kvbuffer verify:  tpkvb (token-parallel) / cgkvb (CuTe sm_90 tensor-core GEMM
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

from cula.ops.kda_decode import kda_decode
from cula.ops.kda_decode_mtp import (
    kda_decode_mtp,
    kda_decode_mtp_small_batch,
    kda_decode_mtp_ws,
)
from cula.ops.kda_decode_mtp_kvbuffer import kda_flush_kvbuffer

# tp-kvbuffer (token-parallel, structure B) is optional too.
try:
    from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_tp_kvbuffer

    _HAVE_TPKVB = True
except Exception:
    _HAVE_TPKVB = False

# gemm-kvbuffer (CuTe sm_90 tensor-core, flat-in-T verify).
try:
    from cula.ops.kda_decode_mtp_kvbuffer import kda_decode_mtp_gemm_kvbuffer_cute

    _HAVE_CGKVB = True
except Exception:
    _HAVE_CGKVB = False


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
    intermediate_states_buffer, same rollback cost as our production vk_v/ws_v."""

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
        return kda_decode_mtp_small_batch(
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


def make_ws_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, inter_buf=None):
    """Production recurrent ws. In verify mode (inter_buf set) it also writes T·d² states."""

    def call():
        return kda_decode_mtp_ws(
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


def make_tpkvb_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, ubufs=None):
    """tp-kvbuffer (token-parallel chunkwise, structure B) — target: verify latency ~flat in T.
    tile_v / ilp_rows overridable via env KDA_TPKVB_TILE_V / KDA_TPKVB_ILP_ROWS (-1 = auto)."""
    u_buf, kinv_buf, b_buf = ubufs if ubufs is not None else (None, None, None)
    _tv = int(os.environ.get("KDA_TPKVB_TILE_V", "-1"))
    _ilp = int(os.environ.get("KDA_TPKVB_ILP_ROWS", "-1"))

    def call():
        return kda_decode_mtp_tp_kvbuffer(
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
            u_buffer=u_buf,
            kinv_buffer=kinv_buf,
            b_buffer=b_buf,
            tile_v=_tv,
            ilp_rows=_ilp,
        )

    return call


def make_cgkvb_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu, ubufs=None):
    """CuTe sm_90 tensor-core gemm-kvbuffer. env KDA_CGKVB_BV / KDA_CGKVB_NUM_V_TILES (-1 = auto)."""
    u_buf, kinv_buf, b_buf = ubufs if ubufs is not None else (None, None, None)
    _bv = int(os.environ.get("KDA_CGKVB_BV", "32"))
    _num_v_tiles = int(os.environ.get("KDA_CGKVB_NUM_V_TILES", "-1"))

    def call():
        return kda_decode_mtp_gemm_kvbuffer_cute(
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
            u_buffer=u_buf,
            kinv_buffer=kinv_buf,
            b_buffer=b_buf,
            bv=_bv,
            num_v_tiles=_num_v_tiles,
        )

    return call


def make_kv_call(q, k, v, a, b, A_log, dt_bias, state, indices, scale, dsu):
    """Forward-only production kv (lane=V small_batch; no intermediate-state support)."""
    state_kv = state.transpose(-2, -1).contiguous()  # vk->kv once, outside timing

    def call():
        return kda_decode_mtp_small_batch(
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
    """kda_decode_mtp dispatch (small_batch vk for N*HV<=512, else ws)."""

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
    u_b, kinv_b, b_b = ubufs

    def call():
        return kda_flush_kvbuffer(state_pool, indices, u_b, kinv_b, b_b, m)

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
    if p == "vk":
        fn = make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
    elif p == "ws":
        fn = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
    elif p == "tpkvb":
        fn = make_tpkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
    elif p == "cgkvb":
        fn = make_cgkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
    elif p == "triton":
        qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
        tri_idx = torch.arange(N, device=device, dtype=torch.int32)
        fn = make_triton_call(
            qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf, tri_idx, T
        )
    elif p == "commit":
        make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)()
        fn = make_scatter_commit_call(state0.clone(), inter_buf, m, N, T, args.HV, args.V, args.K)
    elif p == "kv":
        fn = make_kv_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
    elif p == "auto":
        fn = make_auto_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
    elif p == "loop":
        fn = make_loop_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU)
    elif p == "flush":
        make_tpkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)()
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
        choices=["vk", "ws", "tri", "tpkvb", "cgkvb", "kv", "auto", "loop"],
        help="restrict check/timing to these verify variants (default: all). REC/spd columns show n/a for skipped baselines.",
    )
    ap.add_argument("--check", action="store_true", help="numerical check only, no timing")
    ap.add_argument("--atol", type=float, default=5e-2)
    ap.add_argument(
        "--profile",
        default="",
        choices=["", "vk", "ws", "tpkvb", "cgkvb", "triton", "commit", "flush", "kv", "auto", "loop"],
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
    print(f"shape H={args.H} HV={args.HV} K={args.K} V={args.V}  dsu={DSU} tpkvb_impl={_HAVE_TPKVB} cgkvb_impl={_HAVE_CGKVB}")

    # ---------------- numerical check (vs Triton recurrent) ----------------
    if not _HAVE_TRITON:
        print(f"[warn] Triton baseline unavailable ({_TRITON_ERR}); skipping numerical check.")
    else:
        print(f"\n=== numerical check (max|Δ| vs Triton recurrent, threshold {args.atol}) ===")
        print(f"{'N':>4} {'T':>3} | {'Δ vk':>10} | {'Δ ws':>10} | {'Δ tpkvb':>10} | {'Δ cgkvb':>10} | flag")
        for N in args.batch_sizes:
            for T in args.Ts:
                q, k, v, a, b, A_log, dt_bias, state0, indices = make_dense_inputs(
                    N, T, args.H, args.HV, args.K, args.V, device
                )
                scale = args.K**-0.5
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                o_tri = make_triton_call(qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(), indices, scale, True)()
                o_tri = o_tri.reshape(N, T, args.HV, args.V)
                d_vk = d_ws = float("nan")
                if _want("vk"):
                    o_vk = make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)()
                    d_vk = (o_vk - o_tri).abs().max().item()
                if _want("ws"):
                    o_ws = make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)()
                    d_ws = (o_ws - o_tri).abs().max().item()
                d_tpkvb = float("nan")
                if _HAVE_TPKVB and _want("tpkvb"):
                    o_tpkvb = make_tpkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)()
                    d_tpkvb = (o_tpkvb - o_tri).abs().max().item()
                d_cgkvb = float("nan")
                if _HAVE_CGKVB and _want("cgkvb"):
                    o_cgkvb = make_cgkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, True)()
                    d_cgkvb = (o_cgkvb - o_tri).abs().max().item()
                cand = [x for x in (d_vk, d_ws, d_tpkvb, d_cgkvb) if x == x]
                flag = ("OK" if max(cand) < args.atol else "DIFF!") if cand else "n/a"
                print(f"{N:>4} {T:>3} | {d_vk:>10.2e} | {d_ws:>10.2e} | {d_tpkvb:>10.2e} | {d_cgkvb:>10.2e} | {flag}")

    if args.check:
        return

    _timing_verify_chain(args, DSU, device)


def _timing_verify_chain(args, DSU, device):
    """Fair spec-decode verify CHAIN (each segment timed in its own CUDA graph, summed). All verify
    kernels run dsu=1 + verify-mode: recurrent vk/ws/triton write the T·d² intermediate states,
    kvbuffer writes its compact u-buffer. REC = recurrent verify + commit; KVB = kvbuffer verify +
    flush. spd_vk/spd_ws = REC/KVB vs production vk/ws; spd_vkbf/spd_wsbf = official triton REC chain
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
            if _want("vk"):
                tg["vk_v"] = time_seg(
                    make_vk_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
                )
            if _want("vk") or _want("ws") or _want("tri"):
                if args.commit == "scatter":
                    fn_cmt = make_scatter_commit_call(state0.clone(), inter_buf, m, N, T, args.HV, args.V, args.K)
                else:
                    fn_cmt = make_gather_commit_call(state0.clone(), inter_buf, m)
                tg["cmt"] = time_seg(fn_cmt)
            if _want("ws"):
                tg["ws_v"] = time_seg(
                    make_ws_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, inter_buf)
                )
            # kvbuffer verify (dsu=1, writes u-buffer) + flush
            if _want("tpkvb") or _want("cgkvb"):
                # flush needs a populated u-buffer: run one kvbuffer verify first to fill it
                if _HAVE_TPKVB and _want("tpkvb"):
                    make_tpkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)()
                elif _HAVE_CGKVB and _want("cgkvb"):
                    make_cgkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)()
                tg["flush"] = time_seg(make_flush_call(state0.clone(), indices, ubufs, m))
            if _HAVE_TPKVB and _want("tpkvb"):
                tg["tpkvb_v"] = time_seg(
                    make_tpkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
                )
            if _HAVE_CGKVB and _want("cgkvb"):
                tg["cgkvb_v"] = time_seg(
                    make_cgkvb_call(q, k, v, a, b, A_log, dt_bias, state0.clone(), indices, scale, DSU, ubufs)
                )
            # official triton recurrent verify (dsu=1, writes T·d² states)
            if _HAVE_TRITON and _want("tri"):
                qt, kt, vt, at, bt, cu = to_triton_varlen(q, k, v, a, b)
                tri_inter = torch.empty(N, T, args.HV, args.V, args.K, dtype=torch.float32, device=device)
                tri_idx = torch.arange(N, device=device, dtype=torch.int32)
                tg["tri_v"] = time_seg(
                    make_triton_call(
                        qt, kt, vt, at, bt, cu, A_log, dt_bias, state0.clone(), indices, scale, DSU, tri_inter, tri_idx, T
                    )
                )

            r = {"N": N, "T": T, "m": m, "tg": tg}

            def _sum(av, bv):
                return tg[av] + tg[bv] if (av in tg and bv in tg) else None

            r["REC_vk"] = _sum("vk_v", "cmt")
            r["REC_ws"] = _sum("ws_v", "cmt")
            r["KVB_tp"] = _sum("tpkvb_v", "flush")
            r["KVB_cg"] = _sum("cgkvb_v", "flush")
            r["REC_tri"] = _sum("tri_v", "cmt")
            results.append(r)

    # ---- table 1: chain totals + speedups ----
    print(f"\n=== verify-CHAIN total latency (us) + speedup — accept m={args.accept} commit={args.commit} ===")
    print("  REC_* = recurrent verify (writes T·d² states) + commit;  KVB_* = kvbuffer verify (u-buffer) + flush")
    print("  spd_(vk/ws/tp/cg) = REC_tri (official triton) / (REC_vk/REC_ws/KVB_tp/KVB_cg) -- chain speedup over triton")
    hdr = (
        f"{'N':>4} {'T':>3} {'m':>3} | {'REC_vk':>7} {'REC_ws':>7} {'REC_tri':>7} | {'KVB_tp':>7} {'KVB_cg':>7} | "
        f"{'spd_vk':>7} {'spd_ws':>7} {'spd_tp':>7} {'spd_cg':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['N']:>4} {r['T']:>3} {r['m']:>3} | {us(r['REC_vk']):>7} {us(r['REC_ws']):>7} {us(r['REC_tri']):>7} | "
            f"{us(r['KVB_tp']):>7} {us(r['KVB_cg']):>7} | "
            f"{rat(r['REC_tri'], r['REC_vk']):>7} {rat(r['REC_tri'], r['REC_ws']):>7} {rat(r['REC_tri'], r['KVB_tp']):>7} {rat(r['REC_tri'], r['KVB_cg']):>7}"
        )

    # ---- table 2: per-segment breakdown ----
    print("\n=== per-segment breakdown (us) — verify kernels + shared commit/flush ===")
    hdr2 = (
        f"{'N':>4} {'T':>3} | {'vk_v':>6} {'ws_v':>6} {'tri_v':>6} | {'tpkvb_v':>7} {'cgkvb_v':>7} | {'cmt':>5} {'flush':>6}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for r in results:
        tg = r["tg"]
        print(
            f"{r['N']:>4} {r['T']:>3} | {us(tg.get('vk_v')):>6} {us(tg.get('ws_v')):>6} {us(tg.get('tri_v')):>6} | "
            f"{us(tg.get('tpkvb_v')):>7} {us(tg.get('cgkvb_v')):>7} | "
            f"{us(tg.get('cmt')):>5} {us(tg.get('flush')):>6}"
        )


if __name__ == "__main__":
    main()
