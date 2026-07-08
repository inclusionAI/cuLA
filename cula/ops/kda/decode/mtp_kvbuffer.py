"""CuTe DSL KDA MTP decode — KVBuffer / chunkwise parallel-verification variant.

KVBuffer paper's chunkwise verify form (https://arxiv.org/abs/2605.19049) as a new
operator vs the recurrent vk/kv ops in ``kda_decode_mtp.py``. The T draft tokens
are treated as ONE chunk: per-token outputs come from the FIXED input state S0 plus a
small T×T intra-chunk correction, and the state is updated once at the end — the
S0-matvecs are independent across tokens (no length-T serial chain), the latency angle
at small batch. Infra (grid N*HV*(V//BV), 1 warp/CTA, lane=K, float4 loads, butterfly
reduce-over-K) mirrors the production vk kernel for apples-to-apples comparison.

Chunkwise math (state S0[v,k], decay-first; matches the recurrent op):
    g_t[k]  = exp(-exp(A_log) * softplus(a_t[k] + dt_bias[k]))    # per channel
    b_t[k]  = prod_{i<=t} g_i[k]                                  # cumulative decay
    kdec_t  = k_norm_t * b_t ;  qdec_t = q_scaled_t * b_t
    r(t,i)  = prod_{i<j<=t} g_j <= 1                  # decay ratio
    A[t,i]  = sum_k kn_t kn_i r(t,i) (i<t)   P[t,i] = sum_k qn_t kn_i r(t,i) (i<=t)
    u_t[v]  = beta_t * (v_t[v] - (S0 @ kdec_t)[v] - sum_{i<t} A[t,i] u_i[v])
    o_t[v]  = (S0 @ qdec_t)[v] + sum_{i<=t} P[t,i] u_i[v]
    ksuf_t  = k_norm_t * prod_{j>t} g_j                           # suffix-decayed key
    S_T[v,k]= b_{T-1}[k] * S0[v,k] + sum_i u_i[v] ksuf_i[k]       # full accept

Numerical form: every decay factor is an ORDERED product bounded by 1 — there is
no division by the cumulative gate product (which can underflow to 0 in fp32
under unbounded softplus gates). The op is valid for both softplus and safe-gate
models.
The scratch stores raw (u_i, k_i, g_i) per token — the same triplet as ReplaySSM's
(d, k, g) ring — and the flush rebuilds S_m with descending suffix products.
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.ops.kda.decode.cute import (
    TILE_K,
    _get_cached_stream,
    _normalize_A_log,
    _normalize_dt_bias,
    _normalize_state_indices,
    _normalize_state_source,
    _prepare_output_tensor,
)
from cula.ops.kda.decode.mtp import (
    VEC_SIZE,
    _normalize_mtp_a,
)

logger = logging.getLogger(__name__)


# tile_v by WU=N*HV: <=32->16, <256->32, >=256->64 (H200 sweep).
def _select_kvb_tile_v(V, N, HV):
    """work-unit (N*HV) dependent tile_v. Returns the first candidate that divides V."""
    wu = N * HV
    if wu <= 32:
        order = (16, 32, 8, 64)
    elif wu < 256:
        order = (32, 64, 16, 8)
    else:
        order = (64, 32, 16, 8)
    for tv in order:
        if V % tv == 0:
            return tv
    return 8


# flush BV = smallest tile (DRAM-latency bound; bv=8 > bv=32 ~18% at large N*HV).
def _select_flush_bv(V):
    for bv in (8, 16, 32):
        if V % bv == 0:
            return bv
    raise ValueError(f"V={V} must be divisible by 8, 16 or 32")


# flush kernel: rank-m rebuild of S_m from compact (u,k,g) scratch (Phase-D, lane=K+vk):
#   S_m[v,k] = prod_{j<m} g_j[k] * S0[v,k] + sum_{i<m} u_i[v] * k_i[k] * prod_{i<j<m} g_j[k]
@cute.kernel
def kda_flush_kvbuffer_vk_kernel(
    h0_source: cute.Tensor,  # [pool*HV, V, K] fp32
    d_buf: cute.Tensor,  # [N, T, HV, V] fp32
    k_buf: cute.Tensor,  # [N, T, HV, K] fp32 raw normalized key k_t
    g_buf: cute.Tensor,  # [N, T, HV, K] fp32 per-step gate g_t
    h0_indices: cute.Tensor,
    m_buf: cute.Tensor,  # [N] int32 per-request accept length (first m tokens)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV

    cache_idx = h0_indices[i_n]
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        m_n = m_buf[i_n]  # this request's accept length (runtime; 1 <= m_n <= T)

        r_acc = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
        r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_suf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

        # Overflow-safe rebuild: descending suffix products (all factors <=1, no division).
        for c in cutlass.range_constexpr(vec_size):
            r_suf[c] = cutlass.Float32(1.0)
        for j in cutlass.range_constexpr(BV * vec_size):
            r_acc[j] = cutlass.Float32(0.0)
        for tt in cutlass.range_constexpr(T):
            i_i = T - 1 - tt
            if i_i < m_n:
                k_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size), (i_n, i_i, i_hv, lane))
                cute.autovec_copy(k_tile, r_k)
                g_tile = cute.local_tile(g_buf, (1, 1, 1, vec_size), (i_n, i_i, i_hv, lane))
                cute.autovec_copy(g_tile, r_g)
                for vv in cutlass.range_constexpr(BV):
                    uval = cutlass.Float32(d_buf[i_n, i_i, i_hv, i_v * BV + vv])
                    for c in cutlass.range_constexpr(vec_size):
                        r_acc[vv * vec_size + c] += uval * r_k[c] * r_suf[c]
                for c in cutlass.range_constexpr(vec_size):
                    r_suf[c] = r_suf[c] * r_g[c]

        # S_m = prefix * S0 + acc, write back (contiguous float4)
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h4[c] = r_suf[c] * r_h4[c] + r_acc[vv * vec_size + c]
            h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_flush_kvbuffer_vk_kernel(
    h0_source: cute.Tensor,
    d_buf: cute.Tensor,
    k_buf: cute.Tensor,
    g_buf: cute.Tensor,
    h0_indices: cute.Tensor,
    m_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles
    kda_flush_kvbuffer_vk_kernel(
        h0_source,
        d_buf,
        k_buf,
        g_buf,
        h0_indices,
        m_buf,
        vec_size,
        num_v_tiles,
        BV,
        HV,
        T,
        K,
        V,
    ).launch(grid=(grid_size, 1, 1), block=[32, 1, 1], smem=0, stream=stream)


_compiled_flush_kvbuffer_kernels: dict[tuple, object] = {}


def _get_compiled_flush_kvbuffer_kernel(N, T, HV, K, V, pool_size, BV, opt_level=3):
    key = (N, T, HV, K, V, pool_size, BV, opt_level)
    if key in _compiled_flush_kvbuffer_kernels:
        return _compiled_flush_kvbuffer_kernels[key]

    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    d_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device="cuda")
    k_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    g_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    m_buf = torch.zeros(N, dtype=torch.int32, device="cuda")

    compiled = cute.compile(
        run_kda_flush_kvbuffer_vk_kernel,
        from_dlpack(h0_source, assumed_align=16),
        from_dlpack(d_buf, assumed_align=16),
        from_dlpack(k_buf, assumed_align=16),
        from_dlpack(g_buf, assumed_align=16),
        from_dlpack(h0_indices, assumed_align=16),
        from_dlpack(m_buf, assumed_align=16),
        vec_size=VEC_SIZE,
        BV=BV,
        HV=HV,
        T=T,
        K=K,
        V=V,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_flush_kvbuffer_kernels[key] = compiled
    logger.info(f"CuTe DSL KDA flush KVBuffer kernel compiled: N={N}, T={T}, HV={HV}, K={K}, V={V}, BV={BV}")
    return compiled


def kda_flush_kvbuffer(
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    d_buffer: torch.Tensor,
    k_buffer: torch.Tensor,
    g_buffer: torch.Tensor,
    accept_len,  # int (broadcast to all N) OR per-request [N] int tensor; each in [1, T]
    bv: int = -1,
    opt_level: int = 3,
) -> torch.Tensor:
    N, T, HV, V = d_buffer.shape
    K = k_buffer.shape[3]
    if isinstance(accept_len, torch.Tensor):
        assert accept_len.numel() == N, f"per-request accept_len must have N={N} entries, got {accept_len.numel()}"
        m_buf = accept_len.to(device=d_buffer.device, dtype=torch.int32).contiguous()
    else:
        m = int(accept_len)
        assert 1 <= m <= T, f"accept_len must be in [1,{T}], got {m}"
        m_buf = torch.full((N,), m, dtype=torch.int32, device=d_buffer.device)

    if bv <= 0:
        bv = _select_flush_bv(V)
    assert bv in (8, 16, 32) and V % bv == 0, f"flush bv must be 8/16/32 and divide V, got bv={bv}, V={V}"

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source,
        N=N,
        HV=HV,
        K=K,
        V=V,
        device=initial_state_source.device,
        state_layout="vk",
    )
    initial_state_indices = _normalize_state_indices(
        initial_state_indices, N=N, pool_size=pool_size, device=initial_state_source.device
    )
    stream = _get_cached_stream(initial_state_source.device)

    h0_source_flat = h0_source.view(pool_size * HV, V, K)
    compiled = _get_compiled_flush_kvbuffer_kernel(N, T, HV, K, V, pool_size, bv, opt_level=opt_level)
    compiled(h0_source_flat, d_buffer, k_buffer, g_buffer, initial_state_indices, m_buf, stream)
    return initial_state_source


# ===========================================================================
# MULTILAYER_FLUSH_PATCH: all-layers batched flush, dynamic-N (2D grid x=layer-grid, y=layer).
@cute.kernel
def kda_flush_kvbuffer_vk_ml_kernel(
    h0_source: cute.Tensor,
    d_buf: cute.Tensor,
    k_buf: cute.Tensor,
    g_buf: cute.Tensor,
    h0_indices: cute.Tensor,
    m_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx

    bx, i_l, _ = cute.arch.block_idx()
    i_v = bx % num_v_tiles
    tmp = bx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV

    cache_idx = h0_indices[i_n]
    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        m_n = m_buf[i_n]

        r_acc = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
        r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_suf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

        # Overflow-safe rebuild via descending suffix products (see single-layer flush).
        for c in cutlass.range_constexpr(vec_size):
            r_suf[c] = cutlass.Float32(1.0)
        for j in cutlass.range_constexpr(BV * vec_size):
            r_acc[j] = cutlass.Float32(0.0)
        for tt in cutlass.range_constexpr(T):
            i_i = T - 1 - tt
            if i_i < m_n:
                k_tile = cute.local_tile(k_buf, (1, 1, 1, 1, vec_size), (i_l, i_n, i_i, i_hv, lane))
                cute.autovec_copy(k_tile, r_k)
                g_tile = cute.local_tile(g_buf, (1, 1, 1, 1, vec_size), (i_l, i_n, i_i, i_hv, lane))
                cute.autovec_copy(g_tile, r_g)
                for vv in cutlass.range_constexpr(BV):
                    uval = cutlass.Float32(d_buf[i_l, i_n, i_i, i_hv, i_v * BV + vv])
                    for c in cutlass.range_constexpr(vec_size):
                        r_acc[vv * vec_size + c] += uval * r_k[c] * r_suf[c]
                for c in cutlass.range_constexpr(vec_size):
                    r_suf[c] = r_suf[c] * r_g[c]

        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            h_tile = cute.local_tile(h0_source, (1, 1, 1, vec_size), (i_l, flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h4[c] = r_suf[c] * r_h4[c] + r_acc[vv * vec_size + c]
            h_out = cute.local_tile(h0_source, (1, 1, 1, vec_size), (i_l, flat_state_idx, v_global, lane))
            cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_flush_kvbuffer_vk_ml_kernel(
    h0_source: cute.Tensor,
    d_buf: cute.Tensor,
    k_buf: cute.Tensor,
    g_buf: cute.Tensor,
    h0_indices: cute.Tensor,
    m_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    L = h0_source.layout.shape[0]
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    gx = n_indices * HV * num_v_tiles
    kda_flush_kvbuffer_vk_ml_kernel(
        h0_source,
        d_buf,
        k_buf,
        g_buf,
        h0_indices,
        m_buf,
        vec_size,
        num_v_tiles,
        BV,
        HV,
        T,
        K,
        V,
    ).launch(grid=(gx, L, 1), block=[32, 1, 1], smem=0, stream=stream)


_compiled_flush_kvbuffer_ml_kernels: dict[tuple, object] = {}


def _get_compiled_flush_kvbuffer_ml_kernel(
    L, T, HV, K, V, pool_size, kvb_pool, BV, h0_source, d_buf, k_buf, g_buf, h0_indices, m_buf, opt_level=3
):
    # Trace on the tensors passed in. N not in key (index layout-dynamic).
    key = (L, T, HV, K, V, pool_size, kvb_pool, BV, opt_level)
    if key in _compiled_flush_kvbuffer_ml_kernels:
        return _compiled_flush_kvbuffer_ml_kernels[key]

    compiled = cute.compile(
        run_kda_flush_kvbuffer_vk_ml_kernel,
        from_dlpack(h0_source, assumed_align=16),
        from_dlpack(d_buf, assumed_align=16),
        from_dlpack(k_buf, assumed_align=16),
        from_dlpack(g_buf, assumed_align=16),
        from_dlpack(h0_indices, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(m_buf, assumed_align=16).mark_layout_dynamic(),
        vec_size=VEC_SIZE,
        BV=BV,
        HV=HV,
        T=T,
        K=K,
        V=V,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_flush_kvbuffer_ml_kernels[key] = compiled
    logger.info(f"CuTe DSL KDA flush KVBuffer ML(dyn-N) kernel compiled: L={L}, T={T}, HV={HV}, K={K}, V={V}, BV={BV}")
    return compiled


def kda_flush_kvbuffer_all_layers(
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    d_buffer: torch.Tensor,
    k_buffer: torch.Tensor,
    g_buffer: torch.Tensor,
    accept_len,
    bv: int = -1,
    opt_level: int = 3,
) -> torch.Tensor:
    L, kvb_pool, T, HV, V = d_buffer.shape
    K = k_buffer.shape[4]
    N = initial_state_indices.shape[0]
    if isinstance(accept_len, torch.Tensor):
        assert accept_len.numel() == N, f"per-request accept_len must have N={N} entries, got {accept_len.numel()}"
        m_buf = accept_len.to(device=d_buffer.device, dtype=torch.int32).contiguous()
    else:
        m = int(accept_len)
        assert 1 <= m <= T, f"accept_len must be in [1,{T}], got {m}"
        m_buf = torch.full((N,), m, dtype=torch.int32, device=d_buffer.device)

    if bv <= 0:
        bv = _select_flush_bv(V)
    assert bv in (8, 16, 32) and V % bv == 0, f"flush bv must be 8/16/32 and divide V, got bv={bv}, V={V}"

    pool_size = initial_state_source.shape[1]
    h0_source_flat = initial_state_source.view(L, pool_size * HV, V, K)
    idx = _normalize_state_indices(initial_state_indices, N=N, pool_size=pool_size, device=initial_state_source.device)
    stream = _get_cached_stream(initial_state_source.device)

    compiled = _get_compiled_flush_kvbuffer_ml_kernel(
        L,
        T,
        HV,
        K,
        V,
        pool_size,
        kvb_pool,
        bv,
        h0_source_flat,
        d_buffer,
        k_buffer,
        g_buffer,
        idx,
        m_buf,
        opt_level=opt_level,
    )
    compiled(h0_source_flat, d_buffer, k_buffer, g_buffer, idx, m_buf, stream)
    return initial_state_source


# ---------------------------------------------------------------------------
# shuffle-kvbuffer: token-parallel chunkwise verify (structure B). UT-transform
# W = L^{-1} diag(beta) makes the consumer solve dependence-free: u = W @ (v - S0 kdec).
# ---------------------------------------------------------------------------
@cute.kernel
def kda_mtp_shuffle_kvbuffer_kernel(
    h0_source: cute.Tensor,  # [pool*HV, V, K] fp32 (vk)
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    d_buf: cute.Tensor,  # [N, T, HV, V] fp32
    k_buf: cute.Tensor,  # [N, T, HV, K] fp32 raw normalized key k_t
    g_buf: cute.Tensor,  # [N, T, HV, K] fp32 per-step gate g_t
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    num_warps: cutlass.Constexpr[int] = 4

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    # SMEM. sKdec/sQdec double as staging for k_norm/q_scaled between Stage 1 and 2.
    smem = cutlass.utils.SmemAllocator()
    sKdec = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sKn = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sQdec = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sBrun = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)
    sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16)
    sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16)
    sW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, T), stride=(T, 1)), 16)

    r_qbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_kbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_qf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_tmp = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((ilp_rows, vec_size), stride=(vec_size, 1)), cutlass.Float32)
    # r_part: ilp_rows*T batched partials (Skdec, then reused as x = v - Skdec, then Sqdec).
    r_part = cute.make_rmem_tensor(cute.make_layout((ilp_rows, T), stride=(T, 1)), cutlass.Float32)
    r_u = cute.make_rmem_tensor(cute.make_layout((ilp_rows, T), stride=(T, 1)), cutlass.Float32)
    # Stage-3 pair partials: ceil(T*T/4) per warp.
    ppw: cutlass.Constexpr[int] = (T * T + num_warps - 1) // num_warps
    r_red = cute.make_rmem_tensor(cute.make_layout((ppw,), stride=(1,)), cutlass.Float32)

    if cache_idx >= 0:
        k_start = lane_id * vec_size
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_warps
        flat_state_idx = cache_idx * HV + i_hv

        # ---- Stage 1: token-parallel gating/l2norm (warp w owns tokens w, w+4, ...) ----
        for c in cutlass.range_constexpr(vec_size):
            r_dtb[c] = cutlass.Float32(dt_bias[i_hv, k_start + c])
        tokens_per_warp: cutlass.Constexpr[int] = (T + num_warps - 1) // num_warps
        for tt in cutlass.range_constexpr(tokens_per_warp):
            t_tok = tt * num_warps + warp_idx
            if t_tok < T:
                q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
                k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
                cute.autovec_copy(q_tile, r_qbf)
                cute.autovec_copy(k_tile, r_kbf)
                for c in cutlass.range_constexpr(vec_size):
                    r_qf[c] = cutlass.Float32(r_qbf[c])
                    r_kf[c] = cutlass.Float32(r_kbf[c])

                if cutlass.const_expr(use_qk_l2norm):
                    sum_q = cutlass.Float32(0.0)
                    sum_k = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        sum_q += r_qf[c] * r_qf[c]
                        sum_k += r_kf[c] * r_kf[c]
                    for off in [16, 8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=off, mask=-1, mask_and_clamp=31)
                        sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=off, mask=-1, mask_and_clamp=31)
                    inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
                    inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
                    for c in cutlass.range_constexpr(vec_size):
                        r_qf[c] = r_qf[c] * inv_q
                        r_kf[c] = r_kf[c] * inv_k
                else:
                    for c in cutlass.range_constexpr(vec_size):
                        r_qf[c] = r_qf[c] * scale

                # gate g_t per channel; stage k_norm/q_scaled (decay applied in Stage 2)
                for c in cutlass.range_constexpr(vec_size):
                    x = cutlass.Float32(a[i_n, t_tok, i_hv, k_start + c]) + r_dtb[c]
                    if cutlass.const_expr(use_lower_bound):
                        sigmoid_ax = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-r_exp_A * x, fastmath=fast_math))
                        sG[t_tok, k_start + c] = cute.exp(lower_bound * sigmoid_ax, fastmath=fast_math)
                    else:
                        beta_x = softplus_beta * x
                        exp_bx = cute.exp(beta_x, fastmath=fast_math)
                        sp_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                            cutlass.Float32(1.0) + exp_bx, fastmath=fast_math
                        )
                        use_sp = cutlass.Float32(1.0) if beta_x <= softplus_threshold else cutlass.Float32(0.0)
                        sp_x = use_sp * sp_val + (cutlass.Float32(1.0) - use_sp) * x
                        sG[t_tok, k_start + c] = cute.exp(-r_exp_A * sp_x, fastmath=fast_math)
                    sKdec[t_tok, k_start + c] = r_kf[c]
                    sQdec[t_tok, k_start + c] = r_qf[c]
                if lane_id == 0:
                    sBeta[t_tok] = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, t_tok, i_hv]), fastmath=fast_math)
                    )
        cute.arch.barrier()

        # ---- Stage 2: K-parallel prefix-product scan (thread = one channel).
        kc = tidx  # requires K == 128 == block size
        b_run_s = cutlass.Float32(1.0)
        for i_t in cutlass.range_constexpr(T):
            kn = sKdec[i_t, kc]
            g_t = sG[i_t, kc]
            b_run_s = b_run_s * g_t
            sKdec[i_t, kc] = kn * b_run_s
            sKn[i_t, kc] = kn
            sBrun[i_t, kc] = b_run_s
            if cutlass.const_expr(write_ubuf):
                if i_v == 0:
                    k_buf[i_n, i_t, i_hv, kc] = kn  # raw key (was k/b_run)
                    g_buf[i_n, i_t, i_hv, kc] = g_t  # per-step gate (was b_run)
        cute.arch.barrier()

        # ---- Stage 3: (t,i)-parallel A/P, T^2 pairs round-robined over 4 warps,
        #      ONE batched butterfly per warp. Pair p: p < T*(T-1)/2 -> A, else P. ----
        for j in cutlass.range_constexpr(ppw):
            r_red[j] = cutlass.Float32(0.0)
        p_ctr = 0
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t):  # A[t,i], i<t
                if warp_idx == p_ctr % num_warps:
                    s = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        # decay ratio b_run(t)/b_run(i) = prod_{i<j<=t} g_j <= 1,
                        # accumulated as an ordered product (no division).
                        ratio = cutlass.Float32(1.0)
                        for j in cutlass.range_constexpr(i_t - i_i):
                            ratio = ratio * sG[i_i + 1 + j, k_start + c]
                        s += sKn[i_t, k_start + c] * sKn[i_i, k_start + c] * ratio
                    r_red[p_ctr // num_warps] = s
                p_ctr += 1
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t + 1):  # P[t,i], i<=t
                if warp_idx == p_ctr % num_warps:
                    s = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        ratio = cutlass.Float32(1.0)
                        for j in cutlass.range_constexpr(i_t - i_i):
                            ratio = ratio * sG[i_i + 1 + j, k_start + c]
                        s += sQdec[i_t, k_start + c] * sKn[i_i, k_start + c] * ratio
                    r_red[p_ctr // num_warps] = s
                p_ctr += 1
        for off in [16, 8, 4, 2, 1]:
            for j in cutlass.range_constexpr(ppw):
                r_red[j] = r_red[j] + cute.arch.shuffle_sync_bfly(r_red[j], offset=off, mask=-1, mask_and_clamp=31)
        p_ctr = 0
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t):
                if warp_idx == p_ctr % num_warps:
                    if lane_id == 0:
                        sA[i_t, i_i] = r_red[p_ctr // num_warps]
                p_ctr += 1
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t + 1):
                if warp_idx == p_ctr % num_warps:
                    if lane_id == 0:
                        sP[i_t, i_i] = r_red[p_ctr // num_warps]
                p_ctr += 1
        cute.arch.barrier()

        # ---- Stage 3.5: warp0 builds W = L^{-1} diag(beta), lane j owns column j.
        # Row recurrence W[t,j] = beta_t*[t==j] - beta_t * sum_{i<t} A[t,i] W[i,j];
        # each lane only reads its own column -> no cross-lane sync needed. ----
        if warp_idx == 0:
            if lane_id < T:
                for i_t in cutlass.range_constexpr(T):
                    eq = cutlass.Float32(1.0) if lane_id == i_t else cutlass.Float32(0.0)
                    acc_w = eq
                    for i_i in cutlass.range_constexpr(i_t):
                        acc_w -= sA[i_t, i_i] * sW[i_i, lane_id]
                    sW[i_t, lane_id] = sBeta[i_t] * acc_w
        cute.arch.barrier()

        # ---- Stage 4: consumer (4 warp groups over V rows), zero serial deps. ----
        n_row_groups: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for rg in cutlass.range_constexpr(n_row_groups):
            v_base = i_v * tile_v + warp_idx * rows_per_group + rg * ilp_rows
            for r in cutlass.range_constexpr(ilp_rows):
                h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_base + r, lane_id))
                cute.autovec_copy(h_tile, cute.slice_(r_h, (r, None)))
            # all T Skdec_t for all ilp_rows rows in ONE batched butterfly
            for r in cutlass.range_constexpr(ilp_rows):
                for i_t in cutlass.range_constexpr(T):
                    s = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        s += r_h[r, c] * sKdec[i_t, k_start + c]
                    r_part[r, i_t] = s
            for off in [16, 8, 4, 2, 1]:
                for r in cutlass.range_constexpr(ilp_rows):
                    for i_t in cutlass.range_constexpr(T):
                        r_part[r, i_t] += cute.arch.shuffle_sync_bfly(r_part[r, i_t], offset=off, mask=-1, mask_and_clamp=31)
            # x = v - Skdec (r_part reused), then u = W @ x (token-parallel, no dep chain)
            for r in cutlass.range_constexpr(ilp_rows):
                for i_t in cutlass.range_constexpr(T):
                    r_part[r, i_t] = cutlass.Float32(v[i_n, i_t, i_hv, v_base + r]) - r_part[r, i_t]
            for r in cutlass.range_constexpr(ilp_rows):
                for i_t in cutlass.range_constexpr(T):
                    acc = cutlass.Float32(0.0)
                    for i_i in cutlass.range_constexpr(i_t + 1):
                        acc += sW[i_t, i_i] * r_part[r, i_i]
                    r_u[r, i_t] = acc
            if cutlass.const_expr(write_ubuf):
                if lane_id == 0:
                    for r in cutlass.range_constexpr(ilp_rows):
                        for i_t in cutlass.range_constexpr(T):
                            d_buf[i_n, i_t, i_hv, v_base + r] = r_u[r, i_t]
            # o_t = Sqdec_t + sum_{i<=t} P[t,i] u_i (Sqdec batched butterfly into r_part)
            if cutlass.const_expr(emit_output):
                for r in cutlass.range_constexpr(ilp_rows):
                    for i_t in cutlass.range_constexpr(T):
                        s = cutlass.Float32(0.0)
                        for c in cutlass.range_constexpr(vec_size):
                            s += r_h[r, c] * sQdec[i_t, k_start + c] * sBrun[i_t, k_start + c]
                        r_part[r, i_t] = s
                for off in [16, 8, 4, 2, 1]:
                    for r in cutlass.range_constexpr(ilp_rows):
                        for i_t in cutlass.range_constexpr(T):
                            r_part[r, i_t] += cute.arch.shuffle_sync_bfly(
                                r_part[r, i_t], offset=off, mask=-1, mask_and_clamp=31
                            )
                for r in cutlass.range_constexpr(ilp_rows):
                    for i_t in cutlass.range_constexpr(T):
                        ov = r_part[r, i_t]
                        for i_i in cutlass.range_constexpr(i_t + 1):
                            ov += sP[i_t, i_i] * r_u[r, i_i]
                        if lane_id == 0:
                            o[(i_n, i_t, i_hv, v_base + r)] = cutlass.BFloat16(ov)
            # final state S_T[v,k] = b_{T-1}[k]*S0[v,k] + sum_t u_t k_t[k]*suf(t)[k],
            # suf(t) = prod_{j>t} g_j accumulated descending (bounded <= 1; the
            # running product ends as the full prefix for the S0 term).
            if cutlass.const_expr(not disable_state_update):
                for r in cutlass.range_constexpr(ilp_rows):
                    for c in cutlass.range_constexpr(vec_size):
                        acc = cutlass.Float32(0.0)
                        suf = cutlass.Float32(1.0)
                        for tt in cutlass.range_constexpr(T):
                            i_t = T - 1 - tt
                            acc += r_u[r, i_t] * sKn[i_t, k_start + c] * suf
                            suf = suf * sG[i_t, k_start + c]
                        r_tmp[c] = suf * r_h[r, c] + acc
                    h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_base + r, lane_id))
                    cute.autovec_copy(r_tmp, h_out)


@cute.jit
def run_kda_mtp_shuffle_kvbuffer_kernel(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    d_buf: cute.Tensor,
    k_buf: cute.Tensor,
    g_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    """shuffle-kvbuffer launcher: grid = N*HV*(V//tile_v), block = 128 (4 warps)."""
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, tile_v)
    grid_size = n_indices * HV * num_v_tiles
    smem_bytes = (
        5 * 4 * T * (K + 8)  # sKdec/sKn/sQdec/sG/sBrun
        + 4 * T  # sBeta
        + 3 * 4 * T * T  # sA/sP/sW
        + 256  # alignment slack
    )
    kda_mtp_shuffle_kvbuffer_kernel(
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        d_buf,
        k_buf,
        g_buf,
        vec_size,
        num_v_tiles,
        tile_v,
        ilp_rows,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        fast_math,
        use_lower_bound,
        lower_bound,
    ).launch(grid=(grid_size, 1, 1), block=[128, 1, 1], smem=smem_bytes, stream=stream)


_compiled_mtp_shuffle_kvbuffer_kernels: dict[tuple, object] = {}


def _dlp_qkv(_t, _dyn):
    # dyn-stride: K-contiguous strided view -> dynamic-layout tensor (no copy);
    # contiguous input keeps the compact (byte-identical) descriptor.
    if _dyn:
        return from_dlpack(_t, assumed_align=16).mark_layout_dynamic(leading_dim=3)
    return from_dlpack(_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=_t.dim_order())


def _get_compiled_mtp_shuffle_kvbuffer_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    tile_v,
    ilp_rows,
    scale,
    use_qk_l2norm,
    disable_state_update,
    emit_output,
    write_ubuf,
    softplus_beta,
    softplus_threshold,
    opt_level=3,
    fast_math=True,
    use_lower_bound=False,
    lower_bound=0.0,
    dyn_stride=False,
):
    key = (
        T,
        H,
        HV,
        K,
        V,
        tile_v,
        ilp_rows,
        scale,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        softplus_beta,
        softplus_threshold,
        opt_level,
        fast_math,
        use_lower_bound,
        lower_bound,
        dyn_stride,
    )
    if key in _compiled_mtp_shuffle_kvbuffer_kernels:
        return _compiled_mtp_shuffle_kvbuffer_kernels[key]

    q = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    d_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device="cuda")
    k_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    g_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")

    compiled_kernel = cute.compile(
        run_kda_mtp_shuffle_kvbuffer_kernel,
        from_dlpack(h0_source, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=h0_source.dim_order()),
        from_dlpack(A_log, assumed_align=16),
        from_dlpack(a, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=a.dim_order()),
        from_dlpack(dt_bias, assumed_align=16),
        _dlp_qkv(q, dyn_stride),
        _dlp_qkv(k, dyn_stride),
        _dlp_qkv(v, dyn_stride),
        from_dlpack(b, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=b.dim_order()),
        from_dlpack(o, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=o.dim_order()),
        from_dlpack(h0_indices, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(d_buf, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=d_buf.dim_order()),
        from_dlpack(k_buf, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=k_buf.dim_order()),
        from_dlpack(g_buf, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=g_buf.dim_order()),
        vec_size=VEC_SIZE,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV,
        T=T,
        H=H,
        K=K,
        V=V,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        fast_math=fast_math,
        use_lower_bound=use_lower_bound,
        lower_bound=lower_bound,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_mtp_shuffle_kvbuffer_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP shuffle-KVBuffer kernel compiled: "
        f"N={N}, T={T}, HV={HV}, K={K}, V={V}, tile_v={tile_v}, ilp_rows={ilp_rows}, "
        f"opt_level={opt_level}, fast_math={fast_math}"
    )
    return compiled_kernel


def _select_shuffle_kvb_ilp_rows(tile_v, T):
    """Largest ilp_rows in {4,2,1} dividing rows_per_group with ilp_rows*T <= 16 — the consumer
    holds two (ilp_rows, T) fp32 register arrays (r_part + r_u), so cap their footprint."""
    rows_per_group = tile_v // 4
    for r in (4, 2, 1):
        if rows_per_group % r == 0 and r * T <= 16:
            return r
    return 1


def kda_decode_mtp_shuffle_kvbuffer(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    out: torch.Tensor | None = None,
    disable_state_update: bool = True,
    emit_output: bool = True,
    d_buffer: torch.Tensor | None = None,
    k_buffer: torch.Tensor | None = None,
    g_buffer: torch.Tensor | None = None,
    tile_v: int = -1,
    ilp_rows: int = -1,
    opt_level: int = 3,
    fast_math: bool = True,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """KDA MTP shuffle-KVBuffer verify (token-parallel chunkwise; flush reuses kda_flush_kvbuffer)."""
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    write_ubuf = d_buffer is not None

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    assert K == TILE_K, f"shuffle-kvbuffer requires K={TILE_K}, got {K}"
    assert K == 128, f"shuffle-kvbuffer Stage-2 scan maps 128 threads to K channels; needs K=128, got {K}"
    assert T <= 32, f"shuffle-kvbuffer W-build uses one lane per token column; needs T<=32, got {T}"

    if tile_v <= 0:
        tile_v = _select_kvb_tile_v(V, N, HV)
    assert V % tile_v == 0, f"shuffle-kvbuffer requires V % tile_v == 0, got V={V}, tile_v={tile_v}"
    assert tile_v % 4 == 0, f"shuffle-kvbuffer requires tile_v % 4 == 0 (4 warps), got {tile_v}"
    rows_per_group = tile_v // 4
    if ilp_rows <= 0:
        ilp_rows = _select_shuffle_kvb_ilp_rows(tile_v, T)
    assert rows_per_group % ilp_rows == 0, (
        f"shuffle-kvbuffer requires (tile_v/4) % ilp_rows == 0, got tile_v={tile_v}, ilp_rows={ilp_rows}"
    )

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source,
        N=N,
        HV=HV,
        K=K,
        V=V,
        device=q.device,
        state_layout="vk",
    )

    a = _normalize_mtp_a(a, N=N, T=T, HV=HV, K=K)
    if b.dim() != 3 or tuple(b.shape) != (N, T, HV):
        raise ValueError(f"Unexpected b shape for MTP dense: {tuple(b.shape)}; expected {(N, T, HV)}")

    o = _prepare_output_tensor(q, out, (N, T, HV, V))

    _dyn_kvb = (
        not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
    )
    q = q if (_dyn_kvb or q.is_contiguous()) else q.contiguous()
    k = k if (_dyn_kvb or k.is_contiguous()) else k.contiguous()
    v = v if (_dyn_kvb or v.is_contiguous()) else v.contiguous()
    a = a if a.is_contiguous() else a.contiguous()
    b = b if b.is_contiguous() else b.contiguous()

    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)
    initial_state_indices = _normalize_state_indices(initial_state_indices, N=N, pool_size=pool_size, device=q.device)

    if write_ubuf:
        if tuple(d_buffer.shape) != (N, T, HV, V):
            raise ValueError(f"d_buffer shape must be {(N, T, HV, V)}, got {tuple(d_buffer.shape)}")
        if tuple(k_buffer.shape) != (N, T, HV, K) or tuple(g_buffer.shape) != (N, T, HV, K):
            raise ValueError(f"k_buffer/g_buffer shape must be {(N, T, HV, K)}")
        d_buf, k_buf, g_buf = d_buffer, k_buffer, g_buffer
    else:
        d_buf = torch.empty(N, T, HV, V, dtype=torch.float32, device=q.device)
        k_buf = torch.empty(N, T, HV, K, dtype=torch.float32, device=q.device)
        g_buf = torch.empty(N, T, HV, K, dtype=torch.float32, device=q.device)

    stream = _get_cached_stream(q.device)

    h0_source_flat = h0_source.view(pool_size * HV, V, K)
    compiled_kernel = _get_compiled_mtp_shuffle_kvbuffer_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        tile_v,
        ilp_rows,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        opt_level=opt_level,
        fast_math=fast_math,
        use_lower_bound=lower_bound is not None,
        lower_bound=(0.0 if lower_bound is None else float(lower_bound)),
        dyn_stride=_dyn_kvb,
    )
    compiled_kernel(
        h0_source_flat,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        initial_state_indices,
        d_buf,
        k_buf,
        g_buf,
        stream,
    )
    return o


# ===========================================================================
# tensor_core-kvbuffer (CuTe tensor-core, flat-in-T): every reduction on warp-level
# mma.sync.m16n8k8.tf32 (llvm.inline_asm wrapper); verify = the BT=8 stacked kernel below.
#
# mma.sync m16n8k8 fragment mapping (PTX ISA), gid = lane>>2, tig = lane&3:
#   A row-major [16,8]: a0=A[gid][tig] a1=A[gid+8][tig] a2=A[gid][tig+4] a3=A[gid+8][tig+4]
#   B col-major [8,8]:  b0=B[tig][gid] b1=B[tig+4][gid]
#   C/D [16,8] f32:     c0=C[gid][2tig] c1=C[gid][2tig+1] c2=C[gid+8][2tig] c3=C[gid+8][2tig+1]
# ===========================================================================

from cutlass._mlir.dialects import arith as _arith  # noqa: E402
from cutlass._mlir.dialects import llvm as _llvm  # noqa: E402
from cutlass.cutlass_dsl import T as _T  # noqa: E402
from cutlass.cutlass_dsl import dsl_user_op  # noqa: E402


@dsl_user_op
def _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """One mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32; returns (d0, d1, d2, d3).

    a*/b* are Float32 values reinterpreted as tf32 (raw f32 bits; HW ignores the low
    mantissa bits — same truncation semantics as Triton's tf32 dots)."""
    f32 = _T.f32()
    i32 = _T.i32()

    def _bits(v):
        vv = v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v
        return _arith.bitcast(i32, vv, loc=loc, ip=ip)

    def _f(v):
        return v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v

    res_ty = _llvm.StructType.get_literal([f32, f32, f32, f32])
    res = _llvm.inline_asm(
        res_ty,
        [_bits(a0), _bits(a1), _bits(a2), _bits(a3), _bits(b0), _bits(b1), _f(c0), _f(c1), _f(c2), _f(c3)],
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = cutlass.Float32(_llvm.extractvalue(f32, res, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(_llvm.extractvalue(f32, res, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(_llvm.extractvalue(f32, res, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(_llvm.extractvalue(f32, res, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


@dsl_user_op
def _tf32_lo(v, *, loc=None, ip=None):
    """Residual v - tf32(v): the low-13-mantissa-bit part of an fp32, as Float32."""
    i32 = _T.i32()
    f32 = _T.f32()
    vv = v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v
    bits = _arith.bitcast(i32, vv, loc=loc, ip=ip)
    mask = _arith.constant(i32, -8192, loc=loc, ip=ip)  # 0xFFFFE000: zero low 13 mantissa bits
    hi_bits = _arith.andi(bits, mask, loc=loc, ip=ip)
    hi = _arith.bitcast(f32, hi_bits, loc=loc, ip=ip)
    lo = _arith.subf(vv, hi, loc=loc, ip=ip)
    return cutlass.Float32(lo)


@dsl_user_op
def _mma_m16n8k8_3xtf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """3xTF32-emulated m16n8k8 GEMM (~fp32 accuracy). 3 tf32 MMA passes:
    hi*hi + hi*lo + lo*hi, lo = x - tf32(x). ~3x the HMMA of one tf32 mma."""
    a0l = _tf32_lo(a0)
    a1l = _tf32_lo(a1)
    a2l = _tf32_lo(a2)
    a3l = _tf32_lo(a3)
    b0l = _tf32_lo(b0)
    b1l = _tf32_lo(b1)
    c0, c1, c2, c3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)
    c0, c1, c2, c3 = _mma_m16n8k8_tf32(a0, a1, a2, a3, b0l, b1l, c0, c1, c2, c3)
    c0, c1, c2, c3 = _mma_m16n8k8_tf32(a0l, a1l, a2l, a3l, b0, b1, c0, c1, c2, c3)
    return c0, c1, c2, c3


_compiled_tensor_core_kvbuffer_kernels: dict[tuple, object] = {}


def _get_compiled_tensor_core_kvbuffer_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    pool_size,
    bv,
    num_v_tiles,
    scale,
    use_qk_l2norm,
    disable_state_update,
    emit_output,
    write_ubuf,
    softplus_beta,
    softplus_threshold,
    opt_level=3,
    fast_math=True,
    use_lower_bound=False,
    lower_bound=0.0,
    dyn_stride=False,
):
    key = (
        T,
        H,
        HV,
        K,
        V,
        bv,
        num_v_tiles,
        scale,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        softplus_beta,
        softplus_threshold,
        opt_level,
        fast_math,
        use_lower_bound,
        lower_bound,
        dyn_stride,
    )
    if key in _compiled_tensor_core_kvbuffer_kernels:
        return _compiled_tensor_core_kvbuffer_kernels[key]

    q = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(N, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    d_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device="cuda")
    k_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")
    g_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device="cuda")

    run_fn = run_kda_mtp_tensor_core_kvbuffer_kernel
    compiled_kernel = cute.compile(
        run_fn,
        from_dlpack(h0_source, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=h0_source.dim_order()),
        from_dlpack(A_log, assumed_align=16),
        from_dlpack(a, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=a.dim_order()),
        from_dlpack(dt_bias, assumed_align=16),
        _dlp_qkv(q, dyn_stride),
        _dlp_qkv(k, dyn_stride),
        _dlp_qkv(v, dyn_stride),
        from_dlpack(b, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=b.dim_order()),
        from_dlpack(o, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=o.dim_order()),
        from_dlpack(h0_indices, assumed_align=16).mark_layout_dynamic(),
        from_dlpack(d_buf, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=d_buf.dim_order()),
        from_dlpack(k_buf, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=k_buf.dim_order()),
        from_dlpack(g_buf, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=g_buf.dim_order()),
        vec_size=VEC_SIZE,
        BV=bv,
        num_v_tiles=num_v_tiles,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        HV=HV,
        T=T,
        H=H,
        K=K,
        V=V,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        fast_math=fast_math,
        use_lower_bound=use_lower_bound,
        lower_bound=lower_bound,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_tensor_core_kvbuffer_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA MTP tensor_core-KVBuffer (tensor-core mma) kernel compiled: "
        f"N={N}, T={T}, HV={HV}, K={K}, V={V}, BV={bv}, num_v_tiles={num_v_tiles}, opt_level={opt_level}"
    )
    return compiled_kernel


def kda_decode_mtp_tensor_core_kvbuffer(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    out: torch.Tensor | None = None,
    disable_state_update: bool = True,
    emit_output: bool = True,
    d_buffer: torch.Tensor | None = None,
    k_buffer: torch.Tensor | None = None,
    g_buffer: torch.Tensor | None = None,
    bv: int = 32,
    num_v_tiles: int = -1,
    opt_level: int = 3,
    fast_math: bool = True,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """KDA MTP decode — CuTe tensor-core kvbuffer VERIFY (port of the Triton gemm op)."""
    N, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    write_ubuf = d_buffer is not None

    if scale is None:
        scale = K**-0.5
    assert K == TILE_K == 128, f"tensor_core-kvbuffer requires K=128, got {K}"
    assert T <= 8, f"tensor_core-kvbuffer (BT stacked) needs T<=8, got {T}"
    assert bv == 32, f"tensor_core-kvbuffer (BT) requires bv=32 (one n-tile per warp), got {bv}"
    assert V % bv == 0 and bv % 16 == 0, f"bv must divide V and be 16-aligned, got {bv}"
    if num_v_tiles <= 0:
        # auto: split V across CTAs until the grid reaches ~512 (fills H200's 132 SMs
        # at small batch); producer redundancy per extra slice is negligible.
        num_v_tiles = 1
        while num_v_tiles < V // bv and N * HV * num_v_tiles < 512:
            num_v_tiles *= 2
    assert (V // bv) % num_v_tiles == 0, f"num_v_tiles must divide V//bv, got num_v_tiles={num_v_tiles}"

    h0_source, pool_size, _ = _normalize_state_source(
        initial_state_source,
        N=N,
        HV=HV,
        K=K,
        V=V,
        device=q.device,
        state_layout="vk",
    )
    a = _normalize_mtp_a(a, N=N, T=T, HV=HV, K=K)
    if b.dim() != 3 or tuple(b.shape) != (N, T, HV):
        raise ValueError(f"Unexpected b shape for MTP dense: {tuple(b.shape)}; expected {(N, T, HV)}")
    o = _prepare_output_tensor(q, out, (N, T, HV, V))
    _dyn_kvb = (
        not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
    )
    q = q if (_dyn_kvb or q.is_contiguous()) else q.contiguous()
    k = k if (_dyn_kvb or k.is_contiguous()) else k.contiguous()
    v = v if (_dyn_kvb or v.is_contiguous()) else v.contiguous()
    a = a if a.is_contiguous() else a.contiguous()
    b = b if b.is_contiguous() else b.contiguous()
    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)
    initial_state_indices = _normalize_state_indices(initial_state_indices, N=N, pool_size=pool_size, device=q.device)

    if write_ubuf:
        if tuple(d_buffer.shape) != (N, T, HV, V):
            raise ValueError(f"d_buffer shape must be {(N, T, HV, V)}, got {tuple(d_buffer.shape)}")
        if tuple(k_buffer.shape) != (N, T, HV, K) or tuple(g_buffer.shape) != (N, T, HV, K):
            raise ValueError(f"k_buffer/g_buffer shape must be {(N, T, HV, K)}")
        d_buf, k_buf, g_buf = d_buffer, k_buffer, g_buffer
    else:
        d_buf = torch.empty(N, T, HV, V, dtype=torch.float32, device=q.device)
        k_buf = torch.empty(N, T, HV, K, dtype=torch.float32, device=q.device)
        g_buf = torch.empty(N, T, HV, K, dtype=torch.float32, device=q.device)

    stream = _get_cached_stream(q.device)
    h0_source_flat = h0_source.view(pool_size * HV, V, K)
    compiled_kernel = _get_compiled_tensor_core_kvbuffer_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        bv,
        num_v_tiles,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        opt_level=opt_level,
        fast_math=fast_math,
        use_lower_bound=lower_bound is not None,
        lower_bound=(0.0 if lower_bound is None else float(lower_bound)),
        dyn_stride=_dyn_kvb,
    )
    compiled_kernel(
        h0_source_flat,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        initial_state_indices,
        d_buf,
        k_buf,
        g_buf,
        stream,
    )
    return o


# ---------------------------------------------------------------------------
# BT=8 stacked variant of the tensor_core kernel (T <= 8). mma.sync m16n8k8 has a
# hard M=16, so instead of padding tokens to 16 the spare 8 M-rows carry a
# SECOND matrix — pad waste becomes a ~2x instruction saving:
#   P3: [kdec; qdec] @ kinv^T   -> A (top) and P (bottom) in one GEMM chain
#   P4: Neumann inverse in plain fp32 (precision); L_s is strictly-lower 8x8 so
#       L_s^8 = 0 -> inv = (I+L_s)(I+L_s^2)(I+L_s^4), exactly 3 doubling steps
#   P5: [kdec; qdec] @ S0^T     -> Skdec + Sqdec together; u = inv @ (beta*x) on
#       tensor cores; o-combine P@u in exact fp32 from SMEM (16 FMA/lane)
# Requires BV=32 (4 n-tiles = 1 per warp, keeps barriers warp-uniform).
# ---------------------------------------------------------------------------
BT = 8


@cute.kernel
def kda_mtp_tensor_core_kvbuffer_kernel(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    d_buf: cute.Tensor,
    k_buf: cute.Tensor,
    g_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    gid = lane_id // 4
    tig = lane_id % 4

    num_warps: cutlass.Constexpr[int] = 4
    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    smem = cutlass.utils.SmemAllocator()
    # stacked feature maps: rows 0..7 = kdec(tokens, pad-zeroed), rows 8..15 = qdec
    sKQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((2 * BT, K), stride=(K + 4, 1)), 16)
    # suffix-decayed keys ksuf_t = kn_t * prod_{j>t} g_j (bounded; replaces kinv)
    sKsuf = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, K), stride=(K + 8, 1)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, K), stride=(K + 8, 1)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT,)), 16)
    sBlast = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    # P3 cross-warp partial tiles: row = warp*16 + stacked-row
    sPart = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4 * 16, 12), stride=(12, 1)), 16)
    sL = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sInv = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sLp = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sX = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BV), stride=(BV + 1, 1)), 16)
    sU = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BV), stride=(BV + 1, 1)), 16)
    sS0 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BV, K), stride=(K + 4, 1)), 16)

    r_qbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_kbf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16)
    r_qf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_s = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    # P2a pair partials: ceil(2*T*T/4) per warp
    ppw_tc: cutlass.Constexpr[int] = (2 * T * T + num_warps - 1) // num_warps
    r_red = cute.make_rmem_tensor(cute.make_layout((ppw_tc,), stride=(1,)), cutlass.Float32)

    if cache_idx >= 0:
        k_start = lane_id * vec_size
        flat_state_idx = cache_idx * HV + i_hv

        # ---- P1: token-parallel l2norm + staging (k_norm -> sKQ top, q_scaled -> bottom) ----
        tokens_per_warp: cutlass.Constexpr[int] = (T + num_warps - 1) // num_warps
        for tt in cutlass.range_constexpr(tokens_per_warp):
            t_tok = tt * num_warps + warp_idx
            if t_tok < T:
                q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
                k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, t_tok, i_h, lane_id))
                cute.autovec_copy(q_tile, r_qbf)
                cute.autovec_copy(k_tile, r_kbf)
                for c in cutlass.range_constexpr(vec_size):
                    r_qf[c] = cutlass.Float32(r_qbf[c])
                    r_kf[c] = cutlass.Float32(r_kbf[c])
                if cutlass.const_expr(use_qk_l2norm):
                    sum_q = cutlass.Float32(0.0)
                    sum_k = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        sum_q += r_qf[c] * r_qf[c]
                        sum_k += r_kf[c] * r_kf[c]
                    for off in [16, 8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=off, mask=-1, mask_and_clamp=31)
                        sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=off, mask=-1, mask_and_clamp=31)
                    inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
                    inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
                    for c in cutlass.range_constexpr(vec_size):
                        r_qf[c] = r_qf[c] * inv_q
                        r_kf[c] = r_kf[c] * inv_k
                else:
                    for c in cutlass.range_constexpr(vec_size):
                        r_qf[c] = r_qf[c] * scale
                # gate g_t per channel into sG (decay applied in P2)
                for c in cutlass.range_constexpr(vec_size):
                    x = cutlass.Float32(a[i_n, t_tok, i_hv, k_start + c]) + cutlass.Float32(dt_bias[i_hv, k_start + c])
                    if cutlass.const_expr(use_lower_bound):
                        sigmoid_ax = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-r_exp_A * x, fastmath=fast_math))
                        sG[t_tok, k_start + c] = cute.exp(lower_bound * sigmoid_ax, fastmath=fast_math)
                    else:
                        beta_x = softplus_beta * x
                        exp_bx = cute.exp(beta_x, fastmath=fast_math)
                        sp_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                            cutlass.Float32(1.0) + exp_bx, fastmath=fast_math
                        )
                        use_sp = cutlass.Float32(1.0) if beta_x <= softplus_threshold else cutlass.Float32(0.0)
                        sp_x = use_sp * sp_val + (cutlass.Float32(1.0) - use_sp) * x
                        sG[t_tok, k_start + c] = cute.exp(
                            -r_exp_A * sp_x, fastmath=fast_math
                        )  # g_t directly (exact prefix product in P2)
                    sKQ[t_tok, k_start + c] = r_kf[c]
                    sKQ[BT + t_tok, k_start + c] = r_qf[c]
                if lane_id == 0:
                    sBeta[t_tok] = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, t_tok, i_hv]), fastmath=fast_math)
                    )
        for rp in cutlass.range_constexpr(BT - T):
            sKQ[T + rp, tidx] = cutlass.Float32(0.0)
            sKQ[BT + T + rp, tidx] = cutlass.Float32(0.0)
            sKsuf[T + rp, tidx] = cutlass.Float32(0.0)
        if tidx >= T:
            if tidx < BT:
                sBeta[tidx] = cutlass.Float32(0.0)
        cute.arch.barrier()

        # ---- P2a: T*T scores in plain fp32 with bounded decay-ratio chains.
        # Runs BEFORE the prefix scaling so sKQ still holds raw kn/q_scaled:
        #   A[t,i] = sum_k kn_t kn_i * r(t,i),  P[t,i] = sum_k qn_t kn_i * r(t,i),
        #   r(t,i) = prod_{i<j<=t} g_j <= 1 (ordered product, no division).
        # T*T pairs round-robined over 4 warps, one butterfly per warp
        # (vec_size channels per lane, 32 lanes = K). ----
        for j in cutlass.range_constexpr(ppw_tc):
            r_red[j] = cutlass.Float32(0.0)
        p_ctr = 0
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t):  # A[t,i], i<t
                if warp_idx == p_ctr % num_warps:
                    s = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        ratio = cutlass.Float32(1.0)
                        for j in cutlass.range_constexpr(i_t - i_i):
                            ratio = ratio * sG[i_i + 1 + j, k_start + c]
                        s += sKQ[i_t, k_start + c] * sKQ[i_i, k_start + c] * ratio
                    r_red[p_ctr // num_warps] = s
                p_ctr += 1
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t + 1):  # P[t,i], i<=t
                if warp_idx == p_ctr % num_warps:
                    s = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        ratio = cutlass.Float32(1.0)
                        for j in cutlass.range_constexpr(i_t - i_i):
                            ratio = ratio * sG[i_i + 1 + j, k_start + c]
                        s += sKQ[BT + i_t, k_start + c] * sKQ[i_i, k_start + c] * ratio
                    r_red[p_ctr // num_warps] = s
                p_ctr += 1
        for off in [16, 8, 4, 2, 1]:
            for j in cutlass.range_constexpr(ppw_tc):
                r_red[j] = r_red[j] + cute.arch.shuffle_sync_bfly(r_red[j], offset=off, mask=-1, mask_and_clamp=31)
        # zero-init L/P (covers padding rows), then scatter the pair results
        if tidx < BT * BT:
            sL[tidx // BT, tidx % BT] = cutlass.Float32(0.0)
            sP[tidx // BT, tidx % BT] = cutlass.Float32(0.0)
        cute.arch.barrier()
        p_ctr = 0
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t):
                if warp_idx == p_ctr % num_warps:
                    if lane_id == 0:
                        sL[i_t, i_i] = -sBeta[i_t] * r_red[p_ctr // num_warps]
                p_ctr += 1
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t + 1):
                if warp_idx == p_ctr % num_warps:
                    if lane_id == 0:
                        sP[i_t, i_i] = r_red[p_ctr // num_warps]
                p_ctr += 1
        cute.arch.barrier()

        # ---- P2b: K-parallel scans (thread = channel kc). Backward suffix pass
        # first (raw kn still in sKQ) -> sKsuf; then forward prefix scaling
        # kdec/qdec; scratch stores raw (k, g) for the bounded flush rebuild. ----
        kc = tidx  # requires K == 128 == block size
        suf_s = cutlass.Float32(1.0)
        for tt in cutlass.range_constexpr(T):
            i_t = T - 1 - tt
            sKsuf[i_t, kc] = sKQ[i_t, kc] * suf_s
            suf_s = suf_s * sG[i_t, kc]
        bcum = cutlass.Float32(1.0)
        for i_t in cutlass.range_constexpr(T):
            g_t = sG[i_t, kc]
            bcum = bcum * g_t
            kn = sKQ[i_t, kc]
            sKQ[i_t, kc] = kn * bcum
            sKQ[BT + i_t, kc] = sKQ[BT + i_t, kc] * bcum
            if cutlass.const_expr(write_ubuf):
                if i_v == 0:
                    k_buf[i_n, i_t, i_hv, kc] = kn  # raw key (was k/b_run)
                    g_buf[i_n, i_t, i_hv, kc] = g_t  # per-step gate (was b_run)
        sBlast[kc] = bcum
        cute.arch.barrier()
        if tidx < BT * BT:
            ri = tidx // BT
            ci = tidx % BT
            one = cutlass.Float32(1.0) if ri == ci else cutlass.Float32(0.0)
            sInv[ri, ci] = one  # inv starts at I: each doubling step does inv += inv@Lp_old
            # (with Lp_old = Ls^(2^step)), so I+Ls is produced by step 0
            sLp[ri, ci] = sL[ri, ci]
        cute.arch.barrier()

        # ---- P4: doubling chain + Pinv on the 8x8 mats in PLAIN fp32
        ri = tidx // BT
        ci = tidx % BT
        for step in cutlass.range_constexpr(3):  # 3 steps: (I+Ls)(I+Ls^2)(I+Ls^4), nilpotency 8
            if tidx < 2 * BT * BT:  # rows 0..7 -> Lp@Lp, rows 8..15 -> inv@Lp
                rr = ri % BT
                acc = cutlass.Float32(0.0)
                for l in cutlass.range_constexpr(BT):
                    if ri < BT:
                        acc += sLp[rr, l] * sLp[l, ci]
                    else:
                        acc += sInv[rr, l] * sLp[l, ci]
                sPart[ri, ci] = acc
            cute.arch.barrier()
            if tidx < BT * BT:
                sLp[ri, ci] = sPart[ri, ci]
                sInv[ri, ci] = sInv[ri, ci] + sPart[BT + ri, ci]
            cute.arch.barrier()

        # ---- P5 consumer. V tiled 3 ways (outer->inner):
        #   num_v_tiles  : V split across CTAs (grid=N*HV*num_v_tiles)
        #   BV=32        : V rows/block = 4 warps x mma-N(8); 1 n-tile/warp, uniform barriers
        #   num_v_blocks : BV-blocks each CTA walks serially
        num_v_blocks: cutlass.Constexpr[int] = V // BV // num_v_tiles
        for vb in cutlass.range_constexpr(num_v_blocks):
            v_base = (i_v * num_v_blocks + vb) * BV  # global V-row start of this block
            row_vecs = K // vec_size  # float4s per V row
            # stage S0[BV,K] -> sS0: 128 threads (blockDim), one float4 each;
            # passes = BV*K / (128*vec_size)
            for j in cutlass.range_constexpr(BV * K // (128 * vec_size)):
                flat = j * 128 + tidx  # float4-group id
                s_row = flat // row_vecs  # V row
                s_col = flat % row_vecs  # float4 within row
                h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_base + s_row, s_col))
                cute.autovec_copy(h_tile, r_s)
                for cc in cutlass.range_constexpr(vec_size):
                    sS0[s_row, s_col * vec_size + cc] = r_s[cc]
            cute.arch.barrier()

            nb = warp_idx * 8  # current warp's n-tile = V rows [nb, nb+8) within the BV block
            # the two adjacent V indices this lane owns (mma N-frag: 2*tig, 2*tig+1)
            vc0 = nb + 2 * tig
            vc1 = nb + 2 * tig + 1
            # GEMM1: [kdec; qdec] @ S0^T -> Skdec (rows 0..7) + Sqdec (rows 8..15)
            e0 = cutlass.Float32(0.0)
            e1 = cutlass.Float32(0.0)
            e2 = cutlass.Float32(0.0)
            e3 = cutlass.Float32(0.0)
            for ks in cutlass.range_constexpr(K // 8):
                kb = ks * 8
                a0 = sKQ[gid, kb + tig]
                a1 = sKQ[gid + 8, kb + tig]
                a2 = sKQ[gid, kb + tig + 4]
                a3 = sKQ[gid + 8, kb + tig + 4]
                b0 = sS0[nb + gid, kb + tig]
                b1 = sS0[nb + gid, kb + tig + 4]
                e0, e1, e2, e3 = _mma_m16n8k8_3xtf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
            # x = beta * (v - Skdec) from the top half; Sqdec (e2/e3) stays in registers
            vmask = cutlass.Float32(1.0) if gid < T else cutlass.Float32(0.0)
            vv0 = cutlass.Float32(v[i_n, gid % T, i_hv, v_base + vc0]) * vmask
            vv1 = cutlass.Float32(v[i_n, gid % T, i_hv, v_base + vc1]) * vmask
            sX[gid, vc0] = sBeta[gid] * (vv0 - e0)
            sX[gid, vc1] = sBeta[gid] * (vv1 - e1)
            cute.arch.barrier()

            # u = inv @ x in exact fp32
            f0 = cutlass.Float32(0.0)
            f1 = cutlass.Float32(0.0)
            for l in cutlass.range_constexpr(BT):
                f0 += sInv[gid, l] * sX[l, vc0]
                f1 += sInv[gid, l] * sX[l, vc1]
            sU[gid, vc0] = f0
            sU[gid, vc1] = f1
            if cutlass.const_expr(write_ubuf):
                if gid < T:
                    d_buf[i_n, gid, i_hv, v_base + vc0] = f0
                    d_buf[i_n, gid, i_hv, v_base + vc1] = f1
            cute.arch.barrier()
            # o = Sqdec + P@u combined in exact fp32 from sU (16 FMA/lane — removes the
            # extra tf32 hop that the stacked [inv;Pinv]@x route put on the output path)
            if cutlass.const_expr(emit_output):
                if gid < T:
                    ov0 = e2
                    ov1 = e3
                    for l in cutlass.range_constexpr(BT):
                        ov0 += sP[gid, l] * sU[l, vc0]
                        ov1 += sP[gid, l] * sU[l, vc1]
                    o[(i_n, gid, i_hv, v_base + vc0)] = cutlass.BFloat16(ov0)
                    o[(i_n, gid, i_hv, v_base + vc1)] = cutlass.BFloat16(ov1)

            # state: S_T = b_last * S0 + u^T @ ksuf (ksuf bounded; b_last only
            # rescales the S0 term), M = v rows, single k-slab
            if cutlass.const_expr(not disable_state_update):
                m_tiles: cutlass.Constexpr[int] = BV // 16
                pairs: cutlass.Constexpr[int] = m_tiles * (K // 8)
                for pp in cutlass.range_constexpr((pairs + num_warps - 1) // num_warps):
                    pidx = pp * num_warps + warp_idx
                    if pidx < pairs:
                        m_t = pidx % m_tiles
                        n_t = pidx // m_tiles
                        mb = m_t * 16
                        nb = n_t * 8
                        g0 = cutlass.Float32(0.0)
                        g1 = cutlass.Float32(0.0)
                        g2 = cutlass.Float32(0.0)
                        g3 = cutlass.Float32(0.0)
                        a0 = sU[tig, mb + gid]
                        a1 = sU[tig, mb + gid + 8]
                        a2 = sU[tig + 4, mb + gid]
                        a3 = sU[tig + 4, mb + gid + 8]
                        b0 = sKsuf[tig, nb + gid]
                        b1 = sKsuf[tig + 4, nb + gid]
                        # 3xTF32 for near-fp32 state precision; only the dsu=0
                        # path hits this GEMM (serving verify commits via flush).
                        g0, g1, g2, g3 = _mma_m16n8k8_3xtf32(a0, a1, a2, a3, b0, b1, g0, g1, g2, g3)
                        for fi in cutlass.range_constexpr(4):
                            vrow = mb + gid + (fi // 2) * 8
                            kcol = nb + 2 * tig + (fi % 2)
                            gv = g0
                            if cutlass.const_expr(fi == 1):
                                gv = g1
                            if cutlass.const_expr(fi == 2):
                                gv = g2
                            if cutlass.const_expr(fi == 3):
                                gv = g3
                            h0_source[(flat_state_idx, v_base + vrow, kcol)] = sBlast[kcol] * sS0[vrow, kcol] + gv
            cute.arch.barrier()


@cute.jit
def run_kda_mtp_tensor_core_kvbuffer_kernel(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    d_buf: cute.Tensor,
    k_buf: cute.Tensor,
    g_buf: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    """BT=8 stacked tensor_core launcher: grid = N*HV*num_v_tiles, block = 128."""
    n_indices = h0_indices.layout.shape[0]
    grid_size = n_indices * HV * num_v_tiles
    smem_bytes = (
        2 * 4 * BT * (K + 8)  # sKQ (stacked)
        + 2 * 4 * BT * (K + 8)  # sKsuf + sG
        + 4 * BT
        + 4 * K  # sBeta + sBlast
        + 4 * 64 * 12  # sPart
        + 4 * 4 * BT * (BT + 1)  # sL/sP/sInv/sLp
        + 2 * 4 * BT * (BV + 1)  # sX/sU
        + 4 * BV * (K + 8)  # sS0
        + 512
    )
    kda_mtp_tensor_core_kvbuffer_kernel(
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        d_buf,
        k_buf,
        g_buf,
        vec_size,
        BV,
        num_v_tiles,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        fast_math,
        use_lower_bound,
        lower_bound,
    ).launch(grid=(grid_size, 1, 1), block=[128, 1, 1], smem=smem_bytes, stream=stream)


# ---------------------------------------------------------------------------
# KVBuffer verify dispatch: route between the two kvbuffer verify ops by T.
# ---------------------------------------------------------------------------
def _select_kvb_variant(N: int, HV: int, T: int) -> str:
    """Pick "shuffle" or "tensor_core" kvbuffer variant; wu = N*HV."""
    wu = N * HV
    if T <= 2:
        return "shuffle"
    if T == 3:
        return "shuffle" if wu <= 64 else "tensor_core"
    if T == 4:
        return "shuffle" if wu <= 32 else "tensor_core"
    return "tensor_core"


def _kvbuffer_prefer_tensor_core(N: int, HV: int, T: int) -> bool:
    """True iff the kvbuffer dispatch picks tensor_core."""
    return _select_kvb_variant(N, HV, T) == "tensor_core"


def kda_decode_mtp_kvbuffer(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    out: torch.Tensor | None = None,
    disable_state_update: bool = True,
    emit_output: bool = True,
    d_buffer: torch.Tensor | None = None,
    k_buffer: torch.Tensor | None = None,
    g_buffer: torch.Tensor | None = None,
    t_crossover: int | None = None,
    opt_level: int = 3,
    fast_math: bool = True,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """KDA MTP KVBuffer verify dispatch between shuffle-kvbuffer (token-parallel SIMT) and
    tensor_core-kvbuffer (CuTe tensor-core GEMM, flat-in-T). With ``t_crossover=None``
    (default) the choice follows the kernel-level chain bench via
    ``_kvbuffer_prefer_tensor_core`` (a function of the work size S = HV*N and T); pass an
    int to force the legacy T-only rule (tensor_core iff T >= t_crossover). Routes only
    among kvbuffer ops; the recurrent fallback is a higher-layer concern.
    """
    T = q.shape[1]
    N = q.shape[0]
    HV = v.shape[2]
    common = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        out=out,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        d_buffer=d_buffer,
        k_buffer=k_buffer,
        g_buffer=g_buffer,
        opt_level=opt_level,
        fast_math=fast_math,
        lower_bound=lower_bound,
    )
    if t_crossover is None:
        use_tensor_core = _select_kvb_variant(N, HV, T) == "tensor_core"
    else:
        use_tensor_core = t_crossover <= T
    if use_tensor_core:
        return kda_decode_mtp_tensor_core_kvbuffer(**common)
    return kda_decode_mtp_shuffle_kvbuffer(**common)
