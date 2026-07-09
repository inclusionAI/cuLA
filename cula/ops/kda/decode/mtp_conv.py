"""Fused KDA causal-conv1d + MTP verify, CuTe DSL. Two variants + auto dispatch.

Fuses the depthwise causal conv1d (width 4, SiLU) into the sigmoid-gating
delta-rule MTP verify recurrence, consuming packed ``mixed_qkv`` directly (no
split/transpose). Matches the sglang Triton ``fused_kda_conv_gating_verify``
semantics: per-step conv-window + SSM snapshots, conv_state rolled at the
epilogue, real SSM state not written back (verify). Scope: chain, W=4, T>=W-1.

vk variant (small batch): 1 warp/CTA, grid = N*HV*(V//BV); lane holds
  K[4*lane:4*lane+4]. q/k conv per-lane; v conv Option B (lane<BV computes one
  v-col, shfl.idx broadcast). auto bv: 8 (<192 work) else 32.

ws variant (large batch): 4 warps/CTA, grid = N*HV*(V//tile_v), tile_v=4*BVW.
  Warp 0 produces shared q/k conv + l2norm + gate + beta -> SMEM (no per-v-tile
  redundant q/k); each warp does BVW v-cols of recurrence+v-conv+snapshot. Low
  state regs (r_h=BVW*vec_size) -> high occupancy -> beats triton at large N.

Shared q/k conv_state (read by every v-tile/v-head sharing a q/k head, rolled +
written once) is written by the LAST CTA in the sharing group (largest bidx),
so its epilogue write lands after every earlier-dispatched non-owner has read
the history -> race-free, same rolled output as the reference.

Dispatch (variant="auto"): ws if N*HV>=64 else vk (graph-timed vk/ws crossover).
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

logger = logging.getLogger(__name__)

VEC_SIZE = 4
WCONV = 4  # conv width (KDA short_conv_kernel_size)


@cute.kernel
def kda_conv_verify_vk_kernel(
    mixed_qkv: cute.Tensor,     # [N*T, D] bf16 pre-conv
    conv_weight: cute.Tensor,   # [D, W] fp32
    conv_bias: cute.Tensor,     # [D] fp32
    conv_state: cute.Tensor,    # [lines, D, W-1] fp32 (dim contiguous)
    conv_state_indices: cute.Tensor,  # [B] int32 mamba slot
    inter_conv_window: cute.Tensor,   # [lines, T, D, W-1] fp32
    inter_state_indices: cute.Tensor,  # [B] int32
    h0_source: cute.Tensor,     # [pool*HV, V, K] fp32
    A_log: cute.Tensor,
    a: cute.Tensor,             # [N, T, HV, K] bf16
    dt_bias: cute.Tensor,       # [HV, K] fp32
    b: cute.Tensor,             # [N, T, HV] bf16
    o: cute.Tensor,             # [N, T, HV, V] bf16
    intermediate_states: cute.Tensor,  # [N*T*HV, V, K] fp32
    h0_indices: cute.Tensor,    # [B] int32
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    save_conv_window: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx  # 1 warp = 32 lanes

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cs_idx = conv_state_indices[i_n]
    cache_idx = h0_indices[i_n]
    iw_idx = inter_state_indices[i_n]
    # owner = LAST CTA (largest bidx) among the CTAs sharing this q/k head, so
    # its epilogue conv_state write happens after every non-owner has read the
    # history at its preamble (bidx ~ dispatch order) -> race-free shared write.
    # Same rolled value as any owner would write, so behavior matches the reference.
    is_qk_owner = (i_v == num_v_tiles - 1) and (i_hv % (HV // H) == (HV // H) - 1)

    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    # packed channel bases
    q_base = i_h * K
    k_base = H * K + i_h * K
    v_base = 2 * H * K + i_hv * V + i_v * BV

    # ---- state tile [BV*vec_size] (vk: r_h[vv*vec+c]=state[i_v*BV+vv, vec*lane+c]) ----
    r_h = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
    r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

    # ---- conv registers ----
    r_qhist = cute.make_rmem_tensor(cute.make_layout(((W - 1) * vec_size,), stride=(1,)), cutlass.Float32)
    r_khist = cute.make_rmem_tensor(cute.make_layout(((W - 1) * vec_size,), stride=(1,)), cutlass.Float32)
    r_qw = cute.make_rmem_tensor(cute.make_layout((W * vec_size,), stride=(1,)), cutlass.Float32)
    r_kw = cute.make_rmem_tensor(cute.make_layout((W * vec_size,), stride=(1,)), cutlass.Float32)
    r_qb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    # v conv (Option B): each lane owns ONE v-channel (lane < BV), then broadcast.
    r_vhist = cute.make_rmem_tensor(cute.make_layout((W - 1,), stride=(1,)), cutlass.Float32)
    r_vw = cute.make_rmem_tensor(cute.make_layout((W,), stride=(1,)), cutlass.Float32)
    r_vb = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32)

    # ---- recurrence scratch ----
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_v = cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.Float32)
    r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_gx = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_red = cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.Float32)

    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv

        # ---- state load (float4: lane owns K[4*lane:4*lane+4]) ----
        for vv in cutlass.range_constexpr(BV):
            v_global = i_v * BV + vv
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]

        # ---- conv preload: q/k history + weights + bias (per lane's 4 K-channels) ----
        for c in cutlass.range_constexpr(vec_size):
            qch = q_base + vec_size * lane + c
            kch = k_base + vec_size * lane + c
            r_dtb[c] = cutlass.Float32(dt_bias[i_hv, vec_size * lane + c])
            for w in cutlass.range_constexpr(W - 1):
                r_qhist[w * vec_size + c] = cutlass.Float32(conv_state[cs_idx, qch, w])
                r_khist[w * vec_size + c] = cutlass.Float32(conv_state[cs_idx, kch, w])
            for w in cutlass.range_constexpr(W):
                r_qw[w * vec_size + c] = cutlass.Float32(conv_weight[qch, w])
                r_kw[w * vec_size + c] = cutlass.Float32(conv_weight[kch, w])
            if cutlass.const_expr(has_bias):
                r_qb[c] = cutlass.Float32(conv_bias[qch])
                r_kb[c] = cutlass.Float32(conv_bias[kch])
            else:
                r_qb[c] = cutlass.Float32(0.0)
                r_kb[c] = cutlass.Float32(0.0)

        # ---- conv preload: v (Option B: lane owns v-channel `lane`, lane<BV) ----
        if lane < BV:
            vch = v_base + lane
            r_vhist[0] = cutlass.Float32(conv_state[cs_idx, vch, 0])
            r_vhist[1] = cutlass.Float32(conv_state[cs_idx, vch, 1])
            r_vhist[2] = cutlass.Float32(conv_state[cs_idx, vch, 2])
            r_vw[0] = cutlass.Float32(conv_weight[vch, 0])
            r_vw[1] = cutlass.Float32(conv_weight[vch, 1])
            r_vw[2] = cutlass.Float32(conv_weight[vch, 2])
            r_vw[3] = cutlass.Float32(conv_weight[vch, 3])
            if cutlass.const_expr(has_bias):
                r_vb[0] = cutlass.Float32(conv_bias[vch])
            else:
                r_vb[0] = cutlass.Float32(0.0)

        for i_t in cutlass.range_constexpr(T):
            row = i_n * T + i_t

            # ===== q conv (per lane, 4 K-channels) =====
            for c in cutlass.range_constexpr(vec_size):
                qch = q_base + vec_size * lane + c
                xq = cutlass.Float32(mixed_qkv[row, qch])
                acc = r_qb[c]
                acc = acc + r_qhist[0 * vec_size + c] * r_qw[0 * vec_size + c]
                acc = acc + r_qhist[1 * vec_size + c] * r_qw[1 * vec_size + c]
                acc = acc + r_qhist[2 * vec_size + c] * r_qw[2 * vec_size + c]
                acc = acc + xq * r_qw[3 * vec_size + c]
                # roll
                r_qhist[0 * vec_size + c] = r_qhist[1 * vec_size + c]
                r_qhist[1 * vec_size + c] = r_qhist[2 * vec_size + c]
                r_qhist[2 * vec_size + c] = xq
                if cutlass.const_expr(save_conv_window):
                    if is_qk_owner:
                        inter_conv_window[iw_idx, i_t, qch, 0] = r_qhist[0 * vec_size + c]
                        inter_conv_window[iw_idx, i_t, qch, 1] = r_qhist[1 * vec_size + c]
                        inter_conv_window[iw_idx, i_t, qch, 2] = r_qhist[2 * vec_size + c]
                silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                r_q[c] = cutlass.Float32(cutlass.BFloat16(silu))

            # ===== k conv =====
            for c in cutlass.range_constexpr(vec_size):
                kch = k_base + vec_size * lane + c
                xk = cutlass.Float32(mixed_qkv[row, kch])
                acc = r_kb[c]
                acc = acc + r_khist[0 * vec_size + c] * r_kw[0 * vec_size + c]
                acc = acc + r_khist[1 * vec_size + c] * r_kw[1 * vec_size + c]
                acc = acc + r_khist[2 * vec_size + c] * r_kw[2 * vec_size + c]
                acc = acc + xk * r_kw[3 * vec_size + c]
                r_khist[0 * vec_size + c] = r_khist[1 * vec_size + c]
                r_khist[1 * vec_size + c] = r_khist[2 * vec_size + c]
                r_khist[2 * vec_size + c] = xk
                if cutlass.const_expr(save_conv_window):
                    if is_qk_owner:
                        inter_conv_window[iw_idx, i_t, kch, 0] = r_khist[0 * vec_size + c]
                        inter_conv_window[iw_idx, i_t, kch, 1] = r_khist[1 * vec_size + c]
                        inter_conv_window[iw_idx, i_t, kch, 2] = r_khist[2 * vec_size + c]
                silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                r_k[c] = cutlass.Float32(cutlass.BFloat16(silu))

            # ===== v conv (Option B: lane computes channel `lane`, shuffle-broadcast) =====
            my_v = cutlass.Float32(0.0)
            if lane < BV:
                vch = v_base + lane
                xv = cutlass.Float32(mixed_qkv[row, vch])
                acc = r_vb[0]
                acc = acc + r_vhist[0] * r_vw[0]
                acc = acc + r_vhist[1] * r_vw[1]
                acc = acc + r_vhist[2] * r_vw[2]
                acc = acc + xv * r_vw[3]
                r_vhist[0] = r_vhist[1]
                r_vhist[1] = r_vhist[2]
                r_vhist[2] = xv
                if cutlass.const_expr(save_conv_window):
                    inter_conv_window[iw_idx, i_t, vch, 0] = r_vhist[0]
                    inter_conv_window[iw_idx, i_t, vch, 1] = r_vhist[1]
                    inter_conv_window[iw_idx, i_t, vch, 2] = r_vhist[2]
                silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                my_v = cutlass.Float32(cutlass.BFloat16(silu))
            # broadcast lane vv's post-conv v to all lanes (shfl.idx)
            for vv in cutlass.range_constexpr(BV):
                r_v[vv] = cute.arch.shuffle_sync(my_v, vv, mask=-1, mask_and_clamp=31)

            # ===== gate stage 1: x = a + dt_bias =====
            for c in cutlass.range_constexpr(vec_size):
                r_gx[c] = cutlass.Float32(a[i_n, i_t, i_hv, vec_size * lane + c]) + r_dtb[c]

            # ===== l2norm + scale (q/k) =====
            if cutlass.const_expr(use_qk_l2norm):
                sum_q = cutlass.Float32(0.0)
                sum_k = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    sum_q += r_q[c] * r_q[c]
                    sum_k += r_k[c] * r_k[c]
                for off in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=off, mask=-1, mask_and_clamp=31)
                    sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=off, mask=-1, mask_and_clamp=31)
                inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
                inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
                for c in cutlass.range_constexpr(vec_size):
                    r_q[c] = r_q[c] * inv_q
                    r_k[c] = r_k[c] * inv_k
            else:
                for c in cutlass.range_constexpr(vec_size):
                    r_q[c] = r_q[c] * scale

            # ===== gate stage 2: per-channel decay r_g =====
            if cutlass.const_expr(use_lower_bound):
                for c in cutlass.range_constexpr(vec_size):
                    sigmoid_ax = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.exp(-r_exp_A * r_gx[c], fastmath=fast_math)
                    )
                    r_g[c] = cute.exp(lower_bound * sigmoid_ax, fastmath=fast_math)
            else:
                for c in cutlass.range_constexpr(vec_size):
                    beta_x = softplus_beta * r_gx[c]
                    sp_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                        cutlass.Float32(1.0) + cute.exp(softplus_beta * r_gx[c], fastmath=fast_math), fastmath=fast_math
                    )
                    use_sp = cutlass.Float32(1.0) if beta_x <= softplus_threshold else cutlass.Float32(0.0)
                    r_g[c] = use_sp * sp_val + (cutlass.Float32(1.0) - use_sp) * r_gx[c]
                for c in cutlass.range_constexpr(vec_size):
                    r_g[c] = cute.exp(-r_exp_A * r_g[c], fastmath=fast_math)

            r_beta = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, i_t, i_hv]), fastmath=fast_math)
            )

            # ===== recurrence =====
            for vv in cutlass.range_constexpr(BV):
                sv = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    r_h[vv * vec_size + c] = r_h[vv * vec_size + c] * r_g[c]
                    sv += r_h[vv * vec_size + c] * r_k[c]
                r_red[vv] = sv
            for off in [16, 8, 4, 2, 1]:
                for vv in cutlass.range_constexpr(BV):
                    r_red[vv] = r_red[vv] + cute.arch.shuffle_sync_bfly(r_red[vv], offset=off, mask=-1, mask_and_clamp=31)
            for vv in cutlass.range_constexpr(BV):
                v_new = (r_v[vv] - r_red[vv]) * r_beta
                ovv = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    r_h[vv * vec_size + c] = r_h[vv * vec_size + c] + r_k[c] * v_new
                    ovv += r_h[vv * vec_size + c] * r_q[c]
                r_red[vv] = ovv
            for off in [16, 8, 4, 2, 1]:
                for vv in cutlass.range_constexpr(BV):
                    r_red[vv] = r_red[vv] + cute.arch.shuffle_sync_bfly(r_red[vv], offset=off, mask=-1, mask_and_clamp=31)
            for vv in cutlass.range_constexpr(BV):
                o[(i_n, i_t, i_hv, i_v * BV + vv)] = cutlass.BFloat16(r_red[vv])

            if cutlass.const_expr(cache_intermediate_states):
                flat_idx = i_n * T * HV + i_t * HV + i_hv
                for vv in cutlass.range_constexpr(BV):
                    for c in cutlass.range_constexpr(vec_size):
                        r_h4[c] = r_h[vv * vec_size + c]
                    inter_tile = cute.local_tile(intermediate_states, (1, 1, vec_size), (flat_idx, i_v * BV + vv, lane))
                    cute.autovec_copy(r_h4, inter_tile)

        # ===== epilogue: conv_state rolling writeback (all-accept temp) =====
        if is_qk_owner:
            for c in cutlass.range_constexpr(vec_size):
                qch = q_base + vec_size * lane + c
                kch = k_base + vec_size * lane + c
                conv_state[cs_idx, qch, 0] = r_qhist[0 * vec_size + c]
                conv_state[cs_idx, qch, 1] = r_qhist[1 * vec_size + c]
                conv_state[cs_idx, qch, 2] = r_qhist[2 * vec_size + c]
                conv_state[cs_idx, kch, 0] = r_khist[0 * vec_size + c]
                conv_state[cs_idx, kch, 1] = r_khist[1 * vec_size + c]
                conv_state[cs_idx, kch, 2] = r_khist[2 * vec_size + c]
        if lane < BV:
            vch = v_base + lane
            conv_state[cs_idx, vch, 0] = r_vhist[0]
            conv_state[cs_idx, vch, 1] = r_vhist[1]
            conv_state[cs_idx, vch, 2] = r_vhist[2]

        # ===== epilogue: SSM state writeback (skipped for verify) =====
        if cutlass.const_expr(not disable_state_update):
            for vv in cutlass.range_constexpr(BV):
                v_global = i_v * BV + vv
                for c in cutlass.range_constexpr(vec_size):
                    r_h4[c] = r_h[vv * vec_size + c]
                h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_global, lane))
                cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_conv_verify_vk_kernel(
    mixed_qkv: cute.Tensor,
    conv_weight: cute.Tensor,
    conv_bias: cute.Tensor,
    conv_state: cute.Tensor,
    conv_state_indices: cute.Tensor,
    inter_conv_window: cute.Tensor,
    inter_state_indices: cute.Tensor,
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    intermediate_states: cute.Tensor,
    h0_indices: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    save_conv_window: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles

    kda_conv_verify_vk_kernel(
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        conv_state_indices,
        inter_conv_window,
        inter_state_indices,
        h0_source,
        A_log,
        a,
        dt_bias,
        b,
        o,
        intermediate_states,
        h0_indices,
        vec_size,
        num_v_tiles,
        BV,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        T,
        H,
        K,
        V,
        W,
        use_qk_l2norm,
        disable_state_update,
        cache_intermediate_states,
        save_conv_window,
        has_bias,
        fast_math,
        use_lower_bound,
        lower_bound,
    ).launch(grid=(grid_size, 1, 1), block=[32, 1, 1], smem=0, stream=stream)


_compiled_conv_verify_kernels: dict = {}


def _get_compiled(N, T, H, HV, K, V, D, pool_size, lines, BV, scale, use_qk_l2norm,
                  disable_state_update, cache_intermediate_states, save_conv_window,
                  has_bias, softplus_beta, softplus_threshold, use_lower_bound,
                  lower_bound, opt_level=3, fast_math=True):
    key = (T, H, HV, K, V, D, BV, scale, use_qk_l2norm, disable_state_update,
           cache_intermediate_states, save_conv_window, has_bias, softplus_beta,
           softplus_threshold, use_lower_bound, lower_bound, opt_level, fast_math)
    if key in _compiled_conv_verify_kernels:
        return _compiled_conv_verify_kernels[key]

    dev = "cuda"
    mixed_qkv = torch.zeros(N * T, D, dtype=torch.bfloat16, device=dev)
    conv_weight = torch.zeros(D, WCONV, dtype=torch.float32, device=dev)
    conv_bias = torch.zeros(D, dtype=torch.float32, device=dev)
    conv_state = torch.zeros(lines, D, WCONV - 1, dtype=torch.float32, device=dev)  # contiguous [lines,D,W-1]
    conv_state_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    inter_conv_window = torch.zeros(lines, T, D, WCONV - 1, dtype=torch.float32, device=dev)
    inter_state_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device=dev)
    A_log = torch.zeros(HV, dtype=torch.float32, device=dev)
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device=dev)
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device=dev)
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device=dev)
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device=dev)
    h0_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    if cache_intermediate_states:
        inter_states = torch.zeros(N * T * HV, V, K, dtype=torch.float32, device=dev)
    else:
        inter_states = torch.empty(1, 1, 1, dtype=torch.float32, device=dev)

    def dl(t, dyn0=False):
        x = from_dlpack(t, assumed_align=16)
        if dyn0:
            return x.mark_compact_shape_dynamic(mode=0, stride_order=t.dim_order())
        return x

    mixed_t = dl(mixed_qkv, True)
    cw_t = dl(conv_weight)
    cb_t = dl(conv_bias)
    cs_t = dl(conv_state, True)
    csi_t = from_dlpack(conv_state_indices, assumed_align=16).mark_layout_dynamic()
    iw_t = dl(inter_conv_window, True)
    isi_t = from_dlpack(inter_state_indices, assumed_align=16).mark_layout_dynamic()
    h0_t = dl(h0_source, True)
    A_t = dl(A_log)
    a_t = dl(a, True)
    dtb_t = dl(dt_bias)
    b_t = dl(b, True)
    o_t = dl(o, True)
    is_t = dl(inter_states, True) if cache_intermediate_states else dl(inter_states)
    h0i_t = from_dlpack(h0_indices, assumed_align=16).mark_layout_dynamic()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled = cute.compile(
        run_kda_conv_verify_vk_kernel,
        mixed_t, cw_t, cb_t, cs_t, csi_t, iw_t, isi_t, h0_t, A_t, a_t, dtb_t, b_t,
        o_t, is_t, h0i_t,
        vec_size=VEC_SIZE, BV=BV, softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold, scale=scale, HV=HV, T=T, H=H, K=K, V=V,
        W=WCONV, use_qk_l2norm=use_qk_l2norm, disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states, save_conv_window=save_conv_window,
        has_bias=has_bias, fast_math=fast_math, use_lower_bound=use_lower_bound,
        lower_bound=lower_bound, stream=stream,
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_conv_verify_kernels[key] = compiled
    logger.info(f"cuLA fused conv+verify vk compiled: N={N} T={T} HV={HV} K={K} V={V} BV={BV}")
    return compiled


# =========================================================================== #
# Large-batch warp-spec variant: 4 warps/CTA, warp 0 produces shared q/k conv +
# gate to SMEM, each warp does BVW v-cols of recurrence+v-conv+snapshot.
# r_h = BVW*vec_size (low regs -> high occupancy) + shared q/k (no v-tile redun).
# =========================================================================== #
NWARP = 4


@cute.kernel
def kda_conv_verify_ws_kernel(
    mixed_qkv: cute.Tensor,
    conv_weight: cute.Tensor,
    conv_bias: cute.Tensor,
    conv_state: cute.Tensor,
    conv_state_indices: cute.Tensor,
    inter_conv_window: cute.Tensor,
    inter_state_indices: cute.Tensor,
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    intermediate_states: cute.Tensor,
    h0_indices: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    BVW: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    save_conv_window: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.warp_idx()
    warp = cute.arch.make_warp_uniform(warp)

    bidx, _, _ = cute.arch.block_idx()
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cs_idx = conv_state_indices[i_n]
    cache_idx = h0_indices[i_n]
    iw_idx = inter_state_indices[i_n]
    # owner = LAST CTA (largest bidx) among the CTAs sharing this q/k head, so
    # its epilogue conv_state write happens after every non-owner has read the
    # history at its preamble (bidx ~ dispatch order) -> race-free shared write.
    # Same rolled value as any owner would write, so behavior matches the reference.
    is_qk_owner = (i_v == num_v_tiles - 1) and (i_hv % (HV // H) == (HV // H) - 1)
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    q_base = i_h * K
    k_base = H * K + i_h * K
    # this warp's v-cols: global col = i_v*tile_v + warp*BVW + [0..BVW)
    v_col0 = i_v * tile_v + warp * BVW

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,), stride=(1,)), 16)

    if cache_idx >= 0:
        # =========== Producer: 4 warps each compute a subset of tokens ===========
        # Each token's conv is independent given raw inputs, so warp `warp`
        # handles tokens i_t with i_t % NWARP == warp (T=4 -> one token/warp),
        # 4x-parallelizing the previously warp-0-only producer.
        r_qw = cute.make_rmem_tensor(cute.make_layout((W * vec_size,), stride=(1,)), cutlass.Float32)
        r_kw = cute.make_rmem_tensor(cute.make_layout((W * vec_size,), stride=(1,)), cutlass.Float32)
        r_qb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_kb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_pq = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_pk = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        bos = i_n * T
        for c in cutlass.range_constexpr(vec_size):
            qch = q_base + vec_size * lane + c
            kch = k_base + vec_size * lane + c
            r_dtb[c] = cutlass.Float32(dt_bias[i_hv, vec_size * lane + c])
            for w in cutlass.range_constexpr(W):
                r_qw[w * vec_size + c] = cutlass.Float32(conv_weight[qch, w])
                r_kw[w * vec_size + c] = cutlass.Float32(conv_weight[kch, w])
            if cutlass.const_expr(has_bias):
                r_qb[c] = cutlass.Float32(conv_bias[qch])
                r_kb[c] = cutlass.Float32(conv_bias[kch])
            else:
                r_qb[c] = cutlass.Float32(0.0)
                r_kb[c] = cutlass.Float32(0.0)

        for i_t in cutlass.range_constexpr(T):
            if i_t % NWARP == warp:
                # q conv: tap m at abs position p=i_t-(W-1)+m; p<0 -> conv_state history col p+W-1
                for c in cutlass.range_constexpr(vec_size):
                    qch = q_base + vec_size * lane + c
                    acc = r_qb[c]
                    for m in cutlass.range_constexpr(W):
                        p = i_t - (W - 1) + m
                        if cutlass.const_expr(p >= 0):
                            xq = cutlass.Float32(mixed_qkv[bos + p, qch])
                        else:
                            xq = cutlass.Float32(conv_state[cs_idx, qch, p + (W - 1)])
                        acc = acc + xq * r_qw[m * vec_size + c]
                    silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                    r_pq[c] = cutlass.Float32(cutlass.BFloat16(silu))
                # k conv
                for c in cutlass.range_constexpr(vec_size):
                    kch = k_base + vec_size * lane + c
                    acc = r_kb[c]
                    for m in cutlass.range_constexpr(W):
                        p = i_t - (W - 1) + m
                        if cutlass.const_expr(p >= 0):
                            xk = cutlass.Float32(mixed_qkv[bos + p, kch])
                        else:
                            xk = cutlass.Float32(conv_state[cs_idx, kch, p + (W - 1)])
                        acc = acc + xk * r_kw[m * vec_size + c]
                    silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                    r_pk[c] = cutlass.Float32(cutlass.BFloat16(silu))
                # window snapshot (owner): window ending at i_t = positions i_t-2,i_t-1,i_t
                if cutlass.const_expr(save_conv_window):
                    if is_qk_owner:
                        for c in cutlass.range_constexpr(vec_size):
                            qch = q_base + vec_size * lane + c
                            kch = k_base + vec_size * lane + c
                            for wv in cutlass.range_constexpr(W - 1):
                                pw = i_t - (W - 2) + wv
                                if cutlass.const_expr(pw >= 0):
                                    inter_conv_window[iw_idx, i_t, qch, wv] = cutlass.Float32(mixed_qkv[bos + pw, qch])
                                    inter_conv_window[iw_idx, i_t, kch, wv] = cutlass.Float32(mixed_qkv[bos + pw, kch])
                                else:
                                    inter_conv_window[iw_idx, i_t, qch, wv] = cutlass.Float32(conv_state[cs_idx, qch, pw + (W - 1)])
                                    inter_conv_window[iw_idx, i_t, kch, wv] = cutlass.Float32(conv_state[cs_idx, kch, pw + (W - 1)])
                # l2norm + scale (butterfly within this warp = all K for token i_t)
                if cutlass.const_expr(use_qk_l2norm):
                    sq = cutlass.Float32(0.0)
                    sk = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(vec_size):
                        sq += r_pq[c] * r_pq[c]
                        sk += r_pk[c] * r_pk[c]
                    for off in [16, 8, 4, 2, 1]:
                        sq += cute.arch.shuffle_sync_bfly(sq, offset=off, mask=-1, mask_and_clamp=31)
                        sk += cute.arch.shuffle_sync_bfly(sk, offset=off, mask=-1, mask_and_clamp=31)
                    inv_q = cute.rsqrt(sq + 1e-6, fastmath=fast_math) * scale
                    inv_k = cute.rsqrt(sk + 1e-6, fastmath=fast_math)
                    for c in cutlass.range_constexpr(vec_size):
                        r_pq[c] = r_pq[c] * inv_q
                        r_pk[c] = r_pk[c] * inv_k
                else:
                    for c in cutlass.range_constexpr(vec_size):
                        r_pq[c] = r_pq[c] * scale
                # gate + beta
                for c in cutlass.range_constexpr(vec_size):
                    gx = cutlass.Float32(a[i_n, i_t, i_hv, vec_size * lane + c]) + r_dtb[c]
                    if cutlass.const_expr(use_lower_bound):
                        sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-r_exp_A * gx, fastmath=fast_math))
                        g = cute.exp(lower_bound * sig, fastmath=fast_math)
                    else:
                        beta_x = softplus_beta * gx
                        sp = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                            cutlass.Float32(1.0) + cute.exp(beta_x, fastmath=fast_math), fastmath=fast_math
                        )
                        use_sp = cutlass.Float32(1.0) if beta_x <= softplus_threshold else cutlass.Float32(0.0)
                        spx = use_sp * sp + (cutlass.Float32(1.0) - use_sp) * gx
                        g = cute.exp(-r_exp_A * spx, fastmath=fast_math)
                    sQ[(i_t, vec_size * lane + c)] = r_pq[c]
                    sK[(i_t, vec_size * lane + c)] = r_pk[c]
                    sG[(i_t, vec_size * lane + c)] = g
                if lane == 0:
                    r_bb = cutlass.Float32(b[i_n, i_t, i_hv])
                    sBeta[i_t] = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-r_bb, fastmath=fast_math))

        cute.arch.barrier()

        # =============== Consumer: each warp does BVW v-cols =====================
        r_h = cute.make_rmem_tensor(cute.make_layout((BVW * vec_size,), stride=(1,)), cutlass.Float32)
        r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_v = cute.make_rmem_tensor(cute.make_layout((BVW,), stride=(1,)), cutlass.Float32)
        r_red = cute.make_rmem_tensor(cute.make_layout((BVW,), stride=(1,)), cutlass.Float32)
        r_vhist = cute.make_rmem_tensor(cute.make_layout((W - 1,), stride=(1,)), cutlass.Float32)
        r_vw = cute.make_rmem_tensor(cute.make_layout((W,), stride=(1,)), cutlass.Float32)
        r_vb = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32)

        flat_state_idx = cache_idx * HV + i_hv
        for vv in cutlass.range_constexpr(BVW):
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_col0 + vv, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]

        # v-conv preload (Option B within warp: lane < BVW owns v-col `lane`)
        if lane < BVW:
            vch = 2 * H * K + i_hv * V + v_col0 + lane
            r_vhist[0] = cutlass.Float32(conv_state[cs_idx, vch, 0])
            r_vhist[1] = cutlass.Float32(conv_state[cs_idx, vch, 1])
            r_vhist[2] = cutlass.Float32(conv_state[cs_idx, vch, 2])
            r_vw[0] = cutlass.Float32(conv_weight[vch, 0])
            r_vw[1] = cutlass.Float32(conv_weight[vch, 1])
            r_vw[2] = cutlass.Float32(conv_weight[vch, 2])
            r_vw[3] = cutlass.Float32(conv_weight[vch, 3])
            if cutlass.const_expr(has_bias):
                r_vb[0] = cutlass.Float32(conv_bias[vch])
            else:
                r_vb[0] = cutlass.Float32(0.0)

        for i_t in cutlass.range_constexpr(T):
            row = i_n * T + i_t
            # v conv (lane<BVW computes its col, shuffle-broadcast within warp)
            my_v = cutlass.Float32(0.0)
            if lane < BVW:
                vch = 2 * H * K + i_hv * V + v_col0 + lane
                xv = cutlass.Float32(mixed_qkv[row, vch])
                acc = r_vb[0] + r_vhist[0] * r_vw[0] + r_vhist[1] * r_vw[1] + r_vhist[2] * r_vw[2] + xv * r_vw[3]
                r_vhist[0] = r_vhist[1]
                r_vhist[1] = r_vhist[2]
                r_vhist[2] = xv
                if cutlass.const_expr(save_conv_window):
                    inter_conv_window[iw_idx, i_t, vch, 0] = r_vhist[0]
                    inter_conv_window[iw_idx, i_t, vch, 1] = r_vhist[1]
                    inter_conv_window[iw_idx, i_t, vch, 2] = r_vhist[2]
                silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                my_v = cutlass.Float32(cutlass.BFloat16(silu))
            for vv in cutlass.range_constexpr(BVW):
                r_v[vv] = cute.arch.shuffle_sync(my_v, vv, mask=-1, mask_and_clamp=31)

            # read shared q/k/g/beta from SMEM
            for c in cutlass.range_constexpr(vec_size):
                r_q[c] = sQ[(i_t, vec_size * lane + c)]
                r_k[c] = sK[(i_t, vec_size * lane + c)]
                r_g[c] = sG[(i_t, vec_size * lane + c)]
            r_beta = sBeta[i_t]

            # recurrence
            for vv in cutlass.range_constexpr(BVW):
                sv = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    r_h[vv * vec_size + c] = r_h[vv * vec_size + c] * r_g[c]
                    sv += r_h[vv * vec_size + c] * r_k[c]
                r_red[vv] = sv
            for off in [16, 8, 4, 2, 1]:
                for vv in cutlass.range_constexpr(BVW):
                    r_red[vv] = r_red[vv] + cute.arch.shuffle_sync_bfly(r_red[vv], offset=off, mask=-1, mask_and_clamp=31)
            for vv in cutlass.range_constexpr(BVW):
                v_new = (r_v[vv] - r_red[vv]) * r_beta
                ovv = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(vec_size):
                    r_h[vv * vec_size + c] = r_h[vv * vec_size + c] + r_k[c] * v_new
                    ovv += r_h[vv * vec_size + c] * r_q[c]
                r_red[vv] = ovv
            for off in [16, 8, 4, 2, 1]:
                for vv in cutlass.range_constexpr(BVW):
                    r_red[vv] = r_red[vv] + cute.arch.shuffle_sync_bfly(r_red[vv], offset=off, mask=-1, mask_and_clamp=31)
            for vv in cutlass.range_constexpr(BVW):
                o[(i_n, i_t, i_hv, v_col0 + vv)] = cutlass.BFloat16(r_red[vv])
            if cutlass.const_expr(cache_intermediate_states):
                flat_idx = i_n * T * HV + i_t * HV + i_hv
                for vv in cutlass.range_constexpr(BVW):
                    for c in cutlass.range_constexpr(vec_size):
                        r_h4[c] = r_h[vv * vec_size + c]
                    inter_tile = cute.local_tile(intermediate_states, (1, 1, vec_size), (flat_idx, v_col0 + vv, lane))
                    cute.autovec_copy(r_h4, inter_tile)

        # consumer epilogue: conv_state writeback at kernel end (race-free).
        # q/k written by warp 0 owner from mixed_qkv directly (rolled state after
        # T>=W-1 tokens = the last W-1 raw inputs); v by each warp's lanes.
        if is_qk_owner and warp == 0:
            # rolled window = last W-1 abs positions (p = T-(W-1)+w). p>=0 from
            # mixed_qkv; p<0 (only when T<W-1) from conv_state history col p+(W-1).
            # Read cols before overwriting (w ascending, hazard-free at T<W-1).
            for c in cutlass.range_constexpr(vec_size):
                qch = q_base + vec_size * lane + c
                kch = k_base + vec_size * lane + c
                for w in cutlass.range_constexpr(W - 1):
                    p = T - (W - 1) + w
                    if cutlass.const_expr(p >= 0):
                        conv_state[cs_idx, qch, w] = cutlass.Float32(mixed_qkv[i_n * T + p, qch])
                        conv_state[cs_idx, kch, w] = cutlass.Float32(mixed_qkv[i_n * T + p, kch])
                    else:
                        conv_state[cs_idx, qch, w] = cutlass.Float32(conv_state[cs_idx, qch, p + (W - 1)])
                        conv_state[cs_idx, kch, w] = cutlass.Float32(conv_state[cs_idx, kch, p + (W - 1)])
        if lane < BVW:
            vch = 2 * H * K + i_hv * V + v_col0 + lane
            conv_state[cs_idx, vch, 0] = r_vhist[0]
            conv_state[cs_idx, vch, 1] = r_vhist[1]
            conv_state[cs_idx, vch, 2] = r_vhist[2]
        if cutlass.const_expr(not disable_state_update):
            for vv in cutlass.range_constexpr(BVW):
                for c in cutlass.range_constexpr(vec_size):
                    r_h4[c] = r_h[vv * vec_size + c]
                h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_col0 + vv, lane))
                cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_conv_verify_ws_kernel(
    mixed_qkv, conv_weight, conv_bias, conv_state, conv_state_indices,
    inter_conv_window, inter_state_indices, h0_source, A_log, a, dt_bias, b, o,
    intermediate_states, h0_indices,
    vec_size: cutlass.Constexpr[int],
    BVW: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    save_conv_window: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, tile_v)
    grid_size = n_indices * HV * num_v_tiles
    smem_bytes = 3 * (4 * T * (K + 8)) + 4 * T + 128  # sQ+sK+sG [T,K+8] fp32 + sBeta + slack
    kda_conv_verify_ws_kernel(
        mixed_qkv, conv_weight, conv_bias, conv_state, conv_state_indices,
        inter_conv_window, inter_state_indices, h0_source, A_log, a, dt_bias, b, o,
        intermediate_states, h0_indices,
        vec_size, num_v_tiles, BVW, tile_v, softplus_beta, softplus_threshold, scale,
        HV, T, H, K, V, W, use_qk_l2norm, disable_state_update, cache_intermediate_states,
        save_conv_window, has_bias, fast_math, use_lower_bound, lower_bound,
    ).launch(grid=(grid_size, 1, 1), block=[NWARP * 32, 1, 1], smem=smem_bytes, stream=stream)


_compiled_conv_verify_ws_kernels: dict = {}


def _get_compiled_ws(N, T, H, HV, K, V, D, pool_size, lines, BVW, tile_v, scale,
                     use_qk_l2norm, disable_state_update, cache_intermediate_states,
                     save_conv_window, has_bias, softplus_beta, softplus_threshold,
                     use_lower_bound, lower_bound, opt_level=3, fast_math=True):
    key = (T, H, HV, K, V, D, BVW, tile_v, scale, use_qk_l2norm, disable_state_update,
           cache_intermediate_states, save_conv_window, has_bias, softplus_beta,
           softplus_threshold, use_lower_bound, lower_bound, opt_level, fast_math)
    if key in _compiled_conv_verify_ws_kernels:
        return _compiled_conv_verify_ws_kernels[key]
    dev = "cuda"
    mixed_qkv = torch.zeros(N * T, D, dtype=torch.bfloat16, device=dev)
    conv_weight = torch.zeros(D, WCONV, dtype=torch.float32, device=dev)
    conv_bias = torch.zeros(D, dtype=torch.float32, device=dev)
    conv_state = torch.zeros(lines, D, WCONV - 1, dtype=torch.float32, device=dev)
    conv_state_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    inter_conv_window = torch.zeros(lines, T, D, WCONV - 1, dtype=torch.float32, device=dev)
    inter_state_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device=dev)
    A_log = torch.zeros(HV, dtype=torch.float32, device=dev)
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device=dev)
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device=dev)
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device=dev)
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device=dev)
    h0_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    if cache_intermediate_states:
        inter_states = torch.zeros(N * T * HV, V, K, dtype=torch.float32, device=dev)
    else:
        inter_states = torch.empty(1, 1, 1, dtype=torch.float32, device=dev)

    def dl(t, dyn0=False):
        x = from_dlpack(t, assumed_align=16)
        if dyn0:
            return x.mark_compact_shape_dynamic(mode=0, stride_order=t.dim_order())
        return x

    def dli(t):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        run_kda_conv_verify_ws_kernel,
        dl(mixed_qkv, True), dl(conv_weight), dl(conv_bias), dl(conv_state, True),
        dli(conv_state_indices), dl(inter_conv_window, True), dli(inter_state_indices),
        dl(h0_source, True), dl(A_log), dl(a, True), dl(dt_bias), dl(b, True), dl(o, True),
        dl(inter_states, True) if cache_intermediate_states else dl(inter_states), dli(h0_indices),
        vec_size=VEC_SIZE, BVW=BVW, tile_v=tile_v, softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold, scale=scale, HV=HV, T=T, H=H, K=K, V=V,
        W=WCONV, use_qk_l2norm=use_qk_l2norm, disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states, save_conv_window=save_conv_window,
        has_bias=has_bias, fast_math=fast_math, use_lower_bound=use_lower_bound,
        lower_bound=lower_bound, stream=stream,
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_conv_verify_ws_kernels[key] = compiled
    logger.info(f"cuLA fused conv+verify WS compiled: N={N} T={T} HV={HV} BVW={BVW} tile_v={tile_v}")
    return compiled


def kda_conv_decode_mtp_verify(
    mixed_qkv: torch.Tensor,      # [N*T, D] bf16
    conv_weight: torch.Tensor,    # [D, W] fp32
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,     # [lines, D, W-1] fp32 (dim contiguous)
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,  # [lines, T, D, W-1] fp32
    intermediate_state_indices: torch.Tensor,
    a: torch.Tensor,              # [N, T, HV, K]
    b: torch.Tensor,              # [N, T, HV]
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,     # [slots, HV, V, K] fp32
    cache_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor | None,
    scale: float,
    T: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    bv: int = -1,
    variant: str = "auto",   # "vk" (1-warp small-batch) / "ws" (4-warp large-batch) / "auto"
    bvw: int = -1,           # ws: v-cols per warp (tile_v = 4*bvw); -1 = auto (16 on L20X)
    out: torch.Tensor | None = None,
):
    H, HV, K, V = num_q_heads, num_v_heads, head_k_dim, head_v_dim
    seq_len, D = mixed_qkv.shape
    N = seq_len // T
    assert K == TILE_K, f"requires K={TILE_K}, got {K}"
    assert D == 2 * H * K + HV * V, f"packed dim mismatch: {D} vs {2*H*K+HV*V}"
    work_units = N * HV

    if variant == "auto":
        # small batch -> vk (1 warp, launch-bound floor lowest); larger -> ws
        # (4-warp warp-spec: low state regs + shared q/k -> best occupancy).
        # Threshold 64 from a graph-timed vk-vs-ws crossover sweep: vk wins only
        # at N*HV<=32; at N*HV>=64 ws is faster-or-tied, and at N*HV>=128 ws beats
        # triton (~1.2x) where vk lost (~0.85x). (Earlier 256 left [64,256) on the
        # slower vk path.)
        variant = "ws" if (work_units >= 64 and V % (NWARP * bvw) == 0) else "vk"

    lines = conv_state.shape[0]
    slots = ssm_states.shape[0]
    h0_source = ssm_states.reshape(slots * HV, V, K)  # [slots*HV, V, K]
    o = _prepare_output_tensor(mixed_qkv, out, (N, T, HV, V))

    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        inter_states_flat = intermediate_states_buffer.reshape(N * T * HV, V, K)
    else:
        inter_states_flat = torch.empty(1, 1, 1, dtype=torch.float32, device=mixed_qkv.device)

    has_bias = conv_bias is not None
    conv_bias_t = conv_bias if has_bias else torch.zeros(D, dtype=torch.float32, device=mixed_qkv.device)

    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)
    cache_indices = cache_indices.to(torch.int32)
    intermediate_state_indices = intermediate_state_indices.to(torch.int32)
    stream = _get_cached_stream(mixed_qkv.device)

    lb_val = 0.0 if lower_bound is None else float(lower_bound)
    use_lb = lower_bound is not None

    if variant == "ws":
        if bvw <= 0:  # auto: bvw=16 (tile_v=64) best on L20X — fewer v-tiles cut
            # redundant q/k producer work; r_h=64 keeps occupancy (bvw=32 r_h=128 worse).
            bvw = 16 if V % (NWARP * 16) == 0 else (8 if V % (NWARP * 8) == 0 else V // NWARP)
        tile_v = NWARP * bvw
        assert V % tile_v == 0, f"ws requires V%(4*bvw)==0: V={V} bvw={bvw}"
        compiled = _get_compiled_ws(
            N, T, H, HV, K, V, D, slots, lines, bvw, tile_v, scale, use_qk_l2norm_in_kernel,
            True, cache_intermediate_states, True, has_bias, softplus_beta,
            softplus_threshold, use_lb, lb_val,
        )
    else:  # vk
        if bv <= 0:
            bv = 8 if (work_units < 192 and V % 8 == 0) else (32 if V % 32 == 0 else 8)
        assert V % bv == 0, f"V%bv!=0: V={V} bv={bv}"
        compiled = _get_compiled(
            N, T, H, HV, K, V, D, slots, lines, bv, scale, use_qk_l2norm_in_kernel,
            True, cache_intermediate_states, True, has_bias, softplus_beta,
            softplus_threshold, use_lb, lb_val,
        )

    compiled(
        mixed_qkv, conv_weight, conv_bias_t, conv_state, cache_indices,
        intermediate_conv_window, intermediate_state_indices, h0_source, A_log, a,
        dt_bias, b, o, inter_states_flat, cache_indices, stream,
    )
    return o
