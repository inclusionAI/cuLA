"""Fused causal-conv1d + KDA MTP KVBuffer verify kernels.

The operators consume packed pre-convolution ``mixed_qkv`` and preserve the
compact ``(d, k, g)`` scratch format consumed by :mod:`mtp_kvbuffer` flushes.
Conv state and intermediate windows use the same layout and mutation semantics
as :mod:`mtp_conv`.
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

from cula.ops.kda.decode.cute import (
    TILE_K,
    _get_cached_stream,
    _normalize_A_log,
    _normalize_dt_bias,
    _normalize_state_indices,
    _normalize_state_source,
    _prepare_output_tensor,
)
from cula.ops.kda.decode.mtp import VEC_SIZE, _normalize_mtp_a
from cula.ops.kda.decode.mtp_conv import (
    WCONV,
    _announce_qk_read_complete,
    _get_qk_arrival_counters,
)
from cula.ops.kda.decode.mtp_kvbuffer import (
    BT,
    _mma_m16n8k8_3xtf32,
)

logger = logging.getLogger(__name__)
_zero_conv_bias_cache: dict[tuple, torch.Tensor] = {}


def _get_zero_conv_bias(D: int, device: torch.device) -> torch.Tensor:
    key = (D, str(device))
    bias = _zero_conv_bias_cache.get(key)
    if bias is None:
        bias = torch.zeros(D, dtype=torch.float32, device=device)
        _zero_conv_bias_cache[key] = bias
    return bias


def _dl(tensor: torch.Tensor, dynamic_mode0: bool = False):
    result = from_dlpack(tensor, assumed_align=16)
    if dynamic_mode0:
        return result.mark_compact_shape_dynamic(mode=0, stride_order=tensor.dim_order())
    return result


def _dli(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic()


def _validate_ubuffers(d_buffer, k_buffer, g_buffer, shape_d, shape_kg):
    supplied = (d_buffer is not None, k_buffer is not None, g_buffer is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("d_buffer, k_buffer and g_buffer must be supplied together")
    if not all(supplied):
        return False
    if tuple(d_buffer.shape) != shape_d:
        raise ValueError(f"d_buffer shape must be {shape_d}, got {tuple(d_buffer.shape)}")
    if tuple(k_buffer.shape) != shape_kg or tuple(g_buffer.shape) != shape_kg:
        raise ValueError(f"k_buffer/g_buffer shape must be {shape_kg}")
    for name, tensor in (("d_buffer", d_buffer), ("k_buffer", k_buffer), ("g_buffer", g_buffer)):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {tensor.dtype}")
    return True


@cute.jit
def _conv_channel_segment(
    mixed_qkv,
    conv_weight,
    conv_bias,
    conv_state,
    inter_conv_window,
    conv_out,
    final_hist_out,
    cs_idx,
    iw_idx,
    row_base,
    channel,
    out_row_base,
    out_col,
    save_window,
    save_final_hist,
    segment_idx,
    num_segments: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
    has_bias: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
):
    """Compute one balanced contiguous token segment for one packed channel."""
    base_tokens: cutlass.Constexpr[int] = T // num_segments
    remainder: cutlass.Constexpr[int] = T % num_segments
    short_segments: cutlass.Constexpr[int] = num_segments - remainder
    max_segment_tokens: cutlass.Constexpr[int] = (T + num_segments - 1) // num_segments
    segment_tokens = base_tokens
    token_start = segment_idx * base_tokens
    if cutlass.const_expr(remainder != 0):
        if segment_idx >= short_segments:
            segment_tokens = segment_tokens + 1
            token_start = token_start + segment_idx - short_segments
    r_hist = cute.make_rmem_tensor(cute.make_layout((W - 1,), stride=(1,)), cutlass.Float32)
    r_weight = cute.make_rmem_tensor(cute.make_layout((W,), stride=(1,)), cutlass.Float32)
    for w in cutlass.range_constexpr(W - 1):
        full_pos = token_start + w
        if full_pos < W - 1:
            r_hist[w] = cutlass.Float32(conv_state[cs_idx, channel, full_pos])
        else:
            r_hist[w] = cutlass.Float32(mixed_qkv[row_base + full_pos - (W - 1), channel])
    for w in cutlass.range_constexpr(W):
        r_weight[w] = cutlass.Float32(conv_weight[channel, w])
    bias = cutlass.Float32(0.0)
    if cutlass.const_expr(has_bias):
        bias = cutlass.Float32(conv_bias[channel])

    for local_t in cutlass.range_constexpr(max_segment_tokens):
        if local_t < segment_tokens:
            i_t = token_start + local_t
            x_new = cutlass.Float32(mixed_qkv[row_base + i_t, channel])
            acc = bias
            for w in cutlass.range_constexpr(W - 1):
                acc = acc + r_weight[w] * r_hist[w]
            acc = acc + r_weight[W - 1] * x_new
            silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
            conv_out[out_row_base + i_t, out_col] = cutlass.Float32(cutlass.BFloat16(silu))

            for w in cutlass.range_constexpr(W - 2):
                r_hist[w] = r_hist[w + 1]
            r_hist[W - 2] = x_new
            if save_window:
                for w in cutlass.range_constexpr(W - 1):
                    inter_conv_window[iw_idx, i_t, channel, w] = r_hist[w]

    if save_final_hist:
        for w in cutlass.range_constexpr(W - 1):
            final_hist_out[w, out_col] = r_hist[w]


@cute.jit
def _roll_conv_channel(
    mixed_qkv,
    conv_state,
    cs_idx,
    row_base,
    channel,
    T: cutlass.Constexpr[int],
    W: cutlass.Constexpr[int],
):
    """Roll one channel to the last ``W-1`` raw inputs."""
    r_out = cute.make_rmem_tensor(cute.make_layout((W - 1,), stride=(1,)), cutlass.Float32)
    for w in cutlass.range_constexpr(W - 1):
        full_pos = T + w
        if cutlass.const_expr(full_pos < W - 1):
            r_out[w] = cutlass.Float32(conv_state[cs_idx, channel, full_pos])
        else:
            r_out[w] = cutlass.Float32(mixed_qkv[row_base + full_pos - (W - 1), channel])
    for w in cutlass.range_constexpr(W - 1):
        conv_state[cs_idx, channel, w] = r_out[w]


@cute.kernel
def kda_conv_mtp_tensor_core_kvbuffer_kernel(
    mixed_qkv: cute.Tensor,
    conv_weight: cute.Tensor,
    conv_bias: cute.Tensor,
    conv_state: cute.Tensor,
    conv_state_indices: cute.Tensor,
    qk_arrival_counters: cute.Tensor,
    inter_conv_window: cute.Tensor,
    inter_state_indices: cute.Tensor,
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
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
    W: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # Pack up to two independent four-warp V groups into one CTA.
    cta_v_groups: cutlass.Constexpr[int] = 1
    if cutlass.const_expr(num_v_tiles >= 2):
        cta_v_groups = 2
    physical_v_tiles: cutlass.Constexpr[int] = num_v_tiles // cta_v_groups
    warp_group = cute.arch.make_warp_uniform(warp_idx // 4)
    local_warp = cute.arch.make_warp_uniform(warp_idx % 4)
    local_tidx = tidx - warp_group * 128
    gid = lane_id // 4
    tig = lane_id % 4
    num_warps: cutlass.Constexpr[int] = 4

    # Map each CTA to one request, one V head, and one physical V tile.
    bidx, _, _ = cute.arch.block_idx()
    cta_i_v = bidx % physical_v_tiles
    tmp = bidx // physical_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)
    i_v = cta_i_v * cta_v_groups + warp_group
    single_qk_cta: cutlass.Constexpr[bool] = physical_v_tiles == 1 and HV == H

    cs_idx = conv_state_indices[i_n]
    cache_idx = h0_indices[i_n]
    iw_idx = inter_state_indices[i_n]
    row_base = i_n * T
    q_base = i_h * K
    k_base = H * K + i_h * K
    is_qk_owner = (cta_i_v == physical_v_tiles - 1) and (i_hv % (HV // H) == (HV // H) - 1)
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    smem = cutlass.utils.SmemAllocator()
    # Keep MMA-only padding; sG is read row-wise and uses the compact K stride.
    sKQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((2 * BT, K), stride=(K + 4, 1)), 16)
    sKsuf = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, K), stride=(K + 8, 1)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, K), stride=(K, 1)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT,)), 16)
    sBlast = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sPart = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4 * 16, 12), stride=(12, 1)), 16)
    sL = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sInv = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    sLp = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, BT), stride=(BT + 1, 1)), 16)
    group_bv: cutlass.Constexpr[int] = cta_v_groups * BV
    sX = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, group_bv), stride=(group_bv + 1, 1)), 16)
    sU = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BT, group_bv), stride=(group_bv + 1, 1)), 16)
    sS0 = smem.allocate_tensor(cutlass.Float32, cute.make_layout((group_bv, K), stride=(K + 4, 1)), 16)

    r_qf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_kf = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_s = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    ppw_tc: cutlass.Constexpr[int] = (2 * T * T + num_warps - 1) // num_warps
    r_red = cute.make_rmem_tensor(cute.make_layout((ppw_tc,), stride=(1,)), cutlass.Float32)

    if cache_idx >= 0:
        # Produce the K and Q width-4 conv results in separate warp groups.
        k_start = lane_id * vec_size
        flat_state_idx = cache_idx * HV + i_hv
        kc = local_tidx
        num_v_blocks: cutlass.Constexpr[int] = V // BV // num_v_tiles
        if local_tidx < K:
            if warp_group == 0:
                _conv_channel_segment(
                    mixed_qkv,
                    conv_weight,
                    conv_bias,
                    conv_state,
                    inter_conv_window,
                    sKQ,
                    sS0,
                    cs_idx,
                    iw_idx,
                    row_base,
                    k_base + kc,
                    0,
                    kc,
                    is_qk_owner,
                    False,
                    0,
                    1,
                    T,
                    W,
                    has_bias,
                    fast_math,
                )
            if warp_group == cta_v_groups - 1:
                _conv_channel_segment(
                    mixed_qkv,
                    conv_weight,
                    conv_bias,
                    conv_state,
                    inter_conv_window,
                    sKQ,
                    sS0,
                    cs_idx,
                    iw_idx,
                    row_base,
                    q_base + kc,
                    BT,
                    kc,
                    is_qk_owner,
                    False,
                    0,
                    1,
                    T,
                    W,
                    has_bias,
                    fast_math,
                )
        cute.arch.barrier()
        # Normalize q/k and materialize per-token decay gates and beta values.
        tokens_per_warp: cutlass.Constexpr[int] = (T + num_warps - 1) // num_warps
        for tt in cutlass.range_constexpr(tokens_per_warp):
            i_t = tt * num_warps + warp_idx
            if i_t < T and warp_group == 0:
                for c in cutlass.range_constexpr(vec_size):
                    r_qf[c] = sKQ[BT + i_t, k_start + c]
                    r_kf[c] = sKQ[i_t, k_start + c]
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
                for c in cutlass.range_constexpr(vec_size):
                    x = cutlass.Float32(a[i_n, i_t, i_hv, k_start + c]) + cutlass.Float32(dt_bias[i_hv, k_start + c])
                    if cutlass.const_expr(use_lower_bound):
                        sigmoid_ax = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-r_exp_A * x, fastmath=fast_math))
                        sG[i_t, k_start + c] = cute.exp(lower_bound * sigmoid_ax, fastmath=fast_math)
                    else:
                        beta_x = softplus_beta * x
                        sp_x = x
                        if beta_x <= softplus_threshold:
                            sp_x = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                                cutlass.Float32(1.0) + cute.exp(beta_x, fastmath=fast_math),
                                fastmath=fast_math,
                            )
                        sG[i_t, k_start + c] = cute.exp(-r_exp_A * sp_x, fastmath=fast_math)
                    sKQ[i_t, k_start + c] = r_kf[c]
                    sKQ[BT + i_t, k_start + c] = r_qf[c]
                if lane_id == 0:
                    sBeta[i_t] = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, i_t, i_hv]), fastmath=fast_math)
                    )
        # Zero-pad the active T rows to the fixed BT=8 tensor-core shape.
        if tidx < K:
            for rp in cutlass.range_constexpr(BT - T):
                sKQ[T + rp, tidx] = cutlass.Float32(0.0)
                sKQ[BT + T + rp, tidx] = cutlass.Float32(0.0)
                sKsuf[T + rp, tidx] = cutlass.Float32(0.0)
        if tidx >= T:
            if tidx < BT:
                sBeta[tidx] = cutlass.Float32(0.0)
        cute.arch.barrier()

        # Let the last q/k-sharing CTA roll conv state after every reader finishes.
        if cutlass.const_expr(not single_qk_cta):
            if warp_idx == 0:
                counter_idx = i_n * H + i_h
                expected_arrivals = physical_v_tiles * (HV // H)
                is_last_qk_cta = _announce_qk_read_complete(qk_arrival_counters, counter_idx, expected_arrivals, lane_id)
                if is_last_qk_cta:
                    for c in cutlass.range_constexpr(vec_size):
                        _roll_conv_channel(
                            mixed_qkv,
                            conv_state,
                            cs_idx,
                            row_base,
                            q_base + k_start + c,
                            T,
                            W,
                        )
                        _roll_conv_channel(
                            mixed_qkv,
                            conv_state,
                            cs_idx,
                            row_base,
                            k_base + k_start + c,
                            T,
                            W,
                        )
                    if lane_id == 0:
                        qk_arrival_counters[counter_idx] = Int32(0)
        else:
            if warp_idx == 0:
                for c in cutlass.range_constexpr(vec_size):
                    _roll_conv_channel(
                        mixed_qkv,
                        conv_state,
                        cs_idx,
                        row_base,
                        q_base + k_start + c,
                        T,
                        W,
                    )
                    _roll_conv_channel(
                        mixed_qkv,
                        conv_state,
                        cs_idx,
                        row_base,
                        k_base + k_start + c,
                        T,
                        W,
                    )

        # Build the causal K-K and Q-K correction matrices with warp reductions.
        for j in cutlass.range_constexpr(ppw_tc):
            r_red[j] = cutlass.Float32(0.0)
        p_ctr = 0
        for i_t in cutlass.range_constexpr(T):
            for i_i in cutlass.range_constexpr(i_t):
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
            for i_i in cutlass.range_constexpr(i_t + 1):
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
                r_red[j] += cute.arch.shuffle_sync_bfly(r_red[j], offset=off, mask=-1, mask_and_clamp=31)
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

        # Build gated key suffixes and the full-chunk decay for state update.
        if tidx < K:
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
                    if cta_i_v == 0:
                        k_buf[i_n, i_t, i_hv, kc] = kn
                        g_buf[i_n, i_t, i_hv, kc] = g_t
            sBlast[kc] = bcum
        cute.arch.barrier()
        if tidx < BT * BT:
            ri = tidx // BT
            ci = tidx % BT
            sInv[ri, ci] = cutlass.Float32(1.0) if ri == ci else cutlass.Float32(0.0)
            sLp[ri, ci] = sL[ri, ci]
        cute.arch.barrier()

        # Invert the fixed-size lower-triangular correction matrix in shared memory.
        ri = tidx // BT
        ci = tidx % BT
        for step in cutlass.range_constexpr(3):
            if tidx < 2 * BT * BT:
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

        # Process each V tile through V conv, recurrence correction, and writeback.
        for vb in cutlass.range_constexpr(num_v_blocks):
            v_base = (i_v * num_v_blocks + vb) * BV
            shared_v_base = warp_group * BV
            packed_v_base = 2 * H * K + i_hv * V + v_base
            v_num_segments: cutlass.Constexpr[int] = T // 2
            if cutlass.const_expr(v_num_segments < 1):
                v_num_segments = 1
            if cutlass.const_expr(v_num_segments > 128 // BV):
                v_num_segments = 128 // BV
            if local_tidx < v_num_segments * BV:
                v_segment = local_tidx // BV
                v_channel = local_tidx - v_segment * BV
                _conv_channel_segment(
                    mixed_qkv,
                    conv_weight,
                    conv_bias,
                    conv_state,
                    inter_conv_window,
                    sX,
                    sU,
                    cs_idx,
                    iw_idx,
                    row_base,
                    packed_v_base + v_channel,
                    0,
                    shared_v_base + v_channel,
                    True,
                    v_segment == v_num_segments - 1,
                    v_segment,
                    v_num_segments,
                    T,
                    W,
                    has_bias,
                    fast_math,
                )
            # Stage the current VK state tile for the tensor-core matvecs.
            row_vecs = K // vec_size
            for j in cutlass.range_constexpr(BV * K // (128 * vec_size)):
                flat = j * 128 + local_tidx
                s_row = flat // row_vecs
                s_col = flat % row_vecs
                h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_base + s_row, s_col))
                cute.autovec_copy(h_tile, r_s)
                for cc in cutlass.range_constexpr(vec_size):
                    sS0[shared_v_base + s_row, s_col * vec_size + cc] = r_s[cc]
            cute.arch.barrier()
            if local_tidx < BV:
                for w in cutlass.range_constexpr(W - 1):
                    conv_state[cs_idx, packed_v_base + local_tidx, w] = sU[w, shared_v_base + local_tidx]

            nb = local_warp * 8
            vc0 = nb + 2 * tig
            vc1 = nb + 2 * tig + 1
            svc0 = shared_v_base + vc0
            svc1 = shared_v_base + vc1
            e0 = cutlass.Float32(0.0)
            e1 = cutlass.Float32(0.0)
            e2 = cutlass.Float32(0.0)
            e3 = cutlass.Float32(0.0)
            # Compute S0*k and S0*q together with 3xTF32 MMA.
            for ks in cutlass.range_constexpr(K // 8):
                kb = ks * 8
                a0 = sKQ[gid, kb + tig]
                a1 = sKQ[gid + 8, kb + tig]
                a2 = sKQ[gid, kb + tig + 4]
                a3 = sKQ[gid + 8, kb + tig + 4]
                b0 = sS0[shared_v_base + nb + gid, kb + tig]
                b1 = sS0[shared_v_base + nb + gid, kb + tig + 4]
                e0, e1, e2, e3 = _mma_m16n8k8_3xtf32(a0, a1, a2, a3, b0, b1, e0, e1, e2, e3)
            vmask = cutlass.Float32(1.0) if gid < T else cutlass.Float32(0.0)
            vv0 = cutlass.Float32(0.0)
            vv1 = cutlass.Float32(0.0)
            if gid < T:
                vv0 = sX[gid, svc0]
                vv1 = sX[gid, svc1]
            sX[gid, svc0] = sBeta[gid] * (vv0 * vmask - e0)
            sX[gid, svc1] = sBeta[gid] * (vv1 * vmask - e1)
            cute.arch.barrier()

            # Apply the triangular inverse to obtain the pseudo-values u.
            f0 = cutlass.Float32(0.0)
            f1 = cutlass.Float32(0.0)
            for l in cutlass.range_constexpr(BT):
                f0 += sInv[gid, l] * sX[l, svc0]
                f1 += sInv[gid, l] * sX[l, svc1]
            sU[gid, svc0] = f0
            sU[gid, svc1] = f1
            if cutlass.const_expr(write_ubuf):
                if gid < T:
                    d_buf[i_n, gid, i_hv, v_base + vc0] = f0
                    d_buf[i_n, gid, i_hv, v_base + vc1] = f1
            cute.arch.barrier()
            # Add the causal u correction to S0*q and emit the verified output.
            if cutlass.const_expr(emit_output):
                if gid < T:
                    ov0 = e2
                    ov1 = e3
                    for l in cutlass.range_constexpr(BT):
                        ov0 += sP[gid, l] * sU[l, svc0]
                        ov1 += sP[gid, l] * sU[l, svc1]
                    o[i_n, gid, i_hv, v_base + vc0] = cutlass.BFloat16(ov0)
                    o[i_n, gid, i_hv, v_base + vc1] = cutlass.BFloat16(ov1)

            # Optionally commit the full verified chunk into the VK state.
            if cutlass.const_expr(not disable_state_update):
                m_tiles: cutlass.Constexpr[int] = BV // 16
                pairs: cutlass.Constexpr[int] = m_tiles * (K // 8)
                for pp in cutlass.range_constexpr((pairs + num_warps - 1) // num_warps):
                    pidx = pp * num_warps + local_warp
                    if pidx < pairs:
                        m_t = pidx % m_tiles
                        n_t = pidx // m_tiles
                        mb = m_t * 16
                        nb2 = n_t * 8
                        g0 = cutlass.Float32(0.0)
                        g1 = cutlass.Float32(0.0)
                        g2 = cutlass.Float32(0.0)
                        g3 = cutlass.Float32(0.0)
                        aa0 = sU[tig, shared_v_base + mb + gid]
                        aa1 = sU[tig, shared_v_base + mb + gid + 8]
                        aa2 = sU[tig + 4, shared_v_base + mb + gid]
                        aa3 = sU[tig + 4, shared_v_base + mb + gid + 8]
                        bb0 = sKsuf[tig, nb2 + gid]
                        bb1 = sKsuf[tig + 4, nb2 + gid]
                        g0, g1, g2, g3 = _mma_m16n8k8_3xtf32(aa0, aa1, aa2, aa3, bb0, bb1, g0, g1, g2, g3)
                        for fi in cutlass.range_constexpr(4):
                            vrow = mb + gid + (fi // 2) * 8
                            kcol = nb2 + 2 * tig + (fi % 2)
                            gv = g0
                            if cutlass.const_expr(fi == 1):
                                gv = g1
                            if cutlass.const_expr(fi == 2):
                                gv = g2
                            if cutlass.const_expr(fi == 3):
                                gv = g3
                            h0_source[flat_state_idx, v_base + vrow, kcol] = (
                                sBlast[kcol] * sS0[shared_v_base + vrow, kcol] + gv
                            )
            cute.arch.barrier()


@cute.jit
def run_kda_conv_mtp_tensor_core_kvbuffer_kernel(
    mixed_qkv: cute.Tensor,
    conv_weight: cute.Tensor,
    conv_bias: cute.Tensor,
    conv_state: cute.Tensor,
    conv_state_indices: cute.Tensor,
    qk_arrival_counters: cute.Tensor,
    inter_conv_window: cute.Tensor,
    inter_state_indices: cute.Tensor,
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
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
    W: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    emit_output: cutlass.Constexpr[bool],
    write_ubuf: cutlass.Constexpr[bool],
    has_bias: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
    use_lower_bound: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    # Launch one physical CTA for each request, V head, and packed V tile.
    cta_v_groups: cutlass.Constexpr[int] = 1
    if cutlass.const_expr(num_v_tiles >= 2):
        cta_v_groups = 2
    physical_v_tiles: cutlass.Constexpr[int] = num_v_tiles // cta_v_groups
    group_bv: cutlass.Constexpr[int] = cta_v_groups * BV
    grid_size = h0_indices.layout.shape[0] * HV * physical_v_tiles
    smem_bytes = (
        2 * 4 * BT * (K + 4)
        + 4 * BT * (K + 8)
        + 4 * BT * K
        + 4 * BT
        + 4 * K
        + 4 * 64 * 12
        + 4 * 4 * BT * (BT + 1)
        + 2 * 4 * BT * (group_bv + 1)
        + 4 * group_bv * (K + 4)
        + 512
    )
    kda_conv_mtp_tensor_core_kvbuffer_kernel(
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        conv_state_indices,
        qk_arrival_counters,
        inter_conv_window,
        inter_state_indices,
        h0_source,
        A_log,
        a,
        dt_bias,
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
        W,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        has_bias,
        fast_math,
        use_lower_bound,
        lower_bound,
    ).launch(grid=(grid_size, 1, 1), block=[128 * cta_v_groups, 1, 1], smem=smem_bytes, stream=stream)


_compiled_conv_tensor_core_kvbuffer_kernels: dict[tuple, object] = {}


def _get_compiled_conv_tensor_core_kvbuffer_kernel(
    N,
    T,
    H,
    HV,
    K,
    V,
    D,
    pool_size,
    lines,
    inter_lines,
    bv,
    num_v_tiles,
    scale,
    use_qk_l2norm,
    disable_state_update,
    emit_output,
    write_ubuf,
    has_bias,
    softplus_beta,
    softplus_threshold,
    opt_level=3,
    fast_math=True,
    use_lower_bound=False,
    lower_bound=0.0,
):
    # Cache JIT artifacts by every static shape and execution-mode parameter.
    key = (
        T,
        H,
        HV,
        K,
        V,
        D,
        bv,
        num_v_tiles,
        scale,
        use_qk_l2norm,
        disable_state_update,
        emit_output,
        write_ubuf,
        has_bias,
        softplus_beta,
        softplus_threshold,
        opt_level,
        fast_math,
        use_lower_bound,
        lower_bound,
    )
    if key in _compiled_conv_tensor_core_kvbuffer_kernels:
        return _compiled_conv_tensor_core_kvbuffer_kernels[key]

    dev = "cuda"
    mixed_qkv = torch.zeros(N * T, D, dtype=torch.bfloat16, device=dev)
    conv_weight = torch.zeros(D, WCONV, dtype=torch.float32, device=dev)
    conv_bias = torch.zeros(D, dtype=torch.float32, device=dev)
    conv_state = torch.zeros(lines, D, WCONV - 1, dtype=torch.float32, device=dev)
    conv_state_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    qk_arrival_counters = torch.zeros(N * H, dtype=torch.int32, device=dev)
    inter_conv_window = torch.zeros(inter_lines, T, D, WCONV - 1, dtype=torch.float32, device=dev)
    inter_state_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    h0_source = torch.zeros(pool_size * HV, V, K, dtype=torch.float32, device=dev)
    A_log = torch.zeros(HV, dtype=torch.float32, device=dev)
    a = torch.zeros(N, T, HV, K, dtype=torch.bfloat16, device=dev)
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device=dev)
    b = torch.zeros(N, T, HV, dtype=torch.bfloat16, device=dev)
    o = torch.zeros(N, T, HV, V, dtype=torch.bfloat16, device=dev)
    h0_indices = torch.zeros(N, dtype=torch.int32, device=dev)
    d_buf = torch.zeros(N, T, HV, V, dtype=torch.float32, device=dev)
    k_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device=dev)
    g_buf = torch.zeros(N, T, HV, K, dtype=torch.float32, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        run_kda_conv_mtp_tensor_core_kvbuffer_kernel,
        _dl(mixed_qkv, True),
        _dl(conv_weight),
        _dl(conv_bias),
        _dl(conv_state, True),
        _dli(conv_state_indices),
        _dli(qk_arrival_counters),
        _dl(inter_conv_window, True),
        _dli(inter_state_indices),
        _dl(h0_source, True),
        _dl(A_log),
        _dl(a, True),
        _dl(dt_bias),
        _dl(b, True),
        _dl(o, True),
        _dli(h0_indices),
        _dl(d_buf, True),
        _dl(k_buf, True),
        _dl(g_buf, True),
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
        W=WCONV,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        emit_output=emit_output,
        write_ubuf=write_ubuf,
        has_bias=has_bias,
        fast_math=fast_math,
        use_lower_bound=use_lower_bound,
        lower_bound=lower_bound,
        stream=stream,
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_conv_tensor_core_kvbuffer_kernels[key] = compiled
    logger.info(
        "CuTe DSL fused conv + KDA MTP tensor-core-KVBuffer compiled: "
        f"N={N}, T={T}, H={H}, HV={HV}, bv={bv}, num_v_tiles={num_v_tiles}"
    )
    return compiled


def kda_conv_decode_mtp_kvbuffer(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
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
) -> torch.Tensor:
    """Run fused width-4 conv and tensor-core KVBuffer verify.

    This operator requires ``K=128``, ``V % 32 == 0`` and ``1<=T<=8``.
    Conv state/window mutation and optional ``d/k/g`` buffers match the
    existing KVBuffer flush ABI.
    """
    H, HV, K, V = num_q_heads, num_v_heads, head_k_dim, head_v_dim
    seq_len, D = mixed_qkv.shape
    if not 1 <= T <= BT:
        raise ValueError(f"tensor-core-kvbuffer requires 1<=T<={BT}, got {T}")
    if seq_len % T != 0:
        raise ValueError(f"mixed_qkv rows {seq_len} must be divisible by T={T}")
    N = seq_len // T
    if K != TILE_K or K != 128:
        raise ValueError(f"tensor-core-kvbuffer requires K={TILE_K}=128, got {K}")
    if D != 2 * H * K + HV * V:
        raise ValueError(f"packed dim mismatch: {D} vs {2 * H * K + HV * V}")
    if HV < H or HV % H != 0:
        raise ValueError(f"requires HV to be a multiple of H, got H={H}, HV={HV}")
    if bv != 32 or V % bv != 0:
        raise ValueError(f"tensor-core-kvbuffer requires bv=32 dividing V, got V={V}, bv={bv}")
    if conv_weight.shape != (D, WCONV):
        raise ValueError(f"conv_weight shape must be {(D, WCONV)}, got {tuple(conv_weight.shape)}")
    if conv_state.dim() != 3 or tuple(conv_state.shape[1:]) != (D, WCONV - 1):
        raise ValueError(f"conv_state shape must be [lines, {D}, {WCONV - 1}]")
    expected_window_tail = (T, D, WCONV - 1)
    if intermediate_conv_window.dim() != 4 or tuple(intermediate_conv_window.shape[1:]) != expected_window_tail:
        raise ValueError(f"intermediate_conv_window tail must be {expected_window_tail}")
    if mixed_qkv.dtype != torch.bfloat16:
        raise TypeError(f"mixed_qkv must be bfloat16, got {mixed_qkv.dtype}")
    for name, tensor in (
        ("conv_weight", conv_weight),
        ("conv_state", conv_state),
        ("intermediate_conv_window", intermediate_conv_window),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {tensor.dtype}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if num_v_tiles <= 0:
        # Expose more V tiles for small workloads and reduce CTA count at scale.
        work_units = N * HV
        if work_units <= 64:
            num_v_tiles = 4
        elif work_units < 512:
            num_v_tiles = 2
        else:
            num_v_tiles = 1
    if num_v_tiles > 1 and num_v_tiles % 2 != 0:
        raise ValueError(f"num_v_tiles must be 1 or even, got {num_v_tiles}")
    if (V // bv) % num_v_tiles != 0:
        raise ValueError(f"num_v_tiles must divide V//bv, got {num_v_tiles=}")

    h0_source, pool_size, _ = _normalize_state_source(
        ssm_states, N=N, HV=HV, K=K, V=V, device=mixed_qkv.device, state_layout="vk"
    )
    a = _normalize_mtp_a(a, N=N, T=T, HV=HV, K=K)
    if b.dim() != 3 or tuple(b.shape) != (N, T, HV):
        raise ValueError(f"b shape must be {(N, T, HV)}, got {tuple(b.shape)}")
    o = _prepare_output_tensor(mixed_qkv, out, (N, T, HV, V))
    write_ubuf = _validate_ubuffers(d_buffer, k_buffer, g_buffer, (N, T, HV, V), (N, T, HV, K))
    # Reuse caller-provided d/k/g scratch only when a later flush needs it.
    if write_ubuf:
        d_buf, k_buf, g_buf = d_buffer, k_buffer, g_buffer
    else:
        d_buf = torch.empty(N, T, HV, V, dtype=torch.float32, device=mixed_qkv.device)
        k_buf = torch.empty(N, T, HV, K, dtype=torch.float32, device=mixed_qkv.device)
        g_buf = torch.empty(N, T, HV, K, dtype=torch.float32, device=mixed_qkv.device)
    has_bias = conv_bias is not None
    conv_bias_t = conv_bias if has_bias else _get_zero_conv_bias(D, mixed_qkv.device)
    if has_bias and (conv_bias_t.shape != (D,) or conv_bias_t.dtype != torch.float32):
        raise ValueError(f"conv_bias must be float32 with shape {(D,)}")
    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)
    conv_state_indices = _normalize_state_indices(
        conv_state_indices, N=N, pool_size=conv_state.shape[0], device=mixed_qkv.device
    )
    cache_indices = _normalize_state_indices(cache_indices, N=N, pool_size=pool_size, device=mixed_qkv.device)
    intermediate_state_indices = _normalize_state_indices(
        intermediate_state_indices,
        N=N,
        pool_size=intermediate_conv_window.shape[0],
        device=mixed_qkv.device,
    )
    mixed_qkv = mixed_qkv if mixed_qkv.is_contiguous() else mixed_qkv.contiguous()
    conv_weight = conv_weight if conv_weight.is_contiguous() else conv_weight.contiguous()
    if not conv_state.is_contiguous():
        raise ValueError("conv_state must be contiguous because it is mutated in place")
    if not intermediate_conv_window.is_contiguous():
        raise ValueError("intermediate_conv_window must be contiguous because it is an output")
    a = a if a.is_contiguous() else a.contiguous()
    b = b if b.is_contiguous() else b.contiguous()

    # Each completed launch resets its stream-local counters in-kernel; avoiding a
    # separate zero_ launch keeps the fused CUDA Graph path launch-minimal.
    qk_arrival_counters = _get_qk_arrival_counters(mixed_qkv.device, N, H)
    h0_source_flat = h0_source.view(pool_size * HV, V, K)
    compiled = _get_compiled_conv_tensor_core_kvbuffer_kernel(
        N,
        T,
        H,
        HV,
        K,
        V,
        D,
        pool_size,
        conv_state.shape[0],
        intermediate_conv_window.shape[0],
        bv,
        num_v_tiles,
        scale,
        use_qk_l2norm_in_kernel,
        disable_state_update,
        emit_output,
        write_ubuf,
        has_bias,
        softplus_beta,
        softplus_threshold,
        opt_level,
        fast_math,
        lower_bound is not None,
        0.0 if lower_bound is None else float(lower_bound),
    )
    compiled(
        mixed_qkv,
        conv_weight,
        conv_bias_t,
        conv_state,
        conv_state_indices,
        qk_arrival_counters,
        intermediate_conv_window,
        intermediate_state_indices,
        h0_source_flat,
        A_log,
        a,
        dt_bias,
        b,
        o,
        cache_indices,
        d_buf,
        k_buf,
        g_buf,
        _get_cached_stream(mixed_qkv.device),
    )
    return o
