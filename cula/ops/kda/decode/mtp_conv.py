"""Fused KDA causal-conv1d + MTP verify, CuTe DSL. Two variants + auto dispatch.

Fuses the depthwise causal conv1d (width 4, SiLU) into the sigmoid-gating
delta-rule MTP verify recurrence, consuming packed ``mixed_qkv`` directly (no
split/transpose). Matches the sglang Triton ``fused_kda_conv_gating_verify``
semantics: per-step conv-window + SSM snapshots, conv_state rolled at the
epilogue, real SSM state not written back (verify). Scope: chain, W=4, T>=W-1.

small_batch variant (small/medium batch): 2 warps/CTA, grid = N*HV*(V//BV); lane holds
  K[4*lane:4*lane+4] in both warps. warp 0 (producer) runs token t's conv
  (q/k conv+silu+l2norm, v-conv Option B) into a double-buffered SMEM slot;
  warp 1 (consumer) runs token t-1's recurrence + gate/beta (computed one token
  ahead) from the other slot; decoupled named-barrier handoff (producer may lead
  by 2 tokens). Per-tier knobs: bv=8 (N*HV<=32) / bv=16 (>=64);
  weights_in_smem at N*HV>=256 (reg relief pays off once multi-wave).

large_batch variant (large batch): 8 warps/CTA, grid = N*HV*(V//tile_v), tile_v=8*BVW.
  Warp 0 produces shared q/k conv + l2norm + gate + beta -> SMEM (no per-v-tile
  redundant q/k); each warp does BVW v-cols of recurrence+v-conv+snapshot. Low
  state regs (r_h=BVW*vec_size) -> high occupancy -> beats triton at large N.

Shared q/k conv_state (read by every v-tile/v-head sharing a q/k head, rolled +
written once) is written by the LAST CTA in the sharing group (largest bidx),
so its epilogue write lands after every earlier-dispatched non-owner has read
the history -> race-free, same rolled output as the reference.

Dispatch (variant="auto"): large_batch if N*HV>=768 else small_batch (graph-timed crossover, L20X).
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


NWARP2 = 2


@cute.kernel
def kda_conv_verify_kernel(
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
    weights_in_smem: cutlass.Constexpr[bool],
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
    is_qk_owner = (i_v == num_v_tiles - 1) and (i_hv % (HV // H) == (HV // H) - 1)
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    q_base = i_h * K
    k_base = H * K + i_h * K
    v_base = 2 * H * K + i_hv * V + i_v * BV

    # double-buffered producer->consumer handoff (slot = token % 2)
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((2, K), stride=(K, 1)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((2, K), stride=(K, 1)), 16)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((2, BV), stride=(BV, 1)), 16)
    if cutlass.const_expr(weights_in_smem):
        # q/k conv weights in SMEM (reg relief at bv>=16): lane-major, odd stride
        # vec_size*W+1 so fixed-(c,w) reads across the warp hit distinct banks.
        sWq = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32, vec_size, W), stride=(vec_size * W + 1, W, 1)), 16)
        sWk = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32, vec_size, W), stride=(vec_size * W + 1, W, 1)), 16)

    if cache_idx >= 0:
        # ---- producer (warp 0) conv registers + preamble ----
        r_qhist = cute.make_rmem_tensor(cute.make_layout((vec_size, W - 1), stride=(W - 1, 1)), cutlass.Float32)
        r_khist = cute.make_rmem_tensor(cute.make_layout((vec_size, W - 1), stride=(W - 1, 1)), cutlass.Float32)
        r_qw = cute.make_rmem_tensor(cute.make_layout((vec_size, W), stride=(W, 1)), cutlass.Float32)
        r_kw = cute.make_rmem_tensor(cute.make_layout((vec_size, W), stride=(W, 1)), cutlass.Float32)
        r_qb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_kb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_pq = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_pk = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_vhist = cute.make_rmem_tensor(cute.make_layout((W - 1,), stride=(1,)), cutlass.Float32)
        r_vw = cute.make_rmem_tensor(cute.make_layout((W,), stride=(1,)), cutlass.Float32)
        r_vb = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32)
        # producer raw double-buffer: prefetch token t+1's q/k/v while computing t
        r_xq = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
        r_xk = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
        r_xv = [cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]

        if warp == 0:
            # vectorized preamble: this lane's adjacent q/k channels load their conv
            # history + weights as float4 groups; reg layout [c*(W-1)+w]/[c*W+w] (channel-major).
            qh_tile = cute.coalesce(cute.local_tile(conv_state, (1, vec_size, W - 1), (cs_idx, q_base // vec_size + lane, 0)))
            cute.autovec_copy(qh_tile, r_qhist)
            kh_tile = cute.coalesce(cute.local_tile(conv_state, (1, vec_size, W - 1), (cs_idx, k_base // vec_size + lane, 0)))
            cute.autovec_copy(kh_tile, r_khist)
            qw_tile = cute.coalesce(cute.local_tile(conv_weight, (vec_size, W), (q_base // vec_size + lane, 0)))
            kw_tile = cute.coalesce(cute.local_tile(conv_weight, (vec_size, W), (k_base // vec_size + lane, 0)))
            cute.autovec_copy(qw_tile, r_qw)
            cute.autovec_copy(kw_tile, r_kw)
            if cutlass.const_expr(weights_in_smem):
                # vectorized gmem load into regs, then scalar-store to SMEM (odd stride
                # not 16B-aligned); r_qw/r_kw die after preamble -> loop at lower reg count.
                for c in cutlass.range_constexpr(vec_size):
                    for w in cutlass.range_constexpr(W):
                        sWq[(lane, c, w)] = r_qw[c, w]
                        sWk[(lane, c, w)] = r_kw[c, w]
            if cutlass.const_expr(has_bias):
                qb_tile = cute.local_tile(conv_bias, (vec_size,), (q_base // vec_size + lane,))
                cute.autovec_copy(qb_tile, r_qb)
                kb_tile = cute.local_tile(conv_bias, (vec_size,), (k_base // vec_size + lane,))
                cute.autovec_copy(kb_tile, r_kb)
            else:
                for c in cutlass.range_constexpr(vec_size):
                    r_qb[c] = cutlass.Float32(0.0)
                    r_kb[c] = cutlass.Float32(0.0)
            if lane < BV:
                vch = v_base + lane
                vh_tile = cute.local_tile(conv_state, (1, 1, W - 1), (cs_idx, vch, 0))
                cute.autovec_copy(vh_tile, r_vhist)
                vw_tile = cute.local_tile(conv_weight, (1, W), (vch, 0))
                cute.autovec_copy(vw_tile, r_vw)
                if cutlass.const_expr(has_bias):
                    r_vb[0] = cutlass.Float32(conv_bias[vch])
                else:
                    r_vb[0] = cutlass.Float32(0.0)
            # pipeline fill: prefetch token 0's raw q/k/v into buffer 0
            xq0 = cute.local_tile(mixed_qkv, (1, vec_size), (i_n * T, q_base // vec_size + lane))
            cute.autovec_copy(xq0, r_xq[0])
            xk0 = cute.local_tile(mixed_qkv, (1, vec_size), (i_n * T, k_base // vec_size + lane))
            cute.autovec_copy(xk0, r_xk[0])
            xv0 = cute.local_tile(mixed_qkv, (1, BV), (i_n * T, v_base // BV))
            cute.autovec_copy(xv0, r_xv[0])

        # ---- consumer (warp 1) recurrence registers + state preamble ----
        r_h = cute.make_rmem_tensor(cute.make_layout((BV * vec_size,), stride=(1,)), cutlass.Float32)
        r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_v = cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.Float32)
        r_red = cute.make_rmem_tensor(cute.make_layout((BV,), stride=(1,)), cutlass.Float32)
        # consumer a/b double-buffer: prefetch next consumed token's gate inputs
        r_abf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
        r_bbf = [cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32) for _ in range(2)]
        # gate/beta double-buffer: gate(t) computed one iteration ahead (at iter t,
        # concurrent with producer conv(t) and consumer rec(t-1)); rec(ct) reads slot ct%2
        r_gbf = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32) for _ in range(2)]
        r_betabf = [cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32) for _ in range(2)]
        flat_state_idx = cache_idx * HV + i_hv
        if warp == 1:
            dtb_tile = cute.local_tile(dt_bias, (1, vec_size), (i_hv, lane))  # gate moved to consumer
            cute.autovec_copy(dtb_tile, r_dtb)
            a0 = cute.local_tile(a, (1, 1, 1, vec_size), (i_n, 0, i_hv, lane))
            cute.autovec_copy(a0, r_abf[0])
            r_bbf[0][0] = cutlass.Float32(b[i_n, 0, i_hv])
            for vv in cutlass.range_constexpr(BV):
                h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, i_v * BV + vv, lane))
                cute.autovec_copy(h_tile, r_h4)
                for c in cutlass.range_constexpr(vec_size):
                    r_h[vv * vec_size + c] = r_h4[c]

        # decoupled producer/consumer loops: full[s]=id 1+s, empty[s]=id 3+s; each
        # release needs 64 threads; producer may run up to 2 tokens ahead of consumer.
        if warp == 0:
            for i_t in cutlass.range_constexpr(T):
                if cutlass.const_expr(True):
                    ps = i_t % 2
                    # wait for the consumer to free slot ps (first two writes are free)
                    if cutlass.const_expr(i_t >= 2):
                        cute.arch.barrier(barrier_id=3 + ps, number_of_threads=64)
                    # prefetch token i_t+1's raw into the other buffer (loads overlap this
                    # token's conv/silu + the concurrent consumer recurrence)
                    if cutlass.const_expr(i_t + 1 < T):
                        nps = (i_t + 1) % 2
                        rown = i_n * T + i_t + 1
                        xqn = cute.local_tile(mixed_qkv, (1, vec_size), (rown, q_base // vec_size + lane))
                        cute.autovec_copy(xqn, r_xq[nps])
                        xkn = cute.local_tile(mixed_qkv, (1, vec_size), (rown, k_base // vec_size + lane))
                        cute.autovec_copy(xkn, r_xk[nps])
                        xvn = cute.local_tile(mixed_qkv, (1, BV), (rown, v_base // BV))
                        cute.autovec_copy(xvn, r_xv[nps])
                    for c in cutlass.range_constexpr(vec_size):
                        qch = q_base + vec_size * lane + c
                        xq = cutlass.Float32(r_xq[ps][c])
                        acc = r_qb[c]
                        if cutlass.const_expr(weights_in_smem):
                            acc = acc + r_qhist[c, 0] * sWq[(lane, c, 0)]
                            acc = acc + r_qhist[c, 1] * sWq[(lane, c, 1)]
                            acc = acc + r_qhist[c, 2] * sWq[(lane, c, 2)]
                            acc = acc + xq * sWq[(lane, c, 3)]
                        else:
                            acc = acc + r_qhist[c, 0] * r_qw[c, 0]
                            acc = acc + r_qhist[c, 1] * r_qw[c, 1]
                            acc = acc + r_qhist[c, 2] * r_qw[c, 2]
                            acc = acc + xq * r_qw[c, 3]
                        r_qhist[c, 0] = r_qhist[c, 1]
                        r_qhist[c, 1] = r_qhist[c, 2]
                        r_qhist[c, 2] = xq
                        if cutlass.const_expr(save_conv_window):
                            if is_qk_owner:
                                inter_conv_window[iw_idx, i_t, qch, 0] = r_qhist[c, 0]
                                inter_conv_window[iw_idx, i_t, qch, 1] = r_qhist[c, 1]
                                inter_conv_window[iw_idx, i_t, qch, 2] = r_qhist[c, 2]
                        silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                        r_pq[c] = cutlass.Float32(cutlass.BFloat16(silu))
                    for c in cutlass.range_constexpr(vec_size):
                        kch = k_base + vec_size * lane + c
                        xk = cutlass.Float32(r_xk[ps][c])
                        acc = r_kb[c]
                        if cutlass.const_expr(weights_in_smem):
                            acc = acc + r_khist[c, 0] * sWk[(lane, c, 0)]
                            acc = acc + r_khist[c, 1] * sWk[(lane, c, 1)]
                            acc = acc + r_khist[c, 2] * sWk[(lane, c, 2)]
                            acc = acc + xk * sWk[(lane, c, 3)]
                        else:
                            acc = acc + r_khist[c, 0] * r_kw[c, 0]
                            acc = acc + r_khist[c, 1] * r_kw[c, 1]
                            acc = acc + r_khist[c, 2] * r_kw[c, 2]
                            acc = acc + xk * r_kw[c, 3]
                        r_khist[c, 0] = r_khist[c, 1]
                        r_khist[c, 1] = r_khist[c, 2]
                        r_khist[c, 2] = xk
                        if cutlass.const_expr(save_conv_window):
                            if is_qk_owner:
                                inter_conv_window[iw_idx, i_t, kch, 0] = r_khist[c, 0]
                                inter_conv_window[iw_idx, i_t, kch, 1] = r_khist[c, 1]
                                inter_conv_window[iw_idx, i_t, kch, 2] = r_khist[c, 2]
                        silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                        r_pk[c] = cutlass.Float32(cutlass.BFloat16(silu))
                    if cutlass.const_expr(use_qk_l2norm):
                        sum_q = cutlass.Float32(0.0)
                        sum_k = cutlass.Float32(0.0)
                        for c in cutlass.range_constexpr(vec_size):
                            sum_q += r_pq[c] * r_pq[c]
                            sum_k += r_pk[c] * r_pk[c]
                        for off in [16, 8, 4, 2, 1]:
                            sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=off, mask=-1, mask_and_clamp=31)
                            sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=off, mask=-1, mask_and_clamp=31)
                        inv_q = cute.rsqrt(sum_q + 1e-6, fastmath=fast_math) * scale
                        inv_k = cute.rsqrt(sum_k + 1e-6, fastmath=fast_math)
                        for c in cutlass.range_constexpr(vec_size):
                            r_pq[c] = r_pq[c] * inv_q
                            r_pk[c] = r_pk[c] * inv_k
                    else:
                        for c in cutlass.range_constexpr(vec_size):
                            r_pq[c] = r_pq[c] * scale
                    for c in cutlass.range_constexpr(vec_size):
                        sQ[(ps, vec_size * lane + c)] = r_pq[c]
                        sK[(ps, vec_size * lane + c)] = r_pk[c]
                    if lane < BV:
                        vch = v_base + lane
                        xv = cutlass.Float32(0.0)
                        for vv in cutlass.range_constexpr(BV):
                            xv = cutlass.Float32(r_xv[ps][vv]) if lane == vv else xv
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
                        sV[(ps, lane)] = cutlass.Float32(cutlass.BFloat16(silu))
                    # publish slot ps to the consumer
                    cute.arch.barrier_arrive(barrier_id=1 + ps, number_of_threads=64)
            # producer epilogue: conv_state writeback (overlaps consumer drain)
            if is_qk_owner:
                qh_out = cute.coalesce(cute.local_tile(conv_state, (1, vec_size, W - 1), (cs_idx, q_base // vec_size + lane, 0)))
                cute.autovec_copy(r_qhist, qh_out)
                kh_out = cute.coalesce(cute.local_tile(conv_state, (1, vec_size, W - 1), (cs_idx, k_base // vec_size + lane, 0)))
                cute.autovec_copy(r_khist, kh_out)
            if lane < BV:
                vch = v_base + lane
                vh_out = cute.local_tile(conv_state, (1, 1, W - 1), (cs_idx, vch, 0))
                cute.autovec_copy(r_vhist, vh_out)

        # ===== consumer: gate one token ahead + recurrence, own loop =====
        if warp == 1:
            for i_t in cutlass.range_constexpr(T + 1):
                ct = i_t - 1
                cslot = ct % 2
                # wait for producer to fill slot cslot, copy to regs, then free the slot
                if cutlass.const_expr(i_t >= 1):
                    cute.arch.barrier(barrier_id=1 + cslot, number_of_threads=64)
                    cute.autovec_copy(cute.coalesce(cute.local_tile(sQ, (1, vec_size), (cslot, lane))), r_q)
                    cute.autovec_copy(cute.coalesce(cute.local_tile(sK, (1, vec_size), (cslot, lane))), r_k)
                    for vv in cutlass.range_constexpr(BV):
                        r_v[vv] = sV[(cslot, vv)]
                    cute.arch.barrier_arrive(barrier_id=3 + cslot, number_of_threads=64)
                # gate segment: compute gate(i_t)/beta(i_t) into slot i_t%2 (consumed by
                # rec at the next iteration); prefetch a/b(i_t+1) into the other slot
                if cutlass.const_expr(i_t < T):
                    gslot = i_t % 2
                    if cutlass.const_expr(i_t + 1 < T):
                        an = cute.local_tile(a, (1, 1, 1, vec_size), (i_n, i_t + 1, i_hv, lane))
                        cute.autovec_copy(an, r_abf[(i_t + 1) % 2])
                        r_bbf[(i_t + 1) % 2][0] = cutlass.Float32(b[i_n, i_t + 1, i_hv])
                    for c in cutlass.range_constexpr(vec_size):
                        gx = cutlass.Float32(r_abf[gslot][c]) + r_dtb[c]
                        if cutlass.const_expr(use_lower_bound):
                            sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-r_exp_A * gx, fastmath=fast_math))
                            r_gbf[gslot][c] = cute.exp(lower_bound * sig, fastmath=fast_math)
                        else:
                            beta_x = softplus_beta * gx
                            sp = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                                cutlass.Float32(1.0) + cute.exp(softplus_beta * gx, fastmath=fast_math), fastmath=fast_math
                            )
                            use_sp = cutlass.Float32(1.0) if beta_x <= softplus_threshold else cutlass.Float32(0.0)
                            spx = use_sp * sp + (cutlass.Float32(1.0) - use_sp) * gx
                            r_gbf[gslot][c] = cute.exp(-r_exp_A * spx, fastmath=fast_math)
                    r_betabf[gslot][0] = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.exp(-r_bbf[gslot][0], fastmath=fast_math)
                    )
                # rec segment: token ct using gate/beta computed at iteration ct
                if cutlass.const_expr(i_t >= 1):
                    r_beta = r_betabf[cslot][0]
                    for vv in cutlass.range_constexpr(BV):
                        sv = cutlass.Float32(0.0)
                        for c in cutlass.range_constexpr(vec_size):
                            r_h[vv * vec_size + c] = r_h[vv * vec_size + c] * r_gbf[cslot][c]
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
                        o[(i_n, ct, i_hv, i_v * BV + vv)] = cutlass.BFloat16(r_red[vv])
                    if cutlass.const_expr(cache_intermediate_states):
                        flat_idx = i_n * T * HV + ct * HV + i_hv
                        for vv in cutlass.range_constexpr(BV):
                            for c in cutlass.range_constexpr(vec_size):
                                r_h4[c] = r_h[vv * vec_size + c]
                            inter_tile = cute.local_tile(intermediate_states, (1, 1, vec_size), (flat_idx, i_v * BV + vv, lane))
                            cute.autovec_copy(r_h4, inter_tile)

        # ---- consumer epilogue (producer writes conv_state at the end of its own loop) ----
        if warp == 1:
            if cutlass.const_expr(not disable_state_update):
                for vv in cutlass.range_constexpr(BV):
                    for c in cutlass.range_constexpr(vec_size):
                        r_h4[c] = r_h[vv * vec_size + c]
                    h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, i_v * BV + vv, lane))
                    cute.autovec_copy(r_h4, h_out)


@cute.jit
def run_kda_conv_verify_kernel(
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
    weights_in_smem: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    n_indices = h0_indices.layout.shape[0]
    num_v_tiles = cute.ceil_div(V, BV)
    grid_size = n_indices * HV * num_v_tiles
    smem_bytes = (2 * (2 * K) + 2 * BV) * 4 + 128  # sQ+sK [2,K] fp32 + sV + slack
    if cutlass.const_expr(weights_in_smem):
        smem_bytes = smem_bytes + 2 * 32 * (vec_size * W + 1) * 4  # sWq+sWk lane-major
    kda_conv_verify_kernel(
        mixed_qkv, conv_weight, conv_bias, conv_state, conv_state_indices,
        inter_conv_window, inter_state_indices, h0_source, A_log, a, dt_bias, b, o,
        intermediate_states, h0_indices, vec_size, num_v_tiles, BV, softplus_beta,
        softplus_threshold, scale, HV, T, H, K, V, W, use_qk_l2norm,
        disable_state_update, cache_intermediate_states, save_conv_window, has_bias,
        fast_math, use_lower_bound, lower_bound, weights_in_smem,
    ).launch(grid=(grid_size, 1, 1), block=[NWARP2 * 32, 1, 1], smem=smem_bytes, stream=stream)


_compiled_conv_verify_kernels: dict = {}


def _get_compiled(N, T, H, HV, K, V, D, pool_size, lines, BV, scale, use_qk_l2norm,
                      disable_state_update, cache_intermediate_states, save_conv_window,
                      has_bias, softplus_beta, softplus_threshold, use_lower_bound,
                      lower_bound, opt_level=3, fast_math=True, weights_in_smem=False):
    key = (T, H, HV, K, V, D, BV, scale, use_qk_l2norm, disable_state_update,
           cache_intermediate_states, save_conv_window, has_bias, softplus_beta,
           softplus_threshold, use_lower_bound, lower_bound, opt_level, fast_math,
           weights_in_smem)
    if key in _compiled_conv_verify_kernels:
        return _compiled_conv_verify_kernels[key]
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
        run_kda_conv_verify_kernel,
        dl(mixed_qkv, True), dl(conv_weight), dl(conv_bias), dl(conv_state, True),
        dli(conv_state_indices), dl(inter_conv_window, True), dli(inter_state_indices),
        dl(h0_source, True), dl(A_log), dl(a, True), dl(dt_bias), dl(b, True), dl(o, True),
        dl(inter_states, True) if cache_intermediate_states else dl(inter_states), dli(h0_indices),
        vec_size=VEC_SIZE, BV=BV, softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold, scale=scale, HV=HV, T=T, H=H, K=K, V=V,
        W=WCONV, use_qk_l2norm=use_qk_l2norm, disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states, save_conv_window=save_conv_window,
        has_bias=has_bias, fast_math=fast_math, use_lower_bound=use_lower_bound,
        lower_bound=lower_bound, weights_in_smem=weights_in_smem, stream=stream,
        options=f"--enable-tvm-ffi --opt-level {opt_level}",
    )
    _compiled_conv_verify_kernels[key] = compiled
    logger.info(f"cuLA fused conv+verify small_batch compiled: N={N} T={T} HV={HV} K={K} V={V} BV={BV}")
    return compiled


NWARP = 8


@cute.kernel
def kda_conv_verify_large_batch_kernel(
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
    # owner = LAST CTA (largest bidx) sharing this q/k head: its epilogue conv_state
    # write lands after every non-owner read the history (bidx ~ dispatch order) -> race-free.
    is_qk_owner = (i_v == num_v_tiles - 1) and (i_hv % (HV // H) == (HV // H) - 1)
    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=fast_math)

    q_base = i_h * K
    k_base = H * K + i_h * K
    # this warp's v-cols: global col = i_v*tile_v + warp*BVW + [0..BVW)
    v_col0 = i_v * tile_v + warp * BVW

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K, 1)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K, 1)), 16)
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T, K), stride=(K, 1)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,), stride=(1,)), 16)

    if cache_idx >= 0:
        # prefetch recurrent state early so the h0_source load latency overlaps the
        # producer conv compute (long_scoreboard is the dominant low-occupancy stall).
        r_h = cute.make_rmem_tensor(cute.make_layout((BVW * vec_size,), stride=(1,)), cutlass.Float32)
        r_h4 = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        flat_state_idx = cache_idx * HV + i_hv
        for vv in cutlass.range_constexpr(BVW):
            h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_col0 + vv, lane))
            cute.autovec_copy(h_tile, r_h4)
            for c in cutlass.range_constexpr(vec_size):
                r_h[vv * vec_size + c] = r_h4[c]
        # Producer: warp handles tokens i_t with i_t%NWARP==warp (T=4 -> 1 token/warp);
        # weights/history/bias load as float4 groups to cut exposed global-load latency.
        r_qw = cute.make_rmem_tensor(cute.make_layout((vec_size, W), stride=(W, 1)), cutlass.Float32)
        r_kw = cute.make_rmem_tensor(cute.make_layout((vec_size, W), stride=(W, 1)), cutlass.Float32)
        r_qhist = cute.make_rmem_tensor(cute.make_layout((vec_size, W - 1), stride=(W - 1, 1)), cutlass.Float32)
        r_khist = cute.make_rmem_tensor(cute.make_layout((vec_size, W - 1), stride=(W - 1, 1)), cutlass.Float32)
        r_qb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_kb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_pq = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_pk = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_g_out = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_dtb = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        bos = i_n * T
        qw_tile = cute.coalesce(cute.local_tile(conv_weight, (vec_size, W), (q_base // vec_size + lane, 0)))
        cute.autovec_copy(qw_tile, r_qw)
        kw_tile = cute.coalesce(cute.local_tile(conv_weight, (vec_size, W), (k_base // vec_size + lane, 0)))
        cute.autovec_copy(kw_tile, r_kw)
        qh_tile = cute.coalesce(cute.local_tile(conv_state, (1, vec_size, W - 1), (cs_idx, q_base // vec_size + lane, 0)))
        cute.autovec_copy(qh_tile, r_qhist)
        kh_tile = cute.coalesce(cute.local_tile(conv_state, (1, vec_size, W - 1), (cs_idx, k_base // vec_size + lane, 0)))
        cute.autovec_copy(kh_tile, r_khist)
        for c in cutlass.range_constexpr(vec_size):
            r_dtb[c] = cutlass.Float32(dt_bias[i_hv, vec_size * lane + c])
        if cutlass.const_expr(has_bias):
            qb_tile = cute.local_tile(conv_bias, (vec_size,), (q_base // vec_size + lane,))
            cute.autovec_copy(qb_tile, r_qb)
            kb_tile = cute.local_tile(conv_bias, (vec_size,), (k_base // vec_size + lane,))
            cute.autovec_copy(kb_tile, r_kb)
        else:
            for c in cutlass.range_constexpr(vec_size):
                r_qb[c] = cutlass.Float32(0.0)
                r_kb[c] = cutlass.Float32(0.0)

        for i_t in cutlass.range_constexpr(T):
            if i_t % NWARP == warp:
                # q/k conv: tap m at abs pos p=i_t-(W-1)+m (p<0 -> conv_state history col p+W-1);
                # prefetch this token's taps (float4) up front to overlap conv/l2norm/gate.
                r_xqm = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(W)]
                r_xkm = [cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16) for _ in range(W)]
                for m in cutlass.range_constexpr(W):
                    p = i_t - (W - 1) + m
                    if cutlass.const_expr(p >= 0):
                        xqm = cute.coalesce(cute.local_tile(mixed_qkv, (1, vec_size), (bos + p, q_base // vec_size + lane)))
                        cute.autovec_copy(xqm, r_xqm[m])
                        xkm = cute.coalesce(cute.local_tile(mixed_qkv, (1, vec_size), (bos + p, k_base // vec_size + lane)))
                        cute.autovec_copy(xkm, r_xkm[m])
                for c in cutlass.range_constexpr(vec_size):
                    acc = r_qb[c]
                    for m in cutlass.range_constexpr(W):
                        p = i_t - (W - 1) + m
                        if cutlass.const_expr(p >= 0):
                            xq = cutlass.Float32(r_xqm[m][c])
                        else:
                            xq = r_qhist[c, p + (W - 1)]
                        acc = acc + xq * r_qw[c, m]
                    silu = acc * (cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-acc, fastmath=fast_math)))
                    r_pq[c] = cutlass.Float32(cutlass.BFloat16(silu))
                # k conv
                for c in cutlass.range_constexpr(vec_size):
                    acc = r_kb[c]
                    for m in cutlass.range_constexpr(W):
                        p = i_t - (W - 1) + m
                        if cutlass.const_expr(p >= 0):
                            xk = cutlass.Float32(r_xkm[m][c])
                        else:
                            xk = r_khist[c, p + (W - 1)]
                        acc = acc + xk * r_kw[c, m]
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
                    r_g_out[c] = g
                cute.autovec_copy(r_pq, cute.coalesce(cute.local_tile(sQ, (1, vec_size), (i_t, lane))))
                cute.autovec_copy(r_pk, cute.coalesce(cute.local_tile(sK, (1, vec_size), (i_t, lane))))
                cute.autovec_copy(r_g_out, cute.coalesce(cute.local_tile(sG, (1, vec_size), (i_t, lane))))
                if lane == 0:
                    r_bb = cutlass.Float32(b[i_n, i_t, i_hv])
                    sBeta[i_t] = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-r_bb, fastmath=fast_math))

        cute.arch.barrier()

        # =============== Consumer: each warp does BVW v-cols =====================
        r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_g = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        r_v = cute.make_rmem_tensor(cute.make_layout((BVW,), stride=(1,)), cutlass.Float32)
        r_red = cute.make_rmem_tensor(cute.make_layout((BVW,), stride=(1,)), cutlass.Float32)
        r_vhist = cute.make_rmem_tensor(cute.make_layout((W - 1,), stride=(1,)), cutlass.Float32)
        r_vw = cute.make_rmem_tensor(cute.make_layout((W,), stride=(1,)), cutlass.Float32)
        r_vb = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.Float32)


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
        # double-buffer the raw v input: prefetch token t+1's v while consuming t, so its
        # per-token global-load latency overlaps the recurrence (low-occupancy small batch).
        r_xv = [cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), cutlass.BFloat16) for _ in range(2)]
        if lane < BVW:
            vch_pf = 2 * H * K + i_hv * V + v_col0 + lane
            r_xv[0][0] = mixed_qkv[i_n * T, vch_pf]

        for i_t in cutlass.range_constexpr(T):
            row = i_n * T + i_t
            # v conv (lane<BVW computes its col, shuffle-broadcast within warp)
            my_v = cutlass.Float32(0.0)
            if lane < BVW:
                vch = 2 * H * K + i_hv * V + v_col0 + lane
                if cutlass.const_expr(i_t + 1 < T):
                    r_xv[(i_t + 1) % 2][0] = mixed_qkv[i_n * T + i_t + 1, vch]
                xv = cutlass.Float32(r_xv[i_t % 2][0])
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

        # consumer epilogue: conv_state writeback at kernel end (race-free). q/k by
        # warp-0 owner from mixed_qkv (rolled = last W-1 raw inputs); v by each warp's lanes.
        if is_qk_owner and warp == 0:
            # rolled window = last W-1 abs positions p=T-(W-1)+w: p>=0 from mixed_qkv,
            # p<0 (only T<W-1) from conv_state col p+(W-1); read before overwrite (w ascending).
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
def run_kda_conv_verify_large_batch_kernel(
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
    smem_bytes = 3 * (4 * T * K) + 4 * T + 128  # sQ+sK+sG [T,K] fp32 + sBeta + slack
    kda_conv_verify_large_batch_kernel(
        mixed_qkv, conv_weight, conv_bias, conv_state, conv_state_indices,
        inter_conv_window, inter_state_indices, h0_source, A_log, a, dt_bias, b, o,
        intermediate_states, h0_indices,
        vec_size, num_v_tiles, BVW, tile_v, softplus_beta, softplus_threshold, scale,
        HV, T, H, K, V, W, use_qk_l2norm, disable_state_update, cache_intermediate_states,
        save_conv_window, has_bias, fast_math, use_lower_bound, lower_bound,
    ).launch(grid=(grid_size, 1, 1), block=[NWARP * 32, 1, 1], smem=smem_bytes, stream=stream)


_compiled_conv_verify_large_batch_kernels: dict = {}


def _get_compiled_large_batch(N, T, H, HV, K, V, D, pool_size, lines, BVW, tile_v, scale,
                     use_qk_l2norm, disable_state_update, cache_intermediate_states,
                     save_conv_window, has_bias, softplus_beta, softplus_threshold,
                     use_lower_bound, lower_bound, opt_level=3, fast_math=True):
    key = (T, H, HV, K, V, D, BVW, tile_v, scale, use_qk_l2norm, disable_state_update,
           cache_intermediate_states, save_conv_window, has_bias, softplus_beta,
           softplus_threshold, use_lower_bound, lower_bound, opt_level, fast_math)
    if key in _compiled_conv_verify_large_batch_kernels:
        return _compiled_conv_verify_large_batch_kernels[key]
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
        run_kda_conv_verify_large_batch_kernel,
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
    _compiled_conv_verify_large_batch_kernels[key] = compiled
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
    variant: str = "auto",   # "small_batch" (2-warp small/medium-batch) / "large_batch" (8-warp large-batch) / "auto"
    bvw: int = -1,           # large_batch: v-cols per warp (tile_v = 8*bvw); -1 = auto (8 on L20X)
    out: torch.Tensor | None = None,
):
    H, HV, K, V = num_q_heads, num_v_heads, head_k_dim, head_v_dim
    seq_len, D = mixed_qkv.shape
    N = seq_len // T
    assert K == TILE_K, f"requires K={TILE_K}, got {K}"
    assert D == 2 * H * K + HV * V, f"packed dim mismatch: {D} vs {2*H*K+HV*V}"
    work_units = N * HV

    if variant == "auto":
        # small/medium -> small_batch (2-warp: conv producer overlaps recurrence consumer);
        # large -> large_batch (8-warp, shared q/k, bandwidth-bound). Per-tier bv/wis knobs below.
        if work_units >= 512 and V % (NWARP * bvw) == 0:
            variant = "large_batch"
        else:
            variant = "small_batch"

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

    if variant == "large_batch":
        if bvw <= 0:
            # large batch: bvw=8 (tile_v=64) cuts redundant producer work, r_h=64 keeps occupancy;
            # small batch under-fills the GPU, so finer v-tiles (more CTAs) win -> snap to {1,2,4,8}.
            if work_units <= 8:
                bvw = 1
            elif work_units <= 16:
                bvw = 2
            elif work_units <= 32:
                bvw = 4
            else:
                bvw = 8
            if V % (NWARP * bvw) != 0:  # tile_v = NWARP*bvw must divide V
                bvw = 16 if V % (NWARP * 16) == 0 else (8 if V % (NWARP * 8) == 0 else V // NWARP)
        tile_v = NWARP * bvw
        assert V % tile_v == 0, f"large_batch requires V%(8*bvw)==0: V={V} bvw={bvw}"
        compiled = _get_compiled_large_batch(
            N, T, H, HV, K, V, D, slots, lines, bvw, tile_v, scale, use_qk_l2norm_in_kernel,
            True, cache_intermediate_states, True, has_bias, softplus_beta,
            softplus_threshold, use_lb, lb_val,
        )
    elif variant == "small_batch":  # pipelined 2-warp small_batch (small/medium batch)
        if bv <= 0:
            if work_units <= 32 and V % 8 == 0:
                bv = 8  # grid-underutilized: more CTAs, short per-warp v chain
            elif V % 16 == 0:
                bv = 16  # N*HV=64/128: halve CTA count to stay within 1 wave
            elif V % 8 == 0:
                bv = 8
            else:
                bv = 32
        assert V % bv == 0, f"V%bv!=0: V={V} bv={bv}"
        # weights_in_smem trades ~26 regs for SMEM-read latency: loses ~5-7% at
        # N*HV=64/128 (~1 wave) but wins at >=256 (multi-wave, occupancy-bound).
        wis = (work_units >= 256) and (bv >= 16)
        compiled = _get_compiled(
            N, T, H, HV, K, V, D, slots, lines, bv, scale, use_qk_l2norm_in_kernel,
            True, cache_intermediate_states, True, has_bias, softplus_beta,
            softplus_threshold, use_lb, lb_val, weights_in_smem=wis,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}; supported: 'auto', 'small_batch', 'large_batch'")

    compiled(
        mixed_qkv, conv_weight, conv_bias_t, conv_state, cache_indices,
        intermediate_conv_window, intermediate_state_indices, h0_source, A_log, a,
        dt_bias, b, o, inter_states_flat, cache_indices, stream,
    )
    return o
