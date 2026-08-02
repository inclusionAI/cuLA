# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightning Attention KVBuffer state-update kernel (paper Eq. 8 for LA).

After a parallel-verify cycle, advances the pooled state from h_init to
h_state_L for a per-batch accepted prefix length L = accepted_len[b]:

    h_running = h_init
    for i in 0..L-1:
        h_running = exp(-decay_scales[h]) * h_running + k_i ⊗ v_i
    s[cache_idx] = h_running

The loop body is bit-identical to the baseline T-loop body, so at L == T the
result is bit-equivalent to running the baseline with disable_state_update=False.

Reads s and pool-indexed k_buf/v_buf; writes s. Never touches q or o.

Grid: (B * HV * num_v_tiles, 1, 1), 128 threads/block — identical layout to the
baseline verify kernel, so the state write aligns with the verify kernel's h0 read.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import (
    from_dlpack,
    make_fake_compact_tensor,
    make_fake_stream,
)
from cutlass.cute.typing import Int32

from cula.lightning.la_decode_mtp import (
    NUM_THREADS_MTP,
    la_update_pair,
)
from cula.lightning.la_verify_kvbuffer import get_mtp_config
from cula.utils import USE_FAST_MATH, get_device_sm_version


@cute.kernel
def la_state_update_kernel(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] fp32 (read + written in place)
    decay_scales: cute.Tensor,  # [H] fp32
    h0_indices: cute.Tensor,  # [B] int32
    accepted_len: cute.Tensor,  # [B] int32
    k_buf: cute.Tensor,  # [pool_size, T, H, K] fp32
    v_buf: cute.Tensor,  # [pool_size, T, HV, V] fp32
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 4

    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    block_idx, _, _ = cute.arch.block_idx()
    i_v = block_idx % num_v_tiles
    tmp = block_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    L = accepted_len[i_n]

    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32)

    if cache_idx >= 0 and L > 0:
        r_decay = cute.exp(-cutlass.Float32(decay_scales[i_h]), fastmath=USE_FAST_MATH)
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        # Process `ilp_rows` V-rows per iteration. ilp_rows is a compile-time
        # constant, so range_constexpr fully unrolls the slot loops below — the
        # generated SASS is identical to hand-unrolling each ilp_rows value, but
        # one loop covers ilp_rows in {2, 4, 8}.
        num_chunks: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for chunk in cutlass.range_constexpr(num_chunks):
            v_idx_0 = i_v * tile_v + group_idx * rows_per_group + chunk * ilp_rows
            if v_idx_0 + (ilp_rows - 1) < V:
                # Load the ilp_rows h-state rows this thread owns into registers.
                for slot in cutlass.range_constexpr(ilp_rows):
                    h_tile = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_0 + slot, lane_in_group))
                    cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                # Recurrence: h = decay * h + k_i (x) v_i, for i in 0..L-1.
                for i in cutlass.range(0, L, unroll=0):
                    k_tile = cute.local_tile(k_buf, (1, 1, 1, vec_size), (cache_idx, i, i_h, lane_in_group))
                    cute.autovec_copy(k_tile, r_k)
                    for slot in cutlass.range_constexpr(ilp_rows):
                        r_v_s = cutlass.Float32(v_buf[cache_idx, i, i_hv, v_idx_0 + slot])
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[slot, j], r_h[slot, j + 1] = la_update_pair(
                                r_h[slot, j], r_h[slot, j + 1], r_k[j], r_k[j + 1], r_v_s, r_decay, use_packed_fma
                            )

                # Write the advanced state back in place.
                for slot in cutlass.range_constexpr(ilp_rows):
                    h_out = cute.local_tile(h0_source, (1, 1, vec_size), (flat_state_idx, v_idx_0 + slot, lane_in_group))
                    cute.autovec_copy(cute.slice_(r_h, (slot, None)), h_out)


@cute.jit
def run_la_state_update_kernel(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    h0_indices: cute.Tensor,
    accepted_len: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v
    grid_size = B * HV * num_v_tiles

    la_state_update_kernel(
        h0_source,
        decay_scales,
        h0_indices,
        accepted_len,
        k_buf,
        v_buf,
        vec_size,
        num_v_tiles,
        tile_v,
        B,
        T,
        H,
        HV,
        K,
        V,
        ilp_rows,
        use_packed_fma,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        stream=stream,
    )


@functools.cache
def _get_compiled_state_update_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    tile_v: int,
    vec_size: int,
    ilp_rows: int,
    use_packed_fma: bool,
):
    return {}


def _state_update_compile_cache(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    *,
    device: torch.device,
):
    """Return (cache dict, tile config tuple) for the given launch parameters."""
    tile_v, vec_size, ilp_rows = get_mtp_config(B, T, HV, V)
    assert V % ilp_rows == 0, f"V={V} % ilp_rows={ilp_rows} ≠ 0: partial row-blocks would be silently skipped"
    use_packed_fma = get_device_sm_version(device)[0] >= 10
    cache = _get_compiled_state_update_kernel(
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        tile_v,
        vec_size,
        ilp_rows,
        use_packed_fma,
    )
    return cache, (tile_v, vec_size, ilp_rows, use_packed_fma)


def get_compiled_state_update_kvbuffer_handle(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    *,
    device: torch.device,
):
    """Return a pre-compiled state-update kernel handle (benchmark kernel-only path).

    Call ``linear_attention_state_update_kvbuffer`` once with the same config first.
    """
    cache, _ = _state_update_compile_cache(
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        device=device,
    )
    compiled = cache.get("compiled")
    if compiled is None:
        raise RuntimeError(
            "State-update kernel not compiled for this config; call linear_attention_state_update_kvbuffer once first."
        )
    return compiled


def linear_attention_state_update_kvbuffer(
    k_buf: torch.Tensor,  # [pool_size, T, H, K] fp32
    v_buf: torch.Tensor,  # [pool_size, T, HV, V] fp32
    s: torch.Tensor,  # [pool_size, HV, V, K] fp32, WRITTEN IN PLACE
    decay_scales: torch.Tensor,  # [H] fp32
    h0_indices: torch.Tensor,  # [B] int32, -1 to skip
    accepted_len: torch.Tensor,  # [B] int32, in [0, T]
    T: int,
) -> None:
    """
    Advance pooled state from h_init to h_state_L per batch (KVBuffer Eq. 8).

    Reads k/v from fp32 pool-indexed buffers. This matches the SGLang/Ling
    integration path: verify writes per-layer draft k/v into the request pool,
    then commit advances the fp32 temporal state directly from those buffers.
    """
    pool_size, T_k, H, K = k_buf.shape
    assert T_k == T, f"k.shape[1]={T_k} doesn't match T={T}"
    assert K == 128, f"K={K} != 128: kernel hardcodes K=128 (threads_per_group, lane K-coverage)"
    if k_buf.dtype != torch.float32 or v_buf.dtype != torch.float32:
        raise ValueError(f"k_buf/v_buf must be torch.float32, got {k_buf.dtype}/{v_buf.dtype}")
    if s.dtype != torch.float32:
        raise ValueError(f"s must be torch.float32, got {s.dtype}")
    if decay_scales.dtype != torch.float32:
        raise ValueError(f"decay_scales must be torch.float32, got {decay_scales.dtype}")
    if h0_indices.dtype != torch.int32 or accepted_len.dtype != torch.int32:
        raise ValueError(f"h0_indices/accepted_len must be torch.int32, got {h0_indices.dtype}/{accepted_len.dtype}")
    if s.shape[0] != pool_size:
        raise ValueError(f"s pool_size={s.shape[0]} doesn't match k_buf pool_size={pool_size}")
    if v_buf.shape[:3] != (pool_size, T, s.shape[1]):
        raise ValueError(f"v_buf shape {tuple(v_buf.shape)} doesn't match expected prefix {(pool_size, T, s.shape[1])}")
    HV, V = s.shape[1], s.shape[2]
    if v_buf.shape != (pool_size, T, HV, V):
        raise ValueError(f"v_buf shape {tuple(v_buf.shape)} doesn't match expected {(pool_size, T, HV, V)}")
    if s.shape[3] != K:
        raise ValueError(f"s K={s.shape[3]} doesn't match k_buf K={K}")
    if decay_scales.shape[0] != H:
        raise ValueError(f"decay_scales length={decay_scales.shape[0]} doesn't match H={H}")
    B = h0_indices.shape[0]

    cache, (tile_v, vec_size, ilp_rows, use_packed_fma) = _state_update_compile_cache(
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        device=k_buf.device,
    )

    h0_view = s.view(pool_size * HV, V, K)

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            run_la_state_update_kernel,
            from_dlpack(h0_view, assumed_align=16),
            from_dlpack(decay_scales, assumed_align=16),
            from_dlpack(h0_indices, assumed_align=16),
            from_dlpack(accepted_len, assumed_align=16),
            from_dlpack(k_buf, assumed_align=16),
            from_dlpack(v_buf, assumed_align=16),
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_packed_fma=use_packed_fma,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_view,
        decay_scales,
        h0_indices,
        accepted_len,
        k_buf,
        v_buf,
        stream,
    )


# ---------------------------------------------------------------------------
# Layer-fused state-update: one launch advances ALL mamba layers in parallel.
# Replaces the per-layer Python loop (28 FFI launches -> 1). Grid gains a layer
# dimension; k_buf/v_buf/h0_source/decay_scales all gain a leading num_layers
# dim and are indexed by i_layer. Pool-indexed k/v semantics, so no
# host-side gather is needed either.
# ---------------------------------------------------------------------------
@cute.kernel
def la_state_update_kernel_fused(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
    h0_indices: cute.Tensor,
    accepted_len: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    num_layers: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = K // vec_size
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp

    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    # 3D grid: (HV * num_v_tiles, B, num_layers) — B is a runtime grid dim.
    block_idx_x, block_idx_y, block_idx_z = cute.arch.block_idx()
    i_v = block_idx_x % num_v_tiles
    i_hv = block_idx_x // num_v_tiles
    i_n = block_idx_y
    i_layer = block_idx_z
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    L = accepted_len[i_n]

    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_h = cute.make_rmem_tensor(cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32)

    if cache_idx >= 0 and L > 0:
        r_decay = cute.exp(-cutlass.Float32(decay_scales[i_layer, i_h]), fastmath=USE_FAST_MATH)
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        num_chunks: cutlass.Constexpr[int] = rows_per_group // ilp_rows
        for chunk in cutlass.range_constexpr(num_chunks):
            v_idx_0 = i_v * tile_v + group_idx * rows_per_group + chunk * ilp_rows
            if v_idx_0 + (ilp_rows - 1) < V:
                for slot in cutlass.range_constexpr(ilp_rows):
                    h_tile = cute.local_tile(
                        h0_source, (1, 1, 1, vec_size), (i_layer, flat_state_idx, v_idx_0 + slot, lane_in_group)
                    )
                    cute.autovec_copy(h_tile, cute.slice_(r_h, (slot, None)))

                for i in cutlass.range(0, L, unroll=0):
                    k_tile = cute.local_tile(k_buf, (1, 1, 1, 1, vec_size), (i_layer, cache_idx, i, i_h, lane_in_group))
                    cute.autovec_copy(k_tile, r_k)
                    for slot in cutlass.range_constexpr(ilp_rows):
                        r_v_s = cutlass.Float32(v_buf[i_layer, cache_idx, i, i_hv, v_idx_0 + slot])
                        for j in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[slot, j], r_h[slot, j + 1] = la_update_pair(
                                r_h[slot, j], r_h[slot, j + 1], r_k[j], r_k[j + 1], r_v_s, r_decay, use_packed_fma
                            )

                for slot in cutlass.range_constexpr(ilp_rows):
                    h_out = cute.local_tile(
                        h0_source, (1, 1, 1, vec_size), (i_layer, flat_state_idx, v_idx_0 + slot, lane_in_group)
                    )
                    cute.autovec_copy(cute.slice_(r_h, (slot, None)), h_out)


@cute.jit
def run_la_state_update_kernel_fused(
    h0_source: cute.Tensor,
    decay_scales: cute.Tensor,
    k_buf: cute.Tensor,
    v_buf: cute.Tensor,
    h0_indices: cute.Tensor,
    accepted_len: cute.Tensor,
    grid_y: Int32,
    num_layers: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = (V + tile_v - 1) // tile_v

    la_state_update_kernel_fused(
        h0_source,
        decay_scales,
        k_buf,
        v_buf,
        h0_indices,
        accepted_len,
        vec_size,
        num_v_tiles,
        tile_v,
        num_layers,
        T,
        H,
        HV,
        K,
        V,
        ilp_rows,
        use_packed_fma,
    ).launch(
        grid=(HV * num_v_tiles, grid_y, num_layers),
        block=[NUM_THREADS_MTP, 1, 1],
        stream=stream,
    )


@functools.cache
def _get_compiled_state_update_kernel_fused(
    num_layers: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    tile_v: int,
    vec_size: int,
    ilp_rows: int,
    use_packed_fma: bool,
):
    return {}


def linear_attention_state_update_kvbuffer_fused(
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    s: torch.Tensor,
    decay_scales: torch.Tensor,
    h0_indices: torch.Tensor,
    accepted_len: torch.Tensor,
    T: int,
) -> None:
    num_layers, pool_size_l, HV, V, K = s.shape
    num_layers_k, pool_size_k, T_k, H, K_k = k_buf.shape
    assert T_k == T, f"k_buf T={T_k} doesn't match T={T}"
    assert K_k == 128, f"K={K_k} != 128"
    if num_layers_k != num_layers:
        raise ValueError(f"k_buf num_layers={num_layers_k} doesn't match state num_layers={num_layers}")
    if k_buf.dtype != torch.float32 or v_buf.dtype != torch.float32:
        raise ValueError(f"k_buf/v_buf must be torch.float32, got {k_buf.dtype}/{v_buf.dtype}")
    if s.dtype != torch.float32:
        raise ValueError(f"s must be torch.float32, got {s.dtype}")
    if decay_scales.dtype != torch.float32:
        raise ValueError(f"decay_scales must be torch.float32, got {decay_scales.dtype}")
    if h0_indices.dtype != torch.int32 or accepted_len.dtype != torch.int32:
        raise ValueError(f"h0_indices/accepted_len must be torch.int32, got {h0_indices.dtype}/{accepted_len.dtype}")
    if pool_size_k != pool_size_l:
        raise ValueError(f"k_buf pool_size={pool_size_k} doesn't match state pool_size={pool_size_l}")
    if k_buf.shape != (num_layers, pool_size_l, T, H, K):
        raise ValueError(f"k_buf shape {tuple(k_buf.shape)} doesn't match expected {(num_layers, pool_size_l, T, H, K)}")
    if v_buf.shape[:4] != (num_layers, pool_size_l, T, HV):
        raise ValueError(f"v_buf shape {tuple(v_buf.shape)} doesn't match expected prefix {(num_layers, pool_size_l, T, HV)}")
    if v_buf.shape[-1] != V:
        raise ValueError(f"v_buf V={v_buf.shape[-1]} doesn't match state V={V}")
    if decay_scales.shape != (num_layers, H):
        raise ValueError(f"decay_scales shape {tuple(decay_scales.shape)} doesn't match expected {(num_layers, H)}")
    B = h0_indices.shape[0]
    if accepted_len.shape[0] != B:
        raise ValueError(f"accepted_len length={accepted_len.shape[0]} doesn't match h0_indices length={B}")

    tile_v, vec_size, ilp_rows = get_mtp_config(B, T, HV, V)
    assert V % ilp_rows == 0, f"V={V} % ilp_rows={ilp_rows} != 0"
    use_packed_fma = get_device_sm_version(k_buf.device)[0] >= 10

    cache = _get_compiled_state_update_kernel_fused(
        num_layers,
        T,
        H,
        HV,
        K,
        V,
        pool_size_l,
        tile_v,
        vec_size,
        ilp_rows,
        use_packed_fma,
    )

    h0_view = s.view(num_layers, pool_size_l * HV, V, K)

    if "compiled" not in cache:
        sym_b = cute.sym_int()
        pool_hv = pool_size_l * HV
        h0_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (num_layers, pool_hv, V, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        decay_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (num_layers, H),
            stride_order=(1, 0),
            assumed_align=16,
        )
        k_buf_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (num_layers, pool_size_l, T, H, K),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        v_buf_fake = make_fake_compact_tensor(
            cutlass.Float32,
            (num_layers, pool_size_l, T, HV, V),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        idx_fake = make_fake_compact_tensor(
            cutlass.Int32,
            (sym_b,),
            stride_order=(0,),
            assumed_align=16,
        )
        acc_fake = make_fake_compact_tensor(
            cutlass.Int32,
            (sym_b,),
            stride_order=(0,),
            assumed_align=16,
        )
        stream_fake = make_fake_stream()

        compiled = cute.compile(
            run_la_state_update_kernel_fused,
            h0_fake,
            decay_fake,
            k_buf_fake,
            v_buf_fake,
            idx_fake,
            acc_fake,
            Int32(1),  # grid_y (dummy B)
            num_layers=num_layers,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            tile_v=tile_v,
            vec_size=vec_size,
            ilp_rows=ilp_rows,
            use_packed_fma=use_packed_fma,
            stream=stream_fake,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    compiled = cache["compiled"]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_view,
        decay_scales,
        k_buf,
        v_buf,
        h0_indices,
        accepted_len,
        Int32(B),
        stream,
    )
