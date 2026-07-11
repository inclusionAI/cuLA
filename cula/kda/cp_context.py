from __future__ import annotations

import functools
import math
import weakref

import torch
from fla.utils import tensor_cache

from cula.utils import get_device_sm_count


@tensor_cache
def _create_cu_seqlens(batch_size: int, num_tokens: int, device_idx: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(batch_size + 1, dtype=dtype, device=f"cuda:{device_idx}") * num_tokens


@functools.lru_cache(maxsize=32)
def _create_full_cu_seqlens_2(T: int, device_idx: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([0, T], dtype=dtype, device=f"cuda:{device_idx}")


_CP_SEQS_CACHE: dict = {}
_SLOT_MAP_CACHE: dict = {}


DOMINANT_LONG_SEQ_MIN_LEN = 32768
DOMINANT_LONG_SEQ_MAX_H = 16


def is_dominant_long_seq(
    seqlens: list[int],
    H: int,
    min_long_len: int = DOMINANT_LONG_SEQ_MIN_LEN,
    max_H: int = DOMINANT_LONG_SEQ_MAX_H,
) -> bool:
    if not seqlens or max_H < H:
        return False
    longest = max(seqlens)
    if longest < min_long_len:
        return False
    return 2 * longest >= sum(seqlens)


def _seqlens_from_cu(cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None) -> list[int]:
    """Return per-seq lengths from a cu_seqlens tensor (CPU sync if no CPU copy)."""
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    cu_list = src.tolist()
    return [cu_list[i + 1] - cu_list[i] for i in range(len(cu_list) - 1)]


def _get_slot_map(cp_cu_seqlens: torch.Tensor, T: int, chunk_size: int) -> torch.Tensor:
    key = (id(cp_cu_seqlens), T, chunk_size)
    hit = _SLOT_MAP_CACHE.get(key)
    if hit is not None:
        weak_ref, cached = hit
        if weak_ref() is cp_cu_seqlens:
            return cached
        # id was reused after the original tensor was collected — recompute.
    num_chunks = (T + chunk_size - 1) // chunk_size
    cp_starts = (cp_cu_seqlens[:-1] // chunk_size).to(torch.int32)
    slot_map = torch.full((num_chunks,), -1, dtype=torch.int32, device=cp_cu_seqlens.device)
    slots = torch.arange(cp_starts.numel(), dtype=torch.int32, device=cp_cu_seqlens.device)
    slot_map[cp_starts.long()] = slots
    _SLOT_MAP_CACHE[key] = (weakref.ref(cp_cu_seqlens), slot_map)
    return slot_map


def _calc_cp_seqs_cached(raw_cu_seqlens, chunk_size, num_v_heads, sm_count, raw_cu_seqlens_cpu=None):
    key = (id(raw_cu_seqlens), chunk_size, num_v_heads, sm_count)
    hit = _CP_SEQS_CACHE.get(key)
    if hit is not None:
        weak_ref, cached = hit
        if weak_ref() is raw_cu_seqlens:
            return cached

    val = _calc_cp_seqs(raw_cu_seqlens, chunk_size, num_v_heads, sm_count, raw_cu_seqlens_cpu=raw_cu_seqlens_cpu)
    _CP_SEQS_CACHE[key] = (weakref.ref(raw_cu_seqlens), val)
    return val


def _calc_cp_seqs(
    raw_cu_seqlens: torch.Tensor,
    chunk_size: int,
    num_v_heads: int,
    sm_count: int,
    raw_cu_seqlens_cpu: torch.Tensor | None = None,
) -> tuple[bool, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Decide whether intra-card CP pays off and, if so, build the split tables."""
    device = raw_cu_seqlens.device
    seqlen_dtype = raw_cu_seqlens.dtype

    raw_cu_seqlens_list = (raw_cu_seqlens_cpu if raw_cu_seqlens_cpu is not None else raw_cu_seqlens).tolist()
    raw_batch_size = len(raw_cu_seqlens_list) - 1
    seqlens = [raw_cu_seqlens_list[i + 1] - raw_cu_seqlens_list[i] for i in range(raw_batch_size)]
    num_chunks = [(s + chunk_size - 1) // chunk_size for s in seqlens]
    if max(num_chunks) <= 0:
        return False, None, None, None, None

    H = num_v_heads
    V_BLOCKS = 1  # bump to 2 once main kernel supports V-blocking
    target_cp_batch = max(1, sm_count // (H * V_BLOCKS))
    total_chunks = sum(num_chunks)
    # mlc * cp_batch * chunk_size ≈ T per raw seq → mlc = total_chunks / cp_batch.
    target_mlc = max(1, total_chunks // (max(1, target_cp_batch)))
    # Snap to nearest power of 2; clamp to ≥ 4 to keep multi-stage pipelining alive.
    max_local_chunks = 2 ** round(math.log2(max(target_mlc, 1.0)))
    max_local_chunks = max(max_local_chunks, 4)
    max_local_tokens = max_local_chunks * chunk_size

    cp_cu_seqlens: list[int] = []
    ht_mask: list[bool] = []
    seq_map_c2r: list[int] = []
    seq_map_r2c: list[int] = [0]

    for i, c in enumerate(num_chunks):
        s = raw_cu_seqlens_list[i]
        e = raw_cu_seqlens_list[i + 1]
        if c > max_local_chunks:
            cut = s
            while True:
                cp_cu_seqlens.append(cut)
                ht_mask.append(False)
                seq_map_c2r.append(i)
                remaining = e - cut
                if remaining <= max_local_tokens + chunk_size:
                    break
                cut += max_local_tokens
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(s)
            ht_mask.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu_seqlens_list[-1])

    Be = total_chunks / max(num_chunks)
    use_cp = (Be * H <= 40) or (Be * H <= 56 and max(num_chunks) >= 128)
    # Additional cuLA-specific guard: never bother if there is only one CP-chunk
    # (no split happened).
    if len(cp_cu_seqlens) - 1 == raw_batch_size:
        use_cp = False

    if use_cp and raw_batch_size == 1:
        T_max = max(seqlens)
        if H <= 8 or H <= 16:
            if T_max < 4096:
                use_cp = False
        elif H <= 32:
            if T_max < 16384:
                use_cp = False
        else:  # H >= 64
            use_cp = False
    elif use_cp and raw_batch_size > 1:
        T_packed = sum(seqlens)
        native_grid = raw_batch_size * H
        dominant_exception = is_dominant_long_seq(seqlens, H)

        unaligned = any(s % chunk_size != 0 for s in seqlens[:-1]) or (raw_cu_seqlens_list[-1] % chunk_size != 0)
        if unaligned:
            use_cp = False
        elif native_grid > 16 and not dominant_exception:
            # Native grid already big enough that CP's lift is marginal.
            use_cp = False
        elif T_packed * H <= 32768:
            use_cp = False
        elif H <= 8:
            if T_packed < 8192:
                use_cp = False
        elif H <= 16:
            if T_packed < 4096:
                use_cp = False
        else:  # H >= 32 with native_grid <= 16 → only B=1 fits this, handled above
            use_cp = False

    if not use_cp:
        return False, None, None, None, None

    cp_cu_seqlens_t = torch.tensor(cp_cu_seqlens, dtype=seqlen_dtype, device=device)
    seq_map_c2r_t = torch.tensor(seq_map_c2r, dtype=seqlen_dtype, device=device)
    seq_map_r2c_t = torch.tensor(seq_map_r2c, dtype=seqlen_dtype, device=device)
    ht_mask_t = torch.tensor(ht_mask, dtype=torch.bool, device=device)
    return True, cp_cu_seqlens_t, seq_map_r2c_t, seq_map_c2r_t, ht_mask_t


def _build_raw_seq_idx(
    cp_cu_seqlens: torch.Tensor, seq_map_c2r: torch.Tensor, T: int, chunk_size: int
) -> tuple[torch.Tensor, list[int]]:
    """Per-chunk raw seq id: raw_seq_idx[i_t] = raw_seq containing chunk i_t."""
    NT = (T + chunk_size - 1) // chunk_size
    out = torch.empty(NT, dtype=torch.int32, device=cp_cu_seqlens.device)
    # cp_cu_seqlens[:-1] // chunk_size gives the chunk-index start of each CP-chunk.
    cp_starts = (cp_cu_seqlens[:-1] // chunk_size).to(torch.int64)
    cp_ends = ((cp_cu_seqlens[1:] + chunk_size - 1) // chunk_size).to(torch.int64)

    c2r_cpu = seq_map_c2r.tolist()
    starts = cp_starts.tolist()
    ends = cp_ends.tolist()
    out_cpu = [0] * NT
    for i in range(len(starts)):
        out[starts[i] : ends[i]] = c2r_cpu[i]
        for j in range(starts[i], ends[i]):
            out_cpu[j] = c2r_cpu[i]
    return out, out_cpu


_RAW_SEQ_IDX_CACHE: dict = {}


def _get_raw_seq_idx(
    cp_cu_seqlens: torch.Tensor, seq_map_c2r: torch.Tensor, T: int, chunk_size: int
) -> tuple[torch.Tensor, list[int]]:
    """weakref-guarded cache for the per-chunk raw_seq_idx tensor."""
    key = (id(cp_cu_seqlens), T, chunk_size)
    hit = _RAW_SEQ_IDX_CACHE.get(key)
    if hit is not None:
        weak_ref, cached = hit
        if weak_ref() is cp_cu_seqlens:
            return cached
    val = _build_raw_seq_idx(cp_cu_seqlens, seq_map_c2r, T, chunk_size)
    _RAW_SEQ_IDX_CACHE[key] = (weakref.ref(cp_cu_seqlens), val)
    return val


def _compute_cp_h0_via_fla_h(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    raw_h0: torch.Tensor | None,
    cp_cu_seqlens: torch.Tensor,
    seq_map_c2r: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute cp_h0 directly via FLA's per-chunk h tensor — exact, no mt needed."""
    from cula.kda.cp_h_boundary import kda_cp_h0_boundary
    from cula.kda.wy_intra import kda_intra_native

    cp_batch = cp_cu_seqlens.size(0) - 1
    raw_batch = int(seq_map_c2r.max().item()) + 1 if seq_map_c2r.numel() > 0 else 1
    T = k.size(1)

    if T > CP_PREPROCESS_TILE_TOKENS:
        return _compute_cp_h0_via_fla_h_tiled(
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            beta=beta,
            raw_h0=raw_h0,
            cp_cu_seqlens=cp_cu_seqlens,
            seq_map_c2r=seq_map_c2r,
            chunk_size=chunk_size,
        )

    w, u, _, kg = kda_intra_native(
        k=k,
        v=v,
        gk=g_cumsum,
        beta=beta,
        chunk_size=chunk_size,
    )

    slot_map = _get_slot_map(cp_cu_seqlens, T, chunk_size)

    if raw_batch > 1:
        if raw_h0 is None:
            H_v = v.size(2)
            K = k.size(3)
            V = v.size(3)
            raw_h0 = torch.zeros(raw_batch, H_v, V, K, dtype=torch.float32, device=k.device)
        h0_chunk0 = raw_h0[0:1]  # state at chunk 0 (always raw seq 0)
        raw_seq_idx, _ = _get_raw_seq_idx(cp_cu_seqlens, seq_map_c2r, T, chunk_size)
    else:
        h0_chunk0 = raw_h0
        raw_seq_idx = None

    cp_h0 = kda_cp_h0_boundary(
        kg=kg,
        w=w,
        u=u,
        g_cumsum=g_cumsum,
        h0=h0_chunk0,
        slot_map=slot_map,
        num_cp=cp_batch,
        chunk_size=chunk_size,
        raw_h0_dense=raw_h0 if raw_batch > 1 else None,
        raw_seq_idx=raw_seq_idx,
    )
    del w, u, kg
    return cp_h0


CP_PREPROCESS_TILE_TOKENS = 16384


def _compute_cp_h0_via_fla_h_tiled(
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    raw_h0: torch.Tensor | None,
    cp_cu_seqlens: torch.Tensor,
    seq_map_c2r: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    from cula.kda.cp_h_boundary import kda_cp_h0_boundary
    from cula.kda.wy_intra import kda_intra_native

    assert k.size(0) == 1, "tiled CP preprocess assumes packed [1, T, H, K] layout"
    cp_batch = cp_cu_seqlens.size(0) - 1
    raw_batch = int(seq_map_c2r.max().item()) + 1 if seq_map_c2r.numel() > 0 else 1
    T = k.size(1)
    H = k.size(2)
    K = k.size(3)
    V = v.size(3)

    if raw_h0 is None:
        raw_h0 = torch.zeros(raw_batch, H, V, K, dtype=torch.float32, device=k.device)
    device = k.device

    assert CP_PREPROCESS_TILE_TOKENS % chunk_size == 0
    tile_tokens = CP_PREPROCESS_TILE_TOKENS

    global_slot_map = _get_slot_map(cp_cu_seqlens, T, chunk_size)
    # Per-chunk raw-seq map (length NT_total). For raw_batch==1 this is all
    # zeros and we don't even pass it to the kernel.
    if raw_batch > 1:
        global_raw_seq_idx, global_raw_seq_idx_cpu = _get_raw_seq_idx(cp_cu_seqlens, seq_map_c2r, T, chunk_size)
    else:
        global_raw_seq_idx = None
        global_raw_seq_idx_cpu = None

    cp_h0 = torch.empty(cp_batch, H, V, K, dtype=torch.float32, device=device)

    exit_state = torch.empty(H, V, K, dtype=torch.float32, device=device)

    BT = chunk_size
    BC = 16
    max_tile_T = min(tile_tokens, T)
    buf_w = torch.empty(1, max_tile_T, H, K, dtype=k.dtype, device=device)
    buf_u = torch.empty(1, max_tile_T, H, V, dtype=v.dtype, device=device)
    buf_kg = torch.empty(1, max_tile_T, H, K, dtype=k.dtype, device=device)
    buf_Akkd = torch.empty(1, max_tile_T, H, BC, dtype=torch.float32, device=device)
    buf_Akk = torch.empty(1, max_tile_T, H, BT, dtype=k.dtype, device=device)

    n_tiles = (T + tile_tokens - 1) // tile_tokens
    for tile_idx in range(n_tiles):
        s = tile_idx * tile_tokens
        e = min(s + tile_tokens, T)
        is_last_tile = tile_idx == n_tiles - 1
        s_chunk = s // chunk_size
        e_chunk = (e + chunk_size - 1) // chunk_size

        k_t = k[:, s:e]
        v_t = v[:, s:e]
        g_t = g_cumsum[:, s:e]
        beta_t = beta[:, s:e]

        w_t, u_t, _, kg_t = kda_intra_native(
            k=k_t,
            v=v_t,
            gk=g_t,
            beta=beta_t,
            chunk_size=chunk_size,
            out_w=buf_w,
            out_u=buf_u,
            out_kg=buf_kg,
            out_Akkd=buf_Akkd,
            out_Akk=buf_Akk,
        )

        slot_map_t = global_slot_map[s_chunk:e_chunk]

        if raw_batch > 1:
            first_raw_in_tile = global_raw_seq_idx_cpu[s_chunk]
            if tile_idx == 0:
                h0_in = raw_h0[first_raw_in_tile : first_raw_in_tile + 1]
            else:
                prev_tile_last_raw = global_raw_seq_idx_cpu[s_chunk - 1]
                if first_raw_in_tile == prev_tile_last_raw:
                    h0_in = exit_state
                else:
                    h0_in = raw_h0[first_raw_in_tile : first_raw_in_tile + 1]
            raw_seq_idx_t = global_raw_seq_idx[s_chunk:e_chunk]
        else:
            h0_in = raw_h0 if tile_idx == 0 else exit_state
            raw_seq_idx_t = None

        kda_cp_h0_boundary(
            kg=kg_t,
            w=w_t,
            u=u_t,
            g_cumsum=g_t,
            h0=h0_in,
            slot_map=slot_map_t,
            num_cp=cp_batch,
            chunk_size=chunk_size,
            cp_h0_out=cp_h0,
            # Skip writing exit_state on the last tile — no consumer.
            exit_state=None if is_last_tile else exit_state,
            raw_h0_dense=raw_h0 if raw_batch > 1 else None,
            raw_seq_idx=raw_seq_idx_t,
        )

    return cp_h0


def intra_card_cp_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    raw_h0: torch.Tensor | None,
    raw_cu_seqlens: torch.Tensor | None,
    chunk_size: int = 64,
    raw_cu_seqlens_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    assert k.dim() == 4 and k.size(0) == 1, "expected packed [1, T, H, K]"
    num_v_heads = v.size(2)
    sm_count = get_device_sm_count(k.device)

    if raw_cu_seqlens is None:
        raw_cu_seqlens = _create_cu_seqlens(1, k.size(1), k.device.index, torch.int32)

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask = _calc_cp_seqs_cached(
        raw_cu_seqlens,
        chunk_size,
        num_v_heads,
        sm_count,
        raw_cu_seqlens_cpu=raw_cu_seqlens_cpu,
    )
    if not use_cp:
        return None, None, None, None

    cp_h0 = _compute_cp_h0_via_fla_h(
        q=q,
        k=k,
        v=v,
        g_cumsum=g,
        beta=beta,
        scale=scale,
        raw_h0=raw_h0,
        cp_cu_seqlens=cp_cu_seqlens,
        seq_map_c2r=seq_map_c2r,
        chunk_size=chunk_size,
    )

    return cp_h0, cp_cu_seqlens, seq_map_c2r, raw_cu_seqlens
