"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for KDA Decode.

This version uses the production / FLA-compatible VK state layout:
    state.shape == (pool_size, HV, V, K)

The kernel still computes on a logical (K, V) matrix in shared memory. Global
state loads/stores therefore explicitly map:
    global(V, K) <-> shared(K, V)

Notes:
- This is a correctness-first implementation for decode.
- It keeps the original small-batch / large-batch split.
- It preserves the previous PAD semantics: if pool_idx < 0 the block does not
  load / update / write output or state, consistent with the earlier CuTe path.
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Runtime behavior
# -----------------------------------------------------------------------------
# The dominant setup cost in this decode path is kernel compilation. We keep a
# lightweight compile cache so each shape/configuration is compiled once, then
# launched repeatedly. Runtime dispatch uses TVM-FFI so torch tensors can be
# passed directly without rebuilding CuTe wrappers on every call.
_compiled_kernels: dict[tuple, object] = {}
_cu_seqlens_cache: dict[tuple, torch.Tensor] = {}
_stream_cache: dict[tuple, cuda.CUstream] = {}

# -----------------------------------------------------------------------------
# Kernel tuning constants
# -----------------------------------------------------------------------------
# The current decode path is tuned around K=128, which is the primary target
# workload in this project. Small-batch and large-batch paths use different
# CTA organizations to balance launch overhead and throughput.
TILE_K = 128
TILE_V = 32
TILE_V_PADDED = 36
TILE_V_SMALL = 16
TILE_V_SMALL_PADDED = 20
# Decode does not currently overlap state prefetch with compute, so a single
# shared-memory stage avoids unnecessary SMEM pressure and improves occupancy.
NUM_STAGES = 1
NUM_THREADS = 128
# One CTA per state is still the best default for larger decode batches, but
# very small N*H cases underfill the GPU. For those micro-batches we compile a
# dedicated split-state variant that launches multiple CTAs per state.
NUM_BLOCKS_PER_STATE_SMALL = 1
MAX_NUM_BLOCKS_PER_STATE_SMALL = 8
N4_NUM_BLOCKS_PER_STATE_SMALL = 4
NUM_THREADS_LARGE = 256
NUM_WARPS_LARGE = 8
V_PER_WARP = 4
ROWS_PER_ITER = 8
NUM_K_ITERS = TILE_K // ROWS_PER_ITER
SMALL_BATCH_THRESHOLD = 1024
MICRO_BATCH_NH_THRESHOLD = 512
DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD = 8
N4_DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD = 64
DENSE_SMALL_HV_PARALLEL_MAX_N = 16


def _get_cached_cute_tensor(tensor: torch.Tensor, *, leading_dim: int, assumed_align: int = 16):
    """Wrap a torch.Tensor as a CuTe tensor for compilation-time use."""
    return from_dlpack(tensor.detach(), assumed_align=assumed_align).mark_layout_dynamic(leading_dim=leading_dim)


def _get_cached_stream(device: torch.device):
    """Convert the active torch stream to a cached cuda.bindings CUstream."""
    stream_id = int(torch.cuda.current_stream(device=device).cuda_stream)
    cache_key = (str(device), stream_id)
    if cache_key not in _stream_cache:
        _stream_cache[cache_key] = cuda.CUstream(stream_id)
    return _stream_cache[cache_key]


def _get_cached_dispatch_bundle(
    cu_seqlens: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h0_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    o: torch.Tensor,
):
    """Build the full set of CuTe tensor handles required for one kernel launch.

    This helper is now only needed for compile-time preparation. Runtime launch
    uses TVM-FFI and passes torch tensors directly.
    """
    return (
        _get_cached_cute_tensor(cu_seqlens, leading_dim=0),
        _get_cached_cute_tensor(q, leading_dim=q.ndim - 1),
        _get_cached_cute_tensor(k, leading_dim=k.ndim - 1),
        _get_cached_cute_tensor(v, leading_dim=v.ndim - 1),
        _get_cached_cute_tensor(a, leading_dim=a.ndim - 1),
        _get_cached_cute_tensor(b, leading_dim=b.ndim - 1),
        _get_cached_cute_tensor(A_log, leading_dim=0),
        _get_cached_cute_tensor(dt_bias, leading_dim=dt_bias.ndim - 1),
        _get_cached_cute_tensor(h0_source, leading_dim=h0_source.ndim - 1),
        _get_cached_cute_tensor(initial_state_indices, leading_dim=0),
        _get_cached_cute_tensor(o, leading_dim=o.ndim - 1),
    )

def _select_small_blocks_per_state(N: int, H: int, HV: int, V: int) -> int:
    del HV
    num_v_tiles_small = V // TILE_V_SMALL
    if N <= 4:
        # For N=4, the path is launch-overhead dominated. Splitting all the way
        # to 8 CTAs per state over-fragments the work and hurts latency.
        return min(N4_NUM_BLOCKS_PER_STATE_SMALL, num_v_tiles_small)
    if N * H <= MICRO_BATCH_NH_THRESHOLD:
        return min(4, num_v_tiles_small)
    return NUM_BLOCKS_PER_STATE_SMALL


def _try_fast_dense_decode(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float | None,
    use_qk_l2norm_in_kernel: bool,
    softplus_beta: float,
    softplus_threshold: float,
    out: torch.Tensor | None,
    state_layout: str | None,
):
    """Fast path for the common dense decode case used by the benchmark.

    This bypasses the broader compatibility/normalization logic when inputs are
    already in the exact kernel-ready layout and dtype, which materially lowers
    Python-side overhead for tiny N.
    """
    if initial_state_source is None or initial_state_indices is None:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None

    B_q, T_q, H, K = q.shape
    if T_q != 1 or k.shape != q.shape or v.shape[0] != B_q or v.shape[1] != 1:
        return None

    N = initial_state_indices.shape[0]
    HV = v.shape[2]
    V = v.shape[3]
    if B_q != N or K != TILE_K or V % TILE_V_SMALL != 0 or V % TILE_V != 0:
        return None

    if (
        q.device.type != "cuda"
        or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or initial_state_source.dtype != torch.float32
        or initial_state_indices.dtype != torch.int32
        or A_log.dtype != torch.float32
        or dt_bias.dtype != torch.float32
    ):
        return None

    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and initial_state_source.is_contiguous()
        and initial_state_indices.is_contiguous()
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
    ):
        return None

    if A_log.numel() != HV or dt_bias.shape != (HV, K):
        return None

    normalized_layout = "vk" if state_layout is None else str(state_layout).strip().lower()
    if normalized_layout == "vk":
        if initial_state_source.ndim != 4 or initial_state_source.shape[1:] != (HV, V, K):
            return None
        state_layout_is_kv = False
    elif normalized_layout == "kv":
        if initial_state_source.ndim != 4 or initial_state_source.shape[1:] != (HV, K, V):
            return None
        state_layout_is_kv = True
    else:
        return None

    if not a.is_contiguous() or a.device != q.device or a.dtype != torch.bfloat16:
        return None
    if a.dim() == 4 and a.shape == (N, 1, HV, K):
        a_kernel = a
    elif a.dim() == 3 and a.shape == (N, 1, HV * K):
        a_kernel = a.view(N, 1, HV, K)
    elif a.dim() == 3 and a.shape == (N, HV, K):
        a_kernel = a.unsqueeze(1)
    else:
        return None

    if b.device != q.device or b.dtype != torch.bfloat16 or not b.is_contiguous():
        return None
    if b.dim() == 3 and b.shape == (N, 1, HV):
        b_kernel = b
    elif b.dim() == 2 and b.shape == (N, HV):
        b_kernel = b.unsqueeze(1)
    else:
        return None

    if scale is None:
        scale = K**-0.5
    elif scale <= 0:
        return None

    o = _prepare_output_tensor(q, out, (N, 1, HV, V))

    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32 or cu_seqlens.numel() != N + 1:
            return None
        cu_seqlens_to_use = cu_seqlens.contiguous()
    else:
        cache_key = (N, str(q.device))
        if cache_key not in _cu_seqlens_cache:
            _cu_seqlens_cache[cache_key] = torch.arange(N + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_to_use = _cu_seqlens_cache[cache_key]

    use_small_batch = N < SMALL_BATCH_THRESHOLD
    dense_small_hv_parallel_head_threshold = (
        N4_DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD if N <= 4 else DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD
    )
    dense_small_hv_parallel = (
        use_small_batch and H <= dense_small_hv_parallel_head_threshold and N <= DENSE_SMALL_HV_PARALLEL_MAX_N
    )
    num_blocks_per_state_small = _select_small_blocks_per_state(N, H, HV, V)

    compiled_kernel = _get_compiled_kernel(
        N,
        H,
        HV,
        K,
        V,
        initial_state_source.shape[0],
        use_small_batch,
        False,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        state_layout_is_kv=state_layout_is_kv,
        precomputed_decay_beta=False,
        num_blocks_per_state_small=num_blocks_per_state_small,
        dense_small_hv_parallel=dense_small_hv_parallel,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    compiled_kernel(
        cu_seqlens_to_use,
        q,
        k,
        v,
        a_kernel,
        b_kernel,
        A_log,
        dt_bias,
        initial_state_source,
        initial_state_indices,
        o,
        _get_cached_stream(q.device),
    )
    return o


def _define_kernels():
    """Define CuTe DSL kernels for KDA normal and varlen decode modes."""

    NUM_WARPS_SMALL = 4
    V_PER_WARP_SMALL = TILE_V_SMALL // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K // ROWS_PER_ITER_SMALL

    @cute.kernel
    def kda_kernel_small_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        precomputed_decay_beta: cutlass.Constexpr[bool],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
    ):
        """Small-batch dense KDA kernel for q/k/v shaped as (N, 1, ...).

        High-level flow:
        1. Each CTA handles one (token, value-head) pair across several V tiles.
        2. q, k, and the gating decay term g are staged into shared memory.
        3. Optional q/k L2 normalization is computed at block scope.
        4. Each V tile runs one delta-rule update and writes back state/output.
        """
        del tiled_copy_load
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        batch_idx = block_idx // num_blocks_per_state_small
        batch_inner = block_idx % num_blocks_per_state_small
        num_v_tiles_per_block = num_v_tiles // num_blocks_per_state_small
        start_v_tile = batch_inner * num_v_tiles_per_block

        num_value_heads_per_q = HV // H
        i_n = 0
        i_h = 0
        i_hv_base = 0
        num_hv_iters = 1
        if dense_small_hv_parallel:
            i_n = batch_idx // HV
            i_hv_base = batch_idx % HV
            i_h = i_hv_base // num_value_heads_per_q
            num_hv_iters = 1
        else:
            i_n = batch_idx // H
            i_h = batch_idx % H
            i_hv_base = i_h * num_value_heads_per_q
            num_hv_iters = num_value_heads_per_q

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP_SMALL
            v_local = in_warp_tid % V_PER_WARP_SMALL
            v_base = warp_idx * V_PER_WARP_SMALL
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_gk_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)
            sGK = smem.allocate_tensor(cutlass.Float32, smem_gk_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])
            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if warp_idx == 0:
                    for norm_iter in range(4):
                        norm_idx = in_warp_tid + norm_iter * 32
                        q_val = sQ[norm_idx]
                        k_val = sK[norm_idx]
                        sum_q_partial += q_val * q_val
                        sum_k_partial += k_val * k_val

                    for offset in [16, 8, 4, 2, 1]:
                        sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                        sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)

                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(sum_q_partial + 1e-6)
                        smem_o[1] = cute.rsqrt(sum_k_partial + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            for hv_offset in range(num_hv_iters):
                i_hv = i_hv_base + hv_offset

                if precomputed_decay_beta:
                    if tidx < TILE_K:
                        sG[tidx] = cutlass.Float32(a[i_n, 0, i_hv, tidx])
                else:
                    r_exp_A = 0.0
                    if in_warp_tid == 0:
                        r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
                    r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
                    if tidx < TILE_K:
                        r_a_k = cutlass.Float32(a[i_n, 0, i_hv, tidx])
                        r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                        x = r_a_k + r_dt_bias_k
                        beta_x = softplus_beta * x
                        softplus_x = 0.0
                        if beta_x <= softplus_threshold:
                            exp_beta_x = cute.exp(beta_x)
                            log_input = cutlass.Float32(1.0 + exp_beta_x)
                            log_result = cutlass.Float32(cute.log(log_input))
                            softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                        else:
                            softplus_x = x
                        sG[tidx] = cute.exp(-r_exp_A * softplus_x)

                r_beta = 0.0
                if in_warp_tid == 0:
                    r_b = cutlass.Float32(b[i_n, 0, i_hv])
                    if precomputed_decay_beta:
                        r_beta = r_b
                    else:
                        r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_beta = cute.arch.shuffle_sync(r_beta, 0)

                if tidx < TILE_K:
                    sGK[tidx] = sG[tidx] * sK[tidx]
                cute.arch.barrier()

                kv_v_load = 0
                kv_k_load_base = 0
                kv_k_load_step = 0
                vk_k_load = 0
                vk_v_load_base = 0
                vk_v_load_step = 0
                if state_layout_is_kv:
                    kv_v_load = tidx % TILE_V_SMALL
                    kv_k_load_base = tidx // TILE_V_SMALL
                    kv_k_load_step = NUM_THREADS // TILE_V_SMALL
                else:
                    vk_k_load = tidx % TILE_K
                    vk_v_load_base = tidx // TILE_K
                    vk_v_load_step = NUM_THREADS // TILE_K

                for v_tile_offset in range(num_v_tiles_per_block):
                    stage = v_tile_offset % NUM_STAGES
                    v_tile = start_v_tile + v_tile_offset
                    v_global_base = v_tile * TILE_V_SMALL

                    for k_iter in range(NUM_K_ITERS_SMALL):
                        k_load = 0
                        v_load = 0
                        if state_layout_is_kv:
                            k_load = kv_k_load_base + k_iter * kv_k_load_step
                            v_load = kv_v_load
                        else:
                            k_load = vk_k_load
                            v_load = vk_v_load_base + k_iter * vk_v_load_step
                        v_global_load = v_global_base + v_load
                        h_val = 0.0
                        if v_global_load < v.shape[3]:
                            if state_layout_is_kv:
                                h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                            else:
                                h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                        sData[(k_load, v_load, stage)] = h_val

                    cute.arch.barrier()

                    v_global = v_tile * TILE_V_SMALL + v_idx
                    r_v = 0.0
                    if v_global < v.shape[3]:
                        r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

                    sum_hk = 0.0
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        k_base = k_iter * ROWS_PER_ITER_SMALL
                        k_idx = k_base + k_local
                        sum_hk += sData[(k_idx, v_idx, stage)] * sGK[k_idx]

                    for offset in [4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk,
                            offset=offset * V_PER_WARP_SMALL,
                            mask=-1,
                            mask_and_clamp=31,
                        )

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = cute.arch.shuffle_sync(v_new, v_local)

                    sum_hq = 0.0
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        k_base = k_iter * ROWS_PER_ITER_SMALL
                        k_idx = k_base + k_local
                        h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                        h_new = h_old + sK[k_idx] * v_new
                        sData[(k_idx, v_idx, stage)] = h_new
                        sum_hq += h_new * sQ[k_idx]

                    for offset in [4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq,
                            offset=offset * V_PER_WARP_SMALL,
                            mask=-1,
                            mask_and_clamp=31,
                        )

                    if k_local == 0 and v_global < v.shape[3]:
                        o[(i_n, 0, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                    cute.arch.barrier()

                    for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                        k_write = 0
                        v_write = 0
                        if state_layout_is_kv:
                            k_write = kv_k_load_base + k_iter * kv_k_load_step
                            v_write = kv_v_load
                        else:
                            k_write = vk_k_load
                            v_write = vk_v_load_base + k_iter * vk_v_load_step
                        v_global_write = v_global_base + v_write
                        if v_global_write < v.shape[3]:
                            if state_layout_is_kv:
                                h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                            else:
                                h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

                    cute.arch.barrier()

    @cute.kernel
    def kda_kernel_small_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        precomputed_decay_beta: cutlass.Constexpr[bool],
    ):
        """Small batch KDA kernel for varlen decode: q/k/v shapes (1, N, ...)."""
        del tiled_copy_load
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        batch_idx = block_idx // num_blocks_per_state_small
        batch_inner = block_idx % num_blocks_per_state_small
        num_v_tiles_per_block = num_v_tiles // num_blocks_per_state_small
        start_v_tile = batch_inner * num_v_tiles_per_block

        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP_SMALL
            v_local = in_warp_tid % V_PER_WARP_SMALL
            v_base = warp_idx * V_PER_WARP_SMALL
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_gk_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)
            sGK = smem.allocate_tensor(cutlass.Float32, smem_gk_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            if precomputed_decay_beta:
                if tidx < TILE_K:
                    sG[tidx] = cutlass.Float32(a[i_n, i_hv, tidx])
            else:
                r_exp_A = 0.0
                if in_warp_tid == 0:
                    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
                r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
                if tidx < TILE_K:
                    r_a_k = cutlass.Float32(a[i_n, i_hv, tidx])
                    r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                    x = r_a_k + r_dt_bias_k
                    beta_x = softplus_beta * x
                    softplus_x = 0.0
                    if beta_x <= softplus_threshold:
                        exp_beta_x = cute.exp(beta_x)
                        log_input = cutlass.Float32(1.0 + exp_beta_x)
                        log_result = cutlass.Float32(cute.log(log_input))
                        softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                    else:
                        softplus_x = x
                    sG[tidx] = cute.exp(-r_exp_A * softplus_x)

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, i_hv])
                if precomputed_decay_beta:
                    r_beta = r_b
                else:
                    r_beta = 1.0 / (1.0 + cute.exp(-r_b))
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if warp_idx == 0:
                    for norm_iter in range(4):
                        norm_idx = in_warp_tid + norm_iter * 32
                        q_val = sQ[norm_idx]
                        k_val = sK[norm_idx]
                        sum_q_partial += q_val * q_val
                        sum_k_partial += k_val * k_val

                    for offset in [16, 8, 4, 2, 1]:
                        sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                        sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)

                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(sum_q_partial + 1e-6)
                        smem_o[1] = cute.rsqrt(sum_k_partial + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            if tidx < TILE_K:
                sGK[tidx] = sG[tidx] * sK[tidx]
            cute.arch.barrier()

            kv_v_load = 0
            kv_k_load_base = 0
            kv_k_load_step = 0
            vk_k_load = 0
            vk_v_load_base = 0
            vk_v_load_step = 0
            if state_layout_is_kv:
                kv_v_load = tidx % TILE_V_SMALL
                kv_k_load_base = tidx // TILE_V_SMALL
                kv_k_load_step = NUM_THREADS // TILE_V_SMALL
            else:
                vk_k_load = tidx % TILE_K
                vk_v_load_base = tidx // TILE_K
                vk_v_load_step = NUM_THREADS // TILE_K

            for v_tile_offset in range(num_v_tiles_per_block):
                stage = v_tile_offset % NUM_STAGES
                v_tile = start_v_tile + v_tile_offset
                v_global_base = v_tile * TILE_V_SMALL

                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_load = 0
                    v_load = 0
                    if state_layout_is_kv:
                        k_load = kv_k_load_base + k_iter * kv_k_load_step
                        v_load = kv_v_load
                    else:
                        k_load = vk_k_load
                        v_load = vk_v_load_base + k_iter * vk_v_load_step
                    v_global_load = v_global_base + v_load
                    h_val = 0.0
                    if v_global_load < v.shape[3]:
                        if state_layout_is_kv:
                            h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                        else:
                            h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                    sData[(k_load, v_load, stage)] = h_val

                cute.arch.barrier()

                v_global = v_tile * TILE_V_SMALL + v_idx
                r_v = 0.0
                if v_global < v.shape[3]:
                    r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    sum_hk += sData[(k_idx, v_idx, stage)] * sGK[k_idx]

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_base = k_iter * ROWS_PER_ITER_SMALL
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                    h_new = h_old + sK[k_idx] * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * sQ[k_idx]

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq,
                        offset=offset * V_PER_WARP_SMALL,
                        mask=-1,
                        mask_and_clamp=31,
                    )

                if k_local == 0 and v_global < v.shape[3]:
                    o[(0, i_n, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                    k_write = 0
                    v_write = 0
                    if state_layout_is_kv:
                        k_write = kv_k_load_base + k_iter * kv_k_load_step
                        v_write = kv_v_load
                    else:
                        k_write = vk_k_load
                        v_write = vk_v_load_base + k_iter * vk_v_load_step
                    v_global_write = v_global_base + v_write
                    if v_global_write < v.shape[3]:
                        if state_layout_is_kv:
                            h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                        else:
                            h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

                cute.arch.barrier()

    @cute.kernel
    def kda_kernel_large_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
    ):
        """Large batch KDA kernel for dense decode: q/k/v shapes (N, 1, ...)."""
        del tiled_copy_load
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()

        i_nhv = batch_idx // num_v_tiles
        v_tile = batch_idx % num_v_tiles
        i_n = i_nhv // HV
        i_hv = i_nhv % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP
            v_local = in_warp_tid % V_PER_WARP
            v_base = warp_idx * V_PER_WARP
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

            r_exp_A = 0.0
            if in_warp_tid == 0:
                r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
            r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
            if tidx < TILE_K:
                r_a_k = cutlass.Float32(a[i_n, 0, i_hv, tidx])
                r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                x = r_a_k + r_dt_bias_k
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                else:
                    softplus_x = x
                sG[tidx] = cute.exp(-r_exp_A * softplus_x)

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, 0, i_hv])
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                    sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 8] = sum_k_partial
                cute.arch.barrier()

                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_LARGE:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 8]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(local_sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                        local_sum_k += cute.arch.shuffle_sync_bfly(local_sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            stage = 0

            for k_iter in range(NUM_K_ITERS):
                flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                k_load = 0
                v_load = 0
                if state_layout_is_kv:
                    k_load = flat_idx // TILE_V
                    v_load = flat_idx % TILE_V
                else:
                    k_load = flat_idx % TILE_K
                    v_load = flat_idx // TILE_K
                v_global_load = v_tile * TILE_V + v_load
                h_val = 0.0
                if v_global_load < v.shape[3]:
                    if state_layout_is_kv:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                    else:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                sData[(k_load, v_load, stage)] = h_val

            cute.arch.barrier()

            v_global = v_tile * TILE_V + v_idx
            r_v = 0.0
            if v_global < v.shape[3]:
                r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

            sum_hk = 0.0
            for k_iter in range(NUM_K_ITERS):
                k_base = k_iter * ROWS_PER_ITER
                k_idx = k_base + k_local
                sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]

            for offset in [4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk,
                    offset=offset * V_PER_WARP,
                    mask=-1,
                    mask_and_clamp=31,
                )

            v_new = (r_v - sum_hk) * r_beta
            v_new = cute.arch.shuffle_sync(v_new, v_local)

            sum_hq = 0.0
            for k_iter in range(NUM_K_ITERS):
                k_base = k_iter * ROWS_PER_ITER
                k_idx = k_base + k_local
                h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                h_new = h_old + sK[k_idx] * v_new
                sData[(k_idx, v_idx, stage)] = h_new
                sum_hq += h_new * sQ[k_idx]

            for offset in [4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq,
                    offset=offset * V_PER_WARP,
                    mask=-1,
                    mask_and_clamp=31,
                )

            if k_local == 0 and v_global < v.shape[3]:
                o[(i_n, 0, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

            cute.arch.barrier()

            for k_iter in cutlass.range(NUM_K_ITERS, unroll=2):
                flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                k_write = 0
                v_write = 0
                if state_layout_is_kv:
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                else:
                    k_write = flat_idx % TILE_K
                    v_write = flat_idx // TILE_K
                v_global_write = v_tile * TILE_V + v_write
                if v_global_write < v.shape[3]:
                    if state_layout_is_kv:
                        h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                    else:
                        h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

    @cute.kernel
    def kda_kernel_large_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
    ):
        """Large batch KDA kernel for varlen decode: q/k/v shapes (1, N, ...)."""
        del tiled_copy_load
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()

        i_nhv = batch_idx // num_v_tiles
        v_tile = batch_idx % num_v_tiles
        i_n = i_nhv // HV
        i_hv = i_nhv % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP
            v_local = in_warp_tid % V_PER_WARP
            v_base = warp_idx * V_PER_WARP
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[0, i_n, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[0, i_n, i_h, tidx])

            r_exp_A = 0.0
            if in_warp_tid == 0:
                r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
            r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
            if tidx < TILE_K:
                r_a_k = cutlass.Float32(a[i_n, i_hv, tidx])
                r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                x = r_a_k + r_dt_bias_k
                beta_x = softplus_beta * x
                softplus_x = 0.0
                if beta_x <= softplus_threshold:
                    exp_beta_x = cute.exp(beta_x)
                    log_input = cutlass.Float32(1.0 + exp_beta_x)
                    log_result = cutlass.Float32(cute.log(log_input))
                    softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                else:
                    softplus_x = x
                sG[tidx] = cute.exp(-r_exp_A * softplus_x)

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, i_hv])
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                    sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)

                if in_warp_tid == 0:
                    smem_o[warp_idx] = sum_q_partial
                    smem_o[warp_idx + 8] = sum_k_partial
                cute.arch.barrier()

                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS_LARGE:
                        local_sum_q = smem_o[in_warp_tid]
                        local_sum_k = smem_o[in_warp_tid + 8]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(local_sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                        local_sum_k += cute.arch.shuffle_sync_bfly(local_sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                    if in_warp_tid == 0:
                        smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                        smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = smem_o[0]
                inv_norm_k = smem_o[1]

                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            stage = 0

            for k_iter in range(NUM_K_ITERS):
                flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                k_load = 0
                v_load = 0
                if state_layout_is_kv:
                    k_load = flat_idx // TILE_V
                    v_load = flat_idx % TILE_V
                else:
                    k_load = flat_idx % TILE_K
                    v_load = flat_idx // TILE_K
                v_global_load = v_tile * TILE_V + v_load
                h_val = 0.0
                if v_global_load < v.shape[3]:
                    if state_layout_is_kv:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                    else:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                sData[(k_load, v_load, stage)] = h_val

            cute.arch.barrier()

            v_global = v_tile * TILE_V + v_idx
            r_v = 0.0
            if v_global < v.shape[3]:
                r_v = cutlass.Float32(v[0, i_n, i_hv, v_global])

            sum_hk = 0.0
            for k_iter in range(NUM_K_ITERS):
                k_base = k_iter * ROWS_PER_ITER
                k_idx = k_base + k_local
                sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]

            for offset in [4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk,
                    offset=offset * V_PER_WARP,
                    mask=-1,
                    mask_and_clamp=31,
                )

            v_new = (r_v - sum_hk) * r_beta
            v_new = cute.arch.shuffle_sync(v_new, v_local)

            sum_hq = 0.0
            for k_iter in range(NUM_K_ITERS):
                k_base = k_iter * ROWS_PER_ITER
                k_idx = k_base + k_local
                h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                h_new = h_old + sK[k_idx] * v_new
                sData[(k_idx, v_idx, stage)] = h_new
                sum_hq += h_new * sQ[k_idx]

            for offset in [4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq,
                    offset=offset * V_PER_WARP,
                    mask=-1,
                    mask_and_clamp=31,
                )

            if k_local == 0 and v_global < v.shape[3]:
                o[(0, i_n, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

            cute.arch.barrier()

            for k_iter in cutlass.range(NUM_K_ITERS, unroll=2):
                flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                k_write = 0
                v_write = 0
                if state_layout_is_kv:
                    k_write = flat_idx // TILE_V
                    v_write = flat_idx % TILE_V
                else:
                    k_write = flat_idx % TILE_K
                    v_write = flat_idx // TILE_K
                v_global_write = v_tile * TILE_V + v_write
                if v_global_write < v.shape[3]:
                    if state_layout_is_kv:
                        h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                    else:
                        h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

    return (
        kda_kernel_small_batch,
        kda_kernel_small_batch_varlen,
        kda_kernel_large_batch,
        kda_kernel_large_batch_varlen,
    )


def _create_jit_functions():
    """Create JIT-compiled launcher functions for all KDA kernel variants."""

    kda_small, kda_small_varlen, kda_large, kda_large_varlen = _define_kernels()

    @cute.jit
    def run_small_batch(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        precomputed_decay_beta: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del cu_seqlens, B, T, K, use_initial_state
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * (HV if dense_small_hv_parallel else H)

        num_v_tiles_small = cute.ceil_div(V, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        smem_bytes_small = 4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES + 4 * TILE_V_SMALL + 4 * TILE_K * 4 + 64

        kda_small(
            None,
            h0_source,
            smem_layout_small,
            num_v_tiles_small,
            num_blocks_per_state_small,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
            state_layout_is_kv,
            precomputed_decay_beta,
            dense_small_hv_parallel,
        ).launch(
            grid=(batch_size * num_blocks_per_state_small, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_small_batch_varlen(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        precomputed_decay_beta: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del cu_seqlens, B, T, K, use_initial_state, dense_small_hv_parallel
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * HV

        num_v_tiles_small = cute.ceil_div(V, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        smem_bytes_small = 4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES + 4 * TILE_V_SMALL + 4 * TILE_K * 4 + 64

        kda_small_varlen(
            None,
            h0_source,
            smem_layout_small,
            num_v_tiles_small,
            num_blocks_per_state_small,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
            state_layout_is_kv,
            precomputed_decay_beta,
        ).launch(
            grid=(batch_size * num_blocks_per_state_small, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_large_batch(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        precomputed_decay_beta: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del cu_seqlens, B, T, K, use_initial_state, precomputed_decay_beta, num_blocks_per_state_small, dense_small_hv_parallel
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * HV

        num_v_tiles = cute.ceil_div(V, TILE_V)
        smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        smem_bytes = 4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 2 + 4 * TILE_K + 64

        kda_large(
            None,
            h0_source,
            smem_layout,
            num_v_tiles,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
            state_layout_is_kv,
        ).launch(
            grid=(batch_size * num_v_tiles, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    @cute.jit
    def run_large_batch_varlen(
        cu_seqlens: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        B: cutlass.Constexpr[int],
        T: cutlass.Constexpr[int],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        precomputed_decay_beta: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del cu_seqlens, B, T, K, use_initial_state, precomputed_decay_beta, num_blocks_per_state_small, dense_small_hv_parallel
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * HV

        num_v_tiles = cute.ceil_div(V, TILE_V)
        smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        smem_bytes = 4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 2 + 4 * TILE_K + 64

        kda_large_varlen(
            None,
            h0_source,
            smem_layout,
            num_v_tiles,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            h0_indices,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            use_qk_l2norm,
            state_layout_is_kv,
        ).launch(
            grid=(batch_size * num_v_tiles, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return (
        run_small_batch,
        run_small_batch_varlen,
        run_large_batch,
        run_large_batch_varlen,
    )


_jit_functions = None


def _get_jit_functions():
    global _jit_functions
    if _jit_functions is None:
        _jit_functions = _create_jit_functions()
    return _jit_functions


def _get_compiled_kernel(
    N,
    H,
    HV,
    K,
    V,
    pool_size,
    use_small_batch,
    is_varlen_decode,
    scale,
    use_qk_l2norm,
    state_layout_is_kv,
    precomputed_decay_beta,
    num_blocks_per_state_small,
    dense_small_hv_parallel,
    softplus_beta,
    softplus_threshold,
):
    """Get or lazily compile one CuteDSL decode kernel variant.

    Compile-time specialization is still important here, so we cache the result
    by shape, layout, and constexpr options. The compiled function is emitted
    with TVM-FFI enabled so runtime calls can pass torch tensors directly.
    """
    global _compiled_kernels

    key = (
        N,
        H,
        HV,
        K,
        V,
        pool_size,
        use_small_batch,
        is_varlen_decode,
        scale,
        use_qk_l2norm,
        state_layout_is_kv,
        precomputed_decay_beta,
        num_blocks_per_state_small,
        dense_small_hv_parallel,
        softplus_beta,
        softplus_threshold,
    )
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device="cuda")

    if is_varlen_decode:
        q = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(1, N, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, HV, K, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(1, N, HV, V, dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(N, 1, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, 1, HV, K, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, 1, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(N, 1, HV, V, dtype=torch.bfloat16, device="cuda")

    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    if state_layout_is_kv:
        h0_source = torch.zeros(pool_size, HV, K, V, dtype=torch.float32, device="cuda")
    else:
        h0_source = torch.zeros(pool_size, HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")

    cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)
    q_tensor = from_dlpack(q, assumed_align=16)
    k_tensor = from_dlpack(k, assumed_align=16)
    v_tensor = from_dlpack(v, assumed_align=16)
    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    A_log_tensor = from_dlpack(A_log, assumed_align=16)
    dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
    h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
    h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
    o_tensor = from_dlpack(o, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    run_small, run_small_varlen, run_large, run_large_varlen = _get_jit_functions()
    if use_small_batch:
        kernel_func = run_small_varlen if is_varlen_decode else run_small
    else:
        kernel_func = run_large_varlen if is_varlen_decode else run_large

    compiled_kernel = cute.compile(
        kernel_func,
        cu_seqlens_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        a_tensor,
        b_tensor,
        A_log_tensor,
        dt_bias_tensor,
        h0_source_tensor,
        h0_indices_tensor,
        o_tensor,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        B=1 if is_varlen_decode else N,
        T=N if is_varlen_decode else 1,
        H=H,
        K=K,
        V=V,
        HV=HV,
        use_initial_state=True,
        use_qk_l2norm=use_qk_l2norm,
        state_layout_is_kv=state_layout_is_kv,
        precomputed_decay_beta=precomputed_decay_beta,
        num_blocks_per_state_small=num_blocks_per_state_small,
        dense_small_hv_parallel=dense_small_hv_parallel,
        stream=stream,
        options="--enable-tvm-ffi --opt-level 1",
    )

    _compiled_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA kernel compiled: "
        f"N={N}, H={H}, HV={HV}, K={K}, V={V}, pool_size={pool_size}, "
        f"small_batch={use_small_batch}, varlen={is_varlen_decode}"
    )
    return compiled_kernel


def _normalize_A_log(A_log: torch.Tensor, HV: int) -> torch.Tensor:
    if A_log.numel() != HV:
        raise ValueError(f"Unexpected A_log shape: {A_log.shape}; expected numel={HV}")
    return A_log.reshape(HV).contiguous()


def _normalize_dt_bias(dt_bias: torch.Tensor, HV: int, K: int) -> torch.Tensor:
    if dt_bias.numel() != HV * K:
        raise ValueError(f"Unexpected dt_bias shape: {dt_bias.shape}; expected numel={HV * K}")
    return dt_bias.reshape(HV, K).contiguous()


def _canonicalize_state_layout(state_layout: str | None) -> str:
    """Accept only the two explicit state layouts used by the kernel.

    Internal meaning:
        - "vk": state shape (..., V, K)
        - "kv": state shape (..., K, V)
    """
    if state_layout is None:
        return "vk"

    normalized = str(state_layout).strip().lower()
    if normalized not in ("vk", "kv"):
        raise ValueError(f"Unsupported state_layout={state_layout}; expected only 'vk' or 'kv'")
    return normalized


def _normalize_kda_a(a, *, is_varlen_decode, N, HV, K):
    """Normalize `a` to match the compile-time shape expected by the kernel.

    Supports both cuLA-native layouts and the public flattened compatibility layouts.

    varlen kernel compiled shape: (N, HV, K)  -- 3D
    dense kernel compiled shape:  (N, 1, HV, K) -- 4D
    """
    if is_varlen_decode:
        # Target: (N, HV, K) -- 3D
        if a.dim() == 2 and a.shape == (N, HV * K):
            return a.view(N, HV, K)
        if a.dim() == 3 and a.shape == (N, HV, K):
            return a
        if a.dim() == 3 and a.shape == (N, 1, HV * K):
            return a.view(N, HV, K)
        if a.dim() == 3 and a.shape == (1, N, HV * K):
            return a.view(N, HV, K)
        if a.dim() == 4 and a.shape == (1, N, HV, K):
            return a.squeeze(0)
        if a.dim() == 4 and a.shape == (1, N, 1, HV * K):
            return a.view(N, HV, K)
        raise ValueError(f"Unexpected a shape for varlen: {a.shape}")
    else:
        # Target: (N, 1, HV, K) -- 4D
        if a.dim() == 2 and a.shape == (N, HV * K):
            return a.view(N, 1, HV, K)
        if a.dim() == 3 and a.shape == (N, HV, K):
            return a.unsqueeze(1)
        if a.dim() == 3 and a.shape == (N, 1, HV * K):
            return a.view(N, 1, HV, K)
        if a.dim() == 4 and a.shape == (N, 1, HV, K):
            return a
        raise ValueError(f"Unexpected a shape for dense: {a.shape}")


def _normalize_state_source(initial_state_source, *, N, HV, K, V, device, state_layout="vk"):
    """Validate that the incoming state already matches the requested layout."""
    if initial_state_source is None:
        if state_layout == "vk":
            h0_source = torch.zeros(N, HV, V, K, dtype=torch.float32, device=device)
            return h0_source, N, False
        h0_source = torch.zeros(N, HV, K, V, dtype=torch.float32, device=device)
        return h0_source, N, True

    if initial_state_source.dim() != 4:
        raise ValueError(f"Unexpected initial_state_source shape: {initial_state_source.shape}; expected a 4D state tensor")

    if initial_state_source.shape[1] != HV:
        raise ValueError(f"Unexpected initial_state_source shape: {initial_state_source.shape}; expected HV={HV}")

    if state_layout == "vk":
        if initial_state_source.shape[2:] != (V, K):
            raise ValueError(
                f"State layout mismatch for state_layout='vk': got {initial_state_source.shape}, expected (..., {HV}, {V}, {K})"
            )
        return initial_state_source, initial_state_source.shape[0], False

    if initial_state_source.shape[2:] != (K, V):
        raise ValueError(
            f"State layout mismatch for state_layout='kv': got {initial_state_source.shape}, expected (..., {HV}, {K}, {V})"
        )
    return initial_state_source, initial_state_source.shape[0], True


def _normalize_state_indices(initial_state_indices, *, N, pool_size, device):
    """Normalize state indices for decode.

    For compatibility callers, missing indices default to a sequential mapping.
    """
    if initial_state_indices is None:
        if pool_size < N:
            raise ValueError(f"initial_state_source only has pool_size={pool_size}, but N={N}")
        return torch.arange(N, device=device, dtype=torch.int32)

    indices = initial_state_indices.to(device=device, dtype=torch.int32)
    if indices.numel() != N:
        raise ValueError(f"Unexpected initial_state_indices shape: {initial_state_indices.shape}; expected numel={N}")
    return indices.contiguous()


def _prepare_output_tensor(q: torch.Tensor, out: torch.Tensor | None, shape: tuple[int, ...]) -> torch.Tensor:
    if out is None:
        return q.new_empty(shape, dtype=torch.bfloat16)
    if out.shape != shape:
        raise ValueError(f"Unexpected out shape: {out.shape}; expected {shape}")
    if out.device != q.device:
        raise ValueError(f"Unexpected out device: {out.device}; expected {q.device}")
    if out.dtype != torch.bfloat16:
        raise ValueError(f"Unexpected out dtype: {out.dtype}; expected torch.bfloat16")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor | None,
    initial_state_indices: torch.Tensor | None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    is_kda: bool = False,
    out: torch.Tensor | None = None,
    state_layout: str = "vk",
):
    """Public cuLA decode API backed by CuTe DSL.

    Supported state layouts:
    - "vk": state shape (pool_size, HV, V, K), default and recommended
    - "kv": state shape (pool_size, HV, K, V)

    The caller is expected to pass a state tensor that already matches the
    selected layout exactly.
    """
    if not is_kda:
        raise NotImplementedError("cuLA fused decode currently supports only is_kda=True mode")

    return kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        out=out,
        state_layout=state_layout,
    )


def kda_decode(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    out: torch.Tensor | None = None,
    state_layout: str = "vk",
) -> torch.Tensor:
    """CuTe DSL implementation of fused sigmoid gating KDA update.

    State layout contract:
        - "vk": (pool_size, HV, V, K), default
        - "kv": (pool_size, HV, K, V)

    Dense decode:
        q/k: (N, 1, H, K)
        v:   (N, 1, HV, V)
        a:   (N, 1, HV, K)
        b:   (N, 1, HV)

    Varlen decode:
        q/k: (1, N, H, K)
        v:   (1, N, HV, V)
        a:   (N, HV, K) or (1, N, HV, K)
        b:   (N, HV) or (1, N, HV)
    """

    B_q, T_q, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    if initial_state_indices is not None:
        N = initial_state_indices.shape[0]
    else:
        N = T_q if B_q == 1 and T_q > 1 else B_q

    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, f"scale must be positive, got {scale}"

    state_layout = _canonicalize_state_layout(state_layout)

    fast_dense_out = _try_fast_dense_decode(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        scale,
        use_qk_l2norm_in_kernel,
        softplus_beta,
        softplus_threshold,
        out,
        state_layout,
    )
    if fast_dense_out is not None:
        return fast_dense_out

    A_log = A_log.contiguous()

    assert K == TILE_K, f"Current CuTe DSL KDA kernel requires K={TILE_K}, got {K}"
    assert V % TILE_V_SMALL == 0, f"Current CuTe DSL KDA kernel requires V % {TILE_V_SMALL} == 0, got V={V}"
    assert V % TILE_V == 0, f"Current CuTe DSL KDA kernel requires V % {TILE_V} == 0, got V={V}"
    num_blocks_per_state_small = _select_small_blocks_per_state(N, H, HV, V)
    assert (V // TILE_V_SMALL) % num_blocks_per_state_small == 0, (
        f"Small-batch KDA kernel requires num_v_tiles_small divisible by {num_blocks_per_state_small}, got V={V}"
    )

    is_varlen_decode = B_q == 1 and T_q == N and N > 1

    # The public API only accepts the two explicit kernel layouts.

    # Small and large batches use different kernel organizations:
    # small batches prioritize lower launch overhead, while large batches focus
    # on sustained throughput.
    use_small_batch = N < SMALL_BATCH_THRESHOLD
    dense_small_hv_parallel_head_threshold = (
        N4_DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD if N <= 4 else DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD
    )
    dense_small_hv_parallel = (
        use_small_batch
        and (not is_varlen_decode)
        and H <= dense_small_hv_parallel_head_threshold
        and N <= DENSE_SMALL_HV_PARALLEL_MAX_N
    )

    # fast_path means the incoming state already matches one of the expected
    # layouts, so we can skip extra normalization work.
    fast_path = False
    state_layout_is_kv = False
    pool_size = N
    h0_source = initial_state_source

    if h0_source is None:
        if state_layout == "vk":
            h0_source = torch.zeros(N, HV, V, K, dtype=torch.float32, device=q.device)
            state_layout_is_kv = False
        else:
            h0_source = torch.zeros(N, HV, K, V, dtype=torch.float32, device=q.device)
            state_layout_is_kv = True
        pool_size = N
        fast_path = True
    elif h0_source.dim() == 4 and h0_source.shape[1] == HV:
        pool_size = h0_source.shape[0]
        if state_layout == "kv":
            if h0_source.shape[2:] != (K, V):
                raise ValueError(f"State layout mismatch for state_layout='kv': got {h0_source.shape}, expected (..., {HV}, {K}, {V})")
            state_layout_is_kv = True
            fast_path = True
        else:
            if h0_source.shape[2:] != (V, K):
                raise ValueError(f"State layout mismatch for state_layout='vk': got {h0_source.shape}, expected (..., {HV}, {V}, {K})")
            fast_path = True

    if fast_path:
        a_fast = _normalize_kda_a(a, is_varlen_decode=is_varlen_decode, N=N, HV=HV, K=K)
        if is_varlen_decode:
            if b.dim() == 3:
                b = b.squeeze(0)
            o = _prepare_output_tensor(q, out, (1, N, HV, V))
        else:
            if b.dim() == 2:
                b = b.unsqueeze(1)
            o = _prepare_output_tensor(q, out, (N, 1, HV, V))
        a = a_fast
        if initial_state_indices is None:
            if pool_size < N:
                fast_path = False
            else:
                initial_state_indices = torch.arange(N, device=q.device, dtype=torch.int32)
        elif (
            initial_state_indices.device != q.device
            or initial_state_indices.dtype != torch.int32
            or initial_state_indices.numel() != N
        ):
            fast_path = False

    if not fast_path:
        h0_source, pool_size, state_layout_is_kv = _normalize_state_source(
            initial_state_source,
            N=N,
            HV=HV,
            K=K,
            V=V,
            device=q.device,
            state_layout=state_layout,
        )

        a = _normalize_kda_a(a, is_varlen_decode=is_varlen_decode, N=N, HV=HV, K=K)

        if is_varlen_decode:
            # varlen b compiled: (N, HV) -- 2D
            if b.dim() == 3:
                b = b.squeeze(0)  # (1, N, HV) -> (N, HV)
            # b should be 2D (N, HV)
            o = _prepare_output_tensor(q, out, (1, N, HV, V))
        else:
            # dense b compiled: (N, 1, HV) -- 3D
            if b.dim() == 2:
                b = b.unsqueeze(1)
            # b should be 3D (N, 1, HV)
            o = _prepare_output_tensor(q, out, (N, 1, HV, V))

    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    a = a if a.is_contiguous() else a.contiguous()
    b = b if b.is_contiguous() else b.contiguous()
    dt_bias = dt_bias if dt_bias.is_contiguous() else dt_bias.contiguous()

    if cu_seqlens is not None:
        cu_seqlens_to_use = cu_seqlens
    else:
        cache_key = (N, str(q.device))
        if cache_key not in _cu_seqlens_cache:
            _cu_seqlens_cache[cache_key] = torch.arange(N + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_to_use = _cu_seqlens_cache[cache_key]

    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)

    precomputed_decay_beta = False
    a_kernel, b_kernel = a, b

    if not fast_path:
        initial_state_indices = _normalize_state_indices(
            initial_state_indices,
            N=N,
            pool_size=pool_size,
            device=q.device,
        )
    if cu_seqlens is not None:
        cu_seqlens_to_use = cu_seqlens.contiguous()

    stream = _get_cached_stream(q.device)

    compiled_kernel = _get_compiled_kernel(
        N,
        H,
        HV,
        K,
        V,
        pool_size,
        use_small_batch,
        is_varlen_decode,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        state_layout_is_kv=state_layout_is_kv,
        precomputed_decay_beta=precomputed_decay_beta,
        num_blocks_per_state_small=num_blocks_per_state_small,
        dense_small_hv_parallel=dense_small_hv_parallel,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )

    # With TVM-FFI enabled at compile time, the runtime launch can pass torch
    # tensors directly instead of rebuilding CuTe tensor wrappers for each call.
    compiled_kernel(
        cu_seqlens_to_use,
        q,
        k,
        v,
        a_kernel,
        b_kernel,
        A_log,
        dt_bias,
        h0_source,
        initial_state_indices,
        o,
        stream,
    )

    return o
