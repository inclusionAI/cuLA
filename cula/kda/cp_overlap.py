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

import os
import time
import warnings

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_pre_process as _fla_chunk_gated_delta_rule_fwd_h_pre_process,
)

try:
    import nvshmem.core as _nvshmem_core
except ImportError:
    _nvshmem_core = None

try:
    from nvshmem.core import rma as _nvshmem_rma
except ImportError:
    _nvshmem_rma = None

try:
    from nvshmem.core.direct import ComparisonType as _NVSHMEM_CMP_TYPE
except ImportError:
    _NVSHMEM_CMP_TYPE = None

try:
    from nvshmem.core.direct import SignalOp as _NVSHMEM_SIGNAL_OP
except ImportError:
    _NVSHMEM_SIGNAL_OP = None

try:
    from nvshmem.core.interop.torch import tensor_get_buffer as _nvshmem_tensor_get_buffer
except ImportError:
    _nvshmem_tensor_get_buffer = None

try:
    from cuda.core.experimental import Device as _CudaDevice
except ImportError:
    _CudaDevice = None


_NVSHMEM_INIT_ATTEMPTED = False
_NVSHMEM_INIT_OK = False
_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}
_COMM_EVENTS: dict[int, torch.cuda.Event] = {}
_NVSHMEM_STREAMS: dict[int, object] = {}
_NVSHMEM_COMM_CACHE: dict[tuple[int, int, tuple[int, ...], torch.dtype], dict[str, object]] = {}
_AG_HM_CACHE: dict[tuple[int, int, tuple[int, ...], torch.dtype], torch.Tensor] = {}
_COMPILED_MERGE_CACHE: dict[tuple[int, int, int, int], object] = {}
_CP_TELEMETRY: dict[str, float | int] = {"comm_ms": 0.0, "merge_ms": 0.0, "calls": 0}
_CP_DEBUG_LOG_COUNT = 0
_DIRECT_STORE_CONN_GUARD_WARNED = False
_READY_WAIT_CONN_GUARD_WARNED = False


def _cp_comm_stream_priority() -> int:
    v = os.getenv("CULA_CP_COMM_STREAM_PRIORITY", "-1").strip()
    try:
        return int(v)
    except Exception:
        return -1


def _cp_comm_use_current_stream() -> bool:
    return os.getenv("CULA_CP_COMM_USE_CURRENT_STREAM", "0").lower() in {"1", "true", "on", "yes"}


def _get_comm_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    stream = _COMM_STREAMS.get(idx)
    if stream is None:
        stream = torch.cuda.Stream(device=idx, priority=_cp_comm_stream_priority())
        _COMM_STREAMS[idx] = stream
    return stream


def _get_comm_event(device: torch.device) -> torch.cuda.Event:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ev = _COMM_EVENTS.get(idx)
    if ev is None:
        ev = torch.cuda.Event()
        _COMM_EVENTS[idx] = ev
    return ev


def _is_stream_capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _cp_overlap_enabled() -> bool:
    return os.getenv("CULA_CP_OVERLAP", "0").lower() not in {"0", "false", "off", "no"}


def _cp_comm_backend() -> str:
    return os.getenv("CULA_CP_COMM_BACKEND", "auto").lower()


def _cp_allow_fallback() -> bool:
    return os.getenv("CULA_CP_ALLOW_FALLBACK", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_post_barrier() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_POST_BARRIER", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_pre_barrier() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_PRE_BARRIER", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_collective() -> str:
    return os.getenv("CULA_CP_NVSHMEM_COLLECTIVE", "peer").lower()


def _cp_nvshmem_quiet() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_QUIET", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_direct_store() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_DIRECT_STORE", "1").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_fused_remote_merge() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_FUSED_REMOTE_MERGE", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_direct_store_conn1_only() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_DIRECT_STORE_CONN1_ONLY", "1").lower() in {"1", "true", "on", "yes"}


def _cp_device_max_connections() -> int:
    v = os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "1").strip()
    try:
        return int(v)
    except Exception:
        return 1


def _cp_nvshmem_nongraph_require_conn1() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_NONGRAPH_REQUIRE_CONN1", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_allow_ready_wait_conn_gt1() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_ALLOW_READY_WAIT_CONN_GT1", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_enforce_visibility() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_ENFORCE_VISIBILITY", "1").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_ready_wait() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_READY_WAIT", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_ready_signal_quiet() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_READY_SIGNAL_QUIET", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_ready_wait_in_graph() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_READY_WAIT_IN_GRAPH", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_ready_wait_poll() -> bool:
    return os.getenv("CULA_CP_NVSHMEM_READY_WAIT_POLL", "0").lower() in {"1", "true", "on", "yes"}


def _cp_nvshmem_ready_wait_timeout_ms() -> int:
    v = os.getenv("CULA_CP_NVSHMEM_READY_WAIT_TIMEOUT_MS", "0").strip()
    try:
        n = int(v)
    except Exception:
        return 0
    return max(0, n)


def _cp_merge_impl() -> str:
    return os.getenv("CULA_CP_MERGE_IMPL", "triton").lower()


def _cp_merge_compile_mode() -> str:
    return os.getenv("CULA_CP_MERGE_COMPILE_MODE", "max-autotune")


def _cp_merge_compile_in_graph() -> bool:
    return os.getenv("CULA_CP_MERGE_COMPILE_IN_GRAPH", "0").lower() in {"1", "true", "on", "yes"}


def _cp_direct_pred_fetch_enabled() -> bool:
    return os.getenv("CULA_CP_DIRECT_PRED_FETCH", "1").lower() in {"1", "true", "on", "yes"}


def _cp_direct_pred_fetch_in_graph() -> bool:
    return os.getenv("CULA_CP_DIRECT_PRED_FETCH_IN_GRAPH", "0").lower() in {"1", "true", "on", "yes"}


def _cp_telemetry_enabled() -> bool:
    return os.getenv("CULA_CP_TELEMETRY", "0").lower() in {"1", "true", "on", "yes"}


def _cp_debug_enabled() -> bool:
    return os.getenv("CULA_CP_DEBUG", "0").lower() in {"1", "true", "on", "yes"}


def _cp_debug_max_logs() -> int:
    v = os.getenv("CULA_CP_DEBUG_MAX_LOGS", "64").strip()
    try:
        n = int(v)
    except Exception:
        return 64
    return max(1, n)


def _cp_debug_log(rank: int, msg: str) -> None:
    global _CP_DEBUG_LOG_COUNT
    if not _cp_debug_enabled():
        return
    if _cp_debug_max_logs() <= _CP_DEBUG_LOG_COUNT:
        return
    print(f"[cpdbg][rank{rank}] {msg}", flush=True)
    _CP_DEBUG_LOG_COUNT += 1


def reset_cp_overlap_telemetry() -> None:
    _CP_TELEMETRY["comm_ms"] = 0.0
    _CP_TELEMETRY["merge_ms"] = 0.0
    _CP_TELEMETRY["calls"] = 0


def get_cp_overlap_telemetry() -> dict[str, float | int]:
    return {
        "comm_ms": float(_CP_TELEMETRY["comm_ms"]),
        "merge_ms": float(_CP_TELEMETRY["merge_ms"]),
        "calls": int(_CP_TELEMETRY["calls"]),
    }


def _nvshmem_requires_implicit_visibility_barrier() -> bool:
    if _cp_nvshmem_ready_wait():
        return False
    if not _cp_nvshmem_enforce_visibility():
        return False
    if _cp_allow_fallback():
        return False
    if _cp_comm_backend() != "nvshmem":
        return False
    return (not _cp_nvshmem_pre_barrier()) and (not _cp_nvshmem_post_barrier())


def _nvshmem_should_do_visibility_barrier(*, use_signal_mode: bool) -> bool:
    if _cp_nvshmem_pre_barrier():
        return True
    if _nvshmem_requires_implicit_visibility_barrier():
        return True
    # Ready/wait requested but disabled in this path (e.g. graph-safe mode):
    # fall back to strict implicit visibility barrier policy.
    if use_signal_mode:
        return False
    if not _cp_nvshmem_enforce_visibility():
        return False
    if _cp_allow_fallback():
        return False
    if _cp_comm_backend() != "nvshmem":
        return False
    return (not _cp_nvshmem_pre_barrier()) and (not _cp_nvshmem_post_barrier())


def _nvshmem_can_init(group) -> bool:
    if _nvshmem_core is None:
        return False
    try:
        return dist.get_world_size(group=group) == dist.get_world_size() and dist.get_rank(group=group) == dist.get_rank()
    except Exception:
        return False


def _maybe_init_nvshmem(group) -> bool:
    global _NVSHMEM_INIT_ATTEMPTED, _NVSHMEM_INIT_OK
    if _NVSHMEM_INIT_OK:
        return True
    if _NVSHMEM_INIT_ATTEMPTED:
        return False
    _NVSHMEM_INIT_ATTEMPTED = True
    if not _nvshmem_can_init(group):
        return False

    try:
        current_stream = torch.cuda.current_stream()
        ns = _get_nvshmem_stream(current_stream)
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        uid = _nvshmem_core.get_unique_id() if rank == 0 else None
        uid_holder = [uid]
        dist.broadcast_object_list(uid_holder, src=0, group=group)
        init_kwargs = {
            "uid": uid_holder[0],
            "rank": rank,
            "nranks": world_size,
            "initializer_method": "uid",
        }
        if _CudaDevice is not None:
            dev = _CudaDevice(torch.cuda.current_device())
            dev.set_current()
            init_kwargs["device"] = dev
        _cp_debug_log(rank, "nvshmem_init_enter")
        _nvshmem_core.init(**init_kwargs)
        _nvshmem_core.barrier_all(stream=ns)
        current_stream.synchronize()
        _cp_debug_log(rank, "nvshmem_init_done")
        _NVSHMEM_INIT_OK = True
    except Exception as exc:
        if _cp_allow_fallback():
            warnings.warn(f"NVSHMEM init failed, fallback to NCCL all_gather path. reason: {exc}", stacklevel=2)
        else:
            raise RuntimeError(f"NVSHMEM init failed in strict mode: {exc}") from exc
        _NVSHMEM_INIT_OK = False
    return _NVSHMEM_INIT_OK


def _get_nvshmem_stream(stream: torch.cuda.Stream):
    idx = torch.cuda.current_device()
    ns = _NVSHMEM_STREAMS.get(idx)
    if ns is None or getattr(ns, "__cuda_stream__", None) != stream.cuda_stream:
        ns = _nvshmem_core.NvshmemStream(stream)
        _NVSHMEM_STREAMS[idx] = ns
    return ns


def _get_or_create_nvshmem_cache(*, shape: tuple[int, ...], dtype: torch.dtype, group):
    key = (id(group), torch.cuda.current_device(), tuple(shape), dtype)
    cached = _NVSHMEM_COMM_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_stream_capturing():
        raise RuntimeError("CUDA graph capture requires NVSHMEM symmetric buffers to be allocated during warmup.")
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    _cp_debug_log(rank, f"symmetric_cache_alloc_enter shape={shape} dtype={dtype}")
    sym0 = _nvshmem_core.tensor(tuple(shape), dtype=dtype)
    sym1 = _nvshmem_core.tensor(tuple(shape), dtype=dtype)
    peers0 = [_nvshmem_core.get_peer_tensor(sym0, peer_pe=peer) for peer in range(world_size)]
    peers1 = [_nvshmem_core.get_peer_tensor(sym1, peer_pe=peer) for peer in range(world_size)]
    ready_tensors = None
    ready_buffers = None
    if _nvshmem_tensor_get_buffer is not None and _nvshmem_core is not None:
        ready0 = [_nvshmem_core.tensor((1,), dtype=torch.int64) for _ in range(world_size)]
        ready1 = [_nvshmem_core.tensor((1,), dtype=torch.int64) for _ in range(world_size)]
        for t in ready0:
            t.zero_()
        for t in ready1:
            t.zero_()
        ready_tensors = (ready0, ready1)
        ready_buffers = (
            [_nvshmem_tensor_get_buffer(t)[0] for t in ready0],
            [_nvshmem_tensor_get_buffer(t)[0] for t in ready1],
        )
    dummy_i32 = _nvshmem_core.tensor((1,), dtype=torch.int32)
    dummy_i32.zero_()
    bootstrap_stream = torch.cuda.current_stream()
    bootstrap_stream.synchronize()
    _nvshmem_core.barrier_all(stream=_get_nvshmem_stream(bootstrap_stream))
    bootstrap_stream.synchronize()
    _cp_debug_log(rank, "symmetric_cache_alloc_done")
    state = {
        "step": 0,
        "epoch": [0, 0],
        "sym": (sym0, sym1),
        "peers": (peers0, peers1),
        "ready_tensors": ready_tensors,
        "ready_buffers": ready_buffers,
        "dummy_i32": dummy_i32,
    }
    _NVSHMEM_COMM_CACHE[key] = state
    return state


def _get_or_create_ag_hm(hm: torch.Tensor, num_rows: int, group):
    key = (id(group), torch.cuda.current_device(), tuple(hm.shape), hm.dtype, int(num_rows))
    out = _AG_HM_CACHE.get(key)
    if out is None:
        out = torch.empty(int(num_rows), *hm.shape, device=hm.device, dtype=hm.dtype)
        _AG_HM_CACHE[key] = out
    return out


def _get_or_create_ag_hm_from_spec(*, shape: tuple[int, ...], dtype: torch.dtype, num_rows: int, group):
    key = (id(group), torch.cuda.current_device(), tuple(shape), dtype, int(num_rows))
    out = _AG_HM_CACHE.get(key)
    if out is None:
        out = torch.empty(int(num_rows), *shape, device=torch.device("cuda"), dtype=dtype)
        _AG_HM_CACHE[key] = out
    return out


def _ag_hm_cache_key(hm: torch.Tensor, group, num_rows: int):
    return (id(group), torch.cuda.current_device(), tuple(hm.shape), hm.dtype, int(num_rows))


def _ag_hm_cache_key_from_spec(*, shape: tuple[int, ...], dtype: torch.dtype, group, num_rows: int):
    return (id(group), torch.cuda.current_device(), tuple(shape), dtype, int(num_rows))


def _nvshmem_cache_key(inp: torch.Tensor, group):
    return (id(group), torch.cuda.current_device(), tuple(inp.shape), inp.dtype)


def _nvshmem_cache_key_from_spec(*, shape: tuple[int, ...], dtype: torch.dtype, group):
    return (id(group), torch.cuda.current_device(), tuple(shape), dtype)


def _reserve_nvshmem_slot(*, shape: tuple[int, ...], dtype: torch.dtype, group):
    state = _get_or_create_nvshmem_cache(shape=shape, dtype=dtype, group=group)
    slot = int(state["step"]) & 1
    state["step"] = int(state["step"]) + 1
    state["epoch"][slot] = int(state["epoch"][slot]) + 1
    return slot, state["sym"][slot], int(state["epoch"][slot])


def _nvshmem_signal_api_ready(state: dict[str, object]) -> bool:
    return (
        _nvshmem_rma is not None
        and _NVSHMEM_CMP_TYPE is not None
        and _NVSHMEM_SIGNAL_OP is not None
        and state.get("ready_buffers") is not None
    )


def _nvshmem_signal_mode_enabled(state: dict[str, object], *, allow_in_graph: bool) -> bool:
    global _READY_WAIT_CONN_GUARD_WARNED
    if not _cp_nvshmem_ready_wait():
        return False
    if not allow_in_graph:
        return False
    if (not _is_stream_capturing()) and _cp_device_max_connections() != 1 and (not _cp_nvshmem_allow_ready_wait_conn_gt1()):
        if not _READY_WAIT_CONN_GUARD_WARNED:
            warnings.warn(
                "Disabling NVSHMEM ready-wait for CUDA_DEVICE_MAX_CONNECTIONS!=1 in non-graph mode "
                "(set CULA_CP_NVSHMEM_ALLOW_READY_WAIT_CONN_GT1=1 to override).",
                stacklevel=2,
            )
            _READY_WAIT_CONN_GUARD_WARNED = True
        return False
    if _nvshmem_signal_api_ready(state):
        return True
    if _cp_allow_fallback():
        warnings.warn(
            "CULA_CP_NVSHMEM_READY_WAIT=1 requested but signal API is unavailable; falling back to barrier/peer-copy path.",
            stacklevel=2,
        )
        return False
    raise RuntimeError(
        "CULA_CP_NVSHMEM_READY_WAIT=1 requires nvshmem.core.rma + nvshmem.core.direct + tensor_get_buffer support."
    )


def _nvshmem_publish_ready_epoch(
    *,
    state: dict[str, object],
    slot: int,
    epoch: int,
    rank: int,
    world_size: int,
    target_peers: list[int] | None,
    stream,
) -> None:
    ready_buffers = state["ready_buffers"][slot]
    dummy = state["dummy_i32"]
    peers = list(range(world_size)) if target_peers is None else target_peers
    for peer in peers:
        if peer == rank:
            continue
        _nvshmem_rma.put_signal(
            dummy,
            dummy,
            signal_var=ready_buffers[rank],
            signal_val=epoch,
            signal_op=_NVSHMEM_SIGNAL_OP.SIGNAL_SET,
            remote_pe=peer,
            stream=stream,
        )


def _nvshmem_wait_ready_epoch(
    *,
    state: dict[str, object],
    slot: int,
    epoch: int,
    fetch_peers: list[int],
    rank: int,
    stream,
) -> None:
    timeout_ms = _cp_nvshmem_ready_wait_timeout_ms()
    poll_mode = _cp_nvshmem_ready_wait_poll() or timeout_ms > 0
    if poll_mode:
        ready_tensors = state.get("ready_tensors")
        if ready_tensors is None:
            if _cp_allow_fallback():
                warnings.warn(
                    "CULA_CP_NVSHMEM_READY_WAIT_POLL requested but ready_tensors unavailable; using signal_wait.",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    "CULA_CP_NVSHMEM_READY_WAIT_POLL requires ready_tensors support (nvshmem tensor_get_buffer)."
                )
        else:
            ready_slot = ready_tensors[slot]
            sleep_s = 1e-4
            for peer in fetch_peers:
                if peer == rank:
                    continue
                _cp_debug_log(rank, f"wait_ready_poll_enter peer={peer} slot={slot} epoch={epoch}")
                deadline = None if timeout_ms <= 0 else (time.monotonic() + (float(timeout_ms) / 1000.0))
                while True:
                    observed = int(ready_slot[peer].item())
                    if observed >= epoch:
                        _cp_debug_log(rank, f"wait_ready_poll_done peer={peer} slot={slot} epoch={epoch} observed={observed}")
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RuntimeError(
                            "NVSHMEM ready-wait timeout: "
                            f"peer={peer}, slot={slot}, observed={observed}, required_epoch={epoch}, timeout_ms={timeout_ms}"
                        )
                    time.sleep(sleep_s)
            return
    ready_buffers = state["ready_buffers"][slot]
    cmp_op = _NVSHMEM_CMP_TYPE.CMP_GE if hasattr(_NVSHMEM_CMP_TYPE, "CMP_GE") else _NVSHMEM_CMP_TYPE.CMP_EQ
    for peer in fetch_peers:
        if peer == rank:
            continue
        _cp_debug_log(rank, f"wait_ready_signal_enter peer={peer} slot={slot} epoch={epoch}")
        _nvshmem_rma.signal_wait(
            ready_buffers[peer],
            epoch,
            cmp_op,
            stream=stream,
        )
        _cp_debug_log(rank, f"wait_ready_signal_done peer={peer} slot={slot} epoch={epoch}")


@triton.autotune(
    configs=[
        triton.Config({"BV": 32}, num_warps=4, num_stages=2),
        triton.Config({"BV": 64}, num_warps=4, num_stages=2),
    ],
    key=["H", "K", "V", "NUM_PEERS"],
)
@triton.jit
def _merge_remote_states_kernel(
    out,
    peer0,
    peer1,
    peer2,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_PEERS: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_h = tl.program_id(1)
    state = tl.zeros((BK, BV), dtype=tl.float32)
    stride = K * (K + V)

    for peer_idx in tl.static_range(NUM_PEERS):
        if peer_idx == 0:
            peer = peer0
        elif peer_idx == 1:
            peer = peer1
        else:
            peer = peer2
        peer += i_h * stride
        h_block = tl.make_block_ptr(
            peer,
            (K, V),
            (K + V, 1),
            (0, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        m_block = tl.make_block_ptr(
            peer + V,
            (K, K),
            (K + V, 1),
            (0, 0),
            (BK, BK),
            (1, 0),
        )
        local_h = tl.load(h_block, boundary_check=(0, 1)).to(tl.float32)
        local_m = tl.load(m_block, boundary_check=(0, 1)).to(tl.float32)
        state = tl.dot(local_m, state) + local_h

    out_block = tl.make_block_ptr(
        out + i_h * K * V,
        (K, V),
        (V, 1),
        (0, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    tl.store(out_block, state, boundary_check=(0, 1))


def _merge_remote_states(out: torch.Tensor, peer_tensors: list[torch.Tensor]) -> None:
    if not 1 <= len(peer_tensors) <= 3:
        raise RuntimeError("Fused remote merge supports one to three predecessor peers.")
    h_dim, k_dim, v_dim = out.shape
    bk = triton.next_power_of_2(k_dim)
    padded = peer_tensors + [peer_tensors[0]] * (3 - len(peer_tensors))

    def grid(meta):
        return (triton.cdiv(v_dim, meta["BV"]), h_dim)

    _merge_remote_states_kernel[grid](
        out,
        padded[0],
        padded[1],
        padded[2],
        H=h_dim,
        K=k_dim,
        V=v_dim,
        BK=bk,
        NUM_PEERS=len(peer_tensors),
    )


def _nvshmem_all_gather_into_tensor(
    inp: torch.Tensor,
    out: torch.Tensor | None,
    group,
    stream: torch.cuda.Stream,
    fetch_peers: list[int] | None = None,
    signal_target_peers: list[int] | None = None,
    reserved_slot: int | None = None,
    reserved_epoch: int | None = None,
    input_in_symmetric: bool = False,
    allow_signal_mode: bool = True,
    compact_out_layout: bool = False,
    fetch_last_dim: int | None = None,
    merge_remote_output: bool = False,
) -> None:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if _cp_nvshmem_collective() == "fcollect":
        raise RuntimeError(
            "CULA_CP_NVSHMEM_COLLECTIVE=fcollect is not supported in this build; use CULA_CP_NVSHMEM_COLLECTIVE=peer"
        )
    if fetch_peers is None:
        fetch_peers = list(range(world_size))

    state = _get_or_create_nvshmem_cache(shape=tuple(inp.shape), dtype=inp.dtype, group=group)
    if reserved_slot is None:
        slot = int(state["step"]) & 1
        state["step"] = int(state["step"]) + 1
        state["epoch"][slot] = int(state["epoch"][slot]) + 1
        epoch = int(state["epoch"][slot])
    else:
        slot = reserved_slot
        epoch = int(state["epoch"][slot] if reserved_epoch is None else reserved_epoch)
    sym = state["sym"][slot]
    peers = state["peers"][slot]
    use_signal_mode = _nvshmem_signal_mode_enabled(state, allow_in_graph=allow_signal_mode)
    ns = _get_nvshmem_stream(stream)
    if not input_in_symmetric:
        sym.copy_(inp)
    if use_signal_mode:
        _cp_debug_log(rank, f"publish_ready slot={slot} epoch={epoch} targets={signal_target_peers}")
        # Ensure producer writes are globally visible before signaling readiness.
        _nvshmem_core.quiet(stream=ns)
        _nvshmem_publish_ready_epoch(
            state=state,
            slot=slot,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            target_peers=signal_target_peers,
            stream=ns,
        )
        if _cp_nvshmem_ready_signal_quiet():
            _nvshmem_core.quiet(stream=ns)
    if _nvshmem_should_do_visibility_barrier(use_signal_mode=use_signal_mode):
        _cp_debug_log(rank, f"visibility_barrier_before_fetch slot={slot} epoch={epoch}")
        _nvshmem_core.barrier_all(stream=ns)
    if use_signal_mode:
        _cp_debug_log(rank, f"wait_ready_begin slot={slot} epoch={epoch} fetch_peers={fetch_peers}")
        _nvshmem_wait_ready_epoch(
            state=state,
            slot=slot,
            epoch=epoch,
            fetch_peers=fetch_peers,
            rank=rank,
            stream=ns,
        )
        _cp_debug_log(rank, f"wait_ready_end slot={slot} epoch={epoch} fetch_peers={fetch_peers}")
    if merge_remote_output:
        if out is None or out.ndim != 4 or out.shape[0] != 1:
            raise RuntimeError("Fused remote merge requires an output shaped [1, H, K, V].")
        _merge_remote_states(out[0], [peers[peer] for peer in fetch_peers])
        return
    for fetch_idx, peer in enumerate(fetch_peers):
        if out is None:
            raise RuntimeError("out tensor must be provided when fetch_peers is non-empty.")
        out_row = peer
        if compact_out_layout:
            out_row = fetch_idx
        local_src = inp
        peer_src = peers[peer]
        if fetch_last_dim is not None:
            local_src = local_src[..., :fetch_last_dim]
            peer_src = peer_src[..., :fetch_last_dim]
        if peer == rank:
            out[out_row].copy_(local_src, non_blocking=True)
        else:
            out[out_row].copy_(peer_src, non_blocking=True)
    if _cp_nvshmem_quiet():
        _nvshmem_core.quiet(stream=ns)
    if _cp_nvshmem_post_barrier():
        _nvshmem_core.barrier_all(stream=ns)


def _nvshmem_publish_only(
    inp: torch.Tensor,
    group,
    stream: torch.cuda.Stream,
    signal_target_peers: list[int] | None = None,
    reserved_slot: int | None = None,
    reserved_epoch: int | None = None,
    input_in_symmetric: bool = False,
    allow_signal_mode: bool = True,
) -> None:
    state = _get_or_create_nvshmem_cache(shape=tuple(inp.shape), dtype=inp.dtype, group=group)
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if reserved_slot is None:
        slot = int(state["step"]) & 1
        state["step"] = int(state["step"]) + 1
        state["epoch"][slot] = int(state["epoch"][slot]) + 1
        epoch = int(state["epoch"][slot])
    else:
        slot = reserved_slot
        epoch = int(state["epoch"][slot] if reserved_epoch is None else reserved_epoch)
    sym = state["sym"][slot]
    use_signal_mode = _nvshmem_signal_mode_enabled(state, allow_in_graph=allow_signal_mode)
    ns = _get_nvshmem_stream(stream)
    if not input_in_symmetric:
        sym.copy_(inp)
    if use_signal_mode:
        _cp_debug_log(rank, f"publish_only_ready slot={slot} epoch={epoch} targets={signal_target_peers}")
        # Ensure producer writes are globally visible before signaling readiness.
        _nvshmem_core.quiet(stream=ns)
        _nvshmem_publish_ready_epoch(
            state=state,
            slot=slot,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            target_peers=signal_target_peers,
            stream=ns,
        )
        if _cp_nvshmem_ready_signal_quiet():
            _nvshmem_core.quiet(stream=ns)
    if _nvshmem_should_do_visibility_barrier(use_signal_mode=use_signal_mode):
        _cp_debug_log(rank, f"publish_only_visibility_barrier slot={slot} epoch={epoch}")
        _nvshmem_core.barrier_all(stream=ns)
    if _cp_nvshmem_quiet():
        _nvshmem_core.quiet(stream=ns)
    if _cp_nvshmem_post_barrier():
        _nvshmem_core.barrier_all(stream=ns)


def _get_compiled_merge_fn(*, num_ranks: int, h_dim: int, k_dim: int, v_dim: int):
    compile_mode = _cp_merge_compile_mode()
    key = (num_ranks, h_dim, k_dim, v_dim, torch.cuda.current_device(), compile_mode)
    fn = _COMPILED_MERGE_CACHE.get(key)
    if fn is not None:
        return fn

    def _merge_impl(ag_hm_local: torch.Tensor):
        _, h_dim_local, _, vk_dim = ag_hm_local.shape
        v_local = vk_dim - k_dim
        h_state = torch.zeros(h_dim_local, k_dim, v_local, device=ag_hm_local.device, dtype=torch.float32)
        for ridx in range(num_ranks):
            he_local = ag_hm_local[ridx, :, :, :v_local]
            m_local = ag_hm_local[ridx, :, :, v_local:]
            h_state = torch.matmul(m_local, h_state) + he_local
        return h_state

    compiled = torch.compile(_merge_impl, fullgraph=True, mode=compile_mode)
    _COMPILED_MERGE_CACHE[key] = compiled
    return compiled


def _merge_recurrence_compiled(ag_hm: torch.Tensor, *, rank: int, pre_num_ranks: int) -> torch.Tensor:
    start = rank - int(pre_num_ranks)
    end = rank
    ag_local = ag_hm[start:end]
    num_ranks, h_dim, k_dim, vk_dim = ag_local.shape
    v_dim = vk_dim - k_dim
    fn = _get_compiled_merge_fn(num_ranks=num_ranks, h_dim=h_dim, k_dim=k_dim, v_dim=v_dim)
    return fn(ag_local)


def chunk_gated_delta_rule_fwd_h_pre_process_overlap(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    initial_state: torch.Tensor | None = None,
    context=None,
    transpose_state_layout: bool = False,
):
    global _DIRECT_STORE_CONN_GUARD_WARNED
    if context is None or context.group is None or not _cp_overlap_enabled():
        return _fla_chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            bg=bg,
            v=v,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            use_exp2=use_exp2,
            initial_state=initial_state,
            context=context,
            transpose_state_layout=transpose_state_layout,
        )

    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    from fla.ops.cp import chunk_delta_h as cp_chunk_delta_h

    rank = dist.get_rank(group=context.group)
    world_size = dist.get_world_size(group=context.group)
    pre_num_ranks = int(getattr(context, "pre_num_ranks", 0) or 0)
    _, T, H, K = k.shape
    HV, V = u.shape[2:]
    _cp_debug_log(rank, f"overlap_enter T={T} H={H} HV={HV} K={K} V={V} pre={pre_num_ranks} world={world_size}")
    BT = chunk_size
    BK = triton.next_power_of_2(K)
    if cu_seqlens is None:
        N = k.shape[0]
    else:
        N = len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    hm_shape = (HV, K, (V + K))
    hm_dtype = torch.float32
    predecessor_only = bool((not context.is_first_rank) and pre_num_ranks == 1)
    in_capture = _is_stream_capturing()
    backend = _cp_comm_backend()
    use_nvshmem = backend == "nvshmem" or (backend == "auto" and _nvshmem_core is not None)
    if use_nvshmem and (not in_capture) and _cp_nvshmem_nongraph_require_conn1() and _cp_device_max_connections() != 1:
        raise RuntimeError(
            "Non-graph NVSHMEM CP path requires CUDA_DEVICE_MAX_CONNECTIONS=1 for stability in this build. "
            "Set CULA_CP_NVSHMEM_NONGRAPH_REQUIRE_CONN1=0 to override at your own risk."
        )
    ag_hm_shape = (HV, K, V) if (use_nvshmem and predecessor_only) else hm_shape
    direct_store = False
    reserved_slot = None
    reserved_epoch = None
    direct_store_allowed = True
    if _cp_nvshmem_direct_store_conn1_only() and (not in_capture) and _cp_device_max_connections() != 1:
        direct_store_allowed = False
    if use_nvshmem and _cp_nvshmem_direct_store() and direct_store_allowed:
        init_ok = _NVSHMEM_INIT_OK or (not in_capture and _maybe_init_nvshmem(context.group))
        if init_ok:
            reserved_slot, hm, reserved_epoch = _reserve_nvshmem_slot(shape=hm_shape, dtype=hm_dtype, group=context.group)
            direct_store = True
            _cp_debug_log(rank, f"reserve_slot_ok slot={reserved_slot} epoch={reserved_epoch} direct_store=1")
        else:
            hm = k.new_zeros(*hm_shape, dtype=hm_dtype)
            _cp_debug_log(rank, "reserve_slot_init_not_ok direct_store=0")
    else:
        hm = k.new_zeros(*hm_shape, dtype=hm_dtype)
        if use_nvshmem and _cp_nvshmem_direct_store() and (not direct_store_allowed) and (not _DIRECT_STORE_CONN_GUARD_WARNED):
            warnings.warn(
                "Disabling NVSHMEM direct_store because CUDA_DEVICE_MAX_CONNECTIONS!=1 in non-graph mode "
                "(set CULA_CP_NVSHMEM_DIRECT_STORE_CONN1_ONLY=0 to override).",
                stacklevel=2,
            )
            _DIRECT_STORE_CONN_GUARD_WARNED = True
        _cp_debug_log(
            rank,
            f"reserve_slot_bypass use_nvshmem={use_nvshmem} direct_store={direct_store} "
            f"direct_store_allowed={direct_store_allowed}",
        )
    if not context.is_last_rank:
        _cp_debug_log(rank, "preprocess_kernel_enter")
        block_size = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, block_size) + triton.cdiv(K, block_size), HV)
        cp_chunk_delta_h.pre_process_fwd_kernel_merged[grid](
            k=k,
            v=u if v is None else v,
            w=w,
            g=g,
            gk=gk,
            bg=bg,
            u=u,
            hm=hm,
            cu_seqlens=cu_seqlens[-2:],
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=block_size,
            MULTI_SEQS=False,
        )
        _cp_debug_log(rank, "preprocess_kernel_done")

    compact_ag_hm = bool(use_nvshmem and (not context.is_first_rank))
    ag_hm_rows = pre_num_ranks if compact_ag_hm else world_size
    ag_hm_key = _ag_hm_cache_key_from_spec(shape=ag_hm_shape, dtype=hm_dtype, group=context.group, num_rows=ag_hm_rows)
    if in_capture:
        if (not context.is_first_rank) and ag_hm_key not in _AG_HM_CACHE:
            raise RuntimeError(
                "CUDA graph capture requires warmup before capture (missing ag_hm cache for this shape/device)."
            )
        if use_nvshmem:
            if not _NVSHMEM_INIT_OK:
                raise RuntimeError("CUDA graph capture requires NVSHMEM to be initialized before capture.")
            if _nvshmem_cache_key_from_spec(shape=hm_shape, dtype=hm_dtype, group=context.group) not in _NVSHMEM_COMM_CACHE:
                raise RuntimeError(
                    "CUDA graph capture requires warmup before capture (missing NVSHMEM comm cache for this shape/device)."
                )
        # CUDA Graph capture-safe path: keep communication on the capturing stream.
        comm_stream = torch.cuda.current_stream()
        comm_event = None
    elif _cp_comm_use_current_stream():
        comm_stream = torch.cuda.current_stream()
        comm_event = None
    else:
        comm_stream = _get_comm_stream(hm.device)
        comm_event = _get_comm_event(hm.device)
    direct_predecessor_state = bool(
        _cp_direct_pred_fetch_enabled()
        and use_nvshmem
        and predecessor_only
        and (not transpose_state_layout)
        and (N == 1)
        and ((not in_capture) or _cp_direct_pred_fetch_in_graph())
        and (not context.is_first_rank)
    )
    direct_remote_merge = bool(
        _cp_nvshmem_fused_remote_merge()
        and use_nvshmem
        and _NVSHMEM_INIT_OK
        and world_size <= 4
        and pre_num_ranks > 1
        and not in_capture
        and not transpose_state_layout
        and N == 1
    )
    direct_initial_state = direct_predecessor_state or direct_remote_merge

    initial_state = None
    ag_hm = None
    telemetry_enabled = _cp_telemetry_enabled() and (not in_capture)
    comm_start = None
    comm_end = None
    if telemetry_enabled:
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
    merge_ms = 0.0
    if not context.is_first_rank:
        _cp_debug_log(
            rank,
            f"ag_hm_alloc_enter direct_pred_state={direct_predecessor_state} direct_remote_merge={direct_remote_merge}",
        )
        if direct_initial_state:
            # Materialize the final state directly, avoiding a separate receive
            # buffer and a second local merge/copy pass.
            initial_state = k.new_empty(N, HV, K, V, dtype=torch.float32)
            ag_hm = initial_state
            if direct_predecessor_state:
                # Keep capture-time cache guard satisfied by materializing the
                # compact receive cache during warmup.
                _get_or_create_ag_hm_from_spec(
                    shape=ag_hm_shape,
                    dtype=hm_dtype,
                    num_rows=ag_hm_rows,
                    group=context.group,
                )
            _cp_debug_log(rank, "ag_hm_alloc_done direct_initial_state=1")
        else:
            ag_hm = _get_or_create_ag_hm_from_spec(
                shape=ag_hm_shape,
                dtype=hm_dtype,
                num_rows=ag_hm_rows,
                group=context.group,
            )
            _cp_debug_log(rank, "ag_hm_alloc_done direct_pred_state=0")
    post_num_ranks = int(getattr(context, "post_num_ranks", 0) or 0)
    signal_target_peers = list(range(rank + 1, rank + post_num_ranks + 1)) if post_num_ranks > 0 else []
    allow_signal_mode = (not in_capture) or _cp_nvshmem_ready_wait_in_graph()
    if not in_capture and comm_stream != torch.cuda.current_stream():
        comm_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(comm_stream):
        if comm_start is not None:
            comm_start.record(comm_stream)
        _cp_debug_log(
            rank,
            (
                f"comm_begin in_capture={in_capture} use_nvshmem={use_nvshmem} backend={backend} "
                f"pre_num_ranks={pre_num_ranks} post_num_ranks={post_num_ranks} "
                f"direct_store={direct_store} direct_pred_state={direct_predecessor_state}"
            ),
        )
        if use_nvshmem:
            if not _NVSHMEM_INIT_OK and not _maybe_init_nvshmem(context.group):
                if backend == "nvshmem" and not _cp_allow_fallback():
                    raise RuntimeError("NVSHMEM backend requested but initialization failed (strict mode).")
                if ag_hm is None:
                    raise RuntimeError("ag_hm is required for fallback gather path.")
                if ag_hm.shape[0] == world_size:
                    work = dist.all_gather_into_tensor(ag_hm, hm, group=context.group, async_op=True)
                    work.wait()
                else:
                    ag_hm_world = _get_or_create_ag_hm(hm=hm, num_rows=world_size, group=context.group)
                    work = dist.all_gather_into_tensor(ag_hm_world, hm, group=context.group, async_op=True)
                    work.wait()
                    start = rank - pre_num_ranks
                    if predecessor_only:
                        ag_hm.copy_(ag_hm_world[start:rank, :, :, :V], non_blocking=True)
                    else:
                        ag_hm.copy_(ag_hm_world[start:rank], non_blocking=True)
            else:
                if context.is_first_rank:
                    _cp_debug_log(rank, "comm_publish_only_enter")
                    _nvshmem_publish_only(
                        hm,
                        group=context.group,
                        stream=comm_stream,
                        signal_target_peers=signal_target_peers,
                        reserved_slot=reserved_slot,
                        reserved_epoch=reserved_epoch,
                        input_in_symmetric=direct_store,
                        allow_signal_mode=allow_signal_mode,
                    )
                    _cp_debug_log(rank, "comm_publish_only_done")
                else:
                    fetch_peers = list(range(rank - pre_num_ranks, rank))
                    _cp_debug_log(rank, f"comm_fetch_enter fetch_peers={fetch_peers}")
                    _nvshmem_all_gather_into_tensor(
                        hm,
                        ag_hm,
                        group=context.group,
                        stream=comm_stream,
                        fetch_peers=fetch_peers,
                        signal_target_peers=signal_target_peers,
                        reserved_slot=reserved_slot,
                        reserved_epoch=reserved_epoch,
                        input_in_symmetric=direct_store,
                        allow_signal_mode=allow_signal_mode,
                        compact_out_layout=True,
                        fetch_last_dim=(V if predecessor_only else None),
                        merge_remote_output=direct_remote_merge,
                    )
                    _cp_debug_log(rank, "comm_fetch_done")
        elif backend in {"auto", "nccl"}:
            # all_gather is collective: every rank must participate.
            gather_dst = ag_hm
            if gather_dst is None:
                gather_dst = _get_or_create_ag_hm(hm=hm, num_rows=world_size, group=context.group)
            if in_capture:
                # Graph capture path: avoid async handle + host wait inside capture.
                dist.all_gather_into_tensor(gather_dst, hm, group=context.group)
            else:
                work = dist.all_gather_into_tensor(gather_dst, hm, group=context.group, async_op=True)
                work.wait()
        else:
            raise ValueError(f"Unsupported CULA_CP_COMM_BACKEND={backend}")
        if comm_event is not None:
            comm_event.record(comm_stream)
        if comm_end is not None:
            comm_end.record(comm_stream)
        _cp_debug_log(rank, "comm_end")
    if comm_event is not None and (not context.is_first_rank):
        torch.cuda.current_stream().wait_event(comm_event)

    comm_ms = 0.0
    if telemetry_enabled and comm_start is not None and comm_end is not None:
        comm_end.synchronize()
        comm_ms = float(comm_start.elapsed_time(comm_end))

    if initial_state is None:
        if transpose_state_layout:
            initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)

    if not context.is_first_rank:
        merge_start = None
        merge_end = None
        if telemetry_enabled:
            merge_start = torch.cuda.Event(enable_timing=True)
            merge_end = torch.cuda.Event(enable_timing=True)
            merge_start.record(torch.cuda.current_stream())
        assert ag_hm is not None
        merge_rank = pre_num_ranks if compact_ag_hm else rank
        if pre_num_ranks == 1:
            if not direct_initial_state:
                source_idx = pre_num_ranks - 1 if compact_ag_hm else (rank - 1)
                source = ag_hm[source_idx, :, :, :V]
                if transpose_state_layout:
                    initial_state[0].copy_(source.transpose(-2, -1))
                else:
                    initial_state[0].copy_(source)
        elif not direct_remote_merge:
            use_compiled_merge = (
                _cp_merge_impl() == "compile"
                and (not in_capture or _cp_merge_compile_in_graph())
                and not transpose_state_layout
                and ag_hm.dtype == torch.float32
                and pre_num_ranks > 1
            )
            if use_compiled_merge:
                merged = _merge_recurrence_compiled(
                    ag_hm,
                    rank=merge_rank,
                    pre_num_ranks=pre_num_ranks,
                )
                initial_state[0].copy_(merged)
            else:

                def grid(meta):
                    return (triton.cdiv(V, meta["BV"]), HV)

                cp_chunk_delta_h.merge_fwd_bwd_kernel[grid](
                    h=initial_state[0],
                    ag_hm=ag_hm,
                    pre_or_post_num_ranks=pre_num_ranks,
                    rank=merge_rank,
                    seq_offsets=None,
                    init_offsets=None,
                    h0_seq_ids=None,
                    h0=None,
                    HV=HV,
                    K=K,
                    V=V,
                    BK=BK,
                    FORWARD=True,
                    INTRACARD_MODE=False,
                    NUM_SEQ_ENTRIES=0,
                    TRANSPOSE_STATE=transpose_state_layout,
                )
        if telemetry_enabled and merge_start is not None and merge_end is not None:
            merge_end.record(torch.cuda.current_stream())
            merge_end.synchronize()
            merge_ms = float(merge_start.elapsed_time(merge_end))
    if telemetry_enabled:
        _CP_TELEMETRY["comm_ms"] = float(_CP_TELEMETRY["comm_ms"]) + comm_ms
        _CP_TELEMETRY["merge_ms"] = float(_CP_TELEMETRY["merge_ms"]) + merge_ms
        _CP_TELEMETRY["calls"] = int(_CP_TELEMETRY["calls"]) + 1
    return initial_state
