import argparse
import ctypes
from functools import lru_cache
from pathlib import Path


def _import_fla_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda
    except Exception as exc:  # noqa: BLE001
        return None, exc
    return chunk_kda, None


@lru_cache(maxsize=1)
def _torch_modules():
    import torch
    import torch.nn.functional as F

    return torch, F


class _KdaPrefillIOF32(ctypes.Structure):
    _fields_ = [
        ("q_ptr", ctypes.c_void_p),
        ("k_ptr", ctypes.c_void_p),
        ("v_ptr", ctypes.c_void_p),
        ("g_ptr", ctypes.c_void_p),
        ("beta_ptr", ctypes.c_void_p),
        ("w_ptr", ctypes.c_void_p),
        ("u_ptr", ctypes.c_void_p),
        ("o_ptr", ctypes.c_void_p),
        ("batch_size", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("seq_len", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("value_dim", ctypes.c_int),
        ("chunk_size", ctypes.c_int),
        ("num_chunks", ctypes.c_int),
    ]


@lru_cache(maxsize=1)
def _load_local_prefill_library():
    repo_root = Path(__file__).resolve().parent.parent
    lib_path = repo_root / "build" / "libkda_prefill_runtime.so"
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Expected {lib_path} after `cmake --build build`. "
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.kda_prefill_f32.argtypes = [ctypes.POINTER(_KdaPrefillIOF32), ctypes.c_void_p]
    lib.kda_prefill_f32.restype = ctypes.c_int
    return lib


def _make_prefill_inputs(B, H, T, K, V, device="cuda", seed=42):
    torch, _ = _torch_modules()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(B, H, T, K, device=device, dtype=torch.float32, generator=generator)
    k = torch.randn(B, H, T, K, device=device, dtype=torch.float32, generator=generator)
    v = torch.randn(B, H, T, V, device=device, dtype=torch.float32, generator=generator)
    g = torch.randn(B, H, T, K, device=device, dtype=torch.float32, generator=generator)
    beta = torch.randn(B, H, T, device=device, dtype=torch.float32, generator=generator)
    return q, k, v, g, beta


def _num_chunks(seq_len, chunk_size):
    return (seq_len + chunk_size - 1) // chunk_size


def _run_local_prefill(q, k, v, g, beta, chunk_size, w=None, u=None, o=None):
    torch, _ = _torch_modules()
    if q.device.type != "cuda":
        raise ValueError("CUDA tensors required.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    B, H, T, K = q.shape
    V = v.shape[-1]
    num_chunks = _num_chunks(T, chunk_size)
    if w is None:
        w = torch.empty((B, H, num_chunks, chunk_size, K), device=q.device, dtype=torch.float32)
    if u is None:
        u = torch.empty((B, H, num_chunks, chunk_size, V), device=q.device, dtype=torch.float32)
    if o is None:
        o = torch.empty((B, H, T, V), device=q.device, dtype=torch.float32)

    io = _KdaPrefillIOF32(
        q_ptr=ctypes.c_void_p(q.data_ptr()),
        k_ptr=ctypes.c_void_p(k.data_ptr()),
        v_ptr=ctypes.c_void_p(v.data_ptr()),
        g_ptr=ctypes.c_void_p(g.data_ptr()),
        beta_ptr=ctypes.c_void_p(beta.data_ptr()),
        w_ptr=ctypes.c_void_p(w.data_ptr()),
        u_ptr=ctypes.c_void_p(u.data_ptr()),
        o_ptr=ctypes.c_void_p(o.data_ptr()),
        batch_size=B,
        num_heads=H,
        seq_len=T,
        head_dim=K,
        value_dim=V,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
    )

    stream = torch.cuda.current_stream(device=q.device)
    err = _load_local_prefill_library().kda_prefill_f32(
        ctypes.byref(io),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if err != 0:
        raise RuntimeError(f"kda_prefill_f32 failed with cudaError code {err}")
    return o, w, u


def _local_prefill_benchmark(device, args) -> tuple[float, dict]:
    torch, _ = _torch_modules()
    q, k, v, g, beta = _make_prefill_inputs(
        args.B, args.H, args.T, args.K, args.V, device=device, seed=42
    )
    num_chunks = _num_chunks(args.T, args.C)
    w = torch.empty((args.B, args.H, num_chunks, args.C, args.K), device=device, dtype=torch.float32)
    u = torch.empty((args.B, args.H, num_chunks, args.C, args.V), device=device, dtype=torch.float32)
    out = torch.empty((args.B, args.H, args.T, args.V), device=device, dtype=torch.float32)

    for _ in range(3):
        _run_local_prefill(q, k, v, g, beta, args.C, w=w, u=u, o=out)
    torch.cuda.synchronize(device)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_stop = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    for _ in range(args.iters):
        out, _, _ = _run_local_prefill(q, k, v, g, beta, args.C, w=w, u=u, o=out)
    ev_stop.record()
    torch.cuda.synchronize(device)

    return ev_start.elapsed_time(ev_stop) / float(args.iters), {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "out": out,
    }


def _run_fla_prefill(chunk_kda, q, k, v, g, beta):
    torch, _ = _torch_modules()
    B, H, T, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5

    q_fla = q.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    k_fla = k.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    v_fla = v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
    beta_fla = beta.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    g_fla = (
        torch.log(torch.sigmoid(g).clamp_min(1.0e-6))
        .permute(0, 2, 1, 3)
        .contiguous()
        .to(torch.bfloat16)
    )
    h0 = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

    result = chunk_kda(
        q=q_fla,
        k=k_fla,
        v=v_fla,
        g=g_fla,
        beta=beta_fla,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
    )
    out = result[0] if isinstance(result, tuple) else result
    return out.permute(0, 2, 1, 3).contiguous().float()


def _fla_prefill_benchmark(device, args) -> tuple[float, dict]:
    torch, _ = _torch_modules()
    chunk_kda, import_error = _import_fla_chunk_kda()
    if chunk_kda is None:
        raise RuntimeError(
            "Need `fla` with `chunk_kda` for this benchmark: "
            f"{import_error}"
        )

    q, k, v, g, beta = _make_prefill_inputs(
        args.B, args.H, args.T, args.K, args.V, device=device, seed=42
    )
    for _ in range(3):
        _run_fla_prefill(chunk_kda, q, k, v, g, beta)
    torch.cuda.synchronize(device)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_stop = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    for _ in range(args.iters):
        out = _run_fla_prefill(chunk_kda, q, k, v, g, beta)
    ev_stop.record()
    torch.cuda.synchronize(device)

    return ev_start.elapsed_time(ev_stop) / float(args.iters), {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "out": out,
    }


def _prefill_reference(q, k, v, g, beta, chunk_size):
    torch, F = _torch_modules()
    B, H, T, K = q.shape
    V = v.shape[-1]
    out = torch.zeros((B, H, T, V), device=q.device, dtype=torch.float32)
    q_scale = K**-0.5

    for b in range(B):
        for h in range(H):
            state = torch.zeros((K, V), device=q.device, dtype=torch.float32)
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                valid_c = end - start
                if valid_c <= 0:
                    continue

                q_chunk = q[b, h, start:end].float()
                k_chunk = k[b, h, start:end].float()
                v_chunk = v[b, h, start:end].float()
                g_chunk = g[b, h, start:end].float()
                beta_chunk = beta[b, h, start:end].float()

                q_normed = F.normalize(q_chunk, dim=-1) * q_scale
                k_normed = F.normalize(k_chunk, dim=-1)
                g_prefix = torch.cumprod(torch.sigmoid(g_chunk), dim=0)
                M = torch.eye(valid_c, device=q.device, dtype=torch.float32)

                for i in range(valid_c):
                    for j in range(i):
                        decay = g_prefix[i] / g_prefix[j]
                        dot = (k_normed[i] * k_normed[j] * decay).sum()
                        M[i, j] = dot * F.softplus(beta_chunk[i])

                for i in range(1, valid_c):
                    for j in range(i):
                        acc = torch.zeros((), device=q.device, dtype=torch.float32)
                        for kk in range(j, i):
                            acc = acc + M[i, kk] * (1.0 if kk == j else M[kk, j])
                        M[i, j] = -acc

                swish_g = g_chunk * torch.sigmoid(g_chunk)
                W = torch.zeros((valid_c, K), device=q.device, dtype=torch.float32)
                U = torch.zeros((valid_c, V), device=q.device, dtype=torch.float32)
                for r in range(valid_c):
                    coeff = M[r, : r + 1].unsqueeze(-1)
                    W[r] = (coeff * (swish_g[: r + 1] * k_normed[: r + 1])).sum(dim=0)
                    U[r] = (coeff * v_chunk[: r + 1]).sum(dim=0)

                for r in range(valid_c):
                    out[b, h, start + r] = U[r] + (q_normed[r] * g_prefix[r]) @ state

                state = state * g_prefix[valid_c - 1].unsqueeze(-1) + W.transpose(0, 1) @ U

    return out


def _summarize_diff(name, ref, actual):
    diff = (ref.float() - actual.float()).abs()
    return (
        f"{name}: max_abs={diff.max().item():.6e}, "
        f"mean_abs={diff.mean().item():.6e}"
    )


def _run_prefill_accuracy_compare(device, args):
    torch, _ = _torch_modules()
    chunk_kda, import_error = _import_fla_chunk_kda()
    if chunk_kda is None:
        raise RuntimeError(
            "Need `fla` for accuracy compare: "
            f"{import_error}"
        )

    val_B = min(args.B, 1)
    val_H = min(args.H, 4)
    val_T = min(args.T, 256)
    q, k, v, g, beta = _make_prefill_inputs(
        val_B, val_H, val_T, args.K, args.V, device=device, seed=123
    )
    local_out, _, _ = _run_local_prefill(q, k, v, g, beta, args.C)
    torch.cuda.synchronize(device)
    fla_out = _run_fla_prefill(chunk_kda, q, k, v, g, beta)
    torch.cuda.synchronize(device)

    print(
        "Accuracy compare shape: "
        f"B={val_B}, H={val_H}, T={val_T}, K={args.K}, V={args.V}, C={args.C}"
    )
    print("Reference: FLA chunk_kda")
    print(_summarize_diff("Local vs FLA", fla_out, local_out))

    return {
        "local_out": local_out,
        "fla_out": fla_out,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KDA prefill (ctypes + libkda_prefill_runtime.so) vs FLA."
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--suite",
        default="fla-benchmark",
        choices=[
            "local-benchmark",
            "fla-only-benchmark",
            "fla-benchmark",
        ],
    )
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--V", type=int, default=64)
    parser.add_argument("--C", type=int, default=32)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def run_fla_benchmark_suite(device, args) -> int:
    print(
        "Benchmark shapes: "
        f"B={args.B}, H={args.H}, T={args.T}, K={args.K}, V={args.V}, C={args.C}"
    )
    print("\n[Time]")
    local_ms, _ = _local_prefill_benchmark(device, args)
    print(f"local CUDA prefill (avg {args.iters} iters): {local_ms:.4f} ms")

    try:
        fla_ms, _ = _fla_prefill_benchmark(device, args)
    except Exception as exc:  # noqa: BLE001
        print(f"FLA benchmark skipped: {exc}")
        return 0

    print(f"FLA chunk_kda (avg {args.iters} iters): {fla_ms:.4f} ms")
    print(f"FLA / local: {fla_ms / local_ms:.4f}x")
    print(f"local / FLA: {local_ms / fla_ms:.4f}x")
    print("\n[Accuracy]")
    _run_prefill_accuracy_compare(device, args)
    return 0


def run_local_benchmark_suite(args) -> int:
    torch, _ = _torch_modules()
    device = torch.device(args.device)
    print(
        "Benchmark shapes: "
        f"B={args.B}, H={args.H}, T={args.T}, K={args.K}, V={args.V}, C={args.C}"
    )
    local_ms, _ = _local_prefill_benchmark(device, args)
    print(f"local CUDA prefill (avg {args.iters} iters): {local_ms:.4f} ms")
    return 0


def run_fla_only_benchmark_suite(device, args) -> int:
    print(
        "Benchmark shapes: "
        f"B={args.B}, H={args.H}, T={args.T}, K={args.K}, V={args.V}, C={args.C}"
    )
    try:
        fla_ms, _ = _fla_prefill_benchmark(device, args)
    except Exception as exc:  # noqa: BLE001
        print(f"FLA benchmark unavailable: {exc}")
        return 1
    print(f"FLA chunk_kda (avg {args.iters} iters): {fla_ms:.4f} ms")
    return 0


def main() -> int:
    args = _parse_args()
    torch, _ = _torch_modules()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    if args.suite == "local-benchmark":
        return run_local_benchmark_suite(args)
    device = torch.device(args.device)
    if args.suite == "fla-only-benchmark":
        return run_fla_only_benchmark_suite(device, args)
    return run_fla_benchmark_suite(device, args)


if __name__ == "__main__":
    raise SystemExit(main())
