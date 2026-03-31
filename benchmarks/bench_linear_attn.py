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

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from einops import rearrange
from fla.ops.linear_attn import fused_chunk_linear_attn
from fla.ops.linear_attn.utils import normalize_output

# from fla.ops.linear_attn.naive import naive_recurrent_linear_attn
from fla.utils import assert_close, device

from cula.ops.linear_attn import LinearAttentionChunkwise

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

PRINT_DEBUG = False


def print_chunkwise(t, name):
    if not PRINT_DEBUG:
        return
    print(f"--------{name}:")
    c = t.shape[1] // 64
    for i in range(c):
        beg = i * 64
        end = beg + 64
        print(t[:, beg:end])


def print_chunkwise_bhncd(t, name):
    if not PRINT_DEBUG:
        return
    print(f"--------{name}:")
    c = t.shape[2]
    for i in range(c):
        print(t[:, :, i])


def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    # -n, ..., -2, -1, 0
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, : i + 1] = -torch.flip(y, [0])

    return torch.exp(mask)


def get_full_mask(n, slopes):
    if slopes is None:
        mask = torch.tril(torch.ones((n, n)))
    else:
        arr = []
        for slope in slopes:
            arr.append(get_mask(n, slope.item()))
        mask = torch.stack(arr, dim=0)

    return mask


def linear_attn(q, k, v, s=None):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)

    o = o.transpose(1, 2)
    return o


def naive_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    q, k, v = map(lambda x: x.to(torch.float32), (q, k, v))
    o = torch.empty_like(v)

    S = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
    if initial_state is not None:
        S = S + initial_state
    for t in range(T):
        S = S + torch.einsum("b h k, b h v -> b h k v", k[:, t], v[:, t])
        o[:, t] = torch.einsum("b h k v, b h k -> b h v", S, q[:, t] * scale)
    return o.to(dtype), S if output_final_state else None


def naive_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    normalize: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = rearrange(q, "b (n c) h d -> b h n c d", c=chunk_size) * scale
    k = rearrange(k, "b (n c) h d -> b h n c d", c=chunk_size)
    v = rearrange(v, "b (n c) h d -> b h n c d", c=chunk_size)
    # b h n d d
    kv = k.transpose(-1, -2) @ v
    print_chunkwise_bhncd(kv.transpose(-1, -2)[:, :, 0], "kv")
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    qk = q @ k.transpose(-1, -2)
    intra = (
        qk.masked_fill_(
            torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
            0,
        )
    ) @ v
    print_chunkwise_bhncd(inter.transpose(-1, -2), "ointer_naive")
    o = inter + intra
    if normalize:
        o = normalize_output(q * scale, k, o)
    return rearrange(o, "b h n c d -> b (n c) h d")


def test_triton_linear_attn(
    args,
    Q,
    K,
    V,
    decay,
    problem_size,
) -> torch.Tensor:
    B, S, H, D = problem_size
    (chunk_size, acc_dtype, io_dtype, iterations) = args

    # warmup
    for _ in range(2):
        _, _ = fused_chunk_linear_attn(Q, K, V, scale=1, initial_state=None, output_final_state=False, normalize=False)
    torch.cuda.synchronize()

    start = time.perf_counter()
    tri = None
    for _ in range(iterations):
        tri, _ = fused_chunk_linear_attn(Q, K, V, scale=1, initial_state=None, output_final_state=False, normalize=False)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Triton Execution time: {elapsed * 1000 / iterations:.2f} ms (average over {iterations} iterations)")
    return tri, elapsed


def test_cutedsl_linear_attn(
    args,
    Q,
    K,
    V,
    decay,
    problem_size,
) -> torch.Tensor:
    B, S, H, D = problem_size
    (chunk_size, acc_dtype, io_dtype, iterations) = args
    attn_kernel = LinearAttentionChunkwise(
        chunk_size=chunk_size,
        qk_acc_dtype=acc_dtype,
        kv_acc_dtype=acc_dtype,
        io_dtype=io_dtype,
    )

    # Convert to dlpack for CuTe
    q_cute = from_dlpack(Q)
    k_cute = from_dlpack(K)
    v_cute = from_dlpack(V)
    decay_cute = from_dlpack(decay)

    O = torch.zeros_like(Q)
    o_cute = from_dlpack(O)

    # Get default stream
    stream = cutlass.torch.default_stream()

    start_time = time.time()
    compiled = cute.compile(
        attn_kernel,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        o_cute.iterator,
        decay_cute.iterator,
        # (Int32(B), Int32(S), Int32(H), Int32(D)),
        (B, S, H, D),
        stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print(f"B, S, H, D: {(B, S, H, D)}")

    # warm up
    for _ in range(2):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            (B, S, H, D),
            stream,
        )
    torch.cuda.synchronize()

    # Run
    start = time.perf_counter()
    for _ in range(iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            (B, S, H, D),
            stream,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"\nCuteDSL Execution time: {elapsed * 1000 / iterations:.2f} ms (average over {iterations} iterations)")

    return O, elapsed


def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float | None,
    dtype: torch.dtype,
    iterations: int = 10,
):
    torch.manual_seed(42)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    decay = torch.randn(H, dtype=dtype, device=device)
    # h0 = torch.randn((B, H, D, D), dtype=torch.float, device=device)
    # dht = torch.randn_like(h0)

    with torch.no_grad():
        # ref = naive_linear_attn(q, k, v)
        # ref, ref_ht = naive_recurrent_linear_attn(q, k, v, scale=scale, initial_state=h0, output_final_state=False, normalize=False)
        # ((ref * do).sum() + (ref_ht * dht).sum()).backward()
        # ref_dq, q.grad = q.grad.clone(), None
        # ref_dk, k.grad = k.grad.clone(), None
        # ref_dv, v.grad = v.grad.clone(), None
        # ref_dh0, h0.grad = h0.grad.clone(), None

        args = (
            64,
            cutlass.Float32,
            cutlass.BFloat16,
            # test for 10 times
            iterations,
        )

        tri, triton_elapsed = test_triton_linear_attn(args, q, k, v, decay, problem_size=(B, T, H, D))
        # tri, tri_ht = fused_recurrent_linear_attn(q, k, v, scale=scale, initial_state=h0, output_final_state=False, normalize=False)
        # ((tri * do).sum() + (tri_ht * dht).sum()).backward()
        # tri_dq, q.grad = q.grad.clone(), None
        # tri_dk, k.grad = k.grad.clone(), None
        # tri_dv, v.grad = v.grad.clone(), None
        # tri_dh0, h0.grad = h0.grad.clone(), None

        # chunk_size = T if T < 64 else 64
        # ref = naive_chunk_linear_attn(q, k, v, scale=scale, chunk_size=chunk_size, normalize=False)
        # print_chunkwise(ref, "ref_naive_chunk")
        # ref2 = linear_attn(q, k, v)
        # ref3, _ = naive_recurrent_linear_attn(q, k, v, scale=scale, initial_state=h0, output_final_state=False, normalize=False)

        # assert_close('o', ref, tri, 0.05)
        # assert_close('ht', ref_ht, tri_ht, 0.001)
        # assert_close('dq', ref_dq, tri_dq, 0.001)
        # assert_close('dk', ref_dk, tri_dk, 0.001)
        # assert_close('dv', ref_dv, tri_dv, 0.001)
        # assert_close('dh0', ref_dh0, tri_dh0, 0.001)

        if D == 128:
            cutedsl_o, cutedsl_elapsed = test_cutedsl_linear_attn(args, q, k, v, decay, problem_size=(B, T, H, D))
            print_chunkwise(cutedsl_o, "CUTEDSL_O")
            assert_close("o", tri, cutedsl_o, 0.01)

            print(f"Speedup triton_time / cutedsl_time: {float(triton_elapsed) / cutedsl_elapsed:.2f}x")


if __name__ == "__main__":
    B, T, H, D = 64, 4096, 64, 128
    # B, T, H, D = 1, 192, 1, 128
    # B, T, H, D = 2, 4096, 16, 128
    # B, T, H, D = 2, 8*4096, 64, 128
    test_fused_recurrent(B, T, H, D, scale=1.0, dtype=torch.bfloat16, iterations=4)
