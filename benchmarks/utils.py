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

import contextlib
import functools
import random
import subprocess

import numpy as np
import torch
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices


def get_gpu_name():
    """Return the name of the first visible CUDA GPU, or 'Unknown GPU'."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "Unknown GPU"


def get_env_info():
    """Return a dict with gpu, cuda (from nvcc), and torch version info."""
    cuda_version = "Unknown"
    with contextlib.suppress(Exception):
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        # e.g. "Cuda compilation tools, release 12.9, V12.9.41"
        for line in out.splitlines():
            if "release" in line:
                cuda_version = line.split("release")[-1].strip().split(",")[0]
                break
    return {
        "gpu": get_gpu_name(),
        "cuda": cuda_version,
        "torch": torch.__version__,
    }


SEED = 42
CHUNK_SIZE = 64


def set_seed(seed: int):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def exclusive_cumsum(a: list[int]):
    r = [0]
    for v in a:
        r.append(r[-1] + v)
    return r


def multidist_randn(num_dists, dim, mean_mean=0.0, mean_std=1.0, scale_lower=0.5, scale_upper=1.5):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    scales = torch.distributions.Uniform(scale_lower, scale_upper).sample((num_dists,))
    data = torch.distributions.Normal(means, scales).sample((dim,))
    return data.T.contiguous()


def multidist_randu(num_dists, dim, mean_mean=0.0, mean_std=1.0, lower=-1.0, upper=1.0):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    data = torch.distributions.Uniform(means + lower, means + upper).sample((dim,))
    return data.T.contiguous()


def gen_qkv(seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=torch.float16):
    # qkv_rng = functools.partial(multidist_randn, mean_std=0.1)
    qkv_rng = functools.partial(multidist_randu, mean_std=0.05, lower=-0.25, upper=0.25)

    total_seq_lens = sum(seq_lens)
    q = qkv_rng(total_seq_lens * num_q_heads, head_size)
    k = qkv_rng(total_seq_lens * num_k_heads, head_size)
    v = qkv_rng(total_seq_lens * num_v_heads, head_size)

    q = q.reshape(total_seq_lens, num_q_heads, head_size).to(dtype).contiguous()
    k = k.reshape(total_seq_lens, num_k_heads, head_size).to(dtype).contiguous()
    v = v.reshape(total_seq_lens, num_v_heads, head_size).to(dtype).contiguous()

    return q, k, v


def generate_random_seq_lens(num_seqs: int, total_len: int, min_seq_len: int, variance: float = 1.0, seed: int = 42) -> list:
    """
    Generate a list of random sequence lengths satisfying:
    - Number of sequences: num_seqs
    - Total length: total_len
    - Each sequence length >= min_seq_len
    - variance: controls the distribution of lengths
        - 0.0: perfectly balanced, all lengths as equal as possible
        - 1.0: normal random allocation
        - >1.0: more imbalanced, larger differences between lengths
    """
    assert total_len >= num_seqs * min_seq_len, (
        f"total_len ({total_len}) must be >= num_seqs ({num_seqs}) * min_seq_len ({min_seq_len})"
    )

    random.seed(seed)

    # Compute balanced sequence length
    base_len = total_len // num_seqs
    remainder = total_len % num_seqs

    if variance == 0.0:
        # Perfectly balanced allocation
        seq_lens = [base_len] * num_seqs
        # Distribute remainder to the first few sequences
        for i in range(remainder):
            seq_lens[i] += 1
    else:
        # Assign minimum length to each sequence first
        seq_lens = [min_seq_len] * num_seqs
        remaining = total_len - num_seqs * min_seq_len

        if remaining > 0:
            if variance >= 1.0:
                # High variance: use Dirichlet distribution to generate weights
                # Smaller alpha leads to more uneven distribution
                alpha = 1.0 / variance
                weights = [random.gammavariate(alpha, 1.0) for _ in range(num_seqs)]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Distribute remaining length by weights
                extra_lens = [int(remaining * w) for w in weights]
                # Handle rounding error
                diff = remaining - sum(extra_lens)
                for i in range(abs(diff)):
                    idx = random.randint(0, num_seqs - 1)
                    extra_lens[idx] += 1 if diff > 0 else -1

                for i in range(num_seqs):
                    seq_lens[i] += extra_lens[i]
            else:
                # Low variance (0 < variance < 1): interpolate between balanced and random
                # Compute balanced allocation
                balanced = [base_len] * num_seqs
                for i in range(remainder):
                    balanced[i] += 1

                # Compute random allocation
                random_lens = [min_seq_len] * num_seqs
                for _ in range(remaining):
                    idx = random.randint(0, num_seqs - 1)
                    random_lens[idx] += 1

                # Interpolate by variance
                seq_lens = [int(balanced[i] * (1 - variance) + random_lens[i] * variance) for i in range(num_seqs)]
                # Fix total length
                diff = total_len - sum(seq_lens)
                for i in range(abs(diff)):
                    idx = i % num_seqs
                    seq_lens[idx] += 1 if diff > 0 else -1

    # Ensure all sequence lengths >= min_seq_len
    for i in range(num_seqs):
        if seq_lens[i] < min_seq_len:
            deficit = min_seq_len - seq_lens[i]
            seq_lens[i] = min_seq_len
            # Borrow from other sequences
            for j in range(num_seqs):
                if j != i and seq_lens[j] > min_seq_len:
                    take = min(deficit, seq_lens[j] - min_seq_len)
                    seq_lens[j] -= take
                    deficit -= take
                    if deficit == 0:
                        break

    assert sum(seq_lens) == total_len, f"sum(seq_lens)={sum(seq_lens)} != total_len={total_len}"
    assert all(s >= min_seq_len for s in seq_lens), "Some seq_len < min_seq_len"

    return seq_lens


# ==============================================================================
# Varlen sequence length generators
# ==============================================================================


def gen_uniform(N, T):
    """All sequences have equal length."""
    per = T // N
    lens = [per] * N
    lens[0] += T - per * N  # absorb remainder
    return lens


def gen_skewed(N, T):
    """One long sequence + many short ones."""
    if N == 1:
        return [T]
    short = max(1, T // (2 * (N - 1)))
    long_len = T - short * (N - 1)
    return [long_len] + [short] * (N - 1)


def gen_random(N, T, seed=42):
    """Random sequence lengths summing to T."""
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(N))
    lens = np.maximum(1, np.round(raw * T).astype(int))
    diff = T - lens.sum()
    lens[0] += diff
    lens = np.maximum(1, lens)
    return lens.tolist()


def build_varlen_configs(
    num_seqs_list=(1, 5, 10, 20),
    total_lens=(4096, 8192, 16384),
    dists=("uniform", "random", "skewed"),
    random_seed=42,
):
    """Build a list of (seq_lens, total_len, dist_name) configs for varlen benchmarks.

    Returns:
        list of (seq_lens: list[int], total_len: int, dist: str)
    """
    configs = []
    for T in total_lens:
        for N in num_seqs_list:
            if T // N < 1:
                continue
            for d in dists:
                if d == "uniform":
                    seq_lens = gen_uniform(N, T)
                elif d == "skewed":
                    seq_lens = gen_skewed(N, T)
                elif d == "random":
                    seq_lens = gen_random(N, T, seed=random_seed)
                else:
                    raise ValueError(f"Unknown dist: {d}")
                configs.append((seq_lens, T, d))
    return configs


# ==============================================================================
# Common input preparation functions for benchmarks and demos
# ==============================================================================


def prepare_safe_gate_inputs(
    batch_size, T, H, D, device, cu_seqlens=None, chunk_size=CHUNK_SIZE, seed=SEED, has_init_state=False
):
    """Prepare inputs for safe_gate benchmarks (use_gate_in_kernel=True, safe_gate=True).

    All tensors are flattened to (1, B*T, ...) for cu_seqlens compatibility.
    """
    dtype = torch.bfloat16
    scale = D ** (-0.5)

    set_seed(seed)

    q = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    beta = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)

    A_log = torch.randn(H, dtype=torch.float, device=device).requires_grad_(False)
    dt_bias = torch.randn(H * D, dtype=torch.float, device=device).requires_grad_(False)

    # flatten to batch_size=1 for cu_seqlens compatibility
    if batch_size != 1:
        q, k, v, g, beta = map(lambda x: rearrange(x, "b t ... -> 1 (b t) ..."), (q, k, v, g, beta))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    init_state = None
    if has_init_state:
        num_seqs = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch_size
        init_state = torch.randn(num_seqs, H, D, D, dtype=torch.float, device=device).requires_grad_(False)

    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        init_state=init_state,
        lower_bound=-5.0,
    )


def prepare_safe_gate_inputs_gqa(
    batch_size,
    T,
    num_heads,
    num_k_heads,
    D,
    device,
    cu_seqlens=None,
    chunk_size=CHUNK_SIZE,
    seed=SEED,
    has_init_state=False,
):
    """Prepare inputs for the "multi-value" GQA flavor of safe_gate KDA.

    Q and K use ``num_k_heads`` physical heads while V, g, beta, and state
    use the larger ``num_heads`` (= num_value_heads = state heads). This
    matches Qwen3.5-A3B Gated DeltaNet. Callers that need to compare
    against an MHA-only kernel are expected to ``repeat_interleave`` Q and
    K on their side — this helper intentionally does not pre-materialise
    the expansion so the cost stays visible in their timed block.
    """
    assert num_heads % num_k_heads == 0, f"num_heads ({num_heads}) must be a multiple of num_k_heads ({num_k_heads})"
    dtype = torch.bfloat16
    scale = D ** (-0.5)
    set_seed(seed)

    q = torch.randn(batch_size, T, num_k_heads, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(batch_size, T, num_k_heads, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(batch_size, T, num_heads, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(batch_size, T, num_heads, D, dtype=dtype, device=device).requires_grad_(False)
    beta = torch.randn(batch_size, T, num_heads, dtype=torch.float, device=device).sigmoid().requires_grad_(False)

    A_log = torch.randn(num_heads, dtype=torch.float, device=device).requires_grad_(False)
    dt_bias = torch.randn(num_heads * D, dtype=torch.float, device=device).requires_grad_(False)

    if batch_size != 1:
        q, k, v, g, beta = map(lambda x: rearrange(x, "b t ... -> 1 (b t) ..."), (q, k, v, g, beta))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    init_state = None
    if has_init_state:
        num_seqs = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch_size
        init_state = torch.randn(num_seqs, num_heads, D, D, dtype=torch.float, device=device).requires_grad_(False)

    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        init_state=init_state,
        lower_bound=-5.0,
    )


def prepare_intra_inputs(batch_size, T, H, D, device, cu_seqlens=None, chunk_size=CHUNK_SIZE, seed=SEED):
    """Prepare preprocessed inputs ready for chunk_kda_fwd_intra.

    All tensors are flattened to (1, B*T, ...) for cu_seqlens compatibility.
    """
    dtype = torch.bfloat16
    scale = D ** (-0.5)

    set_seed(seed)

    q = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    k = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    v = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    g_raw = torch.randn(batch_size, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid()

    # l2norm q, k
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    # flatten to batch_size=1 for cu_seqlens compatibility
    if batch_size != 1:
        q, k, v, g_raw, beta = map(lambda x: rearrange(x, "b t ... -> 1 (b t) ..."), (q, k, v, g_raw, beta))

    # gate preprocessing
    A_log = torch.randn(H, dtype=torch.float, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float, device=device)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    g = kda_gate_chunk_cumsum(
        g=g_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=-5.0,
    )

    return q, k, v, g, beta, scale, cu_seqlens, chunk_indices
