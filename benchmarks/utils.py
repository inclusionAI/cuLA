import functools
import random

import torch
import torch.nn.functional as F
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices

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

def generate_random_seq_lens(
    num_seqs: int, 
    total_len: int, 
    min_seq_len: int, 
    variance: float = 1.0,
    seed: int = 42
) -> list:
    """
    生成随机的序列长度列表，满足：
    - 序列数量为 num_seqs
    - 总长度为 total_len
    - 每个序列长度 >= min_seq_len
    - variance: 方差控制参数
        - 0.0: 完全均衡，所有序列长度尽可能相等
        - 1.0: 正常随机分配
        - >1.0: 更不均衡，序列长度差异更大
    """
    assert total_len >= num_seqs * min_seq_len, \
        f"total_len ({total_len}) must be >= num_seqs ({num_seqs}) * min_seq_len ({min_seq_len})"
    
    random.seed(seed)
    
    # 计算均衡情况下每个序列的长度
    base_len = total_len // num_seqs
    remainder = total_len % num_seqs
    
    if variance == 0.0:
        # 完全均衡分配
        seq_lens = [base_len] * num_seqs
        # 将余数分配给前几个序列
        for i in range(remainder):
            seq_lens[i] += 1
    else:
        # 先给每个序列分配最小长度
        seq_lens = [min_seq_len] * num_seqs
        remaining = total_len - num_seqs * min_seq_len
        
        if remaining > 0:
            if variance >= 1.0:
                # 高方差：使用 Dirichlet 分布生成权重
                # alpha 越小，分布越不均匀
                alpha = 1.0 / variance
                weights = [random.gammavariate(alpha, 1.0) for _ in range(num_seqs)]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # 按权重分配剩余长度
                extra_lens = [int(remaining * w) for w in weights]
                # 处理舍入误差
                diff = remaining - sum(extra_lens)
                for i in range(abs(diff)):
                    idx = random.randint(0, num_seqs - 1)
                    extra_lens[idx] += 1 if diff > 0 else -1
                
                for i in range(num_seqs):
                    seq_lens[i] += extra_lens[i]
            else:
                # 低方差 (0 < variance < 1)：在均衡和随机之间插值
                # 先计算均衡分配
                balanced = [base_len] * num_seqs
                for i in range(remainder):
                    balanced[i] += 1
                
                # 计算随机分配
                random_lens = [min_seq_len] * num_seqs
                for _ in range(remaining):
                    idx = random.randint(0, num_seqs - 1)
                    random_lens[idx] += 1
                
                # 按 variance 插值
                seq_lens = [
                    int(balanced[i] * (1 - variance) + random_lens[i] * variance)
                    for i in range(num_seqs)
                ]
                # 修正总长度
                diff = total_len - sum(seq_lens)
                for i in range(abs(diff)):
                    idx = i % num_seqs
                    seq_lens[idx] += 1 if diff > 0 else -1
    
    # 确保所有序列长度 >= min_seq_len
    for i in range(num_seqs):
        if seq_lens[i] < min_seq_len:
            deficit = min_seq_len - seq_lens[i]
            seq_lens[i] = min_seq_len
            # 从其他序列借用
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
# Common input preparation functions for benchmarks and demos
# ==============================================================================

def prepare_safe_gate_inputs(batch_size, T, H, D, device, cu_seqlens=None, chunk_size=CHUNK_SIZE, seed=SEED):
    """Prepare inputs for safe_gate benchmarks (use_gate_in_kernel=True, safe_gate=True).

    All tensors are flattened to (1, B*T, ...) for cu_seqlens compatibility.
    """
    dtype = torch.bfloat16
    scale = D ** (-0.5)

    set_seed(seed)

    q = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(batch_size, T, H, D, dtype=torch.float, device=device).requires_grad_(False)
    beta = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)

    A_log = torch.randn(H, dtype=torch.float, device=device).requires_grad_(False)
    dt_bias = torch.randn(H * D, dtype=torch.float, device=device).requires_grad_(False)

    # flatten to batch_size=1 for cu_seqlens compatibility
    if batch_size != 1:
        q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    return dict(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        init_state=None, lower_bound=-5.0,
    )

def prepare_no_gate_inputs(batch_size, T, H, D, device, cu_seqlens=None, chunk_size=CHUNK_SIZE, seed=SEED):
    """Prepare inputs for use_gate_in_kernel=False benchmarks.

    All tensors are flattened to (1, B*T, ...) for cu_seqlens compatibility.
    """
    dtype = torch.bfloat16
    scale = D ** (-0.5)

    set_seed(seed)

    q = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(torch.randn(batch_size, T, H, D, dtype=torch.float, device=device)).requires_grad_(False)
    beta = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)

    # flatten to batch_size=1 for cu_seqlens compatibility
    if batch_size != 1:
        q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    return dict(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )


def prepare_kernel_inputs(batch_size, T, H, D, device, cu_seqlens=None, chunk_size=CHUNK_SIZE, seed=SEED):
    """Prepare inputs for kernel-level benchmarks (l2norm pre-applied).

    All tensors are flattened to (1, B*T, ...) for cu_seqlens compatibility.
    """
    dtype = torch.bfloat16
    scale = D ** (-0.5)

    set_seed(seed)

    q = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(batch_size, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(torch.randn(batch_size, T, H, D, dtype=dtype, device=device)).requires_grad_(False)
    beta = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)

    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    # flatten to batch_size=1 for cu_seqlens compatibility
    if batch_size != 1:
        q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    return dict(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
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
    g_raw = torch.randn(batch_size, T, H, D, dtype=torch.float, device=device)
    beta = torch.randn(batch_size, T, H, dtype=torch.float, device=device).sigmoid()

    # l2norm q, k
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    # flatten to batch_size=1 for cu_seqlens compatibility
    if batch_size != 1:
        q, k, v, g_raw, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g_raw, beta))

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
