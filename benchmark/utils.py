import functools
import random

import torch
import torch.distributions as dist

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
    assert all(s >= min_seq_len for s in seq_lens), f"Some seq_len < min_seq_len"
    
    return seq_lens
