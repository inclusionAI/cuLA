# FlashLA — Flash Linear Attention

FlashLA is a lightweight, high-performance linear attention library inspired by ideas from FlashAttention but specialized and engineered for linear attention mechanisms.

## Features

- O(N) time and memory approximate attention for long sequences (instead of O(N^2)).
- Supports causal (autoregressive) and non-causal modes.
- Numerical-stability controls (normalization, log-space ops, safe modes).

## Environment

We test and benchmark FlashLA on GB200 GPUs with Python 3.12, CUDA 12.9, PyTorch 2.9.1, Triton 3.5.1, and a specific commit of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/5da31d199456ee4004f70186f3391d309e26ca98).

## Installation

- Install from source

```bash
git clone git@code.alipay.com:ling/flashla.git -b rel/v0.1-rc1
git submodule update --init --recursive
# install torch first
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129
# if network error, try the following command
# pip install --trusted-host pypi.nvidia.cn --trusted-host pypi.nvidia.com --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129
# install flash-linear-attention
cd third_party/flash-linear-attention
pip install -e .
# install flashla
cd ../..
pip install -e . --no-build-isolation
```

- [TODO] From PyPI (if published)

```bash
pip install flashla
```

- [TODO] From a prebuilt local wheel (recommended if available)

## Quick start

We maintain a user-friendly interface that is compatible with flash-linear-attention (FLA), so adopting FlashLA only requires a one-line change.

### KDA

```python
# One-line import change
from flashla.kda.chunk import chunk_kda

# Same interface as FLA
o, ht = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta, scale=scale,
    A_log=A_log, 
    dt_bias=dt_bias,
    initial_state=init_state, 
    output_final_state=True,
    use_qk_l2norm_in_kernel=True, 
    cu_seqlens=cu_seqlens,
    use_gate_in_kernel=True, 
    safe_gate=True, 
    lower_bound=lower_bound,
)
o.backward(do)
```

**Notes:**
- We currently only support `safe_gate=True` due to algorithm advancements and its superior performance.
- We currently only support `beta` input as `float32` data type.
- It is highly recommended to compute `cu_seqlens` outside with `int32` data type and pass it in for optimal performance.
- We currently use FP16 precision for matrix inversion.

### Lightning [TODO]


## Performance and benchmarks

### KDA

Benchmarks for KDA (Kimi Delta Attention) run on a single NVIDIA GB200 GPU. The following tables show the execution time (in milliseconds) and the speedup of FlashLA compared to the baseline `flash-linear-attention`.

- Uniform Sequence Length (B=2, H=64, D=128)

| T | flash-linear-attention (ms) | flashla (ms) | Speedup |
|---|---------------------------|--------------|---------|
| 128 | 0.573 | 0.534 | **1.074x** |
| 256 | 0.548 | 0.521 | **1.052x** |
| 512 | 0.557 | 0.534 | **1.043x** |
| 1024 | 0.555 | 0.528 | **1.052x** |
| 2048 | 1.022 | 0.595 | **1.718x** |
| 4096 | 1.965 | 1.121 | **1.753x** |
| 8192 | 3.856 | 2.176 | **1.772x** |
| 16384 | 7.657 | 4.250 | **1.802x** |
| 32768 | 15.488 | 8.565 | **1.808x** |

- Varlen Sequence Length (NUM_SEQS=8, H=64, D=128)

| Total Length | flash-linear-attention (ms) | flashla (ms) | Speedup |
|--------------|---------------------------|--------------|---------|
| 4096 | 1.070 | 0.621 | **1.722x** |
| 8192 | 2.026 | 1.146 | **1.768x** |
| 16384 | 3.953 | 2.187 | **1.808x** |
| 32768 | 7.858 | 4.273 | **1.839x** |

To reproduce the benchmarks:
```bash
python benchmarks/bench_kda.py
```

## Tests

Tests for KDA (Kimi Delta Attention)

```bash
# e2e test for both forward and backward compared with FLA implementation
python -m pytest tests/test_kda_e2e_compare_fla.py

# e2e test compared with Naive KDA implementation
python -m pytest tests/test_kda.py 
```

## Mathematical background (brief)

Linear attention rewrites the attention kernel using a feature map $\phi$:

$$A_{ij} = \phi(q_i)^T \phi(k_j)$$

so the output can be expressed as

$$o_i = \sum_j \frac{\phi(q_i)^T \phi(k_j)}{\sum_{j'} \phi(q_i)^T \phi(k_{j'})} v_j = \frac{\phi(q_i)^T \sum_j \phi(k_j) v_j^T}{\phi(q_i)^T \sum_j \phi(k_j)}$$

For causal (autoregressive) linear attention:

$$o_i = \frac{\phi(q_i)^T S_i}{\phi(q_i)^T z_i}, \quad S_i = \sum_{j \le i} \phi(k_j) v_j^T, \quad z_i = \sum_{j \le i} \phi(k_j)$$

The recurrence $S_i = S_{i-1} + \phi(k_i) v_i^T$ enables $O(N)$ sequential computation, avoiding $O(N^2)$ pairwise interactions.
