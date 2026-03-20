<div align="center">

# cuLA — CUDA Linear Attention

**High-performance CUDA kernels for linear attention variants, written in [CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL) and CUTLASS C++.**

</div>

## Introduction

Linear attention mechanisms reformulate the standard attention computation by replacing the softmax with a feature map $\phi$, enabling the output to be expressed via a recurrent state:

$$S_i = S_{i-1} + \phi(k_i) v_i^T, \quad o_i = \phi(q_i)^T S_i$$

This recurrence reduces the complexity from $O(N^2)$ (standard attention) to $O(N)$, making linear attention particularly attractive for long-sequence modeling in LLMs. Recent variants — such as [GLA](https://arxiv.org/abs/2312.06635), [Delta Rule](https://arxiv.org/abs/2406.06484), [GDN](https://arxiv.org/abs/2505.18788), and [Lightning Attention](https://arxiv.org/abs/2405.17381) — further enhance expressiveness with gating, delta updates, and chunkwise decomposition.

**cuLA** provides hand-tuned CUDA implementations of these linear attention variants, targeting NVIDIA Blackwell (SM100) and Hopper (SM90) GPUs. It is designed as a submodule of [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention), sharing the same interface — adopting cuLA requires only a one-line import change.

## Installation

```bash
git clone <repo-url>
git submodule update --init --recursive

# Install PyTorch (CUDA 13.0)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Install flash-linear-attention
cd third_party/flash-linear-attention
pip install -e .
cd ../..

# Install cuLA
pip install -e . --no-build-isolation
```

> **Requirements:** Python 3.12+, CUDA Toolkit 13.0, NVCC 12.9+ (SM100a support), PyTorch 2.9.1+

## Quick Start

### KDA (Kimi Delta Attention)

cuLA is a drop-in replacement for [FLA](https://github.com/fla-org/flash-linear-attention) — just change the import:

```python
import torch
from flashla.kda.chunk import chunk_kda  # <-- one-line change from fla.ops.kda

B, T, H, K, V = 2, 2048, 4, 128, 128
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1   # gate (log space)
beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

# Forward
o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=1.0,
    A_log=A_log,
    dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)

# Backward
do = torch.randn_like(o)
o.backward(do)

print(f'Output shape: {o.shape}')             # [2, 2048, 4, 128]
print(f'Final state shape: {final_state.shape}')  # [2, 4, 128, 128]
```

**Notes:**
- `safe_gate=True` is required (leverages M=16 TensorCore acceleration).
- `beta` and `initial_state` must be `float32`.
- `cu_seqlens` (for variable-length sequences) must be `int32`.
- Matrix inversion uses FP16 precision.

## Benchmarks

All benchmarks run on a single **NVIDIA GB200** GPU with **CUDA Toolkit 13.0**, **PyTorch 2.9.1**, **Triton 3.5.1**.

FLA baseline: [flash-linear-attention@5da31d19](https://github.com/fla-org/flash-linear-attention/tree/5da31d199456ee4004f70186f3391d309e26ca98).

### KDA — Fixed-Length (H=64, D=128, bf16)

| B | T | FLA Triton (ms) | cuLA (ms) | Speedup |
|---|---|-----------------|-----------|---------|
| 1 | 128 | 0.557 | 0.519 | **1.07x** |
| 1 | 256 | 0.542 | 0.511 | **1.06x** |
| 1 | 512 | 0.540 | 0.507 | **1.06x** |
| 1 | 1024 | 0.534 | 0.505 | **1.06x** |
| 1 | 2048 | 0.550 | 0.510 | **1.08x** |
| 1 | 4096 | 0.881 | 0.597 | **1.48x** |
| 1 | 8192 | 1.683 | 1.126 | **1.50x** |
| 1 | 16384 | 3.295 | 2.180 | **1.51x** |
| 2 | 128 | 0.562 | 0.516 | **1.09x** |
| 2 | 256 | 0.553 | 0.517 | **1.07x** |
| 2 | 512 | 0.547 | 0.514 | **1.06x** |
| 2 | 1024 | 0.537 | 0.508 | **1.06x** |
| 2 | 2048 | 0.819 | 0.598 | **1.37x** |
| 2 | 4096 | 1.560 | 1.129 | **1.38x** |
| 2 | 8192 | 3.044 | 2.189 | **1.39x** |
| 2 | 16384 | 6.020 | 4.336 | **1.39x** |

### KDA — Variable-Length (H=64, D=128, bf16)

| Config | FLA Triton (ms) | cuLA (ms) | Speedup |
|--------|-----------------|-----------|---------|
| 1seq, T=4096 | 0.881 | 0.597 | **1.48x** |
| 1seq, T=8192 | 1.723 | 1.166 | **1.48x** |
| 1seq, T=16384 | 3.306 | 2.197 | **1.50x** |
| 20seqs, T=4096, uniform | 0.898 | 0.679 | **1.32x** |
| 20seqs, T=8192, uniform | 1.604 | 1.190 | **1.35x** |
| 20seqs, T=16384, uniform | 3.057 | 2.235 | **1.37x** |
| 20seqs, T=4096, skewed | 0.866 | 0.616 | **1.41x** |
| 20seqs, T=8192, skewed | 1.693 | 1.137 | **1.49x** |
| 20seqs, T=16384, skewed | 3.363 | 2.189 | **1.54x** |

To reproduce:

```bash
python benchmarks/bench_kda.py --mode both
```

<details>
<summary>Sample output</summary>

```
                       BENCHMARK REPORT: chunk_kda
                       flashla CuTe DSL vs FLA Triton
                       H=64  D=128  dtype=bf16  safe_gate=True
                       Warmup=10  Iters=100
================================================================================
  [Fixed-Length]
  ────────────────────────────────────────────────────────────────────────────
    B      T  │        RMSE     rel_max     mean_diff  │    FLA(ms)  flashla(ms)   Speedup
  ────────────────────────────────────────────────────────────────────────────
    1    128  │    0.000003    0.003311    0.00000023  │     0.5571       0.5194     1.07x
    1   1024  │    0.000003    0.006061    0.00000021  │     0.5343       0.5049     1.06x
    1   4096  │    0.000004    0.005000    0.00000042  │     0.8809       0.5966     1.48x
    1   8192  │    0.000003    0.005376    0.00000028  │     1.6833       1.1257     1.50x
    1  16384  │    0.000004    0.004717    0.00000034  │     3.2949       2.1799     1.51x
    2   4096  │    0.000003    0.005376    0.00000028  │     1.5602       1.1286     1.38x
    2  16384  │    0.000003    0.004717    0.00000030  │     6.0200       4.3357     1.39x
  ...

  [Varlen]
  ────────────────────────────────────────────────────────────────────────────
    1seqs T=16384           │     3.3063       2.1971     1.50x
    20seqs T=8192           │     1.6041       1.1896     1.35x
    20seqs T=16384          │     3.3627       2.1891     1.54x
  ...
```

</details>

## Tests

```bash
# End-to-end test (forward + backward) against FLA Triton implementation
python -m pytest tests/test_kda_e2e_compare_fla.py -v

# End-to-end test against naive KDA reference
python -m pytest tests/test_kda.py -v
```

<details>
<summary>Sample test output</summary>

```
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B1-T63-H1-D128-...]    PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B2-T500-H3-D128-...]   PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B2-T1000-H3-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B3-T1024-H4-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B4-T1024-H4-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B4-T2048-H8-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk_varlen[...]             PASSED
...
======================= 17 passed in 40.95s =======================
```

</details>

CUDA kernel tuning is significantly more labor-intensive than Triton — contributions from the open-source community are warmly welcomed!

## Status & Roadmap

### Status

- [x] **Modular KDA Forward (Blackwell)** — compatible with Context Parallelism (CP)
  - [x] `chunk_intra_subchunk`
  - [x] `chunk_gated_delta_h`
  - [x] `chunk_fwd_o`
- [x] **Fused KDA Forward (Hopper)**
- [x] **Fused Lightning Prefill (Blackwell)**
- [x] **Lightning Decode (Hopper & Blackwell)**

### Roadmap

[ ] Integrate into [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) via FLA's kernel dispatch mechanism
[ ] GDN Modular Forward / Backward (compatible with Kimi CP)
[ ] Fully fused blackwell KDA prefilling
[ ] kernel-level comm & compute overlapping (via nvshmem)
[ ] More aggressive fusion of small neighbor kernels like 

## Acknowledgements

This project is inspired by [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and [CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL). We thank [FLA-org](https://github.com/fla-org) and NVIDIA for their great work.

## Contact

If you're interested in an internship or job opportunity, feel free to reach out: **shuyan.ycf@antgroup.com**  / **chaofanyu@gmail.com**
