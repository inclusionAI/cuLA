# FlashLA — Flash Linear Attention

FlashLA is a lightweight, high-performance linear attention library inspired by ideas from FlashAttention but specialized and engineered for linear attention mechanisms.

## Features

- O(N) time and memory approximate attention for long sequences (instead of O(N^2)).
- Supports causal (autoregressive) and non-causal modes.
- Numerical-stability controls (normalization, log-space ops, safe modes).

Installation

Install from PyPI (if published):

```bash
pip install flashla
```

Install editable from source:

```bash
cd /path/to/flashla
pip install -e .
```

## Quick start

WIP

## Performance and benchmarks

## Mathematical background (brief)

Linear attention rewrites the attention kernel using a feature map phi:

$$A_{ij} = \\phi(q_i)^T\\phi(k_j)$$

so the output can be expressed as

$$out_i = \\sum_j \\mathrm{softmax\\_approx}(q_i^T k_j) v_j \\approx \\phi(q_i)^T \\left(\\sum_j \\phi(k_j) v_j^T\\right)$$

This transforms the O(N^2) pairwise computation into two O(N) accumulation operations.
