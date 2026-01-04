# Copyright (c) 2024-2025 FlashLA Authors
# Licensed under the MIT License

"""
PyTorch Reference Implementations for Linear Attention

This package provides baseline PyTorch implementations of linear attention
algorithms for accuracy verification and benchmarking against optimized kernels.

Available implementations:
- naive_linear_attn_decay: O(n²) reference implementation
- naive_linear_attn_decay_recurrent: O(n) sequential RNN-style implementation
- chunkwise_linear_attn_decay: O(n) chunkwise parallel implementation
- chunkwise_linear_attn_decay_parallel: Fully parallel chunkwise implementation
- linear_attn_decay: Alias for recommended implementation

FLA-compatible implementations (BTHD layout):
- naive_linear_attn_fla_style: FLA-style with g_gamma parameter
- naive_linear_attn_decay_bthd: Our style with s parameter (BTHD layout)
"""

from .linear_attn_decay import (
    naive_linear_attn_decay,
    naive_linear_attn_decay_recurrent,
    chunkwise_linear_attn_decay,
    chunkwise_linear_attn_decay_parallel,
    linear_attn_decay,
)

from .compare_fla import (
    naive_linear_attn_fla_style,
    naive_linear_attn_decay_bthd,
)

__all__ = [
    # BHND layout (our primary)
    "naive_linear_attn_decay",
    "naive_linear_attn_decay_recurrent", 
    "chunkwise_linear_attn_decay",
    "chunkwise_linear_attn_decay_parallel",
    "linear_attn_decay",
    # BTHD layout (FLA-compatible)
    "naive_linear_attn_fla_style",
    "naive_linear_attn_decay_bthd",
]

__all__ = [
    "naive_linear_attn_decay",
    "naive_linear_attn_decay_recurrent", 
    "chunkwise_linear_attn_decay",
    "chunkwise_linear_attn_decay_parallel",
    "linear_attn_decay",
]
