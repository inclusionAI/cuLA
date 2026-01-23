__version__ = "0.1.0"

from flashla.flashla_interface import lightning_prefill_fwd

# CuTe DSL-based linear attention (Blackwell SM100)
from flashla.linear_attn import LinearAttentionChunkwise
from flashla.lightning_attn import LinearAttentionChunkwiseDecay

# Matrix inversion kernel
from flashla.inv import MatrixInverse64x64

__all__ = [
    "lightning_prefill_fwd",
    "LinearAttentionChunkwise",
    "LinearAttentionChunkwiseDecay",
    "MatrixInverse64x64",
]
