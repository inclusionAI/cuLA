import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import math

import torch
import torch.nn.functional as F

from einops import rearrange
from fla.ops.kda import chunk_kda as fla_chunk_kda
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.utils import assert_close
from benchmark.utils import set_seed, exclusive_cumsum
from torch.profiler import profile, record_function, ProfilerActivity

from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as fla_chunk_kda_fwd_intra

from flashla.kda.chunk import chunk_kda as flashla_chunk_kda
from flashla.kda.chunk_intra import chunk_kda_fwd_intra as flat_chunk_kda_fwd_intra

# Constant params
B, H, D = 1, 1, 128
T = 128
BT = 64  # chunk size

def prepare_intra_inputs(batch_size, T, H, D, device, cu_seqlens=None):
    """Prepare preprocessed inputs ready for chunk_kda_fwd_intra.

    All tensors are flattened to (1, B*T, ...) for cu_seqlens compatibility.
    """
    dtype = torch.bfloat16
    chunk_size = BT
    scale = D ** (-0.5)

    set_seed(42)

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

def test_kda_chunk_intra():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)

    flat_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=BT, chunk_indices=chunk_indices,
                safe_gate=True,)

    fla_chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta,
        scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=BT, chunk_indices=chunk_indices,
        safe_gate=True,
    )

if __name__ == "__main__":
    test_kda_chunk_intra()