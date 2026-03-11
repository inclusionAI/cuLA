import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda as fla_chunk_kda
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.utils import assert_close
from benchmarks.utils import set_seed, exclusive_cumsum, prepare_intra_inputs, prepare_safe_gate_inputs, SEED
from torch.profiler import profile, record_function, ProfilerActivity

from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as fla_chunk_kda_fwd_intra
from fla.ops.kda import chunk_kda as fla_chunk_kda

from flashla.kda.chunk import chunk_kda as flashla_chunk_kda
from flashla.kda.chunk_intra import chunk_kda_fwd_intra as flat_chunk_kda_fwd_intra
from flashla.kda_wrapper import flash_kda_prefill as flashla_fully_fused_kda

# Constant params
B, H, D = 2, 64, 128
T = 500
BT = 64  # chunk size

def test_kda_chunk_intra():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)

    set_seed(SEED)
    w, u, qg, kg, Aqk, Akk = flat_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=BT, chunk_indices=chunk_indices,
                safe_gate=True,)

    set_seed(SEED)
    w_fla, u_fla, qg_fla, kg_fla, Aqk_fla, Akk_fla = fla_chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta,
        scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=BT, chunk_indices=chunk_indices,
        safe_gate=True,
    )

    # assert error because of empty init of Aqk in FLA
    # assert_close("Aqk: fla vs. flashla", Aqk_fla, Aqk, 0.005)
    assert_close("Akk: fla vs. flashla", Akk_fla, Akk, 0.005)
    assert_close("w: fla vs. flashla", w_fla, w, 0.005)
    assert_close("u: fla vs. flashla", u_fla, u, 0.005)
    assert_close("kg: fla vs. flashla", kg_fla, kg, 0.005)

def test_chunk_kda():
    device = torch.device("cuda")
    seq_lens = [T] * B
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    inputs = prepare_safe_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)
    q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
    A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
    scale, init_state, lower_bound = inputs['scale'], inputs['init_state'], inputs['lower_bound']
    chunk_indices = inputs['chunk_indices']

    set_seed(SEED)
    o, final_state = flashla_fully_fused_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        scale=scale, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        initial_state=init_state, output_final_state=True,
        use_gate_in_kernel=True,
        safe_gate=True, use_qk_l2norm_in_kernel=True,
        lower_bound=lower_bound,
    )

    set_seed(SEED)
    o_fla, final_state_fla = fla_chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        scale=scale, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        initial_state=init_state, output_final_state=True,
        use_gate_in_kernel=True,
        safe_gate=True, use_qk_l2norm_in_kernel=True,
        lower_bound=lower_bound,
    )
    # NOTE: >0.005 precision loss
    assert_close("O: fla vs. flashla", o_fla, o, 0.05)
    assert_close("ht: fla vs. flashla", final_state_fla, final_state, 0.005)

def test_chunk_kda_varlen():
    device = torch.device("cuda")
    cu_seqlens = [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192]
    T = cu_seqlens[-1]
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    # NOTE: cu_seqlens must be int32 for FlashLA CUDA Impl
    cu_seqlens = cu_seqlens.to(torch.int32)
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens)

    set_seed(SEED)
    w, u, qg, kg, Aqk, Akk = flat_chunk_kda_fwd_intra(
                q=q, k=k, v=v, gk=g, beta=beta,
                scale=scale, cu_seqlens=cu_seqlens,
                chunk_size=BT, chunk_indices=chunk_indices,
                safe_gate=True,)

    set_seed(SEED)
    w_fla, u_fla, qg_fla, kg_fla, Aqk_fla, Akk_fla = fla_chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta,
        scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=BT, chunk_indices=chunk_indices,
        safe_gate=True,
    )

    # assert error because of empty init of Aqk in FLA
    # assert_close("Aqk: fla vs. flashla", Aqk_fla, Aqk, 0.005)
    assert_close("Akk: fla vs. flashla", Akk_fla, Akk, 0.005)
    assert_close("w: fla vs. flashla", w_fla, w, 0.005)
    assert_close("u: fla vs. flashla", u_fla, u, 0.005)
    assert_close("kg: fla vs. flashla", kg_fla, kg, 0.005)

if __name__ == "__main__":
    # test_kda_chunk_intra()
    # test_chunk_kda()
    test_chunk_kda_varlen()