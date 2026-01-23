import torch

import torch.nn.functional as F

from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import assert_close
from benchmark.utils import set_seed, exclusive_cumsum

import time
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack

from torch.profiler import profile, record_function, ProfilerActivity
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent))
# NOTE: update to your own build path
sys.path.append("/ossfs/workspace/kevinzeng/flashkda/build/python/torch")
# from python.torch import ops
from flashla.kda import KDAChunkwise

# Constant params
B, H, D = 1, 1, 128

S = T = 64

CHUNK_SIZE = 64

WARMUP_ITERATIONS = 0
ITERATIONS = 1

SEED = 42

torch.set_printoptions(edgeitems=8)

compiled_kernel = None

def cutedsl_kda_prefill(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    cu_seqlens,
    use_gate_in_kernel,
    safe_gate,
):
    assert safe_gate == False, "safe_gate=True is not supported in flash_kda_prefill yet."
    assert use_gate_in_kernel == False, "use_gate_in_kernel=True is not supported in cutedsl_kda_prefill yet."
    assert initial_state == None, "initial_state is not supported in cutedsl_kda_prefill yet."
    assert cu_seqlens == None, "cu_seqlens is not supported in cutedsl_kda_prefill yet."
    assert output_final_state == False, "output_final_state=True is not supported in cutedsl_kda_prefill yet."

    g = chunk_local_cumsum(
        g=g,
        chunk_size=CHUNK_SIZE,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        chunk_indices=None
    )

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    q_cute = from_dlpack(q)
    k_cute = from_dlpack(k)
    v_cute = from_dlpack(v)
    g_cute = from_dlpack(g)
    beta_cute = from_dlpack(beta)
    
    o = torch.zeros_like(q)
    o_cute = from_dlpack(o)

    # Get default stream
    stream = cutlass_torch.default_stream()

    global compiled_kernel

    if compiled_kernel is None:
        # Create kernel instance
        attn_kernel = KDAChunkwise(
            chunk_size=CHUNK_SIZE,
            qk_acc_dtype=cutlass.Float32,
            kv_acc_dtype=cutlass.Float32,
            io_dtype=cutlass.BFloat16,
            scale=scale,
        )

        start_time = time.time()
        compiled = cute.compile(
            attn_kernel,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            (B, S, H, D),
            stream,
        )
        compilation_time = time.time() - start_time
        print(f"Compilation time: {compilation_time:.4f} seconds")
        print(f"B, S, H, D: {(B, S, H, D)}")

        compiled_kernel = compiled

    compiled = compiled_kernel

    # Warmup
    # for _ in range(WARMUP_ITERATIONS):
    #     compiled(
    #         q_cute.iterator,
    #         k_cute.iterator,
    #         v_cute.iterator,
    #         g_cute.iterator,
    #         o_cute.iterator,
    #         beta_cute.iterator,
    #         (B, S, H, D),
    #         stream,
    #     )
    
    # Benchmark
    # torch.cuda.synchronize()
    # start = time.perf_counter()

    iterations = 1
    
    for _ in range(iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            (B, S, H, D),
            stream,
        )
    
    torch.cuda.synchronize()
    # elapsed = time.perf_counter() - start
    # print(f"\nExecution time: {elapsed*1000/iterations:.2f} ms (average over {iterations} iterations)")

    return o, None

def get_abs_err(x, y):
    return (x.detach()-y.detach()).flatten().abs().max().item()

def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)

def test_accuracy():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(SEED)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B

    # FIXME: use_gate=True causes NAN for lower_bound_gate=True setting
    use_gate_in_kernel = False
    output_final_state = False

    scale = D ** (-0.5)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    # cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    # init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    init_state = None
    cu_seqlens = None

    if use_gate_in_kernel:
      A_log = torch.randn(H, dtype=torch.float)
      dt_bias = torch.randn(H * D, dtype=torch.float)
      A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(False), (A_log, dt_bias))
    else:
      g = F.logsigmoid(g)

    o, final_states = None, None
    for _ in range(WARMUP_ITERATIONS):
        o, final_states = cutedsl_kda_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=(A_log.clone() if use_gate_in_kernel else None),
            dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
            scale=scale,
            initial_state=init_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=False
        )
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(ITERATIONS):
        o, final_states = cutedsl_kda_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=(A_log.clone() if use_gate_in_kernel else None),
            dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
            scale=scale,
            initial_state=init_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=False
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"\nFlashKDA Execution time: {elapsed*1000/ITERATIONS:.2f} ms (average over {ITERATIONS} iterations)")

    set_seed(SEED)

    o_fla = None
    final_states_fla = None

    for _ in range(WARMUP_ITERATIONS):
        o_fla, final_states_fla = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=(A_log.clone() if use_gate_in_kernel else None),
            dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
            scale=scale,
            initial_state=init_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            # safe_gate=True
            use_gate_in_kernel=use_gate_in_kernel,
        )

    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(ITERATIONS):
        o_fla, final_states_fla = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=(A_log.clone() if use_gate_in_kernel else None),
            dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
            scale=scale,
            initial_state=init_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            # safe_gate=True
            use_gate_in_kernel=use_gate_in_kernel,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"\nFLA Execution time: {elapsed*1000/ITERATIONS:.2f} ms (average over {ITERATIONS} iterations)")

    set_seed(42)
    q, q_rstd = l2norm_fwd(q)
    k, k_rstd = l2norm_fwd(k)
    # o_naive, final_states_naive = naive_chunk_kda(
    #     q=q,
    #     k=k,
    #     v=v,
    #     g=(naive_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
    #     beta=beta,
    #     scale=scale,
    #     initial_state=init_state,
    #     output_final_state=True,
    #     # use_gate_in_kernel=True,
    # )
    # NOTE: use naive_recurrent_kda as reference, same with FLA impl
    o_naive, final_states_naive = naive_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=(naive_kda_gate(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
        beta=beta.clone(),
        scale=scale,
        initial_state=init_state,
        output_final_state=output_final_state,
    )

    print(f"BEGIN CHECK PRECISION")

    print("o flashkda:", o)
    print("o naive:", o_naive)
    print("o fla:", o_fla)

    max_abs_diff = (o - o_naive).abs().max().item()
    print(f"Max absolute difference between flash_kda_prefill and chunk_kda outputs: {max_abs_diff}")

    abs_err = get_abs_err(o, o_naive)
    err_ratio = get_err_ratio(o, o_naive)
    print(f"Absolute error between flash_kda_prefill and naive_recurrent_kda outputs: {abs_err}")
    print(f"Relative error between flash_kda_prefill and naive_recurrent_kda outputs: {err_ratio}")

    abs_err = get_abs_err(o, o_fla)
    err_ratio = get_err_ratio(o, o_fla)
    print(f"Absolute error between flash_kda_prefill and fla outputs: {abs_err}")
    print(f"Relative error between flash_kda_prefill and fla outputs: {err_ratio}")

    # torch.testing.assert_close(o_naive, o_fla), "TORCH & FLA outputs do not match!"
    torch.testing.assert_close(o_naive, o), "TORCH & CUTELDS outputs do not match!"
    # torch.testing.assert_close(o_naive, o_fla), "TORCH & FLA outputs do not match!"
    # assert_close("O accuracy: naive vs. flashkda", o_naive, o, 1e-3)
    # assert_close("O accuracy: naive vs. fla", o_naive, o_fla, 1e-3)
    # assert_close("O accuracy: naive vs. flashkda", o_naive, o, 1e-3)

    print(f"TEST PASSED!")

    if output_final_state:
        assert_close("State accuracy: naive vs. fla", final_states_naive, final_states_fla, 1e5)
        assert_close("State accuracy: naive vs. flashkda", final_states_naive, final_states, 1e5)
        assert_close("State accuracy: fla vs. flashkda", final_states_fla, final_states, 1e5)

def test_accuracy_safe_gate():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(42)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B

    use_gate_in_kernel = False
    safe_gate = True
    # normalize for gate to test numerical stablity
    gate_logit_normalizer = 0.001

    scale = D ** (-0.5)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(B, T, H, D, dtype=torch.float if not use_gate_in_kernel else dtype, device=device).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    cu_seqlens = None
    # cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    # init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    init_state = None

    if use_gate_in_kernel:
      A_log = torch.randn(H, dtype=torch.float)
      dt_bias = torch.randn(H * D, dtype=torch.float)
      A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(False), (A_log, dt_bias))
    else:
      g = F.logsigmoid(g)
      g = g / gate_logit_normalizer

    if safe_gate:
        from fla.ops.kda.gate import naive_kda_lowerbound_gate
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
        naive_kda_gate_fn = naive_kda_lowerbound_gate
    else:
        lower_bound = None
        naive_kda_gate_fn = naive_kda_gate

    o, final_states = flash_kda_prefill(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        scale=scale,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound
    )

    set_seed(42)
    o_fla, final_states_fla = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        scale=scale,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    set_seed(42)
    q, q_rstd = l2norm_fwd(q)
    k, k_rstd = l2norm_fwd(k)
    # NOTE: use naive_recurrent_kda as reference, same with FLA impl
    o_naive, final_states_naive = naive_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=(naive_kda_gate_fn(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
        beta=beta.clone(),
        scale=scale,
        initial_state=init_state,
        output_final_state=True,
    )
    # import pdb;pdb.set_trace()
    print(f"FlashKDA out: {o}")
    print(f"Reference out: {o_naive}")
    print(f"FlashKDA ht: {final_states}")
    print(f"Reference ht: {final_states_naive}")

    # max_abs_diff = (o - o_naive).abs().max().item()
    # print(f"Max absolute difference between flash_kda_prefill and chunk_kda outputs: {max_abs_diff}")
    # torch.testing.assert_close(final_states, final_states_naive, atol=1e-4, rtol=1e-4), "Final States do not match!"
    # assert torch.testing.assert_close(o, o_naive, atol=1e-4, rtol=1e-4), "Outputs do not match!"
    assert_close("O accuracy: naive vs. fla", o_naive, o_fla, 1e5)
    assert_close("O accuracy: naive vs. flashkda", o_naive, o, 1e5)
    assert_close("O accuracy: fla vs. flashkda", o_fla, o, 1e5)

    assert_close("State accuracy: naive vs. fla", final_states_naive, final_states_fla, 1e5)
    assert_close("State accuracy: naive vs. flashkda", final_states_naive, final_states, 1e5)
    assert_close("State accuracy: fla vs. flashkda", final_states_fla, final_states, 1e5)

def test_performance():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(42)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B
    scale = D ** (-0.5)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    A_log = torch.randn(H, dtype=torch.float)
    dt_bias = torch.randn(H * D, dtype=torch.float)
    A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(False), (A_log, dt_bias))
    # init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    init_state = None

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1)
    ) as prof:
      for i in range(5):
        with record_function("flash_kda_prefill_step"):
          o, final_states = flash_kda_prefill(
              q=q,
              k=k,
              v=v,
              g=g,
              beta=beta,
              scale=scale,
              A_log=A_log,
              dt_bias=dt_bias,
              initial_state=init_state,
              output_final_state=True,
              use_qk_l2norm_in_kernel=True,
              cu_seqlens=cu_seqlens,
              use_gate_in_kernel=True,
          )
        prof.step()

      output_file = f"torch_profile_flash_kda_B_{B}_H{H}_T{T}_use_gate_in_kernel"
      prof.export_chrome_trace(f"{output_file}.json")

def test_performance_kernel_kda():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(42)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B
    scale = D ** (-0.5)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    cu_seqlens = None
    # cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    # init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    init_state = None

    use_qk_l2norm_in_kernel = True
    q_rstd, k_rstd = None, None
    if use_qk_l2norm_in_kernel:
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)
    chunk_size = 64
    g = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        chunk_indices=None
    )
    batch_size, seq_len, num_heads, head_dim = q.shape
    # q, k, v, g = map(lambda x: x.reshape(batch_size * seq_len, num_heads, head_dim).contiguous(), (q, k, v, g))
    # beta = beta.reshape(batch_size * seq_len, num_heads)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1)
    ) as prof:
      for i in range(5):
        with record_function("flash_kda_prefill_step"):
          ops.kda_fwd_prefill(
            None, None, q, k, v, None, g, beta, cu_seqlens, scale
          ),
          torch.cuda.synchronize()
        prof.step()

      output_file = f"torch_profile_flash_kda_kernel_B_{B}_H{H}_T{T}"
      prof.export_chrome_trace(f"{output_file}.json")

def test_performance_kernel():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(42)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B
    scale = D ** (-0.5)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    cu_seqlens = None
    # cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    # init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    init_state = None

    use_qk_l2norm_in_kernel = True
    q_rstd, k_rstd = None, None
    if use_qk_l2norm_in_kernel:
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)
    chunk_size = 64
    g = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        chunk_indices=None
    )
    batch_size, seq_len, num_heads, head_dim = q.shape
    # q, k, v, g = map(lambda x: x.reshape(batch_size * seq_len, num_heads, head_dim).contiguous(), (q, k, v, g))
    # beta = beta.reshape(batch_size * seq_len, num_heads)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1)
    ) as prof:
      for i in range(5):
        with record_function("flash_kda_prefill_step"):
          ops.kda_fwd_prefill(
            None, None, q, k, v, None, g, beta, cu_seqlens, scale
          ),
          torch.cuda.synchronize()
        prof.step()

      output_file = f"torch_profile_flash_kda_kernel_B_{B}_H{H}_T{T}"
      prof.export_chrome_trace(f"{output_file}.json")

def test_performance_fla():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(42)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B
    scale = D ** (-0.5)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    # init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    init_state = None

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1)
    ) as prof:
      for i in range(5):
        with record_function("flash_kda_prefill_step"):
          o_fla, final_states_fla = chunk_kda(
              q=q,
              k=k,
              v=v,
              g=g,
              beta=beta,
              scale=scale,
              initial_state=init_state,
              output_final_state=True,
              use_qk_l2norm_in_kernel=True,
              # safe_gate=True
              # use_gate_in_kernel=True,
          )
          torch.cuda.synchronize()
        prof.step()

      output_file = f"torch_profile_fla_B_{B}_H{H}_T{T}"
      prof.export_chrome_trace(f"{output_file}.json")

if __name__ == "__main__":
    test_accuracy()
    # test_accuracy_safe_gate()
    # test_performance()
    # test_performance_kernel()
    # test_performance_fla()
