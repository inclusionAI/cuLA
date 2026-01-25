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

S = T = 64*5

CHUNK_SIZE = 64

WARMUP_ITERATIONS = 0
ITERATIONS = 1

SEED = 42

# torch.set_printoptions(edgeitems=8)

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

def analyze_8x8_blocks(o_pred, o_ref, name=""):
    """
    分析 8x8 块的误差分布
    o_pred: [B, T, H, D] 预测输出
    o_ref: [B, T, H, D] 参考输出
    """
    B, T, H, D = o_pred.shape
    
    print(f"\n{'='*80}")
    print(f"8x8 Block Error Analysis: {name}")
    print(f"{'='*80}")
    
    # 计算每个 8x8 块的误差
    block_size = 8
    num_blocks_T = (T + block_size - 1) // block_size
    num_blocks_D = (D + block_size - 1) // block_size
    
    abs_errors = torch.zeros(num_blocks_T, num_blocks_D)
    rel_errors = torch.zeros(num_blocks_T, num_blocks_D)
    max_elem_rel_errors = torch.zeros(num_blocks_T, num_blocks_D)  # 最大元素级相对误差
    max_vals = torch.zeros(num_blocks_T, num_blocks_D)
    
    for i in range(num_blocks_T):
        for j in range(num_blocks_D):
            t_start = i * block_size
            t_end = min((i + 1) * block_size, T)
            d_start = j * block_size
            d_end = min((j + 1) * block_size, D)
            
            # 提取块
            block_pred = o_pred[:, t_start:t_end, :, d_start:d_end]
            block_ref = o_ref[:, t_start:t_end, :, d_start:d_end]
            
            # 计算绝对误差
            abs_diff = torch.abs(block_pred - block_ref)
            abs_errors[i, j] = abs_diff.max().item()
            
            # 计算相对误差 (norm-based)
            block_ref_norm = torch.norm(block_ref).item()
            block_diff_norm = torch.norm(abs_diff).item()
            rel_errors[i, j] = (block_diff_norm / (block_ref_norm + 1e-10)) * 100
            
            # 计算最大元素级相对误差
            elem_rel_errors = abs_diff / (torch.abs(block_ref) + 1e-10)
            max_elem_rel_errors[i, j] = elem_rel_errors.max().item() * 100
            
            # 记录参考值的最大值
            max_vals[i, j] = torch.abs(block_ref).max().item()
    
    # 打印绝对误差矩阵
    print(f"\nAbsolute Error per 8x8 Block (Max value in each block):")
    print(f"Block grid: {num_blocks_T} x {num_blocks_D}")
    print(f"{'Seq':>6s}", end="")
    for j in range(num_blocks_D):
        print(f" D{j*8:03d}-{min((j+1)*8-1, D-1):03d}", end="")
    print()
    
    for i in range(num_blocks_T):
        print(f"T{i*8:03d}-{min((i+1)*8-1, T-1):03d}", end="")
        for j in range(num_blocks_D):
            val = abs_errors[i, j]
            if val > 1e-2:
                print(f" \033[91m{val:9.2e}\033[0m", end="")  # Red for high error
            elif val > 1e-3:
                print(f" \033[93m{val:9.2e}\033[0m", end="")  # Yellow for medium error
            else:
                print(f" {val:9.2e}", end="")  # Normal for low error
        print()
    
    # 打印相对误差矩阵 (norm-based)
    print(f"\nRelative Error per 8x8 Block (norm-based, %):")
    print(f"{'Seq':>6s}", end="")
    for j in range(num_blocks_D):
        print(f" D{j*8:03d}-{min((j+1)*8-1, D-1):03d}", end="")
    print()
    
    for i in range(num_blocks_T):
        print(f"T{i*8:03d}-{min((i+1)*8-1, T-1):03d}", end="")
        for j in range(num_blocks_D):
            val = rel_errors[i, j]
            if val > 10.0:
                print(f" \033[91m{val:9.2f}\033[0m", end="")  # Red for >10%
            elif val > 1.0:
                print(f" \033[93m{val:9.2f}\033[0m", end="")  # Yellow for >1%
            else:
                print(f" {val:9.2f}", end="")
        print()
    
    # 打印最大元素级相对误差矩阵
    print(f"\nMax Element-wise Relative Error per 8x8 Block (%):")
    print(f"{'Seq':>6s}", end="")
    for j in range(num_blocks_D):
        print(f" D{j*8:03d}-{min((j+1)*8-1, D-1):03d}", end="")
    print()
    
    for i in range(num_blocks_T):
        print(f"T{i*8:03d}-{min((i+1)*8-1, T-1):03d}", end="")
        for j in range(num_blocks_D):
            val = max_elem_rel_errors[i, j]
            if val > 10.0:
                print(f" \033[91m{val:9.2f}\033[0m", end="")  # Red for >10%
            elif val > 1.0:
                print(f" \033[93m{val:9.2f}\033[0m", end="")  # Yellow for >1%
            else:
                print(f" {val:9.2f}", end="")
        print()
    
    # 找出误差最大的块
    max_abs_idx = torch.argmax(abs_errors)
    max_abs_i = max_abs_idx // num_blocks_D
    max_abs_j = max_abs_idx % num_blocks_D
    
    max_rel_idx = torch.argmax(rel_errors)
    max_rel_i = max_rel_idx // num_blocks_D
    max_rel_j = max_rel_idx % num_blocks_D
    
    max_elem_rel_idx = torch.argmax(max_elem_rel_errors)
    max_elem_rel_i = max_elem_rel_idx // num_blocks_D
    max_elem_rel_j = max_elem_rel_idx % num_blocks_D
    
    print(f"\n{'='*80}")
    print(f"Block with Maximum Absolute Error:")
    print(f"  Position: T[{max_abs_i*8}:{min((max_abs_i+1)*8, T)}], D[{max_abs_j*8}:{min((max_abs_j+1)*8, D)}]")
    print(f"  Max abs error: {abs_errors[max_abs_i, max_abs_j]:.6e}")
    print(f"  Norm-based rel error: {rel_errors[max_abs_i, max_abs_j]:.2f}%")
    print(f"  Max elem rel error: {max_elem_rel_errors[max_abs_i, max_abs_j]:.2f}%")
    print(f"  Max ref value: {max_vals[max_abs_i, max_abs_j]:.6e}")
    
    print(f"\nBlock with Maximum Norm-based Relative Error:")
    print(f"  Position: T[{max_rel_i*8}:{min((max_rel_i+1)*8, T)}], D[{max_rel_j*8}:{min((max_rel_j+1)*8, D)}]")
    print(f"  Max abs error: {abs_errors[max_rel_i, max_rel_j]:.6e}")
    print(f"  Norm-based rel error: {rel_errors[max_rel_i, max_rel_j]:.2f}%")
    print(f"  Max elem rel error: {max_elem_rel_errors[max_rel_i, max_rel_j]:.2f}%")
    print(f"  Max ref value: {max_vals[max_rel_i, max_rel_j]:.6e}")
    
    print(f"\nBlock with Maximum Element-wise Relative Error:")
    print(f"  Position: T[{max_elem_rel_i*8}:{min((max_elem_rel_i+1)*8, T)}], D[{max_elem_rel_j*8}:{min((max_elem_rel_j+1)*8, D)}]")
    print(f"  Max abs error: {abs_errors[max_elem_rel_i, max_elem_rel_j]:.6e}")
    print(f"  Norm-based rel error: {rel_errors[max_elem_rel_i, max_elem_rel_j]:.2f}%")
    print(f"  Max elem rel error: {max_elem_rel_errors[max_elem_rel_i, max_elem_rel_j]:.2f}%")
    print(f"  Max ref value: {max_vals[max_elem_rel_i, max_elem_rel_j]:.6e}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean abs error across blocks: {abs_errors.mean():.6e}")
    print(f"  Max abs error across blocks: {abs_errors.max():.6e}")
    print(f"  Mean norm-based rel error across blocks: {rel_errors.mean():.2f}%")
    print(f"  Max norm-based rel error across blocks: {rel_errors.max():.2f}%")
    print(f"  Mean max elem rel error across blocks: {max_elem_rel_errors.mean():.2f}%")
    print(f"  Max elem rel error across blocks: {max_elem_rel_errors.max():.2f}%")
    print(f"{'='*80}\n")

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

    # print("o flashkda:", o)
    print("o flashkda:", o)
    print("o naive:", o_naive)
    # print("o fla:", o_fla)

    # Analyze 8x8 block errors
    # analyze_8x8_blocks(o, o_naive, "FlashKDA vs Naive")
    # analyze_8x8_blocks(o_fla, o_naive, "FLA vs Naive")

    abs_err = get_abs_err(o, o_naive)
    err_ratio = get_err_ratio(o, o_naive)
    print(f"Absolute error between flash_kda_prefill and naive_recurrent_kda outputs: {abs_err}")
    print(f"Relative error between flash_kda_prefill and naive_recurrent_kda outputs: {err_ratio}")

    abs_err = get_abs_err(o_naive, o_fla)
    err_ratio = get_err_ratio(o_naive, o_fla)
    print(f"Absolute error between naive and fla outputs: {abs_err}")
    print(f"Relative error between naive and fla outputs: {err_ratio}")

    # torch.testing.assert_close(o_naive, o_fla), "TORCH & FLA outputs do not match!"
    # torch.testing.assert_close(o_naive, o, atol=1e-5, rtol=5e-3), "TORCH & CUTELDS outputs do not match!"
    # torch.testing.assert_close(o_naive, o_fla), "TORCH & FLA outputs do not match!"
    # assert_close("O accuracy: naive vs. flashkda", o_naive, o, 1e-3)
    # assert_close("O accuracy: naive vs. fla", o_naive, o_fla, 1e-3)
    # assert_close("O accuracy: naive vs. flashkda", o_naive, o, 1e-3)

    print(f"TEST PASSED!")

    if output_final_state:
        assert_close("State accuracy: naive vs. fla", final_states_naive, final_states_fla, 1e5)
        assert_close("State accuracy: naive vs. flashkda", final_states_naive, final_states, 1e5)
        assert_close("State accuracy: fla vs. flashkda", final_states_fla, final_states, 1e5)

if __name__ == "__main__":
    test_accuracy()