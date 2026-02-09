import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate
from fla.modules.l2norm import l2norm_fwd
from fla.utils import assert_close
from benchmark.utils import set_seed
from torch.profiler import profile, record_function, ProfilerActivity

from flashla.kda_wrapper import flash_kda_prefill

# Constant params
B, H, D = 1, 1, 128

S = T = 64*3

WARMUP_ITERATIONS = 0
ITERATIONS = 1

SEED = 42

# torch.set_printoptions(edgeitems=8)

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

def test_performance():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(SEED)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B

    # FIXME: support safe_gate=True
    use_gate_in_kernel = False
    safe_gate = False
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
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

def test_accuracy():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(SEED)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B

    # FIXME: support safe_gate=True
    use_gate_in_kernel = False
    safe_gate = True
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
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    set_seed(SEED)

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

# Stress test with random tensors
def test_random_stress(
    B: int = 2,
    T: int = 2048,
    H: int = 8,
    D: int = 128,
    n_repeat: int = 10000,
    use_gate_in_kernel: bool = True,
    safe_gate: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    initial_state = None,
    output_final_state: bool = False,
    lower_bound: float = -5.0,
):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    set_seed(SEED)

    seq_lens = [T] * B
    num_seqs = len(seq_lens) # TODO: support varlen
    assert num_seqs == B

    scale = D ** (-0.5)
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    beta = torch.randn(B, T, H, dtype=torch.float, device=device).sigmoid().requires_grad_(False)
    # cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int64, device=device)
    # init_state = torch.randn(B, H, D, D, dtype=torch.float, device=device)

    if use_gate_in_kernel:
      A_log = torch.randn(H, dtype=torch.float)
      dt_bias = torch.randn(H * D, dtype=torch.float)
      A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(False), (A_log, dt_bias))
    else:
      g = F.logsigmoid(g)

    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
    else:
        lower_bound = None

    ref_tri = None
    err_ratio_list = []
    for i in range(n_repeat):
        set_seed(SEED)
        tri, tri_ht = flash_kda_prefill(
            q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
            k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            A_log=(A_log.clone() if A_log is not None else None),
            dt_bias=(dt_bias.clone() if dt_bias is not None else None),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        )
        if i == 0:
           ref_tri = tri.clone().detach()
        err_ratio = get_err_ratio(tri, ref_tri)
        print(f"tri shape: {tri.shape}, has nan: {torch.isnan(tri).any()}")
        print(f"Iteration {i}: Relative error to first iter: {err_ratio:.6e}")
        err_ratio_list.append(err_ratio)

    # test if passed
    passed = True
    fail_diff = []
    for i in range(len(err_ratio_list)):
        if err_ratio_list[i] > 1e-8:
            passed = False
            fail_diff.append(err_ratio_list[i])
    if passed:
        print("PASSED")
    else:
        print("FAILED")
        fail_diff.sort(reverse=True)
        print(f"failed counts: {len(fail_diff)}")
        print("failed postitions", fail_diff)

# Stress test with dumped tensors
def test_dumped_stress(dump_path: str = "/tmp/kda_debug/dumped.pt"):
    from fla.ops.kda.gate import naive_kda_lowerbound_gate
    
    print(f"Loading dumped tensors from: {dump_path}")
    device = torch.device("cuda")
    data = torch.load(dump_path, map_location=device)
    n_repeat = 10000
    
    print("\n=== Loaded Tensors ===")
    for name, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{name}: shape={value.shape}, dtype={value.dtype}")
            print(f"  range: [{value.min():.6f}, {value.max():.6f}], nan: {torch.isnan(value).any()}")
        else:
            print(f"{name}: {value}")
    
    q = data["q"]
    k = data["k"]
    v = data["v"]
    g = data["g"]
    beta = data["beta"]
    A_log = data.get("A_log", None)
    dt_bias = data.get("dt_bias", None)
    scale = data.get("scale", q.shape[-1] ** (-0.5))
    use_gate_in_kernel = data.get("use_gate_in_kernel", False)
    use_qk_l2norm_in_kernel = data.get("use_qk_l2norm_in_kernel", False)
    safe_gate = data.get("safe_gate", False)
    lower_bound = data.get("lower_bound", None)
    output_final_state = data.get("output_final_state", False)
    initial_state = data.get("initial_state", None)

    print(f"\n=== Parameters ===")
    print(f"scale={scale}, use_gate_in_kernel={use_gate_in_kernel}, safe_gate={safe_gate}, lower_bound={lower_bound}")
    
    naive_kda_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    
    print("\n=== Running flash_kda_prefill ===")
    err_ratio_list = []
    ref_tri = None
    for i in range(n_repeat):
        set_seed(SEED)
        tri, tri_ht = flash_kda_prefill(
            q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
            k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            A_log=(A_log.clone() if A_log is not None else None),
            dt_bias=(dt_bias.clone() if dt_bias is not None else None),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        )
        if i == 0:
           ref_tri = tri.clone().detach()
        err_ratio = get_err_ratio(tri, ref_tri)
        print(f"tri shape: {tri.shape}, has nan: {torch.isnan(tri).any()}")
        print(f"Iteration {i}: Relative error to first iter: {err_ratio:.6e}")
        err_ratio_list.append(err_ratio)
    
    # test if passed
    passed = True
    fail_diff = []
    for i in range(len(err_ratio_list)):
        if err_ratio_list[i] > 1e-8:
            passed = False
            fail_diff.append(err_ratio_list[i])
    if passed:
        print("PASSED")
    else:
        print("FAILED")
        fail_diff.sort(reverse=True)
        print(f"failed counts: {len(fail_diff)}")
        print("failed diff", fail_diff)
    
    # print("\n=== Running naive_recurrent_kda ===")
    # ref, ref_ht = naive_recurrent_kda(
    #     q=F.normalize(q.clone(), p=2, dim=-1),
    #     k=F.normalize(k.clone(), p=2, dim=-1),
    #     v=v.clone(),
    #     g=(naive_kda_gate_fn(g.clone(), A_log, dt_bias) if use_gate_in_kernel else g.clone()),
    #     beta=beta.clone(),
    #     scale=scale,
    #     initial_state=initial_state,
    #     output_final_state=output_final_state,
    # )
    # print(f"ref shape: {ref.shape}, has nan: {torch.isnan(ref).any()}")
    # print(ref)

    # print("\n=== Accuracy ===")
    # abs_err = get_abs_err(tri, ref)
    # err_ratio = get_err_ratio(tri, ref)
    # print(f"Absolute error: {abs_err:.6e}, Relative error: {err_ratio:.6e}")
    
    # assert_close("o", ref, tri, 0.005)
    # print("PASSED!")

if __name__ == "__main__":
    test_accuracy()
    # test_dumped_stress("B4-T2048-H8-D128-scale0.1-gate_logit_normalizer1-mask_p0-qk_l2normFalse-gateTrue-dtypetorch.bfloat16-safe_gateTrue.pt")
    # test_random_stress()