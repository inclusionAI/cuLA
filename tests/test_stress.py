import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import math

from fla.utils import get_err_ratio, get_abs_err, assert_close
from fla.ops.kda import chunk_kda

from benchmarks.utils import (
    prepare_safe_gate_inputs, set_seed, SEED, exclusive_cumsum
)

from flashla.kda.chunk import chunk_kda as flashla_chunk_kda

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
    initial_state=None,
    output_final_state: bool = True,
    lower_bound: float = -5.0,
):
    device = torch.device("cuda")
    # seq_lens = [T] * B
    cu_seqlens = [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192]
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    inputs = prepare_safe_gate_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens, seed=SEED)
    q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
    cu_seqlens = inputs['cu_seqlens']
    A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
    scale, lower_bound = inputs['scale'], inputs['lower_bound']

    ref_tri = None
    ref_tri_ht = None
    err_ratio_list = []
    err_ratio_ht_list = []
    for i in range(n_repeat):
        set_seed(SEED)
        tri, tri_ht = flashla_chunk_kda(
            q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
            k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
            v=v.clone(), g=g.clone(), beta=beta.clone(),
            A_log=(A_log.clone() if A_log is not None else None),
            dt_bias=(dt_bias.clone() if dt_bias is not None else None),
            scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=use_gate_in_kernel, safe_gate=safe_gate,
            lower_bound=lower_bound,
        )
        if i == 0:
           ref_tri = tri.clone().detach()
           ref_tri_ht = tri_ht.clone().detach()
        err_ratio = get_err_ratio(tri, ref_tri)
        err_ratio_ht = get_err_ratio(tri_ht, ref_tri_ht)
        print(f"tri shape: {tri.shape}, has nan: {torch.isnan(tri).any()}")
        print(f"Iteration {i}: Relative error to first iter: {err_ratio:.6e}")
        print(f"Iteration {i}: Relative error (ht) to first iter: {err_ratio_ht:.6e}")
        err_ratio_list.append(err_ratio)
        err_ratio_ht_list.append(err_ratio_ht)

    # test if passed
    passed = True
    fail_diff = []
    fail_diff_ht = []
    for i in range(len(err_ratio_list)):
        if err_ratio_list[i] > 1e-8 or math.isnan(err_ratio_list[i]):
            passed = False
            fail_diff.append(err_ratio_list[i])
        if err_ratio_ht_list[i] > 1e-8 or math.isnan(err_ratio_ht_list[i]):
            passed = False
            fail_diff_ht.append(err_ratio_ht_list[i])
    if passed:
        print("PASSED")
    else:
        fail_diff.sort(reverse=True)
        fail_diff_ht.sort(reverse=True)
        print("failed diff", fail_diff)
        print("failed diff (ht)", fail_diff_ht)
        print(f"failed counts: {len(fail_diff)}")
        print(f"failed counts (ht): {len(fail_diff_ht)}")
        print("FAILED")

    from fla.ops.kda.gate import naive_kda_lowerbound_gate
    naive_kda_gate_fn = naive_kda_lowerbound_gate

    print("\n=== Running FLA chunk_kda ===")
    set_seed(SEED)
    ref, ref_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(), g=g.clone(), beta=beta.clone(),
        A_log=(A_log.clone() if A_log is not None else None),
        dt_bias=(dt_bias.clone() if dt_bias is not None else None),
        scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=use_gate_in_kernel, safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    print(f"ref shape: {ref.shape}, has nan: {torch.isnan(ref).any()}")

    print("\n=== Accuracy ===")
    abs_err = get_abs_err(ref_tri, ref)
    err_ratio = get_err_ratio(ref_tri, ref)
    abs_err_ht = get_abs_err(ref_tri_ht, ref_ht)
    err_ratio_ht = get_err_ratio(ref_tri_ht, ref_ht)
    print(f"Absolute error (o): {abs_err:.6e}, Relative error (o): {err_ratio:.6e}")
    print(f"Absolute error (ht): {abs_err_ht:.6e}, Relative error (ht): {err_ratio_ht:.6e}")

    assert_close("o", ref, ref_tri, 0.005)
    assert_close("ht", ref_ht, ref_tri_ht, 0.005)

if __name__ == "__main__":
    test_random_stress(T=4007)