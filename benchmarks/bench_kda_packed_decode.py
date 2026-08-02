#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""bench_kda_packed_decode.py — micro-benchmark: packed vs non-packed KDA decode.

Compares single-token (T=1) decode routes that share the same CuTe DSL kernel
body, so the delta isolates host-side orchestration + the q/k/v materialization
that a real caller pays.

Two routes are timed, BOTH seeded from the same packed ``mixed_qkv`` (the form
a conv layer actually produces, e.g. in SGLang's KDA decode):

  1. non-packed (realistic): the caller must turn ``mixed_qkv`` back into
     separate contiguous q/k/v before calling ``kda_decode`` — i.e. a per-call
     ``split + unsqueeze + contiguous``. The ``.contiguous()`` is a REAL copy
     when ``N>1`` because the split leaves row stride = qkv_dim. This is the
     cost the packed path is meant to remove.
  2. packed: ``cula.kda.kda_packed_decode`` feeds q/k/v as strided views
     directly — no materialization, no ``.contiguous()`` copy.

A third "non-packed (pre-split)" column is included as an oracle: it uses
q/k/v that were split once outside the timed loop (no per-call copy). This
shows the floor of the non-packed approach — i.e. how fast decode could be if
the caller already had separate contiguous q/k/v. packed matching or beating
the realistic column while staying near the oracle is the win.

All routes are timed with CUDA events bracketing the full callable (host-side
view construction + cache lookup + kernel launch). The ``mixed_qkv`` is reused
across iterations; only the per-call split+contiguous is inside the timed
region for the realistic non-packed route.

Usage:
    python benchmarks/bench_kda_packed_decode.py
    python benchmarks/bench_kda_packed_decode.py --batch-sizes 1 4 16 64 128
    python benchmarks/bench_kda_packed_decode.py --head-pairs 8:8 16:16 32:32 64:64
    python benchmarks/bench_kda_packed_decode.py --ncu
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmarks.utils import benchmark_cuda_fn
from cula.kda import fused_sigmoid_gating_delta_rule_update as cula_fused
from cula.kda import kda_packed_decode


def make_inputs(N, H, HV, K, V, device="cuda", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(N, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, HV, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = torch.randn(N, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


def run_config(N, H, HV, K, V, warmup, rep):
    device = "cuda"
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V, device)

    q_4d = q.unsqueeze(1).contiguous()
    k_4d = k.unsqueeze(1).contiguous()
    v_4d = v.unsqueeze(1).contiguous()
    a_flat = a.reshape(N, 1, -1).contiguous()
    b_3d = b.unsqueeze(1).contiguous()
    mixed_qkv = torch.cat([q.view(N, -1), k.view(N, -1), v.view(N, -1)], dim=-1).contiguous()
    qk_dim = H * K
    v_dim = HV * V
    indices = torch.arange(N, device=device, dtype=torch.int32)

    state_init = state.clone().contiguous()

    # non-packed (oracle): separate contiguous q/k/v split once outside the loop.
    def call_oracle(state_buf):
        return cula_fused(
            A_log=A_log,
            a=a_flat,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_4d,
            k=k_4d,
            v=v_4d,
            b=b_3d,
            initial_state_source=state_buf,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            state_layout="vk",
        )

    # non-packed (realistic): start from mixed_qkv, split+unsqueeze+contiguous
    # every call — the cost a real SGLang caller pays when q/k/v are not kept
    # pre-split.
    def call_nonpacked(state_buf):
        qq, kk, vv = mixed_qkv.split([qk_dim, qk_dim, v_dim], dim=-1)
        return cula_fused(
            A_log=A_log,
            a=a_flat,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=qq.view(N, 1, H, K).contiguous(),
            k=kk.view(N, 1, H, K).contiguous(),
            v=vv.view(N, 1, HV, V).contiguous(),
            b=b_3d,
            initial_state_source=state_buf,
            initial_state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            state_layout="vk",
        )

    # packed: kda_packed_decode route (mixed_qkv reused every iteration)
    def call_packed(state_buf):
        return kda_packed_decode(
            mixed_qkv,
            a_flat,
            b_3d,
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_buf,
            state_indices=indices,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            state_layout="vk",
        )

    # Correctness sanity: packed must match the oracle.
    state_ora = state_init.clone()
    state_pck = state_init.clone()
    with torch.no_grad():
        o_ora = call_oracle(state_ora)
        o_pck = call_packed(state_pck)
    out_diff = (o_ora.float() - o_pck.float()).abs().max().item()
    state_diff = (state_ora.float() - state_pck.float()).abs().max().item()

    state_bench_ora = state_init.clone()
    state_bench_non = state_init.clone()
    state_bench_pck = state_init.clone()

    def setup_ora():
        state_bench_ora.copy_(state_init)

    def setup_non():
        state_bench_non.copy_(state_init)

    def setup_pck():
        state_bench_pck.copy_(state_init)

    with torch.no_grad():
        t_ora = benchmark_cuda_fn(lambda: call_oracle(state_bench_ora), setup_fn=setup_ora, warmup=warmup, rep=rep)
        t_non = benchmark_cuda_fn(lambda: call_nonpacked(state_bench_non), setup_fn=setup_non, warmup=warmup, rep=rep)
        t_pck = benchmark_cuda_fn(lambda: call_packed(state_bench_pck), setup_fn=setup_pck, warmup=warmup, rep=rep)

    # q/k/v bytes the non-packed path copies per call (split+contiguous): bf16.
    qkv_dim = mixed_qkv.shape[1]
    copy_bytes = 2 * (2 * H * K + HV * V) * N

    return {
        "N": N,
        "H": H,
        "HV": HV,
        "K": K,
        "V": V,
        "qkv_dim": qkv_dim,
        "t_oracle_ms": t_ora,
        "t_non_ms": t_non,
        "t_packed_ms": t_pck,
        "saved_vs_non_us": (t_non - t_pck) * 1e3,
        "saved_vs_ora_us": (t_ora - t_pck) * 1e3,
        "speedup_vs_non": t_non / t_pck if t_pck > 0 else float("inf"),
        "out_diff": out_diff,
        "state_diff": state_diff,
        "copy_bytes": copy_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    parser.add_argument("--Hs", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--HV", type=int, default=16)
    parser.add_argument(
        "--head-pairs",
        type=str,
        nargs="+",
        default=None,
        help="Explicit H:HV pairs, e.g. --head-pairs 8:8 16:16 32:32 64:64",
    )
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--ncu", action="store_true")
    args = parser.parse_args()

    if args.ncu:
        args.warmup, args.rep = 1, 1

    gpu = torch.cuda.get_device_name(0)
    print(f"# cuLA Packed KDA Decode micro-bench  [{gpu}]")
    print(f"# K={args.K} V={args.V}  warmup={args.warmup} rep={args.rep}")
    print()

    if args.head_pairs is None:
        head_pairs = [(H, args.HV if args.HV >= 2 * H else 2 * H) for H in args.Hs]
    else:
        head_pairs = []
        for item in args.head_pairs:
            try:
                h_str, hv_str = item.split(":", 1)
                H, HV = int(h_str), int(hv_str)
            except ValueError as exc:
                raise ValueError(f"Invalid --head-pairs entry {item!r}; expected H:HV, e.g. 8:8") from exc
            if H <= 0 or HV <= 0 or HV % H != 0:
                raise ValueError(f"Invalid head pair H={H}, HV={HV}; expected positive values with HV % H == 0")
            head_pairs.append((H, HV))

    for H, HV in head_pairs:
        print(f"## H={H} HV={HV} K={args.K} V={args.V}")
        hdr = (
            f"{'N':>5} | {'qkv_dim':>7} | {'oracle (ms)':>12} | {'non-packed (ms)':>16} "
            f"| {'cula_packed (ms)':>17} | {'save vs non (us)':>16} | {'speedup vs non':>15} "
            f"| {'out_diff':>9} | {'state_diff':>10}"
        )
        print(hdr)
        print("-" * len(hdr))
        for N in args.batch_sizes:
            r = run_config(N, H, HV, args.K, args.V, args.warmup, args.rep)
            print(
                f"{r['N']:>5} | {r['qkv_dim']:>7} | {r['t_oracle_ms']:>12.4f} | {r['t_non_ms']:>16.4f} "
                f"| {r['t_packed_ms']:>17.4f} | {r['saved_vs_non_us']:>16.2f} | {r['speedup_vs_non']:>14.3f}x "
                f"| {r['out_diff']:>9.2e} | {r['state_diff']:>10.2e}"
            )
        print()


if __name__ == "__main__":
    main()
