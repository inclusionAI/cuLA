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

"""Nsight Compute target for the legacy Qwen scalar CUDA prefill path.

This calls only ``qwen35_scalar_kda_prefill_core``.  It never resolves or
imports the experimental CuTe prefill backend.  Use ``ncu
--profile-from-start off`` so only the launches bracketed by
``cudaProfilerStart/Stop`` are collected.

Examples::

    ncu --profile-from-start off --kernel-name-base demangled \
      --kernel-name 'regex:.*qwen35_chunk_state_output_sm100_ts_kernel.*' \
      -o /tmp/qwen35_scalar_t256_state_output \
      python benchmarks/profile_qwen35_scalar_prefill.py \
        --config-json /data/xinhaowei/qwen_configs/Qwen3.5-27B/config.json \
        --seq-len 256 --rep 1

    ncu --profile-from-start off --kernel-name-base demangled \
      --kernel-name 'regex:.*qwen35_chunk_(preprocess|state_output).*' \
      -o /tmp/qwen35_scalar_t512_stages \
      python benchmarks/profile_qwen35_scalar_prefill.py \
        --config-json /data/xinhaowei/qwen_configs/Qwen3.5-27B/config.json \
        --seq-len 512 --rep 1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_shape(config_path: pathlib.Path, tp_size: int) -> tuple[int, int]:
    root = json.loads(config_path.read_text(encoding="utf-8"))
    config = root.get("text_config", root)
    global_h = int(config["linear_num_key_heads"])
    global_hv = int(config["linear_num_value_heads"])
    if global_h % tp_size or global_hv % tp_size:
        raise ValueError(f"TP={tp_size} must divide global H/HV={global_h}/{global_hv}")
    h, hv = global_h // tp_size, global_hv // tp_size
    if hv % h:
        raise ValueError(f"local HV={hv} must be divisible by local H={h}")
    if int(config["linear_key_head_dim"]) != 128 or int(config["linear_value_head_dim"]) != 128:
        raise ValueError("the scalar CUDA prefill path requires K=V=128")
    return h, hv


def extension_path(cula_cuda) -> str:
    if not hasattr(cula_cuda, "qwen35_scalar_kda_prefill_core"):
        raise RuntimeError("the loaded cuLA extension does not expose qwen35_scalar_kda_prefill_core")
    op = cula_cuda.qwen35_scalar_kda_prefill_core
    module = sys.modules.get(getattr(op, "__module__", ""))
    return str(getattr(module, "__file__", "unavailable"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-json", type=pathlib.Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is available")
    if args.rep < 1 or args.warmup < 0:
        parser.error("--rep must be positive and --warmup must be non-negative")
    if args.seq_len < 32:
        parser.error("the chunk scalar CUDA path requires --seq-len >= 32")

    import cula.cudac as cula_cuda

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", torch.cuda.current_device())
    h, hv = load_shape(args.config_json, args.tp_size)
    torch.manual_seed(args.seed)

    q = torch.randn(1, args.seq_len, h, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, args.seq_len, hv, 128, device=device, dtype=torch.bfloat16)
    a = torch.randn(1, args.seq_len, hv, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = -torch.rand(hv, device=device, dtype=torch.float32)
    dt_bias = torch.randn(hv, device=device, dtype=torch.float32) * 0.1
    g = -torch.exp(A_log).view(1, 1, hv) * F.softplus(a.float() + dt_bias.view(1, 1, hv))
    beta = torch.sigmoid(b.float())
    initial_state = torch.randn(1, hv, 128, 128, device=device, dtype=torch.float32) * 0.01
    cu_seqlens = torch.tensor([0, args.seq_len], device=device, dtype=torch.int32)
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)

    def run() -> None:
        cula_cuda.qwen35_scalar_kda_prefill_core(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            out,
            final_state,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    print(
        f"profile scalar_cuda_core T={args.seq_len} TP={args.tp_size} H/HV={h}/{hv} "
        f"warmup={args.warmup} rep={args.rep} device={torch.cuda.get_device_name(device)}"
    )
    print(f"extension={extension_path(cula_cuda)}")
    torch.cuda.profiler.start()
    for _ in range(args.rep):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
