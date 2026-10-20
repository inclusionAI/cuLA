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

"""Small Nsight Compute target for Qwen3.5 decode kernels.

Use with `ncu --profile-from-start off` so only the decode loop bracketed by
cudaProfilerStart/Stop is collected.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cula.cudac as cula_cuda
from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG as GLOBAL_CONFIG
from cula.qwen35.common import Qwen35LinearAttentionConfig


def local_config_from_tp_size(tp_size: int) -> Qwen35LinearAttentionConfig:
    if tp_size not in (1, 2, 4, 8):
        raise ValueError(f"tp_size must be one of 1, 2, 4, 8, got {tp_size}")
    return Qwen35LinearAttentionConfig(
        hidden_size=GLOBAL_CONFIG.hidden_size // tp_size,
        conv_kernel_size=GLOBAL_CONFIG.conv_kernel_size,
        num_k_heads=GLOBAL_CONFIG.num_k_heads // tp_size,
        num_v_heads=GLOBAL_CONFIG.num_v_heads // tp_size,
        head_k_dim=GLOBAL_CONFIG.head_k_dim,
        head_v_dim=GLOBAL_CONFIG.head_v_dim,
        qkv_dtype=GLOBAL_CONFIG.qkv_dtype,
        state_dtype=GLOBAL_CONFIG.state_dtype,
    )


def make_fused_inputs(tokens: int, device: torch.device, seed: int, config: Qwen35LinearAttentionConfig):
    torch.manual_seed(seed)
    mixed_qkv_conv = torch.randn(tokens, config.conv_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    A_log = -torch.rand(config.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(
        tokens,
        config.num_v_heads,
        config.head_k_dim,
        config.head_v_dim,
        device=device,
        dtype=config.state_dtype,
    ) * 0.01
    state_work = torch.empty_like(state)
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32)
    out = torch.empty(tokens, config.num_v_heads, config.head_v_dim, device=device, dtype=config.qkv_dtype)
    return mixed_qkv_conv, a, b, A_log, dt_bias, state, state_work, state_indices, out


def make_native_inputs(tokens: int, device: torch.device, seed: int, config: Qwen35LinearAttentionConfig):
    torch.manual_seed(seed)
    q = torch.randn(tokens, config.num_v_heads, config.head_k_dim, device=device, dtype=config.qkv_dtype)
    k = torch.randn(tokens, config.num_v_heads, config.head_k_dim, device=device, dtype=config.qkv_dtype)
    v = torch.randn(tokens, config.num_v_heads, config.head_v_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    A_log = -torch.rand(config.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(
        tokens,
        config.num_v_heads,
        config.head_k_dim,
        config.head_v_dim,
        device=device,
        dtype=config.state_dtype,
    ) * 0.01
    state_work = torch.empty_like(state)
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32)
    out = torch.empty_like(v)
    return q, k, v, a, b, A_log, dt_bias, state, state_work, state_indices, out


def profiler_start() -> None:
    torch.cuda.profiler.start()


def profiler_stop() -> None:
    torch.cuda.profiler.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=("fused", "native"), default="fused")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is available")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", torch.cuda.current_device())
    config = local_config_from_tp_size(args.tp_size)

    if args.op == "fused":
        mixed_qkv_conv, a, b, A_log, dt_bias, state, state_work, state_indices, out = make_fused_inputs(
            args.tokens, device, args.seed, config
        )

        def run() -> None:
            cula_cuda.qwen35_layout_scalar_kda_decode(
                mixed_qkv_conv,
                a,
                b,
                A_log,
                dt_bias,
                state_work,
                state_indices,
                out,
            )

    else:
        q, k, v, a, b, A_log, dt_bias, state, state_work, state_indices, out = make_native_inputs(
            args.tokens, device, args.seed, config
        )

        def run() -> None:
            cula_cuda.qwen35_scalar_kda_decode(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state_work,
                state_indices,
                out,
            )

    for _ in range(args.warmup):
        state_work.copy_(state)
        run()
    torch.cuda.synchronize()

    state_work.copy_(state)
    torch.cuda.synchronize()

    print(
        f"profile op={args.op} tokens={args.tokens} tp={args.tp_size} "
        f"warmup={args.warmup} rep={args.rep} device={torch.cuda.get_device_name(device)}"
    )
    profiler_start()
    for _ in range(args.rep):
        run()
    torch.cuda.synchronize()
    profiler_stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
