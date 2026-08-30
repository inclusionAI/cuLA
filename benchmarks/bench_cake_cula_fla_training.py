#!/usr/bin/env python3
"""Compare FlashInfer CAKE, cuLA, and FLA KDA training on SM100.

The benchmark aligns all three public APIs to one mathematical contract:

* BF16 Q/K/V/G/beta logits and FP32 A_log/dt_bias/initial state.
* Q/K L2 normalization, safe gate with lower_bound=-5, and beta sigmoid.
* Both output and final-state gradients participate in backward.

CAKE stores state as [N, HV, V, K], while cuLA and FLA store it as
[N, HV, K, V].  State tensors are transposed outside timed regions.  The
unfused beta sigmoid in cuLA/FLA remains inside their timed paths.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch


D_HEAD = 128
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dA", "ddt", "dh0")


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    length: int
    qk_heads: int
    value_heads: int
    packed_lengths: tuple[int, ...] = ()

    @property
    def tokens(self) -> int:
        return sum(self.packed_lengths) if self.packed_lengths else self.batch * self.length

    @property
    def input_batch(self) -> int:
        return 1 if self.packed_lengths else self.batch

    @property
    def input_length(self) -> int:
        return self.tokens if self.packed_lengths else self.length

    @property
    def sequences(self) -> int:
        return len(self.packed_lengths) if self.packed_lengths else self.batch


DEFAULT_CASES = (
    Case("dense_b1_t1024_h8", 1, 1024, 8, 8),
    Case("dense_b1_t4096_h8", 1, 4096, 8, 8),
    Case("dense_b1_t16384_h8", 1, 16384, 8, 8),
    Case("dense_b4_t2048_h8", 4, 2048, 8, 8),
    Case("dense_b1_t4096_h32", 1, 4096, 32, 32),
    Case("dense_b1_t4096_h64", 1, 4096, 64, 64),
    Case("gva_b1_t4096_h8_hv32", 1, 4096, 8, 32),
    Case("packed_257_997_2048_789_h8", 1, 0, 8, 8, (257, 997, 2048, 789)),
)


def git_head(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flashinfer-root",
        default=os.environ.get("FLASHINFER_ROOT", "/ossfs/workspace/flashinfer-main-e425c7b0"),
    )
    parser.add_argument("--cula-root", default=os.environ.get("CULA_ROOT", "/ossfs/workspace/cuLA"))
    parser.add_argument("--cases", nargs="*", help="Case names; default runs the full matrix")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--disable-recompute",
        action="store_true",
        help="Save cuLA/FLA forward intermediates instead of recomputing them in backward",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print medians and compact error metrics instead of full samples and tensor statistics",
    )
    return parser.parse_args()


def select_cases(names: Sequence[str] | None) -> list[Case]:
    if not names:
        return list(DEFAULT_CASES)
    by_name = {case.name: case for case in DEFAULT_CASES}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"unknown cases: {missing}; choices={sorted(by_name)}")
    return [by_name[name] for name in names]


def make_cu_seqlens(lengths: Sequence[int], dtype: torch.dtype, device: str) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=dtype, device=device)


def tensor_diff(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | list[int] | str]:
    actual_f = actual.float()
    reference_f = reference.float()
    delta = actual_f - reference_f
    abs_delta = delta.abs()
    ref_rms = reference_f.square().mean().sqrt()
    return {
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype).removeprefix("torch."),
        "reference_dtype": str(reference.dtype).removeprefix("torch."),
        "max_abs": abs_delta.max().item(),
        "mean_abs": abs_delta.mean().item(),
        "rel_rms": (delta.square().mean().sqrt() / (ref_rms + 1e-12)).item(),
        "close_pct_at_1e-2": torch.isclose(actual_f, reference_f, atol=1e-2, rtol=1e-2)
        .float()
        .mean()
        .mul(100)
        .item(),
    }


def compact_result(result: dict[str, object]) -> dict[str, object]:
    compact_diffs = {
        pair: {
            tensor: {
                "max_abs": stats["max_abs"],
                "rel_rms": stats["rel_rms"],
                "close_pct_at_1e-2": stats["close_pct_at_1e-2"],
            }
            for tensor, stats in tensors.items()
        }
        for pair, tensors in result["diff"].items()
    }
    compact_timing = {
        phase: {
            "median_ms": stats["median_ms"],
            "speedup_vs_fla": stats["speedup_vs_fla"],
            "cake_speedup_vs_cula": stats["cake_speedup_vs_cula"],
        }
        for phase, stats in result["timing"].items()
    }
    return {
        "case": result["case"],
        "shape": result["shape"],
        "cake_route": result["cake_route"],
        "cula_bwd": result["cula_bwd"],
        "diff": compact_diffs,
        "timing": compact_timing,
    }


def timed_round(fn: Callable[[], object], iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def bench_many(
    functions: dict[str, Callable[[], object]],
    warmup: int,
    iters: int,
    rounds: int,
) -> dict[str, object]:
    for _ in range(warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in functions}
    names = list(functions)
    for round_idx in range(rounds):
        offset = round_idx % len(names)
        order = names[offset:] + names[:offset]
        if (round_idx // len(names)) % 2:
            order.reverse()
        for name in order:
            samples[name].append(timed_round(functions[name], iters))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    fla_ms = medians["fla"]
    return {
        "median_ms": medians,
        "speedup_vs_fla": {name: fla_ms / value for name, value in medians.items()},
        "cake_speedup_vs_cula": medians["cula"] / medians["cake"],
        "samples_ms": samples,
    }


def run_case(
    case: Case,
    flashinfer,
    cula_chunk_kda,
    fla_chunk_kda,
    cula_chunk_intra,
    cula_chunk_bwd,
    args: argparse.Namespace,
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = "cuda"
    B, T, H, HV = case.input_batch, case.input_length, case.qk_heads, case.value_heads

    def rand_bf16(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=torch.bfloat16)

    q = rand_bf16(B, T, H, D_HEAD)
    k = rand_bf16(B, T, H, D_HEAD)
    v = rand_bf16(B, T, HV, D_HEAD)
    g = rand_bf16(B, T, HV, D_HEAD)
    beta_logits = rand_bf16(B, T, HV)
    A_log = torch.randn(HV, device=device, dtype=torch.float32).mul_(0.5).sub_(1.0)
    dt_bias = torch.randn(HV, D_HEAD, device=device, dtype=torch.float32).mul_(0.5)
    h0_vk = torch.randn(case.sequences, HV, D_HEAD, D_HEAD, device=device, dtype=torch.float32).mul_(0.02)

    cake_cu = cake_cu_cpu = cula_cu = cula_cu_cpu = None
    if case.packed_lengths:
        cake_cu = make_cu_seqlens(case.packed_lengths, torch.int64, device)
        cake_cu_cpu = make_cu_seqlens(case.packed_lengths, torch.int64, "cpu")
        cula_cu = make_cu_seqlens(case.packed_lengths, torch.int32, device)
        cula_cu_cpu = make_cu_seqlens(case.packed_lengths, torch.int32, "cpu")

    cake_kwargs = {
        "cu_seqlens": cake_cu,
        "cu_seqlens_cpu": cake_cu_cpu,
        "scale": 1.0 / math.sqrt(D_HEAD),
        "lower_bound": -5.0,
    }
    cake_o, cake_ht, cake_ctx = flashinfer.recurrent_kda_training_forward(
        q, k, v, g, beta_logits, A_log, dt_bias, h0_vk, **cake_kwargs
    )

    def make_autograd_backend(name: str, fn, include_cu_seqlens_cpu: bool) -> dict[str, object]:
        inputs = tuple(x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta_logits))
        bq, bk, bv, bg, bbeta_logits = inputs
        bA = A_log.detach().clone().requires_grad_(True)
        bdt = dt_bias.detach().clone().requires_grad_(True)
        # Native cuLA/FLA state is K x V; CAKE state is V x K.
        bh0_kv = h0_vk.transpose(-1, -2).contiguous().detach().requires_grad_(True)
        leaves = (*inputs, bA, bdt, bh0_kv)

        def forward():
            kwargs = {}
            if include_cu_seqlens_cpu:
                kwargs["cu_seqlens_cpu"] = cula_cu_cpu
            return fn(
                bq,
                bk,
                bv,
                bg,
                # cuLA/FLA do not fuse beta sigmoid.  Keep it in their timed
                # contract and differentiate back to the shared beta logits.
                bbeta_logits.sigmoid(),
                scale=1.0 / math.sqrt(D_HEAD),
                initial_state=bh0_kv,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                cu_seqlens=cula_cu,
                safe_gate=True,
                lower_bound=-5.0,
                disable_recompute=args.disable_recompute,
                A_log=bA,
                # The chunk-64 APIs take the documented flat bias form;
                # view autograd maps the result back to [HV, D].
                dt_bias=bdt.flatten(),
                **kwargs,
            )

        output, final_state_kv = forward()
        return {
            "name": name,
            "leaves": leaves,
            "forward": forward,
            "output": output,
            "final_state_kv": final_state_kv,
        }

    cula_backend = make_autograd_backend("cula", cula_chunk_kda, True)
    fla_backend = make_autograd_backend("fla", fla_chunk_kda, False)
    do = torch.randn_like(cake_o)
    dht_vk = torch.randn_like(cake_ht).mul_(0.1)
    dht_kv = dht_vk.transpose(-1, -2).contiguous()
    cake_grads = flashinfer.recurrent_kda_training_backward(cake_ctx, do, dht_vk)

    def materialize_backend_grads(backend: dict[str, object]) -> list[torch.Tensor]:
        grads = list(
            torch.autograd.grad(
                (backend["output"], backend["final_state_kv"]),
                backend["leaves"],
                grad_outputs=(do, dht_kv),
                retain_graph=True,
            )
        )
        grads[-1] = grads[-1].transpose(-1, -2)
        return grads

    cula_grads = materialize_backend_grads(cula_backend)
    fla_grads = materialize_backend_grads(fla_backend)

    tensors = {
        "cake": {
            "output": cake_o,
            "final_state": cake_ht,
            **dict(zip(GRAD_NAMES, cake_grads)),
        },
        "cula": {
            "output": cula_backend["output"],
            "final_state": cula_backend["final_state_kv"].transpose(-1, -2),
            **dict(zip(GRAD_NAMES, cula_grads)),
        },
        "fla": {
            "output": fla_backend["output"],
            "final_state": fla_backend["final_state_kv"].transpose(-1, -2),
            **dict(zip(GRAD_NAMES, fla_grads)),
        },
    }

    def pairwise_diff(actual: str, reference: str) -> dict[str, object]:
        return {
            name: tensor_diff(tensors[actual][name], tensors[reference][name])
            for name in ("output", "final_state", *GRAD_NAMES)
        }

    diffs = {
        "cake_vs_fla": pairwise_diff("cake", "fla"),
        "cula_vs_fla": pairwise_diff("cula", "fla"),
        "cake_vs_cula": pairwise_diff("cake", "cula"),
    }
    # In safe-gate mode y = lower_bound * sigmoid(exp(A_log) * x), where
    # x = g + dt_bias.  Thus ddt is the B/T reduction of raw dg and dA is the
    # reduction of dg*x.  These checks distinguish a pointwise-gradient
    # mismatch from cancellation/accumulation differences in FP32 parameters.
    gate_x = g.float() + dt_bias[None, None, :, :]

    def reduced_parameter_grads(raw_dg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_dg_f = raw_dg.float()
        return (
            (raw_dg_f * gate_x).sum(dim=(0, 1, 3)),
            raw_dg_f.sum(dim=(0, 1)),
        )

    cake_dA_from_dg, cake_ddt_from_dg = reduced_parameter_grads(cake_grads[3])
    cula_dA_from_dg, cula_ddt_from_dg = reduced_parameter_grads(cula_grads[3])
    fla_dA_from_dg, fla_ddt_from_dg = reduced_parameter_grads(fla_grads[3])
    reduction_diagnostics = {
        "cake_dA_vs_reduced_returned_dg": tensor_diff(cake_grads[5], cake_dA_from_dg),
        "cula_dA_vs_reduced_returned_dg": tensor_diff(cula_grads[5], cula_dA_from_dg),
        "fla_dA_vs_reduced_returned_dg": tensor_diff(fla_grads[5], fla_dA_from_dg),
        "cake_ddt_vs_reduced_returned_dg": tensor_diff(cake_grads[6], cake_ddt_from_dg),
        "cula_ddt_vs_reduced_returned_dg": tensor_diff(cula_grads[6], cula_ddt_from_dg),
        "fla_ddt_vs_reduced_returned_dg": tensor_diff(fla_grads[6], fla_ddt_from_dg),
    }

    # CAKE exposes caller-owned output/context buffers.  Prime them once, then
    # use the steady-state mode in all timed regions.
    cake_grad_out = tuple(torch.empty_like(grad) for grad in cake_grads)

    def cake_forward():
        return flashinfer.recurrent_kda_training_forward(
            q,
            k,
            v,
            g,
            beta_logits,
            A_log,
            dt_bias,
            h0_vk,
            out=cake_o,
            final_state_out=cake_ht,
            context_out=cake_ctx,
            **cake_kwargs,
        )

    def cake_backward():
        return flashinfer.recurrent_kda_training_backward(cake_ctx, do, dht_vk, out=cake_grad_out)

    def make_backward(backend: dict[str, object]):
        def backward():
            return torch.autograd.grad(
                (backend["output"], backend["final_state_kv"]),
                backend["leaves"],
                grad_outputs=(do, dht_kv),
                retain_graph=True,
            )

        return backward

    cula_backward = make_backward(cula_backend)
    fla_backward = make_backward(fla_backend)

    def cake_pair():
        cake_forward()
        return cake_backward()

    def make_pair(backend: dict[str, object]):
        def pair():
            out, ht = backend["forward"]()
            return torch.autograd.grad(
                (out, ht),
                backend["leaves"],
                grad_outputs=(do, dht_kv),
                retain_graph=False,
            )

        return pair

    cula_pair = make_pair(cula_backend)
    fla_pair = make_pair(fla_backend)

    timing = {
        "forward": bench_many(
            {"cake": cake_forward, "cula": cula_backend["forward"], "fla": fla_backend["forward"]},
            args.warmup,
            args.iters,
            args.rounds,
        ),
        "backward": bench_many(
            {"cake": cake_backward, "cula": cula_backward, "fla": fla_backward},
            args.warmup,
            args.iters,
            args.rounds,
        ),
        "fwd_bwd": bench_many(
            {"cake": cake_pair, "cula": cula_pair, "fla": fla_pair},
            args.warmup,
            args.iters,
            args.rounds,
        ),
    }
    route = getattr(cake_ctx, "_route", None)
    route_fields = vars(route) if route is not None and hasattr(route, "__dict__") else {"repr": repr(route)}
    return {
        "case": case.name,
        "shape": {
            "B": B,
            "T": T,
            "H": H,
            "HV": HV,
            "D": D_HEAD,
            "tokens": case.tokens,
            "sequences": case.sequences,
            "packed_lengths": list(case.packed_lengths),
        },
        "cake_route": route_fields,
        "cula_bwd": {
            "intra_requested": cula_chunk_intra._normalize_bwd_intra_backend(),
            "intra_cutedsl_supported": cula_chunk_intra._is_bwd_intra_sm100_supported(
                cula_backend["leaves"][0], cula_backend["leaves"][3], 64, True
            ),
            "wy_dqkg_impl": (
                cula_chunk_bwd._select_chunk_kda_bwd_wy_dqkg_fused(
                    cula_backend["leaves"][0], cula_backend["leaves"][2]
                ).__module__
                if hasattr(cula_chunk_bwd, "_select_chunk_kda_bwd_wy_dqkg_fused")
                else cula_chunk_bwd.chunk_kda_bwd_wy_dqkg_fused_cutedsl.__module__
            ),
            "recompute_impl": (
                cula_chunk_bwd._select_recompute_w_u_backend(cula_backend["leaves"][0].device)
                if hasattr(cula_chunk_bwd, "_select_recompute_w_u_backend")
                else "cula.cudac.recompute_w_u_cuda"
            ),
            "disable_recompute": args.disable_recompute,
        },
        "diff": diffs,
        "reduction_diagnostics": reduction_diagnostics,
        "timing": timing,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.cula_root)))
    sys.path.insert(0, str(Path(args.flashinfer_root)))
    import flashinfer
    import fla
    import cula.kda.chunk_bwd as cula_chunk_bwd
    import cula.kda.chunk_intra as cula_chunk_intra
    from cula.kda import chunk_kda as cula_chunk_kda
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    fla_root = str(Path(fla.__file__).resolve().parent.parent)

    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "flashinfer_root": args.flashinfer_root,
        "cake_commit": git_head(args.flashinfer_root),
        "cula_root": args.cula_root,
        "cula_commit": git_head(args.cula_root),
        "fla_root": fla_root,
        "fla_commit": git_head(fla_root),
        "cula_bwd_intra_backend": cula_chunk_intra._normalize_bwd_intra_backend(),
        "disable_recompute": args.disable_recompute,
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
        "timing_note": "CUDA-event steady-state; state transposes excluded; cuLA/FLA beta sigmoid included",
    }
    print("METADATA " + json.dumps(metadata, sort_keys=True), flush=True)
    for case in select_cases(args.cases):
        print(f"RUN {case.name}", flush=True)
        try:
            result = run_case(
                case,
                flashinfer,
                cula_chunk_kda,
                fla_chunk_kda,
                cula_chunk_intra,
                cula_chunk_bwd,
                args,
            )
            emitted_result = compact_result(result) if args.compact else result
            print("RESULT " + json.dumps(emitted_result, sort_keys=True), flush=True)
        except Exception as exc:
            print(
                "RESULT "
                + json.dumps({"case": case.name, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True),
                flush=True,
            )
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
