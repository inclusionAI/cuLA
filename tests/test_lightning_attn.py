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

"""
Test suite for LinearAttentionChunkwiseDecay (CuTeDSL Blackwell kernel).

Compares against:
  1. PyTorch reference implementation (exact numerical match)
  2. FLA chunk_simple_gla (exact numerical match via g_gamma = -s)
"""

import argparse
import pathlib
import sys
import warnings

import pytest
import torch

pytestmark = pytest.mark.sm100_only

# Suppress third-party deprecation warnings (e.g. torch.jit)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from cula.ops.lightning_attn import lightning_attn_fwd, lightning_attn_fwd_varlen  # noqa: E402

try:
    from fla.ops.simple_gla import chunk_simple_gla

    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    print("Warning: fla library not available, skipping FLA comparison tests")


# ---------------------------------------------------------------------------
# Helper: run CuTeDSL kernel
# ---------------------------------------------------------------------------


def run_cute_kernel(
    Q,
    K,
    V,
    decay,
    scale=1.0,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
):
    """Run the CuTeDSL LinearAttentionChunkwiseDecay kernel.

    Uses TVM-FFI compile cache: first call per config compiles, subsequent reuse.

    Args:
        Q, K, V: (B, S, H, D) bfloat16 tensors on CUDA
        decay: (H,) float32 per-head decay parameter s (s > 0)
        scale: attention scale factor
        chunk_size: chunk size C
        initial_state: (B, H, D, D) float32 or None
        output_final_state: whether to allocate and return final state

    Returns:
        O: (B, S, H, D) bfloat16 output
        ht: (B, H, D, D) float32 final state (or None)
    """
    O, ht = lightning_attn_fwd(
        Q,
        K,
        V,
        decay,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    torch.cuda.synchronize()
    return O, ht


def run_cute_kernel_varlen(
    Q,
    K,
    V,
    decay,
    cu_seqlens,
    scale=1.0,
    chunk_size=64,
    state_pool=None,
    initial_state_indices=None,
):
    """Run the CuTeDSL varlen kernel.

    Args:
        Q, K, V: (1, T, H, D) bfloat16 — packed sequences
        decay: (H,) float32
        cu_seqlens: (N+1,) int32
        state_pool: (pool_size, H, D, D) float32 or None
        initial_state_indices: (N,) int32 or None

    Returns:
        O: (1, T, H, D) bfloat16
        state_pool: (pool_size, H, D, D) float32
    """
    O, sp = lightning_attn_fwd_varlen(
        Q,
        K,
        V,
        decay,
        cu_seqlens,
        scale=scale,
        state_pool=state_pool,
        initial_state_indices=initial_state_indices,
        chunk_size=chunk_size,
    )
    torch.cuda.synchronize()
    return O, sp


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def pytorch_reference(Q, K, V, decay, chunk_size=64, scale=1.0, initial_state=None, output_final_state=False):
    """PyTorch reference for chunkwise linear attention with exponential decay.

    Args:
        Q, K, V: (B, T, H, D) — any dtype, computed in float32
        decay: (H,) float32 per-head s (s >= 0)
        chunk_size: C
        scale: scalar multiplier applied to final output
        initial_state: (B, H, D, D) float32 or None
        output_final_state: bool

    Returns:
        O: (B, T, H, D) float32
        final_state: (B, H, D, D) float32 or None
    """
    B, T, H, D = Q.shape
    C = chunk_size
    Q, K, V = Q.float(), K.float(), V.float()
    O = torch.zeros(B, T, H, D, device=Q.device, dtype=torch.float32)

    state = (
        initial_state.clone().float()
        if initial_state is not None
        else torch.zeros(B, H, D, D, device=Q.device, dtype=torch.float32)
    )

    num_chunks = (T + C - 1) // C
    for ci in range(num_chunks):
        cs, ce = ci * C, min((ci + 1) * C, T)
        cl = ce - cs

        Qc = Q[:, cs:ce]  # (B, cl, H, D)
        Kc = K[:, cs:ce]
        Vc = V[:, cs:ce]

        # --- intra-chunk: QK with causal decay mask ---
        QK = torch.einsum("bthd,bshd->bhts", Qc, Kc)
        pos_q = torch.arange(cl, device=Q.device).view(cl, 1)
        pos_k = torch.arange(cl, device=Q.device).view(1, cl)
        dist = pos_q - pos_k  # (cl, cl)

        s = decay.view(1, H, 1, 1)
        mask = torch.exp(-s * dist.unsqueeze(0).unsqueeze(0).float())
        mask = mask * (pos_q >= pos_k).unsqueeze(0).unsqueeze(0).float()
        O_intra = torch.einsum("bhts,bshd->bthd", QK * mask, Vc)

        # --- inter-chunk: Q @ state with per-position decay ---
        pos_in = torch.arange(cl, device=Q.device).float()
        per_pos = torch.exp(-decay.view(1, 1, H, 1) * (pos_in.view(1, -1, 1, 1) + 1.0))
        O_inter = torch.einsum("bthd,bhde->bthe", Qc, state) * per_pos

        O[:, cs:ce] = (O_intra + O_inter) * scale

        # --- state update ---
        block_decay = torch.exp(-decay.view(1, H, 1, 1) * C)
        pos_w = torch.exp(-decay.view(1, 1, H, 1) * (C - 1 - pos_in.view(1, -1, 1, 1)))
        state = state * block_decay + torch.einsum("bthd,bthe->bhde", Kc * pos_w, Vc)

    return O, (state if output_final_state else None)


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def _compare(name, actual, expected, atol=5e-3, rtol=5e-2, verbose=True):
    """Compare two tensors, return True if within tolerance."""
    diff = (actual.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mag = expected.float().abs().max().item()
    rel = max_diff / (mag + 1e-8)
    if verbose:
        print(f"  {name:8s}  max_diff={max_diff:.6f}  rel={rel:.6f}")
    return max_diff < atol or rel < rtol


# ===========================================================================
# Test functions
# ===========================================================================


def test_basic_execution():
    """Kernel compiles and produces non-NaN, non-Inf output."""
    print("\nTesting basic execution...")
    B, S, H, D = 1, 64, 2, 128
    torch.manual_seed(42)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), 0.1, device="cuda", dtype=torch.float32)

    try:
        O, _ = run_cute_kernel(Q, K, V, decay)
        assert not torch.isnan(O).any(), "NaN in output"
        assert not torch.isinf(O).any(), "Inf in output"
        assert O.abs().max() < 100, "Output values too large"
        print("  ✓ PASSED")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_decay_values():
    """Different decay values produce distinct outputs."""
    print("\nTesting different decay values...")
    B, S, H, D = 1, 128, 4, 128
    torch.manual_seed(42)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1

    try:
        outputs = {}
        for dv in [0.05, 0.1, 0.5]:
            decay = torch.full((H,), dv, device="cuda", dtype=torch.float32)
            O, _ = run_cute_kernel(Q, K, V, decay)
            outputs[dv] = O.clone()

        diff = (outputs[0.05] - outputs[0.5]).abs().mean().item()
        assert diff > 1e-4, f"Decay has no effect (diff={diff:.2e})"
        print("  ✓ PASSED")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_against_reference(B=1, S=128, H=4, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True):
    """Compare against PyTorch reference (exact match)."""
    if verbose:
        print(f"\nRef: B={B}, S={S}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)

    O_ref, _ = pytorch_reference(Q, K, V, decay, chunk_size=C)
    O_ref_bf16 = O_ref.to(torch.bfloat16)

    O_cute, _ = run_cute_kernel(Q, K, V, decay, chunk_size=C)

    passed = _compare("output", O_cute, O_ref_bf16, atol=atol, rtol=rtol, verbose=verbose)
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_initial_and_final_state(B=1, S=128, H=4, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True):
    """Test h0/ht against PyTorch reference.

    NOTE: This test is placed BEFORE FLA tests so that the (has_initial_state=True,
    output_final_state=True) kernel variant is compiled before any Triton/FLA code
    runs.  Running Triton corrupts state needed by cute.compile.
    """
    if verbose:
        print(f"\nh0/ht: B={B}, S={S}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)
    h0 = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32).transpose(-1, -2).contiguous().transpose(-1, -2) * 0.01

    O_ref, ht_ref = pytorch_reference(
        Q,
        K,
        V,
        decay,
        chunk_size=C,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    O_ref_bf16 = O_ref.to(torch.bfloat16)

    O_cute, ht_cute = run_cute_kernel(
        Q,
        K,
        V,
        decay,
        chunk_size=C,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    p1 = _compare("output", O_cute, O_ref_bf16, atol=atol, rtol=rtol, verbose=verbose)
    p2 = True
    if ht_ref is not None and ht_cute is not None:
        p2 = _compare("state", ht_cute, ht_ref, atol=atol, rtol=rtol, verbose=verbose)

    passed = p1 and p2
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_against_fla(B=1, S=128, H=4, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True):
    """Compare against FLA chunk_simple_gla using g_gamma = -s.

    FLA's g_gamma is the per-head log-decay (negative). Our decay parameter s
    maps as g_gamma = -s, giving identical per-timestep decay exp(-s).
    """
    if not HAS_FLA:
        print("\n  ⊘ SKIPPED: fla library not available")
        return True

    if verbose:
        print(f"\nFLA: B={B}, S={S}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1

    # Our decay s -> FLA g_gamma = -s
    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)
    g_gamma = -decay

    # FLA reference (scale=1.0 to match our kernel)
    O_fla, _ = chunk_simple_gla(Q, K, V, g_gamma=g_gamma, scale=1.0, head_first=False)

    # Our kernel
    O_cute, _ = run_cute_kernel(Q, K, V, decay, scale=1.0, chunk_size=C)

    if verbose:
        print(
            f"  O_fla:  mean={O_fla.float().mean():.6f}, std={O_fla.float().std():.6f}, "
            f"range=[{O_fla.min():.4f}, {O_fla.max():.4f}]"
        )
        print(
            f"  O_cute: mean={O_cute.float().mean():.6f}, std={O_cute.float().std():.6f}, "
            f"range=[{O_cute.min():.4f}, {O_cute.max():.4f}]"
        )

    passed = _compare("output", O_cute, O_fla, atol=atol, rtol=rtol, verbose=verbose)
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_against_fla_with_state(B=1, S=128, H=4, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True):
    """Compare h0/ht against FLA chunk_simple_gla."""
    if not HAS_FLA:
        print("\n  ⊘ SKIPPED: fla library not available")
        return True

    if verbose:
        print(f"\nFLA h0/ht: B={B}, S={S}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    h0 = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32).transpose(-1, -2).contiguous().transpose(-1, -2) * 0.01

    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)
    g_gamma = -decay

    # FLA
    O_fla, ht_fla = chunk_simple_gla(
        Q,
        K,
        V,
        g_gamma=g_gamma,
        scale=1.0,
        initial_state=h0.clone(),
        output_final_state=True,
        head_first=False,
    )

    # Ours
    O_cute, ht_cute = run_cute_kernel(
        Q,
        K,
        V,
        decay,
        scale=1.0,
        chunk_size=C,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    p1 = _compare("output", O_cute, O_fla, atol=atol, rtol=rtol, verbose=verbose)
    p2 = True
    if ht_fla is not None and ht_cute is not None:
        p2 = _compare("state", ht_cute, ht_fla, atol=atol, rtol=rtol, verbose=verbose)

    passed = p1 and p2
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


# ===========================================================================
# Varlen tests
# ===========================================================================


def test_varlen_single_seq(H=4, S=128, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True) -> bool:
    """Varlen with a single sequence vs non-varlen reference."""
    if verbose:
        print(f"\nVarlen single: S={S}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    Q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)

    # Non-varlen reference
    O_ref, ht_ref = run_cute_kernel(Q, K, V, decay, chunk_size=C, output_final_state=True)

    # Varlen
    cu_seqlens = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    O_var, sp = run_cute_kernel_varlen(Q, K, V, decay, cu_seqlens, chunk_size=C)

    p1 = _compare("output", O_var, O_ref, atol=atol, rtol=rtol, verbose=verbose)
    p2 = _compare("state", sp[0], ht_ref, atol=atol, rtol=rtol, verbose=verbose)
    passed = p1 and p2
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_varlen_multi_seq(seq_lens=None, H=4, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True) -> bool:
    """Varlen with multiple packed sequences vs per-sequence non-varlen reference."""
    if seq_lens is None:
        seq_lens = [128, 64, 192]  # all multiples of C
    if verbose:
        print(f"\nVarlen multi: seqs={seq_lens}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    T = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    Q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)

    O_var, sp = run_cute_kernel_varlen(Q, K, V, decay, cu_seqlens, chunk_size=C)

    all_pass = True
    for i, slen in enumerate(seq_lens):
        bos = cu_seqlens[i].item()
        eos = cu_seqlens[i + 1].item()
        Qi = Q[:, bos:eos].contiguous()
        Ki = K[:, bos:eos].contiguous()
        Vi = V[:, bos:eos].contiguous()
        O_ref_i, ht_ref_i = run_cute_kernel(Qi, Ki, Vi, decay, chunk_size=C, output_final_state=True)

        po = _compare(f"O[{i}]", O_var[:, bos:eos], O_ref_i, atol=atol, rtol=rtol, verbose=verbose)
        ps = _compare(f"ht[{i}]", sp[i], ht_ref_i, atol=atol, rtol=rtol, verbose=verbose)
        all_pass = all_pass and po and ps

    print(f"  {'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_varlen_with_initial_state(seq_lens=None, H=4, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True) -> bool:
    """Varlen with initial state from state pool (non-contiguous indices)."""
    if seq_lens is None:
        seq_lens = [128, 64]
    if verbose:
        print(f"\nVarlen h0: seqs={seq_lens}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    T = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    Q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)

    # State pool with 3 slots, use indices [2, 0]
    pool_size = 3
    state_pool = torch.randn(pool_size, H, D, D, dtype=torch.float32, device="cuda").transpose(-1, -2).contiguous().transpose(-1, -2) * 0.01
    indices = torch.tensor([2, 0], dtype=torch.int32, device="cuda")

    O_var, sp = run_cute_kernel_varlen(
        Q,
        K,
        V,
        decay,
        cu_seqlens,
        chunk_size=C,
        state_pool=state_pool.clone(),
        initial_state_indices=indices,
    )

    all_pass = True
    for i, slen in enumerate(seq_lens):
        bos = cu_seqlens[i].item()
        eos = cu_seqlens[i + 1].item()
        idx = indices[i].item()
        Qi = Q[:, bos:eos].contiguous()
        Ki = K[:, bos:eos].contiguous()
        Vi = V[:, bos:eos].contiguous()
        h0_i = state_pool[idx : idx + 1].clone()

        O_ref_i, ht_ref_i = run_cute_kernel(
            Qi,
            Ki,
            Vi,
            decay,
            chunk_size=C,
            initial_state=h0_i,
            output_final_state=True,
        )

        po = _compare(f"O[{i}]", O_var[:, bos:eos], O_ref_i, atol=atol, rtol=rtol, verbose=verbose)
        ps = _compare(f"ht[{i}]", sp[idx], ht_ref_i, atol=atol, rtol=rtol, verbose=verbose)
        all_pass = all_pass and po and ps

    print(f"  {'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


def test_varlen_against_pytorch_ref(
    seq_lens=None, H=4, D=128, C=64, decay_val=0.1, atol=5e-3, rtol=5e-2, verbose=True
) -> bool:
    """Varlen against the PyTorch reference with initial state."""
    if seq_lens is None:
        seq_lens = [128, 192]
    if verbose:
        print(f"\nVarlen vs ref: seqs={seq_lens}, H={H}, D={D}, C={C}, decay={decay_val}")

    torch.manual_seed(42)
    T = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    Q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), decay_val, device="cuda", dtype=torch.float32)
    state_pool = torch.randn(N, H, D, D, dtype=torch.float32, device="cuda").transpose(-1, -2).contiguous().transpose(-1, -2) * 0.01

    O_var, sp = run_cute_kernel_varlen(
        Q,
        K,
        V,
        decay,
        cu_seqlens,
        chunk_size=C,
        state_pool=state_pool.clone(),
    )

    all_pass = True
    for i, slen in enumerate(seq_lens):
        bos = cu_seqlens[i].item()
        eos = cu_seqlens[i + 1].item()
        Qi = Q[:, bos:eos].contiguous()
        Ki = K[:, bos:eos].contiguous()
        Vi = V[:, bos:eos].contiguous()
        h0_i = state_pool[i : i + 1].clone()

        O_ref_i, ht_ref_i = pytorch_reference(
            Qi,
            Ki,
            Vi,
            decay,
            chunk_size=C,
            scale=1.0,
            initial_state=h0_i,
            output_final_state=True,
        )
        O_ref_bf16 = O_ref_i.to(torch.bfloat16)

        po = _compare(f"O[{i}]", O_var[:, bos:eos], O_ref_bf16, atol=atol, rtol=rtol, verbose=verbose)
        ps = _compare(f"ht[{i}]", sp[i], ht_ref_i, atol=atol, rtol=rtol, verbose=verbose)
        all_pass = all_pass and po and ps

    print(f"  {'✓ PASSED' if all_pass else '✗ FAILED'}")
    return all_pass


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Lightning Attention test suite")
    parser.add_argument("--test", choices=["basic", "ref", "fla", "h0ht", "varlen", "all"], default="all")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return False

    print("=" * 60)
    print("LIGHTNING ATTENTION — TEST SUITE")
    print("=" * 60)

    results = []

    # ---- Basic sanity ----
    if args.test in ("basic", "all"):
        print("\n" + "=" * 60)
        print("BASIC TESTS")
        print("=" * 60)
        results.append(("Basic execution", test_basic_execution()))
        results.append(("Different decay values", test_different_decay_values()))

    # ---- PyTorch reference ----
    if args.test in ("ref", "all"):
        print("\n" + "=" * 60)
        print("PYTORCH REFERENCE TESTS")
        print("=" * 60)
        for tag, kw in [
            ("Small (64x64)", dict(B=1, S=64, H=2, D=128, C=64, decay_val=0.1)),
            ("Zero decay", dict(B=1, S=64, H=2, D=128, C=64, decay_val=0.0)),
            ("Multi-chunk (256)", dict(B=1, S=256, H=4, D=128, C=64, decay_val=0.1)),
            ("Decay 0.05", dict(B=1, S=128, H=4, D=128, C=64, decay_val=0.05)),
            ("Decay 0.2", dict(B=1, S=128, H=4, D=128, C=64, decay_val=0.2)),
            ("Decay 0.5", dict(B=1, S=128, H=4, D=128, C=64, decay_val=0.5)),
            ("Batch", dict(B=2, S=128, H=4, D=128, C=64, decay_val=0.1)),
        ]:
            results.append((f"Ref {tag}", test_against_reference(**kw, verbose=args.verbose)))

    # ---- FLA comparison ----
    if args.test in ("fla", "all"):
        print("\n" + "=" * 60)
        print("FLA COMPARISON TESTS")
        print("=" * 60)
        if not HAS_FLA:
            print("⊘ SKIPPED: fla library not available")
        else:
            for tag, kw in [
                ("Small (64x64)", dict(B=1, S=64, H=2, D=128, C=64, decay_val=0.1)),
                ("Zero decay", dict(B=1, S=64, H=2, D=128, C=64, decay_val=0.0)),
                ("Multi-chunk (256)", dict(B=1, S=256, H=4, D=128, C=64, decay_val=0.1)),
                ("Decay 0.05", dict(B=1, S=128, H=4, D=128, C=64, decay_val=0.05)),
                ("Decay 0.2", dict(B=1, S=128, H=4, D=128, C=64, decay_val=0.2)),
                ("Decay 0.5", dict(B=1, S=128, H=4, D=128, C=64, decay_val=0.5)),
                ("Batch", dict(B=2, S=128, H=4, D=128, C=64, decay_val=0.1)),
            ]:
                results.append((f"FLA {tag}", test_against_fla(**kw, verbose=args.verbose)))

    # ---- h0 / ht state tests ----
    if args.test in ("h0ht", "all"):
        print("\n" + "=" * 60)
        print("H0/HT STATE TESTS (vs PyTorch ref)")
        print("=" * 60)
        for tag, kw in [
            ("Small", dict(B=1, S=64, H=2, D=128, C=64, decay_val=0.1)),
            ("Multi-chunk", dict(B=1, S=256, H=4, D=128, C=64, decay_val=0.1)),
            ("Batch", dict(B=2, S=128, H=4, D=128, C=64, decay_val=0.2)),
        ]:
            results.append((f"h0/ht {tag}", test_initial_and_final_state(**kw, verbose=args.verbose)))

        print("\n" + "=" * 60)
        print("H0/HT STATE TESTS (vs FLA)")
        print("=" * 60)
        if not HAS_FLA:
            print("⊘ SKIPPED: fla library not available")
        else:
            for tag, kw in [
                ("Small", dict(B=1, S=64, H=2, D=128, C=64, decay_val=0.1)),
                ("Multi-chunk", dict(B=1, S=256, H=4, D=128, C=64, decay_val=0.1)),
                ("Batch", dict(B=2, S=128, H=4, D=128, C=64, decay_val=0.2)),
            ]:
                results.append((f"FLA h0/ht {tag}", test_against_fla_with_state(**kw, verbose=args.verbose)))

    # ---- Varlen tests ----
    if args.test in ("varlen", "all"):
        print("\n" + "=" * 60)
        print("VARLEN TESTS")
        print("=" * 60)
        for tag, kw in [
            ("Single seq", dict(H=4, S=128, D=128, C=64, decay_val=0.1)),
            ("Single long", dict(H=4, S=256, D=128, C=64, decay_val=0.1)),
            ("Decay 0.5", dict(H=4, S=128, D=128, C=64, decay_val=0.5)),
        ]:
            results.append((f"Varlen {tag}", test_varlen_single_seq(**kw, verbose=args.verbose)))

        for tag, kw in [
            ("Multi 3-seq", dict(seq_lens=[128, 64, 192], H=4, D=128, C=64, decay_val=0.1)),
            ("Multi 2-seq", dict(seq_lens=[256, 128], H=4, D=128, C=64, decay_val=0.1)),
            ("Multi decay", dict(seq_lens=[128, 128], H=4, D=128, C=64, decay_val=0.5)),
        ]:
            results.append((f"Varlen {tag}", test_varlen_multi_seq(**kw, verbose=args.verbose)))

        for tag, kw in [
            ("h0 indirect", dict(seq_lens=[128, 64], H=4, D=128, C=64, decay_val=0.1)),
            ("h0 decay 0.2", dict(seq_lens=[128, 64], H=4, D=128, C=64, decay_val=0.2)),
        ]:
            results.append((f"Varlen {tag}", test_varlen_with_initial_state(**kw, verbose=args.verbose)))

        print("\n" + "=" * 60)
        print("VARLEN TESTS (vs PyTorch ref)")
        print("=" * 60)
        for tag, kw in [
            ("vs ref 2-seq", dict(seq_lens=[128, 192], H=4, D=128, C=64, decay_val=0.1)),
            ("vs ref decay", dict(seq_lens=[64, 128], H=4, D=128, C=64, decay_val=0.5)),
        ]:
            results.append((f"Varlen {tag}", test_varlen_against_pytorch_ref(**kw, verbose=args.verbose)))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"{'✓ PASSED' if passed else '✗ FAILED'}: {name}")

    total = sum(p for _, p in results)
    print(f"\nTotal: {total}/{len(results)} tests passed")
    return total == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
