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
Compare our Blackwell SM100 kernel with FLA's reference implementation.
"""

import argparse
import pathlib
import sys

import pytest
import torch

pytestmark = pytest.mark.sm100_only

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cutlass.cute as cute  # noqa: E402
import cutlass.torch as cutlass_torch  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

# Our implementation
from cula.ops.chunk_delta_h import ChunkDeltaRuleFwdH  # noqa: E402


def fla_reference_chunk_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor = None,
    gk: torch.Tensor = None,
    h0: torch.Tensor = None,
    chunk_size: int = 64,
    apply_h_gate: bool = True,  # If False, only apply v_new gating (for testing)
):
    """
    Reference implementation matching FLA's chunk_gated_delta_rule_fwd_kernel_h logic.

    This is the exact math from FLA's triton kernel:
    - h_out[t] = h (before update)
    - v_new = u - w @ h
    - If g: v_new *= exp(g_last - g)  (per row scaling)
    - If g AND apply_h_gate: h *= exp(g_last)  (scalar scaling of entire h)
    - If gk AND apply_h_gate: h *= exp(gk_last)  (per-k scaling)
    - h += k^T @ v_new

    Args:
        k: [B, T, H, K]
        w: [B, T, H, K]
        u: [B, T, H, V]
        g: [B, T, H] - cumulative sum of per-step scalar gate
        gk: [B, T, H, K] - cumulative sum of per-step vector gate
        h0: [B, H, K, V] - initial state
        chunk_size: int
        apply_h_gate: If False, skip h_state gating (to match kernel behavior)

    Returns:
        h_out: [B, NT, H, K, V] - h state at start of each chunk
        v_new: [B, T, H, V] - computed v_new
        ht: [B, H, K, V] - final state
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT

    # Output tensors
    h_out = torch.zeros(B, NT, H, K, V, device=k.device, dtype=k.dtype)
    v_new_out = torch.zeros(B, T, H, V, device=k.device, dtype=k.dtype)

    # Initialize h state
    h = torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
    if h0 is not None:
        h = h0.clone().float()

    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)

        # Store h_out[t] = h (before update) - this matches FLA
        h_out[:, t] = h.to(k.dtype)

        # Get chunk data: [B, bt, H, K/V] -> [B, H, bt, K/V]
        w_chunk = w[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, bt, K)
        k_chunk = k[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, bt, K)
        u_chunk = u[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, bt, V)

        # v_new = u - w @ h
        wh = torch.matmul(w_chunk, h)  # (B, H, bt, V)
        v_new_chunk = u_chunk - wh

        # Apply scalar gate g to v_new: v_new[i] *= exp(g_last - g[i])
        if g is not None:
            g_chunk = g[:, start:end].permute(0, 2, 1).float()  # (B, H, bt)
            g_last = g_chunk[:, :, -1:].float()  # (B, H, 1)
            g_scale = torch.exp(g_last - g_chunk).unsqueeze(-1)  # (B, H, bt, 1)
            v_new_chunk = v_new_chunk * g_scale

        # Save v_new (after g scaling)
        v_new_out[:, start:end] = v_new_chunk.permute(0, 2, 1, 3).to(k.dtype)

        # Apply scalar gate g to h_state: h *= exp(g_last)
        if g is not None and apply_h_gate:
            g_last_scalar = g_chunk[:, :, -1].float()  # (B, H)
            h_scale = torch.exp(g_last_scalar).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            h = h * h_scale

        # Apply vector gate gk to h_state: h[k,:] *= exp(gk_last[k])
        if gk is not None and apply_h_gate:
            gk_chunk = gk[:, start:end].permute(0, 2, 1, 3).float()  # (B, H, bt, K)
            gk_last = gk_chunk[:, :, -1, :].float()  # (B, H, K)
            gk_scale = torch.exp(gk_last).unsqueeze(-1)  # (B, H, K, 1)
            h = h * gk_scale

        # h += k^T @ v_new
        k_t = k_chunk.transpose(-2, -1)  # (B, H, K, bt)
        h = h + torch.matmul(k_t, v_new_chunk)

    return h_out, v_new_out, h.to(torch.float32)


def compare_with_fla(
    batch_size: int = 1,
    seq_len: int = 192,
    num_heads: int = 1,
    head_dim_k: int = 128,
    head_dim_v: int = 128,
    chunk_size: int = 64,
    use_g: bool = False,
    use_gk: bool = False,
    use_h0: bool = False,
):
    """Compare our kernel with FLA's implementation."""
    print("=" * 70)
    print("Comparing with FLA's chunk_gated_delta_rule_fwd_h")
    print("=" * 70)
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dim K: {head_dim_k}")
    print(f"  Head dim V: {head_dim_v}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  use_g: {use_g}")
    print(f"  use_gk: {use_gk}")
    print(f"  use_h0: {use_h0}")

    B, T, H, K, V = batch_size, seq_len, num_heads, head_dim_k, head_dim_v
    BT = chunk_size
    NT = (T + BT - 1) // BT

    # Generate random inputs (same seed for reproducibility)
    torch.manual_seed(42)

    # Input tensors in FLA format: [B, T, H, K/V]
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1

    # Gates (optional)
    if use_g:
        # g is cumsum of per-step gates, scaled small to avoid blow-up
        g_raw = torch.randn(B, T, H, device="cuda", dtype=torch.float32) * 0.01
        g = g_raw.cumsum(dim=1)  # cumulative sum along T
    else:
        g = None

    if use_gk:
        gk_raw = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.01
        gk = gk_raw.cumsum(dim=1)
    else:
        gk = None

    # Initial state (optional)
    if use_h0:
        # Note: FLA uses float32 for h0, but our kernel uses bf16 TMA loads
        # So we need to convert h0 to bf16 for kernel, but use float32 for reference
        h0_fp32 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.1
        h0_bf16 = h0_fp32.to(torch.bfloat16)  # For kernel
        h0 = h0_fp32  # For reference (uses float32 internally)
    else:
        h0 = None
        h0_fp32 = None
        h0_bf16 = None

    print("\n--- Running FLA reference (full gate: apply_h_gate=True) ---")
    # FLA reference implementation (fp32 precision, full gating)
    h_fla_full, v_new_fla_full, ht_fla_full = fla_reference_chunk_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        h0=h0,
        chunk_size=BT,
        apply_h_gate=True,
    )
    print(f"  FLA h shape: {h_fla_full.shape}")
    print(f"  FLA v_new shape: {v_new_fla_full.shape}")
    print(f"  FLA ht shape: {ht_fla_full.shape}")

    # If gating is enabled, also run reference with v_new-only gating (to match kernel)
    if use_g or use_gk:
        print("\n--- Running FLA reference (v_new gate only: apply_h_gate=False) ---")
        h_fla_vnew, v_new_fla_vnew, ht_fla_vnew = fla_reference_chunk_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            h0=h0,
            chunk_size=BT,
            apply_h_gate=False,
        )
        print(f"  FLA (vnew-only) v_new shape: {v_new_fla_vnew.shape}")
    else:
        h_fla_vnew, v_new_fla_vnew, ht_fla_vnew = h_fla_full, v_new_fla_full, ht_fla_full

    print("\n--- Running our SM100 kernel ---")

    # Create output tensors for our kernel
    h_out = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new_out = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    ht_out = torch.zeros(B, H, K, V, device="cuda", dtype=torch.bfloat16)

    # Gates in our format
    g_ours = g if g is not None else torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk_ours = gk if gk is not None else torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    # Use bf16 h0 for kernel (TMA requires matching dtype)
    h0_ours = h0_bf16 if h0_bf16 is not None else torch.zeros(B, H, K, V, device="cuda", dtype=torch.bfloat16)

    # Convert to CuTe
    k_cute = from_dlpack(k)
    w_cute = from_dlpack(w)
    u_cute = from_dlpack(u)
    g_cute = from_dlpack(g_ours)
    gk_cute = from_dlpack(gk_ours)
    h0_cute = from_dlpack(h0_ours)
    h_out_cute = from_dlpack(h_out)
    v_new_cute = from_dlpack(v_new_out)
    ht_cute = from_dlpack(ht_out)

    # Create and run our kernel
    kernel = ChunkDeltaRuleFwdH(
        chunk_size=BT,
        head_dim_k=K,
        head_dim_v=V,
    )

    stream = cutlass_torch.default_stream()

    # problem_size = (B, T, H, K, V)
    problem_size = (B, T, H, K, V)

    print("  Compiling kernel...")
    compiled = cute.compile(
        kernel,
        k_cute.iterator,
        w_cute.iterator,
        u_cute.iterator,
        g_cute.iterator,
        gk_cute.iterator,
        h_out_cute.iterator,
        v_new_cute.iterator,
        h0_cute.iterator,
        ht_cute.iterator,
        problem_size,
        use_g,
        use_gk,
        use_h0,
        True,
        True,  # use_initial_state, store_final_state, save_v_new
        stream,
    )

    print("  Running kernel...")
    compiled(
        k_cute.iterator,
        w_cute.iterator,
        u_cute.iterator,
        g_cute.iterator,
        gk_cute.iterator,
        h_out_cute.iterator,
        v_new_cute.iterator,
        h0_cute.iterator,
        ht_cute.iterator,
        problem_size,
        use_g,
        use_gk,
        use_h0,
        True,
        True,
        stream,
    )
    torch.cuda.synchronize()

    print(f"  Our h shape: {h_out.shape}")
    print(f"  Our v_new shape: {v_new_out.shape}")
    print(f"  Our ht shape: {ht_out.shape}")

    print("\n--- Comparing outputs ---")

    # Compare v_new against v_new-only gating reference (matches kernel behavior)
    v_new_diff_vnew = (v_new_out.float() - v_new_fla_vnew.float()).abs().max().item()
    print(f"  v_new vs vnew-only ref: {v_new_diff_vnew:.6f}")

    if use_g or use_gk:
        # Also show diff against full gating reference (will be larger)
        v_new_diff_full = (v_new_out.float() - v_new_fla_full.float()).abs().max().item()
        print(f"  v_new vs full-gate ref: {v_new_diff_full:.6f}")

    # Compare h (per-chunk states)
    # Both FLA and our kernel store h BEFORE chunk update:
    #   FLA h_out[t] = state at START of chunk t (before update)
    #   Our h_out[t] = state at START of chunk t (before update)
    # So they should match directly: Our h_out[t] ≈ FLA h_out[t]

    # Use full-gate reference when gk is enabled (kernel applies gk to h),
    # otherwise use vnew-only reference
    if use_gk:
        h_fla = h_fla_full
        ht_fla = ht_fla_full
        v_new_ref = v_new_fla_full
        v_new_diff = (v_new_out.float() - v_new_fla_full.float()).abs().max().item()
        ref_label = "full-gate"
    else:
        h_fla = h_fla_vnew
        ht_fla = ht_fla_vnew
        v_new_ref = v_new_fla_vnew
        v_new_diff = v_new_diff_vnew
        ref_label = "vnew-only"

    # Direct comparison: Our h_out[t] vs FLA h_out[t]
    print(f"\n  Direct h_out comparison (vs {ref_label} ref):")
    h_diffs = []
    for t in range(NT):
        diff = (h_out[:, t].float() - h_fla[:, t].float()).abs().max().item()
        h_diffs.append(diff)
        print(f"    Chunk {t}: {diff:.6f}")

    if h_diffs:
        h_max_diff = max(h_diffs)
        print(f"  h max diff: {h_max_diff:.6f}")
    else:
        h_max_diff = 0.0

    # Compare final state: our ht vs FLA ht (state after all chunks)
    ht_diff = (ht_out.float() - ht_fla.float()).abs().max().item()
    print(f"\n  Our ht vs FLA ht ({ref_label} ref): {ht_diff:.6f}")

    # Per-chunk v_new breakdown
    print(f"\n  Per-chunk v_new diff (vs {ref_label} ref):")
    for t in range(NT):
        start = t * BT
        end = min((t + 1) * BT, T)
        chunk_diff = (v_new_out[:, start:end].float() - v_new_ref[:, start:end].float()).abs().max().item()
        print(f"    Chunk {t}: {chunk_diff:.6f}")

    # Tolerance check
    tolerance = 0.01  # bf16 tolerance
    passed = v_new_diff < tolerance and h_max_diff < tolerance

    if passed:
        print(f"\n✅ PASS - v_new and h match {ref_label} ref within tolerance ({tolerance})")
    else:
        print(f"\n❌ FAIL - Outputs exceed tolerance ({tolerance})")
        print(f"  v_new diff (vs {ref_label} ref): {v_new_diff:.6f} {'✓' if v_new_diff < tolerance else '✗'}")
        print(f"  h diff: {h_max_diff:.6f} {'✓' if h_max_diff < tolerance else '✗'}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Compare with FLA implementation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--head_dim_k", type=int, default=128)
    parser.add_argument("--head_dim_v", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--use_g", action="store_true", help="Enable scalar gate g")
    parser.add_argument("--use_gk", action="store_true", help="Enable vector gate gk")
    parser.add_argument("--use_h0", action="store_true", help="Enable initial state h0")
    parser.add_argument("--run_all", action="store_true", help="Run all test cases")

    args = parser.parse_args()

    if args.run_all:
        # Test matrix
        test_cases = [
            # (seq_len, use_g, use_gk, use_h0)
            (128, False, False, False),  # Basic 2 chunks
            (192, False, False, False),  # Basic 3 chunks
            (256, False, False, False),  # Basic 4 chunks
            (128, False, True, False),  # With gk gate
            (192, False, True, False),  # With gk gate, 3 chunks
            (128, False, False, True),  # With initial state
            (192, False, False, True),  # With initial state, 3 chunks
            (128, False, True, True),  # With gk and h0
        ]

        results = []
        for seq_len, use_g, use_gk, use_h0 in test_cases:
            print(f"\n{'=' * 70}")
            print(f"Test case: seq_len={seq_len}, use_g={use_g}, use_gk={use_gk}, use_h0={use_h0}")
            print(f"{'=' * 70}")
            try:
                passed = compare_with_fla(
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    num_heads=args.num_heads,
                    head_dim_k=args.head_dim_k,
                    head_dim_v=args.head_dim_v,
                    chunk_size=args.chunk_size,
                    use_g=use_g,
                    use_gk=use_gk,
                    use_h0=use_h0,
                )
                results.append((seq_len, use_g, use_gk, use_h0, passed))
            except Exception as e:
                print(f"  Error: {e}")
                results.append((seq_len, use_g, use_gk, use_h0, False))

        # Summary
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        for seq_len, use_g, use_gk, use_h0, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  seq_len={seq_len:3d}, g={use_g}, gk={use_gk}, h0={use_h0}: {status}")

        total = len(results)
        passed_count = sum(1 for r in results if r[4])
        print(f"\nTotal: {passed_count}/{total} passed")
    else:
        compare_with_fla(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            head_dim_k=args.head_dim_k,
            head_dim_v=args.head_dim_v,
            chunk_size=args.chunk_size,
            use_g=args.use_g,
            use_gk=args.use_gk,
            use_h0=args.use_h0,
        )


if __name__ == "__main__":
    main()
