# Misc Analysis Tools

Utility programs for analyzing CuTe MMA layouts and deriving the BF16→TF32 register shuffle.

These are host/device programs used during development to understand the TV layout differences
between `SM80_16x8x8_F32BF16BF16F32_TN` and `SM80_16x8x8_F32TF32TF32F32_TN`, and to verify
the correctness of the register-level layout conversion.

## Files

| File | Description | Runs on |
|---|---|---|
| `analyze_layout.cu` | Print TV layouts, fragment shapes, and tiled copy layouts for BF16 vs TF32 MMA | Host (compile-time CuTe introspection) |
| `analyze_layout2.cu` | Extended version: also prints retile_D layouts and copy atom details | Host |
| `analyze_permutation.cu` | Compute the full BF16→TF32 permutation table for operand A and B, check if shuffles stay within warp | Host |
| `analyze_shuffle.cu` | Derive per-thread `(row, k)` ownership and the exact `__shfl_sync` source pattern | Host |
| `verify_conversion.cu` | GPU kernel that prints TV→(row,k) mapping for all 32 threads, verifying the layout formulas | Device (needs SM80+) |

## Build

```bash
# Host-only programs (no GPU needed):
nvcc -std=c++17 -I<cutlass_include_path> analyze_layout.cu -o analyze_layout
nvcc -std=c++17 -I<cutlass_include_path> analyze_permutation.cu -o analyze_permutation
nvcc -std=c++17 -I<cutlass_include_path> analyze_shuffle.cu -o analyze_shuffle

# Device program (needs SM80+ GPU):
nvcc -std=c++17 -arch=sm_80 -I<cutlass_include_path> verify_conversion.cu -o verify_conversion
```

## See Also

- [`miscs/bf16_to_tf32_layout_conversion.md`](bf16_to_tf32_layout_conversion.md) — Full documentation of the conversion algorithm
- [`csrc/kda/sm90/collective/common.hpp`](../csrc/kda/sm90/collective/common.hpp) — Production implementation
