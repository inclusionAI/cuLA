# BF16 → TF32 MMA Register Layout Conversion

## Overview

The KDA forward kernel uses TF32 MMA (`SM80_16x8x8_F32TF32TF32F32_TN`) for subchunk QK/KK
computation, but loads Q/K from shared memory using BF16 MMA LDSM (`ldmatrix`) for efficiency.
Since BF16 and TF32 MMA use **different thread-value (TV) layouts**, a register-level shuffle is
required to rearrange values across threads before the TF32 MMA can consume them.

This document derives the shuffle algorithm step by step.

---

## 1. TV Layout Definitions

CuTe represents a MMA's operand mapping as a **TV layout**: `(Thread, Value) → (M, K)`.
Both BF16 and TF32 atoms have the same *shape* per thread, but different *strides*.

### Operand A — 16 values per thread (4 k-atoms × 4 values each)

```
BF16  LayoutA_TV:  ((_4,_8), (_2,_2)) : ((_32,_1), (_16,_8))
TF32  LayoutA_TV:  ((_4,_8), (_2,_2)) : ((_16,_1), (_8,_64))
                     ↑T: 32 threads     ↑V: 4 values per k-atom
```

The thread index decomposes as `tid = t0 + 4*t1` where `t0 ∈ [0,4)`, `t1 ∈ [0,8)`.
The value index decomposes as `v = v0 + 2*v1` where `v0, v1 ∈ {0,1}`.

Applying the strides to get the `(M, K)` coordinate each thread-value pair maps to:

| | BF16 `(M, K)` | TF32 `(M, K)` |
|---|---|---|
| Formula | `(t1 + 8*v1, 32*t0 + 16*v0)` | `(t1 + 8*v0, 16*t0 + 64*v1)` |
| **K per thread** | `32*t0 + {0, 16}` → **two consecutive** at stride 16 | `16*t0 + {0, 64}` → **two at stride 64** |
| **v0 selects** | **K offset** (stride 16) | **M offset** (stride 8) |
| **v1 selects** | **M offset** (stride 8) | **K offset** (stride 64) |

**However**, the above strides are for the full 16×8 MMA atom (one k-iteration).
For the `SM80_16x8x8` shape the MMA is tiled across K with 4 atoms (`K_per_atom = 8` for
BF16 (each bf16 = 2B, so 8 values pack into 16B), `K_per_atom = 8` for TF32 (each tf32 = 4B,
so 8 values pack into 32B — but the atom shape is 16×8×8)).

In practice, the fragment for each k-atom has 4 float values and the critical fact is:

| | BF16 MMA | TF32 MMA |
|---|---|---|
| K values owned by thread `t0` | `{2*t0, 2*t0+1}` | `{t0, t0+4}` |
| v0 = 0 | K = 2*t0 | M = row 0 |
| v0 = 1 | K = 2*t0+1 | M = row 1 |
| v1 = 0 | M = row 0 | K = t0 |
| v1 = 1 | M = row 1 | K = t0+4 |

### Operand B — 8 values per thread (4 k-atoms × 2 values each)

```
BF16  LayoutB_TV:  ((_4,_8), _2) : ((_16,_1), _8)
TF32  LayoutB_TV:  ((_4,_8), _2) : ((_8,_1), _32)
```

K ownership is the same as operand A:

| | BF16 MMA | TF32 MMA |
|---|---|---|
| K values per thread `t0` | `{2*t0, 2*t0+1}` | `{t0, t0+4}` |

---

## 2. Deriving the Shuffle (Operand B — simpler case first)

### Goal

Thread `t0` currently holds K = `{2*t0, 2*t0+1}` (BF16 layout). After conversion it must
hold K = `{t0, t0+4}` (TF32 layout).

### Step 1: Where does each target K value live in BF16 layout?

For TF32 value `v_tf32 = 0` (target K = `t0`):
- In BF16, K = `t0` is held by thread `t0_src = t0 / 2` (integer division).
- Within that thread: if `t0` is even → `v0_bf16 = 0` (K = 2·(t0/2) = t0 ✓), if `t0` is odd → `v0_bf16 = 1` (K = 2·(t0/2)+1 = t0 ✓).

For TF32 value `v_tf32 = 1` (target K = `t0 + 4`):
- In BF16, K = `t0+4` is held by thread `t0_src = (t0+4) / 2 = t0/2 + 2`.
- Same even/odd selection for `v0_bf16`.

### Step 2: Compute source lane IDs

Thread groups of 4 (`t0 ∈ [0,4)`) share a row. The full lane index is `lane = t0_src + (tid & ~3)`:

```
src_lane_lo = (t0 / 2)     + (tid & ~3)   // source for K = t0
src_lane_hi = (t0 / 2 + 2) + (tid & ~3)   // source for K = t0 + 4
```

### Step 3: Shuffle and select

Each thread shuffles both `v0_bf16 = 0` and `v0_bf16 = 1` values from the source thread,
then picks the correct one based on parity:

```cuda
float val0 = frag_B(2*j);      // v0_bf16=0, K = 2*t0
float val1 = frag_B(2*j + 1);  // v0_bf16=1, K = 2*t0+1

// Get both v0_bf16 values from source thread for K = t0
float recv0_lo = __shfl_sync(0xFFFFFFFF, val0, src_lane_lo);
float recv1_lo = __shfl_sync(0xFFFFFFFF, val1, src_lane_lo);
// Get both v0_bf16 values from source thread for K = t0+4
float recv0_hi = __shfl_sync(0xFFFFFFFF, val0, src_lane_hi);
float recv1_hi = __shfl_sync(0xFFFFFFFF, val1, src_lane_hi);

bool sel_odd = (t0 & 1);
frag_B(2*j)     = sel_odd ? recv1_lo : recv0_lo;  // v_tf32=0 → K = t0
frag_B(2*j + 1) = sel_odd ? recv1_hi : recv0_hi;  // v_tf32=1 → K = t0+4
```

**Cost per k-atom:** 4 `__shfl_sync` + 2 selects. 4 k-atoms × 4 shuffles = **16 shuffles total** for operand B.

### Concrete example (t0 = 3)

```
BF16: t0=3 holds K={6,7}  →  val0=data[K=6], val1=data[K=7]

Target (TF32): K={3, 7}
  K=3:  src_t0 = 3/2 = 1, src has K={2,3}, need v0_bf16=1 (K=3)  ← sel_odd=true → recv1_lo ✓
  K=7:  src_t0 = 3/2+2 = 3, src has K={6,7}, need v0_bf16=1 (K=7)  ← sel_odd=true → recv1_hi ✓
```

---

## 3. Deriving the Shuffle (Operand A)

Operand A has 4 values per k-atom instead of 2, indexed as `v = v0 + 2*v1`:

| Index in fragment | BF16 `(v0, v1)` meaning | TF32 `(v0, v1)` meaning |
|---|---|---|
| 4j + 0 | `(v0=0, v1=0)` → (K-lo, M-0) | `(v0=0, v1=0)` → (M-0, K-lo) |
| 4j + 1 | `(v0=1, v1=0)` → (K-hi, M-0) | `(v0=1, v1=0)` → (M-1, K-lo) |
| 4j + 2 | `(v0=0, v1=1)` → (K-lo, M-1) | `(v0=0, v1=1)` → (M-0, K-hi) |
| 4j + 3 | `(v0=1, v1=1)` → (K-hi, M-1) | `(v0=1, v1=1)` → (M-1, K-hi) |

The K-dimension shuffle is **identical** to operand B. The difference is that v0/v1 swap their
roles (K ↔ M), so for each `v0_tf32` (M selector in TF32), we must pick the right `v1_bf16`
(M selector in BF16):

```
TF32 output at (v0_tf32, v1_tf32):
  v0_tf32 selects M → maps to v1_bf16 = v0_tf32  (same M row)
  v1_tf32 selects K → same shuffle as operand B
```

So the algorithm processes two M rows (`v0_tf32 ∈ {0,1}`), applying the same K-shuffle to each:

```cuda
// Read all 4 inputs BEFORE writing any output (avoids read-after-write hazard)
float in0 = frag_A(4*j + 0);  // (v0_bf16=0, v1_bf16=0)
float in1 = frag_A(4*j + 1);  // (v0_bf16=1, v1_bf16=0)
float in2 = frag_A(4*j + 2);  // (v0_bf16=0, v1_bf16=1)
float in3 = frag_A(4*j + 3);  // (v0_bf16=1, v1_bf16=1)

float out_vals[4];
for (int v0_tf32 = 0; v0_tf32 < 2; v0_tf32++) {
    // Select inputs for this M row: v1_bf16 = v0_tf32
    float val0 = (v0_tf32 == 0) ? in0 : in2;  // v0_bf16=0
    float val1 = (v0_tf32 == 0) ? in1 : in3;  // v0_bf16=1

    // K-dimension shuffle (same as operand B)
    float recv0_lo = __shfl_sync(0xFFFFFFFF, val0, src_lane_lo);
    float recv1_lo = __shfl_sync(0xFFFFFFFF, val1, src_lane_lo);
    float recv0_hi = __shfl_sync(0xFFFFFFFF, val0, src_lane_hi);
    float recv1_hi = __shfl_sync(0xFFFFFFFF, val1, src_lane_hi);

    out_vals[v0_tf32 + 0] = sel_odd ? recv1_lo : recv0_lo;  // v1_tf32=0
    out_vals[v0_tf32 + 2] = sel_odd ? recv1_hi : recv0_hi;  // v1_tf32=1
}

frag_A(4*j + 0) = out_vals[0];
frag_A(4*j + 1) = out_vals[1];
frag_A(4*j + 2) = out_vals[2];
frag_A(4*j + 3) = out_vals[3];
```

**Cost per k-atom:** 8 shuffles (2 M rows × 4 shuffles). 4 k-atoms × 8 = **32 shuffles total** for operand A.

### Why read-before-write matters (in-place conversion)

The output indices `{v0_tf32+0, v0_tf32+2}` overlap with input indices `{0,1,2,3}` in the
same k-iteration. If `v0_tf32=0` writes to position 2 before `v0_tf32=1` reads from position 2,
the input is corrupted. Reading all 4 inputs into local variables first avoids this hazard.
Operand B does not have this problem because its 2 values per k-atom don't alias.

---

## 4. Summary of the Full Conversion Pipeline

```
                    ┌─────────────────────────────┐
                    │  Shared Memory (BF16 data)  │
                    └─────────────┬───────────────┘
                                  │ ldmatrix (BF16 MMA tiled copy)
                                  ▼
                    ┌─────────────────────────────┐
                    │  Registers: BF16 MMA layout │
                    │  (bf16_t values)            │
                    └─────────────┬───────────────┘
                                  │ convert bf16 → float
                                  ▼
                    ┌─────────────────────────────┐
                    │  Registers: float values    │
                    │  (still BF16 MMA layout)    │
                    ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
                    │  ★ Gating: Q*alpha here ★   │  ← alpha also in BF16 layout
                    │  (alpha freed immediately)  │
                    └─────────────┬───────────────┘
                                  │ __shfl_sync shuffle
                                  │ (32 shuffles for A, 16 for B)
                                  ▼
                    ┌─────────────────────────────┐
                    │  Registers: float values    │
                    │  (now TF32 MMA layout)      │
                    └─────────────┬───────────────┘
                                  │ recast<tf32_t>(...) [zero-cost]
                                  │ (MMA hw truncates mantissa)
                                  ▼
                    ┌─────────────────────────────┐
                    │  TF32 MMA (SM80_16x8x8)    │
                    └─────────────────────────────┘
```

Gating (element-wise multiply with alpha/gate) is done **before** the shuffle while Q/K and
alpha share the same BF16 MMA layout. This lets alpha be freed immediately, avoiding 16 extra
registers staying live across 32+ shuffle calls.

---

## 5. Generalizing to BK=64 (NumKAtoms Template Parameter)

### Motivation

The original conversion functions were hardcoded for BK=32, which corresponds to 4 k-atoms
(each SM80_16x8x8 atom covers K=8, so 32/8 = 4 atoms). To support BK=64 subchunks, we need
8 k-atoms. The shuffle pattern per k-atom is **identical** — only the loop count changes.

### Design: Template Parameter `NumKAtoms`

All 5 conversion functions in `csrc/kda/sm90/collective/common.hpp` are templatized with `NumKAtoms`:

```cpp
template <int NumKAtoms = 4, class FragA>
CUTE_DEVICE void convert_bf16_to_tf32_operandA_layout(FragA& frag_A, int local_thread_idx);

template <int NumKAtoms = 4, class FragB>
CUTE_DEVICE void convert_bf16_to_tf32_operandB_layout(FragB& frag_B, int local_thread_idx);

template <int NumKAtoms = 4, class FragA, class FragB>
CUTE_DEVICE void broadcast_row0_operandA_to_operandB_bf16_layout(FragA const& frag_A, FragB& frag_B_first, int local_thread_idx);

template <int NumKAtoms = 4, class FragA, class FragAFirst>
CUTE_DEVICE void broadcast_row0_operandA_bf16_layout(FragA const& frag_A, FragAFirst& frag_A_first, int local_thread_idx);

template <int NumKAtoms = 4, class FragA, class FragB>
CUTE_DEVICE void extract_broadcast_operandA_to_operandB_bf16_layout(FragA const& frag_A, FragB& frag_B);
```

Default `NumKAtoms=4` preserves backward compatibility with BK=32 callers. BK=64 callers
pass `<8>` explicitly.

### What Changes Per Function

The only code change in each function body is the loop bound and the fragment size assertion:

| | BK=32 (`NumKAtoms=4`) | BK=64 (`NumKAtoms=8`) |
|---|---|---|
| Operand A fragment size | 4 × 4 = 16 floats | 8 × 4 = 32 floats |
| Operand B fragment size | 4 × 2 = 8 floats | 8 × 2 = 16 floats |
| Loop bound | `j < 4` | `j < 8` |
| Shuffle count (operand A) | 4 × 8 = 32 | 8 × 8 = 64 |
| Shuffle count (operand B) | 4 × 4 = 16 | 8 × 4 = 32 |

The shuffle pattern within each k-atom iteration is unchanged:
- Source lane computation (`src_lane_lo`, `src_lane_hi`) depends only on `t0` and `tid`, not on `j`.
- The `sel_odd` parity check is the same.
- The read-before-write hazard protection in operand A (reading all 4 inputs into locals) applies identically.

### Why This Works

The BF16 and TF32 TV layouts tile K in atoms of 8. Each atom's thread-value mapping is
self-contained — thread `t0` always holds K = `{2*t0, 2*t0+1}` in BF16 and K = `{t0, t0+4}`
in TF32, regardless of which atom index `j` we're processing. The fragment simply concatenates
atoms: positions `[4j..4j+3]` for operand A, `[2j..2j+1]` for operand B. Doubling the atom
count doubles the fragment size and loop iterations, but each iteration is identical.

### Static Assertions

Each function enforces correctness at compile time:

```cpp
static_assert(NumKAtoms == 4 || NumKAtoms == 8,
              "Only BK=32 (4 k-atoms) and BK=64 (8 k-atoms) supported");
static_assert(decltype(size(frag_A))::value == NumKAtoms * 4);  // operand A
static_assert(decltype(size(frag_B))::value == NumKAtoms * 2);  // operand B
```

This catches mismatches between the caller's `TileShape_SubChunk` K dimension and the
`NumKAtoms` template argument at compile time.

### Caller-Side Changes (Intra Kernel)

In `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp`, the BK=64 configuration uses:

```cpp
// BK=64: 64/8 = 8 k-atoms
constexpr int BK = 64;
constexpr int NK = 2;  // number of K slices (was 4 for BK=32)
using TileShape_SubChunk = Shape<_16, _16, _64>;

// Gating loop processes 8 k-atoms
for (int k = 0; k < 8; k++) { ... }

// All conversion calls use <8>
convert_bf16_to_tf32_operandA_layout<8>(frag_Q_gated, ...);
convert_bf16_to_tf32_operandB_layout<8>(frag_K_gated, ...);
broadcast_row0_operandA_to_operandB_bf16_layout<8>(frag_A, frag_B_first, ...);
broadcast_row0_operandA_bf16_layout<8>(frag_A, frag_A_first, ...);
extract_broadcast_operandA_to_operandB_bf16_layout<8>(frag_A_first, frag_B_first);
```

The shared memory tiling also changes: `flat_divide` uses `Shape<_64, _1>` to produce
NK=2 K-slices (was `Shape<_32, _1>` for NK=4). Smem indexing simplifies from
`make_coord(j%2, j/2)` to `make_coord(_0{}, j)` since each K-slice now spans the full
64-element subchunk dimension.

---

## 6. Bugs Fixed (Summary)

| Bug | Root cause | Fix |
|---|---|---|
| Read-after-write hazard in operand A conversion | In-place write at `v0_tf32=0` corrupted input needed by `v0_tf32=1` | Read all 4 inputs into locals before any writes |
| Unnecessary register pressure from gating | Alpha loaded in TF32 layout, staying alive across the BF16→TF32 shuffle | Load alpha in BF16 layout, gate before shuffle |
| Separate float→tf32 transform pass | `convert_*_layout` wrote float in-place, then a separate `cute::transform` + `make_fragment_like<ElementGatedMMA>` converted float→tf32 | Removed entirely: MMA hardware truncates float→tf32 automatically. Conversion functions now operate purely in-place on float fragments (with `static_assert` enforcing element type). Caller uses `recast<ElementGatedMMA>(float_frag)` for zero-cost typed view. No extra tensor allocation, transform pass, or explicit truncation needed. |

---

## 7. Files

| File | What |
|---|---|
| `csrc/kda/sm90/collective/common.hpp` | `convert_bf16_to_tf32_operandA_layout`, `convert_bf16_to_tf32_operandB_layout` |
| `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp` | Alpha S2R uses BF16 MMA layout; gating before shuffle in `s2r_compute_subchunk_operandA/B` |
| `miscs/analyze_layout.cu` | Print TV layouts, fragment shapes, tiled copy layouts |
| `miscs/analyze_layout2.cu` | Extended: retile_D layouts, copy atom details |
| `miscs/analyze_permutation.cu` | Full BF16→TF32 permutation table, warp-locality check |
| `miscs/analyze_shuffle.cu` | Per-thread (row, k) ownership, `__shfl_sync` source lane derivation |
| `miscs/verify_conversion.cu` | GPU kernel verifying TV→(row,k) mapping on device |
