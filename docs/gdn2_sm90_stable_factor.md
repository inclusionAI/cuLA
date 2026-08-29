# GDN2 SM90 blockwise-rebased intra-chunk factorization

## Problem

The released kernel formed the intra-chunk causal matrices from a
chunk-start-referenced split:

```text
A[i,j] = sum_c ( q_i[c] * exp(G_i[c]) ) * ( k_j[c] * exp(-G_j[c]) )
```

with `G` the chunk-local inclusive per-channel log-decay prefix. The
`exp(-G_j)` factor is an unbounded inverse decay: any 64-token chunk whose
per-channel accumulated decay exceeds `ln(FP32_MAX) ~ 88.72` overflows the
`k`-side operand even though every masked product `exp(G_i - G_j) <= 1`
stays finite. With a uniform per-element `g`, the cliff sits at
`|g| > 88.72 / 64 ~ 1.386`, inside the public contract `g <= 0`.
Because `G` is monotone, the overflow always pairs with an underflowed
`q`-side row, so masked-region entries become `0 * inf = NaN` and the NaN
propagates through the erase-Gram inverse and the recurrent state to every
later token of the sequence.

## Factorization

The chunk is split into four 16-token sub-blocks. `Gs(I)` denotes the
inclusive prefix at the first token of sub-block `I` (clamped to the last
valid token for partial tails). The prepared operands are:

```text
q~[i]  = q_i  * exp(G_i - Gs(B(i)))          # <= 1, span <= 15 tokens
e~[i]  = b_i * k_i * exp(G_i - Gs(B(i)))     # <= 1, span <= 15 tokens
k~'[j] = k_j  * exp(Gs(B(j)) - G_j)          # >= 1, span <= 15 tokens
```

where `B(t) = t // 16`. Per-pair block products then satisfy:

- diagonal pairs `(I,I)`: `q~ k~'` is exactly `q k exp(G_i - G_j)`;
- off-diagonal pairs `(I,J), J < I`: the product must be corrected by the
  per-channel factor `s'[c] = exp(Gs(I) - Gs(J))[c] <= 1`, folded into the
  left fragment before the MMA. The folded left operand equals
  `q exp(G_i - Gs(J)) <= 1`.

The only intermediate with a positive exponent is `k~'`, whose span is 15
token gaps instead of 63, moving the uniform-`g` overflow cliff from
`~1.386` to `88.72 / 15 ~ 5.91`. The documented contract is
`g in [-5, 0]`, matching the pinned FLA GDN2 `safe_gate` documented range;
at `g = -5` the largest `k~'` exponent is `75`, a factor `~9e5` below the
BF16/FP32 overflow boundary.

The per-channel correction factors are served from a small FP32 SMEM
buffer of block-boundary decay ratios written during preparation:

```text
delta[0][c] = exp(Gs(0)[c])                  # = exp(g_0[c]) <= 1
delta[m][c] = exp(Gs(m)[c] - Gs(m-1)[c])     # <= 1, m in {1,2,3}
```

`s'` for pair `(I,J)` is the running product `delta[J+1] * ... * delta[I]`,
computed in registers by the factor warp group; the recurrent-state warp
groups consume the same rows to advance the state scale block by block
(below). Fully-invalid tail blocks store `delta = 1`.

## Factor stage

`A_qk` (causal, scaled, BF16) and the strict-lower erase Gram (FP16, into
the collective-inverse workspace) are computed per sub-block pair by the
factor warp group with warp-level `m16n8k16` MMAs: ten lower/diagonal
16x16 blocks per matrix, four warps, the `k~'` right fragment shared
between both matrices of a pair. Upper blocks are zero-filled. The
collective inverse, the BF16 `A_kk` publication, and every downstream
consumer tile are unchanged.

## Recurrent-state warp groups

The inter-chunk output and erase projections previously consumed
`q_bar = q exp(G_i)` and `erase_bar = b k exp(G_i)` as single
`n=64` WGMMAs against the register-resident state. Those tiles now hold
`q~`/`e~`, so both projections are issued as four `n=16` slices with the
state fragments rescaled by `delta[I]` before slice `I`; after slice 3 the
state carries `exp(Gs(3))`, and the end-of-chunk update multiplies by the
retuned `gamma_end = exp(G_end - Gs(3)) <= 1` so the recurrence
`S <- S exp(G_end) + k_tail v_new^T` is unchanged. `key_tail`
(`k exp(G_end - G)`) and the final-tail carry path are untouched: they were
already in stable form.

## Contract

- `g` must be finite, non-positive, and elementwise `>= -5.0`
  (`docs/gdn2_sm90_api.md`); `validate_inputs=True` enforces the bound
  synchronously, and the default path documents it as a caller
  precondition, exactly like the existing `g <= 0` and gate-range
  preconditions.
- The bound is a per-15-token-channel-sum requirement
  (`sum |g|` over any 15 consecutive in-block gaps `< 88.7`); the
  elementwise `-5` bound is the documented sufficient condition.
