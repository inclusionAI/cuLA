# KDA + CSA 1M context optimization study

This note records measured decisions for the packed-THD workload with 20
sequences of 1,024 tokens and two long sequences. Unless stated otherwise,
the run uses four physical GPUs, 1,048,576 global tokens, BF16 operands, FP32
Kimi CP merge, deterministic CSA selection, `top_k=512`, and block size 4.

## Exact KDA checkpoint reuse

cuDNN KDA can retain the state at every native 16-token chunk and consume it
directly in backward. cuLA exposes this as the explicit
`checkpoint_every_n_tokens` option. The default remains zero because saved
states live across the forward/backward interval for every layer, while the
recompute workspace can be reused between layers.

| Workload | Checkpoint | Critical E2E | Critical KDA | Peak allocated |
|---|---:|---:|---:|---:|
| 128K / CP4 | 0 | 9.780 ms | 9.092 ms | 6.362 GiB |
| 128K / CP4 | 16 | 8.987 ms | 8.159 ms | 5.974 GiB |
| 1M / CP4 | 0 | 208.798 ms | 135.050 ms | 49.904 GiB |
| 1M / CP4 | 16 | 204.927 ms | 111.352 ms | 49.750 GiB |

At 128K, checkpoint reuse improves the hybrid critical path by 8.1%. At 1M,
it improves the slowest KDA rank by 17.5%, but hybrid E2E improves only 1.9%
because deterministic CSA selection becomes the critical path. Output, proxy
loss, and all five input-gradient norms were exactly equal to the recompute
path on every rank in both measured workloads.

## CSA scratch is not the remaining answer

The default adaptive policy caps the compact-logit scratch at 8 GiB. Increasing
the cap reduces the number of exact query windows, but larger selector kernels
contend more strongly with KDA on the overlap stream.

| Scratch cap | Max windows | Critical E2E | Critical select | Critical KDA | Peak allocated |
|---:|---:|---:|---:|---:|---:|
| 8 GiB | 12 | 204.927 ms | 204.524 ms | 111.352 ms | 49.750 GiB |
| 16 GiB | 6 | 205.090 ms | 204.742 ms | 194.178 ms | 59.701 GiB |
| 32 GiB | 3 | 204.129 ms | 180.216 ms | 203.184 ms | 73.649 GiB |
| 64 GiB | 2 | 204.970 ms | 172.687 ms | 204.300 ms | 105.649 GiB |

The 32 GiB point buys less than 0.4% E2E for about 24 GiB more peak memory;
64 GiB regresses. The 8 GiB cap remains the better default Pareto point.

## Why contiguous CP is imbalanced for CSA

CSA selection work is proportional to the causal position of each query. A
contiguous token partition assigns later regions of each long sequence to a
small subset of ranks. For this workload the exact compact-candidate counts are:

| Layout | Rank 0 | Rank 1 | Rank 2 | Rank 3 | Max / min |
|---|---:|---:|---:|---:|---:|
| Contiguous | 7.303B | 24.428B | 9.235B | 25.099B | 3.44x |
| Alternating by long sequence | 16.516B | 16.516B | 16.516B | 16.516B | 1.00x |

The alternating plan keeps short samples whole. Each rank receives five of the
20 short samples, an ascending shard of the first long sample, and a descending
shard of the second long sample. Every rank still owns exactly 262,144 tokens;
the total arithmetic is unchanged, but early and late causal regions cancel.

A layout-native DSA benchmark includes deterministic index pooling, all-gather,
global block reconstruction, discontiguous key remapping, and exact selection:

| Query layout | Critical DSA time | Peak allocated |
|---|---:|---:|
| Contiguous | 166.497 ms | 12.200 GiB |
| Alternating | 109.173 ms | 12.174 GiB |

The measured critical path improves by 34.4%; remap plus communication stays
below 1.2 ms on every balanced rank. No arithmetic precision, Top-K semantics,
or deterministic mode changes.

This is currently a layout study, not a production KDA E2E claim. A correct
integration needs a multi-span Kimi CP plan: cuDNN must receive one initial
state per local span, and forward/backward summaries must be routed by
`(sequence, segment-order)` rather than by physical rank order. The layout must
be produced before the model stack so layers do not pay token-layout conversion.
The benchmark and CPU planner live in `benchmarks/large_cp_emulation` to keep
this not-yet-production policy separate from the contiguous runtime.
