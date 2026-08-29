# Latest CAKE vs. cuLA FlashKDA Intracard CP on GB200

## TL;DR

For `B=1`, `D=128`, BF16, and `H=2/4/8`, cuLA FlashKDA auto intracard CP
was faster than the latest official CAKE dispatcher at every tested length from
16K through 128K. The measured advantage was **1.74x--4.52x**. There is no
CAKE-to-CP crossover inside this range: FlashKDA CP already wins at 16K.

## Versions

| Implementation | Exact version |
|---|---|
| FlashInfer CAKE | [`flashinfer-ai/flashinfer@e425c7b0`](https://github.com/flashinfer-ai/flashinfer/commit/e425c7b029ca90d5d01ff207913b070863d35a5b), latest `main` at measurement time |
| cuLA FlashKDA CP | [`inclusionAI/cuLA@6c13747c`](https://github.com/inclusionAI/cuLA/commit/6c13747cde06fe3dcfda4b4505dca22c3f019991), [PR #124](https://github.com/inclusionAI/cuLA/pull/124) |

CAKE was called through `recurrent_kda(..., backend="cake")`. cuLA was called
through `cula.kda.flashkda.cula_kda_prefill(..., use_intracard_cp="auto")`;
auto CP was active for all rows. This cuLA path is the SM90-derived FlashKDA
compatibility implementation running on SM100, not the separate SM100-native
modular KDA path. For CAKE kernel background, see
[this CAKE optimization overview](https://zhuanlan.zhihu.com/p/2068499679076259239).

## Results

`1K = 1024` tokens. Speedup is `CAKE latency / FlashKDA CP latency`.

| H | T | CAKE route | CAKE (ms) | FlashKDA CP (ms) | CP speedup |
|---:|---:|---|---:|---:|---:|
| 2 | 16K | small-BH M128 | 0.622 | 0.278 | 2.23x |
| 2 | 32K | small-BH M128 | 1.221 | 0.420 | 2.91x |
| 2 | 64K | BT16 + M64 chain | 1.858 | 0.573 | 3.24x |
| 2 | 128K | BT16 + M64 chain | 3.698 | 0.818 | 4.52x |
| 4 | 16K | small-BH M128 | 0.625 | 0.295 | 2.12x |
| 4 | 32K | small-BH M128 | 1.227 | 0.423 | 2.90x |
| 4 | 64K | BT16 + M64 chain | 1.910 | 0.671 | 2.85x |
| 4 | 128K | BT16 + M64 chain | 3.794 | 1.163 | 3.26x |
| 8 | 16K | small-BH M128 | 0.607 | 0.349 | 1.74x |
| 8 | 32K | small-BH M128 | 1.198 | 0.593 | 2.02x |
| 8 | 64K | BT16 + M64 chain | 2.005 | 1.090 | 1.84x |
| 8 | 128K | BT16 + M64 chain | 3.972 | 2.075 | 1.91x |

## Method and Scope

Measurements used an NVIDIA GB200 (SM100, 152 SMs), `B=1`, `H=HV`,
`D=128`, BF16 Q/K/V/G and beta logits, fused Q/K normalization and gate
processing, no initial/final state, and preallocated output buffers. Each point
uses 10 warmups, 40 CUDA-event samples, and three alternating-order rounds; the
reported value is the median of the round-level middle-half means. The harness
is [`bench_cake_vs_flashkda_cp.py`](https://github.com/inclusionAI/cuLA/blob/6c13747cde06fe3dcfda4b4505dca22c3f019991/benchmarks/bench_cake_vs_flashkda_cp.py).

Across the matrix, output relative RMS difference was `0.492%--0.624%` and
maximum absolute difference was at most `1.10e-3`. Results are specific to the
measured shapes and execution contract; batch size, state output, head
dimension, or another GPU can change the ordering.
