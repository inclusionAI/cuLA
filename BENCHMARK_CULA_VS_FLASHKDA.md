# Benchmark: cuLA Fully-Fused KDA vs FlashKDA

## Environment

- **GPU**: NVIDIA H200 (sm90, Hopper)
- **cuLA kernel**: `cula.kda.hopper_fused_fwd.cula_kda_prefill` (selected automatically via SM version)
- **FlashKDA kernel**: `flash_kda.fwd`
- H=64, D=128, dtype=bfloat16, beta=bfloat16, safe_gate=True
- Warmup=25, Iters=100 (CUDA event timing)

## Reproduction

```bash
source /root/kevinzeng/FlashKDA/.venv/bin/activate
python benchmarks/bench_cula_flashkda.py            # fixed + varlen
python benchmarks/bench_cula_flashkda.py --mode fixed
python benchmarks/bench_cula_flashkda.py --mode varlen
```

---

## Notes

> **N1 — Accuracy metric (`err_ratio`) here is cuLA vs FlashKDA, not vs FLA reference.**
> For cuLA/FlashKDA accuracy relative to the FLA `chunk_kda` ground truth, see [ACCURACY_CULA_VS_FLASHKDA.md](ACCURACY_CULA_VS_FLASHKDA.md).

> **N2 — State dtype.**
> Both cuLA and FlashKDA use **float32** for initial and final states in this benchmark. Currently the benchmark runs with `has_init_state=False` (both kernels receive `None`); pass `--init_state` to test with non-zero initial states.

> **N3 — Beta semantics differ between kernels.**
> FlashKDA applies `sigmoid` to beta **internally** (receives raw bf16 logits). cuLA (and FLA) apply sigmoid **externally** before calling the kernel (receive post-sigmoid bf16). The benchmark produces both variants from the same random seed: `beta` → FlashKDA, `beta_activated = sigmoid(beta).bf16` → cuLA.

> **N4 — cu_seqlens dtype.**
> cuLA uses `torch.int32`; FlashKDA uses `torch.int64` (long). The benchmark wraps the same values in both dtypes.

> **N5 — `use_qk_l2norm_in_kernel=True` for cuLA.**
> cuLA is called with L2-normalization inside the kernel; q and k tensors are also pre-normalized (F.normalize) to keep comparison semantics consistent with FlashKDA.

---

## Fixed-Length Results

| B | T | err_ratio (cuLA vs FlashKDA) | FlashKDA (ms) | cuLA (ms) | Speedup (cuLA/FlashKDA) |
|--:|--:|--:|--:|--:|--:|
| 1 |    512 | 0.005485 | 0.1074 | 0.2210 | 0.49x |
| 1 |   1024 | 0.005547 | 0.1810 | 0.2789 | 0.65x |
| 1 |   4096 | 0.005529 | 0.6240 | 0.9019 | 0.69x |
| 1 |   8192 | 0.005538 | 1.2093 | 1.7651 | 0.69x |
| 1 |  16384 | 0.005509 | 2.3959 | 3.5219 | 0.68x |
| 2 |    512 | 0.005545 | 0.1342 | 0.2217 | 0.61x |
| 2 |   1024 | 0.005547 | 0.2334 | 0.3106 | 0.75x |
| 2 |   4096 | 0.005538 | 0.8294 | 1.1245 | 0.74x |
| 2 |   8192 | 0.005509 | 1.6268 | 2.1934 | 0.74x |
| 2 |  16384 | 0.005523 | 3.2112 | 4.2048 | 0.76x |

**FlashKDA is 1.3x–2.0x faster than cuLA on fixed-length sequences.**  
The gap narrows as T increases (0.49x at T=512 → ~0.75x at T≥4096 for B=2).

---

## Variable-Length Results

### Summary by distribution type

| Distribution | Typical Speedup (cuLA/FlashKDA) | Trend |
|---|---|---|
| **uniform** (equal lengths) | 0.52x – 0.79x | FlashKDA significantly faster; cuLA penalized by many short seqs |
| **random** (varied lengths) | 0.70x – 0.90x | Gap narrows vs uniform |
| **skewed** (a few long seqs) | 0.91x – **1.00x** | Near parity at large T; cuLA catches up with long sequences |

### Full table

| Config | err_ratio | FlashKDA (ms) | cuLA (ms) | Speedup |
|---|--:|--:|--:|--:|
| uniform  5seqs T=4096  [819..820]  avg=819  | 0.005536 | 0.4923 | 0.6712 | 0.73x |
| random   5seqs T=4096  [118..2097] avg=819  | 0.005526 | 0.5584 | 0.6443 | 0.87x |
| skewed   5seqs T=4096  [512..2048] avg=819  | 0.005528 | 0.5847 | 0.6032 | 0.97x |
| uniform 10seqs T=4096  [409..415]  avg=409  | 0.005516 | 0.4700 | 0.7074 | 0.66x |
| random  10seqs T=4096  [24..1201]  avg=409  | 0.005520 | 0.5123 | 0.6608 | 0.78x |
| skewed  10seqs T=4096  [227..2053] avg=409  | 0.005532 | 0.6062 | 0.6684 | 0.91x |
| uniform 20seqs T=4096  [204..220]  avg=204  | 0.005520 | 0.4878 | 0.9325 | 0.52x |
| random  20seqs T=4096  [5..787]    avg=204  | 0.005526 | 0.5173 | 0.7383 | 0.70x |
| skewed  20seqs T=4096  [107..2063] avg=204  | 0.005522 | 0.6638 | 0.7254 | 0.92x |
| uniform  5seqs T=8192  [1638..1640] avg=1638 | 0.005532 | 0.9390 | 1.2360 | 0.76x |
| random   5seqs T=8192  [236..4195]  avg=1638 | 0.005527 | 1.0748 | 1.2106 | 0.89x |
| skewed   5seqs T=8192  [1024..4096] avg=1638 | 0.005538 | 1.1268 | 1.1426 | 0.99x |
| uniform 10seqs T=8192  [819..821]   avg=819  | 0.005532 | 0.8594 | 1.1682 | 0.74x |
| random  10seqs T=8192  [48..2401]   avg=819  | 0.005521 | 0.9528 | 1.2096 | 0.79x |
| skewed  10seqs T=8192  [455..4097]  avg=819  | 0.005534 | 1.1487 | 1.2049 | 0.95x |
| uniform 20seqs T=8192  [409..421]   avg=409  | 0.005524 | 0.8548 | 1.4048 | 0.61x |
| random  20seqs T=8192  [9..1574]    avg=409  | 0.005525 | 0.9394 | 1.2771 | 0.74x |
| skewed  20seqs T=8192  [215..4107]  avg=409  | 0.005536 | 1.2086 | 1.2894 | 0.94x |
| uniform  5seqs T=16384 [3276..3280] avg=3276 | 0.005508 | 1.8146 | 2.3604 | 0.77x |
| random   5seqs T=16384 [473..8389]  avg=3276 | 0.005506 | 2.1023 | 2.3455 | 0.90x |
| skewed   5seqs T=16384 [2048..8192] avg=3276 | 0.005509 | 2.2003 | 2.2787 | 0.97x |
| uniform 10seqs T=16384 [1638..1642] avg=1638 | 0.005508 | 1.6583 | 2.1537 | 0.77x |
| random  10seqs T=16384 [95..4802]   avg=1638 | 0.005503 | 1.8520 | 2.2720 | 0.82x |
| skewed  10seqs T=16384 [910..8194]  avg=1638 | 0.005503 | 2.2368 | 2.3132 | 0.97x |
| uniform 20seqs T=16384 [819..823]   avg=819  | 0.005505 | 1.5901 | 2.3406 | 0.68x |
| random  20seqs T=16384 [19..3147]   avg=819  | 0.005500 | 1.7835 | 2.2928 | 0.78x |
| skewed  20seqs T=16384 [431..8195]  avg=819  | 0.005502 | 2.3203 | 2.3583 | 0.98x |
| uniform  5seqs T=32768 [6553..6556] avg=6553 | 0.005523 | 3.5626 | 4.6100 | 0.77x |
| random   5seqs T=32768 [945..16778] avg=6553 | 0.005526 | 4.1320 | 4.6855 | 0.88x |
| skewed   5seqs T=32768 [4096..16384] avg=6553| 0.005523 | 4.3760 | 4.5208 | 0.97x |
| uniform 10seqs T=32768 [3276..3284] avg=3276 | 0.005525 | 3.2643 | 4.1316 | 0.79x |
| random  10seqs T=32768 [191..9605]  avg=3276 | 0.005526 | 3.6217 | 4.4078 | 0.82x |
| skewed  10seqs T=32768 [1820..16388] avg=3276| 0.005524 | 4.4230 | 4.5072 | 0.98x |
| uniform 20seqs T=32768 [1638..1646] avg=1638 | 0.005525 | 3.1160 | 4.3034 | 0.72x |
| random  20seqs T=32768 [37..6294]   avg=1638 | 0.005525 | 3.4888 | 4.3376 | 0.80x |
| skewed  20seqs T=32768 [862..16390] avg=1638 | 0.005525 | 4.5536 | 4.5735 | **1.00x** |

---

## Key Observations

### Performance

1. **FlashKDA is consistently faster than cuLA on this H200 GPU.** The speedup ranges from 1.3x–2.0x on fixed-length and uniform-distribution varlen workloads.

2. **cuLA narrows the gap on skewed distributions** (a few very long sequences dominating total length). At T=32768 with skewed dist and 20 sequences, cuLA reaches parity (1.00x). This is consistent with cuLA being optimized for large chunk-level parallelism within a single long sequence.

3. **cuLA is disproportionately slower with many short sequences.** E.g., uniform 20seqs T=4096 (avg_len=204): 0.52x. This is the worst case for cuLA's tiling strategy.

4. **Fixed-length performance plateaus at ~0.75x** for B=2 once T is large enough to saturate tile-level parallelism.

### Accuracy

- err_ratio between cuLA and FlashKDA outputs: **~0.0055** (uniform across all configs, independent of T, B, distribution)
- This is consistent with the findings in [ACCURACY_CULA_VS_FLASHKDA.md](ACCURACY_CULA_VS_FLASHKDA.md): cuLA err_ratio vs FLA ≈ 0.0031, FlashKDA err_ratio vs FLA ≈ 0.0048 → combined divergence ≈ 0.005–0.006.
