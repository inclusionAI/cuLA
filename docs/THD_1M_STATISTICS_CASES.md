# Statistics-inspired 1M THD workload catalog

The motivating histogram describes **document counts by character length**. Its largest visible bucket is `(10,100]`; exact percentages,
the full long tail and tokenizer information were not supplied. This catalog
uses explicit synthetic priors to explore that shape. It is not a reconstruction
of the real token distribution or a claim about hidden chart values.

There are **18 new cases**, in addition to the previous five baseline cases.
Every case contains exactly **1,048,576 tokens**, with positive document lengths
divisible by four for the current complete-CSA CP harness. All source CP16
shards consequently have 65,536 tokens. The generator uses only Python's
standard library and does not initialize CUDA or distributed execution.

## Distribution assumptions

Weights below are **sample-count priors**, not token shares or measurements
read from the chart. The primary `mix80` prior approximates the chart's visible
shape; `tiny90` and `mix70` test sensitivity to its dominant short bucket.
The generator samples uniformly over aligned lengths within each bucket.

| Synthetic prior | (10,100] | (100,1K] | (1K,2K] | (2K,4K] | (4K,8K] | (8K,16K] |
|---|---:|---:|---:|---:|---:|---:|
| tiny90 | 90% | 5% | 2% | 2% | 0.8% | 0.2% |
| mix80 | 80% | 7% | 5% | 5% | 2.5% | 0.5% |
| mix70 | 70% | 12% | 8% | 6% | 3% | 1% |

K=1024 is a synthetic convention. Default bucket endpoints act as token-length
proxies. Two additional cases divide those endpoints by 2 and 4 before drawing
aligned token lengths; these are character/token **sensitivity assumptions**,
not measured conversions. Actual tokenization is required to calibrate them.
For these scaled cases, the reported `<=100 token` fraction is not the same
quantity as the source chart's `(10,100] character` fraction.

Sampling stops at the desired mixture token budget. Only the final draw may
be shortened to fill exactly; the manifest records the number of tokens removed.
It never fills a residual with an invented giant document. Explicit long stress
documents are then added without truncation. Fixed seed `20261112` and a
separate deterministic permutation make the complete ordered length list
reproducible. Different seeds provide additional replicas.

## Twelve primary cases

All numbers below are the **generated default-seed values**, not expected
probabilities. Prefix every short name with `1m-stat-` to obtain the benchmark
case id. Randomized cases use a reproducible shuffled order.

| Case | Documents | <=100-token document share | <=100-token token share | Largest document | Explicit long-token share |
|---|---:|---:|---:|---:|---:|
| tiny64 | 16,384 | 100.00% | 100.00% | 64 | 0% |
| tiny90 | 4,100 | 89.68% | 19.75% | 14,148 | 0% |
| mix80 | 1,857 | 79.43% | 7.88% | 15,064 | 0% |
| mix70 | 1,403 | 69.57% | 5.27% | 15,932 | 0% |
| mix80-div2 | 3,860 | 80.00% | 8.37% | 7,532 | 0% |
| mix80-div4 | 7,783 | 82.04% | 9.53% | 4,052 | 0% |
| mix80-1x128k | 1,639 | 79.44% | 6.94% | 131,072 | 12.5% |
| mix80-1x256k | 1,450 | 79.03% | 6.10% | 262,144 | 25% |
| mix80-1x512k | 1,066 | 79.55% | 4.60% | 524,288 | 50% |
| mix80-2x256k | 1,067 | 79.48% | 4.60% | 262,144 | 50% |
| mix80-4x128k | 1,069 | 79.33% | 4.60% | 131,072 | 50% |
| mix80-8x64k | 1,073 | 79.03% | 4.60% | 65,536 | 50% |

The distinction between document share and token share is material: `mix80`
contains nearly 80% tiny documents, but those documents contribute only 7.88%
of tokens. In `mix80-2x256k`, just two of 1,067 documents contribute 50% of all
tokens. These cases expose both many-document overhead and long-prefix causal
work, which the earlier 22-document mixed case could not represent together.

The four 50%-long-token cases share **the identical 524,288-token mixture
background**, while splitting the remaining 524,288 tokens as one, two, four
or eight long documents. This controls total long-token volume and tests
whether the planner handles changing document count and causal lengths.
The 64K–512K spikes are deliberate stress additions; their frequency is not
inferred from the chart.

`tiny64` stresses many short segments, boundaries, padding and scheduling.
`mix70`/`mix80`/`tiny90` vary document-count concentration. `div2`/`div4` increase
document density at a fixed token budget. Long-spike cases stress causal load,
communication and scratch memory without making document-count histograms
look predominantly long.

## Six order-only controls

For **both** `mix80` and `mix80-2x256k`, add these suffixes:

| Suffix | Ordering | Purpose |
|---|---|---|
| `-ascending` | shortest to longest | Cluster longer documents at the end of packed input |
| `-descending` | longest to shortest | Move long-document ownership toward earlier source ranks |
| `-alternating` | shortest, longest, second shortest, second longest, ... | Interleave the same histogram |

Each variant retains exactly the same length multiset and total as its
shuffled counterpart. Thus performance differences can be associated with
packing order/rank ownership rather than a different length histogram. The
control fixes lengths, not document-content tensors across order variants:
the current GPU harness still creates synthetic operands from rank-local RNG.

## Reproducible suites

The generator `benchmarks/thd_1m_cases.py` writes exact lengths, seeds,
assumptions, SHA256 digests and count/token histograms. The matrix runner
`benchmarks/bench_csa_layout_matrix.py` accepts `--list-cases` without CUDA.

- `baseline`: five single, uniform, near-uniform and long-tail cases.
- `counted`: eleven 256--1024-document cases, including order-only controls.
- `heldout`: five lognormal, jittered, tiny-tail and trimodal cases for a supplied seed.
- `statistics`: the eighteen synthetic histogram cases described above.
- `validation`: the five baseline cases, eleven counted cases at seed 20261112,
  and five held-out cases at each of seeds 20261114, 20261115 and 20261116.
  This fixes the exact **31-case** matrix in the [measured report](THD_SELECTION_RESULTS.md).

The statistics suite is an additional stress catalog; it is not part of that
31-case GPU result. All suites use synthetic tensor operands, not training data.
A planner does not receive case names or histogram priors: only CPU sequence
boundaries, its configuration and explicit deployment coefficients.
