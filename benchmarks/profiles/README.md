# THD selection profile examples

These are explicit profiles from real-CP experiments, not library
defaults or physical-link bandwidth specifications. They apply to indexer
H64/D128, Top-K512, BF16 query/weights and int32 IDs/FP32 scores, at 1M total
tokens with eager PyTorch/NCCL variable-split all-to-all.

- `thd_selection_h20_cp16.json`: two nodes, eight H20-3e ranks per node (2026-11-15).
- `thd_selection_b300_cp8.json`: one node, eight B300 ranks. Its transport fit
  was refreshed on 2026-11-17 with diagonal copies and local-retention sweeps.
  The inter-node
  rate is unused; it is not a cross-node Blackwell measurement.

The profiles retain compute-fit provenance and single-peer/dense/striped
transport samples. `min_gain_ms` is a conservative **single-use preparation
budget**, 40 ms and 20 ms respectively, not a measured kernel cost. A caller
reusing a plan can explicitly amortize it; zero keeps throughput-only planning.
No plan is selected by a case name, document-count lookup, or hardware detector.

The refreshed B300 fit uses 20 independent transport fixtures and holds out
15 fixtures, including every half-local route and every 32768-row route.
[Raw samples](../../docs/results/csa_cp_transport_local_20261117.json) include
all ranks and exact row-marker checks. Mean absolute relative prediction error
is 12.69% on the held-out transport fixtures, and 3.97% on the four held-out
65536/131072-row fixtures. Smaller messages show protocol/utilization effects
that this simple model does not capture. Validate the complete operator at the
intended context length; these coefficients are not an accuracy guarantee.

Pass one explicitly to `bench_csa_layout_matrix.py --planner-cost-model` with
`--planner-strategy selection`. Recalibrate with `calibrate_thd_selection.py`
and the independent `bench_thd_transport.py` / `fit_thd_transport.py` workflow for other indexer shapes, process-group topology,
message sizes or NCCL/software configuration. See
[the experiment report](../../docs/THD_SELECTION_RESULTS.md) for measured
latency, precision, memory and limitations.
