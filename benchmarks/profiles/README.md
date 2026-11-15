# THD selection profile examples

These are explicit profiles from the 2026-11-15 real-CP experiment, not library
defaults or physical-link bandwidth specifications. They apply to indexer
H64/D128, Top-K512, BF16 query/weights and int32 IDs/FP32 scores, at 1M total
tokens with eager PyTorch/NCCL variable-split all-to-all.

- `thd_selection_h20_cp16.json`: two nodes, eight H20-3e ranks per node.
- `thd_selection_b300_cp8.json`: one node, eight B300 ranks. The inter-node
  rate is unused; it is not a cross-node Blackwell measurement.

The profiles retain compute-fit provenance and single-peer/dense/striped
transport samples. `min_gain_ms` is a conservative **single-use preparation
budget**, 40 ms and 20 ms respectively, not a measured kernel cost. A caller
reusing a plan can explicitly amortize it; zero keeps throughput-only planning.
No plan is selected by a case name, document-count lookup, or hardware detector.

Pass one explicitly to `bench_csa_layout_matrix.py --planner-cost-model` with
`--planner-strategy selection`. Recalibrate with `calibrate_thd_selection.py`
and the independent `bench_thd_transport.py` / `fit_thd_transport.py` workflow for other indexer shapes, process-group topology,
message sizes or NCCL/software configuration. See
[the experiment report](../../docs/THD_SELECTION_RESULTS.md) for measured
latency, precision, memory and limitations.
