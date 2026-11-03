# Large logical-CP emulation

This directory is a benchmark-only layer built on cuLA's native CP public
APIs. Production code under `cula/distributed` and `cula/hybrid` never imports
this package.

The emulators preserve real per-rank tensor shapes and execute real local
cuDNN kernels plus real NCCL hops on the available physical GPUs. They model a
logical dependency length; they do not claim to reproduce a 64-GPU topology,
cross-node congestion, or aggregate bandwidth.

## Structure

- `topology.py`: logical-to-physical rank mapping and transport-span plans.
- `transport.py`: benchmark-only implementation of the native KDA boundary
  transport protocol and shared dependent-ring cost emulation.
- `workload.py`: reusable packed-THD workload descriptions.
- `bench_components.py`: isolated DSA and ring microbenchmarks.
- `bench_kda.py`: complete native KDA forward/backward with logical CP cost.
- `bench_hybrid.py`: native KDA plus DSA-indexer serial/overlap benchmark.

`full-ring` emulates `logical_cp - 1` hops. `sample-chain` uses the longest
packed sample's actual CP span. Neither mode changes KDA merge precision or
DSA arithmetic.

Run from the repository root, for example:

```bash
torchrun --standalone --nproc-per-node=4 \
  benchmarks/large_cp_emulation/bench_hybrid.py logical \
  --global-tokens 1048576 --logical-cp 64 --logical-transport sample-chain \
  --thd-20s2l --schedule both
```
