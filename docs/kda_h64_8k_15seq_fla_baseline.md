# H64 8K balanced-varlen KDA comparison with FLA as accuracy baseline

## Configuration

- NVIDIA GB200; PyTorch 2.11.0+cu130
- `H=HV=64`, `K=V=128`, BF16 activations and FP32 gate parameters/state
- 15 packed sequences with lengths
  `[320, 384, 416, 448, 480, 512, 528, 544, 560, 576, 608, 640, 672, 720, 784]`
- Total tokens: 8192; maximum/minimum length ratio: 2.45
- `safe_gate=True`, `lower_bound=-5`, chunk size 64
- Output and final-state gradients are both included
- cuLA: CuTeDSL WY/dKQG plus tcgen05 CuTeDSL intra, saved forward intermediates
- FLA: Triton backend with saved forward intermediates
- Timing: four warmups, 20 iterations per round, seven rounds, median

## Performance

| Phase | CAKE | cuLA | FLA | CAKE / FLA | cuLA / FLA |
|---|---:|---:|---:|---:|---:|
| Forward | 0.421 ms | 1.123 ms | 1.516 ms | 3.604x | 1.351x |
| Backward | 1.426 ms | 2.411 ms | 3.471 ms | 2.434x | 1.440x |
| Forward + backward | 1.827 ms | 3.520 ms | 4.987 ms | 2.730x | 1.417x |

## Accuracy relative to FLA

The close percentage uses `atol=1e-2` and `rtol=1e-2`.

| Tensor | CAKE max abs | CAKE rel RMS | CAKE close | cuLA max abs | cuLA rel RMS | cuLA close |
|---|---:|---:|---:|---:|---:|---:|
| output | 2.402e-2 | 1.640e-1 | 99.9990% | 4.883e-4 | 3.850e-5 | 100% |
| final state | 1.928e-2 | 6.644e-3 | 99.9999% | 8.270e-7 | 1.062e-8 | 100% |
| dq | 6.104e-4 | 3.450e-3 | 100% | 4.883e-4 | 2.792e-3 | 100% |
| dk | 3.265e-3 | 4.118e-3 | 100% | 1.953e-3 | 2.791e-3 | 100% |
| dv | 1.953e-3 | 4.072e-3 | 100% | 2.441e-4 | 3.981e-5 | 100% |
| dg | 4.927e-3 | 6.805e-2 | 100% | 6.104e-5 | 1.088e-3 | 100% |
| dbeta | 3.906e-3 | 5.325e-3 | 100% | 4.883e-4 | 9.029e-5 | 100% |
| dA_log | 7.802e-2 | 6.812e-2 | 39.0625% | 4.058e-3 | 2.243e-3 | 100% |
| ddt_bias | 5.225e-2 | 1.809e-1 | 94.4092% | 1.953e-3 | 5.424e-3 | 100% |
| dh0 | 1.872e-4 | 3.285e-3 | 100% | 4.012e-5 | 1.461e-4 | 100% |

cuLA remains tightly aligned with the FLA baseline for every returned tensor.
CAKE is close for the BF16 token/state gradients, but its FP32 `dA_log` and
`ddt_bias` reductions are not numerically interchangeable with FLA at a strict
1e-2 tolerance. The relatively large output relative-RMS value is amplified by
near-zero FLA reference values; the maximum absolute error and close percentage
are more informative for that tensor.
