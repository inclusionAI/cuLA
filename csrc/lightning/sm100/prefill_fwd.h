#pragma once

#include <ATen/Tensor.h>

void FLACutlassSM100FwdRun(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    at::Tensor ht,
    at::Tensor g_gamma,
    float scale,
    at::Tensor initial_state,
    bool output_final_state,
    at::Tensor cu_seqlens);
