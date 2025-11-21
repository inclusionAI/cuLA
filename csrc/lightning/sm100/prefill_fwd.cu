#include "lightning/sm100/prefill_fwd.h"

#include <cutlass/kernel_hardware_info.h>

void FLACutlassSM100FwdRun(at::Tensor q, at::Tensor k, at::Tensor v,
                           at::Tensor o, at::Tensor ht, at::Tensor g_gamma,
                           float scale, at::Tensor initial_state,
                           bool output_final_state, at::Tensor cu_seqlens) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
}
