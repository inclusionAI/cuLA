#pragma once

struct LightningFwdKernelTMAWarpSpecializedSchedule {
  enum class WarpRole { Load, Epilogue, MMA, Exponential, Empty }
};