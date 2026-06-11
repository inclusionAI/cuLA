# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Blackwell fused prefill (CuTe DSL — may fail if tcgen05 not available)
try:
    from cula.kda.blackwell_fused_fwd import flash_kda_prefill as kda_prefill_blackwell
except (ImportError, ModuleNotFoundError):
    kda_prefill_blackwell = None

# Chunk KDA dispatcher (may fail if cula.cudac not built)
try:
    from cula.kda.chunk import chunk_kda
except (ImportError, ModuleNotFoundError):
    chunk_kda = None

# Ampere fused prefill (FLA Triton, always importable — no C++ deps)
from cula.kda.ampere_fused_fwd import cula_kda_prefill_ampere as kda_prefill_ampere

# Decode kernels (always importable)
from cula.ops.kda_decode import fused_sigmoid_gating_delta_rule_update, kda_decode

# Hopper fused prefill (C++ extension — may not be available on non-Hopper builds)
try:
    from cula.kda.hopper_fused_fwd import cula_kda_prefill as kda_prefill_hopper
except (ImportError, ModuleNotFoundError):
    kda_prefill_hopper = None

__all__ = [
    "chunk_kda",
    "kda_prefill_blackwell",
    "kda_prefill_hopper",
    "kda_prefill_ampere",
    "kda_decode",
    "fused_sigmoid_gating_delta_rule_update",
]
