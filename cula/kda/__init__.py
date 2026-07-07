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

"""Public KDA API exports for chunk, prefill, and decode"""

__all__ = [
    "chunk_kda",
    "kda_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "kda_prefill_hopper",
    "kda_prefill_hopper_opt",
    "kda_prefill_hopper_auto",
]

_LAZY = {
    "chunk_kda": ("cula.kda.chunk", "chunk_kda"),
    "kda_prefill_hopper": ("cula.kda.hopper_fused_fwd", "cula_kda_prefill"),
    "kda_decode": ("cula.ops.kda.decode.cute", "kda_decode"),
    "fused_sigmoid_gating_delta_rule_update": (
        "cula.ops.kda.decode.cute",
        "fused_sigmoid_gating_delta_rule_update",
    ),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__():
    return sorted(__all__)
