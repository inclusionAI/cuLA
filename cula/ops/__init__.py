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

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "kda_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "linear_attention_decode",
]

_LAZY_EXPORTS = {
    "kda_decode": ("cula.ops.kda_decode", "kda_decode"),
    "fused_sigmoid_gating_delta_rule_update": (
        "cula.ops.kda_decode",
        "fused_sigmoid_gating_delta_rule_update",
    ),
    "linear_attention_decode": ("cula.ops.la_decode", "linear_attention_decode"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
