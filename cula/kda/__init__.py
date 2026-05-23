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

import importlib

# 公开接口映射到 (模块路径, 真实函数名)
_API_MAP = {
    "chunk_kda": ("cula.kda.chunk", "chunk_kda"),
    "kda_prefill_hopper": ("cula.kda.hopper_fused_fwd", "cula_kda_prefill"),
    "kda_decode": ("cula.ops.kda_decode", "kda_decode"),
    "fused_sigmoid_gating_delta_rule_update": ("cula.ops.kda_decode", "fused_sigmoid_gating_delta_rule_update"),
}


def __getattr__(name: str):
    if name not in _API_MAP:
        raise AttributeError(f"module 'cula.kda' has no attribute {name!r}")

    mod_path, attr_name = _API_MAP[name]

    try:
        mod = importlib.import_module(mod_path)
    except ImportError:
        raise ImportError(
            "cula.kda requires flash-linear-attention. Install with: pip install cuda-linear-attention[fla]"
        ) from None

    try:
        return getattr(mod, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Module {mod_path} does not define {attr_name}") from e


__all__ = list(_API_MAP.keys())
