# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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
            f"cula.kda requires flash-linear-attention. "
            f"Install with: pip install cuda-linear-attention[fla]"
        ) from None

    try:
        return getattr(mod, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Module {mod_path} does not define {attr_name}") from e


__all__ = list(_API_MAP.keys())
