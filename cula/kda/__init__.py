# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib

_PUBLIC_API = {
    "chunk_kda": "cula.kda.chunk",
    "kda_prefill_hopper": "cula.kda.hopper_fused_fwd",
    "kda_decode": "cula.ops.kda_decode",
    "fused_sigmoid_gating_delta_rule_update": "cula.ops.kda_decode",
}

# Reverse map: module -> list of (attr, public_name)
_MODULE_ATTRS = {
    "cula.kda.chunk": [("chunk_kda", "chunk_kda")],
    "cula.kda.hopper_fused_fwd": [("cula_kda_prefill", "kda_prefill_hopper")],
    "cula.ops.kda_decode": [
        ("kda_decode", "kda_decode"),
        ("fused_sigmoid_gating_delta_rule_update", "fused_sigmoid_gating_delta_rule_update"),
    ],
}


def __getattr__(name: str):
    """Lazy-load kda submodules only when accessed."""
    if name in _PUBLIC_API:
        module_path = _PUBLIC_API[name]
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            raise ImportError(
                f"cula.kda requires flash-linear-attention. "
                f"Install with: pip install cuda-linear-attention[fla]"
            ) from None
        # Get the actual attribute name in the module
        for mod_path, attrs in _MODULE_ATTRS.items():
            if mod_path == module_path:
                for attr_name, public_name in attrs:
                    if public_name == name:
                        return getattr(mod, attr_name)
        return getattr(mod, name)
    raise AttributeError(f"module 'cula.kda' has no attribute {name!r}")


__all__ = list(_PUBLIC_API.keys())
