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

"""Unified interface to per-architecture CUDA extensions.

Downstream code can continue to use ``import cula.cudac as cula_cuda``
and call ``cula_cuda.kda_fwd_prefill(...)`` or
``cula_cuda.chunk_kda_fwd_intra_cuda(...)`` without knowing which
extension provides the function.
"""

import importlib
import sys
import threading
from types import ModuleType


class _CudacProxy(ModuleType):
    """Lazy proxy that exposes functions from all built arch extensions."""

    def __init__(self):
        super().__init__(__name__)
        self.__path__ = []
        self._modules_loaded = False
        self._funcs: dict[str, object] = {}
        self._lock = threading.Lock()

    def _load(self):
        if self._modules_loaded:
            return
        with self._lock:
            if self._modules_loaded:
                return
            loaded_any = False
            errors: dict[str, Exception] = {}
            for ext_name in ("cula._cudac_sm100", "cula._cudac_sm90"):
                try:
                    mod = importlib.import_module(ext_name)
                    for attr in dir(mod):
                        if not attr.startswith("_"):
                            self._funcs[attr] = getattr(mod, attr)
                    loaded_any = True
                except ImportError as exc:
                    errors[ext_name] = exc
            if not loaded_any:
                details = "; ".join(f"{name}: {exc}" for name, exc in errors.items())
                raise ImportError(
                    "None of the cuLA CUDA extensions could be imported. "
                    f"Per-extension errors: [{details}]. "
                    "Please make sure cuLA is compiled correctly."
                )
            self.__dict__.update(self._funcs)
            self._modules_loaded = True

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        self._load()
        try:
            return self._funcs[name]
        except KeyError:
            raise AttributeError(f"module 'cula.cudac' has no attribute '{name}'") from None

    def __dir__(self):
        self._load()
        return list(self._funcs.keys())


_proxy = _CudacProxy()
_proxy.__dict__.update({k: globals().get(k) for k in ("__spec__", "__file__", "__package__", "__loader__")})
sys.modules[__name__] = _proxy
