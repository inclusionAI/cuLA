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

Loading is **once per process**: the first attribute access checks the
currently active CUDA device, imports the matching ``cula._cudac_sm*``
extension, and caches the discovered callables on the module instance.
Changing the active CUDA device to a different architecture after a
process has already loaded ``cula.cudac`` will therefore not be picked
up -- callers that need a different extension must restart Python.
"""

import importlib
import sys
import threading
from types import ModuleType


def _current_device_extension() -> tuple[str, str]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("cuLA CUDA extensions require PyTorch to detect the current GPU.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("cuLA CUDA extensions require a visible CUDA GPU, but torch.cuda.is_available() is False.")

    device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)
    sm_label = f"sm_{prop.major}{prop.minor}"
    if prop.major == 10 and prop.minor in (0, 3):
        return "cula._cudac_sm100", sm_label
    if prop.major == 9 and prop.minor == 0:
        return "cula._cudac_sm90", sm_label
    raise RuntimeError(f"Unsupported CUDA compute capability {sm_label}. Supported architectures: sm_100, sm_103, sm_90.")


class _CudacProxy(ModuleType):
    """Lazy proxy that exposes functions from the current GPU arch extension."""

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
            ext_name, sm_label = _current_device_extension()
            try:
                mod = importlib.import_module(ext_name)
                for attr in dir(mod):
                    if not attr.startswith("_"):
                        self._funcs[attr] = getattr(mod, attr)
            except (ImportError, AttributeError, OSError) as exc:
                raise ImportError(
                    f"The cuLA CUDA extension for the current GPU ({sm_label}) could not be imported. "
                    f"Extension {ext_name} failed with: {exc}. "
                    "Please make sure cuLA is compiled correctly."
                ) from exc
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
