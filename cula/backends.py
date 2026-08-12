# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import ClassVar

logger = logging.getLogger("cula.backends")


class BaseBackend:
    backend_type: ClassVar[str] = "base"
    env_var: ClassVar[str | None] = None
    default_enable: ClassVar[bool] = True
    priority: ClassVar[int] = 5

    _available: tuple[bool, str | None] | None = None

    def probe(self) -> tuple[bool, str | None]:
        return True, None

    def is_available(self) -> tuple[bool, str | None]:
        if self._available is None:
            self._available = self.probe()
        return self._available

    def is_enabled(self) -> bool:
        if self.env_var is None:
            return True
        default = "1" if self.default_enable else "0"
        return os.environ.get(self.env_var, default) != "0"

    def verify(self, func_name: str, *args, **kwargs) -> tuple[bool, str | None]:
        verifier = getattr(self, f"{func_name}_verifier", None)
        if verifier is None:
            return True, None
        return verifier(*args, **kwargs)


class BackendRegistry:
    def __init__(self, operation: str):
        self.operation = operation
        self._backends: list[BaseBackend] = []
        self._logged: set[str] = set()

    def register(self, backend: BaseBackend) -> None:
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.priority)

    def backends(self) -> list[BaseBackend]:
        return list(self._backends)

    def _log_once(self, key: str, msg: str) -> None:
        if key not in self._logged:
            self._logged.add(key)
            logger.info(msg)

    def dispatch(self, func_name: str, *args, **kwargs):
        rejections: list[tuple[str, str]] = []
        for be in self._backends:
            name = be.backend_type
            if not be.is_enabled():
                rejections.append((name, f"disabled via {be.env_var}=0"))
                continue
            ok, why = be.is_available()
            if not ok:
                rejections.append((name, why or "unavailable"))
                continue
            ok, why = be.verify(func_name, *args, **kwargs)
            if not ok:
                rejections.append((name, why or "rejected"))
                self._log_once(
                    f"{self.operation}:{func_name}:{name}:reject",
                    f"[cula backend] {self.operation}.{func_name}: {name} rejected: {why}",
                )
                continue
            self._log_once(
                f"{self.operation}:{func_name}:{name}",
                f"[cula backend] {self.operation}.{func_name} -> {name}",
            )
            return getattr(be, func_name)(*args, **kwargs)

        table = "\n".join(f"  {n:<12s}: {r}" for n, r in rejections) or "  (no backends registered)"
        raise RuntimeError(f"no {self.operation} backend accepted this call:\n{table}")
