# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SM90 intracard context-parallel prefill backend."""

__all__ = ["intracard_prefill"]


def __getattr__(name):
    if name == "intracard_prefill":
        from cula.ops.kda.sm90.cp.driver import intracard_prefill

        return intracard_prefill
    raise AttributeError(name)
