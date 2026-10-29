# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional KDA backend adapters."""

__all__ = ["cudnn_kda_summary_thd", "cudnn_kda_thd"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from cula.kda.backends import cudnn

    return getattr(cudnn, name)


def __dir__():
    return sorted(__all__)
