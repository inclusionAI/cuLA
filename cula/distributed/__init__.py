# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Distributed linear-attention wrappers."""

from cula.distributed.kda import cudnn_kda_cp_thd

__all__ = ["cudnn_kda_cp_thd"]
