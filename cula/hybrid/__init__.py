# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-backed hybrid-attention integration primitives."""

from cula.hybrid.distributed_pool import cudnn_index_pool_cp
from cula.hybrid.dsa import (
    DEFAULT_CANDIDATE_BUDGET_BYTES,
    MAX_BALANCED_CANDIDATE_BUDGET_BYTES,
    MEMORY_EFFICIENT_CANDIDATE_BUDGET_BYTES,
    DsaSelectionPlan,
    DsaSelectionResult,
    DsaSelectionWindow,
    DsaSelectionWorkspace,
    cudnn_dsa_select,
)
from cula.hybrid.index_pool import cudnn_index_pool
from cula.hybrid.plan import IndexPoolPlan, IndexPoolShardPlan

__all__ = [
    "DEFAULT_CANDIDATE_BUDGET_BYTES",
    "MAX_BALANCED_CANDIDATE_BUDGET_BYTES",
    "MEMORY_EFFICIENT_CANDIDATE_BUDGET_BYTES",
    "DsaSelectionPlan",
    "DsaSelectionResult",
    "DsaSelectionWindow",
    "DsaSelectionWorkspace",
    "IndexPoolPlan",
    "IndexPoolShardPlan",
    "cudnn_dsa_select",
    "cudnn_index_pool",
    "cudnn_index_pool_cp",
]
