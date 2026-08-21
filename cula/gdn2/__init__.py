# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public Gated DeltaNet-2 operators."""

from .prefill import (
    chunk_gdn2,
    get_sm90_gdn2_backend,
    get_sm90_gdn2_backend_identity,
    is_sm90_gdn2_available,
)

__all__ = [
    "chunk_gdn2",
    "get_sm90_gdn2_backend",
    "get_sm90_gdn2_backend_identity",
    "is_sm90_gdn2_available",
]
