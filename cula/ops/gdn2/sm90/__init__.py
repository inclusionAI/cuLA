# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Private Hopper SM90a GDN2 backend."""

from .config import SM90_BACKEND_ID

__all__ = ["get_sm90_gdn2_backend_identity"]


def get_sm90_gdn2_backend_identity() -> str:
    """Return the stable product backend identity."""

    return SM90_BACKEND_ID
