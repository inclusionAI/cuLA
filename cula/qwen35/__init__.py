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

"""Qwen3.5-specific linear attention support built on top of cuLA primitives."""

from cula.qwen35.common import Qwen35LinearAttentionConfig

try:
    from cula.qwen35.runtime import (
        qwen35_linear_attention_decode,
        qwen35_linear_attention_prefill,
    )
except Exception:  # pragma: no cover - optional runtime dependency during partial imports
    qwen35_linear_attention_decode = None
    qwen35_linear_attention_prefill = None

__all__ = ["Qwen35LinearAttentionConfig", "qwen35_linear_attention_prefill", "qwen35_linear_attention_decode"]
