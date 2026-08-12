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

"""Compatibility helpers for low-level CuTeDSL bindings."""

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Literal


@dataclass(frozen=True)
class Tcgen05LdStApi:
    """Detected keyword interface of the generated tcgen05 load/store ops."""

    ld_has_num: bool
    st_has_num: bool
    st_value_keyword: Literal["r", "val"]


def _parameter_names(op: Callable, op_name: str) -> set[str]:
    try:
        parameters = signature(op).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Unable to inspect the CuTeDSL {op_name} binding") from exc

    if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
        raise RuntimeError(f"Unsupported CuTeDSL {op_name} signature: variadic keyword arguments are ambiguous")
    return set(parameters)


def detect_tcgen05_ldst_api(ld_op: Callable, st_op: Callable) -> Tcgen05LdStApi:
    """Detect supported tcgen05 load/store keyword variants from their signatures."""

    ld_parameters = _parameter_names(ld_op, "tcgen05_ld")
    st_parameters = _parameter_names(st_op, "tcgen05_st")

    missing_ld = {"res", "shape", "tmem_addr"} - ld_parameters
    if missing_ld:
        raise RuntimeError(f"Unsupported CuTeDSL tcgen05_ld signature: missing {sorted(missing_ld)}")

    missing_st = {"shape", "tmem_addr"} - st_parameters
    if missing_st:
        raise RuntimeError(f"Unsupported CuTeDSL tcgen05_st signature: missing {sorted(missing_st)}")

    value_keywords = {"r", "val"} & st_parameters
    if len(value_keywords) != 1:
        raise RuntimeError("Unsupported CuTeDSL tcgen05_st signature: expected exactly one value keyword from ['r', 'val']")

    return Tcgen05LdStApi(
        ld_has_num="num" in ld_parameters,
        st_has_num="num" in st_parameters,
        st_value_keyword=value_keywords.pop(),
    )
