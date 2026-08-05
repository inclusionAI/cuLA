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

import pytest

from cula.ops._cutedsl_compat import Tcgen05LdStApi, detect_tcgen05_ldst_api


def _legacy_ld(res, shape, num, tmem_addr, *, pack=None, half_split_offset=None):
    pass


def _legacy_st(shape, num, tmem_addr, r, *, unpack=None, half_split_offset=None):
    pass


def _inferred_ld(res, shape, tmem_addr, *, pack=None, offset=None):
    pass


def _inferred_st(shape, tmem_addr, val, *, unpack=None, offset=None):
    pass


def test_detect_tcgen05_ldst_legacy_api():
    assert detect_tcgen05_ldst_api(_legacy_ld, _legacy_st) == Tcgen05LdStApi(
        ld_has_num=True,
        st_has_num=True,
        st_value_keyword="r",
    )


def test_detect_tcgen05_ldst_inferred_api():
    assert detect_tcgen05_ldst_api(_inferred_ld, _inferred_st) == Tcgen05LdStApi(
        ld_has_num=False,
        st_has_num=False,
        st_value_keyword="val",
    )


def test_detect_tcgen05_ldst_mixed_api():
    assert detect_tcgen05_ldst_api(_legacy_ld, _inferred_st) == Tcgen05LdStApi(
        ld_has_num=True,
        st_has_num=False,
        st_value_keyword="val",
    )


def test_detect_tcgen05_ldst_rejects_unknown_store_value_keyword():
    def unsupported_st(shape, tmem_addr, value):
        pass

    with pytest.raises(RuntimeError, match="expected exactly one value keyword"):
        detect_tcgen05_ldst_api(_inferred_ld, unsupported_st)
