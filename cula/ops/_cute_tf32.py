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

"""Native CuTe warp-level TF32 MMA compatibility helpers.

CuTeDSL 4.6 exposes the SM80-style ``MmaTF32Op`` Python wrapper.  CuTeDSL
4.4.2 already contains the same native ``MmaAtomSM80Type`` compiler support,
but does not expose that small wrapper.  The backport below fills only that
Python API gap.  Both paths lower through CuTe's TiledMma abstraction to
``mma.sync.m16n8k8``; neither path contains inline PTX or uses WGMMA.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
import cutlass.cute as cute
from cutlass.cute.atom import Trait, make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.nvgpu.common import OpError
from cutlass.cute.nvgpu.warp.mma import WarpMmaOp

_MMA_SHAPE_MNK = (16, 8, 8)


class _MmaTF32TraitBackport(Trait):
    """CuTeDSL 4.4.2 trait for the native SM80 TF32 MMA atom."""


@dataclass(frozen=True)
class _MmaTF32OpBackport(WarpMmaOp):
    """Minimal CuTeDSL 4.4.2 equivalent of warp ``MmaTF32Op``."""

    shape_mnk: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.shape_mnk not in ((16, 8, 4), (16, 8, 8)):
            raise OpError(
                self,
                "expects shape_mnk to be (16, 8, 4) or (16, 8, 8)",
            )

    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        del kwargs
        shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
        atom_type = _cute_nvgpu_ir.MmaAtomSM80Type.get(
            shape_mnk.type.attribute,
            cutlass.TFloat32.mlir_type,
            cutlass.TFloat32.mlir_type,
            cutlass.Float32.mlir_type,
        )
        return _MmaTF32TraitBackport(
            make_atom(atom_type, loc=loc, ip=ip)
        )

    def _verify_fragment_A(self, input, *, loc=None, ip=None) -> bool:
        del input, loc, ip
        return True

    def _verify_fragment_B(self, input, *, loc=None, ip=None) -> bool:
        del input, loc, ip
        return True


def make_tf32_tiled_mma() -> cute.TiledMma:
    """Create one-warp native CuTe TF32 ``m16n8k8`` TiledMma.

    Call this while tracing a ``@cute.jit`` function.  CuTeDSL 4.6 and newer
    use the public operation; CuTeDSL 4.4.2 uses the equivalent backported
    Python wrapper over its existing native compiler atom.
    """

    mma_op_type = getattr(cute.nvgpu.warp, "MmaTF32Op", None)
    if mma_op_type is None:
        mma_op = _MmaTF32OpBackport(_MMA_SHAPE_MNK)
    else:
        mma_op = mma_op_type(_MMA_SHAPE_MNK)
    return cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=cute.make_layout((1, 1, 1)),
    )


__all__ = ["make_tf32_tiled_mma"]
