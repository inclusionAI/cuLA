"""
PyTorch entry point for KDA **prefill** (not decode).

Build the CUDA extension once from the repo root (same env as PyTorch), for example:

    /path/to/envs/kda_fla/bin/python setup.py build_ext --inplace

This imports the compiled module `kda_prefill_cuda`, which registers
`torch.ops.kda_prefill.forward` (CuTe-style layout: ``[B,H,T,K]`` tensors).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_extension_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_extension_path()

import torch

try:
    import kda_prefill_cuda  # noqa: F401 — side effect: registers torch.ops
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kda_prefill_cuda is not built. From the repository root run:\n"
        "  python setup.py build_ext --inplace\n"
        "using the same Python as `import torch`."
    ) from exc


def kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused KDA prefill (float32), matching this repo's C++/CUDA contract.

    Tensor layout (same as cuLA-style KDA prefill):

    - ``q``, ``k``, ``g``: ``(B, H, T, K)``
    - ``v``: ``(B, H, T, V)``
    - ``beta``: ``(B, H, T)``
    - Optional chunk buffers ``w`` / ``u`` (else allocated): ``(B, H, Nc, C, K|V)``
    - Optional output ``o`` (else allocated): ``(B, H, T, V)``

    Supported ``(K, V, chunk_size)``: ``(64, 64, 32|64)``, ``(128, 128, 32)``.

    Returns:
        ``(o, w, u)`` — attention output and intra-chunk intermediates.
    """
    return torch.ops.kda_prefill.forward(
        q, k, v, g, beta, chunk_size, w, u, o
    )


def _rand_inputs(
    device: str,
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(B, H, T, K, device=device, dtype=torch.float32, generator=g)
    k = torch.randn(B, H, T, K, device=device, dtype=torch.float32, generator=g)
    v = torch.randn(B, H, T, V, device=device, dtype=torch.float32, generator=g)
    gv = torch.randn(B, H, T, K, device=device, dtype=torch.float32, generator=g)
    beta = torch.randn(B, H, T, device=device, dtype=torch.float32, generator=g)
    return q, k, v, gv, beta


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestKdaPrefillCuda(unittest.TestCase):
    device = "cuda"

    def _assert_shapes(
        self,
        o: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        B: int,
        H: int,
        T: int,
        K: int,
        V: int,
        C: int,
    ) -> None:
        nc = (T + C - 1) // C
        self.assertEqual(o.shape, (B, H, T, V))
        self.assertEqual(w.shape, (B, H, nc, C, K))
        self.assertEqual(u.shape, (B, H, nc, C, V))
        self.assertTrue(torch.isfinite(o).all().item())

    def test_k64_v64_c32(self) -> None:
        B, H, T, K, V, C = 1, 2, 128, 64, 64, 32
        q, k, v, g, beta = _rand_inputs(self.device, B, H, T, K, V, 0)
        o, w, u = kda_prefill(q, k, v, g, beta, C)
        torch.cuda.synchronize()
        self._assert_shapes(o, w, u, B, H, T, K, V, C)

    def test_k64_v64_c64(self) -> None:
        B, H, T, K, V, C = 1, 1, 96, 64, 64, 64
        q, k, v, g, beta = _rand_inputs(self.device, B, H, T, K, V, 1)
        o, w, u = kda_prefill(q, k, v, g, beta, C)
        torch.cuda.synchronize()
        self._assert_shapes(o, w, u, B, H, T, K, V, C)

    def test_k128_v128_c32(self) -> None:
        B, H, T, K, V, C = 1, 1, 64, 128, 128, 32
        q, k, v, g, beta = _rand_inputs(self.device, B, H, T, K, V, 2)
        o, w, u = kda_prefill(q, k, v, g, beta, C)
        torch.cuda.synchronize()
        self._assert_shapes(o, w, u, B, H, T, K, V, C)

    def test_preallocated_buffers(self) -> None:
        B, H, T, K, V, C = 2, 2, 77, 64, 64, 32
        q, k, v, g, beta = _rand_inputs(self.device, B, H, T, K, V, 4)
        nc = (T + C - 1) // C
        w0 = torch.empty(B, H, nc, C, K, device=self.device, dtype=torch.float32)
        u0 = torch.empty(B, H, nc, C, V, device=self.device, dtype=torch.float32)
        o0 = torch.empty(B, H, T, V, device=self.device, dtype=torch.float32)
        o, w, u = kda_prefill(q, k, v, g, beta, C, w=w0, u=u0, o=o0)
        torch.cuda.synchronize()
        self.assertIs(o, o0)
        self.assertIs(w, w0)
        self.assertIs(u, u0)
        self._assert_shapes(o, w, u, B, H, T, K, V, C)

    def test_torch_op_direct(self) -> None:
        B, H, T, K, V, C = 1, 1, 32, 64, 64, 32
        q, k, v, g, beta = _rand_inputs(self.device, B, H, T, K, V, 5)
        o, w, u = torch.ops.kda_prefill.forward(
            q, k, v, g, beta, C, None, None, None
        )
        torch.cuda.synchronize()
        self._assert_shapes(o, w, u, B, H, T, K, V, C)


if __name__ == "__main__":
    unittest.main(verbosity=2)
