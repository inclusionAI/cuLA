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

from __future__ import annotations

import ast
from contextlib import suppress
from pathlib import Path

import pytest
import torch
from lightning_attn_reference import chunkwise_lightning_reference, tokenwise_lightning_reference
from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPO_ROOT / "cula/ops/lightning/sm90/prefill_kernel.py"
SCHEDULE_PATH = REPO_ROOT / "cula/ops/lightning/sm90/schedule.py"
WRAPPER_PATH = REPO_ROOT / "cula/ops/lightning/prefill_sm90.py"


def _parse() -> tuple[str, ast.Module]:
    source = BACKEND_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(BACKEND_PATH))


def _parse_path(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Call) and _qualified_name(item.func) == name]


def _literal_assignments(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        with suppress(TypeError, ValueError):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def test_kernel_exposes_only_production_entrypoints() -> None:
    source, tree = _parse()
    compile(source, str(BACKEND_PATH), "exec")
    values = _literal_assignments(tree)
    backend = _class(tree, "LightningSm90PrefillKernel")

    assert [_qualified_name(base) for base in backend.bases] == ["LightningSm90PrefillSchedule"]
    specifier = values["SUPPORTED_CUTLASS_DSL_SPECIFIER"]
    assert specifier == ">=4.4.2,<4.7,!=4.5.0"
    accepted = SpecifierSet(specifier)
    assert all(version in accepted for version in ("4.4.2", "4.5.1", "4.5.2", "4.5.3", "4.6.0", "4.6.1"))
    assert all(version not in accepted for version in ("4.4.1", "4.5.0", "4.7.0"))
    assert values["DECAY_LUT_ENTRIES"] == 65
    assert SCHEDULE_PATH.exists()
    assert not any(isinstance(node, ast.FunctionDef) and node.name.startswith("launch_") for node in tree.body)
    assert "backend_identity" not in source
    assert "import torch" not in source


def test_real_bthd_views_head_batch_grid_and_exact_tma_rings() -> None:
    _, tree = _parse()
    backend = _class(tree, "LightningSm90PrefillKernel")
    call = _method(backend, "__call__")
    text = ast.unparse(call)

    assert "T = sequence_length" in text
    assert "specialized_T = self.sequence_length" in text
    assert "q_layout = cute.make_layout((T, D, (H, B)), stride=(D * H, 1, (D, D * H * T)))" in text
    assert "k_layout = cute.make_layout((D, T, (H, B)), stride=(1, D * H, (D, D * H * T)))" in text
    assert "vo_layout = cute.make_layout((D, T, (HV, B)), stride=(1, D * HV, (D, D * HV * T)))" in text
    assert text.count("cpasync.make_tiled_tma_atom(") == 4
    assert "grid = (1, HV, B)" in text
    assert "if cutlass.const_expr(self.is_varlen):" in text
    assert "grid = (1, HV, self.num_sequences)" in text
    assert "block=(self.threads_per_cta, 1, 1)" in text
    assert "min_blocks_per_mp=1" in text
    assert "assert self.dynamic_smem_bytes == DYNAMIC_SMEM_ESTIMATE_BYTES" in text
    assert "decay_lut: cute.struct.Align" in text


def test_kernel_uses_one_cta_per_head_zero_state_and_kernel_decay_lut() -> None:
    _, tree = _parse()
    backend = _class(tree, "LightningSm90PrefillKernel")
    decay_lut = _method(backend, "populate_decay_lut")
    kernel = _method(backend, "kernel_nonpersistent")
    math = _method(backend, "run_math")
    decay_lut_text = ast.unparse(decay_lut)
    kernel_text = ast.unparse(kernel)
    math_text = ast.unparse(math)

    assert "_, value_head_idx, batch_idx = cute.arch.block_idx()" in kernel_text
    assert "qk_head_idx = value_head_idx // cutlass.Int32(self.group_size)" in kernel_text
    assert kernel_text.count("cute.local_tile(") == 3
    assert kernel_text.count("[None, None, (qk_head_idx, tensor_batch_idx)]") == 2
    assert kernel_text.count("[None, None, (value_head_idx, tensor_batch_idx)]") == 1
    assert "self.populate_decay_lut(tidx, decay_s, decay_head_idx, s_decay_lut)" in kernel_text
    assert "decay_lambda = cute.exp(-decay_s[decay_head_idx], fastmath=False)" in decay_lut_text
    assert "for base in [0, 32, 64]:" in decay_lut_text
    assert "for offset in [1, 2, 4, 8, 16]:" in decay_lut_text
    assert "cute.arch.shuffle_sync_bfly" in decay_lut_text
    assert "product = product * (decay_lambda * decay_lut[cutlass.Int32(base - 1)])" in decay_lut_text
    assert "if cutlass.const_expr(base != 64):" in decay_lut_text
    assert decay_lut_text.count("cute.arch.sync_warp()") == 1
    assert decay_lut_text.index("decay_lut[index] = product") < decay_lut_text.index("cute.arch.sync_warp()")
    assert kernel_text.index("cute.arch.sync_threads()") < kernel_text.index("if warp_group_idx == 0:")
    assert "cute.arch.setmaxregister_decrease(self.register_targets[0])" in kernel_text
    assert "cute.arch.setmaxregister_increase(self.register_targets[1])" in kernel_text
    assert (
        kernel_text.index("if warp_group_idx == 0:")
        < kernel_text.index("cute.arch.setmaxregister_decrease(self.register_targets[0])")
        < kernel_text.index("if warp_idx == 0:")
    )
    assert "state_accumulator[item] = cutlass.Float32(0.0)" in math_text
    assert "if cutlass.const_expr(self.needs_initial_state):" in math_text
    assert "if cutlass.const_expr(self.needs_final_state):" in math_text
    assert "decay_lut[token + 1]" in math_text
    assert "decay_lut[valid_tokens]" in math_text
    assert "valid_tokens - cutlass.Int32(1) - token" in math_text


def test_math_preserves_current_token_recurrence_and_pipeline_lifetimes() -> None:
    _, tree = _parse()
    backend = _class(tree, "LightningSm90PrefillKernel")
    math_text = ast.unparse(_method(backend, "run_math"))
    producer_text = ast.unparse(_method(backend, "run_tma_load_producer"))

    assert "handle = load_producer.acquire_and_advance()" in producer_text
    assert "tma_bar_ptr=handle.barrier" in producer_text
    assert "handle.commit()" in producer_text
    for name in ("q", "k", "v"):
        assert f"{name}_handle = {name}_consumer.wait_and_advance()" in math_text
    assert "qk_published_barrier.sync()" in math_text
    assert "if chunk < num_chunks - cutlass.Int32(1):" in math_text
    assert "if warp_group_idx == MATH0_WARP_GROUP_INDEX:" in math_text
    assert "qk_consumed_barrier.wait_unaligned()" in math_text
    assert "qk_consumed_barrier.arrive_unaligned()" in math_text
    assert "qk_consumed_barrier.sync()" not in math_text
    assert math_text.index(
        "warpgroup.wait_group(0)", math_text.index("self.issue_wgmma_rs_accumulate(o2_rs_mma")
    ) < math_text.index("qk_consumed_barrier.wait_unaligned()")
    assert "self.issue_wgmma_rs_zero(o1_rs_mma" in math_text
    assert "self.issue_wgmma_rs_accumulate(o2_rs_mma" in math_text
    assert "self.issue_wgmma_rs_accumulate(state_rs_mma" in math_text
    assert math_text.index("q_handle.release()") < math_text.index("o2_a =")
    assert math_text.index("o_pipeline.producer_commit(o_state)") < math_text.index(
        "state_accumulator[item] = state_accumulator[item] * decay_lut[valid_tokens]"
    )
    assert math_text.index("k_handle.release()") > math_text.index("self.issue_wgmma_rs_accumulate(state_rs_mma")
    assert math_text.index("v_handle.release()") > math_text.index("self.issue_wgmma_rs_accumulate(state_rs_mma")
    assert "o_publication_fragment[item] = cutlass.BFloat16(o_copy_source[item] * scale)" in math_text
    assert "cute.copy(o_r2s_tiled_copy, o_publication_fragment, o_copy_destination)" in math_text


def test_output_store_is_head_batch_bounded_and_fully_drained() -> None:
    _, tree = _parse()
    backend = _class(tree, "LightningSm90PrefillKernel")
    epilogue_text = ast.unparse(_method(backend, "run_epilogue_store"))

    assert "output_head = o_tma_tensor[None, None, (head_idx, batch_idx)]" in epilogue_text
    assert "cute.domain_offset(" in epilogue_text
    assert "cute.zipped_divide(" in epilogue_text
    assert "cpasync.tma_partition(" in epilogue_text
    assert "cpasync.fence_tma_desc_acquire(tail_gmem_ptr)" in epilogue_text
    assert "tma_desc_ptr=tail_generic_ptr" in epilogue_text
    assert "tma_store_pipeline.producer_tail()" in epilogue_text
    assert epilogue_text.index("tma_store_pipeline.producer_tail()") < epilogue_text.index("remaining_stages = num_chunks")
    assert "PROBE_EPILOGUE" not in epilogue_text


def test_public_wrapper_compiles_exact_sm90a_kernel_without_fallback() -> None:
    source, tree = _parse_path(WRAPPER_PATH)
    compile(source, str(WRAPPER_PATH), "exec")
    compile_text = ast.unparse(_function(tree, "_compile_fixed_variant"))
    wrapper_text = ast.unparse(_function(tree, "lightning_attn_fwd"))

    assert "schedule = LightningSm90PrefillKernel(" in compile_text
    assert "cute.GPUArch(TARGET_ARCH)" in compile_text
    assert "cute.EnableTVMFFI(True)" in compile_text
    assert "make_fake_stream(use_tvm_ffi_env_stream=True)" in compile_text
    assert "compiled(" in wrapper_text
    assert "cutlass.Int32(T)" in wrapper_text
    assert "torch.cuda.synchronize" not in wrapper_text
    assert "try:" not in wrapper_text and "except" not in wrapper_text
    assert "triton" not in source.lower()
    assert SCHEDULE_PATH.exists()


def test_state_layout_gva_mapping_and_compile_time_state_paths() -> None:
    _, tree = _parse()
    backend = _class(tree, "LightningSm90PrefillKernel")
    call_text = ast.unparse(_method(backend, "__call__"))
    kernel_text = ast.unparse(_method(backend, "kernel_nonpersistent"))
    math_text = ast.unparse(_method(backend, "run_math"))
    load_text = ast.unparse(_method(backend, "load_state_fragment"))
    store_text = ast.unparse(_method(backend, "store_state_fragment"))

    assert "HV = self.value_heads" in call_text
    assert "state_batches = self.state_pool_size if self.is_varlen else B" in call_text
    assert "expected_state_elements = state_batches * HV * VALUE_DIM * HEAD_DIM" in call_text
    assert (
        "state_layout = cute.make_layout((HEAD_DIM, VALUE_DIM, (HV, state_batches)), "
        "stride=(1, HEAD_DIM, (HEAD_DIM * VALUE_DIM, HV * HEAD_DIM * VALUE_DIM)))"
    ) in call_text
    assert "grid = (1, HV, B)" in call_text
    assert "qk_head_idx = value_head_idx // cutlass.Int32(self.group_size)" in kernel_text
    assert "if cutlass.const_expr(self.decay_heads == self.qk_heads):" in kernel_text
    assert "initial_state_head = initial_state[None, None, (value_head_idx, state_idx)]" in math_text
    assert "final_state_head = final_state[None, None, (value_head_idx, state_idx)]" in math_text
    assert "_swap_first_two_modes(global_state)" in load_text
    assert "_swap_first_two_modes(global_state)" in store_text
    assert "partition_S" in load_text and "partition_D" in store_text


def test_scale_changes_only_output_publication() -> None:
    _, tree = _parse()
    backend = _class(tree, "LightningSm90PrefillKernel")
    math_text = ast.unparse(_method(backend, "run_math"))

    scale_sites = [
        node
        for node in ast.walk(_method(backend, "run_math"))
        if isinstance(node, ast.Name) and node.id == "scale" and isinstance(node.ctx, ast.Load)
    ]
    assert len(scale_sites) == 1
    assert "cutlass.BFloat16(o_copy_source[item] * scale)" in math_text
    state_tail = math_text[math_text.index("state_accumulator[item] = state_accumulator[item] * decay_lut") :]
    assert "* scale" not in state_tail


def test_fixed_public_wrapper_freezes_state_and_gva_contract() -> None:
    kernel_source, _ = _parse()
    wrapper_source, wrapper_tree = _parse_path(WRAPPER_PATH)
    wrapper = _function(wrapper_tree, "lightning_attn_fwd")
    wrapper_text = ast.unparse(wrapper)
    validation_text = ast.unparse(_function(wrapper_tree, "_validate_fixed_inputs"))

    assert "HV < H or HV % H" in validation_text
    assert "decay_s.shape not in {(H,), (HV,)}" in validation_text
    assert "initial_state.shape != (B, HV, VALUE_DIM, HEAD_DIM)" in validation_text
    assert "math.isfinite(float(scale))" in validation_text
    assert "initial_state is not None" in wrapper_text
    assert "output_final_state" in wrapper_text
    assert "torch.empty((B, HV, VALUE_DIM, HEAD_DIM)" in wrapper_text
    assert "initial_arg = initial_state if initial_state is not None else decay" in wrapper_text
    assert "torch.cuda.synchronize" not in wrapper_text
    assert not _calls(wrapper, "torch.Tensor.item")
    assert not _calls(wrapper, "torch.isfinite")
    assert "try:" not in wrapper_text and "except" not in wrapper_text
    assert "triton" not in (kernel_source + wrapper_source).lower()


def test_packed_mapping_tail_store_and_state_pool_contract() -> None:
    source, tree = _parse()
    wrapper_source, wrapper_tree = _parse_path(WRAPPER_PATH)
    backend = _class(tree, "LightningSm90PrefillKernel")
    call_text = ast.unparse(_method(backend, "__call__"))
    kernel_text = ast.unparse(_method(backend, "kernel_nonpersistent"))
    epilogue_text = ast.unparse(_method(backend, "run_epilogue_store"))
    validation = _function(wrapper_tree, "_validate_varlen_inputs")
    validation_text = ast.unparse(validation)
    wrapper = _function(wrapper_tree, "lightning_attn_fwd_varlen")
    wrapper_text = ast.unparse(wrapper)
    persistent_text = ast.unparse(_method(backend, "kernel_varlen_persistent"))
    persistent_scheduler_text = ast.unparse(_method(backend, "run_persistent_scheduler"))
    persistent_work_text = ast.unparse(_method(backend, "run_persistent_work_unit"))
    tail_gmem_text = ast.unparse(_method(backend, "tail_tensormap_gmem_ptr"))
    tail_generic_text = ast.unparse(_method(backend, "tail_tensormap_generic_ptr"))

    wrapper_values = _literal_assignments(wrapper_tree)
    assert wrapper_values["VARLEN_NONPERSISTENT_BACKEND_IDENTITY"] == (
        "cula.lightning.sm90a.cutedsl.prefill.varlen.nonpersistent"
    )
    assert wrapper_values["VARLEN_PERSISTENT_BACKEND_IDENTITY"] == (
        "cula.lightning.sm90a.cutedsl.prefill.varlen.persistent_static"
    )
    assert "cute.size(cu_seqlens_in) != self.num_sequences + 1" in call_text
    assert "grid = (1, HV, self.num_sequences)" in call_text
    assert "sequence_bos = cu_seqlens[batch_idx]" in kernel_text
    assert "tensormap_workspace_slot = cutlass.Int32(0)" in kernel_text
    assert "if cutlass.const_expr(self.is_varlen):" in kernel_text
    assert "tensormap_workspace_slot = batch_idx * cutlass.Int32(self.value_heads) + value_head_idx" in kernel_text
    assert "sequence_length_use = cu_seqlens[batch_idx + cutlass.Int32(1)] - sequence_bos" in kernel_text
    assert "state_idx = initial_state_indices[batch_idx]" in kernel_text
    assert "cute.domain_offset((sequence_bos, cutlass.Int32(0)), q_head)" in kernel_text
    assert "cute.domain_offset((cutlass.Int32(0), sequence_bos), k_head)" in kernel_text
    assert "needs_tail_tensormap = sequence_idx < cutlass.Int32(self.num_sequences - 1)" in epilogue_text
    assert "self.create_tail_tensormap(" in epilogue_text
    assert "use_tail_tensormap = needs_tail_tensormap and valid_tokens != cutlass.Int32(64)" in epilogue_text
    assert "local_epilogue_tid" not in epilogue_text
    assert "if token < valid_tokens:" not in epilogue_text
    assert "cu_seqlens.dtype != torch.int32" in validation_text
    assert "initial_state_indices.dtype != torch.int32" in validation_text
    assert "state_pool.shape[1:] != (HV, VALUE_DIM, HEAD_DIM)" in validation_text
    assert "torch.arange(N, dtype=torch.int32, device=Q.device)" in wrapper_text
    assert "state_pool, state_pool" in wrapper_text
    assert "get_device_sm_count(Q.device)" in wrapper_text
    assert "work_units = N * HV" in wrapper_text
    assert "persistent_ctas = min(work_units, sm_count)" in wrapper_text
    assert "workspace_slots = persistent_ctas" in wrapper_text
    assert "workspace_slots = work_units" in wrapper_text
    assert "_get_cache_buf('lightning_sm90_prefill_tensormaps', workspace_slots * TENSORMAP_BYTES, Q.device)" in wrapper_text
    assert "kernel = self.kernel_varlen_persistent(*kernel_args)" in call_text
    assert "grid = (self.persistent_ctas, 1, 1)" in call_text
    assert "self.run_persistent_scheduler" in persistent_text
    assert "work_idx = cutlass.Int32(cta_idx)" in persistent_scheduler_text
    assert "work_idx = work_idx + work_stride" in persistent_scheduler_text
    assert "while work_idx < total_work_units:" in persistent_scheduler_text
    assert "self.run_persistent_work_unit(tidx, warp_idx, warp_group_idx, work_idx, cta_idx" in (persistent_scheduler_text)
    assert "cute.arch.fence_view_async_shared()" in persistent_scheduler_text
    assert "cute.arch.sync_threads()" in persistent_scheduler_text
    assert "self.run_persistent_work_unit" in persistent_scheduler_text
    assert "PipelineTmaAsync" not in persistent_text
    assert "o_producer_state" in persistent_work_text
    assert "o_wait_state" in persistent_work_text and "o_release_state" in persistent_work_text
    assert "self.run_tma_load_producer" in persistent_work_text
    assert "self.run_math" in persistent_work_text
    assert "self.run_epilogue_store" in persistent_work_text
    assert "sequence_idx, tensormap_workspace_slot, g_tensormaps" in persistent_work_text
    assert "workspace_slot * cutlass.Int32(TENSORMAP_BYTES)" in tail_gmem_text
    assert "workspace_slot * cutlass.Int32(TENSORMAP_BYTES)" in tail_generic_text
    assert "%smid" not in source
    assert "_smid" not in source
    assert persistent_work_text.count("q_pipeline = pipeline.PipelineTmaAsync.create") == 1
    assert "torch.cuda.synchronize" not in wrapper_text
    assert not _calls(validation, "torch.Tensor.item")
    assert not _calls(wrapper, "torch.Tensor.item")
    assert "triton" not in (source + wrapper_source).lower()


@pytest.mark.parametrize(("num_sequences", "value_heads"), [(2, 1), (10, 64), (20, 64)])
def test_packed_tensormap_workspace_slots_are_unique_and_bounded(
    num_sequences: int,
    value_heads: int,
) -> None:
    work_units = num_sequences * value_heads
    nonpersistent_slots = [
        sequence_idx * value_heads + value_head_idx
        for sequence_idx in range(num_sequences)
        for value_head_idx in range(value_heads)
    ]

    assert nonpersistent_slots == list(range(work_units))
    assert max(nonpersistent_slots) < work_units

    persistent_ctas = min(work_units, 78)
    persistent_slots = list(range(persistent_ctas))
    assert len(set(persistent_slots)) == persistent_ctas
    assert max(persistent_slots) < persistent_ctas


def test_reference_gva_continuation_scale_and_basis_orientation() -> None:
    generator = torch.Generator().manual_seed(44017)
    batch, length, qk_heads, value_heads, key_dim, value_dim = 2, 67, 2, 4, 5, 3
    q = (torch.randn(batch, length, qk_heads, key_dim, generator=generator) * 0.03).to(torch.bfloat16)
    k = (torch.randn(batch, length, qk_heads, key_dim, generator=generator) * 0.01).to(torch.bfloat16)
    v = (torch.randn(batch, length, value_heads, value_dim, generator=generator) * 0.1).to(torch.bfloat16)
    h0 = torch.randn(batch, value_heads, value_dim, key_dim, generator=generator) * 0.07
    h0_before = h0.clone()
    decay_hv = torch.tensor((0.0, 0.009, 0.021, 0.037), dtype=torch.float32)

    full_output, full_state = tokenwise_lightning_reference(
        q,
        k,
        v,
        decay_hv,
        scale=0.73,
        initial_state=h0,
        output_final_state=True,
    )
    assert torch.equal(h0, h0_before)
    assert full_state is not None

    split = 33
    prefix_output, prefix_state = tokenwise_lightning_reference(
        q[:, :split],
        k[:, :split],
        v[:, :split],
        decay_hv,
        scale=0.73,
        initial_state=h0,
        output_final_state=True,
    )
    assert prefix_state is not None
    suffix_output, suffix_state = tokenwise_lightning_reference(
        q[:, split:],
        k[:, split:],
        v[:, split:],
        decay_hv,
        scale=0.73,
        initial_state=prefix_state,
        output_final_state=True,
    )
    assert suffix_state is not None
    assert torch.equal(torch.cat((prefix_output, suffix_output), dim=1), full_output)
    assert torch.equal(suffix_state, full_state)

    _, state_other_scale = tokenwise_lightning_reference(
        q,
        k,
        v,
        decay_hv,
        scale=-1.25,
        initial_state=h0,
        output_final_state=True,
    )
    assert torch.equal(state_other_scale, full_state)

    chunk_output, chunk_state = chunkwise_lightning_reference(
        q,
        k,
        v,
        decay_hv,
        scale=0.73,
        chunk_size=64,
        initial_state=h0,
        output_final_state=True,
    )
    torch.testing.assert_close(chunk_output, full_output, atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(chunk_state, full_state, atol=2.0e-6, rtol=2.0e-5)

    basis_q = torch.zeros(1, 1, 1, key_dim, dtype=torch.bfloat16)
    basis_k = torch.zeros_like(basis_q)
    basis_v = torch.zeros(1, 1, 2, value_dim, dtype=torch.bfloat16)
    basis_q[0, 0, 0, 1] = 1
    basis_state = torch.zeros(1, 2, value_dim, key_dim, dtype=torch.float32)
    basis_state[0, 1, 2, 1] = 3
    basis_output, _ = tokenwise_lightning_reference(
        basis_q,
        basis_k,
        basis_v,
        torch.zeros(2),
        initial_state=basis_state,
        output_final_state=False,
    )
    assert basis_output[0, 0, 1, 2] == 3
    assert basis_output.count_nonzero() == 1


@pytest.mark.parametrize("sequence_length", (1, 31, 32, 63, 64, 65, 127, 128, 129, 257))
def test_zero_state_semantics_cover_all_length_boundaries(sequence_length: int) -> None:
    generator = torch.Generator().manual_seed(23000 + sequence_length)
    q = (torch.randn(2, sequence_length, 3, 4, generator=generator) * 0.03).to(torch.bfloat16)
    k = (torch.randn(2, sequence_length, 3, 4, generator=generator) * 0.01).to(torch.bfloat16)
    v = (torch.randn(2, sequence_length, 3, 5, generator=generator) * 0.1).to(torch.bfloat16)
    decay_s = torch.tensor((0.005, 0.017, 0.031), dtype=torch.float32)

    tokenwise, tokenwise_state = tokenwise_lightning_reference(
        q,
        k,
        v,
        decay_s,
        scale=1.0,
        initial_state=None,
        output_final_state=False,
    )
    chunkwise, chunkwise_state = chunkwise_lightning_reference(
        q,
        k,
        v,
        decay_s,
        scale=1.0,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
    )
    assert tokenwise_state is None
    assert chunkwise_state is None
    torch.testing.assert_close(chunkwise, tokenwise, atol=2.0e-6, rtol=2.0e-5)
