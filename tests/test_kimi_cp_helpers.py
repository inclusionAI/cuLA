# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from cula.distributed import kda as cp
from cula.distributed.plan import KimiCPPlan


def test_cudnn_loader_accepts_current_linear_attention_exports(monkeypatch):
    def merge(h, m, *, reverse=False):
        return h, m, reverse

    sentinels = {name: object() for name in ("op", "summary", "summary_bwd")}
    modules = {
        "cudnn": SimpleNamespace(),
        "cudnn.linear_attention": SimpleNamespace(
            kimi_cp_merge=merge,
            kimi_delta_attention=sentinels["op"],
            kimi_delta_attention_summary=sentinels["summary"],
            kimi_delta_attention_summary_bwd=sentinels["summary_bwd"],
        ),
    }

    def import_fake(name):
        try:
            return modules[name]
        except KeyError as error:
            raise ModuleNotFoundError(name) from error

    monkeypatch.setattr(cp, "import_module", import_fake)
    cp._load_cudnn_cp_ops.cache_clear()
    try:
        loaded = cp._load_cudnn_cp_ops()
        assert loaded[0] is merge
        assert loaded[1:] == tuple(sentinels.values())
    finally:
        cp._load_cudnn_cp_ops.cache_clear()


def test_cudnn_loader_accepts_legacy_top_level_merge_export(monkeypatch):
    observed = {}

    def merge(h, m, *, reverse=False, precision="bf16"):
        observed["precision"] = precision
        return h, m, reverse

    sentinels = {name: object() for name in ("op", "summary", "summary_bwd")}
    modules = {
        "cudnn": SimpleNamespace(kimi_cp_merge=merge),
        "cudnn.linear_attention": SimpleNamespace(),
        "cudnn.linear_attention.ops": SimpleNamespace(
            kimi_delta_attention=sentinels["op"],
            kimi_delta_attention_summary=sentinels["summary"],
            kimi_delta_attention_summary_bwd=sentinels["summary_bwd"],
        ),
    }

    def import_fake(name):
        try:
            return modules[name]
        except KeyError as error:
            raise ModuleNotFoundError(name) from error

    monkeypatch.setattr(cp, "import_module", import_fake)
    cp._load_cudnn_cp_ops.cache_clear()
    try:
        loaded = cp._load_cudnn_cp_ops()
        assert loaded[1:] == tuple(sentinels.values())
        loaded[0](object(), object(), reverse=True)
        assert observed["precision"] == "ieee"
    finally:
        cp._load_cudnn_cp_ops.cache_clear()


def test_boundary_summary_pack_round_trip_preserves_fp32():
    h = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    transition = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)

    packed = cp._pack_boundary_summary(h, transition)
    h_result, transition_result = cp._unpack_boundary_summaries(packed, h.shape[-2])

    assert packed.shape == (2, 7, 4)
    assert packed.dtype == torch.float32
    assert torch.equal(h_result, h)
    assert torch.equal(transition_result, transition)


def test_boundary_chain_selects_only_matching_remote_ranks(monkeypatch):
    h_all = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1)
    transition_all = torch.zeros(4, 1, 1, 1)
    calls = []

    def fake_merge(h, transition, *, reverse):
        calls.append((h.flatten().tolist(), reverse))
        assert transition.shape == h.shape
        return h.sum(dim=0)

    monkeypatch.setattr(cp, "_merge_rank_summaries", fake_merge)

    incoming = cp._merge_boundary_chain(
        h_all,
        transition_all,
        rank=2,
        num_remote_ranks=2,
        reverse=False,
    )
    outgoing = cp._merge_boundary_chain(
        h_all,
        transition_all,
        rank=1,
        num_remote_ranks=2,
        reverse=True,
    )
    no_remote_state = cp._merge_boundary_chain(
        h_all,
        transition_all,
        rank=0,
        num_remote_ranks=0,
        reverse=False,
    )

    assert incoming.item() == 3.0
    assert outgoing.item() == 7.0
    assert no_remote_state.item() == 0.0
    assert calls == [([1.0, 2.0], False), ([3.0, 4.0], True)]


def test_boundary_pack_rejects_precision_downgrade():
    h = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    transition = torch.zeros(2, 4, 4, dtype=torch.float32)

    try:
        cp._pack_boundary_summary(h, transition)
    except TypeError as error:
        assert "FP32" in str(error)
    else:
        raise AssertionError("BF16 boundary state must be rejected")


def test_merge_uses_cudnn_fixed_ieee_api_without_precision_option(monkeypatch):
    observed = {}

    def fake_merge(h, transition, *, reverse):
        observed.update(h=h, transition=transition, reverse=reverse)
        return {"final_state": h[-1:]}

    monkeypatch.setattr(cp, "_load_cudnn_cp_ops", lambda: (fake_merge, None, None, None))
    h = torch.ones(2, 1, 1, 1, dtype=torch.float32)
    transition = torch.zeros_like(h)

    result = cp._merge_rank_summaries(h, transition, reverse=True)

    assert result.item() == 1.0
    assert observed["reverse"] is True


def _transport_plan(*, rank=2, pre=2, post=1):
    return KimiCPPlan(
        rank=rank,
        world_size=4,
        token_start=8,
        token_end=12,
        first_sequence=0,
        last_sequence=0,
        local_cu_seqlens=(0, 4),
        first_boundary_end=4,
        last_boundary_start=0,
        pre_num_ranks=pre,
        post_num_ranks=post,
    )


def test_collective_workspace_has_stable_fp32_storage():
    workspace = cp._KimiCPCollectiveWorkspace.allocate(
        world_size=4,
        heads=2,
        key_dim=4,
        value_dim=3,
        device="cpu",
    )

    assert workspace.local_payload.shape == (2, 7, 4)
    assert workspace.gathered_payload.shape == (8, 7, 4)
    assert workspace.local_payload.dtype == workspace.gathered_payload.dtype == torch.float32
    assert workspace.nbytes == (2 * 7 * 4 + 8 * 7 * 4) * 4
    cp64_plan = KimiCPPlan(
        rank=0,
        world_size=64,
        token_start=0,
        token_end=1,
        first_sequence=0,
        last_sequence=0,
        local_cu_seqlens=(0, 1),
        first_boundary_end=1,
        last_boundary_start=0,
        pre_num_ranks=0,
        post_num_ranks=63,
    )
    required = cp.AllGatherKimiCPTransport.required_workspace_bytes(
        cp64_plan,
        heads=32,
        key_dim=128,
        value_dim=128,
    )
    assert required == 260 << 20

    try:
        cp.AllGatherKimiCPTransport(workspace=object())
    except TypeError as error:
        assert "prepare" in str(error)
    else:
        raise AssertionError("transport must reject an unrelated workspace object")


def test_prepared_all_gather_transport_reuses_workspace(monkeypatch):
    plan = _transport_plan()
    transport = cp.AllGatherKimiCPTransport.prepare(
        plan,
        heads=1,
        key_dim=1,
        value_dim=1,
        device="cpu",
    )
    workspace = transport.workspace
    assert workspace is not None
    local_pointer = workspace.local_payload.data_ptr()
    gathered_pointer = workspace.gathered_payload.data_ptr()

    def fake_all_gather(output, tensor, *, group):
        assert group == "cp"
        gathered = output.view(4, *tensor.shape)
        for rank in range(4):
            gathered[rank].copy_(tensor)
            gathered[rank, :, 0, :].fill_(rank + 1)

    calls = []

    def fake_merge(h, transition, *, reverse):
        calls.append((h.flatten().tolist(), transition.flatten().tolist(), reverse))
        return h.sum(dim=0)

    monkeypatch.setattr(cp, "_all_gather_into", fake_all_gather)
    monkeypatch.setattr(cp, "_merge_rank_summaries", fake_merge)
    h = torch.ones(1, 1, 1, dtype=torch.float32)
    transition = torch.zeros(1, 1, 1, dtype=torch.float32)

    incoming = transport.boundary_state(h, transition, plan=plan, group="cp", reverse=False)
    outgoing = transport.boundary_state(
        h,
        transition,
        plan=_transport_plan(rank=1, pre=1, post=2),
        group="cp",
        reverse=True,
    )

    assert incoming.item() == 3
    assert outgoing.item() == 7
    assert calls == [([1.0, 2.0], [0.0, 0.0], False), ([3.0, 4.0], [0.0, 0.0], True)]
    assert workspace.local_payload.data_ptr() == local_pointer
    assert workspace.gathered_payload.data_ptr() == gathered_pointer


def test_collective_workspace_rejects_shape_or_world_mismatch():
    workspace = cp._KimiCPCollectiveWorkspace.allocate(
        world_size=4,
        heads=2,
        key_dim=4,
        value_dim=3,
        device="cpu",
    )
    h = torch.zeros(2, 3, 4, dtype=torch.float32)
    transition = torch.zeros(2, 4, 4, dtype=torch.float32)

    workspace.validate(h, transition, world_size=4)
    for bad_h, world_size, message in (
        (h[:1], 4, "shape mismatch"),
        (h, 8, "world_size"),
    ):
        try:
            workspace.validate(bad_h, transition, world_size=world_size)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("workspace mismatch must be rejected")

    try:
        workspace.validate(h.to(torch.bfloat16), transition, world_size=4)
    except TypeError as error:
        assert "FP32" in str(error)
    else:
        raise AssertionError("workspace must not silently cast BF16 summaries")


def test_boundary_transport_protocol_is_structural():
    class CustomTransport:
        def boundary_state(self, h, transition, *, plan, group, reverse):
            return h

    assert isinstance(CustomTransport(), cp.KimiCPBoundaryTransport)
    assert not isinstance(object(), cp.KimiCPBoundaryTransport)


def test_boundary_transport_state_contract_rejects_precision_or_shape_changes():
    reference = torch.zeros(2, 3, 4, dtype=torch.float32)
    noncontiguous = torch.zeros(3, 2, 4, dtype=torch.float32).transpose(0, 1)
    assert cp._validate_boundary_state(reference, reference) is reference

    for state, error_type, message in (
        (None, TypeError, "torch.Tensor"),
        (reference.to(torch.bfloat16), TypeError, "FP32"),
        (reference[:1], ValueError, "mismatch"),
        (noncontiguous, ValueError, "contiguous"),
    ):
        try:
            cp._validate_boundary_state(state, reference)
        except error_type as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid transport state must be rejected")
