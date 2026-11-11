# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from cula.distributed import kda as cp
from cula.distributed.plan import KimiCPPlan


def test_checkpoint_cadence_validation():
    assert cp._validate_checkpoint_every_n_tokens(0) == 0
    assert cp._validate_checkpoint_every_n_tokens(16) == 16

    for invalid in (-16, 1, 24, 32, 64, 128, 256):
        try:
            cp._validate_checkpoint_every_n_tokens(invalid)
        except ValueError as error:
            assert "native chunk cadence 16" in str(error)
        else:
            raise AssertionError(f"invalid checkpoint cadence {invalid} was accepted")

    for invalid in (True, 16.0, "16"):
        try:
            cp._validate_checkpoint_every_n_tokens(invalid)
        except TypeError as error:
            assert "integer" in str(error)
        else:
            raise AssertionError(f"non-integer checkpoint cadence {invalid!r} was accepted")


def test_attention_kwargs_keep_checkpoints_out_of_summary_calls():
    options = cp._KdaOptions(
        scale=None,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=False,
        lower_bound=None,
        gate_domain="log",
        batch_invariant=True,
        checkpoint_every_n_tokens=16,
        plan_name="kda_frost",
        summary_plan_name="kda_summary_frost",
    )

    assert "checkpoint_every_n_tokens" not in options.kwargs(None, None)
    assert options.attention_kwargs(None, None)["checkpoint_every_n_tokens"] == 16


def test_cudnn_loader_accepts_current_linear_attention_exports(monkeypatch):
    def attention(*args, initial_state=None, preceding_state_update=None, preceding_transition=None, **kwargs):
        return args, preceding_state_update, preceding_transition, kwargs

    def summary(*args, **kwargs):
        return args, kwargs

    def summary_bwd(*args, output_context=False, checkpoint_every_n_tokens=0, **kwargs):
        return args, output_context, checkpoint_every_n_tokens, kwargs

    backward = object()
    modules = {
        "cudnn.linear_attention": SimpleNamespace(
            kimi_delta_attention=attention,
            kimi_delta_attention_summary=summary,
            kimi_delta_attention_summary_bwd=summary_bwd,
        ),
    }

    def import_fake(name):
        try:
            return modules[name]
        except KeyError as error:
            raise ModuleNotFoundError(name) from error

    monkeypatch.setattr(cp, "import_module", import_fake)
    monkeypatch.setattr(cp.torch.ops.cudnn, "kimi_delta_attention_bwd", backward, raising=False)
    cp._load_cudnn_kda_ops.cache_clear()
    cp._load_cudnn_cp_ops.cache_clear()
    try:
        loaded = cp._load_cudnn_cp_ops()
        assert loaded == (attention, summary, summary_bwd, backward)
    finally:
        cp._load_cudnn_cp_ops.cache_clear()
        cp._load_cudnn_kda_ops.cache_clear()


def test_cudnn_kda_loader_does_not_require_frost_summary_exports(monkeypatch):
    def attention(*args, initial_state=None, **kwargs):
        return args, initial_state, kwargs

    backward = object()
    monkeypatch.setattr(cp, "import_module", lambda name: SimpleNamespace(kimi_delta_attention=attention))
    monkeypatch.setattr(cp.torch.ops.cudnn, "kimi_delta_attention_bwd", backward, raising=False)
    cp._load_cudnn_kda_ops.cache_clear()
    try:
        assert cp._load_cudnn_kda_ops() == (attention, backward)
    finally:
        cp._load_cudnn_kda_ops.cache_clear()


def test_cudnn_loader_rejects_pre_concatenated_summary_api(monkeypatch):
    def old_attention(*args, initial_state=None, **kwargs):
        return args, kwargs

    modules = {
        "cudnn.linear_attention": SimpleNamespace(
            kimi_delta_attention=old_attention,
            kimi_delta_attention_summary=object(),
            kimi_delta_attention_summary_bwd=object(),
        ),
    }

    def import_fake(name):
        try:
            return modules[name]
        except KeyError as error:
            raise ModuleNotFoundError(name) from error

    monkeypatch.setattr(cp, "import_module", import_fake)
    cp._load_cudnn_kda_ops.cache_clear()
    cp._load_cudnn_cp_ops.cache_clear()
    try:
        try:
            cp._load_cudnn_cp_ops()
        except ImportError as error:
            assert "concatenated KDA summary" in str(error)
        else:
            raise AssertionError("the retired stand-alone merge API must not be accepted")
    finally:
        cp._load_cudnn_cp_ops.cache_clear()
        cp._load_cudnn_kda_ops.cache_clear()


def test_local_boundary_summary_contract_preserves_fp32():
    h = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    transition = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)

    cp._validate_local_boundary_summaries(h, transition)


def test_boundary_chain_selects_only_matching_remote_ranks():
    h_all = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1)
    transition_all = torch.zeros(4, 1, 1, 1)

    incoming = cp._select_boundary_summaries(
        h_all,
        transition_all,
        rank=2,
        num_remote_ranks=2,
        reverse=False,
    )
    outgoing = cp._select_boundary_summaries(
        h_all,
        transition_all,
        rank=1,
        num_remote_ranks=2,
        reverse=True,
    )
    no_remote = cp._select_boundary_summaries(
        h_all,
        transition_all,
        rank=0,
        num_remote_ranks=0,
        reverse=False,
    )

    assert incoming[0].flatten().tolist() == [1.0, 2.0]
    assert outgoing[0].flatten().tolist() == [3.0, 4.0]
    assert incoming[1].shape == outgoing[1].shape == (2, 1, 1, 1)
    assert no_remote[0].shape == no_remote[1].shape == (0, 1, 1, 1)


def test_local_boundary_summary_contract_rejects_precision_downgrade():
    h = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    transition = torch.zeros(2, 4, 4, dtype=torch.float32)

    try:
        cp._validate_local_boundary_summaries(h, transition)
    except TypeError as error:
        assert "FP32" in str(error)
    else:
        raise AssertionError("BF16 boundary state must be rejected")


def test_affine_summary_composition_uses_causal_order_in_both_directions():
    updates = torch.tensor([2.0, 4.0], dtype=torch.float32).reshape(2, 1, 1, 1)
    transitions = torch.tensor([5.0, 7.0], dtype=torch.float32).reshape(2, 1, 1, 1)

    forward = cp._compose_affine_summaries(updates, transitions, reverse=False)
    reverse = cp._compose_affine_summaries(updates, transitions, reverse=True)

    assert forward.item() == 18.0  # (0 * 5 + 2) * 7 + 4
    assert reverse.item() == 22.0  # (0 * 7 + 4) * 5 + 2


def test_hopper_state_probe_recovers_fp32_affine_map(monkeypatch):
    heads, dim = 2, 4
    state_update = torch.arange(heads * dim * dim, dtype=torch.float32).reshape(1, heads, dim, dim) / 32
    transition = torch.eye(dim, dtype=torch.float32).expand(heads, dim, dim).clone()
    transition[:, 0, 1] = 0.25

    def attention(q, k, v, g, beta, cu, *, initial_state, **kwargs):
        final = torch.matmul(initial_state, transition) + state_update
        return torch.zeros_like(v), final, torch.empty(0), torch.empty(0)

    monkeypatch.setattr(cp, "_load_cudnn_kda_ops", lambda: (attention, object()))
    options = cp._KdaOptions(
        scale=None,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=False,
        lower_bound=None,
        gate_domain="log",
        batch_invariant=True,
        checkpoint_every_n_tokens=0,
        plan_name=None,
        summary_plan_name=None,
    )
    q = torch.zeros(8, heads, dim)
    beta = torch.zeros(8, heads)
    cu = torch.tensor([0, 8], dtype=torch.int32)

    actual_h, actual_transition = cp._state_probe_summary(q, q, q, q, beta, cu, options=options)

    assert actual_h.dtype == actual_transition.dtype == torch.float32
    torch.testing.assert_close(actual_h, state_update[0])
    torch.testing.assert_close(actual_transition, transition)


def test_hopper_state_probe_rejects_in_kernel_backward_transforms():
    base = dict(
        scale=None,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=False,
        lower_bound=None,
        gate_domain="log",
        batch_invariant=True,
        checkpoint_every_n_tokens=0,
        plan_name=None,
        summary_plan_name=None,
    )
    for option in ("use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel", "safe_gate"):
        values = dict(base)
        values[option] = True
        try:
            cp._validate_hopper_state_probe_options(cp._KdaOptions(**values))
        except ValueError as error:
            assert option in str(error)
        else:
            raise AssertionError(f"SM90 accepted unsupported Hopper option {option}")


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

    assert workspace.gathered_h.shape == (8, 3, 4)
    assert workspace.gathered_transition.shape == (8, 4, 4)
    assert workspace.gathered_h.dtype == workspace.gathered_transition.dtype == torch.float32
    assert workspace.nbytes == (8 * 3 * 4 + 8 * 4 * 4) * 4
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
    assert required == 256 << 20

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
    h_pointer = workspace.gathered_h.data_ptr()
    transition_pointer = workspace.gathered_transition.data_ptr()
    calls = 0

    def fake_all_gather(output, tensor, *, group):
        nonlocal calls
        assert group == "cp"
        gathered = output.view(4, *tensor.shape)
        for rank in range(4):
            gathered[rank].copy_(tensor)
            if calls % 2 == 0:
                gathered[rank, :, 0, :].fill_(rank + 1)
        calls += 1

    monkeypatch.setattr(cp, "_all_gather_into", fake_all_gather)
    h = torch.ones(1, 1, 1, dtype=torch.float32)
    transition = torch.zeros(1, 1, 1, dtype=torch.float32)

    incoming = transport.boundary_summaries(h, transition, plan=plan, group="cp", reverse=False)
    outgoing = transport.boundary_summaries(
        h,
        transition,
        plan=_transport_plan(rank=1, pre=1, post=2),
        group="cp",
        reverse=True,
    )

    assert incoming[0].flatten().tolist() == [1.0, 2.0]
    assert outgoing[0].flatten().tolist() == [3.0, 4.0]
    assert incoming[1].flatten().tolist() == outgoing[1].flatten().tolist() == [0.0, 0.0]
    assert workspace.gathered_h.data_ptr() == h_pointer
    assert workspace.gathered_transition.data_ptr() == transition_pointer


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
        def boundary_summaries(self, h, transition, *, plan, group, reverse):
            return h.unsqueeze(0), transition.unsqueeze(0)

    assert isinstance(CustomTransport(), cp.KimiCPBoundaryTransport)
    assert not isinstance(object(), cp.KimiCPBoundaryTransport)


def test_boundary_transport_summary_contract_rejects_precision_or_shape_changes():
    h_reference = torch.zeros(2, 3, 4, dtype=torch.float32)
    transition_reference = torch.zeros(2, 4, 4, dtype=torch.float32)
    valid = (h_reference.unsqueeze(0), transition_reference.unsqueeze(0))
    assert (
        cp._validate_boundary_summaries(
            valid,
            h_reference,
            transition_reference,
            rows=1,
        )
        is valid
    )

    noncontiguous = torch.zeros(1, 3, 2, 4, dtype=torch.float32).transpose(1, 2)
    for summaries, error_type, message in (
        (None, TypeError, "tuple"),
        ((valid[0].to(torch.bfloat16), valid[1]), TypeError, "FP32"),
        ((valid[0][:0], valid[1]), ValueError, "mismatch"),
        ((noncontiguous, valid[1]), ValueError, "contiguous"),
    ):
        try:
            cp._validate_boundary_summaries(
                summaries,
                h_reference,
                transition_reference,
                rows=1,
            )
        except error_type as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid transport summaries must be rejected")


def test_distributed_autograd_delegates_summary_composition_to_cudnn(monkeypatch):
    plan = _transport_plan(rank=2, pre=2, post=1)
    heads, key_dim, value_dim = 2, 4, 3
    q = torch.zeros(4, heads, key_dim)
    k = torch.zeros_like(q)
    v = torch.zeros(4, heads, value_dim)
    g = torch.zeros_like(q)
    beta = torch.zeros(4, heads)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    options = cp._KdaOptions(
        scale=None,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=False,
        lower_bound=None,
        gate_domain="log",
        batch_invariant=True,
        checkpoint_every_n_tokens=16,
        plan_name="kda_frost",
        summary_plan_name="kda_summary_frost",
    )
    observed = {}

    class Transport:
        def boundary_summaries(self, h, transition, *, plan, group, reverse):
            rows = plan.post_num_ranks if reverse else plan.pre_num_ranks
            observed["transport_reverse"] = reverse
            return h.unsqueeze(0).expand(rows, *h.shape).contiguous(), transition.unsqueeze(0).expand(
                rows, *transition.shape
            ).contiguous()

    class Context:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    def summary(k, v, g, beta, cu, **kwargs):
        return torch.ones(1, heads, value_dim, key_dim), torch.full((1, heads, key_dim, key_dim), 2.0)

    def attention(q, k, v, g, beta, cu, **kwargs):
        observed["preceding_h"] = kwargs["preceding_state_update"].clone()
        observed["preceding_transition"] = kwargs["preceding_transition"].clone()
        return (
            torch.ones_like(v),
            torch.empty(0),
            torch.full((1, heads, value_dim, key_dim), 3.0),
            torch.full((1, heads, value_dim, key_dim), 4.0),
        )

    def summary_bwd(q, k, g, beta, d_o, cu, **kwargs):
        assert kwargs["output_context"] is True
        assert kwargs["checkpoint_every_n_tokens"] == 16
        return (
            torch.full((1, heads, value_dim, key_dim), 5.0),
            torch.full((1, heads, key_dim, key_dim), 6.0),
            torch.tensor([7], dtype=torch.uint8),
        )

    def backward(d_o, q, k, v, g, beta, cu, scale, **kwargs):
        observed["following_h"] = kwargs["following_state_grad"].clone()
        observed["following_transition"] = kwargs["following_transition"].clone()
        observed["composed"] = kwargs["composed_initial_state"].clone()
        observed["context"] = kwargs["context"].clone()
        observed["checkpoints"] = kwargs["state_checkpoints"].clone()
        return (
            torch.ones_like(q),
            torch.ones_like(k),
            torch.ones_like(v),
            torch.ones_like(g),
            torch.ones_like(beta),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        )

    monkeypatch.setattr(cp, "_load_cudnn_cp_ops", lambda: (attention, summary, summary_bwd, backward))
    ctx = Context()
    output = cp._CudnnKdaCP.forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        cu,
        None,
        None,
        plan,
        "cp",
        options,
        Transport(),
    )
    gradients = cp._CudnnKdaCP.backward(ctx, torch.ones_like(output))

    assert observed["preceding_h"].shape[0] == 2
    assert observed["preceding_transition"].shape[0] == 2
    assert observed["following_h"].shape[0] == 1
    assert torch.equal(observed["following_transition"], torch.full((1, heads, key_dim, key_dim), 6.0))
    assert torch.equal(observed["composed"], torch.full((1, heads, value_dim, key_dim), 4.0))
    assert torch.equal(observed["context"], torch.tensor([7], dtype=torch.uint8))
    assert torch.equal(observed["checkpoints"], torch.full((1, heads, value_dim, key_dim), 3.0))
    assert len(gradients) == 12
    assert gradients[6:8] == (None, None)


def test_single_rank_fast_path_calls_attention_export(monkeypatch):
    plan = KimiCPPlan.from_global_cu_seqlens([0, 2], rank=0, world_size=1)
    q = torch.zeros(2, 1, 4)
    v = torch.zeros(2, 1, 3)
    beta = torch.zeros(2, 1)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    observed = {}

    def attention(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return torch.ones_like(v), torch.empty(0)

    monkeypatch.setattr(cp, "_validate_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(cp, "_load_cudnn_kda_ops", lambda: (attention, object()))

    result = cp.cudnn_kda_cp(q, q, v, q, beta, cu, plan=plan)

    assert torch.equal(result, torch.ones_like(v))
    assert all(actual is expected for actual, expected in zip(observed["args"][:5], (q, q, v, q, beta), strict=True))
