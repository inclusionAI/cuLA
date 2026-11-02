# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.hybrid import DsaSelectionPlan, DsaSelectionWorkspace, cudnn_dsa_select


@pytest.mark.sm100_only
@pytest.mark.cula_slow
def test_windowed_dsa_selection_is_cuda_graph_set_deterministic():
    pytest.importorskip("cudnn")
    torch.manual_seed(43)
    device = torch.device("cuda")
    tokens = 4096
    blocks = tokens // 4
    top_k = 64
    plan = DsaSelectionPlan.from_cu_seqlens(
        [0, tokens],
        [0, blocks],
        top_k=top_k,
        candidate_budget_bytes=6 << 20,
    )
    assert len(plan.windows) > 1
    workspace = DsaSelectionWorkspace.from_plan(plan, device)
    qi = torch.randn(tokens, 4, 128, dtype=torch.bfloat16, device=device)
    kbar = torch.randn(blocks, 1, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(tokens, 4, dtype=torch.float32, device=device)

    def run():
        return cudnn_dsa_select(qi, kbar, weights, plan=plan, workspace=workspace)

    run()
    torch.cuda.synchronize()
    reference_ids = workspace.block_ids.clone()
    reference_scores = workspace.scores.clone()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()

    reference_sorted, reference_order = torch.sort(reference_ids, dim=-1)
    replay_sorted, replay_order = torch.sort(workspace.block_ids, dim=-1)
    reference_logits = torch.gather(reference_scores, -1, reference_order)
    replay_logits = torch.gather(workspace.scores, -1, replay_order)
    valid = reference_sorted >= 0
    assert torch.equal(reference_sorted, replay_sorted)
    assert torch.equal(reference_logits[valid], replay_logits[valid])
