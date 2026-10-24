import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_INTRA_ENTRY_POINTS = {
    "chunk_kda_fwd_intra_sm100_equal",
    "chunk_kda_fwd_intra_sm100_training",
    "chunk_kda_fwd_intra_sm100_varlen",
}


def _is_mbarrier_init(call: ast.Call) -> bool:
    func = call.func
    return (isinstance(func, ast.Name) and func.id == "mbarrier_init") or (
        isinstance(func, ast.Attribute) and func.attr == "mbarrier_init"
    )


def _is_arch_barrier(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == "barrier"


def _is_elect_one_context(with_item: ast.withitem) -> bool:
    context = with_item.context_expr
    return isinstance(context, ast.Call) and (
        (isinstance(context.func, ast.Name) and context.func.id == "elect_one")
        or (isinstance(context.func, ast.Attribute) and context.func.attr == "elect_one")
    )


def _is_single_thread_condition(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    operands = (test.left, test.comparators[0])
    has_thread_index = any(isinstance(value, ast.Name) and value.id in {"tidx", "thread_idx"} for value in operands)
    has_zero = any(isinstance(value, ast.Constant) and value.value == 0 for value in operands)
    return has_thread_index and has_zero


def test_manual_mbarrier_init_is_single_threaded():
    """Every manual mbarrier initialization must be issued by one CTA thread."""
    failures = []
    for path in (REPO_ROOT / "cula").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_mbarrier_init(node):
                continue

            guarded = False
            ancestor = parents.get(node)
            while ancestor is not None:
                if isinstance(ancestor, (ast.With, ast.AsyncWith)) and any(
                    _is_elect_one_context(item) for item in ancestor.items
                ):
                    guarded = True
                    break
                if isinstance(ancestor, ast.If) and _is_single_thread_condition(ancestor.test):
                    guarded = True
                    break
                ancestor = parents.get(ancestor)

            if not guarded:
                failures.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not failures, "mbarrier_init must run in one CTA thread: " + ", ".join(failures)


def test_sm100_intra_exposes_only_the_exact_gk_path():
    path = REPO_ROOT / "cula/ops/kda/sm100/intra_fused.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    defined_names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    assert LEGACY_INTRA_ENTRY_POINTS.isdisjoint(defined_names | assigned_names)


def test_lightning_persistent_scheduler_has_no_divergent_cta_barrier():
    """Warp-role branches must rendezvous through mbarrier, not bar.sync."""
    path = REPO_ROOT / "cula/ops/lightning/prefill_sm100.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    barrier_one_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_arch_barrier(node):
            continue
        barrier_id = next((keyword.value for keyword in node.keywords if keyword.arg == "barrier_id"), None)
        if isinstance(barrier_id, ast.Constant) and barrier_id.value == 1:
            barrier_one_calls.append(node.lineno)

    assert not barrier_one_calls, (
        "barrier id 1 is reserved by the TMEM allocator; persistent warp-role "
        f"branches must use sched_done_mbar instead: {barrier_one_calls}"
    )
