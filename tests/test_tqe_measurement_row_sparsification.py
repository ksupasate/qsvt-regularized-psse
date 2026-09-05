"""Tests for Workstream C - measurement-row-level sparsification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from robust_qsvt_se.qsvt.output_aware_sparse_selection import _ridge_filter_operator
from robust_qsvt_se.tqe_extensions.row_sparsification import (
    STUDY_ID,
    _entry_mask_topk,
    _paired_stats,
    _row_mask,
    _rows_by_score,
    _rows_greedy_nnz,
    build_support,
    entry_magnitude_score,
    row_leverage_score,
    row_magnitude_score,
    row_sensitivity_scores,
    row_set_feasibility,
    run_row_sparsification,
)

SOURCE_ROOT = "outputs/output_aware_structural_generalization"


# ---------------------------------------------------------------- whole-row masks


def test_row_mask_keeps_whole_rows_only():
    matrix = np.array([[1.0, 0.0, 2.0], [3.0, 4.0, 0.0], [0.0, 5.0, 6.0]])
    mask = _row_mask(matrix, {0, 2})
    # kept rows keep exactly their original nonzeros; removed row fully off; no partial rows.
    assert mask[0].tolist() == [True, False, True]
    assert mask[2].tolist() == [False, True, True]
    assert not mask[1].any()
    for i in range(matrix.shape[0]):
        row_on = mask[i] & (matrix[i] != 0.0)
        assert np.array_equal(mask[i], row_on)  # never a zero entry turned on, never partial


def test_zero_row_mask_equals_explicit_row_slice():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(6, 4))
    residual = rng.normal(size=6)
    alpha = 0.7
    rows = {1, 3, 4}
    mask = _row_mask(matrix, rows)
    x_masked = _ridge_filter_operator(np.where(mask, matrix, 0.0), alpha) @ residual
    idx = sorted(rows)
    x_sliced = _ridge_filter_operator(matrix[idx, :], alpha) @ residual[idx]
    assert np.max(np.abs(x_masked - x_sliced)) < 1e-10


def test_entry_mask_can_be_partial_row():
    matrix = np.array([[3.0, 2.0, 1.0], [0.6, 0.5, 0.4]])
    mask = _entry_mask_topk(entry_magnitude_score(matrix), matrix, 2)
    assert int(mask.sum()) == 2
    # top-2 entries are both in row 0 -> a partial row, which is exactly what row selection forbids.
    assert mask[0].sum() == 2 and mask[1].sum() == 0


# ---------------------------------------------------------------- scores


def test_row_magnitude_is_l2_norm():
    matrix = np.array([[3.0, 4.0], [0.0, 5.0]])
    assert np.allclose(row_magnitude_score(matrix), [5.0, 5.0])


def test_row_leverage_matches_quadratic_form():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(5, 3))
    alpha = 0.3
    resolvent = np.linalg.inv(matrix.T @ matrix + alpha * np.eye(3))
    expected = np.array([matrix[i] @ resolvent @ matrix[i] for i in range(5)])
    assert np.allclose(row_leverage_score(matrix, alpha), expected)


def test_row_sensitivity_worst_is_min_over_tasks():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(4, 3))
    tasks = [(rng.normal(size=4), np.eye(3)[0]), (rng.normal(size=4), np.eye(3)[1])]
    mean = row_sensitivity_scores(matrix, tasks, 0.5, aggregation="mean")
    worst = row_sensitivity_scores(matrix, tasks, 0.5, aggregation="worst_case")
    assert np.all(worst <= mean + 1e-12)  # min over tasks never exceeds the mean


# ---------------------------------------------------------------- budgets / determinism


def test_rows_by_score_respects_count_and_ties():
    score = np.array([1.0, 1.0, 1.0, 0.0])  # three-way tie
    assert _rows_by_score(score, 2) == {0, 1}  # deterministic lexicographic tie-break


def test_greedy_nnz_respects_budget():
    score = np.array([5.0, 4.0, 3.0])
    row_nnz = np.array([3, 3, 3])
    assert _rows_greedy_nnz(score, row_nnz, 6) == {0, 1}  # 2 rows x 3 nnz = 6 <= budget
    assert _rows_greedy_nnz(score, row_nnz, 2) == set()  # nothing fits


def test_coverage_enforcement_flags_missing_type():
    meta = [{"measurement_type": "p"}, {"measurement_type": "q"}, {"measurement_type": "v"}]
    ok, reasons = row_set_feasibility({0, 1}, meta, {"require_type_coverage": True, "min_rows": 1})
    assert not ok and any("missing_measurement_types" in r for r in reasons)
    ok2, _ = row_set_feasibility({0, 1, 2}, meta, {"require_type_coverage": True, "min_rows": 1})
    assert ok2


# ---------------------------------------------------------------- statistics


def test_paired_stats_counts_and_direction():
    diffs = np.array([-0.1, -0.2, -0.05, 0.0, 0.3])  # 3 row-better, 1 tie, 1 row-worse
    stats = _paired_stats(diffs)
    assert stats["wins_row_better"] == 3
    assert stats["losses_row_worse"] == 1
    assert stats["ties"] == 1
    assert stats["median_effect"] == pytest.approx(-0.05)


# ---------------------------------------------------------------- no leakage (structural)


def test_build_support_signature_excludes_truth():
    import inspect

    params = set(inspect.signature(build_support).parameters)
    # Supports are built from (structure, selector, protocol, budget, train_tasks, config) only;
    # truth is never a construction input -> no held-out / ground-truth leakage by construction.
    assert params == {"structure", "selector", "protocol", "budget", "train_tasks", "config"}


# ---------------------------------------------------------------- integration (small)


def _tiny_config(tmp_path) -> Path:
    if not Path(SOURCE_ROOT, "instances").is_dir():
        pytest.skip("frozen structural instances unavailable")
    config = {
        "study_id": STUDY_ID,
        "source_structural_root": SOURCE_ROOT,
        "cases": ["ieee14", "ieee30", "ieee57"],
        "training_seed_count": 4,
        "heldout_seed_count": 4,
        "functional_classification": "physical",
        "support_floor": 1e-6,
        "physical_floor": 1e-6,
        "random_seed": 20260721,
        "row_swap_refinement": True,
        "max_structures": 4,
        "budgets": {"protocol_a_row_counts": [2, 3], "protocol_b_nnz": [4, 8]},
        "coverage_constraints": {"min_rows": 2},
        "contrasts": [
            {"name": "best_row_vs_best_entry", "row": "__best_row__", "entry": "__best_entry__"}
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_orchestrator_outputs_and_separated_metrics(tmp_path):
    cfg = _tiny_config(tmp_path)
    out = tmp_path / "out"
    summary = run_row_sparsification(cfg, out)
    for name in (
        "row_registry.csv",
        "selector_scores.csv",
        "support_registry.csv",
        "raw_evaluation_rows.csv",
        "structure_summary.csv",
        "case_summary.csv",
        "statistical_summary.csv",
        "resource_comparison.csv",
        "infeasibility_registry.csv",
        "claim_support.json",
        "run_manifest.json",
        "checksums.sha256",
        "README.md",
    ):
        assert (out / name).exists(), name
    raw = pd.read_csv(out / "raw_evaluation_rows.csv")
    assert "E_physical_norm" in raw.columns and "E_support_norm" in raw.columns  # kept separate
    assert summary["structures"] == 4
    claim = json.loads((out / "claim_support.json").read_text())
    assert claim["whole_row_retention_enforced"] is True
    assert claim["truth_used_only_for_evaluation"] is True


def test_run_is_reproducible(tmp_path):
    cfg = _tiny_config(tmp_path)
    run_row_sparsification(cfg, tmp_path / "a")
    run_row_sparsification(cfg, tmp_path / "b")
    h1 = hashlib.sha256((tmp_path / "a" / "raw_evaluation_rows.csv").read_bytes()).hexdigest()
    h2 = hashlib.sha256((tmp_path / "b" / "raw_evaluation_rows.csv").read_bytes()).hexdigest()
    assert h1 == h2


def test_infeasible_configs_retained(tmp_path):
    cfg = _tiny_config(tmp_path)
    out = tmp_path / "out"
    run_row_sparsification(cfg, out)
    support_reg = pd.read_csv(out / "support_registry.csv")
    # every support is recorded with a feasibility verdict; infeasible ones are not dropped.
    assert "feasible" in support_reg.columns
    infeasible = support_reg[~support_reg["feasible"].astype(bool)]
    inf_reg = pd.read_csv(out / "infeasibility_registry.csv")
    assert len(inf_reg) == len(infeasible)
