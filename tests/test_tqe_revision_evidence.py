from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robust_qsvt_se.experiments.tqe_revision_evidence import (
    NONLINEAR_ITERATION_COLUMNS,
    QSVT_MATRIX_COLUMNS,
    pseudoinverse_svd_update,
    ridge_svd_update,
    run_larger_qsvt_matrix_validation,
    run_nonlinear_convergence_comparison,
    run_tqe_revision_evidence,
    select_deterministic_block,
    sparsity_pattern_preserved_by_diagonal_weighting,
)


def test_ridge_svd_update_matches_normal_equations() -> None:
    H = np.array([[2.0, 0.1], [0.0, 0.2], [1.0, -0.4]], dtype=np.float64)
    r = np.array([1.0, -0.25, 0.5], dtype=np.float64)
    alpha = 0.2

    svd_update = ridge_svd_update(H, r, alpha=alpha)
    normal_update = np.linalg.solve(H.T @ H + alpha * np.eye(H.shape[1]), H.T @ r)

    np.testing.assert_allclose(svd_update, normal_update, rtol=1.0e-12, atol=1.0e-12)


def test_ridge_damps_weak_singular_direction_relative_to_pinv() -> None:
    H = np.diag([1.0, 1.0e-6])
    r = np.array([0.0, 1.0], dtype=np.float64)

    pinv_update = pseudoinverse_svd_update(H, r, rcond=0.0)
    ridge_update = ridge_svd_update(H, r, alpha=1.0e-4)

    assert abs(ridge_update[1]) < 1.0e-6 * abs(pinv_update[1])


def test_diagonal_weighting_preserves_sparsity_pattern() -> None:
    H = np.array([[1.0, 0.0, -2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 0.0]])
    stds = np.array([2.0, 0.5, 10.0])
    weighted = H / stds[:, None]

    assert sparsity_pattern_preserved_by_diagonal_weighting(H, weighted)


def test_nonlinear_convergence_logging_schema(tmp_path: Path) -> None:
    run = run_nonlinear_convergence_comparison(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee14"],
            "case_source": "builtin",
            "seeds": [0],
            "max_iterations": 2,
            "noise_std": 0.0,
            "missing_ratio": 0.0,
            "save_plots": False,
        }
    )

    frame = run["iteration_metrics"]
    assert set(NONLINEAR_ITERATION_COLUMNS).issubset(frame.columns)
    assert {"pinv", "ridge"}.issubset(set(frame["solver_name"]))
    assert (tmp_path / "nonlinear_convergence_iterations.csv").is_file()
    assert (tmp_path / "nonlinear_convergence_summary.csv").is_file()


def test_qsvt_matrix_validation_schema_and_failure_status(tmp_path: Path) -> None:
    run = run_larger_qsvt_matrix_validation(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee14"],
            "case_source": "builtin",
            "seed": 0,
            "block_sizes": [16, 32],
            "degrees": [5],
            "grid_size": 64,
            "target_epsilon": 1.0,
            "save_plots": False,
        }
    )

    frame = run["validation"]
    assert set(QSVT_MATRIX_COLUMNS).issubset(frame.columns)
    assert len(frame) == 2
    assert frame["failure_reason"].fillna("").str.len().max() > 0
    assert (tmp_path / "larger_qsvt_matrix_validation.csv").is_file()


def test_deterministic_block_selection_repeats_with_same_inputs() -> None:
    rng = np.random.default_rng(123)
    H = rng.normal(size=(20, 12))
    r = rng.normal(size=20)

    first = select_deterministic_block(H, r, row_count=8, col_count=8)
    second = select_deterministic_block(H, r, row_count=8, col_count=8)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left, right)


def test_tqe_revision_evidence_manifest_and_readme(tmp_path: Path) -> None:
    run = run_tqe_revision_evidence(
        {
            "tasks": ["sparse"],
            "cases": ["ieee14"],
            "case_source": "builtin",
            "seeds": [0],
            "output_dir": str(tmp_path / "evidence"),
            "sparse": {"save_plots": False},
        }
    )
    output_dir = run["output_dir"]

    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"
    assert manifest_path.is_file()
    assert readme_path.is_file()
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    assert "sparse_access_diagnostics" in manifest["output_files_generated"]
    assert manifest["skipped_tasks_and_reasons"]["nonlinear"] == "task not selected"
