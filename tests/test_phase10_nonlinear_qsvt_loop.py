"""Tests for Phase 10 WP D nonlinear AC QSVT-in-the-loop simulator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.paper.phase10_nonlinear_qsvt_loop import (
    _matrix_level_qsvt_update,
    build_ieee14_config,
    run_phase10_nonlinear_qsvt_loop,
)


def test_matrix_level_qsvt_matches_ridge_at_matched_alpha_controlled():
    # Controlled small well-conditioned system: the bounded polynomial certifies
    # and the matrix-level QSVT approximant matches Ridge closely.
    rng = np.random.default_rng(0)
    H = rng.standard_normal((12, 5))
    r = rng.standard_normal(12)
    beta = float(np.linalg.svd(H, compute_uv=False).max())
    alpha = 0.068 * beta**2
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        diag = _matrix_level_qsvt_update(H, r, alpha=alpha, phase_cache_dir=tmp)
    assert diag["polynomial_status"] == "bounded_certified"
    # The exact-filter loop step equals Ridge by construction.
    assert diag["update_error_vs_ridge"] == 0.0
    # The degree-d bounded-polynomial approximant matches Ridge closely.
    assert diag["approximant_rel_error_vs_ridge"] < 0.05
    # beta_k and lambda_k are recomputed from the current Jacobian.
    assert diag["beta_k"] == pytest.approx(beta, rel=1e-12)
    assert diag["lambda_k"] == pytest.approx(alpha / beta**2, rel=1e-12)


@pytest.fixture(scope="module")
def loop_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase10_nonlinear_qsvt")
    return run_phase10_nonlinear_qsvt_loop({"output_dir": str(output_dir)})


def test_solvers_present_and_qsvt_tracks_ridge(loop_run):
    summary = {row["solver"]: row for row in loop_run["summary_rows"]}
    for name in (
        "pseudoinverse",
        "ridge",
        "truncated_svd",
        "qsvt_target_matrix_level_canonical_alpha",
        "qsvt_target_matrix_level_degree_aware_alpha",
        "qsvt_statevector_in_loop_degree_aware_alpha",
    ):
        assert name in summary, name
    # Matrix-level QSVT-target trajectory matches Ridge exactly (exact-filter step).
    canonical = summary["qsvt_target_matrix_level_canonical_alpha"]
    assert canonical["max_update_error_vs_ridge"] == pytest.approx(0.0, abs=1e-12)
    # Degree-aware bounded polynomial approximant tracks Ridge tightly.
    degree_aware = summary["qsvt_target_matrix_level_degree_aware_alpha"]
    assert degree_aware["max_approximant_rel_error_vs_ridge"] < 0.05
    # Full-rectangular statevector QSVT-in-loop tracks Ridge at the degree-aware alpha.
    statevector = summary["qsvt_statevector_in_loop_degree_aware_alpha"]
    assert statevector["max_update_error_vs_ridge"] < 0.05


def test_canonical_alpha_degree_limit_recorded_not_hidden(loop_run):
    # The canonical alpha=1e-4 is degree-limited for the bounded polynomial; the
    # per-iteration approximant error must be recorded large, not silently passed.
    summary = {row["solver"]: row for row in loop_run["summary_rows"]}
    canonical = summary["qsvt_target_matrix_level_canonical_alpha"]
    assert canonical["max_approximant_rel_error_vs_ridge"] > 0.05


def test_per_iteration_recomputes_beta_lambda_and_rebuilds(loop_run):
    rows = [
        row
        for row in loop_run["iteration_rows"]
        if row["solver"] == "qsvt_target_matrix_level_degree_aware_alpha"
    ]
    assert len(rows) >= 2
    for row in rows:
        assert "beta_k" in row and np.isfinite(row["beta_k"])
        assert "lambda_k" in row and np.isfinite(row["lambda_k"])
    # beta_k varies across iterations (Jacobian rebuilt each iteration).
    betas = [row["beta_k"] for row in rows]
    assert len(set(np.round(betas, 6))) >= 1


def test_repetition_ledger_counts_reload_per_iteration(loop_run):
    rows = loop_run["repetition_rows"]
    assert rows
    for row in rows:
        assert row["residual_reloaded_this_iteration"] is True
        assert row["jacobian_rebuilt_this_iteration"] is True


def test_outputs_and_claim_safe(loop_run):
    output_dir = loop_run["output_dir"]
    for name in (
        "nonlinear_qsvt_iteration_log.csv",
        "nonlinear_qsvt_summary.csv",
        "nonlinear_qsvt_vs_ridge.csv",
        "nonlinear_qsvt_resource_repetition.csv",
        "README.md",
        "manifest.json",
        "checksums.sha256",
        "command_log.txt",
    ):
        assert (output_dir / name).is_file(), name
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "phase10_nonlinear_qsvt_in_loop"


def test_config_matches_nonlinear_ac_ieee14():
    config = build_ieee14_config(101)
    assert config["system"]["case_name"] == "ieee14"
    assert config["system"]["iteration"]["max_iterations"] == 8
    assert config["scenario"]["missing_ratio"] == 0.1
    assert config["scenario"]["bad_data"]["ratio"] == 0.05
