"""Tests for Workstream B - nonlinear AC QSVT circuit-in-the-loop."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from robust_qsvt_se.tqe_extensions.common import EVIDENCE_TIERS, load_yaml_config
from robust_qsvt_se.tqe_extensions.nonlinear_circuit_loop import (
    _magnitude_support,
    block_reference_outputs,
    build_block_qsvt,
    run_one_workload,
)

CONFIG = "configs/tqe_nonlinear_qsvt_circuit_loop.yaml"


def _sample_system(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(20, 12))
    residual = rng.normal(size=20)
    return matrix, residual


# ---------------------------------------------------------------- block / support


def test_magnitude_support_keeps_k_per_row():
    block = np.array([[3.0, 1.0, 2.0], [0.1, 0.9, 0.5]])
    mask = _magnitude_support(block, keep_per_row=2)
    assert mask.sum(axis=1).tolist() == [2, 2]
    assert mask[0].tolist() == [True, False, True]  # top-2 magnitudes of row 0


def test_build_block_qsvt_matched_alpha_beta_lambda():
    matrix, residual = _sample_system()
    bq = build_block_qsvt(
        matrix,
        residual,
        block_size=4,
        keep_per_row=2,
        lambda_target=0.1,
        degree=31,
        margin=1.05,
        functional_index=0,
    )
    assert bq.alpha_k == pytest.approx(0.1 * bq.beta**2, rel=1e-9)
    assert bq.lambda_k == pytest.approx(bq.alpha_k / bq.beta**2, rel=1e-6)
    assert bq.contraction_c > 0.0
    assert bq.sparse_block.shape == (4, 4)


def test_build_block_qsvt_is_deterministic():
    matrix, residual = _sample_system(3)
    a = build_block_qsvt(
        matrix,
        residual,
        block_size=4,
        keep_per_row=2,
        lambda_target=0.1,
        degree=31,
        margin=1.05,
        functional_index=0,
    )
    b = build_block_qsvt(
        matrix,
        residual,
        block_size=4,
        keep_per_row=2,
        lambda_target=0.1,
        degree=31,
        margin=1.05,
        functional_index=0,
    )
    assert np.array_equal(a.sparse_block, b.sparse_block)
    assert np.allclose(a.coefficients, b.coefficients)


def test_physical_rescaling_recovers_ridge_from_polynomial_action():
    # At a QSVT-feasible lambda the exact polynomial action (rescaled by C/beta) reproduces the
    # matched block Ridge selected output -> correct physical rescaling factor C/beta.
    matrix, residual = _sample_system(7)
    bq = build_block_qsvt(
        matrix,
        residual,
        block_size=4,
        keep_per_row=2,
        lambda_target=0.1,
        degree=31,
        margin=1.05,
        functional_index=0,
    )
    refs = block_reference_outputs(bq)
    assert refs["physical_recovery_factor"] == pytest.approx(bq.contraction_c / bq.beta, rel=1e-12)
    denom = max(abs(refs["y_block_ridge_exact_rational"]), 1e-6)
    rel = abs(refs["y_exact_polynomial_action"] - refs["y_block_ridge_exact_rational"]) / denom
    assert rel < 1e-2  # polynomial action matches the rational filter at feasible lambda


def test_build_block_qsvt_excludes_truth():
    params = set(inspect.signature(build_block_qsvt).parameters)
    # The operating point is built from (matrix, residual) only - no truth leaks into selection.
    assert "truth" not in params and "x_true" not in params


# ---------------------------------------------------------------- loop integration (small)


@pytest.fixture(scope="module")
def _small_run(tmp_path_factory, inprocess_phase_synthesis_guard):
    del inprocess_phase_synthesis_guard
    config = load_yaml_config(CONFIG)
    settings = dict(config["nonlinear_settings"])
    settings["iteration"] = dict(settings["iteration"])
    settings["iteration"]["max_iterations"] = 3
    wsb = dict(config["block_qsvt"])
    wsb["degree"] = 15  # cheaper synthesis for the test
    cache = tmp_path_factory.mktemp("nonlinear_circuit_loop") / "cache"
    rows, _shots = run_one_workload(
        config["scenarios"][0], 101, settings, wsb, cache, do_finite_shots=False
    )
    return rows


def test_jacobian_and_residual_rebuilt_every_iteration(_small_run):
    rows = _small_run
    assert all(r["jacobian_rebuilt_this_iteration"] for r in rows)
    assert all(r["residual_rebuilt_this_iteration"] for r in rows)
    # distinct Jacobian fingerprints once the state moves (no stale residual/Jacobian reuse)
    fps = [r["jacobian_fingerprint"] for r in rows]
    assert len(set(fps)) == len(fps)


def test_evidence_tiers_recorded_and_separated(_small_run):
    rows = _small_run
    for r in rows:
        # rational, polynomial, and (where executed) statevector selected outputs are separate
        # columns
        assert "y_block_ridge_exact_rational" in r
        assert "y_exact_polynomial_action" in r
        assert "y_circuit_statevector" in r
        assert r["evidence_tier"] in {
            EVIDENCE_TIERS["EXACT_MATRIX_ACTION"],
            EVIDENCE_TIERS["STATEVECTOR"],
        }


def test_statevector_reproduces_ridge_when_executed(_small_run):
    executed = [r for r in _small_run if r["evidence_tier"] == EVIDENCE_TIERS["STATEVECTOR"]]
    assert executed, "no statevector iteration executed"
    for r in executed:
        # The circuit faithfully implements the polynomial; the residual gap to matched Ridge is the
        # polynomial's own approximation error (matrix-action error), which the circuit does not add
        # to.
        assert r["circuit_vs_polynomial_error"] < 1e-4
        assert r["circuit_vs_ridge_error"] == pytest.approx(r["qsvt_matrix_action_error"], abs=5e-3)
        assert 0.0 <= r["postselection_probability_executed"] <= 1.0


def test_convergence_status_retained(_small_run):
    rows = _small_run
    assert all("convergence_status" in r for r in rows)
    assert all("run_converged" in r for r in rows)
