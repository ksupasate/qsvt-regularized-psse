from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "full_rectangular_breakthrough"


def _json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_target_configuration_fingerprint():
    payload = _json("target_configuration.json")
    digest = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    recorded = (OUT / "target_configuration.sha256").read_text(encoding="utf-8")
    assert payload["case"] == "ieee14"
    assert payload["degree"] == 255
    assert payload["lambda"] == pytest.approx(1.0e-5)
    assert digest in recorded


def test_polynomial_independent_evaluators():
    rows = {row["polynomial"]: row for row in _rows("polynomial_independent_validation.csv")}
    original = rows["original_rebuilt_scalar"]
    repaired = rows["minimal_contraction_active"]
    assert original["classification"] == "POLYNOMIAL_GLOBALLY_INVALID"
    assert float(original["power_basis_max_abs"]) > 1.0e70
    assert repaired["classification"] == "POLYNOMIAL_GLOBALLY_VALID"


def test_polynomial_global_boundedness():
    repaired = {row["polynomial"]: row for row in _rows("polynomial_independent_validation.csv")}[
        "minimal_contraction_active"
    ]
    assert float(repaired["max_abs"]) <= 1.0
    assert float(repaired["boundedness_margin"]) > 0.0


def test_polynomial_zero_and_parity():
    for row in _rows("polynomial_independent_validation.csv"):
        assert abs(float(row["p_zero"])) <= 1.0e-14
        assert float(row["parity_error"]) <= 1.0e-14


def test_polynomial_contraction_recovery():
    rows = {row["method"]: row for row in _rows("polynomial_repair_comparison.csv")}
    promoted = rows["minimal_contraction"]
    assert promoted["status"] == "PROMOTED"
    assert float(promoted["physical_reconstruction_error"]) <= 1.0e-12
    assert rows["bounded_chebyshev_lp_grid_constraints"]["status"] == "FAILED_DENSE_VALIDATION"


def test_phase_convention_reconstruction():
    decision = _json("final_breakthrough_decision.json")
    assert decision["phase_reconstruction"] == "PHASE_RECONSTRUCTION_VALID"
    for row in _rows("phase_reconstruction_validation.csv"):
        assert float(row["max_abs_error"]) <= 1.0e-10


def test_diagonal_singular_value_qsvt():
    decision = _json("final_breakthrough_decision.json")
    assert decision["diagonal_action"] == "DIAGONAL_ACTION_VALID"
    for row in _rows("diagonal_singular_value_action.csv"):
        assert float(row["max_abs_error"]) <= 1.0e-10


def test_padded_zero_modes():
    row = _rows("zero_mode_leakage.csv")[0]
    assert row["classification"] == "no_zero_mode_leakage"
    assert float(row["zero_response_max_abs"]) <= 1.0e-12


def test_small_rectangular_qsvt():
    decision = _json("final_breakthrough_decision.json")
    assert decision["small_rectangular_action"] == "LEFT_RIGHT_PROJECTOR_MISMATCH"
    assert all(
        row["classification"] == "LEFT_RIGHT_PROJECTOR_MISMATCH"
        for row in _rows("small_rectangular_action.csv")
    )


def test_left_right_projectors():
    rows = _rows("rectangular_projector_tests.csv")
    assert rows
    assert all(
        "PCPhase" in row["observed_issue"] or "pyqsp" in row["observed_issue"] for row in rows
    )


def test_exact_svd_full_rectangular():
    row = _rows("ieee14_exact_svd_selected_output.csv")[0]
    assert float(row["relative_error"]) <= 1.0e-2
    assert row["evidence_label"] == "DIAGNOSTIC_ONLY"


def test_production_block_encoding_against_svd():
    decision = _json("final_breakthrough_decision.json")
    assert decision["production_classification"] == "failed_convention_validation"
    assert float(decision["production_selected_relative_error"]) > 1.0
    assert all(
        row["evidence_label"] == "FAILED_CONFIGURATION"
        for row in _rows("production_vs_exact_svd.csv")
    )


def test_full_ieee14_useful_lambda_statevector():
    rows = {row["quantity"]: row for row in _rows("ieee14_full_statevector_validation.csv")}
    assert rows["exact_svd_polynomial"]["status"] == "reference_pass"
    assert rows["production_pcphase_best_component"]["status"] == "production_failed"


def test_backend_useful_lambda_not_distribution_sampling():
    row = _rows("ieee14_useful_lambda_backend_summary.csv")[0]
    assert row["executed_backend_shots"] == "False"
    assert row["distribution_monte_carlo_used"] == "False"
    assert row["evidence_label"] == "EXCLUDED"


def test_postselection_effective_samples():
    row = _rows("ieee14_useful_lambda_backend_runs.csv")[0]
    assert row["status"] == "blocked_statevector_failed"
    assert row["evidence_label"] == "EXCLUDED"


def test_postselection_mitigation():
    rows = _rows("postselection_mitigation_comparison.csv")
    assert rows
    assert all(row["status"] == "not_run_base_execution_failed" for row in rows)
    assert all(row["evidence_label"] == "EXCLUDED" for row in rows)
