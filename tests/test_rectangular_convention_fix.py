from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import Chebyshev, Polynomial

from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.rectangular_convention import (
    extract_component,
    pcphase_qsvt_top_block,
    production_scalar_emulator_unitary,
    pyqsp_pcphase_component,
    pyqsp_sym_qsp_to_pcphase_phases,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import synthesize_pyqsp_sym_qsp_phases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rectangular_convention_fix"


def _json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_target_configuration_fingerprint():
    payload = _json("convention_target_configuration.json")
    digest = hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    recorded = (OUT / "convention_target_configuration.sha256").read_text(encoding="utf-8")
    assert payload["parent_fingerprint"] == (
        "f2911a84b20204d4b87ce646239365be1aca0d9c44ff8d7d8d6ac13f643c57e3"
    )
    assert payload["degree"] == 255
    assert payload["production_convention"].count("global_plus_pi_over_2") == 1
    assert digest in recorded


def test_production_scalar_emulator_matches_circuit():
    rows = _rows("production_scalar_emulator_validation.csv")
    assert rows
    assert all(row["status"] == "pass" for row in rows)
    assert max(float(row["max_abs_operator_error"]) for row in rows) <= 1.0e-12


def test_identity_rectangular_qsvt():
    rows = [
        row
        for row in _rows("identity_polynomial_rectangular_sweep.csv")
        if row["variant"] == "pyqsp_global_plus_pi_over_2_phase_first_U_then_Udag"
    ]
    assert rows
    assert all(row["status"] == "pass" for row in rows)
    assert max(float(row["relative_spectral_error"]) for row in rows) <= 1.0e-10


def test_cubic_rectangular_qsvt():
    rows = [
        row
        for row in _rows("cubic_polynomial_rectangular_sweep.csv")
        if row["variant"] == "pyqsp_global_plus_pi_over_2_phase_first_U_then_Udag"
    ]
    assert rows
    assert all(row["status"] == "pass" for row in rows)
    assert max(float(row["relative_spectral_error"]) for row in rows) <= 1.0e-9


def test_pyqsp_production_phase_mapping():
    rows = _rows("pyqsp_to_production_mapping_sweep.csv")
    accepted = [row for row in rows if row["status"] == "accepted"]
    assert any(row["candidate"] == "global_plus_pi_over_2_signed_imag" for row in accepted)
    best = next(row for row in accepted if row["candidate"] == "global_plus_pi_over_2_signed_imag")
    assert float(best["scalar_max_error"]) <= 1.0e-10
    assert float(best["small_rectangular_max_error"]) <= 1.0e-8


def test_low_degree_ridge_rectangular():
    rows = _rows("low_degree_ridge_rectangular_validation.csv")
    assert {int(row["degree"]) for row in rows} == {7, 15, 31}
    assert all(row["status"] == "pass" for row in rows)
    degree31 = next(row for row in rows if row["degree"] == "31")
    assert float(degree31["diagnostic_contraction"]) < 1.0
    assert float(degree31["physical_recovery_scale_multiplier"]) > 1.0


def test_rectangular_projector_derivation():
    rows = _rows("rectangular_qsvt_derivation_checks.csv")
    assert rows
    assert all(row["status"] == "pass" for row in rows)
    text = (OUT / "rectangular_qsvt_derivation.md").read_text(encoding="utf-8")
    assert "signed imaginary top-left block" in text


def test_rectangular_adapter_on_scalar_monomials():
    for power_coeffs in (np.array([0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])):
        degree = power_coeffs.size - 1
        cheb_coeffs = Polynomial(power_coeffs).convert(kind=Chebyshev).coef
        phases = pyqsp_sym_qsp_to_pcphase_phases(synthesize_pyqsp_sym_qsp_phases(cheb_coeffs))
        component = pyqsp_pcphase_component(degree)
        poly = Polynomial(power_coeffs)
        errors = []
        for x in np.linspace(-0.9, 0.9, 31):
            response = extract_component(
                production_scalar_emulator_unitary(float(x), phases)[:1, :1],
                component,
            )[0, 0]
            errors.append(abs(response - poly(x)))
        assert max(errors) <= 1.0e-10


def test_small_rectangular_regression():
    rows = _rows("small_rectangular_regression_suite.csv")
    assert len(rows) == 125
    assert all(row["status"] == "pass" for row in rows)
    assert any(float(row["absolute_spectral_error"]) <= 1.0e-12 for row in rows)


def test_corrected_exact_svd_ieee14():
    row = _rows("corrected_exact_svd_ieee14_validation.csv")[0]
    assert row["status"] == "pass"
    assert float(row["selected_relative_error_vs_ridge"]) <= 1.0e-3
    assert float(row["target_quadrature_probability"]) == pytest.approx(0.0028555738108420915)


def test_corrected_production_ieee14_statevector():
    row = _rows("corrected_production_ieee14_statevector.csv")[0]
    assert row["status"] == "pass"
    assert row["component"] == "imag"
    assert float(row["production_vs_exact_svd_block_relative_error"]) <= 1.0e-8
    assert float(row["selected_relative_error_vs_ridge"]) <= 1.0e-3


def test_corrected_backend_execution_not_distribution_sampling():
    row = _rows("corrected_ieee14_backend_summary.csv")[0]
    assert row["executed_backend_shots"] == "True"
    assert row["distribution_monte_carlo_used"] == "False"
    assert row["evidence_label"] == "EXECUTED_BACKEND_SHOTS"
    assert row["ci_contains_statevector"] == "True"
    assert int(row["total_hadamard_shots"]) == 1_031_000


def test_corrected_postselection_statistics():
    row = _rows("corrected_ieee14_backend_summary.csv")[0]
    assert int(row["total_accepted_samples"]) > 0
    assert float(row["empirical_postselection_rate"]) > 0.0
    assert float(row["aggregate_confidence_interval_half_width"]) > 0.0


def test_corrected_postselection_mitigation():
    rows = {row["method"]: row for row in _rows("corrected_postselection_mitigation.csv")}
    assert rows["direct_rejection_sampling"]["evidence_label"] == "EXECUTED_BACKEND_SHOTS"
    assert rows["oblivious_amplitude_amplification"]["evidence_label"] == "MODELED_RESOURCE"
    assert rows["fixed_point_amplitude_amplification"]["evidence_label"] == "EXCLUDED"


def test_final_rectangular_decision():
    decision = _json("final_rectangular_decision.json")
    assert decision["decision"] == "FULL_USEFUL_OVERLAP_EXECUTED"
    assert decision["stage_summary"]["production"]["passed"] is True
    assert decision["stage_summary"]["backend"]["ci_contains_statevector"] is True


def test_manual_rectangular_block_extraction_smoke():
    # A direct tiny case guards the adapter against swapping real/imaginary blocks.
    A = np.array([[0.2], [0.7]], dtype=np.float64)
    norm = float(np.linalg.svd(A, compute_uv=False)[0])
    A = A / norm * 0.8
    coeffs = Polynomial([0.0, 0.0, 0.0, 1.0]).convert(kind=Chebyshev).coef
    phases = pyqsp_sym_qsp_to_pcphase_phases(synthesize_pyqsp_sym_qsp_phases(coeffs))
    component = pyqsp_pcphase_component(3)
    padded = np.zeros((2, 2), dtype=np.float64)
    padded[:2, :1] = A
    encoding = canonical_square_block_encoding(padded, tolerance=1.0e-8)
    block = extract_component(
        pcphase_qsvt_top_block(encoding.unitary, phases, encoded_dimension=2),
        component,
    )[:2, :1]
    U, singulars, Vt = np.linalg.svd(A, full_matrices=False)
    expected = U @ np.diag(singulars**3) @ Vt
    assert np.linalg.norm(block - expected, ord=2) <= 1.0e-10
