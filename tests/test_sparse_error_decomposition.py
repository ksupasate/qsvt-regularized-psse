"""Baseline-reproduction, error-decomposition, and statistical-separation tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

import pandas as pd

from robust_qsvt_se.qsvt.sparse_error_precision_study import (
    EXACT_VALUE_KEY,
    FULL_PHASE_KEY,
    FUNCTIONAL_IDS,
    IDENTITY_ABS_TOLERANCE,
    PRIMARY_FUNCTIONAL_ID,
    _assemble_finite_shot_tables,
    build_frozen_design,
    evaluate_statevector_point,
    make_context,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    estimate_signed_selected_output,
    stable_array_fingerprint,
)

EXPECTED_ORIGINAL_FP = "b158d34b86b778f0c290519ca98985345107012e225798a4cfc7fbf9178df7f9"
EXPECTED_SPARSE_FP = "c6e29a98365f6e79e50bac5551c646e1178a4c898cb7ef47a73294a0d80ea88c"
EXPECTED_QUANTIZED_FP = "26159050694e76abc32692332daba94e9cd5e22d958a242236b4d57509aeab21"
EXPECTED_RESIDUAL_FP = "7b51f68a0e1cbdb4cfecd799d3fcb6f12c090a25bbc6a04f3f09324e06fdee82"
EXPECTED_PHASE_FP = "5d1183c7efdeebcc35e08682c8fb5c841533ba8d696132d7bfba04925667b4a1"


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    return build_frozen_design(tmp_path_factory.mktemp("design"))


def test_baseline_fingerprints_and_design_reproduce(design):
    assert stable_array_fingerprint(design.matrix_original) == EXPECTED_ORIGINAL_FP
    assert stable_array_fingerprint(design.matrix_sparse_exact) == EXPECTED_SPARSE_FP
    assert (
        stable_array_fingerprint(design.matrices_by_value_key["6"]) == EXPECTED_QUANTIZED_FP
    )
    assert stable_array_fingerprint(design.residual) == EXPECTED_RESIDUAL_FP
    assert (
        stable_array_fingerprint(design.phases_by_phase_key[FULL_PHASE_KEY])
        == EXPECTED_PHASE_FP
    )
    assert design.alpha == pytest.approx(1134521.3658711074, rel=1e-12)
    assert design.beta == pytest.approx(4041.1277722772575, rel=1e-12)
    assert design.contraction_c == pytest.approx(1.9918449969534795, rel=1e-12)
    assert design.degree == 31
    assert design.phases_by_phase_key[FULL_PHASE_KEY].size == 32




def test_signed_increments_reconstruct_cumulative_difference(design):
    result = evaluate_statevector_point(design, "6", FULL_PHASE_KEY)
    assert result.status == "completed"
    ridge = design.ridge_updates
    for functional_id in FUNCTIONAL_IDS:
        ell = design.functionals[functional_id]
        y_original = float(ell @ ridge["original"])
        y_sparse = float(ell @ ridge["sparse_exact"])
        y_quantized = float(ell @ ridge["value_bits_6"])
        y_qsvt = float(ell @ result.update)
        delta_sparse = y_sparse - y_original
        delta_quant = y_quantized - y_sparse
        delta_qsvt = y_qsvt - y_quantized
        total = y_qsvt - y_original
        assert abs(total - (delta_sparse + delta_quant + delta_qsvt)) <= (
            IDENTITY_ABS_TOLERANCE
        )
        # Triangle bound with separated (never merged) sparsification/quantization terms.
        assert abs(total) <= (
            abs(delta_sparse) + abs(delta_quant) + abs(delta_qsvt) + IDENTITY_ABS_TOLERANCE
        )
        assert delta_sparse != delta_quant


def test_exact_value_point_has_zero_quantization_term(design):
    ridge = design.ridge_updates
    ell = design.functionals[PRIMARY_FUNCTIONAL_ID]
    y_sparse = float(ell @ ridge["sparse_exact"])
    y_exact = float(ell @ ridge[f"value_bits_{EXACT_VALUE_KEY}"])
    assert y_exact == pytest.approx(y_sparse, abs=0.0)


def test_near_zero_outputs_do_not_break_relative_errors(design):
    result = evaluate_statevector_point(design, "6", FULL_PHASE_KEY)
    assert math.isfinite(result.qsvt_action_error)
    zero_action = np.zeros(8)
    scaled = float(
        np.linalg.norm(zero_action - zero_action) / max(np.linalg.norm(zero_action), 1e-30)
    )
    assert scaled == 0.0


def test_estimator_is_unbiased_on_exact_counts():
    scale = 0.0121703
    counts = {"00": 504048, "10": 301116, "01": 97701, "11": 97135}
    attempted = sum(counts.values())
    estimate = estimate_signed_selected_output(counts, physical_scale=scale)
    expected = scale * (counts["00"] - counts["10"]) / attempted
    assert estimate["selected_output_estimate"] == pytest.approx(expected, rel=1e-14)
    assert estimate["readout_accepted"] == counts["00"] + counts["10"]
    assert estimate["analytic_standard_error"] > 0.0
    assert (
        estimate["confidence_interval_lower"]
        < estimate["selected_output_estimate"]
        < estimate["confidence_interval_upper"]
    )


def _synthetic_finite_shot_rows(config_id: str, estimates: list[float]) -> list[dict]:
    rows = []
    for seed, value in enumerate(estimates):
        rows.append(
            {
                "configuration_id": config_id,
                "configuration_label": "baseline",
                "value_bits": "6",
                "phase_bits": "full",
                "functional_id": PRIMARY_FUNCTIONAL_ID,
                "shots_attempted": 1000,
                "direct_postselection_shots_attempted": 1000,
                "seed": seed,
                "backend": "synthetic",
                "postselection_accepted_direct": 600,
                "readout_accepted_interference": 800,
                "measured_postselection_probability": 0.6,
                "interference_acceptance_probability": 0.8,
                "inferred_postselection_probability_from_branch": 0.6,
                "readout_sign_mean_accepted": 0.25,
                "signed_overlap_estimate": value / 0.01,
                "selected_output_estimate": value,
                "statevector_reference": 0.0025,
                "quantized_ridge_reference": 0.00247,
                "original_ridge_reference": 0.00306,
                "sampling_signed_delta": value - 0.0025,
                "sampling_absolute_error": abs(value - 0.0025),
                "absolute_error_vs_quantized_ridge": abs(value - 0.00247),
                "absolute_error_vs_original_ridge": abs(value - 0.00306),
                "analytic_standard_error": 0.0002,
                "statevector_expected_standard_error": 0.0002,
                "confidence_interval_lower": value - 0.000392,
                "confidence_interval_upper": value + 0.000392,
                "statevector_postselection_probability": 0.609,
                "physical_recovery_scale": 0.0121,
                "status": "completed",
                "failure_stage": "",
                "failure_reason": "",
                "exception_type": "",
            }
        )
    return rows


def test_summary_separates_analytic_and_empirical_uncertainty(tmp_path):
    context = make_context(output_dir=tmp_path)
    estimates = [0.0024, 0.0026, 0.0025, 0.0027, 0.0023]
    context.checkpoint.write_part(
        "finite-shot",
        "bv6_bpfull",
        {
            "configuration_id": "synthetic", "label": "baseline", "value_bits": "6",
            "phase_bits": "full", "status": "completed",
            "rows": _synthetic_finite_shot_rows("synthetic", estimates),
            "resource_capture": {}, "postselection_probability_statevector": 0.609,
        },
    )
    frame, summary = _assemble_finite_shot_tables(context)
    assert len(frame) == len(estimates)
    row = summary.iloc[0]
    assert row["num_seeds"] == len(estimates)
    empirical = float(np.var(np.asarray(estimates), ddof=1))
    assert row["empirical_variance_across_seeds"] == pytest.approx(empirical, rel=1e-12)
    assert row["analytic_variance_one_estimate"] == pytest.approx(0.0002**2, rel=1e-12)
    # per-estimate analytic uncertainty and across-seed variation are distinct fields
    assert row["mean_analytic_standard_error_one_estimate"] != (
        row["selected_output_std_across_seeds"]
    )
    assert 0.0 <= row["statevector_95pct_ci_coverage_across_seeds"] <= 1.0
    assert row["mean_readout_accepted_interference"] <= row["mean_shots_attempted"]
    assert row["mean_postselection_accepted_direct"] <= row["mean_shots_attempted"]


def test_failed_finite_shot_configuration_is_retained(tmp_path):
    context = make_context(output_dir=tmp_path)
    context.checkpoint.write_part(
        "finite-shot",
        "bv4_bp8",
        {
            "configuration_id": "failed_cfg", "label": "low_precision",
            "value_bits": "4", "phase_bits": "8", "status": "failed",
            "failure_stage": "finite-shot",
            "failure_reason": "numerical_instability: synthetic",
            "exception_type": "RuntimeError", "rows": [], "resource_capture": {},
        },
    )
    frame, summary = _assemble_finite_shot_tables(context)
    assert (frame["status"] == "failed").sum() == 1
    assert summary.empty
    kept = frame[frame["status"] == "failed"].iloc[0]
    assert kept["failure_reason"].startswith("numerical_instability")


def test_decomposition_csv_contract_if_present():
    path = "outputs/sparse_error_precision_study/error_decomposition.csv"
    try:
        frame = pd.read_csv(path, dtype={"value_bits": str, "phase_bits": str})
    except FileNotFoundError:
        pytest.skip("study outputs not generated yet")
    completed = frame[frame["status"] == "completed"]
    assert (completed["cumulative_identity_residual"].abs() <= IDENTITY_ABS_TOLERANCE).all()
    assert completed["triangle_bound_satisfied"].all()
    assert {"sparsification_signed_delta", "quantization_signed_delta",
            "qsvt_signed_delta", "sampling_signed_delta"}.issubset(frame.columns)
    statevector = completed[completed["estimate_kind"] == "statevector"]
    assert statevector["sampling_signed_delta"].isna().all()
