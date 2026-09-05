"""Fairness and matrix-identity tests for the common-design QSVT track."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    build_common_padded_wrapper,
    build_frozen_output_aware_design,
    load_support,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint

OUTPUT = Path("outputs/output_aware_sparse_selection")


def _artifacts():
    design_path = OUTPUT / "common_qsvt_design.json"
    results_path = OUTPUT / "qsvt_validation_results.csv"
    if not design_path.is_file() or not results_path.is_file():
        pytest.skip("common-design QSVT campaign artifacts not generated yet")
    return (
        json.loads(design_path.read_text(encoding="utf-8")),
        pd.read_csv(results_path),
    )


def test_common_beta_lambda_c_degree_and_phases_are_shared():
    design, results = _artifacts()
    completed = results[results["status"] == "completed"]
    assert completed["common_design_fingerprint"].nunique() == 1
    assert completed["beta"].nunique() == 1
    assert completed["lambda"].nunique() == 1
    assert completed["C"].nunique() == 1
    assert completed["degree"].nunique() == 1
    assert completed["phase_fingerprint"].nunique() == 1
    assert completed["beta"].iloc[0] == pytest.approx(design["common_beta"])
    assert completed["lambda"].iloc[0] == pytest.approx(design["common_lambda"])
    assert completed["C"].iloc[0] == pytest.approx(design["common_C"])
    assert int(completed["degree"].iloc[0]) == int(design["degree"])
    assert len(design["phases"]) == design["degree"] + 1
    assert stable_array_fingerprint(np.asarray(design["phases"])) == design[
        "phase_fingerprint"
    ]


def test_primary_track_has_no_per_support_phase_refitting():
    design, results = _artifacts()
    assert design["phase_refit_policy"] == (
        "one_common_fit_and_one_common_phase_sequence_no_per_support_refitting"
    )
    assert design["degree_selected_before_support_specific_qsvt_outputs"] is True
    assert not results["per_support_phase_refit"].fillna(False).any()


def test_predeclared_subset_contains_required_roles_at_two_budgets():
    design, _results = _artifacts()
    subset = pd.DataFrame(design["support_subset_declared_by_training_policy"])
    assert set(subset["k_budget"]) == {16, 24}
    required = {
        "best_magnitude_training_only",
        "best_sensitivity_training_only",
        "best_refined_training_only",
        "random_replicate_zero_predeclared",
    }
    for _k, group in subset.groupby("k_budget"):
        assert set(group["role"]) == required
    assert (subset["slot_budget"] == 3).all()


def test_every_padded_support_reconstructs_under_the_same_normalization():
    design, _results = _artifacts()
    frozen = build_frozen_output_aware_design(OUTPUT)
    for selected in design["support_subset_declared_by_training_policy"]:
        support = load_support(OUTPUT, selected["support_file"])
        sparse = np.where(support, frozen.matrix, 0.0)
        wrapper = build_common_padded_wrapper(
            sparse,
            slots=int(design["common_slots"]),
            mu=float(design["common_mu"]),
        )
        assert wrapper.beta == pytest.approx(design["common_beta"])
        np.testing.assert_allclose(
            wrapper.encoded_block,
            sparse.T / design["common_beta"],
            atol=1.0e-9,
        )


def test_qsvt_is_compared_to_ridge_on_the_identical_sparse_matrix():
    _design, results = _artifacts()
    completed = results[results["status"] == "completed"]
    assert completed["ridge_and_qsvt_matrix_fingerprint"].str.len().eq(64).all()
    assert (completed["qsvt_action_error_vs_exact_polynomial"] < 1.0e-6).all()
    assert np.isfinite(completed["ridge_output_sparse_matrix"]).all()
    assert np.isfinite(completed["sparse_qsvt_statevector_output"]).all()
    assert completed["postselection_probability"].between(0.0, 1.0).all()
