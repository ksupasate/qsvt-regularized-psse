from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.output_aware_generalization import (
    validate_common_design_registry,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_common_design_registry_matches_predeclared_instances() -> None:
    config = json.loads((ROOT / "configs/output_aware_generalization.json").read_text())
    designs = json.loads((OUT / "qsvt_instance_designs.json").read_text())
    validate_common_design_registry(designs, config["qsvt"]["predeclared_instance_ids"])
    for design in designs.values():
        accepted = [trial for trial in design["degree_trials"] if trial.get("accepted")]
        assert accepted[0]["degree"] == design["degree"]
        assert design["degree"] in config["qsvt"]["candidate_degrees"]
        assert design["fit_max_abs_error"] <= config["qsvt"][
            "uniform_approximation_tolerance"
        ]


def test_supports_within_instance_share_common_parameters_and_phases() -> None:
    results = pd.read_csv(OUT / "qsvt_validation_results.csv")
    for _instance_id, group in results.groupby("instance_id", sort=True):
        for column in (
            "common_design_fingerprint",
            "phase_fingerprint",
            "polynomial_fingerprint",
            "beta",
            "lambda",
            "C",
            "degree",
            "phase_count",
        ):
            assert group[column].nunique() == 1
        assert (~group["per_support_phase_refit"].astype(bool)).all()


def test_qsvt_uses_same_sparse_matrix_and_separates_errors() -> None:
    results = pd.read_csv(OUT / "qsvt_validation_results.csv")
    assert set(results["ieee_case"]) == {"ieee14", "ieee30", "ieee57"}
    assert (results["status"] == "completed").all()
    assert results["ridge_qsvt_identical_sparse_matrix"].astype(bool).all()
    assert results["support_error_separate_from_qsvt_error"].astype(bool).all()
    assert (results["qsvt_error_on_sparse_matrix"] >= 0.0).all()
    assert (results["support_selection_error"] >= 0.0).all()
    assert (results["postselection_probability"] > 0.0).all()

