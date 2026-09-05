from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.output_aware_generalization import (
    validate_common_design_registry,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_qsvt_common_designs_cover_two_structural_groups_per_case() -> None:
    designs = json.loads((OUT / "qsvt_instance_designs.json").read_text())
    results = pd.read_csv(OUT / "qsvt_validation_results.csv")
    validate_common_design_registry(designs, list(designs))
    selected_groups = results.drop_duplicates("structural_group_id")
    assert (selected_groups.groupby("ieee_case").size() == 2).all()
    assert len(designs) == 6
    for design in designs.values():
        assert design["study_id"] == "output_aware_structural_generalization_v1"
        assert design["common_design_applies_to_all_supports"] is True
        assert len(design["support_subset"]) == 4
        assert not any(support["per_support_phase_refit"] for support in design["support_subset"])


def test_qsvt_outputs_use_identical_sparse_matrices_and_separate_errors() -> None:
    frame = pd.read_csv(OUT / "qsvt_validation_results.csv")
    assert len(frame) == 6 * 4 * 3
    assert (frame["status"] == "completed").all()
    assert frame["ridge_qsvt_identical_sparse_matrix"].astype(bool).all()
    assert frame["support_error_separate_from_qsvt_error"].astype(bool).all()
    assert (~frame["per_support_phase_refit"].astype(bool)).all()
    assert (frame["support_selection_error"] >= 0.0).all()
    assert (frame["qsvt_error_on_sparse_matrix"] >= 0.0).all()
    assert frame["postselection_probability"].between(0.0, 1.0, inclusive="right").all()


def test_every_instance_shares_one_common_design_across_supports() -> None:
    frame = pd.read_csv(OUT / "qsvt_validation_results.csv")
    for _instance, local in frame.groupby("instance_id"):
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
            assert local[column].nunique() == 1
