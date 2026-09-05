from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.ridge_output_certificate import (
    compute_ridge_selected_output_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_certificate_formula_is_unchanged_and_does_not_accept_actual_error() -> None:
    config = json.loads((ROOT / "configs/output_aware_generalization.json").read_text())
    assert config["certificate"]["formula_version"].endswith("forward_reverse_min_v1")
    matrix = np.array([[2.0, 0.5], [0.0, 1.0], [1.0, -0.2]])
    sparse = matrix.copy()
    sparse[0, 1] = 0.0
    certificate = compute_ridge_selected_output_certificate(
        matrix, sparse, np.array([1.0, -0.2, 0.4]), np.array([1.0, -1.0]), 0.3
    )
    assert certificate.actual_selected_output_error is None
    assert certificate.certificate_holds is None
    assert certificate.selected_output_bound >= 0.0
    assert certificate.operator_bound_forward >= 0.0
    assert certificate.operator_bound_reverse >= 0.0




def test_certificate_case_summary_has_zero_violations() -> None:
    summary = pd.read_csv(OUT / "certificate_case_summary.csv")
    by_case = summary[summary["summary_dimension"] == "ieee_case"]
    assert set(by_case["ieee_case"]) == {"ieee14", "ieee30", "ieee57"}
    assert (by_case["coverage"] == 1.0).all()
    assert (by_case["violations"] == 0).all()

