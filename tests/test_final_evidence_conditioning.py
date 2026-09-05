import math
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.evidence.regularized_conditioning import (
    global_ridge_filter_bound,
    regularized_condition_number,
    ridge_solve_residual,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"


def test_conditioning_formulas_and_rank_thresholds() -> None:
    audit = pd.read_csv(OUT / "regularized_conditioning_audit.csv")
    assert audit["matrix_fingerprint_matches"].all()
    for row in audit.itertuples():
        assert row.numerical_rank <= min(map(int, row.matrix_shape.split("x")))
        if row.rank_deficient:
            assert math.isinf(row.raw_condition_number)
            assert row.sigma_min == 0.0
        expected = regularized_condition_number(row.sigma_max, row.sigma_min, row.alpha)
        assert math.isclose(row.regularized_condition_number, expected, rel_tol=1e-12)
        assert math.isclose(
            row.max_ridge_filter_response_global,
            global_ridge_filter_bound(row.alpha),
            rel_tol=1e-12,
        )
        assert row.max_ridge_filter_response_actual <= row.max_ridge_filter_response_global * (
            1 + 1e-12
        )


def test_ridge_solve_residual_definition() -> None:
    matrix = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]])
    residual = np.array([0.5, -0.25, 1.5])
    value = ridge_solve_residual(matrix, residual, alpha=0.7)
    assert value < 1e-14
