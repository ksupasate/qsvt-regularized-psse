from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.phase2_preconditioned_alpha import select_alpha_diagnostics


def test_alpha_selection_summary_has_required_criteria() -> None:
    results = pd.DataFrame(
        [
            _row(alpha=1.0e-4, residual=3.0, rmse=0.3, qsvt_error=0.1, query=403),
            _row(alpha=1.0e-2, residual=1.0, rmse=0.1, qsvt_error=0.01, query=403),
            _row(alpha=1.0, residual=2.0, rmse=0.2, qsvt_error=0.001, query=403),
        ]
    )
    summary, trace = select_alpha_diagnostics(results)

    assert {
        "residual_minimizing_alpha",
        "rmse_minimizing_alpha",
        "qsvt_resource_friendly_alpha",
        "joint_score_alpha",
    }.issubset(set(summary["selection_criterion"]))
    assert not trace.empty
    assert "joint_score" in trace.columns
    assert summary["caveat"].str.contains("Diagnostic alpha-selection").all()


def _row(*, alpha: float, residual: float, rmse: float, qsvt_error: float, query: int) -> dict:
    return {
        "case_name": "ieee118",
        "variant_name": "original_ridge",
        "alpha": alpha,
        "residual_norm": residual,
        "rmse_if_available": rmse,
        "qsvt_full_interval_approx_error": qsvt_error,
        "qsvt_actual_singular_value_error": qsvt_error,
        "qsvt_query_count": query,
        "qsvt_degree": 201,
        "status": "ok",
    }
