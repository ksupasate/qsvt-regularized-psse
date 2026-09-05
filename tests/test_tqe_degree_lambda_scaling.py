"""Tests for Workstream A - degree / normalized-regularization / target-tolerance feasibility "
"map."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.tqe_extensions.common import (
    analytic_bound_c,
    extract_min_feasible_degree,
)
from robust_qsvt_se.tqe_extensions.degree_lambda_scaling import (
    STUDY_ID,
    run_degree_lambda_scaling,
)

# ---------------------------------------------------------------- boundedness factor C


def test_bound_c_interior_maximizer():
    # lambda = 0.01 -> s* = sqrt(0.01) = 0.1 in [1e-3, 1] -> C = 1/(2*sqrt(lambda)) = 5.
    rec = analytic_bound_c(0.01, 1e-3, 1.0, margin=1.0)
    assert rec.maximizer_location == "interior"
    assert rec.s_star == pytest.approx(0.1, rel=1e-12)
    assert rec.c_analytic == pytest.approx(5.0, rel=1e-9)
    assert rec.resulting_max_target_magnitude == pytest.approx(1.0, rel=1e-12)


def test_bound_c_right_endpoint():
    # lambda = 3 -> sqrt(3) > 1 -> maximizer at s_hi = 1 -> C = 1/(1+3) = 0.25.
    rec = analytic_bound_c(3.0, 1e-3, 1.0, margin=1.0)
    assert rec.maximizer_location == "right_endpoint"
    assert rec.c_analytic == pytest.approx(0.25, rel=1e-12)


def test_bound_c_left_endpoint():
    # sqrt(lambda) = 0.01 < s_lo = 0.1 -> maximizer at s_lo -> C = 0.1/(0.01 + 1e-4).
    rec = analytic_bound_c(1e-4, 0.1, 1.0, margin=1.0)
    assert rec.maximizer_location == "left_endpoint"
    assert rec.c_analytic == pytest.approx(0.1 / (0.1**2 + 1e-4), rel=1e-12)


def test_bound_c_margin_scales_selected_only():
    rec = analytic_bound_c(0.05, 1e-3, 1.0, margin=1.1)
    assert rec.c_selected == pytest.approx(1.1 * rec.c_analytic, rel=1e-12)
    assert rec.resulting_max_target_magnitude == pytest.approx(1.0 / 1.1, rel=1e-12)


def test_analytic_c_matches_fitted_numeric_c():
    # The fit's numeric interval maximum (bound_C) must equal margin * analytic interval maximum.
    lam, dmin, margin = 0.2, 5e-3, 1.05
    target = fit_codesigned_bounded_polynomial(
        beta=1.0, alpha=lam, domain_min=dmin, domain_max=1.0, degree=31, margin=margin
    )
    rec = analytic_bound_c(lam, dmin, 1.0, margin=margin)
    assert target.bound_C == pytest.approx(rec.c_selected, rel=1e-3)


# ---------------------------------------------------------------- d_min extraction


def _synthetic_grid() -> pd.DataFrame:
    rows = []
    # Two scopes; degrees 7,15,31; a monotone-then-unbounded fit pattern.
    patterns = {
        "s1": {7: (0.5, True), 15: (5e-4, True), 31: (2e6, False)},
        "s2": {7: (0.9, True), 15: (0.3, True), 31: (1e-3, True)},
    }
    for scope, per_deg in patterns.items():
        for degree, (fit, bounded) in per_deg.items():
            rows.append(
                {
                    "scope_id": scope,
                    "normalized_lambda": 1.0,
                    "degree": degree,
                    "uniform_fit_error": fit,
                    "boundedness_ok": bounded,
                    "phase_synthesis_status": "synthesized" if bounded else "not_attempted",
                }
            )
    return pd.DataFrame(rows)


def test_extract_min_feasible_degree_picks_smallest_bounded_fit():
    grid = _synthetic_grid()
    dmin = extract_min_feasible_degree(grid, [1e-3, 1e-2])
    s1_1e3 = dmin[(dmin.scope_id == "s1") & (dmin.epsilon_target == 1e-3)].iloc[0]
    assert s1_1e3.d_min_fit == 15  # d=15 fit 5e-4 <=1e-3; d=31 unbounded ignored
    s2_1e3 = dmin[(dmin.scope_id == "s2") & (dmin.epsilon_target == 1e-3)].iloc[0]
    assert s2_1e3.d_min_fit == 31
    s2_1e2 = dmin[(dmin.scope_id == "s2") & (dmin.epsilon_target == 1e-2)].iloc[0]
    assert s2_1e2.d_min_fit == 31  # 0.3 > 1e-2, only 31 qualifies


def test_extract_min_feasible_degree_none_when_all_infeasible():
    grid = _synthetic_grid()
    dmin = extract_min_feasible_degree(grid, [1e-9])
    assert dmin["d_min_fit"].isna().all()
    assert (dmin["d_min_fit_status"] == "no_tested_degree_feasible").all()


def test_unbounded_high_degree_never_counts_as_feasible():
    grid = _synthetic_grid()
    dmin = extract_min_feasible_degree(grid, [1e-2])
    # s1 d=31 has fit 2e6 and is unbounded; must never be selected.
    assert (dmin[dmin.scope_id == "s1"]["d_min_fit"] != 31).all()


# ---------------------------------------------------------------- orchestrator (tiny grid)


def _tiny_config() -> dict:
    return {
        "study_id": STUDY_ID,
        "matrix_seed": 123,
        "normalized_lambdas": [1.0, 3.0e-4],  # one feasible, one stiff/infeasible
        "target_tolerances": [1e-2, 1e-4],
        "degrees": [7, 15, 63],  # 63 forces the unbounded/divergent failure branch
        "target_margin": 1.05,
        "bound_tolerance": 0.002,
        "fit_grid_size": 2000,
        "attempt_phase_synthesis": False,  # keep the orchestrator test fast + deterministic
        "phase_synthesis_degree_ceiling": 31,
        "scopes": [
            {
                "scope_id": "scalar_validation",
                "label": "scalar",
                "kind": "scalar",
                "domain_min": 1e-3,
            },
        ],
        "statevector_executions": [],
    }


def _run_tiny(tmp_path, config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import json

    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "out"
    summary = run_degree_lambda_scaling(cfg_path, out)
    grid = pd.read_csv(out / "raw_grid.csv")
    dmin = pd.read_csv(out / "minimum_feasible_degree.csv")
    claim = json.loads((out / "claim_support.json").read_text())
    return grid, dmin, claim, summary, out


def test_orchestrator_produces_all_declared_rows(tmp_path):
    grid, _dmin, claim, summary, _ = _run_tiny(tmp_path, _tiny_config())
    assert len(grid) == 1 * 2 * 3  # scopes x lambda x degrees
    assert claim["all_declared_rows_present"] is True
    assert claim["grid_frozen_before_evaluation"] is True
    assert summary["rows"] == 6


def test_even_degree_is_rejected(tmp_path):
    config = _tiny_config()
    config["degrees"] = [7, 8]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="odd"):
        run_degree_lambda_scaling(cfg_path, tmp_path / "out")


def test_failure_rows_retained(tmp_path):
    grid, _dmin, _claim, _, out = _run_tiny(tmp_path, _tiny_config())
    # degree 63 diverges/unbounded -> a retained failure with an explicit code, not dropped.
    failures = pd.read_csv(out / "failure_registry.csv")
    assert len(failures) >= 1
    assert (grid["degree"] == 63).sum() == 2  # both lambdas keep their d=63 row
    assert "unbounded" in set(grid.loc[grid.degree == 63, "failure_code"])


def test_boundedness_column_present_and_separate_from_fit(tmp_path):
    grid, *_ = _run_tiny(tmp_path, _tiny_config())
    assert "uniform_fit_error" in grid.columns
    assert "singular_point_fit_error" in grid.columns  # separate column, NaN for scalar
    assert grid.loc[grid.scope_kind == "scalar", "singular_point_fit_error"].isna().all()
    assert "boundedness_ok" in grid.columns


def test_grid_is_reproducible(tmp_path):
    import hashlib

    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    _, _, _, _, out1 = _run_tiny(tmp_path / "a", _tiny_config())
    _, _, _, _, out2 = _run_tiny(tmp_path / "b", _tiny_config())
    assert _hash(out1 / "raw_grid.csv") == _hash(out2 / "raw_grid.csv")


def test_reports_regenerate_from_raw_rows(tmp_path):
    _, _, _, _, out = _run_tiny(tmp_path, _tiny_config())
    for name in (
        "raw_grid.csv",
        "minimum_feasible_degree.csv",
        "failure_registry.csv",
        "matrix_action_results.csv",
        "resource_summary.csv",
        "claim_support.json",
        "run_manifest.json",
        "config_resolved.yaml",
        "README.md",
        "checksums.sha256",
    ):
        assert (out / name).exists(), name
