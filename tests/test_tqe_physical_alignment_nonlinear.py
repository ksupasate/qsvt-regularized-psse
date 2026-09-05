from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.iterative_ac import build_ac_nonlinear_problem
from robust_qsvt_se.physical_alignment.config import load_campaign_config
from robust_qsvt_se.physical_alignment.nonlinear_ac import (
    EVIDENCE_FAILED,
    EVIDENCE_MODELED,
    _problem_perturbation_record,
    _statevector_boundary_rows,
    build_problem_config,
    nonlinear_functionals,
    run_nonlinear_campaign,
)


def _settings_and_scenario() -> tuple[dict, dict]:
    config = load_campaign_config("configs/tqe_physical_alignment/campaign.json")
    settings = config["nonlinear_ac"]
    scenario = next(
        row for row in settings["scenarios"] if row["scenario_id"] == "gaussian_noise_baseline"
    )
    return settings, scenario


def test_raw_measurement_problem_and_physical_functionals_are_valid() -> None:
    settings, scenario = _settings_and_scenario()
    problem = build_ac_nonlinear_problem(build_problem_config(settings, scenario, 101))
    perturbation = _problem_perturbation_record(problem)
    assert perturbation["raw_measurement_perturbation_path_used"]
    assert perturbation["measurement_model"] == "z=h(x_true)+e+b"
    assert perturbation["measurement_covariance_model"] == "diagonal implicit R_ii=sigma_i^2"
    assert perturbation["measurement_perturbation_l2_norm"] > 0.0
    functionals = nonlinear_functionals(problem)
    families = {row["functional_family"] for row in functionals}
    assert "coordinate_angle_update" in families
    assert "coordinate_voltage_magnitude_update" in families
    assert "real_branch_angle_difference" in families
    assert "connected_area_angle_aggregate" in families
    assert "connected_area_voltage_aggregate" in families
    for row in functionals:
        np.testing.assert_allclose(np.linalg.norm(row["vector"]), 1.0, atol=1.0e-12)


def test_matched_qsvt_target_and_ridge_nonlinear_loops_agree(tmp_path: Path) -> None:
    validation = run_nonlinear_campaign(
        output_dir=tmp_path,
        scenario_subset=["gaussian_noise_baseline"],
        seed_subset=[101],
        solver_subset=["ridge_fixed_alpha", "qsvt_target_exact_fixed_alpha"],
        run_statevector_boundary=False,
        verbose=False,
    )
    raw = pd.read_csv(tmp_path / "raw_runs.csv")
    iterations = pd.read_csv(tmp_path / "iteration_rows.csv")
    equivalence = pd.read_csv(tmp_path / "qsvt_ridge_equivalence.csv")
    selected = pd.read_csv(tmp_path / "selected_output_rows.csv")
    assert validation["qsvt_target_ridge_all_pass"]
    assert validation["qsvt_target_ridge_max_relative_error"] == 0.0
    assert raw["raw_measurement_perturbation_path_used"].all()
    assert iterations["jacobian_rebuilt_this_iteration"].all()
    assert iterations["residual_rebuilt_from_raw_measurements_this_iteration"].all()
    assert iterations["jacobian_fingerprint"].nunique() > 1
    assert equivalence["status"].eq("pass").all()
    np.testing.assert_allclose(
        equivalence["matched_qsvt_ridge_update_relative_error"], 0.0, atol=1.0e-15
    )
    ridge = raw.loc[raw["solver"].eq("ridge_fixed_alpha")].iloc[0]
    target = raw.loc[raw["solver"].eq("qsvt_target_exact_fixed_alpha")].iloc[0]
    assert ridge["converged"] == target["converged"]
    assert ridge["iteration_count"] == target["iteration_count"]
    assert ridge["final_full_state_rmse"] == target["final_full_state_rmse"]
    assert ridge["final_weighted_residual"] == target["final_weighted_residual"]
    assert not selected["logical_key"].duplicated().any()
    assert selected["functional_norm"].sub(1.0).abs().max() <= 1.0e-12
    assert {"final_full_state_rmse", "final_weighted_residual"}.issubset(raw.columns)


def test_statevector_failures_and_unexecuted_cases_have_distinct_statuses(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_campaign_config("configs/tqe_physical_alignment/campaign.json")
    settings = config["nonlinear_ac"]
    scenario = next(
        row for row in settings["scenarios"] if row["scenario_id"] == "gaussian_noise_baseline"
    )

    def fake_execution(*args, **kwargs):
        return {"status": "phase_synthesis_failed", "failure_reason": "controlled test failure"}

    monkeypatch.setattr(
        "robust_qsvt_se.physical_alignment.nonlinear_ac.run_full_rectangular_qsvt",
        fake_execution,
    )
    rows, equivalence = _statevector_boundary_rows(
        config, settings, [scenario], [101, 202], tmp_path
    )
    statuses = {row["evidence_status"] for row in rows}
    assert EVIDENCE_FAILED in statuses
    assert EVIDENCE_MODELED in statuses
    failed = [row for row in rows if row["evidence_status"] == EVIDENCE_FAILED]
    assert failed
    assert all(row["failure_reason"] == "controlled test failure" for row in failed)
    assert equivalence == []
