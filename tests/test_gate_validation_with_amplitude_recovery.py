from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt import gate_validation_with_amplitude_recovery as gvar
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem

MODULE = "robust_qsvt_se.qsvt.gate_validation_with_amplitude_recovery"


def _feasible_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "ieee14_ac_high_leverage_4x4",
                "alpha": 1.0e-4,
                "degree": 5,
                "target_design": "current_global",
                "norm_recovery_method": "bernoulli_success_amplitude",
                "shots": 1000,
                "condition_number": 7.6,
                "residual_ratio_vs_no_update": 0.05,
                "direction_error_vs_ridge": 0.04,
                "matrix_shape": "2x2",
            }
        ]
    )


def _diagnostic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "ieee14_ac_high_leverage_4x4",
                "alpha": 1.0e-4,
                "degree": 5,
                "target_design": "current_global",
                "norm_recovery_method": "best_scalar_upper_bound",
                "shots": 1000,
                "condition_number": 7.6,
                "residual_ratio_vs_no_update": 0.02,
                "direction_error_vs_ridge": 0.03,
                "matrix_shape": "2x2",
            }
        ]
    )


def _tiny_subproblem(**_: object) -> SelectedSubproblem:
    return SelectedSubproblem(
        H_tilde=np.array([[1.0, 0.2], [0.15, 0.8]], dtype=np.float64),
        r_tilde=np.array([0.4, -0.2], dtype=np.float64),
        metadata={"case": "ieee14"},
    )


def _fake_computation(**kwargs: object) -> types.SimpleNamespace:
    H = np.asarray(kwargs["H_tilde"], dtype=np.float64)
    r = np.asarray(kwargs["r_tilde"], dtype=np.float64)
    ridge = np.linalg.solve(H.T @ H + 1.0e-4 * np.eye(2), H.T @ r)
    gate = 2.5 * ridge  # mis-scaled gate output (correct direction, wrong magnitude)
    return types.SimpleNamespace(
        qsvt_update=gate,
        ridge_update=ridge,
        summary={
            "success_probability": 0.4,
            "circuit_depth": 64,
            "two_qubit_gate_count": 0,
        },
    )


def test_selection_prefers_feasible_then_diagnostic_then_closest() -> None:
    selected, source = gvar.select_amplitude_gate_configs(
        _feasible_frame(), _diagnostic_frame(), pd.DataFrame(), max_configs=3
    )
    assert source == "residual_feasible"
    assert len(selected) == 1

    selected, source = gvar.select_amplitude_gate_configs(
        pd.DataFrame(), _diagnostic_frame(), pd.DataFrame(), max_configs=3
    )
    assert source == "diagnostic_feasible"

    selected, source = gvar.select_amplitude_gate_configs(
        pd.DataFrame(), pd.DataFrame(), _feasible_frame(), max_configs=3
    )
    assert source == "closest_rejected"


def test_run_reads_configs_and_writes_columns(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{MODULE}.extract_state_estimation_subproblem", _tiny_subproblem)
    monkeypatch.setattr(f"{MODULE}.solve_gate_level_state_estimation_problem", _fake_computation)

    feasible_path = tmp_path / "residual_feasible_configs.csv"
    feasible_path.write_text(",".join(_feasible_frame().columns) + "\n", encoding="utf-8")
    diagnostic_path = tmp_path / "diagnostic_feasible_configs.csv"
    _diagnostic_frame().to_csv(diagnostic_path, index=False)

    run = gvar.run_gate_validation_with_amplitude_recovery(
        {
            "input": str(feasible_path),
            "fallback_input": str(diagnostic_path),
            "closest_input": str(tmp_path / "missing.csv"),
            "max_configs": 3,
            "shots": 256,
            "seed": 123,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert run["source"] == "diagnostic_feasible"
    results = pd.read_csv(run["artifacts"]["gate_amplitude_recovery_results"])
    for column in gvar.GATE_AMPLITUDE_COLUMNS:
        assert column in results.columns
    assert set(results["gate_run_status"]).issubset({"completed", "failed"})
    # best_scalar_upper_bound is diagnostic-only, never deployable-feasible after the gate.
    assert bool(results.iloc[0]["residual_feasible_after_gate"]) is False


def test_empty_inputs_produce_no_validation(tmp_path: Path) -> None:
    run = gvar.run_gate_validation_with_amplitude_recovery(
        {
            "input": str(tmp_path / "a.csv"),
            "fallback_input": str(tmp_path / "b.csv"),
            "closest_input": str(tmp_path / "c.csv"),
            "max_configs": 3,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert run["source"] == "none"
    assert run["rows"] == []
    text = run["artifacts"]["gate_amplitude_recovery_interpretation"].read_text(encoding="utf-8")
    assert "No deployable amplitude-recovered configuration" in text
