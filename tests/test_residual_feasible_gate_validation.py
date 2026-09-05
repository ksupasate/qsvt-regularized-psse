from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt import residual_feasible_gate_validation as rfgv
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem

MODULE = "robust_qsvt_se.qsvt.residual_feasible_gate_validation"


def _all_config_frame() -> pd.DataFrame:
    # Two deployable-protocol rows (different degrees) plus a diagnostic row that must be
    # excluded by the fallback selection, and a matrix-free row that is never gate-validated.
    return pd.DataFrame(
        [
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "high_leverage_01",
                "alpha": 1.0e-4,
                "degree": 35,
                "target_design": "current_global",
                "scale_protocol": "success_amplitude_proxy",
                "condition_number": 7.6,
                "residual_ratio_vs_no_update": 0.40,
                "direction_error_vs_ridge": 0.30,
                "residual_feasible": False,
                "matrix_shape": "4x4",
            },
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "residual_supported_03",
                "alpha": 1.0e-6,
                "degree": 51,
                "target_design": "margin_1p10",
                "scale_protocol": "known_C",
                "condition_number": 8.7,
                "residual_ratio_vs_no_update": 0.55,
                "direction_error_vs_ridge": 0.49,
                "residual_feasible": False,
                "matrix_shape": "4x4",
            },
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "high_leverage_01",
                "alpha": 1.0e-4,
                "degree": 35,
                "target_design": "current_global",
                "scale_protocol": "best_scalar_diagnostic",
                "condition_number": 7.6,
                "residual_ratio_vs_no_update": 0.02,
                "direction_error_vs_ridge": 0.05,
                "residual_feasible": False,
                "matrix_shape": "4x4",
            },
            {
                "case": "ieee30",
                "model": "ac_linearized",
                "subproblem_id": "matrix_free_ieee30",
                "alpha": 1.0e-4,
                "degree": 35,
                "target_design": "current_global",
                "scale_protocol": "matrix_free_polynomial_residual",
                "condition_number": 50.0,
                "residual_ratio_vs_no_update": 0.01,
                "direction_error_vs_ridge": 0.01,
                "residual_feasible": False,
                "matrix_shape": "4x4",
            },
        ]
    )


def _tiny_subproblem() -> SelectedSubproblem:
    return SelectedSubproblem(
        H_tilde=np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float64),
        r_tilde=np.array([1.0, 0.25], dtype=np.float64),
        metadata={"case": "toy"},
    )


def _fake_computation(H: np.ndarray, r: np.ndarray) -> types.SimpleNamespace:
    # A deliberately mis-scaled gate output: correct direction but wrong magnitude.
    ridge = np.array([0.5, 0.5], dtype=np.float64)
    gate = 3.0 * ridge
    return types.SimpleNamespace(
        H_tilde=H,
        r_tilde=r,
        qsvt_update=gate,
        ridge_update=ridge,
        summary={
            "synthesized_degree": 35,
            "phase_count": 36,
            "qsvt_query_count": 35,
            "success_probability": 0.5,
            "circuit_depth": 72,
            "two_qubit_gate_count": 0,
        },
    )


def test_fallback_selection_reads_from_files_and_excludes_diagnostics() -> None:
    selected, used_fallback = rfgv.select_gate_validation_configs(
        pd.DataFrame(),  # empty feasible set
        _all_config_frame(),
        max_configs=3,
        fallback=True,
    )
    assert used_fallback is True
    # best_scalar_diagnostic and matrix_free rows must never be gate-validated.
    assert "best_scalar_diagnostic" not in set(selected["scale_protocol"])
    assert "matrix_free_polynomial_residual" not in set(selected["scale_protocol"])
    # Deduplicated to distinct (subproblem, alpha, degree); ranked by residual ratio ascending.
    assert len(selected) == 2
    assert float(selected.iloc[0]["residual_ratio_vs_no_update"]) <= float(
        selected.iloc[1]["residual_ratio_vs_no_update"]
    )


def test_selection_uses_only_file_columns_not_gate_performance() -> None:
    # Feasibility comes from the file; the selector must not call the gate solver.
    feasible = _all_config_frame().head(1).copy()
    feasible["residual_feasible"] = True
    selected, used_fallback = rfgv.select_gate_validation_configs(
        feasible, pd.DataFrame(), max_configs=3, fallback=True
    )
    assert used_fallback is False
    assert len(selected) == 1


def test_run_writes_required_columns_and_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        f"{MODULE}.extract_state_estimation_subproblem",
        lambda **_: _tiny_subproblem(),
    )
    monkeypatch.setattr(
        f"{MODULE}.solve_gate_level_state_estimation_problem",
        lambda **kwargs: _fake_computation(kwargs["H_tilde"], kwargs["r_tilde"]),
    )
    fallback_path = tmp_path / "all_config_results.csv"
    _all_config_frame().to_csv(fallback_path, index=False)
    empty_path = tmp_path / "residual_feasible_configs.csv"
    empty_path.write_text(",".join(_all_config_frame().columns) + "\n", encoding="utf-8")

    run = rfgv.run_residual_feasible_gate_validation(
        {
            "input": str(empty_path),
            "fallback_input": str(fallback_path),
            "max_configs": 3,
            "shots": 64,
            "seed": 123,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert run["used_fallback"] is True
    results = pd.read_csv(run["artifacts"]["gate_validation_results"])
    for column in rfgv.GATE_VALIDATION_COLUMNS:
        assert column in results.columns
    assert not results.empty
    assert set(results["validation_status"]).issubset(
        {"residual_feasible_preserved", "exercised_not_feasible", "feasible_lost_at_gate", "failed"}
    )
    # The mis-scaled gate output (3x ridge magnitude) is not Ridge-feasible under a deployable
    # protocol, so it must be flagged exercised_not_feasible rather than preserved.
    assert "exercised_not_feasible" in set(results["validation_status"])


def test_apply_scale_protocol_best_scalar_minimizes_residual() -> None:
    H = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float64)
    r = np.array([1.0, 0.25], dtype=np.float64)
    gate = np.array([2.0, 1.0], dtype=np.float64)
    ridge = np.array([1.0, 0.5], dtype=np.float64)
    scaled, _scalar = rfgv.apply_scale_protocol(
        "best_scalar_diagnostic",
        gate_update=gate,
        H=H,
        r=r,
        ridge_update=ridge,
        success_probability=0.5,
    )
    raw_residual = float(np.linalg.norm(H @ gate - r))
    scaled_residual = float(np.linalg.norm(H @ scaled - r))
    assert scaled_residual <= raw_residual + 1.0e-12
    # known_C / proxy protocols leave the physical gate output unchanged.
    physical, factor = rfgv.apply_scale_protocol(
        "known_C", gate_update=gate, H=H, r=r, ridge_update=ridge, success_probability=0.5
    )
    assert factor == pytest.approx(1.0)
    assert np.allclose(physical, gate)


def test_empty_inputs_produce_no_feasible_conclusion(tmp_path: Path) -> None:
    run = rfgv.run_residual_feasible_gate_validation(
        {
            "input": str(tmp_path / "missing.csv"),
            "fallback_input": str(tmp_path / "also_missing.csv"),
            "max_configs": 3,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert run["rows"] == []
    interpretation = run["artifacts"]["gate_validation_interpretation"].read_text(encoding="utf-8")
    assert "do not produce" in interpretation
