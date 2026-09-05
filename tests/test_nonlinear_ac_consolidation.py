from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.nonlinear_ac_consolidation import build_nonlinear_ac_consolidation


def test_nonlinear_summary_separates_raw_from_weighted_residual(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    run = build_nonlinear_ac_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase4")}
    )
    summary = Path(run["artifacts"]["nonlinear_ac_manuscript_summary"]).read_text(encoding="utf-8")
    # Raw nonlinear perturbation and iterative Jacobian rebuild.
    assert "z = h(x_{\\mathrm{true}}) + e + b" in summary
    assert "H_k =" in summary
    # Distinct single-step weighted-residual perturbation.
    assert "\\tilde r_{\\mathrm{perturbed}}" in summary


def test_nonlinear_consolidation_records_missing_instead_of_fabricating(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()  # no nonlinear_ac_* dirs -> no convergence data may be invented
    run = build_nonlinear_ac_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase4")}
    )
    convergence = pd.read_csv(run["artifacts"]["paper_table_nonlinear_ac_convergence"])
    assert convergence.empty
    missing = pd.read_csv(run["artifacts"]["missing_nonlinear_ac_outputs"])
    assert not missing.empty


def test_nonlinear_convergence_marks_raw_workflow(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    rel = input_root / "nonlinear_ac_ieee14_seed10"
    rel.mkdir(parents=True)
    pd.DataFrame(
        {
            "estimator": ["ridge", "ridge", "huber_irls", "huber_irls"],
            "sweep_name": ["nonlinear_noise_sweep"] * 4,
            "sweep_value": [0.0, 0.0, 0.0, 0.0],
            "noise_std": [0.01] * 4,
            "missing_ratio": [0.0] * 4,
            "bad_data_ratio": [0.0] * 4,
            "converged": [True, True, True, False],
            "iterations": [4, 5, 5, 8],
            "rmse": [0.1, 0.12, 0.09, 0.5],
            "angle_rmse": [0.05] * 4,
            "voltage_magnitude_rmse": [0.02] * 4,
            "residual_norm": [0.4] * 4,
            "weighted_residual": [0.3] * 4,
            "seed": [101] * 4,
        }
    ).to_csv(rel / "aggregate_metrics.csv", index=False)
    run = build_nonlinear_ac_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase4")}
    )
    convergence = pd.read_csv(run["artifacts"]["paper_table_nonlinear_ac_convergence"])
    assert not convergence.empty
    assert convergence["workflow"].str.contains("raw z=h").all()
