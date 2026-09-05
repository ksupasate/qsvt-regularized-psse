from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.alpha_sensitivity_consolidation import (
    build_alpha_sensitivity_consolidation,
)


def _write_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "estimator": ["ridge", "qsvt_regularized", "pseudoinverse"],
            "sweep_parameter": ["scenario.noise_std"] * 3,
            "sweep_value": [0.01, 0.01, 0.01],
            "rmse_median": [0.1, 0.1, 0.2],
            "residual_norm_median": [0.5, 0.5, 0.6],
            "weighted_residual_norm_median": [0.3, 0.3, 0.4],
            "condition_number_median": [100.0, 100.0, 100.0],
        }
    ).to_csv(path, index=False)


def test_alpha_consolidation_never_fabricates_alpha_when_config_absent(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    # summary present but no config_resolved.yaml -> alpha must not be invented.
    _write_summary(input_root / "real_ieee14_seed10" / "summary_metrics.csv")
    run = build_alpha_sensitivity_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase3")}
    )
    table = pd.read_csv(run["artifacts"]["paper_table_alpha_sensitivity"])
    unresolved = table[table["alpha_resolved"] == "no"]
    assert not unresolved.empty
    # Every unresolved row leaves alpha blank rather than fabricating a value.
    assert unresolved["alpha"].isna().all() or (unresolved["alpha"].astype(str) == "").all()


def test_alpha_consolidation_records_missing_alpha_rows(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    _write_summary(input_root / "real_ieee14_seed10" / "summary_metrics.csv")
    run = build_alpha_sensitivity_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase3")}
    )
    missing = pd.read_csv(run["artifacts"]["missing_alpha_sensitivity_outputs"])
    assert not missing.empty
    assert run["rows_without_alpha"] >= 1


def test_alpha_selection_rule_marks_oracle_best_as_diagnostic_only(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    run = build_alpha_sensitivity_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase3")}
    )
    rules = pd.read_csv(run["artifacts"]["paper_table_alpha_selection_rule"])
    oracle = rules[rules["rule_name"] == "best_alpha_diagnostic_only"].iloc[0]
    assert oracle["uses_oracle_best_alpha"] == "yes"
    assert oracle["allowed_for_main_claim"] == "no"
