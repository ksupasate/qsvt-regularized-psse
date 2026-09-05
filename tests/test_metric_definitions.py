from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.metric_definitions import build_metric_definitions


def test_update_rmse_distinct_from_full_state_rmse(tmp_path: Path) -> None:
    run = build_metric_definitions({"output_dir": str(tmp_path / "metrics")})
    table = pd.read_csv(run["artifacts"]["metric_definitions_table"]).set_index("metric_name")
    assert "single_step_update_rmse" in table.index
    assert "nonlinear_full_state_rmse" in table.index
    # The two RMSE metrics must not measure the same thing.
    single = table.loc["single_step_update_rmse"]
    nonlinear = table.loc["nonlinear_full_state_rmse"]
    assert single["applies_to_workflow"] != nonlinear["applies_to_workflow"]
    md = Path(run["artifacts"]["metric_definitions"]).read_text(encoding="utf-8")
    assert "Single-step update RMSE is not nonlinear full-state RMSE" in md


def test_residual_distinct_from_state_accuracy(tmp_path: Path) -> None:
    run = build_metric_definitions({"output_dir": str(tmp_path / "metrics")})
    table = pd.read_csv(run["artifacts"]["metric_definitions_table"]).set_index("metric_name")
    residual = table.loc["residual_norm"]
    assert "state accuracy" in str(residual["does_not_measure"]).lower()
    md = Path(run["artifacts"]["metric_definitions"]).read_text(encoding="utf-8")
    assert "Residual is measurement fit, not state accuracy" in md


def test_gate_error_and_qsvt_ridge_equivalence_documented(tmp_path: Path) -> None:
    run = build_metric_definitions({"output_dir": str(tmp_path / "metrics")})
    md = Path(run["artifacts"]["metric_definitions"]).read_text(encoding="utf-8")
    assert "validates the circuit" in md.lower()
    assert "QSVT-target classical equals Ridge" in md
    assert "Top-k exactness depends on margin" in md
