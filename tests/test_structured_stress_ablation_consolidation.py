from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.structured_stress_ablation_consolidation import (
    build_structured_stress_ablation_consolidation,
)


def _write_missing_sensitivity(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source_output": ["outputs/nonlinear_ac_ieee14_seed10"] * 3,
            "sweep_parameter": ["scenario.missing_ratio"] * 3,
            "sweep_value": [0.0, 0.1, 0.2],
            "estimator": ["ridge", "ridge", "ridge"],
            "rmse_median": [0.1, 0.2, 0.3],
            "weighted_residual_norm_mean": [0.4, 0.5, 0.6],
            "condition_number_mean": [100.0, 200.0, 300.0],
        }
    ).to_csv(path, index=False)


def test_random_missing_not_labeled_structured(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    _write_missing_sensitivity(input_root / "sensitivity_summary" / "missing_sensitivity.csv")
    run = build_structured_stress_ablation_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase5")}
    )
    stress = pd.read_csv(run["artifacts"]["paper_table_structured_stress_ablation"])
    missing_rows = stress[stress["stress_type"] == "missing_only"]
    assert not missing_rows.empty
    # Random missing is labelled random, never as structured/spatial missing.
    assert (missing_rows["stress_subtype"] == "random").all()
    assert not stress["stress_type"].astype(str).str.contains("structured").any()


def test_structured_stress_absent_keeps_claim_missing(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()  # no sensitivity/redundancy/diagnostic inputs
    run = build_structured_stress_ablation_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase5")}
    )
    stress = pd.read_csv(run["artifacts"]["paper_table_structured_stress_ablation"])
    assert stress.empty
    assert run["structured_stress_available"] is False
    status = Path(run["artifacts"]["structured_stress_status"]).read_text(encoding="utf-8")
    assert "missing_evidence" in status


def test_missing_outputs_list_measurement_type_drops(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    _write_missing_sensitivity(input_root / "sensitivity_summary" / "missing_sensitivity.csv")
    run = build_structured_stress_ablation_consolidation(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase5")}
    )
    missing = pd.read_csv(run["artifacts"]["missing_structured_stress_outputs"])
    blob = " ".join(missing["missing_output"].astype(str)).lower()
    assert "voltage_only" in blob or "drop_voltage_rows" in blob
    assert "field-calibrated" in blob
