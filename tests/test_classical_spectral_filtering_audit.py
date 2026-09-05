from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.classical_spectral_filtering_audit import (
    build_classical_spectral_filtering_audit,
)


def test_estimator_definitions_do_not_claim_qsvt_beats_ridge(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    run = build_classical_spectral_filtering_audit(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase2")}
    )
    estimators = pd.read_csv(run["artifacts"]["paper_table_estimator_definitions"])
    names = set(estimators["estimator"])
    assert {"ridge_tikhonov", "qsvt_target_classical", "pseudoinverse"}.issubset(names)

    qsvt_row = estimators[estimators["estimator"] == "qsvt_target_classical"].iloc[0]
    boundary = str(qsvt_row["claim_boundary"]).lower()
    assert "identical to ridge" in boundary
    assert "not superior" in boundary

    blob = " ".join(estimators.astype(str).to_numpy().ravel()).lower()
    assert "beats ridge" not in blob
    assert "quantum advantage" not in blob


def test_classical_audit_records_missing_main_results_when_absent(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()  # no result directories -> main results unavailable
    run = build_classical_spectral_filtering_audit(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase2")}
    )
    missing = pd.read_csv(run["artifacts"]["missing_classical_outputs"])
    status_text = Path(run["artifacts"]["classical_core_status"]).read_text(encoding="utf-8")

    missing_names = " ".join(missing["missing_output"].astype(str)).lower()
    assert "classical main results" in missing_names
    assert "singular-spectrum" in missing_names
    assert "complete: no" in status_text


def test_filter_comparison_marks_ridge_qsvt_identical(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    run = build_classical_spectral_filtering_audit(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "phase2")}
    )
    figure = pd.read_csv(run["artifacts"]["figure_data_filter_comparison"])
    assert not figure.empty
    assert bool((figure["ridge_filter"] == figure["qsvt_regularized_filter"]).all()) is True
