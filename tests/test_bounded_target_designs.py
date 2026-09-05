from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.bounded_target_designs import (
    QSVT_SAFE_TOLERANCE,
    evaluate_target_designs_for_config,
    write_bounded_target_design_outputs,
)
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem


def _toy_subproblem() -> SelectedSubproblem:
    return SelectedSubproblem(
        H_tilde=np.array(
            [
                [1.0, 0.2, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.6, 0.05],
                [0.0, 0.0, 0.03, 0.4],
            ],
            dtype=np.float64,
        ),
        r_tilde=np.array([0.7, -0.3, 0.2, 0.1], dtype=np.float64),
        metadata={"case": "toy"},
    )


def _rows(degree: int = 11):
    rows, samples = evaluate_target_designs_for_config(
        subproblem=_toy_subproblem(),
        alpha=1.0e-3,
        degree=degree,
        margin_factors=[1.05, 1.25],
        grid_size=512,
        gain_cap_factor=0.5,
    )
    return rows, samples


def test_all_named_designs_present_and_completed() -> None:
    rows, _ = _rows()
    by_name = {row["target_design"]: row for row in rows}
    expected = {
        "global_safe",
        "support_scaled",
        "margin_1.05",
        "margin_1.25",
        "degree_adaptive",
        "clipped_inverse_like",
        "piecewise_support_aware",
    }
    assert expected.issubset(by_name.keys())
    assert all(row["run_status"] == "completed" for row in rows)


def test_qsvt_safe_designs_preserve_boundedness() -> None:
    rows, _ = _rows()
    for row in rows:
        if row["qsvt_compatibility_status"] == "qsvt_safe":
            assert float(row["boundedness_margin"]) <= 1.0 + 1.0e-6
            # boundedness_margin == max|p_raw| / C_design, so max|p| = margin <= 1.
            assert (
                float(row["C_design"])
                >= float(row["raw_max_abs_on_unit_interval"]) / (1.0 + 1.0e-6) - 1.0e-9
            )


def test_degree_adaptive_is_qsvt_safe_by_construction() -> None:
    rows, _ = _rows()
    adaptive = next(row for row in rows if row["target_design"] == "degree_adaptive")
    assert adaptive["qsvt_compatibility_status"] == "qsvt_safe"
    assert float(adaptive["boundedness_margin"]) <= 1.0 + QSVT_SAFE_TOLERANCE


def test_scaling_only_designs_share_known_c_residual() -> None:
    rows, _ = _rows()
    scaling_only = [row for row in rows if row["design_family"] == "scaling_only"]
    residuals = [float(row["C_rescaled_polynomial_residual"]) for row in scaling_only]
    reference = residuals[0]
    # The physical known-C-rescaled update is independent of C_design.
    assert max(abs(value - reference) for value in residuals) <= 1.0e-9 * max(abs(reference), 1.0)


def test_shape_changing_designs_are_instance_aware_and_may_differ() -> None:
    rows, _ = _rows()
    scaling_residual = next(
        float(row["C_rescaled_polynomial_residual"])
        for row in rows
        if row["target_design"] == "global_safe"
    )
    shape = [row for row in rows if row["design_family"] == "shape_changing"]
    assert shape, "expected shape-changing designs"
    for row in shape:
        assert row["generalizability_status"] == "instance_aware"
        assert row["design_classification"] in {
            "qsvt_safe_instance_aware",
            "not_qsvt_safe",
            "diagnostic_only",
        }
    # At least one shape-changing design alters the known-C residual relative to scaling-only.
    assert any(
        abs(float(row["C_rescaled_polynomial_residual"]) - scaling_residual)
        > 1.0e-9 * max(abs(scaling_residual), 1.0)
        for row in shape
    )


def test_filter_values_recorded_for_each_design() -> None:
    rows, samples = _rows()
    names = {row["target_design"] for row in rows}
    sampled_names = {sample["target_design"] for sample in samples}
    assert names == sampled_names
    for sample in samples:
        assert np.isfinite(sample["raw_target_value"])
        assert np.isfinite(sample["bounded_polynomial_value"])


def test_outputs_written_with_required_files(tmp_path: Path) -> None:
    rows, samples = _rows()
    artifacts = write_bounded_target_design_outputs(
        tmp_path, {"output_dir": str(tmp_path)}, rows, samples
    )
    for key in [
        "manifest",
        "target_design_summary",
        "target_design_filter_values",
        "target_design_residuals",
        "target_design_tradeoff",
    ]:
        assert artifacts[key].is_file()
    summary = pd.read_csv(artifacts["target_design_summary"])
    assert "boundedness_margin" in summary.columns
    assert "design_classification" in summary.columns
    tradeoff = artifacts["target_design_tradeoff"].read_text(encoding="utf-8")
    assert "Required Answers" in tradeoff
