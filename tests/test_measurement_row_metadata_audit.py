from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.data.cases import load_ac_case
from robust_qsvt_se.measurement.ac_linear import (
    ac_measurements_and_jacobian,
    default_ac_state_vector,
)
from robust_qsvt_se.paper._estimation import DEFAULT_MEASUREMENT
from robust_qsvt_se.paper.measurement_row_metadata_audit import (
    MASK_CHECK_COLUMNS,
    ROW_METADATA_COLUMNS,
    SUBSET_SUMMARY_COLUMNS,
    _audit_one_subset,
    audit_dc_subset_masks,
    audit_subset_masks,
    build_measurement_row_metadata_audit,
)


def _run(tmp_path: Path) -> dict:
    return build_measurement_row_metadata_audit(
        {"cases": ["ieee14"], "output_dir": str(tmp_path / "row_audit")}
    )


def _synthetic_real_metadata() -> tuple[np.ndarray, list[dict]]:
    case = load_ac_case("ieee14", case_source="pypower")
    state = default_ac_state_vector(case)
    _z, H, rows = ac_measurements_and_jacobian(case, state, dict(DEFAULT_MEASUREMENT))
    stds = np.array([row.std for row in rows], dtype=np.float64)
    H_tilde = H / stds[:, None]
    metadata = [
        {
            "case": "ieee14",
            "workflow": "ac_linearized",
            "row_index": index,
            "measurement_type": row.measurement_type,
        }
        for index, row in enumerate(rows)
    ]
    return H_tilde, metadata


def test_outputs_have_required_columns(tmp_path: Path) -> None:
    run = _run(tmp_path)
    metadata = pd.read_csv(run["artifacts"]["row_metadata_audit"])
    subsets = pd.read_csv(run["artifacts"]["subset_row_composition_summary"])
    checks = pd.read_csv(run["artifacts"]["row_mask_consistency_checks"])
    assert list(metadata.columns) == ROW_METADATA_COLUMNS
    assert list(subsets.columns) == SUBSET_SUMMARY_COLUMNS
    assert list(checks.columns) == MASK_CHECK_COLUMNS
    assert not metadata.empty


def test_s1_s4_masks_contain_exact_expected_types_when_implemented(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = pd.read_csv(run["artifacts"]["subset_row_composition_summary"])
    expected = {
        "S1": {"voltage_magnitude", "p_injection"},
        "S2": {"voltage_magnitude", "p_injection", "p_branch_flow"},
        "S3": {"voltage_magnitude", "p_injection", "p_branch_flow", "q_injection"},
        "S4": {
            "voltage_magnitude",
            "p_injection",
            "q_injection",
            "p_branch_flow",
            "q_branch_flow",
        },
    }
    for subset, types in expected.items():
        row = summary[summary["subset_name"] == subset].iloc[0]
        assert row["status"] == "pass"
        assert set(str(row["actual_measurement_types"]).split("|")) == types


def test_optional_ac_subset_views_have_exact_row_types(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = pd.read_csv(run["artifacts"]["subset_row_composition_summary"])
    ac = summary[summary["workflow"] == "ac_linearized"]
    expected = {
        "injection_only": {"p_injection", "q_injection"},
        "branch_flow_only": {"p_branch_flow", "q_branch_flow"},
    }
    for subset, types in expected.items():
        row = ac[ac["subset_name"] == subset].iloc[0]
        assert row["status"] == "pass"
        assert set(str(row["actual_measurement_types"]).split("|")) == types


def test_optional_dc_subset_views_have_exact_row_types(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = pd.read_csv(run["artifacts"]["subset_row_composition_summary"])
    checks = pd.read_csv(run["artifacts"]["row_mask_consistency_checks"])
    dc = summary[summary["workflow"] == "dc_linearized"]
    expected = {
        "injection_only": {"bus_injection"},
        "branch_flow_only": {"branch_flow"},
    }
    for subset, types in expected.items():
        row = dc[dc["subset_name"] == subset].iloc[0]
        assert row["status"] == "pass"
        assert set(str(row["actual_measurement_types"]).split("|")) == types
        selected_checks = checks[
            (checks["workflow"] == "dc_linearized") & (checks["subset_name"] == subset)
        ]
        assert set(selected_checks["expected_measurement_type"]) == {
            "branch_flow",
            "bus_injection",
            "angle",
        }
        assert (selected_checks["status"] == "pass").all()


def test_optional_subset_views_are_not_reported_not_implemented(tmp_path: Path) -> None:
    run = _run(tmp_path)
    checks = pd.read_csv(run["artifacts"]["row_mask_consistency_checks"])
    optional = checks[checks["subset_name"].isin(["injection_only", "branch_flow_only"])]
    assert not optional.empty
    assert "warning_subset_not_implemented" not in set(optional["status"])


def test_drop_view_masks_remove_intended_row_types(tmp_path: Path) -> None:
    run = _run(tmp_path)
    checks = pd.read_csv(run["artifacts"]["row_mask_consistency_checks"])
    drop_p = checks[
        (checks["subset_name"] == "drop_active_injection")
        & (checks["expected_measurement_type"] == "p_injection")
    ].iloc[0]
    drop_branch = checks[
        (checks["subset_name"] == "drop_branch_flow")
        & checks["expected_measurement_type"].isin(["p_branch_flow", "q_branch_flow"])
    ]
    assert bool(drop_p["expected_included"]) is False
    assert int(drop_p["actual_count"]) == 0
    assert (pd.to_numeric(drop_branch["actual_count"]) == 0).all()
    assert (drop_branch["status"] == "pass").all()


def test_row_metadata_count_equals_mask_count(tmp_path: Path) -> None:
    run = _run(tmp_path)
    checks = pd.read_csv(run["artifacts"]["row_mask_consistency_checks"])
    implemented = checks[checks["status"] != "warning_subset_not_implemented"]
    assert not implemented.empty
    assert (implemented["actual_count"] == implemented["mask_count"]).all()
    assert (implemented["actual_count"] == implemented["metadata_count"]).all()


def test_wrong_row_type_metadata_triggers_failure() -> None:
    H_tilde, metadata = _synthetic_real_metadata()
    bad = [dict(row) for row in metadata]
    for row in bad:
        if row["measurement_type"] == "p_injection":
            row["measurement_type"] = "q_branch_flow"
    summaries, checks = audit_subset_masks(
        case="ieee14", workflow="ac_linearized", H_tilde=H_tilde, metadata_rows=bad
    )
    s1 = next(row for row in summaries if row["subset_name"] == "S1")
    assert s1["status"] == "fail_wrong_row_type"
    assert any(row["status"].startswith("fail") for row in checks)


def test_missing_metadata_triggers_failure() -> None:
    H_tilde, metadata = _synthetic_real_metadata()
    summaries, checks = audit_subset_masks(
        case="ieee14", workflow="ac_linearized", H_tilde=H_tilde, metadata_rows=metadata[:-1]
    )
    assert any(row["status"] == "fail_missing_metadata" for row in summaries)
    assert any(row["status"] == "fail_missing_metadata" for row in checks)


def test_summary_status_fails_if_subset_contains_unexpected_type() -> None:
    H_tilde, metadata = _synthetic_real_metadata()
    summary, _checks = _audit_one_subset(
        case="ieee14",
        workflow="ac_linearized",
        subset_name="bad_expected_types",
        implemented_name="full_ac_measurement_set",
        expected_types={"voltage_magnitude"},
        H_tilde=H_tilde,
        metadata_rows=metadata,
    )
    assert summary["status"] == "fail_wrong_row_type"


def test_rank_deficient_optional_subset_is_diagnostic_not_failure() -> None:
    H_tilde = np.array([[1.0, 0.0], [1.0, 0.0]])
    metadata = [
        {
            "case": "tiny",
            "workflow": "dc_linearized",
            "row_index": 0,
            "measurement_type": "bus_injection",
        },
        {
            "case": "tiny",
            "workflow": "dc_linearized",
            "row_index": 1,
            "measurement_type": "bus_injection",
        },
    ]
    summaries, checks = audit_dc_subset_masks(
        case="tiny", workflow="dc_linearized", H_tilde=H_tilde, metadata_rows=metadata
    )
    injection = next(row for row in summaries if row["subset_name"] == "injection_only")
    assert injection["status"] == "pass"
    assert "rank_deficient diagnostic subset" in injection["notes"]
    injection_checks = [row for row in checks if row["subset_name"] == "injection_only"]
    assert not any(str(row["status"]).startswith("fail") for row in injection_checks)
