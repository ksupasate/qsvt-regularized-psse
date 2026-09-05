"""Implementation verification: measurement row metadata and subset-mask audit.

This module audits the existing AC/DC measurement-row metadata, the AC row masks used
by the paper-facing S0-S4/drop-view analyses, and the optional diagnostic-only AC/DC
injection-only and branch-flow-only views. It is an implementation check only: an
unimplemented subset is reported as ``warning_subset_not_implemented`` rather than
fabricated, and a present subset fails if its mask admits an unexpected row type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robust_qsvt_se.data.cases import load_ac_case, load_dc_case
from robust_qsvt_se.measurement.ac_linear import (
    ACMeasurementRow,
    ac_measurements_and_jacobian,
    default_ac_state_vector,
)
from robust_qsvt_se.measurement.dc_linear import MeasurementRow, build_dc_measurement_matrix
from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper._estimation import DEFAULT_CASE_SOURCE, DEFAULT_MEASUREMENT, subset_spec
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

FloatArray = NDArray[np.float64]

SOURCE_SCRIPT = "scripts/run_measurement_row_metadata_audit.py"
DEFAULT_CASES = ("ieee14", "ieee30", "ieee57", "ieee118")

ROW_METADATA_COLUMNS = [
    "case",
    "workflow",
    "row_index",
    "measurement_type",
    "bus_id",
    "from_bus",
    "to_bus",
    "branch_id",
    "state_variables_involved",
    "weight",
    "sigma",
    "row_norm_unweighted",
    "row_norm_weighted",
    "source_function",
    "status",
    "notes",
]

SUBSET_SUMMARY_COLUMNS = [
    "case",
    "workflow",
    "subset_name",
    "expected_measurement_types",
    "actual_measurement_types",
    "num_rows",
    "num_voltage_rows",
    "num_p_injection_rows",
    "num_q_injection_rows",
    "num_p_branch_flow_rows",
    "num_q_branch_flow_rows",
    "num_angle_rows",
    "rank",
    "condition_number_weighted",
    "status",
    "notes",
]

MASK_CHECK_COLUMNS = [
    "case",
    "workflow",
    "subset_name",
    "expected_measurement_type",
    "expected_included",
    "actual_count",
    "mask_count",
    "metadata_count",
    "consistent",
    "status",
    "notes",
]

AC_TYPES = (
    "voltage_magnitude",
    "p_injection",
    "q_injection",
    "p_branch_flow",
    "q_branch_flow",
)
DC_TYPES = ("branch_flow", "bus_injection", "angle")
SUMMARY_TYPES = (*AC_TYPES, "angle")
ALL_CHECK_TYPES = (*AC_TYPES, *DC_TYPES)

_AC_SOURCE_FUNCTION = "measurement.ac_linear.ac_measurements_and_jacobian"
_DC_SOURCE_FUNCTION = "measurement.dc_linear.build_dc_measurement_matrix"

S_SETUPS: tuple[tuple[str, str, set[str]], ...] = (
    ("S0", "voltage_only", {"voltage_magnitude"}),
    ("S1", "s1_voltage_plus_p_injection", {"voltage_magnitude", "p_injection"}),
    (
        "S2",
        "s2_voltage_plus_p_injection_plus_p_branch_flow",
        {"voltage_magnitude", "p_injection", "p_branch_flow"},
    ),
    (
        "S3",
        "s3_voltage_plus_p_injection_plus_p_branch_flow_plus_q_injection",
        {"voltage_magnitude", "p_injection", "p_branch_flow", "q_injection"},
    ),
    (
        "S4",
        "full_ac_measurement_set",
        {"voltage_magnitude", "p_injection", "q_injection", "p_branch_flow", "q_branch_flow"},
    ),
)

DROP_VIEWS: tuple[tuple[str, str | None, set[str]], ...] = (
    (
        "drop_voltage",
        "drop_voltage_rows",
        {"p_injection", "q_injection", "p_branch_flow", "q_branch_flow"},
    ),
    (
        "drop_active_injection",
        "drop_p_injection_rows",
        {"voltage_magnitude", "q_injection", "p_branch_flow", "q_branch_flow"},
    ),
    (
        "drop_reactive_injection",
        "drop_q_injection_rows",
        {"voltage_magnitude", "p_injection", "p_branch_flow", "q_branch_flow"},
    ),
    (
        "drop_active_branch_flow",
        "drop_p_branch_flow_rows",
        {"voltage_magnitude", "p_injection", "q_injection", "q_branch_flow"},
    ),
    (
        "drop_reactive_branch_flow",
        "drop_q_branch_flow_rows",
        {"voltage_magnitude", "p_injection", "q_injection", "p_branch_flow"},
    ),
    (
        "drop_branch_flow",
        "drop_branch_flow_rows",
        {"voltage_magnitude", "p_injection", "q_injection"},
    ),
    (
        "drop_injection",
        "drop_injection_rows",
        {"voltage_magnitude", "p_branch_flow", "q_branch_flow"},
    ),
    ("voltage_only", "voltage_only", {"voltage_magnitude"}),
    ("injection_only", "injection_only", {"p_injection", "q_injection"}),
    ("branch_flow_only", "branch_flow_only", {"p_branch_flow", "q_branch_flow"}),
)

DC_OPTIONAL_VIEWS: tuple[tuple[str, str, set[str]], ...] = (
    ("injection_only", "injection_only", {"bus_injection"}),
    ("branch_flow_only", "branch_flow_only", {"branch_flow"}),
)


def build_measurement_row_metadata_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Build row metadata and subset-mask audit artifacts."""

    input_root = Path(config.get("input_root", "outputs"))
    output_dir = Path(config.get("output_dir", input_root / "measurement_row_metadata_audit"))
    cases = list(config.get("cases", DEFAULT_CASES))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    ensure_directory(output_dir)

    metadata_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    for case_name in cases:
        try:
            ac_rows, _ac_unweighted, ac_weighted = _build_ac_metadata_rows(case_name, case_source)
            metadata_rows.extend(ac_rows)
            summaries, checks = audit_subset_masks(
                case=case_name,
                workflow="ac_linearized",
                H_tilde=ac_weighted,
                metadata_rows=ac_rows,
            )
            subset_rows.extend(summaries)
            mask_rows.extend(checks)
        except Exception as exc:  # record failure, never fabricate rows
            subset_rows.append(_build_failure_summary(case_name, "ac_linearized", exc))
            mask_rows.append(_build_failure_check(case_name, "ac_linearized", exc))

        try:
            dc_rows, _dc_unweighted, dc_weighted = _build_dc_metadata_rows(case_name, case_source)
            metadata_rows.extend(dc_rows)
            summaries, checks = audit_dc_subset_masks(
                case=case_name,
                workflow="dc_linearized",
                H_tilde=dc_weighted,
                metadata_rows=dc_rows,
            )
            subset_rows.extend(summaries)
            mask_rows.extend(checks)
        except Exception as exc:  # record failure as metadata row
            metadata_rows.append(_metadata_failure_row(case_name, "dc_linearized", exc))

    return _write_outputs(
        output_dir=output_dir,
        metadata_rows=metadata_rows,
        subset_rows=subset_rows,
        mask_rows=mask_rows,
        input_config={
            "input_root": str(input_root),
            "output_dir": str(output_dir),
            "cases": cases,
            "case_source": case_source,
        },
    )


def _build_ac_metadata_rows(
    case_name: str, case_source: str
) -> tuple[list[dict[str, Any]], FloatArray, FloatArray]:
    case = load_ac_case(case_name, case_source=case_source)
    state = default_ac_state_vector(case)
    _z, H, rows = ac_measurements_and_jacobian(case, state, dict(DEFAULT_MEASUREMENT))
    stds = np.array([row.std for row in rows], dtype=np.float64)
    H_tilde = H / stds[:, None]
    table_rows = [
        _metadata_row(
            case_name=case_name,
            workflow="ac_linearized",
            row_index=index,
            row=row,
            unweighted_row=H[index],
            weighted_row=H_tilde[index],
            source_function=_AC_SOURCE_FUNCTION,
        )
        for index, row in enumerate(rows)
    ]
    return table_rows, H, H_tilde


def _build_dc_metadata_rows(
    case_name: str, case_source: str
) -> tuple[list[dict[str, Any]], FloatArray, FloatArray]:
    case = load_dc_case(case_name, case_source=case_source)
    measurement_config = {
        "include_branch_flows": True,
        "include_bus_injections": True,
        "angle_buses": list(case.state_buses[:2]),
        "flow_std": 0.02,
        "injection_std": 0.03,
        "angle_std": 0.005,
    }
    H, rows = build_dc_measurement_matrix(case=case, measurement_config=measurement_config)
    stds = np.array([row.std for row in rows], dtype=np.float64)
    H_tilde = H / stds[:, None]
    table_rows = [
        _metadata_row(
            case_name=case_name,
            workflow="dc_linearized",
            row_index=index,
            row=row,
            unweighted_row=H[index],
            weighted_row=H_tilde[index],
            source_function=_DC_SOURCE_FUNCTION,
        )
        for index, row in enumerate(rows)
    ]
    return table_rows, H, H_tilde


def _metadata_row(
    *,
    case_name: str,
    workflow: str,
    row_index: int,
    row: ACMeasurementRow | MeasurementRow,
    unweighted_row: FloatArray,
    weighted_row: FloatArray,
    source_function: str,
) -> dict[str, Any]:
    bus_id = row.buses[0] if len(row.buses) == 1 else ""
    from_bus = row.buses[0] if len(row.buses) == 2 else ""
    to_bus = row.buses[1] if len(row.buses) == 2 else ""
    state_columns = np.flatnonzero(np.abs(np.asarray(unweighted_row, dtype=np.float64)) > 1e-12)
    return {
        "case": case_name,
        "workflow": workflow,
        "row_index": int(row_index),
        "measurement_type": row.measurement_type,
        "bus_id": bus_id,
        "from_bus": from_bus,
        "to_bus": to_bus,
        "branch_id": f"{from_bus}->{to_bus}" if from_bus != "" and to_bus != "" else "",
        "state_variables_involved": "|".join(str(int(index)) for index in state_columns),
        "weight": 1.0 / float(row.std),
        "sigma": float(row.std),
        "row_norm_unweighted": float(np.linalg.norm(unweighted_row)),
        "row_norm_weighted": float(np.linalg.norm(weighted_row)),
        "source_function": source_function,
        "status": "pass",
        "notes": row.label,
    }


def audit_subset_masks(
    *,
    case: str,
    workflow: str,
    H_tilde: FloatArray,
    metadata_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit S0-S4 and drop-view masks against row metadata.

    Exposed for tests so synthetic bad metadata can exercise failure paths.
    """

    subsets = [*S_SETUPS, *DROP_VIEWS]
    return _audit_subset_collection(
        case=case,
        workflow=workflow,
        H_tilde=H_tilde,
        metadata_rows=metadata_rows,
        subsets=subsets,
        check_types=SUMMARY_TYPES,
        subset_resolver=lambda name: (
            subset_spec(name).included_types if subset_spec(name) is not None else None
        ),
    )


def audit_dc_subset_masks(
    *,
    case: str,
    workflow: str,
    H_tilde: FloatArray,
    metadata_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit optional DC diagnostic masks against row metadata."""

    return _audit_subset_collection(
        case=case,
        workflow=workflow,
        H_tilde=H_tilde,
        metadata_rows=metadata_rows,
        subsets=DC_OPTIONAL_VIEWS,
        check_types=DC_TYPES,
        subset_resolver=_dc_subset_types,
    )


def _audit_subset_collection(
    *,
    case: str,
    workflow: str,
    H_tilde: FloatArray,
    metadata_rows: list[dict[str, Any]],
    subsets: tuple[tuple[str, str | None, set[str]], ...],
    check_types: tuple[str, ...],
    subset_resolver: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for subset_name, implemented_name, expected_types in subsets:
        included_types = subset_resolver(implemented_name) if implemented_name else None
        if implemented_name is None or included_types is None:
            summaries.append(_not_implemented_summary(case, workflow, subset_name, expected_types))
            checks.extend(
                _not_implemented_checks(
                    case, workflow, subset_name, expected_types, check_types=check_types
                )
            )
            continue
        summary, subset_checks = _audit_one_subset(
            case=case,
            workflow=workflow,
            subset_name=subset_name,
            implemented_name=implemented_name,
            included_types=tuple(included_types),
            expected_types=expected_types,
            H_tilde=H_tilde,
            metadata_rows=metadata_rows,
            check_types=check_types,
        )
        summaries.append(summary)
        checks.extend(subset_checks)
    return summaries, checks


def _dc_subset_types(subset_name: str | None) -> tuple[str, ...] | None:
    if subset_name == "injection_only":
        return ("bus_injection",)
    if subset_name == "branch_flow_only":
        return ("branch_flow",)
    return None


def _audit_one_subset(
    *,
    case: str,
    workflow: str,
    subset_name: str,
    implemented_name: str,
    included_types: tuple[str, ...] | None = None,
    expected_types: set[str],
    H_tilde: FloatArray,
    metadata_rows: list[dict[str, Any]],
    check_types: tuple[str, ...] = SUMMARY_TYPES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_types = [str(row.get("measurement_type", "")) for row in metadata_rows]
    if len(full_types) != int(np.asarray(H_tilde).shape[0]):
        summary = _subset_summary(
            case=case,
            workflow=workflow,
            subset_name=subset_name,
            expected_types=expected_types,
            actual_types=set(),
            selected=np.empty((0, np.asarray(H_tilde).shape[1]), dtype=np.float64),
            selected_rows=[],
            status="fail_missing_metadata",
            notes=(
                f"{implemented_name}: metadata row count {len(full_types)} does not match "
                f"H_tilde row count {np.asarray(H_tilde).shape[0]}"
            ),
        )
        return summary, [
            _mask_check_row(
                case=case,
                workflow=workflow,
                subset_name=subset_name,
                measurement_type=mtype,
                expected_included=mtype in expected_types,
                actual_count=0,
                mask_count=0,
                metadata_count=len(full_types),
                consistent=False,
                status="fail_missing_metadata",
                notes="metadata/H_tilde row-count mismatch",
            )
            for mtype in check_types
        ]

    if included_types is None:
        spec = subset_spec(implemented_name)
        included_types = spec.included_types if spec is not None else None
    if included_types is None:
        return _not_implemented_summary(case, workflow, subset_name, expected_types), (
            _not_implemented_checks(
                case, workflow, subset_name, expected_types, check_types=check_types
            )
        )
    mask = np.array([mtype in included_types for mtype in full_types], dtype=bool)
    selected_rows = [row for row, keep in zip(metadata_rows, mask, strict=True) if keep]
    selected = np.asarray(H_tilde, dtype=np.float64)[mask, :]
    actual_types = {str(row.get("measurement_type", "")) for row in selected_rows}
    unexpected = actual_types - expected_types
    missing = expected_types - actual_types
    checks = [
        _type_check(
            case=case,
            workflow=workflow,
            subset_name=subset_name,
            measurement_type=mtype,
            expected_types=expected_types,
            selected_rows=selected_rows,
            full_types=full_types,
            mask=mask,
        )
        for mtype in check_types
    ]
    failing_statuses = {row["status"] for row in checks if str(row["status"]).startswith("fail")}
    if selected.shape[0] == 0:
        status = "warning_empty_subset"
        notes = f"{implemented_name}: mask selected no rows"
    elif unexpected or missing:
        status = "fail_wrong_row_type"
        notes = (
            f"{implemented_name}: unexpected={sorted(unexpected) or 'none'}; "
            f"missing={sorted(missing) or 'none'}"
        )
    elif failing_statuses:
        status = sorted(failing_statuses)[0]
        notes = f"{implemented_name}: row-count consistency failure"
    else:
        status = "pass"
        notes = f"{implemented_name}: exact row-type mask"
    notes = _append_diagnostic_rank_note(notes, selected, status)
    summary = _subset_summary(
        case=case,
        workflow=workflow,
        subset_name=subset_name,
        expected_types=expected_types,
        actual_types=actual_types,
        selected=selected,
        selected_rows=selected_rows,
        status=status,
        notes=notes,
    )
    return summary, checks


def _type_check(
    *,
    case: str,
    workflow: str,
    subset_name: str,
    measurement_type: str,
    expected_types: set[str],
    selected_rows: list[dict[str, Any]],
    full_types: list[str],
    mask: FloatArray,
) -> dict[str, Any]:
    expected = measurement_type in expected_types
    actual_count = sum(
        1 for row in selected_rows if row.get("measurement_type") == measurement_type
    )
    mask_count = sum(
        1
        for mtype, keep in zip(full_types, mask, strict=True)
        if keep and mtype == measurement_type
    )
    metadata_count = actual_count
    consistent = actual_count == mask_count == metadata_count
    if not consistent:
        status = "fail_count_mismatch"
        notes = "selected metadata count differs from mask count"
    elif expected and actual_count == 0:
        status = "fail_missing_metadata"
        consistent = False
        notes = "expected row type missing from selected metadata"
    elif not expected and actual_count > 0:
        status = "fail_wrong_row_type"
        consistent = False
        notes = "unexpected row type included by mask"
    else:
        status = "pass"
        notes = "row type count matches mask expectation"
    return _mask_check_row(
        case=case,
        workflow=workflow,
        subset_name=subset_name,
        measurement_type=measurement_type,
        expected_included=expected,
        actual_count=actual_count,
        mask_count=mask_count,
        metadata_count=metadata_count,
        consistent=consistent,
        status=status,
        notes=notes,
    )


def _subset_summary(
    *,
    case: str,
    workflow: str,
    subset_name: str,
    expected_types: set[str],
    actual_types: set[str],
    selected: FloatArray,
    selected_rows: list[dict[str, Any]],
    status: str,
    notes: str,
) -> dict[str, Any]:
    counts = {
        mtype: sum(1 for row in selected_rows if row.get("measurement_type") == mtype)
        for mtype in ALL_CHECK_TYPES
    }
    if selected.size and selected.shape[0] > 0:
        rank = int(np.linalg.matrix_rank(selected))
        cond = _condition_number(selected)
    else:
        rank = ""
        cond = ""
    return {
        "case": case,
        "workflow": workflow,
        "subset_name": subset_name,
        "expected_measurement_types": "|".join(sorted(expected_types)),
        "actual_measurement_types": "|".join(sorted(actual_types)),
        "num_rows": int(selected.shape[0]) if selected.ndim == 2 else 0,
        "num_voltage_rows": counts["voltage_magnitude"],
        "num_p_injection_rows": counts["p_injection"],
        "num_q_injection_rows": counts["q_injection"],
        "num_p_branch_flow_rows": counts["p_branch_flow"],
        "num_q_branch_flow_rows": counts["q_branch_flow"],
        "num_angle_rows": counts["angle"],
        "rank": rank,
        "condition_number_weighted": cond,
        "status": status,
        "notes": notes,
    }


def _condition_number(matrix: FloatArray) -> float:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular.size == 0:
        return float("nan")
    sigma_min = float(np.min(singular))
    sigma_max = float(np.max(singular))
    return sigma_max / sigma_min if sigma_min > 0.0 else float("inf")


def _append_diagnostic_rank_note(notes: str, selected: FloatArray, status: str) -> str:
    if status != "pass" or selected.size == 0 or selected.shape[0] == 0:
        return notes
    rank = int(np.linalg.matrix_rank(selected))
    dimension_bound = min(int(selected.shape[0]), int(selected.shape[1]))
    cond = _condition_number(selected)
    diagnostics: list[str] = []
    if rank < dimension_bound:
        diagnostics.append("rank_deficient diagnostic subset")
    if np.isinf(cond) or cond > 1.0e12:
        diagnostics.append("ill_conditioned diagnostic subset")
    if not diagnostics:
        return notes
    return f"{notes}; {'; '.join(diagnostics)}"


def _not_implemented_summary(
    case: str, workflow: str, subset_name: str, expected_types: set[str]
) -> dict[str, Any]:
    return _subset_summary(
        case=case,
        workflow=workflow,
        subset_name=subset_name,
        expected_types=expected_types,
        actual_types=set(),
        selected=np.empty((0, 0), dtype=np.float64),
        selected_rows=[],
        status="warning_subset_not_implemented",
        notes="subset/drop view is not implemented in subset_spec; no rows fabricated",
    )


def _not_implemented_checks(
    case: str,
    workflow: str,
    subset_name: str,
    expected_types: set[str],
    *,
    check_types: tuple[str, ...] = SUMMARY_TYPES,
) -> list[dict[str, Any]]:
    return [
        _mask_check_row(
            case=case,
            workflow=workflow,
            subset_name=subset_name,
            measurement_type=mtype,
            expected_included=mtype in expected_types,
            actual_count="",
            mask_count="",
            metadata_count="",
            consistent="",
            status="warning_subset_not_implemented",
            notes="subset/drop view is not implemented in subset_spec; no rows fabricated",
        )
        for mtype in check_types
    ]


def _mask_check_row(
    *,
    case: str,
    workflow: str,
    subset_name: str,
    measurement_type: str,
    expected_included: bool,
    actual_count: Any,
    mask_count: Any,
    metadata_count: Any,
    consistent: Any,
    status: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "case": case,
        "workflow": workflow,
        "subset_name": subset_name,
        "expected_measurement_type": measurement_type,
        "expected_included": bool(expected_included),
        "actual_count": actual_count,
        "mask_count": mask_count,
        "metadata_count": metadata_count,
        "consistent": consistent,
        "status": status,
        "notes": notes,
    }


def _metadata_failure_row(case_name: str, workflow: str, exc: Exception) -> dict[str, Any]:
    return {
        "case": case_name,
        "workflow": workflow,
        "row_index": "",
        "measurement_type": "",
        "bus_id": "",
        "from_bus": "",
        "to_bus": "",
        "branch_id": "",
        "state_variables_involved": "",
        "weight": "",
        "sigma": "",
        "row_norm_unweighted": "",
        "row_norm_weighted": "",
        "source_function": "",
        "status": "fail_missing_metadata",
        "notes": f"{type(exc).__name__}: {exc}",
    }


def _build_failure_summary(case_name: str, workflow: str, exc: Exception) -> dict[str, Any]:
    return {
        "case": case_name,
        "workflow": workflow,
        "subset_name": "all",
        "expected_measurement_types": "",
        "actual_measurement_types": "",
        "num_rows": "",
        "num_voltage_rows": "",
        "num_p_injection_rows": "",
        "num_q_injection_rows": "",
        "num_p_branch_flow_rows": "",
        "num_q_branch_flow_rows": "",
        "num_angle_rows": "",
        "rank": "",
        "condition_number_weighted": "",
        "status": "fail_missing_metadata",
        "notes": f"{type(exc).__name__}: {exc}",
    }


def _build_failure_check(case_name: str, workflow: str, exc: Exception) -> dict[str, Any]:
    return _mask_check_row(
        case=case_name,
        workflow=workflow,
        subset_name="all",
        measurement_type="all",
        expected_included=True,
        actual_count="",
        mask_count="",
        metadata_count="",
        consistent=False,
        status="fail_missing_metadata",
        notes=f"{type(exc).__name__}: {exc}",
    )


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def audit_status(subset_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("status", "")) for row in [*subset_rows, *mask_rows]}
    if any(status.startswith("fail") for status in statuses):
        return "fail"
    if any(status.startswith("warning") for status in statuses):
        return "warning"
    return "pass"


def _summary_markdown(
    metadata_rows: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
) -> str:
    status = audit_status(subset_rows, mask_rows)
    mask_failures = sum(1 for row in mask_rows if str(row.get("status", "")).startswith("fail"))
    subset_counts = _status_counts(subset_rows)
    mask_counts = _status_counts(mask_rows)
    cases = sorted({str(row["case"]) for row in metadata_rows if row.get("case")})
    row_types = sorted(
        {str(row["measurement_type"]) for row in metadata_rows if row.get("measurement_type")}
    )
    return "\n".join(
        [
            "# Measurement Row Metadata / Mask Audit",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "This audit verifies that generated measurement-row metadata and AC subset masks "
            "agree with the implemented measurement type definitions. It does not introduce "
            "new experiments or new scientific claims.",
            "",
            "## Result",
            f"- Overall status: **{status}**.",
            f"- Cases audited: {cases or 'none'}.",
            f"- Row types observed: {row_types or 'none'}.",
            f"- Metadata rows: {len(metadata_rows)}.",
            f"- Mask failure rows: {mask_failures}.",
            "",
            "## Subset status counts",
            *[f"- {name}: {count}" for name, count in sorted(subset_counts.items())],
            "",
            "## Mask-check status counts",
            *[f"- {name}: {count}" for name, count in sorted(mask_counts.items())],
            "",
            "## Notes",
            "- S0-S4 use the implemented AC row-type subsets from `subset_spec`.",
            "- Paper-facing drop-view names are mapped to implemented `*_rows` subset names.",
            "- `warning_subset_not_implemented` records a requested view with no implemented "
            "subset; no row composition is fabricated.",
            "- The weighted condition numbers use the already weighted Jacobian rows.",
            "",
        ]
    )


def _write_outputs(
    *,
    output_dir: Path,
    metadata_rows: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    metadata_path = rows_to_table(
        metadata_rows, output_dir / "row_metadata_audit.csv", ROW_METADATA_COLUMNS
    )
    subset_path = rows_to_table(
        subset_rows,
        output_dir / "subset_row_composition_summary.csv",
        SUBSET_SUMMARY_COLUMNS,
    )
    checks_path = rows_to_table(
        mask_rows, output_dir / "row_mask_consistency_checks.csv", MASK_CHECK_COLUMNS
    )
    summary_path = output_dir / "measurement_row_metadata_summary.md"
    summary_path.write_text(
        _summary_markdown(metadata_rows, subset_rows, mask_rows), encoding="utf-8"
    )
    artifacts = {
        "row_metadata_audit": str(metadata_path),
        "subset_row_composition_summary": str(subset_path),
        "row_mask_consistency_checks": str(checks_path),
        "measurement_row_metadata_summary": str(summary_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "metadata_rows": metadata_rows,
        "subset_rows": subset_rows,
        "mask_rows": mask_rows,
        "status": audit_status(subset_rows, mask_rows),
        "mask_failures": sum(
            1 for row in mask_rows if str(row.get("status", "")).startswith("fail")
        ),
    }
