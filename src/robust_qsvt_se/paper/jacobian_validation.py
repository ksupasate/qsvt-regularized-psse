"""Implementation verification: finite-difference Jacobian validation and weighted audit.

This is an implementation-correctness check, not a new scientific experiment. It confirms
that the analytic AC/DC measurement Jacobian ``H = dh/dx`` matches central finite
differences

    H_ij^FD ~= (h_i(x + eps e_j) - h_i(x - eps e_j)) / (2 eps),

reports the relative Frobenius error

    eps_H = ||H_analytic - H_FD||_F / max(1, ||H_FD||_F),

and audits that the weighted Jacobian ``tilde_H = R^{-1/2} H`` (with diagonal covariance
``R = diag(sigma_i^2)``) is applied consistently and that condition numbers refer to the
weighted Jacobian. It fabricates nothing: an unavailable measurement row type is recorded
as ``not_implemented`` and an invalid variance is reported, never silently corrected.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robust_qsvt_se.data.cases import load_ac_case, load_dc_case
from robust_qsvt_se.measurement.ac_linear import (
    ac_measurement_vector,
    ac_measurements_and_jacobian,
    default_ac_state_vector,
)
from robust_qsvt_se.measurement.dc_linear import build_dc_measurement_matrix
from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper._estimation import DEFAULT_CASE_SOURCE, DEFAULT_MEASUREMENT
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

FloatArray = NDArray[np.float64]

SOURCE_SCRIPT = "scripts/run_jacobian_validation.py"

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57", "ieee118")
DEFAULT_EPSILON = 1.0e-6
PERTURBATION_SEED = 20240617
PERTURBATION_STD = 0.01

# Conservative pass thresholds for the relative-error metrics.
FROBENIUS_THRESHOLD = 1.0e-4
MEDIAN_THRESHOLD = 1.0e-4
SCALING_WARNING_FACTOR = 100.0
NEAR_ZERO = 1.0e-9

# Per-type DC measurement standard deviations (mirrors DEFAULT_MEASUREMENT scaling).
DC_MEASUREMENT_BASE: dict[str, Any] = {
    "include_branch_flows": True,
    "include_bus_injections": True,
    "flow_std": 0.02,
    "injection_std": 0.03,
    "angle_std": 0.005,
}

AC_DETAIL_COLUMNS = [
    "case",
    "operating_point_label",
    "epsilon",
    "measurement_type",
    "row_index",
    "state_index",
    "analytic_value",
    "finite_difference_value",
    "absolute_error",
    "relative_error",
    "row_relative_error",
    "case_relative_frobenius_error",
    "status",
    "source_function",
    "notes",
]
AC_SUMMARY_COLUMNS = [
    "case",
    "operating_point_label",
    "measurement_type",
    "num_rows_checked",
    "num_state_columns_checked",
    "max_absolute_error",
    "max_relative_error",
    "median_relative_error",
    "frobenius_relative_error",
    "pass_threshold",
    "status",
    "notes",
]
DC_DETAIL_COLUMNS = ["workflow", *AC_DETAIL_COLUMNS]
DC_SUMMARY_COLUMNS = ["workflow", *AC_SUMMARY_COLUMNS]

WEIGHTED_COLUMNS = [
    "case",
    "workflow",
    "measurement_type",
    "num_rows",
    "num_states",
    "sigma_min_unweighted",
    "sigma_max_unweighted",
    "condition_unweighted",
    "sigma_min_weighted",
    "sigma_max_weighted",
    "condition_weighted",
    "weighting_source",
    "diagonal_covariance_assumed",
    "status",
    "notes",
]

_AC_SOURCE_FUNCTION = "measurement.ac_linear.ac_measurements_and_jacobian"
_DC_SOURCE_FUNCTION = "measurement.dc_linear.build_dc_measurement_matrix"
_WEIGHTING_SOURCE = "R = diag(measurement_std^2); tilde_H = H / std (R^{-1/2} H)"


# ---------------------------------------------------------------------------
# Generic finite-difference primitives (unit-testable without power-system data)
# ---------------------------------------------------------------------------
def finite_difference_jacobian(
    func: Callable[[FloatArray], FloatArray], x: FloatArray, epsilon: float = DEFAULT_EPSILON
) -> FloatArray:
    """Central finite-difference Jacobian of ``func`` at ``x`` with step ``epsilon``."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    x = np.asarray(x, dtype=np.float64)
    base = np.asarray(func(x), dtype=np.float64)
    jac = np.zeros((base.shape[0], x.shape[0]), dtype=np.float64)
    for j in range(x.shape[0]):
        step = np.zeros_like(x)
        step[j] = epsilon
        forward = np.asarray(func(x + step), dtype=np.float64)
        backward = np.asarray(func(x - step), dtype=np.float64)
        jac[:, j] = (forward - backward) / (2.0 * epsilon)
    return jac


def relative_frobenius_error(analytic: FloatArray, finite_difference: FloatArray) -> float:
    """Relative Frobenius error ``||A - FD||_F / max(1, ||FD||_F)``."""

    analytic = np.asarray(analytic, dtype=np.float64)
    finite_difference = np.asarray(finite_difference, dtype=np.float64)
    denom = max(1.0, float(np.linalg.norm(finite_difference)))
    return float(np.linalg.norm(analytic - finite_difference) / denom)


def compare_jacobian_block(analytic: FloatArray, finite_difference: FloatArray) -> dict[str, Any]:
    """Compare an analytic Jacobian block to its finite-difference estimate.

    Returns the element/row/Frobenius error metrics plus a conservative status. Near-zero
    structural entries are excluded before deciding a fail so that they cannot trigger a
    false failure; a genuinely wrong analytic block (large non-near-zero error) fails.
    """

    analytic = np.atleast_2d(np.asarray(analytic, dtype=np.float64))
    finite_difference = np.atleast_2d(np.asarray(finite_difference, dtype=np.float64))
    diff = analytic - finite_difference
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(1.0, np.abs(finite_difference))
    frob = relative_frobenius_error(analytic, finite_difference)
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    median_rel = float(np.median(rel_err)) if rel_err.size else 0.0
    status = _status_for_block(diff, finite_difference, analytic, frob, median_rel)
    return {
        "num_rows": int(analytic.shape[0]),
        "num_cols": int(analytic.shape[1]),
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "median_relative_error": median_rel,
        "frobenius_relative_error": frob,
        "status": status,
    }


def _status_for_block(
    diff: FloatArray,
    finite_difference: FloatArray,
    analytic: FloatArray,
    frob: float,
    median_rel: float,
) -> str:
    if frob <= FROBENIUS_THRESHOLD and median_rel <= MEDIAN_THRESHOLD:
        return "pass"
    near_zero = (np.abs(finite_difference) < NEAR_ZERO) & (np.abs(analytic) < NEAR_ZERO)
    significant = ~near_zero
    if significant.any():
        denom = max(1.0, float(np.linalg.norm(finite_difference[significant])))
        frob_significant = float(np.linalg.norm(diff[significant]) / denom)
        rel_significant = np.abs(diff[significant]) / np.maximum(
            1.0, np.abs(finite_difference[significant])
        )
        median_significant = float(np.median(rel_significant)) if rel_significant.size else 0.0
        if frob_significant <= FROBENIUS_THRESHOLD and median_significant <= MEDIAN_THRESHOLD:
            return "warning_near_zero_row"
        if frob_significant <= SCALING_WARNING_FACTOR * FROBENIUS_THRESHOLD:
            return "warning_scaling_sensitive"
        return "fail"
    return "warning_near_zero_row"


def weighted_jacobian(jacobian: FloatArray, stds: FloatArray) -> FloatArray:
    """Return ``tilde_H = R^{-1/2} H`` for diagonal ``R = diag(std^2)`` (validates std>0)."""

    stds = np.asarray(stds, dtype=np.float64)
    if has_nonpositive_variance(stds):
        raise ValueError("measurement standard deviations must be strictly positive")
    return np.asarray(jacobian, dtype=np.float64) / stds[:, None]


def has_nonpositive_variance(stds: FloatArray) -> bool:
    """True iff any standard deviation is non-positive or non-finite (invalid variance)."""

    stds = np.asarray(stds, dtype=np.float64)
    return bool(stds.size == 0 or np.any(~np.isfinite(stds)) or np.any(stds <= 0.0))


def _singular_extremes(matrix: FloatArray) -> tuple[float, float, float]:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular.size == 0:
        return float("nan"), float("nan"), float("inf")
    sigma_min = float(np.min(singular))
    sigma_max = float(np.max(singular))
    condition = sigma_max / sigma_min if sigma_min > 0.0 else float("inf")
    return sigma_min, sigma_max, condition


# ---------------------------------------------------------------------------
# AC finite-difference Jacobian validation (Phase 2)
# ---------------------------------------------------------------------------
def _ac_operating_points(case: Any) -> dict[str, FloatArray]:
    base = default_ac_state_vector(case)
    rng = np.random.default_rng(PERTURBATION_SEED)
    perturbed = base + rng.normal(0.0, PERTURBATION_STD, size=base.shape[0])
    n_angles = len(case.angle_state_buses)
    # Keep voltage magnitudes strictly positive (unpack_ac_state requires it).
    perturbed[n_angles:] = np.maximum(perturbed[n_angles:], 0.5)
    return {"base_operating_point": base, "small_perturbed_operating_point": perturbed}


def _validate_ac_case(
    case_name: str, case_source: str, epsilon: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case = load_ac_case(case_name, case_source=case_source)
    measurement_config = dict(DEFAULT_MEASUREMENT)
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    available_types: set[str] = set()
    for label, state in _ac_operating_points(case).items():
        _z, analytic, rows = ac_measurements_and_jacobian(case, state, measurement_config)
        finite = finite_difference_jacobian(
            lambda x: ac_measurement_vector(case, x, measurement_config)[0], state, epsilon
        )
        types = [row.measurement_type for row in rows]
        for measurement_type in sorted(set(types)):
            available_types.add(measurement_type)
            indices = [index for index, mtype in enumerate(types) if mtype == measurement_type]
            block_analytic = analytic[indices, :]
            block_finite = finite[indices, :]
            metrics = compare_jacobian_block(block_analytic, block_finite)
            note = (
                f"seed={PERTURBATION_SEED}"
                if label == "small_perturbed_operating_point"
                else "case voltage profile"
            )
            summary.append(
                _ac_summary_row(case_name, label, measurement_type, epsilon, metrics, note)
            )
            detail.append(
                _ac_worst_detail_row(
                    case_name,
                    label,
                    measurement_type,
                    epsilon,
                    indices,
                    block_analytic,
                    block_finite,
                    metrics,
                )
            )
    # AC workflow has no angle/phasor-angle measurement row: record it, do not fabricate.
    if "angle" not in available_types:
        summary.append(_not_implemented_ac_summary(case_name, epsilon))
    return detail, summary


def _ac_summary_row(
    case_name: str,
    label: str,
    measurement_type: str,
    epsilon: float,
    metrics: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    return {
        "case": case_name,
        "operating_point_label": label,
        "measurement_type": measurement_type,
        "num_rows_checked": metrics["num_rows"],
        "num_state_columns_checked": metrics["num_cols"],
        "max_absolute_error": metrics["max_absolute_error"],
        "max_relative_error": metrics["max_relative_error"],
        "median_relative_error": metrics["median_relative_error"],
        "frobenius_relative_error": metrics["frobenius_relative_error"],
        "pass_threshold": FROBENIUS_THRESHOLD,
        "status": metrics["status"],
        "notes": note,
    }


def _ac_worst_detail_row(
    case_name: str,
    label: str,
    measurement_type: str,
    epsilon: float,
    indices: list[int],
    block_analytic: FloatArray,
    block_finite: FloatArray,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    abs_err = np.abs(block_analytic - block_finite)
    local_row, state_index = np.unravel_index(int(np.argmax(abs_err)), abs_err.shape)
    global_row = indices[int(local_row)]
    fd_value = float(block_finite[local_row, state_index])
    analytic_value = float(block_analytic[local_row, state_index])
    row_denom = max(1.0, float(np.linalg.norm(block_finite[local_row])))
    row_rel = float(np.linalg.norm(block_analytic[local_row] - block_finite[local_row]) / row_denom)
    return {
        "case": case_name,
        "operating_point_label": label,
        "epsilon": epsilon,
        "measurement_type": measurement_type,
        "row_index": int(global_row),
        "state_index": int(state_index),
        "analytic_value": analytic_value,
        "finite_difference_value": fd_value,
        "absolute_error": abs(analytic_value - fd_value),
        "relative_error": abs(analytic_value - fd_value) / max(1.0, abs(fd_value)),
        "row_relative_error": row_rel,
        "case_relative_frobenius_error": metrics["frobenius_relative_error"],
        "status": metrics["status"],
        "source_function": _AC_SOURCE_FUNCTION,
        "notes": "worst absolute-error element in this measurement-type block",
    }


def _not_implemented_ac_summary(case_name: str, epsilon: float) -> dict[str, Any]:
    return {
        "case": case_name,
        "operating_point_label": "base_operating_point",
        "measurement_type": "angle_row",
        "num_rows_checked": 0,
        "num_state_columns_checked": 0,
        "max_absolute_error": "",
        "max_relative_error": "",
        "median_relative_error": "",
        "frobenius_relative_error": "",
        "pass_threshold": FROBENIUS_THRESHOLD,
        "status": "not_implemented",
        "notes": "no angle/phasor-angle measurement row in the AC workflow",
    }


# ---------------------------------------------------------------------------
# DC finite-difference Jacobian validation (Phase 3)
# ---------------------------------------------------------------------------
def _dc_measurement_config(case: Any) -> dict[str, Any]:
    config = dict(DC_MEASUREMENT_BASE)
    # Exercise the dc_angle_row path on two non-slack state buses when available.
    config["angle_buses"] = list(case.state_buses[:2])
    return config


def _dc_measurement_vector(matrix: FloatArray, theta: FloatArray) -> FloatArray:
    return np.asarray(matrix, dtype=np.float64) @ np.asarray(theta, dtype=np.float64)


def _validate_dc_case(
    case_name: str, case_source: str, epsilon: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case = load_dc_case(case_name, case_source=case_source)
    measurement_config = _dc_measurement_config(case)
    matrix, rows = build_dc_measurement_matrix(case=case, measurement_config=measurement_config)
    types = [row.measurement_type for row in rows]
    type_label = {
        "branch_flow": "dc_branch_flow",
        "bus_injection": "dc_bus_injection",
        "angle": "dc_angle_row",
    }
    n_states = matrix.shape[1]
    rng = np.random.default_rng(PERTURBATION_SEED)
    operating_points = {
        "base_operating_point": np.zeros(n_states, dtype=np.float64),
        "small_perturbed_operating_point": rng.normal(0.0, PERTURBATION_STD, size=n_states),
    }
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for label, theta in operating_points.items():
        finite = finite_difference_jacobian(
            lambda x: _dc_measurement_vector(matrix, x), theta, epsilon
        )
        for measurement_type in sorted(set(types)):
            indices = [index for index, mtype in enumerate(types) if mtype == measurement_type]
            block_analytic = matrix[indices, :]
            block_finite = finite[indices, :]
            metrics = compare_jacobian_block(block_analytic, block_finite)
            note = "DC model is linear; analytic Jacobian = H (operating-point invariant)"
            summary.append(
                _dc_summary_row(
                    case_name, label, type_label[measurement_type], epsilon, metrics, note
                )
            )
            detail.append(
                _dc_worst_detail_row(
                    case_name,
                    label,
                    type_label[measurement_type],
                    epsilon,
                    indices,
                    block_analytic,
                    block_finite,
                    metrics,
                )
            )
    for missing_type in ("branch_flow", "bus_injection", "angle"):
        if missing_type not in types:
            summary.append(
                _not_implemented_dc_summary(case_name, type_label[missing_type], epsilon)
            )
    return detail, summary


def _dc_summary_row(
    case_name: str,
    label: str,
    measurement_type: str,
    epsilon: float,
    metrics: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    row = _ac_summary_row(case_name, label, measurement_type, epsilon, metrics, note)
    return {"workflow": "dc", **row}


def _dc_worst_detail_row(
    case_name: str,
    label: str,
    measurement_type: str,
    epsilon: float,
    indices: list[int],
    block_analytic: FloatArray,
    block_finite: FloatArray,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = _ac_worst_detail_row(
        case_name, label, measurement_type, epsilon, indices, block_analytic, block_finite, metrics
    )
    row["source_function"] = _DC_SOURCE_FUNCTION
    return {"workflow": "dc", **row}


def _not_implemented_dc_summary(
    case_name: str, measurement_type: str, epsilon: float
) -> dict[str, Any]:
    return {
        "workflow": "dc",
        "case": case_name,
        "operating_point_label": "base_operating_point",
        "measurement_type": measurement_type,
        "num_rows_checked": 0,
        "num_state_columns_checked": 0,
        "max_absolute_error": "",
        "max_relative_error": "",
        "median_relative_error": "",
        "frobenius_relative_error": "",
        "pass_threshold": FROBENIUS_THRESHOLD,
        "status": "not_implemented",
        "notes": "measurement row type not generated by the default DC config",
    }


# ---------------------------------------------------------------------------
# Weighted Jacobian consistency audit (Phase 4)
# ---------------------------------------------------------------------------
def _weighted_audit_row(
    case_name: str, workflow: str, jacobian: FloatArray, stds: FloatArray
) -> dict[str, Any]:
    invalid = has_nonpositive_variance(stds)
    sigma_min_u, sigma_max_u, cond_u = _singular_extremes(jacobian)
    if invalid:
        sigma_min_w = sigma_max_w = cond_w = float("nan")
        status = "invalid_variance"
        note = "non-positive or non-finite measurement variance detected; weighting undefined"
    else:
        tilde = weighted_jacobian(jacobian, stds)
        sigma_min_w, sigma_max_w, cond_w = _singular_extremes(tilde)
        status = "consistent"
        note = "condition_weighted uses tilde_H = R^{-1/2} H; not conflated with unweighted H"
    return {
        "case": case_name,
        "workflow": workflow,
        "measurement_type": "all",
        "num_rows": int(np.asarray(jacobian).shape[0]),
        "num_states": int(np.asarray(jacobian).shape[1]),
        "sigma_min_unweighted": sigma_min_u,
        "sigma_max_unweighted": sigma_max_u,
        "condition_unweighted": cond_u,
        "sigma_min_weighted": sigma_min_w,
        "sigma_max_weighted": sigma_max_w,
        "condition_weighted": cond_w,
        "weighting_source": _WEIGHTING_SOURCE,
        "diagonal_covariance_assumed": True,
        "status": status,
        "notes": note,
    }


def _weighted_audit(cases: list[str], case_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name in cases:
        try:
            ac_case = load_ac_case(case_name, case_source=case_source)
            state = default_ac_state_vector(ac_case)
            _z, ac_jac, ac_rows = ac_measurements_and_jacobian(
                ac_case, state, dict(DEFAULT_MEASUREMENT)
            )
            ac_stds = np.array([row.std for row in ac_rows], dtype=np.float64)
            rows.append(_weighted_audit_row(case_name, "ac_linearized", ac_jac, ac_stds))
        except Exception as exc:  # record build failure, never fabricate
            rows.append(_weighted_audit_failure(case_name, "ac_linearized", exc))
        try:
            dc_case = load_dc_case(case_name, case_source=case_source)
            dc_jac, dc_rows = build_dc_measurement_matrix(
                case=dc_case, measurement_config=_dc_measurement_config(dc_case)
            )
            dc_stds = np.array([row.std for row in dc_rows], dtype=np.float64)
            rows.append(_weighted_audit_row(case_name, "dc_linearized", dc_jac, dc_stds))
        except Exception as exc:  # record build failure, never fabricate
            rows.append(_weighted_audit_failure(case_name, "dc_linearized", exc))
    return rows


def _weighted_audit_failure(case_name: str, workflow: str, exc: Exception) -> dict[str, Any]:
    return {
        "case": case_name,
        "workflow": workflow,
        "measurement_type": "all",
        "num_rows": "",
        "num_states": "",
        "sigma_min_unweighted": "",
        "sigma_max_unweighted": "",
        "condition_unweighted": "",
        "sigma_min_weighted": "",
        "sigma_max_weighted": "",
        "condition_weighted": "",
        "weighting_source": _WEIGHTING_SOURCE,
        "diagonal_covariance_assumed": True,
        "status": "build_failed",
        "notes": f"{type(exc).__name__}: {exc}",
    }


# ---------------------------------------------------------------------------
# Orchestration and output
# ---------------------------------------------------------------------------
def build_jacobian_validation(config: dict[str, Any]) -> dict[str, Any]:
    cases = list(config.get("cases", DEFAULT_CASES))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    epsilon = float(config.get("epsilon", DEFAULT_EPSILON))
    input_root = Path(config.get("input_root", "outputs"))
    output_dir = Path(config.get("output_dir", input_root / "jacobian_validation"))
    ensure_directory(output_dir)

    ac_detail: list[dict[str, Any]] = []
    ac_summary: list[dict[str, Any]] = []
    dc_detail: list[dict[str, Any]] = []
    dc_summary: list[dict[str, Any]] = []
    for case_name in cases:
        case_detail, case_summary = _safe_validate(
            _validate_ac_case, case_name, case_source, epsilon
        )
        ac_detail.extend(case_detail)
        ac_summary.extend(case_summary)
        dc_case_detail, dc_case_summary = _safe_validate(
            _validate_dc_case, case_name, case_source, epsilon
        )
        dc_detail.extend(dc_case_detail)
        dc_summary.extend(dc_case_summary)
    weighted = _weighted_audit(cases, case_source)

    return _write_outputs(
        output_dir=output_dir,
        ac_detail=ac_detail,
        ac_summary=ac_summary,
        dc_detail=dc_detail,
        dc_summary=dc_summary,
        weighted=weighted,
        input_config={
            "input_root": str(input_root),
            "output_dir": str(output_dir),
            "cases": cases,
            "case_source": case_source,
            "epsilon": epsilon,
        },
    )


def _safe_validate(
    fn: Callable[[str, str, float], tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    case_name: str,
    case_source: str,
    epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        return fn(case_name, case_source, epsilon)
    except Exception as exc:  # record case build failure honestly
        workflow = "dc" if fn is _validate_dc_case else None
        summary_row = {
            "case": case_name,
            "operating_point_label": "base_operating_point",
            "measurement_type": "all",
            "num_rows_checked": 0,
            "num_state_columns_checked": 0,
            "max_absolute_error": "",
            "max_relative_error": "",
            "median_relative_error": "",
            "frobenius_relative_error": "",
            "pass_threshold": FROBENIUS_THRESHOLD,
            "status": "build_failed",
            "notes": f"{type(exc).__name__}: {exc}",
        }
        if workflow == "dc":
            summary_row = {"workflow": "dc", **summary_row}
        return [], [summary_row]


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    return counts


def overall_status(summary_rows: list[dict[str, Any]]) -> str:
    """Aggregate per-group statuses into a single AC/DC validation status."""

    statuses = {str(row["status"]) for row in summary_rows}
    if "fail" in statuses or "build_failed" in statuses:
        return "fail"
    if statuses - {"pass", "not_implemented"}:
        return "warning"
    return "pass" if "pass" in statuses else "not_implemented"


def _max_relative_error(summary_rows: list[dict[str, Any]]) -> float:
    values = [
        float(row["frobenius_relative_error"])
        for row in summary_rows
        if isinstance(row.get("frobenius_relative_error"), int | float)
        and np.isfinite(row["frobenius_relative_error"])
    ]
    return max(values) if values else float("nan")


def _validated_cases(summary_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["case"]) for row in summary_rows if str(row["status"]) == "pass"})


def _summary_markdown(
    workflow: str,
    summary_rows: list[dict[str, Any]],
) -> str:
    counts = _status_counts(summary_rows)
    status = overall_status(summary_rows)
    max_rel = _max_relative_error(summary_rows)
    cases = _validated_cases(summary_rows)
    title = "AC" if workflow == "ac" else "DC"
    return "\n".join(
        [
            f"# {title} Finite-Difference Jacobian Validation",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            f"This is an implementation-correctness check of the {title} measurement Jacobian "
            "against central finite differences; it is not a new scientific experiment and it "
            "does not demonstrate quantum speedup or QSVT-over-Ridge superiority.",
            "",
            "The central finite-difference Jacobian is",
            "",
            r"\[",
            r"H_{ij}^{\mathrm{FD}}",
            r"\approx",
            r"\frac{",
            r"h_i(x+\epsilon e_j)-h_i(x-\epsilon e_j)",
            r"}{",
            r"2\epsilon",
            r"}.",
            r"\]",
            "",
            "and the relative Jacobian error is",
            "",
            r"\[",
            r"\varepsilon_H",
            r"=",
            r"\frac{",
            r"\|H_{\mathrm{analytic}}-H_{\mathrm{FD}}\|_F",
            r"}{",
            r"\max(1,\|H_{\mathrm{FD}}\|_F)",
            r"}.",
            r"\]",
            "",
            "## Result",
            f"- Overall status: **{status}**.",
            f"- Pass threshold: frobenius_relative_error <= {FROBENIUS_THRESHOLD:g} and "
            f"median_relative_error <= {MEDIAN_THRESHOLD:g}.",
            "- Worst relative Frobenius error across checked blocks: "
            + (f"{max_rel:.3e}." if np.isfinite(max_rel) else "unavailable."),
            f"- Cases passing: {cases or 'none'}.",
            "",
            "## Status counts",
            *[f"- {name}: {count}" for name, count in sorted(counts.items())],
            "",
            "## Notes",
            "- A `not_implemented` row records a measurement row type that the workflow does not "
            "generate (e.g. the AC angle/phasor-angle row); it is not fabricated.",
            "- `warning_near_zero_row` marks blocks whose only large relative deviations are "
            "structural near-zero entries; `warning_scaling_sensitive` marks ill-scaled blocks.",
            "- DC rows are linear, so the analytic Jacobian equals the constant measurement "
            "matrix and is operating-point invariant."
            if workflow == "dc"
            else "- AC rows are validated at a base and a deterministic small-perturbed operating "
            "point (seed recorded).",
            "",
        ]
    )


def _weighted_summary_markdown(weighted: list[dict[str, Any]]) -> str:
    invalid = [row for row in weighted if row["status"] == "invalid_variance"]
    failed = [row for row in weighted if row["status"] == "build_failed"]
    consistent = [row for row in weighted if row["status"] == "consistent"]
    return "\n".join(
        [
            "# Weighted Jacobian Consistency Audit",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "This audit confirms that the weighted Jacobian is applied consistently as",
            "",
            r"\[",
            r"\tilde H = R^{-1/2}H.",
            r"\]",
            "",
            "with diagonal measurement covariance "
            r"\(R_{ii}=\sigma_i^2\), and that the reported condition number refers to the "
            "weighted Jacobian",
            "",
            r"\[",
            r"\kappa(\tilde H)",
            r"=",
            r"\frac{\sigma_{\max}(\tilde H)}{\sigma_{\min}(\tilde H)}.",
            r"\]",
            "",
            "## Convention",
            "- Covariance: diagonal, `R = diag(sigma_i^2)`; the weighting source is "
            "`tilde_H = H / sigma` (`R^{-1/2} H`).",
            "- The weighted and unweighted condition numbers are recorded in separate columns and "
            "are not conflated.",
            "",
            "## Result",
            f"- Consistent rows: {len(consistent)}.",
            f"- Invalid-variance rows: {len(invalid)}.",
            f"- Build-failed rows: {len(failed)}.",
            "- Invalid (non-positive / non-finite) variances are reported, never silently "
            "corrected.",
            "",
        ]
    )


def _write_outputs(
    *,
    output_dir: Path,
    ac_detail: list[dict[str, Any]],
    ac_summary: list[dict[str, Any]],
    dc_detail: list[dict[str, Any]],
    dc_summary: list[dict[str, Any]],
    weighted: list[dict[str, Any]],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ac_detail_path = rows_to_table(
        ac_detail, output_dir / "ac_jacobian_finite_difference_validation.csv", AC_DETAIL_COLUMNS
    )
    ac_summary_path = rows_to_table(
        ac_summary, output_dir / "ac_jacobian_validation_summary.csv", AC_SUMMARY_COLUMNS
    )
    dc_detail_path = rows_to_table(
        dc_detail, output_dir / "dc_jacobian_finite_difference_validation.csv", DC_DETAIL_COLUMNS
    )
    dc_summary_path = rows_to_table(
        dc_summary, output_dir / "dc_jacobian_validation_summary.csv", DC_SUMMARY_COLUMNS
    )
    weighted_path = rows_to_table(
        weighted, output_dir / "weighted_jacobian_consistency_audit.csv", WEIGHTED_COLUMNS
    )
    ac_md_path = output_dir / "ac_jacobian_validation_summary.md"
    dc_md_path = output_dir / "dc_jacobian_validation_summary.md"
    weighted_md_path = output_dir / "weighted_jacobian_consistency_summary.md"
    ac_md_path.write_text(_summary_markdown("ac", ac_summary), encoding="utf-8")
    dc_md_path.write_text(_summary_markdown("dc", dc_summary), encoding="utf-8")
    weighted_md_path.write_text(_weighted_summary_markdown(weighted), encoding="utf-8")

    artifacts = {
        "ac_jacobian_finite_difference_validation": str(ac_detail_path),
        "ac_jacobian_validation_summary": str(ac_summary_path),
        "ac_jacobian_validation_summary_md": str(ac_md_path),
        "dc_jacobian_finite_difference_validation": str(dc_detail_path),
        "dc_jacobian_validation_summary": str(dc_summary_path),
        "dc_jacobian_validation_summary_md": str(dc_md_path),
        "weighted_jacobian_consistency_audit": str(weighted_path),
        "weighted_jacobian_consistency_summary": str(weighted_md_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "ac_summary": ac_summary,
        "dc_summary": dc_summary,
        "weighted": weighted,
        "ac_status": overall_status(ac_summary),
        "dc_status": overall_status(dc_summary),
        "ac_max_relative_error": _max_relative_error(ac_summary),
        "dc_max_relative_error": _max_relative_error(dc_summary),
        "ac_cases": _validated_cases(ac_summary),
        "dc_cases": _validated_cases(dc_summary),
        "artifacts": artifacts,
    }
