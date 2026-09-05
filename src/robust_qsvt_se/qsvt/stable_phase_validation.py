from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.phase_response_conventions import pennylane_qsvt_response
from robust_qsvt_se.qsvt.phase_sanity_regression import (
    run_phase_sanity_regression,
    sanity_regression_passed,
)
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.stable_phase_candidates import build_stable_phase_candidates
from robust_qsvt_se.utils.io import ensure_directory, write_json

TARGET_TOLERANCE = 1.0e-3
VALIDATION_CAVEAT = (
    "Target-level scalar phase-response validation attempt. Passing requires a "
    "safe polynomial candidate, passing sanity-polynomial regression, and bounded "
    "Ridge/Tikhonov target phase-response error <= 1e-3. This is not hardware "
    "execution, quantum speedup, quantum advantage, or QSVT superiority over Ridge."
)

SUMMARY_COLUMNS = [
    "candidate_name",
    "alpha",
    "degree",
    "backend_name",
    "backend_version",
    "input_basis",
    "phase_count",
    "phase_order",
    "signal_convention",
    "response_component",
    "native_max_error",
    "post_conversion_max_error",
    "phase_response_max_error",
    "phase_response_mean_error",
    "phase_response_rms_error",
    "passed_1e_minus_3",
    "status",
    "failure_reason_if_any",
    "caveat",
]

PHASE_COLUMNS = [
    "candidate_name",
    "backend_name",
    "phase_index",
    "phase_angle",
]

RESPONSE_COLUMNS = [
    "candidate_name",
    "backend_name",
    "sigma_normalized",
    "target_value",
    "certified_polynomial_value",
    "phase_response_value",
]

ERROR_COLUMNS = [
    "candidate_name",
    "backend_name",
    "sigma_normalized",
    "target_value",
    "certified_polynomial_value",
    "phase_response_value",
    "phase_response_abs_error",
    "phase_minus_polynomial_abs_error",
]


def run_stable_target_phase_validation(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    candidate_dir = Path(resolved["candidate_output_dir"])
    if bool(resolved["rebuild_candidates"]) or not _candidate_outputs_exist(candidate_dir):
        build_stable_phase_candidates(
            {**resolved["candidate_config"], "output_dir": str(candidate_dir)}
        )

    sanity = run_phase_sanity_regression(
        {**resolved["sanity_config"], "output_dir": str(resolved["sanity_output_dir"])}
    )
    sanity_passed = sanity_regression_passed(sanity["summary"])

    candidate_summary = pd.read_csv(candidate_dir / "stable_phase_candidate_summary.csv")
    monomial_coefficients = pd.read_csv(candidate_dir / "candidate_coefficients_monomial.csv")
    error_grid = pd.read_csv(candidate_dir / "candidate_error_grid.csv")

    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    phase_error_rows: list[dict[str, Any]] = []

    for row in candidate_summary.itertuples(index=False):
        row_dict = row._asdict()
        if not bool(row_dict["safe_for_phase_synthesis"]):
            summary_rows.append(_skipped_candidate_row(row_dict, "skipped_candidate_safety_gate"))
            continue
        if not sanity_passed:
            summary_rows.append(
                _skipped_candidate_row(row_dict, "skipped_sanity_regression_failed")
            )
            continue
        if importlib.util.find_spec("pennylane") is None or bool(
            resolved["force_dependency_missing"]
        ):
            summary_rows.append(_skipped_candidate_row(row_dict, "skipped_backend_unavailable"))
            continue
        summary, phases, responses, errors = _validate_safe_candidate(
            candidate=row_dict,
            monomial_coefficients=monomial_coefficients,
            error_grid=error_grid,
            output_dir=output_dir,
            config=resolved,
        )
        summary_rows.append(summary)
        phase_rows.extend(phases)
        response_rows.extend(responses)
        phase_error_rows.extend(errors)

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    phases = pd.DataFrame(phase_rows, columns=PHASE_COLUMNS)
    responses = pd.DataFrame(response_rows, columns=RESPONSE_COLUMNS)
    errors = pd.DataFrame(phase_error_rows, columns=ERROR_COLUMNS)

    summary_csv = output_dir / "stable_target_phase_validation_summary.csv"
    summary_json = output_dir / "stable_target_phase_validation_summary.json"
    phases_csv = output_dir / "phase_angles.csv"
    responses_csv = output_dir / "phase_response_values.csv"
    errors_csv = output_dir / "phase_response_error_grid.csv"
    report_md = output_dir / "stable_target_phase_validation_report.md"
    summary.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": summary_rows, "caveat": VALIDATION_CAVEAT})
    phases.to_csv(phases_csv, index=False)
    responses.to_csv(responses_csv, index=False)
    errors.to_csv(errors_csv, index=False)
    report_md.write_text(
        _validation_report(summary, sanity_passed=sanity_passed),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "stable_target_phase_validation_summary_csv": str(summary_csv),
            "stable_target_phase_validation_summary_json": str(summary_json),
            "phase_angles_csv": str(phases_csv),
            "phase_response_values_csv": str(responses_csv),
            "phase_response_error_grid_csv": str(errors_csv),
            "stable_target_phase_validation_report_md": str(report_md),
            "sanity_regression_summary_csv": str(
                Path(resolved["sanity_output_dir"]) / "phase_sanity_regression_summary.csv"
            ),
            "candidate_summary_csv": str(candidate_dir / "stable_phase_candidate_summary.csv"),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "stable_target_phase_validation_summary_csv": summary_csv,
            "stable_target_phase_validation_summary_json": summary_json,
            "phase_angles_csv": phases_csv,
            "phase_response_values_csv": responses_csv,
            "phase_response_error_grid_csv": errors_csv,
            "stable_target_phase_validation_report_md": report_md,
            "manifest": manifest,
        },
    }


def _validate_safe_candidate(
    *,
    candidate: dict[str, Any],
    monomial_coefficients: pd.DataFrame,
    error_grid: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]] | dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_name = str(candidate["candidate_name"])
    backend_name = "pennylane.poly_to_angles"
    try:
        coefficients = _candidate_coefficients(monomial_coefficients, candidate_name)
        validate_qsvt_polynomial(
            coefficients,
            parity="odd",
            grid_size=int(config["bound_validation_grid_size"]),
            bound_tolerance=float(config["bound_tolerance"]),
        )
        phase_result = synthesize_pennylane_phases_cached(
            coefficients,
            angle_solver=str(config["angle_solver"]),
            cache_dir=output_dir / "phase_cache",
            cache_metadata={
                "candidate_name": candidate_name,
                "alpha": float(candidate["alpha"]),
                "degree": int(candidate["degree"]),
            },
        )
        grid = error_grid[error_grid["candidate_name"].astype(str) == candidate_name].copy()
        grid = grid.sort_values("sigma_normalized")
        sigma = grid["sigma_normalized"].to_numpy(dtype=np.float64)
        target = grid["target_value"].to_numpy(dtype=np.float64)
        polynomial = grid["converted_value"].to_numpy(dtype=np.float64)
        response = pennylane_qsvt_response(
            sigma,
            phase_result.phases,
            phase_order=str(config["phase_order"]),
            phase_sign=str(config["phase_sign"]),
            phase_offset_rule=str(config["phase_offset_rule"]),
            signal_operator_convention=str(config["signal_operator_convention"]),
            response_component=str(config["response_component"]),
        )
        phase_errors = np.abs(response - target)
        phase_minus_poly = np.abs(response - polynomial)
        max_error = float(np.max(phase_errors))
        passed = bool(max_error <= TARGET_TOLERANCE)
        status = "passed" if passed else "failed_phase_response"
        failure_reason = "" if passed else "phase response exceeds strict 1e-3 tolerance"
        backend_version = str(phase_result.metadata.get("phase_synthesis_backend", "pennylane"))
        summary = {
            "candidate_name": candidate_name,
            "alpha": float(candidate["alpha"]),
            "degree": int(candidate["degree"]),
            "backend_name": backend_name,
            "backend_version": backend_version,
            "input_basis": "monomial_power_low_to_high",
            "phase_count": int(phase_result.phases.size),
            "phase_order": str(config["phase_order"]),
            "signal_convention": str(config["signal_operator_convention"]),
            "response_component": str(config["response_component"]),
            "native_max_error": float(candidate["native_max_error"]),
            "post_conversion_max_error": float(candidate["conversion_max_error"]),
            "phase_response_max_error": max_error,
            "phase_response_mean_error": float(np.mean(phase_errors)),
            "phase_response_rms_error": float(np.sqrt(np.mean(phase_errors**2))),
            "passed_1e_minus_3": passed,
            "status": status,
            "failure_reason_if_any": failure_reason,
            "caveat": VALIDATION_CAVEAT,
        }
        phase_rows = [
            {
                "candidate_name": candidate_name,
                "backend_name": backend_name,
                "phase_index": int(index),
                "phase_angle": float(phase),
            }
            for index, phase in enumerate(phase_result.phases)
        ]
        response_rows = [
            {
                "candidate_name": candidate_name,
                "backend_name": backend_name,
                "sigma_normalized": float(x),
                "target_value": float(target_value),
                "certified_polynomial_value": float(poly_value),
                "phase_response_value": float(response_value),
            }
            for x, target_value, poly_value, response_value in zip(
                sigma,
                target,
                polynomial,
                response,
                strict=True,
            )
        ]
        error_rows = [
            {
                "candidate_name": candidate_name,
                "backend_name": backend_name,
                "sigma_normalized": float(x),
                "target_value": float(target_value),
                "certified_polynomial_value": float(poly_value),
                "phase_response_value": float(response_value),
                "phase_response_abs_error": float(error),
                "phase_minus_polynomial_abs_error": float(poly_error),
            }
            for x, target_value, poly_value, response_value, error, poly_error in zip(
                sigma,
                target,
                polynomial,
                response,
                phase_errors,
                phase_minus_poly,
                strict=True,
            )
        ]
        return summary, phase_rows, response_rows, error_rows
    except Exception as exc:
        return _failed_candidate_row(candidate, str(exc)), [], [], []


def _candidate_coefficients(frame: pd.DataFrame, candidate_name: str) -> np.ndarray:
    subset = frame[frame["candidate_name"].astype(str) == candidate_name].copy()
    if subset.empty:
        raise ValueError(f"no monomial coefficients found for {candidate_name}")
    subset = subset.sort_values("coefficient_index")
    return subset["monomial_coefficient"].to_numpy(dtype=np.float64)


def _skipped_candidate_row(candidate: dict[str, Any], status: str) -> dict[str, Any]:
    reason = str(candidate.get("failure_reason_if_any", ""))
    if status == "skipped_sanity_regression_failed":
        reason = "sanity-polynomial phase-response regression did not pass"
    elif status == "skipped_backend_unavailable":
        reason = "no safe phase backend is available"
    return {
        "candidate_name": str(candidate["candidate_name"]),
        "alpha": float(candidate["alpha"]),
        "degree": int(candidate["degree"]),
        "backend_name": "none",
        "backend_version": "",
        "input_basis": str(candidate.get("conversion_method", "")),
        "phase_count": 0,
        "phase_order": "",
        "signal_convention": "",
        "response_component": "",
        "native_max_error": float(candidate["native_max_error"]),
        "post_conversion_max_error": float(candidate["conversion_max_error"]),
        "phase_response_max_error": np.nan,
        "phase_response_mean_error": np.nan,
        "phase_response_rms_error": np.nan,
        "passed_1e_minus_3": False,
        "status": status,
        "failure_reason_if_any": reason,
        "caveat": VALIDATION_CAVEAT,
    }


def _failed_candidate_row(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    row = _skipped_candidate_row(candidate, "failed_phase_synthesis_or_response")
    row["backend_name"] = "pennylane.poly_to_angles"
    row["input_basis"] = "monomial_power_low_to_high"
    row["failure_reason_if_any"] = reason
    return row


def _validation_report(summary: pd.DataFrame, *, sanity_passed: bool) -> str:
    passed = (
        summary[summary["passed_1e_minus_3"] == True]  # noqa: E712
        if not summary.empty
        else pd.DataFrame()
    )
    if not passed.empty:
        verdict = "passed"
        message = "At least one safe bounded Ridge/Tikhonov target candidate passed."
    else:
        verdict = "unresolved"
        message = (
            "No bounded Ridge/Tikhonov target candidate passed full phase-response "
            "validation. The phase-response convention is validated on sanity "
            "polynomials, but target-level phase synthesis remains unresolved."
            if sanity_passed
            else "Sanity-polynomial phase-response regression did not pass."
        )
    lines = [
        "# Stable Target Phase Validation",
        "",
        "## Verdict",
        "",
        f"Target-level phase validation status: `{verdict}`.",
        "",
        message,
        "",
        f"Sanity-polynomial regression passed: `{sanity_passed}`.",
        "",
        "## Candidate Status",
        "",
        "| candidate | degree | backend | phase max error | passed | status |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in summary.itertuples(index=False):
        max_error = (
            ""
            if not np.isfinite(row.phase_response_max_error)
            else f"{row.phase_response_max_error:.6g}"
        )
        lines.append(
            "| "
            f"{row.candidate_name} | {row.degree} | {row.backend_name} | "
            f"{max_error} | {row.passed_1e_minus_3} | {row.status} |"
        )
    lines.extend(["", "## Claim Boundary", "", VALIDATION_CAVEAT, ""])
    return "\n".join(lines)


def _candidate_outputs_exist(candidate_dir: Path) -> bool:
    return all(
        (candidate_dir / name).is_file()
        for name in [
            "stable_phase_candidate_summary.csv",
            "candidate_coefficients_monomial.csv",
            "candidate_error_grid.csv",
        ]
    )


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": "outputs/qsvt_stable_target_phase_validation",
        "candidate_output_dir": "outputs/qsvt_stable_phase_candidates",
        "sanity_output_dir": "outputs/qsvt_phase_sanity_regression",
        "rebuild_candidates": False,
        "candidate_config": {},
        "sanity_config": {},
        "angle_solver": "root-finding",
        "bound_validation_grid_size": 2049,
        "bound_tolerance": 1.0e-5,
        "phase_order": "original",
        "phase_sign": "phi",
        "phase_offset_rule": "none",
        "signal_operator_convention": "pennylane_rx_pcphase",
        "response_component": "real_u00",
        "force_dependency_missing": False,
    }
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run stable bounded Ridge/Tikhonov target phase validation"
    )
    parser.parse_args(argv)
    run = run_stable_target_phase_validation()
    print(f"QSVT stable target phase validation complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
