from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.phase_response_conventions import pennylane_qsvt_response
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.utils.io import ensure_directory, write_json

SANITY_CAVEAT = (
    "Sanity-polynomial scalar phase-response regression only. Passing these rows "
    "does not imply bounded Ridge/Tikhonov target phase validation, hardware "
    "execution, quantum speedup, quantum advantage, or QSVT superiority over Ridge."
)

SUMMARY_COLUMNS = [
    "polynomial_name",
    "degree",
    "backend_name",
    "phase_count",
    "max_error",
    "mean_error",
    "rms_error",
    "passed",
    "status",
    "failure_reason_if_any",
]


def run_phase_sanity_regression(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    dependency_available = importlib.util.find_spec("pennylane") is not None
    summary_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []

    for spec in _sanity_specs():
        if not dependency_available or bool(resolved["force_dependency_missing"]):
            summary_rows.append(_skip_row(spec, dependency_available, resolved))
            continue
        row, values = _run_sanity_spec(spec, resolved)
        summary_rows.append(row)
        value_rows.extend(values)

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    values = pd.DataFrame(value_rows)
    summary_csv = output_dir / "phase_sanity_regression_summary.csv"
    summary_json = output_dir / "phase_sanity_regression_summary.json"
    values_csv = output_dir / "phase_sanity_response_values.csv"
    summary.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": summary_rows, "caveat": SANITY_CAVEAT})
    values.to_csv(values_csv, index=False)
    manifest = write_manifest(
        output_dir,
        artifacts={
            "phase_sanity_regression_summary_csv": str(summary_csv),
            "phase_sanity_regression_summary_json": str(summary_json),
            "phase_sanity_response_values_csv": str(values_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "phase_sanity_regression_summary_csv": summary_csv,
            "phase_sanity_regression_summary_json": summary_json,
            "phase_sanity_response_values_csv": values_csv,
            "manifest": manifest,
        },
    }


def sanity_regression_passed(summary: pd.DataFrame) -> bool:
    if summary.empty or "passed" not in summary.columns:
        return False
    return bool(summary["passed"].fillna(False).all())


def _run_sanity_spec(
    spec: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import pennylane as qml  # type: ignore[import-not-found]

        coefficients = np.asarray(spec["coefficients"], dtype=np.float64)
        validate_qsvt_polynomial(
            coefficients,
            parity="odd",
            grid_size=int(config["bound_validation_grid_size"]),
            bound_tolerance=float(config["bound_tolerance"]),
        )
        phases = np.asarray(
            qml.poly_to_angles(
                coefficients,
                "QSVT",
                angle_solver=str(config["angle_solver"]),
            ),
            dtype=np.float64,
        )
        grid = np.linspace(
            -float(config["grid_abs_max"]),
            float(config["grid_abs_max"]),
            int(config["grid_size"]),
            dtype=np.float64,
        )
        target = np.polynomial.Polynomial(coefficients)(grid)
        response = pennylane_qsvt_response(
            grid,
            phases,
            phase_order=str(config["phase_order"]),
            phase_sign=str(config["phase_sign"]),
            phase_offset_rule=str(config["phase_offset_rule"]),
            signal_operator_convention=str(config["signal_operator_convention"]),
            response_component=str(config["response_component"]),
        )
        errors = np.abs(response - target)
        max_error = float(np.max(errors))
        passed = bool(max_error <= float(config["sanity_tolerance"]))
        status = "passed" if passed else "failed_validation"
        reason = "" if passed else "sanity polynomial response exceeds tolerance"
        row = {
            "polynomial_name": spec["polynomial_name"],
            "degree": int(coefficients.size - 1),
            "backend_name": f"pennylane.poly_to_angles-{qml.__version__}",
            "phase_count": int(phases.size),
            "max_error": max_error,
            "mean_error": float(np.mean(errors)),
            "rms_error": float(np.sqrt(np.mean(errors**2))),
            "passed": passed,
            "status": status,
            "failure_reason_if_any": reason,
        }
        values = [
            {
                "polynomial_name": spec["polynomial_name"],
                "degree": int(coefficients.size - 1),
                "backend_name": row["backend_name"],
                "sigma_normalized": float(sigma),
                "target_value": float(target_value),
                "phase_response_value": float(response_value),
                "pointwise_error": float(error),
            }
            for sigma, target_value, response_value, error in zip(
                grid,
                target,
                response,
                errors,
                strict=True,
            )
        ]
        return row, values
    except Exception as exc:
        return _failure_row(spec, str(exc)), []


def _sanity_specs() -> list[dict[str, Any]]:
    return [
        {"polynomial_name": "x", "coefficients": [0.0, 1.0]},
        {"polynomial_name": "0.5x", "coefficients": [0.0, 0.5]},
        {"polynomial_name": "x^3", "coefficients": [0.0, 0.0, 0.0, 1.0]},
        {
            "polynomial_name": "0.5x_plus_0.25x^3",
            "coefficients": [0.0, 0.5, 0.0, 0.25],
        },
    ]


def _skip_row(
    spec: dict[str, Any],
    dependency_available: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "polynomial_name": spec["polynomial_name"],
        "degree": int(len(spec["coefficients"]) - 1),
        "backend_name": "pennylane.poly_to_angles",
        "phase_count": 0,
        "max_error": np.nan,
        "mean_error": np.nan,
        "rms_error": np.nan,
        "passed": False,
        "status": "skipped_dependency_missing",
        "failure_reason_if_any": (
            "PennyLane unavailable"
            if not dependency_available
            else "dependency forced missing by config"
        ),
    }


def _failure_row(spec: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "polynomial_name": spec["polynomial_name"],
        "degree": int(len(spec["coefficients"]) - 1),
        "backend_name": "pennylane.poly_to_angles",
        "phase_count": 0,
        "max_error": np.nan,
        "mean_error": np.nan,
        "rms_error": np.nan,
        "passed": False,
        "status": "failed",
        "failure_reason_if_any": reason,
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": "outputs/qsvt_phase_sanity_regression",
        "sanity_tolerance": 1.0e-6,
        "grid_size": 101,
        "grid_abs_max": 0.95,
        "angle_solver": "root-finding",
        "bound_validation_grid_size": 1001,
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
    parser = argparse.ArgumentParser(description="Run QSVT sanity phase regression")
    parser.parse_args(argv)
    run = run_phase_sanity_regression()
    print(f"QSVT phase sanity regression complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
