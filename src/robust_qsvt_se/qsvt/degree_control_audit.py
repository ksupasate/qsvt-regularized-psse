from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import _configure_mpl_cache, _phase_timeout
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
)
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEGREE_CONTROL_CLAIM = (
    "This audit reports requested, constructed, synthesized, and effective QSVT "
    "degrees separately. It is degree-control evidence only; it does not claim "
    "quantum speedup, hardware execution, or QSVT superiority over Ridge/Tikhonov."
)

AUDIT_COLUMNS = [
    "requested_degree",
    "constructed_polynomial_degree",
    "synthesized_phase_degree",
    "effective_qsvt_degree",
    "phase_count",
    "cache_key",
    "cache_hit",
    "backend_name",
    "backend_status",
    "tolerance",
    "parity",
    "fallback_used",
    "failure_reason_if_any",
]


def run_degree_control_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75, 101, 151, 201],
        "seed": 123,
        "grid_size": 2048,
        "angle_solver": "iterative",
        "phase_timeout_seconds": 20,
        "bound_tolerance": 1.0e-5,
        "phase_cache_dir": "outputs/qsvt_degree_control_audit/phase_cache",
        "output_dir": "outputs/qsvt_degree_control_audit",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    cache_dir = ensure_directory(resolved.get("phase_cache_dir") or (output_dir / "phase_cache"))
    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    rows = audit_degree_grid(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        grid_size=int(resolved["grid_size"]),
        angle_solver=str(resolved["angle_solver"]),
        phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
        bound_tolerance=float(resolved["bound_tolerance"]),
        cache_dir=cache_dir,
    )
    artifacts = write_degree_control_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def audit_degree_grid(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    grid_size: int = 2048,
    angle_solver: str = "iterative",
    phase_timeout_seconds: int = 20,
    bound_tolerance: float = 1.0e-5,
    cache_dir: str | Path = "outputs/qsvt_degree_control_audit/phase_cache",
) -> list[dict[str, Any]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    B = H.T
    beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
    singular_values_A = np.linalg.svd(B / beta, compute_uv=False)
    positive = singular_values_A[singular_values_A > 1.0e-14]
    if positive.size == 0:
        raise ValueError("selected subproblem has no positive singular values")
    domain_min = min(max(1.0e-6, 0.9 * float(np.min(positive))), 0.95)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        alpha_norm = float(alpha) / beta**2
        for requested_degree in degrees:
            rows.append(
                audit_single_degree_request(
                    alpha=float(alpha),
                    alpha_norm=float(alpha_norm),
                    requested_degree=int(requested_degree),
                    domain_min=domain_min,
                    grid_size=int(grid_size),
                    angle_solver=str(angle_solver),
                    phase_timeout_seconds=int(phase_timeout_seconds),
                    bound_tolerance=float(bound_tolerance),
                    cache_dir=cache_dir,
                )
            )
    return rows


def audit_single_degree_request(
    *,
    alpha: float,
    alpha_norm: float,
    requested_degree: int,
    domain_min: float,
    grid_size: int = 2048,
    angle_solver: str = "iterative",
    phase_timeout_seconds: int = 20,
    bound_tolerance: float = 1.0e-5,
    cache_dir: str | Path = "outputs/qsvt_degree_control_audit/phase_cache",
) -> dict[str, Any]:
    backend_name = f"pennylane_poly_to_angles:{angle_solver}"
    row = _base_row(
        alpha=alpha,
        requested_degree=requested_degree,
        backend_name=backend_name,
        tolerance=bound_tolerance,
    )
    try:
        approximation = fit_odd_regularized_polynomial(
            alpha=float(alpha_norm),
            block_encoding_normalization=1.0,
            degree=int(requested_degree),
            domain_min=float(domain_min),
            domain_max=1.0,
            grid_size=max(int(grid_size), int(requested_degree) + 2),
        )
        coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64)
        constructed_degree = int(coefficients.size - 1)
        scaled_coefficients = coefficients / float(approximation.scale_factor)
        row.update(
            {
                "constructed_polynomial_degree": constructed_degree,
                "polynomial_scale_factor": float(approximation.scale_factor),
            }
        )
        validation = validate_qsvt_polynomial(
            scaled_coefficients,
            parity="odd",
            grid_size=max(8193, 128 * int(requested_degree) + 1),
            bound_tolerance=float(bound_tolerance),
        )
        row.update(
            {
                "max_abs_on_unit_interval": float(validation["max_abs_on_unit_interval"]),
            }
        )
        try:
            phase_result = synthesize_phases_with_cache(
                scaled_coefficients,
                angle_solver=str(angle_solver),
                cache_dir=cache_dir,
                cache_metadata={
                    "alpha": float(alpha),
                    "alpha_norm": float(alpha_norm),
                    "requested_degree": int(requested_degree),
                    "constructed_polynomial_degree": constructed_degree,
                    "domain_min": float(domain_min),
                    "bound_tolerance": float(bound_tolerance),
                    "scale_factor": float(approximation.scale_factor),
                },
                timeout_seconds=int(phase_timeout_seconds),
            )
        except Exception as exc:
            row.update(
                {
                    "backend_status": "failed",
                    "failure_reason_if_any": f"{type(exc).__name__}: {exc}",
                }
            )
            return row
        phase_count = int(phase_result["phase_count"])
        synthesized = max(phase_count - 1, 0)
        row.update(
            {
                "synthesized_phase_degree": synthesized,
                "effective_qsvt_degree": synthesized,
                "phase_count": phase_count,
                "cache_key": phase_result["cache_key"],
                "cache_hit": bool(phase_result["cache_hit"]),
                "backend_status": "cache_hit" if phase_result["cache_hit"] else "synthesized",
                "fallback_used": bool(synthesized != constructed_degree),
                "failure_reason_if_any": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "backend_status": "failed",
                "failure_reason_if_any": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def synthesize_phases_with_cache(
    coefficients: np.ndarray,
    *,
    angle_solver: str,
    cache_dir: str | Path,
    cache_metadata: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    values = np.asarray(coefficients, dtype=np.float64)
    cache_root = ensure_directory(cache_dir)
    cache_key = _phase_cache_key(values, angle_solver=angle_solver, metadata=cache_metadata)
    phase_path = cache_root / f"{cache_key}_phase_angles.csv"
    metadata_path = cache_root / f"{cache_key}_metadata.json"
    if phase_path.is_file() and metadata_path.is_file():
        phases = np.loadtxt(phase_path, delimiter=",", skiprows=1, usecols=1)
        phases = np.atleast_1d(np.asarray(phases, dtype=np.float64))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "phases": phases,
            "phase_count": int(phases.size),
            "cache_key": cache_key,
            "cache_hit": True,
            "metadata": metadata,
        }

    _configure_mpl_cache()
    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("PennyLane is required for QSVT phase synthesis") from exc

    with _phase_timeout(int(timeout_seconds)):
        phases = np.asarray(
            qml.poly_to_angles(values, "QSVT", angle_solver=str(angle_solver)),
            dtype=np.float64,
        )
    if phases.ndim != 1 or phases.size == 0 or not np.all(np.isfinite(phases)):
        raise RuntimeError("phase synthesis returned invalid phases")
    pd.DataFrame({"phase_index": np.arange(phases.size), "phase_angle": phases}).to_csv(
        phase_path,
        index=False,
    )
    metadata = {
        **cache_metadata,
        "phase_synthesis_backend": f"pennylane-{qml.__version__}",
        "angle_solver": str(angle_solver),
        "phase_count": int(phases.size),
        "cache_key": cache_key,
        "cache_hit": False,
    }
    write_json(metadata_path, metadata)
    return {
        "phases": phases,
        "phase_count": int(phases.size),
        "cache_key": cache_key,
        "cache_hit": False,
        "metadata": metadata,
    }


def write_degree_control_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = pd.DataFrame(rows)
    for column in AUDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    audit_path = output_dir / "degree_control_audit.csv"
    cache_path = output_dir / "phase_cache_audit.csv"
    requested_path = output_dir / "requested_vs_constructed_degree.csv"
    backend_path = output_dir / "phase_synthesis_backend_summary.json"
    interpretation_path = output_dir / "degree_control_interpretation.md"
    frame.to_csv(audit_path, index=False)
    frame[
        [
            "alpha",
            "requested_degree",
            "constructed_polynomial_degree",
            "cache_key",
            "cache_hit",
            "backend_status",
            "failure_reason_if_any",
        ]
    ].to_csv(cache_path, index=False)
    frame[
        [
            "alpha",
            "requested_degree",
            "constructed_polynomial_degree",
            "synthesized_phase_degree",
            "effective_qsvt_degree",
            "fallback_used",
            "backend_status",
        ]
    ].to_csv(requested_path, index=False)
    backend_summary = phase_backend_summary(frame)
    write_json(backend_path, backend_summary)
    interpretation_path.write_text(
        degree_control_interpretation(frame, backend_summary),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "degree_control_audit": str(audit_path),
            "phase_cache_audit": str(cache_path),
            "requested_vs_constructed_degree": str(requested_path),
            "phase_synthesis_backend_summary": str(backend_path),
            "degree_control_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=DEGREE_CONTROL_CLAIM,
    )
    return {
        "manifest": manifest,
        "degree_control_audit": audit_path,
        "phase_cache_audit": cache_path,
        "requested_vs_constructed_degree": requested_path,
        "phase_synthesis_backend_summary": backend_path,
        "degree_control_interpretation": interpretation_path,
    }


def phase_backend_summary(frame: pd.DataFrame) -> dict[str, Any]:
    completed = frame[frame["backend_status"].isin(["synthesized", "cache_hit"])].copy()
    requested = sorted({int(value) for value in frame["requested_degree"].dropna()})
    synthesized = sorted({int(value) for value in completed["synthesized_phase_degree"].dropna()})
    true_higher = [degree for degree in synthesized if degree > 35]
    failed = frame[frame["backend_status"] == "failed"]
    return {
        "requested_degrees": requested,
        "synthesized_phase_degrees": synthesized,
        "true_higher_degrees_synthesized": true_higher,
        "completed_rows": len(completed),
        "failed_rows": len(failed),
        "cache_hits": int(frame["cache_hit"].fillna(False).astype(bool).sum()),
        "fallback_rows": int(frame["fallback_used"].fillna(False).astype(bool).sum()),
        "legacy_degree_cap_cause": (
            "Earlier residual refinement used max_synthesis_degree=35 in wrappers "
            "and solve_gate_level_state_estimation_problem used min(degree, 35)."
        ),
        "current_degree_control_status": (
            "true_higher_degrees_generated"
            if true_higher
            else "backend_or_validation_limited_for_requested_high_degrees"
        ),
    }


def degree_control_interpretation(
    frame: pd.DataFrame,
    backend_summary: dict[str, Any],
) -> str:
    requested = ", ".join(str(value) for value in backend_summary["requested_degrees"])
    synthesized = ", ".join(str(value) for value in backend_summary["synthesized_phase_degrees"])
    true_higher = backend_summary["true_higher_degrees_synthesized"]
    status_line = (
        f"True higher synthesized degrees were generated: {true_higher}."
        if true_higher
        else "No true higher synthesized degrees were generated in this run."
    )
    failed = frame[frame["backend_status"] == "failed"]
    failure_lines = (
        ["- No failed phase-synthesis rows."]
        if failed.empty
        else [
            f"- degree {int(row.requested_degree)} alpha {float(row.alpha):.3g}: "
            f"{row.failure_reason_if_any}"
            for row in failed.itertuples()
        ][:12]
    )
    return "\n".join(
        [
            "# Degree-Control Audit",
            "",
            DEGREE_CONTROL_CLAIM,
            "",
            f"- Requested degrees: {requested}",
            f"- Synthesized phase degrees: {synthesized}",
            f"- {status_line}",
            "- Previous cap cause: wrappers passed `max_synthesis_degree=35` or "
            "`min(degree, 35)` into the dense phase path.",
            "- Current audit path attempts the requested degree directly and records "
            "validation/synthesis failures rather than silently falling back.",
            "",
            "## Failed Rows",
            *failure_lines,
            "",
        ]
    )


def _base_row(
    *,
    alpha: float,
    requested_degree: int,
    backend_name: str,
    tolerance: float,
) -> dict[str, Any]:
    return {
        "alpha": float(alpha),
        "requested_degree": int(requested_degree),
        "constructed_polynomial_degree": np.nan,
        "synthesized_phase_degree": np.nan,
        "effective_qsvt_degree": np.nan,
        "phase_count": np.nan,
        "cache_key": "",
        "cache_hit": False,
        "backend_name": backend_name,
        "backend_status": "not_run",
        "tolerance": float(tolerance),
        "parity": "odd",
        "fallback_used": False,
        "failure_reason_if_any": "",
        "polynomial_scale_factor": np.nan,
        "max_abs_on_unit_interval": np.nan,
    }


def _phase_cache_key(
    coefficients: np.ndarray,
    *,
    angle_solver: str,
    metadata: dict[str, Any],
) -> str:
    payload = {
        "coefficients": [float(value) for value in np.asarray(coefficients, dtype=np.float64)],
        "angle_solver": str(angle_solver),
        "metadata": metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
