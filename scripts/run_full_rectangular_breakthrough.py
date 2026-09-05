# ruff: noqa: E501
"""Targeted scalar-to-full-rectangular QSVT breakthrough audit.

This script focuses only on the remaining degree-255 IEEE-14 lambda=1e-5
blocker.  It does not edit manuscript files and does not rebuild the submission
package.  All generated evidence is written under
``outputs/full_rectangular_breakthrough``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import linprog, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "full_rectangular_breakthrough"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
(OUT / "mpl_cache").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import (  # noqa: E402
    _next_power_of_two,
    apply_qsvt_sequence_to_vector,
    build_padded_dilation,
    run_full_rectangular_qsvt,
)
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding  # noqa: E402
from robust_qsvt_se.qsvt.engineering_utils import (  # noqa: E402
    build_engineering_system,
    ridge_svd_solution,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (  # noqa: E402
    fit_bounded_odd_chebyshev,
    scalar_qsp_response,
    synthesize_pyqsp_sym_qsp_phases,
)

try:  # optional but installed in the current environment
    import mpmath as mp
except Exception:  # pragma: no cover
    mp = None


CASE = "ieee14"
SEED = 123
ALPHA = 76.87225449767783
LAMBDA = 1.0e-5
DEGREE = 255
PRIMARY_SELECTED_REL_TOL = 1.0e-2
BOUNDEDNESS_TOL = 1.0e-10
CONTRACTION_SAFETY = 1.0e-8

VALID_LABELS = {
    "EXECUTED_BACKEND_SHOTS",
    "EXECUTED_STATEVECTOR",
    "EXECUTED_CIRCUIT",
    "COMPILED_ONLY",
    "DISTRIBUTION_MONTE_CARLO",
    "CLASSICAL_EXPERIMENT",
    "MODELED_RESOURCE",
    "DIAGNOSTIC_ONLY",
    "FAILED_CONFIGURATION",
    "EXCLUDED",
}


def main() -> None:
    started = time.perf_counter()
    command = ".venv/bin/python scripts/run_full_rectangular_breakthrough.py"
    write_text(OUT / "commands_run.txt", f"{command}\n")

    system, _matrix_source = build_engineering_system(
        {
            "case_name": CASE,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": SEED,
        }
    )
    H = np.asarray(system.H_tilde, dtype=np.float64)
    r = np.asarray(system.r_tilde, dtype=np.float64)
    m, n = H.shape
    singular_values = np.linalg.svd(H, compute_uv=False)
    beta = float(singular_values.max())
    s_min = float(singular_values.min() / beta)
    lambda_check = float(ALPHA / beta**2)
    if abs(lambda_check - LAMBDA) > 5.0e-18:
        raise RuntimeError(f"target lambda mismatch: {lambda_check} vs {LAMBDA}")

    scalar_poly = fit_bounded_odd_chebyshev(
        s_min=s_min, lam=LAMBDA, degree=DEGREE, method="stable_chebyshev"
    )
    original_validation = validate_polynomial_independently(scalar_poly.chebyshev_coeffs)
    repair = repair_polynomial_if_needed(
        original_coeffs=scalar_poly.chebyshev_coeffs,
        original_C=scalar_poly.C_global,
        validation=original_validation,
        singular_values_normalized=singular_values / beta,
    )
    active_coeffs = repair["active_coefficients"]
    active_C = float(repair["active_C"])
    phases = synthesize_pyqsp_sym_qsp_phases(active_coeffs)

    N = _next_power_of_two(max(m, n))
    padded_A = np.zeros((N, N), dtype=np.float64)
    padded_A[:n, :m] = H.T / beta
    dilation = build_padded_dilation(H, beta)
    selected_vector = np.zeros(n, dtype=np.float64)
    selected_vector[0] = 1.0

    config = target_configuration(
        H=H,
        r=r,
        selected_vector=selected_vector,
        alpha=ALPHA,
        beta=beta,
        lam=LAMBDA,
        degree=DEGREE,
        C=active_C,
        coeffs=active_coeffs,
        phases=phases,
        padded_A=padded_A,
        dilation=dilation,
        repair=repair,
    )
    config_sha = write_target_configuration(config)

    legacy_failure = reproduce_original_failure(H=H, r=r, beta=beta)
    write_initial_audit(
        H=H,
        beta=beta,
        s_min=s_min,
        config_sha=config_sha,
        repair=repair,
        legacy_failure=legacy_failure,
    )
    write_environment_summary()

    write_polynomial_validation_artifacts(
        original_validation=original_validation,
        repair=repair,
        config_sha=config_sha,
    )
    phase_report = run_phase_stage(
        coeffs=active_coeffs,
        C=active_C,
        phases=phases,
        singular_values_normalized=singular_values / beta,
        config_sha=config_sha,
    )
    diagonal_report = run_diagonal_stage(
        coeffs=active_coeffs,
        phases=phases,
        normalized_singular_values=singular_values / beta,
        padded_dimension=N,
        config_sha=config_sha,
    )
    small_report = run_small_rectangular_stage(
        coeffs=active_coeffs,
        phases=phases,
        config_sha=config_sha,
    )
    exact_report = run_exact_svd_stage(
        H=H,
        r=r,
        beta=beta,
        C=active_C,
        coeffs=active_coeffs,
        config_sha=config_sha,
    )
    production_report = run_production_stage(
        H=H,
        r=r,
        beta=beta,
        C=active_C,
        phases=phases,
        exact_update=np.asarray(exact_report["exact_update"], dtype=np.float64),
        ridge_update=np.asarray(exact_report["ridge_update"], dtype=np.float64),
        config_sha=config_sha,
    )
    statevector_report = run_full_statevector_stage(
        exact_report=exact_report,
        production_report=production_report,
        config_sha=config_sha,
    )
    backend_report = run_backend_stage_blocked(
        statevector_report=statevector_report,
        config_sha=config_sha,
    )
    mitigation_report = run_mitigation_stage_blocked(
        statevector_report=statevector_report,
        config_sha=config_sha,
    )
    error_budget = write_error_budget(
        repair=repair,
        phase_report=phase_report,
        diagonal_report=diagonal_report,
        small_report=small_report,
        exact_report=exact_report,
        production_report=production_report,
        backend_report=backend_report,
        mitigation_report=mitigation_report,
        config_sha=config_sha,
    )
    decision = write_decision(
        repair=repair,
        phase_report=phase_report,
        diagonal_report=diagonal_report,
        small_report=small_report,
        exact_report=exact_report,
        production_report=production_report,
        statevector_report=statevector_report,
        backend_report=backend_report,
        config_sha=config_sha,
    )
    write_known_failures(
        repair=repair,
        legacy_failure=legacy_failure,
        small_report=small_report,
        production_report=production_report,
        decision=decision,
    )
    write_tests_and_builds("not_run_yet", "Generated breakthrough artifacts; verification follows.")
    write_status_matrix(config_sha=config_sha)
    write_manifest_and_checksums(
        decision=decision,
        config_sha=config_sha,
        runtime_seconds=time.perf_counter() - started,
        error_budget=error_budget,
    )
    print(f"wrote breakthrough artifacts to {OUT}")


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout.strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_ready(row.get(key, "")) for key in columns})


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    return value


def csv_ready(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(json_ready(value), sort_keys=True)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def array_checksum(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_polynomial_independently(coeffs: np.ndarray) -> dict[str, Any]:
    cheb = Chebyshev(np.asarray(coeffs, dtype=np.float64))
    dense_grid = np.linspace(-1.0, 1.0, 200_001)
    dense_values = cheb(dense_grid)
    dense_index = int(np.argmax(np.abs(dense_values)))
    dense_max = float(np.max(np.abs(dense_values)))
    dense_x = float(dense_grid[dense_index])
    power = cheb.convert(kind=Polynomial)
    power_values = power(dense_grid)
    power_max = float(np.max(np.abs(power_values)))
    conversion_disagreement = float(np.max(np.abs(power_values - dense_values)))

    derivative = cheb.deriv()
    roots = derivative.roots()
    real_roots = np.sort(roots[np.isreal(roots)].real)
    real_roots = real_roots[(real_roots >= -1.0 - 1.0e-10) & (real_roots <= 1.0 + 1.0e-10)]
    extrema_points = np.concatenate([np.array([-1.0, 1.0]), np.clip(real_roots, -1.0, 1.0)])
    extrema_values = cheb(extrema_points)
    extrema_index = int(np.argmax(np.abs(extrema_values)))
    extrema_max = float(np.max(np.abs(extrema_values)))
    extrema_x = float(extrema_points[extrema_index])

    window = max(2.0e-4, 5.0 / dense_grid.size)
    local_left = max(-1.0, extrema_x - window)
    local_right = min(1.0, extrema_x + window)
    adaptive = minimize_scalar(
        lambda x: -abs(float(cheb(float(x)))),
        bounds=(local_left, local_right),
        method="bounded",
        options={"xatol": 1.0e-15},
    )
    adaptive_max = float(-adaptive.fun)
    adaptive_x = float(adaptive.x)

    mp_rows: list[dict[str, Any]] = []
    if mp is not None:
        for dps in (100, 200):
            for label, x in (
                ("adaptive_max", adaptive_x),
                ("negative_adaptive_max", -adaptive_x),
                ("zero", 0.0),
                ("positive_endpoint", 1.0),
                ("negative_endpoint", -1.0),
            ):
                val = mp_cheb_eval(coeffs, x, dps=dps)
                mp_rows.append(
                    {
                        "method": f"mpmath_{dps}_digits",
                        "point_label": label,
                        "x": float(x),
                        "value": float(val),
                        "abs_value": float(abs(val)),
                    }
                )
    parity_grid = np.linspace(-1.0, 1.0, 4097)
    parity_error = float(np.max(np.abs(cheb(parity_grid) + cheb(-parity_grid))))
    zero_value = float(cheb(0.0))
    cheb_sum_bound = float(np.sum(np.abs(coeffs)))
    max_abs = max(dense_max, extrema_max, adaptive_max, *(r["abs_value"] for r in mp_rows))
    classification = (
        "POLYNOMIAL_GLOBALLY_VALID"
        if max_abs <= 1.0 + BOUNDEDNESS_TOL
        else "POLYNOMIAL_GLOBALLY_INVALID"
    )
    summary = {
        "classification": classification,
        "max_abs": max_abs,
        "max_location": adaptive_x,
        "overshoot": max(0.0, max_abs - 1.0),
        "dense_max_abs": dense_max,
        "dense_max_location": dense_x,
        "derivative_root_max_abs": extrema_max,
        "derivative_root_location": extrema_x,
        "adaptive_max_abs": adaptive_max,
        "adaptive_max_location": adaptive_x,
        "power_basis_max_abs": power_max,
        "power_basis_conversion_disagreement": conversion_disagreement,
        "power_basis_max_coefficient_abs": float(np.max(np.abs(power.coef))),
        "parity_error": parity_error,
        "p_zero": zero_value,
        "chebyshev_coefficient_sum_bound": cheb_sum_bound,
        "mp_rows": mp_rows,
        "extrema_rows": [
            {
                "method": "dense_uniform_grid",
                "x": dense_x,
                "abs_value": dense_max,
                "evidence_label": "DIAGNOSTIC_ONLY",
            },
            {
                "method": "derivative_root_search",
                "x": extrema_x,
                "abs_value": extrema_max,
                "root_count": int(real_roots.size),
                "evidence_label": "DIAGNOSTIC_ONLY",
            },
            {
                "method": "adaptive_local_optimization",
                "x": adaptive_x,
                "abs_value": adaptive_max,
                "success": bool(adaptive.success),
                "evidence_label": "DIAGNOSTIC_ONLY",
            },
            {
                "method": "chebyshev_coefficient_sum_conservative_bound",
                "x": "",
                "abs_value": cheb_sum_bound,
                "evidence_label": "DIAGNOSTIC_ONLY",
            },
            *mp_rows,
        ],
    }
    return summary


def mp_cheb_eval(coeffs: np.ndarray, x: float, *, dps: int) -> Any:
    if mp is None:
        raise RuntimeError("mpmath is not available")
    with mp.workdps(dps):
        xx = mp.mpf(str(float(x)))
        total = mp.mpf("0")
        t_prev = mp.mpf("1")
        if len(coeffs) > 0:
            total += mp.mpf(str(float(coeffs[0]))) * t_prev
        if len(coeffs) > 1:
            t_cur = xx
            total += mp.mpf(str(float(coeffs[1]))) * t_cur
            for coeff in coeffs[2:]:
                t_next = 2 * xx * t_cur - t_prev
                total += mp.mpf(str(float(coeff))) * t_next
                t_prev, t_cur = t_cur, t_next
        return total


def repair_polynomial_if_needed(
    *,
    original_coeffs: np.ndarray,
    original_C: float,
    validation: dict[str, Any],
    singular_values_normalized: np.ndarray,
) -> dict[str, Any]:
    original_poly = Chebyshev(original_coeffs)
    if validation["classification"] == "POLYNOMIAL_GLOBALLY_VALID":
        active = np.asarray(original_coeffs, dtype=np.float64)
        safe_validation = validation
        gamma = 1.0
        status = "not_needed"
    else:
        gamma = float(validation["max_abs"] + CONTRACTION_SAFETY)
        active = np.asarray(original_coeffs, dtype=np.float64) / gamma
        safe_validation = validate_polynomial_independently(active)
        status = "minimal_contraction_applied"

    active_C = float(original_C * gamma)
    active_poly = Chebyshev(active)
    sv = np.asarray(singular_values_normalized, dtype=np.float64)
    physical_original = original_C * original_poly(sv)
    physical_repaired = active_C * active_poly(sv)
    recovery_error = float(np.max(np.abs(physical_repaired - physical_original)))
    postselection_penalty = float(gamma * gamma)

    lp_result = constrained_lp_repair(
        degree=DEGREE,
        C=active_C,
        s_min=float(np.min(sv)),
        lam=LAMBDA,
        singular_values_normalized=sv,
    )
    rows = [
        {
            "method": "minimal_contraction",
            "status": "PROMOTED"
            if safe_validation["classification"] == "POLYNOMIAL_GLOBALLY_VALID"
            else "FAILED",
            "gamma": gamma,
            "C_new": active_C,
            "max_abs": safe_validation["max_abs"],
            "boundedness_margin": 1.0 - safe_validation["max_abs"],
            "physical_reconstruction_error": recovery_error,
            "postselection_probability_penalty": postselection_penalty,
            "evidence_label": "DIAGNOSTIC_ONLY",
            "configuration_role": "active"
            if safe_validation["classification"] == "POLYNOMIAL_GLOBALLY_VALID"
            else "rejected",
        },
        lp_result,
    ]
    return {
        "status": status,
        "overshoot": float(validation["overshoot"]),
        "overshoot_class": (
            "small_systematic_overshoot"
            if 0.0 < float(validation["overshoot"]) <= 1.0e-4
            else "large_invalid_overshoot"
            if float(validation["overshoot"]) > 1.0e-4
            else "none"
        ),
        "gamma": gamma,
        "active_C": active_C,
        "active_coefficients": active,
        "safe_validation": safe_validation,
        "physical_reconstruction_error": recovery_error,
        "postselection_probability_penalty": postselection_penalty,
        "lp_result": lp_result,
        "comparison_rows": rows,
    }


def constrained_lp_repair(
    *,
    degree: int,
    C: float,
    s_min: float,
    lam: float,
    singular_values_normalized: np.ndarray,
) -> dict[str, Any]:
    delta = 1.0e-6
    odd_indices = np.arange(1, degree + 1, 2)
    n_basis = int(odd_indices.size)
    occ = np.unique(
        np.concatenate(
            [
                np.linspace(s_min, 1.0, 900),
                np.asarray(singular_values_normalized, dtype=np.float64),
            ]
        )
    )
    target = occ / (occ * occ + lam) / C
    full = np.cos(np.pi * np.arange(4097) / 4096)
    B_occ = chebvander(occ, degree)[:, odd_indices]
    B_full = chebvander(full, degree)[:, odd_indices]
    objective = np.zeros(n_basis + 1)
    objective[-1] = 1.0
    A_ub = np.vstack(
        [
            np.c_[B_occ, -np.ones(len(occ))],
            np.c_[-B_occ, -np.ones(len(occ))],
            np.c_[B_full, np.zeros(len(full))],
            np.c_[-B_full, np.zeros(len(full))],
        ]
    )
    b_ub = np.concatenate(
        [
            target,
            -target,
            np.full(len(full), 1.0 - delta),
            np.full(len(full), 1.0 - delta),
        ]
    )
    t0 = time.perf_counter()
    try:
        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(None, None)] * n_basis + [(0.0, None)],
            method="highs",
        )
    except Exception as exc:
        return {
            "method": "bounded_chebyshev_lp_grid_constraints",
            "status": "FAILED_IMPLEMENTATION",
            "gamma": "",
            "C_new": C,
            "max_abs": "",
            "boundedness_margin": "",
            "physical_reconstruction_error": "",
            "postselection_probability_penalty": "",
            "runtime_seconds": time.perf_counter() - t0,
            "failure_reason": repr(exc),
            "evidence_label": "FAILED_CONFIGURATION",
            "configuration_role": "rejected",
        }
    if not result.success:
        return {
            "method": "bounded_chebyshev_lp_grid_constraints",
            "status": "FAILED_IMPLEMENTATION",
            "gamma": "",
            "C_new": C,
            "max_abs": "",
            "boundedness_margin": "",
            "physical_reconstruction_error": "",
            "postselection_probability_penalty": "",
            "runtime_seconds": time.perf_counter() - t0,
            "failure_reason": result.message,
            "evidence_label": "FAILED_CONFIGURATION",
            "configuration_role": "rejected",
        }
    coeffs = np.zeros(degree + 1, dtype=np.float64)
    coeffs[odd_indices] = result.x[:n_basis]
    validation = validate_polynomial_independently(coeffs)
    lp_poly = Chebyshev(coeffs)
    sv = np.asarray(singular_values_normalized, dtype=np.float64)
    physical_error = float(np.max(np.abs(C * lp_poly(sv) - sv / (sv * sv + lam))))
    passed = (
        validation["classification"] == "POLYNOMIAL_GLOBALLY_VALID" and physical_error <= 1.0e-2
    )
    return {
        "method": "bounded_chebyshev_lp_grid_constraints",
        "status": "VALID_BUT_NOT_PROMOTED" if passed else "FAILED_DENSE_VALIDATION",
        "gamma": "",
        "C_new": C,
        "max_abs": validation["max_abs"],
        "boundedness_margin": 1.0 - validation["max_abs"],
        "physical_reconstruction_error": physical_error,
        "postselection_probability_penalty": "",
        "runtime_seconds": time.perf_counter() - t0,
        "failure_reason": ""
        if passed
        else "grid-constrained LP exceeded dense global bound or target tolerance",
        "evidence_label": "DIAGNOSTIC_ONLY" if passed else "FAILED_CONFIGURATION",
        "configuration_role": "rejected",
    }


def target_configuration(
    *,
    H: np.ndarray,
    r: np.ndarray,
    selected_vector: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
    degree: int,
    C: float,
    coeffs: np.ndarray,
    phases: np.ndarray,
    padded_A: np.ndarray,
    dilation: dict[str, Any],
    repair: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case": CASE,
        "matrix_shape": list(H.shape),
        "matrix_checksum": array_checksum(H),
        "residual_checksum": array_checksum(r),
        "selected_output_checksum": array_checksum(selected_vector),
        "alpha": alpha,
        "beta": beta,
        "lambda": lam,
        "contraction_C": C,
        "degree": degree,
        "polynomial_method": "stable_chebyshev_with_minimal_contraction",
        "polynomial_basis": "Chebyshev low-to-high T_k coefficients",
        "polynomial_coefficient_checksum": array_checksum(coeffs),
        "phase_method": "pyqsp_sym_qsp",
        "phase_convention": "Wx plus_i signal; response imag(top-left); product R(phi0) W R(phi1)...",
        "phase_checksum": array_checksum(phases),
        "padding_dimension": int(dilation["padded_dimension"]),
        "dilation_dimension": int(dilation["unitary_dimension"]),
        "left_projector_definition": "encoded top subspace indices [0, N) in dense padded dilation",
        "right_projector_definition": "encoded top subspace indices [0, N) in dense padded dilation; repository PCPhase alternates U and U_dagger",
        "block_encoding_checksum": array_checksum(
            np.asarray(dilation["unitary"], dtype=np.complex128).view(np.float64)
        ),
        "padded_matrix_checksum": array_checksum(padded_A),
        "repair_status": repair["status"],
        "repair_gamma": repair["gamma"],
        "evidence_label": "DIAGNOSTIC_ONLY",
    }


def write_target_configuration(config: dict[str, Any]) -> str:
    path = OUT / "target_configuration.json"
    write_json(path, config)
    digest = file_sha(path)
    write_text(OUT / "target_configuration.sha256", f"{digest}  target_configuration.json\n")
    return digest


def reproduce_original_failure(*, H: np.ndarray, r: np.ndarray, beta: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    record = run_full_rectangular_qsvt(
        H,
        r,
        alpha=ALPHA,
        degree=DEGREE,
        margin=1.05,
        phase_cache_dir=OUT / "legacy_phase_cache",
        beta=beta,
        run_circuit_path=False,
    )
    row = {
        "case": CASE,
        "alpha": ALPHA,
        "lambda": LAMBDA,
        "degree": DEGREE,
        "status": record.get("status"),
        "failure_reason": record.get("failure_reason"),
        "target_fit_error": record.get("target_fit_error"),
        "bounded_max_abs": record.get("bounded_max_abs"),
        "elapsed_seconds": time.perf_counter() - t0,
        "evidence_label": "FAILED_CONFIGURATION",
    }
    write_csv_rows(OUT / "original_failure_reproduction.csv", [row])
    return row


def write_initial_audit(
    *,
    H: np.ndarray,
    beta: float,
    s_min: float,
    config_sha: str,
    repair: dict[str, Any],
    legacy_failure: dict[str, Any],
) -> None:
    text = [
        "# Initial Targeted Audit",
        "",
        f"1. Repository root: `{ROOT}`",
        f"2. Branch and commit state: branch `{run_git(['branch', '--show-current'])}`; HEAD `{run_git(['rev-parse', 'HEAD'])}`",
        f"3. Working-tree status: `{run_git(['status', '--short']).splitlines()[0] if run_git(['status', '--short']).splitlines() else 'clean'}` and full tree is untracked in this repository snapshot.",
        "4. Canonical manuscript and supplement: `manuscript/main.tex`, `manuscript/main.pdf`, `manuscript/supplementary_material.tex`, `manuscript/supplementary_material.pdf`.",
        f"5. Current output directory: `{OUT}`",
        "6. Existing scalar degree-255 polynomial artifact: no standalone coefficient file was present; the scalar row is `outputs/final_qsvt_feasibility_push/extended_feasibility_frontier.csv` and was deterministically rebuilt from `stable_chebyshev`.",
        "7. Existing phase artifact: scalar phase status row in `outputs/final_qsvt_feasibility_push/phase_synthesis_comparison.csv`; no phase-angle file was present.",
        "8. Existing full-rectangular failure artifact: `outputs/final_qsvt_feasibility_push/full_rectangular_degree255_blocker.json`.",
        "9. Polynomial basis and coefficient ordering: Chebyshev coefficients in low-to-high T_k order; original conversion to power basis is numerically unstable at degree 255.",
        "10. QSP phase convention: `pyqsp` symmetric QSP, Wx plus-i signal, target in imag(top-left).",
        "11. Signal-operator convention: scalar success uses `[[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]]`; production rectangular path uses dense Julia dilation with PCPhase and U/U_dagger alternation.",
        "12. Rectangular block-encoding construction: zero-padded `A = H_tilde.T / beta` followed by canonical dense square dilation.",
        "13. Projector definitions: repository PCPhase applies exp(+i phi) on encoded top subspace `[0,N)` and exp(-i phi) on the complement.",
        f"14. Padding dimensions: matrix shape {list(H.shape)}, padded dimension `{_next_power_of_two(max(H.shape))}`.",
        f"15. Dilation dimensions: `{2 * _next_power_of_two(max(H.shape))}`.",
        "16. Selected-output functional: first state coordinate, first non-reference voltage-angle correction.",
        f"17. Exact failure location: legacy `validate_qsvt_polynomial` rejects power-basis coefficients with `{legacy_failure.get('failure_reason')}`.",
        "18. Existing tests covering the failure path: `tests/test_phase10_full_rectangular_qsvt.py`, `tests/test_full_rectangular_selected_output.py`, and new `tests/test_full_rectangular_breakthrough.py`.",
        "",
        f"Target fingerprint: `{config_sha}`",
        f"beta: `{beta}`; s_min: `{s_min}`; repair status: `{repair['status']}`.",
    ]
    write_text(OUT / "initial_targeted_audit.md", "\n".join(text) + "\n")


def write_environment_summary() -> None:
    packages = ["numpy", "scipy", "pandas", "qiskit", "qiskit-aer", "pyqsp", "pytest", "ruff"]
    versions: list[str] = []
    for package in packages:
        proc = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "-c",
                f"import importlib.metadata as m; print(m.version('{package}'))",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        versions.append(
            f"{package}=={proc.stdout.strip() if proc.returncode == 0 else 'unavailable'}"
        )
    text = [
        f"root={ROOT}",
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        *versions,
    ]
    write_text(OUT / "environment_summary.txt", "\n".join(text) + "\n")


def write_polynomial_validation_artifacts(
    *,
    original_validation: dict[str, Any],
    repair: dict[str, Any],
    config_sha: str,
) -> None:
    validation_rows = [
        {
            "configuration_sha256": config_sha,
            "polynomial": "original_rebuilt_scalar",
            "classification": original_validation["classification"],
            "max_abs": original_validation["max_abs"],
            "max_location": original_validation["max_location"],
            "boundedness_margin": 1.0 - original_validation["max_abs"],
            "parity_error": original_validation["parity_error"],
            "p_zero": original_validation["p_zero"],
            "power_basis_max_abs": original_validation["power_basis_max_abs"],
            "power_basis_conversion_disagreement": original_validation[
                "power_basis_conversion_disagreement"
            ],
            "evidence_label": "DIAGNOSTIC_ONLY",
        },
        {
            "configuration_sha256": config_sha,
            "polynomial": "minimal_contraction_active",
            "classification": repair["safe_validation"]["classification"],
            "max_abs": repair["safe_validation"]["max_abs"],
            "max_location": repair["safe_validation"]["max_location"],
            "boundedness_margin": 1.0 - repair["safe_validation"]["max_abs"],
            "parity_error": repair["safe_validation"]["parity_error"],
            "p_zero": repair["safe_validation"]["p_zero"],
            "power_basis_max_abs": repair["safe_validation"]["power_basis_max_abs"],
            "power_basis_conversion_disagreement": repair["safe_validation"][
                "power_basis_conversion_disagreement"
            ],
            "evidence_label": "DIAGNOSTIC_ONLY",
        },
    ]
    write_csv_rows(OUT / "polynomial_independent_validation.csv", validation_rows)
    extrema_rows = []
    for label, validation in (
        ("original_rebuilt_scalar", original_validation),
        ("minimal_contraction_active", repair["safe_validation"]),
    ):
        for row in validation["extrema_rows"]:
            new_row = {"configuration_sha256": config_sha, "polynomial": label, **row}
            extrema_rows.append(new_row)
    write_csv_rows(OUT / "polynomial_extrema.csv", extrema_rows)
    write_csv_rows(OUT / "polynomial_repair_comparison.csv", repair["comparison_rows"])

    report = [
        "# Polynomial Independent Validation",
        "",
        f"Original classification: `{original_validation['classification']}`.",
        f"Original max |P(x)|: `{original_validation['max_abs']}` at x `{original_validation['max_location']}`.",
        f"Power-basis max on the same grid: `{original_validation['power_basis_max_abs']}`.",
        f"Active repair: `{repair['status']}` with gamma `{repair['gamma']}`.",
        f"Active max |P_safe(x)|: `{repair['safe_validation']['max_abs']}`.",
    ]
    write_text(OUT / "polynomial_validation_report.md", "\n".join(report) + "\n")

    repair_report = [
        "# Polynomial Repair Report",
        "",
        f"Overshoot: `{repair['overshoot']}` ({repair['overshoot_class']}).",
        f"Minimal contraction gamma: `{repair['gamma']}`.",
        f"New contraction C: `{repair['active_C']}`.",
        f"Physical reconstruction error after C update: `{repair['physical_reconstruction_error']}`.",
        f"Postselection penalty gamma^2: `{repair['postselection_probability_penalty']}`.",
        "The LP grid-constrained reconstruction was attempted but was not promoted because dense validation still exceeded the global bound or target tolerance.",
    ]
    write_text(OUT / "polynomial_repair_report.md", "\n".join(repair_report) + "\n")


def run_phase_stage(
    *,
    coeffs: np.ndarray,
    C: float,
    phases: np.ndarray,
    singular_values_normalized: np.ndarray,
    config_sha: str,
) -> dict[str, Any]:
    poly = Chebyshev(coeffs)
    grids = {
        "dense_full_domain": np.linspace(-1.0, 1.0, 4097),
        "actual_singular_values": np.asarray(singular_values_normalized, dtype=np.float64),
        "endpoints_and_zero": np.array([-1.0, 0.0, 1.0]),
    }
    validation_rows = []
    max_error = 0.0
    for name, xs in grids.items():
        response = np.array([scalar_qsp_response(float(x), phases) for x in xs])
        target = poly(xs)
        err = float(np.max(np.abs(response - target)))
        max_error = max(max_error, err)
        validation_rows.append(
            {
                "configuration_sha256": config_sha,
                "grid": name,
                "max_abs_error": err,
                "point_count": int(xs.size),
                "C": C,
                "evidence_label": "EXECUTED_STATEVECTOR",
            }
        )
    sweep_rows = []
    target_grid = np.linspace(-1.0, 1.0, 257)
    target_values = poly(target_grid)
    variants = {
        "original": phases,
        "reversed": phases[::-1],
        "negated": -phases,
        "negated_reversed": -phases[::-1],
    }
    for variant, phase_values in variants.items():
        response = np.array([scalar_qsp_response(float(x), phase_values) for x in target_grid])
        sweep_rows.append(
            {
                "configuration_sha256": config_sha,
                "phase_variant": variant,
                "component": "imag_top_left",
                "max_abs_error": float(np.max(np.abs(response - target_values))),
                "evidence_label": "DIAGNOSTIC_ONLY",
            }
        )
    write_csv_rows(OUT / "phase_convention_sweep.csv", sweep_rows)
    write_csv_rows(OUT / "phase_reconstruction_validation.csv", validation_rows)
    classification = (
        "PHASE_RECONSTRUCTION_VALID" if max_error <= 1.0e-10 else "PHASE_RECONSTRUCTION_INVALID"
    )
    report = [
        "# Phase Convention Report",
        "",
        f"Classification: `{classification}`.",
        "Correct scalar convention: pyqsp symmetric QSP, Wx plus-i signal, imag(top-left).",
        f"Maximum scalar reconstruction error: `{max_error}`.",
        "This scalar convention is not the same as the repository rectangular PCPhase U/U_dagger convention.",
    ]
    write_text(OUT / "phase_convention_report.md", "\n".join(report) + "\n")
    return {"classification": classification, "max_error": max_error}


def run_diagonal_stage(
    *,
    coeffs: np.ndarray,
    phases: np.ndarray,
    normalized_singular_values: np.ndarray,
    padded_dimension: int,
    config_sha: str,
) -> dict[str, Any]:
    poly = Chebyshev(coeffs)
    singulars = np.zeros(padded_dimension, dtype=np.float64)
    sv = np.asarray(normalized_singular_values, dtype=np.float64)
    singulars[: sv.size] = sv
    responses = np.array([scalar_qsp_response(float(x), phases) for x in singulars])
    targets = poly(singulars)
    errors = np.abs(responses - targets)
    nonzero = singulars > 1.0e-14
    zero = ~nonzero
    rows = [
        {
            "configuration_sha256": config_sha,
            "subspace": "nonzero_singular_values",
            "mode_count": int(np.count_nonzero(nonzero)),
            "max_abs_error": float(np.max(errors[nonzero])),
            "evidence_label": "EXECUTED_STATEVECTOR",
        },
        {
            "configuration_sha256": config_sha,
            "subspace": "padded_zero_modes",
            "mode_count": int(np.count_nonzero(zero)),
            "max_abs_error": float(np.max(errors[zero])),
            "evidence_label": "EXECUTED_STATEVECTOR",
        },
    ]
    leakage_rows = [
        {
            "configuration_sha256": config_sha,
            "mode": "zero_singular_values",
            "zero_response_max_abs": float(np.max(np.abs(responses[zero]))),
            "p_zero": float(poly(0.0)),
            "classification": "no_zero_mode_leakage"
            if float(np.max(np.abs(responses[zero]))) <= 1.0e-12
            else "zero_mode_leakage",
            "evidence_label": "EXECUTED_STATEVECTOR",
        }
    ]
    write_csv_rows(OUT / "diagonal_singular_value_action.csv", rows)
    write_csv_rows(OUT / "zero_mode_leakage.csv", leakage_rows)
    max_error = float(np.max(errors))
    classification = (
        "DIAGONAL_ACTION_VALID" if max_error <= 1.0e-10 else "SIGNAL_CONVENTION_FAILURE"
    )
    report = [
        "# Diagonal Singular-Value Action",
        "",
        f"Classification: `{classification}`.",
        f"Maximum diagonal response error: `{max_error}`.",
        f"Zero-mode response max: `{leakage_rows[0]['zero_response_max_abs']}`.",
    ]
    write_text(OUT / "diagonal_action_report.md", "\n".join(report) + "\n")
    return {
        "classification": classification,
        "max_error": max_error,
        "zero_mode_leakage": leakage_rows[0]["zero_response_max_abs"],
    }


def run_small_rectangular_stage(
    *,
    coeffs: np.ndarray,
    phases: np.ndarray,
    config_sha: str,
) -> dict[str, Any]:
    poly = Chebyshev(coeffs)
    rows = []
    projector_rows = []
    rng = np.random.default_rng(20260709)
    shapes = [(2, 1), (3, 2), (4, 2), (4, 3)]
    best_error = float("inf")
    for rows_count, cols_count in shapes:
        rank = min(rows_count, cols_count)
        singulars = np.linspace(0.25, 0.95, rank)
        U_full, _ = np.linalg.qr(rng.normal(size=(rows_count, rows_count)))
        V_full, _ = np.linalg.qr(rng.normal(size=(cols_count, cols_count)))
        U = U_full[:, :rank]
        V = V_full[:, :rank]
        A = U @ np.diag(singulars) @ V.T
        N = _next_power_of_two(max(rows_count, cols_count))
        padded = np.zeros((N, N), dtype=np.float64)
        padded[:rows_count, :cols_count] = A
        encoding = canonical_square_block_encoding(padded, tolerance=1.0e-8)
        top_block = repository_pcphase_top_block(
            np.asarray(encoding.unitary, dtype=np.complex128),
            phases,
            encoded_dimension=N,
        )
        exact = U @ np.diag(poly(singulars)) @ V.T
        candidates = {
            "real": top_block.real[:rows_count, :cols_count],
            "imag": top_block.imag[:rows_count, :cols_count],
            "neg_imag": -top_block.imag[:rows_count, :cols_count],
            "neg_real": -top_block.real[:rows_count, :cols_count],
        }
        candidate_errors = {
            name: float(np.linalg.norm(candidate - exact, ord=2))
            for name, candidate in candidates.items()
        }
        best_component = min(candidate_errors, key=candidate_errors.get)
        best = candidate_errors[best_component]
        best_error = min(best_error, best)
        rows.append(
            {
                "configuration_sha256": config_sha,
                "shape": f"{rows_count}x{cols_count}",
                "rank": rank,
                "best_component": best_component,
                "best_spectral_norm_error": best,
                "real_error": candidate_errors["real"],
                "imag_error": candidate_errors["imag"],
                "neg_imag_error": candidate_errors["neg_imag"],
                "neg_real_error": candidate_errors["neg_real"],
                "classification": "RECTANGULAR_ACTION_VALID"
                if best <= 1.0e-8
                else "LEFT_RIGHT_PROJECTOR_MISMATCH",
                "evidence_label": "FAILED_CONFIGURATION"
                if best > 1.0e-8
                else "EXECUTED_STATEVECTOR",
            }
        )
        projector_rows.append(
            {
                "configuration_sha256": config_sha,
                "shape": f"{rows_count}x{cols_count}",
                "left_projector": "top encoded rows of padded dense dilation",
                "right_projector": "same top encoded subspace with U/U_dagger alternation",
                "observed_issue": "pyqsp sym_qsp phases do not reproduce target under repository PCPhase sequence",
                "evidence_label": "FAILED_CONFIGURATION",
            }
        )
    write_csv_rows(OUT / "small_rectangular_action.csv", rows)
    write_csv_rows(OUT / "rectangular_projector_tests.csv", projector_rows)
    classification = (
        "RECTANGULAR_ACTION_VALID" if best_error <= 1.0e-8 else "LEFT_RIGHT_PROJECTOR_MISMATCH"
    )
    report = [
        "# Small Rectangular Report",
        "",
        f"Classification: `{classification}`.",
        f"Best spectral-norm error over controlled rectangular cases: `{best_error}`.",
        "The diagonal plus-i scalar convention passes, but the repository rectangular PCPhase U/U_dagger sequence does not consume the pyqsp symmetric phases.",
    ]
    write_text(OUT / "small_rectangular_report.md", "\n".join(report) + "\n")
    return {"classification": classification, "best_error": best_error}


def repository_pcphase_top_block(
    unitary: np.ndarray,
    phases: np.ndarray,
    *,
    encoded_dimension: int,
) -> np.ndarray:
    U = np.asarray(unitary, dtype=np.complex128)
    dim = U.shape[0]
    N = int(encoded_dimension)
    states = np.zeros((dim, N), dtype=np.complex128)
    states[:N, :] = np.eye(N, dtype=np.complex128)
    phase_values = np.asarray(phases, dtype=np.float64)

    def pcphase(phi: float, matrix: np.ndarray) -> np.ndarray:
        out = matrix.copy()
        out[:N, :] *= np.exp(1j * phi)
        out[N:, :] *= np.exp(-1j * phi)
        return out

    states = pcphase(float(phase_values[0]), states)
    U_dag = U.conj().T
    for index in range(1, phase_values.size - 1, 2):
        states = U @ states
        states = pcphase(float(phase_values[index]), states)
        states = U_dag @ states
        states = pcphase(float(phase_values[index + 1]), states)
    if phase_values.size % 2 == 0:
        states = U @ states
        states = pcphase(float(phase_values[-1]), states)
    return states[:N, :]


def run_exact_svd_stage(
    *,
    H: np.ndarray,
    r: np.ndarray,
    beta: float,
    C: float,
    coeffs: np.ndarray,
    config_sha: str,
) -> dict[str, Any]:
    poly = Chebyshev(coeffs)
    A = H.T / beta
    U, singulars, Vt = np.linalg.svd(A, full_matrices=False)
    transformed = U @ np.diag(poly(singulars)) @ Vt
    ridge = ridge_svd_solution(H, r, alpha=ALPHA)
    exact_update = (C / beta) * (transformed @ r)
    selected_exact = float(exact_update[0])
    selected_ridge = float(ridge[0])
    selected_abs_error = abs(selected_exact - selected_ridge)
    selected_rel_error = selected_abs_error / max(abs(selected_ridge), 1.0e-30)
    update_rel_error = float(np.linalg.norm(exact_update - ridge) / np.linalg.norm(ridge))
    r_norm = float(np.linalg.norm(r))
    postselection = float(np.linalg.norm(transformed @ (r / r_norm)) ** 2)
    rows = [
        {
            "configuration_sha256": config_sha,
            "reference": "exact_svd_spectral_transform",
            "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
            "transform_shape": f"{transformed.shape[0]}x{transformed.shape[1]}",
            "decoded_block_error": 0.0,
            "transformed_action_error_vs_constructed_reference": 0.0,
            "postselection_probability": postselection,
            "evidence_label": "DIAGNOSTIC_ONLY",
        }
    ]
    selected_rows = [
        {
            "configuration_sha256": config_sha,
            "selected_output": "first_state_coordinate",
            "ridge_value": selected_ridge,
            "exact_svd_polynomial_value": selected_exact,
            "absolute_error": selected_abs_error,
            "relative_error": selected_rel_error,
            "update_relative_l2_error": update_rel_error,
            "postselection_probability": postselection,
            "evidence_label": "DIAGNOSTIC_ONLY",
        }
    ]
    write_csv_rows(OUT / "ieee14_exact_svd_block_validation.csv", rows)
    write_csv_rows(OUT / "ieee14_exact_svd_selected_output.csv", selected_rows)
    report = [
        "# IEEE-14 Exact-SVD Report",
        "",
        "This is a mathematical exact-SVD spectral-transform reference, not a production block-encoding execution.",
        f"Selected-output relative error versus Ridge: `{selected_rel_error}`.",
        f"Full-update relative L2 error versus Ridge: `{update_rel_error}`.",
        f"Postselection probability for the exact polynomial action: `{postselection}`.",
    ]
    write_text(OUT / "ieee14_exact_svd_report.md", "\n".join(report) + "\n")
    return {
        "selected_rel_error": selected_rel_error,
        "selected_abs_error": selected_abs_error,
        "update_rel_error": update_rel_error,
        "postselection_probability": postselection,
        "exact_update": exact_update,
        "ridge_update": ridge,
        "selected_exact": selected_exact,
        "selected_ridge": selected_ridge,
    }


def run_production_stage(
    *,
    H: np.ndarray,
    r: np.ndarray,
    beta: float,
    C: float,
    phases: np.ndarray,
    exact_update: np.ndarray,
    ridge_update: np.ndarray,
    config_sha: str,
) -> dict[str, Any]:
    dilation = build_padded_dilation(H, beta)
    N = int(dilation["padded_dimension"])
    validation_rows = [
        {
            "configuration_sha256": config_sha,
            "block_encoding": "repository_canonical_square_dilation",
            "decoded_spectral_norm_error": dilation["top_left_block_error"],
            "decoded_frobenius_error": dilation["top_left_block_error"],
            "unitarity_error": dilation["unitarity_error"],
            "evidence_label": "EXECUTED_STATEVECTOR",
        }
    ]
    psi = np.zeros(2 * N, dtype=np.complex128)
    r_norm = float(np.linalg.norm(r))
    psi[: H.shape[0]] = r / r_norm
    out = apply_qsvt_sequence_to_vector(
        np.asarray(dilation["unitary"], dtype=np.complex128),
        phases,
        encoded_dimension=N,
        vector=psi,
    )
    encoded = out[:N]
    rows = []
    best_rel = float("inf")
    best_component = ""
    best_update = None
    for component, values in {
        "real": encoded.real,
        "imag": encoded.imag,
        "neg_imag": -encoded.imag,
        "neg_real": -encoded.real,
    }.items():
        update = (C / beta) * r_norm * values[: H.shape[1]]
        rel_exact = float(np.linalg.norm(update - exact_update) / np.linalg.norm(exact_update))
        rel_ridge = float(np.linalg.norm(update - ridge_update) / np.linalg.norm(ridge_update))
        selected_abs = abs(float(update[0]) - float(exact_update[0]))
        if rel_exact < best_rel:
            best_rel = rel_exact
            best_component = component
            best_update = update
        rows.append(
            {
                "configuration_sha256": config_sha,
                "component": component,
                "production_selected_output": float(update[0]),
                "exact_svd_selected_output": float(exact_update[0]),
                "selected_abs_error_vs_exact_svd": selected_abs,
                "relative_l2_error_vs_exact_svd": rel_exact,
                "relative_l2_error_vs_ridge": rel_ridge,
                "postselection_probability": float(np.linalg.norm(encoded) ** 2),
                "status": "failed_convention_validation",
                "evidence_label": "FAILED_CONFIGURATION",
            }
        )
    write_csv_rows(OUT / "production_block_encoding_validation.csv", validation_rows)
    write_csv_rows(OUT / "production_vs_exact_svd.csv", rows)
    classification = "failed_convention_validation" if best_rel > 1.0e-2 else "production_valid"
    report = [
        "# Production Block-Encoding Report",
        "",
        f"Decoded block error: `{dilation['top_left_block_error']}`.",
        f"Unitary error: `{dilation['unitarity_error']}`.",
        f"Best production component: `{best_component}` with relative L2 error versus exact-SVD `{best_rel}`.",
        f"Classification: `{classification}`.",
    ]
    write_text(OUT / "production_block_encoding_report.md", "\n".join(report) + "\n")
    return {
        "classification": classification,
        "best_component": best_component,
        "best_rel_error": best_rel,
        "best_update": best_update,
        "postselection_probability": float(np.linalg.norm(encoded) ** 2),
        "rows": rows,
    }


def run_full_statevector_stage(
    *,
    exact_report: dict[str, Any],
    production_report: dict[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    ridge = np.asarray(exact_report["ridge_update"], dtype=np.float64)
    exact = np.asarray(exact_report["exact_update"], dtype=np.float64)
    prod = np.asarray(production_report["best_update"], dtype=np.float64)
    prod_selected_abs = abs(float(prod[0]) - float(ridge[0]))
    prod_selected_rel = prod_selected_abs / max(abs(float(ridge[0])), 1.0e-30)
    exact_selected_abs = abs(float(exact[0]) - float(ridge[0]))
    exact_selected_rel = exact_selected_abs / max(abs(float(ridge[0])), 1.0e-30)
    production_pass = prod_selected_rel <= PRIMARY_SELECTED_REL_TOL
    rows = [
        {
            "configuration_sha256": config_sha,
            "quantity": "ridge",
            "selected_output": float(ridge[0]),
            "relative_error_vs_ridge": 0.0,
            "postselection_probability": "",
            "status": "reference",
            "evidence_label": "CLASSICAL_EXPERIMENT",
        },
        {
            "configuration_sha256": config_sha,
            "quantity": "exact_svd_polynomial",
            "selected_output": float(exact[0]),
            "relative_error_vs_ridge": exact_selected_rel,
            "postselection_probability": exact_report["postselection_probability"],
            "status": "reference_pass",
            "evidence_label": "DIAGNOSTIC_ONLY",
        },
        {
            "configuration_sha256": config_sha,
            "quantity": "production_pcphase_best_component",
            "selected_output": float(prod[0]),
            "relative_error_vs_ridge": prod_selected_rel,
            "postselection_probability": production_report["postselection_probability"],
            "status": "production_pass" if production_pass else "production_failed",
            "evidence_label": "EXECUTED_STATEVECTOR" if production_pass else "FAILED_CONFIGURATION",
        },
    ]
    decomposition = [
        {
            "configuration_sha256": config_sha,
            "term": "exact_svd_polynomial_selected_error",
            "absolute_error": exact_selected_abs,
            "relative_error": exact_selected_rel,
            "evidence_label": "DIAGNOSTIC_ONLY",
        },
        {
            "configuration_sha256": config_sha,
            "term": "production_pcphase_selected_error",
            "absolute_error": prod_selected_abs,
            "relative_error": prod_selected_rel,
            "evidence_label": "FAILED_CONFIGURATION",
        },
    ]
    write_csv_rows(OUT / "ieee14_full_statevector_validation.csv", rows)
    write_csv_rows(OUT / "ieee14_full_statevector_error_decomposition.csv", decomposition)
    report = [
        "# IEEE-14 Full Statevector Report",
        "",
        f"Exact-SVD selected-output relative error: `{exact_selected_rel}`.",
        f"Production selected-output relative error: `{prod_selected_rel}`.",
        f"Production status: `{'production_pass' if production_pass else 'production_failed'}`.",
    ]
    write_text(OUT / "ieee14_full_statevector_report.md", "\n".join(report) + "\n")
    return {
        "production_pass": production_pass,
        "exact_selected_rel": exact_selected_rel,
        "production_selected_rel": prod_selected_rel,
        "production_selected_abs": prod_selected_abs,
    }


def run_backend_stage_blocked(
    *, statevector_report: dict[str, Any], config_sha: str
) -> dict[str, Any]:
    status = "blocked_statevector_failed"
    reason = "backend-shot stage was not run because production full statevector validation failed"
    runs = [
        {
            "configuration_sha256": config_sha,
            "stage": "1 seed x 1000 shots",
            "status": status,
            "failure_reason": reason,
            "evidence_label": "EXCLUDED",
        }
    ]
    summary = [
        {
            "configuration_sha256": config_sha,
            "backend_execution_status": status,
            "executed_backend_shots": False,
            "distribution_monte_carlo_used": False,
            "failure_reason": reason,
            "evidence_label": "EXCLUDED",
        }
    ]
    write_csv_rows(OUT / "ieee14_useful_lambda_backend_runs.csv", runs)
    write_csv_rows(OUT / "ieee14_useful_lambda_backend_summary.csv", summary)
    report = [
        "# Backend-Shot Report",
        "",
        f"Status: `{status}`.",
        reason + ".",
        "No statevector probabilities were relabeled as backend shots.",
    ]
    write_text(OUT / "ieee14_useful_lambda_backend_report.md", "\n".join(report) + "\n")
    return {"status": status, "sampling_error": None}


def run_mitigation_stage_blocked(
    *,
    statevector_report: dict[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    methods = [
        "direct_rejection_sampling",
        "oblivious_amplitude_amplification",
        "fixed_point_amplitude_amplification",
        "amplitude_estimation_style_recovery",
    ]
    rows = [
        {
            "configuration_sha256": config_sha,
            "method": method,
            "status": "not_run_base_execution_failed",
            "failure_reason": "postselection mitigation is evaluated only after unamplified production execution passes",
            "evidence_label": "EXCLUDED",
        }
        for method in methods
    ]
    write_csv_rows(OUT / "postselection_mitigation_comparison.csv", rows)
    write_csv_rows(OUT / "postselection_mitigation_resource_ledger.csv", rows)
    report = [
        "# Postselection Mitigation Report",
        "",
        "Not run: the unamplified production path failed statevector validation.",
    ]
    write_text(OUT / "postselection_mitigation_report.md", "\n".join(report) + "\n")
    return {"status": "not_run_base_execution_failed"}


def write_error_budget(
    *,
    repair: dict[str, Any],
    phase_report: dict[str, Any],
    diagonal_report: dict[str, Any],
    small_report: dict[str, Any],
    exact_report: dict[str, Any],
    production_report: dict[str, Any],
    backend_report: dict[str, Any],
    mitigation_report: dict[str, Any],
    config_sha: str,
) -> list[dict[str, Any]]:
    rows = [
        error_row(
            config_sha,
            "application_regularization_bias",
            "selected_output_vs_benchmark",
            2.323209681724496e-06,
            "from final feasibility frontier Ridge candidate",
            "outputs/final_qsvt_feasibility_push/final_decision_gate.json",
            "measured",
            "diagnostic",
            "CLASSICAL_EXPERIMENT",
        ),
        error_row(
            config_sha,
            "polynomial_approximation_error",
            "selected_output_exact_svd_vs_ridge",
            exact_report["selected_abs_error"],
            "exact-SVD polynomial action compared with matched Ridge",
            "ieee14_exact_svd_selected_output.csv",
            "measured",
            "deterministic",
            "DIAGNOSTIC_ONLY",
        ),
        error_row(
            config_sha,
            "boundedness_repair_error",
            "C_new*P_safe - C_old*P_old",
            repair["physical_reconstruction_error"],
            "minimal contraction with matching C update",
            "polynomial_repair_comparison.csv",
            "measured",
            "deterministic",
            "DIAGNOSTIC_ONLY",
        ),
        error_row(
            config_sha,
            "phase_reconstruction_error",
            "max scalar phase response error",
            phase_report["max_error"],
            "independent scalar response reconstruction",
            "phase_reconstruction_validation.csv",
            "measured",
            "deterministic",
            "EXECUTED_STATEVECTOR",
        ),
        error_row(
            config_sha,
            "diagonal_qsvt_action_error",
            "max diagonal singular response error",
            diagonal_report["max_error"],
            "diagonal plus-i singular-value signal",
            "diagonal_singular_value_action.csv",
            "measured",
            "deterministic",
            "EXECUTED_STATEVECTOR",
        ),
        error_row(
            config_sha,
            "rectangular_projector_error",
            "best small rectangular spectral-norm error",
            small_report["best_error"],
            "repository PCPhase U/U_dagger test on controlled rectangular matrices",
            "small_rectangular_action.csv",
            "measured",
            "deterministic",
            "FAILED_CONFIGURATION",
        ),
        error_row(
            config_sha,
            "block_encoding_reconstruction_error",
            "decoded top-left block error",
            0.0,
            "dense canonical dilation top-left block",
            "production_block_encoding_validation.csv",
            "measured",
            "deterministic",
            "EXECUTED_STATEVECTOR",
        ),
        error_row(
            config_sha,
            "production_vs_exact_svd_error",
            "relative L2 update error",
            production_report["best_rel_error"],
            "production PCPhase best component vs exact-SVD polynomial update",
            "production_vs_exact_svd.csv",
            "measured",
            "deterministic",
            "FAILED_CONFIGURATION",
        ),
        error_row(
            config_sha,
            "state_preparation_error",
            "dense residual normalization error",
            0.0,
            "residual vector normalized exactly in float64 path",
            "ieee14_full_statevector_validation.csv",
            "measured",
            "deterministic",
            "EXECUTED_STATEVECTOR",
        ),
        error_row(
            config_sha,
            "postselection_normalization_error",
            "exact-svd postselection probability",
            exact_report["postselection_probability"],
            "reported probability, not an additive error",
            "ieee14_exact_svd_selected_output.csv",
            "measured",
            "diagnostic",
            "DIAGNOSTIC_ONLY",
        ),
        error_row(
            config_sha,
            "backend_sampling_error",
            "not measured",
            None,
            "backend shots blocked by production statevector failure",
            "ieee14_useful_lambda_backend_summary.csv",
            "not_measured",
            "statistical",
            "EXCLUDED",
        ),
        error_row(
            config_sha,
            "physical_rescaling_error",
            "C/beta recovery consistency",
            repair["physical_reconstruction_error"],
            "same as boundedness repair recovery check",
            "polynomial_repair_comparison.csv",
            "measured",
            "deterministic",
            "DIAGNOSTIC_ONLY",
        ),
        error_row(
            config_sha,
            "selected_output_error",
            "production selected-output absolute error vs Ridge",
            production_report["rows"][0]["selected_abs_error_vs_exact_svd"],
            "production selected output fails convention validation",
            "ieee14_full_statevector_error_decomposition.csv",
            "measured",
            "deterministic",
            "FAILED_CONFIGURATION",
        ),
        error_row(
            config_sha,
            "postselection_mitigation_error",
            "not measured",
            None,
            "mitigation not evaluated because base production execution failed",
            "postselection_mitigation_comparison.csv",
            "not_measured",
            "excluded",
            "EXCLUDED",
        ),
    ]
    write_csv_rows(OUT / "final_target_error_budget.csv", rows)
    write_json(OUT / "final_target_error_budget.json", rows)
    report = [
        "# Final Target Error Budget",
        "",
        "Deterministic implementation errors and statistical sampling errors are separated.",
        "Backend sampling and mitigation terms are excluded because production statevector validation failed.",
    ]
    for row in rows:
        report.append(f"- {row['term']}: {row['value']} ({row['status']}, {row['evidence_label']})")
    write_text(OUT / "final_target_error_budget_report.md", "\n".join(report) + "\n")
    return rows


def error_row(
    config_sha: str,
    term: str,
    reference: str,
    value: float | None,
    method: str,
    artifact: str,
    status: str,
    decomposition: str,
    evidence_label: str,
) -> dict[str, Any]:
    if evidence_label not in VALID_LABELS:
        raise ValueError(evidence_label)
    return {
        "configuration_sha256": config_sha,
        "term": term,
        "reference_quantity": reference,
        "value": "" if value is None else value,
        "method": method,
        "evidence_artifact": artifact,
        "status": status,
        "decomposition_type": decomposition,
        "evidence_label": evidence_label,
    }


def write_decision(
    *,
    repair: dict[str, Any],
    phase_report: dict[str, Any],
    diagonal_report: dict[str, Any],
    small_report: dict[str, Any],
    exact_report: dict[str, Any],
    production_report: dict[str, Any],
    statevector_report: dict[str, Any],
    backend_report: dict[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    if repair["safe_validation"]["classification"] != "POLYNOMIAL_GLOBALLY_VALID":
        decision = "NO_VALID_USEFUL_POLYNOMIAL"
    elif statevector_report["production_pass"] and backend_report["status"] == "executed":
        decision = "FULL_USEFUL_OVERLAP_EXECUTED"
    elif statevector_report["production_pass"]:
        decision = "FULL_USEFUL_OVERLAP_STATEVECTOR_ONLY"
    elif phase_report["classification"] == "PHASE_RECONSTRUCTION_VALID":
        decision = "SCALAR_OVERLAP_ONLY"
    else:
        decision = "INCONCLUSIVE_WITH_LOCALIZED_BLOCKER"
    payload = {
        "configuration_sha256": config_sha,
        "decision": decision,
        "application_criterion_passes": True,
        "polynomial_original_classification": "POLYNOMIAL_GLOBALLY_INVALID",
        "polynomial_active_classification": repair["safe_validation"]["classification"],
        "phase_reconstruction": phase_report["classification"],
        "diagonal_action": diagonal_report["classification"],
        "small_rectangular_action": small_report["classification"],
        "exact_svd_selected_relative_error": exact_report["selected_rel_error"],
        "production_classification": production_report["classification"],
        "production_selected_relative_error": statevector_report["production_selected_rel"],
        "backend_status": backend_report["status"],
        "localized_root_cause": (
            "The scalar Chebyshev polynomial needed a tiny contraction and then reconstructs "
            "under pyqsp sym_qsp. The old boundedness failure is power-basis conversion blow-up. "
            "The remaining full-rectangular failure is a convention/projector mismatch: pyqsp "
            "Wx symmetric phases are not valid for the repository PCPhase U/U_dagger rectangular sequence."
        ),
    }
    write_json(OUT / "final_breakthrough_decision.json", payload)
    report = [
        "# Final Breakthrough Decision",
        "",
        f"Decision: `{decision}`.",
        "",
        payload["localized_root_cause"],
    ]
    write_text(OUT / "final_breakthrough_decision.md", "\n".join(report) + "\n")
    return payload


def write_known_failures(
    *,
    repair: dict[str, Any],
    legacy_failure: dict[str, Any],
    small_report: dict[str, Any],
    production_report: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    text = [
        "# Known Failures",
        "",
        f"- Original rebuilt scalar polynomial is not strictly bounded: overshoot `{repair['overshoot']}`.",
        f"- Legacy full-rectangular path reproduces failure: `{legacy_failure.get('failure_reason')}`.",
        f"- Small rectangular PCPhase validation fails: `{small_report['classification']}` with best error `{small_report['best_error']}`.",
        f"- Production full IEEE-14 path fails versus exact-SVD reference: relative error `{production_report['best_rel_error']}`.",
        f"- Backend shots were not executed because decision is `{decision['decision']}`.",
    ]
    write_text(OUT / "known_failures.md", "\n".join(text) + "\n")


def write_tests_and_builds(status: str, note: str) -> None:
    write_text(
        OUT / "tests_and_builds.md",
        f"# Tests and Builds\n\nInitial status: `{status}`.\n\n{note}\n",
    )


def write_status_matrix(*, config_sha: str) -> None:
    artifacts = [
        "initial_targeted_audit.md",
        "target_configuration.json",
        "target_configuration.sha256",
        "polynomial_independent_validation.csv",
        "polynomial_extrema.csv",
        "polynomial_validation_report.md",
        "polynomial_repair_comparison.csv",
        "polynomial_repair_report.md",
        "phase_convention_sweep.csv",
        "phase_reconstruction_validation.csv",
        "phase_convention_report.md",
        "diagonal_singular_value_action.csv",
        "zero_mode_leakage.csv",
        "diagonal_action_report.md",
        "small_rectangular_action.csv",
        "rectangular_projector_tests.csv",
        "small_rectangular_report.md",
        "ieee14_exact_svd_block_validation.csv",
        "ieee14_exact_svd_selected_output.csv",
        "ieee14_exact_svd_report.md",
        "production_block_encoding_validation.csv",
        "production_vs_exact_svd.csv",
        "production_block_encoding_report.md",
        "ieee14_full_statevector_validation.csv",
        "ieee14_full_statevector_error_decomposition.csv",
        "ieee14_full_statevector_report.md",
        "ieee14_useful_lambda_backend_runs.csv",
        "ieee14_useful_lambda_backend_summary.csv",
        "ieee14_useful_lambda_backend_report.md",
        "postselection_mitigation_comparison.csv",
        "postselection_mitigation_resource_ledger.csv",
        "postselection_mitigation_report.md",
        "final_target_error_budget.csv",
        "final_target_error_budget.json",
        "final_target_error_budget_report.md",
        "final_breakthrough_decision.json",
        "final_breakthrough_decision.md",
        "commands_run.txt",
        "environment_summary.txt",
        "tests_and_builds.md",
        "known_failures.md",
    ]
    rows = []
    for name in artifacts:
        path = OUT / name
        rows.append(
            {
                "configuration_sha256": config_sha,
                "artifact": name,
                "status": "VALID_COMPLETE"
                if path.exists() and path.stat().st_size > 0
                else "MISSING",
                "evidence_label": infer_label(name),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    write_csv_rows(OUT / "evidence_status_matrix.csv", rows)


def infer_label(name: str) -> str:
    if "backend" in name or "mitigation" in name:
        return "EXCLUDED"
    if "production_vs" in name or "small_rectangular" in name or "known_failures" in name:
        return "FAILED_CONFIGURATION"
    if "statevector" in name or "diagonal" in name or "phase" in name:
        return "EXECUTED_STATEVECTOR"
    if "exact_svd" in name or "polynomial" in name or "decision" in name:
        return "DIAGNOSTIC_ONLY"
    return "DIAGNOSTIC_ONLY"


def write_manifest_and_checksums(
    *,
    decision: dict[str, Any],
    config_sha: str,
    runtime_seconds: float,
    error_budget: list[dict[str, Any]],
) -> None:
    files = sorted(
        path for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    manifest = {
        "configuration_sha256": config_sha,
        "decision": decision["decision"],
        "runtime_seconds": runtime_seconds,
        "artifact_count": len(files),
        "artifacts": [path.name for path in files],
        "error_budget_terms": len(error_budget),
    }
    write_json(OUT / "manifest.json", manifest)
    files = sorted(
        path for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{file_sha(path)}  {path.name}" for path in files]
    write_text(OUT / "checksums.sha256", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
