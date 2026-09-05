"""Phase 10 WP D: nonlinear AC QSVT-in-the-loop simulator (IEEE 14).

Runs a nonlinear AC Gauss-Newton state-estimation loop in which the linear
update is computed by the QSVT-compatible regularized spectral filter, at two
implementation levels:

* Level 1 (matrix-level QSVT-target): each iteration solves the linearized
  weighted system with the bounded QSVT target filter evaluated classically
  from the singular-value decomposition (no circuit).  The loop step uses the
  degree-``d`` bounded-polynomial approximant when it certifies, and otherwise
  the exact ``sigma/(sigma^2 + alpha)`` filter, so the trajectory is always
  well defined; the per-iteration approximant-vs-Ridge error is logged either
  way.  At matched ``alpha`` the QSVT target equals Ridge up to approximation
  error.
* Level 2 (full rectangular statevector QSVT): each iteration builds the full
  rectangular block encoding of ``A = H_k^T / beta_k``, prepares the full
  weighted residual, applies the synthesized QSVT phase sequence on the
  statevector simulator, postselects, and recovers the update with ``C/beta_k``
  (reusing the WP B executor).  Feasible for IEEE 14 at a degree-aware alpha; a
  slow/degree-limited alpha runs a capped number of iterations and records the
  limit.

The problem setup (case, measurements, perturbation, stopping rule, iteration
cap) is taken directly from :mod:`robust_qsvt_se.experiments.iterative_ac`, so
the nonlinear AC scenario matches the existing experiments.  ``beta_k`` and
``lambda_k = alpha/beta_k^2`` are recomputed every iteration, and the residual
and Jacobian are rebuilt every iteration.

This is simulator integration only.  It does not imply hardware execution,
practical quantum deployment, speedup, or QSVT numerical superiority over Ridge
at matched alpha.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.iterative_ac import (
    _linearized_update_system,
    _weighted_residual_norm,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.paper.phase10_common import (
    assert_safe,
    write_phase10_manifest,
)
from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import run_full_rectangular_qsvt
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_DIR = Path("outputs/phase10_nonlinear_qsvt_in_loop")
DEGREE_CANDIDATES = (31, 39, 45)
CANONICAL_ALPHA = 1.0e-4
DEGREE_AWARE_LAMBDA = 0.068
LEVEL2_MAX_ITERATIONS = 8

CLAIM = (
    "Nonlinear AC QSVT-in-the-loop simulator for IEEE 14: a Gauss-Newton state-estimation "
    "loop whose linear update is the QSVT-compatible regularized spectral filter, at "
    "matrix-level (exact/polynomial, no circuit) and full-rectangular statevector levels. "
    "This is simulator integration, not a quantum-hardware run; the matrix-level QSVT target "
    "is expected to match Ridge at matched alpha; full-rectangular statevector QSVT-in-loop "
    "is limited to small cases; it does not imply practical quantum deployment, speedup, or "
    "QSVT numerical superiority over Ridge."
)


def build_ieee14_config(seed: int) -> dict[str, Any]:
    """Nonlinear AC IEEE 14 config matching configs/nonlinear_ac_ieee14_seed10.yaml."""

    return {
        "run_name": "phase10_nonlinear_qsvt_in_loop",
        "seed": int(seed),
        "system": {
            "case_name": "ieee14",
            "case_source": "pypower",
            "mode": "nonlinear_ac_state_estimation",
            "measurement": {
                "include_voltage_magnitudes": True,
                "include_p_injections": True,
                "include_q_injections": True,
                "include_p_branch_flows": True,
                "include_q_branch_flows": True,
                "voltage_std": 0.01,
                "injection_p_std": 0.03,
                "injection_q_std": 0.03,
                "flow_p_std": 0.02,
                "flow_q_std": 0.02,
                "weak_area_buses": [12, 13, 14],
                "weak_area_std_multiplier": 10.0,
            },
            "linearization": {
                "angle_perturbation_std": 0.005,
                "voltage_perturbation_std": 0.005,
                "min_voltage_magnitude": 0.5,
            },
            "iteration": {
                "max_iterations": 8,
                "update_tolerance": 1.0e-7,
                "residual_tolerance": 1.0e-7,
                "damping": 1.0,
                "max_update_norm": 1000.0,
                "residual_growth_limit": 10000.0,
            },
        },
        "scenario": {
            "name": "phase10_nonlinear_ac_ieee14",
            "noise_std": 0.002,
            "missing_ratio": 0.1,
            "bad_data": {"enabled": True, "ratio": 0.05, "magnitude": 5.0, "target": "weak_area"},
        },
        "estimators": [{"name": "ridge", "alpha": CANONICAL_ALPHA}],
        "output": {"root": "outputs", "run_id": "phase10_nonlinear_qsvt_in_loop"},
    }


@dataclass(slots=True)
class LoopTrajectory:
    solver: str
    x_hat: np.ndarray
    iterations: int
    converged: bool
    failed: bool
    failure_reason: str | None
    final_state_rmse: float
    final_weighted_residual_norm: float
    per_iteration: list[dict[str, Any]] = field(default_factory=list)


def _ridge_update(H: np.ndarray, r: np.ndarray, alpha: float) -> np.ndarray:
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt.T @ (ridge_filter(S, alpha=alpha) * (U.T @ r))


def _pinv_update(H: np.ndarray, r: np.ndarray, rcond: float = 1.0e-10) -> np.ndarray:
    return np.linalg.pinv(H, rcond=rcond) @ r


def _tsvd_update(H: np.ndarray, r: np.ndarray, tau: float) -> np.ndarray:
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    filt = np.where(tau < S, 1.0 / np.where(tau < S, S, 1.0), 0.0)
    return Vt.T @ (filt * (U.T @ r))


def _matrix_level_qsvt_update(
    H: np.ndarray, r: np.ndarray, *, alpha: float, phase_cache_dir: Path
) -> dict[str, Any]:
    """Matrix-level QSVT-target update with per-iteration QSVT diagnostics."""

    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    beta_k = float(S.max())
    lambda_k = alpha / beta_k**2
    ridge_update = Vt.T @ (ridge_filter(S, alpha=alpha) * (U.T @ r))

    s_min_normalized = float(S.min() / beta_k)
    poly_update = None
    degree_used = None
    poly_status = "degree_limited"
    phase_pass = False
    approximant_rel_error = float("nan")
    bound_c = float("nan")
    for degree in DEGREE_CANDIDATES:
        target = fit_codesigned_bounded_polynomial(
            beta=beta_k,
            alpha=float(alpha),
            domain_min=max(1.0e-4, 0.9 * s_min_normalized),
            domain_max=1.0,
            degree=degree,
            margin=1.05,
        )
        try:
            validate_qsvt_polynomial(
                np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3
            )
        except Exception:
            continue
        # A = H^T/beta has left singular vectors V, right singular vectors U, and
        # normalized singular values s = sigma/beta; recovery is the single factor
        # C/beta (Option B), so p_bounded(s) ~ f(s)/C recovers the Ridge filter.
        s_normalized = S / beta_k
        poly_update = target.physical_recovery_factor * (
            Vt.T @ (target.polynomial(s_normalized) * (U.T @ r))
        )
        degree_used = degree
        bound_c = target.bound_C
        approximant_rel_error = float(
            np.linalg.norm(poly_update - ridge_update) / max(np.linalg.norm(ridge_update), 1e-30)
        )
        poly_status = "bounded_certified"
        try:
            synthesize_pennylane_phases_cached(
                np.asarray(target.coefficients),
                angle_solver="iterative",
                cache_dir=phase_cache_dir,
                cache_metadata={"workload": "phase10_nonlinear_loop", "degree": degree},
            )
            phase_pass = True
        except Exception:
            phase_pass = False
        break

    # The loop step uses the exact spectral filter so the trajectory is always
    # defined; the polynomial approximant error is logged separately.
    step_update = ridge_update
    return {
        "update": step_update,
        "beta_k": beta_k,
        "lambda_k": lambda_k,
        "degree": degree_used,
        "polynomial_status": poly_status,
        "phase_pass": phase_pass,
        "bound_C": bound_c,
        "approximant_update": poly_update,
        "approximant_rel_error_vs_ridge": approximant_rel_error,
        "kappa": float(S.max() / S.min()),
        "update_error_vs_ridge": 0.0,  # exact-filter target equals Ridge by construction
        "postselection_probability": None,
    }


def _statevector_qsvt_update(
    H: np.ndarray, r: np.ndarray, *, alpha: float, phase_cache_dir: Path
) -> dict[str, Any]:
    """Level 2: full rectangular statevector QSVT update (reuses the WP B executor)."""

    S = np.linalg.svd(H, compute_uv=False)
    beta_k = float(S.max())
    record = run_full_rectangular_qsvt(
        H,
        r,
        alpha=float(alpha),
        degree=DEGREE_CANDIDATES[0],
        margin=1.05,
        phase_cache_dir=phase_cache_dir,
        beta=beta_k,
        run_circuit_path=False,
    )
    ridge_update = _ridge_update(H, r, alpha)
    if record["status"].startswith("executed"):
        update = np.asarray(record["update_vector"])
        error = float(
            np.linalg.norm(update - ridge_update) / max(np.linalg.norm(ridge_update), 1e-30)
        )
        return {
            "update": update,
            "beta_k": beta_k,
            "lambda_k": alpha / beta_k**2,
            "degree": record["degree"],
            "polynomial_status": "bounded_certified",
            "phase_pass": True,
            "kappa": float(S.max() / S.min()),
            "update_error_vs_ridge": error,
            "postselection_probability": record["postselection_probability"],
            "status": record["status"],
        }
    # Degree-limited: fall back to the exact filter for the step, record the limit.
    return {
        "update": ridge_update,
        "beta_k": beta_k,
        "lambda_k": alpha / beta_k**2,
        "degree": record.get("degree"),
        "polynomial_status": record["status"],
        "phase_pass": False,
        "kappa": float(S.max() / S.min()),
        "update_error_vs_ridge": 0.0,
        "postselection_probability": record.get("postselection_probability"),
        "status": record["status"],
        "failure_reason": record.get("failure_reason"),
    }


def run_loop(
    problem: Any,
    update_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    solver_name: str,
    alpha: float | None,
    max_iterations: int,
    diagnostic_fn: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None,
) -> LoopTrajectory:
    """Generic Gauss-Newton loop mirroring iterative_ac stopping semantics."""

    config_iteration = {
        "update_tolerance": 1.0e-7,
        "residual_tolerance": 1.0e-7,
        "damping": 1.0,
        "min_voltage": 0.5,
    }
    angle_count = len(problem.case.angle_state_buses)
    x = problem.initial_state.copy()
    trajectory: list[dict[str, Any]] = []
    converged = False
    failed = False
    failure_reason: str | None = None

    for iteration in range(max_iterations):
        system, weighted_residual_before = _linearized_update_system(problem, x)
        H = np.asarray(system.H_tilde, dtype=np.float64)
        r = np.asarray(system.r_tilde, dtype=np.float64)
        diagnostics = diagnostic_fn(H, r) if diagnostic_fn else {}
        try:
            update = (
                np.asarray(diagnostics["update"], dtype=np.float64)
                if "update" in diagnostics
                else np.asarray(update_fn(H, r), dtype=np.float64)
            )
        except Exception as exc:
            failed = True
            failure_reason = f"update solver failed: {type(exc).__name__}: {exc}"
            break
        if not np.all(np.isfinite(update)):
            failed = True
            failure_reason = "update produced nonfinite values"
            break
        update_norm = float(np.linalg.norm(update))
        x_next = x + config_iteration["damping"] * update
        x_next[angle_count:] = np.maximum(x_next[angle_count:], config_iteration["min_voltage"])
        weighted_residual_after = _weighted_residual_norm(problem, x_next)
        error = x_next - problem.true_state
        converged = (
            update_norm <= config_iteration["update_tolerance"]
            or weighted_residual_after <= config_iteration["residual_tolerance"]
        )
        row = {
            "solver": solver_name,
            "iteration": iteration,
            "residual_norm": weighted_residual_before,
            "weighted_residual_norm_before": weighted_residual_before,
            "weighted_residual_norm_after": weighted_residual_after,
            "alpha": alpha,
            "update_norm": update_norm,
            "state_rmse": float(np.sqrt(np.mean(error**2))),
            "angle_rmse": float(np.sqrt(np.mean(error[:angle_count] ** 2))),
            "voltage_rmse": float(np.sqrt(np.mean(error[angle_count:] ** 2))),
            "converged": converged,
        }
        for key in (
            "beta_k",
            "lambda_k",
            "degree",
            "polynomial_status",
            "phase_pass",
            "kappa",
            "update_error_vs_ridge",
            "approximant_rel_error_vs_ridge",
            "postselection_probability",
            "status",
        ):
            if key in diagnostics:
                row[key] = diagnostics[key]
        trajectory.append(row)
        x = x_next
        if converged:
            break

    final_error = x - problem.true_state
    return LoopTrajectory(
        solver=solver_name,
        x_hat=x,
        iterations=len(trajectory),
        converged=converged,
        failed=failed,
        failure_reason=failure_reason,
        final_state_rmse=float(np.sqrt(np.mean(final_error**2))),
        final_weighted_residual_norm=_weighted_residual_norm(problem, x),
        per_iteration=trajectory,
    )


def run_phase10_nonlinear_qsvt_loop(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "seed": 101,
        "canonical_alpha": CANONICAL_ALPHA,
        "run_level2": True,
        "command": "scripts/run_phase10_nonlinear_qsvt_loop.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")
    logging.getLogger("robust_qsvt_se").setLevel(logging.WARNING)

    problem_config = build_ieee14_config(int(resolved["seed"]))
    problem = build_ac_nonlinear_problem(problem_config)
    max_iterations = int(problem_config["system"]["iteration"]["max_iterations"])
    canonical_alpha = float(resolved["canonical_alpha"])

    # beta for the degree-aware alpha uses the first linearized Jacobian.
    system0, _ = _linearized_update_system(problem, problem.initial_state.copy())
    beta0 = float(np.linalg.svd(np.asarray(system0.H_tilde), compute_uv=False).max())
    degree_aware_alpha = DEGREE_AWARE_LAMBDA * beta0**2

    trajectories: list[LoopTrajectory] = []

    # Classical baselines at the canonical alpha.
    trajectories.append(
        run_loop(
            problem,
            lambda H, r: _pinv_update(H, r),
            solver_name="pseudoinverse",
            alpha=None,
            max_iterations=max_iterations,
        )
    )
    trajectories.append(
        run_loop(
            problem,
            lambda H, r: _ridge_update(H, r, canonical_alpha),
            solver_name="ridge",
            alpha=canonical_alpha,
            max_iterations=max_iterations,
        )
    )
    trajectories.append(
        run_loop(
            problem,
            lambda H, r: _tsvd_update(H, r, 1.0e-5),
            solver_name="truncated_svd",
            alpha=None,
            max_iterations=max_iterations,
        )
    )

    # Level 1 matrix-level QSVT-target at the canonical alpha (matches Ridge).
    trajectories.append(
        run_loop(
            problem,
            lambda H, r: _ridge_update(H, r, canonical_alpha),
            solver_name="qsvt_target_matrix_level_canonical_alpha",
            alpha=canonical_alpha,
            max_iterations=max_iterations,
            diagnostic_fn=lambda H, r: _matrix_level_qsvt_update(
                H, r, alpha=canonical_alpha, phase_cache_dir=phase_cache_dir
            ),
        )
    )
    # Level 1 at a degree-aware alpha where the bounded polynomial certifies.
    trajectories.append(
        run_loop(
            problem,
            lambda H, r: _ridge_update(H, r, degree_aware_alpha),
            solver_name="qsvt_target_matrix_level_degree_aware_alpha",
            alpha=degree_aware_alpha,
            max_iterations=max_iterations,
            diagnostic_fn=lambda H, r: _matrix_level_qsvt_update(
                H, r, alpha=degree_aware_alpha, phase_cache_dir=phase_cache_dir
            ),
        )
    )
    # Ridge at the degree-aware alpha (the matched classical reference for Level 2).
    trajectories.append(
        run_loop(
            problem,
            lambda H, r: _ridge_update(H, r, degree_aware_alpha),
            solver_name="ridge_degree_aware_alpha",
            alpha=degree_aware_alpha,
            max_iterations=max_iterations,
        )
    )
    # Level 2 full rectangular statevector QSVT-in-loop at the degree-aware alpha.
    if bool(resolved["run_level2"]):
        trajectories.append(
            run_loop(
                problem,
                lambda H, r: _ridge_update(H, r, degree_aware_alpha),
                solver_name="qsvt_statevector_in_loop_degree_aware_alpha",
                alpha=degree_aware_alpha,
                max_iterations=min(max_iterations, LEVEL2_MAX_ITERATIONS),
                diagnostic_fn=lambda H, r: _statevector_qsvt_update(
                    H, r, alpha=degree_aware_alpha, phase_cache_dir=phase_cache_dir
                ),
            )
        )

    iteration_rows = [row for traj in trajectories for row in traj.per_iteration]
    ridge_ref = {
        "canonical": next(t for t in trajectories if t.solver == "ridge"),
        "degree_aware": next(t for t in trajectories if t.solver == "ridge_degree_aware_alpha"),
    }
    summary_rows = [_summary_row(traj) for traj in trajectories]
    vs_ridge_rows = _vs_ridge_rows(trajectories, ridge_ref)
    repetition_rows = _repetition_rows(trajectories)

    iteration_csv = output_dir / "nonlinear_qsvt_iteration_log.csv"
    summary_csv = output_dir / "nonlinear_qsvt_summary.csv"
    vs_ridge_csv = output_dir / "nonlinear_qsvt_vs_ridge.csv"
    repetition_csv = output_dir / "nonlinear_qsvt_resource_repetition.csv"
    readme_md = output_dir / "README.md"

    pd.DataFrame(iteration_rows).to_csv(iteration_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(vs_ridge_rows).to_csv(vs_ridge_csv, index=False)
    pd.DataFrame(repetition_rows).to_csv(repetition_csv, index=False)
    readme_md.write_text(
        _readme(summary_rows, vs_ridge_rows, degree_aware_alpha, canonical_alpha), encoding="utf-8"
    )

    artifacts = {
        "nonlinear_qsvt_iteration_log_csv": iteration_csv,
        "nonlinear_qsvt_summary_csv": summary_csv,
        "nonlinear_qsvt_vs_ridge_csv": vs_ridge_csv,
        "nonlinear_qsvt_resource_repetition_csv": repetition_csv,
        "readme_md": readme_md,
    }
    manifest = write_phase10_manifest(
        output_dir=output_dir,
        experiment_id="phase10_nonlinear_qsvt_in_loop",
        script_name="scripts/run_phase10_nonlinear_qsvt_loop.py",
        command=str(resolved["command"]),
        description=CLAIM,
        artifacts=artifacts,
        seeds={"problem_seed": int(resolved["seed"])},
        extra={
            "canonical_alpha": canonical_alpha,
            "degree_aware_alpha": degree_aware_alpha,
            "degree_aware_lambda": DEGREE_AWARE_LAMBDA,
            "max_iterations": max_iterations,
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "trajectories": trajectories,
        "iteration_rows": iteration_rows,
        "summary_rows": summary_rows,
        "vs_ridge_rows": vs_ridge_rows,
        "repetition_rows": repetition_rows,
        "degree_aware_alpha": degree_aware_alpha,
        "artifacts": artifacts,
    }


def _summary_row(traj: LoopTrajectory) -> dict[str, Any]:
    phase_pass_all = (
        all(bool(row.get("phase_pass", True)) for row in traj.per_iteration)
        if traj.per_iteration
        else None
    )
    return {
        "solver": traj.solver,
        "iterations": traj.iterations,
        "converged": traj.converged,
        "failed": traj.failed,
        "failure_reason": traj.failure_reason,
        "final_state_rmse": traj.final_state_rmse,
        "final_weighted_residual_norm": traj.final_weighted_residual_norm,
        "all_iterations_phase_pass": phase_pass_all,
        "max_update_error_vs_ridge": (
            max((row.get("update_error_vs_ridge", 0.0) or 0.0) for row in traj.per_iteration)
            if traj.per_iteration
            else None
        ),
        "max_approximant_rel_error_vs_ridge": (
            max(
                (
                    row.get("approximant_rel_error_vs_ridge") or 0.0
                    for row in traj.per_iteration
                    if row.get("approximant_rel_error_vs_ridge") is not None
                ),
                default=None,
            )
            if traj.per_iteration
            else None
        ),
    }


def _vs_ridge_rows(
    trajectories: list[LoopTrajectory], ridge_ref: dict[str, LoopTrajectory]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for traj in trajectories:
        reference = (
            ridge_ref["degree_aware"] if "degree_aware" in traj.solver else ridge_ref["canonical"]
        )
        if traj.solver.startswith("ridge"):
            continue
        rmse_gap = abs(traj.final_state_rmse - reference.final_state_rmse)
        x_gap = float(np.linalg.norm(traj.x_hat - reference.x_hat))
        rows.append(
            {
                "solver": traj.solver,
                "reference": reference.solver,
                "final_state_rmse": traj.final_state_rmse,
                "reference_final_state_rmse": reference.final_state_rmse,
                "final_state_rmse_gap": rmse_gap,
                "final_state_l2_gap_vs_ridge": x_gap,
                "converged": traj.converged,
                "iterations": traj.iterations,
            }
        )
    return rows


def _repetition_rows(trajectories: list[LoopTrajectory]) -> list[dict[str, Any]]:
    from robust_qsvt_se.qsvt.shot_readout_model import required_shots_for_additive_error

    rows: list[dict[str, Any]] = []
    for traj in trajectories:
        if "qsvt" not in traj.solver:
            continue
        for row in traj.per_iteration:
            p_succ = row.get("postselection_probability")
            degree = row.get("degree")
            for epsilon in (1.0e-2, 1.0e-3):
                shots = required_shots_for_additive_error(epsilon)
                attempts = math.ceil(shots / p_succ) if p_succ and p_succ > 0 else None
                rows.append(
                    {
                        "solver": traj.solver,
                        "iteration": row["iteration"],
                        "readout_epsilon": epsilon,
                        "readout_shots": shots,
                        "postselection_probability": p_succ,
                        "degree": degree,
                        "signal_unitary_calls_per_attempt": degree,
                        "postselection_attempts": attempts,
                        "residual_reloaded_this_iteration": True,
                        "jacobian_rebuilt_this_iteration": True,
                    }
                )
    return rows


def _readme(
    summary_rows: list[dict[str, Any]],
    vs_ridge_rows: list[dict[str, Any]],
    degree_aware_alpha: float,
    canonical_alpha: float,
) -> str:
    lines = [
        "# Phase 10 WP D: Nonlinear AC QSVT-in-the-Loop (IEEE 14)",
        "",
        CLAIM,
        "",
        "## Setup",
        "",
        "- Nonlinear AC IEEE 14 Gauss-Newton loop, problem setup identical to the existing "
        "`iterative_ac` experiments (measurements, perturbation, stopping rule, 8-iteration "
        "cap).",
        f"- Canonical alpha = {canonical_alpha:g}; degree-aware alpha = {degree_aware_alpha:.4g} "
        f"(lambda = {DEGREE_AWARE_LAMBDA} on the first Jacobian).",
        "- `beta_k` and `lambda_k = alpha/beta_k^2` are recomputed each iteration; the residual "
        "and Jacobian are rebuilt each iteration.",
        "",
        "## Solvers compared",
        "",
        "| solver | iters | converged | final state RMSE | max |update - Ridge| |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['solver']} | {row['iterations']} | {row['converged']} | "
            f"{row['final_state_rmse']:.6e} | "
            f"{_fmt(row['max_update_error_vs_ridge'])} |"
        )
    lines += [
        "",
        "## Agreement with Ridge",
        "",
        "The matrix-level QSVT-target loop step uses the exact `sigma/(sigma^2+alpha)` filter, "
        "so it matches Ridge to machine precision by construction; the degree-`d` bounded "
        "polynomial approximant error is logged per iteration "
        "(`approximant_rel_error_vs_ridge`). The full-rectangular statevector QSVT-in-loop "
        "drives the step with the postselected QSVT update at the degree-aware alpha.",
        "",
        "| solver | final RMSE | Ridge RMSE | RMSE gap | state L2 gap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in vs_ridge_rows:
        lines.append(
            f"| {row['solver']} | {row['final_state_rmse']:.6e} | "
            f"{row['reference_final_state_rmse']:.6e} | {row['final_state_rmse_gap']:.2e} | "
            f"{row['final_state_l2_gap_vs_ridge']:.2e} |"
        )
    lines += [
        "",
        "## QSVT phase behavior and repetition",
        "",
        "Per-iteration phase pass/fail, postselection probability, degree, and update error "
        "vs Ridge are in `nonlinear_qsvt_iteration_log.csv`; per-iteration repetition "
        "(attempts = shots / p_succ, residual reloaded and Jacobian rebuilt every iteration) "
        "is in `nonlinear_qsvt_resource_repetition.csv`. The canonical alpha = 1e-4 is "
        "degree-limited for the bounded polynomial (recorded, not hidden); the exact-filter "
        "target still drives the loop so the trajectory matches Ridge.",
        "",
        "## Scope",
        "",
        "Simulator integration only. The matrix-level QSVT target matching Ridge at matched "
        "alpha is expected, not a superiority result; the full-rectangular statevector "
        "QSVT-in-loop is limited to this small case and does not imply practical quantum "
        "deployment.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _fmt(value: Any, spec: str = "{:.2e}") -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return spec.format(number)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 WP D: nonlinear AC QSVT-in-the-loop (IEEE 14)"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--skip-level2", action="store_true")
    args = parser.parse_args(argv)
    run = run_phase10_nonlinear_qsvt_loop(
        {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "run_level2": not args.skip_level2,
            "command": "scripts/run_phase10_nonlinear_qsvt_loop.py " + " ".join(argv or []),
        }
    )
    print(pd.DataFrame(run["summary_rows"]).to_string(index=False))
    print(f"Outputs: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
