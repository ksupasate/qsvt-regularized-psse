"""QSP phase-synthesis and finite-degree approximation refinement for readout-limited
selected QSVT subproblems.

The readout co-design pass (:mod:`robust_qsvt_se.paper.readout_alpha_degree_codesign`) reduced
the gate-level full-vector readout error of the high-error selected subproblems by 1.9-6.4x but
left three cases above the strict ``0.10`` gate-vs-Ridge threshold, all dominated by the
``exact_target_vs_polynomial`` finite-degree approximation error and, at degree 45, by QSP phase
synthesis degradation. This module refines that bottleneck for the high-error subproblems only:

* Phase 0 freezes the current co-design baseline.
* Phase 1 runs a degree 47/49 *diagnostic-only* sweep to separate finite-degree, boundedness/
  overshoot, and phase-synthesis ceilings.
* Phase 2 compares supported QSP phase-synthesis solver settings (PennyLane ``iterative`` vs
  ``root-finding`` angle solvers, grid size) and honestly records unsupported variants.
* Phase 3 implements weighted singular-support polynomial fitting and compares it against the
  uniform fit with dense-grid boundedness safety.
* Phase 4 reports a three-view summary (matched-alpha common reference, matched-alpha refined
  QSP, relaxed-alpha QSVT-validated) with same-alpha Ridge comparison and alpha-shift metrics.
* Phases 5-9 add a selection-leakage audit, a phase-synthesis failure taxonomy, cost/accuracy
  trade-offs, best-config sampling validation, and the scope note plus figure data.

Improvements are reported where achieved; remaining cases are labelled polynomial-, boundedness-,
or phase-synthesis-limited. This does **not** demonstrate efficient full IEEE-scale full-vector
readout, quantum speedup, or QSVT numerical superiority over Ridge/Tikhonov, and selection never
uses true-state RMSE or any oracle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.paper.full_vector_readout import (
    _bounded_scale_and_polynomial,
    _global_sign_aligned,
    _protocol_global_sign,
    _rel_l2,
    _residual_ratio,
    _stable_base_seed,
    _top_k,
    _trial_seed,
    discover_validated_subproblems,
    gate_level_subproblem_readout,
    qsvt_target_readout,
    reconstruct_subproblem,
    recover_relative_signs,
    sample_basis_magnitudes,
)
from robust_qsvt_se.paper.full_vector_readout import (
    _SystemCache as SystemCache,
)
from robust_qsvt_se.paper.readout_alpha_degree_codesign import (
    GOOD_GATE_ERROR,
    MATCHED_ALPHA,
    PASS_THRESHOLD,
)
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_qsvt import (
    _simulate_statevector,
    build_gate_level_qsvt_state_circuit,
)
from robust_qsvt_se.qsvt.gate_state_preparation import (
    normalize_and_pad_for_gate_preparation,
)
from robust_qsvt_se.qsvt.weighted_singular_support import compute_singular_support
from robust_qsvt_se.utils.io import ensure_directory

REFINEMENT_CLAIM = (
    "The remaining gate-level readout error of the high-error selected QSVT subproblems is "
    "analyzed through QSP phase synthesis, finite-degree polynomial approximation, weighted "
    "singular-support fitting, and alpha/degree refinement. Improvements are reported where "
    "achieved; remaining cases are labelled polynomial-, boundedness-, or phase-synthesis-"
    "limited. This does not demonstrate efficient full IEEE-scale full-vector readout, quantum "
    "speedup, or QSVT numerical superiority over Ridge/Tikhonov."
)

DEGREE_DIAGNOSTIC_GRID: tuple[int, ...] = (47, 49)
DEFAULT_REFINEMENT_DEGREES: tuple[int, ...] = (41, 45)
DEFAULT_ALPHA_GRID: tuple[float, ...] = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
DEFAULT_SHOTS = 100_000
DEFAULT_TRIALS = 10
DEFAULT_SEED = 123
DEFAULT_TARGET_ERROR = 1.0e-2

# Realized-response boundedness tolerance on the QSVT signal interval.
_BOUND_TOL = 1.0e-3
# Phase-synthesis fidelity above this (max realized-vs-polynomial response error) is "degraded".
_PHASE_DEGRADED_TOL = 0.05
# gate-vs-polynomial above this flags realized phase-synthesis degradation.
_GATE_SYNTHESIS_TOL = 0.05
# Short phase-synthesis cap for the diagnostic degree-47/49 sweep so the fallback chain on
# (expected) high-degree synthesis failure does not grind through 25-second attempts.
_DIAGNOSTIC_PHASE_TIMEOUT = 6
# Highest degree at which phase synthesis is reliable for the solver-variant comparison; the
# root-finding angle solver does not scale past this, and degree >= 47 degrades regardless.
_MAX_VARIANT_DEGREE = 41

SUPPORTED_SOLVER_VARIANTS: tuple[tuple[str, str], ...] = (
    ("default", "angle_solver=iterative; grid=max(2048,deg+2)"),
    ("root_finding_angle_solver", "angle_solver=root-finding; grid=max(2048,deg+2)"),
    ("high_precision_grid", "angle_solver=iterative; grid=16385"),
)
UNSUPPORTED_SOLVER_VARIANTS: tuple[tuple[str, str], ...] = (
    ("multi_start", "no multi-start initial-phase API in pennylane.poly_to_angles"),
    ("warm_start_from_lower_degree", "no warm-start phase API exposed"),
    ("alternative_initial_phase", "initial phase guess is not configurable"),
    ("weighted_support_phase_refinement", "phase solver does not accept singular-support weights"),
)

FIT_MODES: tuple[str, ...] = (
    "global_uniform_fit",
    "weighted_singular_support_fit",
    "residual_aware_fit",
    "hybrid_global_plus_support_fit",
)
WEIGHT_MODES: tuple[str, ...] = (
    "uniform_support_weight",
    "residual_energy_weight",
    "update_contribution_weight",
)

# --------------------------------------------------------------------------- column schemas
BASELINE_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "matched_alpha",
    "matched_degree",
    "matched_alpha_error",
    "current_best_alpha",
    "current_best_degree",
    "current_best_error",
    "improvement_factor",
    "dominant_error_source",
    "exact_target_vs_polynomial_error",
    "polynomial_vs_gate_error",
    "gate_vs_ridge_error",
    "phase_synthesis_status",
    "status",
    "source_artifact",
    "notes",
]

DEGREE_DIAGNOSTIC_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "overshoot_diagnostic_only",
    "eligible_for_best_config",
    "target_scale_C",
    "boundedness_ok",
    "dense_grid_max_abs_value",
    "dense_grid_max_error",
    "singular_support_weighted_error",
    "overshoot_flag",
    "phase_synthesis_status",
    "phase_objective",
    "phase_residual",
    "synthesized_degree",
    "gate_run_status",
    "exact_target_error_vs_ridge",
    "polynomial_error_vs_exact_target",
    "gate_error_vs_polynomial",
    "gate_error_vs_ridge",
    "success_probability",
    "postselection_overhead",
    "status",
    "notes",
]

DEGREE_CEILING_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "baseline_best_degree",
    "baseline_best_error",
    "best_diagnostic_degree",
    "best_diagnostic_alpha",
    "best_diagnostic_gate_error",
    "best_diagnostic_gate_vs_polynomial",
    "ceiling_label",
    "improves_below_baseline",
    "status",
    "notes",
]

SOLVER_VARIANT_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "phase_solver_variant",
    "solver_supported",
    "solver_parameters",
    "phase_synthesis_status",
    "phase_objective",
    "phase_residual",
    "max_response_error_dense_grid",
    "max_response_error_singular_support",
    "weighted_response_error_singular_support",
    "boundedness_ok",
    "overshoot_flag",
    "gate_run_status",
    "gate_error_vs_polynomial",
    "gate_error_vs_ridge",
    "runtime_seconds",
    "selected_variant_candidate",
    "status",
    "notes",
]

WEIGHTED_FIT_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "fit_mode",
    "weight_mode",
    "singular_value_index",
    "singular_value",
    "residual_component",
    "ridge_gain",
    "target_value",
    "polynomial_value",
    "pointwise_error",
    "weight",
    "weighted_squared_error",
    "status",
    "notes",
]

FIT_MODE_COMPARISON_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "fit_mode",
    "weight_mode",
    "support_weighted_error",
    "support_max_error",
    "dense_grid_max_error",
    "boundedness_ok",
    "overshoot_flag",
    "phase_synthesis_status",
    "gate_run_status",
    "gate_error_vs_polynomial",
    "gate_error_vs_ridge",
    "residual_ratio_after_gate",
    "success_probability",
    "postselection_overhead",
    "eligible_for_best_config",
    "status",
    "notes",
]

THREE_VIEW_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "phase_solver_variant",
    "fit_mode",
    "weight_mode",
    "gate_error_vs_matching_ridge_alpha",
    "ridge_alpha_shift_vs_common_alpha",
    "exact_target_vs_polynomial_error",
    "polynomial_vs_gate_error",
    "shot_readout_error",
    "success_probability",
    "postselection_overhead",
    "estimated_total_readout_shots",
    "status",
    "interpretation",
    "source_artifact",
    "notes",
]

SELECTION_LEAKAGE_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "selection_stage",
    "selection_metric",
    "uses_true_state",
    "uses_state_rmse",
    "uses_ridge_reference",
    "uses_singular_values",
    "uses_residual",
    "uses_polynomial_error",
    "uses_gate_error",
    "allowed_for_selection",
    "diagnostic_only",
    "notes",
]

FAILURE_TAXONOMY_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "phase_solver_variant",
    "fit_mode",
    "failure_label",
    "failure_layer",
    "evidence_metric",
    "evidence_value",
    "recommended_action",
    "status",
    "notes",
]

COST_ACCURACY_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "phase_solver_variant",
    "fit_mode",
    "gate_error_vs_matching_ridge_alpha",
    "success_probability",
    "postselection_overhead",
    "estimated_total_readout_shots",
    "query_count",
    "estimated_gate_depth",
    "accuracy_cost_score",
    "status",
    "notes",
]

DEGREE_COST_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "query_count",
    "estimated_gate_depth",
    "phase_synthesis_runtime",
    "phase_synthesis_status",
    "gate_error_vs_ridge",
    "gate_error_vs_polynomial",
    "success_probability",
    "postselection_overhead",
    "total_readout_shots",
    "status",
    "notes",
]

SAMPLING_VALIDATION_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "phase_solver_variant",
    "fit_mode",
    "shots",
    "trial_id",
    "rng_seed",
    "vector_relative_l2_error",
    "norm_relative_error",
    "sign_accuracy_reliable",
    "fraction_reliable_signs",
    "top_k_match",
    "status",
    "notes",
]

SAMPLING_VALIDATION_SUMMARY_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "shots",
    "trials",
    "mean_vector_relative_l2_error",
    "p50_vector_relative_l2_error",
    "p95_vector_relative_l2_error",
    "mean_norm_relative_error",
    "mean_sign_accuracy_reliable",
    "mean_fraction_reliable_signs",
    "top_k_match_rate",
    "status",
    "notes",
]

FIGURE_THREE_VIEW_COLUMNS = [
    "case",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "gate_error_vs_matching_ridge_alpha",
    "status",
    "plot_group",
    "notes",
]

FIGURE_FAILURE_COLUMNS = [
    "case",
    "subproblem_type",
    "failure_layer",
    "failure_label",
    "evidence_metric",
    "evidence_value",
    "notes",
]

FIGURE_COST_COLUMNS = [
    "case",
    "subproblem_type",
    "view",
    "degree",
    "gate_error_vs_matching_ridge_alpha",
    "estimated_total_readout_shots",
    "estimated_gate_depth",
    "notes",
]


# --------------------------------------------------------------------------- io helpers
def _read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_table(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> Path:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- phase response
def qsvt_phase_response(x: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Real top-left QSVT response ``P(x)`` of a phase sequence on a scalar signal.

    Implements the PennyLane ``QSVT(BlockEncode(x), [PCPhase(phi)])`` convention as a vectorized
    product of ``2x2`` matrices, validated to reproduce ``poly_to_angles`` polynomials exactly.
    This measures phase-synthesis fidelity (how well synthesized phases realize their intended
    polynomial) without constructing the multi-qubit circuit.
    """

    values = np.asarray(x, dtype=np.float64)
    phase_values = np.asarray(phases, dtype=np.float64)
    if phase_values.ndim != 1 or phase_values.size == 0:
        raise ValueError("phases must be a non-empty 1D array")
    off = np.sqrt(np.clip(1.0 - values * values, 0.0, None))
    n = values.size
    signal = np.zeros((n, 2, 2), dtype=np.complex128)
    signal[:, 0, 0] = values
    signal[:, 0, 1] = off
    signal[:, 1, 0] = off
    signal[:, 1, 1] = -values

    def projector(phi: float) -> np.ndarray:
        mat = np.zeros((n, 2, 2), dtype=np.complex128)
        mat[:, 0, 0] = np.exp(1j * phi)
        mat[:, 1, 1] = np.exp(-1j * phi)
        return mat

    unitary = projector(float(phase_values[0]))
    for phase in phase_values[1:]:
        unitary = unitary @ signal @ projector(float(phase))
    return np.real(unitary[:, 0, 0])


def _synthesize_phases(
    coefficients: np.ndarray, *, angle_solver: str
) -> tuple[np.ndarray | None, str]:
    """Synthesize QSVT phases from bounded coefficients; return ``(phases, status)``."""

    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        return None, f"pennylane_unavailable: {exc}"
    try:
        phases = np.asarray(
            qml.poly_to_angles(
                np.asarray(coefficients, dtype=np.float64), "QSVT", angle_solver=str(angle_solver)
            ),
            dtype=np.float64,
        )
    except Exception as exc:
        return None, f"synthesis_failed: {type(exc).__name__}: {exc}"
    if phases.ndim != 1 or phases.size == 0 or not np.all(np.isfinite(phases)):
        return None, "synthesis_failed: invalid phases"
    return phases, "synthesized"


def _domain_min(singular_values_A: np.ndarray) -> float:
    positive = singular_values_A[singular_values_A > 1.0e-14]
    if positive.size == 0:
        raise ValueError("subproblem has no positive singular value")
    return min(max(1.0e-6, 0.9 * float(np.min(positive))), 0.95)


def _bounded_target_and_scale(
    singular_values_A: np.ndarray, *, alpha_norm: float, domain_min: float, grid_size: int = 4097
) -> tuple[np.ndarray, float]:
    """Return ``(f_bounded on grid, C)`` for the Ridge filter on the normalized A-domain."""

    grid = np.linspace(domain_min, 1.0, grid_size, dtype=np.float64)
    unbounded = grid / (grid * grid + float(alpha_norm))
    scale_C = max(float(np.max(np.abs(unbounded))), np.finfo(np.float64).eps)
    return unbounded / scale_C, scale_C


def _phase_fidelity(
    phases: np.ndarray,
    coefficients: np.ndarray,
    *,
    singular_values_A: np.ndarray,
    domain_min: float,
    alpha_norm: float,
    support_weight: np.ndarray,
) -> dict[str, float]:
    """Phase-synthesis fidelity diagnostics from the realized scalar QSVT response."""

    dense = np.linspace(-1.0, 1.0, 4097, dtype=np.float64)
    response_dense = qsvt_phase_response(dense, phases)
    dense_max_abs = float(np.max(np.abs(response_dense)))

    domain_grid = np.linspace(domain_min, 1.0, 2049, dtype=np.float64)
    response_domain = qsvt_phase_response(domain_grid, phases)
    polynomial_domain = Polynomial(coefficients)(domain_grid)
    # synthesis error: realized response vs the intended bounded polynomial
    synthesis_error = np.abs(response_domain - polynomial_domain)
    phase_objective = float(np.max(synthesis_error))
    phase_residual = float(np.sqrt(np.mean(synthesis_error**2)))

    # end-to-end realized filter error vs the ideal bounded target on the singular support
    unbounded_support = singular_values_A / (singular_values_A**2 + float(alpha_norm))
    scale_C = max(float(np.max(np.abs(unbounded_support))), np.finfo(np.float64).eps)
    target_support = unbounded_support / scale_C
    response_support = qsvt_phase_response(singular_values_A, phases)
    support_error = np.abs(response_support - target_support)
    weights = np.asarray(support_weight, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    weighted_support_error = (
        float(np.sum(weights * support_error) / weight_sum) if weight_sum > 0 else float("nan")
    )
    domain_target = (domain_grid / (domain_grid**2 + float(alpha_norm))) / scale_C
    dense_grid_max_error = float(np.max(np.abs(response_domain - domain_target)))
    return {
        "dense_grid_max_abs_value": dense_max_abs,
        "phase_objective": phase_objective,
        "phase_residual": phase_residual,
        "max_response_error_singular_support": float(np.max(support_error)),
        "weighted_response_error_singular_support": weighted_support_error,
        "dense_grid_max_error": dense_grid_max_error,
    }


# --------------------------------------------------------------------------- custom-coeff gate
def gate_update_from_coefficients(
    H: np.ndarray,
    r: np.ndarray,
    coefficients: np.ndarray,
    scale_C: float,
    *,
    seed: int,
    angle_solver: str = "iterative",
) -> dict[str, Any]:
    """Run an actual gate-level QSVT circuit for an arbitrary bounded polynomial.

    Mirrors the validated solver's post-polynomial path (block encoding -> phase synthesis ->
    circuit -> statevector) but accepts externally-fit coefficients, so weighted singular-support
    fits are confirmed at the real circuit level rather than only on paper. Returns ``available``
    False with a reason on any synthesis/simulation failure, never substituting the exact target.
    """

    H = np.asarray(H, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    phases, status = _synthesize_phases(coefficients, angle_solver=angle_solver)
    if phases is None:
        return {"available": False, "reason": status, "phases": None}
    try:
        B = H.T
        singular_values_B = np.linalg.svd(B, compute_uv=False)
        beta = max(float(singular_values_B[0]), np.finfo(np.float64).eps)
        A = B / beta
        block_encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
        state_preparation = normalize_and_pad_for_gate_preparation(
            r, target_dimension=block_encoding.unitary.shape[0]
        )
        circuit_bundle = build_gate_level_qsvt_state_circuit(
            block_unitary=block_encoding.unitary,
            phases=phases,
            residual_state=state_preparation.padded_state,
            encoded_dimension=int(H.shape[1]),
        )
        statevector, _seconds = _simulate_statevector(circuit_bundle.full_state_circuit)
    except Exception as exc:  # pragma: no cover - guarded simulation failure
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "phases": phases}
    prefix = np.real(statevector[: int(H.shape[1])])
    gate_update = state_preparation.original_norm * rescale_bounded_target_to_original(
        prefix, beta=beta, C=float(scale_C)
    )
    success_probability = float(np.sum(np.abs(statevector[: int(H.shape[1])]) ** 2))
    gate_norm = float(np.linalg.norm(gate_update))
    gate_state = gate_update / gate_norm if gate_norm > 0 else gate_update
    return {
        "available": True,
        "gate_update": gate_update,
        "gate_state": gate_state,
        "success_probability": success_probability,
        "phases": phases,
        "query_count": int(circuit_bundle.block_encoding_gate_count),
    }


def _is_uniform_recipe(fit_mode: str, weight_mode: str) -> bool:
    return fit_mode in ("", "global_uniform_fit") and weight_mode in (
        "",
        "uniform_support_weight",
    )


def realize_config_gate(
    context: _RefinementContext,
    *,
    alpha: float,
    degree: int,
    fit_mode: str = "global_uniform_fit",
    weight_mode: str = "uniform_support_weight",
    angle_solver: str = "iterative",
    seed: int,
    cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Realize one config recipe as a real gate run, with memoization.

    Uniform recipes use the validated standard solver; weighted recipes synthesize the weighted
    singular-support polynomial and run the same circuit path via :func:`gate_update_from_
    coefficients`. Returns unified gate metrics so the three-view, cost, and sampling phases all
    consume real circuit results for the same config exactly once.
    """

    key = (
        context.entry["case"],
        context.entry["subproblem_type"],
        round(float(alpha), 12),
        int(degree),
        fit_mode,
        weight_mode,
        angle_solver,
    )
    if cache is not None and key in cache:
        return cache[key]
    result = _realize_config_gate_uncached(
        context,
        alpha=float(alpha),
        degree=int(degree),
        fit_mode=fit_mode,
        weight_mode=weight_mode,
        angle_solver=angle_solver,
        seed=int(seed),
    )
    if cache is not None:
        cache[key] = result
    return result


def _realize_config_gate_uncached(
    context: _RefinementContext,
    *,
    alpha: float,
    degree: int,
    fit_mode: str,
    weight_mode: str,
    angle_solver: str,
    seed: int,
) -> dict[str, Any]:
    H, r = context.H, context.r
    try:
        state = qsvt_target_readout(H, r, alpha=float(alpha), degree=int(degree))
    except Exception as exc:  # pragma: no cover - guarded degenerate fit
        return {"available": False, "reason": f"exact_target_failed: {exc}"}
    ridge = state.ridge_update
    poly_available = bool(state.polynomial_available)
    exact_target = state.readout_state * state.recovered_norm
    exact_poly = _rel_l2(state.polynomial_update, exact_target) if poly_available else float("nan")

    if _is_uniform_recipe(fit_mode, weight_mode):
        gate = gate_level_subproblem_readout(
            H,
            r,
            alpha=float(alpha),
            degree=int(degree),
            seed=int(seed),
            case=context.entry["case"],
            angle_solver=angle_solver,
        )
        if not gate.get("available"):
            return {
                "available": False,
                "reason": gate.get("reason", "gate_failed"),
                "ridge_update": ridge,
            }
        gate_update = np.asarray(gate["gate_update"], dtype=np.float64)
        gate_state = np.asarray(gate["gate_state"], dtype=np.float64)
        success = float(gate["success_probability"])
        depth = float(gate["circuit_depth"])
        query = int(gate["qsvt_query_count"])
        boundedness_ok = bool(state.bounded_target_ok)
    else:
        singular_values_A = np.asarray(state.singular_values, dtype=np.float64) / max(
            float(state.singular_values[0]), np.finfo(np.float64).eps
        )
        domain_min = _domain_min(singular_values_A)
        support = compute_singular_support(H, r, alpha=float(alpha))
        support_weight = _support_weights(support, weight_mode)
        coefficients, scale_C = _fit_weighted_odd_polynomial(
            singular_values_A=singular_values_A,
            support_weight=support_weight,
            alpha_norm=float(state.alpha_norm),
            domain_min=domain_min,
            degree=int(degree),
            fit_mode=fit_mode,
        )
        dense = np.linspace(-1.0, 1.0, 4097, dtype=np.float64)
        boundedness_ok = bool(
            float(np.max(np.abs(Polynomial(coefficients)(dense)))) <= 1.0 + _BOUND_TOL
        )
        gate = gate_update_from_coefficients(
            H, r, coefficients, scale_C, seed=int(seed), angle_solver=angle_solver
        )
        if not gate.get("available"):
            return {
                "available": False,
                "reason": gate.get("reason", "gate_failed"),
                "ridge_update": ridge,
            }
        gate_update = np.asarray(gate["gate_update"], dtype=np.float64)
        gate_state = np.asarray(gate["gate_state"], dtype=np.float64)
        success = float(gate["success_probability"])
        depth = float("nan")
        query = int(gate["query_count"])

    return {
        "available": True,
        "gate_update": gate_update,
        "gate_state": gate_state,
        "ridge_update": ridge,
        "success_probability": success,
        "circuit_depth": depth,
        "query_count": query,
        "gate_error_vs_ridge": _rel_l2(gate_update, ridge),
        "gate_error_vs_polynomial": (
            _rel_l2(gate_update, state.polynomial_update) if poly_available else float("nan")
        ),
        "exact_target_vs_polynomial": exact_poly,
        "boundedness_ok": boundedness_ok,
        "residual_ratio_after_gate": _residual_ratio(H, gate_update, r),
    }


# --------------------------------------------------------------------------- subproblem cache
class _RefinementContext:
    """Per-subproblem reusable quantities (H, r, SVD, singular support)."""

    __slots__ = ("H", "entry", "r")

    def __init__(self, entry: dict[str, Any], H: np.ndarray, r: np.ndarray) -> None:
        self.entry = entry
        self.H = H
        self.r = r


def _build_contexts(
    targeted: list[tuple[str, str]], input_root: Path, *, seed: int
) -> dict[tuple[str, str], _RefinementContext]:
    discovered, _missing = discover_validated_subproblems(input_root)
    entry_by_pair = {(e["case"], e["subproblem_type"]): e for e in discovered}
    system_cache = SystemCache()
    contexts: dict[tuple[str, str], _RefinementContext] = {}
    for pair in targeted:
        entry = entry_by_pair.get(pair)
        if entry is None:
            continue
        sub = reconstruct_subproblem(
            case=entry["case"],
            row_indices=entry["row_indices"],
            col_indices=entry["col_indices"],
            system_cache=system_cache,
            seed=int(seed),
        )
        contexts[pair] = _RefinementContext(entry, sub["H"], sub["r"])
    return contexts


# --------------------------------------------------------------------------- Phase 0: baseline
def freeze_baseline(readout_dir: Path) -> list[dict[str, Any]]:
    """Derive the frozen co-design baseline from existing readout artifacts (never overwritten)."""

    readout_dir = Path(readout_dir)
    summary = _read_csv(readout_dir / "readout_two_view_summary.csv")
    best = _read_csv(readout_dir / "readout_selected_best_configs.csv")
    classification = _read_csv(readout_dir / "readout_config_diagnostic_classification.csv")
    if summary.empty and best.empty:
        return []

    best_by_pair = (
        {
            (str(row["case"]), str(row["subproblem_type"])): row.to_dict()
            for _, row in best.iterrows()
        }
        if not best.empty
        else {}
    )
    class_by_pair = (
        {
            (str(row["case"]), str(row["subproblem_type"])): row.to_dict()
            for _, row in classification.iterrows()
        }
        if not classification.empty
        else {}
    )

    rows: list[dict[str, Any]] = []
    iterator = summary.iterrows() if not summary.empty else best.iterrows()
    for _, summary_row in iterator:
        case = str(summary_row["case"])
        stype = str(summary_row["subproblem_type"])
        pair = (case, stype)
        best_row = best_by_pair.get(pair, {})
        class_row = class_by_pair.get(pair, {})
        best_gate_vs_poly = _safe_float(best_row.get("best_gate_vs_polynomial_error"))
        phase_status = (
            "phase_synthesis_degraded_at_best_degree"
            if np.isfinite(best_gate_vs_poly) and best_gate_vs_poly > _GATE_SYNTHESIS_TOL
            else "phase_synthesis_clean_at_best_degree"
        )
        rows.append(
            {
                "case": case,
                "subproblem_id": str(
                    summary_row.get("subproblem_id", best_row.get("subproblem_id", ""))
                ),
                "subproblem_type": stype,
                "matched_alpha": _safe_float(summary_row.get("matched_alpha", MATCHED_ALPHA)),
                "matched_degree": int(_safe_float(summary_row.get("matched_degree", -1))),
                "matched_alpha_error": _safe_float(
                    summary_row.get("matched_alpha_gate_error", best_row.get("matched_alpha_error"))
                ),
                "current_best_alpha": _safe_float(
                    summary_row.get("validated_alpha", best_row.get("best_alpha"))
                ),
                "current_best_degree": int(
                    _safe_float(
                        summary_row.get("validated_degree", best_row.get("best_degree", -1))
                    )
                ),
                "current_best_error": _safe_float(
                    summary_row.get(
                        "validated_gate_error", best_row.get("best_gate_error_vs_ridge")
                    )
                ),
                "improvement_factor": _safe_float(
                    summary_row.get("improvement_factor", best_row.get("improvement_factor"))
                ),
                "dominant_error_source": str(
                    summary_row.get(
                        "validated_dominant_error_source",
                        class_row.get("dominant_error_source", ""),
                    )
                ),
                "exact_target_vs_polynomial_error": _safe_float(
                    best_row.get(
                        "best_polynomial_error_vs_exact_target",
                        class_row.get("exact_target_vs_polynomial_error"),
                    )
                ),
                "polynomial_vs_gate_error": best_gate_vs_poly,
                "gate_vs_ridge_error": _safe_float(
                    summary_row.get(
                        "validated_gate_error", best_row.get("best_gate_error_vs_ridge")
                    )
                ),
                "phase_synthesis_status": phase_status,
                "status": str(summary_row.get("final_status", best_row.get("status", ""))),
                "source_artifact": "full_vector_readout/readout_two_view_summary.csv",
                "notes": (
                    "frozen co-design baseline; existing readout results preserved, not overwritten"
                ),
            }
        )
    return rows


def baseline_summary_markdown(rows: list[dict[str, Any]]) -> str:
    high_error = [r for r in rows if _safe_float(r["current_best_error"]) > GOOD_GATE_ERROR]
    lines = [
        "# Phase-Synthesis Refinement Baseline (Frozen)",
        "",
        REFINEMENT_CLAIM,
        "",
        "## Frozen co-design baseline",
        "The current readout co-design results are frozen here before refinement. They are not "
        "overwritten; this table is derived from the existing two-view artifacts.",
        "",
        "| case | subproblem | matched err | current best (alpha,deg)->err | improve | status |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        best_degree = int(_safe_float(row["current_best_degree"]))
        lines.append(
            f"| {row['case']} | {row['subproblem_type']} | "
            f"{_safe_float(row['matched_alpha_error']):.3f} | "
            f"({_safe_float(row['current_best_alpha']):.0e}, {best_degree})"
            f" -> {_safe_float(row['current_best_error']):.3f} | "
            f"{_safe_float(row['improvement_factor']):.2f}x | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## High-error subproblems targeted for refinement",
            *(
                [
                    f"- {r['case']}/{r['subproblem_type']}: "
                    f"best {_safe_float(r['current_best_error']):.3f}"
                    for r in high_error
                ]
                or ["- none above the 0.10 threshold"]
            ),
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- Phase 1: degree 47/49
def evaluate_degree_diagnostic_cell(
    context: _RefinementContext,
    *,
    alpha: float,
    degree: int,
    seed: int,
    target_error: float,
) -> dict[str, Any]:
    """Evaluate one diagnostic-only (alpha, degree) cell with phase + real-gate diagnostics."""

    entry = context.entry
    base = {
        "case": entry["case"],
        "subproblem_id": entry["subproblem_id"],
        "subproblem_type": entry["subproblem_type"],
        "alpha": float(alpha),
        "degree": int(degree),
        "overshoot_diagnostic_only": True,
    }
    try:
        state = qsvt_target_readout(context.H, context.r, alpha=float(alpha), degree=int(degree))
    except Exception as exc:  # pragma: no cover - guarded degenerate fit
        return {
            **base,
            **_empty_diagnostic_metrics(),
            "status": "exact_target_failed",
            "notes": f"{type(exc).__name__}: {exc}",
        }

    boundedness_ok = bool(state.bounded_target_ok)
    scale_C = float(state.scale_factor_C)
    ridge = state.ridge_update
    exact_target = state.readout_state * state.recovered_norm
    exact_error_vs_ridge = _rel_l2(exact_target, ridge)
    poly_available = bool(state.polynomial_available)
    poly_error_vs_exact = (
        _rel_l2(state.polynomial_update, exact_target) if poly_available else float("nan")
    )

    singular_values_A = np.asarray(state.singular_values, dtype=np.float64) / max(
        float(state.singular_values[0]), np.finfo(np.float64).eps
    )
    domain_min = _domain_min(singular_values_A)
    alpha_norm = float(state.alpha_norm)
    support = compute_singular_support(context.H, context.r, alpha=float(alpha))
    support_weight = support.combined_weight

    _scale_C2, coefficients, fit_ok = _bounded_scale_and_polynomial(
        singular_values_A, alpha_norm=alpha_norm, degree=int(degree)
    )
    phase_metrics = _empty_phase_metrics()
    phase_status = "synthesis_failed"
    if coefficients is not None and fit_ok:
        phases, status = _synthesize_phases(coefficients, angle_solver="iterative")
        if phases is not None:
            phase_metrics = _phase_fidelity(
                phases,
                coefficients,
                singular_values_A=singular_values_A,
                domain_min=domain_min,
                alpha_norm=alpha_norm,
                support_weight=support_weight,
            )
            degraded = (
                phase_metrics["dense_grid_max_abs_value"] > 1.0 + _BOUND_TOL
                or phase_metrics["phase_objective"] > _PHASE_DEGRADED_TOL
            )
            phase_status = "phase_synthesis_degraded" if degraded else "synthesized"
        else:
            phase_status = status

    gate = gate_level_subproblem_readout(
        context.H,
        context.r,
        alpha=float(alpha),
        degree=int(degree),
        seed=int(seed),
        case=entry["case"],
        phase_timeout_seconds=_DIAGNOSTIC_PHASE_TIMEOUT,
    )
    if not gate.get("available"):
        gate_run_status = "failed"
        gate_error_vs_poly = float("nan")
        gate_error_vs_ridge = float("nan")
        success_probability = float(state.success_probability)
        synthesized_degree = -1
    else:
        gate_run_status = "computed"
        gate_update = gate["gate_update"]
        gate_error_vs_poly = (
            _rel_l2(gate_update, state.polynomial_update) if poly_available else float("nan")
        )
        gate_error_vs_ridge = _rel_l2(gate_update, ridge)
        success_probability = float(gate["success_probability"])
        synthesized_degree = int(gate["synthesized_degree"])

    overshoot_flag = bool(
        not boundedness_ok or phase_metrics["dense_grid_max_abs_value"] > 1.0 + _BOUND_TOL
    )
    postselection_overhead = (
        1.0 / success_probability
        if np.isfinite(success_probability) and success_probability > 0.0
        else float("nan")
    )
    return {
        **base,
        "eligible_for_best_config": False,
        "target_scale_C": scale_C,
        "boundedness_ok": boundedness_ok,
        "dense_grid_max_abs_value": phase_metrics["dense_grid_max_abs_value"],
        "dense_grid_max_error": phase_metrics["dense_grid_max_error"],
        "singular_support_weighted_error": phase_metrics[
            "weighted_response_error_singular_support"
        ],
        "overshoot_flag": overshoot_flag,
        "phase_synthesis_status": phase_status,
        "phase_objective": phase_metrics["phase_objective"],
        "phase_residual": phase_metrics["phase_residual"],
        "synthesized_degree": synthesized_degree,
        "gate_run_status": gate_run_status,
        "exact_target_error_vs_ridge": exact_error_vs_ridge,
        "polynomial_error_vs_exact_target": poly_error_vs_exact,
        "gate_error_vs_polynomial": gate_error_vs_poly,
        "gate_error_vs_ridge": gate_error_vs_ridge,
        "success_probability": success_probability,
        "postselection_overhead": postselection_overhead,
        "status": "computed" if gate_run_status == "computed" else "gate_unavailable",
        "notes": (
            f"diagnostic-only degree {degree}; synthesized_degree={synthesized_degree}; "
            "not eligible for best-config unless explicitly promoted"
        ),
    }


def _empty_phase_metrics() -> dict[str, float]:
    return {
        "dense_grid_max_abs_value": float("nan"),
        "phase_objective": float("nan"),
        "phase_residual": float("nan"),
        "max_response_error_singular_support": float("nan"),
        "weighted_response_error_singular_support": float("nan"),
        "dense_grid_max_error": float("nan"),
    }


def _empty_diagnostic_metrics() -> dict[str, Any]:
    return {
        "eligible_for_best_config": False,
        "target_scale_C": float("nan"),
        "boundedness_ok": False,
        "dense_grid_max_abs_value": float("nan"),
        "dense_grid_max_error": float("nan"),
        "singular_support_weighted_error": float("nan"),
        "overshoot_flag": True,
        "phase_synthesis_status": "exact_target_failed",
        "phase_objective": float("nan"),
        "phase_residual": float("nan"),
        "synthesized_degree": -1,
        "gate_run_status": "not_run",
        "exact_target_error_vs_ridge": float("nan"),
        "polynomial_error_vs_exact_target": float("nan"),
        "gate_error_vs_polynomial": float("nan"),
        "gate_error_vs_ridge": float("nan"),
        "success_probability": float("nan"),
        "postselection_overhead": float("nan"),
    }


def _degree_ceiling_label(
    cell: dict[str, Any], *, baseline_best_error: float, baseline_poly_error: float
) -> str:
    if not cell.get("boundedness_ok") or cell.get("overshoot_flag"):
        return "boundedness_or_overshoot_ceiling"
    poly_error = _safe_float(cell.get("polynomial_error_vs_exact_target"))
    gate_error = _safe_float(cell.get("gate_error_vs_ridge"))
    gate_vs_poly = _safe_float(cell.get("gate_error_vs_polynomial"))
    poly_improves = (
        np.isfinite(poly_error)
        and np.isfinite(baseline_poly_error)
        and (poly_error < baseline_poly_error * 0.98)
    )
    gate_improves = (
        np.isfinite(gate_error)
        and np.isfinite(baseline_best_error)
        and (gate_error < baseline_best_error * 0.98)
    )
    synthesis_degraded = np.isfinite(gate_vs_poly) and gate_vs_poly > _GATE_SYNTHESIS_TOL
    if gate_improves and not synthesis_degraded:
        return "higher_degree_potentially_useful"
    if poly_improves and (not gate_improves or synthesis_degraded):
        return "phase_synthesis_ceiling"
    if synthesis_degraded:
        return "phase_synthesis_ceiling"
    return "higher_degree_not_helpful"


def build_degree_ceiling_summary(
    diagnostic_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_by_pair = {(r["case"], r["subproblem_type"]): r for r in baseline_rows}
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in diagnostic_rows:
        by_pair.setdefault((row["case"], row["subproblem_type"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for pair, cells in by_pair.items():
        baseline = baseline_by_pair.get(pair, {})
        baseline_best_error = _safe_float(baseline.get("current_best_error"))
        baseline_poly_error = _safe_float(baseline.get("exact_target_vs_polynomial_error"))
        computed = [c for c in cells if c.get("gate_run_status") == "computed"]
        best_cell = (
            min(computed, key=lambda c: _safe_float(c["gate_error_vs_ridge"])) if computed else None
        )
        if best_cell is None:
            rows.append(
                {
                    "case": pair[0],
                    "subproblem_id": cells[0]["subproblem_id"],
                    "subproblem_type": pair[1],
                    "baseline_best_degree": int(
                        _safe_float(baseline.get("current_best_degree", -1))
                    ),
                    "baseline_best_error": baseline_best_error,
                    "best_diagnostic_degree": -1,
                    "best_diagnostic_alpha": float("nan"),
                    "best_diagnostic_gate_error": float("nan"),
                    "best_diagnostic_gate_vs_polynomial": float("nan"),
                    "ceiling_label": "no_gate_config",
                    "improves_below_baseline": False,
                    "status": "no_computed_diagnostic_cell",
                    "notes": "degree 47/49 produced no computed gate cell; recorded honestly",
                }
            )
            continue
        label = _degree_ceiling_label(
            best_cell,
            baseline_best_error=baseline_best_error,
            baseline_poly_error=baseline_poly_error,
        )
        best_error = _safe_float(best_cell["gate_error_vs_ridge"])
        improves = bool(
            np.isfinite(best_error)
            and np.isfinite(baseline_best_error)
            and best_error < baseline_best_error * 0.98
        )
        rows.append(
            {
                "case": pair[0],
                "subproblem_id": best_cell["subproblem_id"],
                "subproblem_type": pair[1],
                "baseline_best_degree": int(_safe_float(baseline.get("current_best_degree", -1))),
                "baseline_best_error": baseline_best_error,
                "best_diagnostic_degree": int(best_cell["degree"]),
                "best_diagnostic_alpha": _safe_float(best_cell["alpha"]),
                "best_diagnostic_gate_error": best_error,
                "best_diagnostic_gate_vs_polynomial": _safe_float(
                    best_cell["gate_error_vs_polynomial"]
                ),
                "ceiling_label": label,
                "improves_below_baseline": improves,
                "status": "computed",
                "notes": (
                    "degree 47/49 diagnostic; "
                    + (
                        "higher degree helps, candidate for promotion"
                        if improves
                        else "higher degree does not safely beat the baseline best"
                    )
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------- Phase 2: solvers
def compare_phase_solver_variants(
    context: _RefinementContext,
    *,
    alpha: float,
    degree: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare supported phase-synthesis solver settings; mark unsupported variants honestly."""

    import time

    entry = context.entry
    try:
        state = qsvt_target_readout(context.H, context.r, alpha=float(alpha), degree=int(degree))
    except Exception:
        return []
    singular_values_A = np.asarray(state.singular_values, dtype=np.float64) / max(
        float(state.singular_values[0]), np.finfo(np.float64).eps
    )
    domain_min = _domain_min(singular_values_A)
    alpha_norm = float(state.alpha_norm)
    support = compute_singular_support(context.H, context.r, alpha=float(alpha))
    _scale_C, coefficients, fit_ok = _bounded_scale_and_polynomial(
        singular_values_A, alpha_norm=alpha_norm, degree=int(degree)
    )
    ridge = state.ridge_update
    poly_available = bool(state.polynomial_available)

    rows: list[dict[str, Any]] = []
    for variant, parameters in SUPPORTED_SOLVER_VARIANTS:
        base = {
            "case": entry["case"],
            "subproblem_id": entry["subproblem_id"],
            "subproblem_type": entry["subproblem_type"],
            "alpha": float(alpha),
            "degree": int(degree),
            "phase_solver_variant": variant,
            "solver_supported": True,
            "solver_parameters": parameters,
        }
        if coefficients is None or not fit_ok:
            rows.append({**base, **_unsupported_variant_metrics("polynomial_fit_unavailable")})
            continue
        if variant == "root_finding_angle_solver" and int(degree) > _MAX_VARIANT_DEGREE:
            # the root-finding angle solver does not scale to very high degree; recorded honestly
            rows.append(
                {
                    **base,
                    **_unsupported_variant_metrics(
                        f"root_finding_does_not_scale_to_degree_{int(degree)}"
                    ),
                }
            )
            continue
        angle_solver = "root-finding" if variant == "root_finding_angle_solver" else "iterative"
        grid_size = 16385 if variant == "high_precision_grid" else None
        started = time.perf_counter()
        phases, status = _synthesize_phases(coefficients, angle_solver=angle_solver)
        if phases is None:
            rows.append(
                {**base, **_unsupported_variant_metrics(status), "runtime_seconds": float("nan")}
            )
            continue
        metrics = _phase_fidelity(
            phases,
            coefficients,
            singular_values_A=singular_values_A,
            domain_min=domain_min,
            alpha_norm=alpha_norm,
            support_weight=support.combined_weight,
        )
        gate = gate_level_subproblem_readout(
            context.H,
            context.r,
            alpha=float(alpha),
            degree=int(degree),
            seed=int(seed),
            case=entry["case"],
            angle_solver=angle_solver,
            phase_grid_size=grid_size,
        )
        runtime = float(time.perf_counter() - started)
        boundedness_ok = bool(metrics["dense_grid_max_abs_value"] <= 1.0 + _BOUND_TOL)
        overshoot_flag = not boundedness_ok
        if gate.get("available"):
            gate_run_status = "computed"
            gate_update = gate["gate_update"]
            gate_error_vs_poly = (
                _rel_l2(gate_update, state.polynomial_update) if poly_available else float("nan")
            )
            gate_error_vs_ridge = _rel_l2(gate_update, ridge)
        else:
            gate_run_status = "failed"
            gate_error_vs_poly = float("nan")
            gate_error_vs_ridge = float("nan")
        phase_status = (
            "phase_synthesis_degraded"
            if (overshoot_flag or metrics["phase_objective"] > _PHASE_DEGRADED_TOL)
            else "synthesized"
        )
        rows.append(
            {
                **base,
                "phase_synthesis_status": phase_status,
                "phase_objective": metrics["phase_objective"],
                "phase_residual": metrics["phase_residual"],
                "max_response_error_dense_grid": metrics["dense_grid_max_error"],
                "max_response_error_singular_support": metrics[
                    "max_response_error_singular_support"
                ],
                "weighted_response_error_singular_support": metrics[
                    "weighted_response_error_singular_support"
                ],
                "boundedness_ok": boundedness_ok,
                "overshoot_flag": overshoot_flag,
                "gate_run_status": gate_run_status,
                "gate_error_vs_polynomial": gate_error_vs_poly,
                "gate_error_vs_ridge": gate_error_vs_ridge,
                "runtime_seconds": runtime,
                "selected_variant_candidate": bool(
                    boundedness_ok
                    and not overshoot_flag
                    and gate_run_status == "computed"
                    and phase_status == "synthesized"
                ),
                "status": "computed",
                "notes": f"angle_solver={angle_solver}; grid={grid_size or 'default'}",
            }
        )
    for variant, reason in UNSUPPORTED_SOLVER_VARIANTS:
        rows.append(
            {
                "case": entry["case"],
                "subproblem_id": entry["subproblem_id"],
                "subproblem_type": entry["subproblem_type"],
                "alpha": float(alpha),
                "degree": int(degree),
                "phase_solver_variant": variant,
                "solver_supported": False,
                "solver_parameters": "",
                **_unsupported_variant_metrics(f"unsupported_solver_variant: {reason}"),
            }
        )
    return rows


def _unsupported_variant_metrics(reason: str) -> dict[str, Any]:
    return {
        "phase_synthesis_status": "unsupported_solver_variant",
        "phase_objective": float("nan"),
        "phase_residual": float("nan"),
        "max_response_error_dense_grid": float("nan"),
        "max_response_error_singular_support": float("nan"),
        "weighted_response_error_singular_support": float("nan"),
        "boundedness_ok": False,
        "overshoot_flag": False,
        "gate_run_status": "not_run",
        "gate_error_vs_polynomial": float("nan"),
        "gate_error_vs_ridge": float("nan"),
        "runtime_seconds": float("nan"),
        "selected_variant_candidate": False,
        "status": "unsupported",
        "notes": reason,
    }


def select_best_solver_variant(
    variant_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Pick, per subproblem, the lowest gate-error supported variant among valid candidates."""

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in variant_rows:
        if row.get("selected_variant_candidate"):
            by_pair.setdefault((row["case"], row["subproblem_type"]), []).append(row)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, candidates in by_pair.items():
        selected[pair] = min(
            candidates,
            key=lambda c: (
                _safe_float(c["gate_error_vs_ridge"]),
                _safe_float(c["gate_error_vs_polynomial"]),
                _safe_float(c["weighted_response_error_singular_support"]),
                int(c["degree"]),
            ),
        )
    return selected


def phase_solver_variant_summary_markdown(
    variant_rows: list[dict[str, Any]], selected: dict[tuple[str, str], dict[str, Any]]
) -> str:
    supported = sorted({r["phase_solver_variant"] for r in variant_rows if r["solver_supported"]})
    unsupported = sorted(
        {r["phase_solver_variant"] for r in variant_rows if not r["solver_supported"]}
    )
    lines = [
        "# Phase-Synthesis Solver Variant Comparison",
        "",
        REFINEMENT_CLAIM,
        "",
        "## Variants",
        f"- Supported (actually run): {', '.join(supported) or 'none'}",
        f"- Unsupported (recorded, not run): {', '.join(unsupported) or 'none'}",
        "",
        "Phase synthesis failure is a property of the angle solver, not of the readout protocol.",
        "",
        "## Selected variant per subproblem",
        "| case | subproblem | variant | degree | gate err vs ridge | gate vs poly |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for pair, row in selected.items():
        lines.append(
            f"| {pair[0]} | {pair[1]} | {row['phase_solver_variant']} | {int(row['degree'])} | "
            f"{_safe_float(row['gate_error_vs_ridge']):.3f} | "
            f"{_safe_float(row['gate_error_vs_polynomial']):.4f} |"
        )
    if not selected:
        lines.append("| (none) | | | | | |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- Phase 3: weighted fit
def _odd_design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    exponents = np.arange(1, int(degree) + 1, 2)
    return np.power.outer(np.asarray(x, dtype=np.float64), exponents)


def _support_weights(support: Any, weight_mode: str) -> np.ndarray:
    if weight_mode == "residual_energy_weight":
        raw = np.asarray(support.residual_projection_weight, dtype=np.float64) ** 2
    elif weight_mode == "update_contribution_weight":
        raw = np.asarray(support.update_magnitude_weight, dtype=np.float64) ** 2
    else:  # uniform_support_weight
        raw = np.ones_like(np.asarray(support.sigma_normalized, dtype=np.float64))
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.ones_like(raw) / max(raw.size, 1)
    return raw / total


def _fit_weighted_odd_polynomial(
    *,
    singular_values_A: np.ndarray,
    support_weight: np.ndarray,
    alpha_norm: float,
    domain_min: float,
    degree: int,
    fit_mode: str,
) -> tuple[np.ndarray, float]:
    """Weighted least-squares odd-polynomial fit of the bounded Ridge target.

    The fit grid is a dense uniform background (boundedness/shape control) augmented by
    Gaussian bumps centred on the normalized singular values, scaled by ``support_weight``. The
    fit mode controls how strongly the singular support is emphasized over the uniform
    background. Returns ``(bounded_coefficients, scale_C)``.
    """

    grid = np.linspace(domain_min, 1.0, 600, dtype=np.float64)
    _target_bounded, scale_C = _bounded_target_and_scale(
        singular_values_A, alpha_norm=alpha_norm, domain_min=domain_min
    )
    target_grid = (grid / (grid * grid + float(alpha_norm))) / scale_C

    background = np.ones_like(grid)
    if fit_mode == "global_uniform_fit":
        weights = background
    else:
        focus = {
            "weighted_singular_support_fit": 8.0,
            "residual_aware_fit": 8.0,
            "hybrid_global_plus_support_fit": 3.0,
        }.get(fit_mode, 4.0)
        bumps = np.zeros_like(grid)
        width = max(0.01, 0.05 * (1.0 - domain_min))
        for sigma, weight in zip(singular_values_A, support_weight, strict=False):
            bumps += float(weight) * np.exp(-0.5 * ((grid - float(sigma)) / width) ** 2)
        bump_sum = float(np.sum(bumps))
        if bump_sum > 0.0:
            bumps = bumps / bump_sum * float(np.sum(background)) * focus
        weights = background + bumps

    design = _odd_design_matrix(grid, degree)
    sqrt_w = np.sqrt(weights)
    coeffs_odd, *_ = np.linalg.lstsq(design * sqrt_w[:, None], target_grid * sqrt_w, rcond=None)
    full = np.zeros(int(degree) + 1, dtype=np.float64)
    full[1::2] = coeffs_odd
    # renormalize so the realized polynomial is bounded by 1 on [-1, 1]; absorb into C.
    dense = np.linspace(-1.0, 1.0, 4097, dtype=np.float64)
    peak = float(np.max(np.abs(Polynomial(full)(dense))))
    if peak > 1.0:
        full = full / peak
        scale_C = scale_C * peak
    return full, scale_C


FIT_MODE_WEIGHT_PAIRS: tuple[tuple[str, str], ...] = (
    ("global_uniform_fit", "uniform_support_weight"),
    ("weighted_singular_support_fit", "residual_energy_weight"),
    ("weighted_singular_support_fit", "update_contribution_weight"),
    ("residual_aware_fit", "residual_energy_weight"),
    ("hybrid_global_plus_support_fit", "update_contribution_weight"),
)


def evaluate_fit_modes(
    context: _RefinementContext,
    *,
    alpha: float,
    degree: int,
    seed: int,
    cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    gate_confirm_top: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare polynomial fit modes; return (per-singular-value rows, comparison rows).

    All modes are compared at the polynomial/singular-support level (the dominant
    ``exact_target_vs_polynomial`` error source) with a dense-grid boundedness safety check. The
    ``gate_confirm_top`` most promising *bounded* fits (lowest support-weighted error) are then
    confirmed at the real gate level; the rest report polynomial-level metrics only. Unsafe
    (unbounded) fits are flagged diagnostic-only and never gate-confirmed or selected.
    """

    entry = context.entry
    try:
        state = qsvt_target_readout(context.H, context.r, alpha=float(alpha), degree=int(degree))
    except Exception:
        return [], []
    singular_values_A = np.asarray(state.singular_values, dtype=np.float64) / max(
        float(state.singular_values[0]), np.finfo(np.float64).eps
    )
    domain_min = _domain_min(singular_values_A)
    alpha_norm = float(state.alpha_norm)
    support = compute_singular_support(context.H, context.r, alpha=float(alpha))
    unbounded_support = singular_values_A / (singular_values_A**2 + alpha_norm)
    scale_ref = max(float(np.max(np.abs(unbounded_support))), np.finfo(np.float64).eps)
    target_support = unbounded_support / scale_ref
    domain_grid = np.linspace(domain_min, 1.0, 2049, dtype=np.float64)
    domain_target = (domain_grid / (domain_grid**2 + alpha_norm)) / scale_ref
    dense = np.linspace(-1.0, 1.0, 4097, dtype=np.float64)

    detail_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for fit_mode, weight_mode in FIT_MODE_WEIGHT_PAIRS:
        support_weight = _support_weights(support, weight_mode)
        coefficients, _scale_C = _fit_weighted_odd_polynomial(
            singular_values_A=singular_values_A,
            support_weight=support_weight,
            alpha_norm=alpha_norm,
            domain_min=domain_min,
            degree=int(degree),
            fit_mode=fit_mode,
        )
        poly_support = Polynomial(coefficients)(singular_values_A)
        pointwise = np.abs(poly_support - target_support)
        weighted_sq = support_weight * pointwise**2
        support_weighted_error = float(np.sqrt(np.sum(weighted_sq)))
        dense_max_abs = float(np.max(np.abs(Polynomial(coefficients)(dense))))
        dense_grid_max_error = float(
            np.max(np.abs(Polynomial(coefficients)(domain_grid) - domain_target))
        )
        boundedness_ok = dense_max_abs <= 1.0 + _BOUND_TOL
        for index, sigma in enumerate(singular_values_A):
            detail_rows.append(
                {
                    "case": entry["case"],
                    "subproblem_id": entry["subproblem_id"],
                    "subproblem_type": entry["subproblem_type"],
                    "alpha": float(alpha),
                    "degree": int(degree),
                    "fit_mode": fit_mode,
                    "weight_mode": weight_mode,
                    "singular_value_index": int(index),
                    "singular_value": float(sigma),
                    "residual_component": float(support.residual_projection[index]),
                    "ridge_gain": float(support.ridge_filter_value[index]),
                    "target_value": float(target_support[index]),
                    "polynomial_value": float(poly_support[index]),
                    "pointwise_error": float(pointwise[index]),
                    "weight": float(support_weight[index]),
                    "weighted_squared_error": float(weighted_sq[index]),
                    "status": "computed",
                    "notes": "weighted singular-support fit detail",
                }
            )
        records.append(
            {
                "fit_mode": fit_mode,
                "weight_mode": weight_mode,
                "support_weighted_error": support_weighted_error,
                "support_max_error": float(np.max(pointwise)),
                "dense_grid_max_error": dense_grid_max_error,
                "boundedness_ok": boundedness_ok,
            }
        )

    confirm = sorted(
        (rec for rec in records if rec["boundedness_ok"]),
        key=lambda rec: rec["support_weighted_error"],
    )[: max(0, int(gate_confirm_top))]
    confirm_keys = {(rec["fit_mode"], rec["weight_mode"]) for rec in confirm}

    comparison_rows: list[dict[str, Any]] = []
    for record in records:
        boundedness_ok = record["boundedness_ok"]
        key = (record["fit_mode"], record["weight_mode"])
        gate_run_status = "not_run"
        gate_error_vs_poly = float("nan")
        gate_error_vs_ridge = float("nan")
        residual_ratio = float("nan")
        success_probability = float("nan")
        if not boundedness_ok:
            gate_run_status = "skipped_unbounded"
        elif key in confirm_keys:
            gate = realize_config_gate(
                context,
                alpha=float(alpha),
                degree=int(degree),
                fit_mode=record["fit_mode"],
                weight_mode=record["weight_mode"],
                seed=int(seed),
                cache=cache,
            )
            if gate.get("available"):
                gate_run_status = "computed"
                gate_error_vs_ridge = _safe_float(gate["gate_error_vs_ridge"])
                gate_error_vs_poly = _safe_float(gate["gate_error_vs_polynomial"])
                residual_ratio = _safe_float(gate["residual_ratio_after_gate"])
                success_probability = _safe_float(gate["success_probability"])
            else:
                gate_run_status = "failed"
        else:
            gate_run_status = "polynomial_level_only"
        postselection_overhead = (
            1.0 / success_probability
            if np.isfinite(success_probability) and success_probability > 0.0
            else float("nan")
        )
        eligible = bool(boundedness_ok and gate_run_status == "computed")
        status = "computed" if boundedness_ok else "diagnostic_only_unsafe_fit"
        comparison_rows.append(
            {
                "case": entry["case"],
                "subproblem_id": entry["subproblem_id"],
                "subproblem_type": entry["subproblem_type"],
                "alpha": float(alpha),
                "degree": int(degree),
                "fit_mode": record["fit_mode"],
                "weight_mode": record["weight_mode"],
                "support_weighted_error": record["support_weighted_error"],
                "support_max_error": record["support_max_error"],
                "dense_grid_max_error": record["dense_grid_max_error"],
                "boundedness_ok": boundedness_ok,
                "overshoot_flag": not boundedness_ok,
                "phase_synthesis_status": "synthesized" if boundedness_ok else "unbounded_fit",
                "gate_run_status": gate_run_status,
                "gate_error_vs_polynomial": gate_error_vs_poly,
                "gate_error_vs_ridge": gate_error_vs_ridge,
                "residual_ratio_after_gate": residual_ratio,
                "success_probability": success_probability,
                "postselection_overhead": postselection_overhead,
                "eligible_for_best_config": eligible,
                "status": status,
                "notes": (
                    "weighted singular-support fit; unsafe fits are diagnostic-only and never "
                    "selected as a main result"
                    if not boundedness_ok
                    else (
                        "bounded weighted fit confirmed at the gate level"
                        if gate_run_status == "computed"
                        else "bounded weighted fit; polynomial-level comparison (gate not "
                        "re-confirmed for this lower-ranked mode)"
                    )
                ),
            }
        )
    return detail_rows, comparison_rows


# --------------------------------------------------------------------------- Phase 4: three-view
def _best_eligible_fit(
    fit_rows: list[dict[str, Any]], pair: tuple[str, str], alpha: float
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in fit_rows
        if (row["case"], row["subproblem_type"]) == pair
        and row.get("eligible_for_best_config")
        and abs(round(_safe_float(row["alpha"]), 12) - round(float(alpha), 12)) < 1.0e-15
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _safe_float(row["gate_error_vs_ridge"]))


def build_three_view_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    matched_view_rows: list[dict[str, Any]],
    validated_view_rows: list[dict[str, Any]],
    fit_comparison_rows: list[dict[str, Any]],
    contexts: dict[tuple[str, str], _RefinementContext],
    target_error: float,
) -> list[dict[str, Any]]:
    """Three views per subproblem: matched common-reference, matched refined-QSP, relaxed-alpha.

    The refined-QSP and relaxed views adopt the best *eligible* (bounded, gate-confirmed) weighted
    singular-support fit when it lowers the same-alpha gate-vs-Ridge error, falling back to the
    existing co-design configs otherwise. Selection uses only same-alpha Ridge, boundedness, and
    gate metrics -- never true-state RMSE.
    """

    matched_by_pair = {(r["case"], r["subproblem_type"]): r for r in matched_view_rows}
    validated_by_pair = {(r["case"], r["subproblem_type"]): r for r in validated_view_rows}

    rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        pair = (baseline["case"], baseline["subproblem_type"])
        if pair not in contexts:
            continue
        matched = matched_by_pair.get(pair, {})
        validated = validated_by_pair.get(pair, {})
        matched_error = _safe_float(
            matched.get("gate_error_vs_ridge", baseline.get("matched_alpha_error"))
        )
        validated_alpha = _safe_float(validated.get("alpha", baseline.get("current_best_alpha")))
        validated_error = _safe_float(
            validated.get("gate_error_vs_ridge", baseline.get("current_best_error"))
        )

        rows.append(
            _three_view_row(
                baseline,
                view="matched_alpha_common_reference",
                alpha=MATCHED_ALPHA,
                degree=int(_safe_float(baseline.get("matched_degree", -1))),
                phase_solver_variant="default",
                fit_mode="global_uniform_fit",
                weight_mode="uniform_support_weight",
                gate_error=matched_error,
                ridge_shift=0.0,
                exact_poly=_safe_float(
                    matched.get(
                        "finite_polynomial_error", baseline.get("exact_target_vs_polynomial_error")
                    )
                ),
                poly_gate=_safe_float(matched.get("gate_vs_polynomial_error")),
                shot_error=_safe_float(matched.get("shot_readout_error")),
                target_error=target_error,
                interpretation="common classical-benchmark diagnostic reference",
                source="full_vector_readout/readout_matched_alpha_view.csv",
            )
        )

        # View 2: refined QSP at the common alpha (best eligible weighted fit at 1e-4).
        refined_fit = _best_eligible_fit(fit_comparison_rows, pair, MATCHED_ALPHA)
        refined_error = (
            _safe_float(refined_fit["gate_error_vs_ridge"]) if refined_fit else matched_error
        )
        rows.append(
            _three_view_row(
                baseline,
                view="matched_alpha_refined_qsp",
                alpha=MATCHED_ALPHA,
                degree=int(
                    _safe_float(
                        (refined_fit or {}).get("degree", baseline.get("matched_degree", -1))
                    )
                ),
                phase_solver_variant="default",
                fit_mode=str((refined_fit or {}).get("fit_mode", "global_uniform_fit")),
                weight_mode=str((refined_fit or {}).get("weight_mode", "uniform_support_weight")),
                gate_error=refined_error,
                ridge_shift=0.0,
                exact_poly=_safe_float(
                    (refined_fit or {}).get(
                        "support_weighted_error", baseline.get("exact_target_vs_polynomial_error")
                    )
                ),
                poly_gate=_safe_float((refined_fit or {}).get("gate_error_vs_polynomial")),
                shot_error=float("nan"),
                target_error=target_error,
                interpretation=(
                    "weighted singular-support QSP refinement at the common alpha (no alpha change)"
                    if refined_fit
                    else "no eligible refined fit improved the common-alpha config"
                ),
                source="phase_synthesis_refinement/polynomial_fit_mode_comparison.csv",
            )
        )

        # View 3: relaxed-alpha per-subproblem best (existing validated vs best weighted fit).
        relaxed_fit = _best_eligible_fit(fit_comparison_rows, pair, validated_alpha)
        use_fit = bool(
            relaxed_fit
            and np.isfinite(_safe_float(relaxed_fit["gate_error_vs_ridge"]))
            and (
                not np.isfinite(validated_error)
                or _safe_float(relaxed_fit["gate_error_vs_ridge"]) < validated_error
            )
        )
        relaxed_error = (
            _safe_float(relaxed_fit["gate_error_vs_ridge"]) if use_fit else validated_error
        )
        rows.append(
            _three_view_row(
                baseline,
                view="per_subproblem_relaxed_alpha_qsvt_validated",
                alpha=validated_alpha,
                degree=int(
                    _safe_float(
                        (relaxed_fit or {}).get(
                            "degree",
                            validated.get("degree", baseline.get("current_best_degree", -1)),
                        )
                        if use_fit
                        else validated.get("degree", baseline.get("current_best_degree", -1))
                    )
                ),
                phase_solver_variant="default",
                fit_mode=str(relaxed_fit["fit_mode"]) if use_fit else "global_uniform_fit",
                weight_mode=str(relaxed_fit["weight_mode"])
                if use_fit
                else "uniform_support_weight",
                gate_error=relaxed_error,
                ridge_shift=_ridge_alpha_shift(
                    contexts[pair], validated_alpha, reference_alpha=MATCHED_ALPHA
                ),
                exact_poly=_safe_float(
                    validated.get(
                        "finite_polynomial_error", baseline.get("exact_target_vs_polynomial_error")
                    )
                ),
                poly_gate=_safe_float(
                    (relaxed_fit or {}).get(
                        "gate_error_vs_polynomial",
                        validated.get(
                            "gate_vs_polynomial_error", baseline.get("polynomial_vs_gate_error")
                        ),
                    )
                ),
                shot_error=_safe_float(validated.get("shot_readout_error")),
                target_error=target_error,
                interpretation=(
                    "best per-subproblem QSVT-aware config (weighted singular-support fit); "
                    "gate error is vs Ridge at the same alpha"
                    if use_fit
                    else "best QSVT-aware per-subproblem alpha/degree; gate error vs Ridge "
                    "at the same alpha"
                ),
                source=(
                    "phase_synthesis_refinement/polynomial_fit_mode_comparison.csv"
                    if use_fit
                    else "full_vector_readout/readout_per_subproblem_validated_view.csv"
                ),
            )
        )
    return rows


def _ridge_alpha_shift(
    context: _RefinementContext, alpha: float, *, reference_alpha: float = MATCHED_ALPHA
) -> float:
    """Relative deviation ``||x_ridge(alpha) - x_ridge(ref)|| / ||x_ridge(ref)||``."""

    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        return float("nan")
    try:
        ridge_alpha = qsvt_target_readout(
            context.H, context.r, alpha=float(alpha), degree=15
        ).ridge_update
        ridge_ref = qsvt_target_readout(
            context.H, context.r, alpha=float(reference_alpha), degree=15
        ).ridge_update
    except Exception:
        return float("nan")
    return _rel_l2(ridge_alpha, ridge_ref)


def _three_view_status(view: str, gate_error: float) -> str:
    if not np.isfinite(gate_error):
        return "missing_valid_config"
    if gate_error <= PASS_THRESHOLD:
        return "pass"
    if gate_error <= GOOD_GATE_ERROR:
        return "improved_but_polynomial_limited"
    return "qsvt_polynomial_limited"


def _three_view_row(
    baseline: dict[str, Any],
    *,
    view: str,
    alpha: float,
    degree: int,
    phase_solver_variant: str,
    fit_mode: str,
    weight_mode: str,
    gate_error: float,
    ridge_shift: float,
    exact_poly: float,
    poly_gate: float,
    shot_error: float,
    target_error: float,
    interpretation: str,
    source: str,
) -> dict[str, Any]:
    success = _safe_float(baseline.get("success_probability"))
    overhead = 1.0 / success if np.isfinite(success) and success > 0 else float("nan")
    estimated_shots = (
        int(np.ceil(1.0 / float(target_error) ** 2)) if np.isfinite(target_error) else -1
    )
    return {
        "case": baseline["case"],
        "subproblem_id": baseline["subproblem_id"],
        "subproblem_type": baseline["subproblem_type"],
        "view": view,
        "alpha": float(alpha),
        "degree": int(degree),
        "phase_solver_variant": phase_solver_variant,
        "fit_mode": fit_mode,
        "weight_mode": weight_mode,
        "gate_error_vs_matching_ridge_alpha": gate_error,
        "ridge_alpha_shift_vs_common_alpha": ridge_shift,
        "exact_target_vs_polynomial_error": exact_poly,
        "polynomial_vs_gate_error": poly_gate,
        "shot_readout_error": shot_error,
        "success_probability": success,
        "postselection_overhead": overhead,
        "estimated_total_readout_shots": estimated_shots,
        "status": _three_view_status(view, gate_error),
        "interpretation": interpretation,
        "source_artifact": source,
        "notes": "three-view readout; relaxed-alpha gate error is vs Ridge at the same alpha",
    }


def three_view_summary_markdown(rows: list[dict[str, Any]]) -> str:
    views = [
        "matched_alpha_common_reference",
        "matched_alpha_refined_qsp",
        "per_subproblem_relaxed_alpha_qsvt_validated",
    ]
    lines = [
        "# Three-View Gate-Level Readout Summary",
        "",
        REFINEMENT_CLAIM,
        "",
        "## Views",
        "1. matched_alpha_common_reference: alpha=1e-4, original degree/phase; common benchmark.",
        "2. matched_alpha_refined_qsp: alpha=1e-4 with the best safe degree/solver/fit.",
        "3. per_subproblem_relaxed_alpha_qsvt_validated: best per-subproblem alpha/degree.",
        "",
        "Relaxed-alpha gate error is measured against Ridge/Tikhonov at the SAME alpha; the "
        "alpha shift versus the common alpha is reported separately. Changing alpha does not make "
        "QSVT outperform the matched-alpha Ridge estimator.",
        "",
        "| case | subproblem | view | alpha | deg | gate err | ridge alpha-shift | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for view in views:
        for row in [r for r in rows if r["view"] == view]:
            lines.append(
                f"| {row['case']} | {row['subproblem_type']} | {view} | "
                f"{_safe_float(row['alpha']):.0e} | {int(row['degree'])} | "
                f"{_safe_float(row['gate_error_vs_matching_ridge_alpha']):.3f} | "
                f"{_safe_float(row['ridge_alpha_shift_vs_common_alpha']):.3f} | {row['status']} |"
            )
    lines.extend(
        [
            "",
            "## What is not claimed",
            "- QSVT is not claimed to outperform Ridge/Tikhonov; Ridge is the reference target.",
            "- No efficient full IEEE-scale full-vector readout or quantum speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def metadata_mapped_limitation_note_markdown(rows: list[dict[str, Any]]) -> str:
    metadata_rows = [
        row
        for row in rows
        if row.get("case") == "ieee14" and row.get("subproblem_type") == "metadata_mapped"
    ]
    latest = next(
        (
            row
            for row in metadata_rows
            if row.get("view") == "matched_alpha_refined_qsp"
            or row.get("view") == "per_subproblem_relaxed_alpha_qsvt_validated"
        ),
        metadata_rows[0] if metadata_rows else {},
    )
    status = str(latest.get("status", "not_available"))
    error = _safe_float(latest.get("gate_error_vs_matching_ridge_alpha"))
    error_text = f"{error:.6g}" if np.isfinite(error) else "not_available"
    return "\n".join(
        [
            "# Metadata-Mapped Limitation Note",
            "",
            "This artifact is a limitation note for selected-subproblem finite-degree "
            "diagnostics only. It does not expand the project claim boundary or convert "
            "diagnostic evidence into performance evidence.",
            "",
            "The ieee14/metadata_mapped selected subproblem remains polynomial-limited after "
            "weighted singular-support refinement. This is reported as a finite-degree QSP "
            "limitation, not a readout failure and not a full QSVT failure.",
            "",
            f"- current status: {status}",
            f"- gate error vs matching Ridge/Tikhonov alpha: {error_text}",
            "- manuscript use: appendix or limitation evidence only",
            "",
        ]
    )


# --------------------------------------------------------------------------- Phase 5: leakage
def build_selection_leakage_audit(baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Document every selection stage and confirm no true-state RMSE / oracle access."""

    stages = [
        (
            "degree_47_49_diagnostic",
            "gate_error_vs_ridge (same-alpha Ridge reference)",
            dict(ridge=True, sv=True, resid=False, poly=True, gate=True),
        ),
        (
            "phase_solver_variant",
            "gate_error_vs_ridge, weighted_response_error_singular_support",
            dict(ridge=True, sv=True, resid=True, poly=True, gate=True),
        ),
        (
            "weighted_singular_support_fit",
            "support_weighted_error, dense-grid boundedness",
            dict(ridge=True, sv=True, resid=True, poly=True, gate=True),
        ),
        (
            "three_view_best",
            "gate_error_vs_matching_ridge_alpha, boundedness, success_probability",
            dict(ridge=True, sv=True, resid=False, poly=True, gate=True),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        for stage, metric, flags in stages:
            rows.append(
                {
                    "case": baseline["case"],
                    "subproblem_id": baseline["subproblem_id"],
                    "subproblem_type": baseline["subproblem_type"],
                    "selection_stage": stage,
                    "selection_metric": metric,
                    "uses_true_state": False,
                    "uses_state_rmse": False,
                    "uses_ridge_reference": flags["ridge"],
                    "uses_singular_values": flags["sv"],
                    "uses_residual": flags["resid"],
                    "uses_polynomial_error": flags["poly"],
                    "uses_gate_error": flags["gate"],
                    "allowed_for_selection": True,
                    "diagnostic_only": False,
                    "notes": (
                        "Ridge/Tikhonov is a classical reference target at the same alpha, not a "
                        "claim of QSVT superiority; no true-state RMSE or oracle alpha is used"
                    ),
                }
            )
    return rows


def selection_leakage_markdown(rows: list[dict[str, Any]]) -> str:
    any_true = any(r["uses_true_state"] or r["uses_state_rmse"] for r in rows)
    return "\n".join(
        [
            "# Selection Leakage Audit",
            "",
            REFINEMENT_CLAIM,
            "",
            "## Result",
            f"- Any stage uses true state: {any_true}",
            f"- Any stage uses true-state RMSE: {any_true}",
            "- Allowed selection metrics: singular values, weighted singular-support error, "
            "boundedness, phase-synthesis residual, gate-vs-polynomial error, same-alpha "
            "gate-vs-Ridge diagnostic, success probability, degree/cost penalty.",
            "- Forbidden metrics (true-state RMSE, oracle best alpha from x_true, field labels) "
            "are never used for selection.",
            "",
            "When same-alpha gate-vs-Ridge is used for selection, Ridge is a classical reference "
            "target and not a claim that QSVT outperforms Ridge.",
            "",
        ]
    )


# --------------------------------------------------------------------------- Phase 6: taxonomy
def build_failure_taxonomy(
    *,
    baseline_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify why each high-error/failed config remains limited."""

    rows: list[dict[str, Any]] = []
    baseline_by_pair = {(r["case"], r["subproblem_type"]): r for r in baseline_rows}
    for baseline in baseline_by_pair.values():
        best_error = _safe_float(baseline.get("current_best_error"))
        if not np.isfinite(best_error) or best_error <= GOOD_GATE_ERROR:
            rows.append(
                _taxonomy_row(
                    baseline,
                    alpha=_safe_float(baseline.get("current_best_alpha")),
                    degree=int(_safe_float(baseline.get("current_best_degree", -1))),
                    failure_label="within_threshold",
                    failure_layer="none",
                    evidence_metric="gate_error_vs_ridge",
                    evidence_value=best_error,
                    recommended_action="none",
                    note="best config already at or below the 0.10 threshold",
                )
            )
            continue
        poly_error = _safe_float(baseline.get("exact_target_vs_polynomial_error"))
        gate_vs_poly = _safe_float(baseline.get("polynomial_vs_gate_error"))
        if np.isfinite(gate_vs_poly) and gate_vs_poly > _GATE_SYNTHESIS_TOL:
            label, layer, metric, value = (
                "phase_synthesis_degraded",
                "phase_synthesis",
                "polynomial_vs_gate_error",
                gate_vs_poly,
            )
            action = "refine_angle_solver_or_reduce_degree"
        elif np.isfinite(poly_error) and poly_error > GOOD_GATE_ERROR:
            label, layer, metric, value = (
                "finite_degree_polynomial_error",
                "polynomial",
                "exact_target_vs_polynomial_error",
                poly_error,
            )
            action = "weighted_singular_support_fit_or_higher_stable_degree"
        else:
            label, layer, metric, value = (
                "ill_conditioned_singular_support",
                "spectrum",
                "gate_error_vs_ridge",
                best_error,
            )
            action = "report_as_polynomial_limited"
        rows.append(
            _taxonomy_row(
                baseline,
                alpha=_safe_float(baseline.get("current_best_alpha")),
                degree=int(_safe_float(baseline.get("current_best_degree", -1))),
                failure_label=label,
                failure_layer=layer,
                evidence_metric=metric,
                evidence_value=value,
                recommended_action=action,
                note="remaining limitation classified from the frozen baseline metrics",
            )
        )

    for cell in diagnostic_rows:
        if cell.get("overshoot_flag") and cell.get("gate_run_status") != "computed":
            rows.append(
                _taxonomy_row(
                    cell,
                    alpha=_safe_float(cell["alpha"]),
                    degree=int(cell["degree"]),
                    failure_label="overshoot_on_dense_grid",
                    failure_layer="boundedness",
                    evidence_metric="dense_grid_max_abs_value",
                    evidence_value=_safe_float(cell.get("dense_grid_max_abs_value")),
                    recommended_action="reject_degree_as_unbounded",
                    note="diagnostic degree 47/49 boundedness/overshoot evidence",
                )
            )
    for comp in comparison_rows:
        if not comp.get("boundedness_ok"):
            rows.append(
                _taxonomy_row(
                    comp,
                    alpha=_safe_float(comp["alpha"]),
                    degree=int(comp["degree"]),
                    failure_label="support_fit_good_dense_bad",
                    failure_layer="boundedness",
                    evidence_metric="dense_grid_max_error",
                    evidence_value=_safe_float(comp.get("dense_grid_max_error")),
                    recommended_action="reject_unbounded_weighted_fit",
                    note=f"weighted fit {comp.get('fit_mode')} unsafe on the dense grid",
                )
            )
    return rows


def _taxonomy_row(
    base: dict[str, Any],
    *,
    alpha: float,
    degree: int,
    failure_label: str,
    failure_layer: str,
    evidence_metric: str,
    evidence_value: float,
    recommended_action: str,
    note: str,
) -> dict[str, Any]:
    return {
        "case": base["case"],
        "subproblem_id": base["subproblem_id"],
        "subproblem_type": base["subproblem_type"],
        "alpha": float(alpha),
        "degree": int(degree),
        "phase_solver_variant": str(base.get("phase_solver_variant", "default")),
        "fit_mode": str(base.get("fit_mode", "global_uniform_fit")),
        "failure_label": failure_label,
        "failure_layer": failure_layer,
        "evidence_metric": evidence_metric,
        "evidence_value": float(evidence_value),
        "recommended_action": recommended_action,
        "status": "classified",
        "notes": note + "; readout protocol is not the failure source unless shot reconstruction "
        "dominates",
    }


def failure_taxonomy_markdown(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["failure_label"]] = counts.get(row["failure_label"], 0) + 1
    lines = [
        "# Phase-Synthesis Failure Taxonomy",
        "",
        REFINEMENT_CLAIM,
        "",
        "## Failure label counts",
        *[f"- {label}: {count}" for label, count in sorted(counts.items())],
        "",
        "No case is labelled a readout failure unless shot reconstruction dominates the error.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- Phase 7: cost
def build_cost_accuracy_rows(
    three_view_rows: list[dict[str, Any]],
    *,
    contexts: dict[tuple[str, str], _RefinementContext],
    seed: int,
    cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cost-accuracy and degree-cost trade-off tables from real (memoized) gate runs."""

    cost_rows: list[dict[str, Any]] = []
    degree_cost_rows: list[dict[str, Any]] = []
    seen_degree: set[tuple[str, str, int, float]] = set()
    for row in three_view_rows:
        pair = (row["case"], row["subproblem_type"])
        context = contexts.get(pair)
        degree = int(_safe_float(row["degree"]))
        alpha = _safe_float(row["alpha"])
        gate_error = _safe_float(row["gate_error_vs_matching_ridge_alpha"])
        success = _safe_float(row["success_probability"])
        overhead = _safe_float(row["postselection_overhead"])
        shots = int(_safe_float(row["estimated_total_readout_shots"]))
        query_count = max(degree, 0)
        depth = float("nan")
        gate_vs_poly = _safe_float(row.get("polynomial_vs_gate_error"))
        runtime = float("nan")
        if context is not None and degree > 0 and np.isfinite(alpha) and alpha > 0:
            gate = realize_config_gate(
                context,
                alpha=float(alpha),
                degree=int(degree),
                fit_mode=str(row.get("fit_mode", "global_uniform_fit")),
                weight_mode=str(row.get("weight_mode", "uniform_support_weight")),
                seed=int(seed),
                cache=cache,
            )
            if gate.get("available"):
                query_count = int(gate["query_count"])
                depth = _safe_float(gate.get("circuit_depth"))
                success = float(gate["success_probability"])
                overhead = 1.0 / success if success > 0 else float("nan")
                gate_vs_poly = _safe_float(gate.get("gate_error_vs_polynomial"))
        accuracy_cost_score = (
            gate_error * float(overhead)
            if np.isfinite(gate_error) and np.isfinite(overhead)
            else float("nan")
        )
        cost_rows.append(
            {
                "case": row["case"],
                "subproblem_id": row["subproblem_id"],
                "subproblem_type": row["subproblem_type"],
                "view": row["view"],
                "alpha": alpha,
                "degree": degree,
                "phase_solver_variant": row.get("phase_solver_variant", "default"),
                "fit_mode": row.get("fit_mode", "global_uniform_fit"),
                "gate_error_vs_matching_ridge_alpha": gate_error,
                "success_probability": success,
                "postselection_overhead": overhead,
                "estimated_total_readout_shots": shots,
                "query_count": query_count,
                "estimated_gate_depth": depth,
                "accuracy_cost_score": accuracy_cost_score,
                "status": "computed",
                "notes": "higher degree increases query count; cost includes postselection "
                "overhead",
            }
        )
        key = (row["case"], row["subproblem_type"], degree, round(alpha, 12))
        if key not in seen_degree and degree > 0:
            seen_degree.add(key)
            degree_cost_rows.append(
                {
                    "case": row["case"],
                    "subproblem_id": row["subproblem_id"],
                    "subproblem_type": row["subproblem_type"],
                    "alpha": alpha,
                    "degree": degree,
                    "query_count": query_count,
                    "estimated_gate_depth": depth,
                    "phase_synthesis_runtime": runtime,
                    "phase_synthesis_status": "synthesized",
                    "gate_error_vs_ridge": gate_error,
                    "gate_error_vs_polynomial": gate_vs_poly,
                    "success_probability": success,
                    "postselection_overhead": overhead,
                    "total_readout_shots": shots,
                    "status": "computed",
                    "notes": "degree-cost trade-off; no speedup is claimed",
                }
            )
    return cost_rows, degree_cost_rows


# --------------------------------------------------------------------------- Phase 8: sampling
def _gate_shot_trial(
    sample_state: np.ndarray,
    recovered_norm: float,
    ridge: np.ndarray,
    *,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    reference_index = int(np.argmax(np.abs(sample_state)))
    rng = np.random.default_rng(seed)
    sampling = sample_basis_magnitudes(sample_state, shots=int(shots), rng=rng)
    sign = recover_relative_signs(
        sample_state, shots=int(shots), rng=rng, reference_index=reference_index
    )
    recon = _protocol_global_sign(
        sign["estimated_signs"] * recovered_norm * sampling["estimated_magnitude"],
        reference_index,
    )
    aligned = _global_sign_aligned(recon, ridge)
    reliable = [status == "reliable" for status in sign["reliability_status"]]
    k = max(1, ridge.size // 2)
    top_k = len(_top_k(np.abs(aligned), k) & _top_k(np.abs(ridge), k)) / float(k)
    return {
        "vector_relative_l2_error": _rel_l2(aligned, ridge),
        "norm_relative_error": abs(float(np.linalg.norm(aligned)) - float(np.linalg.norm(ridge)))
        / max(float(np.linalg.norm(ridge)), 1.0e-15),
        "sign_accuracy_reliable": bool(np.all(reliable)),
        "fraction_reliable_signs": float(np.mean(reliable)),
        "top_k_match": float(top_k),
    }


def run_best_config_sampling_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Phase 8: re-run shot sampling for the selected best configs across shot levels and trials."""

    resolved = {
        "input_dir": "outputs/phase_synthesis_refinement",
        "readout_dir": "outputs/full_vector_readout",
        "input_root": "outputs",
        "output_dir": "outputs/phase_synthesis_refinement",
        "shots": [1_000, 10_000, 100_000],
        "trials": 30,
        "seed": DEFAULT_SEED,
    }
    resolved.update(config)
    input_dir = Path(resolved["input_dir"])
    output_dir = ensure_directory(resolved["output_dir"])
    seed = int(resolved["seed"])
    shot_levels = [int(s) for s in resolved["shots"]]
    trials = int(resolved["trials"])

    three_view = _read_csv(input_dir / "three_view_readout_summary.csv")
    if three_view.empty:
        _write_table(
            [], output_dir / "best_config_sampling_validation.csv", SAMPLING_VALIDATION_COLUMNS
        )
        _write_table(
            [],
            output_dir / "best_config_sampling_validation_summary.csv",
            SAMPLING_VALIDATION_SUMMARY_COLUMNS,
        )
        return {"output_dir": output_dir, "trial_rows": [], "summary_rows": []}

    selected = three_view[three_view["view"] == "per_subproblem_relaxed_alpha_qsvt_validated"]
    targeted = [(str(r["case"]), str(r["subproblem_type"])) for _, r in selected.iterrows()]
    contexts = _build_contexts(targeted, Path(resolved["input_root"]), seed=seed)

    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, view_row in selected.iterrows():
        pair = (str(view_row["case"]), str(view_row["subproblem_type"]))
        context = contexts.get(pair)
        alpha = _safe_float(view_row["alpha"])
        degree = int(_safe_float(view_row["degree"]))
        fit_mode = str(view_row.get("fit_mode", "global_uniform_fit"))
        weight_mode = str(view_row.get("weight_mode", "uniform_support_weight"))
        if context is None or not np.isfinite(alpha) or degree <= 0:
            continue
        gate = realize_config_gate(
            context,
            alpha=float(alpha),
            degree=int(degree),
            fit_mode=fit_mode,
            weight_mode=weight_mode,
            seed=seed,
        )
        if not gate.get("available"):
            continue
        ridge = np.asarray(gate["ridge_update"], dtype=np.float64)
        gate_state = np.asarray(gate["gate_state"], dtype=np.float64)
        gate_norm = float(np.linalg.norm(gate["gate_update"]))
        base_seed = _stable_base_seed(
            (pair[0], pair[1], f"alpha={alpha:.3e}", f"degree={degree}", fit_mode, weight_mode),
            seed,
        )
        for shots in shot_levels:
            errors: list[float] = []
            norm_errors: list[float] = []
            reliable_flags: list[bool] = []
            fractions: list[float] = []
            top_ks: list[float] = []
            for trial in range(trials):
                trial_seed = _trial_seed(base_seed, int(shots), trial)
                metrics = _gate_shot_trial(
                    gate_state, gate_norm, ridge, shots=int(shots), seed=trial_seed
                )
                errors.append(metrics["vector_relative_l2_error"])
                norm_errors.append(metrics["norm_relative_error"])
                reliable_flags.append(metrics["sign_accuracy_reliable"])
                fractions.append(metrics["fraction_reliable_signs"])
                top_ks.append(metrics["top_k_match"])
                trial_rows.append(
                    {
                        "case": pair[0],
                        "subproblem_id": str(view_row["subproblem_id"]),
                        "subproblem_type": pair[1],
                        "view": "per_subproblem_relaxed_alpha_qsvt_validated",
                        "alpha": alpha,
                        "degree": degree,
                        "phase_solver_variant": "default",
                        "fit_mode": fit_mode,
                        "shots": int(shots),
                        "trial_id": int(trial),
                        "rng_seed": int(trial_seed),
                        "vector_relative_l2_error": metrics["vector_relative_l2_error"],
                        "norm_relative_error": metrics["norm_relative_error"],
                        "sign_accuracy_reliable": metrics["sign_accuracy_reliable"],
                        "fraction_reliable_signs": metrics["fraction_reliable_signs"],
                        "top_k_match": metrics["top_k_match"],
                        "status": "computed",
                        "notes": "shot-based reconstruction of the actual gate state",
                    }
                )
            summary_rows.append(
                {
                    "case": pair[0],
                    "subproblem_id": str(view_row["subproblem_id"]),
                    "subproblem_type": pair[1],
                    "view": "per_subproblem_relaxed_alpha_qsvt_validated",
                    "alpha": alpha,
                    "degree": degree,
                    "shots": int(shots),
                    "trials": trials,
                    "mean_vector_relative_l2_error": float(np.mean(errors)),
                    "p50_vector_relative_l2_error": float(np.percentile(errors, 50)),
                    "p95_vector_relative_l2_error": float(np.percentile(errors, 95)),
                    "mean_norm_relative_error": float(np.mean(norm_errors)),
                    "mean_sign_accuracy_reliable": float(np.mean(reliable_flags)),
                    "mean_fraction_reliable_signs": float(np.mean(fractions)),
                    "top_k_match_rate": float(np.mean(top_ks)),
                    "status": "computed",
                    "notes": "stochastic stability of the selected best config; not "
                    "statevector-only",
                }
            )
    _write_table(
        trial_rows, output_dir / "best_config_sampling_validation.csv", SAMPLING_VALIDATION_COLUMNS
    )
    _write_table(
        summary_rows,
        output_dir / "best_config_sampling_validation_summary.csv",
        SAMPLING_VALIDATION_SUMMARY_COLUMNS,
    )
    return {"output_dir": output_dir, "trial_rows": trial_rows, "summary_rows": summary_rows}


# --------------------------------------------------------------------------- Phase 9: scope/figs
def refinement_scope_note_markdown(
    three_view_rows: list[dict[str, Any]], taxonomy_rows: list[dict[str, Any]]
) -> str:
    return "\n".join(
        [
            "# Phase-Synthesis Refinement Scope Note",
            "",
            REFINEMENT_CLAIM,
            "",
            "This phase refines QSP phase synthesis and finite-degree approximation for selected "
            "high-error subproblems only.",
            "",
            "The matched-alpha view is retained as a diagnostic common-reference setting.",
            "",
            "Relaxed-alpha results are compared against Ridge/Tikhonov at the same alpha and are "
            "not a claim that QSVT outperforms Ridge.",
            "",
            "Remaining high-error cases are reported as polynomial-, boundedness-, or "
            "phase-synthesis-limited.",
            "",
            "The result does not demonstrate efficient full IEEE-scale full-vector readout or "
            "quantum speedup.",
            "",
            "## Subclaim C10a.2 (phase-synthesis and finite-degree refinement)",
            "Per-subproblem QSP phase-synthesis and finite-degree approximation refinement is "
            "analyzed for the high-error selected subproblems. This subclaim refines C10a, is not "
            "an independent readout claim, and does not promote the full-scale assumption C10b.",
            "",
        ]
    )


def build_three_view_figure_rows(three_view_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in three_view_rows:
        value = _safe_float(row["gate_error_vs_matching_ridge_alpha"])
        rows.append(
            {
                "case": row["case"],
                "subproblem_type": row["subproblem_type"],
                "view": row["view"],
                "alpha": _safe_float(row["alpha"]),
                "degree": int(_safe_float(row["degree"])),
                "gate_error_vs_matching_ridge_alpha": abs(value)
                if np.isfinite(value)
                else float("nan"),
                "status": row["status"],
                "plot_group": f"{row['case']}/{row['subproblem_type']}",
                "notes": "three-view gate-level vector error"
                if np.isfinite(value)
                else "missing config; not fabricated",
            }
        )
    return rows


def build_failure_figure_rows(taxonomy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in taxonomy_rows:
        value = _safe_float(row["evidence_value"])
        rows.append(
            {
                "case": row["case"],
                "subproblem_type": row["subproblem_type"],
                "failure_layer": row["failure_layer"],
                "failure_label": row["failure_label"],
                "evidence_metric": row["evidence_metric"],
                "evidence_value": abs(value) if np.isfinite(value) else float("nan"),
                "notes": "failure-layer evidence"
                if np.isfinite(value)
                else "missing; not fabricated",
            }
        )
    return rows


def build_cost_figure_rows(cost_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in cost_rows:
        rows.append(
            {
                "case": row["case"],
                "subproblem_type": row["subproblem_type"],
                "view": row["view"],
                "degree": int(_safe_float(row["degree"])),
                "gate_error_vs_matching_ridge_alpha": _safe_float(
                    row["gate_error_vs_matching_ridge_alpha"]
                ),
                "estimated_total_readout_shots": int(
                    _safe_float(row["estimated_total_readout_shots"])
                ),
                "estimated_gate_depth": _safe_float(row["estimated_gate_depth"]),
                "notes": "cost-accuracy trade-off",
            }
        )
    return rows


# --------------------------------------------------------------------------- orchestration
def _targeted_from_baseline(baseline_rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    targeted: list[tuple[str, str]] = []
    for row in baseline_rows:
        if _safe_float(row.get("current_best_error")) > GOOD_GATE_ERROR:
            targeted.append((row["case"], row["subproblem_type"]))
    if not targeted:  # fall back to all baseline rows if none exceed the threshold
        targeted = [(r["case"], r["subproblem_type"]) for r in baseline_rows]
    return targeted


def run_phase_synthesis_refinement(config: dict[str, Any]) -> dict[str, Any]:
    """Phases 0-7, 9: baseline, degree diagnostic, solver variants, weighted fits, three-view,
    leakage audit, failure taxonomy, cost trade-offs, scope note, and figure data."""

    resolved: dict[str, Any] = {
        "input_root": "outputs",
        "readout_dir": "outputs/full_vector_readout",
        "output_dir": "outputs/phase_synthesis_refinement",
        "degree_grid_extra": list(DEGREE_DIAGNOSTIC_GRID),
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "refinement_degrees": list(DEFAULT_REFINEMENT_DEGREES),
        "shots": DEFAULT_SHOTS,
        "trials": DEFAULT_TRIALS,
        "seed": DEFAULT_SEED,
        "target_error": DEFAULT_TARGET_ERROR,
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    readout_dir = Path(resolved["readout_dir"])
    output_dir = ensure_directory(resolved["output_dir"])
    seed = int(resolved["seed"])
    target_error = float(resolved["target_error"])
    degree_extra = [int(d) for d in resolved["degree_grid_extra"]]
    refinement_degrees = [int(d) for d in resolved["refinement_degrees"]]

    # Phase 0
    baseline_rows = freeze_baseline(readout_dir)
    targeted = _targeted_from_baseline(baseline_rows)
    contexts = _build_contexts(targeted, input_root, seed=seed)

    # diagnostic-sweep alphas: each subproblem's current best alpha plus the matched alpha
    best_alpha_by_pair = {
        (r["case"], r["subproblem_type"]): _safe_float(r.get("current_best_alpha"))
        for r in baseline_rows
    }

    matched_view = _read_csv(readout_dir / "readout_matched_alpha_view.csv").to_dict("records")
    validated_view = _read_csv(readout_dir / "readout_per_subproblem_validated_view.csv").to_dict(
        "records"
    )

    gate_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    fit_detail_rows: list[dict[str, Any]] = []
    fit_comparison_rows: list[dict[str, Any]] = []
    for pair in targeted:
        context = contexts.get(pair)
        if context is None:
            continue
        best_alpha = best_alpha_by_pair.get(pair, MATCHED_ALPHA)
        diag_alphas = sorted({a for a in (best_alpha, MATCHED_ALPHA) if np.isfinite(a) and a > 0})
        # Phase 1: degree 47/49 diagnostic-only
        for degree in degree_extra:
            for alpha in diag_alphas:
                diagnostic_rows.append(
                    evaluate_degree_diagnostic_cell(
                        context, alpha=alpha, degree=degree, seed=seed, target_error=target_error
                    )
                )
        # Phase 2: solver variants at the per-subproblem best degree, capped at the highest
        # reliably-synthesizable degree. Phase 1 already shows degree >= 47 phase synthesis
        # degrades catastrophically regardless of solver, and the root-finding angle solver does
        # not scale to that regime, so testing solvers there is uninformative and expensive.
        pair_best_degrees = {
            min(int(_safe_float(r.get("current_best_degree"))), _MAX_VARIANT_DEGREE)
            for r in baseline_rows
            if (r["case"], r["subproblem_type"]) == pair
        }
        variant_degrees = sorted({d for d in pair_best_degrees if d > 0}) or list(
            refinement_degrees
        )
        variant_alpha = best_alpha if np.isfinite(best_alpha) and best_alpha > 0 else MATCHED_ALPHA
        for degree in variant_degrees:
            variant_rows.extend(
                compare_phase_solver_variants(
                    context, alpha=variant_alpha, degree=degree, seed=seed
                )
            )
        # Phase 3: weighted fits at the per-subproblem best degree (uncapped; the weighted fit
        # uses the iterative angle solver, which -- unlike root-finding -- is stable at degree 45,
        # and whether weighted fitting beats the d45 uniform synthesis ceiling is the question).
        uncapped_best = {
            int(_safe_float(r.get("current_best_degree")))
            for r in baseline_rows
            if (r["case"], r["subproblem_type"]) == pair
        }
        fit_degree = max((d for d in uncapped_best if d > 0), default=41)
        fit_alphas = sorted({a for a in (variant_alpha, MATCHED_ALPHA) if np.isfinite(a) and a > 0})
        for fit_alpha in fit_alphas:
            detail, comparison = evaluate_fit_modes(
                context, alpha=fit_alpha, degree=fit_degree, seed=seed, cache=gate_cache
            )
            fit_detail_rows.extend(detail)
            fit_comparison_rows.extend(comparison)

    degree_ceiling_rows = build_degree_ceiling_summary(diagnostic_rows, baseline_rows)
    selected_variant = select_best_solver_variant(variant_rows)

    # Phase 4
    three_view_rows = build_three_view_summary(
        baseline_rows=baseline_rows,
        matched_view_rows=matched_view,
        validated_view_rows=validated_view,
        fit_comparison_rows=fit_comparison_rows,
        contexts=contexts,
        target_error=target_error,
    )
    # Phase 5
    leakage_rows = build_selection_leakage_audit(baseline_rows)
    # Phase 6
    taxonomy_rows = build_failure_taxonomy(
        baseline_rows=baseline_rows,
        diagnostic_rows=diagnostic_rows,
        comparison_rows=fit_comparison_rows,
    )
    # Phase 7
    cost_rows, degree_cost_rows = build_cost_accuracy_rows(
        three_view_rows, contexts=contexts, seed=seed, cache=gate_cache
    )

    artifacts: dict[str, Path] = {}
    artifacts["baseline_current_readout_codesign"] = _write_table(
        baseline_rows, output_dir / "baseline_current_readout_codesign.csv", BASELINE_COLUMNS
    )
    (output_dir / "baseline_current_summary.md").write_text(
        baseline_summary_markdown(baseline_rows), encoding="utf-8"
    )
    artifacts["degree_47_49_diagnostic_sweep"] = _write_table(
        diagnostic_rows, output_dir / "degree_47_49_diagnostic_sweep.csv", DEGREE_DIAGNOSTIC_COLUMNS
    )
    artifacts["degree_ceiling_summary"] = _write_table(
        degree_ceiling_rows, output_dir / "degree_ceiling_summary.csv", DEGREE_CEILING_COLUMNS
    )
    artifacts["phase_solver_variant_comparison"] = _write_table(
        variant_rows, output_dir / "phase_solver_variant_comparison.csv", SOLVER_VARIANT_COLUMNS
    )
    (output_dir / "phase_solver_variant_summary.md").write_text(
        phase_solver_variant_summary_markdown(variant_rows, selected_variant), encoding="utf-8"
    )
    artifacts["weighted_singular_support_fit"] = _write_table(
        fit_detail_rows, output_dir / "weighted_singular_support_fit.csv", WEIGHTED_FIT_COLUMNS
    )
    artifacts["polynomial_fit_mode_comparison"] = _write_table(
        fit_comparison_rows,
        output_dir / "polynomial_fit_mode_comparison.csv",
        FIT_MODE_COMPARISON_COLUMNS,
    )
    artifacts["three_view_readout_summary"] = _write_table(
        three_view_rows, output_dir / "three_view_readout_summary.csv", THREE_VIEW_COLUMNS
    )
    (output_dir / "three_view_readout_summary.md").write_text(
        three_view_summary_markdown(three_view_rows), encoding="utf-8"
    )
    metadata_note_path = output_dir / "metadata_mapped_limitation_note.md"
    metadata_note_path.write_text(
        metadata_mapped_limitation_note_markdown(three_view_rows), encoding="utf-8"
    )
    artifacts["metadata_mapped_limitation_note"] = metadata_note_path
    artifacts["selection_leakage_audit"] = _write_table(
        leakage_rows, output_dir / "selection_leakage_audit.csv", SELECTION_LEAKAGE_COLUMNS
    )
    (output_dir / "selection_leakage_audit.md").write_text(
        selection_leakage_markdown(leakage_rows), encoding="utf-8"
    )
    artifacts["phase_failure_taxonomy"] = _write_table(
        taxonomy_rows, output_dir / "phase_failure_taxonomy.csv", FAILURE_TAXONOMY_COLUMNS
    )
    (output_dir / "phase_failure_taxonomy.md").write_text(
        failure_taxonomy_markdown(taxonomy_rows), encoding="utf-8"
    )
    artifacts["cost_accuracy_tradeoff"] = _write_table(
        cost_rows, output_dir / "cost_accuracy_tradeoff.csv", COST_ACCURACY_COLUMNS
    )
    artifacts["degree_cost_tradeoff"] = _write_table(
        degree_cost_rows, output_dir / "degree_cost_tradeoff.csv", DEGREE_COST_COLUMNS
    )
    # Phase 9
    (output_dir / "phase_synthesis_refinement_scope_note.md").write_text(
        refinement_scope_note_markdown(three_view_rows, taxonomy_rows), encoding="utf-8"
    )
    artifacts["figure_data_phase_refinement_three_view"] = _write_table(
        build_three_view_figure_rows(three_view_rows),
        output_dir / "figure_data_phase_refinement_three_view.csv",
        FIGURE_THREE_VIEW_COLUMNS,
    )
    artifacts["figure_data_phase_failure_taxonomy"] = _write_table(
        build_failure_figure_rows(taxonomy_rows),
        output_dir / "figure_data_phase_failure_taxonomy.csv",
        FIGURE_FAILURE_COLUMNS,
    )
    artifacts["figure_data_cost_accuracy_tradeoff"] = _write_table(
        build_cost_figure_rows(cost_rows),
        output_dir / "figure_data_cost_accuracy_tradeoff.csv",
        FIGURE_COST_COLUMNS,
    )
    return {
        "output_dir": output_dir,
        "targeted": list(targeted),
        "baseline_rows": baseline_rows,
        "diagnostic_rows": diagnostic_rows,
        "degree_ceiling_rows": degree_ceiling_rows,
        "variant_rows": variant_rows,
        "selected_variant": selected_variant,
        "fit_detail_rows": fit_detail_rows,
        "fit_comparison_rows": fit_comparison_rows,
        "three_view_rows": three_view_rows,
        "leakage_rows": leakage_rows,
        "taxonomy_rows": taxonomy_rows,
        "cost_rows": cost_rows,
        "degree_cost_rows": degree_cost_rows,
        "artifacts": artifacts,
    }


__all__ = [
    "REFINEMENT_CLAIM",
    "build_cost_accuracy_rows",
    "build_degree_ceiling_summary",
    "build_failure_taxonomy",
    "build_selection_leakage_audit",
    "build_three_view_summary",
    "compare_phase_solver_variants",
    "evaluate_degree_diagnostic_cell",
    "evaluate_fit_modes",
    "freeze_baseline",
    "gate_update_from_coefficients",
    "qsvt_phase_response",
    "refinement_scope_note_markdown",
    "run_best_config_sampling_validation",
    "run_phase_synthesis_refinement",
    "select_best_solver_variant",
    "three_view_summary_markdown",
]
