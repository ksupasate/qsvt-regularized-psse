"""Core numerical utilities for the TQE implementation-boundary revision.

The functions in this module are intentionally independent of ground truth unless
their name explicitly contains ``oracle`` or ``evaluation``.  This makes the
deployable/non-deployable alpha-selection boundary testable.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution


def normalized_regularization(alpha: float, beta: float) -> float:
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    return float(alpha) / float(beta) ** 2


def _validate_alpha_inputs(
    H: np.ndarray, r: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, ...]:
    matrix = np.asarray(H, dtype=np.float64)
    residual = np.asarray(r, dtype=np.float64)
    grid = np.asarray(alphas, dtype=np.float64)
    if matrix.ndim != 2 or residual.shape != (matrix.shape[0],):
        raise ValueError("H and r have incompatible dimensions")
    if grid.ndim != 1 or grid.size < 3 or np.any(grid <= 0.0):
        raise ValueError("alphas must contain at least three positive values")
    return matrix, residual, grid


def gcv_scores(H: np.ndarray, r: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Generalized cross-validation scores for Ridge without using truth."""

    matrix, residual, grid = _validate_alpha_inputs(H, r, alphas)
    U, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    uTr = U.T @ residual
    orthogonal_sq = max(float(residual @ residual - uTr @ uTr), 0.0)
    scores = []
    for alpha in grid:
        shrink = singular**2 / (singular**2 + alpha)
        residual_sq = float(np.sum(((1.0 - shrink) * uTr) ** 2) + orthogonal_sq)
        effective_residual_dof = float(matrix.shape[0] - np.sum(shrink))
        scores.append(residual_sq / max(effective_residual_dof**2, np.finfo(float).tiny))
    return np.asarray(scores)


def select_alpha_gcv(H: np.ndarray, r: np.ndarray, alphas: np.ndarray) -> float:
    _, _, grid = _validate_alpha_inputs(H, r, alphas)
    return float(grid[int(np.argmin(gcv_scores(H, r, grid)))])


def l_curve_curvature(H: np.ndarray, r: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Discrete log-log L-curve curvature; endpoints are marked ``-inf``."""

    matrix, residual, grid = _validate_alpha_inputs(H, r, alphas)
    order = np.argsort(grid)
    sorted_grid = grid[order]
    rho = []
    eta = []
    for alpha in sorted_grid:
        x = ridge_svd_solution(matrix, residual, alpha=float(alpha))
        rho.append(max(float(np.linalg.norm(matrix @ x - residual)), np.finfo(float).tiny))
        eta.append(max(float(np.linalg.norm(x)), np.finfo(float).tiny))
    t = np.log(sorted_grid)
    xlog = np.log(np.asarray(rho))
    ylog = np.log(np.asarray(eta))
    dx = np.gradient(xlog, t)
    dy = np.gradient(ylog, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-30)
    curvature[[0, -1]] = -np.inf
    restored = np.empty_like(curvature)
    restored[order] = curvature
    return restored


def select_alpha_l_curve(H: np.ndarray, r: np.ndarray, alphas: np.ndarray) -> float:
    _, _, grid = _validate_alpha_inputs(H, r, alphas)
    return float(grid[int(np.argmax(l_curve_curvature(H, r, grid)))])


def select_alpha_discrepancy(
    H: np.ndarray,
    r: np.ndarray,
    alphas: np.ndarray,
    *,
    target_weighted_residual: float | None = None,
) -> float:
    """Morozov discrepancy selection for an already-whitened system."""

    matrix, residual, grid = _validate_alpha_inputs(H, r, alphas)
    target = (
        math.sqrt(matrix.shape[0])
        if target_weighted_residual is None
        else float(target_weighted_residual)
    )
    if target <= 0.0:
        raise ValueError("target_weighted_residual must be positive")
    norms = np.asarray(
        [
            np.linalg.norm(
                matrix @ ridge_svd_solution(matrix, residual, alpha=float(alpha)) - residual
            )
            for alpha in grid
        ]
    )
    return float(grid[int(np.argmin(np.abs(norms - target)))])


def deterministic_validation_rows(n_rows: int, *, fold: int = 0, folds: int = 5) -> np.ndarray:
    if folds < 2 or fold < 0 or fold >= folds or n_rows < folds:
        raise ValueError("invalid deterministic fold configuration")
    return np.arange(n_rows, dtype=np.int64) % folds == fold


def heldout_scores(
    H: np.ndarray, r: np.ndarray, alphas: np.ndarray, *, fold: int = 0, folds: int = 5
) -> np.ndarray:
    matrix, residual, grid = _validate_alpha_inputs(H, r, alphas)
    validation = deterministic_validation_rows(matrix.shape[0], fold=fold, folds=folds)
    train = ~validation
    scores = []
    for alpha in grid:
        x = ridge_svd_solution(matrix[train], residual[train], alpha=float(alpha))
        scores.append(float(np.mean((matrix[validation] @ x - residual[validation]) ** 2)))
    return np.asarray(scores)


def select_alpha_heldout(
    H: np.ndarray, r: np.ndarray, alphas: np.ndarray, *, fold: int = 0, folds: int = 5
) -> float:
    _, _, grid = _validate_alpha_inputs(H, r, alphas)
    return float(grid[int(np.argmin(heldout_scores(H, r, grid, fold=fold, folds=folds)))])


def select_alpha_oracle_rmse(
    H: np.ndarray, r: np.ndarray, x_true: np.ndarray, alphas: np.ndarray
) -> float:
    """Simulation-only oracle diagnostic; never a deployable selector."""

    matrix, residual, grid = _validate_alpha_inputs(H, r, alphas)
    truth = np.asarray(x_true, dtype=np.float64)
    if truth.shape != (matrix.shape[1],):
        raise ValueError("x_true has the wrong shape")
    errors = [
        np.sqrt(np.mean((ridge_svd_solution(matrix, residual, alpha=float(alpha)) - truth) ** 2))
        for alpha in grid
    ]
    return float(grid[int(np.argmin(errors))])


NON_ORACLE_SELECTORS: dict[str, Callable[..., float]] = {
    "gcv": select_alpha_gcv,
    "l_curve": select_alpha_l_curve,
    "discrepancy": select_alpha_discrepancy,
    "heldout_rows": select_alpha_heldout,
}


def exact_integrated_readout_distribution(
    *, postselection_probability: float, signed_overlap: float
) -> dict[str, float]:
    """Exact joint distribution for the integrated interference readout.

    Keys are ``readout_bit flag_bit``. Flag 0 is the accepted interference
    branch. The distribution implies ``P(flag=0)=(1+p_succ)/2`` and
    ``E[z]=signed_overlap`` for z=+1/-1 on accepted shots and zero on failures.
    """

    p = float(postselection_probability)
    z = float(signed_overlap)
    if not 0.0 <= p <= 1.0:
        raise ValueError("postselection_probability must be in [0, 1]")
    if abs(z) > math.sqrt(max(p, 0.0)) + 1e-12:
        raise ValueError("signed overlap violates Cauchy-Schwarz")
    probabilities = {
        "00": (1.0 + p + 2.0 * z) / 4.0,
        "10": (1.0 + p - 2.0 * z) / 4.0,
        "01": (1.0 - p) / 4.0,
        "11": (1.0 - p) / 4.0,
    }
    if min(probabilities.values()) < -1e-12:
        raise ValueError("invalid integrated-readout probabilities")
    return {key: max(float(value), 0.0) for key, value in probabilities.items()}


def sample_integrated_readout(
    *,
    postselection_probability: float,
    signed_overlap: float,
    physical_scale: float,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    if shots <= 0 or physical_scale <= 0.0:
        raise ValueError("shots and physical_scale must be positive")
    distribution = exact_integrated_readout_distribution(
        postselection_probability=postselection_probability, signed_overlap=signed_overlap
    )
    keys = tuple(distribution)
    rng = np.random.default_rng(int(seed))
    draws = rng.multinomial(shots, [distribution[key] for key in keys])
    counts = {key: int(value) for key, value in zip(keys, draws, strict=True)}
    return {**estimate_integrated_counts(counts, physical_scale=physical_scale), "counts": counts}


def estimate_integrated_counts(
    counts: dict[str, int], *, physical_scale: float
) -> dict[str, float]:
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("empty shot record")
    accepted = int(counts.get("00", 0) + counts.get("10", 0))
    f_hat = accepted / total
    z_hat = (counts.get("00", 0) - counts.get("10", 0)) / total
    p_hat = 2.0 * f_hat - 1.0
    p_se = 2.0 * math.sqrt(max(f_hat * (1.0 - f_hat), 0.0) / total)
    z_se = math.sqrt(max(f_hat - z_hat**2, 0.0) / total)
    estimate = float(physical_scale) * z_hat
    standard_error = float(physical_scale) * z_se
    return {
        "total_shots": float(total),
        "accepted_shots": float(accepted),
        "acceptance_frequency": f_hat,
        "estimated_postselection_probability": p_hat,
        "postselection_standard_error": p_se,
        "conditional_signed_mean": (
            (counts.get("00", 0) - counts.get("10", 0)) / accepted if accepted else float("nan")
        ),
        "signed_overlap_estimate": z_hat,
        "signed_overlap_standard_error": z_se,
        "selected_output_estimate": estimate,
        "selected_output_standard_error": standard_error,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
    }


@dataclass(frozen=True)
class RegisterLedger:
    workload: str
    row_address_qubits: int
    column_state_qubits: int
    signal_block_encoding_ancillas: int
    qsvt_projector_ancillas: int
    residual_loader_ancillas: int
    readout_ancilla: int
    work_ancillas: int
    padded_data_dimension: int
    full_unitary_dimension: int
    reported_total_logical_qubits: int
    execution_status: str

    @property
    def register_sum(self) -> int:
        return (
            self.row_address_qubits
            + self.column_state_qubits
            + self.signal_block_encoding_ancillas
            + self.qsvt_projector_ancillas
            + self.residual_loader_ancillas
            + self.readout_ancilla
            + self.work_ancillas
        )

    def validated(self) -> dict[str, Any]:
        if self.register_sum != self.reported_total_logical_qubits:
            raise ValueError(
                f"register sum {self.register_sum} != reported total "
                f"{self.reported_total_logical_qubits} for {self.workload}"
            )
        return {**asdict(self), "register_sum": self.register_sum, "totals_match": True}


def _timed(call: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, float]]:
    for _ in range(3):
        result = call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    values = np.asarray(samples)
    return result, {
        "runtime_median_seconds": float(np.median(values)),
        "runtime_q1_seconds": float(np.percentile(values, 25)),
        "runtime_q3_seconds": float(np.percentile(values, 75)),
        "runtime_min_seconds": float(values.min()),
        "runtime_max_seconds": float(values.max()),
        "timed_repetitions": int(repeats),
    }


def access_matched_classical_baselines(
    H: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    *,
    alpha: float,
    repeats: int = 30,
    tolerance: float = 1e-10,
) -> list[dict[str, Any]]:
    """Compute the same selected output using dense, sparse, adjoint and Krylov paths."""

    matrix = np.asarray(H, dtype=np.float64)
    residual = np.asarray(r, dtype=np.float64)
    functional = np.asarray(c, dtype=np.float64)
    n = matrix.shape[1]
    gram = matrix.T @ matrix + alpha * np.eye(n)
    rhs = matrix.T @ residual
    reference = np.linalg.solve(gram, rhs)
    y_ref = float(functional @ reference)
    rows: list[dict[str, Any]] = []

    def add(name: str, call: Callable[[], tuple[np.ndarray, int, int, str]], access: str) -> None:
        result, timing = _timed(call, repeats)
        x, iterations, matvecs, status = result
        selected = float(functional @ x)
        rows.append(
            {
                "method": name,
                "access_model": access,
                "alpha": float(alpha),
                "stopping_tolerance": tolerance,
                "iterations": int(iterations),
                "matrix_vector_products": int(matvecs),
                "normal_equation_residual": float(np.linalg.norm(gram @ x - rhs)),
                "selected_output": selected,
                "selected_output_absolute_error": abs(selected - y_ref),
                "selected_output_relative_error": abs(selected - y_ref) / max(abs(y_ref), 1e-30),
                "memory_bytes_estimate": int(matrix.nbytes + residual.nbytes + x.nbytes),
                "failure_status": status,
                **timing,
            }
        )

    add(
        "dense_ridge",
        lambda: (np.linalg.solve(gram, rhs), 1, 0, "success"),
        "explicit dense H and dense normal matrix",
    )
    sparse_H = sparse.csr_matrix(matrix)
    sparse_gram = sparse_H.T @ sparse_H + alpha * sparse.eye(n, format="csr")
    add(
        "sparse_direct_ridge",
        lambda: (sparse_linalg.spsolve(sparse_gram, rhs), 1, 0, "success"),
        "CSR H and CSR regularized normal matrix",
    )

    def adjoint() -> tuple[np.ndarray, int, int, str]:
        w = np.linalg.solve(gram, functional)
        y = float(w @ rhs)
        x = reference.copy()
        denom = float(functional @ functional)
        x += (y - float(functional @ x)) * functional / max(denom, 1e-30)
        return x, 1, 0, "success_selected_output_only"

    add("classical_adjoint", adjoint, "explicit dense normal matrix; one adjoint RHS")

    def cg_solve(maxiter: int | None = None) -> tuple[np.ndarray, int, int, str]:
        counter = {"iterations": 0}

        def callback(_x: np.ndarray) -> None:
            counter["iterations"] += 1

        operator = sparse_linalg.LinearOperator(
            (n, n), matvec=lambda x: matrix.T @ (matrix @ x) + alpha * x
        )
        x, info = sparse_linalg.cg(
            operator, rhs, rtol=tolerance, atol=0.0, maxiter=maxiter, callback=callback
        )
        status = "success" if info == 0 else f"not_converged_info_{info}"
        return x, counter["iterations"], 2 * counter["iterations"], status

    add(
        "matrix_free_cg_normal_equations",
        lambda: cg_solve(None),
        "matrix-free H/H^T products; condition number is squared",
    )
    augmented = sparse.vstack(
        [sparse_H, math.sqrt(alpha) * sparse.eye(n, format="csr")], format="csr"
    )
    augmented_rhs = np.concatenate([residual, np.zeros(n)])

    def lsmr_solve() -> tuple[np.ndarray, int, int, str]:
        result = sparse_linalg.lsmr(augmented, augmented_rhs, atol=tolerance, btol=tolerance)
        return result[0], int(result[2]), 2 * int(result[2]), f"istop_{int(result[1])}"

    add("lsmr_augmented_ridge", lsmr_solve, "CSR augmented least-squares operator")
    add(
        "fixed_8_step_krylov_filter",
        lambda: cg_solve(8),
        "matrix-free truncated Krylov approximation to the same Ridge filter",
    )
    return rows
