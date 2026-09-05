"""Ridge selected-output derivatives and conservative perturbation certificates.

The first-order derivative and the operator-norm certificate serve deliberately distinct
roles.  The derivative is a local sensitivity model.  The certificate is a global,
conservative bound obtained from resolvent identities and does not use an observed output
error in its computation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class RidgeSelectedOutputCertificate:
    """Conservative selected-output perturbation bound and optional validation fields."""

    matrix_delta_spectral: float
    operator_bound_forward: float
    operator_bound_reverse: float
    selected_output_bound: float
    actual_selected_output_error: float | None
    certificate_holds: bool | None
    tightness_ratio: float | None


def _validated_inputs(
    matrix: np.ndarray,
    residual: np.ndarray,
    functional: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    raw_matrix = np.asarray(matrix)
    raw_residual = np.asarray(residual)
    raw_functional = np.asarray(functional)
    if np.iscomplexobj(raw_matrix) or np.iscomplexobj(raw_residual) or np.iscomplexobj(
        raw_functional
    ):
        raise ValueError("this Ridge selected-output study supports real arrays only")
    values = np.asarray(raw_matrix, dtype=np.float64)
    rhs = np.asarray(raw_residual, dtype=np.float64)
    ell = np.asarray(raw_functional, dtype=np.float64)
    regularization = float(alpha)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if rhs.ndim != 1 or rhs.shape != (values.shape[0],):
        raise ValueError(f"residual must have shape {(values.shape[0],)}")
    if ell.ndim != 1 or ell.shape != (values.shape[1],):
        raise ValueError(f"functional must have shape {(values.shape[1],)}")
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(rhs))
        and np.all(np.isfinite(ell))
    ):
        raise ValueError("matrix, residual, and functional must be finite")
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("alpha must be positive and finite")
    return values, rhs, ell, regularization


def ridge_selected_output_gradient(
    matrix: np.ndarray,
    residual: np.ndarray,
    functional: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return the exact first derivative of a real Ridge selected output.

    With ``M=H.T@H+alpha*I``, ``x=solve(M,H.T@r)``, ``z=solve(M,ell)``, and
    ``e=r-H@x``, the gradient is ``e z.T - (H z) x.T``.  No explicit inverse is
    formed.
    """

    values, rhs, ell, regularization = _validated_inputs(
        matrix, residual, functional, alpha
    )
    gram = values.T @ values
    system = gram + regularization * np.eye(values.shape[1], dtype=np.float64)
    try:
        solved = np.linalg.solve(system, np.column_stack((values.T @ rhs, ell)))
    except np.linalg.LinAlgError as exc:  # pragma: no cover - alpha>0 is SPD in exact math
        raise RuntimeError("regularized normal-equation solve failed") from exc
    x = solved[:, 0]
    z = solved[:, 1]
    error = rhs - values @ x
    gradient = np.outer(error, z) - np.outer(values @ z, x)
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("Ridge selected-output gradient is non-finite")
    return np.asarray(gradient, dtype=np.float64)


def ridge_selected_output_via_solve(
    matrix: np.ndarray,
    residual: np.ndarray,
    functional: np.ndarray,
    alpha: float,
) -> float:
    """Evaluate the study's Ridge selected output with a stable linear solve."""

    values, rhs, ell, regularization = _validated_inputs(
        matrix, residual, functional, alpha
    )
    system = values.T @ values + regularization * np.eye(
        values.shape[1], dtype=np.float64
    )
    try:
        solution = np.linalg.solve(system, values.T @ rhs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - alpha>0 is SPD in exact math
        raise RuntimeError("regularized normal-equation solve failed") from exc
    return float(ell @ solution)


def ridge_linearized_signed_error(
    matrix: np.ndarray,
    perturbed_matrix: np.ndarray,
    residual: np.ndarray,
    functional: np.ndarray,
    alpha: float,
) -> float:
    """Predict ``y(perturbed_matrix)-y(matrix)`` using the local derivative."""

    values, rhs, ell, regularization = _validated_inputs(
        matrix, residual, functional, alpha
    )
    perturbed = np.asarray(perturbed_matrix)
    if np.iscomplexobj(perturbed):
        raise ValueError("this Ridge selected-output study supports real arrays only")
    candidate = np.asarray(perturbed, dtype=np.float64)
    if candidate.shape != values.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("perturbed_matrix must be finite and match matrix shape")
    gradient = ridge_selected_output_gradient(values, rhs, ell, regularization)
    return float(np.sum(gradient * (candidate - values)))


def compute_ridge_selected_output_certificate(
    matrix: np.ndarray,
    perturbed_matrix: np.ndarray,
    residual: np.ndarray,
    functional: np.ndarray,
    alpha: float,
) -> RidgeSelectedOutputCertificate:
    """Compute the conservative certificate without consulting an actual output error."""

    values, rhs, ell, regularization = _validated_inputs(
        matrix, residual, functional, alpha
    )
    raw_perturbed = np.asarray(perturbed_matrix)
    if np.iscomplexobj(raw_perturbed):
        raise ValueError("this Ridge selected-output study supports real arrays only")
    candidate = np.asarray(raw_perturbed, dtype=np.float64)
    if candidate.shape != values.shape:
        raise ValueError("perturbed_matrix must match matrix shape")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("perturbed_matrix must be finite")

    delta = float(np.linalg.norm(values - candidate, ord=2))
    norm_a = float(np.linalg.norm(values, ord=2))
    norm_b = float(np.linalg.norm(candidate, ord=2))
    common = norm_a + norm_b
    forward = delta * (
        1.0 / regularization + common * norm_b / regularization**2
    )
    reverse = delta * (
        1.0 / regularization + common * norm_a / regularization**2
    )
    operator_bound = min(forward, reverse)
    selected_bound = float(np.linalg.norm(ell) * np.linalg.norm(rhs) * operator_bound)
    if not all(np.isfinite(value) and value >= 0.0 for value in (
        delta, forward, reverse, selected_bound
    )):
        raise FloatingPointError("Ridge perturbation certificate is non-finite")
    return RidgeSelectedOutputCertificate(
        matrix_delta_spectral=delta,
        operator_bound_forward=float(forward),
        operator_bound_reverse=float(reverse),
        selected_output_bound=selected_bound,
        actual_selected_output_error=None,
        certificate_holds=None,
        tightness_ratio=None,
    )


def validate_ridge_selected_output_certificate(
    certificate: RidgeSelectedOutputCertificate,
    actual_selected_output_error: float,
    *,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-10,
) -> RidgeSelectedOutputCertificate:
    """Attach post-computation validation fields to an independently computed bound."""

    actual = float(actual_selected_output_error)
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    if not np.isfinite(actual) or actual < 0.0:
        raise ValueError("actual_selected_output_error must be finite and nonnegative")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("certificate tolerances must be finite and nonnegative")
    allowed = certificate.selected_output_bound + atol + rtol * max(
        certificate.selected_output_bound, actual
    )
    holds = bool(actual <= allowed)
    tightness = (
        float(certificate.selected_output_bound / actual) if actual > 0.0 else None
    )
    return replace(
        certificate,
        actual_selected_output_error=actual,
        certificate_holds=holds,
        tightness_ratio=tightness,
    )


def ridge_selected_output_certificate(
    matrix: np.ndarray,
    perturbed_matrix: np.ndarray,
    residual: np.ndarray,
    functional: np.ndarray,
    alpha: float,
    *,
    actual_selected_output_error: float | None = None,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-10,
) -> RidgeSelectedOutputCertificate:
    """Convenience wrapper that always computes the bound before optional validation."""

    certificate = compute_ridge_selected_output_certificate(
        matrix, perturbed_matrix, residual, functional, alpha
    )
    if actual_selected_output_error is None:
        return certificate
    return validate_ridge_selected_output_certificate(
        certificate,
        actual_selected_output_error,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
