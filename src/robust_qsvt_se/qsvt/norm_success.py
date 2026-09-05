from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NORM_RECOVERY_LIMITATION = (
    "Unless an amplitude-estimation or norm-estimation routine is explicitly "
    "implemented, norm recovery is simulator metadata, not a fully implemented "
    "quantum readout routine."
)


@dataclass(frozen=True, slots=True)
class NormSuccessDiagnostics:
    ridge_update_norm: float
    qsvt_state_norm_before_normalization: float
    qsvt_state_norm_after_normalization: float
    success_probability_proxy: float
    bounded_filter_scaling_C: float
    beta_matrix_scaling: float
    residual_norm: float
    diagnostic_rescaled_update_error: float
    norm_recovery_method: str
    limitation: str = NORM_RECOVERY_LIMITATION

    def to_row(self) -> dict[str, Any]:
        return {
            "ridge_update_norm": self.ridge_update_norm,
            "qsvt_state_norm_before_normalization": self.qsvt_state_norm_before_normalization,
            "qsvt_state_norm_after_normalization": self.qsvt_state_norm_after_normalization,
            "success_probability_proxy": self.success_probability_proxy,
            "bounded_filter_scaling_C": self.bounded_filter_scaling_C,
            "beta_matrix_scaling": self.beta_matrix_scaling,
            "residual_norm": self.residual_norm,
            "diagnostic_rescaled_update_error": self.diagnostic_rescaled_update_error,
            "norm_recovery_method": self.norm_recovery_method,
            "limitation": self.limitation,
        }


@dataclass(frozen=True, slots=True)
class SuccessSamplingResult:
    true_success_probability: float
    shots: int
    seed: int
    success_counts: int
    estimate: float
    standard_error: float
    absolute_error: float
    limitation: str = NORM_RECOVERY_LIMITATION

    def to_row(self) -> dict[str, Any]:
        return {
            "true_success_probability": self.true_success_probability,
            "shots": self.shots,
            "seed": self.seed,
            "success_counts": self.success_counts,
            "shot_estimated_success_probability": self.estimate,
            "estimated_standard_error": self.standard_error,
            "absolute_error": self.absolute_error,
            "limitation": self.limitation,
        }


@dataclass(frozen=True, slots=True)
class AmplitudeEstimationProxyResult:
    true_success_probability: float
    max_queries: int
    seed: int
    estimate: float
    standard_error_proxy: float
    absolute_error: float
    limitation: str = (
        "Amplitude-estimation-style scaling is a simulator proxy, not a "
        "implemented quantum amplitude-estimation circuit."
    )

    def to_row(self) -> dict[str, Any]:
        return {
            "true_success_probability": self.true_success_probability,
            "max_queries": self.max_queries,
            "seed": self.seed,
            "amplitude_proxy_estimate": self.estimate,
            "standard_error_proxy": self.standard_error_proxy,
            "absolute_error": self.absolute_error,
            "limitation": self.limitation,
        }


def compute_norm_success_diagnostics(
    *,
    ridge_update: np.ndarray,
    qsvt_vector: np.ndarray,
    bounded_scaling_C: float,
    beta: float,
    residual_norm: float,
    success_probability_proxy: float,
    norm_recovery_method: str,
) -> dict[str, Any]:
    ridge = np.asarray(ridge_update, dtype=np.float64)
    qsvt = np.asarray(qsvt_vector, dtype=np.float64)
    if ridge.shape != qsvt.shape:
        raise ValueError("ridge_update and qsvt_vector must have the same shape")
    if bounded_scaling_C <= 0.0:
        raise ValueError("bounded_scaling_C must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if residual_norm < 0.0:
        raise ValueError("residual_norm must be nonnegative")
    p_success = _clip_probability(success_probability_proxy)
    qsvt_norm = float(np.linalg.norm(qsvt))
    diagnostics = NormSuccessDiagnostics(
        ridge_update_norm=float(np.linalg.norm(ridge)),
        qsvt_state_norm_before_normalization=qsvt_norm,
        qsvt_state_norm_after_normalization=0.0 if qsvt_norm == 0.0 else 1.0,
        success_probability_proxy=p_success,
        bounded_filter_scaling_C=float(bounded_scaling_C),
        beta_matrix_scaling=float(beta),
        residual_norm=float(residual_norm),
        diagnostic_rescaled_update_error=float(np.linalg.norm(qsvt - ridge)),
        norm_recovery_method=str(norm_recovery_method),
    )
    return diagnostics.to_row()


def estimate_success_probability_from_shots(
    p_success: float,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    result = estimate_success_probability_by_sampling(p_success, shots, seed)
    row = result.to_row()
    row.update(
        {
            "success_probability": result.true_success_probability,
            "estimate": result.estimate,
            "standard_error": result.standard_error,
            "absolute_sampling_error": result.absolute_error,
        }
    )
    return row


def estimate_success_probability_by_sampling(
    p_success: float,
    shots: int,
    seed: int,
) -> SuccessSamplingResult:
    probability = _clip_probability(p_success)
    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError("shots must be positive")
    rng = np.random.default_rng(int(seed))
    successes = int(rng.binomial(shot_count, probability))
    estimate = successes / float(shot_count)
    standard_error = float(np.sqrt(probability * (1.0 - probability) / shot_count))
    return SuccessSamplingResult(
        true_success_probability=probability,
        shots=shot_count,
        seed=int(seed),
        success_counts=successes,
        estimate=float(estimate),
        standard_error=standard_error,
        absolute_error=abs(float(estimate) - probability),
    )


def estimate_success_probability_iterative_proxy(
    p_success: float,
    max_queries: int,
    seed: int,
) -> AmplitudeEstimationProxyResult:
    probability = _clip_probability(p_success)
    queries = int(max_queries)
    if queries <= 0:
        raise ValueError("max_queries must be positive")
    rng = np.random.default_rng(int(seed))
    sigma = np.sqrt(max(probability * (1.0 - probability), 1.0e-16)) / queries
    estimate = float(np.clip(probability + rng.normal(0.0, sigma), 0.0, 1.0))
    return AmplitudeEstimationProxyResult(
        true_success_probability=probability,
        max_queries=queries,
        seed=int(seed),
        estimate=estimate,
        standard_error_proxy=float(sigma),
        absolute_error=abs(estimate - probability),
    )


def _clip_probability(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("probability must be finite")
    return float(np.clip(float(value), 0.0, 1.0))
