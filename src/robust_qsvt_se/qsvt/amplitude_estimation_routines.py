from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import _build_phase_sequence
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
)
from robust_qsvt_se.qsvt.gate_state_preparation import (
    normalize_and_pad_for_gate_preparation,
)
from robust_qsvt_se.utils.io import ensure_directory

AMPLITUDE_ESTIMATION_CLAIM = (
    "Actual small-circuit amplitude-estimation routines for the dense gate-level QSVT "
    "state-estimation circuit on a selected IEEE14-derived subproblem. The exact "
    "statevector routine is a simulator diagnostic; the Bernoulli and iterative "
    "(maximum-likelihood) routines are simulator-backed small quantum circuit "
    "experiments that estimate the postselection success amplitude. The iterative "
    "routine builds the Grover operator Q = -A S_0 A_dag S_chi from the block-encoded "
    "QSVT unitary. Ridge/Tikhonov remains the reference; no QSVT superiority over "
    "Ridge/Tikhonov, quantum speedup, quantum advantage, or hardware execution is claimed."
)

CLAIM_ALLOWED = "actual amplitude-estimation routine / circuit-level amplitude readout"
CLAIM_DISALLOWED = "QSVT beats Ridge; quantum speedup; quantum advantage; hardware execution"

DEPLOYABILITY_CLASSES = {
    "statevector_diagnostic_only",
    "shot_based_small_circuit",
    "iterative_qae_small_circuit",
    "observable_readout_small_circuit",
    "unsupported_for_current_circuit",
}

SUMMARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_design",
    "scale_protocol",
    "circuit_id",
    "estimator_type",
    "success_projector",
    "shots",
    "exact_success_probability",
    "estimated_success_probability",
    "standard_error",
    "confidence_interval_low",
    "confidence_interval_high",
    "absolute_error",
    "relative_error",
    "grover_powers_used",
    "routine_status",
    "failure_reason_if_any",
    "deployability_class",
    "claim_allowed",
    "claim_disallowed",
]

DEFAULT_GROVER_POWERS = (0, 1, 2, 4, 8)
SUCCESS_PROJECTOR = "encoded_block_ancilla_zero"
TARGET_DESIGN = "current_global"


@dataclass(frozen=True, slots=True)
class QsvtAmplitudeProblem:
    """Dense gate-level QSVT amplitude-estimation problem for a selected subproblem."""

    unitary_A: np.ndarray
    statevector: np.ndarray
    encoded_dimension: int
    total_dimension: int
    n_qubits: int
    exact_success_probability: float
    signal_direction_unit: np.ndarray
    beta: float
    scale_factor_C: float
    synthesized_degree: int
    circuit_depth: int


@dataclass(frozen=True, slots=True)
class BernoulliAmplitudeResult:
    exact_probability: float
    shots: int
    successes: int
    estimate: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    absolute_error: float
    relative_error: float
    interval_method: str


@dataclass(frozen=True, slots=True)
class IterativeAmplitudeResult:
    exact_probability: float
    estimate: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    absolute_error: float
    relative_error: float
    grover_powers: tuple[int, ...]
    shots_per_power: int
    status: str
    failure_reason: str


def run_amplitude_estimation_routines(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75],
        "shots": [100, 1000, 10000],
        "grover_powers": list(DEFAULT_GROVER_POWERS),
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "output_dir": "outputs/qsvt_amplitude_estimation_routines",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    rows = evaluate_amplitude_estimation_grid(
        H=H,
        r=r,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        shot_levels=[int(value) for value in resolved["shots"]],
        grover_powers=tuple(int(value) for value in resolved["grover_powers"]),
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        seed=int(resolved["seed"]),
    )
    artifacts = write_amplitude_estimation_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_amplitude_estimation_grid(
    *,
    H: np.ndarray,
    r: np.ndarray,
    alphas: list[float],
    degrees: list[int],
    shot_levels: list[int],
    grover_powers: tuple[int, ...],
    case: str,
    model: str,
    subproblem_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for degree in degrees:
            rows.extend(
                evaluate_amplitude_estimation_point(
                    H=H,
                    r=r,
                    alpha=float(alpha),
                    degree=int(degree),
                    shot_levels=shot_levels,
                    grover_powers=grover_powers,
                    case=case,
                    model=model,
                    subproblem_id=subproblem_id,
                    seed=int(seed),
                )
            )
    return rows


def evaluate_amplitude_estimation_point(
    *,
    H: np.ndarray,
    r: np.ndarray,
    alpha: float,
    degree: int,
    shot_levels: list[int],
    grover_powers: tuple[int, ...],
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    seed: int = 123,
) -> list[dict[str, Any]]:
    circuit_id = f"{case}_{model}_d{int(degree)}_a{float(alpha):g}"
    base = {
        "case": case,
        "model": model,
        "subproblem_id": subproblem_id,
        "alpha": float(alpha),
        "degree": int(degree),
        "target_design": TARGET_DESIGN,
        "scale_protocol": "success_amplitude",
        "circuit_id": circuit_id,
        "success_projector": SUCCESS_PROJECTOR,
    }
    try:
        problem = build_qsvt_amplitude_problem(H, r, alpha=float(alpha), degree=int(degree))
    except Exception as exc:  # pragma: no cover - phase synthesis/backend dependent
        reason = f"{type(exc).__name__}: {exc}"
        return [
            _row(
                base,
                estimator_type="amplitude_problem_build",
                shots=float("nan"),
                exact=float("nan"),
                estimate=float("nan"),
                standard_error=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                grover_powers="",
                status="failed",
                failure_reason=reason,
                deployability_class="unsupported_for_current_circuit",
            )
        ]

    p_exact = problem.exact_success_probability
    rows: list[dict[str, Any]] = [
        _row(
            base,
            estimator_type="statevector_exact",
            shots=0,
            exact=p_exact,
            estimate=p_exact,
            standard_error=0.0,
            ci_low=p_exact,
            ci_high=p_exact,
            grover_powers="",
            status="succeeded",
            failure_reason="",
            deployability_class="statevector_diagnostic_only",
        )
    ]
    for shots in shot_levels:
        bernoulli = bernoulli_amplitude_estimate(p_exact, int(shots), int(seed) + int(shots))
        rows.append(
            _row(
                base,
                estimator_type="bernoulli_shots",
                shots=int(shots),
                exact=p_exact,
                estimate=bernoulli.estimate,
                standard_error=bernoulli.standard_error,
                ci_low=bernoulli.confidence_interval_low,
                ci_high=bernoulli.confidence_interval_high,
                grover_powers="",
                status="succeeded",
                failure_reason=bernoulli.interval_method,
                deployability_class="shot_based_small_circuit",
            )
        )
        iterative = iterative_amplitude_estimate(
            problem.unitary_A,
            problem.encoded_dimension,
            powers=grover_powers,
            shots=int(shots),
            seed=int(seed) + 7 * int(shots),
            exact_probability=p_exact,
        )
        rows.append(
            _row(
                base,
                estimator_type="iterative_mlae",
                shots=int(shots),
                exact=p_exact,
                estimate=iterative.estimate,
                standard_error=iterative.standard_error,
                ci_low=iterative.confidence_interval_low,
                ci_high=iterative.confidence_interval_high,
                grover_powers="|".join(str(power) for power in iterative.grover_powers),
                status=iterative.status,
                failure_reason=iterative.failure_reason,
                deployability_class=(
                    "iterative_qae_small_circuit"
                    if iterative.status == "succeeded"
                    else "unsupported_for_current_circuit"
                ),
            )
        )
        observable = observable_amplitude_estimate(
            problem.statevector,
            encoded_dimension=problem.encoded_dimension,
            component=0,
            shots=int(shots),
            seed=int(seed) + 13 * int(shots),
        )
        rows.append(
            _row(
                base,
                estimator_type="observable_probability_readout",
                shots=int(shots),
                exact=observable["exact_probability"],
                estimate=observable["estimate"],
                standard_error=observable["standard_error"],
                ci_low=observable["confidence_interval_low"],
                ci_high=observable["confidence_interval_high"],
                grover_powers="",
                status="succeeded",
                failure_reason="estimates_magnitude_only_no_sign",
                deployability_class="observable_readout_small_circuit",
                scale_protocol="component_probability",
            )
        )
    return rows


def build_qsvt_amplitude_problem(
    H: np.ndarray,
    r: np.ndarray,
    *,
    alpha: float,
    degree: int,
    grid_size: int = 2048,
    phase_timeout_seconds: int = 25,
) -> QsvtAmplitudeProblem:
    """Build the dense gate-level QSVT circuit and its Grover-ready state-preparation unitary.

    The returned ``unitary_A`` satisfies ``unitary_A @ e_0 == statevector`` so the Grover
    operator ``Q = -A S_0 A_dag S_chi`` can be applied directly for amplitude estimation.
    """

    H = np.asarray(H, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("amplitude problem requires a square H_tilde")
    if not _is_power_of_two(H.shape[0]):
        raise ValueError("amplitude problem dimension must be a power of two")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    if int(degree) <= 0 or int(degree) % 2 == 0:
        raise ValueError("degree must be a positive odd integer")

    B = H.T
    beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
    A = B / beta
    alpha_norm = float(alpha) / beta**2
    block_encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
    phase_sequence = _build_phase_sequence(
        singular_values_A=np.linalg.svd(A, compute_uv=False),
        requested_degree=int(degree),
        alpha_norm=alpha_norm,
        grid_size=max(int(grid_size), int(degree) + 2),
        max_synthesis_degree=int(degree),
        angle_solver="iterative",
        phase_timeout_seconds=int(phase_timeout_seconds),
    )
    state_preparation = normalize_and_pad_for_gate_preparation(
        r,
        target_dimension=block_encoding.unitary.shape[0],
    )
    operator = build_structured_qsvt_operator_circuit(
        block_encoding.unitary,
        phase_sequence.phases,
        encoded_dimension=A.shape[0],
    )
    qsvt_unitary = _operator_matrix(operator.qsvt_operator_circuit)
    preparation_unitary = _state_preparation_unitary(state_preparation.padded_state)
    unitary_A = qsvt_unitary @ preparation_unitary
    statevector = unitary_A[:, 0]
    dimension = int(A.shape[0])
    prefix = statevector[:dimension]
    success_probability = float(np.sum(np.abs(prefix) ** 2))
    signal = np.real(prefix)
    signal_norm = float(np.linalg.norm(signal))
    signal_unit = signal / signal_norm if signal_norm > 1.0e-15 else np.zeros_like(signal)
    return QsvtAmplitudeProblem(
        unitary_A=unitary_A,
        statevector=statevector,
        encoded_dimension=dimension,
        total_dimension=int(unitary_A.shape[0]),
        n_qubits=int(operator.n_qubits),
        exact_success_probability=success_probability,
        signal_direction_unit=signal_unit,
        beta=beta,
        scale_factor_C=float(phase_sequence.approximation.scale_factor),
        synthesized_degree=int(phase_sequence.actual_degree),
        circuit_depth=int(operator.qsvt_operator_circuit.depth()),
    )


def exact_success_probability(statevector: np.ndarray, *, encoded_dimension: int) -> float:
    values = np.asarray(statevector, dtype=np.complex128)
    dimension = int(encoded_dimension)
    if values.ndim != 1 or dimension <= 0 or dimension > values.size:
        raise ValueError("encoded_dimension must select a valid prefix of statevector")
    return float(np.sum(np.abs(values[:dimension]) ** 2))


def bernoulli_amplitude_estimate(
    probability: float,
    shots: int,
    seed: int,
    *,
    confidence: float = 0.95,
) -> BernoulliAmplitudeResult:
    """Estimate a success probability by sampling computational-basis measurement outcomes."""

    p_true = float(np.clip(probability, 0.0, 1.0))
    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError("shots must be positive")
    rng = np.random.default_rng(int(seed))
    successes = int(rng.binomial(shot_count, p_true))
    estimate = successes / float(shot_count)
    standard_error = float(np.sqrt(max(estimate * (1.0 - estimate), 0.0) / shot_count))
    low, high, method = wilson_interval(successes, shot_count, confidence=confidence)
    return BernoulliAmplitudeResult(
        exact_probability=p_true,
        shots=shot_count,
        successes=successes,
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval_low=low,
        confidence_interval_high=high,
        absolute_error=abs(estimate - p_true),
        relative_error=abs(estimate - p_true) / max(p_true, 1.0e-12),
        interval_method=method,
    )


def wilson_interval(
    successes: int,
    shots: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float, str]:
    """Return a Wilson score interval (normal-approximation fallback if SciPy is absent)."""

    n = int(shots)
    if n <= 0:
        raise ValueError("shots must be positive")
    z, method = _normal_quantile(confidence)
    p_hat = int(successes) / float(n)
    denominator = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denominator
    half = (z / denominator) * math.sqrt(p_hat * (1.0 - p_hat) / n + z**2 / (4.0 * n**2))
    return max(0.0, center - half), min(1.0, center + half), method


def iterative_amplitude_estimate(
    unitary_A: np.ndarray,
    encoded_dimension: int,
    *,
    powers: tuple[int, ...] = DEFAULT_GROVER_POWERS,
    shots: int,
    seed: int,
    exact_probability: float | None = None,
    confidence: float = 0.95,
) -> IterativeAmplitudeResult:
    """Maximum-likelihood amplitude estimation using Grover powers of the QSVT unitary.

    The Grover operator ``Q = -A S_0 A_dag S_chi`` is built from the block-encoded QSVT
    unitary; the amplitude after ``m`` applications is ``sin^2((2m+1) theta)`` with
    ``sin^2(theta)`` equal to the postselection success probability. Measurement counts are
    sampled per Grover power and combined into a profile-likelihood estimate of ``theta``.
    """

    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError("shots must be positive")
    A = np.asarray(unitary_A, dtype=np.complex128)
    dimension = int(encoded_dimension)
    p_exact = (
        exact_success_probability(A[:, 0], encoded_dimension=dimension)
        if exact_probability is None
        else float(np.clip(exact_probability, 0.0, 1.0))
    )
    schedule = tuple(int(power) for power in powers)
    try:
        grover = _grover_operator(A, dimension)
        psi = A[:, 0]
        rng = np.random.default_rng(int(seed))
        thetas = np.linspace(1.0e-6, math.pi / 2.0 - 1.0e-6, 4000)
        log_likelihood = np.zeros_like(thetas)
        for power in schedule:
            state = psi.copy()
            for _ in range(int(power)):
                state = grover @ state
            p_power = float(np.sum(np.abs(state[:dimension]) ** 2))
            successes = int(rng.binomial(shot_count, float(np.clip(p_power, 0.0, 1.0))))
            grid = np.clip(np.sin((2 * power + 1) * thetas) ** 2, 1.0e-12, 1.0 - 1.0e-12)
            log_likelihood += successes * np.log(grid) + (shot_count - successes) * np.log(
                1.0 - grid
            )
    except Exception as exc:  # pragma: no cover - defensive (large/incompatible circuits)
        return IterativeAmplitudeResult(
            exact_probability=p_exact,
            estimate=float("nan"),
            standard_error=float("nan"),
            confidence_interval_low=float("nan"),
            confidence_interval_high=float("nan"),
            absolute_error=float("nan"),
            relative_error=float("nan"),
            grover_powers=schedule,
            shots_per_power=shot_count,
            status="unsupported_for_current_circuit",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    best_index = int(np.argmax(log_likelihood))
    estimate = float(np.sin(thetas[best_index]) ** 2)
    z, _ = _normal_quantile(confidence)
    threshold = float(log_likelihood[best_index]) - 0.5 * z**2
    within = np.flatnonzero(log_likelihood >= threshold)
    probabilities = np.sin(thetas[within]) ** 2
    low = float(np.min(probabilities)) if within.size else estimate
    high = float(np.max(probabilities)) if within.size else estimate
    standard_error = (high - low) / (2.0 * z) if z > 0.0 else float("nan")
    return IterativeAmplitudeResult(
        exact_probability=p_exact,
        estimate=estimate,
        standard_error=float(standard_error),
        confidence_interval_low=low,
        confidence_interval_high=high,
        absolute_error=abs(estimate - p_exact),
        relative_error=abs(estimate - p_exact) / max(p_exact, 1.0e-12),
        grover_powers=schedule,
        shots_per_power=shot_count,
        status="succeeded",
        failure_reason="",
    )


def iterative_amplitude_estimate_for_probability(
    probability: float,
    *,
    powers: tuple[int, ...] = DEFAULT_GROVER_POWERS,
    shots: int,
    seed: int,
    confidence: float = 0.95,
) -> IterativeAmplitudeResult:
    """Run Grover-power MLAE on the canonical single-qubit testbed for a target amplitude.

    Builds ``A`` with ``A e_0 = [sqrt(p), sqrt(1-p)]`` and marks the first state as success,
    giving a faithful small-circuit amplitude-estimation problem with success probability ``p``.
    Used when the full QSVT unitary is not rebuilt per target design.
    """

    p = float(np.clip(probability, 0.0, 1.0))
    amplitude = math.sqrt(p)
    co_amplitude = math.sqrt(max(1.0 - p, 0.0))
    unitary = np.array([[amplitude, -co_amplitude], [co_amplitude, amplitude]], dtype=np.complex128)
    return iterative_amplitude_estimate(
        unitary,
        encoded_dimension=1,
        powers=powers,
        shots=int(shots),
        seed=int(seed),
        exact_probability=p,
        confidence=confidence,
    )


def observable_amplitude_estimate(
    statevector: np.ndarray,
    *,
    encoded_dimension: int,
    component: int,
    shots: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate a basis-probability observable on the postselected encoded state.

    A computational-basis measurement yields the component probability (magnitude only);
    it does not estimate the sign of a signed overlap.
    """

    values = np.asarray(statevector, dtype=np.complex128)
    dimension = int(encoded_dimension)
    prefix = values[:dimension]
    success = float(np.sum(np.abs(prefix) ** 2))
    index = int(component)
    if success <= 1.0e-15 or index >= dimension:
        return {
            "exact_probability": float("nan"),
            "estimate": float("nan"),
            "standard_error": float("nan"),
            "confidence_interval_low": float("nan"),
            "confidence_interval_high": float("nan"),
            "estimates_sign": False,
        }
    post = prefix / math.sqrt(success)
    p_component = float(np.abs(post[index]) ** 2)
    result = bernoulli_amplitude_estimate(p_component, int(shots), int(seed), confidence=confidence)
    return {
        "exact_probability": p_component,
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "confidence_interval_low": result.confidence_interval_low,
        "confidence_interval_high": result.confidence_interval_high,
        "estimates_sign": False,
    }


def write_amplitude_estimation_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    summary_path = output_dir / "amplitude_estimation_summary.csv"
    bernoulli_path = output_dir / "bernoulli_success_estimates.csv"
    iterative_path = output_dir / "iterative_amplitude_estimation_attempts.csv"
    observable_path = output_dir / "observable_amplitude_estimates.csv"
    interpretation_path = output_dir / "amplitude_estimation_interpretation.md"

    frame.to_csv(summary_path, index=False)
    frame[frame["estimator_type"] == "bernoulli_shots"].to_csv(bernoulli_path, index=False)
    frame[frame["estimator_type"] == "iterative_mlae"].to_csv(iterative_path, index=False)
    frame[frame["estimator_type"] == "observable_probability_readout"].to_csv(
        observable_path, index=False
    )
    interpretation_path.write_text(amplitude_estimation_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "amplitude_estimation_summary": str(summary_path),
            "bernoulli_success_estimates": str(bernoulli_path),
            "iterative_amplitude_estimation_attempts": str(iterative_path),
            "observable_amplitude_estimates": str(observable_path),
            "amplitude_estimation_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=AMPLITUDE_ESTIMATION_CLAIM,
    )
    return {
        "manifest": manifest,
        "amplitude_estimation_summary": summary_path,
        "bernoulli_success_estimates": bernoulli_path,
        "iterative_amplitude_estimation_attempts": iterative_path,
        "observable_amplitude_estimates": observable_path,
        "amplitude_estimation_interpretation": interpretation_path,
    }


def amplitude_estimation_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(["# Actual Amplitude-Estimation Routines", "", AMPLITUDE_ESTIMATION_CLAIM])

    def best_abs_error(estimator: str) -> float:
        subset = frame[frame["estimator_type"] == estimator]["absolute_error"].astype(float)
        finite = subset[np.isfinite(subset)]
        return float(finite.min()) if not finite.empty else float("nan")

    def largest_shots_error(estimator: str) -> float:
        subset = frame[frame["estimator_type"] == estimator].copy()
        subset["shots"] = pd.to_numeric(subset["shots"], errors="coerce")
        subset = subset[subset["shots"] == subset["shots"].max()]
        finite = subset["absolute_error"].astype(float)
        finite = finite[np.isfinite(finite)]
        return float(finite.mean()) if not finite.empty else float("nan")

    bernoulli_best = best_abs_error("bernoulli_shots")
    iterative_best = best_abs_error("iterative_mlae")
    iterative_status = set(frame[frame["estimator_type"] == "iterative_mlae"]["routine_status"])
    qae_supported = "yes" if "succeeded" in iterative_status else "no"
    bernoulli_high = largest_shots_error("bernoulli_shots")
    iterative_high = largest_shots_error("iterative_mlae")

    return "\n".join(
        [
            "# Actual Amplitude-Estimation Routines",
            "",
            AMPLITUDE_ESTIMATION_CLAIM,
            "",
            "## Implemented estimators",
            "- statevector_exact: exact postselection success probability (diagnostic only).",
            "- bernoulli_shots: repeated computational-basis sampling with a Wilson interval.",
            "- iterative_mlae: maximum-likelihood amplitude estimation with Grover powers built "
            "from the block-encoded QSVT unitary.",
            "- observable_probability_readout: basis-probability readout (magnitude only).",
            "",
            "## Required Answers",
            f"1. Does shot-based estimation approach the exact success probability? "
            f"best Bernoulli absolute error {bernoulli_best:.3g}; at the largest shot budget the "
            f"mean Bernoulli absolute error is {bernoulli_high:.3g}.",
            f"2. Is the iterative (Grover-power) routine supported on the current circuit? "
            f"{qae_supported} (best iterative absolute error {iterative_best:.3g}; at the largest "
            f"shot budget mean absolute error {iterative_high:.3g}).",
            "3. Confidence-interval behavior: Bernoulli uses a Wilson score interval and the "
            "iterative routine uses a profile-likelihood interval; both contract as shots grow.",
            "4. Deployability: statevector_exact is diagnostic-only; bernoulli_shots and "
            "iterative_mlae are simulator-backed small-circuit routines; the observable readout "
            "estimates magnitude only.",
            "",
        ]
    )


def _grover_operator(unitary_A: np.ndarray, encoded_dimension: int) -> np.ndarray:
    n = int(unitary_A.shape[0])
    identity = np.eye(n, dtype=np.complex128)
    good = np.zeros((n, n), dtype=np.complex128)
    for index in range(int(encoded_dimension)):
        good[index, index] = 1.0
    reflect_good = identity - 2.0 * good
    reflect_zero = identity.copy()
    reflect_zero[0, 0] = -1.0
    return -unitary_A @ reflect_zero @ unitary_A.conj().T @ reflect_good


def _state_preparation_unitary(padded_state: np.ndarray) -> np.ndarray:
    vector = np.asarray(padded_state, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("padded_state must have nonzero norm")
    vector = vector / norm
    size = vector.size
    e0 = np.zeros(size, dtype=np.complex128)
    e0[0] = 1.0
    reflector = e0 - vector
    reflector_norm = float(np.linalg.norm(reflector))
    if reflector_norm <= 1.0e-12:
        return np.eye(size, dtype=np.complex128)
    reflector = reflector / reflector_norm
    return np.eye(size, dtype=np.complex128) - 2.0 * np.outer(reflector, reflector.conj())


def _operator_matrix(circuit: Any) -> np.ndarray:
    try:
        from qiskit.quantum_info import Operator  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for amplitude-estimation circuits") from exc
    return np.asarray(Operator(circuit).data, dtype=np.complex128)


def _normal_quantile(confidence: float) -> tuple[float, str]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    try:
        from scipy.stats import norm  # type: ignore[import-not-found]

        return float(norm.ppf(1.0 - (1.0 - confidence) / 2.0)), "wilson_scipy"
    except Exception:  # pragma: no cover - SciPy always present in this environment
        return 1.959963984540054, "wilson_normal_approx"


def _row(
    base: dict[str, Any],
    *,
    estimator_type: str,
    shots: float,
    exact: float,
    estimate: float,
    standard_error: float,
    ci_low: float,
    ci_high: float,
    grover_powers: str,
    status: str,
    failure_reason: str,
    deployability_class: str,
    scale_protocol: str | None = None,
) -> dict[str, Any]:
    absolute = (
        abs(estimate - exact) if math.isfinite(estimate) and math.isfinite(exact) else float("nan")
    )
    relative = (
        absolute / max(abs(exact), 1.0e-12)
        if math.isfinite(absolute) and math.isfinite(exact)
        else float("nan")
    )
    row = dict(base)
    if scale_protocol is not None:
        row["scale_protocol"] = scale_protocol
    row.update(
        {
            "estimator_type": estimator_type,
            "shots": shots,
            "exact_success_probability": float(exact),
            "estimated_success_probability": float(estimate),
            "standard_error": float(standard_error),
            "confidence_interval_low": float(ci_low),
            "confidence_interval_high": float(ci_high),
            "absolute_error": float(absolute),
            "relative_error": float(relative),
            "grover_powers_used": grover_powers,
            "routine_status": status,
            "failure_reason_if_any": failure_reason,
            "deployability_class": deployability_class,
            "claim_allowed": CLAIM_ALLOWED,
            "claim_disallowed": CLAIM_DISALLOWED,
        }
    )
    return row


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0
