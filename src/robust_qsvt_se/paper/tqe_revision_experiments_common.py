"""Shared scaffolding for the TQE-revision *experiment* package.

These five experiments (readout statistics, conditioning boundary, end-to-end
resource ledger, sparse-access oracle demo, revision readiness) produce **new
auditable evidence** for the revised, feasibility-boundary framing of the paper.
They never change any estimator, never retune anything to flatter QSVT, never
overwrite an existing output, and never fabricate a phase-synthesis result: a
failed synthesis is recorded as a failure with its exact status.

Everything lives under ``outputs/tqe_revision_experiments/``. The classical
Ridge/Tikhonov filter stays the only reference; the QSVT-target filter is the
*same* regularized spectral filter evaluated at the *same* ``alpha`` and is
numerically equivalent to Ridge in the classical simulator. No speedup, no
quantum advantage, no full-vector readout, no IEEE-scale hardware execution, and
no field-data validation is claimed anywhere.

Normalization convention (Option B, project memory): the QSVT polynomial
approximates the *normalized* filter ``f(s) = s/(s^2 + alpha/beta^2)`` on
``s = sigma/beta in [0, 1]`` with ``beta = sigma_max``; the single physical
rescale factor back to the matched Ridge update is ``C/beta`` (never ``C*beta``).
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    checksum,
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.paper.tqe_revision_support_common import (
    find_forbidden,
    git_commit_hash,
    now_iso,
    package_versions,
)
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

# ---------------------------------------------------------------------------
# Output roots. Existing outputs elsewhere are never touched.
# ---------------------------------------------------------------------------
EXPERIMENTS_ROOT = Path("outputs/tqe_revision_experiments")
READOUT_DIR = EXPERIMENTS_ROOT / "readout_statistics"
BOUNDARY_DIR = EXPERIMENTS_ROOT / "conditioning_boundary"
RESOURCE_DIR = EXPERIMENTS_ROOT / "end_to_end_resource_case"
SPARSE_DIR = EXPERIMENTS_ROOT / "sparse_block_encoding_demo"
READINESS_DIR = EXPERIMENTS_ROOT / "revision_readiness"

PHASE_CACHE_DIR = EXPERIMENTS_ROOT / "_phase_cache"

# Conservative framing reused verbatim in every generated report header.
EXPERIMENTS_CLAIM_BOUNDARY = (
    "Controlled IEEE/PYPOWER benchmark and QSVT feasibility-boundary study for "
    "regularized spectral filtering in ill-conditioned power-system state estimation. "
    "IEEE/PYPOWER cases provide benchmark network models; measurement rows are generated "
    "from the network equations by code, not from field PMU/SCADA data. The QSVT-target "
    "filter is a QSVT-compatible implementation pathway for the same Ridge/Tikhonov "
    "regularized spectral filter and is numerically equivalent to Ridge at the same alpha "
    "in the classical simulator. This study quantifies a feasibility boundary. It does not "
    "demonstrate a quantum speed-up, does not demonstrate a quantum-computational "
    "advantage, does not show the QSVT-target filter beating Ridge/Tikhonov numerically, "
    "does not implement full-vector quantum readout, does not perform full IEEE-scale "
    "execution on quantum devices, and is not validated against real PMU/SCADA field "
    "measurements."
)

# Additional exact overclaim strings banned on top of the repository-wide list.
_EXPERIMENT_BANNED_OVERCLAIMS = (
    "quantum speedup demonstrated",
    "qsvt outperforms ridge",
    "qsvt beats ridge",
    "qsvt beats tikhonov",
    "quantum advantage",
    "field pmu/scada validation",
    "pmu/scada validation",
    "full ieee-scale hardware execution",
    "full-vector readout solved",
    "full vector readout solved",
    "readout bottleneck solved",
)


def forbidden_in(text: str) -> list[str]:
    """Return forbidden phrases present in ``text`` (repository + experiment sets)."""

    lowered = text.lower()
    repo_hits = find_forbidden(text)
    exp_hits = [phrase for phrase in _EXPERIMENT_BANNED_OVERCLAIMS if phrase in lowered]
    return sorted(set(repo_hits) | set(exp_hits))


def assert_safe(text: str) -> None:
    """Raise if ``text`` contains any repository-wide or experiment-banned overclaim."""

    violations = forbidden_in(text)
    if violations:
        raise RuntimeError(f"generated text contains forbidden wording: {violations}")


# ---------------------------------------------------------------------------
# Matplotlib (Agg, no seaborn) helper.
# ---------------------------------------------------------------------------
def get_pyplot() -> Any:
    """Return a headless ``matplotlib.pyplot`` (Agg backend, never seaborn)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# Provenance manifest writer (records seeds, warnings, failures, boundary).
# ---------------------------------------------------------------------------
def write_experiment_manifest(
    *,
    output_dir: str | Path,
    experiment_id: str,
    script_name: str,
    command: str,
    description: str,
    artifacts: dict[str, Path],
    inputs_used: list[str],
    random_seeds: dict[str, Any],
    warnings: list[str],
    failures: list[dict[str, Any]],
    interpretation_boundary: str,
    extra: dict[str, Any] | None = None,
    manifest_name: str = "manifest.json",
) -> Path:
    """Write a provenance manifest with the full field set required by the task."""

    directory = ensure_directory(output_dir)
    path = directory / manifest_name
    generated = {name: str(value) for name, value in sorted(artifacts.items())}
    generated["manifest"] = str(path)
    checksums = {name: checksum(value) for name, value in sorted(artifacts.items())}
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "script_name": script_name,
        "command": command,
        "description": description,
        "timestamp": now_iso(),
        "git_commit_hash": git_commit_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "key_package_versions": package_versions(
            ["numpy", "scipy", "pandas", "matplotlib", "pennylane", "qiskit", "qiskit-aer"]
        ),
        "random_seeds": random_seeds,
        "inputs_used": sorted(inputs_used),
        "outputs_generated": generated,
        "checksums": checksums,
        "warnings": list(warnings),
        "failures": list(failures),
        "interpretation_boundary": interpretation_boundary,
        "claim_boundary": EXPERIMENTS_CLAIM_BOUNDARY,
        "changes_estimator_behavior": False,
        "fabricates_results": False,
        "overwrites_existing_outputs": False,
    }
    if extra:
        manifest.update(extra)
    write_json(path, manifest)
    return path


# ---------------------------------------------------------------------------
# Controlled-SVD matrix family (Family 1).
# ---------------------------------------------------------------------------
def controlled_singular_values(size: int, kappa: float) -> np.ndarray:
    """Log-spaced singular values from ``sigma_max = 1`` down to ``sigma_min = 1/kappa``."""

    if size < 1:
        raise ValueError("size must be positive")
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1")
    if size == 1:
        return np.array([1.0], dtype=np.float64)
    return np.logspace(0.0, -np.log10(float(kappa)), num=int(size), dtype=np.float64)


def _random_orthogonal(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    gaussian = rng.standard_normal((size, size))
    q, r = np.linalg.qr(gaussian)
    # Fix the sign ambiguity so the map is a deterministic function of the seed.
    q *= np.sign(np.diag(r))
    return q


def controlled_svd_matrix(size: int, kappa: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(H, singular_values)`` for a controlled-conditioning matrix.

    ``H = U diag(sv) V^T`` with random orthogonal ``U, V`` (seeded) and prescribed
    log-spaced singular values. ``sigma_max = 1`` and ``sigma_min = 1/kappa``.
    """

    singular_values = controlled_singular_values(size, kappa)
    left = _random_orthogonal(size, 2 * int(seed) + 1)
    right = _random_orthogonal(size, 2 * int(seed) + 2)
    matrix = left @ np.diag(singular_values) @ right.T
    return matrix, singular_values


# ---------------------------------------------------------------------------
# Bounded-target + phase-synthesis attempt (the Experiment B / C core).
# ---------------------------------------------------------------------------
# Exact status labels (mirrors the task spec; do not silently drop any of these).
STATUS_SUCCESS = "success"
STATUS_PHASE_FAILED = "phase_synthesis_failed"
STATUS_DEGREE_CEILING = "degree_above_supported_ceiling"
STATUS_TARGET_FIT_FAILED = "target_fit_failed"
STATUS_MATRIX_GEN_FAILED = "matrix_generation_failed"
STATUS_NOT_ATTEMPTED_RUNTIME = "not_attempted_due_to_runtime_limit"
STATUS_NOT_APPLICABLE = "not_applicable"

# Boundedness tolerance for a valid QSVT polynomial (matches the demo convention).
BOUND_TOLERANCE = 2.0e-3


@dataclass(frozen=True, slots=True)
class BoundaryAttempt:
    """Result of fitting the bounded target and attempting phase synthesis."""

    alpha: float
    beta: float
    alpha_normalized: float
    domain_min: float
    bound_C: float
    degree_requested: int
    degree_effective: int
    phase_count: int
    phase_synthesis_status: str
    phase_synthesis_error_message: str
    pipeline_status: str
    target_grid_max_error: float
    bounded_max_abs: float
    boundedness_ok: bool
    coefficients: tuple[float, ...]

    @property
    def polynomial(self) -> Any:
        from numpy.polynomial import Polynomial

        return Polynomial(np.asarray(self.coefficients, dtype=np.float64))


def _phase_key(domain_min: float, alpha_normalized: float, degree: int, margin: float) -> tuple:
    # Round to a stable key so seed-/size-invariant polynomials share one synthesis.
    return (
        round(float(domain_min), 12),
        round(float(alpha_normalized), 12),
        int(degree),
        round(float(margin), 6),
    )


# The scalar bounded-target design + phases depend only on this key (not on U, V or
# matrix size), so caching here turns the full grid into a handful of syntheses.
_ATTEMPT_CACHE: dict[tuple, BoundaryAttempt] = {}


def attempt_bounded_target_phases(
    *,
    beta: float,
    alpha: float,
    domain_min: float,
    degree: int,
    margin: float = 1.05,
    max_synthesis_degree: int = 45,
    phase_cache_dir: str | Path = PHASE_CACHE_DIR,
) -> BoundaryAttempt:
    """Fit the bounded Ridge target ``p(s) ~ f(s)/C`` and attempt QSVT phase synthesis.

    ``max_synthesis_degree`` is the known synthesis ceiling: requested degrees above
    it are recorded as ``degree_above_supported_ceiling`` (the fit is still attempted
    to report the polynomial blow-up honestly, but a genuinely unbounded polynomial
    also yields ``target_fit_failed``). A synthesis exception is a real
    ``phase_synthesis_failed`` with the exact message; nothing here is fabricated.
    """

    alpha_normalized = float(alpha) / float(beta) ** 2
    domain_min = float(np.clip(domain_min, 1.0e-12, 0.999))
    cache_key = _phase_key(domain_min, alpha_normalized, degree, margin)
    cached = _ATTEMPT_CACHE.get(cache_key)
    if cached is not None:
        # Re-key beta/alpha (they only affect provenance, not the scalar design).
        return BoundaryAttempt(
            alpha=float(alpha),
            beta=float(beta),
            alpha_normalized=alpha_normalized,
            domain_min=domain_min,
            bound_C=cached.bound_C,
            degree_requested=int(degree),
            degree_effective=cached.degree_effective,
            phase_count=cached.phase_count,
            phase_synthesis_status=cached.phase_synthesis_status,
            phase_synthesis_error_message=cached.phase_synthesis_error_message,
            pipeline_status=cached.pipeline_status,
            target_grid_max_error=cached.target_grid_max_error,
            bounded_max_abs=cached.bounded_max_abs,
            boundedness_ok=cached.boundedness_ok,
            coefficients=cached.coefficients,
        )

    # 1. Fit the bounded odd polynomial (may blow up numerically at high degree).
    try:
        target = fit_codesigned_bounded_polynomial(
            beta=float(beta),
            alpha=float(alpha),
            domain_min=domain_min,
            domain_max=1.0,
            degree=int(degree),
            margin=float(margin),
        )
        coefficients = np.asarray(target.coefficients, dtype=np.float64)
        bound_c = float(target.bound_C)
        grid_error = float(target.fit_max_abs_error)
        bounded_max_abs = float(target.bounded_max_abs)
    except Exception as exc:  # pragma: no cover - defensive
        result = BoundaryAttempt(
            alpha=float(alpha),
            beta=float(beta),
            alpha_normalized=alpha_normalized,
            domain_min=domain_min,
            bound_C=float("nan"),
            degree_requested=int(degree),
            degree_effective=0,
            phase_count=0,
            phase_synthesis_status="not_attempted",
            phase_synthesis_error_message=f"fit_error:{type(exc).__name__}: {exc}",
            pipeline_status=STATUS_TARGET_FIT_FAILED,
            target_grid_max_error=float("nan"),
            bounded_max_abs=float("nan"),
            boundedness_ok=False,
            coefficients=(),
        )
        _ATTEMPT_CACHE[cache_key] = result
        return result

    # 2. Boundedness validation (a valid QSVT polynomial must satisfy |p| <= 1). This also
    #    rejects non-finite coefficients, so no separate finiteness gate is needed.
    try:
        validation = validate_qsvt_polynomial(
            coefficients, parity="odd", bound_tolerance=BOUND_TOLERANCE
        )
        bounded_max_abs = float(validation["max_abs_on_unit_interval"])
    except Exception as exc:
        message = f"boundedness:{type(exc).__name__}: {exc}"
        status = (
            STATUS_DEGREE_CEILING
            if int(degree) > max_synthesis_degree
            else (STATUS_TARGET_FIT_FAILED)
        )
        result = BoundaryAttempt(
            alpha=float(alpha),
            beta=float(beta),
            alpha_normalized=alpha_normalized,
            domain_min=domain_min,
            bound_C=bound_c,
            degree_requested=int(degree),
            degree_effective=0,
            phase_count=0,
            phase_synthesis_status="not_attempted",
            phase_synthesis_error_message=message,
            pipeline_status=status,
            target_grid_max_error=grid_error,
            bounded_max_abs=bounded_max_abs,
            boundedness_ok=False,
            coefficients=tuple(float(c) for c in coefficients),
        )
        _ATTEMPT_CACHE[cache_key] = result
        return result

    # 3. Above the known synthesis ceiling we do not attempt PennyLane synthesis.
    if int(degree) > max_synthesis_degree:
        result = BoundaryAttempt(
            alpha=float(alpha),
            beta=float(beta),
            alpha_normalized=alpha_normalized,
            domain_min=domain_min,
            bound_C=bound_c,
            degree_requested=int(degree),
            degree_effective=0,
            phase_count=0,
            phase_synthesis_status="not_attempted",
            phase_synthesis_error_message=(
                f"degree {degree} exceeds known synthesis ceiling {max_synthesis_degree}"
            ),
            pipeline_status=STATUS_DEGREE_CEILING,
            target_grid_max_error=grid_error,
            bounded_max_abs=bounded_max_abs,
            boundedness_ok=True,
            coefficients=tuple(float(c) for c in coefficients),
        )
        _ATTEMPT_CACHE[cache_key] = result
        return result

    # 4. Attempt PennyLane phase synthesis (a failure is recorded, never faked).
    try:
        cached_phases = synthesize_pennylane_phases_cached(
            coefficients,
            angle_solver="iterative",
            cache_dir=phase_cache_dir,
            cache_metadata={
                "domain_min": domain_min,
                "lambda": alpha_normalized,
                "degree": int(degree),
            },
        )
        phases = np.asarray(cached_phases.phases, dtype=np.float64)
        result = BoundaryAttempt(
            alpha=float(alpha),
            beta=float(beta),
            alpha_normalized=alpha_normalized,
            domain_min=domain_min,
            bound_C=bound_c,
            degree_requested=int(degree),
            degree_effective=int(phases.size - 1),
            phase_count=int(phases.size),
            phase_synthesis_status="completed",
            phase_synthesis_error_message="",
            pipeline_status=STATUS_SUCCESS,
            target_grid_max_error=grid_error,
            bounded_max_abs=bounded_max_abs,
            boundedness_ok=True,
            coefficients=tuple(float(c) for c in coefficients),
        )
    except Exception as exc:
        result = BoundaryAttempt(
            alpha=float(alpha),
            beta=float(beta),
            alpha_normalized=alpha_normalized,
            domain_min=domain_min,
            bound_C=bound_c,
            degree_requested=int(degree),
            degree_effective=0,
            phase_count=0,
            phase_synthesis_status="failed",
            phase_synthesis_error_message=f"{type(exc).__name__}: {exc}",
            pipeline_status=STATUS_PHASE_FAILED,
            target_grid_max_error=grid_error,
            bounded_max_abs=bounded_max_abs,
            boundedness_ok=True,
            coefficients=tuple(float(c) for c in coefficients),
        )
    _ATTEMPT_CACHE[cache_key] = result
    return result


def bounded_filter_gains(
    singular_values: np.ndarray, *, beta: float, alpha: float, bound_C: float
) -> np.ndarray:
    """Return the ideal bounded filter gains ``f(s_i)/C`` with ``s_i = sigma_i/beta``.

    ``f(s) = s/(s^2 + alpha/beta^2)`` is the normalized Ridge filter; dividing by the
    bound ``C`` gives the QSVT-realizable amplitude ``|p(s_i)| <= 1``.
    """

    s = np.asarray(singular_values, dtype=np.float64) / float(beta)
    alpha_normalized = float(alpha) / float(beta) ** 2
    f = s / (s**2 + alpha_normalized)
    if bound_C <= 0.0 or not np.isfinite(bound_C):
        return np.full_like(s, np.nan)
    return f / float(bound_C)


def uniform_postselection_probability(
    singular_values: np.ndarray, *, beta: float, alpha: float, bound_C: float
) -> float:
    """Matrix-intrinsic postselection success probability for a uniform input.

    For a QSVT block-encoding of ``A = H^T/beta`` applying the bounded filter, the
    ancilla-success probability on input ``|b> = sum_i b_i v_i`` is
    ``sum_i |b_i|^2 (f(s_i)/C)^2``. For a uniform superposition over the right
    singular directions this is ``mean_i (f(s_i)/C)^2`` -- a reproducible,
    input-independent proxy (it is *not* the residual-specific number reported by the
    4x4 demo). It collapses as ``C`` grows (light regularization).
    """

    gains = bounded_filter_gains(singular_values, beta=beta, alpha=alpha, bound_C=bound_C)
    if not np.all(np.isfinite(gains)):
        return float("nan")
    return float(np.mean(gains**2))


SUCCESS_PROBABILITY_DEFINITION = (
    "mean over right-singular directions of the squared bounded filter gain "
    "(f(sigma_i/beta)/C)^2 for a uniform-superposition input; a reproducible, "
    "input-independent postselection proxy (not the residual-specific demo number)"
)


def python_version() -> str:
    return sys.version.split()[0]


@lru_cache(maxsize=1)
def all_package_versions() -> dict[str, str | None]:
    return package_versions(
        ["numpy", "scipy", "pandas", "matplotlib", "pennylane", "qiskit", "qiskit-aer"]
    )
