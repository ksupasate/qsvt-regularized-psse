"""Shared scaffolding for the selected-observable QSVT demonstration package.

This package adds a *complete small selected-observable QSVT demonstration* and
the surrounding resource/readout/oracle evidence requested for the TQE revision.
It does **not** change any estimator. Ridge/Tikhonov stays the matched classical
reference, and the QSVT pathway implements the *same* regularized spectral filter
``sigma / (sigma^2 + alpha)`` at the *same* ``alpha``. Every artifact is
feasibility / boundary evidence, not a speedup result, not full-vector readout,
not field-data validation, and not a quantum-hardware run.

Centralised here:

* the co-designed *bounded* QSVT target fit (the well-conditioned O(1) target
  ``f(s) = s / (s^2 + alpha/beta^2)`` rescaled by its bound ``C``),
* deterministic IEEE-derived block selection,
* the conservative claim-boundary wording and a dual forbidden-phrase check,
* a provenance-manifest writer with checksums.

Normalization convention (Option B, recorded in the project memory):
``p_bounded(s) ~ f(s)/C`` with ``f(s) = s/(s^2 + alpha/beta^2) = beta * P_alpha(beta s)``.
The QSVT singular-value transform of ``A = H_B^T / beta`` with ``p_bounded`` realises
``(1/C) * V diag(f(s_i)) U^T`` so the *single* physical rescale factor that maps the
postselected output back to the matched Ridge update is ``C/beta`` (never ``C*beta``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

from robust_qsvt_se.paper.tqe_revision_support_common import (
    find_forbidden,
    git_commit_hash,
    now_iso,
    package_versions,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

# Output roots for the five task deliverables. Existing outputs are never touched.
DEMO_DIR = Path("outputs/selected_observable_qsvt_demo")
ORACLE_DIR = Path("outputs/reversible_sparse_oracle")
RESOURCE_STACK_DIR = Path("outputs/resource_boundary_stack")
CLASSICAL_BASELINE_DIR = Path("outputs/classical_selected_observable_baseline")
EVIDENCE_TIERING_DIR = Path("outputs/qsvt_evidence_tiering")

# Conservative framing reused verbatim in every generated report header.
DEMO_CLAIM_BOUNDARY = (
    "Controlled IEEE/PYPOWER benchmark and QSVT feasibility study for regularized "
    "spectral filtering in ill-conditioned power-system state estimation. IEEE/PYPOWER "
    "cases provide benchmark network models; measurement rows are generated from the "
    "network equations (not field PMU/SCADA data). The QSVT component is a "
    "QSVT-compatible implementation pathway for the same Ridge/Tikhonov regularized "
    "spectral filter, evaluated at the same alpha on selected submatrices. The isolated "
    "shot experiment assumes direct preparation of the computed output state and is not "
    "integrated with residual loading, QSVT, and postselection. This work does not claim speedup, "
    "QSVT-over-Ridge numerical superiority, full-vector quantum readout, or full "
    "IEEE-scale execution on quantum hardware."
)

# Exact overclaim strings banned in addition to the repository-wide list.
DEMO_BANNED_OVERCLAIMS = (
    "quantum speedup demonstrated",
    "qsvt outperforms ridge",
    "qsvt beats ridge",
    "field pmu/scada validation",
    "pmu/scada validation",
    "full ieee-scale hardware execution",
    "full-vector readout solved",
    "full vector readout solved",
)


def forbidden_in(text: str) -> list[str]:
    """Return all forbidden phrases present in ``text`` (repository + demo sets)."""

    lowered = text.lower()
    repo_hits = find_forbidden(text)
    demo_hits = [phrase for phrase in DEMO_BANNED_OVERCLAIMS if phrase in lowered]
    return sorted(set(repo_hits) | set(demo_hits))


def assert_safe(text: str) -> None:
    """Raise if ``text`` contains any repository-wide or demo-banned overclaim."""

    violations = forbidden_in(text)
    if violations:
        raise RuntimeError(f"generated text contains forbidden wording: {violations}")


def checksum(path: str | Path) -> str | None:
    """Return the SHA-256 hex digest of ``path`` or ``None`` if it is absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_checksum(values: np.ndarray) -> str:
    """Deterministic SHA-256 over the rounded array bytes (provenance fingerprint)."""

    rounded = np.round(np.asarray(values, dtype=np.float64), 10) + 0.0  # normalise -0.0
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class CodesignedBoundedTarget:
    """A co-designed bounded odd QSVT polynomial for the regularized filter.

    ``coefficients`` are the *bounded* power-basis coefficients of ``p_bounded``
    with ``|p_bounded| <= 1`` on ``[-1, 1]``. The matched physical update is
    recovered from the QSVT transform of ``A = H_B^T/beta`` via the single factor
    ``C/beta`` (see module docstring).
    """

    degree: int
    alpha: float
    beta: float
    alpha_normalized: float
    domain_min: float
    domain_max: float
    bound_C: float
    margin: float
    coefficients: tuple[float, ...]
    fit_max_abs_error: float
    bounded_max_abs: float

    @property
    def polynomial(self) -> Polynomial:
        return Polynomial(np.asarray(self.coefficients, dtype=np.float64))

    @property
    def physical_recovery_factor(self) -> float:
        """The single physical rescale factor ``C/beta``."""

        return float(self.bound_C / self.beta)


def fit_codesigned_bounded_polynomial(
    *,
    beta: float,
    alpha: float,
    domain_min: float,
    domain_max: float,
    degree: int,
    grid_size: int = 6000,
    margin: float = 1.05,
) -> CodesignedBoundedTarget:
    """Fit an odd ``p_bounded(s) ~ f(s)/C`` for ``f(s) = s/(s^2 + alpha/beta^2)``.

    The odd parity is enforced exactly via ``p(s) = s * q(s^2)`` where ``q`` is a
    Chebyshev fit of ``1 / (C (s^2 + alpha/beta^2))`` on the squared domain. ``C``
    bounds ``f`` on the (margin-widened) domain so the returned polynomial is a
    valid QSVT polynomial (``|p_bounded| <= 1`` on ``[-1, 1]``; verified by the
    caller). Approximating the well-conditioned O(1) target ``f ~ 1/s`` keeps the
    relative accuracy controlled, unlike approximating the tiny ``P_alpha(sigma)``.
    """

    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be positive and finite")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not 0.0 < domain_min < domain_max <= 1.0:
        raise ValueError("domain must satisfy 0 < domain_min < domain_max <= 1")
    if degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    if margin < 1.0:
        raise ValueError("margin must be >= 1")
    if grid_size <= degree + 1:
        raise ValueError("grid_size must exceed degree + 1")

    alpha_normalized = float(alpha) / float(beta) ** 2
    reduced_degree = (degree - 1) // 2

    s_grid = np.linspace(domain_min, domain_max, grid_size, dtype=np.float64)
    f_grid = s_grid / (s_grid**2 + alpha_normalized)
    bound_c = float(margin * np.max(f_grid))

    y_min, y_max = domain_min**2, domain_max**2
    y_grid = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    q_target = 1.0 / (bound_c * (y_grid + alpha_normalized))
    q_poly = Chebyshev.fit(y_grid, q_target, deg=reduced_degree, domain=[y_min, y_max]).convert(
        kind=Polynomial
    )

    coefficients = np.zeros(degree + 1, dtype=np.float64)
    for index, coefficient in enumerate(q_poly.coef):
        target_index = 2 * index + 1
        if target_index <= degree:
            coefficients[target_index] = coefficient

    bounded = Polynomial(coefficients)
    fit_error = float(np.max(np.abs(bounded(s_grid) - f_grid / bound_c)))

    qsvt_grid = np.linspace(-1.0, 1.0, max(8193, 4 * grid_size + 1), dtype=np.float64)
    bounded_max_abs = float(np.max(np.abs(bounded(qsvt_grid))))

    return CodesignedBoundedTarget(
        degree=int(degree),
        alpha=float(alpha),
        beta=float(beta),
        alpha_normalized=alpha_normalized,
        domain_min=float(domain_min),
        domain_max=float(domain_max),
        bound_C=bound_c,
        margin=float(margin),
        coefficients=tuple(float(value) for value in coefficients),
        fit_max_abs_error=fit_error,
        bounded_max_abs=bounded_max_abs,
    )


def write_demo_manifest(
    *,
    output_dir: str | Path,
    artifact_name: str,
    description: str,
    command: str,
    artifacts: dict[str, Path],
    input_files: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    manifest_name: str = "manifest.json",
) -> Path:
    """Write a provenance manifest with checksums, git hash, and command line."""

    directory = ensure_directory(output_dir)
    path = directory / manifest_name
    generated = {name: str(value) for name, value in sorted(artifacts.items())}
    generated["manifest"] = str(path)
    checksums = {name: checksum(value) for name, value in sorted(artifacts.items())}
    manifest: dict[str, Any] = {
        "artifact_name": artifact_name,
        "description": description,
        "timestamp": now_iso(),
        "git_commit_hash": git_commit_hash(),
        "command_used": command,
        "input_files": sorted(input_files or []),
        "generated_file_paths": generated,
        "checksums": checksums,
        "key_package_versions": package_versions(
            ["numpy", "pandas", "scipy", "pennylane", "qiskit", "qiskit-aer"]
        ),
        "claim_boundary": DEMO_CLAIM_BOUNDARY,
        "changes_estimator_behavior": False,
        "fabricates_results": False,
    }
    if extra:
        manifest.update(extra)
    write_json(path, manifest)
    return path
