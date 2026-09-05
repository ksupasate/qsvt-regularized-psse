"""Shared scaffolding for the TQE reviewer-blocking extensions (Workstreams A/B/C).

Kept import-light: only numpy/pandas/yaml at module load.  The heavy qiskit/pennylane stack is
imported lazily inside the workstream modules that execute circuits.  Every IO helper delegates to
the already-validated :mod:`robust_qsvt_se.reviewer_blocking.common` writers so provenance and
checksum formats stay identical across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.reviewer_blocking.common import (
    atomic_write_csv,
    atomic_write_json,
    environment_record,
    now_iso,
    write_manifest_and_checksums,
)

from . import CLAIM_BOUNDARY

__all__ = [
    "CLAIM_BOUNDARY",
    "EVIDENCE_TIERS",
    "BoundCRecord",
    "analytic_bound_c",
    "atomic_write_csv",
    "atomic_write_json",
    "environment_record",
    "extract_min_feasible_degree",
    "load_yaml_config",
    "now_iso",
    "resolved_config_record",
    "write_manifest_and_checksums",
]

# Explicit, non-mergeable evidence tiers (task section 2).  Every result row records exactly one.
EVIDENCE_TIERS = {
    "SCALAR_POLY_FIT": "scalar_polynomial_approximation",
    "EXACT_MATRIX_ACTION": "exact_polynomial_matrix_action",
    "STATEVECTOR": "explicit_statevector_circuit_execution",
    "FINITE_SHOT": "finite_shot_circuit_simulation",
    "TRANSPILE_ONLY": "transpilation_only",
    "MODELED_RESOURCE": "modeled_resource_estimate",
    "SUPPORT_ERROR": "sparse_support_error",
    "PHYSICAL_ERROR": "truth_referenced_physical_error",
}


# --------------------------------------------------------------------------- config


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dict (frozen grid definitions)."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration at {path} must be a mapping")
    return payload


def resolved_config_record(config_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "config_path": str(config_path),
        "resolved_config": config,
        "environment": environment_record(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ------------------------------------------------------------------- boundedness factor C


@dataclass(frozen=True, slots=True)
class BoundCRecord:
    """Explicit boundedness factor ``C`` for ``g_lambda(s)=s/(s^2+lambda)`` on ``[s_lo, s_hi]``.

    The unconstrained maximizer of ``g`` on ``(0, inf)`` is ``s*=sqrt(lambda)`` with value
    ``1/(2 sqrt(lambda))``.  On the restricted synthesis interval the interval-dependent maximizer
    is ``s*`` clipped to ``[s_lo, s_hi]`` (``g`` increases on ``(0, sqrt(lambda))`` and decreases
    after), so the maximum is attained at the interior point when ``s_lo <= sqrt(lambda) <= s_hi``
    and at the nearer endpoint otherwise.
    """

    normalized_lambda: float
    s_lo: float
    s_hi: float
    s_star: float
    maximizer_location: str  # "interior" | "left_endpoint" | "right_endpoint"
    c_analytic: float  # g(s_star) - the interval maximum of g (no margin)
    c_selected: float  # margin * c_analytic - the value used to normalize the target
    margin: float
    resulting_max_target_magnitude: float  # max |f/C| on the interval = c_analytic / c_selected


def analytic_bound_c(
    normalized_lambda: float, s_lo: float, s_hi: float = 1.0, *, margin: float = 1.05
) -> BoundCRecord:
    """Return the interval-dependent boundedness factor for the normalized filter.

    ``margin`` widens ``C`` so the fitted bounded polynomial stays ``<= 1`` on ``[-1, 1]`` with a
    safety cushion; ``c_analytic`` is the exact interval maximum, ``c_selected=margin*c_analytic``
    the value the target is divided by, and ``resulting_max_target_magnitude = 1/margin``.
    """

    if not (0.0 < s_lo < s_hi):
        raise ValueError("require 0 < s_lo < s_hi")
    if normalized_lambda <= 0.0 or not np.isfinite(normalized_lambda):
        raise ValueError("normalized_lambda must be positive and finite")
    if margin < 1.0:
        raise ValueError("margin must be >= 1")

    sqrt_lambda = float(np.sqrt(normalized_lambda))

    def g(s: float) -> float:
        return float(s / (s * s + normalized_lambda))

    if s_lo <= sqrt_lambda <= s_hi:
        s_star, where = sqrt_lambda, "interior"
    elif sqrt_lambda < s_lo:
        s_star, where = s_lo, "left_endpoint"
    else:
        s_star, where = s_hi, "right_endpoint"

    c_analytic = g(s_star)
    c_selected = float(margin) * c_analytic
    return BoundCRecord(
        normalized_lambda=float(normalized_lambda),
        s_lo=float(s_lo),
        s_hi=float(s_hi),
        s_star=float(s_star),
        maximizer_location=where,
        c_analytic=float(c_analytic),
        c_selected=float(c_selected),
        margin=float(margin),
        resulting_max_target_magnitude=float(c_analytic / c_selected),
    )


# ----------------------------------------------------- minimum feasible degree extraction


def extract_min_feasible_degree(
    grid: pd.DataFrame,
    tolerances: list[float],
    *,
    group_keys: tuple[str, ...] = ("scope_id", "normalized_lambda"),
    degree_col: str = "degree",
    fit_error_col: str = "uniform_fit_error",
    bounded_col: str = "boundedness_ok",
    phase_status_col: str = "phase_synthesis_status",
) -> pd.DataFrame:
    """Extract ``d_min(group, epsilon_target)`` reproducibly from a raw feasibility grid.

    A degree is *fit-feasible* at ``epsilon_target`` iff it is bounded and its uniform fit error is
    ``<= epsilon_target``.  A degree is *synthesis-feasible* iff it is additionally fit-feasible and
    phase synthesis succeeded (``synthesized``); ``d_min_synthesis`` is left ``NaN`` when synthesis
    was never attempted at any feasible degree (recorded via ``d_min_synthesis_status``).
    """

    out: list[dict[str, Any]] = []
    for keys, block in grid.groupby(list(group_keys), dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record_base = dict(zip(group_keys, key_values, strict=True))
        block = block.sort_values(degree_col)
        for eps in tolerances:
            bounded = block[bounded_col].astype(bool)
            fit_ok = block[fit_error_col].astype(float) <= float(eps)
            feasible = bounded & fit_ok
            fit_degrees = block.loc[feasible, degree_col]
            d_min_fit = int(fit_degrees.min()) if not fit_degrees.empty else None
            synth = feasible & (block[phase_status_col].astype(str) == "synthesized")
            synth_degrees = block.loc[synth, degree_col]
            d_min_synth = int(synth_degrees.min()) if not synth_degrees.empty else None
            attempted = bool(
                (feasible & (block[phase_status_col].astype(str) != "not_attempted")).any()
            )
            out.append(
                {
                    **record_base,
                    "epsilon_target": float(eps),
                    "d_min_fit": d_min_fit,
                    "d_min_fit_status": (
                        "feasible" if d_min_fit is not None else "no_tested_degree_feasible"
                    ),
                    "d_min_synthesis": d_min_synth,
                    "d_min_synthesis_status": (
                        "synthesized"
                        if d_min_synth is not None
                        else (
                            "synthesis_failed_or_not_attempted"
                            if attempted or not fit_degrees.empty
                            else "no_fit_feasible_degree"
                        )
                    ),
                    "n_feasible_degrees": int(feasible.sum()),
                    "max_tested_degree": int(block[degree_col].max()),
                }
            )
    return pd.DataFrame(out)
