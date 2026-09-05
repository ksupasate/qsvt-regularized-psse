from __future__ import annotations

from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.engineering_utils import RESOURCE_CAVEAT

FULL_VECTOR_CAVEAT = (
    "This shot-level proxy targets selected observables only. Full-vector "
    "reconstruction may require many separate estimates and is not demonstrated."
)


def readout_summary_markdown(
    resource_row: dict[str, Any],
    shot_rows: list[dict[str, Any]] | None = None,
) -> str:
    shot_section = ""
    if shot_rows:
        shot_section = "\n## Shot-Level Proxy Rows\n\n" + "\n".join(
            (
                f"- `{row['target_observable']}` with {row['shot_count']} shots: "
                f"standard error proxy {row['estimated_standard_error']:.6g}"
            )
            for row in shot_rows
        )
        shot_section += "\n"
    return f"""# QSVT Readout Analysis

{RESOURCE_CAVEAT}

## Readout Models

- Full vector reconstruction: estimates every component of the state update. This can
  dominate the cost and is not assumed by the resource summary.
- Selected bus/state-component readout: estimates targeted entries such as
  `Delta theta_i` or `Delta V_i` when only local corrections are needed.
- Observable estimation: estimates scalar quantities from the prepared solution state.

## PSSE-Relevant Observables

- `||Delta x||_2`
- `||H_tilde Delta x - r_tilde||_2`
- `|Delta theta_i|`
- `|Delta V_i|`
- weak-area correction magnitudes when row or state metadata identifies a weak area

## Selected Report Settings

- Readout model: `{resource_row.get("readout_model")}`
- Full-vector readout required by this report: `{resource_row.get("full_vector_readout_required")}`
- Caveat: {resource_row.get("readout_caveat")}
{shot_section}
"""


def resource_assumptions_markdown(resource_row: dict[str, Any]) -> str:
    return f"""# QSVT Resource Estimate Assumptions

{RESOURCE_CAVEAT}

The estimates are proxy calculations for a QSVT-compatible implementation pathway
for the same regularized spectral filter used by Ridge/Tikhonov. They do not
include a hardware-native oracle decomposition, state-preparation implementation,
phase synthesis for every reported row, error-corrected compilation, or full
readout sampling analysis.

## Model

- Matrix dimensions: `m={resource_row.get("m")}`, `n={resource_row.get("n")}`
- Matrix density reported in `sparsity`: nonzero entries divided by total entries.
- Normalization beta: spectral-norm bound for `H_tilde`.
- Polynomial degree: first configured Chebyshev proxy degree meeting epsilon when
  available, otherwise the maximum configured degree.
- Query count: `2 * degree + 1`.
- Depth estimate: query count multiplied by the number of nonzero matrix entries.
- State preparation: placeholder only.
- Readout: selected-component or observable estimation is assumed; full-vector
  reconstruction remains a major limitation.
"""


def shot_level_readout_rows(
    *,
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    x_hat: np.ndarray,
    shot_count: int,
) -> list[dict[str, Any]]:
    if shot_count <= 0:
        raise ValueError("shot_count must be positive")
    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    x = np.asarray(x_hat, dtype=np.float64)
    if H.ndim != 2 or r.ndim != 1 or x.ndim != 1:
        raise ValueError("H_tilde must be 2D and vectors must be 1D")
    if H.shape[0] != r.shape[0] or H.shape[1] != x.shape[0]:
        raise ValueError("incompatible H_tilde, r_tilde, and x_hat shapes")

    residual = H @ x - r
    selected_component = float(x[0])
    update_norm = float(np.linalg.norm(x))
    residual_norm = float(np.linalg.norm(residual))
    component_bound = max(1.0, update_norm)
    residual_bound = max(
        1.0,
        float(np.linalg.norm(r)) + float(np.linalg.norm(H, ord=2)) * update_norm,
    )
    rows = [
        _observable_row(
            target_observable="selected_state_component_0",
            target_value=selected_component,
            shot_count=shot_count,
            normalization_bound=component_bound,
            measurement_model="bounded signed component estimator",
        ),
        _observable_row(
            target_observable="update_vector_norm",
            target_value=update_norm,
            shot_count=shot_count,
            normalization_bound=max(1.0, update_norm),
            measurement_model="bounded nonnegative norm estimator",
        ),
        _observable_row(
            target_observable="residual_norm_proxy",
            target_value=residual_norm,
            shot_count=shot_count,
            normalization_bound=residual_bound,
            measurement_model="bounded residual-norm proxy estimator",
        ),
    ]
    for row in rows:
        row["full_vector_reconstruction_caveat"] = FULL_VECTOR_CAVEAT
    return rows


def _observable_row(
    *,
    target_observable: str,
    target_value: float,
    shot_count: int,
    normalization_bound: float,
    measurement_model: str,
) -> dict[str, Any]:
    bound = max(float(normalization_bound), np.finfo(float).eps)
    normalized_value = float(np.clip(target_value / bound, -1.0, 1.0))
    # Conservative bounded-estimator standard error proxy. It is not a backend
    # sampling result and is intended only to make readout scaling explicit.
    estimated_standard_error = bound * 0.5 / np.sqrt(float(shot_count))
    return {
        "target_observable": target_observable,
        "target_value": float(target_value),
        "shot_count": int(shot_count),
        "estimated_standard_error": float(estimated_standard_error),
        "normalization_bound": bound,
        "normalized_target_value": normalized_value,
        "measurement_model": measurement_model,
        "readout_scope": "shot_level_proxy_not_hardware_execution",
    }
