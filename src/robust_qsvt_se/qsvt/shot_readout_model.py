from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import (
    build_engineering_system,
    required_case_name,
    ridge_svd_solution,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

SHOT_READOUT_CAVEAT = (
    "Shot-level readout analysis is a sampling-cost model for selected observables. "
    "It is not a hardware experiment and does not solve the full state-vector "
    "readout problem."
)
FULL_VECTOR_CAVEAT = (
    "Full-vector reconstruction would require many separate component estimates; "
    "this report models selected observables only."
)
DEFAULT_SHOT_LEVELS = [100, 1000, 10000, 100000]


def build_shot_readout_model(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    alpha = float(resolved["alpha"])
    x_hat = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=alpha)
    observables = selected_observable_specs(
        H_tilde=np.asarray(system.H_tilde, dtype=np.float64),
        r_tilde=np.asarray(system.r_tilde, dtype=np.float64),
        x_hat=x_hat,
        selected_component=int(resolved["selected_component"]),
    )
    shot_rows = _shot_rows(
        observables=observables,
        shot_levels=list(resolved["shot_levels"]),
        target_epsilon=float(resolved["target_epsilon"]),
        case_name=required_case_name(system),
        matrix_source=matrix_source,
        alpha=alpha,
        state_dimension=system.n_states,
    )
    estimate_rows = _observable_estimate_rows(
        observables=observables,
        target_epsilons=list(resolved["target_epsilon_grid"]),
        case_name=required_case_name(system),
        matrix_source=matrix_source,
        alpha=alpha,
        state_dimension=system.n_states,
    )

    shot_frame = pd.DataFrame(shot_rows)
    estimate_frame = pd.DataFrame(estimate_rows)
    summary_csv = output_dir / "shot_readout_summary.csv"
    summary_json = output_dir / "shot_readout_summary.json"
    estimates_csv = output_dir / "observable_estimates.csv"
    caveats_md = output_dir / "readout_caveats.md"
    shot_frame.to_csv(summary_csv, index=False)
    estimate_frame.to_csv(estimates_csv, index=False)
    write_json(summary_json, {"rows": shot_rows, "observable_estimates": estimate_rows})
    caveats_md.write_text(_caveats_markdown(), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "shot_readout_summary_csv": str(summary_csv),
            "shot_readout_summary_json": str(summary_json),
            "observable_estimates_csv": str(estimates_csv),
            "readout_caveats_md": str(caveats_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": shot_frame,
        "artifacts": {
            "shot_readout_summary_csv": summary_csv,
            "shot_readout_summary_json": summary_json,
            "observable_estimates_csv": estimates_csv,
            "readout_caveats_md": caveats_md,
            "manifest": manifest_path,
        },
    }


def selected_observable_specs(
    *,
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    x_hat: np.ndarray,
    selected_component: int,
) -> list[dict[str, Any]]:
    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    x = np.asarray(x_hat, dtype=np.float64)
    if H.ndim != 2 or r.ndim != 1 or x.ndim != 1:
        raise ValueError("H_tilde must be 2D and vectors must be 1D")
    if H.shape[0] != r.shape[0] or H.shape[1] != x.shape[0]:
        raise ValueError("incompatible H_tilde, r_tilde, and x_hat shapes")
    component = min(max(int(selected_component), 0), x.size - 1)
    residual = H @ x - r
    update_norm = float(np.linalg.norm(x))
    residual_norm = float(np.linalg.norm(residual))
    component_bound = max(update_norm, 1.0)
    residual_bound = 1.1 * max(
        residual_norm,
        float(np.linalg.norm(r)) + float(np.linalg.norm(H, ord=2)) * update_norm,
        1.0,
    )
    update_bound = 1.1 * max(update_norm, 1.0)
    component_amplitude = float(x[component] / component_bound)
    return [
        {
            "observable_name": "selected_state_component_0",
            "observable_type": "selected_state_component_probability_proxy",
            "target_component_if_any": int(component),
            "true_value_or_proxy": float(x[component]),
            "normalization_bound": float(component_bound),
            "probability_proxy": _clip_probability(component_amplitude**2),
            "measurement_note": "component amplitude squared in normalized solution proxy",
        },
        {
            "observable_name": "update_vector_norm",
            "observable_type": "update_vector_norm_probability_proxy",
            "target_component_if_any": None,
            "true_value_or_proxy": update_norm,
            "normalization_bound": float(update_bound),
            "probability_proxy": _clip_probability((update_norm / update_bound) ** 2),
            "measurement_note": "norm proxy with explicit normalization bound",
        },
        {
            "observable_name": "residual_norm_proxy",
            "observable_type": "residual_norm_probability_proxy",
            "target_component_if_any": None,
            "true_value_or_proxy": residual_norm,
            "normalization_bound": float(residual_bound),
            "probability_proxy": _clip_probability((residual_norm / residual_bound) ** 2),
            "measurement_note": "residual norm proxy, not direct full residual readout",
        },
    ]


def required_shots_for_additive_error(epsilon_obs: float) -> int:
    if epsilon_obs <= 0.0:
        raise ValueError("epsilon_obs must be positive")
    return ceil(0.25 / float(epsilon_obs) ** 2)


def _shot_rows(
    *,
    observables: list[dict[str, Any]],
    shot_levels: list[int],
    target_epsilon: float,
    case_name: str,
    matrix_source: str,
    alpha: float,
    state_dimension: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observable in observables:
        p = float(observable["probability_proxy"])
        for shots in shot_levels:
            standard_error = float(np.sqrt(p * (1.0 - p) / float(shots)))
            rows.append(
                {
                    "case_name": case_name,
                    "matrix_source": matrix_source,
                    "alpha": float(alpha),
                    "state_dimension": int(state_dimension),
                    "observable_name": observable["observable_name"],
                    "observable_type": observable["observable_type"],
                    "target_component_if_any": observable["target_component_if_any"],
                    "true_value_or_proxy": observable["true_value_or_proxy"],
                    "probability_proxy": p,
                    "shots": int(shots),
                    "shot_count": int(shots),
                    "estimated_value": p,
                    "standard_error": standard_error,
                    "estimated_standard_error": standard_error,
                    "relative_standard_error": float(
                        standard_error / max(abs(p), np.finfo(float).eps)
                    ),
                    "target_epsilon": float(target_epsilon),
                    "required_shots_estimate": required_shots_for_additive_error(target_epsilon),
                    "full_vector_reconstruction_required": False,
                    "readout_caveat": SHOT_READOUT_CAVEAT,
                    "full_vector_readout_caveat": FULL_VECTOR_CAVEAT,
                }
            )
    return rows


def _observable_estimate_rows(
    *,
    observables: list[dict[str, Any]],
    target_epsilons: list[float],
    case_name: str,
    matrix_source: str,
    alpha: float,
    state_dimension: int,
) -> list[dict[str, Any]]:
    return [
        {
            "case_name": case_name,
            "matrix_source": matrix_source,
            "alpha": float(alpha),
            "state_dimension": int(state_dimension),
            "observable_name": observable["observable_name"],
            "observable_type": observable["observable_type"],
            "target_component_if_any": observable["target_component_if_any"],
            "true_value_or_proxy": observable["true_value_or_proxy"],
            "normalization_bound": observable["normalization_bound"],
            "probability_proxy": observable["probability_proxy"],
            "target_epsilon": float(epsilon),
            "required_shots_estimate": required_shots_for_additive_error(float(epsilon)),
            "measurement_note": observable["measurement_note"],
            "readout_caveat": SHOT_READOUT_CAVEAT,
        }
        for observable in observables
        for epsilon in target_epsilons
    ]


def _caveats_markdown() -> str:
    return f"""# QSVT Shot-Level Readout Model

{SHOT_READOUT_CAVEAT}

The model reports Bernoulli standard errors for probability proxies associated
with selected observables: one state-component amplitude/probability proxy, the
update-vector norm proxy, and a residual-norm proxy.

{FULL_VECTOR_CAVEAT}

The estimates are sampling-cost diagnostics only. They are not QSVT hardware
execution, full-vector tomography, field-data validation, or evidence that QSVT
outperforms Ridge/Tikhonov under the same alpha and filter.
"""


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_shot_readout",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "alpha": 1.0e-2,
        "selected_component": 0,
        "shot_levels": DEFAULT_SHOT_LEVELS,
        "target_epsilon": 1.0e-2,
        "target_epsilon_grid": [1.0e-1, 5.0e-2, 1.0e-2],
    }
    if config:
        resolved.update(config)
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    resolved["shot_levels"] = [int(shots) for shots in resolved["shot_levels"]]
    if any(shots <= 0 for shots in resolved["shot_levels"]):
        raise ValueError("shot_levels must be positive")
    resolved["target_epsilon"] = float(resolved["target_epsilon"])
    if resolved["target_epsilon"] <= 0.0:
        raise ValueError("target_epsilon must be positive")
    resolved["target_epsilon_grid"] = [
        float(epsilon) for epsilon in resolved["target_epsilon_grid"]
    ]
    if any(epsilon <= 0.0 for epsilon in resolved["target_epsilon_grid"]):
        raise ValueError("target_epsilon_grid values must be positive")
    return resolved


def _clip_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build selected-observable shot readout model")
    parser.parse_args(argv)
    run = build_shot_readout_model()
    print(f"QSVT shot-readout model complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
