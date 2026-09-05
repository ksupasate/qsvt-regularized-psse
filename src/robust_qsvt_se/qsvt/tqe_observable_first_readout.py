from __future__ import annotations

import argparse
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    OBSERVABLE_READOUT_DIR,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    package_versions,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import load_sweep_subproblem
from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import ridge_update_svd
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import (
    qsvt_rescaled_update_from_transform,
    run_ieee_selected_block,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_COLUMNS = [
    "case_name",
    "subproblem_size",
    "alpha",
    "degree",
    "observable_name",
    "observable_type",
    "coordinate_indices",
    "metadata_label",
    "shot_access_model",
    "signed_observable",
    "sign_access_required",
    "shots",
    "seed",
    "ridge_value",
    "qsvt_statevector_value",
    "shot_estimate",
    "shot_abs_error_vs_ridge",
    "shot_rel_error_vs_ridge",
    "shot_abs_error_vs_qsvt_statevector",
    "shot_rel_error_vs_qsvt_statevector",
    "empirical_mean",
    "empirical_std",
    "ci95_lower",
    "ci95_upper",
    "theoretical_std",
    "norm_convention",
    "qsvt_update_norm",
    "ridge_update_norm",
    "success_probability",
    "simulation_status",
    "failure_or_skip_reason",
]

SUMMARY_COLUMNS = [
    "observable_name",
    "observable_type",
    "shots",
    "shot_accessible",
    "sign_access_required",
    "mean_shot_estimate",
    "mean_abs_error_vs_ridge",
    "mean_rel_error_vs_ridge",
    "mean_abs_error_vs_qsvt_statevector",
    "mean_rel_error_vs_qsvt_statevector",
    "ci95_width",
    "empirical_std",
    "theoretical_std",
    "ridge_value",
    "qsvt_statevector_value",
    "simulation_status",
]

DEFAULT_SHOTS = [100, 1000, 10000, 100000]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
SMALL_TOL = 1.0e-15
NORM_CONVENTION = (
    "amplitude-encoded QSVT-compatible update-state diagnostic; estimates use "
    "||Delta x_qsvt||^2 times computational-basis probability mass"
)


@dataclass(frozen=True, slots=True)
class ReadoutState:
    case_name: str
    subproblem_size: int
    alpha: float
    degree: int
    ridge_update: np.ndarray
    qsvt_update: np.ndarray
    success_probability: float
    metadata: dict[str, Any]
    source_note: str


@dataclass(frozen=True, slots=True)
class ObservableDefinition:
    name: str
    observable_type: str
    indices: tuple[int, ...]
    metadata_label: str
    shot_access_model: str
    signed_observable: bool
    sign_access_required: bool
    estimator: str


def run_observable_first_readout(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / OBSERVABLE_READOUT_DIR)
    tables_dir = paths["tables"]
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    state = load_readout_state(resolved)
    observables = build_observable_definitions(
        state.ridge_update,
        metadata=state.metadata,
        force_unsupported_signed=bool(resolved["force_unsupported_signed_readout"]),
    )
    rows, counts_payload = evaluate_observable_shots(
        state=state,
        observables=observables,
        shots_grid=[int(value) for value in resolved["shots_grid"]],
        seed_grid=[int(value) for value in resolved["seed_grid"]],
    )
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary = summarize_readout_results(results)

    results_csv = output_dir / "observable_first_readout_results.csv"
    counts_json = output_dir / "observable_first_readout_counts.json"
    metadata_json = output_dir / "observable_first_readout_metadata.json"
    summary_csv = tables_dir / "table_observable_first_readout_summary.csv"
    error_figure = figures_dir / "figure_observable_readout_error_vs_shots.png"
    ci_figure = figures_dir / "figure_observable_readout_ci_width.png"
    ridge_qsvt_figure = figures_dir / "figure_observable_readout_ridge_vs_qsvt.png"
    report_path = reports_dir / "observable_first_readout_report.md"

    results.to_csv(results_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_json(counts_json, counts_payload)
    _plot_error_vs_shots(summary, error_figure)
    _plot_ci_width(summary, ci_figure)
    _plot_ridge_qsvt_largest_shots(summary, ridge_qsvt_figure)
    report_path.write_text(
        _report_markdown(
            config=resolved,
            state=state,
            observables=observables,
            results=results,
            summary=summary,
            results_csv=results_csv,
            counts_json=counts_json,
            summary_csv=summary_csv,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "results_csv": str(results_csv),
        "counts_json": str(counts_json),
        "metadata_json": str(metadata_json),
        "summary_table_csv": str(summary_csv),
        "error_vs_shots_figure": str(error_figure),
        "ci_width_figure": str(ci_figure),
        "ridge_vs_qsvt_figure": str(ridge_qsvt_figure),
        "report": str(report_path),
    }
    ended_at = utc_timestamp()
    metadata = reproducibility_metadata(
        config=resolved,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts=artifacts,
    )
    metadata.update(
        {
            "input_artifact_paths": _input_artifact_paths(resolved),
            "shot_counts": list(resolved["shots_grid"]),
            "seed_grid": list(resolved["seed_grid"]),
            "readout_model": "computational-basis sampling of normalized update amplitudes",
            "confidence_interval_method": (
                "empirical mean +/- 1.96 standard errors across deterministic seeds; "
                "binomial or delta-method theoretical standard deviations reported"
            ),
            "input_qsvt_update_state_source": state.source_note,
            "status_counts": _status_counts(results),
        }
    )
    write_json(metadata_json, metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results,
        "summary": summary,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def normalized_update_probabilities(update: np.ndarray) -> np.ndarray:
    values = np.asarray(update, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("update must be a nonempty vector")
    norm = float(np.linalg.norm(values))
    if norm <= SMALL_TOL:
        raise ValueError("update vector norm is too small")
    probabilities = values**2 / norm**2
    return probabilities / float(np.sum(probabilities))


def single_coordinate_energy_estimate(
    counts: dict[int, int],
    *,
    index: int,
    update_norm: float,
    shots: int,
) -> float:
    p_hat = int(counts.get(int(index), 0)) / float(shots)
    return float(update_norm) ** 2 * p_hat


def subset_energy_estimate(
    counts: dict[int, int],
    *,
    indices: tuple[int, ...],
    update_norm: float,
    shots: int,
) -> float:
    selected = sum(int(counts.get(int(index), 0)) for index in indices)
    return float(update_norm) ** 2 * selected / float(shots)


def binomial_ci95(
    *,
    mean: float,
    std: float,
    trials: int,
) -> tuple[float, float]:
    if int(trials) <= 0 or not np.isfinite(std):
        return (np.nan, np.nan)
    half_width = 1.96 * float(std) / np.sqrt(int(trials))
    return float(mean - half_width), float(mean + half_width)


def theoretical_observable_std(
    *,
    probability: float,
    shots: int,
    update_norm: float,
    estimator: str,
) -> float:
    p = float(np.clip(probability, 0.0, 1.0))
    n = max(int(shots), 1)
    if estimator in {"single_coordinate_energy", "subset_energy", "pair_energy"}:
        return float(update_norm) ** 2 * np.sqrt(p * (1.0 - p) / n)
    if estimator == "single_coordinate_magnitude":
        if p <= SMALL_TOL:
            return np.nan
        return float(update_norm) * np.sqrt((1.0 - p) / (4.0 * n))
    return np.nan


def build_observable_definitions(
    ridge_update: np.ndarray,
    *,
    metadata: dict[str, Any],
    force_unsupported_signed: bool = False,
) -> list[ObservableDefinition]:
    ridge = np.asarray(ridge_update, dtype=np.float64)
    if ridge.ndim != 1 or ridge.size < 2:
        raise ValueError("observable definitions require at least two update coordinates")
    order = np.lexsort((np.arange(ridge.size), -np.abs(ridge)))
    dominant = int(order[0])
    partner = int(order[1])
    top2 = tuple(sorted(int(index) for index in order[: min(2, ridge.size)]))
    label_i = _coordinate_label(dominant, metadata)
    label_j = _coordinate_label(partner, metadata)
    signed_model = (
        "unsupported_signed_readout_forced_for_test"
        if force_unsupported_signed
        else "statevector_signed_diagnostic_not_computational_basis_shot_accessible"
    )
    return [
        ObservableDefinition(
            name="selected_coordinate_energy",
            observable_type="single_coordinate_energy",
            indices=(dominant,),
            metadata_label=f"{label_i}; largest-magnitude Ridge update coordinate",
            shot_access_model="computational_basis_probability",
            signed_observable=False,
            sign_access_required=False,
            estimator="single_coordinate_energy",
        ),
        ObservableDefinition(
            name="selected_coordinate_magnitude",
            observable_type="single_coordinate_magnitude",
            indices=(dominant,),
            metadata_label=f"{label_i}; magnitude proxy from sqrt(probability)",
            shot_access_model="computational_basis_probability_delta_method",
            signed_observable=False,
            sign_access_required=False,
            estimator="single_coordinate_magnitude",
        ),
        ObservableDefinition(
            name="branch_coordinate_difference_signed_proxy",
            observable_type="signed_coordinate_difference",
            indices=(dominant, partner),
            metadata_label=(
                f"coordinate-pair proxy {label_i} minus {label_j}; no branch endpoint "
                "metadata asserted"
            ),
            shot_access_model=signed_model,
            signed_observable=True,
            sign_access_required=True,
            estimator="signed_difference",
        ),
        ObservableDefinition(
            name="branch_pair_energy_sum_proxy",
            observable_type="pair_energy_sum_proxy",
            indices=tuple(sorted((dominant, partner))),
            metadata_label=(
                f"shot-accessible pair energy proxy over {label_i} and {label_j}; "
                "ordinary basis shots do not recover the signed difference"
            ),
            shot_access_model="computational_basis_probability_subset",
            signed_observable=False,
            sign_access_required=False,
            estimator="pair_energy",
        ),
        ObservableDefinition(
            name="selected_area_top2_energy",
            observable_type="subset_energy",
            indices=top2,
            metadata_label="top-2 largest Ridge update coordinates; selected-area proxy",
            shot_access_model="computational_basis_probability_subset",
            signed_observable=False,
            sign_access_required=False,
            estimator="subset_energy",
        ),
    ]


def load_readout_state(config: dict[str, Any]) -> ReadoutState:
    if "ridge_update" in config and "qsvt_update" in config:
        ridge = np.asarray(config["ridge_update"], dtype=np.float64)
        qsvt = np.asarray(config["qsvt_update"], dtype=np.float64)
        if ridge.shape != qsvt.shape:
            raise ValueError("ridge_update and qsvt_update must have the same shape")
        return ReadoutState(
            case_name=str(config.get("case_name", "synthetic")),
            subproblem_size=int(config.get("subproblem_size", ridge.size)),
            alpha=float(config.get("alpha", np.nan)),
            degree=int(config.get("degree", 0)),
            ridge_update=ridge,
            qsvt_update=qsvt,
            success_probability=float(config.get("success_probability", np.nan)),
            metadata=dict(config.get("metadata", {})),
            source_note="direct test/config update vectors",
        )

    integrated_results_path = Path(config["integrated_results_path"])
    if integrated_results_path.exists():
        integrated = pd.read_csv(integrated_results_path)
        matches = integrated[
            (integrated["run_type"] == "ieee_selected_block")
            & (integrated["simulation_status"] == "completed")
        ]
        if not matches.empty:
            row = matches.iloc[0]
            return _reconstruct_integrated_readout_state(config, row)
    return _fallback_end_to_end_readout_state(config)


def evaluate_observable_shots(
    *,
    state: ReadoutState,
    observables: list[ObservableDefinition],
    shots_grid: list[int],
    seed_grid: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qsvt_probabilities = normalized_update_probabilities(state.qsvt_update)
    qsvt_norm = float(np.linalg.norm(state.qsvt_update))
    ridge_norm = float(np.linalg.norm(state.ridge_update))
    rows: list[dict[str, Any]] = []
    counts_payload: dict[str, Any] = {
        "probabilities": qsvt_probabilities.tolist(),
        "counts_by_shots_seed": {},
    }
    for shots in shots_grid:
        for seed in seed_grid:
            counts = sample_counts(qsvt_probabilities, shots=int(shots), seed=int(seed))
            counts_payload["counts_by_shots_seed"][f"shots_{shots}_seed_{seed}"] = {
                str(index): int(count) for index, count in counts.items()
            }
            for observable in observables:
                rows.append(
                    _observable_seed_row(
                        state=state,
                        observable=observable,
                        counts=counts,
                        probabilities=qsvt_probabilities,
                        qsvt_norm=qsvt_norm,
                        ridge_norm=ridge_norm,
                        shots=int(shots),
                        seed=int(seed),
                    )
                )
    rows = _attach_empirical_statistics(rows)
    return rows, counts_payload


def sample_counts(probabilities: np.ndarray, *, shots: int, seed: int) -> dict[int, int]:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a nonempty vector")
    if int(shots) <= 0:
        raise ValueError("shots must be positive")
    probs = probs / float(np.sum(probs))
    rng = np.random.default_rng(int(seed))
    samples = rng.multinomial(int(shots), probs)
    return {int(index): int(count) for index, count in enumerate(samples) if count > 0}


def summarize_readout_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(
        ["observable_name", "observable_type", "shots"], dropna=False
    ):
        observable_name, observable_type, shots = keys
        estimates = pd.to_numeric(group["shot_estimate"], errors="coerce")
        shot_accessible = estimates.notna().any()
        mean_estimate = float(estimates.mean()) if shot_accessible else np.nan
        abs_ridge = pd.to_numeric(group["shot_abs_error_vs_ridge"], errors="coerce")
        rel_ridge = pd.to_numeric(group["shot_rel_error_vs_ridge"], errors="coerce")
        abs_qsvt = pd.to_numeric(group["shot_abs_error_vs_qsvt_statevector"], errors="coerce")
        rel_qsvt = pd.to_numeric(group["shot_rel_error_vs_qsvt_statevector"], errors="coerce")
        ci_width = pd.to_numeric(group["ci95_upper"], errors="coerce") - pd.to_numeric(
            group["ci95_lower"], errors="coerce"
        )
        rows.append(
            {
                "observable_name": observable_name,
                "observable_type": observable_type,
                "shots": int(shots),
                "shot_accessible": bool(shot_accessible),
                "sign_access_required": bool(group["sign_access_required"].iloc[0]),
                "mean_shot_estimate": mean_estimate,
                "mean_abs_error_vs_ridge": float(abs_ridge.mean()) if shot_accessible else np.nan,
                "mean_rel_error_vs_ridge": float(rel_ridge.mean()) if shot_accessible else np.nan,
                "mean_abs_error_vs_qsvt_statevector": (
                    float(abs_qsvt.mean()) if shot_accessible else np.nan
                ),
                "mean_rel_error_vs_qsvt_statevector": (
                    float(rel_qsvt.mean()) if shot_accessible else np.nan
                ),
                "ci95_width": float(ci_width.mean()) if shot_accessible else np.nan,
                "empirical_std": float(
                    pd.to_numeric(group["empirical_std"], errors="coerce").mean()
                )
                if shot_accessible
                else np.nan,
                "theoretical_std": float(
                    pd.to_numeric(group["theoretical_std"], errors="coerce").mean()
                )
                if shot_accessible
                else np.nan,
                "ridge_value": float(group["ridge_value"].iloc[0]),
                "qsvt_statevector_value": float(group["qsvt_statevector_value"].iloc[0]),
                "simulation_status": _status_join(group["simulation_status"]),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _observable_seed_row(
    *,
    state: ReadoutState,
    observable: ObservableDefinition,
    counts: dict[int, int],
    probabilities: np.ndarray,
    qsvt_norm: float,
    ridge_norm: float,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    ridge_value = _observable_exact_value(state.ridge_update, observable)
    qsvt_value = _observable_exact_value(state.qsvt_update, observable)
    probability = _observable_probability(probabilities, observable)
    theoretical_std = theoretical_observable_std(
        probability=probability,
        shots=shots,
        update_norm=qsvt_norm,
        estimator=observable.estimator,
    )
    if observable.sign_access_required:
        estimate = np.nan
        status = "skipped_sign_access_required"
        reason = (
            "ordinary computational-basis sampling gives squared amplitudes; signed "
            "coordinate differences require phase/sign-aware readout"
        )
    else:
        estimate = _shot_estimate(observable, counts, qsvt_norm=qsvt_norm, shots=shots)
        status = "completed"
        reason = ""
    return {
        "case_name": state.case_name,
        "subproblem_size": state.subproblem_size,
        "alpha": state.alpha,
        "degree": state.degree,
        "observable_name": observable.name,
        "observable_type": observable.observable_type,
        "coordinate_indices": " ".join(str(index) for index in observable.indices),
        "metadata_label": observable.metadata_label,
        "shot_access_model": observable.shot_access_model,
        "signed_observable": bool(observable.signed_observable),
        "sign_access_required": bool(observable.sign_access_required),
        "shots": int(shots),
        "seed": int(seed),
        "ridge_value": ridge_value,
        "qsvt_statevector_value": qsvt_value,
        "shot_estimate": estimate,
        "shot_abs_error_vs_ridge": abs(estimate - ridge_value) if np.isfinite(estimate) else np.nan,
        "shot_rel_error_vs_ridge": _relative_error(estimate, ridge_value),
        "shot_abs_error_vs_qsvt_statevector": (
            abs(estimate - qsvt_value) if np.isfinite(estimate) else np.nan
        ),
        "shot_rel_error_vs_qsvt_statevector": _relative_error(estimate, qsvt_value),
        "empirical_mean": np.nan,
        "empirical_std": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "theoretical_std": theoretical_std,
        "norm_convention": NORM_CONVENTION,
        "qsvt_update_norm": qsvt_norm,
        "ridge_update_norm": ridge_norm,
        "success_probability": state.success_probability,
        "simulation_status": status,
        "failure_or_skip_reason": reason,
    }


def _observable_exact_value(update: np.ndarray, observable: ObservableDefinition) -> float:
    values = np.asarray(update, dtype=np.float64)
    if observable.estimator == "single_coordinate_energy":
        return float(values[observable.indices[0]] ** 2)
    if observable.estimator == "single_coordinate_magnitude":
        return float(abs(values[observable.indices[0]]))
    if observable.estimator == "signed_difference":
        i, j = observable.indices
        return float(values[i] - values[j])
    if observable.estimator in {"pair_energy", "subset_energy"}:
        return float(np.sum(values[list(observable.indices)] ** 2))
    raise ValueError(f"unsupported estimator: {observable.estimator}")


def _observable_probability(probabilities: np.ndarray, observable: ObservableDefinition) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    if observable.estimator in {"single_coordinate_energy", "single_coordinate_magnitude"}:
        return float(probs[observable.indices[0]])
    if observable.estimator in {"pair_energy", "subset_energy"}:
        return float(np.sum(probs[list(observable.indices)]))
    return np.nan


def _shot_estimate(
    observable: ObservableDefinition,
    counts: dict[int, int],
    *,
    qsvt_norm: float,
    shots: int,
) -> float:
    if observable.estimator == "single_coordinate_energy":
        return single_coordinate_energy_estimate(
            counts,
            index=observable.indices[0],
            update_norm=qsvt_norm,
            shots=shots,
        )
    if observable.estimator == "single_coordinate_magnitude":
        energy = single_coordinate_energy_estimate(
            counts,
            index=observable.indices[0],
            update_norm=qsvt_norm,
            shots=shots,
        )
        return float(np.sqrt(max(energy, 0.0)))
    if observable.estimator in {"pair_energy", "subset_energy"}:
        return subset_energy_estimate(
            counts,
            indices=observable.indices,
            update_norm=qsvt_norm,
            shots=shots,
        )
    raise ValueError(f"unsupported shot estimator: {observable.estimator}")


def _attach_empirical_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows
    for _, group in frame.groupby(["observable_name", "shots"], dropna=False):
        values = pd.to_numeric(group["shot_estimate"], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        lower, upper = binomial_ci95(mean=mean, std=std, trials=len(values))
        for index in group.index:
            frame.loc[index, "empirical_mean"] = mean
            frame.loc[index, "empirical_std"] = std
            frame.loc[index, "ci95_lower"] = lower
            frame.loc[index, "ci95_upper"] = upper
    return frame[RESULT_COLUMNS].to_dict(orient="records")


def _reconstruct_integrated_readout_state(config: dict[str, Any], row: pd.Series) -> ReadoutState:
    spec = dict(config["subproblem_spec"])
    subproblem = load_sweep_subproblem(spec, seed=int(config["seed"]))
    evaluation = run_ieee_selected_block(
        {
            "seed": int(config["seed"]),
            "subproblem_spec": spec,
            "alpha": float(row["alpha"]),
            "epsilon_target": float(config["epsilon_target"]),
            "degree": int(row["degree"]),
            "block_results_path": str(config["block_results_path"]),
            "block_matrices_dir": str(config["block_matrices_dir"]),
            "end_to_end_results_path": str(config["end_to_end_results_path"]),
            "angle_solver": str(config["angle_solver"]),
            "basis_gates": list(config["basis_gates"]),
            "transpile_qubit_limit": int(config["transpile_qubit_limit"]),
            "transpile_optimization_level": int(config["transpile_optimization_level"]),
            "artifact_match_rtol": float(config["artifact_match_rtol"]),
            "artifact_match_atol": float(config["artifact_match_atol"]),
        }
    )
    if evaluation.transformed_block is None:
        raise ValueError("integrated QSVT reconstruction did not return a transform block")
    A = np.asarray(subproblem.H_tilde, dtype=np.float64)
    b = np.asarray(subproblem.r_tilde, dtype=np.float64)
    transformed = evaluation.transformed_block[: A.shape[0], : A.shape[1]]
    qsvt_update = qsvt_rescaled_update_from_transform(
        transformed,
        b,
        C_alpha=float(evaluation.row["C_alpha"]),
    )
    ridge_update = ridge_update_svd(A, b, alpha=float(row["alpha"]))
    return ReadoutState(
        case_name=str(row["case_name"]),
        subproblem_size=int(row["subproblem_size"]),
        alpha=float(row["alpha"]),
        degree=int(row["degree"]),
        ridge_update=ridge_update,
        qsvt_update=qsvt_update,
        success_probability=float(row.get("success_probability_residual_state", np.nan)),
        metadata=dict(subproblem.metadata),
        source_note=(
            "reconstructed integrated small-QSVT circuit transform from Experiment 3B; "
            "normalized QSVT-compatible update vector used as amplitude-encoded readout state"
        ),
    )


def _fallback_end_to_end_readout_state(config: dict[str, Any]) -> ReadoutState:
    path = Path(config["end_to_end_components_path"])
    if not path.exists():
        raise FileNotFoundError(f"end-to-end update components not found: {path}")
    frame = pd.read_csv(path)
    matches = frame[
        (frame["case_name"] == str(config["case_name"]))
        & (frame["subproblem_size"].astype(int) == int(config["subproblem_size"]))
        & np.isclose(frame["alpha"].astype(float), float(config["alpha"]))
        & np.isclose(frame["epsilon_target"].astype(float), float(config["epsilon_target"]))
    ]
    if matches.empty:
        raise ValueError("no matching end-to-end update components were available")
    selected = matches.sort_values(["degree", "component_index"])
    return ReadoutState(
        case_name=str(config["case_name"]),
        subproblem_size=int(config["subproblem_size"]),
        alpha=float(config["alpha"]),
        degree=int(selected["degree"].iloc[0]),
        ridge_update=selected["ridge_update_component"].to_numpy(dtype=np.float64),
        qsvt_update=selected["qsvt_poly_update_component"].to_numpy(dtype=np.float64),
        success_probability=np.nan,
        metadata={},
        source_note=(
            "fallback amplitude-encoded update-state diagnostic from Experiment 3 "
            "end-to-end QSVT-compatible update components"
        ),
    )


def _coordinate_label(index: int, metadata: dict[str, Any]) -> str:
    selected = metadata.get("selected_state_indices")
    if isinstance(selected, list) and int(index) < len(selected):
        return f"selected state coordinate {index} (source state index {selected[int(index)]})"
    return f"selected state coordinate {index}; no bus metadata available"


def _relative_error(estimate: float, reference: float) -> float:
    if not np.isfinite(estimate):
        return np.nan
    return float(abs(estimate - reference) / max(abs(reference), SMALL_TOL))


def _status_join(values: pd.Series) -> str:
    return ";".join(sorted(set(str(value) for value in values)))


def _status_counts(results: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "simulation_status": {
            str(key): int(value)
            for key, value in results["simulation_status"].value_counts(dropna=False).items()
        },
        "observable_name": {
            str(key): int(value)
            for key, value in results["observable_name"].value_counts(dropna=False).items()
        },
    }


def _input_artifact_paths(config: dict[str, Any]) -> dict[str, str]:
    return {
        "integrated_results_path": str(config["integrated_results_path"]),
        "end_to_end_components_path": str(config["end_to_end_components_path"]),
        "block_results_path": str(config["block_results_path"]),
        "block_matrices_dir": str(config["block_matrices_dir"]),
    }


def _plot_error_vs_shots(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    accessible = _shot_accessible_summary(summary)
    if accessible.empty:
        ax.text(0.5, 0.5, "No shot-accessible observables", ha="center", va="center")
    else:
        for name, group in accessible.groupby("observable_name"):
            ordered = group.sort_values("shots")
            ax.plot(
                ordered["shots"],
                _positive_for_log(ordered["mean_rel_error_vs_qsvt_statevector"]),
                marker="o",
                label=str(name),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots")
        ax.set_ylabel("mean relative shot error vs QSVT statevector")
        ax.set_title("Observable-First Readout Error vs Shots")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_ci_width(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    accessible = _shot_accessible_summary(summary)
    if accessible.empty:
        ax.text(0.5, 0.5, "No confidence intervals", ha="center", va="center")
    else:
        for name, group in accessible.groupby("observable_name"):
            ordered = group.sort_values("shots")
            ax.plot(
                ordered["shots"],
                _positive_for_log(ordered["ci95_width"]),
                marker="o",
                label=str(name),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots")
        ax.set_ylabel("empirical 95% CI width")
        ax.set_title("Observable Readout CI Width")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_ridge_qsvt_largest_shots(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    accessible = _shot_accessible_summary(summary)
    if accessible.empty:
        ax.text(0.5, 0.5, "No shot-accessible observables", ha="center", va="center")
    else:
        largest = int(accessible["shots"].max())
        rows = accessible[accessible["shots"] == largest].copy()
        labels = rows["observable_name"].astype(str).tolist()
        x = np.arange(len(rows))
        width = 0.25
        ax.bar(x - width, rows["ridge_value"].astype(float), width, label="Ridge")
        ax.bar(
            x,
            rows["qsvt_statevector_value"].astype(float),
            width,
            label="QSVT statevector",
        )
        ax.bar(
            x + width,
            rows["mean_shot_estimate"].astype(float),
            width,
            label=f"shots={largest}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("observable value")
        ax.set_title("Ridge, QSVT Statevector, and Largest-Shot Estimates")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _positive_for_log(values: pd.Series) -> np.ndarray:
    return np.maximum(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(), 1.0e-18)


def _shot_accessible_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "shot_accessible" not in summary:
        return summary
    return summary[summary["shot_accessible"].astype(bool)]


def _report_markdown(
    *,
    config: dict[str, Any],
    state: ReadoutState,
    observables: list[ObservableDefinition],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    results_csv: Path,
    counts_json: Path,
    summary_csv: Path,
) -> str:
    accessible = _shot_accessible_summary(summary)
    signed = [obs.name for obs in observables if obs.sign_access_required]
    metadata_available = any("source state index" in obs.metadata_label for obs in observables)
    metric_lines = _report_metric_lines(accessible, summary)
    return "\n".join(
        [
            "# Observable-First Readout with Shot Simulation Report",
            "",
            "## Goal",
            "",
            "This experiment estimates selected update observables from the "
            "QSVT-compatible output-state distribution using deterministic "
            "shot-based simulation.",
            "",
            "## Input QSVT State/Update Source",
            "",
            f"- {state.source_note}",
            f"- Case: {state.case_name}, size={state.subproblem_size}, alpha={state.alpha}, "
            f"degree={state.degree}.",
            "",
            "## Command and Environment",
            "",
            f"- Command: `{current_command()}`",
            f"- Python: {platform.python_version()}",
            f"- Platform: {platform.platform()}",
            f"- Package versions: {package_versions()}",
            "",
            "## Input Artifact Paths",
            "",
            *[f"- {key}: `{value}`" for key, value in _input_artifact_paths(config).items()],
            "",
            "## Observables Selected",
            "",
            *[
                f"- {obs.name}: {obs.observable_type}, indices={obs.indices}, "
                f"shot model={obs.shot_access_model}"
                for obs in observables
            ],
            "",
            "## Metadata Labels",
            "",
            f"- Metadata-mapped source state indices available: {metadata_available}.",
            "- Bus/branch physical labels are not asserted unless present in subproblem metadata.",
            "",
            "## Readout Model",
            "",
            "- Computational-basis sampling estimates probabilities |x_i|^2 of the "
            "normalized QSVT-compatible update state.",
            "- Energy observables use ||Delta x_qsvt||^2 times sampled probability mass.",
            "- Signed coordinate differences require phase/sign-aware readout and are "
            "reported separately from directly shot-accessible energy observables.",
            "",
            "## Shot Simulation and Confidence Intervals",
            "",
            f"- Shot counts: {config['shots_grid']}",
            f"- Seeds: {config['seed_grid']}",
            "- Empirical 95% intervals use mean +/- 1.96 standard errors across seeds.",
            "- Theoretical standard deviations use binomial or delta-method approximations.",
            "",
            "## Results",
            "",
            *metric_lines,
            f"- Signed diagnostics requiring sign-aware readout: {signed}.",
            f"- Status counts: {_status_counts(results)}",
            "",
            "## Claim-Safe Interpretation",
            "",
            "This experiment estimates selected update observables from the "
            "QSVT-compatible output-state distribution using shot-based simulation. "
            "Energy-style observables are naturally accessible from computational-basis "
            "sampling of an amplitude-encoded update state.",
            "",
            "Signed coordinate differences require phase/sign-aware readout and are "
            "reported separately from directly shot-accessible energy observables. "
            "The experiment characterizes selected-observable readout in simulation "
            "and does not solve full-vector recovery.",
            "",
            "## Limitations",
            "",
            "- The experiment uses a selected IEEE14 4x4 subproblem by default.",
            "- Shot simulation is classical and is not hardware execution.",
            "- Complete update-vector recovery, scalable state preparation, and "
            "full IEEE-scale readout remain outside scope.",
            "",
            "## Recommended Manuscript Wording",
            "",
            "Selected PSSE-relevant energy observables can be estimated from the "
            "QSVT-compatible output-state distribution in simulation, while signed "
            "differences and complete update-vector recovery require additional "
            "readout resources and remain outside this experiment.",
            "",
            "## Artifacts",
            "",
            f"- Results CSV: `{results_csv}`",
            f"- Counts JSON: `{counts_json}`",
            f"- Summary table: `{summary_csv}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _report_metric_lines(accessible: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    if accessible.empty:
        return ["- No shot-accessible observable rows were generated."]
    rel = pd.to_numeric(accessible["mean_rel_error_vs_qsvt_statevector"], errors="coerce")
    ci = pd.to_numeric(accessible["ci95_width"], errors="coerce")
    largest = accessible[accessible["shots"] == int(accessible["shots"].max())]
    ridge_gap = pd.to_numeric(summary["qsvt_statevector_value"], errors="coerce") - pd.to_numeric(
        summary["ridge_value"], errors="coerce"
    )
    return [
        "- Mean relative shot error vs QSVT statevector range: "
        f"{rel.min():.3e} to {rel.max():.3e}.",
        f"- Empirical 95% CI width range: {ci.min():.3e} to {ci.max():.3e}.",
        "- Largest-shot accessible observable mean relative error vs QSVT range: "
        f"{largest['mean_rel_error_vs_qsvt_statevector'].min():.3e} to "
        f"{largest['mean_rel_error_vs_qsvt_statevector'].max():.3e}.",
        "- Ridge-vs-QSVT statevector absolute discrepancy range across summary rows: "
        f"{np.nanmin(np.abs(ridge_gap)):.3e} to {np.nanmax(np.abs(ridge_gap)):.3e}.",
    ]


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(config or {})
    root = Path(supplied.get("output_root", OUTPUT_ROOT))
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "case_name": "ieee14",
        "subproblem_size": 4,
        "subproblem_spec": {
            "case_name": "ieee14",
            "subproblem_size": 4,
            "selection_mode": "high_leverage",
        },
        "alpha": 1.0e-2,
        "epsilon_target": 1.0e-3,
        "seed": 123,
        "shots_grid": DEFAULT_SHOTS,
        "seed_grid": DEFAULT_SEEDS,
        "integrated_results_path": str(
            root / "integrated_small_qsvt_circuit" / "integrated_small_qsvt_circuit_results.csv"
        ),
        "end_to_end_components_path": str(
            root / "end_to_end_qsvt_vs_ridge" / "end_to_end_qsvt_vs_ridge_update_components.csv"
        ),
        "end_to_end_results_path": str(
            root / "end_to_end_qsvt_vs_ridge" / "end_to_end_qsvt_vs_ridge_results.csv"
        ),
        "block_results_path": str(
            root / "explicit_block_encoding_demo" / "block_encoding_demo_results.csv"
        ),
        "block_matrices_dir": str(root / "explicit_block_encoding_demo" / "matrices"),
        "angle_solver": "root-finding",
        "basis_gates": ["rz", "sx", "x", "cx"],
        "transpile_qubit_limit": 4,
        "transpile_optimization_level": 1,
        "artifact_match_rtol": 1.0e-9,
        "artifact_match_atol": 1.0e-8,
        "force_unsupported_signed_readout": False,
    }
    resolved.update(supplied)
    resolved["subproblem_spec"] = dict(resolved["subproblem_spec"])
    resolved["shots_grid"] = [int(value) for value in resolved["shots_grid"]]
    resolved["seed_grid"] = [int(value) for value in resolved["seed_grid"]]
    if any(value <= 0 for value in resolved["shots_grid"]):
        raise ValueError("all shot counts must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TQE observable-first readout")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    run = run_observable_first_readout(
        {
            "output_root": args.output_root,
            "integrated_results_path": str(
                Path(args.output_root)
                / "integrated_small_qsvt_circuit"
                / "integrated_small_qsvt_circuit_results.csv"
            ),
            "end_to_end_components_path": str(
                Path(args.output_root)
                / "end_to_end_qsvt_vs_ridge"
                / "end_to_end_qsvt_vs_ridge_update_components.csv"
            ),
            "end_to_end_results_path": str(
                Path(args.output_root)
                / "end_to_end_qsvt_vs_ridge"
                / "end_to_end_qsvt_vs_ridge_results.csv"
            ),
            "block_results_path": str(
                Path(args.output_root)
                / "explicit_block_encoding_demo"
                / "block_encoding_demo_results.csv"
            ),
            "block_matrices_dir": str(
                Path(args.output_root) / "explicit_block_encoding_demo" / "matrices"
            ),
        }
    )
    print(f"TQE observable-first readout complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
