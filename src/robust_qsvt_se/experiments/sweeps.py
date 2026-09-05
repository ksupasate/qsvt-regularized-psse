from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.estimators.base import EstimatorResult
from robust_qsvt_se.experiments.checkpointing import (
    append_progress,
    append_trial_record,
    clear_checkpoint_files,
    completed_trial_ids,
    dataframe_from_records,
    load_trial_records,
    trial_payloads_from_records,
    write_checkpoint_state,
)
from robust_qsvt_se.experiments.metrics import metrics_row
from robust_qsvt_se.experiments.reporting import write_sweep_artifacts
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.utils.config import validate_config

TrialRunner = Callable[[dict[str, Any], Logger], tuple[WeightedSystem, list[EstimatorResult]]]


@dataclass(frozen=True, slots=True)
class SweepTrial:
    trial_id: str
    sweep_name: str
    parameter: str
    value: float
    seed: int
    config: dict[str, Any]


def get_by_dot_path(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for part in _split_dot_path(path):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        raise KeyError(f"config path does not exist: {path}")
    return current


def set_by_dot_path(config: dict[str, Any], path: str, value: Any) -> None:
    current: Any = config
    parts = _split_dot_path(path)
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        raise KeyError(f"config path does not exist: {path}")
    final = parts[-1]
    if isinstance(current, dict) and final in current:
        current[final] = value
        return
    if isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = value
        return
    raise KeyError(f"config path does not exist: {path}")


def generate_sweep_trials(config: dict[str, Any]) -> list[SweepTrial]:
    trials: list[SweepTrial] = []
    for sweep in config.get("sweeps", []):
        sweep_name = str(sweep["name"])
        parameter = str(sweep["parameter"])
        get_by_dot_path(config, parameter)
        for value in sweep["values"]:
            for seed in sweep["seeds"]:
                trial_config = deepcopy(config)
                trial_config["seed"] = int(seed)
                set_by_dot_path(trial_config, parameter, value)
                for linked_parameter in sweep.get("linked_parameters", []) or []:
                    set_by_dot_path(trial_config, str(linked_parameter), value)
                validate_config(trial_config)
                trial_id = f"{sweep_name}_{_value_slug(value)}_seed{seed}"
                trials.append(
                    SweepTrial(
                        trial_id=trial_id,
                        sweep_name=sweep_name,
                        parameter=parameter,
                        value=float(value),
                        seed=int(seed),
                        config=trial_config,
                    )
                )
    return trials


def run_sweeps(
    *,
    config: dict[str, Any],
    output_dir: Path,
    logger: Logger,
    trial_runner: TrialRunner,
    resume: bool = False,
) -> dict[str, Any]:
    trials = generate_sweep_trials(config)
    if not resume:
        _clear_checkpoint_files(output_dir)
    records = load_trial_records(output_dir) if resume else []
    completed_ids = completed_trial_ids(records)
    skipped_trials = 0
    started_at = perf_counter()

    logger.info("Starting %d sweep trials", len(trials))
    write_checkpoint_state(
        output_dir,
        status="running",
        total_trials=len(trials),
        completed_trials=_record_count(records, "completed"),
        failed_trials=_record_count(records, "failed"),
        skipped_trials=0,
        started_at=started_at,
        message="sweep run started",
    )
    try:
        for trial in trials:
            if trial.trial_id in completed_ids:
                skipped_trials += 1
                logger.info("Skipping completed trial %s", trial.trial_id)
                continue
            logger.info(
                "Running trial %s parameter=%s value=%s seed=%s",
                trial.trial_id,
                trial.parameter,
                trial.value,
                trial.seed,
            )
            trial_start = perf_counter()
            try:
                record = _run_sweep_trial_record(trial, trial_runner, logger)
            except Exception as exc:
                logger.exception("Trial %s failed; recording structured failure", trial.trial_id)
                record = _failed_trial_record(trial, config, exc, perf_counter() - trial_start)
            append_trial_record(output_dir, record)
            records.append(record)
            completed_ids.add(trial.trial_id)
            _write_progress_update(
                output_dir=output_dir,
                records=records,
                total_trials=len(trials),
                skipped_trials=skipped_trials,
                started_at=started_at,
                trial_id=trial.trial_id,
                status=str(record.get("status", "completed")),
            )
    except BaseException as exc:
        write_checkpoint_state(
            output_dir,
            status="incomplete",
            total_trials=len(trials),
            completed_trials=_record_count(records, "completed"),
            failed_trials=_record_count(records, "failed"),
            skipped_trials=skipped_trials,
            started_at=started_at,
            message=f"interrupted: {type(exc).__name__}: {exc}",
        )
        raise

    aggregate_metrics = dataframe_from_records(records, "aggregate_rows")
    singular_values = dataframe_from_records(records, "singular_rows")
    trial_payloads = trial_payloads_from_records(records)
    summary_metrics = summarize_sweep_metrics(aggregate_metrics)
    artifacts = write_sweep_artifacts(
        output_dir=output_dir,
        config=config,
        aggregate_metrics=aggregate_metrics,
        summary_metrics=summary_metrics,
        singular_values=singular_values,
        trial_results=trial_payloads,
        save_plots=bool(config["output"].get("save_plots", False)),
    )
    status = "complete" if _processed_count(records) >= len(trials) else "incomplete"
    write_checkpoint_state(
        output_dir,
        status=status,
        total_trials=len(trials),
        completed_trials=_record_count(records, "completed"),
        failed_trials=_record_count(records, "failed"),
        skipped_trials=skipped_trials,
        started_at=started_at,
        message=f"completed {len(records)} checkpointed trial records",
    )
    logger.info("Completed %d sweep trials", len(trials))
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "aggregate_metrics": aggregate_metrics,
        "summary_metrics": summary_metrics,
        "singular_values": singular_values,
        "trial_results": trial_payloads,
        "trials": trials,
    }


def _run_sweep_trial_record(
    trial: SweepTrial,
    trial_runner: TrialRunner,
    logger: Logger,
) -> dict[str, Any]:
    trial_start = perf_counter()
    system, results = trial_runner(trial.config, logger)
    elapsed = perf_counter() - trial_start
    aggregate_rows = []
    for result in results:
        row = metrics_row(result, system)
        row.update(_trial_metadata(trial))
        aggregate_rows.append(row)
    singular_rows = [
        {
            "trial_id": trial.trial_id,
            "sweep_name": trial.sweep_name,
            "parameter": trial.parameter,
            "value": trial.value,
            "seed": trial.seed,
            "singular_index": singular_index,
            "singular_value": float(singular_value),
        }
        for singular_index, singular_value in enumerate(system.singular_values())
    ]
    return {
        "trial_id": trial.trial_id,
        "status": "completed",
        "elapsed_seconds": elapsed,
        "aggregate_rows": aggregate_rows,
        "singular_rows": singular_rows,
        "trial_payload": {
            "trial_id": trial.trial_id,
            "sweep_name": trial.sweep_name,
            "parameter": trial.parameter,
            "value": trial.value,
            "seed": trial.seed,
            "system_metadata": system.metadata,
            "results": [result.to_dict() for result in results],
        },
    }


def _failed_trial_record(
    trial: SweepTrial,
    config: dict[str, Any],
    exc: Exception,
    elapsed_seconds: float,
) -> dict[str, Any]:
    error = f"{type(exc).__name__}: {exc}"
    rows = []
    for estimator_config in config.get("estimators", []):
        row = {
            "estimator": estimator_config.get("name"),
            "rmse": np.nan,
            "residual_norm": np.nan,
            "weighted_residual": np.nan,
            "weighted_residual_norm": np.nan,
            "weighted_residual_quadratic": np.nan,
            "condition_number": np.nan,
            "rank": np.nan,
            "effective_rank": np.nan,
            "runtime_seconds": elapsed_seconds,
            "converged": False,
            "failed": True,
            "failure_reason": error,
        }
        row.update(_trial_metadata(trial))
        rows.append(row)
    return {
        "trial_id": trial.trial_id,
        "status": "failed",
        "elapsed_seconds": elapsed_seconds,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "aggregate_rows": rows,
        "singular_rows": [],
        "trial_payload": {
            "trial_id": trial.trial_id,
            "sweep_name": trial.sweep_name,
            "parameter": trial.parameter,
            "value": trial.value,
            "seed": trial.seed,
            "system_metadata": {},
            "failed": True,
            "failure_reason": error,
            "results": [
                {
                    "name": estimator_config.get("name"),
                    "failed": True,
                    "failure_reason": error,
                    "runtime_seconds": elapsed_seconds,
                }
                for estimator_config in config.get("estimators", [])
            ],
        },
    }


def _trial_metadata(trial: SweepTrial) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "sweep_name": trial.sweep_name,
        "sweep_parameter": trial.parameter,
        "sweep_value": trial.value,
        "seed": trial.seed,
    }


def _write_progress_update(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    total_trials: int,
    skipped_trials: int,
    started_at: float,
    trial_id: str,
    status: str,
) -> None:
    completed = _record_count(records, "completed")
    failed = _record_count(records, "failed")
    elapsed = perf_counter() - started_at
    processed = completed + failed
    average = elapsed / processed if processed else None
    remaining = max(0, total_trials - processed)
    eta = None if average is None else average * remaining
    append_progress(
        output_dir,
        trial_id=trial_id,
        status=status,
        completed_trials=completed,
        failed_trials=failed,
        total_trials=total_trials,
        elapsed_seconds=elapsed,
        average_trial_seconds=average,
        estimated_remaining_seconds=eta,
    )
    write_checkpoint_state(
        output_dir,
        status="running",
        total_trials=total_trials,
        completed_trials=completed,
        failed_trials=failed,
        skipped_trials=skipped_trials,
        started_at=started_at,
        last_trial_id=trial_id,
        message=f"{processed}/{total_trials} trials checkpointed",
    )


def _record_count(records: list[dict[str, Any]], status: str) -> int:
    return sum(1 for record in records if record.get("status") == status)


def _processed_count(records: list[dict[str, Any]]) -> int:
    return _record_count(records, "completed") + _record_count(records, "failed")


def _clear_checkpoint_files(output_dir: Path) -> None:
    clear_checkpoint_files(output_dir)


def summarize_sweep_metrics(aggregate_metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["sweep_name", "sweep_parameter", "sweep_value", "estimator"]
    rows = [
        _summary_row(group_key, group_frame, group_columns)
        for group_key, group_frame in aggregate_metrics.groupby(group_columns, dropna=False)
    ]
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _summary_row(
    group_key: Any,
    group_frame: pd.DataFrame,
    group_columns: list[str],
) -> dict[str, Any]:
    if not isinstance(group_key, tuple):
        group_key = (group_key,)
    row = dict(zip(group_columns, group_key, strict=True))
    failed = (
        group_frame["failed"].astype(bool)
        if "failed" in group_frame
        else pd.Series(False, index=group_frame.index)
    )
    success_frame = group_frame.loc[~failed]
    row["n_trials"] = int(group_frame["trial_id"].nunique())
    row["n_successful_trials"] = int(success_frame["trial_id"].nunique())
    row["n_failed_trials"] = int(row["n_trials"] - row["n_successful_trials"])
    row["failed_count"] = int(failed.sum())
    row["failure_rate"] = float(failed.mean()) if len(failed) else 0.0

    for column in (
        "rmse",
        "angle_rmse",
        "voltage_magnitude_rmse",
        "residual_norm",
        "weighted_residual",
        "weighted_residual_norm",
        "weighted_residual_quadratic",
        "condition_number",
        "runtime_seconds",
        "iterations",
    ):
        if column in group_frame.columns:
            row.update(_metric_summary(success_frame[column], prefix=column))
    return row


def _metric_summary(values: pd.Series, *, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q1": np.nan,
            f"{prefix}_q3": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_ci95_low": np.nan,
            f"{prefix}_ci95_high": np.nan,
        }
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    ci_half_width = 1.96 * std / float(np.sqrt(len(numeric))) if len(numeric) > 1 else 0.0
    ci_low = mean - ci_half_width
    if _is_nonnegative_metric(prefix):
        ci_low = max(0.0, ci_low)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": q3 - q1,
        f"{prefix}_ci95_low": ci_low,
        f"{prefix}_ci95_high": mean + ci_half_width,
    }


def _is_nonnegative_metric(prefix: str) -> bool:
    return prefix in {
        "rmse",
        "angle_rmse",
        "voltage_magnitude_rmse",
        "residual_norm",
        "weighted_residual",
        "weighted_residual_norm",
        "weighted_residual_quadratic",
        "condition_number",
        "runtime_seconds",
        "iterations",
    }


def _split_dot_path(path: str) -> list[str]:
    if not isinstance(path, str) or not path:
        raise KeyError("config path must be a non-empty string")
    parts = path.split(".")
    if any(not part for part in parts):
        raise KeyError(f"invalid config path: {path}")
    return parts


def _value_slug(value: Any) -> str:
    return str(value).replace("-", "neg").replace(".", "p")
