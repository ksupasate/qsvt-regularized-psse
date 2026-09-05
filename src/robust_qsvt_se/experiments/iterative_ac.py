from __future__ import annotations

from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.data.cases import ACCase, load_ac_case
from robust_qsvt_se.estimators.base import EstimatorResult
from robust_qsvt_se.estimators.hhl_style_inverse_proxy import HHLStyleInverseProxyEstimator
from robust_qsvt_se.estimators.huber_irls import HuberIRLSEstimator
from robust_qsvt_se.estimators.lav import LAVEstimator
from robust_qsvt_se.estimators.normal_equation_wls import NormalEquationWLSEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.qsvt_spectral import QSVTSpectralEstimator
from robust_qsvt_se.estimators.qsvt_unregularized_inverse import (
    QSVTUnregularizedInverseEstimator,
)
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.estimators.truncated_svd import TruncatedSVDEstimator
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
from robust_qsvt_se.experiments.reporting import (
    write_iterative_artifacts,
    write_iterative_sweep_artifacts,
)
from robust_qsvt_se.experiments.sweeps import generate_sweep_trials, summarize_sweep_metrics
from robust_qsvt_se.measurement.ac_linear import (
    ac_measurements_and_jacobian,
    default_ac_state_vector,
)
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.measurement.perturbations import add_bad_data_to_measurements
from robust_qsvt_se.utils.config import validate_config
from robust_qsvt_se.utils.seed import make_rng


@dataclass(slots=True)
class ACNonlinearProblem:
    case: ACCase
    true_state: np.ndarray
    initial_state: np.ndarray
    z: np.ndarray
    measurement_stds: np.ndarray
    measurement_labels: list[str]
    measurement_types: list[str]
    measurement_config: dict[str, Any]
    kept_row_indices: np.ndarray
    config_metadata: dict[str, Any]


@dataclass(slots=True)
class IterativeACResult:
    name: str
    x_hat: np.ndarray
    rmse: float
    angle_rmse: float
    voltage_magnitude_rmse: float
    residual_norm: float
    weighted_residual: float
    weighted_residual_norm: float
    weighted_residual_quadratic: float
    condition_number: float
    rank: int
    effective_rank: int
    singular_values: np.ndarray
    iterations: int
    converged: bool
    failed: bool
    failure_reason: str | None
    runtime_seconds: float
    extra_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x_hat": self.x_hat.tolist(),
            "rmse": self.rmse,
            "angle_rmse": self.angle_rmse,
            "voltage_magnitude_rmse": self.voltage_magnitude_rmse,
            "residual_norm": self.residual_norm,
            "weighted_residual": self.weighted_residual,
            "weighted_residual_norm": self.weighted_residual_norm,
            "weighted_residual_quadratic": self.weighted_residual_quadratic,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "effective_rank": self.effective_rank,
            "singular_values": self.singular_values.tolist(),
            "iterations": self.iterations,
            "converged": self.converged,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "runtime_seconds": self.runtime_seconds,
            "extra_diagnostics": self.extra_diagnostics,
        }


def run_iterative_ac_experiment(
    *,
    config: dict[str, Any],
    output_dir: Path,
    logger: Logger,
    resume: bool = False,
) -> dict[str, Any]:
    if config.get("sweeps"):
        return _run_iterative_ac_sweeps(
            config=config,
            output_dir=output_dir,
            logger=logger,
            resume=resume,
        )

    problem = build_ac_nonlinear_problem(config)
    results, trace = run_iterative_estimators(config=config, problem=problem, logger=logger)
    metrics = pd.DataFrame([_iterative_metrics_row(result, problem) for result in results])
    singular_values = _singular_values_frame(results)
    artifacts = write_iterative_artifacts(
        output_dir=output_dir,
        config=config,
        problem_metadata=problem_metadata(problem),
        metrics=metrics,
        iteration_trace=trace,
        singular_values=singular_values,
        results=[result.to_dict() for result in results],
        save_plots=bool(config["output"].get("save_plots", False)),
    )
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "metrics": metrics,
        "iteration_trace": trace,
        "results": results,
    }


def build_ac_nonlinear_problem(config: dict[str, Any]) -> ACNonlinearProblem:
    validate_config(config)
    rng = make_rng(int(config["seed"]))
    system_config = config["system"]
    measurement_config = dict(system_config["measurement"])
    linearization_config = dict(system_config["linearization"])
    scenario_config = config["scenario"]
    case = load_ac_case(
        str(system_config.get("case_name", "ieee14")),
        case_source=str(system_config.get("case_source", "builtin")),
    )
    true_state = default_ac_state_vector(case)
    angle_count = len(case.angle_state_buses)
    angle_perturbation_std = float(linearization_config.get("angle_perturbation_std", 0.01))
    voltage_perturbation_std = float(linearization_config.get("voltage_perturbation_std", 0.01))
    perturbation = np.concatenate(
        [
            rng.normal(0.0, angle_perturbation_std, size=angle_count),
            rng.normal(0.0, voltage_perturbation_std, size=len(case.voltage_state_buses)),
        ]
    )
    initial_state = true_state + perturbation
    min_voltage = float(linearization_config.get("min_voltage_magnitude", 0.5))
    initial_state[angle_count:] = np.maximum(initial_state[angle_count:], min_voltage)

    z_true, _, rows = ac_measurements_and_jacobian(case, true_state, measurement_config)
    noise_std = float(scenario_config.get("noise_std", 0.0))
    z_full = z_true + rng.normal(0.0, noise_std, size=z_true.shape)
    missing_ratio = float(scenario_config.get("missing_ratio", 0.0))
    n_drop = round(len(z_full) * missing_ratio)
    if n_drop:
        dropped_rows = np.sort(rng.choice(len(z_full), size=n_drop, replace=False))
        keep_mask = np.ones(len(z_full), dtype=bool)
        keep_mask[dropped_rows] = False
        kept_row_indices = np.flatnonzero(keep_mask)
    else:
        dropped_rows = np.array([], dtype=int)
        kept_row_indices = np.arange(len(z_full), dtype=int)
    z = z_full[kept_row_indices]
    kept_rows = [rows[index] for index in kept_row_indices]
    measurement_stds = np.array([row.std for row in kept_rows], dtype=np.float64)
    bad_data_metadata = {
        "measurement_buses": [list(row.buses) for row in kept_rows],
        "weak_area_buses": [int(bus) for bus in measurement_config.get("weak_area_buses", ())],
    }
    z, bad_data_metadata = add_bad_data_to_measurements(
        z,
        measurement_stds=measurement_stds,
        metadata=bad_data_metadata,
        bad_data_config=scenario_config.get("bad_data", {}),
        rng=rng,
    )
    return ACNonlinearProblem(
        case=case,
        true_state=true_state,
        initial_state=initial_state,
        z=z,
        measurement_stds=measurement_stds,
        measurement_labels=[row.label for row in kept_rows],
        measurement_types=[row.measurement_type for row in kept_rows],
        measurement_config=measurement_config,
        kept_row_indices=kept_row_indices,
        config_metadata={
            "case_name": case.name,
            "case_source": str(system_config.get("case_source", "builtin")),
            "dataset_source": case.source,
            "dataset_source_detail": case.source_detail,
            "external_case": case.source == "pypower",
            "mode": str(system_config.get("mode", "ac_iterative_state_estimation")),
            "scenario_name": scenario_config.get("name"),
            "seed": int(config["seed"]),
            "noise_std": noise_std,
            "missing_ratio": missing_ratio,
            "dropped_rows": dropped_rows.astype(int).tolist(),
            "angle_state_buses": list(case.angle_state_buses),
            "voltage_state_buses": list(case.voltage_state_buses),
            "angle_state_indices": list(range(angle_count)),
            "voltage_magnitude_state_indices": list(
                range(angle_count, angle_count + len(case.voltage_state_buses))
            ),
            "linearization_angle_perturbation_std": angle_perturbation_std,
            "linearization_voltage_perturbation_std": voltage_perturbation_std,
            "measurement_labels": [row.label for row in rows],
            "measurement_types": [row.measurement_type for row in rows],
            "measurement_buses": [list(row.buses) for row in kept_rows],
            "measurement_stds": measurement_stds.tolist(),
            "weak_area_buses": [int(bus) for bus in measurement_config.get("weak_area_buses", ())],
            **bad_data_metadata,
        },
    )


def run_iterative_estimators(
    *,
    config: dict[str, Any],
    problem: ACNonlinearProblem,
    logger: Logger,
) -> tuple[list[IterativeACResult], pd.DataFrame]:
    results: list[IterativeACResult] = []
    traces: list[dict[str, Any]] = []
    for estimator_config in config["estimators"]:
        estimator = _build_update_estimator(estimator_config)
        logger.info("Running iterative AC estimator %s", estimator.name)
        result, trace_rows = _run_one_iterative_estimator(config, problem, estimator)
        results.append(result)
        traces.extend(trace_rows)
        logger.info(
            "Iterative estimator %s finished converged=%s failed=%s iterations=%s rmse=%s",
            result.name,
            result.converged,
            result.failed,
            result.iterations,
            result.rmse,
        )
    return results, pd.DataFrame(traces)


def _run_one_iterative_estimator(
    config: dict[str, Any],
    problem: ACNonlinearProblem,
    estimator: Any,
) -> tuple[IterativeACResult, list[dict[str, Any]]]:
    iteration_config = config["system"]["iteration"]
    max_iterations = int(iteration_config["max_iterations"])
    update_tolerance = float(iteration_config["update_tolerance"])
    residual_tolerance = float(iteration_config["residual_tolerance"])
    damping = float(iteration_config["damping"])
    max_update_norm = float(iteration_config.get("max_update_norm", 1.0e6))
    residual_growth_limit = float(iteration_config.get("residual_growth_limit", 1.0e6))
    log_metrics = bool(iteration_config.get("log_iteration_metrics", False))
    log_condition_number = bool(iteration_config.get("log_condition_number", False))
    min_voltage = float(config["system"]["linearization"].get("min_voltage_magnitude", 0.5))
    x = problem.initial_state.copy()
    trace: list[dict[str, Any]] = []
    start = perf_counter()
    final_system: WeightedSystem | None = None
    final_update_result: EstimatorResult | None = None
    converged = False
    failed = False
    failure_reason: str | None = None

    for iteration in range(max_iterations):
        system, weighted_residual_before = _linearized_update_system(problem, x)
        final_system = system
        update_result = estimator.solve(system)
        final_update_result = update_result
        if update_result.failed or not np.all(np.isfinite(update_result.x_hat)):
            failed = True
            failure_reason = update_result.failure_reason or "update solver failed"
            trace.append(
                _trace_row(
                    estimator.name,
                    iteration,
                    weighted_residual_before,
                    float("nan"),
                    weighted_residual_before,
                    failed=True,
                    converged=False,
                )
            )
            break

        update = update_result.x_hat
        update_norm = float(np.linalg.norm(update))
        if update_norm > max_update_norm:
            failed = True
            failure_reason = f"update norm exceeded max_update_norm={max_update_norm}"
            trace.append(
                _trace_row(
                    estimator.name,
                    iteration,
                    weighted_residual_before,
                    float("nan"),
                    update_norm,
                    failed=True,
                    converged=False,
                )
            )
            break
        x_next = x + damping * update
        x_next[len(problem.case.angle_state_buses) :] = np.maximum(
            x_next[len(problem.case.angle_state_buses) :],
            min_voltage,
        )
        weighted_residual_after = _weighted_residual_norm(problem, x_next)
        if not np.isfinite(weighted_residual_after):
            failed = True
            failure_reason = "weighted residual became nonfinite"
            trace.append(
                _trace_row(
                    estimator.name,
                    iteration,
                    weighted_residual_before,
                    weighted_residual_after,
                    update_norm,
                    failed=True,
                    converged=False,
                )
            )
            break
        if (
            weighted_residual_before > 0.0
            and weighted_residual_after / weighted_residual_before > residual_growth_limit
        ):
            failed = True
            failure_reason = (
                f"weighted residual growth exceeded residual_growth_limit={residual_growth_limit}"
            )
            trace.append(
                _trace_row(
                    estimator.name,
                    iteration,
                    weighted_residual_before,
                    weighted_residual_after,
                    update_norm,
                    failed=True,
                    converged=False,
                )
            )
            break
        converged = update_norm <= update_tolerance or weighted_residual_after <= residual_tolerance
        trace.append(
            _trace_row(
                estimator.name,
                iteration,
                weighted_residual_before,
                weighted_residual_after,
                update_norm,
                failed=False,
                converged=converged,
                extra=_iteration_metrics(
                    problem,
                    system,
                    x_next,
                    log_metrics=log_metrics,
                    log_condition_number=log_condition_number,
                    converged=converged,
                ),
            )
        )
        x = x_next
        if not np.all(np.isfinite(x)):
            failed = True
            failure_reason = "state update produced nonfinite values"
            break
        if converged:
            break

    if final_system is None:
        final_system, _ = _linearized_update_system(problem, x)
    residual_norm = _weighted_residual_norm(problem, x)
    state_error = x - problem.true_state
    runtime = perf_counter() - start
    iterations = len(trace)
    if not converged and not failed and iterations >= max_iterations:
        failure_reason = "max_iterations reached"

    singular_values = final_system.singular_values()
    extra = {
        "damping": damping,
        "max_iterations": max_iterations,
        "update_tolerance": update_tolerance,
        "residual_tolerance": residual_tolerance,
        "max_update_norm": max_update_norm,
        "residual_growth_limit": residual_growth_limit,
    }
    if final_update_result is not None:
        extra["last_update_solver_failed"] = final_update_result.failed
        extra["last_update_solver_reason"] = final_update_result.failure_reason
        extra.update(final_update_result.extra_diagnostics)
    return (
        IterativeACResult(
            name=estimator.name,
            x_hat=x,
            rmse=_rmse(x, problem.true_state),
            angle_rmse=_rmse(
                state_error[: len(problem.case.angle_state_buses)],
                np.zeros(len(problem.case.angle_state_buses)),
            ),
            voltage_magnitude_rmse=_rmse(
                state_error[len(problem.case.angle_state_buses) :],
                np.zeros(len(problem.case.voltage_state_buses)),
            ),
            residual_norm=residual_norm,
            weighted_residual=residual_norm,
            weighted_residual_norm=residual_norm,
            weighted_residual_quadratic=residual_norm**2,
            condition_number=final_system.condition_number(),
            rank=final_system.rank(),
            effective_rank=final_system.effective_rank(),
            singular_values=singular_values,
            iterations=iterations,
            converged=converged,
            failed=failed,
            failure_reason=failure_reason,
            runtime_seconds=runtime,
            extra_diagnostics=extra,
        ),
        trace,
    )


def _linearized_update_system(
    problem: ACNonlinearProblem,
    state: np.ndarray,
) -> tuple[WeightedSystem, float]:
    values_full, jacobian_full, rows = ac_measurements_and_jacobian(
        problem.case,
        state,
        problem.measurement_config,
    )
    values = values_full[problem.kept_row_indices]
    jacobian = jacobian_full[problem.kept_row_indices, :]
    kept_labels = [rows[index].label for index in problem.kept_row_indices]
    if kept_labels != problem.measurement_labels:
        raise ValueError("iterative AC measurement layout changed during solve")
    residual = problem.z - values
    H_tilde = jacobian / problem.measurement_stds[:, None]
    r_tilde = residual / problem.measurement_stds
    system = WeightedSystem(
        H_tilde=H_tilde,
        r_tilde=r_tilde,
        x_true=problem.true_state - state,
        metadata=problem_metadata(problem),
    )
    return system, float(np.linalg.norm(r_tilde))


def _weighted_residual_norm(problem: ACNonlinearProblem, state: np.ndarray) -> float:
    values_full, _, rows = ac_measurements_and_jacobian(
        problem.case,
        state,
        problem.measurement_config,
    )
    values = values_full[problem.kept_row_indices]
    kept_labels = [rows[index].label for index in problem.kept_row_indices]
    if kept_labels != problem.measurement_labels:
        raise ValueError("iterative AC measurement layout changed during residual evaluation")
    return float(np.linalg.norm((problem.z - values) / problem.measurement_stds))


def _run_iterative_ac_sweeps(
    *,
    config: dict[str, Any],
    output_dir: Path,
    logger: Logger,
    resume: bool = False,
) -> dict[str, Any]:
    trials = generate_sweep_trials(config)
    if not resume:
        clear_checkpoint_files(output_dir)
    records = load_trial_records(output_dir) if resume else []
    completed_ids = completed_trial_ids(records)
    skipped_trials = 0
    started_at = perf_counter()
    write_checkpoint_state(
        output_dir,
        status="running",
        total_trials=len(trials),
        completed_trials=_record_count(records, "completed"),
        failed_trials=_record_count(records, "failed"),
        skipped_trials=0,
        started_at=started_at,
        message="iterative AC sweep started",
    )
    try:
        for trial in trials:
            if trial.trial_id in completed_ids:
                skipped_trials += 1
                logger.info("Skipping completed iterative AC trial %s", trial.trial_id)
                continue
            logger.info("Running iterative AC trial %s", trial.trial_id)
            trial_start = perf_counter()
            try:
                record = _run_iterative_ac_trial_record(trial, logger)
            except Exception as exc:
                logger.exception("Iterative AC trial %s failed; recording failure", trial.trial_id)
                record = _failed_iterative_trial_record(
                    trial,
                    config,
                    exc,
                    perf_counter() - trial_start,
                )
            append_trial_record(output_dir, record)
            records.append(record)
            completed_ids.add(trial.trial_id)
            _write_iterative_progress_update(
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
    summary_metrics = summarize_sweep_metrics(aggregate_metrics)
    iteration_trace = dataframe_from_records(records, "trace_rows")
    singular_values = dataframe_from_records(records, "singular_rows")
    trial_payloads = trial_payloads_from_records(records)
    artifacts = write_iterative_sweep_artifacts(
        output_dir=output_dir,
        config=config,
        aggregate_metrics=aggregate_metrics,
        summary_metrics=summary_metrics,
        iteration_trace=iteration_trace,
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
        message=f"completed {len(records)} checkpointed iterative AC trial records",
    )
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "aggregate_metrics": aggregate_metrics,
        "summary_metrics": summary_metrics,
        "iteration_trace": iteration_trace,
        "singular_values": singular_values,
        "trial_results": trial_payloads,
        "trials": trials,
    }


def _run_iterative_ac_trial_record(trial: Any, logger: Logger) -> dict[str, Any]:
    trial_start = perf_counter()
    problem = build_ac_nonlinear_problem(trial.config)
    results, trace = run_iterative_estimators(
        config=trial.config,
        problem=problem,
        logger=logger,
    )
    elapsed = perf_counter() - trial_start
    trace = trace.copy()
    for key, value in _trial_metadata(trial).items():
        trace[key] = value

    aggregate_rows = []
    singular_rows = []
    for result in results:
        row = _iterative_metrics_row(result, problem)
        row.update(_trial_metadata(trial))
        aggregate_rows.append(row)
        for singular_index, singular_value in enumerate(result.singular_values):
            singular_rows.append(
                {
                    "trial_id": trial.trial_id,
                    "sweep_name": trial.sweep_name,
                    "parameter": trial.parameter,
                    "value": trial.value,
                    "seed": trial.seed,
                    "estimator": result.name,
                    "singular_index": singular_index,
                    "singular_value": float(singular_value),
                }
            )
    return {
        "trial_id": trial.trial_id,
        "status": "completed",
        "elapsed_seconds": elapsed,
        "aggregate_rows": aggregate_rows,
        "trace_rows": trace.to_dict(orient="records"),
        "singular_rows": singular_rows,
        "trial_payload": {
            "trial_id": trial.trial_id,
            "sweep_name": trial.sweep_name,
            "parameter": trial.parameter,
            "value": trial.value,
            "seed": trial.seed,
            "problem_metadata": problem_metadata(problem),
            "results": [result.to_dict() for result in results],
        },
    }


def _failed_iterative_trial_record(
    trial: Any,
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
            "angle_rmse": np.nan,
            "voltage_magnitude_rmse": np.nan,
            "residual_norm": np.nan,
            "weighted_residual": np.nan,
            "weighted_residual_norm": np.nan,
            "weighted_residual_quadratic": np.nan,
            "condition_number": np.nan,
            "rank": np.nan,
            "effective_rank": np.nan,
            "runtime_seconds": elapsed_seconds,
            "iterations": 0,
            "converged": False,
            "failed": True,
            "failure_reason": error,
            "mode": config["system"].get("mode"),
            "case_name": config["system"].get("case_name"),
            "dataset_source": config["system"].get("case_source"),
            "n_measurements": np.nan,
            "n_states": np.nan,
            "scenario_name": config.get("scenario", {}).get("name"),
            "noise_std": config.get("scenario", {}).get("noise_std"),
            "missing_ratio": config.get("scenario", {}).get("missing_ratio"),
            "bad_data_ratio": config.get("scenario", {}).get("bad_data", {}).get("ratio"),
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
        "trace_rows": [],
        "singular_rows": [],
        "trial_payload": {
            "trial_id": trial.trial_id,
            "sweep_name": trial.sweep_name,
            "parameter": trial.parameter,
            "value": trial.value,
            "seed": trial.seed,
            "problem_metadata": {},
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


def _trial_metadata(trial: Any) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "sweep_name": trial.sweep_name,
        "sweep_parameter": trial.parameter,
        "sweep_value": trial.value,
        "seed": trial.seed,
    }


def _write_iterative_progress_update(
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
        message=f"{processed}/{total_trials} iterative AC trials checkpointed",
    )


def _record_count(records: list[dict[str, Any]], status: str) -> int:
    return sum(1 for record in records if record.get("status") == status)


def _processed_count(records: list[dict[str, Any]]) -> int:
    return _record_count(records, "completed") + _record_count(records, "failed")


def _iterative_metrics_row(
    result: IterativeACResult,
    problem: ACNonlinearProblem,
) -> dict[str, Any]:
    row = {
        "estimator": result.name,
        "rmse": result.rmse,
        "angle_rmse": result.angle_rmse,
        "voltage_magnitude_rmse": result.voltage_magnitude_rmse,
        "residual_norm": result.residual_norm,
        "weighted_residual": result.weighted_residual,
        "weighted_residual_norm": result.weighted_residual_norm,
        "weighted_residual_quadratic": result.weighted_residual_quadratic,
        "condition_number": result.condition_number,
        "rank": result.rank,
        "effective_rank": result.effective_rank,
        "runtime_seconds": result.runtime_seconds,
        "iterations": result.iterations,
        "converged": result.converged,
        "failed": result.failed,
        "failure_reason": result.failure_reason,
        "case_name": problem.case.name,
        "dataset_source": problem.config_metadata.get("dataset_source"),
        "dataset_source_detail": problem.config_metadata.get("dataset_source_detail"),
        "external_case": problem.config_metadata.get("external_case"),
        "mode": problem.config_metadata.get("mode", "ac_iterative_state_estimation"),
        "n_measurements": len(problem.z),
        "n_states": len(problem.true_state),
        "scenario_name": problem.config_metadata.get("scenario_name"),
        "noise_std": problem.config_metadata.get("noise_std", 0.0),
        "missing_ratio": problem.config_metadata.get("missing_ratio", 0.0),
        "bad_data_ratio": problem.config_metadata.get("bad_data_ratio", 0.0),
        "bad_data_count": problem.config_metadata.get("bad_data_count", 0),
        "bad_data_magnitude": problem.config_metadata.get("bad_data_magnitude"),
        "bad_data_target": problem.config_metadata.get("bad_data_target"),
        "target_condition_number": None,
        "achieved_condition_number": result.condition_number,
    }
    row.update(_selected_iterative_extra_diagnostics(result))
    return row


def _selected_iterative_extra_diagnostics(result: IterativeACResult) -> dict[str, Any]:
    diagnostics = result.extra_diagnostics
    return {
        key: diagnostics.get(key)
        for key in (
            "normal_matrix_condition_number",
            "singular_values_below_cutoff",
            "unstable_ablation",
            "hhl_effective_condition_number",
            "hhl_resource_proxy",
            "hhl_instability_flag",
        )
        if key in diagnostics
    }


def _singular_values_frame(results: list[IterativeACResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for index, singular_value in enumerate(result.singular_values):
            rows.append(
                {
                    "estimator": result.name,
                    "singular_index": index,
                    "singular_value": float(singular_value),
                }
            )
    return pd.DataFrame(rows)


def _trace_row(
    estimator: str,
    iteration: int,
    weighted_residual_before: float,
    weighted_residual_after: float,
    update_norm: float,
    *,
    failed: bool,
    converged: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "estimator": estimator,
        "iteration": iteration,
        "weighted_residual_before": weighted_residual_before,
        "weighted_residual_after": weighted_residual_after,
        "update_norm": update_norm,
        "failed": failed,
        "converged": converged,
    }
    if extra:
        row.update(extra)
    return row


def _iteration_metrics(
    problem: ACNonlinearProblem,
    system: WeightedSystem,
    state: np.ndarray,
    *,
    log_metrics: bool,
    log_condition_number: bool,
    converged: bool,
) -> dict[str, Any] | None:
    """Per-iteration RMSE and conditioning, only when explicitly enabled.

    RMSE fields use the known controlled-benchmark true state; conditioning fields require
    an SVD of the iteration Jacobian and are gated separately so the cost is opt-in.
    """

    extra: dict[str, Any] = {}
    if log_metrics:
        angle_count = len(problem.case.angle_state_buses)
        error = np.asarray(state, dtype=np.float64) - np.asarray(
            problem.true_state, dtype=np.float64
        )
        extra["state_rmse"] = float(np.sqrt(np.mean(error**2)))
        extra["angle_rmse"] = float(np.sqrt(np.mean(error[:angle_count] ** 2)))
        extra["voltage_rmse"] = float(np.sqrt(np.mean(error[angle_count:] ** 2)))
        extra["converged_so_far"] = bool(converged)
    if log_condition_number:
        singular_values = system.singular_values()
        extra["sigma_min"] = (
            float(np.min(singular_values)) if singular_values.size else float("nan")
        )
        extra["sigma_max"] = (
            float(np.max(singular_values)) if singular_values.size else float("nan")
        )
        extra["condition_number"] = system.condition_number()
        extra["numerical_rank"] = system.rank()
        extra["effective_rank"] = system.effective_rank()
    return extra or None


def problem_metadata(problem: ACNonlinearProblem) -> dict[str, Any]:
    return {
        **problem.config_metadata,
        "initial_state": problem.initial_state.tolist(),
        "true_state": problem.true_state.tolist(),
        "measurement_count": len(problem.z),
        "kept_row_indices": problem.kept_row_indices.astype(int).tolist(),
    }


def _build_update_estimator(config: dict[str, Any]) -> Any:
    name = config["name"]
    if name == "pseudoinverse":
        return PseudoinverseEstimator(rcond=float(config.get("rcond", 1e-12)))
    if name == "normal_equation_wls":
        return NormalEquationWLSEstimator(
            max_gain_condition_number=float(config.get("max_gain_condition_number", float("inf")))
        )
    if name == "ridge":
        return RidgeEstimator(alpha=float(config["alpha"]))
    if name == "truncated_svd":
        return TruncatedSVDEstimator(tau=float(config["tau"]))
    if name == "qsvt_regularized":
        return QSVTSpectralEstimator(alpha=float(config["alpha"]))
    if name == "qsvt_unregularized_inverse":
        return QSVTUnregularizedInverseEstimator(cutoff=float(config.get("cutoff", 1.0e-8)))
    if name == "hhl_style_inverse_proxy":
        return HHLStyleInverseProxyEstimator(
            cutoff=float(config.get("cutoff", 1.0e-8)),
            precision=float(config.get("precision", 1.0e-3)),
            instability_condition_threshold=float(
                config.get("instability_condition_threshold", 1.0e8)
            ),
            fail_on_instability=bool(config.get("fail_on_instability", False)),
        )
    if name == "huber_irls":
        return HuberIRLSEstimator(
            delta=float(config["delta"]),
            max_iterations=int(config.get("max_iterations", 50)),
            tolerance=float(config.get("tolerance", 1.0e-8)),
        )
    if name == "lav":
        return LAVEstimator()
    raise ValueError(f"unknown estimator name: {name}")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))
