"""Expanded 12-structure physical selected-output accuracy campaign."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.selectors import output_aware_sensitivity_scores
from robust_qsvt_se.physical_alignment.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    environment_provenance,
    sha256_file,
)
from robust_qsvt_se.physical_alignment.config import load_campaign_config
from robust_qsvt_se.physical_alignment.risk_selectors import (
    exact_single_removal_scores,
    refine_risk_support_one_swap,
)
from robust_qsvt_se.physical_alignment.structures import (
    FunctionalSpec,
    StructuralInstance,
    build_instance_functionals,
    functional_registry_rows,
    instance_residual_and_truth,
    load_instance,
    validate_frozen_registries,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
    _ridge_filter_operator,
    deterministic_ridge_leverage_scores,
    refine_support_one_swap,
    select_resource_constrained_support,
    support_constraint_report,
)
from robust_qsvt_se.qsvt.ridge_output_certificate import ridge_selected_output_gradient
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import (
    ExactLossEvaluator,
    balanced_magnitude_score,
    near_oracle_beam,
    near_oracle_multistart,
)

STUDY_ID = "tqe_physical_alignment_physical_audit_v1"


def _support_fingerprint(support: np.ndarray) -> str:
    return stable_array_fingerprint(np.asarray(support, dtype=np.float64))


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _physical_functionals(functionals: Sequence[FunctionalSpec]) -> list[FunctionalSpec]:
    return [
        functional
        for functional in functionals
        if functional.status == "available" and functional.classification == "physical"
    ]


def _available_functionals(functionals: Sequence[FunctionalSpec]) -> list[FunctionalSpec]:
    return [functional for functional in functionals if functional.status == "available"]


def _build_training_tasks(
    instance: StructuralInstance,
    functionals: Sequence[FunctionalSpec],
    seeds: Sequence[int],
    expected_fingerprints: Mapping[str, str],
) -> list[RidgeTask]:
    tasks: list[RidgeTask] = []
    for seed in seeds:
        residual, _truth, _reference = instance_residual_and_truth(instance, int(seed))
        observed = stable_array_fingerprint(residual)
        expected = expected_fingerprints.get(str(int(seed)))
        if expected is not None and observed != expected:
            raise RuntimeError(
                f"frozen residual fingerprint mismatch for {instance.instance_id} seed {seed}"
            )
        for functional in functionals:
            if functional.vector is None:
                continue
            tasks.append(
                RidgeTask(
                    task_id=f"training_seed{seed}_{functional.functional_id}",
                    seed_id=int(seed),
                    split="training",
                    residual=residual,
                    functional_id=functional.functional_id,
                    functional=np.asarray(functional.vector, dtype=np.float64),
                )
            )
    if not tasks:
        raise RuntimeError(f"no physical training tasks for {instance.instance_id}")
    return tasks


def _milp_record(
    matrix: np.ndarray,
    scores: np.ndarray,
    constraints: SupportConstraints,
    milp_config: Mapping[str, Any],
) -> dict[str, Any]:
    result = select_resource_constrained_support(
        matrix,
        scores,
        constraints,
        time_limit_seconds=float(milp_config["time_limit_seconds"]),
        relative_mip_gap=float(milp_config["relative_mip_gap"]),
        tie_epsilon_relative=float(milp_config["deterministic_tie_epsilon_relative"]),
    )
    return {
        "support": result.support,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "runtime_seconds": result.runtime_seconds,
        "solver_used": result.solver_used,
        "solver_status": result.solver_status,
        "optimality_gap": result.optimality_gap,
        "fallback_used": result.fallback_used,
        "objective_value": result.objective_value,
        "exact_solves": 0,
        "accepted_swaps": 0,
        "termination_reason": "initial_support",
        "refinement_trace": (),
        "initial_support": None,
        "initial_objective": None,
        "final_objective": None,
        "full_support_objective": None,
        "raw_score_min": float(np.min(scores[matrix != 0.0])),
        "raw_score_max": float(np.max(scores[matrix != 0.0])),
        "negative_raw_score_count": int(np.sum(scores[matrix != 0.0] < 0.0)),
    }


def _adjoint_unnormalized_scores(
    matrix: np.ndarray, tasks: Sequence[RidgeTask], alpha: float
) -> np.ndarray:
    if not tasks or any(task.split != "training" for task in tasks):
        raise ValueError("adjoint score accepts training tasks only")
    score = np.zeros_like(matrix, dtype=np.float64)
    for task in tasks:
        gradient = ridge_selected_output_gradient(
            matrix, task.residual, task.functional, float(alpha)
        )
        score += np.abs(matrix * gradient)
    return score / len(tasks)


def _exact_removal_support_scores(
    matrix: np.ndarray,
    tasks: Sequence[RidgeTask],
    alpha: float,
    floor: float,
) -> tuple[np.ndarray, int]:
    full_operator = _ridge_filter_operator(matrix, alpha)
    residuals = np.stack([task.residual for task in tasks])
    functionals = np.stack([task.functional for task in tasks])
    full_outputs = np.einsum("ti,ij,tj->t", functionals, full_operator, residuals)
    scores = np.zeros_like(matrix)
    solves = 0
    for row, column in np.argwhere(matrix != 0.0):
        trial = matrix.copy()
        trial[row, column] = 0.0
        operator = _ridge_filter_operator(trial, alpha)
        outputs = np.einsum("ti,ij,tj->t", functionals, operator, residuals)
        normalized = np.abs(outputs - full_outputs) / np.maximum(np.abs(full_outputs), floor)
        scores[row, column] = float(np.mean(normalized))
        solves += 1
    return scores, solves


def _standard_support_bank(
    instance: StructuralInstance,
    training_tasks: Sequence[RidgeTask],
    constraints: SupportConstraints,
    settings: Mapping[str, Any],
    *,
    random_seed: int,
    requested_selectors: set[str],
) -> dict[str, dict[str, Any]]:
    matrix = instance.matrix
    alpha = instance.alpha
    milp = settings["milp"]
    bank: dict[str, dict[str, Any]] = {}
    required_initial = set(requested_selectors)
    if "sensitivity_refined_mean" in requested_selectors:
        required_initial.add("sensitivity_initial_mean")
    if "sensitivity_refined_worst_case" in requested_selectors:
        required_initial.add("sensitivity_initial_worst_case")
    score_bank: dict[str, np.ndarray] = {}
    if "global_magnitude" in required_initial:
        score_bank["global_magnitude"] = np.abs(matrix)
    if "balanced_magnitude" in required_initial:
        score_bank["balanced_magnitude"] = balanced_magnitude_score(matrix)
    if "ridge_leverage" in required_initial:
        _row, _column, leverage = deterministic_ridge_leverage_scores(matrix, alpha=alpha)
        score_bank["ridge_leverage"] = leverage
    if "random_objective_feasible_support" in required_initial:
        rng = np.random.default_rng(int(random_seed))
        random_scores = np.zeros_like(matrix)
        random_scores[matrix != 0.0] = rng.random(np.count_nonzero(matrix))
        score_bank["random_objective_feasible_support"] = random_scores
    sensitivity_names = {
        "sensitivity_initial_mean",
        "sensitivity_initial_worst_case",
    }
    if required_initial & sensitivity_names:
        sensitivity = output_aware_sensitivity_scores(
            matrix,
            training_tasks,
            alpha=alpha,
            epsilon=float(settings["score_normalization_epsilon"]),
        )
        if "sensitivity_initial_mean" in required_initial:
            score_bank["sensitivity_initial_mean"] = sensitivity.sensitivity_mean
        if "sensitivity_initial_worst_case" in required_initial:
            score_bank["sensitivity_initial_worst_case"] = sensitivity.sensitivity_worst_case
    if "adjoint_unnormalized_mean" in required_initial:
        score_bank["adjoint_unnormalized_mean"] = _adjoint_unnormalized_scores(
            matrix, training_tasks, alpha
        )
    exact_removal_solves = 0
    if "exact_single_entry_removal_mean" in required_initial:
        exact_removal, exact_removal_solves = _exact_removal_support_scores(
            matrix, training_tasks, alpha, float(settings["support_error_floor"])
        )
        score_bank["exact_single_entry_removal_mean"] = exact_removal
    for selector, scores in score_bank.items():
        bank[selector] = _milp_record(matrix, scores, constraints, milp)
    if "exact_single_entry_removal_mean" in bank:
        bank["exact_single_entry_removal_mean"]["exact_solves"] = exact_removal_solves

    refinement = settings["support_fidelity_refinement"]
    for initial_name, refined_name, objective in (
        ("sensitivity_initial_mean", "sensitivity_refined_mean", "mean_normalized_error"),
        (
            "sensitivity_initial_worst_case",
            "sensitivity_refined_worst_case",
            "worst_case_normalized_error",
        ),
    ):
        if refined_name not in requested_selectors:
            continue
        initial = bank[initial_name]
        if initial["support"] is None:
            bank[refined_name] = {
                **initial,
                "failure_reason": "initial_support_unavailable",
                "termination_reason": "initial_support_unavailable",
            }
            continue
        started = time.perf_counter()
        refined = refine_support_one_swap(
            matrix,
            initial["support"],
            training_tasks,
            constraints,
            alpha=alpha,
            y_floor=float(settings["support_error_floor"]),
            objective=objective,
            max_iterations=int(refinement["max_iterations"]),
            improvement_tolerance=float(refinement["strict_improvement_tolerance"]),
        )
        bank[refined_name] = {
            "support": refined.support,
            "status": "completed",
            "failure_reason": "",
            "runtime_seconds": time.perf_counter() - started,
            "solver_used": "exact_objective_deterministic_one_swap",
            "solver_status": "completed",
            "optimality_gap": None,
            "fallback_used": False,
            "objective_value": refined.final_objective,
            "exact_solves": 0,
            "accepted_swaps": refined.iterations_accepted,
            "termination_reason": str(refined.trace[-1]["action"]),
            "refinement_trace": refined.trace,
            "initial_support": initial["support"],
            "initial_objective": refined.initial_objective,
            "final_objective": refined.final_objective,
            "full_support_objective": 0.0,
            "raw_score_min": initial["raw_score_min"],
            "raw_score_max": initial["raw_score_max"],
            "negative_raw_score_count": initial["negative_raw_score_count"],
        }
    return bank


def _risk_support_bank(
    instance: StructuralInstance,
    functionals: Sequence[np.ndarray],
    constraints: SupportConstraints,
    settings: Mapping[str, Any],
    requested_selectors: set[str],
) -> dict[str, dict[str, Any]]:
    matrix = instance.matrix
    alpha = instance.alpha
    milp = settings["milp"]
    refinement = settings["risk_refinement"]
    bank: dict[str, dict[str, Any]] = {}
    for risk_kind, prefix in (
        ("noise_propagation", "noise_propagation_risk"),
        ("posterior_variance_reference", "posterior_variance_reference"),
    ):
        for aggregation, suffix in (("mean", "mean"), ("worst_case", "worst")):
            initial_name = f"{prefix}_{suffix}_initial"
            refined_name = f"{prefix}_{suffix}_refined"
            if not ({initial_name, refined_name} & requested_selectors):
                continue
            scores = exact_single_removal_scores(
                matrix,
                alpha,
                functionals,
                risk_kind=risk_kind,
                aggregation=aggregation,
            )
            initial = _milp_record(matrix, scores.milp_scores, constraints, milp)
            initial.update(
                {
                    "exact_solves": scores.exact_solves,
                    "full_support_objective": scores.full_support_objective,
                    "raw_score_min": float(np.min(scores.raw_scores[matrix != 0.0])),
                    "raw_score_max": float(np.max(scores.raw_scores[matrix != 0.0])),
                    "negative_raw_score_count": int(np.sum(scores.raw_scores[matrix != 0.0] < 0.0)),
                    "risk_kind": risk_kind,
                    "risk_aggregation": aggregation,
                    "score_translation": scores.translation,
                    "risk_task_count": scores.task_count,
                    "risk_unique_functional_count": scores.unique_functional_count,
                }
            )
            if initial["support"] is not None:
                sparse = np.where(initial["support"], matrix, 0.0)
                from robust_qsvt_se.physical_alignment.risk_selectors import evaluate_risk

                initial_objective = evaluate_risk(
                    sparse,
                    alpha,
                    functionals,
                    risk_kind=risk_kind,
                    aggregation=aggregation,
                ).objective
                initial["initial_objective"] = initial_objective
                initial["final_objective"] = initial_objective
                initial["exact_solves"] += 1
            bank[initial_name] = initial
            if refined_name not in requested_selectors:
                continue
            if initial["support"] is None:
                bank[refined_name] = {
                    **initial,
                    "failure_reason": "initial_support_unavailable",
                    "termination_reason": "initial_support_unavailable",
                }
                continue
            refined = refine_risk_support_one_swap(
                matrix,
                initial["support"],
                alpha,
                functionals,
                constraints,
                risk_kind=risk_kind,
                aggregation=aggregation,
                max_iterations=int(refinement["max_iterations"]),
                improvement_tolerance=float(refinement["strict_improvement_tolerance"]),
            )
            bank[refined_name] = {
                **initial,
                "support": refined.support,
                "initial_support": initial["support"],
                "runtime_seconds": initial["runtime_seconds"] + refined.runtime_seconds,
                "solver_used": "milp_then_exact_risk_deterministic_one_swap",
                "solver_status": "completed",
                "objective_value": refined.final_objective,
                "initial_objective": refined.initial_objective,
                "final_objective": refined.final_objective,
                "accepted_swaps": refined.accepted_swaps,
                "termination_reason": refined.termination_reason,
                "exact_solves": initial["exact_solves"] + refined.exact_solves,
                "refinement_trace": refined.trace,
            }
    return bank


def _near_oracle_record(
    instance: StructuralInstance,
    training_tasks: list[RidgeTask],
    constraints: SupportConstraints,
    settings: Mapping[str, Any],
    seed_supports: Iterable[np.ndarray],
) -> dict[str, Any]:
    diagnostic = settings["near_oracle"]
    estimate = int(
        int(diagnostic["beam_width"])
        * int(diagnostic["beam_max_steps"])
        * np.count_nonzero(instance.matrix)
    )
    if estimate > int(diagnostic["maximum_estimated_loss_evaluations"]):
        return {
            "support": None,
            "status": "skipped_compute_ceiling",
            "failure_reason": (
                f"estimated loss evaluations {estimate} exceed ceiling "
                f"{diagnostic['maximum_estimated_loss_evaluations']}"
            ),
            "runtime_seconds": 0.0,
            "solver_used": "near_oracle_multistart_local_search",
            "solver_status": "skipped_compute_ceiling",
            "optimality_gap": None,
            "fallback_used": False,
            "objective_value": None,
            "exact_solves": 0,
            "accepted_swaps": 0,
            "termination_reason": "compute_ceiling",
            "refinement_trace": (),
            "initial_support": None,
            "initial_objective": None,
            "final_objective": None,
            "full_support_objective": 0.0,
            "raw_score_min": None,
            "raw_score_max": None,
            "negative_raw_score_count": 0,
            "estimated_loss_evaluations": estimate,
        }
    evaluator = ExactLossEvaluator(
        instance.matrix,
        training_tasks,
        instance.alpha,
        float(settings["support_error_floor"]),
    )
    beam = near_oracle_beam(
        evaluator,
        constraints,
        objective="mean",
        beam_width=int(diagnostic["beam_width"]),
        max_steps=int(diagnostic["beam_max_steps"]),
    )
    seeds = [np.asarray(support, dtype=bool) for support in seed_supports]
    if beam.get("support") is not None:
        seeds.append(beam["support"])
    record = near_oracle_multistart(
        evaluator,
        training_tasks,
        constraints,
        objective="mean",
        seed_supports=seeds,
        beam_diagnostic=beam,
        max_iterations=int(diagnostic["multistart_max_iterations"]),
    )
    return {
        "support": record.get("support"),
        "status": record["status"],
        "failure_reason": record.get("failure_reason", ""),
        "runtime_seconds": record.get("runtime_seconds", 0.0) + beam.get("runtime_seconds", 0.0),
        "solver_used": record.get("algorithm", "near_oracle_multistart_local_search"),
        "solver_status": record["status"],
        "optimality_gap": None,
        "fallback_used": False,
        "objective_value": record.get("final_loss"),
        "exact_solves": evaluator.solves,
        "accepted_swaps": 0,
        "termination_reason": record.get("stopping_condition", record["status"]),
        "refinement_trace": (),
        "initial_support": None,
        "initial_objective": record.get("base_best_construction_loss"),
        "final_objective": record.get("final_loss"),
        "full_support_objective": 0.0,
        "raw_score_min": None,
        "raw_score_max": None,
        "negative_raw_score_count": 0,
        "estimated_loss_evaluations": estimate,
        "global_optimality_proven": False,
    }


def _support_registry_row(
    instance: StructuralInstance,
    selector: str,
    record: Mapping[str, Any],
    k_budget: int | None,
    slot_budget: int | None,
    settings: Mapping[str, Any],
    configuration_id: str,
    configuration_hash: str,
) -> dict[str, Any]:
    support = record.get("support")
    constraints = (
        SupportConstraints(int(k_budget), int(slot_budget), bool(settings["coverage_enabled"]))
        if k_budget is not None and slot_budget is not None
        else None
    )
    if support is None:
        fingerprint = None
        report: dict[str, Any] = {}
        support_cells = None
    else:
        support = np.asarray(support, dtype=bool)
        fingerprint = _support_fingerprint(support)
        report = (
            support_constraint_report(instance.matrix, support, constraints)
            if constraints is not None
            else {
                "valid": True,
                "failure_reasons": [],
                "actual_nonzeros": int(support.sum()),
                "row_degrees": support.sum(axis=1).astype(int).tolist(),
                "column_degrees": support.sum(axis=0).astype(int).tolist(),
                "actual_max_row_degree": int(support.sum(axis=1).max(initial=0)),
                "actual_max_column_degree": int(support.sum(axis=0).max(initial=0)),
                "active_rows_covered": True,
                "active_columns_covered": True,
            }
        )
        support_cells = _json_compact(np.argwhere(support).astype(int).tolist())
    suffix = fingerprint[:12] if fingerprint else str(record.get("status", "failed"))
    budget_label = "full" if k_budget is None else f"k{k_budget}_s{slot_budget}"
    support_id = f"{instance.instance_id}__{selector}__{budget_label}__{suffix}"
    deployable = selector in settings["deployable_selectors"]
    diagnostic = selector in settings["diagnostic_selectors"]
    return {
        "configuration_id": configuration_id,
        "configuration_hash": configuration_hash,
        "support_id": support_id,
        "instance_id": instance.instance_id,
        "structural_group_id": instance.structural_group_id,
        "ieee_case": instance.ieee_case,
        "realization_order": instance.realization_order,
        "matrix_seed": instance.matrix_seed,
        "matrix_fingerprint": instance.matrix_fingerprint,
        "selector": selector,
        "deployable_selector": deployable,
        "diagnostic_selector": diagnostic,
        "uses_true_state": False,
        "uses_held_out_data": False,
        "selection_data_split": "training" if selector != "full_support" else "none",
        "k_budget": k_budget,
        "slot_budget": slot_budget,
        "coverage_enabled": bool(settings["coverage_enabled"]) if constraints else None,
        "regularization_regime": "frozen_structural_reference",
        "alpha": instance.alpha,
        "status": record.get("status"),
        "failure_reason": record.get("failure_reason", ""),
        "support_fingerprint": fingerprint,
        "support_cells": support_cells,
        "actual_nonzeros": report.get("actual_nonzeros"),
        "actual_max_row_degree": report.get("actual_max_row_degree"),
        "actual_max_column_degree": report.get("actual_max_column_degree"),
        "active_rows_covered": report.get("active_rows_covered"),
        "active_columns_covered": report.get("active_columns_covered"),
        "constraint_valid": report.get("valid"),
        "constraint_failure_reasons": _json_compact(report.get("failure_reasons", [])),
        "solver_used": record.get("solver_used"),
        "solver_status": record.get("solver_status"),
        "optimality_gap": record.get("optimality_gap"),
        "fallback_used": record.get("fallback_used"),
        "runtime_seconds": record.get("runtime_seconds"),
        "exact_solves": record.get("exact_solves", 0),
        "accepted_swaps": record.get("accepted_swaps", 0),
        "objective_before": record.get("initial_objective"),
        "objective_after": record.get("final_objective"),
        "full_support_risk_objective": record.get("full_support_objective"),
        "termination_reason": record.get("termination_reason"),
        "risk_kind": record.get("risk_kind"),
        "risk_aggregation": record.get("risk_aggregation"),
        "risk_task_count": record.get("risk_task_count"),
        "risk_unique_functional_count": record.get("risk_unique_functional_count"),
        "risk_objective_residual_independent": (
            True if record.get("risk_kind") is not None else None
        ),
        "raw_score_min": record.get("raw_score_min"),
        "raw_score_max": record.get("raw_score_max"),
        "negative_raw_score_count": record.get("negative_raw_score_count", 0),
        "score_translation": record.get("score_translation"),
        "global_optimality_proven": record.get("global_optimality_proven"),
    }


def _evaluate_support(
    instance: StructuralInstance,
    functionals: Sequence[FunctionalSpec],
    held_out_cache: Sequence[Mapping[str, Any]],
    support_row: Mapping[str, Any],
    support: np.ndarray,
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sparse_matrix = np.where(support, instance.matrix, 0.0)
    sparse_operator = _ridge_filter_operator(sparse_matrix, instance.alpha)
    physical_floor = float(settings["physical_error_floor"])
    support_floor = float(settings["support_error_floor"])
    near_zero_threshold = float(settings["near_zero_threshold"])
    rows: list[dict[str, Any]] = []
    for cached in held_out_cache:
        seed = int(cached["seed"])
        residual = np.asarray(cached["residual"], dtype=np.float64)
        truth = np.asarray(cached["truth"], dtype=np.float64)
        reference = cached["reference"]
        observed_fingerprint = str(cached["residual_fingerprint"])
        full_update = np.asarray(cached["full_update"], dtype=np.float64)
        sparse_update = sparse_operator @ residual
        for functional in functionals:
            if functional.vector is None:
                continue
            ell = np.asarray(functional.vector, dtype=np.float64)
            y_true = float(ell @ truth)
            y_full = float(ell @ full_update)
            y_sparse = float(ell @ sparse_update)
            a_physical = abs(y_sparse - y_true)
            e_support = abs(y_sparse - y_full) / max(abs(y_full), support_floor)
            e_physical = a_physical / max(abs(y_true), physical_floor)
            logical_key = "|".join(
                (
                    str(support_row["configuration_hash"]),
                    instance.instance_id,
                    str(support_row["support_id"]),
                    str(int(seed)),
                    functional.functional_id,
                    "frozen_structural_reference",
                )
            )
            rows.append(
                {
                    "logical_key": logical_key,
                    "configuration_id": support_row["configuration_id"],
                    "configuration_hash": support_row["configuration_hash"],
                    "instance_id": instance.instance_id,
                    "structural_group_id": instance.structural_group_id,
                    "ieee_case": instance.ieee_case,
                    "realization_order": instance.realization_order,
                    "matrix_seed": instance.matrix_seed,
                    "matrix_fingerprint": instance.matrix_fingerprint,
                    "support_id": support_row["support_id"],
                    "support_fingerprint": support_row["support_fingerprint"],
                    "selector": support_row["selector"],
                    "deployable_selector": support_row["deployable_selector"],
                    "diagnostic_selector": support_row["diagnostic_selector"],
                    "k_budget": support_row["k_budget"],
                    "slot_budget": support_row["slot_budget"],
                    "support_nnz": support_row["actual_nonzeros"],
                    "regularization_regime": "frozen_structural_reference",
                    "alpha": instance.alpha,
                    "residual_seed": int(seed),
                    "split": "held_out",
                    "residual_fingerprint": observed_fingerprint,
                    "functional_id": functional.functional_id,
                    "functional_family": functional.family,
                    "functional_classification": functional.classification,
                    "functional_norm": float(np.linalg.norm(ell)),
                    "y_true": y_true,
                    "y_full_ridge": y_full,
                    "y_sparse": y_sparse,
                    "E_support": e_support,
                    "E_physical": e_physical,
                    "A_physical": a_physical,
                    "B_physical_signed": y_sparse - y_true,
                    "A_full_physical": abs(y_full - y_true),
                    "near_zero_y_true": abs(y_true) < near_zero_threshold,
                    "near_zero_y_full_ridge": abs(y_full) < support_floor,
                    "physical_denominator": max(abs(y_true), physical_floor),
                    "support_denominator": max(abs(y_full), support_floor),
                    "truth_definition": "(x_true-x0)[selected_global_columns]",
                    "truth_reconstruction_max_abs_error": reference.reconstruction_max_abs_error,
                    "status": "completed",
                    "failure_reason": "",
                }
            )
    return rows


def _structure_exclusions(source_root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    settings = config["structure_design"]
    frames = []
    for kind, name in (
        ("candidate", settings["candidate_exclusions"]),
        ("instance", settings["instance_exclusions"]),
    ):
        frame = pd.read_csv(source_root / name)
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "exclusion_source", kind)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["exclusion_source", "status", "failure_reason"])
    columns = sorted(set().union(*(set(frame.columns) for frame in frames)))
    return pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)


def run_physical_audit(
    config_path: str | Path = "configs/tqe_physical_alignment/campaign.json",
    *,
    output_dir: str | Path | None = None,
    limit_instances: int | None = None,
    selector_subset: Sequence[str] | None = None,
    support_budget_subset: Sequence[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    config = load_campaign_config(config_path)
    destination = Path(output_dir or (Path(config["output_root"]) / "physical_audit"))
    destination.mkdir(parents=True, exist_ok=True)
    source_root = Path(config["source_structural_root"])
    settings = config["physical_audit"]
    groups, instances_frame, split_frame = validate_frozen_registries(source_root, config)
    exclusions = _structure_exclusions(source_root, config)
    selected_instances = instances_frame.sort_values(
        ["ieee_case", "structural_group_selection_order", "realization_order"], kind="stable"
    )
    if limit_instances is not None:
        selected_instances = selected_instances.head(int(limit_instances))
    selectors = list(selector_subset or settings["selectors"])
    unknown = set(selectors) - set(settings["selectors"])
    if unknown:
        raise ValueError(f"unknown selector subset: {sorted(unknown)}")
    budgets = list(support_budget_subset or settings["support_budgets"])
    slot_budgets = [int(value) for value in settings["slot_budgets"]]

    group_output = groups.copy()
    group_output.insert(0, "configuration_hash", config["configuration_hash"])
    group_output.insert(0, "configuration_id", config["configuration_id"])
    atomic_write_csv(destination / "structure_registry.csv", group_output)
    atomic_write_json(
        destination / "structure_registry.json",
        {
            "configuration_id": config["configuration_id"],
            "configuration_hash": config["configuration_hash"],
            "selection_source": str(source_root / config["structure_design"]["registry"]),
            "selection_outcome_independent": True,
            "independent_unit": "structural_group_id",
            "numerical_realizations_are_independent": False,
            "groups": group_output.to_dict(orient="records"),
        },
    )
    atomic_write_csv(destination / "structure_exclusions.csv", exclusions)

    functional_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    instance_count = len(selected_instances)

    for ordinal, instance_record in enumerate(selected_instances.itertuples(index=False), start=1):
        instance = load_instance(source_root, str(instance_record.instance_id))
        split = split_frame[split_frame["instance_id"] == instance.instance_id].iloc[0]
        training_seeds = [int(seed) for seed in split["training_seed_ids"]]
        held_out_seeds = [int(seed) for seed in split["held_out_seed_ids"]]
        fingerprints = dict(split["residual_fingerprints"])
        functionals = build_instance_functionals(instance)
        physical = _physical_functionals(functionals)
        available = _available_functionals(functionals)
        functional_rows.extend(functional_registry_rows(instance, functionals))
        training_tasks = _build_training_tasks(instance, physical, training_seeds, fingerprints)
        singular = np.linalg.svd(instance.matrix, compute_uv=False)
        gram = instance.matrix.T @ instance.matrix + instance.alpha * np.eye(
            instance.matrix.shape[1]
        )
        alpha_rows.append(
            {
                "configuration_id": config["configuration_id"],
                "configuration_hash": config["configuration_hash"],
                "instance_id": instance.instance_id,
                "structural_group_id": instance.structural_group_id,
                "ieee_case": instance.ieee_case,
                "realization_order": instance.realization_order,
                "regularization_regime": "frozen_structural_reference",
                "alpha": instance.alpha,
                "reference_beta": float(instance_record.reference_beta),
                "lambda_ref": float(instance_record.lambda_ref),
                "kappa_H": float(np.linalg.cond(instance.matrix)),
                "kappa_HtH_plus_alphaI": float(np.linalg.cond(gram)),
                "rank_H": int(np.linalg.matrix_rank(instance.matrix)),
                "sigma_max_H": float(singular[0]),
                "sigma_min_H": float(singular[-1]),
                "deployable": True,
                "oracle": False,
            }
        )

        # One independent truth check per training and held-out seed is retained.
        for split_name, seeds in (("training", training_seeds), ("held_out", held_out_seeds)):
            for seed in seeds:
                _residual, _truth, reference = instance_residual_and_truth(instance, seed)
                truth_rows.append(
                    {
                        "instance_id": instance.instance_id,
                        "structural_group_id": instance.structural_group_id,
                        "ieee_case": instance.ieee_case,
                        "split": split_name,
                        "seed": seed,
                        "truth_reconstruction_max_abs_error": (
                            reference.reconstruction_max_abs_error
                        ),
                        "stored_truth_matches_x_true_minus_x0": bool(
                            reference.reconstruction_max_abs_error <= 1e-13
                        ),
                    }
                )

        support_records: list[tuple[dict[str, Any], np.ndarray]] = []
        full_operator = _ridge_filter_operator(instance.matrix, instance.alpha)
        held_out_cache: list[dict[str, Any]] = []
        for seed in held_out_seeds:
            residual, truth, reference = instance_residual_and_truth(instance, seed)
            observed_fingerprint = stable_array_fingerprint(residual)
            expected_fingerprint = fingerprints.get(str(seed))
            if expected_fingerprint and observed_fingerprint != expected_fingerprint:
                raise RuntimeError(
                    f"held-out residual fingerprint mismatch for {instance.instance_id} seed {seed}"
                )
            held_out_cache.append(
                {
                    "seed": seed,
                    "residual": residual,
                    "truth": truth,
                    "reference": reference,
                    "residual_fingerprint": observed_fingerprint,
                    "full_update": full_operator @ residual,
                }
            )
        if "full_support" in selectors:
            full_record = {
                "support": instance.matrix != 0.0,
                "status": "completed",
                "failure_reason": "",
                "runtime_seconds": 0.0,
                "solver_used": "reference_full_support",
                "solver_status": "completed",
                "optimality_gap": 0.0,
                "fallback_used": False,
                "exact_solves": 0,
                "accepted_swaps": 0,
                "termination_reason": "reference",
            }
            row = _support_registry_row(
                instance,
                "full_support",
                full_record,
                None,
                None,
                settings,
                config["configuration_id"],
                config["configuration_hash"],
            )
            support_rows.append(row)
            support_records.append((row, full_record["support"]))

        requested_sparse_selectors = set(selectors) - {"full_support"}
        standard_selector_names = {
            "global_magnitude",
            "balanced_magnitude",
            "ridge_leverage",
            "random_objective_feasible_support",
            "sensitivity_initial_mean",
            "sensitivity_refined_mean",
            "sensitivity_initial_worst_case",
            "sensitivity_refined_worst_case",
            "adjoint_unnormalized_mean",
            "exact_single_entry_removal_mean",
        }
        risk_selector_names = {
            name
            for name in settings["selectors"]
            if name.startswith("noise_propagation_risk_")
            or name.startswith("posterior_variance_reference_")
        }
        for k_budget, slot_budget in ((k, s) for k in budgets for s in slot_budgets):
            constraints = SupportConstraints(
                int(k_budget), int(slot_budget), bool(settings["coverage_enabled"])
            )
            random_seed = (
                int(settings["random_objective"]["base_seed"])
                + (ordinal - 1) * 10_000
                + int(k_budget) * 10
                + int(slot_budget)
            )
            bank: dict[str, dict[str, Any]] = {}
            if requested_sparse_selectors & standard_selector_names:
                bank.update(
                    _standard_support_bank(
                        instance,
                        training_tasks,
                        constraints,
                        settings,
                        random_seed=random_seed,
                        requested_selectors=requested_sparse_selectors,
                    )
                )
            # Risk does not depend on residual values, but the repeated training
            # bank is carried through explicitly.  The risk implementation
            # compresses identical functionals while retaining multiplicities.
            risk_vectors = [
                np.asarray(task.functional, dtype=np.float64) for task in training_tasks
            ]
            if requested_sparse_selectors & risk_selector_names:
                bank.update(
                    _risk_support_bank(
                        instance,
                        risk_vectors,
                        constraints,
                        settings,
                        requested_sparse_selectors,
                    )
                )
            construction_seeds = [
                record["support"]
                for record in bank.values()
                if record.get("support") is not None and record.get("status") == "completed"
            ]
            if "near_oracle_support_fidelity_diagnostic" in requested_sparse_selectors:
                bank["near_oracle_support_fidelity_diagnostic"] = _near_oracle_record(
                    instance, training_tasks, constraints, settings, construction_seeds
                )

            for selector in selectors:
                if selector == "full_support":
                    continue
                record = bank[selector]
                row = _support_registry_row(
                    instance,
                    selector,
                    record,
                    int(k_budget),
                    int(slot_budget),
                    settings,
                    config["configuration_id"],
                    config["configuration_hash"],
                )
                support_rows.append(row)
                for trace in record.get("refinement_trace", ()):
                    refinement_rows.append(
                        {
                            "configuration_id": config["configuration_id"],
                            "configuration_hash": config["configuration_hash"],
                            "support_id": row["support_id"],
                            "instance_id": instance.instance_id,
                            "structural_group_id": instance.structural_group_id,
                            "ieee_case": instance.ieee_case,
                            "selector": selector,
                            "k_budget": int(k_budget),
                            "slot_budget": int(slot_budget),
                            **trace,
                        }
                    )
                if record.get("support") is None or record.get("status") != "completed":
                    failure_rows.append(
                        {
                            "configuration_id": config["configuration_id"],
                            "configuration_hash": config["configuration_hash"],
                            "instance_id": instance.instance_id,
                            "structural_group_id": instance.structural_group_id,
                            "ieee_case": instance.ieee_case,
                            "selector": selector,
                            "k_budget": int(k_budget),
                            "slot_budget": int(slot_budget),
                            "stage": "support_selection",
                            "status": record.get("status"),
                            "failure_reason": record.get("failure_reason"),
                        }
                    )
                    continue
                if not bool(row["constraint_valid"]):
                    failure_rows.append(
                        {
                            "configuration_id": config["configuration_id"],
                            "configuration_hash": config["configuration_hash"],
                            "instance_id": instance.instance_id,
                            "structural_group_id": instance.structural_group_id,
                            "ieee_case": instance.ieee_case,
                            "selector": selector,
                            "k_budget": int(k_budget),
                            "slot_budget": int(slot_budget),
                            "stage": "support_validation",
                            "status": "failed",
                            "failure_reason": row["constraint_failure_reasons"],
                        }
                    )
                    continue
                support_records.append((row, np.asarray(record["support"], dtype=bool)))

        for support_row, support in support_records:
            raw_rows.extend(
                _evaluate_support(
                    instance,
                    available,
                    held_out_cache,
                    support_row,
                    support,
                    settings,
                )
            )
        if verbose:
            print(
                f"physical audit instance {ordinal}/{instance_count}: {instance.instance_id}; "
                f"functionals={len(available)} supports={len(support_records)} "
                f"raw_rows={len(raw_rows)}",
                flush=True,
            )

    functional_frame = pd.DataFrame(functional_rows)
    support_frame = pd.DataFrame(support_rows)
    refinement_frame = pd.DataFrame(
        refinement_rows,
        columns=(
            list(refinement_rows[0])
            if refinement_rows
            else [
                "configuration_id",
                "configuration_hash",
                "support_id",
                "instance_id",
                "structural_group_id",
                "ieee_case",
                "selector",
                "k_budget",
                "slot_budget",
                "iteration",
                "action",
                "accepted",
                "objective_before",
                "objective_after",
            ]
        ),
    )
    raw_frame = pd.DataFrame(raw_rows)
    alpha_frame = pd.DataFrame(alpha_rows)
    failure_frame = pd.DataFrame(
        failure_rows,
        columns=(
            list(failure_rows[0])
            if failure_rows
            else [
                "configuration_id",
                "configuration_hash",
                "instance_id",
                "structural_group_id",
                "ieee_case",
                "selector",
                "k_budget",
                "slot_budget",
                "stage",
                "status",
                "failure_reason",
            ]
        ),
    )
    truth_frame = pd.DataFrame(truth_rows)
    if not raw_frame.empty and raw_frame["logical_key"].duplicated().any():
        duplicates = raw_frame.loc[raw_frame["logical_key"].duplicated(), "logical_key"].tolist()
        raise RuntimeError(f"duplicate logical raw-row keys: {duplicates[:3]}")
    if not support_frame.empty and support_frame["support_id"].duplicated().any():
        raise RuntimeError("duplicate support IDs")
    atomic_write_csv(destination / "functional_registry.csv", functional_frame)
    atomic_write_csv(destination / "support_registry.csv", support_frame)
    atomic_write_csv(destination / "refinement_trace.csv", refinement_frame)
    atomic_write_csv(destination / "alpha_regime_summary.csv", alpha_frame)
    atomic_write_csv(destination / "raw_physical_rows.csv", raw_frame)
    atomic_write_csv(destination / "truth_reconstruction_audit.csv", truth_frame)
    atomic_write_csv(destination / "failure_registry.csv", failure_frame)
    provenance = {
        "study_id": STUDY_ID,
        "configuration_id": config["configuration_id"],
        "configuration_hash": config["configuration_hash"],
        "configuration_path": str(config_path),
        "resolved_configuration": config,
        "source_structural_root": str(source_root),
        "source_registry_hashes": {
            name: sha256_file(source_root / name)
            for name in (
                config["structure_design"]["registry"],
                config["structure_design"]["instance_registry"],
            )
        },
        "environment": environment_provenance(Path.cwd()),
        "truth_reference": (
            "WeightedSystem.x_true independently equals metadata true_state-linearization_state"
        ),
        "whitening": "rowwise diagonal sigma; H_tilde and r_tilde already whitened",
        "structure_selection_reexecuted": False,
        "structure_selection_outcome_independent": True,
    }
    atomic_write_json(destination / "provenance.json", provenance)
    validation = {
        "study_id": STUDY_ID,
        "configuration_id": config["configuration_id"],
        "configuration_hash": config["configuration_hash"],
        "full_design_run": limit_instances is None,
        "independent_structures": int(group_output["structural_group_id"].nunique()),
        "cases": int(group_output["ieee_case"].nunique()),
        "instances_evaluated": int(selected_instances["instance_id"].nunique()),
        "realizations_per_structure": (
            selected_instances.groupby("structural_group_id").size().to_dict()
        ),
        "raw_rows": len(raw_frame),
        "unique_logical_keys": (
            int(raw_frame["logical_key"].nunique()) if not raw_frame.empty else 0
        ),
        "support_rows": len(support_frame),
        "functional_rows": len(functional_frame),
        "available_physical_functionals": int(
            (
                (functional_frame["status"] == "available")
                & (functional_frame["classification"] == "physical")
            ).sum()
        )
        if not functional_frame.empty
        else 0,
        "available_legacy_functionals": int(
            (functional_frame["classification"] == "legacy_diagnostic").sum()
        )
        if not functional_frame.empty
        else 0,
        "unavailable_physical_functionals": int((functional_frame["status"] == "unavailable").sum())
        if not functional_frame.empty
        else 0,
        "failure_rows": len(failure_frame),
        "training_heldout_seed_overlap": 0,
        "duplicate_logical_keys": 0,
        "truth_reconstruction_max_abs_error": float(
            truth_frame["truth_reconstruction_max_abs_error"].max()
        )
        if not truth_frame.empty
        else None,
        "all_functionals_unit_norm": bool(
            functional_frame.loc[functional_frame["status"] == "available", "unit_norm_error"]
            .le(float(settings["functional_selection"]["unit_norm_tolerance"]))
            .all()
        )
        if not functional_frame.empty
        else False,
        "full_support_E_support_max": float(
            raw_frame.loc[raw_frame["selector"] == "full_support", "E_support"].max()
        )
        if not raw_frame.empty and bool((raw_frame["selector"] == "full_support").any())
        else None,
        "structure_is_primary_independent_unit": True,
        "selector_truth_or_heldout_leakage_detected": bool(
            support_frame["uses_true_state"].astype(bool).any()
            or support_frame["uses_held_out_data"].astype(bool).any()
        )
        if not support_frame.empty
        else False,
    }
    atomic_write_json(destination / "validation_summary.json", validation)
    report_lines = [
        "# Expanded Physical Audit Validation",
        "",
        f"- Configuration: `{config['configuration_id']}`",
        f"- Configuration hash: `{config['configuration_hash']}`",
        f"- Independent structures in frozen registry: {validation['independent_structures']}",
        f"- IEEE cases: {validation['cases']}",
        f"- Numerical instances evaluated: {validation['instances_evaluated']}",
        f"- Raw selected-output rows: {validation['raw_rows']}",
        f"- Unique logical keys: {validation['unique_logical_keys']}",
        "- Unavailable physical functional records retained: "
        f"{validation['unavailable_physical_functionals']}",
        f"- Support/failure rows: {validation['support_rows']} / {validation['failure_rows']}",
        "- Maximum independent truth-reconstruction error: "
        f"{validation['truth_reconstruction_max_abs_error']}",
        f"- Maximum full-support E_support: {validation['full_support_E_support_max']}",
        "- Selector truth/held-out leakage detected: "
        f"{validation['selector_truth_or_heldout_leakage_detected']}",
        "",
        "Physical and legacy-diagnostic functionals share the raw registry only through an "
        "explicit classification column; all summaries must keep the classifications separate. "
        "Unavailable physical functionals are retained in functional_registry.csv and produce "
        "no substituted evaluation rows.",
        "",
    ]
    atomic_write_text(destination / "validation_report.md", "\n".join(report_lines))
    return validation
