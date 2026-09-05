"""Track A - cross-case validation on one IEEE-30-derived 8x8 structure.

Runs the identical output-aware protocol - physical functionals, full selector suite, joint
application-utility / QSVT-feasibility grid, and resource-accuracy thresholds - on the
deterministic IEEE-30 8x8 block, reusing every frozen primitive.  Nothing here is tuned to
IEEE-30; the block, functionals, grids, and thresholds are fixed by the frozen protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import (
    build_case_design,
    build_case_full_system,
)
from robust_qsvt_se.cross_case_validation.inventory import (
    write_block_inventory,
    write_functional_inventory,
)
from robust_qsvt_se.cross_case_validation.selector_comparison import run_selector_comparison
from robust_qsvt_se.cross_case_validation.selectors import select_support_generalized
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    support_constraint_report,
)
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    array_fingerprint,
    atomic_write_csv,
    atomic_write_json,
    load_config,
    provenance_block,
    write_manifest_and_checksums,
)
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import ExactLossEvaluator
from robust_qsvt_se.reviewer_blocking.joint_feasibility import (
    _alpha_grid,
    _rmse,
    application_metrics,
    classify,
    evaluate_qsvt_feasibility,
    regularization_selection,
)
from robust_qsvt_se.reviewer_blocking.resource_pareto import (
    _best_class_at_threshold,
    _fixed_error_costs,
    _pareto_fronts,
    _resource_status_ledger,
    build_common_qsvt_design,
    cost_model,
    executed_support_resources,
)

STUDY_ID = "cross_case_validation_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/cross_case_larger_block_validation/cross_case")
DEFAULT_CONFIG_PATH = Path("configs/cross_case_larger_block_validation/cross_case.json")


# --------------------------------------------------------------- joint feasibility


def run_joint_feasibility_track(
    case_name: str, design, full_system, config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    from robust_qsvt_se.cross_case_validation.common import build_case_tasks

    jf = config["joint_feasibility"]
    cache_dir = destination / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    small = design.small
    y_floor = float(config.get("y_floor", 1e-6))
    useful_ratio = float(jf["useful_rmse_ratio_threshold"])
    sparsification_threshold = float(jf.get("sparsification_error_threshold", 0.1))
    uniform_tol = float(jf["uniform_approximation_tolerance"])
    bound_tol = float(jf["bound_tolerance"])
    margin = float(jf["target_margin"])
    action_tol = float(jf["action_error_tolerance"])
    degrees = [int(d) for d in jf["degrees"]]
    selectors = list(jf["joint_selectors"])
    training_seeds = [int(s) for s in config["training_seed_ids"]]
    execute_statevector = bool(config.get("execute_statevector", True))

    alpha_grid = _alpha_grid(full_system, small.alpha, {**jf})
    reg_records = regularization_selection(full_system, alpha_grid)
    atomic_write_csv(
        destination / "regularization_selection_records.csv", pd.DataFrame(reg_records)
    )
    rmses = np.asarray([
        _rmse(_ridge(full_system, a), full_system.x_true) for a in alpha_grid
    ])
    oracle_best_rmse = float(rmses.min())
    unregularized_rmse = float(rmses[np.argmin(alpha_grid)])
    selected_alphas = {rec["selected_alpha"] for rec in reg_records}
    anchor_alphas = set(float(a) for a in np.quantile(alpha_grid, [0.0, 0.25, 0.5, 0.75, 1.0]))
    joint_alphas = sorted(selected_alphas | anchor_alphas | {float(small.alpha)})

    score_ids = design.physical_functional_ids
    training_tasks = build_case_tasks(case_name, small, training_seeds, "training", score_ids)
    residual_unit = None
    if training_tasks:
        base = training_tasks[0].residual
        residual_unit = base / np.linalg.norm(base)

    joint_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for selector in selectors:
        for k in jf["joint_support_budgets"]:
            for s in jf["joint_slot_budgets"]:
                constraints = SupportConstraints(int(k), int(s), True)
                support = select_support_generalized(
                    small, selector, constraints, training_tasks, score_ids
                )
                feasible = support is not None and support_constraint_report(
                    small.matrix, support, constraints
                )["valid"]
                if not feasible:
                    failure_rows.append({
                        "stage": "support_selection", "selector": selector,
                        "k_budget": int(k), "slot_budget": int(s),
                        "reason": "support_infeasible_or_none",
                    })
                    continue
                sparse_block = np.where(support, small.matrix, 0.0)
                support_fp = array_fingerprint(support.astype(np.float64))
                for alpha in joint_alphas:
                    app = application_metrics(
                        full_system, alpha, oracle_best_rmse=oracle_best_rmse,
                        unregularized_rmse=unregularized_rmse,
                    )
                    application_useful = bool(app["rmse_ratio_to_oracle_best"] <= useful_ratio)
                    evaluator = ExactLossEvaluator(
                        small.matrix, training_tasks, float(alpha), y_floor
                    )
                    selected_error = float(np.mean(evaluator.normalized(support)))
                    sparsification_acceptable = bool(selected_error <= sparsification_threshold)
                    for degree in degrees:
                        feas = evaluate_qsvt_feasibility(
                            sparse_block, float(alpha), degree, margin=margin,
                            bound_tolerance=bound_tol, uniform_tolerance=uniform_tol,
                            execute_statevector=execute_statevector, residual_unit=residual_unit,
                            cache_dir=cache_dir,
                        )
                        qsvt_feasible = bool(
                            feas.boundedness_ok
                            and feas.phase_synthesis_status == "synthesized"
                            and feas.uniform_fit_error <= uniform_tol
                            and (
                                np.isnan(feas.statevector_action_error)
                                or feas.statevector_action_error <= action_tol
                            )
                        )
                        region = classify(application_useful, qsvt_feasible)
                        operating_point_viable = bool(qsvt_feasible and sparsification_acceptable)
                        joint_rows.append({
                            "case": case_name, "selector": selector, "k_budget": int(k),
                            "slot_budget": int(s), "support_fingerprint": support_fp,
                            "alpha": float(alpha), "degree": int(degree),
                            "beta": feas.beta, "normalized_lambda": feas.normalized_lambda,
                            "contraction_c": feas.contraction_c, "phase_count": feas.phase_count,
                            "uniform_fit_error": feas.uniform_fit_error,
                            "singular_point_fit_error": feas.singular_point_fit_error,
                            "bounded_max_abs": feas.bounded_max_abs,
                            "boundedness_ok": feas.boundedness_ok,
                            "phase_synthesis_status": feas.phase_synthesis_status,
                            "postselection_probability": feas.postselection_probability,
                            "statevector_action_error": feas.statevector_action_error,
                            "qsvt_evidence_status": feas.evidence_status,
                            "qsvt_failure_reason": feas.failure_reason,
                            "selected_output_normalized_error": selected_error,
                            "sparsification_acceptable": sparsification_acceptable,
                            "full_state_rmse": app["full_state_rmse"],
                            "angle_rmse": app["angle_rmse"], "voltage_rmse": app["voltage_rmse"],
                            "rmse_ratio_to_oracle_best": app["rmse_ratio_to_oracle_best"],
                            "beats_unregularized": app["beats_unregularized"],
                            "application_useful_full_state": application_useful,
                            "qsvt_feasible": qsvt_feasible,
                            "selected_output_operating_point_viable": operating_point_viable,
                            "region": region,
                        })
                        if feas.failure_reason:
                            failure_rows.append({
                                "stage": "qsvt_feasibility", "selector": selector,
                                "k_budget": int(k), "slot_budget": int(s),
                                "alpha": float(alpha), "degree": int(degree),
                                "reason": feas.failure_reason,
                            })

    joint_frame = pd.DataFrame(joint_rows)
    atomic_write_csv(destination / "joint_feasibility_grid.csv", joint_frame)
    region_counts = (
        joint_frame["region"].value_counts().to_dict() if not joint_frame.empty else {}
    )
    quadrants = {
        "N_useful_and_feasible": int(region_counts.get("application_useful_qsvt_feasible", 0)),
        "N_useful_and_infeasible": int(region_counts.get("application_useful_qsvt_infeasible", 0)),
        "N_not_useful_and_feasible": int(
            region_counts.get("application_not_useful_qsvt_feasible", 0)
        ),
        "N_neither": int(region_counts.get("neither_useful_nor_qsvt_feasible", 0)),
    }
    summary_frame = pd.DataFrame([{
        "case": case_name, "joint_rows": len(joint_frame),
        "executed_statevector_rows": int(
            (joint_frame["qsvt_evidence_status"] == "executed_statevector").sum()
        ) if not joint_frame.empty else 0,
        "selected_output_operating_points": int(
            joint_frame["selected_output_operating_point_viable"].sum()
        ) if not joint_frame.empty else 0,
        **quadrants,
    }])
    atomic_write_csv(destination / "joint_feasibility_summary.csv", summary_frame)
    atomic_write_csv(
        destination / "joint_failure_registry.csv",
        pd.DataFrame(failure_rows) if failure_rows
        else pd.DataFrame(columns=["stage", "selector", "reason"]),
    )
    atomic_write_csv(
        destination / "figure_data" / "application_rmse_vs_alpha.csv",
        pd.DataFrame({"alpha": alpha_grid, "full_state_rmse": rmses}),
    )
    return {"joint_rows": len(joint_frame), "quadrants": quadrants,
            "region_counts": region_counts, "reg_records": reg_records,
            "best_feasible_selected_error": (
                float(joint_frame[joint_frame["qsvt_feasible"]]["selected_output_normalized_error"].min())
                if (not joint_frame.empty and joint_frame["qsvt_feasible"].any()) else float("nan")
            ),
            "max_feasible_rmse_ratio": (
                float(joint_frame[joint_frame["qsvt_feasible"]]["rmse_ratio_to_oracle_best"].max())
                if (not joint_frame.empty and joint_frame["qsvt_feasible"].any()) else float("nan")
            )}


def _ridge(system, alpha: float) -> np.ndarray:
    from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution

    return ridge_svd_solution(system.matrix, system.residual, alpha=float(alpha))


# --------------------------------------------------------------- resource pareto


OUTPUT_AWARE_SELECTORS = ("sensitivity_initial_mean", "sensitivity_refined_mean")


def run_resource_pareto_track(
    case_name: str, design, config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    from robust_qsvt_se.cross_case_validation.common import build_case_tasks
    from robust_qsvt_se.reviewer_blocking.resource_pareto import _accuracy

    rp = config["resource_pareto"]
    (destination / "figure_data").mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    small = design.small
    degree = int(rp["degree"])
    shots = int(rp["modeled_shots"])
    basis_gates = list(rp["basis_gates"])
    optimization_level = int(rp["optimization_level"])
    thresholds = [float(t) for t in rp["error_thresholds"]]
    selectors = list(rp["resource_selectors"])
    training_seeds = [int(s) for s in config["training_seed_ids"]]
    heldout_seeds = [int(s) for s in config["held_out_seed_ids"]]
    score_ids = design.physical_functional_ids

    training_tasks = build_case_tasks(case_name, small, training_seeds, "training", score_ids)
    heldout_tasks = build_case_tasks(case_name, small, heldout_seeds, "held_out", score_ids)
    residual_unit = training_tasks[0].residual / np.linalg.norm(training_tasks[0].residual)

    common_designs: dict[int, Any] = {}
    for s in rp["resource_slot_budgets"]:
        common_designs[int(s)] = build_common_qsvt_design(
            small.matrix, small.alpha, int(s), degree, cache_dir
        )

    rows: list[dict[str, Any]] = []
    for selector in selectors:
        selector_class = "output_aware" if selector in OUTPUT_AWARE_SELECTORS else "output_agnostic"
        for k in rp["resource_support_budgets"]:
            for s in rp["resource_slot_budgets"]:
                constraints = SupportConstraints(int(k), int(s), True)
                support = select_support_generalized(
                    small, selector, constraints, training_tasks, score_ids
                )
                base = {"case": case_name, "selector": selector, "selector_class": selector_class,
                        "k_budget": int(k), "slot_budget": int(s), "degree": degree}
                feasible = support is not None and support_constraint_report(
                    small.matrix, support, constraints
                )["valid"]
                if not feasible:
                    rows.append(
                        {**base, "status": "failed", "failure_reason": "support_infeasible"}
                    )
                    continue
                common = common_designs[int(s)]
                from robust_qsvt_se.qsvt.bipartite_slot_assignment import minimum_slot_count
                accuracy = _accuracy(small, support, heldout_tasks)
                record = {**base, "status": "completed", "failure_reason": "",
                          "actual_nonzeros": int(support.sum()),
                          "slot_count": int(
                              minimum_slot_count(np.where(support, small.matrix, 0.0).T != 0.0)
                          ),
                          "support_fingerprint": array_fingerprint(support.astype(np.float64)),
                          "common_beta": common.beta, "normalized_lambda": common.normalized_lambda,
                          "qsvt_bounded_at_degree": common.bounded, **accuracy}
                if not common.bounded:
                    record["status"] = "qsvt_infeasible_at_common_degree"
                    record["failure_reason"] = "common_polynomial_unbounded"
                    rows.append(record)
                    continue
                executed = executed_support_resources(
                    support, small.matrix, common, residual_unit,
                    basis_gates=basis_gates, optimization_level=optimization_level,
                )
                costs = cost_model(executed, dimension=small.dimension, degree=degree, shots=shots)
                rows.append({**record, **executed, **costs})

    frame = pd.DataFrame(rows)
    atomic_write_csv(destination / "raw_resource_accuracy.csv", frame)
    completed = frame[frame["status"] == "completed"].copy() if not frame.empty else frame
    pareto_frame = _pareto_fronts(completed)
    atomic_write_csv(destination / "resource_pareto_fronts.csv", pareto_frame)
    fixed_error = _fixed_error_costs(completed, thresholds)
    threshold_summary = _threshold_cost_summary(fixed_error, thresholds, case_name)
    atomic_write_csv(destination / "threshold_cost_summary.csv", threshold_summary)
    atomic_write_csv(destination / "resource_status_ledger.csv", _resource_status_ledger())
    if not completed.empty:
        atomic_write_csv(
            destination / "figure_data" / "resource_error_vs_cost.csv",
            completed[[
                "selector", "selector_class", "k_budget", "slot_budget", "actual_nonzeros",
                "slot_count", "mean_heldout_normalized_error", "c_total_gates",
                "executed_c_signal_gates", "postselection_probability",
            ]],
        )
    return {"cells": len(frame), "completed": len(completed),
            "retained_failures": (
                int((frame["status"] != "completed").sum()) if not frame.empty else 0
            ),
            "threshold_summary": threshold_summary}


def _threshold_cost_summary(
    fixed_error: pd.DataFrame, thresholds: list[float], case_name: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        aware = _best_class_at_threshold(fixed_error, threshold, "output_aware")
        agnostic = _best_class_at_threshold(fixed_error, threshold, "output_agnostic")
        aware_min = aware["min_c_total_gates"]
        agnostic_min = agnostic["min_c_total_gates"]
        if not aware["feasible"] and not agnostic["feasible"]:
            verdict = "both_infeasible"
        elif aware["feasible"] and not agnostic["feasible"]:
            verdict = "output_aware_only"
        elif agnostic["feasible"] and not aware["feasible"]:
            verdict = "output_agnostic_only"
        elif aware_min < agnostic_min:
            verdict = "output_aware_cheaper"
        elif np.isclose(aware_min, agnostic_min, rtol=1e-12, atol=1e-12):
            verdict = "tied"
        else:
            verdict = "output_aware_more_expensive"
        rows.append({
            "case": case_name, "error_threshold": threshold,
            "output_aware_feasible": bool(aware["feasible"]),
            "output_aware_selector": aware["selector"],
            "output_aware_min_c_total": aware_min, "output_agnostic_min_c_total": agnostic_min,
            "output_agnostic_feasible": bool(agnostic["feasible"]),
            "output_agnostic_selector": agnostic["selector"],
            "ratio_agnostic_over_aware": (
                float(agnostic_min / aware_min)
                if np.isfinite(aware_min) and aware_min > 0 and np.isfinite(agnostic_min)
                else np.nan
            ),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- orchestrator


def run_cross_case(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"unexpected study_id: {config.get('study_id')}")
    destination = Path(output_dir)
    (destination / "figure_data").mkdir(parents=True, exist_ok=True)
    (destination / "figures").mkdir(parents=True, exist_ok=True)

    case_name = str(config["case_name"])
    seed = int(config.get("matrix_seed", 123))
    dimension = int(config.get("dimension", 8))
    design = build_case_design(case_name, seed, dimension=dimension)

    block_payload = write_block_inventory(case_name, design, destination, STUDY_ID)
    inventory = write_functional_inventory(design, destination, STUDY_ID)

    selector_result = run_selector_comparison(
        case_name, design, output_dir=destination,
        support_budgets=[int(k) for k in config["support_budgets"]],
        slot_budgets=[int(s) for s in config["slot_budgets"]],
        training_seeds=[int(s) for s in config["training_seed_ids"]],
        held_out_seeds=[int(s) for s in config["held_out_seed_ids"]],
        selectors_subset=None,
        beam_width=int(config.get("near_oracle_beam_width", 6)),
        near_oracle_max_loss_evals=int(config.get("near_oracle_max_loss_evals", 200000)),
        random_seed_base=int(config.get("random_seed_base", 314159)),
        y_floor=float(config.get("y_floor", 1e-6)),
        failure_threshold=float(config.get("failure_threshold", 0.1)),
    )

    full_system = build_case_full_system(case_name, seed)
    joint_result = run_joint_feasibility_track(case_name, design, full_system, config, destination)
    resource_result = run_resource_pareto_track(case_name, design, config, destination)

    _write_cross_case_report(
        destination, case_name, design, block_payload, inventory,
        selector_result, joint_result, resource_result,
    )
    atomic_write_json(
        destination / "provenance.json",
        provenance_block(config_path, config) | {"study_id": STUDY_ID},
    )
    write_manifest_and_checksums(
        destination, study_id=STUDY_ID,
        extra={
            "case": case_name, "block_shape": [dimension, dimension],
            "physical_functionals": len(design.physical_functional_ids),
            "unavailable_functionals": [u.requested_functional_id for u in design.unavailable],
            "quadrants": joint_result["quadrants"],
        },
    )
    return {
        "case": case_name,
        "selector_records": selector_result["support_records"],
        "feasible_supports": selector_result["feasible_supports"],
        "joint_quadrants": joint_result["quadrants"],
        "resource_cells": resource_result["cells"],
    }


def _write_cross_case_report(
    destination, case_name, design, block_payload, inventory,
    selector_result, joint_result, resource_result,
) -> None:
    summary = selector_result["summary_frame"]
    mean_summary = summary[summary["objective"] == "mean"] if not summary.empty else summary
    cond = design.conditioning
    families = sorted({
        rec.family for rec in design.functional_records if rec.family != "legacy_predetermined"
    })
    lines = [
        f"# Cross-Case Validation Report - {case_name.upper()} 8x8",
        "", CLAIM_BOUNDARY, "",
        "## Block (identical deterministic policy, seed 123)", "",
        f"- selected global rows: `{block_payload['selected_global_rows']}`",
        f"- selected global columns: `{block_payload['selected_global_columns']}`",
        f"- rank {cond['rank']}/{cond['cols']}, raw kappa `{cond['raw_condition_number']:.4g}`",
        "- regularized normal-system kappa "
        f"`{cond['regularized_normal_system_condition_number']:.4g}`",
        f"- nonzeros {cond['nonzeros']} (density {cond['density']:.3f})",
        f"- block alpha = 4*sigma_min_pos^2 = `{design.small.alpha:.6g}`",
        "",
        "## Physical functionals", "",
        f"- representable physical functionals: {len(design.physical_functional_ids)} "
        f"({', '.join(families)})",
        f"- unavailable (recorded, never substituted): "
        f"`{[u.requested_functional_id for u in design.unavailable]}`",
        "",
        "## Selector held-out comparison (mean objective, averaged over budget grid)", "",
        "| selector | feasibility | mean held-out err | median | mean overlap w/ near-oracle |",
        "|---|---:|---:|---:|---:|",
    ]
    if not mean_summary.empty:
        for _, row in mean_summary.sort_values("mean_heldout_normalized_error").iterrows():
            lines.append(
                f"| `{row['selector']}` | {row['feasibility_rate']:.2f} | "
                f"{row['mean_heldout_normalized_error']:.4g} | "
                f"{row['median_heldout_normalized_error']:.4g} | "
                f"{row['mean_overlap_with_near_oracle']:.3f} |"
            )
    q = joint_result["quadrants"]
    lines += [
        "",
        "## Joint application-utility x QSVT-feasibility (four quadrants)", "",
        f"- N(useful & feasible) = **{q['N_useful_and_feasible']}**",
        f"- N(useful & infeasible) = {q['N_useful_and_infeasible']}",
        f"- N(not useful & feasible) = {q['N_not_useful_and_feasible']}",
        f"- N(neither) = {q['N_neither']}",
        f"- best selected-output error inside QSVT-feasible band: "
        f"`{joint_result['best_feasible_selected_error']:.4g}`; "
        f"max full-state RMSE ratio there: `{joint_result['max_feasible_rmse_ratio']:.4g}`",
        "",
        "Grid cells are (alpha, degree, k, s) operating points over ONE IEEE-30 structure - "
        "not independent power systems.",
        "",
        "## Resource-accuracy at predeclared thresholds", "",
        "| threshold | output-aware min C_total | output-agnostic min C_total | verdict |",
        "|---:|---:|---:|---|",
    ]
    for _, row in resource_result["threshold_summary"].iterrows():
        lines.append(
            f"| {row['error_threshold']:g} | {row['output_aware_min_c_total']:.4g} | "
            f"{row['output_agnostic_min_c_total']:.4g} | {row['verdict']} |"
        )
    lines += [
        "",
        "Executed C_signal / p_post + modeled loader/readout; not a hardware speedup claim.",
        "",
    ]
    (destination / "cross_case_report.md").write_text("\n".join(lines), encoding="utf-8")
