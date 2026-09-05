"""Track B - larger-block (16x16) validation on the IEEE-14 structure.

Isolates the effect of block size against the frozen IEEE-14 8x8 result: same case, same
deterministic policy, same selectors, physical functionals on the 16x16 block.  Adds runtime /
memory scaling, support stability, QSVT matrix-action validation (exact + statevector where the
wrapper is tractable), and modeled resource estimates with explicit evidence status.  A full
16x16 transpiled QSVT circuit is not required.
"""

from __future__ import annotations

import resource as _resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import (
    build_case_design,
    build_case_tasks,
    json_safe_floats,
)
from robust_qsvt_se.cross_case_validation.inventory import (
    write_block_inventory,
    write_functional_inventory,
)
from robust_qsvt_se.cross_case_validation.selector_comparison import run_selector_comparison
from robust_qsvt_se.cross_case_validation.selectors import select_support_generalized
from robust_qsvt_se.qsvt.bipartite_slot_assignment import minimum_slot_count
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    support_constraint_report,
)
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    atomic_write_csv,
    atomic_write_json,
    load_config,
    provenance_block,
    write_manifest_and_checksums,
)
from robust_qsvt_se.reviewer_blocking.joint_feasibility import evaluate_qsvt_feasibility
from robust_qsvt_se.reviewer_blocking.resource_pareto import (
    build_common_qsvt_design,
    cost_model,
    executed_support_resources,
)

STUDY_ID = "larger_block_validation_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/cross_case_larger_block_validation/larger_block_16x16")
DEFAULT_CONFIG_PATH = Path("configs/cross_case_larger_block_validation/larger_block_16x16.json")

OUTPUT_AWARE_SELECTORS = ("sensitivity_initial_mean", "sensitivity_refined_mean")


def _peak_rss_mb() -> float:
    usage = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return float(usage) / (1024 * 1024) if sys.platform == "darwin" else float(usage) / 1024


# --------------------------------------------------------------- runtime scaling


def _runtime_scaling(
    support_frame: pd.DataFrame, case_name: str, nnz: int, dimension: int
) -> pd.DataFrame:
    if support_frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for selector, group in support_frame.groupby("selector", sort=True):
        feasible = group[group["feasible"]]
        rows.append({
            "case": case_name, "block_dimension": dimension, "candidate_nonzeros": nnz,
            "selector": selector,
            "budgets_evaluated": len(group),
            "feasible_supports": int(group["feasible"].sum()),
            "total_selection_runtime_seconds": float(group["selection_runtime_seconds"].sum()),
            "mean_selection_runtime_seconds": float(group["selection_runtime_seconds"].mean()),
            "total_milp_solves": int(group["milp_solves"].sum()),
            "mean_exact_ridge_solves_per_cell": float(
                group["exact_ridge_solves_this_cell"].mean()
            ),
            "mean_heldout_normalized_error": float(feasible["heldout_mean_normalized_error"].mean())
            if not feasible.empty else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- QSVT + resources


def _representative_constraints(
    support_budgets: list[int], validation_slot: int
) -> list[SupportConstraints]:
    return [SupportConstraints(int(k), int(validation_slot), True) for k in support_budgets]


def run_qsvt_and_resources(
    case_name: str, design, config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    small = design.small
    score_ids = design.physical_functional_ids
    training_seeds = [int(s) for s in config["training_seed_ids"]]
    heldout_seeds = [int(s) for s in config["held_out_seed_ids"]]
    cache_dir = destination / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (destination / "figure_data").mkdir(parents=True, exist_ok=True)

    qv = config["qsvt_validation"]
    degree = int(qv["degree"])
    validation_slot = int(qv.get("validation_slot_budget", 3))
    statevector_max_dim = int(config.get("statevector_max_wrapper_dim", 4096))
    selectors = list(qv["qsvt_selectors"])
    budgets = [int(k) for k in qv["qsvt_support_budgets"]]

    training_tasks = build_case_tasks(case_name, small, training_seeds, "training", score_ids)
    heldout_tasks = build_case_tasks(case_name, small, heldout_seeds, "held_out", score_ids)
    residual_unit = training_tasks[0].residual / np.linalg.norm(training_tasks[0].residual)

    from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import evaluate_support

    qsvt_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    common_cache: dict[int, Any] = {}

    for selector in selectors:
        selector_class = "output_aware" if selector in OUTPUT_AWARE_SELECTORS else "output_agnostic"
        for constraints in _representative_constraints(budgets, validation_slot):
            support = select_support_generalized(
                small, selector, constraints, training_tasks, score_ids
            )
            base = {"case": case_name, "selector": selector, "selector_class": selector_class,
                    "k_budget": constraints.k_budget, "slot_budget": constraints.slot_budget,
                    "degree": degree}
            feasible = support is not None and support_constraint_report(
                small.matrix, support, constraints
            )["valid"]
            if not feasible:
                qsvt_rows.append({**base, "status": "support_infeasible",
                                  "evidence_status": "infeasible"})
                resource_rows.append({**base, "status": "support_infeasible",
                                      "evidence_status": "infeasible"})
                continue
            sparse = np.where(support, small.matrix, 0.0)
            slots = int(minimum_slot_count(sparse.T != 0.0))
            wrapper_slots = max(slots, constraints.slot_budget)
            mu = float(np.max(np.abs(sparse)))
            wrapper_dim = 0
            # QSVT feasibility: exact bounded-poly fit + phase synthesis + statevector action.
            execute_sv = True
            try:
                wrapper_dim = _wrapper_dim_estimate(sparse, wrapper_slots, mu)
                execute_sv = wrapper_dim <= statevector_max_dim
            except Exception:
                execute_sv = False
            feas = evaluate_qsvt_feasibility(
                sparse, float(small.alpha), degree, margin=1.05,
                bound_tolerance=2e-3, uniform_tolerance=2e-3,
                execute_statevector=execute_sv, residual_unit=residual_unit,
                cache_dir=cache_dir,
            )
            evidence = (
                "executed_statevector" if feas.evidence_status == "executed_statevector"
                else ("exact_matrix_action" if feas.phase_synthesis_status == "synthesized"
                      else "modeled_analytic_fit")
            )
            qsvt_rows.append({
                **base, "status": "evaluated",
                "actual_nonzeros": int(support.sum()), "min_slot_count": slots,
                "wrapper_slots": wrapper_slots, "wrapper_unitary_dim": wrapper_dim,
                "beta": feas.beta, "normalized_lambda": feas.normalized_lambda,
                "bounded_max_abs": feas.bounded_max_abs, "boundedness_ok": feas.boundedness_ok,
                "uniform_fit_error": feas.uniform_fit_error,
                "singular_point_fit_error": feas.singular_point_fit_error,
                "phase_synthesis_status": feas.phase_synthesis_status,
                "phase_count": feas.phase_count,
                "statevector_action_error": feas.statevector_action_error,
                "postselection_probability": feas.postselection_probability,
                "qsvt_failure_reason": feas.failure_reason,
                "statevector_executed": bool(execute_sv),
                "evidence_status": evidence,
            })
            # Resource estimate: common QSVT design (per slot) + executed signal + cost model.
            if wrapper_slots not in common_cache:
                common_cache[wrapper_slots] = build_common_qsvt_design(
                    small.matrix, small.alpha, wrapper_slots, degree, cache_dir
                )
            common = common_cache[wrapper_slots]
            accuracy = evaluate_support(small, support, heldout_tasks)
            heldout_err = float(np.mean([r["normalized_error"] for r in accuracy]))
            if not common.bounded:
                resource_rows.append({
                    **base, "status": "qsvt_infeasible_at_common_degree",
                    "actual_nonzeros": int(support.sum()),
                    "mean_heldout_normalized_error": heldout_err,
                    "evidence_status": "modeled_common_polynomial_unbounded",
                })
                continue
            if not execute_sv:
                resource_rows.append({
                    **base, "status": "statevector_skipped_ceiling",
                    "actual_nonzeros": int(support.sum()),
                    "wrapper_unitary_dim": wrapper_dim,
                    "mean_heldout_normalized_error": heldout_err,
                    "evidence_status": "skipped_statevector_ceiling",
                })
                continue
            executed = executed_support_resources(
                support, small.matrix, common, residual_unit,
                basis_gates=list(qv.get("basis_gates", ["u3", "cx"])),
                optimization_level=int(qv.get("optimization_level", 1)),
            )
            costs = cost_model(executed, dimension=small.dimension, degree=degree,
                               shots=int(qv.get("modeled_shots", 100000)))
            resource_rows.append({
                **base, "status": "completed",
                "actual_nonzeros": int(support.sum()),
                "slot_count": int(minimum_slot_count(sparse.T != 0.0)),
                "mean_heldout_normalized_error": heldout_err,
                "evidence_status": "executed_signal_plus_modeled_loader_readout",
                **executed, **costs,
            })

    qsvt_frame = pd.DataFrame(qsvt_rows)
    atomic_write_csv(destination / "qsvt_validation.csv", qsvt_frame)
    resource_frame = pd.DataFrame(resource_rows)
    atomic_write_csv(destination / "resource_estimates.csv", resource_frame)
    if not resource_frame.empty and "c_total_gates" in resource_frame:
        completed = resource_frame[resource_frame["status"] == "completed"]
        if not completed.empty:
            atomic_write_csv(
                destination / "figure_data" / "resource_error_vs_cost_16x16.csv",
                completed[["selector", "selector_class", "k_budget", "actual_nonzeros",
                           "mean_heldout_normalized_error", "c_total_gates",
                           "executed_c_signal_gates", "postselection_probability"]],
            )
    return {"qsvt_rows": len(qsvt_frame), "resource_rows": len(resource_frame),
            "qsvt_frame": qsvt_frame, "resource_frame": resource_frame}


def _wrapper_dim_estimate(sparse: np.ndarray, slots: int, mu: float) -> int:
    """Cheap padded-wrapper dimension probe (build the wrapper once to read its unitary size)."""

    from robust_qsvt_se.qsvt.output_aware_sparse_selection import build_common_padded_wrapper

    wrapper = build_common_padded_wrapper(sparse, slots=slots, mu=mu)
    return int(wrapper.unitary.shape[0])


# --------------------------------------------------------------- orchestrator


def run_larger_block(
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
    dimension = int(config.get("dimension", 16))
    design = build_case_design(case_name, seed, dimension=dimension)

    block_payload = write_block_inventory(case_name, design, destination, STUDY_ID)
    write_functional_inventory(design, destination, STUDY_ID)

    started = time.perf_counter()
    selector_result = run_selector_comparison(
        case_name, design, output_dir=destination,
        support_budgets=[int(k) for k in config["support_budgets"]],
        slot_budgets=[int(s) for s in config["slot_budgets"]],
        training_seeds=[int(s) for s in config["training_seed_ids"]],
        held_out_seeds=[int(s) for s in config["held_out_seed_ids"]],
        selectors_subset=tuple(config["required_selectors"]),
        beam_width=int(config.get("near_oracle_beam_width", 6)),
        near_oracle_max_loss_evals=int(config.get("near_oracle_max_loss_evals", 200000)),
        random_seed_base=int(config.get("random_seed_base", 314159)),
        y_floor=float(config.get("y_floor", 1e-6)),
        failure_threshold=float(config.get("failure_threshold", 0.1)),
    )
    selector_wall_seconds = time.perf_counter() - started
    peak_rss_mb = _peak_rss_mb()

    support_frame = selector_result["support_frame"]
    scaling = _runtime_scaling(
        support_frame, case_name, design.conditioning["nonzeros"], dimension
    )
    scaling["selector_wall_seconds_total"] = selector_wall_seconds
    scaling["peak_rss_mb"] = peak_rss_mb
    atomic_write_csv(destination / "runtime_scaling.csv", scaling)

    qsvt_resource = run_qsvt_and_resources(case_name, design, config, destination)

    _write_larger_block_report(
        destination, case_name, design, block_payload, selector_result, scaling, qsvt_resource,
        peak_rss_mb,
    )
    atomic_write_json(
        destination / "provenance.json",
        provenance_block(config_path, config) | {"study_id": STUDY_ID},
    )
    atomic_write_json(
        destination / "scaling_summary.json",
        json_safe_floats({
            "case": case_name, "dimension": dimension,
            "conditioning": design.conditioning,
            "coverage_floor_k": dimension,
            "peak_rss_mb": peak_rss_mb,
            "selector_wall_seconds_total": selector_wall_seconds,
        }),
    )
    write_manifest_and_checksums(
        destination, study_id=STUDY_ID,
        extra={
            "case": case_name, "block_shape": [dimension, dimension],
            "physical_functionals": len(design.physical_functional_ids),
            "coverage_floor_k": dimension,
            "peak_rss_mb": peak_rss_mb,
        },
    )
    return {
        "case": case_name, "dimension": dimension,
        "selector_records": selector_result["support_records"],
        "feasible_supports": selector_result["feasible_supports"],
        "qsvt_rows": qsvt_resource["qsvt_rows"],
        "resource_rows": qsvt_resource["resource_rows"],
        "peak_rss_mb": peak_rss_mb,
    }


def _write_larger_block_report(
    destination, case_name, design, block_payload, selector_result, scaling, qsvt_resource,
    peak_rss_mb,
) -> None:
    cond = design.conditioning
    summary = selector_result["summary_frame"]
    mean_summary = summary[summary["objective"] == "mean"] if not summary.empty else summary
    lines = [
        f"# Larger-Block (16x16) Validation Report - {case_name.upper()}",
        "", CLAIM_BOUNDARY, "",
        "## Block (identical deterministic policy, seed 123)", "",
        f"- shape 16x16, {cond['nonzeros']} nonzeros (density {cond['density']:.3f}), "
        f"coverage floor k = 16",
        f"- rank {cond['rank']}/16, raw kappa `{cond['raw_condition_number']:.4g}`, "
        "- regularized normal-system kappa "
        f"`{cond['regularized_normal_system_condition_number']:.4g}`",
        f"- block alpha = 4*sigma_min_pos^2 = `{design.small.alpha:.6g}`",
        f"- peak RSS during selector comparison: {peak_rss_mb:.1f} MB",
        "",
        "## Selector held-out comparison (mean objective, required subset)", "",
        "| selector | feasibility | mean held-out err | median | overlap w/ near-oracle |",
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
    qframe = qsvt_resource["qsvt_frame"]
    executed = (
        int((qframe.get("evidence_status") == "executed_statevector").sum())
        if not qframe.empty else 0
    )
    lines += [
        "",
        "## QSVT validation (where practical)", "",
        f"- QSVT rows: {len(qframe)}; executed-statevector: {executed}; "
        "remaining are exact-matrix-action / modeled.",
        "- A full 16x16 transpiled QSVT circuit is NOT required; the padded wrapper unitary is "
        "128-dimensional (2^7), so statevector action is executed exactly where feasible.",
        "",
        "## Support stability (mean pairwise Jaccard across budget cells)", "",
        "| selector | feasible cells | mean Jaccard |",
        "|---|---:|---:|",
    ]
    stability = selector_result["stability_frame"]
    if not stability.empty:
        for _, row in stability.sort_values("selector").iterrows():
            lines.append(
                f"| `{row['selector']}` | {int(row['feasible_cells'])} | "
                f"{row['mean_pairwise_jaccard']:.3f} |"
            )
    lines += [
        "",
        "## Runtime / memory scaling vs 8x8", "",
        "See `runtime_scaling.csv` (runtime, MILP + exact-Ridge solve counts, per selector) and "
        "the comparison track for normalized 8x8-vs-16x16 quantities. Peak RSS "
        f"{peak_rss_mb:.1f} MB. Coverage floor forces k >= 16 (k < 16 is infeasible, retained).",
        "",
    ]
    (destination / "larger_block_report.md").write_text("\n".join(lines), encoding="utf-8")
