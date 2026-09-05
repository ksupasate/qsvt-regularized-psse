"""Workstream 4 - resource-accuracy Pareto analysis.

Tests the QSVT-oriented question: does output-aware support reach a fixed selected-output
accuracy using a smaller access-resource budget than output-agnostic support?  Executed /
transpiled resources (signal-unitary gate counts, post-selection probability) are kept
strictly separate from modeled resources (loader, readout, repetition, shots) via an
``evidence_status`` field, and combined through the declared cost model

    C_total = (C_load + d * C_signal(k, s) + C_readout) / p_post.

No result claims quantum speedup, quantum advantage, or superiority over the matched Ridge
filter; the comparison is only whether output-aware selection lowers the access budget
needed to hit a fixed selected-output accuracy on this controlled 8x8 workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.bipartite_slot_assignment import minimum_slot_count
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    build_common_padded_wrapper,
    compute_output_aware_entry_scores,
    deterministic_ridge_leverage_scores,
    refine_support_one_swap,
    select_resource_constrained_support,
    support_constraint_report,
)
from robust_qsvt_se.qsvt.sparse_error_precision_study import pareto_nondominated
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    array_fingerprint,
    atomic_write_csv,
    atomic_write_json,
    load_config,
    provenance_block,
    write_manifest_and_checksums,
)
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import (
    balanced_magnitude_score,
    build_tasks,
    evaluate_support,
    frozen_8x8_design_as_small,
)

STUDY_ID = "reviewer_blocking_resource_pareto_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/tqe_reviewer_blocking_experiments/resource_pareto")
DEFAULT_CONFIG_PATH = Path("configs/tqe_reviewer_blocking/resource_pareto.json")

LEGACY_FUNCTIONALS = (
    "coordinate_e0",
    "signed_difference_e0_minus_e1",
    "aggregate_e0_to_e3",
)
OUTPUT_AWARE_SELECTORS = ("sensitivity_initial_mean", "sensitivity_refined_mean")
OUTPUT_AGNOSTIC_SELECTORS = ("global_magnitude", "balanced_magnitude", "ridge_leverage")

FIXED_ERROR_COLUMNS = (
    "error_threshold",
    "selector",
    "selector_class",
    "reached",
    "min_c_total_gates",
    "min_total_estimated_gates",
    "min_executed_signal_gates",
    "min_k",
    "min_slots",
    "argmin_c_total_k",
    "argmin_c_total_s",
)


def build_support(design, selector: str, constraints: SupportConstraints, training_tasks):
    matrix = design.matrix
    legacy = [t for t in training_tasks if t.functional_id in LEGACY_FUNCTIONALS]
    if selector == "global_magnitude":
        return select_resource_constrained_support(matrix, np.abs(matrix), constraints).support
    if selector == "balanced_magnitude":
        return select_resource_constrained_support(
            matrix, balanced_magnitude_score(matrix), constraints
        ).support
    if selector == "ridge_leverage":
        _r, _c, leverage = deterministic_ridge_leverage_scores(matrix, alpha=design.alpha)
        return select_resource_constrained_support(matrix, leverage, constraints).support
    if selector in {"sensitivity_initial_mean", "sensitivity_refined_mean"}:
        scores = compute_output_aware_entry_scores(
            matrix, legacy, alpha=design.alpha, epsilon=1e-15
        )
        initial = select_resource_constrained_support(matrix, scores.sensitivity_mean, constraints)
        if selector == "sensitivity_initial_mean" or initial.support is None:
            return initial.support
        refined = refine_support_one_swap(
            matrix,
            initial.support,
            legacy,
            constraints,
            alpha=design.alpha,
            y_floor=1e-6,
            objective="mean_normalized_error",
            max_iterations=8,
            improvement_tolerance=1e-12,
        )
        return refined.support
    raise ValueError(f"unknown selector {selector}")


@dataclass(slots=True)
class CommonQSVTDesign:
    slots: int
    mu: float
    beta: float
    normalized_lambda: float
    degree: int
    phases: np.ndarray
    target: Any
    bounded: bool
    fit_error: float


def build_common_qsvt_design(
    matrix: np.ndarray, alpha: float, common_slots: int, degree: int, cache_dir: Path
) -> CommonQSVTDesign:
    from robust_qsvt_se.qsvt.phase_synthesis import (
        synthesize_pennylane_phases_cached,
        validate_qsvt_polynomial,
    )

    mu = float(np.max(np.abs(matrix)))
    beta = float(common_slots * mu)
    singular = np.linalg.svd(matrix, compute_uv=False)
    positive = singular[singular > 1e-10]
    domain_min = float(np.clip(0.9 * positive.min() / beta, 1e-4, 0.999))
    target = fit_codesigned_bounded_polynomial(
        beta=beta, alpha=float(alpha), domain_min=domain_min, domain_max=1.0, degree=int(degree),
        margin=1.05,
    )
    coefficients = np.asarray(target.coefficients, dtype=np.float64)
    bounded = bool(target.bounded_max_abs <= 1.0 + 2e-3)
    try:
        validate_qsvt_polynomial(coefficients, parity="odd", bound_tolerance=2e-3)
    except Exception:
        bounded = False
    phases = np.zeros(0)
    if bounded:
        result = synthesize_pennylane_phases_cached(
            coefficients,
            angle_solver="iterative",
            cache_dir=cache_dir,
            cache_metadata={"study_id": STUDY_ID, "slots": common_slots, "degree": degree},
        )
        phases = np.asarray(result.phases, dtype=np.float64)
        if phases.size != degree + 1:
            bounded = False
    return CommonQSVTDesign(
        slots=int(common_slots),
        mu=mu,
        beta=beta,
        normalized_lambda=float(alpha) / beta**2,
        degree=int(degree),
        phases=phases,
        target=target,
        bounded=bounded,
        fit_error=float(target.fit_max_abs_error),
    )


def executed_support_resources(
    support: np.ndarray,
    matrix: np.ndarray,
    design: CommonQSVTDesign,
    residual_unit: np.ndarray,
    *,
    basis_gates: list[str],
    optimization_level: int,
) -> dict[str, Any]:
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit

    sparse = np.where(support, matrix, 0.0)
    wrapper = build_common_padded_wrapper(sparse, slots=design.slots, mu=design.mu)
    compiled = transpile(
        wrapper.circuit, basis_gates=basis_gates, optimization_level=optimization_level
    )
    counts = {str(k): int(v) for k, v in compiled.count_ops().items()}
    signal_gate_count = int(sum(counts.values()))
    result: dict[str, Any] = {
        "signal_unitary_gate_count": signal_gate_count,
        "signal_unitary_depth": int(compiled.depth()),
        "cx_count": int(counts.get("cx", 0)),
        "controlled_value_rotations": int(design.slots * sparse.shape[1]),
        "wrapper_reconstruction_error": float(wrapper.reconstruction_error),
        "resource_evidence": "executed_qiskit_transpile_u3_cx",
    }
    bundle = build_structured_qsvt_operator_circuit(
        wrapper.unitary, design.phases, encoded_dimension=sparse.shape[1]
    )
    initial = np.zeros(wrapper.unitary.shape[0], dtype=np.complex128)
    initial[: sparse.shape[0]] = residual_unit
    evolved = Statevector(initial).evolve(bundle.qsvt_operator_circuit).data
    encoded = np.asarray(evolved[: sparse.shape[1]], dtype=np.complex128)
    postselection = float(np.vdot(encoded, encoded).real)
    normalized_matrix = sparse.T / design.beta
    from numpy.polynomial import Polynomial

    left, singular_values, right_t = np.linalg.svd(normalized_matrix, full_matrices=False)
    poly = Polynomial(np.asarray(design.target.coefficients, dtype=np.float64))
    exact = (left @ np.diag(poly(singular_values)) @ right_t) @ residual_unit
    action_error = float(
        np.linalg.norm(np.real(encoded) - exact) / max(np.linalg.norm(exact), 1e-30)
    )
    result["postselection_probability"] = postselection
    result["qsvt_action_error"] = action_error
    result["postselection_evidence"] = "executed_statevector"
    return result


def cost_model(
    executed: dict[str, Any],
    *,
    dimension: int,
    degree: int,
    shots: int,
) -> dict[str, Any]:
    """Combine executed signal cost with modeled loader / readout / repetition."""

    c_load = int(2 * dimension - 2)  # modeled binary-tree amplitude loader
    c_readout = int(dimension)  # modeled selected-output readout
    c_signal = int(executed["signal_unitary_gate_count"])
    p_post = float(executed["postselection_probability"])
    total_estimated_gates = c_load + degree * c_signal + c_readout
    expected_attempts = float("inf") if p_post <= 0 else 1.0 / p_post
    c_total = float("inf") if p_post <= 0 else total_estimated_gates / p_post
    return {
        "modeled_c_load_gates": c_load,
        "modeled_c_readout_gates": c_readout,
        "executed_c_signal_gates": c_signal,
        "signal_unitary_calls": int(degree),
        "total_signal_unitary_calls": int(degree),
        "total_estimated_gates_per_attempt": int(total_estimated_gates),
        "expected_postselection_attempts": expected_attempts,
        "c_total_gates": c_total,
        "modeled_shots": int(shots),
        "modeled_total_shot_attempts": (
            float("inf") if p_post <= 0 else float(shots) * expected_attempts
        ),
        "load_evidence": "modeled_binary_tree_loader_linear",
        "readout_evidence": "modeled_readout_linear",
        "repetition_evidence": "modeled_uniform_postselection_repetition",
        "cost_model": "C_total = (C_load + d*C_signal + C_readout) / p_post",
    }


# ------------------------------------------------------------- orchestrator


def _family(functional_id: str) -> str:
    if functional_id.startswith("coordinate"):
        return "coordinate"
    if functional_id.startswith("signed_difference") or "branch" in functional_id:
        return "branch_difference"
    return "aggregate"


def _accuracy(design_small, support, heldout_tasks) -> dict[str, Any]:
    rows = evaluate_support(design_small, support, heldout_tasks)
    frame = pd.DataFrame(rows)
    per_family = {
        f"heldout_error_family_{family}": float(group["normalized_error"].mean())
        for family, group in frame.assign(
            family=frame["functional_id"].map(_family)
        ).groupby("family")
    }
    return {
        "mean_heldout_normalized_error": float(frame["normalized_error"].mean()),
        "worst_heldout_normalized_error": float(frame["normalized_error"].max()),
        "worst_heldout_absolute_error": float(frame["absolute_error"].max()),
        **per_family,
    }


def run_resource_pareto(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"unexpected study_id: {config.get('study_id')}")
    destination = Path(output_dir)
    (destination / "figure_data").mkdir(parents=True, exist_ok=True)
    (destination / "figures").mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    degree = int(config.get("degree", 31))
    shots = int(config.get("modeled_shots", 100000))
    basis_gates = list(config.get("basis_gates", ["u3", "cx"]))
    optimization_level = int(config.get("optimization_level", 1))
    error_thresholds = [float(t) for t in config.get("error_thresholds", [0.5, 0.75, 1.0])]
    selectors = list(config["selectors"])
    training_seeds = [int(s) for s in config["training_seed_ids"]]
    heldout_seeds = [int(s) for s in config["held_out_seed_ids"]]

    design = frozen_8x8_design_as_small(destination)
    training_tasks = build_tasks(design, training_seeds, "training", LEGACY_FUNCTIONALS)
    heldout_tasks = build_tasks(design, heldout_seeds, "held_out", LEGACY_FUNCTIONALS)
    residual_unit = training_tasks[0].residual / np.linalg.norm(training_tasks[0].residual)

    common_designs: dict[int, CommonQSVTDesign] = {}
    for s in config["slot_budgets"]:
        common_designs[int(s)] = build_common_qsvt_design(
            design.matrix, design.alpha, int(s), degree, cache_dir
        )

    rows: list[dict[str, Any]] = []
    for selector in selectors:
        selector_class = (
            "output_aware" if selector in OUTPUT_AWARE_SELECTORS else "output_agnostic"
        )
        for k in config["support_budgets"]:
            for s in config["slot_budgets"]:
                constraints = SupportConstraints(int(k), int(s), True)
                support = build_support(design, selector, constraints, training_tasks)
                base = {
                    "selector": selector,
                    "selector_class": selector_class,
                    "k_budget": int(k),
                    "slot_budget": int(s),
                    "degree": degree,
                }
                feasible = support is not None and support_constraint_report(
                    design.matrix, support, constraints
                )["valid"]
                if not feasible:
                    rows.append(
                        {**base, "status": "failed", "failure_reason": "support_infeasible"}
                    )
                    continue
                common = common_designs[int(s)]
                accuracy = _accuracy(design, support, heldout_tasks)
                record = {
                    **base,
                    "status": "completed",
                    "failure_reason": "",
                    "actual_nonzeros": int(support.sum()),
                    "slot_count": int(
                        minimum_slot_count(np.where(support, design.matrix, 0.0).T != 0.0)
                    ),
                    "support_fingerprint": array_fingerprint(support.astype(np.float64)),
                    "common_beta": common.beta,
                    "normalized_lambda": common.normalized_lambda,
                    "qsvt_bounded_at_degree": common.bounded,
                    **accuracy,
                }
                if not common.bounded:
                    record["status"] = "qsvt_infeasible_at_common_degree"
                    record["failure_reason"] = "common_polynomial_unbounded"
                    rows.append(record)
                    continue
                executed = executed_support_resources(
                    support,
                    design.matrix,
                    common,
                    residual_unit,
                    basis_gates=basis_gates,
                    optimization_level=optimization_level,
                )
                costs = cost_model(executed, dimension=design.dimension, degree=degree, shots=shots)
                rows.append({**record, **executed, **costs})

    frame = pd.DataFrame(rows)
    atomic_write_csv(destination / "raw_resource_accuracy.csv", frame)
    completed = frame[frame["status"] == "completed"].copy()

    pareto_frame = _pareto_fronts(completed)
    atomic_write_csv(destination / "pareto_fronts.csv", pareto_frame)
    fixed_error = _fixed_error_costs(completed, error_thresholds)
    atomic_write_csv(destination / "fixed_error_costs.csv", fixed_error)
    ledger = _resource_status_ledger()
    atomic_write_csv(destination / "resource_status_ledger.csv", ledger)
    _write_figure_data(destination, completed, pareto_frame, fixed_error)
    _render_figures(destination, completed, error_thresholds)
    _write_summary(destination, config, frame, completed, fixed_error, error_thresholds)
    atomic_write_json(
        destination / "provenance.json",
        provenance_block(config_path, config) | {"study_id": STUDY_ID},
    )
    write_manifest_and_checksums(
        destination,
        study_id=STUDY_ID,
        extra={
            "cells": len(frame),
            "completed": len(completed),
            "selectors": selectors,
        },
    )
    return {
        "cells": len(frame),
        "completed": len(completed),
        "retained_failures": int((frame["status"] != "completed").sum()),
        "pareto_points": int(pareto_frame["nondominated"].sum()) if not pareto_frame.empty else 0,
        "error_thresholds": error_thresholds,
    }


PARETO_COST_AXES = (
    ("actual_nonzeros", "support_nonzeros_k"),
    ("slot_count", "slots_s"),
    ("executed_c_signal_gates", "executed_signal_gates"),
    ("total_estimated_gates_per_attempt", "total_estimated_gates"),
    ("c_total_gates", "c_total"),
    ("expected_postselection_attempts", "expected_attempts"),
)


def _pareto_fronts(completed: pd.DataFrame) -> pd.DataFrame:
    if completed.empty:
        return pd.DataFrame(
            columns=[
                "cost_axis", "selector", "k_budget", "slot_budget", "error", "cost",
                "nondominated",
            ]
        )
    frames: list[pd.DataFrame] = []
    error_column = "mean_heldout_normalized_error"
    for cost_column, axis_name in PARETO_COST_AXES:
        subset = completed[
            ["selector", "selector_class", "k_budget", "slot_budget", error_column, cost_column]
        ].copy()
        subset = subset[np.isfinite(subset[cost_column])]
        if subset.empty:
            continue
        subset = subset.rename(columns={error_column: "error", cost_column: "cost"})
        subset["nondominated"] = pareto_nondominated(
            subset, error_column="error", cost_column="cost"
        )
        subset["cost_axis"] = axis_name
        frames.append(subset)
    return (
        pd.concat(frames, ignore_index=True).sort_values(
            ["cost_axis", "cost", "error"], kind="stable"
        )
        if frames
        else pd.DataFrame()
    )


def _fixed_error_costs(completed: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if completed.empty:
        # Retain every declared threshold even when there is no Pareto point and therefore
        # no selector identity.  Nullable values serialize as blank fields rather than an
        # invented infinite minimum.
        return pd.DataFrame(
            [
                {
                    "error_threshold": threshold,
                    "selector": pd.NA,
                    "selector_class": pd.NA,
                    "reached": False,
                }
                for threshold in thresholds
            ],
            columns=FIXED_ERROR_COLUMNS,
        )
    for threshold in thresholds:
        for selector, group in completed.groupby("selector", sort=True):
            # A resource threshold is reportable only when both the requested error and a
            # finite mixed cost are available.  NaN/inf costs remain unavailable evidence.
            reachable = group[
                (group["mean_heldout_normalized_error"] <= threshold)
                & np.isfinite(group["c_total_gates"])
            ]
            selector_class = group["selector_class"].iloc[0]
            if reachable.empty:
                rows.append(
                    {
                        "error_threshold": threshold,
                        "selector": selector,
                        "selector_class": selector_class,
                        "reached": False,
                    }
                )
                continue
            best = reachable.sort_values("c_total_gates", kind="stable").iloc[0]
            rows.append(
                {
                    "error_threshold": threshold,
                    "selector": selector,
                    "selector_class": selector_class,
                    "reached": True,
                    "min_c_total_gates": float(reachable["c_total_gates"].min()),
                    "min_total_estimated_gates": float(
                        reachable["total_estimated_gates_per_attempt"].min()
                    ),
                    "min_executed_signal_gates": float(reachable["executed_c_signal_gates"].min()),
                    "min_k": int(reachable["actual_nonzeros"].min()),
                    "min_slots": int(reachable["slot_count"].min()),
                    "argmin_c_total_k": int(best["k_budget"]),
                    "argmin_c_total_s": int(best["slot_budget"]),
                }
            )
    return pd.DataFrame(rows, columns=FIXED_ERROR_COLUMNS)


def _best_class_at_threshold(
    fixed_error: pd.DataFrame, threshold: float, selector_class: str
) -> dict[str, Any]:
    """Return a deterministic, nullable reporting record for one selector class."""

    required = {"error_threshold", "selector", "selector_class", "reached"}
    if fixed_error.empty or not required.issubset(fixed_error.columns):
        return {"feasible": False, "selector": pd.NA, "min_c_total_gates": np.nan}
    sub = fixed_error[
        (fixed_error["error_threshold"] == threshold)
        & (fixed_error["selector_class"] == selector_class)
        & fixed_error["reached"].fillna(False).astype(bool)
    ]
    if "min_c_total_gates" not in sub.columns or sub.empty:
        return {"feasible": False, "selector": pd.NA, "min_c_total_gates": np.nan}
    costs = pd.to_numeric(sub["min_c_total_gates"], errors="coerce")
    finite = sub[np.isfinite(costs)].copy()
    if finite.empty:
        return {"feasible": False, "selector": pd.NA, "min_c_total_gates": np.nan}
    finite["_report_cost"] = pd.to_numeric(
        finite["min_c_total_gates"], errors="coerce"
    )
    minimum = float(finite["_report_cost"].min())
    tied = finite[np.isclose(finite["_report_cost"], minimum, rtol=1e-12, atol=1e-12)]
    selectors = sorted(str(value) for value in tied["selector"].dropna().unique())
    return {
        "feasible": True,
        "selector": ", ".join(selectors) if selectors else pd.NA,
        "min_c_total_gates": minimum,
    }


def _format_available_cost(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{float(number):.4g}" if np.isfinite(number) else ""


def _resource_status_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"component": "signal_unitary_gates", "symbol": "C_signal",
             "evidence_status": "executed_transpiled"},
            {"component": "postselection_probability", "symbol": "p_post",
             "evidence_status": "executed_statevector"},
            {"component": "signal_unitary_calls", "symbol": "d",
             "evidence_status": "executed_degree_count"},
            {"component": "residual_loader", "symbol": "C_load",
             "evidence_status": "modeled_linear"},
            {"component": "selected_output_readout", "symbol": "C_readout",
             "evidence_status": "modeled_linear"},
            {"component": "postselection_repetition", "symbol": "1/p_post",
             "evidence_status": "modeled_uniform"},
            {"component": "shot_attempts", "symbol": "shots/p_post", "evidence_status": "modeled"},
            {"component": "c_total", "symbol": "C_total",
             "evidence_status": "mixed_executed_and_modeled"},
        ]
    )


def _write_figure_data(
    destination: Path,
    completed: pd.DataFrame,
    pareto_frame: pd.DataFrame,
    fixed_error: pd.DataFrame,
) -> None:
    figure_dir = destination / "figure_data"
    if not completed.empty:
        atomic_write_csv(
            figure_dir / "error_vs_resources.csv",
            completed[
                [
                    "selector", "selector_class", "k_budget", "slot_budget", "actual_nonzeros",
                    "slot_count", "mean_heldout_normalized_error", "worst_heldout_normalized_error",
                    "executed_c_signal_gates", "total_estimated_gates_per_attempt",
                    "c_total_gates", "expected_postselection_attempts",
                    "postselection_probability",
                ]
            ],
        )
    atomic_write_csv(figure_dir / "pareto_fronts.csv", pareto_frame)
    atomic_write_csv(figure_dir / "fixed_error_costs.csv", fixed_error)


def _render_figures(destination: Path, completed: pd.DataFrame, thresholds: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = destination / "figures"
    if completed.empty:
        return
    error_col = "mean_heldout_normalized_error"

    # error vs support nonzeros, per selector
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for selector, group in completed.groupby("selector"):
        agg = group.groupby("actual_nonzeros")[error_col].min().reset_index()
        ax.plot(agg["actual_nonzeros"], agg[error_col], marker="o", label=selector)
    ax.set_xlabel("selected support nonzeros (k)")
    ax.set_ylabel("held-out selected-output error")
    ax.set_yscale("log")
    ax.set_title("Selected-output error vs support size")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "error_vs_support_nonzeros.png", dpi=130)
    plt.close(fig)

    # Pareto: c_total vs error, per selector class
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    finite = completed[np.isfinite(completed["c_total_gates"])]
    for selector, group in finite.groupby("selector"):
        ax.scatter(group["c_total_gates"], group[error_col], s=22, label=selector)
    ax.set_xlabel("modeled C_total (gates / accepted sample)")
    ax.set_ylabel("held-out selected-output error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Resource-accuracy scatter (executed signal + modeled loader/readout)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "error_vs_c_total.png", dpi=130)
    plt.close(fig)

    # min C_total at fixed error, output-aware vs agnostic
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for threshold in thresholds:
        reachable = finite[finite[error_col] <= threshold]
        if reachable.empty:
            continue
        by_class = reachable.groupby("selector_class")["c_total_gates"].min()
        for klass, value in by_class.items():
            colour = "C0" if klass == "output_aware" else "C1"
            ax.bar(f"{threshold:g}\n{klass}", value, color=colour)
    ax.set_ylabel("min modeled C_total (gates)")
    ax.set_yscale("log")
    ax.set_title("Min resource cost to reach fixed selected-output error")
    fig.tight_layout()
    fig.savefig(figure_dir / "min_cost_at_fixed_error.png", dpi=130)
    plt.close(fig)


def _write_summary(
    destination: Path,
    config: dict[str, Any],
    frame: pd.DataFrame,
    completed: pd.DataFrame,
    fixed_error: pd.DataFrame,
    thresholds: list[float],
) -> None:
    retained_failures = (
        int((frame["status"] != "completed").sum()) if "status" in frame.columns else 0
    )
    lines = [
        "# Resource-Accuracy Pareto Summary",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Cost model",
        "",
        "`C_total = (C_load + d * C_signal + C_readout) / p_post` with C_signal and p_post "
        "executed (qiskit transpile + statevector) and C_load / C_readout / repetition modeled. "
        "Evidence status per component is in `resource_status_ledger.csv`.",
        "",
        f"- cells: {len(frame)}; completed: {len(completed)}; "
        f"retained failures: {retained_failures}",
        f"- common degree: {config.get('degree', 31)}; "
        f"slot budgets: {config.get('slot_budgets', [])}",
        "",
        "## Does output-aware selection lower the access budget at fixed accuracy?",
        "",
        "| threshold | aware feasible | aware selector | aware min C_total | "
        "agnostic feasible | agnostic selector | agnostic min C_total | verdict |",
        "|---:|:--:|---|---:|:--:|---|---:|---|",
    ]
    for threshold in thresholds:
        aware = _best_class_at_threshold(fixed_error, threshold, "output_aware")
        agnostic = _best_class_at_threshold(fixed_error, threshold, "output_agnostic")
        aware_min = aware["min_c_total_gates"]
        agnostic_min = agnostic["min_c_total_gates"]
        if not aware["feasible"] and not agnostic["feasible"]:
            verdict = "both infeasible"
        elif aware["feasible"] and not agnostic["feasible"]:
            verdict = "aware only"
        elif agnostic["feasible"] and not aware["feasible"]:
            verdict = "agnostic only"
        elif aware_min < agnostic_min:
            verdict = "aware cheaper"
        elif np.isclose(aware_min, agnostic_min, rtol=1e-12, atol=1e-12):
            verdict = "tie"
        else:
            verdict = "agnostic cheaper"
        aware_selector = aware["selector"] if pd.notna(aware["selector"]) else "unavailable"
        agnostic_selector = (
            agnostic["selector"] if pd.notna(agnostic["selector"]) else "unavailable"
        )
        lines.append(
            f"| {threshold:g} | {str(aware['feasible']).lower()} | {aware_selector} | "
            f"{_format_available_cost(aware_min)} | {str(agnostic['feasible']).lower()} | "
            f"{agnostic_selector} | {_format_available_cost(agnostic_min)} | {verdict} |"
        )
    lines += [
        "",
        "Interpretation is a controlled resource-accounting comparison on one 8x8 workload, "
        "not a quantum-advantage or speedup claim.",
        "",
    ]
    (destination / "resource_pareto_summary.md").write_text("\n".join(lines), encoding="utf-8")
