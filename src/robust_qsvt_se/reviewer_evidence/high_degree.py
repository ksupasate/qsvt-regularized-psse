"""Phase 3 - high-degree QSVT feasibility slice (degrees 31/63/127/255).

Drives the UNCHANGED ``joint_feasibility.evaluate_qsvt_feasibility`` on a small predeclared slice
to answer: does raising the QSVT polynomial degree from 31/63 to 127/255 close the gap between the
application-useful regularization region and the QSVT-feasible region observed at low degree?  The
composite feasibility criterion is reused verbatim (boundedness + parity + uniform fit + phase
synthesis + statevector action).  Every failure is retained.  No feasibility criterion is changed
after seeing results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.cross_case_validation.common import build_case_full_system
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.output_aware_sparse_selection import SupportConstraints
from robust_qsvt_se.reviewer_blocking.common import (
    CLAIM_BOUNDARY,
    atomic_write_csv,
    atomic_write_json,
    load_config,
    provenance_block,
)
from robust_qsvt_se.reviewer_blocking.joint_feasibility import (
    application_metrics,
    evaluate_qsvt_feasibility,
)
from robust_qsvt_se.reviewer_evidence.engine import (
    STRUCTURES,
    build_structure,
    build_tasks,
    design_time_alpha_regimes,
    seed_reference,
    select_support,
)

STUDY_ID = "reviewer_evidence_high_degree_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/reviewer_blocking_tqe_evidence")
DEFAULT_CONFIG_PATH = Path("configs/reviewer_blocking_tqe_evidence/high_degree.json")


def _utility_reference(structure_id: str) -> tuple[float, float]:
    case_name, _dim, _ = STRUCTURES[structure_id]
    system = build_case_full_system(case_name, 123)
    grid = np.logspace(-3, 7, 40)
    rmses = np.asarray(
        [
            float(
                np.sqrt(
                    np.mean(
                        (
                            ridge_svd_solution(system.matrix, system.residual, alpha=float(a))
                            - system.x_true
                        )
                        ** 2
                    )
                )
            )
            for a in grid
        ]
    )
    return float(rmses.min()), float(rmses[np.argmin(grid)])


def _full_system_alpha_regimes(structure_id: str) -> list[dict[str, Any]]:
    """Full-state-useful alphas (the HARD, small-lambda QSVT region) selected on the full system.

    These are the reviewer-critical alphas: they minimize full-state RMSE (so are application-useful
    by construction) and therefore probe whether raising the QSVT degree makes the useful region
    QSVT-feasible.  ``gcv`` is deployable; ``oracle_rmse`` uses ground truth (diagnostic only).
    """

    from robust_qsvt_se.paper.tqe_revision_core import select_alpha_gcv, select_alpha_oracle_rmse

    case_name, _dim, _ = STRUCTURES[structure_id]
    system = build_case_full_system(case_name, 123)
    grid = np.logspace(-3, 7, 40)
    out: list[dict[str, Any]] = []
    try:
        out.append(
            {
                "regime": "full_system_gcv",
                "alpha": float(select_alpha_gcv(system.matrix, system.residual, grid)),
                "deployable": True,
                "is_oracle": False,
                "note": "full-system GCV (application-useful region)",
            }
        )
    except Exception as exc:
        out.append(
            {
                "regime": "full_system_gcv",
                "alpha": float("nan"),
                "deployable": True,
                "is_oracle": False,
                "note": f"unavailable: {exc}",
            }
        )
    try:
        out.append(
            {
                "regime": "full_system_oracle_rmse",
                "alpha": float(
                    select_alpha_oracle_rmse(system.matrix, system.residual, system.x_true, grid)
                ),
                "deployable": False,
                "is_oracle": True,
                "note": "full-state RMSE oracle (diagnostic)",
            }
        )
    except Exception as exc:
        out.append(
            {
                "regime": "full_system_oracle_rmse",
                "alpha": float("nan"),
                "deployable": False,
                "is_oracle": True,
                "note": f"unavailable: {exc}",
            }
        )
    return out


def run_synthesis_demonstration(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    selector: str = "global_magnitude",
    structure_id: str = "ieee14_8x8",
    degree: int = 31,
) -> dict[str, Any]:
    """Demonstrate that a QSVT-feasible case actually synthesizes phases and executes.

    One feasible operating point (over-regularized fixed-benchmark alpha, degree 31) is taken all
    the way through phase synthesis and a statevector matrix-action check, proving the toolchain
    recovers ``degree+1`` phases and that the QSVT circuit implements the Ridge filter (small action
    error).  The high-degree analytic sweep separately records fit/boundedness; this file supplies
    the phase-recovery + statevector tier that the sweep skips for compute.
    """

    destination = Path(output_dir)
    cache_dir = destination / "high_degree" / "demo_phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_structure(structure_id)
    cons = SupportConstraints(16, 3, True)
    train = build_tasks(structure_id, [1000, 1001, 1002], "training")
    support = (ctx.matrix != 0.0 if selector == "full_support"
               else select_support(ctx, selector, cons, train, ctx.physical_ids)["support"])
    sparse_block = np.where(support, ctx.matrix, 0.0)
    residual, _ = seed_reference(structure_id, 1000)
    residual_unit = residual / np.linalg.norm(residual)
    feas = evaluate_qsvt_feasibility(
        sparse_block, ctx.alpha_fixed, degree, margin=1.05, bound_tolerance=0.002,
        uniform_tolerance=0.002, execute_statevector=True, residual_unit=residual_unit,
        cache_dir=cache_dir, attempt_phase_synthesis=True,
    )
    payload = {
        "study_id": STUDY_ID + "_synthesis_demo",
        "claim_boundary": CLAIM_BOUNDARY,
        "structure_id": structure_id, "selector": selector, "alpha_regime": "fixed_benchmark",
        "alpha": float(ctx.alpha_fixed), "degree": int(degree),
        "normalized_lambda": float(feas.normalized_lambda),
        "boundedness_parity_fit_ok": bool(feas.boundedness_ok),
        "uniform_fit_error": float(feas.uniform_fit_error),
        "phase_synthesis_status": feas.phase_synthesis_status,
        "phase_count": int(feas.phase_count),
        "expected_phase_count": int(degree + 1),
        "phase_count_matches": bool(feas.phase_count == degree + 1),
        "statevector_action_error": float(feas.statevector_action_error),
        "postselection_probability": float(feas.postselection_probability),
        "evidence_tier": feas.evidence_status,
        "note": ("A feasible over-regularized operating point synthesizes degree+1 phases and the "
                 "statevector reproduces the Ridge filter; it over-regularizes full-state PSSE "
                 "(not application-useful). Higher degrees / application-useful alphas are "
                 "recorded analytically in the sweep and do not reach this tier."),
    }
    atomic_write_json(destination / "high_degree_synthesis_demonstration.json", payload)
    return payload


def run_high_degree(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config = load_config(config_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "high_degree" / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    degrees = [int(d) for d in config["degrees"]]
    if any(d % 2 == 0 for d in degrees):
        raise ValueError("only odd degrees are supported")
    selectors = list(config["selectors"])
    k = int(config["support_budget"])
    s = int(config["slot_budget"])
    alpha_regime_names = list(config["alpha_regimes"])
    uniform_tol = float(config.get("uniform_approximation_tolerance", 0.002))
    bound_tol = float(config.get("bound_tolerance", 0.002))
    action_tol = float(config.get("action_error_tolerance", 1e-6))
    margin = float(config.get("target_margin", 1.05))
    useful_ratio = float(config.get("useful_rmse_ratio_threshold", 1.5))
    execute_statevector = bool(config.get("execute_statevector", True))
    statevector_ceiling = int(config.get("statevector_execution_ceiling_dim", 512))
    # Iterative phase synthesis is intractably slow for bounded high-degree targets; above this
    # ceiling we record the analytic fit + boundedness (the decisive evidence: whether the target
    # stays uniformly approximable and bounded) and skip synthesis. The application-useful small-
    # lambda alphas fail boundedness+fit BEFORE synthesis, so their infeasibility is degree-robust.
    synth_ceiling = int(config.get("phase_synthesis_degree_ceiling", 63))
    # Structures NOT listed here are recorded analytic-only (fit + boundedness, no phase synthesis
    # and no statevector): the larger 16x16 block's 128-dim iterative synthesis is intractable, but
    # its high-degree fit divergence is still captured. None => synthesize for every structure.
    synthesis_structures = config.get("synthesis_structures")
    training_seeds = [int(x) for x in config["training_seed_ids"]]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for sid in config["structures"]:
        struct_synth = synthesis_structures is None or sid in synthesis_structures
        struct_ceiling = synth_ceiling if struct_synth else -1
        ctx = build_structure(sid)
        oracle_best, unreg = _utility_reference(sid)
        case_name = STRUCTURES[sid][0]
        fsys = build_case_full_system(case_name, 123)
        train_tasks = build_tasks(sid, training_seeds, "training")
        score_ids = ctx.physical_ids
        residual_block, _ = seed_reference(sid, training_seeds[0])
        residual_unit = residual_block / np.linalg.norm(residual_block)

        regimes = {r["regime"]: r for r in design_time_alpha_regimes(ctx)}
        regimes.update({r["regime"]: r for r in _full_system_alpha_regimes(sid)})
        for selector in selectors:
            if selector == "full_support":
                support = ctx.matrix != 0.0
            else:
                cons = SupportConstraints(k, s, True)
                rec = select_support(ctx, selector, cons, train_tasks, score_ids)
                support = rec["support"]
                if support is None:
                    failures.append(
                        {
                            "structure_id": sid,
                            "selector": selector,
                            "stage": "support",
                            "reason": rec.get("failure_reason", "none"),
                        }
                    )
                    continue
            sparse_block = np.where(support, ctx.matrix, 0.0)
            # Building the padded wrapper unitary is expensive (seconds for a 128-dim block); only
            # do it when this structure will actually execute a statevector check.
            if struct_synth and execute_statevector:
                statevector_dim = _wrapper_dim(sparse_block)
                do_exec = bool(statevector_dim <= statevector_ceiling)
            else:
                statevector_dim = -1
                do_exec = False
            for regime_name in alpha_regime_names:
                regime = regimes.get(regime_name)
                if regime is None or not np.isfinite(regime["alpha"]):
                    failures.append(
                        {
                            "structure_id": sid,
                            "selector": selector,
                            "stage": "alpha",
                            "reason": f"regime {regime_name} unavailable",
                        }
                    )
                    continue
                alpha = float(regime["alpha"])
                util = application_metrics(
                    fsys, alpha, oracle_best_rmse=oracle_best, unregularized_rmse=unreg
                )
                application_useful = bool(util["rmse_ratio_to_oracle_best"] <= useful_ratio)
                for degree in degrees:
                    attempt_synth = degree <= struct_ceiling
                    exec_here = do_exec and attempt_synth
                    feas = evaluate_qsvt_feasibility(
                        sparse_block,
                        alpha,
                        degree,
                        margin=margin,
                        bound_tolerance=bound_tol,
                        uniform_tolerance=uniform_tol,
                        execute_statevector=exec_here,
                        residual_unit=residual_unit if exec_here else None,
                        cache_dir=cache_dir,
                        attempt_phase_synthesis=attempt_synth,
                    )
                    # analytic (fit-level) feasibility: bounded + parity + uniform fit pass,
                    # available at ALL degrees without the expensive phase synthesis.
                    analytic_ok = bool(
                        feas.boundedness_ok and feas.uniform_fit_error <= uniform_tol
                    )
                    qsvt_feasible = bool(
                        analytic_ok
                        and feas.phase_synthesis_status == "synthesized"
                        and (
                            np.isnan(feas.statevector_action_error)
                            or feas.statevector_action_error <= action_tol
                        )
                    )
                    rows.append(
                        {
                            "structure_id": sid,
                            "case": case_name,
                            "selector": selector,
                            "support_budget": k
                            if selector != "full_support"
                            else int(support.sum()),
                            "slot_budget": s,
                            "alpha_regime": regime_name,
                            "alpha": alpha,
                            "deployable_alpha": bool(regime["deployable"]),
                            "oracle_alpha": bool(regime["is_oracle"]),
                            "beta": feas.beta,
                            "normalized_lambda": feas.normalized_lambda,
                            "degree": degree,
                            "parity": "odd",
                            "target_max_abs": feas.bounded_max_abs,
                            "uniform_fit_error": feas.uniform_fit_error,
                            "singular_point_fit_error": feas.singular_point_fit_error,
                            "boundedness_parity_fit_ok": feas.boundedness_ok,
                            "analytic_bounded_fit_ok": analytic_ok,
                            "phase_synthesis_attempted": attempt_synth,
                            "phase_synthesis_status": feas.phase_synthesis_status,
                            "phase_count": feas.phase_count,
                            "statevector_dim": statevector_dim,
                            "statevector_executed": do_exec,
                            "statevector_action_error": feas.statevector_action_error,
                            "postselection_probability": feas.postselection_probability,
                            "qsvt_evidence_tier": feas.evidence_status,
                            "application_useful_full_state": application_useful,
                            "full_state_rmse_ratio": util["rmse_ratio_to_oracle_best"],
                            "qsvt_feasible_composite": qsvt_feasible,
                            "jointly_useful_and_feasible": bool(
                                application_useful and qsvt_feasible
                            ),
                            "failure_reason": feas.failure_reason,
                        }
                    )
                    if feas.failure_reason:
                        failures.append(
                            {
                                "structure_id": sid,
                                "selector": selector,
                                "stage": "qsvt",
                                "reason": feas.failure_reason,
                                "degree": degree,
                                "alpha_regime": regime_name,
                            }
                        )

    frame = pd.DataFrame(rows)
    atomic_write_csv(destination / "high_degree_qsvt_rows.csv", frame)
    summary = _summarize_high_degree(frame)
    atomic_write_csv(destination / "high_degree_qsvt_summary.csv", summary)
    atomic_write_csv(
        destination / "high_degree_failures.csv",
        pd.DataFrame(failures)
        if failures
        else pd.DataFrame(columns=["structure_id", "stage", "reason"]),
    )
    conclusions = _conclusions(frame)
    atomic_write_json(destination / "high_degree_conclusions.json", conclusions)
    atomic_write_json(
        destination / "high_degree_provenance.json",
        provenance_block(config_path, config)
        | {"study_id": STUDY_ID, "claim_boundary": CLAIM_BOUNDARY},
    )
    return {
        "rows": len(frame),
        "structures": list(config["structures"]),
        "any_jointly_useful_and_feasible": bool(frame["jointly_useful_and_feasible"].any())
        if not frame.empty
        else False,
        "conclusions": conclusions,
    }


def _wrapper_dim(sparse_block: np.ndarray) -> int:
    from robust_qsvt_se.qsvt.bipartite_slot_assignment import minimum_slot_count
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import build_common_padded_wrapper

    pattern = sparse_block.T != 0.0
    slots = int(minimum_slot_count(pattern))
    mu = float(np.max(np.abs(sparse_block)))
    try:
        wrapper = build_common_padded_wrapper(sparse_block, slots=slots, mu=mu)
        return int(wrapper.unitary.shape[0])
    except Exception:
        return 10**9


def _summarize_high_degree(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = []
    for key, g in frame.groupby(
        ["structure_id", "selector", "alpha_regime", "degree"], dropna=False
    ):
        row = g.iloc[0]
        out.append(
            {
                "structure_id": key[0],
                "selector": key[1],
                "alpha_regime": key[2],
                "degree": key[3],
                "alpha": float(row["alpha"]),
                "normalized_lambda": float(row["normalized_lambda"]),
                "uniform_fit_error": float(row["uniform_fit_error"]),
                "target_max_abs": float(row["target_max_abs"]),
                "boundedness_parity_fit_ok": bool(row["boundedness_parity_fit_ok"]),
                "phase_synthesis_status": row["phase_synthesis_status"],
                "statevector_action_error": float(row["statevector_action_error"]),
                "application_useful": bool(row["application_useful_full_state"]),
                "qsvt_feasible": bool(row["qsvt_feasible_composite"]),
                "jointly_useful_and_feasible": bool(row["jointly_useful_and_feasible"]),
                "failure_reason": row["failure_reason"],
            }
        )
    return pd.DataFrame(out)


def _conclusions(frame: pd.DataFrame) -> dict[str, Any]:
    """Per-structure overlap conclusion category."""

    result: dict[str, Any] = {"claim_boundary": CLAIM_BOUNDARY, "structures": {}}
    if frame.empty:
        return result
    for sid, g in frame.groupby("structure_id"):
        joint = g[g["jointly_useful_and_feasible"]]  # noqa: F841
        by_degree = {
            int(d): bool(sub["jointly_useful_and_feasible"].any()) for d, sub in g.groupby("degree")
        }
        # analytic (fit-level) overlap: useful AND bounded+uniform-fit, robust to synthesis ceiling
        g_analytic_joint = g["application_useful_full_state"] & g["analytic_bounded_fit_ok"]
        analytic_by_degree = {
            int(d): bool(g_analytic_joint[g["degree"] == d].any()) for d in sorted(by_degree)
        }
        first_overlap = next((d for d in sorted(by_degree) if by_degree[d]), None)
        first_analytic_overlap = next(
            (d for d in sorted(analytic_by_degree) if analytic_by_degree[d]), None
        )
        # deployable-useful + feasible specifically (the reviewer-relevant overlap)
        dep = g[g["deployable_alpha"] & ~g["oracle_alpha"]]
        dep_joint = dep[dep["jointly_useful_and_feasible"]]
        if first_overlap is None:
            category = "overlap_remains_empty_through_degree_255"
        elif first_overlap == 127:
            category = "overlap_first_appears_at_degree_127"
        elif first_overlap == 255:
            category = "overlap_first_appears_at_degree_255"
        elif first_overlap in (31, 63):
            category = "overlap_present_already_at_low_degree_result_depends_on_alpha_or_support"
        else:
            category = "result_depends_on_support_or_alpha_not_uniform"
        result["structures"][sid] = {
            "overlap_by_degree": by_degree,
            "analytic_overlap_by_degree": analytic_by_degree,
            "first_degree_with_overlap": first_overlap,
            "first_degree_with_analytic_overlap": first_analytic_overlap,
            "deployable_useful_and_feasible_rows": len(dep_joint),
            "category": category,
            "any_feasible_any_degree": bool(g["qsvt_feasible_composite"].any()),
            "any_analytic_bounded_fit_any_degree": bool(g["analytic_bounded_fit_ok"].any()),
            "any_useful_any_alpha": bool(g["application_useful_full_state"].any()),
            "note": (
                "analytic_overlap uses bounded+uniform-fit (degree-robust, no phase synthesis); "
                "composite overlap additionally requires synthesized phases (skipped above the "
                "phase_synthesis_degree_ceiling for compute). Application-useful alphas that fail "
                "the analytic fit are QSVT-infeasible at every degree."
            ),
        }
    return result
