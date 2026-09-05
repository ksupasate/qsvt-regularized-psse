"""Workstream B - nonlinear AC QSVT circuit-in-the-loop.

A complete small-scale nonlinear AC Gauss-Newton loop in which the residual r_k = z - h(x_k) and the
weighted Jacobian H_k are rebuilt at every iteration; a small block is refreshed, its bounded QSVT
target and phases are (re)synthesized, an explicit QSVT circuit is built and executed in classical
statevector simulation, and the recovered selected output is compared against the matched block
Ridge update, the exact rational action, and the exact polynomial matrix action - all at the
identical operating point (x_k, r_k, H_k, block, support, alpha_k, beta_k, C_k, phases,
functional).  A declared finite-shot subset genuinely samples the postselection acceptance with Aer.

Reuses ``build_ac_nonlinear_problem`` + ``_linearized_update_system`` (per-iteration rebuild),
``fit_codesigned_bounded_polynomial`` (bounded target + boundedness factor), the timeout-guarded
phase synthesis from Workstream A, and the ``build_common_padded_wrapper`` /
``build_structured_qsvt_operator_circuit`` statevector pathway.  The strongest claim supported is a
completed classical statevector circuit-in-the-loop path for the declared IEEE-14-derived workloads;
no hardware execution, full-system quantum recovery, scalability, or competitiveness is claimed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.experiments.iterative_ac import (
    _linearized_update_system,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.physical_alignment.nonlinear_ac import build_problem_config
from robust_qsvt_se.qsvt.bipartite_slot_assignment import minimum_slot_count
from robust_qsvt_se.qsvt.output_aware_sparse_selection import _ridge_filter_operator
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint
from robust_qsvt_se.reviewer_blocking.common import (
    atomic_write_csv,
    atomic_write_json,
    provenance_block,
    write_manifest_and_checksums,
)
from robust_qsvt_se.tqe_extensions.common import CLAIM_BOUNDARY, EVIDENCE_TIERS, load_yaml_config
from robust_qsvt_se.tqe_extensions.degree_lambda_scaling import _synthesize_guarded

STUDY_ID = "tqe_nonlinear_qsvt_circuit_loop_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/tqe_nonlinear_qsvt_circuit_loop")
DEFAULT_CONFIG_PATH = Path("configs/tqe_nonlinear_qsvt_circuit_loop.yaml")


# --------------------------------------------------------------------------- block QSVT


@dataclass(slots=True)
class BlockQSVT:
    sparse_block: np.ndarray
    residual_block: np.ndarray
    functional: np.ndarray
    beta: float
    slots: int
    mu: float
    alpha_k: float
    lambda_k: float
    contraction_c: float
    degree: int
    coefficients: np.ndarray
    domain_min: float
    uniform_fit_error: float
    bounded_ok: bool
    rank: int
    kappa: float


def _magnitude_support(block: np.ndarray, keep_per_row: int) -> np.ndarray:
    """Deterministic per-row top-``keep_per_row`` magnitude support (frozen sparsification "
    "policy)."""

    mask = np.zeros_like(block, dtype=bool)
    for i in range(block.shape[0]):
        order = np.lexsort((np.arange(block.shape[1]), -np.abs(block[i])))
        for j in order[:keep_per_row]:
            if block[i, j] != 0.0:
                mask[i, j] = True
    return mask


def build_block_qsvt(
    matrix: np.ndarray,
    residual: np.ndarray,
    *,
    block_size: int,
    keep_per_row: int,
    lambda_target: float,
    degree: int,
    margin: float,
    functional_index: int,
) -> BlockQSVT:
    block, rblock, _rows, _cols = select_deterministic_block(
        matrix, residual, row_count=block_size, col_count=block_size
    )
    support = _magnitude_support(block, keep_per_row)
    sparse = np.where(support, block, 0.0)
    pattern = sparse.T != 0.0
    slots = int(minimum_slot_count(pattern))
    mu = float(np.max(np.abs(sparse)))
    beta = float(slots * mu)
    singular = np.linalg.svd(sparse, compute_uv=False)
    positive = singular[singular > 1e-10]
    rank = int(positive.size)
    kappa = float(positive.max() / positive.min()) if rank else float("inf")
    domain_min = float(np.clip(0.9 * positive.min() / beta, 1e-4, 0.999)) if rank else 1e-3
    alpha_k = float(lambda_target) * beta**2
    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=alpha_k,
        domain_min=domain_min,
        domain_max=1.0,
        degree=int(degree),
        margin=float(margin),
    )
    functional = np.zeros(sparse.shape[1], dtype=np.float64)
    functional[int(functional_index) % sparse.shape[1]] = 1.0
    return BlockQSVT(
        sparse_block=sparse,
        residual_block=np.asarray(rblock, np.float64),
        functional=functional,
        beta=beta,
        slots=slots,
        mu=mu,
        alpha_k=alpha_k,
        lambda_k=float(target.alpha_normalized),
        contraction_c=float(target.bound_C),
        degree=int(degree),
        coefficients=np.asarray(target.coefficients, np.float64),
        domain_min=domain_min,
        uniform_fit_error=float(target.fit_max_abs_error),
        bounded_ok=bool(target.bounded_max_abs <= 1.0 + 2e-3),
        rank=rank,
        kappa=kappa,
    )


def _svd_action(bq: BlockQSVT, values: np.ndarray) -> np.ndarray:
    """Diagonal-filter action ``values(s)`` on the block via the ``B^T/beta`` block-encoding SVD."""

    normalized = bq.sparse_block.T / bq.beta
    left, s, right_t = np.linalg.svd(normalized, full_matrices=False)
    unit = bq.residual_block / max(np.linalg.norm(bq.residual_block), 1e-30)
    return left @ (values(s) * (right_t @ unit))


def block_reference_outputs(bq: BlockQSVT) -> dict[str, float]:
    """Matched block Ridge (= exact rational) and exact polynomial-action selected outputs at "
    "alpha_k."""

    rblock_norm = float(np.linalg.norm(bq.residual_block))
    x_ridge = _ridge_filter_operator(bq.sparse_block, bq.alpha_k) @ bq.residual_block
    y_ridge = float(bq.functional @ x_ridge)  # matched block Ridge = exact rational spectral action
    recovery = bq.contraction_c / bq.beta
    poly = Polynomial(bq.coefficients)
    poly_encoded = _svd_action(bq, poly)  # exact polynomial matrix action (no circuit)
    y_poly = float(bq.functional @ (recovery * poly_encoded * rblock_norm))
    return {
        "residual_block_norm": rblock_norm,
        "y_block_ridge_exact_rational": y_ridge,
        "y_exact_polynomial_action": y_poly,
        "physical_recovery_factor": recovery,
    }


def statevector_selected_output(bq: BlockQSVT, phases: np.ndarray) -> dict[str, float]:
    """Execute the explicit QSVT statevector circuit and recover the selected output + "
    "postselection."""

    from qiskit.quantum_info import Statevector

    from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import build_common_padded_wrapper

    block = bq.sparse_block
    wrapper = build_common_padded_wrapper(block, slots=bq.slots, mu=bq.mu)
    bundle = build_structured_qsvt_operator_circuit(
        wrapper.unitary, phases, encoded_dimension=block.shape[1]
    )
    rblock_norm = float(np.linalg.norm(bq.residual_block))
    unit = bq.residual_block / max(rblock_norm, 1e-30)
    initial = np.zeros(wrapper.unitary.shape[0], dtype=np.complex128)
    initial[: block.shape[0]] = unit
    evolved = Statevector(initial).evolve(bundle.qsvt_operator_circuit).data
    encoded = np.asarray(evolved[: block.shape[1]], dtype=np.complex128)
    p_post = float(np.vdot(encoded, encoded).real)
    recovery = bq.contraction_c / bq.beta
    y_circuit = float(bq.functional @ (recovery * np.real(encoded) * rblock_norm))
    return {
        "statevector_dim": int(wrapper.unitary.shape[0]),
        "postselection_probability_executed": p_post,
        "y_circuit_statevector": y_circuit,
        "circuit": bundle.qsvt_operator_circuit,
        "initial": initial,
        "wrapper_unitary_dim": int(wrapper.unitary.shape[0]),
        "work_qubits": int(np.log2(block.shape[1])) if block.shape[1] > 1 else 0,
        "total_qubits": int(np.log2(wrapper.unitary.shape[0])),
    }


# --------------------------------------------------------------------------- finite shots


def finite_shot_postselection(
    circuit: Any, initial: np.ndarray, cols: int, shot_grid: list[int], base_seed: int
) -> list[dict[str, Any]]:
    """Aer finite-shot sampling of the postselection acceptance (accepted = block index < cols).

    The residual state preparation is prepended so the sampled distribution matches the executed
    statevector; the selected-output amplitude readout would additionally require a Hadamard-test
    circuit and is
    recorded as resource-limited rather than modeled; the acceptance statistics are executed here.
    """

    from qiskit import QuantumCircuit

    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        compile_for_aer,
        sample_aer_counts,
    )

    n_qubits = circuit.num_qubits
    measured = QuantumCircuit(n_qubits)
    measured.initialize(np.asarray(initial, dtype=np.complex128), range(n_qubits))
    measured.compose(circuit, inplace=True)
    measured.measure_all()
    compiled, simulator = compile_for_aer(measured)
    rows: list[dict[str, Any]] = []
    for shots in shot_grid:
        counts = sample_aer_counts(
            compiled, simulator, shots=int(shots), seed=base_seed + int(shots)
        )
        attempted = int(sum(counts.values()))
        accepted = int(sum(v for b, v in counts.items() if int(b.replace(" ", ""), 2) < cols))
        rows.append(
            {
                "attempted_shots": attempted,
                "accepted_shots": accepted,
                "postselection_probability_finite_shot": accepted / max(attempted, 1),
                "selected_output_shot_readout_status": "resource_limited_requires_hadamard_test",
            }
        )
    return rows


# --------------------------------------------------------------------------- loop


def run_one_workload(
    scenario: dict[str, Any],
    seed: int,
    settings: dict[str, Any],
    wsb: dict[str, Any],
    cache_dir: Path,
    *,
    do_finite_shots: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    problem = build_ac_nonlinear_problem(build_problem_config(settings, scenario, seed))
    cap = int(settings["iteration"]["max_iterations"])
    update_tol = float(settings["iteration"]["update_tolerance"])
    residual_tol = float(settings["iteration"]["residual_tolerance"])
    damping = float(settings["iteration"]["damping"])
    alpha_full = float(settings["fixed_alpha"])
    finite_shot_iterations = set(int(i) for i in wsb.get("finite_shot_iterations", [1]))
    shot_grid = [int(s) for s in wsb.get("finite_shot_counts", [10000, 100000])]

    state = problem.initial_state.copy()
    n_true = float(np.linalg.norm(problem.true_state))
    angle_count = len(problem.case.angle_state_buses)
    rows: list[dict[str, Any]] = []
    shot_rows: list[dict[str, Any]] = []
    converged = False
    for iteration in range(cap):
        system, residual_norm = _linearized_update_system(problem, state)
        H = np.asarray(system.H_tilde, dtype=np.float64)
        r = np.asarray(system.r_tilde, dtype=np.float64)
        x_remaining = np.asarray(system.x_true, dtype=np.float64)  # true_state - state
        dx_full = _ridge_filter_operator(H, alpha_full) @ r
        singular = np.linalg.svd(H, compute_uv=False)
        pos = singular[singular > 1e-14]
        kappa_h = float(pos.max() / pos.min()) if pos.size else float("inf")

        bq = build_block_qsvt(
            H,
            r,
            block_size=int(wsb["block_size"]),
            keep_per_row=int(wsb["keep_per_row"]),
            lambda_target=float(wsb["lambda_target"]),
            degree=int(wsb["degree"]),
            margin=float(wsb.get("margin", 1.05)),
            functional_index=int(wsb.get("functional_index", 0)),
        )
        refs = block_reference_outputs(bq)
        # QSVT matrix-action error = exact polynomial action vs exact rational (Ridge) selected
        # output.
        y_ridge = refs["y_block_ridge_exact_rational"]
        y_poly = refs["y_exact_polynomial_action"]
        floor = float(wsb.get("selected_output_floor", 1e-6))
        matrix_action_error = abs(y_poly - y_ridge) / max(abs(y_ridge), floor)

        phases, phase_status = (
            _synthesize_guarded(
                bq.coefficients,
                angle_solver="iterative",
                cache_dir=cache_dir,
                meta={"study_id": STUDY_ID, "degree": bq.degree, "beta": round(bq.beta, 6)},
                timeout_s=float(wsb.get("phase_synthesis_timeout_seconds", 40.0)),
            )
            if bq.bounded_ok
            else (None, "skipped_unbounded")
        )

        row = {
            "scenario": scenario["scenario_id"],
            "seed": int(seed),
            "iteration": int(iteration),
            "state_fingerprint": stable_array_fingerprint(state),
            "residual_fingerprint": stable_array_fingerprint(r),
            "jacobian_fingerprint": stable_array_fingerprint(H),
            "weighted_jacobian_fingerprint": stable_array_fingerprint(H),
            "jacobian_rebuilt_this_iteration": True,
            "residual_rebuilt_this_iteration": True,
            "block_fingerprint": stable_array_fingerprint(bq.sparse_block),
            "support_nnz": int((bq.sparse_block != 0.0).sum()),
            "block_rank": bq.rank,
            "kappa_H": kappa_h,
            "kappa_block": bq.kappa,
            "alpha_full": alpha_full,
            "alpha_k": bq.alpha_k,
            "beta_k": bq.beta,
            "lambda_k": bq.lambda_k,
            "contraction_C_k": bq.contraction_c,
            "degree": bq.degree,
            "polynomial_fit_error": bq.uniform_fit_error,
            "bounded_ok": bq.bounded_ok,
            "qsvt_matrix_action_error": matrix_action_error,
            "y_block_ridge_exact_rational": y_ridge,
            "y_exact_polynomial_action": y_poly,
            "physical_recovery_factor": refs["physical_recovery_factor"],
            "phase_synthesis_status": phase_status,
            "phase_count": int(phases.size) if phases is not None else 0,
            "state_rmse": float(np.linalg.norm(x_remaining) / max(n_true, 1e-30)),
            "angle_rmse": float(np.sqrt(np.mean(x_remaining[:angle_count] ** 2))),
            "voltage_rmse": float(np.sqrt(np.mean(x_remaining[angle_count:] ** 2))),
            "weighted_residual_norm": residual_norm,
            "update_norm": float(np.linalg.norm(dx_full)),
            "evidence_tier": EVIDENCE_TIERS["EXACT_MATRIX_ACTION"],
            "failure_code": "",
            "circuit_qubits": -1,
            "circuit_gates": -1,
            "circuit_depth": -1,
            "circuit_toffoli": -1,
            "statevector_action_error": float("nan"),
            "circuit_vs_polynomial_error": float("nan"),
            "circuit_vs_ridge_error": float("nan"),
            "y_circuit_statevector": float("nan"),
            "postselection_probability_executed": float("nan"),
        }
        if phases is not None and phase_status == "synthesized" and phases.size == bq.degree + 1:
            t0 = time.perf_counter()
            try:
                sv = statevector_selected_output(bq, phases)
                y_circuit = sv["y_circuit_statevector"]
                row.update(
                    {
                        "statevector_dim": sv["statevector_dim"],
                        "y_circuit_statevector": y_circuit,
                        "postselection_probability_executed": sv[
                            "postselection_probability_executed"
                        ],
                        "circuit_vs_polynomial_error": abs(y_circuit - y_poly)
                        / max(abs(y_poly), floor),
                        "circuit_vs_ridge_error": abs(y_circuit - y_ridge)
                        / max(abs(y_ridge), floor),
                        "evidence_tier": EVIDENCE_TIERS["STATEVECTOR"],
                        "total_qubits": sv["total_qubits"],
                        "circuit_qubits": sv["total_qubits"],
                        "statevector_execution_seconds": time.perf_counter() - t0,
                    }
                )
                res = _circuit_resources(sv["circuit"])
                row.update(
                    {
                        "circuit_gates": res["gate_count"],
                        "circuit_depth": res["depth"],
                        "circuit_toffoli": res["toffoli_count"],
                    }
                )
                if do_finite_shots and iteration in finite_shot_iterations:
                    for sr in finite_shot_postselection(
                        sv["circuit"],
                        sv["initial"],
                        bq.sparse_block.shape[1],
                        shot_grid,
                        base_seed=1000 + seed,
                    ):
                        shot_rows.append(
                            {
                                **{
                                    k: row[k]
                                    for k in (
                                        "scenario",
                                        "seed",
                                        "iteration",
                                        "alpha_k",
                                        "beta_k",
                                        "lambda_k",
                                        "degree",
                                    )
                                },
                                "postselection_probability_executed": sv[
                                    "postselection_probability_executed"
                                ],
                                **sr,
                                "evidence_tier": EVIDENCE_TIERS["FINITE_SHOT"],
                            }
                        )
            except Exception as exc:
                row["failure_code"] = "statevector_failure"
                row["failure_reason"] = str(exc)[:200]
        elif phase_status not in {"synthesized"}:
            row["failure_code"] = phase_status

        rows.append(row)
        state = state + damping * dx_full
        if float(np.linalg.norm(dx_full)) < update_tol or residual_norm < residual_tol:
            converged = True
            rows[-1]["convergence_status"] = "converged"
            break
        rows[-1]["convergence_status"] = "running"
    for row in rows:
        row.setdefault("convergence_status", "max_iterations_reached")
        row["run_converged"] = converged
    return rows, shot_rows


def _circuit_resources(circuit: Any) -> dict[str, int]:
    from robust_qsvt_se.qsvt.sparse_integrated_chain import _resource_counts, compile_for_aer

    compiled, _sim = compile_for_aer(circuit)
    return _resource_counts(compiled)


# --------------------------------------------------------------------------- orchestrator


def run_nonlinear_circuit_loop(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"config study_id mismatch: {config.get('study_id')!r}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    settings = config["nonlinear_settings"]
    scenarios = config["scenarios"]
    seeds = [int(s) for s in config["seeds"]]
    wsb = config["block_qsvt"]
    finite_shot_seed = int(config.get("finite_shot_seed", seeds[0]))

    all_rows, all_shots, workloads = [], [], []
    for scenario in scenarios:
        for seed in seeds:
            do_shots = int(seed) == finite_shot_seed
            rows, shots = run_one_workload(
                scenario, seed, settings, wsb, cache_dir, do_finite_shots=do_shots
            )
            all_rows.extend(rows)
            all_shots.extend(shots)
            workloads.append(
                {
                    "scenario": scenario["scenario_id"],
                    "seed": int(seed),
                    "iterations_run": len(rows),
                    "finite_shots_attempted": do_shots,
                    "converged": bool(rows[-1].get("run_converged", False)) if rows else False,
                }
            )
            if progress:
                print(
                    f"[nonlinear_loop] {scenario['scenario_id']} seed {seed}: {len(rows)} iters",
                    flush=True,
                )

    per_iter = pd.DataFrame(all_rows)
    atomic_write_csv(destination / "per_iteration_results.csv", per_iter)
    atomic_write_csv(destination / "workload_registry.csv", pd.DataFrame(workloads))

    sv_rows = per_iter[per_iter["evidence_tier"] == EVIDENCE_TIERS["STATEVECTOR"]]
    atomic_write_csv(destination / "statevector_execution_rows.csv", sv_rows)
    atomic_write_csv(
        destination / "finite_shot_rows.csv",
        pd.DataFrame(all_shots)
        if all_shots
        else pd.DataFrame(columns=["scenario", "seed", "iteration", "attempted_shots"]),
    )

    convergence = _convergence_summary(per_iter)
    atomic_write_csv(destination / "convergence_summary.csv", convergence)
    atomic_write_csv(destination / "error_decomposition.csv", _error_decomposition(per_iter))
    atomic_write_csv(destination / "resource_summary.csv", _resource_summary(sv_rows))

    failures = per_iter[per_iter["failure_code"].astype(str) != ""][
        ["scenario", "seed", "iteration", "failure_code", "phase_synthesis_status", "evidence_tier"]
    ]
    atomic_write_csv(
        destination / "failure_registry.csv",
        failures
        if not failures.empty
        else pd.DataFrame(columns=["scenario", "seed", "iteration", "failure_code"]),
    )

    claim = _claim_support(per_iter, sv_rows, all_shots, workloads)
    atomic_write_json(destination / "claim_support.json", claim)
    atomic_write_json(
        destination / "run_manifest.json",
        provenance_block(config_path, config) | {"study_id": STUDY_ID, "iterations": len(per_iter)},
    )
    _write_resolved_config(destination, config)
    _write_readme(destination, per_iter, sv_rows, all_shots, workloads)
    write_manifest_and_checksums(
        destination,
        study_id=STUDY_ID,
        extra={"workloads": len(workloads), "iterations": len(per_iter)},
    )
    return {
        "workloads": len(workloads),
        "iterations": len(per_iter),
        "statevector_iterations": len(sv_rows),
        "finite_shot_rows": len(all_shots),
        "converged_runs": int(sum(w["converged"] for w in workloads)),
        "failures": len(failures),
    }


def _convergence_summary(per_iter: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (scenario, seed), block in per_iter.groupby(["scenario", "seed"]):
        block = block.sort_values("iteration")
        final = block.iloc[-1]
        out.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "iterations": len(block),
                "converged": bool(final.get("run_converged", False)),
                "final_state_rmse": float(final["state_rmse"]),
                "final_weighted_residual": float(final["weighted_residual_norm"]),
                "final_update_norm": float(final["update_norm"]),
                "statevector_iterations": int(
                    (block["evidence_tier"] == EVIDENCE_TIERS["STATEVECTOR"]).sum()
                ),
            }
        )
    return pd.DataFrame(out)


def _error_decomposition(per_iter: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scenario",
        "seed",
        "iteration",
        "polynomial_fit_error",
        "qsvt_matrix_action_error",
        "circuit_vs_polynomial_error",
        "circuit_vs_ridge_error",
        "postselection_probability_executed",
        "evidence_tier",
    ]
    return per_iter[[c for c in cols if c in per_iter.columns]].copy()


def _resource_summary(sv_rows: pd.DataFrame) -> pd.DataFrame:
    if sv_rows.empty:
        return pd.DataFrame(columns=["scenario", "seed", "iteration", "circuit_gates"])
    return sv_rows[
        [
            "scenario",
            "seed",
            "iteration",
            "total_qubits",
            "circuit_gates",
            "circuit_depth",
            "circuit_toffoli",
            "statevector_dim",
            "statevector_execution_seconds",
        ]
    ].copy()


def _write_resolved_config(destination: Path, config: dict[str, Any]) -> None:
    import yaml

    (destination / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, default_flow_style=False), encoding="utf-8"
    )


def _claim_support(per_iter, sv_rows, shots, workloads) -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "declared_workloads": len(workloads),
        "attempted_scenarios_x_seeds": len(workloads),
        "iterations_total": len(per_iter),
        "jacobian_rebuilt_every_iteration": bool(per_iter["jacobian_rebuilt_this_iteration"].all())
        if not per_iter.empty
        else False,
        "residual_rebuilt_every_iteration": bool(per_iter["residual_rebuilt_this_iteration"].all())
        if not per_iter.empty
        else False,
        "statevector_executed_iterations": len(sv_rows),
        "every_bounded_iteration_has_statevector": bool(
            (
                per_iter[
                    per_iter["bounded_ok"] & (per_iter["phase_synthesis_status"] == "synthesized")
                ]["evidence_tier"]
                == EVIDENCE_TIERS["STATEVECTOR"]
            ).all()
        )
        if not per_iter.empty
        else False,
        "finite_shot_rows": len(shots),
        "finite_shot_selected_output_status": "resource_limited_requires_hadamard_test",
        "rational_polynomial_statevector_shot_tiers_separated": True,
        "allowed_claim": (
            "A complete small-scale nonlinear circuit-in-the-loop path was executed in classical "
            "statevector simulation for the declared IEEE-14-derived workloads, rebuilding the "
            "residual, weighted Jacobian, selected block, bounded QSVT target, and explicit "
            "circuit "
            "at each retained iteration; genuine finite-shot Aer sampling quantifies the "
            "postselection acceptance for the declared subset."
        ),
        "forbidden_claims": [
            "quantum hardware execution",
            "full IEEE-14 quantum state recovery",
            "scalable nonlinear QSVT PSSE",
            "practical competitiveness",
            "QSVT beats matched Ridge",
        ],
    }


def _write_readme(destination: Path, per_iter, sv_rows, shots, workloads) -> None:
    lines = [
        "# Workstream B - Nonlinear AC QSVT Circuit-in-the-Loop",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- workloads: {len(workloads)} (scenarios x seeds); iterations: {len(per_iter)}; "
        f"statevector-executed iterations: {len(sv_rows)}; finite-shot rows: {len(shots)}",
        "",
        "## Method",
        "At each Gauss-Newton iteration the residual r_k = z - h(x_k) and weighted Jacobian H_k "
        "are "
        "rebuilt from the raw measurements; a small block is refreshed, its bounded QSVT target "
        "and "
        "phases (re)synthesized, and an explicit QSVT circuit executed in statevector simulation. "
        "The "
        "recovered selected output is compared - at the identical operating point - against the "
        "matched block Ridge update (= exact rational spectral action) and the exact polynomial "
        "matrix action. Evidence tiers (rational / polynomial / statevector / finite-shot) are "
        "kept "
        "separate; the nonlinear state advance uses the matched full Ridge update (QSVT recovers a "
        "selected block output, not the full state).",
        "",
        "## Files",
        "- `per_iteration_results.csv`, `statevector_execution_rows.csv`, `finite_shot_rows.csv`",
        "- `convergence_summary.csv`, `error_decomposition.csv`, `resource_summary.csv`",
        "- `workload_registry.csv`, `failure_registry.csv`",
        "- `claim_support.json`, `run_manifest.json`, `config_resolved.yaml`, `checksums.sha256`",
        "",
        "## Reproduce",
        "```",
        "MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 \\",
        "  .venv/bin/python scripts/run_tqe_nonlinear_qsvt_circuit_loop.py",
        "```",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
