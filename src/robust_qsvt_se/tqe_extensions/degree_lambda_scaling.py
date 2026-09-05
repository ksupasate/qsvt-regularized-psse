"""Workstream A - systematic degree / normalized-regularization / target-tolerance feasibility map.

For a frozen grid of normalized regularizations ``lambda``, target tolerances ``epsilon_target`` and
odd degrees ``d`` we record, over six declared matrix scopes, whether the bounded QSVT target

    f_{lambda,C}(s) = (1/C) * s / (s^2 + lambda),   C = margin * max_{s in [s_lo,1]} s/(s^2+lambda),

is (i) approximable to ``epsilon_target`` by a degree-``d`` odd polynomial, (ii) bounded by
1 on ``[-1,1]``, (iii) synthesizable into ``d+1`` QSVT phases, and - on executable small blocks -
(iv) reproduced by an explicit statevector circuit.  The empirical map ``d_min(lambda,epsilon)`` is
extracted reproducibly.  Every scalar / exact-matrix-action / statevector tier is kept separate; no
grid point is added or removed after evaluation; failures are retained with an explicit code.

Reuses the frozen primitives ``fit_codesigned_bounded_polynomial`` (target + boundedness factor),
``synthesize_pennylane_phases_cached`` + ``qsp_response`` (phase synthesis + convention transfer),
and the ``build_common_padded_wrapper`` / ``build_structured_qsvt_operator_circuit`` statevector
pathway.  No QSVT or Ridge definition is modified.
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.reviewer_blocking.common import provenance_block
from robust_qsvt_se.tqe_extensions.common import (
    CLAIM_BOUNDARY,
    EVIDENCE_TIERS,
    analytic_bound_c,
    atomic_write_csv,
    atomic_write_json,
    extract_min_feasible_degree,
    load_yaml_config,
    write_manifest_and_checksums,
)

STUDY_ID = "tqe_degree_lambda_error_scaling_v1"
DEFAULT_OUTPUT_DIR = Path("outputs/tqe_degree_lambda_error_scaling")
DEFAULT_CONFIG_PATH = Path("configs/tqe_degree_lambda_error_scaling.yaml")


# --------------------------------------------------------------------------- scopes


@dataclass(slots=True)
class MatrixScope:
    """One evaluation scope: a normalized filter domain plus an optional matrix spectrum."""

    scope_id: str
    label: str
    kind: str  # "scalar" | "sparse_block" | "full_jacobian"
    beta: float
    slots: int
    mu: float
    domain_min: float
    singular_values_norm: np.ndarray | None  # positive s_i = sigma_i / beta in (0, 1]
    residual_unit: np.ndarray | None  # measurement-space unit residual (matrix scopes)
    sparse_block: np.ndarray | None  # only for executable sparse blocks
    executable: bool
    normalization_rule: str
    case: str
    dimension: int


def _sparse_block_scope(
    scope_id: str,
    label: str,
    structure_id: str,
    *,
    support_budget: int,
    slot_budget: int,
    reference_seed: int,
) -> MatrixScope:
    from robust_qsvt_se.qsvt.bipartite_slot_assignment import minimum_slot_count
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import SupportConstraints
    from robust_qsvt_se.reviewer_evidence.engine import (
        build_structure,
        build_tasks,
        seed_reference,
        select_support,
    )

    ctx = build_structure(structure_id)
    train = build_tasks(
        structure_id, [reference_seed, reference_seed + 1, reference_seed + 2], "training"
    )
    cons = SupportConstraints(support_budget, slot_budget, True)
    rec = select_support(ctx, "global_magnitude", cons, train, ctx.physical_ids)
    support = rec["support"]
    if support is None:
        raise RuntimeError(f"canonical support unavailable for {structure_id}: {rec}")
    sparse_block = np.where(support, ctx.matrix, 0.0)
    pattern = sparse_block.T != 0.0
    slots = int(minimum_slot_count(pattern))
    mu = float(np.max(np.abs(sparse_block)))
    beta = float(slots * mu)
    singular = np.linalg.svd(sparse_block, compute_uv=False)
    positive = singular[singular > 1e-10] / beta
    domain_min = float(np.clip(0.9 * positive.min(), 1e-4, 0.999))
    residual, _ = seed_reference(structure_id, reference_seed)
    residual_unit = residual / np.linalg.norm(residual)
    return MatrixScope(
        scope_id=scope_id,
        label=label,
        kind="sparse_block",
        beta=beta,
        slots=slots,
        mu=mu,
        domain_min=domain_min,
        singular_values_norm=np.sort(positive)[::-1],
        residual_unit=residual_unit,
        sparse_block=sparse_block,
        executable=True,
        normalization_rule="beta = slots * max|H_block| (sparse block-encoding subnormalization)",
        case=ctx.case_name,
        dimension=ctx.dimension,
    )


def _full_jacobian_scope(scope_id: str, label: str, case: str, seed: int) -> MatrixScope:
    from robust_qsvt_se.cross_case_validation.common import build_case_full_system

    system = build_case_full_system(case, seed)
    matrix = np.asarray(system.matrix, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    beta = float(singular.max())  # spectral-norm block-encoding subnormalization
    positive = singular[singular > 1e-10] / beta
    domain_min = float(np.clip(0.9 * positive.min(), 1e-6, 0.999))
    residual = np.asarray(system.residual, dtype=np.float64)
    residual_unit = residual / np.linalg.norm(residual)
    return MatrixScope(
        scope_id=scope_id,
        label=label,
        kind="full_jacobian",
        beta=beta,
        slots=matrix.shape[1],
        mu=float(np.max(np.abs(matrix))),
        domain_min=domain_min,
        singular_values_norm=np.sort(positive)[::-1],
        residual_unit=residual_unit,
        sparse_block=None,
        executable=False,
        normalization_rule="beta = sigma_max (dense spectral-norm subnormalization); exact "
        "matrix-action only (no circuit propagation)",
        case=case,
        dimension=int(matrix.shape[1]),
    )


def _scalar_scope(scope_id: str, label: str, domain_min: float) -> MatrixScope:
    return MatrixScope(
        scope_id=scope_id,
        label=label,
        kind="scalar",
        beta=1.0,
        slots=1,
        mu=1.0,
        domain_min=float(domain_min),
        singular_values_norm=None,
        residual_unit=None,
        sparse_block=None,
        executable=False,
        normalization_rule=(
            "beta = 1 (controlled scalar validation grid; pure approximation theory)"
        ),
        case="none",
        dimension=0,
    )


def build_scopes(config: dict[str, Any]) -> list[MatrixScope]:
    """Build the declared scopes.  Unavailable scopes are recorded, never silently dropped."""

    spec = config["scopes"]
    scopes: list[MatrixScope] = []
    seed = int(config.get("matrix_seed", 123))
    for entry in spec:
        kind = entry["kind"]
        sid = entry["scope_id"]
        label = entry["label"]
        if kind == "scalar":
            scopes.append(_scalar_scope(sid, label, float(entry.get("domain_min", 1e-3))))
        elif kind == "sparse_block":
            scopes.append(
                _sparse_block_scope(
                    sid,
                    label,
                    entry["structure_id"],
                    support_budget=int(entry.get("support_budget", 16)),
                    slot_budget=int(entry.get("slot_budget", 3)),
                    reference_seed=int(entry.get("reference_seed", 1000)),
                )
            )
        elif kind == "full_jacobian":
            scopes.append(_full_jacobian_scope(sid, label, entry["case"], seed))
        else:
            raise ValueError(f"unknown scope kind {kind}")
    return scopes


# ------------------------------------------------------ phase synthesis with a hard timeout


def _synthesis_worker(
    coefficients: list[float], angle_solver: str, cache_dir: str, meta: dict[str, Any]
) -> list[float]:
    """Top-level worker (spawn-picklable) that synthesizes and caches phases in a child process."""

    import numpy as _np

    from robust_qsvt_se.qsvt.phase_synthesis import synthesize_pennylane_phases_cached

    result = synthesize_pennylane_phases_cached(
        _np.asarray(coefficients, dtype=_np.float64),
        angle_solver=angle_solver,
        cache_dir=cache_dir,
        cache_metadata=meta,
    )
    return [float(x) for x in result.phases]


def _phase_cache_hit(
    coefficients: list[float], angle_solver: str, cache_dir: Path, meta: dict[str, Any]
) -> np.ndarray | None:
    """Load already-synthesized phases in-process (no hang risk) if the cache holds them."""

    import hashlib
    import json

    payload = {"coefficients": coefficients, "angle_solver": angle_solver, "metadata": meta}
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    path = cache_dir / f"{key}_phase_angles.csv"
    if path.is_file():
        phases = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
        return np.atleast_1d(np.asarray(phases, dtype=np.float64))
    return None


def _synthesize_guarded(
    coefficients: np.ndarray,
    *,
    angle_solver: str,
    cache_dir: Path,
    meta: dict[str, Any],
    timeout_s: float,
) -> tuple[np.ndarray | None, str]:
    """Synthesize phases with a wall-clock timeout so a stiff target cannot hang the campaign.

    Iterative phase synthesis for smooth / stiff bounded targets is known to hang; a cache miss runs
    in a fresh single-worker pool and is abandoned (status ``phase_synthesis_timeout``) if it
    exceeds ``timeout_s``.  Cache hits load in-process (no hang risk).  Successful phases are cached
    by coefficient hash for reuse.
    """

    coeffs = [float(x) for x in np.asarray(coefficients, dtype=np.float64)]
    cached = _phase_cache_hit(coeffs, angle_solver, cache_dir, meta)
    if cached is not None:
        return cached, "synthesized"
    try:
        with cf.ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_synthesis_worker, coeffs, angle_solver, str(cache_dir), meta)
            try:
                phases = np.asarray(future.result(timeout=timeout_s), dtype=np.float64)
            except cf.TimeoutError:
                for process in pool._processes.values():  # hard-kill the stuck child
                    process.terminate()
                return None, "phase_synthesis_timeout"
    except Exception as exc:
        return None, f"phase_synthesis_failure_{type(exc).__name__}"
    return phases, "synthesized"


# --------------------------------------------------------------------------- row evaluation


def _exact_matrix_action(
    scope: MatrixScope, coefficients: np.ndarray, c_selected: float, normalized_lambda: float
) -> dict[str, float]:
    """Exact polynomial matrix-action vs the ideal bounded rational filter on the residual.

    Uses the same ``A.T / beta`` convention as the statevector path so the two tiers are comparable.
    """

    if scope.singular_values_norm is None or scope.residual_unit is None:
        return {
            "singular_point_fit_error": float("nan"),
            "exact_matrix_action_error": float("nan"),
            "postselection_probability_modeled": float("nan"),
        }
    poly = Polynomial(np.asarray(coefficients, dtype=np.float64))
    s = np.asarray(scope.singular_values_norm, dtype=np.float64)
    ideal = s / (c_selected * (s**2 + normalized_lambda))
    point_error = float(np.max(np.abs(poly(s) - ideal)))
    # Full-vector action via the block-encoding SVD convention.
    if scope.sparse_block is not None:
        normalized_matrix = scope.sparse_block.T / scope.beta
    else:  # full Jacobian: reconstruct A.T/beta lazily from the cached case system matrix
        normalized_matrix = _full_matrix_transpose_over_beta(scope)
    left, sv, right_t = np.linalg.svd(normalized_matrix, full_matrices=False)
    poly_diag = poly(sv)
    ideal_diag = sv / (c_selected * (sv**2 + normalized_lambda))
    proj = right_t @ scope.residual_unit
    poly_action = left @ (poly_diag * proj)
    ideal_action = left @ (ideal_diag * proj)
    denom = max(float(np.linalg.norm(ideal_action)), 1e-30)
    action_error = float(np.linalg.norm(poly_action - ideal_action) / denom)
    p_post = float(np.dot(poly_action, poly_action))
    return {
        "singular_point_fit_error": point_error,
        "exact_matrix_action_error": action_error,
        "postselection_probability_modeled": p_post,
    }


_FULL_MATRIX_CACHE: dict[str, np.ndarray] = {}


def _full_matrix_transpose_over_beta(scope: MatrixScope) -> np.ndarray:
    from robust_qsvt_se.cross_case_validation.common import build_case_full_system

    key = f"{scope.case}"
    if key not in _FULL_MATRIX_CACHE:
        system = build_case_full_system(scope.case, 123)
        _FULL_MATRIX_CACHE[key] = np.asarray(system.matrix, dtype=np.float64)
    return _FULL_MATRIX_CACHE[key].T / scope.beta


def _statevector_row(
    scope: MatrixScope,
    coefficients: np.ndarray,
    phases: np.ndarray,
    c_selected: float,
    normalized_lambda: float,
) -> dict[str, float]:
    """Explicit statevector circuit action on an executable sparse block (reused primitives)."""

    from qiskit.quantum_info import Statevector

    from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
    from robust_qsvt_se.qsvt.output_aware_sparse_selection import build_common_padded_wrapper

    block = scope.sparse_block
    wrapper = build_common_padded_wrapper(block, slots=scope.slots, mu=scope.mu)
    bundle = build_structured_qsvt_operator_circuit(
        wrapper.unitary, phases, encoded_dimension=block.shape[1]
    )
    initial = np.zeros(wrapper.unitary.shape[0], dtype=np.complex128)
    initial[: block.shape[0]] = scope.residual_unit
    evolved = Statevector(initial).evolve(bundle.qsvt_operator_circuit).data
    encoded = np.asarray(evolved[: block.shape[1]], dtype=np.complex128)
    p_post = float(np.vdot(encoded, encoded).real)
    normalized_matrix = block.T / scope.beta
    left, sv, right_t = np.linalg.svd(normalized_matrix, full_matrices=False)
    ideal_diag = sv / (c_selected * (sv**2 + normalized_lambda))
    ideal = left @ (ideal_diag * (right_t @ scope.residual_unit))
    action_error = float(
        np.linalg.norm(np.real(encoded) - ideal) / max(np.linalg.norm(ideal), 1e-30)
    )
    return {
        "statevector_dim": int(wrapper.unitary.shape[0]),
        "statevector_action_error": action_error,
        "postselection_probability_executed": p_post,
    }


def evaluate_row(
    scope: MatrixScope,
    normalized_lambda: float,
    degree: int,
    config: dict[str, Any],
    cache_dir: Path,
    *,
    execute_statevector: bool,
) -> dict[str, Any]:
    margin = float(config.get("target_margin", 1.05))
    bound_tol = float(config.get("bound_tolerance", 0.002))
    fit_grid = int(config.get("fit_grid_size", 6000))
    synth_ceiling = int(config.get("phase_synthesis_degree_ceiling", 63))
    synth_timeout = float(config.get("phase_synthesis_timeout_seconds", 45.0))
    attempt_synth_policy = bool(config.get("attempt_phase_synthesis", True))

    alpha = float(normalized_lambda) * scope.beta**2
    crec = analytic_bound_c(normalized_lambda, scope.domain_min, 1.0, margin=margin)

    row: dict[str, Any] = {
        "scope_id": scope.scope_id,
        "scope_label": scope.label,
        "scope_kind": scope.kind,
        "case": scope.case,
        "dimension": scope.dimension,
        "normalized_lambda": float(normalized_lambda),
        "physical_alpha": alpha,
        "beta": scope.beta,
        "slots": scope.slots,
        "mu": scope.mu,
        "normalization_rule": scope.normalization_rule,
        "degree": int(degree),
        "parity": "odd",
        "synthesis_interval_lo": scope.domain_min,
        "synthesis_interval_hi": 1.0,
        "bound_C_analytic": crec.c_analytic,
        "bound_C_selected": crec.c_selected,
        "bound_C_maximizer": crec.maximizer_location,
        "bound_C_s_star": crec.s_star,
        "bound_C_margin": crec.margin,
        "resulting_max_target_magnitude": crec.resulting_max_target_magnitude,
        "signal_unitary_query_count": int(degree),
        "phase_operation_count": int(degree + 1),
        "failure_code": "",
        "failure_reason": "",
    }

    try:
        target = fit_codesigned_bounded_polynomial(
            beta=scope.beta,
            alpha=alpha,
            domain_min=scope.domain_min,
            domain_max=1.0,
            degree=int(degree),
            grid_size=fit_grid,
            margin=margin,
        )
    except Exception as exc:
        row.update(
            {
                "failure_code": "polynomial_fit_failure",
                "failure_reason": str(exc)[:200],
                "uniform_fit_error": float("nan"),
                "bounded_max_abs": float("nan"),
                "boundedness_ok": False,
                "bound_C_numeric": float("nan"),
                "singular_point_fit_error": float("nan"),
                "exact_matrix_action_error": float("nan"),
                "phase_synthesis_status": "not_attempted",
                "phase_count": 0,
                "convention_transfer_ok": False,
                "convention_transfer_error": float("nan"),
                "statevector_dim": -1,
                "statevector_action_error": float("nan"),
                "postselection_probability_modeled": float("nan"),
                "postselection_probability_executed": float("nan"),
                "evidence_tier": EVIDENCE_TIERS["SCALAR_POLY_FIT"],
            }
        )
        return row

    coefficients = np.asarray(target.coefficients, dtype=np.float64)
    bounded_max = float(target.bounded_max_abs)
    bounded_ok = bool(bounded_max <= 1.0 + bound_tol)
    # C consistency: fit's numeric interval max (bound_C) vs the analytic interval maximizer.
    row["bound_C_numeric"] = float(target.bound_C)
    row["bound_C_analytic_vs_numeric_rel_gap"] = float(
        abs(target.bound_C - crec.c_selected) / max(crec.c_selected, 1e-30)
    )
    row["uniform_fit_error"] = float(target.fit_max_abs_error)
    row["bounded_max_abs"] = bounded_max
    row["boundedness_ok"] = bounded_ok

    # Exact-matrix-action tier (matrix scopes) + modeled postselection.
    action = _exact_matrix_action(scope, coefficients, target.bound_C, target.alpha_normalized)
    row.update(action)

    # Phase synthesis tier (parity + boundedness gate; declared degree ceiling; timeout-guarded).
    parity_ok = bool(np.all(np.abs(coefficients[::2]) <= max(bound_tol, 1e-9)))
    row["parity_ok"] = parity_ok
    phase_status = "not_attempted"
    phases: np.ndarray | None = None
    conv_ok, conv_err = False, float("nan")
    if scope.kind == "full_jacobian":
        # Full Jacobians are recorded at the exact-matrix-action tier (task section 5.5); circuit
        # phase synthesis and statevector propagation are out of scope for them by declared policy.
        phase_status = "not_attempted_exact_action_tier"
    elif not attempt_synth_policy:
        phase_status = "not_attempted_by_policy"
    elif degree > synth_ceiling:
        phase_status = "not_attempted_ceiling"
    elif not (bounded_ok and parity_ok):
        phase_status = "skipped_unbounded_or_wrong_parity"
    else:
        phases, phase_status = _synthesize_guarded(
            coefficients,
            angle_solver="iterative",
            cache_dir=cache_dir,
            meta={"study_id": STUDY_ID, "degree": int(degree), "beta": float(scope.beta)},
            timeout_s=synth_timeout,
        )
        if phases is not None and phases.size != degree + 1:
            phase_status = "phase_count_mismatch"
        elif phases is not None:
            conv_ok, conv_err = _convention_transfer(coefficients, phases, scope.domain_min)
    row["phase_synthesis_status"] = phase_status
    row["phase_count"] = int(phases.size) if phases is not None else 0
    row["convention_transfer_ok"] = bool(conv_ok)
    row["convention_transfer_error"] = float(conv_err)

    # Statevector tier (executable sparse blocks only, declared subset).
    row["statevector_dim"] = -1
    row["statevector_action_error"] = float("nan")
    row["postselection_probability_executed"] = float("nan")
    tier = EVIDENCE_TIERS["SCALAR_POLY_FIT"]
    if scope.kind in {"sparse_block", "full_jacobian"}:
        tier = EVIDENCE_TIERS["EXACT_MATRIX_ACTION"]
    if (
        execute_statevector
        and scope.executable
        and phases is not None
        and phase_status == "synthesized"
    ):
        try:
            sv_row = _statevector_row(
                scope, coefficients, phases, target.bound_C, target.alpha_normalized
            )
            row.update(sv_row)
            tier = EVIDENCE_TIERS["STATEVECTOR"]
        except Exception as exc:
            row["failure_code"] = "statevector_failure"
            row["failure_reason"] = str(exc)[:200]
    row["evidence_tier"] = tier

    # Record a non-empty failure code for the registry when a stage did not pass.
    if not row["failure_code"]:
        if not bounded_ok:
            row["failure_code"] = "unbounded"
            row["failure_reason"] = f"bounded_max_abs={bounded_max:.4g} exceeds 1+{bound_tol}"
        elif not parity_ok:
            row["failure_code"] = "wrong_parity"
        elif phase_status.startswith("phase_synthesis_"):
            row["failure_code"] = phase_status
            row["failure_reason"] = "phase synthesis did not converge within the timeout"
        elif phase_status == "phase_count_mismatch":
            row["failure_code"] = "phase_count_mismatch"
    return row


def _convention_transfer(
    coefficients: np.ndarray, phases: np.ndarray, domain_min: float
) -> tuple[bool, float]:
    """Verify the synthesized phases reproduce the target polynomial under the documented QSVT
    convention (``pennylane_rx_pcphase`` scalar response, ``real_u00`` component)."""

    from robust_qsvt_se.qsvt.phase_response_conventions import pennylane_qsvt_response

    grid = np.linspace(domain_min, 1.0, 257, dtype=np.float64)
    response = pennylane_qsvt_response(grid, phases)
    target = Polynomial(np.asarray(coefficients, dtype=np.float64))(grid)
    err = float(np.max(np.abs(response - target)))
    return bool(err <= 1e-3), err


# --------------------------------------------------------------------------- orchestrator


def run_degree_lambda_scaling(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"config study_id mismatch: {config.get('study_id')!r} != {STUDY_ID!r}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_dir = destination / "phase_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    lambdas = [float(x) for x in config["normalized_lambdas"]]
    tolerances = [float(x) for x in config["target_tolerances"]]
    degrees = [int(x) for x in config["degrees"]]
    if any(d % 2 == 0 or d < 1 for d in degrees):
        raise ValueError("degrees must be positive and odd")
    exec_grid = {
        (str(item["scope_id"]), float(item["normalized_lambda"]), int(item["degree"]))
        for item in config.get("statevector_executions", [])
    }

    scopes = build_scopes(config)
    spectrum_rows = []
    for scope in scopes:
        if scope.singular_values_norm is not None:
            for rank, s in enumerate(scope.singular_values_norm):
                spectrum_rows.append(
                    {
                        "scope_id": scope.scope_id,
                        "scope_label": scope.label,
                        "case": scope.case,
                        "rank": rank,
                        "normalized_singular_value": float(s),
                        "beta": scope.beta,
                        "domain_min": scope.domain_min,
                    }
                )

    rows: list[dict[str, Any]] = []
    total = len(scopes) * len(lambdas) * len(degrees)
    done = 0
    for scope in scopes:
        for lam in lambdas:
            for degree in degrees:
                execute = (scope.scope_id, lam, degree) in exec_grid
                rows.append(
                    evaluate_row(scope, lam, degree, config, cache_dir, execute_statevector=execute)
                )
                done += 1
                if progress and done % 25 == 0:
                    print(f"[degree_lambda] {done}/{total} rows", flush=True)

    grid = pd.DataFrame(rows)
    atomic_write_csv(destination / "raw_grid.csv", grid)
    atomic_write_csv(destination / "singular_spectrum_rows.csv", pd.DataFrame(spectrum_rows))

    d_min = extract_min_feasible_degree(grid, tolerances)
    atomic_write_csv(destination / "minimum_feasible_degree.csv", d_min)

    failures = grid[grid["failure_code"].astype(str) != ""][
        [
            "scope_id",
            "normalized_lambda",
            "degree",
            "failure_code",
            "failure_reason",
            "phase_synthesis_status",
            "evidence_tier",
        ]
    ]
    atomic_write_csv(
        destination / "failure_registry.csv",
        failures
        if not failures.empty
        else pd.DataFrame(
            columns=["scope_id", "normalized_lambda", "degree", "failure_code", "failure_reason"]
        ),
    )

    action_cols = [
        "scope_id",
        "scope_kind",
        "case",
        "normalized_lambda",
        "degree",
        "beta",
        "singular_point_fit_error",
        "exact_matrix_action_error",
        "statevector_action_error",
        "postselection_probability_modeled",
        "postselection_probability_executed",
        "evidence_tier",
    ]
    matrix_action = grid[grid["scope_kind"] != "scalar"][action_cols]
    atomic_write_csv(destination / "matrix_action_results.csv", matrix_action)

    resource_cols = [
        "scope_id",
        "normalized_lambda",
        "degree",
        "signal_unitary_query_count",
        "phase_operation_count",
        "bound_C_selected",
        "postselection_probability_modeled",
        "postselection_probability_executed",
        "phase_synthesis_status",
        "evidence_tier",
    ]
    atomic_write_csv(destination / "resource_summary.csv", grid[resource_cols])

    claim = _claim_support(grid, d_min, scopes, lambdas, tolerances, degrees)
    atomic_write_json(destination / "claim_support.json", claim)
    atomic_write_json(
        destination / "run_manifest.json",
        provenance_block(config_path, config) | {"study_id": STUDY_ID, "row_count": len(grid)},
    )
    _write_resolved_config(destination, config)
    _write_readme(destination, config, grid, d_min, scopes)
    write_manifest_and_checksums(destination, study_id=STUDY_ID, extra={"row_count": len(grid)})
    return {
        "rows": len(grid),
        "scopes": [s.scope_id for s in scopes],
        "d_min_records": len(d_min),
        "failures": len(failures),
        "synthesized_rows": int((grid["phase_synthesis_status"] == "synthesized").sum()),
        "statevector_rows": int((grid["evidence_tier"] == EVIDENCE_TIERS["STATEVECTOR"]).sum()),
    }


def _write_resolved_config(destination: Path, config: dict[str, Any]) -> None:
    import yaml

    text = yaml.safe_dump(config, sort_keys=True, default_flow_style=False)
    (destination / "config_resolved.yaml").write_text(text, encoding="utf-8")


def _claim_support(grid, d_min, scopes, lambdas, tolerances, degrees) -> dict[str, Any]:
    scalar = grid[grid["scope_kind"] == "scalar"]
    scalar_dmin = d_min[d_min["scope_id"].isin(scalar["scope_id"].unique())]
    any_feasible = scalar_dmin["d_min_fit"].notna().any() if not scalar_dmin.empty else False
    return {
        "study_id": STUDY_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "grid": {
            "normalized_lambdas": lambdas,
            "target_tolerances": tolerances,
            "degrees": degrees,
            "scopes": [
                {
                    "scope_id": s.scope_id,
                    "kind": s.kind,
                    "case": s.case,
                    "beta": s.beta,
                    "domain_min": s.domain_min,
                    "executable": s.executable,
                }
                for s in scopes
            ],
        },
        "grid_frozen_before_evaluation": True,
        "all_declared_rows_present": bool(len(grid) == len(scopes) * len(lambdas) * len(degrees)),
        "failures_retained": True,
        "bound_C_explicitly_computed": True,
        "min_feasible_degree_extracted": bool(not d_min.empty),
        "any_scalar_tolerance_feasible_at_a_tested_degree": bool(any_feasible),
        "evidence_tiers_present": sorted(grid["evidence_tier"].unique().tolist()),
        "allowed_claim": (
            "The controlled grid identifies, per matrix scope, the minimum tested odd degree "
            "satisfying the declared uniform-fit (epsilon_target), boundedness, and (where "
            "attempted) phase-synthesis criteria as a function of normalized regularization lambda "
            "and target tolerance. Scalar, exact-matrix-action, and statevector tiers are recorded "
            "separately. No asymptotic degree theorem is claimed."
        ),
        "forbidden_claims": [
            "quantum speedup",
            "quantum advantage",
            "hardware execution",
            "QSVT beats matched Ridge",
            "asymptotic degree bound proven",
        ],
    }


def _write_readme(destination: Path, config, grid, d_min, scopes) -> None:
    lines = [
        "# Workstream A - Degree / Normalized-Regularization / Target-Tolerance Feasibility Map",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- study_id: `{STUDY_ID}`",
        f"- rows: {len(grid)} = {len(scopes)} scopes x {len(config['normalized_lambdas'])} lambda "
        f"x "
        f"{len(config['degrees'])} degrees",
        "- scopes: " + ", ".join(f"`{s.scope_id}`({s.kind})" for s in scopes),
        "",
        "## Files",
        "- `raw_grid.csv` - every (scope, lambda, degree) row with all recorded metrics.",
        "- `singular_spectrum_rows.csv` - normalized singular values per matrix scope.",
        "- `minimum_feasible_degree.csv` - d_min(scope, lambda, epsilon_target), fit + synthesis.",
        "- `failure_registry.csv` - retained failures with explicit codes.",
        "- `matrix_action_results.csv` - exact-matrix-action + statevector errors and "
        "postselection.",
        "- `resource_summary.csv` - query/phase counts, C, postselection per row.",
        "- `claim_support.json`, `run_manifest.json`, `config_resolved.yaml`, `checksums.sha256`.",
        "",
        "## Reproduce",
        "```",
        "MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 \\",
        "  .venv/bin/python scripts/run_tqe_degree_lambda_error_scaling.py",
        "```",
        "",
        "## Evidence tiers (kept separate, never merged)",
        "- scalar polynomial approximation; exact polynomial matrix action; explicit statevector "
        "circuit execution; modeled postselection.",
        "",
        "Boundedness factor C is computed explicitly from the interval maximizer of "
        "`g(s)=s/(s^2+lambda)` on `[s_lo,1]` (`bound_C_analytic`) and cross-checked against the "
        "fitted numeric maximum (`bound_C_numeric`); no result claims QSVT numerical superiority "
        "over matched Ridge.",
    ]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
