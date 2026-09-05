"""Shared fixtures for the phase-synthesis refinement test suite.

The gate-level circuit path is expensive (phase synthesis + statevector simulation), so these
fixtures provide a deterministic well-conditioned subproblem and synthetic replacements for the
gate entry points. The synthetic gate reproduces the Ridge update plus a controlled relative
error that depends on the fit mode, exercising the selection/limitation logic without invoking
PennyLane or Qiskit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robust_qsvt_se.paper import phase_synthesis_refinement as psr
from robust_qsvt_se.paper.full_vector_readout import ridge_update


@pytest.fixture(scope="module")
def inprocess_phase_synthesis_guard(tmp_path_factory):
    """Run the existing small-test phase synthesis without a process-pool semaphore.

    The production guard uses ``ProcessPoolExecutor`` to enforce a wall-clock timeout.
    The managed macOS test environment denies its semaphore-limit query before a child
    can start. Closed-loop tests opt into this fixture and still execute the same
    production synthesis worker, coefficients, PennyLane backend, and cache format,
    but use only pytest-owned writable directories and the current process.
    """

    from robust_qsvt_se.tqe_extensions import closed_loop_nonlinear_update as clnu
    from robust_qsvt_se.tqe_extensions import degree_lambda_scaling as dls
    from robust_qsvt_se.tqe_extensions import nonlinear_circuit_loop as ncl

    original_closed_loop = clnu._synthesize_guarded
    original_nonlinear_loop = ncl._synthesize_guarded
    cache_environment = tmp_path_factory.mktemp("phase_synthesis_environment")
    previous_mpl_config = os.environ.get("MPLCONFIGDIR")
    previous_xdg_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["MPLCONFIGDIR"] = str(cache_environment / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache_environment / "xdg")

    def synthesize_in_process(
        coefficients: np.ndarray,
        *,
        angle_solver: str,
        cache_dir: Path,
        meta: dict[str, Any],
        timeout_s: float,
    ) -> tuple[np.ndarray | None, str]:
        del timeout_s  # process timeout is the only production behavior replaced in this test.
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        values = [float(value) for value in np.asarray(coefficients, dtype=np.float64)]
        cached = dls._phase_cache_hit(values, angle_solver, cache_root, meta)
        if cached is not None:
            return cached, "synthesized"
        try:
            phases = dls._synthesis_worker(values, angle_solver, str(cache_root), meta)
        except Exception as exc:
            return None, f"phase_synthesis_failure_{type(exc).__name__}"
        return np.asarray(phases, dtype=np.float64), "synthesized"

    clnu._synthesize_guarded = synthesize_in_process
    ncl._synthesize_guarded = synthesize_in_process
    try:
        yield cache_environment
    finally:
        clnu._synthesize_guarded = original_closed_loop
        ncl._synthesize_guarded = original_nonlinear_loop
        if previous_mpl_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_mpl_config
        if previous_xdg_cache is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_xdg_cache


def _unit_perturbation(size: int) -> np.ndarray:
    pert = np.arange(1, size + 1, dtype=np.float64)
    pert[0] *= -1.0
    return pert / np.linalg.norm(pert)


@pytest.fixture
def tiny_context() -> psr._RefinementContext:
    """A deterministic, well-conditioned 4x4 subproblem as a refinement context."""

    rng = np.random.default_rng(7)
    H = np.diag([3.0, 2.0, 1.4, 0.7]) + 0.05 * rng.standard_normal((4, 4))
    r = rng.standard_normal(4)
    entry = {
        "case": "ieee14",
        "subproblem_id": "metadata_mapped_02",
        "subproblem_type": "metadata_mapped",
    }
    return psr._RefinementContext(entry, H, r)


def _fit_error(fit_mode: str, weight_mode: str) -> float:
    """Weighted singular-support fits get a smaller synthetic gate error than the uniform fit."""

    if fit_mode == "global_uniform_fit":
        return 0.18
    if weight_mode == "residual_energy_weight":
        return 0.03
    return 0.08


@pytest.fixture
def fake_gate():
    """Factory for a synthetic ``gate_level_subproblem_readout`` (error = scale/degree)."""

    def factory(error_scale: float = 1.5, *, available: bool = True):
        def gate(
            H,
            r,
            *,
            alpha,
            degree,
            seed,
            shots=1000,
            case="custom",
            angle_solver="iterative",
            phase_grid_size=None,
            phase_timeout_seconds=25,
        ):
            if not available:
                return {"available": False, "reason": "synthetic synthesis failure"}
            ridge = ridge_update(H, r, alpha=float(alpha))
            err = float(error_scale) / float(degree)
            gate_update = ridge + err * float(np.linalg.norm(ridge)) * _unit_perturbation(
                ridge.size
            )
            gate_state = gate_update / float(np.linalg.norm(gate_update))
            return {
                "available": True,
                "gate_update": gate_update,
                "gate_state": gate_state,
                "statevector": np.concatenate([gate_state, np.zeros_like(gate_state)]),
                "ridge_update": ridge,
                "success_probability": 0.5,
                "synthesized_degree": int(degree),
                "state_error_vs_ridge": err,
                "circuit_depth": 100 + int(degree),
                "two_qubit_gate_count": 10 * int(degree),
                "qsvt_query_count": int(degree),
                "phase_selection_reason": "synthetic",
                "angle_solver": str(angle_solver),
            }

        return gate

    return factory


@pytest.fixture
def fake_realize():
    """Factory for a synthetic ``realize_config_gate`` whose error depends on the fit mode."""

    def factory(**overrides: Any):
        def realize(
            context,
            *,
            alpha,
            degree,
            fit_mode="global_uniform_fit",
            weight_mode="uniform_support_weight",
            angle_solver="iterative",
            seed,
            cache=None,
        ):
            ridge = ridge_update(context.H, context.r, alpha=float(alpha))
            err = overrides.get("error", _fit_error(fit_mode, weight_mode))
            gate_update = ridge + err * float(np.linalg.norm(ridge)) * _unit_perturbation(
                ridge.size
            )
            gate_state = gate_update / float(np.linalg.norm(gate_update))
            return {
                "available": overrides.get("available", True),
                "gate_update": gate_update,
                "gate_state": gate_state,
                "ridge_update": ridge,
                "success_probability": 0.5,
                "circuit_depth": float(100 + int(degree)),
                "query_count": int(degree),
                "gate_error_vs_ridge": err,
                "gate_error_vs_polynomial": err * 0.5,
                "exact_target_vs_polynomial": err,
                "boundedness_ok": True,
                "residual_ratio_after_gate": err,
            }

        return realize

    return factory


_BASELINE = [
    ("ieee14", "metadata_mapped_02", "metadata_mapped", 1e-5, 45, 3.0189, 0.4718, 6.40, 0.705),
    ("ieee57", "high_leverage_00", "high_leverage", 1e-2, 41, 0.3230, 0.1717, 1.88, 0.0013),
    (
        "ieee14",
        "residual_supported_03",
        "residual_supported",
        1e-5,
        45,
        0.2877,
        0.1313,
        2.19,
        0.064,
    ),
]


@pytest.fixture
def baseline_rows() -> list[dict[str, Any]]:
    """Three synthetic frozen-baseline rows mirroring the real high-error subproblems."""

    rows: list[dict[str, Any]] = []
    for case, sid, stype, alpha, degree, matched, best, factor, gate_poly in _BASELINE:
        rows.append(
            {
                "case": case,
                "subproblem_id": sid,
                "subproblem_type": stype,
                "matched_alpha": 1e-4,
                "matched_degree": 35,
                "matched_alpha_error": matched,
                "current_best_alpha": alpha,
                "current_best_degree": degree,
                "current_best_error": best,
                "improvement_factor": factor,
                "dominant_error_source": "exact_target_vs_polynomial",
                "exact_target_vs_polynomial_error": best,
                "polynomial_vs_gate_error": gate_poly,
                "gate_vs_ridge_error": best,
                "phase_synthesis_status": "phase_synthesis_degraded_at_best_degree"
                if gate_poly > 0.05
                else "phase_synthesis_clean_at_best_degree",
                "status": "polynomial_limited_after_sweep",
                "source_artifact": "full_vector_readout/readout_two_view_summary.csv",
                "notes": "synthetic baseline",
            }
        )
    return rows
