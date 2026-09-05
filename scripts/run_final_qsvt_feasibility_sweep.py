"""Final QSVT feasibility sweep (Work Packages B, C, D, F core).

Sweeps the validated stable spectrum-aware -> pyqsp -> circuit-action pipeline over
real IEEE-14 / IEEE-30 spectra, polynomial methods, the pre-registered lambda grid,
and a progressive degree grid. Produces:

  polynomial_method_comparison.csv  (WP-B)
  phase_synthesis_comparison.csv    (WP-C)
  spectrum_aware_results.csv        (WP-D)
  extended_feasibility_frontier.csv (WP-F)

All rows carry configuration + provenance + an evidence label. The physical
Ridge/Tikhonov estimator is invariant: lambda = alpha / beta^2, and at matched
alpha the QSVT target is the same spectral filter as Ridge. No ground truth is
used; degrees escalate progressively and stop on a documented limit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.polynomial_approximation import build_approximation_context
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (
    CIRCUIT_ACTION_CLAIM,
    fit_bounded_odd_chebyshev,
    synthesize_pyqsp_sym_qsp_phases,
    validate_circuit_action,
)

LAMBDA_GRID = [0.069, 0.02, 1.0e-2, 5.0e-3, 1.0e-3, 5.0e-4, 1.0e-4, 5.0e-5, 1.0e-5]
# Degree grid capped at 255: pyqsp sym_qsp runtime exceeds the documented budget at
# degree >= 511 (>45 s per point on this machine; see RUNTIME_LIMIT_FAILED rows and
# the runtime_limit_probe). 511/1021 are probed separately to document the ceiling.
DEGREE_GRID = [31, 45, 63, 95, 127, 191, 255]
# minimax_lp (scipy linprog/HiGHS) becomes slow AND numerically unstable at high
# degree (the audit documented HiGHS minimax-LP failure modes). Cap it for the
# automated sweep; stable_chebyshev carries the high-degree feasibility evidence.
MINIMAX_MAX_DEGREE = 127
COMBO_BUDGET_S = 110  # hard wall-clock guard per (case, method, lam, degree) combo
METHODS = ["stable_chebyshev", "minimax_lp"]
CASES = ["ieee14", "ieee30"]
PRIMARY_TOL = 1.0e-2
STRICT_TOL = 1.0e-3
DEGREE_TIMEOUT_S = 60.0


def _case_context(case: str):
    return build_approximation_context(
        {
            "case_name": case,
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
            "fallback_to_synthetic": False,
        }
    )


def _run_one(case, ctx, method, lam, degree):
    s_min = float(ctx.domain_min)
    beta = float(ctx.beta)
    alpha_phys = lam * beta * beta  # recover physical alpha from normalized lambda
    actual_sv = np.asarray(ctx.normalized_singular_values, dtype=np.float64)
    t0 = time.perf_counter()
    row = {
        "case": case,
        "method": method,
        "lambda": lam,
        "alpha_physical": alpha_phys,
        "beta": beta,
        "s_min": s_min,
        "degree": degree,
        "kappa": float(beta / s_min),
    }
    try:
        poly = fit_bounded_odd_chebyshev(s_min=s_min, lam=lam, degree=degree, method=method)
        phases = synthesize_pyqsp_sym_qsp_phases(poly.chebyshev_coeffs)
        if time.perf_counter() - t0 > DEGREE_TIMEOUT_S:
            row.update(_fail("RUNTIME_LIMIT_FAILED", degree, time.perf_counter() - t0))
            return row
        rep = validate_circuit_action(poly=poly, phases=phases, actual_singular_values=actual_sv)
        row.update(
            {
                "C_global": poly.C_global,
                "phase_count": rep.phase_count,
                "occupied_recon_error": rep.occupied_recon_error,
                "actual_sv_error": rep.occupied_actual_sv_error,
                "global_bounded_max": rep.global_bounded_max,
                "circuit_vs_scalar_error": rep.circuit_vs_scalar_max_error,
                "circuit_vs_target_error": rep.circuit_vs_target_max_error,
                "bounded_passes": bool(rep.bounded_passes),
                "poly_primary_pass": bool(rep.occupied_recon_error <= PRIMARY_TOL),
                "poly_strict_pass": bool(rep.occupied_recon_error <= STRICT_TOL),
                "phase_status": "passed_synthesis",
                "circuit_action_status": rep.evidence_label,
                "runtime_seconds": time.perf_counter() - t0,
                "status": "ok",
                "failure_reason": "",
            }
        )
    except Exception as exc:  # pragma: no cover - per-row diagnostics
        row.update(
            _fail(f"FAILED:{type(exc).__name__}", degree, time.perf_counter() - t0, str(exc)[:80])
        )
    return row


def _fail(reason, degree, runtime, msg=""):
    return {
        "C_global": np.nan,
        "phase_count": 0,
        "occupied_recon_error": np.nan,
        "actual_sv_error": np.nan,
        "global_bounded_max": np.nan,
        "circuit_vs_scalar_error": np.nan,
        "circuit_vs_target_error": np.nan,
        "bounded_passes": False,
        "poly_primary_pass": False,
        "poly_strict_pass": False,
        "phase_status": "failed_synthesis",
        "circuit_action_status": "DIAGNOSTIC_ONLY",
        "runtime_seconds": runtime,
        "status": "failed",
        "failure_reason": f"{reason}: {msg}",
    }


def run_sweep(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    class _ComboTimeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _ComboTimeout

    import signal

    old_handler = signal.signal(signal.SIGALRM, _alarm)
    try:
        for case in CASES:
            ctx = _case_context(case)
            for method in METHODS:
                for lam in LAMBDA_GRID:
                    # Progressive degree escalation: once strict tol met, stop escalating.
                    strict_met = False
                    for degree in DEGREE_GRID:
                        if method == "minimax_lp" and degree > MINIMAX_MAX_DEGREE:
                            # minimax is slow/unstable above its cap; record and skip.
                            rows.append(
                                _fail(
                                    "RUNTIME_LIMIT_FAILED",
                                    degree,
                                    0.0,
                                    f"minimax_lp capped at degree {MINIMAX_MAX_DEGREE}",
                                )
                            )
                            continue
                        if strict_met and degree > 255:
                            break
                        t0 = time.perf_counter()
                        try:
                            signal.alarm(COMBO_BUDGET_S)
                            row = _run_one(case, ctx, method, lam, degree)
                            signal.alarm(0)
                        except _ComboTimeout:
                            signal.alarm(0)
                            row = _fail(
                                "RUNTIME_LIMIT_FAILED",
                                degree,
                                time.perf_counter() - t0,
                                f"combo exceeded {COMBO_BUDGET_S}s wall clock",
                            )
                        rows.append(row)
                        if row.get("poly_strict_pass"):
                            strict_met = True
                        if row.get("status") != "ok" and "RUNTIME_LIMIT" in str(
                            row.get("failure_reason", "")
                        ):
                            break
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "polynomial_method_comparison.csv", index=False)
    # WP-C phase synthesis view (one row per passed-approx point)
    phase_cols = [
        "case",
        "method",
        "lambda",
        "alpha_physical",
        "degree",
        "phase_count",
        "occupied_recon_error",
        "actual_sv_error",
        "circuit_vs_scalar_error",
        "circuit_vs_target_error",
        "phase_status",
        "circuit_action_status",
        "status",
    ]
    df[df["status"] == "ok"][phase_cols].to_csv(
        output_dir / "phase_synthesis_comparison.csv", index=False
    )
    # WP-D spectrum-aware view (stable_chebyshev only, with actual-SV vs grid error)
    d = df[(df["method"] == "stable_chebyshev") & (df["status"] == "ok")].copy()
    d.to_csv(output_dir / "spectrum_aware_results.csv", index=False)
    # WP-F frontier: best (smallest occupied_recon_error) point per (case, lambda)
    frontier = (
        df[df["status"] == "ok"]
        .sort_values("occupied_recon_error")
        .groupby(["case", "lambda"], as_index=False)
        .first()
    )
    frontier["useful_feasible_overlap"] = (
        frontier["poly_primary_pass"]
        & frontier["bounded_passes"]
        & (frontier["circuit_vs_target_error"] <= PRIMARY_TOL)
    )
    frontier.to_csv(output_dir / "extended_feasibility_frontier.csv", index=False)
    return {
        "polynomial_method_comparison": output_dir / "polynomial_method_comparison.csv",
        "phase_synthesis_comparison": output_dir / "phase_synthesis_comparison.csv",
        "spectrum_aware_results": output_dir / "spectrum_aware_results.csv",
        "extended_feasibility_frontier": output_dir / "extended_feasibility_frontier.csv",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/final_qsvt_feasibility_push")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    t0 = time.perf_counter()
    artifacts = run_sweep(out)
    print(f"# Final QSVT feasibility sweep complete in {time.perf_counter() - t0:.1f}s")
    for key, path in artifacts.items():
        print(f"  {key}: {path}")
    print(f"# Claim: {CIRCUIT_ACTION_CLAIM[:120]}...")


if __name__ == "__main__":
    main()
