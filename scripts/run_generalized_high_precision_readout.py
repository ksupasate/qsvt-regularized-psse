"""IEEE-14 high-precision shot-based readout (Work Package J).

Strengthens the shot-based evidence for the IEEE-14 headline selected output by
running the Hadamard-test signed-overlap readout on qiskit Aer with multiple
independent backend/transpiler seeds over a shots grid, and checking the
preregistered precision target (relative 95% CI half-width <= 10%).

Construction (faithful to the convention-validated selected output):
  output_state = dx_qsvt / ||dx_qsvt||  (the physical update direction, n-dim),
  observable   = e_0 (theta_2),
  mu = Re<l_hat | output_state>,
  selected output y = ||dx_qsvt|| * mu  ==  l^T dx_qsvt  (the headline value).
The Hadamard test estimates mu via genuine Aer shots; y_shot scatters around the
statevector value y_sv = l^T dx_qsvt with shot noise.

LABEL: state preparation uses the classically-computed update direction loaded by
StatePreparation (dense amplitude loading); the READOUT (Hadamard test +
measurement) is genuinely shot-based on Aer. This is NOT hardware execution, NOT
full-vector recovery, and NOT direct probability sampling mislabeled as backend
execution. Building the full degree-255 QSVT circuit for shot sampling is out of
scope; the state is prepared exactly and the readout is shot-based.
"""

# ruff: noqa: E501,E741

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.circuit_signed_readout import estimate_overlap
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"


def _psd_sqrt(M):
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def main():
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H = np.asarray(system.H_tilde, float)
    r = np.asarray(system.r_tilde, float)
    m, n = H.shape
    beta = float(np.linalg.svd(H, compute_uv=False)[0])
    lam = 1e-5
    alpha = lam * beta * beta

    # convention-validated dx_qsvt via the block action (recomputed here independently)
    from robust_qsvt_se.generalized.convention_api import (
        convert_pyqsp_to_production,
        make_request_from_phases,
    )
    from robust_qsvt_se.qsvt.rectangular_convention import (
        pcphase_qsvt_top_block,
    )
    from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (
        fit_bounded_odd_chebyshev,
        synthesize_pyqsp_sym_qsp_phases,
    )

    s_min = float(np.linalg.svd(H, compute_uv=False)[-1] / beta)
    bop = fit_bounded_odd_chebyshev(s_min=s_min, lam=lam, degree=255)
    phases = synthesize_pyqsp_sym_qsp_phases(bop.chebyshev_coeffs)
    res = convert_pyqsp_to_production(
        make_request_from_phases(phases, degree=255, configuration_id="ieee14::d255")
    )
    A = H / beta
    pad = 1
    while pad < max(m, n):
        pad *= 2
    M = np.zeros((pad, pad))
    M[:m, :n] = A
    I = np.eye(pad)
    W = np.block([[M, _psd_sqrt(I - M @ M.T)], [_psd_sqrt(I - M.T @ M), -M.T]])
    top = pcphase_qsvt_top_block(W, res.phases, encoded_dimension=pad)
    B_prod = (
        np.imag(top[:pad, :pad])
        if res.extraction_component == "imag"
        else -np.imag(top[:pad, :pad])
    )[:m, :n]
    dx_qsvt = (bop.C_global / beta) * (B_prod.T @ r)
    dx_ridge = np.linalg.solve(H.T @ H + alpha * np.eye(n), H.T @ r)

    l = np.zeros(n)
    l[0] = 1.0  # theta_2
    norm_dx = float(np.linalg.norm(dx_qsvt))
    y_sv = float(l @ dx_qsvt)  # statevector selected output (== headline 0.0049368)
    y_ridge = float(l @ dx_ridge)
    output_state = dx_qsvt / norm_dx  # normalized update direction

    shots_grid = [10_000, 100_000, 1_000_000]
    seeds = list(range(770300, 770306))  # 6 independent backend seeds
    rows = []
    for shots in shots_grid:
        per_seed = []
        for seed in seeds:
            t0 = time.perf_counter()
            est = estimate_overlap(
                l, output_state, shots=int(shots), seed=int(seed), prefer_aer=True
            )
            dt = time.perf_counter() - t0
            mu_shot = est.overlap_estimate
            y_shot = norm_dx * mu_shot
            # 95% CI on y via the binomial overlap variance
            p0 = (1.0 + est.overlap_exact) / 2.0
            se_mu = 2.0 * math.sqrt(max(p0 * (1 - p0), 0.0) / shots)
            se_y = norm_dx * se_mu
            ci_low, ci_high = y_shot - 1.96 * se_y, y_shot + 1.96 * se_y
            per_seed.append(y_shot)
            rows.append(
                dict(
                    shots=shots,
                    backend_seed=seed,
                    backend=est.backend,
                    mu_exact=est.overlap_exact,
                    mu_shot=mu_shot,
                    y_shot=y_shot,
                    y_statevector=y_sv,
                    y_ridge=y_ridge,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    ci_half_width=1.96 * se_y,
                    relative_ci_half_width=(1.96 * se_y) / max(abs(y_shot), 1e-300),
                    theoretical_se_y=se_y,
                    execution_time_s=dt,
                    ci_contains_statevector=bool(ci_low <= y_sv <= ci_high),
                    ci_contains_ridge=bool(ci_low <= y_ridge <= ci_high),
                    transpiler_seed=seed,
                )
            )
        # aggregate over seeds for this shot count
        ys = np.array(per_seed)
        emp_var = float(np.var(ys, ddof=1)) if len(ys) > 1 else float("nan")
        agg = dict(
            shots=shots,
            n_seeds=len(seeds),
            total_shots=shots * len(seeds),
            aggregate_y_estimate=float(np.mean(ys)),
            aggregate_y_std=float(np.std(ys, ddof=1)),
            empirical_variance=emp_var,
            theoretical_variance_per_shot=float(
                (norm_dx * 2 * math.sqrt(max(p0 * (1 - p0), 0.0))) ** 2 / shots
            ),
            y_statevector=y_sv,
            y_ridge=y_ridge,
            aggregate_relative_ci_half_width=(
                1.96 * norm_dx * 2 * math.sqrt(max(p0 * (1 - p0), 0.0) / shots)
            )
            / max(abs(float(np.mean(ys))), 1e-300),
            meets_precision_target=bool(
                (1.96 * norm_dx * 2 * math.sqrt(max(p0 * (1 - p0), 0.0) / shots))
                / max(abs(float(np.mean(ys))), 1e-300)
                <= 0.10
            ),
        )
        rows.append({**agg, "_row_type": "aggregate"})

    pd.DataFrame(rows).to_csv(OUT / "ieee14_high_precision_backend_runs.csv", index=False)
    agg_rows = [r for r in rows if r.get("_row_type") == "aggregate"]
    for r in agg_rows:
        r.pop("_row_type", None)
    pd.DataFrame(agg_rows).to_csv(OUT / "ieee14_high_precision_backend_summary.csv", index=False)

    lines = [
        "# IEEE-14 High-Precision Shot Readout (WP-J)",
        "",
        "Hadamard-test signed-overlap readout on qiskit Aer. State preparation",
        "loads the convention-validated update direction; the READOUT is shot-based.",
        "Selected output = theta_2 (statevector value = l^T dx_qsvt).",
        "",
        f"y_statevector = {y_sv:.10f} (headline 0.004936843087993346)",
        f"y_ridge       = {y_ridge:.10f} (headline 0.004938350292269933)",
        "",
        "| shots | seeds | total shots | agg y estimate | rel CI half-width | meets <=10% |",
        "|---|---|---|---|---|---|",
    ]
    for r in agg_rows:
        lines.append(
            f"| {r['shots']:,} | {r['n_seeds']} | {r['total_shots']:,} | {r['aggregate_y_estimate']:.6f} "
            f"| {r['aggregate_relative_ci_half_width']:.4f} | {r['meets_precision_target']} |"
        )
    high_prec = agg_rows[-1] if agg_rows else {}
    if high_prec:
        lines += [
            "",
            f"Highest-precision setting: {high_prec['shots']:,} shots x {high_prec['n_seeds']} seeds "
            f"=> rel CI half-width {high_prec['aggregate_relative_ci_half_width']:.4f}.",
            f"'High precision' label applies iff rel CI half-width <= 0.10: {high_prec['meets_precision_target']}.",
        ]
    (OUT / "ieee14_high_precision_backend_report.md").write_text("\n".join(lines))
    print(
        f"[WP-J] shot readout: {len(rows)} rows; y_sv={y_sv:.6f}; "
        f"1M-shot rel CI hw={agg_rows[-1]['aggregate_relative_ci_half_width']:.4f}"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
