"""Updated classical baselines for the generalized QSVT study (section 18).

For the IEEE-14 headline configuration, times and compares classical solvers for
the SAME regularized-inverse problem the QSVT circuit approximates:
  dense Ridge, sparse-direct Ridge, adjoint selected-output solve, CG on the
  normal equations, LSQR, LSMR, classical application of the same polynomial,
  fixed-step Krylov. 30 timing repetitions for the small case.

The goal is an honest comparison: classical methods are FAST and EXACT here; the
QSVT contribution is NOT a speedup. We do NOT suppress classical superiority.

Outputs: generalized_classical_baselines.csv, generalized_classical_timings.csv,
         generalized_classical_report.md
"""

# ruff: noqa: E501

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import Chebyshev
from scipy.linalg import lu_factor, lu_solve

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import fit_bounded_odd_chebyshev

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
REPS = 30


def _time(fn, reps=REPS):
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        xs.append(time.perf_counter() - t0)
    return float(np.median(xs)), float(np.mean(xs)), out


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
    Hs = sp.csr_matrix(H)
    # normal-equation matrix (n x n) and vector
    G = H.T @ H + alpha * np.eye(n)
    Gs = sp.csr_matrix(G)
    b = H.T @ r
    # reference: dense Ridge solve
    dx_ref = np.linalg.solve(G, b)

    # classical polynomial reference (the SAME degree-255 bounded odd poly, applied classically)
    s_min = float(np.linalg.svd(H, compute_uv=False)[-1] / beta)
    bop = fit_bounded_odd_chebyshev(s_min=s_min, lam=lam, degree=255)
    Pcheb = Chebyshev(bop.chebyshev_coeffs, domain=[-1, 1])
    U, sv, Vh = np.linalg.svd(H, full_matrices=False)
    sigma = sv / beta
    penc = np.array([float(Pcheb(min(1.0, x))) for x in sigma])
    dx_poly = (bop.C_global / beta) * (Vh.T @ (penc * (U.T @ r)))

    rows = []
    timing_rows = []

    # dense Ridge (factor + solve)
    t_fac_med, _t_fac_mean, _ = _time(lambda: lu_factor(G), reps=5)
    lu = lu_factor(G)
    t_solve_med, t_solve_mean, _dx_dense = _time(lambda: lu_solve(lu, b))
    rows.append(
        dict(
            method="dense_Ridge",
            preprocessing="LU factor (n x n)",
            factorization_s=f"{t_fac_med:.2e}",
            per_query_s=f"{t_solve_med:.2e}",
            matvecs=0,
            memory=f"{G.nbytes / 1e3:.1f} KB",
            output_error_vs_ridge=0.0,
            note="exact dense solve; reference",
        )
    )
    timing_rows.append(
        dict(method="dense_Ridge", median_s=t_solve_med, mean_s=t_solve_mean, reps=REPS)
    )

    # sparse direct Ridge
    t_sp_fac, _, solve = _time(lambda: spla.splu(Gs), reps=5)
    t_sp_solve, t_sp_mean, dx_sp = _time(lambda: solve.solve(b))
    rows.append(
        dict(
            method="sparse_direct_Ridge",
            preprocessing="SPARSE LU",
            factorization_s=f"{t_sp_fac:.2e}",
            per_query_s=f"{t_sp_solve:.2e}",
            matvecs=0,
            memory=f"{Gs.data.nbytes / 1e3:.1f} KB",
            output_error_vs_ridge=float(np.linalg.norm(dx_sp - dx_ref)),
            note="sparse direct; identical answer to dense",
        )
    )
    timing_rows.append(
        dict(method="sparse_direct_Ridge", median_s=t_sp_solve, mean_s=t_sp_mean, reps=REPS)
    )

    # adjoint selected-output solve (single output theta_2 = index 0): solve G e_0 then dot b
    e0 = np.zeros(n)
    e0[0] = 1.0
    t_adj_med, t_adj_mean, _ = _time(lambda: lu_solve(lu, e0))
    # selected output = e0^T G^{-1} b = (G^{-T} e0)^T b
    g_inv_e0 = lu_solve(lu, e0)
    y_adj = float(g_inv_e0 @ b)
    y_ref = float(dx_ref[0])
    rows.append(
        dict(
            method="adjoint_selected_output",
            preprocessing="reuse LU",
            factorization_s="0 (amortized)",
            per_query_s=f"{t_adj_med:.2e}",
            matvecs=0,
            memory="shared",
            output_error_vs_ridge=abs(y_adj - y_ref) / max(abs(y_ref), 1e-300),
            note="single selected output via adjoint; cheapest classical route",
        )
    )
    timing_rows.append(
        dict(method="adjoint_selected_output", median_s=t_adj_med, mean_s=t_adj_mean, reps=REPS)
    )

    # CG on normal equations
    def cg():
        x, info = spla.cg(Gs, b, rtol=1e-10, maxiter=2 * n)
        return x, info

    t_cg, _, (dx_cg, info) = _time(cg, reps=10)
    rows.append(
        dict(
            method="CG_normal_equations",
            preprocessing="none",
            factorization_s="0",
            per_query_s=f"{t_cg:.2e}",
            matvecs=int(2 * n),
            memory=f"{Gs.data.nbytes / 1e3:.1f} KB",
            output_error_vs_ridge=float(np.linalg.norm(dx_cg - dx_ref)),
            note=f"iterative; info={info}",
        )
    )
    timing_rows.append(dict(method="CG_normal_equations", median_s=t_cg, mean_s=t_cg, reps=10))

    # LSQR
    def lsqr():
        res = spla.lsqr(Hs, r, damp=math.sqrt(alpha), atol=1e-10, btol=1e-10)[:1][0]
        return res

    t_lsqr, _, dx_lsqr = _time(lsqr, reps=10)
    rows.append(
        dict(
            method="LSQR",
            preprocessing="none",
            factorization_s="0",
            per_query_s=f"{t_lsqr:.2e}",
            matvecs=0,
            memory=f"{Hs.data.nbytes / 1e3:.1f} KB",
            output_error_vs_ridge=float(np.linalg.norm(dx_lsqr - dx_ref)),
            note="iterative on (H, r) with Tikhonov damp=sqrt(alpha)",
        )
    )
    timing_rows.append(dict(method="LSQR", median_s=t_lsqr, mean_s=t_lsqr, reps=10))

    # LSMR
    def lsmr():
        res = spla.lsmr(Hs, r, damp=math.sqrt(alpha), atol=1e-10, btol=1e-10)[0]
        return res

    t_lsmr, _, dx_lsmr = _time(lsmr, reps=10)
    rows.append(
        dict(
            method="LSMR",
            preprocessing="none",
            factorization_s="0",
            per_query_s=f"{t_lsmr:.2e}",
            matvecs=0,
            memory=f"{Hs.data.nbytes / 1e3:.1f} KB",
            output_error_vs_ridge=float(np.linalg.norm(dx_lsmr - dx_ref)),
            note="iterative",
        )
    )
    timing_rows.append(dict(method="LSMR", median_s=t_lsmr, mean_s=t_lsmr, reps=10))

    # classical application of the same polynomial (the QSVT target, computed classically)
    def clpoly():
        return (bop.C_global / beta) * (Vh.T @ (penc * (U.T @ r)))

    t_poly, _, _ = _time(clpoly, reps=30)
    rows.append(
        dict(
            method="classical_polynomial_action",
            preprocessing="SVD",
            factorization_s="SVD (cached)",
            per_query_s=f"{t_poly:.2e}",
            matvecs=2,
            memory=f"{H.nbytes / 1e3:.1f} KB",
            output_error_vs_ridge=float(np.linalg.norm(dx_poly - dx_ref)),
            note="the SAME degree-255 filter applied classically; matches QSVT target by construction",
        )
    )
    timing_rows.append(
        dict(method="classical_polynomial_action", median_s=t_poly, mean_s=t_poly, reps=30)
    )

    # fixed-step Krylov (minres-like on G x = b, fixed 10 steps)
    def krylov():
        return spla.cg(Gs, b, rtol=1e-8, maxiter=10)[0]

    t_kry, _, dx_kry = _time(krylov, reps=10)
    rows.append(
        dict(
            method="fixed_step_Krylov",
            preprocessing="none",
            factorization_s="0",
            per_query_s=f"{t_kry:.2e}",
            matvecs=10,
            memory=f"{Gs.data.nbytes / 1e3:.1f} KB",
            output_error_vs_ridge=float(np.linalg.norm(dx_kry - dx_ref)),
            note="CG capped at 10 steps",
        )
    )
    timing_rows.append(dict(method="fixed_step_Krylov", median_s=t_kry, mean_s=t_kry, reps=10))

    pd.DataFrame(rows).to_csv(OUT / "generalized_classical_baselines.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUT / "generalized_classical_timings.csv", index=False)

    lines = [
        "# Classical Baselines (section 18)",
        "",
        f"IEEE-14 [{m}x{n}], alpha={alpha:.4f}. {REPS} timing reps (10 for iterative, 5 for factorizations).",
        "Classical methods are exact and microsecond-to-millisecond; the QSVT",
        "contribution is NOT a speedup over these.",
        "",
        "| method | per-query (s) | output err vs Ridge | note |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['per_query_s']} | {r['output_error_vs_ridge']:.2e} | {r['note']} |"
        )
    lines += [
        "",
        "Repeated-query amortization: dense/sparse factorizations are reused across",
        "queries; the adjoint selected-output route reuses the LU factorization and is the",
        "cheapest classical way to obtain a single selected output.",
    ]
    (OUT / "generalized_classical_report.md").write_text("\n".join(lines))
    print(
        f"[section 18] classical baselines: {len(rows)} methods; dense Ridge per-query {t_solve_med:.2e}s"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
