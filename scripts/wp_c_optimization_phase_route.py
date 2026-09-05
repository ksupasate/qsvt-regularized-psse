"""WP C third route: optimization-based phase synthesis (scipy least_squares) on the pyqsp
sym_qsp response (imag/Wx), warm-started from pyqsp's degree-191 phases. Targets IEEE-14
alpha=1e-4 at degree 255. If this converges below the 1e-2 circuit tolerance it provides an
independent phase-synthesis route where pyqsp's algebraic (laurent/newton) method is unstable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import probe_high_degree_lambda_feasibility as P  # noqa: E402
from pyqsp.angle_sequence import QuantumSignalProcessingPhases  # noqa: E402
from pyqsp.response import ComputeQSPResponse  # noqa: E402

from robust_qsvt_se.qsvt.engineering_utils import (  # noqa: E402
    build_engineering_system,
    ridge_svd_solution,
)

OUT = ROOT / "outputs" / "final_qsvt_feasibility_push"


def response(grid, phases):
    return np.imag(
        np.asarray(
            ComputeQSPResponse(
                np.asarray(grid, dtype=np.float64),
                phases,
                signal_operator="Wx",
                measurement="x",
                sym_qsp=True,
            )["pdat"]
        )
    )


def main() -> None:
    sys_, _ = build_engineering_system(
        {"case_name": "ieee14", "case_source": "pypower", "seed": 123}
    )
    H = np.asarray(sys_.H_tilde, dtype=np.float64)
    r = np.asarray(sys_.r_tilde, dtype=np.float64)
    U, sv, Vt = np.linalg.svd(H, full_matrices=False)
    beta = float(sv.max())
    s_min = float(sv.min() / beta)
    alpha = 1.0e-4
    lam = alpha / beta**2
    g = lambda s: s / (s**2 + lam)  # noqa: E731
    dx_exact = ridge_svd_solution(H, r, alpha=float(alpha))
    s_norm = sv / beta
    # dense optimization grid on occupied interval + a few negative points for parity/boundedness
    opt_grid = np.concatenate([np.linspace(s_min, 1.0, 400), -np.linspace(s_min, 1.0, 200)])
    target_degree = 255
    mm = P.odd_minimax_cheb_coeffs(g, s_min, target_degree)
    coeffs, _ = mm
    cg = float(np.max(np.abs(chebval(np.linspace(-1, 1, 8193), coeffs))))
    target = np.concatenate(
        [g(np.linspace(s_min, 1.0, 400)) / cg, -g(np.linspace(s_min, 1.0, 200)) / cg]
    )  # odd extension
    # warm-start: pyqsp degree-191 phases, pad/truncate to target_degree+1
    mm191 = P.odd_minimax_cheb_coeffs(g, s_min, 191)
    c191, _ = mm191
    cg191 = float(np.max(np.abs(chebval(np.linspace(-1, 1, 8193), c191))))
    from contextlib import redirect_stdout
    from io import StringIO

    buf = StringIO()
    with redirect_stdout(buf):
        res191 = QuantumSignalProcessingPhases(
            np.asarray(c191 / cg191, dtype=np.float64), method="sym_qsp", chebyshev_basis=True
        )
    warm = np.asarray(res191[0], dtype=np.float64)
    x0 = np.zeros(target_degree + 1, dtype=np.float64)
    n = min(warm.size, x0.size)
    x0[:n] = warm[:n]
    print(
        f"warm-start from deg191 phases (n={warm.size}); optimizing deg255 ({x0.size} phases)",
        flush=True,
    )

    def resid(phases):
        return response(opt_grid, phases) - target

    sol = least_squares(resid, x0, max_nfev=4000, ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=0)
    phases_opt = sol.x
    occ_err = float(np.max(np.abs(response(s_norm, phases_opt) - g(s_norm) / cg)))
    bounded = float(np.max(np.abs(response(np.linspace(-1, 1, 8193), phases_opt))))
    recovered = (cg / beta) * (Vt.T @ (response(s_norm, phases_opt) * (U.T @ r)))
    rel = float(np.linalg.norm(recovered - dx_exact) / np.linalg.norm(dx_exact))
    out = {
        "route": "scipy_least_squares_warmstart_from_pyqsp191",
        "degree": target_degree,
        "alpha": alpha,
        "lambda": lam,
        "C_global": cg,
        "phase_realized_occ_err": occ_err,
        "global_max_response": bounded,
        "bounded_le_1": bool(bounded <= 1.0 + 1e-7),
        "update_rel_err_vs_ridge": rel,
        "n_phases": int(phases_opt.size),
        "passes_1e-2": bool(rel <= 1e-2 and occ_err <= 1e-2),
        "passes_1e-3": bool(rel <= 1e-3 and occ_err <= 1e-3),
        "least_squares_cost": float(sol.cost),
        "nfev": int(sol.nfev),
    }
    print(json.dumps(out, indent=2), flush=True)
    (OUT / "wp_c_optimization_route_result.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
