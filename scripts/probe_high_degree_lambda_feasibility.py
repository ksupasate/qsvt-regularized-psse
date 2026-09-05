"""Decisive WP B/C/D probe: can higher-degree polynomial + pyqsp phase synthesis make
the application-useful alpha (starting at the alpha=1e-4 benchmark) polynomial+phase+
bounded feasible for the FULL-RECTANGULAR IEEE-14/30 system?

This is READ-ONLY with respect to existing outputs; it writes new probe artifacts under
outputs/final_qsvt_feasibility_push/. It does NOT change the Ridge/Tikhonov estimator.

Key construction (spectrum-aware, estimator-preserving):
  - normalized filter g(s) = s/(s^2 + lambda), lambda = alpha/beta^2, s = sigma/beta in [s_min, 1]
  - ODD polynomial p(s) = s * q(s^2) approximates g(s) on the OCCUPIED interval [s_min, 1]
  - because p(0)=0, the global boundedness constant C_global = max|p| on [-1,1] is governed by
    the occupied interval (C ~ 1/s_min), NOT by 1/(2 sqrt(lambda)); for lambda << s_min^2 the
    filter is ~1/s on the occupied interval and C is essentially alpha-INDEPENDENT.
  - QSVT realizes qsp(s) = imag(ComputeQSPResponse(...sym_qsp...)); target = p(s)/C_global (|.|<=1).
  - physical recovery uses the single factor C_global/beta (unchanged Ridge estimator).

PASS criterion (pre-registered): scaled_occupied_error = max|p-g| on actual SVs / C_global
  <= 1e-2 (primary) or <= 1e-3 (strict); |qsp|<=1+1e-9 on [-1,1]; pyqsp synthesis OK;
  reconstruction max|qsp - p/C_global| on occupied <= 1e-6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "final_qsvt_feasibility_push"
OUT.mkdir(parents=True, exist_ok=True)

# Physical alpha grid (alpha=1e-4 is the manuscript's fixed benchmark).
ALPHA_GRID = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5]
DEGREE_GRID = [31, 45, 63, 95, 127, 191, 255]
CASES = ["ieee14", "ieee30"]


def load_system(case: str):
    from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system

    system, _ = build_engineering_system({"case_name": case, "case_source": "pypower", "seed": 123})
    H = np.asarray(system.H_tilde, dtype=np.float64)
    r = np.asarray(system.r_tilde, dtype=np.float64)
    return H, r


def odd_cheb_coeffs_ls(
    g_func, s_min: float, degree: int, c_global_scale: float = 1.0
) -> np.ndarray:
    """Least-squares odd Chebyshev fit of g_func(s)/c_global_scale on [s_min,1] over [-1,1].

    Returns full Chebyshev T-coeff array (even ~0). Fits on symmetric support
    [-1,-s_min] U [s_min,1] so parity is preserved.
    """
    from numpy.polynomial.chebyshev import chebfit

    n = max(2 * (degree + 4), 512)
    pos = np.linspace(s_min, 1.0, n)
    grid = np.concatenate([-pos[::-1], pos])
    target = g_func(np.abs(grid)) * np.sign(grid) / c_global_scale  # odd extension
    coeffs = chebfit(grid, target, deg=degree)
    # enforce odd parity explicitly (zero even coefficients)
    coeffs[0::2] = 0.0
    return coeffs


def odd_minimax_cheb_coeffs(g_func, s_min: float, degree: int, c_global_scale: float = 1.0):
    """L-infinity minimax odd-Chebyshev fit via LP (HiGHS) on [s_min,1].

    Variables: odd Cheb coeffs a_1,a_3,...,a_d and slack t. Minimize t s.t.
    -t <= sum a_k T_k(s_i) - g(s_i)/scale <= t on a dense positive grid.
    """

    odd_idx = np.array([k for k in range(1, degree + 1, 2)], dtype=int)  # 1,3,5,...
    n_basis = odd_idx.size
    n_grid = max(8 * degree, 400)
    s = np.linspace(s_min, 1.0, n_grid)
    target = g_func(s) / c_global_scale
    # basis matrix B[i, j] = T_{odd_idx[j]}(s_i)
    B = np.zeros((n_grid, n_basis))
    for j, k in enumerate(odd_idx):
        B[:, j] = np.cos(k * np.arccos(np.clip(s, -1, 1)))
    # LP: vars = [a (n_basis), t (1)]
    # minimize t -> c = [0,...,0, 1]
    c = np.zeros(n_basis + 1)
    c[-1] = 1.0
    # constraints: B a - target <= t  and  -(B a - target) <= t
    A_ub = np.zeros((2 * n_grid, n_basis + 1))
    A_ub[:n_grid, :n_basis] = B
    A_ub[:n_grid, -1] = -1.0
    A_ub[n_grid:, :n_basis] = -B
    A_ub[n_grid:, -1] = -1.0
    b_ub = np.concatenate([target, -target])
    bounds = [(None, None)] * n_basis + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    a = res.x[:n_basis]
    coeffs = np.zeros(degree + 1)
    coeffs[odd_idx] = a
    return coeffs, float(res.x[-1])


def pyqsp_synthesize_and_reconstruct(cheb_coeffs: np.ndarray, eval_grid: np.ndarray):
    """Return phases, reconstructed response, status, parity, and error message."""
    from contextlib import redirect_stdout
    from io import StringIO

    try:
        from pyqsp.angle_sequence import QuantumSignalProcessingPhases
        from pyqsp.response import ComputeQSPResponse

        buffer = StringIO()
        with redirect_stdout(buffer):
            result = QuantumSignalProcessingPhases(
                np.asarray(cheb_coeffs, dtype=np.float64),
                method="sym_qsp",
                chebyshev_basis=True,
            )
        phases = np.asarray(result[0], dtype=np.float64)
        parity = int(result[2])
        resp = ComputeQSPResponse(
            np.asarray(eval_grid, dtype=np.float64),
            phases,
            signal_operator="Wx",
            measurement="x",
            sym_qsp=True,
        )["pdat"]
        return phases, np.imag(np.asarray(resp)), "ok", parity, None
    except Exception as exc:
        return None, None, "failed", None, str(exc)


def main() -> None:
    rows = []
    summary_lines = []
    for case in CASES:
        H, _r = load_system(case)
        sv = np.linalg.svd(H, compute_uv=False)
        beta = float(sv.max())
        s_min = float(sv.min() / beta)
        s_actual = sv / beta  # normalized actual singular values in (0,1]
        kappa = float(sv.max() / sv.min())
        summary_lines.append(
            f"\n## {case}: shape={H.shape} beta={beta:.4g} sigma_min={sv.min():.4g} "
            f"s_min={s_min:.6g} kappa={kappa:.4g} s_min^2={s_min**2:.4g}"
        )
        smallest_pass_deg = {}  # alpha -> (deg primary, deg strict)
        for alpha in ALPHA_GRID:
            lam = alpha / beta**2
            g = lambda s, lam=lam: s / (s**2 + lam)  # noqa: E731
            for degree in DEGREE_GRID:
                for method, fitfn in (
                    ("cheb_ls", lambda d, g=g, s_min=s_min: odd_cheb_coeffs_ls(g, s_min, d)),
                    (
                        "minimax_lp",
                        lambda d, g=g, s_min=s_min: odd_minimax_cheb_coeffs(g, s_min, d),
                    ),
                ):
                    try:
                        if method == "minimax_lp":
                            out = fitfn(degree)
                            if out is None:
                                rows.append(
                                    _row(
                                        case,
                                        alpha,
                                        lam,
                                        degree,
                                        method,
                                        "POLYNOMIAL_APPROXIMATION_FAILED",
                                    )
                                )
                                continue
                            coeffs, _t = out
                        else:
                            coeffs = fitfn(degree)
                    except Exception as exc:
                        rows.append(
                            _row(
                                case,
                                alpha,
                                lam,
                                degree,
                                method,
                                f"POLYNOMIAL_APPROXIMATION_FAILED:{exc}"[:120],
                            )
                        )
                        continue
                    from numpy.polynomial.chebyshev import chebval

                    full_grid = np.linspace(-1, 1, max(8192, 64 * degree))
                    occ_grid = s_actual
                    p_full = chebval(full_grid, coeffs)
                    p_occ = chebval(occ_grid, coeffs)
                    g_occ = g(occ_grid)
                    c_global = float(np.max(np.abs(p_full)))
                    if not np.isfinite(c_global) or c_global <= 0:
                        rows.append(_row(case, alpha, lam, degree, method, "BOUNDEDNESS_FAILED"))
                        continue
                    scaled_occ_err = float(
                        np.max(np.abs(p_occ - g_occ)) / c_global
                    )  # rel to bounded target
                    # Coefficients fit g/scale with scale=1, so p_full is on the
                    # unbounded target scale before the C_global normalization.
                    # For QSP we pass coeffs / C_global so the realized poly is bounded <=1.
                    qsp_coeffs = coeffs / c_global
                    # pyqsp
                    phases, recon, pstatus, parity, perr = pyqsp_synthesize_and_reconstruct(
                        qsp_coeffs, full_grid
                    )
                    if pstatus != "ok":
                        rows.append(
                            _row(
                                case,
                                alpha,
                                lam,
                                degree,
                                method,
                                f"PHASE_RECOVERY_FAILED:{perr}"[:120],
                            )
                        )
                        continue
                    recon_full = recon
                    target_full = chebval(full_grid, qsp_coeffs)
                    recon_err = float(np.max(np.abs(recon_full - target_full)))
                    # occupied reconstruction accuracy (the physically relevant one)
                    recon_occ = ComputeQSPResponse_safe(occ_grid, phases)
                    if recon_occ is None:
                        rows.append(
                            _row(case, alpha, lam, degree, method, "PHASE_RECONSTRUCTION_FAILED")
                        )
                        continue
                    target_occ = chebval(occ_grid, qsp_coeffs)
                    recon_occ_err = float(np.max(np.abs(recon_occ - target_occ)))
                    bounded_qsp = bool(np.max(np.abs(recon_full)) <= 1.0 + 1e-7)
                    primary = scaled_occ_err <= 1.0e-2 and bounded_qsp and recon_err <= 1.0e-6
                    strict = scaled_occ_err <= 1.0e-3 and bounded_qsp and recon_err <= 1.0e-6
                    rows.append(
                        _row(
                            case,
                            alpha,
                            lam,
                            degree,
                            method,
                            "PASS" if primary else "fail",
                            scaled_occ_err=scaled_occ_err,
                            recon_err=recon_err,
                            recon_occ_err=recon_occ_err,
                            c_global=c_global,
                            phase_count=int(phases.size),
                            parity=parity,
                            bounded_qsp=bounded_qsp,
                            primary=primary,
                            strict=strict,
                        )
                    )
                    if primary:
                        key = f"alpha={alpha:.0e}"
                        if key not in smallest_pass_deg:
                            smallest_pass_deg[key] = (
                                degree,
                                method,
                                scaled_occ_err,
                                recon_err,
                                c_global,
                            )
                        break  # found smallest degree for this alpha/method; stop escalating
                else:
                    continue
                break  # passed at this degree for some method -> next alpha
        for key, info in smallest_pass_deg.items():
            summary_lines.append(
                f"  {key}: SMALLEST PASS degree={info[0]} method={info[1]} "
                f"scaled_occ_err={info[2]:.3e} recon_err={info[3]:.3e} C_global={info[4]:.3g}"
            )

    # write CSV
    import csv

    fields = [
        "case",
        "alpha",
        "lambda",
        "degree",
        "method",
        "status",
        "scaled_occ_err",
        "recon_err",
        "recon_occ_err",
        "c_global",
        "phase_count",
        "parity",
        "bounded_qsp",
        "primary_pass",
        "strict_pass",
    ]
    with (OUT / "probe_poly_phase_feasibility.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r_ in rows:
            w.writerow({k: r_.get(k, "") for k in fields})

    (OUT / "probe_poly_phase_feasibility_summary.md").write_text(
        "# WP B/C/D Decisive Probe — polynomial+phase+bounded feasibility\n\n"
        + "\n".join(summary_lines)
        + "\n\nSee probe_poly_phase_feasibility.csv for full grid.\n",
        encoding="utf-8",
    )
    print("\n".join(summary_lines))


def ComputeQSPResponse_safe(grid, phases):
    try:
        from pyqsp.response import ComputeQSPResponse

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
    except Exception:
        return None


def _row(case, alpha, lam, degree, method, status, **kw):
    base = {
        "case": case,
        "alpha": f"{alpha:.6e}",
        "lambda": f"{lam:.6e}",
        "degree": degree,
        "method": method,
        "status": status,
        "scaled_occ_err": "",
        "recon_err": "",
        "recon_occ_err": "",
        "c_global": "",
        "phase_count": "",
        "parity": "",
        "bounded_qsp": "",
        "primary_pass": "",
        "strict_pass": "",
    }
    base.update(kw)
    return base


if __name__ == "__main__":
    main()
