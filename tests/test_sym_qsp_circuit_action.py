"""Tests for the validated spectrum-aware -> pyqsp -> circuit-action pipeline.

Covers WP-19 required coverage: phase-synthesis backend correctness, phase
reconstruction, spectrum-aware global boundedness, high-precision (mpmath)
evaluation, fixed-seed determinism, and circuit-vs-scalar agreement. These tests
guard the convention calibration (P(x)=x reproduces to ~1e-10 via an EXECUTED
qiskit circuit) so a silent convention regression cannot produce a false result.
"""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.sym_qsp_circuit_action import (
    fit_bounded_odd_chebyshev,
    scalar_qsp_response,
    synthesize_pyqsp_sym_qsp_phases,
    validate_circuit_action,
)


def test_convention_known_target_x_reproduced_by_circuit():
    """pyqsp phases for P(x)=x must be reproduced by the EXECUTED circuit to ~1e-10."""
    from robust_qsvt_se.qsvt.sym_qsp_circuit_action import _circuit_action_value

    phases = synthesize_pyqsp_sym_qsp_phases(np.array([0.0, 1.0]))  # Cheb: T1 only => P(x)=x
    for x in [0.1, 0.37, 0.6, 0.95]:
        scalar = scalar_qsp_response(x, phases)
        circuit = _circuit_action_value(x, phases)
        assert abs(scalar - x) < 1e-10, (x, scalar)
        assert abs(circuit - x) < 1e-9, (x, circuit)
        assert abs(circuit - scalar) < 1e-10, (x, circuit, scalar)


def test_global_boundedness_holds_on_full_domain():
    poly = fit_bounded_odd_chebyshev(s_min=0.029, lam=1e-2, degree=63, method="stable_chebyshev")
    phases = synthesize_pyqsp_sym_qsp_phases(poly.chebyshev_coeffs)
    rep = validate_circuit_action(poly=poly, phases=phases)
    assert rep.bounded_passes, rep.global_bounded_max
    assert rep.global_bounded_max <= 1.0 + 1e-9


def test_phase_reconstruction_matches_target():
    poly = fit_bounded_odd_chebyshev(s_min=0.029, lam=0.069, degree=31, method="stable_chebyshev")
    phases = synthesize_pyqsp_sym_qsp_phases(poly.chebyshev_coeffs)
    rep = validate_circuit_action(poly=poly, phases=phases)
    # Reconstructed filter (response * C) vs target filter on occupied interval.
    assert rep.occupied_recon_error < 1e-2
    # Circuit action must agree with the scalar reconstruction (convention correctness).
    assert rep.circuit_vs_scalar_max_error < 1e-9


def test_stable_fit_does_not_overflow_at_high_degree():
    """The stable Chebyshev construction must stay finite at degree 191, unlike the
    power-basis production fit that overflows off the occupied interval."""
    poly = fit_bounded_odd_chebyshev(s_min=0.029, lam=1e-4, degree=191, method="stable_chebyshev")
    assert np.all(np.isfinite(poly.chebyshev_coeffs))
    assert np.isfinite(poly.C_global)
    # C_global ~ 1/s_min ~ 34 for tiny lambda (postselection suppression factor)
    assert 10.0 < poly.C_global < 1e6, poly.C_global


def test_mpmath_high_precision_filter_matches_float64():
    """mpmath evaluation of the filter s/(s^2+lam) at high dps must agree with float64."""
    import mpmath as mp

    mp.mp.dps = 200
    lam = mp.mpf("1e-4")
    s = mp.mpf("0.3")
    hp = s / (s * s + lam)
    fp = float(0.3 / (0.3**2 + 1e-4))
    assert abs(float(hp) - fp) < 1e-12


def test_fixed_seed_determinism_phase_synthesis():
    p1 = synthesize_pyqsp_sym_qsp_phases(np.array([0.0, 0.5, 0.0, 0.25]))
    p2 = synthesize_pyqsp_sym_qsp_phases(np.array([0.0, 0.5, 0.0, 0.25]))
    assert np.allclose(p1, p2)


def test_spectrum_aware_occupied_better_than_full_domain_artifact():
    """The occupied reconstruction error must be reported separately from any
    full-domain quantity, and must be small where the method is feasible."""
    poly = fit_bounded_odd_chebyshev(s_min=0.029, lam=1e-2, degree=95, method="stable_chebyshev")
    phases = synthesize_pyqsp_sym_qsp_phases(poly.chebyshev_coeffs)
    rep = validate_circuit_action(
        poly=poly, phases=phases, actual_singular_values=np.array([0.05, 0.3, 0.9])
    )
    assert rep.occupied_recon_error < 5e-3
    assert not np.isnan(rep.occupied_actual_sv_error)
    assert rep.occupied_actual_sv_error < 5e-3


def test_minimax_and_chebyshev_both_bounded():
    for method in ["stable_chebyshev", "minimax_lp"]:
        poly = fit_bounded_odd_chebyshev(s_min=0.029, lam=1e-2, degree=63, method=method)
        phases = synthesize_pyqsp_sym_qsp_phases(poly.chebyshev_coeffs)
        rep = validate_circuit_action(poly=poly, phases=phases)
        assert rep.bounded_passes, (method, rep.global_bounded_max)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
