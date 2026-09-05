"""Stable spectrum-aware polynomial -> pyqsp phases -> executed circuit-action pipeline.

This module implements the validated high-degree QSVT/QSP feasibility path used by
the final QSVT feasibility push. It closes three gaps that previously forced the
QSVT-realizable regularization to stay near lambda in [0.02, 0.069]:

1. Numerical stability: the production ``odd_chebyshev_reduced_y`` fit returns
   *power-basis* coefficients that overflow / diverge off the occupied interval at
   degree >= ~63 (Runge phenomenon in the monomial basis). Here every polynomial is
   constructed and evaluated **entirely in the Chebyshev basis** (Clenshaw), so it
   remains stable to degree >= 255.
2. High-degree phase synthesis: phases are produced by pyqsp ``sym_qsp`` (PGV),
   which is empirically stable to degree >= 255, unlike the PennyLane primary
   backend that is capped at degree <= 35.
3. Executable validation: phases are validated by an **executed** qiskit
   ``Statevector`` circuit implementing the symmetric-QSP signal model, not by a
   scalar phase-response proxy alone.

Convention (empirically calibrated against the known target P(x)=x on
2026-07-09): pyqsp ``sym_qsp`` phases reproduce the target polynomial under the
signal operator ``[[x, +i sqrt(1-x^2)], [+i sqrt(1-x^2), x]]`` ("plus_i",
identical to the repository ``repository_plus_i`` convention), response component
``imag( <0|U|0> )``, product order ``R(phi0) W R(phi1) W ...`` ("existing_order").
The qiskit circuit reproduces the scalar matrix-product response to machine
precision (error ~1e-13 on the known target).

LABEL: the scalar ``ComputeQSPResponse``/matrix-product checks are
``EXECUTED_STATEVECTOR`` for the scalar signal and ``DIAGNOSTIC_ONLY`` for the
matrix-product form; the qiskit ``Statevector`` circuit-action check is
``EXECUTED_CIRCUIT`` (statevector backend, noise-free). None of this is hardware
execution, quantum speedup, or QSVT-over-Ridge superiority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial import Chebyshev

from robust_qsvt_se.qsvt.polynomial import regularized_filter_on_normalized_domain

# Calibrated symmetric-QSP convention (see module docstring).
SIGNAL_CONVENTION = "plus_i"
RESPONSE_COMPONENT = "imag"
PRODUCT_ORDER = "existing_order"

Method = Literal["stable_chebyshev", "minimax_lp", "chebyshev_ls"]

_CIRCUIT_ACTION_CLAIM = (
    "Circuit-action validation uses the qiskit quantum_info.Statevector simulator "
    "on the symmetric-QSP signal model. It is executed-circuit evidence at the "
    "scalar-signal / small-matrix scale only; it is not hardware execution, not a "
    "fault-tolerant resource estimate, and not evidence of quantum speedup or "
    "QSVT-over-Ridge numerical superiority."
)


@dataclass(frozen=True, slots=True)
class BoundedOddPolynomial:
    """A bounded, pure-odd Chebyshev polynomial approximating the Ridge filter.

    ``coeffs`` are the *bounded* Chebyshev coefficients (max |P| <= 1 on [-1,1]).
    ``C_global`` is the scale such that ``C_global * P(s)`` recovers the physical
    Ridge filter ``s/(s^2 + lambda)`` on the occupied interval. ``C_global`` is the
    QSVT success-amplitude suppression factor (postselection scales like ~1/C^2).
    """

    chebyshev_coeffs: np.ndarray
    C_global: float
    occupied_max_error: float
    occupied_max_target: float
    global_max_abs: float
    method: str
    degree: int
    s_min: float
    lam: float


@dataclass(frozen=True, slots=True)
class CircuitActionReport:
    phase_count: int
    occupied_recon_error: float
    occupied_actual_sv_error: float
    global_bounded_max: float
    circuit_vs_scalar_max_error: float
    circuit_vs_target_max_error: float
    n_test_points: int
    evidence_label: str
    bounded_passes: bool


def _signal_operator(x: float) -> np.ndarray:
    off = 1j * np.sqrt(max(0.0, 1.0 - x * x))
    return np.array([[x, off], [off, x]], dtype=np.complex128)


def _phase_rotation(phi: float) -> np.ndarray:
    return np.array([[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]], dtype=np.complex128)


def scalar_qsp_response(x: float, phases: np.ndarray) -> float:
    """Symmetric-QSP response ``imag(<0| R W R W ... |0>)`` (calibrated convention)."""
    signal = _signal_operator(float(x))
    unitary = _phase_rotation(float(phases[0]))
    for phase in phases[1:]:
        unitary = unitary @ signal @ _phase_rotation(float(phase))
    return float(np.imag(unitary[0, 0]))


def synthesize_pyqsp_sym_qsp_phases(chebyshev_coeffs: np.ndarray) -> np.ndarray:
    """Return pyqsp ``sym_qsp`` phases for (bounded, pure-parity) Chebyshev coeffs."""
    from contextlib import redirect_stdout
    from io import StringIO

    from pyqsp.angle_sequence import QuantumSignalProcessingPhases

    coeffs = np.asarray(chebyshev_coeffs, dtype=np.float64)
    if coeffs.ndim != 1 or coeffs.size < 2:
        raise ValueError("need at least 2 Chebyshev coefficients")
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = QuantumSignalProcessingPhases(coeffs, method="sym_qsp", chebyshev_basis=True)
    return np.asarray(result[0], dtype=np.float64)


def fit_bounded_odd_chebyshev(
    *,
    s_min: float,
    lam: float,
    degree: int,
    method: Method = "stable_chebyshev",
    grid_size: int | None = None,
) -> BoundedOddPolynomial:
    """Fit a bounded pure-odd Chebyshev polynomial to the filter ``s/(s^2+lam)``.

    The fit is spectrum-aware: it approximates the filter on the *occupied* interval
    ``[s_min, 1]`` only (via the reduced-``y`` map ``q(y)=1/(y+lam)``, ``y=s^2``),
    then enforces global boundedness by scaling with ``C_global = max|P|`` on a dense
    ``[-1,1]`` grid. The Chebyshev basis is used throughout, so the result is stable
    at high degree (no power-basis overflow).
    """
    if not 0.0 < s_min < 1.0:
        raise ValueError("s_min must lie in (0, 1)")
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    if degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    n_grid = int(grid_size or max(2048, 8 * degree))

    if method == "stable_chebyshev":
        y_min = s_min * s_min
        y_grid = np.linspace(y_min, 1.0, n_grid)
        # q(y) = 1/(y+lam): p(s) = s*q(s^2) reproduces s/(s^2+lam) on occupied
        q_cheb = Chebyshev.fit(
            y_grid, 1.0 / (y_grid + lam), deg=(degree - 1) // 2, domain=[y_min, 1.0]
        )
        dense = np.linspace(-1.0, 1.0, max(8193, 32 * degree + 1))
        poly_values = dense * q_cheb(dense**2)  # stable Clenshaw evaluation of q
        p_cheb = Chebyshev.fit(dense, poly_values, deg=degree, domain=[-1.0, 1.0])
        p_cheb.coef[0::2] = 0.0  # enforce pure odd parity
    elif method in {"minimax_lp", "chebyshev_ls"}:
        # Delegate to the production approximation harness, then re-derive a stable
        # bounded odd Chebyshev polynomial from its coefficients.
        from robust_qsvt_se.qsvt.polynomial_approximation import (
            _fit_odd_chebyshev_ls,
            _fit_odd_minimax_lp,
            as_odd_degree,
        )

        grid = np.linspace(s_min, 1.0, n_grid)
        bounded_target = grid / (grid**2 + lam)
        c_occ = float(np.max(np.abs(bounded_target))) or 1.0
        fit_grid = np.unique(np.concatenate([grid]))
        fit_target = (fit_grid / (fit_grid**2 + lam)) / c_occ
        if method == "minimax_lp":
            _, _, _, cheb_coeffs, _ = _fit_odd_minimax_lp(
                fit_grid=fit_grid,
                fit_target=fit_target,
                evaluation_points=fit_grid,
                degree=as_odd_degree(degree),
            )
        else:
            _, _, _, cheb_coeffs, _ = _fit_odd_chebyshev_ls(
                fit_grid=fit_grid,
                fit_target=fit_target,
                evaluation_points=fit_grid,
                degree=as_odd_degree(degree),
            )
        cheb_coeffs = np.asarray(cheb_coeffs, dtype=np.float64)
        # Re-scale the bounded (on occupied) poly to be globally bounded on [-1,1].
        dense = np.linspace(-1.0, 1.0, max(8193, 32 * degree + 1))
        eval_cheb = Chebyshev(cheb_coeffs)
        poly_values = eval_cheb(dense) * c_occ
        p_cheb = Chebyshev.fit(dense, poly_values, deg=degree, domain=[-1.0, 1.0])
        p_cheb.coef[0::2] = 0.0
    else:
        raise ValueError(f"unknown method: {method}")

    dense = np.linspace(-1.0, 1.0, max(8193, 32 * degree + 1))
    global_max = float(np.max(np.abs(p_cheb(dense))))
    c_global = max(global_max, np.finfo(float).eps)
    bounded_coeffs = p_cheb.coef / c_global
    bounded_coeffs[0::2] = 0.0

    occ = np.linspace(s_min, 1.0, max(2048, 8 * degree))
    target_occ = regularized_filter_on_normalized_domain(
        occ, alpha=lam, block_encoding_normalization=1.0
    )
    occ_max_target = float(np.max(np.abs(target_occ)))
    return BoundedOddPolynomial(
        chebyshev_coeffs=np.asarray(bounded_coeffs, dtype=np.float64),
        C_global=float(c_global),
        occupied_max_error=float("nan"),  # filled by the caller via phase reconstruction
        occupied_max_target=occ_max_target,
        global_max_abs=global_max,
        method=str(method),
        degree=int(degree),
        s_min=float(s_min),
        lam=float(lam),
    )


def _circuit_action_value(x: float, phases: np.ndarray) -> float:
    """Executed symmetric-QSP circuit response via qiskit Statevector (imag <0|U|0>)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector

    signal_gate = UnitaryGate(_signal_operator(float(x)), label="W")
    circuit = QuantumCircuit(1)
    circuit.append(UnitaryGate(_phase_rotation(float(phases[0])), label="R0"), [0])
    for phase in phases[1:]:
        circuit.append(signal_gate, [0])
        circuit.append(UnitaryGate(_phase_rotation(float(phase)), label="R"), [0])
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    return float(np.imag(state[0]))


def validate_circuit_action(
    *,
    poly: BoundedOddPolynomial,
    phases: np.ndarray,
    actual_singular_values: np.ndarray | None = None,
    n_test_points: int = 33,
    bounded_tolerance: float = 1.0e-9,
) -> CircuitActionReport:
    """Validate phases by scalar response + executed circuit action.

    Reports occupied reconstruction error (response*C_global vs filter), error on the
    *actual* singular values (separately, per the preregistered criteria), global
    boundedness, and the circuit-vs-scalar / circuit-vs-target agreement.
    """
    occ = np.linspace(poly.s_min, 1.0, n_test_points)
    target_occ = regularized_filter_on_normalized_domain(
        occ, alpha=poly.lam, block_encoding_normalization=1.0
    )
    scalar_resp = np.array([scalar_qsp_response(x, phases) for x in occ]) * poly.C_global
    occupied_recon_error = float(np.max(np.abs(scalar_resp - target_occ)))

    actual_sv_error = float("nan")
    if actual_singular_values is not None and actual_singular_values.size:
        sv = np.asarray(actual_singular_values, dtype=np.float64)
        sv = sv[(sv > poly.s_min) & (sv <= 1.0)]
        if sv.size:
            sv_target = regularized_filter_on_normalized_domain(
                sv, alpha=poly.lam, block_encoding_normalization=1.0
            )
            sv_resp = np.array([scalar_qsp_response(x, phases) for x in sv]) * poly.C_global
            actual_sv_error = float(np.max(np.abs(sv_resp - sv_target)))

    dense = np.linspace(-1.0, 1.0, max(2049, 8 * poly.degree))
    bounded_resp = np.array([scalar_qsp_response(x, phases) for x in dense])
    global_bounded_max = float(np.max(np.abs(bounded_resp)))

    # Executed circuit action at a subset of occupied points.
    circuit_points = np.linspace(poly.s_min, 1.0, min(9, n_test_points))
    circ_vs_scalar = []
    circ_vs_target = []
    for x in circuit_points:
        cv = _circuit_action_value(x, phases) * poly.C_global
        circ_vs_scalar.append(abs(cv - scalar_qsp_response(x, phases) * poly.C_global))
        circ_vs_target.append(
            abs(
                cv
                - regularized_filter_on_normalized_domain(
                    np.array([x]), alpha=poly.lam, block_encoding_normalization=1.0
                )[0]
            )
        )
    return CircuitActionReport(
        phase_count=int(phases.size),
        occupied_recon_error=occupied_recon_error,
        occupied_actual_sv_error=actual_sv_error,
        global_bounded_max=global_bounded_max,
        circuit_vs_scalar_max_error=float(max(circ_vs_scalar)),
        circuit_vs_target_max_error=float(max(circ_vs_target)),
        n_test_points=int(circuit_points.size),
        evidence_label="EXECUTED_CIRCUIT",
        bounded_passes=bool(global_bounded_max <= 1.0 + bounded_tolerance),
    )


CIRCUIT_ACTION_CLAIM = _CIRCUIT_ACTION_CLAIM
