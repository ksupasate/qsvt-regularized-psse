"""Algebraic identity behind the PyQSP -> PCPhase convention proposition.

The derivation in docs/RECTANGULAR_QSVT_CONVENTION_DERIVATION.md rests on the
operator identity (for ANY real phase vector, not only synthesized phases):

    R(phi_0 + pi/2) W R(phi_1 + pi/2) W ... R(phi_d + pi/2)
        = i^{d+1} e^{-i pi/4 Z} [ R(phi_0) Q R(phi_1) Q ... R(phi_d) ] e^{+i pi/4 Z} Z,

with W = [[x, s], [s, -x]] (reflection signal), Q = [[x, i s], [i s, x]]
(plus-i rotation signal), R(phi) = diag(e^{i phi}, e^{-i phi}), s = sqrt(1-x^2).
Because e^{+-i pi/4 Z} acts on |0> as a pure phase and Z|0> = |0>, the top-left
entries satisfy  <0|Prod_shift|0> = i^{d+1} <0|Prod_pyqsp|0>, which for odd d is
the signed-imaginary extraction rule sign(d) = (-1)^{(d+1)/2}.

These tests verify the identity numerically for arbitrary phases and both
degree-parity classes, plus phase 2*pi-periodicity and residual-state
global-phase equivariance of the production sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.rectangular_convention import (
    apply_pcphase_qsvt_sequence,
    pcphase_qsvt_operator,
    pyqsp_pcphase_imag_sign,
    scalar_julia_signal,
)


def rotation_phase(phi: float) -> np.ndarray:
    return np.diag([np.exp(1j * phi), np.exp(-1j * phi)])


def plus_i_signal(x: float) -> np.ndarray:
    s = np.sqrt(max(0.0, 1.0 - x * x))
    return np.array([[x, 1j * s], [1j * s, x]], dtype=np.complex128)


def pyqsp_product(x: float, phases: np.ndarray) -> np.ndarray:
    op = rotation_phase(phases[0])
    for phi in phases[1:]:
        op = rotation_phase(phi) @ plus_i_signal(x) @ op
    return op


@pytest.mark.parametrize("degree", [1, 3, 5, 7, 9, 11])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pi_over_2_offset_identity_for_arbitrary_phases(degree: int, seed: int) -> None:
    rng = np.random.default_rng(9_000 + 13 * degree + seed)
    phases = rng.uniform(-np.pi, np.pi, degree + 1)
    z_pauli = np.diag([1.0, -1.0]).astype(np.complex128)
    quarter = np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])
    for x in rng.uniform(-1.0, 1.0, 5):
        shifted = pcphase_qsvt_operator(
            scalar_julia_signal(float(x)), phases + np.pi / 2, encoded_dimension=1
        )
        reference = (
            (1j) ** (degree + 1)
            * quarter
            @ pyqsp_product(float(x), phases)
            @ quarter.conj().T
            @ z_pauli
        )
        assert np.max(np.abs(shifted - reference)) < 1e-12
        # top-left consequence: <0|shifted|0> = i^{d+1} <0|pyqsp|0>
        lhs = shifted[0, 0]
        rhs = (1j) ** (degree + 1) * pyqsp_product(float(x), phases)[0, 0]
        assert abs(lhs - rhs) < 1e-12


@pytest.mark.parametrize("degree", [1, 3, 5, 7])
def test_sign_rule_matches_i_power(degree: int) -> None:
    predicted = pyqsp_pcphase_imag_sign(degree)
    from_identity = int(np.real((1j) ** (degree + 1)))
    assert predicted == from_identity


@pytest.mark.parametrize("degree", [3, 5])
def test_phase_two_pi_periodicity(degree: int) -> None:
    rng = np.random.default_rng(31 + degree)
    phases = rng.uniform(-np.pi, np.pi, degree + 1)
    x = 0.37
    base = pcphase_qsvt_operator(scalar_julia_signal(x), phases, encoded_dimension=1)
    wrapped = pcphase_qsvt_operator(
        scalar_julia_signal(x), phases + 2 * np.pi, encoded_dimension=1
    )
    assert np.max(np.abs(base - wrapped)) < 1e-12


def test_residual_state_global_phase_equivariance() -> None:
    rng = np.random.default_rng(77)
    dim = 8
    matrix = rng.normal(size=(dim // 2, dim // 2))
    matrix /= 2.0 * np.linalg.norm(matrix, 2)
    identity = np.eye(dim // 2)

    def psd_sqrt(mat: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(0.5 * (mat + mat.T))
        return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T

    dilation = np.block(
        [
            [matrix, psd_sqrt(identity - matrix @ matrix.T)],
            [psd_sqrt(identity - matrix.T @ matrix), -matrix.T],
        ]
    )
    phases = rng.uniform(-np.pi, np.pi, 6)
    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    state /= np.linalg.norm(state)
    theta = 1.234
    out = apply_pcphase_qsvt_sequence(
        dilation, phases, encoded_dimension=dim // 2, vector=state
    )
    out_phased = apply_pcphase_qsvt_sequence(
        dilation, phases, encoded_dimension=dim // 2, vector=np.exp(1j * theta) * state
    )
    # the sequence is linear: a residual-state global phase passes through and
    # cancels in every measured probability or |estimate|.
    assert np.max(np.abs(out_phased - np.exp(1j * theta) * out)) < 1e-12
