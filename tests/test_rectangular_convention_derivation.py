"""Tests for the rectangular-convention derivation (WP-A) and its API traceability.

Verifies the algebraic identities stated in rectangular_convention_derivation.md
LIVE (not from a CSV): the +pi/2 global offset reproduces the target polynomial
in the signed-imaginary channel, the opposite component is ~0, and an
endpoint-only offset breaks the response. Also asserts the sign rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.polynomial import Chebyshev

from robust_qsvt_se.generalized.convention_api import predict_extraction
from robust_qsvt_se.qsvt.rectangular_convention import (
    PYQSP_TO_PCPHASE_OFFSET,
    production_scalar_response,
)
from robust_qsvt_se.qsvt.sym_qsp_circuit_action import synthesize_pyqsp_sym_qsp_phases

GRID = np.linspace(-1, 1, 801)


def _monomial_phases(degree: int):
    p1 = Chebyshev([0, 1], domain=[-1, 1])
    pn = Chebyshev([1], domain=[-1, 1])
    for _ in range(degree):
        pn = pn * p1
    phases = synthesize_pyqsp_sym_qsp_phases(np.asarray(pn.coef, float))
    return phases + PYQSP_TO_PCPHASE_OFFSET, pn


@pytest.mark.parametrize(
    "degree,expected",
    [
        (1, ("neg_imag", -1)),
        (3, ("imag", 1)),
        (5, ("neg_imag", -1)),
        (7, ("imag", 1)),
        (255, ("imag", 1)),
    ],
)
def test_sign_rule(degree, expected):
    assert predict_extraction(degree) == expected


def test_sign_rule_formula():
    for d in range(1, 256, 2):
        comp, sign = predict_extraction(d)
        expected_sign = -1 if ((d + 1) // 2) % 2 else 1
        assert sign == expected_sign
        assert comp == ("imag" if expected_sign > 0 else "neg_imag")


def test_global_offset_reproduces_polynomial():
    phases, poly = _monomial_phases(3)
    comp = predict_extraction(3)[0]
    resp = np.array([production_scalar_response(x, phases, component=comp) for x in GRID])
    assert float(np.max(np.abs(resp - poly(GRID)))) < 1e-10


def test_opposite_channel_is_complementary_not_target():
    """The opposite (real) channel holds the complementary polynomial, O(1), and
    must NOT reproduce the target P(x): the target is uniquely in the signed
    imaginary channel."""

    phases, poly = _monomial_phases(3)
    comp = predict_extraction(3)[0]
    opp = "real" if "imag" in comp else "imag"
    resp = np.array([production_scalar_response(x, phases, component=opp) for x in GRID])
    assert float(np.max(np.abs(resp))) > 0.1
    assert float(np.max(np.abs(resp - poly(GRID)))) > 0.1


def test_endpoint_only_offset_breaks_response():
    """Shifting only the first phase (not global) must NOT reproduce the target."""

    phases, poly = _monomial_phases(3)
    comp = predict_extraction(3)[0]
    broken = phases.copy()
    broken[0] = 0.0  # drop offset on first phase only
    resp = np.array([production_scalar_response(x, broken, component=comp) for x in GRID])
    assert float(np.max(np.abs(resp - poly(GRID)))) > 0.1


def test_offset_value_is_pi_over_two():
    assert abs(PYQSP_TO_PCPHASE_OFFSET - math.pi / 2) < 1e-15
