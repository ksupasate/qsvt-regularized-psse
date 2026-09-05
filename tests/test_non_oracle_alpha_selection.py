import inspect

import numpy as np

from robust_qsvt_se.paper.tqe_revision_core import NON_ORACLE_SELECTORS


def _problem():
    rng = np.random.default_rng(7)
    H = rng.normal(size=(18, 6))
    r = rng.normal(size=18)
    alphas = np.logspace(-5, 1, 17)
    return H, r, alphas


def test_non_oracle_selectors_are_deterministic_and_truth_free():
    H, r, alphas = _problem()
    for selector in NON_ORACLE_SELECTORS.values():
        assert "x_true" not in inspect.signature(selector).parameters
        assert selector(H, r, alphas) == selector(H, r, alphas)


def test_non_oracle_selectors_return_grid_members():
    H, r, alphas = _problem()
    for selector in NON_ORACLE_SELECTORS.values():
        assert selector(H, r, alphas) in alphas
