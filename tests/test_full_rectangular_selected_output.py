import numpy as np

from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import build_padded_dilation
from robust_qsvt_se.paper.tqe_revision_core import normalized_regularization


def test_full_rectangular_weighted_dimensions_and_regularization_conversion():
    rng = np.random.default_rng(12)
    H = rng.normal(size=(11, 5))
    beta = np.linalg.norm(H, 2)
    dilation = build_padded_dilation(H, beta)
    assert dilation["n_states"] == 5
    assert dilation["m_measurements"] == 11
    assert dilation["padded_dimension"] == 16
    assert dilation["unitary_dimension"] == 32
    alpha = 0.068 * beta**2
    assert normalized_regularization(alpha, beta) == np.float64(0.068)
