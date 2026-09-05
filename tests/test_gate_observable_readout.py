from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.gate_observable_readout import (
    _map_protocols,
    _topk_jaccard,
)
from robust_qsvt_se.qsvt.power_observable_protocols import build_observable_protocols


def test_gate_observable_readout_does_not_emphasize_full_vector() -> None:
    subproblem = extract_state_estimation_subproblem(
        case="ieee14", model="ac_linearized", submatrix_size=4, seed=123
    )
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    ridge = ridge_tikhonov_update(H, subproblem.r_tilde, alpha=1.0e-4)
    protocols = build_observable_protocols(
        H_tilde=H, ridge_update=ridge, metadata=subproblem.metadata, topk=2
    )
    mapping = _map_protocols(protocols)

    # Full-vector reconstruction is never mapped into the emphasized observable vocabulary.
    assert "full_state_vector_reconstruction" not in mapping
    for protocol in mapping.values():
        assert protocol.requires_full_vector_readout is False
    # Top-k identification is one of the emphasized observables and needs no norm recovery.
    assert "top_k_update_identification" in mapping
    assert mapping["top_k_update_identification"].requires_norm_recovery is False


def test_topk_jaccard_is_deterministic_and_exact_for_identical_vectors() -> None:
    a = np.array([0.4, -0.3, 0.05, 0.6])
    b = 2.5 * a  # same ranking, different scale
    assert _topk_jaccard(a, b, 2) == 1.0
    # Deterministic across repeated calls.
    assert _topk_jaccard(a, a, 2) == _topk_jaccard(a, a, 2) == 1.0
    # A disjoint top set yields a strictly smaller match.
    c = np.array([0.6, 0.05, -0.3, 0.4])
    assert 0.0 <= _topk_jaccard(a, c, 1) <= 1.0
