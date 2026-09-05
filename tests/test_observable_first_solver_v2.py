from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.observable_first_solver_v2 import (
    SUMMARY_COLUMNS,
    evaluate_observable_first_v2,
)


def _subproblem() -> SelectedSubproblem:
    rng = np.random.default_rng(2)
    u, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    singular_values = np.array([4.0, 3.2, 1.5, 1.0])
    H = u @ np.diag(singular_values) @ v.T
    r = H @ np.array([0.5, -0.4, 0.3, 0.2])
    metadata = {
        "state_index_mapping": [
            {"local_index": 0, "state_type": "angle", "bus_id": 4, "label": "theta_4"},
            {"local_index": 1, "state_type": "angle", "bus_id": 5, "label": "theta_5"},
            {"local_index": 2, "state_type": "voltage", "bus_id": 4, "label": "v_4"},
        ]
    }
    return SelectedSubproblem(H_tilde=H, r_tilde=r, metadata=metadata)


def _rows() -> list[dict]:
    return evaluate_observable_first_v2(
        subproblem=_subproblem(),
        configs=[{"alpha": 1.0e-3, "degree": 35, "target_family": "weighted_support_ls"}],
        topk=2,
        subproblem_id="toy_4x4",
    )


def test_observable_first_v2_does_not_emphasize_full_vector() -> None:
    rows = _rows()
    assert rows
    full_vector = [row for row in rows if bool(row["requires_full_vector_readout"])]
    # The only full-vector row is the not-recommended pseudo-observable.
    for row in full_vector:
        assert row["practical_status"] == "full_vector_like_not_recommended"

    practical = [
        row for row in rows if row["practical_status"] != "full_vector_like_not_recommended"
    ]
    assert len(practical) >= len(full_vector)


def test_topk_identification_is_practical_without_norm_recovery() -> None:
    rows = _rows()
    topk = [row for row in rows if "topk" in str(row["observable_name"])]
    assert topk
    row = topk[0]
    assert row["practical_status"] == "practical_without_norm_recovery"
    assert bool(row["requires_norm_recovery"]) is False
    assert 0.0 <= float(row["top_k_match_if_applicable"]) <= 1.0


def test_observable_rows_have_required_columns() -> None:
    rows = _rows()
    for row in rows:
        for column in SUMMARY_COLUMNS:
            assert column in row
        assert row["claim_disallowed"]
