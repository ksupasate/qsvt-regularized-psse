from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.observable_error_decomposition import (
    decompose_observable_errors,
    dominant_observable_error_source,
)


def test_observable_error_decomposition_identifies_dominant_source() -> None:
    row = {
        "metadata_status": "mapped_to_state_metadata",
        "raw_absolute_error": 1.0,
        "best_scalar_absolute_error": 1.0,
        "shot_absolute_error": 0.1,
        "ridge_value": 1.0,
    }

    assert (
        dominant_observable_error_source(row)
        == "update_state_approximation_or_signed_norm_recovery"
    )


def test_probability_shot_standard_error_decreases() -> None:
    metadata = {"state_index_mapping": []}
    rows = decompose_observable_errors(
        H_tilde=np.eye(3),
        r_tilde=np.array([1.0, 0.0, 0.0]),
        qsvt_update=np.array([0.6, 0.0, 0.8]),
        ridge_update=np.array([0.6, 0.0, 0.8]),
        metadata=metadata,
        shot_counts=[100, 10000],
        seed=123,
    )
    subset_rows = [row for row in rows if row["observable_type"] == "selected_subset_update_energy"]
    se_by_shot = {row["shot_count"]: row["shot_standard_error"] for row in subset_rows}

    assert se_by_shot[10000] < se_by_shot[100]
