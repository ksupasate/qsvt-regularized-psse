from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.state_preparation_model import (
    EXACT_DENSE_LOADING_LIMITATION,
    QRAM_LOADING_LIMITATION,
    StatePreparationModel,
    estimate_state_preparation,
)


def test_state_preparation_estimate_reports_required_fields() -> None:
    vector = np.array([1.0, 0.0, -2.0])
    estimate = estimate_state_preparation(
        vector,
        StatePreparationModel.EXACT_DENSE_AMPLITUDE_LOADING,
    )
    row = estimate.to_row()

    assert row["dimension"] == 3
    assert row["padded_dimension"] == 4
    assert row["input_norm"] == pytest.approx(np.sqrt(5.0))
    assert row["nonzero_count"] == 2
    assert row["index_qubits"] == 2
    assert EXACT_DENSE_LOADING_LIMITATION in row["limitations"]


def test_qram_state_preparation_is_labeled_as_assumption() -> None:
    estimate = estimate_state_preparation(
        np.array([0.5, -0.5]),
        StatePreparationModel.QRAM_AMPLITUDE_ORACLE,
    )

    assert estimate.estimated_query_cost == 1
    assert QRAM_LOADING_LIMITATION in estimate.limitations
