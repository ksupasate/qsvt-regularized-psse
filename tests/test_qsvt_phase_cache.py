from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.phase_synthesis import synthesize_pennylane_phases_cached


def test_pennylane_phase_cache_returns_finite_non_dummy_phases(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    coefficients = np.array([0.0, 0.5, 0.0, -0.1], dtype=float)

    first = synthesize_pennylane_phases_cached(
        coefficients,
        angle_solver="iterative",
        cache_dir=tmp_path,
        cache_metadata={"test": "phase-cache"},
    )
    second = synthesize_pennylane_phases_cached(
        coefficients,
        angle_solver="iterative",
        cache_dir=tmp_path,
        cache_metadata={"test": "phase-cache"},
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert np.all(np.isfinite(first.phases))
    assert not np.allclose(first.phases, 0.0)
    assert np.allclose(first.phases, second.phases)
