from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_qsvt_se.paper.claim_lint import scan_file
from robust_qsvt_se.paper.full_vector_readout import (
    _sign_measurement_protocol_markdown,
    _sign_projector_rows,
    qsvt_target_readout,
    sign_projector_probabilities,
)

_CONTEXT = {
    "case": "toy",
    "subproblem_id": "toy_00",
    "subproblem_type": "high_leverage",
    "alpha": 1.0e-4,
    "degree": 15,
}


def _state():
    H = np.array(
        [
            [1.2, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.2, 0.1],
            [0.0, 0.3, 1.1, 0.2],
            [0.1, 0.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    r = np.array([0.5, -0.3, 0.4, -0.2], dtype=np.float64)
    return qsvt_target_readout(H, r, alpha=1.0e-4, degree=15)


def test_projector_matches_amplitude_formula_on_synthetic() -> None:
    psi = np.array([0.6, -0.5, 0.4, -0.45])
    psi = psi / np.linalg.norm(psi)
    for index in range(psi.size):
        out = sign_projector_probabilities(psi, reference_index=1, index=index)
        assert abs(out["cross_from_projector"] - out["cross_from_amplitudes"]) < 1.0e-12
        assert abs(out["p_plus"] - 0.5 * (psi[index] + psi[1]) ** 2) < 1.0e-12


def test_projector_error_at_numerical_floor_for_readout_state() -> None:
    state = _state()
    rows = _sign_projector_rows(state, 0, ["a", "b", "c", "d"], _CONTEXT)
    assert rows
    assert max(row["projector_error"] for row in rows) < 1.0e-12


def test_small_margin_signs_flagged() -> None:
    # A coordinate near zero has a tiny interference cross term and must be flagged.
    psi = np.array([0.9, 1.0e-6, 0.3, 0.3])
    psi = psi / np.linalg.norm(psi)
    state = _state()
    object.__setattr__(state, "readout_state", psi)
    rows = _sign_projector_rows(state, 0, ["a", "b", "c", "d"], _CONTEXT)
    assert rows[1]["status"] == "small_margin_sign_unreliable"


def test_protocol_note_makes_no_hardware_claim(tmp_path: Path) -> None:
    note = tmp_path / "sign_measurement_protocol.md"
    note.write_text(_sign_measurement_protocol_markdown(), encoding="utf-8")
    text = note.read_text("utf-8").lower()
    assert "no hadamard-test hardware circuit is executed" in text
    assert not [r for r in scan_file(note) if r["risk_level"] == "high"]
