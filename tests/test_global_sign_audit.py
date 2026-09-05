from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_qsvt_se.paper.claim_lint import scan_file
from robust_qsvt_se.paper.full_vector_readout import (
    _global_sign_aligned,
    _global_sign_audit_row,
    _global_sign_scope_markdown,
    _protocol_global_sign,
    qsvt_target_readout,
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


def test_protocol_global_sign_fixes_reference_non_negative() -> None:
    vector = np.array([-0.2, 0.9, -0.3, 0.1])  # reference (index 1, largest) already positive
    out = _protocol_global_sign(vector, reference_index=1)
    assert out[1] >= 0.0 and np.allclose(out, vector)
    flipped = np.array([0.2, -0.9, 0.3, -0.1])
    out2 = _protocol_global_sign(flipped, reference_index=1)
    assert out2[1] >= 0.0 and np.allclose(out2, -flipped)


def test_evaluation_alignment_matches_reference_but_is_separate() -> None:
    reference = np.array([1.0, -0.5, 0.3, -0.2])
    reconstruction = -reference  # opposite global sign
    aligned = _global_sign_aligned(reconstruction, reference)
    assert np.allclose(aligned, reference)


def test_audit_reports_no_ridge_leakage() -> None:
    state = _state()
    row = _global_sign_audit_row(state, 0, dict(_CONTEXT))
    assert row["uses_ridge_for_reconstruction"] == "no"
    assert row["uses_ridge_for_evaluation_only"] == "yes"
    assert row["reference_selection_rule"] == "largest_magnitude_coordinate"
    assert "convention" in row["global_sign_resolved_for_protocol"]


def test_scope_note_not_flagged_by_claim_lint(tmp_path: Path) -> None:
    note = tmp_path / "global_sign_scope_note.md"
    note.write_text(_global_sign_scope_markdown(), encoding="utf-8")
    rows = scan_file(note)
    assert not [r for r in rows if r["risk_level"] == "high"]
