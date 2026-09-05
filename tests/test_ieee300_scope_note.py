"""Gap-resolution: IEEE300 reduced-scope note and QSVT/Ridge equivalence."""

from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.paper.claim_lint import build_claim_lint
from robust_qsvt_se.paper.ieee300_runtime_extension import (
    _qsvt_ridge_equivalent,
    _scope_note_markdown,
)

_ROOT = Path(__file__).resolve().parents[1]


def _alpha_rows() -> list[dict]:
    rows = []
    for subset in ("full_ac_measurement_set", "drop_branch_flow_rows"):
        for alpha in (1e-5, 1e-4):
            rows.append(
                {
                    "measurement_subset": subset,
                    "estimator": "ridge_tikhonov",
                    "alpha": alpha,
                    "seed": 0,
                    "rmse": 0.0005 + alpha,
                    "result_status": "computed",
                }
            )
            rows.append(
                {
                    "measurement_subset": subset,
                    "estimator": "qsvt_target_classical",
                    "alpha": alpha,
                    "seed": 0,
                    "rmse": 0.0005 + alpha,
                    "result_status": "computed",
                }
            )
    return rows


def _ablation_rows() -> list[dict]:
    return [
        {"measurement_subset": "full_ac_measurement_set"},
        {"measurement_subset": "drop_branch_flow_rows"},
    ]


def test_scope_note_documents_reduced_scope() -> None:
    note = _scope_note_markdown(
        "ieee300", _alpha_rows(), _ablation_rows(), [], {"case": "ieee300"}, True
    ).lower()
    assert "reduced scope" in note
    assert "not" in note and "exhaustive" in note
    assert "no manuscript claim depends solely" in note


def test_scope_note_states_qsvt_equals_ridge() -> None:
    note = _scope_note_markdown(
        "ieee300", _alpha_rows(), _ablation_rows(), [], {"case": "ieee300"}, True
    )
    assert "equals Ridge" in note


def test_qsvt_ridge_equivalence_holds() -> None:
    assert _qsvt_ridge_equivalent(_alpha_rows()) is True


def test_scope_note_is_claim_safe(tmp_path: Path) -> None:
    note_dir = tmp_path / "ieee300"
    note_dir.mkdir()
    (note_dir / "ieee300_scope_note.md").write_text(
        _scope_note_markdown(
            "ieee300", _alpha_rows(), _ablation_rows(), [], {"case": "ieee300"}, True
        ),
        "utf-8",
    )
    run = build_claim_lint({"input_root": str(note_dir), "output_dir": str(tmp_path / "lint")})
    assert run["high_risk_count"] == 0


def test_package_scope_note_present_and_runtime_limited_file_exists() -> None:
    ext = _ROOT / "outputs" / "final_manuscript_package" / "ieee300_runtime_extension"
    if not ext.is_dir():
        return
    assert (ext / "ieee300_scope_note.md").is_file()
    assert (ext / "ieee300_runtime_limited_rows.csv").is_file()
    text = (ext / "ieee300_scope_note.md").read_text("utf-8").lower()
    assert "reduced scope" in text
