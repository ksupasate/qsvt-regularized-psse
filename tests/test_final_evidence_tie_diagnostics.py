import json
from pathlib import Path

from robust_qsvt_se.evidence.tie_diagnostics import (
    TieDiagnosticInputs,
    classify_primary_tie,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"


def _input(**overrides: bool | str) -> TieDiagnosticInputs:
    values: dict[str, bool | str] = {
        "original_outcome": "tie",
        "full_support_saturation": False,
        "support_fingerprints_equal": False,
        "effective_matrices_equal": False,
        "near_zero_reference": False,
        "functional_insensitivity": False,
        "numerically_negligible_difference": False,
        "frozen_tie_rule_holds": True,
    }
    values.update(overrides)
    return TieDiagnosticInputs(**values)  # type: ignore[arg-type]


def test_tie_classification_precedence_is_deterministic() -> None:
    assert (
        classify_primary_tie(_input(full_support_saturation=True, support_fingerprints_equal=True))
        == "full_support_saturation"
    )
    assert (
        classify_primary_tie(_input(support_fingerprints_equal=True, near_zero_reference=True))
        == "identical_selected_support"
    )
    assert classify_primary_tie(_input(near_zero_reference=True)) == "near_zero_reference_output"
    assert classify_primary_tie(_input()) == "genuine_selector_tie"
    assert classify_primary_tie(_input(original_outcome="win")) == "not_a_tie"


def test_frozen_primary_result_and_all_tie_causes_are_retained() -> None:
    summary = json.loads((OUT / "primary_tie_diagnostic_summary.json").read_text())
    assert summary["original_primary_win_tie_loss"] == {"win": 6, "tie": 5, "loss": 1}
    assert summary["saturated_ties"] == 2
    assert summary["identical_support_ties"] == 1
    assert summary["genuine_selector_ties"] == 1
    assert summary["diagnostic_category_counts_by_group"]["functional_insensitivity"] == 1
    assert summary["diagnostic_category_counts_by_group"]["mixed_or_ambiguous_tie"] == 0
