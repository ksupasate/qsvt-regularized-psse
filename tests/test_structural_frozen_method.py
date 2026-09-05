from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"




def test_primary_and_secondary_method_fields_are_unchanged() -> None:
    frozen = json.loads((OUT / "frozen_method_configuration.json").read_text())
    assert (frozen["primary_selector"], frozen["primary_baseline"]) == (
        "sensitivity_initial_mean",
        "balanced_magnitude",
    )
    assert (frozen["primary_k"], frozen["primary_slot_budget"]) == (16, 3)
    assert (frozen["secondary_selector"], frozen["secondary_k"]) == (
        "sensitivity_refined_mean",
        24,
    )
    assert frozen["immutable_after_benchmark_evaluation_begins"] is True
