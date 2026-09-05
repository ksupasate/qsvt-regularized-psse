from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from robust_qsvt_se.qsvt.output_aware_generalization import (
    configuration_fingerprint,
    load_generalization_configuration,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"




def test_frozen_configuration_fingerprint_is_immutable() -> None:
    frozen = json.loads((OUT / "frozen_selector_configuration.json").read_text())
    assert frozen["configuration_fingerprint"] == configuration_fingerprint(frozen)
    changed = deepcopy(frozen)
    changed["normalized_error_floor"] *= 2.0
    assert configuration_fingerprint(changed) != frozen["configuration_fingerprint"]


def test_no_case_specific_retuning_fields_exist() -> None:
    config = load_generalization_configuration(
        ROOT / "configs/output_aware_generalization.json"
    )
    serialized = json.dumps(
        {
            key: config[key]
            for key in (
                "score_normalization_epsilon",
                "normalized_error_floor",
                "support_budgets",
                "slot_budgets",
                "milp_solver_options",
                "refinement",
            )
        },
        sort_keys=True,
    )
    assert "ieee14" not in serialized
    assert "ieee30" not in serialized
    assert "ieee57" not in serialized

