from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.output_aware_generalization import (
    deterministic_instance_id,
    extract_coverage_preserving_block,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_instance_ids_and_coverage_extraction_are_deterministic() -> None:
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [1.0, 0.0, 1.0, 4.0],
            [2.0, 1.0, 0.0, 1.0],
        ]
    )
    first = extract_coverage_preserving_block(matrix, row_count=4, column_count=4)
    second = extract_coverage_preserving_block(matrix, row_count=4, column_count=4)
    assert deterministic_instance_id("ieee-30", 30101) == (
        "ieee30_eval_seed_30101_block_8x8"
    )
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
    assert np.all(np.count_nonzero(first[0], axis=0) > 0)
    assert np.all(np.count_nonzero(first[0], axis=1) > 0)


def test_primary_registry_excludes_development_block_and_represents_cases() -> None:
    registry = pd.read_csv(OUT / "instance_registry.csv")
    config = json.loads((ROOT / "configs/output_aware_generalization.json").read_text())
    assert len(registry) == 15
    assert set(registry["ieee_case"]) == {"ieee14", "ieee30", "ieee57"}
    assert config["development_matrix_fingerprint"] not in set(
        registry["matrix_fingerprint"]
    )
    assert (~registry["selector_outcomes_used_for_inclusion"].astype(bool)).all()
    assert (registry.groupby("ieee_case").size() == 5).all()





