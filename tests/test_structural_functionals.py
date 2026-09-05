from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_every_realization_has_three_unit_norm_metadata_grounded_functionals() -> None:
    instances = pd.read_csv(OUT / "instance_registry.csv")
    functionals = pd.read_csv(OUT / "functional_registry.csv")
    assert len(functionals) == 3 * len(instances) == 72
    assert (functionals.groupby("instance_id").size() == 3).all()
    assert set(functionals["functional_family"]) == {
        "coordinate",
        "difference",
        "aggregate",
    }
    assert np.allclose(functionals["functional_norm"], 1.0)
    assert functionals["semantic_status"].str.startswith("metadata").all()
    assert functionals["selection_data_used"].eq("state_metadata_only_no_output_accuracy").all()


def test_functional_vectors_match_frozen_local_policy() -> None:
    frame = pd.read_csv(OUT / "functional_registry.csv")
    for row in frame.itertuples(index=False):
        vector = np.asarray(json.loads(row.functional_vector), dtype=float)
        if row.functional_family == "coordinate":
            assert np.array_equal(vector, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif row.functional_family == "difference":
            assert np.allclose(vector[:2], [1 / np.sqrt(2), -1 / np.sqrt(2)])
            assert np.count_nonzero(vector) == 2
        else:
            assert np.array_equal(vector[:4], [0.5, 0.5, 0.5, 0.5])
            assert np.count_nonzero(vector) == 4
