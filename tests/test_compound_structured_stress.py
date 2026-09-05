from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.compound_structured_stress import (
    ALL_COLUMNS,
    build_compound_structured_stress,
)


def _run(tmp_path: Path) -> dict:
    return build_compound_structured_stress(
        {
            "cases": ["ieee14"],
            "stress_types": [
                "missing_only",
                "bad_data_only",
                "noise_plus_missing",
                "weak_area_only",
                "contiguous_area_drop",
            ],
            "estimators": ["ridge_tikhonov", "huber_irls", "qsvt_target_classical"],
            "seeds": [0, 1],
            "input_root": str(tmp_path / "outputs"),
            "output_dir": str(tmp_path / "compound"),
        }
    )


def test_compound_type_present_and_schema(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["compound_stress_all_results"])
    assert list(frame.columns) == ALL_COLUMNS
    compound = frame[frame["stress_type"] == "noise_plus_missing"]
    assert not compound.empty
    assert (compound["stress_subtype"] == "compound_random").all()
    # Compound stress actually activates two stress axes.
    assert (compound["noise_scale"] > 0).all()
    assert (compound["missing_ratio"] > 0).all()


def test_random_missing_not_labelled_structured(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["compound_stress_all_results"])
    missing_only = frame[frame["stress_type"] == "missing_only"]
    assert not missing_only.empty
    # Random missing must stay labelled random, never structured / spatial.
    assert (missing_only["stress_subtype"] == "random").all()
    assert not missing_only["stress_subtype"].astype(str).str.contains("structured").any()
    assert not missing_only["stress_subtype"].astype(str).str.contains("spatial").any()


def test_weak_area_is_controlled_assumption(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["compound_stress_all_results"])
    weak = frame[frame["stress_type"].isin(["weak_area_only", "contiguous_area_drop"])]
    assert not weak.empty
    assert (weak["result_status"] == "controlled_assumption").all()
    assert (weak["stress_subtype"] == "controlled_topology_assumption").all()
    # Huber may outperform Ridge under outlier-heavy stress (robust estimator allowed to win).
    robustness = pd.read_csv(run["artifacts"]["estimator_robustness_by_stress"])
    bad = robustness[
        (robustness["stress_type"] == "bad_data_only") & (robustness["estimator"] == "huber_irls")
    ]
    assert not bad.empty
    assert bool(bad.iloc[0]["outperforms_ridge"]) is True
