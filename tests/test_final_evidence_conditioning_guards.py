from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"


def test_conditioning_labels_are_never_conflated() -> None:
    audit = pd.read_csv(OUT / "regularized_conditioning_audit.csv")
    assert set(audit["raw_conditioning_label"]) == {"raw_matrix_conditioning"}
    assert set(audit["regularized_conditioning_label"]) == {
        "regularized_normal_system_conditioning"
    }
    assert set(audit["ridge_filter_label"]) == {"ridge_filter_amplification"}
    rank_deficient = audit[audit["rank_deficient"]]
    assert np_is_infinite(rank_deficient["raw_condition_number"]).all()


def np_is_infinite(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value == float("inf"))


def test_interpretation_hazards_are_resolved_not_hidden() -> None:
    guards = pd.read_csv(OUT / "conditioning_interpretation_violations.csv")
    assert not (guards["status"] == "unresolved").any()
    assert not guards["blocking"].any()
    assert set(guards["status"]) <= {"resolved_in_canonical_export"}
