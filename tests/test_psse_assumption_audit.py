from __future__ import annotations

from robust_qsvt_se.paper.psse_assumption_audit import parameter_rows, rank_rows


def test_parameter_audit_covers_required_assumptions() -> None:
    items = {row["item"] for row in parameter_rows()}
    assert {
        "measurement rows",
        "row standard deviations",
        "weak-area multiplier",
        "missing rows",
        "bad data",
        "seeds",
        "rank / pinv rule",
        "perturbation location",
    } <= items


def test_rank_audit_uses_explicit_cutoff() -> None:
    rows = rank_rows(seed=123)
    assert len(rows) == 5
    assert all(row["numerical_rank"] <= row["state_dimension"] for row in rows)
    assert all(row["absolute_singular_value_threshold"] > 0.0 for row in rows)
    assert all(row["pseudoinverse_rcond"] == 1.0e-10 for row in rows)
