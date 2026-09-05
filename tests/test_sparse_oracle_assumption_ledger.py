from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.qsvt.sparse_oracle_assumption_ledger import (
    REQUIRED_ASSUMPTIONS,
    assumption_ledger_rows,
    build_sparse_oracle_assumption_ledger,
)


def test_sparse_oracle_assumption_ledger_includes_required_assumptions() -> None:
    rows = assumption_ledger_rows()
    names = {row["assumption_name"] for row in rows}

    assert set(REQUIRED_ASSUMPTIONS).issubset(names)


def test_sparse_oracle_assumption_ledger_writes_outputs(tmp_path: Path) -> None:
    run = build_sparse_oracle_assumption_ledger(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee14"],
            "degree": 9,
        }
    )

    assert len(run["ledger_rows"]) >= len(REQUIRED_ASSUMPTIONS)
    for name in [
        "manifest",
        "oracle_assumption_ledger",
        "ieee_resource_table",
        "oracle_cost_model",
        "assumption_risk_ranking",
    ]:
        assert run["artifacts"][name].is_file()
