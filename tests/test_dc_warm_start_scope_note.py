"""Gap-resolution: DC warm-start scope note and scope table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.claim_lint import build_claim_lint
from robust_qsvt_se.paper.initializer_ablation import build_dc_warm_start_scope


def _scope_table(tmp_path: Path) -> pd.DataFrame:
    run = build_dc_warm_start_scope({"output_dir": str(tmp_path / "init")})
    return pd.read_csv(run["artifacts"]["initializer_scope_table"])


def test_dc_warm_start_does_not_use_true_state(tmp_path: Path) -> None:
    table = _scope_table(tmp_path).set_index("initializer")
    assert table.loc["dc_warm_start", "uses_true_state"] == "no"


def test_dc_warm_start_is_not_operational_ems(tmp_path: Path) -> None:
    table = _scope_table(tmp_path).set_index("initializer")
    assert table.loc["dc_warm_start", "is_operational_ems_initializer"] == "no"
    assert table.loc["dc_warm_start", "uses_dc_linear_proxy"] == "yes"
    assert table.loc["dc_warm_start", "uses_measurements"] == "yes"


def test_scope_note_has_required_wording(tmp_path: Path) -> None:
    run = build_dc_warm_start_scope({"output_dir": str(tmp_path / "init")})
    note = Path(run["artifacts"]["dc_warm_start_scope_note"]).read_text("utf-8")
    assert "controlled DC linear proxy" in note
    assert "not a utility EMS initializer" in note
    assert "does not establish operational deployment realism" in note


def test_scope_note_is_claim_safe(tmp_path: Path) -> None:
    build_dc_warm_start_scope({"output_dir": str(tmp_path / "init")})
    run = build_claim_lint(
        {"input_root": str(tmp_path / "init"), "output_dir": str(tmp_path / "l")}
    )
    assert run["high_risk_count"] == 0


def test_no_initializer_is_operational_ems(tmp_path: Path) -> None:
    table = _scope_table(tmp_path)
    assert (table["is_operational_ems_initializer"] == "no").all()
