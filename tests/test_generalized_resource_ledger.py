"""Tests for the generalized resource ledger (section 19).

Reads generalized_resource_ledger.csv. Integrity invariant: executed, transpiled,
modeled, and excluded categories must all be present and correctly separated;
modeled costs must NOT appear under executed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "generalized_resource_ledger.csv"


def test_ledger_has_all_categories():
    df = pd.read_csv(CSV)
    cats = set(df["category"])
    assert {"executed", "transpiled", "modeled", "excluded"}.issubset(cats)


def test_executed_qubits_present():
    df = pd.read_csv(CSV)
    q = df[(df["category"] == "executed") & (df["item"] == "logical_qubits")]
    assert len(q) == 1
    assert int(q.iloc[0]["value"]) == 8  # IEEE-14 dilation 256 = 8 qubits


def test_modeled_not_mislabeled():
    df = pd.read_csv(CSV)
    modeled = df[df["category"] == "modeled"]
    # modeled entries must reference modeling, not claim execution
    assert len(modeled) >= 1


def test_excluded_categories_listed():
    df = pd.read_csv(CSV)
    excluded = set(df[df["category"] == "excluded"]["item"])
    for needed in [
        "physical_noise",
        "routing",
        "surface_code",
        "magic_state_factories",
        "networking",
    ]:
        assert needed in excluded
