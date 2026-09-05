from __future__ import annotations

from tests.final_useful_overlap_helpers import rows


def test_degree255_resource_ledger():
    data = rows("degree255_resource_ledger.csv")
    categories = {row["category"] for row in data}
    assert {"EXECUTED", "TRANSPILED", "MODELED", "EXCLUDED"} <= categories
    degree = next(row for row in data if row["item"] == "polynomial_degree")
    assert degree["value"] == "255"
    shots = next(row for row in data if row["item"] == "total_hadamard_shots")
    assert int(shots["value"]) == 5_000_000
