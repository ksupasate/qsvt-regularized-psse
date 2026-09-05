from __future__ import annotations

from tests.final_useful_overlap_helpers import rows


def test_rectangular_convention_degree_generalization():
    data = rows("degree_generalization_validation.csv")
    assert {int(row["degree"]) for row in data} == {1, 3, 7, 15, 31, 63, 127, 255}
    assert all(row["status"] == "pass" for row in data)
    for row in data:
        degree = int(row["degree"])
        expected = "neg_imag" if ((degree + 1) // 2) % 2 else "imag"
        assert row["component"] == expected
        assert row["expected_component_from_degree"] == expected
