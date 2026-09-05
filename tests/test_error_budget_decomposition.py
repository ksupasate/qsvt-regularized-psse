def test_error_budget_keeps_deterministic_and_statistical_sources_separate():
    rows = [
        {"source": "polynomial", "class": "deterministic", "value": 1e-3},
        {"source": "finite_shot", "class": "statistical", "value": 2e-3},
    ]
    assert {row["class"] for row in rows} == {"deterministic", "statistical"}
    assert sum(row["value"] for row in rows) == 3e-3
