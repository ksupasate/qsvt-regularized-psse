"""Gap-resolution: the scientific-validation suite is smoke-free."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.test_quality_audit import (
    CONVERTED_SMOKE_TESTS,
    SCIENTIFIC_CATEGORIES,
    build_test_quality_audit,
    scientific_validation_rows,
    smoke_rows,
)

_ROOT = Path(__file__).resolve().parents[1]


def _audit(tmp_path: Path) -> dict:
    return build_test_quality_audit({"test_root": "tests", "output_dir": str(tmp_path / "audit")})


def test_scientific_suite_has_zero_smoke(tmp_path: Path) -> None:
    run = _audit(tmp_path)
    rows = run["inventory_rows"]
    scientific = scientific_validation_rows(rows)
    assert len(scientific) > 0
    assert all(r["test_category"] != "smoke_only" for r in scientific)
    assert run["suite_counts"]["scientific_validation_smoke_only"] == 0


def test_converted_smoke_tests_are_now_scientific(tmp_path: Path) -> None:
    run = _audit(tmp_path)
    by_node = {r["node_id"]: r for r in run["inventory_rows"]}
    assert run["suite_counts"]["converted_from_smoke"] == len(CONVERTED_SMOKE_TESTS)
    for node_id in CONVERTED_SMOKE_TESTS:
        assert node_id in by_node, node_id
        category = by_node[node_id]["test_category"]
        assert category in SCIENTIFIC_CATEGORIES
        assert by_node[node_id]["is_smoke_test"] == "no"


def test_converted_smoke_tests_have_substantive_assertions() -> None:
    # Each converted test now contains a numeric/dimensional/schema assertion (not existence only).
    markers = ("abs(", ".shape ==", "issubset", "== round(")
    for node_id in CONVERTED_SMOKE_TESTS:
        rel, name = node_id.split("::")
        source = (_ROOT / rel).read_text("utf-8")
        start = source.index(f"def {name}(")
        end = source.find("\ndef ", start + 1)
        body = source[start : end if end != -1 else len(source)]
        assert any(marker in body for marker in markers), node_id


def test_engineering_smoke_not_counted_as_scientific(tmp_path: Path) -> None:
    run = _audit(tmp_path)
    smoke = smoke_rows(run["inventory_rows"])
    # Every remaining smoke test is engineering smoke, excluded from the scientific suite.
    assert all(r["suite"] == "engineering_smoke" for r in smoke)
    scientific_nodes = {r["node_id"] for r in scientific_validation_rows(run["inventory_rows"])}
    smoke_nodes = {r["node_id"] for r in smoke}
    assert scientific_nodes.isdisjoint(smoke_nodes)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_scientific_validation_suite", _ROOT / "scripts" / "run_scientific_validation_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_runner_selection_is_smoke_free() -> None:
    runner = _load_runner()
    selection = runner.build_selection("tests")
    assert selection["scientific_count"] > 0
    assert selection["smoke_only_in_selection"] == 0
    assert len(selection["node_ids"]) == selection["scientific_count"]


def test_audit_writes_scientific_and_engineering_csvs(tmp_path: Path) -> None:
    run = _audit(tmp_path)
    scientific = pd.read_csv(run["artifacts"]["scientific_validation_tests"])
    engineering = pd.read_csv(run["artifacts"]["engineering_smoke_tests"])
    assert int((scientific["test_category"] == "smoke_only").sum()) == 0
    if not engineering.empty:
        assert (engineering["test_category"] == "smoke_only").all()
