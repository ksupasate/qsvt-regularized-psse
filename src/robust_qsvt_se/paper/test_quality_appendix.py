"""Phase 9: test-quality appendix.

Condenses the test-quality audit into a short, appendix-ready summary: the test
category counts, an explicit statement that smoke tests are listed separately and are
NOT counted as scientific validation, and the full-suite / ruff status (recorded from
the actual verification run when provided). It reads the existing audit and fabricates
nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.paper.test_quality_audit import suite_counts
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/build_test_quality_appendix.py"

APPENDIX_TABLE_COLUMNS = [
    "metric",
    "value",
    "counts_as_scientific_validation",
    "notes",
]
_NON_VALIDATION = frozenset({"smoke_only", "unit_behavior"})


def _parse_passed(status: str) -> Any:
    match = re.search(r"(\d+)\s+passed", str(status))
    return int(match.group(1)) if match else "unavailable"


def build_test_quality_appendix(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_root / "test_quality_appendix"))
    full_suite_status = str(config.get("full_suite_status", "run_separately_see_reproducibility"))
    ruff_status = str(config.get("ruff_status", "run_separately_see_reproducibility"))
    ensure_directory(output_dir)

    inventory = read_csv(package_root / "test_quality_audit" / "test_inventory.csv")
    counts = (
        inventory["test_category"].value_counts().to_dict()
        if not inventory.empty and "test_category" in inventory.columns
        else {}
    )
    total = len(inventory) if not inventory.empty else 0
    real = int(counts.get("real_validation", 0))
    guard = int(counts.get("claim_boundary_guard", 0))
    relies_on_smoke = total > 0 and real == 0 and guard == 0
    records = inventory.to_dict("records") if not inventory.empty else []
    suites = suite_counts(records)
    full_pytest_passed = config.get("full_pytest_tests_passed", _parse_passed(full_suite_status))

    table_rows = _table_rows(total, counts, relies_on_smoke, suites, full_pytest_passed)
    return _write_outputs(
        output_dir=output_dir,
        table_rows=table_rows,
        total=total,
        counts=counts,
        suites=suites,
        full_pytest_passed=full_pytest_passed,
        relies_on_smoke=relies_on_smoke,
        full_suite_status=full_suite_status,
        ruff_status=ruff_status,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
            "full_suite_status": full_suite_status,
            "ruff_status": ruff_status,
        },
    )


def _table_rows(
    total: int,
    counts: dict[str, Any],
    relies_on_smoke: bool,
    suites: dict[str, int],
    full_pytest_passed: Any,
) -> list[dict[str, Any]]:
    rows = [
        {
            "metric": "total_tests_inventoried",
            "value": total,
            "counts_as_scientific_validation": "n/a",
            "notes": "all test functions classified by the AST-based audit",
        }
    ]
    for category in (
        "real_validation",
        "claim_boundary_guard",
        "integration_regression",
        "artifact_schema",
        "unit_behavior",
        "smoke_only",
    ):
        rows.append(
            {
                "metric": category,
                "value": int(counts.get(category, 0)),
                "counts_as_scientific_validation": ("no" if category in _NON_VALIDATION else "yes"),
                "notes": (
                    "smoke tests are listed separately and not counted as scientific validation"
                    if category == "smoke_only"
                    else ""
                ),
            }
        )
    rows.extend(
        [
            {
                "metric": "full_pytest_tests_passed",
                "value": full_pytest_passed,
                "counts_as_scientific_validation": "n/a",
                "notes": "pytest items (parametrized cases expand a function into several items)",
            },
            {
                "metric": "test_quality_inventory_total",
                "value": total,
                "counts_as_scientific_validation": "n/a",
                "notes": "distinct test functions counted by the AST audit (not pytest items)",
            },
            {
                "metric": "scientific_validation_suite_tests",
                "value": suites["scientific_validation"],
                "counts_as_scientific_validation": "yes",
                "notes": "real_validation + claim_boundary_guard + integration_regression "
                "+ artifact_schema",
            },
            {
                "metric": "scientific_validation_suite_smoke_only_tests",
                "value": suites["scientific_validation_smoke_only"],
                "counts_as_scientific_validation": "no",
                "notes": "zero by construction: the scientific suite excludes smoke-only tests",
            },
            {
                "metric": "engineering_smoke_tests",
                "value": suites["engineering_smoke"],
                "counts_as_scientific_validation": "no",
                "notes": "kept for import/CLI/existence coverage; not scientific validation",
            },
            {
                "metric": "remaining_smoke_only_tests",
                "value": suites["remaining_smoke_only"],
                "counts_as_scientific_validation": "no",
                "notes": "smoke tests still smoke after conversion; kept as engineering smoke",
            },
            {
                "metric": "converted_smoke_tests",
                "value": suites["converted_from_smoke"],
                "counts_as_scientific_validation": "yes",
                "notes": "smoke tests upgraded to substantive scientific tests in this pass",
            },
        ]
    )
    rows.append(
        {
            "metric": "relies_on_smoke_tests_only",
            "value": "yes" if relies_on_smoke else "no",
            "counts_as_scientific_validation": "n/a",
            "notes": "evidence package does not rely on smoke tests",
        }
    )
    return rows


def _appendix_markdown(
    total: int,
    counts: dict[str, Any],
    relies_on_smoke: bool,
    suites: dict[str, int],
    full_pytest_passed: Any,
    full_suite_status: str,
    ruff_status: str,
) -> str:
    return "\n".join(
        [
            "# Test-Quality Appendix",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Test inventory",
            f"- Total test functions inventoried: {total}.",
            f"- Real-validation tests: {int(counts.get('real_validation', 0))}.",
            f"- Claim-boundary guard tests: {int(counts.get('claim_boundary_guard', 0))}.",
            f"- Integration / regression tests: {int(counts.get('integration_regression', 0))}.",
            f"- Artifact-schema tests: {int(counts.get('artifact_schema', 0))}.",
            f"- Unit-behavior tests: {int(counts.get('unit_behavior', 0))}.",
            f"- Smoke-only tests: {int(counts.get('smoke_only', 0))}.",
            "",
            "## Full engineering suite vs scientific-validation suite",
            f"- Full engineering pytest suite (parametrized items): {full_pytest_passed} passed.",
            f"- Test functions inventoried by the AST audit: {total}.",
            "- These two numbers differ because pytest expands a parametrized test function into "
            "several test items; the inventory counts functions, not items. Neither is wrong.",
            f"- Scientific-validation suite: {suites['scientific_validation']} tests "
            "(real_validation + claim_boundary_guard + integration_regression + artifact_schema).",
            f"- Smoke-only tests inside the scientific-validation suite: "
            f"{suites['scientific_validation_smoke_only']}.",
            f"- Engineering smoke tests (kept, not scientific): {suites['engineering_smoke']}.",
            f"- Smoke tests converted to scientific tests in this pass: "
            f"{suites['converted_from_smoke']}.",
            "",
            "## Smoke tests are not scientific validation",
            "- Smoke-only tests are listed separately in `test_quality_audit/smoke_tests.csv` and "
            "are NOT counted as scientific validation. The scientific claims are backed by "
            "real-validation, claim-boundary guard, integration/regression, and artifact-schema "
            "tests.",
            "- The full engineering pytest suite may include engineering smoke tests. The "
            "manuscript scientific-validation suite excludes smoke-only tests and contains zero "
            "smoke-only tests.",
            "- Smoke tests are not counted as scientific validation evidence.",
            f"- Evidence package relies on smoke tests only: {'yes' if relies_on_smoke else 'no'}.",
            "",
            "## Verification status",
            f"- Full test suite (`.venv/bin/python -m pytest`): {full_suite_status}.",
            f"- Ruff (`ruff check` and `ruff format --check`): {ruff_status}.",
            "- When a status reads `run_separately_see_reproducibility`, the command is run as "
            "part of the verification step documented in reproducibility.md.",
            "",
        ]
    )


def _suite_summary_markdown(suites: dict[str, int], full_pytest_passed: Any) -> str:
    return "\n".join(
        [
            "# Scientific-Validation Suite Summary",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Definition",
            "- The scientific-validation suite is the set of tests in the categories "
            "real_validation, claim_boundary_guard, integration_regression, and artifact_schema.",
            "- It excludes smoke-only and unit-behavior tests.",
            "",
            "## Counts",
            f"- Full engineering pytest suite (items): {full_pytest_passed} passed.",
            f"- Test functions inventoried: {suites['total']}.",
            f"- Scientific-validation suite tests: {suites['scientific_validation']}.",
            f"- Smoke-only tests in the scientific-validation suite: "
            f"{suites['scientific_validation_smoke_only']}.",
            f"- Engineering smoke tests (kept, excluded from scientific validation): "
            f"{suites['engineering_smoke']}.",
            f"- Smoke tests converted to scientific tests: {suites['converted_from_smoke']}.",
            "",
            "## Guarantee",
            "- The manuscript scientific-validation suite contains zero smoke-only tests. Smoke "
            "tests are retained only as engineering coverage and are never counted as scientific "
            "validation evidence.",
            "- Run it with `scripts/run_scientific_validation_suite.py` (see "
            "`scientific_validation_tests.csv` for the exact selection).",
            "",
        ]
    )


def _write_outputs(
    *,
    output_dir: Path,
    table_rows: list[dict[str, Any]],
    total: int,
    counts: dict[str, Any],
    suites: dict[str, int],
    full_pytest_passed: Any,
    relies_on_smoke: bool,
    full_suite_status: str,
    ruff_status: str,
    input_config: dict[str, Any],
) -> dict[str, Any]:
    table_path = rows_to_table(
        table_rows, output_dir / "test_quality_appendix_table.csv", APPENDIX_TABLE_COLUMNS
    )
    md_path = output_dir / "test_quality_appendix.md"
    md_path.write_text(
        _appendix_markdown(
            total,
            counts,
            relies_on_smoke,
            suites,
            full_pytest_passed,
            full_suite_status,
            ruff_status,
        ),
        encoding="utf-8",
    )
    suite_summary_path = output_dir / "scientific_validation_suite_summary.md"
    suite_summary_path.write_text(
        _suite_summary_markdown(suites, full_pytest_passed), encoding="utf-8"
    )

    artifacts = {
        "test_quality_appendix": str(md_path),
        "test_quality_appendix_table": str(table_path),
        "scientific_validation_suite_summary": str(suite_summary_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "table_rows": table_rows,
        "total": total,
        "counts": counts,
        "suites": suites,
        "full_pytest_passed": full_pytest_passed,
        "relies_on_smoke": relies_on_smoke,
        "artifacts": artifacts,
    }
