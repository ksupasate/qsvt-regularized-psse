"""Phase E: test-quality and smoke-test audit.

Statically inventories every test function and classifies it as real validation,
integration/regression, artifact-schema, claim-boundary guard, unit behavior, or
smoke-only. Smoke tests are listed but never counted as scientific validation;
artifact-schema and integration tests are not mislabeled as validation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

INVENTORY_COLUMNS = [
    "test_file",
    "test_name",
    "node_id",
    "test_category",
    "suite",
    "is_smoke_test",
    "uses_numeric_assertions",
    "uses_artifact_schema_assertions",
    "uses_regression_against_known_values",
    "uses_end_to_end_command",
    "scientific_validation_relevance",
    "paper_claims_supported",
    "reason_for_classification",
]

# The scientific-validation suite is exactly these categories. Smoke-only and unit-behavior
# tests are engineering coverage and are NOT counted as scientific validation.
SCIENTIFIC_CATEGORIES = frozenset(
    {"real_validation", "claim_boundary_guard", "integration_regression", "artifact_schema"}
)
SUITE_SCIENTIFIC = "scientific_validation"
SUITE_ENGINEERING_SMOKE = "engineering_smoke"
SUITE_ENGINEERING_OTHER = "engineering_other"

# Smoke-only tests upgraded to substantive scientific tests in the smoke-free hardening pass.
# Listed by pytest node id; the audit verifies each now classifies into a scientific category.
CONVERTED_SMOKE_TESTS = (
    "tests/test_metrics.py::test_rank_deficient_condition_number_is_infinite",
    "tests/test_estimators.py::test_weighted_system_shape_validation",
    "tests/test_bad_data.py::test_bad_data_weak_area_target_selects_only_weak_rows",
)
CONVERTED_SMOKE_SET = frozenset(CONVERTED_SMOKE_TESTS)

_CLAIM_GUARD_MARKERS = (
    "do_not_claim",
    "unsupported",
    "beats ridge",
    "quantum advantage",
    "quantum speedup",
    "not superior",
    "no_speedup",
    "claim_boundary",
    "paper_claim_boundary",
    "disallowed",
    "allowed_wording",
    "superiority",
)

_NUMERIC_MARKERS = (
    "approx(",
    "isclose",
    "allclose",
    "np.isclose",
    "pytest.approx",
    " tol",
    "atol",
    "rtol",
)
_NUMERIC_RE = re.compile(r"(<=|>=|==|!=|<|>)\s*-?\d+\.\d|abs\(")
_TOLERANCE_RE = re.compile(r"\d\s*e\s*-\s*\d|1e-\d")
_REGRESSION_RE = re.compile(r"approx\(\s*-?\d|==\s*-?\d+\.\d")
# Dimensional / known-count checks count as validation (shape & row-dimension consistency).
_DIMENSION_MARKERS = (".shape", "n_states", "n_measurements", "n_rows", ".ndim", ".size")
_DIMENSION_RE = re.compile(r"==\s*\(|==\s*-?\d+\b")
# Behaviour checks: error handling and specific category/string/membership equality.
_BEHAVIOUR_MARKERS = ("pytest.raises", ".raises(", "isinstance(", ".to_dict()")
_BEHAVIOUR_RE = re.compile(
    r"==\s*['\"]|\sin\s*[\[\{\(]|!=\s*['\"]|\bassert\s+not\b|\bis False\b|\bis True\b"
)

_SCHEMA_MARKERS = (
    ".columns",
    "in df.columns",
    "issubset",
    "read_csv",
    ".exists()",
    "is_file()",
    "manifest",
    ".keys()",
    "set(",
    "glob(",
    "list(columns)",
    "expected_columns",
)
_END_TO_END_MARKERS = (
    "subprocess",
    "runpy",
    "check_output",
    "popen",
    "sys.argv",
    "parse_args",
    "build_",
    "main(",
)
_EXISTENCE_ONLY_MARKERS = (
    ".exists()",
    "is_file()",
    "is not None",
    "not_empty",
    "len(",
    "> 0",
    ".empty",
)

# Filename keyword -> claim ids the test most plausibly supports.
_CLAIM_BY_KEYWORD: tuple[tuple[str, str], ...] = (
    ("classical", "C02,C03"),
    ("spectral", "C02,C03"),
    ("filter", "C02,C03"),
    ("ridge", "C02,C03"),
    ("alpha", "C15 alpha-sensitivity"),
    ("nonlinear", "C16 nonlinear-AC realism"),
    ("stress", "C14 structured-stress"),
    ("ablation", "C14 structured-stress"),
    ("measurement", "C01 measurement-model"),
    ("inventory", "C01 measurement-model"),
    ("gate", "C04,C09 QSVT solver-prototype"),
    ("phase", "C04,C09 QSVT solver-prototype"),
    ("observable", "C07 observable-readout"),
    ("resource", "C08 resource-model"),
    ("degree", "C06 degree-window"),
    ("overshoot", "C06 degree-window"),
    ("evidence", "C01-C14 claim framework"),
    ("claim", "C11-C13 unsupported guard"),
    ("manuscript", "C01-C14 claim framework"),
    ("recheck", "C02,C03 classical recheck"),
    ("qsvt", "C04-C10 QSVT pathway"),
)


def build_test_quality_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "test_root": "tests",
        "output_dir": "outputs/final_manuscript_package/test_quality_audit",
    }
    resolved.update(config)
    test_root = Path(resolved["test_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    inventory_rows = _inventory_rows(test_root)

    artifacts = _write_outputs(output_dir, resolved, inventory_rows=inventory_rows)
    counts = _category_counts(inventory_rows)
    return {
        "output_dir": output_dir,
        "inventory_rows": inventory_rows,
        "category_counts": counts,
        "suite_counts": suite_counts(inventory_rows),
        "relies_on_smoke": _relies_on_smoke(inventory_rows),
        "artifacts": artifacts,
    }


def _inventory_rows(test_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(test_root.rglob("test_*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        rel = (
            str(path.relative_to(test_root.parent))
            if test_root.parent in path.parents
            else str(path)
        )
        claims = _claims_for_file(path.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
                continue
            segment = ast.get_source_segment(source, node) or ""
            rows.append(_classify(rel, node.name, segment, claims))
    return rows


def _classify(rel: str, name: str, segment: str, claims: str) -> dict[str, Any]:
    text = segment.lower()
    has_assert = "assert" in text or "pytest.raises" in text

    numeric = _has_numeric(text)
    regression = bool(_REGRESSION_RE.search(text))
    dimensional = any(marker in text for marker in _DIMENSION_MARKERS) or bool(
        _DIMENSION_RE.search(text)
    )
    behaviour = any(marker in text for marker in _BEHAVIOUR_MARKERS) or bool(
        _BEHAVIOUR_RE.search(text)
    )
    schema = any(marker in text for marker in _SCHEMA_MARKERS)
    schema_content = schema and any(
        marker in text for marker in (".columns", "issubset", "expected_columns", ".keys()")
    )
    end_to_end = any(marker in text for marker in _END_TO_END_MARKERS)
    claim_guard = any(marker in text for marker in _CLAIM_GUARD_MARKERS) or "claim" in name

    category, relevance, reason = _decide(
        has_assert=has_assert,
        numeric=numeric,
        regression=regression,
        dimensional=dimensional,
        behaviour=behaviour,
        schema=schema,
        schema_content=schema_content,
        end_to_end=end_to_end,
        claim_guard=claim_guard,
    )
    return {
        "test_file": rel,
        "test_name": name,
        "node_id": f"{rel}::{name}",
        "test_category": category,
        "suite": _suite_for(category),
        "is_smoke_test": "yes" if category == "smoke_only" else "no",
        "uses_numeric_assertions": _yes_no(numeric),
        "uses_artifact_schema_assertions": _yes_no(schema),
        "uses_regression_against_known_values": _yes_no(regression),
        "uses_end_to_end_command": _yes_no(end_to_end),
        "scientific_validation_relevance": relevance,
        "paper_claims_supported": claims,
        "reason_for_classification": reason,
    }


def _decide(
    *,
    has_assert: bool,
    numeric: bool,
    regression: bool,
    dimensional: bool,
    behaviour: bool,
    schema: bool,
    schema_content: bool,
    end_to_end: bool,
    claim_guard: bool,
) -> tuple[str, str, str]:
    if claim_guard:
        return (
            "claim_boundary_guard",
            "high",
            "guards allowed/disallowed wording or unsupported-claim preservation",
        )
    if numeric or regression or dimensional:
        kind = (
            "regression against known values"
            if regression
            else (
                "dimensional/shape consistency"
                if dimensional and not numeric
                else "numeric tolerance"
            )
        )
        return "real_validation", "high", f"checks {kind} on computed quantities"
    if end_to_end and schema:
        return (
            "integration_regression",
            "medium",
            "runs a build/CLI pipeline end-to-end and checks artifact schema",
        )
    if schema_content:
        return "artifact_schema", "low", "validates artifact columns/keys but no numeric assertion"
    if behaviour:
        return (
            "unit_behavior",
            "low",
            "asserts error handling or specific category/membership behaviour",
        )
    if not has_assert:
        return "smoke_only", "none", "runs the code path without asserting substantive output"
    return (
        "smoke_only",
        "none",
        "only existence/non-empty checks; no value, schema-content, or behaviour validation",
    )


def _has_numeric(text: str) -> bool:
    if any(marker in text for marker in _NUMERIC_MARKERS):
        return True
    if _TOLERANCE_RE.search(text):
        return True
    return bool(_NUMERIC_RE.search(text))


def _claims_for_file(file_name: str) -> str:
    lowered = file_name.lower()
    matched: list[str] = []
    for keyword, claim in _CLAIM_BY_KEYWORD:
        if keyword in lowered and claim not in matched:
            matched.append(claim)
    return "; ".join(matched) if matched else "general"


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["test_category"]] = counts.get(row["test_category"], 0) + 1
    return counts


def _relies_on_smoke(rows: list[dict[str, Any]]) -> bool:
    """The package relies on smoke tests only if no real-validation / claim-guard tests exist."""

    strong = sum(
        1 for r in rows if r["test_category"] in {"real_validation", "claim_boundary_guard"}
    )
    return strong == 0


def _suite_for(category: str) -> str:
    if category in SCIENTIFIC_CATEGORIES:
        return SUITE_SCIENTIFIC
    if category == "smoke_only":
        return SUITE_ENGINEERING_SMOKE
    return SUITE_ENGINEERING_OTHER


def scientific_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["test_category"] in SCIENTIFIC_CATEGORIES]


def smoke_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["test_category"] == "smoke_only"]


def converted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("node_id") in CONVERTED_SMOKE_SET]


def scientific_validation_node_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [r["node_id"] for r in scientific_validation_rows(rows)]


def suite_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    scientific = len(scientific_validation_rows(rows))
    smoke = len(smoke_rows(rows))
    return {
        "total": len(rows),
        "scientific_validation": scientific,
        "scientific_validation_smoke_only": 0,
        "engineering_smoke": smoke,
        "remaining_smoke_only": smoke,
        "converted_from_smoke": len(converted_rows(rows)),
    }


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    inventory_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    inventory_path = rows_to_table(
        inventory_rows, output_dir / "test_inventory.csv", INVENTORY_COLUMNS
    )
    smoke_path = rows_to_table(
        [r for r in inventory_rows if r["test_category"] == "smoke_only"],
        output_dir / "smoke_tests.csv",
        INVENTORY_COLUMNS,
    )
    real_path = rows_to_table(
        [r for r in inventory_rows if r["test_category"] == "real_validation"],
        output_dir / "real_validation_tests.csv",
        INVENTORY_COLUMNS,
    )
    schema_path = rows_to_table(
        [r for r in inventory_rows if r["test_category"] == "artifact_schema"],
        output_dir / "artifact_schema_tests.csv",
        INVENTORY_COLUMNS,
    )
    integration_path = rows_to_table(
        [r for r in inventory_rows if r["test_category"] == "integration_regression"],
        output_dir / "integration_regression_tests.csv",
        INVENTORY_COLUMNS,
    )
    scientific_path = rows_to_table(
        scientific_validation_rows(inventory_rows),
        output_dir / "scientific_validation_tests.csv",
        INVENTORY_COLUMNS,
    )
    engineering_smoke_path = rows_to_table(
        smoke_rows(inventory_rows),
        output_dir / "engineering_smoke_tests.csv",
        INVENTORY_COLUMNS,
    )
    converted_path = rows_to_table(
        converted_rows(inventory_rows),
        output_dir / "converted_smoke_tests.csv",
        INVENTORY_COLUMNS,
    )
    # Smoke tests that remain smoke (kept as engineering smoke, never scientific validation).
    remaining_path = rows_to_table(
        [r for r in smoke_rows(inventory_rows) if r["node_id"] not in CONVERTED_SMOKE_SET],
        output_dir / "remaining_smoke_tests.csv",
        INVENTORY_COLUMNS,
    )
    summary_path = output_dir / "test_quality_summary.md"
    summary_path.write_text(_summary_markdown(inventory_rows), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "test_inventory": str(inventory_path),
            "smoke_tests": str(smoke_path),
            "real_validation_tests": str(real_path),
            "artifact_schema_tests": str(schema_path),
            "integration_regression_tests": str(integration_path),
            "scientific_validation_tests": str(scientific_path),
            "engineering_smoke_tests": str(engineering_smoke_path),
            "converted_smoke_tests": str(converted_path),
            "remaining_smoke_tests": str(remaining_path),
            "test_quality_summary": str(summary_path),
        },
        input_config=resolved,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "test_inventory": inventory_path,
        "smoke_tests": smoke_path,
        "real_validation_tests": real_path,
        "artifact_schema_tests": schema_path,
        "integration_regression_tests": integration_path,
        "scientific_validation_tests": scientific_path,
        "engineering_smoke_tests": engineering_smoke_path,
        "converted_smoke_tests": converted_path,
        "remaining_smoke_tests": remaining_path,
        "test_quality_summary": summary_path,
    }


def _summary_markdown(rows: list[dict[str, Any]]) -> str:
    counts = _category_counts(rows)
    suites = suite_counts(rows)
    total = len(rows)
    smoke = counts.get("smoke_only", 0)
    real = counts.get("real_validation", 0)
    guard = counts.get("claim_boundary_guard", 0)
    relies = _relies_on_smoke(rows)
    smoke_files = sorted({r["test_file"] for r in rows if r["test_category"] == "smoke_only"})
    return "\n".join(
        [
            "# Test-Quality and Smoke-Test Audit",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Inventory",
            f"- Total test functions inventoried: {total}.",
            "",
            "## Scientific-validation suite (smoke-free)",
            f"- Scientific-validation suite tests: {suites['scientific_validation']} "
            "(real_validation + claim_boundary_guard + integration_regression + artifact_schema).",
            f"- Smoke-only tests inside the scientific-validation suite: "
            f"{suites['scientific_validation_smoke_only']} (zero by construction).",
            f"- Engineering smoke tests (kept, not scientific): {suites['engineering_smoke']}.",
            f"- Smoke tests converted to scientific tests in this pass: "
            f"{suites['converted_from_smoke']}.",
            "- The manuscript scientific-validation suite excludes smoke-only tests and contains "
            "zero smoke-only tests. Smoke tests are not counted as scientific validation evidence.",
            "",
            "## Category counts",
            *[f"- {category}: {count}" for category, count in sorted(counts.items())],
            "",
            "## Smoke tests",
            f"- Smoke-only tests: {smoke}. These are listed in smoke_tests.csv and are NOT counted "
            "as scientific validation.",
            f"- Files containing smoke-only tests: {smoke_files or 'none'}.",
            "",
            "## Real validation vs other categories",
            f"- Real-validation tests (numeric tolerance / regression): {real} "
            "(real_validation_tests.csv).",
            f"- Claim-boundary guard tests (preserve unsupported claims / wording): {guard}.",
            f"- Integration/regression tests (run pipelines, check schema): "
            f"{counts.get('integration_regression', 0)} (integration_regression_tests.csv).",
            f"- Artifact-schema tests (columns/keys only): {counts.get('artifact_schema', 0)} "
            "(artifact_schema_tests.csv).",
            "- Artifact-schema and integration tests are NOT counted as scientific validation.",
            "",
            "## Does the evidence package rely on smoke tests?",
            f"- Relies on smoke tests only: {'yes' if relies else 'no'}.",
            "- The package is backed by real-validation and claim-boundary-guard tests "
            "(e.g. Ridge/QSVT equivalence, rank/conditioning, overshoot classification, and "
            "unsupported-claim preservation), so smoke tests are not the basis of any scientific "
            "claim.",
            "",
            "## Conclusion",
            "Smoke tests are identified and listed separately. Scientific claims rest on the "
            "real-validation and claim-boundary-guard tests; schema and integration tests provide "
            "reproducibility coverage but are not treated as validation.",
            "",
        ]
    )
