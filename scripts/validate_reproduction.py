#!/usr/bin/env python3
"""Run the publication-artifact preflight and write a Markdown validation report."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from validate_outputs import (  # noqa: E402
    DEFAULT_MANIFEST,
    ValidationCheck,
    validate_manifest_outputs,
)

DEFAULT_REPORT = REPO_ROOT / "outputs" / "reproducibility_audit" / "validation_report.md"
QUANTUM_PACKAGES = {"mpmath", "pennylane", "pyqsp", "qiskit", "qiskit-aer", "sympy"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Do not require optional quantum/phase packages.",
    )
    parser.add_argument(
        "--skip-regeneration",
        action="store_true",
        help="Check registered files and schemas without rebuilding a temporary report bundle.",
    )
    return parser


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return f"<external>/{path.name}"


def _metadata_checks() -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    for name in (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "VERSION",
        "environment.yml",
        "requirements.txt",
    ):
        path = REPO_ROOT / name
        checks.append(
            ValidationCheck(
                "artifact metadata",
                name,
                "PASS" if path.is_file() else "FAIL",
                "present" if path.is_file() else "missing",
            )
        )
    output_root = REPO_ROOT / "outputs" / "reproducibility_audit"
    writable = output_root.is_dir() and os.access(output_root, os.W_OK)
    checks.append(
        ValidationCheck(
            "environment",
            "audit output directory",
            "PASS" if writable else "FAIL",
            "available and writable" if writable else "missing or not writable",
        )
    )
    return checks


def _python_checks() -> list[ValidationCheck]:
    supported = sys.version_info >= (3, 11)
    checks = [
        ValidationCheck(
            "environment",
            "Python",
            "PASS" if supported else "FAIL",
            f"{sys.version.split()[0]} via {Path(sys.executable).name}; requires >=3.11",
        )
    ]
    conda = shutil.which("conda")
    checks.append(
        ValidationCheck(
            "environment",
            "conda command",
            "PASS" if conda else "WARN",
            conda or "not on PATH; an already-created venv remains usable",
        )
    )
    return checks


def _requirement_pins(path: Path) -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw_line.strip())
        if match:
            pins.append((match.group(1), match.group(2)))
    return pins


def _package_checks(*, core_only: bool) -> list[ValidationCheck]:
    requirements_path = REPO_ROOT / "requirements.txt"
    if not requirements_path.is_file():
        return [
            ValidationCheck("package", "requirements.txt", "FAIL", "requirements file missing")
        ]
    checks: list[ValidationCheck] = []
    for distribution, expected in _requirement_pins(requirements_path):
        if core_only and distribution.casefold() in QUANTUM_PACKAGES:
            checks.append(
                ValidationCheck(
                    "package",
                    distribution,
                    "WARN",
                    "optional package check skipped by --core-only",
                )
            )
            continue
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            checks.append(
                ValidationCheck(
                    "package",
                    distribution,
                    "FAIL",
                    f"not installed; publication snapshot is {expected}",
                )
            )
            continue
        status = "PASS" if installed == expected else "FAIL"
        checks.append(
            ValidationCheck(
                "package",
                distribution,
                status,
                f"installed={installed}; expected={expected}",
            )
        )
    return checks


def _regeneration_checks(*, skip: bool) -> list[ValidationCheck]:
    if skip:
        return [
            ValidationCheck(
                "regeneration",
                "derived tables and figures",
                "WARN",
                "temporary rebuild skipped by request",
            )
        ]

    try:
        from robust_qsvt_se.experiments.paper_ready_results import build_paper_ready_results

        with tempfile.TemporaryDirectory(prefix="qsvt-reproduction-") as temp_dir:
            destination = Path(temp_dir) / "paper_ready_results"
            result = build_paper_ready_results(destination)
            table_paths = result["manifest"].get("generated_tables", {})
            figure_paths = result["manifest"].get("generated_figures", {})
            tables_present = bool(table_paths) and all(
                Path(path).is_file()
                for bundle in table_paths.values()
                for path in bundle.values()
            )
            figures_present = bool(figure_paths) and all(
                Path(path).is_file()
                for bundle in figure_paths.values()
                for path in bundle.values()
            )
            return [
                ValidationCheck(
                    "regeneration",
                    "derived tables",
                    "PASS" if tables_present else "FAIL",
                    f"temporary rebuild produced {len(table_paths)} table bundles",
                ),
                ValidationCheck(
                    "regeneration",
                    "derived figures",
                    "PASS" if figures_present else "FAIL",
                    f"temporary rebuild produced {len(figure_paths)} figure bundles",
                ),
            ]
    except Exception as exc:  # report a structured failure rather than hiding the traceback
        return [
            ValidationCheck(
                "regeneration",
                "derived tables and figures",
                "FAIL",
                f"temporary rebuild failed: {type(exc).__name__}: {exc}",
            )
        ]


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _report_text(checks: list[ValidationCheck], *, elapsed: float) -> str:
    failures = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    passed = sum(check.status == "PASS" for check in checks)
    overall = "PASS" if failures == 0 else "FAIL"
    warning_label = "warning" if warnings == 1 else "warnings"
    rows = [
        "# Reproduction Validation Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        f"Overall status: **{overall}**",
        "",
        f"- Branch: `{_git_value('branch', '--show-current')}`",
        f"- Commit: `{_git_value('rev-parse', 'HEAD')}`",
        f"- Working tree: {'dirty' if _git_value('status', '--porcelain') else 'clean'}",
        f"- Runtime: {elapsed:.3f} seconds",
        f"- Checks: {passed} passed, {warnings} {warning_label}, {failures} failed",
        "",
        "## Checks",
        "",
        "| Status | Category | Target | Detail |",
        "|---|---|---|---|",
    ]
    rows.extend(
        (
            f"| {check.status} | {_escape_table(check.category)} | "
            f"{_escape_table(check.target)} | {_escape_table(check.detail)} |"
        )
        for check in checks
    )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A PASS confirms that the declared environment, registered configs, existing "
                "output schemas, and isolated table/figure rebuild were available in this "
                "working copy. It does not independently re-run every long IEEE sweep, prove "
                "field-data validity, execute full-scale quantum hardware, or establish speedup."
            ),
            "",
        ]
    )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    cache_root = REPO_ROOT / "outputs" / "reproducibility_audit" / "_runtime_cache"
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    started = perf_counter()
    checks = [
        *_metadata_checks(),
        *_python_checks(),
        *_package_checks(core_only=args.core_only),
        *validate_manifest_outputs(args.manifest),
        *_regeneration_checks(skip=args.skip_regeneration),
    ]
    elapsed = perf_counter() - started
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_text(checks, elapsed=elapsed), encoding="utf-8")

    failures = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    print(f"Validation report: {_relative(report_path)}")
    print(f"Checks: {len(checks)}; warnings: {warnings}; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
