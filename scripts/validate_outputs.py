#!/usr/bin/env python3
"""Validate registered experiment outputs against the experiment manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "reproducibility_audit" / "experiment_manifest.json"
REQUIRED_EXPERIMENT_FIELDS = {
    "name",
    "config",
    "config_path",
    "git_commit",
    "model_type",
    "measurement_types",
    "rng",
    "seed",
    "state_dimension",
    "row_count",
    "estimators",
    "metrics",
    "output_path",
}
STANDARD_METRIC_COLUMNS = {
    "estimator",
    "rmse",
    "residual_norm",
    "condition_number",
    "runtime_seconds",
    "failed",
}
AGGREGATE_METRIC_COLUMNS = STANDARD_METRIC_COLUMNS | {
    "seed",
    "sweep_name",
    "sweep_value",
}
SUMMARY_METRIC_COLUMNS = {
    "estimator",
    "rmse_mean",
    "failure_rate",
    "condition_number_mean",
    "runtime_seconds_mean",
}
PORTABLE_OMISSION_FILE = "config_resolved.omission.json"


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    category: str
    target: str
    status: str
    detail: str


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_experiment_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = _resolve(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise ValueError("experiment manifest must contain an experiments list")
    return payload


def _csv_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            return {column.strip() for column in next(reader) if column.strip()}
        except StopIteration:
            return set()


def _csv_check(path: Path, required: set[str]) -> ValidationCheck:
    display = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    if not path.is_file():
        return ValidationCheck("output schema", display, "FAIL", "required CSV is missing")
    columns = _csv_columns(path)
    missing = sorted(required - columns)
    if missing:
        return ValidationCheck(
            "output schema",
            display,
            "FAIL",
            f"missing required columns: {', '.join(missing)}",
        )
    return ValidationCheck(
        "output schema",
        display,
        "PASS",
        f"{len(columns)} columns; required schema present",
    )


def _artifact_checks(experiment: dict[str, Any], output_dir: Path) -> list[ValidationCheck]:
    model_type = str(experiment["model_type"])
    if model_type == "qsvt_phase_response":
        return [
            _csv_check(
                output_dir / "qsp_validation_grid.csv",
                {
                    "normalized_singular_value",
                    "target",
                    "phase_response",
                    "phase_abs_error",
                },
            )
        ]
    if model_type == "qsvt_circuit_scaling":
        return [
            _csv_check(
                output_dir / "circuit_scaling_results.csv",
                {
                    "case_name",
                    "matrix_size",
                    "polynomial_degree",
                    "status",
                    "feasible",
                    "max_error_vs_classical",
                },
            )
        ]
    if model_type == "qsvt_resource_estimation":
        return [
            _csv_check(
                output_dir / "qsvt_resource_estimates.csv",
                {
                    "case_name",
                    "matrix_rows",
                    "matrix_columns",
                    "condition_number",
                    "polynomial_degree",
                    "estimated_total_qubits",
                },
            )
        ]

    metrics_path = output_dir / "metrics.csv"
    if metrics_path.is_file():
        return [_csv_check(metrics_path, STANDARD_METRIC_COLUMNS)]
    return [
        _csv_check(output_dir / "aggregate_metrics.csv", AGGREGATE_METRIC_COLUMNS),
        _csv_check(output_dir / "summary_metrics.csv", SUMMARY_METRIC_COLUMNS),
    ]


def _resolved_config_check(experiment: dict[str, Any], output_dir: Path) -> ValidationCheck:
    """Validate resolved config provenance without exporting a machine-local path.

    Five legacy smoke snapshots record the development checkout in ``output.root``.
    The public export omits those snapshots and publishes a hash-bearing omission
    record instead. All other resolved configurations remain byte-identical.
    """
    display = str(experiment["output_path"])
    resolved_config = output_dir / "config_resolved.yaml"
    if resolved_config.is_file():
        return ValidationCheck(
            "output provenance",
            display,
            "PASS",
            "resolved configuration is present",
        )

    omission_path = output_dir / PORTABLE_OMISSION_FILE
    if omission_path.is_file():
        try:
            omission = json.loads(omission_path.read_text(encoding="utf-8"))
            digest = str(omission.get("excluded_source_sha256", ""))
            source_config = str(omission.get("source_config_path", ""))
            digest_valid = len(digest) == hashlib.sha256().digest_size * 2 and all(
                character in "0123456789abcdef" for character in digest
            )
            record_valid = (
                omission.get("status") == "omitted_from_public_export"
                and omission.get("reason") == "contains_machine_local_output_path"
                and source_config == str(experiment["config_path"])
                and digest_valid
            )
        except (OSError, json.JSONDecodeError, TypeError):
            record_valid = False
        if record_valid:
            return ValidationCheck(
                "output provenance",
                display,
                "WARN",
                (
                    "resolved snapshot intentionally omitted because output.root contained a "
                    "machine-local path; hash-bearing omission record and source config present"
                ),
            )

    return ValidationCheck(
        "output provenance",
        display,
        "FAIL",
        "config_resolved.yaml and a valid portable omission record are missing",
    )


def validate_manifest_outputs(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> list[ValidationCheck]:
    path = _resolve(manifest_path)
    checks: list[ValidationCheck] = []
    if not path.is_file():
        return [ValidationCheck("manifest", str(path), "FAIL", "manifest is missing")]
    try:
        payload = load_experiment_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [ValidationCheck("manifest", str(path), "FAIL", f"cannot load manifest: {exc}")]

    experiments = payload["experiments"]
    checks.append(
        ValidationCheck(
            "manifest",
            str(path.relative_to(REPO_ROOT)),
            "PASS",
            f"loaded {len(experiments)} experiment records",
        )
    )
    names: set[str] = set()
    for index, experiment in enumerate(experiments):
        label = f"experiment[{index}]"
        if not isinstance(experiment, dict):
            checks.append(ValidationCheck("manifest", label, "FAIL", "record is not an object"))
            continue
        missing_fields = sorted(REQUIRED_EXPERIMENT_FIELDS - set(experiment))
        if missing_fields:
            checks.append(
                ValidationCheck(
                    "manifest",
                    label,
                    "FAIL",
                    f"missing fields: {', '.join(missing_fields)}",
                )
            )
            continue
        name = str(experiment["name"])
        if not name or name in names:
            checks.append(
                ValidationCheck("manifest", label, "FAIL", f"empty or duplicate name: {name!r}")
            )
        else:
            names.add(name)
            checks.append(ValidationCheck("manifest", name, "PASS", "required fields present"))

        config_display = str(experiment["config"])
        config_provenance = str(experiment["config_path"])
        seed = experiment["seed"]
        seed_present = (
            isinstance(seed, int)
            or (isinstance(seed, list) and bool(seed))
            or seed == "seed provenance unavailable"
        )
        provenance_ok = (
            config_provenance == config_display
            and str(experiment["git_commit"]) == str(payload.get("source_commit", ""))
            and bool(str(experiment["rng"]).strip())
            and seed_present
        )
        checks.append(
            ValidationCheck(
                "manifest provenance",
                name,
                "PASS" if provenance_ok else "FAIL",
                (
                    "seed, RNG, config path, and source commit are recorded"
                    if provenance_ok
                    else "seed/RNG/config/commit provenance is incomplete or inconsistent"
                ),
            )
        )

        config_path = _resolve(str(experiment["config"]))
        config_detail = (
            "configuration is available" if config_path.is_file() else "configuration is missing"
        )
        checks.append(
            ValidationCheck(
                "config",
                config_display,
                "PASS" if config_path.is_file() else "FAIL",
                config_detail,
            )
        )
        output_dir = _resolve(str(experiment["output_path"]))
        output_exists = output_dir.is_dir()
        checks.append(
            ValidationCheck(
                "output folder",
                str(experiment["output_path"]),
                "PASS" if output_exists else "FAIL",
                "output directory is available" if output_exists else "output directory is missing",
            )
        )
        if not output_exists:
            continue
        checks.append(_resolved_config_check(experiment, output_dir))
        checks.extend(_artifact_checks(experiment, output_dir))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args(argv)
    checks = validate_manifest_outputs(args.manifest)
    for check in checks:
        print(f"{check.status:4s} [{check.category}] {check.target}: {check.detail}")
    failures = sum(check.status == "FAIL" for check in checks)
    print(f"\nValidation checks: {len(checks)}; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
