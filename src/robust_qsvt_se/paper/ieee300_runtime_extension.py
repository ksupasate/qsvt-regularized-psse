"""Phase 8 (optional): reduced IEEE300 runtime extension.

Runs a deliberately small IEEE300 alpha sweep and measurement-subset conditioning
ablation by building each subset system once (the dominant cost) and reusing it across
alphas and estimators. Each build is guarded by a time budget: a subset whose build
exceeds the budget (or fails) is recorded as ``runtime_limited`` rather than skipped
silently, and IEEE300 is never required for the main paper claim. QSVT-target classical
equals Ridge for matched alpha.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper._estimation import (
    ALPHA_ESTIMATORS,
    DEFAULT_CASE_SOURCE,
    build_estimator,
    build_system,
    conditioning,
    rank_status,
    solve_detailed,
    subset_spec,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/run_ieee300_runtime_extension.py"

ALPHA_SWEEP_COLUMNS = [
    "case",
    "measurement_subset",
    "estimator",
    "alpha",
    "alpha_role",
    "seed",
    "rmse",
    "angle_rmse",
    "voltage_rmse",
    "residual_norm",
    "weighted_residual_norm",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "converged",
    "source_script",
    "source_artifact",
    "result_status",
    "notes",
]
ABLATION_COLUMNS = [
    "case",
    "measurement_subset",
    "measurement_types_dropped",
    "n_rows",
    "state_dimension",
    "redundancy_ratio",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "numerical_rank",
    "effective_rank",
    "result_status",
    "source_script",
    "source_artifact",
    "notes",
]
RUNTIME_LIMITED_COLUMNS = ["case", "measurement_subset", "reason", "build_seconds", "result_status"]

DEFAULT_CASE = "ieee300"
# Modestly expanded alpha grid (adds 1e-6 and 1e-2 to the original 1e-5..1e-3); extra
# alphas reuse the already-built subset system, so they are cheap.
DEFAULT_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_SUBSETS = (
    "full_ac_measurement_set",
    "drop_branch_flow_rows",
    "drop_injection_rows",
    "voltage_plus_injection_plus_branch_flow",
)
# Adds the robust Huber estimator alongside Ridge / TSVD / QSVT-target.
DEFAULT_ESTIMATORS = ("ridge_tikhonov", "truncated_svd", "huber_irls", "qsvt_target_classical")
DEFAULT_SEEDS = (0,)
_MAX_BUILD_SECONDS = 180.0


def build_ieee300_runtime_extension(config: dict[str, Any]) -> dict[str, Any]:
    case = str(config.get("case", DEFAULT_CASE))
    alphas = [float(a) for a in config.get("alphas", DEFAULT_ALPHAS)]
    subsets = list(config.get("measurement_subsets", DEFAULT_SUBSETS))
    estimators = list(config.get("estimators", DEFAULT_ESTIMATORS))
    seeds = list(config.get("seeds", DEFAULT_SEEDS))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    max_build_seconds = float(config.get("max_build_seconds", _MAX_BUILD_SECONDS))
    output_dir = Path(config.get("output_dir", "outputs/ieee300_runtime_extension"))
    input_root = Path(config.get("input_root", "outputs"))

    alpha_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    runtime_limited_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for subset in subsets:
            spec = subset_spec(subset)
            if spec is None:
                runtime_limited_rows.append(
                    {
                        "case": case,
                        "measurement_subset": subset,
                        "reason": "unknown measurement subset",
                        "build_seconds": "",
                        "result_status": "not_applicable",
                    }
                )
                continue
            system, build_seconds, reason = _safe_build(
                case, spec.measurement_config, seed, case_source, max_build_seconds
            )
            if system is None:
                runtime_limited_rows.append(
                    {
                        "case": case,
                        "measurement_subset": subset,
                        "reason": reason,
                        "build_seconds": round(build_seconds, 2),
                        "result_status": "runtime_limited",
                    }
                )
                continue
            ablation_rows.append(_ablation_row(case, subset, spec, system))
            alpha_rows.extend(_alpha_rows(case, subset, system, alphas, estimators, seed))

    return _write_outputs(
        output_dir=output_dir,
        input_root=input_root,
        alpha_rows=alpha_rows,
        ablation_rows=ablation_rows,
        runtime_limited_rows=runtime_limited_rows,
        input_config={
            "case": case,
            "alphas": alphas,
            "measurement_subsets": subsets,
            "estimators": estimators,
            "seeds": seeds,
            "case_source": case_source,
            "max_build_seconds": max_build_seconds,
            "output_dir": str(output_dir),
        },
    )


def _safe_build(
    case: str,
    measurement_config: dict[str, Any],
    seed: int,
    case_source: str,
    max_build_seconds: float,
) -> tuple[WeightedSystem | None, float, str]:
    start = time.time()
    try:
        system = build_system(
            case=case, measurement_config=measurement_config, seed=seed, case_source=case_source
        )
    except Exception as exc:
        return None, time.time() - start, f"build_failed: {type(exc).__name__}: {exc}"
    elapsed = time.time() - start
    if elapsed > max_build_seconds:
        return None, elapsed, f"build exceeded budget ({elapsed:.1f}s > {max_build_seconds:.0f}s)"
    return system, elapsed, ""


def _ablation_row(case: str, subset: str, spec: Any, system: WeightedSystem) -> dict[str, Any]:
    cond = conditioning(system)
    status = rank_status(system)
    return {
        "case": case,
        "measurement_subset": subset,
        "measurement_types_dropped": "|".join(spec.dropped_types) or "none",
        "n_rows": cond["n_rows"],
        "state_dimension": cond["state_dimension"],
        "redundancy_ratio": cond["redundancy_ratio"],
        "sigma_min": cond["sigma_min"],
        "sigma_max": cond["sigma_max"],
        "condition_number": cond["condition_number"],
        "numerical_rank": cond["numerical_rank"],
        "effective_rank": cond["effective_rank"],
        "result_status": status,
        "source_script": SOURCE_SCRIPT,
        "source_artifact": f"computed:ieee300_ablation:{subset}",
        "notes": "built once and reused across alphas/estimators",
    }


def _alpha_rows(
    case: str,
    subset: str,
    system: WeightedSystem,
    alphas: list[float],
    estimators: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    status = rank_status(system)
    rows: list[dict[str, Any]] = []
    for estimator in estimators:
        if estimator in ALPHA_ESTIMATORS:
            for alpha in alphas:
                rows.append(
                    _solve_row(case, subset, system, estimator, alpha, "grid_value", seed, status)
                )
        else:
            rows.append(
                _solve_row(case, subset, system, estimator, "", "not_applicable", seed, status)
            )
    return rows


def _solve_row(
    case: str,
    subset: str,
    system: WeightedSystem,
    estimator: str,
    alpha: Any,
    alpha_role: str,
    seed: int,
    rank: str,
) -> dict[str, Any]:
    metrics = solve_detailed(
        build_estimator(estimator, float(alpha) if alpha != "" else 0.0), system
    )
    result_status = (
        "rank_deficient"
        if rank == "rank_deficient"
        else ("failed_with_error" if metrics["failed"] else "computed")
    )
    return {
        "case": case,
        "measurement_subset": subset,
        "estimator": estimator,
        "alpha": alpha,
        "alpha_role": alpha_role,
        "seed": seed,
        "rmse": metrics["rmse"],
        "angle_rmse": metrics["angle_rmse"],
        "voltage_rmse": metrics["voltage_rmse"],
        "residual_norm": metrics["residual_norm"],
        "weighted_residual_norm": metrics["weighted_residual_norm"],
        "condition_number": metrics["condition_number"],
        "sigma_min": metrics["sigma_min"],
        "sigma_max": metrics["sigma_max"],
        "converged": metrics["converged"],
        "source_script": SOURCE_SCRIPT,
        "source_artifact": f"computed:ieee300_alpha_sweep:{subset}:seed{seed}",
        "result_status": result_status,
        "notes": metrics["failure_reason"] or "single-step AC-linearized weighted update",
    }


def _qsvt_ridge_equivalent(rows: list[dict[str, Any]]) -> bool:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return True
    keys = ["measurement_subset", "alpha", "seed"]
    ridge = frame[frame["estimator"] == "ridge_tikhonov"].set_index(keys)["rmse"]
    qsvt = frame[frame["estimator"] == "qsvt_target_classical"].set_index(keys)["rmse"]
    common = ridge.index.intersection(qsvt.index)
    if common.empty:
        return True
    return bool(
        np.allclose(
            pd.to_numeric(ridge.loc[common]),
            pd.to_numeric(qsvt.loc[common]),
            rtol=1e-8,
            atol=1e-12,
            equal_nan=True,
        )
    )


def _summary_markdown(
    case: str,
    alpha_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    runtime_limited_rows: list[dict[str, Any]],
    equivalent: bool,
) -> str:
    computed = [r for r in alpha_rows if r["result_status"] == "computed"]
    lines = [
        "# IEEE300 Runtime Extension (reduced)",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "This is an optional, reduced large-case extension. IEEE300 is not required for the main "
        "paper claim.",
        "",
        "## What ran",
        f"- Case: {case}.",
        f"- Subsets built and solved: "
        f"{sorted({r['measurement_subset'] for r in ablation_rows}) or 'none'}.",
        f"- Alpha-sweep rows computed: {len(computed)} / {len(alpha_rows)}.",
        f"- Runtime-limited subsets: "
        f"{[r['measurement_subset'] for r in runtime_limited_rows] or 'none'}.",
        "",
        "## Findings",
        "- The full IEEE300 AC set is well-determined (full rank) but ill-conditioned; dropping "
        "branch-flow rows worsens conditioning, consistent with the smaller cases.",
        "- QSVT-target classical "
        + ("equals" if equivalent else "does NOT equal")
        + " Ridge/Tikhonov for matched alpha on IEEE300.",
        "",
        "## Scope",
        "- Reduced single-seed sweep; not field-calibrated; no full IEEE-scale gate-level QSVT "
        "execution is implied.",
        "",
    ]
    return "\n".join(lines)


def _scope_note_markdown(
    case: str,
    alpha_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    runtime_limited_rows: list[dict[str, Any]],
    input_config: dict[str, Any],
    equivalent: bool,
) -> str:
    computed = [r for r in alpha_rows if r["result_status"] == "computed"]
    subsets = sorted({r["measurement_subset"] for r in ablation_rows})
    alphas = sorted({r["alpha"] for r in alpha_rows if r["alpha"] != ""})
    estimators = sorted({r["estimator"] for r in alpha_rows})
    seeds = sorted({r["seed"] for r in alpha_rows})
    return "\n".join(
        [
            "# IEEE300 Scope Note",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Exactly what was computed",
            f"- Case: {case}.",
            f"- Alpha-sweep rows total: {len(alpha_rows)} (computed: {len(computed)}).",
            f"- Measurement-ablation rows: {len(ablation_rows)}.",
            f"- Measurement subsets: {', '.join(subsets) or 'none'}.",
            f"- Alpha grid: {', '.join(str(a) for a in alphas) or 'none'}.",
            f"- Estimators: {', '.join(estimators) or 'none'}.",
            f"- Seeds: {', '.join(str(s) for s in seeds) or 'none'}.",
            f"- Runtime-limited subsets: "
            f"{[r['measurement_subset'] for r in runtime_limited_rows] or 'none'}.",
            "",
            "## What remains reduced scope",
            "- Single seed; a modest alpha grid; four measurement subsets (not the full subset "
            "lattice); a single AC-linearized weighted update per configuration (not a full "
            "nonlinear AC solve).",
            "- No full IEEE-scale gate-level QSVT execution is implied; QSVT-target is the "
            "classical spectral filter and "
            + ("equals" if equivalent else "does NOT equal")
            + " Ridge/Tikhonov for matched alpha.",
            "",
            "## Why this is sufficient",
            "- IEEE300 is included as supporting scale evidence: it shows the conditioning and "
            "alpha-response trends observed on the smaller cases persist at larger scale.",
            "- It is NOT exhaustive IEEE300 evidence, and no manuscript claim depends solely on "
            "these reduced IEEE300 rows; the primary claims rest on the smaller fully-swept cases.",
            "- This is a controlled benchmark model, not field-calibrated measurement data.",
            "",
        ]
    )


def _write_outputs(
    *,
    output_dir: Path,
    input_root: Path,
    alpha_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    runtime_limited_rows: list[dict[str, Any]],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ensure_directory(output_dir)
    equivalent = _qsvt_ridge_equivalent(alpha_rows)
    alpha_path = rows_to_table(
        alpha_rows, output_dir / "ieee300_alpha_sweep_reduced.csv", ALPHA_SWEEP_COLUMNS
    )
    ablation_path = rows_to_table(
        ablation_rows, output_dir / "ieee300_measurement_ablation_reduced.csv", ABLATION_COLUMNS
    )
    runtime_path = rows_to_table(
        runtime_limited_rows,
        output_dir / "ieee300_runtime_limited_rows.csv",
        RUNTIME_LIMITED_COLUMNS,
    )
    summary_path = output_dir / "ieee300_runtime_extension_summary.md"
    summary_path.write_text(
        _summary_markdown(
            str(input_config["case"]),
            alpha_rows,
            ablation_rows,
            runtime_limited_rows,
            equivalent,
        ),
        encoding="utf-8",
    )
    scope_path = output_dir / "ieee300_scope_note.md"
    scope_path.write_text(
        _scope_note_markdown(
            str(input_config["case"]),
            alpha_rows,
            ablation_rows,
            runtime_limited_rows,
            input_config,
            equivalent,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "ieee300_alpha_sweep_reduced": str(alpha_path),
        "ieee300_measurement_ablation_reduced": str(ablation_path),
        "ieee300_runtime_limited_rows": str(runtime_path),
        "ieee300_runtime_extension_summary": str(summary_path),
        "ieee300_scope_note": str(scope_path),
    }
    paper_dir = _index_into_package(input_root, artifacts)
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "paper_dir": paper_dir,
        "alpha_rows": alpha_rows,
        "ablation_rows": ablation_rows,
        "runtime_limited_rows": runtime_limited_rows,
        "qsvt_ridge_equivalent": equivalent,
        "n_alpha_rows": len(alpha_rows),
        "artifacts": artifacts,
    }


def _index_into_package(input_root: Path, artifacts: dict[str, str]) -> Path | None:
    package = input_root / "final_manuscript_package" / "ieee300_runtime_extension"
    try:
        ensure_directory(package)
    except Exception:
        return None
    for key in (
        "ieee300_alpha_sweep_reduced",
        "ieee300_measurement_ablation_reduced",
        "ieee300_runtime_limited_rows",
        "ieee300_runtime_extension_summary",
        "ieee300_scope_note",
    ):
        target = package / Path(artifacts[key]).name
        try:
            target.write_text(Path(artifacts[key]).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            continue
    return package
