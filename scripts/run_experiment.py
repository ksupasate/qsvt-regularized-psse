#!/usr/bin/env python3
"""Run a configured experiment without overwriting frozen evidence by default."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standard or QSVT validation config. Unless --output-dir is supplied, "
            "artifacts are isolated under outputs/generated/."
        )
    )
    parser.add_argument("--config", required=True, help="YAML configuration path.")
    parser.add_argument(
        "--output-dir",
        help="Exact output directory (default: outputs/generated/<configured-run-id>).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a standard sweep from its checkpoint files; unsupported for QSVT configs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots for standard experiment configs.",
    )
    parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Refuse to write when the selected output directory is nonempty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dispatch and print the destination without running the experiment.",
    )
    return parser


def _load_raw_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config file must contain a mapping")
    return loaded


def _config_kind(raw: dict[str, Any]) -> str:
    if "run_name" in raw or "system" in raw:
        return "standard"
    if "demo" in raw:
        return "qsvt_phase_response"
    if "scaling" in raw:
        return "qsvt_circuit_scaling"
    if "resource" in raw:
        return "qsvt_resource_estimation"
    raise ValueError(
        "unsupported config schema: expected standard, demo, scaling, or resource top-level keys"
    )


def _configured_run_id(raw: dict[str, Any], kind: str, config_path: Path) -> str:
    if kind == "standard":
        output = raw.get("output", {})
        if isinstance(output, dict) and output.get("run_id"):
            return str(output["run_id"])
        if raw.get("run_name"):
            seed = raw.get("seed", 123)
            return f"{raw['run_name']}_seed{seed}"
    section = {
        "qsvt_phase_response": "demo",
        "qsvt_circuit_scaling": "scaling",
        "qsvt_resource_estimation": "resource",
    }.get(kind)
    if section:
        payload = raw.get(section, {})
        if isinstance(payload, dict) and payload.get("run_id"):
            return str(payload["run_id"])
    return config_path.stem


def _resolve_from_repo(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _check_destination(output_dir: Path, *, fail_if_exists: bool) -> None:
    if fail_if_exists and output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is nonempty: {_display_path(output_dir)}")


def _configure_runtime_cache(output_dir: Path) -> None:
    cache_root = output_dir.parent / "_runtime_cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def _run_standard(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool,
    no_plots: bool,
) -> dict[str, Any]:
    from robust_qsvt_se.experiments.runner import run_experiment
    from robust_qsvt_se.utils.config import load_config

    config = load_config(config_path)
    config["output"]["root"] = str(output_dir.parent)
    config["output"]["run_id"] = output_dir.name
    config["output"]["overwrite"] = not resume
    if no_plots:
        config["output"]["save_plots"] = False
    return run_experiment(config, resume=resume)


def _run_phase_response(config_path: Path, output_dir: Path) -> dict[str, Any]:
    from robust_qsvt_se.qsvt.run_phase_demo import load_phase_demo_config, run_phase_demo

    config = load_phase_demo_config(config_path)
    config["demo"]["output_dir"] = str(output_dir)
    config["demo"]["phase_cache_dir"] = str(output_dir.parent / "_phase_cache")
    return run_phase_demo(config)


def _run_circuit_scaling(config_path: Path, output_dir: Path) -> dict[str, Any]:
    from robust_qsvt_se.qsvt.circuit_scaling import (
        load_circuit_scaling_config,
        run_circuit_scaling,
    )

    config = load_circuit_scaling_config(config_path)
    config["scaling"]["output_dir"] = str(output_dir)
    config["scaling"]["phase_cache_dir"] = str(output_dir.parent / "_phase_cache")
    return run_circuit_scaling(config)


def _run_resource_estimation(config_path: Path, output_dir: Path) -> dict[str, Any]:
    from robust_qsvt_se.qsvt.matrix_resource_estimation import (
        load_resource_config,
        run_resource_estimation,
    )

    config = load_resource_config(config_path)
    config["resource"]["output_dir"] = str(output_dir)
    return run_resource_estimation(config)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    config_path = _resolve_from_repo(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")

    raw = _load_raw_config(config_path)
    kind = _config_kind(raw)
    run_id = _configured_run_id(raw, kind, config_path)
    output_dir = _resolve_from_repo(args.output_dir or Path("outputs/generated") / run_id)
    _check_destination(output_dir, fail_if_exists=args.fail_if_exists)
    if args.resume and kind != "standard":
        raise ValueError("--resume is supported only for standard experiment configs")

    print(f"Config: {_display_path(config_path)}", flush=True)
    print(f"Experiment type: {kind}", flush=True)
    print(f"Output: {_display_path(output_dir)}", flush=True)
    if args.dry_run:
        print("Dry run complete; no artifacts were written.", flush=True)
        return 0

    _configure_runtime_cache(output_dir)
    runners = {
        "standard": lambda: _run_standard(
            config_path,
            output_dir,
            resume=args.resume,
            no_plots=args.no_plots,
        ),
        "qsvt_phase_response": lambda: _run_phase_response(config_path, output_dir),
        "qsvt_circuit_scaling": lambda: _run_circuit_scaling(config_path, output_dir),
        "qsvt_resource_estimation": lambda: _run_resource_estimation(config_path, output_dir),
    }
    started = perf_counter()
    result = runners[kind]()
    elapsed = perf_counter() - started
    actual_output = Path(result.get("output_dir", output_dir))
    generated = sorted(path for path in actual_output.rglob("*") if path.is_file())

    print(f"Run complete in {elapsed:.3f} seconds.", flush=True)
    print(f"Generated {len(generated)} files:", flush=True)
    for path in generated:
        print(f"  - {_display_path(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
