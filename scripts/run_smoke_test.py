#!/usr/bin/env python3
"""Run a fast, isolated smoke suite across the core experiment modes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_experiment.py"
SMOKE_CONFIGS = (
    ("synthetic", "configs/ieee14_spectral_smoke.yaml"),
    ("dc_linearized", "configs/ieee14_dc_smoke.yaml"),
    ("ac_linearized", "configs/ieee14_ac_smoke.yaml"),
    ("iterative_ac", "configs/ieee14_ac_iterative_smoke.yaml"),
    ("robust_bad_data", "configs/ieee14_robust_bad_data_smoke.yaml"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated synthetic, DC, AC, iterative-AC, and robust smoke checks."
    )
    parser.add_argument(
        "--output-root",
        default="outputs/examples/smoke_test",
        help="Parent directory for smoke artifacts.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run only the controlled synthetic smoke config.",
    )
    parser.add_argument(
        "--keep-plots",
        action="store_true",
        help="Generate diagnostic plots; disabled by default for speed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(args.output_root)
    selected = SMOKE_CONFIGS[:1] if args.minimal else SMOKE_CONFIGS
    failures: list[tuple[str, int]] = []
    durations: list[tuple[str, float]] = []
    suite_start = perf_counter()
    runtime_cache = output_root / "_runtime_cache"
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(runtime_cache / "matplotlib"))
    environment.setdefault("XDG_CACHE_HOME", str(runtime_cache / "xdg"))

    for name, config in selected:
        destination = output_root / name
        command = [
            sys.executable,
            str(RUNNER),
            "--config",
            config,
            "--output-dir",
            str(destination),
        ]
        if not args.keep_plots:
            command.append("--no-plots")
        print(f"\n=== Smoke case: {name} ===", flush=True)
        started = perf_counter()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            env=environment,
        )
        durations.append((name, perf_counter() - started))
        if completed.returncode != 0:
            failures.append((name, completed.returncode))

    elapsed = perf_counter() - suite_start
    print("\nSmoke suite summary")
    for name, duration in durations:
        status = "FAIL" if any(item[0] == name for item in failures) else "PASS"
        print(f"  {status:4s} {name:18s} {duration:8.3f} seconds")
    print(f"Total runtime: {elapsed:.3f} seconds")
    print(f"Output root: {output_root}")
    if failures:
        print(f"Smoke test FAILED ({len(failures)} case(s)).")
        return 1
    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
