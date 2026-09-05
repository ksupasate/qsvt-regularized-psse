#!/usr/bin/env python3
"""Run every pytest file in a fresh process to avoid macOS fork/thread aborts.

Durable checkpoint/resume: progress is written atomically to a JSON file after
every completed test file. A restarted session seeds itself from the progress
file and from every definitively flushed ``return_code=`` block in the log, so
only files without a confirmed pass are (re)executed. A file header in the log
without a matching return-code line (an in-flight block cut off by an
interruption) is never counted as a result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "tqe_blocking_revision"

FILE_HEADER = re.compile(r"^===== \[\d+/\d+\] (?P<path>\S+) =====$")
RETURN_CODE = re.compile(r"^return_code=(?P<code>-?\d+)$")
PASSED_COUNT = re.compile(r"(\d+) passed")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_log_return_codes(log_path: Path) -> dict[str, tuple[int, bool]]:
    """Map each test file to (last flushed return code, KeyboardInterrupt seen)."""
    results: dict[str, tuple[int, bool]] = {}
    if not log_path.exists():
        return results
    current: str | None = None
    interrupted_marker = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = FILE_HEADER.match(line)
        if header:
            current = header.group("path")
            interrupted_marker = False
            continue
        if current is None:
            continue
        if "KeyboardInterrupt" in line:
            interrupted_marker = True
        match = RETURN_CODE.match(line)
        if match:
            results[current] = (int(match.group("code")), interrupted_marker)
            current = None
    return results


def classify(return_code: int, interrupted_marker: bool) -> str:
    if return_code == 0:
        return "passed"
    # Negative codes are signal terminations (host-level, not assertion
    # failures); pytest exit 2 with a KeyboardInterrupt trace is a user or
    # session interruption. Both must be rerun, never reported as test
    # failures or silently dropped.
    if return_code < 0 or (return_code == 2 and interrupted_marker):
        return "interrupted"
    return "failed"


def fresh_progress() -> dict:
    return {
        "total_files": 0,
        "completed_files": [],
        "passed_files": [],
        "failed_files": [],
        "interrupted_files": [],
        "last_completed_file": "",
        "started_at": utc_now(),
        "updated_at": "",
        "environment": {"MPLBACKEND": "Agg"},
        "file_results": {},
    }


def load_progress(progress_path: Path, log_path: Path) -> dict:
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress.setdefault("file_results", {})
        progress.setdefault("interrupted_files", [])
    else:
        progress = fresh_progress()
    for name, (code, marker) in parse_log_return_codes(log_path).items():
        if name in progress["file_results"]:
            continue
        status = classify(code, marker)
        progress["file_results"][name] = {
            "return_code": code,
            "status": status,
            "source": "seeded_from_log",
        }
    return progress


def rebuild_lists(progress: dict, discovered: list[str]) -> None:
    known = progress["file_results"]
    progress["total_files"] = len(discovered)
    progress["passed_files"] = [
        name for name in discovered if known.get(name, {}).get("status") == "passed"
    ]
    progress["failed_files"] = [
        name for name in discovered if known.get(name, {}).get("status") == "failed"
    ]
    progress["interrupted_files"] = [
        name
        for name in discovered
        if known.get(name, {}).get("status") == "interrupted"
    ]
    progress["completed_files"] = [
        name
        for name in discovered
        if known.get(name, {}).get("status") in ("passed", "failed")
    ]


def write_progress(progress: dict, progress_path: Path) -> None:
    progress["updated_at"] = utc_now()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_name(progress_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, progress_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        "--log-file",
        dest="log",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "full_pytest.log",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "full_pytest_progress.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files with a confirmed pass; rerun failed/interrupted/missing.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="With --resume: rerun only previously failed or interrupted files.",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Parse the log into the progress file and exit without running pytest.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Allow truncating an existing log/progress pair (destructive).",
    )
    args = parser.parse_args()

    log_path = args.log.resolve()
    progress_path = args.progress_file.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    resume_mode = args.resume or args.seed_only or args.only_failed
    if not resume_mode and not args.fresh and (
        log_path.exists() or progress_path.exists()
    ):
        print(
            "Refusing to truncate an existing log/progress file; "
            "pass --resume to continue it or --fresh to overwrite.",
            file=sys.stderr,
        )
        return 2

    discovered = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "tests").glob("test_*.py"))
    ]

    progress = (
        load_progress(progress_path, log_path) if resume_mode else fresh_progress()
    )
    rebuild_lists(progress, discovered)
    write_progress(progress, progress_path)

    if args.seed_only:
        print(
            f"seeded progress: {len(progress['passed_files'])} passed, "
            f"{len(progress['failed_files'])} failed, "
            f"{len(progress['interrupted_files'])} interrupted, "
            f"total {progress['total_files']}"
        )
        return 0

    known = progress["file_results"]
    if args.only_failed:
        to_run = [
            name
            for name in discovered
            if known.get(name, {}).get("status") in ("failed", "interrupted")
        ]
    else:
        to_run = [
            name
            for name in discovered
            if known.get(name, {}).get("status") != "passed"
        ]
    skipped = len(discovered) - len(to_run)

    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"

    session_passed = 0
    session_failed: list[str] = []
    session_tests_passed = 0
    total = len(discovered)

    with log_path.open("a" if resume_mode else "w", encoding="utf-8") as log:
        log.write(
            f"\n===== SESSION {utc_now()} mode="
            f"{'resume' if resume_mode else 'fresh'} "
            f"skip_confirmed_passed={skipped} to_run={len(to_run)} =====\n\n"
        )
        log.flush()
        for name in to_run:
            position = discovered.index(name) + 1
            start = time.monotonic()
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", name],
                    cwd=ROOT,
                    capture_output=True,
                    env=environment,
                    text=True,
                )
            except KeyboardInterrupt:
                known[name] = {
                    "return_code": None,
                    "status": "interrupted",
                    "source": "runner_keyboard_interrupt",
                }
                rebuild_lists(progress, discovered)
                write_progress(progress, progress_path)
                log.write(
                    f"===== [{position}/{total}] {name} =====\n"
                    "runner interrupted by KeyboardInterrupt before completion\n\n"
                )
                log.flush()
                print(f"interrupted during {name}; progress saved", flush=True)
                return 130
            duration = time.monotonic() - start
            combined = result.stdout + result.stderr
            log.write(f"===== [{position}/{total}] {name} =====\n")
            log.write(combined)
            if not combined.endswith("\n"):
                log.write("\n")
            log.write(f"return_code={result.returncode}\n\n")
            log.flush()

            status = classify(result.returncode, "KeyboardInterrupt" in combined)
            entry = {
                "return_code": result.returncode,
                "status": status,
                "duration_s": round(duration, 2),
                "finished_at": utc_now(),
                "source": "executed",
            }
            if status == "passed":
                session_passed += 1
                matches = PASSED_COUNT.findall(combined)
                tests = sum(int(value) for value in matches[-1:])
                entry["tests_passed"] = tests
                session_tests_passed += tests
                progress["last_completed_file"] = name
            elif status == "failed":
                session_failed.append(name)
                progress["last_completed_file"] = name
            known[name] = entry
            rebuild_lists(progress, discovered)
            write_progress(progress, progress_path)

            done = len(progress["completed_files"])
            if done % 25 == 0 or status != "passed":
                print(
                    f"filewise pytest: {done}/{total} completed; "
                    f"passed={len(progress['passed_files'])} "
                    f"failed={len(progress['failed_files'])} "
                    f"interrupted={len(progress['interrupted_files'])}",
                    flush=True,
                )

        overall_passed = len(progress["passed_files"])
        overall_failed = progress["failed_files"]
        overall_interrupted = progress["interrupted_files"]
        log.write("===== FINAL FILEWISE SUMMARY =====\n")
        log.write(f"total_files={total}\n")
        log.write(f"files_passed={overall_passed}\n")
        log.write(f"files_failed={len(overall_failed)}\n")
        log.write(f"files_interrupted={len(overall_interrupted)}\n")
        log.write(f"session_files_run={len(to_run)}\n")
        log.write(f"session_files_passed={session_passed}\n")
        log.write(f"session_tests_reported_passed={session_tests_passed}\n")
        if overall_failed or overall_interrupted:
            for name in overall_failed:
                log.write(f"failed: {name}\n")
            for name in overall_interrupted:
                log.write(f"interrupted: {name}\n")
            log.write("FULL FILEWISE SUITE FAIL\n")
        else:
            log.write(
                f"FULL FILEWISE SUITE PASS: {overall_passed}/{total} files\n"
            )
    print(
        f"done: {overall_passed}/{total} passed, "
        f"{len(overall_failed)} failed, {len(overall_interrupted)} interrupted",
        flush=True,
    )
    return 1 if overall_failed or overall_interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
