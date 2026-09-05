from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.amplitude_estimation_audit import (  # noqa: E402
    run_amplitude_estimation_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integration audit for actual amplitude/norm-estimation routines"
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/qsvt_actual_amplitude_estimation_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_amplitude_estimation_audit(
        {"input_root": args.input_root, "output_dir": args.output_dir}
    )
    print(f"QSVT amplitude-estimation integration audit complete: {run['output_dir']}")
    print(f"probe_status={run['probe'].get('status')}")


if __name__ == "__main__":
    main()
