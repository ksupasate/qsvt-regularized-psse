from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.codesigned_target_audit import (  # noqa: E402
    run_qsvt_codesigned_target_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit prior amplitude-recovery evidence before co-designing targets."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/qsvt_codesigned_target_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_qsvt_codesigned_target_audit(
        {"input_root": args.input_root, "output_dir": args.output_dir}
    )
    print(f"Wrote co-designed target audit to {run['output_dir']}")
    for row in run["rows"]:
        print(f"- {row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
