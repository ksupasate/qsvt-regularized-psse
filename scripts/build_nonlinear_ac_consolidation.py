from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.nonlinear_ac_consolidation import (  # noqa: E402
    build_nonlinear_ac_consolidation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: consolidate nonlinear AC benchmark runs into paper-level tables."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/phase4_nonlinear_ac",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_nonlinear_ac_consolidation(
        {
            "input_root": args.input_root,
            "config_root": args.config_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote Phase 4 nonlinear AC consolidation to {run['output_dir']}")
    print(
        f"cases={run['cases_covered']} convergence_rows={len(run['convergence_rows'])} "
        f"comparison_rows={len(run['comparison_rows'])} bad_data_rows={len(run['bad_data_rows'])}"
    )


if __name__ == "__main__":
    main()
