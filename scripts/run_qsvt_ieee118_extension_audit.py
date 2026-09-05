from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.ieee118_extension_audit import (  # noqa: E402
    IEEE118_SELECTION_MODES,
    run_qsvt_ieee118_extension_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit IEEE118 readiness for the selected-subproblem QSVT extension."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--case", default="ieee118")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--selection-modes", nargs="+", default=list(IEEE118_SELECTION_MODES))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_ieee118_extension_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input_root": args.input_root,
        "case": args.case,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "selection_modes": args.selection_modes,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_ieee118_extension_audit(config)
    status = run["status"]
    print(f"Wrote IEEE118 extension audit to {run['output_dir']}")
    print(
        f"loads={status['loads']}; supports_4x4={status['supports_4x4_extraction']}; "
        f"positive_evidence_modes={status['positive_evidence_modes']}"
    )


if __name__ == "__main__":
    main()
