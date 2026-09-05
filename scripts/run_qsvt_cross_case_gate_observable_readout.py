from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.cross_case_gate_observable_readout import (  # noqa: E402
    DEFAULT_OBSERVABLES,
    DEFAULT_SHOTS,
    run_qsvt_cross_case_gate_observable_readout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-case gate-level observable readout for surviving IEEE30/57 candidates."
    )
    parser.add_argument(
        "--input", default="outputs/qsvt_cross_case_gate_validation/cross_case_gate_results.csv"
    )
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--observables", nargs="+", default=list(DEFAULT_OBSERVABLES))
    parser.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_cross_case_gate_observable_readout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input": args.input,
        "model": args.model,
        "case_source": args.case_source,
        "observables": args.observables,
        "shots": args.shots,
        "topk": args.topk,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_cross_case_gate_observable_readout(config)
    rows = run["rows"]
    cases = sorted({str(row.get("case")) for row in rows})
    print(f"Wrote cross-case gate observable readout to {run['output_dir']}")
    print(f"Observable rows: {len(rows)}; cases: {cases or 'none'}")


if __name__ == "__main__":
    main()
