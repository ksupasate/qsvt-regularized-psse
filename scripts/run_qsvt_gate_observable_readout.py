from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.gate_observable_readout import (  # noqa: E402
    DEFAULT_OBSERVABLES,
    run_qsvt_gate_observable_readout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate-level observable-first readout for co-designed QSVT targets."
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument(
        "--input",
        default="outputs/qsvt_degree_window_gate_validation/degree_window_gate_results.csv",
    )
    parser.add_argument("--observables", nargs="+", default=list(DEFAULT_OBSERVABLES))
    parser.add_argument("--shots", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--max-configs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_gate_observable_readout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "case": args.case,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "input": args.input,
        "observables": args.observables,
        "shots": args.shots,
        "topk": args.topk,
        "max_configs": args.max_configs,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_gate_observable_readout(config)
    rows = run["rows"]
    confirmed = len({row["observable_name"] for row in rows})
    print(f"Wrote gate observable readout to {run['output_dir']}")
    print(f"Observable rows: {len(rows)}; distinct observables gate-confirmed: {confirmed}")


if __name__ == "__main__":
    main()
