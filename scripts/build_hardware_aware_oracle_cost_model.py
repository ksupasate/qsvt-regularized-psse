from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.hardware_aware_oracle_cost_model import (  # noqa: E402
    build_hardware_aware_oracle_cost_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hardware-aware sparse-oracle cost model")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
    )
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--value-precision-bits", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51, 101])
    parser.add_argument("--observable-readout-shots", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--index-lookup-model", default="table_lookup_or_qrom_proxy")
    parser.add_argument("--value-loading-model", default="fixed_point_value_register")
    parser.add_argument("--rotation-synthesis-model", default="clifford_t_proxy")
    parser.add_argument("--state-preparation-model", default="sparse_residual_loading")
    parser.add_argument("--disable-amplification", action="store_true")
    parser.add_argument("--output-dir", default="outputs/hardware_aware_oracle_cost_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_hardware_aware_oracle_cost_model(
        {
            "cases": args.cases,
            "case_source": args.case_source,
            "value_precision_bits": args.value_precision_bits,
            "degrees": args.degrees,
            "observable_readout_shots": args.observable_readout_shots,
            "index_lookup_model": args.index_lookup_model,
            "value_loading_model": args.value_loading_model,
            "rotation_synthesis_model": args.rotation_synthesis_model,
            "state_preparation_model": args.state_preparation_model,
            "amplitude_amplification_enabled": not args.disable_amplification,
            "output_dir": args.output_dir,
        }
    )
    print(f"Hardware-aware oracle cost model complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])}")


if __name__ == "__main__":
    main()
