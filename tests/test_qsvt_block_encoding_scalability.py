from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.block_encoding_scalability import (
    DENSE_BLOCK_ENCODING_CAVEAT,
    build_block_encoding_scalability_report,
)


def test_block_encoding_scalability_report_includes_dense_caveat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_block_encoding_scalability_report(
        {
            "output_dir": str(tmp_path / "scalability"),
            "cases": ["synthetic"],
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "scalability_summary.csv")

    required = {
        "case_name",
        "m",
        "n",
        "nonzeros",
        "density",
        "estimated_dense_encoding_dimension",
        "estimated_index_qubits",
        "scalability_caveat",
    }
    assert required.issubset(frame.columns)
    assert frame.loc[0, "status"] == "ok"
    assert DENSE_BLOCK_ENCODING_CAVEAT in frame.loc[0, "scalability_caveat"]
    assert int(frame.loc[0, "estimated_dense_encoding_dimension"]) == int(frame.loc[0, "m"]) + int(
        frame.loc[0, "n"]
    )
    assert (output_dir / "scalability_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
