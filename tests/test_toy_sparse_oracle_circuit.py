from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.toy_sparse_oracle_circuit import (
    TOY_SPARSE_ORACLE_LIMITATION,
    build_toy_sparse_oracle_circuit_demo,
)


def test_toy_sparse_oracle_circuit_demo_outputs_summary(tmp_path: Path) -> None:
    run = build_toy_sparse_oracle_circuit_demo({"output_dir": str(tmp_path), "degree": 5})
    frame = pd.read_csv(run["artifacts"]["toy_sparse_oracle_summary"])

    assert frame.loc[0, "matrix_size"] == "4x4"
    assert frame.loc[0, "qsvt_query_count_proxy"] == 11
    assert frame.loc[0, "top_left_block_error"] < 1.0e-12
    assert TOY_SPARSE_ORACLE_LIMITATION in frame.loc[0, "limitation"]
