import numpy as np

from robust_qsvt_se.paper.phase8_bridge_characterization import _coupling_diagnostics
from robust_qsvt_se.paper.phase9_bridge_leakage_aware import leakage_aware_block


def test_leakage_aware_selection_retains_target_and_reduces_column_leakage():
    H = np.zeros((8, 6))
    H[:4, 2] = [9.0, 8.0, 7.0, 6.0]
    H[4:, 2] = 0.1
    H[:, 0] = 1.0
    r = np.arange(8.0)
    block, block_r, rows, cols = leakage_aware_block(H, r, 4)
    assert block.shape == (4, 4)
    assert block_r.shape == (4,)
    assert cols[0] == 2
    diagnostics = _coupling_diagnostics(H, r, rows, cols)
    assert diagnostics["functional_column_leakage"] < 1e-3
