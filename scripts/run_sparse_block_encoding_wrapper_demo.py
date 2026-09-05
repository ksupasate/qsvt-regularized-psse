"""Goal B entry point: toy complete sparse block-encoding wrapper demo."""

from __future__ import annotations

import sys

from robust_qsvt_se.qsvt.sparse_block_encoding_wrapper import main

if __name__ == "__main__":
    main(sys.argv[1:])
