#!/usr/bin/env python
"""Run Experiment C: fixed-case end-to-end selected-observable resource ledger."""

from __future__ import annotations

import sys

from robust_qsvt_se.paper.tqe_revision_resource_ledger import main

if __name__ == "__main__":
    main(sys.argv[1:])
