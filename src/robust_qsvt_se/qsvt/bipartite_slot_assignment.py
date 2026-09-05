"""Deterministic bipartite slot assignment for sparse block-encoding wrappers.

Phase 10 replacement for the Phase 9 blocker: the reused Konig augmenting-path
edge coloring in ``sparse_block_encoding_wrapper.konig_slot_permutations`` did
not terminate on the 8x8 sparsified selected block.  That routine had two
independent defects, both fixed here:

1. it accepted an *infeasible* request without detecting it (the Phase 9
   pattern has maximum column degree 3, so no 2-slot coloring exists; Konig's
   theorem guarantees only ``max_degree`` colors), and
2. its alternating-path recoloring used symmetric-difference bookkeeping that
   desynchronizes when a vertex carries both swap colors, so the path walk can
   cycle forever.

This module uses a different, provably terminating construction:

* pad the nonzero pattern to a ``slots``-regular bipartite *multigraph* by
  adding dummy edge instances (dummy instances may be parallel to real edges;
  they carry value zero in the wrapper, so no entry is double counted),
* peel off ``slots`` perfect matchings with Kuhn's augmenting-path algorithm
  under an explicit visit budget (a d-regular bipartite multigraph always has
  a perfect matching, and removing one leaves a (d-1)-regular multigraph),
* record, per slot, the full column-to-row permutation and which pairs carry a
  real matrix entry.  Every real edge is consumed by exactly one slot.

Everything is deterministic: rows are processed in ascending order, adjacency
is sorted (real edges first, then by column), and padding pairs deficits in
ascending index order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SlotAssignment:
    """A decomposition of a bipartite pattern into full slot permutations.

    ``permutations[k][j] = i`` means slot ``k`` maps column ``j`` to row ``i``.
    ``real_edge_mask[k][j]`` is True when that pair carries a real pattern edge
    assigned to slot ``k`` (False pairs are zero-valued padding).
    """

    n: int
    slots: int
    max_degree: int
    permutations: tuple[tuple[int, ...], ...]
    real_edge_mask: tuple[tuple[bool, ...], ...]
    augmenting_visits: int
    visit_budget: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "slots": self.slots,
            "max_degree": self.max_degree,
            "permutations": [list(pi) for pi in self.permutations],
            "real_edge_mask": [list(mask) for mask in self.real_edge_mask],
            "augmenting_visits": self.augmenting_visits,
            "visit_budget": self.visit_budget,
            "algorithm": (
                "regularize to a slots-regular bipartite multigraph, then peel perfect "
                "matchings with Kuhn augmenting paths under an explicit visit budget"
            ),
        }


class _MatchingGraph:
    """Kuhn augmenting-path matcher with an explicit visit budget.

    Extracted to module scope so the recursive search binds no enclosing-loop
    variables; ``augment`` takes the per-color adjacency and matching explicitly.
    """

    __slots__ = ("budget", "edge_cols", "edge_real", "edge_rows", "visits")

    def __init__(
        self,
        edge_rows: list[int],
        edge_cols: list[int],
        edge_real: list[bool],
        budget: int,
    ) -> None:
        self.edge_rows = edge_rows
        self.edge_cols = edge_cols
        self.edge_real = edge_real
        self.budget = budget
        self.visits = 0

    def augment(
        self,
        row: int,
        adjacency: dict[int, list[int]],
        match_col: dict[int, int],
        seen: set[int],
        color: int,
    ) -> bool:
        for eid in adjacency.get(row, []):
            col = self.edge_cols[eid]
            if col in seen:
                continue
            seen.add(col)
            self.visits += 1
            if self.visits > self.budget:
                raise RuntimeError(
                    f"augmenting-path visit budget exceeded ({self.budget}) while "
                    f"matching color {color}; this indicates a logic error"
                )
            if col not in match_col or self.augment(
                self.edge_rows[match_col[col]], adjacency, match_col, seen, color
            ):
                match_col[col] = eid
                return True
        return False


def minimum_slot_count(pattern: np.ndarray) -> int:
    """Konig lower bound: the maximum row/column degree of the pattern."""

    pat = _as_pattern(pattern)
    if not pat.any():
        return 1
    return int(max(pat.sum(axis=1).max(), pat.sum(axis=0).max()))


def assign_slot_permutations(
    pattern: np.ndarray,
    *,
    slots: int | None = None,
    max_augment_visits: int | None = None,
) -> SlotAssignment:
    """Decompose ``pattern`` into ``slots`` full permutations deterministically.

    Raises ``ValueError`` immediately (instead of looping) when ``slots`` is
    below the Konig minimum, and ``RuntimeError`` with diagnostics if the visit
    budget is ever exceeded (which would indicate a logic error, not an input
    property; the budget makes non-termination impossible by construction).
    """

    pat = _as_pattern(pattern)
    n = pat.shape[0]
    required = minimum_slot_count(pat)
    slot_count = required if slots is None else int(slots)
    if slot_count < required:
        raise ValueError(
            f"slots={slot_count} is infeasible: the pattern has maximum row/column "
            f"degree {required}, and Konig's theorem needs at least that many slots"
        )

    # Edge instances: real pattern edges first, then dummy padding instances
    # pairing row and column degree deficits in ascending index order.
    edge_rows: list[int] = []
    edge_cols: list[int] = []
    edge_real: list[bool] = []
    for i, j in zip(*np.nonzero(pat), strict=True):
        edge_rows.append(int(i))
        edge_cols.append(int(j))
        edge_real.append(True)
    row_deficit = slot_count - pat.sum(axis=1)
    col_deficit = slot_count - pat.sum(axis=0)
    deficit_rows = [i for i in range(n) for _ in range(int(row_deficit[i]))]
    deficit_cols = [j for j in range(n) for _ in range(int(col_deficit[j]))]
    if len(deficit_rows) != len(deficit_cols):  # pragma: no cover - defensive
        raise RuntimeError("row and column degree deficits disagree")
    for i, j in zip(deficit_rows, deficit_cols, strict=True):
        edge_rows.append(i)
        edge_cols.append(j)
        edge_real.append(False)

    budget = 4 * slot_count * n * n + 128 if max_augment_visits is None else int(max_augment_visits)
    visits = 0
    remaining: set[int] = set(range(len(edge_rows)))
    permutations: list[tuple[int, ...]] = []
    real_masks: list[tuple[bool, ...]] = []

    graph = _MatchingGraph(edge_rows, edge_cols, edge_real, budget)
    for color in range(slot_count):
        adjacency: dict[int, list[int]] = {}
        for eid in sorted(remaining):
            adjacency.setdefault(edge_rows[eid], []).append(eid)
        for eids in adjacency.values():
            eids.sort(key=lambda e: (not edge_real[e], edge_cols[e], e))
        match_col: dict[int, int] = {}

        for row in range(n):
            if not graph.augment(row, adjacency, match_col, set(), color):
                raise RuntimeError(
                    f"no perfect matching found for color {color}; a "
                    f"{slot_count - color}-regular bipartite multigraph always has one, "
                    "so this indicates a logic error"
                )
        visits = graph.visits
        pi = np.full(n, -1, dtype=np.int64)
        mask = np.zeros(n, dtype=bool)
        for col, eid in match_col.items():
            pi[col] = edge_rows[eid]
            mask[col] = edge_real[eid]
            remaining.discard(eid)
        if sorted(pi.tolist()) != list(range(n)):  # pragma: no cover - defensive
            raise RuntimeError(f"slot {color} map is not a permutation: {pi.tolist()}")
        permutations.append(tuple(int(v) for v in pi))
        real_masks.append(tuple(bool(v) for v in mask))

    if remaining:  # pragma: no cover - defensive
        raise RuntimeError(f"{len(remaining)} edge instances left unassigned")

    assignment = SlotAssignment(
        n=n,
        slots=slot_count,
        max_degree=required,
        permutations=tuple(permutations),
        real_edge_mask=tuple(real_masks),
        augmenting_visits=visits,
        visit_budget=budget,
    )
    validate_slot_assignment(pat, assignment)
    return assignment


def validate_slot_assignment(pattern: np.ndarray, assignment: SlotAssignment) -> dict[str, Any]:
    """Check permutation validity and exactly-once real-edge coverage."""

    pat = _as_pattern(pattern)
    n = pat.shape[0]
    if assignment.n != n:
        raise ValueError("assignment dimension does not match pattern")
    coverage = np.zeros((n, n), dtype=np.int64)
    for slot, (pi, mask) in enumerate(
        zip(assignment.permutations, assignment.real_edge_mask, strict=True)
    ):
        if sorted(pi) != list(range(n)):
            raise ValueError(f"slot {slot} map is not a permutation")
        for col in range(n):
            if mask[col]:
                coverage[pi[col], col] += 1
    if not np.array_equal(coverage > 0, pat):
        raise ValueError("real-edge coverage does not match the pattern")
    if int(coverage.max(initial=0)) > 1:
        raise ValueError("a real edge is assigned to more than one slot")
    return {
        "valid": True,
        "n": n,
        "slots": assignment.slots,
        "real_edges_covered_exactly_once": True,
        "per_slot_real_edge_counts": [int(sum(mask)) for mask in assignment.real_edge_mask],
        "augmenting_visits": assignment.augmenting_visits,
        "visit_budget": assignment.visit_budget,
    }


def _as_pattern(pattern: np.ndarray) -> np.ndarray:
    pat = np.asarray(pattern, dtype=bool)
    if pat.ndim != 2 or pat.shape[0] != pat.shape[1]:
        raise ValueError("pattern must be a square boolean matrix")
    if pat.shape[0] == 0:
        raise ValueError("pattern must be nonempty")
    return pat
