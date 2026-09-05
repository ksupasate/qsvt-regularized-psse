# Metadata Completion Checklist

Prepared: 2026-09-05
Updated: 2026-09-06 — author identity and version resolved.

| Metadata | Public value | Status |
|---|---|---|
| Authors | Supasate Vorathammathorn; Dhana Phassadawongse; Stephen John Turner | **resolved** |
| Author order | as listed above | **resolved** |
| ORCID (Vorathammathorn) | `https://orcid.org/0009-0009-2751-1023` | **resolved** |
| ORCID (Phassadawongse, Turner) | omitted | open — supply only author-verified ORCIDs |
| Repository owner | `ksupasate` | **resolved** |
| Repository URL | `https://github.com/ksupasate/qsvt-regularized-psse` | **resolved** |
| Released artifact version | `1.0.0` in `VERSION`, `pyproject.toml`, `CITATION.cff` | **resolved** |
| Package `__version__` | `0.1.0`, deliberately unchanged | **resolved by decision** |
| Release date | omitted | open — assigned when the author tags `v1.0.0` |
| Zenodo/data DOI | `10.5281/zenodo.22326883` | **resolved** (record still an unpublished draft) |
| Paper DOI | omitted | open — add if and when assigned |
| Git tag `v1.0.0` | not created | open — the author creates it |

## Why `__version__` stays at 0.1.0

`robust_qsvt_se.__version__` is stamped into the `package_version` field of 20
shipped evidence manifests and resolved configs. Raising it would desynchronize
those frozen records from the code that produced them, and would break the
byte-identical guarantee over `src/`. Raising it was verified not to break any
test (471 passed, 10 skipped), so it can be done deliberately in a future
release alongside regenerated evidence.

## Completion rule

The remaining open items are author decisions. ORCIDs must be copied from the
authoritative registry rather than guessed, and no release date may be recorded
before the release actually happens.
