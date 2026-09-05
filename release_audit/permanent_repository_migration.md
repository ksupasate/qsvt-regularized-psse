# Permanent Repository Migration

Prepared: 2026-09-06

The verified clean release candidate was moved off ephemeral storage into its
permanent location before the author's initial commit.

## Paths

| Role | Path |
|---|---|
| Source candidate (ephemeral) | `<system-temp>/qsvt-regularized-psse-public-stage` |
| Permanent repository | `<workspace>/qsvt-regularized-psse` |

The source candidate was **not** deleted. All subsequent release work continues
from the permanent repository.

## Method

```bash
mkdir -p <permanent-repository>
rsync -a --exclude='.git' <source-candidate>/ <permanent-repository>/
```

`.git` was excluded deliberately. The candidate's index (1556 staged files, zero
commits) is not carried over; a fresh repository is initialized at the permanent
location instead, so the public repository still has no inherited history from
any source.

## Pre-migration cleanup

Two regenerable caches were removed from the candidate **before** copying, so
that source and destination compare exactly:

- `.pytest_cache/`
- every `__pycache__/` directory

Both are `.gitignore`d and neither was ever part of the staged set.

## Verification

| Check | Source | Destination | Result |
|---|---:|---:|:---:|
| Files (excluding `.git`) | 1604 | 1604 | equal |
| Total bytes | 120,164,531 | 120,164,531 | equal |
| Files only in source | — | — | **0** |
| Files only in destination | — | — | **0** |
| SHA-256 content differences | — | — | **0** |
| `.git` present in destination | — | no | correct |

Every one of the 1604 files was hashed on both sides.

**Result: 0 unexplained differences. The migration is byte-for-byte exact.**

## Note on file counts

The permanent repository holds 1604 files on disk, of which **1556** are
intended for the initial commit. The remaining 48 are `.gitignore`d local
evidence retained for offline work: eight `outputs/` roots the public checks do
not read, and `release_audit/test_logs/final_public_pytest.log` (excluded by the
`*.log` rule). They are present on disk only and cannot reach the commit.

## Path-recording convention

Concrete machine-local paths are deliberately not written into this public
record. `<workspace>` denotes the author's local project directory and
`<system-temp>` the platform temporary directory. This keeps the provenance
meaningful without publishing the local filesystem layout.

## Unrelated directory observed

A sibling directory `qsvt-regularized-psse-public` exists and is **empty**
(0 files, created 2026-09-05 11:08). It is a leftover from an earlier aborted
attempt, is unrelated to this release, and was left untouched.
