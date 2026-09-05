# Final Privacy and Secret Scan

Prepared: 2026-09-05
Rescanned after the `outputs/` evidence whitelist was widened.
Scope: the exact set of **1556** files Git would track from the public candidate
(computed with the candidate's own `.gitignore`, verified against a throwaway
index so no `.git` was created inside the candidate during the scan). This
includes the 335 `outputs/` evidence files added during final verification.

Untracked-but-on-disk material (`.pytest_cache/`, `.ruff_cache/`,
`__pycache__/`, generated rerun outputs) is **not** part of the public surface
and was excluded from the scan scope for that reason; it was separately
confirmed to be ignored.

The 335 newly included evidence files introduced **no** new finding: zero
machine-local paths, zero hostnames, zero account names, zero emails, zero
secrets.

## Summary

| Category | Findings | Blocking |
|---|---:|:---:|
| High-confidence secrets (AWS keys, GitHub/Slack/Google tokens, JWTs, private keys) | 0 | — |
| Credential assignments (`password=`, `api_key=`, `token=`, …) | 0 | — |
| Key/cert/credential files (`*.pem`, `*.key`, `id_rsa*`, `.env`, `.netrc`, `.pypirc`, `.npmrc`) | 0 | — |
| Email addresses of any kind | 0 | — |
| `/Users/...` machine-local paths | 0 unexplained | no |
| `/home/...` paths | 0 | — |
| Windows user-profile paths (`C:\Users\`, `\Users\`) | 0 | — |
| Operating-system account name | 0 | — |
| Development-workspace absolute paths | 0 | — |
| Private repository URLs | 0 | — |
| Reviewer notes / rebuttal / "confidential" / "internal only" markers | 0 | — |
| Author-private notes | 0 | — |

**Result: 0 high-confidence secrets, 0 credentials, 0 unexplained concrete
machine-local paths.** No blocking finding.

## Patterns searched

Secrets: `AKIA…`/`ASIA…` (AWS), `ghp_`/`gho_`/`github_pat_` (GitHub),
`sk-…`/`sk-ant-…`, `xox[abposr]-…` (Slack), `AIza…` (Google),
`-----BEGIN … PRIVATE KEY-----`, `eyJ….….` (JWT).
Credentials: case-insensitive `api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|credential` followed by an assignment and a quoted value.
Paths: `/Users/`, `/home/`, `/private/tmp`, bare `/tmp/`, `C:\Users`, `\Users\`,
`Documents and Settings`, the OS account name, `VISTEC_Paper`, `QSVT_paper`.
Identity: RFC-shaped email addresses, `https?://` hosts.
Process leakage: `rebuttal`, `confidential`, `do not distribute`, `internal only`,
`referee report`, `reviewer said/asked/comment/response`.

## Explained matches (reviewed, not leaks)

### 1. `src/robust_qsvt_se/utils/privacy.py`

```python
_REPO_ROOT = re.compile(r"/Users/[^/\s\"',]+/(?:Desktop|Deskto)/VISTEC_Paper/QSVT_paper")
_HOME = re.compile(r"/Users/[^/\s\"',]+")
```

These are the **redaction patterns of the sanitizer itself**, not data. They
contain no account name — the user segment is a wildcard character class. Kept
deliberately: this module is what strips such paths from generated artifacts.

### 2. Generic `/tmp/...` literals (6 occurrences)

| Location | Literal | Nature |
|---|---|---|
| `tests/test_tqe_closed_loop_nonlinear_update.py:288,310,334` | `/tmp/does-not-matter`, `/tmp/cache-bound`, `/tmp/cache-mismatch` | inert test placeholders, never written to |
| `scripts/run_scientific_validation_suite.py:36` | `/tmp/_sci_suite_audit` | hardcoded scratch destination |
| `src/robust_qsvt_se/experiments/tqe_revision_evidence.py:1319` | `/tmp/tqe_revision_evidence_dry_run.json` | dry-run scratch destination |

These are generic POSIX temp paths with no account name and no machine identity.
They are a **portability** wart on Windows, not a privacy finding; recorded as a
non-blocking limitation rather than changed, because altering the destinations of
scientific entry points is out of release-hygiene scope.

### 3. `src/robust_qsvt_se/physical_alignment/reporting.py:430`

A recorded provenance string describing how a historical check was executed
(`"… (cwd: copy-on-write /private/tmp clone)"`). It is part of canonical
scientific evidence text, names no account and no machine, and was deliberately
**not** modified.

### 4. `release_audit/takeover_state.md`

The first draft of this session's takeover record contained the development
workspace's absolute path and the OS account name. **This was corrected during
the session**: workspace locations are now written home-relative (`~/…`) or
against a `<staging-root>` placeholder, and the file carries an explicit
"Path-recording convention" note. Re-scanned after the fix: zero matches for the
account name, `/Users/`, and `/private/tmp`.

### 5. This report's own pattern list

`release_audit/final_privacy_scan.md` matches several of the patterns it
documents, because it quotes them (`/Users/`, `/home/`, `/private/tmp`,
`rebuttal`, `confidential`, `internal only`). These are self-referential
descriptions of the search, not data.

### 6. The pytest evidence artifacts

`release_audit/test_logs/final_public_pytest.xml` is committed. pytest stamps
JUnit XML with the machine hostname, and the console log echoed the
interpreter's absolute path. Both were sanitized after the run: the hostname
became `redacted-local-workstation`, and home and staging paths became `<home>`
and `<repo-root>`. Only those identifiers changed; every test name, status,
duration, and total is verbatim. Re-scanned afterwards: zero matches for the
hostname, the account name, `/Users/`, and `/private/tmp`.

### 7. Public URLs (5 occurrences, all benign)

`https://github.com/ksupasate/qsvt-regularized-psse` (the project's own public
repository, in `CITATION.cff` and audit records), `https://keepachangelog.com`
(`CHANGELOG.md`), and `http://www.w3.org` (an SVG XML namespace in
`scripts/plot_figure1_weighted_jacobian.py`). No private host, no internal
service, no tracker link.

## Machine-local paths removed before this session

Five legacy smoke snapshots recorded the development checkout in their
`output.root` field. The previous session excluded those
`config_resolved.yaml` files from the export and replaced each with a
hash-bearing `config_resolved.omission.json` naming the source config, its
SHA-256, the source commit, and the exact regeneration command.
`scripts/validate_outputs.py` was extended to accept that record in place of the
snapshot. This was verified in this session:

- 5 omission records present under `outputs/examples/smoke_test/*/`.
- 0 files anywhere under `outputs/` contain `/Users/`.
- The validator accepts the omission records (see the validator run record).

## Author metadata

`CITATION.cff` now carries the author-supplied names, one author-verified ORCID,
and the real repository URL. This is intentional public author metadata, not a
secret, and is deliberately not flagged. No ORCID, email address, or affiliation
was invented; the two authors who supplied no ORCID simply have none recorded.
