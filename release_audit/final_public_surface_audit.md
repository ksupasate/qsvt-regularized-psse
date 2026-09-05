# Final Public Surface Audit

Prepared: 2026-09-06
Repository: `qsvt-regularized-psse` (permanent location, staged for the
author's initial commit)

## Scope

| Surface | Status | Evidence |
|---|---|---|
| Manuscript sources, PDFs, LaTeX build products | **excluded** | 0 hits for `manuscript*`, `*.tex`, `*.pdf`, `*.aux`, `*.bbl`, `*.blg`, `*.fls`, `*.toc`, `*.synctex.gz` |
| Supplementary material | **excluded** | 0 hits for `supplementary*` |
| Submission packages and archives | **excluded** | 0 hits for `submission_package*`; 0 archives of any kind (`*.zip`, `*.tar`, `*.tgz`, `*.7z`, `*.rar`) |
| Manuscript-correction workspaces | **excluded** | 0 hits for `final_manuscript_correction*` |
| Internal AI/agent prompts and transcripts | **excluded** | see the AI section below |
| Scientific code, configs, tests, evidence | **retained** | 1562 staged files; `src/` and `configs/` byte-identical to the audited source |

### Reviewer-named scientific evidence — retained deliberately

37 staged paths contain the word *reviewer*:
`configs/reviewer_blocking_tqe_evidence/` (5), `configs/tqe_reviewer_blocking/`
(7), `scripts/run_reviewer_blocking_*.py` (4),
`src/robust_qsvt_se/reviewer_blocking/` (7),
`src/robust_qsvt_se/reviewer_evidence/` (8), `tests/test_reviewer_*.py` (6).

These are **scientific implementation and evidence** named after the review
round that motivated them — physical functionals, joint feasibility, resource
Pareto fronts, exact-loss and task-aware baselines, high-degree studies.
Verified:

- **Scientific purpose:** module docstrings describe deterministic functionals,
  QSVT feasibility questions, and baseline comparisons.
- **No reviewer correspondence:** 0 hits for `referee`, `rebuttal`,
  `reviewer said/wrote/asked/commented`, `response to reviewer`,
  `dear editor/reviewer`, `we thank the reviewer`.
- **No private comments.**
- **Test dependency:** the six dedicated test files pass (54 passed).

The reviewer *delivery* scripts (`run_reviewer_evidence_all.py`,
`build_reviewer_blocking_manifest.py`, `reviewer_evidence_hash_snapshot.py`,
`run_tqe_reviewer_revision_audit.py`) remain excluded and are not staged.

## AI / Internal Workflow

Full detail in `release_audit/ai_provenance_surface_audit.md`.

| Scan | Result |
|---|---|
| Filenames (`AGENT`, `CLAUDE`, `COPILOT`, `PROMPT`, `CHATGPT`, `CODEX`, `GEMINI`, `assistant`, `handoff`, `conversation`, `transcript`) | **0** |
| Directories (`.claude`, `.cursor`, `.copilot`, `.codex`, `.agents`, `chatgpt`, `codex`, `agent`, `assistant`) | **0** |
| Root dotfiles | 1 — `.gitignore` only |
| Content scan (19 AI/assistant terms, 1608 files) | 6 matches in 4 files |
| Prompt-language scan (15 phrases) | 2 matches, both false positives |
| Source-comment scan (1073 files in `src/`, `scripts/`, `tests/`, `docs/`, `examples/`) | **0** |

**Internal prompts found:** 0.
**Conversation transcripts found:** 0.
**Handoff files found:** 0.
**Files removed:** 0.
**Files rewritten:** 1 — `scripts/continue_final_qsvt_feasibility_push.py`, whose
work-package status table listed WP26's evidence as `"final assistant response"`.
Replaced with `"continuation summary reported to the maintainer"`. No numerical
or scientific behavior changed; the row feeds a non-shipped Markdown table and
no test references it.

**Legitimate matches retained:** the `.gitignore` rule `/.claude/`, the
exclusion-manifest rules `.claude/**` and `.codex/**`, the inventory row
recording those artifacts as absent, and two `Definition of done`
acceptance-criterion labels in scientific reporting code.

Verified conclusion:

> No internal AI-assistant prompts, conversation transcripts, agent handoff
> files, or assistant-specific workflow instructions remain in the intended
> public release surface.

This is a statement about artifacts present in the repository. It is **not** a
claim that no AI tooling was used during development, and no such claim is made.

## Privacy

| Category | Findings | Blocking |
|---|---:|:---:|
| High-confidence secrets (AWS/GitHub/Slack/Google keys, JWTs, SSH and RSA private keys) | **0** | — |
| Credential assignments | **0** | — |
| URLs with embedded credentials | **0** | — |
| Email addresses | **0** | — |
| OS account name in any *filesystem path* | **0** | — |
| Machine hostname | **0** | — |
| `/home/<user>` paths | **0** | — |
| Windows profile paths | **0** | — |
| Key/credential files (`*.pem`, `*.key`, `id_rsa*`, `.env`, `.netrc`, `.pypirc`) | **0** | — |

**No credential was found, so no credential is quoted anywhere in these reports.**

Two match classes are deliberately retained as intentional public metadata:

- **`ksupasate`** appears as the **GitHub owner** in the repository URL
  (`CITATION.cff`, `README.md`, `RELEASE.md`, and audit records). The author's
  GitHub handle happens to coincide with the local account name, but it is
  published here as the repository owner, not leaked as a filesystem path. No
  home-directory path containing the account name exists anywhere in the staged
  tree; the only two files matching a `/Users/` pattern are this report and
  `final_privacy_scan.md`, which quote the pattern, plus the sanitizer module
  whose regex uses a wildcard for the account segment.
- **`git@github.com`** in the SSH remote URL matches an email-shaped pattern but
  is a Git transport address, not a contact address.

Explained path matches, all reviewed and retained:

- `src/robust_qsvt_se/utils/privacy.py` — the redaction module's own regexes.
  Its `/Users/` pattern is a wildcard character class, not an account name.
- `src/robust_qsvt_se/physical_alignment/reporting.py` — a frozen provenance
  string describing a historical run.
- Seven frozen `outputs/*/manifest.json` provenance `command` fields, already
  redacted by the repository's own sanitizer to `<home>/…`. They retain the
  development directory layout but no account name or hostname. Modifying them
  would alter frozen evidence provenance and is deliberately not done.
- Three generic `/tmp/...` scratch destinations in non-entry-point code.
- `release_audit/*.md` records that quote the search patterns themselves.

Audit records use `<workspace>` and `<system-temp>` placeholders rather than
concrete local paths.

## Scientific Preservation

Full detail in `release_audit/final_precommit_scientific_preservation.md`.

| Surface | Result |
|---|---|
| `src/**` | **354 / 354 byte-identical** to the curated source |
| `configs/**` | **109 / 109 byte-identical** |
| `examples/**` | **6 / 6 byte-identical** |
| `outputs/**` | 336 / 343 identical; the 7 differences are 5 privacy substitutes and 2 prose edits |
| Algorithms changed | **none** |
| Configurations changed | **none** |
| Results changed | **none** |
| Test-generated drift in staged files | **none** (`git_commit: null`: 0 occurrences) |

## Validation

| Check | Result |
|---|---|
| Reproduction validator | 144 checks, 6 explained warnings, **0 failures** |
| IEEE14 one-command quickstart | **PASS**, 8 artifacts, 17.1 s |
| Ridge/QSVT equivalence guard | **8 passed** |
| Reviewer-named subpackage tests | **54 passed** |
| Manifest/provenance tests | **471 passed, 10 skipped** |
| Release-facing Ruff | **all checks passed** |
| Ruff pyflakes repo-wide | **all checks passed** |
| `compileall` (`src`, `scripts`, `tests`) | clean |
| Package import | **350 / 350 modules** |
| CITATION.cff validation | **PASS** |
| Markdown local links | 22 checked, **0 broken** |
| YAML / JSON parsing | 100 YAML + 145 JSON, **0 failures** |

## Release Readiness

**PASS.**

No blocking condition from the STOP list is present: no credential, no
unexplained scientific modification, no old history, no file over 100 MB, no
manuscript or submission material, no unresolved AI artifact, no DOI mismatch,
the remote points only at `ksupasate/qsvt-regularized-psse`, and the validator
reports zero scientific failures.
