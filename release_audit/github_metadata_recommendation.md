# GitHub Repository Metadata Recommendation

Prepared: 2026-09-06

These are recommendations for the author to apply in the GitHub web UI after
creating the repository. **No GitHub setting was changed through the API or CLI
during this task.**

## Repository

| Field | Value |
|---|---|
| Name | `qsvt-regularized-psse` |
| Owner | `ksupasate` |
| URL | `https://github.com/ksupasate/qsvt-regularized-psse` |
| Visibility | public |
| Default branch | `main` |
| License | MIT (detected automatically from `LICENSE`) |

## Description

```text
Reproducible implementation and benchmark evaluation of QSVT-compatible
regularized spectral filtering for power-system state estimation.
```

## Topics

```text
qsvt
qsp
quantum-computing
power-system-state-estimation
power-systems
regularization
tikhonov-regularization
spectral-filtering
numerical-linear-algebra
reproducible-research
```

## Topics deliberately NOT recommended

| Rejected topic | Reason |
|---|---|
| `quantum-speedup` | The repository explicitly claims no speedup. |
| `quantum-advantage` | Same — the claim boundary forbids it. |
| `real-pmu` | No PMU field data are used; measurements are code-generated. |
| `real-scada` | No SCADA field data are used. |

Applying any of these would contradict the claim boundaries stated in
`README.md`, `docs/CLAIM_SCOPE.md`, and `CITATION.cff`.

## Suggested repository settings

- **Citation:** GitHub reads `CITATION.cff` automatically and will show a
  "Cite this repository" button. No action needed.
- **Releases:** create the `v1.0.0` tag and release only when the author is
  ready. Neither was created by this preparation.
- **Zenodo integration:** a DOI is already reserved
  (`10.5281/zenodo.22326883`) against an unpublished draft. If the GitHub–Zenodo
  webhook is enabled instead, it would mint a *different* DOI on the first
  release — decide which path to use before tagging, to avoid two identifiers
  for the same artifact.
- **Social preview / homepage:** optional; nothing is configured.
- **Branch protection:** optional for a single-maintainer artifact repository.

## About-panel summary

```text
Reproducible implementation and benchmark evaluation of QSVT-compatible
regularized spectral filtering for power-system state estimation.
IEEE/PYPOWER benchmark models; code-generated measurements; QSVT studied as an
implementation pathway; no quantum-speedup claim.
```
