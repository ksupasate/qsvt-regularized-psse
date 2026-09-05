# Reproducibility

The detailed protocol is [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Fast path

```bash
conda env create -f environment.yml
conda activate qsvt-se
bash scripts/reproduce_all.sh
```

The entry point checks the environment, runs artifact validation, and generates
a deterministic IEEE14 example under `outputs/examples/`. It does not run
expensive scientific campaigns or overwrite canonical evidence.

## Individual checks

```bash
python scripts/run_smoke_test.py
python scripts/validate_reproduction.py
python scripts/validate_outputs.py
```

## Artifact map

- Environment and determinism: `docs/REPRODUCIBILITY.md`
- Installation and first run: `docs/QUICKSTART.md`
- Data-generation semantics: `docs/MEASUREMENT_MODEL.md`
- Scientific interpretation limits: `docs/CLAIM_SCOPE.md`
- Large-file access policy: `docs/data_access.md`
- Experiment registry: `outputs/reproducibility_audit/experiment_manifest.json`
- Output index: `RESULTS_INDEX.md`
- Clean-repository validation: `release_audit/final_clean_repository_validation.md`
- Public test scope: `release_audit/public_test_scope_report.md`
- Release decision: `release_audit/github_public_release_final.md`

New outputs belong under `outputs/examples/` or `outputs/generated/`. Do not
rewrite frozen scientific outputs, QSVT phase data, residual banks, support
registries, or evidence registries during ordinary reproduction.
