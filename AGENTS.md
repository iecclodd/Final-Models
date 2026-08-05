# Repository guidance

## Purpose and stack

This repository contains MATLAB single-feature EMG classification pipelines and publication-safe aggregate evaluation artifacts. The added LDA, MLP, and RBF-SVM models use MATLAB R2026a APIs from Statistics and Machine Learning Toolbox, Deep Learning Toolbox, and optionally Parallel Computing Toolbox.

## Important paths

- Root `run_single_feature_*.m`: public model and orchestration entry points.
- `run_single_feature_loso_model.m`: shared LOSO engine; changes affect all three added models.
- `single_feature_loso_locked_config.json`: immutable dataset identity, dimensions, seeds, and scientific hyperparameters.
- `results/<model>/`: aggregate publication outputs only.
- `results/comparison/`: aggregate cross-model comparisons.
- `tests/`: MATLAB unit tests.
- `docs/`: experiment interpretation and follow-up protocols.

## Commands

Run from MATLAB with this repository as the current folder:

```matlab
results = runtests('tests');
assertSuccess(results);
run_all_single_feature_loso_smoke_tests;
run_single_feature_loso_pipeline;
compare_single_feature_loso_models;
```

For static analysis, run `checkcode` on every root `.m` file and test file. Full model execution is expensive and requires the exact locked dataset; use unit tests and isolated smoke tests before a full run.

## Data and artifact policy

- Never commit `capgmyo_ml_dataset.mat`, raw participant data, participant identifiers, or derived participant-level tables without documented rights and privacy review.
- Never commit progress checkpoints, timestamped backups, diary logs, smoke/runtime directories, or local absolute paths.
- Keep aggregate rankings, timings, and plots reproducible from the locked configuration.
- Treat `.mat` files as excluded by default. A deliberately published model binary requires a privacy scan, Git LFS decision, provenance, and an explicit pathspec.
- Do not modify locked scientific hyperparameters without versioning the configuration and invalidating incompatible checkpoints.

## Security and privacy

Scan proposed releases for credentials, personal identifiers, absolute paths, raw dataset leakage, and unexpectedly large binaries. Preserve the private-repository setting unless the data owner explicitly approves a public release. Do not expose configured credentials.

## Agent and thread guidance

Use Luna Low for inventories and exact artifact audits, Terra Medium for MATLAB implementation and test work, and Sol High for scientific, privacy, or release review. Keep agent depth at one and assign one writer per file. Reuse a shared repository map rather than duplicating broad scans.

## Definition of done

A change is complete when scoped files are reviewed, relevant MATLAB checks pass, aggregate outputs remain internally consistent, no excluded artifact is staged, documentation matches behavior, and the final Git diff and pushed commit are verified.

## Known risks

Upstream preprocessing provenance is incomplete. GPU numerical behavior may vary by MATLAB and hardware version. The LOSO outputs are evaluation artifacts, not deployable classifiers refitted on the complete dataset.
