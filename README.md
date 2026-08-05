# Final Models

MATLAB implementations and validated aggregate results for four single-feature EMG model families:

- Random Forest (existing repository model)
- Linear Discriminant Analysis (LDA)
- Multilayer Perceptron (MLP)
- Radial Basis Function Support Vector Machine (RBF-SVM)

The LDA, MLP, and RBF-SVM experiments each completed all 378 leave-one-subject-out folds: 21 feature types by 18 held-out subjects. That is 1,134 completed and validated folds across the three added models.

## Run the three-model pipeline

Place the locally authorized `capgmyo_ml_dataset.mat` beside the scripts. Its SHA-256 must match the value in `single_feature_loso_locked_config.json`. Then run:

```matlab
run_single_feature_loso_pipeline
```

The locked order is LDA, MLP, then RBF-SVM. Smoke-test and runtime-estimation entry points are also included. MATLAB R2026a was used for the recorded experiment; Statistics and Machine Learning Toolbox is required, Deep Learning Toolbox is required for MLP, and Parallel Computing Toolbox enables the tested GPU paths.

## Published results

`results/` contains privacy-safe aggregate rankings, timings, plots, and cross-model rank correlations. See `docs/model-results.md` for the completion evidence and scientific limitations.

The repository intentionally excludes:

- the raw participant dataset;
- subject-level tables and confusion matrices;
- checkpoint, backup, and diary files;
- MAT result files containing local machine paths;
- deployable fitted classifier objects, which were not persisted by the LOSO evaluation pipeline.

The scripts retrain a classifier inside each LOSO fold. The published artifacts therefore document a complete cross-user evaluation, not one classifier refitted on 100% of the data for deployment.

This repository must remain private unless the data owner separately approves public release after privacy, licensing, and provenance review.
