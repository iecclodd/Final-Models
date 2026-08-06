# Recreated LOSO compact models

This directory contains the 1,134 fitted compact classifiers from the
**recreated recovery cohort**: LDA, MLP, and RBF-SVM models for 21 feature
types and 18 leave-one-subject-out folds. They are controlled recreations,
not the unavailable original historical fitted objects.

## Provenance

- Locked source commit: `e568d39a07873e52f049a827cfb65686fb308893`
- Dataset SHA-256: `8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719`
- MATLAB release: R2026a
- Families: 378 LDA, 378 MLP, and 378 RBF-SVM artifacts
- Layout: `<family>/feature_XX/<family>_feature_XX_fold_XX_compact_model.mat`

The `published_model_manifest.csv` file binds every published artifact to
both its published SHA-256 and the validated source-model SHA-256.

## Privacy transformation

Each file preserves the fitted `compactModel` object but replaces the local
recovery metadata with a publication-safe record. Local filesystem paths,
training-subject lists, held-out predictions, confusion matrices, row hashes,
and fold metrics are removed. The exporter reloads every file and verifies a
deterministic prediction/score probe before hashing it.

These classifiers remain derived from participant data. In particular,
kernel SVM models may retain support vectors. The repository is private and
must not be made public without a separate data-owner and privacy review.

## Download and use

The MAT files are stored with Git LFS. Install Git LFS before cloning or run:

```powershell
git lfs install
git lfs pull
```

Load a model in MATLAB with:

```matlab
payload = load(modelFile,"compactModel","publishedModelMetadata");
[labels,scores] = predict(payload.compactModel,X);
```

Predictors must match the locked 128-channel feature representation recorded
in `publishedModelMetadata.PredictorIndices`.
