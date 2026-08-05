# Three-model LOSO results

## Completion

| Model | Compute used | Completed folds | Best feature | Mean balanced accuracy | Total training time | Total prediction time |
|---|---|---:|---|---:|---:|---:|
| LDA | CPU `fitcdiscr` | 378/378 | Zero Crossings | 0.241594 | 24.306 s | 1.410 s |
| MLP | GPU `fitcnet` | 378/378 | Waveform Length | 0.303801 | 804.666 s | 0.875 s |
| RBF-SVM | GPU `fitcecoc` with Gaussian `templateSVM` | 378/378 | Waveform Length | 0.328472 | 3,451.474 s | 46.734 s |

All 1,134 folds completed. Every fold evaluated 1,520 observations across eight classes. The stored confusion counts, per-class recall, balanced accuracy, and ordinary accuracy were reconciled during final validation.

## Aggregate feature findings

- LDA top five: Zero Crossings, Median Frequency, Average Rectified Value, Mean Frequency, and Waveform Length.
- MLP top five: Waveform Length, Zero Crossings, Root Mean Square, Spectral Entropy, and Standard Deviation.
- RBF-SVM top five: Waveform Length, Standard Deviation, Peak-to-Peak, Spectral Entropy, and Minimum.
- Consensus top five: Waveform Length, Zero Crossings, Standard Deviation, Spectral Entropy, and Average Rectified Value.

Pairwise Spearman rank correlations across the 21 feature types were 0.558442 for LDA/MLP, 0.601299 for LDA/RBF-SVM, and 0.883117 for MLP/RBF-SVM.

## Validation provenance

The completed local final-result MAT files were validated before publication. Their SHA-256 values were:

- LDA: `F42A6DCC966E78C49EBA6FE01DB9ED6294FC822449BA53F576289243C570F3B8`
- MLP: `DB061D36ECC0A21984838680D35E3C1CCBDF4EB933D69525FD07FB8DAB56EE47`
- RBF-SVM: `E79D6E17CF33DA6FE02217FD75DF6C67C0C23C67BDD2463DA98E89B068B615BF`

Those MAT files are not committed because they contain local machine paths and subject-level fold results. The aggregate CSV and PNG exports under `results/` are the publication-safe subset.

## Scientific scope

These rankings describe standalone cross-user predictive performance for this dataset and these locked classifiers. They are not causal feature-importance estimates. Exact upstream feature construction, filtering, windowing, and normalization provenance was unavailable, so a definitive leakage-free claim cannot be made. LOSO performance also does not establish within-user performance or the within-user-to-cross-user retention gap.
