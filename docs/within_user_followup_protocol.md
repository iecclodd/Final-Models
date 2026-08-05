# Within-user follow-up protocol

This protocol defines a future grouped within-subject validation. It was not run as part of the completed LOSO experiment.

## Split design

For each subject and class, preserve the ten trial groups. Assign complete trials, not windows or segments, to train, validation, and test partitions. Every segment derived from a trial must remain in the same partition. Use the same trial assignments across competing models and record the assignment seed and manifest.

## Leakage controls

Fit feature selection, normalization, and learned preprocessing on training trials only, then apply the parameters unchanged to validation and test trials. Verify subject, class, trial, and segment identifiers, and assert that each trial appears in exactly one partition.

## Metrics and ranking

Use balanced accuracy across the eight classes as the primary metric. Also report ordinary accuracy, per-class recall, macro-F1, confusion matrices, and subject-level distribution summaries. Rank feature types by held-out balanced accuracy with uncertainty intervals, retaining fold-level predictions and timings in approved private storage.

## Transfer gap

For each feature/model pair, report within-user performance beside the corresponding LOSO performance. Define the transfer gap as within-user balanced accuracy minus LOSO balanced accuracy in percentage points. Interpret it descriptively and record any trial-level bootstrap or repeated grouped-split scheme exactly.

## Reproducibility and stop criteria

Bind outputs to the dataset SHA-256, code hash, settings, split manifest, and compute mode. Stop and invalidate results on duplicate trial identifiers across partitions, window leakage, missing classes, non-finite metrics, or mismatched identities. Do not launch until the upstream feature-provenance limitation is resolved or explicitly retained in the report.
