sEMG Feature Transferability
Identifying Important and Transferable sEMG Features for Cross-User Hand Gesture Classification
Research code for investigating which high-density surface electromyography (sEMG) features distinguish hand gestures—and which retain predictive value when the user changes.
Jacci Zhang · Azaan Noman
Overview
A gesture classifier can perform well on someone it has already seen without generalizing to a new person. This project investigates that gap by separating within-user feature importance from standalone cross-user predictive performance.
Which extracted sEMG features contribute most to within-user hand-gesture classification and retain the greatest predictive value across unseen users?

The broader study compares discriminant analysis, support-vector machines, and random forests. The runnable experiment documented below is the repository's single-feature random-forest leave-one-subject-out (LOSO) benchmark: each feature type is evaluated independently across all 128 recording channels.
This is an offline research project, not a clinically validated or deployed prosthetic-control system.
Dataset
The study uses a prepared feature representation derived from CapgMyo DB-a.[^dataset] The analyzed representation is summarized in the accompanying research poster.[^poster]
Property	Value
Participants	18
Gesture classes	8
Trials per gesture per participant	10
Recording channels	128
Extracted feature types	21
Predictors per observation	2,688 = 21 × 128
Observations in the analyzed dataset	27,360
Experiment input	capgmyo_ml_dataset.mat


The experiment consumes pre-extracted features, not raw EMG waveforms. A single-feature experiment retains one feature type across all channels, producing 128 predictors, not one scalar predictor.
Experimental design
Within-user classification
The broader study evaluates gesture recognition using separate trials from the same participant. Its feature-importance analyses include SHAP values for correctly classified observations and grouped permutation importance, with the 128 channel measurements belonging to a feature type treated as a group.[^poster]
Cross-user transferability
The RF benchmark evaluates every feature type using 18-fold LOSO:
1. Select one of the 21 feature types and retain its 128 channel measurements.
2. Train a random forest on 17 participants and evaluate it on the remaining, unseen participant.
3. Repeat for every participant, then rank feature types by their mean held-out balanced accuracy.
This produces 378 model fits: 21 feature types × 18 held-out participants. Every participant serves as the test subject once per feature type.
Prepared CapgMyo features
          |
Select one feature type: 128 predictors
          |
     Split by participant
        /             \
17 participants     1 unseen participant
   Train RF             Test RF
        \               /
         Balanced accuracy
                 |
       Repeat across 18 folds
                 |
       Rank all 21 feature types
Primary metric: balanced accuracy, calculated as the mean recall across the eight gesture classes. Feature rankings average this metric across the 18 held-out participants. Ordinary accuracy and training/prediction times are also recorded. Uniform random guessing has an expected accuracy of 12.5% across eight classes.
Standalone predictive performance is not the same as importance within a multifeature model: a feature can contribute alongside other features without being the strongest predictor on its own.
Reported study results
The accompanying poster reports the following study-level accuracies.[^poster] These summarize the broader study; they are not the feature-by-feature output of the RF command below.
Model family	Within-user accuracy	Unseen-user accuracy, LOSO
Discriminant analysis†	95.7%	27.5%
Linear SVM	98.6%	23.9%
Random forest	97.1%	32.3%


† The poster labels the within-user estimator LDA and the LOSO estimator diagonal LDA. These entries should not be treated as an identical-configuration comparison. The poster reports these values as accuracy and does not supply uncertainty estimates for this summary table.
High personalized performance did not carry over to unseen users. Random forest had the highest reported LOSO accuracy among these study baselines, while linear SVM had the highest within-user accuracy.
The feature analyses tell a related story: Waveform Length and Spectral Entropy were prominent within users, whereas Zero Crossings had the highest observed standalone LOSO accuracy across the evaluated model families. However, Zero Crossings was not significantly better than most other features after correction for multiple comparisons.[^poster]
These findings motivate further investigation of calibration and adaptation; they do not establish a universal best feature or demonstrate calibration-free prosthetic control.
Run the RF benchmark
Requirements
Use MATLAB with Statistics and Machine Learning Toolbox. The RF implementation uses fitcensemble with bagged decision trees and random predictor sampling.[^matlab] Deep Learning Toolbox is not required for this baseline.
Setup
Clone the repository:
git clone https://github.com/iecclodd/Final-Models.git
cd Final-Models
Place the project's prepared capgmyo_ml_dataset.mat beside run_single_feature_rf_loso.m. Downloading the original CapgMyo recordings alone does not create this project-specific feature file.
Open the repository folder in MATLAB and run:
run("run_single_feature_rf_loso.m");
When the dataset is not beside the script, the script opens a file-selection dialog. For noninteractive execution, place the dataset beside the script beforehand.
Input data contract
The MAT file must contain these variables:
Variable	Expected content
X	Numeric feature matrix with one observation per row and 2,688 predictor columns
Y	Numeric gesture labels, one per observation, spanning eight classes
subjectID	Numeric participant identifiers, one per observation, spanning 18 participants
featureNames	Names of the 21 feature types, in feature-number order
predictorInformation	Predictor metadata containing FeatureNumber, with one entry per column of X


predictorInformation.FeatureNumber must map each feature number from 1 through 21 to exactly 128 predictor columns. Do not infer feature groups from column positions without checking this mapping.
Default model settings
Setting	Value
Ensemble method	Bagging with random predictor sampling
Trees per model	50
Minimum leaf size	5
Maximum splits per tree	500
Candidate predictors per split	11, calculated as floor(sqrt(128))
Class prior	Uniform
Random seed	Deterministic per feature–participant pair; base seed 41,000


These are the script's fixed settings, not a claim that hyperparameter optimization was performed.
Outputs and resuming
The script creates single_feature_rf_loso_results/ beside itself:
single_feature_rf_loso_results/
├── single_feature_rf_loso_progress.mat
├── single_feature_rf_loso_final.mat
├── single_feature_rf_loso_ranking.csv
├── single_feature_rf_loso_by_subject.csv
└── single_feature_rf_loso_ranking.png
The ranking CSV contains each feature's mean, standard deviation, median, minimum, and maximum held-out balanced accuracy. The subject-level CSV retains the individual participant scores. The ranking figure shows mean balanced accuracy with between-participant standard deviations.
Progress is saved after every completed feature–participant fold. Rerun the same script to resume; completed folds are skipped after the checkpoint's settings and dimensions are checked.
Use a fresh results folder after changing the dataset or experimental configuration. The checkpoint checks selected tree settings and dimensions, not a complete fingerprint of the data and code. Matching dimensions alone do not establish that two runs are compatible.
The RF script saves metrics, timings, tables, and a figure—not fitted classifier objects. Its result files are therefore not pretrained models for inference or later SHAP analysis.
Interpretation and reproducibility
The participant split excludes the held-out participant's rows from RF fitting. However, the script loads prepared features as-is; it does not establish how upstream normalization was fitted. Any learned preprocessing must be restricted to training participants before claiming strictly subject-independent evaluation.
Feature rankings are exploratory results from this dataset. Choosing a feature using all outer-fold results and reporting those same results as an unbiased estimate of a selected system would reuse test information. Further model or feature selection should use participant-grouped validation inside each outer training fold.
The study uses one dataset of 18 able-bodied participants. Its within-user SHAP analysis is restricted to correctly classified observations, and neither clinical performance nor real-time prosthetic operation was evaluated.[^poster]
Authors and citation
Jacci Zhang and Azaan Noman
To reference this research repository:
@misc{zhang_noman_semg_features_2026,
  author       = {Zhang, Jacci and Noman, Azaan},
  title        = {Identifying Important and Transferable sEMG Features
                  for Cross-User Hand Gesture Classification},
  year         = {2026},
  howpublished = {GitHub research repository},
  url          = {https://github.com/iecclodd/Final-Models}
}
Please also cite the original CapgMyo publication when using its data.[^dataset]
Documentation scope
RF commands and implementation details are based on the repository snapshot recorded on August 5, 2026, at commit 37c50f4. That snapshot contained README.md and run_single_feature_rf_loso.m on main.
Additional LDA, RBF-SVM, and MLP single-feature experiments were recorded separately on agent/add-three-emg-models at commit e568d39. Their configurations and results are not interchangeable with this RF baseline or the poster's linear-SVM results. Later branch changes are not reflected in this implementation description.
[^poster]: Zhang, J., and Noman, A. Identifying Important and Transferable sEMG Features for Cross-User Hand Gesture Classification. Accompanying research poster, 2026. Study-level results and methodological summary are transcribed from the available poster review, not recomputed by this README.
[^dataset]: Geng, W., Du, Y., Jin, W., Wei, W., Hu, Y., and Li, J. “Gesture recognition by instantaneous surface EMG images.” Scientific Reports 6, 36571 (2016). doi:10.1038/srep36571.
[^matlab]: MathWorks. fitcensemble: Fit ensemble of learners for classification.
