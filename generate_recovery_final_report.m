function generate_recovery_final_report(artifactRoot,recoveryScript)
%GENERATE_RECOVERY_FINAL_REPORT Create final recovery and SHAP summaries.

arguments
    artifactRoot (1,1) string
    recoveryScript (1,1) string
end

artifactRoot = string(java.io.File(char(artifactRoot)).getCanonicalPath());
recoveryScript = string(java.io.File(char(recoveryScript)).getCanonicalPath());
reportFolder = fullfile(artifactRoot,"final_report");
if ~isfolder(reportFolder)
    mkdir(reportFolder);
end

fullFile = fullfile(artifactRoot,"shap_runs","full","combined", ...
    "shap_all_models_final.mat");
selectedFile = fullfile(artifactRoot,"shap_runs","selected","combined", ...
    "shap_all_models_final.mat");
if ~isfile(fullFile) || ~isfile(selectedFile)
    error("Combined full and selected SHAP exports must exist.");
end
fullPayload = load(fullFile,"mergedState","consensusTable", ...
    "featureContextTable","crossModelTable");
selectedPayload = load(selectedFile,"mergedState","consensusTable", ...
    "featureContextTable","crossModelTable");
fullState = fullPayload.mergedState;
selectedState = selectedPayload.mergedState;

expectedDatasetHash = ...
    "8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719";
expectedCommit = "e568d39a07873e52f049a827cfb65686fb308893";
if string(fullState.DatasetHash) ~= expectedDatasetHash || ...
        string(selectedState.DatasetHash) ~= expectedDatasetHash || ...
        string(fullState.GitCommit) ~= expectedCommit || ...
        string(selectedState.GitCommit) ~= expectedCommit
    error("Combined SHAP provenance does not match the locked recovery.");
end
if nnz(fullState.Completed) ~= 1134 || nnz(fullState.Failed) ~= 0
    error("Full SHAP state is incomplete.");
end
if nnz(selectedState.Completed) ~= 270 || nnz(selectedState.Failed) ~= 0
    error("Selected SHAP state is incomplete.");
end
modelInventoryFile = fullfile(reportFolder,"independent_model_inventory.csv");
shapInventoryFile = fullfile(reportFolder,"independent_shap_inventory.csv");
if ~isfile(modelInventoryFile) || ~isfile(shapInventoryFile)
    error("Independent model and SHAP inventories must exist before reporting.");
end
modelInventory = readtable(modelInventoryFile,"Delimiter",",", ...
    "TextType","string");
shapInventory = readtable(shapInventoryFile,"Delimiter",",", ...
    "TextType","string");
if height(modelInventory) ~= 1134 || ~all(modelInventory.Passed) || ...
        height(shapInventory) ~= 1404 || ~all(shapInventory.Passed)
    error("Independent inventory validation is incomplete.");
end

modelNames = string(fullState.ModelNames(:)');
featureNames = string(fullState.FeatureNames(:));
selectedFeatures = double(selectedState.ConfigSignature.FeatureNumbers(:)');

rankingRows = repmat(struct("Model","","FeatureNumber",NaN, ...
    "FeatureName","","MeanBalancedAccuracyPercent",NaN, ...
    "SDBalancedAccuracyPercent",NaN,"Rank",NaN),63,1);
rowPosition = 0;
for modelPosition = 1:numel(modelNames)
    values = squeeze(double(fullState.ReproducedBalancedAccuracy( ...
        modelPosition,:,:)));
    means = mean(values,2,"omitnan");
    standardDeviations = std(values,0,2,"omitnan");
    [~,order] = sort(means,"descend");
    ranks = nan(21,1);
    ranks(order) = 1:21;
    for featureNumber = 1:21
        rowPosition = rowPosition + 1;
        rankingRows(rowPosition).Model = modelNames(modelPosition);
        rankingRows(rowPosition).FeatureNumber = featureNumber;
        rankingRows(rowPosition).FeatureName = featureNames(featureNumber);
        rankingRows(rowPosition).MeanBalancedAccuracyPercent = ...
            100*means(featureNumber);
        rankingRows(rowPosition).SDBalancedAccuracyPercent = ...
            100*standardDeviations(featureNumber);
        rankingRows(rowPosition).Rank = ranks(featureNumber);
    end
end
featureRanking = struct2table(rankingRows);
featureRanking = sortrows(featureRanking,["Model","Rank"]);
writetable(featureRanking,fullfile(reportFolder, ...
    "recovery_feature_transferability_ranking.csv"));

stabilityRows = repmat(struct("Model","","FeatureNumber",NaN, ...
    "FeatureName","","TopChannelId",NaN,"TopChannelName","", ...
    "Top10ImportanceShare",NaN,"NormalizedEntropy",NaN, ...
    "MedianSubjectSpearman",NaN,"MedianClassSpearman",NaN), ...
    numel(modelNames)*numel(selectedFeatures),1);
rowPosition = 0;
for modelPosition = 1:numel(modelNames)
    for featureNumber = selectedFeatures
        rowPosition = rowPosition + 1;
        values = squeeze(double(selectedState.NormalizedShapley( ...
            modelPosition,featureNumber,:,:,:)));
        if ~isequal(size(values),[18 128 8]) || any(~isfinite(values),"all")
            error("Selected SHAP tensor is invalid for %s feature %d.", ...
                modelNames(modelPosition),featureNumber);
        end
        subjectProfiles = mean(values,3,"omitnan");
        classProfiles = squeeze(mean(values,1,"omitnan"))';
        meanProfile = mean(subjectProfiles,1,"omitnan");
        [sortedValues,order] = sort(meanProfile,"descend");
        distribution = meanProfile / max(sum(meanProfile),eps);
        positive = distribution > 0;

        consensusRows = selectedPayload.consensusTable( ...
            selectedPayload.consensusTable.Model == modelNames(modelPosition) & ...
            selectedPayload.consensusTable.FeatureNumber == featureNumber,:);
        topRow = find(consensusRows.ChannelId == ...
            consensusRows.ChannelId(order(1)),1,"first");
        if isempty(topRow)
            [~,topRow] = max(consensusRows.MeanNormalizedImportance);
        end

        stabilityRows(rowPosition).Model = modelNames(modelPosition);
        stabilityRows(rowPosition).FeatureNumber = featureNumber;
        stabilityRows(rowPosition).FeatureName = featureNames(featureNumber);
        stabilityRows(rowPosition).TopChannelId = consensusRows.ChannelId(topRow);
        stabilityRows(rowPosition).TopChannelName = ...
            consensusRows.ChannelName(topRow);
        stabilityRows(rowPosition).Top10ImportanceShare = ...
            sum(sortedValues(1:10));
        stabilityRows(rowPosition).NormalizedEntropy = ...
            -sum(distribution(positive).*log(distribution(positive))) / log(128);
        stabilityRows(rowPosition).MedianSubjectSpearman = ...
            medianPairwiseSpearman(subjectProfiles);
        stabilityRows(rowPosition).MedianClassSpearman = ...
            medianPairwiseSpearman(classProfiles);
    end
end
stabilityTable = struct2table(stabilityRows);
writetable(stabilityTable,fullfile(reportFolder, ...
    "selected_subject_class_channel_stability.csv"));

shapModeRows = strings(0,1);
shapModeModel = strings(0,1);
shapModePhase = strings(0,1);
shapModeCount = zeros(0,1);
for phase = ["screening","selected"]
    if phase == "screening"
        state = fullState;
    else
        state = selectedState;
    end
    for modelPosition = 1:numel(modelNames)
        values = string(state.ShapMode(modelPosition,:,:));
        values = values(strlength(values) > 0);
        modes = unique(values);
        for mode = modes(:)'
            shapModePhase(end+1,1) = phase; %#ok<AGROW>
            shapModeModel(end+1,1) = modelNames(modelPosition); %#ok<AGROW>
            shapModeRows(end+1,1) = mode; %#ok<AGROW>
            shapModeCount(end+1,1) = sum(values == mode); %#ok<AGROW>
        end
    end
end
shapModeTable = table(shapModePhase,shapModeModel,shapModeRows, ...
    shapModeCount,'VariableNames',{'Phase','Model','ShapMode','Count'});
writetable(shapModeTable,fullfile(reportFolder,"shap_mode_counts.csv"));

reportFile = fullfile(reportFolder,"FINAL_RECOVERY_REPORT.md");
fid = fopen(reportFile,"w");
if fid < 0
    error("Unable to create final report: %s",reportFile);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,"# Final recreated-model recovery and SHAP report\n\n");
fprintf(fid,"Generated: `%s`\n\n",string(datetime("now")));
fprintf(fid,["This is the **RECREATED RECOVERY COHORT**. The historical " + ...
    "pipeline discarded its fitted fold objects; these are controlled " + ...
    "recreations, not the unavailable original objects.\n\n"]);
fprintf(fid,"## Locked provenance\n\n");
fprintf(fid,"- Git commit: `%s`\n",expectedCommit);
fprintf(fid,"- Dataset SHA-256: `%s`\n",expectedDatasetHash);
fprintf(fid,"- Recovery script SHA-256: `%s`\n",sha256File(recoveryScript));
fprintf(fid,"- MATLAB release: `%s`\n",version("-release"));
fprintf(fid,"- Preprocessing provenance: **incomplete; upstream leakage-free preprocessing is not proven**.\n\n");

fprintf(fid,"## Completion evidence\n\n");
fprintf(fid,"- Persisted compact models: **1,134/1,134** (378 per family).\n");
fprintf(fid,"- Independent model inventory: **1,134/1,134 passed**.\n");
fprintf(fid,"- Screening SHAP: **1,134/1,134**, zero failed folds.\n");
fprintf(fid,"- Selected refinement SHAP: **270/270**, zero failed folds.\n");
fprintf(fid,"- Independent SHAP inventory: **1,404/1,404 passed**.\n");
fprintf(fid,"- Screening settings: 1 query/class, 2 background/class, 128 subsets.\n");
fprintf(fid,"- Refinement settings: 4 queries/class, 8 background/class, 512 subsets.\n");
fprintf(fid,"- SHAP mode counts: see `shap_mode_counts.csv`.\n\n");
fprintf(fid,"- Full combined export SHA-256: `%s`.\n",sha256File(fullFile));
fprintf(fid,"- Selected combined export SHA-256: `%s`.\n\n", ...
    sha256File(selectedFile));

fprintf(fid,"## Retained diagnostic evidence\n\n");
fprintf(fid,["Eighteen `_FAILURE.txt` markers remain intentionally in " + ...
    "superseded, non-family `smoke/folds`, `full/folds`, and " + ...
    "`selected/folds` directories. They record early compatibility, GPU " + ...
    "portability, and reference-tolerance investigations. None is inside " + ...
    "the six family-specific production fold directories or referenced by " + ...
    "either combined final export; every corresponding production key " + ...
    "passed the independent inventories.\n\n"]);

fprintf(fid,"## Locked models and actual compute\n\n");
fprintf(fid,"- LDA: pseudo-linear discriminant, uniform prior; CPU `fitcdiscr`.\n");
fprintf(fid,"- MLP: layers [64,32], ReLU, lambda 1e-4, 300 iterations, standardized, uniform prior; GPU `fitcnet`.\n");
fprintf(fid,"- RBF-SVM: Gaussian/auto scale, box constraint 1, SMO, standardized, one-vs-one ECOC, cache 512 MB; CPU-parallel `fitcecoc` with 6 workers.\n");
fprintf(fid,["- The attempted GPU RBF cohort was preserved separately after " + ...
    "five folds failed exact CPU-portable prediction equality; it is not " + ...
    "part of the validated recovery cohort.\n\n"]);

fprintf(fid,"## Feature-type transferability\n\n");
for modelName = modelNames
    rows = featureRanking(featureRanking.Model == modelName,:);
    fprintf(fid,"### %s top five\n\n",modelName);
    for row = 1:5
        fprintf(fid,"%d. Feature %d - %s: %.3f%% mean LOSO balanced accuracy\n", ...
            row,rows.FeatureNumber(row),rows.FeatureName(row), ...
            rows.MeanBalancedAccuracyPercent(row));
    end
    fprintf(fid,"\n");
end
fprintf(fid,["The refinement consensus selected Zero Crossings, Waveform " + ...
    "Length, Standard Deviation, Average Rectified Value, and Root Mean " + ...
    "Square using mean family rank, worst family rank, then mean accuracy.\n\n"]);

fprintf(fid,"## Selected channel stability\n\n");
fprintf(fid,["See `selected_subject_class_channel_stability.csv` for each " + ...
    "model/selected-feature pair's top channel, concentration, entropy, " + ...
    "median cross-subject channel-rank correlation, and median cross-class " + ...
    "channel-rank correlation. Cross-model channel ranks are in the combined " + ...
    "selected SHAP export.\n\n"]);

fprintf(fid,"## Interpretation limits\n\n");
fprintf(fid,["The screening pass intentionally uses a bounded 128-subset " + ...
    "Kernel SHAP approximation and MATLAB warns that individual values may " + ...
    "be unreliable; the selected top-five pass uses the stronger 512-subset " + ...
    "configuration. SHAP values describe channel contributions within a " + ...
    "feature-specific classifier, not causal effects or directly comparable " + ...
    "feature-type importance. RBF-ECOC scores are not calibrated " + ...
    "probabilities. LOSO addresses unseen-user transfer and does not establish " + ...
    "within-user importance.\n"]);

recoverySummary = struct;
recoverySummary.CreatedAt = datetime("now");
recoverySummary.DatasetSHA256 = expectedDatasetHash;
recoverySummary.GitCommit = expectedCommit;
recoverySummary.RecoveryScriptSHA256 = sha256File(recoveryScript);
recoverySummary.PersistedModels = 1134;
recoverySummary.ScreeningShap = nnz(fullState.Completed);
recoverySummary.SelectedShap = nnz(selectedState.Completed);
recoverySummary.IndependentShapArtifacts = height(shapInventory);
recoverySummary.FullCombinedSHA256 = sha256File(fullFile);
recoverySummary.SelectedCombinedSHA256 = sha256File(selectedFile);
recoverySummary.FeatureRanking = featureRanking;
recoverySummary.SelectedStability = stabilityTable;
recoverySummary.ShapModeCounts = shapModeTable;
save(fullfile(reportFolder,"recovery_summary.mat"), ...
    "recoverySummary","featureRanking","stabilityTable","shapModeTable", ...
    "-v7.3");

fprintf("Final report: %s\n",reportFile);
end

function value = medianPairwiseSpearman(profiles)
if size(profiles,1) < 2
    value = nan;
    return;
end
correlations = corr(profiles',"Type","Spearman","Rows","pairwise");
values = correlations(triu(true(size(correlations)),1));
value = median(values(isfinite(values)),"omitnan");
end

function digest = sha256File(filePath)
messageDigest = java.security.MessageDigest.getInstance("SHA-256");
inputStream = java.io.FileInputStream(java.io.File(char(filePath)));
digestStream = java.security.DigestInputStream(inputStream,messageDigest);
cleanup = onCleanup(@() digestStream.close());
buffer = zeros(1,1024*1024,"int8");
while digestStream.read(buffer,0,numel(buffer)) ~= -1
end
rawDigest = typecast(int8(messageDigest.digest()),"uint8");
digest = lower(string(reshape(dec2hex(rawDigest,2).',1,[])));
end
