function mergedState = combine_family_shap_outputs(artifactRoot,runMode)
%COMBINE_FAMILY_SHAP_OUTPUTS Merge validated family-isolated SHAP exports.
%   Reads the completed LDA, MLP, and RBF-SVM family outputs without
%   modifying them, validates family completion, and creates a separate
%   cross-model export tree.

arguments
    artifactRoot (1,1) string
    runMode (1,1) string {mustBeMember(runMode,["full","selected"])}
end

artifactRoot = string(java.io.File(char(artifactRoot)).getCanonicalPath());
familyKeys = ["lda","mlp","rbf_svm"];
modelNames = ["LDA","MLP","RBF_SVM"];
familyStates = cell(numel(familyKeys),1);
debugTables = cell(numel(familyKeys),1);
pooledTables = cell(numel(familyKeys),1);
consensusTables = cell(numel(familyKeys),1);
featureContextTables = cell(numel(familyKeys),1);

for familyPosition = 1:numel(familyKeys)
    familyFolder = fullfile(artifactRoot,"shap_runs",runMode, ...
        "families",familyKeys(familyPosition));
    finalFile = fullfile(familyFolder,"shap_all_models_final.mat");
    if ~isfile(finalFile)
        error("Family final export is missing: %s",finalFile);
    end
    payload = load(finalFile,"state","debugTable","pooledTable", ...
        "consensusTable","featureContextTable");
    required = ["state","debugTable","pooledTable", ...
        "consensusTable","featureContextTable"];
    if ~all(isfield(payload,required))
        error("Family final export is incomplete: %s",finalFile);
    end
    state = payload.state;
    if string(state.DatasetHash) ~= ...
            "8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719" || ...
            string(state.GitCommit) ~= ...
            "e568d39a07873e52f049a827cfb65686fb308893"
        error("Family state provenance mismatch: %s",finalFile);
    end
    if ~isequal(string(state.ModelNames(:)'),modelNames)
        error("Family state model order mismatch: %s",finalFile);
    end
    requestedModel = modelNames(familyPosition);
    if ~isequal(string(state.ConfigSignature.Models),requestedModel)
        error("Family configuration key mismatch: %s",finalFile);
    end
    completed = nnz(state.Completed(familyPosition,:,:));
    failed = nnz(state.Failed(familyPosition,:,:));
    if runMode == "full"
        expected = 21*18;
    else
        expected = numel(state.ConfigSignature.FeatureNumbers)*18;
    end
    if completed ~= expected || failed ~= 0
        error("Family %s is incomplete: %d/%d complete, %d failed.", ...
            requestedModel,completed,expected,failed);
    end

    familyStates{familyPosition} = state;
    debugTables{familyPosition} = payload.debugTable;
    pooledTables{familyPosition} = payload.pooledTable;
    consensusTables{familyPosition} = payload.consensusTable;
    featureContextTables{familyPosition} = payload.featureContextTable;
end

mergedState = familyStates{1};
modelFields = ["MeanAbsoluteShapley","NormalizedShapley", ...
    "ReferenceBalancedAccuracy","ReproducedBalancedAccuracy", ...
    "ReproducedOrdinaryAccuracy","AccuracyDelta","TrainingSeconds", ...
    "ShapSeconds","AdditivityError","ComputeMode","ShapMode", ...
    "ModelSaved","ModelFile","ModelSHA256","ShapFile","ShapSHA256", ...
    "Stage","AttemptCount","Completed","Failed","FailureMessage"];
for familyPosition = 1:numel(familyKeys)
    sourceState = familyStates{familyPosition};
    for fieldName = modelFields
        sourceValue = sourceState.(fieldName);
        indices = repmat({':'},1,ndims(sourceValue));
        indices{1} = familyPosition;
        mergedValue = mergedState.(fieldName);
        mergedValue(indices{:}) = sourceValue(indices{:});
        mergedState.(fieldName) = mergedValue;
    end
end
mergedState.ConfigSignature.Models = modelNames;
mergedState.MergedFromFamilyIsolatedStates = true;
mergedState.MergedAt = datetime("now");
mergedState.LastCheckpoint = max(cellfun( ...
    @(value) posixtime(value.LastCheckpoint),familyStates));
mergedState.LastCheckpoint = datetime(mergedState.LastCheckpoint, ...
    "ConvertFrom","posixtime");

debugTable = vertcat(debugTables{:});
pooledTable = vertcat(pooledTables{:});
consensusTable = vertcat(consensusTables{:});
featureContextTable = vertcat(featureContextTables{:});
crossModelTable = buildCrossModelConsensus(consensusTable, ...
    mergedState.FeatureNames,10);

outputFolder = fullfile(artifactRoot,"shap_runs",runMode,"combined");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
writetable(debugTable,fullfile(outputFolder,"shap_debug_reproduction.csv"));
writetable(pooledTable,fullfile(outputFolder, ...
    "shap_channel_importance_pooled.csv"));
writetable(consensusTable,fullfile(outputFolder, ...
    "shap_channel_consensus.csv"));
writetable(featureContextTable,fullfile(outputFolder, ...
    "shap_feature_context.csv"));
writetable(crossModelTable,fullfile(outputFolder, ...
    "shap_cross_model_channel_consensus.csv"));

finalFile = fullfile(outputFolder,"shap_all_models_final.mat");
temporaryFile = fullfile(outputFolder,"." + string(java.util.UUID.randomUUID) + ...
    ".combined.tmp.mat");
save(temporaryFile,"mergedState","debugTable","pooledTable", ...
    "consensusTable","featureContextTable","crossModelTable","-v7.3");
probe = load(temporaryFile,"mergedState");
if nnz(probe.mergedState.Completed) ~= sum(cellfun( ...
        @(value) nnz(value.Completed),familyStates)) || ...
        nnz(probe.mergedState.Failed) ~= 0
    delete(temporaryFile);
    error("Temporary combined SHAP export failed validation.");
end
[success,message] = movefile(temporaryFile,finalFile,"f");
if ~success
    error("Unable to promote combined SHAP export: %s",message);
end

fprintf("Combined %s SHAP export: %d complete, 0 failed.\n", ...
    runMode,nnz(mergedState.Completed));
fprintf("Output: %s\n",finalFile);
end

function result = buildCrossModelConsensus(consensusTable,featureNames,topK)
models = unique(consensusTable.Model,"stable");
scope = strings(0,1);
featureNumbers = zeros(0,1);
featureNameColumn = strings(0,1);
channelIds = zeros(0,1);
channelNames = strings(0,1);
meanRanks = zeros(0,1);
worstRanks = zeros(0,1);
topKAppearances = zeros(0,1);

for featureNumber = unique(consensusTable.FeatureNumber)'
    featureTable = consensusTable( ...
        consensusTable.FeatureNumber == featureNumber,:);
    uniqueChannels = unique(featureTable.ChannelId,"stable");
    rankMatrix = nan(numel(models),numel(uniqueChannels));
    for modelPosition = 1:numel(models)
        modelTable = featureTable(featureTable.Model == models(modelPosition),:);
        [~,order] = sort(modelTable.MeanNormalizedImportance,"descend");
        modelRanks = nan(height(modelTable),1);
        modelRanks(order) = 1:height(modelTable);
        for channelPosition = 1:numel(uniqueChannels)
            row = find(modelTable.ChannelId == ...
                uniqueChannels(channelPosition),1,"first");
            if ~isempty(row)
                rankMatrix(modelPosition,channelPosition) = modelRanks(row);
            end
        end
    end

    for channelPosition = 1:numel(uniqueChannels)
        nameRow = find(featureTable.ChannelId == ...
            uniqueChannels(channelPosition),1,"first");
        scope(end+1,1) = "all_models"; %#ok<AGROW>
        featureNumbers(end+1,1) = featureNumber; %#ok<AGROW>
        featureNameColumn(end+1,1) = featureNames(featureNumber); %#ok<AGROW>
        channelIds(end+1,1) = uniqueChannels(channelPosition); %#ok<AGROW>
        channelNames(end+1,1) = featureTable.ChannelName(nameRow); %#ok<AGROW>
        meanRanks(end+1,1) = mean(rankMatrix(:,channelPosition),"omitnan"); %#ok<AGROW>
        worstRanks(end+1,1) = max( ...
            rankMatrix(:,channelPosition),[],"omitnan"); %#ok<AGROW>
        topKAppearances(end+1,1) = sum( ...
            rankMatrix(:,channelPosition) <= topK); %#ok<AGROW>
    end
end

result = table(scope,featureNumbers,featureNameColumn,channelIds, ...
    channelNames,meanRanks,worstRanks,topKAppearances, ...
    'VariableNames',{'Scope','FeatureNumber','FeatureName','ChannelId', ...
    'ChannelName','MeanRankAcrossModels','WorstRankAcrossModels', ...
    'TopKAppearancesAcrossModels'});
result = sortrows(result,["FeatureNumber","MeanRankAcrossModels"], ...
    ["ascend","ascend"]);
end
