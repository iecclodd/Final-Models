function recover_retrain_persist_shap_all_models(varargin)
%RECOVER_RETRAIN_PERSIST_SHAP_ALL_MODELS Recreate, persist, validate, and explain all LOSO models.
%
% RECOVERY PURPOSE
% ----------------
% The historical LDA, MLP, and RBF-SVM LOSO scripts completed aggregate
% evaluation but discarded each fitted fold-local classifier. MATLAB SHAP
% requires an inference-capable fitted object. This recovery pipeline is
% therefore explicitly authorized to recreate the locked classifiers, save
% a portable compact object for every feature/held-subject fold, reload and
% validate that object, and only then compute SHAP.
%
% The recreated models are a NEW RECOVERY COHORT. They must never be called
% the unavailable original objects, and they must never overwrite historical
% rankings, timings, checkpoints, or reports.
%
% SCIENTIFIC SCOPE
% ----------------
% Each classifier receives one extracted EMG feature type represented across
% 128 channels. LOSO balanced accuracy ranks the standalone cross-user value
% of the 21 feature types. SHAP ranks CHANNEL CONTRIBUTIONS inside one
% feature-specific model. Raw SHAP magnitudes from different feature-specific
% models are not causal effects and must not be treated as a direct 21-feature
% ranking.
%
% TRANSACTION GUARANTEE
% ---------------------
% A fold is complete only after this sequence succeeds:
%   train/load -> predict -> compact/gather -> atomic model save -> reload ->
%   prediction validation -> model SHA-256 -> SHAP -> atomic SHAP save ->
%   SHAP reload/dimension validation -> progress checkpoint.
%
% If the process stops after model persistence but before SHAP, the next run
% reloads that model and does NOT retrain it. Existing artifacts are trusted
% only after metadata, hash, class order, predictor count, and predictions are
% validated.
%
% Recommended staged use:
%
%   recover_retrain_persist_shap_all_models("RunMode","smoke", ...
%       "RepositoryRoot","C:/path/to/Final-Models", ...
%       "DatasetFile","C:/path/to/capgmyo_ml_dataset.mat")
%
%   recover_retrain_persist_shap_all_models("RunMode","full", ...
%       "RepositoryRoot","C:/path/to/Final-Models", ...
%       "DatasetFile","C:/path/to/capgmyo_ml_dataset.mat", ...
%       "QueryPerClass",1,"BackgroundPerClass",2, ...
%       "MaxNumSubsets",128)
%
% COMPUTE TRUTH
% -------------
% LDA uses CPU fitcdiscr. MLP may use a verified gpuArray fitcnet path. RBF-SVM
% may test a gpuArray fitcecoc path, but falls back to the locked CPU-parallel
% path only when allowed. A per-family compute-mode lock prevents silently
% mixing GPU and CPU folds. SHAP for these multiclass nonlinear models is not
% reported as general GPU SHAP.
%
% Codex must diff trainExactModel against the frozen repository scripts and
% run MATLAB Code Analyzer plus isolated smoke tests before any full run.

parser = inputParser;
parser.FunctionName = mfilename;

addParameter(parser,"RunMode","smoke", ...
    @(x) any(strcmpi(string(x),["smoke","selected","full"])));
addParameter(parser,"RepositoryRoot","", ...
    @(x) ischar(x) || isstring(x));
addParameter(parser,"DatasetFile","", ...
    @(x) ischar(x) || isstring(x));
addParameter(parser,"ArtifactRoot","", ...
    @(x) ischar(x) || isstring(x));
addParameter(parser,"PipelineStage","models_and_shap", ...
    @(x) any(strcmpi(string(x), ...
    ["models_only","shap_only","models_and_shap"])));
addParameter(parser,"ExpectedDatasetHash", ...
    "8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719", ...
    @(x) ischar(x) || isstring(x));
addParameter(parser,"Models",["LDA","MLP","RBF_SVM"], ...
    @(x) all(ismember(upper(string(x)),["LDA","MLP","RBF_SVM"])));
addParameter(parser,"FeatureNumbers",[], ...
    @(x) isnumeric(x) && isvector(x));
addParameter(parser,"HeldSubjects",[], ...
    @(x) isnumeric(x) && isvector(x));
addParameter(parser,"QueryPerClass",1, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x,1) == 0);
addParameter(parser,"BackgroundPerClass",2, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x,1) == 0);
addParameter(parser,"MaxNumSubsets",128, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 2 && mod(x,1) == 0);
addParameter(parser,"UseParallelShap",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"PreferredShapWorkers",4, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x,1) == 0);
addParameter(parser,"PreferredRbfWorkers",6, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x,1) == 0);
addParameter(parser,"UseMlpGPU",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"AllowMlpCpuFallback",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"UseRbfGPU",false, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"AllowRbfCpuFallback",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"PersistCompactModels",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"RetryFailed",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"MaxFoldAttempts",4, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x,1) == 0);
addParameter(parser,"MinimumFreeDiskGB",20, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(parser,"StrictReferenceCheck",false, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"LdaAccuracyTolerance",1e-10, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(parser,"MlpAccuracyTolerance",0.01, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(parser,"RbfAccuracyTolerance",1e-8, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(parser,"AdditivityWarningTolerance",0.05, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(parser,"TopKChannels",10, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x,1) == 0);
addParameter(parser,"Resume",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"GenerateFigures",true, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"WriteClassSpecificLongCSV",false, ...
    @(x) islogical(x) || isnumeric(x));
addParameter(parser,"StopOnFoldFailure",true, ...
    @(x) islogical(x) || isnumeric(x));

parse(parser,varargin{:});
cfg = parser.Results;
cfg.RunMode = lower(string(cfg.RunMode));
cfg.RepositoryRoot = string(cfg.RepositoryRoot);
cfg.DatasetFile = string(cfg.DatasetFile);
cfg.ArtifactRoot = string(cfg.ArtifactRoot);
cfg.PipelineStage = lower(string(cfg.PipelineStage));
cfg.ExpectedDatasetHash = lower(string(cfg.ExpectedDatasetHash));
cfg.Models = upper(string(cfg.Models(:)'));
cfg.UseParallelShap = logical(cfg.UseParallelShap);
cfg.UseMlpGPU = logical(cfg.UseMlpGPU);
cfg.AllowMlpCpuFallback = logical(cfg.AllowMlpCpuFallback);
cfg.UseRbfGPU = logical(cfg.UseRbfGPU);
cfg.AllowRbfCpuFallback = logical(cfg.AllowRbfCpuFallback);
cfg.PersistCompactModels = logical(cfg.PersistCompactModels);
cfg.RetryFailed = logical(cfg.RetryFailed);
cfg.StrictReferenceCheck = logical(cfg.StrictReferenceCheck);
cfg.Resume = logical(cfg.Resume);
cfg.GenerateFigures = logical(cfg.GenerateFigures);
cfg.WriteClassSpecificLongCSV = logical(cfg.WriteClassSpecificLongCSV);
cfg.StopOnFoldFailure = logical(cfg.StopOnFoldFailure);
if ~cfg.PersistCompactModels
    error(["PersistCompactModels=false is prohibited in this recovery. " + ...
        "The missing-model disaster must not be recreated."]);
end

fprintf("RECOVERY: RECREATE, PERSIST, VALIDATE, AND SHAP ALL MODELS\n");
fprintf("=========================================================\n\n");

scriptPath = string(mfilename("fullpath"));
if ~isfile(scriptPath) && isfile(scriptPath + ".m")
    scriptPath = scriptPath + ".m";
end
scriptFolder = string(fileparts(scriptPath));

if strlength(cfg.RepositoryRoot) == 0
    repositoryRoot = findGitRoot(scriptFolder);
else
    repositoryRoot = string(java.io.File(char(cfg.RepositoryRoot)).getCanonicalPath());
end

if ~isfolder(repositoryRoot)
    error("Repository root does not exist: %s",repositoryRoot);
end

[gitCommit,gitBranch] = readGitIdentity(repositoryRoot);
datasetFile = resolveDatasetFile(repositoryRoot,scriptFolder,cfg.DatasetFile);
datasetHash = lower(sha256File(datasetFile));
if strlength(cfg.ExpectedDatasetHash) > 0 && datasetHash ~= cfg.ExpectedDatasetHash
    error("Dataset SHA-256 mismatch. Expected %s but found %s.", ...
        cfg.ExpectedDatasetHash,datasetHash);
end

if strlength(cfg.ArtifactRoot) == 0
    artifactRoot = fullfile(fileparts(repositoryRoot), ...
        "Final-Models_recovery_artifacts");
else
    artifactRoot = canonicalPath(cfg.ArtifactRoot);
end
if startsWith(lower(artifactRoot + filesep),lower(repositoryRoot + filesep))
    error(["ArtifactRoot must be outside the Git worktree so model binaries " + ...
        "cannot be accidentally committed: %s"],artifactRoot);
end
modelRoot = fullfile(artifactRoot,"persisted_models");
familyIsolatedShap = cfg.PipelineStage == "shap_only" && ...
    numel(cfg.Models) == 1;
if familyIsolatedShap
    familyRunKey = lower(cfg.Models(1));
    runFolder = fullfile(artifactRoot,"shap_runs",cfg.RunMode, ...
        "families",familyRunKey);
    lockFolder = fullfile(artifactRoot,"locks");
    lockFile = fullfile(lockFolder,sprintf("recovery_shap_%s_%s.lock", ...
        cfg.RunMode,familyRunKey));
else
    familyRunKey = "combined";
    runFolder = fullfile(artifactRoot,"shap_runs",cfg.RunMode);
    lockFolder = artifactRoot;
    lockFile = fullfile(artifactRoot,"RECOVERY_PIPELINE.lock");
end
foldFolder = fullfile(runFolder,"folds");
figureFolder = fullfile(runFolder,"figures");
logFolder = fullfile(runFolder,"logs");
manifestFolder = fullfile(artifactRoot,"manifests");
if familyIsolatedShap
    exportManifestFolder = fullfile(runFolder,"manifests");
else
    exportManifestFolder = manifestFolder;
end
quarantineFolder = fullfile(artifactRoot,"quarantine");

for folder = [artifactRoot,modelRoot,runFolder,foldFolder,figureFolder, ...
        logFolder,manifestFolder,exportManifestFolder,lockFolder, ...
        quarantineFolder]
    ensureFolder(folder);
end

lockScope = cfg.RunMode + "/" + cfg.PipelineStage + "/" + familyRunKey;
lockToken = acquirePipelineLock(lockFile,lockScope,gitCommit,datasetHash);
cleanupLock = onCleanup(@() releasePipelineLock(lockFile,lockToken)); %#ok<NASGU>

assertFreeDiskSpace(artifactRoot,cfg.MinimumFreeDiskGB);

logFile = fullfile(logFolder, ...
    "recovery_shap_" + string(datetime("now","Format","yyyyMMdd_HHmmss")) + ".log");
diary(logFile);
cleanupDiary = onCleanup(@() diary("off")); %#ok<NASGU>

fprintf("Repository root: %s\n",repositoryRoot);
fprintf("Git branch: %s\n",gitBranch);
fprintf("Git commit: %s\n",gitCommit);
fprintf("Dataset: %s\n",datasetFile);
fprintf("Dataset SHA-256: %s\n",datasetHash);
fprintf("Artifact root: %s\n",artifactRoot);
fprintf("Run mode: %s\n",cfg.RunMode);
fprintf("Pipeline stage: %s\n",cfg.PipelineStage);
fprintf("Family-isolated SHAP: %d\n",familyIsolatedShap);
fprintf("Family run key: %s\n",familyRunKey);
fprintf("MATLAB release: %s\n\n",version("-release"));

requiredFunctions = ["fitcdiscr","fitcnet","fitcecoc","templateSVM","shapley"];
for functionName = requiredFunctions
    if exist(functionName,"file") == 0
        error("Required MATLAB function is unavailable: %s",functionName);
    end
end

fprintf("Loading dataset...\n");
D = load(datasetFile);
requiredVariables = ["X","Y","subjectID","featureNames","predictorInformation"];
loadedVariables = string(fieldnames(D));
if ~all(ismember(requiredVariables,loadedVariables))
    missingVariables = requiredVariables(~ismember(requiredVariables,loadedVariables));
    error("Dataset is missing required variables: %s",strjoin(missingVariables,", "));
end

X = D.X;
Y = double(D.Y(:));
subjectID = double(D.subjectID(:));
featureNames = string(D.featureNames(:));
predictorInformation = D.predictorInformation;

if ismember("trialID",loadedVariables)
    trialID = double(D.trialID(:));
else
    trialID = nan(size(Y));
end

if ismember("segmentID",loadedVariables)
    segmentID = double(D.segmentID(:));
else
    segmentID = nan(size(Y));
end

subjects = unique(subjectID(:))';
classOrder = unique(Y(:))';
numberOfSubjects = numel(subjects);
numberOfClasses = numel(classOrder);
numberOfFeatures = numel(featureNames);
numberOfChannels = 128;

validateDataset(X,Y,subjectID,featureNames,predictorInformation, ...
    subjects,classOrder,numberOfChannels);

predictorFeatureNumber = double(predictorInformation.FeatureNumber(:));
[channelIdsByFeature,channelNamesByFeature] = resolveChannelMetadata( ...
    predictorInformation,predictorFeatureNumber,numberOfFeatures,numberOfChannels);

[featureSelection,subjectSelection] = resolveSelections( ...
    cfg,subjects,numberOfFeatures);

fprintf("Selected models: %s\n",strjoin(cfg.Models,", "));
fprintf("Selected features: %s\n",mat2str(featureSelection));
fprintf("Selected held-out subjects: %s\n",mat2str(subjectSelection));
fprintf("Query observations per class: %d\n",cfg.QueryPerClass);
fprintf("Training-background observations per class: %d\n", ...
    cfg.BackgroundPerClass);
fprintf("Maximum Kernel SHAP subsets: %d\n\n",cfg.MaxNumSubsets);

modelNames = ["LDA","MLP","RBF_SVM"];
numberOfModels = numel(modelNames);
modelRequested = ismember(modelNames,cfg.Models);

reference = repmat(emptyReference(numberOfFeatures,numberOfSubjects), ...
    numberOfModels,1);
for modelPosition = 1:numberOfModels
    if modelRequested(modelPosition)
        reference(modelPosition) = loadReferenceResults( ...
            repositoryRoot,scriptFolder,modelNames(modelPosition), ...
            numberOfFeatures,numberOfSubjects,featureNames,subjects);
        describeReference(reference(modelPosition),modelNames(modelPosition));
    end
end

if cfg.RunMode == "full" && cfg.StrictReferenceCheck
    for modelPosition = 1:numberOfModels
        if modelRequested(modelPosition)
            if ~reference(modelPosition).Available
                error("Full SHAP requires validated reference results for %s.", ...
                    modelNames(modelPosition));
            end
            if ~all(reference(modelPosition).Completed,"all")
                error("Reference run for %s is not complete (expected 378 folds).", ...
                    modelNames(modelPosition));
            end
        end
    end
end

progressFile = fullfile(runFolder,"shap_all_models_progress.mat");
state = initializeState(numberOfModels,numberOfFeatures,numberOfSubjects, ...
    numberOfChannels,numberOfClasses,datasetHash,gitCommit,cfg, ...
    featureNames,subjects,classOrder,modelNames);

if cfg.Resume && isfile(progressFile)
    fprintf("Existing SHAP progress file found. Validating...\n");
    loadedState = load(progressFile,"state");
    if ~isfield(loadedState,"state")
        error("The SHAP progress file does not contain a state structure.");
    end
    validateResumeState(loadedState.state,state);
    state = loadedState.state;
    fprintf("Resuming with %d completed and %d failed folds.\n\n", ...
        nnz(state.Completed),nnz(state.Failed));
end

additivityChecked = false(numberOfModels,1);

for modelPosition = 1:numberOfModels
    modelName = modelNames(modelPosition);
    if ~modelRequested(modelPosition)
        continue;
    end

    fprintf("\n============================================================\n");
    fprintf("MODEL: %s\n",modelName);
    fprintf("============================================================\n");

    computeContext = prepareComputeContext(modelName,cfg);
    cleanupContext = onCleanup(@() releaseComputeContext(computeContext)); %#ok<NASGU>

    for featureNumber = featureSelection
        predictorMask = predictorFeatureNumber == featureNumber;
        XFeature = X(:,predictorMask);
        channelIds = channelIdsByFeature(featureNumber,:)';
        channelNames = channelNamesByFeature(featureNumber,:)';

        fprintf("\nFeature %d/%d: %s\n", ...
            featureNumber,numberOfFeatures,featureNames(featureNumber));

        for heldSubject = subjectSelection
            subjectPosition = find(subjects == heldSubject,1,"first");
            if isempty(subjectPosition)
                error("Unknown held-out subject: %g",heldSubject);
            end

            modelArtifactFile = modelArtifactPath( ...
                modelRoot,modelName,featureNumber,heldSubject);
            foldOutputFile = fullfile(foldFolder,sprintf( ...
                "%s_feature_%02d_subject_%02d_shap.mat", ...
                lower(modelName),featureNumber,heldSubject));

            if state.Completed(modelPosition,featureNumber,subjectPosition)
                artifactsExist = isfile(modelArtifactFile) && isfile(foldOutputFile);
                hashesMatch = false;
                if artifactsExist
                    expectedModelHash = state.ModelSHA256( ...
                        modelPosition,featureNumber,subjectPosition);
                    expectedShapHash = state.ShapSHA256( ...
                        modelPosition,featureNumber,subjectPosition);
                    hashesMatch = strlength(expectedModelHash) > 0 && ...
                        strlength(expectedShapHash) > 0 && ...
                        lower(sha256File(modelArtifactFile)) == ...
                        lower(expectedModelHash) && ...
                        lower(sha256File(foldOutputFile)) == ...
                        lower(expectedShapHash);
                end
                if artifactsExist && hashesMatch
                    fprintf(["  Subject %g: complete checkpoint and artifact " + ...
                        "hashes validated; skipped.\n"],heldSubject);
                    continue;
                end
                warning(["Progress said complete but required artifacts are " + ...
                    "missing or changed. The fold will be recovered without " + ...
                    "discarding any valid persisted model."]);
                state.Completed(modelPosition,featureNumber,subjectPosition) = false;
            end

            if state.Failed(modelPosition,featureNumber,subjectPosition) && ...
                    (~cfg.RetryFailed || ...
                    state.AttemptCount(modelPosition,featureNumber,subjectPosition) ...
                    >= cfg.MaxFoldAttempts)
                fprintf(["  Subject %g: prior failure reached retry policy " + ...
                    "(%d attempts); skipped.\n"],heldSubject, ...
                    state.AttemptCount(modelPosition,featureNumber,subjectPosition));
                continue;
            end

            state.AttemptCount(modelPosition,featureNumber,subjectPosition) = ...
                state.AttemptCount(modelPosition,featureNumber,subjectPosition) + 1;
            state.Stage(modelPosition,featureNumber,subjectPosition) = "STARTED";
            state.LastCheckpoint = datetime("now");
            atomicSaveStruct(progressFile,"state",state);

            fprintf("  Subject %g: recovering model and SHAP transaction...\n", ...
                heldSubject);

            try
                trainMask = subjectID ~= heldSubject;
                testMask = subjectID == heldSubject;
                assertNoSubjectLeakage(trainMask,testMask,subjectID,heldSubject, ...
                    Y,classOrder);

                XTrain = XFeature(trainMask,:);
                YTrain = Y(trainMask);
                XTest = XFeature(testMask,:);
                YTest = Y(testMask);
                seed = modelSeed(modelName,featureNumber,heldSubject);

                modelWasLoaded = false;
                modelHash = "";
                portableModel = [];
                trainingMeta = struct;

                if isfile(modelArtifactFile)
                    try
                        [portableModel,modelMetadata] = loadAndValidateModelArtifact( ...
                            modelArtifactFile,modelName,featureNumber,heldSubject, ...
                            datasetHash,gitCommit,classOrder,numberOfChannels, ...
                            XTest,YTest);
                        trainingMeta = modelMetadata.TrainingMeta;
                        modelHash = sha256File(modelArtifactFile);
                        modelWasLoaded = true;
                        fprintf("    Reused validated persisted model: %s\n", ...
                            modelArtifactFile);
                    catch modelLoadError
                        if cfg.PipelineStage == "shap_only"
                            error(["Persisted model failed read-only SHAP " + ...
                                "validation and was not modified: %s | Cause: %s"], ...
                                modelArtifactFile,modelLoadError.message);
                        end
                        quarantined = quarantineArtifact(modelArtifactFile, ...
                            quarantineFolder,"invalid_model_artifact");
                        warning(["Existing model artifact failed validation and " + ...
                            "was preserved at %s. It will be recreated. Cause: %s"], ...
                            quarantined,modelLoadError.message);
                    end
                end

                if ~modelWasLoaded && cfg.PipelineStage == "shap_only"
                    error(["SHAP-only stage requires an existing validated model " + ...
                        "artifact. Missing: %s"],modelArtifactFile);
                end

                if ~modelWasLoaded
                    fprintf("    Fitting the locked %s recovery model...\n",modelName);
                    [fittedModel,trainingMeta] = trainExactModel( ...
                        modelName,XTrain,YTrain,classOrder,seed,cfg,computeContext);
                    enforceComputeModeLock(manifestFolder,modelName, ...
                        trainingMeta.ComputeMode,datasetHash,gitCommit);

                    [fitPrediction,fitScores] = predictExactModel( ...
                        fittedModel,XTest,trainingMeta.UseGPU);
                    fitBalancedAccuracy = balancedAccuracy( ...
                        YTest,fitPrediction,classOrder);
                    fitOrdinaryAccuracy = mean(double(fitPrediction) == YTest);
                    assertBalancedOrdinaryAgreement( ...
                        fitBalancedAccuracy,fitOrdinaryAccuracy);

                    portableModel = makePortableCompactModel(fittedModel);
                    [portablePrediction,portableScores] = predictExactModel( ...
                        portableModel,XTest,false);
                    validatePortablePredictionEquivalence( ...
                        fitPrediction,fitScores,portablePrediction,portableScores);

                    modelMetadata = buildModelMetadata( ...
                        modelName,featureNumber,featureNames(featureNumber), ...
                        heldSubject,subjects,classOrder,find(predictorMask), ...
                        find(trainMask),find(testMask),seed,trainingMeta, ...
                        datasetFile,datasetHash,gitCommit,scriptPath, ...
                        YTest,portablePrediction,portableScores);

                    atomicSaveModelArtifact( ...
                        modelArtifactFile,portableModel,modelMetadata);
                    [portableModel,reloadedMetadata] = loadAndValidateModelArtifact( ...
                        modelArtifactFile,modelName,featureNumber,heldSubject, ...
                        datasetHash,gitCommit,classOrder,numberOfChannels, ...
                        XTest,YTest);
                    trainingMeta = reloadedMetadata.TrainingMeta;
                    modelHash = sha256File(modelArtifactFile);
                    modelWasLoaded = false;
                    clear fittedModel fitScores portableScores

                    state.ModelSaved(modelPosition,featureNumber,subjectPosition) = true;
                    state.ModelFile(modelPosition,featureNumber,subjectPosition) = ...
                        string(modelArtifactFile);
                    state.ModelSHA256(modelPosition,featureNumber,subjectPosition) = ...
                        modelHash;
                    state.Stage(modelPosition,featureNumber,subjectPosition) = ...
                        "MODEL_RELOADED_AND_VALIDATED";
                    state.LastCheckpoint = datetime("now");
                    atomicSaveStruct(progressFile,"state",state);
                else
                    enforceComputeModeLock(manifestFolder,modelName, ...
                        trainingMeta.ComputeMode,datasetHash,gitCommit);
                    state.ModelSaved(modelPosition,featureNumber,subjectPosition) = true;
                    state.ModelFile(modelPosition,featureNumber,subjectPosition) = ...
                        string(modelArtifactFile);
                    state.ModelSHA256(modelPosition,featureNumber,subjectPosition) = ...
                        modelHash;
                    state.Stage(modelPosition,featureNumber,subjectPosition) = ...
                        "MODEL_RELOADED_AND_VALIDATED";
                    state.LastCheckpoint = datetime("now");
                    atomicSaveStruct(progressFile,"state",state);
                end

                [prediction,classScores] = predictExactModel( ...
                    portableModel,XTest,false);
                reproducedBalancedAccuracy = balancedAccuracy( ...
                    YTest,prediction,classOrder);
                reproducedOrdinaryAccuracy = mean(double(prediction) == YTest);
                assertBalancedOrdinaryAgreement( ...
                    reproducedBalancedAccuracy,reproducedOrdinaryAccuracy);

                [referenceAccuracy,accuracyDelta,referencePassed] = ...
                    compareReferenceAccuracy(reference(modelPosition), ...
                    featureNumber,subjectPosition,reproducedBalancedAccuracy, ...
                    modelName,cfg);
                if cfg.StrictReferenceCheck && ~referencePassed
                    error(["The recovery fold does not reproduce the saved " + ...
                        "balanced accuracy within tolerance. Refusing SHAP."]);
                end

                if cfg.PipelineStage == "models_only"
                    state.ReferenceBalancedAccuracy(modelPosition,featureNumber, ...
                        subjectPosition) = referenceAccuracy;
                    state.ReproducedBalancedAccuracy(modelPosition,featureNumber, ...
                        subjectPosition) = reproducedBalancedAccuracy;
                    state.ReproducedOrdinaryAccuracy(modelPosition,featureNumber, ...
                        subjectPosition) = reproducedOrdinaryAccuracy;
                    state.AccuracyDelta(modelPosition,featureNumber, ...
                        subjectPosition) = accuracyDelta;
                    state.TrainingSeconds(modelPosition,featureNumber, ...
                        subjectPosition) = trainingMeta.TrainingSeconds;
                    state.ComputeMode(modelPosition,featureNumber, ...
                        subjectPosition) = trainingMeta.ComputeMode;
                    state.ModelSaved(modelPosition,featureNumber,subjectPosition) = true;
                    state.ModelFile(modelPosition,featureNumber,subjectPosition) = ...
                        string(modelArtifactFile);
                    state.ModelSHA256(modelPosition,featureNumber,subjectPosition) = ...
                        modelHash;
                    state.Stage(modelPosition,featureNumber,subjectPosition) = ...
                        "MODEL_RELOADED_AND_VALIDATED";
                    state.Failed(modelPosition,featureNumber,subjectPosition) = false;
                    state.FailureMessage(modelPosition,featureNumber,subjectPosition) = "";
                    state.LastCheckpoint = datetime("now");
                    atomicSaveStruct(progressFile,"state",state);
                    fprintf("    Model stage complete; SHAP deferred.\n");
                    clear portableModel XTrain YTrain XTest YTest classScores prediction
                    continue;
                end

                backgroundIndices = stratifiedDiverseSample( ...
                    trainMask,Y,classOrder,cfg.BackgroundPerClass, ...
                    subjectID,trialID,seed + 100000);
                queryIndices = stratifiedDiverseSample( ...
                    testMask,Y,classOrder,cfg.QueryPerClass, ...
                    subjectID,trialID,seed + 200000);

                if any(subjectID(backgroundIndices) == heldSubject)
                    error("Held-out subject leaked into SHAP background data.");
                end
                if any(subjectID(queryIndices) ~= heldSubject)
                    error("A SHAP query point is not from the held-out subject.");
                end

                XBackground = XFeature(backgroundIndices,:);
                XQuery = XFeature(queryIndices,:);
                YQuery = Y(queryIndices);

                fprintf("    Computing CPU-side interventional Kernel SHAP...\n");
                shapTimer = tic;
                [meanAbsoluteShapley,shapMeta] = computeShapImportance( ...
                    portableModel,XBackground,XQuery,numberOfClasses, ...
                    false,cfg,computeContext);
                shapSeconds = toc(shapTimer);

                validateShapMatrix(meanAbsoluteShapley, ...
                    numberOfChannels,numberOfClasses);
                normalizedShapley = normalizeShapByClass(meanAbsoluteShapley);

                additivityError = nan;
                if ~additivityChecked(modelPosition)
                    fprintf("    Running one-point SHAP additivity diagnostic...\n");
                    additivityError = checkShapAdditivity( ...
                        portableModel,XBackground,XQuery(1,:),numberOfClasses, ...
                        false,cfg,computeContext,shapMeta.Mode);
                    additivityChecked(modelPosition) = true;
                    if isfinite(additivityError) && ...
                            additivityError > cfg.AdditivityWarningTolerance
                        warning(["SHAP additivity error %.6g exceeds warning " + ...
                            "tolerance %.6g for %s."], ...
                            additivityError,cfg.AdditivityWarningTolerance,modelName);
                    end
                end

                foldResult = struct;
                foldResult.SchemaVersion = 2;
                foldResult.ModelName = modelName;
                foldResult.FeatureNumber = featureNumber;
                foldResult.FeatureName = featureNames(featureNumber);
                foldResult.HeldSubject = heldSubject;
                foldResult.ChannelIds = channelIds;
                foldResult.ChannelNames = channelNames;
                foldResult.ClassOrder = classOrder;
                foldResult.BackgroundIndices = backgroundIndices;
                foldResult.QueryIndices = queryIndices;
                foldResult.QueryLabels = YQuery;
                foldResult.ReferenceBalancedAccuracy = referenceAccuracy;
                foldResult.ReproducedBalancedAccuracy = reproducedBalancedAccuracy;
                foldResult.ReproducedOrdinaryAccuracy = reproducedOrdinaryAccuracy;
                foldResult.AccuracyDelta = accuracyDelta;
                foldResult.ReferencePassed = referencePassed;
                foldResult.MeanAbsoluteShapley = meanAbsoluteShapley;
                foldResult.NormalizedMeanAbsoluteShapley = normalizedShapley;
                foldResult.ShapMode = shapMeta.Mode;
                foldResult.ShapMethod = shapMeta.Method;
                foldResult.ShapSeconds = shapSeconds;
                foldResult.AdditivityMaxAbsoluteError = additivityError;
                foldResult.TrainingMeta = trainingMeta;
                foldResult.ModelArtifactFile = string(modelArtifactFile);
                foldResult.ModelSHA256 = modelHash;
                foldResult.ModelWasLoadedRatherThanRetrained = modelWasLoaded;
                foldResult.DatasetFile = datasetFile;
                foldResult.DatasetHash = datasetHash;
                foldResult.GitCommit = gitCommit;
                foldResult.ScriptSHA256 = sha256File(scriptPath);
                foldResult.MATLABRelease = string(version("-release"));
                foldResult.Config = cfg;
                foldResult.CreatedAt = datetime("now");

                [queryPrediction,queryScores] = predictExactModel( ...
                    portableModel,XQuery,false);
                foldResult.QueryPredictions = double(queryPrediction);
                foldResult.QueryScores = double(queryScores);

                atomicSaveAndValidateShapArtifact( ...
                    foldOutputFile,foldResult,modelHash, ...
                    datasetHash,modelName,featureNumber,heldSubject, ...
                    numberOfChannels,numberOfClasses);
                shapHash = sha256File(foldOutputFile);

                state.MeanAbsoluteShapley(modelPosition,featureNumber, ...
                    subjectPosition,:,:) = reshape(single(meanAbsoluteShapley), ...
                    1,1,1,numberOfChannels,numberOfClasses);
                state.NormalizedShapley(modelPosition,featureNumber, ...
                    subjectPosition,:,:) = reshape(single(normalizedShapley), ...
                    1,1,1,numberOfChannels,numberOfClasses);
                state.ReferenceBalancedAccuracy(modelPosition,featureNumber, ...
                    subjectPosition) = referenceAccuracy;
                state.ReproducedBalancedAccuracy(modelPosition,featureNumber, ...
                    subjectPosition) = reproducedBalancedAccuracy;
                state.ReproducedOrdinaryAccuracy(modelPosition,featureNumber, ...
                    subjectPosition) = reproducedOrdinaryAccuracy;
                state.AccuracyDelta(modelPosition,featureNumber, ...
                    subjectPosition) = accuracyDelta;
                state.TrainingSeconds(modelPosition,featureNumber, ...
                    subjectPosition) = trainingMeta.TrainingSeconds;
                state.ShapSeconds(modelPosition,featureNumber, ...
                    subjectPosition) = shapSeconds;
                state.AdditivityError(modelPosition,featureNumber, ...
                    subjectPosition) = additivityError;
                state.ComputeMode(modelPosition,featureNumber, ...
                    subjectPosition) = trainingMeta.ComputeMode;
                state.ShapMode(modelPosition,featureNumber, ...
                    subjectPosition) = shapMeta.Mode;
                state.ModelSaved(modelPosition,featureNumber,subjectPosition) = true;
                state.ModelFile(modelPosition,featureNumber,subjectPosition) = ...
                    string(modelArtifactFile);
                state.ModelSHA256(modelPosition,featureNumber,subjectPosition) = ...
                    modelHash;
                state.ShapFile(modelPosition,featureNumber,subjectPosition) = ...
                    string(foldOutputFile);
                state.ShapSHA256(modelPosition,featureNumber,subjectPosition) = ...
                    shapHash;
                state.Stage(modelPosition,featureNumber,subjectPosition) = ...
                    "FOLD_COMPLETE";
                state.Completed(modelPosition,featureNumber,subjectPosition) = true;
                state.Failed(modelPosition,featureNumber,subjectPosition) = false;
                state.FailureMessage(modelPosition,featureNumber,subjectPosition) = "";
                state.LastCheckpoint = datetime("now");
                atomicSaveStruct(progressFile,"state",state);

                fprintf(["    Complete | BA %.2f%% | delta %.4f pp | " + ...
                    "model %s | train %.2f s | SHAP %.2f s | %s\n"], ...
                    100*reproducedBalancedAccuracy,100*accuracyDelta, ...
                    ternaryString(modelWasLoaded,"reused","created"), ...
                    trainingMeta.TrainingSeconds,shapSeconds,shapMeta.Mode);

                clear portableModel XTrain YTrain XTest YTest classScores ...
                    prediction XBackground XQuery meanAbsoluteShapley ...
                    normalizedShapley

            catch foldError
                state.Failed(modelPosition,featureNumber,subjectPosition) = true;
                state.Stage(modelPosition,featureNumber,subjectPosition) = "FAILED";
                state.FailureMessage(modelPosition,featureNumber,subjectPosition) = ...
                    string(getReport(foldError,"extended","hyperlinks","off"));
                state.LastCheckpoint = datetime("now");
                atomicSaveStruct(progressFile,"state",state);

                failureFile = fullfile(foldFolder,sprintf( ...
                    "%s_feature_%02d_subject_%02d_FAILURE.txt", ...
                    lower(modelName),featureNumber,heldSubject));
                writeTextFile(failureFile,state.FailureMessage( ...
                    modelPosition,featureNumber,subjectPosition));

                warning("Fold failed for %s feature %d subject %g: %s", ...
                    modelName,featureNumber,heldSubject,foldError.message);
                if cfg.StopOnFoldFailure
                    rethrow(foldError);
                end
            end
        end
    end

    clear cleanupContext
    releaseComputeContext(computeContext);
end

fprintf("\nExporting recovery model and provenance outputs...\n");
if cfg.PipelineStage ~= "models_only"
    exportAggregateOutputs(state,reference,featureNames,subjects,classOrder, ...
        channelIdsByFeature,channelNamesByFeature,runFolder,figureFolder,cfg);
end
exportRecoveryManifests(state,exportManifestFolder);

subjectPositionsSelected = find(ismember(subjects,subjectSelection));
selectedComplete = state.Completed(modelRequested,featureSelection, ...
    subjectPositionsSelected);
selectedFailed = state.Failed(modelRequested,featureSelection, ...
    subjectPositionsSelected);
selectedModelsSaved = state.ModelSaved(modelRequested,featureSelection, ...
    subjectPositionsSelected);
expectedSelected = nnz(modelRequested)*numel(featureSelection)* ...
    numel(subjectPositionsSelected);
completedSelected = nnz(selectedComplete);
failedSelected = nnz(selectedFailed);
modelsSavedSelected = nnz(selectedModelsSaved);

fprintf("\nRECOVERY WORKFLOW STATUS\n");
fprintf("========================\n");
fprintf("Expected selected folds: %d\n",expectedSelected);
fprintf("Validated persisted models: %d\n",modelsSavedSelected);
fprintf("Validated SHAP folds: %d\n",completedSelected);
fprintf("Failed selected folds: %d\n",failedSelected);
fprintf("Artifact root: %s\n",artifactRoot);
fprintf("Results folder: %s\n",runFolder);
fprintf("Progress file: %s\n",progressFile);
fprintf("Log file: %s\n",logFile);

if cfg.RunMode == "full"
    if cfg.PipelineStage == "models_only"
        if modelsSavedSelected ~= expectedSelected || failedSelected ~= 0
            error(["Full model-persistence stage is incomplete: expected %d " + ...
                "validated models, found %d with %d failures."], ...
                expectedSelected,modelsSavedSelected,failedSelected);
        end
    elseif completedSelected ~= expectedSelected || ...
            modelsSavedSelected ~= expectedSelected || failedSelected ~= 0
        error(["Full recovery is incomplete: expected %d validated models and " + ...
            "SHAP artifacts, found %d models and %d SHAP folds with %d failures."], ...
            expectedSelected,modelsSavedSelected,completedSelected,failedSelected);
    end
end
end

%% Dataset and repository helpers

function repositoryRoot = findGitRoot(startFolder)
currentFolder = string(java.io.File(char(startFolder)).getCanonicalPath());
while true
    if isfolder(fullfile(currentFolder,".git")) || ...
            isfile(fullfile(currentFolder,".git"))
        repositoryRoot = currentFolder;
        return;
    end
    parentFolder = string(fileparts(currentFolder));
    if parentFolder == currentFolder || strlength(parentFolder) == 0
        repositoryRoot = string(startFolder);
        warning("No .git directory found; using script folder as repository root.");
        return;
    end
    currentFolder = parentFolder;
end
end

function [commit,branch] = readGitIdentity(repositoryRoot)
commit = "UNKNOWN";
branch = "UNKNOWN";
[statusCommit,outputCommit] = system(sprintf( ...
    'git -C "%s" rev-parse HEAD',repositoryRoot));
if statusCommit == 0
    commit = strtrim(string(outputCommit));
end
[statusBranch,outputBranch] = system(sprintf( ...
    'git -C "%s" branch --show-current',repositoryRoot));
if statusBranch == 0
    branch = strtrim(string(outputBranch));
end
end

function datasetFile = resolveDatasetFile(repositoryRoot,scriptFolder,explicitFile)
if strlength(explicitFile) > 0
    datasetFile = string(java.io.File(char(explicitFile)).getCanonicalPath());
    if ~isfile(datasetFile)
        error("Explicit dataset file does not exist: %s",datasetFile);
    end
    return;
end

candidatePaths = strings(0,1);
preferredNames = ["capgmyo_ml_dataset.mat","capgmyo_ml_dataset(2).mat"];
searchRoots = unique([repositoryRoot;scriptFolder]);
for searchRoot = searchRoots'
    for preferredName = preferredNames
        directPath = fullfile(searchRoot,preferredName);
        if isfile(directPath)
            candidatePaths(end+1,1) = string(directPath); %#ok<AGROW>
        end
    end
    recursiveCandidates = dir(fullfile(searchRoot,"**","capgmyo_ml_dataset*.mat"));
    for k = 1:numel(recursiveCandidates)
        candidatePaths(end+1,1) = string(fullfile( ...
            recursiveCandidates(k).folder,recursiveCandidates(k).name)); %#ok<AGROW>
    end
end
candidatePaths = unique(candidatePaths,"stable");

if isempty(candidatePaths)
    error(["No CapgMyo dataset was found. Place capgmyo_ml_dataset.mat " + ...
        "in the repository or pass DatasetFile explicitly."]);
end

if numel(candidatePaths) == 1
    datasetFile = candidatePaths(1);
    return;
end

hashes = strings(numel(candidatePaths),1);
for k = 1:numel(candidatePaths)
    hashes(k) = sha256File(candidatePaths(k));
end
if numel(unique(hashes)) ~= 1
    candidateTable = table(candidatePaths,hashes, ...
        'VariableNames',{'Path','SHA256'});
    disp(candidateTable);
    error("Multiple nonidentical dataset candidates were found.");
end

canonicalIndex = find(endsWith(candidatePaths,string(filesep) + "capgmyo_ml_dataset.mat"), ...
    1,"first");
if isempty(canonicalIndex)
    canonicalIndex = 1;
end
datasetFile = candidatePaths(canonicalIndex);
end

function hash = sha256File(filePath)
messageDigest = java.security.MessageDigest.getInstance("SHA-256");
inputStream = java.io.FileInputStream(java.io.File(char(filePath)));
digestStream = java.security.DigestInputStream(inputStream,messageDigest);
cleanupStream = onCleanup(@() digestStream.close()); %#ok<NASGU>
buffer = zeros(1,1024*1024,"int8");
while digestStream.read(buffer,0,numel(buffer)) ~= -1
end
hashBytes = typecast(int8(messageDigest.digest()),"uint8");
hash = lower(string(reshape(dec2hex(hashBytes,2).',1,[])));
end

function validateDataset(X,Y,subjectID,featureNames,predictorInformation, ...
        subjects,classOrder,numberOfChannels)
if size(X,1) ~= 27360 || size(X,2) ~= 2688
    error("Expected X to be 27360-by-2688; found %d-by-%d.", ...
        size(X,1),size(X,2));
end
if numel(Y) ~= size(X,1) || numel(subjectID) ~= size(X,1)
    error("X, Y, and subjectID observation counts do not match.");
end
if numel(subjects) ~= 18
    error("Expected 18 subjects; found %d.",numel(subjects));
end
if numel(classOrder) ~= 8
    error("Expected 8 gesture classes; found %d.",numel(classOrder));
end
if numel(featureNames) ~= 21
    error("Expected 21 feature types; found %d.",numel(featureNames));
end
if ~istable(predictorInformation) || ...
        ~ismember("FeatureNumber",string(predictorInformation.Properties.VariableNames))
    error("predictorInformation must be a table with FeatureNumber.");
end
if any(~isfinite(X),"all")
    error("X contains NaN or infinite values.");
end

featureMap = double(predictorInformation.FeatureNumber(:));
for featureNumber = 1:numel(featureNames)
    predictorCount = sum(featureMap == featureNumber);
    if predictorCount ~= numberOfChannels
        error("Feature %d has %d predictors; expected %d.", ...
            featureNumber,predictorCount,numberOfChannels);
    end
end

for heldSubject = subjects
    testMask = subjectID == heldSubject;
    trainMask = ~testMask;
    if nnz(testMask) ~= 1520 || nnz(trainMask) ~= 25840
        error("Unexpected train/test counts for subject %g.",heldSubject);
    end
    for gesture = classOrder
        if nnz(testMask & Y == gesture) ~= 190
            error("Subject %g, class %g does not contain 190 test rows.", ...
                heldSubject,gesture);
        end
        if nnz(trainMask & Y == gesture) ~= 3230
            error("Subject %g, class %g does not contain 3230 training rows.", ...
                heldSubject,gesture);
        end
    end
end
end

function [channelIdsByFeature,channelNamesByFeature] = resolveChannelMetadata( ...
        predictorInformation,featureMap,numberOfFeatures,numberOfChannels)
variableNames = string(predictorInformation.Properties.VariableNames);
channelCandidates = ["ChannelNumber","Channel","ChannelID","ElectrodeNumber", ...
    "Electrode","SensorNumber","Sensor"];
nameCandidates = ["PredictorName","VariableName","Name","ChannelName"];
channelVariable = "";
channelIndex = find(ismember(channelCandidates,variableNames),1,"first");
if ~isempty(channelIndex)
    channelVariable = channelCandidates(channelIndex);
end
nameVariable = "";
nameIndex = find(ismember(nameCandidates,variableNames),1,"first");
if ~isempty(nameIndex)
    nameVariable = nameCandidates(nameIndex);
end

channelIdsByFeature = nan(numberOfFeatures,numberOfChannels);
channelNamesByFeature = strings(numberOfFeatures,numberOfChannels);

for featureNumber = 1:numberOfFeatures
    mask = featureMap == featureNumber;
    if strlength(channelVariable) > 0
        rawChannel = predictorInformation.(channelVariable)(mask);
        if isnumeric(rawChannel) || islogical(rawChannel)
            channelIds = double(rawChannel(:));
        else
            channelIds = (1:numberOfChannels)';
        end
    else
        channelIds = (1:numberOfChannels)';
    end

    if numel(unique(channelIds)) ~= numberOfChannels
        channelIds = (1:numberOfChannels)';
    end

    if strlength(nameVariable) > 0
        rawName = string(predictorInformation.(nameVariable)(mask));
        if numel(rawName) == numberOfChannels && all(strlength(rawName) > 0)
            channelNames = rawName(:);
        else
            channelNames = compose("Channel_%03d",channelIds);
        end
    else
        channelNames = compose("Channel_%03d",channelIds);
    end

    channelIdsByFeature(featureNumber,:) = channelIds(:)';
    channelNamesByFeature(featureNumber,:) = channelNames(:)';
end
end

function [featureSelection,subjectSelection] = resolveSelections(cfg,subjects,nFeatures)
switch cfg.RunMode
    case "smoke"
        if isempty(cfg.FeatureNumbers)
            featureSelection = 1;
        else
            featureSelection = cfg.FeatureNumbers(1);
        end
        if isempty(cfg.HeldSubjects)
            subjectSelection = subjects(1);
        else
            subjectSelection = cfg.HeldSubjects(1);
        end
    case "selected"
        if isempty(cfg.FeatureNumbers) || isempty(cfg.HeldSubjects)
            error("Selected mode requires FeatureNumbers and HeldSubjects.");
        end
        featureSelection = unique(cfg.FeatureNumbers(:)','stable');
        subjectSelection = unique(cfg.HeldSubjects(:)','stable');
    case "full"
        featureSelection = 1:nFeatures;
        subjectSelection = subjects;
    otherwise
        error("Unsupported RunMode: %s",cfg.RunMode);
end
if any(featureSelection < 1 | featureSelection > nFeatures | ...
        mod(featureSelection,1) ~= 0)
    error("Feature selection contains invalid feature numbers.");
end
if ~all(ismember(subjectSelection,subjects))
    error("HeldSubjects contains IDs not present in the dataset.");
end
end

%% Reference-result helpers

function reference = emptyReference(nFeatures,nSubjects)
reference = struct;
reference.Available = false;
reference.SourceFile = "";
reference.BalancedAccuracy = nan(nFeatures,nSubjects);
reference.Completed = false(nFeatures,nSubjects);
reference.Metadata = struct;
end

function reference = loadReferenceResults(repositoryRoot,scriptFolder, ...
        modelName,nFeatures,nSubjects,featureNames,subjects)
reference = emptyReference(nFeatures,nSubjects);

switch modelName
    case "LDA"
        finalName = "single_feature_lda_loso_final.mat";
        progressName = "single_feature_lda_loso_progress.mat";
    case "MLP"
        finalName = "single_feature_mlp_loso_final.mat";
        progressName = "single_feature_mlp_loso_progress.mat";
    case "RBF_SVM"
        finalName = "single_feature_rbf_svm_loso_final.mat";
        progressName = "single_feature_rbf_svm_loso_progress.mat";
    otherwise
        error("Unsupported model name: %s",modelName);
end

finalFile = findUniqueProjectFile([repositoryRoot;scriptFolder],finalName);
progressFile = findUniqueProjectFile([repositoryRoot;scriptFolder],progressName);

if strlength(finalFile) > 0
    sourceFile = finalFile;
elseif strlength(progressFile) > 0
    sourceFile = progressFile;
else
    return;
end

R = load(sourceFile);
if ~isfield(R,"balancedAccuracy") || ...
        ~isequal(size(R.balancedAccuracy),[nFeatures,nSubjects])
    error("Reference result has invalid balancedAccuracy: %s",sourceFile);
end

reference.Available = true;
reference.SourceFile = sourceFile;
reference.BalancedAccuracy = double(R.balancedAccuracy);
if isfield(R,"completed")
    reference.Completed = logical(R.completed);
else
    reference.Completed = isfinite(reference.BalancedAccuracy);
end

if isfield(R,"featureNames") && ...
        ~isequal(string(R.featureNames(:)),string(featureNames(:)))
    error("Feature names in reference file do not match the dataset: %s", ...
        sourceFile);
end
if isfield(R,"subjects") && ...
        ~isequal(double(R.subjects(:)'),double(subjects(:)'))
    error("Subject order in reference file does not match the dataset: %s", ...
        sourceFile);
end
reference.Metadata = rmfieldIfPresent(R,["balancedAccuracy","ordinaryAccuracy", ...
    "trainingTimeSeconds","predictionTimeSeconds","completed"]);
end

function filePath = findUniqueProjectFile(searchRoots,fileName)
found = strings(0,1);
for root = unique(searchRoots(:)')
    directPath = fullfile(root,fileName);
    if isfile(directPath)
        found(end+1,1) = string(directPath); %#ok<AGROW>
    end
    candidates = dir(fullfile(root,"**",fileName));
    for k = 1:numel(candidates)
        found(end+1,1) = string(fullfile(candidates(k).folder,candidates(k).name)); %#ok<AGROW>
    end
end
found = unique(found,"stable");
if isempty(found)
    filePath = "";
elseif numel(found) == 1
    filePath = found(1);
else
    disp(table(found,'VariableNames',{'CandidateReferenceFiles'}));
    error("Multiple candidate reference files found for %s.",fileName);
end
end

function describeReference(reference,modelName)
if reference.Available
    fprintf("Reference %s: %d/%d complete | %s\n",modelName, ...
        nnz(reference.Completed),numel(reference.Completed),reference.SourceFile);
else
    fprintf("Reference %s: NOT FOUND\n",modelName);
end
end

function output = rmfieldIfPresent(input,names)
output = input;
for name = names
    if isfield(output,name)
        output = rmfield(output,name);
    end
end
end

%% Progress state

function state = initializeState(nModels,nFeatures,nSubjects,nChannels,nClasses, ...
        datasetHash,gitCommit,cfg,featureNames,subjects,classOrder,modelNames)
state = struct;
state.StateVersion = 2;
state.DatasetHash = datasetHash;
state.GitCommit = gitCommit;
state.ConfigSignature = configSignature(cfg);
state.FeatureNames = featureNames;
state.Subjects = subjects;
state.ClassOrder = classOrder;
state.ModelNames = modelNames;
state.MeanAbsoluteShapley = nan(nModels,nFeatures,nSubjects,nChannels,nClasses,"single");
state.NormalizedShapley = nan(nModels,nFeatures,nSubjects,nChannels,nClasses,"single");
state.ReferenceBalancedAccuracy = nan(nModels,nFeatures,nSubjects);
state.ReproducedBalancedAccuracy = nan(nModels,nFeatures,nSubjects);
state.ReproducedOrdinaryAccuracy = nan(nModels,nFeatures,nSubjects);
state.AccuracyDelta = nan(nModels,nFeatures,nSubjects);
state.TrainingSeconds = nan(nModels,nFeatures,nSubjects);
state.ShapSeconds = nan(nModels,nFeatures,nSubjects);
state.AdditivityError = nan(nModels,nFeatures,nSubjects);
state.ComputeMode = strings(nModels,nFeatures,nSubjects);
state.ShapMode = strings(nModels,nFeatures,nSubjects);
state.ModelSaved = false(nModels,nFeatures,nSubjects);
state.ModelFile = strings(nModels,nFeatures,nSubjects);
state.ModelSHA256 = strings(nModels,nFeatures,nSubjects);
state.ShapFile = strings(nModels,nFeatures,nSubjects);
state.ShapSHA256 = strings(nModels,nFeatures,nSubjects);
state.Stage = repmat("NOT_STARTED",nModels,nFeatures,nSubjects);
state.AttemptCount = zeros(nModels,nFeatures,nSubjects,"uint16");
state.Completed = false(nModels,nFeatures,nSubjects);
state.Failed = false(nModels,nFeatures,nSubjects);
state.FailureMessage = strings(nModels,nFeatures,nSubjects);
state.CreatedAt = datetime("now");
state.LastCheckpoint = datetime("now");
end

function signature = configSignature(cfg)
signature = struct;
signature.RunMode = cfg.RunMode;
signature.PipelineStage = cfg.PipelineStage;
signature.Models = cfg.Models;
signature.FeatureNumbers = cfg.FeatureNumbers;
signature.HeldSubjects = cfg.HeldSubjects;
signature.QueryPerClass = cfg.QueryPerClass;
signature.BackgroundPerClass = cfg.BackgroundPerClass;
signature.MaxNumSubsets = cfg.MaxNumSubsets;
signature.UseParallelShap = cfg.UseParallelShap;
signature.PreferredShapWorkers = cfg.PreferredShapWorkers;
signature.UseMlpGPU = cfg.UseMlpGPU;
signature.AllowMlpCpuFallback = cfg.AllowMlpCpuFallback;
signature.UseRbfGPU = cfg.UseRbfGPU;
signature.AllowRbfCpuFallback = cfg.AllowRbfCpuFallback;
signature.PreferredRbfWorkers = cfg.PreferredRbfWorkers;
signature.PersistCompactModels = cfg.PersistCompactModels;
signature.StrictReferenceCheck = cfg.StrictReferenceCheck;
end

function validateResumeState(existing,expected)
required = ["StateVersion","DatasetHash","ConfigSignature","FeatureNames", ...
    "Subjects","ClassOrder","ModelNames","Completed","Failed", ...
    "ModelSaved","ModelFile","ModelSHA256","ShapFile","ShapSHA256", ...
    "Stage","AttemptCount","MeanAbsoluteShapley","NormalizedShapley"];
if ~all(isfield(existing,required))
    error("Existing SHAP progress state is incomplete.");
end
if existing.StateVersion ~= expected.StateVersion
    error("SHAP progress state version mismatch.");
end
if string(existing.DatasetHash) ~= string(expected.DatasetHash)
    error("Dataset hash mismatch; do not mix SHAP analyses.");
end
if ~isequaln(existing.ConfigSignature,expected.ConfigSignature)
    error("SHAP configuration mismatch; use a separate output folder.");
end
if ~isequal(string(existing.FeatureNames(:)),string(expected.FeatureNames(:))) || ...
        ~isequal(double(existing.Subjects(:)'),double(expected.Subjects(:)')) || ...
        ~isequal(double(existing.ClassOrder(:)'),double(expected.ClassOrder(:)')) || ...
        ~isequal(string(existing.ModelNames(:)),string(expected.ModelNames(:)))
    error("Dataset metadata mismatch in existing SHAP progress.");
end
end

%% Compute context and exact model fitting

function context = prepareComputeContext(modelName,cfg)
context = struct;
context.ModelName = modelName;
context.UseRbfParallel = false;
context.RbfWorkers = 0;
context.UseShapParallel = false;
context.ShapWorkers = 0;

if cfg.PipelineStage ~= "shap_only" && ...
        ((modelName == "MLP" && cfg.UseMlpGPU) || ...
        (modelName == "RBF_SVM" && cfg.UseRbfGPU))
    pool = gcp("nocreate");
    if ~isempty(pool)
        delete(pool);
    end
else
    if license("test","Distrib_Computing_Toolbox")
        desiredWorkers = cfg.PreferredShapWorkers;
        if modelName == "RBF_SVM"
            desiredWorkers = max(desiredWorkers,cfg.PreferredRbfWorkers);
        end
        try
            pool = gcp("nocreate");
            if isempty(pool)
                pool = parpool("local",desiredWorkers);
            end
            context.ShapWorkers = pool.NumWorkers;
            context.UseShapParallel = cfg.UseParallelShap;
            if modelName == "RBF_SVM"
                context.UseRbfParallel = true;
                context.RbfWorkers = pool.NumWorkers;
            end
        catch poolError
            warning("Parallel pool unavailable; using serial execution: %s", ...
                poolError.message);
        end
    end
end
end

function releaseComputeContext(context) %#ok<INUSD>
% Keep CPU pools alive within a MATLAB invocation to avoid repeated startup.
% GPU resources are released when model and gpuArray variables are cleared.
end

function seed = modelSeed(modelName,featureNumber,heldSubject)
switch modelName
    case "LDA"
        base = 71000;
    case "MLP"
        base = 61000;
    case "RBF_SVM"
        base = 51000;
    otherwise
        error("Unsupported model name: %s",modelName);
end
seed = base + 100*featureNumber + heldSubject;
end

function [model,meta] = trainExactModel(modelName,XTrain,YTrain,classOrder, ...
        seed,cfg,context)
rng(seed,"twister");
meta = struct;
meta.ModelName = modelName;
meta.Seed = seed;
meta.UseGPU = false;
meta.GPUName = "";
meta.Workers = 0;
meta.ComputeMode = "CPU";
meta.Settings = struct;

timerValue = tic;
switch modelName
    case "LDA"
        discriminantType = "pseudoLinear";
        model = fitcdiscr(XTrain,YTrain, ...
            "DiscrimType",discriminantType, ...
            "Prior","uniform", ...
            "ClassNames",classOrder);
        meta.ComputeMode = "CPU / fitcdiscr";
        meta.Settings.DiscrimType = discriminantType;

    case "MLP"
        layerSizes = [64 32];
        activationName = "relu";
        regularizationLambda = 1e-4;
        iterationLimit = 300;
        standardizePredictors = true;

        useGPU = false;
        selectedGPU = [];
        if cfg.UseMlpGPU && ...
                license("test","Distrib_Computing_Toolbox") && ...
                ~verLessThan("matlab","24.2")
            try
                if gpuDeviceCount("available") > 0
                    selectedGPU = gpuDevice;
                    useGPU = true;
                end
            catch gpuError
                warning("MLP GPU initialization failed: %s",gpuError.message);
            end
        end

        try
            if useGPU
                XTrainModel = gpuArray(XTrain);
            else
                XTrainModel = XTrain;
            end
            model = fitcnet(XTrainModel,YTrain, ...
                "LayerSizes",layerSizes, ...
                "Activations",activationName, ...
                "Lambda",regularizationLambda, ...
                "IterationLimit",iterationLimit, ...
                "Standardize",standardizePredictors, ...
                "Prior","uniform", ...
                "ClassNames",classOrder);
            if useGPU
                wait(selectedGPU);
            end
        catch gpuFitError
            if useGPU && cfg.AllowMlpCpuFallback
                warning("MLP GPU fitting failed; retrying on CPU: %s", ...
                    gpuFitError.message);
                clear XTrainModel model
                useGPU = false;
                rng(seed,"twister");
                model = fitcnet(XTrain,YTrain, ...
                    "LayerSizes",layerSizes, ...
                    "Activations",activationName, ...
                    "Lambda",regularizationLambda, ...
                    "IterationLimit",iterationLimit, ...
                    "Standardize",standardizePredictors, ...
                    "Prior","uniform", ...
                    "ClassNames",classOrder);
            else
                rethrow(gpuFitError);
            end
        end

        meta.UseGPU = useGPU;
        if useGPU
            meta.GPUName = string(selectedGPU.Name);
            meta.ComputeMode = "GPU / fitcnet";
        else
            meta.ComputeMode = "CPU / fitcnet";
        end
        meta.Settings.LayerSizes = layerSizes;
        meta.Settings.Activation = activationName;
        meta.Settings.Lambda = regularizationLambda;
        meta.Settings.IterationLimit = iterationLimit;
        meta.Settings.Standardize = standardizePredictors;

    case "RBF_SVM"
        kernelFunction = "gaussian";
        kernelScale = "auto";
        boxConstraint = 1;
        solverName = "SMO";
        codingDesign = "onevsone";
        standardizePredictors = true;
        cacheSizeMB = 512;
        svmLearner = templateSVM( ...
            "KernelFunction",kernelFunction, ...
            "KernelScale",kernelScale, ...
            "BoxConstraint",boxConstraint, ...
            "Solver",solverName, ...
            "Standardize",standardizePredictors, ...
            "CacheSize",cacheSizeMB);

        useGPU = false;
        selectedGPU = [];
        if cfg.UseRbfGPU && license("test","Distrib_Computing_Toolbox")
            try
                if gpuDeviceCount("available") > 0
                    selectedGPU = gpuDevice;
                    useGPU = true;
                end
            catch gpuError
                warning("RBF-SVM GPU initialization failed: %s",gpuError.message);
            end
        end

        if useGPU
            try
                XTrainModel = gpuArray(XTrain);
                model = fitcecoc(XTrainModel,YTrain, ...
                    "Learners",svmLearner, ...
                    "Coding",codingDesign, ...
                    "Prior","uniform", ...
                    "ClassNames",classOrder, ...
                    "Options",statset("UseParallel",false));
                wait(selectedGPU);
                meta.UseGPU = true;
                meta.GPUName = string(selectedGPU.Name);
                meta.ComputeMode = "GPU / fitcecoc";
            catch gpuFitError
                if ~cfg.AllowRbfCpuFallback
                    rethrow(gpuFitError);
                end
                warning(["RBF-SVM GPU fitting failed; using locked CPU " + ...
                    "fallback: %s"],gpuFitError.message);
                clear XTrainModel model
                rng(seed,"twister");
                [parallelOptions,workers,useParallel] = ...
                    prepareRbfCpuFallback(cfg,context);
                model = fitcecoc(XTrain,YTrain, ...
                    "Learners",svmLearner, ...
                    "Coding",codingDesign, ...
                    "Prior","uniform", ...
                    "ClassNames",classOrder, ...
                    "Options",parallelOptions);
                meta.Workers = workers;
                if useParallel
                    meta.ComputeMode = sprintf( ...
                        "CPU parallel / fitcecoc (%d workers)",workers);
                else
                    meta.ComputeMode = "CPU serial / fitcecoc";
                end
            end
        else
            [parallelOptions,workers,useParallel] = ...
                prepareRbfCpuFallback(cfg,context);
            model = fitcecoc(XTrain,YTrain, ...
                "Learners",svmLearner, ...
                "Coding",codingDesign, ...
                "Prior","uniform", ...
                "ClassNames",classOrder, ...
                "Options",parallelOptions);
            meta.Workers = workers;
            if useParallel
                meta.ComputeMode = sprintf( ...
                    "CPU parallel / fitcecoc (%d workers)",workers);
            else
                meta.ComputeMode = "CPU serial / fitcecoc";
            end
        end

        meta.Settings.KernelFunction = kernelFunction;
        meta.Settings.KernelScale = kernelScale;
        meta.Settings.BoxConstraint = boxConstraint;
        meta.Settings.Solver = solverName;
        meta.Settings.Coding = codingDesign;
        meta.Settings.Standardize = standardizePredictors;
        meta.Settings.CacheSizeMB = cacheSizeMB;

    otherwise
        error("Unsupported model name: %s",modelName);
end
meta.TrainingSeconds = toc(timerValue);
meta.ClassNames = classOrder;
meta.MATLABRelease = string(version("-release"));
end

function [options,workers,useParallel] = prepareRbfCpuFallback(cfg,context)
workers = context.RbfWorkers;
useParallel = context.UseRbfParallel;
if ~useParallel && license("test","Distrib_Computing_Toolbox")
    try
        pool = gcp("nocreate");
        if isempty(pool)
            pool = parpool("local",cfg.PreferredRbfWorkers);
        end
        workers = pool.NumWorkers;
        useParallel = true;
    catch poolError
        warning("RBF CPU parallel pool unavailable; using serial: %s", ...
            poolError.message);
        workers = 0;
        useParallel = false;
    end
end
options = statset("UseParallel",useParallel);
end

function [prediction,scores] = predictExactModel(model,X,modelUsesGPU)
if modelUsesGPU
    XModel = gpuArray(X);
else
    XModel = X;
end
[prediction,scores] = predict(model,XModel);
if isa(prediction,"gpuArray")
    prediction = gather(prediction);
end
if isa(scores,"gpuArray")
    scores = gather(scores);
end
prediction = double(prediction);
scores = double(scores);
end

function value = balancedAccuracy(yTrue,yPred,classOrder)
recall = nan(numel(classOrder),1);
for classPosition = 1:numel(classOrder)
    classValue = classOrder(classPosition);
    classMask = yTrue == classValue;
    if ~any(classMask)
        error("A class is absent from the evaluation set: %g",classValue);
    end
    recall(classPosition) = mean(double(yPred(classMask)) == classValue);
end
value = mean(recall);
end

function [referenceAccuracy,delta,passed] = compareReferenceAccuracy( ...
        reference,featureNumber,subjectPosition,reproduced,modelName,cfg)
if reference.Available && reference.Completed(featureNumber,subjectPosition)
    referenceAccuracy = reference.BalancedAccuracy(featureNumber,subjectPosition);
    delta = reproduced - referenceAccuracy;
    switch modelName
        case "LDA"
            tolerance = cfg.LdaAccuracyTolerance;
        case "MLP"
            tolerance = cfg.MlpAccuracyTolerance;
        case "RBF_SVM"
            tolerance = cfg.RbfAccuracyTolerance;
        otherwise
            tolerance = 0;
    end
    passed = isfinite(referenceAccuracy) && abs(delta) <= tolerance;
else
    referenceAccuracy = nan;
    delta = nan;
    passed = ~cfg.StrictReferenceCheck;
end
end

function assertNoSubjectLeakage(trainMask,testMask,subjectID,heldSubject,Y,classOrder)
if any(trainMask & testMask) || any(~(trainMask | testMask))
    error("Train/test masks are invalid.");
end
if any(subjectID(trainMask) == heldSubject)
    error("Held-out subject appears in training data.");
end
if any(subjectID(testMask) ~= heldSubject)
    error("Test data contains a subject other than the held-out subject.");
end
if nnz(trainMask) ~= 25840 || nnz(testMask) ~= 1520
    error("Unexpected LOSO train/test sizes.");
end
for gesture = classOrder
    if nnz(testMask & Y == gesture) ~= 190 || ...
            nnz(trainMask & Y == gesture) ~= 3230
        error("Class balance failed for gesture %g.",gesture);
    end
end
end

%% Sampling and SHAP

function selectedIndices = stratifiedDiverseSample(eligibleMask,Y,classOrder, ...
        numberPerClass,subjectID,trialID,seed)
rng(seed,"twister");
selectedIndices = zeros(numberPerClass*numel(classOrder),1);
writePosition = 1;

for classValue = classOrder
    candidates = find(eligibleMask & Y == classValue);
    if numel(candidates) < numberPerClass
        error("Not enough candidates for class %g.",classValue);
    end

    candidateSubjects = subjectID(candidates);
    candidateTrials = trialID(candidates);
    if all(isfinite(candidateTrials))
        groupMatrix = [candidateSubjects,candidateTrials];
    else
        groupMatrix = candidateSubjects;
    end
    [~,~,groupIndex] = unique(groupMatrix,"rows","stable");
    groups = unique(groupIndex,"stable");
    groups = groups(randperm(numel(groups)));

    chosen = zeros(0,1);
    for group = groups(:)'
        groupCandidates = candidates(groupIndex == group);
        chosen(end+1,1) = groupCandidates(randi(numel(groupCandidates))); %#ok<AGROW>
        if numel(chosen) == numberPerClass
            break;
        end
    end

    if numel(chosen) < numberPerClass
        remaining = setdiff(candidates,chosen,"stable");
        remaining = remaining(randperm(numel(remaining)));
        needed = numberPerClass - numel(chosen);
        chosen = [chosen;remaining(1:needed)]; %#ok<AGROW>
    end

    selectedIndices(writePosition:writePosition+numberPerClass-1) = chosen;
    writePosition = writePosition + numberPerClass;
end

selectedIndices = selectedIndices(randperm(numel(selectedIndices)));
end

function [meanAbs,meta] = computeShapImportance(model,XBackground,XQuery, ...
        nClasses,modelUsesGPU,cfg,context)
meta = struct;
meta.Method = "interventional";
meta.Mode = "";
meta.DirectError = "";

useParallel = context.UseShapParallel && ~modelUsesGPU;
try
    explainer = createShapleyExplainer( ...
        model,XBackground,XQuery,cfg.MaxNumSubsets,useParallel);
    meanAbs = extractMeanAbsoluteShapley(explainer);
    if size(meanAbs,2) ~= nClasses
        error("Direct SHAP returned %d class columns; expected %d.", ...
            size(meanAbs,2),nClasses);
    end
    meta.Mode = "direct-model-object";
    return;
catch directError
    meta.DirectError = string(getReport(directError,"extended", ...
        "hyperlinks","off"));
    warning("Direct model SHAP failed; using per-class score functions: %s", ...
        directError.message);
end

meanAbs = nan(size(XBackground,2),nClasses);
for classPosition = 1:nClasses
    scoreFunction = @(Z) classScoreFunction( ...
        model,Z,classPosition,modelUsesGPU);
    useParallelForFunction = useParallel && ~modelUsesGPU;
    explainer = createShapleyExplainer( ...
        scoreFunction,XBackground,XQuery,cfg.MaxNumSubsets, ...
        useParallelForFunction);
    values = extractMeanAbsoluteShapley(explainer);
    if size(values,2) ~= 1
        error("Per-class SHAP function returned an unexpected shape.");
    end
    meanAbs(:,classPosition) = values(:,1);
end
meta.Mode = "per-class-score-function";
end

function score = classScoreFunction(model,Z,classPosition,modelUsesGPU)
if modelUsesGPU
    ZModel = gpuArray(single(Z));
else
    ZModel = Z;
end
[~,allScores] = predict(model,ZModel);
if isa(allScores,"gpuArray")
    allScores = gather(allScores);
end
score = double(allScores(:,classPosition));
end

function explainer = createShapleyExplainer(blackbox,XBackground,XQuery, ...
        maxNumSubsets,useParallel)
if ~verLessThan("matlab","24.1")
    arguments = { ...
        "QueryPoints",XQuery, ...
        "Method","interventional", ...
        "MaxNumSubsets",maxNumSubsets, ...
        "UseParallel",useParallel ...
        };
    if ~verLessThan("matlab","24.2")
        arguments = [arguments,{"NumObservationsToSample","all"}]; %#ok<AGROW>
    end
    explainer = shapley(blackbox,XBackground,arguments{:});
else
    % Older releases support one query point at a time. Return a lightweight
    % structure with a precomputed mean absolute table compatible with the
    % extraction helper below.
    sumAbsolute = [];
    for queryPosition = 1:size(XQuery,1)
        oneExplainer = shapley(blackbox,XBackground, ...
            "QueryPoint",XQuery(queryPosition,:), ...
            "Method","interventional", ...
            "MaxNumSubsets",maxNumSubsets, ...
            "UseParallel",useParallel);
        oneTable = getShapleyTable(oneExplainer);
        oneValues = double(table2array(oneTable(:,2:end)));
        if isempty(sumAbsolute)
            sumAbsolute = zeros(size(oneValues));
        end
        sumAbsolute = sumAbsolute + abs(oneValues);
    end
    explainer = struct;
    explainer.LegacyMeanAbsolute = sumAbsolute / size(XQuery,1);
end
end

function values = extractMeanAbsoluteShapley(explainer)
if isstruct(explainer) && isfield(explainer,"LegacyMeanAbsolute")
    values = double(explainer.LegacyMeanAbsolute);
    return;
end
if ~isprop(explainer,"MeanAbsoluteShapley") || ...
        isempty(explainer.MeanAbsoluteShapley)
    error("The SHAP explainer does not contain MeanAbsoluteShapley.");
end
T = explainer.MeanAbsoluteShapley;
values = double(table2array(T(:,2:end)));
end

function T = getShapleyTable(explainer)
if isprop(explainer,"Shapley")
    T = explainer.Shapley;
elseif isprop(explainer,"ShapleyValues")
    T = explainer.ShapleyValues;
else
    error("Unable to find Shapley/ShapleyValues property.");
end
end

function validateShapMatrix(values,nChannels,nClasses)
if ~isequal(size(values),[nChannels,nClasses])
    error("Expected a %d-by-%d SHAP matrix; found %s.", ...
        nChannels,nClasses,mat2str(size(values)));
end
if any(~isfinite(values),"all")
    error("SHAP matrix contains NaN or infinite values.");
end
if any(values < -sqrt(eps),"all")
    error("Mean absolute SHAP matrix contains negative values.");
end
end

function normalized = normalizeShapByClass(meanAbs)
normalized = zeros(size(meanAbs));
for classPosition = 1:size(meanAbs,2)
    denominator = sum(meanAbs(:,classPosition));
    if denominator > 0
        normalized(:,classPosition) = meanAbs(:,classPosition) / denominator;
    end
end
end

function maxError = checkShapAdditivity(model,XBackground,queryPoint,nClasses, ...
        modelUsesGPU,cfg,context,preferredMode)
maxError = nan;
useParallel = context.UseShapParallel && ~modelUsesGPU;

try
    if preferredMode == "direct-model-object"
        explainer = createSinglePointExplainer(model,XBackground,queryPoint, ...
            cfg.MaxNumSubsets,useParallel);
        shapTable = getShapleyTable(explainer);
        shapValues = double(table2array(shapTable(:,2:end)));
        [~,scores] = predictExactModel(model,queryPoint,modelUsesGPU);
        intercept = double(explainer.Intercept(:)');
        reconstructed = intercept + sum(shapValues,1);
        maxError = max(abs(reconstructed - scores),[],"all");
    else
        errors = nan(1,nClasses);
        for classPosition = 1:nClasses
            scoreFunction = @(Z) classScoreFunction( ...
                model,Z,classPosition,modelUsesGPU);
            explainer = createSinglePointExplainer(scoreFunction, ...
                XBackground,queryPoint,cfg.MaxNumSubsets, ...
                useParallel && ~modelUsesGPU);
            shapTable = getShapleyTable(explainer);
            shapValues = double(table2array(shapTable(:,2:end)));
            score = scoreFunction(queryPoint);
            reconstructed = double(explainer.Intercept) + sum(shapValues,"all");
            errors(classPosition) = abs(reconstructed - score);
        end
        maxError = max(errors);
    end
catch additivityError
    warning("Additivity diagnostic could not be completed: %s", ...
        additivityError.message);
end
end

function explainer = createSinglePointExplainer(blackbox,XBackground,queryPoint, ...
        maxNumSubsets,useParallel)
if ~verLessThan("matlab","24.1")
    arguments = { ...
        "QueryPoints",queryPoint, ...
        "Method","interventional", ...
        "MaxNumSubsets",maxNumSubsets, ...
        "UseParallel",useParallel ...
        };
    if ~verLessThan("matlab","24.2")
        arguments = [arguments,{"NumObservationsToSample","all"}]; %#ok<AGROW>
    end
    explainer = shapley(blackbox,XBackground,arguments{:});
else
    explainer = shapley(blackbox,XBackground, ...
        "QueryPoint",queryPoint, ...
        "Method","interventional", ...
        "MaxNumSubsets",maxNumSubsets, ...
        "UseParallel",useParallel);
end
end

%% Aggregate exports

function exportAggregateOutputs(state,reference,featureNames,subjects,classOrder, ...
        channelIdsByFeature,channelNamesByFeature,runFolder,figureFolder,cfg)
modelNames = state.ModelNames;
[nModels,nFeatures,nSubjects,nChannels,nClasses] = size(state.NormalizedShapley);

% Fold reproduction/debug table.
modelColumn = strings(0,1);
featureNumberColumn = zeros(0,1);
featureNameColumn = strings(0,1);
subjectColumn = zeros(0,1);
referenceColumn = zeros(0,1);
reproducedColumn = zeros(0,1);
deltaColumn = zeros(0,1);
trainSecondsColumn = zeros(0,1);
shapSecondsColumn = zeros(0,1);
computeModeColumn = strings(0,1);
shapModeColumn = strings(0,1);
statusColumn = strings(0,1);
messageColumn = strings(0,1);

for m = 1:nModels
    for f = 1:nFeatures
        for s = 1:nSubjects
            if ~(state.Completed(m,f,s) || state.Failed(m,f,s))
                continue;
            end
            modelColumn(end+1,1) = modelNames(m); %#ok<AGROW>
            featureNumberColumn(end+1,1) = f; %#ok<AGROW>
            featureNameColumn(end+1,1) = featureNames(f); %#ok<AGROW>
            subjectColumn(end+1,1) = subjects(s); %#ok<AGROW>
            referenceColumn(end+1,1) = state.ReferenceBalancedAccuracy(m,f,s); %#ok<AGROW>
            reproducedColumn(end+1,1) = state.ReproducedBalancedAccuracy(m,f,s); %#ok<AGROW>
            deltaColumn(end+1,1) = state.AccuracyDelta(m,f,s); %#ok<AGROW>
            trainSecondsColumn(end+1,1) = state.TrainingSeconds(m,f,s); %#ok<AGROW>
            shapSecondsColumn(end+1,1) = state.ShapSeconds(m,f,s); %#ok<AGROW>
            computeModeColumn(end+1,1) = state.ComputeMode(m,f,s); %#ok<AGROW>
            shapModeColumn(end+1,1) = state.ShapMode(m,f,s); %#ok<AGROW>
            if state.Completed(m,f,s)
                statusColumn(end+1,1) = "complete"; %#ok<AGROW>
                messageColumn(end+1,1) = ""; %#ok<AGROW>
            else
                statusColumn(end+1,1) = "failed"; %#ok<AGROW>
                messageColumn(end+1,1) = state.FailureMessage(m,f,s); %#ok<AGROW>
            end
        end
    end
end

debugTable = table(modelColumn,featureNumberColumn,featureNameColumn, ...
    subjectColumn,100*referenceColumn,100*reproducedColumn,100*deltaColumn, ...
    trainSecondsColumn,shapSecondsColumn,computeModeColumn,shapModeColumn, ...
    statusColumn,messageColumn, ...
    'VariableNames',{'Model','FeatureNumber','FeatureName','HeldSubject', ...
    'ReferenceBalancedAccuracyPercent','ReproducedBalancedAccuracyPercent', ...
    'AccuracyDeltaPercentagePoints','TrainingSeconds','ShapSeconds', ...
    'ComputeMode','ShapMode','Status','Message'});
writetable(debugTable,fullfile(runFolder,"shap_debug_reproduction.csv"));

% Pooled channel importance by fold (mean across class-normalized values).
pooledRows = nModels*nFeatures*nSubjects*nChannels;
pooledModel = strings(pooledRows,1);
pooledFeature = nan(pooledRows,1);
pooledFeatureName = strings(pooledRows,1);
pooledSubject = nan(pooledRows,1);
pooledChannelId = nan(pooledRows,1);
pooledChannelName = strings(pooledRows,1);
pooledImportance = nan(pooledRows,1);
pooledRow = 0;

for m = 1:nModels
    for f = 1:nFeatures
        for s = 1:nSubjects
            if ~state.Completed(m,f,s)
                continue;
            end
            foldValues = squeeze(double(state.NormalizedShapley(m,f,s,:,:)));
            pooled = mean(foldValues,2,"omitnan");
            for c = 1:nChannels
                pooledRow = pooledRow + 1;
                pooledModel(pooledRow) = modelNames(m);
                pooledFeature(pooledRow) = f;
                pooledFeatureName(pooledRow) = featureNames(f);
                pooledSubject(pooledRow) = subjects(s);
                pooledChannelId(pooledRow) = channelIdsByFeature(f,c);
                pooledChannelName(pooledRow) = channelNamesByFeature(f,c);
                pooledImportance(pooledRow) = pooled(c);
            end
        end
    end
end

pooledTable = table(pooledModel(1:pooledRow),pooledFeature(1:pooledRow), ...
    pooledFeatureName(1:pooledRow),pooledSubject(1:pooledRow), ...
    pooledChannelId(1:pooledRow),pooledChannelName(1:pooledRow), ...
    pooledImportance(1:pooledRow), ...
    'VariableNames',{'Model','FeatureNumber','FeatureName','HeldSubject', ...
    'ChannelId','ChannelName','MeanNormalizedAbsoluteShapleyAcrossClasses'});
writetable(pooledTable,fullfile(runFolder,"shap_channel_importance_pooled.csv"));

if cfg.WriteClassSpecificLongCSV
    writeClassSpecificLongCSV(state,featureNames,subjects,classOrder, ...
        channelIdsByFeature,channelNamesByFeature,runFolder);
end

% Consensus table and feature-level context.
consensusRows = nModels*nFeatures*nChannels;
consensusModel = strings(consensusRows,1);
consensusFeature = nan(consensusRows,1);
consensusFeatureName = strings(consensusRows,1);
consensusChannelId = nan(consensusRows,1);
consensusChannelName = strings(consensusRows,1);
consensusMean = nan(consensusRows,1);
consensusSD = nan(consensusRows,1);
consensusMedianRank = nan(consensusRows,1);
consensusTopKFrequency = nan(consensusRows,1);
consensusRow = 0;

featureContextModel = strings(0,1);
featureContextNumber = zeros(0,1);
featureContextName = strings(0,1);
featureContextMeanBA = zeros(0,1);
featureContextTopKShare = zeros(0,1);
featureContextEntropy = zeros(0,1);
featureContextMedianSubjectRankCorrelation = zeros(0,1);

for m = 1:nModels
    for f = 1:nFeatures
        completedSubjects = find(squeeze(state.Completed(m,f,:)))';
        if isempty(completedSubjects)
            continue;
        end

        subjectImportance = nan(numel(completedSubjects),nChannels);
        for p = 1:numel(completedSubjects)
            s = completedSubjects(p);
            values = squeeze(double(state.NormalizedShapley(m,f,s,:,:)));
            subjectImportance(p,:) = mean(values,2,"omitnan")';
        end

        meanImportance = mean(subjectImportance,1,"omitnan");
        sdImportance = std(subjectImportance,0,1,"omitnan");
        ranks = nan(size(subjectImportance));
        topKMask = false(size(subjectImportance));
        for p = 1:size(subjectImportance,1)
            [~,order] = sort(subjectImportance(p,:),"descend");
            ranks(p,order) = 1:nChannels;
            topKMask(p,order(1:min(cfg.TopKChannels,nChannels))) = true;
        end

        for c = 1:nChannels
            consensusRow = consensusRow + 1;
            consensusModel(consensusRow) = modelNames(m);
            consensusFeature(consensusRow) = f;
            consensusFeatureName(consensusRow) = featureNames(f);
            consensusChannelId(consensusRow) = channelIdsByFeature(f,c);
            consensusChannelName(consensusRow) = channelNamesByFeature(f,c);
            consensusMean(consensusRow) = meanImportance(c);
            consensusSD(consensusRow) = sdImportance(c);
            consensusMedianRank(consensusRow) = median(ranks(:,c),"omitnan");
            consensusTopKFrequency(consensusRow) = mean(topKMask(:,c));
        end

        sortedImportance = sort(meanImportance,"descend");
        topKShare = sum(sortedImportance(1:min(cfg.TopKChannels,nChannels)));
        distribution = meanImportance / max(sum(meanImportance),eps);
        nonzero = distribution > 0;
        normalizedEntropy = -sum(distribution(nonzero).*log(distribution(nonzero))) / ...
            log(nChannels);
        medianRankCorrelation = medianPairwiseSpearman(subjectImportance);

        featureContextModel(end+1,1) = modelNames(m); %#ok<AGROW>
        featureContextNumber(end+1,1) = f; %#ok<AGROW>
        featureContextName(end+1,1) = featureNames(f); %#ok<AGROW>
        if reference(m).Available
            featureContextMeanBA(end+1,1) = mean( ...
                reference(m).BalancedAccuracy(f,:),"omitnan"); %#ok<AGROW>
        else
            featureContextMeanBA(end+1,1) = nan; %#ok<AGROW>
        end
        featureContextTopKShare(end+1,1) = topKShare; %#ok<AGROW>
        featureContextEntropy(end+1,1) = normalizedEntropy; %#ok<AGROW>
        featureContextMedianSubjectRankCorrelation(end+1,1) = ...
            medianRankCorrelation; %#ok<AGROW>

        if cfg.GenerateFigures
            createTopChannelFigure(modelNames(m),f,featureNames(f), ...
                meanImportance,channelNamesByFeature(f,:),figureFolder,cfg.TopKChannels);
        end
    end
end

consensusTable = table(consensusModel(1:consensusRow), ...
    consensusFeature(1:consensusRow),consensusFeatureName(1:consensusRow), ...
    consensusChannelId(1:consensusRow),consensusChannelName(1:consensusRow), ...
    consensusMean(1:consensusRow),consensusSD(1:consensusRow), ...
    consensusMedianRank(1:consensusRow),consensusTopKFrequency(1:consensusRow), ...
    'VariableNames',{'Model','FeatureNumber','FeatureName','ChannelId', ...
    'ChannelName','MeanNormalizedImportance','SDNormalizedImportance', ...
    'MedianChannelRankAcrossSubjects','TopKFrequencyAcrossSubjects'});
writetable(consensusTable,fullfile(runFolder,"shap_channel_consensus.csv"));

featureContextTable = table(featureContextModel,featureContextNumber, ...
    featureContextName,100*featureContextMeanBA,featureContextTopKShare, ...
    featureContextEntropy,featureContextMedianSubjectRankCorrelation, ...
    'VariableNames',{'Model','FeatureNumber','FeatureName', ...
    'MeanLOSOBalancedAccuracyPercent','TopKChannelImportanceShare', ...
    'NormalizedChannelImportanceEntropy', ...
    'MedianCrossSubjectSpearmanChannelRankCorrelation'});
writetable(featureContextTable,fullfile(runFolder,"shap_feature_context.csv"));

crossModelTable = buildCrossModelConsensus(consensusTable,featureNames,cfg.TopKChannels);
writetable(crossModelTable,fullfile(runFolder,"shap_cross_model_channel_consensus.csv"));

save(fullfile(runFolder,"shap_all_models_final.mat"), ...
    "state","debugTable","pooledTable","consensusTable", ...
    "featureContextTable","crossModelTable","-v7.3");

writeMarkdownReport(state,reference,featureContextTable,runFolder,cfg);
end

function writeClassSpecificLongCSV(state,featureNames,subjects,classOrder, ...
        channelIdsByFeature,channelNamesByFeature,runFolder)
modelNames = state.ModelNames;
[nModels,nFeatures,nSubjects,nChannels,nClasses] = size(state.NormalizedShapley);
outputFile = fullfile(runFolder,"shap_channel_importance_by_class.csv");
if isfile(outputFile)
    delete(outputFile);
end
firstWrite = true;
for m = 1:nModels
    for f = 1:nFeatures
        for s = 1:nSubjects
            if ~state.Completed(m,f,s)
                continue;
            end
            values = squeeze(double(state.NormalizedShapley(m,f,s,:,:)));
            for classPosition = 1:nClasses
                T = table( ...
                    repmat(modelNames(m),nChannels,1), ...
                    repmat(f,nChannels,1), ...
                    repmat(featureNames(f),nChannels,1), ...
                    repmat(subjects(s),nChannels,1), ...
                    repmat(classOrder(classPosition),nChannels,1), ...
                    channelIdsByFeature(f,:)', ...
                    channelNamesByFeature(f,:)', ...
                    values(:,classPosition), ...
                    'VariableNames',{'Model','FeatureNumber','FeatureName', ...
                    'HeldSubject','Class','ChannelId','ChannelName', ...
                    'NormalizedAbsoluteShapley'});
                if firstWrite
                    writetable(T,outputFile);
                    firstWrite = false;
                else
                    writetable(T,outputFile,"WriteMode","append", ...
                        "WriteVariableNames",false);
                end
            end
        end
    end
end
end

function correlation = medianPairwiseSpearman(subjectImportance)
if size(subjectImportance,1) < 2
    correlation = nan;
    return;
end
C = corr(subjectImportance',"Type","Spearman","Rows","pairwise");
upperMask = triu(true(size(C)),1);
values = C(upperMask);
correlation = median(values(isfinite(values)),"omitnan");
end

function createTopChannelFigure(modelName,featureNumber,featureName, ...
        meanImportance,channelNames,figureFolder,topK)
[sortedValues,order] = sort(meanImportance,"descend");
numberToPlot = min(max(topK,20),numel(order));
order = order(1:numberToPlot);
sortedValues = sortedValues(1:numberToPlot);
fig = figure("Visible","off","Color","white");
barh(sortedValues);
yticks(1:numberToPlot);
yticklabels(channelNames(order));
set(gca,"YDir","reverse");
xlabel("Mean normalized |SHAP| across classes and held-out subjects");
ylabel("Channel");
title(sprintf("%s | Feature %d: %s",modelName,featureNumber,featureName), ...
    "Interpreter","none");
grid on;
fileName = sprintf("%s_feature_%02d_top_channels.png", ...
    lower(modelName),featureNumber);
exportgraphics(fig,fullfile(figureFolder,fileName),"Resolution",200);
close(fig);
end

function T = buildCrossModelConsensus(consensusTable,featureNames,topK)
models = unique(consensusTable.Model,"stable");
rows = strings(0,1);
featureNumbers = zeros(0,1);
featureNameColumn = strings(0,1);
channelIds = zeros(0,1);
channelNames = strings(0,1);
meanRanks = zeros(0,1);
worstRanks = zeros(0,1);
topKAppearances = zeros(0,1);

for f = unique(consensusTable.FeatureNumber)'
    featureTable = consensusTable(consensusTable.FeatureNumber == f,:);
    uniqueChannels = unique(featureTable.ChannelId,"stable");
    rankMatrix = nan(numel(models),numel(uniqueChannels));
    for m = 1:numel(models)
        modelTable = featureTable(featureTable.Model == models(m),:);
        [~,order] = sort(modelTable.MeanNormalizedImportance,"descend");
        modelRanks = nan(height(modelTable),1);
        modelRanks(order) = 1:height(modelTable);
        for c = 1:numel(uniqueChannels)
            row = find(modelTable.ChannelId == uniqueChannels(c),1,"first");
            if ~isempty(row)
                rankMatrix(m,c) = modelRanks(row);
            end
        end
    end

    for c = 1:numel(uniqueChannels)
        nameRow = find(featureTable.ChannelId == uniqueChannels(c),1,"first");
        rows(end+1,1) = "all_models"; %#ok<AGROW>
        featureNumbers(end+1,1) = f; %#ok<AGROW>
        featureNameColumn(end+1,1) = featureNames(f); %#ok<AGROW>
        channelIds(end+1,1) = uniqueChannels(c); %#ok<AGROW>
        channelNames(end+1,1) = featureTable.ChannelName(nameRow); %#ok<AGROW>
        meanRanks(end+1,1) = mean(rankMatrix(:,c),"omitnan"); %#ok<AGROW>
        worstRanks(end+1,1) = max(rankMatrix(:,c),[],"omitnan"); %#ok<AGROW>
        topKAppearances(end+1,1) = sum(rankMatrix(:,c) <= topK); %#ok<AGROW>
    end
end

T = table(rows,featureNumbers,featureNameColumn,channelIds,channelNames, ...
    meanRanks,worstRanks,topKAppearances, ...
    'VariableNames',{'Scope','FeatureNumber','FeatureName','ChannelId', ...
    'ChannelName','MeanRankAcrossModels','WorstRankAcrossModels', ...
    'TopKAppearancesAcrossModels'});
T = sortrows(T,["FeatureNumber","MeanRankAcrossModels"],["ascend","ascend"]);
end

function writeMarkdownReport(state,reference,featureContextTable,runFolder,cfg)
reportFile = fullfile(runFolder,"shap_report.md");
fid = fopen(reportFile,"w");
if fid < 0
    error("Unable to create report: %s",reportFile);
end
cleanupFile = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid,"# Recreated LDA, MLP, and RBF-SVM model + SHAP report\n\n");
fprintf(fid,["**Recovery status:** These fitted objects were recreated because " + ...
    "the historical pipeline did not persist its fold-local classifiers. " + ...
    "They are not the unavailable original model objects.\n\n"]);
fprintf(fid,"- MATLAB release: `%s`\n",version("-release"));
fprintf(fid,"- Run mode: `%s`\n",cfg.RunMode);
fprintf(fid,"- Pipeline stage: `%s`\n",cfg.PipelineStage);
fprintf(fid,"- Dataset preprocessing provenance: `incomplete / not proven leakage-free`\n");
fprintf(fid,"- Completed SHAP folds: %d\n",nnz(state.Completed));
fprintf(fid,"- Failed SHAP folds: %d\n",nnz(state.Failed));
fprintf(fid,"- Query observations per class: %d\n",cfg.QueryPerClass);
fprintf(fid,"- Training-background observations per class: %d\n", ...
    cfg.BackgroundPerClass);
fprintf(fid,"- Maximum Kernel SHAP subsets: %d\n\n",cfg.MaxNumSubsets);

fprintf(fid,"## Scientific interpretation guardrails\n\n");
fprintf(fid,["The recovery experiment trains one model from one extracted " + ...
    "feature type across 128 channels. Consequently, this SHAP analysis " + ...
    "explains channel contributions *within* a feature-specific classifier. " + ...
    "It does not replace the LOSO balanced-accuracy ranking of the 21 feature " + ...
    "types. Raw SHAP magnitudes from separately trained models are not treated " + ...
    "as directly comparable feature-importance scores.\n\n"]);
fprintf(fid,["The SHAP background contains only observations from the 17 " + ...
    "training subjects. Query points come only from the held-out subject. " + ...
    "This aligns the explanation with cross-user transfer, while avoiding " + ...
    "using the held-out subject as the SHAP baseline.\n\n"]);
fprintf(fid,["LDA and MLP class-score explanations concern posterior class " + ...
    "scores. The default RBF-SVM ECOC explanation concerns the ECOC class " + ...
    "score (negated aggregate binary loss), not a calibrated probability, " + ...
    "unless the final repository model was explicitly trained with posterior " + ...
    "fitting and the script was updated accordingly.\n\n"]);
fprintf(fid,["This LOSO SHAP workflow addresses explanations on unseen users. " + ...
    "It does not by itself establish within-user importance. A personalized " + ...
    "comparison requires separately validated within-subject models with " + ...
    "trial-grouped splits.\n\n"]);

fprintf(fid,"## Model status\n\n");
for m = 1:numel(state.ModelNames)
    fprintf(fid,"### %s\n\n",state.ModelNames(m));
    fprintf(fid,"- Reference available: %d\n",reference(m).Available);
    fprintf(fid,"- SHAP folds complete: %d\n",nnz(state.Completed(m,:,:)));
    fprintf(fid,"- SHAP folds failed: %d\n\n",nnz(state.Failed(m,:,:)));
end

fprintf(fid,"## Feature context\n\n");
fprintf(fid,["See `shap_feature_context.csv` for each model-feature pair's " + ...
    "mean LOSO balanced accuracy, top-channel concentration, channel-importance " + ...
    "entropy, and cross-subject channel-rank stability. Interpret LOSO accuracy " + ...
    "as feature-type transferability and SHAP statistics as within-feature " + ...
    "channel attribution/stability.\n\n"]);

if ~isempty(featureContextTable)
    for modelName = unique(featureContextTable.Model,"stable")'
        T = featureContextTable(featureContextTable.Model == modelName,:);
        [~,order] = sort(T.MeanLOSOBalancedAccuracyPercent,"descend", ...
            "MissingPlacement","last");
        T = T(order,:);
        fprintf(fid,"### %s top feature-transferability rows\n\n",modelName);
        for row = 1:min(5,height(T))
            fprintf(fid,"%d. Feature %d — %s: %.3f%% mean LOSO balanced accuracy\n", ...
                row,T.FeatureNumber(row),T.FeatureName(row), ...
                T.MeanLOSOBalancedAccuracyPercent(row));
        end
        fprintf(fid,"\n");
    end
end
end

%% File utilities

function ensureFolder(folderPath)
if ~isfolder(folderPath)
    mkdir(folderPath);
end
end

function atomicSaveStruct(filePath,variableName,value)
folder = string(fileparts(filePath));
ensureFolder(folder);
temporaryFile = fullfile(folder,"." + string(java.util.UUID.randomUUID) + ".tmp.mat");
S = struct;
S.(variableName) = value;
save(temporaryFile,"-struct","S","-v7.3");
validation = load(temporaryFile,variableName);
if ~isfield(validation,variableName)
    delete(temporaryFile);
    error("Temporary checkpoint validation failed: %s",temporaryFile);
end
[success,message] = movefile(temporaryFile,filePath,"f");
if ~success
    error("Atomic checkpoint replacement failed: %s",message);
end
end

function atomicSaveAndValidateShapArtifact(filePath,foldResult,modelHash, ...
        datasetHash,modelName,featureNumber,heldSubject,nChannels,nClasses)
folder = string(fileparts(filePath));
ensureFolder(folder);
temporaryFile = fullfile(folder,"." + string(java.util.UUID.randomUUID) + ...
    ".shap.tmp.mat");
save(temporaryFile,"foldResult","-v7.3");
try
    validateSavedShapArtifact(temporaryFile,modelHash,datasetHash, ...
        modelName,featureNumber,heldSubject,nChannels,nClasses);
catch validationError
    if isfile(temporaryFile)
        delete(temporaryFile);
    end
    rethrow(validationError);
end
[success,message] = movefile(temporaryFile,filePath,"f");
if ~success
    error("Atomic SHAP-artifact replacement failed: %s",message);
end
validateSavedShapArtifact(filePath,modelHash,datasetHash,modelName, ...
    featureNumber,heldSubject,nChannels,nClasses);
end

function writeTextFile(filePath,textValue)
fid = fopen(filePath,"w");
if fid < 0
    error("Unable to write file: %s",filePath);
end
cleanupFile = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,"%s",textValue);
end

%% Recovery safety, model persistence, and validation helpers

function pathValue = canonicalPath(pathValue)
pathValue = string(java.io.File(char(pathValue)).getCanonicalPath());
end

function token = acquirePipelineLock(lockFile,runMode,gitCommit,datasetHash)
lockFile = string(lockFile);
token = string(java.util.UUID.randomUUID);
if isfile(lockFile)
    existingText = string(fileread(lockFile));
    existingPid = nan;
    try
        existing = jsondecode(existingText);
        if isfield(existing,"PID")
            existingPid = double(existing.PID);
        end
    catch
        % Preserve malformed locks rather than deleting them silently.
    end
    if isfinite(existingPid) && processIsAlive(existingPid)
        error(["A recovery pipeline is already active with PID %d. " + ...
            "Do not launch a duplicate MATLAB process. Lock: %s"], ...
            existingPid,lockFile);
    end
    staleFile = lockFile + ".stale_" + ...
        string(datetime("now","Format","yyyyMMdd_HHmmss"));
    [ok,message] = movefile(lockFile,staleFile,"f");
    if ~ok
        error("Unable to preserve stale lock: %s",message);
    end
end

javaFile = java.io.File(char(lockFile));
if ~javaFile.createNewFile()
    error("Unable to atomically create pipeline lock: %s",lockFile);
end
metadata = struct;
metadata.Token = token;
metadata.PID = matlabProcessID;
metadata.Host = string(java.net.InetAddress.getLocalHost.getHostName);
metadata.StartedAt = string(datetime("now","Format","yyyy-MM-dd'T'HH:mm:ssXXX"));
metadata.RunMode = string(runMode);
metadata.GitCommit = string(gitCommit);
metadata.DatasetHash = string(datasetHash);
writeTextFile(lockFile,jsonencode(metadata,"PrettyPrint",true));
end

function releasePipelineLock(lockFile,token)
if ~isfile(lockFile)
    return;
end
try
    metadata = jsondecode(fileread(lockFile));
    if isfield(metadata,"Token") && string(metadata.Token) == string(token)
        delete(lockFile);
    end
catch
    warning("Pipeline lock could not be parsed during cleanup; preserved: %s", ...
        lockFile);
end
end

function alive = processIsAlive(pid)
if ispc
    [status,output] = system(sprintf('tasklist /FI "PID eq %d" /NH',pid));
    alive = status == 0 && contains(string(output),string(pid));
else
    [status,~] = system(sprintf('kill -0 %d 2>/dev/null',pid));
    alive = status == 0;
end
end

function assertFreeDiskSpace(folderPath,minimumGB)
fileObject = java.io.File(char(folderPath));
freeGB = double(fileObject.getUsableSpace()) / 1024^3;
fprintf("Available artifact-disk space: %.2f GB\n",freeGB);
if freeGB < minimumGB
    error(["Only %.2f GB is free at the artifact location; at least %.2f GB " + ...
        "is required before starting. Do not delete validated models to make " + ...
        "room. Choose a larger ArtifactRoot."],freeGB,minimumGB);
end
end

function filePath = modelArtifactPath(modelRoot,modelName,featureNumber,heldSubject)
familyFolder = fullfile(modelRoot,lower(modelName), ...
    sprintf("feature_%02d",featureNumber));
ensureFolder(familyFolder);
filePath = fullfile(familyFolder,sprintf( ...
    "%s_feature_%02d_subject_%02d_compact_model.mat", ...
    lower(modelName),featureNumber,heldSubject));
end

function portableModel = makePortableCompactModel(model)
portableModel = model;
try
    portableModel = gather(portableModel);
catch
    % CPU models do not require gather; continue to compact.
end
try
    portableModel = compact(portableModel);
catch compactError
    error("Unable to create a compact deployable model: %s", ...
        compactError.message);
end
try
    portableModel = gather(portableModel);
catch
    % Some compact CPU classes do not implement gather.
end
if contains(class(portableModel),"Partitioned","IgnoreCase",true)
    error("A partitioned model is not a deployable fold classifier.");
end
end

function metadata = buildModelMetadata(modelName,featureNumber,featureName, ...
        heldSubject,subjects,classOrder,predictorIndices,trainRows,testRows, ...
        seed,trainingMeta,datasetFile,datasetHash,gitCommit,scriptPath, ...
        yTest,prediction,scores)
metadata = struct;
metadata.SchemaVersion = 2;
metadata.RecoveryCohort = "recreated_after_missing-model-artifact-audit";
metadata.IsOriginalHistoricalObject = false;
metadata.ModelName = string(modelName);
metadata.FeatureNumber = featureNumber;
metadata.FeatureName = string(featureName);
metadata.HeldSubject = heldSubject;
metadata.TrainingSubjects = setdiff(double(subjects(:)'),heldSubject,"stable");
metadata.ClassOrder = double(classOrder(:)');
metadata.NumPredictors = numel(predictorIndices);
metadata.PredictorIndices = double(predictorIndices(:)');
metadata.TrainingRowCount = numel(trainRows);
metadata.TestRowCount = numel(testRows);
metadata.TrainingRowIndexSHA256 = sha256Numeric(trainRows);
metadata.TestRowIndexSHA256 = sha256Numeric(testRows);
metadata.Seed = seed;
metadata.TrainingMeta = trainingMeta;
metadata.DatasetFile = string(datasetFile);
metadata.DatasetHash = string(datasetHash);
metadata.GitCommit = string(gitCommit);
metadata.ScriptFile = string(scriptPath);
metadata.ScriptSHA256 = sha256File(scriptPath);
metadata.MATLABRelease = string(version("-release"));
metadata.ModelClassBeforeSave = "portable compact model";
metadata.TestBalancedAccuracy = balancedAccuracy(yTest,prediction,classOrder);
metadata.TestOrdinaryAccuracy = mean(double(prediction) == double(yTest));
metadata.TestPredictions = double(prediction(:));
metadata.TestPredictionSHA256 = sha256Numeric(double(prediction(:)));
metadata.TestScoresSHA256 = sha256Numeric(double(scores(:)));
metadata.ConfusionMatrix = confusionmat(double(yTest),double(prediction), ...
    "Order",double(classOrder));
metadata.CreatedAt = datetime("now");
end

function atomicSaveModelArtifact(filePath,portableModel,modelMetadata)
folder = string(fileparts(filePath));
ensureFolder(folder);
temporaryFile = fullfile(folder,"." + string(java.util.UUID.randomUUID) + ...
    ".model.tmp.mat");
compactModel = portableModel; %#ok<NASGU>
save(temporaryFile,"compactModel","modelMetadata","-v7.3");
probe = load(temporaryFile,"compactModel","modelMetadata");
if ~isfield(probe,"compactModel") || ~isfield(probe,"modelMetadata")
    delete(temporaryFile);
    error("Temporary model artifact failed reload validation: %s",temporaryFile);
end
[success,message] = movefile(temporaryFile,filePath,"f");
if ~success
    error("Atomic model-artifact replacement failed: %s",message);
end
end

function [model,metadata] = loadAndValidateModelArtifact(filePath, ...
        expectedModelName,expectedFeature,expectedSubject,datasetHash,gitCommit, ...
        classOrder,numberOfChannels,XTest,YTest)
A = load(filePath,"compactModel","modelMetadata");
if ~isfield(A,"compactModel") || ~isfield(A,"modelMetadata")
    error("Model artifact is missing compactModel or modelMetadata.");
end
model = A.compactModel;
metadata = A.modelMetadata;
required = ["SchemaVersion","ModelName","FeatureNumber","HeldSubject", ...
    "ClassOrder","NumPredictors","TrainingMeta","DatasetHash", ...
    "GitCommit","TestBalancedAccuracy","TestPredictions"];
if ~all(isfield(metadata,required))
    error("Model metadata is incomplete.");
end
if string(metadata.ModelName) ~= string(expectedModelName) || ...
        double(metadata.FeatureNumber) ~= double(expectedFeature) || ...
        double(metadata.HeldSubject) ~= double(expectedSubject)
    error("Model key does not match the requested family/feature/subject.");
end
if lower(string(metadata.DatasetHash)) ~= lower(string(datasetHash))
    error("Persisted model dataset hash does not match the locked dataset.");
end
if string(metadata.GitCommit) ~= string(gitCommit)
    error("Persisted model Git commit does not match the frozen source commit.");
end
if double(metadata.NumPredictors) ~= numberOfChannels || ...
        getModelPredictorCount(model) ~= numberOfChannels
    error("Persisted model does not contain exactly %d predictors.", ...
        numberOfChannels);
end
savedClasses = double(metadata.ClassOrder(:)');
modelClasses = double(model.ClassNames(:)');
if ~isequal(savedClasses,double(classOrder(:)')) || ...
        ~isequal(modelClasses,double(classOrder(:)'))
    error("Persisted model class order does not match the locked class order.");
end
validateModelFamilyClass(model,expectedModelName);
[prediction,scores] = predictExactModel(model,XTest,false);
if numel(prediction) ~= numel(YTest) || any(~isfinite(double(scores)),"all")
    error("Persisted model prediction output is malformed.");
end
reloadedBalanced = balancedAccuracy(YTest,prediction,classOrder);
reloadedOrdinary = mean(double(prediction) == double(YTest));
assertBalancedOrdinaryAgreement(reloadedBalanced,reloadedOrdinary);
if abs(reloadedBalanced-double(metadata.TestBalancedAccuracy)) > 1e-12
    error("Reloaded model does not reproduce its saved balanced accuracy.");
end
if ~isequal(double(prediction(:)),double(metadata.TestPredictions(:)))
    error("Reloaded model predictions differ from the saved validation vector.");
end
end

function count = getModelPredictorCount(model)
if isprop(model,"PredictorNames")
    count = numel(model.PredictorNames);
elseif isprop(model,"NumPredictors")
    count = double(model.NumPredictors);
else
    error("Unable to determine model predictor count for class %s.",class(model));
end
end

function validateModelFamilyClass(model,modelName)
className = string(class(model));
switch string(modelName)
    case "LDA"
        valid = contains(className,"ClassificationDiscriminant");
    case "MLP"
        valid = contains(className,"ClassificationNeuralNetwork");
    case "RBF_SVM"
        valid = contains(className,"ClassificationECOC");
    otherwise
        valid = false;
end
if ~valid
    error("Model class %s is incompatible with declared family %s.", ...
        className,modelName);
end
end

function validatePortablePredictionEquivalence(fitPrediction,fitScores, ...
        portablePrediction,portableScores)
if ~isequal(double(fitPrediction(:)),double(portablePrediction(:)))
    error("Compacting/gathering the model changed held-out predictions.");
end
if ~isequal(size(fitScores),size(portableScores)) || ...
        any(~isfinite(double(portableScores)),"all")
    error("Portable model scores are missing or malformed.");
end
scoreDifference = max(abs(double(fitScores)-double(portableScores)),[],"all");
if scoreDifference > 1e-4
    error("Compacting/gathering changed scores by %.6g.",scoreDifference);
end
end

function assertBalancedOrdinaryAgreement(balancedValue,ordinaryValue)
if ~isfinite(balancedValue) || ~isfinite(ordinaryValue) || ...
        balancedValue < 0 || balancedValue > 1 || ...
        ordinaryValue < 0 || ordinaryValue > 1
    error("Accuracy is nonfinite or outside [0,1].");
end
if abs(balancedValue-ordinaryValue) > 1e-10
    error(["Balanced and ordinary accuracy differ despite the exactly " + ...
        "class-balanced held-out subject."]);
end
end

function enforceComputeModeLock(manifestFolder,modelName,computeMode, ...
        datasetHash,gitCommit)
lockPath = fullfile(manifestFolder,lower(modelName) + ...
    "_compute_mode_lock.mat");
if isfile(lockPath)
    L = load(lockPath,"computeModeLock");
    if ~isfield(L,"computeModeLock")
        error("Compute-mode lock is malformed: %s",lockPath);
    end
    expected = L.computeModeLock;
    if string(expected.ComputeMode) ~= string(computeMode) || ...
            string(expected.DatasetHash) ~= string(datasetHash) || ...
            string(expected.GitCommit) ~= string(gitCommit)
        error(["Compute mode would change inside model family %s. Locked: %s; " + ...
            "current: %s. Use a separate ArtifactRoot after deliberate review."], ...
            modelName,expected.ComputeMode,computeMode);
    end
else
    computeModeLock = struct; %#ok<NASGU>
    computeModeLock.ModelName = string(modelName);
    computeModeLock.ComputeMode = string(computeMode);
    computeModeLock.DatasetHash = string(datasetHash);
    computeModeLock.GitCommit = string(gitCommit);
    computeModeLock.CreatedAt = datetime("now");
    atomicSaveStruct(lockPath,"computeModeLock",computeModeLock);
end
end

function quarantinedPath = quarantineArtifact(filePath,quarantineFolder,reason)
[~,name,extension] = fileparts(filePath);
quarantinedPath = fullfile(quarantineFolder,name + "_" + ...
    string(reason) + "_" + ...
    string(datetime("now","Format","yyyyMMdd_HHmmss")) + "_" + ...
    string(java.util.UUID.randomUUID) + extension);
[success,message] = movefile(filePath,quarantinedPath,"f");
if ~success
    error("Unable to quarantine invalid artifact: %s",message);
end
end

function validateSavedShapArtifact(filePath,modelHash,datasetHash,modelName, ...
        featureNumber,heldSubject,nChannels,nClasses)
S = load(filePath,"foldResult");
if ~isfield(S,"foldResult")
    error("Saved SHAP artifact is missing foldResult.");
end
R = S.foldResult;
required = ["ModelName","FeatureNumber","HeldSubject","DatasetHash", ...
    "ModelSHA256","MeanAbsoluteShapley","NormalizedMeanAbsoluteShapley", ...
    "BackgroundIndices","QueryIndices"];
if ~all(isfield(R,required))
    error("Saved SHAP artifact metadata is incomplete.");
end
if string(R.ModelName) ~= string(modelName) || ...
        double(R.FeatureNumber) ~= double(featureNumber) || ...
        double(R.HeldSubject) ~= double(heldSubject) || ...
        lower(string(R.DatasetHash)) ~= lower(string(datasetHash)) || ...
        lower(string(R.ModelSHA256)) ~= lower(string(modelHash))
    error("Saved SHAP artifact key/hash does not match its model.");
end
validateShapMatrix(double(R.MeanAbsoluteShapley),nChannels,nClasses);
if ~isequal(size(R.NormalizedMeanAbsoluteShapley),[nChannels,nClasses]) || ...
        any(~isfinite(double(R.NormalizedMeanAbsoluteShapley)),"all")
    error("Saved normalized SHAP matrix is malformed.");
end
end

function hash = sha256Numeric(values)
bytes = typecast(double(values(:)),"uint8");
digest = java.security.MessageDigest.getInstance("SHA-256");
digest.update(typecast(bytes,"int8"));
hashBytes = typecast(digest.digest(),"uint8");
hash = lower(join(compose("%02x",hashBytes),""));
end

function value = ternaryString(condition,trueValue,falseValue)
if condition
    value = string(trueValue);
else
    value = string(falseValue);
end
end

function exportRecoveryManifests(state,manifestFolder)
modelRows = strings(0,1);
featureRows = zeros(0,1);
subjectRows = zeros(0,1);
modelFileRows = strings(0,1);
modelHashRows = strings(0,1);
shapFileRows = strings(0,1);
shapHashRows = strings(0,1);
stageRows = strings(0,1);
computeRows = strings(0,1);
shapModeRows = strings(0,1);
balancedRows = zeros(0,1);
ordinaryRows = zeros(0,1);
trainingRows = zeros(0,1);
shapSecondsRows = zeros(0,1);
completedRows = false(0,1);
failedRows = false(0,1);
attemptRows = zeros(0,1);

for modelPosition = 1:numel(state.ModelNames)
    for featureNumber = 1:numel(state.FeatureNames)
        for subjectPosition = 1:numel(state.Subjects)
            touched = state.ModelSaved(modelPosition,featureNumber,subjectPosition) || ...
                state.Completed(modelPosition,featureNumber,subjectPosition) || ...
                state.Failed(modelPosition,featureNumber,subjectPosition) || ...
                state.AttemptCount(modelPosition,featureNumber,subjectPosition) > 0;
            if ~touched
                continue;
            end
            modelRows(end+1,1) = state.ModelNames(modelPosition); %#ok<AGROW>
            featureRows(end+1,1) = featureNumber; %#ok<AGROW>
            subjectRows(end+1,1) = state.Subjects(subjectPosition); %#ok<AGROW>
            modelFileRows(end+1,1) = state.ModelFile( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            modelHashRows(end+1,1) = state.ModelSHA256( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            shapFileRows(end+1,1) = state.ShapFile( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            shapHashRows(end+1,1) = state.ShapSHA256( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            stageRows(end+1,1) = state.Stage( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            computeRows(end+1,1) = state.ComputeMode( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            shapModeRows(end+1,1) = state.ShapMode( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            balancedRows(end+1,1) = state.ReproducedBalancedAccuracy( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            ordinaryRows(end+1,1) = state.ReproducedOrdinaryAccuracy( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            trainingRows(end+1,1) = state.TrainingSeconds( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            shapSecondsRows(end+1,1) = state.ShapSeconds( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            completedRows(end+1,1) = state.Completed( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            failedRows(end+1,1) = state.Failed( ...
                modelPosition,featureNumber,subjectPosition); %#ok<AGROW>
            attemptRows(end+1,1) = double(state.AttemptCount( ...
                modelPosition,featureNumber,subjectPosition)); %#ok<AGROW>
        end
    end
end

T = table(modelRows,featureRows,subjectRows,modelFileRows,modelHashRows, ...
    shapFileRows,shapHashRows,stageRows,computeRows,shapModeRows, ...
    balancedRows,ordinaryRows,trainingRows,shapSecondsRows, ...
    completedRows,failedRows,attemptRows, ...
    'VariableNames',{'ModelFamily','FeatureNumber','HeldSubject', ...
    'ModelFile','ModelSHA256','ShapFile','ShapSHA256','Stage', ...
    'ComputeMode','ShapMode','BalancedAccuracy','OrdinaryAccuracy', ...
    'TrainingSeconds','ShapSeconds','Completed','Failed','AttemptCount'});
T = sortrows(T,{'ModelFamily','FeatureNumber','HeldSubject'});
writetable(T,fullfile(manifestFolder,"recovery_model_and_shap_manifest.csv"));
manifest = struct;
manifest.StateVersion = state.StateVersion;
manifest.DatasetHash = state.DatasetHash;
manifest.GitCommit = state.GitCommit;
manifest.FeatureNames = state.FeatureNames;
manifest.Subjects = state.Subjects;
manifest.ClassOrder = state.ClassOrder;
manifest.ConfigSignature = state.ConfigSignature;
manifest.Table = T;
manifest.ExportedAt = datetime("now");
atomicSaveStruct(fullfile(manifestFolder, ...
    "recovery_model_and_shap_manifest.mat"),"manifest",manifest);
end
