function results = run_single_feature_loso_model(modelName, varargin)
%RUN_SINGLE_FEATURE_LOSO_MODEL Shared sequential, crash-safe locked LOSO engine.
% Optional name/value pairs are operational only: DatasetPath, FeatureNumbers,
% SubjectIDs, OutputRoot, WritePipelineState, and smoke-test controls.

p = inputParser;
addParameter(p, "DatasetPath", "", @(x)ischar(x) || isstring(x));
addParameter(p, "FeatureNumbers", 1:21, @isnumeric);
addParameter(p, "SubjectIDs", [], @isnumeric);
addParameter(p, "OutputRoot", "", @(x)ischar(x) || isstring(x));
addParameter(p, "WritePipelineState", false, @(x)islogical(x) && isscalar(x));
addParameter(p, "SkipSmokeGate", false, @(x)islogical(x) && isscalar(x));
addParameter(p, "ComputeModeOverride", "", @(x)ischar(x) || isstring(x));
parse(p, varargin{:}); opt = p.Results;
modelName = lower(string(modelName));
if ~ismember(modelName, ["lda" "mlp" "rbf_svm"]), error("Unknown model: %s", modelName); end

root = string(fileparts(mfilename("fullpath")));
cfg = jsondecode(fileread(fullfile(root, "single_feature_loso_locked_config.json")));
if strlength(string(opt.DatasetPath)) == 0
    datasetPath = fullfile(root, cfg.datasetFile);
else
    datasetPath = string(opt.DatasetPath);
end
if ~isfile(datasetPath), error("Dataset not found: %s", datasetPath); end
datasetHash = localSha256(datasetPath);
if ~strcmpi(datasetHash, string(cfg.datasetSha256))
    error("Dataset SHA256 mismatch. Expected %s, got %s.", cfg.datasetSha256, datasetHash);
end

D = load(datasetPath); data = localValidateDataset(D, cfg);
features = unique(double(opt.FeatureNumbers(:)'));
if any(features < 1 | features > 21 | mod(features,1) ~= 0), error("FeatureNumbers must be 1..21."); end
if isempty(opt.SubjectIDs), subjectsToRun = data.subjects; else, subjectsToRun = unique(double(opt.SubjectIDs(:)')); end
if ~all(ismember(subjectsToRun, data.subjects)), error("SubjectIDs are not in the locked dataset."); end
isFullScope = isequal(features, 1:21) && isequal(subjectsToRun, data.subjects);

folderNames = struct("lda", "single_feature_lda_loso_results", "mlp", "single_feature_mlp_loso_results", "rbf_svm", "single_feature_rbf_svm_loso_results");
progressNames = struct("lda", "single_feature_lda_loso_progress.mat", "mlp", "single_feature_mlp_loso_progress.mat", "rbf_svm", "single_feature_rbf_svm_loso_progress.mat");
if strlength(string(opt.OutputRoot)) == 0
    if ~isFullScope, error("Partial runs require OutputRoot to isolate artifacts."); end
    outputFolder = fullfile(root, folderNames.(char(modelName)));
else
    outputFolder = string(opt.OutputRoot);
end
if ~isfolder(outputFolder), mkdir(outputFolder); end
progressFile = fullfile(outputFolder, progressNames.(char(modelName)));
logFile = fullfile(outputFolder, "single_feature_" + modelName + "_loso_diary.log");
diary(char(logFile)); cleanupDiary = onCleanup(@() diary("off"));

settings = localSettings(modelName, cfg); settings.model = modelName;
settings.configHash = localSha256(fullfile(root, "single_feature_loso_locked_config.json"));
settings.codeHash = localSha256(string(mfilename("fullpath")) + ".m");
if opt.SkipSmokeGate
    settings.compute = localComputeMode(modelName, data, string(opt.ComputeModeOverride));
else
    settings.compute = localLoadSmokeDecision(root, modelName);
end
lockOwner = struct([]);
if opt.WritePipelineState
    if ~isFullScope
        error("Pipeline state is reserved for full 21-by-18 production runs.");
    end
    initialState = localState("starting", modelName, settings.compute, [], NaN, NaN, ...
        progressFile, logFile, "", "Validate checkpoint and begin the next uncommitted fold");
    lockOwner = loso_pipeline_lock("acquire", root, struct(), initialState);
    cleanupLock = onCleanup(@() loso_pipeline_lock("cleanup", root, lockOwner, struct()));
end
if modelName == "rbf_svm" && settings.compute.mode == "cpu_parallel"
    localEnsureRbfPool(settings.compute);
end
fingerprint = localFingerprint(settings, datasetHash, data);
P = localLoadOrInitialize(progressFile, data, settings, fingerprint, datasetPath, datasetHash);
if ~isempty(lockOwner)
    loso_pipeline_lock("update", root, lockOwner, ...
        localState("running", modelName, settings.compute, P, NaN, NaN, progressFile, logFile, "", ...
        "Train the next uncommitted fold"));
end
fprintf("%s LOSO: %d feature(s), %d subject(s), compute=%s\n", upper(modelName), numel(features), numel(subjectsToRun), settings.compute.mode);

for f = features
    predictorMask = data.predictorFeatureNumber == f;
    for heldSubject = subjectsToRun
        s = find(data.subjects == heldSubject, 1);
        if P.completed(f,s), fprintf("feature %d subject %d: checkpoint found\n", f, heldSubject); continue; end
        [trainMask, testMask] = localValidateFold(data, heldSubject);
        seed = settings.seedBase + 100*f + heldSubject;
        rng(seed, "twister");
        XTrain = data.X(trainMask,predictorMask); YTrain = data.Y(trainMask);
        XTest = data.X(testMask,predictorMask); YTest = data.Y(testMask);
        if ~isempty(lockOwner)
            loso_pipeline_lock("update", root, lockOwner, ...
                localState("running", modelName, settings.compute, P, f, heldSubject, progressFile, logFile, "", ...
                "Train, predict, validate, and checkpoint this fold"));
        end
        try
            [prediction, trainTime, predictTime] = localFitPredict(modelName, XTrain, YTrain, XTest, data.classOrder, settings, seed);
            [ba, oa, recall, counts] = localMetrics(prediction, YTest, data.classOrder);
            localValidateFoldMetrics(ba, oa, recall, counts, YTest);
            P.balancedAccuracy(f,s) = ba; P.ordinaryAccuracy(f,s) = oa;
            P.trainingTimeSeconds(f,s) = trainTime; P.predictionTimeSeconds(f,s) = predictTime;
            P.classRecall(f,s,:) = recall; P.confusionCounts(f,s,:,:) = counts;
            P.completed(f,s) = true; P.generation = P.generation + 1; P.lastFold = [f heldSubject]; P.timestampUtc = char(datetime("now", "TimeZone", "UTC"));
            localCheckpoint(progressFile, P, data, settings, fingerprint);
            if ~isempty(lockOwner)
                loso_pipeline_lock("update", root, lockOwner, ...
                    localState("running", modelName, settings.compute, P, f, heldSubject, progressFile, logFile, "", ...
                    "Continue with the next uncommitted fold"));
            end
            fprintf("feature %d subject %d: BA %.2f%% train %.2fs predict %.2fs\n", f, heldSubject, 100*ba, trainTime, predictTime);
        catch ME
            if ~isempty(lockOwner)
                loso_pipeline_lock("fail", root, lockOwner, ...
                    localState("failed", modelName, settings.compute, P, f, heldSubject, progressFile, logFile, ...
                    ME.message, "Inspect the model log and resume from the validated checkpoint"));
            end
            rethrow(ME)
        end
    end
end
results = localWriteOutputs(P, data, settings, outputFolder, modelName, isFullScope);
if ~isempty(lockOwner)
    loso_pipeline_lock("complete", root, lockOwner, ...
        localState("completed", modelName, settings.compute, P, P.lastFold(1), P.lastFold(2), ...
        progressFile, logFile, "", "Advance to the next model in the locked sequence"));
end
end

function data = localValidateDataset(D, cfg)
need = ["X" "Y" "subjectID" "featureNames" "predictorInformation"];
if ~all(ismember(need, string(fieldnames(D)))), error("Dataset missing required variables."); end
data.X = D.X; data.Y = double(D.Y(:)); data.subjectID = double(D.subjectID(:)); data.featureNames = string(D.featureNames(:));
if ~isnumeric(data.X) || ~isequal(size(data.X), [cfg.expectedDimensions.rows cfg.expectedDimensions.predictors]) || any(~isfinite(data.X), "all"), error("X must be finite 27360x2688 numeric."); end
if numel(data.Y) ~= size(data.X,1) || numel(data.subjectID) ~= size(data.X,1) || any(~isfinite(data.Y)) || any(~isfinite(data.subjectID)) || any(mod(data.Y,1)) || any(mod(data.subjectID,1)), error("Y and subjectID must be finite integer and row-aligned."); end
data.subjects = unique(data.subjectID(:))'; data.classOrder = unique(data.Y(:))'; data.predictorFeatureNumber = double(D.predictorInformation.FeatureNumber(:));
if numel(data.subjects) ~= 18 || ~isequal(data.subjects, 1:18) || numel(data.classOrder) ~= 8 || ~isequal(data.classOrder, 1:8) || numel(data.featureNames) ~= 21 || numel(data.predictorFeatureNumber) ~= 2688 || any(strlength(data.featureNames)==0) || numel(unique(data.featureNames))~=21, error("Subjects/classes/features/predictor mapping dimensions are not locked values."); end
columns=["Predictor" "FeatureNumber" "FeatureName" "Channel"]; if ~all(ismember(columns,string(D.predictorInformation.Properties.VariableNames))), error("predictorInformation columns are incomplete."); end
pi=D.predictorInformation; if ~isequal(sort(double(pi.Predictor(:)))',(1:2688)) || any(double(pi.FeatureNumber(:))~=data.predictorFeatureNumber) || ~isequal(string(pi.FeatureName(:)),data.featureNames(data.predictorFeatureNumber)) || any(arrayfun(@(f) ~isequal(sort(double(pi.Channel(data.predictorFeatureNumber==f)))',(1:128)),1:21)), error("predictorInformation mapping is not exact."); end
if any(arrayfun(@(f) sum(data.predictorFeatureNumber == f), 1:21) ~= 128), error("Each feature must map to exactly 128 predictors."); end
for s = data.subjects
    rows = data.subjectID == s;
    if sum(rows) ~= 1520 || any(arrayfun(@(c) sum(data.Y(rows) == c), data.classOrder) ~= 190), error("Subject %d has unexpected counts/classes.", s); end
end
end

function [trainMask, testMask] = localValidateFold(data, heldSubject)
trainMask = data.subjectID ~= heldSubject; testMask = data.subjectID == heldSubject;
if any(trainMask & testMask) || ~all(trainMask | testMask) || sum(testMask) ~= 1520 || sum(trainMask) ~= 25840 || any(arrayfun(@(c)sum(data.Y(trainMask)==c),data.classOrder)~=3230) || any(arrayfun(@(c)sum(data.Y(testMask)==c),data.classOrder)~=190), error("Invalid LOSO partition for subject %d.", heldSubject); end
end

function settings = localSettings(modelName, cfg)
settings = cfg.models.(char(modelName)); settings.cacheSizeMB = 512;
end

function compute = localComputeMode(modelName, data, requestedMode)
compute = struct("mode", "cpu", "detail", "CPU sequential", "parallelWorkers", 0);
if strlength(requestedMode)>0
    compute.mode=requestedMode; compute.detail="smoke candidate";
    if modelName=="rbf_svm" && requestedMode=="cpu_parallel"
        try
            pool=gcp("nocreate"); if isempty(pool), pool=parpool("local",min(6,feature("numcores"))); end
            if pool.NumWorkers>6, compute.mode="cpu"; compute.detail="CPU serial fallback: existing pool exceeds six workers"; else, compute.parallelWorkers=pool.NumWorkers; end
        catch ME
            compute.mode="cpu"; compute.detail="CPU serial fallback: "+string(ME.message);
        end
    end
    return
end
if modelName == "mlp"
    try
        if license("test", "Distrib_Computing_Toolbox") && gpuDeviceCount("available") > 0
            g = gpuDevice; probe = gpuArray(data.X(1:2,1:2)); gather(probe); wait(g);
            compute.mode = "gpu"; compute.detail = string(g.Name);
        end
    catch ME
        compute.detail = "CPU fallback after GPU probe: " + string(ME.message);
    end
elseif modelName == "rbf_svm"
    % fitcecoc GPU support is release-dependent; probe gpuArray only, then use CPU safely.
    try
        probe = gpuArray(data.X(1:2,1:2)); gather(probe);
        compute.detail = "CPU; gpuArray probe passed, fitcecoc remains CPU";
    catch ME
        compute.detail = "CPU fallback after GPU probe: " + string(ME.message);
    end
    if license("test", "Distrib_Computing_Toolbox")
        try
            pool = gcp("nocreate"); if isempty(pool), pool = parpool("local", min(6, feature("numcores"))); end
            if pool.NumWorkers>6, compute.mode="cpu"; compute.detail="CPU serial fallback: existing pool exceeds six workers"; else, compute.mode = "cpu_parallel"; compute.parallelWorkers = pool.NumWorkers; compute.detail = "CPU parallel reproducible streams; outer folds sequential"; end
        catch ME
            compute.detail = "CPU sequential fallback: " + string(ME.message);
        end
    end
end
end

function compute = localLoadSmokeDecision(root, modelName)
decisionFile=fullfile(root,"single_feature_loso_smoke_tests","compute_decisions.json");
if ~isfile(decisionFile), error("No smoke compute decision exists. Run run_single_feature_loso_smoke_test('%s') before full/benchmark execution.",modelName); end
S=jsondecode(fileread(decisionFile)); if ~isfield(S,char(modelName)) || ~S.(char(modelName)).success, error("No successful compatible smoke decision exists for %s.",modelName); end
decision=S.(char(modelName)); cfgPath=fullfile(root,"single_feature_loso_locked_config.json"); cfg=jsondecode(fileread(cfgPath)); datasetPath=fullfile(root,cfg.datasetFile); codePath=fullfile(root,"run_single_feature_loso_model.m");
required=["datasetHash" "configHash" "codeHash" "matlabRelease" "compute"];
if ~all(isfield(decision,required)) || ~strcmpi(string(decision.datasetHash),localSha256(datasetPath)) || ~strcmpi(string(decision.configHash),localSha256(cfgPath)) || ~strcmpi(string(decision.codeHash),localSha256(codePath)) || string(decision.matlabRelease)~=string(version("-release"))
    error("Smoke compute decision for %s is stale or incompatible with the current dataset/config/code/release.",modelName);
end
compute=decision.compute; compute.mode=string(compute.mode); compute.detail=string(compute.detail);
if ~ismember(compute.mode,["cpu" "gpu" "cpu_parallel"]), error("Smoke compute decision has an invalid mode."); end
compute.smokeEvidence=decision;
end

function localEnsureRbfPool(compute)
workers=double(compute.parallelWorkers);
if ~isscalar(workers) || ~isfinite(workers) || workers<1 || workers>6 || mod(workers,1)~=0
    error("The smoke-frozen RBF worker count must be an integer from 1 through 6.");
end
pool=gcp("nocreate");
if isempty(pool)
    parpool("Processes",workers);
elseif pool.NumWorkers~=workers
    error("Existing parallel pool has %d workers; the smoke-frozen RBF mode requires %d.",pool.NumWorkers,workers);
end
end

function [prediction, trainTime, predictTime] = localFitPredict(modelName, XTrain, YTrain, XTest, classOrder, settings, seed)
if modelName == "lda"
    t = tic; model = fitcdiscr(XTrain,YTrain,"DiscrimType",settings.discrimType,"Prior","uniform","ClassNames",classOrder); trainTime = toc(t);
    t = tic; prediction = predict(model,XTest); predictTime = toc(t);
elseif modelName == "mlp"
    if settings.compute.mode == "gpu", XTrain = gpuArray(XTrain); XTest = gpuArray(XTest); g=gpuDevice; wait(g); end
    t = tic; model = fitcnet(XTrain,YTrain,"LayerSizes",settings.layerSizes,"Activations",settings.activations,"Lambda",settings.lambda,"IterationLimit",settings.iterationLimit,"Standardize",true,"Prior","uniform","ClassNames",classOrder); if settings.compute.mode=="gpu",wait(g);end; trainTime = toc(t);
    t = tic; prediction = predict(model,XTest); if settings.compute.mode=="gpu",wait(g);end; prediction = gather(prediction); predictTime = toc(t);
else
    learner = templateSVM("KernelFunction",settings.kernelFunction,"KernelScale",settings.kernelScale,"BoxConstraint",settings.boxConstraint,"Solver",settings.solver,"Standardize",true,"CacheSize",settings.cacheSizeMB);
    if settings.compute.mode == "gpu", XTrain=gpuArray(XTrain); XTest=gpuArray(XTest); g=gpuDevice; wait(g); end
    if settings.compute.mode == "cpu_parallel"
        try
            streams=RandStream("Threefry","Seed",seed);
            options=statset("UseParallel",true,"Streams",streams,"UseSubstreams",true);
        catch ME
            error("RBF parallel reproducible stream configuration unavailable; CPU serial is required: %s",ME.message);
        end
    else
        options = statset("UseParallel", false);
    end
    t = tic; model = fitcecoc(XTrain,YTrain,"Learners",learner,"Coding",settings.coding,"Prior","uniform","ClassNames",classOrder,"Options",options); if settings.compute.mode=="gpu",wait(g);end; trainTime = toc(t);
    t = tic; prediction = predict(model,XTest); if settings.compute.mode=="gpu",wait(g);end; prediction=gather(prediction); predictTime = toc(t);
end
prediction = double(prediction(:));
end

function [ba, oa, recall, counts] = localMetrics(prediction, actual, classes)
counts = confusionmat(actual, prediction, "Order", classes); recall = diag(counts) ./ sum(counts,2); ba = mean(recall); oa = mean(prediction == actual);
end

function localValidateFoldMetrics(ba, oa, recall, counts, actual)
if ~isscalar(ba) || ~isfinite(ba) || ba<0 || ba>1 || ~isscalar(oa) || ~isfinite(oa) || oa<0 || oa>1 || ~isequal(size(recall),[8 1]) || any(~isfinite(recall)) || any(recall<0 | recall>1) || ~isequal(size(counts),[8 8]) || any(~isfinite(counts),"all") || any(counts<0,"all") || any(mod(counts,1),"all") || any(sum(counts,2)~=190) || sum(counts,"all")~=1520 || numel(actual)~=1520, error("Fold metric validation failed; checkpoint was not marked complete."); end
recomputed=diag(counts)./sum(counts,2); if abs(ba-mean(recomputed))>1e-12 || abs(oa-trace(counts)/1520)>1e-12, error("Fold metrics disagree with confusion counts."); end
if max(abs(recall-recomputed))>1e-12 || abs(ba-oa)>1e-12, error("Recall or balanced/ordinary accuracy integrity check failed."); end
end

function P = localLoadOrInitialize(file, data, settings, fingerprint, datasetPath, datasetHash)
if isfile(file)
    try
        S = load(file); if ~isfield(S,"checkpoint"), error("missing checkpoint"); end
        P = S.checkpoint; localValidateCheckpoint(P, data, settings, fingerprint); return
    catch
        localPreserveCorrupt(file);
    end
end
[folder,base,~]=fileparts(file); d=dir(fullfile(folder,base+"*.mat")); valid=struct("P",{},"generation",{});
for k=1:numel(d)
    candidate=fullfile(d(k).folder,d(k).name); if strcmp(candidate,file), continue; end
    try
        S=load(candidate); if ~isfield(S,"checkpoint"), error("missing checkpoint"); end
        localValidateCheckpoint(S.checkpoint,data,settings,fingerprint); valid(end+1)=struct("P",S.checkpoint,"generation",S.checkpoint.generation); %#ok<AGROW>
    catch
        localPreserveCorrupt(candidate);
    end
end
if ~isempty(valid), [~,ix]=max([valid.generation]); P=valid(ix).P; localRestoreCheckpoint(file,P,data,settings,fingerprint); return; end
if isfile(file) || ~isempty(d), error("No compatible checkpoint recovery candidate exists; artifacts were preserved."); end
if ~isfile(file)
    P = struct("schemaVersion","1.0","completed",false(21,18),"balancedAccuracy",nan(21,18),"ordinaryAccuracy",nan(21,18),"trainingTimeSeconds",nan(21,18),"predictionTimeSeconds",nan(21,18),"classRecall",nan(21,18,8),"confusionCounts",zeros(21,18,8,8),"featureNames",data.featureNames,"subjects",data.subjects,"classOrder",data.classOrder,"settings",settings,"fingerprint",fingerprint,"datasetPath",char(datasetPath),"datasetHash",char(datasetHash),"matlabRelease",version("-release"),"generation",0,"timestampUtc",char(datetime("now","TimeZone","UTC")),"lastFold",[NaN NaN]);
end
end

function localValidateCheckpoint(P, data, settings, fingerprint)
need = ["schemaVersion" "completed" "balancedAccuracy" "ordinaryAccuracy" "trainingTimeSeconds" "predictionTimeSeconds" "classRecall" "confusionCounts" "featureNames" "subjects" "classOrder" "settings" "fingerprint"];
if ~all(isfield(P,need)) || ~strcmp(string(P.schemaVersion),"1.0") || ~islogical(P.completed) || ~isequal(size(P.completed),[21 18]) || ~isequal(size(P.balancedAccuracy),[21 18]) || ~isequal(size(P.ordinaryAccuracy),[21 18]) || ~isequal(size(P.trainingTimeSeconds),[21 18]) || ~isequal(size(P.predictionTimeSeconds),[21 18]) || ~isequal(size(P.classRecall),[21 18 8]) || ~isequal(size(P.confusionCounts),[21 18 8 8]) || ~isequaln(P.settings,settings) || ~strcmp(string(P.fingerprint),string(fingerprint)) || ~isequal(string(P.featureNames),data.featureNames) || ~isequal(P.subjects,data.subjects) || ~isequal(P.classOrder,data.classOrder) || ~isfield(P,"generation") || ~isscalar(P.generation) || ~isfinite(P.generation) || P.generation<nnz(P.completed) || mod(P.generation,1)~=0 || ~isfield(P,"lastFold") || ~isequal(size(P.lastFold),[1 2]) || any(~isnan(P.lastFold)&(~isfinite(P.lastFold)|mod(P.lastFold,1)~=0)), error("Progress checkpoint is incompatible; it has not been overwritten."); end
for f=1:21
    for s=1:18
        if P.completed(f,s)
            localValidateFoldMetrics(P.balancedAccuracy(f,s),P.ordinaryAccuracy(f,s), ...
                squeeze(P.classRecall(f,s,:)),squeeze(P.confusionCounts(f,s,:,:)),zeros(1520,1));
            if ~isfinite(P.trainingTimeSeconds(f,s)) || P.trainingTimeSeconds(f,s)<0 || ...
                    ~isfinite(P.predictionTimeSeconds(f,s)) || P.predictionTimeSeconds(f,s)<0
                error("Completed checkpoint timing is invalid.");
            end
        elseif ~isnan(P.balancedAccuracy(f,s)) || ~isnan(P.ordinaryAccuracy(f,s)) || ...
                ~isnan(P.trainingTimeSeconds(f,s)) || ~isnan(P.predictionTimeSeconds(f,s)) || ...
                any(~isnan(squeeze(P.classRecall(f,s,:)))) || any(squeeze(P.confusionCounts(f,s,:,:))~=0,"all")
            error("Incomplete checkpoint cell is not empty.");
        end
    end
end
end

function localCheckpoint(file, P, data, settings, fingerprint)
localValidateCheckpoint(P,data,settings,fingerprint); checkpoint=P; folder=fileparts(file); temp=string(tempname(folder))+".mat"; save(temp,"checkpoint","-v7.3"); S=load(temp); localValidateCheckpoint(S.checkpoint,data,settings,fingerprint);
if isfile(file)
 old=load(file); localValidateCheckpoint(old.checkpoint,data,settings,fingerprint); backup=file+".bak.mat"; btemp=string(tempname(folder))+".mat"; checkpoint=old.checkpoint; save(btemp,"checkpoint","-v7.3"); B=load(btemp); localValidateCheckpoint(B.checkpoint,data,settings,fingerprint); if isfile(backup),copyfile(backup,backup+".archive_"+string(datetime("now","Format","yyyyMMdd_HHmmssSSS")));end; localAtomicReplace(btemp,backup);
end
localAtomicReplace(temp,file);
end

function localRestoreCheckpoint(file,P,data,settings,fingerprint)
localValidateCheckpoint(P,data,settings,fingerprint); checkpoint=P; temp=string(tempname(fileparts(file)))+".mat"; save(temp,"checkpoint","-v7.3"); S=load(temp); localValidateCheckpoint(S.checkpoint,data,settings,fingerprint); localAtomicReplace(temp,file);
end

function results = localWriteOutputs(P, data, settings, folder, modelName, isFullScope)
featureIndex = find(any(P.completed,2));
subjectIndex = find(any(P.completed,1));
meanBA = mean(P.balancedAccuracy(featureIndex,subjectIndex),2,"omitnan");
[score,order] = sort(meanBA,"descend");
rankedFeatures = featureIndex(order);
ranking = table((1:numel(rankedFeatures))',rankedFeatures(:),reshape(data.featureNames(rankedFeatures),[],1),100*score(:), ...
    VariableNames=["Rank","FeatureNumber","FeatureName","MeanBalancedAccuracyPercent"]);
bySubject = array2table(100*P.balancedAccuracy(featureIndex,subjectIndex), ...
    VariableNames=compose("Subject_%02d",data.subjects(subjectIndex)));
bySubject = addvars(bySubject,featureIndex(:),reshape(data.featureNames(featureIndex),[],1), ...
    Before=1,NewVariableNames=["FeatureNumber","FeatureName"]);
results = struct("checkpoint",P,"ranking",ranking,"bySubject",bySubject,"settings",settings);
if isFullScope && all(P.completed,"all")
    balancedAccuracy=P.balancedAccuracy; ordinaryAccuracy=P.ordinaryAccuracy; trainingTimeSeconds=P.trainingTimeSeconds; predictionTimeSeconds=P.predictionTimeSeconds; classRecall=P.classRecall; confusionCounts=P.confusionCounts; completed=P.completed; featureNames=P.featureNames; featureNumbers=(1:21)'; subjects=P.subjects; classOrder=P.classOrder; datasetHash=P.datasetHash; matlabRelease=P.matlabRelease; computeMode=char(P.settings.compute.mode); computeDetails=P.settings.compute; computeFingerprint=P.fingerprint; modelSettings=P.settings;
    save(fullfile(folder,"single_feature_"+modelName+"_loso_final.mat"),"results","balancedAccuracy","ordinaryAccuracy","trainingTimeSeconds","predictionTimeSeconds","classRecall","confusionCounts","completed","featureNames","featureNumbers","subjects","classOrder","datasetHash","matlabRelease","computeMode","computeDetails","computeFingerprint","modelSettings","-v7.3"); writetable(ranking,fullfile(folder,"single_feature_"+modelName+"_loso_ranking.csv")); writetable(bySubject,fullfile(folder,"single_feature_"+modelName+"_loso_by_subject.csv"));
    timing=table(featureNumbers,mean(trainingTimeSeconds,2),mean(predictionTimeSeconds,2),VariableNames=["FeatureNumber","MeanTrainingSeconds","MeanPredictionSeconds"]); writetable(timing,fullfile(folder,"single_feature_"+modelName+"_loso_timing.csv"));
    [ff,ss,cc]=ndgrid(1:21,1:18,1:8); heldSubjects=reshape(subjects(ss(:)),[],1); recalls=table(ff(:),heldSubjects,cc(:),reshape(P.classRecall,[],1),VariableNames=["FeatureNumber","HeldSubject","Class","Recall"]); writetable(recalls,fullfile(folder,"single_feature_"+modelName+"_loso_class_recall.csv"));
    [ff,ss,aa,pp]=ndgrid(1:21,1:18,1:8,1:8); heldSubjects=reshape(subjects(ss(:)),[],1); confusions=table(ff(:),heldSubjects,aa(:),pp(:),reshape(P.confusionCounts,[],1),VariableNames=["FeatureNumber","HeldSubject","ActualClass","PredictedClass","Count"]); writetable(confusions,fullfile(folder,"single_feature_"+modelName+"_loso_confusion_counts.csv"));
    fid=fopen(fullfile(folder,"single_feature_"+modelName+"_loso_warnings_deviations.txt"),"w");fprintf(fid,"Compute mode: %s\nDetail: %s\n",computeMode,computeDetails.detail);fclose(fid);
    fid = fopen(fullfile(folder,"single_feature_"+modelName+"_loso_settings_provenance.json"),"w");
    fprintf(fid,"%s\n",jsonencode(struct("settings",settings,"datasetHash",P.datasetHash,"datasetPath",P.datasetPath,"matlabRelease",P.matlabRelease,"timestampUtc",P.timestampUtc))); fclose(fid);
    fig=figure("Visible","off","Color","white"); barh(100*score); yticks(1:numel(score)); yticklabels(data.featureNames(rankedFeatures)); set(gca,"YDir","reverse"); xlabel("Mean LOSO balanced accuracy (%)"); exportgraphics(fig,fullfile(folder,"single_feature_"+modelName+"_loso_ranking.png"),"Resolution",300); close(fig);
end
end

function localPreserveCorrupt(file), if isfile(file), copyfile(file,file+".corrupt_"+string(datetime("now","Format","yyyyMMdd_HHmmssSSS"))+".mat"); end, end
function localAtomicReplace(temp,dest)
try
    options=javaArray('java.nio.file.CopyOption',2);
    options(1)=java.nio.file.StandardCopyOption.ATOMIC_MOVE;
    options(2)=java.nio.file.StandardCopyOption.REPLACE_EXISTING;
    java.nio.file.Files.move(java.io.File(char(temp)).toPath,java.io.File(char(dest)).toPath,options);
catch
    movefile(temp,dest,"f");
end
end
function h = localSha256(file)
% Keep the bytes inside Java. MATLAB numeric arrays passed to
% FileInputStream.read(byte[]) are copied, so hashing that MATLAB buffer can
% silently hash zeros rather than the file contents.
md = java.security.MessageDigest.getInstance("SHA-256");
in = java.io.FileInputStream(java.io.File(char(file)));
cleanup = onCleanup(@() in.close);
channel = in.getChannel();
buffer = java.nio.ByteBuffer.allocateDirect(1048576);
while channel.read(buffer) >= 0
    buffer.flip();
    md.update(buffer);
    buffer.clear();
end
h = upper(reshape(dec2hex(typecast(md.digest(),"uint8"),2)',1,[]));
clear cleanup
end
function f = localFingerprint(settings, hash, data), f = char(join([string(settings.model) string(settings.configHash) string(settings.codeHash) string(settings.compute.mode) string(hash) string(numel(data.Y))],"|")); end
function state=localState(phase,model,compute,P,featureNumber,subject,checkpoint,logFile,blocker,nextAction)
completedFolds=0; checkpointTimestamp="";
if ~isempty(P), completedFolds=nnz(P.completed); checkpointTimestamp=string(P.timestampUtc); end
state=struct("phase",char(phase),"model",char(model),"compute_mode",char(compute.mode), ...
    "completed_folds",completedFolds,"total_folds",378,"last_feature",featureNumber, ...
    "last_held_subject",subject,"progress_file",char(checkpoint),"console_log",char(logFile), ...
    "last_checkpoint_timestamp",char(checkpointTimestamp),"blocking_issue",char(string(blocker)), ...
    "next_action",char(string(nextAction)),"timestamp_utc",char(datetime("now","TimeZone","UTC")));
end
