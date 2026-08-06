function report = verify_published_recovery_models(modelRoot)
%VERIFY_PUBLISHED_RECOVERY_MODELS Independently audit published model MATs.

arguments
    modelRoot (1,1) string
end

expectedArtifacts = 1134;
expectedPerFamily = 378;
expectedDatasetHash = ...
    "8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719";
expectedGitCommit = "e568d39a07873e52f049a827cfb65686fb308893";
expectedPredictors = 128;
expectedClasses = 1:8;
families = ["LDA","MLP","RBF_SVM"];
forbiddenFields = ["DatasetFile","ScriptFile","HeldSubject", ...
    "TrainingSubjects","TrainingRowIndexSHA256","TestRowIndexSHA256", ...
    "TestPredictions","TestPredictionSHA256","TestScoresSHA256", ...
    "ConfusionMatrix","TestBalancedAccuracy","TestOrdinaryAccuracy", ...
    "TrainingRowCount","TestRowCount"];

modelRoot = canonicalPath(modelRoot);
manifestFile = fullfile(modelRoot,"published_model_manifest.csv");
if ~isfolder(modelRoot) || ~isfile(manifestFile)
    error("Published model root or manifest is missing.");
end
manifest = readtable(manifestFile,"Delimiter",",","TextType","string");
if height(manifest) ~= expectedArtifacts || ~all(manifest.Passed)
    error("Published model manifest is incomplete.");
end
temporaryFiles = dir(fullfile(modelRoot,"**","*.tmp_*.mat"));
if ~isempty(temporaryFiles)
    error("Temporary published model files remain after export.");
end

rows = repmat(struct( ...
    "ModelFamily","", "FeatureNumber",NaN, "FoldNumber",NaN, ...
    "ModelFile","", "SHA256","", "ModelClass","", ...
    "FileBytes",NaN, "PredictionProbeRows",NaN, "Passed",false), ...
    expectedArtifacts,1);

for rowPosition = 1:height(manifest)
    manifestRow = manifest(rowPosition,:);
    family = string(manifestRow.ModelFamily);
    featureNumber = double(manifestRow.FeatureNumber);
    foldNumber = double(manifestRow.FoldNumber);
    modelFile = canonicalPath(fullfile(modelRoot,manifestRow.ModelFile));
    if ~startsWith(lower(modelFile + filesep), ...
            lower(modelRoot + filesep)) || ~isfile(modelFile)
        error("Published manifest references an invalid path: %s",modelFile);
    end
    actualHash = sha256File(modelFile);
    if actualHash ~= lower(manifestRow.PublishedSHA256)
        error("Published model SHA-256 mismatch: %s",modelFile);
    end

    payload = load(modelFile,"compactModel","publishedModelMetadata");
    if ~isfield(payload,"compactModel") || ...
            ~isfield(payload,"publishedModelMetadata")
        error("Published model payload is incomplete: %s",modelFile);
    end
    model = payload.compactModel;
    metadata = payload.publishedModelMetadata;
    requiredFields = ["SchemaVersion","ArtifactType","RecoveryCohort", ...
        "IsOriginalHistoricalObject","ModelName","FeatureNumber", ...
        "FeatureName","FoldNumber","ClassOrder","NumPredictors", ...
        "PredictorIndices","Seed","TrainingSettings", ...
        "TrainingComputeMode","DatasetSHA256","SourceModelSHA256", ...
        "SourceInventorySHA256","SourceGitCommit", ...
        "RecoveryScriptSHA256","MATLABRelease","ModelClass", ...
        "PrivacyTransformation"];
    if ~all(isfield(metadata,requiredFields)) || ...
            any(isfield(metadata,forbiddenFields))
        error("Published metadata field policy failed: %s",modelFile);
    end
    if string(metadata.ModelName) ~= family || ...
            double(metadata.FeatureNumber) ~= featureNumber || ...
            double(metadata.FoldNumber) ~= foldNumber || ...
            lower(string(metadata.DatasetSHA256)) ~= expectedDatasetHash || ...
            string(metadata.SourceGitCommit) ~= expectedGitCommit || ...
            lower(string(metadata.SourceModelSHA256)) ~= ...
                lower(manifestRow.SourceSHA256) || ...
            double(metadata.NumPredictors) ~= expectedPredictors || ...
            numel(metadata.PredictorIndices) ~= expectedPredictors || ...
            modelPredictorCount(model) ~= expectedPredictors || ...
            ~isequal(double(metadata.ClassOrder(:)'),expectedClasses) || ...
            ~isequal(double(model.ClassNames(:)'),expectedClasses)
        error("Published model key/provenance failed: %s",modelFile);
    end
    validateFamilyClass(model,family,modelFile);
    assertNoPrivateText(metadata,modelFile);

    probe = reshape(cos((1:8*expectedPredictors)/19),8,expectedPredictors);
    [prediction,scores] = predict(model,probe);
    if numel(prediction) ~= 8 || ...
            ~isequal(size(scores),[8 numel(expectedClasses)]) || ...
            any(~isfinite(double(scores)),"all")
        error("Published model prediction probe failed: %s",modelFile);
    end

    fileInfo = dir(modelFile);
    rows(rowPosition).ModelFamily = family;
    rows(rowPosition).FeatureNumber = featureNumber;
    rows(rowPosition).FoldNumber = foldNumber;
    rows(rowPosition).ModelFile = string(manifestRow.ModelFile);
    rows(rowPosition).SHA256 = actualHash;
    rows(rowPosition).ModelClass = string(class(model));
    rows(rowPosition).FileBytes = fileInfo.bytes;
    rows(rowPosition).PredictionProbeRows = 8;
    rows(rowPosition).Passed = true;
end

report = struct2table(rows);
if height(report) ~= expectedArtifacts || ~all(report.Passed)
    error("Independent published-model verification is incomplete.");
end
for family = families
    if nnz(report.ModelFamily == family) ~= expectedPerFamily
        error("Published family count failed for %s.",family);
    end
end

fprintf("Independent published-model verification passed: %d/%d.\n", ...
    height(report),expectedArtifacts);
end

function assertNoPrivateText(value,modelFile)
textValues = collectText(value);
lowerText = lower(textValues);
forbiddenPatterns = ["c:\\users", "c:/users", "/users/", ...
    "onedrive", "azaan", "testpredictions", "trainingsubjects", ...
    "confusionmatrix"];
for pattern = forbiddenPatterns
    if any(contains(lowerText,pattern))
        error("Published metadata contains forbidden private text: %s", ...
            modelFile);
    end
end
end

function values = collectText(value)
values = strings(0,1);
if isstring(value) || ischar(value)
    values = string(value(:));
elseif iscategorical(value)
    values = string(value(:));
elseif isstruct(value)
    names = fieldnames(value);
    for elementPosition = 1:numel(value)
        for namePosition = 1:numel(names)
            values = [values; collectText( ...
                value(elementPosition).(names{namePosition}))]; %#ok<AGROW>
        end
    end
elseif iscell(value)
    for elementPosition = 1:numel(value)
        values = [values; collectText(value{elementPosition})]; %#ok<AGROW>
    end
end
end

function count = modelPredictorCount(model)
if isprop(model,"PredictorNames") && ~isempty(model.PredictorNames)
    count = numel(model.PredictorNames);
elseif isprop(model,"NumPredictors")
    count = double(model.NumPredictors);
else
    error("Unable to determine predictor count for %s.",class(model));
end
end

function validateFamilyClass(model,family,modelFile)
className = string(class(model));
switch family
    case "LDA"
        valid = contains(className,"Discriminant","IgnoreCase",true);
    case "MLP"
        valid = contains(className,"NeuralNetwork","IgnoreCase",true) || ...
            contains(className,"ClassificationNet","IgnoreCase",true);
    case "RBF_SVM"
        valid = contains(className,"ECOC","IgnoreCase",true);
    otherwise
        valid = false;
end
if ~valid
    error("Unexpected %s model class %s: %s",family,className,modelFile);
end
end

function pathValue = canonicalPath(pathValue)
pathValue = string(java.io.File(char(pathValue)).getCanonicalPath());
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
