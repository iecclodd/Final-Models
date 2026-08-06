function manifest = export_recovery_models_for_github( ...
        sourceRoot,destinationRoot,modelInventoryFile,options)
%EXPORT_RECOVERY_MODELS_FOR_GITHUB Publish privacy-minimized model copies.
%   Preserves each fitted compactModel exactly while replacing the local
%   recovery metadata with a publication-safe provenance record. Every
%   destination MAT file is saved temporarily, reloaded, prediction-probed,
%   atomically promoted, hashed, and recorded in a CSV manifest.

arguments
    sourceRoot (1,1) string
    destinationRoot (1,1) string
    modelInventoryFile (1,1) string
    options.MaxArtifacts (1,1) double {mustBePositive,mustBeInteger} = 1134
    options.CheckpointEvery (1,1) double {mustBePositive,mustBeInteger} = 18
end

expectedDatasetHash = ...
    "8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719";
expectedGitCommit = "e568d39a07873e52f049a827cfb65686fb308893";
expectedArtifacts = 1134;
expectedPredictors = 128;
expectedClasses = 1:8;

sourceRoot = canonicalPath(sourceRoot);
destinationRoot = canonicalPath(destinationRoot);
modelInventoryFile = canonicalPath(modelInventoryFile);
if ~isfolder(sourceRoot) || ~isfile(modelInventoryFile)
    error("Source model root or independent inventory is missing.");
end
if startsWith(lower(destinationRoot + filesep), ...
        lower(sourceRoot + filesep))
    error("Destination must not be inside the immutable source model tree.");
end
if ~isfolder(destinationRoot)
    mkdir(destinationRoot);
end

inventory = readtable(modelInventoryFile,"Delimiter",",", ...
    "TextType","string");
if height(inventory) ~= expectedArtifacts || ~all(inventory.Passed)
    error("Independent source-model inventory is incomplete.");
end
inventory = sortrows(inventory, ...
    {'ModelFamily','FeatureNumber','HeldSubject'});
numberToExport = min(height(inventory),options.MaxArtifacts);
sourceManifestHash = sha256File(modelInventoryFile);

rows = repmat(struct( ...
    "ModelFamily","", "FeatureNumber",NaN, "FoldNumber",NaN, ...
    "ModelFile","", "PublishedSHA256","", "PublishedBytes",NaN, ...
    "SourceSHA256","", "ModelClass","", "ComputeMode","", ...
    "PredictionProbeRows",NaN, "Passed",false),numberToExport,1);

for rowPosition = 1:numberToExport
    sourceRow = inventory(rowPosition,:);
    family = string(sourceRow.ModelFamily);
    featureNumber = double(sourceRow.FeatureNumber);
    foldNumber = double(sourceRow.HeldSubject);
    sourceFile = canonicalPath(sourceRow.ModelFile);
    if ~startsWith(lower(sourceFile + filesep),lower(sourceRoot + filesep)) || ...
            ~isfile(sourceFile)
        error("Inventory references an invalid source model: %s",sourceFile);
    end
    sourceHash = sha256File(sourceFile);
    if sourceHash ~= lower(sourceRow.SHA256)
        error("Source-model hash mismatch: %s",sourceFile);
    end

    sourcePayload = load(sourceFile,"compactModel","modelMetadata");
    validateSourcePayload(sourcePayload,family,featureNumber,foldNumber, ...
        expectedDatasetHash,expectedGitCommit,expectedPredictors, ...
        expectedClasses,sourceFile);
    compactModel = sourcePayload.compactModel;
    sourceMetadata = sourcePayload.modelMetadata;

    publishedModelMetadata = buildPublishedMetadata( ...
        compactModel,sourceMetadata,sourceHash,sourceManifestHash,foldNumber);
    familyFolder = lower(family);
    destinationFolder = fullfile(destinationRoot,familyFolder, ...
        sprintf("feature_%02d",featureNumber));
    if ~isfolder(destinationFolder)
        mkdir(destinationFolder);
    end
    destinationFile = fullfile(destinationFolder,sprintf( ...
        "%s_feature_%02d_fold_%02d_compact_model.mat", ...
        familyFolder,featureNumber,foldNumber));

    if isfile(destinationFile)
        validatePublishedArtifact(destinationFile,compactModel, ...
            publishedModelMetadata,expectedPredictors,expectedClasses);
    else
        temporaryFile = destinationFile + ".tmp_" + ...
            string(java.util.UUID.randomUUID) + ".mat";
        cleanup = onCleanup(@() deleteIfPresent(temporaryFile));
        save(temporaryFile,"compactModel","publishedModelMetadata","-v7.3");
        validatePublishedArtifact(temporaryFile,compactModel, ...
            publishedModelMetadata,expectedPredictors,expectedClasses);
        [success,message] = movefile(temporaryFile,destinationFile,"f");
        if ~success
            error("Unable to promote published model: %s",message);
        end
        clear cleanup
        validatePublishedArtifact(destinationFile,compactModel, ...
            publishedModelMetadata,expectedPredictors,expectedClasses);
    end

    destinationInfo = dir(destinationFile);
    rows(rowPosition).ModelFamily = family;
    rows(rowPosition).FeatureNumber = featureNumber;
    rows(rowPosition).FoldNumber = foldNumber;
    rows(rowPosition).ModelFile = relativePublishedPath( ...
        destinationFile,destinationRoot);
    rows(rowPosition).PublishedSHA256 = sha256File(destinationFile);
    rows(rowPosition).PublishedBytes = destinationInfo.bytes;
    rows(rowPosition).SourceSHA256 = sourceHash;
    rows(rowPosition).ModelClass = string(class(compactModel));
    rows(rowPosition).ComputeMode = ...
        string(sourceMetadata.TrainingMeta.ComputeMode);
    rows(rowPosition).PredictionProbeRows = 8;
    rows(rowPosition).Passed = true;

    if mod(rowPosition,options.CheckpointEvery) == 0 || ...
            rowPosition == numberToExport
        manifest = struct2table(rows(1:rowPosition));
        writeManifestAtomically(manifest,fullfile(destinationRoot, ...
            "published_model_manifest.partial.csv"));
        fprintf("Published models: %d/%d\n",rowPosition,numberToExport);
    end
    clear compactModel sourcePayload sourceMetadata publishedModelMetadata
end

manifest = struct2table(rows);
if height(manifest) ~= numberToExport || ~all(manifest.Passed)
    error("Published model export did not validate every requested artifact.");
end
finalManifest = fullfile(destinationRoot,"published_model_manifest.csv");
writeManifestAtomically(manifest,finalManifest);
partialManifest = fullfile(destinationRoot, ...
    "published_model_manifest.partial.csv");
if isfile(partialManifest)
    delete(partialManifest);
end

fprintf("Published recovery models validated: %d/%d.\n", ...
    height(manifest),numberToExport);
fprintf("Manifest: %s\n",finalManifest);
end

function metadata = buildPublishedMetadata(model,sourceMetadata, ...
        sourceHash,sourceManifestHash,foldNumber)
metadata = struct;
metadata.SchemaVersion = 1;
metadata.ArtifactType = "privacy-minimized recreated LOSO compact model";
metadata.RecoveryCohort = true;
metadata.IsOriginalHistoricalObject = false;
metadata.ModelName = string(sourceMetadata.ModelName);
metadata.FeatureNumber = double(sourceMetadata.FeatureNumber);
metadata.FeatureName = string(sourceMetadata.FeatureName);
metadata.FoldNumber = foldNumber;
metadata.ClassOrder = double(sourceMetadata.ClassOrder(:)');
metadata.NumPredictors = double(sourceMetadata.NumPredictors);
metadata.PredictorIndices = double(sourceMetadata.PredictorIndices(:)');
metadata.Seed = double(sourceMetadata.Seed);
metadata.TrainingSettings = sourceMetadata.TrainingMeta.Settings;
metadata.TrainingComputeMode = ...
    string(sourceMetadata.TrainingMeta.ComputeMode);
metadata.DatasetSHA256 = lower(string(sourceMetadata.DatasetHash));
metadata.SourceModelSHA256 = lower(string(sourceHash));
metadata.SourceInventorySHA256 = lower(string(sourceManifestHash));
metadata.SourceGitCommit = string(sourceMetadata.GitCommit);
metadata.RecoveryScriptSHA256 = lower(string(sourceMetadata.ScriptSHA256));
metadata.MATLABRelease = string(sourceMetadata.MATLABRelease);
metadata.ModelClass = string(class(model));
metadata.SourceCreatedAt = sourceMetadata.CreatedAt;
metadata.PublishedAt = datetime("now");
metadata.PrivacyTransformation = [ ...
    "Removed local paths, subject lists, held-out predictions, " + ...
    "confusion matrices, row hashes, and fold metrics; the fitted " + ...
    "compactModel object is unchanged."];
end

function validateSourcePayload(payload,family,featureNumber,foldNumber, ...
        expectedDatasetHash,expectedGitCommit,expectedPredictors, ...
        expectedClasses,sourceFile)
if ~isfield(payload,"compactModel") || ~isfield(payload,"modelMetadata")
    error("Source model payload is incomplete: %s",sourceFile);
end
metadata = payload.modelMetadata;
required = ["ModelName","FeatureNumber","FeatureName","HeldSubject", ...
    "ClassOrder","NumPredictors","PredictorIndices","Seed", ...
    "TrainingMeta","DatasetHash","GitCommit","ScriptSHA256", ...
    "MATLABRelease","CreatedAt"];
if ~all(isfield(metadata,required)) || ...
        string(metadata.ModelName) ~= family || ...
        double(metadata.FeatureNumber) ~= featureNumber || ...
        double(metadata.HeldSubject) ~= foldNumber || ...
        lower(string(metadata.DatasetHash)) ~= expectedDatasetHash || ...
        string(metadata.GitCommit) ~= expectedGitCommit || ...
        double(metadata.NumPredictors) ~= expectedPredictors || ...
        ~isequal(double(metadata.ClassOrder(:)'),expectedClasses) || ...
        modelPredictorCount(payload.compactModel) ~= expectedPredictors
    error("Source model metadata validation failed: %s",sourceFile);
end
end

function validatePublishedArtifact(filePath,sourceModel,expectedMetadata, ...
        expectedPredictors,expectedClasses)
payload = load(filePath,"compactModel","publishedModelMetadata");
if ~isfield(payload,"compactModel") || ...
        ~isfield(payload,"publishedModelMetadata")
    error("Published model payload is incomplete: %s",filePath);
end
metadata = payload.publishedModelMetadata;
required = ["SchemaVersion","ArtifactType","ModelName", ...
    "FeatureNumber","FoldNumber","ClassOrder","NumPredictors", ...
    "DatasetSHA256","SourceModelSHA256","SourceGitCommit", ...
    "RecoveryScriptSHA256","ModelClass","PrivacyTransformation"];
if ~all(isfield(metadata,required)) || ...
        string(metadata.ModelName) ~= string(expectedMetadata.ModelName) || ...
        double(metadata.FeatureNumber) ~= ...
            double(expectedMetadata.FeatureNumber) || ...
        double(metadata.FoldNumber) ~= double(expectedMetadata.FoldNumber) || ...
        lower(string(metadata.SourceModelSHA256)) ~= ...
            lower(string(expectedMetadata.SourceModelSHA256)) || ...
        double(metadata.NumPredictors) ~= expectedPredictors || ...
        modelPredictorCount(payload.compactModel) ~= expectedPredictors || ...
        ~isequal(double(metadata.ClassOrder(:)'),expectedClasses) || ...
        ~isequal(double(payload.compactModel.ClassNames(:)'),expectedClasses)
    error("Published model metadata validation failed: %s",filePath);
end

probe = reshape(sin((1:8*expectedPredictors)/17),8,expectedPredictors);
[sourcePrediction,sourceScores] = predict(sourceModel,probe);
[publishedPrediction,publishedScores] = predict(payload.compactModel,probe);
if ~isequal(double(sourcePrediction(:)),double(publishedPrediction(:))) || ...
        ~isequal(size(sourceScores),size(publishedScores)) || ...
        any(~isfinite(double(publishedScores)),"all") || ...
        max(abs(double(sourceScores(:))-double(publishedScores(:)))) > 1e-12
    error("Published model prediction probe failed: %s",filePath);
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

function value = relativePublishedPath(filePath,destinationRoot)
prefix = destinationRoot + filesep;
if ~startsWith(lower(filePath),lower(prefix))
    error("Published file is outside the destination root: %s",filePath);
end
value = replace(extractAfter(filePath,strlength(prefix)), ...
    string(filesep),"/");
end

function writeManifestAtomically(manifest,filePath)
temporaryFile = fullfile(fileparts(filePath), ...
    string(java.util.UUID.randomUUID) + ".tmp.csv");
cleanup = onCleanup(@() deleteIfPresent(temporaryFile));
writetable(manifest,temporaryFile);
[success,message] = movefile(temporaryFile,filePath,"f");
if ~success
    error("Unable to promote published-model manifest: %s",message);
end
clear cleanup
end

function deleteIfPresent(filePath)
if isfile(filePath)
    delete(filePath);
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
