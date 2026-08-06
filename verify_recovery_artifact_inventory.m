function report = verify_recovery_artifact_inventory(datasetFile,artifactRoot)
%VERIFY_RECOVERY_ARTIFACT_INVENTORY Independently validate recovery models.
%   Loads every expected compact model, matches it to the exported manifest,
%   verifies the artifact hash and metadata key, and reproduces a small saved
%   held-out prediction probe. This function does not call the recovery
%   pipeline and must be run in a fresh MATLAB process before full SHAP.

arguments
    datasetFile (1,1) string
    artifactRoot (1,1) string
end

expectedDatasetHash = ...
    "8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719";
expectedGitCommit = "e568d39a07873e52f049a827cfb65686fb308893";
families = ["LDA","MLP","RBF_SVM"];
classOrder = 1:8;
expectedPredictors = 128;
expectedArtifacts = 3*21*18;
probeRows = 8;

datasetFile = string(java.io.File(char(datasetFile)).getCanonicalPath());
artifactRoot = string(java.io.File(char(artifactRoot)).getCanonicalPath());
if ~isfile(datasetFile)
    error("Dataset does not exist: %s",datasetFile);
end
if sha256File(datasetFile) ~= expectedDatasetHash
    error("Dataset SHA-256 does not match the locked dataset.");
end

manifestFile = fullfile(artifactRoot,"manifests", ...
    "recovery_model_and_shap_manifest.csv");
if ~isfile(manifestFile)
    error("Recovery manifest is missing: %s",manifestFile);
end
manifest = readtable(manifestFile,"TextType","string");
if height(manifest) ~= expectedArtifacts
    error("Manifest must contain %d rows; found %d.", ...
        expectedArtifacts,height(manifest));
end

data = matfile(datasetFile);
subjectID = double(data.subjectID);
rows = repmat(struct( ...
    "ModelFamily","", "FeatureNumber",NaN, "HeldSubject",NaN, ...
    "ModelFile","", "SHA256","", "ModelClass","", ...
    "ComputeMode","", "ProbeRows",NaN, "Passed",false), ...
    expectedArtifacts,1);
rowPosition = 0;

for family = families
    familyFolder = lower(family);
    for featureNumber = 1:21
        for heldSubject = 1:18
            rowPosition = rowPosition + 1;
            modelFile = fullfile(artifactRoot,"persisted_models", ...
                familyFolder,sprintf("feature_%02d",featureNumber), ...
                sprintf("%s_feature_%02d_subject_%02d_compact_model.mat", ...
                familyFolder,featureNumber,heldSubject));
            if ~isfile(modelFile)
                error("Expected model artifact is missing: %s",modelFile);
            end

            manifestRow = manifest( ...
                manifest.ModelFamily == family & ...
                manifest.FeatureNumber == featureNumber & ...
                manifest.HeldSubject == heldSubject,:);
            if height(manifestRow) ~= 1
                error("Manifest key is absent or duplicated: %s/%d/%d.", ...
                    family,featureNumber,heldSubject);
            end

            actualHash = sha256File(modelFile);
            if lower(manifestRow.ModelSHA256) ~= actualHash
                error("Manifest SHA-256 mismatch: %s",modelFile);
            end

            artifact = load(modelFile,"compactModel","modelMetadata");
            if ~isfield(artifact,"compactModel") || ...
                    ~isfield(artifact,"modelMetadata")
                error("Model payload is incomplete: %s",modelFile);
            end
            model = artifact.compactModel;
            metadata = artifact.modelMetadata;

            requiredMetadata = ["ModelName","FeatureNumber", ...
                "HeldSubject","DatasetHash","GitCommit","ClassOrder", ...
                "NumPredictors","PredictorIndices","TestRowCount", ...
                "TestPredictions","TrainingMeta"];
            if ~all(isfield(metadata,requiredMetadata))
                error("Model metadata is incomplete: %s",modelFile);
            end
            if string(metadata.ModelName) ~= family || ...
                    double(metadata.FeatureNumber) ~= featureNumber || ...
                    double(metadata.HeldSubject) ~= heldSubject
                error("Model metadata key mismatch: %s",modelFile);
            end
            if lower(string(metadata.DatasetHash)) ~= expectedDatasetHash || ...
                    string(metadata.GitCommit) ~= expectedGitCommit
                error("Dataset/source binding mismatch: %s",modelFile);
            end
            if ~isequal(double(metadata.ClassOrder(:)'),classOrder) || ...
                    ~isequal(double(model.ClassNames(:)'),classOrder)
                error("Class order mismatch: %s",modelFile);
            end
            if double(metadata.NumPredictors) ~= expectedPredictors || ...
                    modelPredictorCount(model) ~= expectedPredictors || ...
                    numel(metadata.PredictorIndices) ~= expectedPredictors
                error("Predictor count mismatch: %s",modelFile);
            end
            validateFamilyClass(model,family,modelFile);

            numberToProbe = min(probeRows,double(metadata.TestRowCount));
            heldOutRows = find(subjectID == heldSubject);
            if numel(heldOutRows) ~= double(metadata.TestRowCount)
                error("Held-out row count mismatch: %s",modelFile);
            end
            testRows = heldOutRows(1:numberToProbe);
            predictors = double(metadata.PredictorIndices(:)');
            XProbe = data.X(testRows,predictors);
            [prediction,scores] = predict(model,XProbe);
            prediction = double(prediction(:));
            scores = double(scores);
            if ~isequal(prediction, ...
                    double(metadata.TestPredictions(1:numberToProbe))) || ...
                    any(~isfinite(scores),"all") || ...
                    ~isequal(size(scores),[numberToProbe numel(classOrder)])
                error("Saved prediction probe failed: %s",modelFile);
            end

            rows(rowPosition).ModelFamily = family;
            rows(rowPosition).FeatureNumber = featureNumber;
            rows(rowPosition).HeldSubject = heldSubject;
            rows(rowPosition).ModelFile = string(modelFile);
            rows(rowPosition).SHA256 = actualHash;
            rows(rowPosition).ModelClass = string(class(model));
            rows(rowPosition).ComputeMode = ...
                string(metadata.TrainingMeta.ComputeMode);
            rows(rowPosition).ProbeRows = numberToProbe;
            rows(rowPosition).Passed = true;
        end
    end
end

report = struct2table(rows);
if height(report) ~= expectedArtifacts || ~all(report.Passed)
    error("Independent inventory did not validate all expected artifacts.");
end
for family = families
    if sum(report.ModelFamily == family) ~= 378
        error("Independent inventory count failed for %s.",family);
    end
end

reportFolder = fullfile(artifactRoot,"final_report");
if ~isfolder(reportFolder)
    mkdir(reportFolder);
end
csvFile = fullfile(reportFolder,"independent_model_inventory.csv");
matFile = fullfile(reportFolder,"independent_model_inventory.mat");
writetable(report,csvFile);
summary = struct;
summary.SchemaVersion = 1;
summary.CreatedAt = datetime("now");
summary.DatasetFile = datasetFile;
summary.DatasetSHA256 = expectedDatasetHash;
summary.GitCommit = expectedGitCommit;
summary.ManifestFile = string(manifestFile);
summary.ManifestSHA256 = sha256File(manifestFile);
summary.ArtifactCount = height(report);
summary.ProbeRowsPerArtifact = probeRows;
summary.FamilyCounts = groupsummary(report,"ModelFamily");
summary.AllPassed = all(report.Passed);
save(matFile,"report","summary","-v7.3");

fprintf("Independent model inventory passed: %d/%d artifacts.\n", ...
    height(report),expectedArtifacts);
fprintf("Report: %s\n",csvFile);
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
