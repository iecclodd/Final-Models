function report = verify_recovery_shap_inventory(datasetFile,artifactRoot)
%VERIFY_RECOVERY_SHAP_INVENTORY Independently validate every SHAP artifact.
%   Verifies the family-isolated manifests and fold artifacts from both the
%   screening and selected-refinement runs. This function is read-only with
%   respect to persisted models and SHAP fold artifacts.

arguments
    datasetFile (1,1) string
    artifactRoot (1,1) string
end

expectedDatasetHash = ...
    "8f84902f12b4e0d1c3f47d2d536e4257379d3f6c5179e4f68b54a139f006f719";
expectedGitCommit = "e568d39a07873e52f049a827cfb65686fb308893";
families = ["LDA","MLP","RBF_SVM"];
phaseNames = ["full","selected"];
phaseExpectedPerFamily = [378 90];
phaseFeatures = {1:21,[1 3 6 12 14]};
expectedChannels = 128;
expectedClasses = 8;
expectedTotal = sum(phaseExpectedPerFamily)*numel(families);

datasetFile = canonicalPath(datasetFile);
artifactRoot = canonicalPath(artifactRoot);
if ~isfile(datasetFile)
    error("Dataset does not exist: %s",datasetFile);
end
if sha256File(datasetFile) ~= expectedDatasetHash
    error("Dataset SHA-256 does not match the locked dataset.");
end

modelInventoryFile = fullfile(artifactRoot,"final_report", ...
    "independent_model_inventory.csv");
if ~isfile(modelInventoryFile)
    error("Independent model inventory is missing: %s",modelInventoryFile);
end
modelInventory = readtable(modelInventoryFile,"Delimiter",",", ...
    "TextType","string");
if height(modelInventory) ~= 1134 || ~all(modelInventory.Passed)
    error("Independent model inventory is incomplete or failed.");
end

dataset = matfile(datasetFile);
subjectID = double(dataset.subjectID);
subjectID = subjectID(:);
labels = double(dataset.Y);
labels = labels(:);
numberOfRows = numel(subjectID);
if numel(labels) ~= numberOfRows
    error("Dataset subject and label vectors differ in length.");
end

rows = repmat(struct( ...
    "Phase","", "ModelFamily","", "FeatureNumber",NaN, ...
    "HeldSubject",NaN, "ShapFile","", "ShapSHA256","", ...
    "ModelFile","", "ModelSHA256","", "BackgroundCount",NaN, ...
    "QueryCount",NaN, "ShapMode","", "Passed",false), ...
    expectedTotal,1);
rowPosition = 0;

for phasePosition = 1:numel(phaseNames)
    phaseName = phaseNames(phasePosition);
    expectedPerFamily = phaseExpectedPerFamily(phasePosition);
    expectedFeatures = phaseFeatures{phasePosition};
    for family = families
        familyFolder = fullfile(artifactRoot,"shap_runs",phaseName, ...
            "families",lower(family));
        foldFolder = fullfile(familyFolder,"folds");
        manifestFile = fullfile(familyFolder,"manifests", ...
            "recovery_model_and_shap_manifest.csv");
        if ~isfile(manifestFile)
            error("SHAP manifest is missing: %s",manifestFile);
        end
        manifest = readtable(manifestFile,"Delimiter",",", ...
            "TextType","string");
        manifest = manifest(manifest.ModelFamily == family,:);
        if height(manifest) ~= expectedPerFamily || ...
                ~all(manifest.Completed) || any(manifest.Failed)
            error("Manifest completion failed for %s/%s.",phaseName,family);
        end
        if ~isequal(unique(double(manifest.FeatureNumber(:)))', ...
                double(expectedFeatures))
            error("Manifest feature set failed for %s/%s.",phaseName,family);
        end
        assertFolderInventory(foldFolder,expectedPerFamily,phaseName,family);

        for manifestPosition = 1:height(manifest)
            manifestRow = manifest(manifestPosition,:);
            featureNumber = double(manifestRow.FeatureNumber);
            heldSubject = double(manifestRow.HeldSubject);
            shapFile = canonicalPath(manifestRow.ShapFile);
            modelFile = canonicalPath(manifestRow.ModelFile);
            if ~isfile(shapFile) || ~isfile(modelFile)
                error("Manifest references a missing artifact: %s",shapFile);
            end

            shapHash = sha256File(shapFile);
            if lower(manifestRow.ShapSHA256) ~= shapHash
                error("SHAP SHA-256 mismatch: %s",shapFile);
            end

            modelRow = modelInventory( ...
                modelInventory.ModelFamily == family & ...
                modelInventory.FeatureNumber == featureNumber & ...
                modelInventory.HeldSubject == heldSubject,:);
            if height(modelRow) ~= 1 || ...
                    canonicalPath(modelRow.ModelFile) ~= modelFile || ...
                    lower(modelRow.SHA256) ~= lower(manifestRow.ModelSHA256)
                error("Model-inventory binding failed: %s",modelFile);
            end

            payload = load(shapFile,"foldResult");
            if ~isfield(payload,"foldResult")
                error("SHAP payload is missing foldResult: %s",shapFile);
            end
            foldResult = payload.foldResult;
            validateFoldResult(foldResult,phaseName,family,featureNumber, ...
                heldSubject,expectedDatasetHash,expectedGitCommit, ...
                manifestRow,subjectID,labels,numberOfRows, ...
                expectedChannels,expectedClasses,shapFile);

            rowPosition = rowPosition + 1;
            rows(rowPosition).Phase = phaseName;
            rows(rowPosition).ModelFamily = family;
            rows(rowPosition).FeatureNumber = featureNumber;
            rows(rowPosition).HeldSubject = heldSubject;
            rows(rowPosition).ShapFile = shapFile;
            rows(rowPosition).ShapSHA256 = shapHash;
            rows(rowPosition).ModelFile = modelFile;
            rows(rowPosition).ModelSHA256 = lower(manifestRow.ModelSHA256);
            rows(rowPosition).BackgroundCount = ...
                numel(foldResult.BackgroundIndices);
            rows(rowPosition).QueryCount = numel(foldResult.QueryIndices);
            rows(rowPosition).ShapMode = string(foldResult.ShapMode);
            rows(rowPosition).Passed = true;
        end
    end
end

report = struct2table(rows);
if rowPosition ~= expectedTotal || height(report) ~= expectedTotal || ...
        ~all(report.Passed)
    error("Independent SHAP inventory did not validate every artifact.");
end

reportFolder = fullfile(artifactRoot,"final_report");
if ~isfolder(reportFolder)
    mkdir(reportFolder);
end
csvFile = fullfile(reportFolder,"independent_shap_inventory.csv");
matFile = fullfile(reportFolder,"independent_shap_inventory.mat");
writetable(report,csvFile);
summary = struct;
summary.SchemaVersion = 1;
summary.CreatedAt = datetime("now");
summary.DatasetFile = datasetFile;
summary.DatasetSHA256 = expectedDatasetHash;
summary.GitCommit = expectedGitCommit;
summary.ArtifactCount = height(report);
summary.ScreeningCount = nnz(report.Phase == "full");
summary.SelectedCount = nnz(report.Phase == "selected");
summary.AllPassed = all(report.Passed);
summary.FamilyPhaseCounts = groupsummary(report,["Phase","ModelFamily"]);
save(matFile,"report","summary","-v7.3");

fprintf("Independent SHAP inventory passed: %d/%d artifacts.\n", ...
    height(report),expectedTotal);
fprintf("Report: %s\n",csvFile);
end

function validateFoldResult(R,phaseName,family,featureNumber,heldSubject, ...
        expectedDatasetHash,expectedGitCommit,manifestRow,subjectID,labels, ...
        numberOfRows,expectedChannels,expectedClasses,shapFile)
required = ["ModelName","FeatureNumber","HeldSubject","DatasetHash", ...
    "GitCommit","ModelArtifactFile","ModelSHA256", ...
    "MeanAbsoluteShapley","NormalizedMeanAbsoluteShapley", ...
    "BackgroundIndices","QueryIndices","QueryLabels","ClassOrder", ...
    "ChannelIds","ShapMode","Config"];
if ~all(isfield(R,required))
    error("SHAP metadata is incomplete: %s",shapFile);
end
if string(R.ModelName) ~= family || ...
        double(R.FeatureNumber) ~= featureNumber || ...
        double(R.HeldSubject) ~= heldSubject || ...
        lower(string(R.DatasetHash)) ~= expectedDatasetHash || ...
        string(R.GitCommit) ~= expectedGitCommit || ...
        canonicalPath(R.ModelArtifactFile) ~= canonicalPath(manifestRow.ModelFile) || ...
        lower(string(R.ModelSHA256)) ~= lower(manifestRow.ModelSHA256)
    error("SHAP key or provenance mismatch: %s",shapFile);
end

meanAbsolute = double(R.MeanAbsoluteShapley);
normalized = double(R.NormalizedMeanAbsoluteShapley);
if ~isequal(size(meanAbsolute),[expectedChannels expectedClasses]) || ...
        ~isequal(size(normalized),[expectedChannels expectedClasses]) || ...
        any(~isfinite(meanAbsolute),"all") || ...
        any(~isfinite(normalized),"all") || ...
        any(meanAbsolute < 0,"all") || any(normalized < 0,"all")
    error("SHAP matrices are malformed: %s",shapFile);
end
if ~isequal(double(R.ClassOrder(:)'),1:expectedClasses) || ...
        numel(R.ChannelIds) ~= expectedChannels
    error("SHAP class/channel metadata is malformed: %s",shapFile);
end

backgroundIndices = double(R.BackgroundIndices(:));
queryIndices = double(R.QueryIndices(:));
if any(~isfinite(backgroundIndices)) || any(~isfinite(queryIndices)) || ...
        any(backgroundIndices < 1) || any(backgroundIndices > numberOfRows) || ...
        any(queryIndices < 1) || any(queryIndices > numberOfRows) || ...
        any(backgroundIndices ~= fix(backgroundIndices)) || ...
        any(queryIndices ~= fix(queryIndices)) || ...
        numel(unique(backgroundIndices)) ~= numel(backgroundIndices) || ...
        numel(unique(queryIndices)) ~= numel(queryIndices) || ...
        any(subjectID(backgroundIndices) == heldSubject) || ...
        any(subjectID(queryIndices) ~= heldSubject)
    error("SHAP background/query provenance failed: %s",shapFile);
end
if ~isequal(double(R.QueryLabels(:)),labels(queryIndices))
    error("SHAP query labels do not match the dataset: %s",shapFile);
end

backgroundPerClass = double(R.Config.BackgroundPerClass);
queryPerClass = double(R.Config.QueryPerClass);
if phaseName == "full"
    expectedConfiguration = [1 2 128];
else
    expectedConfiguration = [4 8 512];
end
actualConfiguration = [queryPerClass backgroundPerClass ...
    double(R.Config.MaxNumSubsets)];
if ~isequal(actualConfiguration,expectedConfiguration) || ...
        numel(backgroundIndices) ~= expectedClasses*backgroundPerClass || ...
        numel(queryIndices) ~= expectedClasses*queryPerClass || ...
        ~isequal(groupcounts(categorical(labels(backgroundIndices),1:8))', ...
            repmat(backgroundPerClass,1,expectedClasses)) || ...
        ~isequal(groupcounts(categorical(labels(queryIndices),1:8))', ...
            repmat(queryPerClass,1,expectedClasses))
    error("SHAP sampling/configuration failed: %s",shapFile);
end
end

function assertFolderInventory(foldFolder,expectedCount,phaseName,family)
artifacts = dir(fullfile(foldFolder,"*_shap.mat"));
failures = dir(fullfile(foldFolder,"*_FAILURE.txt"));
temporaries = [dir(fullfile(foldFolder,"*.tmp*")); ...
    dir(fullfile(foldFolder,"*.temporary*"))];
if numel(artifacts) ~= expectedCount || ~isempty(failures) || ...
        ~isempty(temporaries)
    error("Fold-folder inventory failed for %s/%s.",phaseName,family);
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
