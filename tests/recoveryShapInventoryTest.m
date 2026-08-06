classdef recoveryShapInventoryTest < matlab.unittest.TestCase
    %RECOVERYSHAPINVENTORYTEST End-to-end independent SHAP validation.

    methods (TestClassSetup)
        function addRecoverySourceToPath(testCase)
            sourceFolder = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(sourceFolder));
        end
    end

    methods (Test)
        function testCompleteRecoveryInventory(testCase)
            datasetFile = string(getenv("CAPGMYO_DATASET_FILE"));
            artifactRoot = string(getenv("EMG_RECOVERY_ARTIFACT_ROOT"));
            testCase.assumeNotEmpty(datasetFile, ...
                "Set CAPGMYO_DATASET_FILE to run the recovery inventory test.");
            testCase.assumeNotEmpty(artifactRoot, ...
                "Set EMG_RECOVERY_ARTIFACT_ROOT to run the recovery inventory test.");

            report = verify_recovery_shap_inventory( ...
                datasetFile,artifactRoot);

            testCase.verifyEqual(height(report),1404);
            testCase.verifyTrue(all(report.Passed));
            testCase.verifyEqual(nnz(report.Phase == "full"),1134);
            testCase.verifyEqual(nnz(report.Phase == "selected"),270);
        end
    end
end
