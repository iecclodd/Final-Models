classdef publishedRecoveryModelsTest < matlab.unittest.TestCase
    %PUBLISHEDRECOVERYMODELSTEST Audit Git LFS publication artifacts.

    methods (TestClassSetup)
        function addRecoverySourceToPath(testCase)
            sourceFolder = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(sourceFolder));
        end
    end

    methods (Test)
        function testCompletePublishedInventory(testCase)
            modelRoot = string(getenv("EMG_PUBLISHED_MODEL_ROOT"));
            testCase.assumeNotEmpty(modelRoot, ...
                "Set EMG_PUBLISHED_MODEL_ROOT to audit published models.");

            report = verify_published_recovery_models(modelRoot);

            testCase.verifyEqual(height(report),1134);
            testCase.verifyTrue(all(report.Passed));
            testCase.verifyEqual(nnz(report.ModelFamily == "LDA"),378);
            testCase.verifyEqual(nnz(report.ModelFamily == "MLP"),378);
            testCase.verifyEqual(nnz(report.ModelFamily == "RBF_SVM"),378);
        end
    end
end
