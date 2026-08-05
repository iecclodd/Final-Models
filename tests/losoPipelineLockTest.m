classdef losoPipelineLockTest < matlab.unittest.TestCase
    %losoPipelineLockTest Tests exclusive ownership and atomic state publication.

    properties
        Root (1,1) string
    end

    methods (TestClassSetup)
        function addProjectRoot(testCase)
            projectRoot=fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (TestMethodSetup)
        function createTemporaryRoot(testCase)
            testCase.Root=string(tempname);
            mkdir(testCase.Root);
            testCase.addTeardown(@() rmdir(testCase.Root,"s"));
        end
    end

    methods (Test)
        function testSecondLiveOwnerIsRejected(testCase)
            state=testCase.makeState("running");
            owner=loso_pipeline_lock("acquire",testCase.Root,struct(),state); %#ok<NASGU>
            testCase.verifyError(@() loso_pipeline_lock("acquire",testCase.Root,struct(),state), ...
                "LOSO:ActivePipelineLock");
        end

        function testNonOwnerCannotUpdate(testCase)
            state=testCase.makeState("running");
            owner=loso_pipeline_lock("acquire",testCase.Root,struct(),state);
            impostor=owner;
            impostor.owner_nonce="not-the-owner";
            testCase.verifyError(@() loso_pipeline_lock("update",testCase.Root,impostor,state), ...
                "LOSO:LockOwnership");
        end

        function testOwnerPublishesValidStatus(testCase)
            state=testCase.makeState("starting");
            owner=loso_pipeline_lock("acquire",testCase.Root,struct(),state);
            completed=testCase.makeState("completed");
            completed.completed_folds=378;
            loso_pipeline_lock("complete",testCase.Root,owner,completed);
            lock=jsondecode(fileread(fullfile(testCase.Root,"pipeline_lock.json")));
            status=jsondecode(fileread(fullfile(testCase.Root,"run_status.json")));
            markdown=string(fileread(fullfile(testCase.Root,"RUN_STATUS.md")));
            testCase.verifyEqual(string(lock.phase),"completed");
            testCase.verifyEqual(lock.completed_folds,378);
            testCase.verifyEqual(string(status.owner_nonce),string(owner.owner_nonce));
            testCase.verifyTrue(contains(markdown,"Completed folds: 378"));
        end

        function testCompletedLockIsArchivedBeforeNextModel(testCase)
            state=testCase.makeState("running");
            owner=loso_pipeline_lock("acquire",testCase.Root,struct(),state);
            loso_pipeline_lock("complete",testCase.Root,owner,testCase.makeState("completed"));
            nextOwner=loso_pipeline_lock("acquire",testCase.Root,struct(),state);
            archives=dir(fullfile(testCase.Root,"pipeline_lock.json.archive_*"));
            testCase.verifyNotEqual(string(nextOwner.owner_nonce),string(owner.owner_nonce));
            testCase.verifyGreaterThanOrEqual(numel(archives),1);
        end
    end

    methods (Static, Access=private)
        function state=makeState(phase)
            state=struct("phase",char(phase),"model","lda","compute_mode","cpu", ...
                "completed_folds",0,"total_folds",378,"last_feature",NaN, ...
                "last_held_subject",NaN,"progress_file","checkpoint.mat", ...
                "console_log","run.log","last_checkpoint_timestamp","", ...
                "blocking_issue","","next_action","test","timestamp_utc", ...
                char(datetime("now","TimeZone","UTC")));
        end
    end
end
