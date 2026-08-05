function summary=run_all_single_feature_loso_smoke_tests()
%RUN_ALL_SINGLE_FEATURE_LOSO_SMOKE_TESTS Run exact isolated smokes in locked order.
models=["lda" "mlp" "rbf_svm"];
summary=struct();
for model=models
    summary.(char(model))=run_single_feature_loso_smoke_test(model);
end
root=string(fileparts(mfilename("fullpath")));
folder=fullfile(root,"single_feature_loso_smoke_tests");
save(fullfile(folder,"single_feature_loso_smoke_test_summary.mat"),"summary","-v7.3");
fid=fopen(fullfile(folder,"single_feature_loso_smoke_test_summary.txt"),"w");
if fid<0, error("Could not write consolidated smoke summary."); end
cleanup=onCleanup(@() fclose(fid));
for model=models
    checkpoint=summary.(char(model)).checkpoint;
    fprintf(fid,"%s: completed=%d compute=%s BA=%.12f OA=%.12f train=%.6f predict=%.6f\n", ...
        upper(model),checkpoint.completed(1,1),checkpoint.settings.compute.mode, ...
        checkpoint.balancedAccuracy(1,1),checkpoint.ordinaryAccuracy(1,1), ...
        checkpoint.trainingTimeSeconds(1,1),checkpoint.predictionTimeSeconds(1,1));
end
end
