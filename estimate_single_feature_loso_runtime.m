function estimate = estimate_single_feature_loso_runtime(varargin)
%ESTIMATE_SINGLE_FEATURE_LOSO_RUNTIME Times locked fit/predict paths only.
p=inputParser; addParameter(p,"DatasetPath","",@(x)ischar(x)||isstring(x)); addParameter(p,"Pairs",[1 1;11 9;21 18],@(x)isnumeric(x)&&size(x,2)==2&&all(isfinite(x),"all")&&all(x(:,1)>=1&x(:,1)<=21&x(:,2)>=1&x(:,2)<=18&mod(x,1)==0,"all")); parse(p,varargin{:}); o=p.Results;
root=string(fileparts(mfilename("fullpath"))); output=fullfile(root,"single_feature_loso_runtime_benchmark",string(datetime("now","Format","yyyyMMdd_HHmmssSSS"))); if ~isfolder(output),mkdir(output);end
models=["lda" "mlp" "rbf_svm"]; rows=[];
for m=models
    for k=1:size(o.Pairs,1)
        f=o.Pairs(k,1); s=o.Pairs(k,2); runFolder=fullfile(output,m+"_"+sprintf("f%02d_s%02d",f,s));
        r=run_single_feature_loso_model(m,"DatasetPath",o.DatasetPath,"FeatureNumbers",f,"SubjectIDs",s,"OutputRoot",runFolder);
        c=r.checkpoint; rows=[rows; table(m,f,s,string(c.settings.compute.mode),string(c.settings.compute.detail),c.settings.compute.parallelWorkers,c.trainingTimeSeconds(f,s),c.predictionTimeSeconds(f,s),VariableNames=["Model","FeatureNumber","HeldSubject","ComputeMode","ComputeDetail","Workers","TrainingSeconds","PredictionSeconds"])]; %#ok<AGROW>
    end
end
summary=groupsummary(rows,"Model","median",["TrainingSeconds" "PredictionSeconds"]); summary.ProjectedTrainingSecondsFor378Folds=378*summary.median_TrainingSeconds; summary.ProjectedEndToEndSecondsFor378Folds=378*(summary.median_TrainingSeconds+summary.median_PredictionSeconds); estimate=struct("label","Smoke-frozen exact settings; projection from isolated benchmark folds","foldTimings",rows,"projection",summary,"timestampUtc",char(datetime("now","TimeZone","UTC")));
writetable(rows,fullfile(output,"single_feature_loso_runtime_fold_timings.csv")); writetable(summary,fullfile(output,"single_feature_loso_runtime_estimate.csv")); save(fullfile(output,"single_feature_loso_runtime_estimate.mat"),"estimate");
end
