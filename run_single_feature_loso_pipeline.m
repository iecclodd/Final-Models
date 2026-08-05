function pipeline = run_single_feature_loso_pipeline(varargin)
%RUN_SINGLE_FEATURE_LOSO_PIPELINE Locked order: LDA, then MLP, then RBF-SVM.
% Stops immediately if a model errors; outer LOSO folds are always sequential.
p=inputParser; addParameter(p,"DatasetPath","",@(x)ischar(x)||isstring(x)); parse(p,varargin{:}); o=p.Results;
models=["lda" "mlp" "rbf_svm"]; pipeline=struct();
for m=models
    switch m
        case "lda", pipeline.lda=run_single_feature_lda_loso("DatasetPath",o.DatasetPath);
        case "mlp", pipeline.mlp=run_single_feature_mlp_loso("DatasetPath",o.DatasetPath);
        case "rbf_svm", pipeline.rbf_svm=run_single_feature_rbf_svm_loso("DatasetPath",o.DatasetPath);
    end
end
compare_single_feature_loso_models();
pipeline.comparisonFolder=fullfile(fileparts(mfilename("fullpath")),"single_feature_cross_model_comparison");
end
