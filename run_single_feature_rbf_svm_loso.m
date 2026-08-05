function results = run_single_feature_rbf_svm_loso(varargin)
%RUN_SINGLE_FEATURE_RBF_SVM_LOSO Locked RBF-SVM LOSO analysis; no input UI.
results = run_single_feature_loso_model("rbf_svm", varargin{:}, "WritePipelineState", true);
end
