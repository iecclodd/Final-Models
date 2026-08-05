function results = run_single_feature_mlp_loso(varargin)
%RUN_SINGLE_FEATURE_MLP_LOSO Locked MLP LOSO analysis; no interactive input.
results = run_single_feature_loso_model("mlp", varargin{:}, "WritePipelineState", true);
end
