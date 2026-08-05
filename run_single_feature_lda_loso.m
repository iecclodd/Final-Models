function results = run_single_feature_lda_loso(varargin)
%RUN_SINGLE_FEATURE_LDA_LOSO Locked LDA LOSO analysis; no interactive input.
results = run_single_feature_loso_model("lda", varargin{:}, "WritePipelineState", true);
end
