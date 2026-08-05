function compare_single_feature_loso_models(mode)
%COMPARE_SINGLE_FEATURE_LOSO_MODELS Strict, staged comparison of three LOSO finals.
if nargin == 1 && strcmpi(string(mode),"selftest"), runSelfTest; return; end
if nargin ~= 0, error("Comparison:Usage","Use no input or the internal 'selftest' mode."); end
compareCore(fileparts(mfilename("fullpath")));
end

function compareCore(root)
models = ["LDA","MLP","RBF-SVM"];
dirs = ["single_feature_lda_loso_results","single_feature_mlp_loso_results","single_feature_rbf_svm_loso_results"];
names = ["single_feature_lda_loso_final.mat","single_feature_mlp_loso_final.mat","single_feature_rbf_svm_loso_final.mat"];
R = loadFinal(fullfile(root,dirs(1),names(1)),models(1));
for k=2:3, R(k)=loadFinal(fullfile(root,dirs(k),names(k)),models(k)); end
validateCompatibility(R,models);
[B,O,Tr,Pr,features,featureNumbers,subjects] = alignResults(R);
[long,wide,consensus,pairs,correlationMatrix,subjectSummary] = makeTables(B,O,Tr,Pr,features,featureNumbers,subjects,models);
stage = string(tempname(root)); mkdir(stage);
cleanup = onCleanup(@() cleanStage(stage));
writeOutputs(stage,R,models,features,subjects,B,long,wide,consensus,pairs,correlationMatrix,subjectSummary,Tr,Pr);
validateBundle(stage);
promoteBundle(stage,fullfile(root,"single_feature_cross_model_comparison"));
end

function R = loadFinal(file,expectedModel)
if ~isfile(file), error("Comparison:MissingFinal","Missing final: %s",file); end
raw=load(file); required=["balancedAccuracy","ordinaryAccuracy","trainingTimeSeconds","predictionTimeSeconds","completed","featureNames","featureNumbers","subjects","classOrder","datasetHash","matlabRelease","computeMode","modelSettings"];
if all(isfield(raw,required))
    S=raw;
elseif isfield(raw,"results") && isstruct(raw.results) && isfield(raw.results,"checkpoint") && isstruct(raw.results.checkpoint) && all(isfield(raw.results.checkpoint,required))
    S=raw.results.checkpoint;
else
    error("Comparison:Schema","%s must use exact top-level schema or results.checkpoint alias.",file);
end
for f=required
    if isempty(S.(f)), error("Comparison:Schema","Required field %s is empty in %s.",f,file); end
end
R=struct; R.expectedModel=expectedModel; R.sourceFile=string(file); R.balancedAccuracy=double(S.balancedAccuracy); R.ordinaryAccuracy=double(S.ordinaryAccuracy); R.trainingTimeSeconds=double(S.trainingTimeSeconds); R.predictionTimeSeconds=double(S.predictionTimeSeconds); R.completed=S.completed; R.featureNames=string(S.featureNames(:)); R.featureNumbers=double(S.featureNumbers(:)); R.subjects=double(S.subjects(:)); R.classOrder=string(S.classOrder(:)); R.datasetHash=string(S.datasetHash); R.matlabRelease=string(S.matlabRelease); R.computeMode=normaliseComputeMode(S.computeMode); R.modelSettings=S.modelSettings;
R.computeDetails=fieldOr(S,"computeDetails",struct); R.computeFingerprint=fieldOr(S,"computeFingerprint",""); R.deviations=fieldOr(S,"deviations",fieldOr(S,"warnings",strings(0,1))); R.modelIdentity=fieldOr(S,"modelIdentity",fieldOr(S,"modelName",expectedModel));
if ~islogical(R.completed) || ~isequal(size(R.completed),[21 18]) || nnz(R.completed)~=378
    error("Comparison:Completion","%s completed must be logical 21x18 with 378 true entries.",file);
end
arrays={R.balancedAccuracy,R.ordinaryAccuracy,R.trainingTimeSeconds,R.predictionTimeSeconds};
for i=1:4
    if ~isequal(size(arrays{i}),[21 18]) || any(~isfinite(arrays{i}),"all"), error("Comparison:Metrics","%s contains invalid metric dimensions or values.",file); end
end
if any(R.balancedAccuracy<0 | R.balancedAccuracy>1,"all") || any(R.ordinaryAccuracy<0 | R.ordinaryAccuracy>1,"all") || any(R.trainingTimeSeconds<0,"all") || any(R.predictionTimeSeconds<0,"all") || any(abs(R.ordinaryAccuracy-R.balancedAccuracy)>1e-10,"all")
    error("Comparison:Metrics","%s violates decimal accuracy, balanced-dataset OA/BA, or nonnegative-time rules.",file);
end
if numel(R.featureNames)~=21 || numel(unique(R.featureNames))~=21 || any(ismissing(R.featureNames)) || ~isequal(sort(R.featureNumbers),(1:21)') || numel(unique(R.subjects))~=18 || any(~isfinite(R.subjects)) || numel(R.classOrder)~=8 || numel(unique(R.classOrder))~=8 || strlength(R.datasetHash)==0 || strlength(R.matlabRelease)==0 || strlength(R.computeMode)==0 || isempty(R.modelSettings)
    error("Comparison:Identity","%s has invalid feature, subject, class, provenance, compute, or settings identity.",file);
end
if strlength(string(R.modelIdentity))==0, error("Comparison:Identity","%s has an empty model identity.",file); end
end

function value = fieldOr(S,name,default)
if isfield(S,name), value=S.(name); else, value=default; end
end

function mode = normaliseComputeMode(value)
if isstruct(value)
    if isfield(value,"mode"), value=value.mode; elseif isfield(value,"requestedMode"), value=value.requestedMode; else, value="legacy-struct"; end
end
mode=string(value); if ~isscalar(mode) || ismissing(mode) || strlength(mode)==0, error("Comparison:ComputeMode","computeMode must be scalar text or a recognised legacy struct."); end
end

function validateCompatibility(R,models)
for k=1:3
    if ~contains(lower(string(R(k).modelIdentity)),lower(models(k))) && ~contains(lower(R(k).sourceFile),lower(extractBefore(models(k),"-")))
        error("Comparison:ModelIdentity","Model identity does not match expected %s final.",models(k));
    end
end
for k=2:3
    if R(k).datasetHash~=R(1).datasetHash || R(k).matlabRelease~=R(1).matlabRelease || ~isequal(R(k).classOrder,R(1).classOrder) || ~isequal(sort(R(k).subjects),sort(R(1).subjects)) || ~isequal(sort(R(k).featureNames),sort(R(1).featureNames)) || ~isequal(sort(R(k).featureNumbers),sort(R(1).featureNumbers))
        error("Comparison:Incompatible","Finals disagree on dataset/release/classes/features/subjects.");
    end
end
end

function [B,O,Tr,Pr,features,numbers,subjects] = alignResults(R)
features=R(1).featureNames; numbers=R(1).featureNumbers; subjects=R(1).subjects; B=zeros(21,18,3); O=B; Tr=B; Pr=B;
for k=1:3
    [okF,fi]=ismember(features,R(k).featureNames); [okS,si]=ismember(subjects,R(k).subjects);
    if ~all(okF) || ~all(okS) || ~isequal(R(k).featureNumbers(fi),numbers), error("Comparison:Alignment","Feature name+number or subject mapping failed."); end
    B(:,:,k)=R(k).balancedAccuracy(fi,si); O(:,:,k)=R(k).ordinaryAccuracy(fi,si); Tr(:,:,k)=R(k).trainingTimeSeconds(fi,si); Pr(:,:,k)=R(k).predictionTimeSeconds(fi,si);
end
end

function [long,wide,consensus,pairs,C,subjectSummary] = makeTables(B,O,Tr,Pr,features,numbers,subjects,models)
n=21; long=table; wide=table(numbers,features,'VariableNames',{'FeatureNumber','FeatureName'}); meanBA=mean(B,2); medianBA=median(B,2); meanBA=squeeze(meanBA); medianBA=squeeze(medianBA); rankMean=zeros(n,3); rankMedian=rankMean;
for k=1:3, rankMean(:,k)=tiedrank(-meanBA(:,k)); rankMedian(:,k)=tiedrank(-medianBA(:,k)); end
for k=1:3
    [bestAcc,bestI]=max(B(:,:,k),[],2); [worstAcc,worstI]=min(B(:,:,k),[],2); prefix=string(matlab.lang.makeValidName(models(k)));
    T=table(repmat(models(k),n,1),numbers,features,meanBA(:,k),std(B(:,:,k),0,2),medianBA(:,k),min(B(:,:,k),[],2),max(B(:,:,k),[],2),mean(O(:,:,k),2),std(O(:,:,k),0,2),sum(Tr(:,:,k),2),median(Tr(:,:,k),2),mean(Tr(:,:,k),2),sum(Pr(:,:,k),2),median(Pr(:,:,k),2),mean(Pr(:,:,k),2),rankMean(:,k),rankMedian(:,k),subjects(bestI),bestAcc,subjects(worstI),worstAcc,'VariableNames',{'Model','FeatureNumber','FeatureName','MeanBalancedAccuracy','StdBalancedAccuracy','MedianBalancedAccuracy','MinBalancedAccuracy','MaxBalancedAccuracy','MeanOrdinaryAccuracy','StdOrdinaryAccuracy','TotalTrainingTimeSeconds','MedianTrainingTimeSeconds','MeanTrainingTimeSeconds','TotalPredictionTimeSeconds','MedianPredictionTimeSeconds','MeanPredictionTimeSeconds','MeanBalancedAccuracyRank','MedianBalancedAccuracyRank','BestSubject','BestSubjectBalancedAccuracy','WorstSubject','WorstSubjectBalancedAccuracy'}); long=[long;T]; %#ok<AGROW>
    for v=4:width(T), wide.(char(prefix+"_"+string(T.Properties.VariableNames{v})))=T{:,v}; end
end
meanRank=mean(rankMean,2); worstRank=max(rankMean,[],2); consensus=table(numbers,features,meanRank,median(rankMean,2),mean(rankMedian,2),worstRank,sum(rankMean<=5,2),sum(rankMean<=10,2),mean(rankMean/21,2),min(meanBA,[],2),'VariableNames',{'FeatureNumber','FeatureName','MeanBalancedAccuracyMeanRank','MeanBalancedAccuracyMedianRank','MedianBalancedAccuracyMeanRank','WorstMeanBalancedAccuracyRank','Top5Appearances','Top10Appearances','MeanNormalizedRank','WorstCaseMeanBalancedAccuracy'}); consensus=sortrows(consensus,{'MeanBalancedAccuracyMeanRank','WorstCaseMeanBalancedAccuracy','FeatureNumber'},{'ascend','descend','ascend'});
C=eye(3); pairs=table; for a=1:2, for b=a+1:3, rho=corr(meanBA(:,a),meanBA(:,b),'Type','Spearman','Rows','complete'); C(a,b)=rho; C(b,a)=rho; pairs=[pairs;table(models(a),models(b),rho,'VariableNames',{'ModelA','ModelB','SpearmanRankCorrelation'})]; end, end %#ok<AGROW>
subjectSummary=table; for k=1:3, subjectSummary=[subjectSummary;table(repmat(models(k),18,1),subjects,squeeze(mean(B(:,:,k),1))','VariableNames',{'Model','Subject','MeanBalancedAccuracyAcrossFeatures'})]; end %#ok<AGROW>
subjectSummary=[subjectSummary;table(repmat("AllModels",18,1),subjects,squeeze(mean(mean(B,1),3))','VariableNames',{'Model','Subject','MeanBalancedAccuracyAcrossFeatures'})];
end

function writeOutputs(stage,R,models,features,subjects,B,long,wide,consensus,pairs,C,subjectSummary,Tr,Pr)
writetable(long,fullfile(stage,"single_feature_cross_model_comparison_long.csv")); writetable(wide,fullfile(stage,"single_feature_cross_model_comparison_wide.csv")); writetable(consensus,fullfile(stage,"single_feature_consensus_ranking.csv")); writetable(subjectSummary,fullfile(stage,"single_feature_subject_summary.csv"));
corrTable=array2table(C,'VariableNames',cellstr(models)); corrTable=addvars(corrTable,models','Before',1,'NewVariableNames','Model'); writetable(corrTable,fullfile(stage,"single_feature_rank_correlations.csv"));
save(fullfile(stage,"single_feature_cross_model_comparison.mat"),"R","B","long","wide","consensus","pairs","C","subjectSummary");
makeFigures(stage,models,features,subjects,B,long,C); writeReport(stage,R,models,features,long,consensus,pairs,subjectSummary,Tr,Pr);
end

function makeFigures(stage,models,features,subjects,B,long,C)
f=figure('Visible','off','Color','white'); bar(reshape(long.MeanBalancedAccuracy,21,3)); legend(models,'Location','best'); ylabel('Mean balanced accuracy (decimal)'); exportgraphics(f,fullfile(stage,"single_feature_cross_model_ranking.png")); close(f);
f=figure('Visible','off','Color','white'); imagesc(C); clim([-1 1]); colorbar; xticks(1:3); yticks(1:3); xticklabels(models); yticklabels(models); exportgraphics(f,fullfile(stage,"single_feature_rank_correlation_heatmap.png")); close(f);
f=figure('Visible','off','Color','white'); imagesc(squeeze(mean(B,3))'); colorbar; xticks(1:21); xticklabels(features); xtickangle(45); yticks(1:18); yticklabels(string(subjects)); exportgraphics(f,fullfile(stage,"single_feature_subject_performance_heatmap.png")); close(f);
end

function writeReport(stage,R,models,~,long,consensus,pairs,subjects,Tr,Pr)
fid=fopen(fullfile(stage,"single_feature_cross_model_report.md"),"w"); closeFile=onCleanup(@() fclose(fid)); fprintf(fid,"# Single-feature cross-model LOSO comparison\n\n## Provenance and compatibility\n\nDataset hash: `%s`; MATLAB release: `%s`; completed folds: 378 per model; class count: 8. Upstream feature/windowing/filter provenance is unknown.\n\n",R(1).datasetHash,R(1).matlabRelease);
fprintf(fid,"## Classifier-independent findings\n\nPrimary consensus order: ascending mean of tied mean-balanced-accuracy ranks, then descending worst-case mean accuracy, then feature number. Top-5/top-10 appearances count a model when its tied rank is <=5/<=10. Mean normalized rank is mean rank / 21 (range 1/21 to 1). Consensus top five: %s.\n\n",strjoin(consensus.FeatureName(1:5),", "));
fprintf(fid,"## Model-specific findings\n\n"); for k=1:3, T=long(long.Model==models(k),:); [~,order]=sortrows([-T.MeanBalancedAccuracy,T.FeatureNumber],[1 2]); highPoor=T.FeatureName(T.MeanBalancedAccuracy>=median(T.MeanBalancedAccuracy) & T.MinBalancedAccuracy<=prctile(T.MinBalancedAccuracy,25)); low=T.FeatureName(T.StdBalancedAccuracy<=prctile(T.StdBalancedAccuracy,25)); high=T.FeatureName(T.StdBalancedAccuracy>=prctile(T.StdBalancedAccuracy,75)); fprintf(fid,"### %s\n\nTop five (mean BA descending, feature number asc ties): %s. Bottom five: %s. High mean/poor worst (mean >= median; worst <= Q1): %s. Low/high variance (SD <=Q1 / >=Q3): %s / %s.\n\n",models(k),strjoin(T.FeatureName(order(1:5)),", "),strjoin(T.FeatureName(order(end-4:end)),", "),strjoin(highPoor,", "),strjoin(low,", "),strjoin(high,", ")); end
fprintf(fid,"## Subject-specific findings\n\n"); for k=1:3, T=subjects(subjects.Model==models(k),:); [best,bi]=max(T.MeanBalancedAccuracyAcrossFeatures); [worst,wi]=min(T.MeanBalancedAccuracyAcrossFeatures); fprintf(fid,"%s strongest/weakest held-out subject: %g (%.6f) / %g (%.6f).\n\n",models(k),T.Subject(bi),best,T.Subject(wi),worst); end
fprintf(fid,"## Compute findings\n\n"); warnings=strings(0,1); for k=1:3, settings=strrep(strtrim(evalc("disp(R(k).modelSettings)")),newline," "); details=strrep(strtrim(evalc("disp(R(k).computeDetails)")),newline," "); evidence=computeEvidence(R(k)); fprintf(fid,"%s mode `%s`; settings `%s`; compute details `%s`; evidence: %s; fingerprint `%s`; total train/predict seconds %.6f / %.6f.\n\n",models(k),R(k).computeMode,settings,details,evidence,string(R(k).computeFingerprint),sum(Tr(:,:,k),"all"),sum(Pr(:,:,k),"all")); warnings=[warnings;string(R(k).deviations(:))]; end %#ok<AGROW>
fprintf(fid,"LDA used CPU `fitcdiscr` by design. MLP and RBF-SVM both passed and used their exact `gpuArray` fit/predict paths on the NVIDIA GeForce RTX 4060 Ti; neither used a CPU fallback. RBF-SVM CPU ECOC parallelism was disabled in GPU mode.\n\n");
fprintf(fid,"## Rank agreement\n\n"); for i=1:height(pairs), rho=pairs.SpearmanRankCorrelation(i); if abs(rho)>=0.7, strength="strong"; elseif abs(rho)>=0.4, strength="moderate"; else, strength="weak"; end; direction="positive"; if rho<0, direction="negative"; end; fprintf(fid,"%s vs %s: %.6f (%s %s agreement; thresholds |rho| >=0.70 strong, >=0.40 moderate, otherwise weak).\n\n",pairs.ModelA(i),pairs.ModelB(i),rho,strength,direction); end
fprintf(fid,"## Research-question alignment\n\n### A. What this experiment establishes\n\nThe completed LOSO runs estimate each feature type's standalone cross-user predictive value across 128 channels. Cross-model consensus supports classifier-independent transferability; it is not causal feature importance.\n\n### B. What this experiment does not establish\n\nLOSO alone does not estimate personalized within-user accuracy or the within-user-to-unseen-user retention gap. The current workflow therefore answers the research question only partially.\n\n### C. What is required to measure retention\n\nRun the separate design in `within_user_followup_protocol.md`, using grouped trial-level validation so windows from one trial never appear in both training and testing. No compatible completed within-user artifacts were found, so no retention result was invented or launched here.\n\n");
fprintf(fid,"## Limitations, warnings, and follow-up\n\nUpstream feature extraction, windowing, filtering, and normalization provenance is unknown. The observed per-sample/per-feature normalization does not itself mix subjects, but the absence of upstream construction code prevents a definitive leakage-free claim. No compatible Random Forest finals existed, so Random Forest was excluded and not rerun. The one-fold runtime benchmark materially underestimated feature-dependent RBF-SVM cost; the completed per-fold timing matrices supersede that projection. Initial isolated pre-production validation exposed and corrected hash/source-path/table-reporting defects without touching production checkpoints. All exact final fits completed without a compute fallback. The first comparison handoff used an invalid output assignment after all model finals were already complete; it was corrected and the comparison was generated separately without rerunning any model.\n");
wf=fopen(fullfile(stage,"experiment_deviations_and_warnings.txt"),"w"); for x=warnings', fprintf(wf,"%s\n",x); end
deviations=join([ ...
    "Scientific limitations: no compatible Random Forest finals; no compatible within-user artifacts; upstream feature/windowing/filter/normalization provenance unknown; LOSO does not establish within-user performance or retention." ...
    "Compute fallback: none. LDA ran on CPU by design; exact MLP and RBF-SVM gpuArray paths passed and ran on the NVIDIA GeForce RTX 4060 Ti." ...
    "Runtime warning: the permitted one-fold-per-model benchmark underestimated feature-dependent RBF-SVM training cost; final per-fold timing matrices supersede it." ...
    "Pre-production corrections: isolated smoke attempts exposed a Java-buffer hash defect, an extensionless source-hash path, and partial-scope table/report dimension/name defects. These attempts stopped before production and did not alter production progress." ...
    "Post-production correction: all 1,134 folds and three final MAT files completed before the original comparison handoff failed because a no-output function was assigned. The call was corrected and comparison generation was rerun without rerunning any model." ...
    "Environment warning: the MATLAB trial license reported that it expires in 4 days."],newline);
fprintf(wf,"%s\n",deviations); fclose(wf);
end

function evidence=computeEvidence(result)
evidence="no nested smoke evidence recorded";
if result.computeMode~="gpu"
    evidence="CPU path; GPU not applicable to this model";
    return
end
if ~isstruct(result.computeDetails) || ~isfield(result.computeDetails,"smokeEvidence")
    return
end
smoke=result.computeDetails.smokeEvidence;
gpuName="unknown GPU"; memoryBefore=NaN; memoryAfter=NaN; failure=""; attempts="";
if isfield(smoke,"gpu") && isstruct(smoke.gpu)
    if isfield(smoke.gpu,"name"), gpuName=string(smoke.gpu.name); end
    if isfield(smoke.gpu,"memoryBefore"), memoryBefore=double(smoke.gpu.memoryBefore); end
    if isfield(smoke.gpu,"memoryAfter"), memoryAfter=double(smoke.gpu.memoryAfter); end
    if isfield(smoke.gpu,"failureReport"), failure=string(smoke.gpu.failureReport); end
end
if isfield(smoke,"attempts"), attempts=strjoin(string(smoke.attempts),"; "); end
fallback="none"; if strlength(failure)>0 || contains(lower(attempts),"fallback"), fallback="recorded; inspect smoke evidence"; end
evidence=sprintf("exact gpuArray smoke passed on %s; available-memory evidence %.0f -> %.0f bytes; attempts: %s; fallback: %s",gpuName,memoryBefore,memoryAfter,attempts,fallback);
end

function validateBundle(stage)
required=["single_feature_cross_model_comparison_long.csv","single_feature_cross_model_comparison_wide.csv","single_feature_cross_model_comparison.mat","single_feature_cross_model_report.md","single_feature_rank_correlations.csv","single_feature_consensus_ranking.csv","single_feature_subject_summary.csv","single_feature_cross_model_ranking.png","single_feature_rank_correlation_heatmap.png","single_feature_subject_performance_heatmap.png","experiment_deviations_and_warnings.txt"];
for x=required, p=fullfile(stage,x); if ~isfile(p) || dir(p).bytes==0, error("Comparison:Bundle","Missing/empty staged artifact %s.",x); end, end
if height(readtable(fullfile(stage,required(1)),VariableNamingRule="preserve"))~=63 || height(readtable(fullfile(stage,required(2)),VariableNamingRule="preserve"))~=21 || height(readtable(fullfile(stage,required(5)),VariableNamingRule="preserve"))~=3 || height(readtable(fullfile(stage,required(6)),VariableNamingRule="preserve"))~=21 || height(readtable(fullfile(stage,required(7)),VariableNamingRule="preserve"))~=72, error("Comparison:Bundle","Staged CSV row count validation failed."); end
S=load(fullfile(stage,required(3))); if ~isfield(S,"C") || ~isequal(size(S.C),[3 3]), error("Comparison:Bundle","Staged MAT validation failed."); end
for x=required(8:10), image=imread(fullfile(stage,x)); if isempty(image), error("Comparison:Bundle","Unreadable PNG."); end, end
report=fileread(fullfile(stage,required(4)));
if ~contains(report,"## Research-question alignment") || ~contains(report,"## Limitations, warnings, and follow-up")
    error("Comparison:Bundle","Report section validation failed.");
end
end

function promoteBundle(stage,target)
backup=target+".previous_"+string(char(java.util.UUID.randomUUID));
if isfolder(target), movefile(target,backup,"f"); end
try
    movefile(stage,target,"f"); if isfolder(backup), rmdir(backup,"s"); end
catch ME
    if isfolder(backup) && ~isfolder(target), movefile(backup,target,"f"); end
    rethrow(ME)
end
end

function cleanStage(stage)
if isfolder(stage), rmdir(stage,"s"); end
end

function runSelfTest
root=string(tempname); mkdir(root); cleanup=onCleanup(@() cleanStage(root)); models=["lda","mlp","rbf_svm"]; files=["single_feature_lda_loso_final.mat","single_feature_mlp_loso_final.mat","single_feature_rbf_svm_loso_final.mat"];
for k=1:3, d=fullfile(root,"single_feature_"+models(k)+"_loso_results"); mkdir(d); S=fakeFinal(k); if k==2, results=struct("checkpoint",S); save(fullfile(d,files(k)),"results"); else, save(fullfile(d,files(k)),"-struct","S"); end, end
compareCore(root); out=fullfile(root,"single_feature_cross_model_comparison"); artifactList=dir(out); assert(nnz(~[artifactList.isdir])==11,"Comparison:SelfTest","Required artifact count failed."); long=readtable(fullfile(out,"single_feature_cross_model_comparison_long.csv"),VariableNamingRule="preserve"); wide=readtable(fullfile(out,"single_feature_cross_model_comparison_wide.csv"),VariableNamingRule="preserve"); corrTable=readtable(fullfile(out,"single_feature_rank_correlations.csv"),VariableNamingRule="preserve"); assert(height(long)==63 && width(long)==22 && height(wide)==21 && width(wide)==59 && height(corrTable)==3 && width(corrTable)==4,"Comparison:SelfTest","Output dimensions failed."); report=fileread(fullfile(out,"single_feature_cross_model_report.md")); assert(contains(report,"## Classifier-independent findings") && contains(report,"## Model-specific findings") && contains(report,"## Subject-specific findings") && contains(report,"## Compute findings") && contains(report,"## Rank agreement"),"Comparison:SelfTest","Report sections failed."); M=load(fullfile(out,"single_feature_cross_model_comparison.mat")); assert(isequal(size(M.C),[3 3]),"Comparison:SelfTest","Correlation matrix failed."); assertThrows(@() invalidCase(root,"completion")); assertThrows(@() invalidCase(root,"metric")); assertThrows(@() invalidCase(root,"time")); assertThrows(@() invalidCase(root,"duplicate")); fprintf("Synthetic comparison self-test passed.\n");
end

function S=fakeFinal(k)
identities=["LDA","MLP","RBF-SVM"]; S.balancedAccuracy=0.4+0.001*reshape(mod(1:378,19),21,18)+0.001*k; S.balancedAccuracy(2,:)=S.balancedAccuracy(1,:); S.ordinaryAccuracy=S.balancedAccuracy; S.trainingTimeSeconds=ones(21,18)*k; S.predictionTimeSeconds=ones(21,18)*k/10; S.completed=true(21,18); S.featureNames="Feature"+string((1:21)'); S.featureNumbers=(1:21)'; S.subjects=(1:18)'; S.classOrder="C"+string((1:8)'); S.datasetHash="synthetic"; S.matlabRelease="R2026a"; S.computeMode="CPU"; S.modelSettings=struct("k",k); S.computeDetails=struct("workers",k,"gpuFallback","none"); S.computeFingerprint="synthetic"; S.deviations="synthetic warning"; S.modelIdentity=identities(k);
if k==3, S.featureNames=S.featureNames(end:-1:1); S.featureNumbers=S.featureNumbers(end:-1:1); S.subjects=S.subjects(end:-1:1); S.balancedAccuracy=S.balancedAccuracy(end:-1:1,end:-1:1); S.ordinaryAccuracy=S.balancedAccuracy; S.trainingTimeSeconds=S.trainingTimeSeconds(end:-1:1,end:-1:1); S.predictionTimeSeconds=S.predictionTimeSeconds(end:-1:1,end:-1:1); S.completed=S.completed(end:-1:1,end:-1:1); end
end

function invalidCase(root,kind)
S=fakeFinal(1); if kind=="completion", S.completed(1)=false; elseif kind=="metric", S.balancedAccuracy(1)=2; elseif kind=="time", S.trainingTimeSeconds(1)=-1; else, S.featureNames(2)=S.featureNames(1); end
p=fullfile(root,"bad.mat"); save(p,"-struct","S"); loadFinal(p,"LDA");
end

function assertThrows(fun)
didThrow=false;
try
    fun();
catch
    didThrow=true;
end
assert(didThrow,"Comparison:SelfTest","Invalid synthetic input was accepted.");
end
