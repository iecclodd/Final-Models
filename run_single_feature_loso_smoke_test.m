function smoke = run_single_feature_loso_smoke_test(modelName)
%RUN_SINGLE_FEATURE_LOSO_SMOKE_TEST Exact, isolated full-size feature-1/subject-1 compute proof.
if nargin < 1 || strlength(string(modelName)) == 0, modelName="lda"; end
modelName=lower(string(modelName)); if ~ismember(modelName,["lda" "mlp" "rbf_svm"]), error("Unknown model."); end
root=string(fileparts(mfilename("fullpath"))); stamp=string(datetime("now","Format","yyyyMMdd_HHmmssSSS")); output=fullfile(root,"single_feature_loso_smoke_tests",stamp,modelName);
cfg=jsondecode(fileread(fullfile(root,"single_feature_loso_locked_config.json"))); datasetPath=fullfile(root,cfg.datasetFile);
prod=fullfile(root,"single_feature_"+modelName+"_loso_results","single_feature_"+modelName+"_loso_progress.mat"); before=localInfo(prod); guard=onCleanup(@()localGuard(prod,before));
gpuBefore=localGpuEvidence(); gpu=struct("name",gpuBefore.name,"memoryBefore",gpuBefore.availableMemory,"memoryAfter",NaN,"failureReport",gpuBefore.failureReport); decision=struct("success",false,"model",modelName,"compute",struct("mode","cpu","detail",""),"attempts",{{}},"datasetHash",localHash(datasetPath),"configHash",localHash(fullfile(root,"single_feature_loso_locked_config.json")),"codeHash",localHash(fullfile(root,"run_single_feature_loso_model.m")),"matlabRelease",version("-release"),"gpu",gpu,"timestampUtc",char(datetime("now","TimeZone","UTC")));
if modelName=="lda"
    smoke=run_single_feature_loso_model(modelName,"FeatureNumbers",1,"SubjectIDs",1,"OutputRoot",output,"SkipSmokeGate",true,"ComputeModeOverride","cpu");
    decision.success=true; decision.compute=smoke.checkpoint.settings.compute; decision.attempts={"exact CPU fitcdiscr/predict passed"};
else
    try
        smoke=run_single_feature_loso_model(modelName,"FeatureNumbers",1,"SubjectIDs",1,"OutputRoot",output,"SkipSmokeGate",true,"ComputeModeOverride","gpu");
        decision.success=true; decision.compute=smoke.checkpoint.settings.compute; decision.attempts={"exact gpuArray fit/predict passed"};
    catch ME
        gpuFailure=string(getReport(ME,"basic","hyperlinks","off")); decision.gpu.failureReport=gpuFailure; decision.attempts={"GPU exact fit/predict failed: "+gpuFailure};
        fallback="cpu"; if modelName=="rbf_svm", fallback="cpu_parallel"; end
        try
            smoke=run_single_feature_loso_model(modelName,"FeatureNumbers",1,"SubjectIDs",1,"OutputRoot",output,"SkipSmokeGate",true,"ComputeModeOverride",fallback);
            decision.success=true; decision.compute=smoke.checkpoint.settings.compute; decision.compute.detail="Controlled fallback after exact GPU failure"; decision.attempts{end+1}="exact controlled fallback fit/predict passed";
        catch fallbackError
            decision.attempts{end+1}="fallback failed: "+string(getReport(fallbackError,"basic","hyperlinks","off")); localWriteDecision(root,modelName,decision); rethrow(fallbackError)
        end
    end
end
gpuAfter=localGpuEvidence(); decision.gpu.memoryAfter=gpuAfter.availableMemory;
localWriteDecision(root,modelName,decision); save(fullfile(output,"smoke_summary.mat"),"smoke","decision"); fid=fopen(fullfile(output,"smoke_summary.json"),"w"); fprintf(fid,"%s\n",jsonencode(decision)); fclose(fid);
end
function localWriteDecision(root,model,decision)
folder=fullfile(root,"single_feature_loso_smoke_tests");
if ~isfolder(folder), mkdir(folder); end
file=fullfile(folder,"compute_decisions.json"); S=struct();
if isfile(file)
    try
        S=jsondecode(fileread(file));
    catch
        copyfile(file,file+".invalid_"+string(datetime("now","Format","yyyyMMdd_HHmmssSSS"))); S=struct();
    end
end
S.(char(model))=decision; temp=string(tempname(folder))+".json"; fid=fopen(temp,"w"); fprintf(fid,"%s\n",jsonencode(S)); fclose(fid); jsondecode(fileread(temp)); movefile(temp,file,"f");
end
function g=localGpuEvidence()
    g=struct("name","unavailable","availableMemory",NaN,"failureReport","");
    try
        d=gpuDevice; wait(d); g.name=string(d.Name); g.availableMemory=d.AvailableMemory;
    catch ME
        g.failureReport=string(getReport(ME,"basic","hyperlinks","off"));
    end
end
function localGuard(path,before), after=localInfo(path); if ~isequal(before,after), error("Smoke altered production progress artifact."); end; end
function info=localInfo(path), if isfile(path),d=dir(path);info=struct("exists",true,"bytes",d.bytes,"datenum",d.datenum,"sha256",localHash(path));else,info=struct("exists",false,"bytes",0,"datenum",0,"sha256","");end,end
function h=localHash(file)
md=java.security.MessageDigest.getInstance("SHA-256");
in=java.io.FileInputStream(java.io.File(char(file)));
cleanup=onCleanup(@()in.close);
channel=in.getChannel();
buffer=java.nio.ByteBuffer.allocateDirect(1048576);
while channel.read(buffer)>=0
    buffer.flip();
    md.update(buffer);
    buffer.clear();
end
h=upper(reshape(dec2hex(typecast(md.digest(),"uint8"),2)',1,[]));
clear cleanup
end
