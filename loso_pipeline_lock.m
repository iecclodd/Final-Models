function owner=loso_pipeline_lock(action,root,owner,state)
%LOSO_PIPELINE_LOCK Exclusively acquire and owner-update the pipeline lock.
arguments
    action (1,1) string
    root (1,1) string
    owner struct = struct()
    state struct = struct()
end

lockFile=fullfile(root,"pipeline_lock.json");
statusFile=fullfile(root,"run_status.json");
markdownFile=fullfile(root,"RUN_STATUS.md");
action=lower(action);

switch action
    case "acquire"
        pid=matlabProcessID;
        if isfile(lockFile)
            existing=localReadJson(lockFile);
            existingPid=localPid(existing);
            existingPhase=localFieldString(existing,"phase");
            if existingPid>0 && localPidAlive(existingPid) && ismember(existingPhase,["starting" "running"])
                error("LOSO:ActivePipelineLock","Active pipeline lock belongs to live PID %d.",existingPid);
            end
            localArchive(lockFile);
            delete(lockFile);
        end
        try
            attributes=javaArray('java.nio.file.attribute.FileAttribute',0);
            java.nio.file.Files.createFile(java.io.File(char(lockFile)).toPath,attributes);
        catch ME
            error("LOSO:LockAcquire","Could not exclusively create pipeline lock: %s",ME.message);
        end
        owner=struct("matlab_pid",pid,"owner_nonce",char(java.util.UUID.randomUUID), ...
            "start_time",char(datetime("now","TimeZone","UTC")));
        record=localMerge(state,owner);
        try
            localPublish(record,lockFile,statusFile,markdownFile);
        catch ME
            if isfile(lockFile), localArchive(lockFile); delete(lockFile); end
            rethrow(ME)
        end
    case {"update" "complete" "fail"}
        localRequireOwner(lockFile,owner);
        record=localMerge(state,owner);
        localPublish(record,lockFile,statusFile,markdownFile);
    case "cleanup"
        if ~isfile(lockFile), return; end
        try
            current=localReadJson(lockFile);
            if ~localOwnerMatches(current,owner), return; end
            if ismember(localFieldString(current,"phase"),["completed" "failed"]), return; end
            current.phase="failed";
            current.blocking_issue="MATLAB exited before normal completion.";
            current.next_action="Inspect the console log and resume from the validated checkpoint.";
            current.timestamp_utc=char(datetime("now","TimeZone","UTC"));
            localPublish(current,lockFile,statusFile,markdownFile);
        catch
            % Cleanup must not mask the original experiment error.
        end
    otherwise
        error("LOSO:LockAction","Unknown lock action: %s",action);
end
end

function localPublish(record,lockFile,statusFile,markdownFile)
localAtomicJson(record,lockFile);
localAtomicJson(record,statusFile);
text="# RUN STATUS"+newline+newline+ ...
    "- Current phase: "+localDisplay(record.phase)+newline+ ...
    "- Current model: "+localDisplay(record.model)+newline+ ...
    "- MATLAB PID: "+localDisplay(record.matlab_pid)+newline+ ...
    "- Compute mode: "+localDisplay(record.compute_mode)+newline+ ...
    "- Completed folds: "+localDisplay(record.completed_folds)+newline+ ...
    "- Total folds: "+localDisplay(record.total_folds)+newline+ ...
    "- Last completed feature: "+localDisplay(record.last_feature)+newline+ ...
    "- Last held-out subject: "+localDisplay(record.last_held_subject)+newline+ ...
    "- Progress-file path: `"+localDisplay(record.progress_file)+"`"+newline+ ...
    "- Console-log path: `"+localDisplay(record.console_log)+"`"+newline+ ...
    "- Last checkpoint timestamp: "+localDisplay(record.last_checkpoint_timestamp)+newline+ ...
    "- Blocking issue: "+localDisplay(record.blocking_issue)+newline+ ...
    "- Next action: "+localDisplay(record.next_action)+newline;
localAtomicText(text,markdownFile);
end

function localAtomicJson(value,path)
folder=fileparts(path); temp=string(tempname(folder))+".json";
fid=fopen(temp,"w"); if fid<0, error("LOSO:LockWrite","Could not create temporary state file."); end
cleanup=onCleanup(@() localClose(fid)); fprintf(fid,"%s\n",jsonencode(value)); if fclose(fid)~=0, error("LOSO:LockWrite","Could not close temporary state file."); end; clear cleanup
jsondecode(fileread(temp)); localAtomicReplace(temp,path);
end

function localAtomicText(value,path)
folder=fileparts(path); temp=string(tempname(folder))+".md";
fid=fopen(temp,"w"); if fid<0, error("LOSO:StatusWrite","Could not create temporary status file."); end
value=string(value); value(ismissing(value))=""; value=join(value,"");
cleanup=onCleanup(@() localClose(fid)); fprintf(fid,"%s",char(value)); if fclose(fid)~=0, error("LOSO:StatusWrite","Could not close temporary status file."); end; clear cleanup
if strlength(string(fileread(temp)))==0, error("LOSO:StatusWrite","Temporary status file is empty."); end
localAtomicReplace(temp,path);
end

function localAtomicReplace(temp,dest)
options=javaArray('java.nio.file.CopyOption',2);
options(1)=java.nio.file.StandardCopyOption.ATOMIC_MOVE;
options(2)=java.nio.file.StandardCopyOption.REPLACE_EXISTING;
java.nio.file.Files.move(java.io.File(char(temp)).toPath,java.io.File(char(dest)).toPath,options);
end

function localRequireOwner(lockFile,owner)
if ~isfile(lockFile), error("LOSO:LockLost","Pipeline lock no longer exists."); end
current=localReadJson(lockFile);
if ~localOwnerMatches(current,owner), error("LOSO:LockOwnership","Pipeline lock is not owned by this run."); end
end

function match=localOwnerMatches(record,owner)
match=isfield(record,"matlab_pid") && isfield(record,"owner_nonce") && ...
    isfield(owner,"matlab_pid") && isfield(owner,"owner_nonce") && ...
    double(record.matlab_pid)==double(owner.matlab_pid) && string(record.owner_nonce)==string(owner.owner_nonce);
end

function record=localMerge(state,owner)
record=state; record.matlab_pid=owner.matlab_pid; record.owner_nonce=owner.owner_nonce;
record.start_time=owner.start_time;
end

function value=localReadJson(path)
try
    value=jsondecode(fileread(path));
catch ME
    error("LOSO:LockInvalid","Invalid pipeline lock: %s",ME.message);
end
end

function pid=localPid(record)
pid=0;
if isfield(record,"matlab_pid") && isnumeric(record.matlab_pid) && isscalar(record.matlab_pid) && isfinite(record.matlab_pid), pid=double(record.matlab_pid);
elseif isfield(record,"pid") && isnumeric(record.pid) && isscalar(record.pid) && isfinite(record.pid), pid=double(record.pid); end
end

function value=localFieldString(record,name)
value=""; if isfield(record,name), value=string(record.(name)); end
end

function localArchive(path)
archive=path+".archive_"+string(datetime("now","Format","yyyyMMdd_HHmmssSSS"));
copyfile(path,archive);
end

function value=localDisplay(input)
if isempty(input)
    value="";
else
    value=string(input);
    value(ismissing(value))="";
end
end

function localClose(fid)
try
    fclose(fid);
catch
    % Best-effort cleanup after a failed state write.
end
end

function alive=localPidAlive(pid)
if ispc
    [status,out]=system(sprintf('tasklist /FI "PID eq %d" /NH',pid));
    alive=status==0 && contains(string(out),string(pid));
else
    [status,~]=system(sprintf('kill -0 %d',pid)); alive=status==0;
end
end
