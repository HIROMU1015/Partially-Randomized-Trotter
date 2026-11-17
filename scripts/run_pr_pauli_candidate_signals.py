"""Complete one B2-selected Pauli candidate's signal carriers on an owned checkout.

The scientific implementation is A's existing descriptor-aware entry. This
wrapper never chooses L_D, formula, width, kappa or shots. By default it plans;
--execute acquires the plan's missing rounds and publishes raw/packet commits.
B intake, prices and Q3 remain the calling candidate coordinator's next stages.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.rpe_preparation_paths import runtime_request

ENTRY_SOURCE='6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb'
ENTRY_FILES=('src/trottertracks/pauli_signal_entry.py','scripts/run_pauli_signal_entry.py',
             'src/trottertracks/pauli_split_descriptor.py')
SUCCESS='ENTRY_CARRIERS_COMPLETE_NOT_RPE_QUALIFIED'
GIT_ID=['-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp']


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,path)


def fixed_file(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('FIXED_INPUT_CHANGED:'+str(path))
    else:write(path,value)


def git(workspace,*args):
    return subprocess.check_output(['git',*args],cwd=workspace,text=True).strip()


@contextmanager
def owned_checkout_lock(workspace):
    """Protect this checkout's source/HEAD through its scientific child."""
    path=Path(git(workspace,'rev-parse','--git-path','d2-pauli-signal-entry.lock'))
    if not path.is_absolute():path=workspace/path
    with path.open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def commit(workspace,paths,message):
    rel=[str(Path(p).relative_to(workspace))for p in paths if Path(p).exists()]
    if not rel:return git(workspace,'rev-parse','HEAD')
    subprocess.run(['git','add','--sparse','--',*rel],cwd=workspace,check=True,stdout=subprocess.DEVNULL)
    changed=subprocess.check_output(['git','diff','--cached','--name-only','--',*rel],cwd=workspace,text=True).strip()
    if changed:
        subprocess.run(['git',*GIT_ID,'commit','--only','-q','-m',message+' [skip ci]','--',*rel],
                       cwd=workspace,check=True)
    return git(workspace,'rev-parse','HEAD')


def publish(workspace,enabled=True):
    branch=git(workspace,'branch','--show-current')
    if not branch:raise ValueError('OWNED_PUBLICATION_BRANCH_REQUIRED')
    if enabled:
        subprocess.run(['git','push','origin',branch],cwd=workspace,check=True)
        remote=git(workspace,'ls-remote','--heads','origin',branch).split()[0]
        if remote!=git(workspace,'rev-parse','HEAD'):raise ValueError('PUBLICATION_REMOTE_MISMATCH')
    return branch


def live(marker,run):
    pid=marker.get('pid')
    if type(pid)is not int:return False
    try:return str(run)in Path('/proc',str(pid),'cmdline').read_bytes().decode(errors='replace')
    except OSError:return False


def entry(workspace,interpreter,arguments,log,marker=None):
    cmd=[interpreter,'-B',str(workspace/'scripts/run_pauli_signal_entry.py'),*arguments]
    if marker is not None:write(marker,dict(status='STARTING',UTC=utc(),command=cmd,automatic_retry=False))
    start=time.monotonic()
    with log.open('a')as stream:
        child=subprocess.Popen(cmd,cwd=workspace,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                               stdout=stream,stderr=subprocess.STDOUT)
        if marker is not None:write(marker,dict(status='RUNNING',UTC=utc(),pid=child.pid,command=cmd,automatic_retry=False))
        code=child.wait()
    if marker is not None:write(marker.with_name('SIGNALS_RETURN.json'),dict(returncode=code if code>=0 else None,
        signal=-code if code<0 else None,wall_seconds=time.monotonic()-start,UTC=utc(),automatic_retry=False))
    return code


def run(request,*,execute=False,push=True):
    request=runtime_request(request,('workspace','interpreter'))
    workspace=Path(request['workspace']).resolve()
    relative=Path(request['directory'])
    if relative.is_absolute()or'..'in relative.parts:raise ValueError('REPOSITORY_RELATIVE_OWNED_DIRECTORY_REQUIRED')
    base=(workspace/relative).resolve();base.relative_to(workspace)
    config=request['configuration'];descriptor=request.get('tail_descriptor')
    interpreter=request.get('interpreter','/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python')
    with owned_checkout_lock(workspace):
        for path in ENTRY_FILES:
            expected=subprocess.check_output(['git','show',ENTRY_SOURCE+':'+path],cwd=workspace)
            if(workspace/path).read_bytes()!=expected:raise ValueError('FIXED_A_ENTRY_SOURCE_DIFFERS:'+path)
        base.mkdir(parents=True,exist_ok=True)
        cfg=base/'CONFIGURATION.json';desc=base/'DESCRIPTOR.json';packets=base/'SAVED_PACKETS.json'
        fixed_file(cfg,config)
        if descriptor is not None:fixed_file(desc,descriptor)
        elif desc.exists():raise ValueError('FIXED_DESCRIPTOR_REMOVED')
        fixed_file(packets,request.get('saved_packets',[]))
        common=['--config',str(cfg),'--saved-packets',str(packets)]
        if descriptor is not None:common+=['--tail-descriptor',str(desc)]
        result=dict(schema='PR_PAULI_CANDIDATE_SIGNALS_V1',configuration_id=config['id'],
                    workspace=str(workspace),directory=str(relative),entry_source=ENTRY_SOURCE,
                    new_signal_calls=0,compile_calls=0,Q3_calls=0,automatic_retry=False)
        plan_dir=base/'plan'
        if not(plan_dir/'PLAN.json').exists():
            code=entry(workspace,interpreter,['--plan',*common,'--out',str(plan_dir)],base/'PLAN.log')
            if code!=0:return dict(result,status='STOP_PLAN_FAILED',reason='A_ENTRY_PLAN_FAILED',returncode=code)
        plan=read(plan_dir/'PLAN.json')
        result.update(request_rounds=plan.get('request_rounds',[]),reuse_rounds=plan.get('reuse_rounds',[]),
                      rebind_rounds=plan.get('rebind_rounds',[]),counts=plan.get('counts'))
        if plan.get('configuration_id')!=config['id']:raise ValueError('SAVED_PLAN_CONFIGURATION_DIFFERS')
        if plan.get('unsupported_rounds')or plan.get('configuration_reasons'):
            return dict(result,status='HELD_UNSUPPORTED_SIGNAL',reason=plan.get('configuration_reasons'),
                        unsupported_rounds=plan.get('unsupported_rounds'))
        if not execute:return dict(result,status='PLANNED_NOT_EXECUTED')
        state_file=base/'SIGNAL_STATE.json'
        state=read(state_file)if state_file.exists()else{}
        if state.get('status')=='SIGNALS_READY':
            # No file/job is resubmitted; retain its original execution and publication locators.
            return dict(state,resumed=True,new_signal_calls_this_invocation=0)
        source=state.get('execution_source')
        if source is None:
            source=commit(workspace,[cfg,desc,packets,plan_dir,base/'PLAN.log'],'Fix selected Pauli candidate signal inputs')
            publish(workspace,push)
            state=dict(result,status='INPUTS_FIXED',execution_source=source,
                       configuration_source=dict(commit=source,path=str(relative/'CONFIGURATION.json')),
                       descriptor_source=None if descriptor is None else dict(commit=source,path=str(relative/'DESCRIPTOR.json')))
            write(state_file,state)
        run_dir=base/'run01';marker=base/'SIGNALS_STARTED.json';terminal=run_dir/'result.json'
        if terminal.exists():
            if marker.exists()and read(marker).get('status')=='STARTING'and not(base/'SIGNALS_RETURN.json').exists():
                return dict(state,status='HELD_AMBIGUOUS_CHILD_LIFETIME',reason='PRESERVE_RESULT_NO_DUPLICATE_EXECUTION')
            saved=read(terminal)
            if saved.get('scientific_source')!=source:raise ValueError('TERMINAL_EXECUTION_SOURCE_DIFFERS')
            if marker.exists()and live(read(marker),run_dir):
                return dict(state,status='IN_PROGRESS',reason='ORIGINAL_SCIENTIFIC_CHILD_STILL_RUNNING')
        elif marker.exists()or run_dir.exists():
            if marker.exists()and live(read(marker),run_dir):
                return dict(state,status='IN_PROGRESS',reason='ORIGINAL_SCIENTIFIC_CHILD_STILL_RUNNING')
            return dict(state,status='HELD_INCOMPLETE_PRIOR_EXECUTION',reason='NO_AUTOMATIC_RETRY',new_signal_calls_this_invocation=0)
        else:
            if git(workspace,'rev-parse','HEAD')!=source:
                return dict(state,status='HELD_EXECUTION_SOURCE_CHANGED',reason='FIXED_SOURCE_NOT_CURRENT_NO_NEW_MEANS')
            code=entry(workspace,interpreter,['--execute','--approve-new-means','--source',source,*common,
                         '--out',str(run_dir)],base/'EXECUTE.log',marker)
            if not terminal.exists():return dict(state,status='STOP_MISSING_TERMINAL_RESULT',returncode=code)
            saved=read(terminal)
        raw=state.get('raw_commit')or commit(workspace,[run_dir,marker,base/'SIGNALS_RETURN.json',base/'EXECUTE.log'],
                   'Save selected Pauli candidate signal carriers')
        publish(workspace,push)
        state.update(raw_commit=raw,raw_result=dict(commit=raw,path=str(relative/'run01/result.json')),
                     signal_status=saved.get('status'),new_signal_calls=saved.get('scope',{}).get('new_mean_calls',0))
        write(state_file,state)
        if saved.get('status')!=SUCCESS:
            state.update(status='HELD_SIGNAL_GATE_OR_EXECUTION',reason=saved.get('error')or saved.get('status'))
            write(state_file,state);return state
        packet_dir=base/'packet';handoff=packet_dir/'MEAN_HANDOFF.json'
        if not handoff.exists():
            if packet_dir.exists()and any(packet_dir.iterdir()):
                return dict(state,status='HELD_INCOMPLETE_PACKET_ASSEMBLY',reason='PRESERVE_PARTIAL_ASSEMBLY')
            code=entry(workspace,interpreter,['--assemble',*common,'--source',source,'--raw-commit',raw,
                '--run-dir',str(relative/'run01')+'/', '--out',str(packet_dir)],base/'ASSEMBLE.log')
            if code!=0 or not handoff.exists():
                state.update(status='HELD_PACKET_ASSEMBLY',reason='A_ENTRY_ASSEMBLE_FAILED',returncode=code)
                write(state_file,state);return state
        packet=read(handoff);matches=[p for p in packet.get('packets',[])if(p.get('configuration_id')or(p.get('configuration')or{}).get('id'))==config['id']]
        if len(matches)!=1 or not matches[0].get('signal_coverage_complete'):
            state.update(status='HELD_PACKET_INCOMPLETE',reason='NO_COMPLETE_MATCHING_CONFIGURATION_PACKET')
            write(state_file,state);return state
        carrier=commit(workspace,[packet_dir,base/'ASSEMBLE.log'],'Assemble selected Pauli candidate signal packet')
        publish(workspace,push)
        state.update(status='SIGNALS_READY',packet_commit=carrier,
            packet_source=dict(commit=carrier,path=str(relative/'packet/MEAN_HANDOFF.json')),
            configuration_source=dict(commit=source,path=str(relative/'CONFIGURATION.json')),
            executed_source_distinct_from_result_commit=True,resumed=bool(state.get('raw_commit')))
        write(state_file,state);return state


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--no-push',action='store_true',help='technical fixture only; normal acquisition publishes its locators')
    a=p.parse_args(argv)
    try:result=run(read(a.request),execute=a.execute,push=not a.no_push)
    except (ValueError,OSError,subprocess.CalledProcessError)as exc:
        result=dict(schema='PR_PAULI_CANDIDATE_SIGNALS_V1',status='HELD_ORCHESTRATION_FAILURE',reason=repr(exc),automatic_retry=False)
    write(a.output,result);print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']in('SIGNALS_READY','PLANNED_NOT_EXECUTED')else 2


if __name__=='__main__':raise SystemExit(main())
