"""Technical seals and explicitly selected independent one-shot contracts.

No scientific imports, preparation restoration or numerical accounting at import.
Approval is separate from a technical seal. The preparation CLI cannot approve.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import pr_cost_contract as old
require,digest,read,save,sha=old.require,old.digest,old.read,old.save,old.sha
SCOPES=tuple(s+'_'+t for s in ('H4','H6','H8') for t in ('A_COST','D_ACCOUNTING','B_ANALYSIS'))
FILES=('scripts/pr_cost_batch.py','scripts/pr_cost_batch_science.py','scripts/prepare_pr_cost_batch.py',
       'tests/test_pr_cost_batch.py','scripts/pr_cost_candidate_parallel.py',
       'scripts/prepare_pr_candidate_parallel.py','scripts/h8_key_compile_parallel.py',
       'tests/test_pr_cost_candidate_parallel.py',*old.SOURCES)


def metadata_binding(spec):
    copy=dict(spec);copy.pop('technical_digest',None)
    return digest(copy)


def sealed(spec):
    require(spec['scope'] in SCOPES and spec['technical_sealed'] is True,'TECHNICAL_SEAL')
    require(spec['technical_digest']==metadata_binding(spec),'TECHNICAL_DIGEST')
    require(spec['approved'] is False and spec['runtime_authorization'] is False,'TECHNICAL_SEAL_IS_NOT_AUTHORIZATION')
    require(spec['retry'] is False and spec['next_stage_authorized'] is False,'ONE_SHOT_SCOPE')
    return True


def authorize(spec,grant):
    sealed(spec)
    if spec.get('A_scheduling'):
        require(spec['A_scheduling']=='H8_CANDIDATE_CONTROL_CONTINUATION_V1' and spec['scope']=='H8_A_COST'
                and grant.get('candidate_parallel_handoff_authorized') is True
                and grant.get('origin_SOURCE')==spec['continuation_origin']['origin_SOURCE'],
                'EXPLICIT_CANDIDATE_HANDOFF_APPROVAL_REQUIRED')
    require(grant.get('scope')==spec['scope'] and grant.get('technical_digest')==spec['technical_digest'],
            'INDIVIDUAL_SCOPE_BINDING')
    require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True
            and grant.get('one_shot') is True and grant.get('retry') is False
            and grant.get('next_stage_authorized') is False,'EXPLICIT_SELECTED_SCOPE_APPROVAL_REQUIRED')
    require(grant.get('SOURCE')==spec['SOURCE'] and grant.get('approval_receipt_sha256'), 'APPROVAL_SOURCE_RECEIPT')
    require(grant.get('allowed_cpus') and set(grant['allowed_cpus'])<=os.sched_getaffinity(0),'FRESH_OWN_CPUS')
    require(len(grant['allowed_cpus'])==len(set(grant['allowed_cpus'])),'UNIQUE_OWN_CPUS')
    require(type(grant.get('memory_admission_estimate_bytes')) is int
            and grant['memory_admission_estimate_bytes']>=spec['memory_estimate_per_worker_bytes']
                * (len(grant['allowed_cpus'])+1 if spec['stage']=='A_COST' else 1),'MEMORY_ESTIMATE_NOT_A_CAP')
    require(sha(grant['approval_receipt'])==grant['approval_receipt_sha256'],'USER_APPROVAL_RECEIPT')


def fresh_admission(spec,grant):
    """Own-input resources only; unrelated worktree/PSI changes are observations."""
    require(set(grant['allowed_cpus']).isdisjoint(grant['concurrent_job_cpus']),'CONCURRENT_JOB_CPU_EXCLUSION')
    observation=old.fresh(Path(spec['output']).parent)
    require(observation['memory_available_bytes']>=grant['memory_admission_estimate_bytes']
            and observation['disk_free_bytes']>=spec['output_estimate_bytes']
            and observation['free_inodes']>=spec['inode_estimate'],'FRESH_RESOURCE_ADMISSION')
    # Absence of quota tools never implies unlimited quota. Accepting this
    # site limitation must be an explicit part of the future selected approval.
    require(grant.get('quota_policy') in ('AVAILABLE_VERIFIED','NO_QUOTA_CONFIRMED',
            'USER_ACCEPTS_UNEXPOSED_QUOTA_ENOSPC_FAIL_CLOSED'),'EXPLICIT_QUOTA_POLICY_REQUIRED')
    if grant['quota_policy']=='AVAILABLE_VERIFIED':
        require(grant['quota_remaining_bytes']>=spec['output_estimate_bytes']
                and grant['quota_remaining_inodes']>=spec['inode_estimate'],'QUOTA_ADMISSION')
    return observation


def verify(spec,*,runtime=True,source=True):
    """Byte-only verification; never imports the scientific dispatch module."""
    sealed(spec)
    if source:old.verify_source(ROOT,spec['SOURCE'],spec['source_hashes'])
    old.verify_byte_rows(spec['inputs'])
    require(sha(spec['python'])==spec['python_sha256'],'PYTHON_BINARY_CHANGED')
    if runtime:
        receipt=read(spec['runtime_receipt'])
        require(sha(spec['runtime_receipt'])==spec['runtime_receipt_sha256'],'RUNTIME_RECEIPT')
        # The saved SHA observation is reused only while each actual leaf's
        # device/inode/size/mtime is unchanged. A changed leaf is rehashed.
        for path,h in receipt['expected'].items():
            st=Path(path).stat();facts=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]
            if facts!=receipt['verified_stats'][path]:require(sha(path)==h,'RUNTIME_CHANGED')
    for directory,files in spec.get('provider_closure',{}).items():
        for path,h in files.items():
            relative=Path(path);require(not relative.is_absolute() and '..' not in relative.parts,'PROVIDER_PATH')
            require(sha(Path(directory)/relative)==h,'PROVIDER_SOURCE_CHANGED')
    if spec.get('A_scheduling'):
        from pr_cost_candidate_parallel import POLICY,verify_manifest,verify_origin
        require(spec['A_scheduling']==POLICY and spec['scope']=='H8_A_COST','EXACT_CANDIDATE_SCOPE')
        payload=read(spec['payload']);tasks=payload['new_seed_registry']['tasks'];origin=spec['continuation_origin']
        require(spec['seed_derivation_SOURCE']==origin['origin_SOURCE'],'ORIGINAL_SEED_DERIVATION_SOURCE')
        verify_origin(origin,tasks,spec['binding'])
        verify_manifest(read(spec['resume_manifest']),tasks,spec['binding'],origin)
    return dict(status='PASS_BYTES_ONLY',scope=spec['scope'],SOURCE=spec['SOURCE'],scientific_calls=0)


def select(overview,approval):
    require(approval.get('approved_by_user') is True and approval.get('runtime_authorization') is True
            and approval.get('retry') is False and approval.get('next_stage_authorized') is False,
            'EXPLICIT_BATCH_SELECTION_APPROVAL_REQUIRED')
    require(approval.get('batch_digest')==overview['batch_digest'],'BATCH_DIGEST')
    value=dict(overview);value.pop('batch_digest')
    require(digest(value)==overview['batch_digest'],'SEALED_BATCH_CONTENT')
    selected=approval.get('selected_scopes',[])
    require(selected and len(selected)==len(set(selected)) and set(selected)<=set(SCOPES),'EXPLICIT_UNIQUE_SCOPE_LIST')
    require(set(approval['individual_grants'])==set(selected),'EXACT_SELECTED_GRANTS_NO_AUTOMATIC_OTHER_STAGE')
    by={x['scope']:x for x in overview['contracts']}
    return selected


def verify_scoped_paths(spec):
    output=old.home_path(spec['output']);lock=old.home_path(spec['lock'])
    require(output!=lock and not output.exists() and not lock.exists(),'UNUSED_ONE_SHOT_OUTPUT')
    require(not output.is_relative_to(lock) and not lock.is_relative_to(output),'INDEPENDENT_OUTPUT_LOCK')
    return output,lock


def run_contract(spec,grant):
    authorize(spec,grant);verify(spec)
    fresh=fresh_admission(spec,grant)
    require(sha(grant['approval_receipt'])==grant['approval_receipt_sha256'],'USER_APPROVAL_RECEIPT')
    output,lock=verify_scoped_paths(spec);lock.parent.mkdir(parents=True,exist_ok=True)
    old.claim_attempt(lock,spec['SOURCE'],spec['scope']);output.mkdir(parents=True,exist_ok=False)
    # Affinity and environment settings apply only to this newly authorized child.
    os.sched_setaffinity(0,set(grant['allowed_cpus']))
    for name in ('TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR'):
        path=output/'scratch'/name;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):os.environ[name]='1'
    try:
        save(output/'startup_gate.json',dict(status='PASS',SOURCE=spec['SOURCE'],scope=spec['scope'],fresh=fresh))
        import pr_cost_batch_science as science
        result=science.execute(spec,grant,output)
        save(output/'result.json',result)
        verify(spec)
        save(output/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',scope=spec['scope'],
            SOURCE=spec['SOURCE'],result_sha256=sha(output/'result.json'),next_stage_authorized=False))
        return 0
    except BaseException as error:
        old.first_stop(output,str(error) or type(error).__name__)
        raise


def identity(pid):
    path=Path('/proc')/str(pid)
    text=(path/'stat').read_text();tail=text[text.rfind(')')+2:].split()
    return dict(pid=pid,starttime=int(tail[19]),uid=path.stat().st_uid,pgid=os.getpgid(pid))


def stop_owned(process,owned):
    if process.poll() is not None:
        process.wait();return
    try:now=identity(process.pid)
    except (FileNotFoundError,ProcessLookupError):
        process.wait();return
    require(now==owned and owned['uid']==os.getuid() and owned['pgid']==process.pid,'OWN_SESSION_STOP_ONLY')
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        process.wait();return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:current=identity(process.pid)
        except (FileNotFoundError,ProcessLookupError):
            process.wait();return
        require(current==owned,'OWN_IDENTITY_BEFORE_KILL')
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


def cleanup_all(processes):
    errors=[]
    for scope,(process,owned) in processes.items():
        try:stop_owned(process,owned)
        except Exception as error:errors.append(dict(scope=scope,error=str(error)))
    return errors


def launch(overview,approval):
    selected=select(overview,approval);processes={};results={};handles=[]
    # All selected grants are checked before the first scientific child starts.
    runnable=[];used_cpus=set();total_memory_estimate=0;total_output=0;total_inodes=0
    by={x['scope']:x for x in overview['contracts']}
    for scope in selected:
        try:
            require(by[scope]['status'] in ('READY','READY_PARTIAL_DIAGNOSTIC'),'CONTRACT_EVIDENCE_NOT_READY')
            spec=read(overview['private_specs'][scope]);grant=approval['individual_grants'][scope]
            authorize(spec,grant);verify(spec);verify_scoped_paths(spec)
            fresh_admission(spec,grant)
            require(set(grant['allowed_cpus']).isdisjoint(used_cpus),'PARALLEL_CPU_COLLISION')
            used_cpus.update(grant['allowed_cpus'])
            total_memory_estimate+=grant['memory_admission_estimate_bytes']
            total_output+=spec['output_estimate_bytes'];total_inodes+=spec['inode_estimate'];runnable.append(scope)
        except Exception as error:
            results[scope]=dict(status='PRESTART_STOP',reason=str(error),scientific_calls=0)
    require(runnable,'NO_READY_SELECTED_CONTRACTS')
    fresh=old.fresh(ROOT)
    require(fresh['memory_available_bytes']>=total_memory_estimate
            and fresh['disk_free_bytes']>=total_output and fresh['free_inodes']>=total_inodes,
            'FRESH_BATCH_RESOURCE_ADMISSION')
    batch_lock=old.home_path(overview['batch_lock']);batch_lock.parent.mkdir(parents=True,exist_ok=True)
    old.claim_attempt(batch_lock,overview['SOURCE'],'SELECTED_BATCH')
    try:
        for scope in runnable:
            try:
                spec=read(overview['private_specs'][scope]);grant_path=Path(overview['authorization_directory'])/(scope+'_APPROVED.json')
                save(grant_path,approval['individual_grants'][scope])
                log=Path(overview['authorization_directory'])/(scope+'.log');handle=log.open('x');handles.append(handle)
                command=[spec['python'],'-P','-B',str(ROOT/'scripts/pr_cost_batch.py'),'--contract',
                         overview['private_specs'][scope],'--grant',str(grant_path)]
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                try:owned=identity(process.pid)
                except (FileNotFoundError,ProcessLookupError):
                    results[scope]=dict(exit_code=process.wait(),reaped=True,exited_before_identity_observation=True);continue
                processes[scope]=(process,owned)
            except Exception as error:
                results[scope]=dict(status='START_STOP',reason=str(error),automatic_retry=False)
        # A failed scope is recorded while independent siblings keep running.
        while any(scope not in results for scope in processes):
            for scope,(process,owned) in processes.items():
                if scope not in results and process.poll() is not None:
                    results[scope]=dict(exit_code=process.wait(),reaped=True,owned_identity=owned)
            if any(scope not in results for scope in processes):time.sleep(.5)
    except BaseException:
        cleanup_errors=cleanup_all(processes)
        save(Path(overview['authorization_directory'])/'batch_interruption_cleanup.json',dict(errors=cleanup_errors))
        raise
    finally:
        for handle in handles:handle.close()
    save(Path(overview['authorization_directory'])/'batch_native_completion.json',
         dict(results=results,automatic_retry=False,next_stage_authorized=False,mandatory_STOP=True))
    return results


def main():
    def interrupted(signum,frame):raise KeyboardInterrupt('OWN_RUN_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path);parser.add_argument('--grant',type=Path)
    parser.add_argument('--batch',type=Path);parser.add_argument('--approval',type=Path)
    parser.add_argument('--verify-technical',action='store_true')
    args=parser.parse_args()
    try:
        if args.verify_technical:
            require(args.contract and not args.grant and not args.approval,'TECHNICAL_ONLY_MODE')
            print(json.dumps(verify(read(args.contract))));return 0
        if args.batch:
            require(args.approval,'EXPLICIT_BATCH_APPROVAL_REQUIRED')
            launch(read(args.batch),read(args.approval));return 0
        require(args.contract and args.grant,'CONTRACT_AND_GRANT_REQUIRED')
        return run_contract(read(args.contract),read(args.grant))
    except (Exception,KeyboardInterrupt) as error:
        print('STOP: '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
