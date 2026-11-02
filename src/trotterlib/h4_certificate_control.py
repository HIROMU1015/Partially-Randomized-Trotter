"""C authority and durable native audit. Standard library only until authorization.

No grant is issued implicitly. All scientific reservations are non-refundable.
Each detached witness persists its own child's native wait result independently
of the top-level supervisor. No host-wide controls or foreign process signals.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

LIMITS = dict(operator_preparation_reconstruction_batches=2, probability_mapping_batches=2,
    reference_signal_enclosures=6, nominal_law_mean_enclosures=12,
    implemented_PR_law_mean_enclosures=6, PR_normalization_enclosures=6,
    dense_matrix_exponential_requests=39)
ZERO = ('SCF','FCI','new_molecular_DF_generation','new_state_optimization','MC_sampling',
        'RNG_replay','circuit_compilation','shot_calculation','G_accounting')
PRECISIONS = (64,96,128)
ROOT = Path(__file__).resolve().parents[2]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(p, value, *, exclusive=True):
    """Durable exclusive artifacts; mutable run ledger uses an atomic rename."""
    p=Path(p);data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    flags=os.O_WRONLY|os.O_CREAT|(os.O_EXCL if exclusive else os.O_TRUNC)
    target=p if exclusive else p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    fd=os.open(target,flags,0o600)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if not exclusive:os.replace(target,p)
        d=os.open(p.parent,os.O_DIRECTORY);os.fsync(d);os.close(d)
    finally:
        if not exclusive and target.exists():target.unlink()


def hashes(manifest):
    for name, expected in manifest.items():
        p=Path(name);require(p.is_file() and not p.is_symlink() and sha(p)==expected,'HASH_MISMATCH:'+name)


def validate_contract(c, *, active=False):
    require(c['precision_schedule_bits']==list(PRECISIONS),'PRECISION_CONTRACT')
    require({k:c['total_call_budget_max'][k] for k in LIMITS}==LIMITS,'CALL_CONTRACT')
    require(all(c['total_call_budget_max'].get(k)==0 for k in ZERO),'FORBIDDEN_CALL_CONTRACT')
    require(c['cases']==['pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR'],'CASE_CONTRACT')
    require(c['epsilon_sig']==[.05,.01,.005,.001,.0005] and c['axes']==['cosine','sine'],'AXIS_CONTRACT')
    require(c['fixed_model']==dict(T=.8,q=8,r=4,K=4,product_formula='S2',full_dimension=256),'MODEL_CONTRACT')
    require(c['mandatory_STOP'] is True and c['automatic_retry'] is False,'STOP_CONTRACT')
    if active:
        require(all(c[k] is True for k in ('execution_authorized','grant_valid','dispatch_enabled')),'C_EXECUTION_NOT_AUTHORIZED')
        require(c.get('approval_sha256') and c.get('approval_file'),'EXPLICIT_APPROVAL_REQUIRED')
        require(sha(c['approval_file'])==c['approval_sha256'],'APPROVAL_HASH')
    return c


def verify_seal(seal):
    require(seal['kind']=='C_PRODUCTION_TECHNICAL_SOURCE_NOT_EXECUTION_GRANT','WRONG_SEAL')
    for group in ('code_files','runtime_files','legacy_source_files','legacy_runtime_files','input_files'):
        hashes(seal[group])
    require(sha(sys.executable)==seal['python_executable_sha256'],'WRONG_EXECUTABLE')
    # Commit identity is pinned, while unrelated worktree HEAD/status are irrelevant.
    for path,digest_value in seal['code_git_files'].items():
        data=subprocess.check_output(['git','show',seal['SOURCE']+':'+path],cwd=ROOT)
        require(hashlib.sha256(data).hexdigest()==digest_value,'SOURCE_COMMIT_BINDING:'+path)



def validate_routing(c,seal):
    for name,value in seal['routing'].items():
        require(c.get(name)==value,'SEALED_ROUTING_MISMATCH:'+name)
    require(c['historical_scientific_SOURCE']==seal['historical_scientific_SOURCE'],'HISTORICAL_SOURCE_BINDING')
    require(c['input_manifest_sha256']==seal['input_manifest_sha256'],'INPUT_MANIFEST_BINDING')


def issue_grant(contract_path, seal_path, namespace):
    c=validate_contract(read(contract_path),active=True);seal=read(seal_path);verify_seal(seal)
    validate_routing(c,seal)
    require(sha(seal_path)==c['technical_seal_sha256'],'SEAL_BINDING')
    ns=Path(namespace).resolve();require(str(ns)==c['output_namespace'],'NAMESPACE_BINDING')
    require(not ns.exists(),'NAMESPACE_ALREADY_USED')
    ns.mkdir(parents=True)
    grant=dict(kind='C_SCIENCE_ONE_SHOT',namespace=str(ns),contract_sha256=sha(contract_path),
        seal_sha256=sha(seal_path),approval_sha256=c['approval_sha256'],one_shot=True)
    grant['grant_id']=digest(grant)
    save(ns/'contract.json',c);save(ns/'seal.json',seal);save(ns/'grant.json',grant)
    return grant


def authorize(namespace):
    ns=Path(namespace).resolve();c=validate_contract(read(ns/'contract.json'),active=True)
    grant=read(ns/'grant.json');seal=read(ns/'seal.json')
    require(grant['kind']=='C_SCIENCE_ONE_SHOT' and grant['namespace']==str(ns)==c['output_namespace'],'C_GRANT_IDENTITY')
    require(grant['grant_id']==digest({k:v for k,v in grant.items() if k!='grant_id'}),'C_GRANT_DIGEST')
    require(sha(ns/'contract.json')==grant['contract_sha256'] and sha(ns/'seal.json')==grant['seal_sha256']==c['technical_seal_sha256'],'C_GRANT_BINDING')
    verify_seal(seal);validate_routing(c,seal)
    require(not (ns/'STOP.json').exists() and not (ns/'CONSUMED.json').exists(),'C_ALREADY_CONSUMED_OR_STOPPED')
    # Exclusive claim is the final authority operation before any scientific import.
    save(ns/'CONSUMED.json',dict(grant_id=grant['grant_id'],UTC=utc(),nonrefundable=True))
    return c,seal


def utc():return datetime.now(timezone.utc).isoformat()


class CallBudget:
    def __init__(self, directory, limits=None):
        self.directory=Path(directory);self.limits=dict(LIMITS if limits is None else limits)
        self.path=self.directory/'call_ledger.json'
    def take(self, name):
        require(name in self.limits,'FORBIDDEN_SCIENTIFIC_CALL:'+name)
        with (self.directory/'call_ledger.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=read(self.path) if self.path.exists() else dict(limits=self.limits,used={k:0 for k in self.limits},events=[])
            require(ledger['limits']==self.limits,'BUDGET_CHANGED')
            require(ledger['used'][name]<self.limits[name],'CALL_BUDGET_EXHAUSTED:'+name)
            ledger['used'][name]+=1
            ledger['events'].append(dict(call=name,reserved_before_call=True,pid=os.getpid(),UTC=utc()))
            save(self.path,ledger,exclusive=False)


def precision_schedule(evaluate, *, on_tier=None):
    """Only an explicitly finite WIDTH_INSUFFICIENT result permits advancement."""
    for bits in PRECISIONS:
        result=evaluate(bits)
        require(result['status'] in ('TARGET_MET','WIDTH_INSUFFICIENT','MODEL_UNRESOLVED','TARGET_NOT_MET'),'INVALID_NUMERICAL_STATUS')
        if on_tier:on_tier(bits,result)
        if result['status']!='WIDTH_INSUFFICIENT':return result
    return dict(status='UNDETERMINED_PRECISION_EXHAUSTED',last=result)


def _proc_identity(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=int(fields[19]))
    except (FileNotFoundError,ProcessLookupError):return None


def _children(pid):
    # Only direct owned proc children, never an all-host process scan.
    try:return [int(x) for x in Path('/proc',str(pid),'task',str(pid),'children').read_text().split()]
    except FileNotFoundError:return []


def witness(directory, command):
    """Durable per-worker wait owned by a detached process, not the supervisor.

    Linux child-subreaper adoption covers orphaned descendants. Native exits are
    fsync'd when each wait returns. An abnormal worker exit stops only its owned
    descendants; independent runs and PSI/CPU competition are never stop causes.
    """
    import ctypes
    d=Path(directory);d.mkdir(exist_ok=False)
    require(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,'SUBREAPER_UNAVAILABLE')
    with (d/'stdout.log').open('xb') as out,(d/'stderr.log').open('xb') as err:
        child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
        identity=_proc_identity(child.pid);save(d/'worker_identity.json',identity)
        code=None;reaped=[];known={};cancel=[False]
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:cancel.__setitem__(0,True))
        stopped_at=None;descendants_terminated=False
        while True:
            todo=[child.pid,*_children(os.getpid()),*[p for p,i in known.items() if _proc_identity(p)==i]]
            visited=set()
            while todo:
                pid=todo.pop()
                if pid in visited:continue
                visited.add(pid)
                found=_proc_identity(pid)
                if found and (pid not in known or found==known[pid]):
                    known[pid]=found;todo.extend(_children(pid))
            while True:
                try:pid,status,usage=os.wait4(-1,os.WNOHANG)
                except ChildProcessError:pid=0;break
                if not pid:break
                native=os.waitstatus_to_exitcode(status);reaped.append(pid)
                save(d/('native_exit_'+str(pid)+'.json'),dict(pid=pid,exit_code=native,native_wait=True,reaped=True,UTC=utc()))
                if pid==child.pid:
                    code=native;child.returncode=native
                    save(d/'worker_native_exit.json',dict(pid=pid,exit_code=native,reaped=True,UTC=utc(),scope='NEW_C_WORKER_ONLY'))
            live=[p for p,ident in known.items() if p not in reaped and _proc_identity(p)==ident]
            if code is not None and not live:break
            if cancel[0] or (code is not None and live):
                if code is not None and live:descendants_terminated=True
                if stopped_at is None:stopped_at=time.monotonic()
                sig=signal.SIGKILL if time.monotonic()-stopped_at>5 else signal.SIGTERM
                for pid in live:
                    if _proc_identity(pid)==known[pid]:
                        try:os.kill(pid,sig)
                        except ProcessLookupError:pass
            time.sleep(.02)
        save(d/'STOP.json',dict(status='NATIVE_EXIT_REAP_RECORDED' if code==0 and not cancel[0] and not descendants_terminated else 'FAILED_NO_RETRY',
            exit_code=code,all_owned_reaped=True,descendants_terminated=descendants_terminated,reaped_pids=reaped,original_H4_native_verified=False,UTC=utc()))
        return 0 if code==0 and not cancel[0] and not descendants_terminated else 1


def launch(namespace):
    ns=Path(namespace).resolve();c,seal=authorize(ns)
    save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=utc(),grant_id=read(ns/'grant.json')['grant_id']))
    processes=[]
    try:
        for rep in ('pauli_jw','df_native'):
            command=[sys.executable,'-B',str(ROOT/'scripts/run_h4_numerical_evidence.py'),'--worker',str(ns),'--representation',rep]
            witness_command=[sys.executable,'-B',str(ROOT/'scripts/run_h4_numerical_evidence.py'),'--witness',str(ns/(rep+'_native')),'--command-json',json.dumps(command)]
            p=subprocess.Popen(witness_command,start_new_session=True)
            processes.append((rep,p))
        statuses={}
        for rep,p in processes:
            statuses[rep]=p.wait();save(ns/(rep+'_witness_exit.json'),dict(exit_code=statuses[rep],reaped=True,UTC=utc()))
        require(all(v==0 for v in statuses.values()),'C_WORKER_FAILURE')
        verify_seal(seal)
        numerical={rep:read(ns/(rep+'_result.json'))['status'] for rep in statuses}
        final='C_CANDIDATE_EVIDENCE_COMPLETE_PENDING_SCIENTIFIC_ACCEPTANCE' if all(s=='TARGET_MET' for s in numerical.values()) else 'C_UNDETERMINED_MANDATORY_STOP'
        save(ns/'STOP.json',dict(status=final,numerical_status=numerical,
            native_witness_exits=statuses,automatic_retry=False,original_H4_native_verified=False,UTC=utc()))
    except BaseException as exc:
        for rep,p in processes:
            if p.poll() is None:p.terminate()
        for rep,p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        # Detached witnesses retain native evidence even if this supervisor dies.
        if not (ns/'STOP.json').exists():save(ns/'STOP.json',dict(status='FAILED_NO_RETRY',error=repr(exc),UTC=utc(),original_H4_native_verified=False))
        raise
    return dict(status=read(ns/'STOP.json')['status'])


def worker_authorize(namespace, representation, *, bridge=False):
    ns=Path(namespace).resolve();require(representation in ('pauli_jw','df_native'),'WORKER_REPRESENTATION')
    c=validate_contract(read(ns/'contract.json'),active=True);grant=read(ns/'grant.json');seal=read(ns/'seal.json')
    require(grant['grant_id']==digest({k:v for k,v in grant.items() if k!='grant_id'}),'C_GRANT_DIGEST')
    require(grant['namespace']==c['output_namespace']==str(ns),'WORKER_NAMESPACE')
    require(sha(ns/'contract.json')==grant['contract_sha256'] and sha(ns/'seal.json')==grant['seal_sha256']==c['technical_seal_sha256'],'WORKER_BINDING')
    validate_routing(c,seal)
    require(read(ns/'CONSUMED.json')['grant_id']==grant['grant_id'] and (ns/'LAUNCH_CLAIM.json').is_file(),'WORKER_NOT_DISPATCHED')
    require(not (ns/'STOP.json').exists(),'C_ALREADY_STOPPED')
    if bridge:
        require((ns/(representation+'_worker_claim.json')).is_file(),'BRIDGE_WITHOUT_WORKER')
        hashes(seal['code_files']);hashes(seal['legacy_source_files']);hashes(seal['legacy_runtime_files']);hashes(seal['input_files'])
        require(sha(sys.executable)==seal['legacy_python_executable_sha256'],'WRONG_LEGACY_EXECUTABLE')
    else:verify_seal(seal)
    save(ns/(representation+('_bridge_claim.json' if bridge else '_worker_claim.json')),dict(pid=os.getpid(),UTC=utc()))
    return c,seal
