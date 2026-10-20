"""Explicit USER-approved 512 correctness runner; prepare does not claim or sample.

The management source is pinned with the package. Execute consumes the existing
scope once; failures are preserved and never retried. Inspection reads files only.
"""
from pathlib import Path
from hashlib import sha256
import argparse,json,os,sys,time,subprocess,traceback
sys.dont_write_bytecode=True
from . import contract as K,activation,gate,guard as G,offline as O
from ..g6_w2c_b3_live_preflight_v1 import spec as S


def save(path,value):
    path=Path(path)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def resources():
    v=dict(time_ns=time.time_ns(),affinity=sorted(os.sched_getaffinity(0)),loadavg=os.getloadavg())
    for p in ('/proc/meminfo','/proc/pressure/cpu','/proc/pressure/memory','/proc/pressure/io','/proc/self/status','/proc/self/io'):
        try:v[p]=Path(p).read_text()
        except OSError as exc:v[p]=dict(error=type(exc).__name__)
    return v


def prepare(root,packet):
    c=K.candidate(root)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    check=K.verify_source_commit(c,source)
    approval=O.read(packet/'user_approval.json');text=approval['approval_text']
    G.require(sha256(text.encode()).hexdigest()==approval['approval_text_sha256'],'ORIGINAL_USER_INSTRUCTION')
    G.require(sha256(b'LOCAL_USER_ATTACHMENT_BINDING/B3_CORRECTNESS_512\0'+text.encode()).hexdigest()==activation.APPROVED_ACTIVATION_ID,'LOCAL_ACTIVATION_CONTENT_ID')
    m=K.run_manifest_candidate(c,source_commit=source)
    state=Path(m['state_directory']);state.mkdir(parents=True,exist_ok=False)
    m.update(SCIENCE_RUN_AUTHORIZED=True,artificial_only=False,run_ready=True,content_adopted=True,
        contract_grants_execution=True,science_release=source,scientific_adoption='USER_ADOPTED_512_CORRECTNESS',
        approval_status='EXPLICIT_USER_EXECUTION_APPROVED',activation_record=dict(id=activation.APPROVED_ACTIVATION_ID,
        purpose='USER_APPROVED_ONE_SHOT_512_CORRECTNESS'))
    m['approval_record']=dict(approver='USER',explicit_execution_approval=True,approval_text=text,
        approval_text_sha256=approval['approval_text_sha256'],measured_SOURCE=source,
        manifest_binding=S.digest({k:v for k,v in m.items() if k!='approval_record'}))
    permit=activation.admit(c,m,state)
    K.preflight_runtime(c,m,permit)
    # Gate rejection exercises no science resolver, store, or bit provider.
    from copy import deepcopy
    denied=[]
    for key,value in [('SCIENCE_RUN_AUTHORIZED',False),('planned',513),('retry',1),('resume',True),
        ('seed_namespace','ARTIFICIAL_UNAPPROVED'),('activation_record',None),('approval_record',None),
        ('measured_SOURCE','0'*40),('global_claim','0'*64),('storage_destination',str(state/'wrong'))]:
        wrong=deepcopy(m);wrong[key]=value
        try:activation.admit(c,wrong,state)
        except Exception as exc:denied.append(dict(field=key,rejected=type(exc).__name__,reason=str(exc)))
        else:raise RuntimeError('UNAUTHORIZED_MUTATION_ADMITTED:'+key)
    G.require(not list(state.iterdir()),'PREFLIGHT_SCOPE_ALREADY_CONSUMED')
    G.require(not G.observation()['core_loaded'] and not G.ATTEMPTS,'PRESTART_CORE_IMPORT')
    fs=os.statvfs(state)
    save(packet/'candidate.json',c);save(packet/'approved_manifest.json',m)
    save(packet/'source_inventory.json',dict(measured_SOURCE=source,source_release_candidate=c['source_release_candidate'],
        source_pins=c['source_pins'],input_pins=c['input_pins'],runtime=c['python'],worker_runtime=c['worker_runtime'],
        resolver_binding=c['final_runtime_binding'],source_git_check=check))
    save(packet/'preflight.json',dict(status='PASS',measured_SOURCE=source,candidate_digest=S.digest(c),
        manifest_digest=S.digest(m),contract_digest=S.digest([m['input'],m['precision'],m['A6'],m['tasks'],m['call_budgets'],m['byte_limits'],m['simple_control_contract']]),
        approval_digest=S.digest(m['approval_record']),approval_text_sha256=approval['approval_text_sha256'],
        activation_id=activation.APPROVED_ACTIVATION_ID,source_git_check=check,denied_mutations=denied,
        new_scope_empty=True,claim_consumed=False,scientific_imports=0,scientific_sampling=0,
        guard=G.observation(),filesystem=dict(device=state.stat().st_dev,available_bytes=fs.f_bavail*fs.f_frsize),resources=resources()))
    return dict(status='PREFLIGHT_PASS',source=source,manifest_digest=S.digest(m),run_id=m['run_id'],state=str(state))


def execute(root,packet):
    c=O.read(packet/'candidate.json');m=O.read(packet/'approved_manifest.json');pre=O.read(packet/'preflight.json')
    G.require(S.digest(c)==pre['candidate_digest'] and S.digest(m)==pre['manifest_digest'],'FROZEN_PREFLIGHT_BINDING')
    G.require(Path(c['runtime_root'])==root,'FROZEN_ROOT')
    permit=activation.admit(c,m,Path(m['state_directory']));K.preflight_runtime(c,m,permit)
    G.require(not list(Path(m['state_directory']).iterdir()),'ONE_SHOT_SCOPE_NOT_FRESH')
    save(packet/'execution_invocation.json',dict(one_shot=True,retry=0,resume=False,run_id=m['run_id'],
        manifest_digest=S.digest(m),source=m['measured_SOURCE'],begin=resources()))
    try:
        from .integration import run_live
        result=run_live(c,m,Path(m['state_directory']),permit)
        save(packet/'execution_return.json',result)
        return dict(status=result['classification'],complete=result['complete'])
    except BaseException as exc:
        save(packet/'execution_exception.json',dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),mandatory_STOP=True,retry=0,resume=False))
        raise
    finally:
        save(packet/'resources_after.json',resources())


def inspect(packet):
    c=O.read(packet/'candidate.json');m=O.read(packet/'approved_manifest.json')
    result=O.inspect(m['state_directory'],c,m,artificial=False)
    save(packet/'saved_evidence_check.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute','inspect']);p.add_argument('packet',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parents[4];packet=args.packet.resolve()
    value=inspect(packet) if args.action=='inspect' else (prepare(root,packet) if args.action=='prepare' else execute(root,packet))
    print(json.dumps(value,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=='__main__':main()
