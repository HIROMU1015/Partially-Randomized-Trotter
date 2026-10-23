"""D1 one-shot control. Stdlib only before authorization; old science untouched."""
import ast
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import h4_certificate_control as C

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_d1_execution/2026-10-24'
PREP=ROOT/'artifacts/non_df_h4_d1_preparation/2026-10-24'
LIMITS=dict(full_input_build=4,ablation_input_build=5,cost_transpile=9)
FULL=('pauli_jw_PF_full','pauli_jw_PR_full','df_native_PF_full','df_native_PR_full')
PYTHON='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
SCRIPT=ROOT/'scripts/run_h4_d1.py'

def relative_hashes(files):
    for path,expected in files.items():
        C.require(C.sha(ROOT/path)==expected,'SOURCE_OR_INPUT_CHANGED:'+path)

def scope(c):
    C.require(c.get('kind')=='H4_D1_ACTIVE_ONE_SHOT_V1','D1_CONTRACT_KIND')
    C.require(all(c.get(k) is True for k in ('execution_authorized','dispatch_enabled','grant_valid')),'D1_NOT_AUTHORIZED')
    C.require(c['maximum_cost_transpile_calls']==9 and c['call_limits']==LIMITS,'D1_BUDGET')
    C.require(c['maximum_scope']==list(FULL)+[
        'pauli_jw_PR_retained','pauli_jw_PR_event','df_native_PR_retained','df_native_PR_event','empty'],'D1_VARIANTS')
    C.require(c['reproduction_gate']['fingerprint_mismatch']=='STOP_NO_FALLBACK','D1_NO_HASH_BYPASS')
    C.require(c['mandatory_STOP'] and not c['automatic_retry'],'D1_STOP')
    C.require(c['fixed']==C.read(PREP/'contract_DRAFT.json')['fixed'],'D1_FIXED_CONDITIONS')
    C.require(c['D2_D3_authorized'] is False and c['N_G_authorized'] is False,'D1_SCOPE_EXPANSION')

def verify_seal(seal, *, runtime=True):
    C.require(seal['kind']=='D1_SAVED_DESCRIPTOR_SOURCE_RUNTIME_INPUT_SEAL','D1_SEAL_KIND')
    C.require(C.sha(sys.executable)==seal['python_sha256'],'D1_PYTHON')
    relative_hashes(seal['code_files'])
    for path,expected in seal['code_files'].items():
        import hashlib
        b=subprocess.check_output(['git','show',seal['SOURCE']+':'+path],cwd=ROOT)
        C.require(hashlib.sha256(b).hexdigest()==expected,'UNCOMMITTED_D1_CODE:'+path)
    relative_hashes(seal['input_files'])
    C.require(C.sha(BASE/'approval.txt')==seal['approval_sha256'],'D1_APPROVAL')
    C.require(C.sha(BASE/'runtime.json')==seal['runtime_manifest_sha256'],'D1_RUNTIME_MANIFEST')
    if runtime:C.hashes(C.read(BASE/'runtime.json')['files'])
    previous=C.read(ROOT/seal['preservation_map'])
    relative_hashes(previous)

def prepare(approval):
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(),'D1_NAMESPACE_USED_NO_RETRY')
    proof=C.read(PREP/'preparation.json')
    C.require(proof['artificial_tests']['failures']==0 and proof['artificial_tests']['errors']==0,'PREPARATION_TESTS')
    artificial=C.read(BASE/'artificial/result.json')
    C.require(artificial['failures']==0 and artificial['errors']==0 and artificial['tests']>=8,'ADAPTER_TESTS')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=set(artificial['project_modules'])|{
        'src/trotterlib/h4_d1_control.py','src/trotterlib/h4_certificate_control.py',
        'scripts/run_h4_d1.py','tests/test_h4_d1_adapter.py','tests/test_h4_d1_authority.py',
        'artifacts/non_df_h4_d1_preparation/2026-10-24/check_preparation.py'}
    code={}
    for p in sorted(paths):
        C.require((ROOT/p).read_bytes()==subprocess.check_output(['git','show',source+':'+p],cwd=ROOT),'D1_CODE_NOT_COMMITTED:'+p)
        code[p]=C.sha(ROOT/p)
        if p.startswith('src/trotterlib/') and not p.startswith('src/trotterlib/h4_d1_'):
            reference='3282f74cd4d219141e4358c215a225dc2e3ec9c3' if p=='src/trotterlib/h4_certificate_control.py' else '0c329b30a81accb35c212be8defa66fee0dd0f12'
            old=subprocess.check_output(['git','show',reference+':'+p],cwd=ROOT)
            C.require(old==(ROOT/p).read_bytes(),'ORIGINAL_SOURCE_CHANGED:'+p)
    text=Path(approval).read_text()
    C.require('限定診断D1の実行を承認' in text and '最大9回' in text and 'mandatory STOP' in text,'D1_EXPLICIT_APPROVAL')
    (BASE/'approval.txt').write_bytes(Path(approval).read_bytes())
    legacy=C.read(ROOT/proof['reference_runtime_seal'])
    C.require(C.sha(sys.executable)==legacy['legacy_python_executable_sha256'],'REFERENCE_PYTHON_CHANGED')
    files=dict(legacy['legacy_runtime_files']);C.hashes(files)
    rt=dict(python=sys.executable,version=sys.version,files=files,
            origin=proof['reference_runtime_seal']+'#/legacy_runtime_files',
            origin_manifest_sha256=proof['reference_runtime_seal_sha256'],package_installs=0)
    C.save(BASE/'runtime.json',rt)
    inputs=dict(proof['input_source_sha256'])
    for path in ('artifacts/non_df_h4_d1_preparation/2026-10-24/preparation.json',
                 'artifacts/non_df_h4_d1_preparation/2026-10-24/contract_DRAFT.json',
                 'artifacts/non_df_h4_d1_execution/2026-10-24/artificial/result.json',
                 'artifacts/non_df_h4_d1_execution/2026-10-24/artificial/authority.json'):
        inputs[path]=C.sha(ROOT/path)
    relative_hashes(inputs)
    seal=dict(kind='D1_SAVED_DESCRIPTOR_SOURCE_RUNTIME_INPUT_SEAL',SOURCE=source,
        historical_SOURCE='0c329b30a81accb35c212be8defa66fee0dd0f12',code_files=code,input_files=inputs,
        python_sha256=C.sha(sys.executable),runtime_manifest_sha256=C.sha(BASE/'runtime.json'),
        approval_sha256=C.sha(BASE/'approval.txt'),preservation_map=proof['previous_preservation_map'],
        preservation_map_sha256=proof['previous_preservation_map_sha256'],
        UTC=C.utc(),real_H4_calls_before_seal=0)
    C.save(BASE/'seal.json',seal)
    plan=C.read(PREP/'contract_DRAFT.json')
    c=dict(plan,kind='H4_D1_ACTIVE_ONE_SHOT_V1',status='APPROVED_SEALED_ONE_SHOT',
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,
        science_SOURCE=source,production_adapter='src/trotterlib/h4_d1_adapter.py',
        call_limits=LIMITS,maximum_scope=[v['id'] for v in plan['variants']],
        selected=proof['selected'],approval_file=str(BASE/'approval.txt'),
        approval_sha256=seal['approval_sha256'],seal_sha256=C.sha(BASE/'seal.json'),
        output_namespace=str(BASE/'run01'),mandatory_STOP=True,automatic_retry=False,
        D2_D3_authorized=False,N_G_authorized=False,
        launch_command=[PYTHON,'-B',str(SCRIPT),'--launch'])
    scope(c);C.save(BASE/'active_contract.json',c);verify_seal(seal)
    ns=BASE/'run01';ns.mkdir()
    grant=dict(kind='H4_D1_SCIENCE_ONE_SHOT',SOURCE=source,namespace=str(ns),
        contract_sha256=C.sha(BASE/'active_contract.json'),seal_sha256=C.sha(BASE/'seal.json'),
        approval_sha256=seal['approval_sha256'],one_shot=True,automatic_retry=False,call_limits=LIMITS)
    grant['grant_id']=C.digest(grant);C.save(ns/'grant.json',grant)
    C.save(ns/'call_ledger.json',dict(limits=LIMITS,used=dict.fromkeys(LIMITS,0),events=[]))
    return grant

def authority(ns=None, *, consume=False, task=False):
    ns=BASE/'run01' if ns is None else Path(ns).resolve()
    C.require(ns==BASE/'run01','D1_NAMESPACE')
    c=C.read(BASE/'active_contract.json');seal=C.read(BASE/'seal.json');g=C.read(ns/'grant.json')
    scope(c)
    C.require(g['kind']=='H4_D1_SCIENCE_ONE_SHOT' and g['namespace']==str(ns),'D1_GRANT_KIND')
    C.require(g['one_shot'] is True and g['automatic_retry'] is False,'D1_GRANT_ONESHOT')
    C.require(g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}),'D1_GRANT_DIGEST')
    C.require(g['SOURCE']==c['science_SOURCE']==seal['SOURCE'],'D1_GRANT_SOURCE')
    C.require(g['contract_sha256']==C.sha(BASE/'active_contract.json') and
              g['seal_sha256']==C.sha(BASE/'seal.json')==c['seal_sha256'],'D1_GRANT_BINDING')
    C.require(g['call_limits']==LIMITS,'D1_GRANT_BUDGET')
    C.require(C.sha(BASE/'approval.txt')==g['approval_sha256']==c['approval_sha256']==seal['approval_sha256'],'D1_GRANT_APPROVAL')
    C.require(not (ns/'STOP.json').exists() and not (ns/'ENGINE_STOP.json').exists(),'D1_STOP_ALREADY_PRESENT')
    if task:
        C.require(C.read(ns/'CONSUMED.json')['grant_id']==g['grant_id'] and (ns/'LAUNCH_CLAIM.json').exists(),'D1_TASK_NOT_LAUNCHED')
        C.require(C.sha(sys.executable)==seal['python_sha256'],'D1_TASK_PYTHON')
        relative_hashes(seal['code_files']);relative_hashes(seal['input_files'])
        C.require(C.sha(BASE/'runtime.json')==seal['runtime_manifest_sha256'],'D1_TASK_RUNTIME')
    else:verify_seal(seal)
    if consume:
        C.require(not (ns/'CONSUMED.json').exists(),'D1_GRANT_CONSUMED_NO_RETRY')
        C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id'],UTC=C.utc(),nonrefundable=True,pid=os.getpid()))
    return c,seal,g

def record_resources(ns):
    snapshot=dict(UTC=C.utc(),own_pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)))
    for name,path in [('memory','/proc/meminfo'),('cpu_pressure','/proc/pressure/cpu'),('memory_pressure','/proc/pressure/memory')]:
        try:snapshot[name]=Path(path).read_text()
        except OSError:snapshot[name]=None
    with (ns/'resource_observations.jsonl').open('a') as f:
        f.write(json.dumps(snapshot)+'\n');f.flush();os.fsync(f.fileno())

def task_run(ns, kind, variant):
    ns=Path(ns);c,seal,g=authority(ns,task=True)
    matches=[v for v in c['variants'] if v['id']==variant]
    C.require(len(matches)==1,'D1_VARIANT_NOT_ALLOWED');v=matches[0]
    d=ns/'variants'/variant
    if kind=='build':
        if v['historical_baseline']:
            C.require(not (ns/'stage_A.json').exists(),'D1_DUPLICATE_FULL_BUILD')
        else:
            C.require(C.read(ns/'stage_B.json')['status']=='PASS_FULL4_RZ_REPRODUCED','D1_ABLATION_BEFORE_BASELINE')
        d.mkdir();C.save(d/'build_claim.json',dict(pid=os.getpid(),identity=C._proc_identity(os.getpid()),UTC=C.utc()))
    elif kind=='compile':
        gate='stage_A.json' if v['historical_baseline'] else 'stage_B.json'
        expected='PASS_FULL4_INPUT_FINGERPRINTS' if v['historical_baseline'] else 'PASS_FULL4_RZ_REPRODUCED'
        C.require(C.read(ns/gate)['status']==expected,'D1_COMPILE_STAGE_GATE')
        C.save(d/'compile_claim.json',dict(pid=os.getpid(),UTC=C.utc()))
    else:raise ValueError('D1_TASK_KIND')
    try:
        from . import h4_d1_adapter as A
        a=A.api();C.require(a.qiskit.__version__=='1.3.0','D1_QISKIT_VERSION')
        if kind=='build':
            if variant=='empty':built=A.empty(8);old=None
            else:
                old=next(x for x in c['selected'] if x['case']==v['case'])
                join=C.read(ROOT/old['price_join']);a.verify_join(join,expected_case=v['case'])
                rep=join['representation'];method=join['method']
                raw=C.read(ROOT/('artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/'+rep+'_model.json'))
                # Never use the saved full matrix/state or regenerate a Hamiltonian.
                model=dict(representation=rep,n_qubits=raw['n_qubits'],step_time=raw['step_time'],
                    hamiltonian=dict(constant=raw['hamiltonian']['constant']),
                    cases={method:{k:raw['cases'][method][k] for k in
                        ('config','blocks','components','native_scalar_phases')}})
                built=A.build(model,method,join,v['part'])
            expected=old['input_fingerprint'] if v['historical_baseline'] else None
            data=A.save_input(built,d,expected)
            data.update(variant=variant,case=v['case'],part=v['part'],SOURCE=seal['SOURCE'],
                input_descriptor_hash=None if old is None else old['trajectory_record_hash'],
                scientific_gate='NOT_TESTED' if expected is None else
                    ('PASS' if data['input_fingerprint_matches'] else 'STOP_INPUT_FINGERPRINT_MISMATCH'))
            C.save(d/'input.json',data)
        else:
            inp=C.read(d/'input.json')
            with (d/'input.qpy').open('rb') as f:q=a.qpy.load(f)
            C.require(len(q)==1 and a.canonical_qiskit_circuit_fingerprint(q[0])==inp['input_fingerprint'],'D1_QPY_INPUT_HASH_CHANGED')
            # Release temporary object; compile_saved reuses unmodified legacy function once.
            del q
            C.save(d/'cost_transpile_started.json',dict(UTC=C.utc(),reserved_before_call=True,variant=variant))
            data=A.compile_saved(d)
            C.require(data['input_fingerprint']==inp['input_fingerprint'],'D1_COMPILED_INPUT_HASH_CHANGED')
            if v['historical_baseline']:
                old=next(x for x in c['selected'] if x['case']==v['case'])
                data['old_RZ_matches']=data['RZ']==old['saved_cosine_RZ']
            data.update(variant=variant,SOURCE=seal['SOURCE'])
            C.save(d/'metrics.json',data)
        C.save(d/(kind+'_finished.json'),dict(UTC=C.utc(),pid=os.getpid(),scope='NEW_D1_TASK_ONLY'))
        return 0
    except BaseException:
        C.save(d/(kind+'_error.json'),dict(traceback=traceback.format_exc(),UTC=C.utc(),no_retry=True))
        raise

def batch(ns, kind, variants):
    budget=C.CallBudget(ns,LIMITS);processes=[]
    try:
        for variant in variants:
            budget.take('cost_transpile' if kind=='compile' else
                        'full_input_build' if variant in FULL else 'ablation_input_build')
            taskdir=ns/'native_tasks'/(kind+'_'+variant);taskdir.mkdir(parents=True)
            command=[PYTHON,'-B',str(SCRIPT),'--task',str(ns),'--kind',kind,'--variant',variant]
            out=(taskdir/'stdout.log').open('xb');err=(taskdir/'stderr.log').open('xb')
            p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
            out.close();err.close()
            C.save(taskdir/'launch.json',dict(pid=p.pid,identity=C._proc_identity(p.pid),argv=command,UTC=C.utc()))
            processes.append((variant,p,taskdir))
        last=0
        while any(p.poll() is None for _,p,_ in processes):
            if time.monotonic()-last>30:record_resources(ns);last=time.monotonic()
            time.sleep(.2)
        codes={}
        for variant,p,d in processes:
            codes[variant]=p.wait()
            C.save(d/'native_exit.json',dict(pid=p.pid,exit_code=codes[variant],native_wait=True,reaped=True,UTC=C.utc()))
        C.require(all(x==0 for x in codes.values()),'D1_TASK_FAILURE:'+str(codes))
        return codes
    finally:
        for variant,p,d in processes:
            if p.poll() is None:p.terminate()
        for variant,p,d in processes:
            if not (d/'native_exit.json').exists():
                code=p.wait()
                C.save(d/'native_exit.json',dict(pid=p.pid,exit_code=code,native_wait=True,reaped=True,UTC=C.utc()))

def worker():
    ns=BASE/'run01';c,seal,g=authority(ns,task=True)
    C.save(ns/'ENGINE_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    (ns/'variants').mkdir();record_resources(ns)
    try:
        batch(ns,'build',FULL)
        rows={v:C.read(ns/'variants'/v/'input.json') for v in FULL}
        ok=all(x['input_fingerprint_matches'] is True for x in rows.values())
        C.save(ns/'stage_A.json',dict(status='PASS_FULL4_INPUT_FINGERPRINTS' if ok else 'STOP_INPUT_FINGERPRINT_MISMATCH',
                                     full_inputs=rows,top_level_cost_compile_calls=0))
        if not ok:
            C.save(ns/'ENGINE_STOP.json',dict(status='STAGE_A_FINGERPRINT_STOP_NO_RETRY',stage='A',
                cost_compiles=0,stage_B='NOT_EXECUTED',stage_C='NOT_EXECUTED',mandatory_STOP=True))
            return 0
        metrics={}
        # Stop on the first historical price mismatch; no further full compile is launched.
        for variant in FULL:
            batch(ns,'compile',[variant])
            data=C.read(ns/'variants'/variant/'metrics.json');metrics[variant]=data
            if not data['old_RZ_matches']:
                C.save(ns/'stage_B.json',dict(status='STOP_FULL_RZ_MISMATCH',full_metrics=metrics))
                C.save(ns/'ENGINE_STOP.json',dict(status='STAGE_B_RZ_STOP_NO_RETRY',stage='B',
                    stage_C='NOT_EXECUTED',mandatory_STOP=True))
                return 0
        C.save(ns/'stage_B.json',dict(status='PASS_FULL4_RZ_REPRODUCED',full_metrics=metrics))
        parts=[v['id'] for v in c['variants'] if not v['historical_baseline']]
        batch(ns,'build',parts);batch(ns,'compile',parts)
        for variant in parts:metrics[variant]=C.read(ns/'variants'/variant/'metrics.json')
        from .h4_d1_adapter import interactions
        C.save(ns/'comparisons.json',dict(metrics=metrics,interactions=interactions(metrics),
            scope='FIXED_REPRESENTATIVES_ONLY_NO_UNIQUE_CAUSAL_SHARES'))
        C.save(ns/'stage_C.json',dict(status='COMPLETE_9_COST_COMPILES',variants=list(metrics)))
        C.save(ns/'ENGINE_STOP.json',dict(status='D1_COMPLETE_MANDATORY_STOP',stage='C',cost_compiles=9,mandatory_STOP=True))
        return 0
    except BaseException:
        if not (ns/'ENGINE_STOP.json').exists():
            C.save(ns/'ENGINE_STOP.json',dict(status='FAILED_NO_RETRY',traceback=traceback.format_exc(),mandatory_STOP=True))
        raise

def launch():
    ns=BASE/'run01';c,seal,g=authority(ns,consume=True)
    C.save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc(),grant_id=g['grant_id']))
    tree=ast.parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='witness')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='NEW_C_WORKER_ONLY':n.value='NEW_D1_ENGINE_ONLY'
    env=dict(vars(C));exec(compile(ast.Module(body=[node],type_ignores=[]),'<reused D1 native witness>','exec'),env)
    status=env['witness'](ns/'native',[PYTHON,'-B',str(SCRIPT),'--worker'])
    verification_error=None
    try:verify_seal(seal)
    except BaseException as exc:verification_error=repr(exc)
    engine=C.read(ns/'ENGINE_STOP.json') if (ns/'ENGINE_STOP.json').exists() else dict(status='MISSING_ENGINE_STOP')
    native=C.read(ns/'native/worker_native_exit.json')
    native_stop=C.read(ns/'native/STOP.json')
    tasks=list((ns/'native_tasks').glob('*/native_exit.json')) if (ns/'native_tasks').exists() else []
    C.save(ns/'STOP.json',dict(status=engine['status'] if status==0 and not verification_error else 'FAILED_NATIVE_OR_PRESERVATION_NO_RETRY',
        stage=engine.get('stage'),native_engine_exit=native,all_owned_reaped=native_stop['all_owned_reaped'],
        direct_task_native_receipts=len(tasks),direct_task_exits=[C.read(p) for p in tasks],
        preservation_error=verification_error,old_preservation_PASS=verification_error is None,
        ledger=C.read(ns/'call_ledger.json'),mandatory_STOP=True,automatic_retry=False,D2_D3_started=False,
        original_H4_native_verified=False,formal_certificate_accepted=False,UTC=C.utc()))
    inventory={str(p.relative_to(ns)):C.sha(p) for p in ns.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.name!='output_inventory.json'}
    C.save(ns/'output_inventory.json',dict(files=inventory))
    return dict(status=C.read(ns/'STOP.json')['status'],native_engine_exit=native['exit_code'],
                stage=engine.get('stage'),ledger=C.read(ns/'call_ledger.json')['used'])
