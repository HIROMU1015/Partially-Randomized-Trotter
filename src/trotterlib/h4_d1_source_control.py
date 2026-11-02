"""New one-shot D1 connector: saved full QPY + adopted SOURCE criterion.

No quantum imports before authority. Original adapter/compiler and prior run
records are unmodified. SOURCE correspondence and historical hashes are separate.
"""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from . import h4_d1_control as D
from . import h4_d1_reproducibility as R

C=D.C
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_d1_source_execution/2026-10-24'
DESIGN=ROOT/'artifacts/non_df_h4_d1_reproducibility_design/2026-10-24'
PREVIOUS=ROOT/'artifacts/non_df_h4_d1_execution/2026-10-24'
SCRIPT=ROOT/'scripts/run_h4_d1_source.py'
PYTHON=D.PYTHON
FULL=D.FULL
LIMITS=dict(full_QPY_validation=4,full_input_build=0,ablation_input_build=5,cost_transpile=9)
KIND='H4_D1_SOURCE_ACTIVE_ONE_SHOT_V1'


def scope(c):
    plan=C.read(DESIGN/'contract_DRAFT.json')
    C.require(c.get('kind')==KIND,'SOURCE_D1_CONTRACT_KIND')
    C.require(all(c.get(k) is True for k in ('execution_authorized','criterion_adopted','dispatch_enabled','grant_valid')),'SOURCE_D1_NOT_AUTHORIZED')
    for key in ('fixed','variants','full_inputs','ablation','comparisons','metrics','old_full_RZ'):
        C.require(c[key]==plan[key],'SOURCE_D1_FIXED:'+key)
    C.require(c['call_limits']==LIMITS and c['maximum_cost_transpile_calls']==9,'SOURCE_D1_BUDGET')
    C.require(c['maximum_full_input_build_calls']==0 and c['maximum_ablation_input_build_calls']==5,'SOURCE_D1_NO_FULL_BUILD')
    C.require(c['mandatory_STOP'] is True and c['automatic_retry'] is False,'SOURCE_D1_STOP')
    C.require(c['D2_D3_authorized'] is False and c['N_G_authorized'] is False,'SOURCE_D1_SCOPE_EXPANSION')


def verify_seal(seal, *, runtime=True, committed=True):
    C.require(seal['kind']=='D1_SOURCE_REUSE_QPY_SEAL_V1','SOURCE_D1_SEAL_KIND')
    C.require(C.sha(sys.executable)==seal['python_sha256'],'SOURCE_D1_PYTHON')
    for group in ('code_files','input_files'):
        D.relative_hashes(seal[group])
    if committed:
        for path,value in seal['code_files'].items():
            data=subprocess.check_output(['git','show',seal['SOURCE']+':'+path],cwd=ROOT)
            C.require(hashlib.sha256(data).hexdigest()==value,'SOURCE_D1_UNCOMMITTED_CODE:'+path)
    C.require(C.sha(BASE/'approval.json')==seal['approval_sha256'],'SOURCE_D1_APPROVAL')
    manifest=ROOT/seal['runtime_manifest']
    C.require(C.sha(manifest)==seal['runtime_manifest_sha256'],'SOURCE_D1_RUNTIME_MANIFEST')
    if runtime:C.hashes(C.read(manifest)['files'])
    C.require(C.sha(BASE/'preservation_before.json')==seal['preservation_map_sha256'],'SOURCE_D1_PRESERVATION_MAP')
    D.relative_hashes(C.read(BASE/'preservation_before.json'))


def prepare():
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(),'SOURCE_D1_NAMESPACE_USED_NO_RETRY')
    approval=C.read(BASE/'approval.json')
    C.require(approval['verbatim_user']=='次の作業に進んで大丈夫' and approval['criterion_B_adopted'] is True
              and approval['finite_D1_execution_authorized'] is True and approval['maximum_cost_compile_calls']==9,'SOURCE_D1_APPROVAL_SCOPE')
    tests=C.read(BASE/'artificial/result.json')
    C.require(tests['failures']==0 and tests['errors']==0 and tests['tests']>=10,'SOURCE_D1_TESTS')
    reused=C.read(BASE/'artificial/reused_17.json')
    C.require(reused['failures']==0 and reused['errors']==0 and reused['tests']==17,'SOURCE_D1_REUSED_TESTS')
    previous=C.read(PREVIOUS/'seal.json')
    C.require(C.sha(sys.executable)==previous['python_sha256'],'SOURCE_D1_REFERENCE_PYTHON_CHANGED')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code=dict(previous['code_files'])
    extra=['src/trotterlib/h4_d1_source_control.py','src/trotterlib/h4_d1_reproducibility.py',
           'scripts/run_h4_d1_source.py','tests/test_h4_d1_source_control.py',
           'tests/test_h4_d1_reproducibility.py','tests/test_native_df_gate_model.py',
           'artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/inspect_saved_evidence.py']
    code.update({p:C.sha(ROOT/p) for p in extra})
    for path,value in code.items():
        C.require(hashlib.sha256(subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)).hexdigest()==value,'SOURCE_D1_NOT_COMMITTED:'+path)
    inputs=dict(previous['input_files'])
    plan=C.read(DESIGN/'contract_DRAFT.json')
    for row in plan['full_inputs'].values():
        p=row['path'];C.require(C.sha(ROOT/p)==row['sha256'],'SOURCE_D1_SAVED_QPY_CHANGED')
        inputs[p]=row['sha256']
        for name in ('input.json','body_before_control.qpy'):
            sibling=(ROOT/p).with_name(name)
            if sibling.exists():inputs[str(sibling.relative_to(ROOT))]=C.sha(sibling)
    for folder,names in [(DESIGN,['contract_DRAFT.json','correspondence.json','inspection.json','saved_QPY_structure.json','preservation_before.json','completion.json']),
                         (PREVIOUS,['seal.json','runtime.json','run01/grant.json','run01/CONSUMED.json','run01/call_ledger.json','run01/STOP.json']),
                         (BASE,['approval.json','artificial/result.json','artificial/reused_17.json'])]:
        for name in names:inputs[str((folder/name).relative_to(ROOT))]=C.sha(folder/name)
    protected=dict(C.read(DESIGN/'preservation_before.json'))
    protected.update({str(p.relative_to(ROOT)):C.sha(p) for p in DESIGN.iterdir() if p.is_file()})
    C.save(BASE/'preservation_before.json',protected)
    seal=dict(kind='D1_SOURCE_REUSE_QPY_SEAL_V1',SOURCE=source,
        historical_SOURCE=previous['historical_SOURCE'],previous_D1_SOURCE=previous['SOURCE'],
        code_files=code,input_files=inputs,python_sha256=C.sha(sys.executable),
        runtime_manifest=str((PREVIOUS/'runtime.json').relative_to(ROOT)),runtime_manifest_sha256=C.sha(PREVIOUS/'runtime.json'),
        approval_sha256=C.sha(BASE/'approval.json'),preservation_map_sha256=C.sha(BASE/'preservation_before.json'),
        UTC=C.utc(),real_H4_calls_before_seal=0)
    verify_seal(seal)
    C.save(BASE/'seal.json',seal)
    c=dict(plan,kind=KIND,status='APPROVED_SOURCE_CRITERION_SEALED_ONE_SHOT',
        criterion_adopted=True,execution_authorized=True,dispatch_enabled=True,grant_valid=True,
        science_SOURCE=source,production_controller_patch='src/trotterlib/h4_d1_source_control.py',
        call_limits=LIMITS,selected=C.read(D.PREP/'preparation.json')['selected'],
        seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=seal['approval_sha256'],
        automatic_retry=False,D2_D3_authorized=False,N_G_authorized=False,
        output_namespace=str(BASE/'run01'),launch_command=[PYTHON,'-B',str(SCRIPT),'--launch'])
    # The disabled design's prose remains a reference, not the active authority.
    c['prohibited_now']=['new full H4 generation','sampling/RNG replay','SCF/FCI/DF','H4 unitary/state/signal','N/G','D2/D3','certificate acceptance']
    c['reproduction_gate']=dict(c['reproduction_gate'],B='ADOPTED_SOURCE_SEMANTIC_CORRESPONDENCE; twelve documented links and actual saved tokens; no historical identity claim')
    c['required_next_approval']=[]
    c['new_grants_issued']=1
    c['grant_location']=str(BASE/'run01/grant.json')
    scope(c);C.save(BASE/'active_contract.json',c)
    ns=BASE/'run01';ns.mkdir()
    g=dict(kind='H4_D1_SOURCE_SCIENCE_ONE_SHOT',SOURCE=source,namespace=str(ns),
           contract_sha256=C.sha(BASE/'active_contract.json'),seal_sha256=C.sha(BASE/'seal.json'),
           approval_sha256=seal['approval_sha256'],one_shot=True,automatic_retry=False,call_limits=LIMITS)
    g['grant_id']=C.digest(g);C.save(ns/'grant.json',g)
    C.save(ns/'call_ledger.json',dict(limits=LIMITS,used=dict.fromkeys(LIMITS,0),events=[]))
    return g


def authority(*,consume=False,task=False):
    ns=BASE/'run01';c=C.read(BASE/'active_contract.json');s=C.read(BASE/'seal.json');g=C.read(ns/'grant.json')
    scope(c)
    C.require(g['kind']=='H4_D1_SOURCE_SCIENCE_ONE_SHOT' and g['namespace']==str(ns),'SOURCE_D1_GRANT_KIND')
    C.require(g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}),'SOURCE_D1_GRANT_DIGEST')
    C.require(g['SOURCE']==c['science_SOURCE']==s['SOURCE'],'SOURCE_D1_GRANT_SOURCE')
    C.require(g['contract_sha256']==C.sha(BASE/'active_contract.json') and g['seal_sha256']==C.sha(BASE/'seal.json')==c['seal_sha256'],'SOURCE_D1_GRANT_BINDING')
    C.require(g['approval_sha256']==c['approval_sha256']==s['approval_sha256'] and g['call_limits']==LIMITS,'SOURCE_D1_GRANT_SCOPE')
    C.require(g['one_shot'] is True and g['automatic_retry'] is False,'SOURCE_D1_ONE_SHOT')
    C.require(not (ns/'STOP.json').exists() and not (ns/'ENGINE_STOP.json').exists(),'SOURCE_D1_STOP_ALREADY_PRESENT')
    verify_seal(s,runtime=not task,committed=not task)
    if task:
        C.require(C.read(ns/'CONSUMED.json')['grant_id']==g['grant_id'] and (ns/'LAUNCH_CLAIM.json').is_file(),'SOURCE_D1_TASK_NOT_LAUNCHED')
    if consume:
        C.require(not (ns/'CONSUMED.json').exists(),'SOURCE_D1_CONSUMED_NO_RETRY')
        C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id'],pid=os.getpid(),UTC=C.utc(),nonrefundable=True))
    return c,s,g


def inspector():
    path=DESIGN/'inspect_saved_evidence.py'
    spec=importlib.util.spec_from_file_location('source_d1_saved_inspector',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def require_compiler(api,expected):
    # Dataclass tuples and their saved JSON lists encode the same fixed profile.
    C.require(C.digest(asdict(api.default_compiler()))==C.digest(expected),'SOURCE_D1_COMPILER_PROFILE')


def adopt_QPY(row, expected, folder, source, a):
    """Read exact bytes; recursive hash may hydrate lazy local definitions only."""
    C.require(C.sha(ROOT/expected['path'])==expected['sha256'],'SOURCE_D1_FULL_QPY_SHA')
    with (ROOT/expected['path']).open('rb') as f:q=a.qpy.load(f)
    C.require(len(q)==1,'SOURCE_D1_ONE_QPY')
    actual=a.canonical_qiskit_circuit_fingerprint(q[0])
    origin=(ROOT/expected['path']).with_name('input.json')
    data=dict(C.read(origin),SOURCE=source,origin_SOURCE=C.read(origin)['SOURCE'],
        origin_input_qpy=expected['path'],origin_input_qpy_sha256=expected['sha256'],
        input_fingerprint=actual,expected_D1_fingerprint=expected['D1_input_fingerprint'],
        current_D1_fingerprint_matches=actual==expected['D1_input_fingerprint'],
        historical_fingerprint=expected['legacy_input_fingerprint'],
        historical_fingerprint_matches=actual==expected['legacy_input_fingerprint'],
        historical_status='LEGACY_FINGERPRINT_MATCH' if actual==expected['legacy_input_fingerprint'] else 'LEGACY_FINGERPRINT_MISMATCH',
        source_semantic_status='PENDING_PARENT_SOURCE_TOKEN_GATE',
        scientific_gate='PENDING_ADOPTED_SOURCE_GATE',full_input_build_calls=0,
        lazy_definition_hydration='Recursive fingerprint only, no new full emitter/control, state, unitary or cost compile')
    # Keep mismatch evidence even when the mandatory current-integrity gate fails.
    C.save(folder/'input.json',data)
    C.require(data['current_D1_fingerprint_matches'],'SOURCE_D1_CURRENT_QPY_FINGERPRINT_MISMATCH')
    shutil.copyfile(ROOT/expected['path'],folder/'input.qpy')
    C.require(C.sha(folder/'input.qpy')==expected['sha256'],'SOURCE_D1_QPY_COPY_CHANGED')
    body=(ROOT/expected['path']).with_name('body_before_control.qpy')
    if body.exists():shutil.copyfile(body,folder/body.name)
    return data


def task_run(kind,variant):
    ns=BASE/'run01';c,s,g=authority(task=True)
    found=[v for v in c['variants'] if v['id']==variant]
    C.require(len(found)==1,'SOURCE_D1_VARIANT');v=found[0];folder=ns/'variants'/variant
    C.require(kind in ('validate','build','compile'),'SOURCE_D1_TASK_KIND')
    if kind=='validate':
        C.require(v['historical_baseline'] and not (ns/'stage_A.json').exists(),'SOURCE_D1_VALIDATION_SCOPE')
        folder.mkdir()
    elif kind=='build':
        C.require(not v['historical_baseline'],'SOURCE_D1_NEW_FULL_FORBIDDEN')
        C.require(C.read(ns/'stage_B.json')['status']=='PASS_FULL4_RZ_REPRODUCED','SOURCE_D1_ABLATION_BEFORE_BASELINE')
        folder.mkdir()
    else:
        gate='stage_A.json' if v['historical_baseline'] else 'stage_B.json'
        state='PASS_FULL4_ADOPTED_SOURCE_CORRESPONDENCE' if v['historical_baseline'] else 'PASS_FULL4_RZ_REPRODUCED'
        C.require(C.read(ns/gate)['status']==state,'SOURCE_D1_COMPILE_STAGE_GATE')
    C.save(folder/(kind+'_claim.json'),dict(pid=os.getpid(),UTC=C.utc(),variant=variant))
    try:
        from . import h4_d1_adapter as A
        a=A.api();C.require(a.qiskit.__version__=='1.3.0','SOURCE_D1_QISKIT_VERSION')
        require_compiler(a,c['fixed']['compiler'])
        if kind=='validate':
            row=next(x for x in c['selected'] if x['case']==v['case'])
            adopt_QPY(row,c['full_inputs'][v['case']],folder,s['SOURCE'],a)
        elif kind=='build':
            if variant=='empty':built=A.empty(8);old=None
            else:
                old=next(x for x in c['selected'] if x['case']==v['case'])
                join=C.read(ROOT/old['price_join']);a.verify_join(join,expected_case=v['case'])
                raw=C.read(ROOT/('artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/'+join['representation']+'_model.json'))
                method=join['method']
                model=dict(representation=join['representation'],n_qubits=raw['n_qubits'],step_time=raw['step_time'],
                    hamiltonian=dict(constant=raw['hamiltonian']['constant']),cases={method:{k:raw['cases'][method][k]
                        for k in ('config','blocks','components','native_scalar_phases')}})
                built=A.build(model,method,join,v['part'])
            data=A.save_input(built,folder)
            data.update(variant=variant,case=v['case'],part=v['part'],SOURCE=s['SOURCE'],scientific_gate='ISOLATION_DIAGNOSTIC_NOT_VALID_SIMULATION')
            C.save(folder/'input.json',data)
        else:
            inp=C.read(folder/'input.json')
            with (folder/'input.qpy').open('rb') as f:q=a.qpy.load(f)
            C.require(len(q)==1 and a.canonical_qiskit_circuit_fingerprint(q[0])==inp['input_fingerprint'],'SOURCE_D1_INPUT_HASH_CHANGED')
            del q
            C.save(folder/'cost_transpile_started.json',dict(UTC=C.utc(),reserved_before_call=True,variant=variant))
            data=A.compile_saved(folder)
            C.require(data['input_fingerprint']==inp['input_fingerprint'],'SOURCE_D1_COMPILED_INPUT_HASH_CHANGED')
            if v['historical_baseline']:data['old_RZ_matches']=data['RZ']==c['old_full_RZ'][v['case']]
            data.update(variant=variant,SOURCE=s['SOURCE'])
            C.save(folder/'metrics.json',data)
        C.save(folder/(kind+'_finished.json'),dict(UTC=C.utc(),pid=os.getpid(),scope='NEW_SOURCE_D1_TASK_ONLY'))
        return 0
    except BaseException:
        C.save(folder/(kind+'_error.json'),dict(traceback=traceback.format_exc(),UTC=C.utc(),no_retry=True))
        raise


def batch(ns,kind,variants):
    budget=C.CallBudget(ns,LIMITS);children=[]
    try:
        for variant in variants:
            budget.take({'validate':'full_QPY_validation','build':'ablation_input_build','compile':'cost_transpile'}[kind])
            directory=ns/'native_tasks'/(kind+'_'+variant);directory.mkdir(parents=True)
            command=[PYTHON,'-B',str(SCRIPT),'--task','--kind',kind,'--variant',variant]
            with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
                child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
            children.append((variant,child,directory))
            C.save(directory/'launch.json',dict(pid=child.pid,identity=C._proc_identity(child.pid),argv=command,UTC=C.utc()))
        last=0
        while any(p.poll() is None for _,p,_ in children):
            if time.monotonic()-last>30:D.record_resources(ns);last=time.monotonic()
            time.sleep(.2)
        codes={}
        for variant,p,d in children:
            codes[variant]=p.wait()
            C.save(d/'native_exit.json',dict(pid=p.pid,exit_code=codes[variant],native_wait=True,reaped=True,UTC=C.utc()))
        C.require(all(v==0 for v in codes.values()),'SOURCE_D1_TASK_FAILURE:'+str(codes))
        return codes
    finally:
        for _,p,_ in children:
            if p.poll() is None:p.terminate()
        for _,p,d in children:
            if not (d/'native_exit.json').exists():
                code=p.wait();C.save(d/'native_exit.json',dict(pid=p.pid,exit_code=code,native_wait=True,reaped=True,UTC=C.utc()))


def stage_A(c,ns):
    # Reuse previous read-only token/phase-source comparison, no real build.
    saved=inspector().inspect_qpy(c['selected'])
    C.save(ns/'source_saved_structure.json',saved)
    links=C.read(DESIGN/'correspondence.json')['links']
    support={x['id']:x['assessment']=='SUPPORTED_SOURCE_RULE' for x in links}
    rows={}
    batch(ns,'validate',FULL)
    for variant in FULL:
        row=C.read(ns/'variants'/variant/'input.json')
        assessment=R.assess(historical_match=row['historical_fingerprint_matches'],source_links=support)
        C.require(assessment['research_GO_candidate'],'SOURCE_D1_UNRESOLVED_SEMANTIC_FIELD')
        rows[variant]=dict(current_fingerprint=row['input_fingerprint'],current_D1_fingerprint_matches=row['current_D1_fingerprint_matches'],
            historical_fingerprint=row['historical_fingerprint'],historical_status=row['historical_status'],
            historical_match=row['historical_fingerprint_matches'],SOURCE_semantic_status='SOURCE_SEMANTIC_CORRESPONDENCE',
            criterion_B_adopted=True,source_link_count=len(support),new_certificate=False,
            historical_program_identity='NOT_ESTABLISHED',accepted=True)
    C.save(ns/'stage_A.json',dict(status='PASS_FULL4_ADOPTED_SOURCE_CORRESPONDENCE',full_inputs=rows,
        SOURCE_semantic_rule='Twelve pinned SOURCE links + saved token/phase/control mapping + current QPY integrity; not lost historical program identity',
        full_input_build_calls=0,cost_compile_calls=0))


def worker():
    ns=BASE/'run01';c,s,g=authority(task=True)
    C.save(ns/'ENGINE_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    (ns/'variants').mkdir();D.record_resources(ns)
    stage='A'
    try:
        stage_A(c,ns)
        stage='B'
        metrics={}
        for variant in FULL:
            batch(ns,'compile',[variant]);data=C.read(ns/'variants'/variant/'metrics.json');metrics[variant]=data
            if not data['old_RZ_matches']:
                C.save(ns/'stage_B.json',dict(status='STOP_FULL_RZ_MISMATCH',full_metrics=metrics))
                C.save(ns/'ENGINE_STOP.json',dict(status='SOURCE_D1_STAGE_B_RZ_STOP_NO_RETRY',stage='B',stage_C='NOT_EXECUTED',mandatory_STOP=True))
                return 0
        C.save(ns/'stage_B.json',dict(status='PASS_FULL4_RZ_REPRODUCED',full_metrics=metrics))
        stage='C'
        parts=[v['id'] for v in c['variants'] if not v['historical_baseline']]
        batch(ns,'build',parts);batch(ns,'compile',parts)
        for variant in parts:metrics[variant]=C.read(ns/'variants'/variant/'metrics.json')
        from .h4_d1_adapter import interactions
        C.save(ns/'comparisons.json',dict(metrics=metrics,interactions=interactions(metrics),
            scope='ADOPTED_SOURCE_SELECTED_REPRESENTATIVES; NO_HISTORICAL_PROGRAM_IDENTITY_OR_UNIQUE_CAUSAL_SHARES'))
        C.save(ns/'stage_C.json',dict(status='COMPLETE_9_COST_COMPILES',variants=list(metrics)))
        C.save(ns/'ENGINE_STOP.json',dict(status='SOURCE_D1_COMPLETE_MANDATORY_STOP',stage='C',cost_compiles=9,mandatory_STOP=True))
        return 0
    except BaseException:
        if not (ns/'ENGINE_STOP.json').exists():C.save(ns/'ENGINE_STOP.json',dict(status='SOURCE_D1_FAILED_NO_RETRY',stage=stage,traceback=traceback.format_exc(),mandatory_STOP=True))
        raise


def launch():
    ns=BASE/'run01';c,s,g=authority(consume=True)
    C.save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc(),grant_id=g['grant_id']))
    tree=ast.parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='witness')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='NEW_C_WORKER_ONLY':n.value='NEW_SOURCE_D1_ENGINE_ONLY'
    env=dict(vars(C));exec(compile(ast.Module(body=[node],type_ignores=[]),'<reused SOURCE D1 native witness>','exec'),env)
    status=env['witness'](ns/'native',[PYTHON,'-B',str(SCRIPT),'--worker'])
    problem=None
    try:verify_seal(s)
    except BaseException as exc:problem=repr(exc)
    engine=C.read(ns/'ENGINE_STOP.json') if (ns/'ENGINE_STOP.json').exists() else dict(status='MISSING_ENGINE_STOP')
    native=C.read(ns/'native/worker_native_exit.json');stop=C.read(ns/'native/STOP.json')
    tasks=[C.read(p) for p in (ns/'native_tasks').glob('*/native_exit.json')] if (ns/'native_tasks').exists() else []
    C.save(ns/'STOP.json',dict(status=engine['status'] if status==0 and problem is None else 'SOURCE_D1_NATIVE_OR_PRESERVATION_FAILURE_NO_RETRY',
        stage=engine.get('stage'),native_engine_exit=native,all_owned_reaped=stop['all_owned_reaped'],
        direct_task_native_receipts=len(tasks),direct_task_exits=tasks,preservation_error=problem,
        old_preservation_PASS=problem is None,ledger=C.read(ns/'call_ledger.json'),mandatory_STOP=True,automatic_retry=False,
        original_H4_native_verified=False,formal_certificate_accepted=False,UTC=C.utc()))
    C.save(ns/'output_inventory.json',dict(files={str(p.relative_to(ns)):C.sha(p) for p in ns.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name!='output_inventory.json'}))
    return dict(status=C.read(ns/'STOP.json')['status'],stage=engine.get('stage'),
        native_engine_exit=native['exit_code'],ledger=C.read(ns/'call_ledger.json')['used'])
