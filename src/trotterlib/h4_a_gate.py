"""One saved order2 trajectory, one body/lift, two axis cost compiles only.

No quantum import before authority. D1 append/schedule and D2 lift/compiler,
seal verification, parallel compile batch and native witness are reused.
"""
import ast
from types import FunctionType, SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import traceback
from . import h4_d2_control as D2

C=D2.C
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_a_gate_order2/2026-10-24'
DESIGN=ROOT/'artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24'
SCRIPT=ROOT/'scripts/run_h4_a_gate.py'
PYTHON=D2.PYTHON
AXES=('cosine','sine')
ID='596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa'
EVENT='9cea6ce6d3659094b6b0d0a6da8b935cf32860c97a2e01a75a0761eded9f1779'
TRAJECTORY='0d16d201c87fdbb5f3e219dffb7dca7d48a69799290453e9ebb110a267db0e4b'
JOIN='artifacts/non_df_h4_pilot/2026-10-23/run01/run/pauli_jw_PR/trajectories/'+ID+'/price_join.json'
MODEL='artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json'
LIMITS=dict(body_replay=1,central_control_build=1,axis_wrapper_build=2,cost_transpile=2,
            generic_control_build=0,sampling=0,state_signal=0,N_G=0)
KIND='H4_A_GATE_ORDER2_ACTIVE_V1'


def body_only(model,join):
    """Literal D1 Pauli full schedule without its generic-control wrapper."""
    from . import h4_d1_adapter as A
    a=A.api();case=model['cases']['PR'];cfg=case['config'];dt=model['step_time']
    C.require(model['representation']=='pauli_jw' and join['representation']=='pauli_jw'
              and join['method']==cfg['method']=='PR','AG_BODY_CASE')
    C.require(cfg['T']/cfg['q']==dt,'AG_STEP_TIME')
    groups=A.grouped_events(join,cfg['q'],cfg['r'])
    labels={str(i):c['label'] for i,c in enumerate(case['components'])}
    body=a.QuantumCircuit(model['n_qubits']);tags=[];phases=[]
    for step,events in enumerate(groups):
        for reverse in (False,True):
            if reverse:
                for event in events:
                    start=len(body.data);before=float(body.global_phase)
                    a.append_pauli_event(body,event,labels)
                    tags.append(dict(step=step,role='event',order=event.taylor_order,start=start,stop=len(body.data)))
                    phases.append(dict(step=step,owner='event',before=before,after=float(body.global_phase),
                        rule='original append_pauli_event: negative product pi then order phase',
                        order=event.taylor_order,phase=event.to_dict()['phase']))
            start=len(body.data)
            for block in (reversed(case['blocks']) if reverse else case['blocks']):
                a.append_pauli_evolution(body,block['label'],block['coefficient'],dt/2)
            tags.append(dict(step=step,role='retained',direction='reverse' if reverse else 'forward',start=start,stop=len(body.data)))
        before=float(body.global_phase)
        body.global_phase-=model['hamiltonian']['constant']*dt
        phases.append(dict(step=step,owner='constant',before=before,after=float(body.global_phase),
                           rule='global_phase -= constant * step_time'))
    return SimpleNamespace(body=body,sidecar=dict(tags=tags,phase_updates=phases,
        aggregate_phase=float(body.global_phase),control_policy='BODY_ONLY_NO_WRAPPER_OR_GENERIC_CONTROL'))


def scope(c):
    plan=C.read(DESIGN/'contracts_DRAFT.json')
    target=next(x for x in plan['candidates'] if x['id']=='A_GATE_ORDER2')['fixed_selection']
    C.require(c['kind']==KIND and all(c[k] is True for k in ('execution_authorized','dispatch_enabled','grant_valid')),'AG_AUTHORIZATION')
    C.require(c['target']==target and target['trajectory_index']==1 and target['trajectory_id']==ID,'AG_TARGET')
    C.require(target['event_record_hash']==EVENT and target['trajectory_record_hash']==TRAJECTORY,'AG_TARGET_HASH')
    C.require(c['fixed']==C.read(D2.BASE/'active_contract.json')['fixed'],'AG_FIXED')
    C.require(c['call_limits']==LIMITS and c['per_axis_compile_max']==dict(cosine=1,sine=1),'AG_BUDGET')
    C.require(c['mandatory_STOP'] is True and c['automatic_retry'] is False and c['automatic_next_stage'] is False,'AG_STOP')


def verify_seal(s,*,runtime=True,committed=True):
    C.require(s['kind']=='H4_A_GATE_SOURCE_INPUT_RUNTIME_SEAL_V1','AG_SEAL_KIND')
    # Rebind only the artifact root. Reuse every D2 hash/approval/runtime check.
    f=FunctionType(D2.verify_seal.__code__,dict(D2.__dict__,BASE=BASE))
    return f(dict(s,kind='H4_D2_SOURCE_INPUT_RUNTIME_SEAL_V1'),runtime=runtime,committed=committed)


def prepare():
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(),'AG_NAMESPACE_USED')
    approval=C.read(BASE/'approval.json');text=(BASE/'user_instruction.txt').read_text()
    C.require('A_GATE_ORDER2の限定科学実行を正式に承認' in text and ID in text and '最大2回' in text,'AG_EXPLICIT_APPROVAL')
    C.require(approval['instruction_sha256']==C.sha(BASE/'user_instruction.txt'),'AG_APPROVAL_BYTES')
    t=C.read(BASE/'artificial/new_tests.json');C.require(t['tests']==12 and t['failures']==t['errors']==0,'AG_TESTS')
    for p,count in [('additional.json',8),('reused_d2.json',12)]:
        r=C.read(DESIGN/'artificial'/p);C.require(r['tests']==count and r['errors']==r['failures']==0,'AG_REUSED_TESTS')
    old=C.read(D2.BASE/'seal.json');code=dict(old['code_files']);inputs=dict(old['input_files'])
    for p in ('src/trotterlib/h4_a_gate.py','scripts/run_h4_a_gate.py','tests/test_h4_a_gate.py','tests/test_h4_controlled_comparison.py'):
        code[p]=C.sha(ROOT/p)
    for p in (ROOT/JOIN,DESIGN/'contracts_DRAFT.json',DESIGN/'cohort_manifest.json',BASE/'approval.json',BASE/'user_instruction.txt',BASE/'output_schema.json',BASE/'artificial/new_tests.json'):
        inputs[str(p.relative_to(ROOT))]=C.sha(p)
    protected=dict(C.read(DESIGN/'preservation_before.json'))
    for p in DESIGN.rglob('*'):
        if p.is_file() and 'mplconfig' not in p.parts:protected[str(p.relative_to(ROOT))]=C.sha(p)
    C.save(BASE/'preservation_before.json',protected)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    s=dict(kind='H4_A_GATE_SOURCE_INPUT_RUNTIME_SEAL_V1',SOURCE=source,code_files=code,input_files=inputs,
        python_sha256=C.sha(sys.executable),runtime_manifest=old['runtime_manifest'],runtime_manifest_sha256=old['runtime_manifest_sha256'],
        approval_sha256=C.sha(BASE/'approval.json'),preservation_sha256=C.sha(BASE/'preservation_before.json'),real_H4_calls_before_seal=0,UTC=C.utc())
    verify_seal(s);C.save(BASE/'seal.json',s)
    target=C.read(DESIGN/'cohort_manifest.json')['selected']
    c=dict(kind=KIND,status='APPROVED_SEALED_A_GATE_ORDER2',target=target,fixed=C.read(D2.BASE/'active_contract.json')['fixed'],
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,science_SOURCE=source,
        call_limits=LIMITS,per_axis_compile_max=dict(cosine=1,sine=1),mandatory_STOP=True,automatic_retry=False,automatic_next_stage=False,
        seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=s['approval_sha256'],output_schema_sha256=C.sha(BASE/'output_schema.json'),
        output_namespace=str(BASE/'run01'),launch_command=[PYTHON,'-B',str(SCRIPT),'--launch'])
    scope(c);C.save(BASE/'active_contract.json',c)
    ns=BASE/'run01';ns.mkdir()
    g=dict(kind='H4_A_GATE_ORDER2_ONE_SHOT',SOURCE=source,namespace=str(ns),contract_sha256=C.sha(BASE/'active_contract.json'),
        seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=s['approval_sha256'],one_shot=True,automatic_retry=False,call_limits=LIMITS)
    g['grant_id']=C.digest(g);C.save(ns/'grant.json',g)
    C.save(ns/'call_ledger.json',dict(limits=LIMITS,used=dict.fromkeys(LIMITS,0),events=[]));return g


def authority(*,consume=False,task=False):
    ns=BASE/'run01';c=C.read(BASE/'active_contract.json');s=C.read(BASE/'seal.json');g=C.read(ns/'grant.json');scope(c)
    C.require(g['kind']=='H4_A_GATE_ORDER2_ONE_SHOT' and g['namespace']==str(ns),'AG_GRANT_NAMESPACE')
    C.require(g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}),'AG_GRANT_DIGEST')
    C.require(g['SOURCE']==s['SOURCE']==c['science_SOURCE'] and g['call_limits']==LIMITS,'AG_GRANT_SOURCE_SCOPE')
    C.require(g['contract_sha256']==C.sha(BASE/'active_contract.json') and g['seal_sha256']==C.sha(BASE/'seal.json')==c['seal_sha256'],'AG_GRANT_BINDING')
    C.require(g['approval_sha256']==s['approval_sha256']==c['approval_sha256'] and g['one_shot'] is True and g['automatic_retry'] is False,'AG_GRANT_APPROVAL')
    C.require(not (ns/'STOP.json').exists() and not (ns/'ENGINE_STOP.json').exists(),'AG_ALREADY_STOPPED')
    verify_seal(s,runtime=not task,committed=not task)
    if task:C.require(C.read(ns/'CONSUMED.json')['grant_id']==g['grant_id'] and (ns/'LAUNCH_CLAIM.json').exists(),'AG_NOT_LAUNCHED')
    if consume:
        C.require(not (ns/'CONSUMED.json').exists(),'AG_CONSUMED_NO_RETRY')
        C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id'],pid=os.getpid(),UTC=C.utc(),nonrefundable=True))
    return c,s,g


def stage_A(c,ns,a):
    from . import h4_d1_adapter as A
    j=C.read(ROOT/JOIN);a.verify_join(j,expected_case='pauli_jw_PR')
    for k,v in [('trajectory_index',1),('trajectory_id',ID),('event_record_hash',EVENT),('trajectory_record_hash',TRAJECTORY),('q',8),('r',4)]:
        C.require(j[k]==v,'AG_SAVED_JOIN:'+k)
    C.require(j['binding']['SOURCE']=='0c329b30a81accb35c212be8defa66fee0dd0f12','AG_ORIGINAL_SOURCE')
    C.require(any(e['taylor_order']==2 for e in j['events']) and not any(e['taylor_order']==4 for e in j['events']),'AG_FIXED_EVENT_COVERAGE')
    raw=C.read(ROOT/MODEL);case=raw['cases']['PR'];cfg=case['config']
    C.require(raw['n_qubits']==8 and raw['step_time']==.1 and len(raw['hamiltonian']['terms'])==184,'AG_MODEL')
    C.require(len(case['blocks'])==27 and all(cfg[k]==v for k,v in [('T',.8),('q',8),('r',4),('K',4)]),'AG_SCHEDULE')
    groups=A.grouped_events(j,8,4)
    for record in j['events']:
        for app in record['application_sequence']:
            comp=case['components'][int(app['component_id'])]
            C.require(comp['sign']==app['coefficient_sign'] and comp['coefficient_abs']==app['coefficient_abs'] and set(comp['label'])!={'I'},'AG_COMPONENT_SIGN_LABEL')
    model=dict(representation='pauli_jw',n_qubits=8,step_time=raw['step_time'],hamiltonian=dict(constant=raw['hamiltonian']['constant']),
        cases={'PR':{k:case[k] for k in ('config','blocks','components')}})
    expected,phase_rules=D2.S.inspector().pauli_recipe(model,'PR',j)
    C.save(ns/'stage_A.json',dict(status='PASS_FIXED_ORDER2_SOURCE',trajectory_id=ID,event_record_hash=EVENT,
        trajectory_record_hash=TRAJECTORY,event_orders=[e['taylor_order'] for e in j['events']],
        saved_rotation_angles=[e['rotation_angle'] for e in j['events']],phase_SOURCE_rules=phase_rules,
        compiler=c['fixed']['compiler'],historical_hash='FAIL_UNCHANGED_NOT_RECONSTRUCTED'))
    return model,j,expected


def stage_B(c,s,ns,a,model,j,expected):
    from . import h4_d2_pauli_control as P
    budget=C.CallBudget(ns,LIMITS)
    budget.take('body_replay');C.save(ns/'body_replay_claim.json',dict(pid=os.getpid(),UTC=C.utc()))
    replay=body_only(model,j)
    inspector=D2.S.inspector();actual=[inspector.primitive_token(x,replay.body) for x in replay.body.data]
    match=actual==expected
    C.save(ns/'body_source_check.json',dict(numeric_tokens_match=match,actual_tokens_sha256=C.digest(actual),expected_tokens_sha256=C.digest(expected),
        source_instruction_count=len(actual),sidecar=replay.sidecar,historical_program_identity='NOT_ESTABLISHED'))
    C.require(match,'AG_REPLAY_SOURCE_TOKENS')
    with (ns/'source_body.qpy').open('xb') as f:a.qpy.dump(replay.body,f)
    budget.take('central_control_build');result=P.central_control_body(replay.body)
    with (ns/'controlled_body.qpy').open('xb') as f:a.qpy.dump(result.body,f)
    C.save(ns/'controlled_mapping.json',result.sidecar)
    C.require(result.sidecar['source_instruction_count']==len(actual) and result.sidecar['aggregate_source_global_phase']==replay.sidecar['aggregate_phase'],'AG_PHASE_MAPPING')
    C.save(ns/'stage_B.json',dict(status='PASS_BODY_ONLY_AND_CENTRAL_ONCE',source_body_sha256=C.sha(ns/'source_body.qpy'),
        controlled_body_sha256=C.sha(ns/'controlled_body.qpy'),aggregate_phase=replay.sidecar['aggregate_phase'],new_certificate=False))
    return result


def wrappers(ns,s,a,result):
    from . import h4_d1_adapter as A
    (ns/'variants').mkdir()
    for axis in AXES:
        C.CallBudget(ns,LIMITS).take('axis_wrapper_build');folder=ns/'variants'/axis;folder.mkdir()
        circuit=a.hadamard_wrapper(result.body,axis,already_controlled=True)
        data=A.save_input(SimpleNamespace(circuit=circuit,body=result.body,sidecar=result.sidecar),folder)
        data.update(axis=axis,SOURCE=s['SOURCE'],trajectory_id=ID,input_qpy_sha256=C.sha(folder/'input.qpy'),controlled_body_sha256=C.sha(ns/'controlled_body.qpy'))
        C.save(folder/'input.json',data)
    C.save(ns/'stage_C_inputs.json',dict(status='PASS_BOTH_AXIS_INPUTS_SAVED',axes=list(AXES)))


def compile_task(axis):
    c,s,g=authority(task=True);C.require(axis in AXES,'AG_AXIS');ns=BASE/'run01';d=ns/'variants'/axis
    C.require(C.read(ns/'stage_C_inputs.json')['status']=='PASS_BOTH_AXIS_INPUTS_SAVED','AG_COMPILE_GATE')
    r=C.read(ns/'native_tasks'/('compile_'+axis)/'reservation.json')
    C.require(r['kind']=='compile' and r['case']==axis and r['grant_id']==g['grant_id'],'AG_RESERVATION')
    C.save(d/'compile_claim.json',dict(pid=os.getpid(),axis=axis,UTC=C.utc()))
    try:
        from . import h4_d1_adapter as A
        a=A.api();D2.S.require_compiler(a,c['fixed']['compiler']);C.require(a.qiskit.__version__=='1.3.0','AG_QISKIT')
        i=C.read(d/'input.json');C.require(C.sha(d/'input.qpy')==i['input_qpy_sha256'],'AG_INPUT_CHANGED')
        C.save(d/'cost_transpile_started.json',dict(UTC=C.utc(),nonrefundable=True))
        m=A.compile_saved(d);C.require(m['input_fingerprint']==i['input_fingerprint'],'AG_COMPILED_INPUT_HASH')
        m.update(axis=axis,SOURCE=s['SOURCE'],trajectory_id=ID,output_qpy_sha256=C.sha(d/'output.qpy'))
        C.save(d/'metrics.json',m);C.save(d/'compile_finished.json',dict(pid=os.getpid(),UTC=C.utc()));return 0
    except BaseException:
        C.save(d/'compile_error.json',dict(traceback=traceback.format_exc(),no_retry=True));raise


def compile_both(ns):
    # Same tested D2 parallel dispatcher; only namespace/axis/call bindings differ.
    f=FunctionType(D2.batch.__code__,dict(D2.__dict__,CASES=AXES,CALLS=dict(compile='cost_transpile'),LIMITS=LIMITS,SCRIPT=SCRIPT))
    return f(ns,'compile',AXES)


def limited_comparison(ns,j):
    metrics={axis:C.read(ns/'variants'/axis/'metrics.json') for axis in AXES}
    old0=C.read(D2.BASE/'run01/variants/pauli_jw_PR/metrics.json')
    rows={axis:dict(generic_index1_RZ=j['compiled_RZ'][axis],central_index1_RZ=m['RZ'],
        signed_central_minus_generic_RZ=m['RZ']-j['compiled_RZ'][axis],
        D2_index0_cosine_RZ=old0['RZ'],signed_index1_minus_index0_cosine_RZ=m['RZ']-old0['RZ'] if axis=='cosine' else None)
        for axis,m in metrics.items()}
    delta={k:metrics['sine'][k]-metrics['cosine'][k] for k in ('RZ','CX','depth','rz_depth','cx_depth')}
    C.save(ns/'comparisons.json',dict(metrics=metrics,comparisons=rows,signed_sine_minus_cosine=delta,
        scope='ONE_POSTSELECTED_STRUCTURAL_COVERAGE_TRAJECTORY_NOT_COHORT_OR_POPULATION',new_N_G=False))


def worker():
    c,s,g=authority(task=True);ns=BASE/'run01';C.save(ns/'ENGINE_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()));stage='A'
    try:
        from . import h4_d1_adapter as A
        a=A.api();D2.S.require_compiler(a,c['fixed']['compiler']);C.require(a.qiskit.__version__=='1.3.0','AG_QISKIT')
        model,j,expected=stage_A(c,ns,a)
        stage='B';result=stage_B(c,s,ns,a,model,j,expected)
        stage='C';wrappers(ns,s,a,result);compile_both(ns);limited_comparison(ns,j)
        C.save(ns/'ENGINE_STOP.json',dict(status='A_GATE_COMPLETE_MANDATORY_STOP',stage=stage,mandatory_STOP=True));return 0
    except BaseException:
        C.save(ns/'ENGINE_STOP.json',dict(status='A_GATE_FAILED_NO_RETRY',stage=stage,traceback=traceback.format_exc(),mandatory_STOP=True));raise


def launch():
    c,s,g=authority(consume=True);ns=BASE/'run01';C.save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc(),grant_id=g['grant_id']))
    tree=ast.parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='witness')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='NEW_C_WORKER_ONLY':n.value='NEW_A_GATE_ENGINE_ONLY'
    env=dict(vars(C));exec(compile(ast.Module(body=[node],type_ignores=[]),'<reused A_GATE native witness>','exec'),env)
    status=env['witness'](ns/'native',[PYTHON,'-B',str(SCRIPT),'--worker'])
    error=None
    try:verify_seal(s)
    except BaseException as exc:error=repr(exc)
    engine=C.read(ns/'ENGINE_STOP.json') if (ns/'ENGINE_STOP.json').exists() else dict(status='MISSING_ENGINE_STOP')
    native=C.read(ns/'native/worker_native_exit.json');stop=C.read(ns/'native/STOP.json')
    tasks=[C.read(p) for p in (ns/'native_tasks').glob('*/native_exit.json')]
    success=status==0 and error is None and stop['all_owned_reaped'] and len(tasks)==2 and all(t['exit_code']==0 and t['reaped'] for t in tasks) and engine['status']=='A_GATE_COMPLETE_MANDATORY_STOP'
    C.save(ns/'STOP.json',dict(status='A_GATE_COMPLETE_MANDATORY_STOP' if success else 'A_GATE_FAILED_NO_RETRY',native_engine_exit=native,
        all_owned_reaped=stop['all_owned_reaped'],task_native_receipts=tasks,preservation_error=error,ledger=C.read(ns/'call_ledger.json'),
        original_H4_native_verified=False,formal_certificate_accepted=False,mandatory_STOP=True,automatic_next_stage=False,UTC=C.utc()))
    C.save(ns/'output_inventory.json',dict(files={str(p.relative_to(ns)):C.sha(p) for p in ns.rglob('*') if p.is_file() and p.name!='output_inventory.json'}))
    return dict(status=C.read(ns/'STOP.json')['status'],ledger=C.read(ns/'call_ledger.json')['used'])
