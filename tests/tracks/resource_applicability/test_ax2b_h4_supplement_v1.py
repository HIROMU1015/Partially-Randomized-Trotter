"""Synthetic arithmetic + metadata/mock regressions; no molecular execution."""
import ast
import builtins
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import mpmath as mp
import pytest

from trottertracks.resource_applicability import ax2b_supplement_launch_v1 as launch
from trottertracks.resource_applicability.ax2b_mp_cache_v1 import OperatorCache, oracle_identity
from trottertracks.resource_applicability.ax2b_stage_validation_v2 import mp_cell as old_mp
from trottertracks.resource_applicability.ax2b_stage_validation_v3 import mp_cell as cached_mp
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter, Progress, latest_progress
from trottertracks.resource_applicability.ax2b_supplement_watchdog_v1 import supervise
from trottertracks.resource_applicability.ax2b_coverage_binding_v3 import canonical_coverage
from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
from trottertracks.resource_applicability.ax2a_preparation import digest

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT/'src/trottertracks/resource_applicability/ax2b_h4_supplement_port_v1.py'


@pytest.fixture(autouse=True)
def forbid_molecular_io_and_circuit_imports(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('pyscf','qiskit','openfermion'):
            raise AssertionError('MOLECULAR_OR_CIRCUIT_IMPORT_FORBIDDEN')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    original_open = Path.open
    def opened(path, *args, **kwargs):
        if str(path).endswith('.npz'):
            raise AssertionError('MOLECULAR_SNAPSHOT_READ_FORBIDDEN')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)


def toy():
    # Artificial two-orbital one-particle sector; no molecule or saved input.
    return NS(constant=.11, one_body=np.array([[.5,.13],[.13,-.2]]),
              lambdas=np.array([.07,-.04]),
              g_matrices=np.array([[[.2,.1],[.1,-.1]],[[-.12,.07],[.07,.25]]]))


def cell(method='B2', formula='2nd', q=4, K=2):
    return {'id':'SYNTHETIC','method':method,'formula':formula,
            'prefix':2 if method == 'B1' else 0 if method == 'B3' else 1,
            'q':q,'R':8 if method in ('B2','B3') else None,
            'K':K if method in ('B2','B3') else None}


@pytest.mark.parametrize('spec', [('B1','2nd',4,2),('B1','4th',1,2),('B1','4th',4,2),
                                  ('B0','2nd',4,2),('B2','2nd',4,2),('B2','2nd',4,4),('B3','2nd',4,6)])
@pytest.mark.parametrize('dps',[80,120])
@pytest.mark.parametrize('T',[.8,-.8])
def test_old_new_all_decimal_signals_states_and_stages_identical(spec,dps,T,monkeypatch):
    kwargs = dict(T=T,scalar=.133,lambda_r=.8,extracted_identity=.023,dps=dps)
    calls = []
    original = mp.expm
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(mp,'expm',counted)
    before = old_mp(toy(),[2,1],[1.,0.],cell(*spec),**kwargs)
    old_calls=len(calls);calls.clear(); snapshots=[]
    after = cached_mp(toy(),[2,1],[1.,0.],cell(*spec),progress=snapshots.append,**kwargs)
    work=after.pop('oracle_work')
    assert after == before  # Includes every serialized MP state, norm and stage.
    assert len(calls) == work['completed']['exp'] == work['attempted']['exp']
    assert len(calls) < old_calls
    assert work['hits'] > 0 and work['logical_stage_actions'] == len(before['trace'])
    assert work['heap_bytes'] <= 64*2**20 and work['entries'] <= 128
    assert snapshots[0]['point'] == 'operator_started'
    assert snapshots[0]['oracle_work']['attempted']['exp'] == 1
    assert snapshots[0]['oracle_work']['completed']['exp'] == 0
    assert snapshots[-1]['oracle_work']['completed']['exp'] == len(calls)
    assert before['certified'] is False


def test_no_cross_cell_or_precision_reuse():
    results=[cached_mp(toy(),[2,1],[1.,0.],cell(),T=.8,scalar=.11,lambda_r=.8,dps=d)
             for d in (80,120,80)]
    assert results[0] == results[2]
    assert results[0]['oracle_work']['misses'] == results[1]['oracle_work']['misses']
    assert results[0]['signals'] != results[1]['signals']


@pytest.mark.parametrize('change',['coefficient','basis','state','time','signed_zero','dps','lambda','identity','K','formula','backend'])
def test_cache_identity_separates_exact_semantics(change):
    args=dict(ham=toy(),basis=[2,1],state=[1.,0.],cell=cell(),T=.8,scalar=.11,
              lambda_r=.8,extracted_identity=0.,dps=80,backend='synthetic')
    base=oracle_identity(**args)
    if change=='coefficient':args['ham'].one_body[0,0]=math.nextafter(.5,1.)
    elif change=='basis':args['basis'].reverse()
    elif change=='state':args['state']=[0.,1.]
    elif change=='time':args['T']=-.8
    elif change=='signed_zero':args['extracted_identity']=-0.
    elif change=='dps':args['dps']=120
    elif change=='lambda':args['lambda_r']=.9
    elif change=='identity':args['extracted_identity']=.01
    elif change=='K':args['cell']['K']=4
    elif change=='formula':args['cell']['formula']='4th'
    else:args['backend']='other'
    assert oracle_identity(**args)!=base


def test_matrix_mutation_and_signed_times():
    cache=OperatorCache('synthetic')
    m=mp.eye(2)
    first=cache.get(('primitive',1,.2.hex()),lambda:m)
    first[0,0]=7;m[1,1]=8
    again=cache.get(('primitive',1,.2.hex()),lambda:pytest.fail('unexpected miss'))
    assert again==mp.eye(2)
    cache.get(('primitive',1,(-.2).hex()),lambda:mp.eye(2))
    assert cache.stats['misses']==2 and cache.stats['hits']==1


def test_precision_context_rejects_before_factory():
    with mp.workdps(80):
        cache=OperatorCache('synthetic',mp_precision=mp.mp.prec)
        cache.get(('test',),lambda:mp.eye(2))
    with mp.workdps(120),pytest.raises(ValueError,match='PRECISION_CONTEXT'):
        cache.get(('test',),lambda:pytest.fail('factory should not run'))


@pytest.mark.parametrize('limit',['entries','exp','lookups','heap'])
def test_cache_caps_and_attempted_completed(limit):
    kwargs={'entries':1} if limit=='entries' else {'exp_generations':1} if limit=='exp' else {'lookups':1} if limit=='lookups' else {'heap_bytes':1}
    cache=OperatorCache('synthetic',**kwargs)
    if limit=='heap':
        with pytest.raises(RuntimeError,match='HEAP_CAP'):cache.get(('a',),lambda:mp.eye(2))
        assert cache.stats['completed']['exp']==1 and cache.stats['entries']==0
    else:
        cache.get(('a',),lambda:mp.eye(2))
        with pytest.raises(RuntimeError):cache.get(('b',),lambda:pytest.fail('over-cap factory ran'))
        assert cache.stats['attempted']['exp']==cache.stats['completed']['exp']==1


def test_failed_factory_is_attempted_not_completed():
    records=[];cache=OperatorCache('synthetic',progress=records.append)
    def fail():raise RuntimeError('injected')
    with pytest.raises(RuntimeError,match='injected'):cache.get(('a',),fail)
    assert cache.stats['attempted']['exp']==1 and cache.stats['completed']['exp']==0
    assert len(records)==1 and records[0]['point']=='operator_started'


def test_atomic_exclusive_output_progress_caps_and_no_guessing(tmp_path):
    writer=AtomicWriter(tmp_path,byte_cap=2**20)
    progress=Progress(writer,'S4_MP',cap=2)
    progress.update(mp_attempted=1,dps=80,point='started')
    writer.write('toy_mp80.json',{'synthetic':True})
    last=latest_progress(tmp_path)
    assert last['mp_attempted']==last['mp_completed']==1 and last['last_completed_record']=='toy_mp80.json'
    with pytest.raises(RuntimeError,match='PROGRESS_RECORD_CAP'):progress.update(mp_attempted=2)
    writer.observer=None
    with pytest.raises(FileExistsError):writer.write('toy_mp80.json',{'synthetic':False})
    assert json.loads((tmp_path/'toy_mp80.json').read_text())=={'synthetic':True}
    assert not list(tmp_path.glob('.pending_*'))


def test_atomic_preserves_existing_pending_and_output_caps(tmp_path):
    writer=AtomicWriter(tmp_path,byte_cap=65550)
    with pytest.raises(RuntimeError,match='OUTPUT_WRITE_CAP'):writer.write('large.json',{'v':'x'*100})
    writer=AtomicWriter(tmp_path,byte_cap=2**20)
    (tmp_path/'.pending_toy.json').write_text('protected')
    with pytest.raises(FileExistsError):writer.write('toy.json',{})
    assert (tmp_path/'.pending_toy.json').read_text()=='protected'


@pytest.mark.parametrize('unit',launch.UNITS)
def test_plan_retains_full_prerequisites_and_only_missing_obligations(unit):
    p=launch.plan(unit)
    assert len(p['cells'])==8 and p['dps']==[80,120]
    assert len(p['selected_correctness_ids'])==(2 if unit=='S4_MP' else 0)
    assert len(p['explicit_groups'])==(0 if unit=='S4_MP' else 4)
    assert p['caps_proposed']['primitive']==537
    assert all(p['caps_proposed'][k]==0 for k in ('compile','trajectory','occurrence'))
    assert p['H6_status']=='H6_NOT_AUTHORIZED' and p['N'] is None and p['G'] is None
    for c in p['cells']:
        if c['id'] in ('H4_B1_S4_q1','H4_B1_S4_q4'):
            schedule=primitive_time_schedule(dict(c,formula=c['order']),T=p['T'])
            pairs=set(map(tuple,schedule['ordinary_one_outer_step']))
            assert len(pairs)+1<=p['cache_policy']['exp_generations']
            assert any(t<0 for _,t in pairs)


def isolated_method(name, namespace):
    cls=next(n for n in ast.parse(PORT.read_text()).body if isinstance(n,ast.ClassDef))
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PORT),'exec'),namespace)
    return namespace[name]


@pytest.mark.parametrize('unit',launch.UNITS)
def test_split_unit_routing_does_not_make_events_depend_on_s4(unit):
    calls=[]
    port=NS(unit=unit,setup=lambda:calls.append('setup'),phase=lambda p:calls.append(p),
            primitives=lambda:calls.append('primitives'),correctness=lambda:calls.append('S4_MP'),
            explicit_estimator_probes=lambda:calls.append('EVENT_CONTROL'))
    isolated_method('execute_unit',{})(port)
    assert calls==['setup','primitive_validation','primitives']+(['S4_MP'] if unit=='S4_MP' else ['validation','EVENT_CONTROL'])


def test_correctness_selects_only_two_missing_cells_and_keeps_two_precisions(tmp_path):
    writer=AtomicWriter(tmp_path,byte_cap=2**20);progress=Progress(writer,'S4_MP')
    selected=[];mp_calls=[]
    cells=launch.plan('S4_MP')['cells']
    preps={c['id']:NS(constant_coefficient=.11,extracted_identity_coefficient=0.) for c in cells}
    actual={'signals':{'corrected':1+0j},'intermediate_norm_max':1.,'trace':[],
            'counts':{'deterministic':0,'tail':0},'log_B':0.}
    def signal(raw):selected.append(raw['id']);return actual
    def oracle(*args,**kwargs):
        mp_calls.append((args[3]['id'],kwargs['dps']))
        return {'signals':{'corrected':{'real':'1','imag':'0'},'reference':{'real':'1','imag':'0'}},'trace':[]}
    namespace={'time':__import__('time'),'resource':__import__('resource'), 'math':math,
        'canonical_cell':lambda c:dict(c,formula=c['order']), 'complex_record':lambda z:{'real':z.real,'imag':z.imag},
        'mp_cell':oracle,'finite_scale_guard':lambda **kw:None,'compare_stages':lambda a,b:{},
        'compare_mp_records':lambda a,b:{'signal_differences':{}},'S4_IDS':('H4_B1_S4_q1','H4_B1_S4_q4')}
    port=NS(cells=cells,preps=preps,progress=progress,plan=launch.plan('S4_MP'),kind='H4_LIMITED',
        phase=lambda p:writer.write('phase_'+p+'.json',{'phase':p,'elapsed':0}),cell_signal=signal,
        ham=None,sector=NS(basis_indices=[2,1]),saved_state=[1.,0.],reference=1+0j,writer=writer,completed=0)
    isolated_method('correctness',namespace)(port)
    assert selected==['H4_B1_S4_q1','H4_B1_S4_q4']
    assert mp_calls==[(i,d) for i in selected for d in (80,120)]
    assert progress.row['correctness_attempted']==progress.row['correctness_completed']==port.completed==2
    assert progress.row['mp_attempted']==progress.row['mp_completed']==4


def test_control_failure_is_recorded_before_action_and_not_completed(tmp_path):
    progress=Progress(AtomicWriter(tmp_path,byte_cap=2**20),'EVENT_CONTROL')
    from trottertracks.resource_applicability.ax2b_limits import CallBudget
    port=NS(calls=CallBudget(control_probe=1),progress=progress)
    def fail(*args):raise RuntimeError('injected control error')
    action=isolated_method('control_state',{'simulate_statevector':fail})
    with pytest.raises(RuntimeError,match='injected'):action(port,None,None)
    assert latest_progress(tmp_path)['control_attempted']==1
    assert latest_progress(tmp_path)['control_completed']==0
    with pytest.raises(RuntimeError,match='CALL_BUDGET'):action(port,None,None)
    assert progress.row['control_attempted']==1


def bound_mock_manifest(monkeypatch,tmp_path):
    m=launch.preparation('S4_MP');m['execution_plan_sealed']=True
    m['assigned_resources']={'assigned_cpu':0,'science_workers':1,'blas_threads':1}
    out=tmp_path/'new';m['intended_exclusive_output']={'absolute_path':str(out.resolve()),
            'repository_path':m['plan']['output_namespace']+'synthetic-launch'}
    saved=json.loads((ROOT/'artifacts/resource_applicability/track_a_ax2b_h4_limited_seal_v2/2026-10-10/sealed_preparation_manifest_v2.json').read_text())
    m['coverage_binding']=saved['coverage_binding'];m['environment']={'synthetic':True}
    monkeypatch.setattr(launch,'safe_path',lambda root,name:out.resolve())
    monkeypatch.setattr(launch,'verify_sources',lambda *a:None)
    monkeypatch.setattr(launch,'verify_input',lambda *a:12)
    monkeypatch.setattr(launch,'environment',lambda:{'synthetic':True})
    monkeypatch.setattr(launch.os,'sched_getaffinity',lambda pid:{0})
    g={'schema':'track_a_ax2b_supplement_authorization_v1','approved_by_user':True,'manifest_digest':digest(m),
       'unit':'S4_MP','assigned_cpu':0,'exclusive_output':str(out.resolve()),'retry':False,'resume':False}
    return m,g,out


def test_unapproved_and_old_grants_rejected_before_io(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'verify_sources',lambda *a:pytest.fail('I/O before grant'))
    for g,reason in (({},'EXPLICIT_USER_GRANT'),({'approved_by_user':True,'schema':'track_a_ax2b_bound_authorization_v3'},'GRANT_SCHEMA')):
        with pytest.raises(ValueError,match=reason):launch.validate_launch(ROOT,{},g,requested=True,output=tmp_path)


@pytest.mark.parametrize('change',['unit','plan','plan_bool','resources_bool','coverage','sealed','cpu','output','resume','environment','H6','next'])
def test_bound_launch_mutations_fail_closed(change,tmp_path,monkeypatch):
    m,g,out=bound_mock_manifest(monkeypatch,tmp_path)
    assert launch.validate_launch(ROOT,m,g,requested=True,output=out)==0
    if change=='unit':g['unit']='EVENT_CONTROL'
    elif change=='plan':m['plan']['dps']=[80]
    elif change=='plan_bool':m['plan']['cells'][0]['q']=True
    elif change=='resources_bool':m['assigned_resources']['science_workers']=True
    elif change=='coverage':m['coverage_binding']['expected_bounds']['primitive_actions']=536
    elif change=='sealed':m['execution_plan_sealed']=False
    elif change=='cpu':g['assigned_cpu']=True
    elif change=='output':out.mkdir()
    elif change=='resume':g['resume']=True
    elif change=='environment':m['environment']={}
    elif change=='H6':m['kind']='H6_TECHNICAL'
    else:m['next_stage_authorized']=True
    g['manifest_digest']=digest(m)
    with pytest.raises(ValueError):launch.validate_launch(ROOT,m,g,requested=True,output=out)


@pytest.mark.parametrize('unit',launch.UNITS)
def test_cli_default_metadata_has_no_numerical_imports(unit,tmp_path):
    out=tmp_path/'metadata.json'
    result=subprocess.run([sys.executable,'-S',str(ROOT/launch.RUNNER),'--unit',unit,'--output',str(out)],
                          capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    m=json.loads(out.read_text())
    assert m['unit']==unit and m['assigned_resources'] is None and m['execution_plan_sealed'] is False
    assert m['science_authorized'] is False and m['status']=='H4_SUPPLEMENT_NOT_AUTHORIZED'
    assert 'DRAFT_NOT_AUTHORIZATION' in result.stdout


def test_watchdog_kill_preserves_last_attempt_snapshot(tmp_path):
    code="""import json,pathlib,time
p=pathlib.Path(__import__('sys').argv[1])
(p/'progress_0000.json').write_text(json.dumps({'mp_attempted':1,'mp_completed':0,'dps':80}))
time.sleep(10)
"""
    caps=dict(launch.plan('S4_MP')['caps_proposed']);caps['phase_wall_seconds']=dict.fromkeys(launch.PHASES,.15);caps['total_wall_seconds']=.5
    result=supervise([sys.executable,'-S','-c',code,str(tmp_path)],tmp_path,caps=caps,unit='S4_MP')
    assert result['status']=='H4_SUPPLEMENT_STOP' and result['worker_terminal'] is None
    assert result['latest_progress']=={'mp_attempted':1,'mp_completed':0,'dps':80}
    assert result['worker_exit_code']!=0 and result['mandatory_stop'] is True


@pytest.mark.parametrize('unit',launch.UNITS)
def test_watchdog_technical_success_requires_unit_specific_counts(unit,tmp_path):
    count,mp_count,event_count=(2,4,0) if unit=='S4_MP' else (0,0,4)
    terminal={'status':'H4_SUPPLEMENT_COMPLETE','unit':unit,'completed_correctness_cells':count,
       'compiled_wrappers':0,'completed_mp_records':mp_count,'completed_event_groups':event_count,
       'primitive_completed':537,'mandatory_stop':True,'next_stage_authorized':False,'N':None,'G':None,
       'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
       'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION'}
    code="""import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);terminal=json.loads(sys.argv[2])
for phase in ('input_reference','primitive_validation','validation'):
 (p/('phase_'+phase+'.json')).write_text(json.dumps({'phase':phase,'elapsed':0}))
(p/'worker_terminal.json').write_text(json.dumps(terminal))
"""
    caps=dict(launch.plan(unit)['caps_proposed']);caps['phase_wall_seconds']=dict.fromkeys(launch.PHASES,3);caps['total_wall_seconds']=5
    result=supervise([sys.executable,'-S','-c',code,str(tmp_path),json.dumps(terminal)],tmp_path,caps=caps,unit=unit)
    assert result['status']=='H4_SUPPLEMENT_COMPLETE' and result['next_stage_authorized'] is False
