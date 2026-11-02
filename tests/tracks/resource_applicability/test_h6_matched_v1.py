"""Preparation safety, toy arithmetic, injected compilation and dummy children.

Real molecular archives, sampling and transpilation are forbidden throughout.
"""
from copy import deepcopy
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.linalg import expm
from trottertracks.resource_applicability import h6_matched_contract_v1 as contract
from trottertracks.resource_applicability import h6_matched_accounting_v1 as accounting
from trottertracks.resource_applicability import h6_matched_execution_v1 as execution
from trottertracks.resource_applicability import h6_matched_port_v1 as portmod
from trottertracks.resource_applicability.ax2b_stage_validation_v2 import traced_signal
from trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 import recurrence_paths
from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    old=builtins.open;oldpath=Path.open
    def check(path):
        if isinstance(path,(str,Path)) and (str(path).endswith('.npz') or '/artifacts/' in str(path)):
            raise AssertionError('REAL_SCIENCE_FORBIDDEN:'+str(path))
    def guarded(path,*a,**k):check(path);return old(path,*a,**k)
    def guardedpath(path,*a,**k):check(path);return oldpath(path,*a,**k)
    def forbidden(*a,**k):raise AssertionError('REAL_SCIENCE_FORBIDDEN')
    monkeypatch.setattr(builtins,'open',guarded);monkeypatch.setattr(Path,'open',guardedpath)
    monkeypatch.setattr(np,'load',forbidden)
    import qiskit
    import trotterlib.rte as rte
    import trotterlib.rte_compiled_cost as costs
    monkeypatch.setattr(qiskit,'transpile',forbidden);monkeypatch.setattr(costs,'transpile',forbidden)
    monkeypatch.setattr(rte,'sample_rte_events',forbidden)


def test_plan_finite_fair_axes_and_no_time_caps():
    p=contract.plan();cs=p['cells']
    assert len(cs)==92 and len({c['id'] for c in cs})==92
    assert [sum(c['method']==m for c in cs) for m in ('B0','B1','B2')]==[12,8,72]
    assert {c['prefix'] for c in cs if c['method']=='B0'}=={c['prefix'] for c in cs if c['method']=='B2'}
    assert all(c['R']==c['q']*c['r'] for c in cs if c['R'])
    assert p['caps_proposed']['phase_wall_seconds'] is None and p['caps_proposed']['total_wall_seconds'] is None
    assert p['maximum_cost_groups']==12+8+72*8+8*32
    assert p['maximum_primary_wrappers']+p['maximum_sensitivity_wrappers']<=p['caps_proposed']['compile']


@pytest.mark.parametrize('method,formula,q,r,K',[('B0','2nd',2,None,None),('B1','2nd',4,None,None),
    ('B1','4th',2,None,None),('B2','2nd',2,4,2),('B2','2nd',4,2,4)])
def test_compact_preserves_frozen_arithmetic_and_forward_oracle(method,formula,q,r,K):
    psi=np.array([.6,.8j]);terms=[np.array([[.3,.1j],[-.1j,-.2]]),np.array([[.4,-.12],[-.12,.1]])]
    actions=[lambda v,t,h=h:expm(-1j*t*h)@v for h in terms]
    tail=np.array([[.1,.2],[.2,-.3]])
    c=dict(id='toy',method=method,formula=formula,prefix=1,q=q,r=r,R=q*r if r else None,K=K)
    kw=dict(cell=c,T=.8,scalar=.23,lambda_r=.9 if r else 0.)
    a=portmod.compact_signal(psi,actions,lambda v:tail@v,**kw,budget=ActionBudget(1000,10000))
    b=traced_signal(psi,actions,lambda v:tail@v,**kw,budget=ActionBudget(1000,10000))
    o=recurrence_paths(psi,actions,tail if r else None,**kw,maximum_tail=1000)
    assert a['signals']==b['signals'] and a['log_B']==b['log_B'] and a['b']==b['b']
    assert a['intermediate_norm_max']==b['intermediate_norm_max']
    for k in a['signals']:
        np.testing.assert_array_equal(a['vectors'][k],b['vectors'][k])
        np.testing.assert_allclose(a['vectors'][k],o['vectors'][k],atol=2e-14)


def test_negative_s4_times_and_nonrenormalization():
    from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
    c=dict(method='B1',prefix=1,q=2,formula='4th',R=None,K=None)
    assert any(t<0 for _,t in primitive_time_schedule(c,T=.8)['ordinary_one_outer_step'])
    with pytest.raises(ValueError,match='NORMALIZATION'):
        portmod.compact_signal(np.array([2.,0.]),[],None,cell=c,T=.8,scalar=0.,lambda_r=0.,budget=ActionBudget(0,0))


def signal_record(cell,plan,bias=0.,u=1e-10):
    s=dict(cell=cell,error_decomposition={'total_signed':{'real':bias,'imag':bias}},log_B=0.,
           allowance=dict(real=u,imag=u,log_B_margin=u))
    s['precision_rows']=accounting.precision_rows(s,plan)
    return s


def test_boundary_and_unknown_integer_guard():
    p=contract.plan();c=p['cells'][0]
    assert accounting.eligible(signal_record(c,p))
    s=signal_record(c,p,bias=.1)
    assert not accounting.eligible(s)
    p['epsilons']=[.001];s=signal_record(c,p,bias=.001/np.sqrt(2)-1e-9,u=9e-10)
    assert not accounting.eligible(s)
    with pytest.raises(ValueError,match='NONFINITE'):
        accounting.empirical_allowance(reference_error=float('nan'),path_errors=[],normalization_error=0.,
            raw_closure=0.,maximum_norm=1.,work=1,settings=p['numerical'])


def cost_rows(c, phase='exploration', n=8, base=100):
    return [dict(cell_id=c['id'],phase=phase,replica=i,seed=accounting.seed(c['id'],phase,i),
                 axis=a,control='symmetric_directional',metrics={'RZ':base+i,'CX':base+i,'depth':base+i,'size':base+i})
            for i in range(n) for a in ('cosine','sine')]


def test_independent_confirmation_seeds_and_paired_variance():
    p=contract.plan();cs=[c for c in p['cells'] if c['method']=='B2'][:3]
    signals=[signal_record(c,p) for c in cs]
    rows=[r for i,c in enumerate(cs) for r in cost_rows(c,base=100+100*i)]
    assert accounting.confirmation_ids(signals,rows,p)==sorted(c['id'] for c in cs[:2])
    explore=accounting.tasks(signals,p,'exploration')
    confirm=accounting.tasks(signals,p,'confirmation',[cs[0]['id']])
    assert len(explore)==24 and len(confirm)==32
    assert not {t['seed'] for t in explore}&{t['seed'] for t in confirm}
    assert accounting.seed(cs[0]['id'],'exploration',0)==accounting.seed(cs[0]['id'],'exploration',0)
    row=accounting.resource_row(signals[0],cost_rows(cs[0]),.05)
    assert row['axes']['cosine']['n']==8 and row['axes']['sine']['n']==8
    N=row['N_total']; expected=N*np.std(np.arange(8),ddof=1)/np.sqrt(8)
    assert row['G_standard_error']==pytest.approx(expected)
    broken=cost_rows(cs[0])[:-1]
    with pytest.raises(ValueError,match='PAIRED'):accounting.resource_row(signals[0],broken,.05)


def test_rare_order_diagnostics_do_not_bound_mean():
    d=accounting.order_diagnostic({'orders':[0,2],'order_probabilities':[.999,.001]},[0]*32,R=4,trajectories=8)
    assert d['all_zero_order_trajectory_probability']==.999**4
    assert d['probability_trajectory_contains_unseen_order']==pytest.approx(1-.999**4)
    assert d['cost_mean_tail_bound'] is None and not d['population_mean_certified']


def sealed_fixture(tmp_path,monkeypatch):
    cpus=sorted(os.sched_getaffinity(0))[:4]
    m=contract.preparation();out=tmp_path/'launch'
    m.update(source_commit='a'*40,source_hashes={},input_identity={'toy':True},environment={'toy':True},
             assigned_resources=contract.resources(cpus),exclusive_output=dict(repository_path=contract.NAMESPACE+'launch_v1',absolute_path=str(out)),execution_plan_sealed=True)
    g=dict(schema='h6_matched_authorization_v1',kind=contract.KIND,approved_by_user=True,
           source_commit=m['source_commit'],manifest_digest=digest(m),one_shot=True,retry=False,resume=False,
           assigned_cpus=cpus,exclusive_output=str(out))
    monkeypatch.setattr(contract,'verify_sources',lambda *a:None)
    monkeypatch.setattr(contract,'verify_parent',lambda *a:{'toy':True})
    monkeypatch.setattr(contract,'environment',lambda:{'toy':True})
    monkeypatch.setattr(contract,'safe_path',lambda *a:out)
    return m,g,out


@pytest.mark.parametrize('mutation',['none','unapproved','old_grant','plan','bind','source','retry','resume','one_shot','flags','resources','existing_output','environment','input'])
def test_launch_guard_before_scientific_imports(tmp_path,monkeypatch,mutation):
    m,g,out=sealed_fixture(tmp_path,monkeypatch)
    if mutation=='unapproved':g['approved_by_user']=False
    if mutation=='old_grant':g['schema']='track_a_h6_technical_pilot_authorization_v2'
    if mutation=='plan':m['plan']['cells'].pop();g['manifest_digest']=digest(m)
    if mutation=='bind':g['manifest_digest']='changed'
    if mutation=='source':g['source_commit']='b'*40
    if mutation in ('retry','resume'):g[mutation]=True
    if mutation=='one_shot':g['one_shot']=False
    if mutation=='flags':m['launch_allowed']=True;g['manifest_digest']=digest(m)
    if mutation=='resources':m['assigned_resources']['cost_workers']=4;g['manifest_digest']=digest(m)
    if mutation=='existing_output':out.mkdir()
    if mutation=='environment':monkeypatch.setattr(contract,'environment',lambda:{'changed':True})
    if mutation=='input':monkeypatch.setattr(contract,'verify_parent',lambda *a:{'changed':True})
    if mutation=='none':assert contract.validate_launch(tmp_path,m,g,out,requested=True)==m['assigned_resources']
    else:
        with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,out,requested=True)


def test_default_runner_no_scientific_imports_and_denied_execute(tmp_path):
    script=ROOT/contract.RUNNER;out=tmp_path/'metadata.json'
    code='import runpy,sys\nsys.argv='+repr([str(script),'--output',str(out)])+'\ntry:runpy.run_path('+repr(str(script))+',run_name="__main__")\nexcept SystemExit as e:assert e.code==0\nassert not any(n in sys.modules for n in ("numpy","scipy","numba","qiskit","openfermion"))'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert json.loads(out.read_text())['science_authorized'] is False
    target=tmp_path/'denied'
    r=subprocess.run([sys.executable,str(script),'--execute','--output',str(target)],capture_output=True,text=True)
    assert r.returncode!=0 and not target.exists()


def test_supervisor_two_disposable_children_without_wall_caps(tmp_path,monkeypatch):
    cpus=sorted(os.sched_getaffinity(0))[:4];m=contract.preparation();m['assigned_resources']=contract.resources(cpus)
    real=subprocess.Popen;started=[]
    def dummy(command,**kwargs):
        directory=Path(command[command.index('--worker')+1]);started.append(directory)
        code='import json,time;from pathlib import Path;p=Path('+repr(str(directory))+');time.sleep(.15);(p/"worker_terminal.json").write_text(json.dumps({"status":"TASK_COMPLETE"}))'
        return real([sys.executable,'-c',code],**kwargs)
    monkeypatch.setattr(execution.subprocess,'Popen',dummy)
    registrations=[(i,dict(phase='exploration',id=i)) for i in range(3)]
    paths,peak=execution.supervise_batch(ROOT,tmp_path,m,registrations)
    assert len(paths)==3 and len(started)==3 and peak>0
    bindings=[json.loads((p/'task_binding.json').read_text()) for p in started]
    assert bindings[0]['cpus']!=bindings[1]['cpus']
    assert not any(p.name.startswith('phase_wall') for p in tmp_path.iterdir())


def test_explicit_cancel_and_rejection_of_time_cap(tmp_path,monkeypatch):
    cpus=sorted(os.sched_getaffinity(0))[:4];m=contract.preparation();m['assigned_resources']=contract.resources(cpus)
    (tmp_path/'STOP_REQUEST').write_text('stop')
    with pytest.raises(RuntimeError,match='USER_STOP_REQUEST'):
        execution.supervise_batch(ROOT,tmp_path,m,[])
    m['plan']['caps_proposed']['total_wall_seconds']=1
    with pytest.raises(ValueError,match='NO_WALL_CAP'):
        execution.supervise_batch(ROOT,tmp_path,m,[])


def test_no_worker_claim_without_pinned_parent_grant(tmp_path):
    run=tmp_path/'run';run.mkdir();directory=run/'signal';directory.mkdir()
    m=contract.preparation()
    (run/'frozen_manifest.json').write_text(json.dumps(m))
    (run/'authorization.json').write_text(json.dumps({'approved_by_user':False}))
    (run/'launch_binding.json').write_text('{}')
    (directory/'task_binding.json').write_text('{}')
    with pytest.raises(ValueError,match='GRANT_BINDING'):
        execution.verify_worker(ROOT,directory,run)
    assert not (directory/'worker_claim.json').exists()


def test_actual_metadata_coverage_for_all_92_cells():
    from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import actual_bounds
    from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec
    def prep(prefix):
        one=DFDeterministicOneBodySpec('toy','one_body',None,None,'toy','toy',(),(0.,)*12,12,0,())
        return NS(deterministic_blocks=(one,)+tuple(NS(original_fragment_index=i,num_system_qubits=12,
            runtime_basis_operations=(),basis_hash='toy') for i in range(prefix)),
            rte_preparation=NS(component_specs=(NS(diagonal_pauli_support=(0,),basis_change_operations=()),)))
    p=contract.plan();bounds=actual_bounds(p['cells'],{c['id']:prep(c['prefix']) for c in p['cells']},T=.8)
    assert digest([dict(cell_id=r['cell_id'],schedule=r['schedule']) for r in bounds['cells']])==digest(p['coverage']['cells'])
    assert bounds['primitive_actions']==p['coverage']['primitive_actions']


def test_oracle_cache_reuses_matrices_without_changing_actions(tmp_path,monkeypatch):
    p=contract.preparation();writer=AtomicWriter(tmp_path,byte_cap=2**20)
    port=portmod.MatchedPort(tmp_path,p,writer,execution.Progress(writer))
    port.ham=NS(one_body=np.zeros((2,2)),lambdas=np.array([.2]),g_matrices=(np.eye(2),),constant=.1)
    port.sector=NS(basis_indices=np.arange(2));port.state=np.array([1.,0.],complex)
    port.oracle_tails={};port.truncated_references={}
    prep=NS(randomized_block_indices=(0,),exact_rte_lambda_r=.4,constant_coefficient=.1,extracted_identity_coefficient=0.)
    cs=[dict(id=str(K),method='B2',prefix=1,q=2,R=4,r=2,K=K,formula='2nd') for K in (2,4)]
    port.preps={c['id']:prep for c in cs};calls=[]
    monkeypatch.setattr(portmod,'occupation_df_matrix',lambda *a:(calls.append(True) or np.diag([.1,.2])))
    port.actions=lambda *a,**k:[lambda v,t:v,lambda v,t:v]
    for c in cs:
        out=port.oracle_signal(c)
        assert set(out['signals'])=={'corrected','raw','exact_tail'}
    assert len(calls)==1


@pytest.mark.parametrize('ordinary,validate,fail',[(False,False,False),(True,True,False),(False,False,True)])
def test_cost_task_pairing_primary_scope_and_compile_stop(tmp_path,monkeypatch,ordinary,validate,fail):
    m=contract.preparation();m['input_identity']=dict(snapshot_path='toy',snapshot_receipt_path='a.json',df_receipt_path='b.json')
    for name in ('a.json','b.json'):(tmp_path/name).write_text('{}')
    state=np.array([1.,0.],complex);ham=NS(n_qubits=1)
    monkeypatch.setattr(portmod,'load_h6_snapshot',lambda *a:(ham,NS(),state,state,{},None))
    monkeypatch.setattr(portmod,'checked_basis_bridge',lambda *a:(np.array([0,1]),state))
    monkeypatch.setattr(portmod,'_prepare_discard',lambda *a:NS(deterministic_blocks=(),constant_coefficient=0.))
    monkeypatch.setattr(portmod,'actual_bounds',lambda *a,**k:dict(cells=[dict(wrapper_instruction_upper_bounds={'ordinary':1,'symmetric_directional':1})]))
    builds=[];compiles=[]
    monkeypatch.setattr(portmod,'build_deterministic_native',lambda *a,**k:(builds.append(k['control_policy']) or NS(circuit='toy')))
    monkeypatch.setattr(portmod,'build_native_hadamard_wrapper',lambda *a,**k:k['axis'])
    monkeypatch.setattr(portmod,'simulate_statevector',lambda circuit,v:v)
    monkeypatch.setattr(portmod,'hadamard_expectation',lambda v,n:1. if len(compiles)==0 else 0.)
    # Wrapper expectation injection depends on axis, not on real circuit simulation.
    def simulate(circuit,v):
        if circuit=='cosine':return np.array([1.,0.,0.,0.],complex)
        if circuit=='sine':return np.array([2**-.5,0.,2**-.5,0.],complex)
        return v
    monkeypatch.setattr(portmod,'simulate_statevector',simulate)
    monkeypatch.setattr(portmod,'hadamard_expectation',lambda v,n:float(abs(v[0])**2-abs(v[2])**2))
    monkeypatch.setattr(portmod,'canonical_qiskit_circuit_fingerprint',lambda x:'toy_'+x)
    def compile_cost(wrapper,*a,**k):
        compiles.append(wrapper)
        if fail:raise RuntimeError('INJECTED_COMPILE_STOP')
        return NS(transpiled_circuit=NS(data=[]),rz_count=3,rz_depth=2,cx_count=4,cx_depth=3,
            total_depth=8,circuit_size=9,actual_circuit_fingerprint='toy_'+wrapper,compiler_settings_hash='toy')
    monkeypatch.setattr(portmod,'transpile_and_measure_cost',compile_cost)
    c=m['plan']['cells'][0];task=dict(cell=c,cell_id=c['id'],phase='exploration',replica=0,seed=1,
                                    ordinary=ordinary,validate=validate,expected_signal={'real':1.,'imag':0.})
    writer=AtomicWriter(tmp_path,byte_cap=2**20)
    if fail:
        with pytest.raises(RuntimeError,match='INJECTED_COMPILE'):portmod.cost_task(tmp_path,m,task,writer,execution.Progress(writer))
        assert not (tmp_path/'cost_result.json').exists()
        assert len(compiles)==1
    else:
        portmod.cost_task(tmp_path,m,task,writer,execution.Progress(writer))
        rows=json.loads((tmp_path/'cost_result.json').read_text())['rows']
        assert len(rows)==(4 if ordinary else 2)
        assert len({r['event_digest'] for r in rows})==1
        assert builds==(['symmetric_directional','ordinary'] if ordinary else ['symmetric_directional'])


@pytest.mark.parametrize('fail',[False,True])
def test_injected_end_to_end_selection_accounting_stop_and_saved_audit(tmp_path,monkeypatch,fail):
    m=contract.preparation();m['source_commit']='a'*40;m['source_hashes']={};m['input_identity']={'toy':True};m['environment']={'toy':True}
    m['assigned_resources']=contract.resources(sorted(os.sched_getaffinity(0))[:4])
    m['plan']['cells']=[next(c for c in m['plan']['cells'] if c['method']==method) for method in ('B0','B1','B2')]
    g=dict(schema='h6_matched_authorization_v1',kind=m['kind'],approved_by_user=True,one_shot=True,
           retry=False,resume=False,source_commit=m['source_commit'],manifest_digest=digest(m))
    for name,obj in [('frozen_manifest.json',m),('authorization.json',g),('authorization_source.json',g)]:
        (tmp_path/name).write_text(json.dumps(obj))
    (tmp_path/'launch_binding.json').write_text(json.dumps(dict(manifest_digest=digest(m),authorization_digest=digest(g),
        authorization_source_sha256=contract.file_hash(tmp_path/'authorization_source.json'))))
    monkeypatch.setattr(execution,'install_limits',lambda *a:None)
    monkeypatch.setattr(execution,'verify_sources',lambda *a:None)
    monkeypatch.setattr(execution,'verify_parent',lambda *a:{'toy':True})
    monkeypatch.setattr(execution,'environment',lambda:{'toy':True})
    def injected(root,out,manifest,registrations,signal_stage=False,progress=None):
        paths=[]
        if signal_stage:
            path=out/'signal';path.mkdir();w=AtomicWriter(path,byte_cap=2**20)
            w.write('signal_summary.json',dict(signals=[dict(signal_record(c,manifest['plan']),signals={'corrected':{'real':1.,'imag':0.}}) for c in manifest['plan']['cells']],
                calls_attempted={k:0 for k in ('primitive','control_probe','compile','trajectory','occurrence','reference_matvec')}))
            return [path],100
        for i,t in registrations:
            path=out/f'cost_{t["phase"]}_{i:04d}';path.mkdir();w=AtomicWriter(path,byte_cap=2**20)
            if fail:raise RuntimeError('INJECTED_WORKER_STOP')
            rows=cost_rows(t['cell'],t['phase'],n=1,base=100)
            for row in rows:row.update(replica=t['replica'],seed=t['seed'],event_digest=digest(None))
            w.write('cost_result.json',dict(task=t,rows=rows,calls_attempted={k:0 for k in ('primitive','control_probe','compile','trajectory','occurrence','reference_matvec')}))
            w.write('trajectory.json',dict(task=t,events=None,event_digest=digest(None),finite_distribution={'orders':[0,2],'order_probabilities':[.9,.1]} if t['cell']['R'] else None,observed_orders=[0]))
            w.write('worker_terminal.json',dict(status='TASK_COMPLETE'));paths.append(path)
        return paths,100
    monkeypatch.setattr(execution,'supervise_batch',injected)
    w=AtomicWriter(tmp_path,byte_cap=2**30)
    rc=execution.orchestrate(ROOT,tmp_path,m,w)
    terminal=json.loads((tmp_path/'execution_terminal.json').read_text())
    assert terminal['mandatory_stop'] and not terminal['next_stage_authorized'] and terminal['H6_status']=='H6_NOT_AUTHORIZED'
    assert rc==(1 if fail else 0)
    if fail:
        assert 'INJECTED_WORKER_STOP' in terminal['reason']
        assert not (tmp_path/'confirmation_tasks.json').exists()
    else:
        assert len(json.loads((tmp_path/'confirmation_tasks.json').read_text())['tasks'])==32
        assert json.loads((tmp_path/'confirmation_resource_map.json').read_text())['comparisons']
        import runpy
        module=runpy.run_path(str(ROOT/contract.AUDITOR))
        monkeypatch.setitem(module['audit'].__globals__,'verify_sources',lambda *a:None)
        monkeypatch.setitem(module['audit'].__globals__,'verify_parent',lambda *a:{'toy':True})
        assert module['audit'](ROOT,tmp_path)['status']=='STATIC_IDENTITY_AND_COVERAGE_PASS'
        path=tmp_path/'cost_exploration_0000/cost_result.json'
        path.write_text('{}')
        with pytest.raises(ValueError,match='AUDIT_BYTES'):module['audit'](ROOT,tmp_path)


def test_failed_dummy_child_stops_without_retry(tmp_path,monkeypatch):
    m=contract.preparation();m['assigned_resources']=contract.resources(sorted(os.sched_getaffinity(0))[:4])
    real=subprocess.Popen;launches=[]
    def dummy(command,**kwargs):
        launches.append(command)
        return real([sys.executable,'-c','raise SystemExit(7)'],**kwargs)
    monkeypatch.setattr(execution.subprocess,'Popen',dummy)
    with pytest.raises(RuntimeError,match='WORKER_EXIT:7'):
        execution.supervise_batch(ROOT,tmp_path,m,[(i,dict(phase='exploration',id=i)) for i in range(8)])
    assert len(launches)==2 and len(list(tmp_path.glob('cost_exploration_*')))==2
