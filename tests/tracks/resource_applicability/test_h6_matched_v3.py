"""Synthetic circuits/metadata/processes only. No molecular inputs or science."""
import builtins
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref
import pytest
from trottertracks.resource_applicability import h6_matched_contract_v3 as contract
from trottertracks.resource_applicability import h6_matched_reuse_v3 as reuse
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.h6_matched_accounting_v1 import seed

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def forbid_real_inputs(monkeypatch):
    old,oldpath=builtins.open,Path.open
    def check(p):
        if isinstance(p,(str,Path)) and (str(p).endswith('.npz') or '/artifacts/' in str(p)):
            raise AssertionError('REAL_SCIENCE_FORBIDDEN:'+str(p))
    def guarded(p,*a,**k):check(p);return old(p,*a,**k)
    def guardedpath(p,*a,**k):check(p);return oldpath(p,*a,**k)
    monkeypatch.setattr(builtins,'open',guarded);monkeypatch.setattr(Path,'open',guardedpath)


def toy_evolution(q=2,formula='4th',policy='symmetric_directional'):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec,DFDeterministicFragmentSpec
    from trottertracks.resource_applicability.ax2b_native_df_v5 import build_deterministic_native
    basis=QuantumCircuit(2);basis.p(.17,0);basis.append(XXPlusYYGate(.41,-.31),[0,1])
    ops=tuple((i.operation,tuple(basis.find_bit(x).index for x in i.qubits)) for i in basis.data)
    one=DFDeterministicOneBodySpec('toy_one','one_body',None,None,'toy_U','toy_h',(),(.3,-.17),2,0,ops)
    frag=DFDeterministicFragmentSpec('toy_df','df_fragment',0,0,'toy_U1','toy_h1',(),(.7,-.4),-.37,2,1,ops)
    return build_deterministic_native((one,frag),num_system_qubits=2,T=.8,q=q,formula=formula,
        scalar=.23,control_policy=policy,max_instructions=10000)


@pytest.mark.parametrize('axis',['cosine','sine'])
@pytest.mark.parametrize('measurement',[False,True])
@pytest.mark.parametrize('formula',['2nd','4th'])
@pytest.mark.parametrize('policy',['ordinary','symmetric_directional'])
def test_consumed_wrapper_has_exact_legacy_numeric_bytes(axis,measurement,formula,policy):
    from trottertracks.resource_applicability.ax2b_native_df_v5 import build_native_hadamard_wrapper
    from trottertracks.resource_applicability.h6_matched_memory_v3 import take_wrapper
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import write_canonical_bytes
    from qiskit.quantum_info import Operator
    evolution=toy_evolution(formula=formula,policy=policy)
    old=build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=measurement,max_instructions=10000)
    owner=[evolution];original_circuit=evolution.circuit
    new=take_wrapper(owner,axis=axis,include_measurement=measurement,max_instructions=10000)
    assert not owner and new is original_circuit
    a,b=bytearray(),bytearray();write_canonical_bytes(old,a.extend);write_canonical_bytes(new,b.extend)
    assert a==b
    assert old.count_ops()==new.count_ops()
    if not measurement:
        import numpy as np
        assert np.linalg.norm(Operator(old).data-Operator(new).data)<1e-12


@pytest.mark.parametrize('axis',['cosine','sine'])
def test_actual_toy_compile_metrics_settings_and_input_lifetime(tmp_path,monkeypatch,axis):
    import qiskit
    from trotterlib.rte import CompilerSettings
    from trotterlib import rte_compiled_cost as old
    from trottertracks.resource_applicability.ax2b_native_df_v5 import build_native_hadamard_wrapper
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fp
    from trottertracks.resource_applicability.h6_matched_memory_v3 import compile_owned
    from trottertracks.resource_applicability.ax2b_diagnostics_v4 import ResourceTrace
    compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
    legacy=build_native_hadamard_wrapper(toy_evolution(q=8),axis=axis,max_instructions=10000)
    expected=old.transpile_and_measure_cost(legacy,compiler,actual_circuit_fingerprint=fp(legacy))
    owner=[toy_evolution(q=8)];reference=weakref.ref(owner[0].circuit)
    depth=old._filtered_depth
    def checked_depth(*a,**k):
        gc.collect();assert reference() is None  # Input released before metric traversal.
        return depth(*a,**k)
    monkeypatch.setattr(old,'_filtered_depth',checked_depth)
    records={};trace=ResourceTrace(lambda n,v:records.setdefault(n,v),lambda:0.)
    actual=compile_owned(owner,compiler,axis=axis,max_instructions=10000,transpiled_cap=50000,trace=trace)
    for field in ('pretranspile_gate_counts','posttranspile_gate_counts','rz_count','cx_count','rz_depth',
        'cx_depth','total_depth','circuit_size','qubit_count','global_phase','compiler_settings_hash','actual_circuit_fingerprint'):
        assert getattr(actual,field)==getattr(expected,field)
    assert not owner
    stages={v['context']['stage'] for v in records.values()}
    assert {'wrapper_build','wrapper_fingerprint','transpile','metric_measurement'}<=stages
    assert all('VmSize_bytes' in v['memory'] and 'VmRSS_bytes' in v['memory'] for v in records.values())


def fixture_group(tmp_path,ordinary=False):
    t=dict(cell=dict(method='B1'),cell_id='SYNTHETIC',phase='exploration',replica=0,
        seed=seed('SYNTHETIC','exploration',0),ordinary=ordinary,validate=True)
    rows=[]
    for p in ['symmetric_directional','ordinary'] if ordinary else ['symmetric_directional']:
        for a in ('cosine','sine'):
            rows.append(dict(cell_id=t['cell_id'],phase=t['phase'],replica=0,seed=t['seed'],axis=a,control=p,
                event_digest=digest(None),compiler_hash=reuse.COMPILER_HASH,fingerprint='a'*64,
                metrics={k:1 for k in ('RZ','CX','RZ_depth','CX_depth','depth','size')}))
    objects={'worker_terminal.json':dict(status='TASK_COMPLETE',reason=None,task=t,source_commit=reuse.OLD_SOURCE),
        'worker_claim.json':dict(source_commit=reuse.OLD_SOURCE,task_digest=digest(t),resource_limits={'rlimit_as':[8*2**30]*2}),
        'task_binding.json':dict(task=t),
        'trajectory.json':dict(task=t,event_digest=digest(None),events=None,finite_distribution=None,observed_orders=[]),
        'cost_result.json':dict(task=t,rows=rows,calls_attempted={'compile':len(rows)})}
    for row in rows:objects[f'wrapper_{row["control"]}_{row["axis"]}.json']=row
    for p in ['symmetric_directional','ordinary'] if ordinary else ['symmetric_directional']:
        objects['validation_'+p+'.json']={'max_error':1e-14}
    for n,obj in objects.items():(tmp_path/n).write_text(json.dumps(obj))
    return t,objects


@pytest.mark.parametrize('tamper',['valid','seed_float','wrong_seed','one_axis','duplicate_axis','compiler',
    'event','failed_terminal','task','source','missing_validation','wrapper_bytes','limit'])
def test_complete_group_admission_rejects_misbound_or_partial_metadata(tmp_path,tamper):
    t,obj=fixture_group(tmp_path,ordinary=True)
    if tamper=='seed_float':t['seed']=float(t['seed'])
    if tamper=='wrong_seed':t['seed']+=1
    if tamper=='one_axis':obj['cost_result.json']['rows'].pop()
    if tamper=='duplicate_axis':obj['cost_result.json']['rows'].append(obj['cost_result.json']['rows'][0])
    if tamper=='compiler':obj['cost_result.json']['rows'][0]['compiler_hash']='b'*64
    if tamper=='event':obj['trajectory.json']['event_digest']='b'*64
    if tamper=='failed_terminal':obj['worker_terminal.json']['status']='TASK_STOP'
    if tamper=='task':obj['task_binding.json']={'task':{}}
    if tamper=='source':obj['worker_claim.json']['source_commit']='b'*40
    if tamper=='missing_validation':(tmp_path/'validation_ordinary.json').unlink()
    if tamper=='wrapper_bytes':obj['wrapper_ordinary_sine.json']=dict(axis='wrong')
    if tamper=='limit':obj['worker_claim.json']['resource_limits']['rlimit_as']=[2*2**30]*2
    for n,v in obj.items():
        if tamper!='missing_validation' or n!='validation_ordinary.json':(tmp_path/n).write_text(json.dumps(v))
    if tamper=='valid':assert len(reuse.admit_group(tmp_path,t)['rows'])==4
    else:
        with pytest.raises((ValueError,FileNotFoundError)):reuse.admit_group(tmp_path,t)


def test_exact_exploration_gate_before_partial_selection(tmp_path):
    task,obj=fixture_group(tmp_path);result=obj['cost_result.json']
    registered=[task,{**task,'replica':1}]
    with pytest.raises(ValueError,match='NOT_COMPLETE'):reuse.complete_exploration(registered,{0:result})
    with pytest.raises(ValueError,match='MISMATCH'):reuse.complete_exploration(registered,{0:result,1:result})
    with pytest.raises(ValueError,match='NOT_COMPLETE'):reuse.complete_exploration(registered,{0:result,1:result,2:result})
    assert len(reuse.complete_exploration([task],{0:result}))==2


@pytest.mark.parametrize('tamper',['valid','old_grant','unapproved','retry','resume','reused_digest','cpu','source','H8'])
def test_new_grant_required_for_exact_new_scope(tamper):
    m=contract.preparation();m.update(source_commit='a'*40,reuse_registry={'old':'bytes'},assigned_resources={'assigned_cpus':[0,2,5,6]})
    g=dict(schema='h6_matched_authorization_v3',kind=contract.KIND,approved_by_user=True,authorization_only=True,
        source_commit=m['source_commit'],manifest_digest=digest(m),reuse_registry_digest=digest(m['reuse_registry']),
        assigned_cpus=[0,2,5,6],one_shot=True,retry=False,resume=False,exclusive_output='/tmp/new_only',H8_authorized=False)
    if tamper=='old_grant':g['schema']='h6_matched_authorization_v2'
    if tamper=='unapproved':g['approved_by_user']=False
    if tamper=='retry':g['retry']=True
    if tamper=='resume':g['resume']=True
    if tamper=='reused_digest':g['reuse_registry_digest']='changed'
    if tamper=='cpu':g['assigned_cpus']=[0,1,2,3]
    if tamper=='source':g['source_commit']='b'*40
    if tamper=='H8':g['H8_authorized']=True
    if tamper=='valid':contract.validate_grant(m,g,Path('/tmp/new_only'))
    else:
        with pytest.raises(ValueError):contract.validate_grant(m,g,Path('/tmp/new_only'))


def test_unapproved_runner_has_no_heavy_import_or_output_claim(tmp_path):
    script=ROOT/contract.RUNNER
    code='import runpy,sys\nsys.argv='+repr([str(script),'--output',str(tmp_path/'preparation.json')])+'\ntry:runpy.run_path('+repr(str(script))+',run_name="__main__")\nexcept SystemExit as e:assert e.code==0\nassert not any(x in sys.modules for x in ("numpy","scipy","qiskit","numba","openfermion"))'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr
    m=json.loads((tmp_path/'preparation.json').read_text());assert not m['science_authorized']
    out=tmp_path/'denied'
    r=subprocess.run([sys.executable,str(script),'--execute','--output',str(out)],capture_output=True,text=True,timeout=30)
    assert r.returncode!=0 and not out.exists()


def test_memoryerror_trace_releases_frames_and_keeps_stage(tmp_path):
    from trottertracks.resource_applicability.ax2b_diagnostics_v4 import ResourceTrace
    records={};trace=ResourceTrace(lambda n,v:records.setdefault(n,v),lambda:0.)
    try:
        with trace.stage('transpile',instructions=7):raise MemoryError('synthetic only')
    except MemoryError as exc:
        trace.failure(exc);assert exc.__traceback__ is None
    f=records['failure_diagnostics.json']
    assert f['context']['stage']=='transpile' and f['traceback_frames'] and not f['locals_recorded']


@pytest.mark.parametrize('phase',['exploration','confirmation'])
def test_real_disposable_children_preserve_limits_and_original_indices(tmp_path,phase):
    root=tmp_path/'synthetic_root';root.mkdir();out=tmp_path/'output';out.mkdir()
    for n in contract.source_paths(root):
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic source\n')
    for n in ('snapshot.bin','snapshot_receipt.json','df_receipt.json'):(root/n).write_text('{}')
    m=contract.preparation();cpus=sorted(os.sched_getaffinity(0))[:4]
    task=dict(phase=phase,cell_id='SYNTHETIC',seed=seed('SYNTHETIC',phase,0),replica=0)
    index=447 if phase=='exploration' else 0
    m.update(source_commit='a'*40,source_hashes={n:contract.file_hash(root/n) for n in contract.source_paths(root)},
        assigned_resources=contract.resources(cpus),environment=contract.environment(),execution_plan_sealed=True,
        reuse_registry={'missing':[{'index':index,'task':task}]},
        input_identity={key+suffix:v for key,n in [('snapshot','snapshot.bin'),('snapshot_receipt','snapshot_receipt.json'),('df_receipt','df_receipt.json')]
            for suffix,v in [('_path',n),('_sha256',contract.file_hash(root/n))]})
    g=dict(schema='h6_matched_authorization_v3',kind=contract.KIND,authorization_only=True,approved_by_user=True,
        manifest_digest=digest(m),source_commit=m['source_commit'],reuse_registry_digest=digest(m['reuse_registry']),
        assigned_cpus=cpus,one_shot=True,retry=False,resume=False,exclusive_output=str(out),H8_authorized=False)
    for n,v in [('frozen_manifest.json',m),('authorization.json',g),('authorization_source.json',g),
        (phase+'_tasks.json',{'tasks':[{'skipped':i} for i in range(index)]+[task]})]:
        (out/n).write_text(json.dumps(v))
    (out/'launch_binding.json').write_text(json.dumps(dict(manifest_digest=digest(m),authorization_digest=digest(g),
        authorization_source_sha256=contract.file_hash(out/'authorization_source.json'))))
    child=r'''
import sys,json,mmap,resource,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability import h6_matched_execution_v3 as e
root,directory,out=map(Path,sys.argv[2:])
m,t,w=e.verify_worker(root,directory,out)
assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','numba','openfermion'))
with mmap.mmap(-1,2*2**30) as probe:assert len(probe)==2*2**30
w.write('synthetic_limits.json',dict(limits=resource.getrlimit(resource.RLIMIT_AS),cpus=sorted(os.sched_getaffinity(0)),synthetic_only=True))
w.write('worker_terminal.json',dict(status='TASK_COMPLETE',synthetic_only=True))
'''
    outer=r'''
import sys,json,subprocess,resource
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability import h6_matched_execution_v3 as e
root,out=map(Path,sys.argv[2:4]);phase=sys.argv[4];index=int(sys.argv[5]);child=sys.argv[6]
m=json.loads((out/'frozen_manifest.json').read_text());e.install_coordinator_limits(m['assigned_resources'],2*2**30)
real=e.subprocess.Popen
def launch(command,**kw):
 d=command[command.index('--worker')+1]
 return real([sys.executable,'-c',child,sys.argv[1],str(root),d,str(out)],**kw)
e.subprocess.Popen=launch
t=json.loads((out/(phase+'_tasks.json')).read_text())['tasks'][index]
paths,peak=e.supervise_batch(root,out,m,[(index,t)])
assert resource.getrlimit(resource.RLIMIT_AS)==(2*2**30,8*2**30)
print(json.dumps({'paths':[str(p) for p in paths],'peak':peak}))
'''
    r=subprocess.run([sys.executable,'-c',outer,str(ROOT/'src'),str(root),str(out),phase,str(index),child],
        capture_output=True,text=True,timeout=30,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert r.returncode==0,r.stderr+r.stdout
    data=json.loads(r.stdout);assert data['peak']>0
    path=Path(data['paths'][0]);assert path.name==f'cost_{phase}_{index:04d}'
    claim=json.loads((path/'worker_claim.json').read_text())
    assert claim['resource_limits']['rlimit_as']==[8*2**30]*2
    assert claim['resource_limits']['inherited_rlimit_as']==[2*2**30,8*2**30]
    assert json.loads((path/'synthetic_limits.json').read_text())['limits']==[8*2**30]*2


@pytest.mark.parametrize('failure',['none','first','partial'])
def test_injected_complete_supplement_or_stop_before_selection(tmp_path,monkeypatch,failure):
    """Exercise actual integration and original selection with synthetic records."""
    from trottertracks.resource_applicability import h6_matched_execution_v3 as e
    from trottertracks.resource_applicability.h6_matched_accounting_v1 import precision_rows,tasks
    from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
    root=tmp_path/'root';root.mkdir();out=root/'new_run';out.mkdir()
    m=contract.preparation();p=m['plan'];signals=[]
    for c in p['cells']:
        s=dict(cell=c,log_B=0.,error_decomposition={'total_signed':{'real':0.,'imag':0.}},
            allowance={'real':1e-10,'imag':1e-10,'log_B_margin':1e-10},signals={'corrected':{'real':1.,'imag':0.}})
        s['precision_rows']=precision_rows(s,p)
        if c['method']=='B0' and c['prefix']==5:
            for v in s['precision_rows'].values():
                for a in v.values():a['cost_acquisition_eligible']=False
        signals.append(s)
    registered=tasks(signals,p,'exploration')
    for t in registered:
        if t['cell']['method']!='B2':t['expected_signal']={'real':1.,'imag':0.}
    assert len(registered)==592 and registered[447]['cell_id']=='B1_p19_q8_4th'
    zero={k:0 for k in ('compile','control_probe','trajectory','occurrence','primitive','reference_matvec')}
    legacy=root/'legacy'/'signal';legacy.mkdir(parents=True)
    (legacy/'signal_summary.json').write_text(json.dumps({'signals':signals}))
    monkeypatch.setattr(e,'OLD_RAW','legacy')
    def save(path,t):
        path.mkdir();w=AtomicWriter(path,byte_cap=2**20)
        rows=[dict(cell_id=t['cell_id'],phase=t['phase'],replica=t['replica'],seed=t['seed'],axis=a,control=control,
            metrics={k:100+t['replica'] for k in ('RZ','CX','RZ_depth','CX_depth','depth','size')})
            for control in (['symmetric_directional','ordinary'] if t['ordinary'] else ['symmetric_directional']) for a in ('cosine','sine')]
        calls=dict(zero,compile=len(rows));w.write('cost_result.json',dict(task=t,rows=rows,calls_attempted=calls))
        w.write('worker_terminal.json',dict(status='TASK_COMPLETE',synthetic_only=True))
        w.write('trajectory.json',dict(task=t,finite_distribution={'orders':[0,2],'order_probabilities':[.999,.001]} if t['cell']['R'] else None,
            observed_orders=[0]*(t['cell']['R'] or 0)))
    accepted=[];missing=[]
    for i,t in enumerate(registered):
        if i in range(447) or i in (448,449):
            directory=f'old_{i:04d}';save(root/directory,t);accepted.append({'index':i,'directory':directory})
        else:missing.append({'index':i,'task':t})
    registry=dict(source_commit=reuse.OLD_SOURCE,result_commit=reuse.OLD_RESULT,original_tasks=registered,
        accepted=accepted,missing=missing,missing_first_index=447,original_signal_calls=zero,original_raw_path='legacy')
    m.update(source_commit='a'*40,source_hashes={},input_identity={'synthetic':True},environment={},reuse_registry=registry,execution_plan_sealed=True,
        assigned_resources=contract.resources(sorted(os.sched_getaffinity(0))[:4]))
    monkeypatch.setattr(e,'admit_group',lambda path,*a,**k:json.loads((path/'cost_result.json').read_text()))
    monkeypatch.setattr(e,'install_coordinator_limits',lambda *a:{'rlimit_as':[2*2**30,8*2**30]})
    monkeypatch.setattr(e,'verify_sources',lambda *a:None);monkeypatch.setattr(e,'verify_evidence',lambda *a:None)
    monkeypatch.setattr(e,'verify_parent',lambda *a:{'synthetic':True});monkeypatch.setattr(e,'environment',lambda:{})
    batches=[]
    def injected(root,out,m,registrations,progress=None):
        batches.append(registrations)
        assert registrations and all(t['phase']!='signal' for i,t in registrations)
        if failure=='first':raise RuntimeError('INJECTED_MEMORY_STOP')
        paths=[]
        for i,t in registrations:
            path=out/f'cost_{t["phase"]}_{i:04d}';save(path,t);paths.append(path)
        if failure=='partial' and len(registrations)>1:paths.pop()
        return list(reversed(paths)),100
    monkeypatch.setattr(e,'supervise_batch',injected)
    g=dict(schema='h6_matched_authorization_v3',kind=m['kind'],approved_by_user=True,authorization_only=True,
        source_commit=m['source_commit'],manifest_digest=digest(m),reuse_registry_digest=digest(registry),
        assigned_cpus=m['assigned_resources']['assigned_cpus'],one_shot=True,retry=False,resume=False,exclusive_output=str(out),H8_authorized=False)
    for n,v in [('frozen_manifest.json',m),('authorization.json',g),('authorization_source.json',g)]:
        (out/n).write_text(json.dumps(v))
    (out/'launch_binding.json').write_text(json.dumps(dict(manifest_digest=digest(m),authorization_digest=digest(g),
        authorization_source_sha256=contract.file_hash(out/'authorization_source.json'))))
    status=e.orchestrate(root,out,m,AtomicWriter(out,byte_cap=2**30))
    term=json.loads((out/'execution_terminal.json').read_text())
    assert len(batches[0])==1 and batches[0][0][0]==447
    assert term['mandatory_stop'] and not term['next_stage_authorized']
    if failure!='none':
        assert status==1 and not (out/'confirmation_tasks.json').exists()
        assert not (out/'exploration_resource_map.json').exists()
    else:
        assert status==0,term
        selection=json.loads((out/'confirmation_tasks.json').read_text())
        assert len(selection['selected'])==2 and len(selection['tasks'])==64
        assert not {t['seed'] for t in registered}&{t['seed'] for t in selection['tasks']}
        assert len(batches[1])==142 and len(batches[2])==64
        assert json.loads((out/'cost_coverage.json').read_text())['wrapper_count']==1316
        assert json.loads((out/'aggregate_calls.json').read_text())['old_aborted_attempts'] is None
        assert (out/'cost_order_diagnostics.json').exists()
    import runpy
    module=runpy.run_path(str(ROOT/contract.AUDITOR));namespace=module['audit'].__globals__
    for name in ('verify_sources','verify_evidence'):monkeypatch.setitem(namespace,name,lambda *a:None)
    monkeypatch.setitem(namespace,'verify_parent',lambda *a:{'synthetic':True})
    monkeypatch.setitem(namespace,'environment',lambda:{})
    monkeypatch.setitem(namespace,'admit_group',lambda path,*a,**k:json.loads((path/'cost_result.json').read_text()))
    result=module['audit'](root,out)
    assert result['full_registered_coverage_verified']==(failure=='none')
    assert not result['scientific_GO_issued']


@pytest.mark.parametrize('scenario',['deterministic','random','ordinary','partial_digest_bad','compile_failure'])
def test_actual_cost_port_reuses_steps_once_and_never_signal_setup(tmp_path,monkeypatch,scenario):
    import numpy as np
    from types import SimpleNamespace as NS
    from trottertracks.resource_applicability import h6_matched_cost_port_v3 as port
    from trottertracks.resource_applicability.ax2b_diagnostics_v4 import ResourceTrace
    from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
    from trottertracks.resource_applicability.h6_matched_execution_v3 import Progress
    m=contract.preparation();m['input_identity']=dict(snapshot_path='synthetic.bin',snapshot_receipt_path='a.json',df_receipt_path='b.json')
    for n in ('a.json','b.json'):(tmp_path/n).write_text('{}')
    random=scenario!='deterministic'
    c=next(c for c in m['plan']['cells'] if c['method']==('B2' if random else 'B1') and c['q']==2 and
        (not random or c['prefix']==10 and c['r']==1 and c['K']==2))
    task=dict(cell=c,cell_id=c['id'],phase='exploration',replica=0,seed=seed(c['id'],'exploration',0),
        ordinary=scenario=='ordinary',validate=True,expected_signal={'real':1.,'imag':0.})
    event=NS(taylor_order=0,to_dict=lambda:{'taylor_order':0})
    steps=[NS(rte_occurrence=NS(events=[event])) for _ in range(2)]
    events=[[event.to_dict()] for _ in range(2)] if random else None
    m['reuse_registry']={'missing':[dict(task=task,partial_event_digest='bad' if scenario=='partial_digest_bad' else digest(events))]}
    state=np.array([1.,0.],complex);ham=NS(n_qubits=1)
    monkeypatch.setattr(port,'load_h6_snapshot',lambda *a:(ham,NS(),state,state,{},None))
    monkeypatch.setattr(port,'checked_basis_bridge',lambda *a:(np.array([0,1]),state))
    prep=NS(deterministic_blocks=(),constant_coefficient=0.,exact_rte_lambda_r=.1)
    monkeypatch.setattr(port,'_prepare',lambda *a:prep)
    monkeypatch.setattr(port,'actual_bounds',lambda *a,**k:dict(cells=[dict(wrapper_instruction_upper_bounds={'symmetric_directional':1})]))
    def forbidden(*a,**k):raise AssertionError('SIGNAL_SETUP_FORBIDDEN')
    monkeypatch.setattr(port.MatchedPort,'setup',forbidden)
    sampled=[];builds=[];compiles=[]
    def sample(self,cell,s):
        sampled.append((cell,s));return steps
    monkeypatch.setattr(port.MatchedPort,'sampled_steps',sample)
    def make(*a,**k):
        if random:assert a[0] is steps
        builds.append(k['control_policy']);return NS(circuit=NS(data=[]))
    monkeypatch.setattr(port,'partial_native_from_step_requests',make)
    monkeypatch.setattr(port,'build_deterministic_native',make)
    monkeypatch.setattr(port,'build_native_hadamard_wrapper',lambda *a,**k:k['axis'])
    monkeypatch.setattr(port,'simulate_statevector',lambda circuit,v:(np.array([1.,0.,0.,0.],complex) if circuit=='cosine' else
        np.array([2**-.5,0.,2**-.5,0.],complex) if circuit=='sine' else v))
    monkeypatch.setattr(port,'hadamard_expectation',lambda v,n:float(abs(v[0])**2-abs(v[2])**2))
    def compile(owner,compiler,**k):
        owner.pop();k['before_compile']();compiles.append(k['axis'])
        if scenario=='compile_failure':raise MemoryError('synthetic injected')
        return NS(rz_count=3,rz_depth=2,cx_count=4,cx_depth=3,total_depth=8,circuit_size=9,
            actual_circuit_fingerprint='a'*64,compiler_settings_hash=reuse.COMPILER_HASH)
    monkeypatch.setattr(port,'compile_owned',compile)
    writer=AtomicWriter(tmp_path,byte_cap=2**20);trace=ResourceTrace(writer.write,lambda:0.)
    if scenario in ('partial_digest_bad','compile_failure'):
        with pytest.raises((ValueError,MemoryError)):port.cost_task(tmp_path,m,task,writer,Progress(writer),trace)
        assert not (tmp_path/'cost_result.json').exists()
    else:
        port.cost_task(tmp_path,m,task,writer,Progress(writer),trace)
        result=json.loads((tmp_path/'cost_result.json').read_text())
        assert len(result['rows'])==(4 if task['ordinary'] else 2)
        assert len({r['event_digest'] for r in result['rows']})==1
        assert result['calls_attempted']['compile']==len(result['rows'])
        assert builds==(['symmetric_directional']*2+['ordinary']*2 if task['ordinary'] else ['symmetric_directional']*2)
    assert len(sampled)==(1 if random else 0)
