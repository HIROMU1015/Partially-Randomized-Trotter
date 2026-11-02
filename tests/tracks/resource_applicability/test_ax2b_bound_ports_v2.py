"""Synthetic-only port/gate tests. No molecular snapshot, sampling or circuits."""
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
import pytest
from scipy.linalg import expm

from trottertracks.resource_applicability import ax2b_bound_launch_v2 as launch
from trottertracks.resource_applicability.ax2b_stage_validation_v2 import traced_signal, mp_cell
from trottertracks.resource_applicability.ax2b_molecular_ports_v2 import (
    occupation_column, compare_mp_records, compare_stages, check_bounds,
    local_matrix_action, explicit_event_action, hadamard_expectation, MolecularPort, FreshShotRequests,
    load_h6_snapshot as reject_synthetic_header, validation_times)
from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
from trottertracks.resource_applicability.ax2b_h6_controller import BoundedWriter
from trottertracks.resource_applicability.ax2b_launch_watchdog_v2 import supervise
from trottertracks.resource_applicability.ax2a_preparation import digest


@pytest.fixture(autouse=True)
def forbidden_science(monkeypatch):
    original = Path.open
    def guarded(path,*a,**kw):
        if '/artifacts/' in str(path) or str(path).endswith('.npz') or '/.runtime/' in str(path):
            raise AssertionError('MOLECULAR_IO_FORBIDDEN')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    def deny(*a,**kw):
        raise AssertionError('MOLECULAR_SAMPLING_CIRCUIT_FORBIDDEN')
    import qiskit
    import trotterlib.df_hamiltonian as df
    import trotterlib.rte as rte
    import trotterlib.df_partial_s2_repeated as repeated
    monkeypatch.setattr(np,'load',deny)
    monkeypatch.setattr(qiskit.QuantumCircuit,'__init__',deny)
    monkeypatch.setattr(qiskit,'transpile',deny)
    monkeypatch.setattr(df,'build_df_h_d_from_molecule',deny)
    monkeypatch.setattr(df,'low_rank_two_body_decomposition',deny)
    for name in ('sample_rte_events','iter_sample_rte_events'):
        monkeypatch.setattr(rte,name,deny)
    monkeypatch.setattr(repeated,'make_df_partial_s2_repeated_request',deny)
    from trottertracks.resource_applicability import ax2b_molecular_ports_v2 as ports
    for name in ('make_df_partial_s2_repeated_request','_load_snapshot_once','load_h6_snapshot',
                 'make_native_block_action','build_deterministic_native','partial_native_from_step_requests',
                 'build_native_hadamard_wrapper','transpile_and_measure_cost'):
        monkeypatch.setattr(ports,name,deny)


def toy():
    one = np.array([[.2,.3],[.3,-.1]],dtype=complex)
    g = np.array([[.8,.2],[.2,-.3]],dtype=complex)
    return NS(constant=.125,one_body=one,lambdas=np.array([.7]),g_matrices=(g,)),[2,1],np.array([.6,.8],dtype=complex)


@pytest.mark.parametrize('formula',['2nd','4th'])
@pytest.mark.parametrize('q',[1,2])
def test_native_action_endpoint_vs_independent_mp(formula,q):
    ham,basis,state = toy()
    terms = [ham.one_body,ham.lambdas[0]*ham.g_matrices[0]@ham.g_matrices[0]]
    actions = [lambda v,t,h=h:expm(-1j*t*h)@v for h in terms]
    cell = {'id':'toy','method':'B1','prefix':1,'q':q,'R':None,'K':None,'formula':formula}
    value = traced_signal(state,actions,None,cell=cell,T=-.7,scalar=ham.constant,lambda_r=0.,budget=ActionBudget(0,1000))
    oracle = mp_cell(ham,basis,state,cell,T=-.7,scalar=ham.constant,dps=80)
    comparison = compare_stages(value,oracle)
    assert len(comparison['stages']) == len(value['trace'])
    assert max(float(x['state_difference']) for x in comparison['stages']) < 1e-13
    assert not oracle['certified']
    if formula == '4th':
        assert any(x['time']>0 for x in value['trace'])  # negative global T includes reverse S4 piece


@pytest.mark.parametrize('K',[0,2,6])
@pytest.mark.parametrize('T',[.7,-.7])
def test_finite_stage_endpoint_pairing_raw_and_corrected(K,T):
    ham,basis,state = toy(); identity=.15; lam=1.4
    tail=(ham.lambdas[0]*ham.g_matrices[0]@ham.g_matrices[0]-identity*np.eye(2))/lam
    cell={'id':'toy','method':'B2','prefix':0,'q':2,'R':4,'K':K,'formula':'2nd'}
    action=lambda v,t:expm(-1j*t*ham.one_body)@v
    budget=ActionBudget(4*2*(K+1),1000)
    value=traced_signal(state,[action],lambda v:tail@v,cell=cell,T=T,scalar=ham.constant+identity,lambda_r=lam,budget=budget)
    oracle=mp_cell(ham,basis,state,cell,T=T,scalar=ham.constant+identity,lambda_r=lam,extracted_identity=identity,dps=120)
    comparison=compare_stages(value,oracle)
    assert comparison['stages'] and max(float(x['state_difference']) for x in comparison['stages']) < 1e-13
    assert budget.tail_matvecs==8*(K+1)
    assert abs(value['signals']['raw']*math.exp(value['log_B'])-value['signals']['corrected']) < 1e-13
    assert any(abs(x['norm_after']-1)>1e-4 for x in value['trace'])


def test_mp_precision_difference_keeps_decimal_digits():
    ham,basis,state=toy()
    cell={'id':'toy','method':'B0','prefix':0,'q':2,'R':None,'K':None,'formula':'2nd'}
    records=[mp_cell(ham,basis,state,cell,T=.8,scalar=ham.constant,dps=p) for p in (80,120)]
    difference=compare_mp_records(*records)
    assert set(difference['signal_differences'])=={'corrected','reference','exact_truncated'}
    assert max(float(v) for v in difference['signal_differences'].values()) < 1e-75
    assert len(records[1]['signals']['reference']['real'])>110


def test_full_occupation_squared_before_projection():
    ham=NS(constant=0.,one_body=np.zeros((2,2)),lambdas=[1.],g_matrices=(np.array([[0.,1.],[1.,0.]]),))
    assert np.array_equal(occupation_column(ham,[2],2),[1.])
    ham.one_body=np.array([[0.,1.],[1.,0.]])
    with pytest.raises(ValueError,match='LEAVES_SECTOR'):
        occupation_column(ham,[2],2)


@pytest.mark.parametrize('qubits',[(0,),(2,),(0,2),(2,0)])
def test_local_little_endian_action(qubits):
    size=2**len(qubits); matrix=np.fliplr(np.eye(size))
    v=np.arange(8,dtype=complex)
    actual=local_matrix_action(v,matrix,qubits)
    mask=sum(1<<q for q in qubits)
    assert np.array_equal(actual,v[np.arange(8)^mask])


def test_event_relative_phase_signed_identity_and_y_axis():
    app=NS(is_identity=True,coefficient_sign=-1,role='product')
    rotation=NS(is_identity=True,coefficient_sign=1,role='rotation')
    event=NS(application_sequence=(app,rotation),rotation_angle=.3,phase=-1)
    psi=np.array([1.,0.],dtype=complex)
    z=explicit_event_action(psi,event,None)
    assert np.allclose(z,np.exp(-.3j)*psi)
    # Actual H/Sdg/H algebra, no circuit construction.
    h=np.array([[1.,1.],[1.,-1.]])/math.sqrt(2)
    for axis,expected in (('cosine',math.cos(.3)),('sine',-math.sin(.3))):
        state=np.concatenate((psi,z))/math.sqrt(2)
        if axis=='sine': state[2:]*=-1j
        state=local_matrix_action(state,h,(1,))
        assert abs(hadamard_expectation(state,1)-expected)<1e-14


def test_local_matrix_asymmetric_control_respects_qubit_order():
    matrix=np.eye(4)[[0,3,2,1]]
    v=np.arange(8,dtype=complex)
    for control,target in ((0,2),(2,0)):
        expected=[i^(1<<target) if (i>>control)&1 else i for i in range(8)]
        assert np.array_equal(local_matrix_action(v,matrix,(control,target)),v[expected])


def test_event_basis_signed_rotation_and_negative_phase():
    from qiskit.circuit.library import HGate
    definition=NS(metadata=NS(basis_hash='fixture'),runtime_operations=((HGate(),(0,)),))
    registry=NS(definition=lambda _:definition)
    product=NS(is_identity=False,coefficient_sign=1,role='product',basis_id='h',basis_hash='fixture',diagonal_pauli_support=(0,))
    rotation=NS(is_identity=False,coefficient_sign=-1,role='rotation',basis_id='h',basis_hash='fixture',diagonal_pauli_support=(0,))
    event=NS(application_sequence=(product,product,rotation),phase=-1,rotation_angle=.2)
    psi=np.array([1.,0.],dtype=complex)
    actual=explicit_event_action(psi,event,registry)
    assert np.allclose(actual,[-math.cos(.2),-1j*math.sin(.2)],atol=1e-14)
    rotation.basis_hash='changed'
    with pytest.raises(ValueError,match='BASIS_HASH'):
        explicit_event_action(psi,event,registry)


def sealed_fixture(tmp_path,monkeypatch,kind='H6_TECHNICAL'):
    manifest=launch.preparation(kind)
    rank=2 if kind=='H6_TECHNICAL' else 12
    manifest.update(execution_plan_sealed=True,source_commit='a'*40,source_hashes={'fake':'hash'},environment={'fake':1},
                    input_binding={'fake':1},assigned_resources={'assigned_cpu':0,'science_workers':1,'blas_threads':1})
    manifest['plan']=launch.scope_plan(kind,rank if kind=='H6_TECHNICAL' else None)
    manifest['coverage_binding']={'sealed':True,'actual_rank':rank,'expected_bounds':{},
              'schedule_digest':digest(manifest['plan'].get('primitive_schedules',manifest['plan'].get('cells')))}
    auth={'schema':'track_a_ax2b_bound_authorization_v2','kind':kind,'approved_by_user':True,
          'manifest_digest':digest(manifest),'assigned_cpu':0,'exclusive_output':str(tmp_path/'future'),
          'retry':False,'resume':False}
    monkeypatch.setattr(launch,'verify_sources',lambda *a:None)
    monkeypatch.setattr(launch,'verify_input',lambda *a:rank)
    monkeypatch.setattr(launch,'environment',lambda:{'fake':1})
    monkeypatch.setattr(launch.os,'sched_getaffinity',lambda _: {0})
    return manifest,auth


@pytest.mark.parametrize('kind',['H4_LIMITED','H6_TECHNICAL'])
def test_separate_grant_and_all_bound_fields(tmp_path,monkeypatch,kind):
    m,a=sealed_fixture(tmp_path,monkeypatch,kind)
    assert launch.validate_launch(tmp_path,m,a,requested=True,output=tmp_path/'future')==0
    for mutation in ('approval','digest','unsealed','scope','cpu','retry','coverage'):
        badm,bada=copy.deepcopy(m),copy.deepcopy(a)
        if mutation=='approval': bada['approved_by_user']=False
        elif mutation=='digest': bada['manifest_digest']='changed'
        elif mutation=='unsealed': badm['execution_plan_sealed']=False
        elif mutation=='scope': badm['plan']['retry']=True
        elif mutation=='cpu': bada['assigned_cpu']=True
        elif mutation=='retry': bada['retry']=True
        elif mutation=='coverage': badm['coverage_binding']['sealed']=False
        if mutation!='digest': bada['manifest_digest']=digest(badm)
        with pytest.raises((ValueError,KeyError)):
            launch.validate_launch(tmp_path,badm,bada,requested=True,output=tmp_path/'future')


def test_no_grant_stops_before_any_input_or_source_access(tmp_path,monkeypatch):
    def deny(*a,**kw): raise AssertionError('GATE_ORDER')
    monkeypatch.setattr(launch,'verify_input',deny); monkeypatch.setattr(launch,'verify_sources',deny)
    with pytest.raises(ValueError,match='GRANT'):
        launch.validate_launch(tmp_path,launch.preparation(),{},requested=True,output=tmp_path/'future')


def test_commit_blob_and_local_byte_binding(tmp_path,monkeypatch):
    file=tmp_path/'fake.py';file.write_bytes(b'exact bytes')
    monkeypatch.setattr(launch,'source_paths',lambda _:['fake.py'])
    monkeypatch.setattr(launch.subprocess,'run',lambda *a,**kw:None)
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**kw:b'exact bytes')
    hashes={'fake.py':hashlib.sha256(b'exact bytes').hexdigest()}
    launch.verify_sources(tmp_path,'a'*40,hashes)
    file.write_bytes(b'drift')
    with pytest.raises(ValueError,match='LOCAL_SOURCE_CHANGED'):
        launch.verify_sources(tmp_path,'a'*40,hashes)
    file.write_bytes(b'exact bytes')
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**kw:b'foreign bytes')
    with pytest.raises(ValueError,match='COMMIT_BYTES'):
        launch.verify_sources(tmp_path,'a'*40,hashes)


def test_bound_gate_caps_are_before_action():
    caps={'primitive':2000,'untranspiled_instructions':100}
    with pytest.raises(RuntimeError,match='COVERAGE'):
        check_bounds({'primitive_actions':2001,'cells':[]},caps)
    with pytest.raises(RuntimeError,match='INSTRUCTION'):
        check_bounds({'primitive_actions':10,'cells':[{'wrapper_instruction_upper_bounds':{'ordinary':101}}]},caps)


def test_h4_estimator_microstep_half_times_are_registered():
    cell={'id':'H4_B2_K2','method':'B2','prefix':6,'q':4,'R':8,'r':2,'K':2,'formula':'2nd'}
    values=set(map(tuple,validation_times(cell,T=.8)))
    for i in range(7):
        assert (i,.05) in values and (i,-.05) in values
        assert (i,.1) in values and (i,-.1) in values


def test_h6_metadata_rank_policy_and_generation_provenance(tmp_path,monkeypatch):
    path=tmp_path/'toy.fixture';path.write_bytes(b'synthetic metadata fixture')
    hm={'input_policy':'TOL_ONLY_NO_CONFIG_FALLBACK','df_tol_requested':1e-8,
        'decomposition_kwargs':{'truncation_threshold':1e-8},'final_rank_supplied':False,
        'coefficient_order':'decomposer_generation_order','subsequent_coefficient_cutoff':0.,
        'df_rank_actual':2,'input_tensors':{},'one_body_correction':{},'hermitization':[], 'df_truncation_value':0.}
    metadata={'model':'linear_H6','geometry_angstrom':1.,'basis':'sto-3g','hamiltonian_metadata':hm,
              'sector':{'n_qubits':12,'nelec_alpha':3,'nelec_beta':3},
              'input_generation':{'source_commit':'a'*40,'authorization_sha256':'b'*64}}
    binding={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metadata':metadata}
    monkeypatch.setattr(launch,'metadata_only',lambda _:metadata)
    assert launch.verify_input(tmp_path,binding,'H6_TECHNICAL')==2
    hm['decomposition_kwargs']['final_rank']=11
    with pytest.raises(ValueError,match='TOL_ONLY'):launch.verify_input(tmp_path,binding,'H6_TECHNICAL')
    del hm['decomposition_kwargs']['final_rank']
    metadata['input_generation']['source_commit']=None
    with pytest.raises(ValueError,match='GENERATION_PROVENANCE'):launch.verify_input(tmp_path,binding,'H6_TECHNICAL')


def test_h6_header_rejection_precedes_np_load(tmp_path):
    import zipfile
    from trottertracks.resource_applicability import ax2b_molecular_ports_v2 as ports
    path=tmp_path/'malformed.fixture'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('one_body.npy',b'bad')
    with pytest.raises(ValueError,match='H6_NPZ_KEYS'):
        reject_synthetic_header(path,{'hamiltonian_metadata':{'df_rank_actual':2}})


def test_cli_default_stdlib_only_and_execute_without_grant(tmp_path):
    runner=Path(__file__).resolve().parents[3]/'scripts/resource_applicability/run_track_a_ax2b_bound_v2.py'
    output=tmp_path/'proposal.json'
    run=subprocess.run([sys.executable,'-S',str(runner),'--output',str(output)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    proposal=json.loads(output.read_text())
    assert proposal['status']=='H6_NOT_AUTHORIZED' and not proposal['execution_plan_sealed']
    bad=subprocess.run([sys.executable,'-S',str(runner),'--output',str(tmp_path/'science'),'--execute'],capture_output=True,text=True)
    assert bad.returncode!=0 and not (tmp_path/'science').exists()


@pytest.mark.parametrize('case',['complete','missing','wall','completed_phase_overrun','claim'])
def test_watchdog_dummy_workers(tmp_path,case):
    caps={'phase_wall_seconds':dict.fromkeys(launch.PHASES,.2),'total_wall_seconds':3.,
          'output_bytes':2**20,'log_bytes':65536}
    terminal={'status':'H6_TECHNICAL_COMPLETE','completed_correctness_cells':7,'compiled_wrappers':36,
              'mandatory_stop':True,'next_stage_authorized':False,'N':None,'G':None,
              'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED'}
    if case=='claim':terminal['N']=1
    worker=tmp_path/'dummy.py'; output=tmp_path/'out';output.mkdir()
    worker.write_text('import pathlib,json,time\np=pathlib.Path('+repr(str(output))+')\n'+
       ("time.sleep(.5)\n" if case=='wall' else '')+
       ''.join("(p/"+repr('phase_'+phase+'.json')+").write_text("+repr(json.dumps({'phase':phase,'elapsed': .4 if case=='completed_phase_overrun' and i else i*.001}))+ ")\n" for i,phase in enumerate(launch.PHASES))+
       ('' if case=='missing' else "(p/'worker_terminal.json').write_text("+repr(json.dumps(terminal))+")\n"))
    result=supervise([sys.executable,str(worker)],output,caps=caps,kind='H6_TECHNICAL')
    assert result['status']==('H6_TECHNICAL_COMPLETE' if case=='complete' else 'H6_TECHNICAL_STOP')
    assert result['mandatory_stop'] and not result['next_stage_authorized']


def test_h6_setup_synthetic_sector_reference_only(tmp_path,monkeypatch):
    from trottertracks.resource_applicability import ax2b_molecular_ports_v2 as ports
    from trotterlib.df_hamiltonian import PhysicalSector
    sector=PhysicalSector.spin_sector(n_qubits=12,nelec_alpha=3,nelec_beta=3)
    state=np.zeros(400,dtype=complex);state[0]=1
    full=np.zeros(4096,dtype=complex);full[sector.basis_indices[0]]=1
    ham=NS(n_qubits=12,n_blocks=2,constant=.125,one_body=np.zeros((12,12)),
           lambdas=np.zeros(2),g_matrices=(np.zeros((12,12)),)*2)
    manifest=launch.preparation();manifest['plan']=launch.scope_plan('H6_TECHNICAL',2)
    manifest['input_binding']={'path':'fake.fixture','metadata':{}}
    bounds={'primitive_actions':0,'cells':[]}
    manifest['coverage_binding']={'expected_bounds':bounds}
    monkeypatch.setattr(ports,'verify_input',lambda *a:2)
    monkeypatch.setattr(ports,'load_h6_snapshot',lambda *a:(ham,sector,full,state,{},{}))
    monkeypatch.setattr(ports,'_prepare',lambda *a:NS())
    monkeypatch.setattr(ports,'_prepare_discard',lambda *a:NS())
    monkeypatch.setattr(ports,'actual_bounds',lambda *a,**kw:bounds)
    class FakeOperator:
        def __matmul__(self,v):return .125*v
    monkeypatch.setattr(ports,'df_linear_operator',lambda *a,**kw:(FakeOperator(),{}))
    port=MolecularPort(tmp_path,manifest,BoundedWriter(tmp_path,byte_cap=2**20))
    port.setup()
    assert port.calls.used['reference_matvec']==400
    assert abs(port.reference-np.exp(-.1j))<1e-14
    assert port.oracle_cache is None
    assert not hasattr(port,'fragments') and not hasattr(port,'eigensystems')


@pytest.mark.parametrize('fail_at',[None,5])
def test_h6_mock_backend_36_wrappers_same_prepared_group(tmp_path,monkeypatch,fail_at):
    from trottertracks.resource_applicability import ax2b_molecular_ports_v2 as ports
    manifest=launch.preparation();manifest['plan']=launch.scope_plan('H6_TECHNICAL',2)
    port=MolecularPort(tmp_path,manifest,BoundedWriter(tmp_path,byte_cap=2**20))
    port.ham=NS(n_qubits=1);port.qstate=np.array([1.,0.],dtype=complex);port.qindices=(0,1)
    port.completed=7
    port.preps={c['id']:NS(deterministic_blocks=(),constant_coefficient=.125) for c in port.cells}
    trajectories=[]
    def fake_sample(cell,seed):
        port.calls.take('trajectory');port.calls.take('occurrence',cell['q'])
        events=[NS(to_dict=lambda:{'fake':'event'})]
        steps=tuple(NS(rte_occurrence=NS(events=events)) for _ in range(cell['q']))
        trajectories.append((cell['id'],seed,id(steps)));return steps
    monkeypatch.setattr(port,'sampled_steps',fake_sample)
    def evolution(*a,**kw):return NS(circuit=NS(role='evolution'))
    monkeypatch.setattr(ports,'build_deterministic_native',evolution)
    monkeypatch.setattr(ports,'partial_native_from_step_requests',evolution)
    def wrapper(e,*,axis,**kw):return NS(role=axis,data=[1])
    monkeypatch.setattr(ports,'build_native_hadamard_wrapper',wrapper)
    def simulate(circuit,v):
        z=np.exp(-.2j)
        if circuit.role=='evolution':return np.concatenate((v[:2],z*v[2:]))
        result=np.concatenate((v[:2],z*v[:2]))/math.sqrt(2)
        if circuit.role=='sine':result[2:]*=-1j
        return local_matrix_action(result,np.array([[1.,1.],[1.,-1.]])/math.sqrt(2),(1,))
    monkeypatch.setattr(ports,'simulate_statevector',simulate)
    monkeypatch.setattr(ports,'canonical_qiskit_circuit_fingerprint',lambda _: 'fake')
    attempts=[]
    def compile_(wrapper,compiler,**kw):
        attempts.append(1)
        if len(attempts)==fail_at:raise RuntimeError('FAKE_COMPILE_STOP')
        return NS(transpiled_circuit=NS(data=[1]),actual_circuit_fingerprint='fake',compiler_settings_hash='fake',
                  **dict.fromkeys(('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size'),1))
    monkeypatch.setattr(ports,'transpile_and_measure_cost',compile_)
    if fail_at:
        with pytest.raises(RuntimeError,match='FAKE_COMPILE_STOP'):port.costs()
        assert port.compiled==4 and port.calls.used['compile']==5
        assert len(list(tmp_path.glob('wrapper_*.json')))==4
    else:
        port.costs()
        assert port.compiled==36 and port.calls.used['compile']==36
        assert len(trajectories)==4 and port.calls.used['occurrence']==8
        assert port.calls.used['control_probe']==216
        assert len(list(tmp_path.glob('wrapper_*.json')))==36


def test_fresh_shot_factory_is_called_every_time_before_cap():
    seen=[]
    def factory(cell,seed):seen.append(seed);return {'cell':cell['id'],'seed':seed}
    request=FreshShotRequests(factory,max_shots=3,master_seed=17)
    values=[request.draw({'id':'synthetic'}) for _ in range(3)]
    assert len(seen)==len(set(seen))==3 and len({id(v) for v in values})==3
    with pytest.raises(RuntimeError,match='CALL_BUDGET'):request.draw({'id':'synthetic'})
    assert len(seen)==3


def test_h6_correctness_method_all_seven_synthetic_cells(tmp_path,monkeypatch):
    from trottertracks.resource_applicability import ax2b_molecular_ports_v2 as ports
    ham,basis,state=toy()
    ham.lambdas=np.array([.7,.2]);ham.g_matrices+=(np.array([[.3,.1],[.1,.7]],dtype=complex),)
    ham.n_blocks=2;ham.n_qubits=2
    manifest=launch.preparation();manifest['plan']=launch.scope_plan('H6_TECHNICAL',2)
    port=MolecularPort(tmp_path,manifest,BoundedWriter(tmp_path,byte_cap=2**20))
    port.ham,port.sector,port.state=ham,NS(basis_indices=basis,dimension=2),state
    port.preps={c['id']:NS(exact_rte_lambda_r=2.,extracted_identity_coefficient=0.,constant_coefficient=.125)
                for c in port.cells}
    monkeypatch.setattr(port,'primitives',lambda:None)
    # Native callback substitution; independent occupation/tail oracle stays live.
    def actions(prep,*,oracle=False):
        cell=next(c for c in port.cells if port.preps[c['id']] is prep)
        terms=[ham.one_body]+[w*g@g for w,g in zip(ham.lambdas,ham.g_matrices)]
        return [lambda v,t,h=h:expm(-1j*t*h)@v for h in terms[:cell['prefix']+1]]
    monkeypatch.setattr(port,'actions',actions)
    def tail(ham,sector,indices,**kw):
        matrix=sum((ham.lambdas[i]*ham.g_matrices[i]@ham.g_matrices[i] for i in indices),np.zeros((2,2)))
        class Operator:
            def __matmul__(self,v):return matrix@v
        return Operator(),{}
    monkeypatch.setattr(ports,'df_tail_operator',tail)
    port.correctness()
    assert port.completed==7 and port.compiled==0
    assert len(list(tmp_path.glob('H6_*_correctness.json')))==7
    for path in tmp_path.glob('H6_*_correctness.json'):
        row=json.loads(path.read_text())
        assert row['N'] is None and row['G'] is None and not row['numerical_allowance_certified']
        if row['cell']['method']=='B3':assert row['action_counts']['tail']==56


def test_h4_correctness_method_mp_two_precisions_eight_toy_cells(tmp_path,monkeypatch):
    from trottertracks.resource_applicability import ax2b_molecular_ports_v2 as ports
    ham,basis,state=toy()
    ham.lambdas=np.array([.7,.2]);ham.g_matrices+=(np.array([[.3,.1],[.1,.7]],dtype=complex),)
    ham.n_blocks=2;ham.n_qubits=2
    cells=[{'id':f'toy_{i}','method':m,'order':order,'prefix':prefix,'q':q,'R':R,'K':K}
           for i,(m,order,prefix,q,R,K) in enumerate([
              ('B1','2nd',2,1,None,None),('B1','2nd',2,4,None,None),('B0','2nd',1,4,None,None),
              ('B2','2nd',1,4,8,2),('B2','2nd',1,4,8,4),('B3','2nd',0,4,8,6),
              ('B1','4th',2,1,None,None),('B1','4th',2,4,None,None)])]
    manifest=launch.preparation('H4_LIMITED');manifest['plan']['cells']=cells
    port=MolecularPort(tmp_path,manifest,BoundedWriter(tmp_path,byte_cap=8*2**20))
    port.ham,port.sector,port.state,port.saved_state=ham,NS(basis_indices=basis,dimension=2),state,state.copy()
    target=ham.constant*np.eye(2)+ham.one_body+sum(w*g@g for w,g in zip(ham.lambdas,ham.g_matrices))
    port.reference=complex(np.vdot(state,expm(-.8j*target)@state))
    port.preps={c['id']:NS(exact_rte_lambda_r=2.,extracted_identity_coefficient=0.,constant_coefficient=.125) for c in cells}
    monkeypatch.setattr(port,'primitives',lambda:None)
    def actions(prep,**kw):
        cell=next(c for c in cells if port.preps[c['id']] is prep)
        terms=[ham.one_body]+[w*g@g for w,g in zip(ham.lambdas,ham.g_matrices)]
        return [lambda v,t,h=h:expm(-1j*t*h)@v for h in terms[:cell['prefix']+1]]
    monkeypatch.setattr(port,'actions',actions)
    def tail(ham,sector,indices,**kw):
        matrix=sum((ham.lambdas[i]*ham.g_matrices[i]@ham.g_matrices[i] for i in indices),np.zeros((2,2)))
        class Operator:
            def __matmul__(self,v):return matrix@v
        return Operator(),{}
    monkeypatch.setattr(ports,'df_tail_operator',tail)
    port.correctness()
    assert port.completed==8 and port.compiled==0
    assert len(list(tmp_path.glob('*_mp*.json')))==16
    for path in tmp_path.glob('toy_*_correctness.json'):
        row=json.loads(path.read_text())
        assert row['stage_comparison_mp120']['stages'] and row['precision_comparison']['signal_differences']
