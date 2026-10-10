"""Synthetic matrices, injected costs and dummy children only; no H6 NPZ decode."""
from copy import deepcopy
import builtins
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import sys
import numpy as np
import pytest
from scipy.linalg import expm
from trottertracks.resource_applicability import ax2b_h6_pilot_contract_v2 as contract
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h6_pilot_port_v2 import H6PilotPort
from trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 import recurrence_paths,reference_pair,differences
from trottertracks.resource_applicability.ax2b_h6_pilot_watchdog_v1 import PilotProgress,supervise
from trottertracks.resource_applicability.ax2b_h6_pilot_audit_v2 import audit_saved
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2b_stage_validation_v2 import traced_signal
from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
from trottertracks.resource_applicability import ax2b_h6_pilot_coverage_v2 as binding
from trottertracks.resource_applicability import ax2b_molecular_ports_v3 as runtime
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec
ROOT=Path(__file__).resolve().parents[3]


def synthetic_preparation(prefix):
    one = DFDeterministicOneBodySpec('toy_one', 'one_body', None, None, 'toy',
        'toy_basis', (), (0.,) * 12, 12, 0, ())
    blocks = (one,) + tuple(NS(original_fragment_index=i, num_system_qubits=12,
        runtime_basis_operations=(), basis_hash='toy_basis') for i in range(prefix))
    return NS(deterministic_blocks=blocks, deterministic_fragment_indices=tuple(range(prefix)),
        randomized_block_indices=tuple(range(prefix,19)), coefficient_atol=0.,
        threshold_dropped_component_count=0, hamiltonian_hash='toy_h', preparation_hash='toy_p',
        partition_hash='toy_partition', exact_rte_lambda_r=1., ranking_proxy_lambda_r=1.,
        extracted_identity_coefficient=0., constant_coefficient=0.,
        rte_preparation=NS(component_specs=(NS(diagonal_pauli_support=(0,), basis_change_operations=()),)))


def synthetic_bounds():
    cells=contract.plan()['cells']
    # Exercise the real runtime function and its real registered-time function.
    # Only prepared metadata is synthetic; no preparation, circuit or action occurs.
    return runtime.actual_bounds(cells, {c['id']:synthetic_preparation(c['prefix']) for c in cells}, T=.8)


def test_real_runtime_bounds_connect_to_fixed_v2_after_json_roundtrip(tmp_path):
    coverage=json.loads(json.dumps(contract.coverage()))
    bounds=synthetic_bounds()
    writer=AtomicWriter(tmp_path,byte_cap=4*2**20)
    receipt=binding.assert_actual_coverage(coverage,bounds,writer)
    assert receipt['equal'] is True and receipt['differences']==[]
    assert bounds['primitive_actions']==coverage['primitive_actions']==735
    assert len(coverage['physical_times'])==245
    for row in bounds['cells']:
        s=row['schedule']
        assert s['registered_validation_times_v2']==s['unique_primitive_times']
    # Precisely reproduce the old schema failure without opening H6 input.
    from trottertracks.resource_applicability import ax2b_h6_pilot_contract_v1 as old
    assert digest([{'cell_id':r['cell_id'],'schedule':r['schedule']} for r in bounds['cells']])!=digest(old.coverage()['cells'])


@pytest.mark.parametrize('mutation', ['time','index','order','registered_time','registered_order',
    'missing_registered','extra_registered','missing_cell','extra_cell','cell_order','count',
    'bool_count','float_count','probe_count','extra_schedule_field','missing_schedule_field',
    'extra_bounds_field','extra_row_field','missing_control','negative_bound','float_bound'])
def test_strict_binding_saves_evidence_before_rejecting_mutation(tmp_path,mutation):
    b=synthetic_bounds();s=b['cells'][0]['schedule']
    if mutation=='time':s['ordinary_one_outer_step'][0]=(0,np.nextafter(.1,np.inf).item())
    elif mutation=='index':s['ordinary_one_outer_step'][0]=(99,s['ordinary_one_outer_step'][0][1])
    elif mutation=='order':
        seq=s['ordinary_one_outer_step'];seq[0],seq[1]=seq[1],seq[0]
    elif mutation=='registered_time':s['registered_validation_times_v2'][0]=(0,-.123)
    elif mutation=='registered_order':s['registered_validation_times_v2'].reverse()
    elif mutation=='missing_registered':s['registered_validation_times_v2'].pop()
    elif mutation=='extra_registered':s['registered_validation_times_v2'].append((0,.123))
    elif mutation=='missing_cell':b['cells'].pop()
    elif mutation=='extra_cell':b['cells'].append(deepcopy(b['cells'][0]))
    elif mutation=='cell_order':b['cells'].reverse()
    elif mutation=='count':b['primitive_actions']+=1
    elif mutation=='bool_count':b['primitive_probe_count']=True
    elif mutation=='float_count':b['primitive_actions']=735.
    elif mutation=='probe_count':b['primitive_probe_count']=4
    elif mutation=='extra_schedule_field':s['unregistered']='extra'
    elif mutation=='missing_schedule_field':del s['registered_validation_times_v2']
    elif mutation=='extra_bounds_field':b['extra']=True
    elif mutation=='extra_row_field':b['cells'][0]['extra']=True
    elif mutation=='missing_control':del b['cells'][0]['wrapper_instruction_upper_bounds']['ordinary']
    elif mutation=='negative_bound':b['cells'][0]['wrapper_instruction_upper_bounds']['ordinary']=-1
    elif mutation=='float_bound':b['cells'][0]['wrapper_instruction_upper_bounds']['ordinary']=2.
    writer=AtomicWriter(tmp_path,byte_cap=4*2**20)
    with pytest.raises(ValueError,match='ACTUAL_COVERAGE_CHANGED'):
        binding.assert_actual_coverage(contract.coverage(),b,writer)
    assert json.loads((tmp_path/'actual_bounds_v2.json').read_text())==json.loads(json.dumps(b))
    receipt=json.loads((tmp_path/'coverage_comparison_v3.json').read_text())
    assert receipt['equal'] is False and receipt['differences']
    assert (tmp_path/'actual_coverage.json').exists()


@pytest.mark.parametrize('change', ['none','coverage','instruction_cap','order','cutoff'])
def test_setup_real_bounds_precedes_reference_and_saves_on_gate_failure(tmp_path,monkeypatch,change):
    from trottertracks.resource_applicability import ax2b_h6_pilot_port_v2 as portmod
    m=contract.preparation()
    identity={'snapshot_path':'toy.npz','snapshot_receipt_path':'toy.json','df_receipt_path':'toy_df.json'}
    m['input_identity']=identity
    monkeypatch.setattr(portmod,'verify_parent',lambda root:identity)
    monkeypatch.setattr(portmod,'read_json',lambda path:{'synthetic_only':True})
    state=np.array([1.,0.],complex)
    monkeypatch.setattr(portmod,'load_h6_snapshot',lambda *a:(NS(),NS(),state,state,{},None))
    monkeypatch.setattr(portmod,'primitive_sector_certificate',lambda *a:{'synthetic_only':True})
    monkeypatch.setattr(portmod,'checked_basis_bridge',lambda *a:(np.array([0,1]),state.copy()))
    def prepare(ham,prefix):
        p=synthetic_preparation(prefix)
        if change=='order':p.deterministic_fragment_indices=tuple(reversed(p.deterministic_fragment_indices))
        if change=='cutoff':p.coefficient_atol=1e-8
        return p
    monkeypatch.setattr(portmod,'_prepare',prepare)
    monkeypatch.setattr(portmod,'_prepare_discard',prepare)
    if change=='coverage':m['plan']['coverage']['cells'][0]['schedule']['registered_validation_times_v2'].pop()
    if change=='instruction_cap':m['plan']['caps_proposed']['untranspiled_instructions']=1
    reference_calls=[]
    def reference(*a,**k):
        reference_calls.append(True)
        raise RuntimeError('SYNTHETIC_REFERENCE_BOUNDARY')
    monkeypatch.setattr(portmod,'df_linear_operator',reference)
    writer=AtomicWriter(tmp_path,byte_cap=4*2**20)
    progress=PilotProgress(writer,cap=100)
    port=H6PilotPort(tmp_path,m,writer,progress);writer.observer=port.observe
    reason={'none':'SYNTHETIC_REFERENCE_BOUNDARY','coverage':'ACTUAL_COVERAGE_CHANGED',
            'instruction_cap':'STRUCTURAL_INSTRUCTION_BOUND_CAP','order':'PREPARATION_ORDER_OR_CUTOFF',
            'cutoff':'PREPARATION_ORDER_OR_CUTOFF'}[change]
    with pytest.raises((ValueError,RuntimeError),match=reason):port.setup()
    assert reference_calls==([True] if change=='none' else [])
    assert set(port.calls.used.values())=={0}
    for name in ('actual_prepared_representation.json','actual_bounds_v2.json','actual_coverage.json','coverage_comparison_v3.json'):
        assert (tmp_path/name).exists()
    if change!='coverage':binding.audit_coverage_records(tmp_path,m['plan']['coverage'])


@pytest.mark.parametrize('record', ['actual_bounds_v2.json','actual_coverage.json',
    'actual_prepared_representation.json','coverage_comparison_v3.json'])
def test_saved_coverage_audit_detects_tampering(tmp_path,record):
    writer=AtomicWriter(tmp_path,byte_cap=4*2**20);b=synthetic_bounds()
    writer.write('actual_prepared_representation.json',{'synthetic_only':True,'bounds':b})
    binding.assert_actual_coverage(contract.coverage(),b,writer)
    assert binding.audit_coverage_records(tmp_path,contract.coverage())['equal'] is True
    path=tmp_path/record;row=json.loads(path.read_text())
    if record=='coverage_comparison_v3.json':row['expected_sha256']='0'*64
    elif record=='actual_prepared_representation.json':row['bounds']['primitive_actions']=738
    else:row['primitive_actions']=738
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='PILOT_AUDIT_COVERAGE'):
        binding.audit_coverage_records(tmp_path,contract.coverage())


def test_v2_plan_scientific_fields_unchanged_and_inherited_methods_identical():
    from trottertracks.resource_applicability import ax2b_h6_pilot_contract_v1 as old
    from trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 import H6PilotPort as oldport
    a,b=old.plan(),contract.plan()
    for key in a:
        if key not in ('schema','coverage'):assert digest(a[key])==digest(b[key]),key
    for key in a['coverage']:
        if key!='cells':assert digest(a['coverage'][key])==digest(b['coverage'][key]),key
    for ar,br in zip(a['coverage']['cells'],b['coverage']['cells'],strict=True):
        newer=deepcopy(br);del newer['schedule']['registered_validation_times_v2']
        assert digest(ar)==digest(newer)
    for name in ('correctness','cell_signal','oracle_signal','primitives','sampled_steps','costs','control_and_measurement'):
        assert getattr(H6PilotPort,name) is getattr(oldport,name)


def test_v1_execution_grant_cannot_launch_v2(tmp_path,monkeypatch):
    m,g,out=sealed(tmp_path,monkeypatch)
    g['schema']='track_a_h6_technical_pilot_authorization_v1'
    with pytest.raises(ValueError,match='PILOT_GRANT_SCHEMA'):
        contract.validate_launch(tmp_path,m,g,out,requested=True)


@pytest.fixture(autouse=True)
def no_real_science(monkeypatch,tmp_path):
    def forbidden(*a,**k):raise AssertionError('REAL_SCIENCE_FORBIDDEN')
    import openfermion,openfermionpyscf,pyscf.gto,scipy.sparse.linalg,qiskit
    import trotterlib.df_hamiltonian as df
    import trotterlib.rte as rte
    import trottertracks.resource_applicability.ax2b_h6_pilot_port_v2 as port_v2
    monkeypatch.setattr(port_v2,'_prepare',forbidden)
    monkeypatch.setattr(port_v2,'_prepare_discard',forbidden)
    monkeypatch.setattr(port_v2,'load_h6_snapshot',forbidden)
    for obj,name in ((openfermion,'low_rank_two_body_decomposition'),(openfermion.MolecularData,'__init__'),
        (openfermionpyscf,'run_pyscf'),(pyscf.gto.Mole,'build'),(scipy.sparse.linalg,'eigsh'),
        (df,'build_df_h_d_from_molecule'),(qiskit.QuantumCircuit,'__init__'),(qiskit,'transpile')):
        monkeypatch.setattr(obj,name,forbidden)
    for name in ('sample_rte_events','iter_sample_rte_events','sample_event_mean_operator'):
        monkeypatch.setattr(rte,name,forbidden)
    original=np.load
    def temp_only(path,*a,**k):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve()),'REAL_NPZ_DECODE_FORBIDDEN'
        return original(path,*a,**k)
    monkeypatch.setattr(np,'load',temp_only)
    real_open,real_path=builtins.open,Path.open
    def check(path):
        if isinstance(path,(str,Path)) and '/artifacts/' in str(path) and not Path(path).resolve().is_relative_to(tmp_path.resolve()):
            raise AssertionError('REAL_ARTIFACT_IO_FORBIDDEN')
    def safe_open(path,*a,**k):check(path);return real_open(path,*a,**k)
    def safe_path(path,*a,**k):check(path);return real_path(path,*a,**k)
    monkeypatch.setattr(builtins,'open',safe_open);monkeypatch.setattr(Path,'open',safe_path)


def sealed(tmp_path,monkeypatch):
    m=contract.preparation();cpus=sorted(os.sched_getaffinity(0))[:4];out=tmp_path/contract.NAMESPACE/'launch_v1'
    m.update(execution_plan_sealed=True,source_commit='a'*40,source_hashes={'fixture':'b'*64},
        input_identity={'synthetic_only':True},environment={'fixture':True},assigned_resources=contract.resources(cpus),
        exclusive_output={'repository_path':contract.NAMESPACE+'launch_v1','absolute_path':str(out.resolve())})
    grant={'schema':'track_a_h6_technical_pilot_authorization_v2','approved_by_user':True,'kind':contract.KIND,
        'manifest_digest':digest(m),'source_commit':m['source_commit'],'assigned_cpus':cpus,
        'exclusive_output':str(out.resolve()),'retry':False,'resume':False,'one_shot':True}
    monkeypatch.setattr(contract,'verify_sources',lambda *a:None)
    monkeypatch.setattr(contract,'verify_parent',lambda *a:{'synthetic_only':True})
    monkeypatch.setattr(contract,'environment',lambda:{'fixture':True})
    return m,grant,out


def test_plan_counts_times_and_no_input_generation():
    p=contract.plan();assert len(p['cells'])==7 and len(p['wrapper_tasks'])==36
    assert [c['prefix'] for c in p['cells']]==[10,19,19,19,19,10,0]
    assert sum(c['replicas'] for c in p['cells'] if c['R'])==4
    assert sum(c['replicas']*c['q'] for c in p['cells'] if c['R'])==8
    assert p['coverage']['primitive_actions']<=p['caps_proposed']['primitive']
    times=p['coverage']['physical_times'];assert any(t<0 for _,t in times)
    assert any(k=='one' for k,_ in times) and any(k=='18' for k,_ in times)
    for key in ('integral_build','df_decomposition','state_solver','solver_matvec'):assert p['caps_proposed'][key]==0
    assert p['N'] is None and p['G'] is None and not p['prepared_representation_computed_in_preparation']


@pytest.mark.parametrize('field,value',[('approved_by_user',False),('approved_by_user',1),('schema','track_a_h6_saved_df_completion_authorization_v2'),
    ('kind','H6_SAVED_DF_INPUT_COMPLETION'),('manifest_digest','bad'),('source_commit','c'*40),('retry',True),
    ('resume',True),('one_shot',False),('assigned_cpus',[True,2,3,4]),('exclusive_output','/tmp/wrong')])
def test_wrong_or_consumed_grants_rejected(tmp_path,monkeypatch,field,value):
    m,g,out=sealed(tmp_path,monkeypatch);g[field]=value
    with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,out,requested=True)


@pytest.mark.parametrize('field',['T','cells','coverage','compiler_proposed','caps_proposed'])
def test_changed_plan_is_rejected(tmp_path,monkeypatch,field):
    m,g,out=sealed(tmp_path,monkeypatch);m['plan'][field]='altered';g['manifest_digest']=digest(m)
    with pytest.raises(ValueError,match='FIXED_PLAN'):contract.validate_launch(tmp_path,m,g,out,requested=True)


def test_gate_one_shot_and_environment(tmp_path,monkeypatch):
    m,g,out=sealed(tmp_path,monkeypatch)
    assert contract.validate_launch(tmp_path,m,g,out,requested=True)==g['assigned_cpus']
    with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,out)
    out.mkdir(parents=True)
    with pytest.raises(ValueError,match='EXCLUSIVE_OUTPUT'):contract.validate_launch(tmp_path,m,g,out,requested=True)
    writer=AtomicWriter(out,byte_cap=2**20);writer.write('launch_binding.json',{'manifest_digest':digest(m),'authorization_digest':digest(g)})
    assert contract.validate_launch(tmp_path,m,g,out,requested=True,worker=True)==g['assigned_cpus']
    writer.write('worker_claim.json',{})
    with pytest.raises(ValueError,match='ONE_SHOT'):contract.validate_launch(tmp_path,m,g,out,requested=True,worker=True)
    m,g,out=sealed(tmp_path/'other',monkeypatch);monkeypatch.setattr(contract,'environment',lambda:{'changed':True})
    with pytest.raises(ValueError,match='ENVIRONMENT'):contract.validate_launch(tmp_path/'other',m,g,out,requested=True)


@pytest.mark.parametrize('index',range(7))
def test_forward_recurrence_matches_horner_and_signed_decomposition(index):
    c=contract.plan()['cells'][index];psi=np.array([1,1j],complex)/np.sqrt(2)
    matrices=[np.array([[.1+i*.002,.03j],[-.03j,-.2]],complex) for i in range(c['prefix']+1)]
    actions=[lambda v,t,a=a:expm(-1j*t*a)@v for a in matrices]
    tail=np.array([[.2,.1],[.1,-.3]],complex);random=bool(c['R']);lam=2. if random else 0.
    maximum=contract.plan()['caps_proposed']['tail_matvecs_corrected_and_raw'].get(c['method'],0)
    a=traced_signal(psi,actions,lambda v:tail@v,cell=c,T=.8,scalar=.15,lambda_r=lam,budget=ActionBudget(maximum,100000))
    b=recurrence_paths(psi,actions,tail if random else None,cell=c,T=.8,scalar=.15,lambda_r=lam,maximum_tail=maximum)
    for path in a['signals']:
        assert np.linalg.norm(a['vectors'][path]-b['vectors'][path])<1e-12
        assert abs(a['signals'][path]-b['signals'][path])<1e-12
    if random:
        assert b['counts']['tail']==maximum
        assert abs(a['signals']['raw']*np.exp(a['log_B'])-a['signals']['corrected'])<1e-12
        assert not np.isclose(np.linalg.norm(a['vectors']['corrected']),1.,atol=1e-15,rtol=1e-15)
    if c['method']=='B0':b['signals']['exact_truncated']=.1+.2j
    d=differences(b['signals'],.3+.4j,c['method']);assert d['complex_closure_discrepancy']<1e-15
    assert not d['absolute_additivity_claim']


def test_oracle_budget_before_action_and_scale_rejection():
    c=contract.plan()['cells'][-1];psi=np.array([1.,0.],complex)
    with pytest.raises(RuntimeError,match='TAIL_MATVEC'):recurrence_paths(psi,[lambda v,t:v],np.eye(2),cell=c,T=.8,scalar=0.,lambda_r=1.,maximum_tail=0)
    with pytest.raises(ValueError,match='NORMALIZATION_OVERFLOW|RAW_ATTENUATION'):recurrence_paths(psi,[lambda v,t:v],np.eye(2),cell=c,T=.8,scalar=0.,lambda_r=1e30,maximum_tail=56)
    with pytest.raises(ValueError,match='SAVED_STATE'):recurrence_paths(2*psi,[lambda v,t:v],None,cell=contract.plan()['cells'][0],T=.8,scalar=0.,lambda_r=0.,maximum_tail=0)


def test_reference_pair_detects_nonhermitian_and_mismatch(monkeypatch):
    import trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 as port
    psi=np.array([1.,0.],complex);h=np.array([[.2,.3j],[-.3j,-.1]],complex)
    z,r=reference_pair(h,psi,.8);assert r['state_difference']<1e-12 and abs(z)<=1+1e-12
    with pytest.raises(ValueError,match='HERMITICITY'):reference_pair(np.array([[0,1],[0,0]],complex),psi,.8)
    monkeypatch.setattr(port,'eigh',lambda a:(np.array([10.,20.]),np.eye(2)))
    with pytest.raises(ValueError,match='EXPM_EIGH'):reference_pair(h,psi,.8)


def fake_cost_run(tmp_path,monkeypatch,*,fail=False):
    import trottertracks.resource_applicability.ax2b_molecular_ports_v3 as old
    m=contract.preparation();out=tmp_path/'synthetic_cost';out.mkdir(parents=True)
    writer=AtomicWriter(out,byte_cap=16*2**20);progress=PilotProgress(writer,cap=1024)
    port=H6PilotPort(tmp_path,m,writer,progress);writer.observer=port.observe
    port.ham=NS(n_qubits=2);port.completed=7;port.preps={c['id']:NS(deterministic_blocks=(),constant_coefficient=0.) for c in port.cells}
    builds=[];compiles=[];samples=[]
    def builder(*a,**k):builds.append(k['control_policy']);return NS(circuit='INJECTED_SYNTHETIC')
    monkeypatch.setattr(old,'build_deterministic_native',builder);monkeypatch.setattr(old,'partial_native_from_step_requests',builder)
    monkeypatch.setattr(old,'build_native_hadamard_wrapper',lambda *a,**k:'INJECTED_SYNTHETIC_WRAPPER')
    monkeypatch.setattr(old,'canonical_qiskit_circuit_fingerprint',lambda a:'fixture')
    def compile_cost(*a,**k):
        compiles.append('call')
        if fail:raise RuntimeError('INJECTED_COMPILE_STOP')
        return NS(transpiled_circuit=NS(data=[]),rz_count=2,rz_depth=2,cx_count=3,cx_depth=3,total_depth=5,circuit_size=5,actual_circuit_fingerprint='fixture',compiler_settings_hash='fixture')
    monkeypatch.setattr(old,'transpile_and_measure_cost',compile_cost)
    def draw(cell,seed):
        port.calls.take('trajectory');port.calls.take('occurrence',cell['q']);samples.append((cell['id'],seed))
        event=NS(to_dict=lambda:{'synthetic':True,'seed':seed})
        return tuple(NS(rte_occurrence=NS(events=(event,))) for _ in range(cell['q']))
    port.sampled_steps=draw
    def control(e):port.calls.take('control_probe',24);return 0.
    port.control_and_measurement=control
    return port,out,builds,compiles,samples


def test_paired_cost_schedule_36_without_real_sampling_or_compile(tmp_path,monkeypatch):
    port,out,builds,compiles,samples=fake_cost_run(tmp_path,monkeypatch);port.costs()
    assert len(builds)==18 and len(compiles)==36 and len(samples)==4
    assert port.calls.used['occurrence']==8 and port.calls.used['control_probe']==216
    for group in range(9):assert len({r['event_digest'] for r in port.wrapper_records[4*group:4*group+4]})==1
    summary=json.loads((out/'cost_summary.json').read_text());assert len(summary['groups'])==28
    assert not summary['measurement_shots_sampled'] and not summary['winner_claim']


def test_compile_stop_and_correctness_gate(tmp_path,monkeypatch):
    port,out,_,compiles,_=fake_cost_run(tmp_path,monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='INJECTED_COMPILE'):port.costs()
    assert len(compiles)==1 and port.calls.used['compile']==1 and port.compiled==0
    assert not list(out.glob('wrapper_*.json'))
    port,out,_,compiles,_=fake_cost_run(tmp_path/'other',monkeypatch);port.completed=6
    with pytest.raises(ValueError,match='CORRECTNESS_REQUIRED'):port.costs()
    assert not compiles


def test_watchdog_kills_child_at_wall_cap(tmp_path):
    out=tmp_path/'dummy';out.mkdir();caps=deepcopy(contract.plan()['caps_proposed'])
    caps['phase_wall_seconds']={p:.1 for p in contract.PHASES};caps['total_wall_seconds']=.3
    r=supervise([sys.executable,'-c','import time;time.sleep(10)'],out,caps=caps)
    assert r['status']=='H6_TECHNICAL_PILOT_STOP' and 'WALL_CAP' in r['reason']
    assert r['mandatory_stop'] and not r['next_stage_authorized']


def test_metadata_runner_imports_no_science(tmp_path):
    script=ROOT/contract.RUNNER;out=tmp_path/'metadata.json'
    # SystemExit is success; explicitly inspect eager imports after catching it.
    code='import runpy,sys\nsys.argv='+repr([str(script),'--output',str(out)])+'\ntry:runpy.run_path('+repr(str(script))+',run_name="__main__")\nexcept SystemExit as e:assert e.code==0\nassert not any(n in sys.modules for n in ("numpy","scipy","numba","qiskit","openfermion"))'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert json.loads(out.read_text())['status']=='H6_PILOT_NOT_AUTHORIZED'


def test_missing_required_result_rejected_by_byte_audit(tmp_path):
    out=tmp_path/'dummy_audit';out.mkdir();writer=AtomicWriter(out,byte_cap=2**20)
    m=contract.preparation();g={'schema':'track_a_h6_technical_pilot_authorization_v2','kind':contract.KIND,
        'source_commit':None,'approved_by_user':True,'manifest_digest':digest(m),'one_shot':True,'retry':False,'resume':False}
    writer.write('frozen_pilot.json',m);writer.write('authorization.json',g)
    writer.write('launch_binding.json',{'manifest_digest':digest(m),'authorization_digest':digest(g)})
    writer.write('authorization_source.json',g)
    writer.write('terminal_status.json',{'status':'H6_TECHNICAL_PILOT_COMPLETE','mandatory_stop':True,'next_stage_authorized':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','N':None,'G':None,
        'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED'})
    with pytest.raises(ValueError,match='MISSING'):audit_saved(out)


@pytest.mark.parametrize('tamper',[None,'task','event','source'])
def test_complete_synthetic_saved_byte_audit_and_tampering(tmp_path,monkeypatch,tamper):
    port,out,_,_,_=fake_cost_run(tmp_path,monkeypatch);port.costs();writer=port.writer;writer.observer=None
    m=contract.preparation();m['source_commit']='a'*40
    g={'schema':'track_a_h6_technical_pilot_authorization_v2','kind':contract.KIND,'source_commit':'a'*40,
        'approved_by_user':True,'manifest_digest':digest(m),'one_shot':True,'retry':False,'resume':False}
    for name,obj in [('frozen_pilot.json',m),('authorization.json',g),('authorization_source.json',g),
        ('launch_binding.json',{'manifest_digest':digest(m),'authorization_digest':digest(g)})]:writer.write(name,obj)
    sha=hashlib.sha256((out/'authorization_source.json').read_bytes()).hexdigest()
    writer.write('worker_claim.json',{'authorization_sha256':sha})
    flags={'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
    worker=dict(flags,status='H6_TECHNICAL_PILOT_COMPLETE',source_commit='a'*40,authorization_sha256=sha,
        correctness_completed=7,compiled_wrappers=36,calls_attempted=dict(port.calls.used))
    writer.write('worker_terminal.json',worker);writer.write('terminal_status.json',dict(flags,status='H6_TECHNICAL_PILOT_COMPLETE'))
    for name in ['input_reference.json','primitive_validation.json','parallel_resource_receipt.json']:
        writer.write(name,{'synthetic_only':True})
    bounds=synthetic_bounds()
    writer.write('actual_prepared_representation.json',{'synthetic_only':True,'bounds':bounds})
    binding.assert_actual_coverage(contract.coverage(),bounds,writer)
    for p in ('input_reference','correctness'):writer.write('phase_'+p+'.json',{'phase':p,'elapsed':0.})
    for c in contract.plan()['cells']:writer.write(c['id']+'_correctness.json',{'synthetic_only':True})
    (out/'worker.log').write_text('SYNTHETIC ONLY\n')
    if tamper:
        path=out/('worker_terminal.json' if tamper=='source' else 'wrapper_00.json')
        row=json.loads(path.read_text())
        if tamper=='source':row['source_commit']='b'*40
        elif tamper=='task':row['task']['axis']='altered'
        else:row['event_digest']='altered'
        path.write_text(json.dumps(row))
        with pytest.raises(ValueError,match='PILOT_AUDIT'):audit_saved(out)
    else:
        report=audit_saved(out);assert report['status']=='SAVED_PILOT_BYTES_PASS' and report['missing_records']==[]
        assert not report['scientific_result_recomputed'] and report['mandatory_stop']


def test_watchdog_requires_all_counts_and_stop_flags(tmp_path):
    out=tmp_path/'dummy_complete';out.mkdir();caps=deepcopy(contract.plan()['caps_proposed'])
    caps['phase_wall_seconds']={p:10 for p in contract.PHASES};caps['total_wall_seconds']=30
    flags={'status':'H6_TECHNICAL_PILOT_COMPLETE','correctness_completed':7,'compiled_wrappers':36,
        'calls_attempted':{'compile':36,'trajectory':4,'occurrence':8},'mandatory_stop':True,'next_stage_authorized':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','N':None,'G':None,
        'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED'}
    code='from pathlib import Path;import json;out=Path('+repr(str(out))+');'
    code+=';'.join('(out/'+repr('phase_'+p+'.json')+').write_text('+repr(json.dumps({'phase':p,'elapsed':0.}))+')' for p in contract.PHASES)
    code+=';(out/"worker_terminal.json").write_text('+repr(json.dumps(flags))+')'
    r=supervise([sys.executable,'-c',code],out,caps=caps)
    assert r['status']=='H6_TECHNICAL_PILOT_COMPLETE' and r['mandatory_stop']
