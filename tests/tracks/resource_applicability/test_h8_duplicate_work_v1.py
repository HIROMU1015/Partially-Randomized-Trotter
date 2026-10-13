"""Artificial byte/state/process-slot checks; sampling/real transpile forbidden."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.library import RZGate

from trotterlib.df_trotter.circuit import simulate_statevector
from trotterlib.rte_compiled_cost import _circuit_payload
from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as original
from trottertracks.resource_applicability.h8_hash_stream_reuse_v1 import StreamingFingerprintReuse, use_streaming_hash_reuse
from trottertracks.resource_applicability.h8_control_reuse_v1 import hadamard_states_from_branches, validate_control
from trottertracks.resource_applicability.h8_fast_parallel_plan_v1 import case_waves, classify, resource_proposal
from trottertracks.resource_applicability.ax2b_limits import CallBudget
from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution, build_native_hadamard_wrapper
from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import hadamard_expectation


@pytest.fixture(autouse=True)
def artificial_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('NO_REAL_INPUT_SAMPLING_TRANSPILE')
    monkeypatch.setattr(np, 'load', forbidden)
    import qiskit
    from trotterlib import rte_compiled_cost as compile_module
    monkeypatch.setattr(qiskit, 'transpile', forbidden)
    monkeypatch.setattr(compile_module, 'transpile', forbidden)


def fingerprint_fixture(kind):
    c = QuantumCircuit(17, 2); c.global_phase = -.17
    d = QuantumCircuit(1); d.rz(.23, 0); d.p(-.19, 0); d.global_phase = .31
    gate = Gate('shared', 1, []); gate.definition = d
    if kind == 'matrix':
        a = np.arange(256*256).reshape(256,256).astype(complex)*.001
        a[0,0] = complex(-.0, .0)
        c.append(Instruction('artificial_complex_parameter',8,0,[a]),range(8))
    elif kind == 'control':
        c.append(RZGate(.17).control(ctrl_state=0),[0,1]); c.append(gate.control(),[1,2])
    elif kind == 'conditional':
        c.rz(.27,1).c_if(c.clbits[1],1); c.x(2).c_if(c.clbits[0],1)
    else:
        for _ in range(12):
            c.append(gate,[0],copy=False)
        c.cx(0,1);c.h(16);c.measure(16,1)
    return c


@pytest.mark.parametrize('kind',['nested','matrix','control','conditional'])
def test_exact_canonical_bytes_and_digest(kind):
    c = fingerprint_fixture(kind); expected = json.dumps(_circuit_payload(c),sort_keys=True,separators=(',',':')).encode()
    cache = StreamingFingerprintReuse(); data = bytearray(); cache.write_bytes(c,data.extend)
    assert data == expected
    for _ in range(2):
        assert cache.fingerprint(c) == original(c) == hashlib.sha256(expected).hexdigest()
    assert cache.stats['top_level_prepasses'] == 0
    assert cache.stats['operation_byte_hits'] > 0
    assert cache.stats['canonical_instruction_visits'] >= len(c.data)*3


@pytest.mark.parametrize('change',['phase','definition','parameter','control','array','dtype','shape','signed_zero','qubits','clbits','condition','order'])
def test_mutations_never_reuse_stale_digest(change):
    c = fingerprint_fixture('matrix' if change in ('array','dtype','shape','signed_zero') else
                            'control' if change=='control' else 'conditional' if change=='condition' else 'nested')
    cache = StreamingFingerprintReuse(); before = cache.fingerprint(c)
    if change == 'phase':c.global_phase += .01
    elif change == 'definition':c.data[0].operation.definition.x(0)
    elif change == 'parameter':
        d=c.data[0].operation.definition; item=d.data[0];op=item.operation.copy();op.params=[.251];d.data[0]=item.replace(operation=op)
    elif change == 'control':c.data[0].operation.ctrl_state=1
    elif change == 'array':c.data[0].operation.params[0][1,1]+=.001j
    elif change == 'dtype':c.data[0].operation.params=[c.data[0].operation.params[0].astype(np.complex64)]
    elif change == 'shape':c.data[0].operation.params=[c.data[0].operation.params[0].reshape(128,512)]
    elif change == 'signed_zero':c.data[0].operation.params[0][0,0]=complex(.0,.0)
    elif change == 'qubits':c.data[0]=c.data[0].replace(qubits=(c.qubits[1],))
    elif change == 'clbits':c.data[-1]=c.data[-1].replace(clbits=(c.clbits[0],))
    elif change == 'condition':c.data[0].operation.condition=(c.clbits[0],1)
    else:c.data=list(reversed(c.data))
    assert cache.fingerprint(c) == original(c) != before


@pytest.mark.parametrize('kind',['symbolic','nan','array_nan','object','layout','calibration','recursive'])
def test_invalid_graph_not_hidden_by_cached_payloads(kind):
    c=fingerprint_fixture('nested');cache=StreamingFingerprintReuse();cache.fingerprint(c)
    if kind=='symbolic':c.rz(Parameter('x'),0)
    elif kind=='nan':c.append(Instruction('bad',1,0,[float('nan')]),[0])
    elif kind=='array_nan':c.append(Instruction('bad',1,0,[np.array([float('nan')])]),[0])
    elif kind=='object':c.append(Instruction('bad',1,0,[np.array(['x'],dtype=object)]),[0])
    elif kind=='layout':c._layout=object()
    elif kind=='calibration':c._calibrations={'x':{'synthetic':True}}
    else:c.data[0].operation.definition.append(c.data[0].operation,[0],copy=False)
    with pytest.raises((ValueError,TypeError)):
        cache.fingerprint(c)


@pytest.mark.parametrize('cap',[0,2**20,32*2**20])
def test_bounded_payload_fallback(cap):
    cache=StreamingFingerprintReuse(byte_cap=cap);c=fingerprint_fixture('control')
    for i in range(4):
        c.rz(.01*i,3);assert cache.fingerprint(c)==original(c)
        assert cache.cached_bytes == sum(map(len,cache.entries.values())) <= cap
        assert cache.snapshot()['payload_peak_reserved_bytes'] <= cap
        assert cache.reserved_bytes == 0
    cache.clear();assert not cache.entries


def test_alias_restoration_and_nested_scope_rejected():
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    with pytest.raises(RuntimeError):
        with use_streaming_hash_reuse() as cache:
            assert native.canonical_qiskit_circuit_fingerprint.__self__ is cache
            with pytest.raises(ValueError,match='ALIAS_ALREADY_CHANGED'):
                with use_streaming_hash_reuse():pass
            raise RuntimeError('artificial failure')
    assert native.canonical_qiskit_circuit_fingerprint is original
    assert not cache.entries


def test_changed_builtin_definition_method_is_rechecked(monkeypatch):
    c=QuantumCircuit(1);c.rz(.17,0);cache=StreamingFingerprintReuse()
    before=cache.fingerprint(c)
    def alternate(op):
        d=QuantumCircuit(1);d.x(0);op.definition=d
    monkeypatch.setattr(RZGate,'_define',alternate)
    changed=QuantumCircuit(1);changed.rz(.17,0)
    assert cache.fingerprint(changed)==original(changed)!=before


def test_explicit_builtin_definition_is_not_assumed_default():
    c=QuantumCircuit(1);c.rz(.17,0);cache=StreamingFingerprintReuse();before=cache.fingerprint(c)
    op=RZGate(.17);d=QuantumCircuit(1);d.x(0);op.definition=d
    changed=QuantumCircuit(1);changed.append(op,[0])
    assert cache.fingerprint(changed)==original(changed)!=before


@pytest.mark.parametrize('n',[2,4,16])
@pytest.mark.parametrize('mix_ancilla',[False,True])
def test_branch_reuse_matches_actual_full_XY_wrappers(n,mix_ancilla):
    c=QuantumCircuit(n+1);c.global_phase=.271
    c.h(0);c.rz(-.29,1);c.cx(n,0);c.cp(.71,n,1);c.p(-.17,n)
    if mix_ancilla:c.rx(.17,n)
    e=NativeEvolution(c,n,.8,1,'artificial','ordinary',1000,original(c))
    rng=np.random.default_rng(1729);psi=rng.normal(size=1<<n)+1j*rng.normal(size=1<<n);psi/=np.linalg.norm(psi)
    low=np.concatenate((psi,np.zeros_like(psi)));high=np.concatenate((np.zeros_like(psi),psi))
    a=simulate_statevector(c,low);b=simulate_statevector(c,high)
    actual=hadamard_states_from_branches(a,b,n)
    for axis in ('cosine','sine'):
        w=build_native_hadamard_wrapper(e,axis=axis,include_measurement=False,max_instructions=1000)
        expected=simulate_statevector(w,low)
        assert np.linalg.norm(actual[axis]-expected)<2e-12
        assert abs(hadamard_expectation(actual[axis],n)-hadamard_expectation(expected,n))<2e-12


@pytest.mark.parametrize('bad',['shape','nonfinite','n'])
def test_branch_reuse_rejects_invalid_states(bad):
    a=np.ones(8,dtype=complex);b=a.copy();n=2
    if bad=='shape':b=b[:-1]
    elif bad=='nonfinite':b[0]=np.inf
    else:n=17
    with pytest.raises(ValueError):hadamard_states_from_branches(a,b,n)


def test_control_budget_and_phase_preserved():
    n=3;c=QuantumCircuit(n+1);c.cp(.71,n,1);c.crx(-.3,n,0);c.p(.19,n)
    e=NativeEvolution(c,n,.8,1,'artificial','ordinary',1000,original(c))
    rng=np.random.default_rng(19);psi=rng.normal(size=1<<n)+1j*rng.normal(size=1<<n);psi/=np.linalg.norm(psi)
    budget=CallBudget(control_probe=4);calls=[]
    def run(c,state):calls.append(1);return simulate_statevector(c,state)
    evidence,b=validate_control(e,psi,calls=budget,simulate=run,expectation=hadamard_expectation)
    assert evidence['max_error']<1e-12 and len(calls)==2 and budget.used['control_probe']==4
    assert evidence['validation_method_changed'] is True
    budget=CallBudget(control_probe=4)
    evidence,_=validate_control(e,psi,calls=budget,simulate=run,expectation=hadamard_expectation,
        expected_signal=evidence['signal'],previous_high=b)
    assert evidence['max_error']<1e-12


def test_failed_control_does_not_report_success():
    n=2;c=QuantumCircuit(n+1);c.x(0);e=NativeEvolution(c,n,.8,1,'artificial','ordinary',1000,original(c))
    psi=np.zeros(4,complex);psi[0]=1
    evidence,_=validate_control(e,psi,calls=CallBudget(control_probe=4),simulate=simulate_statevector,
                                 expectation=hadamard_expectation)
    assert evidence['max_error']>1
    with pytest.raises(RuntimeError,match='CALL_BUDGET'):
        validate_control(e,psi,calls=CallBudget(control_probe=1),simulate=simulate_statevector,expectation=hadamard_expectation)


def test_metadata_waves_preserve_all_tasks_and_seeds():
    rows=[dict(index=i,task=dict(seed=100+i,cell_id='artificial_'+str(i))) for i in range(12)]
    receipts={i:dict(complete=True,all_reaped=True,first_stop_reason=None,source_input_compiler_bound=True,
                    peak_rss_bytes=4*2**30,peak_as_bytes=5*2**30) for i in range(8)}
    waves=case_waves(list(reversed(rows)),receipts)
    assert [len(x['registrations']) for x in waves]==[3,1,8]
    found=sorted([r for w in waves for r in w['registrations']],key=lambda r:r['index'])
    assert found==rows
    for w in waves:
        p=w['proposal'];assert p['aggregate_reservation_bytes']<=p['aggregate_RSS_bytes']
        assert p['allowed_cpus']==[] and p['approved'] is False and p['runtime_authorization'] is False
        workers={c[0] for c in p['proposed_worker_cpu_sets']}
        assert workers.isdisjoint(p['proposed_coordinator_cpu_set'])


def test_unknown_or_partial_peak_never_becomes_normal():
    assert classify(None)=='heavy'
    assert classify({'complete':False,'peak_rss_bytes':1})=='heavy'
    with pytest.raises(ValueError):resource_proposal('heavy',4)
    with pytest.raises(ValueError):resource_proposal('normal',9)
    with pytest.raises(ValueError):case_waves([{'index':0},{'index':0}])


@pytest.mark.parametrize('random',[False,True])
def test_full_cost_adapter_preserves_mock_compile_inputs_metrics_calls_and_seed(monkeypatch,random):
    from trottertracks.resource_applicability import h6_matched_cost_port_v3 as old
    from trottertracks.resource_applicability import h8_fast_cost_port_v1 as new
    from trottertracks.resource_applicability.h4_common_panel_port_v1 import bound_function
    from trotterlib import rte_compiled_cost as compilation
    c=QuantumCircuit(3);c.cx(2,0);c.crz(.17,2,1);c.p(.13,2)
    def build(*args,**kwargs):
        qc=c.copy();return NativeEvolution(qc,2,.8,1,'artificial',kwargs['control_policy'],1000,original(qc))
    prep=SimpleNamespace(exact_rte_lambda_r=.1,constant_coefficient=.13,deterministic_blocks=())
    psi=np.array([1.,1j,.2,.1j],complex);psi/=np.linalg.norm(psi)
    low=np.concatenate((psi,np.zeros_like(psi)));high=np.concatenate((np.zeros_like(psi),psi));b=simulate_statevector(c,high)
    z=complex(np.vdot(psi,b[4:]));event=SimpleNamespace(taylor_order=0,to_dict=lambda:{'artificial_event':0})
    steps=(SimpleNamespace(rte_occurrence=SimpleNamespace(events=[event])),)
    class Port:
        def __init__(self,*a,**k):
            self.plan=a[1]['plan'];self.caps=self.plan['caps_proposed']
            self.preps={};self.calls=CallBudget(compile=4,trajectory=int(random),occurrence=int(random),control_probe=8)
        def sampled_steps(self,cell,seed):
            assert seed==173;self.calls.take('trajectory');self.calls.take('occurrence');return steps
    class Writer:
        def __init__(self):self.rows={}
        def write(self,n,v,**kw):
            assert n not in self.rows;self.rows[n]=v
    class Trace:
        def stage(self,*a,**kw):return nullcontext()
    cell=dict(id='artificial',method='B2' if random else 'B1',prefix=1,q=1,formula='2nd',r=1 if random else None,
              R=1 if random else None,K=2 if random else None)
    task=dict(cell=cell,cell_id='artificial',phase='artificial_mock',replica=0,seed=173,ordinary=True,
              validate=True,expected_signal=dict(real=z.real,imag=z.imag))
    m=dict(kind='ARTIFICIAL',plan=dict(T=.8,caps_proposed={'untranspiled_instructions':1000,'transpiled_instructions':10000},
        compiler_proposed={'basis_gates':['rz','sx','x','cx'],'optimization_level':1,'seed_transpiler':17}),
        input_identity={'snapshot_path':'fixture','snapshot_receipt_path':'fixture','df_receipt_path':'fixture'},reuse_registry={'missing':[]})
    loader=lambda *a:(SimpleNamespace(n_qubits=2),None,None,None,None,None)
    bridge=lambda *a: (tuple(range(4)),psi)
    bound=lambda *a,**k:{'cells':[{'wrapper_instruction_upper_bounds':{'one':1000}}]}
    fields=dict(MatchedPort=Port,load_h6_snapshot=loader,safe_path=lambda *a:Path('fixture'),
        checked_basis_bridge=bridge,_prepare=lambda *a:prep,_prepare_discard=lambda *a:prep,actual_bounds=bound,
        build_deterministic_native=build,partial_native_from_step_requests=build)
    previous=bound_function(old.cost_task,fields)
    for name,value in [('MatchedPort',Port),('load_h8_snapshot',loader),('private_path',lambda *a:Path('fixture')),
        ('checked_basis_bridge',bridge),('_prepare',lambda *a:prep),('_prepare_discard',lambda *a:prep),
        ('actual_bounds',bound),('build_deterministic_native',build),('partial_native_from_step_requests',build)]:
        monkeypatch.setattr(new,name,value)
    captures=[]
    def fake_transpile(qc,**kw):
        captures.append((original(qc),kw.copy()));return qc.copy()
    monkeypatch.setattr(compilation,'transpile',fake_transpile)
    with patch.object(Path,'read_text',return_value='{}'):
        a=Writer();previous(Path('/home/AbeHiromu'),m,task,a,None,Trace());old_inputs=captures[:];captures.clear()
        b=Writer()
        with use_streaming_hash_reuse():new.cost_task(Path('/home/AbeHiromu'),m,task,b,None,Trace())
    assert old_inputs==captures and len(captures)==4
    assert a.rows['trajectory.json']==b.rows['trajectory.json']
    assert a.rows['cost_result.json']['calls_attempted']==b.rows['cost_result.json']['calls_attempted']
    for x,y in zip(a.rows['cost_result.json']['rows'],b.rows['cost_result.json']['rows']):
        for key in ('cell_id','phase','seed','replica','axis','control','event_digest','fingerprint','compiler_hash','metrics'):
            assert x[key]==y[key]
    assert b.rows['cost_result.json']['full_evolution_simulations']==4
    assert b.rows['cost_result.json']['expensive_full_wrapper_simulations']==0
