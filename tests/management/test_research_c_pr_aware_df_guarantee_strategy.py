"""Invented circuits only;no saved scientific QPY/numeric arrays."""
from copy import deepcopy
from fractions import Fraction as F
import math
from pathlib import Path
import sys

import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_fixed_certificate as v
from trottertracks import c_pr_aware_df_guarantee_strategy as s
from trottertracks import c_pr_aware_df_zsxx_enclosure as z


@pytest.mark.parametrize('theta',[0.,-0.,math.ulp(0.),-math.ulp(0.),.3,math.pi,
    math.pi+math.ulp(math.pi),1e20,-1e100,float.fromhex('0x1.fffffffffffffp+1023')])
def test_actual_Rust_RZ_capture_never_becomes_uniform(theta):
    r=s.rz_capture(theta)
    assert r['status']=='VALIDATED_CAPTURED_RUST_RZ' and r['uniform_matrix_error_a'] is None
    assert F(r['operator_distance']['upper_exact'])>=0
    assert r['theta_hex']==theta.hex()


@pytest.mark.parametrize('kind',['few_ULP','random_sampling_max_ULP','captured_max','ideal_template_image','held_output_box'])
def test_unknown_ULP_and_non_Rust_images_are_rejected(kind):
    with pytest.raises(ValueError,match='UNIFORM_RUST'):s.uniform_model_status(dict(kind=kind))
    assert s.uniform_model_status(None)['uniform_a'] is None


@pytest.mark.parametrize('name,params',[('h',[]),('s',[]),('sdg',[]),('cx',[]),('rzz',[.31]),
    ('crz',[.31]),('crzz',[.31]),('xx_plus_yy',[.31,-.17])])
def test_independent_exact_local_meaning_matches_artificial_Qiskit_diagnostic(name,params):
    from flint import ctx
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import RZZGate,XXPlusYYGate
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    lock=SharedQiskitCalls(lambda k,l,f:f())
    def build():
        q=QuantumCircuit(v.ARITY[name])
        if name=='crzz':q.append(RZZGate(params[0]).control(1,annotated=False),[0,1,2])
        elif name=='xx_plus_yy':q.append(XXPlusYYGate(*params),[0,1])
        else:getattr(q,name)(*params,*range(v.ARITY[name]))
        return q
    q=lock('kernel_build','artificial local',build)
    r=v.record(q)
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        A=v._local(name,params);D=A-z._matrix(Operator(q).data)
        assert F(z._norm(D)['upper_exact'])<F(1,10**13)
    assert r['gates'][0]['name']==name


def invented():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZZGate,XXPlusYYGate
    q=QuantumCircuit(3);q.global_phase=.271
    q.ry(.31,0);q.h(2);q.cx(2,0);q.append(XXPlusYYGate(.7,-.19),[0,1])
    q.crz(.21,2,1);q.rzz(-.17,0,2)
    q.append(RZZGate(.13).control(1,annotated=False),[1,0,2])
    q.p(.31,2);return q


def test_artificial_whole_P0_full_basis_phase_and_actual_adjoint():
    from trottertracks.c_pr_aware_df_c2a import trace_artificial
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from trottertracks import c1_stage2_native as native
    lock=SharedQiskitCalls(lambda k,l,f:f());q=lock('kernel_build','artificial whole',invented)
    compiled,_=trace_artificial(q)
    a,b=v.record(q),v.record(compiled)
    bound=v.compare(a,b,parallel=False)
    assert bound['dimension']==bound['columns_evaluated']==8
    assert F(bound['upper_exact'])<F(1,10**11) and bound['whole_law_bound'] is None
    with lock.adjoint_builds():actual=native.actual_adjoint(compiled)
    inverse=v.compare(v.adjoint(b),v.record(actual),parallel=False)
    assert F(inverse['upper_exact'])<F(1,10**11)
    bad=deepcopy(b);bad['phase_hex']=(float.fromhex(b['phase_hex'])+.1).hex()
    assert F(v.compare(a,bad,parallel=False)['lower_exact'])>F(1,100)


def test_recentered_long_noncommuting_word_keeps_rigorous_evaluation_radius():
    r=dict(qubits=2,phase_hex=.1.hex(),global_phase_fit=False,
        gates=[dict(name='rz',qubits=[0],params_hex=[.13.hex()]),
               dict(name='sx',qubits=[0],params_hex=[]),
               dict(name='cx',qubits=[0,1],params_hex=[])]*128)
    bound=v.compare(r,r,parallel=False)
    assert F(bound['upper_exact'])<F(1,10**40)
    assert any(F(c['evaluation_radius_left_exact'])>0 for c in bound['columns'])


def test_full_operator_certificate_contains_known_exact_phase_distance():
    from flint import ctx,arb
    left=v.identity(2);right=dict(left,phase_hex=.2.hex())
    r=v.compare(left,right,parallel=False)
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        exact=2*(z._real(.2)/2).sin()
        assert F(r['lower_exact'])<=z._fraction(exact.lower())
        assert z._fraction(exact.upper())<=F(r['upper_exact'])


def test_ancilla_only_reduction_is_analytic_full_embedding_not_sector_projection():
    from qiskit import QuantumCircuit
    q=QuantumCircuit(9);q.h(8);q.sdg(8)
    record=v.record(q,reduce_ancilla=True)
    assert record['qubits']==1 and record['embedded_qubits']==9
    q.rz(.2,0)
    with pytest.raises(ValueError,match='ANCILLA_ONLY'):v.record(q,reduce_ancilla=True)


def test_unknown_gate_open_control_nonfinite_and_phasefit_fail():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from qiskit.circuit.library import CRZGate
    q=QuantumCircuit(2);q.append(CRZGate(.1,ctrl_state=0),[0,1])
    with pytest.raises(ValueError,match='OPEN_CONTROL'):v.record(q)
    q=QuantumCircuit(1);q.append(Gate('h',1,[]),[0])
    with pytest.raises(ValueError,match='CLASS_MISMATCH'):v.record(q)
    q=QuantumCircuit(1);q.rx(.1,0)
    with pytest.raises(ValueError,match='UNSUPPORTED_FIXED_GATE'):v.record(q)
    r=v.identity(1);r['global_phase_fit']=True
    with pytest.raises(ValueError,match='SCOPE'):v.compare(r,r)


def test_statistical_zero_failure_formula_requires_explicit_IID_and_holdout():
    assert s.zero_failure_tail_limit(100,'1/100',iid=False,fresh_holdout=True,certified_decisions=True)['status']=='UNRESOLVED'
    r=s.zero_failure_tail_limit(100,'1/100',iid=True,fresh_holdout=True,certified_decisions=True)
    q=F(r['tail_probability_upper_exact'])
    assert F(4,100)<q<F(5,100) and (1-q)**100<=F(1,100)
    assert r['not_uniform'] and not r['PCG64_IID_verified']


def test_finite_execution_list_does_not_cover_unseen_trajectory_or_IID():
    assert s.finite_claim(['a'],['a','b'])['missing_ids']==['b']
    r=s.finite_claim(['a'],['a','a']);assert r['status']=='FINITE_EXECUTION_SCOPE_ONLY'
    assert r['IID_law_preserved'] is r['whole_law_bound'] is None


def synthetic_records():
    from trottertracks.c_pr_aware_df_guarantee_execution import _shell
    kernel=dict(v.identity(2),gates=[dict(name='rz',qubits=[0],params_hex=[.31.hex()]),
        dict(name='cx',qubits=[0,1],params_hex=[])])
    records=dict(kernel_raw=kernel,kernel_compiled=kernel,
        kernel_raw_adjoint=v.adjoint(kernel),kernel_compiled_adjoint=v.adjoint(kernel))
    for label in ['head','X','Y']:
        for kind in ['raw','compiled']:records[label+'_'+kind]=_shell(label)
    return records


def test_synthetic_pilot_E2E_exact19calls_and_independent_scope(tmp_path):
    from trottertracks import c_pr_aware_df_guarantee_execution as e
    from trottertracks.c1_stage3_integrated import Calls
    r=synthetic_records();c=dict(circuits={k:k for k in r},expected_qubits=2,allowance_exact='1/100000000')
    call=Calls(tmp_path,e.LIMITS,e.MAX_CALLS)
    def loader(path,**kw):return r[path.name]
    result=e.execute(tmp_path,c,tmp_path,call,loader=loader,compare=lambda a,b:v.compare(a,b,parallel=False))
    assert result['status']=='FIXED_B1_CIRCUIT_PILOT_COMPLETE'
    assert sum(call.counts.values())==19 and call.counts==e.LIMITS
    assert result['whole_law_bound'] is result['raw_Hamiltonian_reference_bound'] is None


def test_synthetic_pilot_first_correctness_failure_stops_and_preserves(tmp_path):
    from trottertracks import c_pr_aware_df_guarantee_execution as e
    from trottertracks.c1_stage3_integrated import Calls
    r=synthetic_records();r['kernel_compiled']=dict(r['kernel_compiled'],phase_hex=.1.hex())
    c=dict(circuits={k:k for k in r},expected_qubits=2,allowance_exact='1/100000000')
    call=Calls(tmp_path,e.LIMITS,e.MAX_CALLS)
    with pytest.raises(ValueError,match='FIXED_ALLOWANCE_FAIL'):
        e.execute(tmp_path,c,tmp_path,call,loader=lambda p,**kw:r[p.name],compare=lambda a,b:v.compare(a,b,parallel=False))
    assert sum(call.counts.values())==11 and call.stop.is_set()
    assert (tmp_path/'kernel_compiled_raw.json').exists()
    assert not (tmp_path/'kernel_raw_actual_adjoint.json').exists()


def test_synthetic_pilot_uncertain_enclosure_is_unresolved_not_pass(tmp_path):
    from trottertracks import c_pr_aware_df_guarantee_execution as e
    from trottertracks.c1_stage3_integrated import Calls
    r=synthetic_records();c=dict(circuits={k:k for k in r},expected_qubits=2,allowance_exact='1/100000000')
    call=Calls(tmp_path,e.LIMITS,e.MAX_CALLS)
    result=e.execute(tmp_path,c,tmp_path,call,loader=lambda p,**kw:r[p.name],
        compare=lambda *a:dict(lower_exact='0',upper_exact='1'))
    assert result['status']=='UNRESOLVED_FIXED_PILOT' and sum(call.counts.values())==11


def test_parallel_full_column_path_uses_process_local_Arb():
    # Many tiny artificial gates make the parallel path relevant without any
    # molecular input. Four basis columns are four independent tasks.
    r=dict(v.identity(2),gates=[dict(name='sx',qubits=[0],params_hex=[]),
        dict(name='rz',qubits=[1],params_hex=[.1.hex()])]*3200)
    bound=v.compare(r,r,parallel=True)
    assert bound['columns_evaluated']==4 and F(bound['upper_exact'])<F(1,10**35)


def test_passive_preflight_source_and_input_mismatch_stops_without_QPY_decode(tmp_path,monkeypatch):
    from trottertracks import c_pr_aware_df_guarantee_execution as e
    from trottertracks.c1_stage1_execution import save,sha
    from trottertracks.c_pr_aware_df_c2a import P0
    names=synthetic_records();records={k:k+'.qpy' for k in names}
    for path in records.values():(tmp_path/path).write_bytes(b'artificial hash bytes,not QPY')
    manifest=tmp_path/'SOURCE.json';save(manifest,dict(files={}))
    c=dict(scope=s.SCOPE,implementation=s.VERSION,verifier=v.VERSION,status='UNAPPROVED_FIXED_CIRCUIT_ONLY',
        executable=True,maximum_logical_units=e.MAX_CALLS,call_limits=e.LIMITS,compiler=P0,
        precision_bits=z.PRECISION,retry=0,resume=0,source_manifest='SOURCE.json',source_manifest_sha256=sha(manifest),
        packages={},external_python_source_hashes={},circuits=records,saved_inputs={p:sha(tmp_path/p) for p in records.values()})
    path=tmp_path/'contract.json';save(path,c)
    monkeypatch.setattr(e,'_read_qpy',lambda *a,**kw:pytest.fail('No scientific QPY decode'))
    assert e.preflight(tmp_path,path)['scientific_calls']==0
    (tmp_path/next(iter(records.values()))).write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):e.preflight(tmp_path,path)


def test_real_QPY_loader_on_artificial_data_only(tmp_path):
    from qiskit import qpy
    from trottertracks.c_pr_aware_df_guarantee_execution import _read_qpy
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    lock=SharedQiskitCalls(lambda k,l,f:f());q=lock('kernel_build','artificial QPY',invented)
    p=tmp_path/'artificial.qpy'
    with p.open('wb') as f:qpy.dump(q,f)
    decoded=_read_qpy(p,shell=False,expected_qubits=3)
    r=v.compare(v.record(q),decoded,parallel=False)
    assert F(r['upper_exact'])<F(1,10**40)
    with pytest.raises(ValueError,match='DIMENSION'):_read_qpy(p,shell=False,expected_qubits=9)


def test_only_registered_terminal_ancilla_measurement_is_removed(tmp_path):
    from qiskit import QuantumCircuit,qpy
    from trottertracks.c_pr_aware_df_guarantee_execution import _read_qpy,_shell
    q=QuantumCircuit(3,1);q.sdg(2);q.h(2);q.measure(2,0)
    p=tmp_path/'artificialY.qpy'
    with p.open('wb') as f:qpy.dump(q,f)
    r=_read_qpy(p,shell=True,expected_qubits=3)
    assert r['terminal_measurement_removed'] and r['ancilla_only_reduction']
    assert F(v.compare(r,_shell('Y'),parallel=False)['upper_exact'])<F(1,10**40)
    with pytest.raises(ValueError,match='TERMINAL'):v.record(q)
    bad=QuantumCircuit(3,1);bad.measure(2,0);bad.h(2)
    with pytest.raises(ValueError,match='TERMINAL'):v.record(bad,reduce_ancilla=True,strip_terminal_measurement=True)
