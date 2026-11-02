"""Rigorous captured-pair bounds on invented data;never saved H4 inputs."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_zsxx_enclosure as z
from trottertracks.c_pr_aware_df_zsxx_artificial import angle_cases,matrix_cases


def mp_matrix(M):
    import mpmath
    mp=mpmath.mp.clone();mp.dps=150
    def exact(x):
        n,d=float(x).as_integer_ratio();return mp.mpf(n)/d
    A=mp.matrix([[mp.mpc(exact(v.real),exact(v.imag)) for v in row] for row in M])
    return mp,A,exact


def mp_output(mp,record,exact):
    A=mp.eye(2);I=mp.j
    for g in record['gates']:
        name=g['gate'];t=0 if g['angle_hex'] is None else exact(float.fromhex(g['angle_hex']))
        if name=='rz':B=mp.matrix([[mp.exp(-I*t/2),0],[0,mp.exp(I*t/2)]])
        elif name=='sx':B=mp.matrix([[(1+I)/2,(1-I)/2],[(1-I)/2,(1+I)/2]])
        elif name=='x':B=mp.matrix([[0,1],[1,0]])
        else:raise AssertionError(name)
        A=B*A
    return mp.exp(I*exact(float.fromhex(record['phase_hex'])))*A


def test_import_passive_and_no_science_execute_option():
    subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_zsxx_enclosure;assert not any(x in sys.modules for x in ('numpy','qiskit','flint'))"],cwd=ROOT,check=True)
    r=subprocess.run([sys.executable,'-B','scripts/check_research_c_pr_aware_df_zsxx.py','--execute'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==2 and 'unrecognized arguments' in r.stderr


@pytest.mark.parametrize('label,t,p,l,b',angle_cases())
def test_all_emitter_boundary_cases_have_rigorous_finite_bounds(label,t,p,l,b):
    r=z.captured_angles(t,p,l,b)
    assert r['status']=='VALIDATED_REGION_ENCLOSURE'
    assert 0<=Fraction(r['point_distance']['upper_exact'])<Fraction(1,10**8)
    assert r['whole_law_bound'] is r['M_to_Euler_error'] is None
    assert not r['platform_libm_accuracy_assumed'] and not r['global_branch_coverage']
    assert r['output']['phase_fit'] is False


@pytest.mark.parametrize('label,M',matrix_cases())
def test_arb_bound_contains_independent_high_precision_spectral_diagnostic(label,M):
    r=z.captured_matrix(M);mp,A,exact=mp_matrix(M);B=mp_output(mp,r['output'],exact)
    observed=mp.svd(A-B,compute_uv=False)[0]
    def q(s):v=Fraction(s);return mp.mpf(v.numerator)/v.denominator
    bound=r['point_distance']
    assert q(bound['lower_exact'])<=observed<=q(bound['upper_exact'])
    # mpmath is an independent diagnostic,not the certificate backend.
    assert r['precision_bits']==192 and 'outward' in bound['upper_method']


def test_non_singleton_reference_region_is_not_unseen_compiler_coverage():
    r=z.captured_matrix(np.eye(2),reference_radius='1/1000000000000000')
    assert Fraction(r['region_distance']['upper_exact'])>=Fraction(2,10**15)
    assert 'single recorded binary64' in r['compiler_inputs_covered']
    assert r['whole_law_bound'] is None and not r['unobserved_reexecution_guaranteed']
    angular=z.captured_angles(.7,.3,-.7,.2,theta_reference_radius='1/1000000000000000')
    assert Fraction(angular['region_distance']['upper_exact'])>=Fraction(angular['point_distance']['lower_exact'])


def test_exact_dyadic_subnormal_and_no_silent_float_underflow():
    from flint import acb,acb_mat,ctx
    with ctx.workprec(z.PRECISION):
        smallest=math.ulp(0.);assert z._fraction(z._real(smallest))==Fraction.from_float(smallest)
        tiny=Fraction(1,1<<1100);r=z._norm(acb_mat([[acb(z._real(tiny)),0],[0,0]]))
        assert Fraction(r['upper_exact'])>=tiny>0 and float(tiny)==0.
    r=z.captured_matrix(np.array([[1,complex(-0.,-0.)],[0,1]],complex))
    assert r['input_hex'][0][1]==['-0x0.0p+0','-0x0.0p+0']


def test_invalid_or_uncovered_inputs_never_reach_Rust(monkeypatch):
    from qiskit._accelerate import euler_one_qubit_decomposer as rust
    monkeypatch.setattr(rust,'unitary_to_gate_sequence',lambda *a:pytest.fail('Do not call Rust on invalid domain'))
    for M in (np.full((2,2),np.nan),np.full((2,2),np.inf),np.eye(3)):
        with pytest.raises(ValueError,match='FINITE_2X2'):z.captured_matrix(M)
    assert z.captured_matrix(np.zeros((2,2)))['status']=='UNRESOLVED'
    with pytest.raises(ValueError):z.captured_angles(float('nan'),0,0,0)


def test_unverified_validation_runtime_and_phase_fit_rejected(monkeypatch):
    import flint
    monkeypatch.setattr(flint,'__FLINT_VERSION__','unverified')
    with pytest.raises(ValueError,match='RUNTIME'):z.runtime_model()
    monkeypatch.undo()
    r=z.captured_matrix(np.eye(2));bad=deepcopy(r['output']);bad['phase_fit']=True
    with pytest.raises(ValueError,match='PHASE_FIT'):z.verify_fixed_record(np.eye(2),bad)


def test_unsupported_rounding_model_is_not_silently_adopted(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from unittest.mock import Mock
    monkeypatch.setattr(ctypes,'CDLL',lambda *a:SimpleNamespace(fegetround=Mock(return_value=1)))
    with pytest.raises(ValueError,match='ROUND_TO_NEAREST'):z.runtime_model()


def test_branch_output_patterns_not_just_small_sample_maximum():
    records={name:z.captured_angles(t,p,l,b) for name,t,p,l,b in angle_cases()}
    names=lambda key:[x['gate'] for x in records[key]['output']['gates']]
    assert 'sx' not in names('theta_zero_cut_below') and names('theta_zero_cut_above').count('sx')==2
    assert names('theta_halfpi_cut_below').count('sx')==1 and names('theta_halfpi_cut_above').count('sx')==2
    assert 'x' in names('theta_pi')
    assert names('RZ_drop_below')==[] and names('RZ_drop_above')==['rz']
    clamp=records['mod_endpoint_clamp'];assert float.fromhex(clamp['output']['gates'][0]['angle_hex'])==-math.pi
    assert Fraction(clamp['point_distance']['lower_exact'])>0


def test_phase_loss_is_detected_and_controlled_XY_changes():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    r=z.captured_matrix(1j*np.eye(2));bad=deepcopy(r['output']);bad['phase_hex']=(0.).hex()
    distance=z.verify_fixed_record(1j*np.eye(2),bad)
    assert Fraction(distance['lower_exact'])>1
    lock=SharedQiskitCalls(lambda k,l,f:f())
    def signals(record):
        def build():
            q=QuantumCircuit(2);q.h(1);q.p(float.fromhex(record['phase_hex']),1)
            # This chosen captured iI has no system gates;phase is relative.
            assert not record['gates'];return q
        q=lock('kernel_build','invented controlled phase',build);values=[]
        for axis in ('X','Y'):
            w=q.copy()
            if axis=='Y':w.sdg(1)
            w.h(1);s=Statevector.from_instruction(w).data
            values.append(sum(abs(s[:2])**2)-sum(abs(s[2:])**2))
        return values
    assert signals(r['output'])==pytest.approx([0,1],abs=1e-14)
    assert signals(bad)==pytest.approx([1,0],abs=1e-14)


def test_long_run_separates_input_error_and_never_double_counts():
    r=z.captured_run([('rz',.13),('sx',None),('rz',-.09)]*256)
    assert r['run_length']==768 and Fraction(r['ideal_product_to_output']['upper_exact'])<Fraction(1,10**8)
    assert Fraction(r['triangle_upper_exact'])==Fraction(r['input_product_error']['upper_exact'])+Fraction(r['point_distance']['upper_exact'])
    assert 'not Rust' in r['input_product_model']


def test_captured_whole_P0_initial_translation_phase_and_actual_inverse():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from trottertracks.c_pr_aware_df_c2a import trace_artificial
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from trottertracks import c1_stage2_native as native
    gs=[('ry',.31),('p',-.17),('sx',None),('rz',.23)]
    q=QuantumCircuit(1);q.global_phase=-.19
    for k,v in gs:getattr(q,k)(0) if v is None else getattr(q,k)(v,0)
    compiled,_=trace_artificial(q);r=z.captured_circuit(gs,compiled,input_phase=-.19)
    assert Fraction(r['direct_distance']['upper_exact'])<Fraction(1,10**8) and r['whole_law_bound'] is None
    lock=SharedQiskitCalls(lambda k,l,f:f())
    with lock.adjoint_builds():adj=native.actual_adjoint(compiled)
    inverse=z.captured_inverse(r['output'],adj)
    assert Fraction(inverse['direct_distance']['upper_exact'])<Fraction(1,10**8)
    assert np.max(abs(Operator(adj).data@Operator(compiled).data-np.eye(2)))<1e-10


CONDITIONS=dict(basis_closed=True,target_none=True,zsxx_only=True,parameter_free_cancellation=True,no_calibrations=True)


def test_accepted_only_composition_unknown_reference_and_sixN_conditions():
    r=z.captured_run([('rz',.13),('sx',None),('rz',-.09)]*2)
    accepted=dict(accepted=True,input_gates=6,status=r['status'],phase_fit=False,
        bound_scope='direct_ideal_run_to_captured_output',upper_exact=r['ideal_product_to_output']['upper_exact'])
    rejected=dict(accepted=False,input_gates=10000,status='NUMERICAL_DIAGNOSTIC_ONLY',upper_exact='999')
    k=z.accepted_word_budget(10,[accepted,rejected],conditions=CONDITIONS,initial_translation='0',phase='0')
    assert k['status']=='CONDITIONAL_ENCLOSURE' and k['accepted_gate_visits']==6
    assert k['captured_word_upper_exact']==accepted['upper_exact'] and k['whole_law_bound'] is None
    assert z.accepted_word_budget(10,[accepted],conditions=CONDITIONS)['status']=='UNRESOLVED'
    bad=dict(CONDITIONS,target_none=False)
    assert z.accepted_word_budget(10,[accepted],conditions=bad)['status']=='UNRESOLVED'
    accepted['input_gates']=61
    with pytest.raises(ValueError,match='COVERAGE_COUNT'):z.accepted_word_budget(10,[accepted],conditions=CONDITIONS)


@pytest.mark.parametrize('reverse',[False,True])
def test_legacy_failure_orders_use_primary_journal_not_surface_exception(tmp_path,monkeypatch,reverse):
    # Deterministic coverage of both legacy as_completed error orders.
    from concurrent.futures import Future
    from trottertracks import c_pr_aware_df_stage_c2 as legacy
    from trottertracks.c1_stage3_integrated import Calls,Stopped
    from test_research_c_pr_aware_df_stage_c2 import artificial_saved
    records=dict(C1_result=dict(status='C1_SAVED_SCALAR_SCENARIOS_COMPLETE'),
        C1_SOURCE=dict(execution_commit='b1c735da394d4b0a444cbc618ead9781ef4d9134'),baseline_tail=artificial_saved(),proposed_tail=artificial_saved())
    records['baseline_tail']['components'][0]['probability']*=2;c=dict(records={})
    for k,val in records.items():legacy.io.save(tmp_path/(k+'.json'),val);c['records'][k]=k+'.json'
    class Executor:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,fn,*args):
            f=Future()
            try:f.set_result(fn(*args))
            except BaseException as err:f.set_exception(err)
            return f
    import concurrent.futures
    monkeypatch.setattr(concurrent.futures,'ThreadPoolExecutor',Executor)
    monkeypatch.setattr(concurrent.futures,'as_completed',lambda fs:list(reversed(fs)) if reverse else fs)
    out=tmp_path/'out';out.mkdir();call=Calls(out,legacy.LIMITS,legacy.MAX_CALLS)
    with pytest.raises((ValueError,Stopped)):legacy.execute(tmp_path,c,out,call)
    assert call.stop.is_set() and call.first_failure['kind']=='table_replay'
    assert call.first_failure['error_type']=='ValueError' and 'REPLAY' in call.first_failure['error']


def test_contract_preflight_blocks_science_and_checks_identity(tmp_path,monkeypatch):
    from trottertracks import c_pr_aware_df_zsxx_contract as contract
    from trottertracks.c_pr_aware_df_c2a import P0
    s=tmp_path/'source.json';contract.io.save(s,dict(files={}))
    original=tmp_path/'saved.npz';original.write_bytes(b'synthetic bytes only')
    c=dict(scope=contract.SCOPE,implementation=z.VERSION,executable=False,
        status='BLOCKED_PARTIAL_VALIDATED_DOMAINS',maximum_logical_units=None,
        precision_bits=z.PRECISION,retry=0,resume=0,source_manifest='source.json',
        source_manifest_sha256=contract.io.sha(s),saved_inputs={'saved.npz':contract.io.sha(original)},packages={},
        compiler=P0,external_python_source_hashes={})
    p=tmp_path/'contract.json';contract.io.save(p,c)
    monkeypatch.setattr(np,'load',lambda *args,**kwargs:pytest.fail('No array decode in preflight'))
    assert contract.preflight(tmp_path,p)['maximum_logical_units'] is None
    original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):contract.preflight(tmp_path,p)


def test_non_adopted_model_cannot_turn_diagnostic_into_enclosure():
    row=dict(accepted=True,input_gates=1,status='NUMERICAL_DIAGNOSTIC_ONLY',phase_fit=False,
        bound_scope='direct_ideal_run_to_captured_output',upper_exact='1/1000000000000')
    assert z.accepted_word_budget(10,[row],conditions=CONDITIONS,initial_translation='0',phase='0')['status']=='UNRESOLVED'
    row['status']='VALIDATED_REGION_ENCLOSURE';row['bound_scope']='event_P0_sample_max'
    assert z.accepted_word_budget(10,[row],conditions=CONDITIONS,initial_translation='0',phase='0')['status']=='UNRESOLVED'
