"""Artificial Q2 step-switch/oracle and four-compile integration checks."""
import sys
from pathlib import Path
import numpy as np
import pytest
from qiskit.quantum_info import Operator,Pauli,Statevector

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_b2_q2_streaming_h4 as m
from b2_cost_method_cases import artificial_round,event_pattern,expected_controlled


@pytest.fixture(autouse=True)
def no_molecular_io(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('No saved numerical inputs, m0 execution or solver')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(m.previous,'git',forbidden)
    monkeypatch.setattr(m.m0,'run',forbidden)
    import trotterlib.df_hamiltonian as h
    monkeypatch.setattr(h,'solve_df_ground_state',forbidden)
    monkeypatch.setattr(h,'build_df_h_d_from_molecule',forbidden)


def case():
    h,r=artificial_round(1,3)
    e0,e4=tuple(event_pattern(r,'mixed'))[:2]
    events=m.two_step_events((e0,e4,e4,e0))
    request=r.explicit_request(events)
    ref=r.native(request,max_instructions=m.m0.request_instruction_bound(request))
    stream=m.m0.stream_native(r,iter(events))
    evolutions={'reference':ref,'stream':stream}
    wrappers={name:{axis:m.previous.connection.build_native_hadamard_wrapper(
        evo,axis=axis,max_instructions=len(evo.circuit.data)+4)
        for axis in ('cosine','sine')} for name,evo in evolutions.items()}
    return h,r,events,evolutions,wrappers


def test_fixed_second_step_is_different_and_not_a_palindrome_reversal():
    assert m.two_step_events((0,4,4,0))==(0,4,4,0,4,0,0,4)


@pytest.mark.parametrize('bad',[(0,4,4),(0,4,0,4)])
def test_wrong_first_step_is_rejected(bad):
    with pytest.raises(ValueError,match='FIRST_STEP'):m.two_step_events(bad)


def test_q2_boundary_controlled_phase_order_and_both_measurement_axes():
    h,r,events,evolutions,wrappers=case()
    expected=expected_controlled(h,r,events)
    assert events[:4]!=events[4:]
    assert [e.taylor_order for e in events]==[0,2,2,0,2,0,0,2]
    assert evolutions['reference'].circuit_fingerprint==evolutions['stream'].circuit_fingerprint
    for evolution in evolutions.values():
        assert evolution.q==2 and evolution.T==.04
        assert np.max(abs(Operator(evolution.circuit).data-expected))<1e-10
    psi=np.array([.2,.3j,-.4,.5j]);psi/=np.linalg.norm(psi)
    signal=np.vdot(psi,expected[4:,4:]@psi)
    for axis,target in (('cosine',signal.real),('sine',signal.imag)):
        for name in evolutions:
            wrapper=wrappers[name][axis]
            assert wrapper.count_ops()['measure']==1
            measured=Statevector(np.r_[psi,np.zeros(4)]).evolve(
                wrapper.remove_final_measurements(inplace=False)).expectation_value(Pauli('ZII')).real
            assert abs(measured-target)<1e-10


def test_four_compiles_compare_against_new_q2_reference(tmp_path):
    _,r,_,evolutions,wrappers=case()
    result={'compile_calls_started':0,'axes':{'reference':{},'stream':{}}}
    m.compile_both_paths(result,evolutions,wrappers,r.context.compiler,tmp_path,'ARTIFICIAL_SOURCE')
    assert result['compile_calls_started']==4
    for axis in ('cosine','sine'):
        ref=result['axes']['reference'][axis];stream=result['axes']['stream'][axis]
        assert stream['comparison']['status']=='EXACT_MATCH'
        assert ref['compiled_circuit_fingerprint']==stream['compiled_circuit_fingerprint']
        assert all(v==0 for v in stream['comparison']['signed_difference'].values())


def test_compiler_failure_stops_before_any_additional_call(tmp_path):
    _,r,_,evolutions,wrappers=case()
    result={'compile_calls_started':0,'axes':{'reference':{},'stream':{}}}
    calls=[]
    def failed(*a,**k):
        calls.append(1);raise RuntimeError('synthetic compiler failure')
    with pytest.raises(RuntimeError,match='synthetic compiler failure'):
        m.compile_both_paths(result,evolutions,wrappers,r.context.compiler,tmp_path,
                             'ARTIFICIAL_SOURCE',compile_function=failed)
    assert len(calls)==result['compile_calls_started']==1
