"""Invented two-mode circuits/means only; no real H4 data or sampling."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from test_o1_m1_matched_split import PACKAGE, adapter, toy_inputs

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_o1_m1_h4_operator_correctness as reference
validation=importlib.import_module(PACKAGE+'.o1_m1_circuit_signal_validation')
rte=importlib.import_module(PACKAGE+'.rte')
df_tail=importlib.import_module(PACKAGE+'.df_rte_tail')


@pytest.fixture(autouse=True)
def no_saved_inputs_or_sampling(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No saved molecular input or sampling in synthetic preparation')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(rte,'sample_rte_events',forbidden)
    monkeypatch.setattr(rte,'iter_sample_rte_events',forbidden)


def test_complex_signed_gaussian_controlled_and_matched_means():
    split=adapter.prepare_matched_a(toy_inputs())
    result=validation.validate_matched_circuit_signals(split,reference,
        step_times=(.1,.05),r=2,K_values=(0,2),q=2)
    assert result['status']=='PASS', [c for c in result['checks'] if not c['passed']]
    assert max(c['max_entry'] for c in result['checks']) < 2e-13
    assert result['conditions']['state_squared_norm']>1  # supplied vector unchanged
    assert {e['phase_real'] for e in result['selected_events']}=={1.,-1.}
    assert all(c['passed'] for c in result['checks'] if 'relative_phase' in c['name'])
    for record in result['records']:
        p,s=record['dictionaries']['P'],record['dictionaries']['DF']
        assert complex(p['corrected_signal']['real'],p['corrected_signal']['imag'])==pytest.approx(
            complex(s['corrected_signal']['real'],s['corrected_signal']['imag']),abs=2e-13)
        assert p['normalization']!=pytest.approx(s['normalization'],abs=1e-8)
        for item in (p,s):
            assert item['implementation_error_vs_nominal']['max_entry']<2e-13
            assert item['normalization']*item['attenuation']==pytest.approx(1.)
            assert item['raw_signal']['radius']*item['normalization']==pytest.approx(item['corrected_signal']['radius'])


def test_complete_small_event_circuit_mean_matches_existing_moments():
    split=adapter.prepare_matched_a(toy_inputs())
    extraction=adapter.extract_matched_df_tail(split,identity_policy='faithful_identity_in_tail')
    prep=df_tail.prepare_df_rte_event_inputs(extraction)
    dense=df_tail.extraction_to_normalized_rte_tail(extraction)
    config,dist=rte.make_rte_config(prep.symbolic_tail,evolution_time=.17,rte_steps=1,
                                   truncation_tolerance=1.,finite_taylor_order=2)
    events=rte.enumerate_rte_events(prep.symbolic_tail.components,dist,max_events=1000)
    builder=validation.QiskitDFRTEEventCircuitBuilder(basis_registry=prep.basis_registry)
    total=np.zeros((4,4),complex)
    controlled=np.zeros((8,8),complex)
    for event in events:
        built=builder.build_event(prep.request_for_event(event))
        built_c=builder.build_event(prep.request_for_event(event,controlled=True,ancilla_qubit=2))
        total+=event.event_probability*Operator(built.circuit).data
        controlled+=event.event_probability*Operator(built_c.circuit).data
    moments=rte.finite_rte_operator_moments(dense.normalized_hamiltonian,config)
    np.testing.assert_allclose(total,moments.attenuated_event_mean_operator,rtol=0,atol=2e-13)
    np.testing.assert_allclose(controlled,validation.controlled_reference(moments.attenuated_event_mean_operator),rtol=0,atol=2e-13)


def test_identity_one_body_phase_cannot_be_dropped_in_controlled_block():
    block=validation.build_one_body_gaussian_block(.4*np.eye(2))
    time=.3
    actual=Operator(validation.one_body_circuit(block,time,controlled=True)).data
    expected=np.diag(np.exp(-1j*time*.4*np.array([0,1,1,2])))
    np.testing.assert_allclose(actual,validation.controlled_reference(expected),rtol=0,atol=1e-14)
    assert np.max(abs(actual-validation.controlled_reference(np.exp(1j*time*.4)*expected)))>.1


def test_retained_implementation_error_is_separate_from_finite_rte_error(monkeypatch):
    split=adapter.prepare_matched_a(toy_inputs())
    options=dict(step_times=(.1,),r=2,K_values=(2,),q=1)
    good=validation.validate_matched_circuit_signals(split,reference,**options)
    original=validation.one_body_circuit
    def phase_bug(block,time,*,controlled=False):
        circuit=original(block,time,controlled=controlled)
        if not controlled:
            circuit.global_phase+=.003
        return circuit
    monkeypatch.setattr(validation,'one_body_circuit',phase_bug)
    bad=validation.validate_matched_circuit_signals(split,reference,**options)
    assert good['status']=='PASS' and bad['status']=='FAIL'
    for name in ('P','DF'):
        expected=good['records'][0]['dictionaries'][name]
        actual=bad['records'][0]['dictionaries'][name]
        assert actual['implementation_error_vs_nominal']['max_entry']>.001
        assert actual['finite_RTE_error_vs_PF']==expected['finite_RTE_error_vs_PF']


def test_unsupported_identity_policy_is_rejected_before_build():
    with pytest.raises(ValueError,match='faithful'):
        validation.validate_matched_circuit_signals(None,None,step_times=(.1,),r=1,K_values=(2,),q=1,
            identity_policy='extract_identity_phase')


@pytest.mark.parametrize('coefficient',[-.7,.4,-3e-12])
def test_pauli_identity_rotation_keeps_single_global_and_relative_phase(coefficient):
    module=importlib.import_module(PACKAGE+'.pauli_signal_adapter')
    circuit=validation.QuantumCircuit(2)
    module.append_pauli_evolution(circuit,'II',coefficient,.17)
    expected=np.exp(-1j*coefficient*.17)*np.eye(4)
    np.testing.assert_allclose(Operator(circuit).data,expected,rtol=0,atol=1e-15)
    controlled=validation.QuantumCircuit(3)
    controlled.append(circuit.to_gate().control(1),[2,0,1])
    np.testing.assert_allclose(Operator(controlled).data,validation.controlled_reference(expected),rtol=0,atol=1e-15)
