"""Artificial identity/comparison tests; no saved molecular input or solver."""
from copy import deepcopy
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_b2_m0_streaming_h4 as m
from b2_cost_method_cases import artificial_round,event_pattern


@pytest.fixture(autouse=True)
def no_molecular_inputs(monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError('No saved scientific inputs or solver')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(m.previous,'git',forbidden)
    import trotterlib.df_hamiltonian as h
    monkeypatch.setattr(h,'solve_df_ground_state',forbidden)
    monkeypatch.setattr(h,'build_df_h_d_from_molecule',forbidden)


def pair():
    _,r=artificial_round(0,3)
    events=tuple(event_pattern(r,'mixed'))
    request=r.explicit_request(events)
    original=r.native(request,max_instructions=m.request_instruction_bound(request))
    streamed=m.stream_native(r,iter(events))
    return original,streamed


def test_native_comparison_requires_exact_saved_identity_and_rows():
    reference,candidate=pair()
    record=m.compare_native(reference,candidate,reference.circuit_fingerprint)
    assert record['status']=='EXACT_MATCH'
    assert all(record['checks'].values())
    assert record['phase_fitting'] is False


@pytest.mark.parametrize('defect',['phase','gate','saved_identity'])
def test_native_difference_is_inconclusive_without_phase_fitting(defect):
    reference,candidate=pair()
    fp=reference.circuit_fingerprint
    if defect=='saved_identity':
        fp='different_saved_circuit'
    else:
        changed=candidate.circuit.copy()
        if defect=='phase': changed.global_phase+=.3
        else: changed.rz(.1,2)
        candidate=replace(candidate,circuit=changed,
            circuit_fingerprint=m.previous.canonical_qiskit_circuit_fingerprint(changed))
    assert m.compare_native(reference,candidate,fp)['status']=='INCONCLUSIVE'


def fake_record():
    return dict(metrics=dict(rz_count=17,cx_count=11,rz_depth=9,cx_depth=8,total_depth=21,circuit_size=40),
        actual_circuit_fingerprint='invented_input',compiled_circuit_fingerprint='invented_compiled',
        compiler_hash='invented_compiler',backend_fingerprint='no_backend',
        scope=m.previous.RPE_HADAMARD_INTERROGATION_SCOPE,state_preparation_included=False,qubits=3)


def test_all_six_metrics_and_circuit_context_match():
    saved=fake_record()
    record=m.compare_axis(deepcopy(saved),saved)
    assert record['status']=='EXACT_MATCH'
    assert all(v==0 for v in record['signed_difference'].values())


@pytest.mark.parametrize('defect',['metric','compiled_identity','input_identity','scope','prep'])
def test_equal_prices_alone_do_not_qualify_a_different_circuit(defect):
    saved=fake_record();candidate=deepcopy(saved)
    if defect=='metric': candidate['metrics']['cx_count']+=1
    elif defect=='compiled_identity': candidate['compiled_circuit_fingerprint']='different'
    elif defect=='input_identity': candidate['actual_circuit_fingerprint']='different'
    elif defect=='scope': candidate['scope']='evolution_only'
    else: candidate['state_preparation_included']=True
    assert m.compare_axis(candidate,saved)['status']=='INCONCLUSIVE'
