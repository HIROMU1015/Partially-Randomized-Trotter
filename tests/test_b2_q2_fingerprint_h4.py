"""Artificial adapter/STOP tests; never load real molecular inputs."""

import copy
from dataclasses import replace

import numpy as np
import pytest

import run_b2_q2_fingerprint_h4 as m
from test_b2_q2_streaming_h4 import case


@pytest.fixture(autouse=True)
def no_saved_data_or_old_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No molecular I/O or old real runners')
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(m.previous, 'git', forbidden)
    monkeypatch.setattr(m.q2, 'run', forbidden)
    monkeypatch.setattr(m.m0, 'run', forbidden)
    import trotterlib.df_hamiltonian as h
    monkeypatch.setattr(h, 'solve_df_ground_state', forbidden)
    monkeypatch.setattr(h, 'build_df_h_d_from_molecule', forbidden)


@pytest.fixture(scope='module')
def saved_artificial_reference():
    _, r, events, evolutions, wrappers = case()
    evo = evolutions['reference']
    reference = dict(fixed_conditions=dict(Q=2,T=.04),
        native_comparison=dict(reference_fingerprint=evo.circuit_fingerprint,
            instruction_count=len(evo.circuit.data),
            instruction_rows_digest=m.m0.digest(m.m0.instruction_rows(evo.circuit))),
        wrappers={'reference':{}}, axes={'reference':{}})
    for axis in ('cosine','sine'):
        w = wrappers['reference'][axis]
        cost = m.previous.transpile_and_measure_cost(w,r.context.compiler)
        fp = m.previous.canonical_qiskit_circuit_fingerprint(w)
        reference['wrappers']['reference'][axis] = dict(fingerprint=fp,instructions=len(w.data),
            qubits=w.num_qubits,clbits=w.num_clbits,measurements=1)
        reference['axes']['reference'][axis] = dict(actual_circuit_fingerprint=fp,
            compiled_circuit_fingerprint=m.previous.canonical_qiskit_circuit_fingerprint(cost.transpiled_circuit),
            compiler_hash=cost.compiler_settings_hash,backend_fingerprint=cost.backend_fingerprint,
            scope=m.previous.RPE_HADAMARD_INTERROGATION_SCOPE,state_preparation_included=False,
            qubits=cost.qubit_count,compiled_global_phase=cost.global_phase,
            pretranspile_gate_counts=dict(cost.pretranspile_gate_counts),
            metrics={k:getattr(cost,k) for k in m.previous.RPE_COST_METRICS})
    return r,events,reference


def result():
    return dict(status='ARTIFICIAL_TEST',native_builds_started=0,compile_calls_started=0,
                stages={},wrappers={},axes={})


def test_one_build_two_compiles_reproduce_saved_reference(saved_artificial_reference,tmp_path):
    r,events,ref = saved_artificial_reference
    out = result()
    m.evaluate_candidate(out,r,events,ref,r.context.compiler,tmp_path,'ARTIFICIAL_SOURCE')
    assert out['native_builds_started']==1 and out['compile_calls_started']==2
    assert out['native_comparison']['status']=='EXACT_MATCH'
    assert all(w['status']=='EXACT_MATCH' for w in out['wrappers'].values())
    for a in out['axes'].values():
        assert a['comparison']['status']=='EXACT_MATCH'
        assert not any(a['comparison']['signed_difference'].values())


@pytest.mark.parametrize('changed', ['native','sine_wrapper'])
def test_saved_identity_mismatch_stops_before_compile(saved_artificial_reference,tmp_path,changed):
    r,events,ref = saved_artificial_reference
    ref = copy.deepcopy(ref)
    if changed=='native':
        ref['native_comparison']['reference_fingerprint']='different'
    else:
        ref['wrappers']['reference']['sine']['fingerprint']='different'
    out = result()
    with pytest.raises(ValueError,match='SAVED_NATIVE_IDENTITY|SAVED_WRAPPER_IDENTITY'):
        m.evaluate_candidate(out,r,events,ref,r.context.compiler,tmp_path,'ARTIFICIAL_SOURCE')
    assert out['native_builds_started']==1 and out['compile_calls_started']==0


def test_price_mismatch_stops_after_first_compile(saved_artificial_reference,tmp_path):
    r,events,ref = saved_artificial_reference
    calls = []
    def different(circuit,compiler,**kwargs):
        calls.append(1)
        cost = m.previous.transpile_and_measure_cost(circuit,compiler,**kwargs)
        return replace(cost,cx_count=cost.cx_count+1)
    out = result()
    with pytest.raises(ValueError,match='SAVED_COMPILED_IDENTITY_OR_METRICS'):
        m.evaluate_candidate(out,r,events,ref,r.context.compiler,tmp_path,'ARTIFICIAL_SOURCE',different)
    assert len(calls)==out['compile_calls_started']==1
    assert out['axes']['cosine']['comparison']['signed_difference']['cx_count']==1
    assert 'sine' not in out['axes']


def test_compiler_failure_is_not_retried(saved_artificial_reference,tmp_path):
    r,events,ref = saved_artificial_reference
    calls = []
    def failed(*args,**kwargs):
        calls.append(1)
        raise RuntimeError('synthetic compiler failure')
    out = result()
    with pytest.raises(RuntimeError,match='synthetic compiler failure'):
        m.evaluate_candidate(out,r,events,ref,r.context.compiler,tmp_path,'ARTIFICIAL_SOURCE',failed)
    assert len(calls)==out['compile_calls_started']==1
