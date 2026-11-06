"""Artificial circuits only; full-space parity and independent fault detection."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate, XXPlusYYGate, UnitaryGate
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks.c1_h4_native_fair import decode, compile_checked
from trottertracks import c_pr_aware_df_stage_b2 as b2
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile


def artificial(qubits=9, repeats=8):
    qc = QuantumCircuit(qubits)
    qc.global_phase = .371
    for i in range(repeats):
        p, q = i % qubits, (i+1) % qubits
        qc.h(p); qc.rz(.17*(i+1), q); qc.sx(p); qc.sxdg(q)
        qc.cx(p, q); qc.append(CXGate(ctrl_state=0), [q, p])
        qc.append(XXPlusYYGate(.23, -.19), [p, q])
        qc.append(XXPlusYYGate(.11, .29).control(), [(i+2) % qubits, p, q])
    if qubits >= 4:
        nested = QuantumCircuit(4)
        nested.global_phase = -.233
        nested.h(2); nested.crz(.43, 3, 0); nested.cx(1, 3)
        qc.append(nested.to_gate(), list(reversed(range(4))))
    rng = np.random.default_rng(731)
    z = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8))
    u, _ = np.linalg.qr(z)
    qc.append(UnitaryGate(u), list(reversed(range(3))))
    qc.barrier()
    return qc


@pytest.mark.parametrize('qubits', [3, 5, 9])
def test_full_matrix_and_actual_inverse_match_independent_operator(qubits):
    qc = artificial(qubits)
    expected = decode(qc, backend='operator')
    actual = decode(qc)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
    # Build and simulate the inverse circuit itself, never return actual.H.
    inverse = b2.b.native.actual_adjoint(qc)
    observed_inverse = decode(inverse)
    np.testing.assert_allclose(observed_inverse, Operator(inverse).data,
                               atol=1e-13, rtol=0)
    np.testing.assert_allclose(observed_inverse @ actual, np.eye(1 << qubits),
                               atol=1e-13, rtol=0)


def test_full512_P0_and_adjoint_gates_preserve_reports_and_tolerances():
    body = QuantumCircuit(8)
    body.global_phase = .313
    for i in range(8):
        body.h(i); body.rz(.17*(i+1), i); body.cx(i, (i+1) % 8)
    raw = QuantumCircuit(9)
    raw.append(body.to_gate().control(), [8] + list(range(8)))
    reference = Operator(body).data
    compiled = compile_checked(raw, CompilerProfile())
    R, target, report = b2.raw_gate(raw, reference)
    C, report2 = b2.compiled_gate(compiled, R, target)
    inverses, report3 = b2.adjoint_pair_gate(raw, compiled, R, C, target)
    assert set(inverses) == {'raw', 'compiled'}
    for circuit, matrix in ((raw, R), (compiled, C)):
        np.testing.assert_allclose(matrix, decode(circuit, backend='operator'),
                                   atol=1e-13, rtol=0)
    assert report and report2 and report3


@pytest.mark.parametrize('fault', ['phase', 'outside_sector', 'wrong_inverse'])
def test_independent_faults_fail_fullspace_gates(fault, monkeypatch):
    raw = QuantumCircuit(9)
    raw.crz(.31, 8, 0)
    ref = np.eye(256, dtype=complex)
    ref[np.arange(256) % 2 == 0, np.arange(256) % 2 == 0] = np.exp(-.155j)
    ref[np.arange(256) % 2 == 1, np.arange(256) % 2 == 1] = np.exp(.155j)
    R, target, _ = b2.raw_gate(raw, ref)
    if fault == 'phase':
        raw.global_phase += .001
        with pytest.raises(ValueError):
            b2.raw_gate(raw, ref)
    elif fault == 'outside_sector':
        # Only the fully occupied sector changes; Nalpha=Nbeta=2 is unchanged.
        raw.mcp(.001, list(range(1, 9)), 0)
        with pytest.raises(ValueError):
            b2.raw_gate(raw, ref)
    else:
        monkeypatch.setattr(b2.b.native, 'actual_adjoint', lambda qc: qc.copy())
        with pytest.raises(ValueError):
            b2.adjoint_pair_gate(raw, raw, R, R, target)


def test_terminal_measurement_phase_and_thread_local_backends():
    qc = artificial(3)
    measured = QuantumCircuit(3, 1)
    measured.compose(qc, inplace=True); measured.measure(2, 0)
    expected = Operator(qc).data
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(decode, [measured, qc, qc.inverse()]))
    for actual, wanted in zip(results, [expected, expected, expected.conj().T]):
        np.testing.assert_allclose(actual, wanted, atol=1e-13, rtol=0)


def test_candidate_bound_event_cache_reuses_exact_operators(monkeypatch):
    from fractions import Fraction
    from test_research_c_pr_aware_df_stage_b2 import artificial as candidate, full_context
    from trotterlib import rte
    arm, packet = candidate(n=2, rank=8)
    generators, context = full_context(arm, packet)
    tail, config, dist = b2.b.configure(arm, Fraction(1,40), 16, 0)
    events, _ = b2.sample_trajectory(arm, tail, dist, config, 0, 'ARTIFICIAL')
    original = rte.event_unitary
    calls = []
    def counted(event, atoms):
        calls.append(event)
        return original(event, atoms)
    monkeypatch.setattr(rte, 'event_unitary', counted)
    first = b2.trajectory_reference(arm, context, Fraction(1,40), events)
    count = len(calls)
    second = b2.trajectory_reference(arm, context, Fraction(1,40), events)
    assert len(calls) == count == len(context['event_matrices'])
    np.testing.assert_array_equal(first, second)
    independent = b2.b.trajectory_reference(arm, generators, context['atoms'], Fraction(1,40), events, config)
    np.testing.assert_allclose(first, independent, atol=1e-13, rtol=0)
    assert all(not x.flags.writeable for x in context['event_matrices'].values())


@pytest.mark.parametrize('fault', ['reset', 'initialize', 'mid_measure', 'parameter', 'condition'])
def test_nonunitary_or_unbound_inputs_do_not_become_unitaries(fault):
    qc = QuantumCircuit(3, 1 if fault in ('mid_measure', 'condition') else 0)
    if fault == 'reset': qc.reset(0)
    elif fault == 'initialize': qc.initialize([1, 0], 0)
    elif fault == 'mid_measure': qc.measure(2, 0); qc.x(1)
    elif fault == 'condition': qc.x(0).c_if(qc.cregs[0], 1)
    else: qc.rz(Parameter('theta'), 0)
    with pytest.raises(ValueError): decode(qc)
