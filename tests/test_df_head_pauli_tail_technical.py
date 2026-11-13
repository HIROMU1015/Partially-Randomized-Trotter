"""Artificial operators only: native DF head + existing Pauli D2 tail.

No molecular input, preparation, sampling, transpilation or price measurement.
Reuses the native tests' two-mode blocks and independent Fock-energy oracle.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, Statevector

from trotterlib import rte
from trotterlib import rte_compiled_cost as cost
from trotterlib import pauli_signal_adapter as pauli
from trotterlib import df_partial_s2
from trotterlib.h4_d2_pauli_control import central_control_body
from trottertracks.resource_applicability import ax2b_native_df_v5 as native

spec = importlib.util.spec_from_file_location('toy_native_fixtures',
    Path(__file__).parent/'tracks/resource_applicability/test_ax2b_native_df_v5.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture(autouse=True)
def no_science_or_prices(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No scientific input/preparation/sampling/compile/price in this test')
    for module, name in ((np, 'load'), (qiskit, 'transpile'), (cost, 'transpile'),
                         (cost, 'transpile_and_measure_cost'), (rte, 'sample_rte_events'),
                         (pauli, 'sample_rte_events'), (df_partial_s2, 'prepare_df_partial_s2')):
        monkeypatch.setattr(module, name, forbidden)


def event_matrix(event, labels, n):
    result = np.eye(1 << n, dtype=complex)
    for app in event.application_sequence:
        p = Pauli(labels[app.component_id]).to_matrix()
        if app.role == 'product':
            operation = app.coefficient_sign*p
        else:
            a = app.coefficient_sign*event.rotation_angle
            operation = math.cos(a)*np.eye(1 << n) - 1j*math.sin(a)*p
        result = operation@result
    return event.phase*result


@pytest.mark.parametrize('K', [0, 2])
@pytest.mark.parametrize('tau', [.23, -.23])
@pytest.mark.parametrize('policy', ['ordinary', 'symmetric_directional'])
def test_native_df_head_pauli_tail_xy_scalar_and_signed_order(K, tau, policy):
    # ZX and YI anticommute. Positive products leave the K2 minus phase visible.
    labels = {'a': 'YX', 'b': 'ZX', 'c': 'YI'}
    components = (rte.RTEComponent('a', .6, .3, -1),
                  rte.RTEComponent('b', .2, .1, 1), rte.RTEComponent('c', .2, .1, 1))
    distribution = rte.finite_rte_distribution(tau, K)
    events = (rte._make_event((0, 1, 2) if K else (0,), components, distribution, K//2),
              rte._make_event((1,), components, distribution, 0))
    tails = []
    for e in events:
        source = QuantumCircuit(2)
        pauli.append_pauli_event(source, e, labels)
        expected = event_matrix(e, labels, 2)
        np.testing.assert_allclose(Operator(source).data, expected, atol=2e-13, rtol=0)
        controlled = central_control_body(source)
        np.testing.assert_allclose(Operator(controlled.body).data, fixtures.ordinary(expected), atol=2e-13, rtol=0)
        tails.append(controlled.body)

    blocks = fixtures.blocks()[:2]  # artificial one-body and one squared DF fragment
    T = math.copysign(.4, tau); q = 2; scalar = .13
    modes = ('ORDINARY', 'ORDINARY') if policy == 'ordinary' else ('UNCONTROLLED', 'DIRECTIONAL')
    bound = q*sum(native.block_instruction_bound(b, m) for b in blocks for m in modes)
    bound += sum(len(t.data) for t in tails) + 1
    built = native.build_partial_native(blocks, num_system_qubits=2, T=T, q=q,
        scalar=scalar, controlled_tails=tails, control_policy=policy, max_instructions=bound)
    expected = np.eye(4, dtype=complex)
    for e in events:
        for b in blocks:
            expected = fixtures.independent_term(b, T/q/2)@expected
        expected = event_matrix(e, labels, 2)@expected
        for b in reversed(blocks):
            expected = fixtures.independent_term(b, T/q/2)@expected
    expected *= np.exp(-1j*scalar*T)
    # Full matrix comparison, including the low-control block, without phase alignment.
    np.testing.assert_allclose(Operator(built.circuit).data, fixtures.ordinary(expected), atol=4e-13, rtol=0)
    wrong = built.circuit.copy(); wrong.p(-scalar*T, 2)
    assert np.linalg.norm(Operator(wrong).data-fixtures.ordinary(expected)) > .01
    for axis in ('cosine', 'sine'):
        wrapper = native.build_native_hadamard_wrapper(built, axis=axis,
            include_measurement=True, max_instructions=len(built.circuit.data)+4)
        assert wrapper.num_clbits == 1 and wrapper.data[-1].operation.name == 'measure'
        p = Statevector.from_instruction(wrapper.remove_final_measurements(inplace=False)).probabilities()
        observed = float(sum(p[:4])-sum(p[4:]))
        target = expected[0, 0].real if axis == 'cosine' else expected[0, 0].imag
        assert observed == pytest.approx(target, abs=4e-13)


def test_longer_pauli_support_and_identity_exclusion():
    # Qiskit little-endian label/wire convention, three-wire parity and mixed axes.
    source = QuantumCircuit(3)
    pauli.append_pauli_evolution(source, 'XYZ', -.3, -.2)
    expected = math.cos(.06)*np.eye(8)-1j*math.sin(.06)*Pauli('XYZ').to_matrix()
    np.testing.assert_allclose(Operator(central_control_body(source).body).data,
        fixtures.ordinary(expected), atol=2e-13, rtol=0)
    invalid = QuantumCircuit(2)
    pauli.append_pauli_evolution(invalid, 'II', 1., .2)
    with pytest.raises(ValueError, match='IDENTITY_EVOLUTION'):
        central_control_body(invalid)
