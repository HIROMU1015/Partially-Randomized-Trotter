"""Artificial matrices only: no H4 inputs, molecular loader, RNG or transpile.

Reuse production builders with forced events. References use explicit occupation
energies / signed matrices, rather than extracting the controlled output block.
"""
from dataclasses import replace
from fractions import Fraction
from itertools import product
from pathlib import Path
import importlib.util
import math
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.df_partial_s2_cost import _request_for_events
from trotterlib.df_partial_s2_repeated import (
    DFPartialS2RepeatedRequest, QiskitDFPartialS2RepeatedCircuitBuilder,
)
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.df_rte_tail import basis_change_unitary
from trotterlib.research_direction_full_scope import apply_run_threshold_policy
from trotterlib.research_direction_sequence_policy import register_support_restricted_bases
from trotterlib.rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder, round_index_for_benchmark_repetition_count,
)
from trotterlib.rpe_hadamard_failure_validation import _trajectory_signal
from trotterlib.rte import _make_event, event_unitary, make_rte_config

ATOL = 1e-11


@pytest.fixture(scope="module")
def artificial_preparation():
    # Explicit 2-mode algebraic fixture, patterned on test_df_partial_s2_repeated.
    # No chemistry/state solve. Two tail bases allow noncommuting components.
    h = DFHamiltonian(
        constant=.17,
        one_body=np.array([[.4, .13], [.13, -.2]], complex),
        lambdas=np.array([.6, -.35, .22]),
        g_matrices=tuple(np.array(g, complex) for g in (
            [[1., .19], [.19, .35]], [[.2, -.23], [-.23, 1.2]],
            [[.7, .31], [.31, -.4]],
        )), metadata={"name": "ARTIFICIAL_ONLY"},
    )
    return h, prepare_df_partial_s2(
        h, split_df_hamiltonian_by_ld(h, 1),
        identity_policy="extract_identity_phase", coefficient_atol=1e-12,
    )


def signed_operators(prep):
    out = {}
    for c in prep.rte_preparation.symbolic_tail.components:
        u = basis_change_unitary(prep.tail_extraction.basis_definition(c.basis_id))
        diag = np.array([(-1) ** sum((i >> q) & 1 for q in c.diagonal_pauli_support)
                         for i in range(4)])
        out[c.component_id] = c.coefficient_sign * (u * diag) @ u.conj().T
    return out


def noncommuting_pair(prep, ops):
    cs = prep.rte_preparation.symbolic_tail.components
    return next((i, j) for i, a in enumerate(cs) for j, b in enumerate(cs)
                if a.coefficient_sign == -1 and b.coefficient_sign == 1
                and np.max(np.abs(ops[a.component_id] @ ops[b.component_id]
                                  - ops[b.component_id] @ ops[a.component_id])) > .01)


def halves(prep, dt):
    """Occupation-energy reference, independent of diagonal primitive appending."""
    blocks = []
    for block in prep.deterministic_blocks:
        q = QuantumCircuit(2)
        for gate, wires in block.runtime_basis_operations:
            q.append(gate, list(wires))
        u = Operator(q).data
        if block.block_kind == "one_body":
            energies = [sum(e * ((i >> j) & 1)
                            for j, e in enumerate(block.diagonal_eigenvalues)) for i in range(4)]
        else:
            energies = [block.lam * sum(e * ((i >> j) & 1)
                            for j, e in enumerate(block.diagonal_eta)) ** 2 for i in range(4)]
        blocks.append((u * np.exp(-1j * dt / 2 * np.array(energies))) @ u.conj().T)
    f = np.exp(-1j * dt * (prep.constant_coefficient + prep.extracted_identity_coefficient)) * np.eye(4)
    r = np.eye(4, dtype=complex)
    for block in blocks:
        f = block @ f
    for block in reversed(blocks):
        r = block @ r
    return f, r


def forced_request(prep, r, k, q=2, dt=.17):
    config, law = make_rte_config(prep.rte_preparation.symbolic_tail,
        evolution_time=dt, rte_steps=r, truncation_tolerance=1., finite_taylor_order=k)
    ops = signed_operators(prep)
    i, j = noncommuting_pair(prep, ops)
    components = prep.rte_preparation.symbolic_tail.components
    zero = _make_event((i,), components, law, law.orders.index(0))
    higher = _make_event((i,) + tuple((j, i)[n % 2] for n in range(k)),
                         components, law, law.orders.index(k))
    events = tuple(higher if n % 2 == 0 else zero for n in range(q * r))
    steps = [_request_for_events(prep, step_time=dt, config=config, distribution=law,
                events=events[n*r:(n+1)*r], seed=0, controlled=True, ancilla_qubit=2,
                cancel_adjacent_equal_bases=True) for n in range(q)]
    request = DFPartialS2RepeatedRequest.from_step_requests(
        steps, sampling_policy="artificial_forced_no_rng", construction_policy="boundary_optimized")
    return request, events, ops


def controlled(u):
    z = np.zeros_like(u)
    return np.block([[np.eye(len(u)), z], [z, u]])


@pytest.mark.parametrize("r,k", [(1, 2), (2, 2), (4, 2), (8, 2), (16, 2), (32, 2), (3, 4)])
def test_forced_noncommuting_pipeline(artificial_preparation, r, k):
    _, prep = artificial_preparation
    request, events, ops = forced_request(prep, r, k)
    f, rev = halves(prep, request.step_time)
    expected = np.eye(4, dtype=complex)
    for n in range(2):
        expected = f @ expected
        for ev in events[n*r:(n+1)*r]:
            expected = event_unitary(ev, ops) @ expected
        expected = rev @ expected
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    raw = builder.build(request, construction_policy="raw_concatenation")
    optimized = builder.build(request, construction_policy="boundary_optimized")
    for result in (raw, optimized):
        np.testing.assert_allclose(Operator(result.circuit).data, controlled(expected), atol=ATOL, rtol=0)
    # Compare ordinary control of the independently built uncontrolled circuit too.
    unsteps = [replace(s, controlled=False, ancilla_qubit=None,
                      rte_occurrence=replace(s.rte_occurrence, controlled=False, ancilla_qubit=None))
               for s in request.iter_step_requests()]
    un = builder.build(DFPartialS2RepeatedRequest.from_step_requests(unsteps))
    np.testing.assert_allclose(Operator(un.circuit).data, expected, atol=ATOL, rtol=0)
    # Direct control must retain the uncontrolled circuit's global phase.
    whole = QuantumCircuit(3)
    whole.append(un.circuit.to_gate().control(1), [2, 0, 1])
    np.testing.assert_allclose(Operator(whole).data, controlled(expected), atol=ATOL, rtol=0)
    psi = np.array([.2, .3j, -.4, .5j]); psi /= np.linalg.norm(psi)
    signal = np.vdot(psi, expected @ psi)
    old = _trajectory_signal(initial_state=psi, forward=f, reverse=rev, events=events,
                             operators=ops, q_m=2, rte_steps=r)
    assert abs(old - signal) < ATOL
    for axis in ("cosine", "sine"):
        wrapper = QiskitRPEHadamardBenchmarkCircuitBuilder(maximum_repetition_count=2).build(
            RPEHadamardInterrogationRequest(optimized, axis, include_measurement=True))
        vec = Statevector(np.r_[psi, np.zeros(4)]).evolve(
            wrapper.circuit.remove_final_measurements(inplace=False)).data
        mean = np.sum(abs(vec[:4])**2) - np.sum(abs(vec[4:])**2)
        assert abs(mean - (signal.real if axis == "cosine" else signal.imag)) < ATOL
        assert wrapper.bit_value_mapping == ((0, 1), (1, -1))


def test_support_policy_preserves_forced_multievent_pipeline(artificial_preparation):
    h, prep = artificial_preparation
    request, _, _ = forced_request(prep, 4, 2)
    definitions, proofs = register_support_restricted_bases(h, prep)
    assert all(p["preserved_columns_max_abs_residual"] < ATOL for p in proofs)
    planned = apply_run_threshold_policy(request, definitions, training_fingerprint="a"*64)
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    np.testing.assert_allclose(Operator(builder.build(planned).circuit).data,
                               Operator(builder.build(request).circuit).data, atol=ATOL, rtol=0)
    assert any(c.construction == "support_restricted_preserved_columns_v1"
               for plan in planned.rte_basis_plans for c in plan.choices)


def test_order_and_sign_mutations_are_observable(artificial_preparation):
    _, prep = artificial_preparation
    request, events, ops = forced_request(prep, 2, 2)
    ev = events[0]
    a, b = (ops[c] for c in ev.product_component_ids)
    rot = ops[ev.rotation_component_id]
    eye = np.eye(4)
    expected = event_unitary(ev, ops)
    bad_product = ev.phase * (math.cos(ev.rotation_angle)*eye - 1j*math.sin(ev.rotation_angle)*rot) @ a @ b
    bad_rotation = ev.phase * (math.cos(ev.rotation_angle)*eye + 1j*math.sin(ev.rotation_angle)*rot) @ b @ a
    assert np.max(abs(expected-bad_product)) > .01
    assert np.max(abs(expected-bad_rotation)) > .001
    f, r = halves(prep, request.step_time)
    assert np.max(abs(r @ expected @ f - f @ expected @ r)) > 1000 * ATOL
    assert np.max(abs(controlled(expected)-controlled(-expected))) > 1


def test_categorical_mean_is_amplitude_average_and_needs_no_extra_B():
    # Finite artificial law; exhaustive trajectories are feasible only here.
    x = np.array([[0, 1], [1, 0]], complex); z = np.diag([1., -1.]); eye = np.eye(2)
    matrices = [math.cos(.3)*eye-1j*math.sin(.3)*x, -(math.cos(.2)*eye-1j*math.sin(.2)*z)]
    masses = [Fraction(1, 4), Fraction(3, 4)]
    avg = sum(float(p)*u for p, u in zip(masses, matrices))
    psi = np.array([math.sqrt(.3), 1j*math.sqrt(.7)])
    enumerated = 0j
    for ids in product(range(2), repeat=4):  # r=2, q=2 with independent events
        u = eye.copy(); p = Fraction(1)
        for i in ids:
            u = matrices[i] @ u; p *= masses[i]
        enumerated += float(p) * np.vdot(psi, u @ psi)
    mean = np.vdot(psi, np.linalg.matrix_power(avg, 4) @ psi)
    assert abs(enumerated-mean) < 1e-14
    assert abs(mean-mean/1.7) > .01  # An additional normalization changes the target.
    channel_overlap = sum(float(p)*abs(np.vdot(psi, u @ psi))**2 for p, u in zip(masses, matrices))
    assert abs(channel_overlap-abs(np.vdot(psi, avg @ psi))**2) > .01
    for q in (1, 2, 8, 131072):  # Tiny 2x2 power, never a q-sized circuit.
        a = np.linalg.matrix_power(avg, q)
        normed = np.vdot(3*psi, a @ (3*psi)) / np.vdot(3*psi, 3*psi)
        assert abs(normed-np.vdot(psi, a @ psi)) < 1e-14


def test_paired_component_law_matches_independent_event_enumeration(artificial_preparation):
    _, prep = artificial_preparation
    _, law = make_rte_config(prep.rte_preparation.symbolic_tail, evolution_time=.17,
                            rte_steps=2, truncation_tolerance=1., finite_taylor_order=2)
    ops = signed_operators(prep)
    indices = noncommuting_pair(prep, ops)
    cs = prep.rte_preparation.symbolic_tail.components
    component_mass = [Fraction(1, 4), Fraction(3, 4)]
    order_mass = [Fraction(2, 3), Fraction(1, 3)]
    a = sum(float(p)*ops[cs[i].component_id] for p, i in zip(component_mass, indices))
    enumerated = np.zeros((4, 4), complex)
    factorized = np.zeros((4, 4), complex)
    builder = QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
    for n, pn in zip((0, 2), order_mass):
        theta = math.atan(law.dimensionless_step_time/(n+1))
        factorized += float(pn)*(-1)**(n//2) * (math.cos(theta)*np.eye(4)
                      -1j*math.sin(theta)*a) @ np.linalg.matrix_power(a, n)
        for choices in product(range(2), repeat=n+1):
            ev = _make_event(tuple(indices[i] for i in choices), cs, law, law.orders.index(n))
            probability = pn * math.prod(component_mass[i] for i in choices)
            u = event_unitary(ev, ops)
            circuit = builder.build_event(prep.rte_preparation.request_for_event(
                ev, controlled=True, ancilla_qubit=2)).circuit
            np.testing.assert_allclose(Operator(circuit).data, controlled(u), atol=ATOL, rtol=0)
            enumerated += float(probability)*u
    np.testing.assert_allclose(enumerated, factorized, atol=ATOL, rtol=0)


@pytest.mark.parametrize("m", range(18))
def test_all_rounds_wrapper_metadata_domain_without_build(m):
    assert round_index_for_benchmark_repetition_count(2**m, maximum_repetition_count=2**17) == m


def test_b_phase_and_sine_error_detection_reused():
    # Call only existing artificial fixtures, never B.run()/load_api().
    path = ROOT / "docs/research/rpe_fixed_request_measurement_b_20261103/run_fixed_request_b.py"
    spec = importlib.util.spec_from_file_location("b_definitions_only", path)
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    psi = np.array([1., 0.], complex); phase = np.exp(-.31j)
    evo = QuantumCircuit(2); evo.p(-.31, 1)
    wrappers = {}
    for axis in ("cosine", "sine"):
        c = QuantumCircuit(2, 1); c.h(1); c.compose(evo, inplace=True)
        if axis == "sine": c.sdg(1)
        c.h(1); c.measure(1, 0); wrappers[axis] = c
    job = dict(id="toy", ld=0, q=1, profile="toy", system_state=psi,
               target_system_state=phase*psi, sector_indices=np.array([0, 1]),
               signal=phase, evolution=evo, wrappers=wrappers)
    assert b.evaluate_job(job)["pass"]
    assert not b.evaluate_job(dict(job, signal=phase.conjugate()))["pass"]
    assert not b.evaluate_job(dict(job, target_system_state=-phase*psi))["pass"]
