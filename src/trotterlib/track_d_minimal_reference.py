"""One fixed artificial DF word: direct controls versus a reference gadget.

No molecular I/O, sampler, shot execution, parameter search, or T synthesis.
Loaded only by the explicit runner after its SOURCE has been committed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from . import native_df_gate_model as model
from .df_partial_s2 import DFDeterministicOneBodySpec
from .df_rte_circuit import DFRTEEventSequenceCircuitRequest
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .df_trotter import ops

T, Q, SCALAR, TAU = .8, 2, .17, .4
TOLERANCE = 3e-12
ARMS = ("ordinary_shared_fused", "symmetric_shared_fused", "reference")
AXES = ("X", "Y")


@dataclass
class Fixture:
    physical: QuantumCircuit
    physical_reference: np.ndarray
    step: QuantumCircuit
    later_half: QuantumCircuit
    earlier_half: QuantumCircuit
    cores: dict
    native_controls: dict
    wrappers: dict
    negative_wrappers: dict
    events: tuple
    phase: float
    basis_checks: list


def _primitive(block, time):
    if isinstance(block, DFDeterministicOneBodySpec):
        return ops.one_body_diagonal_primitives(np.asarray(block.diagonal_eigenvalues), time)
    return ops.df_squared_diagonal_primitives(np.asarray(block.diagonal_eta), block.lam, time)


def hadamard_wrapper(core, axis):
    """Native H/U/(Sdg)/H convention, with explicit ancilla-last binding."""
    if axis not in AXES or core.num_clbits:
        raise ValueError("Numeric unitary core and X/Y axis required")
    ancilla = core.num_qubits - 1
    qc = QuantumCircuit(core.num_qubits, 1)
    qc.h(ancilla)
    qc.compose(core, inplace=True)
    if axis == "Y":
        qc.sdg(ancilla)
    qc.h(ancilla)
    qc.measure(ancilla, 0)
    return qc


def reference_gadget(later, earlier, phase):
    """Eq.2 branch order: A† x B versus B x A†, two CSWAP layers."""
    if later.num_qubits != 2 or earlier.num_qubits != 2:
        raise ValueError("Only the selected two-mode fixture is supported")
    qc = QuantumCircuit(5)
    for system, reference in ((0, 2), (1, 3)):
        qc.cswap(4, system, reference)
    qc.compose(later.inverse(), qubits=[0, 1], inplace=True)
    qc.compose(earlier, qubits=[2, 3], inplace=True)
    for system, reference in ((0, 2), (1, 3)):
        qc.cswap(4, system, reference)
    qc.p(phase, 4)
    return qc


def induced_observable(wrapper):
    """4x4 observable for arbitrary system input, other inputs fixed to |0>."""
    if wrapper.num_qubits not in (3, 5) or wrapper.num_clbits != 1:
        raise ValueError("Unexpected full-wrapper register")
    measurements = [i for i, item in enumerate(wrapper.data) if item.operation.name == "measure"]
    if measurements != [len(wrapper.data) - 1]:
        raise ValueError("One terminal ancilla measurement required")
    measured = wrapper.data[-1]
    if wrapper.find_bit(measured.qubits[0]).index != wrapper.num_qubits - 1:
        raise ValueError("Measurement must read the last qubit")
    unitary = wrapper.remove_final_measurements(inplace=False)
    matrix = Operator(unitary).data
    columns = matrix[:, :4]  # original system bits 0,1; reference/ancilla initially zero
    signs = np.ones(matrix.shape[0])
    signs[len(signs) // 2:] = -1
    return columns.conj().T @ (signs[:, None] * columns)


def expected_observable(physical, axis):
    return ((physical + physical.conj().T) / 2 if axis == "X"
            else (physical - physical.conj().T) / (2j))


def build_fixture(support, native):
    registry, components, specs, _ = support.rte_inputs(same_basis=True, identity=False)
    _, events = support.build_events(components, tau=TAU, maximum_order=0)
    if (len(events) != 2 or tuple(e.rotation_component_id for e in events) != ("Zplus", "ZZminus")
            or any(e.taylor_order != 0 or e.product_component_ids for e in events)):
        raise ValueError("Selected fixture word changed")
    _, _, blocks, bases, frames = support.artificial_inputs()
    tail_controlled = support.build_rte(registry, specs, events, True).circuit
    tail_request = DFRTEEventSequenceCircuitRequest(
        events=events, component_specs=specs, controlled=False,
        cancel_adjacent_equal_bases=True)
    tail = QiskitDFRTEEventCircuitBuilder(basis_registry=registry).build_sequence(tail_request).circuit
    lookup = lambda app: model.basis_from_runtime(
        app.basis_id, app.basis_hash, registry.operations(app.basis_id))
    tail_ir = model.existing_rte_occurrence(events, lookup, 2, "B")
    half_time = T / Q / 2

    # Physical step uses the existing actual un-controlled appenders, not the IR counter.
    step = QuantumCircuit(2)
    for block in blocks:
        native._append_block(step, block, half_time, "UNCONTROLLED", None)
    step.compose(tail, inplace=True)
    for block in reversed(blocks):
        native._append_block(step, block, half_time, "UNCONTROLLED", None)
    phase = -SCALAR * T
    physical = QuantumCircuit(2)
    physical.compose(step, inplace=True)
    physical.compose(step, inplace=True)
    physical.global_phase += phase
    later = step.copy()
    later.global_phase += phase

    native_controls = {}
    cores = {}
    for policy, arm in (("ordinary", ARMS[0]), ("symmetric_directional", ARMS[1])):
        native_controls[arm] = native.build_partial_native(
            blocks, num_system_qubits=2, T=T, q=Q, scalar=SCALAR,
            controlled_tails=(tail_controlled, tail_controlled), control_policy=policy,
            max_instructions=2000).circuit
        left, right = (("ordinary", "ordinary") if policy == "ordinary"
                       else ("uncontrolled", "directional"))

        def sandwich(index, time, mode):
            return model.Sandwich(bases[index], model.diagonal_from_primitives(
                _primitive(blocks[index], time), mode, 2))

        parts = []
        for outer in range(Q):
            for j in (range(len(blocks)) if outer == 0 else range(1, len(blocks))):
                parts.append(sandwich(j, half_time, left))
            parts.append(tail_ir)  # closed occurrence, identical to selected saved B scope
            for j in (reversed(range(len(blocks))) if outer == Q - 1
                      else reversed(range(1, len(blocks)))):
                parts.append(sandwich(j, half_time, right))
            if outer != Q - 1:
                # Same-block boundary: ordinary+ordinary gives C[U(2t)].
                # directional+uncontrolled also gives diag(I,U(2t)), NOT a
                # second directional half. Both arms retain the scalar primitive.
                parts.append(sandwich(0, 2 * half_time, "ordinary"))
        parts.append(model.Circuit((model.Gate("p", (2,), phase),)))
        cores[arm] = support.circuit_from_ir(model.assemble(parts, "B"), 3)

    cores["reference"] = reference_gadget(later, step, phase)
    negative = reference_gadget(later, step, 0.0)
    wrappers = {(arm, axis): hadamard_wrapper(core, axis)
                for arm, core in cores.items() for axis in AXES}

    # Independent physical reference: existing exterior-minor lift and direct
    # occupation diagonals. No PF builder or controlled circuit supplies V here.
    occupation = np.array([[(state >> bit) & 1 for bit in range(2)] for state in range(4)])
    lifts = [support.exterior_fock_lift(u) for u in frames]
    one = occupation @ np.asarray(blocks[0].diagonal_eigenvalues)
    square = blocks[1].lam * (occupation @ np.asarray(blocks[1].diagonal_eta)) ** 2
    d0, d1 = [lift @ np.diag(np.exp(-1j * half_time * energies)) @ lift.conj().T
              for lift, energies in zip(lifts, (one, square))]
    z0 = 1 - 2 * occupation[:, 0]
    zz = z0 * (1 - 2 * occupation[:, 1])
    phi = math.atan(TAU)
    eplus = lifts[0] @ np.diag(np.exp(-1j * phi * z0)) @ lifts[0].conj().T
    eminus = lifts[0] @ np.diag(np.exp(1j * phi * zz)) @ lifts[0].conj().T
    ref_step = d0 @ d1 @ eminus @ eplus @ d1 @ d0
    reference = np.exp(1j * phase) * ref_step @ ref_step
    basis_checks = []
    for b, lift in zip(bases, lifts):
        actual = Operator(support.circuit_from_ir(b.circuit(), 2)).data
        basis_checks.append((actual, actual[0, 0] * lift))
    return Fixture(physical, reference, step, later, step.copy(), cores, native_controls,
                   wrappers, {axis: hadamard_wrapper(negative, axis) for axis in AXES},
                   events, phase, basis_checks)


def compile_wrapper(wrapper, compile_gates, profile):
    """Exactly one transpile call, using the existing phase-preserving F2 path."""
    return compile_gates.compiled_wrapper(wrapper.copy(), profile)


def metrics(circuit):
    counts = circuit.count_ops()
    return dict(rz_count=int(counts.get("rz", 0)), cx_count=int(counts.get("cx", 0)),
                rz_depth=int(circuit.depth(lambda item: item.operation.name == "rz") or 0),
                cx_depth=int(circuit.depth(lambda item: item.operation.name == "cx") or 0),
                total_depth=int(circuit.depth() or 0), circuit_size=int(circuit.size()),
                qubit_count=circuit.num_qubits, measurement_count=int(counts.get("measure", 0)),
                global_phase=float(circuit.global_phase))


def complex_array(value):
    value = np.asarray(value)
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}
