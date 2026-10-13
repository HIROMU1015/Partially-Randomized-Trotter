"""Off-domain observer, phase, and cost aggregation checks; no H4 input."""
from dataclasses import replace
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator

from scripts.diagnostics.rte_basis_cost_attribution import TaggedBuilder, partition_circuit, summarize
from trotterlib.df_rte_circuit import DFRTEComponentCircuitSpec, DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.df_rte_tail import DFBasisRegistry
from trotterlib.rte import RTEComponent, _make_event, finite_rte_distribution


def fixture(order=0):
    registry = DFBasisRegistry()
    definition = registry.register(((RYGate(.37), (0,)),), num_system_qubits=1)
    spec = DFRTEComponentCircuitSpec('toy', 1., -1, 'toy-fragment', definition.basis_id,
        (0,), definition.metadata.operations, 1, basis_hash=definition.basis_hash)
    component = RTEComponent('toy', 1., 1., -1, 'toy-fragment', definition.basis_id,
        definition.basis_hash, diagonal_pauli_support=(0,),
        basis_change_operations=definition.metadata.operations)
    event = _make_event((0,) * (order + 1), (component,), finite_rte_distribution(.13, 2), order // 2)
    request = DFRTEEventSequenceCircuitRequest((event, event), (spec,), controlled=True,
                                             ancilla_qubit=1)
    return registry, request


def test_observer_retains_basis_reuse_and_controlled_operator():
    registry, request = fixture()
    baseline = QiskitDFRTEEventCircuitBuilder(basis_registry=registry).build_sequence(request)
    builder = TaggedBuilder(basis_registry=registry)
    tagged = builder.build_sequence(request)
    assert tagged.cancelled_basis_change_pairs == 1
    assert tagged.emitted_basis_change_count == 2
    assert tagged.circuit == baseline.circuit
    parts = partition_circuit(tagged.circuit, builder.spans)
    rebuilt = QuantumCircuit(2)
    rebuilt.global_phase = tagged.circuit.global_phase
    for _, part in parts:
        rebuilt.compose(part, inplace=True)
    assert np.max(np.abs(Operator(rebuilt).data - Operator(baseline.circuit).data)) < 1e-12


def test_relative_phase_is_other_and_not_discarded():
    registry, request = fixture(order=2)
    request = replace(request, events=(request.events[0],))
    builder = TaggedBuilder(basis_registry=registry)
    built = builder.build_sequence(request)
    parts = partition_circuit(built.circuit, builder.spans)
    assert parts[-1][0]['role'] == 'other'
    assert abs(np.exp(1j * built.relative_ancilla_phase) + 1) < 1e-12
    missing = QuantumCircuit(2)
    for group, part in parts:
        if group['role'] != 'other':
            missing.compose(part, inplace=True)
    assert np.max(np.abs(Operator(missing).data - Operator(built.circuit).data)) > .1


def test_overlapping_or_incomplete_segments_rejected():
    circuit = QuantumCircuit(1)
    circuit.rz(.2, 0)
    circuit.ry(.3, 0)
    with pytest.raises(ValueError):
        partition_circuit(circuit, [{'role': 'basis', 'start': 0, 'end': 1}])
    with pytest.raises(ValueError):
        partition_circuit(circuit, [{'role': 'basis', 'start': 0, 'end': 2},
                                    {'role': 'central', 'start': 1, 'end': 2}])


def test_ratio_of_sums_instead_of_mean_row_percentages():
    def row(basis, central):
        return {'additive': {'basis': dict.fromkeys(('rz_count','cx_count','circuit_size'),basis),
                            'central': dict.fromkeys(('rz_count','cx_count','circuit_size'),central),
                            'other': dict.fromkeys(('rz_count','cx_count','circuit_size'),0)},
                'full': dict.fromkeys(('rz_count','cx_count','circuit_size'),basis+central-1)}
    result = summarize([row(9,1),row(1,99)])
    assert result['basis_fraction_exact']['rz_count'] == {'numerator':10,'denominator':110}
    assert result['basis_percent_additive']['rz_count'] == pytest.approx(100/11)
    assert result['additive_minus_whole']['rz_count'] == 2
    assert result['exact_optimized_cost_lineage_claimed'] is False
