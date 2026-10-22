"""H2-only operator checks and random-event diagnostics for the frozen pilot."""
from collections import Counter
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from .h2_non_df_pilot import check, dense_law_mean, pauli_matrix
from .pauli_signal_adapter import append_pauli_event
from .rte import _make_event, event_unitary, sample_rte_events


def event_context(adapter):
    if not adapter.randomized:
        return (), {}
    if adapter.representation == "pauli_jw":
        components = adapter.components
        matrices = {str(j): np.sign(c)*pauli_matrix(s) for j,(s,c) in enumerate(adapter.tail)}
    else:
        from .df_rte_tail import extraction_to_normalized_rte_tail
        tail = extraction_to_normalized_rte_tail(adapter.preparation.tail_extraction)
        components = tail.components
        matrices = {c.component_id: op for c,op in zip(tail.components, tail.operators, strict=True)}
    return components, matrices


def trajectory_events(adapter, config, seed):
    if not adapter.randomized:
        return [()] * config.q
    if adapter.representation == "pauli_jw":
        events = sample_rte_events(adapter.components, adapter.distribution,
                                   sample_count=config.q*config.r, seed=seed)
        return [events[i*config.r:(i+1)*config.r] for i in range(config.q)]
    from .df_partial_s2_repeated import make_df_partial_s2_repeated_request
    request = make_df_partial_s2_repeated_request(adapter.preparation, step_time=adapter.step_time,
        repetition_count=config.q, rte_config=adapter.config, rte_distribution=adapter.distribution, seed=seed)
    return [occ.events for occ in request.rte_occurrences]


def trajectory_reference(adapter, config, sequences, matrices):
    full, blocks, tail, scalar = adapter.operators()
    result = np.eye(full.shape[0], dtype=complex)
    forward = [expm(-.5j*adapter.step_time*block.toarray()) for block in blocks]
    for events in sequences:
        for block in forward:
            result = block@result
        for event in events:
            result = event_unitary(event, matrices)@result
        for block in reversed(forward):
            result = block@result
        result *= np.exp(-1j*scalar*adapter.step_time)
    return result


def validate_case(adapter, state, config, result, plan):
    tolerance = plan["numerical"]["trajectory_unitary_agreement"]
    components, matrices = event_context(adapter)
    gates = {}
    forced = []
    if adapter.randomized:
        reconstructed = adapter.lambda_r * sum((c.probability*matrices[c.component_id] for c in components))
        gates["tail_operator"] = check(np.linalg.norm(reconstructed-adapter.operators()[2].toarray(), 2),
                                        plan["numerical"]["Hamiltonian_operator_agreement"], "TAIL_OPERATOR")
        for c in components:
            op = matrices[c.component_id]
            check(np.linalg.norm(op-op.conj().T, 2), tolerance, "INVOLUTION_HERMITICITY")
            check(np.linalg.norm(op@op-np.eye(len(state)), 2), tolerance, "INVOLUTION_SQUARE")
        # Exercise each supported Taylor order explicitly as algebraic circuit
        # checks; these are not production MC samples or cost estimates.
        for order_index, order in enumerate(adapter.distribution.orders):
            indices = tuple(i % len(components) for i in range(order+1))
            event = _make_event(indices, components, adapter.distribution, order_index)
            expected = event_unitary(event, matrices)
            if adapter.representation == "pauli_jw":
                circuit = QuantumCircuit(adapter.num_qubits)
                append_pauli_event(circuit, event, adapter.labels)
            else:
                from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
                prep = adapter.preparation.rte_preparation
                circuit = QiskitDFRTEEventCircuitBuilder(basis_registry=prep.basis_registry).build_event(prep.request_for_event(event)).circuit
            forced.append(dict(order=order, gate=check(np.linalg.norm(Operator(circuit).data-expected, 2),
                                                      tolerance, "FORCED_ORDER_EVENT")))
    order_counts = Counter()
    trajectory_checks = []
    for index, row in enumerate(result["circuit_cost"]["samples"]):
        sequences = trajectory_events(adapter, config, row["seed"])
        order_counts.update(e.taylor_order for events in sequences for e in events)
        if index < plan["evaluation"]["independent_trajectory_operator_checks"]:
            expected = trajectory_reference(adapter, config, sequences, matrices)
            actual = Operator(adapter.trajectory(q=config.q, seed=row["seed"])).data
            gate = check(np.linalg.norm(actual-expected, 2), tolerance, "TRAJECTORY_OPERATOR")
            control = None
            if adapter.representation == "df_native":
                c = Operator(adapter.trajectory(q=config.q, seed=row["seed"], controlled=True)).data
                n = len(state)
                reference = np.block([[np.eye(n), np.zeros((n,n))], [np.zeros((n,n)), expected]])
                control = check(np.linalg.norm(c-reference, 2), tolerance, "ORDINARY_DF_CONTROL")
            trajectory_checks.append(dict(seed=row["seed"], operator=gate, ordinary_control=control,
                                          phase_alignment=False))
    if not adapter.randomized:
        expected = dense_law_mean(adapter, q=config.q)
        actual = Operator(adapter.trajectory(q=config.q, seed=0)).data
        gates["PF_full_operator"] = check(np.linalg.norm(actual-expected, 2), tolerance, "PF_UNITARY")
    probability_rows = []
    if adapter.randomized:
        positions = len(result["circuit_cost"]["samples"])*config.q*config.r
        for order, probability in zip(adapter.distribution.orders, adapter.distribution.order_probabilities, strict=True):
            probability_rows.append(dict(order=order, probability=probability, observed=order_counts[order],
                event_positions=positions, expected=positions*probability,
                probability_of_zero=(1-probability)**positions,
                compiled_population_contribution_identified=order_counts[order]>0,
                unobserved_cost_is_zero=False))
    return dict(status="PASS", gates=gates, trajectory_operator_checks=trajectory_checks,
                forced_event_orders=forced, sampled_orders=probability_rows,
                full_MC_law_mean_independently_compiled=False,
                all_trajectories_certified=False, scientific_superiority_claimed=False)


def validate_df_primitives(adapter, plan):
    """Compare each retained native block with original second-quantized H."""
    from .df_partial_s2 import QiskitDFPartialS2CircuitBuilder, make_df_partial_s2_step_request
    prep = adapter.preparation
    request = make_df_partial_s2_step_request(prep, step_time=adapter.step_time)
    blocks = adapter.operators()[1]
    if len(blocks) != len(prep.deterministic_blocks):
        raise ValueError("DF_BLOCK_OPERATOR_ALIGNMENT")
    rows = []
    for block, op in zip(prep.deterministic_blocks, blocks, strict=True):
        for time in [adapter.step_time/2, -adapter.step_time/2]:
            circuit = QuantumCircuit(adapter.num_qubits)
            QiskitDFPartialS2CircuitBuilder()._append_block(circuit, block, request, evolution_time=time)
            error = np.linalg.norm(Operator(circuit).data-expm(-1j*time*op.toarray()), 2)
            rows.append(dict(block_id=block.block_id, time=time,
                             gate=check(error, plan["numerical"]["trajectory_unitary_agreement"], "DF_NATIVE_PRIMITIVE"),
                             phase_alignment=False))
    return rows
