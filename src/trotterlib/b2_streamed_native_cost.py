"""B2 event/request streaming with the unchanged native gate emitters.

Only intermediate requests and tails are streamed. The final circuit, recursive
fingerprint and Qiskit whole compilation still occupy length-dependent memory.
No sampling, molecular input loading or execution on import.
"""
from dataclasses import replace
from itertools import islice

from qiskit import QuantumCircuit

from .df_rte_qiskit import (QiskitDFRTEEventCircuitBuilder,
    estimate_df_rte_structural_size_upper_bound)
from .rpe_resource_accounting import RPERoundSpecification
from .rte_compiled_cost import (_filtered_depth, _gate_counts,
    canonical_qiskit_circuit_fingerprint, transpile_and_measure_cost)
from trottertracks.resource_applicability import ax2a_native_df as native


def iter_b2_step_requests(round_, events):
    """Validate and replay r events at a time using B2's existing explicit API."""
    one = replace(round_, specification=RPERoundSpecification(0, round_.delta), q=1)
    stream = iter(events)
    for _ in range(round_.q):
        occurrence = tuple(islice(stream, round_.r))
        yield one.explicit_request(occurrence).iter_step_requests()[0]
    sentinel = object()
    if next(stream, sentinel) is not sentinel:
        raise ValueError('Extra events after Q*r canonical events')


def structural_instruction_bound(round_):
    p = round_.preparation
    return 1 + round_.q * (2 * sum(native.block_instruction_bound(b, 'ORDINARY')
        for b in p.deterministic_blocks) + estimate_df_rte_structural_size_upper_bound(
            p.rte_preparation.component_specs, maximum_taylor_order=round_.K,
            event_count=round_.r, controlled=True))


def stream_native(round_, events):
    """Emit the same ordinary-controlled instruction sequence as B2.native.

    Preserve the reference's T=delta*Q then dt=T/Q floating-point arithmetic,
    forward/tail/reverse ordering, and one final aggregate scalar phase. No
    fusion, normalization, threshold, basis-policy or compiler changes.
    """
    p = round_.preparation
    n = native.validate_blocks(p.deterministic_blocks, p.num_system_qubits)
    bound = structural_instruction_bound(round_)
    total_time = round_.delta * round_.q
    dt = total_time / round_.q
    circuit = QuantumCircuit(n+1, name='ax2a_df_partial_s2')
    builder = QiskitDFRTEEventCircuitBuilder(basis_registry=p.rte_preparation.basis_registry)
    for request in iter_b2_step_requests(round_, events):
        for block in p.deterministic_blocks:
            native._append_block(circuit, block, dt/2, 'ORDINARY', n)
        tail = builder.build_sequence(request.rte_occurrence,
                                      basis_plan=request.rte_basis_plan).circuit
        circuit.compose(tail, inplace=True)
        del tail
        for block in reversed(p.deterministic_blocks):
            native._append_block(circuit, block, dt/2, 'ORDINARY', n)
    scalar = p.constant_coefficient + p.extracted_identity_coefficient
    if scalar * total_time != 0:
        circuit.p(-scalar * total_time, n)
    return native._finish(circuit, n, total_time, round_.q, 'partial_2nd',
                          'ordinary', bound, bound)


def cached_compiler(cache):
    """Inject the existing circuit/compiler/backend keyed cache into a provider.

    It saves compilation work only; callers retain every observation, including
    repeated IID draws. A separate cache instance belongs to each worker.
    """
    def compile_function(circuit, compiler, *, backend=None,
                         actual_circuit_fingerprint=None):
        fingerprint = actual_circuit_fingerprint or canonical_qiskit_circuit_fingerprint(circuit)
        cost, _, _ = cache.get_or_transpile(circuit, compiler,
            circuit_fingerprint=fingerprint, backend=backend)
        return cost
    return compile_function


def measured_metrics(circuit):
    """Measure an already compiled circuit with the existing metric primitives.

    Needed for a concatenation of separately compiled blocks: recompiling that
    concatenation would turn it back into the whole-compile reference.
    """
    counts = dict(_gate_counts(circuit))
    return dict(rz_count=counts.get('rz', 0), cx_count=counts.get('cx', 0),
        rz_depth=_filtered_depth(circuit, 'rz'), cx_depth=_filtered_depth(circuit, 'cx'),
        total_depth=int(circuit.depth() or 0), circuit_size=int(circuit.size()))


def split_compile(circuit, compiler, *, pieces):
    """Diagnostic candidate only: compile contiguous pieces and compose outputs.

    Includes the exact wrapper and measurement once, and measures the composed
    DAG's depths rather than adding block depths. Whole-compile equivalence is
    deliberately not asserted. No compiler/backend fallback.
    """
    if type(pieces) is not int or not 1 <= pieces <= len(circuit.data):
        raise ValueError('Valid explicit partition count required')
    result = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    sums = dict.fromkeys(measured_metrics(result), 0)
    for i in range(pieces):
        chunk = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        if i == 0:
            chunk.global_phase = circuit.global_phase
        for item in circuit.data[i*len(circuit.data)//pieces:(i+1)*len(circuit.data)//pieces]:
            chunk.append(item.operation,
                [circuit.find_bit(q).index for q in item.qubits],
                [circuit.find_bit(c).index for c in item.clbits])
        cost = transpile_and_measure_cost(chunk, compiler, backend=None)
        result.compose(cost.transpiled_circuit, inplace=True)
        for name in sums:
            sums[name] += getattr(cost, name)
    return result, sums
