"""Explicit B2 trajectory pricing through the existing full no-prep pipeline.

No sampling, input loading, signal evaluation, or execution on import. A fixed
trajectory price is deliberately not an RPERoundCompiledCost expectation.
"""
from .b2_rpe_connection import _check_provider
from .df_rte_qiskit import estimate_df_rte_untranspiled_size_upper_bound
from .o1_m1_resource_bridge import _compile_pair
from .pauli_signal_adapter import digest
from .rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE
from .rpe_resource_accounting import RPE_COST_METRICS
from .rte_compiled_cost import compiler_settings_hash
from trottertracks.resource_applicability.ax2a_native_df import (
    block_instruction_bound, build_native_hadamard_wrapper,
)


def request_instruction_bound(request):
    """Reuse native pre-build bounds; this is not a compiled-price estimate."""
    return (request.repetition_count * 2 * sum(
        block_instruction_bound(b, 'ORDINARY')
        for b in request.preparation.deterministic_blocks) + 1 + sum(
        estimate_df_rte_untranspiled_size_upper_bound(o)
        for o in request.rte_occurrences))


def price_explicit_trajectory(round_, provider, events):
    """Compile both measured axes of one canonical explicit native trajectory.

    Reuse the same pair compiler as MatchedNoPrepCostProvider and B2's MC path.
    Derive the instruction allowance from the request, without a resource cap.
    Callers must fix the eligible scientific/compiler context before building.
    """
    _check_provider(round_.context, provider)
    events = tuple(events)
    request = round_.explicit_request(events)
    bound = request_instruction_bound(request)
    evolution = round_.native(request, max_instructions=bound)
    pair = _compile_pair(evolution, provider.compiler,
        compile_function=provider.compile_function, backend=provider.backend,
        wrapper_builder=lambda value, axis: build_native_hadamard_wrapper(
            value, axis=axis, include_measurement=True, max_instructions=bound+4))
    trajectory = [e.to_dict() for e in events]
    return dict(evaluation_method='specified_trajectory_full_no_prep_compiled_cost',
        circuit_cost_scope=RPE_HADAMARD_INTERROGATION_SCOPE,
        state_preparation_included=False, round_index=round_.specification.round_index,
        q=round_.q, delta=round_.delta, round_binding=round_.binding,
        context_hash=round_.context_hash, preparation_hash=round_.preparation_hash,
        trajectory=trajectory, trajectory_identity=digest(trajectory),
        evolution_fingerprint=evolution.circuit_fingerprint,
        evolution_instructions=len(evolution.circuit.data),
        request_instruction_upper_bound=bound,
        native_instruction_upper_bound=evolution.instruction_upper_bound,
        compiler_settings_hash=compiler_settings_hash(provider.compiler),
        backend_fingerprint=provider.backend_hash,
        axes={axis: {name: getattr(cost, name) for name in RPE_COST_METRICS}
              for axis, cost in pair.items()},
        expected_trajectory_cost=None, classical_sample_count=None,
        standard_error=None, confidence_interval=None, total_rpe_cost=None)
