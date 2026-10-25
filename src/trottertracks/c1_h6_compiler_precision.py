"""Keep P0 gate compilation, stabilize only declared global-phase accumulation.

No output state/operator is inspected or fitted. A symbolic unit offset keeps
the existing Qiskit/SymEngine MPFR number in phase additions during P0. Bind
the exact compensating -1 once after compilation and cast to binary64 once.
"""
VERSION='qiskit_P0_mpfr160_declared_phase_v1'
PHASE_BITS=160


def compiled_wrapper(circuit,profile):
    import symengine
    from qiskit.circuit import Parameter,ParameterExpression
    from .pr_dictionary_validation.gates import compiled_wrapper as existing
    if circuit.parameters:raise ValueError('BOUND_INPUT_CIRCUIT_REQUIRED')
    marker=Parameter('__c1_declared_phase_accumulator')
    offset=ParameterExpression(marker._parameter_symbols,
        marker._symbol_expr+symengine.RealMPFR('1.0',PHASE_BITS))
    tracked=circuit.copy();tracked.global_phase=tracked.global_phase+offset
    compiled=existing(tracked,profile)
    phase=compiled.global_phase
    if (not isinstance(phase,ParameterExpression) or set(phase.parameters)!={marker}
            or float(phase.gradient(marker))!=1.):
        raise ValueError('DECLARED_PHASE_TRACKING_LOST')
    # Qiskit1.3 UnrollCustomDefinitions omits global-only parameters from its
    # circuit ParameterView. Bind the declared phase expression directly.
    compiled.global_phase=float(phase.bind({marker:-1.}))
    if compiled.parameters:raise ValueError('UNBOUND_COMPILED_CIRCUIT')
    return compiled
