"""Single-owner wrapper construction and scoped compile telemetry.

No compiler or gate sequence changes. Synthetic bridge tests compare the full
canonical numeric fingerprint and every compiled metric with frozen v2 paths.
Real H6/q8 feasibility remains to be measured by a separately granted run.
"""
import gc
from qiskit import ClassicalRegister
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import HGate
from trotterlib import rte_compiled_cost as legacy
from .ax2b_native_df_v5 import NativeEvolution, _gate_only, _limit
from .ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint


def take_wrapper(owner, *, axis, include_measurement=True, max_instructions):
    """Consume a one-element ownership box instead of copying its evolution.

    Callers must not reuse the consumed evolution. The circuit has the same
    qregs, ordered gates, global phase, classical register and numeric payload
    as build_native_hadamard_wrapper. No QPY/type conversion or block sums.
    """
    if len(owner)!=1 or not isinstance(owner[0],NativeEvolution) or axis not in ('cosine','sine'):
        raise ValueError('WRAPPER_OWNERSHIP')
    if type(include_measurement) is not bool:raise ValueError('WRAPPER_MEASUREMENT')
    e=owner[0];n=e.num_system_qubits;c=e.circuit
    _gate_only(c,n+1)
    if canonical_qiskit_circuit_fingerprint(c)!=e.circuit_fingerprint:raise ValueError('EVOLUTION_MUTATED')
    if len(c.data)+2+(axis=='sine')+include_measurement>_limit(max_instructions):
        raise RuntimeError('PRE_BUILD_INSTRUCTION_BUDGET')
    owner.pop()
    c.name='rpe_hadamard_interrogation'
    c.data.insert(0,CircuitInstruction(HGate(),(c.qubits[n],),()))
    if axis=='sine':c.sdg(n)
    c.h(n)
    if include_measurement:
        c.add_register(ClassicalRegister(1,'rpe_measure'));c.measure(n,0)
    return c


def compile_owned(owner, compiler, *, axis, max_instructions, transpiled_cap, trace, before_compile=lambda:None):
    with trace.stage('wrapper_build',axis=axis,instructions=len(owner[0].circuit.data)):
        wrapper=take_wrapper(owner,axis=axis,max_instructions=max_instructions)
    before_compile()
    with trace.stage('wrapper_fingerprint',axis=axis,instructions=len(wrapper.data)):
        fingerprint=canonical_qiskit_circuit_fingerprint(wrapper)
        pre=legacy._gate_counts(wrapper)
    legacy._validate_transpile_context(compiler,None)
    if (compiler.coupling_map is not None or compiler.layout_method is not None or
        compiler.routing_method is not None):raise ValueError('FIXED_COMPILER_CONTEXT')
    with trace.stage('transpile',axis=axis,instructions=len(wrapper.data)):
        compiled=legacy.transpile(wrapper,basis_gates=list(compiler.basis_gates),
            optimization_level=compiler.optimization_level,seed_transpiler=compiler.transpiler_seed)
    del wrapper;gc.collect()
    if len(compiled.data)>transpiled_cap:raise RuntimeError('TRANSPILED_INSTRUCTION_CAP')
    with trace.stage('metric_measurement',axis=axis,instructions=len(compiled.data)):
        counts=dict(legacy._gate_counts(compiled))
        return legacy.TranspiledCircuitCost(pretranspile_gate_counts=pre,
            posttranspile_gate_counts=tuple(sorted(counts.items())),
            rz_count=counts.get('rz',0),rz_depth=legacy._filtered_depth(compiled,'rz'),
            cx_count=counts.get('cx',0),cx_depth=legacy._filtered_depth(compiled,'cx'),
            total_depth=int(compiled.depth() or 0),circuit_size=int(compiled.size()),
            qubit_count=compiled.num_qubits,global_phase=float(compiled.global_phase),
            compiler=compiler,compiler_settings_hash=legacy.compiler_settings_hash(compiler),
            circuit_fingerprint=fingerprint,actual_circuit_fingerprint=fingerprint,
            backend_fingerprint=None,transpiled_circuit=compiled)
