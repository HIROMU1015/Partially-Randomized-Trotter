"""Dormant numerical gates. No gate is run in stage I."""
from __future__ import annotations
from fractions import Fraction
import math


def matrix_gate(actual, reference, tolerance, *, label):
    import numpy as np
    a=np.asarray(actual,dtype=complex); b=np.asarray(reference,dtype=complex)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(label+': shape/nonfinite')
    error=a-b
    report=dict(label=label,max_entry=float(np.max(np.abs(error))),
                frobenius=float(np.linalg.norm(error)),
                spectral=float(np.linalg.norm(error,ord=2)),tolerance=tolerance,rtol=0)
    if report['max_entry']>tolerance: raise ValueError(str(report))
    return report


def controlled_gate(circuit, system_reference, tolerance=1e-10):
    import numpy as np
    from qiskit.quantum_info import Operator
    u=Operator(circuit).data; n=len(system_reference)
    expected=np.zeros((2*n,2*n),complex); expected[:n,:n]=np.eye(n)
    expected[n:,n:]=system_reference
    report=matrix_gate(u,expected,tolerance,label='CONTROLLED_NO_PHASE_FIT')
    report['unitarity']=matrix_gate(u.conj().T@u,np.eye(2*n),1e-10,label='UNITARITY')
    return report


def wrapper_signal(circuit, axis, *, measured=True):
    from qiskit.quantum_info import Statevector
    if axis not in ('X','Y'): raise ValueError('Axis')
    unitary=circuit.copy()
    if measured:
        positions=[i for i,x in enumerate(unitary.data) if x.operation.name=='measure']
        if positions!=[len(unitary.data)-1]: raise ValueError('Only one terminal measure allowed')
        item=unitary.data[-1]
        if unitary.find_bit(item.qubits[0]).index!=unitary.num_qubits-1:
            raise ValueError('Measurement is not ancilla-last')
        unitary.remove_final_measurements(inplace=True)
    if any(x.operation.name in ('measure','reset') for x in unitary.data):
        raise ValueError('Nonunitary wrapper')
    sv=Statevector.from_instruction(unitary)
    half=len(sv.data)//2
    return float(sum(abs(x)**2 for x in sv.data[:half])-sum(abs(x)**2 for x in sv.data[half:]))


def signal_gate(actual_x,actual_y,reference,tolerance=2e-10):
    value=complex(actual_x,actual_y)
    if not all(math.isfinite(x) for x in (value.real,value.imag)):
        raise ValueError('Nonfinite signal')
    residual=abs(value-complex(reference))
    if residual>tolerance: raise ValueError('SIGNAL error '+str(residual))
    return dict(error=residual,tolerance=tolerance,actual_real=value.real,actual_imag=value.imag)


def high_precision_agreement(a,b):
    # Cast upward before subtraction; never collapse the 90/128 check to float64.
    from .reference import precision_context
    ctx=precision_context(128)
    if (a.rows,a.cols)!=(b.rows,b.cols): raise ValueError('Reference dimensions')
    residual=max(abs(ctx.mpc(a[i,j])-b[i,j]) for i in range(a.rows) for j in range(a.cols))
    if residual>ctx.mpf('1e-60'): raise ValueError('REFERENCE_PRECISION_DISAGREEMENT')
    return str(residual)


def compile_cache_key(circuit, profile, identity_context):
    """Full serialized IR identity, also used for F2 prebinding after approval."""
    from qiskit import qpy
    import io
    import hashlib
    from dataclasses import asdict
    from .fixture import canonical_json
    profile.validate()
    if not identity_context:
        raise ValueError('Compile cache requires full bound context')
    # Keep all angles, global phase and register layout. Ignore descriptions only.
    normalized=circuit.copy(); normalized.metadata={}; normalized.name='full_ir'
    stream=io.BytesIO(); qpy.dump(normalized,stream)
    return hashlib.sha256(stream.getvalue()+canonical_json(dict(
        profile=asdict(profile),identity_context=identity_context))).hexdigest()


def compiled_wrapper(circuit, profile, *, cache=None, identity_context=None):
    from qiskit import transpile
    from qiskit.circuit.equivalence import EquivalenceLibrary
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import UnrollCustomDefinitions
    profile.validate()
    cache_key=None
    if cache is not None:
        cache_key=compile_cache_key(circuit,profile,identity_context)
        if cache_key in cache:
            return cache[cache_key].copy()
    # Qiskit 1.3's nested equivalence translation can discard global phase
    # (reproduced with RY -> R -> U3 and Gaussian XXPlusYY gates). Unroll
    # official gate definitions to U plus the fixed native basis first. This
    # preserves each definition's phase without fitting an output operator or
    # changing the shared SessionEquivalenceLibrary, environment or profile.
    lowered=PassManager([UnrollCustomDefinitions(EquivalenceLibrary(),
        list(profile.basis_gates)+['u'])]).run(circuit)
    compiled=transpile(lowered,basis_gates=list(profile.basis_gates),
        optimization_level=profile.optimization_level,seed_transpiler=profile.seed_transpiler,
        backend=None,target=None,coupling_map=None,initial_layout=None,
        layout_method=None,routing_method=None,translation_method=None,scheduling_method=None,
        approximation_degree=profile.approximation_degree,
        qubits_initially_zero=profile.qubits_initially_zero)
    if compiled.layout is not None:
        permutation=compiled.layout.final_index_layout(filter_ancillas=True)
        if permutation!=list(range(circuit.num_qubits)):
            # P0 permits tracking; this minimal implementation refuses unknown permutations.
            raise ValueError('Nonidentity layout needs explicit reviewed mapping')
    if cache is not None:
        cache[cache_key]=compiled.copy()
    return compiled


def component_derived_mean(tail, distribution, blocks_by_role):
    """Use actual control1 blocks, not candidate R fed into a Taylor helper."""
    import numpy as np
    n=len(next(iter(blocks_by_role.values())))
    p=np.zeros((n,n),complex)
    mass=math.fsum(c.probability for c in tail.components)
    for component in tail.components:
        p+=(component.probability/mass)*blocks_by_role[(component.component_id,'product')]
    result=np.zeros_like(p)
    for order,weight in zip(distribution.orders,distribution.unnormalized_order_weights,strict=True):
        v=np.zeros_like(p)
        for component in tail.components:
            v+=(component.probability/mass)*blocks_by_role[(component.component_id,'rotation',order)]
        result+=weight*((-1)**(order//2))*v@np.linalg.matrix_power(p,order)
    return result


def compose_corrected_mean(step_mean,spec,context):
    import numpy as np
    from .fixture import fraction
    energies=np.array([sum(float(fraction(x))*((index>>p)&1)
        for p,x in enumerate(spec.deterministic)) for index in range(1<<spec.modes)])
    half=np.diag(np.exp(-1j*float(context.delta)*energies/2))
    macro=half@np.linalg.matrix_power(step_mean,context.r)@half
    return np.exp(-1j*float(fraction(spec.scalar))*float(fraction(context.total_time)))*np.linalg.matrix_power(macro,context.q)


def shot_count_from_enclosure(lower:Fraction,upper:Fraction):
    def ceiling(x): return -((-x.numerator)//x.denominator)
    lo,hi=ceiling(lower),ceiling(upper)
    if lower<0 or upper<lower or lo!=hi: raise ValueError('SHOT_COUNT_UNRESOLVED')
    return lo
