"""Artificial-only compiler phase investigation; no production entry point.

Neither helper is installed in the historical native/compiler path.  A future
molecular use needs a new SOURCE and separately authorized execution contract.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(value):
    require(type(value) in (float, int), 'BOUND_REAL_PARAMETER_REQUIRED')
    if type(value) is int:
        return Fraction(value)
    require(math.isfinite(value), 'FINITE_PARAMETER_REQUIRED')
    return Fraction.from_float(value)


def payload(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def _types():
    from qiskit.circuit import Barrier, Measure
    from qiskit.circuit.library import (
        CXGate, HGate, IGate, PhaseGate, RZGate, RZZGate, SXGate, XGate,
        XXPlusYYGate,
    )
    return dict(p=PhaseGate, rz=RZGate, rzz=RZZGate, h=HGate, x=XGate,
                sx=SXGate, cx=CXGate, xx_plus_yy=XXPlusYYGate,
                id=IGate, barrier=Barrier, measure=Measure)


def _supported(op, types):
    return (op.name in types and op.base_class is types[op.name]
            and getattr(op, 'condition', None) is None)


def _rounded_phase(base):
    """Round the exact sum only once, after mathematical 2*pi reduction.

    Extra precision grows with the integer magnitude, preventing catastrophic
    cancellation for a large synthetic sum. Agreement is an internal check,
    not a certified physical error bound.
    """
    from mpmath import mp
    magnitude_digits = len(str(abs(base.numerator) // base.denominator))
    answers = []
    for guard in (80, 120):
        ctx = mp.clone()
        ctx.dps = guard + magnitude_digits
        value = ctx.mpf(base.numerator) / base.denominator
        period = 2 * ctx.pi
        reduced = value - ctx.floor(value / period) * period
        require(0 <= reduced < period, 'PHASE_REDUCTION_NOT_STABLE')
        answers.append(float(reduced))
    require(answers[0].hex() == answers[1].hex(), 'PHASE_ROUNDING_NOT_STABLE')
    return answers[1]


def lower_p_once(circuit, *, artificial=False):
    """P(theta) -> RZ(theta), with its theta/2 scalar summed exactly once.

    Retains every other operation and qubit location. No opposite-P deletion,
    gate-angle fitting, postcompile correction, or accepted-cost override.
    """
    require(artificial is True, 'ARTIFICIAL_ONLY_NOT_APPROVED_FOR_MOLECULES')
    require(not circuit.parameters, 'NO_SYMBOLIC_PARAMETERS')
    from qiskit.circuit.library import RZGate
    types = _types()
    phase = exact(circuit.global_phase)
    count = 0
    measured = False
    for item in circuit.data:
        op = item.operation
        require(_supported(op, types), 'UNSUPPORTED_OPERATION:' + op.name)
        for value in op.params:
            exact(value)
        if op.name == 'measure':
            measured = True
        else:
            require(not measured, 'NO_MID_CIRCUIT_MEASUREMENT')
        if op.name == 'p':
            phase += exact(op.params[0]) / 2
            count += 1
    result = circuit.copy_empty_like()
    for item in circuit.data:
        op = item.operation
        if op.name == 'p':
            op = RZGate(item.operation.params[0], label=item.operation.label)
        result.append(op, item.qubits, item.clbits, copy=False)
    assigned = _rounded_phase(phase)
    result.global_phase = assigned
    return result, dict(
        status='ARTIFICIAL_PROTOTYPE_NOT_ADOPTED', replaced_P=count,
        exact_global_plus_half_P_sum=payload(phase),
        assigned_global_phase_binary64_hex=assigned.hex(),
        stored_global_phase_binary64_hex=float(result.global_phase).hex(),
        gate_order_and_locations_retained=True, deleted_P_pairs=0,
        compiler_profile_adopted=False, production_installed=False,
        postcompile_correction=False, acceptance_override=False,
    )


def lower_native_once(circuit, *, artificial=False):
    """Experimental explicit native-to-basis lowering, never installed.

    Avoid parameterized R/RY equivalence routes: RY(t) is S H RZ(t) H Sdg
    in matrix order. Expand XXPlusYY using its checked built-in definition,
    then use only RZ/SX/X/CX and one exact scalar sum. Fixed Clifford angles
    are still binary64 approximations; whole-operator tests and a future raw
    molecular check are indispensable. This is a new compiler-input policy.
    """
    require(artificial is True, 'ARTIFICIAL_ONLY_NOT_APPROVED_FOR_MOLECULES')
    require(not circuit.parameters, 'NO_SYMBOLIC_PARAMETERS')
    from qiskit.circuit.library import (
        CXGate, HGate, RYGate, RZGate, SGate, SdgGate, SXGate, SXdgGate, XGate,
    )
    types = _types()
    types.update(ry=RYGate, s=SGate, sdg=SdgGate, sxdg=SXdgGate)
    result = circuit.copy_empty_like()
    phase = exact(circuit.global_phase)
    counts = Counter()

    def emit(op, qubits, clbits=()):
        nonlocal phase
        require(_supported(op, types), 'UNSUPPORTED_NATIVE_OPERATION:' + op.name)
        for value in op.params:
            exact(value)
        counts[op.name] += 1
        name = op.name
        if name == 'p':
            phase += exact(op.params[0]) / 2
            emit(RZGate(op.params[0]), qubits)
        elif name == 'h':
            # H = exp(i*pi/4) RZ(pi/2) SX RZ(pi/2).
            phase += exact(math.pi) / 4
            emit(RZGate(math.pi/2), qubits)
            emit(SXGate(), qubits)
            emit(RZGate(math.pi/2), qubits)
        elif name in ('s', 'sdg'):
            sign = 1 if name == 's' else -1
            phase += sign * exact(math.pi) / 4
            emit(RZGate(sign*math.pi/2), qubits)
        elif name == 'sxdg':
            emit(XGate(), qubits)
            emit(SXGate(), qubits)
        elif name == 'ry':
            emit(SdgGate(), qubits)
            emit(HGate(), qubits)
            emit(RZGate(op.params[0]), qubits)
            emit(HGate(), qubits)
            emit(SGate(), qubits)
        elif name == 'rzz':
            emit(CXGate(), qubits)
            emit(RZGate(op.params[0]), [qubits[1]])
            emit(CXGate(), qubits)
        elif name == 'xx_plus_yy':
            definition = op.definition
            phase += exact(definition.global_phase)
            for item in definition.data:
                local = [qubits[definition.find_bit(q).index] for q in item.qubits]
                require(not item.clbits, 'NO_CLASSICAL_XX_DEFINITION')
                emit(item.operation, local)
        else:
            result.append(op, qubits, clbits, copy=False)

    measured = False
    for item in circuit.data:
        if item.operation.name == 'measure':
            measured = True
        else:
            require(not measured, 'NO_MID_CIRCUIT_MEASUREMENT')
        emit(item.operation, list(item.qubits), list(item.clbits))
    assigned = _rounded_phase(phase)
    result.global_phase = assigned
    require(set(result.count_ops()) <= {'rz','sx','x','cx','id','barrier','measure'},
            'ONLY_REGISTERED_BASIS_REMAINS')
    return result, dict(
        status='EXPLICIT_NATIVE_BASIS_ARTIFICIAL_PROTOTYPE_NOT_ADOPTED',
        exact_scalar_before_single_rounding=payload(phase),
        visited_operation_counts=dict(sorted(counts.items())),
        assigned_global_phase_binary64_hex=assigned.hex(),
        stored_global_phase_binary64_hex=float(result.global_phase).hex(),
        changed_compiler_input_policy=True, old_cost_compatibility=False,
        Clifford_pi_policy='BINARY64_MATH_PI_EXPLICIT_NOT_EXACT_CLIFFORD_CERTIFICATE',
        production_installed=False, postcompile_correction=False,
        acceptance_override=False,
    )


def snapshot(dag):
    """Copy scalar metadata only; no mutable DAG or operation references.

    For >=3 qubits every registered H/X/SX/CX constant determinant contribution
    is 1 on the full system. Unknown/classically conditioned gates give null
    scalar evidence, never a complete determinant certificate.
    """
    require(dag.num_qubits() >= 3, 'AT_LEAST_THREE_QUBITS_FOR_CONSTANT_DROP')
    types = _types()
    counts = Counter()
    unknown = Counter()
    phase = exact(dag.global_phase)
    angle_sum = Fraction(0)
    stream = hashlib.sha256()
    for node in dag.topological_op_nodes():
        op = node.op
        counts[op.name] += 1
        supported = _supported(op, types)
        if not supported:
            unknown[op.name] += 1
        params = ([payload(exact(value)) for value in op.params] if supported else
                  [dict(unsupported_parameter_type=type(value).__name__)
                   for value in op.params])
        if op.name == 'measure':
            # Topological iteration may move a terminal measurement ahead of
            # independent gates on other qubits; only its own wire matters.
            require(all(not hasattr(successor, 'op')
                        for successor in dag.quantum_successors(node)),
                    'NONTERMINAL_MEASUREMENT_IN_TRACE')
        if op.name == 'p' and _supported(op, types):
            angle_sum += exact(op.params[0])
        row = dict(name=op.name,
                   qubits=[dag.find_bit(q).index for q in node.qargs],
                   clbits=[dag.find_bit(q).index for q in node.cargs], params=params)
        stream.update((json.dumps(row, sort_keys=True) + '\n').encode())
    return dict(
        status='SUPPORTED_SCALAR_METADATA' if not unknown else 'UNSUPPORTED_DETERMINANT_NULL',
        num_qubits=dag.num_qubits(), gate_counts=dict(sorted(counts.items())),
        unknown_gates=dict(sorted(unknown.items())),
        global_phase_exact=payload(phase),
        P_angle_sum_exact=payload(angle_sum) if not unknown else None,
        F_global_plus_half_P_exact=payload(phase + angle_sum/2) if not unknown else None,
        DAG_topological_metadata_sha256=stream.hexdigest(),
        order_hash_is_QPY_serial_order=False, gate_matrices_evaluated=0,
        determinant_modulo_evaluated=False, acceptance_override=False,
    )


class ArtificialPassRecorder:
    """Qiskit callback prototype; collects evidence and never mutates the DAG."""

    def __init__(self, *, artificial=False):
        require(artificial is True, 'ARTIFICIAL_ONLY_NOT_APPROVED_FOR_MOLECULES')
        self.rows = []

    def boundary(self, circuit, label):
        from qiskit.converters import circuit_to_dag
        self.rows.append(dict(event=len(self.rows), boundary=label,
                              snapshot=snapshot(circuit_to_dag(circuit))))

    def __call__(self, *, pass_, dag, time, property_set, count):
        self.rows.append(dict(event=len(self.rows), pass_name=pass_.name(),
                              callback_count=count, pass_seconds=time,
                              snapshot=snapshot(dag)))


if __name__ == '__main__':
    print(json.dumps(dict(
        status='ARTIFICIAL_PROTOTYPE_ONLY_NO_MOLECULAR_EXECUTION_ENTRY',
        real_QPY_restores=0, new_molecular_compile=0,
        production_installed=False, real_execution_approved=False,
        old_STOP_remains_STOP=True, acceptance_override=False,
    )))
