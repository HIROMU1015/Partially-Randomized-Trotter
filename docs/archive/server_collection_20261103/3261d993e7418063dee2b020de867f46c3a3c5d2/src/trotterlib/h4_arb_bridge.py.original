"""Preparation-only native QPY / Arb bridge.

Real QPY inspection is structural. Numerical entry points reject saved-H4 and
systems with more than three total qubits BEFORE loading Arb. No grant issuer or H4 runner.
The unitary kernel tracks a global L2 error ball to avoid interval wrapping.
"""
from fractions import Fraction as Q
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIND = 'EXACT_BINARY64_NATIVE_PROGRAM_V1'
PRECISIONS = (64, 96, 128)
NATIVE = dict(rz=(1, 1), sx=(1, 0), x=(1, 0), cx=(2, 0))
_C = None


def require(value, reason):
    if not value: raise ValueError(reason)


def exact_binary64(value):
    require(type(value) in (float, int), 'BINARY64_SCALAR_REQUIRED')
    require(math.isfinite(float(value)) and Q(float(value)) == Q(value), 'EXACT_FINITE_BINARY64')
    return Q(float(value))


def encoded_parameter(text):
    require(type(text) is str, 'EXACT_RATIONAL_STRING_REQUIRED')
    value = Q(text)
    require(exact_binary64(float(value)) == value, 'ENCODED_BINARY64_PARAMETER')
    return value


def validate_program(program):
    require(program['kind'] == KIND and program['domain'] in ('ARTIFICIAL', 'SAVED_QPY_NO_EXECUTION'), 'PROGRAM_KIND_DOMAIN')
    n, anc = program['n_qubits'], program['ancilla']
    require(type(n) is int and n >= 2 and program['n_system'] == n - 1 and anc == n - 1, 'PROGRAM_QUBIT_LAYOUT')
    require(program['axis'] in ('cosine', 'sine') and program['measurement'] == dict(qubit=anc, clbit=0, terminal_only=True), 'FINAL_ANCILLA_MEASUREMENT')
    encoded_parameter(program['global_phase_exact'])
    for gate in program['instructions']:
        require(gate['name'] in NATIVE, 'UNSUPPORTED_NATIVE_GATE')
        arity, count = NATIVE[gate['name']]
        wires = gate['qubits']
        require(len(wires) == arity and len(set(wires)) == arity and all(type(q) is int and 0 <= q < n for q in wires), 'GATE_WIRES')
        require(len(gate['params_exact']) == count, 'GATE_PARAMETERS')
        for value in gate['params_exact']: encoded_parameter(value)
    return program


def descriptor_from_circuit(circuit, *, axis, domain='SAVED_QPY_NO_EXECUTION', fixture_id=None):
    # The R0 reader verifies native gates, layout, parameters and final measure.
    from . import h4_r0_empirical as R
    signature = R.validate_program(circuit, compiled=True, n=circuit.num_qubits - 1)
    from qiskit.circuit import Measure
    from qiskit.circuit.library import RZGate, SXGate, XGate, CXGate
    measure = circuit.data[-1].operation
    require(measure.base_class is Measure and getattr(measure, 'condition', None) is None, 'STANDARD_UNCONDITIONAL_MEASUREMENT')
    classes = dict(rz=RZGate, sx=SXGate, x=XGate, cx=CXGate)
    for item in circuit.data[:-1]:
        require(item.operation.base_class is classes[item.operation.name], 'STANDARD_NATIVE_GATE_CLASS')
        if item.operation.name == 'cx': require(item.operation.ctrl_state == 1, 'CLOSED_CX_CONTROL')
    program = dict(kind=KIND, domain=domain, fixture_id=fixture_id, n_system=circuit.num_qubits - 1,
        n_qubits=circuit.num_qubits, ancilla=circuit.num_qubits - 1, axis=axis,
        global_phase_exact=str(exact_binary64(signature['global_phase'])),
        instructions=[dict(name=g['name'], qubits=g['qubits'], params_exact=[str(exact_binary64(x)) for x in g['params']])
                      for g in signature['instructions'][:-1]],
        measurement=dict(qubit=circuit.num_qubits - 1, clbit=0, terminal_only=True))
    return validate_program(program)


def inspect_qpy(path, *, axis, expected_sha256, expected_fingerprint):
    from . import h4_r0_empirical as R
    path = Path(path)
    require(R.C.sha(path) == expected_sha256, 'ACTUAL_QPY_SHA')
    with path.open('rb') as stream: values = R.api().qpy.load(stream)
    require(len(values) == 1 and R.api().fingerprint(values[0]) == expected_fingerprint, 'ACTUAL_QPY_FINGERPRINT')
    program = descriptor_from_circuit(values[0], axis=axis)
    program['origin'] = dict(qpy_sha256=expected_sha256, fingerprint=expected_fingerprint,
                            decode_only=True, state_actions=0, compile=0)
    return program


def _backend():
    global _C
    if _C is None:
        spec = importlib.util.spec_from_file_location('arb_bridge_reused_C', Path(__file__).with_name('h4_ball_evidence.py'))
        _C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_C)
    return _C


def _artificial_guard(program, state, bits, fixture_id):
    # This scope bound is an authorization guard, not a CPU/RAM limit.
    require(program.get('domain') == 'ARTIFICIAL' and 1 <= program.get('n_system', 999) <= 2, 'NO_REAL_H4_NUMERICAL_ENTRYPOINT')
    require(type(fixture_id) is str and fixture_id.startswith('TOY_') and program.get('fixture_id') == fixture_id, 'ARTIFICIAL_FIXTURE_BINDING')
    require(type(bits) is int and bits in PRECISIONS, 'FINITE_PRECISION_SCHEDULE')
    validate_program(program)
    require(len(state) == 1 << program['n_system'], 'RAW_STATE_DIMENSION')
    for pair in state:
        require(type(pair) in (list, tuple) and len(pair) == 2, 'RAW_COMPLEX_PAIR')
        for value in pair: exact_binary64(value)


def _recenter(values, error, C):
    """Enclose G*m by exact dyadic centers plus a global Euclidean radius.

    If ||x-m||<=e and G is unitary, ||G*x-m_new||<=e+r. r encloses
    the L2 norm of all freshly computed rectangular component radii.
    No physical state renormalization or untracked midpoint substitution.
    """
    square = C.arb(0)
    center = []
    for value in values:
        require(value.real.is_finite() and value.imag.is_finite(), 'NONFINITE_GATE_ENCLOSURE')
        mr, mi = value.real.mid(), value.imag.mid()
        er, ei = (value.real - mr).abs_upper(), (value.imag - mi).abs_upper()
        square += er ** 2 + ei ** 2
        z = C.acb(mr, mi)
        require(z.real.is_exact() and z.imag.is_exact(), 'EXACT_DYADIC_CENTER')
        center.append(z)
    radius = C.dyadic((C.real(error) + square.sqrt()).upper())
    return center, radius


def _kernel(program, raw, bits, fixture_id):
    _artificial_guard(program, raw, bits, fixture_id)
    C = _backend()
    with C.ctx.workprec(bits):
        nu = sum((exact_binary64(x) ** 2 + exact_binary64(y) ** 2 for x, y in raw), Q(0))
        require(nu > 0, 'POSITIVE_RAW_NORM')
        vector = [C.complex_ball(pair) for pair in raw] + [C.acb(0) for _ in raw]
        require(all(z.real.is_exact() and z.imag.is_exact() for z in vector), 'EXACT_INITIAL_BINARY64_STATE')
        error, phases = Q(0), {}
        diag, off = C.acb(C.real(Q(1, 2)), C.real(Q(1, 2))), C.acb(C.real(Q(1, 2)), C.real(Q(-1, 2)))
        for gate in program['instructions']:
            name, wires = gate['name'], gate['qubits']
            q = wires[-1]
            mask = 1 << q
            if name in ('x', 'cx'):
                control = 1 << wires[0] if name == 'cx' else None
                for i in range(len(vector)):
                    if not i & mask and (control is None or i & control):
                        vector[i], vector[i | mask] = vector[i | mask], vector[i]
                continue  # Exact permutation, no fresh arithmetic error.
            if name == 'rz':
                theta = encoded_parameter(gate['params_exact'][0])
                if theta not in phases:
                    phases[theta] = (C.phase(-C.real(theta) / 2), C.phase(C.real(theta) / 2))
                lo, hi = phases[theta]
                values = [z * (hi if i & mask else lo) for i, z in enumerate(vector)]
            else:  # SX = ((1+i) I + (1-i) X)/2, including its global phase.
                values = list(vector)
                for i in range(len(vector)):
                    if not i & mask:
                        j = i | mask
                        values[i], values[j] = diag * vector[i] + off * vector[j], off * vector[i] + diag * vector[j]
            vector, error = _recenter(values, error, C)
        global_phase = C.phase(C.real(encoded_parameter(program['global_phase_exact'])))
        vector, error = _recenter([global_phase * z for z in vector], error, C)
        zmid, norm2 = C.arb(0), C.arb(0)
        anc = 1 << program['ancilla']
        for i, z in enumerate(vector):
            square = z.real ** 2 + z.imag ** 2
            norm2 += square
            zmid += -square if i & anc else square
        # ||Z_anc||=1: quadratic-form error <= 2||m||e + e^2.
        e = C.real(error)
        transfer = (2 * norm2.sqrt().upper() * e + e ** 2).upper()
        denominator = C.real(nu)
        require(denominator.lower() > 0, 'NORM_DENOMINATOR_EXCLUDES_ZERO')
        signal = (zmid + C.arb(0, transfer)) / denominator
        require(signal.is_finite(), 'FINITE_SIGNAL_ENCLOSURE')
        interval = C.endpoints(signal)
        record = dict(status='ARTIFICIAL_VALIDATED_SIGNAL_NO_H4_RESULT', fixture_id=fixture_id, precision_bits=bits,
            interval=interval, interval_width_exact=str(Q(interval['upper']) - Q(interval['lower'])),
            raw_norm_exact=str(nu), state_l2_error_upper_exact=str(error),
            observable_transfer_upper_exact=str(C.dyadic(transfer)), instructions=len(program['instructions']),
            normalization='same exact raw nu; no input/output rescaling', clipping=False,
            global_phase_preserved=True, real_H4_state_actions=0, formal_certificate_accepted=False)
        return signal, record


def evaluate_artificial(program, raw_state, *, bits=64, fixture_id):
    return _kernel(program, raw_state, bits, fixture_id)


def residual_artificial(signal, target_interval, *, B_saved_exact, budget_exact, bits=64, fixture_id):
    require(type(fixture_id) is str and fixture_id.startswith('TOY_') and bits in PRECISIONS, 'ARTIFICIAL_RESIDUAL_ONLY')
    C = _backend()
    with C.ctx.workprec(bits):
        lo, hi = Q(target_interval['lower']), Q(target_interval['upper'])
        b, budget = Q(B_saved_exact), Q(budget_exact)
        require(lo <= hi and b > 0 and budget > 0, 'TARGET_SCALE_BUDGET')
        target = C.real(lo).union(C.real(hi))
        rho = abs(C.real(b) * (signal - target))
        ep = C.endpoints(rho)
        lower, upper = Q(ep['lower']), Q(ep['upper'])
        relation = 'UPPER_WITHIN_BUDGET' if upper <= budget else 'LOWER_EXCEEDS_BUDGET' if lower > budget else 'UNDETERMINED'
        return rho, dict(status='ARTIFICIAL_LOCAL_NUMERICAL_CANDIDATE_NO_ACCEPTANCE', interval=ep,
            budget_relation=relation, B_saved_exact=str(b), formal_certificate_accepted=False, LOCAL_BUDGET_MET=False)


def execution_request(*_args, **_kwargs):
    raise PermissionError('PREPARATION_ONLY_NO_REAL_H4_EVALUATION_OR_GRANT')
