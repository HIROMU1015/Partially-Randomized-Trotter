"""Fixed Pauli parity IR and bounded native/T inventory accounting."""
from collections import Counter
from fractions import Fraction as F
import math
import mpmath as mp
from .angles import Angle, clifford_sequence
from .intervals import iv, upper, bounds


def pauli_ir(label, signed_alpha, controlled):
    """Native basis only; global phase of H/S decompositions is retained."""
    if not any(c != 'I' for c in label):
        raise ValueError('Identity Pauli handled as a scalar identity event')
    alpha = float(signed_alpha)
    support = [i for i, c in enumerate(reversed(label)) if c != 'I']
    gates, phase = [], F(0)
    def h(q):
        nonlocal phase
        gates.extend([('rz', [q], math.pi/2), ('sx', [q], None), ('rz', [q], math.pi/2)])
        phase += F(1, 4)
    def s(q, sign):
        nonlocal phase
        gates.append(('rz', [q], sign*math.pi/2))
        phase += sign*F(1, 4)
    for q, char in enumerate(reversed(label)):
        if char == 'Y': s(q, -1)
        if char in 'XY': h(q)
    for q in support[:-1]: gates.append(('cx', [q, support[-1]], None))
    if controlled:
        ancilla = len(label)
        gates.extend([('rz', [support[-1]], -alpha), ('cx', [ancilla, support[-1]], None),
                      ('rz', [support[-1]], alpha), ('cx', [ancilla, support[-1]], None)])
    else:
        gates.append(('rz', [support[-1]], -2*alpha))
    for q in reversed(support[:-1]): gates.append(('cx', [q, support[-1]], None))
    for q, char in reversed(list(enumerate(reversed(label)))):
        if char in 'XY': h(q)
        if char == 'Y': s(q, 1)
    return {'qubits': len(label)+int(controlled), 'label': label,
            'controlled': controlled, 'gates': gates, 'global_phase_pi': str(phase),
            'arbitrary_angle_RZ': 2 if controlled else 1}


def ir_counts(ir):
    counts = Counter(g[0] for g in ir['gates'])
    return {'CX': counts['cx'], 'total_RZ': counts['rz'], 'arbitrary_angle_RZ': ir['arbitrary_angle_RZ'],
            '1Q': sum(v for k, v in counts.items() if k != 'cx'), 'gate_counts': dict(counts)}


def circuit_from_ir(ir):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(ir['qubits'])
    qc.global_phase = float(F(ir['global_phase_pi']))*math.pi
    for name, qubits, angle in ir['gates']:
        if name == 'rz': qc.rz(angle, qubits[0])
        elif name == 'cx': qc.cx(*qubits)
        elif name == 'sx': qc.sx(qubits[0])
        elif name == 'x': qc.x(qubits[0])
        else: raise ValueError('Unexpected native gate')
    return qc


def native_inventory(circuit, *, semantic_alpha=None):
    """Record EVERY key occurrence, with phase/binary64 approximation distance.

    Clifford recognition is an interval proximity proof charged to numerical
    error, not an assertion that a binary64 approximation of pi is exact.
    """
    counts, occurrences = Counter(), []
    angle_distance = F(0)
    for position, item in enumerate(circuit.data):
        name = item.operation.name
        if name not in ('rz', 'sx', 'x', 'cx'):
            raise ArithmeticError('Native basis not closed')
        counts[name] += 1
        if name != 'rz': continue
        raw = F(float(item.operation.params[0]))
        angle = Angle('raw-rad', raw)
        distance = F(0)
        k = round(float(raw)*2/math.pi)
        candidate = Angle('pi', F(k, 2))
        r = candidate.at(mp.iv)-iv(raw)
        lo, hi = bounds(r)
        if max(abs(lo), abs(hi))/2 <= F(1, 160000000000):
            angle = candidate
            distance = max(abs(lo), abs(hi))/2
        elif semantic_alpha is not None:
            for scale in (-2, -1, 1, 2):
                candidate = Angle('atan', semantic_alpha, F(scale))
                lo, hi = bounds(candidate.at(mp.iv)-iv(raw))
                if max(abs(lo), abs(hi))/2 <= F(1, 160000000000):
                    angle = candidate
                    distance = max(abs(lo), abs(hi))/2
                    break
        angle_distance += distance
        occurrences.append({'position': position, 'raw_binary64_radians': str(raw),
                            'angle': angle.as_dict(), 'native_to_semantic_operator_error': str(distance)})
    phase = F(float(circuit.global_phase))
    k = round(float(phase)*4/math.pi)
    lo, hi = bounds(iv(phase)-iv(F(k, 4))*mp.iv.pi)
    phase_distance = max(abs(lo), abs(hi))
    phase_valid = phase_distance <= F(1, 100000000)
    return {'gate_counts': dict(counts), 'occurrences': occurrences,
            'global_phase_raw': str(phase), 'global_phase_W_exponent': k % 8 if phase_valid else None,
            'global_phase_error_upper': str(phase_distance), 'phase_supported': phase_valid,
            'angle_lowering_error_upper': str(angle_distance), 'arbitrary_angle_RZ': sum(
                clifford_sequence(Angle.load(r['angle'])) is None for r in occurrences),
            'total_RZ': counts['rz'], 'CX': counts['cx'],
            '1Q': sum(v for k, v in counts.items() if k != 'cx')}


def selected_positions(count, slots):
    if count <= slots: return list(range(count))
    return [j*(count-1)//(slots-1) for j in range(slots)]


def select_synthesis_inventory(circuits, b_angles):
    """Fixed <=64 slots, B first; no filling dedup/exact-path free slots."""
    b = [a for r in (3, 6) for a in b_angles[r]]
    phase = []  # W phases are exact symbolic; unsupported phases -> inconclusive.
    a = []
    for r in (3, 6):
        for controlled in (False, True):
            rows = circuits[(r, controlled)]['occurrences']
            slots = 16 if controlled else 8
            a += [Angle.load(rows[j]['angle']) for j in selected_positions(len(rows), slots)]
    logical = b+phase+a
    if len(logical) > 64: raise RuntimeError('Logical synthesis slots exceeded')
    ordered = {}
    for angle in logical:
        ordered.setdefault(angle.key, angle.as_dict())
    return {'logical_slots': len(logical), 'unique_keys': list(ordered.values()),
            'B_required_keys': list(dict.fromkeys(x.key for x in b)),
            'phase_required_keys': list(dict.fromkeys(x.key for x in phase)),
            'order': 'B_COMPLETE_THEN_PHASE_COMPLETE_THEN_REGISTERED_A_POSITIONS', 'fill_empty_slots': False}


def t_completion(inventory, sequences, epsilon=F(1, 40000000000)):
    """Completion lower for this fixed additive plan, NOT a compiler lower bound."""
    known, unknown = 0, Counter()
    for row in inventory['occurrences']:
        angle = Angle.load(row['angle'])
        seq = clifford_sequence(angle)
        if seq is not None:
            continue
        result = sequences.get(angle.key)
        if result is None:
            unknown[angle.key] += 1
        else:
            if not result.get('error_pass') or F(result['strict_operator_error_upper']) > epsilon:
                raise ArithmeticError('Invalid sequence cannot enter T lower bound')
            s = result['sequence']
            import hashlib
            if result['angle']['key'] != angle.key or hashlib.sha256(s.encode()).hexdigest() != result['sequence_sha256']:
                raise ArithmeticError('Sequence/key identity mismatch')
            actual = s.count('T')+s.count('t')
            if actual != result['T_count']:
                raise ArithmeticError('Sequence T identity mismatch')
            known += actual
    phase = inventory['phase_supported']
    numeric = inventory.get('numeric_operator_error_upper')
    complete = (not unknown and phase and inventory['total_RZ'] <= 32000
                and numeric is not None and F(numeric) <= F(1,5000000))
    return {'fixed_plan_completion_T_lower': known if phase else None,
            'actual_total_T': known if complete else None,
            'unknown_multiplicities': dict(unknown), 'all_keys_covered': not unknown,
            'global_phase_supported': phase, 'full_circuit_synthesis_certified': complete,
            'A_TOTAL_T_UNKNOWN': not complete,
            'classification': 'FIXED_ADDITIVE_T_COMPLETE' if complete else 'T_INCONCLUSIVE',
            'bound_scope': 'fixed_obtained_sequences_additive_completion_only_no_cross_gate_cancellation'}


def local_t_separation(a, b):
    lo, hi = a['fixed_plan_completion_T_lower'], b['actual_total_T']
    return {'FIXED_PLAN_LOCAL_T_SEPARATION': lo is not None and hi is not None and lo > hi,
            'A_TOTAL_T_UNKNOWN': a['A_TOTAL_T_UNKNOWN'], 'PR_total_winner': None}
