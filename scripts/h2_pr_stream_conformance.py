"""Recognize saved QPY fields, never generate/lower/evaluate an H2 circuit.

This is an online cursor over existing serialized records. Inverse checking
negates only the saved gate's first numeric parameter and reverses saved basis
order (fixed Qiskit Phase/XXPlusYY rules). No Hamiltonian coefficient, rotation,
diagonal primitive, matrix, gate object or output state is computed here.
Deterministic diagonal magnitudes are NOT certified against fresh preparation.
"""
import math
from itertools import combinations
import h2_pr_identity_recovery as r
import h2_pr_saved_qpy as qpy
import audit_h2_pr_original_identity as original


def number(value):
    r.require(value.get('kind') == 'float', 'NUMERIC_SAVED_PARAMETER_REQUIRED')
    x = float.fromhex(value['hex'])
    r.require(math.isfinite(x), 'FINITE_SAVED_PARAMETER')
    return x


class Cursor:
    def __init__(self, circuit):
        self.circuit = circuit
        self.items = circuit['instructions']
        self.pos = 1  # initial Hadamard checked separately
        self.counts = dict(forward_basis_groups=0, inverse_basis_groups=0,
                           reuse_boundaries=0, identity_flushes=0, tail_applications=0,
                           deterministic_pairs=0, relative_phase_gates=0)

    def take(self, name, wires, values=()):
        r.require(self.pos < len(self.items), 'SAVED_STREAM_TRUNCATED')
        item = self.items[self.pos]
        r.require(item['name'] == name and item['wires'] == wires and not item['clbits'],
                  'SAVED_STREAM_OPERATION_WIRE_ORDER@' + str(self.pos))
        controls = 1 if name in self.circuit['custom'] or name in ('CXGate', 'CZGate', 'CRZGate') else 0
        r.require((item['controls'], item['ctrl_state']) == (controls, controls), 'SAVED_STREAM_CONTROL_METADATA')
        r.require(len(item['params']) == len(values), 'SAVED_STREAM_PARAMETER_ARITY')
        for stored, expected in zip(item['params'], values):
            r.require(number(stored).hex() == float(expected).hex(), 'SAVED_STREAM_PARAMETER_BITS@' + str(self.pos))
        self.pos += 1
        return item

    def angle(self, name, wires):
        r.require(self.pos < len(self.items), 'SAVED_DIAGONAL_MISSING')
        item = self.items[self.pos]
        r.require(len(item['params']) == 1, 'SAVED_DIAGONAL_PARAMETER')
        value = number(item['params'][0])
        self.take(name, wires, [value])
        r.require(value != 0.0, 'FIXED_NONZERO_DIAGONAL_PATTERN')
        return value

    def basis(self, operations, inverse=False):
        rows = reversed(operations) if inverse else operations
        for op in rows:
            r.require(op['gate'] in ('p', 'xx_plus_yy'), 'FIXED_GAUSSIAN_GATE')
            values = [float.fromhex(v) for v in op['params']]
            r.require(all(math.isfinite(v) for v in values), 'FINITE_GAUSSIAN_PARAMETER')
            r.require(len(values) == (1 if op['gate'] == 'p' else 2), 'GAUSSIAN_PARAMETER_ARITY')
            if inverse: values[0] = -values[0]
            self.take('PhaseGate' if op['gate'] == 'p' else 'XXPlusYYGate', op['wires'], values)
        self.counts['inverse_basis_groups' if inverse else 'forward_basis_groups'] += 1

    def diagonal(self, operations, fragment):
        self.basis(operations, inverse=True)
        pairs = [(list(pair), self.angle('RZZGate', list(pair)))
                 for pair in combinations(range(4), 2)] if fragment else []
        singles = [([k], self.angle('RZGate', [k])) for k in range(4)]
        self.basis(operations)
        return pairs, singles

    def directional(self, operations, saved):
        self.basis(operations, inverse=True)
        pairs, singles = saved
        for (left, right), angle in pairs:
            self.take('CXGate', [left, right]); self.take('CXGate', [4, right])
            self.take('RZGate', [right], [-angle])
            self.take('CXGate', [4, right]); self.take('CXGate', [left, right])
        for (wire,), angle in singles:
            self.take('CXGate', [4, wire]); self.take('RZGate', [wire], [-angle]); self.take('CXGate', [4, wire])
        # This saved ancilla RZ carries twice this block's original global phase.
        phase = self.angle('RZGate', [4]) / 2.0
        self.basis(operations)
        self.counts['deterministic_pairs'] += 1
        return phase

    def tail(self, events, known):
        active = None
        active_key = None
        phase = 0.0
        # Original SOURCE accumulates event phases BEFORE application phases.
        for event in events:
            p = event['phase']
            r.require(p['imag'] == 0 and p['real'] in (1, -1), 'EXACT_RTE_PHASE')
            phase += math.pi if p['real'] == -1 else 0.0
        for event in events:
            for app in event['application_sequence']:
                self.counts['tail_applications'] += 1
                sign = app['coefficient_sign']
                r.require(type(sign) is int and sign in (-1, 1), 'RTE_COEFFICIENT_SIGN')
                if app['is_identity']:
                    if active is not None:
                        self.basis(active); active = active_key = None
                        self.counts['identity_flushes'] += 1
                    r.require(app['role'] in ('product', 'rotation'), 'RTE_APPLICATION_ROLE')
                    phase += ((math.pi if sign == -1 else 0.0) if app['role'] == 'product'
                              else -(sign * event['rotation_angle']))
                    continue
                operations = [dict(gate=o['name'], params=[float(v).hex() for v in o['parameters']], wires=o['qubits'])
                              for o in app['basis_change_operations']]
                r.require(r.digest(operations) in known, 'COMMON_REGISTERED_GAUSSIAN_FIELDS')
                key = (app['basis_id'], app['basis_hash'])
                if active_key == key:
                    r.require(active == operations, 'REUSED_BASIS_FIELDS_CHANGED')
                    self.counts['reuse_boundaries'] += 1
                else:
                    if active is not None: self.basis(active)
                    self.basis(operations, inverse=True)
                    active, active_key = operations, key
                support = app['diagonal_pauli_support']
                r.require(type(support) is list and len(support) in (1, 2) and
                          len(set(support)) == len(support) and all(type(k) is int and 0 <= k < 4 for k in support),
                          'RTE_SUPPORT')
                if app['role'] == 'product':
                    for k in support: self.take('CZGate', [4, k])
                    phase += math.pi if sign == -1 else 0.0
                else:
                    r.require(app['role'] == 'rotation', 'RTE_APPLICATION_ROLE')
                    angle = 2.0 * (sign * event['rotation_angle'])
                    if len(support) == 1:
                        self.take('CRZGate', [4, *support], [angle])
                    else:
                        # Existing helper verifies custom saved definition/base/body.
                        r.require(self.pos < len(self.items), 'CONTROLLED_RZZ_MISSING')
                        name = self.items[self.pos]['name']
                        r.require(name in self.circuit['custom'], 'CONTROLLED_RZZ_DEFINITION_REQUIRED')
                        self.take(name, [4, *support], [angle])
        if active is not None: self.basis(active)
        if phase != 0.0:
            self.take('PhaseGate', [4], [phase]); self.counts['relative_phase_gates'] += 1


def recognize(circuit, events, receipt, axis, scalar):
    original.wrapper_fields(circuit, axis, scalar)
    first = circuit['instructions'][0]
    r.require(not first['params'] and not first['clbits'] and first['controls'] == first['ctrl_state'] == 0, 'INITIAL_HADAMARD_FIELDS')
    for definition in circuit['custom'].values():
        body = definition['body']; base = definition['base_operation']
        r.require(definition['kind'] == 'c' and definition['nq'] == 3 and definition['nc'] == 0 and
                  body is not None and body['nq'] == 3 and body['nc'] == 0 and not body['custom'] and
                  number(body['phase']) == 0.0 and
                  [(i['controls'], i['ctrl_state']) for i in body['instructions']] == [(2, 3), (1, 1), (2, 3)] and
                  base['controls'] == base['ctrl_state'] == 0, 'CUSTOM_CONTROL_BODY_PHASE_METADATA')
    qpy.event_central_join(circuit, events)
    qpy.event_basis_join(events, receipt)
    r.require(len(receipt['operation_stream']) == 6, 'FIXED_COMMON_SIX_ROTATIONS')
    groups = [x['operations'] for x in receipt['operation_stream']]
    known = {r.digest(g) for g in groups}
    cursor = Cursor(circuit)
    phases = []
    for stream in events:
        first = cursor.diagonal(groups[0], fragment=False)
        second = cursor.diagonal(groups[1], fragment=True)
        cursor.tail(stream, known)
        phi_second = cursor.directional(groups[1], second)
        phi_first = cursor.directional(groups[0], first)
        phases.append((phi_first, phi_second))
    cursor.take('PhaseGate', [4], [-scalar * .8])
    if axis == 'sine': cursor.take('SdgGate', [4])
    cursor.take('HGate', [4])
    # Measurement has a clbit, so recognize it explicitly.
    r.require(cursor.pos == len(cursor.items) - 1, 'FULL_STREAM_EXTRA_OR_MISSING_OPERATION')
    measurement = cursor.items[cursor.pos]
    r.require(measurement['name'] == 'Measure' and measurement['wires'] == [4] and
              measurement['clbits'] == [0] and measurement['params'] == [] and
              measurement['controls'] == measurement['ctrl_state'] == 0, 'FINAL_MEASUREMENT')
    # Compare global-phase consistency using only serialized ancilla angles,
    # not fresh H coefficients. Fixed Qiskit numerical phase normalization.
    phase = 0.0
    for first, second in phases:
        partial = (first % math.tau + second) % math.tau
        phase = (phase + partial) % math.tau
    global_equal = number(circuit['phase']).hex() == phase.hex()
    return dict(status='VERIFIED', conditions=dict(
        gaussian_forward_inverse='VERIFIED', basis_reuse_boundaries='VERIFIED',
        deterministic_diagonal_shape_and_mirror='VERIFIED', event_control_order_wire_angle='VERIFIED',
        relative_event_and_identity_phase='VERIFIED', scalar_once_and_full_wrapper='VERIFIED',
        complete_instruction_coverage='VERIFIED',
        serialized_global_phase_relation='VERIFIED' if global_equal else 'UNRESOLVED',
        absolute_deterministic_diagonal_vs_common_preparation='UNRESOLVED'),
        **cursor.counts, pieces=len(events), serialized_instructions_consumed=len(cursor.items),
        global_phase_relation_exact=global_equal,
        scope='All existing instructions recognized against source grammar and saved fields; absolute diagonal magnitudes vs common preparation remain unproved',
        diagonal_pattern='Fixed H2 nonzero: one-body 4 RZ; fragment 6 RZZ + 4 RZ; omission rejected',
        new_circuit_objects=0, new_lowering=0, fresh_diagonal_or_rotation_computation=0,
        scientific_operator_equivalence=False)


def validate_pair(cosine, sine, cosine_evolution_sha256, sine_evolution_sha256):
    """Sine has one Sdg; compare common evolution/conditions, not wrapper length."""
    r.require(cosine_evolution_sha256 == sine_evolution_sha256, 'PAIRED_EVOLUTION_FIELDS')
    a = {k:v for k,v in cosine.items() if k != 'serialized_instructions_consumed'}
    b = {k:v for k,v in sine.items() if k != 'serialized_instructions_consumed'}
    r.require(a == b and sine['serialized_instructions_consumed'] == cosine['serialized_instructions_consumed'] + 1,
              'PAIRED_STRUCTURE_SCOPE')
