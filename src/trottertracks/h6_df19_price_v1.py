"""Source-derived RZ accounting, with no QuantumCircuit or compiler calls.

The candidate is an explicit native expansion plus literal inverse-basis and
wire-local adjacent-RZ cancellation. It is NOT a qualified P0 price oracle.
Nothing here accepts measured prices, fitted constants, or a sampling seed.
"""
from collections import Counter
import math

from .track_d_native_resources import NativeCounter, NativeInstruction

MODEL = 'H6_DF19_M5_SOURCE_NATIVE_BASIS_CANCEL_ZFOLD_V1'
NQ = 13


class Accounting:
    def __init__(self):
        self.native = NativeCounter(NQ, 1)
        self.folded = NativeCounter(NQ, 1)
        self.pending_z = [None] * NQ
        self.pending_basis = None
        self.features = Counter()
        self.native_rz_by_role = Counter()
        self.global_phase = 0.0

    def _flush_z(self, q):
        angle = self.pending_z[q]
        if angle is not None:
            if angle != 0.0:
                self.folded.feed(NativeInstruction('rz', (q,), params=(angle,)))
            self.pending_z[q] = None

    def _native(self, name, wires, angle=None, role='other'):
        params = () if angle is None else (float(angle),)
        op = NativeInstruction(name, tuple(wires), (0,) if name == 'measure' else (), params)
        self.native.feed(op)
        if name == 'rz':
            self.native_rz_by_role[role] += 1
            if angle == 0.0:
                self.features['literal_zero_rz'] += 1
            if abs(angle) in (math.pi / 2, math.pi, 2 * math.pi):
                self.features['literal_pi_family_rz'] += 1
            q = wires[0]
            previous = self.pending_z[q]
            self.pending_z[q] = float(angle) if previous is None else previous + float(angle)
        else:
            for q in wires:
                self._flush_z(q)
            self.folded.feed(op)

    def _gate(self, name, wires, params=(), role='other'):
        """A fixed analytic basis expansion, not a call to basis translation.

        P(t)=exp(it/2)RZ(t), H=exp(i*pi/4)RZ(pi/2)SX RZ(pi/2),
        SXdg=X SX, and RY(t)=SX RZ(-t) SXdg (matrix product).
        XX+YY uses the literal Qiskit 1.3.0 standard-gate definition.
        """
        q = wires[0]
        if name in ('rz', 'sx', 'x', 'cx', 'measure'):
            self._native(name, wires, params[0] if params else None, role)
        elif name in ('p', 's', 'sdg'):
            t = params[0] if name == 'p' else (math.pi / 2 if name == 's' else -math.pi / 2)
            self.global_phase += t / 2
            self._native('rz', wires, t, role)
        elif name == 'sxdg':
            self._native('x', wires, role=role)
            self._native('sx', wires, role=role)
        elif name == 'h':
            self.global_phase += math.pi / 4
            self._native('rz', wires, math.pi / 2, role)
            self._native('sx', wires, role=role)
            self._native('rz', wires, math.pi / 2, role)
        elif name == 'ry':
            self._gate('sxdg', wires, role=role)
            self._native('rz', wires, -params[0], role)
            self._native('sx', wires, role=role)
        elif name == 'xx_plus_yy':
            a, b = wires
            theta, beta = params
            for gate, qs, ps in [
                ('rz', (a,), (beta,)), ('rz', (b,), (-math.pi/2,)),
                ('sx', (b,), ()), ('rz', (b,), (math.pi/2,)),
                ('s', (a,), ()), ('cx', (b,a), ()),
                ('ry', (b,), (-theta/2,)), ('ry', (a,), (-theta/2,)),
                ('cx', (b,a), ()), ('sdg', (a,), ()),
                ('rz', (b,), (-math.pi/2,)), ('sxdg', (b,), ()),
                ('rz', (b,), (math.pi/2,)), ('rz', (a,), (-beta,)),
            ]:
                self._gate(gate, qs, ps, role)
        elif name == 'rzz':
            a, b = wires
            self._native('cx', (a,b), role=role)
            self._native('rz', (b,), params[0], role)
            self._native('cx', (a,b), role=role)
        elif name == 'crz':
            a, b = wires
            self._native('rz', (b,), params[0]/2, role)
            self._native('cx', (a,b), role=role)
            self._native('rz', (b,), -params[0]/2, role)
            self._native('cx', (a,b), role=role)
        elif name == 'crzz':
            a, b, c = wires
            self._native('cx', (b,c), role=role)
            self._gate('crz', (a,c), params, role)
            self._native('cx', (b,c), role=role)
        elif name == 'cz':
            a, b = wires
            self._gate('h', (b,), role=role)
            self._native('cx', (a,b), role=role)
            self._gate('h', (b,), role=role)
        else:
            raise ValueError('unsupported primitive: ' + name)

    def flush_basis(self):
        if self.pending_basis is None:
            return
        ops, inverse, role = self.pending_basis
        self.pending_basis = None
        for op in reversed(ops) if inverse else ops:
            params = tuple(float(t) for t in op['parameters'])
            if inverse:
                if op['name'] == 'p':
                    params = (-params[0],)
                elif op['name'] == 'xx_plus_yy':
                    params = (-params[0], params[1])
                else:
                    raise ValueError('unsupported inverse basis primitive')
            self._gate(op['name'], tuple(op['qubits']), params, role)
        self.features['basis_transforms_emitted'] += 1

    def basis(self, operations, inverse, role):
        if self.pending_basis is not None:
            old_ops, old_inverse, _ = self.pending_basis
            if old_inverse != inverse and old_ops == operations:
                self.pending_basis = None
                self.features['adjacent_inverse_basis_pairs_cancelled'] += 1
                return
            self.flush_basis()
        self.pending_basis = (operations, inverse, role)

    def gate(self, name, wires, params=(), role='other'):
        self.flush_basis()
        self._gate(name, tuple(wires), params, role)

    def diagonal(self, primitives, directional, role):
        for a, b, angle in primitives.rzz:
            if angle == 0:
                continue
            if directional:
                for name, qs, ps in [('cx',(a,b),()),('cx',(12,b),()),
                                      ('rz',(b,),(-angle,)),('cx',(12,b),()),('cx',(a,b),())]:
                    self.gate(name, qs, ps, role)
            else:
                self.gate('rzz', (a,b), (angle,), role)
        for q, angle in primitives.rz:
            if directional:
                self.gate('cx', (12,q), role=role)
                self.gate('rz', (q,), (-angle,), role)
                self.gate('cx', (12,q), role=role)
            else:
                self.gate('rz', (q,), (angle,), role)
        if directional and primitives.global_phase != 0:
            self.gate('rz', (12,), (2*primitives.global_phase,), role)
        elif not directional:
            self.global_phase += primitives.global_phase

    def block(self, block, directional):
        self.basis(block['basis'], True, 'head_basis')
        self.diagonal(block['diagonal'], directional, 'head_directional' if directional else 'head_uncontrolled')
        self.basis(block['basis'], False, 'head_basis')

    def tail(self, events):
        active = None
        phase = 0.0
        for event in events:
            phase += math.atan2(event['phase']['imag'], event['phase']['real'])
            apps = event['application_sequence']
            if not apps or apps[-1]['role'] != 'rotation':
                raise ValueError('saved application order')
            unsigned = apps[-1]['coefficient_sign'] * event['rotation_angle']
            self.features['event_order_' + str(event['taylor_order'])] += 1
            for i, app in enumerate(apps):
                if app['application_index'] != i:
                    raise ValueError('saved application index')
                if app['is_identity']:
                    if active is not None:
                        self.basis(active[1], False, 'tail_basis')
                        active = None
                    phase += (math.pi if app['coefficient_sign'] == -1 else 0.) if app['role'] == 'product' else -unsigned
                    self.features['identity_applications'] += 1
                    continue
                key = (app['basis_id'], app['basis_hash'])
                if active is None or active[0] != key:
                    if active is not None:
                        self.basis(active[1], False, 'tail_basis')
                    self.basis(app['basis_change_operations'], True, 'tail_basis')
                    active = (key, app['basis_change_operations'])
                    self.features['tail_basis_groups'] += 1
                else:
                    self.features['tail_shared_basis_pairs'] += 1
                support = tuple(app['diagonal_pauli_support'])
                if len(support) not in (1,2):
                    raise ValueError('saved support')
                if app['role'] == 'product':
                    for q in support:
                        self.gate('cz', (12,q), role='tail_product')
                    phase += math.pi if app['coefficient_sign'] == -1 else 0.
                elif app['role'] == 'rotation':
                    self.gate('crz' if len(support) == 1 else 'crzz', (12,*support), (2*unsigned,), 'tail_rotation')
                else:
                    raise ValueError('saved application role')
        if active is not None:
            self.basis(active[1], False, 'tail_basis')
        if phase != 0:
            self.gate('p', (12,), (phase,), 'tail_relative_phase')

    def finish(self):
        self.flush_basis()
        for q in range(NQ):
            self._flush_z(q)
        native = self.native.result()['metrics']['rz_count']
        folded = self.folded.result()['metrics']['rz_count']
        return dict(model=MODEL, structural_rz=folded,
                    expanded_rz_after_basis_cancellation=native,
                    rz_removed_by_literal_zfold=native-folded,
                    native_rz_by_role=dict(self.native_rz_by_role), features=dict(self.features),
                    expansion_global_phase=self.global_phase,
                    qualification='STRUCTURAL_NATIVE_PROXY_P0_UNQUALIFIED',
                    compiler_correction=None, law_expectation=None, confidence_interval=None)



# Gate-expansion/Z-fold kernel above reused from pinned H4 V1, adapted only to
# 12 system wires + control. H6 piece/head/tail rules below follow its provider.
def predict(piece_blocks, q, scalar, T, axis, event_batches=None, allocation=None):
    if (q,T) not in ((1,.8),(1,1.6),(2,.8)) or axis not in ('cosine','sine'):
        raise ValueError('Fixed saved H6 V1 scope only')
    if len(piece_blocks) != 11:
        raise ValueError('Eleven ordered signed pieces required')
    if event_batches is not None:
        if allocation not in ([3,3,3,3,2,5,2,3,2,3,3], [1,2,1,2,1,2,1,2,1,2,1]):
            raise ValueError('Saved R32/R16 allocation required')
        if len(event_batches) != 11*q or any(len(batch) != allocation[i%11] for i,batch in enumerate(event_batches)):
            raise ValueError('R_outer allocated once per outer, not per piece')
    a=Accounting();a.gate('h',(12,),role='wrapper')
    for outer in range(q):
        for j,blocks in enumerate(piece_blocks):
            for block in blocks:a.block(block,False)
            if event_batches is not None:a.tail(event_batches[11*outer+j])
            for block in reversed(blocks):a.block(block,True)
    if scalar*T != 0:a.gate('p',(12,),(-scalar*T,),'scalar_once')
    if axis=='sine':a.gate('sdg',(12,),role='wrapper')
    a.gate('h',(12,),role='wrapper');a.gate('measure',(12,),role='measurement')
    a.features.update(signed_piece_occurrences=11*q,negative_piece_occurrences=q,
                      piece_boundaries=11*q-1,outer_step_boundaries=q-1)
    if event_batches is not None:a.features['head_tail_interfaces']=22*q
    return a.finish()
