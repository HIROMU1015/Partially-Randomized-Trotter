"""Dormant C-S V0 binding and finite residual adapter. No work on import.

Calling bind_slots/coarse/residual_table on molecular data is a SCIENCE call.
This preparation uses invented schema fixtures only. No circuit/state/shot API.
"""
from fractions import Fraction as F
import math

from .same_position_residual import Angle, RotationSite
from .parent_allocation import rational, event_statistics, error_accounting


def binary_angle(raw):
    """Exact binary64 *parameter*, never decimal-round or derive a real angle."""
    if not isinstance(raw, str):
        raise TypeError('saved canonical float.hex string required')
    value = float.fromhex(raw)
    if not math.isfinite(value) or value.hex() != raw:
        raise ValueError('nonfinite/noncanonical saved angle')
    return Angle(rad=F(value))


def lower_slot(site_id, support, angle, role, *, n):
    """diag(I,R_Zsupport(theta)): Ianc P(theta/2), Zanc P(-theta/2).

    Word order is q[n],q[n-1],...,q[0] (ancilla first). V0 picks factor 0;
    factor 1 and all phase/basis boundaries are retained, never fused away.
    """
    support = tuple(support)
    if (type(n) is not int or n < 1 or len(support) not in (1, 2)
            or len(set(support)) != len(support)
            or any(type(q) is not int or not 0 <= q < n for q in support)
            or role not in ('D_DIAGONAL', 'R_DIAGONAL')):
        raise ValueError('explicit Z/ZZ support and diagonal role required')
    word = ''.join('Z' if q in support else 'I' for q in reversed(range(n)))
    return tuple(RotationSite(site_id+'/factor'+str(i), a+word,
                              angle.scaled(s), role)
                 for i, (a, s) in enumerate((('I', F(1, 2)), ('Z', F(-1, 2)))))


def bind_slots(prep, law, uniform, r_angle, *, n=8, policy):
    """DEFERRED: first canonical D central gate; first positive R rotation.

    Inputs must first be byte-bound to the fixed N1/qualification manifests.
    D comes from SAVED half-step gate keys, not a new block build. R angle is
    supplied by resolve_r_angle below using the original event generator.
    Failure of the first chosen spec does not search for a replacement.
    """
    blocks = prep['deterministic_blocks']
    records = uniform['inventory']['deterministic_half_step_records']
    if (not blocks or len(blocks) != len(records)
            or [b['order_index'] for b in blocks] != list(range(len(blocks)))
            or len({b['block_id'] for b in blocks}) != len(blocks)):
        raise ValueError('canonical D block binding failed')
    selected_d = None
    boundaries = []
    for b, rec in zip(blocks, records):
        if (rec['block_id'] != b['block_id'] or b['num_system_qubits'] != n
                or b['block_kind'] not in ('one_body', 'df_fragment')):
            raise ValueError('D role/block identity mismatch')
        basis = prep['basis_operations'][b['basis_hash']]
        if basis != b['basis_change_operations']:
            raise ValueError('D basis boundary mismatch')
        gates = rec['half_step_gates']; k = len(basis)
        if len(gates) < 2*k:
            raise ValueError('truncated saved block')
        # Check both saved boundaries without constructing any gate/matrix.
        for row, src in zip(gates[len(gates)-k:] if k else (), basis):
            if row['key'] != [src['name'], [float(x).hex() for x in src['parameters']]] or row['qubits'] != src['qubits']:
                raise ValueError('forward basis record mismatch')
        for row, src in zip(gates[:k], reversed(basis)):
            if src['name'] not in ('rz', 'xx_plus_yy'):
                raise ValueError('unsupported inverse basis metadata')
            # describe_basis_change_operations saves str(parameter). Restore
            # the same binary64 values as the existing basis restore codec
            # before negating theta; beta of XXPlusYY is left unchanged.
            params = [float(x) for x in src['parameters']]; params[0] = -params[0]
            if row['key'] != [src['name'], [float(x).hex() for x in params]] or row['qubits'] != src['qubits']:
                raise ValueError('inverse basis record mismatch')
        central = gates[k:len(gates)-k if k else len(gates)]
        for j, row in enumerate(central):
            name, params = row['key']
            if name == 'p':
                if row['qubits'] != [n]:
                    raise ValueError('scalar phase wire mismatch')
                continue  # FIXED controlled scalar, not eligible
            if name not in ('crz', 'crzz') or len(params) != 1 or row['qubits'][0] != n or len(row['qubits']) != (2 if name == 'crz' else 3):
                raise ValueError('unknown D central role; no inference from gate name alone')
            if selected_d is None:
                selected_d = lower_slot('D/forward/'+b['block_id']+'/central'+str(j),
                    row['qubits'][1:], binary_angle(params[0]), 'D_DIAGONAL', n=n)
        boundaries.append(dict(block_id=b['block_id'], basis_hash=b['basis_hash'],
                               fixed_basis_gates_per_side=k, central_records=central))
    if selected_d is None:
        raise ValueError('NO_D_SLOT; no alternative policy')
    positive = [r['label'] for r in law['label_masses'] if rational(r['exact']) > 0]
    ids = [c['component_id'] for c in prep['components']]
    if not positive or len(set(ids)) != len(ids) or len(set(positive)) != len(positive) or not set(positive) <= set(ids):
        raise ValueError('positive law/catalog identity mismatch')
    c = prep['components'][ids.index(positive[0])]
    if c['is_identity'] or c['coefficient_sign'] not in (-1, 1):
        raise ValueError('first R rotation unsupported; no substitute label')
    matches = [row['definition'] for row in policy['definitions']
               if row['source_basis_id'] == c['basis_id']
               and row['source_basis_hash'] == c['basis_hash']
               and row['support'] == c['diagonal_pauli_support']]
    if len(matches) != 1 or matches[0]['num_system_qubits'] != n:
        raise ValueError('unique saved selected R basis required')
    angle = r_angle(c)  # exactly one resolver call; never resample
    selected_r = lower_slot('R/order0/rotation/'+c['component_id'],
                            c['diagonal_pauli_support'], angle, 'R_DIAGONAL', n=n)
    return dict(sites=(selected_d[0], selected_r[0]),
                fixed_control_partners=tuple(RotationSite(p.site_id, p.generator, p.angle, 'FIXED')
                                             for p in (selected_d[1], selected_r[1])),
                fixed_D_boundaries=boundaries, selected_R_component=c['component_id'],
                fixed_selected_R_basis=matches[0], policy_identity=policy['policy_fingerprint'])


def resolve_r_angle(component, distribution_record):
    """DEFERRED original order-0 event arithmetic; no builder/compile/RNG.

    These are the same binary64 operations as _make_event and the central
    builder. Their resulting gate parameter is preserved EXACTLY, not replaced
    by the mathematical atan of the saved rational tau.
    """
    from types import SimpleNamespace as NS
    from .shared_cpu import module
    rte = module('rte')
    c = dict(component)
    c['basis_change_operations'] = ()  # unused by event-angle arithmetic
    c = NS(**c)
    event = rte._make_event([0], [c], NS(**distribution_record), 0)
    return binary_angle((2.0*event.unsigned_rotation_angle).hex())


def coarse(angle):
    """Proposed sole policy: keep pi; nearest 2^-8 rad, ties away from zero.

    No H4 call in preparation. Identical D/R policy, no price search. This
    makes the residual rational and <=2^-9 rad without approximating pi.
    """
    if not isinstance(angle, Angle):
        raise TypeError('exact Angle required')
    a = abs(angle.rad)*256
    k = (a+F(1, 2)).numerator//(a+F(1, 2)).denominator
    return Angle(pi=angle.pi, rad=F(k if angle.rad >= 0 else -k, 256))


def residual_table(residual):
    """DEFERRED finite ordinary Kres2/rres1, two events, no RNG.

    Exact rational residual from coarse(); no independent original-path draw
    exists in V0. A future V1 must specify its conditional joint law separately.
    """
    from .g7_generator import CanonicalGenerator
    if not isinstance(residual, Angle) or residual.pi or abs(residual.rad) > F(1, 512):
        raise ValueError('residual outside fixed rational policy; no r/K search')
    if residual.zero:
        return dict(events=(), statistics=dict(coefficient_mass=F(1), range=F(1),
                    weight_second_moment=F(1)), finite_error_upper=F(0), exact_identity=True)
    x = abs(residual.rad)/2
    gen = CanonicalGenerator((F(1),), x, 3, 'ordinary', root_bits=256, probability_bits=160)
    events = tuple(dict(gen.event(i, (0,)*g.degree, 0),
                        rotation_sign=1 if residual.rad > 0 else -1)
                   for i, g in enumerate(gen.groups))
    coefficient_error = sum((z.hi-z.lo)/2 for z in gen.roots)
    truncation = x**4/(24*(1-x/5))
    return dict(events=events, statistics=event_statistics(events),
                coefficient_error_upper=coefficient_error, truncation_upper=truncation,
                finite_error_upper=coefficient_error+truncation, exact_identity=False)


def local_bounds(table, coarse_error, event_errors):
    """Corrected local mean bound, not a whole-interrogation allowance."""
    c = rational(coarse_error)
    if c < 0:
        raise ValueError('nonnegative coarse error required')
    events = table['events']; errors = tuple(event_errors)
    if table['exact_identity']:
        if events or errors:
            raise ValueError('zero residual has no stochastic branch')
        return dict(operator_mean_error_upper=c, confidence_or_shots_computed=False)
    out = error_accounting(events, coefficient_error_upper=table['finite_error_upper'],
        # No independent angle approximation: guard targets exact atan ratio.
        angle_error_bounds=(F(0),)*len(events), native_error_bounds=errors)
    out['coarse_weighted_error_upper'] = table['statistics']['coefficient_mass']*c
    out['operator_mean_error_upper'] += out['coarse_weighted_error_upper']
    return out


class Calls:
    """Tiny durable attempt caps using caller's existing journal writer.

    A failed attempt remains consumed. A fresh V0 output is required by the
    execution draft; this object is not an authorization/grant mechanism.
    """
    CAPS = dict(binding=1, local_comparison=2, residual_table=2,
                positive_key=8, local_price=2)

    def __init__(self, write):
        self.write = write
        self.used = {}; self.stopped = False

    def invoke(self, kind, identity, fn):
        token = (kind, identity)
        if (self.stopped or kind not in self.CAPS or token in self.used
                or sum(k == kind for k, _ in self.used) >= self.CAPS[kind]):
            self.stopped = True
            self.write(dict(event='STOP', kind=kind, identity=identity,
                            reason='duplicate/unknown/cap-exceeding call'))
            raise RuntimeError('STOP: duplicate/unknown/cap-exceeding call')
        self.used[token] = 'START'
        try:
            self.write(dict(event='CALL_START', kind=kind, identity=identity))
            out = fn()
            self.write(dict(event='CALL_RETURN', kind=kind, identity=identity))
            self.used[token] = 'RETURN'
            return out
        except BaseException as exc:
            self.stopped = True
            self.write(dict(event='STOP', kind=kind, identity=identity,
                            reason=type(exc).__name__+': '+str(exc)))
            raise
