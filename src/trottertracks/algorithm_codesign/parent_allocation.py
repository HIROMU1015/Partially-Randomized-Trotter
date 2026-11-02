"""Generic parent allocation and root-only adapters; no scientific execution.

Only rational/interval metadata and gate descriptions are constructed here.
There is no solver, synthesis, matrix evaluator, shot forecast, or authorization.
Reference enumeration is explicit and never part of the sampling path.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from time import perf_counter
import hashlib
import json

from .return_aggregation import Interval, root_interval, dyadic_distribution, dyadic_index, reduce_word
from .g9_p5 import P5Closed
from .g7_generator import CanonicalGenerator


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (F, int, str)):
        raise TypeError('exact Fraction/int/string required; no float conversion')
    return F(value)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def enclosed_root(square, bits, rho):
    bound = root_interval(square, bits)
    if bound.lo <= 0 or bound.hi-bound.lo > 2*rho*bound.lo:
        raise ValueError('fixed square-root resolution insufficient; no adaptive rescue')
    return bound


def normalized_enclosures(values):
    total = Interval(sum(v.lo for v in values), sum(v.hi for v in values))
    if not values or total.lo <= 0:
        raise ValueError('positive total enclosure required')
    return tuple(Interval(v.lo/total.hi, v.hi/total.lo) for v in values)


def digital_proposal(coefficients, *, mode='canonical', costs=None,
                     regularization=None, root_bits=256, probability_bits=160,
                     eta=F(1, 10**12), rho=F(1, 10**12)):
    """Proposal for a supplied finite table, with no cost acquisition or choice.

    The cost-aware optimum concerns cost x second moment, not finite-confidence
    optimality. Zero-cost analytic IS is rejected; no hidden positive floor.
    Regularization is caller supplied and is not a charged execution cost.
    """
    c = tuple(map(rational, coefficients))
    integer(root_bits, 'root bits', 1); integer(probability_bits, 'probability bits', 1)
    eta, rho = rational(eta), rational(rho)
    if not c or min(c) <= 0 or not 0 < eta < 1 or not 0 < rho < 1:
        raise ValueError('positive coefficients and valid tolerances required')
    if mode == 'canonical':
        if costs is not None or regularization is not None:
            raise ValueError('canonical proposal does not consume cost or regularizer')
        scores = tuple(Interval(v, v) for v in c)
    elif mode in ('cost_aware', 'regularized'):
        if costs is None:
            raise ValueError('cost acquisition is external and required')
        costs = tuple(map(rational, costs))
        if len(costs) != len(c) or min(costs) < 0:
            raise ValueError('one nonnegative cost per positive atom required')
        if mode == 'cost_aware':
            if regularization is not None or min(costs) == 0:
                raise ValueError('analytic IS requires positive costs; zero-cost limit is not a law')
            lam = F(0)
        else:
            if regularization is None or rational(regularization) <= 0:
                raise ValueError('regularizer must be explicitly chosen and positive')
            lam = rational(regularization)
        roots = tuple(enclosed_root(v+lam, root_bits, rho) for v in costs)
        scores = tuple(Interval(ci/rt.hi, ci/rt.lo) for ci, rt in zip(c, roots))
    else:
        raise ValueError('unknown proposal mode')
    return dyadic_distribution(normalized_enclosures(scores), probability_bits, eta)


@dataclass(frozen=True)
class Atom:
    child: object
    kind: str
    ratio: object
    coefficient_interval: Interval
    phase_i_power: int
    rotation_sign: int

    @property
    def coefficient(self):
        return self.coefficient_interval.midpoint


class ParentAllocation:
    """One even parent, arbitrary nonnegative rational allocation.

    Zero-b children with positive allocations are merged into one pure parent.
    The common-angle interior uses the old factorized square-root enclosure,
    preserving its digital coefficient/law rather than re-rounding each child.
    """
    def __init__(self, a, b, allocation=None, *, word=(), labels=None, sigma=1,
                 root_bits=256, probability_bits=160, eta=F(1, 10**12),
                 rho=F(1, 10**12), proposal_mode='canonical', costs=None,
                 regularization=None):
        start = perf_counter()
        self.a, self.b = rational(a), tuple(map(rational, b))
        if not self.b or self.a < 0 or min(self.b) < 0:
            raise ValueError('nonnegative parent and nonempty child coefficients required')
        if type(sigma) is not int or sigma not in (-1, 1):
            raise ValueError('time sign must be +1 or -1')
        self.sigma, self.word = sigma, tuple(word)
        for label in self.word:
            integer(label, 'parent label')
        if len(self.word) % 2 or reduce_word(self.word) != self.word:
            raise ValueError('even reduced parent required')
        self.labels = tuple(range(len(self.b))) if labels is None else tuple(labels)
        for label in self.labels:
            integer(label, 'child label')
        if len(self.labels) != len(self.b) or len(set(self.labels)) != len(self.labels):
            raise ValueError('unique labels matching child coefficients required')
        self.K, self.H = integer(root_bits, 'root bits', 1), integer(probability_bits, 'probability bits', 1)
        self.eta, self.rho = rational(eta), rational(rho)
        if not 0 < self.eta < 1 or not 0 < self.rho < 1:
            raise ValueError('tolerances must be strictly between zero and one')
        self.s = sum(self.b)
        if allocation is None:
            allocation = tuple(self.a*bi/self.s for bi in self.b) if self.s else (self.a,)+(F(0),)*(len(self.b)-1)
        self.allocation = tuple(map(rational, allocation))
        if len(self.allocation) != len(self.b) or min(self.allocation) < 0 or sum(self.allocation) != self.a:
            raise ValueError('allocation must be nonnegative with exact sum a')
        self.is_common_angle = self.s > 0 and all(zi*self.s == self.a*bi for zi, bi in zip(self.allocation, self.b))
        phase = len(self.word) % 4
        atoms = []
        if self.is_common_angle and self.a > 0:
            d = enclosed_root(self.a*self.a+self.s*self.s, self.K, self.rho)
            for label, bi in zip(self.labels, self.b):
                if bi:
                    frac = bi/self.s
                    atoms.append(Atom(label, 'rotation', self.s/self.a,
                                      Interval(d.lo*frac, d.hi*frac), phase, sigma))
        else:
            pure_parent = F(0)
            for label, zi, bi in zip(self.labels, self.allocation, self.b):
                if not bi:
                    pure_parent += zi
                elif not zi:
                    atoms.append(Atom(label, 'pure_child', None, Interval(bi, bi),
                                      (phase+(3 if sigma == 1 else 1)) % 4, 0))
                else:
                    ci = enclosed_root(zi*zi+bi*bi, self.K, self.rho)
                    atoms.append(Atom(label, 'rotation', bi/zi, ci, phase, sigma))
            if pure_parent:
                atoms.append(Atom(None, 'pure_parent', None,
                                  Interval(pure_parent, pure_parent), phase, 0))
        self.atoms = tuple(atoms)
        self.mass = Interval(sum(v.coefficient_interval.lo for v in atoms),
                             sum(v.coefficient_interval.hi for v in atoms))
        self.proposal_mode, self.regularization = proposal_mode, regularization
        if atoms:
            self.child_law = digital_proposal([v.coefficient for v in atoms], mode=proposal_mode,
                costs=costs, regularization=regularization, root_bits=self.K,
                probability_bits=self.H, eta=self.eta, rho=self.rho)
        else:
            if proposal_mode != 'canonical' or costs is not None or regularization is not None:
                raise ValueError('zero contribution has no cost-aware law')
            self.child_law = ()
        self.setup_seconds = perf_counter()-start

    def event(self, index, *, group_proposal=F(1)):
        integer(index, 'atom index')
        if index >= len(self.atoms):
            raise ValueError('atom outside positive support')
        group_proposal = rational(group_proposal)
        if not 0 < group_proposal <= 1:
            raise ValueError('group proposal must be in (0,1]')
        atom = self.atoms[index]
        q = group_proposal*self.child_law[index]
        calls = Counter(self.word)
        if atom.child is not None:
            calls[atom.child] += 2 if atom.kind == 'rotation' else 1
        return {'word': self.word, 'raw_word': self.word,
                'circuit_time_word': tuple(reversed(self.word)), 'child': atom.child,
                'event_kind': atom.kind, 'ratio': atom.ratio,
                'rotation_sign': atom.rotation_sign, 'phase_i_power': atom.phase_i_power,
                'coefficient': atom.coefficient, 'coefficient_interval': atom.coefficient_interval,
                'proposal': q, 'weight': atom.coefficient/q,
                'provider_calls': dict(calls), 'provider_calls_are_abstract': True,
                'workspace': 'PROVIDER_DEPENDENT_NOT_ACQUIRED'}

    def sample(self, bits, *, group_proposal=F(1)):
        if not self.child_law:
            raise ValueError('zero contribution cannot be sampled')
        index = dyadic_index(self.child_law, self.H, bits.bits(self.H))
        return self.event(index, group_proposal=group_proposal)

    @property
    def coefficient_error_upper(self):
        return (self.mass.hi-self.mass.lo)/2


class RootAllocationGenerator:
    """Composition over frozen ordinary/P5 constructors, not in-place mutation."""
    def __init__(self, base, allocation=None, *, sigma=1,
                 proposal_mode='canonical', root_costs=None, regularization=None):
        if not isinstance(base, (P5Closed, CanonicalGenerator)) or base.arm not in ('ordinary', 'closed_P5_full'):
            raise ValueError('only ordinary or closed P5 base supported')
        self.base = base
        self.p, self.x, self.m, self.arm = base.p, base.x, base.m, base.arm
        self.K, self.H, self.eta, self.rho = base.K, base.H, base.eta, base.rho
        a, b = (base.a0, base.child0) if isinstance(base, P5Closed) else (F(1), tuple(base.x*pi for pi in base.p))
        self.root = ParentAllocation(a, b, allocation, sigma=sigma, root_bits=self.K,
            probability_bits=self.H, eta=self.eta, rho=self.rho,
            proposal_mode=proposal_mode, costs=root_costs, regularization=regularization)
        self.roots = (self.root.mass,)+tuple(base.roots[1:])
        self.B = Interval(sum(v.lo for v in self.roots), sum(v.hi for v in self.roots))
        self.group_law = dyadic_distribution(normalized_enclosures(self.roots), self.H, self.eta)
        self.legacy_exact = self.root.is_common_angle and proposal_mode == 'canonical' and sigma == 1
        if self.root.is_common_angle:
            if self.root.mass != base.roots[0] or self.B != base.B or self.group_law != base.group_law:
                raise ArithmeticError('factorized common-angle backward compatibility failed')

    @classmethod
    def closed_p5(cls, p, x, allocation=None, *, sigma=1,
                  proposal_mode='canonical', root_costs=None, regularization=None, **precision):
        return cls(P5Closed(tuple(map(rational, p)), rational(x), **precision), allocation, sigma=sigma,
                   proposal_mode=proposal_mode, root_costs=root_costs, regularization=regularization)

    @classmethod
    def ordinary(cls, p, x, allocation=None, *, m=5, sigma=1,
                 proposal_mode='canonical', root_costs=None, regularization=None, **precision):
        return cls(CanonicalGenerator(tuple(map(rational, p)), rational(x), m, 'ordinary', **precision), allocation, sigma=sigma,
                   proposal_mode=proposal_mode, root_costs=root_costs, regularization=regularization)

    def root_event(self, index):
        integer(index, 'root atom index')
        if index >= len(self.root.atoms):
            raise ValueError('root atom outside support')
        if self.legacy_exact:
            return self.base.event(0, (), self.root.atoms[index].child)
        return self.root.event(index, group_proposal=self.group_law[0])

    def event(self, group_index, word, child):
        integer(group_index, 'group index')
        if group_index >= len(self.roots):
            raise ValueError('group outside support')
        if group_index == 0:
            if tuple(word):
                raise ValueError('root has empty parent')
            index = next((i for i, a in enumerate(self.root.atoms) if a.child == child), None)
            if index is None:
                raise ValueError('root child outside support')
            return self.root_event(index)
        old = self.base.event(group_index, word, child)
        if self.legacy_exact:
            return old
        result = dict(old)
        result['proposal'] = old['proposal']*self.group_law[group_index]/self.base.group_law[group_index]
        result['weight'] = result['coefficient']/result['proposal']
        result['event_kind'], result['rotation_sign'] = 'rotation', self.root.sigma
        return result

    def sample(self, bits):
        if self.legacy_exact:
            return self.base.sample(bits)
        draw = lambda law: dyadic_index(law, self.H, bits.bits(self.H))
        index = draw(self.group_law)
        if index == 0:
            return self.root.sample(bits, group_proposal=self.group_law[0])
        if isinstance(self.base, CanonicalGenerator):
            degree = self.base.groups[index].degree
            word = tuple(draw(self.base.label_law) for _ in range(degree))
            child = draw(self.base.label_law)
        else:
            mode, j, k, _ = self.base.groups[index]
            if mode == 'two':
                word = (j, k)
            else:
                word = (j,)
                for remaining in (2, 1, 0):
                    prev = word[-1]
                    ids = tuple(i for i in range(len(self.p)) if i != prev)
                    values = tuple(self.p[i]*self.base.completion(i, remaining)
                                   /self.base.completion(prev, remaining+1) for i in ids)
                    word += (ids[draw(self.base.law(values))],)
            _, s, b = self.base.parent_coefficients(word)
            ids = tuple(i for i, v in enumerate(b) if v)
            child = ids[draw(self.base.law(tuple(b[i]/s for i in ids)))]
        return self.event(index, word, child)

    def reference_events(self, max_events=10000):
        """Bounded reference-only support, never called by sampling."""
        integer(max_events, 'reference event cap', 1)
        count = 0
        for index in range(len(self.roots)):
            if index == 0:
                records = (self.root_event(i) for i in range(len(self.root.atoms)))
            elif isinstance(self.base, CanonicalGenerator):
                degree = self.base.groups[index].degree
                records = (self.event(index, w, child) for w in product(range(len(self.p)), repeat=degree)
                           for child in range(len(self.p)))
            else:
                mode, j, k, _ = self.base.groups[index]
                words = [(j, k)] if mode == 'two' else (
                    (j,)+tail for tail in product(range(len(self.p)), repeat=3)
                    if all(a != b for a, b in zip((j,)+tail, tail)))
                records = (self.event(index, w, child) for w in words
                           for child, b in enumerate(self.base.parent_coefficients(w)[2]) if b)
            for record in records:
                count += 1
                if count > max_events:
                    raise ValueError('reference-only cap exceeded')
                yield record

    def certificate(self):
        return {'normalizer_interval': self.B, 'normalizer_midpoint': self.B.midpoint,
                'coefficient_error_upper': (self.B.hi-self.B.lo)/2,
                'group_count': len(self.roots), 'acceptance': F(1),
                'legacy_exact': self.legacy_exact, 'root_setup_seconds': self.root.setup_seconds,
                'online_sampling_cost': 'NOT_MEASURED', 'native_and_angle_error': 'EXTERNAL_CERTIFICATES_REQUIRED',
                'new_angle_acquisition_cost': 'NOT_ACQUIRED', 'shot_forecast': 'NOT_IMPLEMENTED'}


def reweight_events(events, *, mode='canonical', costs=None, regularization=None, **precision):
    """Explicit table path only. Does not claim cheap global-IS generation."""
    events = tuple(events)
    law = digital_proposal([e['coefficient'] for e in events], mode=mode,
                           costs=costs, regularization=regularization, **precision)
    return tuple(dict(e, proposal=q, weight=rational(e['coefficient'])/q)
                 for e, q in zip(events, law))


def event_statistics(events, *, complete=True):
    events = tuple(events)
    mass = F(0); moment = F(0); bound = F(0); norm = F(0)
    for e in events:
        c, q, w = (rational(e[k]) for k in ('coefficient', 'proposal', 'weight'))
        if c <= 0 or q <= 0 or q > 1 or q*w != c:
            raise ValueError('invalid positive event or corrected weight')
        mass += q; norm += c; moment += q*w*w; bound = max(bound, abs(w))
    if mass > 1 or (complete and mass != 1):
        raise ValueError('proposal does not match declared complete/partial law')
    return {'acceptance': mass, 'coefficient_mass': norm,
            'weight_second_moment': moment, 'range': bound,
            'forecast': 'NOT_COMPUTED'}


def error_accounting(events, *, coefficient_error_upper, angle_error_bounds, native_error_bounds):
    """External certified per-event error inputs; never assumes missing = zero."""
    events = tuple(events)
    ce = rational(coefficient_error_upper)
    angle, native = tuple(map(rational, angle_error_bounds)), tuple(map(rational, native_error_bounds))
    if ce < 0 or len(angle) != len(events) or len(native) != len(events) or any(v < 0 for v in angle+native):
        raise ValueError('complete nonnegative certified error coverage required')
    c = tuple(rational(e['coefficient']) for e in events)
    if any(v <= 0 for v in c):
        raise ValueError('positive digital coefficients required')
    ae = sum(ci*ei for ci, ei in zip(c, angle)); ne = sum(ci*ei for ci, ei in zip(c, native))
    return {'coefficient_error_upper': ce, 'weighted_angle_error_upper': ae,
            'weighted_native_error_upper': ne, 'operator_mean_error_upper': ce+ae+ne,
            'confidence_or_shots_computed': False}


def native_plan(event, controlled_q, controlled_rotation, *, control):
    """Provider callbacks, preserving controlled phase and operator order.

    Descriptions only: callbacks must be separately validated for their provider.
    Phase gates act on ancilla-1, not as a discarded system-global phase.
    """
    integer(control, 'control')
    phase = event['phase_i_power']
    if type(phase) is not int or phase not in range(4):
        raise ValueError('phase exponent must be 0..3')
    kind = event.get('event_kind', 'rotation')
    if kind not in ('rotation', 'pure_child', 'pure_parent'):
        raise ValueError('unknown event kind')
    if (kind == 'pure_child') != bool(phase % 2):
        raise ValueError('phase parity does not match even-parent event kind')
    if len(tuple(event['word'])) % 2:
        raise ValueError('even parent required by this adapter')
    out = [] if phase == 0 else [({1: 'S', 2: 'Z', 3: 's'}[phase], control)]
    for child in reversed(tuple(event['word'])):
        integer(child, 'parent label')
        out += list(controlled_q(child))
    if kind == 'rotation':
        integer(event['child'], 'rotation child')
        ratio = rational(event['ratio']); sign = event.get('rotation_sign', 1)
        if ratio <= 0 or type(sign) is not int or sign not in (-1, 1):
            raise ValueError('positive finite tangent and signed rotation required')
        out += list(controlled_rotation(event['child'], ratio, sign))
    elif kind == 'pure_child':
        integer(event['child'], 'endpoint child')
        if event['ratio'] is not None:
            raise ValueError('pure endpoint cannot have a finite tangent surrogate')
        out += list(controlled_q(event['child']))
    elif event.get('ratio') is not None:
        raise ValueError('pure parent cannot have a tangent surrogate')
    return out


def adjoint_plan(gates, inverse_gate):
    """Invert the entire sequence; do not commute a rotation through its word."""
    return [inverse_gate(g) for g in reversed(tuple(gates))]


def native_ir_g9(event, helper=False):
    """Connection to the frozen three-qubit provider; no compile/synthesis.

    Calling this on a registered fixture requires separate execution permission.
    Tests use mock providers, not the registered Q matrices or circuits.
    """
    if type(helper) is not bool:
        raise ValueError('helper choice must be explicit boolean')
    labels = tuple(event['word']) + (() if event.get('event_kind') == 'pure_parent' else (event['child'],))
    if any(type(i) is not int or i not in range(3) for i in labels):
        raise ValueError('frozen G9 provider supports only labels 0,1,2')
    from . import g9_native as old
    def rotation(child, ratio, sign):
        if helper:
            return [('H', 4)]+old.CQ(child, 4)+[
                ('H', 4), ('R', 4, str(ratio), sign), ('CX', 3, 4),
                ('R', 4, str(ratio), -sign), ('CX', 3, 4), ('H', 4)
            ]+old.CQ(child, 4)+[('H', 4)]
        v = old.V(child)
        axis = ''.join('Z' if i == child else 'I' for i in range(3))
        return v+old.crot(axis, ratio, sign)+old.adjoint(v)
    gates = native_plan(event, old.CQ, rotation, control=3)
    lowered = []
    for g in gates:
        lowered += [('H', g[2]), ('CX', g[1], g[2]), ('H', g[2])] if g[0] == 'CZ' else [g]
    return old.simplify(lowered)


def rotation_key(event, *, precision, tool_identity):
    """Positive RZ primitive key. Signed inverse must use actual adjoint.

    Metadata only; radians = atan(ratio), not the full controlled 2*atan angle.
    Exact endpoints do not request synthesis keys. No tool is selected here.
    """
    kind = event.get('event_kind', 'rotation')
    if kind in ('pure_child', 'pure_parent'):
        if event.get('ratio') is not None:
            raise ValueError('endpoint cannot request a surrogate synthesis key')
        return None
    if kind != 'rotation':
        raise ValueError('unknown event kind')
    eps, ratio = rational(precision), rational(event['ratio'])
    required = ('name', 'version', 'source_sha256', 'seed')
    if eps <= 0 or ratio <= 0 or any(k not in tool_identity for k in required):
        raise ValueError('positive precision/ratio and explicit tool identity required')
    digest = tool_identity['source_sha256']
    if (not all(isinstance(tool_identity[k], str) and tool_identity[k] for k in ('name', 'version'))
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or isinstance(tool_identity['seed'], bool)
            or not isinstance(tool_identity['seed'], (str, int))):
        raise ValueError('tool identity fields are incomplete or malformed')
    spec = {'operation': 'RZ', 'angle_kind': 'atan', 'ratio': str(ratio), 'scale': '1',
            'angle_units': 'radians', 'precision': str(eps), 'up_to_phase': False,
            'signed_inverse': 'actual_sequence_adjoint', 'tool': dict(tool_identity)}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'key': key, 'spec': spec}


def check_cache_metadata(entry, expected):
    """Check normalized saved identities; does NOT rerun an operator guard.

    A future worker must produce/check the strict guard independently. Existing
    cache formats require a provenance-preserving bridge to this metadata shape.
    """
    if expected is None or entry.get('key') != expected['key'] or entry.get('spec') != expected['spec']:
        raise ValueError('angle/precision/tool/control identity mismatch')
    computed = hashlib.sha256(json.dumps(expected['spec'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if computed != expected['key'] or expected['spec'].get('up_to_phase') is not False:
        raise ValueError('key/spec binding or strict-phase policy was changed')
    sequence = entry.get('sequence')
    # The empty product is a valid identity approximation for a tiny angle;
    # its eligibility still requires the saved strict error and sequence hash.
    if not isinstance(sequence, str):
        raise ValueError('saved raw sequence string required')
    if hashlib.sha256(sequence.encode()).hexdigest() != entry.get('sequence_sha256'):
        raise ValueError('saved sequence bytes changed')
    if entry.get('up_to_phase') is not False or entry.get('complete') is not True:
        raise ValueError('complete strict-phase guard receipt required')
    error = rational(entry['operator_error_upper'])
    if error < 0 or error > F(expected['spec']['precision']):
        raise ValueError('saved operator error exceeds requested precision')
    return {'metadata_identity': 'PASS', 'operator_guard_rerun': False,
            'independent_operator_guard_still_required_for_new_acquisition': True}
