"""W2c cost-audit helpers. Standard library only; no science/backend imports.

No CLI, RNG, timer, sampler, compiler, synthesis, authority issuer or file writer.
The supplied generator/clock callbacks are exercised only by artificial fixtures
in preparation. A draft never authorizes a real operation.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re

ARMS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return')
COORDINATES = ('T', 'CX', 'one_Q', 'serial_depth', 'ancilla_peak')
PRICE_KEY_FIELDS = ('operator_id', 'label', 'control_role', 'direction',
                    'compiler_id', 'gate_set_id', 'precision_id', 'phase_policy',
                    'optimization_id')


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def integer(value, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), 'invalid integer')
    return value


def rat(value):
    require(type(value) is Fraction, 'exact Fraction required')
    return value


def _pack(value):
    if type(value) is Fraction:
        return {'rat': [hex(value.numerator), hex(value.denominator)]}
    if type(value) is int:
        return {'int': hex(value)}
    if value is None or type(value) in (str, bool):
        return value
    if type(value) in (tuple, list):
        return [_pack(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _pack(v) for k, v in sorted(value.items())}
    raise InvalidEvidence('nonexact or unsupported evidence type')


def canonical(value):
    return json.dumps(_pack(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def fingerprint(value):
    return sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Seal:
    text: str
    digest: str

    @classmethod
    def create(cls, value):
        text = canonical(value)
        return cls(text, sha256(text.encode()).hexdigest())

    def verify(self):
        require(type(self.text) is str and sha256(self.text.encode()).hexdigest() == self.digest,
                'evidence hash mismatch')
        return self.digest


def check_source_files(root, pins):
    """Read exact named files only, no source imports; rejects path traversal."""
    base = Path(root).resolve()
    require(type(pins) is dict and pins, 'nonempty source pins required')
    for name, expected in pins.items():
        path = (base / name).resolve()
        require(path.is_relative_to(base) and path.is_file(), 'source outside checkout or absent')
        require(sha256(path.read_bytes()).hexdigest() == expected, 'source bytes changed')
    return fingerprint(pins)


def law_witness(law, intervals, bits, eta):
    """Check the frozen dyadic_distribution postconditions, never generate a law."""
    integer(bits, True);rat(eta)
    require(0 < eta < 1 and len(law) == len(intervals) and len(law) > 0, 'law shape')
    require(sum(map(rat, law)) == 1, 'law not normalized')
    lows, highs = [], []
    for q, bounds in zip(law, intervals):
        lo, hi = map(rat, bounds)
        require(0 < lo <= hi and q > 0 and (q * (1 << bits)).denominator == 1,
                'support or dyadic grid')
        require((1-eta)*hi <= q <= (1+eta)*lo, 'relative law precision')
        lows.append(lo);highs.append(hi)
    require(sum(lows) <= 1 <= sum(highs), 'interval normalization')
    return Seal.create({'law': law, 'intervals': intervals, 'bits': bits, 'eta': eta})


def root_witness(square, bounds, bits, rho):
    """Validate the same outward grid and relative width as frozen canonical."""
    square, rho = rat(square), rat(rho);integer(bits, True)
    lo, hi = map(rat, bounds);step = Fraction(1, 1 << bits)
    require(square > 0 and rho > 0 and 0 < lo <= hi, 'root domain')
    require((lo/step).denominator == (hi/step).denominator == 1, 'root grid')
    require(lo*lo <= square <= hi*hi, 'root not enclosing')
    require((lo == hi and lo*lo == square) or
            (hi-lo == step and lo*lo < square < hi*hi), 'root not frozen tight grid')
    require(hi-lo <= 2*rho*lo, 'root relative precision')
    return Seal.create({'square': square, 'bounds': bounds, 'bits': bits, 'rho': rho})


def canonical_state(generator):
    """Persists actual held state only: no extra conditional-law constructions."""
    g = generator
    require(g.arm in ARMS[:-1], 'canonical arm required')
    return {'arm': g.arm, 'p': tuple(g.p), 'x': g.x, 'm': g.m, 'H': g.H, 'K': g.K,
            'eta': g.eta, 'rho': g.rho, 'chi': g.chi,
            'groups': tuple({'mode': z.mode, 'degree': z.degree, 'square': z.square,
                             'ratio': z.ratio, 'first': z.first} for z in g.groups),
            'roots': tuple((z.lo, z.hi) for z in g.roots),
            'B': (g.B.lo, g.B.hi), 'labels': tuple(g.label_law),
            'group_law': tuple(g.group_law),
            'root_child': tuple(getattr(g, 'root_child', ()))}


def canonical_snapshot(generator, law_builder=None):
    """Duck-typed adapter; no generator construction or science import here.

    Captures finite conditional laws at preflight. These extra queries are setup
    audit cost, not free cached production replacements. Source must be pinned by
    the caller; the snapshot alone does not independently prove group formulas.
    """
    g = generator
    require(g.arm in ARMS[:-1], 'canonical arm required')
    p = tuple(g.p)
    require(p and all(type(q) is Fraction and q > 0 for q in p) and sum(p) == 1,
            'canonical p')
    require(type(g.x) is Fraction and 0 < g.x <= 1 and type(g.m) is int and
            g.m >= 5 and g.m % 2 == 1, 'canonical input domain')
    groups = tuple({'mode': z.mode, 'degree': z.degree, 'square': z.square,
                    'ratio': z.ratio, 'first': z.first} for z in g.groups)
    require(groups, 'empty groups')
    roots = tuple((z.lo, z.hi) for z in g.roots)
    require(len(roots) == len(groups), 'group/root mismatch')
    for z, bounds in zip(groups, roots):
        root_witness(z['square'], bounds, g.K, g.rho)
    blo, bhi = sum(z[0] for z in roots), sum(z[1] for z in roots)
    require((g.B.lo, g.B.hi) == (blo, bhi), 'B envelope mismatch')
    law_witness(g.label_law, tuple((q, q) for q in p), g.H, g.eta)
    law_witness(g.group_law, tuple((lo/bhi, hi/blo) for lo, hi in roots), g.H, g.eta)
    # Snapshot every conditional type that this frozen arm can actually use.
    extra = []
    if any(z['mode'] == 'distinct' for z in groups):
        exact, law = g.first_distribution()
        law_witness(law, tuple((q, q) for q in exact), g.H, g.eta)
        extra.append(('first', tuple(exact), tuple(law)))
    excluded = (range(len(p)) if any(z['mode'] == 'distinct' for z in groups)
                else sorted({z['first'] for z in groups if z['mode'] == 'fixed_first'}))
    for i in excluded:
        ids, exact, law = g.conditional(i)
        require(tuple(ids) == tuple(j for j in range(len(p)) if j != i), 'conditional support')
        require(tuple(exact) == tuple(p[j]/(1-p[i]) for j in ids), 'conditional exact values')
        law_witness(law, tuple((q, q) for q in exact), g.H, g.eta)
        extra.append((i, tuple(ids), tuple(exact), tuple(law)))
    child = tuple(g.root_child) if any(z['mode'] == 'root_collected' for z in groups) else ()
    require(not child or (len(child) == len(p) and sum(child) == 1 and min(child) > 0),
            'root child support')
    if child:
        require(callable(law_builder), 'root child law builder required')
        child_law = tuple(law_builder(child, g.H, g.eta))
        law_witness(child_law, tuple((q, q) for q in child), g.H, g.eta)
        extra.append(('root_child', child, child_law))
    require(g.chi == sum(q*q for q in p), 'chi mismatch')
    return canonical_state(g) | {'conditional_witnesses': tuple(extra)}


def event_witness(event, *, full, degree_cap, labels):
    """Shared event invariants; canonical raw reduction keeps original phase."""
    raw, word = tuple(event['raw_word']), tuple(event['word'])
    require(len(raw) % 2 == 0 and len(raw) <= degree_cap-1, 'raw degree')
    require(all(type(i) is int and 0 <= i < labels for i in raw), 'raw labels')
    reduced = []
    for i in raw:
        if reduced and reduced[-1] == i: reduced.pop()
        else: reduced.append(i)
    require(word == tuple(reduced) and (not full or word == raw), 'word remapping')
    child = event['child'];require(type(child) is int and 0 <= child < labels, 'child')
    require(not full or not word or child != word[0], 'Full child support')
    require(event['phase_i_power'] == len(raw) % 4 and
            tuple(event['circuit_time_word']) == tuple(reversed(word)), 'phase or time word')
    for k in ('proposal', 'coefficient', 'weight', 'ratio'):
        require(rat(event[k]) > 0, 'event positivity')
    require(event['proposal'] <= 1 and type(event['phase_i_power']) is int, 'proposal or phase type')
    require(event['proposal']*event['weight'] == event['coefficient'], 'pi weight identity')
    calls = Counter(word);calls[child] += 2
    require(dict(calls) == event['provider_calls'], 'provider call metadata')
    return Seal.create(event | {'provider_calls': {str(k): v for k, v in calls.items()}})


class AuditParity:
    """Same provenance/state/evidence checks on all four arms.

    This is an audit overlay, not a new sampler or a claim of equal overhead.
    Full keeps its original F3 checks; canonical uses its pinned constructor and
    canonical_snapshot. Callbacks may not discard the old guards.
    """
    def __init__(self, arm, identity, snapshot, source_check):
        require(arm in ARMS, 'unknown arm')
        self.arm, self.identity = arm, Seal.create(identity)
        self.snapshot, self.source_check = snapshot, source_check
        self.source_id = source_check()
        self.state_id = fingerprint(snapshot())
        self.failure = None
        self.completed = 0

    def invoke(self, operation):
        require(self.failure is None, 'audit session terminal STOP')
        try:
            self.identity.verify()
            require(self.source_check() == self.source_id, 'source identity drift')
            require(fingerprint(self.snapshot()) == self.state_id, 'law state drift')
            value = operation()
            record = Seal.create({'arm': self.arm, 'identity': self.identity.digest,
                                  'state_id': self.state_id, 'payload': value,
                                  'completed_calls_before': self.completed,
                                  'confidence_certified': False})
            record.verify();self.completed += 1
            return record
        except Exception as exc:
            self.failure = Seal.create({'error': type(exc).__name__, 'message': str(exc),
                                        'completed_calls': self.completed,
                                        'is_random_zero': False, 'mandatory_STOP': True})
            raise


class StageLedger:
    """Exclusive stage CPU/wall records with injected clocks, no timer/RNG here."""
    def __init__(self, clock, sink):
        self.clock, self.sink = clock, sink
        self.records, self.active, self.stopped = [], None, False

    def begin(self, stage):
        require(not self.stopped and self.active is None and type(stage) is str and stage,
                'nested/terminal/unnamed stage')
        self.active = (stage, tuple(self.clock()))

    def end(self, outcome, *, counters=None):
        require(self.active is not None and outcome in ('OK', 'ZERO_R', 'ZERO_C', 'EVENT', 'FAIL'),
                'invalid stage completion')
        stage, before = self.active;after = tuple(self.clock())
        if not (len(before) == len(after) == 2 and all(type(x) is int for x in before+after)
                and all(y >= x for x, y in zip(before, after))):
            self.stopped = True
            self.records.append({'stage': stage, 'outcome': 'FAIL', 'started_clocks': before,
                                 'received_clocks': after, 'reason': 'invalid nanosecond clocks'})
            raise InvalidEvidence('invalid nanosecond clocks')
        row = {'stage': stage, 'cpu_ns': after[0]-before[0], 'wall_ns': after[1]-before[1],
               'outcome': outcome, 'counters': counters or {}, 'is_scientific_confidence': False}
        self.records.append(row);self.active = None
        if outcome == 'FAIL':self.stopped = True
        try:self.sink(Seal.create(row))
        except Exception:
            self.stopped = True
            raise
        return row


def price_key(request):
    require(type(request) is dict and set(request) == set(PRICE_KEY_FIELDS), 'price key fields')
    require(all(type(v) is str and v for v in request.values()), 'price key values')
    require(request['direction'] in ('forward', 'actual_adjoint') and
            request['control_role'] in ('outer', 'helper') and
            request['phase_policy'] == 'STRICT_ABSOLUTE_AND_CONTROLLED_RELATIVE', 'price semantics')
    return fingerprint(request)


def price_row(row, expected_request, *, error_allowance):
    """Validate a supplied price/certificate; never acquire it or assume missing=0."""
    key = price_key(expected_request)
    require(row['request'] == expected_request and row['key'] == key, 'price identity')
    require(row['status'] in ('KNOWN', 'UNKNOWN'), 'price status')
    require(set(row['coordinates']) == set(COORDINATES), 'price coordinates')
    for bounds in row['coordinates'].values():
        if bounds is None:
            continue
        require(type(bounds) is tuple and len(bounds) == 2, 'price interval')
        integer(bounds[0]);integer(bounds[1]);require(bounds[0] <= bounds[1], 'price interval order')
    if row['status'] == 'UNKNOWN':
        require(all(b is None for b in row['coordinates'].values()), 'unknown price must remain null')
        return 'UNRESOLVED'
    require(all(type(row[k]) is str and re.fullmatch('[0-9a-f]{64}', row[k])
                for k in ('artifact_sha256', 'certificate_sha256')) and row['phase_retained'] is True,
            'price artifact or phase evidence missing')
    require(0 <= rat(row['strict_error_upper']) <= rat(error_allowance), 'provider error exceeds budget')
    if expected_request['direction'] == 'actual_adjoint':
        require(type(row.get('forward_artifact_sha256')) is str and
                re.fullmatch('[0-9a-f]{64}', row['forward_artifact_sha256']) and
                row.get('actual_adjoint_verified') is True,
                'actual adjoint evidence absent')
    return 'PARTIAL' if any(b is None for b in row['coordinates'].values()) else 'COMPLETE'


def decode_price_record(record):
    """External JSON shape to exact internal types; does not validate a circuit."""
    require(type(record) is dict and type(record.get('coordinates')) is dict, 'price record object')
    row = dict(record)
    row['coordinates'] = {}
    for key, bounds in record['coordinates'].items():
        require(bounds is None or (type(bounds) is list and len(bounds) == 2 and
                                  all(type(v) is int for v in bounds)), 'JSON price bounds')
        row['coordinates'][key] = None if bounds is None else tuple(bounds)
    if record.get('strict_error_upper') is not None:
        value = record['strict_error_upper']
        require(type(value) is str and re.fullmatch(r'[0-9]+(?:/[1-9][0-9]*)?', value),
                'JSON exact nonnegative rational required')
        row['strict_error_upper'] = Fraction(value)
    return row


def rz_cache_witness(row, *, key, epsilon):
    """Metadata guard for an existing strict RZ row; no matrix/error reevaluation."""
    require(row['key'] == key and row['epsilon'] == str(rat(epsilon)), 'RZ key/precision')
    sequence = row['sequence']
    require(type(sequence) is str and all(g in 'HTtSXW' for g in sequence), 'RZ token alphabet')
    require(sha256(sequence.encode()).hexdigest() == row['sequence_sha256'], 'RZ hash')
    require(row['T_count'] == sequence.count('T')+sequence.count('t') and
            row['Tdagger_count'] == sequence.count('t') and
            row['global_W_count'] == sequence.count('W') and
            row['one_qubit_count'] == len(sequence)-sequence.count('W'), 'RZ saved counts')
    require(row['error_pass'] is True and
            0 <= Fraction(row['strict_operator_error_upper']) <= epsilon, 'RZ saved bound')
    return {'existing_saved_certificate_only': True, 'W_scalar_retained': True,
            'independent_numerical_error_check_performed': False}


def execution_blockers(candidate):
    """No execution or authority creation. Enumerate unfilled adoption boundaries."""
    require(candidate['schema'] == 'g6_w2c_b3_c1_candidate_v1', 'candidate schema')
    require(candidate['arms'] == list(ARMS), 'candidate comparator set')
    require(candidate['contract_grants_execution'] is False, 'candidate cannot grant execution')
    blockers = []
    if candidate['adopted'] is not True:blockers.append('CONTRACT_NOT_ADOPTED')
    for name, value in candidate['decisions_required'].items():
        if value is None:blockers.append(name)
    if candidate.get('separate_execution_approval') is not True:
        blockers.append('SEPARATE_EXECUTION_APPROVAL_REQUIRED')
    return tuple(blockers)


def reject_real_execution(candidate):
    """Preparation release has no real runner, even if metadata is edited."""
    blockers = execution_blockers(candidate)
    raise PermissionError('PREPARATION_ONLY_NO_REAL_RUNNER: ' + ','.join(blockers))
