"""F3 exact input, immutable evidence and provenance. No science inputs or RNG.

This is an opt-in wrapper contract, not a physical-input or platform certificate.
Integer evidence uses hex so huge exact values do not need decimal conversion.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path


SPEC_COMMIT = 'dd03eb08b6d768f5578d743ab5ad390bd9c93b07'
PROOF_COMMIT = 'bc649c592c8c2b625b49094b85c27e380faa3cf4'
G7_COMMIT = 'ab2549f41b3546fb3940342a2162dd9ee93699c4'
POLICY = 'G6_F3_FIXED_SUPPLIED_PRECISION_P0_P5_V1'
ARMS = ('full_return', 'ordinary', 'partial_return_tail', 'closed_P3_tail')
OLD_PINS = {
    'return_aggregation.py': 'f716d3c69fc1063e47b7618022662715f6b34a9c1f35cc3e7ed26813f5ac69f0',
    'g7_generator.py': 'bb4177d26f1cee9721c2b083de01f4aa309f3ad39f22597cfae3b662881269fb',
    'g7_provider.py': '17fff6d4d9fa9c9728091f6f01a4e6af1a389e8949c6e4a2ec6e3112f2fe3c62',
}
F3_FILES = ('g6_f3_contract.py', 'g6_f3_precision.py', 'g6_f3_generator.py')


def _pack(value):
    if type(value) is F:
        return {'$rat': [hex(value.numerator), hex(value.denominator)]}
    if type(value) is int:
        return {'$int': hex(value)}
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is tuple:
        return {'$tuple': [_pack(v) for v in value]}
    if type(value) is list:
        return [_pack(v) for v in value]
    if type(value) is dict:
        if all(type(k) is str for k in value):
            return {k: _pack(value[k]) for k in sorted(value)}
        items = [(_pack(k), _pack(v)) for k, v in value.items()]
        items.sort(key=lambda row: json.dumps(row[0], sort_keys=True))
        return {'$map': items}
    raise TypeError('only exact evidence values may be serialized')


def _unpack(value):
    if type(value) is list:
        return [_unpack(v) for v in value]
    if type(value) is dict:
        if set(value) == {'$rat'}:
            n, d = (int(v, 16) for v in value['$rat'])
            if d <= 0:
                raise ValueError('invalid evidence denominator')
            return F(n, d)
        if set(value) == {'$int'}:
            return int(value['$int'], 16)
        if set(value) == {'$tuple'}:
            return tuple(_unpack(v) for v in value['$tuple'])
        if set(value) == {'$map'}:
            return {_unpack(k): _unpack(v) for k, v in value['$map']}
        return {k: _unpack(v) for k, v in value.items()}
    if value is None or type(value) in (str, bool):
        return value
    raise ValueError('nonexact evidence number')


def canonical(value):
    return json.dumps(_pack(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def fingerprint(value):
    return sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Evidence:
    document: str
    digest: str

    @classmethod
    def create(cls, value):
        text = canonical(value)
        return cls(text, sha256(text.encode()).hexdigest())

    def read(self):
        if type(self.document) is not str or sha256(self.document.encode()).hexdigest() != self.digest:
            raise F3Failure('INVALID_INPUT', 'EVIDENCE_DIGEST_MISMATCH', 'evidence.read')
        try:
            return _unpack(json.loads(self.document))
        except (ValueError, TypeError, KeyError) as exc:
            raise F3Failure('INVALID_INPUT', 'EVIDENCE_ENCODING_INVALID', 'evidence.read',
                            details={'exception_type': type(exc).__name__}) from exc

    def to_dict(self):
        self.read()
        return json.loads(self.document)


class F3Failure(Exception):
    """A failure is terminal for its session. It never represents X=0."""
    def __init__(self, category, reason, stage, *, input_id=None, details=None):
        super().__init__(reason)
        self.category, self.reason, self.stage = category, reason, stage
        self.input_id = input_id
        self.evidence = Evidence.create({
            'schema': 'g6_f3_failure_v1', 'category': category, 'reason': reason,
            'stage': stage, 'input_id': input_id, 'details': details or {},
            'mandatory_STOP': True, 'run_complete': False, 'is_random_zero': False,
        })

    def to_dict(self):
        return self.evidence.to_dict()


def invalid(reason, stage, **details):
    return F3Failure('INVALID_INPUT', reason, stage, details=details)


def exact_integer(value, field, *, positive=False):
    if type(value) is not int or (positive and value < 1):
        raise invalid('INVALID_INTEGER', 'entrance.' + field,
                      field=field, received_type=type(value).__name__, positive_required=positive)
    return value


def exact_rational(value, field):
    if type(value) is F:
        return value
    if type(value) is int:
        return F(value)
    if type(value) is dict and set(value) == {'num', 'den'}:
        n = exact_integer(value['num'], field + '.num')
        d = exact_integer(value['den'], field + '.den', positive=True)
        return F(n, d)
    raise invalid('EXACT_RATIONAL_REQUIRED', 'entrance.' + field,
                  field=field, received_type=type(value).__name__)


@dataclass(frozen=True)
class Request:
    p: tuple
    x: F
    m: int
    arm: str
    sigma: int
    eta: F
    rho: F
    root_bits: int
    probability_bits: int
    angle_tau: F
    angle_root_bits: int
    angle_terms: int

    def payload(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def dimensions(self):
        def bits(q):
            return max(1, abs(q.numerator).bit_length()) + q.denominator.bit_length()
        return {'L': len(self.p), 'm_value': self.m, 'm_binary_length': self.m.bit_length(),
                'H_p_integer_pair_total_bits': sum(bits(q) for q in self.p),
                'H_x_integer_pair_bits': bits(self.x),
                'precision_rational_integer_pair_bits': sum(bits(q) for q in
                                                           (self.eta, self.rho, self.angle_tau)),
                'H_value': self.probability_bits, 'K_root_value': self.root_bits,
                'K_phi_value': self.angle_root_bits, 'J_value': self.angle_terms}

    @property
    def input_id(self):
        return fingerprint({'p': self.p, 'x': self.x, 'm': self.m,
                            'arm': self.arm, 'sigma': self.sigma})

    @property
    def policy_id(self):
        return fingerprint({'policy': POLICY, 'spec_commit': SPEC_COMMIT,
                            **{k: v for k, v in self.payload().items()
                               if k not in ('p', 'x', 'm', 'arm', 'sigma')}})


def make_request(p, x, m, *, arm='full_return', sigma=1, eta, rho,
                 root_bits, probability_bits, angle_tau, angle_root_bits, angle_terms):
    if type(p) not in (tuple, list) or not p:
        raise invalid('FINITE_EXPLICIT_P_TABLE_REQUIRED', 'entrance.p')
    probabilities = tuple(exact_rational(v, 'p') for v in p)
    if any(v <= 0 for v in probabilities):
        raise invalid('REMOVE_ZERO_LABELS_AND_REQUIRE_POSITIVE_P', 'entrance.p')
    if sum(probabilities, F(0)) != 1:
        raise invalid('P_NOT_EXACTLY_NORMALIZED', 'entrance.p')
    x = exact_rational(x, 'x')
    if not 0 <= x <= 1:
        raise invalid('X_OUTSIDE_SHORT_STEP_DOMAIN', 'entrance.x')
    m = exact_integer(m, 'm', positive=True)
    if m % 2 != 1:
        raise invalid('ODD_DEGREE_REQUIRED', 'entrance.m')
    if type(arm) is not str or arm not in ARMS:
        raise invalid('UNKNOWN_ARM', 'entrance.arm')
    if arm != 'full_return' and m < 3:
        raise invalid('CANONICAL_REQUIRES_ODD_DEGREE_AT_LEAST_THREE', 'entrance.arm')
    sigma = exact_integer(sigma, 'sigma')
    if sigma not in (-1, 1):
        raise invalid('SIGNED_TIME_REQUIRED', 'entrance.sigma')
    eta, rho = exact_rational(eta, 'eta'), exact_rational(rho, 'rho')
    if not 0 < eta < 1 or rho <= 0:
        raise invalid('INVALID_ETA_RHO_DOMAIN', 'entrance.precision')
    tau = exact_rational(angle_tau, 'angle_tau')
    if tau <= 0:
        raise invalid('POSITIVE_ANGLE_TOLERANCE_REQUIRED', 'entrance.angle_tau')
    return Request(probabilities, x, m, arm, sigma, eta, rho,
                   exact_integer(root_bits, 'root_bits', positive=True),
                   exact_integer(probability_bits, 'probability_bits', positive=True), tau,
                   exact_integer(angle_root_bits, 'angle_root_bits', positive=True),
                   exact_integer(angle_terms, 'angle_terms', positive=True))


def validate_word(word, request):
    if type(word) not in (list, tuple):
        raise invalid('EXPLICIT_PARENT_WORD_REQUIRED', 'helper.word')
    word = tuple(word)
    if any(type(i) is not int or not 0 <= i < len(request.p) for i in word):
        raise invalid('INVALID_LABEL', 'helper.word')
    if len(word) % 2 or len(word) > request.m - 1:
        raise invalid('VALID_EVEN_PARENT_LENGTH_REQUIRED', 'helper.word')
    if any(a == b for a, b in zip(word, word[1:])):
        raise invalid('REDUCED_PARENT_REQUIRED', 'helper.word')
    return word


def source_identity():
    directory = Path(__file__).resolve().parent
    old = {name: sha256((directory / name).read_bytes()).hexdigest() for name in OLD_PINS}
    if old != OLD_PINS:
        raise invalid('FROZEN_SOURCE_IDENTITY_MISMATCH', 'source_identity', actual=old,
                      expected=OLD_PINS)
    new = {name: sha256((directory / name).read_bytes()).hexdigest() for name in F3_FILES}
    return Evidence.create({'schema': 'g6_f3_source_identity_v1', 'old_SHA256': old,
                            'F3_SHA256': new, 'spec_commit': SPEC_COMMIT,
                            'proof_commit': PROOF_COMMIT, 'G7_commit': G7_COMMIT,
                            'query_model': 'conditional exact controlled-Q',
                            'physical_provider_certified': False})
