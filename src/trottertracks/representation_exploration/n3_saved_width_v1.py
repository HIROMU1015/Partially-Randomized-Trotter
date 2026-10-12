"""Exact fixed-witness width algebra; no numerical proposals or eigensolvers."""
from fractions import Fraction as F
from math import isqrt

BITS = 128
RATIONAL_CAP = 4096
INTEGER_CAP = 8192


class Arithmetic:
    def __init__(self):
        self.max_rational_bits = self.max_integer_bits = self.operations = 0

    def q(self, value):
        if isinstance(value, (float, bool)):
            raise ValueError('exact rational required')
        value = F(value)
        bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        self.max_rational_bits = max(self.max_rational_bits, bits)
        self.operations += 1
        if bits > RATIONAL_CAP:
            raise ValueError('RATIONAL_BIT_CAP')
        return value

    def integer(self, value):
        bits = abs(value).bit_length()
        self.max_integer_bits = max(self.max_integer_bits, bits)
        if bits > INTEGER_CAP:
            raise ValueError('INTEGER_BIT_CAP')
        return value

    def add(self, a, b): return self.q(self.q(a) + self.q(b))
    def sub(self, a, b): return self.q(self.q(a) - self.q(b))
    def mul(self, a, b): return self.q(self.q(a) * self.q(b))
    def div(self, a, b): return self.q(self.q(a) / self.q(b))


def encode(value): return [value.numerator, value.denominator]


def decode(value):
    if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value) or value[1] <= 0:
        raise ValueError('rational pair')
    if max(abs(value[0]).bit_length(), value[1].bit_length()) > RATIONAL_CAP:
        raise ValueError('RATIONAL_BIT_CAP')
    return F(*value)


def interval(g, kappa, q, arithmetic=None):
    """Enclose infimum delta>0 with k<=g+delta and delta(g+delta)>=q."""
    a = arithmetic or Arithmetic()
    g, kappa, q = map(a.q, (g, kappa, q))
    if g <= 0 or kappa < 0:
        raise ValueError('g>0 and kappa>=0 required')
    box = max(F(0), a.sub(kappa, g))
    if q <= 0 or a.mul(box, a.add(g, box)) >= q:
        low = high = box
        root = {'kind': 'box_or_zero', 'value': encode(box)}
    else:
        disc = a.add(a.mul(g, g), a.mul(4, q))
        numerator, denominator = disc.numerator, disc.denominator
        nr, dr = isqrt(numerator), isqrt(denominator)
        if a.integer(nr * nr) == numerator and a.integer(dr * dr) == denominator:
            low = high = a.div(a.sub(a.q(F(nr, dr)), g), 2)
            root = {'kind': 'exact_rational_root', 'discriminant': encode(disc)}
        else:
            scaled = a.integer(numerator << (2 * BITS))
            scaled_floor = scaled // denominator
            n = isqrt(scaled_floor)
            lower_sqrt, upper_sqrt = a.q(F(n, 1 << BITS)), a.q(F(n + 1, 1 << BITS))
            low = max(box, F(0), a.div(a.sub(lower_sqrt, g), 2))
            high = max(box, a.div(a.sub(upper_sqrt, g), 2))
            root = {'kind': 'outward_sqrt_interval', 'discriminant': encode(disc),
                    'sqrt_low': encode(lower_sqrt), 'sqrt_high': encode(upper_sqrt), 'bits': BITS}
    proof_delta = high if high > 0 else a.q(F(1, 1 << BITS))
    if kappa > a.add(g, proof_delta) or a.mul(proof_delta, a.add(g, proof_delta)) < q:
        raise AssertionError('unsafe rational proof endpoint')
    return {'low': encode(low), 'high': encode(high), 'positive_proof_delta': encode(proof_delta),
            'infimum_zero': high == 0, 'root_certificate': root}


def check_interval(g, kappa, q, record):
    """Independent endpoint test, using polynomial inequalities, not isqrt."""
    g, kappa, q = F(g), F(kappa), F(q)
    lo, hi = decode(record['low']), decode(record['high'])
    proof = decode(record['positive_proof_delta'])
    if g <= 0 or kappa < 0 or lo < 0 or hi < lo or proof <= 0:
        raise ValueError('interval domain')
    bound = max(F(0), kappa - g)
    if not (lo <= bound or lo * (g + lo) <= q):
        raise ValueError('lower endpoint is not a bound')
    if hi < bound or hi * (g + hi) < q:
        raise ValueError('upper endpoint is unsafe')
    if proof < hi or kappa > g + proof or proof * (g + proof) < q:
        raise ValueError('positive proof endpoint')
    if record['infimum_zero'] != (hi == 0):
        raise ValueError('zero infimum flag')
    if hi - lo > F(1, 1 << (BITS + 1)):
        raise ValueError('outward enclosure precision')
    return True


def minimum(records):
    if not records:
        return None
    return {'low': encode(min(decode(r['width']['low']) for r in records)),
            'high': encode(min(decode(r['width']['high']) for r in records)),
            'upper_witness_id': min(records, key=lambda r: (decode(r['width']['high']), r['id']))['id'],
            'positive_proof_delta': min(records, key=lambda r: (decode(r['width']['high']), r['id']))['width']['positive_proof_delta'],
            'candidate_count': len(records)}


def compare(occ, base):
    if occ is None or base is None:
        return 'MISSING_SAVED_COVERAGE'
    ol, oh, bl, bh = (decode(x) for x in (occ['low'], occ['high'], base['low'], base['high']))
    if oh < bl: return 'OCC_STRICTLY_NARROWER_IN_SAVED_SET'
    if bh < ol: return 'BASE_STRICTLY_NARROWER_IN_SAVED_SET'
    if ol == oh == bl == bh: return 'EXACT_EQUAL_WIDTH'
    return 'OVERLAPPING_INTERVALS_UNDETERMINED'
