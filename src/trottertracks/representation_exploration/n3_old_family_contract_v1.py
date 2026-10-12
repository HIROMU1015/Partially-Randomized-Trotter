"""Exact input schema and fixed resource contract; no scientific proposals."""
from fractions import Fraction as F
import hashlib
import json

CAPS = {'rational_bits': 4096, 'arithmetic_operations': 2_000_000,
        'numeric_eigh_calls': 64, 'primal_PSD_probes': 96,
        'ray_probes': 80, 'mixture_probes': 1024}
GRID = 16
GOLDEN_STEPS = 60
DYADIC_BITS = 40
INSTANCE_KEYS = {'id', 'h_A', 'v', 'nu', 'U', 'c', 'delta'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def encode(value):
    if isinstance(value, F): return [value.numerator, value.denominator]
    if isinstance(value, list): return [encode(v) for v in value]
    if isinstance(value, dict): return {k: encode(v) for k, v in value.items()}
    return value


def rational(raw):
    if (not isinstance(raw, list) or len(raw) != 2
            or any(type(v) is not int for v in raw) or raw[1] <= 0):
        raise ValueError('RATIONAL_SCHEMA')
    if max(abs(raw[0]).bit_length(), raw[1].bit_length()) > CAPS['rational_bits']:
        raise ArithmeticError('RATIONAL_BIT_CAP')
    return F(*raw)


def matrix(raw):
    if not isinstance(raw, list) or not raw or any(not isinstance(r, list) or len(r) != len(raw) for r in raw):
        raise ValueError('MATRIX_SCHEMA')
    result = [[rational(x) for x in row] for row in raw]
    if any(result[i][j] != result[j][i] for i in range(len(result)) for j in range(len(result))):
        raise ValueError('REAL_SYMMETRY_REQUIRED')
    return result


def decode(instance):
    if not isinstance(instance, dict) or set(instance) != INSTANCE_KEYS:
        raise ValueError('INSTANCE_SCHEMA_OR_ORACLE_FIELD')
    A = matrix(instance['h_A']); n = len(A); nu = instance['nu']
    if not 1 <= n <= 9 or type(nu) is not int or not 1 <= nu <= n:
        raise ValueError('DIMENSION_OR_PARTICLE_SCOPE')
    if not isinstance(instance['id'], str) or not instance['id']:
        raise ValueError('INSTANCE_ID')
    if not isinstance(instance['v'], list) or len(instance['v']) != n:
        raise ValueError('COUPLING_DIMENSION')
    v = [rational(x) for x in instance['v']]
    U, c, delta = [rational(instance[k]) for k in ('U', 'c', 'delta')]
    if c <= U or delta <= 0: raise ValueError('COMMON_BOUND_OR_TARGET_DOMAIN')
    return A, v, nu, U, c, delta


class Arithmetic:
    """Explicit rational operands/results; Fraction internals are not monitored."""
    def __init__(self):
        self.operations = self.max_bits = 0

    def q(self, x):
        if isinstance(x, (float, bool)): raise ValueError('EXACT_RATIONAL_REQUIRED')
        x = F(x); self.operations += 1
        self.max_bits = max(self.max_bits, abs(x.numerator).bit_length(), x.denominator.bit_length())
        if self.max_bits > CAPS['rational_bits']: raise ArithmeticError('RATIONAL_BIT_CAP')
        if self.operations > CAPS['arithmetic_operations']: raise ArithmeticError('ARITHMETIC_OPERATION_CAP')
        return x

    def add(self, x, y): return self.q(self.q(x) + self.q(y))
    def sub(self, x, y): return self.q(self.q(x) - self.q(y))
    def mul(self, x, y): return self.q(self.q(x) * self.q(y))
    def div(self, x, y): return self.q(self.q(x) / self.q(y))
    def total(self, values):
        s = F(0)
        for value in values: s = self.add(s, value)
        return s
