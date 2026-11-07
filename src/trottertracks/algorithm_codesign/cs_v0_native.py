"""C-S exact-angle/strict native adapter; no acquisition at import.

Only the angle boundary is new. Uses existing interval gate matrices,
cache identity check, actual adjoint, and pygridsynth endpoint. Not the SP
projective guard or its channel PAI. Real angle evaluation requires science
permission; preparation tests supply invented angles and mock acquisitions.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import time

import mpmath as mp

from .same_position_residual import Angle
from .parent_allocation import rational, check_cache_metadata, adjoint_plan
from .parent_allocation_native import native_accounting
from .rte_reallocation.numeric import enclosure, ivf
from .synthesis_placement.economics import sequence_matrix, exact_sequence


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@contextmanager
def precision(dps):
    if type(dps) is not int or dps < 1:
        raise ValueError('explicit interval precision required')
    old = mp.mp.dps, mp.iv.dps
    try:
        mp.mp.dps = mp.iv.dps = dps
        yield
    finally:
        mp.mp.dps, mp.iv.dps = old


@dataclass(frozen=True)
class Expression:
    kind: str
    a: F = F(0)
    b: F = F(0)

    def __post_init__(self):
        object.__setattr__(self, 'a', rational(self.a))
        object.__setattr__(self, 'b', rational(self.b))
        if self.kind not in ('linear', 'atan') or (self.kind == 'atan' and self.a <= 0):
            raise ValueError('only exact pi+radian or scaled positive atan supported')

    @classmethod
    def linear(cls, angle):
        if not isinstance(angle, Angle):
            raise TypeError('exact Angle required')
        return cls('linear', angle.pi, angle.rad)

    def value(self, ctx):
        a = ctx.mpf(self.a.numerator)/self.a.denominator
        b = ctx.mpf(self.b.numerator)/self.b.denominator
        return a*ctx.pi+b if self.kind == 'linear' else b*ctx.atan2(a, ctx.mpf(1))

    def record(self):
        return dict(kind=self.kind, a=str(self.a), b=str(self.b))


def positive(expr, *, dps):
    if not isinstance(expr, Expression):
        raise TypeError('unsupported angle class')
    if (expr.kind == 'linear' and expr.a == expr.b == 0) or (expr.kind == 'atan' and expr.b == 0):
        return Expression('linear'), 1
    with precision(dps):
        lo, hi = enclosure(expr.value(mp.iv))
    if lo <= 0 <= hi:
        raise ArithmeticError('angle sign enclosure unresolved; no fallback')
    sign = 1 if lo > 0 else -1
    return (Expression(expr.kind, expr.a*sign, expr.b*sign) if expr.kind == 'linear'
            else Expression('atan', expr.a, expr.b*sign)), sign


def request(expr, epsilon, tool):
    """Expanded positive key, including actual precision/runtime/tool options."""
    required = ('name', 'version', 'source_sha256', 'seed', 'tool_identity_sha256',
                'runtime_executable_sha256', 'package_lock_sha256', 'python',
                'synthesizer_options', 'interval_dps')
    if any(k not in tool for k in required) or tool['name'] != 'pygridsynth' or tool['version'] != '2.0.0':
        raise ValueError('expanded fixed pygridsynth 2.0.0 identity required')
    for k in ('source_sha256', 'tool_identity_sha256', 'runtime_executable_sha256', 'package_lock_sha256'):
        if not isinstance(tool[k], str) or len(tool[k]) != 64 or any(c not in '0123456789abcdef' for c in tool[k]):
            raise ValueError('malformed tool digest')
    opts = tool['synthesizer_options']
    if opts.get('up_to_phase') is not False or opts.get('seed') != tool['seed']:
        raise ValueError('strict phase and consistent seed required')
    eps = rational(epsilon)
    if eps <= 0:
        raise ValueError('positive synthesis precision required')
    expr, sign = positive(expr, dps=tool['interval_dps'])
    spec = dict(operation='RZ', angle=expr.record(), precision=str(eps),
                up_to_phase=False, signed_inverse='actual_sequence_adjoint', tool=tool)
    # Copy nested metadata; mutable caller options cannot alter a saved request.
    spec = json.loads(json.dumps(spec))
    return dict(key=digest(spec), spec=spec), sign


def strict_guard(sequence, expr, *, dps):
    """Same outward Frobenius guard as R1, extended ONLY at angle_value."""
    with precision(dps):
        u = sequence_matrix(sequence); theta = expr.value(mp.iv)
        v = [[mp.iv.exp(-mp.iv.j*theta/2), 0], [0, mp.iv.exp(mp.iv.j*theta/2)]]
        error = mp.iv.sqrt(sum((abs(u[i][j]-v[i][j])**2
                              for i in range(2) for j in range(2)), mp.iv.mpf(0)))
        return enclosure(error)[1]


def exact_strict(expr):
    """Correct old projective fast path only for pi/2 multiples.

    RZ(k*pi/2)=W^(-k) S^k. Strict RZ(pi/4) is NOT bare T: its missing
    exp(-i*pi/8) scalar is not an exact Clifford+T scalar. Use synthesis.
    No angle/key periodicization; the explicit scalar is retained.
    """
    if expr.kind != 'linear' or expr.b or (2*expr.a).denominator != 1:
        return None
    k = int(2*expr.a)
    body = exact_sequence(dict(unit='pi', numerator=expr.a.numerator, denominator=expr.a.denominator))
    return 'W'*((-k) % 8)+body


def backend(expr, epsilon, options):
    """Deferred real acquisition: same pygridsynth endpoint/epsilon headroom."""
    from pygridsynth.config import GridsynthConfig
    from pygridsynth.gridsynth import gridsynth_gates
    return gridsynth_gates(expr.value(mp.mp),
        (mp.mpf(epsilon.numerator)/epsilon.denominator)/4, cfg=GridsynthConfig(**options))


def acquire(expected, *, synth=backend):
    """One positive-key attempt. Real angles/tool calls need separate permission.

    Caller reserves CALL_START in a fresh output BEFORE entry. Failed guard is
    STOP, not a second acquisition. All newly acquired rows are independently
    strict-guarded against the exact expression, including point-roundoff.
    """
    spec = expected['spec']; expr = Expression(**spec['angle'])
    rebuilt, sign = request(expr, spec['precision'], spec['tool'])
    if rebuilt != expected or sign != 1:
        raise ValueError('noncanonical acquisition key')
    start = time.monotonic(); seq = exact_strict(expr); calls = 0
    if seq is None:
        calls = 1
        with precision(spec['tool']['interval_dps']):
            seq = synth(expr, rational(spec['precision']), spec['tool']['synthesizer_options'])
    if not isinstance(seq, str) or any(c not in 'HTtSXW' for c in seq):
        raise ValueError('unsupported synthesis output')
    bound = strict_guard(seq, expr, dps=spec['tool']['interval_dps'])
    if not 0 <= bound <= rational(spec['precision']):
        raise ValueError('strict operator guard failed; no retry')
    row = dict(**expected, sequence=seq, sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),
               operator_error_upper=str(bound), complete=True, up_to_phase=False,
               backend_calls=calls, wall_seconds=time.monotonic()-start,
               T_count=seq.count('T')+seq.count('t'), Tdagger_count=seq.count('t'),
               Clifford_count=sum(seq.count(g) for g in 'HSX'), global_W_count=seq.count('W'),
               one_qubit_count=len(seq)-seq.count('W'))
    row['guard_binding'] = digest([expected, row['sequence_sha256'], str(bound)])
    return row


def checked(row, expected):
    check_cache_metadata(row, expected)
    if (row.get('guard_binding') != digest([expected, row['sequence_sha256'], row['operator_error_upper']])
            or any(c not in 'HTtSXW' for c in row['sequence'])
            or any(row.get(k) != v for k, v in dict(
                T_count=row['sequence'].count('T')+row['sequence'].count('t'),
                Tdagger_count=row['sequence'].count('t'),
                Clifford_count=sum(row['sequence'].count(g) for g in 'HSX'),
                global_W_count=row['sequence'].count('W'),
                one_qubit_count=len(row['sequence'])-row['sequence'].count('W')).items())):
        raise ValueError('unqualified/tampered strict row')
    return row


def time_letters(row, expected, sign):
    checked(row, expected)
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError('explicit signed inverse required')
    time_word = list(reversed(row['sequence']))
    inverse = dict(H='H', X='X', S='s', s='S', T='t', t='T', W='w', w='W')
    return tuple(time_word if sign == 1 else adjoint_plan(time_word, inverse.__getitem__))


def local_native(generator, row, expected, sign, *, phase_i_power=0):
    """Literal Z-string ladder only; no build/compile or FIXED basis pricing.

    Common parity lowering for all arms. Counts are local additive diagnostics,
    not an optimized full interrogation price. W/event scalar retained.
    """
    if (not isinstance(generator, str) or not generator or any(c not in 'IZ' for c in generator)
            or 'Z' not in generator or type(phase_i_power) is not int or phase_i_power not in range(4)):
        raise ValueError('V0 diagonal joint IZ word and exact event phase required')
    support = [q for q, c in enumerate(reversed(generator)) if c == 'Z']
    pivot = support[-1]
    ladder = [('CX', q, pivot) for q in support[:-1]]
    gates = [('W', pivot)]*(2*phase_i_power)+ladder
    gates += [(g, pivot) for g in time_letters(row, expected, sign)]
    gates += adjoint_plan(ladder, lambda g:g)
    cost = native_accounting(gates, {})
    # Expanded letters have no R token; attach the qualified primitive bound
    # explicitly rather than interpreting the helper's zero R count as exact.
    cost['strict_native_error_upper'] = rational(row['operator_error_upper'])
    cost['R_occurrences'] = 1
    cost['Tdagger_count'] = row['Tdagger_count'] if sign == 1 else row['T_count']-row['Tdagger_count']
    return dict(time_gates=tuple(gates), counts=cost,
                primitive_strict_error_upper=rational(row['operator_error_upper']),
                serial_depth=sum(g[0] not in ('W','w') for g in gates),
                compiled=False, fixed_basis_and_wrapper_cost='NOT_ACQUIRED')
