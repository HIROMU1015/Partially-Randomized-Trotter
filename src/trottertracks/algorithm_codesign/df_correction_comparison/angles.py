"""Closed angle types and phase-preserving sequence guard, no synthesis on import."""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import mpmath as mp
from .intervals import iv, upper, record
from ..synthesis_placement.economics import sequence_matrix


@dataclass(frozen=True)
class Angle:
    kind: str
    value: F
    scale: F = F(1)

    def __post_init__(self):
        if self.kind not in ('pi', 'atan', 'raw-rad'):
            raise ValueError('Unknown angle kind')
        object.__setattr__(self, 'value', F(self.value))
        object.__setattr__(self, 'scale', F(self.scale))

    @property
    def key(self):
        return f'{self.kind}:{self.value}:scale:{self.scale}'

    def at(self, ctx):
        q = ctx.mpf(self.value.numerator)/self.value.denominator
        s = ctx.mpf(self.scale.numerator)/self.scale.denominator
        if self.kind == 'pi':
            return s*ctx.pi*q
        if self.kind == 'atan':
            return s*ctx.atan2(q, ctx.mpf(1))
        return s*q

    def as_dict(self):
        return {'kind': self.kind, 'value': str(self.value), 'scale': str(self.scale), 'key': self.key}

    @classmethod
    def load(cls, row):
        x = cls(row['kind'], F(row['value']), F(row.get('scale', '1')))
        if 'key' in row and x.key != row['key']:
            raise ValueError('Angle key mismatch')
        return x


def strict_guard(sequence, angle):
    u = sequence_matrix(sequence)
    theta = angle.at(mp.iv)
    target = [[mp.iv.exp(-mp.iv.j*theta/2), 0], [0, mp.iv.exp(mp.iv.j*theta/2)]]
    return upper(mp.iv.sqrt(sum((abs(u[i][j]-target[i][j])**2
                                 for i in range(2) for j in range(2)), iv(0))))


def clifford_sequence(angle):
    # Exact strict RZ(k*pi/2)=W^(-k) S^k. RZ(pi/4) has NO phase-free fast path.
    if angle.kind != 'pi':
        return '' if angle.value == 0 else None
    k = 2*angle.value*angle.scale
    if k.denominator != 1:
        return None
    k = k.numerator
    return 'W'*((-k) % 8) + 'S'*(k % 4)


def synthesize(angle, epsilon, options):
    """Only called by an authorized synthesis child; tests never call this."""
    from .intervals import configure
    configure()
    if options.get('up_to_phase') is not False:
        raise ValueError('Strict phase required')
    sequence = clifford_sequence(angle)
    route = 'strict_clifford'
    if sequence is None:
        from pygridsynth.config import GridsynthConfig
        from pygridsynth.gridsynth import gridsynth_gates
        cfg = GridsynthConfig(**options)
        cfg.dps = 100  # Do not let the backend replace the frozen precision automatically.
        cfg.seed = int(hashlib.sha256(('20261011:'+angle.key).encode()).hexdigest()[:8], 16)
        sequence = gridsynth_gates(angle.at(mp.mp), mp.mpf(F(epsilon).numerator)/F(epsilon).denominator/4, cfg=cfg)
        route = 'pygridsynth_2.0.0'
    if not isinstance(sequence, str) or len(sequence) > 20000 or any(g not in 'HTtSXW' for g in sequence):
        raise RuntimeError('Sequence cap/alphabet failure')
    error = strict_guard(sequence, angle)
    if error > F(epsilon):
        raise ArithmeticError('Strict synthesis certificate failed; no retry')
    return {'angle': angle.as_dict(), 'route': route, 'sequence': sequence,
            'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
            'T_count': sequence.count('T')+sequence.count('t'), 'Tdagger_count': sequence.count('t'),
            'W_count': sequence.count('W'), 'strict_operator_error_upper': str(error), 'error_pass': True}
