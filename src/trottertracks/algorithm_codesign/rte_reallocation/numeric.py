"""R1 coefficient enclosures and strict, phase-preserving RZ synthesis guard."""
from fractions import Fraction as F
import hashlib
import mpmath as mp
from ..synthesis_placement.economics import sequence_matrix


def configure(dps=100):mp.mp.dps=dps;mp.iv.dps=dps


def fraction_value(raw):
    sign,mantissa,exponent,bits=raw
    if bits<0:raise ArithmeticError('nonfinite interval endpoint; no retry')
    return F(-mantissa if sign else mantissa)*F(2)**exponent


def enclosure(value):
    return tuple(fraction_value(p) for p in value._mpi_)


def ivf(q):
    q=F(q);return mp.iv.mpf(q.numerator)/q.denominator


def record(pair):return {'lo':str(pair[0]),'hi':str(pair[1])}


def angle_value(angle,ctx):
    q=ctx.mpf(angle.value.numerator)/angle.value.denominator
    scale=ctx.mpf(angle.scale.numerator)/angle.scale.denominator
    return scale*(ctx.pi*q if angle.kind=='pi' else ctx.atan2(q,ctx.mpf(1)))


def alpha_enclosure(event):
    # Round the order/group norm once, then multiply the exact rational word
    # probability. This preserves I0 order-then-IID generation after rounding.
    lo,hi=enclosure(mp.iv.sqrt(ivf(event.norm_square)))
    return lo*event.label_probability,hi*event.label_probability


def strict_guard(sequence,angle):
    """Frobenius upper >= strict operator norm, with NO phase minimization."""
    u=sequence_matrix(sequence);theta=angle_value(angle,mp.iv)
    v=[[mp.iv.exp(-mp.iv.j*theta/2),0],[0,mp.iv.exp(mp.iv.j*theta/2)]]
    error=mp.iv.sqrt(sum((abs(u[i][j]-v[i][j])**2 for i in range(2) for j in range(2)),mp.iv.mpf(0)))
    return enclosure(error)[1]


def synthesis_key(angle,epsilon):return angle.key+':epsilon:'+str(epsilon)


def synthesize(angle,epsilon,options,on_synthesis=None,max_characters=20000):
    """Only future authorized runner calls this. No calls in preparation/tests."""
    if options.get('up_to_phase') is not False:raise ValueError('strict phase synthesis required')
    from pygridsynth.config import GridsynthConfig
    from pygridsynth.gridsynth import gridsynth_gates
    theta=angle_value(angle,mp.mp);cfg=GridsynthConfig(**options)
    if on_synthesis is not None:on_synthesis()
    sequence=gridsynth_gates(theta,mp.mpf(epsilon)/4,cfg=cfg)
    if not isinstance(sequence,str) or any(g not in 'HTtSXW' for g in sequence):
        raise ValueError('invalid synthesizer sequence')
    if len(sequence)>max_characters:raise RuntimeError('sequence output cap hit')
    bound=strict_guard(sequence,angle)
    return {'key':synthesis_key(angle,epsilon),'angle_key':angle.key,'epsilon':str(epsilon),
            'sequence':sequence,'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),
            'T_count':sequence.count('T')+sequence.count('t'),
            'Tdagger_count':sequence.count('t'),
            'one_qubit_count':len(sequence)-sequence.count('W'),
            'global_W_count':sequence.count('W'),
            'strict_operator_error_upper':str(bound),'error_pass':bound<=F(epsilon)}


def validate_saved(row,angle,epsilon):
    s=row['sequence']
    if (row['key']!=synthesis_key(angle,epsilon) or row['angle_key']!=angle.key
        or row['epsilon']!=str(epsilon) or any(g not in 'HTtSXW' for g in s)
        or hashlib.sha256(s.encode()).hexdigest()!=row['sequence_sha256']
        or row['T_count']!=s.count('T')+s.count('t')
        or row['Tdagger_count']!=s.count('t') or row['global_W_count']!=s.count('W')
        or row['one_qubit_count']!=len(s)-s.count('W')
        or row['error_pass'] is not True or not 0<=F(row['strict_operator_error_upper'])<=F(epsilon)):
        raise PermissionError('sequence/count/strict-error/key identity mismatch')
    return row
