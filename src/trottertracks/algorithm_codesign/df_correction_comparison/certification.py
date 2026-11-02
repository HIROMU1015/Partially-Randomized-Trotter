"""Frame/assembly/Pauli guards, conditional on IEEE/BLAS forward error.

Ideal columns are exact positive-norm, ordered Gram-Schmidt of the binary64
recipe output. This fixes the ideal input without regenerating random draws.
An exact computational Pauli table approximates that ideal generator; its
assembly distance is charged, including computational zeros. No threshold drop.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cached_property
import itertools
import math
import numpy as np
import mpmath as mp
from .intervals import configure, iv, upper, complex_exact, bounds, record, outward_upper
from ..numerics import Spectrum

EPS = F(1, 2**52)


def gamma(k):
    x = k*EPS
    if x >= F(1, 10):
        raise ArithmeticError('Forward-error regime exceeded')
    return x/(1-x)


def frobenius_upper(a):
    q = sum((F(float(z.real))**2+F(float(z.imag))**2 for z in np.asarray(a).flat), F(0))
    return upper(mp.iv.sqrt(iv(q)))


@dataclass
class Ball:
    value: object
    error: F = F(0)

    def __post_init__(self):
        if self.error<0:raise ArithmeticError('Negative matrix error radius')
        self.error=outward_upper(self.error)
        self.value=np.asarray(self.value,dtype=complex)
        self.value.setflags(write=False)

    @cached_property
    def norm(self):
        return frobenius_upper(self.value)

    def mul(self, other):
        n = self.value.shape[1]
        radius = self.norm*other.error+other.norm*self.error+self.error*other.error
        radius += gamma(8*n+4)*self.norm*other.norm
        return Ball(self.value @ other.value, radius)

    def add(self, other, sign=1):
        radius = self.error+other.error+gamma(2)*(self.norm+other.norm)
        value=self.value-other.value if sign==-1 else self.value+other.value
        return Ball(value, radius)

    def scale(self, scalar):
        q = F(scalar)
        f = float(q)
        return Ball(f*self.value, abs(q)*self.error+abs(F(f)-q)*self.norm+gamma(2)*abs(q)*self.norm)

    def adjoint(self):
        return Ball(self.value.conj().T, self.error)


def frame_guard(columns):
    configure()
    n, rank = columns.shape
    raw = [[complex_exact(columns[i, j]) for i in range(n)] for j in range(rank)]
    ideal = []
    for col in raw:
        v = list(col)
        for prev in ideal:
            # mpmath1.3 iv.mpc.conjugate has an interval-tuple bug. Construct
            # interval conjugation componentwise, with no mp point fallback.
            dot = sum((mp.iv.mpc(prev[i].real,-prev[i].imag)*v[i] for i in range(n)), mp.iv.mpc(0))
            v = [v[i]-dot*prev[i] for i in range(n)]
        norm2 = sum((abs(z)**2 for z in v), iv(0))
        if bounds(norm2)[0] <= 0:
            raise ArithmeticError('Frame rank not certified; no rank dropping')
        scale = mp.iv.sqrt(norm2)
        ideal.append([z/scale for z in v])
    errors = [upper(mp.iv.sqrt(sum((abs(ideal[j][i]-raw[j][i])**2 for i in range(n)), iv(0)))) for j in range(rank)]
    return errors, {'ideal_definition': 'ordered_exact_Gram_Schmidt_of_exact_binary64_recipe_frame',
                    'positive_norms_certified': True, 'column_distance_upper': [str(e) for e in errors]}


def annihilators(n):
    out = []
    for k in range(n):
        a = np.zeros((1 << n, 1 << n), complex)
        for state in range(1 << n):
            if state & (1 << k):
                a[state ^ (1 << k), state] = (-1)**((state & ((1 << k)-1)).bit_count())
        out.append(a)
    return out


def primitive(columns, sign):
    errors, certificate = frame_guard(columns)
    n, rank = columns.shape
    d = 1 << n
    ops, eye = annihilators(n), np.eye(d, dtype=complex)
    result = Ball(sign*eye)
    for j, col in enumerate(columns.T):
        # Same operation order as the frozen source's Python sum and matrix products.
        terms = [col[k].conjugate()*ops[k] for k in range(n)]
        a = sum(terms)
        floating = gamma(8*n+4)*sum(frobenius_upper(t) for t in terms)
        # CAR norm of an annihilator equals the orbital 2-norm.
        b = Ball(a, errors[j]+floating)
        occupied = b.adjoint().mul(b).scale(2)
        factor = Ball(eye).add(occupied, -1)
        result = result.mul(factor)
    certificate['primitive_assembly_operator_error_upper'] = str(result.error)
    certificate['guard_model'] = 'IEEE_binary64_standard_complex_BLAS_forward_error'
    return result, certificate


def comm(a, b):
    return a.mul(b).add(b.mul(a), -1)


def pauli_matrix(label):
    mats = {'I': np.eye(2), 'X': np.array([[0,1],[1,0]]),
            'Y': np.array([[0,-1j],[1j,0]]), 'Z': np.diag([1,-1])}
    out = np.array([[1]], complex)
    # Labels use Qiskit big-endian display; index bit zero is rightmost.
    for c in label:
        out = np.kron(out, mats[c])
    return out


def pauli_table(generator, *, number_conserving=False, traceless=False):
    """Exact rational transform of a Hermitian, symmetry-projected proxy.

    Proxy projection distance plus input/BLAS ball encloses the ideal T.
    Exact zeros here are COMPUTATIONAL zeros, not claims about ideal beta.
    Their ideal contribution is covered by the assembly operator distance.
    """
    configure()
    a = np.asarray(generator.value, complex)
    n = int(round(math.log2(len(a))))
    if a.shape != (1 << n, 1 << n):
        raise ValueError('Invalid Pauli dimension')
    table, records = {}, []
    # Exact dyadic Hermitian projection, evaluated directly as rational entries.
    # Avoid converting Fraction to complex: real/imag exact pairs.
    pairs = [[((F(float(a[i,j].real))+F(float(a[j,i].real)))/2,
               (F(float(a[i,j].imag))-F(float(a[j,i].imag)))/2)
              for j in range(len(a))] for i in range(len(a))]
    for label_tuple in itertools.product('IXYZ', repeat=n):
        label = ''.join(label_tuple)
        beta = F(0)
        for state in range(1 << n):
            dest, phase = state, 1
            for bit, char in enumerate(reversed(label)):
                if char in 'XY':
                    dest ^= 1 << bit
                if char == 'Y':
                    phase *= 1j*(-1 if state & (1 << bit) else 1)
                elif char == 'Z':
                    phase *= -1 if state & (1 << bit) else 1
            re, im = pairs[state][dest]
            beta += int(phase.real)*re-int(phase.imag)*im
        beta /= 1 << n
        structural_zero = (traceless and label == 'I'*n) or (number_conserving and sum(c in 'XY' for c in label) % 2 == 1)
        if structural_zero:
            beta = F(0)  # trace(comm)=0 and number-conserving even CAR parity.
        table[label] = beta
        records.append({'label': label, 'computational_beta': str(beta),
                        'zero_status': 'PROVED_IDEAL_TRACE_OR_PARITY_ZERO' if structural_zero else
                        ('EXACT_COMPUTATIONAL_ZERO_IDEAL_UNRESOLVED_CHARGED' if beta == 0 else 'COMPUTATIONAL_NONZERO'),
                        'ideal_beta_radius_upper': str(generator.error)})
    # Trace/parity proxy correction error <= sum of exact removed Pauli masses;
    # computing exact reconstruction also covers Hermitian projection roundoff.
    rebuilt = np.zeros_like(a)
    round_bound = F(0)
    for label, beta in table.items():
        if beta:
            p = pauli_matrix(label)
            round_bound += abs(F(float(beta))-beta)*frobenius_upper(p)
            rebuilt += float(beta)*p
    round_bound += gamma(2*len(table))*sum(abs(v) for v in table.values())*(1 << n)
    radius = outward_upper(generator.error+frobenius_upper(rebuilt-a)+round_bound)
    for row in records:
        row['ideal_beta_radius_upper'] = str(radius)
    return table, rebuilt, radius, records


def verified_rotation(generator, time):
    """Independent spectral enclosure; exact target exp(i time T_ideal)."""
    source = Spectrum(generator.value)
    columns, errors = [], []
    for i in range(len(generator.value)):
        v = np.eye(len(generator.value), dtype=complex)[:, i]
        out, error, _ = source.action(v, -float(time),
                                     time_error=float(abs(F(float(time))-F(time))),
                                     assembly_error=float(generator.error))
        columns.append(out)
        errors.append(F(float(error)))
    radius = upper(mp.iv.sqrt(iv(sum(e*e for e in errors))))*(1+F(1, 2**40))
    return Ball(np.column_stack(columns), radius)
