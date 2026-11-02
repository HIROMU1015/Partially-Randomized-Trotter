"""Exact rational coefficient algebra; no dense trace coefficient oracle.

Pauli words are tuples in increasing physical qubit order, unlike Qiskit labels.
Every function is dormant until a separately approved B/F1 invocation.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from .fixture import FixtureSpec, FragmentSpec, fraction


@dataclass(frozen=True)
class GaussianRational:
    real: Fraction
    imag: Fraction

    def __add__(self, other):
        return GaussianRational(self.real+other.real, self.imag+other.imag)

    def __mul__(self, other):
        return GaussianRational(self.real*other.real-self.imag*other.imag,
                                self.real*other.imag+self.imag*other.real)

    def conjugate(self):
        return GaussianRational(self.real, -self.imag)

    def nonzero(self):
        return self.real != 0 or self.imag != 0


def gr(real=0, imag=0):
    return GaussianRational(Fraction(real), Fraction(imag))


def add(*maps):
    result = {}
    for terms in maps:
        for word, value in terms.items():
            result[word] = result.get(word, gr())+value
    return {w: c for w, c in result.items() if c.nonzero()}


def scale(terms, factor):
    return {w: c*factor for w, c in terms.items() if (c*factor).nonzero()}


def multiply(left, right):
    cycle = {('X','Y'):'Z', ('Y','Z'):'X', ('Z','X'):'Y'}
    out = {}
    for w, c in left.items():
        for v, d in right.items():
            z, phase = [], gr(1)
            for a, b in zip(w, v, strict=True):
                if a == 'I': z.append(b)
                elif b == 'I': z.append(a)
                elif a == b: z.append('I')
                elif (a,b) in cycle:
                    z.append(cycle[a,b]); phase = phase*gr(0,1)
                else:
                    z.append(cycle[b,a]); phase = phase*gr(0,-1)
            word = tuple(z)
            out[word] = out.get(word, gr())+c*d*phase
    return {w:c for w,c in out.items() if c.nonzero()}


def annihilator(modes, p):
    x = tuple('Z' if j<p else 'X' if j==p else 'I' for j in range(modes))
    y = tuple('Y' if j==p else x[j] for j in range(modes))
    return {x:gr(Fraction(1,2)), y:gr(0,Fraction(1,2))}


def rational_frame(modes: int, fragment: FragmentSpec):
    # Direct matrix product in the declared left-to-right matrix order.
    result = [[Fraction(int(i==j)) for j in range(modes)] for i in range(modes)]
    for g in fragment.rotations:
        g.validate(modes)
        matrix = [[Fraction(int(i==j)) for j in range(modes)] for i in range(modes)]
        c,s = fraction(g.cosine), fraction(g.sine)
        matrix[g.p][g.p]=matrix[g.q][g.q]=c
        matrix[g.p][g.q],matrix[g.q][g.p]=-s,s
        result = [[sum(result[i][k]*matrix[k][j] for k in range(modes))
                   for j in range(modes)] for i in range(modes)]
    return tuple(tuple(row) for row in result)


def frame_coefficients(fragment: FragmentSpec, modes: int):
    """Expand (sum d n)^2 in the original orbital diagonal basis."""
    diagonal = {}
    for p, text in enumerate(fragment.diagonal):
        half = fraction(text)/2
        ident = ('I',)*modes
        z = tuple('Z' if j==p else 'I' for j in range(modes))
        diagonal = add(diagonal, {ident:gr(half),z:gr(-half)})
    return real_map(multiply(diagonal,diagonal))


def pauli_coefficients(fragment: FragmentSpec, modes: int):
    """dGamma(V diag(d) V^T)^2 via exact JW polynomial algebra."""
    v = rational_frame(modes, fragment)
    d = tuple(fraction(x) for x in fragment.diagonal)
    a = tuple(annihilator(modes,p) for p in range(modes))
    one_body = {}
    for i in range(modes):
        for j in range(modes):
            g = sum(v[i][p]*d[p]*v[j][p] for p in range(modes))
            if g:
                adag = {w:c.conjugate() for w,c in a[i].items()}
                one_body = add(one_body,scale(multiply(adag,a[j]),gr(g)))
    return real_map(multiply(one_body,one_body))


def real_map(terms):
    result = {}
    for word,value in terms.items():
        if value.imag:
            raise ValueError('Non-Hermitian exact Pauli coefficient')
        if value.real:
            result[word] = value.real
    return dict(sorted(result.items()))


def materialize_alternatives(spec: FixtureSpec):
    """B-only operation; returning maps is science, even without dense matrices."""
    spec.validate()
    return tuple((f, rational_frame(spec.modes,f),
                  frame_coefficients(f,spec.modes), pauli_coefficients(f,spec.modes))
                 for f in spec.fragments)
