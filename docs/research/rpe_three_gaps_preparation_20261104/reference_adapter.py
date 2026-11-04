"""Certified minimum-eigenvalue bracket for an exact saved-coefficient DF model.

Only pure functions on supplied artificial or separately permitted inputs.
No eigensolver, residual-to-ground assumption, circuit, or molecular generator.
"""
from fractions import Fraction as Q
from dataclasses import dataclass
import math
from types import SimpleNamespace
import numpy as np
from flint import acb, acb_mat, arb
from common import D, ball, encode, excitation_functions, real, upper


@dataclass(frozen=True)
class HermitianEnclosure:
    """Family of Hermitian matrices defined by its real diagonal/lower balls."""
    matrix: acb_mat


def exact_complex(value):
    z = complex(value)
    if not math.isfinite(z.real) or not math.isfinite(z.imag):
        D.stop('nonfinite coefficient/state entry')
    return Q.from_float(z.real), Q.from_float(z.imag)


def coefficient_matrix(values):
    """Require exact Hermiticity of represented coefficients; never symmetrize."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        D.stop('square coefficient matrix required')
    pairs = [[exact_complex(z) for z in row] for row in values]
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            x, y = pairs[i][j], pairs[j][i]
            if x != (y[0], -y[1]):
                D.stop('saved coefficient is not exactly Hermitian')
    return [[acb(real(x), real(y)) for x, y in row] for row in pairs]


def complete_number_sector(n_qubits, n_electrons, basis):
    if type(n_qubits) is not int or type(n_electrons) is not int or not 0 <= n_electrons <= n_qubits:
        D.stop('invalid number sector')
    expected = [i for i in range(1 << n_qubits) if i.bit_count() == n_electrons]
    if list(basis) != expected:
        D.stop('saved basis is not the complete ordered number sector')
    return SimpleNamespace(n_qubits=n_qubits, basis_indices=np.asarray(expected, dtype=np.int64),
                           n_electrons=n_electrons, dimension=len(expected))


def represented_df(constant, one_body, lambdas, fragments, sector):
    """H=cI+T(h)+sum lambda*T(g)^2, exact binary64 operands + Arb arithmetic.

    Reuses the old excitation signs and basis conventions. Does not invoke the
    old rounded matvec, materializer, eigsh, or NumPy eigh.
    """
    c_re, c_im = exact_complex(constant)
    if c_im or len(lambdas) != len(fragments):
        D.stop('invalid real constant/DF inventory')
    functions, records = excitation_functions()
    table = functions['_build_excitation_table'](sector)
    dimension = sector.dimension

    def lift(coefficients):
        coeff = coefficient_matrix(coefficients)
        if len(coeff) != sector.n_qubits:
            D.stop('coefficient/qubit dimension mismatch')
        out = acb_mat(dimension, dimension)
        for i, j, p, q, s in zip(table.target_positions, table.source_positions,
                                  table.p_indices, table.q_indices, table.signs):
            out[int(i), int(j)] += int(s)*coeff[int(p)][int(q)]
        return out

    identity = ball.identity(dimension)
    result = real(c_re)*identity+lift(one_body)
    for weight, fragment in zip(lambdas, fragments):
        re, im = exact_complex(weight)
        if im:
            D.stop('nonreal DF weight')
        g = lift(fragment)
        result += real(re)*(g*g)
    # The exact polynomial target is Hermitian by the checks above; use its
    # lower-triangle enclosures and real diagonals. This is not changing H.
    return hermitian_lower(result), records


def hermitian_lower(matrix):
    """Packed lower bounds representing a Hermitian target; structural contract.

    Caller must establish Hermiticity of the target, not merely interval overlap.
    represented_df does so from exactly Hermitian coefficient operands.
    """
    if matrix.nrows() != matrix.ncols() or matrix.nrows() < 1:
        D.stop('nonempty square matrix required')
    n = matrix.nrows()
    out = acb_mat(n, n)
    for i in range(n):
        if not matrix[i, i].imag.contains(0):
            D.stop('Hermitian target diagonal inconsistent')
        out[i, i] = acb(matrix[i, i].real, 0)
        for j in range(i):
            if not matrix[i, j].is_finite():
                D.stop('nonfinite matrix enclosure')
            out[i, j], out[j, i] = matrix[i, j], matrix[i, j].conjugate()
    return HermitianEnclosure(out)


def positive_shift(matrix, shift):
    """Unpivoted interval LDL*: all positive pivots prove H-shift*I > 0.

    All matrices in the supplied Hermitian lower-triangle enclosure are covered.
    A zero/uncertain/negative pivot is UNKNOWN, not a negative-eigenvalue claim.
    """
    if not isinstance(matrix, HermitianEnclosure):
        D.stop('explicit Hermitian target enclosure required')
    matrix = matrix.matrix
    n, shift = matrix.nrows(), real(shift)
    lower = acb_mat(n, n)
    pivots = []
    for j in range(n):
        pivot = matrix[j, j].real-shift
        for k in range(j):
            pivot -= abs(lower[j, k])**2*pivots[k]
        if not pivot.is_finite() or not pivot > 0:
            return dict(decision='UNKNOWN/STOP', stopped_pivot=j,
                        pivot=encode(pivot) if pivot.is_finite() else None), None
        pivots.append(pivot)
        lower[j, j] = 1
        for i in range(j+1, n):
            numerator = matrix[i, j]
            for k in range(j):
                numerator -= lower[i, k]*pivots[k]*lower[j, k].conjugate()
            lower[i, j] = numerator/pivot
    return dict(decision='STRICT_POSITIVE_DEFINITE', positive_pivots=n,
                pivot_intervals=[encode(x) for x in pivots]), pivots


def ground_bracket(matrix, state, energy_reference, lower_offset):
    """A verified lower bound + Rayleigh upper bound identifies the minimum.

    state need only be nonzero; it need not be a certified ground vector.
    """
    energy, offset = real(energy_reference), D.positive(lower_offset)
    if not isinstance(matrix, HermitianEnclosure):
        D.stop('explicit Hermitian target enclosure required')
    if len(state) != matrix.matrix.nrows():
        D.stop('matrix/state dimension mismatch')
    lower_bound = (energy-offset).lower()
    proof, pivots = positive_shift(matrix, lower_bound)
    if pivots is None:
        return dict(decision='UNKNOWN/STOP', positive_shift=proof, eta_ref_upper=None)
    matrix = matrix.matrix
    vector = acb_mat([[acb(real(x), real(y))] for x, y in map(exact_complex, state)])
    norm2 = (ball.dagger(vector)*vector)[0, 0].real
    if not norm2 > 0:
        D.stop('nonzero saved ray not established')
    rayleigh = (ball.dagger(vector)*matrix*vector)[0, 0]/norm2
    if not rayleigh.is_finite() or not rayleigh.imag.contains(0):
        D.stop('Rayleigh enclosure incompatible with Hermitian target')
    upper_bound = upper(rayleigh.real)
    if not lower_bound < upper_bound:
        D.stop('inconsistent ground bracket')
    eta = upper(abs(energy-lower_bound)).max(upper(abs(energy-upper_bound)))
    return dict(decision='REPRESENTED_DF_SECTOR_GROUND_BRACKET', positive_shift=proof,
                eigenvalue_interval={'lower': str(ball.dyadic(lower_bound)),
                                     'upper': str(ball.dyadic(upper_bound))},
                rayleigh_interval=encode(rayleigh.real), norm_squared=encode(norm2),
                eta_ref_upper=encode(upper(eta)))
