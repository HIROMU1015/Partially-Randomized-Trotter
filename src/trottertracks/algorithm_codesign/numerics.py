"""CPU spectral actions with a posteriori residual and roundoff guards.

The guard is conditional on IEEE binary64, standard BLAS forward-error
assumptions, and the supplied binary64 matrices. It is not a physical DF
truncation bound. No external eigensolver is assumed to return exact vectors.
"""
from __future__ import annotations
import math
import numpy as np

EPS = np.finfo(float).eps


def gamma(operations):
    x = operations*EPS
    if x >= .1:
        raise ValueError("Numerical guard outside its stable regime")
    return x/(1-x)


def phase_values(theta):
    """exp(-i theta) through a scaled Taylor series, with explicit remainder.

    This avoids assuming a black-box complex exp implementation is exact.
    One common scale is used for all eigenvalues; the short Taylor interval
    is bounded by 1/8, followed by repeated squaring.
    """
    theta = np.asarray(theta, dtype=float)
    magnitude = float(np.max(np.abs(theta), initial=0))
    scale = max(0, math.ceil(math.log2(max(1, magnitude*8))))
    x = -1j*theta/2**scale
    term = np.ones_like(x, dtype=complex)
    result = term.copy()
    for n in range(1, 19):
        term = term*x/n
        result += term
    error = math.exp(.125)*.125**19/math.factorial(19) + gamma(18*12)*math.exp(.125)
    for _ in range(scale):
        norm = float(np.max(np.abs(result), initial=0))
        error = (2*norm+error)*error+gamma(8)*norm*norm
        result *= result
    return result, error


def finite_values(eigenvalues, time, r, K):
    x = -1j*time/r*eigenvalues
    radius = float(np.max(np.abs(x), initial=0))
    term = np.ones_like(x, dtype=complex)
    result = term.copy()
    for n in range(1, K+2):
        term = term*x/n
        result += term
    absolute_sum = math.fsum(radius**n/math.factorial(n) for n in range(K+2))
    error = gamma(12*(K+2))*absolute_sum
    norm = float(np.max(np.abs(result), initial=0))
    value = np.ones_like(x, dtype=complex)
    value_error = 0.
    # r <= 80; explicit multiplication gives a transparent forward guard.
    for _ in range(r):
        value_norm = float(np.max(np.abs(value), initial=0))
        value_error = (norm+error)*value_error+error*value_norm+gamma(8)*norm*value_norm
        value *= result
    derivative = abs(time)*math.exp(radius)*(norm+error)**max(0, r-1)
    return value, value_error, derivative


class Spectrum:
    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
            raise ValueError("Invalid finite square generator")
        n = len(matrix)
        hermitian_error = float(np.linalg.norm(matrix-matrix.conj().T, 'fro'))/2
        if hermitian_error > 1e-10*max(1., float(np.linalg.norm(matrix, 'fro'))):
            raise ValueError("Generator is not Hermitian within source tolerance")
        self.values, self.vectors = np.linalg.eigh(.5*(matrix+matrix.conj().T))
        q = self.vectors
        gram = float(np.linalg.norm(q.conj().T@q-np.eye(n), 'fro')) + gamma(8*n)*float(np.linalg.norm(q, 'fro'))**2
        if gram >= .01:
            raise ValueError("Eigensystem orthogonality guard failed")
        delta = gram/(1+math.sqrt(1-gram))
        self.gram_error = delta*(math.sqrt(1+gram)+1)
        reconstruction = q@(self.values[:, None]*q.conj().T)
        residual = float(np.linalg.norm(matrix-reconstruction, 'fro'))
        residual += gamma(16*n+4)*float(np.linalg.norm(q, 'fro'))**2*float(np.max(np.abs(self.values), initial=0))
        self.matrix_error = residual+float(np.max(np.abs(self.values), initial=0))*self.gram_error+hermitian_error
        self.norm = float(np.max(np.abs(self.values), initial=0))+self.matrix_error
        self.dimension = n

    def action(self, state, time, time_error=0., r=None, K=2):
        if r is None:
            diagonal, scalar_error = phase_values(time*self.values)
            # Argument multiplication, supplied time lowering and Hermitian
            # perturbation each receive an explicit Lipschitz contribution.
            operator_error = scalar_error + self.gram_error + abs(time)*self.matrix_error
            operator_error += (time_error+gamma(2)*abs(time))*self.norm
            operator_norm = 1.
        else:
            diagonal, scalar_error, derivative = finite_values(self.values, time, r, K)
            operator_norm = max(1., float(np.max(np.abs(diagonal), initial=0))+scalar_error)
            operator_error = scalar_error+self.gram_error*operator_norm+derivative*self.matrix_error
            micro_radius = (abs(time)+time_error)*self.norm/r
            micro_bound = 1+math.exp(micro_radius)*micro_radius**(K+2)/math.factorial(K+2)
            operator_norm = max(operator_norm, micro_bound**r)
            derivative_upper = (abs(time)+time_error)*math.exp(micro_radius)*micro_bound**max(0, r-1)
            operator_error = max(operator_error, scalar_error+self.gram_error*operator_norm+derivative_upper*self.matrix_error)
            operator_error += (time_error+gamma(2)*abs(time))*self.norm*math.exp(micro_radius)*micro_bound**max(0, r-1)
        q = self.vectors
        result = q@(diagonal*(q.conj().T@state))
        multiply_error = gamma(16*self.dimension+8)*self.dimension*operator_norm*float(np.linalg.norm(state))
        error = operator_error*float(np.linalg.norm(state))+multiply_error
        if not np.all(np.isfinite(result)) or not math.isfinite(error):
            raise ValueError("Unrepresentable spectral action")
        return result, error, operator_norm
