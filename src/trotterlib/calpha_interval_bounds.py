"""Directed interval arithmetic for the supplied C-alpha bound formulas.

Uses basic mpmath 1.3 interval operations. Matrix exponentials have an explicit
Taylor remainder; mpmath.expm's point stopping rule is deliberately not used.
No molecular data or calculation is loaded on import.
"""
from fractions import Fraction
import math


def endpoint(value, upper=False):
    sign, man, exponent, _ = value._mpi_[int(upper)]
    return (-1 if sign else 1)*Fraction(man)*Fraction(2)**exponent


class IntervalBounds:
    def __init__(self, digits=40):
        from mpmath.ctx_iv import MPIntervalContext
        self.c = MPIntervalContext()
        self.c.dps = digits

    def real(self, value):
        if isinstance(value, Fraction):
            return self.c.mpf(value.numerator)/value.denominator
        return self.c.mpf(value)

    def matrix(self, array):
        return self.c.matrix([[self.c.mpc(float(complex(x).real),float(complex(x).imag))
                               for x in row] for row in array])

    def norm(self, matrix):
        return self.c.sqrt(sum((abs(x)**2 for x in matrix),self.c.mpf(0))).b

    def dagger(self, matrix):
        # mpmath 1.3.0 iv.mpc.conjugate incorrectly calls scalar mpf_neg on
        # an interval. Basic real/imag interval arithmetic avoids that bug.
        return matrix.apply(lambda z:self.c.mpc(z.real,-z.imag)).T

    def error_box(self, matrix, error):
        radius = endpoint(error,True)
        box = self.c.mpc(self.c.mpf([-self.real(radius).b,self.real(radius).b]),
                        self.c.mpf([-self.real(radius).b,self.real(radius).b]))
        return matrix.apply(lambda x:x+box)

    def expm(self, matrix, degree=32):
        total = self.c.eye(matrix.rows)
        term = total.copy()
        for k in range(1,degree+1):
            term = term*matrix/k
            total += term
        size = self.norm(matrix)
        remainder = self.c.exp(size)*size**(degree+1)/math.factorial(degree+1)
        return self.error_box(total,remainder)

    def tail_log(self, tail, lam, theta, norm_upper):
        n = tail.rows
        ident = self.c.eye(n)
        x = theta**2*(tail*tail)
        power = ident.copy()
        atan_factor = ident.copy()
        log_matrix = self.c.zeros(n)
        for k in range(1,7):
            power = power*x
            atan_factor += (-1)**k*power/(2*k+1)
            log_matrix += (-1)**(k+1)*power/k
        ratio = (theta*norm_upper)**2
        rt_error = norm_upper*ratio**7/(15*(1-ratio))
        gamma_error = ratio**7/(7*(1-ratio))/(2*theta)
        rt = self.error_box(atan_factor*tail,rt_error)
        gamma = self.error_box((self.c.ln(1+(lam*theta)**2)*ident-log_matrix)/(2*theta),gamma_error)
        return rt,gamma

    def weighted_state_pf(self, defect, generator, eigenvalues, eigenvectors,
                          coefficients, psi, delta, rate, q):
        """Retain contraction in the same exact telescoping identity as (3.6).

        Approximate eigenpairs are only an enclosure device. Their residuals
        are propagated by Duhamel; they are not presumed exact or unitary.
        """
        c = self.c
        values = [c.mpc(float(x.real),float(x.imag)) for x in eigenvalues]
        v = self.matrix(eigenvectors)
        coeff = self.matrix([[x] for x in coefficients])
        residual_state = self.norm(psi-v*coeff)
        defect_norm = self.norm(defect)
        dv = defect*v
        residual = generator*v-v*c.diag(values)
        common = c.exp(-rate*delta*(q-1))
        main = c.mpf(0)
        eigen_error = c.mpf(0)
        for j,value in enumerate(values):
            x = (value.real+rate)*delta
            # If the interval crosses zero, use a continuous safe enclosure.
            if endpoint(x) <= 0 <= endpoint(x,True):
                geom = q*c.exp(abs(x).b*q)
            else:
                geom = c.expm1(q*x)/c.expm1(x)
            main += abs(coeff[j])*self.norm(dv[:,j])*geom
            eigen_error += (abs(coeff[j])*self.norm(residual[:,j])
                            *delta*q*(q-1)/2*c.exp(abs(value.real+rate).b*q*delta))
        return (common*(main+defect_norm*(q*residual_state+eigen_error))).b
