"""Unexecuted conditional binary64 mean-amplitude kernels, never a CPTP solver.

Frobenius radii use the candidate BLAS/trig/norm/underflow model. They are not
certified intervals. Only the future guarded constructor imports NumPy.
"""
from fractions import Fraction
import math
from .control import POLICY, require_permit


class Ball:
    def __init__(self, matrix, radius, op_bound):
        self.m, self.e, self.b = matrix, radius, op_bound


class Kernels:
    def __init__(self, permit):
        require_permit(permit)
        import numpy as np
        if np.__version__ != POLICY['runtime']['numpy']:
            raise RuntimeError('NumPy version')
        self.np = np
        self.products = 0
        self.twirls = 0

    def up(self, x):
        if not math.isfinite(x) or x < 0:
            raise RuntimeError('nonfinite/negative error budget')
        # Positive scalar expressions use <64 elementary operations. Conditional
        # 512u headroom plus nextafter covers their evaluation, absent underflow.
        return math.nextafter(x*(1+2.0**-44), math.inf) if x else 0.0

    def gamma(self, n):
        t = n * 2.0**-53
        if t >= 1:
            raise RuntimeError('gamma model domain')
        return self.up(t/(1-t))

    def rational_up(self, q):
        value = float(q)
        if Fraction(value) < q:
            value = math.nextafter(value, math.inf)
        return value

    def finite(self, matrix):
        np = self.np
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError('nonfinite arithmetic')
        for part in (matrix.real, matrix.imag):
            if np.any((np.abs(part) > 0) & (np.abs(part) < 2.0**-1022)):
                raise RuntimeError('observed subnormal; conditional model STOP')

    def norm(self, a):
        self.finite(a)
        return self.up(float(self.np.linalg.norm(a)) * (1+self.gamma(8*a.size+64)))

    def eye(self, n):
        return Ball(self.np.eye(n, dtype=self.np.complex128), 0.0, 1.0)

    def exact_dense(self, matrix, divisor=Fraction(1), known_op_bound=None):
        exponent, data = matrix
        m = self.np.zeros((256, 256), dtype=self.np.complex128)
        errors = [Fraction(0)]*256
        norms = [Fraction(0)]*256
        for (r, c), (a, b) in data.items():
            real = Fraction(a, 1 << exponent)/divisor
            imag = Fraction(b, 1 << exponent)/divisor
            fr, fi = float(real), float(imag)
            m[r, c] = complex(fr, fi)
            errors[r] += abs(Fraction(fr)-real)+abs(Fraction(fi)-imag)
            norms[r] += abs(real)+abs(imag)
        # ||error||F <= sqrt(256) max_row_l1; Hermitian target ||H||2<=max_row_l1.
        error = self.rational_up(16*max(errors))
        bound = self.rational_up(max(norms)) if known_op_bound is None else known_op_bound
        self.finite(m)
        return Ball(m, error, bound)

    def add(self, a, b):
        m = a.m+b.m
        e = self.up(a.e+b.e+self.gamma(2)*(self.norm(a.m)+self.norm(b.m)))
        self.finite(m)
        return Ball(m, e, self.up(a.b+b.b))

    def scale(self, a, scalar, scalar_error=0.0):
        m = scalar*a.m
        magnitude = self.up(abs(scalar))
        e = self.up((magnitude+scalar_error)*a.e+scalar_error*self.norm(a.m)+self.gamma(8)*magnitude*self.norm(a.m))
        self.finite(m)
        return Ball(m, e, self.up((magnitude+scalar_error)*a.b))

    def mul(self, a, b):
        self.products += 1
        if self.products > POLICY['resources']['max_matrix_products']:
            raise RuntimeError('matrix product cap')
        d = a.m.shape[1]
        m = a.m@b.m
        e = self.up(a.b*b.e+b.b*a.e+a.e*b.e+self.gamma(8*d+32)*self.norm(a.m)*self.norm(b.m))
        self.finite(m)
        return Ball(m, e, self.up(a.b*b.b))

    def power(self, a, count):
        out = self.eye(len(a.m))
        base = a
        while count:
            if count & 1:
                out = self.mul(out, base)
            count //= 2
            if count:
                base = self.mul(base, base)
        return out

    def exp_i(self, h, time):
        time = Fraction(time)
        tf = float(time)
        et = self.rational_up(abs(Fraction(tf)-time))
        k = 0
        while self.up(abs(tf)*h.b+et*h.b) > 2.0**k/8:
            k += 1
            if k > POLICY['precision']['matrix_exp_max_squarings']:
                raise RuntimeError('matrix exponential squaring cap')
        a = self.scale(h, 1j*tf/(1 << k), et/(1 << k))
        value = self.eye(len(h.m)); term = self.eye(len(h.m))
        degree = POLICY['precision']['matrix_exp_degree']
        for j in range(1, degree+1):
            inv = 1.0/j
            einv = self.rational_up(abs(Fraction(inv)-Fraction(1, j)))
            term = self.scale(self.mul(term, a), inv, einv)
            value = self.add(value, term)
        # e^b b^(m+1)/(m+1)! <= (8/7)(1/8)^(m+1)/(m+1)! for b<=1/8.
        tail = Fraction(8, 7)*Fraction(1, 8)**(degree+1)/math.factorial(degree+1)
        square_root = math.isqrt(len(h.m))
        sqrt_bound = square_root+int(square_root*square_root < len(h.m))
        value.e = self.up(value.e+sqrt_bound*self.rational_up(tail))
        value.b = 1.0  # Exact target exp(i*t*Hermitian H) is unitary.
        for _ in range(k):
            value = self.mul(value, value)
            value.b = 1.0
        return value

    def fwht(self, a):
        out = a.copy(); n = out.shape[-1]; width = 1
        while width < n:
            blocks = out.reshape(*out.shape[:-1], -1, 2*width)
            left = blocks[..., :width].copy(); right = blocks[..., width:].copy()
            blocks[..., :width] = left+right
            blocks[..., width:] = left-right
            width *= 2
        return out

    def twirl_plan(self, coefficients, n):
        weights = self.np.zeros((n, n), dtype=self.np.float64)
        mass = sum(abs(c) for c in coefficients.values())
        conversion = Fraction(0)
        for word, coefficient in coefficients.items():
            x = z = 0
            for letter in word:
                x = 2*x+int(letter in 'XY'); z = 2*z+int(letter in 'YZ')
            exact = abs(coefficient)/mass; approximate = float(exact)
            weights[x, z] = approximate
            conversion += abs(Fraction(approximate)-exact)
        spectrum = self.fwht(self.fwht(weights).T)
        error = self.up(self.rational_up(conversion)+self.gamma(32)*(1+self.rational_up(conversion)))
        rows = self.np.broadcast_to(self.np.arange(n), (n, n))
        cols = rows ^ self.np.arange(n)[:, None]
        return spectrum, error, rows, cols

    def twirl(self, a, plan):
        self.twirls += 1
        if self.twirls > POLICY['resources']['max_twirl_applications']:
            raise RuntimeError('twirl application cap')
        spectrum, error, rows, cols = plan
        diagonal = a.m[rows, cols]
        transformed = self.fwht(diagonal)
        result = self.fwht(transformed*spectrum)/len(a.m)
        m = self.np.empty_like(a.m); m[rows, cols] = result
        # Two Walsh transforms, diagonal multiply and power-of-two scaling.
        e = self.up(a.e+(error+self.gamma(128))*(self.norm(a.m)+a.e))
        self.finite(m)
        return Ball(m, e, a.b)

    def verify_twirl_probe(self, coefficients, plan):
        """Future authorized full-size probe: literal tensor PXP vs Walsh path."""
        np=self.np
        n=len(plan[0]);index=np.arange(n*n).reshape(n,n)
        probe=Ball(((index%7)-3+1j*((index%5)-2)).astype(np.complex128),0.0,float(8*n))
        direct=Ball(np.zeros((n,n),dtype=np.complex128),0.0,0.0)
        local={'I':np.eye(2,dtype=np.complex128),'X':np.array([[0,1],[1,0]],dtype=np.complex128),
               'Y':np.array([[0,-1j],[1j,0]],dtype=np.complex128),'Z':np.diag([1.,-1.]).astype(np.complex128)}
        mass=sum(abs(c) for c in coefficients.values())
        if len(coefficients)>POLICY['resources']['max_saved_law_terms']:raise RuntimeError('saved-law resource cap; no cutoff')
        for word,c in coefficients.items():
            p=np.array([[1.+0j]])
            for letter in word:p=np.kron(p,local[letter])
            left=np.argmax(np.abs(p),axis=1);right=np.argmax(np.abs(p),axis=0)
            lp=p[np.arange(n),left];rp=p[right,np.arange(n)]
            # Literal tensor monomial action, no Walsh/XOR masks from main path.
            term=Ball(lp[:,None]*probe.m[left[:,None],right[None,:]]*rp[None,:],0.0,probe.b)
            weight=abs(c)/mass;wf=float(weight)
            direct=self.add(direct,self.scale(term,wf,self.rational_up(abs(Fraction(wf)-weight))))
        transformed=self.twirl(probe,plan)
        if self.norm(direct.m-transformed.m)>self.up(direct.e+transformed.e):raise RuntimeError('correctness FAIL: full-size tensor/Walsh probe')
        return {'status':'PASS','dimension':n,'terms':len(coefficients),'reference':'literal tensor monomial conjugation','reference_kernel_independence':'no Walsh masks; shared binary64 arithmetic model'}

    def sincos(self, theta):
        x = float(theta)
        argument_error = self.rational_up(abs(Fraction(x)-theta))
        error = self.up(argument_error+2.0**-51)
        return math.sin(x), math.cos(x), error

    def mean(self, h, theta):
        s, c, e = self.sincos(theta)
        value = self.add(self.scale(self.eye(len(h.m)), c, e), self.scale(h, 1j*s, e))
        value.b = 1.0
        return value

    def phi(self, x, h, theta, plan):
        # Exact Phi is a convex mean of X -> A_j X A_j with unitary A_j.
        # It is Frobenius nonexpansive even though it is not a CPTP channel.
        # Evaluate rounding/parameter errors at the stored centre Xhat, then
        # transport the original X uncertainty once through exact Phi.
        input_error, target_bound = x.e, x.b
        x = Ball(x.m, 0.0, self.up(x.b+x.e))
        s, c, e = self.sincos(theta)
        scalar_error = self.up(2*e+e*e+self.gamma(8))
        hx = self.add(self.mul(h, x), self.mul(x, h))
        out = self.add(self.add(self.scale(x, c*c, scalar_error), self.scale(hx, 1j*c*s, scalar_error)), self.scale(self.twirl(x, plan), -s*s, scalar_error))
        out.e = self.up(input_error+out.e)
        out.b = target_bound  # E[A_j X A_j], not E[A_j X A_j^dagger].
        return out

    def shared(self, x, d, h, theta, R, plan):
        value = self.mul(self.mul(d, x), d)
        value.b = x.b
        for _ in range(R):
            value = self.phi(value, h, theta, plan)
        value = self.mul(self.mul(d, value), d)
        value.b = x.b
        return value

    def signal(self, a, state, norm2):
        np = self.np
        norm_float = float(norm2)
        value = complex(np.vdot(state, a.m@state)/norm_float)
        norm_error = self.rational_up(abs(Fraction(norm_float)-norm2)/norm2)
        e = self.up(a.e+self.gamma(16*len(state)+64)*self.norm(a.m)*(1+norm_error)+norm_error*(1+a.e))
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise RuntimeError('nonfinite signal')
        return value, e
