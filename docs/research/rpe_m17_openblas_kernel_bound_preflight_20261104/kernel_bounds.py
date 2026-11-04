"""Bounds for the inspected binary64 paths; no loader, RNG or scientific CLI.

The contract is path/FP-mode conditional. It is not a certification of real
H4 norms, all-support completion, historical payloads, or fixed PCG64 IID.
"""
from fractions import Fraction as Q
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_virtual_measurement_bridge_preparation_20261104'))
from virtual_bridge_adapter import KernelBounds

U = Q(1, 1 << 53)
ETA = Q(1, 1 << 1075)  # half the smallest binary64 subnormal
PROFILE = {
    'numpy': '1.26.4', 'blas_core': 'Prescott',
    'blas_sha256': '9254d0854dd7615e11de28d771ae408878ca8123a7ac204f21e4cc7a376cc2e5',
    'numpy_binary_sha256': 'a735e4e8355b75c800112af8a5b1731b891b2016ed57067352eea6cab0aa00bc',
    'libm_sha256': '1a08a427bbf3790aa434d2310e8c724846ec72ba7811982e85b0002dd296ca92',
}


def fp_mode_supported(mxcsr):
    # Sticky exception flags are irrelevant; RC/DAZ/FTZ and exception masks matter.
    return type(mxcsr) is int and 0 <= mxcsr < (1 << 32) and (mxcsr & 0xffc0) == 0x1f80


def gamma(depth):
    if type(depth) is not int or depth < 0 or depth*U >= 1:
        raise ValueError('Invalid binary64 roundoff depth')
    return depth*U/(1-depth*U)


def path_bounds(rows, columns):
    """C-contiguous complex128 GEMVT/DOT, positive contiguous vectors, <=70.

    Caller must establish the documented binary/profile, RNE with gradual
    underflow in every participating thread, stable FP mode, finite/no-overflow
    input envelope, and GEMV alpha=1,beta=0. No such fact is inferred here.
    """
    if any(type(n) is not int or not 1 <= n <= 70 for n in (rows, columns)):
        raise ValueError('Outside the inspected shape family')
    depth, nodes = 2*columns+16, 4*columns+16
    # Per real output: <=nodes rounding sites, signed-unit error transfers,
    # depth<=depth; 2*rows bounds sqrt(2*rows) without irrational rounding.
    factor = 1/(1-depth*U)
    return KernelBounds(mv_rel=2*gamma(depth), mv_abs=2*rows*nodes*ETA*factor,
                        scale_rel=2*gamma(2), scale_abs=8*rows*ETA/(1-2*U),
                        sub_rel=U, sub_abs=2*rows*ETA,
                        dot_rel=2*gamma(depth), dot_abs=2*nodes*ETA*factor)


def selected_path_contract(profile, *, rows, columns, layout, mxcsr_values):
    """Match supplied read-only observations; never claim future mode stability."""
    if (profile != PROFILE or layout != 'C' or not mxcsr_values
            or not all(fp_mode_supported(value) for value in mxcsr_values)):
        return {'status': 'UNRESOLVED', 'bounds': None, 'scientific_execution_ready': False}
    return {'status': 'CURRENT_KERNEL_BOUND_DERIVED', 'bounds': path_bounds(rows, columns),
            'required_future_checks': ['same dispatch family and binary profile',
                'stable FP mode in all participating threads', 'finite intermediate envelope',
                'all-support signal and coordinate guards'],
            'scientific_execution_ready': False}


def finite_intermediate_envelope(*, matrix_frobenius, initial_norm, boundary_norm, scalar_modulus):
    """Sufficient screening for the documented operation graph, not H4 evidence.

    The boundary_norm must cover every state boundary, including product
    actions; a mean or only a typical endpoint norm does not qualify.
    """
    values = [Q(value) for value in (matrix_frobenius, initial_norm, boundary_norm, scalar_modulus)]
    return all(0 <= value <= 1 << 200 for value in values)


class _Error:
    """Normalized polynomial enclosure and absolute forward error, rational."""
    def __init__(self, lo, hi=None, error=0):
        self.lo, self.hi, self.error = Q(lo), Q(lo if hi is None else hi), Q(error)

    def combine(self, other, operation):
        if operation == 'add':
            lo, hi = self.lo+other.lo, self.hi+other.hi
            incoming = self.error+other.error
        elif operation == 'sub':
            lo, hi = self.lo-other.hi, self.hi-other.lo
            incoming = self.error+other.error
        elif operation == 'mul':
            products = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
            lo, hi = min(products), max(products)
            incoming = (max(abs(self.lo), abs(self.hi))*other.error
                        + max(abs(other.lo), abs(other.hi))*self.error+self.error*other.error)
        else:
            raise ValueError('Unsupported proof operation')
        # Underflow floor normalized by a or a^2 is <=U^2 in the scaled kernel.
        rounding = U*(max(abs(lo), abs(hi))+incoming)+U*U
        return _Error(lo, hi, incoming+rounding)

    def __add__(self, other): return self.combine(other, 'add')
    def __sub__(self, other): return self.combine(other, 'sub')
    def __mul__(self, other): return self.combine(other, 'mul')


def hypot_polynomial_error():
    """Exact proof budget for both non-FMA glibc correction polynomials.

    a-normalized inputs: a=1, 0<=b<=1, 1/2<=h<=2. Both exact polynomials
    equal h^2-a^2-b^2. This checks rounding propagation, not sample accuracy.
    """
    a, b, h, two, four = _Error(1), _Error(0, 1), _Error(Q(1, 2), 2), _Error(2), _Error(4)
    delta = h-b
    branch1 = a*(two*delta-a)+(delta-two*(a-b))*delta
    delta = h-a
    branch2 = two*delta*(a-two*b)+((four*delta-b)*b+delta*delta)
    return max(branch1.error, branch2.error)


def hypot_relative_proof_bound():
    error = hypot_polynomial_error()
    # Initial sqrt of a*a+b*b is within 4U of rho; residual <=9U*rho^2.
    correction = (9*U+error)*(1+U)/(2*(1-4*U))+U*U
    return 4*U+correction+U*(1+4*U+correction)+U*U


def complex_abs_error(magnitude_upper):
    """Inspected glibc non-FMA hypot; finite |z|<=2^400 and supported FP mode.

    Includes subnormal inputs/output and power-of-two scaling. <=1024U*|z|+ETA.
    This bound is not used to certify a future signal magnitude automatically.
    """
    magnitude_upper = Q(magnitude_upper)
    if not 0 <= magnitude_upper <= 1 << 400:
        raise ValueError('Outside the inspected finite hypot range')
    if hypot_relative_proof_bound() > 1024*U:
        raise ArithmeticError('Analytic proof budget exceeded')
    return 1024*U*magnitude_upper+ETA
