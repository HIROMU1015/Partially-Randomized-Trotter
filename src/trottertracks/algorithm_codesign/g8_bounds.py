"""G8 rational budgets: finite-provider parameter and separate resource failure."""
from fractions import Fraction as F
from .return_aggregation import root_interval, Interval


def ceil(q): return (q.numerator + q.denominator - 1) // q.denominator


def log_interval(q, terms=96):
    q = F(q)
    if q < 1: raise ValueError('log argument >=1 required')
    power = 0
    while q >= 2: q /= 2; power += 1
    def atanh_log(z):
        lo = 2 * sum(z ** (2 * k + 1) / (2 * k + 1) for k in range(terms))
        return Interval(lo, lo + 2*z**(2*terms+1)/((2*terms+1)*(1-z*z)))
    a, b = atanh_log(F(1, 3)), atanh_log((q - 1) / (q + 1))
    return Interval(power*a.lo+b.lo, power*a.hi+b.hi)


def log_upper(q, terms=96): return log_interval(q, terms).hi


def provider_bias(m, rho, epsilon, delta):
    rho, epsilon, delta = F(rho), F(epsilon), F(delta)
    if min(rho, epsilon, delta) < 0: raise ValueError('nonnegative error parameters required')
    return 2 * (3 * rho + 3 * (1 + rho) * (2 * epsilon + (m + 1) * delta))


def delta_ceiling(m, rho, epsilon, epsilon_axis=F(1, 200)):
    return (epsilon_axis - provider_bias(m, rho, epsilon, 0)) / (6 * (1 + rho) * (m + 1))


def accepted_cap(M, z_upper, t=9, root_bits=256):
    z_upper = min(F(1), F(z_upper))
    if not 0 <= z_upper <= 1: raise ValueError('acceptance bound invalid')
    if z_upper == 1: return M
    v = M * z_upper
    return min(M, ceil(v + root_interval(2 * v * t, root_bits).hi + F(2 * t, 3)))


def acceptance_upper(generator):
    if generator.arm != 'full_return': return F(1)
    return min(F(1), (1 + generator.eta) ** (generator.m + 2) * generator.U.hi / generator.B.lo)


def budget(generator, delta=F(1, 10**6), epsilon=F(1, 10**6), epsilon_axis=F(1, 200)):
    eta, rho, m = generator.eta, generator.rho, generator.m
    kappa = (1 + rho)**2 / (1 - eta)**(m + 2)
    B = generator.B.hi
    moment = kappa * B * (generator.U.hi if generator.arm == 'full_return' else B)
    W = (1 + rho) * B / (1 - eta)**(m + 2)
    bias = provider_bias(m, rho, epsilon, delta)
    s = epsilon_axis - bias
    if s <= 0: raise ArithmeticError('finite-provider margin exhausted; no precision rescue')
    alpha_axis = F(49, 16000)
    N = ceil(log_upper(2 / alpha_axis) * (2 * moment / s**2 + 4 * W / (3 * s)))
    z = acceptance_upper(generator)
    return {'N_per_axis': N, 'm2_upper': moment, 'range_upper': W, 'bias_upper': bias,
        'remaining': s, 'delta_hypothetical': F(delta),
        'delta_strict_ceiling': delta_ceiling(m, rho, epsilon, epsilon_axis),
        'alpha_axis': alpha_axis, 'estimation_failure_16_axes': 16 * alpha_axis,
        'resource_failure_per_row': F(1, 8000), 'resource_failure_8_rows': F(1, 1000),
        'total_failure_upper': F(1, 20), 'resource_tail_t': 9,
        'acceptance_upper_non_enumerative': z,
        'accepted_call_cap_two_axes': accepted_cap(2 * N, z),
        'hard_attempt_cap_two_axes': 2 * N,
        'uses_global_B_new_or_true_signal': False,
        'provider_implementation_validated': False}


def p5_root_formula(p, x):
    p, x = tuple(map(F, p)), F(x)
    chi, mu3, mu4, mu5 = (sum(pi**k for pi in p) for k in (2, 3, 4, 5))
    a = 1 - chi*x**2/2 + (2*chi**2 - mu4)*x**4/24
    s = x - (2*chi - mu3)*x**3/6 + (5*chi**2 - 4*chi*mu3 + 2*mu5 - 2*mu4)*x**5/120
    return a, s
