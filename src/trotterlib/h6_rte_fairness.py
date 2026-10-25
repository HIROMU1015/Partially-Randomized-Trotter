"""H6 study connections; preserve the existing sufficient shot inequality."""
from fractions import Fraction as Q
import math
from types import SimpleNamespace
from . import h4_conditional_cng as A


def integer_shot(axis, rho):
    """Extend only representational SAFE_INTEGER_LIMIT, never the guard.

    The original evaluator already computes an outward rational upper bound.
    Python's integer ceil is exact. Verifying N*h^2 >= A_upper proves the same
    sufficient condition for any returned integer; this is not a new shot law.
    """
    old = A.shot(axis, rho)
    if old['reason'] != 'SAFE_INTEGER_LIMIT':
        return old
    h = Q(axis['s_exact']) - Q(axis['u_enc_exact']) - Q(rho)
    upper = Q(old['bound_upper_exact'])
    lower = Q(axis['scale']['A_lower']) / h**2
    n = max(1, math.ceil(upper))
    assert h > 0 and (Q(axis['u_enc_exact'])+Q(rho))/h <= A.G
    assert n*h*h >= Q(axis['scale']['A_upper'])
    return dict(N=n, reason='CONDITIONAL_ONLY_ARBITRARY_INTEGER',
                previous_reason='SAFE_INTEGER_LIMIT', rho_exact=str(Q(rho)),
                headroom_exact=str(h), u_over_h_exact=str((Q(axis['u_enc_exact'])+Q(rho))/h),
                bound_lower_exact=str(lower), bound_upper_exact=str(upper),
                ceiling_interval_width=math.ceil(upper)-math.ceil(lower),
                sufficient_inequality_verified=True)


# Reuse the complete original account function with only its shot primitive
# replaced in a private namespace. Do not monkeypatch the historical module.
account = A.extract('src/trotterlib/h4_pauli_b0_comparison.py', 'account',
    dict(Q=Q, AXES=A.AXES,
         A=SimpleNamespace(scale_bounds=A.scale_bounds, base_axis=A.base_axis, shot=integer_shot)))


def scale(B):
    return A.scale_bounds(Q(B), A.trans_bounds('ln', Q(B))[1])
