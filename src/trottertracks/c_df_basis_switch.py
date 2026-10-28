"""Fixed-partition switch surrogates. Pure synthetic PoC; no fitter or I/O.

Group labels must come from the existing exact spin-lift key producer. Keeping
these labels fixed gives a J-mass objective, not a differentiable U-merging rule.
"""
from dataclasses import dataclass
import hashlib
import math
from .c_pr_aware_df import coefficients

VERSION = 'fixed_exact_basis_partition_mass_switch_poc_v1'


@dataclass(frozen=True)
class Partition:
    keys: tuple
    ld: int = 2
    r: int = 128
    epsilon: float = 1e-8

    def __post_init__(self):
        if (not isinstance(self.keys, tuple) or not self.keys
                or any(not isinstance(k, str) or not k for k in self.keys)
                or type(self.ld) is not int or self.ld < 0
                or type(self.r) is not int or self.r < 1
                or not math.isfinite(self.epsilon) or self.epsilon <= 0):
            raise ValueError('FIXED_PARTITION_POLICY')

    @property
    def groups(self):
        return tuple(dict.fromkeys(self.keys))


def exact_keys(U, ld):
    """No gauge alignment, threshold, projector equivalence or approximate merge."""
    import numpy as np
    from .c1_general_k_adapter import _matrix, _spin_lift
    from .c_df_basis_surrogate import overlap
    U = np.asarray(U)
    if (U.dtype != np.float64 or U.ndim != 3 or U.shape[1] != U.shape[2]
            or type(ld) is not int or not 0 <= ld < len(U)):
        raise ValueError('ROTATION_BATCH')
    keys = []
    for u in U[ld:]:
        overlap(u, u)
        keys.append(hashlib.sha256(repr(_spin_lift(_matrix(u, 'synthetic_U'))).encode()).hexdigest())
    return tuple(keys)


def _validate(J, spec):
    import numpy as np
    J = np.asarray(J)
    if (J.dtype != np.float64 or J.ndim != 3 or len(J) != spec.ld + len(spec.keys)
            or J.shape[1] != J.shape[2] or not J.shape[1]
            or not np.isfinite(J).all() or not np.array_equal(J, J.transpose(0, 2, 1))):
        raise ValueError('GENERAL_REAL_SYMMETRIC_J')
    return J


def exact_law(J, spec):
    """Stable normalized masses, including subnormal inputs; no epsilon law."""
    import numpy as np
    c = coefficients(_validate(J, spec)[spec.ld:], np)
    scale = float(np.max(np.abs(c)))
    if scale == 0:
        return dict(status='NO_RANDOM_TAIL',lambda_R=0.,q_basis={},q_frame={},
                    P_same=None,expected_switches=0.)
    frame = [math.fsum(abs(float(x/scale)) for x in row) for row in c]
    total = math.fsum(frame); mass = scale * total
    if not math.isfinite(mass) or mass == 0:
        raise ValueError('UNREPRESENTABLE_NONZERO_MASS')
    grouped = {k: math.fsum(frame[i] for i, x in enumerate(spec.keys) if x == k) / total
               for k in spec.groups}
    same = math.fsum(q*q for q in grouped.values())
    values=list(grouped.values())
    switching=2*math.fsum(values[i]*values[j] for i in range(len(values)) for j in range(i+1,len(values)))
    return dict(status='IDEAL_IID_DESCRIPTOR',lambda_R=mass,q_basis=grouped,
                q_frame={str(spec.ld+i):v/total for i,v in enumerate(frame)},
                P_same=same,expected_switches=(spec.r-1)*switching)


def smooth_terms(J, spec, xp):
    """AD-compatible fixed-group terms; all coefficient slots get epsilon mass.

    No data-dependent support mask. Stable rescaling avoids squaring a tiny
    coefficient directly. These probabilities are NOT the actual RTE law.
    Validation is separate so JAX tracers need not be converted to NumPy.
    """
    c = coefficients(J[spec.ld:], xp)
    scale = xp.maximum(xp.max(xp.abs(c)), spec.epsilon)
    weights = xp.hypot(c / scale, spec.epsilon / scale)
    frames = xp.sum(weights, axis=-1)
    groups = xp.stack([xp.sum(frames[xp.asarray([i for i,k in enumerate(spec.keys) if k == key])])
                       for key in spec.groups])
    total = xp.sum(groups); q = groups / total
    # Pair form avoids a cancellation-prone total**2 - sum(groups**2).
    cross = 2 * sum(groups[i]*groups[j] for i in range(len(groups)) for j in range(i+1,len(groups)))
    fraction=cross/(total*total)
    return dict(mass=scale*total, q_basis=q,
                switch_fraction=fraction,
                expected_switches=(spec.r-1)*fraction)


def exact_coefficient_gradient(c, keys, r=128):
    """dT/dc away from zeros, for the registered fixed partition only."""
    import numpy as np
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 2 or len(c) != len(keys) or not np.isfinite(c).all() or np.any(c == 0):
        raise ValueError('NONZERO_FINITE_COEFFICIENTS_FOR_CLASSICAL_GRADIENT')
    m = {k:float(np.sum(abs(c[[i for i,x in enumerate(keys) if x == k]]))) for k in set(keys)}
    total = math.fsum(m.values()); q = {k:v/total for k,v in m.items()}
    same = math.fsum(v*v for v in q.values())
    return np.asarray([-2*(r-1)*(q[k]-same)/total*np.sign(row) for k,row in zip(keys,c)])
