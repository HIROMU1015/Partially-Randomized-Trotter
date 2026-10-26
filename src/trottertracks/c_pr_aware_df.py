"""Fixed-policy PR-tail surrogate around the existing ffsim CDF objective.

Passive import; real-input evaluation/fit needs separate scientific permission.
No new optimizer, DF reconstruction, codec, sampling law or resource estimator.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import math

from . import c1_ffsim_corrected_codec as corrected

VERSION = 'cdf_fixed_prefix_smooth_pauli_tail_mass_v1'
INITIALIZATION = 'qualified_real_schur_snapshot_x0_v2'


@dataclass(frozen=True)
class Objective:
    rank: int
    norb: int
    ld: int
    loss_scale: float
    mass_scale: float
    gamma: float
    epsilon: float = 1e-8

    def __post_init__(self):
        if (type(self.rank) is not int or self.rank < 1 or type(self.norb) is not int
                or self.norb < 1 or type(self.ld) is not int or not 0 <= self.ld < self.rank):
            raise ValueError('FIXED_NONEMPTY_ORIGINAL_FRAME_TAIL_REQUIRED')
        for key in ('loss_scale', 'mass_scale', 'epsilon'):
            x = getattr(self, key)
            if not math.isfinite(x) or x <= 0: raise ValueError('POSITIVE_FINITE_' + key)
        if not math.isfinite(self.gamma) or self.gamma < 0: raise ValueError('INVALID_GAMMA')


def coefficients(J, xp):
    """Differentiable linear map matching the existing general-K convention.

K_ab = J_(a//2,b//2)/2; c_Za=-sum_b K_ab/2; c_ZaZb=K_ab/2.
Identity is excluded from lambda_R and remains an exact deterministic phase.
All supports, including exact zeros, enter the smooth surrogate.
"""
    n = J.shape[-1]; modes = 2 * n
    K = xp.repeat(xp.repeat(J / 2, 2, axis=-2), 2, axis=-1)
    z = -xp.sum(K, axis=-1) / 2
    rows, cols = xp.triu_indices(modes, 1)
    return xp.concatenate((z, K[..., rows, cols] / 2), axis=-1)


def smooth_mass(J, spec, xp):
    c = coefficients(J[spec.ld:], xp)
    return xp.sum(xp.sqrt(c * c + spec.epsilon * spec.epsilon))


def make_value_and_grad(spec, base_factory=None):
    """Reuse ffsim's tensor loss/analytic AD and add only the new penalty AD."""
    import ffsim
    import jax
    import jax.numpy as jnp
    if not jax.config.jax_enable_x64: raise ValueError('JAX_X64_REQUIRED')
    d = importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    util = importlib.import_module('ffsim.linalg.util')
    factory = base_factory or getattr(d._make_compressed_value_and_grad,
        '_c_pr_base', d._make_compressed_value_and_grad)
    base = factory(spec.rank, spec.norb, None)
    if spec.gamma == 0:
        def baseline(x, g):
            value, grad = base(x, g)
            return value / spec.loss_scale, grad / spec.loss_scale
        return baseline

    def penalty(x):
        J, _ = util.df_tensors_from_params_jax(x, spec.rank, spec.norb, None, real=True)
        return (smooth_mass(J, spec, jnp) / spec.mass_scale) ** 2
    penalty_vg = jax.jit(jax.value_and_grad(penalty))

    def combined(x, g):
        value, grad = base(x, g)
        p, dp = penalty_vg(x)
        return value / spec.loss_scale + spec.gamma * p, grad / spec.loss_scale + spec.gamma * dp
    return combined


def prepare(g, J, U, *, ld, epsilon=1e-8):
    """One shared faithful initial state/scaling for the future paired fits."""
    a, u, x, report, _ = corrected.codec.roundtrip(J, U)
    return a, u, x, dict(scales(g,a,x,ld=ld,epsilon=epsilon),codec=report)


def scales(g, J, x, *, ld, epsilon=1e-8):
    import numpy as np
    import jax.numpy as jnp
    provisional = Objective(len(J), J.shape[-1], ld, 1., 1., 0., epsilon)
    value, grad = make_value_and_grad(provisional)(jnp.asarray(x), jnp.asarray(g))
    mass = float(smooth_mass(J, provisional, np))
    if not np.isfinite(grad).all() or not math.isfinite(float(value)) or mass <= 0:
        raise ValueError('NONFINITE_INITIAL_OBJECTIVE_OR_EMPTY_TAIL')
    spec = Objective(len(J), J.shape[-1], ld, max(float(value), 1e-14), mass, 0., epsilon)
    return dict(spec=asdict(spec),
        initial_tensor_objective=float(value), initial_smooth_tail_mass=mass)


def components(g, x, spec):
    """Diagnostics only: proxy is never an attenuation/CA/total-cost verdict."""
    import numpy as np
    import jax.numpy as jnp
    from ffsim.linalg.util import df_tensors_from_params
    a, _ = df_tensors_from_params(np.asarray(x), spec.rank, spec.norb, None, real=True)
    base_spec = Objective(**{**asdict(spec), 'gamma': 0.})
    normalized, _ = make_value_and_grad(base_spec)(jnp.asarray(x), jnp.asarray(g))
    c = coefficients(a[spec.ld:], np)
    mass = float(np.sum(np.abs(c))); sm = float(smooth_mass(a, spec, np))
    return dict(tensor_objective=float(normalized) * spec.loss_scale,
        normalized_tensor_objective=float(normalized), exact_tail_mass_Ha=mass,
        smooth_tail_mass_Ha=sm, normalized_penalty=(sm / spec.mass_scale) ** 2,
        combined_objective=float(normalized) + spec.gamma * (sm / spec.mass_scale) ** 2,
        proxy=VERSION, attenuation=None, PF_error=None, RTE_error=None,
        synthesis_error=None, shots=None, total_cost=None, CA_achieved=None,
        independent_novelty=None)


@contextmanager
def objective_context(spec):
    """Dedicated process only; restore the old factory even on failure."""
    d = importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    original = d._make_compressed_value_and_grad
    def factory(rank, norb, indices):
        if (rank, norb, indices) != (spec.rank, spec.norb, None):
            raise ValueError('PR_OBJECTIVE_STRUCTURE_CHANGED')
        return make_value_and_grad(spec, base_factory=original)
    factory._c_pr_base = original
    try:
        d._make_compressed_value_and_grad = factory
        yield
    finally:
        d._make_compressed_value_and_grad = original


def fit(g, J, U, x0, spec, *, before_optimizer, after_optimizer):
    """Existing fitter/loss hook; forward the single qualified initial x0."""
    if (len(J), J.shape[-1]) != (spec.rank, spec.norb): raise ValueError('FIT_RANK_SHAPE')
    import numpy as np
    with objective_context(spec):
        a,u,result = corrected.fit(g, J, U, x0, before_optimizer=before_optimizer,
                                  after_optimizer=after_optimizer,prepared_initial=True)
    if (a.shape != (spec.rank,spec.norb,spec.norb) or u.shape != a.shape
            or not np.isfinite(a).all() or not np.isfinite(u).all()
            or not np.array_equal(a,a.transpose(0,2,1))
            or np.max(abs(u.transpose(0,2,1)@u-np.eye(spec.norb))) > 1e-10):
        raise ValueError('FINAL_DF_STRUCTURE_OR_ORTHOGONALITY')
    return a,u,result
