"""Tail entry-L1-squared comparator oracle; synthetic preparation only.

Reuse the PR coefficient convention and ffsim tensor value/gradient. This is
not a molecular runner, stable reduced-oracle adapter, or paper RC-DF solver.
Real-input evaluation requires separately permitted scientific execution.
"""
from dataclasses import dataclass

from . import c_pr_aware_df as pr
from . import c_rcdf_comparator as rc2

VERSION = "tail_triangle_entry_l1_squared_matched_zero_floor_v1"


def entry_weights(norb, xp):
    """Ordered J: diagonal 5/4, off-diagonal 3/2 (twice per pair)."""
    return 1.5 - .25 * xp.eye(norb)


@dataclass(frozen=True)
class Objective:
    rank: int
    norb: int
    ld: int
    loss_scale: float
    mass_scale: float
    strength: float
    epsilon: float = 1e-8

    def __post_init__(self):
        pr.Objective(self.rank, self.norb, self.ld, self.loss_scale,
                     self.mass_scale, self.strength, self.epsilon)

    @property
    def entry_epsilon(self):
        # Same mass at all-zero J as PR's n(2n+1) coefficient slots/frame.
        # This matches the floor, NOT the smoothing functions everywhere.
        return self.epsilon * (2*self.norb+1) / (1.5*self.norb-.25)

    @property
    def smoothing_mass_bound(self):
        return (self.rank-self.ld)*self.norb*(2*self.norb+1)*self.epsilon

    def baseline(self):
        return pr.Objective(self.rank, self.norb, self.ld,
                            self.loss_scale, 1., 0., self.epsilon)


def mass(J, spec, xp, *, smooth=True):
    """Sum weights times |J|, or its positive-epsilon smooth continuation."""
    tail = J[spec.ld:]
    magnitude = xp.hypot(tail, spec.entry_epsilon) if smooth else xp.abs(tail)
    return xp.sum(entry_weights(spec.norb, xp)*magnitude)


def initial_spec(J, *, ld, loss_scale, strength, epsilon=1e-8):
    """Set the comparator's own scale once; never renormalize an endpoint."""
    import numpy as np
    J = np.asarray(J)
    if (J.dtype != np.float64 or J.ndim != 3 or J.shape[1] != J.shape[2]
            or not np.isfinite(J).all()
            or not np.array_equal(J, J.transpose(0, 2, 1))):
        raise ValueError("REAL_FINITE_SYMMETRIC_J_REQUIRED")
    provisional = Objective(len(J), J.shape[-1], ld, loss_scale,
                            1., strength, epsilon)
    return Objective(len(J), J.shape[-1], ld, loss_scale,
                     float(mass(J, provisional, np)), strength, epsilon)


def make_value_and_grad(spec, base_factory=None):
    """One existing tensor oracle plus comparator penalty AD, no optimizer."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ffsim.linalg.util import df_tensors_from_params_jax
    base = pr.make_value_and_grad(spec.baseline(), base_factory=base_factory)

    def penalty(x):
        J, _ = df_tensors_from_params_jax(
            x, spec.rank, spec.norb, None, real=True)
        return (mass(J, spec, jnp)/spec.mass_scale)**2
    penalty_vg = jax.jit(jax.value_and_grad(penalty))

    def combined(x, g):
        rc2._inputs(x, g, spec)
        value, grad = base(x, g)
        if spec.strength:
            p, dp = penalty_vg(x)
            value, grad = value+spec.strength*p, grad+spec.strength*dp
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise ValueError("NONFINITE_OBJECTIVE_OR_GRADIENT")
        return value, grad
    return combined


def mass_diagnostics(J, spec):
    """Exact cancellation gap and smoothing bounds, not resource inference."""
    import numpy as np
    tail = J[spec.ld:]
    mapped = float(np.sum(np.abs(pr.coefficients(tail, np))))
    entry = float(mass(J, spec, np, smooth=False))
    gap = float(np.sum(np.sum(np.abs(tail), axis=-1)
                       - np.abs(np.sum(tail, axis=-1))))
    return dict(exact_tail_mass=mapped, exact_entry_mass=entry,
                row_cancellation_gap=gap,
                smooth_tail_mass=float(pr.smooth_mass(J, spec, np)),
                smooth_entry_mass=float(mass(J, spec, np)),
                each_smoothing_mass_bound=spec.smoothing_mass_bound,
                entry_epsilon=spec.entry_epsilon,
                PF_error=None, RTE_error=None, native_price=None,
                total_cost=None, independent_novelty=None)
