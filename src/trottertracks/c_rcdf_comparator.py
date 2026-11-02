"""Uniform full-Z L2 objective, with the existing ffsim solver and real codec.

Z equals the spatial general-J tensor (not the spin-lifted K=J/2).
This compares objectives under a common optimizer; it is not the paper's
alternating RC-DF implementation. Passive import, no molecular evaluation.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math

from . import c_pr_aware_df as pr

VERSION = "rcdf_full_ordered_Z_l2_initial_value_matched_v1"


@dataclass(frozen=True)
class Objective:
    rank: int
    norb: int
    ld: int
    loss_scale: float
    l2_scale: float
    strength: float = 0.1
    epsilon: float = 1e-8

    def __post_init__(self):
        # Keep the existing tail convention for diagnostics, not the L2 penalty.
        pr.Objective(self.rank, self.norb, self.ld, self.loss_scale, 1.,
                     self.strength, self.epsilon)
        if not math.isfinite(self.l2_scale) or self.l2_scale <= 0:
            raise ValueError("POSITIVE_FINITE_INITIAL_L2_REQUIRED")

    def baseline(self):
        return pr.Objective(self.rank, self.norb, self.ld, self.loss_scale,
                            1., 0., self.epsilon)


def full_l2(J, xp):
    """Sum all ordered spatial indices: diag once, off-diagonal twice."""
    return xp.sum(J * J)


def initial_spec(J, *, ld, loss_scale, strength=0.1, epsilon=1e-8):
    import numpy as np
    J = np.asarray(J)
    if (J.dtype != np.float64 or J.ndim != 3 or J.shape[1] != J.shape[2]
            or not np.isfinite(J).all()
            or not np.array_equal(J, J.transpose(0, 2, 1))):
        raise ValueError("REAL_FINITE_SYMMETRIC_J_REQUIRED")
    return Objective(len(J), J.shape[-1], ld, loss_scale,
                     float(full_l2(J, np)), strength, epsilon)


def _inputs(x, g, spec):
    import numpy as np
    x, g = np.asarray(x), np.asarray(g)
    if (x.dtype != np.float64 or x.shape != (spec.rank * spec.norb**2,)
            or g.dtype != np.float64 or g.shape != (spec.norb,) * 4
            or not np.isfinite(x).all() or not np.isfinite(g).all()):
        raise ValueError("REAL_FINITE_PARAMETER_AND_TARGET_SHAPE_REQUIRED")


def make_value_and_grad(spec, base_factory=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ffsim.linalg.util import df_tensors_from_params_jax
    base = pr.make_value_and_grad(spec.baseline(), base_factory=base_factory)

    def penalty(x):
        J, _ = df_tensors_from_params_jax(
            x, spec.rank, spec.norb, None, real=True)
        return full_l2(J, jnp) / spec.l2_scale
    penalty_vg = jax.jit(jax.value_and_grad(penalty))

    def combined(x, g):
        _inputs(x, g, spec)
        value, grad = base(x, g)
        if spec.strength:
            p, dp = penalty_vg(x)
            value, grad = value + spec.strength * p, grad + spec.strength * dp
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise ValueError("NONFINITE_OBJECTIVE_OR_GRADIENT")
        return value, grad
    return combined


def components(g, x, spec):
    import numpy as np
    from ffsim.linalg.util import df_tensors_from_params
    _inputs(x, g, spec)
    # The existing baseline hook unwraps _c_pr_base, even inside our context.
    result = pr.components(g, x, spec.baseline())
    J, _ = df_tensors_from_params(
        np.asarray(x), spec.rank, spec.norb, None, real=True)
    p = float(full_l2(J, np))
    result.update(proxy=VERSION, full_ordered_Z_l2_Ha_squared=p,
                  initial_Z_l2_Ha_squared=spec.l2_scale,
                  normalized_penalty=p/spec.l2_scale,
                  raw_rho=spec.strength*spec.loss_scale/spec.l2_scale,
                  combined_objective=result["normalized_tensor_objective"]
                  + spec.strength*p/spec.l2_scale)
    return result


@contextmanager
def objective_context(spec):
    d = importlib.import_module("ffsim.linalg.double_factorized_decomposition")
    original = d._make_compressed_value_and_grad
    if hasattr(original, "_c_pr_base"):
        raise ValueError("DEDICATED_UNPATCHED_FIT_PROCESS_REQUIRED")

    def factory(rank, norb, indices):
        if (rank, norb, indices) != (spec.rank, spec.norb, None):
            raise ValueError("RCDF_OBJECTIVE_STRUCTURE_CHANGED")
        return make_value_and_grad(spec, base_factory=original)
    factory._c_pr_base = original
    try:
        d._make_compressed_value_and_grad = factory
        yield
    finally:
        d._make_compressed_value_and_grad = original


def fit(g, J, U, x0, spec, *, before_optimizer, after_optimizer):
    """Only a penalty hook; the unchanged corrected bridge owns the one fit."""
    _inputs(x0, g, spec)
    import numpy as np
    if J.shape != (spec.rank, spec.norb, spec.norb) or U.shape != J.shape:
        raise ValueError("FIT_RANK_SHAPE")
    with objective_context(spec):
        result = pr.corrected.fit(
            g, J, U, x0, prepared_initial=True,
            before_optimizer=before_optimizer, after_optimizer=after_optimizer)
    a, u, _ = result
    if (a.shape != J.shape or u.shape != U.shape
            or not np.isfinite(a).all() or not np.isfinite(u).all()
            or not np.array_equal(a, a.transpose(0, 2, 1))
            or np.max(abs(u.transpose(0, 2, 1) @ u
                          - np.eye(spec.norb))) > pr.corrected.codec.ATOL):
        raise ValueError("FINAL_DF_STRUCTURE_OR_ORTHOGONALITY")
    return result
