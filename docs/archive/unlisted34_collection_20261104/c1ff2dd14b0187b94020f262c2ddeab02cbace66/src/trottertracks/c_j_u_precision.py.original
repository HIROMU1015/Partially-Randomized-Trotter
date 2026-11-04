"""Same-array DF residual evaluation with extended accumulation.

No fit or science at import. Existing reconstruction, coefficient adjoint and
ffsim/JAX derivatives are reused. longdouble is arithmetic precision, not a
different scientific objective or a validated enclosure.
"""
from functools import lru_cache
from types import SimpleNamespace
from . import c_j_u_reduced as reduced


def require_extended():
    import numpy as np
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps or np.finfo(np.longdouble).nmant+1<64:
        raise ValueError('EXTENDED_ACCUMULATION_UNAVAILABLE_NO_FALLBACK')
    return np.longdouble


@lru_cache(None)
def _geometry(rank,norb):
    import jax
    import jax.numpy as jnp
    from ffsim.linalg.util import unitaries_from_parameters_jax
    if not jax.config.jax_enable_x64:raise ValueError('JAX_X64_REQUIRED')
    def f(eta,U):
        return U@unitaries_from_parameters_jax(eta/jnp.sqrt(2.),dim=norb,n_mats=rank,real=True)
    return jax.jit(f),jax.jit(jax.jacfwd(f,argnums=0))


def geometry(U,eta):
    import numpy as np
    f,_=_geometry(len(U),U.shape[-1])
    return np.asarray(f(eta,U))


@lru_cache(None)
def _array_jacobian(rank,norb):
    import jax
    f,_=reduced.blocks._residual_factory(rank,norb)
    # Differentiate the EXISTING residual wrt the explicit base-array U at
    # eta=0. Then pull back through the chart geometry separately. This avoids
    # calling unequal NumPy/JAX rotation points an identical derivative point.
    return jax.jit(jax.jacfwd(f,argnums=3))


def parts_gradient_J(g,J,U,spec,*,extended=True):
    """Physical array point J/U is binary64; arithmetic accumulates in LD."""
    import numpy as np
    ld=require_extended() if extended else np.float64
    for a in (g,J,U):
        if a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('FINITE_BINARY64_ARRAY_POINT')
    p=reduced.Coefficients(g,U,spec)
    j,u,target=J.astype(ld),U.astype(ld),g.astype(ld)
    residual=reduced.linear.reconstruct(j,u)-target
    tensor=np.sum(residual*residual,dtype=ld)/ld(2)
    mass=reduced.pr.smooth_mass(j,spec,np)
    l2=np.sum(j*j,dtype=ld)
    view=SimpleNamespace(U=u,n=spec.norb)
    adj=reduced.linear.FixedU.adjoint(view,residual)/p.w.astype(ld)/ld(spec.loss_scale)
    if p.is_pr:
        c=reduced.pr.coefficients(j[spec.ld:],np).ravel()
        h=np.hypot(c,ld(spec.epsilon))
        penalty=ld(spec.gamma)*(mass/ld(spec.mass_scale))**2
        adj+=2*ld(spec.gamma)*mass/ld(spec.mass_scale)**2*(p.T.astype(ld).T@(c/h))
    else:
        penalty=ld(spec.strength)*l2/ld(spec.l2_scale)
        i,k=np.triu_indices(spec.norb);fold=np.tile(np.where(i==k,1,2),spec.rank).astype(ld)
        adj+=2*ld(spec.strength)*j[:,i,k].ravel()*fold/p.w.astype(ld)/ld(spec.l2_scale)
    return dict(objective=float(tensor/ld(spec.loss_scale)+penalty),
        tensor_objective=float(tensor),normalized_tensor_objective=float(tensor/ld(spec.loss_scale)),
        penalty=float(penalty),smooth_tail_mass_Ha=float(mass),
        exact_tail_mass_Ha=float(np.sum(abs(reduced.pr.coefficients(j[spec.ld:],np)),dtype=ld)),
        full_ordered_Z_l2_Ha_squared=float(l2)),np.asarray(adj,dtype=float),residual


def evaluate(g,J,U,spec,*,base_U=None,eta=None,parts_evaluator=None):
    """Primal residual in LD; existing JAX derivative at exactly J/U arrays.

    Mathematical quantities are unchanged. This is NOT a certified oracle.
    The binary64 Jacobian's own rounding remains; decimal synthetic tests
    distinguish residual error from derivative and inner-solution error.
    """
    import numpy as np
    ld=require_extended()
    parts,gj,r=(parts_evaluator or parts_gradient_J)(g,J,U,spec)
    zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    D=np.asarray(_array_jacobian(spec.rank,spec.norb)(zero,g,J,U,spec.loss_scale)).reshape(g.size,U.size)
    normalized=r.ravel()/np.sqrt(ld(spec.loss_scale))
    array_grad=D.astype(ld).T@normalized
    _,jac=_geometry(spec.rank,spec.norb)
    local=np.asarray(jac(zero,U)).reshape(U.size,-1)
    physical=np.asarray(local.astype(ld).T@array_grad,dtype=float)
    if (base_U is None)!=(eta is None):raise ValueError('CHART_BASE_AND_ETA_REQUIRED_TOGETHER')
    chart=physical
    if base_U is not None:
        if not np.array_equal(geometry(base_U,eta),U):raise ValueError('UNEQUAL_GEOMETRY_ARRAY_POINT')
        d=np.asarray(jac(eta,base_U)).reshape(U.size,-1)
        chart=np.asarray(d.astype(ld).T@array_grad,dtype=float)
    if not all(np.isfinite(a).all() for a in (gj,physical,chart,D)) or not np.isfinite(parts['objective']):
        raise ValueError('NONFINITE_EXTENDED_EVALUATION')
    return parts,gj,physical,chart,dict(arithmetic='longdouble_primal_binary64_JAX_array_Jacobian',
        longdouble_mantissa_bits=np.finfo(ld).nmant+1,array_point='exact supplied binary64 J/U',
        rotation_pullback_separated=True,rigorous_gradient_error_bound=None)


def rc2_residual_bound(g,J,U,spec):
    """Analytic conservative bound with FLOATING inputs, not an enclosure.

    For exactly orthogonal U at exact arithmetic, mu=2 alpha/P0,
    dx<=||F_x||/mu and
    ||F_U(x)-F_U(x*)|| <= 4 dx/f*(||Ax-g||+sqrt(rank)*(||x||+dx)).
    The constant uses the full isometric physical tangent. A floating plug-in
    cannot establish a rigorous numerical bound; it can also be very loose.
    """
    import numpy as np
    if isinstance(spec,reduced.pr.Objective) or spec.strength<=0:
        raise ValueError('POSITIVE_RC2_ONLY')
    _,gj,r=parts_gradient_J(g,J,U,spec)
    mu=2*spec.strength/spec.l2_scale;dx=float(np.linalg.norm(gj))/mu
    value=4*dx/spec.loss_scale*(float(np.linalg.norm(r.ravel()))+
        np.sqrt(spec.rank)*(float(np.linalg.norm(J.ravel()))+dx))
    return dict(mu=mu,solution_error_plugin=dx,outer_gradient_error_plugin=float(value),
                exact_arithmetic_theorem=True,rigorous_numerical_enclosure=False,
                qualification='UNRESOLVED_FLOATING_RESIDUAL_AND_ARITHMETIC')
