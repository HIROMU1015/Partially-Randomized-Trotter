"""Process-local solver options and common post-fit engineering diagnostics.

Reuse existing RC2/PR objectives, faithful x0 codec and actual ffsim fitter.
No rescaling, bounds, gauge changes, optimizer or loss implementation here.
"""
from contextlib import contextmanager

VERSION = "common_native_gradient_identity_tangent_v1"
OPTIONS = dict(ftol=1e-15,gtol=1e-6,maxcor=10,maxls=20,
               maxiter=15000,maxfun=15000)
CRITERIA = dict(parameter_gradient_infinity=1e-5,
    identity_tangent_frobenius=1e-5,atol=1e-8,rtol=1e-7,
    value_atol=1e-10,value_rtol=1e-8)


@contextmanager
def solver_options(options):
    """Last-hop options only. Existing corrected bridge still validates x0.

    Dedicated process; old bridge captures this wrapper as original_minimize.
    None reproduces the old options unchanged for synthetic regression only.
    """
    import scipy.optimize
    original = scipy.optimize.minimize
    if getattr(original,"_common_options",False) is True:
        raise ValueError("DEDICATED_UNPATCHED_PROCESS_REQUIRED")
    chosen = None if options is None else dict(options)
    seen = 0
    def forward(fun,x0,**kwargs):
        nonlocal seen
        seen += 1
        if seen != 1 or kwargs != dict(method="L-BFGS-B",jac=True,
                                        callback=None,options=None):
            raise ValueError("COMMON_SOLVER_DISPATCH_CHANGED")
        return original(fun,x0,**{**kwargs,"options":chosen})
    forward._common_options = True
    try:
        scipy.optimize.minimize = forward
        yield
        if seen != 1: raise ValueError("EXACTLY_ONE_OPTIMIZER_INVOCATION_REQUIRED")
    finally:
        scipy.optimize.minimize = original


def fit(module,g,J,U,x0,spec,*,before_optimizer,after_optimizer,options=OPTIONS):
    with solver_options(options):
        return module.fit(g,J,U,x0,spec,before_optimizer=before_optimizer,
                          after_optimizer=after_optimizer)


def first_order(gradient,rank,norb,criteria=CRITERIA):
    """Native-chart norm plus a Frobenius-normalized exact tensor-null space.

    Convert independent upper-J derivative to symmetric-matrix derivative
    (off-diagonal half); its inner product with 11^T/n is sum(upper grad)/n.
    Project frame coefficients to sum-zero. No J/U move is applied.
    """
    import numpy as np
    g = np.asarray(gradient)
    if (g.dtype != np.float64 or g.shape != (rank*norb*norb,)
            or not np.isfinite(g).all()):
        raise ValueError("FINITE_NATIVE_GRADIENT_REQUIRED")
    split = rank*norb*(norb-1)//2
    upper = g[split:].reshape(rank,norb*(norb+1)//2)
    coefficients = upper.sum(axis=1)/norb
    projected = coefficients-coefficients.mean()
    raw = float(np.max(abs(g)))
    tangent = float(np.linalg.norm(projected))
    def gate(size,threshold):
        margin = criteria["atol"]+criteria["rtol"]*max(size,threshold)
        return dict(value=size,threshold=threshold,engineering_margin=margin,
            status="MET" if size+margin<=threshold else
                   "NOT_MET" if size-margin>threshold else "UNRESOLVED_MARGIN")
    a = gate(raw,criteria["parameter_gradient_infinity"])
    b = gate(tangent,criteria["identity_tangent_frobenius"])
    return dict(parameter_gradient=a,projected_parameter_gradient=a.copy(),
        bounds=None,identity_tangent=b,identity_tangent_coefficients=projected.tolist(),
        U_gradient_l2=float(np.linalg.norm(g[:split])),
        J_upper_gradient_l2=float(np.linalg.norm(g[split:])),
        gradient_l2=float(np.linalg.norm(g)),
        convergence_criteria_met=a["status"]==b["status"]=="MET",
        rigorous_stationarity=False,all_tensor_null_directions_qualified=False)


def terminal(value,gradient,result,rank,norb,criteria=CRITERIA):
    """Independent objective evaluation vs raw result; success is separate."""
    import numpy as np
    grad, raw = np.asarray(gradient), np.asarray(result.jac)
    for x in (grad,raw,np.asarray(result.x)):
        if x.dtype!=np.float64 or x.shape!=(rank*norb*norb,) or not np.isfinite(x).all():
            raise ValueError("FINITE_TERMINAL_SCHEMA_REQUIRED")
    if (not np.isfinite(value) or not np.isfinite(result.fun)
            or abs(float(value)-float(result.fun)) > criteria["value_atol"]
                +criteria["value_rtol"]*abs(float(result.fun))
            or not np.allclose(grad,raw,atol=criteria["atol"],rtol=criteria["rtol"])):
        raise ValueError("INDEPENDENT_TERMINAL_VALUE_GRADIENT_MISMATCH")
    assessment = first_order(grad,rank,norb,criteria)
    assessment.update(objective=float(value),solver_success=bool(result.success),
        solver_status=int(result.status),solver_message=str(result.message),
        independent_gradient_maxabs=float(np.max(abs(grad-raw))),
        independent_value_abs=float(abs(value-result.fun)),
        qualified=bool(result.success) and assessment["convergence_criteria_met"])
    return assessment
