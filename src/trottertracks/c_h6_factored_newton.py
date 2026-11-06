"""Opt-in fixed-U Newton search using a factored Hessian, no molecular I/O.

The objective, epsilon and coefficient chart are supplied unchanged. QR avoids
forming normal equations. All columns remain; there is no ridge or truncation.
Final qualification belongs to the existing independent physical/binary gates.
"""
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import OptimizeResult
from scipy.optimize._linesearch import scalar_search_armijo

VERSION = 'H6_factored_Newton_QR_Armijo_diagnostic_v2_safe_failure'


def factor(chart, z):
    """D with D.T@D equal to the original analytic Hessian."""
    _, _, residual, c, h, mass, _ = chart._primal(z)
    s = chart.p.spec
    v = chart.T.T @ (c/h)
    gradient = (chart.A.T @ residual.ravel()/s.loss_scale
        + 2*chart.strength*mass/s.mass_scale**2*v)
    curvature = (np.asarray(chart.epsilon, dtype=chart.ld)/h)**2/h
    D = np.vstack((chart.A/np.sqrt(np.longdouble(s.loss_scale)),
        np.sqrt(np.longdouble(2*chart.strength))/s.mass_scale*v[None, :],
        np.sqrt(2*np.longdouble(chart.strength)*mass)/s.mass_scale
            * np.sqrt(curvature)[:, None]*chart.T))
    rhs = -np.hstack((residual.ravel()/np.sqrt(np.longdouble(s.loss_scale)),
        np.atleast_1d(np.sqrt(np.longdouble(2*chart.strength))*mass/s.mass_scale),
        np.zeros(len(c), dtype=np.longdouble)))
    return D, gradient, rhs


def direction(D, rhs, refinements=2):
    """Column equilibrated full QR, with extended residual refinement."""
    scales = np.sqrt(np.sum(D*D, axis=0, dtype=np.longdouble))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('NONPOSITIVE_NEWTON_COLUMN_SCALE')
    E = D/scales
    Q, R = qr(np.asarray(E, float), mode='economic')
    Q = Q.astype(np.longdouble)
    if np.any(np.diag(R) == 0):
        raise ValueError('FACTORED_NEWTON_SINGULAR_WITHOUT_TRUNCATION')
    def inverse(b):
        return solve_triangular(R, np.asarray(Q.T@b, float))/scales
    delta = inverse(rhs).astype(np.longdouble)
    for _ in range(refinements):
        delta += inverse(rhs-D@delta)
    residual = D.T@(D@delta-rhs)
    return np.asarray(delta, float), dict(
        linear_residual_infinity=float(np.max(abs(residual))),
        linear_rhs_infinity=float(np.max(abs(D.T@rhs))),
        minimum_QR_diagonal=float(np.min(abs(np.diag(R)))),
        directions_dropped=0, objective_regularization_added=False)


def solve(chart, *, maxiter=1000, gtol=1e-13, record=lambda *a: None):
    """One unchanged fixed-U block; records accepted and rejected progress.

    A failed line search or exhausted iterations is returned as such. A small
    coordinate gradient never substitutes for physical endpoint qualification.
    """
    z = np.zeros(chart.dimension)
    trace = []
    evaluations = 0
    def value(point):
        nonlocal evaluations
        evaluations += 1
        return chart.centered(point)[0]
    current = value(z)
    status, message = 1, 'Maximum number of iterations reached.'
    line_search_error = None
    for iteration in range(maxiter):
        D, gradient, rhs = factor(chart, z)
        if np.linalg.norm(np.asarray(gradient, float)) < gtol:
            status, message = 0, 'Coordinate gradient tolerance reached; physical gates still required.'
            break
        step, linear = direction(D, rhs)
        slope = float(gradient @ step)
        if not np.isfinite(step).all() or not np.isfinite(slope) or slope >= 0:
            status, message = 2, 'Factored Newton direction is not a finite descent direction.'
            break
        try:
            alpha, proposed = scalar_search_armijo(
                lambda a: value(z+a*step), current, slope, alpha0=1.)
        except ZeroDivisionError as exc:
            # SciPy1.14.1's cubic interpolation divides by zero for a flat
            # rounded objective with a nonzero supplied derivative. Preserve
            # the last accepted point and expose search failure to the gates.
            line_search_error = repr(exc)
            status, message = 2, 'Armijo interpolation failed (ZeroDivisionError); last accepted point retained.'
            break
        if alpha is None or alpha <= 0 or not np.isfinite(proposed):
            status, message = 2, 'Armijo search failed.'
            break
        displacement = alpha*step
        next_z = z+displacement
        # Avoid hundreds of numerically identical iterations. This is a failed
        # search, never evidence that the physical stationarity gate passed.
        if np.array_equal(chart.point(next_z)[1], chart.point(z)[1]):
            status, message = 2, 'Binary64 coefficient point did not change; independent physical gates still required.'
            break
        predicted = -float(gradient@displacement + .5*np.sum((D@displacement)**2))
        actual = current-proposed
        z = next_z
        current = proposed
        row = dict(iteration=iteration, alpha=float(alpha),
            step_l2=float(np.linalg.norm(displacement)), predicted=predicted,
            actual=float(actual), rho=float(actual/predicted) if predicted else None,
            coordinate_gradient_l2=float(np.linalg.norm(gradient)), **linear)
        trace.append(row)
        record(row, z)
    endpoint = chart.vg(z)[1]
    return OptimizeResult(x=z, fun=current, jac=endpoint, nit=len(trace),
        nfev=evaluations, nhev=len(trace), status=status, success=status==0,
        message=message, iteration_trace=trace, method=VERSION,
        line_search_error=line_search_error)
