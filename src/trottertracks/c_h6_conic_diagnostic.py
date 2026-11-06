"""Experimental exact epigraph of the existing fixed-U convex objective.

Search model only. Every returned J still needs the original independently
evaluated objective/gradient and binary64 gates. No recipe/runtime integration.
"""
import numpy as np
from scipy import sparse

VERSION = 'H6_fixed_U_exact_smooth_mass_SOC_diagnostic_v1'


def problem(chart):
    """min .5||y||² + gamma*m², m=sum(t), M0*t_i>=hypot(c_i,eps_i)."""
    zero=np.zeros(chart.dimension)
    _,_,r,c,_,_,_=chart._primal(zero)
    s=chart.p.spec;k=chart.dimension;nr=r.size;nc=len(c);nv=k+nr+nc+1
    A=sparse.csc_matrix(np.asarray(chart.A/np.sqrt(np.longdouble(s.loss_scale)),float))
    T=sparse.csc_matrix(np.asarray(chart.T/s.mass_scale,float))
    # Variables [z, normalized residual y, normalized epigraph t, total m].
    eq=sparse.hstack((A,-sparse.eye(nr),sparse.csc_matrix((nr,nc+1))))
    mass=sparse.csc_matrix((np.r_[-np.ones(nc),1.],
        (np.zeros(nc+1,dtype=int),np.r_[np.arange(k+nr,k+nr+nc),nv-1])),shape=(1,nv))
    row=np.arange(nc)
    epigraph=sparse.csc_matrix((-np.ones(nc),(3*row,k+nr+row)),shape=(3*nc,nv))
    coo=T.tocoo()
    epigraph+=sparse.csc_matrix((-coo.data,(3*coo.row+1,coo.col)),shape=(3*nc,nv))
    b_soc=np.zeros(3*nc);b_soc[1::3]=np.asarray(c/s.mass_scale,float)
    b_soc[2::3]=np.broadcast_to(np.asarray(chart.epsilon,float),c.shape)/s.mass_scale
    matrix=sparse.vstack((eq,mass,epigraph)).tocsc()
    b=np.r_[-np.asarray(r.ravel()/np.sqrt(np.longdouble(s.loss_scale)),float),0.,b_soc]
    P=sparse.diags(np.r_[np.zeros(k),np.ones(nr),np.zeros(nc),2*chart.strength],format='csc')
    return P,np.zeros(nv),matrix,b,(k,nr,nc)


def solve(chart, *, maxiter=1000):
    import clarabel
    P,q,A,b,(k,nr,nc)=problem(chart)
    cones=[clarabel.ZeroConeT(nr+1)]+[clarabel.SecondOrderConeT(3) for _ in range(nc)]
    settings=clarabel.DefaultSettings()
    settings.verbose=False;settings.max_iter=maxiter;settings.max_threads=1
    settings.tol_gap_abs=1e-12;settings.tol_gap_rel=1e-12;settings.tol_feas=1e-12
    settings.iterative_refinement_reltol=1e-13
    settings.iterative_refinement_abstol=1e-13
    settings.iterative_refinement_max_iter=10
    solver=clarabel.DefaultSolver(P,q,A,b,cones,settings)
    result=solver.solve()
    point=np.asarray(result.x,float)
    return point[:k],dict(method=VERSION,solver='clarabel',version=clarabel.__version__,
        status=str(result.status),iterations=int(result.iterations),
        objective=float(result.obj_val),dual_objective=float(result.obj_val_dual),
        primal_residual=float(result.r_prim),dual_residual=float(result.r_dual),
        solve_seconds=float(result.solve_time),variables=len(point),
        constraints=A.shape[0],coefficient_directions_dropped=0,
        epsilon_unchanged=True,objective_regularization_added=False)


def solve_and_polish(chart, *, maxiter=1000, record=lambda *a:None,
                     record_endpoint=lambda *a:None, record_conic=lambda *a:None):
    """Use the cone solution as a start; retain the original physical gates.

    Conic gap tolerances alone do not resolve an epsilon=1e-8 derivative.
    The same analytic Newton search polishes the unaltered primal objective.
    record_conic observes a copy of the cone coordinates before J decoding.
    """
    from . import c_h6_fit_conditioning as fit
    from . import c_h6_factored_newton as newton
    z,meta=solve(chart,maxiter=maxiter)
    record_conic(dict(meta), z.copy())
    start=chart.point(z)[1]
    local=fit.Chart(chart.p,start,chart.policy)
    remaining=maxiter-meta['iterations']
    if remaining<=0:
        return start,dict(conic=meta,polish=None)
    raw=newton.solve(local,maxiter=remaining,record=record)
    J=local.point(raw.x)[1]
    value,gx,pullback=local.independent(raw.x)
    checks=dict(raw_jac_at_endpoint=fit.inner.point.compare(raw.jac,pullback,
        atol=chart.policy.gradient_atol,rtol=chart.policy.gradient_rtol),
        centered_value=fit.inner.point.compare(raw.fun+float(local.origin_primal[-1]),value,
        atol=chart.policy.value_atol,rtol=chart.policy.value_rtol))
    info=dict(conic=meta,polish=dict(status=int(raw.status),
        message=str(raw.message),iterations=int(raw.nit),
        objective=float(local.vg(raw.x)[0]),
        physical_gradient_infinity=float(np.max(abs(gx))),consistency=checks),
        total_iterations=meta['iterations']+int(raw.nit))
    record_endpoint(info,dict(starting_J=start,z=raw.x,solver_jac=raw.jac,J=J))
    return J,info
