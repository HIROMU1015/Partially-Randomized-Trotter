"""Curvature-aware separated DF blocks. Synthetic validation/science by scope.

Original objectives, physical stationarity and DF gates come from the frozen
v1 implementation. Only PR's coefficient solver and the fixed-J orbital
subproblem change. No argmin differentiation or hidden joint update.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import time
from . import c_separated_j_u as old
from .c_j_u_consistency_repair import compare

linear, pr, rc2 = old.linear, old.pr, old.rc2
Coefficients, state, retract = old.Coefficients, old.state, old.retract
VERSION = 'alternating_whitened_J_U_point_consistency_v3'


@dataclass(frozen=True)
class Policy:
    max_cycles: int = 64
    j_maxiter: int = 1000
    j_xtol: float = 1e-12
    metric_singular_floor: float = 1.
    u_max_nfev: int = 25
    u_solver_gtol: float = 1e-12
    u_xtol: float = 1e-14
    j_gradient_infinity: float = 1e-5
    u_tangent_l2: float = 1e-5
    gradient_atol: float = 1e-8
    gradient_rtol: float = 1e-7
    value_atol: float = 1e-10
    value_rtol: float = 1e-8
    stall_atol: float = 1e-12
    stall_rtol: float = 1e-10
    stall_cycles: int = 3
    orthogonality: float = 1e-10

    def __post_init__(self):
        import math
        for k,v in asdict(self).items():
            if k in ('max_cycles','j_maxiter','u_max_nfev','stall_cycles'):
                if type(v) is not int or v < 1: raise ValueError('POSITIVE_INTEGER_POLICY:'+k)
            elif not math.isfinite(v) or v <= 0: raise ValueError('POSITIVE_FINITE_POLICY:'+k)


DEFAULT = Policy()


def metric(B, floor=1.):
    """Invertible full-direction right coordinate transform; no ridge/truncation."""
    import numpy as np
    from scipy.linalg import svd
    if B.ndim != 2 or not np.isfinite(B).all() or not np.isfinite(floor) or floor <= 0:
        raise ValueError('FINITE_METRIC_REQUIRED')
    _,s,Vh = svd(B, full_matrices=B.shape[0] < B.shape[1],lapack_driver='gesdd')
    s = np.pad(s,(0,B.shape[1]-len(s)))
    W = Vh.T/np.maximum(s,floor)
    if not np.isfinite(W).all(): raise ValueError('NONFINITE_METRIC_NO_FALLBACK')
    return W,s


class CoefficientChart:
    """x=x0+Wz. Affine residual reuse avoids repeated Ax-g cancellation.

    Hessian is formed from AW/TW, not W.T Q W, and includes exact smooth-mass
    curvature. This is the SAME mathematical objective in invertible coordinates.
    Independent original-coordinate contraction remains the admission criterion.
    """
    def __init__(self,p,J,policy=DEFAULT):
        import numpy as np
        self.p,self.x0 = p,p.pack(J)
        B=np.vstack((p.A/np.sqrt(p.spec.loss_scale),
                     np.sqrt(2*p.spec.gamma)*p.T/p.spec.mass_scale))
        self.W,self.singular=metric(B,policy.metric_singular_floor)
        self.A,self.T=p.A@self.W,p.T@self.W
        self.r0,self.c0=p.A@self.x0-p.g.ravel(),p.T@self.x0
        self.Q=self.A.T@self.A/p.spec.loss_scale

    def vg(self,z):
        import numpy as np
        p=self.p;res=self.r0+self.A@z;c=self.c0+self.T@z
        h=np.hypot(c,p.spec.epsilon);mass=float(np.sum(h))
        value=.5*float(res@res)/p.spec.loss_scale+p.spec.gamma*(mass/p.spec.mass_scale)**2
        grad=self.A.T@res/p.spec.loss_scale+2*p.spec.gamma*mass/p.spec.mass_scale**2*(self.T.T@(c/h))
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise ValueError('NONFINITE_J_VALUE_GRADIENT')
        return value,grad

    def hess(self,z):
        import numpy as np
        p=self.p;c=self.c0+self.T@z;h=np.hypot(c,p.spec.epsilon);v=self.T.T@(c/h)
        H=self.Q+2*p.spec.gamma/p.spec.mass_scale**2*(np.outer(v,v)+np.sum(h)*(
            self.T.T@(((p.spec.epsilon/h)**2/h)[:,None]*self.T)))
        if not np.isfinite(H).all(): raise ValueError('NONFINITE_J_HESSIAN')
        return H


def solve_j(p,J,policy=DEFAULT,*,solver=None,record=lambda *a:None):
    import numpy as np
    from scipy.optimize import minimize
    if not p.is_pr:
        # Exactly the existing augmented gelsd problem/solver, never normal equations.
        return old.solve_j(p,J,policy,solver=solver,record=record)
    total,cpu=time.monotonic(),time.process_time();chart=CoefficientChart(p,J,policy)
    evaluations,hessians=0,0;trace=[]
    accepted=[np.zeros(len(chart.x0))]
    def vg(z):
        nonlocal evaluations
        evaluations+=1;return chart.vg(z)
    def hess(z):
        nonlocal hessians
        hessians+=1;return chart.hess(z)
    def callback(intermediate_result):
        trace.append(dict(objective=float(intermediate_result.fun),z_norm=float(np.linalg.norm(intermediate_result.x))))
        accepted.append(np.array(intermediate_result.x,copy=True))
        if len(accepted)>2:accepted.pop(0)
    zero=np.zeros(len(chart.x0));before,gradient0=p.vg(chart.x0,independent=True)
    res=(solver or minimize)(vg,zero,jac=True,hess=hess,method='Newton-CG',
        callback=callback,options=dict(xtol=policy.j_xtol,maxiter=policy.j_maxiter))
    z=np.asarray(res.x);x=chart.x0+chart.W@z
    info=dict(method='affine_residual_whitened_exact_Hessian_Newton_CG',**old._raw(res),
        metric_singular_values=chart.singular.tolist(),metric_directions_dropped=0,
        objective_regularization_added=False,iteration_trace=trace,
        wall_seconds=time.monotonic()-total,CPU_seconds=time.process_time()-cpu,
        initial_objective=before,initial_J_gradient_infinity=float(np.max(abs(gradient0))))
    record(dict(info),dict(x0=chart.x0,x=x,z=z,solver_jac=np.asarray(res.jac),
                          coordinate_transform=chart.W,gradient0=gradient0,
                          last_accepted_z=accepted[-1],last_step_start_z=accepted[0]))
    if z.shape!=zero.shape or z.dtype!=np.float64 or not np.isfinite(x).all():
        raise ValueError('NONFINITE_OR_MALFORMED_J_ENDPOINT')
    value,g=p.vg(x,independent=True);fv,gz=vg(z)
    refs=[(name,point) for name,point in [('last_step_start',accepted[0]),('last_accepted',accepted[-1])]]
    checks={}
    for name,point in refs:
        _,reference=vg(point)
        checks[name]=compare(res.jac,reference,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
    checks['raw_jac_at_endpoint']=compare(res.jac,gz,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
    checks['chart_vs_physical_gradient']=compare(chart.W.T@g,gz,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
    checks['chart_vs_physical_value']=compare(fv,value,atol=policy.value_atol,rtol=policy.value_rtol)
    checks['raw_vs_endpoint_value']=compare(info['fun'],fv,atol=policy.value_atol,rtol=policy.value_rtol) if info['fun'] is not None else dict(PASS=False)
    info['consistency']=checks
    info['raw_jac_semantics']='SciPy1.14.1 Newton-CG gfk may be pre-step;raw jac preserved;physical stationarity uses fresh endpoint evaluation'
    record(dict(info,consistency_record=True),None)
    if (np.asarray(res.jac).dtype!=np.float64 or
        not (checks['last_step_start']['PASS'] or checks['last_accepted']['PASS'] or checks['raw_jac_at_endpoint']['PASS']) or
        not all(checks[k]['PASS'] for k in ['chart_vs_physical_gradient','chart_vs_physical_value','raw_vs_endpoint_value'])):
        raise ValueError('J_POINT_CONSISTENCY_MISMATCH')
    info.update(independent_objective=value,
        stationarity=old.gradient_gate(float(np.max(abs(g))),policy.j_gradient_infinity,policy),
        progress=old.nonincrease(before,value,policy),value_gradient_calls_in_block=evaluations,
        hessian_calls_in_block=hessians)
    return p.unpack(x),info


@lru_cache(None)
def _residual_factory(rank,norb):
    import jax
    import jax.numpy as jnp
    from ffsim.linalg.util import unitaries_from_parameters_jax
    if not jax.config.jax_enable_x64:raise ValueError('JAX_X64_REQUIRED')
    def residual(eta,g,J,U,scale):
        V=U@unitaries_from_parameters_jax(eta/jnp.sqrt(2.),dim=norb,n_mats=rank,real=True)
        tensor=jnp.einsum('kpi,kqi,kij,krj,ksj->pqrs',V,V,J,V,V,optimize=True)
        return ((tensor-g)/jnp.sqrt(scale)).ravel()
    return jax.jit(residual),jax.jit(jax.jacfwd(residual))


def residual_jacobian(eta,g,J,U,spec):
    import numpy as np
    fun,jac=_residual_factory(spec.rank,spec.norb)
    a,b=np.asarray(fun(eta,g,J,U,spec.loss_scale)),np.asarray(jac(eta,g,J,U,spec.loss_scale))
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('NONFINITE_U_RESIDUAL_JACOBIAN')
    return a,b


def solve_u(g,J,U,spec,policy=DEFAULT,*,solver=None,record=lambda *a:None):
    import numpy as np
    from scipy.optimize import least_squares
    total,cpu=time.monotonic(),time.process_time();p=Coefficients(g,U,spec)
    parts=p.parts(p.pack(J),independent=True);zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    r0,D0=residual_jacobian(zero,g,J,U,spec);W,singular=metric(D0,policy.metric_singular_floor)
    value0,gradient0=old.u_value_gradient(zero,g,J,U,spec,parts['penalty'])
    if not np.allclose(D0.T@r0,gradient0,atol=policy.gradient_atol,rtol=policy.gradient_rtol):
        raise ValueError('U_RESIDUAL_INDEPENDENT_GRADIENT_MISMATCH')
    fun,jac=_residual_factory(spec.rank,spec.norb);nf,nj=1,1;trial=[]
    def f(z):
        nonlocal nf
        nf+=1;eta=W@z;a=np.asarray(fun(eta,g,J,U,spec.loss_scale))
        if not np.isfinite(a).all():raise ValueError('NONFINITE_U_RESIDUAL')
        trial.append(dict(residual_cost=.5*float(a@a),eta_norm=float(np.linalg.norm(eta)),
            directional_linear_change=float(gradient0@eta)))
        return a
    def derivative(z):
        nonlocal nj
        nj+=1;a=np.asarray(jac(W@z,g,J,U,spec.loss_scale))@W
        if not np.isfinite(a).all():raise ValueError('NONFINITE_U_JACOBIAN')
        return a
    res=(solver or least_squares)(f,zero,jac=derivative,method='trf',tr_solver='exact',
        loss='linear',ftol=None,xtol=policy.u_xtol,gtol=policy.u_solver_gtol,
        max_nfev=policy.u_max_nfev)
    z=np.asarray(res.x);eta=W@z
    info=dict(method='fixed_J_residual_TRF_exact_SVD_metric',success=bool(res.success),
        status=int(res.status),message=str(res.message),fun=float(res.cost)+parts['penalty'] if np.isfinite(res.cost) else None,
        residual_cost=float(res.cost) if np.isfinite(res.cost) else None,raw_cost_repr=repr(res.cost),nfev=int(res.nfev),njev=int(res.njev or 0),nit=None,
        initial_objective=parts['objective'],initial_U_tangent_l2=float(np.linalg.norm(gradient0)),
        metric_singular_values=singular.tolist(),metric_directions_dropped=0,
        trial_evaluations=trial,trial_points_are_not_accepted_iteration_history=True,
        wall_seconds=time.monotonic()-total,CPU_seconds=time.process_time()-cpu)
    record(dict(info),dict(eta0=zero,eta=eta,z=z,solver_grad=np.asarray(res.grad),
                          coordinate_transform=W,gradient0=gradient0))
    if z.shape!=zero.shape or z.dtype!=np.float64 or not np.isfinite(eta).all():
        raise ValueError('NONFINITE_OR_MALFORMED_U_ENDPOINT')
    V=retract(U,eta)
    if not np.isfinite(V).all() or np.max(abs(V.transpose(0,2,1)@V-np.eye(spec.norb)))>policy.orthogonality:
        raise ValueError('U_ORTHOGONALITY_FAIL_NO_PROJECTION')
    r,D=residual_jacobian(eta,g,J,U,spec);nf+=1;nj+=1
    value,gradient=old.u_value_gradient(eta,g,J,U,spec,parts['penalty'])
    after=Coefficients(g,V,spec).parts(p.pack(J),independent=True)['objective']
    if (info['fun'] is None or not np.isfinite(after) or abs(after-value)>policy.value_atol+policy.value_rtol*abs(value)
        or abs(info['fun']-value)>policy.value_atol+policy.value_rtol*abs(value)
        or not np.allclose(W.T@(D.T@r),res.grad,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
        or not np.allclose(D.T@r,gradient,atol=policy.gradient_atol,rtol=policy.gradient_rtol)):
        raise ValueError('U_RETRACTION_OR_GRADIENT_MISMATCH')
    _,physical=old.u_value_gradient(zero,g,J,V,spec,parts['penalty'])
    info.update(progress=old.nonincrease(parts['objective'],after,policy),
        independent_objective=after,chart_gradient_l2=float(np.linalg.norm(gradient)),
        final_U_tangent_l2_at_fixed_J=float(np.linalg.norm(physical)),
        accepted_endpoint_eta_norm=float(np.linalg.norm(eta)),residual_calls_in_block=nf,
        residual_jacobian_calls_in_block=nj,independent_value_gradient_calls=3,
        total_block_wall_seconds=time.monotonic()-total,total_block_CPU_seconds=time.process_time()-cpu)
    return V,info


def classification(s):
    j=s['J_stationarity']['status']=='MET';u=s['U_stationarity']['status']=='MET'
    return ('BOTH_FIRST_ORDER_MET' if j and u else 'J_UNQUALIFIED' if u else
            'U_UNQUALIFIED' if j else 'BOTH_UNQUALIFIED')


def optimize(g,J,U,spec,policy=DEFAULT,*,call=lambda kind,label,fn:fn(),record=lambda *a:None):
    """Same sequential blocks as v1; no extra cycles, search or retry.

    SciPy TRF status0 means a registered finite evaluation limit, unlike v1's
    L-BFGS-B status1. Every U endpoint is followed by mandatory J refit before
    physical stationarity, stall or finite solver-stop classification.
    """
    J,U=J.copy(),U.copy();states,blocks=[],[]
    def assess(label):
        row,gj,gu=call('state_diagnostic',label,lambda:state(g,J,U,spec,policy))
        row['classification']=classification(row)
        states.append(dict(label=label,**row))
        record('state',label,row,dict(J=J,U=U,gradient_J=gj,gradient_U=gu));return row
    def coefficient(label):
        def block():
            tick,cpu=time.monotonic(),time.process_time()
            v,info=solve_j(Coefficients(g,U,spec),J,policy,
                record=lambda data,arrays:record('J_consistency' if arrays is None else 'J_raw',label,data,arrays))
            info.update(total_block_wall_seconds=time.monotonic()-tick,total_block_CPU_seconds=time.process_time()-cpu)
            return v,info
        v,info=call('J_solve',label,block);blocks.append(dict(block='J',label=label,**info))
        record('J',label,info,dict(J=v,U=U));return v
    assess('initial');J=coefficient('0');current=assess('0')
    status,completed,stalls='OUTER_LIMIT_UNQUALIFIED',0,0
    if current['J_stationarity']['status']!='MET':status='J_STATIONARITY_UNQUALIFIED'
    elif current['first_order_met']:status='FIRST_ORDER_MET'
    else:
        for cycle in range(1,policy.max_cycles+1):
            label=str(cycle);before=current['components']['objective']
            U,info=call('U_solve',label,lambda:solve_u(g,J,U,spec,policy,
                record=lambda data,arrays:record('U_raw',label,data,arrays)))
            blocks.append(dict(block='U',label=label,**info));record('U',label,info,dict(J=J,U=U))
            J=coefficient(label);current=assess(label);completed=cycle
            progress=old.nonincrease(before,current['components']['objective'],policy)
            current['cycle_progress']=progress
            current['U_gradient_before_J_refit']=info['final_U_tangent_l2_at_fixed_J']
            stalls=stalls+1 if abs(progress['decrease'])<=policy.stall_atol+policy.stall_rtol*abs(before) else 0
            if current['J_stationarity']['status']!='MET':status='J_STATIONARITY_UNQUALIFIED';break
            if current['first_order_met']:status='FIRST_ORDER_MET';break
            if not info['success'] and info['status']!=0:status='U_SOLVER_STOP_UNQUALIFIED';break
            if stalls>=policy.stall_cycles:status='STALLED_NOT_STATIONARY';break
    return J,U,dict(status=status,completed_cycles=completed,states=states,blocks=blocks,
        terminal=current,first_order_qualified=current['first_order_met'],
        classification=classification(current),policy=asdict(policy),
        solver_flags=[dict(block=b['block'],label=b['label'],success=b['success'],status=b['status']) for b in blocks],
        retry=0,resume=0,rigorous_stationarity=False)
