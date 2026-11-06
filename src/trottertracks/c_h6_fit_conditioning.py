"""H6 opt-in conditioning repair. Existing objectives and acceptance gates.

Known head gauges are fixed to the starting representative, not penalized.
All nongauge coefficient directions are retained. SVD rank evidence is checked
against an independently assembled longdouble map before the expensive solve.
"""
from functools import partial
from types import SimpleNamespace
import time

from . import c_near_stable as near
from . import c_pr_inner_point_solver as inner

VERSION='H6_known_gauge_bounded_metric_centered_trust_exact_v2'


def fro(a):
    import numpy as np
    return float(np.sqrt(np.sum(a*a,dtype=np.longdouble)))


def physical_map(p):
    import numpy as np
    if not hasattr(p,'_h6_A'):
        view=SimpleNamespace(U=p.mapping.U.astype(np.longdouble),n=p.spec.norb)
        p._h6_A=inner.stable.linear.FixedU.matrix(view)/p.w.astype(np.longdouble)
    return p._h6_A


def head_report(p,policy=inner.DEFAULT):
    import numpy as np
    from scipy.linalg import svd
    if hasattr(p,'_h6_head_report'):return p._h6_head_report
    s=p.spec;d=s.norb*(s.norb+1)//2;columns=s.ld*d
    H=physical_map(p)[:,:columns];u,v,vh=svd(np.asarray(H,float),full_matrices=False)
    expected=columns-max(0,s.ld-1)
    ul,vl=u.astype(np.longdouble),vh.astype(np.longdouble)
    residual=fro(H-(ul*v.astype(np.longdouble))@vl)
    orth_u=fro(ul.T@ul-np.eye(len(v),dtype=np.longdouble))
    orth_v=fro(vl@vl.T-np.eye(len(v),dtype=np.longdouble))
    largest=float(v[0]) if len(v) else 0.
    # A posteriori SVD backward error + orthogonality errors, evaluated in LD.
    # The original rank safety multipliers remain256/1024. This reduces the
    # arithmetic uncertainty, not the required separation from that uncertainty.
    model_guard=policy.rank_guard_factor*np.finfo(np.longdouble).eps*max(H.shape)*max(1.,largest)
    backward=residual+largest*(orth_u+orth_v)
    guard=max(float(model_guard),backward)
    T=p.T[:,columns:];ts=svd(T,compute_uv=False)
    tail=not T.shape[1] or (len(ts)==T.shape[1] and ts[-1]>policy.rank_gap_factor*guard)
    smallest=float(v[expected-1]) if expected else None
    null=float(np.max(v[expected:],initial=0.))
    admitted=bool(tail and (not expected or smallest>policy.rank_gap_factor*guard) and null<=guard)
    old=inner.stable.previous.applicability(p,policy)
    report=dict(old,admitted=admitted,rank_guard=guard,
        minimum_quotient_singular=smallest,extra_null_singular_max=null,
        tail_injective=bool(tail),condition='verified_extended_head_SVD_same_known_gauges',
        original_float64_report=old,head_singular_values=v.tolist(),
        SVD_reconstruction_Frobenius=residual,U_orthogonality_Frobenius=orth_u,
        V_orthogonality_Frobenius=orth_v,model_guard=float(model_guard),
        observed_backward_error=backward,rank_gap_factor=policy.rank_gap_factor,
        arithmetic_mantissa_bits=np.finfo(np.longdouble).nmant+1,
        rigorous_numerical_enclosure=False)
    p._h6_head_report=report;return report


def metric(p,policy):
    import numpy as np
    from scipy.linalg import qr,svd
    if hasattr(p,'_h6_metric'):return p._h6_metric
    s=p.spec;ncoef=len(p.w);gauges=[]
    for frame in range(1,s.ld):
        j=np.zeros_like(p.mapping.U);j[0]=1.;j[frame]=-1.
        gauges.append(p.pack(j))
    if gauges:
        full=qr(np.stack(gauges,axis=1),mode='full')[0]
        N=full[:,:len(gauges)];Q=full[:,len(gauges):]
    else:N=np.empty((ncoef,0));Q=np.eye(ncoef)
    A=physical_map(p);T=p.T.astype(np.longdouble)
    gauge_error=max(fro(A@N.astype(np.longdouble)),fro(T@N.astype(np.longdouble)))
    if gauge_error>1e-10:raise ValueError('KNOWN_GAUGE_MAP_RESIDUAL')
    strength=s.strength if getattr(p,'is_entry',False) else s.gamma
    B=np.vstack((A/np.sqrt(np.longdouble(s.loss_scale)),
        np.sqrt(np.longdouble(2*strength))*T/np.longdouble(s.mass_scale)))
    _,singular,vh=svd(np.asarray(B@Q.astype(np.longdouble),float),full_matrices=False)
    if not len(singular) or not np.isfinite(singular).all() or singular[-1]<=0:
        raise ValueError('NONGAUGE_METRIC_NOT_INVERTIBLE')
    # Do not invert near-null singular values without bound. Every nongauge
    # direction remains present with a positive scale; no truncated SVD or
    # objective regularization is introduced. Bound the inverse metric's
    # condition by1/sqrt(binary64 eps), the coordinate precision of SciPy.
    floor=np.longdouble(singular[0])*np.sqrt(np.longdouble(np.finfo(float).eps))
    scales=np.maximum(singular.astype(np.longdouble),floor)
    W=(Q.astype(np.longdouble)@vh.T.astype(np.longdouble))/scales
    if not np.isfinite(W).all():raise ValueError('NONFINITE_NONGAUGE_METRIC')
    p._h6_metric=SimpleNamespace(W=W,singular=singular,known_gauges=len(gauges),
        gauge_map_residual=gauge_error,scaling_floor=float(floor),
        scaling_condition_limit=float(1/np.sqrt(np.finfo(float).eps)),
        limited_directions=int(np.count_nonzero(singular<floor)))
    return p._h6_metric


class Chart(inner.point.CoefficientChart):
    def __init__(self,p,J,policy):
        import numpy as np
        m=metric(p,policy)
        super().__init__(p,J,policy,original=SimpleNamespace(p=p,x0=p.pack(J),W=m.W,singular=m.singular))
        self.dimension=self.W.shape[1]
        self._primal_key=None;self._primal_value=None
        self.T64=np.asarray(self.T,float);self.Q64=np.asarray(self.Q,float)
        self.origin_primal=self._primal(np.zeros(self.dimension))
        # Anchor the centered difference at the actual decoded point. pack /
        # unpack can move an offdiagonal coefficient by one binary64 ulp.
        self.origin_J=self.origin_primal[1].astype(self.ld)

    def _primal(self,z):
        key=z.tobytes()
        if getattr(self,'_primal_key',None)!=key:
            value=super()._primal(z)
            self._primal_key=key;self._primal_value=value
        return self._primal_value

    def point(self,z):
        import numpy as np
        if z.dtype!=np.float64 or z.shape!=(self.W.shape[1],) or not np.isfinite(z).all():
            raise ValueError('FINITE_QUOTIENT_CHART_POINT')
        x=np.asarray(self.x0.astype(self.ld)+self.W.astype(self.ld)@z.astype(self.ld),dtype=float)
        return x,self.p.unpack(x)

    def centered(self,z):
        import numpy as np
        _,J,_,c,h,mass,_=self._primal(z);p=self.p
        _,_,r0,c0,h0,m0,_=self.origin_primal
        # Stable difference at the SAME actual decoded binary64 J. Original
        # value/gradient checks below still use the independent full primal.
        dr=inner.stable.linear.reconstruct(J.astype(self.ld)-self.origin_J,self.U)
        delta=np.sum(dr*(2*r0+dr),dtype=self.ld)/(2*self.ld(p.spec.loss_scale))
        dm=np.sum((c-c0)*(c+c0)/(h+h0),dtype=self.ld)
        delta+=self.ld(self.strength)*dm*(2*m0+dm)/self.ld(p.spec.mass_scale)**2
        return float(delta),self.vg(z)[1]

    def hess(self,z):
        """Same analytic search Hessian via BLAS; final gates remain LD."""
        import numpy as np
        _,_,_,c,h,mass,_=self._primal(z);p=self.p
        v=np.asarray(self.T.T@(c/h),float)
        curvature=np.asarray((np.asarray(self.epsilon,dtype=self.ld)/h)**2/h,float)
        H=self.Q64+2*self.strength/p.spec.mass_scale**2*(
            np.outer(v,v)+float(mass)*(self.T64.T@(curvature[:,None]*self.T64)))
        if not np.isfinite(H).all():raise ValueError('NONFINITE_SEARCH_HESSIAN')
        return (H+H.T)/2


def solve_j(p,J,policy=inner.DEFAULT,*,record=lambda *a:None,parts_evaluator=None):
    import numpy as np
    from scipy.optimize import minimize
    if parts_evaluator is None:
        parts_evaluator=near.parts_gradient_J if getattr(p,'is_entry',False) else inner.stable.precision.parts_gradient_J
    tick,cpu=time.monotonic(),time.process_time();chart=Chart(p,J,policy)
    zero=np.zeros(chart.dimension);before,g0,_=chart.independent(zero)
    trace=[];counts=dict(vg=0,hess=0)
    def vg(z):counts['vg']+=1;return chart.centered(z)
    def hess(z):counts['hess']+=1;return chart.hess(z)
    def callback(intermediate_result):
        trace.append(dict(objective=float(before+intermediate_result.fun),z_norm=float(np.linalg.norm(intermediate_result.x))))
    raw=minimize(vg,zero,jac=True,hess=hess,method='trust-exact',callback=callback,
        options=dict(maxiter=policy.j_maxiter,gtol=min(policy.j_xtol,1e-13)))
    z=np.asarray(raw.x)
    if z.shape!=zero.shape or z.dtype!=np.float64 or not np.isfinite(z).all():
        raise ValueError('NONFINITE_OR_MALFORMED_J_ENDPOINT')
    x,j=chart.point(z);fv,gz=chart.vg(z);value,gx,pullback=chart.independent(z)
    binary,binary_g,_=parts_evaluator(p.g,j,p.mapping.U,p.spec,extended=False)
    compare=inner.point.compare
    checks=dict(chart_vs_physical_gradient=compare(pullback,gz,atol=policy.gradient_atol,rtol=policy.gradient_rtol),
        chart_vs_physical_value=compare(value,fv,atol=policy.value_atol,rtol=policy.value_rtol),
        binary64_same_array_value=compare(value,binary['objective'],atol=policy.value_atol,rtol=policy.value_rtol),
        centered_vs_full_value=compare(before+raw.fun,fv,atol=policy.value_atol,rtol=policy.value_rtol),
        raw_jac_at_endpoint=compare(raw.jac,gz,atol=policy.gradient_atol,rtol=policy.gradient_rtol))
    info=dict(method=VERSION,**{**inner.stable.old._raw(raw),'fun':fv},
        raw_centered_fun=float(raw.fun),centering_origin=before,
        metric_singular_values=chart.singular.tolist(),metric_directions_dropped=0,
        known_gauge_coordinates_fixed=p._h6_metric.known_gauges,
        nongauge_directions_dropped=0,objective_regularization_added=False,
        gauge_map_residual=p._h6_metric.gauge_map_residual,
        metric_scaling_floor=p._h6_metric.scaling_floor,
        metric_condition_limit=p._h6_metric.scaling_condition_limit,
        metric_condition_limited_directions=p._h6_metric.limited_directions,
        search_Hessian_arithmetic='BLAS binary64 assembly; same analytic formula; acceptance uses independent LD and binary64 gates',
        iteration_trace=trace,initial_objective=before,initial_J_gradient_infinity=float(np.max(abs(g0))),
        consistency=checks,raw_jac_available=True,gradient_qualification_route='RETURNED_ENDPOINT_JAC',
        objective_unchanged=True,thresholds_changed=False,
        rigorous_gradient_error_bound=None)
    arrays=dict(x0=chart.x0,x=x,J=j,z=z,coordinate_transform=chart.W,solver_jac=np.asarray(raw.jac))
    record(dict(info),arrays)
    if not all(v['PASS'] for v in checks.values()):raise ValueError('CONDITIONED_INNER_POINT_CONSISTENCY_MISMATCH')
    info.update(independent_objective=value,
        stationarity=inner.stable.old.gradient_gate(max(float(np.max(abs(gx))),float(np.max(abs(binary_g)))),policy.j_gradient_infinity,policy),
        extended_J_gradient_infinity=float(np.max(abs(gx))),binary64_J_gradient_infinity=float(np.max(abs(binary_g))),
        progress=inner.stable.old.nonincrease(before,value,policy),
        value_gradient_calls_in_block=counts['vg'],hessian_calls_in_block=counts['hess'],
        independent_adjoint_calls_in_block=2,binary64_same_array_calls_in_block=1,
        wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu)
    record(dict(info,inner_summary_record=True),None)
    return j,info


def backend(spec):
    entry=type(spec) is near.near.Objective
    if type(spec) not in (near.near.Objective,near.near.pr.Objective):raise ValueError('REGISTERED_PR_ENTRY_OBJECTIVE')
    parts=near.parts_gradient_J if entry else inner.stable.precision.parts_gradient_J
    state=near.state if entry else inner.stable.state
    def coefficients(g,U,s):
        return (near.Coefficients if entry else inner.stable.Coefficients)(g,U,s)
    def validate(s):
        if type(s) is not type(spec) or s!=spec:raise ValueError('CONDITIONED_OBJECTIVE_BINDING')
    return SimpleNamespace(validate=validate,Coefficients=coefficients,state=state,
        solve_j=partial(solve_j,parts_evaluator=parts),applicability=head_report,
        early_applicability=head_report)


def trajectory(g,J,U,spec,physical,**kwargs):
    from collections import Counter
    b=backend(spec)
    original_call=kwargs.pop('call',lambda k,l,f:f());original_record=kwargs.pop('record',lambda *a:None)
    counts=Counter();solves=Counter()
    def call(k,l,f):counts[k]+=1;return original_call(k,l,f)
    def record(k,l,data,arrays):
        if k=='inner_raw' and data.get('inner_summary_record'):
            solves['all']+=1
            if l.endswith(('_refine1','_refine2')):solves['refinements']+=1
        return original_record(k,l,data,arrays)
    result=near.control.trajectory(g,J,U,spec,physical,call=call,record=record,
        oracle_factory=partial(near.guarded.Oracle,backend=b),
        terminal_assessor=partial(near.control.small.assess,state_evaluator=b.state),**kwargs)
    result.update(inner_solves=solves['all'],refinement_solves=solves['refinements'],
        coefficient_setup_and_solve_macro_calls=counts['inner_J_solve'],numerical_backend=VERSION)
    return result


def recipe_cell(*args,**kwargs):
    from .c_integrated_robustness import recipe_cell as original
    return original(*args,trajectory=trajectory,**kwargs)
