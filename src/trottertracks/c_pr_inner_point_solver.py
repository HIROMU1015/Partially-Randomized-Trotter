"""Opt-in PR Newton-CG and reduced-oracle connection, no molecular I/O.

Frozen legacy routines hard-code their chart/inner call targets. These small
orchestration adapters retain the same SciPy solver, pre-step jac semantics,
physical gates and latest-pair admission without patching any old module.
"""
import time
from . import c_pr_coefficient_point as point
from . import c_j_u_stable as stable

DEFAULT=stable.DEFAULT
VERSION='PR_Newton_CG_same_decoded_point_v1'


def solve_j(p,J,policy=DEFAULT,*,solver=None,record=lambda *a:None,parts_evaluator=None):
    import numpy as np
    from scipy.optimize import minimize
    if not p.is_pr and not getattr(p,'is_entry',False):raise ValueError('OPT_IN_PR_ONLY_NO_RC2_CHANGE_OR_EXPLICIT_ENTRY_BACKEND')
    tick,cpu=time.monotonic(),time.process_time();chart=point.CoefficientChart(p,J,policy)
    evaluations,hessians=0,0;trace=[];accepted=[np.zeros(len(chart.x0))]
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
    zero=np.zeros(len(chart.x0));before,g0,_=chart.independent(zero)
    res=(solver or minimize)(vg,zero,jac=True,hess=hess,method='Newton-CG',callback=callback,
        options=dict(xtol=policy.j_xtol,maxiter=policy.j_maxiter))
    z=np.asarray(res.x)
    if z.shape!=zero.shape or z.dtype!=np.float64 or not np.isfinite(z).all():
        record(dict(method=VERSION,**stable.old._raw(res),failure='NONFINITE_OR_MALFORMED_J_ENDPOINT'),None)
        raise ValueError('NONFINITE_OR_MALFORMED_J_ENDPOINT')
    x,j=chart.point(z)
    info=dict(method=VERSION,**stable.old._raw(res),metric_singular_values=chart.singular.tolist(),
        metric_directions_dropped=0,objective_regularization_added=False,iteration_trace=trace,
        initial_objective=before,initial_J_gradient_infinity=float(np.max(abs(g0))))
    # Preserve raw jac AND the points defining its permitted pre-step meaning.
    record(dict(info),dict(x0=chart.x0,x=x,J=j,z=z,solver_jac=np.asarray(res.jac),
        coordinate_transform=chart.W,gradient0=g0,last_accepted_z=accepted[-1],last_step_start_z=accepted[0]))
    fv,gz=vg(z);value,gx,pullback=chart.independent(z)
    checks={}
    for name,q in [('last_step_start',accepted[0]),('last_accepted',accepted[-1])]:
        _,reference=vg(q);checks[name]=point.compare(res.jac,reference,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
    checks['raw_jac_at_endpoint']=point.compare(res.jac,gz,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
    checks['chart_vs_physical_gradient']=point.compare(pullback,gz,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
    checks['chart_vs_physical_value']=point.compare(value,fv,atol=policy.value_atol,rtol=policy.value_rtol)
    checks['raw_vs_endpoint_value']=point.compare(info['fun'],fv,atol=policy.value_atol,rtol=policy.value_rtol) if info['fun'] is not None else dict(PASS=False)
    # Same decoded arrays for the inherited binary64 observation, not another
    # coefficient pack/unpack. Its J norm still participates in stationarity.
    binary,binary_g,_=(parts_evaluator or stable.precision.parts_gradient_J)(p.g,j,p.mapping.U,p.spec,extended=False)
    checks['binary64_same_array_value']=point.compare(value,binary['objective'],atol=policy.value_atol,rtol=policy.value_rtol)
    info.update(consistency=checks,raw_jac_semantics='SciPy1.14.1 may return pre-step jac;raw preserved;endpoint J gradient independently evaluated',
        rigorous_gradient_error_bound=None,thresholds_changed=False)
    record(dict(info,consistency_record=True),None)
    if (np.asarray(res.jac).dtype!=np.float64 or
        not any(checks[k]['PASS'] for k in ('last_step_start','last_accepted','raw_jac_at_endpoint')) or
        not all(checks[k]['PASS'] for k in ('chart_vs_physical_gradient','chart_vs_physical_value','raw_vs_endpoint_value','binary64_same_array_value'))):
        raise ValueError('PR_INNER_POINT_CONSISTENCY_MISMATCH')
    info.update(independent_objective=value,
        stationarity=stable.old.gradient_gate(max(float(np.max(abs(gx))),float(np.max(abs(binary_g)))),policy.j_gradient_infinity,policy),
        extended_J_gradient_infinity=float(np.max(abs(gx))),binary64_J_gradient_infinity=float(np.max(abs(binary_g))),
        progress=stable.old.nonincrease(before,value,policy),value_gradient_calls_in_block=evaluations,hessian_calls_in_block=hessians,
        independent_adjoint_calls_in_block=2,binary64_same_array_calls_in_block=1,
        wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu)
    record(dict(info,inner_summary_record=True),None)
    return j,info


class PRReducedOracle(stable.StableReducedOracle):
    """Frozen reduced bookkeeping with only its three PR solve calls replaced.

    Explicit opt-in class; old optimize()/paired runners still use old defaults.
    FD/line-search tests use this class through the existing SciPy interface.
    """
    def __init__(self,g,J,U,spec,policy=DEFAULT,*,backend=None,**kwargs):
        if backend is None:
            if not isinstance(spec,stable.pr.Objective):raise ValueError('OPT_IN_PR_ONLY_NO_RC2_CHANGE')
        else:backend.validate(spec)
        self.backend=backend
        super().__init__(g,J,U,spec,policy,**kwargs)

    def _state(self,*args,**kwargs):
        return (self.backend.state if self.backend else stable.state)(*args,**kwargs)

    def _solve(self,*args,**kwargs):
        return (self.backend.solve_j if self.backend else solve_j)(*args,**kwargs)

    def evaluate(self,eta):
        import numpy as np
        eta=np.asarray(eta);s,pcy=self.spec,self.policy
        nrot=s.rank*s.norb*(s.norb-1)//2
        if eta.shape!=(nrot,) or eta.dtype!=np.float64 or not np.isfinite(eta).all():raise ValueError('FINITE_BINARY64_ETA_REQUIRED')
        key=eta.tobytes()
        if key in self.cache:result=self.cache[key]
        else:
            if len(self.points)>=pcy.max_points:raise stable.Unqualified('OUTER_LIMIT_UNQUALIFIED')
            label=str(len(self.points))
            def raw(step):return lambda data,arrays:self.record('inner_raw',label+'_'+step,data,arrays)
            def initial():
                V=stable.precision.geometry(self.U0,eta)
                if np.max(abs(V.transpose(0,2,1)@V-np.eye(s.norb)))>pcy.orthogonality:raise ValueError('U_ORTHOGONALITY_FAIL_NO_PROJECTION')
                p=(self.backend.Coefficients if self.backend else stable.Coefficients)(self.g,V,s);j,info=self._solve(p,self.J0,pcy,record=raw('solve'))
                return V,p,j,info
            V,p,J0,i0=self.call('inner_J_solve',label+'_solve',initial)
            J1,i1=self.call('inner_J_solve',label+'_refine1',lambda:self._solve(p,J0,pcy,record=raw('refine1')))
            a,j1,u1,c1,binary1=self.call('precision_state',label+'_refine1',lambda:self._state(self.g,J1,V,s,pcy,base_U=self.U0,eta=eta))
            self.record('precision_state',label+'_refine1',a,dict(J=J1,U=V,eta=eta,gradient_J=j1,physical_gradient_U=u1,chart_gradient=c1,binary64_gradient_U=binary1))
            J2,i2=self.call('inner_J_solve',label+'_refine2',lambda:self._solve(p,J1,pcy,record=raw('refine2')))
            b,j2,u2,c2,binary2=self.call('precision_state',label+'_refine2',lambda:self._state(self.g,J2,V,s,pcy,base_U=self.U0,eta=eta))
            self.record('precision_state',label+'_refine2',b,dict(J=J2,U=V,eta=eta,gradient_J=j2,physical_gradient_U=u2,chart_gradient=c2,binary64_gradient_U=binary2))
            self.record('inner_summary',label,dict(solves=[i0,i1,i2]),None)
            def admit():
                structure=(self.backend.applicability if self.backend else stable.previous.applicability)(p,pcy)
                parts0=p.parts(p.pack(J0),independent=True)
                _,coarse=stable.old.u_value_gradient(np.zeros(nrot),self.g,J0,V,s,parts0['penalty'])
                shift=max(float(np.linalg.norm(u2-u1)),float(np.linalg.norm(c2-c1)))
                status=('J_UNQUALIFIED' if b['J_stationarity']['status']!='MET' else 'REDUCED_GRADIENT_UNQUALIFIED'
                    if not structure['admitted'] or shift>pcy.refinement_U_l2 else 'NUMERICAL_PILOT_ADMITTED')
                return dict(status=status,applicability=structure,latest_refinement_U_shift_l2=shift,refinement_U_shift_l2=shift,
                    refinement_U_allowance=pcy.refinement_U_l2,legacy_first_pair_physical_shift_observation=float(np.linalg.norm(binary1-coarse)),
                    legacy_result_not_reclassified=True,full_numeric_rank=True,
                    objective_refinement=stable.old.nonincrease(a['components']['objective'],b['components']['objective'],pcy),
                    RC2_analytic_bound_floating_plugin=None,rigorous_gradient_error_bound=None,qualification_level='LATEST_PAIR_NUMERICAL_PILOT_NOT_ERROR_ENCLOSURE',inner_solver=VERSION)
            valid=self.call('reduced_admission',label,admit)
            result=dict(label=label,J=J2,U=V,eta=eta.copy(),state=b,gradient=c2,value=b['components']['objective'],qualification=valid,inner=[i0,i1,i2])
            self.points.append(result);self.cache[key]=result;self.record('envelope',label,valid,dict(eta=eta,gradient=c2))
        if result['qualification']['status']!='NUMERICAL_PILOT_ADMITTED':raise stable.Unqualified(result['qualification']['status'])
        return result
