"""Numerical-pilot reduced oracle: extended residuals, latest refinement pair.

Original objectives/inner solvers/orthogonal chart/physical thresholds remain.
The old coarse-to-refined shift is retained as an observation; the final TWO
refined points are assessed independently. No error enclosure is claimed.
"""
from dataclasses import dataclass,asdict
import time
from . import c_j_u_reduced as previous
from . import c_j_u_precision as precision

blocks,old=previous.blocks,previous.old
linear,pr,rc2=previous.linear,previous.pr,previous.rc2
Coefficients=previous.Coefficients
VERSION='array_point_extended_primal_latest_refinement_pair_v1'


@dataclass(frozen=True)
class Policy(previous.Policy):
    inner_refinements: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.inner_refinements != 2:raise ValueError('EXACTLY_TWO_REGISTERED_REFINEMENTS')


DEFAULT=Policy()
Unqualified=previous.Unqualified


def state(g,J,U,spec,policy=DEFAULT,*,base_U=None,eta=None,parts_evaluator=None):
    import numpy as np
    from .c_j_u_consistency_repair import compare
    tick,cpu=time.monotonic(),time.process_time()
    parts,gj,gu,chart,meta=precision.evaluate(g,J,U,spec,base_U=base_U,eta=eta,parts_evaluator=parts_evaluator)
    zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    # Do not pack/unpack J merely to evaluate a reference: binary64 x=D*j
    # followed by j=x/D can move by one ulp. Both evaluations use J/U directly.
    binary,old_j,_=(parts_evaluator or precision.parts_gradient_J)(g,J,U,spec,extended=False)
    old_value,old_u=old.u_value_gradient(zero,g,J,U,spec,binary['penalty'])
    check=compare(parts['objective'],binary['objective'],atol=policy.value_atol,rtol=policy.value_rtol)
    adcheck=compare(parts['objective'],old_value,atol=policy.value_atol,rtol=policy.value_rtol)
    if not check['PASS'] or not adcheck['PASS']:raise ValueError('SAME_ARRAY_CROSS_PRECISION_VALUE_MISMATCH')
    _,D=blocks.residual_jacobian(zero,g,J,U,spec)
    ld=precision.require_extended()
    residual=linear.reconstruct(J.astype(ld),U.astype(ld))-g.astype(ld)
    direct=np.asarray(D.astype(ld).T@(residual.ravel()/np.sqrt(ld(spec.loss_scale))),dtype=float)
    derivative=compare(gu,direct,atol=policy.gradient_atol,rtol=policy.gradient_rtol)
    if not derivative['PASS']:raise ValueError('SAME_ARRAY_FORWARD_PULLBACK_MISMATCH')
    jnorm,unorm=float(np.max(abs(gj))),float(np.linalg.norm(gu))
    legacy_j,legacy_u=float(np.max(abs(old_j))),float(np.linalg.norm(old_u))
    jgate=old.gradient_gate(max(jnorm,legacy_j),policy.j_gradient_infinity,policy)
    ugate=old.gradient_gate(max(unorm,legacy_u),policy.u_tangent_l2,policy)
    row=dict(components=parts,J_stationarity=jgate,U_stationarity=ugate,
        first_order_met=jgate['status']==ugate['status']=='MET',
        orthogonality_maxabs=float(np.max(abs(U.transpose(0,2,1)@U-np.eye(spec.norb)))),rigorous_stationarity=False,
        stable_J_gradient_infinity=jnorm,stable_U_tangent_l2=unorm,
        binary64_J_gradient_infinity=legacy_j,binary64_U_tangent_l2=legacy_u,
        stationarity_policy='Both same-array stable and binary64 norms must pass original thresholds/margins',
        value_same_array=check,binary64_AD_value_same_array=adcheck,forward_pullback_same_array=derivative,
        binary64_U_difference_observation=float(np.linalg.norm(gu-old_u)),
        precision=meta,wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu)
    return row,gj,gu,chart,old_u


class StableReducedOracle(previous.ReducedOracle):
    def __init__(self,g,J,U,spec,policy=DEFAULT,*,call=lambda k,l,f:f(),record=lambda *a:None):
        super().__init__(g,J,U,spec,policy,call=call,record=record)

    def evaluate(self,eta):
        import numpy as np
        eta=np.asarray(eta);s,pcy=self.spec,self.policy
        nrot=s.rank*s.norb*(s.norb-1)//2
        if eta.shape!=(nrot,) or eta.dtype!=np.float64 or not np.isfinite(eta).all():
            raise ValueError('FINITE_BINARY64_ETA_REQUIRED')
        key=eta.tobytes()
        if key in self.cache:point=self.cache[key]
        else:
            if len(self.points)>=pcy.max_points:raise Unqualified('OUTER_LIMIT_UNQUALIFIED')
            label=str(len(self.points))
            def raw(step):
                return lambda info,arrays:self.record('inner_raw',label+'_'+step,info,arrays)
            def start():
                V=precision.geometry(self.U0,eta)
                if np.max(abs(V.transpose(0,2,1)@V-np.eye(s.norb)))>pcy.orthogonality:
                    raise ValueError('U_ORTHOGONALITY_FAIL_NO_PROJECTION')
                p=Coefficients(self.g,V,s)
                J,info=blocks.solve_j(p,self.J0,pcy,record=raw('solve'))
                return V,p,J,info
            V,p,J0,i0=self.call('inner_J_solve',label+'_solve',start)
            J1,i1=self.call('inner_J_solve',label+'_refine1',lambda:previous.refine_j(p,J0,pcy,record=raw('refine1')))
            a,j1,u1,c1,legacy1=self.call('precision_state',label+'_refine1',lambda:state(
                self.g,J1,V,s,pcy,base_U=self.U0,eta=eta))
            self.record('precision_state',label+'_refine1',a,dict(J=J1,U=V,eta=eta,
                gradient_J=j1,physical_gradient_U=u1,chart_gradient=c1,binary64_gradient_U=legacy1))
            J2,i2=self.call('inner_J_solve',label+'_refine2',lambda:previous.refine_j(p,J1,pcy,record=raw('refine2')))
            b,j2,u2,c2,legacy2=self.call('precision_state',label+'_refine2',lambda:state(
                self.g,J2,V,s,pcy,base_U=self.U0,eta=eta))
            self.record('precision_state',label+'_refine2',b,dict(J=J2,U=V,eta=eta,
                gradient_J=j2,physical_gradient_U=u2,chart_gradient=c2,binary64_gradient_U=legacy2))
            self.record('inner_summary',label,dict(solves=[i0,i1,i2]),None)
            def admission():
                structure=previous.applicability(p,pcy)
                parts0=p.parts(p.pack(J0),independent=True)
                _,coarse=old.u_value_gradient(np.zeros(nrot),self.g,J0,V,s,parts0['penalty'])
                legacy_shift=float(np.linalg.norm(legacy1-coarse))
                shift=max(float(np.linalg.norm(u2-u1)),float(np.linalg.norm(c2-c1)))
                full=p.is_pr or all(i['rank']==len(p.pack(J2)) for i in (i0,i1,i2))
                jmet=b['J_stationarity']['status']=='MET'
                status=('J_UNQUALIFIED' if not jmet else 'REDUCED_GRADIENT_UNQUALIFIED'
                    if not structure['admitted'] or not full or shift>pcy.refinement_U_l2 else 'NUMERICAL_PILOT_ADMITTED')
                bound=None if p.is_pr else precision.rc2_residual_bound(self.g,J2,V,s)
                return dict(status=status,applicability=structure,latest_refinement_U_shift_l2=shift,
                    refinement_U_shift_l2=shift,refinement_U_allowance=pcy.refinement_U_l2,
                    legacy_first_pair_physical_shift_observation=legacy_shift,
                    legacy_result_not_reclassified=True,full_numeric_rank=bool(full),
                    objective_refinement=old.nonincrease(a['components']['objective'],b['components']['objective'],pcy),
                    RC2_analytic_bound_floating_plugin=bound,rigorous_gradient_error_bound=None,
                    qualification_level='LATEST_PAIR_NUMERICAL_PILOT_NOT_ERROR_ENCLOSURE')
            valid=self.call('reduced_admission',label,admission)
            point=dict(label=label,J=J2,U=V,eta=eta.copy(),state=b,gradient=c2,
                value=b['components']['objective'],qualification=valid,inner=[i0,i1,i2])
            self.points.append(point);self.cache[key]=point
            self.record('envelope',label,valid,dict(eta=eta,gradient=c2))
        if point['qualification']['status']!='NUMERICAL_PILOT_ADMITTED':raise Unqualified(point['qualification']['status'])
        return point


def coupled_value_gradient(z,g,U,spec):
    """Synthetic alternative only. Same joint local variables, stable primal."""
    import numpy as np
    nrot=spec.rank*spec.norb*(spec.norb-1)//2
    J=Coefficients(g,U,spec).unpack(z[nrot:])
    V=precision.geometry(U,z[:nrot])
    parts,gj,_,chart,_=precision.evaluate(g,J,V,spec,base_U=U,eta=z[:nrot])
    return parts['objective'],np.r_[chart,gj]


def optimize(g,J,U,spec,policy=DEFAULT,*,call=lambda k,l,f:f(),record=lambda *a:None,solver=None):
    """SciPy BFGS, every line-search point re-solves J. No fixed-J TRF.

    Callback checks physical tangent stationarity, not merely exponential-chart
    stationarity. No retry, recenter restart or automatic alternate solver.
    """
    import numpy as np
    from scipy.optimize import minimize
    oracle=StableReducedOracle(g,J,U,spec,policy,call=call,record=record)
    zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    accepted=[];stalls=0;raw=None;status='U_UNQUALIFIED';tick,cpu=time.monotonic(),time.process_time()
    def callback(intermediate_result):
        nonlocal stalls
        p=oracle.evaluate(intermediate_result.x)
        progress=old.nonincrease(accepted[-1]['value'],p['value'],policy)
        stalls=stalls+1 if abs(progress['decrease'])<=policy.stall_atol+policy.stall_rtol*abs(p['value']) else 0
        accepted.append(p)
        record('accepted',p['label'],dict(objective=p['value'],physical=p['state'],progress=progress),None)
        if p['state']['first_order_met']:raise Unqualified('BOTH_FIRST_ORDER_MET')
        if stalls>=policy.stall_cycles:raise Unqualified('OUTER_STALLED')
    try:
        p=oracle.evaluate(zero);accepted.append(p)
        if p['state']['first_order_met']:
            status='BOTH_FIRST_ORDER_MET'
        else:
            def driver():
                # Expected qualification stops must not become journal FAILs.
                try:
                    res=(solver or minimize)(oracle.fun,zero,jac=True,method='BFGS',callback=callback,
                        options=dict(maxiter=policy.outer_maxiter,gtol=policy.outer_gtol,norm=2,xrtol=0.,c1=1e-4,c2=.9))
                    record('outer_solver_result','returned',old._raw(res),dict(eta=res.x,jac=res.jac))
                    return res,None
                except Unqualified as stop:
                    return None,stop.status
            raw,stop=call('outer_driver','BFGS',driver)
            if stop:status=stop
            else:
                p=oracle.evaluate(np.asarray(raw.x))
                if not np.array_equal(p['eta'],accepted[-1]['eta']):callback(raw)
                if not np.allclose(raw.jac,p['gradient'],atol=policy.gradient_atol,rtol=policy.gradient_rtol):
                    raise ValueError('OUTER_RAW_ENDPOINT_GRADIENT_MISMATCH')
                status='BOTH_FIRST_ORDER_MET' if p['state']['first_order_met'] else 'U_UNQUALIFIED'
    except Unqualified as stop:
        status=stop.status
    if not accepted:
        # Initial unqualified finite pair is diagnostic only, never called a
        # reduced solution or silently replaced by the old initial snapshot.
        if not oracle.points:raise ValueError('NO_FINITE_INITIAL_PAIR')
        terminal=oracle.points[0]
    else:terminal=accepted[-1]
    result=dict(status=status,classification=blocks.classification(terminal['state']),
        physical_first_order_met=terminal['state']['first_order_met'],
        first_order_qualified=status=='BOTH_FIRST_ORDER_MET' and terminal['state']['first_order_met'],
        terminal=terminal['state'],terminal_label=terminal['label'],
        accepted_labels=[p['label'] for p in accepted],completed_cycles=max(0,len(accepted)-1),
        points=len(oracle.points),inner_solves=3*len(oracle.points),
        qualification=terminal['qualification'],stop_trial_qualification=oracle.points[-1]['qualification'],
        outer_raw=None if raw is None else old._raw(raw),policy=asdict(policy),
        outer_raw_unavailable_reason='callback/qualification termination before SciPy return' if raw is None else None,
        wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
        rigorous_stationarity=False,rigorous_reduced_gradient=False,retry=0,resume=0)
    record('outer_raw','terminal',result,None if raw is None else dict(eta=raw.x,jac=raw.jac))
    return terminal['J'],terminal['U'],result
