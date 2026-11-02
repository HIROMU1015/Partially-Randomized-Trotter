"""Opt-in, one qualified Armijo step; frozen PR inner/admission unchanged.

SciPy BFGS has no missing-gradient trial protocol. This small outer controller
does not replace Newton-CG, invent failed values, or update a BFGS matrix.
It proposes a bounded sequence along one admitted steepest-descent direction.
"""
from dataclasses import dataclass, asdict, replace
import math
from . import c_pr_reduced_limited as previous

stable=previous.stable
VERSION='PR_qualified_only_bounded_Armijo_step_v1'
DEFAULT=replace(previous.DEFAULT,outer_maxiter=1,max_points=9)


@dataclass(frozen=True)
class SearchPolicy:
    max_trials: int = 8
    contraction: float = .5
    c1: float = 1e-4

    def __post_init__(self):
        if type(self.max_trials) is not int or self.max_trials<1:raise ValueError('POSITIVE_TRIAL_LIMIT')
        if not 0<self.contraction<1 or not 0<self.c1<1:raise ValueError('ARMIJO_POLICY')


SEARCH=SearchPolicy()


class Oracle(previous.Oracle):
    """Same admission; add only norms of already evaluated refined arrays.

    Future science can distinguish physical/chart shifts and delta J. No new
    gradient/Hessian query, gauge fixing, or admission relaxation is introduced.
    """
    def __init__(self,*args,record=lambda *a:None,**kwargs):
        states={}
        def evidence(kind,label,data,arrays):
            import numpy as np
            if kind=='precision_state':states[label]=arrays
            if kind=='envelope':
                a,b=states[label+'_refine1'],states[label+'_refine2']
                if not np.array_equal(a['U'],b['U']):raise ValueError('PAIR_U_ARRAY_MISMATCH')
                data['pair_diagnostics']=dict(
                    physical_shift_l2=float(np.linalg.norm(b['physical_gradient_U']-a['physical_gradient_U'])),
                    chart_shift_l2=float(np.linalg.norm(b['chart_gradient']-a['chart_gradient'])),
                    delta_J_frobenius=float(np.linalg.norm(b['J']-a['J'])),
                    same_U_arrays=True,gauge_decomposition=None,mixed_Hessian=None,
                    diagnostics_not_error_bounds=True)
            record(kind,label,data,arrays)
        super().__init__(*args,record=evidence,**kwargs)


def _valid(p):
    import numpy as np
    if p['qualification']['status']!='NUMERICAL_PILOT_ADMITTED':raise ValueError('QUALIFIED_ORACLE_CONTRACT')
    x,g=np.asarray(p['eta']),np.asarray(p['gradient'])
    if (x.dtype!=np.float64 or g.dtype!=np.float64 or g.shape!=x.shape or x.ndim!=1
        or not np.isfinite(x).all() or not np.isfinite(g).all() or not math.isfinite(p['value'])):
        raise ValueError('FINITE_QUALIFIED_POINT_REQUIRED')


def search(evaluate,initial,policy=SEARCH,*,conservative=False,record=lambda *a:None):
    """Return an admitted, strictly decreasing Armijo point or explicit STOP.

    Only the two registered numerical qualification exceptions are rejectable.
    A correctness exception escapes. Invalid gradients never enter a direction,
    slope, Armijo comparison or quasi-Newton update (there is no such update).
    """
    import numpy as np
    _valid(initial);x=initial['eta'];g=initial['gradient'];direction=-g
    slope=float(g@direction);norm=float(np.linalg.norm(g))
    if not math.isfinite(slope) or slope>=0:return initial,dict(status='NO_QUALIFIED_DESCENT_DIRECTION',trials=0,accepted_U_updates=0)
    # Same initial scale as SciPy1.14.1's FIRST BFGS line-search query with H0=I.
    alpha0=min(1.,1.01/norm)
    for i in range(policy.max_trials):
        alpha=alpha0*policy.contraction**i;eta=x+alpha*direction
        if np.array_equal(eta,x):return initial,dict(status='STEP_ROUNDING_STALLED',trials=i,accepted_U_updates=0)
        event=dict(trial=i,alpha=float(alpha),initial_slope=slope,objective=None,gradient_used=False)
        record('guarded_trial_attempt',str(i),dict(event),dict(eta=eta))
        try:
            p=evaluate(eta)
        except stable.Unqualified as stop:
            event.update(status=stop.status,decision='REJECT_NUMERICALLY_UNQUALIFIED')
            record('guarded_trial',str(i),event,None)
            if conservative or stop.status not in ('J_UNQUALIFIED','REDUCED_GRADIENT_UNQUALIFIED'):
                return initial,dict(status=stop.status,trials=i+1,accepted_U_updates=0)
            continue
        _valid(p)
        if not np.array_equal(p['eta'],eta):raise ValueError('QUALIFIED_TRIAL_IDENTITY_MISMATCH')
        bound=float(initial['value']+policy.c1*alpha*slope)
        accept=p['value']<=bound and p['value']<initial['value']
        event.update(status='QUALIFIED',objective=p['value'],gradient_used=True,
            qualified_gradient_l2=float(np.linalg.norm(p['gradient'])),Armijo_bound=bound,
            decision='ACCEPT_QUALIFIED_ARMIJO' if accept else 'REJECT_ARMIJO')
        record('guarded_trial',str(i),event,None)
        if accept:return p,dict(status='QUALIFIED_ARMIJO_STEP_ACCEPTED',trials=i+1,
            accepted_U_updates=1,alpha=float(alpha),initial_slope=slope,Armijo_bound=bound,
            actual_decrease=float(initial['value']-p['value']),strong_Wolfe_claim=False)
    return initial,dict(status='GUARDED_TRIAL_LIMIT_NO_ACCEPT',trials=policy.max_trials,accepted_U_updates=0)


def optimize(g,J,U,spec,policy=DEFAULT,*,call=lambda k,l,f:f(),record=lambda *a:None):
    import numpy as np
    oracle=Oracle(g,J,U,spec,policy,call=call,record=record)
    zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    initial=None
    try:
        initial=oracle.evaluate(zero)
        if initial['state']['first_order_met']:
            terminal=initial;step=dict(status='BOTH_FIRST_ORDER_MET',trials=0,accepted_U_updates=0)
        else:terminal,step=search(oracle.evaluate,initial,record=record)
    except stable.Unqualified as stop:
        terminal=oracle.points[0];step=dict(status=stop.status,trials=0,accepted_U_updates=0)
    result=dict(**step,terminal_label=terminal['label'],terminal_kind='accepted' if initial else 'unqualified_initial_diagnostic',
        terminal=terminal['state'],terminal_qualification=terminal['qualification'],
        accepted_labels=([initial['label']] if initial else [])+([terminal['label']] if step['accepted_U_updates'] else []),
        distinct_U_points=len(oracle.points),inner_solves=3*len(oracle.points),refinement_solves=2*len(oracle.points),
        last_trial_label=oracle.points[-1]['label'],last_trial_qualification=oracle.points[-1]['qualification'],
        qualification_by_point=[dict(label=p['label'],**p['qualification']) for p in oracle.points],
        outer_method=VERSION,outer_BFGS_invocations=0,outer_raw=None,policy=asdict(policy),search_policy=asdict(SEARCH),
        rigorous_reduced_gradient=False,DF_budget=None,PR_performance=None,RC2_continuation=0,retry=0,resume=0)
    record('guarded_terminal','final',result,None)
    return terminal['J'],terminal['U'],result
