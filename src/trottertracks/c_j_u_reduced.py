"""Guarded variable projection; no science on import.

The exact envelope theorem and numerical inner-solve admission are distinct.
SciPy owns optimization. Existing DF maps, objectives, Newton-CG point checks,
right-exponential rotations and physical tangent diagnostics remain unchanged.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import time
from . import c_separated_j_u_consistent as blocks

old = blocks.old
linear, pr, rc2 = blocks.linear, blocks.pr, blocks.rc2
Coefficients, retract, state = blocks.Coefficients, blocks.retract, blocks.state
VERSION = 'guarded_reduced_BFGS_refined_inner_v1'


@dataclass(frozen=True)
class Policy(blocks.Policy):
    outer_maxiter: int = 64
    max_points: int = 128
    outer_gtol: float = 1e-7
    refinement_U_l2: float = 1e-6
    rank_guard_factor: float = 256.
    rank_gap_factor: float = 1024.

    def __post_init__(self):
        # The base validator treats new integer fields as positive numbers;
        # enforce their integer nature here too.
        super().__post_init__()
        for k in ('outer_maxiter', 'max_points'):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError('POSITIVE_INTEGER_POLICY:'+k)


DEFAULT = Policy()


class Unqualified(Exception):
    """Expected numerical qualification stop, not a correctness failure."""
    def __init__(self, status):
        self.status = status
        super().__init__(status)


def applicability(p, policy=DEFAULT):
    """Sufficient, conservative structural test; SVD is a numerical pilot.

    For PR, ker H = ker A intersect ker T. T is injective on tail J.
    The head always has LD-1 U-INDEPENDENT gauges: redistribute c_t 11^T
    with sum c_t=0. If these are the ONLY head null directions, the quotient
    is strictly convex locally and all minimizers have the same U derivative.
    Additional/ambiguous nulls are refused, not removed with a pseudoinverse.
    """
    import numpy as np
    if not p.is_pr and not getattr(p,'is_entry',False):
        m = 2*p.spec.strength/p.spec.l2_scale
        return dict(admitted=bool(m > 0), condition='RC2_positive_L2_unique',
                    strong_convexity_lower=m, rigorous_numerical_enclosure=False)
    s = p.spec
    strength=s.strength if getattr(p,'is_entry',False) else s.gamma
    epsilon=s.entry_epsilon if getattr(p,'is_entry',False) else s.epsilon
    if strength <= 0 or epsilon <= 0:
        return dict(admitted=False, condition='PR_positive_gamma_epsilon_required')
    d = s.norb*(s.norb+1)//2
    H = p.A[:, :s.ld*d]
    sv = np.linalg.svd(H, compute_uv=False)
    expected = s.ld*d-max(0, s.ld-1)
    guard = policy.rank_guard_factor*np.finfo(float).eps*max(H.shape)*max(1., sv[0] if len(sv) else 0.)
    smallest = (float(sv[expected-1]) if expected <= len(sv) else 0.) if expected else None
    null_max = float(np.max(sv[expected:], initial=0.))
    T = p.T[:, s.ld*d:]
    ts = np.linalg.svd(T, compute_uv=False)
    tail_injective = T.shape[1] == 0 or (len(ts) == T.shape[1] and ts[-1] > policy.rank_gap_factor*guard)
    admitted = bool(tail_injective and (expected == 0 or smallest > policy.rank_gap_factor*guard)
                    and null_max <= guard)
    return dict(admitted=admitted, condition='ENTRY_only_universal_head_gauge' if getattr(p,'is_entry',False) else 'PR_only_universal_head_gauge',
        expected_head_rank=expected, universal_nullity=max(0, s.ld-1),
        head_singular_values=sv.tolist(), tail_singular_values=ts.tolist(),
        rank_guard=guard, minimum_quotient_singular=smallest,
        extra_null_singular_max=null_max, tail_injective=bool(tail_injective),
        strict_convexity_on_quotient_only=True, rigorous_numerical_enclosure=False)


def refine_j(p, J, policy=DEFAULT, *, record=lambda *a: None):
    """One counted refinement, same F. PR reuses corrected Newton-CG.

    RC2 solves the augmented residual with gelsd; unlike repeating the original
    solve, this can resolve residual rounding. No normal equations or ridge
    beyond the ORIGINAL RC2 L2 are introduced.
    """
    import numpy as np
    from scipy.linalg import lstsq
    if p.is_pr:
        return blocks.solve_j(p, J, policy, record=record)
    tick, cpu = time.monotonic(), time.process_time()
    x = p.pack(J)
    B = np.vstack((p.A/np.sqrt(p.spec.loss_scale),
        np.sqrt(2*p.spec.strength/p.spec.l2_scale)*np.eye(len(x))))
    b = np.r_[p.g.ravel()/np.sqrt(p.spec.loss_scale), np.zeros(len(x))]
    # Extended accumulation only in this residual, LAPACK input stays binary64.
    rhs = np.asarray(b.astype(np.longdouble)-B.astype(np.longdouble)@x.astype(np.longdouble), dtype=float)
    cond = np.finfo(float).eps*max(B.shape)
    dx, _, rank, singular = lstsq(B, rhs, cond=cond, lapack_driver='gelsd')
    y = x+dx
    before, _ = p.vg(x, independent=True)
    value, gradient = p.vg(y, independent=True)
    info = dict(method='same_augmented_residual_correction_gelsd', success=True, status=0,
        rank=int(rank), singular_values=singular.tolist(), linear_solves=1,
        correction_l2=float(np.linalg.norm(dx)), nfev=0, njev=0, nhev=0, nit=0,
        wall_seconds=time.monotonic()-tick, CPU_seconds=time.process_time()-cpu,
        independent_objective=value, normal_equations=False,
        stationarity=old.gradient_gate(float(np.max(abs(gradient))), policy.j_gradient_infinity, policy),
        progress=old.nonincrease(before, value, policy))
    record(info, dict(x0=x, x=y, correction=dx, gradient_J=gradient))
    return p.unpack(y), info


class ReducedOracle:
    """Each new eta: TWO explicit J calls, state, envelope. Exact eta cache.

    Trial gradients are partial derivatives at the SAME eta and refined J;
    the inner solver is never differentiated. Qualification here means a
    NUMERICAL PILOT (stationarity + structural guard + refinement stability),
    not an error enclosure. Rejected trials never become accepted iterates.
    """
    def __init__(self, g, J, U, spec, policy=DEFAULT, *, call=lambda k,l,f:f(), record=lambda *a:None):
        self.g, self.J0, self.U0, self.spec, self.policy = g, J.copy(), U.copy(), spec, policy
        self.call, self.record = call, record
        self.cache, self.points = {}, []

    def evaluate(self, eta):
        import numpy as np
        eta = np.asarray(eta)
        expected = self.spec.rank*self.spec.norb*(self.spec.norb-1)//2
        if eta.shape != (expected,) or eta.dtype != np.float64 or not np.isfinite(eta).all():
            raise ValueError('FINITE_BINARY64_ETA_REQUIRED')
        key = eta.tobytes()
        if key in self.cache:
            point = self.cache[key]
        else:
            if len(self.points) >= self.policy.max_points:
                raise Unqualified('OUTER_LIMIT_UNQUALIFIED')
            label = str(len(self.points)); pcy = self.policy
            def raw(suffix):
                return lambda info, arrays:self.record('inner_raw', label+'_'+suffix, info, arrays)
            def initial_solve():
                V = retract(self.U0, eta)
                if np.max(abs(V.transpose(0,2,1)@V-np.eye(self.spec.norb))) > pcy.orthogonality:
                    raise ValueError('U_ORTHOGONALITY_FAIL_NO_PROJECTION')
                p = Coefficients(self.g,V,self.spec)
                j,info=blocks.solve_j(p,self.J0,pcy,record=raw('solve'))
                return V,p,j,info
            V,p,J0,first = self.call('inner_J_solve', label+'_solve', initial_solve)
            J1, second = self.call('inner_J_solve', label+'_refine', lambda:refine_j(
                p, J0, pcy, record=raw('refine')))
            self.record('inner_summary',label,dict(solves=[first,second]),None)
            row, gj, gu = self.call('reduced_state', label, lambda:state(self.g,J1,V,self.spec,pcy))
            self.record('state', label, row, dict(J=J1,U=V,eta=eta,gradient_J=gj,gradient_U=gu))
            def envelope():
                a = applicability(p,pcy)
                parts0=p.parts(p.pack(J0),independent=True)
                f0,g0=old.u_value_gradient(eta,self.g,J0,self.U0,self.spec,parts0['penalty'])
                value,grad=old.u_value_gradient(eta,self.g,J1,self.U0,self.spec,row['components']['penalty'])
                from .c_j_u_consistency_repair import compare
                check=compare(value,row['components']['objective'],atol=pcy.value_atol,rtol=pcy.value_rtol)
                r,D=blocks.residual_jacobian(eta,self.g,J1,self.U0,self.spec)
                derivative=compare(grad,D.T@r,atol=pcy.gradient_atol,rtol=pcy.gradient_rtol)
                if not check['PASS'] or not derivative['PASS']:
                    raise ValueError('REDUCED_SAME_POINT_VALUE_OR_GRADIENT_MISMATCH')
                zero=np.zeros_like(eta)
                _,physical0=old.u_value_gradient(zero,self.g,J0,V,self.spec,parts0['penalty'])
                shift=max(float(np.linalg.norm(grad-g0)),float(np.linalg.norm(gu-physical0)))
                jmet=row['J_stationarity']['status']=='MET'
                full_rank=p.is_pr or all(info['rank']==len(p.pack(J1)) for info in (first,second))
                status=('J_UNQUALIFIED' if not jmet else
                    'REDUCED_GRADIENT_UNQUALIFIED' if not a['admitted'] or not full_rank or shift>pcy.refinement_U_l2
                    else 'NUMERICAL_PILOT_ADMITTED')
                return dict(status=status, applicability=a, refinement_U_shift_l2=shift,
                    RC2_augmented_full_numeric_rank=bool(full_rank) if not p.is_pr else None,
                    refinement_U_allowance=pcy.refinement_U_l2, value_same_point=check,
                    gradient_same_point=derivative,
                    envelope_reverse_VG_calls=3,forward_residual_Jacobian_calls=1,
                    chart_gradient_l2=float(np.linalg.norm(grad)),
                    exact_envelope_only_at_exact_argmin=True, rigorous_gradient_error_bound=None,
                    refinement_stability_is_not_a_bound=True),value,grad
            validity,value,grad=self.call('envelope_gradient',label,envelope)
            point=dict(label=label,J=J1,U=V,eta=eta.copy(),state=row,gradient=grad,value=value,
                qualification=validity,inner=[first,second])
            self.points.append(point);self.cache[key]=point
            self.record('envelope',label,validity,dict(eta=eta,gradient=grad))
        if point['qualification']['status']!='NUMERICAL_PILOT_ADMITTED':
            raise Unqualified(point['qualification']['status'])
        return point

    def fun(self, eta):
        p=self.evaluate(eta)
        return p['value'],p['gradient']


def optimize(g,J,U,spec,policy=DEFAULT,*,call=lambda k,l,f:f(),record=lambda *a:None,solver=None):
    """SciPy BFGS, every line-search point re-solves J. No fixed-J TRF.

    Callback checks physical tangent stationarity, not merely exponential-chart
    stationarity. No retry, recenter restart or automatic alternate solver.
    """
    import numpy as np
    from scipy.optimize import minimize
    oracle=ReducedOracle(g,J,U,spec,policy,call=call,record=record)
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
        points=len(oracle.points),inner_solves=2*len(oracle.points),
        qualification=terminal['qualification'],stop_trial_qualification=oracle.points[-1]['qualification'],
        outer_raw=None if raw is None else old._raw(raw),policy=asdict(policy),
        outer_raw_unavailable_reason='callback/qualification termination before SciPy return' if raw is None else None,
        wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
        rigorous_stationarity=False,rigorous_reduced_gradient=False,retry=0,resume=0)
    record('outer_raw','terminal',result,None if raw is None else dict(eta=raw.x,jac=raw.jac))
    return terminal['J'],terminal['U'],result


@lru_cache(None)
def _coupled_factory(rank,norb,ld,is_pr):
    """Alternative prototype: joint local block, NO envelope assumption.

    Reuses the existing JAX DF loss/rotation and penalty; no new reconstruction.
    This is synthetic-only in the first H4 contract, never an automatic fallback.
    """
    import jax
    import jax.numpy as jnp
    from ffsim.linalg.util import real_symmetrics_from_parameters_jax
    nrot=rank*norb*(norb-1)//2
    weights=old.weights(rank,norb)
    def loss(z,g,U,scale,mass_scale,strength,epsilon):
        eta,x=z[:nrot],z[nrot:]
        J=real_symmetrics_from_parameters_jax(x/jnp.asarray(weights),dim=norb,n_mats=rank)
        if is_pr:
            coeff=old.pr.coefficients(J[ld:],jnp)
            penalty=strength*(jnp.sum(jnp.hypot(coeff,epsilon))/mass_scale)**2
        else:penalty=strength*jnp.sum(x*x)/mass_scale
        return old._u_factory(rank,norb)(eta,g,J,U,scale,penalty)[0]
    return jax.jit(jax.value_and_grad(loss))


def coupled_value_gradient(z,g,U,spec):
    import numpy as np
    is_pr=isinstance(spec,old.pr.Objective)
    f=_coupled_factory(spec.rank,spec.norb,spec.ld,is_pr)
    v,h=f(z,g,U,spec.loss_scale,spec.mass_scale if is_pr else spec.l2_scale,
          spec.gamma if is_pr else spec.strength,spec.epsilon)
    v,h=float(v),np.asarray(h)
    if not np.isfinite(v) or not np.isfinite(h).all():raise ValueError('NONFINITE_COUPLED_GRADIENT')
    return v,h
