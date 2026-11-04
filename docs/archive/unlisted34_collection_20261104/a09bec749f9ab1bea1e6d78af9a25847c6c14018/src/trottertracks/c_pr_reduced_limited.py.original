"""Opt-in PR-only reduced advancement pilot; no molecular I/O.

Reuse the frozen three-solve oracle and SciPy BFGS. Extra admission metadata
separates raw solver flags from numerical inner/structure/stability gates.
No old entry, objective, solver, allowance or SOURCE is patched.
"""
from dataclasses import asdict, replace
import time
from . import c_pr_inner_point_solver as inner

stable = inner.stable
VERSION = 'PR_same_array_reduced_limited_BFGS_v1'
# Scientific scope limits (not CPU/RAM/wall limits). Include initial eta.
DEFAULT = replace(inner.DEFAULT, outer_maxiter=2, max_points=8)


class Oracle(inner.PRReducedOracle):
    def __init__(self, g, J, U, spec, policy=DEFAULT, *,
                 call=lambda k,l,f:f(), record=lambda *a:None, backend=None):
        self.summaries = {}

        def evidence(kind, label, data, arrays):
            if kind == 'inner_summary':
                self.summaries[label] = data['solves']
            record(kind, label, data, arrays)

        def dispatch(kind, label, fn):
            if kind != 'reduced_admission':
                return call(kind, label, fn)

            def admission():
                q = fn()  # Original rank/nullspace and latest-pair rule.
                solves = self.summaries[label]
                gates = dict(
                    final_inner_J=solves[-1]['stationarity']['status'] == 'MET',
                    final_independent_J=q['status'] != 'J_UNQUALIFIED',
                    rank_nullspace=q['applicability']['admitted'],
                    full_numeric_rank=q['full_numeric_rank'],
                    latest_pair_stability=q['latest_refinement_U_shift_l2'] <= policy.refinement_U_l2,
                    block_nonincrease=all(x['progress']['status'] == 'PASS_WITHIN_ENGINEERING_ALLOWANCE' for x in solves),
                    refinement_nonincrease=q['objective_refinement']['status'] == 'PASS_WITHIN_ENGINEERING_ALLOWANCE')
                q.update(inherited_admission_status=q['status'], admission_gates=gates,
                    inner_J_statuses=[x['stationarity']['status'] for x in solves],
                    raw_solver_flags=[{k:x[k] for k in ('success','status','message')} for x in solves],
                    raw_success_is_not_an_admission_gate=True,
                    first_inner_J_is_observation_final_refined_J_decides=True)
                if not gates['final_inner_J'] or not gates['final_independent_J']:
                    q['status'] = 'J_UNQUALIFIED'
                elif not all(gates.values()):
                    q['status'] = 'REDUCED_GRADIENT_UNQUALIFIED'
                return q
            return call(kind, label, admission)
        super().__init__(g,J,U,spec,policy,call=dispatch,record=evidence,backend=backend)


def optimize(g,J,U,spec,policy=DEFAULT,*,call=lambda k,l,f:f(),
             record=lambda *a:None,solver=None):
    """At most two accepted U updates / eight distinct eta, no restart.

    Every new trial starts from the original J, followed by two refinements.
    Expected qualification/point stops are returned, correctness errors escape
    after raw evidence. Last unaccepted trial is never the accepted endpoint.
    """
    import numpy as np
    from scipy.optimize import minimize
    oracle=Oracle(g,J,U,spec,policy,call=call,record=record)
    zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    accepted=[];raw=None;stop=None;stalls=0;outer_invocations=0
    tick,cpu=time.monotonic(),time.process_time()

    def callback(intermediate_result):
        nonlocal stalls
        p=oracle.evaluate(np.asarray(intermediate_result.x))
        if np.array_equal(p['eta'],accepted[-1]['eta']):return
        if len(accepted)-1 >= policy.outer_maxiter:
            raise stable.Unqualified('LIMITED_UPDATE_TARGET_REACHED')
        progress=stable.old.nonincrease(accepted[-1]['value'],p['value'],policy)
        stalls=stalls+1 if abs(progress['decrease']) <= policy.stall_atol+policy.stall_rtol*abs(p['value']) else 0
        accepted.append(p)
        record('accepted',p['label'],dict(objective=p['value'],physical=p['state'],progress=progress),dict(eta=p['eta']))
        if p['state']['first_order_met']:raise stable.Unqualified('BOTH_FIRST_ORDER_MET')
        if stalls >= policy.stall_cycles:raise stable.Unqualified('OUTER_STALLED')

    try:
        p=oracle.evaluate(zero);accepted.append(p)
        if p['state']['first_order_met']:
            stop='BOTH_FIRST_ORDER_MET'
        else:
            outer_invocations=1
            raw=(solver or minimize)(oracle.fun,zero,jac=True,method='BFGS',callback=callback,
                options=dict(maxiter=policy.outer_maxiter,gtol=policy.outer_gtol,norm=2,xrtol=0.,c1=1e-4,c2=.9))
            record('outer_solver_result','returned',stable.old._raw(raw),dict(eta=raw.x,jac=raw.jac))
            p=oracle.evaluate(np.asarray(raw.x))
            if not np.array_equal(p['eta'],accepted[-1]['eta']):callback(raw)
            for name,actual,expected in [('gradient',raw.jac,p['gradient']),('value',raw.fun,p['value'])]:
                check=inner.point.compare(actual,expected,
                    atol=policy.gradient_atol if name=='gradient' else policy.value_atol,
                    rtol=policy.gradient_rtol if name=='gradient' else policy.value_rtol)
                record('outer_endpoint_check',name,check,None)
                if not check['PASS']:raise ValueError('OUTER_RAW_ENDPOINT_'+name.upper()+'_MISMATCH')
            stop='BOTH_FIRST_ORDER_MET' if p['state']['first_order_met'] else 'LIMITED_OUTER_RETURN'
    except stable.Unqualified as reason:
        stop=reason.status

    terminal=accepted[-1] if accepted else oracle.points[0]
    updates=max(0,len(accepted)-1)
    result=dict(status=stop, advancement='ACCEPTED_U_ADVANCE_OBSERVED' if updates else 'NO_ACCEPTED_U_UPDATE',
        terminal_kind='accepted' if accepted else 'unqualified_initial_diagnostic',
        accepted_labels=[p['label'] for p in accepted],completed_U_updates=updates,
        outer_solver_invocations=outer_invocations,
        distinct_U_points=len(oracle.points),inner_solves=3*len(oracle.points),
        refinement_solves=2*len(oracle.points),terminal_label=terminal['label'],terminal=terminal['state'],
        terminal_qualification=terminal['qualification'],stop_trial_label=oracle.points[-1]['label'],
        stop_trial_qualification=oracle.points[-1]['qualification'],
        qualification_by_point=[dict(label=p['label'],**p['qualification']) for p in oracle.points],
        outer_raw=None if raw is None else stable.old._raw(raw),
        outer_raw_unavailable_reason='qualification/callback stop or initial first-order gate before SciPy return' if raw is None else None,
        physical_first_order_met=terminal['state']['first_order_met'],
        first_order_qualified=bool(accepted) and stop=='BOTH_FIRST_ORDER_MET' and terminal['state']['first_order_met'],
        rigorous_reduced_gradient=False,rigorous_stationarity=False,
        DF_budget=None,PR_performance=None,qualified_tail_comparison=None,
        policy=asdict(policy),wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
        RC2_continuation=0,retry=0,resume=0)
    record('limited_terminal','final',result,None)
    return terminal['J'],terminal['U'],result
