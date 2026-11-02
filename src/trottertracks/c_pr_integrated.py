"""One recentered PR trajectory; frozen inner oracle and physical qualification."""
from dataclasses import replace
from . import c_pr_small_step as small
stable=small.stable
DEFAULT=replace(small.DEFAULT,max_points=4)
FRACTIONS=(1/8,1/32,1/128)
MAX_UPDATES=64


def trajectory(g,Jcold,Ustart,spec,physical,*,call=lambda k,l,f:f(),record=lambda *a:None,max_updates=MAX_UPDATES):
    import numpy as np
    if not 1<=max_updates<=MAX_UPDATES:raise ValueError('REGISTERED_UPDATE_LIMIT')
    zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    history=[];trials=[];updates=0;points=0;stalls=0;best=None
    def emit(prefix):return lambda kind,label,data,arrays:record(kind,prefix+label,data,arrays)
    oracle=small.Oracle(g,Jcold,Ustart,spec,DEFAULT,call=call,record=emit('initial_'))
    try:current=oracle.evaluate(zero)
    except stable.Unqualified as e:
        return dict(status=e.status,can_performance=False,history=history,trials=trials,
            optimization_accepted_updates=0,physically_qualified_updates=0,best=None,point_evaluations=1)
    points+=1;initial_state=current['state']
    def qualify(p,label,update,step=None):
        nonlocal best
        row=call('physical_assessment',label,lambda:physical(label,p['J'],p['U']))
        data=dict(id=label,update=update,state=p['state'],qualification=p['qualification'],physical=row,step=step)
        history.append(data);record('qualified_history',label,data,None)
        if row['budget_PASS']:
            key=(row['exact_tail_lambda_Ha'],p['value'],update)
            if best is None or key<best['key']:best=dict(key=key,row=data,J=p['J'].copy(),U=p['U'].copy())
        return row['budget_PASS']
    if not qualify(current,'B_initial',0):
        return dict(status='B_INITIAL_DF_BUDGET_NOT_MET',can_performance=False,history=history,
            trials=trials,optimization_accepted_updates=0,physically_qualified_updates=0,best=None,point_evaluations=1)
    status='MAXIMUM_ACCEPTED_UPDATES';can=True
    for k in range(max_updates):
        if current['state']['first_order_met']:status='BOTH_FIRST_ORDER_MET';break
        base=current['U'];prefix=f'k{k:03d}_'
        oracle=small.Oracle(g,Jcold,base,spec,DEFAULT,call=call,record=emit(prefix))
        # The independently evaluated physical tangent is the eta=0 pullback.
        center=dict(current,eta=zero.copy(),label='0')
        oracle.points.append(center);oracle.cache[zero.tobytes()]=center
        accepted=False;fatal=False
        for j,q in enumerate(FRACTIONS):
            before=len(oracle.points);saved={}
            def evidence(kind,label,data,arrays):
                if kind=='independent_terminal_state':saved.update(arrays)
                emit(prefix+f'q{j}_')(kind,label,data,arrays)
            candidate,decision=small.propose(oracle.evaluate,center,small.StepPolicy(q),DEFAULT,
                record=emit(prefix+f'q{j}_'))
            points+=len(oracle.points)-before
            event=dict(update_attempt=k,trial_index=j,fraction=q,**{a:b for a,b in decision.items() if a!='fraction'})
            trials.append(event);record('trial_decision',prefix+str(j),event,None)
            if decision['status']!='PROVISIONAL_SMALL_ARMIJO_PASS':
                if decision['trials']==0:status=decision['status'];fatal=True;break
                continue
            post=call('independent_terminal_assessment',prefix+candidate['label'],
                lambda:small.assess(g,candidate,base,spec,DEFAULT,record=evidence))
            if not post['J_numerically_qualified']:status=post['status'];can=False;fatal=True;break
            updates+=1
            advanced=dict(candidate,state=post['state'],gradient=saved['physical_gradient_U'].copy(),eta=zero.copy())
            event.update(decision='OPTIMIZATION_STEP_ACCEPTED_PENDING_DF',independent=post)
            record('optimization_accepted',f'update{updates:03d}',event,
                dict(J=candidate['J'],U=candidate['U'],physical_gradient=saved['physical_gradient_U']))
            if not qualify(advanced,f'update{updates:03d}',updates,event):
                status='ACCEPTED_CANDIDATE_DF_BUDGET_NOT_MET';can=False;fatal=True;break
            record('accepted',f'update{updates:03d}',dict(objective=advanced['value'],
                physically_qualified=True,gradient_recentered=True),dict(J=advanced['J'],U=advanced['U']))
            current=advanced;accepted=True
            stalls=stalls+1 if decision['actual_decrease']<=DEFAULT.stall_atol+DEFAULT.stall_rtol*abs(center['value']) else 0
            break
        if fatal:break
        if not accepted:status='FINITE_STEP_GRID_NO_ACCEPT';break
        if current['state']['first_order_met']:status='BOTH_FIRST_ORDER_MET';break
        if stalls>=DEFAULT.stall_cycles:status='REGISTERED_OBJECTIVE_STALL';break
    return dict(status=status,can_performance=can,history=history,trials=trials,
        optimization_accepted_updates=updates,physically_qualified_updates=sum(x['update']>0 and x['physical']['budget_PASS'] for x in history),
        best=best,terminal=current,initial_state=initial_state,point_evaluations=points,
        inner_solves=3*points,refinement_solves=2*points,
        true_reduced_gradient_qualification=None,certified_descent=None,retry=0,resume=0)
