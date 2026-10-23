"""Reuse exact decomposition and the original account body with two constants adapted."""
from copy import deepcopy
import importlib
from types import FunctionType,SimpleNamespace
import h246_reversal_batch as c
import h246_order_cost_analysis_backend as old
from h2_accepted_compare_backend import Backend as Original


def adapted_account(epsilon):
    c.require(epsilon in c.EPS,'NEW_EPSILONS_ONLY_NO_MAIN_REACCOUNT')
    fn=Original.account;const=fn.__code__.co_consts
    c.require(sum(type(x) is float and x==.0005 for x in const)==1 and const.count('_accounting.json')==1,'ORIGINAL_ACCOUNT_CONSTANT_SCOPE')
    # Preserve every instruction, global, and other constant of the saved numerical method.
    replacement=tuple(epsilon if type(x) is float and x==.0005 else
        '_eps_'+epsilon.hex()+'_accounting.json' if x=='_accounting.json' else x for x in const)
    return FunctionType(fn.__code__.replace(co_consts=replacement),fn.__globals__,fn.__name__,fn.__defaults__,fn.__closure__)


def account(session,row,epsilon,permit,shot_fn):
    old.check(row,permit)
    c.require(row['system']=='H2' and row['subscope']=='B_H2_PRECISION' and row['epsilon']==.0005,'H2_ACCEPTED_BASE_ONLY')
    a=row['account_inputs']
    proxy=SimpleNamespace(s=SimpleNamespace(root=session.root),shots=lambda *args,**kw:session.call('u_aware_shots',shot_fn,*args,**kw))
    costs=[dict(task=p['pair_id'],**p['axes']) for p in row['pairs']]
    result=adapted_account(epsilon)(proxy,SimpleNamespace(id=row['candidate']),a['diagnostic'],a['allowance'],costs)
    out=deepcopy(row);out.update(epsilon=epsilon,status='ELIGIBLE' if result['eligible'] else result['shots']['eligibility_under_declared_allowance'],
        N={ax:result['shots']['axes'][ch]['shots'] for ax,ch in (('cosine','real'),('sine','imag'))},
        G_saved=result['G'],accounting=result,subscope='B_NEW_ENDPOINT')
    if not result['eligible'] and out['status']=='ELIGIBLE':out['status']='INELIGIBLE'
    # New N implies a new G-SE; do not silently quote the old epsilon's SE.
    out['uncertainty']=dict(G_SE=None,saved_axis_statistics=row['uncertainty'].get('saved_axis_statistics'),
        interpretation='SAVED_UNIT_STATISTICS_ONLY_NEW_G_SE_NOT_COMPUTED')
    out['source']=dict(kind='NEW_APPROVED_SAVED_INPUT_ACCOUNT',epsilon=epsilon,accepted_base_source=row['source'])
    return out


def integrate(quoted,new,same_q,data):
    main=deepcopy(quoted)+deepcopy(new)
    c.require(len(main)==30,'THIRTY_MAIN_CONTRASTS')
    matrix=[]
    for system in ('H2','H4','H6'):
        for epsilon in c.prior.EPS:
            pair={m:next(r for r in main if r['system']==system and r['epsilon']==epsilon and r['method']==m) for m in ('FULL','PR')}
            dirs={m:r.get('direction') for m,r in pair.items()}
            matrix.append(dict(system=system,epsilon=epsilon,scope='PRIMARY_DIFFERENT_Q',
                full=pair['FULL'],PR=pair['PR'],reversal=(dirs['FULL']!=dirs['PR']) if all(d in ('S2_LOWER','Y4_LOWER') for d in dirs.values()) else None,
                evidence='FIXED_EXPLORATORY_POINT_NOT_POPULATION_GUARANTEE',
                independent_standard_Y4_at_this_endpoint=False,H2_candidate_law_independent_support_only=system=='H2',
                weak_H4_005=system=='H4' and epsilon==.005))
    controls=[]
    for system in ('H4','H6'):
        for q in (4,8):
            rows={m:next(r for r in same_q if r['system']==system and r['method']==m and r['q']==q) for m in ('FULL','PR')}
            dirs=[rows[m].get('direction') for m in ('FULL','PR')]
            controls.append(dict(system=system,q=q,epsilon=.001,scope='AUXILIARY_SAME_Q',
                full=rows['FULL'],PR=rows['PR'],reversal=(dirs[0]!=dirs[1]) if all(d in ('S2_LOWER','Y4_LOWER') for d in dirs) else None,
                pure_order_causal_effect=None))
    return dict(main_15_system_precision_cells=matrix,same_q_4_system_q_cells=controls,
        n_by_candidate={t['id']:t['n'] for t in c.prior.TARGETS+c.A_TARGETS},
        new_H2_G_SE=None,new_H2_G_SE_reason='No new sample statistics; quote saved unit C statistics and main-epsilon G SE separately',
        cross_candidate_covariance_assumed=False,interpolation=False,cohort_pooling=False,global_candidate_ranking=False)


def pipeline(session,data,*,artificial=False,shot_module=None):
    c.validate(data,artificial=artificial)
    if artificial:permit=old.Permit(True)
    else:
        c.require(type(session) is c.Session and session.used['bootstrap']==1,'APPROVED_BATCH_SESSION_REQUIRED')
        permit=old.Permit(False,old._TOKEN)
    same_q=[]
    for a,b in zip(data['A_rows'][::2],data['A_rows'][1::2]):
        r=session.call('A_decomposition',old.decompose,a,b,permit)
        r.update(q=a['target']['q'],scope='AUXILIARY_SAME_Q');same_q.append(r)
    shot_module=shot_module or importlib.import_module('trottertracks.resource_applicability.ax2b_numerical_accounting')
    original_headroom=shot_module.axis_headroom
    shot_module.axis_headroom=lambda *a,**kw:session.call('axis_headroom',original_headroom,*a,**kw)
    try:
        endpoints=[session.call('B_account',account,session,r,ep,permit,shot_module.u_aware_shots) for ep in c.EPS for r in data['B_bases']]
    finally:shot_module.axis_headroom=original_headroom
    comparisons=[]
    for ep in c.EPS:
        for method in ('FULL','PR'):
            pair=[r for r in endpoints if r['epsilon']==ep and r['target']['method']==method]
            comparisons.append(session.call('B_decomposition',old.decompose,*pair,permit))
    integrated=session.call('integration_summary',integrate,data['quoted_main_comparisons'],comparisons,same_q,data)
    return dict(schema=c.SERIES+'_RESULT',same_q_comparisons=same_q,H2_new_endpoints=endpoints,
        H2_new_comparisons=comparisons,integration=integrated,old_main_comparisons_recomputed=False,
        limitations=c.LIMITS,strict_COMPARE='BLOCKED',science_GO=False,mandatory_STOP=True)
