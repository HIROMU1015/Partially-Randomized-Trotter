"""Preparation policy and ARTIFICIAL-only prototype; no production execution."""
from copy import deepcopy
from fractions import Fraction

AXES=('cosine','sine')
STRATEGIES=('NEW450_ONLY','ALL_SAVED458','MATCHED_NEW450_SUBSET')


def coverage(rows,allowed_ids):
    """Presence/identity checks only; never multiply shots or select a winner."""
    selected=[r for r in rows if r['candidate_id'] in allowed_ids]
    unresolved=[r['candidate_id'] for r in selected if r['eligible'] is None]
    eligible=[r for r in selected if r['eligible'] is True]
    missing_shots=[r['candidate_id'] for r in eligible if any(type(r['saved_shots'].get(a)) is not int or r['saved_shots'][a]<=0 for a in AXES)]
    missing_prices=[r['candidate_id'] for r in eligible if r['price_coverage']!='COMPLETE']
    if unresolved:status='BLOCKED_SAVED_ELIGIBILITY_UNRESOLVED'
    elif missing_shots:status='BLOCKED_SAVED_SHOTS_MISSING'
    elif missing_prices:status='BLOCKED_DIRECT_PRICE_MISSING'
    elif not eligible:status='NO_ELIGIBLE_CANDIDATE'
    else:status='CONDITIONAL_INPUTS_READY'
    return dict(status=status,registered=len(selected),eligible=len(eligible),
        ineligible=sum(r['eligible'] is False for r in selected),unresolved_candidate_ids=unresolved,
        missing_shot_candidate_ids=missing_shots,missing_direct_candidate_ids=missing_prices)


def synthetic_compare(payload):
    """The future arithmetic prototype accepts ARTIFICIAL inputs only.

    The preparation CLI never calls this with project values. Real-data wiring,
    sealing and execution need the user's separate scientific authorization.
    """
    if payload.get('kind')!='ARTIFICIAL':raise PermissionError('REAL_COST_REACCOUNTING_NOT_AUTHORIZED')
    data=deepcopy(payload)
    if set(data['baseline'])!=set(data['direct']):raise ValueError('SAME_CANDIDATE_SET_REQUIRED')
    def minimum(prices):
        totals={}
        for cid,row in prices.items():
            if row['eligible'] is None:raise ValueError('UNRESOLVED_FULL_DOMAIN')
            if row['eligible'] is False:continue
            if row['shots']!=data['baseline'][cid]['shots']:raise ValueError('SAVED_SHOTS_CHANGED')
            if row['eligible']!=data['baseline'][cid]['eligible']:raise ValueError('ELIGIBILITY_CHANGED')
            if set(row['shots'])!=set(AXES) or set(row['RZ'])!=set(AXES):raise ValueError('COMPLETE_AXIS_PAIR_REQUIRED')
            if any(type(n) is not int or n<=0 for n in row['shots'].values()):raise ValueError('SAVED_POSITIVE_SHOTS_REQUIRED')
            if any(row['RZ'][a] is None for a in AXES):raise ValueError('MISSING_NOT_ZERO')
            totals[cid]=sum(row['shots'][a]*Fraction(row['RZ'][a]) for a in AXES)
        order=sorted(totals,key=lambda cid:(totals[cid],cid))
        if not order:return dict(best=None,total=None,ties=[],order=[],totals={})
        best=order[0]
        return dict(best=best,total=str(totals[best]),ties=sorted(cid for cid in totals if totals[cid]==totals[best]),order=order,
            totals={cid:str(v) for cid,v in totals.items()},best_meta={k:prices[best].get(k) for k in ('q','prefix','PF','method')})
    # Check all rows before filtering ineligible rows.
    for cid,row in data['direct'].items():
        if (row['shots'],row['eligible'])!=(data['baseline'][cid]['shots'],data['baseline'][cid]['eligible']):
            raise ValueError('SAVED_POLICY_CHANGED')
    old=minimum(data['baseline']);new=minimum(data['direct'])
    pr=Fraction(data['saved_PR_total']) if data.get('saved_PR_total') is not None else None
    def ratio(result):return str(pr/Fraction(result['total'])) if pr is not None and result['total'] else None
    def relation(r):return None if r is None else 'PR_LOWER' if Fraction(r)<1 else 'PF_LOWER' if Fraction(r)>1 else 'TIE'
    before,after=ratio(old),ratio(new)
    inversions=[];tie_changes=[]
    eligible=sorted(set(old['totals'])&set(new['totals']))
    for i,left in enumerate(eligible):
        for right in eligible[i+1:]:
            a=Fraction(old['totals'][left])-Fraction(old['totals'][right])
            b=Fraction(new['totals'][left])-Fraction(new['totals'][right])
            if a*b<0:inversions.append([left,right])
            elif (a==0)!=(b==0):tie_changes.append([left,right])
    return dict(baseline=old,direct=new,PR_over_PF_before=before,PR_over_PF_after=after,
        best_candidate_changed=old['best']!=new['best'],ordering_changed=old['order']!=new['order'],
        pairwise_strict_inversions=inversions,pairwise_tie_changes=tie_changes,
        PR_PF_relation_before=relation(before),PR_PF_relation_after=relation(after),
        PR_PF_relation_changed=relation(before)!=relation(after),synthetic_only=True)


def reject_production():
    raise PermissionError('PREPARATION_ONLY_NO_REAL_SHOT_COST_OR_RANKING_EXECUTION')
