"""Stdlib-only frozen h schedule and finite-grid refinement policy."""
import math

NAMES=('original','response_selected','known_linear_extraction')
METHODS=tuple(f'{order}:{name}' for name in NAMES for order in ('S2','S4','CS2','EXT2'))

def initial_grid(h0,lam,settings,support=True):
    if not math.isfinite(h0+lam) or min(h0,lam)<=0:
        raise ValueError('INVALID_GRID_SCALE')
    end=settings['maximum_h'];ratio=settings['coarse_ratio']
    if end<=h0 or ratio<=1 or settings['initial_h_cap']<2:
        raise ValueError('INVALID_GRID_CONTRACT')
    p=1/(4-4**(1/3));K=lam*(4*p+abs(1-4*p))
    gap=math.pi/(8*K)
    coarse=[h0];x=h0
    while x<end:
        x=min(end,x*ratio)
        if x<=coarse[-1]:raise ValueError('GRID_STAGNATION')
        coarse.append(x)
    # Derivative ||U'(h)|| <= K includes the signed S4 stage absolute sum.
    bridges=[h0/2,h0]
    for left,right in zip(coarse[:-1],coarse[1:]):
        pieces=max(1,math.ceil((right-left)/gap))
        bridges.extend(left+(right-left)*j/pieces for j in range(1,pieces+1))
    grid=sorted(set(bridges+[x/2 for x in bridges if x/2>=h0/2]))
    if len(grid)>settings['initial_h_cap']:raise ValueError('INITIAL_GRID_CAP')
    if any(b-a>gap*(1+1e-12) for a,b in zip(grid[:-1],grid[1:])):
        raise ValueError('BRIDGE_GAP')
    return (grid if support else sorted(set(x for x in bridges if x>=h0))),K

def eligibility(record,target,contract):
    if record.get('ambiguous',True):return False,'BRANCH_OR_NUMERICAL_UNDETERMINED'
    limit=target*contract['evaluation']['budget_fractions']['deterministic']
    margin=record['numerical_allowance'];error=abs(record['bias'])
    if not all(math.isfinite(x) for x in (limit,margin,error,record['overlap'])) or margin<0:
        raise ValueError('NONFINITE_RECORD')
    if error+margin>=limit:
        return False,('NUMERICAL_BOUNDARY' if error-margin<limit else 'BIAS_TOO_LARGE')
    if record['overlap']-1e-10<contract['evaluation']['minimum_overlap']:
        return False,'OVERLAP_BOUNDARY'
    return True,'ELIGIBLE_DECLARED_DOUBLE_PRECISION_MARGIN'

def refinement_proposals(records,grid,contract,support_grid=None):
    table={(r['method'],float(r['h']).hex()):r for r in records}
    intervals={}
    for method in METHODS:
        present=[(h,table[method,h.hex()]) for h in grid if (method,h.hex()) in table]
        for (lo,a),(hi,b) in zip(present[:-1],present[1:]):
            if hi/lo-1<=contract['schedule']['boundary_relative_width']:continue
            triggers=[]
            for target in contract['evaluation']['energy_targets_Ha']:
                if eligibility(a,target,contract)[0]!=eligibility(b,target,contract)[0]:
                    triggers.append([method,target])
            if bool(a.get('ambiguous'))!=bool(b.get('ambiguous')):triggers.append([method,'branch'])
            if triggers:intervals.setdefault((lo,hi),[]).extend(triggers)
    ranked=sorted(intervals,key=lambda pair:(-(pair[1]/pair[0]-1),pair[0],pair[1]))
    selected=ranked[:contract['schedule']['refinement_intervals_per_round']]
    points=set();assessment=set()
    for lo,hi in selected:
        mid=math.sqrt(lo*hi)
        points.add(mid);assessment.add(mid)
        if mid/2>=contract['schedule']['h_anchor']/2:points.add(mid/2)
    points=sorted(points-set(grid if support_grid is None else support_grid))
    return points,{'selected_intervals':[{'left':a,'right':b,'triggers':intervals[a,b]} for a,b in selected],
                   'unselected_interval_count':max(0,len(ranked)-len(selected)),
                   'new_h':points,'assessment_new_h':sorted(assessment-set(grid)),
                   'rule':'union of all methods/targets;no monotonicity assumption'}

def boundary_status(record,records,target,contract):
    relevant=sorted((r for r in records if r['method']==record['method']),key=lambda r:r['h'])
    eligible=[r for r in relevant if eligibility(r,target,contract)[0]]
    if not eligible:return {'status':'NO_ELIGIBLE_POINT','resolved':False}
    best=max(eligible,key=lambda r:r['h'])
    following=[r for r in relevant if r['h']>best['h']]
    if not following:return {'status':'WINDOW_OR_VALID_BRANCH_UPPER_EDGE','resolved':False,'last_eligible_h':best['h']}
    nxt=following[0];width=nxt['h']/best['h']-1
    reason=eligibility(nxt,target,contract)[1]
    resolved=(width<=contract['schedule']['boundary_relative_width']*(1+1e-10)
              and reason=='BIAS_TOO_LARGE')
    return {'status':'SAMPLED_LOCAL_BOUNDARY_RESOLVED' if resolved else 'BOUNDARY_UNDETERMINED',
            'resolved':resolved,'left_h':best['h'],'right_h':nxt['h'],'relative_width':width,'right_reason':reason,
            'continuum_global_optimum_certified':False}
