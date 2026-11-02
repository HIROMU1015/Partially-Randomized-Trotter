"""Future fixed comparison only; no re-selection or new shot calculation.

Functions receive artificial data in readiness tests. Real data evaluation is
blocked until a matching future explicit execution grant and full end audit.
"""
from fractions import Fraction
import math
from pathlib import Path
import h246_confirmation_contract as c

def paired_statistics(pairs,ncos,nsin):
    c.require(len(pairs)>=2 and type(ncos) is int and type(nsin) is int and min(ncos,nsin)>0,'PAIRED_SAMPLE_AND_FROZEN_SHOTS')
    c.require(all(len(x)==2 and all(type(v) is int and v>0 for v in x) for x in pairs),'POSITIVE_INTEGER_AXIS_COSTS')
    n=len(pairs);xs=[x[0] for x in pairs];ys=[x[1] for x in pairs]
    meanx,meany=Fraction(sum(xs),n),Fraction(sum(ys),n)
    vx=sum((x-meanx)**2 for x in xs)/(n-1);vy=sum((y-meany)**2 for y in ys)/(n-1)
    cov=sum((x-meanx)*(y-meany) for x,y in pairs)/(n-1)
    values=[ncos*x+nsin*y for x,y in pairs];mean=Fraction(sum(values),n)
    variance=sum((g-mean)**2 for g in values)/(n-1)
    independent=ncos*ncos*vx+nsin*nsin*vy+2*ncos*nsin*cov
    c.require(variance==independent and mean==ncos*meanx+nsin*meany,'EXACT_PAIRED_COVARIANCE_CROSSCHECK')
    def exact(x):return dict(numerator=x.numerator,denominator=x.denominator)
    return dict(n=n,mean=float(mean),variance=float(variance),SE=math.sqrt(float(variance)/n),
      mean_exact=exact(mean),variance_exact=exact(variance),axis_means=[float(meanx),float(meany)],
      covariance=float(cov),independent_weighted_covariance_match=True,
      zero_observed_variance=variance==0,population_variance_bound=None,rare_tail_bound=None)

def summarize(comparison,observations):
    groups=[]
    for group,n in zip(comparison['groups'],comparison['recommended']['n_C']):
        key=group['candidate']['id'];items=observations[key]
        c.require(len(items)==n,'EXACT_FIXED_N_NO_TOPUP_OR_POOLING')
        stats=paired_statistics([r['pair'] for r in items],group['cost_row']['shot']['N_cos'],group['cost_row']['shot']['N_sin'])
        strata={s:sum(r['stratum']==s for r in items) for s in c.law.STRATA}
        c.require(sum(strata.values())==n,'ALL_EVENT_STRATA_RECORDED')
        groups.append(dict(candidate=group['candidate'],statistics=stats,observed_strata=strata,
          unobserved_strata=[s for s,k in strata.items() if not k],unobserved_costs=None))
    result=c.d.classify_confirmation(*(g['statistics'] for g in groups),z=comparison['recommended']['z'])
    result.update(relative_point_improvement=result['difference']/groups[0]['statistics']['mean'],
      groups=groups,population_guarantee=False,zero_observed_variance_warning=any(g['statistics']['zero_observed_variance'] for g in groups),
      interval='NORMAL_BONFERRONI_APPROXIMATION;NOT_RIGOROUS_CI',family_size=3,family_alpha=.05,
      ratio_interval=None,conditional_on_saved_signal_bias_normalization_and_shots=True,AD03='DEFERRED')
    return result

def evaluate(cfg):
    c.authorize(cfg);c.consumed(cfg);root=Path(cfg['run_root'])
    audit=c.read(root/'end_audit.json')
    c.require(audit['status']=='ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP' and not audit['errors']
      and audit['SOURCE']==cfg['SOURCE'] and audit['plan_sha256']==cfg['plan_sha256']
      and audit['completed_trajectories']==len(cfg['tasks']),'FULL_CONFIRMATION_END_AUDIT_BEFORE_STATISTICS')
    if cfg['stage']=='H2_CONFIRMATION':
        guard=c.read(root/'compiler_policy_end_audit.json')
        c.require(guard['status']=='PASS' and guard['end_audit_sha256']==c.sha(root/'end_audit.json'),'H2_PHASE_GUARD_AUDIT_REQUIRED')
    records={r['id']:r['result_sha256'] for r in audit['results']}
    marker=root/'COMPARISON_CONSUMED.json'
    c.save(marker,dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],grant_id=cfg['grant']['grant_id'],retry=False))
    try:
        obs={g['candidate']['id']:[] for g in cfg['comparison']['groups']}
        for t in cfg['tasks']:
            path=Path(cfg['output'])/t['id']/'result.json';r=c.read(path)
            c.require(c.sha(path)==records[t['id']] and r['task']==t and r['seed_cohort']==c.d.COHORT
              and r['SOURCE']==cfg['SOURCE'] and r['plan_sha256']==cfg['plan_sha256'],'CONFIRMATION_RESULT_SHA_IDENTITY')
            obs[t['candidate']['id']].append(dict(pair=[a['metrics']['rz_count'] for a in r['axis_costs']],stratum=r['events']['stratum']))
        result=summarize(cfg['comparison'],obs);c.verify(cfg)
        c.save(root/'fixed_comparison.json',dict(status='FIXED_CONFIRMATION_COMPARISON_COMPLETE',SOURCE=cfg['SOURCE'],
          plan_sha256=cfg['plan_sha256'],end_audit_sha256=c.sha(root/'end_audit.json'),**result,mandatory_STOP=True))
    except BaseException as exc:
        c.save(root/'comparison_STOP.json',dict(status='STOP',reason=str(exc),retry=False,mandatory_STOP=True));raise
