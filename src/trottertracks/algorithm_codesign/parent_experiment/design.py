"""Deterministic finite candidates, global proposals, rational sufficient ledger.

No fixture, cache, provider, backend or registered values at import. This module
also supports off-domain artificial validation without executing a science job.
"""
from fractions import Fraction as F
from math import factorial
from ..return_aggregation import root_interval
from ..parent_allocation import rational, reweight_events, event_statistics
from ..g8_bounds import ceil, log_upper
from ..rte_development.accounting import affine_relation


def kkt_candidate(a, b, costs, *, iterations=128, root_bits=256, share_bits=64):
    a = rational(a); b = tuple(map(rational,b)); costs = tuple(map(rational,costs))
    if len(b) != len(costs) or not b:
        raise ValueError('one full-event frozen proxy price per child required')
    if a <= 0 or min(b) <= 0 or min(costs) <= 0:
        return {'status':'MATHEMATICALLY_NOT_APPLICABLE','allocation':None,
                'reason':'strictly positive a, b and proxy prices required'}
    if any(type(v) is not int or v < 1 for v in (iterations,root_bits,share_bits)):
        raise ValueError('fixed positive integer resolutions required')
    lo, hi = F(0), min(costs); performed=0
    def enclosure(lam):
        out=[]
        for bi,ci in zip(b,costs):
            rt=root_interval(ci*ci-lam*lam,root_bits)
            if rt.lo <= 0: raise ArithmeticError('fixed KKT root resolution exhausted')
            out.append((lam*bi/rt.hi,lam*bi/rt.lo))
        return out
    for _ in range(iterations):
        mid=(lo+hi)/2; bounds=enclosure(mid); performed+=1
        lower=sum(v[0] for v in bounds); upper=sum(v[1] for v in bounds)
        if upper < a: lo=mid
        elif lower > a: hi=mid
        else:
            # Interval overlap does not license a directional update.
            # Keep the previous certified bracket; no precision rescue.
            break
    if hi >= min(costs): raise ArithmeticError('KKT root not bracketed below pole')
    at_lo=enclosure(lo); at_hi=enclosure(hi)
    z_bounds=[(l[0],h[1]) for l,h in zip(at_lo,at_hi)]
    mid_bounds=enclosure((lo+hi)/2)
    mids=[(v[0]+v[1])/2 for v in mid_bounds]; total=sum(mids)
    scale=1<<share_bits; exact=[v/total*scale for v in mids]
    counts=[v.numerator//v.denominator for v in exact]
    order=sorted(range(len(b)),key=lambda i:(-(exact[i]-counts[i]),i))
    for i in order[:scale-sum(counts)]: counts[i]+=1
    z=tuple(a*F(n,scale) for n in counts)
    distance=max(max(abs(zi-l),abs(zi-h)) for zi,(l,h) in zip(z,z_bounds))
    return dict(status='RATIONAL_SURROGATE_CANDIDATE',allocation=z,
                lambda_interval=(lo,hi),ideal_z_intervals=z_bounds,
                L_infinity_distance_upper=distance,
                simplex_residual=sum(z)-a,iterations_performed=performed,
                fixed_share_denominator=scale,exact_KKT_optimum=False,
                proxy_scope='old complete common root event T; no new-angle prices')


def candidates(a,b,proxy_costs,policy):
    a=rational(a); b=tuple(map(rational,b))
    if a <= 0 or not b or min(b) <= 0: raise ArithmeticError('central root domain')
    common=tuple(a*v/sum(b) for v in b)
    raw=[('common',common)]
    raw += [('vertex_'+str(i),tuple(a if i==j else F(0) for j in range(len(b))))
            for i in range(len(b))]
    kkt=kkt_candidate(a,b,proxy_costs,**policy)
    if kkt['allocation'] is not None:
        z=kkt['allocation'];raw += [('kkt',z),('halfway',tuple((u+v)/2 for u,v in zip(common,z)))]
    unique=[]; dropped=[];seen={}
    for label,z in raw:
        if z in seen: dropped.append(dict(label=label,duplicate_of=seen[z]))
        else: seen[z]=label;unique.append(dict(label=label,allocation=z))
    return dict(candidates=unique,dropped=dropped,kkt=kkt,
                replacement_candidates=False,new_prices_used=False)


def proposals(events, costs, scale, precision):
    scale=rational(scale); costs=tuple(map(rational,costs))
    if scale < 0: raise ValueError('negative shared scale')
    modes=[('canonical','canonical',None)]
    dropped=[]
    if min(costs)>0: modes.append(('analytic_IS','cost_aware',None))
    else: dropped.append(dict(label='analytic_IS',reason='zero T cost in positive support'))
    if scale>0:
        modes += [('regularized_'+str(f),'regularized',scale*f) for f in (F(1,16),F(1),F(16))]
    else: dropped.append(dict(label='regularized_all',reason='C_star=0; no artificial floor'))
    laws=[];seen={}
    for label,mode,kappa in modes:
        law=reweight_events(events,mode=mode,
            costs=costs if mode!='canonical' else None,regularization=kappa,**precision)
        signature=tuple(e['proposal'] for e in law)
        if signature in seen:
            dropped.append(dict(label=label,duplicate_of=seen[signature]));continue
        seen[signature]=label; laws.append(dict(label=label,kappa=kappa,events=law))
    return laws,dropped


def ledger(events, prices, coefficient_error_upper, c):
    """Exact digital law; interval sqrt/log only use fixed outward certificates."""
    if not events or len(events)!=len(prices): raise ValueError('complete positive event coverage')
    stats=event_statistics(events)
    coeff=rational(coefficient_error_upper)
    if coeff<0: raise ValueError('nonnegative certified coefficient error required')
    for price in prices:
        if any(type(price[k]) is not int or price[k]<0 for k in ('T','CX','1Q')):
            raise ValueError('missing or invalid actual native price')
        if rational(price['strict_native_error_upper'])<0: raise ValueError('negative error')
    native=sum(rational(e['coefficient'])*rational(p['strict_native_error_upper']) for e,p in zip(events,prices))
    tay=abs(F(c['input']['tau']))**6/factorial(6)
    beta=2*(tay+coeff+native); delta=F(c['epsilon_axis'])-beta
    per={k:sum(F(e['proposal'])*p[k] for e,p in zip(events,prices)) for k in ('T','CX','1Q')}
    worst={k:max(p[k] for p in prices) for k in per}
    out=dict(**stats,Taylor_upper=tay,coefficient_error_upper=coeff,
             weighted_native_error_upper=native,operator_rep_error_upper=coeff+native,
             axis_bias_upper=beta,statistical_remaining=delta,per_trial_native=per,
             worst_event_native=worst,alpha_axis=F(c['alpha_axis']),
             native_angle_error_already_in_strict_guard=True,
             probability_rounding_mean_bias=F(0),quantum_shots_executed=0,
             online_classical_cost='NOT_MEASURED',minimum_shot_claim=False)
    if delta<=0: return dict(out,eligibility='BIAS_INELIGIBLE',N_per_axis=None)
    n=ceil(log_upper(2/F(c['alpha_axis']))*(2*stats['weight_second_moment']/delta**2+4*stats['range']/(3*delta)))
    if n>c['caps']['shot_cap_per_axis']:
        return dict(out,eligibility='SHOT_CAP_INELIGIBLE',N_per_axis=n)
    return dict(out,eligibility='ELIGIBLE',N_per_axis=n,
                expected_native_intercept={k:2*n*v for k,v in per.items()},
                prep_readout_slope=2*n,hard_native={k:2*n*v for k,v in worst.items()},
                expected_total_quantum_cost_is_not_high_probability_cap=True)


def comparisons(rows):
    eligible=[r for r in rows if r['ledger']['eligibility']=='ELIGIBLE']
    def best(items):
        return min(items,key=lambda r:(r['ledger']['expected_native_intercept']['T'],r['id'])) if items else None
    mechanisms=[]
    for arm in ('ordinary','closed_P5_full'):
        common=best([r for r in eligible if r['arm']==arm and r['candidate']=='common'])
        non=best([r for r in eligible if r['arm']==arm and r['candidate']!='common'])
        difference=(non['ledger']['expected_native_intercept']['T']-common['ledger']['expected_native_intercept']['T']) if common and non else None
        mechanisms.append(dict(arm=arm,common=common['id'] if common else None,
            noncommon=non['id'] if non else None,difference=difference,
            accounting_relation=('NONCOMMON_LOWER_UNDER_FIXED_SUFFICIENT_POLICY' if difference is not None and difference<0 else
                                 'NO_STRICT_REGISTERED_ALLOCATION_GAIN' if difference is not None else 'NO_COMPARABLE_NONCOMMON'),
            scientific_GO=False,continuous_family_no_go=False))
    relations=[]
    for i,left in enumerate(eligible):
        for right in eligible[i+1:]:
            for metric in ('T','CX','1Q'):
                l,r=left['ledger'],right['ledger']
                relations.append(dict(left=left['id'],right=right['id'],metric=metric,
                    **affine_relation(dict(intercept=l['expected_native_intercept'][metric],slope=l['prep_readout_slope']),
                                      dict(intercept=r['expected_native_intercept'][metric],slope=r['prep_readout_slope']))))
    winner=best(eligible)
    return dict(mechanisms=mechanisms,affine_relations=relations,
                finite_registered_best=winner['id'] if winner else None,
                scientific_adoption=False,priority_established=False,materiality_threshold=None)


def integer_io_limit(digits):
    """Bounded decimal I/O for exact global-IS fractions, not solver precision.

    Dyadic probabilities have unrelated count numerators; the exact moment
    denominator can exceed CPython's default 4300-digit conversion limit.
    A context restores the process policy even on failure.
    """
    from contextlib import contextmanager
    import sys
    @contextmanager
    def context():
        if type(digits) is not int or digits<640:raise ValueError('bounded integer decimal cap')
        old=sys.get_int_max_str_digits() if hasattr(sys,'get_int_max_str_digits') else None
        try:
            if old is not None:sys.set_int_max_str_digits(digits)
            yield
        finally:
            if old is not None:sys.set_int_max_str_digits(old)
    return context()
