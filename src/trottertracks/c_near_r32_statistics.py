"""Affine comparison adapter: existing calibration + SciPy Welch interval.

One prespecified PR-minus-ENTRY RZ comparison. No ratio interval from division
by a random estimated denominator, no optional sample extension or support bound.
"""
import math
from . import c_r32_price_statistics as shared

POLICY=dict(samples=32,confidence=.95,primary_metric='rz_count',primary_lane='ideal_exact_native',
    method='scipy_Welch_fixed_N_independent_cohorts_affine_total_difference',
    precision_halfwidth_fraction_of_ENTRY=.05,precision_halfwidth_fraction_of_D=.05,
    skewness_abs_threshold=2.,maximum_delete_one_shift_fraction_of_halfwidth=.25,
    old_samples_pooled=False,adaptive=False,interim_inference=False,
    family='one primary PR-minus-ENTRY comparison;per-arm PR/D intervals individual secondary only')


def affine(packet,result):
    """Reuse serial unit-price composition to obtain W and shells per lane."""
    m3,m4=shared.ret.m3,shared.ret.m4
    out={}
    for lane,v in result['lanes'].items():
        if not v['energy']['same_accuracy'] or lane=='evidence':
            out[lane]=None;continue
        plans=v['rounds'];target=packet['target_id']
        kernel=dict(target_id=target,mean={k:1 for k in m4.METRICS},policy='native_kernel_repeat_with_fences',simultaneous_interval_qualified=False)
        costs=[m3.serial_cost(packet,kernel,m,axis) for m in range(13) for axis in ('X','Y')]
        totals=m4.accuracy.aggregate_totals(v['energy'],plans,costs,cost_scope=shared.ret.accounting.COST_SCOPE)['totals']
        W=v['weighted_SE']['kernel_reuse_weight']
        if W!=sum(2*p['shots']*(1<<p['m']) for p in plans):raise ValueError('W_SINGLE_OCCURRENCE_ACCOUNTING')
        shell={k:totals[k]-W for k in m4.METRICS}
        if any(not math.isfinite(x) or x<0 for x in shell.values()):raise ValueError('SHELL_ACCOUNTING')
        out[lane]=dict(W=W,shell=shell,shots=v['total_shots'])
    return out


def arm_summary(rows,bindings,deterministic,*,calibration=None,call=lambda k,l,f:f(),expected_n=32,statistics=None):
    """Existing one-sample PR/D inference and descriptive all-metric totals."""
    policy=POLICY if statistics is None else statistics
    shared.validate_rows(rows,expected_n)
    cal=shared.ret.perf.b2.calibration(rows) if calibration is None else calibration
    results={}
    for lane,b in bindings.items():
        if b is None:results[lane]=None;continue
        value=dict(qualification='CONDITIONAL_FINITE_BIT_PRICE_PROXY',samples=expected_n,
            totals={k:b['W']*cal['sample_mean'][k]+b['shell'][k] for k in shared.ret.m4.METRICS},
            SE={k:b['W']*cal['sample_mean_standard_error'][k] for k in shared.ret.m4.METRICS})
        d_lane='ideal_exact_native' if lane=='ideal_exact_native' else 'numerical'
        d=None if deterministic is None else deterministic.get(d_lane)
        if d and d.get('cost',{}).get('totals') is not None:
            value['same_JU_D']=d['cost']['totals']
            value['point_ratios']={k:value['totals'][k]/value['same_JU_D'][k] if value['same_JU_D'][k] else None for k in shared.ret.m4.METRICS}
            value['RZ_PR_D_nominal']=call('price_statistics',lane,lambda:shared.price_statistics(rows,dict(b,D=value['same_JU_D']),expected_n=expected_n,statistics=policy))
        else:value.update(same_JU_D=None,point_ratios=None,RZ_PR_D_nominal=None)
        results[lane]=value
    return dict(calibration=cal,lanes=results,actual_CA_achieved=None,ideal_law_price_transfer_qualified=False)


def primary(rows_pr,rows_entry,affine_pr,affine_entry,*,calibrations=None,expected_n=32,statistics=None):
    """One approximate IID-model difference interval; no validated coverage."""
    import numpy as np
    from scipy.stats import t,ttest_ind
    policy=POLICY if statistics is None else statistics
    if type(expected_n) is not int or expected_n<2 or not 0<policy['confidence']<1:raise ValueError('INVALID_FIXED_N_CONFIDENCE')
    for rows in (rows_pr,rows_entry):shared.validate_rows(rows,expected_n)
    cs=[shared.ret.perf.b2.calibration(rows) for rows in (rows_pr,rows_entry)] if calibrations is None else calibrations
    bs=[affine_pr,affine_entry]
    estimates=[b['W']*c['sample_mean']['rz_count']+b['shell']['rz_count'] for b,c in zip(bs,cs)]
    out=dict(samples_per_arm=expected_n,total_points=dict(PR=estimates[0],ENTRY=estimates[1]),
        point_ratio_PR_ENTRY=estimates[0]/estimates[1],difference_point=estimates[0]-estimates[1],
        difference_interval=None,ratio_interval=None,confidence=policy['confidence'],method=policy['method'],
        interval_kind='APPROXIMATE_FIXED_N_IID_MODEL_NOT_VALIDATED',actual_CA_achieved=None,
        guaranteed_resource_superiority=None,ideal_law_price_transfer_qualified=False,
        simultaneous_secondary_superiority=None,observed_extrema_are_support=False)
    if any(c['sample_mean_standard_error']['rz_count']<=0 for c in cs):
        return dict(out,decision='UNRESOLVED_ZERO_COHORT_VARIANCE',precision_met=False)
    diagnostics={}
    for name,rows,c in zip(('PR','ENTRY'),(rows_pr,rows_entry),cs):
        se=c['sample_mean_standard_error']['rz_count']
        half=float(t.ppf((1+policy['confidence'])/2,expected_n-1))*se
        diagnostics[name]=shared.distribution_diagnostics(rows,c['sample_mean']['rz_count'],se,half,policy=policy)
    # Existing SciPy owns Welch degrees of freedom, CI and test implementation.
    values=[np.asarray([b['W']*row['rz_count']+b['shell']['rz_count'] for row in rows],dtype=float)
        for b,rows in zip(bs,(rows_pr,rows_entry))]
    test=ttest_ind(*values,equal_var=False,nan_policy='raise')
    ci=test.confidence_interval(policy['confidence'])
    interval=[float(ci.low),float(ci.high)]
    if not all(math.isfinite(x) for x in interval):raise ValueError('NONFINITE_WELCH_INTERVAL')
    half=(interval[1]-interval[0])/2
    precise=half/estimates[1]<=policy['precision_halfwidth_fraction_of_ENTRY']
    warning=any(x['influence_warning'] or x['skewness_warning'] for x in diagnostics.values())
    decision=('UNRESOLVED_DISTRIBUTION_DIAGNOSTIC' if warning else 'UNRESOLVED_PRECISION_TARGET' if not precise
        else 'NOMINAL_MODEL_PR_LOWER' if interval[1]<0 else 'NOMINAL_MODEL_ENTRY_LOWER' if interval[0]>0
        else 'UNRESOLVED_SIGN_NOT_EQUIVALENCE')
    return dict(out,difference_interval=interval,Welch_df=float(test.df),diagnostics=diagnostics,
        halfwidth=half,precision_met=precise,decision=decision)
