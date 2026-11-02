"""Saved C/N/G accounting only: no circuit, sampler, signal solver or shot model."""
from fractions import Fraction as F
import math

AXES = ('cosine', 'sine')


def summarize(values, critical):
    """Exact mean and sample variance; nominal t interval, never a rare-tail bound."""
    n = len(values)
    if n == 0:
        raise ValueError('Missing saved samples')
    mean = sum(map(F, values), F()) / n
    variance = sum((F(x) - mean)**2 for x in values) / (n-1) if n > 1 else F()
    se = math.sqrt(float(variance) / n)
    half = critical * se
    return dict(n=n, mean_exact=str(mean), mean=float(mean), SE=se, half95=half,
                lower95=float(mean)-half, upper95=float(mean)+half,
                interval_kind='NOMINAL_STUDENT_T' if n>1 else 'DETERMINISTIC_NO_MC',
                population_guarantee=False)


def ratio_interval(mean, half, lower, upper):
    lower, upper = F(lower), F(upper)
    if not 0 < lower <= upper:
        raise ValueError('Nonpositive/inverted theoretical cost')
    # Separate the theoretical endpoints from the nominal MC interval.
    return dict(lower=float(F(mean)/upper), upper=float(F(mean)/lower),
                nominal95_lower=(float(F(mean))-half)/float(upper),
                nominal95_upper=(float(F(mean))+half)/float(lower))


def account(row, role, prices, critical):
    v = row[role]
    if v.get('theory') is None or v.get('shots') is None or v.get('compiled') is None:
        raise ValueError('Missing comparable saved inputs')
    n = len(prices)
    if n != v['compiled']['n']:
        raise ValueError('Saved sample count mismatch')
    if role == 'PF' and n != 1:
        raise ValueError('PF must be deterministic')
    axis = {}
    for a in AXES:
        if any(type(p[a]) is not int or p[a] < 0 for p in prices):
            raise ValueError('Noninteger/negative RZ price')
        c = summarize([p[a] for p in prices], critical)
        if F(c['mean_exact']) != F(v['compiled'][a+'_mean']):
            raise ValueError('Compiled axis mean mismatch')
        axis[a] = dict(C_theory=v['theory'][a], C_circuit=c,
                       C_ratio=ratio_interval(c['mean_exact'],c['half95'],**{
                           k:v['theory'][a][k] for k in ['lower','upper']}),
                       N_theory=v['shots'][a], N_circuit=None, N_ratio=None)
    shots = v['shots']
    g00 = {b:str(sum((shots[a]*F(v['theory'][a][b]) for a in AXES),F()))
           for b in ['lower','upper']}
    # Paired axes are weighted BEFORE estimating variance; retain covariance.
    g01 = summarize([sum(shots[a]*p[a] for a in AXES) for p in prices], critical)
    for b in ['lower','upper']:
        if F(g00[b]) != F(row['G_'+role+'_theory_'+b]):
            raise ValueError('G00 differs from source')
    if F(g01['mean_exact']) != F(row['G_'+role+'_compiled']):
        raise ValueError('G01 differs from source')
    if not math.isclose(g01['half95'],v['compiled']['half95'],rel_tol=1e-12,abs_tol=1e-7):
        raise ValueError('Paired MC interval differs from source')
    contributions={a:str(shots[a]*(F(axis[a]['C_circuit']['mean_exact'])-
                                     F(axis[a]['C_theory']['lower']))) for a in AXES}
    return dict(system=row['system'],epsilon=row['epsilon'],role=role,
                candidate=row[role+'_candidate_id'],L_D=row[role+'_L_D'],q=row[role+'_q'],
                delta=0.8/row[role+'_q'],r=row['PR_r'] if role=='PR' else None,
                K=row['PR_K'] if role=='PR' else None,DF_rank=row['DF_rank'],
                cohort=row['cohort'],physical_status=row['status'],
                executed_technical_checks_complete=row['executed_technical_checks_complete'],
                preparation_hash=v['preparation_hash'],axes=axis,N_total=sum(shots.values()),
                N_circuit_total=None,N_ratio_total=None,N_independent=False,
                N_status='NO_INDEPENDENT_COMPILED_DISTRIBUTION_SHOT_EVALUATION',
                G00=g00,G01=g01,G10=None,G11=None,
                G01_over_G00=ratio_interval(g01['mean_exact'],g01['half95'],**g00),
                delta_G_C_lower_reference=str(F(g01['mean_exact'])-F(g00['lower'])),
                delta_G_C_by_axis_lower_reference=contributions,delta_G_N=None,
                total_comparison_label='N_FIXED_C_ONLY_REPLACEMENT',
                MC_n=n,MC_kind='HISTORICAL_EXPLORATION_N8' if row['system']=='H2' and role=='PR'
                else 'INDEPENDENT_MAIN' if role=='PR' else 'DETERMINISTIC',
                rare_event_diagnostics_ref='../../pr_pf_s2_recomputed_20261022/rare_event_summary.json',
                compiler_only_causal_effect=False,selection_effect_corrected=False)
