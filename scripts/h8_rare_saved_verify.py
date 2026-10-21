"""Separate Decimal raw-moment path, reading CSV rather than primary JSON rows.

No calls to primary integration/statistics. Same author, not a separate reviewer.
"""
import csv
import io
from decimal import Decimal as D, localcontext
from math import isqrt


def validate(b, b_csv, c_csv, result):
    with localcontext() as ctx:
        ctx.prec = 140
        def rat(s):
            parts = str(s).split('/')
            return D(parts[0])/D(parts[1]) if len(parts)==2 else D(parts[0])
        def close(actual, expected):
            assert abs(rat(actual)-expected) <= D('1e-85')*max(D(1),abs(expected)), 'SEPARATE_ARITHMETIC_MISMATCH'
        zeros = []
        for r in csv.DictReader(io.StringIO(b_csv)):
            if r['candidate'] != b['candidate']:
                continue
            assert r['stratum']=='S0_ONLY'
            g = sum(b['shots'][axis]*int(r['RZ_'+axis]) for axis in ('cosine','sine'))
            assert g==int(r['G_exact'])
            zeros.append(g)
        pairs = {}
        for r in csv.DictReader(io.StringIO(c_csv)):
            assert r['axis'] in ('cosine','sine')
            pair = pairs.setdefault(r['task'],{})
            assert r['axis'] not in pair
            pair[r['axis']] = r
        singles = []
        for pair in pairs.values():
            assert set(pair)=={'cosine','sine'}
            g=sum(b['shots'][a]*int(pair[a]['RZ']) for a in pair)
            assert all(g==int(r['paired_G']) for r in pair.values())
            singles.append(g)
        assert len(zeros)==48 and len(singles)==43
        means, variances = [], []
        for j, values in enumerate((zeros,singles)):
            n=len(values); s=sum(values); ss=sum(x*x for x in values)
            mean=D(s)/n
            variance=D(n*ss-s*s)/D(n*(n-1))
            close(result['known_layers'][j]['mean'],mean)
            close(result['known_layers'][j]['variance'],variance)
            close(result['known_layers'][j]['SE'],(variance/n).sqrt())
            means.append(mean); variances.append(variance)
        p = {g['stratum']:D(g['probability']) for g in b['groups']}
        p0,p1,pm,p4=[p[s] for s in ('S0_ONLY','S2_SINGLE','S2_MULTI','S4_PRESENT')]
        known=p0*means[0]+p1*means[1]
        vk=p0*p0*variances[0]/48+p1*p1*variances[1]/43
        se=vk.sqrt(); gap=D(b['fixed_PF_G_exact'])-known; mass=pm+p4
        # A separate integer square-root route checks Decimal sqrt itself.
        num,den=vk.as_integer_ratio(); scale=10**125
        sqrt_floor=D(isqrt(num*scale*scale//den))/D(scale)
        assert abs(sqrt_floor-se)<D('1e-120')
        checks={'K_known_G':known, 'estimated_variance_K_known_G':vk, 'SE_K_known_G':se,
                'remaining_budget_D_G':gap,'residual_mass':mass,
                'headroom_in_known_SE':gap/se,'K_known_over_PF':known/D(b['fixed_PF_G_exact']),
                'known_SE_over_PF':se/D(b['fixed_PF_G_exact']),
                'residual_mixture_mean_threshold_G':gap/mass,
                'residual_mixture_threshold_over_S0':gap/mass/means[0]}
        for key,value in checks.items(): close(result[key],value)
        for j,weight in enumerate((p0,p1)):
            close(result['known_layers'][j]['contribution_G'],weight*means[j])
            close(result['known_layers'][j]['contribution_SE_G'],weight*(variances[j]/(48 if j==0 else 43)).sqrt())
        old=p1+mass; oldgap=D(b['fixed_PF_G_exact'])-p0*means[0]
        prior=result['stage_B_comparison']
        for key,value in {'previous_unknown_mass':old,
                          'fraction_of_previous_unknown_mass_measured':p1/old,
                          'unknown_mass_reduction_factor':old/mass,
                          'fraction_of_previous_headroom_used':p1*means[1]/oldgap,
                          'single_mean_over_S0':means[1]/means[0]}.items():close(prior[key],value)
        line=result['two_layer_boundary']
        close(line['multi_intercept_G'],gap/pm);close(line['S4_intercept_G'],gap/p4)
        close(line['multi_intercept_over_S0'],gap/pm/means[0]);close(line['S4_intercept_over_S0'],gap/p4/means[0])
        close(line['p_MULTI_over_p_S4'],pm/p4)
        for i,point in enumerate(line['points']):
            t=D(i)/4
            close(point['mu_MULTI_G'],t*gap/pm); close(point['mu_S4_G'],(1-t)*gap/p4)
            close(point['mu_MULTI_over_S0'],t*gap/pm/means[0]);close(point['mu_S4_over_S0'],(1-t)*gap/p4/means[0])
            close(pm*D(point['mu_MULTI_G'])+p4*D(point['mu_S4_G']),gap)
        for k,row in enumerate(result['known_error_sensitivity']):
            for suffix,g in [('low',gap-k*se),('high',gap+k*se)]:
                close(row['residual_budget_'+suffix],g)
                close(row['mixture_threshold_'+suffix],g/mass)
                close(row['multi_intercept_'+suffix],g/pm)
                close(row['S4_intercept_'+suffix],g/p4)
            close(row['K_known_low'],known-k*se);close(row['K_known_high'],known+k*se)
        assert all(result['residual_means'][s] is None and result['residual_variances'][s] is None
                   for s in ('S2_MULTI','S4_PRESENT'))
        return dict(condition=b['condition'],status='PASS_SEPARATE_CSV_DECIMAL140_INTEGER_SQRT',
                    relative_absolute_tolerance='1e-85', means_variances_raw_integer_numerators_checked=True,
                    n_S0=48,n_SINGLE=43, primary_functions_called=False,
                    separate_reviewer=False, population_certificate=False)
