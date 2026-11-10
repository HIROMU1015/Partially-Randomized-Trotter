"""H6 coefficient-prior/D1 adapter. No saved inputs are loaded on import.

Real H6 prior, phase, N and partial-cost evaluations require the separately
specified scientific execution scope. Tests use artificial coefficients.
"""
from copy import deepcopy
import math

from . import h4_df10_rpe_d1 as scalar
from . import h6_df19_rpe_connection as connection

PREPARATION = '99ad77ef949b70192918a8700b092dd33c17af8c'
IDS = tuple(connection.D1_CANDIDATES)

# Exact fixed H4 function: no duplicate prior implementation or E0 parameter.
coefficient_prior = scalar.coefficient_prior


def load_coefficient_payload(snapshot):
    """Decode saved coefficients and occupation indices, never a state array."""
    import numpy as np
    with np.load(snapshot, allow_pickle=False) as z:
        h=z['one_body'].tolist();g=z['g_matrices'].tolist()
        weights=z['lambdas'].tolist();constant=float(z['constant'].reshape(-1)[0])
        indices=z['sector_basis_indices']
        if len(h)!=12 or len(weights)!=19 or len(g)!=19 or len(indices)!=400:
            raise ValueError('H6 DF19/12-mode/400-sector coefficient schema required')
        mask=int(indices[0])
    occupied=[i for i in range(12) if mask>>i&1]
    if (mask<0 or mask>=1<<12 or len(occupied)!=6
            or sum(i%2==0 for i in occupied)!=3):
        raise ValueError('Fixed first determinant must have Nalpha=Nbeta=3')
    return constant,h,weights,g,occupied


def configuration(prior, *, energy_ha=scalar.CHEMICAL_ACCURACY_HA):
    """Try ONLY h=.8 and .1; both failing returns STOP, never a smaller h."""
    attempts=[]
    for h in (.8,.1):
        c=connection.prepare_conditions(base_time=h,energy_ha=energy_ha,
            alpha_total=.05,beta_total=.4,beta_stat=.2,energy_prior=prior,
            prior_provenance='Saved DF coefficients: exact-rational Gershgorin/signed-square and fixed-sector determinant')
        attempts.append(dict(base_time=h,status=c['status']))
        if c['status']=='CONDITIONAL_RPE_POLICY':
            count=c['horizon']['round_count_M_plus_one']
            return dict(status='CONDITIONAL_D1_CONFIGURATION',base_time=h,
                energy_precision_ha=energy_ha,beta_total=.4,beta_stat=.2,
                alpha_total_per_method=.05,alpha_axis=c['alpha_axis'],
                alpha_sum_all_rounds=math.fsum([c['alpha_axis']]*(2*count)),
                horizon=c['horizon'],conditions=c,prior_attempts=attempts,
                energy_prior_ha=list(prior),energy_shift=0,
                decoder='consecutive_dyadic_nearest_branch_principal_energy_v1')
    return dict(status='STOP_NO_ALLOWED_BASE_TIME',prior_attempts=attempts,
                base_time=None,horizon=None,energy_prior_ha=list(prior))


def hydrate_saved_allowances(point, endpoint):
    """Copy original per-axis allowances omitted by the compact projection."""
    if endpoint['candidate']!=point['candidate']:
        raise ValueError('Endpoint differs from the fixed saved candidate')
    p=deepcopy(point)
    p['saved_finite_time_endpoint']=dict(axes={
        a:dict(allowance=endpoint['axes'][a]['allowance']) for a in ('real','imag')})
    return p


def bind(point, config):
    if config['status']!='CONDITIONAL_D1_CONFIGURATION':
        raise ValueError('No allowed prior/base-time configuration')
    h=config['base_time'];T=point['candidate']['T']
    matches=[m for m in range(config['horizon']['maximum_round_index_M']+1)
             if math.ldexp(h,m)==T]
    if len(matches)!=1:
        raise ValueError('Saved time does not match the selected finite horizon')
    return connection.bind_saved_round(point,
        target_id='H6_DF19_GROUND_CA_CONDITIONAL_D1',m=matches[0],base_time=h)


def analyze_point(point, config, *, state_energy, state_residual):
    row=bind(point,config)
    result=scalar.analyze_bound_point(point,row,config,
        state_energy=state_energy,state_residual=state_residual)
    for lane in ('ideal_numerical_pilot','actual_implementation'):
        result[lane].update(energy_target='saved_DF_H_ground_energy',
            energy_ha=config['energy_precision_ha'],
            prior_provenance=config['conditions']['prior_provenance'])
    return result


def partial_resources(points, rows, config):
    """Frozen direct prices; PF/PR and PR cohorts are separate output tables."""
    if len(points)!=4 or {p['candidate']['id'] for p in points}!=set(IDS):
        raise ValueError('Exactly the four fixed H6 D1 points required')
    byid={r['candidate_id']:r for r in rows}
    if len(rows)!=4 or len(byid)!=4 or set(byid)!=set(IDS):
        raise ValueError('Exactly four analyzed candidate rows required')
    results=[]
    for method in ('FULL','PR'):
        selected=sorted((p for p in points if p['candidate']['method']==method),
                        key=lambda p:p['candidate']['T'])
        plans=[byid[p['candidate']['id']]['ideal_numerical_pilot'] for p in selected]
        cohorts=('DETERMINISTIC',) if method=='FULL' else ('EXPLORATION','CONFIRMATION')
        for cohort in cohorts:
            prices=[]
            for p in selected:
                chosen=p['prices'][0]['cohort'] if method=='FULL' else cohort
                if method=='FULL' and len(p['prices'])!=1:
                    raise ValueError('One saved deterministic price per PF point required')
                prices.extend(connection.supplied_price_axes(bind(p,config),cohort=chosen))
            out=connection.aggregate_partial(plans,prices,
                conditions=config['conditions'],compiler=selected[0]['compiler'])
            results.append(dict(out,method='PF' if method=='FULL' else 'PR',
                cohort=cohort,actual_totals=None,
                direct_price_samples=[dict(m=p['m'],axis=p['axis'],n=p['n'],
                    classification=p['classification'],source=p['source']) for p in prices],
                population_expectation_guaranteed=False))
    return results
