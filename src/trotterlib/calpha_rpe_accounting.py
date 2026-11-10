"""Raw-radius RPE accounting for the qualified fixed K0 builder law.

No attenuation is applied to an already raw lower radius. Statistical claims
are conditional on ideal gates and fresh independent bounded outcomes.
"""
from fractions import Fraction
import math


def account_raw_radius(radius,bounds):
    from .finite_rte_signal_validation import _hoeffding_shots
    from .calpha_interval_bounds import endpoint
    radius=Fraction(radius)
    if radius<=0:raise ValueError('POSITIVE_RAW_RADIUS_REQUIRED')
    c=bounds.c;alpha=Fraction(1,720)
    expression=4*c.ln(2/bounds.real(alpha))/(bounds.real(radius)**2*c.sin(bounds.real('0.24'))**2)
    high=endpoint(expression,True)
    enclosed_count=-(-high.numerator//high.denominator)
    helper=_hoeffding_shots(attenuation=1.,reference_radius=math.nextafter(float(radius),-math.inf),
        signal_error_bound=0.,beta_stat_budget=.24,alpha_axis=float(alpha))
    if helper is None:raise ValueError('EXISTING_HOEFFDING_HELPER_INELIGIBLE')
    count=max(helper,enclosed_count)
    return dict(N_X=count,N_Y=count,N_round=2*count,
        existing_helper_count=helper,interval_ceiling=enclosed_count,
        alpha_each_exact=str(alpha),attenuation_applied=1,
        radius_role='raw_lower_radius; no second RTE normalization')


def branch_sufficient_conditions(bounds):
    from .calpha_interval_bounds import endpoint
    c=bounds.c;beta=bounds.real('0.4');delta=bounds.real(.02)
    branch=c.pi-3*beta
    initial=c.pi-(128*delta+beta)
    energy=(beta/(delta*(1<<17))).b
    target=bounds.real(.00015936001019904)
    return dict(beta_total=.4,systematic_budget=.14,statistical_budget=.24,reserve=.02,
        energy_prior_Ha=[-128,128],phase_sign='phi=-delta*E',
        branch_margin_lower=float(endpoint(branch)),initial_alias_margin_lower=float(endpoint(initial)),
        final_energy_error_upper_exact=str(endpoint(energy,True)),
        final_energy_error_upper=float(endpoint(energy,True)),
        energy_precision_target=.00015936001019904,
        sufficient=endpoint(branch)>0 and endpoint(initial)>0 and endpoint(energy,True)<=endpoint(target),
        energy_oracle_used=False,proof='same nearest-branch induction: 3*beta<pi; prior base phase plus beta inside (-pi,pi)')


def evaluate_all_rounds(operands,core,bridge,law):
    import numpy as np
    from .calpha_interval_bounds import IntervalBounds,endpoint
    b=IntervalBounds(40);c=b.c
    layer=law['variants']['builder_finite_53bit_grid']
    if not all(x['satisfies_014'] for x in layer['rows']):
        raise ValueError('FIXED_LAW_ENDPOINT_GUARANTEE_REQUIRED')
    upper=lambda record:b.real(Fraction(record['upper_exact']))
    if [x['m'] for x in layer['rows']]!=[0,17]:raise ValueError('ENDPOINT_SOURCE_SCOPE')
    radius=b.real(Fraction(core['tail_norm_upper_exact']))
    rate=b.real(Fraction(core['rate_lower_exact']))
    gamma_upper=b.real(Fraction(core['gamma_upper_exact']))
    variance_upper=b.real(Fraction(core['variance_upper_exact']))
    delta=b.real(.02);theta=delta/4
    ident=c.eye(36)
    rt,damping=b.tail_log(b.matrix(operands['tail']),b.real(float(operands['lambda_R'])),theta,radius)
    generator=-c.j*rt-damping
    f=b.expm(delta*generator)
    for h in reversed(operands['heads']):
        head=-c.j*b.matrix(h);half=b.expm(delta*head/2)
        f=half*f*half;generator+=head
    defect=f-b.expm(delta*generator)
    psi=b.matrix([[x] for x in operands['psi']])
    normpsi=c.sqrt((b.dagger(psi)*psi)[0,0].real)
    # This field was a nearest binary64 display of an exact interval endpoint;
    # one nextafter upwards safely encloses its discarded final rounding.
    ground_distance=b.real(math.nextafter(core['ground_ray_distance_upper'],math.inf))
    state_rep=upper(core['rounds'][0]['values']['state_representation_upper'])
    model=upper(bridge['model_difference_upper']);connection=upper(bridge['macrostep_connection_upper'])
    raw=upper(bridge['raw_difference_upper']);law_step=upper(layer['macrostep_error'])
    rho=upper(bridge['macrostep_contraction_upper'])
    # Consume the exact inertia certificate, not the old numerical gap.
    gap=b.real(Fraction(operands['gap_lower_exact']))
    rows=[]
    def serialize(x):
        return dict(lower_exact=str(endpoint(x)),upper_exact=str(endpoint(x,True)),
                    lower_float=float(endpoint(x)),upper_float=float(endpoint(x,True)))
    for m in range(18):
        q=1<<m;t=q*delta
        if m in (0,17):
            # Reuse endpoint proofs/records, without operator or signal replay.
            cached=next(x for x in layer['rows'] if x['m']==m)
            d=upper(cached['disk_upper'])
            rec_radius=b.real(Fraction(cached['radius_lower']['lower_exact']))
            beta=upper(cached['phase_upper'])
            parts=dict(source='preserved qualified endpoint',law_connection=cached['law_connection'])
            a=c.exp(-gamma_upper*t).a
        else:
            pf=(b.weighted_state_pf(defect,generator,operands['eigenvalues'],operands['eigenvectors'],
                operands['coefficients'],psi,delta,rate,q)/normpsi.a+2*ground_distance).b
            variance=min([variance_upper*t*t/2,2*variance_upper*t/gap],key=lambda x:endpoint(x,True)).b
            arc=(t*theta**2*radius**3/3).b
            law_error=(law_step*(1-rho**q)/(1-rho)).b
            fields=dict(PF=pf,variance=variance,arctan=arc,state=state_rep,
                        model=t*model,CAR_connection=q*connection,raw=c.expm1(t*raw),law_connection=law_error)
            d=sum(fields.values(),c.mpf(0)).b
            a=c.exp(-gamma_upper*t).a
            rec_radius=(a-d).a
            beta=c.atan2(d,c.sqrt(a*a-d*d)).b if endpoint(d,True)<endpoint(a) else None
            parts={key:serialize(x) for key,x in fields.items()}
        good=beta is not None and endpoint(d,True)<=endpoint(a*c.sin(b.real('0.14')))
        if not good:raise ValueError('ROUND_SYSTEMATIC_GUARANTEE_FAILED:'+str(m))
        count=account_raw_radius(endpoint(rec_radius),b)
        rows.append(dict(m=m,q=q,T_Ha_inverse=float(.02*q),disk=serialize(d),
            raw_radius_lower=serialize(rec_radius),systematic_phase_upper=serialize(beta),
            satisfies_014=True,parts=parts,**count))
    branch=branch_sufficient_conditions(b)
    if not branch['sufficient']:raise ValueError('RPE_BRANCH_ALIAS_OR_ENERGY_CONDITION_FAILED')
    total_x=sum(x['N_X'] for x in rows);total_y=sum(x['N_Y'] for x in rows)
    return dict(status='ALL18_CONDITIONAL_GATE_MODEL_GUARANTEE_AND_SHOTS',rounds=rows,
        totals=dict(N_X=total_x,N_Y=total_y,N_all=total_x+total_y,
                    alpha_sum_exact=str(36*Fraction(1,720)),success_lower_bound_exact='19/20'),
        branch_conditions=branch,qualification='Interval bounds for fixed uncompiled ideal gate semantics and finite CDF law; fresh independent uniform-bit trajectories and +/-1 outcomes assumed.',
        compiled_native_and_device_guarantee='UNKNOWN',new_raw_signals=0,new_damped_signals=0,
        new_ground_state_solves=0,prices=None,total_cost=None)
