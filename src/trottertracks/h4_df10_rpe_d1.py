"""Approved D1 scalar analysis: coefficient prior and TWO saved T=.8 points.

No molecular construction, diagonalization, state-vector load, signal evolution,
trajectory, circuit, fit, resource sum or H6 calculation is performed here.
"""
import cmath
from fractions import Fraction
import math

from . import c1_stage3_accuracy as existing
from . import h4_df10_rpe_connection as connection

CHEMICAL_ACCURACY_HA = 1.59360010199040e-3  # existing config.CA, not TARGET_ERROR
IDS = ('49709717a6c3d405c60d', '14bd81a375400c9b0305')
PREPARATION = 'e91021ae2db0fba755ba7a6404c18e7e003c129f'


def _f(x):
    return Fraction.from_float(float(x))


def _hermitian_rows(matrix):
    """Exact-rational Gershgorin bounds; |Re|+|Im| bounds each off-diagonal."""
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError('Square saved coefficient matrix required')
    lows, highs = [], []
    for i in range(n):
        radius = Fraction(0)
        for j in range(n):
            z = complex(matrix[i][j])
            if not math.isfinite(z.real) or not math.isfinite(z.imag):
                raise ValueError('Nonfinite saved coefficient')
            if z != complex(matrix[j][i]).conjugate():
                raise ValueError('Prior requires the saved Hermitian representation')
            if i != j:
                radius += abs(_f(z.real)) + abs(_f(z.imag))
        diagonal = _f(complex(matrix[i][i]).real)
        lows.append(diagonal-radius); highs.append(diagonal+radius)
    return min(lows), max(highs)


def coefficient_prior(constant, one_body, weights, fragments, occupied):
    """Ground prior from saved coefficients; NO state energy/E0 argument.

    H=c+dGamma(h)+sum lambda*dGamma(G)^2, in a fixed particle/spin sector.
    Lower: N-particle Gershgorin intervals, then signed square intervals.
    Upper: one predeclared occupation determinant, using
      <dGamma(G)^2>=(sum_occ G_pp)^2+sum_(p occ,q empty)|G_qp|^2.
    This evaluates scalar formulas; no Fock Hamiltonian or trial vector is built.
    """
    n = len(one_body); occupied = tuple(occupied)
    if len(set(occupied)) != len(occupied) or not occupied or any(p not in range(n) for p in occupied):
        raise ValueError('Valid existing-sector occupation mask required')
    if len(weights) != len(fragments):
        raise ValueError('Fragment/weight mismatch')
    ne = len(occupied); a, b = _hermitian_rows(one_body)
    lo = _f(constant)+ne*a; hi = _f(constant)+ne*b
    trial = _f(constant)+sum((_f(complex(one_body[p][p]).real) for p in occupied),Fraction(0))
    diagnostics = []
    empty = tuple(q for q in range(n) if q not in occupied)
    for weight, g in zip(weights, fragments):
        if len(g) != n:
            raise ValueError('Coefficient dimension mismatch')
        w = _f(weight); a, b = _hermitian_rows(g); a *= ne; b *= ne
        sqlo = Fraction(0) if a <= 0 <= b else min(a*a,b*b)
        sqhi = max(a*a,b*b)
        contributions = (w*sqlo,w*sqhi)
        lo += min(contributions); hi += max(contributions)
        trace = sum((_f(complex(g[p][p]).real) for p in occupied),Fraction(0))
        second = trace*trace
        for p in occupied:
            for q in empty:
                z = complex(g[q][p]); second += _f(z.real)**2 + _f(z.imag)**2
        trial += w*second
        diagnostics.append(dict(weight=float(weight), generator_interval=[float(a),float(b)],
                                square_interval=[float(sqlo),float(sqhi)]))
    upper = min(hi,trial)
    if lo > upper:
        raise ValueError('Coefficient bounds are inconsistent')
    # Fraction -> nearest float -> one outward step preserves containment.
    lower_float = math.nextafter(float(lo),-math.inf)
    upper_float = math.nextafter(float(upper),math.inf)
    assert _f(lower_float) <= lo and _f(upper_float) >= upper
    return dict(energy_prior_ha=[lower_float,upper_float],
        exact_rational_lower=str(lo), exact_rational_upper=str(upper),
        full_spectral_interval_ha=[float(lo),float(hi)],
        determinant_variational_upper_ha=float(trial), occupied_modes=list(occupied),
        particle_number=ne, fragments=diagnostics, uses_state_energy_or_E0=False,
        qualification='ANALYTIC_COEFFICIENT_BOUND_FOR_SAVED_HERMITIAN_DF_MODEL',
        full_hamiltonian_assembled=False, trial_state_vector_built=False)


def configuration(prior):
    """Select h using the EXISTING principal-decoder condition, then horizon."""
    lo, hi = prior; beta_total, beta_stat, alpha_total = .4,.2,.05
    attempted=[]; selected=None
    # User preference: .8 first, .1 second; smaller dyadic alternatives only if needed.
    candidates=(.8, *(.1/(1 << k) for k in range(32)))
    for h in candidates:
        margin = math.pi-max(abs(lo),abs(hi))*h-beta_total
        attempted.append(dict(base_time=h,principal_alias_margin=margin,qualified=margin>0))
        if margin>0:
            selected=h;break
    if selected is None:
        raise ValueError('No principal energy alias interval at available scalar precision')
    horizon=existing._legacy('rpe_target_round_horizon_validation').required_rpe_round_horizon(
        target_energy_precision=CHEMICAL_ACCURACY_HA,beta_rpe=beta_total,delta_time=selected)
    count=horizon['round_count_M_plus_one'];alpha_axis=alpha_total/(2*count)
    if math.fsum([alpha_axis]*(2*count))>alpha_total:
        alpha_axis=math.nextafter(alpha_axis,0.)
    ms=[m for m in range(horizon['maximum_round_index_M']+1) if math.ldexp(selected,m)==.8]
    if len(ms)!=1:raise ValueError('Saved T=.8 must match one dyadic round')
    return dict(base_time=selected,energy_precision_ha=CHEMICAL_ACCURACY_HA,
        beta_total=beta_total,beta_stat=beta_stat,alpha_total_per_method=alpha_total,
        alpha_axis=alpha_axis,alpha_sum_all_rounds=math.fsum([alpha_axis]*(2*count)),
        horizon=horizon,D1_m=ms[0],D1_q_RPE=1 << ms[0],prior_attempts=attempted,
        energy_prior_ha=list(prior),energy_shift=0,decoder='consecutive_dyadic_nearest_branch_principal_energy_v1')


def _z(record):
    return complex(record['real'],record['imag'])


def analyze_point(point, config, *, state_energy, state_residual):
    """Numerical state/reference pilot to stored solver energy; not a ground proof."""
    c=point['candidate']
    if c['id'] not in IDS or c['T']!=.8:
        raise ValueError('D1 is restricted to the fixed two T=.8 candidates')
    row=connection.bind_saved_round(point,target_id='H4_DF10_GROUND_CONDITIONAL_D1',
                                   m=config['D1_m'],base_time=config['base_time'])
    sig=point['saved_signal'];endpoint=point['saved_finite_time_endpoint']
    ref,corr,raw=(_z(sig[k]) for k in ('reference','corrected','raw'))
    u=[max(point['saved_allowance']['u'],endpoint['axes'][a]['allowance']) for a in ('real','imag')]
    bias=_z(sig['total_signed_bias'])
    if abs((corr-ref)-bias)>math.hypot(*u):
        raise ValueError('Saved bias/reference mismatch exceeds existing allowance')
    phase_reference=cmath.exp(-1j*state_energy*c['T'])
    reference_offset=abs(ref-phase_reference)
    residual_guard=c['T']*state_residual
    # Includes total PF/discard/RTE bias ONCE, original coordinate allowances,
    # the observed full-H-reference offset, and a conservative residual term.
    box=math.hypot(abs(bias.real)+u[0],abs(bias.imag)+u[1])
    corrected_disk=math.fsum((box,reference_offset,residual_guard))
    A=math.exp(-sig['log_B']);B=math.exp(sig['log_B'])
    closure=abs(raw-A*corr)
    margin=point['saved_allowance']['log_B_margin']
    normalization_error=math.fsum((closure,A*(1+corrected_disk)*math.expm1(margin)))
    policy=dict(beta_total=config['beta_total'],beta_stat=config['beta_stat'],
                alpha_X=config['alpha_axis'],alpha_Y=config['alpha_axis'],
                law_model='deterministic' if c['method']!='PR' else 'ideal_finite_RTE_law')
    shared=dict(policy=policy,corrected_signal_disk=corrected_disk,
        normalization_error_raw=normalization_error,
        reference_provenance='Saved reference + numerical solver-energy/residual pilot; exact ground identification conditional',
        iid_provenance='Fresh independent +/-1 outcomes under the reference-defined ideal law',
        error_provenance='Saved total signed bias + preserved u + reference offset + T*saved residual',
        df_accounting='own_DF_target_no_DF_energy_shift')
    ideal=connection.assess_supplied_round(row,implementation_error_raw=0.,law_error_raw=0.,**shared)
    # Unknown actual gate/law distances remain unknown; no numerical gate residual
    # or absence of rare samples is silently promoted to a signal-error bound.
    actual=connection.assess_supplied_round(row,implementation_error_raw=None,law_error_raw=None,**shared)
    prices=[]
    for p in point['prices']:
        if p['candidate']!=c or p['binding']!=point['binding']:
            raise ValueError('Price/scientific identity mismatch')
        metrics=p['mean_metrics']
        prices.append(dict(classification=p['classification'],cohort=p['cohort'],n=p['n'],
            rz_X=metrics['cosine']['rz_count'],rz_Y=metrics['sine']['rz_count'],source=p['source']))
    return dict(candidate_id=c['id'],candidate=c,m=config['D1_m'],q_RPE=config['D1_q_RPE'],
        q_PF=c['q'],reference_solver_energy_ha=state_energy,
        state_residual_ha=state_residual,ground_state_certified=False,
        reference_diagnostic=dict(signed_phase=cmath.phase(ref/phase_reference),
            complex_offset=reference_offset,observed_radius=abs(ref),T_times_saved_residual=residual_guard,
            used_for_prior_or_branch_choice=False),
        normalization=dict(log_B=sig['log_B'],B=B,attenuation=A,
            log_B_margin=margin,raw_corrected_closure=closure,raw_margin=normalization_error),
        error_disk=dict(saved_total_signed_bias=sig['total_signed_bias'],preserved_u_X=u[0],preserved_u_Y=u[1],
            coordinate_box_disk=box,reference_offset=reference_offset,residual_guard=residual_guard,
            corrected_signal_disk=corrected_disk,unconditional_ground_identification_error=None),
        observed_signal=dict(corrected_radius=abs(corr),raw_radius=abs(raw),
            corrected_phase_to_solver=cmath.phase(corr/phase_reference),raw_phase_to_solver=cmath.phase(raw/phase_reference)),
        ideal_numerical_pilot=ideal,actual_implementation=actual,
        compiled_prices=prices,round_resource_product_computed=False,
        full_rpe_energy_success=False,overall_guarantee='CONDITIONAL_NUMERICAL_PILOT_ONLY')
