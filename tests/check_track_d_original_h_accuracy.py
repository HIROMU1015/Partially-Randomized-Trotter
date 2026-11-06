"""Artificial scalars/coordinate means only; never read molecular evidence."""
from dataclasses import asdict, replace
import math
import subprocess
import sys
from unittest.mock import patch

import pytest

from trottertracks import c1_stage3_accuracy as old
from trottertracks import track_d_b1_c1 as b1
from trottertracks import track_d_original_h_accuracy as new


def binding(**changes):
    return new.SignalBinding(**(dict(original_h='TOY_H', cdf='TOY_CDF',
        sector='TOY_INVARIANT_SECTOR', law='TOY_LAW', source='TOY_OPERATOR_SOURCE',
        implementation='TOY_DIRECT') | changes))


def target(**changes):
    return new.OriginalHTarget(**(dict(energy_ha=1/8, base_time=1., beta_total=.4,
        beta_stat=.2, alpha_axis=.001, alpha_total=.01, maximum_round_index=2,
        energy_window_ha=(-1., 1.)) | changes))


def evidence(m=0, **changes):
    return new.OriginalHRoundEvidence(**(dict(m=m, attenuation=.8, operator_error=.015,
        implementation_signal_error=.001, finite_law_signal_error=.002,
        binding=binding(), law_model=new.ACTUAL, alias_branch_verified=True, iid_per_shot=True,
        operator_provenance='artificial operator norm', alias_provenance='toy prior branch',
        iid_provenance='independent toy outcome model', implementation_provenance='toy raw bound',
        law_provenance='toy raw law bound') | changes))


def plans(t=None, **changes):
    t = t or target()
    return [new.assess_round(t, binding(), evidence(m, **changes))
            for m in range(t.maximum_round_index+1)]


def test_passive_import_and_no_molecular_constructor_sampler_or_compiler():
    subprocess.run([sys.executable, '-B', '-c',
        "import sys; from trottertracks import track_d_original_h_accuracy; "
        "assert not any(n in sys.modules for n in ['numpy','scipy','qiskit','ffsim'])"], check=True)
    # Import existing dependencies first; only numerical scalar entries run below.
    old._legacy('rpe_resource_accounting')
    old._legacy('rpe_four_round_phase_validation')
    old._legacy('rpe_target_round_horizon_validation')
    with patch('numpy.linalg.eigh', side_effect=AssertionError('ground solve')), \
         patch('numpy.linalg.matrix_power', side_effect=AssertionError('matrix power')), \
         patch('scipy.linalg.expm', side_effect=AssertionError('matrix exponential')), \
         patch('numpy.random.default_rng', side_effect=AssertionError('sampling')), \
         patch('qiskit.QuantumCircuit', side_effect=AssertionError('circuit build')):
        ps = plans()
        assert new.assess_energy(target(), binding(), ps)['same_accuracy']
        assert new.decode_energy(target(), binding(), ps, [(1., 0.)]*3)['energy_estimate_ha'] == 0


def test_original_state_raw_units_and_exact_existing_shot_helper():
    t, e = target(), evidence()
    r = new.assess_round(t, binding(), e)
    raw = .8*.015+.001+.002
    coordinate = (.8-raw)*math.sin(.2)/math.sqrt(2.)
    assert r['raw_signal_error'] == pytest.approx(raw)
    assert r['radius_lower'] == pytest.approx(.8-raw)
    assert r['beta_deterministic'] == pytest.approx(math.asin(raw/.8))
    assert r['epsilon_coordinate'] == pytest.approx(coordinate)
    assert r['shots'] == old._legacy('rpe_resource_accounting')._hoeffding_shots(coordinate, .001)
    assert r['state_model'] == new.STATE
    assert not r['unconditional_success_claim'] and not r['success_probability_evaluated']
    assert r['input_errors']['implementation_raw'] == .001


def test_existing_own_df_guard_and_values_are_preserved():
    t = old.AccuracyTarget(1/8, 1/4, 1., .4, .2, .001, .05)
    e = old.RoundEvidence(0, .8, .01, .005, .001, .002, 'TOY',
                         'own_DF_ground_state_oracle', True, True)
    a = old.assess_round(t, e)
    b = new.assess_round(target(), binding(), evidence())
    for key in ('raw_signal_error', 'radius_lower', 'beta_deterministic', 'epsilon_coordinate', 'shots'):
        assert a[key] == b[key]
    with pytest.raises(ValueError):
        old.assess_round(t, replace(e, state_model=new.STATE))
    with pytest.raises(ValueError):
        old.assess_energy(t, 0., [], state_model=new.STATE, alias_branch_verified=True)


@pytest.mark.parametrize('field', ['original_h','cdf','sector','law','source','implementation'])
def test_each_binding_mismatch_rejected(field):
    e = evidence(binding=binding(**{field: 'OTHER'}))
    with pytest.raises(ValueError, match='binding mismatch'):
        new.assess_round(target(), binding(), e)


@pytest.mark.parametrize('change', [dict(state_model='own_DF_ground_state_oracle'),
    dict(state_model='HF'), dict(operator_error_kind='raw_signal'),
    dict(operator_error_kind='energy_Ha'), dict(law_model='weighted_estimator')])
def test_incompatible_state_units_and_unsupported_law_rejected(change):
    with pytest.raises(ValueError):
        new.assess_round(target(), binding(), evidence(**change))


def test_b1_synthetic_adapter_df_is_counted_once_and_no_saved_signal_or_N_used():
    t = target()
    rows = [b1.transport_round(dict(attenuation=.8, pf_operator_error=.01,
        rte_corrected_operator_error=.005), m=m, df_error=.001, base_time=t.base_time,
        historical_target=dict(beta_total=t.beta_total,beta_stat=t.beta_stat), finite=old._finite)
        for m in range(3)]
    context = asdict(evidence())
    for key in ('m','attenuation','operator_error','binding','evidence_kind'):
        context.pop(key)
    ps = []
    for row in rows:
        # Impossible stale N/radius must not replace transported operator data.
        row.update(N=999999, observed_radius_lower=-100.)
        e = new.OriginalHRoundEvidence.from_b1_row(row, target=t, binding=binding(), **context)
        r = new.assess_round(t, binding(), e)
        assert r['raw_signal_error'] == pytest.approx(.8*(.015+.001*(1 << row['m']))+.003)
        ps.append(r)
    energy = new.assess_energy(t, binding(), ps)
    assert energy['energy_error_bound_ha'] == .4/4  # no old Weyl/DF addition
    assert energy['df_counting'] == 'already_in_signal_once'
    assert energy['numerical_pilot']
    with pytest.raises(ValueError, match='time differs'):
        new.OriginalHRoundEvidence.from_b1_row(rows[0], target=replace(t,base_time=.5),
                                             binding=binding(), **context)
    with pytest.raises(ValueError, match='q mismatch'):
        new.OriginalHRoundEvidence.from_b1_row(dict(rows[0],q=2),target=t,binding=binding(),**context)


@pytest.mark.parametrize('change', [dict(implementation_signal_error=None),
    dict(finite_law_signal_error=None), dict(implementation_provenance=None),
    dict(law_provenance=None), dict(alias_branch_verified=False),
    dict(iid_per_shot=False), dict(alias_provenance=''), dict(iid_provenance=None)])
def test_unknown_error_or_sampling_premise_never_zero_filled(change):
    r = new.assess_round(target(), binding(), evidence(**change))
    assert r['shots'] is None
    assert r['status'] == 'UNRESOLVED_SIGNAL_OR_SAMPLING_PREMISES'
    ps = plans(**change)
    assert not new.assess_energy(target(), binding(), ps)['same_accuracy']


def test_ideal_model_is_explicit_and_never_actual_evidence():
    r = new.assess_round(target(), binding(), evidence(law_model=new.IDEAL,
        implementation_signal_error=0., finite_law_signal_error=0.,
        implementation_provenance='exact native model', law_provenance='ideal IID model'))
    assert r['qualification'] == 'CONDITIONAL_IDEAL_MODEL'
    assert not r['unconditional_success_claim']
    with pytest.raises(ValueError, match='explicit zero'):
        new.assess_round(target(), binding(), evidence(law_model=new.IDEAL))
    with pytest.raises(ValueError, match='lanes cannot be mixed'):
        new.assess_energy(target(), binding(), [r, *plans()[1:]])


@pytest.mark.parametrize('change, status', [
    (dict(attenuation=0.), 'INFEASIBLE_RADIUS'),
    (dict(implementation_signal_error=1.), 'INFEASIBLE_RADIUS'),
    (dict(operator_error=.4), 'INFEASIBLE_PHASE')])
def test_nonpositive_radius_and_phase_excess_stop_before_shots(change,status):
    with patch.object(old._legacy('rpe_resource_accounting'), '_hoeffding_shots',
                      side_effect=AssertionError('N called for infeasible round')):
        r = new.assess_round(target(), binding(), evidence(**change))
    assert r['status'] == status and r['shots'] is None


def test_strict_full_horizon_failure_union_and_target_binding():
    t, ps = target(), plans()
    assert new.assess_energy(t, binding(), ps)['statistical_failure_upper'] == math.fsum([.002]*3)
    assert new.assess_energy(t, binding(), ps[:-1])['status'] == 'INCOMPLETE_ROUNDS'
    assert new.assess_energy(t, binding(), [ps[0], ps[2], ps[1]])['status'] == 'INCOMPLETE_ROUNDS'
    assert new.assess_energy(t, binding(), [ps[0], ps[1], ps[1]])['status'] == 'INCOMPLETE_ROUNDS'
    under = replace(t,alpha_total=math.nextafter(.006,0.))
    assert new.assess_energy(under,binding(),plans(under))['status'] == 'FAILURE_PROBABILITY_EXCEEDED'
    with pytest.raises(ValueError,match='target/state/SOURCE'):
        new.assess_energy(replace(t,energy_ha=.2),binding(),ps)
    with pytest.raises(ValueError,match='target/state/SOURCE'):
        new.assess_energy(t,binding(source='OTHER'),ps)
    with pytest.raises(ValueError,match='phase/failure model'):
        new.assess_energy(t,binding(),[dict(ps[0],alpha_axis=-1.),*ps[1:]])
    with pytest.raises(ValueError,match='horizon'):
        new.assess_energy(replace(t,maximum_round_index=1),binding(),[])
    extra = replace(t,maximum_round_index=3)
    assert extra.horizon()['achieved_energy_resolution_ha'] == .4/8
    assert not extra.horizon()['minimal_power_of_two_horizon']


@pytest.mark.parametrize('change', [dict(energy_ha=0.), dict(base_time=float('inf')),
    dict(beta_stat=.4), dict(beta_total=math.pi/3), dict(alpha_axis=1.),
    dict(alpha_total=float('nan')), dict(maximum_round_index=True),
    dict(energy_window_ha=(-4.,4.)), dict(energy_window_ha=(1.,-1.)),
    dict(energy_window_ha=(float('nan'),1.)), dict(energy_ha=True)])
def test_invalid_target_rejected(change):
    with pytest.raises(ValueError):
        target(**change)


@pytest.mark.parametrize('change', [dict(attenuation=float('nan')),dict(attenuation=1.1),
    dict(operator_error=-1.), dict(implementation_signal_error=float('inf')),
    dict(finite_law_signal_error=-.1), dict(m=True),dict(m=3),
    dict(iid_per_shot='yes'),dict(operator_provenance=''),dict(evidence_kind='assumed')])
def test_invalid_evidence_rejected(change):
    with pytest.raises(ValueError):
        new.assess_round(target(),binding(),evidence(**change))


def test_numerical_shot_range_is_unresolved_without_clipping():
    r = new.assess_round(target(beta_stat=1e-200),binding(),evidence())
    assert r['status'] == 'UNRESOLVED_NUMERICAL_SHOT_RANGE' and r['shots'] is None


def test_existing_decoder_sign_wrap_alias_and_no_success_simulation():
    t, ps = target(), plans()
    phi = .72  # q4 crosses the principal phase branch
    xy = [(math.cos((1 << m)*phi),math.sin((1 << m)*phi)) for m in range(3)]
    r = new.decode_energy(t,binding(),ps,xy)
    assert r['energy_estimate_ha'] == pytest.approx(-phi)
    assert not r['success_probability_evaluated']
    assert r['decoded'] == old._legacy('rpe_four_round_phase_validation').reconstruct_rpe_phase(
        [math.atan2(y,x) for x,y in xy],[1,2,4])
    assert new.decode_energy(t,binding(),ps,[(0.,0.)]*3)['status'] == 'UNRESOLVED_ZERO_COORDINATES'
    tie = [(1.,0.),(-1.,0.),(1.,0.)]
    assert new.decode_energy(t,binding(),ps,tie)['status'] == 'UNRESOLVED_BRANCH_TIE'
    badphi = 2.
    bad = [(math.cos((1 << m)*badphi),math.sin((1 << m)*badphi)) for m in range(3)]
    assert new.decode_energy(t,binding(),ps,bad)['status'] == 'ENERGY_BRANCH_OUTSIDE_PRIOR'
    for bad in ([(1.,0.)],[(float('nan'),0.)]*3,[(2.,0.)]*3):
        with pytest.raises(ValueError):
            new.decode_energy(t,binding(),ps,bad)


def test_existing_cost_interface_accepts_equal_axis_N_without_new_cost_algorithm():
    ps = plans()
    assessment = new.assess_energy(target(),binding(),ps)
    costs = [dict(m=p['m'],axis=axis,target_id=binding().target_id,
                  scope='TOY_FULL_WRAPPER',long_round_coverage=True,
                  mean={k:2. for k in old.METRICS},simultaneous_interval_qualified=False)
             for p in ps for axis in ('X','Y')]
    r = old.aggregate_totals(assessment,ps,costs,cost_scope='TOY_FULL_WRAPPER',
                             expected_scope='TOY_FULL_WRAPPER')
    assert r['shots'] == 2*sum(p['shots'] for p in ps)
    assert r['totals']['rz_count'] == 2*r['shots']
    assert r['intervals'] is None and not r['unconditional_total_resource_claim']
