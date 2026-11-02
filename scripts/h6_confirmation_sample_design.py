"""H6 signed-margin planning arithmetic, reusing the fixed H8 design helpers.

Artificial inputs are used in tests. Saved H6 values are processed only by the
separately approved, SOURCE/input/runtime-bound one-shot S_STATS entry point.
No native providers, sampling, compiler, shot solver, raw cost arrays or RNG.
"""
from fractions import Fraction
import math

import h8_confirmation_sample_design as shared


def signed_normal_scenario(mean_exact, se, n_pilot, comparator_exact, expected_direction, **kwargs):
    mean, comparator = Fraction(mean_exact), Fraction(comparator_exact)
    delta = comparator - mean
    direction = 1 if delta > 0 else -1 if delta < 0 else 0
    shared.require(type(expected_direction) is int and expected_direction in (-1, 1)
                   and direction == expected_direction, 'SAVED_DIRECTION_AND_NONZERO_MARGIN')
    # H8 takes an ordered positive margin. Its SE input remains the PR pilot SE.
    result = shared.normal_scenario(str(min(mean, comparator)), se, n_pilot,
                                    str(max(mean, comparator)), **kwargs)
    result.update(normal_ceiling_n=result['normal_plugin_n'],
        planning_n=max(2, result['normal_plugin_n']), signed_margin_exact=str(delta),
        expected_direction=expected_direction, saved_PR_mean_exact=str(mean),
        fixed_PF_G_exact=str(comparator), variance_source='saved_PR_paired_SE_n8')
    return result


def planning_power(n, scenario):
    """Conditional normal-model power; never empirical or certified power."""
    shared.require(type(n) is int and n >= 2, 'PLANNING_N')
    sigma = scenario['pilot_SD'] * scenario['sd_multiplier']
    threshold = shared.NormalDist().inv_cdf(1-scenario['alpha_each'])
    return shared.NormalDist().cdf(math.sqrt(n)*float(Fraction(scenario['margin_exact']))/sigma - threshold)


def order2_visibility(p2, events_per_trajectory, n):
    shared.require(type(events_per_trajectory) is int and events_per_trajectory > 0, 'K2_EVENT_COUNT')
    probability = shared.at_least_one(p2, events_per_trajectory)
    return dict(order2_trajectory_probability=probability,
                probability_at_least_one=shared.at_least_one(probability, n), n=n)


def recommended_n(reference_n, normal_n, visibility_n, block):
    shared.require(all(type(n) is int and n > 0 for n in (reference_n, normal_n, block)), 'RECOMMENDATION_INPUTS')
    if visibility_n is None:
        return None
    shared.require(type(visibility_n) is int and visibility_n > 0, 'VISIBILITY_N')
    bound = max(reference_n, normal_n, visibility_n)
    return ((bound+block-1)//block)*block


def analyze(plan, policy):
    """One approved pass over two fixed saved scalar/law records."""
    shared.require(policy['schema'] == 'H6_S_STATS_POLICY_V1'
        and (policy['family_alpha'], policy['comparisons'], policy['planning_power']) == (.05, 2, .8)
        and policy['SD_multipliers'] == [1,2,4] and policy['minimum_planning_n'] == 2
        and policy['H8_reference_n'] == 48 and policy['visibility_failure_probability'] == .025
        and policy['recommendation_block'] == 8
        and policy['recommendation_rule'] == 'round_up_8(max(H8_reference_n, SD4_planning_n, order2_visibility_97_5_min_n))',
        'FIXED_APPROVED_PLANNING_POLICY')
    expected = {'H6_CONFIRM_P4_Q2': ('B2_p4_q2_2nd_r4_K2', 2, 8, .05, 'B0I_p7_q1_2nd', -1),
                'H6_CONFIRM_P4_Q8': ('B2_p4_q8_2nd_r4_K2', 8, 32, .0005, 'B0I_p11_q4_4th', 1)}
    shared.require(len(plan['targets']) == 2 and {t['scope'] for t in plan['targets']} == set(expected), 'EXACT_TWO_H6_TARGETS')
    outputs = []
    for target in plan['targets']:
        cell, pilot, law = target['cell'], target['saved_n8'], target['law']
        shared.require((cell['id'], cell['q'], cell['R'], target['epsilon'], target['fixed_PF_id'],
                        target['expected_direction']) == expected[target['scope']]
                       and cell['prefix'] == 4 and cell['r'] == 4 and cell['K'] == 2
                       and pilot['n'] == policy['pilot_n'] == 8
                       and law['finite_taylor_order'] == 2 and law['orders'] == [0,2]
                       and len(law['order_probabilities']) == 2, 'FIXED_H6_N8_K2_BINDING')
        scenarios = [signed_normal_scenario(pilot['mean_exact'], pilot['paired_SE'], pilot['n'],
            target['fixed_PF_G_exact'], target['expected_direction'], family_alpha=policy['family_alpha'],
            comparisons=policy['comparisons'], power=policy['planning_power'], sd_multiplier=f)
            for f in policy['SD_multipliers']]
        p2 = law['order_probabilities'][1]
        reference = order2_visibility(p2, cell['R'], policy['H8_reference_n'])
        visibility_n = shared.detection_n(reference['order2_trajectory_probability'], policy['visibility_failure_probability'])
        normal_n = next(s['planning_n'] for s in scenarios if s['sd_multiplier'] == 4)
        proposal = recommended_n(policy['H8_reference_n'], normal_n, visibility_n, policy['recommendation_block'])
        for scenario in scenarios:
            scenario['visibility_at_planning_n'] = order2_visibility(p2, cell['R'], scenario['planning_n'])
            scenario['normal_model_power_at_n48'] = planning_power(policy['H8_reference_n'], scenario)
            scenario['normal_model_power_at_recommended_n'] = None if proposal is None else planning_power(proposal, scenario)
        outputs.append(dict(scope=target['scope'], cell=cell, epsilon=target['epsilon'],
            fixed_PF_id=target['fixed_PF_id'], expected_direction=target['expected_direction'],
            saved_n8=pilot, fixed_PF_G_exact=target['fixed_PF_G_exact'],
            original_law_sha256=target['original_law_sha256'], saved_order2_probability=p2,
            scenarios=scenarios, reference_n48=reference, order2_visibility_min_n_97_5=visibility_n,
            visibility_at_min_n=None if visibility_n is None else order2_visibility(p2,cell['R'],visibility_n),
            recommended_n=proposal,
            recommended_visibility=None if proposal is None else order2_visibility(p2,cell['R'],proposal)))
    return dict(status='COMPLETE_MANDATORY_STOP', stage='H6_CONFIRMATION_S_STATS',
        binding=plan['binding'], policy=policy, candidates=outputs, saved_scalar_passes=1,
        population_mean_certified=False, actual_power_certified=False, cost_contribution_bound=None,
        confidence_intervals=None, p_values=None, normal_model_assumptions_verified=False,
        visibility_is_cost_mean_precision=False, final_scientific_n=None, final_scientific_SOURCE=None,
        final_seed_registry=None, final_scientific_call_budget=None, scientific_execution_approved=False,
        new_sampling=0, native_compile=0, signal=0, shot_recalculation=0, new_confirmation_seeds=0,
        raw_variance_reaggregation=0, bootstrap=0, retry=False, next_stage_authorized=False)
