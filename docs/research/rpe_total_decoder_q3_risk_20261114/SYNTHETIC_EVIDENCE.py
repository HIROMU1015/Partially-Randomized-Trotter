"""Artificial evidence for the total estimator Z1+T1 and its Q3 risk (no MC, no saved H4 data).

Run from the repository root:  python -B <this file> <out.json>
Reuses tests/check_rpe_total_decoder_risk.py (W0 laws, independent comb/Fraction
oracle). Section 'bench' compares the lightweight lemma with the exact DP on
larger artificial M=12 laws (h=0.1, H4-like geometry, fictional raws/N).
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import numpy as np
import check_rpe_total_decoder_risk as C
from check_rpe_signal_statistics_caller import synthetic, I, MOCK

R, T, D, a, X = C.R, C.T, C.D, C.a, C.X
F = lambda v: None if v is None else float(X(v))


def law_cases():
    rows = []
    for name in C.CASES:
        o = C.oracle(C.make(name)); out = C.evaluate(C.make(name)); total = out['deterministic_risk']
        b = total['bounds']; partial = out['partial_decoder_risk']; dp = b['total_state_dp']
        rows.append(dict(case=name, histories=o['histories'], decoded_by_original=o['decoded'],
            bit_identical_on_original_decoded=o['identical'],
            any_zero=float(o['zero_any']), partial_unresolved=float(o['unresolved']),
            tie_argmin_by_round=[float(v) for v in o['tie_round']], exact_MSE_total=float(o['mse']),
            bias_squared=float(o['bias']**2), variance=float(o['variance']),
            original_resolved_contribution_eta=float(o['eta']),
            partial_coarse=F(partial.get('coarse_MSE_upper')),
            partial_DP=[F(partial['state_refinement']['MSE_lower']), F(partial['state_refinement']['MSE_upper'])],
            lemma=F(b['lemma']['MSE_upper']), total_DP_exact=F(dp['MSE_exact']),
            selected_upper=F(total['MSE_upper']), selected_method=total['MSE_upper_method'],
            max_path_transition_excess_rad=float(o['path_excess']),
            checks=dict(DP_equals_enumeration=X(dp['MSE_exact']) == o['mse'],
                lemma_valid=X(b['lemma']['MSE_upper']) >= o['mse'],
                coarse_valid=X(partial['coarse_MSE_upper']) >= o['mse'],
                partial_DP_brackets=X(partial['state_refinement']['MSE_lower']) <= o['mse'] <= X(partial['state_refinement']['MSE_upper']),
                eta_identical=X(dp['partial_decoder']['resolved_MSE_contribution']) == o['eta'],
                bias_variance_identity=X(dp['moments']['bias_squared'][0]) + X(dp['moments']['variance'][0]) == o['mse'])))
    return rows


def partial_scopes():
    checked = valid = 0
    for name in C.CASES:
        o = C.oracle(C.make(name))
        for states, pairs in ((0, 0), (0, 10**6), (1, 10**6), (2, 10**6), (3, 10**6), (10**6, 7), (2, 40), (5, 100)):
            dp = C.evaluate(C.make(name, total=dict(C.BIG, maximum_frontier_states=states,
                maximum_transition_pairs=pairs)))['deterministic_risk']['bounds']['total_state_dp']
            if dp['status'] != 'PARTIAL_TOTAL_STATE_DP_WITH_STATE_RESIDUAL_BOUND':
                continue
            checked += 1; mom = dp['moments']
            valid += (X(dp['MSE_lower']) <= o['mse'] <= X(dp['MSE_upper'])
                      and X(mom['mean_energy'][0]) <= o['m1'] <= X(mom['mean_energy'][1])
                      and X(mom['variance'][0]) <= o['variance'] <= X(mom['variance'][1]))
    return dict(partial_scope_runs=checked, all_bounds_and_moment_intervals_valid=checked == valid)


def negative_controls():
    unguarded = lambda radius, *, h, theta, loss_upper: (min(loss_upper, (radius/h+R.division_slack(h))**2), True)
    o = C.oracle(C.make('principal_edge_wrap'))
    args = C.make('principal_edge_wrap', total=dict(C.BIG, maximum_frontier_states=1))
    with patch.object(R, 'energy_loss_cap', side_effect=unguarded):
        bad = C.evaluate(args)['deterministic_risk']['bounds']
    good = C.evaluate(args)['deterministic_risk']['bounds']
    z = C.oracle(C.make('zero_tie_rich')); fam = C.make('zero_tie_rich')
    fam['statistical_contract']['summary_options']['unresolved_loss']['upper'] = 0
    low_family = C.evaluate(fam)
    return dict(principal_edge_wrap_exact_MSE=float(o['mse']),
        lemma_without_guard=F(bad['lemma']['MSE_upper']), lemma_with_guard=F(good['lemma']['MSE_upper']),
        residual_without_guard=F(bad['total_state_dp']['MSE_upper']), residual_with_guard=F(good['total_state_dp']['MSE_upper']),
        guard_needed=X(bad['lemma']['MSE_upper']) < o['mse'] and X(bad['total_state_dp']['MSE_upper']) < o['mse'],
        guarded_valid=X(good['lemma']['MSE_upper']) >= o['mse'] and X(good['total_state_dp']['MSE_upper']) >= o['mse'],
        family_upper_below_L_D=dict(exact_total_MSE=float(z['mse']),
            partial_DP_upper=F(low_family['partial_decoder_risk']['state_refinement']['MSE_upper']),
            excluded_from_total=not low_family['deterministic_risk']['bounds']['partial_state_dp_valid_for_total']))


def stress(n=220000):
    rng = np.random.default_rng(20261114); worst = -1.
    for m in range(1, 13):
        q = 1 << m
        principal = R.F._wrap(rng.uniform(-math.pi, math.pi, n)); estimate = R.F._wrap(rng.uniform(-math.pi, math.pi, n))
        k = rng.integers(0, q, n//10)
        estimate = np.concatenate([estimate, R.F._wrap((principal[:n//10]+2*math.pi*k)/q+math.pi/q)])
        principal = np.concatenate([principal, principal[:n//10]]); zero = rng.random(len(principal)) < .05
        value, tie = T.step(principal, estimate, q, zero)
        assert np.all(value[zero] == estimate[zero])
        move = np.abs(value-estimate); move = np.minimum(move, 2*math.pi-move)
        worst = max(worst, float(np.max(move-math.pi/q)))
    return dict(transitions_per_round=n+n//10, rounds='m=1..12', computed_midpoints=True, zero_rows_held=True,
                max_excess_rad=worst, allowance_rad=float(R.TRANSITION_ALLOWANCE),
                within_allowance=worst <= float(R.TRANSITION_ALLOWANCE),
                meaning='BINARY64_FLOAT_CHECK_NOT_CERTIFIED')


def bench_args(label, vector, h=.1, target=-2.1663874486347634, amplitude=.8, scope=None):
    M = len(vector)-1
    config, packet, _, tgt, contract = synthetic(M=M)
    design = deepcopy(config['spec']); design['h'] = h
    config = D.B.generate(design); packet['configuration_id'] = config['id']; tgt['energy_ha'] = target
    theta = -h*target
    for m, (mean, row) in enumerate(zip(packet['means'], config['rounds'])):
        z = amplitude*complex(math.cos(theta*2**m), math.sin(theta*2**m))
        mean.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
            raw=dict(real=z.real, imag=z.imag), corrected=dict(real=z.real, imag=z.imag), normalization=dict(log_B=0.))
    allocations = [dict(allocation_id=label, source=MOCK, shot_policy='USER_SUPPLIED',
        shots=[dict(m=m, N_X=n, N_Y=n) for m, n in enumerate(vector)])]
    edge = math.nextafter(math.pi/h, math.inf)
    contract['summary_options'] = dict(unresolved_loss=dict(kind='BOUNDED_INTERVAL', lower=0,
        upper=max((edge-target)**2, (edge+target)**2), source=MOCK))
    options = dict(source=MOCK, phase_radii=[1.]*(M+1), probability_rule=D.EXACT_AFFINE,
        maximum_round_outcomes=10**6, maximum_verification_pairs=10**8,
        model_loss_contract=dict(adopted=True, source=MOCK), estimator=T.ESTIMATOR)
    if scope:
        options['total_state_refinement'] = dict(source=MOCK, **scope)
    return dict(config=config, signal_packet=packet, allocations=allocations, evaluation_target=tgt,
                statistical_contract=contract, integration=I, source=MOCK, options=options)


def bench_one(item):
    label, vector, scope = item
    started = time.perf_counter(); lemma_only = C.evaluate(bench_args(label, vector))['deterministic_risk']
    lemma_seconds = time.perf_counter()-started
    started = time.perf_counter(); full = C.evaluate(bench_args(label, vector, scope=scope))
    dp_seconds = time.perf_counter()-started
    total = full['deterministic_risk']; dp = total['bounds']['total_state_dp']; partial = full['partial_decoder_risk']
    return dict(label=label, vector=list(vector), count_cells=[(n+1)**2 for n in vector], scope=scope,
        seconds=dict(B_coarse_plus_lemma=lemma_seconds, B_coarse_plus_lemma_plus_total_DP=dp_seconds),
        partial_coarse=F(partial.get('MSE_upper')), any_unsafe=1-F(partial['all_good']['exact']),
        any_zero=F(partial['any_zero']['exact']), lemma=F(lemma_only['bounds']['lemma']['MSE_upper']),
        lemma_status=lemma_only['bounds']['lemma']['status'],
        total_DP=dict(status=dp['status'], lower=F(dp['MSE_lower']), upper=F(dp['MSE_upper']),
            transitions=dp['transition_pairs'], max_states=max(r['retained_states'] for r in dp['rounds']),
            residual=F(dp['residual_probability']), bias_squared=[F(v) for v in dp['moments']['bias_squared']],
            variance=[F(v) for v in dp['moments']['variance']], partial_unresolved=[F(v) for v in dp['partial_decoder']['unresolved_probability_bounds']]),
        selected=F(total['MSE_upper']), selected_method=total['MSE_upper_method'])


def bench():
    scope = dict(maximum_frontier_states=4096, maximum_transition_pairs=10**8)
    items = [('even_N_zero_mass', (12, 12, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4), scope),
             ('odd_N_one_axis_like', (13, 11, 11, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3), scope),
             ('small_N_heavy_unsafe', (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4), scope),
             ('tight_scope', (12, 12, 10, 10, 8, 8, 8, 6, 6, 6, 4, 4, 4), dict(maximum_frontier_states=64, maximum_transition_pairs=10**8))]
    with ProcessPoolExecutor(max_workers=len(items)) as pool:
        return list(pool.map(bench_one, items))


def main():
    started = time.perf_counter()
    laws = law_cases(); result = dict(scope='ARTIFICIAL_LAWS_ONLY_NO_SAVED_H4_NO_MC', estimator=T.ESTIMATOR,
        laws=laws, partial_scopes=partial_scopes(), negative_controls=negative_controls(), stress=stress(),
        parity=dict(odd_N_never_zero=all(R.S.count_mean(k, n)['mean'] != 0 for n in range(1, 201, 2) for k in range(n+1))),
        bench=bench())
    checks = [all(r['checks'].values()) and r['bit_identical_on_original_decoded'] for r in laws]
    checks += [result['partial_scopes']['all_bounds_and_moment_intervals_valid'], result['negative_controls']['guard_needed'],
               result['negative_controls']['guarded_valid'], result['negative_controls']['family_upper_below_L_D']['excluded_from_total'],
               result['stress']['within_allowance'], result['parity']['odd_N_never_zero']]
    result.update(all_checks_pass=all(checks), check_count=len(checks), wall_seconds=time.perf_counter()-started)
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(all_checks_pass=result['all_checks_pass'], checks=len(checks),
                          wall_seconds=round(result['wall_seconds'], 1))))


if __name__ == '__main__':
    main()
