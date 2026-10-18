"""Registered saved-result reporting and UNADOPTED confirmation planning only."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / 'repo'
sys.path.insert(0, str(ROOT / 'scripts'))
from h46_higher_pf_adapter import kernel, SIGNAL


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    out = Path(sys.argv[1])
    k = kernel()
    result_path = out / 'result.json'
    result = json.loads(result_path.read_text())['comparison']
    old = json.loads((ROOT / 'artifacts/resource_applicability/track_a_h46_higher_pf_B_saved_run01/2026-10-18/result.json').read_text())
    old_map = {x['candidate_id']: x for x in old['rows']}
    rows = result['rows']
    assert len(rows) == 24
    assert sum(x['eligibility'] == 'ELIGIBLE' for x in rows) == 18
    assert sum(x['cost'] is not None for x in rows) == 18
    old_costs = 0
    for row in rows:
        previous = old_map[row['candidate_id']]
        current_shots = {a: b for a, b in row['shots'].items() if a != 'evidence_ref'}
        old_shots = {a: b for a, b in previous['shots'].items() if a != 'evidence_ref'}
        assert current_shots == old_shots
        if previous['cost'] is not None:
            assert row['cost'] == previous['cost']
            old_costs += 1
        if row['cost'] is not None:
            assert row['cost']['summaries']['RZ']['n'] == 8
    assert old_costs == 6
    assert len(result['selections']) == 6
    assert all(x['status'] == 'COMPLETE_POINT_SELECTION' and x['full_grid_winner'] and not x['unresolved'] for x in result['selections'])
    assert result['native_compile_calls'] == result['sampling_calls'] == 0
    assert not result['next_phase_started']
    by_id = {x['candidate_id']: x for x in rows}
    selected = [by_id[x['full_grid_winner']] for x in result['selections']]
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    selection = {'status': 'EXPLORATION_POINT_SELECTION_ONLY',
                 'candidates': sorted(x['candidate_id'] for x in selected),
                 'result_sha256': result_sha,
                 'analysis_SOURCE': 'a65e4f6e7bcc83f90a054f6aafc4ee2e0ca5d80a',
                 'acquisition_SOURCE': 'd77e439a16d6953cb17e6d1ef91b0d9430144848',
                 'exploration_n_per_candidate': 8,
                 'confirmation_seed_union_audit_required_before_execution': True,
                 'population_winner_certified': False}
    selection['selection_sha256'] = k.digest(selection)
    planning = []
    for row in selected:
        cost = row['cost']['summaries']['RZ']
        planning.append({'candidate_id': row['candidate_id'], 'pilot_n': cost['n'],
                         'pilot_mean': cost['mean'], 'pilot_variance': cost['variance'],
                         'relative_MCSE_scenarios': [
                             {'target': target, **k.planning_n(cost['variance'], relative_mcse=target, reference_mean=cost['mean'])}
                             for target in (.01, .02, .05)]})
    comparisons = []
    for system in ('H4', 'H6'):
        ordered = sorted((x for x in selected if x['system'] == system), key=lambda x: x['formula_id'])
        for left, right in itertools.combinations(ordered, 2):
            a, b = (x['cost']['summaries']['RZ'] for x in (left, right))
            effect = .1 * max(a['mean'], b['mean'])
            comparisons.append({'left': left['candidate_id'], 'right': right['candidate_id'],
                                'effect': effect, 'effect_definition': '10_PERCENT_OF_LARGER_PILOT_MEAN',
                                'family_comparisons': 6, 'family_alpha_planning': .05, 'power_planning': .8,
                                'independent_candidate_assumption_for_planning_only': True,
                                **k.planning_equal_n_difference(a['variance'], b['variance'], effect,
                                                               alpha=.05, power=.8, comparisons=6)})
    signals = {s: json.loads((ROOT / SIGNAL / f'{s}_run02_results.json').read_text()) for s in ('H4', 'H6')}
    table = []
    for row in rows:
        formula = next(label for label, identity in k.FORMULAS.items() if identity == row['formula_id'])
        prefix = 3 if row['system'] == 'H4' else 5
        signal = next(x for x in signals[row['system']]['records'] if (x['label'], x['prefix'], x['q']) == (formula, prefix, row['q']))
        finite = next(x for x in signal['finite_rows'] if x['cutoff'] == 4)
        cost = row['cost']['summaries'] if row['cost'] else {}
        table.append({'candidate_id': row['candidate_id'], 'system': row['system'], 'formula': row['formula_id'],
                      'q': row['q'], 'eligibility': row['eligibility'],
                      'bias_real_signed': finite['total_signed_bias']['real'],
                      'bias_imag_signed': finite['total_signed_bias']['imag'],
                      'log_B_saved': finite['log_B'], 'B_saved': math.exp(finite['log_B']),
                      'log_B_margin': 1e-12, 'empirical_allowance': 1e-9,
                      'N_real': row['shots']['axes']['real']['shots'],
                      'N_imag': row['shots']['axes']['imag']['shots'],
                      'n': cost['RZ']['n'] if cost else None,
                      'mean_cosine_RZ': cost['cosine_RZ']['mean'] if cost else None,
                      'mean_sine_RZ': cost['sine_RZ']['mean'] if cost else None,
                      'mean_total_RZ': cost['RZ']['mean'] if cost else None,
                      'mean_total_RZ_exact': row['mean_exact'],
                      'SE_total_RZ': cost['RZ']['SE'] if cost else None,
                      'mean_total_CX': cost['CX']['mean'] if cost else None,
                      'SE_total_CX': cost['CX']['SE'] if cost else None,
                      'selected_q_point': row['candidate_id'] in selection['candidates']})
    design = {'status': 'UNAPPROVED_DESIGN_SCENARIOS_NO_ADOPTED_N', 'selection_sha256': selection['selection_sha256'],
              'candidate_planning': planning, 'difference_planning': comparisons,
              'adopted_n': None, 'new_sampling': 0, 'new_compile': 0,
              'independence_proved': False, 'normal_approximation_or_pilot_variance_is_not_a_guarantee': True,
              'unobserved_rare_event_costs_imputed': False,
              'q_global_optimality_not_tested_by_selected_points_only': True}
    save(out / 'selection.json', selection)
    save(out / 'confirmation_design_UNAPPROVED.json', design)
    save(out / 'audit.json', {'status': 'PASS', 'saved_shot_rows_reproduced': 24,
                             'old_cost_groups_exactly_reproduced': old_costs, 'eligible_measured_groups': 18,
                             'complete_q_strata': 6, 'new_compile': 0, 'new_sampling': 0,
                             'candidate_independence_asserted_in_comparison': False, 'mandatory_STOP': True})
    with (out / 'comparison_table.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    for row in selected:
        print(json.dumps({'selected': row['candidate_id'], 'RZ': row['cost']['summaries']['RZ']}))


if __name__ == '__main__':
    main()
