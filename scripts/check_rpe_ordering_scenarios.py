"""Small ARTIFICIAL ordering comparisons using the shared joint replay only.

The fixed fixture definitions are case-design assumptions. No archived outcome,
physical signal, quantum kernel, compiler or calibration worker is opened/run.
"""
import argparse
from copy import deepcopy
import csv
from fractions import Fraction as F
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_schedule_search as W
import check_rpe_priority_search as CP
import replay_rpe_candidate_ordering as J

POLICIES = ('ROUND05', 'IMPROVEMENT_PER_TIME_V1')


class ArtificialIndex:
    """Materialize a predefined toy answer only for a selected launch key."""
    def __init__(self, function, durations):
        self.function, self.durations = function, durations
        self.rows, self.read_ids, self.clock = {}, [], None

    def get(self, key, default=None):
        identity = J.qid(dict(configuration_id=key[0], vector=key[1]))
        selected = {e['id'] for e in self.clock.events if e['event'] == 'SELECT_FOR_LAUNCH'}
        if identity not in selected:
            raise AssertionError('Artificial future result requested before selected launch')
        self.read_ids.append(identity)
        if key not in self.rows:
            lower, upper = self.function(*key)
            self.rows[key] = dict(seconds=self.durations[key[0]],
                value=dict(status=O.PASS if upper <= CP.TAU else O.FAIL, MSE_lower=lower, MSE_upper=upper))
        return self.rows[key]

    def __getitem__(self, key):
        return self.get(key)

    def values(self):
        raise AssertionError('Future artificial answers/timings must not be enumerated')


class ScenarioReplay(J.JointReplay):
    """Metadata-only overrides; scheduling/execution/adoption use root's run()."""
    def schedule_action(self, proposal, ordinal):
        action = super().schedule_action(proposal, ordinal)
        candidate = self.by_id.get(action['configuration_id'])
        if candidate:
            metadata = candidate['artificial_action']
            action.update(width=metadata['width'], preparation=deepcopy(metadata['preparation']))
            action['q3_seconds'] = metadata['declared_Q3_prior']
            if metadata.get('declared_entry_point_cost') is not None:
                action['cost'] = metadata['declared_entry_point_cost']
        return action

    def shot_actions(self, frame):
        actions = super().shot_actions(frame)
        for action in actions:
            metadata = self.by_id[action['configuration_id']]['artificial_action']
            action['width'] = metadata['width']
            if metadata.get('declared_Q3_prior') is not None:
                action['q3_seconds'] = metadata['declared_Q3_prior']
        return actions


def family(ld):
    return W.Family(ld, [dict(m=m, c=1, q=1, J=1) for m in range(13)])


def design(name):
    candidates = [CP.config('BASE', 1, '1', 1), CP.config('OTHER', 2, '1', F(101, 100))]
    widths = [0.2, 0.2]
    preparations = [dict(signal_reuse=True, price_reuse=True, requires_calibration=False, seconds=0,
                         source='ARTIFICIAL_REUSED_INPUTS') for _ in candidates]
    durations, priors = {'BASE': 100, 'OTHER': 1}, {'BASE': 100, 'OTHER': 1}
    seeded_ids, budget, levels = ['BASE', 'OTHER'], 1, [F(840)]
    base = sum(w/F(n) for w, n in zip(CP.W, CP.CAPS))
    scales = {'BASE': F(2, 5)/base, 'OTHER': F(2, 5)/base}
    forced_failure = set()
    if name == 'CALIBRATION_DELAY_AND_OPAQUE_WIDTH':
        candidates[1] = CP.config('OTHER', 2, '1', F(2, 3))
        widths[1] = '1/5'
        preparations[1] = dict(signal_reuse=True, price_reuse=False, requires_calibration=True, seconds=60,
                               source='ARTIFICIAL_FIXED_60_SECOND_CALIBRATION')
        durations = priors = {'BASE': 1, 'OTHER': 1}
        seeded_ids, budget, levels, forced_failure = ['BASE'], 3, [F(800)], {'OTHER'}
    elif name == 'EARLY_UNKNOWN_SCHEDULE_PAYS_OFF':
        candidates[1]['prices'] = [F(260 if m == 12 else 1) for m in range(13) for _ in O.AXES]
        scales['OTHER'] = F(1, 100)/base
        durations = priors = {'BASE': 1, 'OTHER': 1}
        seeded_ids, budget, levels = ['BASE'], 4, [F(700)]
    elif name != 'FAST_REUSED_SHOT_FIRST':
        raise ValueError(name)
    for candidate, width, preparation in zip(candidates, widths, preparations):
        candidate['artificial_action'] = dict(width=width, preparation=preparation,
            declared_Q3_prior=priors[candidate['configuration_id']])
        if name == 'CALIBRATION_DELAY_AND_OPAQUE_WIDTH' and candidate['configuration_id'] == 'OTHER':
            candidate['artificial_action']['declared_entry_point_cost'] = O.cost(
                tuple(n for n in CP.CAPS for _ in O.AXES), candidate['prices'])
        candidate['artificial_MSE_scale'] = scales[candidate['configuration_id']]
    original = CP.mse_fn(scales)
    def function(cid, vector):
        return (F(2), F(2)) if cid in forced_failure else original(cid, vector)
    seeds = CP.seeded([c for c in candidates if c['configuration_id'] in seeded_ids], function)
    families = {c['key']['L_D']: family(c['key']['L_D']) for c in candidates}
    schedules = {c['configuration_id']: (c['key']['L_D'], families[c['key']['L_D']].R(1), F(1)) for c in candidates}
    return candidates, seeds, families, schedules, function, durations, budget, levels


def compare_case(name):
    candidates, seeds, families, schedules, function, durations, budget, levels = design(name)
    search = dict(CP.SEARCH, explore_every=2, max_steps_per_method=100, lookahead=1, lookahead_mode='HYBRID',
                  adaptive_hybrid=False, allocation_exchange=None)
    results = {}
    for policy in POLICIES:
        archive = ArtificialIndex(function, durations)
        clock = J.ReplayClock(archive, deepcopy(seeds), candidates, 1, budget)
        archive.clock = clock
        driver = ScenarioReplay(candidates, seeds, families, schedules, clock, search, CP.TAU,
                                CP.none_generator, policy, wide=dict(max_kappa='1'), batch=64, max_cycles=2)
        result = driver.run()
        result['timing_qualification'] = 'ARTIFICIAL_PREDECLARED_CONFIGURATION_SERVICE_CLASSES_AND_PREPARATION'
        result['artificial_archive_read_ids'] = archive.read_ids
        result['modeled_job_service_seconds'] = sum(durations[cid] for cid, vector in clock.finishes)
        result['modeled_preparation_seconds'] = sum(next(row['timing']['preparation_seconds']
            for row in decision['selection']['ranks'] if row['id'] == decision['selected'])
            for decision in result['decisions'] if decision['kind'] == 'NEW_SCHEDULE')
        result['total_CPU_seconds'] = None
        result['CPU_reason'] = 'Pure discrete-event artificial fixture; service/preparation time is not measured CPU.'
        result['artificial_target_reach'] = [dict(target=target, reach=next(
            (dict(seconds=row['seconds'], launched=row['launched'], cost=row['best_all'])
             for row in result['curve'] if row['best_all'] is not None and row['best_all'] <= target), None)) for target in levels]
        assert result['Q3_launched'] <= budget
        assert result['physical_scientific_calls'] == 0
        assert result['censor'] is None, result['censor']
        results[policy] = result
    assert results[POLICIES[0]]['decisions'][0]['pool_ids'] == results[POLICIES[1]]['decisions'][0]['pool_ids']
    horizon = min(result['seconds'] for result in results.values())
    return dict(name=name, conditions=dict(artificial=True, system='NONPHYSICAL_13_ROUND_FIXTURE', tau=CP.TAU,
        candidates=candidates, initial_seed_ids=list(seeds), caps=CP.CAPS, search=search, wide=dict(max_kappa='1'),
        slots=1, DP_workers=8, prefix_reuse=False, Q3_budget=budget, hidden_service_by_configuration=durations,
        service_prior_origin='Same predeclared configuration-class constants, not per-candidate archived duration',
        preparation_clock='Selected explicit calibration delay is charged by shared activate_schedule()',
        exact_width_limit='The fixture treats widths as opaque identities; it does NOT establish real mapper acceptance of exact1/5.',
        MSE_definition='CP.mse_fn(scales): exact artificial sum of round weights/shot; OTHER=2 is forced in calibration case.',
        scopes='Identical candidates, initial knowledge, generators, caps, time assumptions and budgets within each A/B pair.'),
        common_wall_horizon=horizon, best_at_common_wall_horizon={policy:min(row['best_all'] for row in result['curve']
            if row['seconds'] <= horizon and row['best_all'] is not None) for policy,result in results.items()},
        policies=results)


def plot_curves(cases, directory):
    """Standard plotting; all values come from these explicit artificial cases."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(cases), 2, figsize=(11, 8), squeeze=False)
    for row, case in enumerate(cases):
        for column, variable in enumerate(('seconds', 'launched')):
            axis = axes[row][column]
            for policy, result in case['policies'].items():
                curve = result['curve']
                axis.step([r[variable] for r in curve], [float(r['best_all']) for r in curve], where='post',
                          marker='.', label='A: ROUND05' if policy == POLICIES[0] else 'B: improvement / time')
            axis.set_title(case['name'].replace('_', ' ').lower(), fontsize=9)
            axis.set_xlabel('Modeled seconds (artificial)' if column == 0 else 'Launched Q3 jobs')
            axis.set_ylabel('Artificial cost units')
            axis.grid(alpha=.25);axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory/'CURVES.svg')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Also write standard matplotlib SVG curves')
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    names = ('FAST_REUSED_SHOT_FIRST', 'CALIBRATION_DELAY_AND_OPAQUE_WIDTH', 'EARLY_UNKNOWN_SCHEDULE_PAYS_OFF')
    cases = [compare_case(name) for name in names]
    first, calibration, early = cases
    A, B = POLICIES
    assert first['policies'][B]['artificial_target_reach'][0]['reach']['seconds'] < first['policies'][A]['artificial_target_reach'][0]['reach']['seconds']
    assert calibration['policies'][B]['best_all_completed'] < calibration['policies'][A]['best_all_completed']
    assert early['policies'][A]['best_all_completed'] < early['policies'][B]['best_all_completed']
    payload = dict(schema='ARTIFICIAL_JOINT_ORDERING_SCENARIOS_V1', cases=cases,
        checks=dict(equal_initial_pool=True, budget_respected=True, no_future_archive_enumeration=True,
                    fast_reuse_B_example=True, costly_calibration_B_example=True, unknown_schedule_A_counterexample=True),
        limits=['Constructed artificial examples, not performance rates or physical MSE predictions.',
                'The paired simulated wall curves charge declared service/preparation time; they are not CPU measurements.',
                'Exact1/5 is an opaque artificial label, not evidence that the actual Pauli/DF mapper accepts that width.',
                'No scientific signal, calibration, compilation, Q3 or archived OUTCOME was run/read.'])
    (args.output/'SCENARIOS.json').write_text(json.dumps(O.jsonable(payload), indent=2, ensure_ascii=False)+'\n')
    with (args.output/'CURVES.csv').open('w', newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['scenario', 'policy', 'modeled_seconds', 'Q3_launched', 'best_all_completed', 'job'])
        for case in cases:
            for policy, result in case['policies'].items():
                for row in result['curve']:
                    writer.writerow([case['name'], policy, row['seconds'], row['launched'], str(row['best_all']), row['job']])
    if args.plot:
        plot_curves(cases, args.output)
    print(json.dumps([{ 'scenario':case['name'], 'policies':{policy:dict(seconds=result['seconds'],
        best=str(result['best_all_completed']), launches=result['Q3_launched'], reach=O.jsonable(result['artificial_target_reach']))
        for policy,result in case['policies'].items()}}for case in cases], ensure_ascii=False))


if __name__ == '__main__':
    main()
