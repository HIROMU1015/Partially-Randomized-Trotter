"""Extract preregistered summaries from saved data only; no bound/proof search.

The code was prepared after the one authorized run. It reads saved rational
width intervals, endpoint decisions and resource counters. Independent proof
validity is established by the separately frozen checker. No numerical libraries
or science producer are imported. Accepts a checkout/export root and a fresh
destination so the published extraction can be reproduced from GitHub data.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import statistics


METHODS = ('NORM', 'FIVE', 'OLD', 'OCC', 'SDP_EQUIVALENT', 'DENSITY_SOS', 'FULL_H_LOWER')
PAIRS = (('OCC', 'CHEAP_OR'), ('OCC', 'SDP_EQUIVALENT'),
         ('OCC', 'DENSITY_SOS'), ('OCC_OPERATIONAL_OR', 'CHEAP_OR'),
         ('SDP_EQUIVALENT_OPERATIONAL_OR', 'CHEAP_OR'),
         ('DENSITY_SOS_OPERATIONAL_OR', 'CHEAP_OR'))
PREFIX = 'artifacts/n3_width_comparison_v2/2026-10-12/'


def frac(x):
    return F(*x)


def enc(x):
    return [x.numerator, x.denominator] if x is not None else None


def interval(width):
    return None if width is None else (frac(width['low']), frac(width['high']))


def min_interval(endpoint):
    values = [interval(x['width']) for x in endpoint['same_mode_endpoints'] if x['width'] is not None]
    return (min(x[0] for x in values), min(x[1] for x in values)) if values else None


def comparison(a, b):
    if a is None or b is None:
        return 'UNDETERMINED_MISSING_SAVED_WIDTH'
    if a == b:
        return 'IDENTICAL_SAVED_INTERVAL'
    if a[1] < b[0]:
        return 'A_STRICTLY_NARROWER_SAVED_INTERVAL'
    if b[1] < a[0]:
        return 'B_STRICTLY_NARROWER_SAVED_INTERVAL'
    return 'OVERLAPPING_SAVED_INTERVALS'


def stats(values):
    return {'minimum': min(values), 'median': statistics.median(values), 'maximum': max(values)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(); dest = a.destination
    dest.mkdir(parents=True, exist_ok=False)
    plan_path = root/'docs/research/n3_width_comparison_v2_execution_plan_20261012.json'
    plan = json.loads(plan_path.read_bytes())
    pack = json.loads((root/plan['input_path']).read_bytes())
    methods = list(METHODS); source_inputs = []
    mode_rows = []; path_rows = []; comparisons = []; H_groups = defaultdict(list)
    acceptance = {}; exact_bound_groups = 0; counters = Counter(); event_counts = Counter()
    max_bits = 0; max_iteration = 0; unverified = 0; audit_operations = 0; audit_PSD = 0
    primitive_counts = Counter(); phases = defaultdict(lambda: defaultdict(float))
    common_bounds = []; worker_setups = []; failure_rows = []
    all_records = {}
    for case in pack['cases']:
        data = (root/PREFIX/'run1'/str(case['id']+'.json')).read_bytes()
        source_inputs.append({'path': PREFIX+'run1/'+case['id']+'.json', 'sha256': sha(data), 'bytes': len(data)})
        record = json.loads(data); all_records[case['id']] = record
        assert record['status'] == 'FINITE_COMPARISON_COMPLETE'
        assert record['source_fingerprint'] == plan['source_fingerprint']
        common_bounds.append({'input_id':case['id'], 'costs':record['common_kernel_bound_costs']})
        worker_setups.append({'input_id':case['id'], 'wall_seconds':record['worker_dependency_setup_wall_seconds']})
        for block in record['blocks']:
            layer, rep = block['layer'], block['repetition']
            for d in case['candidate_modes']:
                bound_values = []
                for name in METHODS[:-1]:
                    row = next(r for r in block['methods'][name]['mode_records'] if r['external'] == d)
                    if row['bounds'] is not None:
                        bound_values.append((frac(row['bounds']['upper']['U']), frac(row['bounds']['c'])))
                assert len(bound_values) == 6 and len(set(bound_values)) == 1
                exact_bound_groups += 1
            for name, path in block['methods'].items():
                endpoint = block['endpoints'][name]; width = min_interval(endpoint)
                row = {'input_id':case['id'], 'family':case['family'], 'full_modes':case['modes'],
                       'nu':case['nu'], 'layer':layer, 'repetition':rep, 'method':name,
                       'H_reduction':endpoint['H_any_mode_reduction'], 'selected_external':endpoint['selected_external'],
                       'modes_removed':endpoint['modes_removed'], 'fallback':endpoint['fallback'],
                       'best_saved_width_low':enc(width[0]) if width else None,
                       'best_saved_width_high':enc(width[1]) if width else None,
                       'width_high_display':float(width[1]) if width else None,
                       'task_wall_seconds':path['task_wall_including_selector_serialization'],
                       'task_cpu_seconds':path['task_cpu_including_selector_serialization'],
                       'selector_wall_seconds':path['selector_wall_seconds'],
                       'serialization_wall_seconds':path['method_output_serialization_wall_seconds'],
                       'serialized_measured_view_bytes':path['serialized_measured_view_bytes']}
                path_rows.append(row); H_groups[(case['id'], layer, name)].append(row)
                resources = [path['resources']] if name == 'FULL_H_LOWER' else [r['resources'] for r in path['mode_records']]
                for resource in resources:
                    primitive_counts.update(resource['calls'])
                    max_bits = max(max_bits, resource['proposal_max_rational_bits'])
                    for phase, times in resource['phase_times'].items():
                        for unit, value in times.items(): phases[phase][unit] += value
                    for event in resource['solver_events']:
                        event_counts.update([event['status']]); counters['solver_events'] += 1
                        max_iteration = max(max_iteration, event['iterations'] or 0)
                        counters['iterations_total'] += event['iterations'] or 0
                        counters['events_at_iteration_limit'] += event['iterations'] == plan['caps']['solver_iterations_per_call']
                for mode, e in zip(path['mode_records'], endpoint['same_mode_endpoints']):
                    assert mode['external'] == e['external']
                    d = mode['external']; w = interval(e['width']); counters[mode['status']] += 1
                    unverified += len(mode.get('unverified_proposals_retained', []))
                    counters['saved_candidates'] += len(mode['candidates']); counters['strict_refutations'] += len(mode['refutations'])
                    for audit in mode['audits']:
                        max_bits = max(max_bits, audit['max_rational_bits'])
                        audit_operations += audit['operations']; audit_PSD += audit['PSD_checks']
                    mr = {'input_id':case['id'], 'family':case['family'], 'full_modes':case['modes'],
                          'nu':case['nu'], 'layer':layer, 'repetition':rep, 'method':name, 'external':d,
                          'status':mode['status'], 'accepted':e['accepted'], 'best_index':e['best_index'],
                          'width_low_fraction':str(w[0]) if w else '', 'width_high_fraction':str(w[1]) if w else '',
                          'width_high_display':float(w[1]) if w else '', 'saved_candidates':len(mode['candidates']),
                          'strict_refutations':len(mode['refutations']), 'unverified_proposals':len(mode.get('unverified_proposals_retained', []))}
                    mode_rows.append(mr)
                    if mode['status'] != 'CERTIFICATES_FOUND' or mode.get('unverified_proposals_retained'):
                        failure_rows.append({'input_id':case['id'], 'layer':layer, 'repetition':rep, 'method':name,
                                             'external':d, 'status':mode['status'],
                                             'unverified_proposals':mode.get('unverified_proposals_retained', [])})
            key = layer+':'+str(rep)
            counts = acceptance.setdefault(key, {})
            for name, endpoint in block['endpoints'].items():
                acc = counts.setdefault(name, {'H_accepted':0, 'H_total':0, 'mode_accepted':0, 'mode_total':0})
                acc['H_accepted'] += endpoint['H_any_mode_reduction']; acc['H_total'] += 1
                acc['mode_accepted'] += sum(r['accepted'] for r in endpoint['same_mode_endpoints'])
                acc['mode_total'] += len(endpoint['same_mode_endpoints'])
            for left, right in PAIRS:
                ea, eb = block['endpoints'][left], block['endpoints'][right]
                for endpoint_scope in ('H_minimum_saved_width', 'same_mode_saved_width'):
                    pairs = [(None, min_interval(ea), min_interval(eb))] if endpoint_scope.startswith('H_') else [
                        (x['external'], interval(x['width']), interval(y['width']))
                        for x, y in zip(ea['same_mode_endpoints'], eb['same_mode_endpoints'])]
                    for external, ia, ib in pairs:
                        comparisons.append({'input_id':case['id'], 'layer':layer, 'repetition':rep,
                                            'scope':endpoint_scope, 'external':external, 'A':left, 'B':right,
                                            'relation':comparison(ia, ib),
                                            'A_interval':[enc(x) for x in ia] if ia else None,
                                            'B_interval':[enc(x) for x in ib] if ib else None})
    assert len(mode_rows) == plan['method_mode_records'] == 504
    assert len(path_rows) == 252 and exact_bound_groups == 72
    assert primitive_counts['solver_calls'] <= plan['managed_solver_call_upper_limit']
    medians = []
    for (case_id, layer, name), rows in H_groups.items():
        assert len(rows) == 3
        intervals = [(frac(r['best_saved_width_low']), frac(r['best_saved_width_high'])) for r in rows]
        medians.append({'input_id':case_id, 'layer':layer, 'method':name,
                        'task_wall_seconds':stats([r['task_wall_seconds'] for r in rows]),
                        'task_cpu_seconds':stats([r['task_cpu_seconds'] for r in rows]),
                        'selector_wall_seconds':stats([r['selector_wall_seconds'] for r in rows]),
                        'serialization_wall_seconds':stats([r['serialization_wall_seconds'] for r in rows]),
                        'saved_width_high_display':stats([float(x[1]) for x in intervals]),
                        'saved_width_high_min':enc(min(x[1] for x in intervals)),
                        'saved_width_high_max':enc(max(x[1] for x in intervals)),
                        'saved_width_repetitions_identical':len(set(intervals)) == 1,
                        'goal_decision_repetitions_identical':len({r['H_reduction'] for r in rows}) == 1,
                        'H_reduction':all(r['H_reduction'] for r in rows)})
    pair_counts = {}
    for r in comparisons:
        key='|'.join((r['layer'], str(r['repetition']), r['scope'], r['A'], r['B']))
        pair_counts.setdefault(key, Counter()).update([r['relation']])
    report = {'status':'PREREGISTERED_SAVED_DATA_EXTRACTION_COMPLETE_AWAITING_GPT_REVIEW',
              'source_commit':plan['source_commit'], 'source_fingerprint':plan['source_fingerprint'],
              'plan_sha256':sha(plan_path.read_bytes()), 'input_sha256':plan['input_sha256'],
              'run_summary':json.loads((root/PREFIX/'run1/result.json').read_bytes()),
              'input_files':source_inputs, 'mode_method_records':len(mode_rows), 'H_method_paths':len(path_rows),
              'independent_Hamiltonians':6, 'timing_repetitions_are_independent_inputs':False,
              'identical_U_c_comparison_cells_verified':exact_bound_groups,
              'acceptance_by_layer_and_repetition':acceptance,
              'pairwise_strict_saved_interval_counts':pair_counts,
              'primitive_counts_excluding_separate_common_kernel_bounds':primitive_counts,
              'solver_event_status_counts':event_counts, 'max_solver_iterations':max_iteration,
              'status_and_proof_counts':counters, 'max_monitored_rational_bits':max_bits,
              'saved_candidate_audit_operations':audit_operations, 'saved_candidate_PSD_checks':audit_PSD,
              'unverified_proposals_retained':unverified, 'failure_rows':failure_rows,
              'exclusive_phase_totals_from_method_resources':phases,
              'common_kernel_bound_costs':common_bounds, 'worker_dependency_setup':worker_setups,
              'H_method_medians_and_ranges':medians,
              'scientific_conclusion_or_GO_assigned':False, 'new_producer_or_solver_calls':0,
              'mandatory_stop':True, 'next_stage_authorized':False, 'central_theme':None}
    for filename, data in [('summary.json', report), ('H_method_paths.json', path_rows), ('pairwise_saved_intervals.json', comparisons)]:
        (dest/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (dest/'mode_method_records.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(mode_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(mode_rows)
    print(json.dumps({'mode_method_records':len(mode_rows), 'H_method_paths':len(path_rows),
                      'identical_U_c_cells':exact_bound_groups, 'managed_solver_calls':primitive_counts['solver_calls'],
                      'saved_certificates':counters['saved_candidates'], 'strict_refutations':counters['strict_refutations'],
                      'unverified_proposals':unverified, 'new_producer_or_solver_calls':0},indent=2))


if __name__ == '__main__':
    main()
