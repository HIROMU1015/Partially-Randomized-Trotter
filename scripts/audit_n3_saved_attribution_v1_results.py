#!/usr/bin/env python3
"""Audit saved posthoc scalars/timer accounting without producers or PSD replay.

Reads the frozen inputs and evaluation1. Outputs one deterministic JSON report.
Independent formulas use only stdlib; this is evidence audit, not another run.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys


def load(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rat(x):
    assert isinstance(x, list) and len(x) == 2
    assert all(type(t) is int for t in x) and x[1] > 0
    assert max(abs(t).bit_length() for t in x) <= 4096
    return Fraction(*x)


def enc(x):
    return [x.numerator, x.denominator]


def ptr(v, path):
    for token in path.strip('/').split('/'):
        v = v[int(token)] if isinstance(v, list) else v[token]
    return v


def span(values):
    return dict(minimum=min(values), median=statistics.median(values), maximum=max(values))


def audit(root):
    prep = root / 'artifacts/n3_saved_attribution_v1/2026-10-13/preparation'
    out = root / 'artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1'
    planpath = root / 'docs/research/n3_saved_attribution_v1_execution_plan_20261013.json'
    plan, reg = load(planpath), load(prep / 'saved_input_registry.json')
    for path, expected in {**reg['files'], **plan['source_sha256'], **plan['preparation_sha256']}.items():
        assert sha(root / path) == expected, path
    assert sha(prep / 'saved_input_registry.json') == plan['registry_sha256']
    receipt, result, stop = (load(out / p) for p in ('launch_receipt.json', 'result.json', 'MANDATORY_STOP.json'))
    authpath = root / 'artifacts/n3_saved_attribution_v1/2026-10-13/execution_authorization_v1/authorization.json'
    auth = load(authpath)
    assert receipt['authorization_sha256'] == sha(authpath)
    assert receipt['plan_sha256'] == sha(planpath) == auth['plan_sha256']
    for key in ('source_commit', 'source_fingerprint', 'registry_sha256'):
        assert receipt[key] == plan[key] == auth[key]
    assert auth['posthoc_execution_authorized'] is True
    assert receipt['attempt'] == 1 and receipt['retries'] == result['retries'] == 0
    assert result['status'] == stop['status'] == 'SAVED_POSTHOC_COMPLETE_AWAITING_SCIENTIFIC_JUDGEMENT'
    assert result['failure'] is None and set(result['completed_cases']) == set(reg['case_ids'])
    assert result['mandatory_stop'] and stop['mandatory_stop'] and not stop['next_stage_authorized']
    assert not result['memory']['budget_violation']
    assert all(result[k] == 0 for k in ('new_producer_calls', 'new_solver_calls', 'new_certificate_searches'))
    cases = {c['id']: c for c in load(root / reg['input_path'])['cases']}
    counts = Counter(); active = Counter(); selected_active = Counter(); inaccurate = []
    timing = []; literal_diagnostics = []; candidate_diagnostics = []; selected_pair_comparison = []
    residual_ratios = []; negative_remainders = Counter()
    for name in reg['case_ids']:
        saved = load(root / (reg['raw_prefix'] + '/' + name + '.json'))
        post = load(out / (name + '.json'))
        assert post['input_id'] == name and post['mandatory_stop']
        for k in ('new_producer_calls', 'new_eigensolver_calls', 'new_solver_calls'):
            assert post[k] == 0
        assert post['causal_time_or_width_difference_fully_identified'] is False
        counts['cases'] += 1
        for row in post['rows']:
            raw = ptr(saved, row['pointer']); summ = row['posthoc_phase_summary']
            assert row['resources'] == raw['resources']
            assert row['saved_U'] == raw['bounds']['upper']['U'] and row['saved_c'] == raw['bounds']['c']
            assert row['row_wall_seconds'] == raw['wall_seconds'] and row['row_cpu_seconds'] == raw['cpu_seconds']
            phases, events = raw['resources']['phase_times'], raw['resources']['solver_events']
            wall = sum(v['wall'] for v in phases.values()); cpu = sum(v['cpu'] for v in phases.values())
            solve = sum(e['reported_solve_time'] for e in events if e['reported_solve_time'] is not None)
            setup = sum(e['reported_setup_time'] for e in events if e['reported_setup_time'] is not None)
            expected = {'exclusive_phase_wall_sum': wall, 'exclusive_phase_cpu_sum': cpu,
                        'row_wall_minus_exclusive_phase_sum_unassigned': raw['wall_seconds'] - wall,
                        'row_cpu_minus_exclusive_phase_sum_unassigned': raw['cpu_seconds'] - cpu,
                        'reported_backend_solve_time_sum': solve, 'reported_backend_setup_time_sum': setup,
                        'numerical_proposal_minus_backend_times_mixed_unassigned': phases.get('numerical_proposal', {}).get('wall', 0) - solve - setup}
            for key, value in expected.items():
                assert summ[key] == value
                if 'unassigned' in key and value < 0:
                    negative_remainders[key] += 1
            assert summ['refutation_only_time'] == 'NOT_SEPARATELY_MEASURED'
            assert summ['hypothetical_refutation_free_or_one_call_time'] == 'NOT_COMPUTED'
            counts['mode_paths'] += 1
        for event in post['solver_events']:
            assert event['saved_event'] == ptr(saved, event['pointer'])
            assert event['tier_delta'] == ([1, 100], [1, 200], [1, 400])[event['event_index']]
            counts['solver_events'] += 1
            if event['saved_event']['status'] == 'optimal_inaccurate':
                inaccurate.append({k: event[k] for k in ('input_id', 'method', 'external', 'layer', 'repetition', 'tier_delta', 'pointer')})
        for item in post['candidates']:
            witness = ptr(saved, item['pointer']); row = ptr(saved, item['pointer'].rsplit('/candidates/', 1)[0])
            summ = item['posthoc_scalar_summary']; p = witness['parameters']
            k, s = rat(witness['kappa']), rat(witness['s'])
            U, c = rat(row['bounds']['upper']['U']), rat(row['bounds']['c'])
            g, q = c - U, k * U - s; box = max(Fraction(), k - g)
            assert k >= 0 and g > 0
            active_term = ('BOX' if box else 'ZERO_FLOOR') if q <= 0 else ('BOX' if box * (g + box) > q else 'BOX_ROOT_TIE' if box * (g + box) == q else 'ROOT')
            for key, value in {'kappa': k, 'saved_lower_s': s, 'U': U, 'c': c, 'g': g, 'q_final': q, 'box_term': box}.items():
                assert rat(summ[key]) == value
            assert summ['active_term'] == active_term and summ['saved_width'] == witness['width']
            assert item['selected_in_saved_row'] == (item['candidate_index'] == row['best_index'])
            counts['candidates'] += 1; active[item['method'] + ':' + active_term] += 1
            if item['selected_in_saved_row']:
                selected_active[item['method'] + ':' + active_term] += 1
            if witness['kind'] == 'SOS':
                ell, e = rat(p['ell']), rat(p['residual_bound'])
                assert ell - e == s and e >= 0
                assert rat(summ['q_using_saved_rounded_ell']) == k * U - ell
                assert rat(summ['residual_penalty_in_q']) == e
                assert rat(summ['saved_ell']) == ell and rat(summ['saved_residual_bound']) == e
                if q > 0:
                    residual_ratios.append(float(e / q))
            else:
                trace = sum((rat(p['Z'][i][i]) for i in range(len(p['Z']))), Fraction())
                assert cases[name]['nu'] * rat(p['tau']) - trace == s
                assert rat(summ['saved_trace_Z']) == trace and rat(summ['saved_tau']) == rat(p['tau'])
            # Saved width endpoints only: no new width search, ideal optimum, or PSD replay.
            hi, lo = rat(witness['width']['high']), rat(witness['width']['low'])
            assert 0 <= lo <= hi and hi >= box and hi * (g + hi) >= q
            assert lo <= box or lo * (g + lo) <= q
            if item['layer'] == 'kernel' and item['repetition'] == 0:
                diagnostic = {k: item[k] for k in ('input_id', 'method', 'external', 'candidate_index', 'selected_in_saved_row', 'origin_tier_delta', 'pointer')}
                diagnostic['exact_scalar_summary'] = summ
                candidate_diagnostics.append(diagnostic)
                if name == 'V2_banded_6' and item['method'] == 'DENSITY_SOS' and item['external'] == 5:
                    literal_diagnostics.append({'tier': item['origin_tier_delta'], 'selected': item['selected_in_saved_row'], 'kappa_approx': float(k), 'q_final_approx': float(q), 'q_using_saved_rounded_ell_approx': float(k * U - ell), 'residual_e_approx': float(e), 'e_over_q_final_approx': float(e / q), 'saved_width_high_approx': float(hi), 'pointer': item['pointer']})
        first = [x for x in candidate_diagnostics if x['input_id'] == name and x['selected_in_saved_row']]
        for external in sorted({x['external'] for x in first}):
            pair = {x['method']: x for x in first if x['external'] == external}
            occ, sdp = (pair[m]['exact_scalar_summary'] for m in ('OCC', 'SDP_EQUIVALENT'))
            assert occ['U'] == sdp['U'] and occ['c'] == sdp['c']
            assert rat(occ['q_final']) < rat(sdp['q_final'])
            assert rat(occ['saved_width']['high']) < rat(sdp['saved_width']['low'])
            selected_pair_comparison.append({'input_id': name, 'external': external, 'same_U_c': True, 'selected_OCC_q_strictly_lower': True, 'selected_OCC_width_strictly_lower': True, 'OCC_pointer': pair['OCC']['pointer'], 'SDP_pointer': pair['SDP_EQUIVALENT']['pointer'], 'OCC_kappa': occ['kappa'], 'SDP_kappa': sdp['kappa'], 'OCC_final_q': occ['q_final'], 'SDP_final_q': sdp['q_final'], 'cause_from_unsaved_proposals_repairs': 'UNDETERMINED'})
        for layer in ('kernel', 'cold'):
            for method in ('OCC', 'SDP_EQUIVALENT', 'DENSITY_SOS'):
                paths = [b['methods'][method] for b in saved['blocks'] if b['layer'] == layer]
                totals = [p['task_wall_including_selector_serialization'] for p in paths]
                assert post['path_wall_medians_and_ranges'][layer + ':' + method] == span(totals)
                phase_names = sorted({k for p in paths for row in p['mode_records'] for k in row['resources']['phase_times']})
                phases = {key: span([sum(row['resources']['phase_times'].get(key, {}).get('wall', 0) for row in p['mode_records']) for p in paths]) for key in phase_names}
                row_unassigned = [sum(row['wall_seconds'] - sum(t['wall'] for t in row['resources']['phase_times'].values()) for row in p['mode_records']) for p in paths]
                path_unassigned = [p['task_wall_including_selector_serialization'] - sum(row['wall_seconds'] for row in p['mode_records']) for p in paths]
                events = lambda p: [e for row in p['mode_records'] for e in row['resources']['solver_events']]
                backend = {key: span([sum(e[key] for e in events(p) if e[key] is not None) for p in paths]) for key in ('reported_solve_time', 'reported_setup_time')}
                ratios = {key: statistics.median([sum(row['resources']['phase_times'].get(key, {}).get('wall', 0) for row in p['mode_records']) / p['task_wall_including_selector_serialization'] for p in paths]) for key in phase_names}
                timing.append({'input_id': name, 'layer': layer, 'method': method, 'saved_path_wall_seconds': span(totals), 'exclusive_phase_wall_seconds': phases, 'median_phase_fraction_of_saved_path': ratios, 'row_unassigned_seconds': span(row_unassigned), 'path_minus_rows_seconds_including_selector_serialization': span(path_unassigned), 'reported_backend_seconds': backend, 'candidate_counts_per_path': [sum(len(row['candidates']) for row in p['mode_records']) for p in paths], 'refutation_record_counts_per_path': [sum(len(row['refutations']) for row in p['mode_records']) for p in paths], 'serialized_measured_view_bytes': span([p['serialized_measured_view_bytes'] for p in paths]), 'refutation_only_seconds': 'NOT_SEPARATELY_MEASURED', 'equal_output_obligation_counterfactual_time': 'NOT_COMPUTED'})
    assert counts == {'cases': 6, 'mode_paths': 216, 'solver_events': 432, 'candidates': 696}
    assert len(candidate_diagnostics) == 116 and len(selected_pair_comparison) == 12
    assert len(inaccurate) == 18 and all(x['input_id'] == 'V2_banded_6' and x['method'] == 'DENSITY_SOS' and x['external'] == 5 for x in inaccurate)
    recheck = load(out / 'selected_existing_proof_recheck.json')
    assert recheck['SOS_audit']['status'] == recheck['other_mode_OCC_audit']['status'] == 'EXACT_CERTIFICATE_PASS'
    assert recheck['OCC_dual_audit']['status'] == 'IDEAL_OCC_CONDITION_INFEASIBLE_AT_TARGET'
    assert not recheck['physical_impossibility_claimed'] and not recheck['additional_H_reduction_claimed']
    assert all(k not in sys.modules for k in ('numpy', 'scipy', 'cvxpy', 'scs'))
    return {'status': 'INDEPENDENT_SAVED_SCALAR_TIMING_IDENTITY_AUDIT_PASS', 'counts': dict(counts), 'source_files_preserved': len(plan['source_sha256']), 'saved_files_preserved': len(reg['files']), 'evaluation_files_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}, 'active_term_counts_including_six_timing_repetitions': dict(active), 'selected_row_active_term_counts': dict(selected_active), 'distinct_candidate_diagnostics_kernel_rep0': candidate_diagnostics, 'selected_OCC_SDP_same_mode_kernel_rep0': selected_pair_comparison, 'inaccurate_event_pointers': inaccurate, 'timing_by_case_layer_method': timing, 'banded6_external5_SOS_kernel_rep0': literal_diagnostics, 'SOS_residual_e_over_positive_final_q_range': span(residual_ratios), 'negative_unassigned_remainders_retained': dict(negative_remainders), 'selected_existing_proof_recheck_status': recheck['status'], 'new_PSD_replays_in_this_audit': 0, 'new_eigensolver_or_solver_calls': 0, 'ideal_minimum_width_computed': False, 'new_independent_inputs': 0, 'causal_width_or_time_difference_fully_identified': False, 'mandatory_stop': True, 'next_stage_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
