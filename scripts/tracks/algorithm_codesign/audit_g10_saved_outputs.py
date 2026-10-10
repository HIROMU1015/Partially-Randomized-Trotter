"""Post-STOP, saved-only G10 identity/accounting audit. Standard library only.

Never imports science code, builds circuits, evaluates matrices or reruns guards,
samplers, synthesis, shot policies, or proposal lower bounds. Technical prefixes
are checked for storage consistency and never ranked or reclassified.
"""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path('artifacts/track_b_g10_degree_result/2026-10-10/v1')
PREP = Path('artifacts/track_b_g10_degree_preparation/2026-10-10')
S = '05c5ef23fce775a822ab5686f5da2f0d77675864'
A = 'f5cd0755424d1b11e2249cc115518d84fb8bb8d3'


def audit(root):
    root = Path(root)
    checked = 0

    def require(condition, label):
        nonlocal checked
        if not condition:
            raise AssertionError(label)
        checked += 1

    def read(relative):
        if str(relative).lower().endswith('.npz'):
            raise PermissionError('NPZ excluded before any access')
        return (root / relative).read_bytes()

    def digest(relative):
        return hashlib.sha256(read(relative)).hexdigest()

    def data(relative):
        return json.loads(read(relative))

    identities = data(OUT / 'original_output_identity_v1.json')
    for name, record in identities.items():
        require(digest(OUT/name) == record['sha256'] and len(read(OUT/name)) == record['bytes'],
                'original raw identity: '+name)
    d = data(OUT/'result_v1.json')
    marker, stop = data(OUT/'one_shot_consumed.json'), data(OUT/'STOP.json')
    c, auth = data(PREP/'contract_v1.json'), data(PREP/'authorization.json')
    pre, process = data(OUT/'preflight_receipt_v1.json'), data(OUT/'process_receipt_v1.json')
    require(d['status'] == 'G10_TECHNICAL_INCONCLUSIVE' == stop['status'], 'original technical classification retained')
    require(d['prefix_rows_usable_for_final_research_decision'] is False, 'prefix forbidden for scientific decision')
    require(d['technical_reason'] == 'MemoryError: SP-1 RSS cap hit; no retry', 'saved RSS cap reason retained')
    require(d['resource']['peak_RSS_KiB'] > c['caps']['RSS_MiB']*1024, 'saved RSS exceeds frozen cap')
    require(identities['result_v1.json']['bytes'] <= c['caps']['output_bytes'], 'raw output under byte cap')
    for record in (d, marker, auth, pre):
        require(record['source_commit'] == S, 'source identity')
        require(record['runs'] == 1 and record['retries'] == 0 and record['mandatory_STOP'] is True,
                'one-shot/retry0/STOP')
        require(record['contract_sha256'] == digest(PREP/'contract_v1.json'), 'contract identity')
    for record in (d, marker, pre, process):
        require(record['execution_HEAD'] == A, 'execution authorization A')
    require(pre['only_parent'] == S and pre['remote_SHA_matched'] is True
            and pre['worktree_clean'] is True and pre['result_directory_and_marker_absent'] is True,
            'saved launch gate passed before acquisition')
    require(process['runner_invocations'] == 1 and process['retries'] == 0
            and process['process_exit_code'] == 0, 'one handled technical run, no restart')
    require(d['authorization_sha256'] == marker['authorization_sha256']
            == pre['authorization_sha256'] == digest(PREP/'authorization.json'), 'authorization bytes unchanged')
    require(marker['authorization'] == auth, 'marker embeds approved authorization')
    require(d['marker_sha256'] == digest(OUT/'one_shot_consumed.json'), 'marker identity unchanged')
    require(stop['mandatory_STOP'] is True and stop['next_science_authorized'] is False
            and d['next_science_authorized'] is False and d['method_or_novelty_adopted'] is False,
            'no next science or method adoption')
    require(d['actual_quantum_shots_trajectory_GPU_DF_molecule_NPZ_LP'] == 0, 'forbidden task counters zero')
    for field in ('protected_before', 'protected_after'):
        require(d[field]['protected_paths'] == 1241 and not d[field]['violations'], 'saved protected history: '+field)

    # Original science source remains protected; result documentation may append.
    manifest = data(c['source_manifest'])
    ledger = data(c['protected_ledger'])
    prefixes = []
    for name, expected in manifest['sha256'].items():
        raw = read(name)
        if hashlib.sha256(raw).hexdigest() != expected:
            # Only declared append-only indexes can have post-result notes.
            require(name in c['append_only_paths'], 'critical source cannot change: '+name)
            # Prefix length at S is recorded in a post-execution receipt.
            rec = data(OUT/'source_prefix_identity_v1.json')[name]
            require(hashlib.sha256(raw[:rec['bytes']]).hexdigest() == expected, 'full S prefix: '+name)
            prefixes.append(name)
        else:
            require(True, 'critical identity: '+name)
    for name, record in ledger.items():
        raw = read(name)
        if name in c['append_only_paths']:
            raw = raw[:record['bytes']]
        require(hashlib.sha256(raw).hexdigest() == record['sha256'], 'protected identity: '+name)

    cache, inventory = d['synthesis_cache'], d['inventory']
    require(len(cache) == len(inventory) == 46 and len(cache) <= c['cache_entries'], '46 fixed cache entries')
    require(d['new_synthesis_calls'] == 27 <= c['caps']['synthesis_keys'] and d['reused_keys'] == 19,
            'new27/reuse19 counters')
    old = data(c['reuse_G9_result'])
    require(digest(c['reuse_G9_result']) == c['reuse_G9_sha256'], 'original G9 anchor')
    require(set(cache) == {r['ratio'] for r in inventory}, 'inventory/cache keys agree')
    require(sum(r['acquisition'] == 'new_G10_once' for r in inventory) == 27
            and sum(r['acquisition'] == 'reuse_G9_fixed_identity' for r in inventory) == 19,
            'acquisition counts agree')
    require(d['synthesis_cache_bytes'] <= c['cache_bytes'], 'reported cache below byte cap')
    for key, s in cache.items():
        seq = s['sequence']
        require(set(seq) <= set('HTtSXW') and len(seq) <= c['caps']['sequence_characters'], 'sequence alphabet/size')
        require(hashlib.sha256(seq.encode()).hexdigest() == s['sequence_sha256'], 'sequence hash')
        require(s['T_count'] == seq.count('T')+seq.count('t') and s['Tdagger_count'] == seq.count('t'), 'T/Tdag recount')
        require(s['one_qubit_count'] == len(seq)-seq.count('W') and s['global_W_count'] == seq.count('W'), '1Q/global-W recount')
        require(s['angle_key'] == 'atan:'+key+':scale:1' and s['epsilon'] == c['primitive_error'], 'angle/error identity')
        require(s['error_pass'] is True and 0 <= F(s['strict_operator_error_upper']) <= F(c['primitive_error']),
                'saved strict error passes; no re-evaluation')
        if key in old['synthesis_cache']:
            require(s == old['synthesis_cache'][key], 'reused sequence full identity')
    for r in inventory:
        require(r['sequence_sha256'] == cache[r['ratio']]['sequence_sha256'] and r['strict_error_pass'] is True,
                'inventory identity/strict error')

    names = ['ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return']
    expected = {(3, arm) for arm in names+['matched_CTS']}
    expected |= {(5, arm) for arm in names+['closed_P5_full', 'matched_CTS']}
    expected |= {(7, arm) for arm in names+['closed_P5_tail', 'matched_CTS']}
    rows = d['rows']
    require(len(rows) == 17 and {(r['degree'], r['arm']) for r in rows} == expected, 'all 17 prefix rows stored')
    total, by_degree, by_row = 0, {}, []
    old_rows = {r['arm']: r for r in old['rows'] if r['primary']}
    for r in rows:
        q = F(0)
        expected_cost = {k: F(0) for k in ('T', 'CX', '1Q')}
        maximum = 0
        for b in r['events']:
            e, price = b['event'], b['cost']
            a, proposal, weight = F(e['coefficient']), F(e['proposal']), F(e['weight'])
            require(a > 0 and proposal > 0 and a == proposal*weight, 'saved coefficient/proposal/weight consistency')
            q += proposal
            count = {'T': 0, 'CX': 0, '1Q': 0}
            for gate in b['native_ir']:
                op = gate[0]
                if op == 'R':
                    s = cache[gate[2]]
                    count['T'] += s['T_count']
                    count['1Q'] += s['one_qubit_count']
                    require(gate[3] in (-1, 1), 'saved signed R primitive')
                elif op in ('CX', 'CZ'):
                    count['CX'] += 1
                    count['1Q'] += 2*(op == 'CZ')
                else:
                    count['T'] += op in ('T', 't')
                    count['1Q'] += op not in ('W', 'w')
            require(all(price[k] == count[k] for k in count), 'literal saved IR T/CX/1Q recount')
            for k in count:
                expected_cost[k] += proposal*price[k]
            maximum = max(maximum, price['T'])
        b, N = r['budget'], r['budget']['N_per_axis']
        require(0 < q <= 1 and q == F(r['reference_acceptance']), 'saved acceptance consistent')
        require(all(expected_cost[k] == F(r['per_trial_native_cost'][k]) for k in expected_cost), 'saved per-trial cost consistency')
        require(all(2*N*expected_cost[k] == F(r['two_axis_expected_native_cost'][k]) for k in expected_cost), 'saved two-axis cost scaling')
        require(2*N*q == F(r['T_prep_readout_affine_coefficient']) == F(r['expected_accepted_calls']), 'saved prep coefficient')
        require(r['registered_worst_event_T'] == maximum and r['hard_attempt_T_upper'] == 2*N*maximum,
                'saved hard T upper consistency')
        require(r['accepted_tail_T_upper'] == b['accepted_call_cap_two_axes']*maximum, 'saved accepted-tail T upper consistency')
        require(b['hard_attempt_cap_two_axes'] == 2*N and N <= c['caps']['shot_cap_per_axis'], 'saved shot/hard-attempt caps')
        require(F(b['alpha_axis']) == F(c['alpha_axis']) and F(b['total_failure_upper']) == F(1,20), 'saved common confidence identity')
        require(r['workspace_beyond_system'] == 1 and r['physical_quantum_shots_executed'] == 0, 'workspace/shots identity')
        require('fixed_dictionary_policy_lower' in r, 'prefix lower stored, not re-evaluated or scored')
        if r['degree'] == 5:
            original = old_rows[r['arm']]
            require(r['events'] == original['events'] and r['original_G9_budget'] == original['budget'],
                    'm5 original bindings/budget retained')
        n = len(r['events'])
        total += n
        by_degree[str(r['degree'])] = by_degree.get(str(r['degree']), 0)+n
        by_row.append({'m': r['degree'], 'arm': r['arm'], 'bindings': n, 'prefix_scientifically_usable': False})
    require(total <= c['caps']['event_bindings'], 'reference binding cap')
    traces = d['production_interface_traces']
    require(len(traces) == 9 and sum(r['fixed_trials'] for r in traces) == 576, 'fixed interface trial count')
    for r in traces:
        require(r['fixed_trials'] == 64 == r['accepted']+r['pre_quantum_zero']
                and r['table_normalizer_native_cost_signal_inputs'] is False
                and r['statistical_or_scaling_inference'] is False, 'stored nonenumeration diagnostic boundary')
    require(d['across_degrees_equal_exponential_accuracy_claim'] is False, 'degree target boundary')
    return {'kind': 'G10_POST_STOP_SAVED_ONLY_AUDIT', 'status': 'SAVED_OUTPUT_IDENTITY_AND_ACCOUNTING_PASS',
            'checks_passed': checked, 'source_commit': S, 'authorization_commit': A,
            'original_classification': d['status'], 'technical_reason': d['technical_reason'],
            'original_output_identity': identities, 'rows_stored': len(rows), 'bindings_checked': total,
            'bindings_by_degree': by_degree, 'prefix_row_shape': by_row,
            'synthesis_cache_entries': len(cache), 'new_synthesis_calls': 27, 'reused_keys': 19,
            'stored_interface_trials': 576, 'critical_paths': len(manifest['sha256']),
            'protected_paths': len(ledger), 'post_result_append_only_paths': prefixes,
            'budget_lower_matrix_synthesis_sampler_rerun': False, 'outcome_reclassified': False,
            'prefix_scientifically_usable': False, 'mandatory_STOP': True, 'next_science_authorized': False}


if __name__ == '__main__':
    report = audit(ROOT)
    path = ROOT/OUT/'saved_output_audit_v1.json'
    with path.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'checks_passed', 'rows_stored', 'bindings_checked',
                                          'original_classification', 'prefix_scientifically_usable')}))
