"""Post-STOP G10 v3 saved-only completion/provenance/native-accounting audit.

Stdlib only. No scientific imports, matrix/error evaluation, circuit construction,
synthesis, sampling, shot-policy execution or proposal-lower recomputation.
Native saved-value arithmetic is adapted from the immutable v1 saved audit.
"""
import hashlib
import json
import subprocess
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path('artifacts/track_b_g10_degree_result/2026-10-10/v3')
PREP = Path('artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3')
S = 'b9ed01455351628c9073748f5ba5751aa794b789'
A = '53a7bc4ca8051bfd76343e98f3122c0f198e37d0'
COMPLETE = 'G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE'


def audit(root):
    root = Path(root)
    checked = 0

    def require(condition, label):
        nonlocal checked
        if not condition:
            raise AssertionError(label)
        checked += 1

    def path(relative):
        if str(relative).lower().endswith('.npz'):
            raise PermissionError('NPZ rejected before access')
        return root/relative

    def read(relative):
        return path(relative).read_bytes()

    def data(relative):
        with path(relative).open() as stream:
            return json.load(stream)

    def identity(relative, limit=None):
        digest, size = hashlib.sha256(), 0
        with path(relative).open('rb') as stream:
            while limit is None or size < limit:
                chunk = stream.read(32768 if limit is None else min(32768, limit-size))
                if not chunk:
                    break
                digest.update(chunk); size += len(chunk)
        return {'bytes': size, 'sha256': digest.hexdigest()}

    def digest(relative):
        return identity(relative)['sha256']

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()

    raw_identities = data(OUT/'original_output_identity_v3.json')
    require(set(raw_identities) == {'one_shot_consumed.json', 'io_stages_v2.jsonl',
                             'result_v1.json', 'STOP.json', 'COMPLETED.v2'}, 'five original outputs')
    for name, record in raw_identities.items():
        require(identity(OUT/name) == record, 'raw output unchanged: '+name)
    require(not path(OUT/'failure_receipt_v2.json').exists()
            and not path(OUT/'result_v1.json.partial').exists(), 'no failure receipt or partial')
    stop, marker = data(OUT/'STOP.json'), data(OUT/'one_shot_consumed.json')
    d = data(OUT/'result_v1.json')
    pre, proc = data(OUT/'preflight_receipt_v3.json'), data(OUT/'process_receipt_v3.json')
    c, auth = data(PREP/'contract_v3.json'), data(PREP/'authorization.json')
    require(stop['status'] == d['status'] == COMPLETE, 'payload and STOP complete status')
    require(stop['completion_protocol'] == 'g10-stream-and-terminal-v2'
            and stop['scientific_result_committed'] is True, 'terminal success protocol')
    require(stop['result']['path'] == 'result_v1.json'
            and {k: stop['result'][k] for k in ('bytes', 'sha256')} == raw_identities['result_v1.json'],
            'terminal binds exact saved payload')
    require(raw_identities['COMPLETED.v2']['bytes'] == 0, 'empty exclusive completion token')
    require(d['technical_reason'] is None and d['prefix_rows_usable_for_final_research_decision'] is True,
            'complete payload, no technical reason')
    require(proc['normal_exit'] is True and proc['exit_code'] == 0
            and proc['termination_signal'] is None and proc['outer_timeout_reason'] is None,
            'normal outer termination')
    terminal = proc['terminal_status_records']
    require(len(terminal) == 1 and terminal[0]['status'] == COMPLETE
            and 'failure_receipt_failed' not in terminal[0], 'outer complete status, no failure display')
    require(proc['stderr_bytes'] == 0, 'empty stderr')
    require(json.loads(read(OUT/'runner_stdout_v3.log')) == terminal[0], 'exact captured status')
    for name, field in [('runner_stdout_v3.log', 'stdout'), ('runner_stderr_v3.log', 'stderr')]:
        ident = identity(OUT/name)
        require(ident == {'bytes': proc[field+'_bytes'], 'sha256': proc[field+'_sha256']}, 'outer log '+field)
    invocation = data(OUT/'outer_invocation_consumed_v3.json')
    require(invocation['command'] == proc['command'] and invocation['runner_invocations'] == 1
            and proc['runner_invocations'] == 1, 'one exclusive invocation')
    require(pre['status'] == 'PREFLIGHT_PASS' and pre['parents'] == [S]
            and pre['worktree_clean'] is True and pre['result_directory_absent'] is True
            and pre['marker_absent'] is True and pre['remote'].split()[0] == A, 'saved fresh launch gate')
    require(git('show', '-s', '--format=%P', A).split() == [S], 'A3 direct single parent S3')
    require(set(git('diff', '--name-only', S, A).splitlines())
            == set(pre['authorization_only_paths'])
            == {c['authorization_path'], c['optional_receipt_path']}, 'exact auth-only diff')
    require(auth['status'] == 'APPROVED_FOR_ONE_G10_RUN'
            and auth['science_execution_authorized'] is True and marker['authorization'] == auth,
            'separate approval embedded in marker')
    for record in (marker, d, pre, auth):
        require(record['source_commit'] == S and record['contract_sha256'] == digest(PREP/'contract_v3.json'),
                'fixed source and contract')
    for record in (marker, d, pre, proc, invocation):
        require(record['execution_HEAD'] == A, 'execution HEAD A3')
    require(d['authorization_commit'] == A, 'payload authorization commit')
    for record in (marker, d, pre):
        require(record['authorization_sha256'] == digest(PREP/'authorization.json'), 'authorization hash')
    require(d['marker_sha256'] == digest(OUT/'one_shot_consumed.json'), 'one-shot marker identity')
    for record in (marker, d, auth, pre):
        require(record['runs'] == 1, 'one run')
    for record in (marker, d, auth, pre, proc, stop, invocation, terminal[0]):
        require(record['retries'] == 0 and record['mandatory_STOP'] is True, 'retry0 / STOP')
    for record in (d, pre, proc, stop, terminal[0]):
        require(record['next_science_authorized'] is False, 'no next science')
    require(d['method_or_novelty_adopted'] is False
            and d['actual_quantum_shots_trajectory_GPU_DF_molecule_NPZ_LP'] == 0, 'claim and forbidden tasks')
    require(d['runtime'] == pre['runtime'] and pre['runtime']['packages_match'] is True, 'saved runtime receipt')
    require(pre['runtime_executable_sha256'] == c['runtime_executable_sha256'], 'runtime executable binding')
    require(digest(c['tool_identity']['path']) == c['tool_identity']['sha256'], 'tool document identity')

    manifest = data(c['source_manifest'])
    prefixes = data(OUT/'source_prefix_identity_v3.json')
    appended = []
    for relative, expected in manifest['sha256'].items():
        require(prefixes[relative]['sha256'] == expected, 'prepublication critical '+relative)
        actual = identity(relative)
        if actual['sha256'] != expected:
            require(relative in c['append_only_paths'], 'declared index only '+relative)
            require(identity(relative, prefixes[relative]['bytes']) == prefixes[relative], 'full S3 prefix '+relative)
            appended.append(relative)
        else:
            require(True, 'unchanged source '+relative)
    ledger = data(c['protected_ledger'])
    for relative, record in ledger.items():
        limit = record['bytes'] if relative in c['append_only_paths'] else None
        require(identity(relative, limit) == record, 'old protected evidence '+relative)
    require(len(ledger) == 1352 and len(manifest['sha256']) == 180, 'protected/critical inventory')
    for receipt in (d['protected_before'], d['protected_after'], pre['protected_before']):
        require(receipt['protected_paths'] == len(ledger) and receipt['violations'] == [], 'saved protected check')
    require(digest(c['reuse_G9_result']) == c['reuse_G9_sha256'], 'fixed G9 anchor')
    telemetry = [json.loads(line) for line in read(OUT/'io_stages_v2.jsonl').splitlines()]
    require(telemetry[-1]['stage'] == 'before_success_STOP', 'last recorded success stage')
    require(any(r['stage'] == 'result_closed_identity_verified' for r in telemetry)
            and any(r['stage'] == 'result_promoted_pending_terminal' for r in telemetry), 'closed and promoted payload')
    require(proc['peak_RSS_MiB'] <= c['caps']['RSS_MiB']
            and proc['wall_seconds'] < c['caps']['wall_seconds']
            and proc['CPU_seconds'] < c['caps']['cpu_seconds'], 'whole-process total caps')
    require(proc['peak_RSS_KiB'] >= stop['resource_after_result_io']['peak_RSS_KiB']
            >= d['resource']['peak_RSS_KiB'], 'resource scopes and after-IO peak')
    raw_bytes = sum(r['bytes'] for r in raw_identities.values())
    require(raw_bytes <= c['caps']['output_bytes'], 'aggregate raw output cap')
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
    require(len(rows) == 17 and {(r['degree'], r['arm']) for r in rows} == expected, 'all 17 registered rows stored')
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
        require('fixed_dictionary_policy_lower' in r, 'fixed dictionary lower stored, not re-evaluated or scored')
        if r['degree'] == 5:
            original = old_rows[r['arm']]
            require(r['events'] == original['events'] and r['original_G9_budget'] == original['budget'],
                    'm5 original bindings/budget retained')
        n = len(r['events'])
        total += n
        by_degree[str(r['degree'])] = by_degree.get(str(r['degree']), 0)+n
        by_row.append({'m': r['degree'], 'arm': r['arm'], 'bindings': n, 'complete_saved_row': True})
    require(total <= c['caps']['event_bindings'], 'reference binding cap')
    traces = d['production_interface_traces']
    require(len(traces) == 9 and sum(r['fixed_trials'] for r in traces) == 576, 'fixed interface trial count')
    for r in traces:
        require(r['fixed_trials'] == 64 == r['accepted']+r['pre_quantum_zero']
                and r['table_normalizer_native_cost_signal_inputs'] is False
                and r['statistical_or_scaling_inference'] is False, 'stored nonenumeration diagnostic boundary')
    require(d['across_degrees_equal_exponential_accuracy_claim'] is False, 'degree target boundary')
    for row_record in rows:
        b = row_record['budget']
        require(b['uses_signal_or_full_normalizer_for_local_budget'] is False, 'no oracle budget reduction')
        require(F(b['estimation_failure_34_axes']) == F(c['estimation_failure'])
                and F(b['resource_failure_17_rows']) == F(c['resource_failure_total']), 'fixed confidence split')
        lower = row_record['fixed_dictionary_policy_lower']
        require(lower['all_full_support_proposals'] is True
                and lower['all_common_nonnegative_prep_T'] is True
                and lower['attained_optimum_or_executable_law_claim'] is False, 'saved analytic lower scope')
        for binding in row_record['events']:
            # CTS events use Pauli bookkeeping without provider_calls. The
            # generator-native events retain normalized provider label keys.
            if 'provider_calls' in binding['event']:
                require(all(isinstance(k, str) for k in binding['event']['provider_calls']), 'saved normalized label keys')
            else:
                require(row_record['arm'] == 'matched_CTS', 'only CTS events omit provider_calls')
    for key, entry in cache.items():
        if key not in old['synthesis_cache']:
            require(entry['resource']['wall_seconds'] < c['caps']['per_key_wall_seconds']
                    and entry['resource']['cpu_seconds'] < c['caps']['per_key_cpu_seconds'], 'saved per-key caps')
    require(set(d['CTS_certificates']) == {'3', '7'}, 'registered CTS certificate inventory')
    summary = []
    for r in rows:
        summary.append({
            'degree': r['degree'], 'arm': r['arm'], 'events': len(r['events']),
            'N_per_axis': r['budget']['N_per_axis'], 'acceptance': r['reference_acceptance'],
            'reference_second_moment': r['reference_m2'], 'reference_range': r['reference_range'],
            'expected_native_cost_per_attempt': r['per_trial_native_cost'],
            'two_axis_expected_native_cost': r['two_axis_expected_native_cost'],
            'T_prep_readout_affine_coefficient': r['T_prep_readout_affine_coefficient'],
            'accepted_tail_T_upper': r['accepted_tail_T_upper'], 'hard_attempt_T_upper': r['hard_attempt_T_upper'],
            'workspace_beyond_system': r['workspace_beyond_system'],
            'saved_fixed_dictionary_intercept_lower': r['fixed_dictionary_policy_lower']['intercept_lower'],
            'saved_fixed_dictionary_prep_slope_lower': r['fixed_dictionary_policy_lower']['prep_slope_lower'],
            'acquisition': r['acquisition'],
        })
    return {
        'kind': 'G10_V3_POST_STOP_SAVED_ONLY_COMPLETION_AND_PROVENANCE_AUDIT',
        'status': 'SAVED_COMPLETION_OUTER_PROCESS_PROVENANCE_AND_ACCOUNTING_PASS', 'checks_passed': checked,
        'source_commit': S, 'authorization_commit': A, 'original_classification': d['status'],
        'original_output_identity': raw_identities, 'raw_runner_output_bytes': raw_bytes,
        'marker_sha256': d['marker_sha256'], 'rows_complete': len(rows), 'axes': 2*len(rows),
        'bindings_checked': total, 'bindings_by_degree': by_degree, 'saved_row_summary': summary,
        'synthesis_cache_entries': len(cache), 'new_synthesis_calls': d['new_synthesis_calls'],
        'reused_keys': d['reused_keys'], 'stored_interface_trials': sum(r['fixed_trials'] for r in traces),
        'critical_paths': len(manifest['sha256']), 'protected_paths': len(ledger), 'protected_violations': [],
        'post_result_append_only_paths': appended, 'outer_process': proc,
        'guard_resource_before_result_io': d['resource'], 'guard_resource_after_result_io': stop['resource_after_result_io'],
        'telemetry_records': len(telemetry), 'saved_strict_error_and_IR_counts_verified': True,
        'actual_matrix_error_or_budget_lower_synthesis_sampler_rerun': False,
        'outcome_reclassified': False, 'method_or_novelty_adopted': False,
        'actual_quantum_shots_trajectory_GPU_DF_molecule_NPZ_LP': 0,
        'run_count': 1, 'retries': 0, 'post_STOP_science_calls': 0,
        'mandatory_STOP': True, 'next_science_authorized': False,
        'evidence_scope': 'source-bound local complete G10 acquisition and saved-value consistency; not immutable CI or independent scientific reproduction',
        'limitations': ['Within each Taylor degree only; no across-degree equal exponential accuracy comparison.',
                        'Fixed synthetic three-qubit native provider; no molecular DF/I0 advantage or PR/QPE total-cost claim.',
                        'Fixed interface traces are not quantum trajectories or scaling evidence.',
                        'Strict errors and analytic lower values are audited as saved values, not independently regenerated.',
                        'One successful run under 512 MiB does not guarantee all future runs or environments.'],
    }


if __name__ == '__main__':
    print(json.dumps(audit(ROOT), ensure_ascii=False, indent=2))
