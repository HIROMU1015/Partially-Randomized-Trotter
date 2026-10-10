"""Post-STOP G10 v2 failure audit: stdlib, saved files and source text only.

No production/science imports, synthesis, matrices, sampling, budgets or lowers.
Never edits a runner output or infers scientific rankings from a failed prefix.
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path('artifacts/track_b_g10_degree_result/2026-10-10/v2')
PREP = Path('artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2')
S = 'a139b91f119d109430ae3154a045d0fdcf722233'
A = '1a2cd261ebe0cf097e71026a9150756dc8c9acc3'


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
            raise PermissionError('NPZ rejected before any access')
        return root / relative

    def read(relative):
        return path(relative).read_bytes()

    def data(relative):
        return json.loads(read(relative))

    def identity(relative, limit=None):
        digest, size = hashlib.sha256(), 0
        with path(relative).open('rb') as stream:
            while limit is None or size < limit:
                block = stream.read(32768 if limit is None else min(32768, limit-size))
                if not block:
                    break
                digest.update(block)
                size += len(block)
        return {'bytes': size, 'sha256': digest.hexdigest()}

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()

    original = data(OUT/'original_output_identity_v2.json')
    require(set(original) == {'one_shot_consumed.json', 'STOP.json',
                             'failure_receipt_v2.json', 'io_stages_v2.jsonl',
                             'result_v1.json.partial'}, 'exact five raw outputs')
    for name, record in original.items():
        require(identity(OUT/name) == record, 'raw bytes/hash unchanged: '+name)
    failure, stop = data(OUT/'failure_receipt_v2.json'), data(OUT/'STOP.json')
    marker = data(OUT/'one_shot_consumed.json')
    pre, proc = data(OUT/'preflight_receipt_v2.json'), data(OUT/'process_receipt_v2.json')
    c, auth = data(PREP/'contract_v2.json'), data(PREP/'authorization.json')
    prov = failure['provenance']
    require(failure['status'] == stop['status'] == 'G10_TECHNICAL_INCONCLUSIVE',
            'original technical classification')
    require(failure['technical_reason'] == 'TypeError: G10 JSON keys must be strings',
            'raw technical reason')
    require(failure['failed_stage'] == 'before_result_stream'
            and failure['failed_phase'] == 'validate_encode_write', 'failure stage')
    require(not path(OUT/'result_v1.json').exists()
            and not path(OUT/'COMPLETED.v2').exists(), 'no final payload or completion token')
    require(original['result_v1.json.partial']['bytes'] == 0
            and identity(OUT/'result_v1.json.partial')['sha256']
            == failure['partial_output']['written_prefix_sha256']
            and failure['partial_output']['written_bytes'] == 0, 'empty partial verified after STOP')
    require(failure['partial_output']['identity_reverified_after_failure'] is False,
            'runner did not claim whole-file post-failure verification')
    require(failure['prefix_rows_usable_for_final_research_decision'] is False
            and failure['scientific_result_committed'] is False
            and stop['scientific_result_committed'] is False, 'scientific completion veto')
    require(proc['runner_invocations'] == 1 and proc['retries'] == 0
            and proc['normal_exit'] is True and proc['exit_code'] == 0
            and proc['termination_signal'] is None, 'one handled failure, normal outer exit')
    terminal = proc['terminal_status_records']
    require(len(terminal) == 1 and terminal[0]['status'] == failure['status'],
            'outer terminal status agrees; exit zero is not science success')
    for record in (failure, marker, auth, stop, proc, terminal[0]):
        require(record['retries'] == 0 and record['mandatory_STOP'] is True, 'retry0 and STOP')
    for record in (failure, auth, marker):
        require(record['runs'] == 1, 'one run')
    for record in (failure, stop, proc, terminal[0]):
        require(record['next_science_authorized'] is False, 'no next science')
    require(pre['status'] == 'PREFLIGHT_PASS' and pre['source_commit'] == S
            and pre['authorization_commit'] == A and pre['parents'] == [S]
            and pre['worktree_clean'] is True and pre['marker_absent'] is True
            and pre['result_directory_absent'] is True, 'saved fresh clean preflight')
    require(pre['remote'].split()[0] == A, 'preflight remote A2')
    require(git('show', '-s', '--format=%P', A).split() == [S], 'single direct parent')
    changed = git('diff', '--name-only', S, A).splitlines()
    require(set(changed) == {c['authorization_path'], c['optional_receipt_path']}
            == set(pre['authorization_only_paths']), 'exact authorization-only diff')
    require(marker['authorization'] == auth and auth['status'] == 'APPROVED_FOR_ONE_G10_RUN'
            and auth['science_execution_authorized'] is True, 'approved marker binding')
    for record in (marker, prov, auth, pre):
        require(record['source_commit'] == S, 'fixed S2 identity')
        require(record['contract_sha256'] == identity(PREP/'contract_v2.json')['sha256'],
                'fixed contract hash')
    for record in (marker, prov):
        require(record['execution_HEAD'] == A, 'execution HEAD A2')
    for record in (marker, prov, pre):
        require(record['authorization_sha256'] == identity(PREP/'authorization.json')['sha256'],
                'authorization hash')
    require(prov['marker_sha256'] == identity(OUT/'one_shot_consumed.json')['sha256'],
            'consumed marker hash')
    require(prov['runtime'] == pre['runtime'] and prov['runtime']['packages_match'] is True,
            'runtime receipt unchanged')
    require(pre['runtime_executable_sha256'] == c['runtime_executable_sha256'], 'runtime executable')
    require(identity(c['tool_identity']['path'])['sha256'] == c['tool_identity']['sha256'],
            'tool identity document')
    for log, prefix in [('runner_stdout_v2.log', 'stdout'), ('runner_stderr_v2.log', 'stderr')]:
        rec = identity(OUT/log)
        require(rec['bytes'] == proc[prefix+'_bytes'] and rec['sha256'] == proc[prefix+'_sha256'],
                'outer captured log '+prefix)
    require(proc['stderr_bytes'] == 0, 'empty stderr')
    require(json.loads(read(OUT/'runner_stdout_v2.log')) == terminal[0], 'exact terminal stdout')
    invocation = data(OUT/'outer_invocation_consumed_v2.json')
    require(invocation['runner_invocations'] == 1 and invocation['command'] == proc['command']
            and invocation['retries'] == 0, 'outer exclusive invocation record')
    manifest = data(c['source_manifest'])
    prefixes = data(OUT/'source_prefix_identity_v2.json')
    appended = []
    for name, digest in manifest['sha256'].items():
        require(prefixes[name]['sha256'] == digest, 'prepublication full source '+name)
        actual = identity(name)
        if actual['sha256'] != digest:
            require(name in c['append_only_paths'] and actual['bytes'] >= prefixes[name]['bytes'],
                    'only append-only source index '+name)
            require(identity(name, prefixes[name]['bytes']) == prefixes[name], 'complete S2 prefix '+name)
            appended.append(name)
        else:
            require(True, 'current critical hash '+name)
    ledger = data(c['protected_ledger'])
    for name, record in ledger.items():
        limit = record['bytes'] if name in c['append_only_paths'] else None
        require(identity(name, limit) == record, 'old protected evidence '+name)
    require(identity(c['reuse_G9_result'])['sha256'] == c['reuse_G9_sha256'], 'fixed G9 anchor')
    require(pre['protected_before']['protected_paths'] == len(ledger)
            and pre['protected_before']['violations'] == [], 'protected preflight')
    stages = [json.loads(line) for line in read(OUT/'io_stages_v2.jsonl').splitlines()]
    require(stages[-1]['stage'] == failure['failed_stage'], 'last saved IO stage')
    require(any(s['stage'] == 'scientific_body_returned_before_output' for s in stages),
            'collect returned before failure')
    require(sum(s['stage'] == 'small_reference_matrix_and_native_accounting' for s in stages)
            == c['new_rows'] == 11, 'eleven new-row stage receipts')
    require(sum(s['stage'] == 'saved_m5_policy_rebudget_only' for s in stages) == 1,
            'saved m5 rebudget stage')
    require(failure['rows_retained_diagnostic_only'] == c['rows'] == 17
            and failure['new_synthesis_calls'] == 27 and failure['reused_keys'] == 19,
            'diagnostic counters only; no saved native rows/sequences')
    require(proc['peak_RSS_KiB'] == failure['resource']['peak_RSS_KiB']
            == max(s['peak_RSS_KiB'] for s in stages), 'whole-process/guard telemetry peak agree')
    require(proc['peak_RSS_MiB'] < c['caps']['RSS_MiB']
            and proc['wall_seconds'] < c['caps']['wall_seconds']
            and proc['CPU_seconds'] < c['caps']['cpu_seconds'], 'observed total caps not exceeded')
    raw_bytes = sum(rec['bytes'] for rec in original.values())
    require(raw_bytes < c['caps']['output_bytes'], 'raw runner output cap')
    # Static source only: preserve uncertainty about the unrecorded exact key/path.
    event_source = read('src/trottertracks/algorithm_codesign/g7_generator.py').decode()
    serial_source = read('src/trottertracks/algorithm_codesign/g10_saved.py').decode()
    io_source = read('src/trottertracks/algorithm_codesign/g10_io.py').decode()
    require('calls = Counter(reduced)' in event_source and 'calls[child] += 2' in event_source
            and "'provider_calls': dict(calls)" in event_source, 'native event contains label-keyed counts')
    require('return {str(k): serial(v) for k, v in value.items()}' in serial_source,
            'old serial explicitly normalized keys')
    require("raise TypeError('G10 JSON keys must be strings')" in io_source,
            'S2 string-key-only boundary')
    return {
        'kind': 'G10_V2_POST_STOP_SAVED_FAILURE_AUDIT',
        'status': 'SAVED_FAILURE_AND_OUTER_PROCESS_PROVENANCE_PASS', 'checks_passed': checked,
        'source_commit': S, 'authorization_commit': A,
        'original_classification': failure['status'], 'technical_reason': failure['technical_reason'],
        'failure_stage': failure['failed_stage'], 'failure_phase': failure['failed_phase'],
        'raw_output_identity': original, 'raw_runner_output_bytes': raw_bytes,
        'marker_sha256': prov['marker_sha256'], 'completed_payload': False,
        'scientifically_usable_rows': 0, 'rows_retained_diagnostic_only': 17,
        'new_synthesis_calls_reported': 27, 'reused_keys_reported': 19,
        'cache_and_sequences_saved': False, 'row_sequence_or_error_reaudit_possible': False,
        'outer_process': proc, 'guard_failure_resource': failure['resource'],
        'telemetry_records': len(stages), 'critical_paths': len(manifest['sha256']),
        'protected_paths': len(ledger), 'protected_violations': [],
        'append_only_index_paths': appended, 'source_code_contract_authorization_marker_unchanged': True,
        'static_compatibility_issue': {
            'field': 'rows[new_degree].events[*].event.provider_calls',
            'producer': 'g7_generator._event: dict(Counter(label_word)); integer child labels',
            'old_boundary': 'g10_saved.serial converts each dict key using str',
            'S2_boundary': 'g10_io.validate_tree requires existing string keys',
            'confidence': 'high: static incompatibility confirmed; exact first failing key/path not saved',
            'no_scientific_inputs_recomputed': True},
        'run_count': 1, 'retries': 0, 'post_STOP_science_calls': 0,
        'matrix_synthesis_sampler_budget_lower_rerun': False, 'outcome_reclassified': False,
        'method_or_novelty_adopted': False, 'mandatory_STOP': True, 'next_science_authorized': False,
        'limits': [
            'No result payload, native rows, new sequences or per-key resource/error records survived.',
            'Runtime synthesizer_calls=0 describes metadata verification, not the 27 reported actual calls.',
            'RSS below cap in this failed attempt does not prove successful full JSON output within cap.',
            'No scientific comparison or winner can be inferred from retained-row counters.']}


if __name__ == '__main__':
    report = audit(ROOT)
    with (ROOT/OUT/'saved_failure_audit_v2.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'checks_passed',
                     'original_classification', 'technical_reason', 'critical_paths',
                     'protected_paths', 'scientifically_usable_rows')}, ensure_ascii=False))
