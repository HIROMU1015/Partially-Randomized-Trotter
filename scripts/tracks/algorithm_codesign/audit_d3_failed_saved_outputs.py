#!/usr/bin/env python3
"""Saved-only D3 v1 failure audit; no runner, numeric backend or science import."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
RESULT = 'artifacts/track_b_d3_result/2026-10-13/v1'
SOURCE_DIR = 'artifacts/track_b_d3_source/2026-10-13/v1'
SOURCE = 'c7d2c826433dcf3b1bb30c98fc05a3bf13970ae1'
AUTHORITY = 'db584d9fb3f80c07ae55d5e1fe477bff2e386af2'
FAILED = 'D3_TECHNICAL_INCONCLUSIVE'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(snapshot):
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'PASS': bool(condition)})

    directory = ROOT / RESULT
    marker = read(directory / 'one_shot_consumed.json')
    terminal = read(directory / 'terminal.json')
    inner = read(directory / 'inner_terminal.json')
    outer = read(directory / 'outer_process_receipt.json')
    stop = read(directory / 'STOP.json')
    authorization = read(ROOT / SOURCE_DIR / 'authorization.json')
    contract = read(ROOT / SOURCE_DIR / 'execution_contract_v1.json')
    manifest = read(ROOT / SOURCE_DIR / 'source_manifest_v1.json')
    protected = read(ROOT / SOURCE_DIR / 'protected_hashes_v1.json')
    for section, ledger in (('critical', manifest['critical_sha256']),
                            ('protected', protected['sha256'])):
        for relative, expected in ledger.items():
            path = ROOT / relative
            check(section + ':' + relative,
                  not relative.lower().endswith('.npz') and digest(path) == expected)
    for entry in snapshot['files']:
        path = ROOT / entry['path']
        check('raw:' + entry['path'], path.stat().st_size == entry['bytes']
              and digest(path) == entry['sha256'])
    check('frozen source / direct authorization parent',
          subprocess.check_output(['git', 'show', '-s', '--format=%P', AUTHORITY],
                                  cwd=ROOT).decode().strip() == SOURCE)
    changed = subprocess.check_output(
        ['git', 'diff', '--name-only', SOURCE, AUTHORITY], cwd=ROOT).decode().splitlines()
    check('authorization-only changes', set(changed) == {
        SOURCE_DIR + '/authorization.json',
        'docs/tracks/algorithm_codesign/d3_execution_authorization_receipt.md'})
    check('authority binding', authorization['source_commit'] == SOURCE
          and authorization['runs'] == 1 and authorization['retries'] == 0
          and authorization['mandatory_STOP'] is True)
    identity = dict(source_commit=SOURCE, execution_HEAD=AUTHORITY,
                    contract_sha256=digest(ROOT / SOURCE_DIR / 'execution_contract_v1.json'),
                    authorization_sha256=digest(ROOT / SOURCE_DIR / 'authorization.json'),
                    source_manifest_sha256=digest(ROOT / SOURCE_DIR / 'source_manifest_v1.json'),
                    design_contract_sha256=authorization['design_contract_sha256'],
                    inventory_sha256=authorization['inventory_sha256'],
                    tool_identity_sha256=contract['acquisition']['tool_identity']['sha256'])
    records = {'marker': marker, 'terminal': terminal, 'inner': inner, 'outer': outer,
               'job_start': read(directory / 'job_start_receipt.json'),
               'ledger_job': read(directory / 'ledger_job.json'),
               'ledger_started': read(directory / 'ledger_started.json'),
               'preflight': read(directory / 'preflight.json')}
    for name, record in records.items():
        for key, value in identity.items():
            check(name + ':identity:' + key, record.get(key) == value)
        if name != 'marker':
            check(name + ':marker', record.get('marker_sha256') == digest(
                directory / 'one_shot_consumed.json'))
    check('inner/outer failure labels agree', terminal['status'] == inner['status']
          == outer['effective_status'] == outer['inner_status'] == FAILED)
    check('outer exit is failure without signal or resource cap',
          outer['exit_code'] == 1 and outer['signal'] is None
          and outer['resource_failure'] is None)
    check('raw terminal -> outer receipt link',
          terminal['outer_process_receipt_sha256'] == digest(directory / 'outer_process_receipt.json'))
    check('outer -> inner link',
          outer['inner_terminal_sha256'] == digest(directory / 'inner_terminal.json'))
    check('STOP -> terminal link', stop['terminal_sha256'] == digest(directory / 'terminal.json'))
    for name, record in (('terminal', terminal), ('inner', inner), ('outer', outer), ('STOP', stop)):
        check(name + ':STOP/no retry', record['retries'] == 0
              and record['mandatory_STOP'] is True and record['next_science_authorized'] is False)
    check('successful prefix never used', terminal['prefix_scientific_use'] is False
          and inner['prefix_scientific_use'] is False)
    jobs = sorted((directory / 'jobs').glob('key_[0-9][0-9][0-9].json'))
    starts = sorted((directory / 'jobs').glob('*.started.json'))
    tokens = sorted((directory / 'jobs').glob('*.backend_call_consumed.json'))
    outputs = sorted((directory / 'jobs').glob('*.result.json'))
    worker_records = []
    for path in jobs:
        job = read(path)
        start = read(path.with_suffix('.started.json'))
        token = read(path.with_suffix('.backend_call_consumed.json'))
        value = read(path.with_suffix('.result.json'))
        for label, item in (('job', job), ('started', start), ('token', token), ('result', value)):
            for key, expected in identity.items():
                check(path.name + ':' + label + ':' + key, item.get(key) == expected)
            check(path.name + ':' + label + ':marker', item['marker_sha256'] == digest(
                directory / 'one_shot_consumed.json'))
        check(path.name + ':owner', job['parent_pid'] == records['ledger_started']['pid'])
        check(path.name + ':job bytes', start['job_path_sha256'] == digest(path))
        check(path.name + ':single token', token['backend_call_token'] == 1 and token['retries'] == 0)
        check(path.name + ':no successful sequence', value['saved'] is None
              and value['status'] == 'KEY_TECHNICAL_INCONCLUSIVE'
              and value['reason'] == 'TypeError: cannot create mpf from Fraction(1, 10000000)'
              and value['synthesis_calls'] == 1 and value['retries'] == 0)
        worker_records.append({'job_index': job['job_index'], 'ratio': job['ratio'],
                               'key': job['key'], 'reason': value['reason'],
                               'token_count': 1, 'completed_sequence': False})
    receipts = sorted((directory / 'jobs').glob('*.process_receipt.json'))
    for path in receipts:
        receipt = read(path)
        result = path.with_name(path.name.replace('.process_receipt.json', '.result.json'))
        check(path.name + ':result bytes', receipt['result_sha256'] == digest(result))
        check(path.name + ':failed process', receipt['exit_code'] == 1)
    check('first two jobs only', len(jobs) == len(starts) == len(tokens) == len(outputs) == 2
          and [v['ratio'] for v in worker_records] == ['1/28', '1/20'])
    check('outer counts actual saved files', outer['key_jobs_started'] == len(starts)
          and outer['backend_call_tokens_consumed'] == len(tokens)
          and outer['key_result_files_saved'] == len(outputs))
    check('no ledger or cache outputs', not (directory / 'rows').exists()
          and not (directory / 'resource_map.json').exists()
          and not (directory / 'synthesis_cache_new.json').exists())
    check('no optimistic final counts', terminal['rows'] is None
          and terminal['new_synthesis_calls'] is None)
    check('zero registered pairs', terminal['registered_pair_evaluations'] == 0)
    helper = ROOT / 'src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py'
    tree = ast.parse(helper.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'synthesize')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    token_line = next(n.lineno for n in calls if isinstance(n.func, ast.Name)
                      and n.func.id == 'on_synthesis')
    backend = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'gridsynth_gates')
    conversion = next(n for n in ast.walk(backend.args[1]) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == 'mpf')
    check('static argument-conversion before backend entry', token_line < conversion.lineno
          and ast.unparse(conversion) == 'mp.mpf(epsilon)')
    # The error text and this Python call structure imply no backend entry.
    # No numeric expression or backend is executed by this audit.
    return {
        'kind': 'D3_SAVED_ONLY_FAILURE_AUDIT_V1', 'audit_date_JST': '2026-10-14',
        'execution_source': SOURCE, 'authorization_commit': AUTHORITY,
        'checks': checks, 'passed': sum(c['PASS'] for c in checks),
        'failed': sum(not c['PASS'] for c in checks),
        'status': 'PASS_FAILURE_EVIDENCE_INTEGRITY' if all(c['PASS'] for c in checks) else 'AUDIT_FAILURE',
        'raw_result_snapshot': snapshot, 'identity': identity,
        'marker_sha256': digest(directory / 'one_shot_consumed.json'),
        'final_classification_unchanged': FAILED,
        'started_jobs': len(jobs), 'reserved_backend_tokens': len(tokens),
        'saved_failure_records': len(outputs), 'completed_synthesis_sequences': 0,
        'backend_entries_inferred_from_exception_and_static_call_order': 0,
        'backend_entry_direct_instrumentation': 'not present; token is a pre-entry reservation',
        'registered_ledger_rows': 0, 'registered_pair_evaluations': 0,
        'strict_2x2_synthesis_guards_completed': 0,
        'worker_records': worker_records,
        'individual_process_receipts_saved': len(receipts),
        'missing_process_receipts': [p.name for p in jobs if not p.with_suffix('.process_receipt.json').exists()],
        'outer_status_consistent_field': outer['status_consistent'],
        'outer_status_consistent_field_meaning': 'source defines status == COMPLETE; false is not a contradiction between failure labels',
        'resource_receipt': outer,
        'audit_science_or_numeric_backend_calls': 0,
        'source_code_repair_performed': False,
        'mandatory_STOP': True, 'next_science_authorized': False,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit(read(a.snapshot))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'passed', 'failed', 'registered_ledger_rows',
                                           'completed_synthesis_sequences', 'mandatory_STOP')}))
    raise SystemExit(1 if report['failed'] else 0)
