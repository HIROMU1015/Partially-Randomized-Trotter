"""Completed DET receipt audit/transcription only. No QPY/NPZ or science import."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import h46_m5_cost_contract as c

GATES = ('saved_normalization_log_B', 'native_low', 'native_leakage',
         'independent_occupation_vector', 'hadamard_cosine',
         'compiled_vector_cosine', 'hadamard_sine', 'compiled_vector_sine')
METRICS = ('rz_count', 'cx_count', 'rz_depth', 'cx_depth', 'total_depth', 'circuit_size')


def hash_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_gate(row):
    tolerance = c.COMPILE_TOLERANCE if row['check'].startswith('compiled_vector_') else c.OTHER_TOLERANCE
    c.require(row['check'] in GATES and row['PASS'] is True and row['phase_alignment'] is False, 'SAVED_GATE_NOT_ACCEPTED')
    c.require(row['strictly_below'] == tolerance, 'SAVED_GATE_FIXED_THRESHOLD')
    value = row['error']
    c.require(type(value) in (int, float) and math.isfinite(value) and 0 <= value < tolerance, 'SAVED_GATE_VALUE')


def validate_receipt(receipt, axis):
    c.require(receipt['axis'] == axis and receipt['status'] == 'DIRECT_MEASURED_CT01_SINGLE_SAVED_INPUT_STATE', 'DIRECT_AXIS_RECEIPT')
    c.require(receipt['compiler_settings_hash'] == c.design.COMPILER_HASH and receipt['native_policy'] == c.design.NATIVE_POLICY, 'SAME_NATIVE_COMPILER')
    c.require(set(receipt['metrics']) == set(METRICS) and all(type(x) is int and x >= 0 for x in receipt['metrics'].values()) and receipt['metrics']['rz_count'] > 0, 'SAVED_INTEGER_METRICS')


def validate_result(cfg, task, entry, result):
    c.require(result['status'] == 'UNIT_COST_PASS_NOT_ACCOUNTING' and result['task'] == task and
              result['SOURCE'] == cfg['SOURCE'] and result['stage'] == cfg['stage'] and
              result['plan_sha256'] == cfg['plan_sha256'] and entry['id'] == task['id'], 'RESULT_IDENTITY')
    c.require(result['paired_trajectory'] is True and result['mandatory_STOP'] is True and
              all(result[k] is False for k in ('conditional_sampling', 'cohort_pooling', 'shot_accounting', 'cost_ranking')) and
              result['seed_cohort'] is None and result['strict_bias_certificate'] is None, 'DETERMINISTIC_UNIT_COST_SCOPE')
    c.require(result['events'] == dict(N2=0, N4=0, stratum='s0') and result['log_B'] == 0, 'NO_DET_RANDOM_EVENTS')
    c.require([x['check'] for x in result['checks']] == list(GATES), 'EXACT_EIGHT_SAVED_GATES')
    c.require([x['axis'] for x in result['axis_costs']] == ['cosine', 'sine'], 'EXACT_PAIRED_AXES')


def write_csv(path, rows):
    c.require(rows and not Path(path).exists(), 'NEW_NONEMPTY_PUBLIC_CSV')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks, axis_rows, gate_rows, stage_rows, raw_inventory = [], [], [], [], []

    def check(name, value, **details):
        c.require(value, name)
        checks.append(dict(check=name, PASS=True, **details))

    for stage in ('H4_DET', 'H6_DET'):
        folder = base / stage
        cfg = c.read(folder / 'config_APPROVED_PRIVATE.json')
        c.authorize(cfg)
        c.consumed(cfg)
        check(stage + '_AUTHORIZE_AND_CONSUMED_SEALS', True, files=len(cfg['sealed_files']))
        root = Path(cfg['run_root'])
        done = c.read(root / 'completion_witness.json')
        exited = c.read(cfg['supervisor']['exit'])
        check(stage + '_SUCCESSFUL_COMPLETE', done['status'] == 'UNIT_COST_STAGE_COMPLETE_MANDATORY_STOP' and done['errors'] == {} and
              done['SOURCE'] == cfg['SOURCE'] and done['plan_sha256'] == cfg['plan_sha256'] and done['stage'] == stage)
        check(stage + '_NATIVE_EXIT_AND_REAP', exited['native_exit'] == 0 and exited['reaped'] is True and
              exited['SOURCE'] == cfg['SOURCE'] and exited['plan_sha256'] == cfg['plan_sha256'] and exited['forwarded_signals'] == [])
        check(stage + '_ALL_WORKERS_EXIT_AND_REAP', len(done['workers']) == c.read(root / 'fresh_gate.json')['workers'] and
              all(x == dict(exit=0, reaped=True) for x in done['workers'].values()))
        check(stage + '_NO_RETRY_ACCOUNTING_OR_NEXT_PHASE', done['retry'] is False and done['next_phase_started'] is False and
              done['mandatory_STOP'] is True and done['shot_calculation'] == 0 and done['cost_accounting'] is False and done['ranking'] is False and
              exited['retry'] is False and exited['next_phase_started'] is False and exited['mandatory_STOP'] is True)
        entries = {x['id']: x for x in done['results']}
        check(stage + '_EXACT_TASKS_AND_OUTPUT_DIRECTORIES', len(entries) == len(done['results']) == len(cfg['tasks']) and
              set(entries) == {t['id'] for t in cfg['tasks']} == {p.name for p in Path(cfg['output']).iterdir()})
        expected_ledger = {}
        for task in cfg['tasks']:
            for kind, count in c.task_limits(task).items():
                for index in range(count):
                    name = task['id'] + '__' + kind + '__' + str(index) + '.json'
                    expected_ledger[name] = dict(task=task['id'], kind=kind, index=index, SOURCE=cfg['SOURCE'], stage=stage,
                                                 state='RESERVED_NONREFUNDABLE', actual_call_completion_separate=True)
        check(stage + '_EXACT_LEDGER_FILE_SET', set(expected_ledger) == {p.name for p in Path(cfg['ledger']).iterdir()})
        check(stage + '_EXACT_NONREFUNDABLE_LEDGER_CONTENT', all(c.read(Path(cfg['ledger']) / name) == row for name, row in expected_ledger.items()))
        for task in cfg['tasks']:
            path = Path(cfg['output']) / task['id']
            row = task['candidate']
            result = c.read(path / 'result.json')
            entry = entries[task['id']]
            check(stage + '_' + task['id'] + '_RESULT_SHA', hash_file(path / 'result.json') == entry['result_sha256'])
            validate_result(cfg, task, entry, result)
            check(stage + '_' + task['id'] + '_IDENTITY_AND_SCOPE', result['input_binding'] == cfg['systems'][row['system']]['binding'] and
                  result['preparation_hash'] == next(x['preparation_hash'] for x in c.manifest()['candidate_table'] if x['candidate']['id'] == row['id']))
            for gate in result['checks']:
                validate_gate(gate)
                check(stage + '_' + task['id'] + '_' + gate['check'], c.read(path / (gate['check'] + '.json')) == gate)
                gate_rows.append(dict(stage=stage, candidate_id=row['id'], system=row['system'], method=row['method'], prefix=row['prefix'], q=row['q'], **gate))
            event = c.read(path / 'events_PRIVATE.json')
            check(stage + '_' + task['id'] + '_DETERMINISTIC_EVENT_RECEIPT', event == dict(events=[], event_digest=c.digest([]), conditional=False, cohort='DETERMINISTIC', N2=0, N4=0, stratum='s0') and result['event_digest'] == c.digest([]))
            for axis, receipt in zip(('cosine', 'sine'), result['axis_costs']):
                validate_receipt(receipt, axis)
                qdir = path / ('quarantine_' + axis)
                raw = c.read(qdir / 'unvalidated_metrics.json')
                check(stage + '_' + task['id'] + '_' + axis + '_RECEIPT', c.read(path / ('cost_' + axis + '.json')) == receipt and
                      raw['metrics'] == receipt['metrics'] and raw['compiler_settings_hash'] == receipt['compiler_settings_hash'] and
                      raw['wrapper_fingerprint'] == receipt['fingerprint'] and receipt['preparation_hash'] == result['preparation_hash'] and receipt['event_digest'] == result['event_digest'])
                for label in ('wrapper', 'compiled'):
                    item = raw[label]
                    p = qdir / item['file']
                    check(stage + '_' + task['id'] + '_' + axis + '_' + label + '_QPY_SHA', item['file'] == label + '.qpy' and
                          p.stat().st_size == item['bytes'] and hash_file(p) == item['sha256'])
                check(stage + '_' + task['id'] + '_' + axis + '_ACCEPTED_COMPILED_SHA', receipt['compiled_SHA'] == raw['compiled']['sha256'] and (qdir / 'vectors.npz').is_file())
                axis_rows.append(dict(stage=stage, candidate_id=row['id'], system=row['system'], method=row['method'], formula=row['formula'], prefix=row['prefix'], q=row['q'], axis=axis,
                                      **receipt['metrics'], SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'], compiler_settings_hash=receipt['compiler_settings_hash'],
                                      native_policy=receipt['native_policy'], preparation_hash=receipt['preparation_hash'], fingerprint=receipt['fingerprint'], compiled_SHA=receipt['compiled_SHA'],
                                      cost_receipt_sha256=hash_file(path / ('cost_' + axis + '.json')), result_sha256=entry['result_sha256'], cost_type='DIRECT_MEASURED_NEW_RUN01', shots=None, total_shot_cost=None))
            progress = [json.loads(line) for line in (path / 'progress.jsonl').read_text().splitlines()]
            phases = [r['phase'] for r in progress]
            check(stage + '_' + task['id'] + '_COSINE_BEFORE_SINE_AND_NO_STOP', phases.index('compiled_action_cosine') < phases.index('wrapper_sine') < phases.index('compile_sine') and not (path / 'STOP.json').exists())
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                raw_inventory.append(dict(path=str(p.relative_to(base)), bytes=p.stat().st_size, sha256=hash_file(p)))
        stage_rows.append(dict(stage=stage, SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'], tasks=len(cfg['tasks']), compile_axes=cfg['call_scope']['compile_axes'], ledger_files=len(expected_ledger), call_scope=cfg['call_scope'],
                               completion=done, external_exit=exited, completion_sha256=hash_file(root / 'completion_witness.json'), external_exit_sha256=hash_file(cfg['supervisor']['exit']),
                               config_sha256=hash_file(folder / 'config_APPROVED_PRIVATE.json'), grants_consumed=True, native_exit=0, all_workers_reaped=True))
    old = c.read(base.parent / 'preservation_before_PRIVATE.json')
    changed = [name for name, digest in old.items() if not Path(name).is_file() or hash_file(name) != digest]
    check('PRESERVED3137_OLD_SOURCE_RAW_GRANT_LEDGER_MARKER_STOP', len(old) == 3137 and changed == [])
    for stage in ('H4_PR', 'H6_PR'):
        cfg = c.read(base.parent / 'closure_v2' / ('proposal_' + stage) / 'config_DRAFT_PRIVATE.json')
        check(stage + '_REMAINS_DISABLED_UNSTARTED', cfg['authorized'] is False and cfg['grant']['valid'] is False and not Path(cfg['run_root']).exists())
    check('EXACT92_DIRECT_AXES_368_SAVED_GATES', len(axis_rows) == 92 and len(gate_rows) == 368)
    write_csv(output / 'direct_measured_axes_92.csv', axis_rows)
    write_csv(output / 'saved_native_gate_checks_368.csv', gate_rows)
    c.save(output / 'stage_completion_receipts.json', stage_rows)
    c.save(output / 'private_raw_inventory.json', raw_inventory)
    c.save(output / 'completion_audit.json', dict(status='PASS_METADATA_AUDIT_ONLY_NO_REPLAY', checks=checks,
           new_molecular_compile_in_audit=0, new_molecular_sampling=0, shot_recalculation=0, cost_reranking=0, vector_norm_recomputed=False,
           vector_NPZ_policy='HASH_RECORDED_AT_END_ONLY_NO_RESTORE_OR_REPLAY', independent_review=False, mandatory_STOP=True))
    c.save(output / 'preservation_receipt.json', dict(files=len(old), changed=changed, status='UNCHANGED', unrelated_worktrees_frozen=False,
           original_snapshot_sha256=hash_file(base.parent / 'preservation_before_PRIVATE.json')))
    print(json.dumps(dict(status='PASS', checks=len(checks), direct_axes=len(axis_rows), saved_gates=len(gate_rows),
                         raw_files=len(raw_inventory), old_files_unchanged=len(old), stages=[dict(stage=x['stage'], tasks=x['tasks'], ledger_files=x['ledger_files']) for x in stage_rows])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--private', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    audit(args.private, args.output)
