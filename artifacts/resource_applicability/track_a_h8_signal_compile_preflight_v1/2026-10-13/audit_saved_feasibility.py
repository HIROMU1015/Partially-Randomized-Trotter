#!/usr/bin/env python3
"""Read-only, stdlib H8 saved evidence audit. Never launches a science worker.

Use --private-root for the existing run and --output for a new exclusive audit
directory under home. Only JSON/CSV/proc/Git and byte hashes are inspected.
Missing costs, counters, exit codes and end timestamps remain null.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

SOURCE = '5c207fb874083022e5d48d69c879d0bcb40f894c'
INPUT_SHA = 'd4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955'
PROFILE_SHA = '4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PREFIX = 'artifacts/resource_applicability/track_a_h8_signal_compile_preflight_v1/2026-10-13/'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path, cap=8*2**20):
    p = Path(path)
    if p.is_symlink() or p.stat().st_size > cap:
        raise ValueError('AUDIT_JSON_CAP_OR_LINK:'+str(p))
    return json.loads(p.read_text())


def proc(expected):
    if expected is None:
        return {'state': 'IDENTITY_NOT_SAVED', 'native_exit_code': None}
    p = Path('/proc')/str(expected['pid'])
    try:
        text = (p/'stat').read_text(); f = text[text.rfind(')')+2:].split()
        same = int(f[19]) == expected['start_ticks'] and p.stat().st_uid == os.getuid()
        result = dict(pid=expected['pid'], expected_start_ticks=expected['start_ticks'],
            observed_start_ticks=int(f[19]), identity_matches=same,
            state=f[0] if same else 'PID_REUSED', pgrp=int(f[2]), session=int(f[3]),
            affinity=None, native_exit_code=None, exact_exit_timestamp=None)
        if same:
            result['affinity'] = sorted(os.sched_getaffinity(expected['pid']))
            if f[0] in ('Z', 'X') and len(f) > 49:
                result['kernel_wait_status'] = int(f[49])
                result['native_exit_code'] = os.waitstatus_to_exitcode(int(f[49]))
        return result
    except FileNotFoundError:
        return dict(pid=expected['pid'], expected_start_ticks=expected['start_ticks'],
            state='ABSENT', identity_matches=None, native_exit_code=None, exact_exit_timestamp=None)


def journal(path):
    rows = []
    if not path.exists():
        return rows
    with path.open('rb') as f:
        for line in f:
            if len(line) > 1024*1024:
                raise ValueError('AUDIT_JOURNAL_LINE_CAP')
            try:
                rows.append(json.loads(line))
            except ValueError:
                if not line.endswith(b'\n'):
                    break  # concurrent last append is not an observation
                raise
    return rows


def stage_summary(directory, observed_wall):
    records = [read(p) for p in sorted(directory.glob('diagnostic_[0-9][0-9][0-9][0-9].json'))]
    stack = []; stages = []; transpile_started = 0
    for r in records:
        context = r['context']
        if r['kind'] == 'begin':
            stack.append(r)
            if context.get('stage') == 'transpile':
                transpile_started += 1
        elif r['kind'] == 'end':
            if not stack or stack[-1]['context'] != context:
                raise ValueError('AUDIT_STAGE_NESTING')
            begin = stack.pop()
            stages.append(dict(context=context, begin=begin['elapsed_seconds'],
                end=r['elapsed_seconds'], duration_seconds=r['elapsed_seconds']-begin['elapsed_seconds'],
                inclusive_duration=True))
    active = [dict(context=r['context'], began_at_elapsed=r['elapsed_seconds'],
                  observed_duration_lower_seconds=max(0., observed_wall-r['elapsed_seconds'])) for r in stack]
    return dict(completed_stages=stages, active_stages=active,
                actual_transpile_start_records=transpile_started,
                instructions_max=max((r['context'].get('instructions', 0) for r in records), default=0) or None,
                diagnostic_peak_rss_bytes=max((r['memory'].get('cumulative_peak_rss_bytes', 0) for r in records), default=0) or None)


def expected_rows(plan):
    rows = []
    for index, c in enumerate(plan['cells']):
        seed = int.from_bytes(hashlib.sha256(('H8_EIGHT_CASE_FEASIBILITY_V1_20261013|'+SOURCE+'|'+INPUT_SHA+'|'+c['id']+'|0').encode()).digest()[:8], 'big')
        policies = ['symmetric_directional', 'ordinary'] if c['id'] in plan['sensitivity_cells'] else ['symmetric_directional']
        for p in policies:
            for axis in ('cosine', 'sine'):
                rows.append(dict(index=index, cell_id=c['id'], axis=axis, control=p, seed=seed,
                                 source_commit=SOURCE, input_sha256=INPUT_SHA, compiler_hash=COMPILER))
    if len(rows) != 20:
        raise ValueError('AUDIT_EXPECTED_20')
    return rows


def audit(root, private):
    errors = []
    def check(value, name):
        if not value:
            errors.append(name)
    scope_reports = {}; case_reports = []; metrics = []; bound_m = None
    source_checked = False; input_checked = False
    for scope_name in ('signal01', 'heavy01', 'remaining01'):
        d = private/scope_name
        if not d.exists():
            scope_reports[scope_name] = dict(state='NOT_DISPATCHED', native_terminal=None)
            continue
        m = read(d/'manifest.json'); g = read(d/'grant.json'); bound_m = m
        check(m['source_commit'] == SOURCE, scope_name+':SOURCE')
        check(m['H8_full_campaign_authorized'] is False and g['H8_full_campaign_authorized'] is False, scope_name+':NO_FULL_GRANT')
        check(g['manifest_sha256'] == digest(m) and g['source_commit'] == SOURCE
              and g['approved_by_user'] is True and g['one_shot'] is True and g['retry'] is False, scope_name+':AUTH')
        lock = read(private/(scope_name+'.one_shot_lock'))
        check(lock['manifest_sha256'] == digest(m) and lock['source_commit'] == SOURCE and lock['retry'] is False, scope_name+':LOCK')
        launch = read(private/('launch-'+scope_name+'.json'))
        check(launch['output'] == str(d) and launch['manifest_sha256'] == sha(private/('auth-'+scope_name)/'manifest.json')
              and launch['grant_sha256'] == sha(private/('auth-'+scope_name)/'grant.json'), scope_name+':LAUNCH_BYTES')
        check(read(private/('auth-'+scope_name)/'manifest.json') == m
              and read(private/('auth-'+scope_name)/'grant.json') == g, scope_name+':SAVED_AUTH_BYTES')
        if not source_checked:
            for name, h in m['source_hashes'].items():
                blob = subprocess.check_output(['git', 'show', SOURCE+':'+name], cwd=root)
                check(hashlib.sha256(blob).hexdigest() == h == sha(root/name), 'SOURCE:'+name)
            source_checked = True
        i = m['input_identity']
        if not input_checked:
            parent = Path(i['directory'])
            for name, h in i['file_hashes'].items():
                check(sha(parent/name) == h, 'INPUT:'+name)
            receipt = read(parent/'snapshot_receipt.json')
            check(sha(i['snapshot_path']) == INPUT_SHA and Path(i['snapshot_path']).stat().st_size == 1299688, 'SNAPSHOT_BYTES')
            check(receipt['metadata']['hamiltonian_metadata']['df_rank_actual'] == 28
                  and receipt['metadata']['sector'] == dict(n_qubits=16, nelec_alpha=4, nelec_beta=4, dimension=4900), 'INPUT_MODEL')
            check(i['hamiltonian_hash'] == receipt['metadata']['hamiltonian_hash'] and i['state_hash'] == receipt['metadata']['state_hash'], 'STATE_HAMILTONIAN_HASH')
            input_checked = True
        parent = proc(launch['identity']); term = read(d/'execution_terminal.json') if (d/'execution_terminal.json').exists() else None
        records = journal(d/'resource_observations.jsonl')
        last = records[-1] if records else None
        state = ('COMPLETE' if term['first_stop_reason'] is None and term['all_dispatched_reaped'] and not term['cleanup_errors'] else 'STOP') if term else (
            'RUNNING' if parent.get('identity_matches') and parent['state'] not in ('Z', 'X') else 'ABNORMAL_OR_UNKNOWN')
        sr = dict(state=state, coordinator_identity=launch['identity'], coordinator_observation=parent,
            native_terminal=term, one_shot_consumed=True, output=str(d), scope=m['scope'],
            source_commit=m['source_commit'], manifest_digest=digest(m), manifest_byte_sha256=sha(d/'manifest.json'),
            grant_byte_sha256=sha(d/'grant.json'), last_observation=last,
            fresh_resources=read(d/'fresh_resources.json'), observation_count=len(records),
            maximum_observation_seconds=max((r['observation_seconds'] for r in records), default=None),
            maximum_staleness_seconds=max((r['staleness_seconds'] for r in records), default=None),
            max_aggregate_rss_bytes=max((r['aggregate_rss_bytes'] for r in records), default=None),
            max_active_slots=max((r['active_slots'] for r in records), default=0))
        scope_reports[scope_name] = sr
        for registration in m['registrations']:
            index = registration['index']; task = registration['task']; work = d/f'cost_exploration_{index:04d}'
            if not work.exists():
                continue
            claim = read(work/'worker_claim.json') if (work/'worker_claim.json').exists() else None
            b = read(work/'task_binding.json') if (work/'task_binding.json').exists() else None
            if claim:
                check(claim['source_commit'] == SOURCE and claim['manifest_sha256'] == digest(m)
                      and claim['affinity'] == m['proposed_resources']['cost_cpu_sets'][b['slot']]
                      and claim['threads'] == 1 and claim['rlimit_as'] == [32*2**30]*2, scope_name+':WORKER_CLAIM')
            if b:
                check(b['manifest_digest'] == digest(m) and b['task'] == task and b['threads'] == 1, scope_name+':TASK_BINDING')
            wt = read(work/'worker_terminal.json') if (work/'worker_terminal.json').exists() else None
            if m['scope'] == 'SIGNAL':
                continue
            obs = [w for r in records for w in r['workers'] if w['index'] == index]
            hwm = max((w['memory']['VmHWM_bytes'] for w in obs), default=0)
            aspeak = max((w['memory']['VmPeak_bytes'] for w in obs), default=0)
            if wt:
                hwm = max(hwm, wt['memory']['VmHWM_bytes']); aspeak = max(aspeak, wt['memory']['VmPeak_bytes'])
            stages = stage_summary(work, last['wall_seconds'] if last else 0.)
            result = read(work/'cost_result.json') if (work/'cost_result.json').exists() else None
            calls = result['calls_attempted'] if result else None
            sampled = read(work/'trajectory.json') if (work/'trajectory.json').exists() else None
            policies = ['symmetric_directional', 'ordinary'] if task['ordinary'] else ['symmetric_directional']
            valid = True
            if result:
                valid = (result['task'] == task and sampled is not None and sampled['task'] == task
                    and sampled['event_digest'] == digest(sampled['events'])
                    and calls['compile'] == 2*len(policies) and calls['trajectory'] == int(bool(task['cell']['R']))
                    and calls['occurrence'] == (task['cell']['q'] if task['cell']['R'] else 0)
                    and calls['control_probe'] == 4*len(policies))
                check(valid, task['cell_id']+':RESULT_CALLS_AND_EVENTS')
            row_count = 0
            for p in policies:
                v = read(work/f'validation_{p}.json') if (work/f'validation_{p}.json').exists() else None
                if v:
                    check(math.isfinite(v['max_error']) and 0 <= v['max_error'] <= 1e-9, task['cell_id']+':CONTROL:'+p)
                for axis in ('cosine', 'sine'):
                    f = work/f'wrapper_{p}_{axis}.json'
                    if not f.exists():
                        continue
                    r = read(f); row_count += 1
                    check(r['cell_id'] == task['cell_id'] and r['axis'] == axis and r['control'] == p
                        and r['seed'] == task['seed'] and r['replica'] == 0 and r['phase'] == task['phase']
                        and r['compiler_hash'] == COMPILER and len(r['fingerprint']) == 64
                        and r['event_digest'] == sampled['event_digest']
                        and set(r['metrics']) == {'RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size'}
                        and all(type(x) is int and x >= 0 for x in r['metrics'].values()), task['cell_id']+':WRAPPER:'+p+':'+axis)
                    if result:
                        check(r in result['rows'], task['cell_id']+':RESULT_ROW')
                    metrics.append(dict(index=index, source_commit=SOURCE, input_sha256=INPUT_SHA,
                                        scope_name=scope_name, group_admitted=bool(wt and wt['status'] == 'COMPLETE' and result and valid), **r))
            admitted = bool(wt and wt['status'] == 'COMPLETE' and wt['reason'] is None and result and valid and row_count == 2*len(policies))
            case_reports.append(dict(index=index, task=task, scope_name=scope_name,
                worker_observation=proc(claim['identity'] if claim else None), worker_terminal=wt,
                group_admitted=admitted, complete_wrapper_records=row_count,
                expected_wrapper_records=2*len(policies), actual_calls= calls,
                compile_before_fingerprint_reservation_records=len(list(work.glob('compile_attempt_*.json'))),
                trajectory_record_saved=sampled is not None,
                trajectory_event_digest=sampled['event_digest'] if sampled else None,
                peak_rss_bytes=hwm or None, peak_as_bytes=aspeak or None,
                peak_rss_fraction_guard=hwm/m['proposed_resources']['worker_rss_stop_bytes'] if hwm else None,
                peak_as_fraction_guard=aspeak/m['proposed_resources']['cost_address_space_bytes'] if aspeak else None,
                stage_summary=stages, hash_reuse=read(work/'hash_reuse_receipt.json') if (work/'hash_reuse_receipt.json').exists() else None))
    if bound_m is None:
        raise ValueError('NO_EXISTING_MANIFEST')
    plan = bound_m['plan']; expected = expected_rows(plan)
    metric_by = {(r['cell_id'], r['control'], r['axis']): r for r in metrics}
    for e in expected:
        row = metric_by.get((e['cell_id'], e['control'], e['axis']))
        e['state'] = 'COMPLETE_ADMITTED_GROUP' if row and row['group_admitted'] else 'COMPLETE_WRAPPER_PARTIAL_GROUP' if row else 'NOT_COMPLETED'
        e['metrics'] = row['metrics'] if row else None
        e['fingerprint'] = row['fingerprint'] if row else None
    signal = read(private/'signal01/cost_exploration_0000/signal_summary.json')
    primitive = read(private/'signal01/cost_exploration_0000/primitive_validation.json')
    check(len(signal['signals']) == 8 and {r['cell']['id'] for r in signal['signals']} == {c['id'] for c in plan['cells']}, 'SIGNAL_COVERAGE')
    check(primitive['actual_time_count'] == 388 and primitive['probe_count'] == 3
          and signal['calls_attempted']['primitive'] == 1164, 'PRIMITIVE_COVERAGE')
    physical = {(str(k), t) for k, t in plan['coverage']['physical_times']}
    actual = [(r['block'], r['time']) for r in primitive['records']]
    check(len(actual) == len(set(actual)) == 388 and set(actual) == physical, 'PRIMITIVE_TIME_IDENTITIES')
    for r in primitive['records']:
        check(len(r['native_spin_errors']) == len(r['spin_CSR_errors']) == 3
              and all(math.isfinite(e) and 0 <= e <= 1e-9 for e in r['native_spin_errors']+r['spin_CSR_errors']), 'PRIMITIVE_ERRORS:'+r['block'])
    for c in signal['signals']:
        check(all(math.isfinite(e) and e <= 1e-9 for e in c['signal_discrepancies']+c['state_discrepancies']), c['cell']['id']+':SIGNAL_AGREEMENT')
        check(c['numerical_allowance_certified'] is False and c['ground_state_certified'] is False, c['cell']['id']+':NO_CERTIFICATE')
        if c['cell']['R']:
            raw = complex(c['signals']['raw']['real'], c['signals']['raw']['imag'])
            corrected = complex(c['signals']['corrected']['real'], c['signals']['corrected']['imag'])
            check(abs(raw*math.exp(c['log_B'])-corrected) <= 1e-9, c['cell']['id']+':NORMALIZATION')
    profile_path = root/'artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/environment_compiler_profile_v2.json'
    check(sha(profile_path) == PROFILE_SHA, 'PROFILE_BYTES')
    profile = read(profile_path, cap=16*2**20); runtime = []
    venv = '/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/'
    for row in profile['critical_file_hashes']:
        p = row['path']
        if p.startswith(venv) and (p.endswith('.so') or '.so.' in p) and not any(k in p.lower() for k in ('cuda', 'cupy', 'nvidia')):
            check(sha(p) == row['sha256'], 'NATIVE_RUNTIME:'+p); runtime.append(row)
    native = profile['native_lowering_dependencies']
    check(sha(native['source_path']) == native['source_sha256'], 'NATIVE_LOWERING_SOURCE')
    pipeline = read(private/'pipeline_launch02.json')
    pipeline_state = proc(dict(pid=pipeline['pid'], start_ticks=pipeline['start_ticks']))
    complete = (len(metrics) == 20 and len(case_reports) == 8 and all(c['group_admitted'] for c in case_reports)
        and all(scope_reports[s]['state'] == 'COMPLETE' for s in ('signal01','heavy01','remaining01')) and not errors)
    any_stop = any(s['state'] in ('STOP','ABNORMAL_OR_UNKNOWN') for s in scope_reports.values())
    status = 'COMPLETE_MANDATORY_STOP' if complete else 'STOP_OR_ABNORMAL' if any_stop else 'RUNNING_PARTIAL_AUDIT'
    return dict(schema='h8_saved_feasibility_audit_v1', audited_utc=datetime.now(timezone.utc).isoformat(),
        status=status, source_commit=SOURCE, input_sha256=INPUT_SHA,
        hamiltonian_hash=bound_m['input_identity']['hamiltonian_hash'], state_hash=bound_m['input_identity']['state_hash'],
        compiler_hash=COMPILER, compiler_options=plan['compiler_proposed'], profile_sha256=PROFILE_SHA,
        source_files_checked=len(bound_m['source_hashes']), input_scientific_arrays_read=False,
        native_runtime_files_checked=len(runtime), full_34273_environment_inventory_repeated=False,
        old_host_compiler_bit_equivalence_verified=False, scopes=scope_reports, cases=case_reports,
        signal_coverage=8, primitive_probes=1164, prior_synthetic_tests=15, synthetic_tests_reexecuted=0,
        validation_errors=errors, expected_compile=20, complete_wrapper_records=len(metrics),
        admitted_complete_groups=sum(c['group_admitted'] for c in case_reports),
        actual_transpile_start_records=sum(c['stage_summary']['actual_transpile_start_records'] for c in case_reports),
        cost_metric_inventory=expected, wrappers=metrics, pipeline_observation=pipeline_state,
        preserved_carry=bound_m['cumulative_carry'], H8_full_campaign_authorized=False,
        new_runner_started_by_audit=False, retry_or_resume_performed=False,
        full_campaign_absolute_command=None, full_campaign_allowed_cpus=[])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--private-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for path in (a.repository, a.private_root, a.output):
        if not path.is_absolute() or not path.resolve().is_relative_to('/home/AbeHiromu') or path.is_symlink():
            raise ValueError('HOME_PATH_REQUIRED')
    r = audit(a.repository, a.private_root)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'audit.json').write_text(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    keys = ['index','cell_id','axis','control','seed','source_commit','input_sha256','compiler_hash','state',
            'RZ','CX','RZ_depth','CX_depth','depth','size','fingerprint']
    with (a.output/'compile_inventory.csv').open('x') as f:
        out = csv.DictWriter(f, fieldnames=keys, lineterminator='\n'); out.writeheader()
        for e in r['cost_metric_inventory']:
            row = {k: e[k] for k in keys if k in e}
            row.update(e['metrics'] or {})
            out.writerow(row)
    print(json.dumps({k:r[k] for k in ('status','audited_utc','complete_wrapper_records','admitted_complete_groups',
         'actual_transpile_start_records','source_files_checked','native_runtime_files_checked','validation_errors')}))
    return 0 if not r['validation_errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
