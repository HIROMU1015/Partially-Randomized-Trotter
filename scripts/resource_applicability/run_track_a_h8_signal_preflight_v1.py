#!/usr/bin/env python3
"""Own-run monitored H8 preflight. Each scope is one-shot and stops on failure."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.h8_signal_preflight_contract_v1 import verify_binding, RUNNER
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.h6_matched_execution_v2 import Progress
from trottertracks.resource_applicability.h6_server_resources_v1 import (
    process_identity, process_memory, install_parent_guard, fresh_resources, THREAD_KEYS)
from trottertracks.resource_applicability.h48_preflight_contract_v1 import verify_candidate_runtime


def read(path):
    p = Path(path)
    if p.is_symlink() or p.stat().st_size > 8*2**20:
        raise ValueError('H8_BOUND_JSON')
    return json.loads(p.read_text())


def validate_result(directory, task, m):
    t = read(directory/'worker_terminal.json')
    if (t['status'] != 'COMPLETE' or t['reason'] is not None or t['source_commit'] != m['source_commit']
        or t['scope'] != m['scope'] or t['H8_full_campaign_authorized'] is not False):
        raise ValueError('H8_INCOMPLETE_TASK')
    if m['scope'] == 'SIGNAL':
        r = read(directory/'signal_result.json'); s = read(directory/'signal_summary.json')
        if r['signal_count'] != 8 or {x['cell']['id'] for x in s['signals']} != {c['id'] for c in m['plan']['cells']}:
            raise ValueError('H8_SIGNAL_COVERAGE')
        return
    result = read(directory/'cost_result.json'); trajectory = read(directory/'trajectory.json')
    if digest(result['task']) != digest(task) or digest(trajectory['task']) != digest(task):
        raise ValueError('H8_COST_TASK_BINDING')
    if trajectory['event_digest'] != digest(trajectory['events']):
        raise ValueError('H8_COST_EVENT_BINDING')
    policies = ['symmetric_directional', 'ordinary'] if task['ordinary'] else ['symmetric_directional']
    expected = {(p, a) for p in policies for a in ('cosine', 'sine')}
    keys = [(x['control'], x['axis']) for x in result['rows']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('H8_COST_PAIRED_AXES')
    for row in result['rows']:
        if (any(row[k] != task[k] for k in ('cell_id', 'phase', 'replica', 'seed'))
            or row['compiler_hash'] != '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
            or row['event_digest'] != trajectory['event_digest'] or len(row['fingerprint']) != 64
            or set(row['metrics']) != {'RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size'}
            or any(type(v) is not int or v < 0 for v in row['metrics'].values())
            or read(directory/f"wrapper_{row['control']}_{row['axis']}.json") != row):
            raise ValueError('H8_COST_ROW_BINDING')
    for p in policies:
        err = read(directory/f'validation_{p}.json')['max_error']
        if not isinstance(err, (float, int)) or not 0 <= err <= 1e-9:
            raise ValueError('H8_NATIVE_CONTROL_VALIDATION')
    calls = result['calls_attempted']
    if (calls['compile'] != len(expected) or calls['trajectory'] != int(bool(task['cell']['R']))
        or calls['occurrence'] != (task['cell']['q'] if task['cell']['R'] else 0)
        or calls['control_probe'] != 4*len(policies)):
        raise ValueError('H8_COST_ACTUAL_COUNTS')


def worker(m, g, directory):
    from trottertracks.resource_applicability.ax2b_limits import install_worker_limits
    from trottertracks.resource_applicability.h6_matched_execution_v3 import ResourceTraceV3
    b = read(directory/'task_binding.json')
    if directory.parent != Path(m['output']) or directory.name != f"cost_exploration_{b['index']:04d}":
        raise ValueError('H8_WORKER_DIRECTORY')
    r = m['proposed_resources']; slot = b['slot']
    if type(slot) is not int or not 0 <= slot < r['cost_workers']:
        raise ValueError('H8_WORKER_SLOT')
    expected = dict(manifest_digest=digest(m), index=b['index'], task=b['task'], slot=slot,
        cpus=r['cost_cpu_sets'][slot], threads=1, address_space_bytes=r['cost_address_space_bytes'])
    if digest(b) != digest(expected) or digest({'index': b['index'], 'task': b['task']}) not in [digest(x) for x in m['registrations']]:
        raise ValueError('H8_WORKER_BINDING')
    install_parent_guard(read(Path(m['output'])/'parent_identity.json'))
    if any(os.environ.get(k) != '1' for k in THREAD_KEYS) or os.environ.get('QISKIT_PARALLEL') != 'FALSE':
        raise ValueError('H8_WORKER_THREADS')
    os.sched_setaffinity(0, set(b['cpus']))
    install_worker_limits(b['cpus'][0], b['address_space_bytes'], 32*2**20)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    w = AtomicWriter(directory, byte_cap=32*2**20, diagnostics_cap=5000)
    w.write('worker_claim.json', dict(source_commit=m['source_commit'], scope=m['scope'],
        identity=process_identity(os.getpid()), affinity=sorted(os.sched_getaffinity(0)),
        rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)), threads=1, manifest_sha256=digest(m)))
    p = Progress(w, cap=5000); trace = ResourceTraceV3(w.write, lambda: time.monotonic()-p.start)
    status = 'COMPLETE'; reason = None; port = None
    try:
        from trottertracks.resource_applicability.h8_signal_preflight_port_v1 import H8SignalPort, signal_port, cost_port
        with trace.bind():
            if m['scope'] == 'SIGNAL':
                port = H8SignalPort(ROOT, m, w, p)
                signal_port(ROOT, m, w, p, port=port)
            else:
                cost_port(ROOT, m, b['task'], w, p, trace)
        verify_binding(ROOT, m, g, worker=True)
    except Exception as exc:
        status = 'STOP'; reason = type(exc).__name__+':'+str(exc)[:512]
        try:
            trace.failure(exc)
        except Exception:
            pass
    w.write('worker_terminal.json', dict(status=status, reason=reason, source_commit=m['source_commit'],
        scope=m['scope'], memory=process_memory(os.getpid()),
        calls_attempted=port.calls.used if port is not None else None,
        mandatory_stop=True, retry=False, H8_full_campaign_authorized=False), terminal=True)
    return 0 if reason is None else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--grant', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True); p.add_argument('--grant-sha256', required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--worker', type=Path)
    a = p.parse_args()
    if file_hash(a.manifest) != a.manifest_sha256 or file_hash(a.grant) != a.grant_sha256:
        raise ValueError('H8_PINNED_PLAN_AUTH_BYTES')
    m, g = read(a.manifest), read(a.grant); verify_binding(ROOT, m, g, worker=bool(a.worker))
    if not a.execute:
        print('H8_BOUND_METADATA_CHECK_ONLY'); return 0
    if a.worker:
        return worker(m, g, a.worker.resolve())
    out = Path(m['output'])
    if out.exists() or (out.parent/(out.name+'.one_shot_lock')).exists():
        raise ValueError('H8_ONE_SHOT_USED')
    native = verify_candidate_runtime(ROOT)
    fresh = fresh_resources(m['proposed_resources'], m['observation_policy'], out)
    with (out.parent/(out.name+'.one_shot_lock')).open('x') as f:
        f.write(json.dumps(dict(manifest_sha256=digest(m), source_commit=m['source_commit'], retry=False))+'\n')
    out.mkdir(exist_ok=False); w = AtomicWriter(out, byte_cap=2*2**30)
    for name, value in (('manifest.json', m), ('grant.json', g), ('fresh_resources.json', fresh),
                        ('fresh_native_profile_subset.json', native), ('parent_identity.json', process_identity(os.getpid()))):
        w.write(name, value)
    r = m['proposed_resources']
    resource.setrlimit(resource.RLIMIT_AS, (r['coordinator_address_space_bytes'], r['coordinator_hard_address_space_bytes']))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); os.sched_setaffinity(0, set(r['coordinator_cpu_set']))
    from trottertracks.resource_applicability.h6_parallel_execution_v1 import supervise
    def command(row, directory, slot):
        return [sys.executable, '-I', '-P', '-B', str(ROOT/RUNNER), '--manifest', str(a.manifest.resolve()),
            '--grant', str(a.grant.resolve()), '--manifest-sha256', a.manifest_sha256,
            '--grant-sha256', a.grant_sha256, '--execute', '--worker', str(directory)]
    report = supervise(ROOT, out, m, command=command, admission=lambda d, t: validate_result(d, t, m))
    if report['first_stop_reason'] is None:
        verify_binding(ROOT, m, g)
    w.write('execution_terminal.json', report, terminal=True)
    print(json.dumps(dict(scope=m['scope'], output=str(out), first_stop_reason=report['first_stop_reason'],
                          completed=report['completed_indices'], all_reaped=report['all_dispatched_reaped'])))
    return 0 if report['first_stop_reason'] is None and report['all_dispatched_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
