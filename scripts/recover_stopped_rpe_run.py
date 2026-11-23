#!/usr/bin/env python3
"""Inspect already-owned RPE attempts and recover saved completions without scheduling."""
import argparse
import base64
from contextlib import ExitStack, contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from queue import Empty
from runpy import run_path
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time

OWNED = ('ASSIGNED', 'STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER', 'UNKNOWN')


def load(path):
    return json.loads(path.read_bytes())


def optional(path):
    return load(path) if path.is_file() else {}


def live_driver(run):
    candidates = ('LAUNCH.json', 'NORMAL_DRIVER.json', 'distributed/ENDPOINT.json')
    for name in candidates:
        pid = optional(run/name).get('pid')
        if type(pid) is not int or pid <= 0:
            continue
        try:
            args = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if any(Path(os.fsdecode(arg)).resolve() == run for arg in args if arg.startswith(b'/')):
            return True
    return False


def safe_name(value):
    if not isinstance(value, str) or value in ('', '.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:_.-' for c in value):
        raise ValueError('Invalid saved job, method or attempt ID')
    return value


def read_rows(run):
    database = run/'distributed/QUEUE.sqlite3'
    if not database.is_file():
        return []
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row, request=json.loads(row['request'])) for row in db.execute(
            'SELECT job_id,method,state,host_id,attempt_id,request_sha256,request,adopted_epoch,completion_sha256 FROM jobs ORDER BY enqueued_epoch')]


@contextmanager
def stopped_lock(state, run, launch):
    with ExitStack() as stack:
        central = stack.enter_context((state/'REGISTRY.lock').open('rb'))
        fcntl.flock(central, fcntl.LOCK_EX)
        for name in ('FAST_DRIVER.lock', 'SCHEDULE_DRIVER.lock', 'distributed/pbs/MANAGER.lock'):
            path = run/name
            if path.is_file():
                owner = stack.enter_context(path.open('rb'))
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if optional(run/'LAUNCH.json') != launch or live_driver(run):
            raise BlockingIOError('Original invocation ownership changed')
        yield


def fixed_modules(run):
    pin = load(run/'CONTROLLER_CONFIG.json')
    root = Path(pin['code_root']).resolve(strict=True)
    sys.path[:0] = [str(root/'src'), str(root/'scripts')]
    path = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_completion_acceptance.py'
    spec = importlib.util.spec_from_file_location('trottertracks._offline_completion_acceptance', path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    return pin, check.validate_preparation_result


def owner(row):
    return {key: row[key] for key in ('job_id', 'attempt_id', 'host_id', 'request_sha256')}


def check_assignment(row, assignment):
    if any(assignment.get(k, row[k] if k == 'host_id' else None) != v for k,v in owner(row).items()):
        raise ValueError('Saved worker assignment differs from the central owner')
    if assignment.get('request') != row['request']:
        raise ValueError('Saved worker request differs from the central request')


read_preparation_readiness = run_path(str(Path(__file__).resolve().parents[1] /
    'src/trottertracks/rpe_preparation_artifacts.py'))['read_preparation_readiness']

DRY_CHECK_OBSERVER = (Path(__file__).resolve().parents[1] /
                      'src/trottertracks/rpe_dry_check_recovery.py').read_text()

# This probe is intentionally independent from mailbox status. That action
# can start a detached supervisor for previously delivered ASSIGNED work.
PROBE = DRY_CHECK_OBSERVER + r"""
import json, os, sys
from pathlib import Path
profile = json.loads(Path(sys.argv[1]).read_bytes())
payload = json.load(sys.stdin)
root = Path(profile['work_dir']).resolve()
attempt = root/payload['job_id']/payload['attempt_id']
assignment = attempt/'ASSIGNMENT.json'
if attempt.is_symlink() or not attempt.resolve().is_relative_to(root):
    raise ValueError('Invalid saved attempt path')
if not assignment.is_file():
    print(json.dumps({'state':'NO_SAVED_ASSIGNMENT'})); sys.exit(0)
def read(name):
    p = attempt/name
    return json.loads(p.read_bytes()) if p.is_file() else {}
bound = read('ASSIGNMENT.json')
for key in ('job_id','attempt_id','request_sha256'):
    if bound.get(key) != payload[key]:
        raise ValueError('Different saved assignment')
if profile['host_id'] != payload['host_id'] or bound.get('host_id', profile['host_id']) != payload['host_id']:
    raise ValueError('Different saved host')
state = read('WORKER_STATE.json')
live = False
for name in ('SUPERVISOR_STARTED.json','SUPERVISOR_LAUNCH.json'):
    pid = read(name).get('pid')
    if type(pid) is not int: continue
    try:
        live |= os.fsencode(attempt) in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        pass
result = dict(state=state.get('state','NO_WORKER_STATE'), supervisor_live=live,
              exit_receipt=(attempt/'PROCESS_RETURN.json').is_file(), outcome=(attempt/'OUTCOME.json').is_file())
if state.get('state') == 'UNKNOWN' and not live:
    proof = dry_check_rejection(attempt, bound)
    if proof:
        result['prelaunch_detail'] = dict(state['detail'], resource_released=True, prelaunch_rejection=proof)
print(json.dumps(result))
"""

PBS_PRELAUNCH_PROBE = DRY_CHECK_OBSERVER + r"""
import json, sys
from pathlib import Path
payload = json.load(sys.stdin)
verified = {}
for owner in payload['owners']:
    for work in payload['work_dirs']:
        attempt = Path(work)/owner['job_id']/owner['attempt_id']
        marker = attempt/'ASSIGNMENT.json'
        if not marker.is_file() or attempt.is_symlink() or marker.is_symlink():
            continue
        assignment = _read_saved(marker)
        if any(assignment.get(k) != v for k,v in owner.items() if k != 'host_id'):
            continue
        if assignment.get('host_id', owner['host_id']) != owner['host_id']:
            continue
        proof = dry_check_rejection(attempt, assignment)
        if proof:
            detail = _read_saved(attempt/'WORKER_STATE.json')['detail']
            verified[owner['job_id']] = dict(detail, resource_released=True, prelaunch_rejection=proof)
            break
print(json.dumps(verified))
"""


def probe_local(profile, row):
    result = subprocess.check_output([sys.executable, '-B', '-c', PROBE, str(profile)],
                                     input=json.dumps(owner(row)), text=True)
    return json.loads(result)


def worker_results(run, rows, cache):
    from run_rpe_q3_relay import Relay
    from trottertracks.rpe_q3_worker import mailbox, result_bundle, worker_cache
    relay_path = run/'distributed/relay/CONFIG.json'
    relay = Relay(relay_path) if relay_path.is_file() else None
    workers = {w['host_id']: w for w in relay.config['workers'] if w.get('enabled', True)} if relay else {}
    local = run/'distributed/LOCAL_WORKER.json'
    local_profile = optional(local)
    for row in rows:
        host = row['host_id']
        if host.startswith('pbs-'):
            continue
        try:
            safe_name(row['job_id'])
            safe_name(row['attempt_id'])
            if local_profile.get('host_id') == host:
                status = probe_local(local, row)
                ready = status['state'] == 'COMPLETED_PENDING_TRANSFER'
                if (not ready and not status.get('supervisor_live')
                        and status.get('exit_receipt') and status.get('outcome')):
                    attempt = Path(local_profile['work_dir'])/row['job_id']/row['attempt_id']
                    check_assignment(row, load(attempt/'ASSIGNMENT.json'))
                    saved = dict(files=result_bundle(attempt),
                                 artifact_manifest=load(attempt/'OUTCOME.json').get('artifacts', {}))
                    ready = True
                elif ready:
                    saved = mailbox(local_profile, 'result', owner(row))
                fetch = lambda hashes: worker_cache(local_profile).get(hashes)
            elif host in workers:
                worker = workers[host]
                command = [worker.get('python','python3'), '-B', '-c', PROBE, worker['profile']]
                status, _ = relay.remote_command(worker, command, 'recovery-observe', owner(row))
                ready = status['state'] == 'COMPLETED_PENDING_TRANSFER' or (
                    status['state'] == 'UNKNOWN' and not status.get('supervisor_live')
                    and status.get('exit_receipt') and status.get('outcome'))
                if ready:
                    saved, _ = relay.remote(worker, 'result', owner(row))
                fetch = lambda hashes: relay.remote(worker,'artifacts-get',dict(hashes=hashes))[0]['blobs']
            else:
                yield row, dict(state='NO_PINNED_RECOVERY_ROUTE'), None
                continue
            if ready:
                for sha in cache.missing(saved.get('artifact_manifest', {}).values()):
                    cache.put(fetch([sha]))
            yield row, status, saved['files'] if ready else None
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            yield row, dict(state='RECOVERY_PENDING', reason=str(error)[:450]), None


def pbs_results(run, rows, cache):
    from trottertracks.rpe_pbs import Gateway
    rows = {row['job_id']:row for row in rows if row['host_id'].startswith('pbs-')}
    if not rows:
        return
    directory = run/'distributed/pbs'
    configured = optional(directory/'SETTINGS.json')
    if not configured:
        for row in rows.values(): yield row, dict(state='NO_PINNED_PBS_RECOVERY_ROUTE'), None
        return
    saved_allocations = optional(directory/'STATE.json').get('allocations', {})
    gateway = Gateway(configured)
    pending = {job:dict(state='NO_SAVED_PBS_COMPLETION') for job in rows}
    try:
        for allocation in gateway.call({'action':'snapshot'})['allocations']:
            token = allocation['token']
            host = saved_allocations.get(token, {}).get('spec', {}).get('profile', {}).get('host_id')
            candidates = {job:row for job,row in rows.items() if row['host_id'] == host}
            if not candidates:
                continue
            try:
                # Harvest exchanges saved bytes only. No ensure, control or qdel.
                query = dict(action='harvest', token=token,
                    owners=[dict(job_id=job, attempt_id=row['attempt_id'])
                            for job, row in sorted(candidates.items()) if job in pending])
                if not query['owners']:
                    continue
                data = gateway.call(query)
                if data['host_id'] != host:
                    raise ValueError('PBS harvest returned a different fixed host')
                for item in data.get('incomplete', []):
                    row = candidates.get(item['assignment']['job_id'])
                    if row is None or item['assignment']['attempt_id'] != row['attempt_id']:
                        continue
                    check_assignment(row, item['assignment'])
                    state = item.get('state', {})
                    pending[row['job_id']] = dict(state=state.get('state','INCOMPLETE'),
                                                  allocation=token, pbs_state=allocation['status'], exit_receipt=False)
                    if allocation['status'] in ('F', 'X') and state.get('detail', {}).get('prelaunch_rejection'):
                        pending[row['job_id']]['prelaunch_detail'] = state['detail']
                for item in data.get('attempts', []):
                    row = candidates.get(item['assignment']['job_id'])
                    if row is None or item['assignment']['attempt_id'] != row['attempt_id'] or row['job_id'] not in pending:
                        continue
                    check_assignment(row, item['assignment'])
                    manifest = json.loads(base64.b64decode(item['files']['OUTCOME.json'])).get('artifacts', {})
                    for sha in cache.missing(manifest.values()):
                        recovered = gateway.call(dict(query, hashes=[sha]))
                        cache.put({sha:recovered['blobs'][sha]})
                    del pending[row['job_id']]
                    yield row, dict(state='SAVED_COMPLETION', allocation=token, pbs_state=allocation['status']), item['files']
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, Empty) as error:
                # Fixed older gateways expose queue.Empty and may leave a
                # late response on the connection. Reconnect for the next
                # allocation and keep ownership of this one unchanged.
                gateway.close()
                reason = str(error) or 'PBS harvest response timed out'
                for job in candidates.keys() & pending.keys():
                    pending[job] = dict(state='PBS_RECOVERY_PENDING', allocation=token,
                                        reason=reason[:450])
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, Empty) as error:
        reason = str(error) or 'PBS snapshot response timed out'
        for job in pending:
            pending[job] = dict(state='PBS_RECOVERY_PENDING', reason=reason[:450])
    finally:
        gateway.close()
    # Old gateway archives may only preserve the common worker spool, or
    # omit a prelaunch rejection from terminal harvest. Inspect its original
    # assigned files directly using the same read-only evidence checks.
    pending_rejections = {job: rows[job] for job, status in pending.items()
                          if rows[job]['state'] == 'UNKNOWN' and not status.get('prelaunch_detail')}
    if pending_rejections and configured.get('ssh_alias') and configured.get('gateway_python'):
        for host in sorted({row['host_id'] for row in pending_rejections.values()}):
            work_dirs = {profile['work_dir'] for allocation in saved_allocations.values()
                         if allocation.get('status') in ('F', 'X', 'FINISHED_SAVED_WORKER')
                         for profile in [allocation.get('spec', {}).get('profile', {})]
                         if profile.get('host_id') == host and profile.get('work_dir')}
            selected = [owner(row) for row in pending_rejections.values() if row['host_id'] == host]
            if not work_dirs:
                continue
            try:
                for item in selected:
                    safe_name(item['job_id'])
                    safe_name(item['attempt_id'])
                command = [*(configured.get('ssh_command') or ['ssh']), '-T', '-o', 'BatchMode=yes',
                           '-o', 'ConnectTimeout=8', configured['ssh_alias'],
                           shlex.join([configured['gateway_python'], '-B', '-c', PBS_PRELAUNCH_PROBE])]
                raw = subprocess.check_output(command, input=json.dumps(dict(
                    owners=selected, work_dirs=sorted(work_dirs))), text=True, timeout=20, stderr=subprocess.PIPE)
                verified = json.loads(raw)
                for item in selected:
                    if item['job_id'] in verified:
                        pending[item['job_id']].update(prelaunch_detail=verified[item['job_id']])
            except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
                for item in selected:
                    pending[item['job_id']]['prelaunch_observation_error'] = str(error)[-450:]
    for job, status in pending.items():
        yield rows[job], status, None


def accept(run, row, files, store, cache, validate_preparation, pin):
    from trottertracks import rpe_method_optimizer as optimizer
    from trottertracks import rpe_q3_coordinator as coordinator
    current = store.row(row['job_id'])
    if current is None or owner(current) != owner(row):
        raise ValueError('Attempt ownership changed before recovery')
    if current['state'] == 'ACCEPTED':
        return 'ALREADY_RECEIVED'
    request = current['request']
    job, method, attempt = map(safe_name,(row['job_id'],row['method'],row['attempt_id']))
    destination = run/'jobs'/method/job
    if load(destination/'REQUEST.json') != request:
        raise ValueError('Frozen on-disk request differs from the queue')
    # Validate in a private directory before putting any worker bytes at the
    # normal transfer path. The ordinary controller will do scientific adoption.
    with tempfile.TemporaryDirectory(prefix='rpe-recovered-check-') as scratch:
        check = Path(scratch)
        coordinator.install_files(check, files)
        receipt = load(check/'PROCESS_RETURN.json')
        if type(receipt.get('returncode')) is not int or receipt['returncode'] != 0 or receipt.get('signal') is not None:
            return 'HELD_NO_CLEAN_RECEIPT'
        if request.get('route', {}).get('kind') == 'preparation':
            outcome = load(check/'OUTCOME.json')
            if outcome.get('status') != 'PREPARATION_COMPLETE':
                return 'HELD_NON_SUCCESSFUL_OUTCOME'
            prepared = load(check/'PREPARATION_RECEIPT.json')
            if (prepared.get('returncode') != 0 or prepared.get('signal') is not None
                    or (check/'ARTIFACT_MANIFEST.json').is_file()
                    and load(check/'ARTIFACT_MANIFEST.json') != outcome.get('artifacts')):
                raise ValueError('Preparation completion evidence differs')
            validate_preparation(current,receipt,outcome,cache)
        else:
            outcome, _ = optimizer.job_outcome(check)
            if outcome.get('status') not in (optimizer.PASS,optimizer.FAIL,'INCONCLUSIVE'):
                return 'HELD_NON_SUCCESSFUL_OUTCOME'
            if outcome.get('request_id') != job or outcome.get('route') != request['route']['kind']:
                raise ValueError('Q3 completion identity differs')
            cid = request['configuration_id']
            view = optimizer.Executor(pin['code_root'],run,{method:[request['route']]}, {cid:request['expected']},
                                      load(run/'PROBLEM.json'),launch_disabled=True).for_method(method)
            proposal = dict(configuration_id=cid,vector=optimizer.interleave(request['N_X'],request['N_Y']))
            if view.ingest(check,proposal,calls=0)['status'] == 'UNKNOWN':
                raise ValueError('Q3 completion failed the pinned controller checks')
        stage = run/'distributed/transfers'/job/attempt
        stage.mkdir(parents=True,exist_ok=True)
        coordinator.install_files(stage,files)
        store.complete(current['host_id'],job,attempt,current['request_sha256'],files)
        coordinator.publish_staged_files(stage,destination)
        return 'RECEIVED'


def repair_dependencies(run, rows, cache):
    """Recover named inputs from the originating accepted job's immutable mirror."""
    from trottertracks.rpe_preparation_artifacts import artifact_path
    by_id = {row['job_id']:row for row in rows}
    repaired = []
    prepared = {row['job_id']:row['request'] for row in rows if row['state'] == 'QUEUED'}
    for job, entry in read_preparation_readiness(run/'distributed').items():
        if job not in by_id:
            prepared[job] = entry['request']
    for job, request in prepared.items():
        prep = request.get('preparation') or {}
        for prior, names in prep.get('dependency_inputs', {}).items():
            source = by_id.get(prior)
            if not source or source['state'] != 'ACCEPTED':
                continue
            root = run/'jobs'/safe_name(source['method'])/safe_name(prior)
            outcome = optional(root/'OUTCOME.json')
            if (outcome.get('status') != 'PREPARATION_COMPLETE'
                    or outcome.get('request_sha256') != source['request_sha256']):
                continue
            for name in names:
                sha = outcome.get('artifacts', {}).get(name)
                if not sha or not cache.missing([sha]):
                    continue
                mirror = root/'artifacts'/artifact_path(name)
                if not mirror.is_file() or mirror.is_symlink() or not mirror.resolve().is_relative_to(root/'artifacts'):
                    continue
                # put verifies the recorded SHA and installs atomically.
                cache.put({sha:base64.b64encode(mirror.read_bytes()).decode()})
                repaired.append(dict(job_id=job, predecessor=prior, sha256=sha))
    return repaired


def publish_received(run, row):
    """Complete a publish interrupted after the existing ACCEPT transaction."""
    from trottertracks.rpe_q3_coordinator import publish_staged_files
    from trottertracks.rpe_q3_store import digest
    stage = run/'distributed/transfers'/safe_name(row['job_id'])/safe_name(row['attempt_id'])
    target = run/'jobs'/safe_name(row['method'])/safe_name(row['job_id'])
    if (target/'OUTCOME.json').exists() or not (stage/'OUTCOME.json').is_file():
        return False
    files = {p.relative_to(stage).as_posix():base64.b64encode(p.read_bytes()).decode()
             for p in stage.rglob('*') if p.is_file() and not p.is_symlink() and p != stage/'REQUEST.json'}
    if digest(files) != row['completion_sha256']:
        raise ValueError('Staged completion differs from the accepted digest')
    publish_staged_files(stage,target)
    return True


def recover(state, run, *, remote=True):
    pin, validate_preparation = fixed_modules(run)
    from trottertracks.rpe_preparation_artifacts import ArtifactCache
    from trottertracks.rpe_q3_store import JobStore
    launch = optional(run/'LAUNCH.json')
    rows = read_rows(run)
    cache = ArtifactCache(run/'distributed/artifacts')
    report = dict(run_id=run.name, invocation_id=launch.get('invocation_id'), epoch=time.time(),
                  jobs=[], repaired_inputs=[], new_scientific_calls=0)
    try:
        report['repaired_inputs'] = repair_dependencies(run,rows,cache)
    except (OSError, ValueError, KeyError) as error:
        report['delivery_hold'] = str(error)
    if not launch or live_driver(run):
        return dict(report,status='ACTIVE_OR_UNLAUNCHED')
    for row in rows:
        if row['state'] != 'ACCEPTED' or not row['attempt_id'] or not row['completion_sha256']:
            continue
        if (run/'jobs'/safe_name(row['method'])/safe_name(row['job_id'])/'OUTCOME.json').is_file():
            continue
        try:
            with stopped_lock(state,run,launch):
                if publish_received(run,row):
                    report['jobs'].append(dict(owner(row),recovery='REPUBLISHED'))
        except (OSError, ValueError, KeyError) as error:
            report['jobs'].append(dict(owner(row),recovery='HELD',reason=str(error)))
    unresolved = [row for row in rows if row['state'] in OWNED and row['host_id'] and row['attempt_id']]
    if not unresolved:
        return dict(report,status='NO_OWNED_ATTEMPTS')
    from itertools import chain
    observed = chain(worker_results(run,unresolved,cache),pbs_results(run,unresolved,cache)) if remote else ()
    for row,status,files in observed:
        item = dict(owner(row),ledger_state=row['state'],worker=status)
        if files is not None:
            try:
                with stopped_lock(state,run,launch):
                    store = JobStore(run/'distributed/QUEUE.sqlite3')
                    item['recovery'] = accept(run,row,files,store,cache,validate_preparation,pin)
            except (OSError, ValueError, KeyError, sqlite3.Error) as error:
                item['recovery'] = 'HELD'
                item['reason'] = str(error)
        elif prelaunch := status.get('prelaunch_detail'):
            try:
                with stopped_lock(state, run, launch):
                    store = JobStore(run/'distributed/QUEUE.sqlite3')
                    current = store.row(row['job_id'])
                    if current['state'] in OWNED and current['request_sha256'] == row['request_sha256']:
                        store.update(row['host_id'], row['job_id'], row['attempt_id'], 'UNKNOWN', prelaunch)
                        item['recovery'] = 'PRELAUNCH_EVIDENCE_RECORDED'
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
                item.update(recovery='HELD', reason=str(error))
        report['jobs'].append(item)
    return dict(report,status='CHECKED' if remote else 'OFFLINE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--offline',action='store_true')
    args = parser.parse_args()
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)  # Preserve the exact integers already saved by fixed Q3.
    state = args.state_directory.resolve(strict=True)
    row = load(state/'RUNS.json')[args.run_id]
    run = Path(row['run_directory']).resolve(strict=True)
    if run.name != args.run_id:
        raise ValueError('Registered run directory differs from run ID')
    print(json.dumps(recover(state,run,remote=not args.offline),ensure_ascii=False))


if __name__ == '__main__':
    main()
