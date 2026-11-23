#!/usr/bin/env python3
"""Reclaim an existing run's acknowledged worker copies without changing its release."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

from run_rpe_q3_relay import Relay, acquire_lock
from trottertracks.rpe_local_pack_maintenance import compact_local_metadata, registered_storage_owner_is_exclusive
from trottertracks.rpe_preparation_artifacts import read_preparation_readiness
from trottertracks.rpe_worker_storage import (duplicate_reclaim_target, prefix_caches, report,
    q3_result_copy_inventory, reclaim_q3_result_copies, _Q3_COPY_SWEEP_BYTES)


# Windows SSH has a command-line limit even when the mailbox input uses SCP.
# Send the report body with the existing file payload and keep the command short.
REMOTE_ENTRY = "import json, sys; payload = json.load(sys.stdin); exec(payload.pop('report_code'))"
REMOTE_REPORT = """
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
from trottertracks import rpe_worker_storage as storage
exec(compile(payload['source'], 'rpe_worker_storage.py', 'exec'), vars(storage))
profile = json.loads(Path(sys.argv[2]).read_bytes()) if sys.argv[2] else payload['profile']
spools = (profile['work_dir'],) if sys.argv[2] else payload['spools']
q3_spools = set(spools)
if payload.get('pbs_runs'):
    q3_spools.update(Path(payload['pbs_runs']).glob('*/workers/' + profile['host_id']))
if payload.get('q3_audit'):
    print(json.dumps({'q3_inventory': storage.q3_result_copy_inventory(profile, q3_spools)}))
    raise SystemExit(0)
receipts = payload.get('pbs_ack_receipts')
if receipts:
    acknowledgements = storage.acknowledge_pbs_completions(profile, spools[0], receipts)
options = dict(spools=spools, pbs_runs=payload.get('pbs_runs'), cache_directories=storage.prefix_caches(profile), compact_metadata=True, proactive=True)
result = storage.report(profile.get('storage_limit'), **options)
if result and storage.duplicate_reclaim_target(result) > 0 and payload.get('duplicate_inventory'):
    result['duplicate_inventory'] = storage.duplicate_inventory(profile)
if result and storage.duplicate_reclaim_target(result) > 0 and payload.get('verified_duplicates'):
    guarded_spools = set(spools)
    if payload.get('pbs_runs'):
        guarded_spools.update(Path(payload['pbs_runs']).glob('*/workers/*'))
    removed = storage.reclaim_verified_duplicates(profile, payload['verified_duplicates'], spools=guarded_spools)
    if removed:
        result = storage.report(profile['storage_limit'], refresh=True, **options)
        result.update(reclaimed_duplicate_artifacts=len(removed), reclaimed_duplicate_bytes=sum(removed.values()))
if result and storage.duplicate_reclaim_target(result) > 0:
    if payload.get('verified_q3_copies'):
        target = min(storage.duplicate_reclaim_target(result), storage._Q3_COPY_SWEEP_BYTES)
        removed = storage.reclaim_q3_result_copies(profile, q3_spools, payload['verified_q3_copies'], target)
        if removed['files']:
            result = storage.report(profile['storage_limit'], refresh=True, **options)
        result['q3_copies_removed'] = removed
    elif payload.get('duplicate_inventory'):
        result['q3_inventory'] = storage.q3_result_copy_inventory(profile, q3_spools)
if result and storage.duplicate_reclaim_target(result) > 0 and payload.get('pbs_namespace'):
    if payload.get('verified_completion_rpcs'):
        target = min(storage.duplicate_reclaim_target(result), 128 * 1024**2)
        removed = storage.reclaim_pbs_completion_copies(profile, payload['pbs_namespace'],
                      payload['verified_completion_rpcs'], payload['adopted_completion_owners'], target)
        if removed['pairs']:
            q3_removed = result.get('q3_copies_removed')
            result = storage.report(profile['storage_limit'], refresh=True, **options)
            if q3_removed is not None:
                result['q3_copies_removed'] = q3_removed
        result['completion_rpcs_removed'] = removed
    elif payload.get('duplicate_inventory'):
        result['ended_rpc_inventory'] = storage.pbs_completion_copy_inventory(payload['pbs_namespace'])
if result is not None and payload.get('pbs_ack_inventory'):
    result['pbs_ack_inventory'] = storage.pending_pbs_completions(spools[0])
if result is not None and receipts:
    result['pbs_acknowledgements'] = acknowledgements
print(json.dumps(result))
"""


def read(path):
    return json.loads(path.read_bytes())


def _hashes(value):
    if isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value):
        yield value
    elif isinstance(value, (dict, list)):
        for item in value.values() if isinstance(value, dict) else value:
            yield from _hashes(item)


def central_artifact_caches(run):
    """Current CAS and ended CAS owners of the same evaluation only."""
    state_dir = Path(read(run/'CONTROLLER_CONFIG.json')['state_directory'])
    index = read(state_dir/'RUNS.json')
    evaluation = (index.get(run.name) or {}).get('evaluation_identity')
    candidates = [run]
    if evaluation:
        for item in index.values():
            if item.get('evaluation_identity') != evaluation:
                continue
            prior = Path(item['run_directory']).resolve()
            if prior != run and (prior/'LAUNCH.json').is_file() and (prior/'EXIT.json').is_file() and not invocation_active(prior):
                candidates.append(prior)
    return [(owner/'distributed/artifacts').resolve() for owner in dict.fromkeys(candidates)]


def local_reclaim_target(policy, latest, worker):
    """Measure the local filesystem alongside the worker quota."""
    usage = dict(latest, filesystem_free_bytes=shutil.disk_usage(worker).free,
                 filesystem_min_free_bytes=policy['limit_bytes'] // 10)
    return duplicate_reclaim_target(usage)


def retained_artifact_hashes(run, db):
    requests = db.execute("SELECT request FROM jobs WHERE state NOT IN ('ACCEPTED','CANCELLED')").fetchall()
    retained = {sha for (request,) in requests for sha in _hashes(json.loads(request))}
    pending = run/'distributed/PREPARATION_PENDING.json'
    if pending.is_file():
        record = read(pending)
        if not isinstance(record.get('tasks'), dict):
            retained.update(_hashes(record))
    tasks = read_preparation_readiness(run/'distributed', db)
    finished = {row[0] for row in db.execute(
        "SELECT job_id FROM jobs WHERE state IN ('ACCEPTED','CANCELLED')")} if tasks else set()
    for job_id, task in tasks.items():
        if job_id not in finished:
            retained.update(_hashes(task))
    return retained


def local_duplicate_cache(run, profile):
    """Drop only idle local cache copies with a byte-verified central copy."""
    policy = profile['storage_limit']
    sample = Path(policy['sample_path'])
    worker = Path(profile['artifact_cache_directory']).resolve()
    spool = Path(profile['work_dir']).resolve()
    if str(worker) not in policy['roots']:
        return {}
    state_dir = Path(read(run/'CONTROLLER_CONFIG.json')['state_directory'])
    canonical = [central for central in central_artifact_caches(run)
                 if not central.is_relative_to(worker) and not worker.is_relative_to(central)]

    def protected():
        if not registered_storage_owner_is_exclusive(state_dir, run, spool, worker, artifact_cache=True):
            return None
        with sqlite3.connect((run/'distributed/QUEUE.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
            if db.execute("""SELECT 1 FROM jobs WHERE host_id=? AND state IN
                            ('ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER')""",
                          (profile['host_id'],)).fetchone():
                return None
            retained = retained_artifact_hashes(run, db)
        for marker in spool.glob('*/*/WORKER_STATE.json'):
            status = read(marker)['state']
            if status == 'TRANSFERRED':
                continue
            if status != 'UNKNOWN':
                return None
            for name in ('ASSIGNMENT.json', 'OUTCOME.json', 'ARTIFACT_MANIFEST.json'):
                path = marker.with_name(name)
                if path.is_file():
                    retained.update(_hashes(read(path)))
        return retained

    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink

    with sample.with_suffix('.lock').open('r+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        measurement = read(sample)
        latest = measurement['report']
        if measurement['policy'] != dict(limit_bytes=policy['limit_bytes'], roots=policy['roots']):
            return {}
        target = local_reclaim_target(policy, latest, worker)
        if target <= 0:
            return {}
        retained = protected()
        if retained is None:
            return {}
        verified = []
        verified_bytes = 0
        # Large duplicates restore intake with fewer registry checks.
        for path in sorted(worker.glob('*/*'), key=lambda path: -path.stat().st_size):
            if (path.name in retained or path.parent.name != path.name[:2]
                    or not re.fullmatch('[a-f0-9]{64}', path.name) or path.is_symlink() or not path.is_file()):
                continue
            stat = path.stat()
            if stat.st_nlink != 1:
                continue
            relative = path.relative_to(worker)
            for cache in canonical:
                original = cache/relative
                if not original.is_file() or original.is_symlink():
                    continue
                central_stat = original.stat()
                if central_stat.st_size == stat.st_size and sha(original) == path.name:
                    break
            else:
                continue
            if sha(path) != path.name:
                continue
            size = stat.st_blocks * 512
            verified.append((path, identity(stat), original, identity(central_stat), size))
            verified_bytes += size
            if verified_bytes >= target:
                break
        if not verified:
            return {}
        try:
            with (state_dir/'REGISTRY.lock').open('rb') as registry:
                fcntl.flock(registry, fcntl.LOCK_SH | fcntl.LOCK_NB)
                current = protected()
                if current is None:
                    return {}
                removed = {}
                for path, local_id, original, central_id, size in verified:
                    if path.name in current or path.is_symlink() or original.is_symlink():
                        continue
                    try:
                        if identity(path.stat()) != local_id or identity(original.stat()) != central_id:
                            continue
                        path.unlink()
                    except FileNotFoundError:
                        continue
                    removed[path.name] = size
                return removed
        except BlockingIOError:
            return {}


def local_report(run, profile):
    policy = profile.get('storage_limit')
    options = dict(spools=(profile['work_dir'],), cache_directories=prefix_caches(profile), proactive=True)
    result = report(policy, **options)
    compacted = compact_local_metadata(run, profile, result)
    reclaimed = {key: value for key, value in (result or {}).items()
                 if key.startswith(('reclaimed_', 'evicted_'))}
    packed = bool(compacted.get('compacted_metadata_bytes'))
    if packed:
        result = report(policy, refresh=True, **options)
    removed = {}
    if (result and profile.get('artifact_cache_directory')
            and local_reclaim_target(policy, result, profile['artifact_cache_directory']) > 0):
        removed = local_duplicate_cache(run, profile)
        if removed:
            result = report(policy, refresh=True, **options)
            result.update(reclaimed_duplicate_artifacts=len(removed), reclaimed_duplicate_bytes=sum(removed.values()))
    if result and duplicate_reclaim_target(result) > 0:
        items = q3_result_copy_inventory(profile, options['spools'])
        proofs = verified_q3_copies(run, profile['host_id'], items, duplicate_reclaim_target(result))
        if proofs:
            q3 = reclaim_q3_result_copies(profile, options['spools'], proofs, _Q3_COPY_SWEEP_BYTES)
            if q3['files']:
                result = report(policy, refresh=True, **options)
            result['q3_copies_removed'] = q3
    if result is not None:
        if packed or removed:
            result['reclaim_checked'] = True
            for key, value in reclaimed.items():
                result[key] = result.get(key, 0) + value
        result.update(compacted)
    return result


def wsl_running(worker, ssh):
    name = worker['distribution'].replace("'", "''")
    executable = worker.get('wsl_executable', 'wsl.exe').replace("'", "''")
    command = (f"$running = & '{executable}' --list --running --quiet | "
               "ForEach-Object { ($_ -replace [string][char]0, '').Trim() }; "
               f"if ($running -contains '{name}') {{ 'running' }}")
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    raw = subprocess.check_output(
        [*worker.get('ssh_command', ssh), '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
         worker['ssh_alias'], 'powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded],
        stderr=subprocess.PIPE, timeout=15)
    return raw.replace(b'\x00', b'').strip() == b'running'


def verified_remote_duplicates(run, host, inventory, target):
    """Prove remote copies exist in the current or ended same-evaluation CAS."""
    with sqlite3.connect((run/'distributed/QUEUE.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        if db.execute("""SELECT 1 FROM jobs WHERE host_id=? AND state IN
                      ('STARTING','RUNNING','COMPLETED_PENDING_TRANSFER')""", (host,)).fetchone():
            return {}
        retained = retained_artifact_hashes(run, db)
    proof = {}
    size_verified = 0
    canonical = central_artifact_caches(run)
    for sha, size in sorted(inventory.items(), key=lambda pair: -pair[1]):
        if sha in retained or not re.fullmatch('[a-f0-9]{64}', sha):
            continue
        verified = False
        for central in canonical:
            original = central/sha[:2]/sha
            if original.is_symlink() or not original.is_file() or original.stat().st_size != size:
                continue
            with original.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() == sha:
                    verified = True
                    break
        if not verified:
            continue
        proof[sha] = size
        size_verified += size
        if size_verified >= target:
            break
    return proof


def accepted_pbs_receipts(run, host, attempts):
    """Return durable completion identities for this worker's saved attempts."""
    if not attempts:
        return []
    matches = []
    uri = (run/'distributed/QUEUE.sqlite3').as_uri()+'?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        for item in attempts:
            row = db.execute(
                """SELECT job_id, attempt_id, request_sha256, completion_sha256 FROM jobs
                   WHERE job_id=? AND host_id=? AND attempt_id=? AND state='ACCEPTED'
                   AND completion_sha256 IS NOT NULL""",
                (item['job_id'], host, item['attempt_id'])).fetchone()
            if row:
                matches.append(dict(row))
    return matches


def verified_q3_copies(run, host, inventory, target):
    """Bind each proposed copy to an adopted central result and hash its saved bytes."""
    target = min(target, _Q3_COPY_SWEEP_BYTES)
    if target <= 0 or not inventory:
        return []
    state_dir = Path(read(run/'CONTROLLER_CONFIG.json')['state_directory'])
    index = read(state_dir/'RUNS.json')
    directories = list(dict.fromkeys([run.resolve(), *(Path(item['run_directory']).resolve()
                    for item in index.values())]))
    pending = {}
    for item in inventory:
        key = item['job'], item['attempt'], item['request_sha256']
        if (re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', key[0])
                and re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', key[1])
                and item['name'] in ('RESULT.json', 'API_RETURN.json')):
            pending.setdefault(key[0], {}).setdefault(key[1:], []).append(item)
    verified = []
    saved_bytes = 0
    for directory in directories:
        dbpath = directory/'distributed/QUEUE.sqlite3'
        if not pending or not dbpath.is_file():
            continue
        with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro', uri=True)) as db:
            for offset in range(0, len(pending), 400):
                jobs = tuple(pending)[offset:offset+400]
                marks = ','.join('?' for _ in jobs)
                rows = db.execute(f"""SELECT job_id,attempt_id,request_sha256 FROM jobs
                    WHERE state='ACCEPTED' AND adopted_epoch IS NOT NULL AND completion_sha256 IS NOT NULL
                    AND host_id=? AND job_id IN ({marks})""", (host,*jobs)).fetchall()
                for job, attempt, req in rows:
                    candidates = pending[job].pop((attempt,req), ())
                    for item in candidates:
                        root = directory/'distributed/transfers'/job/attempt/job
                        path = root/item['name']
                        if path.resolve() != path or not path.is_file() or path.stat().st_size != item['size']:
                            continue
                        with path.open('rb') as stream:
                            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                        verified.append(dict(item, sha256=digest))
                        saved_bytes += item['blocks']
                        if saved_bytes >= target:
                            return verified
        pending = {job: grouped for job, grouped in pending.items() if grouped}
    return verified


def audit_q3(run, host, result):
    inventory = result['q3_inventory']
    verified = verified_q3_copies(run, host, inventory, _Q3_COPY_SWEEP_BYTES)
    return dict(candidate_files=len(inventory), candidate_bytes=sum(x['blocks'] for x in inventory),
                eligible_for_remote_hash_check=len(verified),
                eligible_bytes_this_sweep=sum(x['blocks'] for x in verified))


def verified_completion_rpcs(run, host, items):
    """Proof for ended sessions whose successful completion response is central."""
    if not items:
        return [], {}
    pbs = run/'distributed/pbs'
    allocations = read(pbs/'STATE.json')['allocations']
    receipts = []
    size = 0
    for token, rid, count in items:
        if (not re.fullmatch(r'[a-f0-9]{32}', token)
                or not re.fullmatch(r'(?:complete-[a-f0-9]{20}-)?[a-f0-9]{32}', rid)
                or (allocations.get(token,{}).get('observed') or {}).get('status') != 'F'):
            continue
        path = pbs/'responses'/token/(rid+'.json')
        if not path.is_file():
            continue
        record = read(path)
        response = record['response']
        result = response.get('result') or {}
        if response.get('ok') is not True or result.get('ok') is not True or type(result.get('duplicate')) is not bool:
            continue
        receipts.append(dict(token=token, request_id=rid, response=response, request_sha256=record['request_sha256']))
        size += count
        if size >= 128 * 1024**2:
            break
    if not receipts:
        return [], {}
    dbpath = run/'distributed/QUEUE.sqlite3'
    with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro', uri=True)) as db:
        owners = {'|'.join((job,attempt,req)): digest for job,attempt,req,digest in db.execute("""
            SELECT job_id,attempt_id,request_sha256,completion_sha256 FROM jobs
            WHERE host_id=? AND state='ACCEPTED' AND adopted_epoch IS NOT NULL
              AND attempt_id IS NOT NULL AND completion_sha256 IS NOT NULL
            """,(host,))}
    return receipts, owners


def cleanup(run, source, *, audit=False):
    distributed = run/'distributed'
    conf = distributed/'relay'/'CONFIG.json'
    relay = Relay(conf) if conf.exists() else None
    if relay is not None:
        relay.config['request_timeout_seconds'] = max(300, relay.config.get('request_timeout_seconds', 60))
    admission_path = run/'HOST_ADMISSION.json'
    admission = read(admission_path).get('hosts', {}) if admission_path.exists() else {}
    local = distributed/'LOCAL_WORKER.json'
    pbs = distributed/'pbs'/'SETTINGS.json'
    report_payload = dict(source=source, report_code=REMOTE_REPORT)
    work = []
    if local.exists():
        profile = read(local)
        def local_check():
            if audit:
                return audit_q3(run, 'local', {'q3_inventory': q3_result_copy_inventory(profile, [profile['work_dir']])})
            return local_report(run, profile)
        work.append(('local', local_check))
    if relay is not None:
        for worker in relay.config['workers']:
            host = worker['host_id']
            if not worker.get('enabled', True) or admission.get(host, {}).get('accepting') is False:
                continue
            def remote(w=worker):
                wsl = w.get('transport') == 'ssh-wsl'
                if wsl and not wsl_running(w, relay.ssh):
                    return dict(status='WSL_STOPPED')
                command = [w['python'], '-B', '-c', REMOTE_ENTRY, w['root'], w['profile']]
                def check(**options):
                    return relay.remote_command(w, command, 'storage-reclaim', dict(report_payload, **options))[0]
                if audit:
                    return audit_q3(run, w['host_id'], check(q3_audit=True))
                result = check(duplicate_inventory=True)
                inventory = result.pop('duplicate_inventory', {}) if result else {}
                copies = result.pop('q3_inventory', []) if result else []
                target = duplicate_reclaim_target(result) if result else 0
                proof = verified_remote_duplicates(run, w['host_id'], inventory, target) if inventory else {}
                q3_proof = verified_q3_copies(run, w['host_id'], copies, target)
                if proof or q3_proof:
                    result = check(verified_duplicates=proof, verified_q3_copies=q3_proof)
                return result
            work.append((host, remote))
    if pbs.exists():
        settings = read(pbs)
        namespace = Path(settings['remote_root'])
        runs = str(namespace.parent) if settings.get('namespace_by_run', True) else None
        for worker in settings['workers']:
            def gateway(w=worker):
                command = [settings['gateway_python'], '-B', '-c', REMOTE_ENTRY,
                           settings['transport_root'], '']
                payload = dict(report_payload, profile=w['profile'],
                               spools=[str(namespace/'workers'/w['host_id'])], pbs_runs=runs, pbs_namespace=str(namespace))
                def check(**options):
                    output = subprocess.check_output(
                        ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                         settings['ssh_alias'], shlex.join(command)],
                        input=json.dumps(dict(payload, **options)), text=True, stderr=subprocess.PIPE, timeout=300)
                    return json.loads(output)
                if audit:
                    return audit_q3(run, w['host_id'], check(q3_audit=True))
                result = check(duplicate_inventory=True, pbs_ack_inventory=True)
                copies = result.pop('q3_inventory', []) if result else []
                inventory = result.pop('duplicate_inventory', {}) if result else {}
                rpc_items = result.pop('ended_rpc_inventory', []) if result else []
                pending = result.pop('pbs_ack_inventory', []) if result else []
                receipts = accepted_pbs_receipts(run, w['host_id'], pending)
                proof = (verified_remote_duplicates(run, w['host_id'], inventory, duplicate_reclaim_target(result))
                         if inventory else {})
                q3_proof = verified_q3_copies(run, w['host_id'], copies, duplicate_reclaim_target(result)) if copies else []
                rpc_proof, rpc_owners = verified_completion_rpcs(run, w['host_id'], rpc_items)
                if receipts or proof or q3_proof or rpc_proof:
                    result = check(verified_duplicates=proof, pbs_ack_receipts=receipts, verified_q3_copies=q3_proof,
                                   verified_completion_rpcs=rpc_proof, adopted_completion_owners=rpc_owners)
                return result
            work.append((worker['host_id'], gateway))
    with ThreadPoolExecutor(max_workers=max(1, len(work))) as pool:
        for host, future in [(host, pool.submit(check)) for host, check in work]:
            try:
                result = future.result()
            except (OSError, KeyError, ValueError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as error:
                result = dict(error_type=type(error).__name__)
            print(json.dumps(dict(epoch=time.time(), host_id=host, storage=result)), flush=True)


def invocation_active(run):
    launch = run/'LAUNCH.json'
    if not launch.is_file():
        return False
    invocation = read(launch)['invocation_id']
    finished = run/'EXIT.json'
    return not finished.is_file() or read(finished).get('invocation_id') != invocation


def cleanup_stopped_local(state):
    """Use the last selected local profile during a naturally stopped period."""
    target = state/'CURRENT_TARGET.json'
    if not target.is_file():
        return
    run = Path(read(target)['run_directory']).resolve()
    profile = run/'distributed/LOCAL_WORKER.json'
    if not profile.is_file() or invocation_active(run):
        return
    with (profile.parent/'STORAGE_RECLAIM.lock').open('a+b') as lock:
        try:
            acquire_lock(lock)
        except BlockingIOError:
            return
        result = local_report(run, read(profile))
        if result and (result.get('reclaim_checked') or result.get('metadata_compaction')):
            print(json.dumps(dict(epoch=time.time(), host_id='local', storage=result)), flush=True)


def cleanup_active_runs(state, source):
    for run_id, entry in read(state/'RUNS.json').items():
        try:
            run = Path(entry['run_directory']).resolve()
            distributed = run/'distributed'
            if not distributed.is_dir() or not (run/'CONTROLLER_CONFIG.json').is_file() or not invocation_active(run):
                continue
            with (distributed/'STORAGE_RECLAIM.lock').open('a+b') as run_lock:
                try:
                    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                cleanup(run, source)
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
            print(json.dumps(dict(event='STORAGE_RUN_DEFERRED', run_id=run_id,
                                  error_type=type(error).__name__)), flush=True)


def watch_active_runs(state, source):
    """Follow normal Start and Resume, including after the active run changes."""
    with closing((state/'STORAGE_RECLAIM.lock').open('a+b')) as lock:
        acquire_lock(lock)
        while True:
            try:
                cleanup_active_runs(state, source)
                cleanup_stopped_local(state)
            except (OSError, KeyError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
                print(json.dumps(dict(event='STORAGE_WATCH_WAIT', error_type=type(error).__name__)), flush=True)
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('run', type=Path, nargs='?')
    target.add_argument('--state-dir', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--audit-q3', action='store_true')
    args = parser.parse_args()
    source = (Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_worker_storage.py').read_text()
    if args.audit_q3 and (args.state_dir or args.watch):
        parser.error('--audit-q3 requires one run')
    if args.state_dir:
        watch_active_runs(args.state_dir.resolve(strict=True), source)
        return
    run = args.run.resolve(strict=True)
    if not (run/'CONTROLLER_CONFIG.json').is_file():
        parser.error('Expected an existing central run directory')
    if not args.watch:
        cleanup(run, source, audit=args.audit_q3)
        return
    with closing((run/'distributed'/'STORAGE_RECLAIM.lock').open('a+b')) as lock:
        acquire_lock(lock)
        while invocation_active(run):
            cleanup(run, source)
            time.sleep(60)


if __name__ == '__main__':
    main()
