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
from trottertracks.rpe_local_pack_maintenance import compact_local_metadata
from trottertracks.rpe_worker_storage import duplicate_reclaim_target, prefix_caches, report


REMOTE_REPORT = """
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
from trottertracks import rpe_worker_storage as storage
payload = json.load(sys.stdin)
exec(compile(payload['source'], 'rpe_worker_storage.py', 'exec'), vars(storage))
profile = json.loads(Path(sys.argv[2]).read_bytes()) if sys.argv[2] else payload['profile']
spools = (profile['work_dir'],) if sys.argv[2] else payload['spools']
options = dict(spools=spools, pbs_runs=payload.get('pbs_runs'), cache_directories=storage.prefix_caches(profile), compact_metadata=True, proactive=True)
result = storage.report(profile.get('storage_limit'), **options)
if result and storage.duplicate_reclaim_target(result) > 0 and payload.get('duplicate_inventory'):
    result['duplicate_inventory'] = storage.duplicate_inventory(profile)
if result and storage.duplicate_reclaim_target(result) > 0 and payload.get('verified_duplicates'):
    removed = storage.reclaim_verified_duplicates(profile, payload['verified_duplicates'])
    if removed:
        result = storage.report(profile['storage_limit'], refresh=True, **options)
        result.update(reclaimed_duplicate_artifacts=len(removed), reclaimed_duplicate_bytes=sum(removed.values()))
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
        if Path(read(state_dir/'CURRENT_TARGET.json')['run_directory']).resolve() != run:
            return None
        for item in read(state_dir/'RUNS.json').values():
            other = Path(item['run_directory']).resolve()
            if other == run or not (other/'LAUNCH.json').is_file():
                continue
            if (not (other/'EXIT.json').is_file() or
                    read(other/'LAUNCH.json')['invocation_id'] != read(other/'EXIT.json')['invocation_id']):
                return None
        with sqlite3.connect((run/'distributed/QUEUE.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
            if db.execute("""SELECT 1 FROM jobs WHERE host_id=? AND state IN
                            ('ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER')""",
                          (profile['host_id'],)).fetchone():
                return None
            queued = [r[0] for r in db.execute("SELECT request FROM jobs WHERE state='QUEUED'")]
        retained = {sha for request in queued for sha in _hashes(json.loads(request))}
        pending = run/'distributed/PREPARATION_PENDING.json'
        if pending.is_file():
            retained.update(_hashes(read(pending)))
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
        removed = {}
        for path in sorted(worker.glob('*/*'), key=lambda path: path.stat().st_mtime_ns):
            if (path.name in retained or path.parent.name != path.name[:2]
                    or not re.fullmatch('[a-f0-9]{64}', path.name) or path.is_symlink() or not path.is_file()):
                continue
            stat = path.stat()
            if stat.st_nlink != 1:
                continue
            relative = path.relative_to(worker)
            originals = (cache/relative for cache in canonical)
            if not any(original.is_file() and not original.is_symlink()
                       and original.stat().st_size == stat.st_size and sha(original) == path.name
                       for original in originals) or sha(path) != path.name:
                continue
            with (state_dir/'REGISTRY.lock').open('rb') as registry:
                try:
                    fcntl.flock(registry, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    break
                current = protected()
                if current is None:
                    break
                if path.name in current:
                    continue
                path.unlink()
            removed[path.name] = stat.st_blocks * 512
            if sum(removed.values()) >= target:
                break
        return removed


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
        requests = db.execute("SELECT request FROM jobs WHERE state IN ('QUEUED','ASSIGNED')").fetchall()
    retained = {sha for (request,) in requests for sha in _hashes(json.loads(request))}
    pending = run/'distributed/PREPARATION_PENDING.json'
    if pending.exists():
        retained.update(_hashes(read(pending)))
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


def cleanup(run, source):
    distributed = run/'distributed'
    conf = distributed/'relay'/'CONFIG.json'
    relay = Relay(conf) if conf.exists() else None
    if relay is not None:
        relay.config['request_timeout_seconds'] = max(300, relay.config.get('request_timeout_seconds', 60))
    admission_path = run/'HOST_ADMISSION.json'
    admission = read(admission_path).get('hosts', {}) if admission_path.exists() else {}
    local = distributed/'LOCAL_WORKER.json'
    pbs = distributed/'pbs'/'SETTINGS.json'
    work = []
    if local.exists():
        profile = read(local)
        work.append(('local', lambda p=profile: local_report(run,p)))
    if relay is not None:
        for worker in relay.config['workers']:
            host = worker['host_id']
            if not worker.get('enabled', True) or admission.get(host, {}).get('accepting') is False:
                continue
            def remote(w=worker):
                wsl = w.get('transport') == 'ssh-wsl'
                if wsl and not wsl_running(w, relay.ssh):
                    return dict(status='WSL_STOPPED')
                command = [w['python'], '-B', '-c', REMOTE_REPORT, w['root'], w['profile']]
                result = relay.remote_command(w, command, 'storage-reclaim',
                                              dict(source=source, duplicate_inventory=True))[0]
                inventory = result.pop('duplicate_inventory', {}) if result else {}
                if inventory:
                    target = duplicate_reclaim_target(result)
                    proof = verified_remote_duplicates(run, w['host_id'], inventory, target)
                    if proof:
                        result = relay.remote_command(w, command, 'storage-reclaim-verified',
                                                      dict(source=source, verified_duplicates=proof))[0]
                return result
            work.append((host, remote))
    if pbs.exists():
        settings = read(pbs)
        namespace = Path(settings['remote_root'])
        runs = str(namespace.parent) if settings.get('namespace_by_run', True) else None
        for worker in settings['workers']:
            def gateway(w=worker):
                command = [settings['gateway_python'], '-B', '-c', REMOTE_REPORT,
                           settings['transport_root'], '']
                payload = dict(source=source, profile=w['profile'],
                               spools=[str(namespace/'workers'/w['host_id'])], pbs_runs=runs)
                output = subprocess.check_output(
                    ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                     settings['ssh_alias'], shlex.join(command)],
                    input=json.dumps(payload), text=True, stderr=subprocess.PIPE, timeout=300)
                return json.loads(output)
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


def watch_active_runs(state, source):
    """Follow normal Start and Resume, including after the active run changes."""
    with closing((state/'STORAGE_RECLAIM.lock').open('a+b')) as lock:
        acquire_lock(lock)
        while True:
            try:
                index = read(state/'RUNS.json')
                for entry in index.values():
                    run = Path(entry['run_directory']).resolve()
                    if (run/'CONTROLLER_CONFIG.json').is_file() and invocation_active(run):
                        with (run/'distributed'/'STORAGE_RECLAIM.lock').open('a+b') as run_lock:
                            try:
                                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                            cleanup(run, source)
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
    args = parser.parse_args()
    source = (Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_worker_storage.py').read_text()
    if args.state_dir:
        watch_active_runs(args.state_dir.resolve(strict=True), source)
        return
    run = args.run.resolve(strict=True)
    if not (run/'CONTROLLER_CONFIG.json').is_file():
        parser.error('Expected an existing central run directory')
    if not args.watch:
        cleanup(run, source)
        return
    with closing((run/'distributed'/'STORAGE_RECLAIM.lock').open('a+b')) as lock:
        acquire_lock(lock)
        while invocation_active(run):
            cleanup(run, source)
            time.sleep(60)


if __name__ == '__main__':
    main()
