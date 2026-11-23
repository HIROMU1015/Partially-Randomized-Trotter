"""Stage lossless Git pack compaction without taking a running worker offline."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import time

from .rpe_q3_coordinator import atomic_json
from .rpe_q3_worker import resource_state
from .rpe_worker_storage import (
    SAMPLE_SECONDS,
    _compact_pack_directory,
    _exchange_pack_directories,
    _packed_objects,
    _used_bytes,
)

PACK = re.compile(r'pack-[0-9a-f]{40,64}\.(?:pack|idx|rev)')
TEMPORARY_PACK = re.compile(r'tmp_pack_[A-Za-z0-9]+')
LIVE_STATES = ('ASSIGNED', 'STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER')


def read(path):
    return json.loads(path.read_bytes())


def lock(stack, path):
    handle = stack.enter_context(path.open('rb'))
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def sync_directory(path):
    handle = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def snapshot(directory):
    """Fingerprint immutable pack files. Preserve unindexed temporary packs verbatim."""
    result = {}
    for path in directory.iterdir():
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222
                or not (PACK.fullmatch(path.name) or TEMPORARY_PACK.fullmatch(path.name))):
            raise ValueError('Unsupported file in metadata pack directory')
        result[path.name] = [info.st_ino, info.st_size, info.st_mtime_ns]
    return result


def _choose_candidate(pool, roots):
    candidates = []
    for repo in pool.glob('*/objects.git'):
        if repo.resolve() != repo or not any(repo.is_relative_to(root) for root in roots):
            continue
        directory = repo/'objects/pack'
        try:
            packs = list(directory.glob('*.pack'))
            if len(packs) < 8:
                continue
            amount = sum(file.stat().st_blocks*512 for file in packs)
            if amount >= 256*1024**2:
                candidates.append((amount, directory))
        except OSError:
            continue
    return [directory for _, directory in sorted(candidates, reverse=True)]


def stage_pack(directory, staging, min_free_bytes):
    """Pack a snapshot in scratch while allowing new imports into the live cache."""
    shadow = staging/'objects.git'
    target = shadow/'objects/pack'
    with tempfile.TemporaryDirectory(prefix='build-', dir=staging) as tmp:
        temporary = Path(tmp)
        repo = temporary/'objects.git'
        subprocess.run(['git', 'init', '--bare', '-q', str(repo)], check=True, capture_output=True, timeout=10)
        packs = repo/'objects/pack'
        with ExitStack() as held:
            lock(held, directory.parents[2]/'IMPORT.lock')
            baseline = snapshot(directory)
            maximum = sum((directory/name).stat().st_blocks*512 for name in baseline
                          if PACK.fullmatch(name))
            if shutil.disk_usage(staging).free < maximum + min_free_bytes:
                return None
            for name in baseline:
                if PACK.fullmatch(name):
                    os.link(directory/name, packs/name)
            alternate = directory.parent/'info/alternates'
            if alternate.is_file():
                entries = [str((directory.parent/line).resolve())
                           for line in alternate.read_text().splitlines() if line]
                (repo/'objects/info/alternates').write_text('\n'.join(entries)+'\n')
        saved = _compact_pack_directory(packs, temporary)
        if not saved:
            return None
        with ExitStack() as held:
            lock(held, directory.parents[2]/'IMPORT.lock')
            if snapshot(directory) != baseline:
                return None
            for name in baseline:
                if TEMPORARY_PACK.fullmatch(name):
                    os.link(directory/name, packs/name)
        for path in packs.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        sync_directory(packs)
        staged = snapshot(packs)
        # Write the recovery record before publishing the staged directory.
        atomic_json(staging/'PACK.json', dict(original=str(directory), baseline=baseline,
                                             staged=staged, saved_bytes=saved))
        repo.rename(shadow)
        sync_directory(staging)
    return target


def _resolved_refs(pack):
    repo = pack.parent.parent
    command = ['git', '-C', str(repo)]
    refs = subprocess.check_output([*command,'for-each-ref','--format=%(refname) %(objectname)'],
                                   stderr=subprocess.PIPE, timeout=8)
    objects = b''.join(line.rsplit(b' ', 1)[1]+b'\n' for line in refs.splitlines())
    resolved = subprocess.check_output([*command,'cat-file','--batch-check'], input=objects,
                                       stderr=subprocess.PIPE, timeout=8)
    if b' missing\n' in resolved or len(resolved.splitlines()) != len(objects.splitlines()):
        raise ValueError('Derived metadata reference no longer resolves')
    return refs, resolved


def _controller_storage_hold(run, host):
    """Confirm the controller has consumed the zero-slot offer."""
    profile = run/'distributed/LOCAL_WORKER.json'
    daemon = read(profile.with_name('LOCAL_DAEMON.json'))
    if daemon.get('profile') != str(profile) or type(daemon.get('pid')) is not int:
        return False
    arguments = (Path('/proc')/str(daemon['pid'])/'cmdline').read_bytes().split(b'\0')
    if not any(name.endswith(b'/run_rpe_q3_worker.py') for name in arguments) or os.fsencode(profile) not in arguments:
        return False
    with (run/'PROGRESS.jsonl').open('rb') as stream:
        offset = max(0, stream.seek(0, os.SEEK_END) - 1024**2)
        stream.seek(offset)
        if offset:
            stream.readline()
        lines = stream.read().split(b'\n')[:-1]
    for line in reversed(lines):
        if b'"QUEUE_WAIT_STATE"' in line:
            event = json.loads(line)
            if event.get('event') == 'QUEUE_WAIT_STATE':
                offer = event.get('hosts', {}).get(host, {})
                return (offer.get('fresh_free_slots') == 0
                        and (offer.get('storage') or {}).get('reason') == 'STORAGE_LIMIT_REACHED')
    return False


def _check_local_jobs(run, spool, host):
    database = run/'distributed/QUEUE.sqlite3'
    if not database.is_file():
        return False
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=1) as db:
        if db.execute(
                "SELECT 1 FROM jobs WHERE host_id=? AND state IN (?,?,?,?) LIMIT 1",
                (host, *LIVE_STATES)).fetchone():
            return False
    for assignment in spool.glob('*/*/ASSIGNMENT.json'):
        marker = assignment.with_name('WORKER_STATE.json')
        if not marker.is_file():
            return False
        state = read(marker)['state']
        if state == 'TRANSFERRED':
            continue
        if state != 'UNKNOWN' or resource_state(marker.parent, state) != 'EXITED':
            return False
    return True


def _registered_owner_is_exclusive(state, run, spool, directory):
    registered = {Path(entry['run_directory']).resolve() for entry in read(state/'RUNS.json').values()}
    if run not in registered:
        return False
    for other in registered - {run}:
        profile = other/'distributed/LOCAL_WORKER.json'
        launch = other/'LAUNCH.json'
        if not profile.is_file() or not launch.is_file():
            continue
        exit_path = other/'EXIT.json'
        if exit_path.is_file() and read(exit_path)['invocation_id'] == read(launch)['invocation_id']:
            continue
        options = read(profile)
        other_spool = Path(options['work_dir']).resolve()
        other_pool = Path(options.get('preparation_metadata_cache_directory')
                          or other_spool.parent/'preparation-metadata-cache').resolve()
        if spool == other_spool or directory.is_relative_to(other_pool):
            return False
    return True


def _switch(run, profile, scratch, directory, staged, record):
    policy = profile['storage_limit']
    sample = Path(policy['sample_path'])
    spool = Path(profile['work_dir']).resolve()
    state = Path(read(run/'CONTROLLER_CONFIG.json')['state_directory']).resolve()
    worker_lock = spool/'WORKER.lock'
    with ExitStack() as held:
        lock(held, sample.with_suffix('.lock'))
        lock(held, state/'REGISTRY.lock')
        if (read(run/'distributed/LOCAL_WORKER.json') != profile
                or not _registered_owner_is_exclusive(state, run, spool, directory)):
            return 'PACK_SWITCH_WAIT_LOCAL_OWNER', 0
        try:
            lock(held, worker_lock)
        except BlockingIOError:
            # A polling daemon retains WORKER.lock while full. Its actual
            # quota hold fences new intake while the sample lock is held.
            saved = read(sample)
            observed = saved['report']
            if (saved['policy'] != {key:policy[key] for key in ('limit_bytes','roots')}
                    or not 0 <= time.time()-saved['sampled_epoch'] < SAMPLE_SECONDS
                    or observed.get('reason') != 'STORAGE_LIMIT_REACHED'
                    or observed.get('accepting_new_jobs') is not False
                    or not _controller_storage_hold(run, profile['host_id'])
                    or _used_bytes(policy['roots']) < policy['limit_bytes']):
                return 'PACK_SWITCH_WAIT_LOCAL_IDLE_OR_STORAGE_HOLD', 0
        if not _check_local_jobs(run, spool, profile['host_id']):
            return 'PACK_SWITCH_WAIT_OWNED_ATTEMPTS', 0
        lock(held, directory.parents[2]/'IMPORT.lock')
        current = snapshot(directory)
        prepared = snapshot(staged)
        if current != record['baseline'] or prepared != record['staged']:
            return 'PACK_STAGE_STALE', 0
        packed = _packed_objects(directory)
        if packed != _packed_objects(staged):
            raise ValueError('Compacted pack object set differs')
        refs = _resolved_refs(directory)
        # No controller, polling daemon, or scientific supervisor is stopped.
        if not _exchange_pack_directories(directory, staged):
            return 'PACK_SWITCH_UNAVAILABLE', 0
        try:
            if _packed_objects(directory) != packed or _resolved_refs(directory) != refs:
                raise ValueError('Installed pack changed the Git objects or refs')
            sync_directory(directory.parent)
            sync_directory(staged.parent)
            atomic_json(scratch/'APPLIED.json', record)
        except BaseException:
            _exchange_pack_directories(directory, staged)
            raise
    # The staged path now contains the previous representation, whose complete
    # object set and temporary files were verified before relinquishing locks.
    return 'PACK_COMPACTED', record['saved_bytes']


def compact_local_metadata(run, profile, observation):
    """Prepare at the soft limit and install during a provable local intake hold."""
    if not observation or profile.get('host_id') != 'local':
        return {}
    run = Path(run).resolve()
    policy = profile['storage_limit']
    spool = Path(profile['work_dir']).resolve()
    roots = [Path(root).resolve() for root in policy['roots']]
    pool = Path(profile.get('preparation_metadata_cache_directory') or spool.parent/'preparation-metadata-cache').resolve()
    scratch = Path(policy['sample_path']).resolve().parent/'metadata-repack'
    stage = scratch/'pending'
    due = observation.get('reclaim_pending')
    if not due and not stage.is_dir():
        return {}
    if not any(pool.is_relative_to(root) for root in roots) or any(scratch.is_relative_to(root) for root in roots):
        return {}
    if not (spool/'WORKER.lock').is_file():
        return {}
    scratch.mkdir(exist_ok=True)
    lock_path = scratch/'REPACK.lock'
    lock_path.touch(exist_ok=True)
    try:
        with ExitStack() as held:
            lock(held, lock_path)
            if stage.is_symlink():
                raise ValueError('Unexpected metadata staging path')
            metadata = stage/'PACK.json'
            staged = stage/'objects.git/objects/pack'
            if metadata.is_file() and staged.is_dir():
                record = read(metadata)
                directory = Path(record['original'])
                if directory != directory.resolve() or not directory.is_relative_to(pool):
                    raise ValueError('Unexpected saved pack source')
                current = snapshot(directory)
                prepared = snapshot(staged)
                if (prepared == record['baseline']
                        and all(current.get(name) == identity for name, identity in record['staged'].items())):
                    # Resume cleanup only when the installed representation
                    # still holds all old objects, including after a new import.
                    if not _packed_objects(staged) <= _packed_objects(directory):
                        raise ValueError('Interrupted pack switch has different objects')
                    _resolved_refs(directory)
                    shutil.rmtree(stage)
                    return dict(metadata_compaction='PACK_COMPACTED', compacted_metadata_bytes=record['saved_bytes'])
                if prepared != record['staged']:
                    raise ValueError('Staged pack changed during interrupted maintenance')
                if current != record['baseline']:
                    shutil.rmtree(stage)
                    return dict(metadata_compaction='PACK_STAGE_STALE')
            else:
                if stage.is_dir():
                    shutil.rmtree(stage)  # Owned staging from an interrupted build.
                if not due:
                    return {}
                directories = _choose_candidate(pool, roots)
                if not directories:
                    return {}
                stage.mkdir()
                for directory in directories:
                    try:
                        candidate = stage_pack(directory, stage, policy['limit_bytes']//10)
                    except (OSError, ValueError, subprocess.SubprocessError):
                        continue
                    if candidate is not None:
                        break
                else:
                    shutil.rmtree(stage)
                    return dict(metadata_compaction='NO_VERIFIED_SAVINGS')
                record = read(metadata)
            status, saved = _switch(run, profile, scratch, directory, staged, record)
            if status in ('PACK_COMPACTED', 'PACK_STAGE_STALE'):
                shutil.rmtree(stage)
            result = dict(metadata_compaction=status)
            if saved:
                result['compacted_metadata_bytes'] = saved
            return result
    except (OSError, ValueError, KeyError, sqlite3.Error, subprocess.SubprocessError) as error:
        return dict(metadata_compaction='PACK_MAINTENANCE_DEFERRED', metadata_compaction_reason=str(error)[-500:])
