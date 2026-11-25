#!/usr/bin/env python3
"""Move old immutable delivery caches to the attached cold storage disk."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tempfile
import time

from compact_rpe_git_objects import GIB, free_bytes
from recover_stopped_rpe_run import live_driver, optional, stopped_lock
from trottertracks.rpe_q3_coordinator import atomic_json

MOUNT = Path('/media/abe/T7')
ARCHIVE_FOLDER = 'RomuTail-auto-coldstore'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mounted_separately(mount, local):
    return mount.is_mount() and mount.stat().st_dev != local.stat().st_dev


def exchange(left, right):
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(left))


def eligible(state, run, min_age):
    current = optional(state/'CURRENT_TARGET.json').get('run_id')
    launch, end = optional(run/'LAUNCH.json'), optional(run/'EXIT.json')
    if (run.name == current or not launch.get('invocation_id') or
            end.get('invocation_id') != launch['invocation_id'] or
            time.time()-(run/'EXIT.json').stat().st_mtime < min_age or live_driver(run)):
        return None
    database = run/'distributed/QUEUE.sqlite3'
    if not database.is_file():
        return None
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True, timeout=2) as db:
        if db.execute("SELECT 1 FROM jobs WHERE state NOT IN ('ACCEPTED','CANCELLED') LIMIT 1").fetchone():
            return None
    return launch


def inventory(cache):
    items = {}
    for shard in cache.iterdir():
        if shard.is_symlink() or not shard.is_dir() or not re.fullmatch('[0-9a-f]{2}', shard.name):
            raise ValueError('Unexpected delivery cache directory: '+str(shard))
        for path in shard.iterdir():
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or not re.fullmatch('[0-9a-f]{64}', path.name)
                    or path.name[:2] != shard.name):
                raise ValueError('Unexpected delivery cache entry: '+str(path))
            items[path.relative_to(cache).as_posix()] = (info.st_dev, info.st_ino, info.st_size,
                                                        info.st_mtime_ns, info.st_blocks*512, info.st_nlink)
    return items


def same_inventory(before, after):
    return {k:v[:4] for k,v in before.items()} == {k:v[:4] for k,v in after.items()}


def copy_verified(source, target, digest, size):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.stat().st_size != size or sha256(target) != digest:
            raise ValueError('Existing archive artifact differs: '+str(target))
        if sha256(source) != digest:
            raise ValueError('Source artifact differs: '+str(source))
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.copy-', delete=False) as out:
        temporary = Path(out.name)
        try:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, out, length=4*1024**2)
            out.flush()
            os.fsync(out.fileno())
            if temporary.stat().st_size != size or sha256(temporary) != digest:
                raise ValueError('Delivery artifact SHA does not match: '+str(source))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def archive_one(state, run, archive_root, *, min_age=86400, minimum_bytes=GIB):
    source = run/'distributed/artifacts'
    if source.is_symlink() or not source.is_dir():
        return None
    launch = eligible(state, run, min_age)
    if launch is None:
        return None
    items = inventory(source)
    exclusive = sum(v[4] for v in items.values() if v[5] == 1)
    if exclusive < minimum_bytes:
        return None
    total = sum(v[2] for v in items.values())
    if free_bytes(archive_root) < total+GIB:
        return None
    destination = archive_root/run.name/launch['invocation_id']/'artifacts'
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, info in items.items():
        digest = Path(rel).name
        copy_verified(source/rel, destination/rel, digest, info[2])
        entries.append(dict(path=rel, sha256=digest, bytes=info[2]))
    # Retain the former local data until the complete archive is durable.
    fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if ctypes.CDLL(None, use_errno=True).syncfs(fd) != 0:
            raise OSError(ctypes.get_errno(), 'Cold storage sync failed')
    finally:
        os.close(fd)
    marker = destination.parent/'MANIFEST.json'
    record = dict(schema='RPE_COLD_ARTIFACTS_V1', state='VERIFIED_COPY', run_id=run.name,
                  original_invocation=launch['invocation_id'], source=str(source), archive=str(destination),
                  verified_bytes=total, estimated_exclusive_bytes=exclusive, entries=entries)
    atomic_json(marker, record)
    local = state/'maintenance/cold-artifacts'/f'{run.name}.json'
    local.parent.mkdir(parents=True, exist_ok=True)
    parked = source.with_name('.artifacts-cold-'+launch['invocation_id'])
    record['parked'] = str(parked)
    atomic_json(local, record)
    with stopped_lock(state, run, launch):
        if eligible(state, run, min_age) != launch or not same_inventory(items, inventory(source)):
            return None
        if parked.exists() or parked.is_symlink():
            raise ValueError('Previous cold-artifact exchange requires reconciliation')
        parked.symlink_to(destination, target_is_directory=True)
        exchange(source, parked)
        try:
            if not source.is_symlink() or source.resolve() != destination or not same_inventory(items, inventory(parked)):
                raise ValueError('Cold cache path exchange differs')
        except BaseException:
            exchange(source, parked)
            parked.unlink()
            raise
        record['state'] = 'MOVED_VERIFIED'
        atomic_json(local, record)
    # The source path now resolves to the verified T7 copy. The parked tree is
    # detached from the run and contains only immutable, previously verified files.
    shutil.rmtree(parked)
    atomic_json(marker, record)
    return {k:v for k,v in record.items() if k != 'entries'}


def reconcile_parked(state, mount):
    """Finish an interrupted exchange only after the original data is recoverable."""
    for path in (state/'maintenance/cold-artifacts').glob('*.json'):
        record = optional(path)
        if not record.get('parked'):
            continue
        source = Path(record['source'])
        parked = Path(record['parked'])
        target = Path(record['archive'])
        if (not target.is_relative_to(mount) or parked.parent != source.parent
                or not source.is_relative_to(state.parent) or not parked.name.startswith('.artifacts-cold-')):
            continue
        if parked.is_symlink() and parked.resolve() == target and not source.is_symlink():
            parked.unlink()  # Interrupted before exchange; local cache is intact.
        elif parked.is_dir() and not parked.is_symlink() and source.is_symlink() and source.resolve() == target:
            expected = {entry['path']:entry['sha256'] for entry in record['entries']}
            if set(inventory(parked)) == set(expected) == set(inventory(target)) and all(
                    sha256(parked/rel) == value == sha256(target/rel) for rel,value in expected.items()):
                shutil.rmtree(parked)
                record['state'] = 'MOVED_VERIFIED'
                atomic_json(path, record)


def archive_old(state, mount=MOUNT, *, min_age=86400, pressure_bytes=120*GIB):
    state = Path(state).resolve()
    mount = Path(mount).resolve()
    if free_bytes(state) >= pressure_bytes:
        return {'status':'ENOUGH_DISK_SPACE'}
    if not mounted_separately(mount, state):
        return {'status':'COLD_DISK_NOT_MOUNTED'}
    archive_root = mount/ARCHIVE_FOLDER
    archive_root.mkdir(exist_ok=True)
    with (state/'maintenance/COLD_ARTIFACTS.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status':'ALREADY_ARCHIVING'}
        reconcile_parked(state, mount)
        errors = []
        checked = 0
        for item in optional(state/'RUNS.json').values():
            run = Path(item['run_directory']).resolve()
            # Only local registered run stores, never external or live trees.
            if not run.is_relative_to(state.parent):
                continue
            checked += 1
            try:
                result = archive_one(state, run, archive_root, min_age=min_age)
            except (OSError, ValueError, KeyError, sqlite3.Error) as error:
                errors.append(dict(run_id=run.name, reason=str(error)[:160]))
                continue
            if result:
                return dict(status='MOVED_VERIFIED', **result)
    return {'status':'NO_ELIGIBLE_OLD_CACHE', 'runs_checked':checked, 'errors':errors[:10], 'error_count':len(errors)}


def restore_for_resume(run, mount=MOUNT):
    """Called by the normal gateway while it owns the run registry lock."""
    run = Path(run).resolve()
    source = run/'distributed/artifacts'
    if not source.is_symlink():
        return {'status':'LOCAL_CACHE'}
    target = source.resolve()
    mount = Path(mount).resolve()
    if not target.is_relative_to(mount):
        return {'status':'OTHER_STORAGE_LAYOUT'}
    if not mounted_separately(mount, run):
        raise ValueError('T7 is unavailable for the archived input cache')
    record = optional(target.parent/'MANIFEST.json')
    entries = record.get('entries', record.get('files_manifest'))
    # Earlier verified cold stores used a keyed files manifest; retain their
    # ability to resume through the same ordinary entry point.
    if entries is None and isinstance(record.get('files'), dict) and record.get('verified') is True:
        entries = [dict(path=path, sha256=entry['sha256'], bytes=entry['bytes'])
                   for path, entry in record['files'].items()]
    verified = (record.get('state') in ('MOVED_VERIFIED','VERIFIED_COPY') or record.get('verified') is True)
    if (record.get('run_id') not in (None, run.name) or Path(record.get('source','')).resolve() != target
            or Path(record.get('archive',record.get('target',''))).resolve() != target
            or not verified or not isinstance(entries, list)):
        raise ValueError('Cold cache manifest does not match the registered run')
    required = sum(entry['bytes'] for entry in entries)
    if free_bytes(run) < required + 6*GIB:
        raise ValueError('Insufficient local capacity to restore the archived preparation cache')
    with tempfile.TemporaryDirectory(prefix='.artifacts-restore-', dir=source.parent) as scratch:
        restored = Path(scratch)/'artifacts'
        restored.mkdir()
        for entry in entries:
            rel = Path(entry['path'])
            if (len(rel.parts) != 2 or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])
                    or rel.as_posix() != entry['sha256'][:2]+'/'+entry['sha256']):
                raise ValueError('Invalid cold archive entry')
            copy_verified(target/rel, restored/rel, entry['sha256'], entry['bytes'])
        if set(inventory(target)) != set(inventory(restored)):
            raise ValueError('Cold archive changed since verification')
        exchange(source, restored)
        if not restored.is_symlink() or source.is_symlink():
            exchange(source, restored)
            raise ValueError('Cold cache restore exchange differs')
        restored.unlink()
    return dict(status='RESTORED_FROM_COLD_STORAGE', bytes=required, files=len(entries))


def preflight_with_cold_storage(preflight, request_error):
    """Restore a cold cache before any ordinary bundle recovery or new execution."""
    def wrapped(run, problem, spec):
        try:
            recovered = restore_for_resume(run)
        except (OSError, ValueError, KeyError) as error:
            raise request_error('cold_storage', str(error)) from error
        report = preflight(run, problem, spec)
        if recovered['status'] == 'RESTORED_FROM_COLD_STORAGE':
            report['cold_storage_recovery'] = recovered
        return report
    if getattr(preflight, 'restores_preparation_archives', False):
        wrapped.restores_preparation_archives = True
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory', type=Path, required=True)
    args = parser.parse_args()
    report = archive_old(args.state_directory)
    atomic_json(args.state_directory/'maintenance/COLD_ARTIFACTS.json', dict(report, epoch=time.time()))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
