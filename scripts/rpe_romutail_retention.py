#!/usr/bin/env python3
"""Retire redundant central storage independently of scientific execution."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

from recover_stopped_rpe_run import live_driver, optional, stopped_lock
from rpe_romutail_cold_storage import MOUNT, copy_verified, mounted_separately, sha256
from trottertracks.rpe_q3_coordinator import atomic_json


def stopped_runs(state):
    current = optional(state/'CURRENT_TARGET.json').get('run_id')
    for row in optional(state/'RUNS.json').values():
        run = Path(row['run_directory']).resolve()
        if not run.is_relative_to(state.parent) or run.name == current:
            continue
        launch, ended = optional(run/'LAUNCH.json'), optional(run/'EXIT.json')
        if launch.get('invocation_id') and ended.get('invocation_id') == launch['invocation_id'] and not live_driver(run):
            yield run, launch


def identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Not a regular immutable file')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def deduplicate_old_artifacts(state, *, minimum_bytes=1024**2):
    """Only immutable content-addressed blobs, preserving every original path."""
    seen, checked = {}, set()
    links = reclaimed = 0
    for run, launch in stopped_runs(state):
        cache = run/'distributed/artifacts'
        if cache.is_symlink() or not cache.is_dir():
            continue
        for path in cache.glob('*/*'):
            if not re.fullmatch('[0-9a-f]{64}', path.name) or path.parent.name != path.name[:2]:
                continue
            try:
                right = identity(path)
                if right[2] < minimum_bytes:
                    continue
                key = (right[0], path.name, right[2])
                source = seen.setdefault(key, path)
                left = identity(source)
                if left[:2] == right[:2]:
                    continue
                if left not in checked:
                    if sha256(source) != path.name or identity(source) != left:
                        continue
                    checked.add(left)
                if sha256(path) != path.name or identity(path) != right:
                    continue
                with stopped_lock(state, run, launch):
                    if identity(source) != left or identity(path) != right:
                        continue
                    info = path.stat()
                    tmp = path.with_name('.share-'+uuid.uuid4().hex)
                    try:
                        os.link(source, tmp)
                        os.replace(tmp, path)
                    finally:
                        tmp.unlink(missing_ok=True)
                    links += 1
                    if info.st_nlink == 1:
                        reclaimed += info.st_blocks*512
            except (OSError, ValueError, KeyError):
                continue
    return dict(shared_files=links, reclaimed_bytes=reclaimed)


def compact_readonly_pbs_replies(state, *, minimum_bytes=1024**2, progress=None):
    """Discard old GET response copies only when every blob remains in the CAS.

    The old dispatcher can regenerate a deleted read-only reply. New releases
    also preserve and check its exact RPC request digest in a small replay file.
    """
    reclaimed = count = 0
    known = set()
    for run, launch in stopped_runs(state):
        replies = run/'distributed/pbs/responses'
        cache = run/'distributed/artifacts'
        for path in replies.glob('*/*.json'):
            try:
                before = identity(path)
                if before[2] < minimum_bytes:
                    continue
                # Avoid decoding large assignment or completion responses.
                with path.open('rb') as stream:
                    if not re.search(rb'"result"\s*:\s*\{\s*"blobs"\s*:', stream.read(4096)):
                        continue
                with path.open() as stream:
                    record = json.load(stream)
                response = record['response']
                if response.get('ok') is not True or set(response['result']) != {'blobs'}:
                    continue
                blobs = response['result']['blobs']
                if not blobs or not re.fullmatch('[0-9a-f]{64}', record['request_sha256']):
                    continue
                valid = True
                for sha, encoded in blobs.items():
                    if not re.fullmatch('[0-9a-f]{64}', sha):
                        valid = False; break
                    source = cache/sha[:2]/sha
                    info = identity(source)
                    if info not in known:
                        if sha256(source) != sha or identity(source) != info:
                            valid = False; break
                        known.add(info)
                    if hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest() != sha:
                        valid = False; break
                if not valid:
                    continue
                replay = replies.parent/'responses-replay'/path.relative_to(replies)
                with stopped_lock(state, run, launch):
                    if identity(path) != before:
                        continue
                    info = path.stat()
                    replay.parent.mkdir(parents=True, exist_ok=True)
                    compact = dict(request_sha256=record['request_sha256'], hashes=list(blobs))
                    if 'clock' in record:
                        compact['clock'] = record['clock']
                    if replay.exists() and optional(replay) != compact:
                        continue
                    atomic_json(replay, compact)
                    path.unlink()
                    count += 1
                    if info.st_nlink == 1:
                        reclaimed += info.st_blocks*512
                if progress:
                    progress(count, reclaimed)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return dict(compacted_responses=count, reclaimed_bytes=reclaimed)


def archive_old_logs(state, mount=MOUNT, *, minimum_bytes=64*1024**2):
    """Keep old diagnostic logs accessible without keeping them on the SSD."""
    mount = Path(mount).resolve()
    if not mounted_separately(mount, state):
        return dict(status='COLD_DISK_NOT_MOUNTED', archived_bytes=0)
    root = mount/'RomuTail-auto-coldstore'/'diagnostics'
    archived = []
    reclaimed = 0
    for run, launch in stopped_runs(state):
        for name in ('PROGRESS.jsonl', 'DRIVER.log'):
            path = run/name
            try:
                before = identity(path)
                if before[2] < minimum_bytes:
                    continue
                target = root/run.name/launch['invocation_id']/name
                digest = sha256(path)
                copy_verified(path, target, digest, before[2])
                with stopped_lock(state, run, launch):
                    if identity(path) != before:
                        continue
                    info = path.stat()
                    atomic_json(target.with_name(name+'.archive.json'), dict(source=str(path),
                        archive=str(target), sha256=digest, bytes=before[2], state='VERIFIED_COPY'))
                    tmp = path.with_name('.cold-'+uuid.uuid4().hex)
                    try:
                        tmp.symlink_to(target)
                        os.replace(tmp, path)
                    finally:
                        tmp.unlink(missing_ok=True)
                    archived.append(str(path))
                    if info.st_nlink == 1:
                        reclaimed += info.st_blocks*512
            except (OSError, ValueError, KeyError):
                continue
    return dict(status='CHECKED', archived_files=archived, archived_bytes=reclaimed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory', type=Path, required=True)
    parser.add_argument('--mount', type=Path, default=MOUNT)
    parser.add_argument('--logs-only', action='store_true')
    args = parser.parse_args()
    state = args.state_directory.resolve()
    result = archive_old_logs(state, args.mount) if args.logs_only else dict(
        duplicates=deduplicate_old_artifacts(state), replies=compact_readonly_pbs_replies(state),
        logs=archive_old_logs(state, args.mount))
    print(result)

if __name__ == '__main__':
    main()
