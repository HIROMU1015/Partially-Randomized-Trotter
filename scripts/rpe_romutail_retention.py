#!/usr/bin/env python3
"""Retire redundant central storage independently of scientific execution."""
import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import time
import uuid

from recover_stopped_rpe_run import live_driver, optional, safe_name, stopped_lock
from rpe_romutail_cold_storage import MOUNT, copy_verified, exchange, mounted_separately, sha256
from trottertracks.rpe_q3_coordinator import atomic_json
from trottertracks.rpe_q3_store import digest as request_digest


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
                # A terminal session on cold storage is already a verified
                # archive, not a disposable local replay cache.
                if path.parent.is_symlink() or replies.is_symlink():
                    continue
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



def share_stopped_event_originals(state, run_id=None):
    """Share only byte-identical completed common/X/Y event files; keep all paths."""
    shared = reclaimed = groups = 0
    for run, launch in stopped_runs(state):
        if run_id is not None and run.name != run_id:
            continue
        for marker in (run/'completion/candidates').rglob('SHARED_EVENTS.json'):
            parent = marker.parent
            paths = [parent/'events_PRIVATE.json', parent/'axes_X/events_PRIVATE.json',
                     parent/'axes_Y/events_PRIVATE.json']
            try:
                if marker.is_symlink() or not marker.is_file() or not optional(marker).get('digest'):
                    continue
                if any(p.resolve() != p for p in paths):
                    continue
                before = [identity(p) for p in paths]
                if len({v[2] for v in before}) != 1 or len({v[:2] for v in before}) == 1:
                    continue
                if len({sha256(p) for p in paths}) != 1:
                    continue
                with stopped_lock(state, run, launch):
                    if [identity(p) for p in paths] != before:
                        continue
                    for p in paths[1:]:
                        if paths[0].samefile(p):
                            continue
                        info = p.stat()
                        tmp = p.with_name('.event-share-'+uuid.uuid4().hex)
                        try:
                            os.link(paths[0], tmp)
                            os.replace(tmp, p)
                        finally:
                            tmp.unlink(missing_ok=True)
                        shared += 1
                        if info.st_nlink == 1:
                            reclaimed += info.st_blocks*512
                    groups += 1
            except (OSError, ValueError, KeyError):
                continue
    return dict(groups=groups, shared_files=shared, reclaimed_bytes=reclaimed)



def share_stopped_accepted_requests(state, run_id=None):
    """Share byte-identical accepted REQUEST copies with the saved queue owner."""
    shared = reclaimed = 0
    for run, launch in stopped_runs(state):
        if run_id is not None and run.name != run_id:
            continue
        database = run/'distributed/QUEUE.sqlite3'
        if not database.is_file():
            continue
        with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as db:
            rows = db.execute("SELECT job_id,method,attempt_id,request_sha256 FROM jobs WHERE state='ACCEPTED'").fetchall()
            for job, method, attempt, digest in rows:
                try:
                    source = run/'jobs'/safe_name(method)/safe_name(job)/'REQUEST.json'
                    target = run/'distributed/transfers'/safe_name(job)/safe_name(attempt)/'REQUEST.json'
                    if source.resolve()!=source or target.resolve()!=target:
                        continue
                    left, right = identity(source), identity(target)
                    if left[:2]==right[:2] or left[2]!=right[2]:
                        continue
                    raw=source.read_bytes()
                    if request_digest(json.loads(raw)) != digest or target.read_bytes()!=raw:
                        continue
                    with stopped_lock(state, run, launch):
                        if (identity(source)!=left or identity(target)!=right
                                or optional(run/'EXIT.json').get('invocation_id')!=launch['invocation_id']):
                            continue
                        info = target.stat()
                        tmp=target.with_name('.request-share-'+uuid.uuid4().hex)
                        try:
                            os.link(source,tmp);os.replace(tmp,target)
                        finally:
                            tmp.unlink(missing_ok=True)
                        shared+=1
                        if info.st_nlink==1:
                            reclaimed+=info.st_blocks*512
                except (OSError, ValueError, KeyError, TypeError):
                    continue
    return dict(shared_files=shared,reclaimed_bytes=reclaimed)



def share_stopped_provider_git_packs(state, run_id=None):
    """Share immutable pack/index pairs, leaving mutable checkout files alone."""
    shared = reclaimed = 0
    for run, launch in stopped_runs(state):
        if run_id is not None and run.name != run_id:
            continue
        pf=run/'pf_auto'
        seen={}
        for candidates in [pf/'candidates', *pf.glob('price-recovery/recheck-*/candidates')]:
            for workspace in candidates.glob('*/pf_auto/pf_completion/provider_workspace'):
                directory=workspace/'.git/objects/pack'
                if directory.resolve()!=directory or not directory.is_dir():
                    continue
                for pack in directory.glob('pack-*.pack'):
                    if not re.fullmatch('pack-[0-9a-f]{40}\.pack',pack.name):
                        continue
                    pair=(pack, pack.with_suffix('.idx'))
                    try:
                        identities=tuple(identity(path) for path in pair)
                        key=(pack.name,tuple(i[2] for i in identities))
                        original=seen.setdefault(key,pair)
                        if original==pair:
                            continue
                        old=tuple(identity(path) for path in original)
                        if all(a[:2]==b[:2] for a,b in zip(old,identities)):
                            continue
                        if any(sha256(a)!=sha256(b) for a,b in zip(original,pair)):
                            continue
                        with stopped_lock(state,run,launch):
                            if (tuple(identity(path) for path in original)!=old
                                    or tuple(identity(path) for path in pair)!=identities):
                                continue
                            for a,b in zip(original,pair):
                                if a.samefile(b):
                                    continue
                                info=b.stat();tmp=b.with_name('.pack-share-'+uuid.uuid4().hex)
                                try:
                                    os.link(a,tmp);os.replace(tmp,b)
                                finally:
                                    tmp.unlink(missing_ok=True)
                                shared+=1
                                if info.st_nlink==1:
                                    reclaimed+=info.st_blocks*512
                    except (OSError, ValueError, KeyError):
                        continue
    return dict(shared_files=shared,reclaimed_bytes=reclaimed)


def rpc_session_inventory(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Not a local response directory')
    items = {}
    for path in directory.iterdir():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.json', path.name):
            raise ValueError('Unexpected response record: '+str(path))
        items[path.name] = identity(path)
    if len({name.casefold() for name in items}) != len(items):
        raise ValueError('Response filenames collide on cold storage')
    return items


def terminal_response_session(run, session):
    # Local saved allocation metadata only; never contact a PBS gateway.
    allocation = optional(run/'distributed/pbs/STATE.json').get('allocations', {}).get(session, {})
    return allocation.get('status') in ('F', 'X', 'FINISHED_SAVED_WORKER', 'SUBMIT_REJECTED')


def archive_terminal_rpc_sessions(state, mount=MOUNT, *, run_id=None, session_id=None, minimum_bytes=1):
    """Relocate finished-session JSON byte-for-byte, retaining the replay path."""
    mount = Path(mount).resolve()
    if not mounted_separately(mount, state):
        return dict(status='COLD_DISK_NOT_MOUNTED', sessions=[], reclaimed_bytes=0)
    results = []
    for run, launch in stopped_runs(state):
        if run_id is not None and run.name != run_id:
            continue
        replies = run/'distributed/pbs/responses'
        if not replies.is_dir() or replies.is_symlink():
            continue
        for source in sorted(replies.iterdir()):
            session = source.name
            if (session_id is not None and session != session_id or source.is_symlink()
                    or not re.fullmatch('[0-9a-f]{32}', session)
                    or not terminal_response_session(run, session)):
                continue
            items = rpc_session_inventory(source)
            total = sum(info[2] for info in items.values())
            if total < minimum_bytes:
                continue
            target = mount/'RomuTail-auto-coldstore'/'pbs-responses'/run.name/launch['invocation_id']/session
            target.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or not target.resolve().is_relative_to(mount):
                raise ValueError('Invalid response archive directory')
            if set(p.name for p in target.iterdir()) - items.keys():
                raise ValueError('Unexpected response archive entries')
            entries = {}
            for name, before in items.items():
                original, saved = source/name, target/name
                digest = sha256(original)
                if saved.exists():
                    if saved.is_symlink() or saved.stat().st_size != before[2] or sha256(saved) != digest:
                        raise ValueError('Existing response archive differs: '+name)
                else:
                    tmp = saved.with_name('.copy-'+uuid.uuid4().hex)
                    try:
                        with original.open('rb') as reader, tmp.open('xb') as writer:
                            shutil.copyfileobj(reader, writer)
                        if tmp.stat().st_size != before[2] or sha256(tmp) != digest:
                            raise ValueError('Archived response differs: '+name)
                        os.replace(tmp, saved)
                    finally:
                        tmp.unlink(missing_ok=True)
                if identity(original) != before:
                    raise ValueError('Response changed during archive: '+name)
                entries[name] = dict(sha256=digest, bytes=before[2])
            record = dict(run_id=run.name, invocation=launch['invocation_id'], session=session,
                          source=str(source), archive=str(target), entries=entries, status='VERIFIED_COPY')
            manifest = target.with_name(session+'.archive.json')
            atomic_json(manifest, record)
            fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
            try:
                if ctypes.CDLL(None, use_errno=True).syncfs(fd) != 0:
                    raise OSError(ctypes.get_errno(), 'Cold response storage sync failed')
            finally:
                os.close(fd)
            parked = source.with_name('.responses-cold-'+session)
            with stopped_lock(state, run, launch):
                if (not terminal_response_session(run, session) or rpc_session_inventory(source) != items
                        or optional(run/'EXIT.json').get('invocation_id') != launch['invocation_id']):
                    raise ValueError('Stopped response session changed')
                if parked.exists() or parked.is_symlink():
                    raise ValueError('Previous response archive exchange requires reconciliation')
                reclaimed = sum((source/name).stat().st_blocks*512 for name in items
                                if (source/name).stat().st_nlink == 1)
                parked.symlink_to(target, target_is_directory=True)
                exchange(source, parked)
                try:
                    if source.resolve() != target or rpc_session_inventory(parked) != items:
                        raise ValueError('Response archive exchange differs')
                except BaseException:
                    exchange(source, parked); parked.unlink(); raise
            # The original JSON paths now refer to byte-verified archival copies.
            shutil.rmtree(parked)
            record.update(status='MOVED_VERIFIED', reclaimed_bytes=reclaimed)
            atomic_json(manifest, record)
            results.append(dict(session=session, files=len(items), reclaimed_bytes=reclaimed, archive=str(target)))
            print(json.dumps(results[-1]), flush=True)
    return dict(status='CHECKED', sessions=results, reclaimed_bytes=sum(r['reclaimed_bytes'] for r in results))



def archive_terminal_rpc_large_files(state, mount=MOUNT, *, run_id=None, session_id=None,
                                     minimum_bytes=256*1024):
    """Preserve large old reply records at their exact paths, avoiding tiny-file churn."""
    mount=Path(mount).resolve()
    if not mounted_separately(mount,state):
        return dict(status='COLD_DISK_NOT_MOUNTED',sessions=[],reclaimed_bytes=0)
    results=[]
    for run,launch in stopped_runs(state):
        if run_id is not None and run.name!=run_id:
            continue
        replies=run/'distributed/pbs/responses'
        if not replies.is_dir() or replies.is_symlink():
            continue
        for directory in sorted(replies.iterdir()):
            session=directory.name
            if (session_id is not None and session!=session_id or directory.is_symlink()
                    or not re.fullmatch('[0-9a-f]{32}',session)
                    or not terminal_response_session(run,session)):
                continue
            target=mount/'RomuTail-auto-coldstore'/'pbs-responses'/run.name/launch['invocation_id']/session
            target.mkdir(parents=True,exist_ok=True)
            if target.is_symlink() or not target.resolve().is_relative_to(mount):
                raise ValueError('Invalid response archive directory')
            manifest=target.with_name(session+'.large-records.json')
            previous=optional(manifest)
            entries=dict(previous.get('entries',{}))
            pending=[]
            for path in directory.iterdir():
                if path.is_symlink() or not re.fullmatch(r'[A-Za-z0-9_.-]+\.json',path.name):
                    continue
                before=identity(path)
                if before[2]<minimum_bytes:
                    continue
                archived=target/path.name
                digest=sha256(path)
                copy_verified(path,archived,digest,before[2])
                if identity(path)!=before:
                    raise ValueError('Response changed during archive: '+str(path))
                entries[path.name]=dict(sha256=digest,bytes=before[2])
                pending.append((path,archived,before))
            if not pending:
                continue
            record=dict(run_id=run.name,invocation=launch['invocation_id'],session=session,
                        source=str(directory),archive=str(target),entries=entries,status='VERIFIED_COPY')
            atomic_json(manifest,record)
            fd=os.open(target,os.O_RDONLY|os.O_DIRECTORY)
            try:
                if ctypes.CDLL(None,use_errno=True).syncfs(fd)!=0:
                    raise OSError(ctypes.get_errno(),'Cold response storage sync failed')
            finally:
                os.close(fd)
            reclaimed=linked=0
            for start in range(0,len(pending),128):
                with stopped_lock(state,run,launch):
                    if (not terminal_response_session(run,session)
                            or optional(run/'EXIT.json').get('invocation_id')!=launch['invocation_id']):
                        raise ValueError('Stopped response session changed')
                    for path,archived,before in pending[start:start+128]:
                        if identity(path)!=before:
                            continue
                        info=path.stat();tmp=path.with_name('.response-cold-'+uuid.uuid4().hex)
                        try:
                            tmp.symlink_to(archived)
                            os.replace(tmp,path)
                        finally:
                            tmp.unlink(missing_ok=True)
                        linked+=1
                        if info.st_nlink==1:
                            reclaimed+=info.st_blocks*512
            record.update(status='MOVED_VERIFIED' if linked==len(pending) else 'PARTIALLY_MOVED',
                          reclaimed_bytes=int(previous.get('reclaimed_bytes',0))+reclaimed)
            atomic_json(manifest,record)
            results.append(dict(session=session,files=linked,reclaimed_bytes=reclaimed,archive=str(target)))
            print(json.dumps(results[-1]),flush=True)
    return dict(status='CHECKED',sessions=results,reclaimed_bytes=sum(r['reclaimed_bytes'] for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory', type=Path, required=True)
    parser.add_argument('--mount', type=Path, default=MOUNT)
    parser.add_argument('--logs-only', action='store_true')
    parser.add_argument('--share-events', action='store_true')
    parser.add_argument('--share-requests', action='store_true')
    parser.add_argument('--share-provider-packs', action='store_true')
    parser.add_argument('--archive-terminal-responses', action='store_true')
    parser.add_argument('--large-files-only', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--session-id')
    args = parser.parse_args()
    state = args.state_directory.resolve()
    if args.share_provider_packs:
        print(share_stopped_provider_git_packs(state, args.run_id)); return
    if args.share_requests:
        print(share_stopped_accepted_requests(state, args.run_id)); return
    if args.share_events:
        print(share_stopped_event_originals(state, args.run_id)); return
    if args.archive_terminal_responses:
        archive = archive_terminal_rpc_large_files if args.large_files_only else archive_terminal_rpc_sessions
        print(archive(state, args.mount, run_id=args.run_id, session_id=args.session_id)); return
    result = archive_old_logs(state, args.mount) if args.logs_only else dict(
        duplicates=deduplicate_old_artifacts(state), replies=compact_readonly_pbs_replies(state),
        logs=archive_old_logs(state, args.mount))
    print(result)

if __name__ == '__main__':
    main()
