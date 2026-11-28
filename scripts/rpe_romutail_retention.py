#!/usr/bin/env python3
"""Retire redundant central storage independently of scientific execution."""
import argparse
import base64
import ctypes
from contextlib import closing
from itertools import zip_longest
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import time
import uuid

from recover_stopped_rpe_run import live_driver, optional, safe_name, stopped_lock, protected_run_directories
from rpe_romutail_cold_storage import MOUNT, copy_verified, exchange, mounted_separately, sha256
from trottertracks.rpe_q3_coordinator import atomic_json
from trottertracks.rpe_q3_store import digest as request_digest


def stopped_runs(state):
    protected = protected_run_directories(state)
    for row in optional(state/'RUNS.json').values():
        run = Path(row['run_directory']).resolve()
        if not run.is_relative_to(state.parent) or run in protected:
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
                with stopped_lock(state, run, launch, protect_references=True):
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
                with stopped_lock(state, run, launch, protect_references=True):
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
        # The root logs are reused by normal Resume. Only ended invocation
        # snapshots have an immutable lifecycle; never link an append target.
        invocation = run/'invocations'/launch['invocation_id']
        for name in ('PROGRESS.jsonl', 'DRIVER.log'):
            path = invocation/name
            try:
                before = identity(path)
                if before[2] < minimum_bytes:
                    continue
                target = root/run.name/launch['invocation_id']/name
                digest = sha256(path)
                copy_verified(path, target, digest, before[2])
                with stopped_lock(state, run, launch, protect_references=True):
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
                with stopped_lock(state, run, launch, protect_references=True):
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
                    with stopped_lock(state, run, launch, protect_references=True):
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
                        with stopped_lock(state,run,launch,protect_references=True):
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


def compact_stopped_provider_sources(state,run_id):
    """Compact only clean provider files whose pinned Resume can restore them.

    Reuse the ordinary provider compactor; never change a fixed code root or
    Git objects. Old releases without the restore path remain untouched.
    """
    from trottertracks.rpe_worker_storage import compact_provider_source
    import ast
    result=dict(compacted=0,reclaimed_bytes=0,skipped=[])
    for run,launch in stopped_runs(state):
        if run.name!=run_id:continue
        config=optional(run/'CONTROLLER_CONFIG.json')
        try:
            code=Path(config['code_root']).resolve(strict=True)/'src/trottertracks/pf_distributed_preparation.py'
            tree=ast.parse(code.read_text())
            if not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                       and node.func.id=='restore_provider_source' for node in ast.walk(tree)):
                raise ValueError('Pinned Resume does not restore retained provider source')
        except (OSError,KeyError,ValueError,SyntaxError) as error:
            result['skipped'].append(dict(run=run.name,reason=str(error)));continue
        pf=run/'pf_auto'
        for candidates in [pf/'candidates',*pf.glob('price-recovery/recheck-*/candidates')]:
            for workspace in candidates.glob('*/pf_auto/pf_completion/provider_workspace'):
                try:
                    if workspace.resolve()!=workspace:continue
                    marker=optional(workspace.parent/'PROVIDER_SOURCE.json')
                    source=marker['source']
                    if not re.fullmatch('[0-9a-f]{40,64}',source):raise ValueError('Unfixed provider source')
                    with stopped_lock(state,run,launch,protect_references=True):
                        size=compact_provider_source(workspace,source)
                    if size:result['compacted']+=1;result['reclaimed_bytes']+=size
                except (OSError,KeyError,ValueError,subprocess.SubprocessError) as error:
                    result['skipped'].append(dict(workspace=str(workspace),reason=str(error)))
    return result


def restore_stopped_provider_sources(state, run_id):
    """Restore retained bytes before handing a stopped run to a legacy release."""
    from trottertracks.rpe_worker_storage import restore_provider_source
    run, launch = next(((r,l) for r,l in stopped_runs(state) if r.name == run_id), (None,None))
    if run is None:
        raise ValueError('No stopped registered run')
    restored = 0
    with stopped_lock(state, run, launch, protect_references=True):
        pf = run/'pf_auto'
        for candidates in [pf/'candidates', *pf.glob('price-recovery/recheck-*/candidates')]:
            for workspace in candidates.glob('*/pf_auto/pf_completion/provider_workspace'):
                if workspace.resolve() != workspace:
                    raise ValueError('Provider workspace is not local')
                if (workspace.parent/'RETAINED_PROVIDER_SOURCE.json').is_file():
                    restore_provider_source(workspace)
                    restored += 1
    return dict(restored=restored)


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
            with stopped_lock(state, run, launch, protect_references=True):
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
                with stopped_lock(state,run,launch,protect_references=True):
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



def _queue_db_signature(path):
    return tuple(identity(item) if item.is_file() and not item.is_symlink() else None
                 for item in (path,Path(str(path)+'-wal'),Path(str(path)+'-shm')))


def _offline_queue(path):
    # No WAL/SHM is allowed; immutable reads do not create either sidecar.
    return closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True))


def _queue_is_settled(path):
    with _offline_queue(path) as db:
        return db.execute("SELECT 1 FROM jobs WHERE state NOT IN ('ACCEPTED','CANCELLED') LIMIT 1").fetchone() is None


def _same_queue_contents(original, compacted):
    """Compare every schema entry and ordered row, including original rowids."""
    with _offline_queue(original) as a, _offline_queue(compacted) as b:
        if b.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:return False
        schema="SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        left=a.execute(schema).fetchall();right=b.execute(schema).fetchall()
        if left!=right:return False
        for pragma in ('user_version','application_id'):
            if a.execute('PRAGMA '+pragma).fetchone()!=b.execute('PRAGMA '+pragma).fetchone():return False
        for kind,name,_,definition in left:
            if kind!='table':continue
            quoted='"'+name.replace('"','""')+'"'
            query='SELECT rowid,* FROM '+quoted+' ORDER BY rowid'
            if definition and 'WITHOUT ROWID' in definition.upper():
                columns=a.execute('PRAGMA table_info('+quoted+')').fetchall()
                keys=[row[1] for row in sorted((r for r in columns if r[5]),key=lambda r:r[5])]
                query='SELECT * FROM '+quoted+' ORDER BY '+','.join('"'+k.replace('"','""')+'"' for k in keys)
            if any(x!=y for x,y in zip_longest(a.execute(query),b.execute(query))):return False
        return True


def compact_stopped_queues(state, run_id=None, *, minimum_bytes=64*1024**2):
    """Reclaim freelist pages only in settled ended ledgers, retaining every row."""
    results=[]
    for run,launch in stopped_runs(state):
        if run_id is not None and run.name!=run_id:continue
        dbpath=run/'distributed/QUEUE.sqlite3'
        if not dbpath.is_file() or dbpath.is_symlink():continue
        wal=Path(str(dbpath)+'-wal');shm=Path(str(dbpath)+'-shm')
        # A WAL-backed active or unresolved ledger is never replaced.
        if (wal.exists() or wal.is_symlink() or shm.exists() or shm.is_symlink()
                or dbpath.stat().st_nlink != 1 or not _queue_is_settled(dbpath)):
            results.append(dict(run_id=run.name,status='OWNED_OR_WAL_HELD'));continue
        before=_queue_db_signature(dbpath)
        with _offline_queue(dbpath) as db:
            free=db.execute('PRAGMA freelist_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
        if free<minimum_bytes:continue
        tmp=dbpath.with_name('.queue-vacuum-'+uuid.uuid4().hex+'.sqlite3')
        exchanged=False
        try:
            if shutil.disk_usage(dbpath.parent).free<dbpath.stat().st_size:
                results.append(dict(run_id=run.name,status='INSUFFICIENT_WORKSPACE'));continue
            with _offline_queue(dbpath) as db:
                db.execute('VACUUM INTO ?', (str(tmp),))
            # VACUUM INTO defaults to rollback journaling; preserve a WAL ledger's format.
            with dbpath.open('rb') as stream:
                stream.seek(18); journal_header=stream.read(2)
            if journal_header==b'\x02\x02':
                with closing(sqlite3.connect(tmp)) as db:
                    if db.execute('PRAGMA journal_mode=WAL').fetchone()!=('wal',):
                        raise ValueError('Unable to preserve WAL format')
            if not _same_queue_contents(dbpath,tmp):
                results.append(dict(run_id=run.name,status='CONTENT_OR_ROWID_DIFFERS'));continue
            with tmp.open('rb') as stream:os.fsync(stream.fileno())
            with stopped_lock(state,run,launch,protect_references=True):
                if (_queue_db_signature(dbpath)!=before or wal.exists() or wal.is_symlink()
                        or shm.exists() or shm.is_symlink() or dbpath.stat().st_nlink!=1
                        or not _queue_is_settled(dbpath)
                        or optional(run/'EXIT.json').get('invocation_id')!=launch['invocation_id']):
                    results.append(dict(run_id=run.name,status='OWNERSHIP_CHANGED'));continue
                info=dbpath.stat();os.chmod(tmp,stat.S_IMODE(info.st_mode))
                reclaimed=info.st_blocks*512-tmp.stat().st_blocks*512
                if reclaimed<=0:continue
                exchange(dbpath,tmp)
                exchanged=True
                # The verified new DB is now visible; the original remains at tmp until synced.
                fd=os.open(dbpath.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
                tmp.unlink()
                results.append(dict(run_id=run.name,status='COMPACTED_VERIFIED',reclaimed_bytes=reclaimed))
        finally:
            # An exception after exchange must retain the old DB for recovery.
            if tmp.exists() and not exchanged:tmp.unlink()
    return dict(runs=results,reclaimed_bytes=sum(r.get('reclaimed_bytes',0) for r in results))



def pack_stopped_diagnostics(state,run_id,archive):
    """Create a lossless diagnostic archive; do not delete any original path."""
    import lzma
    archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():raise ValueError('Diagnostic archive already exists')
    run,launch=next(((r,l) for r,l in stopped_runs(state) if r.name==run_id),(None,None))
    if run is None:raise ValueError('No stopped registered run')
    db_path=run/'distributed/QUEUE.sqlite3'
    names=('TRANSPORT_TIMING.jsonl','JOB_STAGES.jsonl','WORKER_COLLECT_RETURN.json')
    temporary=archive.with_name(archive.name+'.tmp-'+uuid.uuid4().hex);count=size=0
    try:
        with stopped_lock(state,run,launch,protect_references=True),closing(sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True)) as db, lzma.open(temporary,'wt',encoding='utf-8') as writer:
            writer.write(json.dumps({'schema':'RPE_DIAGNOSTIC_ARCHIVE_V1','run_id':run_id})+'\n')
            for job,method,attempt in db.execute("SELECT job_id,method,attempt_id FROM jobs WHERE state='ACCEPTED'"):
                for folder in (run/'jobs'/safe_name(method)/safe_name(job),run/'distributed/transfers'/safe_name(job)/safe_name(attempt)):
                    for name in names:
                        path=folder/name
                        if path.resolve()!=path or not path.is_file():continue
                        before=identity(path);raw=path.read_bytes()
                        if identity(path)!=before:raise ValueError('Diagnostic changed during archive')
                        writer.write(json.dumps({'path':str(path.relative_to(run)),'sha256':hashlib.sha256(raw).hexdigest(),
                                                 'data':base64.b64encode(raw).decode()},separators=(',',':'))+'\n')
                        count+=1;size+=len(raw)
        if verify_diagnostic_archive(temporary,run_id)!=count:raise ValueError('Diagnostic archive verification differs')
        os.link(temporary,archive)
        return dict(files=count,original_bytes=size,archive_bytes=archive.stat().st_size,originals_preserved=True)
    finally:temporary.unlink(missing_ok=True)


def verify_diagnostic_archive(archive,run_id):
    import lzma
    count=0;seen=set()
    with lzma.open(archive,'rt',encoding='utf-8') as reader:
        if json.loads(reader.readline())!={'schema':'RPE_DIAGNOSTIC_ARCHIVE_V1','run_id':run_id}:raise ValueError('Diagnostic archive identity differs')
        for line in reader:
            row=json.loads(line);path=Path(row['path'])
            if path.is_absolute() or '..' in path.parts or row['path'] in seen:raise ValueError('Invalid diagnostic archive path')
            raw=base64.b64decode(row['data'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Diagnostic archive digest differs')
            seen.add(row['path']);count+=1
    return count


def restore_diagnostic_archive(state,run_id,archive):
    """Restore missing diagnostic paths only from a verified stopped-run pack."""
    import lzma
    expected=verify_diagnostic_archive(archive,run_id)
    run,launch=next(((r,l) for r,l in stopped_runs(state) if r.name==run_id),(None,None))
    if run is None:raise ValueError('No stopped registered run')
    names={'TRANSPORT_TIMING.jsonl','JOB_STAGES.jsonl','WORKER_COLLECT_RETURN.json'}
    restored=0
    with stopped_lock(state,run,launch,protect_references=True),lzma.open(archive,'rt',encoding='utf-8') as reader:
        header=json.loads(reader.readline())
        if header!={'schema':'RPE_DIAGNOSTIC_ARCHIVE_V1','run_id':run_id}:raise ValueError('Diagnostic archive identity differs')
        seen=set()
        for line in reader:
            row=json.loads(line);relative=Path(row['path']);parts=relative.parts
            if (relative.is_absolute() or '..' in parts or not parts or parts[-1] not in names
                    or not (len(parts)==4 and parts[0]=='jobs' or len(parts)==5 and parts[:2]==('distributed','transfers'))
                    or row['path'] in seen):raise ValueError('Invalid diagnostic restore path')
            seen.add(row['path']);raw=base64.b64decode(row['data'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Diagnostic archive digest differs')
            path=run/relative
            if path.resolve()!=path:raise ValueError('Diagnostic restore escapes original path')
            if path.exists():
                if path.read_bytes()!=raw:raise ValueError('Diagnostic original differs; preserve it')
                continue
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.restore-'+uuid.uuid4().hex)
            try:
                with temporary.open('xb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
                os.link(temporary,path);restored+=1
            finally:temporary.unlink(missing_ok=True)
        if len(seen)!=expected:raise ValueError('Diagnostic archive count differs')
    return dict(restored=restored,files=expected)


def alternate_relocation_readiness(repository, target_objects):
    """Conservative read-only check: target alone must cover every reachable object.

    A PASS is object availability evidence, not authorization to change an
    alternate or delete any source. Include refs, reflogs and index objects.
    The target must not itself rely on the old store being retired.
    """
    from compact_rpe_git_objects import git_environment
    repository = Path(repository).resolve(strict=True)
    target_objects = Path(target_objects).resolve(strict=True)
    if not target_objects.is_dir() or (target_objects/'info/alternates').exists():
        return dict(status='HOLD_TARGET_NOT_INDEPENDENT')
    env = git_environment()
    base = ['git','-C',str(repository)]
    # Stream the two Git processes; no persistent list or object copy.
    missing = total = 0
    with subprocess.Popen([*base,'rev-list','--objects','--all','--reflog','--indexed-objects','--no-object-names','HEAD'],
                          stdout=subprocess.PIPE, env=env) as listing:
        with subprocess.Popen([*base,'cat-file','--batch-check'], stdin=listing.stdout,
                              stdout=subprocess.PIPE, env=dict(env,GIT_OBJECT_DIRECTORY=str(target_objects))) as check:
            listing.stdout.close()
            for row in check.stdout:
                total += 1
                values = row.split()
                if len(values) != 3 or not values[-1].isdigit():
                    missing += 1
            if check.wait() != 0 or listing.wait() != 0:
                return dict(status='HOLD_GIT_ERROR', checked=total, missing=missing)
    return dict(status='OBJECTS_AVAILABLE' if total and not missing else 'HOLD_MISSING_OBJECTS',
                checked=total, missing=missing, mutation=False)


def storage_dependency_inventory(state, reference_files=()):
    """Read-only legacy storage inventory, never a declaration of safe deletion."""
    central=state.parent.resolve();repo=central.parent
    categories=('releases','deployment','planning','acceptance','validation','price-validation','operation-acceptance')
    entries={}
    for category in categories:
        for path in (central/category).glob('*'):
            if path.is_dir():entries[str(path)]=dict(category=category,path=str(path),references=[])
    for name in ('publication','results-publication'):
        path=central/name
        if path.is_dir():entries[str(path)]=dict(category='publication',path=str(path),references=[])
    prefix=re.escape(str(central))+r'/([^/\s"\\]+)(?:/([^/\s"\\]+))?'
    refs=0;errors=[]
    def inspect(text, origin):
        nonlocal refs
        for first,second in re.findall(prefix,text):
            key=str(central/first/second) if first in categories and second else str(central/first)
            if key in entries and origin not in entries[key]['references']:
                entries[key]['references'].append(origin);refs+=1
    def scan(path):
        if not path.is_file():return
        try:inspect(path.read_text(),str(path))
        except (OSError,UnicodeError) as error:errors.append(dict(path=str(path),reason=str(error)))
    for path in (state/'RUNS.json',central/'RELEASES.json',central/'SERVICE.json',central/'CLIENT.json',
                 state/'CURRENT_TARGET.json',*map(Path,reference_files)):
        scan(path)
    units=Path.home()/'.config/systemd/user'
    for pattern in ('rpe-*.service','rpe-*.service.d/*.conf','rpe-*.timer','rpe-*.timer.d/*.conf'):
        for path in units.glob(pattern):scan(path)
    for row in optional(state/'RUNS.json').values():
        try:run=Path(row['run_directory']).resolve(strict=True)
        except (OSError,KeyError,TypeError) as error:errors.append(dict(path=str(row),reason=str(error)));continue
        inspect(str(run),'registered run '+run.name)
        for filename in ('CONTROLLER_CONFIG.json','ROUND.json','ROUND.requested.json','APPLIED_SCOPE.json',
                         'LAUNCH.json','NORMAL_DRIVER.json','RUN_LINEAGE.json','distributed/LOCAL_DAEMON.json',
                         'distributed/LOCAL_WORKER.json'):
            scan(run/filename)
        for path in (run/'invocations').glob('*/LAUNCH.json'):scan(path)
        pf=run/'pf_auto'
        for candidates in [pf/'candidates',*pf.glob('price-recovery/recheck-*/candidates')]:
            for path in candidates.glob('*/pf_auto/pf_completion/provider_workspace/.git/objects/info/alternates'):scan(path)
    scan(repo/'.git/objects/info/alternates')
    for pattern in ('releases/*/.git/objects/info/alternates','deployment/*/.git/objects/info/alternates',
                    'deployment/*/*/.git/objects/info/alternates','planning/*/*/objects/info/alternates',
                    'transport-flat/*/objects.git/objects/info/alternates','publication/.git/objects/info/alternates',
                    'results-publication/.git/objects/info/alternates'):
        for path in central.glob(pattern):scan(path)
    # Report only matched paths, not unrelated process arguments or contents.
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if process.stat().st_uid==os.getuid():
                inspect((process/'cmdline').read_bytes().decode(errors='replace').replace('\0',' '),
                        'local process '+process.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        except OSError as error:errors.append(dict(path=str(process),reason=str(error)))
    for category in ('deployment','releases'):
        for entry in (central/category).glob('*'):
            if entry.is_symlink():
                inspect(str(entry.resolve()),str(entry))
            elif entry.is_dir():
                for child in entry.iterdir():
                    if child.is_symlink():inspect(str(child.resolve()),str(child))
    for row in entries.values():
        row['status']='REFERENCED' if row['references'] else 'UNCONFIRMED_NOT_A_DELETION_CANDIDATE'
    return dict(status='INVENTORY_ONLY',entries=sorted(entries.values(),key=lambda r:r['path']),references=refs,errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory', type=Path, required=True)
    parser.add_argument('--mount', type=Path, default=MOUNT)
    parser.add_argument('--logs-only', action='store_true')
    parser.add_argument('--share-events', action='store_true')
    parser.add_argument('--compact-settled-queues', action='store_true')
    parser.add_argument('--legacy-inventory', action='store_true')
    parser.add_argument('--reference-file', type=Path, action='append', default=[])
    parser.add_argument('--share-requests', action='store_true')
    parser.add_argument('--share-provider-packs', action='store_true')
    parser.add_argument('--compact-provider-sources',action='store_true')
    parser.add_argument('--restore-provider-sources',action='store_true')
    parser.add_argument('--check-alternate',type=Path,nargs=2,metavar=('REPOSITORY','TARGET_OBJECTS'))
    parser.add_argument('--pack-diagnostics',type=Path)
    parser.add_argument('--restore-diagnostics',type=Path)
    parser.add_argument('--archive-terminal-responses', action='store_true')
    parser.add_argument('--large-files-only', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--session-id')
    args = parser.parse_args()
    state = args.state_directory.resolve()
    if args.check_alternate:
        print(alternate_relocation_readiness(*args.check_alternate));return
    if args.legacy_inventory:
        print(json.dumps(storage_dependency_inventory(state,args.reference_file),ensure_ascii=False));return
    if args.compact_settled_queues:
        print(compact_stopped_queues(state,args.run_id));return
    if args.restore_diagnostics:
        if not args.run_id:parser.error('--restore-diagnostics requires --run-id')
        print(restore_diagnostic_archive(state,args.run_id,args.restore_diagnostics));return
    if args.pack_diagnostics:
        if not args.run_id:parser.error('--pack-diagnostics requires --run-id')
        print(pack_stopped_diagnostics(state,args.run_id,args.pack_diagnostics));return
    if args.compact_provider_sources:
        if not args.run_id:parser.error('--compact-provider-sources requires --run-id')
        print(compact_stopped_provider_sources(state,args.run_id));return
    if args.restore_provider_sources:
        if not args.run_id:parser.error('--restore-provider-sources requires --run-id')
        print(restore_stopped_provider_sources(state,args.run_id));return
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
