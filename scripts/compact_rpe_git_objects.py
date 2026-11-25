#!/usr/bin/env python3
"""Keep the local research Git object store compact without dropping objects."""
import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zlib

from trottertracks.rpe_q3_coordinator import atomic_json

GIB = 1024**3
MIN_FREE = 6 * GIB
BATCH_BYTES = 2 * GIB


def free_bytes(path):
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def git_environment():
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1',
               GIT_ALLOW_PROTOCOL='', GIT_TERMINAL_PROMPT='0')
    for key in ('GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_DIR', 'GIT_COMMON_DIR', 'GIT_WORK_TREE'):
        env.pop(key, None)
    return env


def flat_object_stores(repository, objects):
    """Recognize only the project's declared private transport stores."""
    directory = repository/'.rpe-central/transport-flat'
    stores = []
    for manifest in directory.glob('*/TRANSPORT.json'):
        try:
            value = json.loads(manifest.read_text())
            store = manifest.parent/'objects.git/objects'
            if (value.get('alternate_depth') == 1 and value.get('paths')
                    and all(Path(path).is_relative_to(manifest.parent/'checkouts') for path in value['paths'].values())
                    and store.resolve() == store
                    and store.is_dir() and store.stat().st_dev == objects.stat().st_dev
                    and (store.parent/'config').is_file() and (store.parent/'HEAD').is_file()):
                stores.append(store)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(stores)


def inventory(objects, *, min_age, minimum_object_bytes, rejected, flat_stores=()):
    groups = defaultdict(list)
    cutoff = time.time() - min_age
    for shard in objects.iterdir():
        if not re.fullmatch('[0-9a-f]{2}', shard.name) or not shard.is_dir() or shard.is_symlink():
            continue
        for path in shard.iterdir():
            oid = shard.name + path.name
            if not re.fullmatch('[0-9a-f]{40}', oid) or oid in rejected:
                continue
            try:
                info = path.lstat()
                if (not stat.S_ISREG(info.st_mode)
                        or info.st_size < minimum_object_bytes or info.st_mtime > cutoff):
                    continue
                stores = [objects]
                if info.st_nlink != 1:
                    for store in flat_stores:
                        try:
                            other = (store/shard.name/path.name).lstat()
                        except FileNotFoundError:
                            continue
                        if (other.st_dev, other.st_ino) == (info.st_dev, info.st_ino):
                            stores.append(store)
                    if len(stores) != info.st_nlink:
                        continue  # Unknown external links would still retain disk space.
                with path.open('rb') as stream:
                    raw = zlib.decompressobj().decompress(stream.read(65536), 100)
                header, _, body = raw.partition(b'\0')
                if not header.startswith(b'blob '):
                    continue
                kind = 'bundle' if body.startswith(b'# v') else 'pack' if body.startswith(b'PACK') else 'json' if body.startswith((b'{', b'[')) else 'other'
                groups[(kind, tuple(stores))].append((info.st_size, info.st_blocks * 512, oid, path))
            except (OSError, zlib.error):
                continue
    return groups


def batches(groups, batch_bytes):
    result = []
    for (kind, stores), rows in groups.items():
        group, total = [], 0
        for row in sorted(rows, reverse=True):
            if group and (total + row[0] > batch_bytes or len(group) >= 24):
                if len(group) >= 2:
                    result.append((kind, stores, group))
                group, total = [], 0
            group.append(row)
            total += row[0]
        if len(group) >= 2:
            result.append((kind, stores, group))
    return sorted(result, key=lambda batch: sum(row[1] for row in batch[2]), reverse=True)


def compact(repository, state, *, min_age=600, minimum_object_bytes=10*1024**2,
            batch_bytes=BATCH_BYTES, pressure_fraction=0.20, force=False, workers=3,
            proactive_bytes=5*GIB, min_free=MIN_FREE):
    repository, state = Path(repository).resolve(), Path(state).resolve()
    env = git_environment()

    def git(*args, **kwargs):
        return subprocess.run(['git', '-C', str(repository), *args], env=env, check=True, **kwargs)

    remote = git('remote', 'get-url', 'origin', text=True, capture_output=True).stdout.strip()
    if not remote.startswith(('git@github.com:HIROMU1015/', 'https://github.com/HIROMU1015/')):
        return {'status': 'NOT_OWNED_RESEARCH_REPOSITORY'}
    directory = Path(git('rev-parse', '--git-common-dir', text=True, capture_output=True).stdout.strip())
    if not directory.is_absolute():
        directory = repository/directory
    objects = directory.resolve()/'objects'
    if not objects.is_dir() or objects.is_symlink():
        return {'status': 'NO_LOCAL_OBJECT_STORE'}
    state.mkdir(parents=True, exist_ok=True)
    report_path = state/'GIT_COMPACTION.json'
    rejected_path = state/'GIT_COMPACTION_NO_SAVINGS.json'
    with (state/'GIT_COMPACTION.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'ALREADY_COMPACTING'}
        try:
            old = json.loads(rejected_path.read_text())
            rejected = {oid: at for oid, at in old.items() if time.time()-at < 6*3600}
        except (OSError, ValueError, AttributeError, TypeError):
            rejected = {}
        worklist = batches(inventory(objects, min_age=min_age, minimum_object_bytes=minimum_object_bytes,
                                    rejected=rejected, flat_stores=flat_object_stores(repository, objects)), batch_bytes)
        starting = free_bytes(objects)
        allocated = sum(row[1] for _, _, rows in worklist for row in rows)
        pressure = starting < shutil.disk_usage(objects).total * pressure_fraction
        report = dict(status='RUNNING', available_before_bytes=starting, planned_loose_bytes=allocated,
                      planned_objects=sum(len(rows) for _, _, rows in worklist), reclaimed_bytes=0, batches=[])

        def save(status):
            report.update(status=status, epoch=time.time(), available_after_bytes=free_bytes(objects))
            atomic_json(report_path, report)

        if not worklist or (not force and not pressure and allocated < proactive_bytes):
            save('NO_COMPACTION_NEEDED')
            return report
        save('RUNNING')

        def store_git(store, *args, **kwargs):
            if store == objects:
                return git(*args, **kwargs)
            return subprocess.run(['git', '--git-dir='+str(store.parent), *args], env=env, check=True, **kwargs)

        def check(ids, store=objects):
            lines = store_git(store,'cat-file', '--batch-check', input='\n'.join(ids)+'\n', text=True,
                        capture_output=True, timeout=120).stdout.splitlines()
            if len(lines) != len(ids) or any(len(line.split()) != 3 or line.split()[0] != oid
                    or line.split()[1] != 'blob' or not line.split()[2].isdigit() for oid, line in zip(ids, lines)):
                raise ValueError('Git object identity or availability changed')

        def index_ids(index):
            with index.open('rb') as stream:
                return {line.split()[1] for line in git('show-index', stdin=stream, capture_output=True).stdout.decode().splitlines()}

        def generate(rows, work):
            ids = [row[2] for row in rows]
            pack = work/'new.pack'
            with pack.open('wb') as output:
                subprocess.run(['nice','-n','10','git','-C',str(repository),'pack-objects','--stdout',
                                '--window=5','--depth=10','--threads=1','--window-memory=1g','--compression=1'],
                               env=env, input=('\n'.join(ids)+'\n').encode(), stdout=output,
                               stderr=subprocess.PIPE, check=True, timeout=900)
                output.flush()
                os.fsync(output.fileno())
            packed = pack.stat().st_size
            if packed >= sum(row[1] for row in rows)*.95:
                return packed, None
            index = pack.with_suffix('.idx')
            key = git('index-pack','--strict','-o',str(index),str(pack),text=True,capture_output=True,timeout=900).stdout.strip()
            git('verify-pack',str(index),capture_output=True,timeout=900)
            if index_ids(index) != set(ids):
                raise ValueError('New pack does not contain exactly the selected objects')
            return packed, key

        def install(rows, stores, work, packed, key):
            ids = [row[2] for row in rows]
            if key is None:
                rejected.update({oid: time.time() for oid in ids})
                return 0, 'NO_MATERIAL_SAVINGS'
            with (work/'new.pack').open('rb') as stream:
                installed = git('index-pack','--stdin','--strict',stdin=stream,text=True,capture_output=True,timeout=900).stdout.split()[-1]
            index = objects/'pack'/f'pack-{key}.idx'
            if installed != key or index_ids(index) != set(ids):
                raise ValueError('Installed pack index differs from the verified pack')
            check(ids)
            # The declared flat stores are native Git stores on the same SSD.
            # Reuse the one immutable pack and index in each, then verify that
            # Git itself can read the exact object IDs before pruning loose data.
            for store in stores[1:]:
                for original in (index.with_suffix('.pack'), index):
                    destination = store/'pack'/original.name
                    destination.parent.mkdir(exist_ok=True)
                    try:
                        os.link(original, destination)
                    except FileExistsError:
                        if original.stat().st_ino != destination.stat().st_ino:
                            with original.open('rb') as a, destination.open('rb') as b:
                                if hashlib.file_digest(a,'sha256').digest() != hashlib.file_digest(b,'sha256').digest():
                                    raise ValueError('Conflicting flat-store Git pack')
                linked_index = store/'pack'/index.name
                if index_ids(linked_index) != set(ids):
                    raise ValueError('Flat-store pack index differs')
                check(ids, store)
            before = {}
            for _, _, oid, path in rows:
                try:
                    info = path.lstat()
                    before[oid] = info.st_blocks*512 if info.st_nlink == len(stores) else 0
                except FileNotFoundError:
                    pass
            # Native Git removes only loose duplicates whose packed object is
            # present. It never prunes unreachable history in this operation.
            for store in stores:
                store_git(store,'prune-packed','-q',timeout=300)
                check(ids, store)
            return sum(before.get(oid,0) for _,_,oid,path in rows
                       if not any((store/path.relative_to(objects)).exists() for store in stores))-packed, 'PACKED_VERIFIED'

        position = 0
        try:
            with tempfile.TemporaryDirectory(prefix='git-pack-', dir=state) as tmp, ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {}
                while position < len(worklist) or pending:
                    while position < len(worklist) and len(pending) < workers:
                        kind, stores, rows = worklist[position]
                        size = sum(row[0] for row in rows)
                        reserved = sum(entry[4] for entry in pending.values())
                        if free_bytes(objects) < min_free + reserved + 2*size:
                            break
                        position += 1
                        work = Path(tmp)/str(position)
                        work.mkdir()
                        pending[pool.submit(generate, rows, work)] = (kind, stores, rows, work, 2*size)
                    if not pending:
                        break
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        kind, stores, rows, work, _ = pending.pop(future)
                        try:
                            packed, key = future.result()
                            saved, status = install(rows, stores, work, packed, key)
                            entry = dict(kind=kind, objects=len(rows), old_bytes=sum(row[1] for row in rows),
                                         packed_bytes=packed, pack=key, reclaimed_bytes=saved, status=status, stores=len(stores))
                            report['reclaimed_bytes'] += saved
                        except (OSError, ValueError, subprocess.SubprocessError) as error:
                            entry = dict(kind=kind, objects=len(rows), status='BATCH_DEFERRED', reason=str(error)[:350])
                        report['batches'].append(entry)
                        save('RUNNING')
                        atomic_json(rejected_path, rejected)
                        shutil.rmtree(work)
                        print(json.dumps(dict(entry, available_bytes=free_bytes(objects))), flush=True)
            save('COMPLETE' if position == len(worklist) else 'PACK_WORKSPACE_LOW')
        except BaseException:
            save('INTERRUPTED')
            raise
        history = state/'git-compaction-history'
        history.mkdir(exist_ok=True)
        atomic_json(history/f'{int(report["epoch"])}.json', report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--state-directory', type=Path, required=True)
    args = parser.parse_args()
    result = compact(args.repository, args.state_directory)
    print(json.dumps({key: value for key, value in result.items() if key != 'batches'}), flush=True)

if __name__ == '__main__':
    main()
