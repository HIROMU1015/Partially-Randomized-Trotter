"""Host-wide intake limits and reclamation of acknowledged worker copies."""
from contextlib import ExitStack
import ctypes
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

from .rpe_q3_coordinator import atomic_json
from .rpe_preparation_artifacts import worker_cache
from .rpe_dp_prefix_reuse import PrefixStore


SAMPLE_SECONDS = 5
PREPARATION_COPIES = ('inputs', 'workspace', 'workspaces')


def duplicate_inventory(profile):
    policy = profile.get('storage_limit') or {}
    cache_path = profile.get('artifact_cache_directory')
    if not cache_path or cache_path not in policy.get('roots', []):
        return {}
    cache = Path(cache_path)
    return {path.name: path.stat().st_size for path in cache.glob('*/*')
            if path.parent.name == path.name[:2] and re.fullmatch('[a-f0-9]{64}', path.name)
            and path.resolve() == path and path.is_file() and path.stat().st_nlink == 1}


def _referenced_hashes(value):
    if isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value):
        yield value
    elif isinstance(value, (dict, list)):
        for item in value.values() if isinstance(value, dict) else value:
            yield from _referenced_hashes(item)


def reclaim_verified_duplicates(profile, proof):
    """Reclaim transport cache copies verified from the central CAS."""
    policy = profile['storage_limit']
    sample = Path(policy['sample_path'])
    spool = Path(profile['work_dir'])
    cache = Path(profile.get('artifact_cache_directory') or '.')
    if str(cache) not in policy['roots'] or not spool.is_dir():
        return {}
    with (spool/'WORKER.lock').open('a+b') as worker_lock, sample.with_suffix('.lock').open('r+b') as storage_lock:
        try:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {}
        fcntl.flock(storage_lock, fcntl.LOCK_EX)
        observation = json.loads(sample.read_bytes())
        report = observation['report']
        if (observation['policy'] != dict(limit_bytes=policy['limit_bytes'], roots=policy['roots'])
                or report['reason'] not in ('STORAGE_LIMIT_REACHED','FILESYSTEM_SPACE_LOW')):
            return {}
        retained = set()
        for marker in spool.glob('*/*/WORKER_STATE.json'):
            status = json.loads(marker.read_bytes())['state']
            if status == 'TRANSFERRED':
                continue
            if status != 'UNKNOWN':
                return {}
            for name in ('ASSIGNMENT.json', 'ARTIFACT_MANIFEST.json', 'OUTCOME.json'):
                path = marker.with_name(name)
                if path.is_file():
                    retained.update(_referenced_hashes(json.loads(path.read_bytes())))
        inventory = duplicate_inventory(profile)
        removed = {}
        target = max(0, report['used_bytes'] - report['limit_bytes'] * 9 // 10,
                     2*report.get('filesystem_min_free_bytes',0)-report.get('filesystem_free_bytes',0))
        for sha, size in proof.items():
            if sha in retained or inventory.get(sha) != size:
                continue
            path = cache/sha[:2]/sha
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
                    continue
            removed[sha] = path.stat().st_blocks * 512
            path.unlink()
            if sum(removed.values()) >= target:
                break
        return removed


def prefix_caches(profile):
    profiles = [profile, *(rule['profile'] for rule in profile.get('route_profiles', []))]
    return tuple(dict.fromkeys(p['prefix_cache_directory'] for p in profiles
                               if p.get('prefix_cache_directory')))


def _used_bytes(roots):
    existing = set()
    for path in roots:
        try:
            root = Path(path).resolve(strict=True)
        except FileNotFoundError:
            continue  # A worker may not have created this cache yet.
        if not root.is_dir():
            raise ValueError('Worker storage root is not a directory: ' + path)
        existing.add(root)
    ordered = sorted(existing, key=lambda root: (len(root.parts), str(root)))
    distinct = [root for i, root in enumerate(ordered)
                if not any(root.is_relative_to(parent) for parent in ordered[:i])]
    if not distinct:
        return 0
    output = subprocess.check_output(
        ['du', '--summarize', '--total', '--block-size=1', '--', *map(str, distinct)],
        text=True, stderr=subprocess.PIPE, timeout=10)
    return int(output.splitlines()[-1].split('\t', 1)[0])


def _attempt_copies(marker):
    attempt = marker.parent
    if attempt.resolve() != attempt or marker.is_symlink():
        return
    try:
        state = json.loads(marker.read_bytes())
        if state.get('state') != 'TRANSFERRED':
            return
        assignment = json.loads((attempt/'ASSIGNMENT.json').read_bytes())
        if (assignment['job_id'], assignment['attempt_id']) != (attempt.parent.name, attempt.name):
            return
        paths = [(0, attempt/'_dry_check')]
        if assignment['request']['route']['kind'] == 'preparation':
            outcome = json.loads((attempt/'OUTCOME.json').read_bytes())
            receipt = json.loads((attempt/'PROCESS_RETURN.json').read_bytes())
            if (outcome['status'] == 'PREPARATION_COMPLETE'
                    and outcome['request_id'] == assignment['job_id']
                    and outcome['request_sha256'] == assignment['request_sha256']
                    and receipt.get('returncode') == 0 and receipt.get('signal') is None):
                artifacts = outcome['artifacts']
                profile = json.loads((attempt/'WORKER_PROFILE.json').read_bytes())
                cache = worker_cache(profile)
                if all(cache.path(sha).is_file() for sha in set(artifacts.values())):
                    paths += [(1, attempt/name) for name in PREPARATION_COPIES]
                    paths += [(2, path) for path in attempt.glob('*-output.bundle') if path.name in artifacts]
        for priority, path in paths:
            if path.exists() and not path.is_symlink():
                yield priority, marker.stat().st_mtime_ns, path
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return


def _acknowledged_copies(spools, roots):
    monitored = {Path(root).resolve() for root in roots}
    for spool in sorted({Path(path).resolve() for path in spools}):
        if any(spool.is_relative_to(root) for root in monitored):
            for marker in spool.glob('*/*/WORKER_STATE.json'):
                yield from _attempt_copies(marker)


def _remove_copy(path):
    lock = path.parent/'WORKER_PROCESS.lock'
    try:
        if not stat.S_ISREG(lock.lstat().st_mode):
            return False
        with lock.open('r+b') as process_lock:
            fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.is_symlink():
                return False
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return True
    except OSError:
        return False


def reclaim_transferred_attempt(policy, attempt):
    """Discard acknowledged working copies, retaining receipts and result blobs."""
    if policy is None or not any(attempt.is_relative_to(Path(root).resolve()) for root in policy['roots']):
        return
    for _, _, path in _attempt_copies(attempt/'WORKER_STATE.json'):
        _remove_copy(path)


def _reclaim(spools, roots, used, target):
    copies = sorted(_acknowledged_copies(spools, roots))
    removed = 0
    # Recheck the host total after each batch. Separate copies can share blocks.
    for offset in range(0, len(copies), 32):
        paths = [str(path) for _, _, path in copies[offset:offset+32]]
        output = subprocess.check_output(
            ['du', '--summarize', '--block-size=1', '--', *paths],
            text=True, stderr=subprocess.PIPE, timeout=10)
        sizes = {name: int(size) for size, name in (line.split('\t', 1) for line in output.splitlines())}
        remaining = used
        for filename in paths:
            if _remove_copy(Path(filename)):
                removed += 1
                remaining -= sizes.get(filename, 0)
                if remaining <= target:
                    break
        used = _used_bytes(roots)
        if used <= target:
            break
    return used, removed


def _share_transferred_git_packs(spools, roots, bytes_needed):
    """Share byte-identical immutable Git packs in acknowledged workspaces."""
    monitored = [Path(root).resolve() for root in roots]
    reclaimed = 0
    for spool in sorted({Path(path).resolve() for path in spools}):
        if not any(spool.is_relative_to(root) for root in monitored):
            continue
        for marker in spool.glob('*/*/WORKER_STATE.json'):
            if reclaimed >= bytes_needed:
                return reclaimed
            attempt = marker.parent
            if marker.resolve() != marker or not (attempt/'workspaces').is_dir():
                continue
            lock = attempt/'WORKER_PROCESS.lock'
            try:
                if not stat.S_ISREG(lock.lstat().st_mode):
                    continue
                with lock.open('r+b') as process_lock:
                    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if json.loads(marker.read_bytes()).get('state') != 'TRANSFERRED':
                        continue
                    reclaimed += _share_attempt_git_packs(attempt)
            except (OSError, ValueError):
                continue
    return reclaimed


def _share_attempt_git_packs(attempt):
    copies = sorted(attempt.glob('workspaces/*/.git/objects/pack/pack-*'))
    original = attempt/'workspace/.git/objects/pack'
    reclaimed = 0
    for path in copies:
        source = original/path.name
        if (not re.fullmatch(r'pack-[a-f0-9]{40,64}\.(pack|idx)', path.name)
                or source.resolve() != source or path.resolve() != path):
            continue
        try:
            left, right = source.lstat(), path.lstat()
        except FileNotFoundError:
            continue
        if (not stat.S_ISREG(left.st_mode) or left.st_mode != right.st_mode
                or left.st_mode & 0o222 or left.st_size != right.st_size
                or left.st_dev != right.st_dev or left.st_ino == right.st_ino
                or (left.st_uid, left.st_gid) != (right.st_uid, right.st_gid)):
            continue
        with source.open('rb') as a, path.open('rb') as b:
            if hashlib.file_digest(a, 'sha256').digest() != hashlib.file_digest(b, 'sha256').digest():
                continue
        temporary = path.with_name('.shared-'+path.name)
        os.link(source, temporary)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if right.st_nlink == 1:
            reclaimed += right.st_blocks * 512
    return reclaimed


def _packed_objects(directory):
    objects = set()
    for index in sorted(directory.glob('*.idx')):
        with index.open('rb') as stream:
            output = subprocess.check_output(['git', 'show-index'], stdin=stream, timeout=30)
        objects.update(line.split()[1] for line in output.splitlines())
    return objects


def _compact_metadata_packs(policy, spools, pbs_runs):
    """Compact an idle worker's derived Git store without pruning objects."""
    roots = [Path(root).resolve() for root in policy['roots']]
    scratch_root = Path(policy['sample_path']).resolve().parent
    if any(scratch_root.is_relative_to(root) for root in roots):
        return 0
    spools = set(spools)
    if pbs_runs:
        spools.update(str(spool) for spool in Path(pbs_runs).glob('*/workers/*')
                      if (spool/'WORKER.lock').is_file())
    candidates = {}
    for spool_name in spools:
        spool = Path(spool_name).resolve()
        pool = spool.parent/'preparation-metadata-cache'
        for repository in pool.glob('*/objects.git'):
            if repository.resolve() == repository and any(repository.is_relative_to(root) for root in roots):
                candidates.setdefault(repository, [spool])
    if pbs_runs:
        shared = Path(pbs_runs).resolve()/'preparation-metadata-cache'
        for repository in shared.glob('*/objects.git'):
            if repository.resolve() == repository and any(repository.is_relative_to(root) for root in roots):
                candidates[repository] = sorted({Path(name).resolve() for name in spools})
    packed = []
    for repository, owners in candidates.items():
        directory = repository/'objects/pack'
        packs = list(directory.glob('*.pack'))
        if len(packs) >= 8:
            packed.append((sum(p.stat().st_blocks * 512 for p in packs), directory, owners))
    for size, directory, owners in sorted(packed, reverse=True):
        if size < 256 * 1024**2:
            break
        try:
            with ExitStack() as stack:
                for spool in owners:
                    worker_lock = stack.enter_context((spool/'WORKER.lock').open('rb'))
                    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if any(json.loads(marker.read_bytes()).get('state') in
                           ('ASSIGNED', 'STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER')
                           for marker in spool.glob('*/*/WORKER_STATE.json')):
                        raise BlockingIOError('Active worker attempts still own inputs')
                import_lock = stack.enter_context((directory.parents[2]/'IMPORT.lock').open('rb'))
                fcntl.flock(import_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                reclaimed = _compact_pack_directory(directory, scratch_root)
                if reclaimed:
                    return reclaimed
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return 0


def _compact_pack_directory(directory, scratch_root):
    entries = list(directory.iterdir())
    if any(path.resolve() != path or not path.is_file() or path.suffix not in ('.pack', '.idx', '.rev')
           for path in entries):
        return 0
    reverse_indexes = [path for path in entries if path.suffix == '.rev']
    if any(not path.with_suffix('.pack').is_file() or not path.with_suffix('.idx').is_file()
           for path in reverse_indexes):
        return 0
    for path in reverse_indexes:
        subprocess.run(['git', '-C', str(directory.parent.parent), 'index-pack', '--verify', '--rev-index',
                        str(path.with_suffix('.pack'))], check=True, capture_output=True, timeout=60)
    if directory.stat().st_dev != scratch_root.stat().st_dev:
        return 0
    before = sum(path.stat().st_blocks * 512 for path in entries)
    if shutil.disk_usage(scratch_root).free < before:
        return 0
    originals = _packed_objects(directory)
    if not originals:
        return 0
    repo = directory.parent.parent
    with tempfile.TemporaryDirectory(prefix='rpe-cache-compact-', dir=scratch_root) as temp:
        scratch = Path(temp)/'objects'
        stage = scratch/'pack'
        stage.mkdir(parents=True, mode=directory.stat().st_mode & 0o777)
        env = dict(os.environ, GIT_OBJECT_DIRECTORY=str(scratch),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo/'objects'))
        git = ['git', '-C', str(repo)]
        name = subprocess.check_output([*git, '-c', 'pack.threads=2', 'pack-objects',
                                        str(stage/'pack')],
            input=b'\n'.join(sorted(originals)) + b'\n',
            env=env, stderr=subprocess.PIPE, timeout=180).decode().strip()
        subprocess.run([*git, 'index-pack', '--verify', str(stage/f'pack-{name}.pack')],
                       check=True, capture_output=True, timeout=60)
        if _packed_objects(stage) != originals:
            raise ValueError('Compacted metadata pack has a different object set')
        after = sum(path.stat().st_blocks * 512 for path in stage.iterdir())
        if after >= before:
            return 0
        rename = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
        if rename is None:
            return 0
        if rename(-100, os.fsencode(stage), -100, os.fsencode(directory), 2) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot atomically exchange metadata packs')
        return before - after


def report(policy, *, spools=(), pbs_runs=None, prefix_cache=None, cache_directories=(), refresh=False, compact_metadata=False):
    """Reclaim acknowledged copies and recoverable cache before reaching the intake limit."""
    if policy is None:
        return None
    limit, roots = policy['limit_bytes'], policy['roots']
    sample = Path(policy['sample_path'])
    if (type(limit) is not int or limit <= 0 or not roots
            or not isinstance(roots, list) or any(not Path(p).is_absolute() for p in roots)
            or not sample.is_absolute()):
        raise ValueError('Worker storage requires a positive byte limit and absolute paths')
    identity = dict(limit_bytes=limit, roots=roots)
    caches = list(dict.fromkeys([*([prefix_cache] if prefix_cache else []), *cache_directories]))
    cleanup = dict(spools=list(spools), pbs_runs=pbs_runs, prefix_caches=caches)

    def recent():
        if refresh:
            return None
        try:
            saved = json.loads(sample.read_bytes())
            if saved['policy'] == identity and 0 <= time.time() - saved['sampled_epoch'] < SAMPLE_SECONDS:
                report = saved['report']
                if 'filesystem_free_bytes' not in report:
                    return None
                if (report['used_bytes'] is None or report['used_bytes'] < limit * 95 // 100
                        or saved.get('cleanup') == cleanup):
                    return report
        except (OSError, KeyError, TypeError, ValueError):
            pass
        return None

    try:
        cached = recent()
        if cached is not None:
            return cached
        sample.parent.mkdir(parents=True, exist_ok=True)
        compacted = 0
        if compact_metadata:
            observation = report(policy, spools=spools, pbs_runs=pbs_runs,
                                 cache_directories=caches, refresh=refresh)
            if observation['used_bytes'] is not None and observation['used_bytes'] >= limit * 95 // 100:
                compacted = _compact_metadata_packs(policy, spools, pbs_runs)
            if not compacted:
                return observation
            refresh = True
        with sample.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = recent()
            if cached is not None:
                return cached
            try:
                used = _used_bytes(roots)
                before, reclaimed = used, 0
                target = limit * 9 // 10
                cleanup_due = used >= limit * 95 // 100
                if cleanup_due and (spools or pbs_runs):
                    candidates = list(spools)
                    if pbs_runs:
                        candidates.extend(Path(pbs_runs).glob('*/workers/*'))
                    # Leave room for new attempts before the next cleanup pass.
                    used, reclaimed = _reclaim(candidates, roots, used, target)
                if cleanup_due and used > target and (spools or pbs_runs):
                    if _share_transferred_git_packs(candidates, roots, used-target):
                        used = _used_bytes(roots)
                evicted = 0
                if cleanup_due and used > target:
                    for path in caches:
                        cache = Path(path).resolve()
                        if cache.is_dir() and any(cache.is_relative_to(Path(root).resolve()) for root in roots):
                            cache_size = _used_bytes([str(cache)])
                            evicted += PrefixStore(cache).reclaim(cache_size - (used-target))
                            used = _used_bytes(roots)
                            if used <= target:
                                break
                filesystem_free = {}
                for name in roots:
                    root = Path(name)
                    while not root.exists():
                        root = root.parent
                    filesystem_free[name] = shutil.disk_usage(root).free
                min_free = limit // 10
                low_roots = [name for name,free in filesystem_free.items() if free < min_free]
            except (OSError, ValueError, subprocess.SubprocessError):
                measured = dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False,
                                reason='STORAGE_USAGE_UNAVAILABLE',filesystem_free_bytes=None,
                                filesystem_min_free_bytes=limit//10)
            else:
                measured = dict(limit_bytes=limit, used_bytes=used, accepting_new_jobs=used < limit and not low_roots,
                                reason='STORAGE_LIMIT_REACHED' if used >= limit else 'FILESYSTEM_SPACE_LOW' if low_roots else None,
                                filesystem_free_bytes=min(filesystem_free.values()),
                                filesystem_min_free_bytes=min_free,filesystem_low_space_roots=low_roots)
                if used < before:
                    measured.update(reclaimed_copies=reclaimed, reclaimed_bytes=max(0, before-used))
                if evicted:
                    measured['evicted_prefix_checkpoints'] = evicted
                if compacted:
                    measured['compacted_metadata_bytes'] = compacted
            atomic_json(sample, dict(policy=identity, cleanup=cleanup, sampled_epoch=time.time(), report=measured))
            return measured
    except OSError:
        return dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
