"""Host-wide intake limits and reclamation of acknowledged worker copies."""
import fcntl
import json
from pathlib import Path
import shutil
import stat
import subprocess
import time

from .rpe_q3_coordinator import atomic_json
from .rpe_preparation_artifacts import worker_cache


SAMPLE_SECONDS = 5
PREPARATION_COPIES = ('inputs', 'workspace', 'workspaces')


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


def _acknowledged_copies(spools, roots):
    monitored = {Path(root).resolve() for root in roots}
    for spool in sorted({Path(path).resolve() for path in spools}):
        if not any(spool.is_relative_to(root) for root in monitored):
            continue
        for marker in spool.glob('*/*/WORKER_STATE.json'):
            attempt = marker.parent
            if attempt.resolve() != attempt or marker.is_symlink():
                continue
            try:
                state = json.loads(marker.read_bytes())
                if state.get('state') != 'TRANSFERRED':
                    continue
                assignment = json.loads((attempt/'ASSIGNMENT.json').read_bytes())
                if (assignment['job_id'], assignment['attempt_id']) != (attempt.parent.name, attempt.name):
                    continue
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
                continue


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
            path = Path(filename)
            lock = path.parent/'WORKER_PROCESS.lock'
            try:
                if not stat.S_ISREG(lock.lstat().st_mode):
                    continue
                with lock.open('r+b') as process_lock:
                    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # The supervisor has exited. Keep its tombstone and receipts.
                    if path.is_symlink():
                        continue
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                removed += 1
                remaining -= sizes.get(filename, 0)
                if remaining <= target:
                    break
            except OSError:
                continue
        used = _used_bytes(roots)
        if used <= target:
            break
    return used, removed


def report(policy, *, spools=(), pbs_runs=None, refresh=False):
    """At the limit, reclaim acknowledged copies before computing intake capacity."""
    if policy is None:
        return None
    limit, roots = policy['limit_bytes'], policy['roots']
    sample = Path(policy['sample_path'])
    if (type(limit) is not int or limit <= 0 or not roots
            or not isinstance(roots, list) or any(not Path(p).is_absolute() for p in roots)
            or not sample.is_absolute()):
        raise ValueError('Worker storage requires a positive byte limit and absolute paths')
    identity = dict(limit_bytes=limit, roots=roots)
    cleanup = dict(spools=list(spools), pbs_runs=pbs_runs) if spools or pbs_runs else None

    def recent():
        if refresh:
            return None
        try:
            saved = json.loads(sample.read_bytes())
            if saved['policy'] == identity and 0 <= time.time() - saved['sampled_epoch'] < SAMPLE_SECONDS:
                report = saved['report']
                if (cleanup is None or report['reason'] != 'STORAGE_LIMIT_REACHED'
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
        with sample.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = recent()
            if cached is not None:
                return cached
            try:
                used = _used_bytes(roots)
                before, reclaimed = used, 0
                if used >= limit and (spools or pbs_runs):
                    candidates = list(spools)
                    if pbs_runs:
                        candidates.extend(Path(pbs_runs).glob('*/workers/*'))
                    # Leave room for new attempts before the next cleanup pass.
                    used, reclaimed = _reclaim(candidates, roots, used, limit * 9 // 10)
            except (OSError, ValueError, subprocess.SubprocessError):
                measured = dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
            else:
                measured = dict(limit_bytes=limit, used_bytes=used, accepting_new_jobs=used < limit,
                                reason='STORAGE_LIMIT_REACHED' if used >= limit else None)
                if reclaimed:
                    measured.update(reclaimed_copies=reclaimed, reclaimed_bytes=max(0, before-used))
            atomic_json(sample, dict(policy=identity, cleanup=cleanup, sampled_epoch=time.time(), report=measured))
            return measured
    except OSError:
        return dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
